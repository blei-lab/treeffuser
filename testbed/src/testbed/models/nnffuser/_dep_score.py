"""
This file should contain a general abstraction of the score models and
should function as a wrapper for different models we might want to use.

TODO: Cleanup naming and such.
"""

import abc
import tempfile
from functools import partial
from typing import Callable
from typing import List
from typing import Optional

import lightning as L
import numpy as np
import torch
from jaxtyping import Float
from jaxtyping import Int
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from numpy import ndarray
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch import nn
from torch.optim import Adam

from testbed.models._preprocessors import Preprocessor
from testbed.models.nnfuser._data_module import GenericDataModule
from treeffuser.sde import DiffusionSDE

###################################################
# Helper functions & classes
###################################################


class MLP(nn.Module):
    """
    A simple multilayer perceptron (MLP) for regression tasks with flexible depth and hidden layer sizes.

    Parameters:
        n_inputs (int): Number of input features.
        n_hidden (List[int]): List containing the size of each hidden layer.
        n_outputs (int): Number of outputs.
        activation_fn (Callable): Activation function to use in hidden layers (default is ReLU).
    """

    def __init__(
        self,
        n_inputs: int,
        n_hidden: List[int],
        n_outputs: int,
        activation_fn: Callable = nn.ReLU,
    ):
        super().__init__()
        layers = []
        input_dim = n_inputs
        for hidden_dim in n_hidden:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, n_outputs))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Float[Tensor, "batch x_dim"]) -> Float[Tensor, "batch 2 * y_dim"]:
        return self.network(x)


class Regression(L.LightningModule):
    """
    Model for simple L2 regression.

    Parameters:
        model (nn.Module): The neural network model that outputs both mean and log-variance.
        optimizer_func (Callable): Function to create the optimizer.
        burnin_epochs (int): Number of epochs for burn-in phase.
    """

    def __init__(self, model: nn.Module, optimizer_func: Callable, burnin_epochs: int):
        super().__init__()
        self.model = model
        self.optimizer_func = optimizer_func
        self.burnin_epochs = burnin_epochs

    def forward(self, x: Float[ndarray, "batch x_dim"]) -> Float[Tensor, "batch y_dim"]:
        mean = self.model(x)
        return mean

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        x, y = batch["input"], batch["target"]
        mean = self(x)
        loss = self.loss_fn(mean, y)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        x, y = batch["input"], batch["target"]
        mean = self(x)
        loss = self.loss_fn(mean, y)
        self.log("val_loss", loss)
        return loss

    def loss_fn(
        self,
        mean: Float[Tensor, "batch y_dim"],
        y: Float[Tensor, "batch y_dim"],
    ) -> Float[Tensor, ""]:
        return ((mean - y) ** 2).mean()

    def configure_optimizers(self) -> Callable:
        return self.optimizer_func(self.parameters())


def _make_training_data(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    sde: DiffusionSDE,
    n_repeats: int,
    eval_percent: Optional[float],
    seed: Optional[int] = None,
):
    """
    Creates the training data for the score model. This functions assumes that
    1.  Score is parametrized as score(y, x, t) = GBT(y, x, t) / std(t)
    2.  The loss that we want to use is
        || std(t) * score(y_perturbed, x, t) - (mean(y, t) - y_perturbed)/std(t) ||^2
        Which corresponds to the standard denoising objective with weights std(t)**2
        This ends up meaning that we optimize
        || GBT(y_perturbed, x, t) - (-z)||^2
        where z is the noise added to y_perturbed.

    Returns:
    - predictors_train: X_train=[y_perturbed_train, x_train, t_train] for lgbm
    - predictors_val: X_val=[y_perturbed_val, x_val, t_val] for lgbm
    - predicted_train: y_train=[-z_train] for lgbm
    - predicted_val: y_val=[-z_val] for lgbm
    """
    EPS = 1e-5  # smallest step we can sample from
    T = sde.T
    if seed is not None:
        np.random.seed(seed)

    X_train, X_test, y_train, y_test = X, None, y, None
    predictors_train, predictors_val = None, None
    predicted_train, predicted_val = None, None

    if eval_percent is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=eval_percent, random_state=seed
        )

    # TRAINING DATA
    X_train = np.tile(X, (n_repeats, 1))
    y_train = np.tile(y, (n_repeats, 1))
    t_train = np.random.uniform(0, 1, size=(y_train.shape[0], 1)) * (T - EPS) + EPS
    z_train = np.random.normal(size=y_train.shape)

    train_mean, train_std = sde.get_mean_std_pt_given_y0(y_train, t_train)
    perturbed_y_train = train_mean + train_std * z_train

    predictors_train = np.concatenate([perturbed_y_train, X_train, t_train], axis=1)
    predicted_train = -1.0 * z_train

    # VALIDATION DATA
    if eval_percent is not None:
        t_val = np.random.uniform(0, 1, size=(y_test.shape[0], 1)) * (T - EPS) + EPS
        z_val = np.random.normal(size=(y_test.shape[0], y_test.shape[1]))

        val_mean, val_std = sde.get_mean_std_pt_given_y0(y_test, t_val)
        perturbed_y_val = val_mean + val_std * z_val
        predictors_val = np.concatenate(
            [perturbed_y_val, X_test, t_val.reshape(-1, 1)], axis=1
        )
        predicted_val = -1.0 * z_val

    return predictors_train, predictors_val, predicted_train, predicted_val


###################################################
# Main models
###################################################


class Score(abc.ABC):
    @abc.abstractmethod
    def score(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        t: Int[np.ndarray, "batch"],
    ):

        pass

    @abc.abstractmethod
    def fit(self, X: Float[np.ndarray, "batch x_dim"], y: Float[np.ndarray, "batch y_dim"]):
        pass


# lightgbm score
class NNScore(Score):
    def __init__(
        self,
        sde: DiffusionSDE,
        n_repeats: Optional[int] = 1,
        n_layers: int = 1,
        hidden_size: int = 50,
        max_epochs: int = 300,
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        use_gpu: bool = False,
        patience: int = 10,
        seed: int = 42,
        burnin_epochs: int = 10,
        eval_percent: float = 0.1,
        enable_progress_bar: bool = False,
    ) -> None:
        """
        Args:
        This model doesn't do any model checking or validation. It's assumed that
        that the main user is the `Treeffuser` class and that the user has already
        checked that the inputs are valid.

            Diffusion model args
            -------------------------------
            sde (SDE): A member from the SDE class specifying the sde that is implied
                by the score model.
            n_repeats (int): How many times to repeat the training dataset. i.e how
                many noisy versions of a point to generate for training.

            NN args
            -------------------------------
            Attributes:
                n_layers (int): Number of hidden layers in each model of the ensemble.
                hidden_size (int): Size of each hidden layer.
                max_epochs (int): Maximum number of training epochs for each model.
                learning_rate (float): Learning rate for the optimizer.
                batch_size (int): Batch size used during training.
                use_gpu (bool): If True, training is performed on GPU.
                patience (int): Number of epochs with no improvement after which training will be stopped early.
                seed (int): Random seed for reproducibility.
                n_ensembles (int): Number of models in the ensemble.
                burnin_epochs (int): Number of initial epochs during which the model parameters stabilize.
                enable_progress_bar (bool): If True, enables the display of a progress bar during training.
                    eval_percent (float): Percentage of the training data to use for validation.
        """

        # Diffusion model args
        self._sde = sde
        self._n_repeats = n_repeats

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.patience = patience
        self.seed = seed
        self.burnin_epochs = burnin_epochs
        self.enable_progress_bar = enable_progress_bar
        self.eval_percent = eval_percent

        # Other stuff part of internal state
        self.model = None  # Convention inputs are (y, x, t)
        self.is_fitted = False

        self._x_scaler = None
        self._y_scaler = None

        np.random.seed(seed)
        torch.manual_seed(seed)

        self._my_temp_dir = tempfile.mkdtemp()
        self.y_dim = None

        np.random.seed(seed)
        torch.manual_seed(seed)

    def score(
        self,
        y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Int[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        scores = []
        predictors = np.concatenate([y, X, t], axis=1)
        _, std = self._sde.get_mean_std_pt_given_y0(y, t)

        predictors = self._x_scaler.transform(predictors)
        predictors = torch.from_numpy(predictors).float()
        score_p_not_norm = self.model(predictors).detach().numpy()
        score_p = self._y_scaler.inverse_transform(score_p_not_norm)
        scores = score_p / std

        return scores

    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
    ):
        """
        Fit the score model to the data.

        Args:
            X: input data
            y: target data
            n_repeats: How many times to repeat the training dataset.
            likelihood_reweighting: Whether to reweight the likelihoods.
            likelihood_weighting: If `True`, weight the mixture of score
                matching losses according to https://arxiv.org/abs/2101.09258;
                otherwise use the weighting recommended in song's SDEs paper.
        """
        nn_X_train, nn_X_val, nn_y_train, nn_y_val = _make_training_data(
            X=X,
            y=y,
            sde=self._sde,
            n_repeats=self._n_repeats,
            eval_percent=self.eval_percent,
            seed=self.seed,
        )

        self._x_scaler = Preprocessor()
        self._y_scaler = Preprocessor()

        nn_X_train = self._x_scaler.fit_transform(nn_X_train)
        nn_X_val = self._x_scaler.transform(nn_X_val)
        nn_y_train = self._y_scaler.fit_transform(nn_y_train)
        nn_y_val = self._y_scaler.transform(nn_y_val)

        dm = GenericDataModule(
            X_train=nn_X_train,
            y_train=nn_y_train,
            X_val=nn_X_val,
            y_val=nn_y_val,
            batch_size=self.batch_size,
        )

        # Fit the model
        model = MLP(
            n_inputs=nn_X_train.shape[1],
            n_hidden=[self.hidden_size] * self.n_layers,
            n_outputs=nn_y_train.shape[1],
        )
        optimizer_func = partial(Adam, lr=self.learning_rate)
        regression_model = Regression(model, optimizer_func, self.burnin_epochs)

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=self.patience,
            verbose=False,
            mode="min",
        )

        trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if self.use_gpu else "cpu",
            enable_checkpointing=False,
            enable_progress_bar=self.enable_progress_bar,
            default_root_dir=self._my_temp_dir,
            callbacks=[early_stop_callback],
        )
        trainer.fit(model=regression_model, datamodule=dm)
        self.model = model
