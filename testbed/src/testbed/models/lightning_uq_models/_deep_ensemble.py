import tempfile
from functools import partial
from typing import Callable
from typing import List
from typing import Tuple

import lightning as L
import numpy as np
import torch
from jaxtyping import Float
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from numpy import ndarray
from sklearn.base import MultiOutputMixin
from skopt.space import Integer
from skopt.space import Real
from torch import Tensor
from torch import nn
from torch.optim import Adam

from testbed.models.base_model import ProbabilisticModel
from testbed.models.lightning_uq_models._data_module import GenericDataModule
from treeffuser.scaler import ScalerMixedTypes


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


class MVERegression(L.LightningModule):
    """
    Model for regression with mean-variance estimation (MVE), encapsulating the prediction model.

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

    def forward(
        self, x: Float[ndarray, "batch x_dim"]
    ) -> Tuple[Float[Tensor, "batch y_dim"], Float[Tensor, "batch y_dim"]]:
        preds = self.model(x)
        y_dim = preds.shape[1] // 2
        mean = preds[:, :y_dim]
        log_varish = preds[:, y_dim:]
        var = nn.functional.softplus(log_varish) + 1e-6
        return mean, var

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        x, y = batch["input"], batch["target"]
        mean, var = self(x)
        loss = self.loss_fn(mean, var, y)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        x, y = batch["input"], batch["target"]
        mean, var = self(x)
        loss = self.loss_fn(mean, var, y)
        self.log("val_loss", loss)
        return loss

    def loss_fn(
        self,
        mean: Float[Tensor, "batch y_dim"],
        var: Float[Tensor, "batch y_dim"],
        y: Float[Tensor, "batch y_dim"],
    ) -> Float[Tensor, ""]:
        dist = torch.distributions.Normal(mean, var.sqrt())
        log_likelihood = dist.log_prob(y).mean()
        return -log_likelihood

    def sample(
        self, x: Float[ndarray, "batch x_dim"], n_samples: int
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        mean, var = self(x)
        samples = torch.randn(n_samples, *mean.shape)
        samples = samples * torch.sqrt(var) + mean
        return samples

    def configure_optimizers(self) -> Callable:
        return self.optimizer_func(self.parameters())


class DeepEnsemble(ProbabilisticModel, MultiOutputMixin):
    """
    Implements a deep ensemble for regression tasks, where each model in the ensemble outputs both mean and variance.
    This approach is designed to provide predictions with associated uncertainty estimates.

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
    """

    def __init__(
        self,
        n_layers: int = 1,
        hidden_size: int = 50,
        max_epochs: int = 300,
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        use_gpu: bool = False,
        patience: int = 10,
        seed: int = 42,
        n_ensembles: int = 5,
        burnin_epochs: int = 10,
        enable_progress_bar: bool = False,
    ):
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.patience = patience
        self.seed = seed
        self.n_ensembles = n_ensembles
        self.burnin_epochs = burnin_epochs
        self.enable_progress_bar = enable_progress_bar
        self._models: List[MVERegression] = []
        self.scaler_x = None
        self.scaler_y = None
        np.random.seed(seed)
        torch.manual_seed(seed)

        self._my_temp_dir = tempfile.mkdtemp()
        self.y_dim = None

        np.random.seed(seed)
        torch.manual_seed(seed)

    def fit(self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]):
        """
        Fits the ensemble models to the provided training data.

        Parameters:
            X (np.ndarray): The input features for training.
            y (np.ndarray): The target outputs for training.
        """
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        self.scaler_x = ScalerMixedTypes()
        self.scaler_y = ScalerMixedTypes()

        X = self.scaler_x.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        self.y_dim = y.shape[1]
        dm = GenericDataModule(X, y, batch_size=self.batch_size)

        for _ in range(self.n_ensembles):
            model = MLP(
                n_inputs=X.shape[1],
                n_hidden=[self.hidden_size] * self.n_layers,
                n_outputs=y.shape[1] * 2,  # outputs for both mean and log-variance
            )
            optimizer_func = partial(Adam, lr=self.learning_rate)
            mve_model = MVERegression(model, optimizer_func, self.burnin_epochs)

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
            trainer.fit(model=mve_model, datamodule=dm)
            self._models.append(mve_model)

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        """
        Predicts the mean response for new data using the trained ensemble models.

        Parameters:
            X (np.ndarray): The input features for prediction.

        Returns:
            np.ndarray: The mean predictions from the ensemble.
        """
        X = self.scaler_x.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float)
        predictions = [
            model(X_tensor)[0].detach().numpy() for model in self._models
        ]  # Only extracting means
        mean_predictions = np.mean(predictions, axis=0)
        return self.scaler_y.inverse_transform(mean_predictions)

    def sample(
        self,
        X: Float[ndarray, "batch x_dim"],
        n_samples: int = 10,
        seed=None,
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Samples from the predictive distribution for given inputs using the trained ensemble models.

        Parameters:
            X (np.ndarray): Input features.
            n_samples (int): Number of samples to draw from the predictive distribution.

        Returns:
            np.ndarray: Samples drawn from the ensemble's predictive distribution.
        """
        X = self.scaler_x.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float)
        samples = []
        for model in self._models:
            model.eval()
            model_samples = model.sample(X_tensor, n_samples)
            model_samples = model_samples.detach().numpy()
            samples.append(model_samples)

        samples = np.concatenate(samples, axis=0)
        # Choose random samples from the ensemble
        indices = np.random.choice(range(samples.shape[0]), n_samples)
        samples = samples[indices]

        samples = samples.reshape(-1, self.y_dim)
        samples = self.scaler_y.inverse_transform(samples)
        samples = samples.reshape(n_samples, -1, self.y_dim)
        return samples

    def log_likelihood(
        self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]
    ) -> float:
        """
        Computes the log likelihood of the observed data under the predictive distribution of the ensemble.
        The log-likelihood is under the scaled data

        Parameters:
            X (np.ndarray): Input features.
            y (np.ndarray): Observed target values.

        Returns:
            float: The average log likelihood across all ensemble models.
        """
        X = self.scaler_x.transform(X)
        y = self.scaler_y.transform(y)

        log_sum_std = np.sum(np.log(self.scaler_y._scaler.scale_))

        X_tensor = torch.tensor(X, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.float)
        log_likelihoods = []
        for model in self._models:
            model.eval()
            mean, var = model(X_tensor)
            dist = torch.distributions.Normal(mean, var.sqrt())
            # using jacobian to calculate the log likelihood
            log_likelihood = (
                dist.log_prob(y_tensor).mean().detach().numpy()
                - log_sum_std
                - np.log(y.shape[0])
            )
            log_likelihoods.append(log_likelihood)

        return np.sum(log_likelihoods)

    @torch.no_grad()
    def predict_distribution(self, X: Float[ndarray, "batch x_dim"]):
        """
        Predicts the distribution using the DeepEnsemble model.
        """
        X = self.scaler_x.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float)
        parameters = []
        for model in self._models:
            model.eval()
            mean, var = model(X_tensor)
            std = torch.sqrt(var)
            mean = mean * self.scaler_y._scaler.scale_ + self.scaler_y._scaler.mean_
            std = std * self.scaler_y._scaler.scale_
            parameters.append((mean, std))

        mix = torch.distributions.Categorical(
            torch.ones(
                self.n_ensembles,
            )
        )
        batched_means = torch.stack([mean for mean, _ in parameters])
        batched_means = batched_means.permute(1, 0, 2)
        batched_stds = torch.stack([std for _, std in parameters])
        batched_stds = batched_stds.permute(1, 0, 2)
        comp = torch.distributions.Independent(
            torch.distributions.Normal(batched_means, batched_stds),
            1,
        )
        dist = torch.distributions.MixtureSameFamily(mix, comp)
        return dist

    @staticmethod
    def search_space():
        return {
            "n_layers": Integer(1, 5),
            "hidden_size": Integer(10, 500),
            "learning_rate": Real(1e-5, 1e-2, prior="log-uniform"),
            "n_ensembles": Integer(2, 10),
            # "patience": Integer(5, 50),
            # "burnin_epochs": Integer(1, 30),
            # "batch_size": Integer(16, 512),
        }
