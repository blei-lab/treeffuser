"""
Implementation of CARD model for probabilistic regression according to the uq_box library.

This follows the tutorial from the uq_box library:
https://lightning-uq-box.readthedocs.io/en/latest/tutorials/regression/card.html
"""

import tempfile
from functools import partial

import numpy as np
import torch
import torch as t
import torch.nn as nn
from jaxtyping import Float
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning_uq_box.models import MLP
from lightning_uq_box.models import ConditionalGuidedLinearModel
from lightning_uq_box.uq_methods import CARDRegression
from lightning_uq_box.uq_methods import DeterministicRegression
from numpy import ndarray
from sklearn.base import MultiOutputMixin
from skopt.space import Integer
from skopt.space import Real
from torch.optim import Adam
from tqdm import tqdm

from testbed.models.base_model import ProbabilisticModel
from testbed.models.lightning_uq_models._data_module import GenericDataModule
from testbed.models.lightning_uq_models._utils import _to_tensor
from treeffuser.scaler import ScalerMixedTypes


class Card(ProbabilisticModel, MultiOutputMixin):

    def __init__(
        self,
        n_layers: int = 3,
        hidden_size: int = 100,
        max_epochs: int = 10000,
        dropout: float = 0.01,
        learning_rate: float = 1e-3,
        n_steps: int = 1000,
        batch_size: int = 64,
        enable_progress_bar: bool = False,
        use_gpu: bool = False,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        patience: int = 50,
        seed: int = 42,
    ):
        """
        Implements CARD using the uq_box library.

        CARD is a method for probabilistic regression.
        This class trains both a conditional model and a diffusion model which
        are used jointly to create a sample based probabilistic model.
        See https://arxiv.org/abs/2206.07275 for more details.

        Args:
            n_layers: The number of hidden layers to use in the conditional model.
            hidden_size: The size of the hidden layers to use in the conditional model.
            max_epochs: The maximum number of epochs to train the model. This is used
                for training both the conditional and diffusion models.
            dropout: The dropout rate to use when training the models.
            learning_rate: The learning rate to use when training the models.
            n_steps: The number of steps for the diffusion model.
            batch_size: The batch size to use when training the models.
            enable_progress_bar: Whether to display a progress bar during training.
            use_gpu: Whether to use the GPU for training.
            beta_start: The starting value of beta for the diffusion model.
                See https://arxiv.org/abs/2006.11239 for more details.
            beta_end: The ending value of beta for the diffusion model.
                see https://arxiv.org/abs/2006.11239 for more details.
            beta_schedule: The schedule to use for beta. This can be
                "linear", "const", "quad", "jsd", "sigmoid","cosine", "cosine_anneal",
            patience: The number of epochs to wait before early stopping if the validation loss
                does not improve.

        """
        super().__init__()
        self._cond_model: nn.Module = None
        self._diff_model: nn.Module = None

        self._y_dim = None
        self._x_dim = None

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self._layers = [hidden_size] * n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.max_epochs = max_epochs
        self.enable_progress_bar = enable_progress_bar
        self.patience = patience
        self.use_gpu = use_gpu

        self._my_temp_dir = tempfile.mkdtemp()
        self._x_scaler = None
        self._y_scaler = None

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def fit(
        self,
        X: Float[t.Tensor, "batch x_dim"],
        y: Float[t.Tensor, "batch y_dim"],
    ) -> None:

        self._y_dim = y.shape[1]
        self._x_dim = X.shape[1]

        self._x_scaler = ScalerMixedTypes()
        self._y_scaler = ScalerMixedTypes()

        X = self._x_scaler.fit_transform(X)
        y = self._y_scaler.fit_transform(y)

        self._fit_conditional_model(X, y)
        self._fit_diffusion_model(X, y)

    def _fit_conditional_model(
        self,
        X: Float[t.Tensor, "batch x_dim"],
        y: Float[t.Tensor, "batch y_dim"],
    ) -> None:
        """
        Fits the conditional mean model that is used later
        as part of the diffusion model in CARD.
        """
        dm = GenericDataModule(X, y, batch_size=self.batch_size)
        network = MLP(
            n_inputs=X.shape[1],
            n_hidden=self._layers,
            n_outputs=y.shape[1],
            dropout_p=self.dropout,
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=self.patience,
            verbose=False,
            mode="max",
        )
        _cond_trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if self.use_gpu else "cpu",
            enable_checkpointing=False,
            enable_progress_bar=self.enable_progress_bar,
            check_val_every_n_epoch=1,
            default_root_dir=self._my_temp_dir,
            callbacks=[early_stop_callback],
            log_every_n_steps=1,
        )
        self._cond_model = DeterministicRegression(
            model=network,
            loss_fn=nn.MSELoss(),
            optimizer=partial(Adam, lr=self.learning_rate),
        )
        _cond_trainer.fit(self._cond_model, dm)

    def _fit_diffusion_model(
        self,
        X: Float[t.Tensor, "batch x_dim"],
        y: Float[t.Tensor, "batch y_dim"],
    ) -> None:
        """
        Fits the diffusion model that is used to generate samples in CARD.
        This function should be called only after the conditional model has been trained.
        """
        if self._cond_model is None:
            raise ValueError(
                "The conditional model must be trained before the diffusion model."
            )

        dm = GenericDataModule(X, y, batch_size=self.batch_size)
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=self.patience,
            verbose=False,
            mode="max",
        )
        guidance_model = ConditionalGuidedLinearModel(
            n_steps=self.n_steps,
            x_dim=X.shape[1],
            y_dim=y.shape[1],
            n_hidden=self._layers,
            cat_x=True,  # condition on x through concatenation
            cat_y_pred=True,  # condition on y_0_hat (not super sure what this means)
        )
        self._diff_model = CARDRegression(
            cond_mean_model=self._cond_model.model,
            guidance_model=guidance_model,
            guidance_optim=partial(Adam, lr=self.learning_rate),
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            n_steps=self.n_steps,
            n_z_samples=1,
        )

        _diff_trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if self.use_gpu else "cpu",
            enable_checkpointing=False,
            enable_progress_bar=self.enable_progress_bar,
            check_val_every_n_epoch=1,
            default_root_dir=self._my_temp_dir,
            callbacks=[early_stop_callback],
        )
        _diff_trainer.fit(self._diff_model, dm)

    @t.no_grad()
    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[t.Tensor, "batch y_dim"]:
        """
        Predicts the mean of the distribution for each input using the conditional model.

        The output of this function might be different from the mean of the samples
        as they don't use the same model.
        """
        if self._cond_model is None:
            raise ValueError("The conditional model must be trained before calling predict.")

        X = self._x_scaler.transform(X)
        X = _to_tensor(X)

        self._cond_model.eval()
        y_tensor = self._cond_model.predict_step(X)["pred"]
        y_np = y_tensor.detach().numpy()
        y_np = self._y_scaler.inverse_transform(y_np)
        return y_np

    @t.no_grad()
    def sample(
        self,
        X: Float[ndarray, "batch x_dim"],
        n_samples: int,
        batch_size: int = 64,
        seed=None,
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Geneters n_samples of p(y|x) for each input x in X using the diffusion model.

        Args:
            X: The input data.
            n_samples: The number of samples to generate for each input.
            batch_size: The batch size to use when generating samples. For example
                if batch_size=2 and n_samples=10, and X has 5 datapoints, then there
                will be a total of 50 samples generated with batches of 2 generated at
                a time, therefore the diffusion model will be run 25 times. This
                is useful for managing memory usage.
        """
        # TODO: check if seed is used
        self._diff_model.eval()
        self._cond_model.eval()

        X = self._x_scaler.transform(X)
        X = _to_tensor(X)
        repeated_X = X.repeat(n_samples, 1)
        samples = torch.zeros(repeated_X.shape[0], self._y_dim)

        # Track the number of datapoints sampled
        datapoints_sampled = 0

        # Iterate until all datapoints are sampled

        pbar = tqdm(total=repeated_X.shape[0], disable=not self.enable_progress_bar)
        while datapoints_sampled < repeated_X.shape[0]:
            # Get the current batch of datapoints
            batch_start = datapoints_sampled
            batch_end = datapoints_sampled + batch_size
            batch = repeated_X[batch_start:batch_end]

            # Generate samples for the current batch
            generated_samples = self._diff_model.predict_step(batch)["samples"][-1]
            generated_samples = generated_samples[0]  # get rid of the first dimension

            # Determine the number of samples to use from the generated batch
            samples_to_use = min(batch_size, repeated_X.shape[0] - batch_start)

            # Update the samples tensor with the generated samples
            samples[batch_start : batch_start + samples_to_use] = generated_samples[
                :samples_to_use
            ]

            # Update the number of datapoints sampled
            datapoints_sampled += samples_to_use
            pbar.update(samples_to_use)

        # Reshape the samples tensor to (n_samples, batch, y_dim)
        samples = samples.detach().numpy()
        samples = self._y_scaler.inverse_transform(samples)
        samples = samples.reshape(n_samples, X.shape[0], self._y_dim)
        return samples

    def log_likelihood(
        self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]
    ) -> Float[ndarray, "batch"]:
        raise NotImplementedError

    @staticmethod
    def search_space() -> dict:

        return {
            "n_layers": Integer(2, 3),
            "hidden_size": Integer(16, 128),
            "dropout": Real(0.0, 0.5),
            "learning_rate": Real(1e-5, 1e-1, prior="log-uniform"),
        }
