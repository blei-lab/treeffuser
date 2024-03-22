import tempfile
from functools import partial
from typing import List
from typing import Optional

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
from torch.optim import Adam

from testbed.models.base_model import ProbabilisticModel
from testbed.models.card._data_module import GenericDataModule


def _to_tensor(X: Float[np.ndarray, "batch x_dim"]) -> Float[t.Tensor, "batch x_dim"]:
    return t.tensor(X, dtype=t.float32)


class Card(ProbabilisticModel):
    def __init__(
        self,
        layers: List[int] = [50, 50, 50],  # noqa
        max_epochs: int = 1000,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        n_steps: int = 1000,
        enable_progress_bar: bool = False,
        enable_checkpointing: bool = False,
        use_gpu: bool = False,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        patience: int = 10,
        seed: int = 42,
    ):
        self._cond_model: nn.Module = None
        self._diff_model: nn.Module = None

        self._y_dim = None
        self._x_dim = None

        self._layers = layers
        self._dropout = dropout
        self._learning_rate = learning_rate
        self._n_steps = n_steps
        self._beta_start = beta_start
        self._beta_end = beta_end
        self._beta_schedule = beta_schedule
        self._max_epochs = max_epochs
        self._enable_progress_bar = enable_progress_bar
        self._enable_checkpointing = enable_checkpointing
        self._patience = patience
        self._use_gpu = use_gpu

        self._seed = seed
        self._my_temp_dir = tempfile.mkdtemp()

    def fit(
        self,
        X: Float[t.Tensor, "batch x_dim"],
        y: Float[t.Tensor, "batch y_dim"],
        cat_idx: Optional[List[int]] = None,
    ) -> None:

        self._y_dim = y.shape[1]
        self._x_dim = X.shape[1]

        self._fit_conditional_model(X, y, cat_idx)
        self._fit_diffusion_model(X, y, cat_idx)

    def _fit_conditional_model(
        self,
        X: Float[t.Tensor, "batch x_dim"],
        y: Float[t.Tensor, "batch y_dim"],
        cat_idx: Optional[List[int]] = None,
    ) -> None:
        dm = GenericDataModule(X, y)
        network = MLP(
            n_inputs=X.shape[1],
            n_hidden=self._layers,
            n_outputs=y.shape[1],
            dropout_p=self._dropout,
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=self._patience,
            verbose=False,
            mode="max",
        )
        _cond_trainer = Trainer(
            max_epochs=self._max_epochs,
            accelerator="gpu" if self._use_gpu else "cpu",
            enable_checkpointing=self._enable_checkpointing,
            enable_progress_bar=self._enable_progress_bar,
            check_val_every_n_epoch=1,
            default_root_dir=self._my_temp_dir,
            callbacks=[early_stop_callback],
        )
        self._cond_model = DeterministicRegression(
            model=network,
            loss_fn=nn.MSELoss(),
            optimizer=partial(Adam, lr=self._learning_rate),
        )
        _cond_trainer.fit(self._cond_model, dm)

    def _fit_diffusion_model(
        self,
        X: Float[t.Tensor, "batch x_dim"],
        y: Float[t.Tensor, "batch y_dim"],
        cat_idx: Optional[List[int]] = None,
    ) -> None:
        dm = GenericDataModule(X, y)
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=self._patience,
            verbose=False,
            mode="max",
        )
        guidance_model = ConditionalGuidedLinearModel(
            n_steps=self._n_steps,
            x_dim=X.shape[1],
            y_dim=y.shape[1],
            n_hidden=self._layers,
            cat_x=True,  # condition on x through concatenation
            cat_y_pred=True,  # condition on y_0_hat (not super sure what this means)
        )
        self._diff_model = CARDRegression(
            cond_mean_model=self._cond_model.model,
            guidance_model=guidance_model,
            guidance_optim=partial(Adam, lr=self._learning_rate),
            beta_start=self._beta_start,
            beta_end=self._beta_end,
            beta_schedule=self._beta_schedule,
            n_steps=self._n_steps,
            n_z_samples=1,
        )

        _diff_trainer = Trainer(
            max_epochs=self._max_epochs,
            accelerator="gpu" if self._use_gpu else "cpu",
            enable_checkpointing=self._enable_checkpointing,
            enable_progress_bar=self._enable_progress_bar,
            check_val_every_n_epoch=1,
            default_root_dir=self._my_temp_dir,
            callbacks=[early_stop_callback],
        )
        _diff_trainer.fit(self._diff_model, dm)

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[t.Tensor, "batch y_dim"]:
        X = _to_tensor(X)
        y_tensor = self._cond_model.predict_step(X)["pred"]
        y_np = y_tensor.detach().numpy()
        return y_np

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples: int, batch_size: int = 64
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        X = _to_tensor(X)
        repeated_X = X.repeat(n_samples, 1)
        samples = torch.zeros(repeated_X.shape[0], self._y_dim)

        # Track the number of datapoints sampled
        datapoints_sampled = 0

        # Iterate until all datapoints are sampled
        while datapoints_sampled < repeated_X.shape[0]:
            # Get the current batch of datapoints
            batch_start = datapoints_sampled
            batch_end = datapoints_sampled + batch_size
            batch = repeated_X[batch_start:batch_end]

            # Generate samples for the current batch
            generated_samples = self._diff_model.predict_step(batch)["samples"][-1]
            generated_samples = generated_samples[0]  # get rid of the first dimension

            # Determine the number of samples to use from the generated batch
            samples_to_use = min(batch_size, X.shape[0] - datapoints_sampled)

            # Update the samples tensor with the generated samples
            samples[batch_start : batch_start + samples_to_use] = generated_samples[
                :samples_to_use
            ]

            # Update the number of datapoints sampled
            datapoints_sampled += samples_to_use

        # Reshape the samples tensor to (n_samples, batch, y_dim)
        samples = samples.reshape(X.shape[0], n_samples, self._y_dim)
        samples = samples.permute(1, 0, 2)
        samples = samples.detach().numpy()
        return samples

    def log_likelihood(
        self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]
    ) -> Float[ndarray, "batch"]:
        raise NotImplementedError

    def predict_distribution(self, X: ndarray) -> ndarray:
        raise NotImplementedError
