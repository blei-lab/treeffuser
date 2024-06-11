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
from lightning_uq_box.uq_methods import NLL
from lightning_uq_box.uq_methods import MCDropoutRegression
from numpy import ndarray
from sklearn.base import MultiOutputMixin
from skopt.space import Integer
from skopt.space import Real
from torch.optim import Adam

from testbed.models.base_model import ProbabilisticModel
from testbed.models.lightning_uq_models._data_module import GenericDataModule
from testbed.models.lightning_uq_models._utils import _to_tensor
from treeffuser.scaler import ScalerMixedTypes


class MCDropout(ProbabilisticModel, MultiOutputMixin):

    def __init__(
        self,
        n_layers: int = 4,
        hidden_size: int = 100,
        max_epochs: int = 1000,
        dropout: float = 0.1,
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        enable_progress_bar: bool = False,
        use_gpu: bool = False,
        burnin_epochs: int = 0,
        patience: int = 30,
        seed: int = 42,
    ):
        """
        Implements MCDropout using the uq_box library for uncertainty quantification.

        MCDropout is a Bayesian approximation method that uses dropout during both training
        and inference to approximate the posterior distribution of the network's weights.

        Args:
            n_layers: The number of hidden layers to use in the MLP model.
            hidden_size: The size of the hidden layers to use in the MLP model.
            max_epochs: The maximum number of epochs to train the model.
            dropout: The dropout rate to use when training the model.
            learning_rate: The learning rate to use when training the model.
            batch_size: The batch size to use when training the model.
            enable_progress_bar: Whether to display a progress bar during training.
            use_gpu: Whether to use the GPU for training.
            num_mc_samples: The number of Monte Carlo samples to take during inference.
            burnin_epochs: The number of initial epochs before contributing to the
                loss function.

        """
        super().__init__(seed)
        self._model: nn.Module = None

        self._y_dim = None
        self._x_dim = None

        self.layers = [hidden_size] * n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.enable_progress_bar = enable_progress_bar
        self.use_gpu = use_gpu
        self.burnin_epochs = burnin_epochs
        self.patience = patience
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self._x_scaler = None
        self._y_scaler = None

        self.seed = seed

        self._my_temp_dir = tempfile.mkdtemp()

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def fit(self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]) -> None:
        """
        Fits the MCDropout model using provided training data.
        """
        self._y_dim = y.shape[1]
        self._x_dim = X.shape[1]

        self._x_scaler = ScalerMixedTypes()
        self._y_scaler = ScalerMixedTypes()

        X = self._x_scaler.fit_transform(X)
        y = self._y_scaler.fit_transform(y)

        dm = GenericDataModule(X, y, batch_size=self.batch_size)
        network = MLP(
            n_inputs=self._x_dim,
            n_hidden=self.layers,
            n_outputs=self._y_dim * 2,  # mean and variance
            dropout_p=self.dropout,
        )

        self._model = MCDropoutRegression(
            model=network,
            optimizer=partial(Adam, lr=self.learning_rate),
            loss_fn=NLL(),
            num_mc_samples=1,  # We won't use this
            burnin_epochs=self.burnin_epochs,
        )

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
            check_val_every_n_epoch=1,
            default_root_dir=self._my_temp_dir,
            callbacks=[early_stop_callback],
        )
        trainer.fit(self._model, dm)

    @torch.no_grad()
    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[torch.Tensor, "batch y_dim"]:
        """
        Predicts using the MCDropout model by taking multiple forward passes with dropout enabled.
        """
        if self._model is None:
            raise ValueError("The model must be trained before calling predict.")

        X = self._x_scaler.transform(X)
        X = _to_tensor(X)
        self._model.eval()

        preds = self._model.predict_step(X)
        y = preds["pred"]
        y_np = y.cpu().numpy()
        y_np = self._y_scaler.inverse_transform(y_np)
        return y_np

    @torch.no_grad()
    def sample(
        self,
        X: Float[ndarray, "batch x_dim"],
        n_samples: int = 10,
        seed=None,
    ) -> Float[torch.Tensor, "n_samples batch y_dim"]:
        """
        Samples from the MCDropout model by taking multiple forward passes with dropout enabled.
        """
        if self._model is None:
            raise ValueError("The model must be trained before calling sample.")

        X = self._x_scaler.transform(X)
        X = _to_tensor(X)

        self._model.hparams.num_mc_samples = n_samples
        preds = self._model.predict_step(X)

        mean = preds["pred"]
        std = preds["pred_uct"].reshape(-1, self._y_dim)

        samples = (
            t.distributions.Normal(mean, std).sample((n_samples, 1)).squeeze()
        )  # batch, num_samples
        samples = samples.unsqueeze(-1)
        samples = samples.cpu().numpy()

        # Inverse transform the samples
        samples = samples.reshape(-1, self._y_dim)
        samples = self._y_scaler.inverse_transform(samples)
        samples = samples.reshape(n_samples, -1, self._y_dim)
        return samples

    @staticmethod
    def search_space() -> dict:
        return {
            "n_layers": Integer(1, 7),
            "hidden_size": Integer(10, 500),
            "dropout": Real(0.0, 0.5),
            "learning_rate": Real(1e-5, 1e-1, prior="log-uniform"),
            "burnin_epochs": Integer(1, 10),
        }
