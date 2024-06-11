import tempfile
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import QuantileRegression as QR
from numpy import ndarray
from skopt.space import Integer
from skopt.space import Real
from torch.optim import Adam

from testbed.models.base_model import ProbabilisticModel
from testbed.models.lightning_uq_models._data_module import GenericDataModule
from testbed.models.lightning_uq_models._utils import _to_tensor
from treeffuser.scaler import ScalerMixedTypes


class QuantileRegression(ProbabilisticModel):

    def __init__(
        self,
        n_layers: int = 3,
        hidden_size: int = 50,
        max_epochs: int = 300,
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        enable_progress_bar: bool = False,
        use_gpu: bool = False,
        patience: int = 10,
        seed: int = 42,
        n_quantiles: int = 10,
    ):
        """
        Implements QuantileRegression using the uq_box library for uncertainty quantification.

        QuantileRegression predicts conditional quantiles that approximate the true quantiles
        of the data without making assumptions about the distribution of errors.

        Args:
            n_layers: The number of hidden layers to use in the MLP model.
            hidden_size: The size of the hidden layers to use in the MLP model.
            max_epochs: The maximum number of epochs to train the model.
            learning_rate: The learning rate to use when training the model.
            batch_size: The batch size to use when training the model.
            enable_progress_bar: Whether to display a progress bar during training.
            use_gpu: Whether to use the GPU for training.
            patience: The number of epochs to wait for improvement before early stopping.
            seed: The random seed for reproducibility.
            quantiles: The quantiles to predict.

        """
        super().__init__(seed)
        self._model: nn.Module = None

        self._y_dim = None
        self._x_dim = None

        self.layers = [hidden_size] * n_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.enable_progress_bar = enable_progress_bar
        self.use_gpu = use_gpu
        self.patience = patience
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_quantiles = n_quantiles

        lower_quantile = 1 / (n_quantiles + 1)
        upper_quantile = 1 - lower_quantile
        self.quantiles = np.linspace(lower_quantile, upper_quantile, n_quantiles - 1)
        # add the median
        self.quantiles = np.array([*list(self.quantiles), 0.5])
        self.quantiles = np.sort(self.quantiles)

        self._x_scaler = None
        self._y_scaler = None

        self.seed = seed

        self._my_temp_dir = tempfile.mkdtemp()

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def fit(self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]) -> None:
        """
        Fits the QuantileRegression model using provided training data.
        """
        self._y_dim = y.shape[1]
        self._x_dim = X.shape[1]

        # if y is not 1D, raise an error
        if y.shape[1] > 1:
            raise ValueError("QuantileRegression only accepts 1 dimensional y values.")

        self._x_scaler = ScalerMixedTypes()
        self._y_scaler = ScalerMixedTypes()

        X = self._x_scaler.fit_transform(X.astype(float))
        y = self._y_scaler.fit_transform(y)

        dm = GenericDataModule(X, y, batch_size=self.batch_size)
        network = MLP(
            n_inputs=self._x_dim,
            n_hidden=self.layers,
            n_outputs=len(self.quantiles) * self._y_dim,
            activation_fn=nn.ReLU(),
        )

        self._model = QR(
            model=network,
            optimizer=partial(Adam, lr=self.learning_rate),
            quantiles=list(self.quantiles),
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
    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        """
        Predicts using the QuantileRegression model by outputting the median quantile.
        """
        if self._model is None:
            raise ValueError("The model must be trained before calling predict.")

        X = self._x_scaler.transform(X)
        X = _to_tensor(X)
        self._model.eval()

        preds = self._model.predict_step(X)
        y = preds["pred"]
        y_np = y.cpu().numpy()
        y = self._y_scaler.inverse_transform(y_np)
        return y_np

    @torch.no_grad()
    def sample(
        self,
        X: Float[ndarray, "batch x_dim"],
        n_samples: int = 10,
        seed=None,
    ) -> Float[torch.Tensor, "n_samples batch y_dim"]:
        """
        Samples from the QuantileRegression model by linearly interpolating between predicted quantiles.
        """
        if self._model is None:
            raise ValueError("The model must be trained before calling sample.")
        X = _to_tensor(X)

        quantiles = self._model(X)  # shape: (batch, n_quantiles)
        samples = np.random.default_rng(seed).uniform(
            0, 1, size=(X.shape[0], n_samples)
        )  # shape: (batch, n_samples)
        samples_lst = []
        for i in range(X.shape[0]):
            samples_lst.append(
                np.interp(samples[i, :], self.quantiles, quantiles[i].cpu().numpy())
            )

        samples = np.array(samples_lst).T
        samples = samples[:, :, np.newaxis]

        samples = samples.reshape(-1, self._y_dim)
        samples = self._y_scaler.inverse_transform(samples)
        samples = samples.reshape(n_samples, -1, self._y_dim)

        return samples

    @staticmethod
    def search_space() -> dict:
        return {
            "n_layers": Integer(1, 5),
            "hidden_size": Integer(10, 500),
            "learning_rate": Real(1e-5, 1e-1, prior="log-uniform"),
        }
