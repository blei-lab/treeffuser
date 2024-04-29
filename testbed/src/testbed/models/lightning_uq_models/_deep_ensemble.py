import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch as t
from jaxtyping import Float
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import DeepEnsembleRegression
from lightning_uq_box.uq_methods import MVERegression
from numpy import ndarray
from sklearn.preprocessing import StandardScaler
from skopt.space import Integer
from skopt.space import Real
from torch.optim import Adam

from testbed.models.base_model import ProbabilisticModel
from testbed.models.lightning_uq_models._data_module import GenericDataModule
from testbed.models.lightning_uq_models._utils import _to_tensor


class DeepEnsemble(ProbabilisticModel):

    def __init__(
        self,
        n_layers: int = 3,
        hidden_size: int = 50,
        max_epochs: int = 300,
        burnin_epochs: int = 10,
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        enable_progress_bar: bool = True,
        use_gpu: bool = False,
        patience: int = 10,
        seed: int = 42,
        n_ensembles: int = 5,
    ):
        """
        Implements DeepEnsemble using the uq_box library for uncertainty quantification.

        DeepEnsemble trains multiple neural networks with different initializations and
        combines their predictions to estimate uncertainty.

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
            n_ensembles: The number of ensemble members to train.
            burnin_epochs: The number of initial epochs before contributing to the

        """
        self._models = None

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
        self.n_ensembles = n_ensembles
        self.burnin_epochs = burnin_epochs

        self.seed = seed
        self._my_temp_dir = tempfile.mkdtemp()

        self.scaler_x = None
        self.scaler_y = None

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def fit(
        self, X: Float[torch.Tensor, "batch x_dim"], y: Float[torch.Tensor, "batch y_dim"]
    ) -> None:
        """
        Fits the DeepEnsemble model using provided training data.
        """
        self._y_dim = y.shape[1]
        self._x_dim = X.shape[1]

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        X = self.scaler_x.fit_transform(X.copy())
        y = self.scaler_y.fit_transform(y.copy())

        # print has nans
        print("x has nans", np.isnan(X).any())
        print("y has nans", np.isnan(y).any())

        print("first 10 x", X[:10])
        print("first 10 y", y[:10])

        dm = GenericDataModule(X, y, batch_size=self.batch_size)

        trained_models = []
        for i in range(self.n_ensembles):
            network = MLP(
                n_inputs=self._x_dim,
                n_hidden=self.layers,
                n_outputs=self._y_dim * 2,  # mean and variance
                activation_fn=torch.nn.ReLU(),
            )

            ensemble_member = MVERegression(
                model=network,
                optimizer=partial(Adam, lr=self.learning_rate),
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
                gradient_clip_val=1,
            )
            trainer.fit(ensemble_member, dm)
            save_path = Path(self._my_temp_dir) / f"model_{i}.ckpt"
            trainer.save_checkpoint(save_path)
            trained_models.append({"base_model": ensemble_member, "ckpt_path": save_path})

        self._models = DeepEnsembleRegression(self.n_ensembles, trained_models)

    @torch.no_grad()
    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[torch.Tensor, "batch y_dim"]:
        """
        Predicts using the DeepEnsemble model by combining predictions from multiple models.
        """
        if self._models is None:
            raise ValueError("The model must be trained before calling predict.")

        X = self.scaler_x.transform(X.copy())
        X = _to_tensor(X)

        preds = self._models.predict_step(X)
        y = preds["pred"]
        y_np = y.cpu().numpy()
        y_np = self.scaler_y.inverse_transform(y_np)
        return y_np

    @torch.no_grad()
    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples: int = 10
    ) -> Float[torch.Tensor, "n_samples batch y_dim"]:
        """
        Samples from the DeepEnsemble model by combining samples from multiple models.
        """
        if self._models is None:
            raise ValueError("The model must be trained before calling sample.")
        X = self.scaler_x.transform(X.copy())
        X = _to_tensor(X)

        preds = self._models.predict_step(X)

        mean = preds["pred"]
        # std = preds["pred_uct"].reshape(-1, self._y_dim)
        std = preds["aleatoric_uct"].reshape(-1, self._y_dim)
        # print(preds)

        samples = (
            t.distributions.Normal(mean, std).sample((n_samples, 1)).squeeze()
        )  # batch, num_samples
        samples = samples.cpu().numpy()
        print("samples", samples)
        samples = self.scaler_y.inverse_transform(samples)
        # add extra dim
        return samples[:, :, None]

    @staticmethod
    def search_space() -> dict:
        return {
            "n_layers": Integer(1, 7),
            "hidden_size": Integer(10, 500),
            "learning_rate": Real(1e-5, 1e-1, prior="log-uniform"),
            "n_ensembles": Integer(2, 10),
            "patience": Integer(5, 50),
            "burnin_epochs": Integer(1, 30),
            "batch_size": Integer(16, 512),
        }
