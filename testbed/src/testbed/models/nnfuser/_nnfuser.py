"""
This should be the main file corresponding to the project.
"""

from typing import Dict
from typing import List
from typing import Optional

from jaxtyping import Float
from ml_collections import ConfigDict
from numpy import ndarray
from skopt.space import Integer
from skopt.space import Real

import testbed.models.nnfuser._score as _score
from testbed.models.base_model import ProbabilisticModel
from testbed.models.base_model import SupportsMultioutput
from treeffuser.treeffuser import Treeffuser


class _NNffuser(Treeffuser):
    """
    Private class implementing Treffuser API but
    for neural networks.
    """

    def __init__(
        self,
        # Diffusion model args
        sde_name: str = "vesde",
        sde_initialize_with_data: bool = False,
        sde_manual_hyperparams: Optional[dict] = None,
        n_repeats: int = 30,
        # Score estimator args
        n_layers: int = 1,
        hidden_size: int = 50,
        max_epochs: int = 300,
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        use_gpu: bool = False,
        patience: int = 2,
        seed: int = 42,
        burnin_epochs: int = 0,
        enable_progress_bar: bool = False,
    ):
        """
        Diffusion model args
        -------------------------------
        sde_name (str): The SDE name.
        sde_initialize_with_data (bool): Whether to initialize the SDE hyperparameters
            with data.
        sde_manual_hyperparams: (dict): A dictionary for explicitly setting the SDE
            hyperparameters, overriding default or data-based initializations.
        n_repeats (int): How many times to repeat the training dataset. i.e how
            many noisy versions of a point to generate for training.

        NNfuser args (see _score_modesl) for more details
        -------------------------------
        """
        if sde_initialize_with_data and sde_manual_hyperparams is not None:
            raise Warning(
                "Manual hypeparameter setting will override data-based initialization."
            )

        super().__init__(
            sde_name=sde_name,
            sde_initialize_with_data=sde_initialize_with_data,
            sde_manual_hyperparams=sde_manual_hyperparams,
        )
        self._score_config = ConfigDict(
            {
                "n_repeats": n_repeats,
                "n_layers": n_layers,
                "hidden_size": hidden_size,
                "max_epochs": max_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "use_gpu": use_gpu,
                "patience": patience,
                "seed": seed,
                "burnin_epochs": burnin_epochs,
                "enable_progress_bar": enable_progress_bar,
            }
        )

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        cat_idx: Optional[List[int]] = None,
    ):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape (batch, x_dim).
        y : np.ndarray
            Target data with shape (batch, y_dim).
        cat_idx: list
            List with indices of the columns of X that are categorical.

        Parameters
        ----------
        An instance of the model for chaining.
        """
        if cat_idx:
            self._x_cat_idx = cat_idx
            self._score_config.update({"categorical_features": cat_idx})

        super().fit(X, y)

    @property
    def score_config(self):
        return self._score_config

    @property
    def _score_model_class(self):
        return _score.NNScore


class NNffuser(ProbabilisticModel, SupportsMultioutput):
    """
    Wrapping the LightGBMTreeffuser model as a ProbabilisticModel.
    """

    def __init__(
        self,
        n_layers: int = 3,
        hidden_size: int = 10,
        max_epochs: int = 3000,
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        use_gpu: bool = False,
        patience: int = 4,
        seed: int = 42,
        burnin_epochs: int = 1,
        n_repeats: int = 10,
        enable_progress_bar: bool = False,
        sde_initialize_with_data: bool = False,
        sde_manual_hyperparams: Optional[Dict[str, float]] = None,
    ):
        super().__init__(seed)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.patience = patience
        self.seed = seed
        self.n_repeat = n_repeats
        self.burnin_epochs = burnin_epochs
        self.enable_progress_bar = enable_progress_bar
        self.sde_initialize_with_data = sde_initialize_with_data
        self.sde_manual_hyperparams = sde_manual_hyperparams

        self.model = None

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        cat_idx: Optional[List[int]] = None,
    ) -> "ProbabilisticModel":

        self.model = _NNffuser(
            n_layers=self.n_layers,
            hidden_size=self.hidden_size,
            max_epochs=self.max_epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            use_gpu=self.use_gpu,
            patience=self.patience,
            seed=self.seed,
            n_repeats=self.n_repeat,
            burnin_epochs=self.burnin_epochs,
            enable_progress_bar=self.enable_progress_bar,
            sde_initialize_with_data=self.sde_initialize_with_data,
            sde_manual_hyperparams=self.sde_manual_hyperparams,
        )

        self.model.fit(X, y, cat_idx)
        return self

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        y_samples = self.sample(X, n_samples=50, seed=0)
        return y_samples.mean(axis=0)

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10, seed=None
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        return self.model.sample(X, n_samples, n_parallel=10, n_steps=50, seed=seed)

    @staticmethod
    def search_space() -> dict:
        return {
            "n_layers": Integer(1, 7),
            "hidden_size": Integer(10, 1000),
            "max_epochs": Integer(10, 1000),
            "learning_rate": Real(1e-3, 1),
            "patience": Integer(1, 100),
            "burnin_epochs": Integer(1, 20),
        }
