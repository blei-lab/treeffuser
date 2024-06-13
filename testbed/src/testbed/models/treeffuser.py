from typing import List
from typing import Optional

from jaxtyping import Float
from numpy import ndarray
from sklearn.base import MultiOutputMixin
from skopt.space import Integer
from skopt.space import Real

import treeffuser as tf

from .base_model import ProbabilisticModel


class Treeffuser(ProbabilisticModel, MultiOutputMixin):
    """
    Wrapping the LightGBMTreeffuser model as a ProbabilisticModel.
    """

    def __init__(
        self,
        n_estimators: int = 3000,
        n_repeats: int = 30,
        learning_rate: float = 0.1,
        early_stopping_rounds: int = 50,
        num_leaves: int = 31,
        seed: int = 0,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        verbose: bool = 0,
        sde_initialize_from_data: bool = False,
        sde_hyperparam_min: Optional[float] = None,
        sde_hyperparam_max: Optional[float] = None,
    ):
        super().__init__(seed)
        self.n_estimators = n_estimators
        self.n_repeats = n_repeats
        self.learning_rate = learning_rate
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.verbose = verbose
        self.sde_initialize_from_data = sde_initialize_from_data
        self.sde_hyperparam_min = sde_hyperparam_min
        self.sde_hyperparam_max = sde_hyperparam_max
        self.model = None

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        cat_idx: Optional[List[int]] = None,
    ) -> "ProbabilisticModel":
        self.model = tf.Treeffuser(
            n_estimators=self.n_estimators,
            n_repeats=self.n_repeats,
            sde_name="vesde",
            sde_initialize_from_data=self.sde_initialize_from_data,
            sde_hyperparam_min=self.sde_hyperparam_min,
            sde_hyperparam_max=self.sde_hyperparam_max,
            learning_rate=self.learning_rate,
            early_stopping_rounds=self.early_stopping_rounds,
            num_leaves=self.num_leaves,
            seed=self.seed,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            verbose=self.verbose,
        )

        self.model.fit(X, y, cat_idx)
        return self

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        y_samples = self.sample(X, n_samples=50, seed=0)
        return y_samples.mean(axis=0)

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10, seed=None
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        return self.model.sample(
            X, n_samples, n_parallel=10, n_steps=50, seed=seed, verbose=True
        )

    @staticmethod
    def search_space() -> dict:
        return {
            "n_estimators": Integer(100, 3000, "log-uniform"),
            "n_repeats": Integer(10, 50),
            "learning_rate": Real(0.01, 1, "log-uniform"),
            "early_stopping_rounds": Integer(10, 100),
            "num_leaves": Integer(10, 100),
        }

    def get_extra_stats(self) -> dict:
        n_estimators_true = self.model.n_estimators_true
        if len(n_estimators_true) == 1:
            n_estimators_true = n_estimators_true[0]
        return {"n_estimators_true": n_estimators_true}
