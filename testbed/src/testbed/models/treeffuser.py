from typing import Dict
from typing import Optional

from jaxtyping import Float
from numpy import ndarray
from skopt.space import Integer
from skopt.space import Real

from treeffuser.treeffuser import LightGBMTreeffuser

from .base_model import ProbabilisticModel


class Treeffuser(ProbabilisticModel):
    """
    Wrapping the LightGBMTreeffuser model as a ProbabilisticModel.
    """

    def __init__(
        self,
        n_estimators: int = 10000,
        n_repeats: int = 50,
        learning_rate: float = 0.5,
        early_stopping_rounds: int = 50,
        num_leaves: int = 31,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        verbose: bool = 0,
        sde_manual_hyperparams: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.n_repeats = n_repeats
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.num_leaves = num_leaves

        print(subsample)
        print(subsample_freq)

        self.model = LightGBMTreeffuser(
            n_estimators=n_estimators,
            n_repeats=n_repeats,
            sde_name="vesde",
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            num_leaves=num_leaves,
            subsample=subsample,
            subsample_freq=subsample_freq,
            verbose=verbose,
            sde_manual_hyperparams=sde_manual_hyperparams,
        )

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ) -> "ProbabilisticModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        y_samples = self.sample(X, n_samples=50)
        return y_samples.mean(axis=0)

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        return self.model.sample(X, n_samples, n_parallel=5, n_steps=50)

    @staticmethod
    def search_space() -> dict:
        return {
            "n_estimators": Integer(100, 2000, "log-uniform"),
            "n_repeats": Integer(10, 100),
            "learning_rate": Real(0.01, 1),
            "early_stopping_rounds": Integer(1, 100),
            "num_leaves": Integer(2, 100),
        }
