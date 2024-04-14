from jaxtyping import Float
from numpy import ndarray

from treeffuser.treeffuser import LightGBMTreeffuser

from .base_model import ProbabilisticModel


class Treeffuser(ProbabilisticModel):
    """
    Wrapping the LightGBMTreeffuser model as a ProbabilisticModel.
    """

    def __init__(self, n_estimators: int = 1000):
        super().__init__()
        self.n_estimators = n_estimators
        self.model = LightGBMTreeffuser(
            n_estimators=n_estimators,
            n_repeats=30,
            sde_name="vesde",
            learning_rate=0.01,
            early_stopping_rounds=50,
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
