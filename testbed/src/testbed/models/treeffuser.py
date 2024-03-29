from jaxtyping import Array
from jaxtyping import Float

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
        X: Float[Array, "batch x_dim"],
        y: Float[Array, "batch y_dim"],
    ) -> "ProbabilisticModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: Float[Array, "batch x_dim"]) -> Float[Array, "batch y_dim"]:
        y_samples = self.sample(X, n_samples=50)
        return y_samples.mean(axis=0)

    def sample(
        self, X: Float[Array, "batch x_dim"], n_samples=10
    ) -> Float[Array, "n_samples batch y_dim"]:
        return self.model.sample(X, n_samples, n_parallel=5, n_steps=50)
