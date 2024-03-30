from jaxtyping import Float
from ngboost import NGBRegressor
from numpy import ndarray

from testbed.models.base_model import ProbabilisticModel
from testbed.models.ngboost._gaussian_mixtures import build_gaussian_mixture_model


class NGBoostGaussian(ProbabilisticModel):
    """
    A probabilistic model that uses NGBoost with a Gaussian likelihood.

    NGBoost only accepts 1 dimensional y values.
    """

    def __init__(self, n_estimators: int = 5000, learning_rate: float = 0.05):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ) -> ProbabilisticModel:
        """
        Fit the model to the data.
        """

        if y.shape[1] > 1:
            raise ValueError("NGBoost only accepts 1 dimensional y values.")

        y = y[:, 0]

        self.model = NGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            early_stopping_rounds=10,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        """
        Predict the mean for each input.
        """
        return self.model.predict(X).reshape(-1, 1)

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Sample from the probability distribution for each input.
        """
        return self.model.pred_dist(X).sample(n_samples).reshape(n_samples, -1, 1)


class NGBoostMixtureGaussian(ProbabilisticModel):
    """
    A probabilistic model that uses NGBoost with a Gaussian mixture likelihood.
    """

    def __init__(self, n_estimators: int = 5000, learning_rate: float = 0.05, k=3):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.k = k
        self._dist = build_gaussian_mixture_model(k)

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ) -> ProbabilisticModel:
        """
        Fit the model to the data.
        """
        if y.shape[1] > 1:
            raise ValueError("NGBoost only accepts 1 dimensional y values.")

        y = y[:, 0]

        self.model = NGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            Dist=self._dist,
            early_stopping_rounds=10,
            natural_gradient=False,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        """
        Predict the mean for each input.
        """
        return self.model.predict(X).reshape(-1, 1)

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Sample from the probability distribution for each input.
        """
        return self.model.pred_dist(X).sample(n_samples).reshape(n_samples, -1, 1)
