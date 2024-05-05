from jaxtyping import Float
from ngboost import NGBRegressor
from numpy import ndarray
from skopt.space import Integer
from skopt.space import Real

from testbed.models.base_model import ProbabilisticModel
from testbed.models.ngboost._gaussian_mixtures import build_gaussian_mixture_model

MAX_MINIBATCH_SIZE = 50_000
MAX_VALIDATION_SIZE = 20_000


class NGBoostGaussian(ProbabilisticModel):
    """
    A probabilistic model that uses NGBoost with a Gaussian likelihood.

    NGBoost only accepts 1 dimensional y values.
    """

    def __init__(self, n_estimators: int = 5000, learning_rate: float = 0.05):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = None

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
        minibatch_frac = min(MAX_MINIBATCH_SIZE, X.shape[0]) / X.shape[0]
        validation_fraction = min(int(0.1 * X.shape[0]), MAX_VALIDATION_SIZE) / X.shape[0]

        self.model = NGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            early_stopping_rounds=10,
            minibatch_frac=minibatch_frac,
            validation_fraction=validation_fraction,
            verbose_eval=50,
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

    @staticmethod
    def search_space() -> dict:
        """
        Return the search space for parameters of the model.
        """
        return {
            "n_estimators": Integer(100, 10000),
            "learning_rate": Real(0.01, 1),
        }


class NGBoostMixtureGaussian(ProbabilisticModel):
    """
    A probabilistic model that uses NGBoost with a Gaussian mixture likelihood.
    """

    def __init__(self, n_estimators: int = 5000, learning_rate: float = 0.05, k=10):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.k = k
        self._dist = build_gaussian_mixture_model(k)
        self.model = None

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

        minibatch_frac = min(MAX_MINIBATCH_SIZE, X.shape[0]) / X.shape[0]
        validation_fraction = min(int(0.1 * X.shape[0]), MAX_VALIDATION_SIZE) / X.shape[0]

        self.model = NGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            Dist=self._dist,
            early_stopping_rounds=10,
            natural_gradient=False,
            minibatch_frac=minibatch_frac,
            validation_fraction=validation_fraction,
            verbose_eval=1,
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

    @staticmethod
    def search_space() -> dict:
        """
        Return the search space for parameters of the model.
        """
        return {
            "n_estimators": Integer(100, 10000),
            "learning_rate": Real(0.01, 1),
            "k": Integer(1, 50),
        }
