from abc import ABC
from abc import abstractmethod

from jaxtyping import Float, Array
from sklearn.base import BaseEstimator


class ProbabilisticModel(ABC, BaseEstimator):
    """
    A base class for all probabilistic models. Which produces a probability distribution
    rather than a single output for each input. Subclasses BaseEstimator.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(
        self,
        X: Float[Array, "batch x_dim"],
        y: Float[Array, "batch y_dim"],
    ) -> "ProbabilisticModel":
        """
        Fit the model to the data.
        """

    @abstractmethod
    def predict(self, X: Float[Array, "batch x_dim"]) -> Float[Array, "batch y_dim"]:
        """
        Predict the mean for each input.
        """

    @abstractmethod
    def sample(
        self, X: Float[Array, "batch x_dim"], n_samples=10
    ) -> Float[Array, "n_samples batch y_dim"]:
        """
        Sample from the probability distribution for each input.
        """

    # @abstractmethod
    # def predict_distribution(self, X: Float[Array, "batch x_dim"]):
    #     """
    #     Predict the probability distribution for each input.
    #     """


class CachedProbabilisticModel(ProbabilisticModel):
    """
    A probabilistic model that caches the predictions for each input, for faster evaluation.
    """

    def __init__(self, model: ProbabilisticModel):
        super().__init__()
        self._cache = {}
        self.model = model

    def fit(
        self,
        X: Float[Array, "batch x_dim"],
        y: Float[Array, "batch y_dim"],
    ) -> ProbabilisticModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: Float[Array, "batch x_dim"]) -> Float[Array, "batch y_dim"]:
        if "predict" not in self._cache:
            self._cache["predict"] = self.model.predict(X)
        return self._cache["predict"]

    def sample(
        self, X: Float[Array, "batch x_dim"], n_samples=10
    ) -> Float[Array, "n_samples batch y_dim"]:
        if "sample" not in self._cache:
            self._cache["sample"] = self.model.sample(X, n_samples)
        return self._cache["sample"]

    def clear_cache(self):
        self._cache = {}
