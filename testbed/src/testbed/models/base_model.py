from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional

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
    ) -> None:
        """
        Fit the model to the data.
        """

    @abstractmethod
    def predict(self, X: Float[Array, "batch x_dim"]) -> Float[Array, "batch y_dim"]:
        """
        Predict the mean for each input.
        """

    @abstractmethod
    def predict_distribution(self, X: Float[Array, "batch x_dim"]):
        """
        Predict the probability distribution for each input.
        """


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
        cat_idx: Optional[List[int]] = None,
    ) -> None:
        self.model.fit(X, y, cat_idx)

    def predict(self, X: Float[Array, "batch x_dim"]) -> Float[Array, "batch y_dim"]:
        if "predict" not in self._cache:
            self._cache["predict"] = self.model.predict(X)
        return self._cache["predict"]

    def predict_distribution(self, X: Float[Array, "batch x_dim"]):
        if "predict_distribution" not in self._cache:
            self._cache["predict_distribution"] = self.model.predict_distribution(X)
        return self._cache["predict_distribution"]

    def clear_cache(self):
        self._cache = {}
