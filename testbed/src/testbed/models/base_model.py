from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional

from jaxtyping import Float
from numpy import ndarray
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
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        cat_idx: Optional[List[int]] = None,
    ) -> None:
        """
        Fit the model to the data.
        """

    @abstractmethod
    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        """
        Predict the mean for each input.
        """

    @abstractmethod
    def predict_distribution(self, X: Float[ndarray, "batch x_dim"]):
        """
        Predict the probability distribution for each input.
        """
