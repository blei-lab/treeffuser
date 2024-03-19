from abc import ABC
from abc import abstractmethod

from jaxtyping import Float
from numpy import ndarray


class ProbabilisticModel(ABC):
    """
    A base class for all probabilistic models. Which produce a probability distribution
    rather than a single output for each input.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]):
        """
        Fit the model to the data.
        """

    @abstractmethod
    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        """
        Predict the probability distribution for each input.
        """

    @abstractmethod
    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples: int
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Sample from the probability distribution for each input.
        """

    @abstractmethod
    def log_likelihood(
        self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]
    ) -> Float[ndarray, "batch"]:
        """
        Compute the log likelihood of the data under the model.
        """
