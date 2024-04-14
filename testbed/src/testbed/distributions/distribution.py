from abc import ABC
from abc import abstractmethod

from jaxtyping import Float
from numpy import ndarray


class Distribution(ABC):
    """
    Represents a probability distribution of the form p(y | x).
    where x has shape (batch, x_dim) and y has shape (batch, y_dim).

    Each method should implement its own distribution function.
    """

    @abstractmethod
    def sample(self, n_samples: int) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Sample from the probability distribution for each input.
        """

    @abstractmethod
    def log_likelihood(self, y: Float[ndarray, "batch y_dim"]) -> Float[ndarray, "batch"]:
        """
        Compute the log likelihood of the data under the model.
        """

    @property
    @abstractmethod
    def x_dim(self) -> int:
        """
        The dimension of the input.
        """

    @property
    @abstractmethod
    def y_dim(self) -> int:
        """
        The dimension of the output.
        """

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """
        The batch size of the distribution.
        """

    def __init__(self):
        super().__init__()
