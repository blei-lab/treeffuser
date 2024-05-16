from typing import Tuple

from jaxtyping import Float
from numpy import ndarray


class SimulatedDataset:
    """
    Abstract class for simulated datasets.
    Slightly different than a real dataset.
    """

    def sample(
        self, n: int, seed: int
    ) -> Tuple[Float[ndarray, "n x_dim"], Float[ndarray, "n y_dim"]]:
        """
        Sample n data points from the dataset.
        """
        raise NotImplementedError

    def score(
        self, x: Float[ndarray, "batch x_dim"], y: Float[ndarray, "samples batch y_dim"]
    ) -> Float[ndarray, "n"]:
        """
        Computes the log-likelihood of the data points y
        according to the he distribution $y|x$.
        """
        raise NotImplementedError
