import abc
from typing import Dict

from jaxtyping import Float
from numpy import ndarray

from testbed.models.base_model import ProbabilisticModel


class Metric(abc.ABC):
    def __init__(self):
        self.seed = 0

    @abc.abstractmethod
    def compute(
        self,
        model: ProbabilisticModel,
        X_test: Float[ndarray, "batch n_features"],
        y_test: Float[ndarray, "batch y_dim"],
    ) -> Dict[str, float]:
        pass

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()
