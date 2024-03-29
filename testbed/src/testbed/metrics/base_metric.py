import abc
from typing import Dict

from jaxtyping import Array
from jaxtyping import Float

from testbed.models.base_model import ProbabilisticModel


class Metric(abc.ABC):
    @abc.abstractmethod
    def compute(
        self,
        model: ProbabilisticModel,
        X_test: Float[Array, "batch n_features"],
        y_test: Float[Array, "batch y_dim"],
    ) -> Dict[str, float]:
        pass
