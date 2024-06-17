from typing import List
from typing import Literal
from typing import Union

import numpy as np
from jaxtyping import Float
from sklearn.neighbors import KernelDensity


def _check_unidimensional(array) -> None:
    if array.shpae[-1] > 1:
        raise ValueError("This method only applies to unidimensional responses.")


class Samples(np.ndarray):
    def __new__(cls, input_array):
        if input_array.ndim < 2 or input_array.ndim > 3:
            raise ValueError("Samples must have either 2 or 3 dimensions.")

        return np.asarray(input_array).view(cls)

    def __init__(self):
        pass

    def kde(
        self,
        bandwidth: Union[float, Literal["scott", "silverman"]] = 1.0,
    ) -> List[KernelDensity]:
        batch = self.shape[2]

        kdes = []
        for i in range(batch):
            if self.ndim == 2:
                y_i = self[:, i]
            else:
                y_i = self[:, i, :]
            kde = KernelDensity(bandwidth=bandwidth, algorithm="auto", kernel="gaussian")
            kde.fit(y_i)
            kdes.append(kde)

        return kdes

    def mean(self, axis=0) -> Float[np.ndarray, "batch y_dim"]:
        return np.mean(self, axis=axis)

    def median(self, axis=0) -> Float[np.ndarray, "batch y_dim"]:
        _check_unidimensional(self)
        return np.median(self, axis=axis)

    def quantile(self, q, axis=0) -> Float[np.ndarray, "batch y_dim q_dim"]:
        _check_unidimensional(self)
        return np.quantile(self, q, axis=axis)

    def range(self) -> Float[np.ndarray, "batch 2"]:
        _check_unidimensional(self)
        return np.stack((np.min(self, axis=0), np.max(self, axis=0)), axis=-1)

    def std(self, axis=0) -> Float[np.ndarray, "batch y_dim"]:
        _check_unidimensional(self)
        return np.std(self, axis=axis)
