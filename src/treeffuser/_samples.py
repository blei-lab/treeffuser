from typing import List
from typing import Literal
from typing import Union

import numpy as np
from jaxtyping import Float
from sklearn.neighbors import KernelDensity
from tqdm import tqdm


###################################################
# Helper functions
###################################################
def _check_unidimensional(array) -> None:
    if array.shpae[-1] > 1:
        raise ValueError("This method only applies to unidimensional responses.")


###################################################
# Main class
###################################################
class Samples(np.ndarray):
    def __new__(cls, input_array):
        if input_array.ndim < 2 or input_array.ndim > 3:
            raise ValueError("Samples must have either 2 or 3 dimensions.")

        return np.asarray(input_array).view(cls)

    def __init__(self):
        pass

    def correlation(self) -> Float[np.ndarray, "batch y_dim y_dim"]:
        _, batch, y_dim = self.shape
        correlation = np.empty((batch, y_dim, y_dim))

        for i in range(batch):
            correlation[i, :, :] = np.corrcoef(self[:, i, :], rowvar=False)

        return correlation

    def kde(
        self,
        bandwidth: Union[float, Literal["scott", "silverman"]] = 1.0,
        verbose: bool = True,
    ) -> List[KernelDensity]:
        batch = self.shape[2]

        kdes = []
        for i in tqdm(
            range(batch), disable=not verbose, desc="Fitting kernel densities for each x"
        ):
            if self.ndim == 2:  # samples may have shape (n_samples, batch) when y_dim=1
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

    def mode(
        self, bandwidth: Union[float, Literal["scott", "silverman"]] = 1.0, verbose=True
    ) -> Float[np.ndarray, "batch y_dim"]:
        _check_unidimensional(self)

        batch = self.shape[2]
        kdes = self.kde(bandwidth=bandwidth)

        modes = []
        for i in range(batch, disable=not verbose):
            if self.ndim == 2:
                y_i = self[:, i]
            else:
                y_i = self[:, i, :]
            grid = np.linspace(np.min(y_i), np.max(y_i), 1000).flatten()
            log_density = kdes[i].score_samples(grid)
            modes.append(grid[np.argmax(log_density)])
        return modes

    def quantile(self, q, axis=0) -> Float[np.ndarray, "batch y_dim q_dim"]:
        _check_unidimensional(self)
        return np.quantile(self, q, axis=axis)

    def range(self) -> Float[np.ndarray, "batch 2"]:
        _check_unidimensional(self)
        return np.stack((np.min(self, axis=0), np.max(self, axis=0)), axis=-1)

    def std(self, axis=0) -> Float[np.ndarray, "batch y_dim"]:
        _check_unidimensional(self)
        return np.std(self, axis=axis)
