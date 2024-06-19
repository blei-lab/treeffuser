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
    if array.ndim > 2 and array.shape[-1] > 1:
        raise ValueError("This method only applies to unidimensional responses.")


###################################################
# Main class
###################################################
class Samples(np.ndarray):
    def __new__(cls, input_array):
        if input_array.ndim < 2 or input_array.ndim > 3:
            raise ValueError("Samples must have either 2 or 3 dimensions.")

        samples = np.asarray(input_array).view(cls)
        samples.n_samples = input_array.shape[0]
        samples.batch = input_array.shape[1]
        samples.y_dim = 1 if input_array.ndim == 2 else input_array.shape[-1]
        return samples

    def __array_finalize__(self, obj):
        # this ensures that attributes are propagated to new instances created
        # from numpy operations
        if obj is None:
            return
        self.n_samples = getattr(obj, "n_samples", None)
        self.batch = getattr(obj, "batch", None)
        self.y_dim = getattr(obj, "y_dim", None)

    def sample_confidence_interval(
        self, confidence: float = 0.95
    ) -> Float[np.ndarray, "2 batch y_dim"]:
        _check_unidimensional(self)
        alpha = 1 - confidence
        return self.sample_quantile(q=[alpha / 2, 1 - alpha / 2])

    def sample_correlation(self) -> Float[np.ndarray, "batch y_dim y_dim"]:
        correlation = np.empty((self.batch, self.y_dim, self.y_dim))

        for i in range(self.batch):
            correlation[i, :, :] = np.corrcoef(self[:, i, :], rowvar=False)

        return correlation

    def sample_kde(
        self,
        bandwidth: Union[float, Literal["scott", "silverman"]] = 1.0,
        verbose: bool = False,
    ) -> List[KernelDensity]:

        kdes = []
        for i in tqdm(
            range(self.batch), disable=not verbose, desc="Fitting kernel densities for each x"
        ):
            if self.ndim == 2:  # samples may have shape (n_samples, batch) when y_dim=1
                y_i = self[:, i]
            else:
                y_i = self[:, i, :]
            kde = KernelDensity(bandwidth=bandwidth, algorithm="auto", kernel="gaussian")
            kde.fit(y_i)
            kdes.append(kde)

        return kdes

    def sample_max(self) -> Float[np.ndarray, "batch y_dim"]:
        return super().max(axis=0)

    def sample_mean(self) -> Float[np.ndarray, "batch y_dim"]:
        return super().mean(axis=0)

    def sample_median(self) -> Float[np.ndarray, "batch y_dim"]:
        return np.median(self, axis=0)

    def sample_min(self) -> Float[np.ndarray, "batch y_dim"]:
        return super().min(axis=0)

    def sample_mode(
        self,
        bandwidth: Union[float, Literal["scott", "silverman"]] = 1.0,
        verbose: bool = False,
    ) -> List[np.ndarray]:
        _check_unidimensional(self)

        kdes = self.sample_kde(bandwidth=bandwidth)

        modes = []
        n_grid = np.max([2 * self.batch, 1000])  # heuristic for the grid granularity
        for i in tqdm(range(self.batch), disable=not verbose, desc="Searching for modes"):
            if self.ndim == 2:
                y_i = self[:, i]
            else:
                y_i = self[:, i, :]
            grid = np.linspace(np.min(y_i), np.max(y_i), n_grid)
            log_density = kdes[i].score_samples(grid.reshape(-1, 1))
            modes.append(grid[np.argmax(log_density)])
        return modes

    def sample_quantile(self, q) -> Float[np.ndarray, "q_dim batch y_dim"]:
        return np.quantile(self, q, axis=0)

    def sample_range(self) -> Float[np.ndarray, "batch 2"]:
        _check_unidimensional(self)
        return np.stack((self.min(axis=0), self.max(axis=0)), axis=-1)

    def sample_std(self, axis: int = 0) -> Float[np.ndarray, "batch y_dim"]:
        return super().std(axis=0)
