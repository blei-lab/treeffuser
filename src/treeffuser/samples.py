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
class Samples:
    def __init__(self, input_array):
        if input_array.ndim < 2 or input_array.ndim > 3:
            raise ValueError("Samples must have either 2 or 3 dimensions.")

        self.samples = input_array
        self.n_samples = input_array.shape[0]
        self.batch = input_array.shape[1]
        self.y_dim = 1 if input_array.ndim == 2 else input_array.shape[-1]
        self.shape = input_array.shape
        self.ndim = input_array.ndim

    def sample_mean(self) -> Float[np.ndarray, "batch y_dim"]:
        """
        Compute the mean of the samples for each `x`.
        Estimate: E[Y | X = x] for each x.

        Returns
        -------
        mean : np.ndarray
            The mean of the samples for each `x`.
        """
        return self.samples.mean(axis=0)

    def sample_confidence_interval(
        self, confidence: float = 0.95
    ) -> Float[np.ndarray, "2 batch y_dim"]:
        _check_unidimensional(self.samples)
        alpha = 1 - confidence
        return self.sample_quantile(q=[alpha / 2, 1 - alpha / 2])

    def sample_correlation(self) -> Float[np.ndarray, "batch y_dim y_dim"]:
        correlation = np.empty((self.batch, self.y_dim, self.y_dim))

        for i in range(self.batch):
            correlation[i, :, :] = np.corrcoef(self.samples[:, i, :], rowvar=False)

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
            if self.ndim == 2:
                y_i = self.samples[:, i, None]
            else:
                y_i = self.samples[:, i, :]
            kde = KernelDensity(bandwidth=bandwidth, algorithm="auto", kernel="gaussian")
            kde.fit(y_i)
            kdes.append(kde)

        return kdes

    def sample_max(self) -> Float[np.ndarray, "batch y_dim"]:
        return self.samples.max(axis=0)

    def sample_median(self) -> Float[np.ndarray, "batch y_dim"]:
        return np.median(self.samples, axis=0)

    def sample_min(self) -> Float[np.ndarray, "batch y_dim"]:
        return self.samples.min(axis=0)

    def sample_mode(
        self,
        bandwidth: Union[float, Literal["scott", "silverman"]] = 1.0,
        verbose: bool = False,
    ) -> Float[np.ndarray, "batch"]:
        _check_unidimensional(self.samples)

        kdes = self.sample_kde(bandwidth=bandwidth)

        modes = []
        n_grid = np.max([2 * self.batch, 1000])  # heuristic for the grid granularity
        for i in tqdm(range(self.batch), disable=not verbose, desc="Searching for modes"):
            if self.samples.ndim == 2:
                y_i = self.samples[:, i, None]
            else:
                y_i = self.samples[:, i, :]
            grid = np.linspace(np.min(y_i), np.max(y_i), n_grid)
            log_density = kdes[i].score_samples(grid.reshape(-1, 1))
            modes.append(grid[np.argmax(log_density)])

        modes = np.array(modes)
        return modes

    def sample_quantile(self, q) -> Float[np.ndarray, "q_dim batch y_dim"]:
        return np.quantile(self.samples, q, axis=0)

    def sample_range(self) -> Float[np.ndarray, "batch 2"]:
        _check_unidimensional(self.samples)
        return np.stack((self.samples.min(axis=0), self.samples.max(axis=0)), axis=-1)

    def sample_std(self) -> Float[np.ndarray, "batch y_dim"]:
        return self.samples.std(axis=0)

    def __getitem__(self, key):
        """
        Prevent the user from removing the first or second dimension of the samples.
        """
        if isinstance(key, int):
            key = (key,)
        if isinstance(key[0], int):
            raise ValueError(
                f"Accessing `my_samples[{key}] would remove the first dimension of the samples,"
                f"which is forbidden. Instead, use `my_samples.samples[{key}]`."
            )
        if len(key) >= 2 and isinstance(key[1], int):
            # If key[0] is an ellipsis and self.ndim == 3 then key[1] actually refers to the
            # third dimension, which is allowed
            if not (key[0] is Ellipsis and self.ndim == 3):
                raise ValueError(
                    f"Accessing `my_samples[{key}] would remove the second dimension of the "
                    f"samples which is forbidden. Instead, use `my_samples.samples[{key}]`."
                )
        return Samples(self.samples.__getitem__(key))
