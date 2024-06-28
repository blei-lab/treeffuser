from typing import Callable
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
    """
    A wrapper class for the output of Treeffuser, the samples from the
    conditional distribution `p(y|x)`. It provides convenient methods to
    compute various statistics from the samples.

    Parameters
    ----------
    input_array : np.ndarray
        An array containing samples with shape (n_samples, batch, y_dim).

    Attributes
    ----------
    n_samples : int
        Number of samples.
    batch : int
        Batch size.
    y_dim : int
        Dimension of the response variable.
    shape : tuple
        Shape of the samples array.
    ndim : int
        Number of dimensions of the samples array.
    """

    def __init__(self, input_array: Float[np.ndarray, "n_samples batch y_dim"]):
        if input_array.ndim < 2 or input_array.ndim > 3:
            raise ValueError("Samples must have either 2 or 3 dimensions.")

        self._samples = input_array
        self.n_samples = input_array.shape[0]
        self.batch = input_array.shape[1]
        self.y_dim = 1 if input_array.ndim == 2 else input_array.shape[-1]
        self.shape = input_array.shape
        self.ndim = input_array.ndim

    def sample_apply(
        self, fun: Callable[[np.ndarray], np.ndarray]
    ) -> Float[np.ndarray, "batch y_dim"]:
        """
        Apply a function to the samples for each `x`.

        Parameters
        ----------
        func : callable
            A function to apply to each sample. The function should take a numpy array of shape
            (n_samples,) and return a numpy array of the same shape.

        Returns
        -------
        result : np.ndarray
            The result of applying the function to each row of the samples.
        """
        result = np.apply_along_axis(fun, 0, self._samples)
        return result

    def sample_confidence_interval(
        self, confidence: float = 0.95
    ) -> Float[np.ndarray, "2 batch y_dim"]:
        """
        Estimate the confidence interval of the samples for each `x` using
        the empirical quantiles of the samples.

        Parameters
        ----------
        confidence : float
            The confidence level for the interval.

        Returns
        -------
        ci : np.ndarray
            The confidence interval of the samples for each `x`.
        """
        _check_unidimensional(self._samples)
        alpha = 1 - confidence
        return self.sample_quantile(q=[alpha / 2, 1 - alpha / 2])

    def sample_correlation(self) -> Float[np.ndarray, "batch y_dim y_dim"]:
        """
        Compute the correlation matrix of the samples for each `x`.
        Estimate: `corr[Y | X = x]` for each `x`.

        Returns
        -------
        correlation : np.ndarray
            The correlation matrix of the samples for each `x`.
        """
        correlation = np.empty((self.batch, self.y_dim, self.y_dim))

        for i in range(self.batch):
            correlation[i, :, :] = np.corrcoef(self._samples[:, i, :], rowvar=False)

        return correlation

    def sample_kde(
        self,
        bandwidth: Union[float, Literal["scott", "silverman"]] = 1.0,
        verbose: bool = False,
    ) -> List[KernelDensity]:
        """
        Compute the Kernel Density Estimate (KDE) for each `x`.
        Estimate: `KDE[Y | X = x]` for each `x` using Gaussian kernels from `sklearn.neighbors`.

        Parameters
        ----------
        bandwidth : float or {'scott', 'silverman'}, default=1.0
            The bandwidth of the kernel. Bandwidth can be specified as a scalar value
            or as a string:
            - 'scott': Scott's rule of thumb.
            - 'silverman': Silverman's rule of thumb.
        verbose : bool, default=False
            Whether to display progress bars.

        Returns
        -------
        kdes : list of KernelDensity
            A list of `KernelDensity` objects, one for each `x`.
        """
        kdes = []
        for i in tqdm(
            range(self.batch),
            disable=not verbose,
            desc="Fitting kernel densities for each `x`",
        ):
            if self.ndim == 2:
                y_i = self._samples[:, i, None]
            else:
                y_i = self._samples[:, i, :]
            kde = KernelDensity(bandwidth=bandwidth, algorithm="auto", kernel="gaussian")
            kde.fit(y_i)
            kdes.append(kde)

        return kdes

    def sample_max(self) -> Float[np.ndarray, "batch y_dim"]:
        """
        Compute the maximum of the samples for each `x`.
        Estimate: `max[Y | X = x]` for each `x`.
        Equivalent to `np.max(samples.to_numpy(), axis=0)`.

        Returns
        -------
        max : np.ndarray
            The maximum of the samples for each `x`.
        """
        return self._samples.max(axis=0)

    def sample_mean(self) -> Float[np.ndarray, "batch y_dim"]:
        """
        Compute the mean of the samples for each `x`.
        Estimate: `E[Y | X = x]` for each `x`.
        Equivalent to `np.mean(samples.to_numpy(), axis=0)`.

        Returns
        -------
        mean : np.ndarray
            The mean of the samples for each `x`.
        """
        return self._samples.mean(axis=0)

    def sample_median(self) -> Float[np.ndarray, "batch y_dim"]:
        """
        Compute the median of the samples for each `x`.
        Estimate: `median[Y | X = x]` for each `x`.
        Equivalent to `np.median(samples.to_numpy(), axis=0)`.

        Returns
        -------
        median : np.ndarray
            The median of the samples for each `x`.
        """
        return np.median(self._samples, axis=0)

    def sample_min(self) -> Float[np.ndarray, "batch y_dim"]:
        """
        Compute the minimum of the samples for each `x`.
        Estimate: `min[Y | X = x]` for each `x`.
        Equivalent to `np.min(samples.to_numpy(), axis=0)`.

        Returns
        -------
        min : np.ndarray
            The minimum of the samples for each `x`.
        """
        return self._samples.min(axis=0)

    def sample_mode(
        self,
        bandwidth: Union[float, Literal["scott", "silverman"]] = 1.0,
        verbose: bool = False,
    ) -> Float[np.ndarray, "batch"]:
        """
        Compute the mode of the samples for each `x`.
        Estimate: `mode[Y | X = x]` for each `x` using Kernel Density Estimation.

        Parameters
        ----------
        bandwidth : float or {'scott', 'silverman'}, default=1.0
            The bandwidth of the kernel. Bandwidth can be specified as a scalar value
            or as a string:
            - 'scott': Scott's rule of thumb.
            - 'silverman': Silverman's rule of thumb.
        verbose : bool, default=False
            Whether to display progress bars.

        Notes
        -----
        The mode is computed via grid search on the Kernel Density Estimate (KDE). The step size
        of the grid is set to be equal to the maximum between twice the number of batches and
        1,000.

        Returns
        -------
        mode : np.ndarray
            The mode of the samples for each `x`.
        """
        _check_unidimensional(self._samples)

        kdes = self.sample_kde(bandwidth=bandwidth)

        modes = []
        n_grid = np.max([2 * self.batch, 1000])  # heuristic for the grid granularity
        for i in tqdm(range(self.batch), disable=not verbose, desc="Searching for modes"):
            if self._samples.ndim == 2:
                y_i = self._samples[:, i, None]
            else:
                y_i = self._samples[:, i, :]
            grid = np.linspace(np.min(y_i), np.max(y_i), n_grid)
            log_density = kdes[i].score_samples(grid.reshape(-1, 1))
            modes.append(grid[np.argmax(log_density)])

        modes = np.array(modes)
        return modes

    def sample_quantile(
        self, q: Union[float, List[float]]
    ) -> Float[np.ndarray, "q_dim batch y_dim"]:
        """
        Compute the quantiles of the samples for each `x`.
        Estimate: `q-th quantile[Y | X = x]` for each `x`.
        Equivalent to `np.quantile(samples.to_numpy(), q, axis=0)`.

        Parameters
        ----------
        q : float or list[float]
            Quantile or sequence of quantiles to compute.
        """
        quantiles = np.quantile(self._samples, q, axis=0)
        return (
            quantiles
            if isinstance(q, list)
            else quantiles.reshape((1, self.batch, self.y_dim))
        )

    def sample_range(self) -> Float[np.ndarray, "batch 2"]:
        """
        Compute the range of the samples for each `x` using the empirical minimum and
        maximum of the samples, `np.min(samples.to_numpy(), axis=0)` and
        `np.max(samples.to_numpy(), axis=0)`.

        Returns
        -------
        range : np.ndarray
            The range of the samples for each `x`.
        """
        _check_unidimensional(self._samples)
        return np.stack((self._samples.min(axis=0), self._samples.max(axis=0)), axis=-1)

    def sample_std(self, ddof: int = 0) -> Float[np.ndarray, "batch y_dim"]:
        """
        Compute the standard deviation of the samples for each `x`.
        Estimate: `std[Y | X = x]` for each `x`.
        Equivalent to `np.std(samples.to_numpy(), axis=0, ddof=ddof)`.

        Parameters
        ----------
        ddof : int
            Delta Degrees of Freedom. The divisor used in the calculation is `N - ddof`,
            where N represents the number of elements.

        Returns
        -------
        std : np.ndarray
            The standard deviation of the samples for each `x`.
        """
        return self._samples.std(axis=0, ddof=ddof)

    def to_numpy(self) -> Float[np.ndarray, "n_samples batch y_dim"]:
        """
        Return the samples as a numpy array.

        Returns
        -------
        samples : np.ndarray
            The numpy array of the samples.
        """
        return self._samples

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
        return Samples(self._samples.__getitem__(key))
