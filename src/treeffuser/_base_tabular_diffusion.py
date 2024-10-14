"""
This should be the main file corresponding to the project.
"""

import abc
import warnings
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import einops
import numpy as np
from jaxtyping import Float
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from treeffuser._score_models import ScoreModel
from treeffuser._warnings import CastFloat32Warning
from treeffuser._warnings import ConvergenceWarning
from treeffuser.scaler import ScalerMixedTypes
from treeffuser.sde import DiffusionSDE
from treeffuser.sde import sdeint

###################################################
# Helper functions
###################################################


def _check_array(array: ndarray[float]):
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    # Check shapes
    if array.ndim > 2:
        raise ValueError("Input array cannot have more than three dimensions.")
    elif array.ndim == 1:
        array = array.reshape(-1, 1)

    # Cast floats
    if not np.issubdtype(array.dtype, np.floating):
        try:
            array = np.asarray(array, dtype=np.float32)
            warnings.warn(
                "Input array is not float; it has been recast to float32.",
                CastFloat32Warning,
                stacklevel=2,
            )
        except ValueError as e:
            # raise the ValueError preserving the original exception context, see B904 from flake8-bugbear
            raise ValueError(
                "Input array is not float and cannot be converted to float32."
                "Please check if you have encoded the categorical variables as numerical values."
            ) from e

    return array


###################################################
# Main class
###################################################


class BaseTabularDiffusion(BaseEstimator, abc.ABC):
    """
    Abstract class for the tabular diffusions. Every particular
    score function has a slightly different implementation with
    different parameters and methods.
    """

    def __init__(
        self,
        sde_initialize_from_data: bool = False,
    ):
        self.sde = None
        self.sde_initialize_from_data = sde_initialize_from_data
        self.score_model = None
        self._is_fitted = False
        self._x_scaler = None
        self._x_dim = None
        self._x_cat_idx = None
        self._y_scaler = None
        self._y_dim = None

    @abc.abstractmethod
    def get_new_sde(self) -> DiffusionSDE:
        """
        Return the SDE model.
        """

    @abc.abstractmethod
    def get_new_score_model(self) -> ScoreModel:
        """
        Return the score model.
        """

    def _validate_data(
        self,
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        validate_X: bool = True,
        validate_y: bool = True,
    ) -> Tuple[
        Optional[Float[ndarray, "batch x_dim"]], Optional[Float[ndarray, "batch y_dim"]]
    ]:
        """Reshape X and/or y to adhere to the (batch, n_dim) convention, and cast them as flows."""
        if validate_X and X is not None:
            X = _check_array(X)

        if validate_y and y is not None:
            y = _check_array(y)

        return X, y

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        cat_idx: Optional[List[int]] = None,
    ):
        """
        Fit the conditional diffusion model to the tabular data (X, y).

        Parameters
        ----------
        X : np.ndarray
            Input data with shape (batch, x_dim).
        y : np.ndarray
            Target data with shape (batch, y_dim).
        cat_idx : List[int], optional
            List of indices of categorical features in X. Default is None.

        Returns
        -------
        self : TabularDiffusion
            The fitted model.

        Note
        ----
        The method handles 2D inputs (["batch x_dim"], ["batch y_dim"]) by convention,
        but also works with 1D inputs (["batch"]) for single-dimensional data.
        """
        if cat_idx is not None:
            for idx in cat_idx:
                if idx < 0 or idx >= X.shape[1]:
                    raise ValueError(
                        f"Invalid indices in `cat_idx`: {idx} is not in "
                        f"[0, {X.shape[1]}-1] (the shape of X)."
                    )
        self.sde = self.get_new_sde()
        self.score_model = self.get_new_score_model()
        self._x_scaler = ScalerMixedTypes()
        self._y_scaler = ScalerMixedTypes()

        # store the original number of dimensions of the response
        # before reshaping the data so as to ensure that predictions
        # have the same shape as the user-inputted response
        self._y_original_ndim = y.ndim
        X, y = self._validate_data(X=X, y=y)

        self._y_dim = y.shape[1]
        x_transformed = self._x_scaler.fit_transform(
            X,
            cat_idx=cat_idx,
        )
        y_transformed = self._y_scaler.fit_transform(y)

        if self.sde_initialize_from_data:
            self.sde.initialize_hyperparams_from_data(y_transformed)
        self.score_model.fit(x_transformed, y_transformed, self.sde, cat_idx)

        self._is_fitted = True
        return self

    def sample(
        self,
        X: Float[ndarray, "batch x_dim"],
        n_samples: int,
        n_parallel: int = 10,
        n_steps: int = 50,
        seed=None,
        verbose: bool = False,
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Sample responses from the diffusion model conditional on the given input data `X`.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape (batch, x_dim).
        n_samples : int
            Number of samples to draw for each input.
        n_parallel : int, optional
            Number of parallel samples to draw. Default is 10.
        n_steps : int, optional
            Number of steps to take by the SDE solver. Default is 100.
        seed : int, optional
            Seed for the random number generator of the sampling. Default is None.
        verbose : bool, optional
            Show a progress bar indicating the number of samples drawn. Default is False.

        Returns
        -------
        Float[ndarray, "n_samples batch y_dim"]
            Samples drawn from the diffusion model.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Note
        ----
        The method handles 2D inputs (["batch x_dim"], ["batch y_dim"]) by convention,
        but also works with 1D inputs (["batch"]) for single-dimensional data.
        """
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        X, _ = self._validate_data(X=X, validate_y=False)

        y_samples = self._sample_without_validation(
            X, n_samples, n_parallel, n_steps, seed, verbose
        )

        # Ensure output aligns with original shape provided by user
        if self._y_original_ndim == 1:
            y_samples = y_samples.squeeze(axis=-1)

        return y_samples

    def _sample_without_validation(
        self,
        X: Float[ndarray, "batch x_dim"],
        n_samples: int,
        n_parallel: int = 10,
        n_steps: int = 100,
        seed=None,
        verbose: bool = False,
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Sampling method that preserves shape conventions.
        """
        x_transformed = self._x_scaler.transform(X)
        batch_size_x = x_transformed.shape[0]
        y_dim = self._y_dim

        n_samples_sampled = 0
        y_samples = []
        x_batched = None

        pbar = tqdm(total=n_samples, disable=not verbose)
        while n_samples_sampled < n_samples:
            batch_size_samples = min(n_parallel, n_samples - n_samples_sampled)
            y_batch = self.sde.sample_from_theoretical_prior(
                (batch_size_samples * batch_size_x, y_dim),
                seed=seed,
            )
            if x_batched is None or x_batched.shape[0] != batch_size_samples:
                # Reuse the same batch of x as much as possible
                x_batched = np.tile(x_transformed, [batch_size_samples, 1])

            def _score_fn(y, t):
                return self.score_model.score(y=y, X=x_batched, t=t)  # noqa: B023
                # B023 highlights that x_batched might change in the future. But we
                # use _score_fn immediately inside the loop, so there are no risks.

            y_batch_samples = sdeint(
                self.sde,
                y_batch,
                self.sde.T,
                0,
                n_steps=n_steps,
                method="euler",
                seed=seed + n_samples_sampled if seed is not None else None,
                score_fn=_score_fn,
            )
            n_samples_sampled += batch_size_samples
            y_samples.append(y_batch_samples)
            pbar.update(batch_size_samples)
        pbar.close()

        y_transformed = np.concatenate(y_samples, axis=0)
        y_untransformed = self._y_scaler.inverse_transform(y_transformed)

        y_untransformed = einops.rearrange(
            y_untransformed,
            "(n_samples batch) y_dim -> n_samples batch y_dim",
            n_samples=n_samples,
        )
        return y_untransformed

    def predict(
        self,
        X: Float[ndarray, "batch x_dim"],
        tol: float = 1e-3,
        max_samples: int = 100,
        verbose: bool = False,
    ):
        """
        Predict the conditional mean of the response given the input data X using Monte Carlo estimates.

        The method iteratively samples from the model until the change in the norm of the mean estimate is within a specified
        tolerance, or until a maximum number of samples is reached.

        Parameters
        ----------
        X : Float[ndarray, "batch x_dim"]
            Input data with shape (batch, x_dim).
        tol : float, optional
            Tolerance for the stopping criterion based on the relative change in the mean estimate. Default is 1e-3.
        max_samples : int, optional
            Maximum number of samples to draw in the Monte Carlo simulation to ensure convergence. Default is 100.
        verbose : bool, optional
            If True, displays a progress bar indicating the sampling progress. Default is False.

        Returns
        -------
        Float[ndarray, "batch y_dim"]
            The predicted conditional mean of the response for each input in X, shaped according to the original dimensionality
            of the target data provided during training.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Note
        ----
        The method handles 2D inputs (["batch x_dim"], ["batch y_dim"]) by convention,
        but also works with 1D inputs (["batch"]) for single-dimensional data.
        """
        X, _ = self._validate_data(X=X, validate_y=False)

        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        y_preds = self._predict_from_sample(X, tol, max_samples, verbose)

        if self._y_original_ndim == 1:
            y_preds = y_preds.squeeze(axis=-1)

        return y_preds

    def _predict_from_sample(
        self,
        X: Float[ndarray, "batch x_dim"],
        tol: float,
        max_samples: int,
        verbose: bool,
    ) -> Float[ndarray, "batch y_dim"]:
        """
        Predict the conditional mean of the response given the input data `X` using Monte Carlo estimates.

        This method iteratively generates samples of size 10, until the relative change in
        the norm of each estimate is within the specified tolerance `tol`.
        """
        n_samples = n_samples_increment = 10

        mean_prev = self._sample_without_validation(
            X=X, n_samples=n_samples, verbose=verbose
        ).mean(axis=0)
        norm_prev = np.sqrt((mean_prev**2).sum(axis=1))
        max_change = np.inf

        while max_change > tol and n_samples < max_samples:
            sum_increment = self._sample_without_validation(
                X=X,
                n_samples=n_samples_increment,
                verbose=verbose,
            ).sum(axis=0)

            mean = (sum_increment + mean_prev * n_samples) / (n_samples + n_samples_increment)
            norm = np.sqrt((mean**2).sum(axis=1))
            n_samples += n_samples_increment

            max_change = np.max(np.abs(norm / norm_prev - 1))
            mean_prev = mean

        if n_samples > max_samples:
            warnings.warn(
                f"Predict method did not converge on {max_samples} samples. Consider increasing "
                f"`max_samples` for more accurate estimates.",
                ConvergenceWarning,
                stacklevel=2,
            )

        return mean_prev

    def predict_distribution(
        self,
        X: Float[ndarray, "batch x_dim"],
        n_samples: int = 100,
        bandwidth: Union[float, Literal["scott", "silverman"]] = 1.0,
        verbose: bool = False,
    ) -> List[KernelDensity]:
        """
        Estimate the distribution of the predicted responses for the given input data `X` using Gaussian
        KDEs from `sklearn.neighbors.KernelDensity`.

        Parameters
        ----------
        X : Float[ndarray, "batch x_dim"]
            Input data with shape (batch, x_dim).
        n_samples : int, optional
            Number of samples to draw for each input. Default is 100.
        bandwidth : Union[float, Literal["scott", "silverman"]], optional
            The bandwidth of the kernel for the Kernel Density Estimation. If a float, it defines the bandwidth of the kernel. If a string, one of the "scott" or "silverman" estimation methods. Default is 1.0.
        verbose : bool, optional
            If True, displays a progress bar indicating the number of samples drawn. Default is False.

        Returns
        -------
        List[KernelDensity]
            A list of KernelDensity objects representing the estimated distributions for each input in X.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Note
        ----
        The method handles 2D inputs (`["batch x_dim"]`, `["batch y_dim"]`) by convention, but also works with 1D inputs (`["batch"]`) for single-dimensional data.
        """
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        X, _ = self._validate_data(X=X, validate_y=False)

        y_samples = self._sample_without_validation(X=X, n_samples=n_samples, verbose=verbose)

        batch = y_samples.shape[1]

        kdes = []
        for i in range(batch):
            y_i = y_samples[:, i, :]
            kde = KernelDensity(bandwidth=bandwidth, algorithm="auto", kernel="gaussian")
            kde.fit(y_i)
            kdes.append(kde)

        return kdes

    def compute_nll(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        n_samples: int = 10,
        bandwidth: Union[float, Literal["scott", "silverman"]] = 1.0,
        verbose: bool = False,
    ) -> float:
        """
        Compute the negative log likelihood, \\sum_{(y, x) in [y, X]} \\log p(y|x), where p
        is the conditional distribution learned by the model.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape (batch, x_dim).
        y : np.ndarray
            Target data with shape (batch, y_dim).
        n_samples : int, optional
            Number of samples to draw if computing the negative log likelihood from samples. Default is 10.
        bandwidth : Union[float, Literal["scott", "silverman"]], optional
            The bandwidth of the kernel. If bandwidth is a float, it defines the bandwidth of the kernel.
            If bandwidth is a string, one of the  "scott" and "silverman" estimation methods. Default is 1.0.
        verbose : bool, optional
            If True, displays a progress bar for the sampling. Default is False.

        Returns
        -------
        float
            The computed negative log likelihood value.

        Note
        ----
        The method handles 2D inputs (`["batch x_dim"]`, `["batch y_dim"]`) by convention, but also works with 1D inputs (`["batch"]`) for single-dimensional data.
        """
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        X, y = self._validate_data(X=X, y=y)

        return self._compute_nll_from_sample(X, y, n_samples, bandwidth, verbose)

    def _compute_nll_from_sample(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        n_samples: int = 10,
        bandwidth: Union[float, Literal["scott", "silverman"]] = 1.0,
        verbose: bool = False,
    ) -> float:
        y_samples = self._sample_without_validation(X=X, n_samples=n_samples, verbose=verbose)

        def fit_and_evaluate_kde(y_train, y_test):
            kde = KernelDensity(bandwidth=bandwidth, algorithm="auto", kernel="gaussian")
            kde.fit(y_train)
            return kde.score_samples(y_test).item()

        n_samples, batch, _ = y_samples.shape

        nll = 0
        for i in range(batch):
            y_train_xi = y_samples[:, i, :]
            y_test_xi = y[i, :]
            nll -= fit_and_evaluate_kde(y_train_xi, [y_test_xi])

        return nll
