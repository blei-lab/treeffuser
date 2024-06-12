"""
This should be the main file corresponding to the project.
"""

import abc
import warnings
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import einops
import numpy as np
from jaxtyping import Float
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from treeffuser._preprocessors import Preprocessor
from treeffuser._score_models import ScoreModel
from treeffuser._warnings import ConvergenceWarning
from treeffuser.sde import DiffusionSDE
from treeffuser.sde import sdeint


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
        self._x_preprocessor = None
        self._x_dim = None
        self._x_cat_idx = None
        self._y_preprocessor = None
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

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        cat_idx: Optional[List[int]] = None,
    ):
        """
        Fit the tabular diffusion model to the data.

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
        """
        self.sde = self.get_new_sde()
        self.score_model = self.get_new_score_model()
        self._x_preprocessor = Preprocessor()
        self._y_preprocessor = Preprocessor()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._y_dim = y.shape[1]
        x_transformed = self._x_preprocessor.fit_transform(
            X,
            cat_idx=cat_idx,
        )
        y_transformed = self._y_preprocessor.fit_transform(y)

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
        n_steps: int = 100,
        seed=None,
        verbose: bool = False,
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Sample from the diffusion model.

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
        """
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        x_transformed = self._x_preprocessor.transform(X)
        batch_size_x = x_transformed.shape[0]
        y_dim = self._y_dim

        n_samples_sampled = 0
        y_samples = []
        x_batched = None

        pbar = tqdm(total=n_samples, disable=~verbose)
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
        y_untransformed = self._y_preprocessor.inverse_transform(y_transformed)
        y_untransformed = einops.rearrange(
            y_untransformed,
            "(n_samples batch) y_dim -> n_samples batch y_dim",
            n_samples=n_samples,
        )
        return y_untransformed

    def predict(
        self,
        X: Float[ndarray, "batch x_dim"],
        ode: bool = False,
        tol: float = 1e-3,
        max_samples: int = 100,
        verbose: bool = False,
    ):
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        if ode:
            return self._predict_from_ode(X, tol, verbose)
        else:
            return self._predict_from_sample(X, tol, max_samples, verbose)

    def _predict_from_ode(
        self, X: Float[ndarray, "batch x_dim"], tol: float = 1e-3, verbose: bool = False
    ) -> Float[ndarray, "batch y_dim"]:
        raise NotImplementedError

    def _predict_from_sample(
        self,
        X: Float[ndarray, "batch x_dim"],
        tol: float,
        max_samples: int,
        verbose: bool,
    ) -> Float[ndarray, "batch y_dim"]:
        """
        Predict the conditional mean of y given x via Monte Carlo estimates.

        This method iteratively generates samples of size 10, until the relative change in
        the norm of each estimate is within a specified tolerance.
        """
        n_samples = n_samples_increment = 10

        mean_prev = self.sample(X=X, n_samples=n_samples, verbose=verbose).mean(axis=0)
        norm_prev = np.sqrt((mean_prev**2).sum(axis=1))
        max_change = np.inf

        while max_change > tol and n_samples < max_samples:
            sum_increment = self.sample(
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
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        y_samples = self.sample(X=X, n_samples=n_samples, verbose=verbose)

        n_samples, batch, _ = y_samples.shape

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
        ode: bool = False,
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
        ode : bool, optional
            If True, computes the negative log likelihood from ODE.
            If False, computes it from samples. Default is True.
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
        """
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        if ode:
            return self._compute_nll_from_ode(X, y, verbose)
        else:
            return self._compute_nll_from_sample(X, y, n_samples, bandwidth, verbose)

    def _compute_nll_from_ode(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        verbose: bool = False,
    ):
        raise NotImplementedError

    def _compute_nll_from_sample(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        n_samples: int = 10,
        bandwidth: float | Literal["scott", "silverman"] = 1.0,
        verbose: bool = False,
    ) -> float:
        y_samples = self.sample(X=X, n_samples=n_samples, verbose=verbose)

        def fit_and_evaluate_kde(y_train, y_test):
            kde = KernelDensity(bandwidth=bandwidth, algorithm="auto", kernel="gaussian")
            kde.fit(y_train)
            return kde.score_samples(y_test).item()

        n_samples, batch, y_dim = y_samples.shape

        nll = 0
        for i in range(batch):
            y_train_xi = y_samples[:, i, :]
            y_test_xi = y[i, :]
            nll -= fit_and_evaluate_kde(y_train_xi, [y_test_xi])

        return nll
