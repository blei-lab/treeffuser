"""
This should be the main file corresponding to the project.
"""

import abc
import warnings
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
from einops import rearrange
from jaxtyping import Float
from ml_collections import ConfigDict
from ml_collections import FrozenConfigDict
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

import treeffuser._score_models as _score_models
from treeffuser._preprocessors import Preprocessor
from treeffuser._score_models import Score
from treeffuser._warnings import ConvergenceWarning
from treeffuser.sde import get_sde
from treeffuser.sde import sdeint
from treeffuser.sde.initialize import initialize_sde


def _check_arguments(
    X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"] = None
) -> None:
    """
    Check the arguments for the model.

    Raises an error if the arguments are not valid.
    """
    # TODO: Implement this function
    return


class Treeffuser(BaseEstimator, abc.ABC):
    """
    Abstract class for the Treeffuser model. Every particular
    score function has a slightly different implementation with
    different parameters and methods.
    """

    def __init__(
        self,
        sde_name: str = "vesde",
        sde_initialize_with_data: Optional[bool] = False,
        sde_manual_hyperparams: Optional[dict] = None,
    ):
        self._sde_name = sde_name
        self._sde_initialize_with_data = sde_initialize_with_data
        self._sde_manual_hyperparams = sde_manual_hyperparams
        self._score_model = None
        self._is_fitted = False
        self._x_preprocessor = Preprocessor()
        self._x_cat_idx = None
        self._y_preprocessor = Preprocessor()
        self._y_dim = None

    @property
    @abc.abstractmethod
    def score_config(self) -> FrozenConfigDict:
        """
        Should return the score config for the model.
        These are the parameters that will be used to initialize
        the score model.
        """

    @property
    @abc.abstractmethod
    def _score_model_class(self) -> Score:
        """
        Should return the class of the score model.
        """

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape (batch, x_dim).
        y : np.ndarray
            Target data with shape (batch, y_dim).

        Parameters
        ----------
        An instance of the model for chaining.
        """
        _check_arguments(X, y)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        x_transformed = self._x_preprocessor.fit_transform(
            X,
            cat_idx=self._x_cat_idx,  # _x_cat_idx: temporary fix that leads to non-local behavior
        )
        y_transformed = self._y_preprocessor.fit_transform(y)

        if self._sde_initialize_with_data:
            self._sde = initialize_sde(self._sde_name, y_transformed)
        else:
            sde_cls = get_sde(self._sde_name)
            if self._sde_manual_hyperparams:
                self._sde = sde_cls(**self._sde_manual_hyperparams)
            else:
                self._sde = sde_cls()
        self._score_config.update({"sde": self._sde})

        self._score_config = FrozenConfigDict(self._score_config)

        self._score_model = self._score_model_class(**self.score_config)
        self._score_model.fit(x_transformed, y_transformed)

        self._y_dim = y.shape[1]
        self._is_fitted = True
        return self

    def sample(
        self,
        X: Float[ndarray, "batch x_dim"],
        n_samples: int,
        n_parallel: int = 100,
        n_steps: int = 100,
        seed=None,
        verbose: int = 0,
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Sample from the diffusion model.
        """
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        x_transformed = self._x_preprocessor.transform(X)

        batch_size_x = x_transformed.shape[0]
        y_dim = self._y_dim

        n_samples_sampled = 0
        y_samples = []
        x_batched = None

        pbar = tqdm(total=n_samples, disable=verbose < 1)
        while n_samples_sampled < n_samples:
            batch_size_samples = min(n_parallel, n_samples - n_samples_sampled)
            y_batch = self._sde.sample_from_theoretical_prior(
                (batch_size_samples * batch_size_x, y_dim),
                seed=seed,
            )
            if x_batched is None or x_batched.shape[0] != batch_size_samples:
                # Reuse the same batch of x as much as possible
                x_batched = np.tile(x_transformed, [batch_size_samples, 1])

            def _score_fn(y, t):
                return self._score_model.score(y=y, X=x_batched, t=t)  # noqa: B023
                # B023 highlights that x_batched might change in the future. But we
                # use _score_fn immediately inside the loop, so there are no risks.

            y_batch_samples = sdeint(
                self._sde,
                y_batch,
                self._sde.T,
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
        y_untransformed = rearrange(
            y_untransformed,
            "(n_samples batch) y_dim -> n_samples batch  y_dim",
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
                f"Predict method did not converge on {max_samples} samples. Consider increasing `max_samples` for more accurate estimates.",
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
        bandwidth: float = 1.0,
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


class LightGBMTreeffuser(Treeffuser):
    def __init__(
        self,
        # Diffusion model args
        sde_name: str = "vesde",
        sde_initialize_with_data: bool = False,
        sde_manual_hyperparams: Optional[dict] = None,
        n_repeats: int = 10,
        # Score estimator args
        n_estimators: int = 100,
        eval_percent: Optional[float] = None,
        early_stopping_rounds: Optional[int] = None,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        max_bin: int = 255,
        subsample_for_bin: int = 200000,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        verbose: int = 0,
        seed: Optional[int] = None,
        n_jobs: int = -1,
    ):
        """
        Diffusion model args
        -------------------------------
        sde_name (str): The SDE name.
        sde_initialize_with_data (bool): Whether to initialize the SDE hyperparameters
            with data.
        sde_manual_hyperparams: (dict): A dictionary for explicitly setting the SDE
            hyperparameters, overriding default or data-based initializations.
        n_repeats (int): How many times to repeat the training dataset. i.e how
            many noisy versions of a point to generate for training.
        LightGBM args
        -------------------------------
        eval_percent (float): Percentage of the training data to use for validation.
            If `None`, no validation set is used.
        early_stopping_rounds (int): If `None`, no early stopping is performed. Otherwise,
            the model will stop training if no improvement is observed in the validation
            set for `early_stopping_rounds` consecutive iterations.
        n_estimators (int): Number of boosting iterations.
        num_leaves (int): Maximum tree leaves for base learners.
        max_depth (int): Maximum tree depth for base learners, <=0 means no limit.
        learning_rate (float): Boosting learning rate.
        max_bin (int): Max number of bins that feature values will be bucketed in. This
            is used for lightgbm's histogram binning algorithm.
        subsample_for_bin (int): Number of samples for constructing bins (can ignore).
        min_child_samples (int): Minimum number of data needed in a child (leaf). If
            less than this number, will not create the child.
        subsample (float): Subsample ratio of the training instance.
        subsample_freq (int): Frequence of subsample, <=0 means no enable.
            How often to subsample the training data.
        seed (int): Random seed.
        early_stopping_rounds (int): If `None`, no early stopping is performed. Otherwise,
            the model will stop training if no improvement is observed in the validation
        n_jobs (int): Number of parallel threads. If set to -1, the number is set to the
            number of available cores.
        """
        if sde_initialize_with_data and sde_manual_hyperparams is not None:
            raise Warning(
                "Manual hypeparameter setting will override data-based initialization."
            )

        super().__init__(
            sde_name=sde_name,
            sde_initialize_with_data=sde_initialize_with_data,
            sde_manual_hyperparams=sde_manual_hyperparams,
        )
        self._score_config = ConfigDict(
            {
                "n_repeats": n_repeats,
                "n_estimators": n_estimators,
                "eval_percent": eval_percent,
                "early_stopping_rounds": early_stopping_rounds,
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "max_bin": max_bin,
                "subsample_for_bin": subsample_for_bin,
                "min_child_samples": min_child_samples,
                "subsample": subsample,
                "subsample_freq": subsample_freq,
                "verbose": verbose,
                "seed": seed,
                "n_jobs": n_jobs,
            }
        )

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        cat_idx: Optional[List[int]] = None,
    ):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape (batch, x_dim).
        y : np.ndarray
            Target data with shape (batch, y_dim).
        cat_idx: list
            List with indices of the columns of X that are categorical.

        Parameters
        ----------
        An instance of the model for chaining.
        """
        if cat_idx:
            self._x_cat_idx = cat_idx
            self._score_config.update({"categorical_features": cat_idx})

        super().fit(X, y)

    @property
    def score_config(self):
        return self._score_config

    @property
    def _score_model_class(self):
        return _score_models.LightGBMScore
