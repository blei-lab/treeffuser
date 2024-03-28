"""
This should be the main file corresponding to the project.
"""

import abc
import warnings
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
from einops import rearrange
from jaxtyping import Float
from ml_collections import ConfigDict
from ml_collections import FrozenConfigDict
from numpy import ndarray
from scipy.integrate import odeint
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

import treeffuser._score_models as _score_models
from treeffuser._preprocessors import Preprocessor
from treeffuser._score_models import Score
from treeffuser._tree import _compute_score_divergence_numerical
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

    def fit(self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]):
        """
        Fit the model to the data.

        Returns an instance of the model for chaining.
        """
        _check_arguments(X, y)

        if self._sde_initialize_with_data:
            self._sde = initialize_sde(self._sde_name, y)
        else:
            sde_cls = get_sde(self._sde_name)
            if self._sde_manual_hyperparams:
                self._sde = sde_cls(**self._sde_manual_hyperparams)
            else:
                self._sde = sde_cls()

        self._score_config.update({"sde": self._sde})
        self._score_config = FrozenConfigDict(self._score_config)

        x_transformed = self._x_preprocessor.fit_transform(X)
        y_transformed = self._y_preprocessor.fit_transform(y)

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
        verbose: int = 1,
    ) -> Float[ndarray, "batch n_samples y_dim"]:
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
                (batch_size_samples * batch_size_x, y_dim)
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
            "(n_samples batch) y_dim ->  batch n_samples y_dim",
            n_samples=n_samples,
        )
        return y_untransformed

    def predict(
        self,
        X: Float[ndarray, "batch x_dim"],
        ode: bool = True,
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

        mean_prev = self.sample(X=X, n_samples=n_samples, verbose=verbose).mean(axis=1)
        norm_prev = np.sqrt((mean_prev**2).sum(axis=1))
        max_change = np.inf

        while max_change > tol and n_samples < max_samples:
            sum_increment = self.sample(
                X=X,
                n_samples=n_samples_increment,
                verbose=verbose,
            ).sum(axis=1)

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

    def compute_nll(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        ode: bool = True,
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
            return self._compute_nll_from_ode(X, y, verbose=verbose)
        else:
            return self._compute_nll_from_sample(X, y, n_samples, bandwidth, verbose)

    def _compute_nll_from_ode(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        n_steps: int = 10,
        verbose: bool = False,
    ):
        raise NotImplementedError

    def _compute_nll_from_sample(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        n_samples: Optional[int] = 10,
        bandwidth: Optional[float] = 1.0,
        verbose: bool = False,
    ) -> float:
        y_sample = self.sample(X=X, n_samples=n_samples, verbose=verbose)
        batch_size = y.shape[0]

        def fit_and_evaluate_kde(y_train, y_test):
            kde = KernelDensity(bandwidth=bandwidth, algorithm="auto", kernel="gaussian")
            kde.fit(y_train)
            return kde.score_samples(y_test).item()

        nll = 0
        # for i, y_i in enumerate(y_sample):
        for i in range(batch_size):
            nll -= fit_and_evaluate_kde(
                y_sample[:, i, :].reshape(-1, 1), y[i, :].reshape(-1, 1)
            )

        return nll

    # def probability_flow(
    #     self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    # ) -> Float[ndarray, "batch y_dim"]:
    #     drift, diffusion = self.drift_and_diffusion(y, t)
    #     return drift - diffusion**2 / 2


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
        n_jobs: Optional[int] = -1,
        linear_tree: bool = False,
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
        linear_tree (bool): Fit piecewise linear gradient boosting tree.
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

        # BaseEstimator requirement: every keyword argument should correspond to an attribute on the instance
        self.sde_name = sde_name
        self.n_repeats = n_repeats
        self.n_estimators = n_estimators
        self.eval_percent = eval_percent
        self.early_stopping_rounds = early_stopping_rounds
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_bin = max_bin
        self.subsample_for_bin = subsample_for_bin
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.verbose = verbose
        self.seed = seed
        self.n_jobs = n_jobs
        self.linear_tree = linear_tree

        self._score_config = FrozenConfigDict(
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
                "linear_tree": linear_tree,
            }
        )

    @property
    def score_config(self):
        return self._score_config

    @property
    def _score_model_class(self):
        return _score_models.LightGBMScore

    # def _compute_nll_from_ode(
    #     self,
    #     X: Float[ndarray, "batch x_dim"],
    #     y: Float[ndarray, "batch y_dim"],
    #     verbose: bool = False,
    # ):
    #     if self.linear_tree is False:
    #         raise ValueError(
    #             "Cannot compute ode-based negative log likelihood when `linear_tree` is set to False."
    #         )

    #     forest = self._dump_model()

    #     nll = 0
    #     for i in range(y.shape[0]):
    #         nll += self._sde.get_likelihood_theoretical_prior(y[i].reshape(-1))
    #         nll += _integrate_divergence_over_time(
    #             forest[1:],
    #             self.learning_rate,
    #             y[i].reshape(-1),
    #             X[i, :].reshape(-1),
    #             self._sde.T,
    #         )
    #     return nll

    def _compute_nll_from_ode(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        n_steps: int = 100,
        verbose: bool = False,
    ):
        """
        Compute the log likelihood using the instantaneous change of variable formula.

        It first solves the forward probability flow ODE and uses its solution to solve the ODE resulting from the instantaneous change of variable formula.

        It assumes that the drift is linear in y and the diffusion coefficient doesn't depend on y.

        Write down formula.
        ------
        References:
            Song, Y., et al. Score-based generative modeling through stochastic differential equations. ICLR (2021).

        ------
        TO DO:
        - create a divergence score function:
          - get dict representation of score
          - for each leaf that has y as a feature:
            - set the intercept equal to the sum of all the coefficients that refer to y
            - set all the coefficients to 0, or set leaf_features = [0] and leaf_coeff = [0]
          - initialize a LightGBM model from the modified dict
        - update documentation to add formula of divergence. should this go to the public method?
        """
        if self.linear_tree is False:
            raise ValueError(
                "Cannot compute ode-based negative log likelihood when `linear_tree` is set to False."
            )

        y_dim = y.shape[1]
        x_dim = X.shape[1]

        score_dict = self._dump_model()

        nll = 0
        ite = -1
        for y0, x in zip(y, X):  # tuttapposto
            ite += 1
            print(f"{'#' * 20}")
            print(f"{ite} of {len(y)}")
            print(f"y0={y0}")
            print(f"x={x}")

            # first, solve the probability flow
            def _score_fn(y, x, t):
                score = self._score_model.score(
                    y=y.reshape(1, y_dim), X=x.reshape(1, x_dim), t=np.array(t).reshape(-1, 1)
                )
                return score.reshape(-1)

            def _score_fn_given_x(y, t):
                score = self._score_model.score(
                    y=y.reshape(1, y_dim), X=x.reshape(1, x_dim), t=np.array(t).reshape(-1, 1)
                )
                return score.reshape(-1)

            def _probability_flow(y, t):
                drift, diffusion = self._sde.drift_and_diffusion(y, t)
                return drift - 0.5 * diffusion**2 * _score_fn_given_x(y, t)

            ts = np.arange(0.01, self._sde.T, 1 / n_steps)
            y_ts = odeint(func=_probability_flow, y0=y0, t=ts)

            # next, solve the instantaneous change of variable formula
            drift_div_ts = np.array(
                [
                    self._sde._get_drift_and_diffusion_divergence(y, t)[0]
                    for y, t in zip(y_ts, ts)
                ]  # y_ts[0] is y(T)
            )
            diffusion_coeff_ts = np.array(
                [self._sde.drift_and_diffusion(y, t)[1] for y, t in zip(y_ts, ts)]
            )
            score_div_ts = np.array(
                [
                    # _compute_score_divergence(score_dict, self.learning_rate, y, x, t)
                    _compute_score_divergence_numerical(_score_fn, y, x, t)
                    for y, t in zip(y_ts, ts)
                ]
            )
            log_p_T = self._sde.get_likelihood_theoretical_prior(y_ts[-1])

            nll -= (
                log_p_T
                + (drift_div_ts - 0.5 * diffusion_coeff_ts**2 * score_div_ts).sum() / n_steps
            )

            temp = (
                log_p_T
                + (drift_div_ts - 0.5 * diffusion_coeff_ts**2 * score_div_ts).sum() / n_steps
            )
            print(f"log_p_0_ode={temp}")
            print(
                f"log_p_0_sample: {self.compute_nll(x.reshape(1, x_dim), y0.reshape(1, y_dim), ode=False)}"
            )
            # ts = np.sort(np.arange(self._sde.T, 0, -1 / n_steps))
            # y_ts = odeint(_probability_flow, y0, t=ts)
            #
            # def _score_divergence(
            #     y: Float[np.ndarray, "y_dim"], x: Float[np.ndarray, "x_dim"], t: float
            # ):
            #     return _compute_score_divergence(score_dict, self.learning_rate, y, x, t)
            #
            # def _get_log_likelihood_derivative(y, t):
            #     t_idx = np.where(np.arange(0, 1, 1 / n_steps) == t)
            #     if t_idx:
            #         _, diff_coeff = self._sde.drift_and_diffusion(y_ts[t_idx], t)
            #         drift_div, diff_div = self._sde._get_drift_and_diffusion_divergence(
            #             y_ts[t_idx], t
            #         )  # they depend on y
            #         score = (
            #             _score_fn(y_ts[t_idx], t) if diff_div != 0 else 0
            #         )  # don't compute score if diff_div = 0
            #         return (
            #             drift_div
            #             - 0.5 * diff_coeff**2 * _score_divergence(y_ts[t_idx], x, t)
            #             - diff_coeff * diff_div * score
            #         )
            #     else:
            #         return ValueError
            #
            # log_p_T = self._sde.get_likelihood_theoretical_prior(y_ts[-1])
            # nll -= odeint(  # do this manually
            #     _get_log_likelihood_derivative,
            #     log_p_T,
            #     t=[1, 0],
            #     h0=1 / n_steps,
            #     hmin=1 / n_steps,
            #     hmax=1 / n_steps,
            # )
        return nll

    def _dump_model(self):
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        return self._score_model.models[0]._Booster.dump_model()["tree_info"]  # multiple dims?
