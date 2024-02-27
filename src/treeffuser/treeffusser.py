"""
This should be the main file corresponding to the project.
"""

import abc
from typing import Optional

from einops import rearrange
from jaxtyping import Float
from ml_collections import FrozenConfigDict
from numpy import ndarray
from sklearn.base import BaseEstimator

import treeffuser._sampling as _sampling
import treeffuser._score_models as _score_models
import treeffuser._sdes as _sdes
from treeffuser._preprocessors import Preprocessor
from treeffuser._score_models import Score


def _check_arguments(
    X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"] = None
) -> None:
    """
    Check the arguments for the model.

    Raises an error if the arguments are not valid.
    """
    # TODO: Implement this function
    return


class Treeffusser(BaseEstimator, abc.ABC):
    """
    Abstract class for the Treeffuser model. Every particular
    score function has a slightly different implementation with
    different parameters and methods.
    """

    def __init__(self, sde_name: str = "vesde"):
        self._score_model = None
        self._is_fitted = False
        self._x_preprocessor = Preprocessor()
        self._y_preprocessor = Preprocessor()
        self._y_dim = None

        # TODO: We are using the defaults but we should change this
        self._sde = _sdes.get_sde(sde_name)()
        self._sde_name = sde_name

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
        denoise: bool = False,
        n_steps: int = 100,
        seed=None,
    ) -> Float[ndarray, "batch n_samples y_dim"]:
        """
        Sample from the model.
        """
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        # We need a new SDE in case we change discretization steps
        sde = _sdes.get_sde(self._sde_name)(N=n_steps)

        x_transformed = self._x_preprocessor.transform(X)

        y_untransformed = _sampling.sample(
            X=x_transformed,
            y_dim=self._y_dim,
            n_samples=n_samples,
            score_fn=self._score_model.score,
            n_parallel=n_parallel,
            sde=sde,
            n_steps=n_steps,
            denoise=denoise,
            seed=seed,
            predictor_name="euler_maruyama",  # TODO: This should be a parameter
            corrector_name="none",  # TODO: This should be a parameter
            verbose=1,
        )

        y_untransformed = rearrange(
            y_untransformed, "n_preds n_samples y_dim -> (n_preds n_samples) y_dim"
        )
        y_transformed = self._y_preprocessor.inverse_transform(y_untransformed)
        y_transformed = rearrange(
            y_transformed,
            "(n_preds n_samples) y_dim -> n_preds n_samples y_dim",
            n_samples=n_samples,
        )
        return y_transformed


class LightGBMTreeffusser(Treeffusser):
    def __init__(
        self,
        # Diffusion model args
        sde_name: Optional[str] = "vesde",
        # Score estimator args
        n_repeats: Optional[int] = 10,
        n_estimators: Optional[int] = 100,
        eval_percent: Optional[float] = None,
        early_stopping_rounds: Optional[int] = None,
        num_leaves: Optional[int] = 31,
        max_depth: Optional[int] = -1,
        learning_rate: Optional[float] = 0.1,
        max_bin: Optional[int] = 255,
        subsample_for_bin: Optional[int] = 200000,
        min_child_samples: Optional[int] = 20,
        subsample: Optional[float] = 1.0,
        subsample_freq: Optional[int] = 0,
        verbose: Optional[int] = 0,
        seed: Optional[int] = 0,
        n_jobs: Optional[int] = -1,
    ):
        """
        Diffusion model args
        -------------------------------
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
        super().__init__(sde_name=sde_name)
        self._score_config = FrozenConfigDict(
            {
                "sde": self._sde,
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

    @property
    def score_config(self):
        return self._score_config

    @property
    def _score_model_class(self):
        return _score_models.LightGBMScore
