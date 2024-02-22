"""
This should be the main file corresponding to the project.
"""

import abc

import numpy as np
from jaxtyping import Float
from ml_collections import FrozenConfigDict
from numpy import ndarray
from sklearn.base import BaseEstimator

from treeffuser._preprocessors import Preprocessor
from treeffuser.score_models import ScoreModel


def _check_arguments(
    X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]
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

    def __init__(self):
        self._score_model = None
        self._is_fitted = False
        self._x_preprocessor = Preprocessor()
        self._y_preprocessor = Preprocessor()

    @abc.abstractmethod
    @property
    def _score_config(self) -> FrozenConfigDict:
        """
        Should return the score config for the model.
        These are the parameters that will be used to initialize
        the score model.
        """

    @abc.abstractmethod
    @property
    def _score_model_class(self) -> ScoreModel:
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

        self._score_model = self.score_model_class(**self.score_config)
        self._score_model.fit(x_transformed, y_transformed)

        self._is_fitted = True
        return self

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples: int
    ) -> Float[ndarray, "batch n_samples y_dim"]:
        """
        Sample from the model.
        """
        if not self._is_fitted:
            raise ValueError("The model has not been fitted yet.")

        return np.zeros((X.shape[0], n_samples, X.shape[1]))


# STILL NEEDS TO BE IMPLEMENTED
#
#
# class Treeffuser(BaseEstimator):
#
#
#    def __init__(self,
#        likelihood_reweighting: Optional[bool] = False,
#        n_repeats: Optional[int] = 1,
#        n_estimators: Optional[int] = 100,
#        eval_percent: Optional[float] = None,
#        early_stopping_rounds: Optional[int] = None,
#        num_leaves: Optional[int] = 31,
#        max_depth: Optional[int] = -1,
#        learning_rate: Optional[float] = 0.1,
#        max_bin: Optional[int] = 255,
#        subsample_for_bin: Optional[int] = 200000,
#        min_child_samples: Optional[int] = 20,
#        subsample: Optional[float] = 1.0,
#        subsample_freq: Optional[int] = 0,
#        verbose: Optional[int] = 0,
#        seed: Optional[int] = 0,
#        n_jobs: Optional[int] = -1,
#    ):
#        """
#        Args:
#        This model doesn't do any model checking or validation. It's assumed that
#        that the main user is the `Treeffuser` class and that the user has already
#        checked that the inputs are valid.
#
#        Diffusion model args
#        -------------------------------
#        likelihood_reweighting (bool): Whether to reweight the likelihoods.
#        n_repeats (int): How many times to repeat the training dataset. i.e how
#            many noisy versions of a point to generate for training.
#
#        LightGBM args
#        -------------------------------
#        eval_percent (float): Percentage of the training data to use for validation.
#            If `None`, no validation set is used.
#        early_stopping_rounds (int): If `None`, no early stopping is performed. Otherwise,
#            the model will stop training if no improvement is observed in the validation
#            set for `early_stopping_rounds` consecutive iterations.
#        n_estimators (int): Number of boosting iterations.
#        num_leaves (int): Maximum tree leaves for base learners.
#        max_depth (int): Maximum tree depth for base learners, <=0 means no limit.
#        learning_rate (float): Boosting learning rate.
#        max_bin (int): Max number of bins that feature values will be bucketed in. This
#            is used for lightgbm's histogram binning algorithm.
#        subsample_for_bin (int): Number of samples for constructing bins (can ignore).
#        min_child_samples (int): Minimum number of data needed in a child (leaf). If
#            less than this number, will not create the child.
#        subsample (float): Subsample ratio of the training instance.
#        subsample_freq (int): Frequence of subsample, <=0 means no enable.
#            How often to subsample the training data.
#        seed (int): Random seed.
#        early_stopping_rounds (int): If `None`, no early stopping is performed. Otherwise,
#            the model will stop training if no improvement is observed in the validation
#        n_jobs (int): Number of parallel threads. If set to -1, the number is set to the
#            number of available cores.
#        """
#        # Stuff related to the diffusion model
#        self.score_config =
#
#
#    def fit(self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]):
#        """
#        Fit the model to the data.
#
#        Returns an instance of the model for chaining.
#        """
#        _check_arguments(X, y)
#        pass
#
#
#    def predict(self, X):
#        pass
#
#    def sample(self, X):
#        pass
#
#    def likelihood(self, X, y):
#        """
#        Something that computes the log-likelihood of the model.
#        """
#        pass
#
#    def pred_distribution(self, X):
#        """
#        Maybe the CDF?
#        """
#        pass
