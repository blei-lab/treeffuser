"""
This file should contain a general abstraction of the score models and
should function as a wrapper for different models we might want to use.

The idea is to "hide" the particular tree we want to use so that
we can easily switch between different models without having to change
the rest of the code.
"""

import abc
from typing import Optional

import lightgbm as lgb
import numpy as np
from jaxtyping import Float, Int
from numpy import ndarray
from sklearn.model_selection import train_test_split

from treeffuser.sde import SDE


###################################################
# Helper functions
###################################################
def _score_normal_distribution(
    y: Float[np.ndarray, "batch y_dim"],
    mean: Float[np.ndarray, "batch y_dim"],
    std: Float[np.ndarray, "batch y_dim"],
) -> Float[np.ndarray, "batch y_dim"]:
    """Compute the score of a normal distribution."""
    # TODO: We might have issues if the std is too small
    # might need to consider implementing a custom loss
    # for lightgbm
    return (y - mean) / (std**2)


def _lgbm_loss(y_true, y_pred, weights):
    """
    Compute the score matching loss for lightgbm.
    """
    grad = y_true - y_pred
    hess = np.ones_like(grad)
    return grad, hess


def _fit_one_lgbm_model(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    X_val: Float[np.ndarray, "batch x_dim"],
    y_val: Float[np.ndarray, "batch y_dim"],
    weights: Float[np.ndarray, "batch"],
    n_estimators: int,
    num_leaves: int,
    max_depth: int,
    learning_rate: float,
    max_bin: int,
    subsample_for_bin: int,
    min_child_samples: int,
    subsample: float,
    subsample_freq: int,
    seed: int,
    verbose: int,
    early_stopping_rounds: int,
    n_jobs: int = -1,
) -> lgb.Booster:
    """
    Simple wrapper for fitting a lightgbm model. See
    the lightgbm score function documentation for more details.
    """
    callbacks = None
    if early_stopping_rounds is not None:
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=verbose)]

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_bin=max_bin,
        subsample_for_bin=subsample_for_bin,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=subsample_freq,
        random_state=seed,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    eval_set = None if X_val is None else (X_val, y_val)
    model.fit(X=X, y=y, sample_weight=weights, eval_set=eval_set, callbacks=callbacks)

    return model


###################################################
# Main models
###################################################


class Score(abc.ABC):
    @abc.abstractmethod
    def score(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        t: Int[np.ndarray, "batch"],
    ):

        pass

    @abc.abstractmethod
    def fit(self, X: Float[np.ndarray, "batch x_dim"], y: Float[np.ndarray, "batch y_dim"]):
        pass


# lightgbm score
class LightGBMScore(Score):
    def __init__(
        self,
        sde: SDE,
        likelihood_reweighting: Optional[bool] = False,
        n_repeats: Optional[int] = 1,
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
    ) -> None:
        """
        Args:
        This model doesn't do any model checking or validation. It's assumed that
        that the main user is the `Treeffuser` class and that the user has already
        checked that the inputs are valid.

            Diffusion model args
            -------------------------------
            sde (SDE): A member from the SDE class specifying the sde that is implied
                by the score model.
            likelihood_reweighting (bool): Whether to reweight the likelihoods.
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

        # Diffusion model args
        self._sde = sde
        self._likelihood_reweighting = likelihood_reweighting
        self._n_repeats = n_repeats
        self._eval_percent = eval_percent

        # LightGBM args
        self._lgbm_args = {
            "early_stopping_rounds": early_stopping_rounds,
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "max_bin": max_bin,
            "subsample_for_bin": subsample_for_bin,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "subsample_freq": subsample_freq,
            "seed": seed,
            "verbose": verbose,
            "n_jobs": n_jobs,
        }

        # Other stuff part of internal state
        self.models = None  # Convention inputs are (y, x, t)
        self.is_fitted = False

    def score(
        self,
        y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Int[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        scores = []
        for i in range(y.shape[1]):
            predictors = np.concatenate([y, X, t.reshape(-1, 1)], axis=1)
            scores.append(self.models[i].predict(predictors))
        return np.array(scores).T

    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
    ):
        """
        Fit the score model to the data.

        Args:
            X: input data
            y: target data
            n_repeats: How many times to repeat the training dataset.
            likelihood_reweighting: Whether to reweight the likelihoods.
            likelihood_weighting: If `True`, weight the mixture of score
                matching losses according to https://arxiv.org/abs/2101.09258;
                otherwise use the weighting recommended in song's SDEs paper.
        """
        EPS = 1e-6  # smallest step we can sample from
        T = self._sde.T
        y_dim = y.shape[1]

        X_train, X_test, y_train, y_test = X, None, y, None
        if self._eval_percent is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self._eval_percent
            )

        X_train: Float[ndarray, "batch x_dim"] = np.tile(X, (self._n_repeats, 1))
        y_train: Float[ndarray, "batch y_dim"] = np.tile(y, (self._n_repeats, 1))
        t_train: Float[ndarray, "batch"] = (
            np.random.uniform(0, 1, size=(y_train.shape[0], 1)) * (T - EPS) + EPS
        )
        z_train: Float[ndarray, "batch y_dim"] = np.random.normal(
            size=(y_train.shape[0], y_train.shape[1])
        )

        if self._eval_percent is not None:
            t_val = np.random.uniform(0, 1, size=(y_test.shape[0])) * (T - EPS) + EPS
            z_val = np.random.normal(size=(y_test.shape[0], y_test.shape[1]))

        train_mean, train_std = self._sde.marginal_prob(y_train, t_train)
        if self._eval_percent is not None:
            val_mean, val_std = self._sde.marginal_prob(y_test, t_val)

        perturbed_y_train = train_mean + train_std * z_train
        if self._eval_percent is not None:
            perturbed_y_val = val_mean + val_std * z_val

        predictors_train = np.concatenate(
            [perturbed_y_train, X_train, t_train.reshape(-1, 1)], axis=1
        )
        predictors_val = None
        if self._eval_percent is not None:
            predictors_val = np.concatenate(
                [perturbed_y_val, X_test, t_val.reshape(-1, 1)], axis=1
            )

        score_normal_train = _score_normal_distribution(
            perturbed_y_train, train_mean, train_std
        )
        score_normal_val = None
        if self._eval_percent is not None:
            score_normal_val = _score_normal_distribution(perturbed_y_val, val_mean, val_std)

        weights = np.ones(score_normal_train.shape[0])
        if self._likelihood_reweighting:
            norm = np.sum(score_normal_train**2, axis=1)
            weights = 1 / (norm + 1e-6)

        models = []
        for i in range(y_dim):
            score_normal_val_i = None
            if self._eval_percent is not None:
                score_normal_val_i = score_normal_val[:, i]
            score_model_i = _fit_one_lgbm_model(
                X=predictors_train,
                y=score_normal_train[:, i],
                X_val=predictors_val,
                y_val=score_normal_val_i,
                weights=weights,
                **self._lgbm_args,
            )
            models.append(score_model_i)
        self.models = models
