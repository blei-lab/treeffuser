"""
Contains different score models to be used to approximate the score of a given SDE.
"""

import abc
from typing import List
from typing import Optional

import lightgbm as lgb
import numpy as np
from jaxtyping import Float
from jaxtyping import Int
from sklearn.model_selection import train_test_split

from treeffuser.sde import DiffusionSDE

###################################################
# Helper functions
###################################################


def _fit_one_lgbm_model(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    X_val: Float[np.ndarray, "batch x_dim"],
    y_val: Float[np.ndarray, "batch y_dim"],
    seed: int,
    verbose: int,
    cat_idx: Optional[List[int]] = None,
    n_jobs: int = -1,
    early_stopping_rounds: Optional[int] = None,
    **lgbm_args,
) -> lgb.LGBMRegressor:
    """
    Simple wrapper for fitting a lightgbm model. See
    the lightgbm score function documentation for more details.
    """
    callbacks = None
    if early_stopping_rounds is not None:
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=verbose > 0)]

    model = lgb.LGBMRegressor(
        random_state=seed,
        verbose=verbose,
        n_jobs=n_jobs,
        linear_tree=False,
        **lgbm_args,
    )
    eval_set = None if X_val is None else (X_val, y_val)
    if cat_idx is None:
        cat_idx = "auto"
    model.fit(
        X=X,
        y=y,
        eval_set=eval_set,
        callbacks=callbacks,
        categorical_feature=cat_idx,
    )
    return model


def _make_training_data(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    sde: DiffusionSDE,
    n_repeats: int,
    eval_percent: Optional[float],
    cat_idx: Optional[List[int]] = None,
    seed: Optional[int] = None,
):
    """
    Creates the training data for the score model. This functions assumes that
    1.  Score is parametrized as score(x, y, t) = GBT(x, y, t) / std(t)
    2.  The loss that we want to use is
        || std(t) * score(y_perturbed, x, t) - (mean(y, t) - y_perturbed)/std(t) ||^2
        Which corresponds to the standard denoising objective with weights std(t)**2
        This ends up meaning that we optimize
        || GBT(y_perturbed, x, t) - (-z)||^2
        where z is the noise added to y_perturbed.

    Returns:
    - predictors_train: X_train=[x_train, y_perturbed_train, t_train] for lgbm
    - predictors_val: X_val=[x_val, y_perturbed_val, t_val] for lgbm
    - predicted_train: y_train=[-z_train] for lgbm
    - predicted_val: y_val=[-z_val] for lgbm
    """
    EPS = 1e-5  # smallest step we can sample from
    T = sde.T
    if seed is not None:
        np.random.seed(seed)

    X_train, X_test, y_train, y_test = X, None, y, None
    predictors_train, predictors_val = None, None
    predicted_train, predicted_val = None, None

    if eval_percent is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=eval_percent, random_state=seed
        )

    # TRAINING DATA
    X_train = np.tile(X, (n_repeats, 1))
    y_train = np.tile(y, (n_repeats, 1))
    t_train = np.random.uniform(0, 1, size=(y_train.shape[0], 1)) * (T - EPS) + EPS
    z_train = np.random.normal(size=y_train.shape)

    train_mean, train_std = sde.get_mean_std_pt_given_y0(y_train, t_train)
    perturbed_y_train = train_mean + train_std * z_train

    predictors_train = np.concatenate([perturbed_y_train, X_train, t_train], axis=1)
    predicted_train = -1.0 * z_train
    cat_idx = [c + y_train.shape[1] for c in cat_idx] if cat_idx is not None else None

    # VALIDATION DATA
    if eval_percent is not None:
        t_val = np.random.uniform(0, 1, size=(y_test.shape[0], 1)) * (T - EPS) + EPS
        z_val = np.random.normal(size=(y_test.shape[0], y_test.shape[1]))

        val_mean, val_std = sde.get_mean_std_pt_given_y0(y_test, t_val)
        perturbed_y_val = val_mean + val_std * z_val
        predictors_val = np.concatenate([perturbed_y_val, X_test, t_val.reshape(-1, 1)], axis=1)
        predicted_val = -1.0 * z_val

    # cat_idx is not changed
    return predictors_train, predictors_val, predicted_train, predicted_val, cat_idx


###################################################
# Main models
###################################################


class ScoreModel(abc.ABC):
    @abc.abstractmethod
    def score(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        t: Int[np.ndarray, "batch"],
    ):

        pass

    @abc.abstractmethod
    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        sde: DiffusionSDE,
        cat_idx: Optional[List[int]] = None,
    ):
        pass


class LightGBMScoreModel(ScoreModel):
    """
    A score model that uses a LightGBM model (trees) to approximate the score of a given SDE.

    Parameters
    ----------
    n_repeats : int
        How many times to repeat the training dataset when fitting the score. That is, how many
        noisy versions of a point to generate for training.
    eval_percent : float
        Percentage of the training data to use for validation for optional early stopping. It is
        ignored if `early_stopping_rounds` is not set in the `lgbm_args`.
    n_jobs : int
        LightGBM: Number of parallel threads. If set to -1, the number is set to the number of available cores.
    seed : int
        Random seed for generating the training data and fitting the model.
    verbose : int
        Verbosity of the score model.
    **lgbm_args
        Additional arguments to pass to the LightGBM model. See the LightGBM documentation for more
        information. E.g. `early_stopping_rounds`, `n_estimators`, `learning_rate`, `max_depth`,
    """

    def __init__(
        self,
        n_repeats: Optional[int] = 10,
        eval_percent: float = 0.1,
        n_jobs: Optional[int] = -1,
        seed: Optional[int] = None,
        **lgbm_args,
    ) -> None:
        self.n_repeats = n_repeats
        self.eval_percent = eval_percent
        self.n_jobs = n_jobs
        self.seed = seed

        self._lgbm_args = lgbm_args
        self.sde = None
        self.models = None  # Convention inputs are (x, y, t)

    def score(
        self,
        y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Int[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        if self.sde is None:
            raise ValueError("The model has not been fitted yet.")

        scores = []
        predictors = np.concatenate([y, X, t], axis=1)
        _, std = self.sde.get_mean_std_pt_given_y0(y, t)
        for i in range(y.shape[-1]):
            # The score is parametrized: score(x, y, t) = GBT(x, y, t) / std(t)
            score_p = self.models[i].predict(predictors, num_threads=self.n_jobs)
            score = score_p / std[:, i]
            scores.append(score)
        return np.array(scores).T

    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        sde: DiffusionSDE,
        cat_idx: Optional[List[int]] = None,
    ):
        """
        Fit the score model to the data and the given SDE.

        Parameters
        ----------
        X : Float[np.ndarray, "batch x_dim"]
            The input data.
        y : Float[np.ndarray, "batch y_dim"]
            The true output values.
        sde : DiffusionSDE
            The SDE that the model is supposed to approximate the score of.
        cat_idx : Optional[List[int]]
            List of indices of categorical features in the input data. If `None`, all features are
            assumed to be continuous.
        """
        y_dim = y.shape[1]
        self.sde = sde

        lgb_X_train, lgb_X_val, lgb_y_train, lgb_y_val, cat_idx = _make_training_data(
            X=X,
            y=y,
            sde=self.sde,
            n_repeats=self.n_repeats,
            eval_percent=self.eval_percent,
            cat_idx=cat_idx,
            seed=self.seed,
        )

        models = []
        for i in range(y_dim):
            lgb_y_val_i = lgb_y_val[:, i] if lgb_y_val is not None else None
            score_model_i = _fit_one_lgbm_model(
                X=lgb_X_train,
                y=lgb_y_train[:, i],
                X_val=lgb_X_val,
                y_val=lgb_y_val_i,
                cat_idx=cat_idx,
                seed=self.seed,
                n_jobs=self.n_jobs,
                **self._lgbm_args,
            )
            models.append(score_model_i)
        self.models = models
