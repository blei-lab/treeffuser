"""
This should be the main file corresponding to the project.
"""

from typing import List
from typing import Optional

from jaxtyping import Float
from ml_collections import ConfigDict
from numpy import ndarray

import treeffuser._score_models as _score_models
from treeffuser._base_tabular_diffusion import BaseTabularDiffusion


class Treeffuser(BaseTabularDiffusion):
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
