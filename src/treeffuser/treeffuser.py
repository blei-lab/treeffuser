from typing import Literal
from typing import Optional
from typing import Union

from treeffuser._base_tabular_diffusion import BaseTabularDiffusion
from treeffuser._score_models import LightGBMScoreModel
from treeffuser._score_models import ScoreModel
from treeffuser.sde import DiffusionSDE
from treeffuser.sde import get_diffusion_sde


class Treeffuser(BaseTabularDiffusion):
    def __init__(
        self,
        n_repeats: int = 10,
        n_estimators: int = 100,
        eval_percent: Optional[float] = None,
        early_stopping_rounds: Optional[int] = None,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        max_bin: int = 255,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        n_jobs: int = -1,
        sde_name: str = "vesde",
        sde_initialize_from_data: bool = False,
        sde_hyperparam_min: Union[float, Literal["default"]] = None,
        sde_hyperparam_max: Union[float, Literal["default"]] = None,
        seed: Optional[int] = None,
        verbose: int = 0,
        extra_lightgbm_params: Optional[dict] = None,
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
        super().__init__(
            sde_initialize_from_data=sde_initialize_from_data,
        )
        self.sde_name = sde_name
        self.n_repeats = n_repeats
        self.n_estimators = n_estimators
        self.eval_percent = eval_percent
        self.early_stopping_rounds = early_stopping_rounds
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_bin = max_bin
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose
        self.sde_initialize_from_data = sde_initialize_from_data
        self.sde_hyperparam_min = sde_hyperparam_min
        self.sde_hyperparam_max = sde_hyperparam_max
        self.extra_lightgbm_params = extra_lightgbm_params or {}

    def get_new_sde(self) -> DiffusionSDE:
        sde_cls = get_diffusion_sde(self.sde_name)
        if not issubclass(sde_cls, DiffusionSDE):
            raise ValueError(f"SDE {sde_cls} is not a subclass of DiffusionSDE")
        sde_kwargs = {}
        if self.sde_hyperparam_min is not None:
            sde_kwargs["hyperparam_min"] = self.sde_hyperparam_min
        if self.sde_hyperparam_max is not None:
            sde_kwargs["hyperparam_max"] = self.sde_hyperparam_max
        sde = sde_cls(**sde_kwargs)
        return sde

    def get_new_score_model(self) -> ScoreModel:
        score_model = LightGBMScoreModel(
            n_repeats=self.n_repeats,
            n_estimators=self.n_estimators,
            eval_percent=self.eval_percent,
            early_stopping_rounds=self.early_stopping_rounds,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_bin=self.max_bin,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            verbose=self.verbose,
            seed=self.seed,
            n_jobs=self.n_jobs,
            **self.extra_lightgbm_params,
        )
        return score_model
