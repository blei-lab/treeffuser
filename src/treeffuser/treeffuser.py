from typing import List
from typing import Literal
from typing import Optional

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
        early_stopping_rounds: Optional[int] = None,
        eval_percent: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        max_bin: int = 255,
        subsample_for_bin: int = 200000,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        n_jobs: int = -1,
        sde_name: str = "vesde",
        sde_initialize_from_data: bool = False,
        sde_hyperparam_min: Optional[float | Literal["default"]] = None,
        sde_hyperparam_max: Optional[float | Literal["default"]] = None,
        seed: Optional[int] = None,
        verbose: int = 0,
        extra_lightgbm_params: Optional[dict] = None,
    ):
        """
        n_repeats : int
            How many times to repeat the training dataset when fitting the score. That is, how many
            noisy versions of a point to generate for training.
        n_estimators : int
            LightGBM: Number of boosting iterations.
        early_stopping_rounds : int
            LightGBM: If `None`, no early stopping is performed. Otherwise, the model will stop training
            if no improvement is observed in the validation set for `early_stopping_rounds` consecutive
            iterations.
        eval_percent : float
            LightGBM: Percentage of the training data to use for validation if `early_stopping_rounds`
            is not `None`.
        num_leaves : int
            LightGBM: Maximum tree leaves for base learners.
        max_depth : int
            LightGBM: Maximum tree depth for base learners, <=0 means no limit.
        learning_rate : float
            LightGBM: Boosting learning rate.
        max_bin : int
            LightGBM: Max number of bins that feature values will be bucketed in. This is used for
            lightgbm's histogram binning algorithm.
        subsample_for_bin : int
            LightGBM: Number of samples for constructing bins.
        min_child_samples : int
            LightGBM: Minimum number of data needed in a child (leaf). If less than this number, will
            not create the child.
        subsample : float
            LightGBM: Subsample ratio of the training instance.
        subsample_freq : int
            LightGBM: Frequency of subsample, <=0 means no enable. How often to subsample the training
            data.
        n_jobs : int
            LightGBM: Number of parallel threads. If set to -1, the number is set to the number of available cores.
        sde_name : str
            SDE: Name of the SDE to use. See `treeffuser.sde.get_diffusion_sde` for available SDEs.
        sde_initialize_from_data : bool
            SDE: Whether to initialize the SDE from the data. If `True`, the SDE hyperparameters are
            initialized with a heuristic based on the data (see `treeffuser.sde.initialize.py`).
            Otherwise, sde_hyperparam_min and sde_hyperparam_max are used. (default: False)
        sde_hyperparam_min : float or "default"
            SDE: The scale of the SDE at t=0 (see `VESDE`, `VPSDE`, `SubVPSDE`).
        sde_hyperparam_max : float or "default"
            SDE: The scale of the SDE at t=T (see `VESDE`, `VPSDE`, `SubVPSDE`).
        seed : int
            Random seed for generating the training data and fitting the model.
        verbose : int
            Verbosity of the score model.
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
        self.subsample_for_bin = subsample_for_bin
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
            subsample_for_bin=self.subsample_for_bin,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            verbose=self.verbose,
            seed=self.seed,
            n_jobs=self.n_jobs,
            **self.extra_lightgbm_params,
        )
        return score_model

    @property
    def n_estimators_true(self) -> List[int]:
        """
        The number of estimators that are actually used in the models (after early stopping),
        one for each dimension of the score (i.e. the dimension of y).
        """
        return self.score_model.n_estimators_true
