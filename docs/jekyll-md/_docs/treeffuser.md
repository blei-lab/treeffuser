# treeffuser package

## Subpackages

* [treeffuser.sde package](treeffuser.sde.md)
  * [Submodules](treeffuser.sde.md#submodules)
    * [treeffuser.sde.base_sde module](treeffuser.sde.base_sde.md)
      * [`BaseSDE`](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.BaseSDE)
      * [`CustomSDE`](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.CustomSDE)
      * [`ReverseSDE`](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.ReverseSDE)
    * [treeffuser.sde.base_solver module](treeffuser.sde.base_solver.md)
      * [`BaseSDESolver`](treeffuser.sde.base_solver.md#treeffuser.sde.base_solver.BaseSDESolver)
      * [`get_solver()`](treeffuser.sde.base_solver.md#treeffuser.sde.base_solver.get_solver)
      * [`sdeint()`](treeffuser.sde.base_solver.md#treeffuser.sde.base_solver.sdeint)
    * [treeffuser.sde.diffusion_sdes module](treeffuser.sde.diffusion_sdes.md)
      * [`DiffusionSDE`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.DiffusionSDE)
      * [`SubVPSDE`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.SubVPSDE)
      * [`VESDE`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VESDE)
      * [`VPSDE`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VPSDE)
      * [`get_diffusion_sde()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.get_diffusion_sde)
    * [treeffuser.sde.initialize module](treeffuser.sde.initialize.md)
      * [`ConvergenceWarning`](treeffuser.sde.initialize.md#treeffuser.sde.initialize.ConvergenceWarning)
      * [`initialize_subvpsde()`](treeffuser.sde.initialize.md#treeffuser.sde.initialize.initialize_subvpsde)
      * [`initialize_vesde()`](treeffuser.sde.initialize.md#treeffuser.sde.initialize.initialize_vesde)
      * [`initialize_vpsde()`](treeffuser.sde.initialize.md#treeffuser.sde.initialize.initialize_vpsde)
    * [treeffuser.sde.parameter_schedule module](treeffuser.sde.parameter_schedule.md)
      * [`ExponentialSchedule`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.ExponentialSchedule)
      * [`LinearSchedule`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.LinearSchedule)
      * [`ParameterSchedule`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.ParameterSchedule)
    * [treeffuser.sde.solvers module](treeffuser.sde.solvers.md)
      * [`EulerMaruyama`](treeffuser.sde.solvers.md#treeffuser.sde.solvers.EulerMaruyama)
  * [Module contents](treeffuser.sde.md#module-treeffuser.sde)
    * [`BaseSDE`](treeffuser.sde.md#treeffuser.sde.BaseSDE)
      * [`BaseSDE.drift_and_diffusion()`](treeffuser.sde.md#treeffuser.sde.BaseSDE.drift_and_diffusion)
    * [`CustomSDE`](treeffuser.sde.md#treeffuser.sde.CustomSDE)
      * [`CustomSDE.drift_and_diffusion()`](treeffuser.sde.md#treeffuser.sde.CustomSDE.drift_and_diffusion)
    * [`DiffusionSDE`](treeffuser.sde.md#treeffuser.sde.DiffusionSDE)
      * [`DiffusionSDE.T`](treeffuser.sde.md#treeffuser.sde.DiffusionSDE.T)
      * [`DiffusionSDE.get_hyperparams()`](treeffuser.sde.md#treeffuser.sde.DiffusionSDE.get_hyperparams)
      * [`DiffusionSDE.get_marginalized_perturbation_kernel()`](treeffuser.sde.md#treeffuser.sde.DiffusionSDE.get_marginalized_perturbation_kernel)
      * [`DiffusionSDE.get_mean_std_pt_given_y0()`](treeffuser.sde.md#treeffuser.sde.DiffusionSDE.get_mean_std_pt_given_y0)
      * [`DiffusionSDE.initialize_hyperparams_from_data()`](treeffuser.sde.md#treeffuser.sde.DiffusionSDE.initialize_hyperparams_from_data)
      * [`DiffusionSDE.sample_from_theoretical_prior()`](treeffuser.sde.md#treeffuser.sde.DiffusionSDE.sample_from_theoretical_prior)
    * [`ReverseSDE`](treeffuser.sde.md#treeffuser.sde.ReverseSDE)
      * [`ReverseSDE.drift_and_diffusion()`](treeffuser.sde.md#treeffuser.sde.ReverseSDE.drift_and_diffusion)
    * [`SubVPSDE`](treeffuser.sde.md#treeffuser.sde.SubVPSDE)
      * [`SubVPSDE.drift_and_diffusion()`](treeffuser.sde.md#treeffuser.sde.SubVPSDE.drift_and_diffusion)
      * [`SubVPSDE.get_hyperparams()`](treeffuser.sde.md#treeffuser.sde.SubVPSDE.get_hyperparams)
      * [`SubVPSDE.get_mean_std_pt_given_y0()`](treeffuser.sde.md#treeffuser.sde.SubVPSDE.get_mean_std_pt_given_y0)
      * [`SubVPSDE.initialize_hyperparams_from_data()`](treeffuser.sde.md#treeffuser.sde.SubVPSDE.initialize_hyperparams_from_data)
      * [`SubVPSDE.sample_from_theoretical_prior()`](treeffuser.sde.md#treeffuser.sde.SubVPSDE.sample_from_theoretical_prior)
      * [`SubVPSDE.set_hyperparams()`](treeffuser.sde.md#treeffuser.sde.SubVPSDE.set_hyperparams)
    * [`VESDE`](treeffuser.sde.md#treeffuser.sde.VESDE)
      * [`VESDE.drift_and_diffusion()`](treeffuser.sde.md#treeffuser.sde.VESDE.drift_and_diffusion)
      * [`VESDE.get_hyperparams()`](treeffuser.sde.md#treeffuser.sde.VESDE.get_hyperparams)
      * [`VESDE.get_mean_std_pt_given_y0()`](treeffuser.sde.md#treeffuser.sde.VESDE.get_mean_std_pt_given_y0)
      * [`VESDE.initialize_hyperparams_from_data()`](treeffuser.sde.md#treeffuser.sde.VESDE.initialize_hyperparams_from_data)
      * [`VESDE.sample_from_theoretical_prior()`](treeffuser.sde.md#treeffuser.sde.VESDE.sample_from_theoretical_prior)
      * [`VESDE.set_hyperparams()`](treeffuser.sde.md#treeffuser.sde.VESDE.set_hyperparams)
    * [`VPSDE`](treeffuser.sde.md#treeffuser.sde.VPSDE)
      * [`VPSDE.drift_and_diffusion()`](treeffuser.sde.md#treeffuser.sde.VPSDE.drift_and_diffusion)
      * [`VPSDE.get_hyperparams()`](treeffuser.sde.md#treeffuser.sde.VPSDE.get_hyperparams)
      * [`VPSDE.get_mean_std_pt_given_y0()`](treeffuser.sde.md#treeffuser.sde.VPSDE.get_mean_std_pt_given_y0)
      * [`VPSDE.initialize_hyperparams_from_data()`](treeffuser.sde.md#treeffuser.sde.VPSDE.initialize_hyperparams_from_data)
      * [`VPSDE.sample_from_theoretical_prior()`](treeffuser.sde.md#treeffuser.sde.VPSDE.sample_from_theoretical_prior)
      * [`VPSDE.set_hyperparams()`](treeffuser.sde.md#treeffuser.sde.VPSDE.set_hyperparams)
    * [`get_diffusion_sde()`](treeffuser.sde.md#treeffuser.sde.get_diffusion_sde)
    * [`get_solver()`](treeffuser.sde.md#treeffuser.sde.get_solver)
    * [`sdeint()`](treeffuser.sde.md#treeffuser.sde.sdeint)

## Submodules

* [treeffuser.scaler module](treeffuser.scaler.md)
  * [`ScalerMixedTypes`](treeffuser.scaler.md#treeffuser.scaler.ScalerMixedTypes)
    * [`ScalerMixedTypes.fit()`](treeffuser.scaler.md#treeffuser.scaler.ScalerMixedTypes.fit)
    * [`ScalerMixedTypes.fit_transform()`](treeffuser.scaler.md#treeffuser.scaler.ScalerMixedTypes.fit_transform)
    * [`ScalerMixedTypes.inverse_transform()`](treeffuser.scaler.md#treeffuser.scaler.ScalerMixedTypes.inverse_transform)
    * [`ScalerMixedTypes.transform()`](treeffuser.scaler.md#treeffuser.scaler.ScalerMixedTypes.transform)
* [treeffuser.treeffuser module](treeffuser.treeffuser.md)
  * [`Treeffuser`](treeffuser.treeffuser.md#treeffuser.treeffuser.Treeffuser)
    * [`Treeffuser.get_new_score_model()`](treeffuser.treeffuser.md#treeffuser.treeffuser.Treeffuser.get_new_score_model)
    * [`Treeffuser.get_new_sde()`](treeffuser.treeffuser.md#treeffuser.treeffuser.Treeffuser.get_new_sde)
    * [`Treeffuser.n_estimators_true`](treeffuser.treeffuser.md#treeffuser.treeffuser.Treeffuser.n_estimators_true)
    * [`Treeffuser.set_fit_request()`](treeffuser.treeffuser.md#treeffuser.treeffuser.Treeffuser.set_fit_request)
    * [`Treeffuser.set_predict_request()`](treeffuser.treeffuser.md#treeffuser.treeffuser.Treeffuser.set_predict_request)

## Module contents

### *class* treeffuser.Treeffuser(n_repeats: int = 30, n_estimators: int = 3000, early_stopping_rounds: int | None = 50, eval_percent: float = 0.1, num_leaves: int = 31, max_depth: int = -1, learning_rate: float = 0.1, max_bin: int = 255, subsample_for_bin: int = 200000, min_child_samples: int = 20, subsample: float = 1.0, subsample_freq: int = 0, n_jobs: int = -1, sde_name: str = 'vesde', sde_initialize_from_data: bool = False, sde_hyperparam_min: float | Literal['default'] | None = None, sde_hyperparam_max: float | Literal['default'] | None = None, seed: int | None = None, verbose: int = 0, extra_lightgbm_params: dict | None = None)

Bases: `BaseTabularDiffusion`

n_repeats
: How many times to repeat the training dataset when fitting the score. That is, how many
  noisy versions of a point to generate for training.

n_estimators
: LightGBM: Number of boosting iterations.

early_stopping_rounds
: LightGBM: If None, no early stopping is performed. Otherwise, the model will stop training
  if no improvement is observed in the validation set for early_stopping_rounds consecutive
  iterations.

eval_percent
: LightGBM: Percentage of the training data to use for validation if early_stopping_rounds
  is not None.

num_leaves
: LightGBM: Maximum tree leaves for base learners.

max_depth
: LightGBM: Maximum tree depth for base learners, <=0 means no limit.

learning_rate
: LightGBM: Boosting learning rate.

max_bin
: LightGBM: Max number of bins that feature values will be bucketed in. This is used for
  lightgbm’s histogram binning algorithm.

subsample_for_bin
: LightGBM: Number of samples for constructing bins.

min_child_samples
: LightGBM: Minimum number of data needed in a child (leaf). If less than this number, will
  not create the child.

subsample
: LightGBM: Subsample ratio of the training instance.

subsample_freq
: LightGBM: Frequency of subsample, <=0 means no enable. How often to subsample the training
  data.

n_jobs
: LightGBM: Number of parallel threads. If set to -1, the number is set to the number of available cores.

sde_name
: SDE: Name of the SDE to use. See treeffuser.sde.get_diffusion_sde for available SDEs.

sde_initialize_from_data
: SDE: Whether to initialize the SDE from the data. If True, the SDE hyperparameters are
  initialized with a heuristic based on the data (see treeffuser.sde.initialize.py).
  Otherwise, sde_hyperparam_min and sde_hyperparam_max are used. (default: False)

sde_hyperparam_min
: SDE: The scale of the SDE at t=0 (see VESDE, VPSDE, SubVPSDE).

sde_hyperparam_max
: SDE: The scale of the SDE at t=T (see VESDE, VPSDE, SubVPSDE).

seed
: Random seed for generating the training data and fitting the model.

verbose
: Verbosity of the score model.

#### get_new_score_model()

Return the score model.

#### get_new_sde()

Return the SDE model.

#### *property* n_estimators_true *: List[int]*

The number of estimators that are actually used in the models (after early stopping),
one for each dimension of the score (i.e. the dimension of y).

#### set_fit_request(\*, cat_idx: bool | None | str = '$UNCHANGED$')

Request metadata passed to the `fit` method.

Note that this method is only relevant if
`enable_metadata_routing=True` (see `sklearn.set_config()`).
Please see User Guide on how the routing
mechanism works.

The options for each parameter are:

- `True`: metadata is requested, and passed to `fit` if provided. The request is ignored if metadata is not provided.
- `False`: metadata is not requested and the meta-estimator will not pass it to `fit`.
- `None`: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
- `str`: metadata should be passed to the meta-estimator with this given alias instead of the original name.

The default (`sklearn.utils.metadata_routing.UNCHANGED`) retains the
existing request. This allows you to change the request for some
parameters and not others.

#### Versionadded
Added in version 1.3.

#### NOTE
This method is only relevant if this estimator is used as a
sub-estimator of a meta-estimator, e.g. used inside a
`Pipeline`. Otherwise it has no effect.

* **Parameters:**
  **cat_idx** (*str* *,* *True* *,* *False* *, or* *None* *,*                     *default=sklearn.utils.metadata_routing.UNCHANGED*) – Metadata routing for `cat_idx` parameter in `fit`.
* **Returns:**
  **self** – The updated object.
* **Return type:**
  object

#### set_predict_request(\*, max_samples: bool | None | str = '$UNCHANGED$', tol: bool | None | str = '$UNCHANGED$', verbose: bool | None | str = '$UNCHANGED$')

Request metadata passed to the `predict` method.

Note that this method is only relevant if
`enable_metadata_routing=True` (see `sklearn.set_config()`).
Please see User Guide on how the routing
mechanism works.

The options for each parameter are:

- `True`: metadata is requested, and passed to `predict` if provided. The request is ignored if metadata is not provided.
- `False`: metadata is not requested and the meta-estimator will not pass it to `predict`.
- `None`: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
- `str`: metadata should be passed to the meta-estimator with this given alias instead of the original name.

The default (`sklearn.utils.metadata_routing.UNCHANGED`) retains the
existing request. This allows you to change the request for some
parameters and not others.

#### Versionadded
Added in version 1.3.

#### NOTE
This method is only relevant if this estimator is used as a
sub-estimator of a meta-estimator, e.g. used inside a
`Pipeline`. Otherwise it has no effect.

* **Parameters:**
  * **max_samples** (*str* *,* *True* *,* *False* *, or* *None* *,*                     *default=sklearn.utils.metadata_routing.UNCHANGED*) – Metadata routing for `max_samples` parameter in `predict`.
  * **tol** (*str* *,* *True* *,* *False* *, or* *None* *,*                     *default=sklearn.utils.metadata_routing.UNCHANGED*) – Metadata routing for `tol` parameter in `predict`.
  * **verbose** (*str* *,* *True* *,* *False* *, or* *None* *,*                     *default=sklearn.utils.metadata_routing.UNCHANGED*) – Metadata routing for `verbose` parameter in `predict`.
* **Returns:**
  **self** – The updated object.
* **Return type:**
  object
