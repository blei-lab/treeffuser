# treeffuser.sde package

## Submodules

* [treeffuser.sde.base_sde module](treeffuser.sde.base_sde.md)
  * [`BaseSDE`](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.BaseSDE)
    * [`BaseSDE.drift_and_diffusion()`](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.BaseSDE.drift_and_diffusion)
  * [`CustomSDE`](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.CustomSDE)
    * [`CustomSDE.drift_and_diffusion()`](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.CustomSDE.drift_and_diffusion)
  * [`ReverseSDE`](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.ReverseSDE)
    * [`ReverseSDE.drift_and_diffusion()`](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.ReverseSDE.drift_and_diffusion)
* [treeffuser.sde.base_solver module](treeffuser.sde.base_solver.md)
  * [`BaseSDESolver`](treeffuser.sde.base_solver.md#treeffuser.sde.base_solver.BaseSDESolver)
    * [`BaseSDESolver.integrate()`](treeffuser.sde.base_solver.md#treeffuser.sde.base_solver.BaseSDESolver.integrate)
    * [`BaseSDESolver.step()`](treeffuser.sde.base_solver.md#treeffuser.sde.base_solver.BaseSDESolver.step)
  * [`get_solver()`](treeffuser.sde.base_solver.md#treeffuser.sde.base_solver.get_solver)
  * [`sdeint()`](treeffuser.sde.base_solver.md#treeffuser.sde.base_solver.sdeint)
* [treeffuser.sde.diffusion_sdes module](treeffuser.sde.diffusion_sdes.md)
  * [`DiffusionSDE`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.DiffusionSDE)
    * [`DiffusionSDE.T`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.DiffusionSDE.T)
    * [`DiffusionSDE.get_hyperparams()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.DiffusionSDE.get_hyperparams)
    * [`DiffusionSDE.get_marginalized_perturbation_kernel()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.DiffusionSDE.get_marginalized_perturbation_kernel)
    * [`DiffusionSDE.get_mean_std_pt_given_y0()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.DiffusionSDE.get_mean_std_pt_given_y0)
    * [`DiffusionSDE.initialize_hyperparams_from_data()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.DiffusionSDE.initialize_hyperparams_from_data)
    * [`DiffusionSDE.sample_from_theoretical_prior()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.DiffusionSDE.sample_from_theoretical_prior)
  * [`SubVPSDE`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.SubVPSDE)
    * [`SubVPSDE.drift_and_diffusion()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.SubVPSDE.drift_and_diffusion)
    * [`SubVPSDE.get_hyperparams()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.SubVPSDE.get_hyperparams)
    * [`SubVPSDE.get_mean_std_pt_given_y0()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.SubVPSDE.get_mean_std_pt_given_y0)
    * [`SubVPSDE.initialize_hyperparams_from_data()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.SubVPSDE.initialize_hyperparams_from_data)
    * [`SubVPSDE.sample_from_theoretical_prior()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.SubVPSDE.sample_from_theoretical_prior)
    * [`SubVPSDE.set_hyperparams()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.SubVPSDE.set_hyperparams)
  * [`VESDE`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VESDE)
    * [`VESDE.drift_and_diffusion()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VESDE.drift_and_diffusion)
    * [`VESDE.get_hyperparams()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VESDE.get_hyperparams)
    * [`VESDE.get_mean_std_pt_given_y0()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VESDE.get_mean_std_pt_given_y0)
    * [`VESDE.initialize_hyperparams_from_data()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VESDE.initialize_hyperparams_from_data)
    * [`VESDE.sample_from_theoretical_prior()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VESDE.sample_from_theoretical_prior)
    * [`VESDE.set_hyperparams()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VESDE.set_hyperparams)
  * [`VPSDE`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VPSDE)
    * [`VPSDE.drift_and_diffusion()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VPSDE.drift_and_diffusion)
    * [`VPSDE.get_hyperparams()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VPSDE.get_hyperparams)
    * [`VPSDE.get_mean_std_pt_given_y0()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VPSDE.get_mean_std_pt_given_y0)
    * [`VPSDE.initialize_hyperparams_from_data()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VPSDE.initialize_hyperparams_from_data)
    * [`VPSDE.sample_from_theoretical_prior()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VPSDE.sample_from_theoretical_prior)
    * [`VPSDE.set_hyperparams()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.VPSDE.set_hyperparams)
  * [`get_diffusion_sde()`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.get_diffusion_sde)
* [treeffuser.sde.initialize module](treeffuser.sde.initialize.md)
  * [`ConvergenceWarning`](treeffuser.sde.initialize.md#treeffuser.sde.initialize.ConvergenceWarning)
  * [`initialize_subvpsde()`](treeffuser.sde.initialize.md#treeffuser.sde.initialize.initialize_subvpsde)
  * [`initialize_vesde()`](treeffuser.sde.initialize.md#treeffuser.sde.initialize.initialize_vesde)
  * [`initialize_vpsde()`](treeffuser.sde.initialize.md#treeffuser.sde.initialize.initialize_vpsde)
* [treeffuser.sde.parameter_schedule module](treeffuser.sde.parameter_schedule.md)
  * [`ExponentialSchedule`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.ExponentialSchedule)
    * [`ExponentialSchedule.get_derivative()`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.ExponentialSchedule.get_derivative)
    * [`ExponentialSchedule.get_integral()`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.ExponentialSchedule.get_integral)
    * [`ExponentialSchedule.get_value()`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.ExponentialSchedule.get_value)
  * [`LinearSchedule`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.LinearSchedule)
    * [`LinearSchedule.get_derivative()`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.LinearSchedule.get_derivative)
    * [`LinearSchedule.get_integral()`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.LinearSchedule.get_integral)
    * [`LinearSchedule.get_value()`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.LinearSchedule.get_value)
  * [`ParameterSchedule`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.ParameterSchedule)
    * [`ParameterSchedule.get_derivative()`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.ParameterSchedule.get_derivative)
    * [`ParameterSchedule.get_integral()`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.ParameterSchedule.get_integral)
    * [`ParameterSchedule.get_value()`](treeffuser.sde.parameter_schedule.md#treeffuser.sde.parameter_schedule.ParameterSchedule.get_value)
* [treeffuser.sde.solvers module](treeffuser.sde.solvers.md)
  * [`EulerMaruyama`](treeffuser.sde.solvers.md#treeffuser.sde.solvers.EulerMaruyama)
    * [`EulerMaruyama.step()`](treeffuser.sde.solvers.md#treeffuser.sde.solvers.EulerMaruyama.step)

## Module contents

### *class* treeffuser.sde.BaseSDE

Bases: `ABC`

This abstract class represents a stochastic differential equation (SDE) of the form
dY = f(Y, t) dt + g(Y, t) dW, where:
- Y is the variable of the SDE
- t is time
- f is the drift function
- g is the diffusion function

Any class that inherits from BaseSDE must implement the drift_and_diffusion(y, t) method,
which returns a tuple containing the drift and the diffusion at time t for a given state Y=y.

### References

[1] [https://en.wikipedia.org/wiki/Stochastic_differential_equation](https://en.wikipedia.org/wiki/Stochastic_differential_equation)

#### *abstract* drift_and_diffusion(y: Float[ndarray, 'batch y_dim'], t: Float[ndarray, 'batch 1'])

Computes the drift and diffusion at a given time t for a given state Y=y.

* **Parameters:**
  * **y** (*Float* *[**ndarray* *,*  *"batch y_dim"* *]*) – The state of the SDE.
  * **t** (*Float* *[**ndarray* *,*  *"batch 1"* *]*) – The time at which to compute the drift and diffusion.
* **Returns:**
  A tuple containing the drift and the diffusion at time t for a given state Y=y.
* **Return type:**
  tuple

### *class* treeffuser.sde.CustomSDE(drift_fn: Callable[[Float[ndarray, 'batch y_dim'], Float[ndarray, 'batch 1']], Float[ndarray, 'batch y_dim']], diffusion_fn: Callable[[Float[ndarray, 'batch y_dim'], Float[ndarray, 'batch 1']], Float[ndarray, 'batch y_dim']])

Bases: [`BaseSDE`](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.BaseSDE)

SDE defined by a custom drift and diffusion functions.

### Parameters:

drift_fn
: Drift function of the SDE.

diffusion_fn
: Diffusion function of the SDE.

#### drift_and_diffusion(y: Float[ndarray, 'batch y_dim'], t: Float[ndarray, 'batch 1'])

Computes the drift and diffusion at a given time t for a given state Y=y.

* **Parameters:**
  * **y** (*Float* *[**ndarray* *,*  *"batch y_dim"* *]*) – The state of the SDE.
  * **t** (*Float* *[**ndarray* *,*  *"batch 1"* *]*) – The time at which to compute the drift and diffusion.
* **Returns:**
  A tuple containing the drift and the diffusion at time t for a given state Y=y.
* **Return type:**
  tuple

### *class* treeffuser.sde.DiffusionSDE

Bases: [`BaseSDE`](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.BaseSDE)

Abstract class representing a diffusion SDE:
dY = (A(t) + B(t) Y) dt + C(t) dW where C(t) is a time-varying
diffusion coefficient independent of Y, and the drift is an affine function of Y.
As a result, the conditional distribution p_t(y | y0) is Gaussian.

#### *property* T *: float*

End time of the SDE.

#### *abstract* get_hyperparams()

Return a dictionary with the hyperparameters of the SDE.

The hyperparameters parametrize the drift and diffusion coefficients of the
SDE.

#### get_marginalized_perturbation_kernel(y0: Float[ndarray, 'batch y_dim'])

> Compute the marginalized perturbation kernel density function induced by the data y0.

> The marginalized perturbation kernel is defined as:
> : ```
>   `
>   ```
>   <br/>
>   p(y, t) =

rac{1}{n}sum_{y’ in y0}p_t(y | y’)\`
: where n is the number of data points in y0. Each p_t(y | y’) is a Gaussian
  density with conditional mean and standard deviation given by marginal_prob.
  <br/>
  Args:
  : y0: data
  <br/>
  Returns:
  : kernel_density_fn: function taking y_prime and t as input and returning
    the perturbation kernel density function induced by the diffusion of data y0
    for time t.

#### *abstract* get_mean_std_pt_given_y0(y0: Float[ndarray, 'batch y_dim'], t: Float[ndarray, 'batch 1'])

Our diffusion SDEs have conditional distributions p_t(y | y0) that
are Gaussian. This method returns their mean and standard deviation.

* **Parameters:**
  * **y0** (ndarray of shape (

    ```
    *
    ```

    batch, y_dim)) – Initial value at t=0.
  * **t** (*float*) – Time at which to compute the conditional distribution.
* **Returns:**
  * **mean** (*ndarray of shape (\*batch, y_dim)*) – Mean of the conditional distribution.
  * **std** (*ndarray of shape (\*batch, y_dim)*) – Standard deviation of the conditional distribution.

#### initialize_hyperparams_from_data(y0: Float[ndarray, 'batch y_dim'])

Initialize the hyperparameters of the SDE from the data y0.

This method can be implemented by subclasses to initialize the hyperparameters.

* **Parameters:**
  **y0** (ndarray of shape (

  ```
  *
  ```

  batch, y_dim)) – Data y0.

#### *abstract* sample_from_theoretical_prior(shape: tuple[int, ...], seed: int | None = None)

Sample from the theoretical distribution that p_T(y) converges to.

* **Parameters:**
  * **shape** (*tuple*) – Shape of the output array.
  * **seed** (*int* *(**optional* *)*) – Random seed. If None, the random number generator is not seeded.

### *class* treeffuser.sde.ReverseSDE(sde: [BaseSDE](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.BaseSDE), t_reverse_origin: float, score_fn: Callable[[Float[ndarray, 'batch y_dim'], Float[ndarray, 'batch']], Float[ndarray, 'batch']])

Bases: [`BaseSDE`](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.BaseSDE)

The ReverseSDE class represents a stochastic differential equation (SDE) reversed
in time.

An SDE requires a transformation of the drift term to be reversed, which is based on
the score function of the marginal distributions induced by the original SDE [1].
The original SDE dY = f(Y, t) dt + g(Y, t) dW can be reversed from time T to
define a new SDE:
dY(T-t) = (-f(Y, T-t) + g(Y, T-t)² ∇[log p(Y(T-t))]) dt + g(Y, T-t) dW.

* **Parameters:**
  * **sde** ([*BaseSDE*](#treeffuser.sde.BaseSDE)) – The original SDE.
  * **t_reverse_origin** (*float*) – The time from which to reverse the SDE.
  * **score_fn** – The score function of the original SDE induced marginal distributions.

### References

[1] [https://openreview.net/pdf?id=PxTIG12RRHS](https://openreview.net/pdf?id=PxTIG12RRHS)

#### drift_and_diffusion(y: Float[ndarray, 'batch y_dim'], t: Float[ndarray, 'batch 1'])

Computes the drift and diffusion at a given time t for a given state Y=y.

* **Parameters:**
  * **y** (*Float* *[**ndarray* *,*  *"batch y_dim"* *]*) – The state of the SDE.
  * **t** (*Float* *[**ndarray* *,*  *"batch 1"* *]*) – The time at which to compute the drift and diffusion.
* **Returns:**
  A tuple containing the drift and the diffusion at time t for a given state Y=y.
* **Return type:**
  tuple

### *class* treeffuser.sde.SubVPSDE(hyperparam_min=0.01, hyperparam_max=20)

Bases: [`DiffusionSDE`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.DiffusionSDE)

Sub-Variance-preserving SDE (SubVPSDE):
dY = -0.5 beta(t) Y dt + sqrt{beta(t) (1 - e^{-2 int_0^t beta(s) ds})} dW
where beta(t) is a time-varying coefficient with linear schedule:
beta(t) = hyperparam_min + (hyperparam_max - hyperparam_min) \* t.

The SDE converges to a standard normal distribution for large hyperparam_max.

* **Parameters:**
  * **hyperparam_min** (*float*) – Minimum value of the time-varying coefficient beta(t).
  * **hyperparam_max** (*float*) – Maximum value of the time-varying coefficient beta(t).

#### drift_and_diffusion(y: Float[ndarray, 'batch y_dim'], t: Float[ndarray, 'batch 1'])

Computes the drift and diffusion at a given time t for a given state Y=y.

* **Parameters:**
  * **y** (*Float* *[**ndarray* *,*  *"batch y_dim"* *]*) – The state of the SDE.
  * **t** (*Float* *[**ndarray* *,*  *"batch 1"* *]*) – The time at which to compute the drift and diffusion.
* **Returns:**
  A tuple containing the drift and the diffusion at time t for a given state Y=y.
* **Return type:**
  tuple

#### get_hyperparams()

Return a dictionary with the hyperparameters of the SDE.

The hyperparameters parametrize the drift and diffusion coefficients of the
SDE.

#### get_mean_std_pt_given_y0(y0: Float[ndarray, 'batch y_dim'], t: Float[ndarray, 'batch 1'])

The conditional distribution is Gaussian with:
: mean: y0 \* exp(-0.5 \* int_0^t1 beta(s) ds)
  variance: [1 - exp(-int_0^t1 beta(s) ds)]^2

#### initialize_hyperparams_from_data(y0: Float[ndarray, 'batch y_dim'])

Initialize the hyperparameters of the SDE from the data y0.

* **Parameters:**
  **y0** (ndarray of shape (

  ```
  *
  ```

  batch, y_dim)) – Data y0.

#### sample_from_theoretical_prior(shape: tuple[int, ...], seed: int | None = None)

Sample from the theoretical distribution that p_T(y) converges to.

* **Parameters:**
  * **shape** (*tuple*) – Shape of the output array.
  * **seed** (*int* *(**optional* *)*) – Random seed. If None, the random number generator is not seeded.

#### set_hyperparams(hyperparam_min, hyperparam_max)

### *class* treeffuser.sde.VESDE(hyperparam_min=0.01, hyperparam_max=20)

Bases: [`DiffusionSDE`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.DiffusionSDE)

Variance-exploding SDE (VESDE):
: dY = 0 dt + sqrt{2 sigma(t) sigma’(t)} dW

where sigma(t) is a time-varying diffusion coefficient with exponential schedule:

sigma(t) = hyperparam_min \* (hyperparam_max / hyperparam_min) ^ t

The SDE converges to a normal distribution with variance hyperparam_max ^ 2.

* **Parameters:**
  * **hyperparam_min** (*float*) – Minimum value of the diffusion coefficient.
  * **hyperparam_max** (*float*) – Maximum value of the diffusion coefficient.

#### drift_and_diffusion(y: Float[ndarray, 'batch y_dim'], t: Float[ndarray, 'batch 1'])

Computes the drift and diffusion at a given time t for a given state Y=y.

* **Parameters:**
  * **y** (*Float* *[**ndarray* *,*  *"batch y_dim"* *]*) – The state of the SDE.
  * **t** (*Float* *[**ndarray* *,*  *"batch 1"* *]*) – The time at which to compute the drift and diffusion.
* **Returns:**
  A tuple containing the drift and the diffusion at time t for a given state Y=y.
* **Return type:**
  tuple

#### get_hyperparams()

Return a dictionary with the hyperparameters of the SDE.

The hyperparameters parametrize the drift and diffusion coefficients of the
SDE.

#### get_mean_std_pt_given_y0(y0: Float[ndarray, 'batch y_dim'], t: Float[ndarray, 'batch 1'])

The conditional distribution is Gaussian with:
: mean: y0
  variance: hyperparam(t)\*\*2 - hyperparam(0)\*\*2

#### initialize_hyperparams_from_data(y0: Float[ndarray, 'batch y_dim'])

Initialize the hyperparameters of the SDE from the data y0.

* **Parameters:**
  **y0** (ndarray of shape (

  ```
  *
  ```

  batch, y_dim)) – Data y0.

#### sample_from_theoretical_prior(shape: tuple[int, ...], seed: int | None = None)

Sample from the theoretical distribution that p_T(y) converges to.

* **Parameters:**
  * **shape** (*tuple*) – Shape of the output array.
  * **seed** (*int* *(**optional* *)*) – Random seed. If None, the random number generator is not seeded.

#### set_hyperparams(hyperparam_min, hyperparam_max)

### *class* treeffuser.sde.VPSDE(hyperparam_min=0.01, hyperparam_max=20)

Bases: [`DiffusionSDE`](treeffuser.sde.diffusion_sdes.md#treeffuser.sde.diffusion_sdes.DiffusionSDE)

Variance-preserving SDE (VPSDE):
dY = -0.5 beta(t) Y dt + sqrt{beta(t)} dW
where beta(t) is a time-varying coefficient with linear schedule:
beta(t) = hyperparam_min + (hyperparam_max - hyperparam_min) \* t.

The SDE converges to a standard normal distribution for large hyperparam_max.

* **Parameters:**
  * **hyperparam_min** (*float*) – Minimum value of the time-varying coefficient beta(t).
  * **hyperparam_max** (*float*) – Maximum value of the time-varying coefficient beta(t).

#### drift_and_diffusion(y: Float[ndarray, 'batch y_dim'], t: Float[ndarray, 'batch 1'])

Computes the drift and diffusion at a given time t for a given state Y=y.

* **Parameters:**
  * **y** (*Float* *[**ndarray* *,*  *"batch y_dim"* *]*) – The state of the SDE.
  * **t** (*Float* *[**ndarray* *,*  *"batch 1"* *]*) – The time at which to compute the drift and diffusion.
* **Returns:**
  A tuple containing the drift and the diffusion at time t for a given state Y=y.
* **Return type:**
  tuple

#### get_hyperparams()

Return a dictionary with the hyperparameters of the SDE.

The hyperparameters parametrize the drift and diffusion coefficients of the
SDE.

#### get_mean_std_pt_given_y0(y0: Float[ndarray, 'batch y_dim'], t: Float[ndarray, 'batch 1'])

The conditional distribution is Gaussian with:
: mean: y0 \* exp(-0.5 \* int_0^t1 beta(s) ds)
  variance: 1 - exp(-int_0^t1 beta(s) ds)

#### initialize_hyperparams_from_data(y0: Float[ndarray, 'batch y_dim'])

Initialize the hyperparameters of the SDE from the data y0.

* **Parameters:**
  **y0** (ndarray of shape (

  ```
  *
  ```

  batch, y_dim)) – Data y0.

#### sample_from_theoretical_prior(shape: tuple[int, ...], seed: int | None = None)

Sample from the theoretical distribution that p_T(y) converges to.

* **Parameters:**
  * **shape** (*tuple*) – Shape of the output array.
  * **seed** (*int* *(**optional* *)*) – Random seed. If None, the random number generator is not seeded.

#### set_hyperparams(hyperparam_min, hyperparam_max)

### treeffuser.sde.get_diffusion_sde(name: str | None = None)

Function to retrieve a registered diffusion SDE by its name.

### Parameters:

name: str
: The name of the SDE to retrieve. If None, returns a dictionary of all available SDEs.

### Returns:

BaseSDE or dict: The SDE class corresponding to the given name, or a dictionary of all available SDEs.

### Raises:

ValueError: If the given name is not registered.

### Examples:

```pycon
>>> sde_class = get_diffusion_sde("my_sde")
>>> sde_instance = sde_class()
```

### treeffuser.sde.get_solver(name)

Function to retrieve a registered solver by its name.

* **Parameters:**
  **name** (*str*) – The name of the solver.
* **Raises:**
  **ValueError** – If the solver with the given name is not registered.
* **Returns:**
  The class of the registered solver.

### Examples

```pycon
>>> solver_class = get_solver("my_solver")
>>> solver_instance = solver_class()
```

### treeffuser.sde.sdeint(sde, y0, t0=0.0, t1=1.0, method='euler', n_steps=20, score_fn=None, n_samples=1, seed=None)

Integrate an SDE (i.e. sample from an SDE).

* **Parameters:**
  * **sde** ([*BaseSDE*](#treeffuser.sde.BaseSDE)) – The SDE to integrate.
  * **y0** (*ndarray* *of* *shape* *(**batch* *,* *y_dim* *)*) – The initial value of the SDE.
  * **t0** (*float*) – The initial time.
  * **t1** (*float*) – The final time.
  * **method** (*str*) – The integration method to use. Currently only “euler” is supported.
  * **n_steps** (*int*) – The number of steps to use for the integration.
  * **score_fn** (*callable*) – The score function for the reverse SDE. Needed only if the SDE is reversed (i.e. t1 < t0).
  * **n_samples** (*int*) – The number of samples to generate per input point.
  * **seed** (*int*) – Random seed.
