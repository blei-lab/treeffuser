# treeffuser.sde.diffusion_sdes module

### *class* treeffuser.sde.diffusion_sdes.DiffusionSDE

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

### *class* treeffuser.sde.diffusion_sdes.SubVPSDE(hyperparam_min=0.01, hyperparam_max=20)

Bases: [`DiffusionSDE`](#treeffuser.sde.diffusion_sdes.DiffusionSDE)

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

### *class* treeffuser.sde.diffusion_sdes.VESDE(hyperparam_min=0.01, hyperparam_max=20)

Bases: [`DiffusionSDE`](#treeffuser.sde.diffusion_sdes.DiffusionSDE)

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

### *class* treeffuser.sde.diffusion_sdes.VPSDE(hyperparam_min=0.01, hyperparam_max=20)

Bases: [`DiffusionSDE`](#treeffuser.sde.diffusion_sdes.DiffusionSDE)

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

### treeffuser.sde.diffusion_sdes.get_diffusion_sde(name: str | None = None)

Function to retrieve a registered diffusion SDE by its name.

## Parameters:

name: str
: The name of the SDE to retrieve. If None, returns a dictionary of all available SDEs.

## Returns:

BaseSDE or dict: The SDE class corresponding to the given name, or a dictionary of all available SDEs.

## Raises:

ValueError: If the given name is not registered.

## Examples:

```pycon
>>> sde_class = get_diffusion_sde("my_sde")
>>> sde_instance = sde_class()
```
