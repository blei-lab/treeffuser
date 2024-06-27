# treeffuser.sde.base_sde module

### *class* treeffuser.sde.base_sde.BaseSDE

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

### *class* treeffuser.sde.base_sde.CustomSDE(drift_fn: Callable[[Float[ndarray, 'batch y_dim'], Float[ndarray, 'batch 1']], Float[ndarray, 'batch y_dim']], diffusion_fn: Callable[[Float[ndarray, 'batch y_dim'], Float[ndarray, 'batch 1']], Float[ndarray, 'batch y_dim']])

Bases: [`BaseSDE`](#treeffuser.sde.base_sde.BaseSDE)

SDE defined by a custom drift and diffusion functions.

## Parameters:

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

### *class* treeffuser.sde.base_sde.ReverseSDE(sde: [BaseSDE](#treeffuser.sde.base_sde.BaseSDE), t_reverse_origin: float, score_fn: Callable[[Float[ndarray, 'batch y_dim'], Float[ndarray, 'batch']], Float[ndarray, 'batch']])

Bases: [`BaseSDE`](#treeffuser.sde.base_sde.BaseSDE)

The ReverseSDE class represents a stochastic differential equation (SDE) reversed
in time.

An SDE requires a transformation of the drift term to be reversed, which is based on
the score function of the marginal distributions induced by the original SDE [1].
The original SDE dY = f(Y, t) dt + g(Y, t) dW can be reversed from time T to
define a new SDE:
dY(T-t) = (-f(Y, T-t) + g(Y, T-t)² ∇[log p(Y(T-t))]) dt + g(Y, T-t) dW.

* **Parameters:**
  * **sde** ([*BaseSDE*](#treeffuser.sde.base_sde.BaseSDE)) – The original SDE.
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
