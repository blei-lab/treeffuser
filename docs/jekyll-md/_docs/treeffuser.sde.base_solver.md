# treeffuser.sde.base_solver module

### *class* treeffuser.sde.base_solver.BaseSDESolver(sde: [BaseSDE](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.BaseSDE), n_steps: int, seed: int | None = None)

Bases: `ABC`

Abstract class representing a solver for stochastic differential equations (SDEs).

* **Parameters:**
  * **sde** ([*BaseSDE*](treeffuser.sde.md#treeffuser.sde.BaseSDE)) – The SDE to solve.
  * **n_steps** (*int*) – The number of steps to use for the integration.
  * **seed** (*int*) – Random seed.

#### integrate(y0: Float[ndarray, 'batch y_dim'], t0: float, t1: float)

Integrate the SDE from time t0 to time t1 using self.n_steps steps.

* **Parameters:**
  * **y0** – The value of the SDE at time t0.
  * **t0** (*float*) – The initial time.
  * **t1** (*float*) – The final time.

#### *abstract* step(y0: Float[ndarray, 'batch y_dim'], t0: float, t1: float)

Perform a single discrete step of the SDE solver from time t0 to time t1.

* **Parameters:**
  * **y0** – The value of the SDE at time t0.
  * **t0** (*float*) – The source time.
  * **t1** (*float*) – The target time.

### treeffuser.sde.base_solver.get_solver(name)

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

### treeffuser.sde.base_solver.sdeint(sde, y0, t0=0.0, t1=1.0, method='euler', n_steps=20, score_fn=None, n_samples=1, seed=None)

Integrate an SDE (i.e. sample from an SDE).

* **Parameters:**
  * **sde** ([*BaseSDE*](treeffuser.sde.md#treeffuser.sde.BaseSDE)) – The SDE to integrate.
  * **y0** (*ndarray* *of* *shape* *(**batch* *,* *y_dim* *)*) – The initial value of the SDE.
  * **t0** (*float*) – The initial time.
  * **t1** (*float*) – The final time.
  * **method** (*str*) – The integration method to use. Currently only “euler” is supported.
  * **n_steps** (*int*) – The number of steps to use for the integration.
  * **score_fn** (*callable*) – The score function for the reverse SDE. Needed only if the SDE is reversed (i.e. t1 < t0).
  * **n_samples** (*int*) – The number of samples to generate per input point.
  * **seed** (*int*) – Random seed.
