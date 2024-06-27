# treeffuser.sde.solvers module

### *class* treeffuser.sde.solvers.EulerMaruyama(sde: [BaseSDE](treeffuser.sde.base_sde.md#treeffuser.sde.base_sde.BaseSDE), n_steps: int, seed: int | None = None)

Bases: [`BaseSDESolver`](treeffuser.sde.base_solver.md#treeffuser.sde.base_solver.BaseSDESolver)

Euler-Maruyama solver for SDEs [1].

### References

[1] [https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)

#### step(y0: Float[ndarray, 'batch y_dim'], t0: float, t1: float)

Perform a single discrete step of the SDE solver from time t0 to time t1.

* **Parameters:**
  * **y0** – The value of the SDE at time t0.
  * **t0** (*float*) – The source time.
  * **t1** (*float*) – The target time.
