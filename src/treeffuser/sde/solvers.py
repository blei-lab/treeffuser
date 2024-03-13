import numpy as np
from jaxtyping import Float

from treeffuser.sde.base_solver import BaseSDESolver
from treeffuser.sde.base_solver import _register_solver


@_register_solver(name="euler")
class EulerMaruyama(BaseSDESolver):
    """
    Euler-Maruyama solver for SDEs [1].

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
    """

    def step(
        self, y0: Float[np.ndarray, "batch y_dim"], t0: float, t1: float
    ) -> Float[np.ndarray, "batch y_dim"]:
        dt = t1 - t0
        t0 = np.broadcast_to(t0, (y0.shape[:-1] + (1,)))
        drift, diffusion = self.sde.drift_and_diffusion(y0, t0)
        dW = self._rng.normal(size=y0.shape)
        return y0 + drift * dt + diffusion * np.sqrt(dt) * dW
