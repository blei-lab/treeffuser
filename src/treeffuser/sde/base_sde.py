import abc
from typing import Callable

from jaxtyping import Float
from numpy import ndarray

_AVAILABLE_SDES = {}


def _register_sde(name):
    """A decorator for registering available SDE."""

    def _register(cls):
        if name in _AVAILABLE_SDES:
            raise ValueError(f"Already registered SDE with name: {name}")
        _AVAILABLE_SDES[name] = cls
        return cls

    return _register


def get_sde(name):
    """Get an SDE by name."""
    if name not in _AVAILABLE_SDES:
        raise ValueError(f"Unknown SDE {name}. Available SDEs: {list(_AVAILABLE_SDES.keys())}")
    return _AVAILABLE_SDES[name]


class BaseSDE(abc.ABC):
    """
    Abstract class representing a stochastic differential equation (SDE):
    `dY = f(Y, t) dt + g(Y, t) dW` where `Y` is the state, `t` is time,
    `f` is the drift and `g` is the diffusion [1].

    BaseSDE must implement:
    - `drift_and_diffusion(y, t)`, which return a tuple of the drift and the
    diffusion at time `t` when `Y=y`.

    References:
    -----------
    [1] https://en.wikipedia.org/wiki/Stochastic_differential_equation
    """

    @abc.abstractmethod
    def drift_and_diffusion(
        self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    ) -> (Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]):
        """
        Drift and diffusion of the SDE at time `t` and value `y`.
        """

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ReverseSDE(BaseSDE):
    """
    ReverseSDE, is the SDE reversed in time. To be reversed, an SDE requires a
    transformation of the drift term, based on the score function of the marginal
    distributions induced by the original SDE [1].

    The original SDE `dY = f(Y, t) dt + g(Y, t) dW` can be reversed from
    time `T` to define a new SDE:
    `dY(T-t) = (-f(Y, T-t) + g(Y, T-t)² ∇[log p(Y(T-t))]) dt + g(Y, T-t) dW`

    Parameters:
    -----------
    sde : BaseSDE
        The original SDE.
    t_reverse_origin : float
        The time from which to reverse the SDE.
    score_fn : Callable[[Float[ndarray, "batch y_dim"], Float[ndarray, "batch"]], Float[ndarray, "batch"]]
        The score function of the original SDE induced marginal distributions.

    References:
    -----------
    [1] https://openreview.net/pdf?id=PxTIG12RRHS

    """

    def __init__(
        self,
        sde: BaseSDE,
        t_reverse_origin: float,
        score_fn: Callable[
            [Float[ndarray, "batch y_dim"], Float[ndarray, "batch"]], Float[ndarray, "batch"]
        ],
    ):
        self.sde = sde
        self.t_reverse_origin = t_reverse_origin
        self.score_fn = score_fn

    def drift_and_diffusion(
        self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    ) -> (Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]):
        drift, diffusion = self.sde.drift_and_diffusion(y, self.t_reverse_origin - t)
        drift = -drift + diffusion**2 * self.score_fn(y, self.t_reverse_origin - t)
        return drift, diffusion

    def __repr__(self):
        return f"ReverseSDE(sde={self.sde}, t_origin={self.t_reverse_origin}, score_fn={self.score_fn})"


class CustomSDE(BaseSDE):
    """
    SDE defined by a custom drift and diffusion functions.

    Parameters:
    -----------
    drift_fn : Callable[[Float[ndarray, "batch y_dim"], Float[ndarray, "batch 1"]], Float[ndarray, "batch y_dim"]]
        Drift function of the SDE.
    diffusion_fn : Callable[[Float[ndarray, "batch y_dim"], Float[ndarray, "batch 1"]], Float[ndarray, "batch y_dim"]]
        Diffusion function of the SDE.

    """

    def __init__(
        self,
        drift_fn: Callable[
            [Float[ndarray, "batch y_dim"], Float[ndarray, "batch 1"]],
            Float[ndarray, "batch y_dim"],
        ],
        diffusion_fn: Callable[
            [Float[ndarray, "batch y_dim"], Float[ndarray, "batch 1"]],
            Float[ndarray, "batch y_dim"],
        ],
    ):
        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn

    def drift_and_diffusion(
        self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    ) -> (Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]):
        return self.drift_fn(y, t), self.diffusion_fn(y, t)

    def __repr__(self):
        return f"CustomSDE(drift_fn={self.drift_fn}, diffusion_fn={self.diffusion_fn})"
