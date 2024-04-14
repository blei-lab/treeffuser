import abc
from typing import Callable

from jaxtyping import Float
from numpy import ndarray

_AVAILABLE_SDES = {}


class BaseSDE(abc.ABC):
    """
    This abstract class represents a stochastic differential equation (SDE) of the form
    `dY = f(Y, t) dt + g(Y, t) dW`, where:
    - `Y` is the variable of the SDE
    - `t` is time
    - `f` is the drift function
    - `g` is the diffusion function

    Any class that inherits from `BaseSDE` must implement the `drift_and_diffusion(y, t)` method,
    which returns a tuple containing the drift and the diffusion at time `t` for a given state `Y=y`.

    References:
        [1] https://en.wikipedia.org/wiki/Stochastic_differential_equation
    """

    @abc.abstractmethod
    def drift_and_diffusion(
        self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    ) -> tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
        """
        Computes the drift and diffusion at a given time `t` for a given state `Y=y`.

        Args:
            y (Float[ndarray, "batch y_dim"]): The state of the SDE.
            t (Float[ndarray, "batch 1"]): The time at which to compute the drift and diffusion.

        Returns:
            tuple: A tuple containing the drift and the diffusion at time `t` for a given state `Y=y`.
        """

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ReverseSDE(BaseSDE):
    """
    The `ReverseSDE` class represents a stochastic differential equation (SDE) reversed
    in time.

    An SDE requires a transformation of the drift term to be reversed, which is based on
    the score function of the marginal distributions induced by the original SDE [1].
    The original SDE `dY = f(Y, t) dt + g(Y, t) dW` can be reversed from time `T` to
    define a new SDE:
    `dY(T-t) = (-f(Y, T-t) + g(Y, T-t)² ∇[log p(Y(T-t))]) dt + g(Y, T-t) dW`.

    Args:
        sde (BaseSDE): The original SDE.
        t_reverse_origin (float): The time from which to reverse the SDE.
        score_fn: The score function of the original SDE induced marginal distributions.

    References:
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
    ) -> tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
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
    ) -> tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
        return self.drift_fn(y, t), self.diffusion_fn(y, t)

    def __repr__(self):
        return f"CustomSDE(drift_fn={self.drift_fn}, diffusion_fn={self.diffusion_fn})"


def _register_sde(name: str):
    """
    A decorator for registering available SDEs and making them accessible by name,
    with the `get_sde` function.

    Args:
        name (str): Name of the SDE.

    Examples:
        >>> @_register_sde(name="my_sde")
        ... class MySDE(BaseSDE):
        ...     def drift_and_diffusion(self, y, t):
        ...         ...
        >>> sde_cls = get_sde("my_sde")
        >>> sde_instance = sde_cls()

    See Also:
        get_sde: Function to get an SDE by name.
    """

    def _register(cls):
        if name in _AVAILABLE_SDES:
            raise ValueError(f"Already registered SDE with name: {name}")
        _AVAILABLE_SDES[name] = cls
        return cls

    return _register


def get_sde(name: str):
    """
    Function to retrieve a registered SDE by its name.

    Args:
        name (str): The name of the SDE.

    Raises:
        ValueError: If the SDE with the given name is not registered.

    Returns:
        The class of the registered SDE.

    Examples:
        >>> sde_class = get_sde("my_sde")
        >>> sde_instance = sde_class()
    """
    if name not in _AVAILABLE_SDES:
        raise ValueError(f"Unknown SDE {name}. Available SDEs: {list(_AVAILABLE_SDES.keys())}")
    return _AVAILABLE_SDES[name]
