import abc
from typing import Optional

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from .base_sde import BaseSDE
from .base_sde import ReverseSDE

_AVAILABLE_SOLVERS = {}


def sdeint(
    sde,
    y0,
    t0=0.0,
    t1=1.0,
    method="euler",
    n_steps=20,
    score_fn=None,
    n_samples=1,
    seed=None,
):
    """
    Integrate an SDE (i.e. sample from an SDE).

    Parameters
    ----------
    sde : BaseSDE
        The SDE to integrate.
    y0 : ndarray of shape (batch, y_dim)
        The initial value of the SDE.
    t0 : float
        The initial time.
    t1 : float
        The final time.
    method : str
        The integration method to use. Currently only "euler" is supported.
    n_steps : int
        The number of steps to use for the integration.
    score_fn : callable
        The score function for the reverse SDE. Needed only if the SDE is reversed (i.e. t1 < t0).
    n_samples : int
        The number of samples to generate per input point.
    seed : int
        Random seed.
    """
    if t1 < t0:
        # Reverse SDE
        if score_fn is None:
            raise ValueError(
                "`score_fn` must be provided for reverse SDE (the SDE is reversed "
                f"because `t1` is smaller than `t0`: t0={t0}, t1={t1})."
            )
        sde = ReverseSDE(sde, t0, score_fn)
        t0, t1 = 0.0, t0 - t1
    if n_samples > 1:
        y0 = np.broadcast_to(y0, (n_samples, *y0.shape))
    solver = get_solver(method)(sde=sde, n_steps=n_steps, seed=seed)
    return solver.integrate(y0, t0, t1)


class BaseSDESolver(abc.ABC):
    """
    Abstract class representing a solver for stochastic differential equations (SDEs).

    Parameters
    ----------
    sde : BaseSDE
        The SDE to solve.
    n_steps : int
        The number of steps to use for the integration.
    seed : int
        Random seed.
    """

    def __init__(self, sde: BaseSDE, n_steps: int, seed: Optional[int] = None):
        self.sde = sde
        self.n_steps = n_steps
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def step(
        self, y0: Float[ndarray, "batch y_dim"], t0: float, t1: float
    ) -> Float[ndarray, "batch y_dim"]:
        """
        Perform a single discrete step of the SDE solver from time t0 to time t1.

        Parameters
        ----------
        y0 :
            The value of the SDE at time t0.
        t0 : float
            The source time.
        t1 : float
            The target time.
        """
        raise NotImplementedError

    def integrate(
        self, y0: Float[ndarray, "batch y_dim"], t0: float, t1: float
    ) -> Float[ndarray, "batch y_dim"]:
        """
        Integrate the SDE from time t0 to time t1 using `self.n_steps` steps.

        Parameters
        ----------
        y0 :
            The value of the SDE at time t0.
        t0 : float
            The initial time.
        t1 : float
            The final time.
        """
        assert t1 > t0, "t1 must be greater than t0"
        dt = (t1 - t0) / self.n_steps
        t, y = t0, y0
        for _ in range(self.n_steps):
            y = self.step(y, t, t + dt)
            t += dt
        return y


def _register_solver(name):
    """
    A decorator for registering available solvers and making them accessible by name,
    with the `get_solver` function.

    Args:
        name (str): Name of the solver.

    Examples:
        >>> @_register_solver(name="my_solver")
        ... class MySolver(BaseSDESolver):
        ...     def step(self, y0, t0, t1):
        ...         ...
        ...
        >>> solver_cls = get_solver("my_solver")
        >>> solver_instance = solver_cls()

    See Also:
        get_solver: Function to get a solver by name.
    """

    def _register(cls):
        if name in _AVAILABLE_SOLVERS:
            raise ValueError(f"Already registered solver with name: {name}")
        _AVAILABLE_SOLVERS[name] = cls
        return cls

    return _register


def get_solver(name):
    """
    Function to retrieve a registered solver by its name.

    Args:
        name (str): The name of the solver.

    Raises:
        ValueError: If the solver with the given name is not registered.

    Returns:
        The class of the registered solver.

    Examples:
        >>> solver_class = get_solver("my_solver")
        >>> solver_instance = solver_class()
    """
    if name not in _AVAILABLE_SOLVERS:
        raise ValueError(
            f"Unknown solver {name}. Available solvers: {list(_AVAILABLE_SOLVERS.keys())}"
        )
    return _AVAILABLE_SOLVERS[name]
