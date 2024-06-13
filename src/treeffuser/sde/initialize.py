import warnings
from typing import Callable
from typing import Tuple

import numpy as np
from jaxtyping import Float

from .parameter_schedule import LinearSchedule


class ConvergenceWarning(Warning):
    # Indicates hyperparameter initialization didn't converge.
    pass


def initialize_vesde(y0: Float[np.ndarray, "batch y_dim"]) -> Tuple[float, float]:
    hyperparam_min = 0.01
    if y0.shape[1] == 1:
        max_pairwise_difference = y0.max() - y0.min()
    else:
        y0_aug = y0[:, np.newaxis, :]
        max_pairwise_difference = np.max(np.sqrt(np.sum((y0_aug - y0) ** 2, axis=-1)))
    return hyperparam_min, max_pairwise_difference


def initialize_vpsde(
    y0: Float[np.ndarray, "batch y_dim"], T: float = 1, kl_tol: float = 10 ** (-5)
) -> Tuple[float, float]:
    hyperparam_min = 0.01
    y0_max = y0.max()

    def kl_helper(hyperparam_max_local):
        schedule = LinearSchedule(hyperparam_min, hyperparam_max_local)
        hyperparam_integral = schedule.get_integral(T)
        kl = _kl_univariate_gaussians(
            y0_max * np.exp(-0.5 * hyperparam_integral),
            (1 - np.exp(-hyperparam_integral)) ** 0.5,
            0,
            1,
        )
        return kl

    hyperparam_max_ini = 100
    hyperparam_max = _bisect(kl_helper, hyperparam_min, hyperparam_max_ini, kl_tol)

    if hyperparam_max == hyperparam_max_ini:
        warnings.warn(
            f"Hyperparameter initialization did not converge: setting `hyperparam_max` to {hyperparam_max_ini}.",
            ConvergenceWarning,
            stacklevel=1,
        )

    return hyperparam_min, hyperparam_max


def initialize_subvpsde(
    y0: Float[np.ndarray, "batch y_dim"], T: float = 1, kl_tol: float = 10 ** (-5)
) -> Tuple[float, float]:
    hyperparam_min = 0.01
    y0_max = y0.max()

    def kl_helper(hyperparam_max_local):
        schedule = LinearSchedule(hyperparam_min, hyperparam_max_local)
        hyperparam_integral = schedule.get_integral(T)
        kl = _kl_univariate_gaussians(
            y0_max * np.exp(-0.5 * hyperparam_integral), 1 - np.exp(-hyperparam_integral), 0, 1
        )
        return kl

    hyperparam_max_ini = 100
    hyperparam_max = _bisect(kl_helper, hyperparam_min, hyperparam_max_ini, kl_tol)

    if hyperparam_max == hyperparam_max_ini:
        warnings.warn(
            f"Hyperparameter initialization did not converge: setting `hyperparam_max` to {hyperparam_max_ini}.",
            ConvergenceWarning,
            stacklevel=1,
        )

    return hyperparam_min, hyperparam_max


def _kl_univariate_gaussians(
    loc_1: float, scale_1: float, loc_2: float = 0, scale_2: float = 1
) -> float:
    """
    Computes the kl divergence between two univariate Gaussians.

    The kl divergence between two Gaussians `N(loc_1, scale_1)` and `N(loc_2, scale_2)` is:
    `log (scale_2 / scale_1) + (scale_1^2 + (loc_1 - loc_2) ^ 2) / (2 * scale_2^2) - .5`.
    """
    return (
        np.log(scale_2 / scale_1)
        + (scale_1**2 + (loc_1 - loc_2) ** 2) / (2 * scale_2**2)
        - 1 / 2
    )


def _bisect(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-5,
    x_tol: float = 1e-5,
    max_iter: int = 1000,
) -> float:
    """
    Return an x such that |f(x)| <= tol and |f(x - x_tol)| > tol.

    It assumes that:
    - f is continuous and decreasing in [a, b]
    - |f(a)| > tol and |f(b)| <= tol.
    """
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than 0.")
    x = 0
    for _ in range(max_iter):
        x = (a + b) / 2
        if np.abs(f(x)) > tol:
            a = x
        elif np.abs(f(x - x_tol)) > tol:
            return x
        else:
            b = x
    return x
