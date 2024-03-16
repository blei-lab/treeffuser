import warnings
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from jaxtyping import Float

from .base_sde import get_sde
from .parameter_schedule import LinearSchedule
from .sdes import VESDE
from .sdes import VPSDE
from .sdes import SubVPSDE


class ConvergenceWarning(Warning):
    # Indicates hyperparameter initialization didn't converge.
    pass


def initialize_sde(
    name: str,
    y0: Float[np.ndarray, "batch y_dim"],
    T: float = 1,
    KL_tol: Optional[float] = 10 ** (-5),
    verbose: bool = False,
) -> Union[VESDE, VPSDE, SubVPSDE]:
    """
    Initializes an SDE model based on the given name and initial data.

    For all SDEs, it sets `hyperparam_min` to 0.01. For VESDE, it sets `hyperparam_max`
    to the maxiumum pairwise distance in y0, following [1]. For VPSDE and Sub-VPSDE, it
    sets `hyperparam_max` to the smallest value that controls the KL divergence between
    the perturbation kernel at T and the theorethical prior.

    Parameters
    ----------
    name : str
        The SDE model to initialize ('vesde', 'vpsde', 'sub-vpsde').
    y0 : np.ndarray
        The data array with the training outcome.
    T : float, optional
        End time of the SDE, default is 1.
    KL_tol : float, optional
        KL divergence tolerance for initialization, default is 1e-5. This is only used
        for VPSDE and Sub-VPSDE.

    Returns
    -------
    An instance of the specified SDE model.

    References
    -------
        [1] Song, Y. and Ermon, S., 2020. Improved techniques for training score-based
        generative models. NeurIPS (2020).
    """
    hyperparams = None
    if name.lower() == "vesde":
        hyperparams = _initialize_vesde(y0)
    elif name.lower() == "vpsde":
        hyperparams = _initialize_vpsde(y0, T, KL_tol)
    elif name.lower() == "sub-vpsde":
        hyperparams = _initialize_subvpsde(y0, T, KL_tol)
    else:
        raise NotImplementedError

    sde = get_sde(name)(*hyperparams)
    if verbose:
        print(sde)
    return sde


def _initialize_vesde(y0: Float[np.ndarray, "batch y_dim"]) -> List[float, float]:
    hyperparam_min = 0.01
    if y0.shape[1] == 1:
        max_pairwise_difference = y0.max() - y0.min()
    else:
        y0_aug = y0[:, np.newaxis, :]
        max_pairwise_difference = np.max(np.sqrt(np.sum((y0_aug - y0) ** 2, axis=-1)))
    return [hyperparam_min, max_pairwise_difference]


def _initialize_vpsde(
    y0: Float[np.ndarray, "batch y_dim"], T: float = 1, KL_tol: float = 10 ** (-5)
) -> List[float, float]:
    hyperparam_min = 0.01
    y0_max = y0.max()

    def KL_helper(hyperparam_max):
        schedule = LinearSchedule(hyperparam_min, hyperparam_max)
        hyperparam_integral = schedule.get_integral(T)
        KL = _KL_univariate_gaussians(
            y0_max * np.exp(-0.5 * hyperparam_integral),
            (1 - np.exp(-hyperparam_integral)) ** 0.5,
            0,
            1,
        )
        return KL

    hyperparam_max_ini = 100
    hyperparam_max = _bisect(KL_helper, hyperparam_min, hyperparam_max_ini, KL_tol)

    if hyperparam_max == hyperparam_max_ini:
        warnings.warn(
            f"Hyperparameter initialization did not converge: setting `hyperparam_max` to {hyperparam_max_ini}.",
            ConvergenceWarning,
            stacklevel=1,
        )

    return [hyperparam_min, hyperparam_max]


def _initialize_subvpsde(
    y0: Float[np.ndarray, "batch y_dim"], T: float = 1, KL_tol: float = 10 ** (-5)
) -> List[float, float]:
    hyperparam_min = 0.01
    y0_max = y0.max()

    def KL_helper(hyperparam_max):
        schedule = LinearSchedule(hyperparam_min, hyperparam_max)
        hyperparam_integral = schedule.get_integral(T)
        KL = _KL_univariate_gaussians(
            y0_max * np.exp(-0.5 * hyperparam_integral), 1 - np.exp(-hyperparam_integral), 0, 1
        )
        return KL

    hyperparam_max_ini = 100
    hyperparam_max = _bisect(KL_helper, hyperparam_min, hyperparam_max_ini, KL_tol)

    if hyperparam_max == hyperparam_max_ini:
        warnings.warn(
            f"Hyperparameter initialization did not converge: setting `hyperparam_max` to {hyperparam_max_ini}.",
            ConvergenceWarning,
            stacklevel=1,
        )

    return [hyperparam_min, hyperparam_max]


def _KL_univariate_gaussians(
    loc_1: float, scale_1: float, loc_2: float = 0, scale_2: float = 1
) -> float:
    """
    Computes the KL divergence between two univariate Gaussians.

    The KL divergence between two Gaussians `N(loc_1, scale_1)` and `N(loc_2, scale_2)` is:
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
    Return an x such that |f(x)| <= tol and |f(x - tol)| > tol.

    It assumes that f is continuous and increasing in [a, b], and that |f(a)| > tol and |f(b)| <= tol.
    """
    for _ in range(max_iter):
        x = (a + b) / 2
        if np.abs(f(x)) > tol:
            a = x
        elif np.abs(f(x - x_tol)) > tol:
            return x
        else:
            b = x
    return x
