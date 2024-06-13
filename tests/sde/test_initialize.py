"""
Contains tests for sdes and solvers.
"""

from functools import partial

import numpy as np
import pytest
from jaxtyping import Float
from numpy import ndarray

from treeffuser.sde import get_diffusion_sde
from treeffuser.sde import sdeint


def _score_linear_vesde(
    y: Float[ndarray, "batch 1"],
    t: Float[ndarray, "batch 1"],
    x: Float[ndarray, "batch 1"],
    alpha: float,
    gamma: float,
    hyperparam_min: float,
    hyperparam_max: float,
) -> Float[ndarray, "batch 1"]:
    """
    This function computes the score under the data generating process:
        alpha ~ N(0, 1)
        for i in 1, ..., n
            x_i ~ N(0, 1)
            eps_i ~ N(0, gamma^2)
            y_i = alpha * x_i + eps_i *  x_i

    We assume that the diffusion model is a VESDE. The resulting perturbation kernel is Gaussian
    Song et al. (2021), Appendix B,  Eq. 29.

    Reference:
        Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).
        Score-based generative modeling through stochastic differential equations.
        ICLR 2021.
    """
    kernel_std = hyperparam_min * (hyperparam_max / hyperparam_min) ** t

    mean = alpha * x
    std = np.sqrt((gamma * x) ** 2 + kernel_std**2)

    score = (mean - y) / (std**2)
    return score


def _score_linear_vpsde(
    y: Float[ndarray, "batch 1"],
    t: Float[ndarray, "batch 1"],
    x: Float[ndarray, "batch 1"],
    alpha: float,
    gamma: float,
    hyperparam_min: float,
    hyperparam_max: float,
) -> Float[ndarray, "batch 1"]:
    """
    Same as _score_linear_vesde but for VPSDE.
    """
    hyperparam_integral = t * hyperparam_min + 0.5 * t**2 * (hyperparam_max - hyperparam_min)
    kernel_std = np.sqrt(1 - np.exp(-hyperparam_integral))

    mean = alpha * x * np.exp(-0.5 * hyperparam_integral)
    std = np.sqrt((gamma * x * np.exp(-0.5 * hyperparam_integral)) ** 2 + kernel_std**2)

    score = (mean - y) / (std**2)
    return score


def _score_linear_subvpsde(
    y: Float[ndarray, "batch 1"],
    t: Float[ndarray, "batch 1"],
    x: Float[ndarray, "batch 1"],
    alpha: float,
    gamma: float,
    hyperparam_min: float,
    hyperparam_max: float,
) -> Float[ndarray, "batch 1"]:
    """
    Same as _score_linear_vesde but for subVPSDE.
    """
    hyperparam_integral = t * hyperparam_min + 0.5 * t**2 * (hyperparam_max - hyperparam_min)
    kernel_std = 1 - np.exp(-hyperparam_integral)

    mean = alpha * x * np.exp(-0.5 * hyperparam_integral)
    std = np.sqrt((gamma * x * np.exp(-0.5 * hyperparam_integral)) ** 2 + kernel_std**2)

    score = (mean - y) / (std**2)
    return score


@pytest.mark.parametrize(
    ("sde_name", "score_fn"),
    [
        ("vesde", _score_linear_vesde),
        ("vpsde", _score_linear_vpsde),
        ("sub-vpsde", _score_linear_subvpsde),
    ],
)
def test_linear_sde(sde_name, score_fn):
    # Set seed
    seed = 0
    np.random.seed(seed)

    # Generate data
    n = 1000
    y_dim = 1
    alpha = np.random.normal(size=(y_dim, y_dim))
    gamma = 1
    x = np.random.normal(size=(n, y_dim))
    y = alpha * x + np.random.normal(size=(n, y_dim)) * gamma * x

    # Define SDE
    sde = get_diffusion_sde(sde_name)()
    sde.initialize_hyperparams_from_data(y)
    print(sde)
    hyperparams = sde.get_hyperparams()
    score_fn = partial(score_fn, **hyperparams, x=x, alpha=alpha, gamma=gamma)

    n_samples = 100
    y1 = sde.sample_from_theoretical_prior((n_samples, n, y_dim), seed=seed)
    y_samples = sdeint(
        sde,
        y1,
        score_fn=score_fn,
        t0=sde.T,
        t1=0,
        method="euler",
        seed=seed,
        n_steps=20,
    )  # shape: (n_samples, n_preds, y_dim)

    # Check that the samples are roughly correct
    true_mean = alpha * x
    true_std = gamma * np.abs(x)

    pred_mean = y_samples.mean(axis=0)
    pred_std = y_samples.std(axis=0)

    diff_mean = np.abs(pred_mean - true_mean)
    diff_std = np.abs(pred_std - true_std)

    assert diff_mean.mean() < 0.1, f"{sde_name}: diff_mean.mean() = {diff_mean.mean()}"
    assert diff_std.mean() < 0.1, f"{sde_name}: diff_std.mean() = {diff_std.mean()}"
