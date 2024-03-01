"""
Contains tests for _sampling.py.
"""

from functools import partial

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from treeffuser._sampling import sample
from treeffuser._sdes import VESDE
from treeffuser._sdes import VPSDE
from treeffuser._sdes import subVPSDE


def _score_linear_vesde(
    y: Float[ndarray, "batch 1"],
    x: Float[ndarray, "batch 1"],
    t: Float[ndarray, "batch 1"],
    alpha: Float[ndarray, "1 1"],
    gamma: Float[ndarray, "1 1"],
    sigma_min: Float[ndarray, "1 1"],
    sigma_max: Float[ndarray, "1 1"],
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
    kernel_std = sigma_min * (sigma_max / sigma_min) ** t

    mean = alpha * x
    std = np.sqrt((gamma * x) ** 2 + kernel_std**2)

    score = (mean - y) / (std**2)
    return score


def _score_linear_vpsde(
    y: Float[ndarray, "batch 1"],
    x: Float[ndarray, "batch 1"],
    t: Float[ndarray, "batch 1"],
    alpha: Float[ndarray, "1 1"],
    gamma: Float[ndarray, "1 1"],
    beta_min: Float[ndarray, "1 1"],
    beta_max: Float[ndarray, "1 1"],
) -> Float[ndarray, "batch 1"]:
    """
    Same as _score_linear_vesde but for VPSDE.
    """
    beta_integral = t * beta_min + 0.5 * t**2 * (beta_max - beta_min)
    kernel_std = np.sqrt(1 - np.exp(-beta_integral))

    mean = alpha * x * np.exp(-0.5 * beta_integral)
    std = np.sqrt((gamma * x * np.exp(-0.5 * beta_integral)) ** 2 + kernel_std**2)

    score = (mean - y) / (std**2)
    return score


def _score_linear_subvpsde(
    y: Float[ndarray, "batch 1"],
    x: Float[ndarray, "batch 1"],
    t: Float[ndarray, "batch 1"],
    alpha: Float[ndarray, "1 1"],
    gamma: Float[ndarray, "1 1"],
    beta_min: Float[ndarray, "1 1"],
    beta_max: Float[ndarray, "1 1"],
) -> Float[ndarray, "batch 1"]:
    """
    Same as _score_linear_vesde but for subVPSDE.
    """
    beta_integral = t * beta_min + 0.5 * t**2 * (beta_max - beta_min)
    kernel_std = 1 - np.exp(-beta_integral)

    mean = alpha * x * np.exp(-0.5 * beta_integral)
    std = np.sqrt((gamma * x * np.exp(-0.5 * beta_integral)) ** 2 + kernel_std**2)

    score = (mean - y) / (std**2)
    return score


def test_linear_vesde():
    """
    We assume that the data come from a linear model with data generating process:
        alpha ~ N(0, 1)
        for i in 1, ..., n
            x_i ~ N(0, 1)
            eps_i ~ N(0, gamma^2)
            y_i = alpha * x_i + eps_i *  x_i

    We assume that the diffusion model is either a VESDE.
    """
    # Set seed
    np.random.seed(0)

    # Generate data
    n = 1000
    n_features = 1
    y_dim = 1
    alpha = np.random.normal(size=(1, n_features))
    gamma = 1
    x = np.random.normal(size=(n, n_features))

    # Define SDE and score
    y = alpha * x + np.random.normal(size=(n, n_features)) * gamma * x
    sigma_min = 0.01
    sigma_max = y.std() + 4
    N = 100
    sde = VESDE(sigma_min, sigma_max, N)

    score_fn = partial(
        _score_linear_vesde,
        alpha=alpha,
        gamma=gamma,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )

    # Generate samples
    y_samples = sample(
        X=x,
        y_dim=y_dim,
        score_fn=score_fn,
        n_samples=100,
        n_parallel=100,
        sde=sde,
        predictor_name="euler_maruyama",
        corrector_name="none",
        denoise=False,
        seed=0,
        verbose=0,
    )  # shape: (n_preds, n_samples, y_dim)

    # Check that the samples are roughly correct
    true_mean = alpha * x
    true_std = gamma * np.abs(x)

    pred_mean = y_samples.mean(axis=1)
    pred_std = y_samples.std(axis=1)

    diff_mean = np.abs(pred_mean - true_mean)
    diff_std = np.abs(pred_std - true_std)

    assert diff_mean.mean() < 0.1, f"VESDE: diff_mean.mean() = {diff_mean.mean()}"
    assert diff_std.mean() < 0.1, f"VESDE: diff_std.mean() = {diff_std.mean()}"


def test_linear_vpsde():
    """
    Same as test_linear_vesde, but for VPSDE.
    """
    # Set seed
    np.random.seed(0)

    # Generate data
    n = 1000
    n_features = 1
    y_dim = 1
    alpha = np.random.normal(size=(1, n_features))
    gamma = 1
    x = np.random.normal(size=(n, n_features))

    # Define SDE and score
    beta_min = 0.01
    beta_max = 20
    N = 100
    sde = VPSDE(beta_min, beta_max, N)

    score_fn = partial(
        _score_linear_vpsde,
        alpha=alpha,
        gamma=gamma,
        beta_min=beta_min,
        beta_max=beta_max,
    )

    # Generate samples
    y_samples = sample(
        X=x,
        y_dim=y_dim,
        score_fn=score_fn,
        n_samples=100,
        n_parallel=100,
        sde=sde,
        predictor_name="euler_maruyama",
        corrector_name="none",
        denoise=False,
        seed=0,
        verbose=0,
    )  # shape: (n_preds, n_samples, y_dim)

    # Check that the samples are roughly correct
    true_mean = alpha * x
    true_std = gamma * np.abs(x)

    pred_mean = y_samples.mean(axis=1)
    pred_std = y_samples.std(axis=1)

    diff_mean = np.abs(pred_mean - true_mean)
    diff_std = np.abs(pred_std - true_std)

    assert diff_mean.mean() < 0.1, f"VPSDE: diff_mean.mean() = {diff_mean.mean()}"
    assert diff_std.mean() < 0.1, f"VPSDE: diff_std.mean() = {diff_std.mean()}"


def test_linear_subvpsde():
    """
    Same as test_linear_vesde, but for subVPSDE.
    """
    # Set seed
    np.random.seed(0)

    # Generate data
    n = 1000
    n_features = 1
    y_dim = 1
    alpha = np.random.normal(size=(1, n_features))
    gamma = 1
    x = np.random.normal(size=(n, n_features))

    # Define SDE and score
    beta_min = 0.01
    beta_max = 20
    N = 100
    sde = subVPSDE(beta_min, beta_max, N)

    score_fn = partial(
        _score_linear_subvpsde,
        alpha=alpha,
        gamma=gamma,
        beta_min=beta_min,
        beta_max=beta_max,
    )

    # Generate samples
    y_samples = sample(
        X=x,
        y_dim=y_dim,
        score_fn=score_fn,
        n_samples=100,
        n_parallel=100,
        sde=sde,
        predictor_name="euler_maruyama",
        corrector_name="none",
        denoise=False,
        seed=0,
        verbose=0,
    )  # shape: (n_preds, n_samples, y_dim)

    # Check that the samples are roughly correct
    true_mean = alpha * x
    true_std = gamma * np.abs(x)

    pred_mean = y_samples.mean(axis=1)
    pred_std = y_samples.std(axis=1)

    diff_mean = np.abs(pred_mean - true_mean)
    diff_std = np.abs(pred_std - true_std)

    assert diff_mean.mean() < 0.1, f"subVPSDE: diff_mean.mean() = {diff_mean.mean()}"
    assert diff_std.mean() < 0.1, f"subVPSDE: diff_std.mean() = {diff_std.mean()}"
