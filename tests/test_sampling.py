"""
Contains tests for _sampling.py module.
"""

from functools import partial

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from treeffuser._sampling import sample
from treeffuser._sdes import VESDE


def _linear_score_vesde(
    y: Float[ndarray, "batch 1"],
    x: Float[ndarray, "batch 1"],
    t: Float[ndarray, "batch 1"],
    beta: Float[ndarray, "1 1"],
    original_sigma: Float,
    sigma_min: Float,
    sigma_max: Float,
) -> Float[ndarray, "batch 1"]:
    """
    We assume that the data come from a linear model with process
        beta ~ N(0, 1)
        for i in 1, ..., n
            x_i ~ N(0, 1)
            eps_i ~ N(0, sigma^2)
            y_i = beta * x_i + eps_i *  x_i

    Furthermore, we assume that the score model is a VESDE
    which noises the data with kernel N(0, std= sigma_min * (sigma_max / sigma_min)^t).
    """
    mean = beta[0, 0] * x
    kernel_std = sigma_min * (sigma_max / sigma_min) ** t
    std = np.sqrt((original_sigma * x) ** 2 + kernel_std**2)

    # Compute the score
    score = (mean - y) / (std**2)
    return score


def test_vesde_linear_model():
    """
    We assume that the data come from a linear model with
    the following process:
        beta ~ N(0, 1)
        for i in 1, ..., n
            x_i ~ N(0, 1)
            eps_i ~ N(0, sigma^2)
            y_i = beta * x_i + eps_i *  x_i

    This test computes various statistics of the score model
    """
    np.random.seed(0)

    # Params
    n = 1000
    n_features = 1
    y_dim = 1

    # Generate data
    sigma = 1
    x = np.random.normal(size=(n, n_features))
    beta = np.random.normal(size=(1, n_features))

    mean = x * beta
    y = mean + np.random.normal(size=(n, n_features)) * sigma * x

    # Instantiate score function and SDE
    sigma_min = 0.01
    sigma_max = y.std() * 4

    score_fn = partial(
        _linear_score_vesde,
        beta=beta,
        original_sigma=sigma,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )

    sde = VESDE(sigma_min=sigma_min, sigma_max=sigma_max, N=100)

    # Draw samples
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
    )  # Shape: (n_preds, n_samples, y_dim)

    # Check that the samples are roughly correct
    pred_mean = y_samples.mean(axis=1)
    pred_std = y_samples.std(axis=1)

    effective_std = np.sqrt((sigma * x) ** 2 + sigma_min**2)
    diff_mean = np.abs(pred_mean - mean)
    diff_std = np.abs(pred_std - effective_std)

    assert diff_mean.mean() < 0.4, f"diff_mean.mean() = {diff_mean.mean()}"
    assert diff_std.mean() < 0.3, f"diff_std.mean() = {diff_std.mean()}"
