"""
Contains tests for _sampling.py.
"""

from functools import partial

import numpy as np
from jaxtyping import Float
from ml_collections import ConfigDict
from numpy import ndarray

from treeffuser._sampling import sample
from treeffuser._sdes import get_sde


def _score_linear(
    y: Float[ndarray, "batch 1"],
    x: Float[ndarray, "batch 1"],
    t: Float[ndarray, "batch 1"],
    gamma: Float[ndarray, "1 1"],
    sigma: Float[ndarray, "1 1"],
    sde_name: str,
    sde_params: ConfigDict,
) -> Float[ndarray, "batch 1"]:
    """
    This function returns the score of the perturbation kernel under the data generating process:
        gamma ~ N(0, 1)
        for i in 1, ..., n
            x_i ~ N(0, 1)
            eps_i ~ N(0, sigma^2)
            y_i = gamma * x_i + eps_i *  x_i

    We assume that the diffusion model is either a VESDE, VPSDE, or subVPSDE. For these models,
    the perturbation kernels are Gaussian. Their mean and variances can be found in
    Song et al. (2021), Appendix B,  Eq. 29.

    Reference:
        Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).
        Score-based generative modeling through stochastic differential equations.
        ICLR 2021.
    """
    # Compute mean and std for each diffusion model
    if sde_name.lower() == "vesde":
        mean = gamma * x
        kernel_std = sde_params.sigma_min * (sde_params.sigma_max / sde_params.sigma_min) ** t
        std = np.sqrt((sigma * x) ** 2 + kernel_std**2)
    else:
        beta_integral = t * sde_params.beta_min + 0.5 * t**2 * (
            sde_params.beta_max - sde_params.beta_min
        )
        mean = gamma * x * np.exp(-0.5 * beta_integral)
        if sde_name.lower() == "vpsde":
            kernel_std = np.sqrt(1 - np.exp(-beta_integral))
        elif sde_name.lower() == "subvpsde":
            kernel_std = 1 - np.exp(-beta_integral)
        std = np.sqrt((sigma * x * np.exp(-0.5 * beta_integral)) ** 2 + kernel_std**2)

    # Compute the score
    score = (mean - y) / (std**2)
    return score


def test_vesde_linear_model():
    """
    We assume that the data come from a linear model with data generating process:
        gamma ~ N(0, 1)
        for i in 1, ..., n
            x_i ~ N(0, 1)
            eps_i ~ N(0, sigma^2)
            y_i = gamma * x_i + eps_i *  x_i

    We assume that the diffusion model is either a VESDE, VPSDE, or subVPSDE.
    """
    # Set seed
    np.random.seed(42)

    # Generate data
    n = 1000
    n_features = 1
    y_dim = 1
    sigma = 1
    x = np.random.normal(size=(n, n_features))
    gamma = np.random.normal(size=(1, n_features))

    # Define parameters for SDEs
    sde_params = ConfigDict()
    sde_params.vesde = vesde = ConfigDict()
    sde_params.vpsde = vpsde = ConfigDict()
    sde_params.subvpsde = subvpsde = ConfigDict()

    y = gamma * x + np.random.normal(size=(n, n_features)) * sigma * x

    vesde.sigma_min, vesde.sigma_max, vesde.N = 0.01, y.std() + 4, 100
    vpsde.beta_min, vpsde.beta_max, vpsde.N = 0.01, 20, 100
    subvpsde.beta_min, subvpsde.beta_max, subvpsde.N = 0.01, 20, 100

    # Run test for each diffusion model
    for sde_name in ["vesde", "subvpsde", "vpsde"]:
        score_fn = partial(
            _score_linear,
            gamma=gamma,
            sigma=sigma,
            sde_name=sde_name,
            sde_params=sde_params[sde_name],
        )

        sde = get_sde(sde_name)(**sde_params[sde_name])

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
        true_mean = gamma * x
        true_std = sigma * x

        pred_mean = y_samples.mean(axis=1)
        pred_std = y_samples.std(axis=1)

        diff_mean = np.abs(pred_mean - true_mean)
        diff_std = np.abs(pred_std - true_std)

        assert diff_mean.mean() < 0.1, f"{sde_name} diff_mean.mean() = {diff_mean.mean()}"
        assert diff_std.mean() < 1, f"{sde_name} diff_std.mean() = {diff_std.mean()}"
