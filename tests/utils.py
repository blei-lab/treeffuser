"""
This file doesn't actually contain any tests. It just contains various utilities.
"""

from typing import Tuple

import numpy as np
from jaxtyping import Float
from numpy import ndarray


def generate_bimodal_linear_regression_data(
    n: int, p: int, sigma: float, bimodal: bool, seed=None
) -> Tuple[Float[ndarray, "n p"], Float[ndarray, "n"]]:
    """
    Generate a dataset for linear regression with a bimodal distribution for the response variable.

    Args:
    n: The number of observations.
    p: The number of features.
    sigma: The standard deviation of the noise.
    bimodal: Whether the response variable should have a bimodal distribution.
    seed: The random seed to use.

    Returns:
    A tuple containing the features and the response variable.
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.normal(size=(n, p))
    if bimodal:
        y = np.random.choice([0, 1], size=n)
        y = y * (X @ np.random.normal(size=p) * 2) + (1 - y) * (
            X @ np.random.normal(size=p) * 2
        )
    else:
        y = X @ np.random.normal(size=p)
    y += np.random.normal(size=n) * sigma
    y = y[:, None]
    return X, y


def gaussian_pdf(
    x: Float[np.ndarray, "batch_size 1"], loc: float, scale: float, log=False
) -> Float[np.ndarray, "batch_size 1"]:
    log_density = -0.5 * np.log(2 * np.pi * scale**2) - (x - loc) ** 2 / (2 * scale**2)
    return log_density if log else np.exp(log_density)


def gaussian_mixture_pdf(
    x: Float[np.ndarray, "batch_size 1"],
    loc1: float,
    scale1: float,
    loc2: float,
    scale2: float,
    weight1: float,
    log=False,
) -> Float[np.ndarray, "batch_size 1"]:
    density = weight1 * gaussian_pdf(x, loc1, scale1, log=False)
    density += (1 - weight1) * gaussian_pdf(x, loc2, scale2, log=False)
    return np.log(density) if log else density
