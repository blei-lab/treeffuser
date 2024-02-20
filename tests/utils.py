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
        y = y * (X @ np.random.normal(size=p)) + (1 - y) * (X @ np.random.normal(size=p))
    else:
        y = X @ np.random.normal(size=p)
    y += np.random.normal(scale=sigma, size=n)
    y = y[:, None]
    return X, y
