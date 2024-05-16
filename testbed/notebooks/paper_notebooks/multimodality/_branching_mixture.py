import numpy as np
from numpy import ndarray
from numpy.random import Generator

from typing import List
from typing import Callable
from typing import Optional
from jaxtyping import Float


class CustomRandomGenerator(Generator):
    def gaussian_mixture(
        self,
        locs: List[float],
        scales: List[float],
        weights: Optional[List[float]] = None,
        size: int = 1,
    ) -> Float[ndarray, "size"]:
        n_components = len(locs)

        if weights is None:
            weights = [1 / n_components] * n_components

        component = self.choice(n_components, size=size, p=weights)
        samples = self.normal(np.array(locs)[component], np.array(scales)[component])
        return samples

    def branching_mixture(
        self, x_splits=[0, 0.33, 0.66], x_max: float = 1.0, scale: float = 0.1, size: int = 1
    ):
        x = self.uniform(low=0, high=x_max, size=size).reshape(-1, 1)
        y = np.zeros_like(x)

        for i, x_i in enumerate(x):
            idx = np.searchsorted(x_splits, x_i, side="right").item()
            locs = [x_i] + [2 * split - x_i for split in x_splits[:idx]]
            scales = [scale] * len(locs)
            y[i] = self.gaussian_mixture(locs, scales)

        return x, y.reshape(-1)


def gaussian_density(x, loc, scale):
    return np.exp(-0.5 * ((x - loc) ** 2) / (scale**2)) / (np.sqrt(2 * np.pi) * scale)


def branching_mixture_density(
    x: float, x_splits=[0, 0.33, 0.66], x_max: float = 1.0, scale: float = 0.1, size: int = 1
) -> Callable:
    idx = np.searchsorted(x_splits, x, side="right").item()
    locs = [x] + [2 * split - x for split in x_splits[:idx]]
    scales = [scale] * len(locs)

    def density_fn(y):
        density = 0
        for loc, scale in zip(locs, scales):
            density += gaussian_density(y, loc, scale)
        return density / len(locs)

    return density_fn
