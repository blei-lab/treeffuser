from typing import Tuple

import numpy as np
import torch as t
from jaxtyping import Float
from numpy import ndarray


class NormalLinearDataset:
    def __init__(self, d: int = 10, noise_std: Float = 1.0, seed: int = 0):
        self.d = d
        self.noise_std = noise_std
        self.seed = seed

        rng = np.random.RandomState(seed)
        self.w = rng.randn(d, 1)

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=100, seed=0
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        rng = np.random.RandomState(seed)
        y = X @ self.w  # (batch, y_dim)

        # add extra coordinate (n_samples, batch, y_dim)
        y = np.expand_dims(y, axis=0)
        y = np.repeat(y, n_samples, axis=0)
        y += rng.randn(*y.shape) * self.noise_std
        return y

    def sample_dataset(
        self, n_samples: int, seed: int
    ) -> Tuple[Float[ndarray, "n x_dim"], Float[ndarray, "n y_dim"]]:
        rng = np.random.RandomState(seed)
        X = rng.randn(n_samples, self.d)
        y = X @ self.w + rng.randn(n_samples, 1) * self.noise_std
        return X, y

    def score(
        self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "samples batch y_dim"]
    ) -> Float:
        mean = t.Tensor(X @ self.w)  # (batch, y_dim)
        std = t.ones_like(mean) * self.noise_std
        y = t.Tensor(y)

        dist = t.distributions.Normal(mean, std)
        log_prob = dist.log_prob(y).mean()
        return log_prob.item()
