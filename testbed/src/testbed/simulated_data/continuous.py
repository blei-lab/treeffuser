"""
Contains two simple datasets with simulated data.

The coding could be improved but it is not important to
do so at the moment.

TODO: Make this prettier
"""

from typing import Tuple

import numpy as np
import torch as t
from jaxtyping import Float
from numpy import ndarray


class ContinuousDataset:
    """
    Abstract class for continuous datasets.

    See the code for further details.
    """

    def __init__(self):
        # Check x_dim, noise_scale, is_linear, is_heteroscedastic, seed
        # w  are define
        if not hasattr(self, "x_dim"):
            raise NotImplementedError
        if not hasattr(self, "noise_scale"):
            raise NotImplementedError
        if not hasattr(self, "is_linear"):
            raise NotImplementedError
        if not hasattr(self, "is_heteroscedastic"):
            raise NotImplementedError
        if not hasattr(self, "seed"):
            raise NotImplementedError

    def _get_mean(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        y = X @ self.w
        if not self.is_linear:
            y = np.sin(y**2)
        return y

    def _get_scale(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        if self.is_heteroscedastic:
            return np.abs(X @ self.w) * self.noise_scale
        else:
            return np.ones_like(self._get_mean(X)) * self.noise_scale

    def _get_noise(self, std: Float[ndarray, "... y_dim"]) -> Float[ndarray, "batch y_dim"]:
        raise NotImplementedError

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=100, seed=0
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        mean = self._get_mean(X)
        scale = self._get_scale(X)

        mean = np.expand_dims(mean, axis=0)
        mean = np.repeat(mean, n_samples, axis=0)

        scale = np.expand_dims(scale, axis=0)
        scale = np.repeat(scale, n_samples, axis=0)

        noise = self._get_noise(scale)
        y = mean + noise * scale
        return y

    def sample_dataset(
        self, n_samples: int, seed: int
    ) -> Tuple[Float[ndarray, "n x_dim"], Float[ndarray, "n y_dim"]]:

        rng = np.random.RandomState(seed)
        X = rng.randn(n_samples, self.x_dim)

        mean = self._get_mean(X)
        std = self._get_scale(X)
        noise = self._get_noise(std)

        y = mean + noise * std
        return X, y

    def score(
        self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "samples batch y_dim"]
    ) -> Float:
        mean = self._get_mean(X)  # (batch, y_dim)
        scale = self._get_scale(X)

        mean = t.Tensor(mean)
        scale = t.Tensor(scale)
        y = t.Tensor(y)

        distribution = self.Distribution(loc=mean, scale=scale)

        log_prob = distribution.log_prob(y).mean()
        return log_prob.item()

    @property
    def Distribution(
        self, loc: Float[ndarray, "batch y_dim"], scale: Float[ndarray, "batch y_dim"]
    ) -> t.distributions.Distribution:
        """
        Should return a class of the distibution that
        takes in parameters of the form loc, scale
        """
        raise NotImplementedError


class NormalDataset(ContinuousDataset):
    def __init__(
        self,
        x_dim: int = 10,
        noise_scale: Float = 1.0,
        is_linear=False,
        is_heteroscedastic=False,
        seed: int = 0,
    ):
        self.x_dim = x_dim
        self.noise_scale = noise_scale
        self.seed = seed
        self.is_linear = is_linear
        self.is_heteroscedastic = is_heteroscedastic

        rng = np.random.RandomState(seed)
        self.w = rng.randn(x_dim, 1) * 10

    def _get_noise(self, std: Float[ndarray, "... y_dim"]) -> Float[ndarray, "batch y_dim"]:
        rng = np.random.RandomState(self.seed)
        return rng.randn(*std.shape)

    def Distribution(
        self, loc: Float[ndarray, "batch y_dim"], scale: Float[ndarray, "batch y_dim"]
    ) -> t.distributions.Normal:
        return t.distributions.Normal(loc=loc, scale=scale)


class StudentTDataset(ContinuousDataset):
    def __init__(
        self,
        x_dim: int = 10,
        noise_scale=1,
        df: Float = 1.0,
        is_linear=False,
        is_heteroscedastic=False,
        seed: int = 0,
    ):
        self.x_dim = x_dim
        self.df = df
        self.seed = seed
        self.noise_scale = noise_scale
        self.is_linear = is_linear
        self.is_heteroscedastic = is_heteroscedastic

        rng = np.random.RandomState(seed)
        self.w = rng.randn(x_dim, 1) * 10

    def _get_noise(self, std: Float[ndarray, "... y_dim"]) -> Float[ndarray, "batch y_dim"]:
        rng = np.random.RandomState(self.seed)

        # noise is student t distributed
        return rng.standard_t(df=self.df, size=std.shape)

    def Distribution(
        self, loc: Float[ndarray, "batch y_dim"], scale: Float[ndarray, "batch y_dim"]
    ) -> t.distributions.StudentT:
        return t.distributions.StudentT(df=self.df, loc=loc, scale=scale)
