"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.
Adapted from: http://tinyurl.com/torch-sde-lib-song

The preferred way to use this module is to use the `get_sde` function to
get the SDE class by name. For example:

```python
sde = get_sde("vesde")(sigma_min=0.01, sigma_max=50, N=1000)
```

The notice from the original code is as follows:
 coding=utf-8
 Copyright 2020 The Google Research Authors.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import abc
from typing import Tuple

import numpy as np
from einops import repeat
from jaxtyping import Float
from numpy import ndarray

_SDES = {}


def _register_sde(cls=None, *, name=None):
    """
    A simple decorator to register SDE classes.
    """

    def _register(cls):
        if name is None:
            local_name = cls.__name__.lower()
        else:
            local_name = name
        if local_name in _SDES:
            raise ValueError(f"Duplicate SDE name: {local_name}")

        _SDES[local_name] = cls
        return cls

    if cls is None:
        return _register
    return _register(cls)


def get_sde(name):
    """
    Get the SDE class by name.
    """
    if name not in _SDES:
        msg = f"SDE {name} not found. Available SDEs: {list(_SDES.keys())}"
        raise ValueError(msg)
    return _SDES[name]


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N: int):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self) -> float:
        """End time of the SDE."""

    @abc.abstractmethod
    def sde(
        self, y: Float[np.ndarray, "batch y_dim"], t: Float[np.ndarray, "batch 1"]
    ) -> Tuple[Float[np.ndarray, "batch y_dim"], Float[np.ndarray, "batch y_dim"]]:
        """
        Returns the drift and diffusion functions of the SDE which should
        be of the same shape as the input.
        """

    @abc.abstractmethod
    def marginal_prob(
        self, y: Float[np.ndarray, "batch y_dim"], t: Float[np.ndarray, "batch 1"]
    ):
        """Parameters to determine the marginal distribution of the SDE, $p_t( |y)$."""

    def marginalized_perturbation_kernel(self, y0: Float[np.ndarray, "batch y_dim"]):
        """Compute the perturbation kernel density function induced by the data `y0`.
        Defined as: `p(y', t | y0) = \frac{1}{n}\\sum_{y \\in y0}p_t(y' | y)` where
        `n` is the number of data points in `y0`. Each `p_t(y' | y)` is a Gaussian
        density with conditional mean and standard deviation given by `marginal_prob`.

        Args:
            y0: data
        Returns:
            kernel_density_fn: function taking `y_prime` and `t` as input and returning
            the perturbation kernel density function induced by the diffusion of data `y0`
            for time `t`.

        """
        if len(y0.shape) == 1:
            y0 = y0[:, None]

        def kernel_density_fn(
            y_prime: Float[ndarray, "batch_2 y_dim"], t: Float[np.ndarray, "batch 1"]
        ):
            if t is None:
                t = self.T
            if len(y_prime.shape) == 1:
                y_prime = y_prime[:, None]
            if isinstance(t, float):
                t = np.ones_like(y_prime) * t
            means, stds = self.marginal_prob(y0, t)
            if isinstance(means, float):
                means = np.ones_like(y_prime) * means
            if isinstance(stds, float):
                stds = np.ones_like(y_prime) * stds
            means = means[:, None, :]
            stds = stds[:, None, :]

            return np.mean(
                np.exp(-0.5 * ((y_prime - means) / stds) ** 2) / (stds * np.sqrt(2 * np.pi)),
                axis=0,
            ).sum(axis=-1)

        return kernel_density_fn

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """

    def discretize(
        self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    ) -> Tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a NumPy array
          t: a NumPy float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(y, t)
        f = drift * dt
        G = diffusion * np.sqrt(dt)
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(
                self,
                y: Float[ndarray, "batch y_dim"],
                X: Float[ndarray, "batch x_dim"],
                t: Float[ndarray, "batch 1"],
            ) -> Tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(y, t)
                score = score_fn(y, X, t)
                drift = drift - diffusion**2 * score * (0.5 if self.probability_flow else 1.0)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(
                self,
                y: Float[ndarray, "batch y_dim"],
                X: Float[ndarray, "batch x_dim"],
                t: Float[ndarray, "batch 1"],
            ) -> Tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(y, X, t)
                score = score_fn(y, X, t)
                rev_f = f - G**2 * score * (0.5 if self.probability_flow else 1.0)
                rev_G = np.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    def __str__(self):
        return self.__class__.__name__


@_register_sde(name="vpsde")
class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = np.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t.reshape((-1,) + (1,) * (len(y.shape) - 1)) * y
        diffusion = np.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]):
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = np.exp(log_mean_coeff).reshape((-1,) + (1,) * (len(y.shape) - 1)) * y
        std = np.sqrt(1.0 - np.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return np.random.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - np.sum(z**2, axis=(1, 2, 3)) / 2.0
        return logps

    def discretize(self, y, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas[timestep]
        alpha = self.alphas[timestep]
        sqrt_beta = np.sqrt(beta)
        f = np.sqrt(alpha).reshape((-1,) + (1,) * (len(y.shape) - 1)) * y - y
        G = sqrt_beta
        return f, G

    def __str__(self):
        return f"VPSDE(beta_min={self.beta_0}, beta_max={self.beta_1})"


@_register_sde(name="subvpsde")
class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, y, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t.reshape((-1,) + (1,) * (len(y.shape) - 1)) * y
        discount = 1.0 - np.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2)
        diffusion = np.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]):
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = np.exp(log_mean_coeff).reshape((-1,) + (1,) * (len(y.shape) - 1)) * y
        std = 1 - np.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return np.random.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - np.sum(z**2, axis=(1, 2, 3)) / 2.0


@_register_sde(name="vesde")
class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = np.exp(
            np.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
        )
        self.N = N

    @property
    def T(self):
        return 1

    def sde(
        self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    ) -> Tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        sigma = repeat(sigma, "b 1 -> b y_dim", y_dim=y.shape[1])
        drift = np.zeros_like(y)
        diffusion = sigma * np.sqrt(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)))
        return drift, diffusion

    def marginal_prob(
        self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    ) -> Tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:

        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        std = repeat(std, "b 1 -> b y_dim", y_dim=y.shape[1])
        mean = y
        return mean, std

    def prior_sampling(self, shape):
        return np.random.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - np.sum(
            z**2, axis=(1, 2, 3)
        ) / (2 * self.sigma_max**2)

    def discretize(self, y, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas[timestep]
        adjacent_sigma = np.where(
            timestep == 0, np.zeros_like(t), self.discrete_sigmas[timestep - 1]
        )
        f = np.zeros_like(y)
        G = np.sqrt(sigma**2 - adjacent_sigma**2)
        return f, G

    def __str__(self):
        return f"VESDE(sigma_min={self.sigma_min}, sigma_max={self.sigma_max})"
