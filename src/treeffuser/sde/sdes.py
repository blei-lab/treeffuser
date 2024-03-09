import abc
from typing import Optional

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from .base_sde import BaseSDE, _register_sde
from .parameter_schedule import ExponentialSchedule, LinearSchedule


class DiffusionSDE(BaseSDE):
    """
    Abstract class representing a diffusion SDE:
    `dY = f(Y, t) dt + \\sigma(t) dW` where `\\sigma(t)` is a time-varying
    diffusion coefficient independent of `Y`. As a result, the conditional
    distribution p_t(y | y0) is Gaussian.
    """

    @property
    def T(self) -> float:
        """End time of the SDE."""
        return 1.0

    @abc.abstractmethod
    def sample_from_theoretical_prior(
        self, shape: tuple[int, ...], seed: Optional[int] = None
    ) -> Float[ndarray, "*shape"]:
        """Sample from the theoretical distribution that p_T(y) converges to.

        Parameters
        ----------
        shape : tuple
            Shape of the output array.
        seed : int (optional)
            Random seed. If None, the random number generator is not seeded.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_mean_std_conditional_pt_given_y0(
        self,
        y0: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> (Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]):
        """Our diffusion SDEs have conditional distributions p_t(y | y0) that
         are Gaussian. This method returns their mean and standard deviation.

        Parameters
        ----------
        y0 : ndarray of shape (*batch, y_dim)
            Initial value at t=0.
        t : float
            Time at which to compute the conditional distribution.

        Returns
        -------
        mean : ndarray of shape (*batch, y_dim)
            Mean of the conditional distribution.
        std : ndarray of shape (*batch, y_dim)
            Standard deviation of the conditional distribution.
        """
        raise NotImplementedError


@_register_sde(name="vesde")
class VESDE(DiffusionSDE):
    """
    Variance-exploding SDE (VESDE):
        `dY = 0 dt + \\sqrt{2 \\sigma(t) \\sigma'(t)} dW`
    where `sigma(t)` is a time-varying diffusion coefficient.

    The SDE converges to a normal distribution with variance `sigma_max**2`.

    Parameters
    ----------
    sigma_min : float
        Minimum value of the diffusion coefficient.
    sigma_max : float
        Maximum value of the diffusion coefficient.
    """

    def __init__(self, sigma_min=0.01, sigma_max=20):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_schedule = ExponentialSchedule(sigma_min, sigma_max)

    def drift_and_diffusion(
        self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    ) -> (Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]):
        sigma = self.sigma_schedule(t)
        sigma_prime = self.sigma_schedule.get_derivative(t)
        drift = 0
        diffusion = np.sqrt(2 * sigma * sigma_prime)
        return drift, diffusion

    def sample_from_theoretical_prior(
        self, shape: tuple[int, ...], seed: Optional[int] = None
    ) -> Float[ndarray, "batch x_dim y_dim"]:
        # This assumes that sigma_max is large enough
        # TODO: maybe (sigma_max**2 + 1 - sigma_min**2)**0.5 would be more accurate
        rng = np.random.default_rng(seed)
        return rng.normal(0, self.sigma_max, shape)

    def get_mean_std_conditional_pt_given_y0(
        self,
        y0: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> (Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]):
        """
        The conditional distribution is Gaussian with:
        * mean = `y0`
        * variance = `sigma(t)**2 - sigma(0)**2`
        """
        mean = y0
        std = (self.sigma_schedule(t) ** 2 - self.sigma_schedule(np.zeros_like(t)) ** 2) ** 0.5

        std = np.broadcast_to(std, y0.shape)
        return mean, std

    def __repr__(self):
        return f"VESDE(sigma_min={self.sigma_min}, sigma_max={self.sigma_max})"


@_register_sde(name="vpsde")
class VPSDE(DiffusionSDE):
    """
    Variance-preserving SDE (VPSDE):
    `dY = -0.5 \\beta(t) Y dt + \\sqrt{\\beta(t)} dW`
    where `beta(t)` is a time-varying diffusion coefficient.

    The SDE converges to a standard normal distribution for large `beta(t)`.

    Parameters
    ----------
    beta_min : float
        Minimum value of the diffusion coefficient.
    beta_max : float
        Maximum value of the diffusion coefficient.
    """

    def __init__(self, beta_min=0.01, beta_max=20):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_schedule = LinearSchedule(beta_min, beta_max)

    def drift_and_diffusion(
        self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    ) -> (Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]):
        beta_t = self.beta_schedule(t)
        drift = -0.5 * beta_t * y
        diffusion = np.sqrt(beta_t)
        return drift, diffusion

    def sample_from_theoretical_prior(
        self, shape: tuple[int, ...], seed: Optional[int] = None
    ) -> Float[ndarray, "batch x_dim y_dim"]:
        # Assume that beta_max is large enough so that the SDE has converged
        # to a standard normal distribution.
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, shape)

    def get_mean_std_conditional_pt_given_y0(
        self,
        y0: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> (Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]):
        """
        The conditional distribution is Gaussian with:
        * mean = `y0 * exp(-0.5 * \\int_0^t1 beta(s) ds)`
        * variance = `1 - exp(-\\int_0^t1 beta(s) ds)`
        """
        beta_integral = self.beta_schedule.get_integral(t)
        mean = y0 * np.exp(-0.5 * beta_integral)
        std = (1 - np.exp(-beta_integral)) ** 0.5

        mean = np.broadcast_to(mean, y0.shape)
        std = np.broadcast_to(std, y0.shape)
        return mean, std
