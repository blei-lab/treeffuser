import abc
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from .base_sde import BaseSDE
from .base_sde import _register_sde
from .parameter_schedule import ExponentialSchedule
from .parameter_schedule import LinearSchedule


class DiffusionSDE(BaseSDE):
    """
    Abstract class representing a diffusion SDE:
    `dY = (A(t) + B(t) Y) dt + C(t) dW` where `C(t)` is a time-varying
    diffusion coefficient independent of `Y`, and the drift is an affine function of Y.
    As a result, the conditional distribution p_t(y | y0) is Gaussian.
    """

    @property
    def T(self) -> float:
        """End time of the SDE."""
        return 1.0

    @abc.abstractmethod
    def get_hyperparams(self) -> Dict:
        """
        Return a dictionary with the hyperparameters of the SDE.

        The hyperparameters parametrize the drift and diffusion coefficients of the
        SDE.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_from_theoretical_prior(
        self, shape: tuple[int, ...], seed: Optional[int] = None
    ) -> Float[ndarray, "*shape"]:
        """
        Sample from the theoretical distribution that p_T(y) converges to.

        Parameters
        ----------
        shape : tuple
            Shape of the output array.
        seed : int (optional)
            Random seed. If None, the random number generator is not seeded.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_mean_std_pt_given_y0(
        self,
        y0: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
        """
        Our diffusion SDEs have conditional distributions p_t(y | y0) that
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

    def get_marginalized_perturbation_kernel(self, y0: Float[np.ndarray, "batch y_dim"]):
        """
        Compute the marginalized perturbation kernel density function induced by the data `y0`.

        The marginalized perturbation kernel is defined as:
            `p(y, t) = \frac{1}{n}\\sum_{y' \\in y0}p_t(y | y')`
        where `n` is the number of data points in `y0`. Each `p_t(y | y')` is a Gaussian
        density with conditional mean and standard deviation given by `marginal_prob`.

        Args:
            y0: data
        Returns:
            kernel_density_fn: function taking `y_prime` and `t` as input and returning
            the perturbation kernel density function induced by the diffusion of data `y0`
            for time `t`.

        """

        def kernel_density_fn(
            y: Float[ndarray, "batch_ y_dim"], t: Union[float, Float[np.ndarray, "batch_ 1"]]
        ):
            if isinstance(t, float):
                t = np.ones_like(y) * t
            means, stds = self.get_mean_std_pt_given_y0(y0, t)
            means = means[:, None, :]
            stds = stds[:, None, :]

            return np.mean(
                np.exp(-0.5 * ((y - means) / stds) ** 2) / (stds * np.sqrt(2 * np.pi)),
                axis=0,
            ).sum(axis=-1)

        return kernel_density_fn


@_register_sde(name="vesde")
class VESDE(DiffusionSDE):
    """
    Variance-exploding SDE (VESDE):
        `dY = 0 dt + \\sqrt{2 \\sigma(t) \\sigma'(t)} dW`
    where `sigma(t)` is a time-varying diffusion coefficient with exponential schedule:

    `\\sigma(t) = hyperparam_min * (hyperparam_max / hyperparam_min) ^ t`

    The SDE converges to a normal distribution with variance `hyperparam_max ^ 2`.

    Parameters
    ----------
    hyperparam_min : float
        Minimum value of the diffusion coefficient.
    hyperparam_max : float
        Maximum value of the diffusion coefficient.
    """

    def __init__(self, hyperparam_min=0.01, hyperparam_max=20):
        self.hyperparam_min = hyperparam_min
        self.hyperparam_max = hyperparam_max
        self.hyperparam_schedule = ExponentialSchedule(hyperparam_min, hyperparam_max)

    def get_hyperparams(self):
        return {"hyperparam_min": self.hyperparam_min, "hyperparam_max": self.hyperparam_max}

    def drift_and_diffusion(
        self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    ) -> tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
        hyperparam = self.hyperparam_schedule(t)
        hyperparam_prime = self.hyperparam_schedule.get_derivative(t)
        drift = 0
        diffusion = np.sqrt(2 * hyperparam * hyperparam_prime)
        return drift, diffusion

    def sample_from_theoretical_prior(
        self, shape: tuple[int, ...], seed: Optional[int] = None
    ) -> Float[ndarray, "batch x_dim y_dim"]:
        # This assumes that hyperparam_max is large enough
        rng = np.random.default_rng(seed)
        return rng.normal(0, self.hyperparam_max, shape)

    def get_mean_std_pt_given_y0(
        self,
        y0: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
        """
        The conditional distribution is Gaussian with:
            mean: `y0`
            variance: `hyperparam(t)**2 - hyperparam(0)**2`
        """
        mean = y0
        std = (
            self.hyperparam_schedule(t) ** 2 - self.hyperparam_schedule(np.zeros_like(t)) ** 2
        ) ** 0.5

        std = np.broadcast_to(std, y0.shape)
        return mean, std

    def __repr__(self):
        return f"VESDE(hyperparam_min={self.hyperparam_min}, hyperparam_max={self.hyperparam_max})"


@_register_sde(name="vpsde")
class VPSDE(DiffusionSDE):
    """
    Variance-preserving SDE (VPSDE):
    `dY = -0.5 \\beta(t) Y dt + \\sqrt{\\beta(t)} dW`
    where `beta(t)` is a time-varying coefficient with linear schedule:
    `\\beta(t) = hyperparam_min + (hyperparam_max - hyperparam_min) * t.`

    The SDE converges to a standard normal distribution for large `hyperparam_max`.

    Parameters
    ----------
    hyperparam_min : float
        Minimum value of the time-varying coefficient `\\beta(t)`.
    hyperparam_max : float
        Maximum value of the time-varying coefficient `\\beta(t)`.
    """

    def __init__(self, hyperparam_min=0.01, hyperparam_max=20):
        self.hyperparam_min = hyperparam_min
        self.hyperparam_max = hyperparam_max
        self.hyperparam_schedule = LinearSchedule(hyperparam_min, hyperparam_max)

    def get_hyperparams(self):
        return {"hyperparam_min": self.hyperparam_min, "hyperparam_max": self.hyperparam_max}

    def drift_and_diffusion(
        self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    ) -> tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
        hyperparam_t = self.hyperparam_schedule(t)
        drift = -0.5 * hyperparam_t * y
        diffusion = np.sqrt(hyperparam_t)
        return drift, diffusion

    def sample_from_theoretical_prior(
        self, shape: tuple[int, ...], seed: Optional[int] = None
    ) -> Float[ndarray, "batch x_dim y_dim"]:
        # Assume that hyperparam_max is large enough so that the SDE has converged to N(0,1).
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, shape)

    def get_mean_std_pt_given_y0(
        self,
        y0: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
        """
        The conditional distribution is Gaussian with:
            mean: `y0 * exp(-0.5 * \\int_0^t1 \\beta(s) ds)`
            variance: `1 - exp(-\\int_0^t1 \\beta(s) ds)`
        """
        hyperparam_integral = self.hyperparam_schedule.get_integral(t)
        mean = y0 * np.exp(-0.5 * hyperparam_integral)
        std = (1 - np.exp(-hyperparam_integral)) ** 0.5

        mean = np.broadcast_to(mean, y0.shape)
        std = np.broadcast_to(std, y0.shape)
        return mean, std

    def __repr__(self):
        return f"VPSDE(hyperparam_min={self.hyperparam_min}, hyperparam_max={self.hyperparam_max})"


@_register_sde(name="sub-vpsde")
class SubVPSDE(DiffusionSDE):
    """
    Sub-Variance-preserving SDE (SubVPSDE):
    `dY = -0.5 \\beta(t) Y dt + \\sqrt{\\beta(t) (1 - e^{-2 \\int_0^t \\beta(s) ds})} dW`
    where `beta(t)` is a time-varying coefficient with linear schedule:
    `\\beta(t) = hyperparam_min + (hyperparam_max - hyperparam_min) * t.`

    The SDE converges to a standard normal distribution for large `hyperparam_max`.

    Parameters
    ----------
    hyperparam_min : float
        Minimum value of the time-varying coefficient `\\beta(t)`.
    hyperparam_max : float
        Maximum value of the time-varying coefficient `\\beta(t)`.
    """

    def __init__(self, hyperparam_min=0.01, hyperparam_max=20):
        self.hyperparam_min = hyperparam_min
        self.hyperparam_max = hyperparam_max
        self.hyperparam_schedule = LinearSchedule(hyperparam_min, hyperparam_max)

    def get_hyperparams(self):
        return {"hyperparam_min": self.hyperparam_min, "hyperparam_max": self.hyperparam_max}

    def drift_and_diffusion(
        self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    ) -> tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
        hyperparam_t = self.hyperparam_schedule(t)
        drift = -0.5 * hyperparam_t * y
        hyperparam_integral = self.hyperparam_schedule.get_integral(t)
        discount = 1.0 - np.exp(-2 * hyperparam_integral)
        diffusion = np.sqrt(hyperparam_t * discount)
        return drift, diffusion

    def sample_from_theoretical_prior(
        self, shape: tuple[int, ...], seed: Optional[int] = None
    ) -> Float[ndarray, "batch x_dim y_dim"]:
        # Assume that hyperparam_max is large enough so that the SDE has converged to N(0,1).
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, shape)

    def get_mean_std_pt_given_y0(
        self,
        y0: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
        """
        The conditional distribution is Gaussian with:
            mean: `y0 * exp(-0.5 * \\int_0^t1 \\beta(s) ds)`
            variance: `[1 - exp(-\\int_0^t1 \\beta(s) ds)]^2`
        """
        hyperparam_integral = self.hyperparam_schedule.get_integral(t)
        mean = y0 * np.exp(-0.5 * hyperparam_integral)
        std = 1 - np.exp(-hyperparam_integral)

        mean = np.broadcast_to(mean, y0.shape)
        std = np.broadcast_to(std, y0.shape)
        return mean, std

    def __repr__(self):
        return f"VPSDE(hyperparam_min={self.hyperparam_min}, hyperparam_max={self.hyperparam_max})"
