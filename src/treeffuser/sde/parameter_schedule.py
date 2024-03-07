import abc

import numpy as np
from jaxtyping import Float
from numpy import ndarray


class ParameterSchedule(abc.ABC):
    """
    Base class for a parameter schedule, representing a parameter as a function of time.
    """

    @abc.abstractmethod
    def get_value(self, t: Float[ndarray, "batch"]) -> Float[ndarray, "batch"]:
        """Get the value of the parameter at time `t`."""

    def get_derivative(self, t: Float[ndarray, "batch"]) -> Float[ndarray, "batch"]:
        """Get the derivative of the parameter at time `t`."""
        raise NotImplementedError

    def get_integral(
        self, t: Float[ndarray, "batch"], t0: Float[ndarray, "batch"]
    ) -> Float[ndarray, "batch"]:
        """Get the integral of the parameter at time `t` from time `t0`."""
        raise NotImplementedError

    def __call__(self, t: Float[ndarray, "batch"]) -> Float[ndarray, "batch"]:
        return self.get_value(t)


class LinearSchedule(ParameterSchedule):
    """
    Linear schedule for a parameter, between `min_value` and `max_value`.
    The value of the parameter at time `t` is given by:
    .. math:: min_value + (max_value - min_value) * t

    Parameters
    ----------
    min_value : float
        Minimum value of the parameter (at time 0).
    max_value : float
        Maximum value of the parameter (at time 1).
    """

    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value

    def get_value(self, t: Float[ndarray, "batch"]) -> Float[ndarray, "batch"]:
        return self.min_value + (self.max_value - self.min_value) * t

    def get_derivative(self, t: Float[ndarray, "batch"]) -> Float[ndarray, "batch"]:
        return self.max_value - self.min_value

    def get_integral(
        self, t: Float[ndarray, "batch"], t0: Float[ndarray, "batch"] = 0
    ) -> Float[ndarray, "batch"]:
        return self.min_value * (t - t0) + 0.5 * (self.max_value - self.min_value) * (
            t**2 - t0**2
        )

    def __repr__(self):
        return f"LinearSchedule(min_value={self.min_value}, max_value={self.max_value})"


class ExponentialSchedule(ParameterSchedule):
    """
    Exponential schedule for a parameter, between `min_value` and `max_value`.
    The value of the parameter at time `t` is given by:
    .. math:: min_value * (max_value / min_value) ** t

    Parameters
    ----------
    min_value : float
        Minimum value of the parameter (at time 0).
    max_value : float
        Maximum value of the parameter (at time 1).
    """

    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value

    def get_value(self, t: Float[ndarray, "batch"]) -> Float[ndarray, "batch"]:
        return self.min_value * (self.max_value / self.min_value) ** t

    def get_derivative(self, t: Float[ndarray, "batch"]) -> Float[ndarray, "batch"]:
        return self.get_value(t) * np.log(self.max_value / self.min_value)

    def get_integral(
        self, t: Float[ndarray, "batch"], t0: Float[ndarray, "batch"] = 0
    ) -> Float[ndarray, "batch"]:
        return (self.get_value(t) - self.get_value(t0)) / np.log(
            self.max_value / self.min_value
        )

    def __repr__(self):
        return f"ExponentialSchedule(min_value={self.min_value}, max_value={self.max_value})"
