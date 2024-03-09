import abc
from typing import Union

import numpy as np
from numpy import ndarray

array_like = Union[ndarray, float]


class ParameterSchedule(abc.ABC):
    """
    Base class representing a parameter as a function of time.
    """

    @abc.abstractmethod
    def get_value(self, t: array_like) -> array_like:
        """Get the value of the parameter at time `t`."""

    def get_derivative(self, t: array_like) -> array_like:
        """Get the derivative of the parameter at time `t`."""
        raise NotImplementedError

    def get_integral(self, t: array_like) -> array_like:
        """Get the integral of the parameter at time `t` from time `0`."""
        raise NotImplementedError

    def __call__(self, t: array_like) -> array_like:
        return self.get_value(t)


class LinearSchedule(ParameterSchedule):
    """
    Linear schedule for a parameter, between `min_value` and `max_value`.
    The value of the parameter at time `t` is given by:
    `min_value + (max_value - min_value) * t`

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

    def get_value(self, t: array_like) -> array_like:
        return self.min_value + (self.max_value - self.min_value) * t

    def get_derivative(self, t: array_like) -> array_like:
        if isinstance(t, float):
            return self.max_value - self.min_value
        return np.broadcast_to(self.max_value - self.min_value, t.shape)

    def get_integral(self, t: array_like) -> array_like:
        integral = self.min_value * t
        integral += 0.5 * (self.max_value - self.min_value) * t**2
        return integral

    def __repr__(self):
        return f"LinearSchedule(min_value={self.min_value}, max_value={self.max_value})"


class ExponentialSchedule(ParameterSchedule):
    """
    Exponential schedule for a parameter, between `min_value` and `max_value`.
    The value of the parameter at time `t` is given by:
    `min_value * (max_value / min_value) ** t`

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

    def get_value(self, t: array_like) -> array_like:
        return self.min_value * (self.max_value / self.min_value) ** t

    def get_derivative(self, t: array_like) -> array_like:
        return self.get_value(t) * np.log(self.max_value / self.min_value)

    def get_integral(self, t: array_like) -> array_like:
        return (self.get_value(t) - self.min_value) / np.log(self.max_value / self.min_value)

    def __repr__(self):
        return f"ExponentialSchedule(min_value={self.min_value}, max_value={self.max_value})"
