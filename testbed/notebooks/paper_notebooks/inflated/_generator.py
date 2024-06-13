import numpy as np
from numpy.random import Generator
from scipy.special import loggamma

from typing import Callable
from typing import List


class CustomRandomGenerator(Generator):
    def mixed(
        self,
        p_atoms: List[float],
        loc_atoms: List[float],
        continuous_distr_name: str,
        continuous_distr_params: dict,
        continuous_shift: float = 0,
        size: int = 1,
    ):
        p_atoms = np.array(p_atoms)
        p_continuous = 1 - np.sum(p_atoms)

        assert len(p_atoms) == len(loc_atoms)
        assert p_atoms.min() >= 0
        assert p_continuous > 0

        is_discrete = self.choice([False, True], p=[p_continuous, 1 - p_continuous], size=size)
        n_discrete = is_discrete.sum()

        sample = np.zeros(size)
        if (is_discrete == False).any():
            sample[is_discrete == False] = continuous_shift + getattr(
                self, continuous_distr_name
            )(**continuous_distr_params, size=size - n_discrete)
        if is_discrete.any():
            sample[is_discrete] = self.choice(
                loc_atoms, p=p_atoms / (1 - p_continuous), size=n_discrete
            )

        return sample

    def CIG(
        self,
        p_atom_fn: Callable,
        shape: float,
        scale: float,
        size: int,
        x_inf: float = 0,
        x_sup: float = 1,
        x_shift_coefficient: float = 1.0,
        x_atom_coefficient: float = 1.0,
    ):
        """
        Covariate-Inflated Gamma

        This generates values from a mixed random variable that combines:
        - An atomic mass at x.
        - A continuous gamma distribution, shifted at x.

        The atmoic mass at x can be a function of x.
        """
        x = self.uniform(low=x_inf, high=x_sup, size=size)
        y = np.zeros_like(x)
        for i, atom in enumerate(x):
            y[i] = self.mixed(
                p_atoms=[p_atom_fn(atom)],
                loc_atoms=[atom * x_atom_coefficient],
                continuous_distr_name="gamma",
                continuous_distr_params={"shape": shape, "scale": scale},
                continuous_shift=atom * x_shift_coefficient,
                size=1,
            )

        return x, y.reshape(-1, 1)


def gamma_density(x, shape, scale, shift):
    x_shifted = x - shift
    log_density = (shape - 1) * np.log(x_shifted) - x_shifted / scale
    log_density -= loggamma(shape) + shape * np.log(scale)
    return np.exp(log_density)


def CIG_conditional_density(
    x,
    p_atom,
    shape: float,
    scale: float,
    x_shift_coefficient: float = 1.0,
    x_atom_coefficient: float = 1.0,
):
    def density_fn(y):
        density = np.zeros_like(y)
        density[y >= x] = (1 - p_atom) * gamma_density(
            y[y >= x], shape=shape, scale=scale, shift=x * x_shift_coefficient
        )
        density[y == x * x_atom_coefficient] += p_atom
        return density

    return density_fn
