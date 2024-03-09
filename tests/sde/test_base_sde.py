import numpy as np
import pytest
from jaxtyping import Float
from numpy import ndarray

from treeffuser.sde.base_sde import BaseSDE, ReverseSDE
from unittest.mock import MagicMock


class ConcreteBaseSDE(BaseSDE):
    def drift_and_diffusion(
        self, y: Float[ndarray, "batch y_dim"], t: Float[ndarray, "batch 1"]
    ) -> (Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]):
        raise NotImplementedError


@pytest.fixture
def base_sde():
    return ConcreteBaseSDE()


@pytest.fixture
def reverse_sde():
    return ReverseSDE(ConcreteBaseSDE(), 0.0, lambda y, t: y)


def test_base_sde_drift_and_diffusion(base_sde):
    base_sde.drift_and_diffusion = MagicMock(return_value=(1.0, 2.0))
    assert base_sde.drift_and_diffusion(np.array([[1.0]]), np.array([[0.0]])) == (1.0, 2.0)


def test_reverse_sde_drift_and_diffusion(reverse_sde):
    reverse_sde.sde.drift_and_diffusion = MagicMock(return_value=(1.0, 2.0))
    reverse_sde.score_fn = MagicMock(return_value=3.0)
    assert reverse_sde.drift_and_diffusion(np.array([[1.0]]), np.array([[0.0]])) == (
        -1.0 + 2.0**2 * 3.0,
        2.0,
    )


def test_base_sde_raises_error_drift_and_diffusion_not_implemented(base_sde):
    with pytest.raises(RuntimeError):
        base_sde.drift_and_diffusion(np.array([[1.0]]), np.array([[0.0]]))
