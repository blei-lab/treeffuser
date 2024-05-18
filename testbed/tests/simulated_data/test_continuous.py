import numpy as np
import pytest
from testbed.simulated_data.continuous import NormalDataset
from testbed.simulated_data.continuous import StudentTDataset


def test_normal_linear():
    """
    Simple sanity check to verify that the NormalDataset is working as expected
    and the log likelihood is correct.

    TODO: Make this test better
    """
    seed = 0
    x_dim = 10
    is_linear = True
    is_heteroscedastic = False
    n = 100

    dataset = NormalDataset(
        x_dim=x_dim,
        noise_scale=1.0,
        is_linear=is_linear,
        is_heteroscedastic=is_heteroscedastic,
        seed=seed,
    )
    X, y = dataset.sample_dataset(n_samples=n, seed=seed)
    assert X.shape == (n, x_dim)
    assert y.shape == (n, 1)

    # Now for a sanity check
    X_ones = np.ones((n, x_dim))
    y_mean = X_ones @ dataset.w
    y_ones_samples = dataset.sample(X_ones, n_samples=n, seed=seed)

    # should be the same for all
    y_mean = y_mean[0, 0]
    y_std = dataset.noise_scale

    # check that the log likelihood is correct
    score = dataset.score(X_ones, y_ones_samples)
    real_score = np.mean(
        np.log(
            np.exp(-0.5 * ((y_mean - y_ones_samples) / y_std) ** 2)
            / np.sqrt(2 * np.pi * y_std**2)
        )
    )
    assert np.isclose(score, real_score)


@pytest.mark.parametrize(
    "cls, is_linear, is_heteroscedastic",  # noqa
    [
        (NormalDataset, True, True),
        (NormalDataset, True, False),
        (NormalDataset, False, True),
        (NormalDataset, False, False),
        (StudentTDataset, True, True),
        (StudentTDataset, True, False),
        (StudentTDataset, False, True),
        (StudentTDataset, False, False),
    ],
)
def test_continuous(cls, is_linear, is_heteroscedastic):
    """
    Simple sanity check to verify that everything runs without crashing
    """
    n = 1000
    x_dim = 10
    noise_scale = 1.0
    n_samples = 2

    dataset = cls(
        x_dim=x_dim,
        noise_scale=noise_scale,
        is_linear=is_linear,
        is_heteroscedastic=is_heteroscedastic,
        seed=0,
    )

    X, y = dataset.sample_dataset(n_samples=n, seed=0)
    assert X.shape == (n, x_dim)
    assert y.shape == (n, 1)

    y_samples = dataset.sample(X, n_samples=n_samples, seed=0)
    assert y_samples.shape == (n_samples, n, 1)

    score = dataset.score(X, y_samples)
    assert np.isscalar(score)
