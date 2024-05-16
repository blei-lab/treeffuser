import numpy as np
from testbed.simulated_data.normal_linear import NormalLinearDataset


def test_normal_linear():
    seed = 0
    n = 100
    d = 10

    dataset = NormalLinearDataset(d=d)
    X, y = dataset.sample_dataset(n_samples=n, seed=seed)
    assert X.shape == (n, d)
    assert y.shape == (n, 1)

    # Now for a sanity check
    X_ones = np.ones((n, d))
    y_mean = X_ones @ NormalLinearDataset(d=d).w
    y_ones_samples = dataset.sample(X_ones, n_samples=n, seed=seed)

    # should be the same for all
    y_mean = y_mean[0, 0]
    y_std = dataset.noise_std

    # check that the log likelihood is correct
    score = dataset.score(X_ones, y_ones_samples)
    real_score = np.mean(
        np.log(
            np.exp(-0.5 * ((y_mean - y_ones_samples) / y_std) ** 2)
            / np.sqrt(2 * np.pi * y_std**2)
        )
    )
    assert np.isclose(score, real_score)
