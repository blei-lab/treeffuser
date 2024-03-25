import numpy as np
from testbed.models import Card


def test_linear_regression():
    """
    Simple test to verify that Card can perform linear regression and
    can be trained and evaluated.
    """
    seed = 42
    np.random.seed(seed)

    n_samples = 5000
    X = np.random.rand(n_samples, 1)
    beta = np.random.rand(1, 1) * 10
    mean = X @ beta
    std = 0.1

    epsilon = np.random.randn(n_samples, 1) * std
    y = mean + epsilon

    model = Card(max_epochs=500, enable_progress_bar=True, n_steps=100, patience=3)

    model.fit(X, y)
    y_pred = model.predict(X)

    samples = model.sample(X, 1, batch_size=256).flatten()

    std_sampled = np.std(samples)
    mean_samples = np.mean(samples)
    geq_0_samples = np.sum(samples >= 0)

    std_real = np.std(y)
    mean_real = np.mean(y)
    geq_0_real = np.sum(y >= 0)

    abs_diff_y = np.mean(np.abs(y_pred - mean))
    abs_diff_std = np.abs(std_sampled - std_real)
    abs_diff_mean = np.abs(mean_samples - mean_real)
    abs_diff_geq_0 = np.abs(geq_0_samples - geq_0_real) / n_samples

    ERR_TOL = 0.2
    assert abs_diff_std < ERR_TOL, f"abs_diff_std: {abs_diff_std}"
    assert abs_diff_mean < ERR_TOL, f"abs_diff_mean: {abs_diff_mean}"
    assert abs_diff_y < ERR_TOL, f"abs_diff_y: {abs_diff_y}"
    assert abs_diff_geq_0 < ERR_TOL, f"abs_diff_geq_0: {abs_diff_geq_0}"
