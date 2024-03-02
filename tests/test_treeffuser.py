import numpy as np
from scipy.stats import ks_2samp

from treeffuser import LightGBMTreeffuser


def test_treeffuser_bimodal_linear_regression():
    """
    We do a very simple sanity check that for a very simple model
    with not enough data the samples from the model should
    be statistically indistinguishable from the data.
    """
    n = 500
    n_samples = 1
    X_1 = np.random.rand(n, 1)
    y_1 = X_1 + np.random.randn(n, 1) * 0.05 * (X_1 + 1) ** 2

    X_2 = np.random.rand(n, 1)
    y_2 = X_2 + np.random.randn(n, 1) * 0.05 * (X_2 + 1) ** 2

    X = np.concatenate([X_1, X_2], axis=0)
    y = np.concatenate([y_1, y_2], axis=0)

    # Shuffle and split the data
    idx = np.random.permutation(2 * n)
    X = X[idx]
    y = y[idx]

    X_train = X[:n]
    y_train = y[:n]

    X_test = X[n:]
    y_test = y[n:]

    model = LightGBMTreeffuser(
        verbose=1,
        n_repeats=20,
        n_estimators=10000,
        sde_name="vesde",
        learning_rate=0.1,
        early_stopping_rounds=20,
        seed=0,
    )
    model.fit(X_train, y_train)

    y_samples = model.sample(
        X_test, n_samples=n_samples, n_parallel=50, denoise=True, n_steps=30, seed=0
    )

    y_samples = y_samples.flatten()
    y_test = y_test.flatten()

    # Check that the samples are statistically indistinguishable from the data
    result = ks_2samp(y_samples, y_test)
    assert result.pvalue > 0.05
