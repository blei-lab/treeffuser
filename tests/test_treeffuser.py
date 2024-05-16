import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from treeffuser import LightGBMTreeffuser
from utils import gaussian_mixture_pdf


def test_treeffuser_bimodal_linear_regression():
    """
    We do a very simple sanity check that for a very simple model with not enough data the
    samples from the model should be statistically indistinguishable from the data.
    """
    n = 500
    n_samples = 1
    rng = np.random.default_rng(seed=0)

    X_1 = rng.uniform(size=(n, 1))
    y_1 = X_1 + rng.normal(size=(n, 1)) * 0.05 * (X_1 + 1) ** 2

    X_2 = rng.uniform(size=(n, 1))
    y_2 = X_2 + rng.normal(size=(n, 1)) * 0.05 * (X_2 + 1) ** 2

    X = np.concatenate([X_1, X_2], axis=0)
    y = np.concatenate([y_1, y_2], axis=0)

    # Shuffle and split the data
    idx = rng.permutation(2 * n)
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

    y_samples = model.sample(X_test, n_samples=n_samples, n_parallel=50, n_steps=30, seed=0)

    y_samples = y_samples.flatten()
    y_test = y_test.flatten()

    # Check that the samples are statistically indistinguishable from the data
    result = ks_2samp(y_samples, y_test)
    assert result.pvalue > 0.05


def test_sample_based_nll_gaussian_mixture():
    """
    The data are generated from a Gaussian mixture model with conditional density:
    p(y_i | x_i) = .5 * N(x_i, x_i ** 2) + (1 - .5) * N(-x_i, x_i ** 2)
    """
    n = 10**3
    rng = np.random.default_rng(seed=0)

    x = rng.uniform(low=1, high=2, size=(n, 1))
    sign = 2 * rng.binomial(n=1, p=0.5, size=(n, 1)) - 1
    y = rng.normal(loc=sign * x, scale=abs(x), size=(n, 1))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

    model = LightGBMTreeffuser(
        verbose=1,
        n_repeats=20,
        n_estimators=10000,
        sde_name="vesde",
        learning_rate=0.1,
        early_stopping_rounds=20,
        seed=0,
    )
    model.fit(x_train, y_train)

    nll_treeffuser = model.compute_nll(x_test, y_test, ode=False, n_samples=10**3, bandwidth=1)
    nll_true = -(
        gaussian_mixture_pdf(
            y_test, x_test, np.abs(x_test), -x_test, np.abs(x_test), 0.5, log=True
        )
        .sum()
        .item()
    )

    relative_error = np.abs(nll_treeffuser / nll_true - 1)
    assert relative_error < 0.05, f"relative error: {relative_error}"


def test_categorical():
    """Basic test for categorical variable support."""
    n = 10**3
    rng = np.random.default_rng(seed=0)

    X_noncat = rng.uniform(low=1, high=2, size=(n, 1))
    X_cat = rng.choice(1, size=(n, 1))
    X = np.concatenate([X_noncat, X_cat], axis=1)

    y = rng.normal(loc=X_noncat + 2 * X_cat, scale=1, size=(n, 1))

    for cat_idx in [None, [1]]:
        model = LightGBMTreeffuser()
        model.fit(X=X, y=y, cat_idx=cat_idx)


test_categorical()
