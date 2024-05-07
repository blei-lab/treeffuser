import numpy as np
from sklearn.metrics import r2_score
from testbed.models import QuantileRegression


def test_quantile_regression_works():
    """
    Simple test to check if our wrapper for bayesian optimization works
    """
    # add seed
    np.random.seed(0)

    n_samples = 1000
    n_pred_samples = 100
    n_features = 1
    n_targets = 1

    X = np.random.randn(n_samples, n_features)
    epsilon = np.random.randn(n_samples, n_targets) * 0.2
    beta = np.random.randn(n_features, n_targets)

    y = X @ beta + epsilon
    model = QuantileRegression()
    model.fit(X, y)

    # Check that everything runs
    y_pred = model.predict(X)
    y_samples = model.sample(X, n_samples=n_pred_samples)
    y_pred_via_samples = np.mean(y_samples, axis=0)

    # y_samples = model.sample(X, n_samples=n_pred_samples)
    r2 = r2_score(y.flatten(), y_pred.flatten())
    r2_via_samples = r2_score(y.flatten(), y_pred_via_samples.flatten())
    print(r2)
    print(r2_via_samples)

    predict_0_5_quantile = model.predict_quantiles(X, np.ones(n_samples) * 0.5)
    predict_0_95_quantile = model.predict_quantiles(X, np.ones(n_samples) * 0.95)
    predict_0_05_quantile = model.predict_quantiles(X, np.ones(n_samples) * 0.05)

    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]
    zero_quantile_sorted = predict_0_5_quantile[sort_idx]
    five_quantile_sorted = predict_0_05_quantile[sort_idx]
    ninety_five_quantile_sorted = predict_0_95_quantile[sort_idx]

    import matplotlib.pyplot as plt

    plt.plot(X_sorted, zero_quantile_sorted, label="0.5 quantile")
    plt.plot(X_sorted, five_quantile_sorted, label="0.05 quantile")
    plt.plot(X_sorted, ninety_five_quantile_sorted, label="0.95 quantile")
    plt.scatter(X, y, color="black", label="data", alpha=0.1, s=1)
    plt.legend()
    for idx in range(3):
        plt.scatter(
            [X[idx] for n in range(n_pred_samples)],
            y_samples[:, idx, 0],
            color="red",
            alpha=0.1,
        )

    plt.show()

    assert y_pred.shape == (n_samples, n_targets)
    assert y_samples.shape == (n_pred_samples, n_samples, n_targets)

    assert r2 > 0.9
    assert r2_via_samples > 0.9

    # assert y_samples.shape == (n_pred_samples, n_samples, n_targets)


test_quantile_regression_works()
