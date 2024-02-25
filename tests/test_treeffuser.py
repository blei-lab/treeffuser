import numpy as np
from matplotlib import pyplot as plt

from treeffuser import LightGBMTreeffusser
from utils import generate_bimodal_linear_regression_data


def test_treeffuser_bimodal_linear_regression():

    n = 1000
    p = 1
    sigma = 0.1
    n_samples = 10

    X, y = generate_bimodal_linear_regression_data(n, p, sigma, bimodal=True, seed=0)

    model = LightGBMTreeffusser(
        verbose=1,
        n_repeats=100,
        n_estimators=200,
        likelihood_reweighting=True,
        sde_name="vesde",
        learning_rate=0.01,
    )
    model.fit(X, y)

    # shape (n , n_samples, y_dim)
    y_samples = model.sample(
        X, n_samples=n_samples, n_parallel=100, denoise=False, n_steps=100, seed=0
    )
    X_repeated = np.repeat(X, n_samples, axis=0)

    plt.scatter(X_repeated, y_samples, alpha=0.1, color="blue", label="Samples")
    plt.scatter(X, y, alpha=0.1, color="red", label="Data")
    plt.show()


test_treeffuser_bimodal_linear_regression()
