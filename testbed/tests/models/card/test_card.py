import numpy as np
from testbed.models import Card


def test_linear_regression():
    """
    Simple test to verify that Card can perform linear regression and
    can be trained and evaluated.
    """
    n_samples = 100
    n_features = 10

    X = np.random.rand(n_samples, n_features)
    beta = np.random.rand(n_features, 1)
    epsilon = np.random.rand(n_samples, 1) * 0.1

    y = X @ beta + epsilon

    card = Card()
    card.fit(X=X, y=y)
