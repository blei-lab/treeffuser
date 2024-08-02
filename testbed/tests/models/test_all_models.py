"""
Contains test for simple sanity checks for the models
"""

from typing import Type

import numpy as np
import pytest
from sklearn.metrics import r2_score
from testbed.models import ProbabilisticModel
from testbed.models.ibug_kde_ import IBugXGBoostKDE


@pytest.mark.parametrize(
    "model_class, params",
    [
        (IBugXGBoostKDE, {"k": 10}),
    ],
)
def test_all_models_sanity(model_class: Type[ProbabilisticModel], params: dict):
    """
    Simple test to check if our wrapper for bayesian optimization works
    """
    # add seed
    np.random.seed(0)

    n_samples = 1000
    n_pred_samples = 10
    n_features = 1
    n_targets = 1

    X = np.random.rand(n_samples, n_features)
    epsilon = np.random.rand(n_samples, n_targets) * 0.1
    beta = np.random.rand(n_features, n_targets)

    y = X @ beta + epsilon
    model = model_class(**params)
    model.fit(X, y)

    # Check that everything runs
    y_pred = model.predict(X)
    y_samples = model.sample(X, n_samples=n_pred_samples)

    assert y_pred.shape == (n_samples, n_targets)
    assert y_samples.shape == (n_pred_samples, n_samples, n_targets)

    r2_pred = r2_score(y.flatten(), y_pred.flatten())
    r2_samples = r2_score(y.flatten(), y_samples.mean(axis=0).flatten())

    assert r2_pred > 0.9
    assert r2_samples > 0.9
