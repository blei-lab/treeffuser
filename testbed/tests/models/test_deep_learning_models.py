"""
Simple test to make sure that the deep learning models are learning sensible stuff
"""

from typing import Type

import numpy as np
import pytest
from sklearn.metrics import r2_score
from testbed.models import ProbabilisticModel
from testbed.models.lightning_uq_models import Card
from testbed.models.lightning_uq_models import DeepEnsemble
from testbed.models.lightning_uq_models import MCDropout
from testbed.models.nnffuser import NNffuser


@pytest.mark.parametrize(
    ("model_class, n_samples, p_train"),  # type: ignore
    [
        (NNffuser, 2_000, 0.9),
        (DeepEnsemble, 5_000, 0.9),
        (MCDropout, 1_000, 0.9),
        (Card, 10_000, 0.99),  # Card needs more samples to work properly
    ],
)
def test_deep_model_work(
    model_class: Type[ProbabilisticModel], n_samples: int, p_train: float
):
    """
    Simple test to check that card works
    """
    # add seed
    np.random.seed(0)
    n_pred_samples = 50
    n_features = 1
    n_targets = 1
    scale = 100  # to make sure scaling works

    n_train = int(p_train * n_samples)
    n_test = n_samples - n_train
    std = 1

    X = np.random.rand(n_samples, n_features) * scale
    epsilon = np.random.randn(n_samples, n_targets) * std
    beta = np.random.randn(n_features, n_targets)
    mean = X @ beta
    y = mean + epsilon

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    model = model_class(enable_progress_bar=True)
    model.fit(X_train, y_train)

    # Check that everything runs
    y_pred = model.predict(X_test)
    y_samples = model.sample(X_test, n_samples=n_pred_samples)
    y_pred_samples = np.mean(y_samples, axis=0)

    assert y_pred.shape == (n_test, n_targets)
    assert y_samples.shape == (n_pred_samples, n_test, n_targets)

    r2 = r2_score(y_test.flatten(), y_pred.flatten())
    assert r2 > 0.9, f"R2 score is {r2}"

    r2_samples = r2_score(y_test.flatten(), y_pred_samples.flatten())
    assert r2_samples > 0.9, f"R2 score is {r2_samples}"

    if not isinstance(model, MCDropout):  # MCDropout doesn't work :(
        pred_std = np.mean(np.std(y_samples, axis=0))
        assert np.abs(pred_std - std) < 0.1, f"Predicted std is {pred_std}"
