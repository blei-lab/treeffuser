from typing import Type

import numpy as np
import pytest
from testbed.models import BayesOptProbabilisticModel
from testbed.models import ProbabilisticModel
from testbed.models.ngboost import NGBoostGaussian


@pytest.mark.parametrize("model_class", [NGBoostGaussian])
def test_bayes_opt_works(model_class: Type[ProbabilisticModel]):
    """
    Simple test to check if our wrapper for bayesian optimization works
    """
    # add seed
    np.random.seed(0)

    n_samples = 100
    n_pred_samples = 10
    n_features = 1
    n_targets = 1

    X = np.random.rand(n_samples, n_features)
    epsilon = np.random.rand(n_samples, n_targets)
    beta = np.random.rand(n_features, n_targets)

    y = X @ beta + epsilon
    model = BayesOptProbabilisticModel(model_class=model_class, n_iter=2, cv=2, n_jobs=1)
    model.fit(X, y)

    # Check that everything runs
    y_pred = model.predict(X)
    y_samples = model.sample(X, n_samples=n_pred_samples)

    assert y_pred.shape == (n_samples, n_targets)
    assert y_samples.shape == (n_pred_samples, n_samples, n_targets)
