"""
Contains all of the test for the different score model classes.
"""

import numpy as np
import pytest
from einops import repeat
from sklearn.metrics import r2_score

from treeffuser._score_models import LightGBMScore
from treeffuser._sdes import VESDE

from .utils import generate_bimodal_linear_regression_data


def test_linear_regression():
    """
    This test checks that the score model can fit a simple linear regression model.
    We do this by using the fact that for the VESDE model the score
    is -(y_perturbed - y_true)/sigma^2.  Hence

    Hence
        y_true = -score(y_perturbed; x, t) * sigma^2 + y_perturbed
    """

    # Params
    n = 1000
    x_dim = 1
    y_dim = 1
    sigma = 0.00001
    n_estimators = 100
    learning_rate = 0.01
    n_repeats = 10

    X, y = generate_bimodal_linear_regression_data(n, x_dim, sigma, bimodal=False, seed=0)

    assert X.shape == (n, x_dim)
    assert y.shape == (n, y_dim)

    # Fit a score model
    sigma_min = 0.01
    sigma_max = y.std()
    sde = VESDE(sigma_min=sigma_min, sigma_max=sigma_max)
    score_model = LightGBMScore(
        sde=sde,
        verbose=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_repeats=n_repeats,
    )
    score_model.fit(X, y)

    # Check that the score model is able to fit the data
    random_t = np.random.uniform(0, sde.T // 2, size=n)
    random_t = repeat(random_t, "n -> n 1")
    z = np.random.randn(n)
    z = repeat(z, "n -> n y_dim", y_dim=y_dim)

    mean, std = sde.marginal_prob(y, random_t)
    y_perturbed = mean + z * std

    scores = score_model.score(y=y_perturbed, X=X, t=random_t)
    y_pred = (-1.0) * scores * sigma**2 + y_perturbed

    # Check that the R^2 is close to 1
    r2 = r2_score(y.flatten(), y_pred.flatten())
    assert r2 > 0.95, f"R^2 is {r2}"


@pytest.mark.parametrize("seed", [None, 0])
def test_is_deterministic(seed):
    # Params
    n = 1000
    x_dim = 1
    y_dim = 1
    sigma = 0.00001
    n_estimators = 100
    learning_rate = 0.01
    n_repeats = 10

    X, y = generate_bimodal_linear_regression_data(n, x_dim, sigma, bimodal=False, seed=0)

    assert X.shape == (n, x_dim)
    assert y.shape == (n, y_dim)

    # Fit a score model
    sigma_min = 0.01
    sigma_max = y.std()
    sde = VESDE(sigma_min=sigma_min, sigma_max=sigma_max)

    # First fit
    score_model_a = LightGBMScore(
        sde=sde,
        verbose=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_repeats=n_repeats,
        seed=seed,
    )
    score_model_a.fit(X, y)

    # Second fit
    score_model_b = LightGBMScore(
        sde=sde,
        verbose=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_repeats=n_repeats,
        seed=seed,
    )
    score_model_b.fit(X, y)

    # Check that the two results are the same
    random_t = np.random.uniform(0, sde.T // 2, size=n)
    random_t = repeat(random_t, "n -> n 1")
    z = np.random.randn(n)
    z = repeat(z, "n -> n y_dim", y_dim=y_dim)

    mean, std = sde.marginal_prob(y, random_t)
    y_perturbed = mean + z * std

    scores_a = score_model_a.score(y=y_perturbed, X=X, t=random_t)
    scores_b = score_model_b.score(y=y_perturbed, X=X, t=random_t)

    if seed is None:
        # Check that the score model gives different results
        msg = "The score model is deterministic when it shouldn't be"
        assert not np.allclose(scores_a, scores_b), msg
    else:
        # Check that the score model gives the same results
        msg = "The score model is not deterministic"
        assert np.allclose(scores_a, scores_b), msg
