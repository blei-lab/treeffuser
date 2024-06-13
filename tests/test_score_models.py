"""
Contains all of the test for the different score model classes.
"""

import numpy as np
from einops import repeat

from treeffuser._score_models import LightGBMScoreModel
from treeffuser.sde.diffusion_sdes import VESDE

from .utils import generate_bimodal_linear_regression_data
from .utils import r2_score


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
    hyperparam_min = 0.01
    hyperparam_max = y.std()
    sde = VESDE(hyperparam_min=hyperparam_min, hyperparam_max=hyperparam_max)
    score_model = LightGBMScoreModel(
        verbose=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_repeats=n_repeats,
    )
    score_model.fit(X, y, sde)

    # Check that the score model is able to fit the data
    random_t = np.random.uniform(1e-5, sde.T // 2, size=n)
    random_t = repeat(random_t, "n -> n 1")
    z = np.random.randn(n)
    z = repeat(z, "n -> n y_dim", y_dim=y_dim)

    mean, std = sde.get_mean_std_pt_given_y0(y, random_t)
    y_perturbed = mean + z * std

    scores = score_model.score(y=y_perturbed, X=X, t=random_t)
    y_pred = (-1.0) * scores * sigma**2 + y_perturbed

    # Check that the R^2 is close to 1
    r2 = r2_score(y.flatten(), y_pred.flatten())
    assert r2 > 0.95, f"R^2 is {r2}"


def test_can_be_deterministic():
    # Params
    n = 200
    x_dim = 1
    y_dim = 1
    sigma = 0.00001
    n_estimators = 50
    learning_rate = 0.1
    n_repeats = 1

    X, y = generate_bimodal_linear_regression_data(n, x_dim, sigma, bimodal=False, seed=0)

    assert X.shape == (n, x_dim)
    assert y.shape == (n, y_dim)

    # Fit a score model
    hyperparam_min = 0.01
    hyperparam_max = y.std()
    sde = VESDE(hyperparam_min=hyperparam_min, hyperparam_max=hyperparam_max)
    seed = 0

    # First fit
    score_model_a = LightGBMScoreModel(
        verbose=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_repeats=n_repeats,
        seed=seed,
    )
    score_model_a.fit(X, y, sde)

    # Second fit
    score_model_b = LightGBMScoreModel(
        verbose=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_repeats=n_repeats,
        seed=seed,
    )
    score_model_b.fit(X, y, sde)

    # Check that the two results are the same
    random_t = np.random.uniform(1e-5, sde.T // 2, size=n)
    random_t = repeat(random_t, "n -> n 1")
    z = np.random.randn(n)
    z = repeat(z, "n -> n y_dim", y_dim=y_dim)

    mean, std = sde.get_mean_std_pt_given_y0(y, random_t)
    y_perturbed = mean + z * std

    scores_a = score_model_a.score(y=y_perturbed, X=X, t=random_t)
    scores_b = score_model_b.score(y=y_perturbed, X=X, t=random_t)

    msg = "The score model is not deterministic"
    assert np.allclose(scores_a, scores_b), msg


def test_different_seeds_do_not_give_same_results():
    # Params
    n = 200
    x_dim = 1
    y_dim = 1
    sigma = 0.00001
    n_estimators = 50
    learning_rate = 0.1
    n_repeats = 5

    X, y = generate_bimodal_linear_regression_data(n, x_dim, sigma, bimodal=False, seed=0)

    assert X.shape == (n, x_dim)
    assert y.shape == (n, y_dim)

    # Fit a score model
    hyperparam_min = 0.01
    hyperparam_max = y.std()
    sde = VESDE(hyperparam_min=hyperparam_min, hyperparam_max=hyperparam_max)

    seed_a = 0
    seed_b = 1

    # First fit
    score_model_a = LightGBMScoreModel(
        verbose=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_repeats=n_repeats,
        seed=seed_a,
    )
    score_model_a.fit(X, y, sde)

    # Second fit
    score_model_b = LightGBMScoreModel(
        verbose=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_repeats=n_repeats,
        seed=seed_b,
    )
    score_model_b.fit(X, y, sde)

    # Check that the two results are the same
    random_t = np.random.uniform(1e-5, sde.T // 2, size=n)
    random_t = repeat(random_t, "n -> n 1")
    z = np.random.randn(n)
    z = repeat(z, "n -> n y_dim", y_dim=y_dim)

    mean, std = sde.get_mean_std_pt_given_y0(y, random_t)
    y_perturbed = mean + z * std

    scores_a = score_model_a.score(y=y_perturbed, X=X, t=random_t)
    scores_b = score_model_b.score(y=y_perturbed, X=X, t=random_t)

    # Check that the score model gives different results
    msg = "The score model gives the same results for different seeds"
    assert not np.allclose(scores_a, scores_b), msg
