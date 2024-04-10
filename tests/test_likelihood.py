import numpy as np
from sklearn.model_selection import train_test_split

from treeffuser import LightGBMTreeffuser
from treeffuser._tree import _compute_score_divergence


def test_ode_based_nll():
    """
    The data are generated from a Gaussian mixture model with conditional density:
    p(y_i | x_i) = .5 * N(x_i, x_i ** 2) + (1 - .5) * N(-x_i, x_i ** 2)
    """
    n = 10**3
    rng = np.random.default_rng(seed=0)

    y_dim = 1
    x_dim = 2

    X = rng.normal(size=(n, x_dim))
    y = np.sin(X.dot(np.array([2, 1]).reshape(2, 1))) + rng.uniform(
        low=-0.5, high=0.5, size=(n, 1)
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    model = LightGBMTreeffuser(
        verbose=1,
        n_repeats=20,
        n_estimators=10000,
        sde_name="vesde",
        sde_manual_hyperparams={"hyperparam_min": 0.001, "hyperparam_max": 20},
        learning_rate=0.1,
        early_stopping_rounds=20,
        seed=0,
        linear_tree=True,
    )
    model.fit(X_train, y_train, transform_data=False)

    nll_sample = model.compute_nll(X_test, y_test, ode=False)
    nll_ode = model.compute_nll(X_test, y_test, ode=True)

    relative_error = np.abs(nll_sample / nll_ode - 1)
    assert relative_error < 0.05, f"relative error: {relative_error}"


def test_divergence():
    n = 10**3
    rng = np.random.default_rng(seed=0)

    y_dim = 1
    x_dim = 2

    X = rng.normal(size=(n, x_dim))
    y = np.sin(X.dot(np.array([2, 1]).reshape(2, 1))) + rng.uniform(
        low=-0.5, high=0.5, size=(n, 1)
    )

    model = LightGBMTreeffuser(
        verbose=1,
        n_repeats=20,
        n_estimators=10000,
        sde_name="vesde",
        # sde_manual_hyperparams={"hyperparam_min": 0.001, "hyperparam_max": 20},
        learning_rate=0.1,
        early_stopping_rounds=20,
        seed=0,
        linear_tree=True,
    )
    model.fit(X, y)

    def _score_fn(y, x, t):
        return model._score_model.score(
            y=y.reshape(1, y_dim), X=x.reshape(1, x_dim), t=t.reshape(1, 1)
        )

    def _score_fn_divergence_numerical(y, x, t, eps=10 ** (-5)):
        return (_score_fn(y + eps, x, t) - _score_fn(y, x, t)) / eps

    score_dict = model._dump_model()

    for _ in range(10):
        y_test, X_test, t_test = (
            rng.normal(size=(1, 1)),
            rng.normal(size=(1, 2)),
            rng.uniform(size=1),
        )

        temp_num = _score_fn_divergence_numerical(y_test, X_test, t_test)

        _, std = model._sde.get_mean_std_pt_given_y0(y_test, t_test)
        temp_th = (
            _compute_score_divergence(
                score_dict,
                model.learning_rate,
                y_test.reshape(-1),
                X_test.reshape(-1),
                t_test.reshape(-1),
            )
            / std
        )

        print(f"temp_num: {temp_num}")
        print(f"temp_th: {temp_th}")

        print("Ready for next comparison.")


test_ode_based_nll()

# test_divergence()
