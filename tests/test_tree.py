import numpy as np
from sklearn.model_selection import train_test_split

from treeffuser import LightGBMTreeffuser
from treeffuser._tree import _compute_prediction_divergence
from treeffuser._tree import _predict


def test_score():
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
        learning_rate=0.5,
        early_stopping_rounds=20,
        seed=0,
        linear_tree=True,
    )
    model.fit(X_train, y_train)

    # def _score_fn(y, x, t):
    #     return model._score_model.score(
    #         y=y.reshape(1, y_dim), X=x.reshape(1, x_dim), t=t.reshape(1, 1)
    #     )

    score_dict = model._dump_model()

    for i in range(y_test.shape[0]):
        t_test = rng.uniform(size=1)

        score_ligthgbm = model._score_model.score(
            y_test[i, :].reshape((1, -1)), X_test[i, :].reshape((1, -1)), t_test.reshape(1, 1)
        ).item()

        _, std = model._sde.get_mean_std_pt_given_y0(y_test[i, :].reshape(-1), t_test)
        score_from_dump_model = (
            _predict(
                score_dict,
                y_test[i, :].reshape(-1),
                X_test[i, :].reshape(-1),
                t_test,
            )
            / std
        ).item()

        assert (
            np.abs(score_ligthgbm / score_from_dump_model - 1) < 1e-5
        ), f"LightGBM score: {score_ligthgbm}, \nScore from dump_model(): {score_from_dump_model}"


def test_divergence():
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
        learning_rate=0.1,
        early_stopping_rounds=20,
        seed=0,
        linear_tree=True,
    )
    model.fit(X_train, y_train)

    def _score_fn(y, x, t):
        return model._score_model.score(
            y=y.reshape(1, y_dim), X=x.reshape(1, x_dim), t=t.reshape(1, 1)
        )

    def _score_fn_divergence_numerical(y, x, t, eps=10 ** (-5)):
        return (_score_fn(y + eps, x, t) - _score_fn(y - eps, x, t)) / (2 * eps)

    score_dict = model._dump_model()

    for i in range(y_test.shape[0]):
        t_test = rng.uniform(size=1)

        divergence_num = _score_fn_divergence_numerical(
            y_test[i, :].reshape(-1), X_test[i, :].reshape(-1), t_test
        ).item()

        _, std = model._sde.get_mean_std_pt_given_y0(y_test[i, :].reshape(-1), t_test)
        divergence_from_dump_model = (
            _compute_prediction_divergence(
                score_dict,
                y_test[i, :].reshape(-1),
                X_test[i, :].reshape(-1),
                t_test,
            )
            / std.item()
        )

        assert (
            np.abs(divergence_num / divergence_from_dump_model - 1) < 1e-5
        ), f"Numerical divergence: {divergence_num}, \nDivergence from dump_model(): {divergence_from_dump_model}"
