import numpy as np

from treeffuser import LightGBMTreeffuser
from treeffuser._tree import _compute_score_divergence


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
        y_test, x_test, t_test = (
            rng.normal(size=(1, 1)),
            rng.normal(size=(1, 2)),
            rng.uniform(size=1),
        )

        temp_num = _score_fn_divergence_numerical(y_test, x_test, t_test)
        temp_th = _compute_score_divergence(
            score_dict,
            model.learning_rate,
            y_test.reshape(-1),
            x_test.reshape(-1),
            t_test.reshape(-1),
        )

        print(f"temp_num: {temp_num}")
        print(f"temp_th: {temp_th}")

        print("Ready for next comparison.")


test_divergence()
