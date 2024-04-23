from sklearn.model_selection import train_test_split

from treeffuser import LightGBMTreeffuser
from treeffuser.likelihood.instantaneous_change_of_variable import (
    _instantaneous_change_of_variable_formula,
)
from treeffuser.likelihood.instantaneous_change_of_variable import (
    _reverse_instantaneous_change_of_variable_formula,
)
from treeffuser.likelihood.utils import _generate_data


def main(n_steps=10**3):
    n = 10**4

    std_x = 1
    y, X = _generate_data(n, std_x=1)

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

    for i, (y_i_test, x_i_test) in enumerate(zip(y_test, X_test)):
        print(f"{'#' * 20}")
        print(f"{i} of {len(y_test)-1}, \nx={x_i_test}, y={y_i_test}")

        print("Computing ICVF...", end="")
        log_p_0 = _instantaneous_change_of_variable_formula(
            y_i_test.reshape(1, -1), x_i_test.reshape(1, -1), model, n_steps
        )
        print("done.")

        print("Computing reverse ICVF...", end="")
        log_p_0_reverse = _reverse_instantaneous_change_of_variable_formula(
            y_i_test.reshape(1, -1), x_i_test.reshape(1, -1), model, n_steps
        )
        print("done.")

        print(f"ICVF: {log_p_0}")
        print(f"Reverse ICVF: {log_p_0_reverse}")


main(n_steps=10**2)
