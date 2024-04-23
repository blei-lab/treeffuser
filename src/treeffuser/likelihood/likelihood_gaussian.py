from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from sklearn.model_selection import train_test_split

from treeffuser import LightGBMTreeffuser
from treeffuser.likelihood.utils import _compute_gaussian_likelihood
from treeffuser.likelihood.utils import _compute_log_prior_T
from treeffuser.likelihood.utils import _compute_score_divergence_numerical
from treeffuser.likelihood.utils import _diffuse_via_probability_flow
from treeffuser.likelihood.utils import _generate_data
from treeffuser.likelihood.utils import _integrate_divergence_pflow_derivative

std_x = 1


def test_ode_based_nll(
    n_steps=10**3, use_treeffuser=False, press_key_for_next=False, empirical_prior=True
):
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

    # # Some useful plots
    # _compare_divergence(
    #     y_range=(y_test.min(), y_test.max()),
    #     x=X_test[0, :].reshape(1, X_test.shape[1]),
    #     t_min=0.01,
    #     model=model,
    #     std_x=std_x,
    # )

    # _compare_score(
    #     y_range=(y_test.min(), y_test.max()),
    #     x=X_test[0, :].reshape(1, X_test.shape[1]),
    #     t_min=0.01,
    #     model=model,
    #     std_x=std_x,
    # )

    # _validate_divergence(
    #     y_range=(y_test.min(), y_test.max()),
    #     x=X_test[0, :].reshape(1, X_test.shape[1]),
    #     t_min=0.01,
    #     model=model,
    #     std_x=std_x,
    # )

    compute_nll_from_ode_gaussian(
        X_test,
        y_test,
        model,
        std_x,
        n_steps,
        use_treeffuser,
        press_key_for_next,
        empirical_prior,
    )


def compute_nll_from_ode_gaussian(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    model: LightGBMTreeffuser,
    std_x: float = 1,
    n_steps: int = 10**4,
    use_treeffuser: bool = False,
    press_key_for_next: bool = False,
    empirical_prior: bool = False,
):
    y_dim = y.shape[1]
    x_dim = X.shape[1]

    ite = -1
    for y0, x in zip(y, X):
        y0 = y0.reshape(1, y_dim)
        x = x.reshape(1, x_dim)

        # transform data
        if model.transform_data:
            raise Warning("Data have been transformed.")
        y0_new = (
            model._y_preprocessor.transform(y0.reshape(1, y_dim))
            if model.transform_data
            else y0
        )
        x_new = (
            model._x_preprocessor.transform(x.reshape(1, x_dim)) if model.transform_data else x
        )

        ite += 1
        print(f"{'#' * 20}")
        print(f"{ite} of {len(y)-1}")
        print(f"y0={y0}")
        print(f"x={x}")

        if ite != 2 and ite != 6:
            print("Skipping ite!")
            continue

        # set discretization parameters
        dt = 1 / n_steps

        # diffuse y0 via probability flow ODE
        y_diffused = _diffuse_via_probability_flow(
            y0_new, x_new, dt, model=model, std_x=std_x, use_treeffuser=use_treeffuser
        )

        # compute likelihood via instantaneous change of variable formula
        integral = _integrate_divergence_pflow_derivative(
            y_diffused,
            x,
            dt,
            model=model,
            std_x=std_x,
            use_treeffuser=use_treeffuser,
        )
        log_p_T = _compute_log_prior_T(
            y_diffused[-1],
            y0,
            model,
            use_treeffuser=use_treeffuser,
            empirical=empirical_prior,
            dt=dt,
        )
        log_p_0 = log_p_T + integral

        if model.transform_data:
            log_p_0 = log_p_0 + np.log(
                model._y_preprocessor._scaler.scale_
            )  # rescale log likelihood

        if use_treeffuser:
            print(f"-log_p_0_treeffuser_ode: {-log_p_0}")
            print(
                f"-log_p_0_treeffuser_sample_pflow: {model.compute_nll(x, y0, sample=True, probability_flow=True, n_samples=1000)}"
            )
            print(
                f"-log_p_0_treeffuser_sample_reverseSDE: {model.compute_nll(x, y0, sample=True, probability_flow=False, n_samples=1000)}"
            )
        else:
            print(f"-log_p_0_ode_true={-log_p_0}")
        print(
            f"-log_p_0_theory_VESDE: {-_compute_gaussian_likelihood(y0_new, loc=x_new, scale=std_x, log=True)}"
        )

        if press_key_for_next:
            input("press key for next point in test set")


########################################### Utilities for some useful plots ###########################################
def _compare_score(
    y_range: Tuple[float, float],
    x: Float[np.ndarray, "1 x_dim"],
    t_min: float,
    model: LightGBMTreeffuser,
    std_x: float,
):
    """
    Compares true and learnt scores and their divergences for Gaussian data under VESDE.
    """

    # define score function
    def _score_estimate(ys, x, t):
        score = []
        for y in ys:
            score_temp = model._score_model.score(y.reshape(1, 1), x, t.reshape(1, 1))
            score.append(score_temp.item())
        return np.array(score)

    def _score_true(ys, x, t):
        score = []
        mu_x = x
        mu_t = 1
        for y in ys:
            _, std_t = model._sde.get_mean_std_pt_given_y0(y, t)
            score_temp = -(y - mu_t * mu_x) / (std_t**2 + mu_t**2 * std_x**2)
            score.append(score_temp.item())
        return np.array(score)

    ts = np.linspace(t_min, 1, 4)
    ys = np.linspace(y_range[0], y_range[1], 1000)

    # make plots
    fig, axs = plt.subplots(2, int(len(ts) / 2), figsize=(15, 3))
    fig.suptitle(f"Score of y_t given x = {x.item():.2f}")
    axs = axs.flatten()

    for i, t in enumerate(ts):
        axs[i].plot(ys, _score_true(ys, x, t), label="ground truth")
        axs[i].plot(ys, _score_estimate(ys, x, t), label="treeffuser")
        axs[i].set_title(f"t = {t:.2f}")
        axs[i].set_xlabel("y")

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower center")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # adjust the top margin to fit the suptitle
    plt.show()


def _compare_divergence(
    y_range: Tuple[float, float],
    x: Float[np.ndarray, "1 x_dim"],
    t_min: float,
    model: LightGBMTreeffuser,
    std_x: float,
):
    """
    Compares true and learnt scores and their divergences for Gaussian data under VESDE.
    """

    # define divergence functions
    def _divergence_estimate(ys, x, t):
        div = []
        for y in ys:
            div_temp = _compute_score_divergence_numerical(
                model._score_model.score, y.reshape(1, 1), x, t.reshape(1, 1), eps=10 ** (-5)
            )
            div.append(div_temp.item())
        return np.array(div)

    def _divergence_true(ys, x, t):
        div = []
        for y in ys:
            mu_t = 1
            _, std_t = model._sde.get_mean_std_pt_given_y0(y.reshape(1, 1), t.reshape(1, 1))
            div_temp = -1 / (std_t**2 + mu_t**2 * std_x**2)
            div.append(div_temp.item())
        return np.array(div)

    ts = np.linspace(t_min, 1, 4)
    ys = np.linspace(y_range[0], y_range[1], 1000)

    # make plots
    fig, axs = plt.subplots(2, int(len(ts) / 2), figsize=(15, 3))
    fig.suptitle(f"Score divergence of y_t given x = {x.item():.2f}")
    axs = axs.flatten()

    for i, t in enumerate(ts):
        axs[i].plot(ys, _divergence_true(ys, x, t), label="ground truth")
        axs[i].plot(ys, _divergence_estimate(ys, x, t), label="treeffuser")
        axs[i].set_title(f"t = {t:.2f}")
        axs[i].set_xlabel("y")

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower center")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # adjust the top margin to fit the suptitle
    plt.show()


def _validate_divergence(
    y_range: Tuple[float, float],
    x: Float[np.ndarray, "1 x_dim"],
    t_min: float,
    model: LightGBMTreeffuser,
    std_x: float,
):
    # define divergence and score functions
    def _divergence_estimate(ys, x, t):
        div = []
        for y in ys:
            div_temp = _compute_score_divergence_numerical(
                model._score_model.score, y.reshape(1, 1), x, t.reshape(1, 1), eps=10 ** (-5)
            )
            div.append(div_temp.item())
        return np.array(div)

    def _score_estimate(ys, x, t):
        score = []
        for y in ys:
            score_temp = model._score_model.score(y.reshape(1, 1), x, t.reshape(1, 1))
            score.append(score_temp.item())
        return np.array(score)

    ts = np.linspace(t_min, 1, 4)
    ys = np.linspace(y_range[0], y_range[1], 1000)

    # make plots
    fig, axs = plt.subplots(2, int(len(ts) / 2), figsize=(15, 3))
    fig.suptitle(f"Score function and its divergence given x = {x.item():.2f}")
    axs = axs.flatten()

    for i, t in enumerate(ts):
        ax1 = axs[i]
        ax1.plot(ys, _score_estimate(ys, x, t), "b-", label="score")
        ax1.set_title(f"t = {t:.2f}")
        ax1.set_xlabel("y")
        ax1.tick_params(axis="y", labelcolor="b")

        ax1_twin = ax1.twinx()
        ax1_twin.plot(ys, _divergence_estimate(ys, x, t), "r-", label="divergence")
        ax1.tick_params(axis="y", labelcolor="r")

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower center")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # adjust the top margin to fit the suptitle
    plt.show()


#######################################################################################################################

test_ode_based_nll(
    n_steps=10**3, use_treeffuser=True, press_key_for_next=True, empirical_prior=True
)
