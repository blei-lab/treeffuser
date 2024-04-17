from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from sklearn.model_selection import train_test_split

from treeffuser import LightGBMTreeffuser
from treeffuser._tree import _compute_score_divergence_numerical

std_x = 1


def _generate_data(n, std_x=1):
    rng = np.random.default_rng(seed=0)
    X = rng.uniform(-1, 1, size=(n, 1))
    y = rng.normal(loc=X, scale=std_x, size=(n, 1))
    return y, X


def _compute_gaussian_likelihood(x, loc, scale, log=True):
    out = -0.5 * np.log(2 * np.pi * scale**2) - 0.5 * ((x - loc) / scale) ** 2
    return out.sum() if log else np.exp(out.sum())


def _score(y, x, t, model, std_x):
    """
    True score function for Gaussian data under VESDE.
    """
    mu_x = x
    mu_t = 1
    _, std_t = model._sde.get_mean_std_pt_given_y0(y, t)
    return -(y - mu_t * mu_x) / (std_t**2 + mu_t**2 * std_x**2)


def _score_divergence(
    y: Float[np.ndarray, "1 y_dim"],
    x: Float[np.ndarray, "1 x_dim"],
    t: Float[np.ndarray, "1 1"],
    model: LightGBMTreeffuser,
    std_x: float,
    use_treeffuser=False,
):
    """
    True score divergence function for Gaussian data under VESDE.
    """
    if use_treeffuser:
        div = _compute_score_divergence_numerical(
            model._score_model.score, y, x, t, eps=10 ** (-5)
        )
    else:
        mu_t = 1
        _, std_t = model._sde.get_mean_std_pt_given_y0(y, t)
        div = -1 / (std_t**2 + mu_t**2 * std_x**2)
    return div


def _probability_flow(y, t, x, model, score_fn):
    """
    Derivative function defining the probability flow ODE.
    """
    drift, diffusion = model._sde.drift_and_diffusion(y, t)
    return drift - 0.5 * diffusion**2 * score_fn(y=y, x=x, t=t)


def _diffuse_via_probability_flow(y0, x, t, model, std_x):
    """
    Diffuse observation via probability flow ODE for Gaussian data under VESDE.
    """
    mu_x = x
    mu_t = 1
    yt = [y0]
    for s in t:
        _, std_t = model._sde.get_mean_std_pt_given_y0(y0, s)
        y = (y0 - mu_t * mu_x) * np.sqrt(
            (std_t**2 + mu_t**2 * std_x**2) / (mu_t**2 * std_x**2)
        ) + mu_t * mu_x
        yt.append(y)
    return np.array(yt)


def _integrate_divergence_pflow_derivative(
    y_diffused: Float[np.ndarray, "n_steps y_dim"],
    timestamps: Float[np.ndarray, "n_steps 1"],
    x: Float[np.ndarray, "x_dim"],
    dt: float,
    model: LightGBMTreeffuser,
    std_x: float,
    use_treeffuser: False,
):
    y_dim = y_diffused.shape[1]  # ys.shape[1]

    integral = 0
    for y, t in zip(y_diffused, timestamps):
        drift_div = model._sde._get_drift_and_diffusion_divergence(
            y.reshape(1, y_dim), t.reshape(1, 1)
        )[0]

        diffusion_coeff = model._sde.drift_and_diffusion(y.reshape(1, y_dim), t.reshape(1, 1))[
            1
        ]

        score_div = _score_divergence(
            y.reshape(1, y_dim),
            x,
            t.reshape(1, 1),
            model=model,
            std_x=std_x,
            use_treeffuser=use_treeffuser,
        )

        integral += (drift_div - 0.5 * diffusion_coeff**2 * score_div).sum()

    integral *= dt
    return integral


def _compute_log_prior_T(y, y0, model: LightGBMTreeffuser):
    y_dim = y0.shape[1]
    if model.sde_name.lower == "vesde":
        hyperparam_max = model._sde.get_hyperparams()["hyperparam_max"]
        log_p_T = _compute_gaussian_likelihood(
            y.reshape(y_dim),
            loc=y0.reshape(y_dim),
            scale=np.sqrt(hyperparam_max**2 + std_x**2),
            log=True,
        )
    else:
        log_p_T = model._sde.get_likelihood_theoretical_prior(y.reshape(y_dim))
    return log_p_T


def test_ode_based_nll():
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

    # Some useful plots
    _compare_divergence(
        y_range=(y_test.min(), y_test.max()),
        x=X_test[0, :].reshape(1, X_test.shape[1]),
        t_min=0.01,
        model=model,
        std_x=std_x,
    )

    _compare_score(
        y_range=(y_test.min(), y_test.max()),
        x=X_test[0, :].reshape(1, X_test.shape[1]),
        t_min=0.01,
        model=model,
        std_x=std_x,
    )

    _validate_divergence(
        y_range=(y_test.min(), y_test.max()),
        x=X_test[0, :].reshape(1, X_test.shape[1]),
        t_min=0.01,
        model=model,
        std_x=std_x,
    )

    compute_nll_from_ode_gaussian(X_test, y_test, model, std_x)


def compute_nll_from_ode_gaussian(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    model: LightGBMTreeffuser,
    std_x: float = 1,
    n_steps: int = 10**4,
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
        print(f"{ite} of {len(y)}")
        print(f"y0={y0}")
        print(f"x={x}")

        # set discretization parameters
        dt = 1 / n_steps
        timestamps = np.arange(0.01, model._sde.T, dt)

        # diffuse y0 via probability flow ODE
        y_diffused = _diffuse_via_probability_flow(
            y0_new, x_new, timestamps, model=model, std_x=std_x
        )

        # compute likelihood via instantaneous change of variable formula
        integral = _integrate_divergence_pflow_derivative(
            y_diffused, timestamps, x, dt, model=model, std_x=std_x, use_treeffuser=False
        )
        log_p_T = _compute_log_prior_T(y_diffused[-1], y0, model)
        log_p_0 = log_p_T + integral

        if model.transform_data:
            log_p_0 = log_p_0 + np.log(
                model._y_preprocessor._scaler.scale_
            )  # rescale log likelihood

        print(f"log_p_0_ode={-log_p_0}")
        print(
            f"log_p_0_theory_VESDE: {-_compute_gaussian_likelihood(y0_new, loc=x_new, scale=std_x, log=True)}"
        )


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

test_ode_based_nll()
