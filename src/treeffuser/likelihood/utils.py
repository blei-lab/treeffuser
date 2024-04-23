from typing import Callable
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from scipy.integrate import solve_ivp
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from treeffuser import LightGBMTreeffuser

std_x = 1


def _generate_data(n, std_x=1, given_x=None):
    rng = np.random.default_rng(seed=0)
    if given_x:
        return rng.normal(loc=given_x, scale=std_x, size=(n, 1))
    else:
        X = rng.uniform(-1, 1, size=(n, 1))
        y = rng.normal(loc=X, scale=std_x, size=(n, 1))
        return y, X


def _compute_gaussian_likelihood(x, loc, scale, log=True):
    out = -0.5 * np.log(2 * np.pi * scale**2) - 0.5 * ((x - loc) / scale) ** 2
    return out.sum() if log else np.exp(out.sum())


def _score(y, x, t, model, std_x, use_treeffuser):
    """
    True score function for Gaussian data under VESDE.
    """
    if use_treeffuser:
        score = model._score_model.score(
            y=y.reshape((1, -1)), X=x.reshape((1, -1)), t=np.array(t).reshape(1, 1)
        )
    else:
        mu_x = x
        mu_t = 1
        _, std_t = model._sde.get_mean_std_pt_given_y0(y, t)
        score = -(y - mu_t * mu_x) / (std_t**2 + mu_t**2 * std_x**2)
    return score


def _score_divergence(
    y: Float[np.ndarray, "1 y_dim"],
    x: Float[np.ndarray, "1 x_dim"],
    t: Float[np.ndarray, "1 1"],
    model: LightGBMTreeffuser,
    std_x: float,
    use_treeffuser=False,
):
    """
    When use_treeffuser=False, returns the true score divergence for Gaussian data under VESDE.
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


def _compute_score_divergence_numerical(score_fn, y, x, t, eps=10 ** (-5)):
    """
    Temporary function for numerical divergence for debugging.
    """
    div = (
        score_fn((y + eps).reshape(y.shape), x, t)
        - score_fn(y - eps, x, t)  # centered differences
    ) / (2 * eps)
    return div.reshape(-1)


def _probability_flow(y, x, t, model, score_fn):
    """
    Derivative function defining the probability flow ODE.
    """
    drift, diffusion = model._sde.drift_and_diffusion(y, t)
    return (drift - 0.5 * diffusion**2 * score_fn(y=y, x=x, t=t)).reshape(-1)


def _diffuse_via_probability_flow(
    y0,
    x,
    dt,
    model,
    std_x,
    use_treeffuser=False,
    use_exact_solution=False,
    use_scipy=False,
    plot_ode_problem=False,
):
    """
    Diffuse observation via probability flow ODE for Gaussian data under VESDE.

    If use_treeffuser is False:
    - and use_exact_solution is True, it diffuses y(0)=y0 using the closed form solution for y(t).
    - and use_exact_solution is False, it diffuse y(0)=y0 using the discretized ODE via the true score fuction.
    """
    timestamps = np.arange(0.01, model._sde.T, dt)

    y_t = [y0.reshape(-1)]
    if not use_treeffuser and use_exact_solution:  # diffuse via exact solution
        mu_x = x.reshape(-1)
        mu_t = 1
        for s in timestamps:
            _, std_t = model._sde.get_mean_std_pt_given_y0(y0, s)
            y = (y0 - mu_t * mu_x) * np.sqrt(
                (std_t**2 + mu_t**2 * std_x**2) / (mu_t**2 * std_x**2)
            ) + mu_t * mu_x
            y_t.append(y.reshape(-1))
    else:

        def _score_fn(y, x, t):
            return _score(y, x, t, model, std_x, use_treeffuser)

        if use_scipy:

            def _probability_flow_scipy_helper(t, y):
                return _probability_flow(
                    y.reshape(1, -1),
                    x=x,
                    t=np.array(t).reshape(1, 1),
                    model=model,
                    score_fn=_score_fn,
                )

            if plot_ode_problem:
                _plot_probability_flow_ODE_derivative(
                    _probability_flow_scipy_helper,
                    y0=y0.item(),
                    n_steps=len(timestamps),
                    model=model,
                )

            print("Scipy: The solver started working on the probability flow ODE.")
            y_t = solve_ivp(
                fun=_probability_flow_scipy_helper,
                t_span=[timestamps[0], timestamps[-1]],
                y0=y0.reshape(-1),
                method="BDF",
                t_eval=timestamps,
            )
            print(f"Scipy: {y_t['message']}")
            y_t = y_t["y"].T
        else:
            for s in timestamps:
                y = (
                    y_t[-1]
                    + _probability_flow(
                        y_t[-1].reshape(1, -1), x, s.reshape(1, 1), model, _score_fn
                    )
                    * dt
                )
                y_t.append(y)
    return np.array(y_t)


def _integrate_divergence_pflow_derivative(
    y_diffused: Float[np.ndarray, "n_steps y_dim"],
    x: Float[np.ndarray, "x_dim"],
    dt: float,
    model: LightGBMTreeffuser,
    std_x: float,
    use_treeffuser: False,
):
    y_dim = y_diffused.shape[1]  # ys.shape[1]
    timestamps = np.arange(0.01, model._sde.T, dt)

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


def _compute_log_prior_T(
    y,
    x,
    model: LightGBMTreeffuser,
    use_treeffuser: bool,
    empirical: bool = False,
    dt: Optional[float] = None,
):
    if empirical:
        n = 10**3
        y0_sims = _generate_data(n, std_x=std_x, given_x=x)
        yT_sims = []
        for y0 in tqdm(y0_sims):
            yT_temp = _diffuse_via_probability_flow(
                y0, x, dt=dt, model=model, std_x=std_x, use_treeffuser=use_treeffuser
            )[-1]
            yT_sims.append(yT_temp)
        yT_sims = np.sort(yT_sims, axis=0)

        kde = KernelDensity(bandwidth=1.0, algorithm="auto", kernel="gaussian")
        kde.fit(yT_sims)

        # plot empirical distribution vs SDE prior
        plt.figure()
        plt.hist(yT_sims, bins=30, density=True)
        prior_density = np.array(
            [
                model._sde.get_likelihood_theoretical_prior(yT_sim, log=False)
                for yT_sim in yT_sims
            ]
        )
        plt.plot(yT_sims, prior_density, label=f"SDE prior at T={model._sde.T}")
        plt.plot(
            yT_sims,
            np.exp([kde.score(yT_sim.reshape(1, -1)) for yT_sim in yT_sims]),
            label="KDE estimate",
        )
        plt.legend()
        plt.show()

        return kde.score(y.reshape(1, -1))

    if model.sde_name.lower == "vesde" and not use_treeffuser:
        x_dim = x.shape[1]
        mu_t = 1
        _, std_t = model._sde.get_mean_std_pt_given_y0(y.reshape(1, -1), model._sde.T)
        log_p_T = _compute_gaussian_likelihood(
            y,
            loc=x.reshape(x_dim),
            scale=np.sqrt(std_t**2 + mu_t**2 * std_x**2),
            log=True,
        )
    else:
        log_p_T = model._sde.get_likelihood_theoretical_prior(y)
    return log_p_T


def _plot_probability_flow_ODE_derivative(
    probability_flow_fn: Callable, y0: float, n_steps: int, model: LightGBMTreeffuser
):
    t_values = np.concatenate((np.linspace(0.01, 0.5, 5), np.linspace(0.51, model._sde.T, 20)))
    y_values = np.linspace(-5 * np.abs(y0), 5 * abs(y0), 20)
    T, Y = np.meshgrid(t_values, y_values)
    F = np.zeros_like(T)

    # Compute f(t, y) for each element in T and Y
    for i in tqdm(range(T.shape[0]), desc="Plotting pflow problem"):
        for j in range(T.shape[1]):
            F[i, j] = probability_flow_fn(T[i, j], Y[i, j]).item()

    # Create a 3D plot to visualize the derivatives
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T, Y, F, cmap="viridis")

    # Add labels and title
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title("Probability flow ODE as a function of (t, y)")

    # Add a color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()
