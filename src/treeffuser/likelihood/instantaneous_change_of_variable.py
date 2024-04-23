import numpy as np
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from treeffuser.likelihood.utils import _compute_log_prior_T
from treeffuser.likelihood.utils import _diffuse_via_probability_flow
from treeffuser.likelihood.utils import _generate_data
from treeffuser.likelihood.utils import _integrate_divergence_pflow_derivative

std_x = 1


def _estimate_distribution_via_pflow(
    initial_samples, x, direction, dt, model, n_samples=10**2
):
    sign = 1 if direction == "forward" else -1

    # generate samples from prior via forward pflow
    y_target = []
    for y_ini in tqdm(initial_samples):
        y_temp = (
            sign
            * _diffuse_via_probability_flow(
                y_ini, x, dt=dt, model=model, std_x=std_x, use_treeffuser=True
            )[-1]
        )
        y_target.append(y_temp)
    y_target = np.sort(y_target, axis=0)

    # estimate prior density
    kde = KernelDensity(bandwidth=1.0, algorithm="auto", kernel="gaussian")
    kde.fit(y_target)

    return kde


def _reverse_instantaneous_change_of_variable_formula(y, x, model, n_steps, n_samples=10**2):
    dt = 1 / n_steps

    y0_sims = _generate_data(n_samples, std_x=std_x, given_x=x)

    prior = _estimate_distribution_via_pflow(
        y0_sims, x, direction="forward", dt=dt, model=model, n_samples=n_samples
    )

    yT_sims = prior.sample(n_samples=n_samples)

    p0 = _estimate_distribution_via_pflow(
        yT_sims, x, direction="backward", dt=dt, model=model, n_samples=n_samples
    )

    return p0.score(y.reshape(1, -1))


def _instantaneous_change_of_variable_formula(y, x, model, n_steps):
    # set discretization parameters
    dt = 1 / n_steps

    # diffuse y0 via probability flow ODE
    y_diffused = _diffuse_via_probability_flow(
        y, x, dt, model=model, std_x=std_x, use_treeffuser=True
    )

    # compute likelihood via instantaneous change of variable formula
    integral = _integrate_divergence_pflow_derivative(
        y_diffused,
        x,
        dt,
        model=model,
        std_x=std_x,
        use_treeffuser=True,
    )
    log_p_T = _compute_log_prior_T(
        y_diffused[-1],
        y,
        model,
        use_treeffuser=True,
        empirical=True,
        dt=dt,
    )

    return log_p_T + integral
