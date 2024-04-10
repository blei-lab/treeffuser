import numpy as np
from jaxtyping import Float
from sklearn.model_selection import train_test_split

from treeffuser import LightGBMTreeffuser

std_x = 1


def _generate_data(n, std_x=1):
    rng = np.random.default_rng(seed=0)
    X = rng.uniform(-1, 1, size=(n, 1))
    y = rng.normal(loc=X, scale=std_x, size=(n, 1))
    return y, X


# def _score_fn(y, x, t, sde, std_x=1):
#     mu_x = x
#     mu_t, std_t = sde.get_mean_std_pt_given_y0(y, t)
#     return -(y - mu_t * mu_x) / (std_t**2 + mu_t**2 * std_x**2)


# def _divergence_fn(y, x, t, sde, std_x=1):
#     mu_t, std_t = sde.get_mean_std_pt_given_y0(y, t)
#     return -1 / (std_t**2 + mu_t**2 * std_x**2)


def test_ode_based_nll():
    n = 10**3
    y_dim = 1

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

    compute_nll_from_ode_gaussian(model, X_test, y_test, std_x)

    nll_sample = model.compute_nll(X_test, y_test, ode=False)
    nll_ode = model.compute_nll(X_test, y_test, ode=True)

    relative_error = np.abs(nll_sample / nll_ode - 1)
    assert relative_error < 0.05, f"relative error: {relative_error}"


def compute_nll_from_ode_gaussian(
    self: LightGBMTreeffuser,
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    std_x: float = 1,
    n_steps: int = 100,
    verbose: bool = False,
):
    y_dim = y.shape[1]
    x_dim = X.shape[1]

    ite = -1
    for y0, x in zip(y, X):  # tuttapposto
        y0 = y0.reshape(1, y_dim)
        x = x.reshape(1, x_dim)

        # transform data
        y0_new = (
            self._y_preprocessor.transform(y0.reshape(1, y_dim)) if self.transform_data else y0
        )
        x_new = (
            self._x_preprocessor.transform(x.reshape(1, x_dim)) if self.transform_data else x
        )

        ite += 1
        print(f"{'#' * 20}")
        print(f"{ite} of {len(y)}")
        print(f"y0={y0}")
        print(f"x={x}")

        # define score helper functions for transformed data
        def _score_fn(y, x, t, std_x=std_x):  # true score function
            mu_x = x
            mu_t, std_t = self._sde.get_mean_std_pt_given_y0(y, t)
            return -(y - mu_t * mu_x) / (std_t**2 + mu_t**2 * std_x**2)

        # define helper function for score divergence
        def _divergence_fn(y, x, t, std_x=1):
            mu_t, std_t = self._sde.get_mean_std_pt_given_y0(y, t)
            return -1 / (std_t**2 + mu_t**2 * std_x**2)

        # diffuse the transformed data via the probability flow
        def _probability_flow(y, t, x=x_new):
            drift, diffusion = self._sde.drift_and_diffusion(y, t)
            return drift - 0.5 * diffusion**2 * _score_fn(y=y, x=x, t=t)

        n_steps = 10**3
        ts = np.arange(0.01, self._sde.T, 1 / n_steps)
        if True:
            y_new = [y0_new]
            for t in ts:
                y_next = y_new[-1] + _probability_flow(y_new[-1], t) / n_steps
                y_new.append(y_next)
            y_new = np.array(y_new).reshape(-1)
            y_new = y_new[1:]
        else:
            y_new = odeint(func=_probability_flow, y0=y0_new.reshape(-1), t=ts).reshape(-1)[1:]

        # compute divergences w.r.t. transformed data
        drift_div_ts = np.array(
            [
                self._sde._get_drift_and_diffusion_divergence(
                    y.reshape(1, y_dim), t.reshape(1, 1)
                )[0]
                for y, t in zip(y_new, ts)
            ]  # y_ts[0] is y(T)
        ).reshape(-1)

        diffusion_coeff_ts = np.array(
            [
                self._sde.drift_and_diffusion(y.reshape(1, y_dim), t.reshape(1, 1))[1]
                for y, t in zip(y_new, ts)
            ]
        ).reshape(-1)

        score_div_ts = np.array(
            [
                # _compute_score_divergence(score_dict, self.learning_rate, y, x, t)
                # _compute_score_divergence_numerical(
                #     _score_fn, y.reshape(1, y_dim), x_new, t.reshape(1, 1)
                # )
                _divergence_fn(y.reshape(1, y_dim), x_new, t.reshape(1, 1))
                for y, t in zip(y_new, ts)
            ]
        ).reshape(-1)

        # compute likelihood of transformed data via the instantaneous change of variable formula
        if self.sde_name.lower == "vesde":
            log_p_T = self._sde.get_likelihood_theoretical_prior(
                y_new[-1].reshape(y_dim),  # y0_new.reshape(y_dim)
            )
        else:
            log_p_T = self._sde.get_likelihood_theoretical_prior(y_new[-1].reshape(y_dim))

        log_p_0 = (
            log_p_T
            + (drift_div_ts - 0.5 * diffusion_coeff_ts**2 * score_div_ts).sum() / n_steps
        )

        if self.transform_data:
            log_p_0 = log_p_0 + np.log(
                self._y_preprocessor._scaler.scale_
            )  # rescale log likelihood

        print(f"log_p_0_ode={-log_p_0}")
        print(
            f"log_p_0_sample: {self.compute_nll(x.reshape(1, x_dim), y0.reshape(1, y_dim), ode=False, n_samples=100)}"
        )


test_ode_based_nll()
