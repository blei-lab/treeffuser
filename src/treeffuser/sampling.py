"""
Abstract classes for sampling methods.
Adapted from: http://tinyurl.com/torch-sampling-song
The notice from the original code is as follows:

 coding=utf-8
 Copyright 2020 The Google Research Authors.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file eycept in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either eypress or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import abc
import functools
from typing import Callable
from typing import Literal

import numpy as np
from einops import rearrange
from einops import repeat
from jaxtyping import Float
from numpy import ndarray

import treeffuser.sde as sde_lib

_PREDICTORS = {}
_CORRECTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    if name not in _PREDICTORS:
        msg = f"Predictor {name} not found. Available predictors: {list(_PREDICTORS.keys())}"
        raise ValueError(msg)
    return _PREDICTORS[name]


def get_corrector(name):
    if name not in _CORRECTORS:
        msg = f"Corrector {name} not found. Available correctors: {list(_CORRECTORS.keys())}"
        raise ValueError(msg)
    return _CORRECTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(
        self,
        y: Float[ndarray, "batch y_dim"],
        X: Float[ndarray, "batch x_dim"],
        t: Float[ndarray, "batch 1"],
    ):
        """One update of the predictor.

        Args:
          y: A NumPy array representing the current state.
          X: A NumPy array representing the covariates.
          t: A NumPy array representing the current time step.

        Returns:
          y: A NumPy array of the neyt state.
          y_mean: A NumPy array. The neyt state without random noise. Useful for denoising.
        """


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(
        self,
        y: Float[ndarray, "batch y_dim"],
        X: Float[ndarray, "batch x_dim"],
        t: Float[ndarray, "batch 1"],
    ):
        """One update of the corrector.

        Args:
          y: A NumPy array representing the current state.
          X: A NumPy array representing the covariates.
          t: A NumPy array representing the current time step.

        Returns:
          y: A NumPy array of the neyt state.
          y_mean: A NumPy array. The neyt state without random noise. Useful for denoising.
        """


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(
        self,
        y: Float[ndarray, "batch y_dim"],
        X: Float[ndarray, "batch x_dim"],
        t: Float[ndarray, "batch 1"],
    ):
        dt = -1.0 / self.rsde.N
        z = np.random.randn(*y.shape)
        drift, diffusion = self.rsde.sde(y, t)
        y_mean = y + drift * dt
        y = y_mean + diffusion.reshape((-1,) + (1,) * (len(z.shape) - 1)) * np.sqrt(-dt) * z
        return y, y_mean


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(
        self,
        y: Float[ndarray, "batch y_dim"],
        X: Float[ndarray, "batch x_dim"],
        t: Float[ndarray, "batch 1"],
    ):
        f, G = self.rsde.discretize(y, t)
        z = np.random.randn(*y.shape)
        y_mean = y - f
        y = y_mean + G.reshape((-1,) + (1,) * (len(z.shape) - 1)) * z
        return y, y_mean


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(
        self,
        y: Float[ndarray, "batch y_dim"],
        X: Float[ndarray, "batch x_dim"],
        t: Float[ndarray, "batch 1"],
    ):
        return y, y


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(
        self,
        y: Float[ndarray, "batch y_dim"],
        X: Float[ndarray, "batch x_dim"],
        t: Float[ndarray, "batch 1"],
    ):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas[timestep]
        else:
            alpha = np.ones_like(t)

        for _ in range(n_steps):
            grad = score_fn(y, X, t)
            noise = np.random.randn(*y.shape)
            grad_norm = np.linalg.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.linalg.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            y_mean = y + step_size.reshape((-1,) + (1,) * (len(grad) - 1)) * grad
            y = (
                y_mean
                + np.sqrt(step_size * 2).reshape((-1,) + (1,) * (len(noise.shape) - 1)) * noise
            )

        return y, y_mean


@register_corrector(name="ald")
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(
        self,
        y: Float[ndarray, "batch y_dim"],
        X: Float[ndarray, "batch x_dim"],
        t: Float[ndarray, "batch 1"],
    ):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas[timestep]
        else:
            alpha = np.ones_like(t)

        std = self.sde.marginal_prob(y, t)[1]

        for _ in range(n_steps):
            grad = score_fn(y, X, t)
            noise = np.random.randn(*y.shape)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            y_mean = y + step_size.reshape((-1,) + (1,) * (len(grad.shape) - 1)) * grad
            y = (
                y_mean
                + np.sqrt(step_size * 2).reshape((-1,) + (1,) * (len(noise.shape) - 1)) * noise
            )

        return y, y_mean


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(
        self,
        y: Float[ndarray, "batch y_dim"],
        X: Float[ndarray, "batch x_dim"],
        t: Float[ndarray, "batch 1"],
    ):
        return y, y


def shared_predictor_update_fn(y, X, t, sde, score_fn, predictor, probability_flow):
    """A wrapper that configures and returns the update function of predictors."""
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(y, X, t)


def shared_corrector_update_fn(y, X, t, score_fn, sde, corrector, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(y, X, t)


def batch_sample(
    X_batch: Float[np.ndarray, "batch x_dim"],
    y_dim: int,
    sde: sde_lib.SDE,
    eps: float,
    denoise: bool,
    predictor_update_fn: Callable,
    corrector_update_fn: Callable,
):
    timesteps = np.linspace(sde.T, eps, sde.N)

    y = sde.prior_sampling((X_batch.shape[0], y_dim))
    for i in range(sde.N):
        t = timesteps[i]
        vec_t = np.ones((X_batch.shape[0], 1)) * t
        y, y_mean = corrector_update_fn(y, X_batch, vec_t)
        y, y_mean = predictor_update_fn(y, X_batch, vec_t)

    return y_mean if denoise else y


def sample(
    X: Float[np.ndarray, "batch x_dim"],
    y_dim: int,
    n_samples: int,
    score_fn: Callable,
    sde: sde_lib.SDE,
    predictor_name: Literal["euler_maruyama", "reverse_diffusion", "none"],
    corrector_name: Literal["langevin", "ald", "none"],
    snr: float,
    n_steps: int,
    n_batches: int = 10,
    probability_flow: bool = False,
    denoise: bool = True,
    eps: float = 1e-3,
):
    """Sampling.

    Args:
        X: The data to condition on when sampling y.
        y_dim: The dimension of a single sample.
        n_samples: The number of samples to generate.
        score: A callalbe that takes input y[batch, x_dim], X[batch, x_dim] and
            t[batch, 1] and returns the score at the given time.
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        predictor_name: A string representing the name of the predictor algorithm.
        corrector_name: A string representing the name of the corrector algorithm.
        snr: The signal-to-noise ratio for configuring correctors.
        n_steps: The number of corrector steps per predictor update.
        n_batches: The number of batches to sample in parallel. This should be less than or equal to the batch size `X.shape[0]`.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE is only integrated to `eps`
            for numerical stability.
    """
    # For efficiency we will sample multiple samples in parallel
    # and then stack them together. A batch is a single sample.

    total_samples = n_samples * X.shape[0]

    predictor = get_predictor(predictor_name)
    corrector = get_corrector(corrector_name)

    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        score_fn=score_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        score_fn=score_fn,
        sde=sde,
        corrector=corrector,
        snr=snr,
        n_steps=n_steps,
    )

    sampled = 0
    y_samples = np.zeros((X.shape[0], n_samples, y_dim))
    while sampled < total_samples:
        X_batches = repeat(X, "x_dim -> (b x_dim)", b=n_batches)
        y_batch_samples = batch_sample(
            X_batch=X_batches,
            y_dim=y_dim,
            sde=sde,
            eps=eps,
            denoise=denoise,
            predictor_update_fn=predictor_update_fn,
            corrector_update_fn=corrector_update_fn,
        )

        y_batch_samples = rearrange(
            y_batch_samples, "(b n_samples) y_dim -> b n_samples y_dim", b=X.shape[0]
        )
        start, end = sampled, min(sampled + y_batch_samples.shape[1], total_samples)
        y_samples[:, start:end] = y_batch_samples[:, : end - start]
        sampled += y_batch_samples.shape[1]

    return y_samples
