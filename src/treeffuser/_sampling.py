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
from typing import Optional

import numpy as np
from einops import rearrange
from einops import repeat
from jaxtyping import Float
from numpy import ndarray
from tqdm import tqdm

import treeffuser._sdes as _sdes

_PREDICTORS = {}
_CORRECTORS = {}


def _register_predictor(cls=None, *, name=None):
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


def _register_corrector(cls=None, *, name=None):
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


def _get_predictor(name):
    if name not in _PREDICTORS:
        msg = f"Predictor {name} not found. Available predictors: {list(_PREDICTORS.keys())}"
        raise ValueError(msg)
    return _PREDICTORS[name]


def _get_corrector(name):
    if name not in _CORRECTORS:
        msg = f"Corrector {name} not found. Available correctors: {list(_CORRECTORS.keys())}"
        raise ValueError(msg)
    return _CORRECTORS[name]


class _Predictor(abc.ABC):
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


class _Corrector(abc.ABC):
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


@_register_predictor(name="euler_maruyama")
class _EulerMaruyamaPredictor(_Predictor):
    """
    See:
        http://tinyurl.com/wiki-euler-maruyama
    """

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
        drift, diffusion = self.rsde.sde(y, X, t)
        y_mean = y + drift * dt
        y = y_mean + diffusion * np.sqrt(-dt) * z
        return y, y_mean


@_register_predictor(name="reverse_diffusion")
class _ReverseDiffusionPredictor(_Predictor):
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


@_register_predictor(name="none")
class _NonePredictor(_Predictor):
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


@_register_corrector(name="langevin")
class _LangevinCorrector(_Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, _sdes.VPSDE)
            and not isinstance(sde, _sdes.VESDE)
            and not isinstance(sde, _sdes.subVPSDE)
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
        if isinstance(sde, _sdes.VPSDE) or isinstance(sde, _sdes.subVPSDE):
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


@_register_corrector(name="ald")
class _AnnealedLangevinDynamics(_Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, _sdes.VPSDE)
            and not isinstance(sde, _sdes.VESDE)
            and not isinstance(sde, _sdes.subVPSDE)
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
        if isinstance(sde, _sdes.VPSDE) or isinstance(sde, _sdes.subVPSDE):
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


@_register_corrector(name="none")
class _NoneCorrector(_Corrector):
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


def _shared_predictor_update_fn(y, X, t, sde, score_fn, predictor, probability_flow):
    """A wrapper that configures and returns the update function of predictors."""
    predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(y, X, t)


def _shared_corrector_update_fn(y, X, t, score_fn, sde, corrector, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(y, X, t)


def batch_sample(
    X_batch: Float[np.ndarray, "batch x_dim"],
    y_dim: int,
    sde: _sdes.SDE,
    eps: float,
    denoise: bool,
    predictor_update_fn: Callable,
    corrector_update_fn: Callable,
    seed=None,
) -> Float[np.ndarray, "batch y_dim"]:
    if seed is not None:
        np.random.seed(seed)
    timesteps = np.linspace(sde.T, eps, sde.N)

    batch_dim = X_batch.shape[0]
    y = sde.prior_sampling((batch_dim, y_dim))
    for i in range(sde.N):
        t = timesteps[i]
        vec_t = np.ones((batch_dim, 1)) * t
        y, y_mean = corrector_update_fn(y, X_batch, vec_t)
        y, y_mean = predictor_update_fn(y, X_batch, vec_t)

    return y_mean if denoise else y


def sample(
    X: Float[np.ndarray, "n_predictions x_dim"],
    y_dim: int,
    n_samples: int,
    score_fn: Callable,
    sde: _sdes.SDE,
    predictor_name: Literal["euler_maruyama", "reverse_diffusion", "none"],
    corrector_name: Literal["langevin", "ald", "none"],
    snr: Optional[float] = None,
    n_steps: Optional[int] = None,
    n_parallel: int = 10,
    probability_flow: bool = False,
    denoise: bool = True,
    eps: float = 1e-5,
    seed=None,
    verbose: int = 0,
) -> Float[np.ndarray, "n_predictions n_samples y_dim"]:
    """Sampling.

    Args:
        X: The data to condition on when sampling y.
        y_dim: The dimension of a single sample.
        n_samples: The number of samples to generate.
        score: A callalbe that takes input y[batch, x_dim], X[batch, x_dim] and
            t[batch, 1] and returns the score at the given time.
        sde: A `_sdes.SDE` object that represents the forward SDE.
        predictor_name: A string representing the name of the predictor algorithm.
        corrector_name: A string representing the name of the corrector algorithm.
        snr: The signal-to-noise ratio for configuring correctors.
        n_steps: The number of corrector steps per predictor update.
        n_parallel: How many samples of y to draw in parallel.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE is only integrated to `eps`
            for numerical stability.
        verbose: A `int` number. The verbosity level.
    """
    # For efficiency we will sample multiple samples in parallel
    # and then stack them together. A batch is a single sample.
    predictor = _get_predictor(predictor_name)
    corrector = _get_corrector(corrector_name)

    predictor_update_fn = functools.partial(
        _shared_predictor_update_fn,
        score_fn=score_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
    )
    corrector_update_fn = functools.partial(
        _shared_corrector_update_fn,
        score_fn=score_fn,
        sde=sde,
        corrector=corrector,
        snr=snr,
        n_steps=n_steps,
    )

    n_sampled = 0
    n_predictions, _ = X.shape
    y_samples = np.zeros((n_predictions, n_samples, y_dim))

    pbar = tqdm(total=n_samples, disable=verbose == 0)
    while n_sampled < n_samples:
        # This funny code is simple meant to make sampling a bit more efficient by grouping
        # parallel samples and treating them all as a single "batch".
        X_batch = repeat(
            X, "n_predict x_dim -> (n_parallel n_predict) x_dim", n_parallel=n_parallel
        )
        y_batch_samples = batch_sample(
            X_batch=X_batch,
            y_dim=y_dim,
            sde=sde,
            eps=eps,
            denoise=denoise,
            predictor_update_fn=predictor_update_fn,
            corrector_update_fn=corrector_update_fn,
            seed=seed + n_sampled if seed is not None else None,
        )

        y_batch_samples = rearrange(
            y_batch_samples,
            "(n_parallel n_predict) y_dim -> n_predict n_parallel y_dim",
            n_parallel=n_parallel,
        )
        n_samples_batch = y_batch_samples.shape[1]
        start, end = n_sampled, min(n_sampled + n_samples_batch, n_samples)
        y_samples[:, start:end] = y_batch_samples[:, : end - start]
        n_sampled += n_samples_batch

        pbar.update(n_samples_batch)

    return y_samples
