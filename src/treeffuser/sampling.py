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
from typing import Literal

import numpy as np

import treeffuser.sde as sde_lib
import treeffuser.utils as utils

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


def get_sampling_fn(
    predictor_name: Literal["euler_maruyama", "reverse_diffusion", "none"],
    corrector_name: Literal["langevin", "ald", "none"],
    snr: float,
    n_steps: int,
    sde: sde_lib.SDE,
    inverse_scaler,
    probability_flow: bool = False,
    denoise: bool = True,
    eps: float = 1e-3,
):
    """Create a sampling function.

    Args:
        predictor_name: A string representing the name of the predictor algorithm.
        corrector_name: A string representing the name of the corrector algorithm.
        snr: The signal-to-noise ratio for configuring correctors.
        n_steps: The number of corrector steps per predictor update.
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers representing the eypected shape of a single sample.
        denoise: If `True`, add one-step denoising to the final samples.
        inverse_scaler: The inverse data normalizer function.
        eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """
    predictor = get_predictor(predictor_name)
    corrector = get_corrector(corrector_name)

    sampling_fn = get_pc_sampler(
        sde=sde,
        predictor=predictor,
        corrector=corrector,
        inverse_scaler=inverse_scaler,
        snr=snr,
        n_steps=n_steps,
        probability_flow=probability_flow,
        denoise=denoise,
        eps=eps,
    )

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, y, t):
        """One update of the predictor.

        Args:
          y: A NumPy array representing the current state
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
    def update_fn(self, y, t):
        """One update of the corrector.

        Args:
          y: A NumPy array representing the current state
          t: A NumPy array representing the current time step.

        Returns:
          y: A NumPy array of the neyt state.
          y_mean: A NumPy array. The neyt state without random noise. Useful for denoising.
        """


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, y, t):
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

    def update_fn(self, y, t):
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

    def update_fn(self, y, t):
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

    def update_fn(self, y, t):
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
            grad = score_fn(y, t)
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

    def update_fn(self, y, t):
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
            grad = score_fn(y, t)
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

    def update_fn(self, y, t):
        return y, y


def shared_predictor_update_fn(y, t, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = utils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(y, t)


def shared_corrector_update_fn(y, t, sde, model, corrector, continuous, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = utils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(y, t)


def get_pc_sampler(
    sde,
    shape,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-3,
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The eypected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def pc_sampler(model):
        """The PC sampler funciton.

        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        # Initial sample
        y = sde.prior_sampling(shape)
        timesteps = np.linspace(sde.T, eps, sde.N)

        for i in range(sde.N):
            t = timesteps[i]
            vec_t = np.ones(shape[0]) * t
            y, y_mean = corrector_update_fn(y, vec_t, model=model)
            y, y_mean = predictor_update_fn(y, vec_t, model=model)

        return inverse_scaler(y_mean if denoise else y), sde.N * (n_steps + 1)

    return pc_sampler
