"""
Abstract classes for sampling methods.
Adapted from: http://tinyurl.com/torch-sampling-song
The notice from the original code is as follows:

 coding=utf-8
 Copyright 2020 The Google Research Authors.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import abc
import functools

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
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """Create a sampling function.

    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers representing the expected shape of a single sample.
      inverse_scaler: The inverse data normalizer function.
      eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    if sampler_name.lower() == "pc":
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(
            sde=sde,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=inverse_scaler,
            snr=config.sampling.snr,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=eps,
        )
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

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
    def update_fn(self, x, t):
        """One update of the predictor.

        Args:
          x: A NumPy array representing the current state
          t: A NumPy array representing the current time step.

        Returns:
          x: A NumPy array of the next state.
          x_mean: A NumPy array. The next state without random noise. Useful for denoising.
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
    def update_fn(self, x, t):
        """One update of the corrector.

        Args:
          x: A NumPy array representing the current state
          t: A NumPy array representing the current time step.

        Returns:
          x: A NumPy array of the next state.
          x_mean: A NumPy array. The next state without random noise. Useful for denoising.
        """


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1.0 / self.rsde.N
        z = np.random.randn(*x.shape)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion.reshape((-1,) + (1,) * (len(z.shape) - 1)) * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = np.random.randn(*x.shape)
        x_mean = x - f
        x = x_mean + G.reshape((-1,) + (1,) * (len(z.shape) - 1)) * z
        return x, x_mean


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x


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

    def update_fn(self, x, t):
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
            grad = score_fn(x, t)
            noise = np.random.randn(*x.shape)
            grad_norm = np.linalg.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.linalg.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size.reshape((-1,) + (1,) * (len(grad) - 1)) * grad
            x = (
                x_mean
                + np.sqrt(step_size * 2).reshape((-1,) + (1,) * (len(noise.shape) - 1)) * noise
            )

        return x, x_mean


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

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas[timestep]
        else:
            alpha = np.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for _ in range(n_steps):
            grad = score_fn(x, t)
            noise = np.random.randn(*x.shape)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size.reshape((-1,) + (1,) * (len(grad.shape) - 1)) * grad
            x = (
                x_mean
                + np.sqrt(step_size * 2).reshape((-1,) + (1,) * (len(noise.shape) - 1)) * noise
            )

        return x, x_mean


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = utils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = utils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)


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
      shape: A sequence of integers. The expected shape of a single sample.
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
        x = sde.prior_sampling(shape)
        timesteps = np.linspace(sde.T, eps, sde.N)

        for i in range(sde.N):
            t = timesteps[i]
            vec_t = np.ones(shape[0]) * t
            x, x_mean = corrector_update_fn(x, vec_t, model=model)
            x, x_mean = predictor_update_fn(x, vec_t, model=model)

        return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

    return pc_sampler
