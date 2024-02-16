"""
Abstract classes for sampling methods.
Adapted from: http://tinyurl.com/torch-sde-lib-song
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

import numpy as np

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
    return _PREDICTORS[name]


def get_corrector(name):
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
