"""
This file contains the main and only public class for using Card.

Although the API is general a lot of design decisions are based on the
specific structure of the Card model.
"""

import argparse
from argparse import Namespace

from jaxtyping import Float
from numpy import ndarray

from testbed.models.card._card_regression import Diffusion

####################################################
# Helper functions
####################################################


def _dict2namespace(config) -> Namespace:
    """
    Takes a dictionary and returns a namespace object with the same
    structure.

    Example usage:
    ```
    >>  config = {
            "model": {
                "var" : 0.1,
            }
        }
    >>  namespace = _dict2namespace(config)
    >>  print(namespace.model.var)
    0.1
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = _dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def _make_config(
    # Model parameters
    var_type: str,
    type: str,
    data_dim: int,
    x_dim: int,
    y_dim: int,
    z_dim: int,
    cat_x: bool,
    feature_dim: int,
    ema_rate: float,
    ema: int,
    # Beta Schedule
    beta_schedule: str,
    beta_start: float,
    beta_end: float,
    timesteps: int,
    vis_step: int,
    num_figs: int,
    conditioning_signal: str,
    # Non-linear guidance
    pre_train: bool,
    joint_train: bool,
    n_pretrain_epochs: int,
    logging_interval: int,
    hid_layers: int,
    use_batchnorm: bool,
    negative_slope: float,
    dropout_rate: float,
    apply_early_stopping: bool,
    n_pretrain_max_epochs: int,
    train_ratio: int,
    patience: int,
    delta: int,
) -> Namespace:
    """
    Creates a valid config object with the structure required by the
    `Diffusion` class in card.

    The config object is simply a namespace object (which is
    essentially a dictionary with dot access to keys). We use it only
    to mantain compatibility with Card.
    """

    config = {
        "model": {
            "type": type,
            "data_dim": data_dim,
            "x_dim": x_dim,
            "y_dim": y_dim,
            "z_dim": z_dim,
            "cat_x": cat_x,
            "feature_dim": feature_dim,
            "var_type": var_type,
            "ema_rate": ema_rate,
            "ema": ema,
        },
        "diffusion": {
            "beta_schedule": beta_schedule,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "timesteps": timesteps,
            "vis_step": vis_step,
            "num_figs": num_figs,
            "conditioning_signal": conditioning_signal,
            "nonlinear_guidance": {
                "pre_train": pre_train,
                "joint_train": joint_train,
                "n_pretrain_epochs": n_pretrain_epochs,
                "logging_interval": logging_interval,
                "hid_layers": hid_layers,
                "use_batchnorm": use_batchnorm,
                "negative_slope": negative_slope,
                "dropout_rate": dropout_rate,
                "apply_early_stopping": apply_early_stopping,
                "n_pretrain_max_epochs": n_pretrain_max_epochs,
                "train_ratio": train_ratio,
                "patience": patience,
                "delta": delta,
            },
        },
    }

    config: Namespace = _dict2namespace(config)
    return config


####################################################
# Main class
####################################################


class Card:
    """
    Implmenetation of the Card model as presented in
    https://arxiv.org/abs/2206.07275
    """

    def __init__(
        self,
        # Model parameters
        var_type: str = "fixedlarge",
        type: str = "simple",
        data_dim: int = 15,
        x_dim: int = 14,
        y_dim: int = 1,
        z_dim: int = 2,
        cat_x: bool = True,
        feature_dim: int = 128,
        ema_rate: float = 0.9999,
        ema: int = True,
        # Beta Schedule
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        timesteps: int = 1000,
        vis_step: int = 100,
        num_figs: int = 10,
        conditioning_signal: str = "NN",
        # Non-linear guidance
        pre_train: bool = True,
        joint_train: bool = False,
        n_pretrain_epochs: int = 100,
        logging_interval: int = 10,
        hid_layers: int = [100, 50],
        use_batchnorm: bool = False,
        negative_slope: float = 0.01,
        dropout_rate: float = 0.05,
        apply_early_stopping: bool = True,
        n_pretrain_max_epochs: int = 1000,
        train_ratio: int = 0.6,  # for splitting original train into train and validation set for hyperparameter tuning
        patience: int = 50,
        delta: int = 0,  # hyperparameter for improvement measurement in the early stopping scheme
    ):

        # Config settings
        self.var_type = var_type
        self.type = type
        self.data_dim = data_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.cat_x = cat_x
        self.feature_dim = feature_dim
        self.ema_rate = ema_rate
        self.ema = ema

        self._beta_schedule = beta_schedule
        self._beta_start = beta_start
        self._beta_end = beta_end
        self._timesteps = timesteps
        self._vis_step = vis_step
        self._num_figs = num_figs
        self._conditioning_signal = conditioning_signal

        self._pre_train = pre_train
        self._joint_train = joint_train
        self._n_pretrain_epochs = n_pretrain_epochs
        self._logging_interval = logging_interval
        self._hid_layers = hid_layers
        self._use_batchnorm = use_batchnorm
        self._negative_slope = negative_slope
        self._dropout_rate = dropout_rate
        self._apply_early_stopping = apply_early_stopping
        self._n_pretrain_max_epochs = n_pretrain_max_epochs
        self._train_ratio = train_ratio
        self._patience = patience
        self._delta = delta

        # Actual state
        self._diffusion = None  # Diffusion model from card

        super().__init__()

    def fit(self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]):
        """
        Fit the model to the data.
        """
        config = _make_config(
            self.var_type,
            self.type,
            self.data_dim,
            self.x_dim,
            self.y_dim,
            self.z_dim,
            self.cat_x,
            self.feature_dim,
            self.ema_rate,
            self.ema,
            self._beta_schedule,
            self._beta_start,
            self._beta_end,
            self._timesteps,
            self._vis_step,
            self._num_figs,
            self._conditioning_signal,
            self._pre_train,
            self._joint_train,
            self._n_pretrain_epochs,
            self._logging_interval,
            self._hid_layers,
            self._use_batchnorm,
            self._negative_slope,
            self._dropout_rate,
            self._apply_early_stopping,
            self._n_pretrain_max_epochs,
            self._train_ratio,
            self._patience,
            self._delta,
        )
        self._diffusion = Diffusion(config=config, X=X, y=y)

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        """
        Predict the probability distribution for each input.
        """

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples: int
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Sample from the probability distribution for each input.
        """

    def log_likelihood(
        self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]
    ) -> Float[ndarray, "batch"]:
        """
        Compute the log likelihood of the data under the model.
        """
