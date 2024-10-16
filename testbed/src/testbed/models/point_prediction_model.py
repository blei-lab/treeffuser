import abc

import numpy as np
from jaxtyping import Float
from numpy import ndarray
from skopt.space import Categorical
from skopt.space import Integer
from skopt.space import Real

from testbed.models import ProbabilisticModel


class PointPredictionModel(ProbabilisticModel, abc.ABC):
    """
    A probabilistic model that predicts a single point estimate from an sklearn model.
    """

    def __init__(self, **hyperparameters):
        super().__init__()
        self.model_hyperparameters = hyperparameters
        self.model = None

    @staticmethod
    @abc.abstractmethod
    def get_model_class():
        """
        Return the model class to use.
        """
        raise NotImplementedError

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ) -> "PointPredictionModel":
        """
        Fit the model to the data.
        """
        model_class = self.get_model_class()
        # filter out hyperparameters that are not valid for the model
        tmp_model_hyperparameters = model_class().get_params()
        valid_hyperparameters = {
            k: v
            for k, v in self.model_hyperparameters.items()
            if k in tmp_model_hyperparameters
        }
        self.model = self.get_model_class()(**valid_hyperparameters)
        self.model.fit(X, y)
        return self

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        """
        Predict the mean for each input.
        """
        predictions = self.model.predict(X)
        if len(predictions.shape) == 1:
            predictions = predictions[:, None]
        return predictions

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10, seed=None
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Sample from the probability distribution for each input.
        """
        single_sample = self.predict(X)
        samples = np.array([single_sample] * n_samples)
        return samples


class PPMLightGBM(PointPredictionModel):
    """
    A probabilistic model that predicts a single point estimate from a LightGBM model.
    """

    @staticmethod
    def get_model_class():
        from lightgbm import LGBMRegressor as LightGBMRegressor

        return LightGBMRegressor

    @staticmethod
    def search_space() -> dict:
        return {
            "n_estimators": Integer(10, 1000),
            "learning_rate": Real(0.01, 0.5, prior="log-uniform"),
            "num_leaves": Integer(10, 100),
            "early_stopping_rounds": Integer(10, 100),
        }


class PPMXGBoost(PointPredictionModel):
    """
    A probabilistic model that predicts a single point estimate from a XGBoost model.
    """

    @staticmethod
    def get_model_class():
        from xgboost import XGBRegressor

        return XGBRegressor

    @staticmethod
    def search_space() -> dict:
        return {
            "n_estimators": Integer(10, 1000),
            "learning_rate": Real(0.01, 0.5, prior="log-uniform"),
            "max_depth": Integer(1, 100),
        }


class PPMMLP(PointPredictionModel):
    """
    A probabilistic model that predicts a single point estimate from a MLP model.
    """

    @staticmethod
    def get_model_class():
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor

    def score(
        self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"], **kwargs
    ) -> float:
        """
        Return the negative RMSE score for the model.
        The higher the score, the better the model.
        This function is used for hyperparameter optimization and
        compatibility with scikit-learn.
        """
        # Avoid circular import
        import testbed.metrics.accuracy as accuracy

        metric = accuracy.AccuracyMetric()
        return -1.0 * metric.compute(self, X, y)["rmse"]

    @staticmethod
    def search_space() -> dict:
        hidden_layer_size = [16, 32, 64, 128]
        n_layers = [1, 2]
        hidden_layer_sizes = []
        for n in n_layers:
            for size in hidden_layer_size:
                hidden_layer_sizes.append((size,) * n)
        return {
            "hidden_layer_sizes": Categorical(hidden_layer_sizes),
            "alpha": Real(0.0001, 0.01, prior="log-uniform"),
            "learning_rate_init": Real(0.001, 0.1, prior="log-uniform"),
        }
