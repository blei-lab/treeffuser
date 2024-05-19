"""
Module for computing metrics.
Adapted from: https://tinyurl.com/uncertainty-toolbox-accuracy
"""

from typing import Dict

import numpy as np
from jaxtyping import Float
from numpy import ndarray
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

from testbed.metrics.base_metric import Metric
from testbed.models.base_model import ProbabilisticModel


class AccuracyMetric(Metric):
    """
    Computes prediction error metrics.
    """

    def compute(
        self,
        model: ProbabilisticModel,
        X_test: Float[ndarray, "batch n_features"],
        y_test: Float[ndarray, "batch y_dim"],
    ) -> Dict[str, float]:
        """
        Computes prediction error metrics.

        Parameters:
        ----------
        model: ProbabilisticModel
            The model to evaluate.
        X_test: ndarray of shape (batch, n_features)
            The input data.
        y_test: ndarray of shape (batch, y_dim)
            The true output values.

        Returns:
        -------
        metrics: dict
            A dictionary containing the following metrics:
            - `mae`: Mean average error.
            - `rmse`: Root mean squared error.
            - `mdae`: Median absolute error.
            - `marpd`: Mean absolute relative percent difference.
        """
        y_pred = model.predict(X_test)

        # Check shapes
        if y_pred.shape != y_test.shape:
            raise ValueError(
                f"Mismatch in shape between predicted and true values: {y_pred.shape} != {y_test.shape}"
            )

        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mdae = median_absolute_error(y_test, y_pred)
        residuals = y_test - y_pred
        marpd = np.abs(2 * residuals / (np.abs(y_pred) + np.abs(y_test))).mean() * 100
        metrics = {"mae": mae, "rmse": rmse, "mdae": mdae, "marpd": marpd}

        y_dim = y_pred.shape[1]
        if y_dim == 1:
            r2 = r2_score(y_test, y_pred)
            corr = np.corrcoef(y_test.T, y_pred.T)[0, 1]
            metrics.update({"r2": r2, "corr": corr})

        return metrics
