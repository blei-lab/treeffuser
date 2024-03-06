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


def compare_accuracy(
    y_preds: Dict[str, Float[ndarray, "batch y_dim"]], y_true: Float[ndarray, "batch y_dim"]
) -> Dict[str, Dict[str, float]]:
    """
    Computes error metrics for different prediciton methods.

    Args:
        y_preds: A dictionary where keys are method names (str) and values
                    are ndarrays with predicted means. Each prediction
                    array should match the shape of `y_true`.
        y_true: ndarray of true values.

    Returns:
        A dictionary where each key is a method name corresponding to those
        in `y_preds` and each value is another dictionary containing the accuracy
        metrics.
    """

    metrics = {method: compute_accuracy(y, y_true) for method, y in y_preds.items()}
    return metrics


def compute_accuracy(
    y_pred: Float[ndarray, "batch y_dim"],
    y_true: Float[ndarray, "batch y_dim"],
) -> Dict[str, float]:
    """
    Computes prediction error metrics.

    Args:
        y_pred: ndarray of predicted means.
        y_true: ndarray of true values.

    Returns:
        A dictionary with Mean average error ('mae'), Root mean squared
        error ('rmse'), Median absolute error ('mdae'),  Mean absolute
        relative percent difference ('marpd'), r^2 ('r2'), and Pearson's
        correlation coefficient ('corr').
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("Mismatch in shape between predicted and true values.")

    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mdae = median_absolute_error(y_true, y_pred)
    residuals = y_true - y_pred
    marpd = np.abs(2 * residuals / (np.abs(y_pred) + np.abs(y_true))).mean() * 100
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mdae": mdae,
        "marpd": marpd,
    }

    y_dim = y_pred.shape[1]
    if y_dim == 1:
        r2 = r2_score(y_true, y_pred)
        corr = np.corrcoef(y_true.T, y_pred.T)[0, 1]
        metrics.update(
            {
                "r2": r2,
                "corr": corr,
            }
        )

    return metrics
