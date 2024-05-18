"""
Metrics for measuring calibration of predictive uncertainty of a scalar output.

Adapted from: https://tinyurl.com/uncertainty-toolbox-calibration
"""

import warnings
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from testbed.metrics.base_metric import Metric
from testbed.models.base_model import ProbabilisticModel


class SharpnessFromSamplesMetric(Metric):
    """
    Computes the sharpness of the predictive distribution (i.e. the average standard deviation).
    """

    def __init__(self, n_samples: int = 20):
        super().__init__()
        self.n_samples = n_samples

    def compute(
        self,
        model: ProbabilisticModel,
        X_test: Float[ndarray, "batch n_features"],
        y_test: Float[ndarray, "batch y_dim"],
    ) -> Dict[str, float]:
        """
        Compute the empirical sharpness of the predictive distribution (i.e. the average standard deviation).

        Parameters
        ----------
        model : ProbabilisticModel
            The model to evaluate.
        X_test : ndarray of shape (batch, n_features)
            The input data.
        y_test : ndarray of shape (batch, y_dim)
            The true output values.

        Returns
        -------
        sharpness : dict
            A single scalar which quantifies the average of the standard deviations.
        """
        y_preds = model.sample(X_test, self.n_samples)
        y_stds = np.std(y_preds, axis=0)
        sharpness = np.sqrt(np.mean(y_stds**2))
        return {"sharpness": sharpness}


class QuantileCalibrationErrorMetric(Metric):
    """
    Computes quantile calibration metrics based on samples from the predictive distribution.

    - root mean squared calibration error (RMSCE)
    - mean absolute calibration error (MACE)
    """

    def compute(
        self,
        model: ProbabilisticModel,
        X_test: Float[ndarray, "batch n_features"],
        y_test: Float[ndarray, "batch y_dim"],
    ) -> Dict[str, float]:
        """
        Compute quantile calibration metrics based on samples from the predictive distribution.

        Parameters
        ----------
        model : ProbabilisticModel
            The model to evaluate.
        X_test : ndarray of shape (batch, n_features)
            The input data.
        y_test : ndarray of shape (batch, y_dim)
            The true output values.

        Returns
        -------
        metrics : dict
            Dictionary with keys:
            - "rmsce": root mean squared calibration error
            - "mace": mean absolute calibration error
        """
        y_preds = model.sample(X_test, n_samples=50, seed=self.seed)
        metrics = _compute_quantile_calibration_error(y_preds, y_test)
        return metrics


def _compute_quantile_calibration_error(
    y_preds: Float[ndarray, "n_samples batch y_dim"],
    y_true: Float[ndarray, "batch y_dim"],
) -> Dict[str, float]:
    """
    Compute quantile calibration metrics based on samples from the predictive distribution.

    - root mean squared calibration error (RMSCE)
    - mean absolute calibration error (MACE)

    Parameters
    ----------
    y_preds : ndarray of shape (n_samples, batch, y_dim)
        ndarray of `n_samples` of `y` from the predictive distribution, for a batch of data.
    y_true : ndarray of shape (batch, y_dim)
        True `y` values for the batch of data.

    Returns
    -------
    metrics : dict
        Dictionary with keys:
        - "rmsce": root mean squared calibration error
        - "mace": mean absolute calibration error


    """

    empirical_quantiles = np.mean(y_true <= y_preds, axis=0)
    empirical_quantiles = np.sort(empirical_quantiles, axis=0)
    expected_quantiles = np.linspace(0, 1, y_true.shape[0])[:, np.newaxis]
    rmsce = np.sqrt(np.mean((empirical_quantiles - expected_quantiles) ** 2))
    mace = np.mean(np.abs(empirical_quantiles - expected_quantiles))

    metrics = {
        "rmsce": rmsce,
        "mace": mace,
    }

    return metrics


def compute_adversarial_group_calibration_error(
    y_preds: Float[ndarray, "n_samples batch"],
    y_true: Float[ndarray, "batch"],
    group_size_ratios: Optional[List[float]] = None,
    n_group_draws=100,
    n_full_repeats=10,
    metric_name="rmsce",
    seed=0,
):
    """
    Compute adversarial group calibration metrics based on samples from the predictive
    distribution.

    For each group size, sample groups of data of that size and measure the calibration
    error. Record the worst calibration error for each group size.
    Repeat this process multiple times and report the mean and standard error of the worst
    calibration error for each group size.
    """
    rng = np.random.default_rng(seed)
    group_size_ratios = (
        np.linspace(0.1, 0.9, 10) if group_size_ratios is None else group_size_ratios
    )

    calibration_results = {
        "group_size_ratios": group_size_ratios,
        f"{metric_name}_mean": [],
        f"{metric_name}_stderr": [],
    }
    group_size_list = []

    for group_size_ratio in group_size_ratios:
        group_size = int(group_size_ratio * len(y_true))
        if group_size < 10:
            warnings.warn(
                f"Group size {group_size} is very small to compute adversarial group "
                f"calibration. Consider increasing the group size ratio.",
                stacklevel=2,
            )
        group_size_list.append(group_size)
        worst_calibrations_for_group_size = []
        for _ in range(n_full_repeats):
            worst_calibration = 0  # 0 is the best calibration
            for _ in range(n_group_draws):
                idx = rng.choice(y_true.shape[0], group_size, replace=False)
                y_true_group = y_true[idx]
                y_preds_group = y_preds[:, idx]
                calibration_error = _compute_quantile_calibration_error(
                    y_preds_group, y_true_group
                )[metric_name]
                worst_calibration = max(worst_calibration, calibration_error)
            worst_calibrations_for_group_size.append(worst_calibration)
        mean_worst_calibration = np.mean(worst_calibrations_for_group_size)
        std_worst_calibration = np.std(worst_calibrations_for_group_size, ddof=1)
        calibration_results[f"{metric_name}_mean"].append(mean_worst_calibration)
        calibration_results[f"{metric_name}_stderr"].append(std_worst_calibration)

    for key in calibration_results:
        calibration_results[key] = np.array(calibration_results[key])

    return calibration_results


def _fix_shape(data: ndarray, target_shape: tuple[int, ...]):
    if data.shape == target_shape:
        return data

    # data has one extra dimension of size 1 at the end
    if data.shape[:-1] == target_shape and data.shape[-1] == 1:
        warnings.warn(
            f"Reshaping data from shape {data.shape} to target shape {target_shape}.",
            stacklevel=1,
        )
        return data[..., 0]

    # data is missing one dimension of size 1 at the end
    if data.shape == target_shape[:-1] and target_shape[-1] == 1:
        warnings.warn(
            f"Reshaping data from shape {data.shape} to target shape {target_shape}.",
            stacklevel=1,
        )
        return data[..., np.newaxis]

    raise ValueError(f"Data shape {data.shape} does not match target shape {target_shape}.")
