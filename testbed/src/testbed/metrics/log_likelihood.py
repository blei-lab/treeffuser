from typing import Dict

from jaxtyping import Float, Array

from testbed.metrics.base_metric import Metric
from testbed.models.base_model import ProbabilisticModel


class LogLikelihoodMetric(Metric):
    """
    Computes the log likelihood of the predictive distribution.
    """

    def compute(
        self,
        model: ProbabilisticModel,
        X_test: Float[Array, "batch n_features"],
        y_test: Float[Array, "batch y_dim"],
    ) -> Dict[str, float]:
        """
        Compute the log likelihood of the predictive distribution.

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
        log_likelihood : dict
            A single scalar which quantifies the log likelihood of the predictive distribution.
        """
        predictive_distribution = model.predict_distribution(X_test)
        log_likelihood = predictive_distribution.log_prob(y_test).mean().item()

        return {"log_likelihood": log_likelihood}
