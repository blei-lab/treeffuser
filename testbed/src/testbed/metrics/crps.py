from typing import Dict

import properscoring
from jaxtyping import Float
from numpy import ndarray

from testbed.metrics import Metric
from testbed.models import ProbabilisticModel


class CRPS(Metric):
    """
    Computes the Continuous Ranked Probability Score (CRPS) of a model's predictive
    distribution given empirical samples of the model.
    """

    def __init__(
        self,
        n_samples: int = 100,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples

    def compute(
        self,
        model: ProbabilisticModel,
        X_test: Float[ndarray, "batch n_features"],
        y_test: Float[ndarray, "batch y_dim"],
        samples: Float[ndarray, "n_samples batch y_dim"] = None,
    ) -> Dict[str, float]:
        """
        Compute the continuous ranked probability score (CRPS) of the predictive distribution
        from empirical samples.

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
         dict with key 'crps' : float
            The CRPS of the predictive distribution from empirical samples.

        """
        if samples is not None:
            y_samples = samples
        else:
            y_samples: Float[ndarray, "n_samples batch y_dim"] = model.sample(
                X=X_test, n_samples=self.n_samples, seed=self.seed
            )
        _, batch, y_dim = y_samples.shape

        assert batch == X_test.shape[0], f"batch={batch} != X_test.shape[0]={X_test.shape[0]}"
        assert y_dim == y_test.shape[1], f"y_dim={y_dim} != y_test.shape[1]={y_test.shape[1]}"

        res = properscoring.crps_ensemble(
            y_test,
            y_samples,
            axis=0,
        )
        return {f"crps_{self.n_samples}": res.mean()}
