import numpy as np
from drf import drf
from jaxtyping import Float
from numpy import ndarray
from sklearn.base import MultiOutputMixin
from skopt.space import Integer

from testbed.models import ProbabilisticModel


class DistributionalRandomForest(ProbabilisticModel, MultiOutputMixin):
    """
    Distributional Random Forest.

    Note: This model requires:
        - the `drf` package to be installed from  GitHub (not on PyPI, and outdated on test PyPI).
          `pip install -e "git+https://github.com/lorismichel/drf#egg=drf&subdirectory=python-package"`
        - the R package `drf` to be installed in the R environment.

    On MacOS, you may need to install: https://www.xquartz.org to install the R package.
    (e.g. if errors involve '/Library/Frameworks/R.framework/Resources/modules//R_X11.so')
    """

    def __init__(self, min_node_size=10, num_trees=1000, seed=0):
        super().__init__(seed)
        self.model = None
        self.min_node_size = min_node_size
        self.num_trees = num_trees

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ) -> ProbabilisticModel:
        """
        Fit the model to the data.
        """

        self.model = drf(
            min_node_size=self.min_node_size,
            num_trees=self.num_trees,
            splitting_rule="FourierMMD",
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        """
        Predict the mean for each input.
        """
        # predict mean and variance for unseen instances
        out = self.model.predict(newdata=X, functional="mean")
        return out.mean

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10, seed=None
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Sample from the probability distribution for each input.
        """
        np.random.seed(seed)
        out = self.model.predict(newdata=X, functional="sample", n=n_samples)
        out = out.sample  # dim = (batch, y_dim, n_samples)
        # reorder axes
        out = np.transpose(out, (2, 0, 1))
        return out

    @staticmethod
    def search_space() -> dict:
        """
        Return the search space for parameters of the model.
        """
        return {
            "min_node_size": Integer(5, 30),
            "num_trees": Integer(250, 3000),
        }
