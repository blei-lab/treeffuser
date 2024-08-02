import numpy as np
from ibug import IBUGWrapper
from jaxtyping import Float
from numpy import ndarray
from sklearn.model_selection import train_test_split
from skopt.space import Integer
from skopt.space import Real
from xgboost import XGBRegressor

from testbed.models import ProbabilisticModel


class IBugXGBoost(ProbabilisticModel):
    """
    IBug wrapper for XGBoost.
    """

    def __init__(self, k=100, n_estimators=100, max_depth=2, learning_rate=0.1, seed=0):
        super().__init__(seed)
        self.model = None
        self.k = k
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ) -> ProbabilisticModel:
        """
        Fit the model to the data.
        """

        if y.shape[1] > 1:
            raise ValueError("IBugXGBoost only accepts 1 dimensional y values.")

        y = y[:, 0]

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.1,
            random_state=self.seed,
        )

        # train GBRT model
        self.gbrt_model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_alpha=0,
            scale_pos_weight=1,
            base_score=0.5,
        ).fit(X_train, y_train)

        # extend GBRT model into a probabilistic estimator
        self.model = IBUGWrapper(k=self.k).fit(
            self.gbrt_model, X_train, y_train, X_val=X_val, y_val=y_val
        )
        return self

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        """
        Predict the mean for each input.
        """
        # predict mean and variance for unseen instances
        location, scale = self.model.pred_dist(X)
        return location.reshape(-1, 1)

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10, seed=None
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Sample from the probability distribution for each input.
        """
        location, scale = self.model.pred_dist(X)
        # sample from a normal distribution
        return np.random.normal(location, scale, size=(n_samples, len(X)))[..., None]

    @staticmethod
    def search_space() -> dict:
        """
        Return the search space for parameters of the model.
        """
        return {
            "k": Integer(20, 250),
            "n_estimators": Integer(10, 1000),
            "learning_rate": Real(0.01, 0.5, prior="log-uniform"),
            "max_depth": Integer(1, 10),
        }
