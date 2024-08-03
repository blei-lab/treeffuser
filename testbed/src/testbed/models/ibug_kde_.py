import numpy as np
from ibug import IBUGWrapper
from jaxtyping import Float
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from skopt.space import Integer
from skopt.space import Real
from xgboost import XGBRegressor

from testbed.models import ProbabilisticModel

_BANDWIDTH_SEARCH_SPACE = np.logspace(-2, 10, 10)


class IBugXGBoostKDE(ProbabilisticModel):
    """
    IBug wrapper for XGBoost.
    """

    def __init__(self, k=100, n_estimators=100, max_depth=2, learning_rate=0.1, seed=0):
        super().__init__(seed)
        self.model = None
        self.best_bandwith = None

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
        self.k = min(self.k, len(X))  # make sure k is not larger than the number of instances

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

        _, _, _, y_val_neighbor_vals = self.model.pred_dist(X_val, return_kneighbors=True)
        self.best_bandwith = _find_best_bandwith(y_val_neighbor_vals.T, y_val)
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
        location, scale, _, neighbor_values = self.model.pred_dist(X, return_kneighbors=True)
        assert location.shape == (len(X),)
        assert scale.shape == (len(X),)
        assert neighbor_values.shape == (
            len(X),
            self.model.k_,
        )  # might be different than our k

        samples = []
        for i in range(len(X)):
            kde = KernelDensity(
                bandwidth=self.best_bandwith, algorithm="auto", kernel="gaussian"
            )
            kde.fit(neighbor_values[i].reshape(-1, 1))
            samples.append(kde.sample(n_samples, random_state=seed))

        samples = np.array(samples).transpose(1, 0, 2)
        assert samples.shape == (n_samples, len(X), 1)
        return samples

    @staticmethod
    def search_space() -> dict:
        """
        Return the search space for parameters of the model.
        """
        return {
            "k": Integer(20, 250),
            "n_estimators": Integer(10, 1000),
            "learning_rate": Real(0.01, 0.5, prior="log-uniform"),
            "max_depth": Integer(1, 100),
        }


def _find_best_bandwith(
    y_samples: Float[ndarray, "n_samples batch"],
    y_true: Float[ndarray, "batch"],
) -> float:
    batch = y_true.shape[0]
    assert y_samples.shape == (y_samples.shape[0], batch)

    best_bandwith = 0
    best_log_prob = -np.inf

    for bandwidth in _BANDWIDTH_SEARCH_SPACE:
        log_prob = 0
        for i in range(batch):
            y_i = y_samples[:, i].reshape(-1, 1)
            kde = KernelDensity(bandwidth=bandwidth, algorithm="auto", kernel="gaussian")
            kde.fit(y_i)
            log_prob += kde.score(y_true[i].reshape(-1, 1))

        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_bandwith = bandwidth

    return best_bandwith
