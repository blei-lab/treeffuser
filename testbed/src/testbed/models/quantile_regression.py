import lightgbm as lgb  # noqa

from typing import Tuple


import numpy as np
from jaxtyping import Float
from numpy import ndarray
from sklearn.model_selection import train_test_split
from skopt.space import Integer
from skopt.space import Real

from .base_model import ProbabilisticModel


def pinball_loss(
    preds, train_data: lgb.Dataset
) -> Tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
    """
    Pinball loss for quantile regression.

    Returns the gradient and hessian of the pinball loss.

    The pinball loss is defined as:
        loss = q * (y - pred) if y > pred else (1 - q) * (pred - y)
    Hence the gradient is:
        grad = - q if y > pred else (1 - q)
    And the hessian is:
        hess = 0
    We set the hessian to 1 tho so that lightgbm works this trick
    makes the model essentially act like standard gradient boosting
    and is implemented in the lightgbm source code as well.
    """
    Xq, y = train_data.data, train_data.get_label()
    q = Xq[:, -1]

    y = y.reshape(-1)
    preds = preds.reshape(-1)
    q = q.reshape(-1)

    grad = np.where(y > preds, -q, 1 - q)  # + np.random.randn(len(q)) * 1e-6
    hess = np.ones_like(grad)

    return grad, hess


def pinball_eval(preds, train_data: lgb.Dataset) -> Tuple[str, float, bool]:
    """
    Pinball loss for quantile regression.

    Returns the name of the metric, the value of the metric, and whether
    a higher value is better.
    """
    Xq, y = train_data.data, train_data.get_label()
    q = Xq[:, -1]

    y = y.reshape(-1)
    preds = preds.reshape(-1)
    q = q.reshape(-1)

    loss = np.where(y > preds, q * (y - preds), (1 - q) * (preds - y))
    return "pinball", np.mean(loss), False


class QuantileRegressionTree(ProbabilisticModel):
    """
    Creates a quantile regression model using LightGBM.

    The model works by fitting a model f(X, q) that predicts the q-th quantile of
    the target distribution. This is done by minimizing the loss

    E_q[loss(y, f(X, q))], where loss is the pinball loss and the
    distribution of q is uniform.
    """

    def __init__(
        self,
        n_estimators: int = 10000,
        n_repeats: int = 100,
        learning_rate: float = 0.5,
        early_stopping_rounds: int = 50,
        num_leaves: int = 31,
        validation_fraction: float = 0.1,
        verbose: bool = 0,
        seed: int = 0,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.n_repeats = n_repeats
        self.learning_rate = learning_rate
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.num_leaves = num_leaves
        self.seed = seed
        self.validation_fraction = validation_fraction
        self.verbose = verbose

        self.model = None

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ) -> "ProbabilisticModel":

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_fraction, random_state=self.seed
        )

        train_examples = self.n_repeats * X_train.shape[0]
        q_train = np.random.uniform(0, 1, size=train_examples)
        X_train = np.tile(X_train, (self.n_repeats, 1))
        Xq_train = np.concatenate([X_train, q_train[:, None]], axis=1)
        y_train = np.tile(y_train, (self.n_repeats, 1))

        validation_examples = self.n_repeats * X_val.shape[0]
        q_val = np.random.uniform(0, 1, size=validation_examples)
        X_val = np.tile(X_val, (self.n_repeats, 1))
        Xq_val = np.concatenate([X_val, q_val[:, None]], axis=1)
        y_val = np.tile(y_val, (self.n_repeats, 1))

        train_dataset = lgb.Dataset(Xq_train, y_train, free_raw_data=False)
        test_dataset = lgb.Dataset(Xq_val, y_val, free_raw_data=False)

        params = {
            "boosting_type": "gbdt",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "random_state": self.seed,
            "objective": pinball_loss,
            "verbose": self.verbose,
        }

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=int(self.early_stopping_rounds), verbose=self.verbose
            )
        ]

        self.model = lgb.train(
            params,
            train_set=train_dataset,
            num_boost_round=self.n_estimators,
            valid_sets=test_dataset,
            callbacks=callbacks,
            feval=pinball_eval,
        )

        return self

    def predict(
        self, X: Float[ndarray, "batch x_dim"], n_quantiles=100
    ) -> Float[ndarray, "batch y_dim"]:
        q_pred = self._predict_quantile_function(X, n_quantiles)
        mean = q_pred.mean(axis=0)
        return mean

    def predict_quantiles(
        self, X: Float[ndarray, "batch x_dim"], quantiles: Float[ndarray, "batch"]
    ) -> Float[ndarray, "batch y_dim"]:
        Xq = np.concatenate([X, quantiles[:, None]], axis=1)
        return self.model.predict(Xq, num_iteration=self.model.best_iteration)

    def _predict_quantile_function(
        self, X: Float[ndarray, "batch x_dim"], n_quantiles=100
    ) -> Float[ndarray, "n_quantiles batch y_dim"]:
        X_repeated = np.repeat(X, n_quantiles, axis=0)
        q = np.linspace(0, 1, n_quantiles)
        q_repeated = np.tile(q, X.shape[0])

        Xq = np.concatenate([X_repeated, q_repeated[:, None]], axis=1)
        # (n_quantiles * batch, 1)
        q_pred = self.model.predict(Xq, num_iteration=self.model.best_iteration).flatten()
        # reshape to have (n_quantiles, batch, y_dim)
        q_pred = q_pred.reshape(n_quantiles, X.shape[0], -1, order="F")

        # sort (sometimes the quantiles might cross)
        q_pred = np.sort(q_pred, axis=0)
        return q_pred

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10, n_quantiles=1000, seed=None
    ) -> Float[ndarray, "n_samples batch 1"]:
        q: Float[ndarray, "n_quantiles+1 batch 1"] = self._predict_quantile_function(
            X, n_quantiles + 1
        )
        if seed is not None:
            np.random.seed(seed)

        # we will sample (and construct the index of the quantile)
        rng = np.random.default_rng(seed)
        q_index = rng.choice(n_quantiles, size=(n_samples, X.shape[0]))
        batch_index = np.arange(X.shape[0])
        batch_index = np.tile(batch_index, (n_samples, 1))

        q_lower_bound = q[q_index, batch_index]
        q_upper_bound = q[q_index + 1, batch_index]

        # sample from the uniform distribution
        u = rng.uniform(size=(n_samples, X.shape[0], 1))

        # sample from the quantile
        y = q_lower_bound + u * (q_upper_bound - q_lower_bound)
        return y

    @staticmethod
    def search_space() -> dict:
        return {
            "n_estimators": Integer(100, 3000, "log-uniform"),
            "n_repeats": Integer(10, 100),
            "learning_rate": Real(0.01, 1),
            "early_stopping_rounds": Integer(10, 100),
            "num_leaves": Integer(10, 100),
        }
