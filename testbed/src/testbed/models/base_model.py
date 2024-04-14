from abc import ABC
from abc import abstractmethod
from typing import Type

from jaxtyping import Float
from numpy import ndarray
from sklearn.base import BaseEstimator
from skopt import BayesSearchCV
from skopt.space import Categorical
from skopt.space import Integer
from skopt.space import Real


class ProbabilisticModel(ABC, BaseEstimator):
    """
    A base class for all probabilistic models. Which produces a probability distribution
    rather than a single output for each input. Subclasses BaseEstimator.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ) -> "ProbabilisticModel":
        """
        Fit the model to the data.
        """

    @abstractmethod
    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        """
        Predict the mean for each input.
        """

    @abstractmethod
    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        """
        Sample from the probability distribution for each input.
        """

    # @abstractmethod
    # def predict_distribution(self, X: Float[ndarray, "batch x_dim"]):
    #     """
    #     Predict the probability distribution for each input.
    #     """

    @abstractmethod
    @property
    def search_space(self) -> dict:
        """
        Return the search space for parameters of the model.

        It should be a dictionary using the conventions specified by
        skopt. For example:

        {
            "param1": Real(0.0, 1.0),
            "param2": Real(0.0, 1.0, prior="log-uniform"),
            "param3": Integer(1, 10),
            "param4": Categorical(["a", "b", "c"]),
        }
        """


class CachedProbabilisticModel(ProbabilisticModel):
    """
    A probabilistic model that caches the predictions for each input, for faster evaluation.
    """

    def __init__(self, model: ProbabilisticModel):
        super().__init__()
        self._cache = {}
        self.model = model

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ) -> ProbabilisticModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        if "predict" not in self._cache:
            self._cache["predict"] = self.model.predict(X)
        return self._cache["predict"]

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        if "sample" not in self._cache:
            self._cache["sample"] = self.model.sample(X, n_samples)
        return self._cache["sample"]

    def clear_cache(self):
        self._cache = {}


class BayesSearchProbabilisticModel(ProbabilisticModel):
    """
    A probabilistic model that uses Bayesian optimization to find the best hyperparameters.
    It is a wrapper around a probabilistic model that uses the skopt library.
    """

    def __init__(
        self,
        model_class: Type[ProbabilisticModel],
        n_iter: int = 50,
        cv: int = 5,
        n_jobs: int = -1,
    ):
        """
        model_class: The class of the model to be optimized.
        n_iter: The number of iterations for the Bayesian optimization.
        cv: The number of cross-validation folds.
        n_jobs: The number of parallel jobs to run. -1 means using all processors.
        """
        super().__init__()
        self._model_class = model_class
        self._model = None
        self._search_space = None
        self._n_iter = n_iter
        self._cv = cv
        self._n_jobs = n_jobs

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ) -> ProbabilisticModel:
        model = self._model_class()
        self._search_space = self._model_class.search_space
        search_space = {
            key: self._convert_space(space) for key, space in self._search_space.items()
        }

        opt = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=self._n_iter,
            cv=self._cv,
            n_jobs=self._n_jobs,
            verbose=1,
            random_state=42,
        )

        opt.fit(X, y)

        self._model = opt.best_estimator_
        return self

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        return self._model.predict(X)

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        return self._model.sample(X, n_samples)

    def _convert_space(self, space):
        if isinstance(space, tuple):
            if len(space) == 2:
                return Real(*space)
            elif len(space) == 3:
                if space[2] == "log-uniform":
                    return Real(*space[:2], prior="log-uniform")
                elif space[2] == "int":
                    return Integer(*space[:2])
        elif isinstance(space, list):
            return Categorical(space)
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")
