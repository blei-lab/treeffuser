from abc import ABC
from abc import abstractmethod
from typing import Type

from jaxtyping import Float
from numpy import ndarray
from sklearn.base import BaseEstimator
from skopt import BayesSearchCV


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

    @staticmethod
    @abstractmethod
    def search_space() -> dict:
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

    def score(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        n_samples: int = 50,
        bandwidth: float = 1.0,
    ) -> float:
        """
        Return the negative log-likelihood of the model on the data.
        This function is used for hyperparameter optimization and
        compatibility with scikit-learn.

        n_samples: The number of samples to draw from the model's predictive
            distribution to compute an estimate of the log likelihood.
        bandwidth: The bandwidth of the kernel density estimator used to fit the samples.
        """
        # Avoid circular import
        import testbed.metrics.log_likelihood as log_likelihood

        metric = log_likelihood.LogLikelihoodFromSamplesMetric(
            n_samples=n_samples, bandwidth=bandwidth
        )
        return -1.0 * metric.compute(self, X, y)["nll"]


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


class BayesOptProbabilisticModel(ProbabilisticModel):
    """
    A probabilistic model that uses Bayesian optimization to find the best hyperparameters.
    It is a wrapper around a probabilistic model that uses the skopt library.
    """

    def __init__(
        self,
        model_class: Type[ProbabilisticModel],
        n_iter_bayes_opt: int = 50,
        cv: int = 5,
        n_jobs: int = -1,
    ):
        """
        model_class: The class of the model to be optimized.
        n_iter_bayes_opt: The number of iterations for the Bayesian optimization.
        cv: The number of cross-validation folds.
        n_jobs: The number of parallel jobs to run. -1 means using all processors.
        """
        super().__init__()
        self._model_class = model_class
        self._model = None
        self._model_search_space = None
        self._n_iter_bayes_opt = n_iter_bayes_opt
        self._cv = cv
        self._n_jobs = n_jobs

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ) -> ProbabilisticModel:
        model = self._model_class()
        self._model_search_space = self._model_class.search_space()

        opt = BayesSearchCV(
            estimator=model,
            search_spaces=self._model_search_space,
            n_iter=self._n_iter_bayes_opt,
            cv=self._cv,
            n_jobs=self._n_jobs,
            verbose=2,
            random_state=0,
            optimizer_kwargs={"base_estimator": "GBRT"},
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

    @staticmethod
    def search_space() -> dict:
        """
        This has no hyperparameters to optimize.
        """
        return {}
