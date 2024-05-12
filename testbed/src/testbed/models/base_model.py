from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Type

import torch.distributions
import wandb
from jaxtyping import Float
from numpy import ndarray
from sklearn.base import BaseEstimator
from skopt import BayesSearchCV
from skopt.utils import use_named_args


class ProbabilisticModel(ABC, BaseEstimator):
    """
    A base class for all probabilistic models. Which produces a probability distribution
    rather than a single output for each input. Subclasses BaseEstimator.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.seed = seed

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

    def predict_distribution(
        self, X: Float[ndarray, "batch x_dim"]
    ) -> torch.distributions.Distribution:
        """
        Predict the distribution for each input.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10, seed: Optional[int] = None
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
        n_samples: int = 100,
    ) -> float:
        """
        Return the negative CRPS score for the model.
        The higher the score, the better the model.
        This function is used for hyperparameter optimization and
        compatibility with scikit-learn.

        n_samples: The number of samples to draw from the model's predictive
            distribution to compute an estimate of the log likelihood.
        """
        # Avoid circular import
        import testbed.metrics.crps as crps

        metric = crps.CRPS(
            n_samples=n_samples,
        )
        return -1.0 * next(iter(metric.compute(self, X, y).values()))


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


class BayesOptCVProbabilisticModel(ProbabilisticModel):
    """
    A probabilistic model that uses Bayesian optimization to find the best hyperparameters.
    It is a wrapper around a probabilistic model that uses the skopt library.
    It finds the best hyperparameters by cross-validation.

    See also: BayesOptProbabilisticModel
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
            verbose=3,
            random_state=0,
            optimizer_kwargs={"base_estimator": "GBRT"},
            error_score=-1000000,
        )

        callbacks = []
        # check if wandb is on
        if wandb.run is not None:

            def wandb_callback(res):
                wandb.log(
                    {
                        "bayes_opt_score": res.fun,
                        "bayes_opt_params": res.x_iters[-1],
                    }
                )

            callbacks.append(wandb_callback)
        opt.fit(X, y, callback=callbacks)

        self._model = opt.best_estimator_
        self._model.fit(X, y)
        return self

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        return self._model.predict(X)

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10, seed: Optional[int] = None
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        return self._model.sample(X, n_samples, seed=seed)

    @staticmethod
    def search_space() -> dict:
        """
        This has no hyperparameters to optimize.
        """
        return {}

    def get_params(self, deep=True):
        res = self._model.get_params(deep=deep)
        res["n_iter_bayes_opt"] = self._n_iter_bayes_opt
        res["cv_bayes_opt"] = self._cv
        return res


class BayesOptProbabilisticModel(ProbabilisticModel):
    """
    A probabilistic model that uses Bayesian optimization to find the best hyperparameters.
    It is a wrapper around a probabilistic model that uses the skopt library.
    It finds the best hyperparameters on a train-validation split.

    See also: BayesOptCVProbabilisticModel
    """

    def __init__(
        self,
        model_class: Type[ProbabilisticModel],
        n_iter_bayes_opt: int = 50,
        n_jobs: int = -1,
        frac_validation: float = 0.1,
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
        self._frac_validation = frac_validation
        self._n_jobs = n_jobs

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ) -> ProbabilisticModel:
        # not CV, need to split data
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self._frac_validation,
            random_state=0,
        )

        callbacks = []
        # check if wandb is on
        if wandb.run is not None:

            def wandb_callback(res):
                wandb.log(
                    {
                        "bayes_opt_score": res.fun,
                        **dict(zip(space_args.keys(), res.x_iters[-1])),
                    }
                )

            callbacks.append(wandb_callback)

        space_args = self._model_class.search_space()
        space = []
        for k, v in space_args.items():
            v.name = k
            space.append(v)

        @use_named_args(space)
        def objective(**params):
            model = self._model_class(**params)
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            return -score

        from skopt import forest_minimize

        res = forest_minimize(
            objective,
            space,
            n_initial_points=5,
            n_calls=self._n_iter_bayes_opt,
            n_jobs=self._n_jobs,
            verbose=True,
            random_state=0,
            callback=callbacks,
        )

        self._model = self._model_class(**dict(zip(space_args.keys(), res.x)))
        self._model.fit(X, y)
        return self

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        return self._model.predict(X)

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=10, seed: Optional[int] = None
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        return self._model.sample(X, n_samples, seed=seed)

    @staticmethod
    def search_space() -> dict:
        """
        This has no hyperparameters to optimize.
        """
        return {}

    def get_params(self, deep=True):
        res = self._model.get_params(deep=deep)
        res["n_iter_bayes_opt"] = self._n_iter_bayes_opt
        res["frac_validation"] = self._frac_validation
        return res
