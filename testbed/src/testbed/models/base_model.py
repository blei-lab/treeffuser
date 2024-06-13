from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple
from typing import Type

import numpy as np
import wandb
from jaxtyping import Float
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.base import MultiOutputMixin
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

    @classmethod
    def supports_multioutput(cls) -> bool:
        """
        Whether the model supports multioutput data.
        Determined by whether the class is a subclass of SupportsMultioutput.
        """
        return issubclass(cls, MultiOutputMixin)

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

    def predict_distribution(self, X: Float[ndarray, "batch x_dim"]):
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


def make_autoregressive_probabilistic_model(
    model_class: Type[ProbabilisticModel],
) -> Type[ProbabilisticModel]:
    """
    A somewhat hacky and complicated way to create an autoregressive model from
    a given model class.

    It returns a class that is a subclass of ProbabilisticModel
    and has as a name "AutoRegressive" + model_class.__name__.
    """

    # Strings to make the kwargs of the init method
    params = model_class.search_space().keys()
    args = [f"{param}=None" for param in params]
    args_string = ", ".join(args)

    class AutoRegressiveProbabilisticModel(ProbabilisticModel, SupportsMultioutput):
        # A probabilistic model that models multi-output data by using an autoregressive model.
        # In particular if p(y_1|x) is a ProbabilisticModel, then: we model p(y_i|x, y_{i-1}, ..., y_1)
        # and then we sample sequentially to get the output.

        def _make_input_for_ith_model(
            self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"], i: int
        ) -> Tuple[Float[ndarray, "batch x_dim+i"], Float[ndarray, "batch 1"]]:

            X_i = np.concatenate([X, y[:, :i]], axis=1)
            y_i = y[:, i].reshape(-1, 1)
            return X_i, y_i

        def fit(
            self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]
        ) -> ProbabilisticModel:
            for i in range(y.shape[1]):
                X_i, y_i = self._make_input_for_ith_model(X, y, i)
                model = model_class(**self.hyperparameters)
                model.fit(X_i, y_i)
                self.models.append(model)
            return self

        def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
            y = np.zeros((X.shape[0], len(self.models)))
            for i, model in enumerate(self.models):
                X_i, _ = self._make_input_for_ith_model(X, y, i)
                y[:, i] = model.predict(X_i).flatten()
            return y

        def sample(
            self,
            X: Float[ndarray, "batch x_dim"],
            n_samples: int = 10,
            seed: Optional[int] = None,
        ) -> Float[ndarray, "n_samples batch y_dim"]:
            samples_rep = np.zeros((n_samples * X.shape[0], len(self.models)))
            X_rep = np.repeat(X, n_samples, axis=0)
            for i, model in enumerate(self.models):
                X_i, _ = self._make_input_for_ith_model(X_rep, samples_rep, i)
                y_i_samples = model.sample(X_i, n_samples=1, seed=seed).squeeze()
                samples_rep[:, i] = y_i_samples.flatten()

            samples = samples_rep.reshape(n_samples, X.shape[0], len(self.models))
            return samples

        @staticmethod
        def search_space() -> dict:
            """
            This has no hyperparameters to optimize.
            """
            return model_class.search_space()

    # We need to set the __init__ method of the class to have the
    # correct signature which is important for BayesOptProbabilisticModel
    # There probably is a better way to do this, but I don't know it.
    init_func_string = f"""
def __init__(self, {args_string}, **kwargs):
    self.models = []
    self.hyperparameters = kwargs
"""
    for param in params:
        init_func_string += f"\n    self.{param} = {param}"

    exec(init_func_string)  # noqa

    AutoRegressiveProbabilisticModel.__init__ = locals()["__init__"]
    AutoRegressiveProbabilisticModel.__name__ = "AutoRegressive" + model_class.__name__

    return AutoRegressiveProbabilisticModel
