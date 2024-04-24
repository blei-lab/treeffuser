import argparse
import logging
import warnings
from typing import Dict
from typing import List

import namesgenerator
import pandas as pd
from jaxtyping import Float
from numpy import ndarray
from sklearn.model_selection import train_test_split

from testbed.data.utils import get_data
from testbed.data.utils import list_data
from testbed.metrics import AccuracyMetric
from testbed.metrics import LogLikelihoodFromSamplesMetric
from testbed.metrics import Metric

# from testbed.metrics import QuantileCalibrationErrorMetric
from testbed.models.base_model import BayesOptProbabilisticModel
from testbed.models.ngboost import NGBoostGaussian
from testbed.models.ngboost import NGBoostMixtureGaussian

logger = logging.getLogger(__name__)

###########################################################
#                 CONSTANTS                               #
###########################################################

AVAILABLE_DATASETS = list(list_data().keys())


# Treeffuser is not imported by default as it causes a segmentation fault
# of Card model. If adequate it gets added to this dictionary.
# by the proc_model_names function.
MODEL_TO_CLASS = {
    "ngboost_gaussian": NGBoostGaussian,
    "ngboost_mixture_gaussian": NGBoostMixtureGaussian,
    "card": None,
    "treeffuser": None,
}
AVAILABLE_MODELS = list(MODEL_TO_CLASS.keys())

METRIC_TO_CLASS = {
    "accuracy": AccuracyMetric,
    # "quantile_calibration_error": QuantileCalibrationErrorMetric,
    "log_likelihood": LogLikelihoodFromSamplesMetric,
}
AVAILABLE_METRICS = list(METRIC_TO_CLASS.keys())


BARS = "-" * 50 + "\n"
TITLE = "\n" + BARS + "TESTBED".center(50) + "\n" + BARS


###########################################################
#               HELPER FUNCTIONS                          #
###########################################################


def lst_to_new_line(lst: list) -> str:
    """
    Helper function to convert list to a string with each element on a new line.

    >> lst_to_new_line(["a", "b", "c"])
        - a
        - b
        - c
    """
    string = ""
    for item in lst:
        string += f"- {item}\n"
    return string


def update_metric_to_class(models: List[str]):
    """
    Adds Treeffuser to MODEL_TO_CLASS if it is in the list of models.
    Changes global variable MODEL_TO_CLASS.

    There is an odd bug such that if Treeffuser is imported then Card
    doesn't run and we get a segmentation fault. We solve this by
    a conditional import statement (unfortunately).

    This function verifies that we are not asked to run both Treeffuser
    and Card models together and imports Treeffuser if needed.

    This is extremely hacky and should be fixed in the future.

    Args:
        models (List[str]): List of models to run.
    """

    if "treeffuser" in models and "card" in models:
        msg = "Treeffuser and Card models can't be run together. Segmentation fault occurs."
        raise ValueError(msg)

    if "treeffuser" in models:
        from testbed.models.treeffuser import Treeffuser

        MODEL_TO_CLASS["treeffuser"] = Treeffuser
    if "card" in models:
        from testbed.models.card import Card

        MODEL_TO_CLASS["card"] = Card


def parse_args():
    parser = argparse.ArgumentParser(description="Run models on available datasets.")

    msg = "List of datasets to run models on. Default: all available datasets."
    msg += f" Available datasets: {lst_to_new_line(AVAILABLE_DATASETS)}"
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=AVAILABLE_DATASETS,
        help=msg,
    )

    msg = "List of models to run on the datasets. Default: all available models except card."
    msg += f" Available models: {lst_to_new_line(AVAILABLE_MODELS)}"
    default_models = [model for model in AVAILABLE_MODELS if model != "card"]
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=default_models,
        help=msg,
    )

    msg = "List of metrics to compute. Default: all available metrics."
    msg += f" Available metrics: {lst_to_new_line(AVAILABLE_METRICS)}"
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=AVAILABLE_METRICS,
        help=msg,
    )

    msg = "Seed for reproducibility."
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=msg,
    )

    msg = "Directory to save the results. Default: None."
    msg += " If None, the results are not saved."
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help=msg,
    )

    msg = "Whether to optimize the hyperparameters of the models."
    parser.add_argument(
        "--optimize_hyperparameters",
        action="store_true",
        help=msg,
    )

    msg = "Number of iterations for the Bayesian optimization. To use"
    msg += " this option, the --optimize_hyperparameters flag must be set."
    parser.add_argument(
        "--n_iter_bayes_opt",
        type=int,
        default=20,
        help=msg,
    )

    return parser.parse_args()


def check_args(args):
    """
    Check the arguments passed to the script.

    Args:
        args (argparse.Namespace): Arguments passed to the script.
    """
    # check model name is valid
    for model_name in args.models:
        if model_name not in AVAILABLE_MODELS:
            msg = f"Model {model_name} is not available."
            msg += f" Available models: {lst_to_new_line(AVAILABLE_MODELS)}"

    # check dataset name is valid
    for dataset_name in args.datasets:
        if dataset_name not in AVAILABLE_DATASETS:
            msg = f"Dataset {dataset_name} is not available."
            msg += f" Available datasets: {lst_to_new_line(AVAILABLE_DATASETS)}"
            raise ValueError(msg)

    # check metric name is valid
    for metric_name in args.metrics:
        if metric_name not in AVAILABLE_METRICS:
            msg = f"Metric {metric_name} is not available."
            msg += f" Available metrics: {lst_to_new_line(AVAILABLE_METRICS)}"
            raise ValueError(msg)

    # check that n_iter_bayes_opt is positive
    if args.n_iter_bayes_opt <= 0:
        msg = "The number of iterations for the Bayesian optimization must be positive."
        raise ValueError(msg)


def format_results(model_name: str, dataset_name: str, results: Dict[str, float]) -> str:
    """
    Format the results of a model on a dataset.

    Args:
        model_name (str): Name of the model.
        dataset_name (str): Name of the dataset.
        results (Dict[str, float]): Results of the model on the dataset.
    """
    results_string = "\n\n"
    results_string += BARS
    results_string += (
        f" MODEL: {model_name.capitalize()} | DATASET: {dataset_name.capitalize()}\n"
    )
    results_string += BARS
    for key, value in results.items():
        results_string += f"{key}: {value}\n"
    results_string += BARS
    results_string += "\n\n"

    return results_string


def format_header(args: argparse.Namespace, run_name: str) -> str:
    """
    Format the header of the script for logging.
    """
    header = "\n"
    header += TITLE
    header += "\n"
    header += f"Running models: {args.models}\n"
    header += f"On datasets: {args.datasets}\n"
    header += f"Computing metrics: {args.metrics}\n"
    header += f"Seed: {args.seed}\n"
    header += f"Results will be saved in: {args.save_dir}\n"
    header += f"Run name: {run_name}\n"
    header += BARS
    return header


def run_model_on_dataset(
    X_train: Float[ndarray, "train_size n_features"],
    X_test: Float[ndarray, "test_size n_features"],
    y_train: Float[ndarray, "train_size 1"],
    y_test: Float[ndarray, "test_size 1"],
    model_name: str,
    metrics: List[Metric],
    optimize_hyperparameters: bool,
    n_iter_bayes_opt: int = 20,
) -> Dict[str, float]:
    """
    Run a model on a dataset and compute the metrics specified.

    Args:
        model_name (str): Name of the model to run.
        dataset_name (str): Name of the dataset to run the model on.
        metrics (List[Metric]): List of metrics to compute.
        n_iter_bayes_opt (int): Number of iterations for the Bayesian optimization.
            Not used if optimize_hyperparameters is False.
        optimize_hyperparameters: Wether to use bayesian optimizatoin when
            fitting the model or not

    Returns:
        Dict[str, float]: Results of the model on the dataset.
    """
    model_class = MODEL_TO_CLASS[model_name]
    if optimize_hyperparameters:
        model = BayesOptProbabilisticModel(
            model_class=model_class, n_iter_bayes_opt=n_iter_bayes_opt, cv=4, n_jobs=1
        )
    else:
        model = model_class()

    model.fit(X_train, y_train)

    results = {}
    for metric in metrics:
        metric = METRIC_TO_CLASS[metric]()
        res = metric.compute(model=model, X_test=X_test, y_test=y_test)
        results.update(res)

    if optimize_hyperparameters:
        results.update(model._model.get_params())
        results["n_iter_bayes_opt"] = n_iter_bayes_opt
    else:
        results.update(model.get_params())
    return results


###########################################################
#               MAIN FUNCTION                            #
###########################################################


def main() -> None:
    args = parse_args()
    check_args(args)

    # This line should be added before everything else
    # see the docstring of the function for more information.
    update_metric_to_class(args.models)

    run_name = namesgenerator.get_random_name()
    full_results = []

    header = format_header(args, run_name)
    logger.info(header)

    for model_name in args.models:
        for dataset_name in args.datasets:
            data = get_data(dataset_name, verbose=True)
            if "test" not in data:
                X_train, X_test, y_train, y_test = train_test_split(
                    data["x"], data["y"], test_size=0.2, random_state=args.seed
                )
            else:
                warnings.warn(
                    f"Warning: The dataset '{dataset_name}' includes a prescribed test set. The 'seed' argument will be ignored.",
                    stacklevel=2,
                )
                X_train, X_test, y_train, y_test = (
                    data["x"],
                    data["test"]["x"],
                    data["y"],
                    data["test"]["y"],
                )

            results = run_model_on_dataset(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_name=model_name,
                metrics=args.metrics,
                optimize_hyperparameters=args.optimize_hyperparameters,
                n_iter_bayes_opt=args.n_iter_bayes_opt,
            )
            results["model"] = model_name
            results["dataset"] = dataset_name
            full_results.append(results)

            results_string = format_results(model_name, dataset_name, results)
            logger.info(results_string)

            if args.save_dir is not None:
                df = pd.DataFrame(full_results)
                df.to_csv(f"{args.save_dir}/{run_name}.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
