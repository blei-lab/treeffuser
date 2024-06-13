# noinspection PyUnresolvedReferences
import lightgbm as lgb  # noqa F401

import argparse
import logging
import os
import warnings
from pathlib import Path
import time
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Type
import sys
from typing import Tuple

import namesgenerator
import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy import ndarray
from sklearn.model_selection import train_test_split


current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / "../"))
sys.path.append(str(current_dir / "../../../src"))
print(sys.path)


from testbed.data.utils import get_data  # noqa E402
from testbed.data.utils import list_data  # noqa E402
from testbed.metrics import AccuracyMetric  # noqa E402
from testbed.metrics import CRPS  # noqa E402
from testbed.metrics import LogLikelihoodExactMetric  # noqa E402
from testbed.metrics import LogLikelihoodFromSamplesMetric  # noqa E402
from testbed.metrics import Metric  # noqa E402
from testbed.metrics import QuantileCalibrationErrorMetric  # noqa E402
from testbed.models.base_model import BayesOptProbabilisticModel  # noqa E402
from testbed.models.base_model import make_autoregressive_probabilistic_model  # noqa E402
from testbed.models.base_model import ProbabilisticModel  # noqa E402

logger = logging.getLogger(__name__)

# Disable wandb console logs (might limit the log size and avoid out of memory errors; but hurts debugging)
os.environ["WANDB_CONSOLE"] = "off"


def get_model(
    model_name: Optional[str] = None, return_available_models: bool = False
) -> List[str] | Type[ProbabilisticModel]:
    # noinspection PyListCreation
    available_models = []

    available_models.append("ngboost")
    if model_name == "ngboost":
        from testbed.models.ngboost import NGBoostGaussian

        return NGBoostGaussian

    available_models.append("ngboost_mixture_gaussian")
    if model_name == "ngboost_mixture_gaussian":
        from testbed.models.ngboost import NGBoostMixtureGaussian

        return NGBoostMixtureGaussian

    available_models.append("treeffuser")
    if model_name == "treeffuser":
        from testbed.models.treeffuser import Treeffuser

        return Treeffuser

    available_models.append("card")
    if model_name == "card":
        from testbed.models.lightning_uq_models import Card

        return Card
    available_models.append("deep_ensemble")
    if model_name == "deep_ensemble":
        from testbed.models.lightning_uq_models import DeepEnsemble

        return DeepEnsemble

    available_models.append("mc_dropout")
    if model_name == "mc_dropout":
        from testbed.models.lightning_uq_models import MCDropout

        return MCDropout

    available_models.append("quantile_regression_tree")
    if model_name == "quantile_regression_tree":
        from testbed.models.quantile_regression import QuantileRegressionTree

        return QuantileRegressionTree

    available_models.append("quantile_regression_nn")
    if model_name == "quantile_regression_nn":
        from testbed.models.lightning_uq_models import QuantileRegression

        return QuantileRegression

    available_models.append("ibug")
    if model_name == "ibug":
        from testbed.models.ibug_ import IBugXGBoost

        return IBugXGBoost

    if model_name == "drf":
        from testbed.models.drf_ import DistributionalRandomForest

        return DistributionalRandomForest

    available_models.append("nnffuser")
    if model_name == "nnffuser":
        from testbed.models.nnffuser import NNffuser

        return NNffuser

    if return_available_models:
        return available_models

    raise ValueError(
        f"Model {model_name} is not available. Available models: {available_models}"
    )


###########################################################
#                 CONSTANTS                               #
###########################################################

AVAILABLE_DATASETS = list(list_data().keys())
AVAILABLE_MODELS = get_model(return_available_models=True)

SUPPORT_CATEGORICAL = ["treeffuser"]

METRIC_TO_CLASS = {
    "accuracy": AccuracyMetric(),
    "quantile_calibration_error": QuantileCalibrationErrorMetric(),
    "log_likelihood_closed_form": LogLikelihoodExactMetric(),
    "crps100": CRPS(n_samples=100),
    # "crps500": CRPS(n_samples=500),
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

    msg = "Mode of model evaluation."
    msg += "cross_val: evaluate the split in --split_idx with the default parameters"
    msg += " or bayes_opt: optimize the hyperparameters with bayesian optimization on a single split."
    msg += " Default: cross_val."
    parser.add_argument(
        "--evaluation_mode",
        type=str,
        help=msg,
        default="cross_val",
    )

    msg = "Number of iterations for the Bayesian optimization. To use"
    msg += " this option, set --evaluation_mode bayes_opt."
    parser.add_argument(
        "--n_iter_bayes_opt",
        type=int,
        default=20,
        help=msg,
    )

    msg = "Append some columns of x to y to increase the dimension of y. Specify"
    msg += "the number of columns to append. They will be selected randomly. Default: 0."
    parser.add_argument(
        "--append_x_to_y",
        type=int,
        default=0,
        help=msg,
    )

    msg = "Which split to evaluate the model on. Default: 0. To use"
    msg += " this option, set --evaluation_mode cross_val."
    parser.add_argument(
        "--split_idx",
        type=int,
        default=-1,
        help=msg,
    )

    msg = "Wandb project name. Disable wandb logging if not provided."
    parser.add_argument(
        "--wandb_project",
        type=str,
        help=msg,
        default=None,
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

    # There is an odd bug such that Card gets a segmentation fault if Treeffuser was imported.
    # Hence, we cannot use both of them in the same run.
    if "card" in args.models and "treeffuser" in args.models:
        msg = "Card and Treeffuser cannot be run in the same script."
        raise ValueError(msg)

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

    if args.append_x_to_y is not None:
        if args.append_x_to_y < 0:
            msg = "The number of extra dimension to add to y cannot be negative."
            raise ValueError(msg)

    # check that split_idx is in [0, 9] if evaluation_mode is cross_val
    if args.evaluation_mode == "cross_val":
        if args.split_idx < 0 or args.split_idx > 9:
            msg = "The split index must be in [0, 9] if evaluation_mode is cross_val."
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
    y_train: Float[ndarray, "train_size n_targets"],
    y_test: Float[ndarray, "test_size n_targets"],
    cat_idx: List[int],
    model_name: str,
    metrics: List[Metric],
    evaluation_mode: Literal["bayes_opt", "cross_val"] = "cross_val",
    n_iter_bayes_opt: int = 20,
    seed: int = 0,
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
    model_class = get_model(model_name)

    use_autoregressive = y_train.shape[1] > 1 and not model_class.supports_multioutput()
    if use_autoregressive:
        model_class = make_autoregressive_probabilistic_model(model_class)

    if evaluation_mode == "bayes_opt":
        model = BayesOptProbabilisticModel(
            model_class=model_class,
            n_iter_bayes_opt=n_iter_bayes_opt,
            frac_validation=0.2,
            n_jobs=1,
        )
    else:
        model = model_class(seed=seed)

    train_start = time.time()
    if model_name in SUPPORT_CATEGORICAL:
        model.fit(X_train, y_train, cat_idx)
    else:
        model.fit(X_train, y_train)
    train_end = time.time()

    results = {}
    results["train_time"] = train_end - train_start

    for metric_name in metrics:
        metric = METRIC_TO_CLASS[metric_name]
        metric_time_start = time.time()
        res = metric.compute(model=model, X_test=X_test, y_test=y_test)
        metric_time_end = time.time()
        results.update(res)
        results[f"{metric_name}_time"] = metric_time_end - metric_time_start

    results.update(model.get_params())
    results.update(model.get_extra_stats())
    return results


def make_multi_output_dataset(
    X: Float[ndarray, "batch n_features"],
    y: Float[ndarray, "batch n_targets"],
    append_x_to_y: int,
    seed: int = 0,
) -> Tuple[Float[ndarray, "batch n_features"], Float[ndarray, "batch n_targets"]]:
    """
    Switch some features to the output to increase the dimension of the output.

    A random subset of Xy is selected as the new features and the rest as the new output.
    and 0.001 * std(y) is added to the new output to add some noise for the conditional regression.

    Args:
        X (Float[ndarray, "n_samples n_features"]): Features of the dataset.
        y (Float[ndarray, "n_samples n_targets"]): Target of the dataset.
        dim_output (int): Dimension of the output.

    Returns:
        Tuple[Float[ndarray, "n_samples n_features"], Float[ndarray, "n_samples n_targets"]]: Multi-output dataset.
    """
    _, n_features = X.shape
    n_targets = y.shape[1]
    n_total = n_features + n_targets

    new_n_features = n_features - append_x_to_y

    if new_n_features < 1:
        raise ValueError(
            "The number of remaining features of X must be at least 1, "
            "make sure that append_x_to_y < n_features."
        )

    # seed with get_default_rng to avoid issues with jax
    rng = np.random.default_rng(seed)

    # randomly select the new features
    new_features = rng.choice(n_total, new_n_features, replace=False)
    new_output = np.array([i for i in range(n_total) if i not in new_features])
    Xy = np.concatenate([X, y], axis=1)

    # create the new dataset
    new_X = Xy[:, new_features]
    new_y = Xy[:, new_output]

    # We add a bit of noise to new_y
    noise = np.random.normal(0, 0.001, new_y.shape) * np.std(new_y)
    new_y += noise

    return new_X, new_y


###########################################################
#               MAIN FUNCTION                            #
###########################################################


def main() -> None:
    args = parse_args()
    check_args(args)

    run_name = namesgenerator.get_random_name()
    full_results = []

    header = format_header(args, run_name)
    logger.info(header)

    for model_name in args.models:
        for dataset_name in args.datasets:
            data = get_data(dataset_name, verbose=True)

            if args.append_x_to_y is not None and args.append_x_to_y > 0:
                data["x"], data["y"] = make_multi_output_dataset(
                    data["x"], data["y"], args.dim_output, args.seed
                )

            if args.split_idx != -1:
                X_train = data["x"][data["k_fold_splits"] != args.split_idx]
                y_train = data["y"][data["k_fold_splits"] != args.split_idx]
                X_test = data["x"][data["k_fold_splits"] == args.split_idx]
                y_test = data["y"][data["k_fold_splits"] == args.split_idx]
            else:
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

            if args.wandb_project is not None:
                # setup wandb
                import wandb

                wandb.init(
                    project=args.wandb_project,
                    name=f"{model_name}_{dataset_name}",
                    # config=args,
                )
                wandb.log(
                    {"model": model_name, "dataset": dataset_name, "split_idx": args.split_idx}
                )

            results = run_model_on_dataset(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                cat_idx=data.get("categorical", None),
                model_name=model_name,
                metrics=args.metrics,
                evaluation_mode=args.evaluation_mode,
                n_iter_bayes_opt=args.n_iter_bayes_opt,
                seed=args.seed,
            )
            results["model"] = model_name
            results["dataset"] = dataset_name
            results["evaluation_mode"] = args.evaluation_mode
            results["seed"] = args.seed
            if args.evaluation_mode == "cross_val":
                results["split_idx"] = args.split_idx
            if args.evaluation_mode == "bayes_opt":
                results["n_iter_bayes_opt"] = args.n_iter_bayes_opt

            if args.wandb_project is not None:
                wandb.log(results)
                wandb.finish()

            full_results.append(results)
            results_string = format_results(model_name, dataset_name, results)
            logger.info(results_string)

            if args.save_dir is not None:
                df = pd.DataFrame(full_results)
                df.to_csv(f"{args.save_dir}/{run_name}.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
