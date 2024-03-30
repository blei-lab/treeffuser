import argparse
from typing import Dict
from typing import List

import namesgenerator
import pandas as pd
from sklearn.model_selection import train_test_split

from testbed.data.utils import get_data
from testbed.data.utils import list_data
from testbed.metrics import AccuracyMetric
from testbed.metrics import Metric
from testbed.metrics import QuantileCalibrationErrorMetric
from testbed.models.card import Card
from testbed.models.ngboost import NGBoostGaussian
from testbed.models.ngboost import NGBoostMixtureGaussian

###########################################################
#                 CONSTANTS                               #
###########################################################

AVAILABLE_DATASETS = list_data()

# Treeffuser is not imported by default as it causes a segmentation fault
# of Card model. If adequate it gets added to this dictionary.
# by the proc_model_names function.
AVAILABLE_MODELS = {
    "ngboost_gaussian": NGBoostGaussian,
    "ngboost_mixture_gaussian": NGBoostMixtureGaussian,
    "card": Card,
    "treeffuser": None,
}


AVAILABLE_METRICS = {
    "accuracy": AccuracyMetric,
    "quantile_calibration_error": QuantileCalibrationErrorMetric,
}

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


def update_available_models(models: List[str], available_models: dict) -> List[str]:
    """
    Adds Treeffuser to AVAILABLE_MODELS if needed and return
    the updated list of available models.

    There is an odd bug such that if Treeffuser is imported then Card
    doesn't run and we get a segmentation fault. We solve this by
    a conditional import statement (unfortunately).

    This function verifies that we are not asked to run both Treeffuser
    and Card models together and imports Treeffuser if needed.

    This is extremely hacky and should be fixed in the future.

    Args:
        models (List[str]): List of models to run.
        available_models (dict): Dictionary of available models. Should be set
            to AVAILABLE_MODELS always. Names as a parameter for
            clarity.
    """

    if "treeffuser" in models and "card" in models:
        msg = "Treeffuser and Card models can't be run together. Segmentation fault occurs."
        raise ValueError(msg)

    if "treeffuser" in models:
        from testbed.models.treeffuser import Treeffuser

        available_models["treeffuser"] = Treeffuser

    return available_models


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
    msg += f" Available models: {lst_to_new_line(AVAILABLE_MODELS.keys())}"
    available_models = [model for model in AVAILABLE_MODELS if model != "card"]
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=available_models,
        help=msg,
    )

    msg = "List of metrics to compute. Default: all available metrics."
    msg += f" Available metrics: {lst_to_new_line(AVAILABLE_METRICS.keys())}"
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=AVAILABLE_METRICS.keys(),
        help=msg,
    )

    msg = "Seed for reproducibility."
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    return parser.parse_args()


def check_args(args):
    """
    Check the arguments passed to the script.

    Args:
        args (argparse.Namespace): Arguments passed to the script.
    """
    # check model name is valid
    for model_name in args.models:
        print(model_name)
        print(AVAILABLE_MODELS)
        if model_name not in AVAILABLE_MODELS:
            msg = f"Model {model_name} is not available."
            msg += f" Available models: {lst_to_new_line(AVAILABLE_MODELS.keys())}"

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
            msg += f" Available metrics: {lst_to_new_line(AVAILABLE_METRICS.keys())}"
            raise ValueError(msg)


def print_results(model_name: str, dataset_name: str, results: Dict[str, float]):
    """
    Print the results for a model on a dataset.

    Args:
        model_name (str): Name of the model.
        dataset_name (str): Name of the dataset.
        results (Dict[str, float]): Results of the model on the dataset.
    """
    print("\n")
    print("##################################################")
    print(f"Results for {model_name} on {dataset_name}")
    for key, value in results.items():
        print(f"{key}: {value}")
    print("##################################################")


def run_model_on_dataset(
    model_name: str, dataset_name: str, metrics: List[Metric], seed: int
) -> Dict[str, float]:
    """
    Run a model on a dataset and compute the metrics specified.

    Args:
        model_name (str): Name of the model to run.
        dataset_name (str): Name of the dataset to run the model on.
        metrics (List[Metric]): List of metrics to compute.
        seed (int): Seed for reproducibility.
    """
    model = AVAILABLE_MODELS[model_name]()
    data = get_data(dataset_name, verbose=True)

    X_train, X_test, y_train, y_test = train_test_split(
        data["x"], data["y"], test_size=0.2, random_state=seed
    )

    model.fit(X_train, y_train)

    results = {}
    for metric in metrics:
        metric = metric()
        res = metric.compute(model=model, X_test=X_test, y_test=y_test)
        results.update(res)
    return results


###########################################################
#               MAIN FUNCTION                            #
###########################################################


def main():
    global AVAILABLE_MODELS

    args = parse_args()

    # This line should be added before everything else
    # see the docstring of the function for more information.
    AVAILABLE_MODELS = update_available_models(args.models, AVAILABLE_MODELS)

    check_args(args)

    full_df = pd.DataFrame()

    run_name = namesgenerator.get_random_name()
    for model_name in args.models:
        for dataset_name in args.datasets:
            metrics = [AVAILABLE_METRICS[metric_name] for metric_name in args.metrics]
            results: Dict[str, float] = run_model_on_dataset(
                model_name, dataset_name, metrics, args.seed
            )

            df = pd.DataFrame(results, index=[0])
            df["model"] = model_name
            df["dataset"] = dataset_name
            full_df = pd.concat([full_df, df])

            print_results(model_name, dataset_name, results)

            if args.save_dir is not None:
                full_df.to_csv(f"{args.save_dir}/{run_name}.csv")


if __name__ == "__main__":
    main()
