"""
This notebook contains the code neccesary for generating plots and tables needed
for the run-time analysis of the algorithm.
"""

from testbed.data.utils import get_data  # noqa E402
from testbed.data.utils import list_data  # noqa E402
from testbed.models.treeffuser import Treeffuser  # noqa E402


from dataclasses import dataclass
from argparse import ArgumentParser
from typing import List
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import seaborn as sns
import os

import time
from jaxtyping import Float
import pandas as pd
import pickle

# make plots pretty
sns.set(style="whitegrid")

N_ATTEMPTS = 5
FILE_NAME_TRAIN_TIMES = "train_times.pdf"
FILE_NAME_SAMPLE_TIMES = "sample_times.csv"
FILE_NAME_M5_SUBSET = "m5_subset.pdf"
FILE_NAME_DATAPOINT_PER_DATASET = "datapoint_per_dataset.pkl"
FILE_NAME_DATAPOINT_PER_FRACTION = "datapoint_per_fraction.pkl"
ATTEMPT_LOAD_DATAPOINTS = True

N_SUBSETS = 10


# Simply to match the paper (we could run on all but it would take too long)

NAMES_TO_PLOT = {
    "kin8nm": "kin8nm",
    "m5_subset": "m5",
    "bike": "bike",
    "energy": "energy",
    "kin8nm": "kin8nm",
    "movies": "movies",
    "naval": "naval",
    "news": "news",
    "power": "power",
    "superconductor": "superc",
    "wine": "wine",
    "yacht": "yacht",
}


@dataclass
class _Datapoint:
    """
    Class representing a single datapoint to plot. Just for convenience
    and for cleaner code.
    """

    # name of the dataset
    dataset_name: str
    # dataset shape used for fitting the model
    dataset_shape: tuple[int, int]
    # Number of samples taken (different from n_samples which should be 1)
    n_samples: int
    # mean and std of the time taken to fit the model
    fit_time_mean: float
    fit_time_std: float
    # mean and std of the time taken to sample from the mode2
    sample_time_mean: float
    sample_time_std: float


def parse_args():
    parser = ArgumentParser()
    msg = "Where to save the outputs of the notebook"
    parser.add_argument("--out-dir", type=str, default="output_speed", help=msg)
    return parser.parse_args()


def make_datapoint_from_dataset(
    dataset_name: str,
    x: Float[ndarray, "batch x_dim"],
    y: Float[ndarray, "batch y_dim"],
) -> _Datapoint:
    """
    Create a _Datapoint object from a dataset by fitting the model
    N_ATTEMPTS times and taking the mean and std of the time taken
    to fit and sample from the model.
    """

    t_fits = []
    t_samples = []
    n_sampled = None


    for _ in range(N_ATTEMPTS):
        t_start = time.time()
        model = Treeffuser()
        model.fit(x, y)
        t_fit = time.time() - t_start
        print(f"Time taken to fit model on {dataset_name}: {t_fit}")

        x_to_sample = np.concatenate([x, x, x, x, x, x ])
        x_to_sample = x_to_sample[: 1000]
        n_sampled = len(x_to_sample)

        t_start = time.time()
        _ = model.sample(x_to_sample, n_samples=1)
        t_sample = (time.time() - t_start)
        print(f"Time taken to sample from model on {dataset_name}: {t_sample}")

        t_fits.append(t_fit)
        t_samples.append(t_sample)

    t_fit_mean = np.mean(t_fits)
    t_fit_std = np.std(t_fits) / np.sqrt(N_ATTEMPTS)
    t_sample_mean = np.mean(t_samples)
    t_sample_std = np.std(t_samples) / np.sqrt(N_ATTEMPTS)

    datapoint = _Datapoint(
        dataset_name=dataset_name,
        dataset_shape=x.shape,
        n_samples=n_sampled,
        fit_time_mean=t_fit_mean,
        fit_time_std=t_fit_std,
        sample_time_mean=t_sample_mean,
        sample_time_std=t_sample_std,
    )
    return datapoint


def compute_fit_and_sample_time(dataset_name: str) -> _Datapoint:
    """
    This function computes the time taken to fit and
    sample from treeffuser on a given dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to use.

    Returns
    -------
    _Datapoint
        A _Datapoint object containing the results of the computation.
        See the _Datapoint class for more information.
    """
    data = get_data(dataset_name)
    x, y = data["x"], data["y"]
    # convert to float
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    print(f"Running on dataset: {dataset_name}")
    print(f"Shape of x: {x.shape}")
    return make_datapoint_from_dataset(dataset_name, x, y)


def plot_train_times(datapoints: List[_Datapoint], save_pth: str, annotate=True):
    # array with color for each datapoint
    colors = sns.color_palette("tab10", n_colors=len(datapoints))
    # Extract data for plotting
    x = [dp.dataset_shape[0] for dp in datapoints]
    y = [dp.fit_time_mean for dp in datapoints]
    yerr = [dp.fit_time_std for dp in datapoints]
    labels = [f"{NAMES_TO_PLOT[dp.dataset_name]}\n{dp.dataset_shape}" for dp in datapoints]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    for i in range(len(datapoints)):
        plt.errorbar(x[i], y[i], yerr=yerr[i], fmt="o", capsize=5, capthick=1, ecolor="red", color=colors[i], label=labels[i])


    # Annotate each point with the dataset name and shape
    if annotate:
        for i, label in enumerate(labels):
            plt.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(5, 10))
    else:
        plt.legend()

    # Add titles and labels
    plt.title("Training Times vs. Dataset Size")
    plt.xlabel("Number of Samples in Dataset")
    plt.ylabel("Mean Training Time (seconds)")
    plt.grid(True)
    plt.tight_layout()
    # set x lim
    plt.xlim(0, 1.2 * max(x))
    plt.ylim(0, 1.2 * max(y))

    # Save plot
    plt.savefig(save_pth)
    # save also png
    plt.savefig(save_pth.replace(".pdf", ".png"), dpi=200)


def make_table_for_sample_times(datapoints: List[_Datapoint], save_pth: str):

    df = pd.DataFrame(
        {
            "Dataset": [dp.dataset_name for dp in datapoints],
            "Time per sample (seconds)": [
                dp.sample_time_mean / dp.n_samples for dp in datapoints
            ],
        }
    )
    df.to_csv(save_pth, index=False, sep=",")
    df.to_latex(save_pth.replace(".csv", ".tex"), index=False)


def create_a_datapoint_per_dataset(
    dataset_names: List[str] = None,
    ) -> List[_Datapoint]:

    # remove m5 subset / not uciml datasets
    datapoints = []
    for dataset_name in dataset_names:
        dp = compute_fit_and_sample_time(dataset_name)
        datapoints.append(dp)
    return datapoints


def create_a_datapoint_per_fraction_of_dataset(
    dataset_name: str, n_fractions: int
) -> List[_Datapoint]:
    dataset = get_data(dataset_name)
    x, y = dataset["x"], dataset["y"]

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    datapoints = []
    fractions = np.linspace(0.1, 1, n_fractions)

    for fraction in fractions:
        subset = int(len(x) * fraction)
        x_subset = x[:subset]
        y_subset = y[:subset]
        dp = make_datapoint_from_dataset(dataset_name, x_subset, y_subset)
        datapoints.append(dp)

    return datapoints


def get_pkls(pkl_name: str):
    try:
        with open(pkl_name, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
def save_pkls(pkl_name: str, data):
    with open(pkl_name, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    dataset_names = list(list_data().keys())
    dataset_names = [name for name in dataset_names if name in NAMES_TO_PLOT.keys()]

    datapoints = get_pkls(os.path.join(args.out_dir, FILE_NAME_DATAPOINT_PER_DATASET))
    if datapoints is None or not ATTEMPT_LOAD_DATAPOINTS:
        datapoints = create_a_datapoint_per_dataset(dataset_names)
        save_pkls(os.path.join(args.out_dir, FILE_NAME_DATAPOINT_PER_DATASET), datapoints)

    plot_train_times(datapoints, os.path.join(args.out_dir, FILE_NAME_TRAIN_TIMES), annotate=False)
    make_table_for_sample_times(datapoints, os.path.join(args.out_dir, FILE_NAME_SAMPLE_TIMES))


    datapoints = get_pkls(os.path.join(args.out_dir, FILE_NAME_DATAPOINT_PER_FRACTION))
    if datapoints is None or not ATTEMPT_LOAD_DATAPOINTS:
        datapoints = create_a_datapoint_per_fraction_of_dataset("m5_subset", N_SUBSETS)
        save_pkls(os.path.join(args.out_dir, FILE_NAME_DATAPOINT_PER_FRACTION), datapoints)

    plot_train_times(datapoints, os.path.join(args.out_dir, FILE_NAME_M5_SUBSET), annotate=True)
