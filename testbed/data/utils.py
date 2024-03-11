"""Helper functions to import preprocessed datasets in ./data/."""

import json
import subprocess
from pathlib import Path

import numpy as np
import requests


def _get_links():
    path_links = Path("./data/links.json")
    with path_links.open() as links:
        return json.load(links)


def _download_raw_data(url: str, path_raw_dataset_dir: Path, verbose: bool = False):
    response = requests.get(url, allow_redirects=False, timeout=10)
    if response.status_code == 200:
        path_raw_dataset_file = path_raw_dataset_dir / "temp.zip"
        path_raw_dataset_file.open("wb").write(response.content)
        if verbose:
            print(f"Raw data files downloaded successfully in {path_raw_dataset_dir}.")

    else:
        raise Exception(
            f"The URL {url} is unavailable. Download dataset manually and then follow instructions in readme file. Status code: {response.status_code}."
        )


def _preprocess_raw_data(path_dataset_dir: Path, verbose: bool = False):
    path_preprocess_script = path_dataset_dir / "preprocess.py"
    path_raw_dataset_dir = path_dataset_dir / "raw"
    subprocess.run(
        [  # noqa: S603, S607 (S603 `subprocess` call: check for execution of untrusted input,  S607 Starting a process with a partial executable path)
            "python3",
            path_preprocess_script,
            path_raw_dataset_dir,
        ],
        check=True,
    )
    if verbose:
        print("Preprocessing completed.")


def get_uci_data(dataset: str, data_dir: str = "./data", verbose: bool = False) -> np.ndarray:
    """
    Retrieve preprocessed data files from UCI directory.
    """
    if verbose:
        print(f"Getting UCI {dataset} dataset.")

    path_dataset_dir = Path(data_dir) / "uci" / dataset
    path_dataset_file = path_dataset_dir / "data.npy"

    if path_dataset_dir.exists():
        if not path_dataset_file.exists():
            links = _get_links()
            path_raw_dataset_dir = path_dataset_dir / "raw"
            _download_raw_data(links[dataset], path_raw_dataset_dir, verbose)
            _preprocess_raw_data(path_dataset_dir, verbose)

        data = np.load(path_dataset_file, allow_pickle=True).item()
        if verbose:
            n_obs, n_cov = data["x"].shape
            y_dim = data["y"].shape[1]
            print(
                f"# of observations: {n_obs}, # of covariates: {n_cov}, dimension of outcome: {y_dim}"
            )
        return data

    else:
        raise FileNotFoundError(
            f"Dataset '{dataset}' not found in UCI directory. Use `list_uci_data` to print a list of available datasets."
        )


def list_uci_data(data_dir: str = "./data"):
    """
    Return list of datasets from UCI directory.
    """
    path = Path(data_dir) / "uci"
    datasets = sorted(x.name for x in path.iterdir() if x.is_dir())
    datasets = sorted(datasets)
    return datasets


def get_uci_data_all(data_dir: str = "./data", verbose: bool = False):
    return {dataset: get_uci_data(dataset, verbose=verbose) for dataset in list_uci_data()}
