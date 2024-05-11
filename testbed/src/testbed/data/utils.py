"""Helper functions to import preprocessed datasets in ./data/."""

import json
import subprocess
import zipfile
from pathlib import Path
from typing import List
from typing import Union

import numpy as np
import requests


def _get_links():
    current_dir = Path(__file__).resolve().parent
    path_links = Path(current_dir) / "links.json"

    with path_links.open() as links:
        return json.load(links)


def _download_raw_data(url: str, path_raw_dataset_dir: Path, verbose: bool = False):
    response = requests.get(url, allow_redirects=False, timeout=10)
    if response.status_code == 200:
        format = url.rsplit(".", 1)[-1]
        path_raw_dataset_file = path_raw_dataset_dir / ("temp." + format)
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


def _load_data(path_dataset_file: Path, verbose: bool = False) -> np.ndarray:
    data = np.load(path_dataset_file, allow_pickle=True).item()
    if verbose:
        n_obs, n_cov = data["x"].shape
        y_dim = data["y"].shape[1]
        print(
            f"# of observations: {n_obs}, # of covariates: {n_cov}, dimension of outcome: {y_dim}"
        )

    return data


def list_data(verbose: bool = False) -> dict:
    """
    Lists all datasets in './data/' (current directory of this script).
    """
    current_dir = Path(__file__).resolve().parent
    path_data_dir = Path(current_dir)

    # extract all leaves of data folder
    datasets = []
    for d in path_data_dir.glob("**/"):
        if d.is_dir() and "raw" not in d.parts and not d.name.startswith((".", "_")):
            subdirs = [
                subdir
                for subdir in d.iterdir()
                if subdir.is_dir()
                and subdir.name != "raw"
                and not subdir.name.startswith((".", "_"))
            ]
            if len(subdirs) == 0:
                datasets.append(d)

    datasets = {d.name: str(d) for d in sorted(datasets)}
    if verbose:
        print(datasets)

    return datasets


def _get_data_path(dataset: str, verbose: bool = False) -> Path:
    datasets = list_data()
    if dataset in datasets:
        return Path(datasets[dataset])

    else:
        path_data_dir = Path(__file__).resolve().parent
        raise FileNotFoundError(
            f"Dataset '{dataset}' not found: {dataset} is not a subfolder of '{path_data_dir}'. Use `list_data` to print a list of available datasets."
        )


def get_data(
    datasets: Union[str, List[str]], verbose: bool = False
) -> Union[dict, List[dict]]:
    """
    Download or retrieve data files for one or multiple datasets. For each dataset, it checks if the
    preprocessed data file exists; if not, it downloads the raw data and then preprocesses it. Each dataset
    is loaded into a dictionary containing the keys `x` for covariates, `y` for outcome, and `categorical`
    for indices of categorical covariates.

    Parameters:
    - datasets (Union[str, List[str]]): A single dataset name as a string or a list of dataset names.

    Returns:
    - Union[dict, List[dict]]: A dictionary for a single dataset or a list of dictionaries for multiple
      datasets. The dictionaries contain the following keys:
         - `x`: a np.ndarray with features.
         - `y`: a np.ndarray with the outcome.
         - `categorical`: a list with the indices of categorical features.
    """
    if isinstance(datasets, list):
        return {dataset: get_data(dataset, verbose=verbose) for dataset in datasets}
    else:
        dataset = datasets

    if verbose:
        print(f"Getting {dataset} dataset.")

    path_dataset_dir = _get_data_path(dataset)
    path_dataset_file = path_dataset_dir / "data.npy"
    if not path_dataset_file.exists():
        path_raw_dataset_dir = path_dataset_dir / "raw"
        if not path_raw_dataset_dir.exists():
            path_raw_dataset_dir.mkdir()

        if not any(path_raw_dataset_dir.iterdir()):
            # download raw files
            links = _get_links()
            if dataset not in links:
                raise Exception(
                    f"Cannot download {dataset} dataset: no download link available."
                )

            _download_raw_data(links[dataset], path_raw_dataset_dir, verbose)

        # preprocess raw files
        path_preprocess_file = path_dataset_dir / "preprocess.py"
        if not path_preprocess_file.exists():
            raise Exception(
                f"Cannot preprocess {dataset} dataset. Preprocessing script {path_preprocess_file} is missing."
            )

        _preprocess_raw_data(path_dataset_dir, verbose)

    # load and return preprocessed data file
    return _load_data(path_dataset_file, verbose)


def _extract_and_delete_zipfile(path_raw_dataset_dir: Path, zipfile_name: str = "temp.zip"):
    path_raw_data_file = path_raw_dataset_dir / zipfile_name
    with zipfile.ZipFile(path_raw_data_file, "r") as archive:
        for file in archive.namelist():
            archive.extract(file, path_raw_dataset_dir)
    path_raw_data_file.unlink()
