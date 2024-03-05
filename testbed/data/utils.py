"""Helper functions to import preprocessed datasets in ./data/."""

from pathlib import Path

import numpy as np


def get_uci_data(dataset: str, dir_data: str = "./data", verbose: bool = False) -> np.ndarray:
    """
    Retrieve preprocessed data files from UCI directory.
    """
    path_dir = Path(dir_data) / "uci" / dataset
    path_file = path_dir / "data.npy"

    if path_dir.exists():
        if path_file.exists():
            data = np.load(path_file, allow_pickle=True).item()
            if verbose:
                n_obs, n_cov = data["x"].shape
                y_dim = data["y"].shape[1]
                print(f"dataset: {dataset}")
                print(
                    f"# of observations: {n_obs}, # of covariates: {n_cov}, dimension of outcome: {y_dim}"
                )  # TO DO: use logger
            return data
        else:
            path_readme = path_dir / "readme.md"
            raise FileNotFoundError(
                f"Data folder for '{dataset}' is present, but no preprocessed data were found. See {path_readme}."
            )
    else:
        raise FileNotFoundError(
            f"Dataset '{dataset}' not found in UCI directory. Use `list_uci_data` to print a list of available datasets."
        )


def list_uci_data(dir_data: str = "./data"):
    """
    Return list of datasets from UCI directory.
    """
    path = Path(dir_data) / "uci"
    datasets = sorted(x.name for x in path.iterdir() if x.is_dir())
    datasets = sorted(datasets)
    return datasets
