import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from testbed.data.utils import _assign_k_splits


def main(path_raw_dataset_dir: Path):
    # unzip and delete original arhchive with raw files
    # extract outcome and covariates

    column_names = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "MEDV",
    ]

    # Read the .data file
    df = pd.read_csv(path_raw_dataset_dir / "temp.data", sep=r"\s+", names=column_names)
    x = df.values
    y = x[:, -1].copy().reshape((-1, 1))
    x = np.delete(x, -1, 1)
    categorical = []

    k_fold_splits = _assign_k_splits(x.shape[0], 10, 0)

    np.save(
        path_raw_dataset_dir.parent / "data.npy",
        {"x": x, "y": y, "categorical": categorical, "k_fold_splits": k_fold_splits},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
