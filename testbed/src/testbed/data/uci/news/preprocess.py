import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from testbed.data.utils import _assign_k_splits
from testbed.data.utils import _extract_and_delete_zipfile


def main(path_raw_dataset_dir: Path):
    # unzip and delete original arhchive with raw files
    _extract_and_delete_zipfile(path_raw_dataset_dir)

    # import data
    data = pd.read_csv(path_raw_dataset_dir / "OnlineNewsPopularity/OnlineNewsPopularity.csv")

    # remove url and timedelta columns
    data = data.drop(columns=["url", " timedelta"])

    # extract outcome and covariates
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    categorical = []  # TODO: it contains some binary variables if we want to support them

    k_fold_splits = _assign_k_splits(X.values.shape[0], 10, 0)

    # save preprocessed data
    np.save(
        path_raw_dataset_dir.parent / "data.npy",
        {
            "x": X.values,
            "y": y.values.reshape(-1, 1),
            "categorical": categorical,
            "k_fold_splits": k_fold_splits,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
