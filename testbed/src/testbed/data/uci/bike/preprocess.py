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
    data = pd.read_csv(path_raw_dataset_dir / "hour.csv")

    # remove index and date columns
    data = data.drop(columns=["instant", "dteday", "casual", "registered"])

    # extract outcome and covariates
    X = data[[x for x in data.columns if x != "cnt"]]
    y = data["cnt"]
    categorical_cols = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]
    categorical = [data.columns.get_loc(idx) for idx in categorical_cols]

    k_fold_splits = _assign_k_splits(X.values.shape[0], 10, 0)

    # save preprocessed data
    np.save(
        path_raw_dataset_dir.parent / "data.npy",
        {
            "x": X.values,
            "y": y.values,
            "categorical": categorical,
            "k_fold_splits": k_fold_splits,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
