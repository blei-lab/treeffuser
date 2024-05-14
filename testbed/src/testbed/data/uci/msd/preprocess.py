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
    data = pd.read_csv(path_raw_dataset_dir / "YearPredictionMSD.txt", header=None)

    # add custom test set
    i_test = 51630
    X_test = data.iloc[-i_test:, 1:]
    y_test = y = data.iloc[-i_test:, 0]

    # extract outcome and covariates
    X = data.iloc[:-i_test, 1:]
    y = data.iloc[:-i_test, 0]
    categorical = []

    k_fold_splits = _assign_k_splits(X.values.shape[0], 10, 0)

    # save preprocessed data
    np.save(
        path_raw_dataset_dir.parent / "data.npy",
        {
            "x": X.values,
            "y": y.values,
            "categorical": categorical,
            "test": {"x": X_test.values, "y": y_test.values},
            "k_fold_splits": k_fold_splits,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
