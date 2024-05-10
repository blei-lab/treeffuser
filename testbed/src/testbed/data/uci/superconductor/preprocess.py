import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from testbed.data.utils import _extract_and_delete_zipfile


def main(path_raw_dataset_dir: Path):
    # unzip and delete original arhchive with raw files
    _extract_and_delete_zipfile(path_raw_dataset_dir)

    # import data
    data = pd.read_csv(path_raw_dataset_dir / "train.csv")

    # extract outcome and covariates
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    categorical = []

    # save preprocessed data
    np.save(
        path_raw_dataset_dir.parent / "data.npy",
        {"x": X.values, "y": y.values.reshape(-1, 1), "categorical": categorical},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
