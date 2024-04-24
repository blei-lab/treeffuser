import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from testbed.data.utils import _extract_and_delete_zipfile


def main(path_raw_dataset_dir: Path):
    # unzip and delete original arhchive with raw files
    _extract_and_delete_zipfile(path_raw_dataset_dir)

    # import data
    data = pd.read_excel(path_raw_dataset_dir / "ENB2012_data.xlsx")

    # extract outcome and covariates
    X = data.iloc[:, :-2]
    y = data.iloc[:, -2:]
    categorical = []

    # save preprocessed data
    np.save(
        path_raw_dataset_dir.parent / "data.npy",
        {"x": X.values, "y": y.values, "categorical": categorical},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
