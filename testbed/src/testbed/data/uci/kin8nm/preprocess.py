import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from testbed.data.utils import _assign_k_splits


def main(path_raw_dataset_dir: Path):
    # Process the raw data file named temp.csv
    raw_data_path = path_raw_dataset_dir / "data.zip"
    # unzip the data
    with zipfile.ZipFile(raw_data_path, "r") as zip_ref:
        zip_ref.extractall(path_raw_dataset_dir)
    raw_data_path = path_raw_dataset_dir / "data.csv"

    df = pd.read_csv(raw_data_path)
    y = df["y"].values
    x = df.drop(columns=["y"]).values
    categorical = []

    k_fold_splits = _assign_k_splits(x.shape[0], 10, 0)

    # Save preprocessed data
    np.save(
        path_raw_dataset_dir.parent / "data.npy",
        {
            "x": x,
            "y": y,
            "categorical": categorical,
            "k_fold_splits": k_fold_splits,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
