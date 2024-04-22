import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def main(path_raw_dataset_dir: Path):
    # unzip and delte original raw files
    path_raw_data_file = path_raw_dataset_dir / "temp.zip"
    with zipfile.ZipFile(path_raw_data_file, "r") as archive:
        for file in archive.namelist():
            archive.extract(file, path_raw_dataset_dir)
    path_raw_data_file.unlink()

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

    # save preprocessed data
    np.save(
        path_raw_dataset_dir.parent / "data.npy",
        {
            "x": X.values,
            "y": y.values,
            "categorical": categorical,
            "test": {"x": X_test.values, "y": y_test.values},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
