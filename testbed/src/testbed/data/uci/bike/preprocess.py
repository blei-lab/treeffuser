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
    data = pd.read_csv(path_raw_dataset_dir / "hour.csv")

    # remove index and date columns
    data = data.drop(columns=["instant", "dteday"])

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
