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
    data = pd.read_csv(
        path_raw_dataset_dir / "Dataset/Training/Features_Variant_1.csv", header=None
    )

    # extract outcome and covariates
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    categorical = [3]

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
