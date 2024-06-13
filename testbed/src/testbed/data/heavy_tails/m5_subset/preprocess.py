import argparse
import pickle as pkl
from pathlib import Path

import numpy as np


def main(path_raw_dataset_dir: Path):
    # Process the raw data file named temp.csv
    raw_data_path = path_raw_dataset_dir / "data.pkl"

    with open(raw_data_path, "rb") as f:  # noqa: PTH123
        data = pkl.load(f)  # noqa: S301

    # gross
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"].round()
    y_test = data["y_test"].round()
    categorical = data["cat_cols"]
    col_names = data["col_names"]
    categorical = [col_names.index(col) for col in categorical]

    new_data_dict = {
        "x": X_train,
        "y": y_train,
        "test": {"x": X_test, "y": y_test},
        "categorical": categorical,
    }

    # Save the preprocessed data
    np.save(path_raw_dataset_dir.parent / "data.npy", new_data_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
