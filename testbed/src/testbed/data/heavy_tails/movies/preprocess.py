import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from testbed.data.utils import _assign_k_splits

final_X_cols = [
    "genre",  # cat
    "rating",  # cat
    "score",  # num
    "votes",  # num
    "director",  # cat
    "star",  # cat
    "country",  # cat
    "company",  # cat
    "runtime",  # num
]

final_y_cols = ["gross"]

categorical = [0, 1, 4, 5, 6, 7]


def main(path_raw_dataset_dir: Path):
    # Process the raw data file named temp.csv
    raw_data_path = path_raw_dataset_dir / "temp.csv"

    # Read the raw data
    data = pd.read_csv(raw_data_path)

    # gross
    X = data[final_X_cols]
    y = data[final_y_cols]

    # Convert categorical columns to integers
    for col in categorical:
        nan_mask = X.iloc[:, col].isna()
        X.iloc[:, col] = X.iloc[:, col].astype("category").cat.codes
        # Put nans back
        X.loc[nan_mask, X.columns[col]] = np.nan

    # remove rows with nans
    x_has_nan = X.isna().any(axis=1)
    y_has_nan = y.isna().any(axis=1)

    has_nan = x_has_nan | y_has_nan

    X = X[~has_nan]
    y = y[~has_nan]

    k_fold_splits = _assign_k_splits(X.values.shape[0], 10, 0)

    # Save the preprocessed data
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
