import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from testbed.data.utils import _assign_k_splits


def main(path_raw_dataset_dir: Path):
    # Process the raw data file named temp.csv
    raw_data_path = path_raw_dataset_dir / "temp.arff"

    # Read the raw data
    data, _ = arff.loadarff(raw_data_path)
    data = pd.DataFrame(data).values

    X, y = data[:, :-1], data[:, -1]

    k_fold_splits = _assign_k_splits(X.shape[0], 10, 0)

    # Save the preprocessed data
    np.save(
        path_raw_dataset_dir.parent / "data.npy",
        {
            "x": X,
            "y": y,
            "categorical": [],
            "k_fold_splits": k_fold_splits,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
