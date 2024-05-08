import argparse
from pathlib import Path

import numpy as np
from testbed.data.utils import _extract_and_delete_zipfile


def main(path_raw_dataset_dir: Path):
    # unzip and delete original arhchive with raw files
    _extract_and_delete_zipfile(path_raw_dataset_dir)

    # extract outcome and covariates
    x = np.genfromtxt(path_raw_dataset_dir / "CASP.csv", delimiter=",", skip_header=True)
    y = x[:, 0].copy().reshape((-1, 1))
    x = np.delete(x, 0, 1)
    categorical = []

    np.save(
        path_raw_dataset_dir.parent / "data.npy", {"x": x, "y": y, "categorical": categorical}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
