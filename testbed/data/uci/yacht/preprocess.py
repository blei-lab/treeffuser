import argparse
import zipfile
from pathlib import Path

import numpy as np


def main(path_raw_dataset_dir: Path):
    # unzip and delete original raw files
    path_raw_data_file = path_raw_dataset_dir / "temp.zip"
    with zipfile.ZipFile(path_raw_data_file, "r") as raw_zip:
        raw_zip.extractall(path_raw_dataset_dir)
    path_raw_data_file.unlink()

    # extract outcome and covariates
    x = np.genfromtxt(path_raw_dataset_dir / "yacht_hydrodynamics.data", skip_header=False)
    y = x[:, -1].copy().reshape((-1, 1))
    x = np.delete(x, -1, 1)
    categorical = []

    np.save(
        path_raw_dataset_dir.parent / "data.npy", {"x": x, "y": y, "categorical": categorical}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
