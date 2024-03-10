import argparse
from pathlib import Path

import numpy as np


def main(path_raw_dataset_dir: Path):
    x = np.genfromtxt(path_raw_dataset_dir / "data.txt", delimiter="   ", skip_header=False)

    # extract outcome and covariates
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
