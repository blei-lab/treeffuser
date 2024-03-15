import argparse
import zipfile
from pathlib import Path
from shutil import rmtree  # delete non-empty folders

import numpy as np


def main(path_raw_dataset_dir: Path):
    # unzip and delte original raw files
    path_raw_data_file = path_raw_dataset_dir / "temp.zip"
    with zipfile.ZipFile(path_raw_data_file, "r") as archive:
        for file in archive.namelist():
            if file.startswith("UCI CBM Dataset/"):
                archive.extract(file, path_raw_dataset_dir)
    path_raw_data_file.unlink()

    # delete "UCI CBM Dataset" folder after moving raw files
    path_uci_cbm_dir = path_raw_dataset_dir / "UCI CBM Dataset"
    for file in path_uci_cbm_dir.iterdir():
        if file.suffix == ".txt":
            file.rename(path_raw_dataset_dir / file.name)
    rmtree(path_uci_cbm_dir)

    # extract outcome and covariates
    x = np.genfromtxt(path_raw_dataset_dir / "data.txt", delimiter="   ", skip_header=False)
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
