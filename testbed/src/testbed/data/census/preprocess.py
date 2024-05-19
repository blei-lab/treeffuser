import argparse
from pathlib import Path

import numpy as np
from testbed.data.utils import _assign_k_splits


def main(path_raw_dataset_dir: Path):
    # extract dat file
    # from ipumspy import readers, ddi
    # path_to_cookbook = path_raw_dataset_dir / "usa_00004.xml"
    # path_to_datafile = path_raw_dataset_dir / "usa_00004.dat"
    # ddi_codebook = readers.read_ipums_ddi(path_to_cookbook)
    # data = readers.read_microdata(ddi_codebook, path_to_datafile)
    # np.save(path_raw_dataset_dir / "usa_00004.npy", data)

    # import data
    data = np.load(
        path_raw_dataset_dir / "usa_00004.npy", allow_pickle=True
    )  # to access metadata for a given column, visit https://usa.ipums.org/usa-action/variables/COLUMN_NAME#description_section

    # drop year, sample
    drop_cols = [
        "YEAR",
        "SAMPLE",
        "SERIAL",
        "CBSERIAL",
        "HHWT",
        "CLUSTER",
        "STRATA",
        "PERNUM",
        "PERWT",
        "RACED",
        "HISPAND",
        "DEGFIELDD",
    ]
    data = data.drop(drop_cols, axis=1)

    # convert to float
    convert_to_float_cols = ["INCTOT", "INCWELFR", "INCRETIR", "MIGRATE1"]
    data[convert_to_float_cols] = data[convert_to_float_cols].astype("float64")

    # extract outcome and covariates
    y_cols = ["INCTOT"]  # or wage income?
    X = data.loc[:, [col for col in data.columns if col not in y_cols]]
    y = data.iloc[:, y_cols]

    # categorical columns
    categorical_cols = [
        "GQ",
        "RACE",
        "HISPAN",
        "MIGRATE1",
        "MIGRATE1D",
        "HCOVANY",
        "SCHOOL",
        "SCHLTYPE",
        "DEGFIELD",
    ]
    categorical = [data.columns.get_loc(col) for col in categorical_cols]

    k_fold_splits = _assign_k_splits(X.values.shape[0], 10, 0)

    # save preprocessed data
    np.save(
        path_raw_dataset_dir.parent / "data.npy",
        {
            "x": X.values,
            "y": y.values,
            "categorical": categorical,
            "k_fold_splits": k_fold_splits,
            "x_names": [col for col in data.columns if col not in y_cols],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
