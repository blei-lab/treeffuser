# Data Utilities

This folder contains utilities to automate the downloading and preprocessing of datasets.

## Supported datasets

The utilities currently support the following datasets:

- **UCI Datasets:**
  - `naval`
  - `protein`
  - `wine`
  - `yacht`

## Importing supported datasets

The utility consists of two primary functions: `list_data` and `get_data`.

- `list_data` lists all the available datasets in the `./data/` directory.

    ```python
    list_data()
    ```

- `get_data` is used to import a supported dataset or datasets. To import a single dataset, such as `"naval"`, simply call:

    ```python
    get_data("naval")
    ```

    Multiple datasets can be imported simultaneously by passing a list of dataset names.

    ```python
    get_data(["naval", "yacht"])
    ```


## Adding a new dataset

To introduce a new dataset:

1. Create a folder within `./data`, e.g., `./data/new-dataset/`.
2. Prepare the dataset by **either**:
   - Adding a preprocessed `data.npy` file directly to `new-dataset/`. See [preprocessed data files](#preprocessed-data-files) for format. **or**
   - Including a `preprocess.py` script, and then either:
     - Manually place raw files in `new-dataset/raw/`, **or**
     - Add a download link in `links.json` ([how to add links](#adding-download-links)).
3. Verify the dataset has been added successfully:
   - Use `list_data()` to see if the new dataset is listed.
   - Import the dataset with `get_data("new-dataset")` to ensure it loads correctly.

**Note**: Dataset folders can be nested to organize datasets into categories. For example, all UCI datasets are placed under `./data/uci/`.

## Preprocessed data files

Preprocessed data should be saved in `data.npy` files. These files contain a dictionary with three keys:
- `x`: a `np.ndarray` with features.
- `y`: a `np.ndarray` with the outcome.
- `categorical`: a list with the indices of categorical features. Use an empty list if there are no categorical variables.

## Adding download links

To add download links for automatic data retrieval, edit the `links.json` file in `./data`. Add the dataset name with the respective download URL: `"new-dataset": "https://example.com/.../raw.format"`. [Note that the link should end with the data file's format](#note). When downloaded automatically, the raw data file will be saved as `temp.format`.

## Preprocessing script

The `preprocess.py` script, placed in `./data/new-dataset`, should create a `data.npy` file in the same directory (see [preprocessed data files](#preprocessed-data-files) for format). Raw data downloaded automatically are stored as `temp.format` in `./data/new-dataset/raw`.

Here's a template for `preprocess.py`, that assumes that the raw file is named `temp.csv`.

```python
import argparse
from pathlib import Path
import numpy as np

def main(path_raw_dataset_dir: Path):
    # Process the raw data file named temp.csv
    raw_data_path = path_raw_dataset_dir / "temp.format"

    # Example: extracting outcome and covariates
    x = np.genfromtxt(raw_data_path, delimiter=",", skip_header=True)
    y = x[:, 0].copy().reshape((-1, 1))
    x = np.delete(x, 0, 1)
    categorical = [12]  # Specify categorical feature indices if any

    # Save preprocessed data
    np.save(path_raw_dataset_dir.parent / "data.npy", {"x": x, "y": y, "categorical": categorical})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
```

## Note

The dataset links are updated as of March 9, 2024. Any changes to the raw files after this date may require adjustments to the preprocessing scripts.
