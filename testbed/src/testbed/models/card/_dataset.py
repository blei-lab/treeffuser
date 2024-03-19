"""
Contains a dataset class that generalizes the datasets used in the CARD repo.

This file is mainly an adaptation from so that it can be used in the testbed.
Most of the design decisions are based on the specific structure of the Card model.

https://github.com/XzwHan/CARD/blob/main/regression/data_loader.py
"""

from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import torch as t
from jaxtyping import Float
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _onehot_encode_cat_feature(
    X: Float[np.ndarray, "n_samples x_dim"], cat_idx: List[int]
) -> Tuple[Float[np.ndarray, "n_samples new_dim"], int]:
    """
    Apply one-hot encoding to the categorical variable(s) in the feature set,
    they get added to the end of the feature set.
    """
    # select numerical features
    X_num = np.delete(arr=X, obj=cat_idx, axis=1)
    # select categorical features
    X_cat = X[:, cat_idx]
    X_onehot_cat = []
    for col in range(X_cat.shape[1]):
        X_onehot_cat.append(pd.get_dummies(X_cat[:, col], drop_first=True))

    X_onehot_cat = np.concatenate(X_onehot_cat, axis=1).astype(np.float32)
    dim_cat = X_onehot_cat.shape[1]  # number of categorical feature(s)
    X = np.concatenate([X_num, X_onehot_cat], axis=1)
    return X, dim_cat


def _preprocess_feature_set(
    X: Float[np.ndarray, "n_samples x_dim"], cat_idx: Optional[List[int]] = None
) -> Tuple[Float[np.ndarray, "n_samples x_dim"], int]:
    """
    Adds one-hot encoding to the feature set if there are categorical features.
    The categorical features get added to the end of the feature set as one-hot
    encoded features.
    """
    dim_cat = 0
    if cat_idx is not None and len(cat_idx) > 0:
        X, dim_cat = _onehot_encode_cat_feature(X, cat_idx)
    return X, dim_cat


class Dataset:
    def __init__(
        self,
        X: Float[np.ndarray, "n_samples x_dim"],
        y: Float[np.ndarray, "n_samples x_dim"],
        cat_idx: Optional[List[int]] = None,
        validation: bool = False,
        test_ratio: float = 0.2,
        validation_ratio: float = 0.2,
        seed: int = 0,
    ) -> None:

        np.random.seed(seed)

        self.x, self.dim_cat = _preprocess_feature_set(X, cat_idx)
        self.y = y

        # Split the data into train and test
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=test_ratio, random_state=seed
        )

        if validation:
            # Split the train data into train and validation
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=validation_ratio, random_state=seed
            )
            self.x_val, self.y_val = x_val, y_val

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.train_n_samples = self.x_train.shape[0]
        self.test_n_samples = self.x_test.shape[0]

        self.train_dim_x = self.x_train.shape[1]
        self.train_dim_y = self.y_train.shape[1]

        self.test_n_samples = self.x_test.shape[0]
        self.test_dim_x = self.x_test.shape[1]
        self.test_dim_y = self.y_test.shape[1]

        self._normalize_train_test_x()
        self._normalize_train_test_y()

        # Convert to torch tensors
        self.x_train = t.tensor(self.x_train, dtype=t.float32)
        self.y_train = t.tensor(self.y_train, dtype=t.float32)
        self.x_test = t.tensor(self.x_test, dtype=t.float32)
        self.y_test = t.tensor(self.y_test, dtype=t.float32)

    def _normalize_train_test_x(self):
        """
        When self.dim_cat > 0, we have one-hot encoded number of categorical variables,
        on which we don't conduct standardization. They are arranged as the last
        columns of the feature set.
        """
        self.scaler_x = StandardScaler(with_mean=True, with_std=True)
        if self.dim_cat == 0:
            self.x_train = self.scaler_x.fit_transform(self.x_train)
            self.x_test = self.scaler_x.transform(self.x_test)
        else:
            x_train_num, x_train_cat = (
                self.x_train[:, : -self.dim_cat],
                self.x_train[:, -self.dim_cat :],
            )
            x_test_num, x_test_cat = (
                self.x_test[:, : -self.dim_cat],
                self.x_test[:, -self.dim_cat :],
            )
            x_train_num = self.scaler_x.fit_transform(x_train_num)
            x_test_num = self.scaler_x.transform(x_test_num)

            self.x_train = np.concatenate([x_train_num, x_train_cat], axis=1)
            self.x_test = np.concatenate([x_test_num, x_test_cat], axis=1)

    def _normalize_train_test_y(self):
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        self.y_train = self.scaler_y.fit_transform(self.y_train)
        self.y_test = self.scaler_y.transform(self.y_test)

    def return_dataset(self, split="train"):
        if split == "train":
            dataset = t.cat([self.x_train, self.y_train], dim=1)
        elif split == "test":
            dataset = t.cat([self.x_test, self.y_test], dim=1)
        return dataset

    def summary_dataset(self, split="train"):
        if split == "train":
            return {
                "n_samples": self.train_n_samples,
                "dim_x": self.train_dim_x,
                "dim_y": self.train_dim_y,
            }
        else:
            return {
                "n_samples": self.test_n_samples,
                "dim_x": self.test_dim_x,
                "dim_y": self.test_dim_y,
            }
