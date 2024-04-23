"""This file contains simple classes for specifying datamodules to be used in PyTorch Lightning models."""

import torch
from jaxtyping import Float
from lightning import LightningDataModule
from numpy import ndarray
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split


def collate_fn_tensordataset(batch):
    """
    Collate function for tensor dataset to UQ framework.
    this is how uq framework expects the data to be in the dataloader
    """
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return {"input": inputs, "target": targets}


class _SimpleDataset(Dataset):
    """A simple dataset class that wraps NumPy arrays.

    Args:
        X (ndarray): Input features array of shape (batch_size, x_dim).
        y (ndarray): Target labels array of shape (batch_size, y_dim).
    """

    def __init__(self, X: Float[ndarray, "batch x_dim"], y: Float[ndarray, "batch y_dim"]):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GenericDataModule(LightningDataModule):
    """A generic data module for PyTorch Lightning.

    This class handles the data splitting and provides data loaders for training,
    validation, and testing.

    Args:
        X (ndarray): Input features array of shape (batch_size, x_dim).
        y (ndarray): Target labels array of shape (batch_size, y_dim).
        batch_size (int): Batch size for data loaders (default: 32).
        train_split (float): Ratio of data to be used for training (default: 0.8).
        val_split (float): Ratio of data to be used for validation (default: 0.1).
        test_split (float): Ratio of data to be used for testing (default: 0.1).
    """

    def __init__(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        batch_size=32,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        num_workers=0,
    ):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        cpu_count = torch.multiprocessing.cpu_count() // 2
        self.num_workers = num_workers if num_workers != -1 else cpu_count

        assert (
            self.train_split + self.val_split + self.test_split == 1.0
        ), "Split percentages should sum to 1.0"

    def setup(self, stage=None):
        """Set up the datasets for training, validation, and testing.

        This method is called by PyTorch Lightning to prepare the data.

        Args:
            stage (str): The stage of the training process (fit, validate, test, or predict).
        """
        dataset = _SimpleDataset(self.X, self.y)
        dataset_len = len(dataset)

        train_len = int(dataset_len * self.train_split)
        val_len = int(dataset_len * self.val_split)
        test_len = dataset_len - train_len - val_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_len, val_len, test_len]
        )

    def train_dataloader(self):
        """Return the data loader for the training dataset."""
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_tensordataset,
        )
        return loader

    def val_dataloader(self):
        """Return the data loader for the validation dataset."""
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )
        return loader

    def test_dataloader(self):
        """Return the data loader for the testing dataset."""
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )
        return loader
