"""This file contains simple classes for specifying datamodules to be used in PyTorch Lightning models."""

import torch
from jaxtyping import Float
from lightning import LightningDataModule
from numpy import ndarray
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


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
        X_train (ndarray): Input features array of shape (batch_size, x_dim).
        y_train (ndarray): Target labels array of shape (batch_size, y_dim).
        X_val (ndarray): Input features array of shape (batch_size, x_dim).
        y_val (ndarray): Target labels array of shape (batch_size, y_dim).
        batch_size (int): Batch size for data loaders (default: 32).
    """

    def __init__(
        self,
        X_train: Float[ndarray, "batch x_dim"],
        y_train: Float[ndarray, "batch y_dim"],
        X_val: Float[ndarray, "batch x_dim"],
        y_val: Float[ndarray, "batch y_dim"],
        batch_size=32,
        num_workers=0,
    ):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size
        cpu_count = torch.multiprocessing.cpu_count() // 2
        self.num_workers = num_workers if num_workers != -1 else cpu_count

    def setup(self, stage=None):
        """Set up the datasets for training, validation, and testing.

        This method is called by PyTorch Lightning to prepare the data.

        Args:
            stage (str): The stage of the training process (fit, validate, test, or predict).
        """
        self.train_dataset = _SimpleDataset(self.X_train, self.y_train)
        self.val_dataset = _SimpleDataset(self.X_val, self.y_val)

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
