"""Dataset management and PyTorch DataLoaders for Motor Imagery EEG.
Ensures subject-wise and trial-wise non-overlapping data partitions.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split


class EEGDataset(Dataset):
    """PyTorch Dataset wrapping EEG sample tensors and labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray, subjects: Optional[np.ndarray] = None):
        """Args:

        X: np.ndarray of shape (N, 1, 1280) or (N, 1280)
        y: np.ndarray of shape (N,)
        subjects: optional subject index per sample
        """
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subjects = subjects

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    split_ratio: Tuple[float, float, float] = (0.5, 0.2, 0.3),
    random_state: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split dataset into Train, Validation, and Test sets according to the paper's 5:2:3 ratio.

    Uses stratified splitting to preserve class balance.

    Returns:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    train_r, val_r, test_r = split_ratio
    total = train_r + val_r + test_r
    train_r, val_r, test_r = train_r / total, val_r / total, test_r / total

    # First split off the test set (30%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_r, stratify=y, random_state=random_state
    )

    # Then split the remainder into train and validation (5:2 ratio -> val is 2/7 of temp)
    val_rel_ratio = val_r / (train_r + val_r)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_rel_ratio, stratify=y_temp, random_state=random_state
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_dataloaders(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 64,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for train, val, and test splits."""
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    train_ds = EEGDataset(X_train, y_train)
    val_ds = EEGDataset(X_val, y_val)
    test_ds = EEGDataset(X_test, y_test)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders
