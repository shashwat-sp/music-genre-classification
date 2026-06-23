"""Dataset classes for music genre classification."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import platform


class MelDataset(Dataset):
    """Dataset for mel-spectrogram based classification."""
    
    def __init__(self, X: torch.Tensor, y: torch.Tensor, augment: bool = False):
        """
        Args:
            X: Mel-spectrogram tensors
            y: Labels
            augment: Whether to apply augmentation
        """
        self.X = X
        self.y = y
        self.augment = augment
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        if self.augment:
            x = self.apply_augmentation(x)
        return x, self.y[idx]
    
    def apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to the spectrogram."""
        # Time shift
        if torch.rand(1) < 0.4:
            shift = torch.randint(-20, 20, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=2)
        
        # Frequency masking
        if torch.rand(1) < 0.4:
            mask_size = torch.randint(5, 15, (1,)).item()
            start = torch.randint(0, x.shape[1] - mask_size, (1,)).item()
            if start >= 0 and start + mask_size <= x.shape[1]:
                x[:, start:start+mask_size, :] = 0
        
        # Time masking
        if torch.rand(1) < 0.4:
            mask_size = torch.randint(10, 40, (1,)).item()
            start = torch.randint(0, x.shape[2] - mask_size, (1,)).item()
            if start >= 0 and start + mask_size <= x.shape[2]:
                x[:, :, start:start+mask_size] = 0
        
        # Add noise
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
            x = torch.clamp(x, 0, 1)
        
        return x


def create_data_loaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 32,
    augment: bool = True,
    num_workers: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size for dataloaders
        augment: Whether to apply augmentation on training set
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if num_workers is None:
        is_windows = platform.system() == 'Windows'
        num_workers = 0 if is_windows else 4
    
    pin_memory = not platform.system() == 'Windows'
    
    train_dataset = MelDataset(X_train, y_train, augment=augment)
    val_dataset = MelDataset(X_val, y_val, augment=False)
    test_dataset = MelDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def get_segmented_data(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    segment_duration: float = 3.0,
    hop_duration: float = 1.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return X_train, y_train, X_val, y_val, X_test, y_test