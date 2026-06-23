"""Training utilities for PyTorch models."""

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Tuple, List


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 15,
    weight_decay: float = 1e-4,
    save_path: str = "best_model.pth",
) -> Tuple[nn.Module, List[float], List[float], List[float], List[float]]:
    """
    Train a PyTorch model with early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        weight_decay: L2 regularization
        save_path: Path to save best model
    
    Returns:
        Tuple of (trained_model, train_losses, val_losses, train_accs, val_accs)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
            train_loop.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ", leave=False)
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)
                
                val_loop.set_postfix(loss=loss.item())
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved (val acc: {best_val_acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accs, val_accs


def train_improved_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 60,
    initial_lr: float = 1e-3,
    finetune_lr: float = 2e-4,
    weight_decay: float = 1e-4,
    patience: int = 12,
    unfreeze_epoch: int = 5,
    grad_clip: float = 1.0,
    save_path: str = "best_improved_model.pth",
) -> Tuple[nn.Module, List[float], List[float], List[float], List[float]]:
    """
    Train the improved model with two-phase training.
    
    Phase 1: Backbone frozen, only classifier head trains
    Phase 2: All parameters train with lower learning rate
    
    Args:
        model: ImprovedGenreClassifier instance
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Maximum number of epochs
        initial_lr: Learning rate for phase 1
        finetune_lr: Learning rate for phase 2
        weight_decay: L2 regularization
        patience: Early stopping patience
        unfreeze_epoch: Epoch to unfreeze backbone
        grad_clip: Gradient clipping value
        save_path: Path to save best model
    
    Returns:
        Tuple of (trained_model, train_losses, val_losses, train_accs, val_accs)
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Phase 1 optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=unfreeze_epoch, eta_min=1e-5
    )
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, num_epochs + 1):
        # Phase transition
        if epoch == unfreeze_epoch + 1:
            print(f"\n[Epoch {epoch}] Unfreezing backbone — switching to lr={finetune_lr}")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=finetune_lr, weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(num_epochs - unfreeze_epoch), eta_min=1e-6
            )
        
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_loop = tqdm(train_loader, desc=f"Ep {epoch:3d}/{num_epochs} [Train]", leave=False)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply SpecAugment on GPU
            inputs = apply_spec_augment_on_gpu(inputs)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            train_loop.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Ep {epoch:3d}/{num_epochs} [Val]  ", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:3d}/{num_epochs}  "
              f"Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}  Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved (val acc: {best_val_acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {patience} epochs).")
                break
    
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accs, val_accs


def apply_spec_augment_on_gpu(
    x: torch.Tensor,
    freq_mask_param: int = 20,
    time_mask_param: int = 40,
) -> torch.Tensor:
    """
    Apply SpecAugment on GPU tensors.
    
    Args:
        x: Input tensor of shape (batch, channels, height, width)
        freq_mask_param: Maximum frequency mask size
        time_mask_param: Maximum time mask size
    
    Returns:
        Augmented tensor
    """
    # Frequency masking
    if torch.rand(1) < 0.5:
        f = torch.randint(0, freq_mask_param, (1,)).item()
        f0 = torch.randint(0, max(1, x.shape[2] - f), (1,)).item()
        if f > 0 and f0 + f <= x.shape[2]:
            x[:, :, f0:f0+f, :] = 0
    
    # Time masking
    if torch.rand(1) < 0.5:
        t = torch.randint(0, time_mask_param, (1,)).item()
        t0 = torch.randint(0, max(1, x.shape[3] - t), (1,)).item()
        if t > 0 and t0 + t <= x.shape[3]:
            x[:, :, :, t0:t0+t] = 0
    
    return x