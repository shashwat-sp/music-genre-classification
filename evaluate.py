"""Evaluation utilities for music genre classification models."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any
from collections import defaultdict


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    label_encoder,
    plot_confusion: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a PyTorch model on test data.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        label_encoder: Label encoder for inverse transform
        plot_confusion: Whether to plot confusion matrix
    
    Returns:
        Dictionary with evaluation metrics
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(1)
            
            total_correct += (preds == y_batch).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    n = len(test_loader.dataset)
    accuracy = total_correct / n
    avg_loss = total_loss / n
    class_names = label_encoder.classes_
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Test Loss     : {avg_loss:.4f}")
    
    # Classification Report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        class_correct[label] += int(pred == label)
    
    print("\n" + "="*60)
    print("PER-CLASS ACCURACY")
    print("="*60)
    per_class_acc = {}
    for idx in sorted(class_total):
        name = label_encoder.inverse_transform([idx])[0]
        acc = class_correct[idx] / class_total[idx]
        per_class_acc[name] = acc
        bar = "█" * int(acc * 30)
        print(f"  {name:12s}  {acc:.4f}  {bar}")
    
    # Confusion Matrix
    if plot_confusion:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Genre', fontsize=12)
        plt.ylabel('True Genre', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'per_class_accuracy': per_class_acc,
    }


def get_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get predictions and probabilities from a model.
    
    Args:
        model: PyTorch model
        loader: Data loader
        device: Device to evaluate on
    
    Returns:
        Tuple of (labels, predictions, probabilities)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs)
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Compute ROC curves for multi-class classification.
    
    Args:
        y_true: True labels
        y_probs: Probability predictions
        num_classes: Number of classes
    
    Returns:
        Dictionary with ROC data for each class
    """
    y_bin = label_binarize(y_true, classes=range(num_classes))
    roc_data = {}
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data[i] = (fpr, tpr, roc_auc)
    
    return roc_data