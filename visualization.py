"""Visualization utilities for music genre classification."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict


# Style constants
FONT_FAMILY = "DejaVu Sans"
PALETTE = {
    "train": "#2563EB",
    "val": "#DC2626",
    "grid": "#E5E7EB",
    "bg": "#FAFAFA",
    "panel": "#FFFFFF",
    "text": "#111827",
    "subtext": "#6B7280",
    "accent": "#059669",
    "warn": "#D97706",
}
GENRE_COLORS = [
    "#1E40AF", "#7C3AED", "#BE185D", "#B45309",
    "#065F46", "#0E7490", "#92400E", "#1D4ED8",
    "#6D28D9", "#047857",
]


def apply_base_style(
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    grid: bool = True,
) -> None:
    """Apply consistent styling to matplotlib axes."""
    ax.set_facecolor(PALETTE["panel"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D1D5DB")
    ax.tick_params(colors=PALETTE["subtext"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["subtext"])
    ax.yaxis.label.set_color(PALETTE["subtext"])
    
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color=PALETTE["text"], pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if grid:
        ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accs: Training accuracies per epoch
        val_accs: Validation accuracies per epoch
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             facecolor=PALETTE["bg"])
    fig.subplots_adjust(hspace=0.35, wspace=0.28,
                        left=0.07, right=0.97, top=0.88, bottom=0.12)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax = axes[0]
    apply_base_style(ax, title="Training & Validation Loss",
                     xlabel="Epoch", ylabel="Cross-Entropy Loss")
    ax.plot(epochs, train_losses, color=PALETTE["train"],
            linewidth=2, label="Train", zorder=3)
    ax.plot(epochs, val_losses, color=PALETTE["val"],
            linewidth=2, label="Validation", zorder=3)
    ax.fill_between(epochs, train_losses, val_losses,
                    alpha=0.08, color=PALETTE["val"])
    
    best_val_ep = int(np.argmin(val_losses)) + 1
    ax.axvline(best_val_ep, color=PALETTE["accent"],
               linestyle="--", linewidth=1.2, zorder=2,
               label=f"Best val (ep {best_val_ep})")
    ax.legend(fontsize=8.5, framealpha=0, labelcolor=PALETTE["text"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Accuracy plot
    ax = axes[1]
    apply_base_style(ax, title="Training & Validation Accuracy",
                     xlabel="Epoch", ylabel="Accuracy")
    ax.plot(epochs, train_accs, color=PALETTE["train"],
            linewidth=2, label="Train", zorder=3)
    ax.plot(epochs, val_accs, color=PALETTE["val"],
            linewidth=2, label="Validation", zorder=3)
    ax.fill_between(epochs, train_accs, val_accs,
                    alpha=0.08, color=PALETTE["val"])
    
    best_acc_ep = int(np.argmax(val_accs)) + 1
    ax.axvline(best_acc_ep, color=PALETTE["accent"],
               linestyle="--", linewidth=1.2, zorder=2,
               label=f"Best val (ep {best_acc_ep})")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend(fontsize=8.5, framealpha=0, labelcolor=PALETTE["text"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Annotation
    gap = train_accs[-1] - val_accs[-1]
    fig.text(0.97, 0.92,
             f"Final overfit gap: {gap*100:.1f} pp",
             ha="right", va="top", fontsize=8.5,
             color=PALETTE["warn"], style="italic")
    
    fig.suptitle("Model Training Dynamics", fontsize=14,
                 fontweight="bold", color=PALETTE["text"], y=0.97)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Saved → {save_path}")
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot confusion matrix with counts and normalized versions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                             facecolor=PALETTE["bg"])
    fig.subplots_adjust(wspace=0.35, left=0.05, right=0.97,
                        top=0.90, bottom=0.12)
    
    cmap_raw = LinearSegmentedColormap.from_list(
        "blues_custom", ["#EFF6FF", "#1E40AF"], N=256)
    cmap_norm = LinearSegmentedColormap.from_list(
        "greens_custom", ["#F0FDF4", "#065F46"], N=256)
    
    for ax, data, cmap, fmt, title in [
        (axes[0], cm, cmap_raw, "d", "Confusion Matrix (counts)"),
        (axes[1], cm_norm, cmap_norm, ".2f", "Confusion Matrix (normalised)"),
    ]:
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0,
                       vmax=data.max())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = data[i, j]
                color = "white" if val > thresh else PALETTE["text"]
                ax.text(j, i, format(val, fmt),
                        ha="center", va="center",
                        fontsize=7.5, color=color, fontweight="bold")
        
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=8.5)
        ax.set_yticklabels(class_names, fontsize=8.5)
        ax.set_xlabel("Predicted label", fontsize=9,
                      color=PALETTE["subtext"])
        ax.set_ylabel("True label", fontsize=9,
                      color=PALETTE["subtext"])
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color=PALETTE["text"], pad=10)
        ax.spines[:].set_color("#D1D5DB")
        ax.tick_params(colors=PALETTE["subtext"])
    
    fig.suptitle("Confusion Matrix — Test Set", fontsize=14,
                 fontweight="bold", color=PALETTE["text"], y=0.97)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Saved → {save_path}")
    plt.show()


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot per-class precision, recall, and F1 scores.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the figure
    """
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    bar_w = 0.26
    bar_colors = [PALETTE["train"], PALETTE["val"], PALETTE["accent"]]
    
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=PALETTE["bg"])
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.18)
    apply_base_style(ax, title="Per-class Precision / Recall / F1",
                     xlabel="Genre", ylabel="Score")
    
    for i, (metric, color) in enumerate(zip(metrics, bar_colors)):
        vals = [report[c][metric] for c in class_names]
        bars = ax.bar(x + (i - 1) * bar_w, vals,
                      width=bar_w, color=color, alpha=0.88,
                      label=metric.capitalize(), zorder=3,
                      edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, vals):
            if v > 0.06:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color=PALETTE["subtext"])
    
    # Macro avg line
    macro_f1 = report["macro avg"]["f1-score"]
    ax.axhline(macro_f1, color=PALETTE["warn"], linestyle="--",
               linewidth=1.2, zorder=4,
               label=f"Macro F1 avg = {macro_f1:.3f}")
    
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8.5, framealpha=0, labelcolor=PALETTE["text"],
              loc="upper right")
    
    fig.suptitle("Per-class Classification Metrics — Test Set",
                 fontsize=14, fontweight="bold",
                 color=PALETTE["text"], y=0.97)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Saved → {save_path}")
    plt.show()


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot one-vs-rest ROC curves for all classes.
    
    Args:
        y_true: True labels
        y_probs: Probability predictions
        class_names: List of class names
        save_path: Path to save the figure
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 7),
                             facecolor=PALETTE["bg"])
    fig.subplots_adjust(hspace=0.45, wspace=0.32,
                        left=0.05, right=0.97,
                        top=0.90, bottom=0.07)
    
    for i, (ax, name) in enumerate(zip(axes.flat, class_names)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        color = GENRE_COLORS[i % len(GENRE_COLORS)]
        
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"AUC = {roc_auc:.3f}", zorder=3)
        ax.fill_between(fpr, tpr, alpha=0.12, color=color)
        ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=1, zorder=2)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        
        apply_base_style(ax, title=name.capitalize(),
                         xlabel="FPR", ylabel="TPR")
        ax.legend(fontsize=8, framealpha=0, loc="lower right",
                  labelcolor=PALETTE["text"])
        ax.set_aspect("equal")
    
    fig.suptitle("One-vs-Rest ROC Curves — Test Set",
                 fontsize=14, fontweight="bold",
                 color=PALETTE["text"], y=0.97)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Saved → {save_path}")
    plt.show()


def plot_confidence_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot confidence distribution and reliability diagram.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Probability predictions
        save_path: Path to save the figure
    """
    max_conf = y_probs.max(axis=1)
    correct = (y_true == y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             facecolor=PALETTE["bg"])
    fig.subplots_adjust(wspace=0.3, left=0.07, right=0.97,
                        top=0.88, bottom=0.12)
    
    bins = np.linspace(0, 1, 26)
    
    # Histogram
    ax = axes[0]
    apply_base_style(ax,
                     title="Prediction Confidence Distribution",
                     xlabel="Max softmax probability",
                     ylabel="Count")
    ax.hist(max_conf[correct], bins=bins, color=PALETTE["accent"],
            alpha=0.75, label="Correct", zorder=3)
    ax.hist(max_conf[~correct], bins=bins, color=PALETTE["val"],
            alpha=0.75, label="Incorrect", zorder=3)
    ax.legend(fontsize=9, framealpha=0, labelcolor=PALETTE["text"])
    
    # Reliability diagram
    ax = axes[1]
    apply_base_style(ax,
                     title="Reliability Diagram (Calibration)",
                     xlabel="Mean predicted confidence",
                     ylabel="Fraction of correct predictions")
    
    bin_edges = np.linspace(0, 1, 11)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    frac_correct, mean_conf = [], []
    
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (max_conf >= lo) & (max_conf < hi)
        if mask.sum() > 0:
            frac_correct.append(correct[mask].mean())
            mean_conf.append(max_conf[mask].mean())
    
    ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=1.2,
            label="Perfect calibration", zorder=2)
    ax.plot(mean_conf, frac_correct, "o-",
            color=PALETTE["train"], linewidth=2,
            markersize=6, zorder=3, label="Model")
    ax.fill_between(mean_conf, frac_correct, mean_conf,
                    alpha=0.12, color=PALETTE["warn"],
                    label="Calibration gap")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5, framealpha=0, labelcolor=PALETTE["text"])
    
    fig.suptitle("Model Confidence Analysis", fontsize=14,
                 fontweight="bold", color=PALETTE["text"], y=0.97)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Saved → {save_path}")
    plt.show()