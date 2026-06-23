"""Training script for traditional ML models."""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple


def tune_hyperparameters(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict,
    n_iter: int = 50,
    cv: int = 5
) -> Tuple[Any, Dict]:
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    Args:
        model: Base model instance
        X_train: Training features
        y_train: Training labels
        param_grid: Hyperparameter grid
        n_iter: Number of combinations to try
        cv: Number of cross-validation folds
    
    Returns:
        Tuple of (best_model, best_params)
    """
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='accuracy'
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    print(f"Randomized Search completed in {time.time() - start_time:.2f} seconds")
    
    print("\nBest parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"\nBest cross-validation score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
    class_names: list
) -> Dict[str, Any]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder for inverse transform
        class_names: List of class names
    
    Returns:
        Dictionary with evaluation metrics
    """
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    
    results = {
        'accuracy': accuracy,
        'inference_time': inference_time / len(X_test),
        'predictions': y_pred,
        'labels': y_test
    }
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print(f"Inference time: {inference_time / len(X_test):.4f} seconds per sample")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - Accuracy: {accuracy:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = y_test == i
        if mask.sum() > 0:
            per_class_acc[name] = accuracy_score(y_test[mask], y_pred[mask])
    
    print("\nPer-Class Accuracy:")
    for name, acc in sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:12s}: {acc:.4f}")
    
    return results


def train_traditional_model(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    config: Dict = None,
    tune: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train and evaluate a traditional ML model.
    
    Args:
        model_type: Type of model to train
        X: Feature matrix
        y: Labels
        config: Configuration dictionary
        tune: Whether to perform hyperparameter tuning
    
    Returns:
        Tuple of (trained_model, results)
    """
    from models.traditional_models import TraditionalModelTrainer, get_hyperparameter_grid
    
    trainer = TraditionalModelTrainer(model_type, config)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    
    # Hyperparameter tuning
    if tune:
        param_grid = get_hyperparameter_grid(model_type)
        if param_grid:
            best_model, best_params = tune_hyperparameters(
                trainer.model, X_train, y_train, param_grid
            )
            trainer.model = best_model
        else:
            print(f"No hyperparameter grid defined for {model_type}")
            trainer.train(X_train, y_train)
    else:
        trainer.train(X_train, y_train)
    
    # Evaluate
    results = evaluate_model(
        trainer.model, X_test, y_test,
        trainer.label_encoder,
        trainer.label_encoder.classes_
    )
    
    return trainer, results