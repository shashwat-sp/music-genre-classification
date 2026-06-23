#!/usr/bin/env python
"""Script to run traditional ML models on music genre data."""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_segmented_data
from data.preprocessing import generate_mel_segments
from training.train_traditional import train_traditional_model
from utils.helpers import load_config, set_seed


def load_gtzan_features(csv_path: str):
    """
    Load GTZAN features from CSV.
    
    Args:
        csv_path: Path to features CSV file
    
    Returns:
        Tuple of (features, labels)
    """
    df = pd.read_csv(csv_path)
    
    # Drop non-feature columns
    if 'filename' in df.columns:
        df = df.drop(columns=['filename'])
    
    X = df.drop(columns=['label']).values
    y = df['label'].values
    
    return X, y


def main():
    parser = argparse.ArgumentParser(description='Train traditional ML models for genre classification')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to features CSV file')
    parser.add_argument('--model_type', type=str, default='random_forest',
                       choices=['random_forest', 'svm', 'knn', 'gradient_boosting'],
                       help='Type of model to train')
    parser.add_argument('--config_path', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--no_tune', action='store_true',
                       help='Skip hyperparameter tuning')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    X, y = load_gtzan_features(args.data_path)
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Get model configuration
    model_config = config.get('traditional_models', {}).get(args.model_type, {})
    
    # Train model
    print(f"\nTraining {args.model_type}...")
    trainer, results = train_traditional_model(
        args.model_type,
        X, y,
        model_config,
        tune=not args.no_tune
    )
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    print(f"\nModel saved to {output_dir}")
    
    # Save results
    results_file = output_dir / f"{args.model_type}_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Inference time per sample: {results['inference_time']:.6f}s\n")
    
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()