#!/usr/bin/env python
"""Script to train CNN model for music genre classification."""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_data_loaders
from data.preprocessing import generate_mel_spectrogram
from models.cnn_model import CNNModel
from training.train import train_model
from evaluation.evaluate import evaluate_model
from utils.helpers import load_config, set_seed, get_device
from data.load_data import load_gtzan_data


def main():
    parser = argparse.ArgumentParser(description='Train CNN for genre classification')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to GTZAN genres folder')
    parser.add_argument('--config_path', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config_path)
    cnn_config = config.get('cnn', {})
    data_config = config.get('data', {})
    
    # Override config with command line arguments
    epochs = args.epochs or cnn_config.get('training', {}).get('epochs', 100)
    batch_size = args.batch_size or cnn_config.get('training', {}).get('batch_size', 64)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_gtzan_data(
        args.data_path,
        data_config
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=batch_size,
        augment=True
    )
    
    # Create model
    input_channels = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    model = CNNModel(input_channels, num_classes).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nTraining model...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=epochs,
        patience=cnn_config.get('training', {}).get('patience', 15),
        save_path=f"{args.output_dir}/best_cnn_model.pth"
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(
        model,
        test_loader,
        device,
        label_encoder,
        plot_confusion=True
    )
    
    # Save final model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / 'final_cnn_model.pth')
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()