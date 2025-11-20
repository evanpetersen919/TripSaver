"""
Train Landmark Detector with Optuna Hyperparameter Optimization

This script uses Optuna to automatically find the best hyperparameters
to maximize validation accuracy.

Author: Evan Petersen
Date: November 2025
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import json
import time
import mlflow
import mlflow.pytorch
import random
import numpy as np
import optuna
from optuna.integration import MLflowCallback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.landmark_detector import LandmarkDetector


# Import the dataset class from the original training script
sys.path.insert(0, str(Path(__file__).parent))
from train_landmark_detector import GoogleLandmarksDataset


def create_dataloaders(args, batch_size, augmentation_strength=0.5):
    """Create data loaders with configurable augmentation"""
    
    # Augmentation strength controls the intensity of transformations
    rotation = int(15 * augmentation_strength)
    color_jitter = 0.2 * augmentation_strength
    
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation),
        transforms.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=color_jitter/2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets
    data_dir = Path(args.data_dir)
    csv_path = data_dir / 'metadata' / 'train.csv'
    
    train_dataset = GoogleLandmarksDataset(
        data_dir=data_dir / 'train',
        csv_path=csv_path,
        transform=train_transform,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class
    )
    
    val_dataset = GoogleLandmarksDataset(
        data_dir=data_dir / 'train',
        csv_path=csv_path,
        transform=val_transform,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class
    )
    
    # Use same train/val split
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    _, val_dataset = torch.utils.data.random_split(val_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.dataset.num_classes


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(val_loader), 100. * correct / total


def objective(trial, args):
    """Optuna objective function to maximize validation accuracy"""
    
    # Set random seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Hyperparameters to optimize
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    augmentation_strength = trial.suggest_float("augmentation_strength", 0.3, 1.0)
    
    # Create dataloaders with trial-specific hyperparameters
    train_loader, val_loader, num_classes = create_dataloaders(
        args, batch_size, augmentation_strength
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = LandmarkDetector(num_classes=num_classes, device=device)
    model = detector.model
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training loop (reduced epochs for faster optimization)
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 5
    
    print(f"\nTrial {trial.number}")
    print(f"  LR: {lr:.6f}")
    print(f"  Batch size: {batch_size}")
    print(f"  Weight decay: {weight_decay:.6f}")
    print(f"  Augmentation: {augmentation_strength:.2f}")
    
    for epoch in range(args.optuna_epochs):
        print(f"\nEpoch {epoch+1}/{args.optuna_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Report intermediate value
        trial.report(val_acc, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train Landmark Detector with Optuna")
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data/google_landmarks',
                        help='Directory with Google Landmarks dataset')
    parser.add_argument('--num-classes', type=int, default=100,
                        help='Number of landmark classes to use')
    parser.add_argument('--samples-per-class', type=int, default=None,
                        help='Balance dataset to N samples per class (None = no balancing)')
    
    # Optuna parameters
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of Optuna trials')
    parser.add_argument('--optuna-epochs', type=int, default=10,
                        help='Epochs per trial (reduced for speed)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--study-name', type=str, default='landmark_optimization',
                        help='Optuna study name')
    
    # Final training parameters (after optimization)
    parser.add_argument('--final-epochs', type=int, default=40,
                        help='Epochs for final training with best hyperparameters')
    parser.add_argument('--skip-final-training', action='store_true',
                        help='Skip final training (only run optimization)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LANDMARK DETECTOR - OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Number of classes: {args.num_classes}")
    print(f"Samples per class: {args.samples_per_class or 'All available'}")
    print(f"Optuna trials: {args.n_trials}")
    print(f"Epochs per trial: {args.optuna_epochs}")
    print(f"Final training epochs: {args.final_epochs}")
    print("=" * 80)
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # MLflow callback for Optuna
    mlflc = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="val_accuracy"
    )
    
    # Run optimization
    print("\nStarting hyperparameter optimization...")
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        callbacks=[mlflc]
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.2f}%")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Save best hyperparameters
    best_params_path = Path('data/checkpoints/best_hyperparameters.json')
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest hyperparameters saved to: {best_params_path}")
    
    # Final training with best hyperparameters
    if not args.skip_final_training:
        print("\n" + "=" * 80)
        print("FINAL TRAINING WITH BEST HYPERPARAMETERS")
        print("=" * 80)
        
        # You can now run the regular training script with these parameters
        print("\nTo train with these optimized hyperparameters, run:")
        print(f"python scripts/train_landmark_detector.py \\")
        print(f"  --num-classes {args.num_classes} \\")
        print(f"  --samples-per-class {args.samples_per_class or 'None'} \\")
        print(f"  --epochs {args.final_epochs} \\")
        print(f"  --batch-size {study.best_params['batch_size']} \\")
        print(f"  --lr {study.best_params['lr']:.6f} \\")
        print(f"  --weight-decay {study.best_params['weight_decay']:.6f}")


if __name__ == '__main__':
    main()
