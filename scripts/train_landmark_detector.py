"""
Train Landmark Detector on Google Landmarks Dataset

Supports incremental learning - start with small subset, expand later.

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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.landmark_detector import LandmarkDetector


# ============================================================================
# DATASET CLASS
# ============================================================================

class GoogleLandmarksDataset(Dataset):
    """Dataset for Google Landmarks with class balancing"""
    
    def __init__(self, data_dir: Path, csv_path: Path, transform=None, num_classes=None, landmark_ids=None, samples_per_class=None):
        """
        Args:
            data_dir: Directory with extracted images
            csv_path: Path to train.csv or train_clean.csv
            transform: Image transformations
            num_classes: Limit to top N classes (None = all)
            landmark_ids: List of specific landmark IDs to use (overrides num_classes)
            samples_per_class: Balance dataset to N samples per class (None = no balancing)
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Read CSV
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Filter by landmark_ids or num_classes
        if landmark_ids:
            # Use specific curated landmark IDs
            df = df[df['landmark_id'].isin(landmark_ids)]
            
            # Remap landmark IDs to 0-N
            self.landmark_map = {old_id: new_id for new_id, old_id in enumerate(landmark_ids)}
            df['landmark_id'] = df['landmark_id'].map(self.landmark_map)
            
            print(f"Selected {len(landmark_ids)} curated landmarks")
            print(f"Total images before balancing: {len(df)}")
        elif num_classes:
            # Get top N most common landmarks
            landmark_counts = df['landmark_id'].value_counts()
            top_landmarks = landmark_counts.head(num_classes).index.tolist()
            df = df[df['landmark_id'].isin(top_landmarks)]
            
            # Remap landmark IDs to 0-N
            self.landmark_map = {old_id: new_id for new_id, old_id in enumerate(top_landmarks)}
            df['landmark_id'] = df['landmark_id'].map(self.landmark_map)
            
            print(f"Selected top {num_classes} most common landmarks")
            print(f"Total images before balancing: {len(df)}")
        
        # Balance classes if requested
        if samples_per_class:
            print(f"\nBalancing dataset to {samples_per_class} samples per class...")
            balanced_dfs = []
            
            for landmark_id in df['landmark_id'].unique():
                landmark_df = df[df['landmark_id'] == landmark_id]
                n_samples = len(landmark_df)
                
                if n_samples >= samples_per_class:
                    # Downsample: randomly select N samples
                    balanced_df = landmark_df.sample(n=samples_per_class, random_state=42)
                else:
                    # Oversample: repeat samples to reach target
                    n_repeats = samples_per_class // n_samples
                    n_remainder = samples_per_class % n_samples
                    
                    # Repeat full dataset n_repeats times
                    repeated_df = pd.concat([landmark_df] * n_repeats, ignore_index=True)
                    
                    # Add random remainder samples
                    if n_remainder > 0:
                        remainder_df = landmark_df.sample(n=n_remainder, random_state=42, replace=True)
                        balanced_df = pd.concat([repeated_df, remainder_df], ignore_index=True)
                    else:
                        balanced_df = repeated_df
                
                balanced_dfs.append(balanced_df)
            
            df = pd.concat(balanced_dfs, ignore_index=True)
            print(f"Balanced dataset: {len(df)} total images ({samples_per_class} per class)")
        
        self.df = df.reset_index(drop=True)
        self.num_classes = df['landmark_id'].nunique()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Build image path: {a}/{b}/{c}/{id}.jpg
        img_id = row['id']
        img_path = self.data_dir / img_id[0] / img_id[1] / img_id[2] / f"{img_id}.jpg"
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return black image if load fails
            print(f"Warning: Failed to load {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        label = int(row['landmark_id'])
        
        return image, label


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def get_transforms(augment=True):
    """Get image transforms"""
    if augment:
        return transforms.Compose([
            transforms.Resize(320),
            transforms.RandomCrop(300),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{running_loss / (pbar.n + 1):.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f"{running_loss / (pbar.n + 1):.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model(args):
    """Main training function"""
    
    # Initialize MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("landmark-detector-training")
    
    # Load curated landmarks if specified
    landmark_ids = None
    landmark_names = {}
    if args.use_curated:
        project_root = Path(__file__).parent.parent
        curated_path = project_root / "data" / "curated_landmarks.json"
        if not curated_path.exists():
            print(f"ERROR: Curated landmarks not found at {curated_path}")
            print("Run: python scripts/find_famous_landmarks.py")
            return
        
        with open(curated_path, 'r', encoding='utf-8') as f:
            curated_data = json.load(f)
        
        landmark_ids = [lm['landmark_id'] for lm in curated_data['landmarks']]
        landmark_names = {lm['landmark_id']: lm['name'] for lm in curated_data['landmarks']}
        
        print("=" * 80)
        print("TRAINING ON CURATED FAMOUS LANDMARKS")
        print("=" * 80)
        print(f"Total curated landmarks: {len(landmark_ids)}")
        print(f"Examples: {', '.join(list(landmark_names.values())[:5])}")
    else:
        print("=" * 80)
        print("LANDMARK DETECTOR TRAINING")
        print("=" * 80)
        print(f"Number of classes: {args.num_classes}")
    
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir / "train"
    csv_path = project_root / args.data_dir / "metadata" / "train.csv"
    
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        print("Run: python scripts/download_google_landmarks.py --metadata-only")
        return
    
    if not data_dir.exists():
        print(f"ERROR: Training images not found at {data_dir}")
        print("Run: python scripts/download_google_landmarks.py --num-tars 10")
        return
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = GoogleLandmarksDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        transform=get_transforms(augment=True),
        num_classes=None if args.use_curated else args.num_classes,
        landmark_ids=landmark_ids,
        samples_per_class=args.samples_per_class
    )
    
    # Split into train/val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print()
    
    # Initialize model
    print("Initializing model...")
    detector = LandmarkDetector(num_classes=train_dataset.num_classes, device=args.device)
    model = detector.model
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    best_acc = 0.0
    training_history = []
    patience_counter = 0
    
    print("Starting training...")
    print("=" * 80)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("num_classes", train_dataset.num_classes)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("samples_per_class", args.samples_per_class)
        mlflow.log_param("total_samples", len(train_dataset))
        mlflow.log_param("train_samples", len(train_subset))
        mlflow.log_param("val_samples", len(val_subset))
        mlflow.log_param("use_curated", args.use_curated)
        mlflow.log_param("early_stopping_patience", args.early_stopping)
    
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print("-" * 80)
            
            start_time = time.time()
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, args.device)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)
            
            # Print epoch results
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr'],
                'time': epoch_time
            })
            
            # Save best model and check early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0  # Reset patience
                model_suffix = "curated" if args.use_curated else f"{args.num_classes}classes"
                checkpoint_path = project_root / "data" / "checkpoints" / f"landmark_detector_{model_suffix}_best.pth"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'num_classes': train_dataset.num_classes,
                    'landmark_map': train_dataset.landmark_map if hasattr(train_dataset, 'landmark_map') else None,
                    'landmark_names': landmark_names if args.use_curated else None
                }, checkpoint_path)
                
                print(f"  ✓ Saved best model: {checkpoint_path} (acc: {best_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{args.early_stopping}")
                
                # Early stopping
                if patience_counter >= args.early_stopping:
                    print(f"\n{'='*80}")
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    print(f"Best validation accuracy: {best_acc:.2f}%")
                    print(f"{'='*80}")
                    break
        
        # Log final metrics
        mlflow.log_metric("best_val_acc", best_acc)
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
    
    # Save final model
    model_suffix = "curated" if args.use_curated else f"{args.num_classes}classes"
    final_path = project_root / "data" / "checkpoints" / f"landmark_detector_{model_suffix}_final.pth"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'num_classes': train_dataset.num_classes,
        'landmark_map': train_dataset.landmark_map if hasattr(train_dataset, 'landmark_map') else None,
        'landmark_names': landmark_names if args.use_curated else None
    }, final_path)
    
    # Save training history
    history_path = project_root / "data" / "checkpoints" / f"training_history_{model_suffix}.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Best model saved to: {checkpoint_path}")
    print(f"Final model saved to: {final_path}")
    print(f"Training history: {history_path}")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Landmark Detector on Google Landmarks")
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/google_landmarks',
                       help='Path to Google Landmarks dataset')
    parser.add_argument('--num-classes', type=int, default=50,
                       help='Number of landmark classes to train on (ignored if --use-curated)')
    parser.add_argument('--use-curated', action='store_true',
                       help='Use curated list of famous landmarks from data/curated_landmarks.json')
    parser.add_argument('--samples-per-class', type=int, default=None,
                       help='Balance dataset to N samples per class (default: None = no balancing)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--early-stopping', type=int, default=7,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    
    args = parser.parse_args()
    
    train_model(args)


if __name__ == "__main__":
    main()
