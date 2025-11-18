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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.landmark_detector import LandmarkDetector


# ============================================================================
# DATASET CLASS
# ============================================================================

class GoogleLandmarksDataset(Dataset):
    """Dataset for Google Landmarks"""
    
    def __init__(self, data_dir: Path, csv_path: Path, transform=None, num_classes=None):
        """
        Args:
            data_dir: Directory with extracted images
            csv_path: Path to train.csv or train_clean.csv
            transform: Image transformations
            num_classes: Limit to top N classes (None = all)
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Read CSV
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Filter by num_classes if specified
        if num_classes:
            # Get top N most common landmarks
            landmark_counts = df['landmark_id'].value_counts()
            top_landmarks = landmark_counts.head(num_classes).index.tolist()
            df = df[df['landmark_id'].isin(top_landmarks)]
            
            # Remap landmark IDs to 0-N
            self.landmark_map = {old_id: new_id for new_id, old_id in enumerate(top_landmarks)}
            df['landmark_id'] = df['landmark_id'].map(self.landmark_map)
            
            print(f"Selected top {num_classes} landmarks")
            print(f"Total images: {len(df)}")
        
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
        num_classes=args.num_classes
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
    
    print("Starting training...")
    print("=" * 80)
    
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
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = project_root / "data" / "checkpoints" / f"landmark_detector_{args.num_classes}classes_best.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'num_classes': train_dataset.num_classes,
                'landmark_map': train_dataset.landmark_map if hasattr(train_dataset, 'landmark_map') else None
            }, checkpoint_path)
            
            print(f"  ✓ Saved best model: {checkpoint_path} (acc: {best_acc:.2f}%)")
    
    # Save final model
    final_path = project_root / "data" / "checkpoints" / f"landmark_detector_{args.num_classes}classes_final.pth"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'num_classes': train_dataset.num_classes,
        'landmark_map': train_dataset.landmark_map if hasattr(train_dataset, 'landmark_map') else None
    }, final_path)
    
    # Save training history
    history_path = project_root / "data" / "checkpoints" / f"training_history_{args.num_classes}classes.json"
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
                       help='Number of landmark classes to train on')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    
    args = parser.parse_args()
    
    train_model(args)


if __name__ == "__main__":
    main()
