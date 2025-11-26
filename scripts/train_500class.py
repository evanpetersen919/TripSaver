"""
Train 1000-Class Landmark Detector
===================================

Enhanced training script for scaling to 1000 classes with 90%+ accuracy.
Includes MixUp, CutMix, label smoothing, and transfer learning from 100-class model.

Author: Evan Petersen
Date: November 2025
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm
import time
import mlflow
from models.landmark_detector import LandmarkDetector
from models.augmentation import MixUp, CutMix, RandAugment, mixup_criterion
from models.social_media_augmentation import SocialMediaTransform
from scripts.training_monitor import TrainingMonitor, CheckpointManager


# Simple 1000-class dataset
class Landmarks1000Dataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        
        # Load class mapping
        with open(Path(data_dir) / 'class_mapping.json', 'r') as f:
            mapping = json.load(f)
        
        self.num_classes = mapping['num_classes']
        
        # Collect all images and count samples per class
        self.samples = []
        self.class_counts = {}
        
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                class_idx = int(class_dir.name)
                class_images = list(class_dir.glob('*.jpg'))
                self.class_counts[class_idx] = len(class_images)
                
                for img_path in class_images:
                    self.samples.append((str(img_path), class_idx))
        
        print(f"Loaded {len(self.samples)} images from {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self):
        """Get dictionary of class_idx -> sample_count for weighting."""
        return self.class_counts


def get_advanced_transforms(input_size=300, use_social_media_aug=True):
    """
    Get transforms with RandAugment + Social Media Augmentation.
    
    Social media augmentation makes model robust to screenshots with UI elements.
    """
    transform_list = []
    
    # CRITICAL: Add social media UI elements FIRST (30% of images)
    # This trains the model to ignore Instagram/TikTok buttons, captions, etc.
    if use_social_media_aug:
        transform_list.append(SocialMediaTransform(probability=0.3))
    
    # Then apply standard augmentations
    transform_list.extend([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        RandAugment(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5)
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(input_size=300):
    """Get validation transforms."""
    return transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def train_epoch_with_augmentation(
    model, dataloader, criterion, optimizer, device,
    mixup=None, cutmix=None, use_mixup_prob=0.5
):
    """Train one epoch with optional MixUp/CutMix."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Randomly apply MixUp or CutMix
        use_mix = (mixup is not None or cutmix is not None) and torch.rand(1).item() < use_mixup_prob
        
        if use_mix and torch.rand(1).item() < 0.5 and mixup is not None:
            # Apply MixUp
            mixed_images, labels_a, labels_b, lam = mixup(images, labels)
            
            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy (approximate)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (lam * predicted.eq(labels_a).sum().item() + 
                       (1 - lam) * predicted.eq(labels_b).sum().item())
        
        elif use_mix and cutmix is not None:
            # Apply CutMix
            mixed_images, labels_a, labels_b, lam = cutmix(images, labels)
            
            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy (approximate)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (lam * predicted.eq(labels_a).sum().item() + 
                       (1 - lam) * predicted.eq(labels_b).sum().item())
        
        else:
            # Normal training
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct_top1 += predicted.eq(labels).sum().item()
            
            # Top-5 accuracy
            _, top5 = outputs.topk(5, 1, True, True)
            top5 = top5.t()
            correct_top5 += top5.eq(labels.view(1, -1).expand_as(top5)).sum().item()
    
    return (total_loss / len(dataloader), 
            100. * correct_top1 / total,
            100. * correct_top5 / total)


def main():
    """Train 1000-class model."""
    
    print("=" * 80)
    print("TRAINING 500-CLASS LANDMARK DETECTOR")
    print("=" * 80)
    
    # Configuration
    DATA_DIR = "data/landmarks_500class"
    CHECKPOINT_100 = "data/checkpoints/landmark_detector_100classes_best.pth"
    NUM_CLASSES = 500
    BATCH_SIZE = 32
    EPOCHS = 60
    LR = 0.0001  # Lower LR for transfer learning
    LABEL_SMOOTHING = 0.1
    USE_MIXUP = True
    USE_CUTMIX = True
    MIXUP_ALPHA = 1.0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Data: {DATA_DIR}")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LR}")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")
    print(f"  MixUp: {USE_MIXUP}")
    print(f"  CutMix: {USE_CUTMIX}")
    print(f"  Transfer learning from: {CHECKPOINT_100}")
    print(f"  Device: {device}")
    
    # Check data
    if not Path(DATA_DIR).exists():
        print(f"\nERROR: Dataset not found at {DATA_DIR}")
        print("Run: python scripts/prepare_1000class_dataset.py (configured for 500 classes)")
        return
    
    # Create datasets
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    train_dataset = Landmarks1000Dataset(
        DATA_DIR,
        split='train',
        transform=get_advanced_transforms(use_social_media_aug=True)  # CRITICAL for real-world performance
    )
    
    val_dataset = Landmarks1000Dataset(
        DATA_DIR,
        split='val',
        transform=get_val_transforms()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    
    detector = LandmarkDetector(num_classes=NUM_CLASSES, device=device)
    model = detector.model
    
    # Load 100-class checkpoint for transfer learning
    if Path(CHECKPOINT_100).exists():
        print(f"Loading checkpoint: {CHECKPOINT_100}")
        checkpoint = torch.load(CHECKPOINT_100, map_location=device)
        
        # Load backbone weights (skip classifier)
        state_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        # Filter out classifier weights
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and 'classifier' not in k}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"✓ Loaded backbone weights from 100-class model")
        print(f"  (Classifier randomly initialized for 1000 classes)")
    else:
        print(f"WARNING: Checkpoint not found, training from ImageNet weights")
    
    # Setup training with class weighting for imbalanced dataset
    print("\nCalculating class weights for balanced training...")
    class_counts = train_dataset.get_class_counts()
    
    # Calculate inverse frequency weights (underrepresented classes get higher weight)
    total_samples = sum(class_counts.values())
    class_weights = torch.zeros(NUM_CLASSES, device=device)
    
    for class_idx, count in class_counts.items():
        # Weight = total / (num_classes * count)
        class_weights[class_idx] = total_samples / (NUM_CLASSES * count)
    
    # Normalize weights to mean=1.0
    class_weights = class_weights / class_weights.mean()
    
    print(f"  Class weight range: {class_weights.min():.2f} - {class_weights.max():.2f}")
    print(f"  (Underrepresented classes get higher weight to balance learning)")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Setup augmentation
    mixup = MixUp(alpha=MIXUP_ALPHA) if USE_MIXUP else None
    cutmix = CutMix(alpha=MIXUP_ALPHA) if USE_CUTMIX else None
    
    # Setup monitoring and checkpoint management
    monitor = TrainingMonitor(
        log_file="training_monitor_500classes.log",
        alert_on_nan=True,
        alert_on_plateau=True,
        alert_on_overfitting=True
    )
    monitor.start()
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir="data/checkpoints",
        keep_best=3,
        keep_every_n=10
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    best_acc = 0.0
    target_acc = 90.0
    
    with mlflow.start_run(run_name="landmark_500classes"):
        mlflow.log_params({
            "num_classes": NUM_CLASSES,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "label_smoothing": LABEL_SMOOTHING,
            "mixup": USE_MIXUP,
            "cutmix": USE_CUTMIX,
            "transfer_learning": True
        })
        
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print("-" * 80)
            
            start_time = time.time()
            
            # Train
            train_loss, train_acc = train_epoch_with_augmentation(
                model, train_loader, criterion, optimizer, device,
                mixup=mixup, cutmix=cutmix
            )
            
            # Validate
            val_loss, val_acc, val_acc_top5 = validate(model, val_loader, criterion, device)
            
            # Step scheduler
            scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Log to monitor
            monitor.log_epoch(epoch + 1, {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_acc_top5': val_acc_top5,
                'lr': optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
                'total_epochs': EPOCHS
            })
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_acc_top5": val_acc_top5,
                "lr": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Print summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Val Top-5: {val_acc_top5:.2f}%")
            print(f"  Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint (best and periodic)
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                print(f"  ✓ New best accuracy: {best_acc:.2f}%")
            
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                val_acc=val_acc,
                val_acc_top5=val_acc_top5,
                is_best=is_best
            )
            
            # Check for early stopping
            if monitor.should_stop_early(patience=15):
                print("\n⚠️  Early stopping triggered")
                break
                
            # Check if target reached
            if best_acc >= target_acc:
                print(f"\n🎉 TARGET ACHIEVED: {best_acc:.2f}% >= {target_acc}%")
    
    # Save training history
    monitor.save_history("training_history_1000classes.json")
    
    # Final summary
    best_epoch, best_val = monitor.get_best_epoch()
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Best epoch: {best_epoch}")
    print(f"Target: {target_acc}%")
    print(f"Status: {'✓ ACHIEVED' if best_acc >= target_acc else '✗ NOT YET'}")
    
    if monitor.alerts:
        print(f"\n⚠️  Total alerts: {len(monitor.alerts)}")
        print("  Check training_monitor_1000classes.log for details")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
