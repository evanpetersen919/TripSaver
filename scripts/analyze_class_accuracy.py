"""
Per-Class Accuracy Analysis

Analyzes the trained model's performance on each landmark class.
Shows which landmarks the model recognizes well and which ones need improvement.

Author: Evan Petersen
Date: November 2025
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.landmark_detector import LandmarkDetector
sys.path.insert(0, str(Path(__file__).parent))
from train_landmark_detector import GoogleLandmarksDataset


def analyze_per_class_accuracy(model_path: str, 
                               data_dir: str,
                               csv_path: str,
                               landmark_names_path: str,
                               num_classes: int = 100,
                               samples_per_class: int = None,
                               batch_size: int = 32):
    """
    Analyze accuracy for each landmark class
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Path to image data
        csv_path: Path to train.csv
        landmark_names_path: Path to landmark names JSON
        num_classes: Number of classes
        samples_per_class: Samples per class for validation
        batch_size: Batch size for evaluation
    """
    
    print("=" * 80)
    print("PER-CLASS ACCURACY ANALYSIS")
    print("=" * 80)
    
    # Load landmark names
    with open(landmark_names_path, 'r') as f:
        landmark_data = json.load(f)
        landmark_names = landmark_data['landmark_names']
        landmark_map = landmark_data['landmark_map']
    
    # Load model
    print("\nLoading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = LandmarkDetector(
        model_path=model_path,
        landmark_names_path=landmark_names_path,
        device=device
    )
    model = detector.model
    model.eval()
    
    # Create validation dataset
    print("\nCreating validation dataset...")
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = GoogleLandmarksDataset(
        data_dir=Path(data_dir) / 'train',
        csv_path=Path(csv_path),
        transform=val_transform,
        num_classes=num_classes,
        samples_per_class=samples_per_class
    )
    
    # Use 10% for validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Per-class tracking
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_top5_correct = defaultdict(int)
    
    # Confusion tracking (for most common misclassifications)
    confusion = defaultdict(lambda: defaultdict(int))
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Processing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Top-1 predictions
            _, predicted = outputs.max(1)
            
            # Top-5 predictions
            _, top5_pred = outputs.topk(5, dim=1)
            
            # Update per-class stats
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                
                class_total[label] += 1
                
                if pred == label:
                    class_correct[label] += 1
                else:
                    confusion[label][pred] += 1
                
                # Top-5 accuracy
                if label in top5_pred[i]:
                    class_top5_correct[label] += 1
    
    # Calculate per-class accuracy
    results = []
    for class_idx in sorted(class_total.keys()):
        total = class_total[class_idx]
        correct = class_correct[class_idx]
        top5_correct = class_top5_correct[class_idx]
        
        accuracy = (correct / total * 100) if total > 0 else 0
        top5_accuracy = (top5_correct / total * 100) if total > 0 else 0
        
        # Get landmark name
        landmark_name = landmark_names.get(str(class_idx), f"class_{class_idx}")
        landmark_id = landmark_map.get(str(class_idx), class_idx)
        
        # Get most common misclassification
        if class_idx in confusion and confusion[class_idx]:
            most_confused = max(confusion[class_idx].items(), key=lambda x: x[1])
            confused_with = landmark_names.get(str(most_confused[0]), f"class_{most_confused[0]}")
            confusion_count = most_confused[1]
        else:
            confused_with = "N/A"
            confusion_count = 0
        
        results.append({
            'class_idx': class_idx,
            'landmark_id': landmark_id,
            'landmark_name': landmark_name,
            'total_samples': total,
            'correct': correct,
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'confused_with': confused_with,
            'confusion_count': confusion_count
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Overall statistics
    overall_acc = (sum(class_correct.values()) / sum(class_total.values()) * 100)
    overall_top5 = (sum(class_top5_correct.values()) / sum(class_total.values()) * 100)
    
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"Overall Top-1 Accuracy: {overall_acc:.2f}%")
    print(f"Overall Top-5 Accuracy: {overall_top5:.2f}%")
    print(f"Total validation samples: {sum(class_total.values())}")
    print(f"Number of classes: {len(class_total)}")
    
    # Best performing classes
    print("\n" + "=" * 80)
    print("TOP 10 BEST PERFORMING CLASSES")
    print("=" * 80)
    best = df.nlargest(10, 'accuracy')
    for idx, row in best.iterrows():
        print(f"{row['landmark_name']:<40} {row['accuracy']:>6.2f}% ({row['correct']}/{row['total_samples']})")
    
    # Worst performing classes
    print("\n" + "=" * 80)
    print("TOP 10 WORST PERFORMING CLASSES")
    print("=" * 80)
    worst = df.nsmallest(10, 'accuracy')
    for idx, row in worst.iterrows():
        conf_info = f"→ {row['confused_with']} ({row['confusion_count']}x)" if row['confusion_count'] > 0 else ""
        print(f"{row['landmark_name']:<40} {row['accuracy']:>6.2f}% ({row['correct']}/{row['total_samples']}) {conf_info}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("ACCURACY DISTRIBUTION")
    print("=" * 80)
    print(f"Mean accuracy: {df['accuracy'].mean():.2f}%")
    print(f"Median accuracy: {df['accuracy'].median():.2f}%")
    print(f"Std deviation: {df['accuracy'].std():.2f}%")
    print(f"Min accuracy: {df['accuracy'].min():.2f}%")
    print(f"Max accuracy: {df['accuracy'].max():.2f}%")
    
    # Count classes by accuracy ranges
    print("\nClasses by accuracy range:")
    ranges = [(90, 100), (80, 90), (70, 80), (60, 70), (0, 60)]
    for low, high in ranges:
        count = len(df[(df['accuracy'] >= low) & (df['accuracy'] < high)])
        print(f"  {low}-{high}%: {count} classes")
    
    # Save detailed results
    output_path = Path("data/checkpoints/per_class_accuracy.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze per-class accuracy")
    parser.add_argument('--model-path', type=str, 
                       default='data/checkpoints/landmark_detector_100classes_best.pth',
                       help='Path to trained model')
    parser.add_argument('--data-dir', type=str,
                       default='data/google_landmarks',
                       help='Path to dataset')
    parser.add_argument('--csv-path', type=str,
                       default='data/google_landmarks/metadata/train.csv',
                       help='Path to train.csv')
    parser.add_argument('--landmark-names', type=str,
                       default='data/checkpoints/landmark_names_100classes.json',
                       help='Path to landmark names JSON')
    parser.add_argument('--num-classes', type=int, default=100,
                       help='Number of classes')
    parser.add_argument('--samples-per-class', type=int, default=1275,
                       help='Samples per class')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    
    args = parser.parse_args()
    
    analyze_per_class_accuracy(
        model_path=args.model_path,
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        landmark_names_path=args.landmark_names,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size
    )
