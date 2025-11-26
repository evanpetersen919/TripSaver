"""
Prepare 1000-Class Dataset
===========================

Prepares training dataset with top 1000 most popular landmarks
from Google Landmarks v2 for scaling up from 100 to 1000 classes.

Author: Evan Petersen
Date: November 2025
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import shutil
from collections import Counter, defaultdict
from tqdm import tqdm
import random


def analyze_landmark_distribution(train_dir: str):
    """
    Analyze distribution of landmarks using train.csv metadata.
    
    Args:
        train_dir: Path to Google Landmarks training directory
        
    Returns:
        Dictionary with landmark_id -> image_count
    """
    print("Analyzing landmark distribution from CSV metadata...")
    print("This will process 4.1M entries - estimated time: 2-3 minutes...")
    
    # Load CSV with image_id -> landmark_id mapping
    csv_path = Path(train_dir).parent / "metadata" / "train.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"train.csv not found at {csv_path}")
    
    import pandas as pd
    
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"✓ Loaded {len(df):,} image entries")
    
    # Count images per landmark
    landmark_counts = df['landmark_id'].value_counts().to_dict()
    
    print(f"✓ Found {len(landmark_counts):,} unique landmarks")
    print(f"✓ Total images: {sum(landmark_counts.values()):,}")
    
    return landmark_counts


def select_top_landmarks(landmark_counts: dict, num_classes: int = 1000, min_images: int = 500):
    """
    Select top N landmarks with sufficient training data.
    
    Args:
        landmark_counts: Dictionary of landmark_id -> count
        num_classes: Number of classes to select
        min_images: Minimum images per class
        
    Returns:
        List of selected landmark IDs
    """
    print(f"\nSelecting top {num_classes} landmarks with min {min_images} images...")
    
    # Filter landmarks with sufficient data
    valid_landmarks = {lid: count for lid, count in landmark_counts.items() 
                      if count >= min_images}
    
    print(f"✓ {len(valid_landmarks)} landmarks have {min_images}+ images")
    
    # Select top N by count
    top_landmarks = sorted(valid_landmarks.items(), key=lambda x: x[1], reverse=True)[:num_classes]
    selected_ids = [lid for lid, count in top_landmarks]
    
    print(f"✓ Selected {len(selected_ids)} landmarks")
    print(f"  Image count range: {top_landmarks[-1][1]} to {top_landmarks[0][1]}")
    
    return selected_ids


def create_train_val_splits(
    train_dir: str,
    selected_landmarks: list,
    output_dir: str,
    val_split: float = 0.15,
    images_per_class: int = 1000,
    balance_classes: bool = True
):
    """
    Create train/val splits from selected landmarks.
    
    Args:
        train_dir: Source directory with all landmarks
        selected_landmarks: List of landmark IDs to use
        output_dir: Output directory for organized dataset
        val_split: Fraction of data for validation
        images_per_class: Target images per class for balanced dataset
        balance_classes: If True, ensure all classes have equal images
    """
    print(f"\nCreating train/val splits (val={val_split*100:.0f}%)...")
    
    train_path = Path(train_dir)
    output_path = Path(output_dir)
    
    train_out = output_path / "train"
    val_out = output_path / "val"
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)
    
    # Create class mapping
    class_to_idx = {lid: idx for idx, lid in enumerate(selected_landmarks)}
    
    train_manifest = []
    val_manifest = []
    
    total_train = 0
    total_val = 0
    
    # Load CSV to get image_id -> landmark_id mapping
    import pandas as pd
    csv_path = train_path.parent / "metadata" / "train.csv"
    print(f"Loading image mappings from CSV...")
    df = pd.read_csv(csv_path)
    
    # Process each landmark
    for landmark_id in tqdm(selected_landmarks, desc="Processing landmarks"):
        # Get all image IDs for this landmark from CSV
        landmark_images = df[df['landmark_id'] == landmark_id]['id'].tolist()
        
        if not landmark_images:
            print(f"  Warning: No images found for landmark {landmark_id}")
            continue
        
        # Find actual image files (structure: train/[0]/[0]/[0]/[image_id].jpg)
        image_files = []
        for img_id in landmark_images:
            if len(img_id) >= 3:
                char1 = img_id[0].lower()
                char2 = img_id[1].lower()
                char3 = img_id[2].lower()
                
                img_path = train_path / char1 / char2 / char3 / f"{img_id}.jpg"
                if img_path.exists():
                    image_files.append(img_path)
        
        # Balance classes - ensure equal samples per class
        if balance_classes:
            if len(image_files) < images_per_class:
                print(f"  Warning: {landmark_id} has only {len(image_files)} images (< {images_per_class})")
            else:
                # Random sample to get exactly images_per_class
                image_files = random.sample(image_files, images_per_class)
        else:
            # Just cap at max if not balancing
            if len(image_files) > images_per_class:
                image_files = random.sample(image_files, images_per_class)
        
        # Shuffle and split
        random.shuffle(image_files)
        split_idx = int(len(image_files) * (1 - val_split))
        
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        class_idx = class_to_idx[landmark_id]
        
        # Create class directories
        train_class_dir = train_out / str(class_idx)
        val_class_dir = val_out / str(class_idx)
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        
        # Copy training images
        for img_path in train_images:
            dest = train_class_dir / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)
            train_manifest.append({
                'path': str(dest.relative_to(output_path)),
                'landmark_id': landmark_id,
                'class_idx': class_idx
            })
            total_train += 1
        
        # Copy validation images
        for img_path in val_images:
            dest = val_class_dir / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)
            val_manifest.append({
                'path': str(dest.relative_to(output_path)),
                'landmark_id': landmark_id,
                'class_idx': class_idx
            })
            total_val += 1
    
    # Save manifests
    train_manifest_path = output_path / "train_manifest.json"
    val_manifest_path = output_path / "val_manifest.json"
    
    with open(train_manifest_path, 'w') as f:
        json.dump(train_manifest, f, indent=2)
    
    with open(val_manifest_path, 'w') as f:
        json.dump(val_manifest, f, indent=2)
    
    # Save class mapping
    class_map_path = output_path / "class_mapping.json"
    with open(class_map_path, 'w') as f:
        json.dump({
            'num_classes': len(selected_landmarks),
            'landmark_to_idx': class_to_idx,
            'idx_to_landmark': {str(idx): lid for lid, idx in class_to_idx.items()}
        }, f, indent=2)
    
    print(f"\n✓ Dataset created successfully!")
    print(f"  Training images: {total_train:,}")
    print(f"  Validation images: {total_val:,}")
    print(f"  Classes: {len(selected_landmarks)}")
    print(f"  Avg train images/class: {total_train / len(selected_landmarks):.1f}")
    print(f"  Avg val images/class: {total_val / len(selected_landmarks):.1f}")
    print(f"  Output directory: {output_path}")
    print(f"\nManifests saved:")
    print(f"  - {train_manifest_path}")
    print(f"  - {val_manifest_path}")
    print(f"  - {class_map_path}")
    
    if balance_classes:
        print(f"\n✓ Classes balanced: Each class has exactly {images_per_class} images")
        print(f"  (before train/val split)")


def main():
    """Prepare 1000-class dataset."""
    
    print("=" * 80)
    print("PREPARE 1000-CLASS LANDMARK DATASET")
    print("=" * 80)
    
    # Configuration
    TRAIN_DIR = "data/google_landmarks/train"
    OUTPUT_DIR = "data/landmarks_500class"
    NUM_CLASSES = 500
    MIN_IMAGES = 100  # Minimum images per class (Google Landmarks has long-tail distribution)
    VAL_SPLIT = 0.15
    IMAGES_PER_CLASS = 500  # Target images per class for balanced dataset
    BALANCE_CLASSES = True  # Ensure equal samples per class
    
    # Check source directory
    if not Path(TRAIN_DIR).exists():
        print(f"ERROR: Source directory not found: {TRAIN_DIR}")
        print("Please ensure Google Landmarks data is extracted.")
        return
    
    print(f"\nConfiguration:")
    print(f"  Source: {TRAIN_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Target classes: {NUM_CLASSES}")
    print(f"  Min images/class: {MIN_IMAGES}")
    print(f"  Images per class: {IMAGES_PER_CLASS}")
    print(f"  Balance classes: {BALANCE_CLASSES}")
    print(f"  Validation split: {VAL_SPLIT*100:.0f}%")
    print(f"\n  Strategy: Select top {NUM_CLASSES} landmarks BY POPULARITY")
    print(f"           (popularity = image count, more images = more popular)")
    
    # Analyze distribution
    print("\n" + "=" * 80)
    print("STEP 1: ANALYZE LANDMARK DISTRIBUTION")
    print("=" * 80)
    
    landmark_counts = analyze_landmark_distribution(TRAIN_DIR)
    
    # Select top landmarks
    print("\n" + "=" * 80)
    print("STEP 2: SELECT TOP LANDMARKS")
    print("=" * 80)
    
    selected_landmarks = select_top_landmarks(
        landmark_counts,
        num_classes=NUM_CLASSES,
        min_images=MIN_IMAGES
    )
    
    if len(selected_landmarks) < NUM_CLASSES:
        print(f"\nWARNING: Only found {len(selected_landmarks)} landmarks with {MIN_IMAGES}+ images")
        print(f"Consider reducing MIN_IMAGES or NUM_CLASSES")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Create splits
    print("\n" + "=" * 80)
    print("STEP 3: CREATE TRAIN/VAL SPLITS")
    print("=" * 80)
    
    create_train_val_splits(
        TRAIN_DIR,
        selected_landmarks,
        OUTPUT_DIR,
        val_split=VAL_SPLIT,
        images_per_class=IMAGES_PER_CLASS,
        balance_classes=BALANCE_CLASSES
    )
    
    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\n✨ Next step: Train 1000-class model")
    print(f"   python scripts/train_landmark_detector.py \\")
    print(f"       --data_dir {OUTPUT_DIR} \\")
    print(f"       --num_classes {NUM_CLASSES} \\")
    print(f"       --checkpoint data/checkpoints/landmark_detector_100classes_best.pth \\")
    print(f"       --epochs 60 \\")
    print(f"       --batch_size 32 \\")
    print(f"       --augmentation advanced \\")
    print(f"       --use_mixup \\")
    print(f"       --use_cutmix \\")
    print(f"       --label_smoothing 0.1")
    print("=" * 80)


if __name__ == "__main__":
    main()
