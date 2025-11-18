"""
Google Landmarks Dataset v2 Download Script

Downloads and prepares the Google Landmarks v2 dataset for training.
Dataset: 5M images, 200K landmark classes

Author: Evan Petersen
Date: November 2025
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset URLs
TRAIN_CSV = "https://s3.amazonaws.com/google-landmark/metadata/train.csv"
TRAIN_CLEAN_CSV = "https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv"
TRAIN_LABEL_TO_CATEGORY = "https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv"

# Base URL for training images
TRAIN_BASE_URL = "https://s3.amazonaws.com/google-landmark/train/"

# Dataset size
NUM_TRAIN_TARS = 500  # 500 TAR files, each ~1GB


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_file(url: str, output_path: Path):
    """Download a file using curl or wget"""
    print(f"Downloading {url}...")
    
    try:
        # Try curl first (usually faster)
        subprocess.run(
            ["curl", "-o", str(output_path), url],
            check=True,
            capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Fall back to wget
            subprocess.run(
                ["wget", "-O", str(output_path), url],
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ERROR: Neither curl nor wget found. Please install one of them.")
            sys.exit(1)
    
    print(f"✓ Downloaded to {output_path}")


def download_metadata(data_dir: Path):
    """Download dataset metadata (CSVs)"""
    print("=" * 80)
    print("DOWNLOADING METADATA")
    print("=" * 80)
    
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Download CSV files
    download_file(TRAIN_CSV, metadata_dir / "train.csv")
    download_file(TRAIN_CLEAN_CSV, metadata_dir / "train_clean.csv")
    download_file(TRAIN_LABEL_TO_CATEGORY, metadata_dir / "train_label_to_category.csv")
    
    print(f"\n✓ Metadata downloaded to {metadata_dir}\n")


def download_images(data_dir: Path, start_idx: int = 0, end_idx: int = 499, num_parallel: int = 4, auto_delete: bool = False):
    """Download training images (TAR files)
    
    Args:
        data_dir: Base data directory
        start_idx: Start TAR file index (0-499)
        end_idx: End TAR file index (0-499)
        num_parallel: Number of parallel downloads
        auto_delete: Automatically delete TARs after extraction
    """
    print("=" * 80)
    print(f"DOWNLOADING IMAGES (TAR files {start_idx} to {end_idx})")
    print("=" * 80)
    print(f"WARNING: Each TAR is ~1GB. Total download: ~{(end_idx - start_idx + 1)} GB")
    print("This may take several hours depending on your internet connection.")
    print("=" * 80)
    
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Download TAR files
    for i in range(start_idx, end_idx + 1):
        tar_name = f"images_{i:03d}.tar"
        tar_url = TRAIN_BASE_URL + tar_name
        tar_path = train_dir / tar_name
        
        if tar_path.exists():
            print(f"✓ {tar_name} already exists, skipping...")
            continue
        
        print(f"\nDownloading {tar_name} ({i - start_idx + 1}/{end_idx - start_idx + 1})...")
        download_file(tar_url, tar_path)
        
        # Extract TAR
        print(f"Extracting {tar_name}...")
        subprocess.run(
            ["tar", "-xf", str(tar_path), "-C", str(train_dir)],
            check=True
        )
        
        # Optionally delete TAR to save space
        if auto_delete:
            tar_path.unlink()
            print(f"✓ Deleted {tar_name} (auto-delete enabled)")
        else:
            if input(f"Delete {tar_name} to save space? (y/n): ").lower() == 'y':
                tar_path.unlink()
                print(f"✓ Deleted {tar_name}")
    
    print(f"\n✓ Images downloaded and extracted to {train_dir}\n")


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_dataset(data_dir: Path, num_classes: int = 1000, min_images_per_class: int = 100):
    """Prepare training dataset by filtering top landmarks
    
    Args:
        data_dir: Base data directory
        num_classes: Number of landmark classes to use
        min_images_per_class: Minimum images required per class
    """
    print("=" * 80)
    print("PREPARING DATASET")
    print("=" * 80)
    
    import pandas as pd
    from collections import Counter
    
    # Read metadata
    train_clean_path = data_dir / "metadata" / "train_clean.csv"
    if not train_clean_path.exists():
        print("ERROR: train_clean.csv not found. Run download_metadata() first.")
        return
    
    print(f"Reading {train_clean_path}...")
    df = pd.read_csv(train_clean_path)
    
    # Count images per landmark
    landmark_counts = df['images'].str.split().apply(len)
    df['num_images'] = landmark_counts
    
    # Filter by minimum images
    df_filtered = df[df['num_images'] >= min_images_per_class]
    
    # Take top N classes
    df_top = df_filtered.nlargest(num_classes, 'num_images')
    
    print(f"\n✓ Selected {len(df_top)} landmarks with {min_images_per_class}+ images each")
    print(f"Total images: {df_top['num_images'].sum()}")
    print(f"Average images per landmark: {df_top['num_images'].mean():.1f}")
    
    # Save filtered dataset
    output_path = data_dir / "metadata" / f"train_top{num_classes}.csv"
    df_top.to_csv(output_path, index=False)
    print(f"\n✓ Saved filtered dataset to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download Google Landmarks v2 dataset")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/google_landmarks',
        help='Directory to save dataset'
    )
    parser.add_argument(
        '--metadata-only',
        action='store_true',
        help='Download only metadata (CSVs)'
    )
    parser.add_argument(
        '--num-tars',
        type=int,
        default=10,
        help='Number of TAR files to download (each ~1GB)'
    )
    parser.add_argument(
        '--auto-delete',
        action='store_true',
        help='Automatically delete TAR files after extraction to save space'
    )
    parser.add_argument(
        '--prepare',
        action='store_true',
        help='Prepare dataset by filtering top landmarks'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=1000,
        help='Number of landmark classes to prepare'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GOOGLE LANDMARKS V2 DATASET DOWNLOADER")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Dataset info: https://github.com/cvdfoundation/google-landmark")
    print("=" * 80)
    print()
    
    # Download metadata
    download_metadata(data_dir)
    
    if not args.metadata_only:
        # Download images
        download_images(data_dir, start_idx=0, end_idx=args.num_tars - 1, auto_delete=args.auto_delete)
    
    if args.prepare:
        # Prepare filtered dataset
        prepare_dataset(data_dir, num_classes=args.num_classes)
    
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the dataset structure")
    print("2. Create a training script (scripts/train_landmark_detector.py)")
    print("3. Train the model on your target landmarks")
    print("4. Load trained weights in LandmarkDetector(model_path='path/to/weights.pth')")
    print("=" * 80)


if __name__ == "__main__":
    main()
