"""
Download Places365 Pretrained Weights

Downloads ResNet50 model trained on Places365 dataset for scene classification.

Author: Evan Petersen
Date: November 2025
"""

import urllib.request
import sys
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

PLACES365_URL = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
MODEL_NAME = "resnet50_places365.pth.tar"


# ============================================================================
# DOWNLOAD FUNCTION
# ============================================================================

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_with_progress(url: str, output_path: Path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("PLACES365 PRETRAINED WEIGHTS DOWNLOADER")
    print("=" * 80)
    print(f"Model: ResNet50 trained on Places365")
    print(f"Source: MIT CSAIL")
    print(f"Size: ~100MB")
    print("=" * 80)
    print()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    checkpoints_dir = project_root / "data" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = checkpoints_dir / MODEL_NAME
    
    # Check if already exists
    if output_path.exists():
        print(f"✓ Model already exists at {output_path}")
        response = input("Download again? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    # Download
    print(f"\nDownloading from: {PLACES365_URL}")
    print(f"Saving to: {output_path}")
    print()
    
    try:
        download_with_progress(PLACES365_URL, output_path)
        print(f"\n✓ Download complete!")
        print(f"✓ Saved to: {output_path}")
        print()
        
        # Usage instructions
        print("=" * 80)
        print("USAGE INSTRUCTIONS")
        print("=" * 80)
        print("To use these weights in your pipeline:")
        print()
        print("Option 1 - Python code:")
        print("  from models.scene_classifier import SceneClassifier")
        print(f"  classifier = SceneClassifier(model_path='{output_path}')")
        print()
        print("Option 2 - Vision Pipeline:")
        print("  from core.vision_pipeline import VisionPipeline")
        print("  pipeline = VisionPipeline()")
        print(f"  pipeline.scene_classifier = SceneClassifier(model_path='{output_path}')")
        print()
        print("Option 3 - Demo script:")
        print(f"  python demo_cv.py --image photo.jpg --all")
        print("  (Will automatically find weights in data/checkpoints/)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nManual download instructions:")
        print(f"1. Visit: {PLACES365_URL}")
        print(f"2. Save file to: {output_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
