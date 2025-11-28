"""
Build Lambda Layers for AWS Deployment
=======================================

Creates two Lambda layers:
1. Python dependencies (torch, transformers, fastapi, etc.)
2. Trained models and data (EfficientNet weights, landmark database)

Lambda layer structure:
layers/
  python-dependencies/
    python/
      lib/
        python3.11/
          site-packages/
  models/
    opt/
      models/
      data/

Author: Evan Petersen
Date: January 2025
"""

import shutil
import subprocess
import sys
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).parent.parent
LAYERS_DIR = PROJECT_ROOT / "layers"
PYTHON_LAYER_DIR = LAYERS_DIR / "python-dependencies"
MODELS_LAYER_DIR = LAYERS_DIR / "models"


def create_python_dependencies_layer():
    """
    Build Python dependencies layer.
    
    Lambda layers must follow this structure:
    python/lib/python3.11/site-packages/
    """
    print("=" * 80)
    print("BUILDING PYTHON DEPENDENCIES LAYER")
    print("=" * 80)
    
    # Clean and create directories
    if PYTHON_LAYER_DIR.exists():
        print(f"Removing existing layer: {PYTHON_LAYER_DIR}")
        shutil.rmtree(PYTHON_LAYER_DIR)
    
    site_packages = PYTHON_LAYER_DIR / "python" / "lib" / "python3.11" / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory: {site_packages}")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    requirements_file = PROJECT_ROOT / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"ERROR: requirements.txt not found at {requirements_file}")
        return False
    
    # Read requirements and filter out local packages
    with open(requirements_file, 'r') as f:
        requirements = f.read()
    
    # Create temporary requirements file for Lambda (exclude local dev tools)
    lambda_requirements = []
    exclude_packages = ['jupyter', 'notebook', 'ipython', 'streamlit', 'gradio']
    
    for line in requirements.splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            # Check if line contains excluded package
            is_excluded = any(pkg in line.lower() for pkg in exclude_packages)
            if not is_excluded:
                lambda_requirements.append(line)
    
    lambda_req_file = LAYERS_DIR / "lambda_requirements.txt"
    with open(lambda_req_file, 'w') as f:
        f.write('\n'.join(lambda_requirements))
    
    print(f"Created Lambda requirements file: {lambda_req_file}")
    print(f"Installing {len(lambda_requirements)} packages...")
    
    # Install using pip with platform-specific flags for Lambda (Linux x86_64)
    cmd = [
        sys.executable, "-m", "pip", "install",
        "-r", str(lambda_req_file),
        "-t", str(site_packages),
        "--platform", "manylinux2014_x86_64",
        "--only-binary=:all:",
        "--upgrade"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Failed to install dependencies")
        print(result.stderr)
        return False
    
    print("✓ Dependencies installed successfully")
    
    # Clean up unnecessary files to reduce layer size
    print("\nCleaning up unnecessary files...")
    cleanup_patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.dist-info",
        "**/*.egg-info",
        "**/tests",
        "**/test",
        "**/*.so.debug"
    ]
    
    removed_count = 0
    for pattern in cleanup_patterns:
        for item in site_packages.glob(pattern):
            if item.is_file():
                item.unlink()
                removed_count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                removed_count += 1
    
    print(f"✓ Removed {removed_count} unnecessary files/directories")
    
    # Calculate layer size
    layer_size = sum(f.stat().st_size for f in site_packages.rglob('*') if f.is_file())
    layer_size_mb = layer_size / (1024 * 1024)
    
    print(f"\n{'=' * 80}")
    print(f"PYTHON LAYER BUILT SUCCESSFULLY")
    print(f"{'=' * 80}")
    print(f"Location: {PYTHON_LAYER_DIR}")
    print(f"Size: {layer_size_mb:.2f} MB")
    print(f"Lambda Layer Limit: 250 MB (uncompressed)")
    
    if layer_size_mb > 250:
        print(f"WARNING: Layer exceeds 250 MB limit!")
        print(f"Consider splitting into multiple layers or removing heavy dependencies")
        return False
    
    print(f"{'=' * 80}\n")
    return True


def create_models_layer():
    """
    Build models and data layer.
    
    Lambda layers must follow this structure:
    opt/models/ and opt/data/
    """
    print("=" * 80)
    print("BUILDING MODELS LAYER")
    print("=" * 80)
    
    # Clean and create directories
    if MODELS_LAYER_DIR.exists():
        print(f"Removing existing layer: {MODELS_LAYER_DIR}")
        shutil.rmtree(MODELS_LAYER_DIR)
    
    opt_models = MODELS_LAYER_DIR / "opt" / "models"
    opt_data = MODELS_LAYER_DIR / "opt" / "data"
    opt_models.mkdir(parents=True, exist_ok=True)
    opt_data.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directories:")
    print(f"  - {opt_models}")
    print(f"  - {opt_data}")
    
    # Copy trained models
    print("\nCopying trained models...")
    models_src = PROJECT_ROOT / "data" / "checkpoints"
    
    if not models_src.exists():
        print(f"ERROR: Models directory not found: {models_src}")
        return False
    
    files_to_copy = [
        "landmark_detector_500classes_best.pth",
        "landmark_names_100classes.json",
        "best_hyperparameters.json"
    ]
    
    copied_count = 0
    for filename in files_to_copy:
        src_file = models_src / filename
        if src_file.exists():
            dst_file = opt_models / filename
            shutil.copy2(src_file, dst_file)
            file_size_mb = src_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ Copied {filename} ({file_size_mb:.2f} MB)")
            copied_count += 1
        else:
            print(f"  ⚠ Skipped {filename} (not found)")
    
    # Copy landmark database
    print("\nCopying landmark database...")
    data_files = [
        "landmarks_unified.json",
        "landmarks_clip_embeddings.npy",
        "landmarks_id_mapping.json"
    ]
    
    data_src = PROJECT_ROOT / "data"
    for filename in data_files:
        src_file = data_src / filename
        if src_file.exists():
            dst_file = opt_data / filename
            shutil.copy2(src_file, dst_file)
            file_size_mb = src_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ Copied {filename} ({file_size_mb:.2f} MB)")
            copied_count += 1
        else:
            print(f"  ⚠ Skipped {filename} (not found)")
    
    # Calculate layer size
    layer_size = sum(f.stat().st_size for f in MODELS_LAYER_DIR.rglob('*') if f.is_file())
    layer_size_mb = layer_size / (1024 * 1024)
    
    print(f"\n{'=' * 80}")
    print(f"MODELS LAYER BUILT SUCCESSFULLY")
    print(f"{'=' * 80}")
    print(f"Location: {MODELS_LAYER_DIR}")
    print(f"Files copied: {copied_count}")
    print(f"Size: {layer_size_mb:.2f} MB")
    print(f"Lambda Layer Limit: 250 MB (uncompressed)")
    
    if layer_size_mb > 250:
        print(f"WARNING: Layer exceeds 250 MB limit!")
        print(f"Consider using S3 for large model files")
        return False
    
    print(f"{'=' * 80}\n")
    return True


def create_layer_metadata():
    """Create metadata files for layers"""
    print("Creating layer metadata...")
    
    metadata = {
        "python_dependencies": {
            "description": "Python dependencies for CV Location Classifier",
            "runtime": "python3.11",
            "size_mb": 0,
            "created_at": None
        },
        "models": {
            "description": "Trained models and landmark database",
            "runtime": "python3.11",
            "size_mb": 0,
            "created_at": None
        }
    }
    
    # Calculate sizes
    if PYTHON_LAYER_DIR.exists():
        python_size = sum(f.stat().st_size for f in PYTHON_LAYER_DIR.rglob('*') if f.is_file())
        metadata["python_dependencies"]["size_mb"] = round(python_size / (1024 * 1024), 2)
    
    if MODELS_LAYER_DIR.exists():
        models_size = sum(f.stat().st_size for f in MODELS_LAYER_DIR.rglob('*') if f.is_file())
        metadata["models"]["size_mb"] = round(models_size / (1024 * 1024), 2)
    
    # Save metadata
    from datetime import datetime
    metadata["python_dependencies"]["created_at"] = datetime.utcnow().isoformat()
    metadata["models"]["created_at"] = datetime.utcnow().isoformat()
    
    metadata_file = LAYERS_DIR / "layer_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Layer metadata saved to {metadata_file}")


def main():
    """Build all Lambda layers"""
    print("\n" + "=" * 80)
    print("AWS LAMBDA LAYER BUILDER")
    print("=" * 80)
    print("This script builds Lambda layers for deployment to AWS")
    print("Target runtime: Python 3.11 on Linux x86_64")
    print("=" * 80 + "\n")
    
    # Create layers directory
    LAYERS_DIR.mkdir(exist_ok=True)
    
    # Build layers
    success = True
    
    success &= create_python_dependencies_layer()
    success &= create_models_layer()
    
    if success:
        create_layer_metadata()
        
        print("\n" + "=" * 80)
        print("ALL LAYERS BUILT SUCCESSFULLY")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Review layer sizes and ensure they're under 250 MB")
        print("2. Set your Hugging Face API token and JWT secret in samconfig.toml")
        print("3. Deploy with: sam build && sam deploy --guided")
        print("=" * 80 + "\n")
    else:
        print("\n" + "=" * 80)
        print("LAYER BUILD FAILED")
        print("=" * 80)
        print("Please review the errors above and fix them before deploying")
        print("=" * 80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
