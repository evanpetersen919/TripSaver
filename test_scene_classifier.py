"""
Test script to verify Scene Classifier is working correctly

Tests the scene classifier on a single image and shows top-5 predictions.
"""

import sys
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from models.scene_classifier import SceneClassifier


def test_scene_classifier(image_path: str):
    """Test the scene classifier on a single image"""
    
    print("=" * 80)
    print("SCENE CLASSIFIER TEST")
    print("=" * 80)
    
    # Initialize classifier
    print("\nLoading Scene Classifier...")
    classifier = SceneClassifier()
    print(f"✓ Loaded with {classifier.num_classes} scene categories")
    
    # Load image
    print(f"\nLoading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    # Get predictions
    print("\nRunning inference...")
    results = classifier.predict(image, top_k=10)
    
    # Display results
    print("\n" + "=" * 80)
    print("TOP-10 SCENE PREDICTIONS")
    print("=" * 80)
    for i, pred in enumerate(results, 1):
        category = pred['category']
        confidence = pred['confidence']
        print(f"{i:2d}. {category:40s} {confidence:.2%}")
    
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Test scene classifier")
    parser.add_argument('image', type=str, help='Path to image file')
    args = parser.parse_args()
    
    test_scene_classifier(args.image)
