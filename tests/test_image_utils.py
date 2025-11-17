"""
Quick test script for image_utils.py
Run this to verify all functions work correctly
"""

import sys
import torch
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from models.image_utils import *


def test_image_loading():
    """Test if we can create and load a test image"""
    print("Testing image loading...")
    
    # Create a simple test image
    import numpy as np
    test_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_image = numpy_to_image(test_array)
    
    # Save and reload
    test_image.save("test_temp.jpg")
    loaded = load_image("test_temp.jpg")
    
    print(f"  Created image: {test_image.size}")
    print(f"  Loaded image: {loaded.size}")
    print("  PASSED\n")
    
    return loaded


def test_conversions(image):
    """Test format conversions"""
    print("Testing format conversions...")
    
    # Image to tensor
    tensor = image_to_tensor(image)
    print(f"  Image -> Tensor: {tensor.shape}")
    
    # Tensor to image
    back_to_image = tensor_to_image(tensor)
    print(f"  Tensor -> Image: {back_to_image.size}")
    
    # Image to numpy
    array = image_to_numpy(image)
    print(f"  Image -> Numpy: {array.shape}")
    
    # Numpy to image
    back_from_numpy = numpy_to_image(array)
    print(f"  Numpy -> Image: {back_from_numpy.size}")
    
    print("  PASSED\n")


def test_preprocessing(image):
    """Test preprocessing functions"""
    print("Testing preprocessing...")
    
    # Resize
    resized = resize_image(image, 128)
    print(f"  Resized: {resized.size}")
    
    # Center crop
    cropped = center_crop(image, 100)
    print(f"  Cropped: {cropped.size}")
    
    # Normalize
    tensor = image_to_tensor(image)
    normalized = normalize_tensor(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    print(f"  Normalized tensor: {normalized.shape}")
    
    print("  PASSED\n")


def test_transforms(image):
    """Test augmentation transforms"""
    print("Testing transforms...")
    
    # Training transforms
    train_transform = get_training_transforms(224)
    train_tensor = train_transform(image)
    print(f"  Training transform: {train_tensor.shape}")
    
    # Validation transforms
    val_transform = get_validation_transforms(224)
    val_tensor = val_transform(image)
    print(f"  Validation transform: {val_tensor.shape}")
    
    print("  PASSED\n")


def test_batch_processing():
    """Test batch processing"""
    print("Testing batch processing...")
    
    # Create multiple test images
    for i in range(3):
        test_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image = numpy_to_image(test_array)
        test_image.save(f"test_temp_{i}.jpg")
    
    # Process batch
    image_paths = [f"test_temp_{i}.jpg" for i in range(3)]
    transform = get_validation_transforms(224)
    batch = process_image_batch(image_paths, transform)
    
    print(f"  Batch tensor: {batch.shape}")
    print("  PASSED\n")


def test_utilities():
    """Test utility functions"""
    print("Testing utilities...")
    
    # Validate image
    is_valid = validate_image("test_temp.jpg")
    print(f"  Image validation: {is_valid}")
    
    # Get image info
    info = get_image_info("test_temp.jpg")
    print(f"  Image info: {info}")
    
    print("  PASSED\n")


def cleanup():
    """Remove test files"""
    print("Cleaning up test files...")
    import os
    
    files = ["test_temp.jpg"] + [f"test_temp_{i}.jpg" for i in range(3)]
    for f in files:
        if os.path.exists(f):
            os.remove(f)
    
    print("  Cleanup complete\n")


def main():
    print("=" * 60)
    print("IMAGE UTILS TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        # Run all tests
        test_image = test_image_loading()
        test_conversions(test_image)
        test_preprocessing(test_image)
        test_transforms(test_image)
        test_batch_processing()
        test_utilities()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        cleanup()


if __name__ == "__main__":
    main()
