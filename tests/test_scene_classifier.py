"""
Quick test script for scene_classifier.py
Tests scene classification on sample images
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.scene_classifier import SceneClassifier


def create_test_image(scene_type='beach'):
    """Create a colored test image representing different scenes."""
    # Create different colored images for different scene types
    colors = {
        'beach': (255, 220, 150),      # Sandy color
        'mountain': (139, 137, 137),   # Gray/rocky
        'forest': (34, 139, 34),       # Green
        'indoor': (200, 200, 200),     # Neutral gray
    }
    
    color = colors.get(scene_type, (128, 128, 128))
    img_array = np.full((224, 224, 3), color, dtype=np.uint8)
    return Image.fromarray(img_array)


def test_classifier_init():
    """Test classifier initialization."""
    print("=" * 60)
    print("TEST 1: Classifier Initialization")
    print("=" * 60)
    
    try:
        classifier = SceneClassifier()
        print(f"✓ Classifier initialized successfully")
        print(f"  Device: {classifier.device}")
        print(f"  Categories: {len(classifier.categories)}")
        print(f"  Model: {type(classifier.model).__name__}")
        print()
        return classifier
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_single_prediction(classifier):
    """Test single image prediction."""
    print("=" * 60)
    print("TEST 2: Single Image Prediction")
    print("=" * 60)
    
    try:
        # Create test image
        test_img = create_test_image('beach')
        
        # Get predictions
        predictions = classifier.predict(test_img, top_k=5)
        
        print(f"✓ Prediction successful")
        print(f"\nTop 5 Scene Predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred['category']:30s} - {pred['confidence']*100:5.2f}%")
        print()
        
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_prediction(classifier):
    """Test batch prediction."""
    print("=" * 60)
    print("TEST 3: Batch Prediction")
    print("=" * 60)
    
    try:
        # Create multiple test images
        images = [
            create_test_image('beach'),
            create_test_image('mountain'),
            create_test_image('forest')
        ]
        
        # Get batch predictions
        results = classifier.predict_batch(images, top_k=3)
        
        print(f"✓ Batch prediction successful")
        print(f"\nProcessed {len(results)} images:")
        
        for i, predictions in enumerate(results, 1):
            print(f"\n  Image {i} - Top 3 predictions:")
            for pred in predictions:
                print(f"    {pred['category']:25s} - {pred['confidence']*100:5.2f}%")
        print()
        
        return True
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utility_methods(classifier):
    """Test utility methods."""
    print("=" * 60)
    print("TEST 4: Utility Methods")
    print("=" * 60)
    
    try:
        test_img = create_test_image('beach')
        
        # Test get_top_category
        top_cat = classifier.get_top_category(test_img)
        print(f"✓ Top category: {top_cat}")
        
        # Test is_travel_relevant
        predictions = classifier.predict(test_img, top_k=5)
        is_relevant = classifier.is_travel_relevant(predictions)
        print(f"✓ Travel relevant: {is_relevant}")
        
        # Test filter by type
        outdoor_preds = classifier.filter_by_category_type(predictions, 'outdoor')
        print(f"✓ Outdoor scenes found: {len(outdoor_preds)}")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Utility methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance(classifier):
    """Test inference speed."""
    print("=" * 60)
    print("TEST 5: Performance Benchmark")
    print("=" * 60)
    
    try:
        import time
        
        test_img = create_test_image('beach')
        
        # Warm-up
        classifier.predict(test_img, top_k=1)
        
        # Benchmark single prediction
        iterations = 10
        start = time.time()
        for _ in range(iterations):
            classifier.predict(test_img, top_k=5)
        end = time.time()
        
        avg_time = (end - start) / iterations * 1000
        print(f"✓ Average inference time: {avg_time:.2f}ms")
        print(f"  ({iterations} iterations)")
        
        # Benchmark batch
        images = [create_test_image('beach') for _ in range(10)]
        start = time.time()
        classifier.predict_batch(images, top_k=5)
        end = time.time()
        
        batch_time = (end - start) * 1000
        per_image = batch_time / len(images)
        print(f"✓ Batch inference (10 images): {batch_time:.2f}ms")
        print(f"  ({per_image:.2f}ms per image)")
        print()
        
        return True
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "SCENE CLASSIFIER TEST SUITE" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Initialize classifier
    classifier = test_classifier_init()
    if not classifier:
        print("\n✗ Failed to initialize classifier. Aborting tests.")
        return
    
    # Run tests
    tests_passed = 0
    total_tests = 5
    
    if test_single_prediction(classifier):
        tests_passed += 1
    
    if test_batch_prediction(classifier):
        tests_passed += 1
    
    if test_utility_methods(classifier):
        tests_passed += 1
    
    if test_performance(classifier):
        tests_passed += 1
    
    tests_passed += 1  # Init test
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ {total_tests - tests_passed} test(s) failed")
    print()


if __name__ == "__main__":
    main()
