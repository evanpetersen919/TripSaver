"""
Quick test script for landmark_detector.py
Tests landmark recognition on sample images
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.landmark_detector import LandmarkDetector


def create_test_image(color=(128, 128, 128), size=(300, 300)):
    """Create a colored test image."""
    img_array = np.full((size[0], size[1], 3), color, dtype=np.uint8)
    return Image.fromarray(img_array)


def test_detector_init():
    """Test landmark detector initialization."""
    print("=" * 60)
    print("TEST 1: Landmark Detector Initialization")
    print("=" * 60)
    
    try:
        detector = LandmarkDetector()
        print(f"✓ Detector initialized successfully")
        print(f"  Device: {detector.device}")
        print(f"  Number of landmarks: {detector.num_classes}")
        print(f"  Sample landmarks: {detector.landmarks[:5]}")
        print()
        return detector
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_single_prediction(detector):
    """Test single image prediction."""
    print("=" * 60)
    print("TEST 2: Single Image Prediction")
    print("=" * 60)
    
    try:
        # Create test image
        test_img = create_test_image((100, 150, 200))
        
        # Get predictions
        predictions = detector.predict(test_img, top_k=5)
        
        print(f"✓ Prediction successful")
        print(f"\nTop 5 Landmark Predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred['landmark']:30s} - {pred['confidence']*100:5.2f}%")
        print()
        
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_prediction(detector):
    """Test batch prediction."""
    print("=" * 60)
    print("TEST 3: Batch Prediction")
    print("=" * 60)
    
    try:
        # Create multiple test images
        images = [
            create_test_image((255, 100, 100)),
            create_test_image((100, 255, 100)),
            create_test_image((100, 100, 255)),
        ]
        
        # Get batch predictions
        results = detector.predict_batch(images, top_k=3)
        
        print(f"✓ Batch prediction successful")
        print(f"\nProcessed {len(results)} images:")
        
        for i, predictions in enumerate(results, 1):
            print(f"\n  Image {i} - Top 3 predictions:")
            for pred in predictions:
                print(f"    {pred['landmark']:25s} - {pred['confidence']*100:5.2f}%")
        print()
        
        return True
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utility_methods(detector):
    """Test utility methods."""
    print("=" * 60)
    print("TEST 4: Utility Methods")
    print("=" * 60)
    
    try:
        test_img = create_test_image((150, 150, 150))
        
        # Test get_top_landmark
        top_landmark = detector.get_top_landmark(test_img)
        print(f"✓ Top landmark: {top_landmark}")
        
        # Test is_known_landmark
        is_known = detector.is_known_landmark(test_img, threshold=0.5)
        print(f"✓ Is known landmark: {is_known}")
        
        # Test get_landmark_metadata
        metadata = detector.get_landmark_metadata('eiffel_tower')
        print(f"✓ Landmark metadata: {metadata}")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Utility methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_mode(detector):
    """Test training mode switching."""
    print("=" * 60)
    print("TEST 5: Training Mode")
    print("=" * 60)
    
    try:
        # Set training mode (freeze backbone)
        detector.set_training_mode(freeze_backbone=True)
        print(f"✓ Training mode set (backbone frozen)")
        
        # Set training mode (full model)
        detector.set_training_mode(freeze_backbone=False)
        print(f"✓ Training mode set (full model)")
        
        # Back to eval mode
        detector.set_eval_mode()
        print(f"✓ Evaluation mode set")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Training mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load(detector):
    """Test model save/load."""
    print("=" * 60)
    print("TEST 6: Save/Load Model")
    print("=" * 60)
    
    try:
        import tempfile
        import shutil
        
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())
        model_path = temp_dir / "test_model.pth"
        
        # Save model
        detector.save_model(str(model_path))
        print(f"✓ Model saved")
        
        # Load model
        detector.load_model(str(model_path))
        print(f"✓ Model loaded")
        
        # Test prediction after load
        test_img = create_test_image()
        predictions = detector.predict(test_img, top_k=1)
        print(f"✓ Prediction after load: {predictions[0]['landmark']}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"✓ Cleanup complete")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Save/load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance(detector):
    """Test inference speed."""
    print("=" * 60)
    print("TEST 7: Performance Benchmark")
    print("=" * 60)
    
    try:
        import time
        
        test_img = create_test_image()
        
        # Warm-up
        detector.predict(test_img, top_k=1)
        
        # Benchmark single prediction
        iterations = 10
        start = time.time()
        for _ in range(iterations):
            detector.predict(test_img, top_k=5)
        end = time.time()
        
        avg_time = (end - start) / iterations * 1000
        print(f"✓ Average inference time: {avg_time:.2f}ms")
        print(f"  ({iterations} iterations)")
        
        # Benchmark batch
        images = [create_test_image() for _ in range(10)]
        start = time.time()
        detector.predict_batch(images, top_k=5)
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
    print("║" + " " * 13 + "LANDMARK DETECTOR TEST SUITE" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Initialize detector
    detector = test_detector_init()
    if not detector:
        print("\n✗ Failed to initialize detector. Aborting tests.")
        return
    
    # Run tests
    tests_passed = 1  # Init test
    total_tests = 7
    
    if test_single_prediction(detector):
        tests_passed += 1
    
    if test_batch_prediction(detector):
        tests_passed += 1
    
    if test_utility_methods(detector):
        tests_passed += 1
    
    if test_training_mode(detector):
        tests_passed += 1
    
    if test_save_load(detector):
        tests_passed += 1
    
    if test_performance(detector):
        tests_passed += 1
    
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
