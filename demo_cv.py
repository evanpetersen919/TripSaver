"""
CV Pipeline Demonstration Script

Showcases the complete computer vision pipeline for photo location detection.
Demonstrates parallel model execution, result aggregation, and performance benchmarking.

Author: Evan Petersen
Date: November 2025
"""

import sys
from pathlib import Path
import argparse
from PIL import Image
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.vision_pipeline import VisionPipeline
from core.config import config


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def demo_single_image(pipeline: VisionPipeline, image_path: str):
    """Demonstrate single image prediction"""
    print("\n" + "=" * 80)
    print("DEMO 1: SINGLE IMAGE PREDICTION")
    print("=" * 80)
    
    # Load image
    print(f"\nLoading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    # Run prediction
    print("\nRunning parallel CV models...")
    results = pipeline.predict(image)
    
    # Display results
    pipeline.print_results(results, show_timing=True)
    
    # Show aggregated prediction
    aggregated = pipeline.aggregate_predictions(results)
    print("\n" + "=" * 80)
    print("AGGREGATED LOCATION PREDICTION")
    print("=" * 80)
    print(f"Location Type: {aggregated['location_type']}")
    print(f"Location Name: {aggregated['location_name']}")
    print(f"Confidence: {aggregated['confidence']:.2%}")
    print(f"\nEvidence from {len(aggregated['evidence'])} models:")
    for evidence in aggregated['evidence']:
        print(f"  - {evidence['source']}: {evidence['value']} ({evidence['confidence']:.2%})")
    print("=" * 80)


def demo_batch_processing(pipeline: VisionPipeline, image_dir: str, num_images: int = 5):
    """Demonstrate batch processing"""
    print("\n" + "=" * 80)
    print("DEMO 2: BATCH PROCESSING")
    print("=" * 80)
    
    # Load images
    image_paths = list(Path(image_dir).glob("*.jpg"))[:num_images]
    if not image_paths:
        image_paths = list(Path(image_dir).glob("*.png"))[:num_images]
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\nLoading {len(image_paths)} images from {image_dir}...")
    images = [Image.open(path).convert('RGB') for path in image_paths]
    
    # Process batch
    print("\nProcessing batch...")
    start = time.time()
    results = pipeline.predict_batch(images)
    total_time = time.time() - start
    
    # Display results
    print("\n" + "=" * 80)
    print("BATCH RESULTS")
    print("=" * 80)
    for i, (path, result) in enumerate(zip(image_paths, results), 1):
        aggregated = pipeline.aggregate_predictions(result)
        print(f"\nImage {i}: {path.name}")
        print(f"  Location: {aggregated['location_name']}")
        print(f"  Type: {aggregated['location_type']}")
        print(f"  Confidence: {aggregated['confidence']:.2%}")
        print(f"  Time: {result['total_time_ms']:.2f}ms")
    
    print("\n" + "=" * 80)
    print(f"Total batch time: {total_time:.2f}s")
    print(f"Average per image: {(total_time / len(images)):.2f}s")
    print("=" * 80)


def demo_performance_benchmark(pipeline: VisionPipeline, image_path: str, iterations: int = 10):
    """Demonstrate performance benchmarking"""
    print("\n" + "=" * 80)
    print("DEMO 3: PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    # Load image
    print(f"\nLoading test image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # Run benchmark
    print(f"\nRunning {iterations} iterations...")
    stats = pipeline.benchmark(image, iterations=iterations)
    
    # Display detailed stats
    print("\nPERFORMANCE ANALYSIS:")
    print(f"  Mean: {stats['mean_ms']:.2f}ms")
    print(f"  Min: {stats['min_ms']:.2f}ms")
    print(f"  Max: {stats['max_ms']:.2f}ms")
    print(f"  Throughput: {1000 / stats['mean_ms']:.1f} images/second")
    print("=" * 80)


def demo_model_comparison(pipeline: VisionPipeline, image_path: str):
    """Demonstrate individual model outputs"""
    print("\n" + "=" * 80)
    print("DEMO 4: MODEL-BY-MODEL COMPARISON")
    print("=" * 80)
    
    # Load image
    print(f"\nLoading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # Run prediction
    results = pipeline.predict(image)
    
    # Scene Classifier Analysis
    if results.get('scene_classifier'):
        scene = results['scene_classifier']
        print("\n" + "-" * 80)
        print("SCENE CLASSIFIER (Places365)")
        print("-" * 80)
        print(f"Primary Scene: {scene['top_scene']}")
        print(f"Confidence: {scene['confidence']:.2%}")
        print(f"Inference Time: {scene['elapsed_ms']:.2f}ms")
        print(f"\nTop 10 Scene Predictions:")
        for i, pred in enumerate(scene['predictions'][:10], 1):
            print(f"  {i:2d}. {pred['category']:30s} - {pred['confidence']:6.2%}")
    
    # CLIP Embedder Analysis
    if results.get('clip_embedder') and results['clip_embedder']['results']:
        clip = results['clip_embedder']
        print("\n" + "-" * 80)
        print("CLIP VISUAL SIMILARITY SEARCH")
        print("-" * 80)
        print(f"Top Match: {clip['top_match'].get('image_id', 'N/A')}")
        print(f"Similarity: {clip['similarity']:.2%}")
        print(f"Inference Time: {clip['elapsed_ms']:.2f}ms")
        print(f"\nTop 10 Similar Images:")
        for i, match in enumerate(clip['results'][:10], 1):
            print(f"  {i:2d}. {match.get('image_id', 'N/A'):30s} - {match['similarity']:6.2%}")
    
    # Landmark Detector Analysis
    if results.get('landmark_detector'):
        landmark = results['landmark_detector']
        print("\n" + "-" * 80)
        print("LANDMARK DETECTOR (EfficientNet-B3)")
        print("-" * 80)
        print(f"Primary Landmark: {landmark['top_landmark']}")
        print(f"Confidence: {landmark['confidence']:.2%}")
        print(f"Inference Time: {landmark['elapsed_ms']:.2f}ms")
        print(f"\nTop 10 Landmark Predictions:")
        for i, pred in enumerate(landmark['predictions'][:10], 1):
            print(f"  {i:2d}. {pred['landmark']:30s} - {pred['confidence']:6.2%}")
    
    print("\n" + "-" * 80)


def demo_configuration():
    """Demonstrate configuration system"""
    print("\n" + "=" * 80)
    print("DEMO 5: CONFIGURATION SYSTEM")
    print("=" * 80)
    
    config.print_summary()
    
    print("\nConfiguration Details:")
    print(f"  Scene Classifier Threshold: {config.scene_classifier.confidence_threshold}")
    print(f"  CLIP Similarity Threshold: {config.clip.similarity_threshold}")
    print(f"  Landmark Confidence Threshold: {config.landmark_detector.confidence_threshold}")
    print(f"  Parallel Execution: {config.pipeline.parallel_execution}")
    print(f"  Async Mode: {config.pipeline.use_asyncio}")
    print("=" * 80)


def create_test_image():
    """Create a test image if no real image available"""
    import numpy as np
    
    print("\nNo test image provided - creating random test image...")
    array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    
    # Save test image
    test_path = Path("data/test_image.jpg")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(test_path)
    print(f"Test image saved to {test_path}")
    
    return str(test_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CV Pipeline Demo - Photo Location Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all demos with a specific image
  python demo_cv.py --image path/to/photo.jpg --all
  
  # Run single image demo
  python demo_cv.py --image path/to/photo.jpg --single
  
  # Run batch processing demo
  python demo_cv.py --batch-dir path/to/images/ --num-images 10
  
  # Run performance benchmark
  python demo_cv.py --image path/to/photo.jpg --benchmark --iterations 20
  
  # Show configuration
  python demo_cv.py --config
        """
    )
    
    # Image options
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--batch-dir', type=str, help='Directory with images for batch processing')
    parser.add_argument('--num-images', type=int, default=5, help='Number of images for batch demo')
    
    # Demo options
    parser.add_argument('--all', action='store_true', help='Run all demos')
    parser.add_argument('--single', action='store_true', help='Run single image demo')
    parser.add_argument('--batch', action='store_true', help='Run batch processing demo')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--comparison', action='store_true', help='Run model comparison demo')
    parser.add_argument('--config', action='store_true', help='Show configuration')
    
    # Pipeline options
    parser.add_argument('--no-scene', action='store_true', help='Disable scene classifier')
    parser.add_argument('--no-landmark', action='store_true', help='Disable landmark detector')
    parser.add_argument('--clip-index', type=str, help='Path to CLIP index')
    parser.add_argument('--landmark-weights', type=str, help='Path to trained landmark weights')
    
    # Benchmark options
    parser.add_argument('--iterations', type=int, default=10, help='Benchmark iterations')
    
    args = parser.parse_args()
    
    # Display header
    print("=" * 80)
    print("CV PIPELINE DEMONSTRATION")
    print("Photo Location Detection using Parallel AI Models")
    print("=" * 80)
    print(f"Device: {config.device.device.upper()}")
    print(f"Scene Classifier: {'Enabled' if not args.no_scene else 'Disabled'}")
    print(f"Landmark Detector: {'Enabled' if not args.no_landmark else 'Disabled'}")
    print(f"CLIP Embedder: {'Enabled' if args.clip_index else 'Disabled (no index)'}")
    print("=" * 80)
    
    # Show config demo
    if args.config or args.all:
        demo_configuration()
        if args.config and not args.all:
            return
    
    # Initialize pipeline
    print("\nInitializing CV Pipeline...")
    pipeline = VisionPipeline(
        enable_scene=not args.no_scene,
        enable_clip=bool(args.clip_index),
        enable_landmark=not args.no_landmark,
        clip_index_path=args.clip_index,
        landmark_weights_path=args.landmark_weights
    )
    
    # Get test image
    if not args.image and not args.batch_dir:
        args.image = create_test_image()
    
    # Run demos
    if args.single or args.all:
        if args.image:
            demo_single_image(pipeline, args.image)
        else:
            print("\nSkipping single image demo (no image provided)")
    
    if args.batch or args.all:
        batch_dir = args.batch_dir or (Path(args.image).parent if args.image else None)
        if batch_dir:
            demo_batch_processing(pipeline, batch_dir, args.num_images)
        else:
            print("\nSkipping batch demo (no batch directory provided)")
    
    if args.benchmark or args.all:
        if args.image:
            demo_performance_benchmark(pipeline, args.image, args.iterations)
        else:
            print("\nSkipping benchmark (no image provided)")
    
    if args.comparison or args.all:
        if args.image:
            demo_model_comparison(pipeline, args.image)
        else:
            print("\nSkipping model comparison (no image provided)")
    
    # If no specific demo requested, run single image demo
    if not any([args.all, args.single, args.batch, args.benchmark, args.comparison, args.config]):
        if args.image:
            demo_single_image(pipeline, args.image)
        else:
            print("\nNo demo specified. Use --help for options.")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
