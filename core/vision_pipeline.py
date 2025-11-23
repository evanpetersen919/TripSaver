"""
Vision Pipeline Orchestrator

Coordinates multiple computer vision models in parallel for photo location detection.
Runs Scene Classifier, CLIP Embedder, and Landmark Detector simultaneously for
minimum latency and maximum accuracy.

Author: Evan Petersen
Date: November 2025
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor
import warnings

from models.scene_classifier import SceneClassifier
from models.clip_embedder import ClipEmbedder
from models.landmark_detector import LandmarkDetector
from core.config import config


# ============================================================================
# VISION PIPELINE CLASS
# ============================================================================

class VisionPipeline:
    """
    Orchestrates parallel execution of multiple CV models.
    
    Combines predictions from:
    - Scene Classifier (365 scene categories)
    - CLIP Embedder (visual similarity search)
    - Landmark Detector (famous landmarks)
    
    Provides unified location prediction with confidence scores.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        enable_scene: bool = True,
        enable_clip: bool = True,
        enable_landmark: bool = True,
        clip_index_path: Optional[str] = None,
        landmark_weights_path: Optional[str] = None,
        landmark_names_path: Optional[str] = None
    ):
        """Initialize the vision pipeline
        
        Args:
            device: Device to run models on ('cuda' or 'cpu')
            enable_scene: Enable scene classification
            enable_clip: Enable CLIP similarity search
            enable_landmark: Enable landmark detection
            clip_index_path: Path to FAISS index for CLIP
            landmark_weights_path: Path to trained landmark weights
            landmark_names_path: Path to JSON file with landmark names
        """
        self.device = device or config.device.get_device()
        
        print("=" * 80)
        print("INITIALIZING VISION PIPELINE")
        print("=" * 80)
        
        # Initialize models
        self.scene_classifier = None
        self.clip_embedder = None
        self.landmark_detector = None
        
        if enable_scene:
            print("Loading Scene Classifier...")
            self.scene_classifier = SceneClassifier(device=str(self.device))
            print(f"✓ Scene Classifier ready ({self.scene_classifier.num_classes} categories)")
        
        if enable_clip:
            print("Loading CLIP Embedder...")
            self.clip_embedder = ClipEmbedder(device=str(self.device))
            if clip_index_path and Path(clip_index_path).exists():
                self.clip_embedder.load_index(clip_index_path)
                print(f"✓ CLIP Embedder ready (index loaded: {self.clip_embedder.index.ntotal} images)")
            else:
                print("✓ CLIP Embedder ready (no index loaded)")
        
        if enable_landmark:
            print("Loading Landmark Detector...")
            self.landmark_detector = LandmarkDetector(
                model_path=landmark_weights_path if landmark_weights_path else None,
                landmark_names_path=landmark_names_path if landmark_names_path else None
            )
            print(f"✓ Landmark Detector ready ({self.landmark_detector.num_classes} landmarks)")
        
        print("=" * 80)
        print(f"✓ Pipeline initialized on {self.device}")
        print("=" * 80)
        print()
    
    
    # ========================================================================
    # PARALLEL EXECUTION
    # ========================================================================
    
    async def _run_scene_classifier(self, image: Image.Image) -> Dict[str, Any]:
        """Run scene classifier asynchronously"""
        if not self.scene_classifier:
            return None
        
        start = time.time()
        predictions = self.scene_classifier.predict(image, top_k=5)
        elapsed = (time.time() - start) * 1000
        
        return {
            'model': 'scene_classifier',
            'predictions': predictions,
            'top_scene': predictions[0]['category'],
            'confidence': predictions[0]['confidence'],
            'elapsed_ms': elapsed
        }
    
    
    async def _run_clip_embedder(self, image: Image.Image, top_k: int = 10) -> Dict[str, Any]:
        """Run CLIP similarity search asynchronously"""
        if not self.clip_embedder or self.clip_embedder.index is None:
            return None
        
        start = time.time()
        
        # Encode image
        embedding = self.clip_embedder.encode_image(image)
        
        # Search index
        results = self.clip_embedder.search(embedding, top_k=top_k)
        
        elapsed = (time.time() - start) * 1000
        
        return {
            'model': 'clip_embedder',
            'results': results,
            'top_match': results[0] if results else None,
            'similarity': results[0]['similarity'] if results else 0.0,
            'elapsed_ms': elapsed
        }
    
    
    async def _run_landmark_detector(self, image: Image.Image) -> Dict[str, Any]:
        """Run landmark detector asynchronously"""
        if not self.landmark_detector:
            return None
        
        start = time.time()
        predictions = self.landmark_detector.predict(image, top_k=5)
        elapsed = (time.time() - start) * 1000
        
        return {
            'model': 'landmark_detector',
            'predictions': predictions,
            'top_landmark': predictions[0]['landmark'],
            'confidence': predictions[0]['confidence'],
            'elapsed_ms': elapsed
        }
    
    
    async def predict_async(
        self,
        image: Image.Image,
        clip_top_k: int = 10
    ) -> Dict[str, Any]:
        """Run all models in parallel asynchronously
        
        Args:
            image: PIL Image to analyze
            clip_top_k: Number of similar images to return from CLIP
            
        Returns:
            Dictionary with results from all models
        """
        start_time = time.time()
        
        # Run all models in parallel
        tasks = []
        
        if self.scene_classifier:
            tasks.append(self._run_scene_classifier(image))
        
        if self.clip_embedder:
            tasks.append(self._run_clip_embedder(image, clip_top_k))
        
        if self.landmark_detector:
            tasks.append(self._run_landmark_detector(image))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = (time.time() - start_time) * 1000
        
        # Organize results
        output = {
            'scene_classifier': None,
            'clip_embedder': None,
            'landmark_detector': None,
            'total_time_ms': total_time,
            'timestamp': time.time()
        }
        
        for result in results:
            if isinstance(result, Exception):
                warnings.warn(f"Model error: {result}")
                continue
            
            if result is None:
                continue
            
            model_name = result['model']
            output[model_name] = result
        
        return output
    
    
    def predict(
        self,
        image: Image.Image,
        clip_top_k: int = 10
    ) -> Dict[str, Any]:
        """Run all models in parallel (synchronous wrapper)
        
        Args:
            image: PIL Image to analyze
            clip_top_k: Number of similar images to return from CLIP
            
        Returns:
            Dictionary with results from all models
        """
        # Run async function in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.predict_async(image, clip_top_k))
    
    
    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================
    
    def predict_batch(
        self,
        images: List[Image.Image],
        clip_top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Process multiple images in parallel
        
        Args:
            images: List of PIL Images
            clip_top_k: Number of similar images to return from CLIP
            
        Returns:
            List of prediction dictionaries
        """
        start_time = time.time()
        
        print(f"Processing batch of {len(images)} images...")
        
        results = []
        for i, image in enumerate(images):
            result = self.predict(image, clip_top_k)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(images)} images...")
        
        total_time = time.time() - start_time
        avg_time = (total_time / len(images)) * 1000
        
        print(f"✓ Batch complete: {len(images)} images in {total_time:.2f}s ({avg_time:.2f}ms/image)")
        
        return results
    
    
    # ========================================================================
    # RESULT AGGREGATION
    # ========================================================================
    
    def aggregate_predictions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate predictions from all models into unified location prediction
        
        Args:
            results: Output from predict() or predict_async()
            
        Returns:
            Aggregated prediction with location and confidence
        """
        aggregated = {
            'location_type': 'unknown',
            'location_name': 'unknown',
            'confidence': 0.0,
            'evidence': []
        }
        
        # Check landmark detector first (most specific)
        if results.get('landmark_detector'):
            landmark = results['landmark_detector']
            if landmark['confidence'] > config.landmark_detector.confidence_threshold:
                aggregated['location_type'] = 'landmark'
                aggregated['location_name'] = landmark['top_landmark']
                aggregated['confidence'] = landmark['confidence']
                aggregated['evidence'].append({
                    'source': 'landmark_detector',
                    'value': landmark['top_landmark'],
                    'confidence': landmark['confidence']
                })
        
        # Check CLIP similarity (for niche locations)
        if results.get('clip_embedder') and results['clip_embedder']['top_match']:
            clip_result = results['clip_embedder']
            if clip_result['similarity'] > config.clip.similarity_threshold:
                if aggregated['location_type'] == 'unknown':
                    aggregated['location_type'] = 'similar_location'
                    aggregated['location_name'] = clip_result['top_match'].get('location', 'unknown')
                    aggregated['confidence'] = clip_result['similarity']
                
                aggregated['evidence'].append({
                    'source': 'clip_embedder',
                    'value': clip_result['top_match'].get('location', 'unknown'),
                    'confidence': clip_result['similarity']
                })
        
        # Use scene classification (most general)
        if results.get('scene_classifier'):
            scene = results['scene_classifier']
            if scene['confidence'] > config.scene_classifier.confidence_threshold:
                if aggregated['location_type'] == 'unknown':
                    aggregated['location_type'] = 'scene'
                    aggregated['location_name'] = scene['top_scene']
                    aggregated['confidence'] = scene['confidence']
                
                aggregated['evidence'].append({
                    'source': 'scene_classifier',
                    'value': scene['top_scene'],
                    'confidence': scene['confidence']
                })
        
        return aggregated
    
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def print_results(self, results: Dict[str, Any], show_timing: bool = True):
        """Pretty print results from prediction
        
        Args:
            results: Output from predict()
            show_timing: Whether to show timing information
        """
        print("\n" + "=" * 80)
        print("VISION PIPELINE RESULTS")
        print("=" * 80)
        
        # Scene Classifier
        if results.get('scene_classifier'):
            scene = results['scene_classifier']
            print("\nSCENE CLASSIFIER:")
            print(f"  Top Scene: {scene['top_scene']}")
            print(f"  Confidence: {scene['confidence']:.2%}")
            if show_timing:
                print(f"  Time: {scene['elapsed_ms']:.2f}ms")
            print(f"  Top 5 Predictions:")
            for pred in scene['predictions'][:5]:
                print(f"    - {pred['category']}: {pred['confidence']:.2%}")
        
        # CLIP Embedder
        if results.get('clip_embedder') and results['clip_embedder']['results']:
            clip = results['clip_embedder']
            print("\nCLIP SIMILARITY SEARCH:")
            print(f"  Top Match: {clip['top_match'].get('image_id', 'N/A')}")
            print(f"  Similarity: {clip['similarity']:.2%}")
            if show_timing:
                print(f"  Time: {clip['elapsed_ms']:.2f}ms")
            print(f"  Top 5 Matches:")
            for match in clip['results'][:5]:
                print(f"    - {match.get('image_id', 'N/A')}: {match['similarity']:.2%}")
        
        # Landmark Detector
        if results.get('landmark_detector'):
            landmark = results['landmark_detector']
            print("\nLANDMARK DETECTOR:")
            print(f"  Top Landmark: {landmark['top_landmark']}")
            print(f"  Confidence: {landmark['confidence']:.2%}")
            if show_timing:
                print(f"  Time: {landmark['elapsed_ms']:.2f}ms")
            print(f"  Top 5 Predictions:")
            for pred in landmark['predictions'][:5]:
                print(f"    - {pred['landmark']}: {pred['confidence']:.2%}")
        
        # Overall timing
        if show_timing:
            print(f"\n{'=' * 80}")
            print(f"TOTAL PIPELINE TIME: {results['total_time_ms']:.2f}ms")
            print(f"{'=' * 80}\n")
    
    
    def benchmark(self, image: Image.Image, iterations: int = 10) -> Dict[str, float]:
        """Benchmark pipeline performance
        
        Args:
            image: Test image
            iterations: Number of iterations
            
        Returns:
            Dictionary with timing statistics
        """
        print(f"\nBenchmarking pipeline ({iterations} iterations)...")
        
        times = []
        for i in range(iterations):
            start = time.time()
            self.predict(image)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        stats = {
            'mean_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'iterations': iterations
        }
        
        print(f"\n{'=' * 80}")
        print("BENCHMARK RESULTS")
        print(f"{'=' * 80}")
        print(f"Average: {stats['mean_ms']:.2f}ms")
        print(f"Min: {stats['min_ms']:.2f}ms")
        print(f"Max: {stats['max_ms']:.2f}ms")
        print(f"Iterations: {stats['iterations']}")
        print(f"{'=' * 80}\n")
        
        return stats


# ============================================================================
# QUICK START
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    print("Initializing Vision Pipeline...")
    
    pipeline = VisionPipeline(
        enable_scene=True,
        enable_clip=False,  # No index loaded by default
        enable_landmark=True
    )
    
    # Create test image
    test_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_array)
    
    print("\nRunning prediction...")
    results = pipeline.predict(test_image)
    
    pipeline.print_results(results)
    
    # Aggregate predictions
    aggregated = pipeline.aggregate_predictions(results)
    print("\nAGGREGATED PREDICTION:")
    print(f"  Location Type: {aggregated['location_type']}")
    print(f"  Location Name: {aggregated['location_name']}")
    print(f"  Confidence: {aggregated['confidence']:.2%}")
    print(f"  Evidence Sources: {len(aggregated['evidence'])}")
    
    # Benchmark
    pipeline.benchmark(test_image, iterations=5)
