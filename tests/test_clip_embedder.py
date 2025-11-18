"""
Quick test script for clip_embedder.py
Tests CLIP embedding and FAISS similarity search
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.clip_embedder import ClipEmbedder


def create_test_image(color=(128, 128, 128), size=(224, 224)):
    """Create a colored test image."""
    img_array = np.full((size[0], size[1], 3), color, dtype=np.uint8)
    return Image.fromarray(img_array)


def test_embedder_init():
    """Test CLIP embedder initialization."""
    print("=" * 60)
    print("TEST 1: CLIP Embedder Initialization")
    print("=" * 60)
    
    try:
        embedder = ClipEmbedder(model_name="ViT-B/32")
        print(f"✓ Embedder initialized successfully")
        print(f"  Device: {embedder.device}")
        print(f"  Model: {embedder.model_name}")
        print(f"  Embedding dimension: {embedder.embedding_dim}")
        print()
        return embedder
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_image_encoding(embedder):
    """Test image to embedding conversion."""
    print("=" * 60)
    print("TEST 2: Image Encoding")
    print("=" * 60)
    
    try:
        # Create test image
        test_img = create_test_image((255, 0, 0))  # Red image
        
        # Generate embedding
        embedding = embedder.encode_image(test_img)
        
        print(f"✓ Encoding successful")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding dtype: {embedding.dtype}")
        print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
        print()
        
        return True
    except Exception as e:
        print(f"✗ Encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_encoding(embedder):
    """Test batch encoding."""
    print("=" * 60)
    print("TEST 3: Batch Encoding")
    print("=" * 60)
    
    try:
        # Create multiple test images
        images = [
            create_test_image((255, 0, 0)),    # Red
            create_test_image((0, 255, 0)),    # Green
            create_test_image((0, 0, 255)),    # Blue
        ]
        
        # Batch encode
        embeddings = embedder.encode_batch(images)
        
        print(f"✓ Batch encoding successful")
        print(f"  Batch shape: {embeddings.shape}")
        print(f"  Number of images: {len(images)}")
        print()
        
        return True
    except Exception as e:
        print(f"✗ Batch encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_encoding(embedder):
    """Test text encoding."""
    print("=" * 60)
    print("TEST 4: Text Encoding")
    print("=" * 60)
    
    try:
        # Encode text
        text = "a photo of the Eiffel Tower"
        text_embedding = embedder.encode_text(text)
        
        print(f"✓ Text encoding successful")
        print(f"  Text: '{text}'")
        print(f"  Embedding shape: {text_embedding.shape}")
        print()
        
        return True
    except Exception as e:
        print(f"✗ Text encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_index_building(embedder):
    """Test FAISS index building and search."""
    print("=" * 60)
    print("TEST 5: Index Building and Search")
    print("=" * 60)
    
    try:
        # Build index
        embedder.build_index(use_gpu=True)
        print(f"✓ Index built")
        
        # Add images with metadata
        images = [
            create_test_image((255, 0, 0)),
            create_test_image((0, 255, 0)),
            create_test_image((0, 0, 255)),
            create_test_image((255, 255, 0)),
            create_test_image((255, 0, 255)),
        ]
        
        metadata = [
            {'name': 'Red Beach', 'location': 'Greece', 'category': 'beach'},
            {'name': 'Green Forest', 'location': 'Amazon', 'category': 'forest'},
            {'name': 'Blue Ocean', 'location': 'Maldives', 'category': 'ocean'},
            {'name': 'Yellow Desert', 'location': 'Sahara', 'category': 'desert'},
            {'name': 'Purple Mountain', 'location': 'Alps', 'category': 'mountain'},
        ]
        
        embedder.add_images_batch(images, metadata)
        print(f"✓ Added {len(images)} images to index")
        print(f"  Index size: {embedder.get_index_size()}")
        
        # Search for similar images
        query_img = create_test_image((255, 10, 10))  # Slightly different red
        results = embedder.search(query_img, k=3)
        
        print(f"\n✓ Search completed")
        print(f"  Top 3 matches:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. {result['name']:20s} - {result['confidence']*100:5.2f}%")
        print()
        
        return True
    except Exception as e:
        print(f"✗ Index test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_search(embedder):
    """Test text-based search."""
    print("=" * 60)
    print("TEST 6: Text-Based Search")
    print("=" * 60)
    
    try:
        # Search by text
        query_text = "a beautiful beach"
        results = embedder.search_by_text(query_text, k=3)
        
        print(f"✓ Text search completed")
        print(f"  Query: '{query_text}'")
        print(f"  Top 3 matches:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. {result['name']:20s} - {result['confidence']*100:5.2f}%")
        print()
        
        return True
    except Exception as e:
        print(f"✗ Text search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load(embedder):
    """Test saving and loading index."""
    print("=" * 60)
    print("TEST 7: Save/Load Index")
    print("=" * 60)
    
    try:
        import tempfile
        import shutil
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Save index
        embedder.save_index(temp_dir)
        print(f"✓ Index saved to {temp_dir}")
        
        # Create new embedder and load
        new_embedder = ClipEmbedder(model_name="ViT-B/32")
        new_embedder.load_index(temp_dir)
        print(f"✓ Index loaded")
        print(f"  Loaded index size: {new_embedder.get_index_size()}")
        
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


def test_performance(embedder):
    """Test encoding performance."""
    print("=" * 60)
    print("TEST 8: Performance Benchmark")
    print("=" * 60)
    
    try:
        import time
        
        test_img = create_test_image((128, 128, 128))
        
        # Warm-up
        embedder.encode_image(test_img)
        
        # Single image encoding
        iterations = 10
        start = time.time()
        for _ in range(iterations):
            embedder.encode_image(test_img)
        end = time.time()
        
        avg_time = (end - start) / iterations * 1000
        print(f"✓ Single image encoding: {avg_time:.2f}ms")
        
        # Batch encoding
        images = [create_test_image() for _ in range(10)]
        start = time.time()
        embedder.encode_batch(images)
        end = time.time()
        
        batch_time = (end - start) * 1000
        per_image = batch_time / len(images)
        print(f"✓ Batch encoding (10 images): {batch_time:.2f}ms")
        print(f"  ({per_image:.2f}ms per image)")
        
        # Search performance
        query_img = create_test_image()
        start = time.time()
        for _ in range(iterations):
            embedder.search(query_img, k=5)
        end = time.time()
        
        search_time = (end - start) / iterations * 1000
        print(f"✓ Search time (k=5): {search_time:.2f}ms")
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
    print("║" + " " * 15 + "CLIP EMBEDDER TEST SUITE" + " " * 19 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Initialize embedder
    embedder = test_embedder_init()
    if not embedder:
        print("\n✗ Failed to initialize embedder. Aborting tests.")
        print("\nMake sure CLIP and FAISS are installed:")
        print("  pip install git+https://github.com/openai/CLIP.git")
        print("  pip install faiss-gpu")
        return
    
    # Run tests
    tests_passed = 1  # Init test
    total_tests = 8
    
    if test_image_encoding(embedder):
        tests_passed += 1
    
    if test_batch_encoding(embedder):
        tests_passed += 1
    
    if test_text_encoding(embedder):
        tests_passed += 1
    
    if test_index_building(embedder):
        tests_passed += 1
    
    if test_text_search(embedder):
        tests_passed += 1
    
    if test_save_load(embedder):
        tests_passed += 1
    
    if test_performance(embedder):
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
