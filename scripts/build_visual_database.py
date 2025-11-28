"""
Build CLIP visual embeddings database from Google Landmarks v2 images.

This script:
1. Loads landmarks from landmarks_unified.json
2. Finds corresponding images from Google Landmarks v2 dataset
3. Encodes images with CLIP (batch processing)
4. Saves embeddings to data/landmarks_clip_embeddings.npy

Usage:
    python scripts/build_visual_database.py --images_per_landmark 5 --batch_size 32
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.clip_embedder import ClipEmbedder


def load_landmarks(landmarks_path: Path) -> dict:
    """Load landmarks from JSON file."""
    print(f"Loading landmarks from {landmarks_path}...")
    with open(landmarks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data['landmarks'])} landmarks")
    return {lm['landmark_id']: lm for lm in data['landmarks']}


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load Google Landmarks metadata CSV."""
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} image records")
    return df


def find_images_for_landmark(landmark_id: int, metadata_df: pd.DataFrame, 
                             images_dir: Path, max_images: int = 5) -> list:
    """Find image paths for a given landmark_id."""
    # Filter metadata for this landmark
    landmark_images = metadata_df[metadata_df['landmark_id'] == landmark_id]
    
    if len(landmark_images) == 0:
        return []
    
    # Get up to max_images image IDs
    image_ids = landmark_images['id'].head(max_images).tolist()
    
    # Build file paths (images are organized in subdirectories by first 3 chars of ID)
    image_paths = []
    for img_id in image_ids:
        # Google Landmarks format: first_char/second_char/third_char/image_id.jpg
        img_id_str = str(img_id)
        if len(img_id_str) >= 3:
            subdir = images_dir / img_id_str[0] / img_id_str[1] / img_id_str[2]
            img_path = subdir / f"{img_id}.jpg"
            if img_path.exists():
                image_paths.append(img_path)
    
    return image_paths


def load_and_preprocess_image(image_path: Path, preprocess):
    """Load and preprocess an image for CLIP."""
    try:
        image = Image.open(image_path).convert('RGB')
        return preprocess(image)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def encode_landmark_images(landmark_id: int, image_paths: list, 
                           clip_embedder: ClipEmbedder) -> np.ndarray:
    """Encode multiple images for a landmark and average the embeddings."""
    if not image_paths:
        return None
    
    embeddings = []
    for img_path in image_paths:
        try:
            # Load and encode image
            image = Image.open(img_path).convert('RGB')
            embedding = clip_embedder.encode_image(image)
            embeddings.append(embedding)
        except Exception as e:
            print(f"  Warning: Failed to encode {img_path.name}: {e}")
            continue
    
    if not embeddings:
        return None
    
    # Average embeddings and normalize
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    return avg_embedding


def main():
    parser = argparse.ArgumentParser(description='Build CLIP visual embeddings database')
    parser.add_argument('--images_per_landmark', type=int, default=5,
                       help='Maximum images to use per landmark (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for encoding (default: 32)')
    parser.add_argument('--use_clean', action='store_true',
                       help='Use train_clean.csv instead of train.csv (smaller but cleaner)')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    landmarks_path = base_dir / 'data' / 'landmarks_unified.json'
    metadata_file = 'train_clean.csv' if args.use_clean else 'train.csv'
    metadata_path = base_dir / 'data' / 'google_landmarks' / 'metadata' / metadata_file
    images_dir = base_dir / 'data' / 'google_landmarks' / 'train'
    output_path = base_dir / 'data' / 'landmarks_clip_embeddings.npy'
    
    # Load data
    landmarks = load_landmarks(landmarks_path)
    metadata_df = load_metadata(metadata_path)
    
    # Initialize CLIP
    print("Initializing CLIP embedder...")
    clip_embedder = ClipEmbedder()
    embedding_dim = 512  # CLIP ViT-B/32 dimension
    
    # Filter to landmarks with BOTH images AND coordinates
    # - Google landmarks: landmark_id is int AND has coordinates
    # - WikiData landmarks: has google_landmark_id field AND has coordinates
    landmark_ids = []
    for lm_id, lm_data in landmarks.items():
        has_coords = 'latitude' in lm_data and 'longitude' in lm_data
        has_images = (isinstance(lm_id, int)) or ('google_landmark_id' in lm_data)
        
        if has_coords and has_images:
            # Use google_landmark_id for image lookup if available (WikiData landmarks)
            # Otherwise use the landmark_id itself (Google landmarks)
            google_id = lm_data.get('google_landmark_id', lm_id)
            if isinstance(google_id, int):
                landmark_ids.append(google_id)
    
    landmark_ids = sorted(set(landmark_ids))  # Remove duplicates
    embeddings_dict = {}
    
    print(f"\nProcessing {len(landmark_ids)} landmarks with images AND coordinates...")
    print(f"Using up to {args.images_per_landmark} images per landmark")
    print(f"Images directory: {images_dir}")
    
    stats = {
        'processed': 0,
        'with_images': 0,
        'without_images': 0,
        'total_images_encoded': 0
    }
    
    # Process each landmark
    for landmark_id in tqdm(landmark_ids, desc="Encoding landmarks"):
        # Find images for this landmark
        image_paths = find_images_for_landmark(
            landmark_id, metadata_df, images_dir, args.images_per_landmark
        )
        
        stats['processed'] += 1
        
        if image_paths:
            # Encode images and average
            embedding = encode_landmark_images(landmark_id, image_paths, clip_embedder)
            
            if embedding is not None:
                embeddings_dict[landmark_id] = embedding
                stats['with_images'] += 1
                stats['total_images_encoded'] += len(image_paths)
            else:
                stats['without_images'] += 1
        else:
            stats['without_images'] += 1
    
    # Convert to arrays aligned with landmark_ids
    print("\nBuilding final embedding array...")
    embeddings_list = []
    missing_landmarks = []
    
    for landmark_id in landmark_ids:
        if landmark_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[landmark_id])
        else:
            # Use zero vector for landmarks without images
            embeddings_list.append(np.zeros(embedding_dim))
            missing_landmarks.append(landmark_id)
    
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    
    # Save embeddings
    print(f"Saving embeddings to {output_path}...")
    np.save(output_path, embeddings_array)
    
    # Save landmark ID mapping
    mapping_path = output_path.parent / 'landmarks_id_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump({'landmark_ids': landmark_ids}, f)
    
    # Print statistics
    print("\n" + "="*60)
    print("ENCODING COMPLETE")
    print("="*60)
    print(f"Total landmarks: {stats['processed']}")
    print(f"Landmarks with images: {stats['with_images']} ({stats['with_images']/stats['processed']*100:.1f}%)")
    print(f"Landmarks without images: {stats['without_images']} ({stats['without_images']/stats['processed']*100:.1f}%)")
    print(f"Total images encoded: {stats['total_images_encoded']}")
    print(f"Average images per landmark: {stats['total_images_encoded']/stats['with_images']:.1f}")
    print(f"\nEmbedding shape: {embeddings_array.shape}")
    print(f"Embedding dtype: {embeddings_array.dtype}")
    print(f"Output saved to: {output_path}")
    print(f"Mapping saved to: {mapping_path}")
    
    if missing_landmarks:
        print(f"\nWarning: {len(missing_landmarks)} landmarks have no images (using zero vectors)")
        print(f"First 10 missing: {missing_landmarks[:10]}")


if __name__ == '__main__':
    main()

