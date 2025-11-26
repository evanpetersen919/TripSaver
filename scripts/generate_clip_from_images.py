"""
Generate CLIP Embeddings from Google Landmarks Images
======================================================

Much faster than downloading from Wikipedia - uses images you already have!

Author: Evan Petersen
Date: November 2025
"""

import json
import numpy as np
from pathlib import Path
import sys
import time
from typing import List, Dict
from PIL import Image
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.clip_embedder import ClipEmbedder


def main():
    """Generate CLIP embeddings from downloaded Google Landmarks images."""
    
    print("=" * 80)
    print("CLIP EMBEDDING GENERATOR (from local images)")
    print("=" * 80)
    print()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    images_dir = data_dir / 'google_landmarks' / 'train'
    
    # Load existing embeddings
    print("Loading existing CLIP embeddings...")
    if (data_dir / 'landmarks_clip_embeddings.npy').exists():
        existing_embeddings = np.load(data_dir / 'landmarks_clip_embeddings.npy')
        with open(data_dir / 'landmarks_id_mapping.json', 'r') as f:
            mapping = json.load(f)
            existing_ids = set(mapping['landmark_ids'])
        print(f"✓ Found {len(existing_ids)} existing embeddings")
    else:
        existing_embeddings = None
        existing_ids = set()
        print("No existing embeddings found - starting fresh")
    
    # Initialize CLIP on GPU
    print("\nInitializing CLIP model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip = ClipEmbedder(device=device)
    print(f"✓ CLIP ready on {device}")
    
    # Find all landmark ID folders (0-9, a-f)
    print("\nScanning for landmark images...")
    landmark_folders = []
    for folder in sorted(images_dir.iterdir()):
        if folder.is_dir() and folder.name in '0123456789abcdef':
            # Each folder contains subfolders with landmark IDs
            for landmark_folder in folder.iterdir():
                if landmark_folder.is_dir():
                    try:
                        landmark_id = int(landmark_folder.name)
                        if landmark_id not in existing_ids:
                            # Find first image in this landmark folder
                            images = list(landmark_folder.glob('*.jpg'))
                            if images:
                                landmark_folders.append((landmark_id, images[0]))
                    except ValueError:
                        continue
    
    print(f"✓ Found {len(landmark_folders)} landmarks with images (missing embeddings)")
    
    if not landmark_folders:
        print("\n✅ All landmarks already have CLIP embeddings!")
        return
    
    # Process in batches
    print("\n" + "=" * 80)
    print("GENERATING CLIP EMBEDDINGS")
    print("=" * 80)
    
    new_embeddings = []
    new_ids = []
    failed = []
    
    batch_size = 50  # Save progress every 50
    
    for i, (landmark_id, image_path) in enumerate(tqdm(landmark_folders, desc="Processing"), 1):
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Generate embedding
            embedding = clip.encode_image(image)
            
            new_embeddings.append(embedding)
            new_ids.append(landmark_id)
            
        except Exception as e:
            failed.append({'id': landmark_id, 'path': str(image_path), 'error': str(e)})
        
        # Save progress every batch_size
        if i % batch_size == 0 or i == len(landmark_folders):
            # Combine with existing
            if existing_embeddings is not None:
                all_embeddings = np.vstack([existing_embeddings, np.array(new_embeddings)])
                all_ids = list(existing_ids) + new_ids
            else:
                all_embeddings = np.array(new_embeddings)
                all_ids = new_ids
            
            # Save
            np.save(data_dir / 'landmarks_clip_embeddings.npy', all_embeddings)
            with open(data_dir / 'landmarks_id_mapping.json', 'w') as f:
                json.dump({'landmark_ids': all_ids}, f)
            
            print(f"\n💾 Saved progress: {len(all_ids)} total embeddings")
            
            # Update for next batch
            existing_embeddings = all_embeddings
            existing_ids = set(all_ids)
            new_embeddings = []
            new_ids = []
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"✅ Generated: {len(landmark_folders) - len(failed)} embeddings")
    print(f"❌ Failed: {len(failed)} landmarks")
    print(f"📊 Total coverage: {len(all_ids)} landmarks")
    
    if failed:
        with open(data_dir / 'clip_failed_images.json', 'w') as f:
            json.dump({'failed_count': len(failed), 'landmarks': failed}, f, indent=2)
        print(f"📝 Saved failure log to clip_failed_images.json")


if __name__ == '__main__':
    main()
