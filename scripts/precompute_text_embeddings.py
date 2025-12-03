"""
Pre-compute text embeddings for all landmarks for Lambda deployment.
Uses sentence-transformers locally, saves embeddings as numpy arrays.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    landmarks_path = base_dir / 'data' / 'landmarks_unified.json'
    embeddings_output = base_dir / 'data' / 'embeddings' / 'landmark_text_embeddings.npy'
    metadata_output = base_dir / 'data' / 'embeddings' / 'landmark_text_metadata.json'
    
    print("Loading landmarks...")
    with open(landmarks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        landmarks = [lm for lm in data['landmarks'] if 'latitude' in lm]
    
    print(f"Found {len(landmarks)} landmarks with coordinates")
    
    # Use lightweight but effective model (384 dimensions, ~80MB)
    print("\nLoading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare texts for embedding
    print("\nPreparing landmark descriptions...")
    texts = []
    landmark_ids = []
    
    for lm in landmarks:
        # Combine name + description for better semantic matching
        desc = lm.get('description', '')
        text = f"{lm['name']}. {desc}" if desc else lm['name']
        texts.append(text)
        landmark_ids.append(lm['landmark_id'])
    
    # Batch encode (much faster)
    print(f"\nComputing embeddings for {len(texts)} landmarks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True  # Pre-normalize for faster cosine similarity
    )
    
    # Save embeddings
    print(f"\nSaving embeddings to {embeddings_output}...")
    embeddings_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_output, embeddings)
    
    # Save metadata (landmark IDs in same order as embeddings)
    metadata = {
        'landmark_ids': landmark_ids,
        'model': 'all-MiniLM-L6-v2',
        'dimensions': embeddings.shape[1],
        'count': len(embeddings),
        'normalized': True
    }
    
    print(f"Saving metadata to {metadata_output}...")
    with open(metadata_output, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Done!")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  File size: {embeddings_output.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\nYou can now deploy these embeddings with your Lambda function.")

if __name__ == '__main__':
    main()
