"""
Generate CLIP text embeddings for all landmarks using HuggingFace Space API.
Replaces old MiniLM embeddings with CLIP ViT-B/32 embeddings (512-dim).
"""

import json
import numpy as np
import requests
from tqdm import tqdm
import time

# Configuration
HF_SPACE_URL = "https://evanpetersen919-cv-location-classifier.hf.space"
LANDMARKS_FILE = "data/landmarks_unified.json"
OUTPUT_EMBEDDINGS = "data/embeddings/landmark_text_embeddings.npy"
OUTPUT_METADATA = "data/embeddings/landmark_text_metadata.json"
BATCH_SIZE = 50  # Process in batches to show progress
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds


def get_clip_text_embedding(text: str) -> np.ndarray:
    """Get CLIP text embedding from HuggingFace Space."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.post(
                f"{HF_SPACE_URL}/clip/text",
                data={"text": text},  # Use form data, not JSON
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            return np.array(result["embedding"], dtype=np.float32)
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                print(f"  Retry {attempt + 1}/{RETRY_ATTEMPTS} for '{text[:50]}...' ({str(e)})")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  FAILED after {RETRY_ATTEMPTS} attempts: '{text[:50]}...'")
                return None


def load_landmarks():
    """Load landmark data from unified JSON."""
    print(f"Loading landmarks from {LANDMARKS_FILE}...")
    with open(LANDMARKS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    landmarks = []
    for landmark in data["landmarks"]:
        # Clean up landmark names (remove URLs, fix formatting)
        name = landmark.get("name", "")
        if name.startswith("http://"):
            # Skip URL-based names
            continue
        
        landmarks.append({
            "landmark_id": landmark["landmark_id"],
            "name": name
        })
    
    print(f"Found {len(landmarks)} valid landmarks")
    return landmarks


def generate_embeddings():
    """Generate CLIP embeddings for all landmarks."""
    landmarks = load_landmarks()
    
    embeddings = []
    landmark_ids = []
    failed_count = 0
    
    print(f"\nGenerating CLIP embeddings for {len(landmarks)} landmarks...")
    print(f"Using HuggingFace Space: {HF_SPACE_URL}")
    print(f"This will take approximately {len(landmarks) * 0.5 / 60:.1f} minutes\n")
    
    # Process with progress bar
    for landmark in tqdm(landmarks, desc="Processing landmarks"):
        landmark_id = landmark["landmark_id"]
        name = landmark["name"]
        
        # Get embedding
        embedding = get_clip_text_embedding(name)
        
        if embedding is not None:
            embeddings.append(embedding)
            landmark_ids.append(landmark_id)
        else:
            failed_count += 1
        
        # Small delay to avoid overwhelming the Space
        time.sleep(0.1)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    print(f"\n✅ Successfully generated {len(embeddings)} embeddings")
    print(f"❌ Failed: {failed_count}")
    print(f"📊 Embedding shape: {embeddings_array.shape}")
    print(f"💾 Size: {embeddings_array.nbytes / 1024 / 1024:.2f} MB")
    
    return embeddings_array, landmark_ids


def save_embeddings(embeddings, landmark_ids):
    """Save embeddings and metadata to disk."""
    print(f"\nSaving embeddings to {OUTPUT_EMBEDDINGS}...")
    np.save(OUTPUT_EMBEDDINGS, embeddings)
    
    metadata = {
        "landmark_ids": landmark_ids,
        "model": "CLIP ViT-B/32",
        "dimensions": embeddings.shape[1],
        "total_landmarks": len(landmark_ids),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(f"Saving metadata to {OUTPUT_METADATA}...")
    with open(OUTPUT_METADATA, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ CLIP embeddings generated successfully!")
    print(f"   - Model: {metadata['model']}")
    print(f"   - Dimensions: {metadata['dimensions']}")
    print(f"   - Total landmarks: {metadata['total_landmarks']}")


if __name__ == "__main__":
    try:
        # Generate embeddings
        embeddings, landmark_ids = generate_embeddings()
        
        # Save to disk
        save_embeddings(embeddings, landmark_ids)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise
