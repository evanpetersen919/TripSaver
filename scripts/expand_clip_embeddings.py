"""
Expand CLIP Embedding Coverage
================================

Generate CLIP visual embeddings for all 15,873 landmarks in the database.
Currently only 4,248 have embeddings. This script will:

1. Load landmarks_unified.json (15,873 landmarks)
2. Identify which landmarks are missing CLIP embeddings
3. Search for images using Wikidata/Wikipedia APIs
4. Generate CLIP embeddings for found images
5. Update landmarks_clip_embeddings.npy and landmarks_id_mapping.json

Author: Evan Petersen
Date: November 2025
"""

import json
import numpy as np
from pathlib import Path
import sys
import time
from typing import List, Dict, Any, Optional
import requests
from PIL import Image
from io import BytesIO
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.clip_embedder import ClipEmbedder


class CLIPExpander:
    """Expand CLIP embeddings to all landmarks in database."""
    
    def __init__(self, data_dir: str = None):
        """Initialize expander.
        
        Args:
            data_dir: Path to data directory
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / 'data'
        self.data_dir = Path(data_dir)
        
        # Load existing data
        print("Loading landmark database...")
        with open(self.data_dir / 'landmarks_unified.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.landmarks = data['landmarks']
        print(f"✓ Loaded {len(self.landmarks)} landmarks")
        
        # Load existing CLIP embeddings
        print("Loading existing CLIP embeddings...")
        if (self.data_dir / 'landmarks_clip_embeddings.npy').exists():
            self.existing_embeddings = np.load(self.data_dir / 'landmarks_clip_embeddings.npy')
            with open(self.data_dir / 'landmarks_id_mapping.json', 'r') as f:
                mapping = json.load(f)
                self.existing_ids = set(mapping['landmark_ids'])
            print(f"✓ Found {len(self.existing_ids)} existing embeddings")
        else:
            self.existing_embeddings = None
            self.existing_ids = set()
            print("No existing embeddings found")
        
        # Initialize CLIP model
        print("Initializing CLIP model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip = ClipEmbedder(device=device)
        print(f"✓ CLIP ready on {device}")
        
        # Track progress
        self.new_embeddings = []
        self.new_ids = []
        self.failed_landmarks = []
    
    
    def find_missing_landmarks(self) -> List[Dict[str, Any]]:
        """Find landmarks without CLIP embeddings.
        
        Returns:
            List of landmark dicts without embeddings
        """
        missing = []
        for landmark in self.landmarks:
            lm_id = landmark['landmark_id']
            if lm_id not in self.existing_ids:
                missing.append(landmark)
        
        print(f"\n📊 Missing embeddings for {len(missing)}/{len(self.landmarks)} landmarks")
        return missing
    
    
    def search_wikipedia_image(self, landmark_name: str, wikidata_id: str = None) -> Optional[str]:
        """Search for landmark image on Wikipedia/Wikidata.
        
        Args:
            landmark_name: Name of landmark
            wikidata_id: Optional Wikidata ID (e.g., 'Q243')
            
        Returns:
            Image URL if found, else None
        """
        try:
            # Try Wikidata first if ID provided
            if wikidata_id:
                url = f"https://www.wikidata.org/w/api.php"
                params = {
                    'action': 'wbgetclaims',
                    'entity': wikidata_id,
                    'property': 'P18',  # Image property
                    'format': 'json'
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'claims' in data and 'P18' in data['claims']:
                        filename = data['claims']['P18'][0]['mainsnak']['datavalue']['value']
                        # Convert to Commons URL
                        import hashlib
                        md5 = hashlib.md5(filename.replace(' ', '_').encode()).hexdigest()
                        return f"https://upload.wikimedia.org/wikipedia/commons/{md5[0]}/{md5[:2]}/{filename.replace(' ', '_')}"
            
            # Fallback: Search Wikipedia
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'titles': landmark_name,
                'prop': 'pageimages',
                'piprop': 'original',
                'pilimit': 1
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pages = data['query']['pages']
                for page_id, page_data in pages.items():
                    if 'original' in page_data:
                        return page_data['original']['source']
            
            return None
            
        except Exception as e:
            print(f"   Error searching Wikipedia for {landmark_name}: {e}")
            return None
    
    
    def download_and_embed_image(self, image_url: str) -> Optional[np.ndarray]:
        """Download image and generate CLIP embedding.
        
        Args:
            image_url: URL to image
            
        Returns:
            CLIP embedding (512-dim) or None if failed
        """
        try:
            # Download image
            response = requests.get(image_url, timeout=15)
            if response.status_code != 200:
                return None
            
            # Load image
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Generate embedding
            embedding = self.clip.encode_image(image)
            
            return embedding
            
        except Exception as e:
            print(f"   Error processing image: {e}")
            return None
    
    
    def process_landmarks(self, max_landmarks: int = None, batch_size: int = 100):
        """Process missing landmarks and generate embeddings.
        
        Args:
            max_landmarks: Maximum number to process (None for all)
            batch_size: Save progress every N landmarks
        """
        missing = self.find_missing_landmarks()
        
        if max_landmarks:
            missing = missing[:max_landmarks]
            print(f"Processing first {max_landmarks} missing landmarks...")
        else:
            print(f"Processing all {len(missing)} missing landmarks...")
        
        print("\n" + "=" * 80)
        print("STARTING CLIP EMBEDDING GENERATION")
        print("=" * 80)
        
        start_time = time.time()
        
        for i, landmark in enumerate(missing, 1):
            lm_id = landmark['landmark_id']
            lm_name = landmark['name']
            wikidata_id = landmark.get('wikidata_id')
            
            print(f"\n[{i}/{len(missing)}] Processing: {lm_name} (ID: {lm_id})")
            
            # Search for image
            image_url = self.search_wikipedia_image(lm_name, wikidata_id)
            
            if image_url:
                print(f"   ✓ Found image: {image_url[:80]}...")
                
                # Generate embedding
                embedding = self.download_and_embed_image(image_url)
                
                if embedding is not None:
                    self.new_embeddings.append(embedding)
                    self.new_ids.append(lm_id)
                    print(f"   ✓ Generated embedding ({embedding.shape})")
                else:
                    self.failed_landmarks.append({
                        'id': lm_id,
                        'name': lm_name,
                        'reason': 'Failed to download/embed image'
                    })
                    print(f"   ✗ Failed to generate embedding")
            else:
                self.failed_landmarks.append({
                    'id': lm_id,
                    'name': lm_name,
                    'reason': 'No image found'
                })
                print(f"   ✗ No image found")
            
            # Save progress every batch_size landmarks
            if i % batch_size == 0:
                self.save_progress()
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (len(missing) - i) * avg_time
                success_rate = len(self.new_ids) / i * 100 if i > 0 else 0
                print(f"\n📊 Progress: {i}/{len(missing)} ({i/len(missing)*100:.1f}%)")
                print(f"⏱️  Elapsed: {elapsed/60:.1f}min | Remaining: ~{remaining/60:.1f}min")
                print(f"✅ Success rate: {success_rate:.1f}% ({len(self.new_ids)} embeddings)")
                
            # Small delay to avoid hammering APIs
            time.sleep(0.5)
        
        # Final save
        self.save_progress()
        
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"New embeddings: {len(self.new_ids)}")
        print(f"Failed: {len(self.failed_landmarks)}")
        print(f"Success rate: {len(self.new_ids)/(len(self.new_ids)+len(self.failed_landmarks))*100:.1f}%")
    
    
    def save_progress(self):
        """Save current progress to disk."""
        if not self.new_embeddings:
            print("   No new embeddings to save")
            return
        
        print(f"\n💾 Saving {len(self.new_embeddings)} new embeddings...")
        
        # Combine with existing
        if self.existing_embeddings is not None:
            all_embeddings = np.vstack([
                self.existing_embeddings,
                np.array(self.new_embeddings)
            ])
            all_ids = list(self.existing_ids) + self.new_ids
        else:
            all_embeddings = np.array(self.new_embeddings)
            all_ids = self.new_ids
        
        # Save embeddings
        np.save(self.data_dir / 'landmarks_clip_embeddings.npy', all_embeddings)
        print(f"   ✓ Saved embeddings: {all_embeddings.shape}")
        
        # Save mapping
        mapping = {'landmark_ids': all_ids}
        with open(self.data_dir / 'landmarks_id_mapping.json', 'w') as f:
            json.dump(mapping, f)
        print(f"   ✓ Saved mapping: {len(all_ids)} IDs")
        
        # Save failed landmarks log
        if self.failed_landmarks:
            with open(self.data_dir / 'clip_failed_landmarks.json', 'w') as f:
                json.dump({
                    'failed_count': len(self.failed_landmarks),
                    'landmarks': self.failed_landmarks
                }, f, indent=2)
            print(f"   ✓ Saved failed landmarks log: {len(self.failed_landmarks)} entries")
        
        # Update existing for next save
        self.existing_embeddings = all_embeddings
        self.existing_ids = set(all_ids)
        self.new_embeddings = []
        self.new_ids = []


def main():
    """Main execution."""
    print("=" * 80)
    print("CLIP EMBEDDING EXPANDER")
    print("=" * 80)
    print()
    
    # Initialize expander
    expander = CLIPExpander()
    
    # Process ALL landmarks overnight
    print("\n🎯 Processing ALL missing landmarks (overnight run)...")
    print("Progress will be saved every 50 landmarks")
    print("You can stop and resume anytime - progress is saved automatically")
    print()
    
    expander.process_landmarks(
        max_landmarks=None,  # Process ALL missing landmarks
        batch_size=50  # Save progress every 50 (more frequent for safety)
    )
    
    print("\n✅ DONE! All landmarks processed.")


if __name__ == '__main__':
    main()
