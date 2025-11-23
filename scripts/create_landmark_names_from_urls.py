"""
Create human-readable landmark names from Wikipedia URLs

Extracts landmark names from the Wikipedia URLs in the training data
to provide better display names than generic "landmark_12345".

Author: Evan Petersen
Date: November 2025
"""

import pandas as pd
import json
from pathlib import Path
from urllib.parse import unquote
import re


def extract_landmark_name_from_url(url):
    """Extract landmark name from Wikipedia/Wikimedia URL"""
    if pd.isna(url) or not isinstance(url, str):
        return None
    
    # Extract from Wikipedia/Wikimedia URL
    # Format: https://en.wikipedia.org/wiki/Landmark_Name or
    #         http://commons.wikimedia.org/wiki/Category:Landmark_Name
    if '/wiki/' in url:
        # Get the part after /wiki/
        name = url.split('/wiki/')[-1]
        # URL decode
        name = unquote(name)
        # Remove file extensions
        name = re.sub(r'\.(jpg|jpeg|png|JPG|JPEG|PNG)$', '', name)
        # Replace underscores with spaces
        name = name.replace('_', ' ')
        # Remove disambiguation
        name = re.sub(r'\s*\([^)]*\)', '', name)
        # Clean up
        name = name.strip()
        return name if name else None
    
    return None


def create_landmark_names_from_urls(csv_path: str, 
                                    model_path: str,
                                    output_path: str,
                                    category_csv: str,
                                    num_classes: int = 100):
    """
    Create landmark names from category file
    
    Args:
        csv_path: Path to train.csv
        model_path: Path to trained model (for landmark_map)
        output_path: Path to save updated landmark names JSON
        category_csv: Path to train_label_to_category.csv
        num_classes: Number of classes
    """
    import torch
    
    print(f"Loading category mapping from {category_csv}...")
    category_df = pd.read_csv(category_csv)
    
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loading model checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if 'landmark_map' in checkpoint and checkpoint['landmark_map'] is not None:
        old_landmark_map = checkpoint['landmark_map']
        # The landmark_map has landmark_ids as keys, need to convert to class_idx -> landmark_id
        # Sort by keys to get consistent class indices
        sorted_landmark_ids = [old_landmark_map[k] for k in sorted(old_landmark_map.keys())]
        landmark_map = {i: lid for i, lid in enumerate(sorted_landmark_ids)}
        print(f"Converted landmark_map: {len(landmark_map)} classes")
    else:
        # Get top N classes
        landmark_counts = df['landmark_id'].value_counts()
        top_landmark_ids = landmark_counts.head(num_classes).index.tolist()
        landmark_map = {i: lid for i, lid in enumerate(top_landmark_ids)}
    
    print(f"\nExtracting names for {len(landmark_map)} landmarks...")
    
    landmark_names = {}
    landmark_id_to_name = {}
    
    for class_idx, landmark_id in landmark_map.items():
        # Look up category URL for this landmark
        category_row = category_df[category_df['landmark_id'] == landmark_id]
        
        if len(category_row) > 0:
            category_url = category_row.iloc[0]['category']
            # Extract name from category URL
            # Format: http://commons.wikimedia.org/wiki/Category:Landmark_Name
            name = extract_landmark_name_from_url(category_url)
            
            if name and len(name) < 100:
                # Remove "Category:" prefix if present
                if name.startswith('Category:'):
                    name = name[9:].strip()
                
                if name:
                    landmark_names[int(class_idx)] = name
                    landmark_id_to_name[landmark_id] = name
                else:
                    landmark_names[int(class_idx)] = f"Landmark {landmark_id}"
            else:
                landmark_names[int(class_idx)] = f"Landmark {landmark_id}"
        else:
            landmark_names[int(class_idx)] = f"Landmark {landmark_id}"
    
    # Save to JSON
    output = {
        'num_classes': len(landmark_map),
        'landmark_map': {int(k): int(v) for k, v in landmark_map.items()},
        'landmark_names': landmark_names,
        'landmark_id_to_name': landmark_id_to_name
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved landmark names to: {output_path}")
    print(f"\nSample landmarks:")
    sorted_keys = sorted(landmark_names.keys())
    for idx in sorted_keys[:20]:
        print(f"  Class {idx}: {landmark_names[idx]}")
    
    # Statistics
    generic_names = sum(1 for name in landmark_names.values() if name.startswith('Landmark '))
    real_names = len(landmark_names) - generic_names
    print(f"\nExtracted {real_names}/{len(landmark_names)} real names ({real_names/len(landmark_names)*100:.1f}%)")


if __name__ == '__main__':
    csv_path = "data/google_landmarks/metadata/train.csv"
    category_csv = "data/google_landmarks/metadata/train_label_to_category.csv"
    model_path = "data/checkpoints/landmark_detector_100classes_best.pth"
    output_path = "data/checkpoints/landmark_names_100classes.json"
    
    create_landmark_names_from_urls(csv_path, model_path, output_path, category_csv)
