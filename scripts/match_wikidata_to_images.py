"""
Match WikiData landmarks (with coordinates) to Google Landmarks (with images).

This enables building CLIP embeddings for thousands of landmarks instead of just 78.

Strategy:
1. Load WikiData landmarks (15,795 with coordinates)
2. Load Google Landmarks train.csv (4.1M images, 203K landmarks)
3. Match by normalized name similarity
4. Update unified database with Google landmark_ids for matched WikiData entries
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re
from difflib import SequenceMatcher
import urllib.parse


def normalize_name(name):
    """Normalize landmark name for matching."""
    if not name:
        return ""
    
    # Extract from Wikimedia URLs if needed
    if 'wikimedia.org' in name or 'wikipedia.org' in name:
        if '/Category:' in name:
            name = name.split('/Category:')[-1]
        elif '/wiki/' in name:
            name = name.split('/wiki/')[-1]
    
    # URL decode and clean
    name = urllib.parse.unquote(name)
    name = name.replace('_', ' ')
    
    # Lowercase
    name = name.lower()
    
    # Remove common suffixes and prefixes
    patterns_to_remove = [
        r'\(.*?\)',  # Remove parenthetical content
        r'\[.*?\]',  # Remove bracketed content
    ]
    for pattern in patterns_to_remove:
        name = re.sub(pattern, '', name)
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    return name.strip()


def calculate_similarity(name1, name2):
    """Calculate similarity score between two names (0-1)."""
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Exact match
    if norm1 == norm2:
        return 1.0
    
    # One contains the other (good indicator)
    if norm1 in norm2 or norm2 in norm1:
        return 0.9
    
    # Use SequenceMatcher for fuzzy matching
    return SequenceMatcher(None, norm1, norm2).ratio()


def match_landmarks(wikidata_landmarks, google_metadata):
    """Match WikiData landmarks to Google landmarks by name.
    
    Returns: dict mapping wikidata landmark_id -> google landmark_id
    """
    print("\nBuilding Google landmarks lookup by name...")
    
    # Get unique Google landmarks with their names
    google_lookup = {}
    for _, row in tqdm(google_metadata.iterrows(), total=len(google_metadata), desc="Processing Google metadata"):
        google_id = row['landmark_id']
        if google_id not in google_lookup:
            google_lookup[google_id] = {
                'id': google_id,
                'name': row.get('landmark_name', ''),
                'normalized': normalize_name(row.get('landmark_name', '')),
                'image_count': 0
            }
        google_lookup[google_id]['image_count'] += 1
    
    print(f"Found {len(google_lookup):,} unique Google landmarks")
    
    # Build index by normalized name for fast lookup
    print("Building fast lookup index...")
    google_by_name = {}
    for google_id, data in google_lookup.items():
        norm_name = data['normalized']
        if norm_name:
            if norm_name not in google_by_name:
                google_by_name[norm_name] = []
            google_by_name[norm_name].append(google_id)
    
    # Match WikiData to Google
    print(f"\nMatching WikiData landmarks to Google by name...")
    print("Using fast exact + fuzzy matching strategy...")
    matches = {}
    match_stats = {
        'exact': 0,
        'high_confidence': 0,  # > 0.85
        'medium_confidence': 0,  # 0.70-0.85
        'no_match': 0
    }
    
    for lm in tqdm(wikidata_landmarks, desc="Matching"):
        wikidata_id = lm['landmark_id']
        wikidata_name = lm.get('name', '')
        
        if not wikidata_name:
            match_stats['no_match'] += 1
            continue
        
        norm_wikidata = normalize_name(wikidata_name)
        if not norm_wikidata:
            match_stats['no_match'] += 1
            continue
        
        # Fast exact match first
        if norm_wikidata in google_by_name:
            google_id = google_by_name[norm_wikidata][0]  # Take first match
            matches[wikidata_id] = {
                'google_landmark_id': google_id,
                'confidence': 1.0,
                'wikidata_name': wikidata_name,
                'google_name': google_lookup[google_id]['name'],
                'image_count': google_lookup[google_id]['image_count']
            }
            match_stats['exact'] += 1
            continue
        
        # Fuzzy match only for candidates with similar start (much faster)
        best_match = None
        best_score = 0.0
        
        # Only compare with Google landmarks that start with same letter
        first_word = norm_wikidata.split()[0] if norm_wikidata else ''
        if len(first_word) >= 3:
            prefix = first_word[:3]
            
            for google_id, google_data in google_lookup.items():
                google_norm = google_data['normalized']
                if not google_norm or not google_norm.startswith(prefix):
                    continue
                
                score = SequenceMatcher(None, norm_wikidata, google_norm).ratio()
                
                if score > best_score:
                    best_score = score
                    best_match = google_id
        
        # Accept matches with score >= 0.70
        if best_score >= 0.70:
            matches[wikidata_id] = {
                'google_landmark_id': best_match,
                'confidence': best_score,
                'wikidata_name': wikidata_name,
                'google_name': google_lookup[best_match]['name'],
                'image_count': google_lookup[best_match]['image_count']
            }
            
            if best_score >= 0.85:
                match_stats['high_confidence'] += 1
            else:
                match_stats['medium_confidence'] += 1
        else:
            match_stats['no_match'] += 1
    
    # Print statistics
    print("\n" + "="*60)
    print("MATCHING RESULTS")
    print("="*60)
    print(f"WikiData landmarks processed: {len(wikidata_landmarks):,}")
    print(f"Total matches found: {len(matches):,}")
    print(f"  Exact matches (1.0): {match_stats['exact']:,}")
    print(f"  High confidence (>0.85): {match_stats['high_confidence']:,}")
    print(f"  Medium confidence (0.70-0.85): {match_stats['medium_confidence']:,}")
    print(f"No matches: {match_stats['no_match']:,}")
    
    return matches, match_stats


def update_unified_database(matches):
    """Update landmarks_unified.json with matched Google IDs."""
    base_dir = Path(__file__).parent.parent
    unified_path = base_dir / 'data' / 'landmarks_unified.json'
    
    print(f"\nLoading {unified_path}...")
    with open(unified_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Updating {len(matches):,} WikiData landmarks with Google IDs...")
    updated = 0
    
    for lm in tqdm(data['landmarks'], desc="Updating"):
        wikidata_id = lm.get('landmark_id')
        
        # Check if this WikiData landmark has a match
        if wikidata_id in matches:
            match_info = matches[wikidata_id]
            lm['google_landmark_id'] = match_info['google_landmark_id']
            lm['image_count'] = match_info['image_count']
            lm['match_confidence'] = match_info['confidence']
            updated += 1
    
    # Update statistics
    with_both = sum(1 for lm in data['landmarks'] 
                    if 'latitude' in lm and 'google_landmark_id' in lm)
    
    data['statistics']['with_google_matches'] = len(matches)
    data['statistics']['landmarks_with_coords_and_images'] = with_both
    
    # Save
    print(f"Saving updated database...")
    with open(unified_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Updated {updated:,} landmarks")
    print(f"✓ {with_both:,} landmarks now have BOTH coordinates AND images")
    
    return with_both


def main():
    base_dir = Path(__file__).parent.parent
    unified_path = base_dir / 'data' / 'landmarks_unified.json'
    metadata_path = base_dir / 'data' / 'google_landmarks' / 'metadata' / 'train.csv'
    
    print("="*60)
    print("MATCH WIKIDATA LANDMARKS TO GOOGLE IMAGES")
    print("="*60)
    
    # Load unified database
    print(f"\nLoading {unified_path}...")
    with open(unified_path, 'r', encoding='utf-8') as f:
        unified_data = json.load(f)
    
    # Get WikiData landmarks (string IDs with coordinates)
    wikidata_landmarks = [
        lm for lm in unified_data['landmarks']
        if isinstance(lm.get('landmark_id'), str) and 'latitude' in lm
    ]
    
    print(f"WikiData landmarks with coordinates: {len(wikidata_landmarks):,}")
    
    # Load Google metadata
    print(f"\nLoading Google Landmarks metadata from {metadata_path}...")
    google_metadata = pd.read_csv(metadata_path)
    print(f"Loaded {len(google_metadata):,} image records")
    
    # Load landmark names
    names_path = base_dir / 'data' / 'google_landmarks' / 'metadata' / 'train_label_to_category.csv'
    if names_path.exists():
        print(f"Loading landmark names from {names_path}...")
        names_df = pd.read_csv(names_path)
        
        # Merge names into metadata
        google_metadata = google_metadata.merge(
            names_df[['landmark_id', 'category']].rename(columns={'category': 'landmark_name'}),
            on='landmark_id',
            how='left'
        )
        print(f"Merged landmark names")
    
    # Match landmarks
    matches, stats = match_landmarks(wikidata_landmarks, google_metadata)
    
    # Update database
    landmarks_with_both = update_unified_database(matches)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"1. Run build_visual_database.py to encode {landmarks_with_both:,} landmarks")
    print("2. This will give you visual similarity for thousands of landmarks!")
    print("3. Your itinerary planner will have rich data for recommendations")


if __name__ == '__main__':
    main()
