"""
Geocode landmarks using WikiData API.

This script enriches the expanded landmark database with coordinates
by querying WikiData based on landmark names.

WARNING: This will make 203K API requests and may take 4-8 hours.
"""

import json
import requests
import time
from pathlib import Path
from tqdm import tqdm
import argparse
import urllib.parse

def clean_landmark_name(name):
    """Extract actual place name from Wikimedia URLs."""
    if not name:
        return ""
    
    # If it's a Wikimedia URL, extract the category name
    if 'wikimedia.org' in name or 'wikipedia.org' in name:
        if '/Category:' in name:
            name = name.split('/Category:')[-1]
        elif '/wiki/' in name:
            name = name.split('/wiki/')[-1]
    
    # Clean up: decode URL encoding, replace underscores
    name = urllib.parse.unquote(name)
    name = name.replace('_', ' ')
    
    return name.strip()


def search_wikidata(landmark_name, max_results=3):
    """
    Search WikiData for a landmark by name.
    
    Returns list of potential matches with coordinates.
    """
    # Clean the landmark name first
    clean_name = clean_landmark_name(landmark_name)
    if not clean_name or len(clean_name) < 2:
        return []
    
    try:
        # Search for the landmark
        search_url = "https://www.wikidata.org/w/api.php"
        search_params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': 'en',
            'type': 'item',
            'search': clean_name,
            'limit': max_results
        }
        
        response = requests.get(search_url, params=search_params, timeout=10)
        response.raise_for_status()
        search_results = response.json()
        
        if 'search' not in search_results or not search_results['search']:
            return []
        
        # For each result, get coordinates
        matches = []
        for result in search_results['search'][:max_results]:
            entity_id = result['id']
            
            # Get entity data including coordinates
            entity_url = "https://www.wikidata.org/w/api.php"
            entity_params = {
                'action': 'wbgetentities',
                'ids': entity_id,
                'format': 'json',
                'props': 'claims|labels|descriptions'
            }
            
            entity_response = requests.get(entity_url, params=entity_params, timeout=10)
            entity_response.raise_for_status()
            entity_data = entity_response.json()
            
            if 'entities' not in entity_data:
                continue
            
            entity = entity_data['entities'].get(entity_id, {})
            claims = entity.get('claims', {})
            
            # Get coordinates (P625)
            if 'P625' in claims:
                coord_claim = claims['P625'][0]
                value = coord_claim.get('mainsnak', {}).get('datavalue', {}).get('value', {})
                
                if 'latitude' in value and 'longitude' in value:
                    match = {
                        'wikidata_id': entity_id,
                        'latitude': value['latitude'],
                        'longitude': value['longitude'],
                        'label': entity.get('labels', {}).get('en', {}).get('value', ''),
                        'description': entity.get('descriptions', {}).get('en', {}).get('value', '')
                    }
                    
                    # Get country (P17)
                    if 'P17' in claims:
                        country_id = claims['P17'][0].get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id')
                        if country_id:
                            # Get country name
                            country_response = requests.get(entity_url, params={
                                'action': 'wbgetentities',
                                'ids': country_id,
                                'format': 'json',
                                'props': 'labels'
                            }, timeout=10)
                            if country_response.ok:
                                country_data = country_response.json()
                                country_name = country_data.get('entities', {}).get(country_id, {}).get('labels', {}).get('en', {}).get('value')
                                if country_name:
                                    match['country'] = country_name
                    
                    matches.append(match)
        
        return matches
        
    except Exception as e:
        return []


def geocode_landmarks(input_path, output_path, start_from=0, batch_size=100, delay=0.1):
    """
    Geocode all landmarks in the database.
    
    Args:
        input_path: Path to landmarks_expanded_full.json
        output_path: Path to save geocoded results
        start_from: Index to start from (for resuming)
        batch_size: Save progress every N landmarks
        delay: Delay between API requests (seconds)
    """
    # Load landmarks
    print(f"Loading landmarks from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    landmarks = data['landmarks']
    total = len(landmarks)
    
    print(f"Total landmarks: {total:,}")
    print(f"Starting from index: {start_from}")
    print(f"Delay between requests: {delay}s")
    print(f"Estimated time: {(total - start_from) * delay / 3600:.1f} hours")
    
    # Statistics
    stats = {
        'processed': start_from,
        'geocoded': sum(1 for lm in landmarks[:start_from] if 'latitude' in lm),
        'failed': 0,
        'skipped': 0
    }
    
    # Process landmarks
    try:
        for i in tqdm(range(start_from, total), desc="Geocoding", initial=start_from, total=total):
            landmark = landmarks[i]
            
            # Skip if already has coordinates
            if 'latitude' in landmark:
                stats['skipped'] += 1
                stats['processed'] += 1
                continue
            
            # Clean landmark name for search
            name = landmark['name']
            # Remove URL parts if present
            if 'http://' in name or 'wiki' in name.lower():
                # Extract readable part from URL
                parts = name.split('/')
                if len(parts) > 0:
                    name = parts[-1].replace('_', ' ')
            
            # Search WikiData
            matches = search_wikidata(name)
            
            if matches:
                # Use best match (first result)
                best_match = matches[0]
                landmark['latitude'] = best_match['latitude']
                landmark['longitude'] = best_match['longitude']
                landmark['wikidata_id'] = best_match['wikidata_id']
                
                if 'description' in best_match and best_match['description']:
                    landmark['description'] = best_match['description']
                if 'country' in best_match:
                    landmark['country'] = best_match['country']
                
                stats['geocoded'] += 1
            else:
                stats['failed'] += 1
            
            stats['processed'] += 1
            
            # Rate limiting
            time.sleep(delay)
            
            # Save progress periodically
            if stats['processed'] % batch_size == 0:
                print(f"\n[Checkpoint] Saving progress at {stats['processed']:,}/{total:,}...")
                data['statistics'] = {
                    'total_landmarks': total,
                    'geocoded': stats['geocoded'],
                    'processed': stats['processed'],
                    'failed': stats['failed'],
                    'skipped': stats['skipped'],
                    'progress_percent': (stats['processed'] / total) * 100
                }
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"Progress: {stats['geocoded']:,} geocoded, {stats['failed']:,} failed")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving progress...")
    
    # Final save
    print(f"\nSaving final results to {output_path}...")
    data['statistics'] = {
        'total_landmarks': total,
        'geocoded': stats['geocoded'],
        'processed': stats['processed'],
        'failed': stats['failed'],
        'skipped': stats['skipped'],
        'progress_percent': (stats['processed'] / total) * 100
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Print final statistics
    print("\n" + "="*60)
    print("GEOCODING COMPLETE")
    print("="*60)
    print(f"Total processed: {stats['processed']:,}/{total:,}")
    print(f"Successfully geocoded: {stats['geocoded']:,} ({stats['geocoded']/total*100:.1f}%)")
    print(f"Already had coords: {stats['skipped']:,}")
    print(f"Failed to geocode: {stats['failed']:,}")
    print(f"\nOutput: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Geocode landmarks using WikiData')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Index to start from (for resuming)')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between API requests in seconds (default: 0.1)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Save progress every N landmarks (default: 100)')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / 'data' / 'landmarks_unified.json'
    output_path = base_dir / 'data' / 'landmarks_unified.json'
    
    if not input_path.exists():
        print(f"Error: {input_path} not found!")
        print("Run expand_landmarks_database.py first.")
        return
    
    print("="*60)
    print("LANDMARK GEOCODING")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"\n⚠️  WARNING: This will take several hours to complete.")
    print("You can press Ctrl+C to stop and resume later with --start-from")
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    geocode_landmarks(
        input_path,
        output_path,
        start_from=args.start_from,
        batch_size=args.batch_size,
        delay=args.delay
    )


if __name__ == '__main__':
    main()
