"""
Enrich landmark database with geocoordinates and metadata.
Fetches data from Wikipedia/Wikidata APIs.
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, Any, Optional, List


class LandmarkEnricher:
    """Fetch coordinates and metadata for landmarks."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LandmarkEnricher/1.0 (Educational Project)'
        })
    
    def search_wikidata(self, landmark_name: str) -> Optional[Dict[str, Any]]:
        """
        Search Wikidata for landmark information.
        
        Args:
            landmark_name: Name of the landmark
            
        Returns:
            Dictionary with coordinates and metadata, or None
        """
        try:
            # Search for entity
            search_url = "https://www.wikidata.org/w/api.php"
            search_params = {
                'action': 'wbsearchentities',
                'format': 'json',
                'language': 'en',
                'type': 'item',
                'search': landmark_name,
                'limit': 1
            }
            
            response = self.session.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('search'):
                print(f"  ⚠ No Wikidata entity found for: {landmark_name}")
                return None
            
            entity_id = data['search'][0]['id']
            
            # Get entity details
            entity_url = "https://www.wikidata.org/w/api.php"
            entity_params = {
                'action': 'wbgetentities',
                'format': 'json',
                'ids': entity_id,
                'props': 'claims|labels|descriptions'
            }
            
            response = self.session.get(entity_url, params=entity_params, timeout=10)
            response.raise_for_status()
            entity_data = response.json()
            
            entity = entity_data['entities'][entity_id]
            claims = entity.get('claims', {})
            
            # Extract coordinates (P625)
            coords = None
            if 'P625' in claims:
                coord_claim = claims['P625'][0]['mainsnak']['datavalue']['value']
                coords = {
                    'latitude': coord_claim['latitude'],
                    'longitude': coord_claim['longitude']
                }
            
            # Extract country (P17)
            country = None
            if 'P17' in claims:
                country_id = claims['P17'][0]['mainsnak']['datavalue']['value']['id']
                country = self._get_entity_label(country_id)
            
            # Extract description
            description = entity.get('descriptions', {}).get('en', {}).get('value', '')
            
            return {
                'coordinates': coords,
                'country': country,
                'description': description,
                'wikidata_id': entity_id
            }
            
        except Exception as e:
            print(f"  ⚠ Error fetching data for {landmark_name}: {e}")
            return None
    
    def _get_entity_label(self, entity_id: str) -> Optional[str]:
        """Get label for a Wikidata entity."""
        try:
            url = "https://www.wikidata.org/w/api.php"
            params = {
                'action': 'wbgetentities',
                'format': 'json',
                'ids': entity_id,
                'props': 'labels'
            }
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            return data['entities'][entity_id]['labels']['en']['value']
        except:
            return None
    
    def enrich_landmark(self, landmark: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single landmark with metadata.
        
        Args:
            landmark: Landmark dictionary with 'name' field
            
        Returns:
            Enriched landmark dictionary
        """
        print(f"Enriching: {landmark['name']}")
        
        # Search Wikidata
        metadata = self.search_wikidata(landmark['name'])
        
        if metadata and metadata.get('coordinates'):
            landmark.update({
                'latitude': metadata['coordinates']['latitude'],
                'longitude': metadata['coordinates']['longitude'],
                'country': metadata.get('country', 'Unknown'),
                'description': metadata.get('description', ''),
                'wikidata_id': metadata.get('wikidata_id', '')
            })
            print(f"  ✓ Found: {landmark['latitude']}, {landmark['longitude']}")
        else:
            # Fallback: manual coordinates for famous landmarks
            coords = self._get_fallback_coords(landmark['name'])
            if coords:
                landmark.update(coords)
                print(f"  ✓ Using fallback coordinates")
            else:
                print(f"  ✗ No coordinates found")
        
        time.sleep(0.5)  # Rate limiting
        return landmark
    
    def _get_fallback_coords(self, name: str) -> Optional[Dict[str, Any]]:
        """Fallback coordinates for famous landmarks."""
        fallbacks = {
            'Eiffel Tower': {'latitude': 48.8584, 'longitude': 2.2945, 'country': 'France'},
            'Golden Gate Bridge': {'latitude': 37.8199, 'longitude': -122.4783, 'country': 'USA'},
            'Taj Mahal': {'latitude': 27.1751, 'longitude': 78.0421, 'country': 'India'},
            'Great Wall of China': {'latitude': 40.4319, 'longitude': 116.5704, 'country': 'China'},
            'Sydney Opera House': {'latitude': -33.8568, 'longitude': 151.2153, 'country': 'Australia'},
            'Statue of Liberty': {'latitude': 40.6892, 'longitude': -74.0445, 'country': 'USA'},
            'Machu Picchu': {'latitude': -13.1631, 'longitude': -72.5450, 'country': 'Peru'},
            'Colosseum': {'latitude': 41.8902, 'longitude': 12.4922, 'country': 'Italy'},
            'Big Ben': {'latitude': 51.5007, 'longitude': -0.1246, 'country': 'UK'},
            'Niagara Falls': {'latitude': 43.0962, 'longitude': -79.0377, 'country': 'Canada/USA'},
            'Grand Canyon': {'latitude': 36.1069, 'longitude': -112.1129, 'country': 'USA'},
            'Louvre': {'latitude': 48.8606, 'longitude': 2.3376, 'country': 'France'},
            'Edinburgh Castle': {'latitude': 55.9486, 'longitude': -3.1999, 'country': 'Scotland'},
            'Tokyo Tower': {'latitude': 35.6586, 'longitude': 139.7454, 'country': 'Japan'},
            'Burj Khalifa': {'latitude': 25.1972, 'longitude': 55.2744, 'country': 'UAE'},
        }
        return fallbacks.get(name)


def main():
    """Enrich curated landmarks with metadata."""
    
    print("="*80)
    print("LANDMARK METADATA ENRICHMENT")
    print("="*80)
    
    # Load curated landmarks
    data_dir = Path(__file__).parent.parent / 'data'
    curated_path = data_dir / 'curated_landmarks.json'
    
    with open(curated_path, 'r') as f:
        data = json.load(f)
    
    landmarks = data['landmarks']
    print(f"\nLoaded {len(landmarks)} curated landmarks")
    
    # Enrich landmarks
    enricher = LandmarkEnricher()
    enriched_landmarks = []
    
    for landmark in landmarks:
        enriched = enricher.enrich_landmark(landmark)
        enriched_landmarks.append(enriched)
    
    # Save enriched data
    enriched_data = {
        'description': 'Curated landmarks with geocoordinates and metadata',
        'total': len(enriched_landmarks),
        'landmarks': enriched_landmarks
    }
    
    output_path = data_dir / 'landmarks_enriched.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)
    
    # Statistics
    with_coords = sum(1 for lm in enriched_landmarks if 'latitude' in lm)
    
    print("\n" + "="*80)
    print(f"✓ Enrichment complete!")
    print(f"  Total landmarks: {len(enriched_landmarks)}")
    print(f"  With coordinates: {with_coords} ({with_coords/len(enriched_landmarks)*100:.1f}%)")
    print(f"  Saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
