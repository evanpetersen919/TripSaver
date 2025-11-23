"""
Download landmarks from Wikidata using SPARQL queries.
Fetches tourist attractions, monuments, and landmarks worldwide.
"""

import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


class WikidataLandmarkDownloader:
    """Download landmark data from Wikidata SPARQL endpoint."""
    
    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LandmarkDownloader/1.0 (Educational Project)'
        })
    
    def query_sparql(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute SPARQL query against Wikidata.
        
        Args:
            query: SPARQL query string
            
        Returns:
            List of result dictionaries
        """
        params = {
            'query': query,
            'format': 'json'
        }
        
        print("Executing SPARQL query...")
        response = self.session.get(
            self.SPARQL_ENDPOINT,
            params=params,
            timeout=60
        )
        response.raise_for_status()
        
        data = response.json()
        return data['results']['bindings']
    
    def download_tourist_attractions(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """
        Download tourist attractions from Wikidata.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of landmark dictionaries
        """
        print(f"\nQuerying Wikidata for top {limit} tourist attractions...")
        
        query = f"""
        SELECT DISTINCT ?place ?placeLabel ?coord ?countryLabel ?description ?image
        WHERE {{
          ?place wdt:P31/wdt:P279* wd:Q570116.  # tourist attraction or subclass
          ?place wdt:P625 ?coord.                # has coordinates
          OPTIONAL {{ ?place wdt:P17 ?country }}  # country
          OPTIONAL {{ ?place wdt:P18 ?image }}    # image
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en". 
            ?place rdfs:label ?placeLabel.
            ?country rdfs:label ?countryLabel.
            ?place schema:description ?description.
          }}
        }}
        LIMIT {limit}
        """
        
        results = self.query_sparql(query)
        
        print(f"✓ Retrieved {len(results)} tourist attractions")
        return self._parse_results(results)
    
    def download_monuments(self, limit: int = 5000) -> List[Dict[str, Any]]:
        """Download monuments and memorials."""
        print(f"\nQuerying Wikidata for {limit} monuments...")
        
        query = f"""
        SELECT DISTINCT ?place ?placeLabel ?coord ?countryLabel ?description
        WHERE {{
          ?place wdt:P31/wdt:P279* wd:Q4989906.  # monument or subclass
          ?place wdt:P625 ?coord.
          OPTIONAL {{ ?place wdt:P17 ?country }}
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en". 
            ?place rdfs:label ?placeLabel.
            ?country rdfs:label ?countryLabel.
            ?place schema:description ?description.
          }}
        }}
        LIMIT {limit}
        """
        
        results = self.query_sparql(query)
        print(f"✓ Retrieved {len(results)} monuments")
        return self._parse_results(results)
    
    def download_natural_landmarks(self, limit: int = 3000) -> List[Dict[str, Any]]:
        """Download natural landmarks (mountains, waterfalls, etc.)."""
        print(f"\nQuerying Wikidata for {limit} natural landmarks...")
        
        query = f"""
        SELECT DISTINCT ?place ?placeLabel ?coord ?countryLabel ?description
        WHERE {{
          {{
            ?place wdt:P31 wd:Q8502.  # mountain
          }} UNION {{
            ?place wdt:P31 wd:Q34038.  # waterfall
          }} UNION {{
            ?place wdt:P31 wd:Q35666.  # national park
          }}
          ?place wdt:P625 ?coord.
          OPTIONAL {{ ?place wdt:P17 ?country }}
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en". 
            ?place rdfs:label ?placeLabel.
            ?country rdfs:label ?countryLabel.
            ?place schema:description ?description.
          }}
        }}
        LIMIT {limit}
        """
        
        results = self.query_sparql(query)
        print(f"✓ Retrieved {len(results)} natural landmarks")
        return self._parse_results(results)
    
    def _parse_results(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """Parse SPARQL results into our landmark format."""
        landmarks = []
        
        for result in results:
            try:
                # Extract Wikidata ID from URI
                place_uri = result['place']['value']
                wikidata_id = place_uri.split('/')[-1]
                
                # Parse coordinates (format: "Point(lon lat)")
                coord_str = result['coord']['value']
                coord_parts = coord_str.replace('Point(', '').replace(')', '').split()
                longitude = float(coord_parts[0])
                latitude = float(coord_parts[1])
                
                # Extract other fields
                name = result.get('placeLabel', {}).get('value', 'Unknown')
                country = result.get('countryLabel', {}).get('value', 'Unknown')
                description = result.get('description', {}).get('value', '')
                
                landmark = {
                    'name': name,
                    'landmark_id': wikidata_id,
                    'latitude': latitude,
                    'longitude': longitude,
                    'country': country,
                    'description': description,
                    'wikidata_id': wikidata_id,
                    'image_count': 0,  # Placeholder
                    'category_url': f"https://www.wikidata.org/wiki/{wikidata_id}"
                }
                
                landmarks.append(landmark)
                
            except (KeyError, ValueError, IndexError) as e:
                continue  # Skip malformed entries
        
        return landmarks
    
    def download_all_categories(self) -> List[Dict[str, Any]]:
        """Download landmarks from all categories."""
        print("="*80)
        print("DOWNLOADING LANDMARKS FROM WIKIDATA")
        print("="*80)
        
        all_landmarks = []
        
        # Download different categories
        categories = [
            ("Tourist Attractions", self.download_tourist_attractions, 10000),
            ("Monuments", self.download_monuments, 5000),
            ("Natural Landmarks", self.download_natural_landmarks, 3000),
        ]
        
        for category_name, download_func, limit in categories:
            try:
                landmarks = download_func(limit)
                all_landmarks.extend(landmarks)
                print(f"  Added {len(landmarks)} from {category_name}")
                time.sleep(2)  # Rate limiting
            except Exception as e:
                print(f"  ⚠ Error downloading {category_name}: {e}")
        
        # Remove duplicates by name
        seen_names = set()
        unique_landmarks = []
        
        for lm in all_landmarks:
            if lm['name'] not in seen_names:
                seen_names.add(lm['name'])
                unique_landmarks.append(lm)
        
        print(f"\n✓ Total unique landmarks: {len(unique_landmarks)}")
        return unique_landmarks


def merge_with_existing(new_landmarks: List[Dict], existing_path: Path) -> List[Dict]:
    """Merge new landmarks with existing database."""
    print("\nMerging with existing landmarks...")
    
    # Load existing
    with open(existing_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
        existing = existing_data['landmarks']
    
    print(f"  Existing: {len(existing)} landmarks")
    print(f"  New from Wikidata: {len(new_landmarks)} landmarks")
    
    # Create lookup by name
    existing_names = {lm['name'].lower() for lm in existing}
    
    # Add only new landmarks
    merged = existing.copy()
    added = 0
    
    for lm in new_landmarks:
        if lm['name'].lower() not in existing_names:
            merged.append(lm)
            added += 1
    
    print(f"  Added: {added} new landmarks")
    print(f"  Total: {len(merged)} landmarks")
    
    return merged


def main():
    """Download and merge Wikidata landmarks."""
    
    # Download from Wikidata
    downloader = WikidataLandmarkDownloader()
    
    try:
        wikidata_landmarks = downloader.download_all_categories()
    except Exception as e:
        print(f"\n❌ Error downloading from Wikidata: {e}")
        print("\nThis might be due to:")
        print("1. Network timeout (Wikidata queries can be slow)")
        print("2. Rate limiting (try again in a few minutes)")
        print("3. Query too large (reduce limit parameters)")
        return
    
    # Merge with existing
    data_dir = Path(__file__).parent.parent / 'data'
    existing_path = data_dir / 'landmarks_enriched.json'
    
    merged_landmarks = merge_with_existing(wikidata_landmarks, existing_path)
    
    # Save expanded database
    output_data = {
        'description': 'Expanded landmark database with Wikidata tourist attractions',
        'total': len(merged_landmarks),
        'sources': ['Google Landmarks (curated)', 'Wikidata (tourist attractions, monuments, natural)'],
        'landmarks': merged_landmarks
    }
    
    output_path = data_dir / 'landmarks_enriched.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("✓ DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"Total landmarks: {len(merged_landmarks)}")
    print(f"Saved to: {output_path}")
    print("\nRestart Streamlit to use the expanded database:")
    print("  streamlit run streamlit_demo.py")
    print("="*80)


if __name__ == '__main__':
    main()
