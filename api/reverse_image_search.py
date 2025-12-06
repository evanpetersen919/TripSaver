"""
Reverse Image Search Fallback for Ultra-Niche Landmarks

Uses Google Cloud Vision API for landmark detection when EfficientNet + CLIP fail.
Falls back to web entity detection for obscure locations.

Author: Evan Petersen
Date: December 2025
"""

import os
import base64
from typing import Dict, List, Optional
from google.cloud import vision
from google.cloud.vision_v1 import types


def detect_landmarks_vision_api(image_bytes: bytes) -> Optional[Dict]:
    """
    Use Google Cloud Vision API for landmark detection.
    
    This is a paid API but extremely accurate for niche landmarks.
    Covers millions of landmarks worldwide.
    """
    try:
        client = vision.ImageAnnotatorClient()
        
        image = types.Image(content=image_bytes)
        
        # Landmark detection
        response = client.landmark_detection(image=image)
        landmarks = response.landmark_annotations
        
        if landmarks:
            results = []
            for landmark in landmarks[:5]:  # Top 5
                location = landmark.locations[0].lat_lng if landmark.locations else None
                
                results.append({
                    'name': landmark.description,
                    'confidence': landmark.score,
                    'latitude': location.latitude if location else None,
                    'longitude': location.longitude if location else None
                })
            
            return {
                'source': 'google_vision_api',
                'landmarks': results
            }
        
        # Fallback: Web entity detection for non-landmark places
        response = client.web_detection(image=image)
        web_entities = response.web_detection.web_entities
        
        if web_entities:
            # Try to extract location from web entities
            for entity in web_entities[:3]:
                if entity.score > 0.5:
                    return {
                        'source': 'web_entities',
                        'description': entity.description,
                        'confidence': entity.score
                    }
        
        return None
        
    except Exception as e:
        print(f"Vision API error: {e}")
        return None


def detect_with_bing_visual_search(image_url: str, subscription_key: str) -> Optional[Dict]:
    """
    Bing Visual Search API - Free tier available.
    Good for commercial/tourist locations.
    """
    import requests
    
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-Type': 'application/json'
    }
    
    data = {
        'url': image_url
    }
    
    try:
        response = requests.post(
            'https://api.bing.microsoft.com/v7.0/images/visualsearch',
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract landmark information
            tags = result.get('tags', [])
            for tag in tags:
                actions = tag.get('actions', [])
                for action in actions:
                    if action.get('actionType') == 'Entity':
                        return {
                            'source': 'bing_visual_search',
                            'name': action.get('displayName'),
                            'url': action.get('url'),
                            'description': action.get('snippet')
                        }
        
        return None
        
    except Exception as e:
        print(f"Bing search error: {e}")
        return None


def multi_tier_detection(image_bytes: bytes, image_url: Optional[str] = None) -> Dict:
    """
    Multi-tier fallback strategy:
    1. EfficientNet (500 classes) - FREE, FAST
    2. CLIP + Groq (15K landmarks) - FREE, 1-2s
    3. Google Vision API (millions) - $1.50/1000 images
    4. Bing Visual Search - Free tier available
    5. Manual entry
    """
    
    # Tier 3: Google Vision API
    vision_result = detect_landmarks_vision_api(image_bytes)
    if vision_result:
        return vision_result
    
    # Tier 4: Bing Visual Search (if image URL available)
    if image_url:
        bing_key = os.getenv('BING_SEARCH_API_KEY')
        if bing_key:
            bing_result = detect_with_bing_visual_search(image_url, bing_key)
            if bing_result:
                return bing_result
    
    # Tier 5: No results
    return {
        'source': 'none',
        'message': 'Location could not be identified. Please enter manually.'
    }


if __name__ == "__main__":
    # Test with sample image
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python reverse_image_search.py <image_path>")
        sys.exit(1)
    
    with open(sys.argv[1], 'rb') as f:
        image_data = f.read()
    
    result = detect_landmarks_vision_api(image_data)
    print("Result:", result)
