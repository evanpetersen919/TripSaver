"""
Test Google Vision API integration for niche landmark detection

Usage:
    python test_vision_api.py <image_path>

Author: Evan Petersen
Date: December 2025
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_vision_api(image_path: str):
    """Test Google Vision API with an image"""
    
    try:
        from google.cloud import vision
        from google.cloud.vision_v1 import types as vision_types
        
        print("✅ Google Cloud Vision library imported successfully")
        
        # Create client
        client = vision.ImageAnnotatorClient()
        print("✅ Vision API client created")
        
        # Load image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        image = vision_types.Image(content=image_data)
        print(f"✅ Loaded image: {image_path}")
        
        # Landmark detection
        print("\n🔍 Performing landmark detection...")
        response = client.landmark_detection(image=image)
        landmarks = response.landmark_annotations
        
        if landmarks:
            print(f"\n✅ Found {len(landmarks)} landmarks:")
            print("=" * 60)
            for i, landmark in enumerate(landmarks[:5], 1):
                location = landmark.locations[0].lat_lng if landmark.locations else None
                
                print(f"\n{i}. {landmark.description}")
                print(f"   Confidence: {landmark.score * 100:.1f}%")
                if location:
                    print(f"   Location: {location.latitude:.6f}, {location.longitude:.6f}")
        else:
            print("\n⚠️ No landmarks detected")
            
            # Try web entity detection as fallback
            print("\n🔍 Trying web entity detection...")
            response = client.web_detection(image=image)
            web_entities = response.web_detection.web_entities
            
            if web_entities:
                print(f"\n✅ Found {len(web_entities)} web entities:")
                print("=" * 60)
                for i, entity in enumerate(web_entities[:5], 1):
                    print(f"\n{i}. {entity.description}")
                    print(f"   Score: {entity.score * 100:.1f}%")
            else:
                print("⚠️ No web entities found")
        
        return True
        
    except ImportError:
        print("❌ Google Cloud Vision library not installed")
        print("\nInstall with:")
        print("  pip install google-cloud-vision")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            print("\n⚠️ GOOGLE_APPLICATION_CREDENTIALS not set")
            print("\nSet credentials:")
            print("  $env:GOOGLE_APPLICATION_CREDENTIALS=\"path\\to\\credentials.json\"")
        
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_vision_api.py <image_path>")
        print("\nExample:")
        print("  python test_vision_api.py ../data/sample_images/tokyo_tower.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    success = test_vision_api(image_path)
    sys.exit(0 if success else 1)
