"""
Hugging Face Inference API Client
===================================

Client for LLaVA model inference using HF's free GPU inference API.
Replaces local LLaVA model to save compute resources on AWS Lambda.

FREE TIER: 1,000 requests/month on HF Inference API
Alternative: Deploy on HF Spaces (unlimited, but public)

Author: Evan Petersen
Date: November 2025
"""

import os
import requests
import time
from typing import Optional, Dict, Any
from PIL import Image
import io
import base64


class HuggingFaceClient:
    """
    Client for Hugging Face Inference API.
    Handles LLaVA vision-language model inference.
    """
    
    def __init__(self, api_token: Optional[str] = None, model_id: str = "llava-hf/llava-1.5-7b-hf"):
        """
        Initialize HF Inference API client.
        
        Args:
            api_token: HF API token (get from https://huggingface.co/settings/tokens)
            model_id: Model identifier on Hugging Face Hub
        """
        self.api_token = api_token or os.getenv('HUGGINGFACE_API_TOKEN')
        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        if not self.api_token:
            raise ValueError(
                "HuggingFace API token required. "
                "Get one at: https://huggingface.co/settings/tokens"
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Seconds between requests
    
    
    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image
            
        Returns:
            Base64 encoded string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    
    def analyze_location(
        self,
        image: Image.Image,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> Dict[str, Any]:
        """
        Analyze image for location features using LLaVA.
        
        Args:
            image: PIL Image to analyze
            max_retries: Number of retry attempts if model is loading
            retry_delay: Seconds to wait between retries
            
        Returns:
            Dict with description and metadata
        """
        # Prepare the prompt (same as local LLaVA)
        prompt = (
            "Describe the key visual features in this image: "
            "architectural style, colors, materials, landscape type, vegetation, "
            "weather, lighting, and any distinctive characteristics. "
            "Focus on what you see, not where it might be."
        )
        
        # Convert image to base64
        img_base64 = self._image_to_base64(image)
        
        # Prepare payload
        payload = {
            "inputs": {
                "image": img_base64,
                "question": prompt
            },
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.1,  # Low temperature for consistent results
                "top_p": 0.9
            }
        }
        
        # Make request with retries
        for attempt in range(max_retries):
            self._wait_for_rate_limit()
            
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                # Handle different response codes
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract text from response
                    if isinstance(result, list) and len(result) > 0:
                        description = result[0].get('generated_text', '')
                    elif isinstance(result, dict):
                        description = result.get('generated_text', '')
                    else:
                        description = str(result)
                    
                    return {
                        'description': description.strip(),
                        'type': 'feature_analysis',
                        'model': self.model_id,
                        'success': True
                    }
                
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"Model loading, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {
                            'description': 'Model temporarily unavailable',
                            'type': 'error',
                            'error': 'Model loading timeout',
                            'success': False
                        }
                
                elif response.status_code == 429:
                    # Rate limit exceeded
                    return {
                        'description': 'Rate limit exceeded',
                        'type': 'error',
                        'error': 'Too many requests. Free tier: 1000/month',
                        'success': False
                    }
                
                else:
                    # Other error
                    return {
                        'description': f'API error: {response.status_code}',
                        'type': 'error',
                        'error': response.text,
                        'success': False
                    }
            
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {
                    'description': 'Request timeout',
                    'type': 'error',
                    'error': 'API request timed out',
                    'success': False
                }
            
            except Exception as e:
                return {
                    'description': 'Unexpected error',
                    'type': 'error',
                    'error': str(e),
                    'success': False
                }
        
        return {
            'description': 'Max retries exceeded',
            'type': 'error',
            'error': 'Failed after multiple attempts',
            'success': False
        }
    
    
    def ask_question(
        self,
        image: Image.Image,
        question: str,
        max_retries: int = 3
    ) -> str:
        """
        Ask a custom question about the image.
        
        Args:
            image: PIL Image
            question: Question to ask
            max_retries: Retry attempts
            
        Returns:
            Answer text
        """
        img_base64 = self._image_to_base64(image)
        
        payload = {
            "inputs": {
                "image": img_base64,
                "question": question
            },
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.1
            }
        }
        
        for attempt in range(max_retries):
            self._wait_for_rate_limit()
            
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '')
                    elif isinstance(result, dict):
                        return result.get('generated_text', '')
                    return str(result)
                
                elif response.status_code == 503 and attempt < max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                
                else:
                    return f"Error: {response.status_code}"
            
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2.0)
                    continue
                return f"Error: {str(e)}"
        
        return "Error: Max retries exceeded"
    
    
    def check_model_status(self) -> Dict[str, Any]:
        """
        Check if model is loaded and ready.
        
        Returns:
            Dict with model status
        """
        try:
            response = requests.get(
                self.api_url,
                headers=self.headers,
                timeout=10
            )
            
            return {
                'status_code': response.status_code,
                'loaded': response.status_code == 200,
                'response': response.text[:200]
            }
        
        except Exception as e:
            return {
                'status_code': None,
                'loaded': False,
                'error': str(e)
            }


# ============================================================================
# ALTERNATIVE: HUGGING FACE SPACES CLIENT
# ============================================================================

class HuggingFaceSpacesClient:
    """
    Alternative client for HF Spaces deployment.
    Use this if you deploy your own LLaVA space (unlimited, free).
    """
    
    def __init__(self, space_url: str):
        """
        Initialize Spaces client.
        
        Args:
            space_url: Your HF Spaces URL (e.g., https://username-space.hf.space)
        """
        self.space_url = space_url.rstrip('/')
        self.api_url = f"{self.space_url}/api/predict"
    
    
    def analyze_location(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze image using your deployed HF Space.
        
        Args:
            image: PIL Image
            
        Returns:
            Dict with description
        """
        # Convert image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        
        files = {'image': ('image.jpg', buffered.getvalue(), 'image/jpeg')}
        
        try:
            response = requests.post(
                self.api_url,
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'description': result.get('description', ''),
                    'type': 'feature_analysis',
                    'success': True
                }
            else:
                return {
                    'description': 'Error',
                    'type': 'error',
                    'error': f'Status {response.status_code}',
                    'success': False
                }
        
        except Exception as e:
            return {
                'description': 'Error',
                'type': 'error',
                'error': str(e),
                'success': False
            }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    import numpy as np
    
    print("=" * 80)
    print("HUGGING FACE CLIENT TEST")
    print("=" * 80)
    
    # Create test image
    test_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_array)
    
    print("\nInitializing HF client...")
    print("Note: Requires HUGGINGFACE_API_TOKEN environment variable")
    
    try:
        client = HuggingFaceClient()
        
        print("\nChecking model status...")
        status = client.check_model_status()
        print(f"Model loaded: {status['loaded']}")
        
        print("\nAnalyzing test image...")
        result = client.analyze_location(test_image)
        
        if result['success']:
            print(f"✓ Analysis successful!")
            print(f"Description: {result['description'][:100]}...")
        else:
            print(f"✗ Analysis failed: {result.get('error')}")
    
    except ValueError as e:
        print(f"✗ Error: {e}")
        print("\nTo use HF Inference API:")
        print("1. Get API token: https://huggingface.co/settings/tokens")
        print("2. Set environment variable: export HUGGINGFACE_API_TOKEN=your_token")
        print("3. Or pass token to HuggingFaceClient(api_token='your_token')")
    
    print("\n" + "=" * 80)
    print("HF CLIENT READY")
    print("=" * 80)
    print("\nFree Tier Limits:")
    print("  • 1,000 requests/month")
    print("  • Model may take 20-30s to load on first request")
    print("  • Shared infrastructure (variable latency)")
    print("\nFor production, consider deploying your own HF Space (free, unlimited)")
