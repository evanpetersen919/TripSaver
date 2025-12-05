"""
Test Groq vision API
"""
import requests
import base64
from PIL import Image
import io
import os
from dotenv import load_dotenv
import time

load_dotenv()

# Create test image
img = Image.new('RGB', (300, 300), 'blue')
buf = io.BytesIO()
img.save(buf, format='JPEG')
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode('utf-8')

groq_api_key = os.getenv('GROQ_API_KEY')
print(f"Using Groq API key: {groq_api_key[:20]}...")

print("\nTesting Groq Llama 4 Scout (Vision Model)...")
start = time.time()

response = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]
        }],
        "max_completion_tokens": 300,
        "temperature": 0.3
    },
    timeout=30
)

elapsed = time.time() - start

print(f"Status: {response.status_code}")
print(f"Time: {elapsed:.1f}s")

if response.status_code == 200:
    result = response.json()
    description = result['choices'][0]['message']['content']
    print(f"\n✅ SUCCESS!")
    print(f"Description: {description}")
    print(f"\nModel: {result['model']}")
    print(f"Usage: {result.get('usage', {})}")
else:
    print(f"\n❌ FAILED")
    print(f"Response: {response.text}")
