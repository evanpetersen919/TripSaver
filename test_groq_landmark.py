"""
Test Groq vision API with real landmark
"""
import requests
import base64
from PIL import Image
import io
import os
from dotenv import load_dotenv
import time

load_dotenv()

# Download Statue of Liberty image
print("Downloading test landmark image...")
img_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
img_response = requests.get(img_url, headers={'User-Agent': 'Mozilla/5.0'})
img_response.raise_for_status()
img = Image.open(io.BytesIO(img_response.content)).convert('RGB')

# Resize for efficiency
img.thumbnail((800, 800))
buf = io.BytesIO()
img.save(buf, format='JPEG', quality=85)
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode('utf-8')

groq_api_key = os.getenv('GROQ_API_KEY')

print("\n" + "="*80)
print("Testing Groq Llama 4 Scout with REAL LANDMARK")
print("="*80)
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
                {"type": "text", "text": "Describe this landmark in detail. Include its name if recognizable, architectural features, location characteristics, and notable details."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]
        }],
        "max_completion_tokens": 300,
        "temperature": 0.3
    },
    timeout=30
)

elapsed = time.time() - start

print(f"\n⏱️  Response Time: {elapsed:.2f} seconds")
print(f"📊 Status Code: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    description = result['choices'][0]['message']['content']
    usage = result.get('usage', {})
    
    print(f"\n✅ SUCCESS!\n")
    print("="*80)
    print("VISION DESCRIPTION:")
    print("="*80)
    print(description)
    print("="*80)
    print(f"\n📈 Token Usage:")
    print(f"   Prompt: {usage.get('prompt_tokens')} tokens")
    print(f"   Completion: {usage.get('completion_tokens')} tokens")
    print(f"   Total: {usage.get('total_tokens')} tokens")
    print(f"   Total Time: {usage.get('total_time', 0):.3f}s")
    
else:
    print(f"\n❌ FAILED")
    print(f"Response: {response.text}")
