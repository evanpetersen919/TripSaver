# Google Vision API Integration Guide

## Setup Steps

### 1. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or use existing)
3. Enable **Cloud Vision API**
   - Navigation → APIs & Services → Library
   - Search "Cloud Vision API"
   - Click Enable

### 2. Create Service Account

```bash
# Create service account
gcloud iam service-accounts create cv-location-classifier \
    --display-name="CV Location Classifier" \
    --project=YOUR_PROJECT_ID

# Grant Vision API permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:cv-location-classifier@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/cloudvision.annotator"

# Create and download key
gcloud iam service-accounts keys create google-credentials.json \
    --iam-account=cv-location-classifier@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 3. Deploy to Lambda

```bash
# Encode credentials as base64
$credentials = Get-Content google-credentials.json -Raw
$base64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($credentials))

# Deploy with SAM
sam deploy \
  --parameter-overrides \
    GroqAPIKey="YOUR_GROQ_KEY" \
    GoogleCloudCredentials="$base64" \
    JWTSecret="YOUR_JWT_SECRET" \
    HuggingFaceAPIToken="YOUR_HF_TOKEN"
```

### 4. Lambda Configuration

The credentials will be automatically decoded and saved to `/tmp/google-credentials.json` in your Lambda function.

Add this to your Lambda function initialization:

```python
# In api/main.py startup
import os
import base64

# Decode Google Cloud credentials from environment
google_creds_b64 = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if google_creds_b64:
    try:
        creds_json = base64.b64decode(google_creds_b64).decode('utf-8')
        creds_path = '/tmp/google-credentials.json'
        with open(creds_path, 'w') as f:
            f.write(creds_json)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
        print("✅ Google Cloud credentials loaded")
    except Exception as e:
        print(f"⚠️ Failed to load Google credentials: {e}")
```

## Pricing

**Google Cloud Vision API Pricing:**
- First 1,000 requests/month: **FREE**
- 1,001 - 5,000,000: **$1.50 per 1,000 images**
- 5M+: **$0.60 per 1,000 images**

**Your Current Fallback Chain:**
1. EfficientNet (500 classes) - FREE, 0.5s
2. CLIP + Groq (15K landmarks) - FREE, 1-2s
3. **Google Vision API** - $1.50/1000 (only if confidence < 60%)
4. Manual entry - FREE, 10s

**Expected Usage:**
- Tier 3 triggered only when:
  - CLIP confidence < 60%, OR
  - No CLIP results found
- Estimated: ~10-20% of uploads
- Monthly cost at 1,000 uploads: **~$0.15 - $0.30**

## Testing Locally

```bash
# Set credentials
$env:GOOGLE_APPLICATION_CREDENTIALS="d:\VS Code\cv_pipeline\google-credentials.json"

# Test Vision API
cd api
python -c "
from google.cloud import vision
client = vision.ImageAnnotatorClient()
print('✅ Google Vision API connected successfully')
"

# Test with sample image
python reverse_image_search.py ../data/sample_images/tokyo_tower.jpg
```

## Monitoring

Track Vision API usage in [Google Cloud Console](https://console.cloud.google.com/apis/dashboard):
- Navigation → APIs & Services → Dashboard
- Click "Cloud Vision API"
- View usage graphs and quota

## Fallback Logic

```python
# Pseudocode
if efficientnet_top1_confidence < 0.5:
    # User rejects → Tier 2
    clip_results = search_with_clip(image)
    groq_description = analyze_with_groq(image)
    
    if clip_top1_similarity < 0.6:
        # Low confidence → Tier 3
        vision_results = google_vision_api(image)
        
        if vision_top1_confidence > 0.7:
            return vision_results  # High confidence from Vision API
    
    return clip_results
```

## Disable Vision API (Optional)

To disable Tier 3 and use only free tiers:

```bash
# Deploy without Google credentials
sam deploy --parameter-overrides GoogleCloudCredentials=""
```

The code will automatically skip Vision API if credentials are not available.
