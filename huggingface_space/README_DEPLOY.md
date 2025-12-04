# 🚀 Deploying CV Location Classifier to Hugging Face Spaces

Complete guide to deploy CLIP + LLaVA + EfficientNet to HuggingFace for GPU-powered inference.

## 📋 Prerequisites

1. **Hugging Face Account**: Create at https://huggingface.co/join
2. **Git LFS**: Install from https://git-lfs.github.com/
3. **Model Checkpoint**: Have `landmark_detector_500classes_best.pth` ready

## 🛠️ Step-by-Step Deployment

### 1. Create New Space

```bash
# Go to: https://huggingface.co/new-space
# 
# Settings:
# - Name: cv-location-classifier
# - License: MIT
# - SDK: Gradio
# - Hardware: CPU Basic (free) or GPU (for faster inference)
# - Visibility: Public (for free tier)
```

### 2. Clone Your Space Locally

```bash
cd d:\VS Code\cv_pipeline

# Clone your space
git clone https://huggingface.co/spaces/YOUR-USERNAME/cv-location-classifier

# Or if already exists, just add remote
cd huggingface_space
git init
git remote add space https://huggingface.co/spaces/YOUR-USERNAME/cv-location-classifier
```

### 3. Copy Files

```bash
# Copy all files from huggingface_space/ to your cloned space
cd huggingface_space

# Required files:
# ├── app.py                                  # Main Gradio app
# ├── requirements.txt                        # Dependencies
# ├── README.md                               # Space description
# └── landmark_detector_500classes_best.pth   # Model checkpoint (132MB)
```

### 4. Setup Git LFS for Large Files

```bash
cd huggingface_space

# Initialize Git LFS
git lfs install

# Track the model checkpoint
git lfs track "*.pth"

# Add .gitattributes
git add .gitattributes
```

### 5. Push to Hugging Face

```bash
# Add all files
git add app.py requirements.txt README.md landmark_detector_500classes_best.pth

# Commit
git commit -m "Initial deployment: CLIP + LLaVA + EfficientNet pipeline"

# Push to space
git push space main
```

### 6. Wait for Build

- Go to your Space URL: `https://huggingface.co/spaces/YOUR-USERNAME/cv-location-classifier`
- HuggingFace will automatically:
  - Install dependencies (~5-10 minutes)
  - Download CLIP (~338MB) and LLaVA (~13GB)
  - Build and start the app
- First build takes 10-15 minutes

### 7. Test the API

Once deployed, test each endpoint:

```python
import requests
from PIL import Image
import io

# Your Space URL
SPACE_URL = "https://YOUR-USERNAME-cv-location-classifier.hf.space"

# Test image
image = Image.open("test_landmark.jpg")

# 1. Test Landmark Detection
img_bytes = io.BytesIO()
image.save(img_bytes, format='JPEG')
img_bytes.seek(0)

response = requests.post(
    f"{SPACE_URL}/api/predict_landmark",
    files={"data": img_bytes}
)
print("Landmark:", response.json())

# 2. Test CLIP Embedding
img_bytes.seek(0)
response = requests.post(
    f"{SPACE_URL}/api/get_clip_embedding",
    files={"data": img_bytes}
)
print("CLIP:", response.json())

# 3. Test LLaVA Analysis
img_bytes.seek(0)
response = requests.post(
    f"{SPACE_URL}/api/analyze_with_llava",
    files={"data": img_bytes}
)
print("LLaVA:", response.json())
```

## 🔗 Connect Lambda to HuggingFace Space

Update your Lambda function to call the Space:

```python
# In api/main.py
import os
import requests
from PIL import Image
import io

HUGGINGFACE_SPACE_URL = os.getenv('HUGGINGFACE_SPACE_URL', 
    'https://YOUR-USERNAME-cv-location-classifier.hf.space')

@app.post("/vision/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze image using CLIP + LLaVA on HuggingFace Space."""
    
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Call HF Space API
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Get CLIP embedding
    response = requests.post(
        f"{HUGGINGFACE_SPACE_URL}/api/get_clip_embedding",
        files={"data": img_bytes},
        timeout=30
    )
    clip_result = response.json()
    
    # Get LLaVA description
    img_bytes.seek(0)
    response = requests.post(
        f"{HUGGINGFACE_SPACE_URL}/api/analyze_with_llava",
        files={"data": img_bytes},
        timeout=30
    )
    llava_result = response.json()
    
    return {
        "clip_embedding": clip_result.get("embedding"),
        "description": llava_result.get("description"),
        "success": True
    }
```

## ⚙️ Hardware Recommendations

| Hardware | CLIP | LLaVA | EfficientNet | Cost |
|----------|------|-------|--------------|------|
| **CPU Basic** | ✅ ~2s | ⚠️ ~20-30s | ✅ ~1s | **FREE** |
| **CPU Upgrade** | ✅ ~1s | ✅ ~8-10s | ✅ ~500ms | $0.03/hr |
| **T4 Small GPU** | ✅ ~150ms | ✅ ~2-3s | ✅ ~200ms | $0.60/hr |
| **A10G Large GPU** | ✅ ~100ms | ✅ ~1-2s | ✅ ~150ms | $3.15/hr |

**Recommendation**: Start with **CPU Basic (free)** for testing, upgrade to **T4 GPU** for production.

## 🔒 Environment Variables

Set in Space Settings → Variables:

```bash
# Optional: HuggingFace token for private models
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx
```

## 📊 Monitoring

Monitor your Space:
- Usage: https://huggingface.co/spaces/YOUR-USERNAME/cv-location-classifier/settings
- Logs: View in Space → Logs tab
- Analytics: Space Settings → Analytics

## 🐛 Troubleshooting

### Build Fails

```bash
# Check requirements.txt has correct versions
torch>=2.0.0
open-clip-torch>=2.20.0
transformers>=4.37.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
```

### Out of Memory

1. Use CPU Basic (free) - LLaVA loads with 4-bit quantization
2. Or upgrade to T4 GPU ($0.60/hr)

### Model Download Slow

First build downloads ~14GB of models. This is normal and cached for future builds.

### CLIP Not Working

```python
# Check if open_clip is installed
pip install open-clip-torch
```

### LLaVA Not Working

```python
# Check transformers version
pip install transformers>=4.37.0 accelerate>=0.25.0
```

## 💡 Tips

1. **Free Tier**: Use CPU Basic for unlimited inference (slower but free)
2. **Speed**: Upgrade to T4 GPU for 10-15x faster inference
3. **Private Models**: Set `HUGGINGFACE_TOKEN` in Space settings
4. **Cold Start**: First request takes longer as models load into memory
5. **Caching**: Models cached after first load (faster subsequent requests)

## 🔄 Updating Your Space

```bash
cd huggingface_space

# Make changes to app.py or requirements.txt
git add .
git commit -m "Update: improved CLIP similarity"
git push space main

# HF automatically rebuilds and redeploys
```

## 📱 Access Your Space

- **Web UI**: https://huggingface.co/spaces/YOUR-USERNAME/cv-location-classifier
- **API Endpoint**: https://YOUR-USERNAME-cv-location-classifier.hf.space/api/predict
- **Embed**: Get embed code from Space → Embed button

## 🎉 Done!

Your multi-model vision pipeline is now live on HuggingFace Spaces with GPU acceleration!

Next steps:
1. Update Lambda to call your Space URL
2. Add image upload feature to frontend
3. Test end-to-end pipeline
4. Monitor usage and upgrade hardware if needed

---

Built by Evan Petersen | [GitHub](https://github.com/evanpetersen919/CV-Location-Classifier)
