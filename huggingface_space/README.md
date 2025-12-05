---
title: CV Location Classifier
emoji: 🗺️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🗺️ CV Location Classifier - Multi-Model Vision Pipeline

**Two powerful AI models on HuggingFace Space:** EfficientNet-B3 and CLIP ViT-B/32 (vision analysis powered by Groq API)

## 🚀 Features

- **Fast inference**: ~200ms on GPU (T4 free tier)
- **High accuracy**: 81.37% on 500 landmark classes
- **API-ready**: Call via Gradio API or HTTP requests
- **Free hosting**: Runs on Hugging Face Spaces forever

## 📡 API Usage

```python
import requests
from PIL import Image
import io

# API endpoint
API_URL = "https://YOUR-USERNAME-landmark-detector.hf.space/api/predict"

# Upload image
with open("landmark.jpg", "rb") as f:
    response = requests.post(API_URL, files={"data": f})

predictions = response.json()["data"]
print(predictions)
```

## 🔑 Response Format

```json
{
  "success": true,
  "predictions": [
    {"class_id": 42, "confidence": 0.8567},
    {"class_id": 128, "confidence": 0.0891},
    ...
  ],
  "model": "EfficientNet-B3",
  "num_classes": 500,
  "device": "cuda"
}
```

## ℹ️ Model Details

- **Architecture**: EfficientNet-B3 (132MB)
- **Training**: 54 epochs, transfer learning from ImageNet
- **Dataset**: 500 landmark classes
- **Validation Accuracy**: 81.37% (top-1), ~95% (top-5)
- **Input size**: 300x300 RGB
- **Preprocessing**: Resize(320) → CenterCrop(300) → Normalize

## 🔗 Links

- **GitHub**: [CV-Location-Classifier](https://github.com/evanpetersen919/CV-Location-Classifier)
- **Author**: [Evan Petersen](https://www.linkedin.com/in/evan-petersen-b93037386/)
- **License**: MIT

---

Part of the CV Location Classifier project - AI-powered travel recommendations using landmark detection and visual similarity search.
