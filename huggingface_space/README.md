---
title: TripSaver Landmark Detector
emoji: 🗺️
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🗺️ TripSaver - AI Landmark Recognition

**EfficientNet-B3 trained on 500 landmark classes** | Part of the TripSaver full-stack AI platform

## 🚀 Features

- **Fast inference**: ~200ms on CPU (HuggingFace Spaces)
- **Trained on custom hardware**: RTX 4080, 20 hours of training
- **Social media ready**: Robust to Instagram/TikTok overlays and filters
- **API-ready**: Integrate with REST API calls
- **Free hosting**: Runs on HuggingFace Spaces

## 🎯 Model Details

### Architecture
- **Model**: EfficientNet-B3
- **Dataset**: Google Landmarks Dataset v2 (500 classes)
- **Input size**: 300x300 RGB images
- **Output**: Top-5 predictions with confidence scores

### Training
- **Hardware**: RTX 4080 16GB, i9-13900K, 64GB DDR5
- **Duration**: ~20 hours
- **Transfer learning**: ImageNet pretrained weights
- **Augmentation**: RandAugment, MixUp, CutMix
- **Custom augmentation**: Instagram/TikTok UI overlays (30% probability)
- **Optimizer**: AdamW with cosine annealing
- **Precision**: Mixed FP16 for faster training

### Preprocessing
- Template matching removes social media UI elements
- Resize(320) → CenterCrop(300) → ImageNet normalization



## 🌐 Full System

This model is part of the TripSaver platform, which includes:
- **Two-tier detection**: EfficientNet-B3 + Google Vision API validation
- **Visual similarity**: CLIP ViT-B/32 (4,248 landmarks)
- **Scene understanding**: Groq Llama 4 Scout 17B
- **Full-stack app**: Next.js frontend + AWS Lambda backend
- **Deployment**: Vercel (frontend) + AWS (backend) + HuggingFace Spaces (model)

## 🔗 Links

- **GitHub**: [TripSaver](https://github.com/evanpetersen919/CV-Location-Classifier)
- **Live App**: [TripSaver Web App](https://your-vercel-app.vercel.app)
- **Author**: [Evan Petersen](https://www.linkedin.com/in/evan-petersen-b93037386/)

All Rights Reserved © 2025 Evan Petersen
