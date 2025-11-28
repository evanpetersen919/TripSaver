"""
Landmark Detector API - Hugging Face Space
===========================================

EfficientNet-B3 model trained on 500 landmark classes with 81.37% accuracy.
Provides fast GPU inference via Gradio API.

Author: Evan Petersen
Date: November 2025
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List


class LandmarkDetector:
    """EfficientNet-B3 landmark detector."""
    
    def __init__(self, model_path: str, num_classes: int = 500):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        
        # Build model
        self.model = models.efficientnet_b3(weights=None)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        # Load weights
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"✓ Validation accuracy: {checkpoint.get('val_acc', 0):.2f}%")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Ready for inference on {num_classes} classes")
    
    def predict(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        """
        Predict landmarks in image.
        
        Args:
            image: PIL Image
            top_k: Number of top predictions
            
        Returns:
            List of dicts with 'class_id', 'confidence'
        """
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k
        top_probs, top_indices = torch.topk(probabilities[0], top_k)
        
        # Format results
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class_id': int(idx.item()),
                'confidence': float(prob.item())
            })
        
        return predictions


# ============================================================================
# INITIALIZE MODEL
# ============================================================================

print("Initializing Landmark Detector...")
detector = LandmarkDetector(
    model_path="landmark_detector_500classes_best.pth",
    num_classes=500
)
print("✓ Initialization complete!\n")


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def predict_landmark(image: Image.Image, top_k: int = 5) -> Dict:
    """
    Gradio prediction function.
    
    Args:
        image: Uploaded image
        top_k: Number of predictions to return
        
    Returns:
        Dictionary with predictions
    """
    if image is None:
        return {"error": "No image provided"}
    
    try:
        predictions = detector.predict(image, top_k=top_k)
        
        return {
            "success": True,
            "predictions": predictions,
            "model": "EfficientNet-B3",
            "num_classes": 500,
            "device": detector.device
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# BUILD GRADIO APP
# ============================================================================

with gr.Blocks(title="Landmark Detector API", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🗺️ Landmark Detector API")
    gr.Markdown("""
    **EfficientNet-B3 trained on 500 landmark classes**  
    Validation Accuracy: 81.37% | Inference: ~200ms on GPU
    
    Upload an image to detect landmarks. Returns top-5 predictions with confidence scores.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            top_k_slider = gr.Slider(
                minimum=1, 
                maximum=10, 
                value=5, 
                step=1, 
                label="Number of predictions"
            )
            predict_btn = gr.Button("🔍 Predict", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_json = gr.JSON(label="Predictions")
    
    predict_btn.click(
        fn=predict_landmark,
        inputs=[image_input, top_k_slider],
        outputs=[output_json]
    )
    
    gr.Markdown("---")
    gr.Markdown("""
    ### 📡 API Usage
    
    You can call this Space programmatically:
    
    ```python
    import requests
    from PIL import Image
    import io
    
    # Your Space URL
    API_URL = "https://YOUR-USERNAME-landmark-detector.hf.space/api/predict"
    
    # Upload image
    with open("landmark.jpg", "rb") as f:
        response = requests.post(API_URL, files={"data": f})
    
    predictions = response.json()["data"]
    print(predictions)
    ```
    
    ### 🔑 Response Format
    
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
    
    ### ℹ️ About
    
    - **Model**: EfficientNet-B3 (132MB)
    - **Training**: 500 landmark classes, 81.37% validation accuracy
    - **Inference**: GPU-accelerated (T4 free tier)
    - **Response time**: ~200ms per image
    
    Built by [Evan Petersen](https://github.com/evanpetersen919) | [CV Location Classifier](https://github.com/evanpetersen919/CV-Location-Classifier)
    """)


# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
