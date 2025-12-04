"""
Landmark Detector API - HuggingFace Space (FastAPI Direct)
===========================================================

Pure FastAPI implementation bypassing Gradio to avoid schema bugs.
Serves all three models with full API functionality.

Author: Evan Petersen
Date: December 2025
"""

import json
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from typing import List, Dict, Optional
import numpy as np

# ============================================================================
# MODEL CLASSES (Same as before)
# ============================================================================

class LandmarkDetector:
    """EfficientNet-B3 landmark detector."""
    
    def __init__(self, checkpoint_path: str = "landmark_detector_500classes_best.pth", num_classes: int = 500):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load model
        self.model = models.efficientnet_b3(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            predictions.append({
                "class_id": int(idx),
                "confidence": float(prob)
            })
        
        return predictions


class ClipEmbedder:
    """CLIP ViT-B/32 embedder for images and text."""
    
    def __init__(self):
        try:
            import open_clip
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='openai'
            )
            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        except Exception as e:
            print(f"CLIP initialization failed: {e}")
            self.model = None
    
    def encode_image(self, image: Image.Image) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        
        text_tokens = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()[0]


class LLaVAAnalyzer:
    """LLaVA 1.5 7B vision-language model."""
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 4-bit quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.processor = LlavaNextProcessor.from_pretrained(model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
        except Exception as e:
            print(f"LLaVA initialization failed: {e}")
            self.model = None
    
    def analyze_scene(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        if self.model is None:
            return "LLaVA model not available"
        
        if prompt is None or not prompt.strip():
            prompt = "USER: <image>\nDescribe this landmark in detail. Include its name, location, and notable features.\nASSISTANT:"
        else:
            prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
        
        output = self.model.generate(**inputs, max_new_tokens=200)
        description = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "ASSISTANT:" in description:
            description = description.split("ASSISTANT:")[-1].strip()
        
        return description


# ============================================================================
# GLOBAL MODEL INSTANCES (Lazy Loading)
# ============================================================================

detector = None
clip_embedder = None
llava_analyzer = None

def get_detector():
    global detector
    if detector is None:
        print("Loading EfficientNet-B3...")
        detector = LandmarkDetector()
        print("✓ EfficientNet loaded")
    return detector

def get_clip_embedder():
    global clip_embedder
    if clip_embedder is None:
        print("Loading CLIP ViT-B/32...")
        clip_embedder = ClipEmbedder()
        print("✓ CLIP loaded")
    return clip_embedder

def get_llava_analyzer():
    global llava_analyzer
    if llava_analyzer is None:
        print("Loading LLaVA-1.5-7B...")
        llava_analyzer = LLaVAAnalyzer()
        print("✓ LLaVA loaded")
    return llava_analyzer


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="CV Location Classifier API",
    description="Multi-model vision pipeline with EfficientNet, CLIP, and LLaVA",
    version="2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n" + "="*80)
print("INITIALIZING VISION PIPELINE (LAZY LOADING)")
print("="*80)
print("\n✓ Ready for inference (models will load on first use)")
print("="*80 + "\n")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Web interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CV Location Classifier API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #27ae60; font-weight: bold; }
            code { background: #e8e8e8; padding: 2px 6px; border-radius: 3px; }
            .section { margin-top: 30px; }
        </style>
    </head>
    <body>
        <h1>🗺️ CV Location Classifier API</h1>
        <p><strong>Multi-Model Vision Pipeline</strong></p>
        <ul>
            <li>EfficientNet-B3: 500 landmark classes (81.37% accuracy)</li>
            <li>CLIP ViT-B/32: Visual similarity & text-to-image search</li>
            <li>LLaVA-1.5-7B: Natural language scene understanding</li>
        </ul>
        
        <div class="section">
            <h2>📡 API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/predict</code>
                <p>Landmark detection with EfficientNet-B3</p>
                <p><strong>Params:</strong> file (image), top_k (int, default=5)</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/clip/image</code>
                <p>Get CLIP embedding for image (512-dim vector)</p>
                <p><strong>Params:</strong> file (image)</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/clip/text</code>
                <p>Get CLIP embedding for text query</p>
                <p><strong>Params:</strong> text (string)</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/llava</code>
                <p>Analyze image with LLaVA vision-language model</p>
                <p><strong>Params:</strong> file (image), prompt (string, optional)</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/pipeline</code>
                <p>Run all three models on one image</p>
                <p><strong>Params:</strong> file (image)</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/health</code>
                <p>Check API health status</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/docs</code>
                <p>Interactive API documentation (Swagger UI)</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🔗 Links</h2>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="https://github.com/evanpetersen919/CV-Location-Classifier">GitHub Repository</a></li>
            </ul>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "models_loaded": {
            "efficientnet": detector is not None,
            "clip": clip_embedder is not None,
            "llava": llava_analyzer is not None
        }
    }


@app.post("/predict")
async def predict_landmark(file: UploadFile = File(...), top_k: int = Form(5)):
    """Detect landmarks with EfficientNet-B3."""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        det = get_detector()
        predictions = det.predict(image, top_k=top_k)
        
        return JSONResponse({
            "success": True,
            "predictions": predictions,
            "model": "EfficientNet-B3",
            "num_classes": 500
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clip/image")
async def clip_image_embedding(file: UploadFile = File(...)):
    """Get CLIP embedding for image."""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        clip = get_clip_embedder()
        embedding = clip.encode_image(image)
        
        if embedding is None:
            raise HTTPException(status_code=503, detail="CLIP model not available")
        
        return JSONResponse({
            "success": True,
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "model": "CLIP ViT-B/32"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clip/text")
async def clip_text_embedding(text: str = Form(...)):
    """Get CLIP embedding for text."""
    try:
        clip = get_clip_embedder()
        embedding = clip.encode_text(text)
        
        if embedding is None:
            raise HTTPException(status_code=503, detail="CLIP model not available")
        
        return JSONResponse({
            "success": True,
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "query": text,
            "model": "CLIP ViT-B/32"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llava")
async def analyze_with_llava(file: UploadFile = File(...), prompt: Optional[str] = Form(None)):
    """Analyze image with LLaVA."""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        llava = get_llava_analyzer()
        description = llava.analyze_scene(image, prompt=prompt)
        
        return JSONResponse({
            "success": True,
            "description": description,
            "model": "LLaVA-1.5-7B"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline")
async def complete_pipeline(file: UploadFile = File(...)):
    """Run all three models."""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Load all models
        det = get_detector()
        clip = get_clip_embedder()
        llava = get_llava_analyzer()
        
        # Run predictions
        predictions = det.predict(image, top_k=3)
        embedding = clip.encode_image(image)
        description = llava.analyze_scene(image)
        
        return JSONResponse({
            "success": True,
            "landmark_predictions": predictions,
            "clip_embedding_dim": len(embedding),
            "llava_description": description,
            "models_used": ["EfficientNet-B3", "CLIP ViT-B/32", "LLaVA-1.5-7B"]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
