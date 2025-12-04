"""
Landmark Detector API - Hugging Face Space
===========================================

Multi-Model Vision Pipeline:
1. EfficientNet-B3: 500 landmark classes (81.37% accuracy)
2. CLIP: Visual similarity search & text-to-image matching
3. LLaVA: Natural language scene understanding & captioning

Author: Evan Petersen
Date: December 2025
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


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
# CLIP EMBEDDER
# ============================================================================

class ClipEmbedder:
    """CLIP-based visual similarity search."""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        try:
            import open_clip
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            print(f"Loading CLIP model: {model_name}...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained='openai',
                device=self.device
            )
            self.model.eval()
            self.embedding_dim = self.model.visual.output_dim
            
            print(f"✓ CLIP loaded on {self.device} (dim={self.embedding_dim})")
        except ImportError:
            print("⚠️  OpenCLIP not available - CLIP features disabled")
            self.model = None
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Generate CLIP embedding for image."""
        if self.model is None:
            return None
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().astype('float32')[0]
    
    def encode_text(self, text: str) -> np.ndarray:
        """Generate CLIP embedding for text query."""
        if self.model is None:
            return None
        
        try:
            import open_clip
            text_tokens = open_clip.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy().astype('float32')[0]
        except Exception as e:
            print(f"Text encoding failed: {e}")
            return None
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Normalize
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)


# ============================================================================
# LLAVA ANALYZER
# ============================================================================

class LLaVAAnalyzer:
    """Vision-language analyzer using LLaVA."""
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
            
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            print(f"Loading LLaVA model: {model_name}...")
            print("This may take a few minutes (downloading ~13GB model)...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            # Load model with 4-bit quantization
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                print("✓ LLaVA loaded with 4-bit quantization")
            else:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                ).to(self.device)
                print("✓ LLaVA loaded on CPU")
            
            self.model.eval()
        except ImportError:
            print("⚠️  Transformers not available - LLaVA features disabled")
            self.model = None
    
    def analyze_scene(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """Generate description or answer question about image."""
        if self.model is None:
            return "LLaVA not available"
        
        if prompt is None:
            prompt = "Describe this image in detail, including the setting, notable features, architectural style, colors, and overall atmosphere."
        
        # Format conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        prompt_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt"
        )
        
        # Move to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
        
        # Decode
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        elif prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response


# ============================================================================
# INITIALIZE MODELS
# ============================================================================

print("="*80)
print("INITIALIZING VISION PIPELINE")
print("="*80)

# 1. EfficientNet Landmark Detector
print("\n[1/3] Initializing Landmark Detector...")
detector = LandmarkDetector(
    model_path="landmark_detector_500classes_best.pth",
    num_classes=500
)

# 2. CLIP Embedder
print("\n[2/3] Initializing CLIP...")
clip_embedder = ClipEmbedder(model_name="ViT-B/32")

# 3. LLaVA Analyzer
print("\n[3/3] Initializing LLaVA...")
llava_analyzer = LLaVAAnalyzer(model_name="llava-hf/llava-1.5-7b-hf")

print("\n" + "="*80)
print("✓ ALL MODELS INITIALIZED")
print("="*80 + "\n")


# ============================================================================
# GRADIO PREDICTION FUNCTIONS
# ============================================================================

def predict_landmark(image: Image.Image, top_k: int = 5) -> Dict:
    """EfficientNet landmark detection."""
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
        return {"success": False, "error": str(e)}


def get_clip_embedding(image: Image.Image) -> Dict:
    """Generate CLIP embedding for image."""
    if image is None:
        return {"error": "No image provided"}
    
    try:
        embedding = clip_embedder.encode_image(image)
        
        if embedding is None:
            return {"success": False, "error": "CLIP not available"}
        
        return {
            "success": True,
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "model": "CLIP ViT-B/32",
            "device": clip_embedder.device
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_by_text(text_query: str) -> Dict:
    """Generate CLIP text embedding for search."""
    if not text_query or not text_query.strip():
        return {"error": "No text provided"}
    
    try:
        embedding = clip_embedder.encode_text(text_query)
        
        if embedding is None:
            return {"success": False, "error": "CLIP not available"}
        
        return {
            "success": True,
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "query": text_query,
            "model": "CLIP ViT-B/32"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_with_llava(image: Image.Image, custom_prompt: str = "") -> Dict:
    """Analyze image with LLaVA."""
    if image is None:
        return {"error": "No image provided"}
    
    try:
        prompt = custom_prompt if custom_prompt.strip() else None
        description = llava_analyzer.analyze_scene(image, prompt=prompt)
        
        return {
            "success": True,
            "description": description,
            "model": "LLaVA-1.5-7B",
            "device": llava_analyzer.device
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def complete_pipeline(image: Image.Image) -> Tuple[Dict, Dict, Dict]:
    """Run all three models on image."""
    if image is None:
        error = {"error": "No image provided"}
        return error, error, error
    
    # Run all models
    landmark_result = predict_landmark(image, top_k=3)
    clip_result = get_clip_embedding(image)
    llava_result = analyze_with_llava(image)
    
    return landmark_result, clip_result, llava_result


# ============================================================================
# BUILD GRADIO APP
# ============================================================================

with gr.Blocks(title="CV Location Classifier API", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🗺️ CV Location Classifier - Multi-Model Vision Pipeline")
    gr.Markdown("""
    **Three powerful models working together:**
    - **EfficientNet-B3**: 500 landmark classes (81.37% accuracy)
    - **CLIP ViT-B/32**: Visual similarity & text-to-image search
    - **LLaVA-1.5-7B**: Natural language scene understanding
    """)
    
    with gr.Tabs():
        # ====================================================================
        # TAB 1: LANDMARK DETECTION
        # ====================================================================
        with gr.Tab("🏛️ Landmark Detection"):
            gr.Markdown("""
            ### EfficientNet-B3 Landmark Classifier
            Upload an image to identify landmarks from 500 classes.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    landmark_image = gr.Image(type="pil", label="Upload Image")
                    landmark_top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K Predictions")
                    landmark_btn = gr.Button("🔍 Detect Landmark", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    landmark_output = gr.JSON(label="Detection Results")
            
            landmark_btn.click(
                fn=predict_landmark,
                inputs=[landmark_image, landmark_top_k],
                outputs=[landmark_output]
            )
        
        # ====================================================================
        # TAB 2: CLIP EMBEDDINGS
        # ====================================================================
        with gr.Tab("🎨 CLIP Embeddings"):
            gr.Markdown("""
            ### Visual Similarity with CLIP
            Generate embeddings for images or text queries. Use for similarity search.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Image → Embedding")
                    clip_image = gr.Image(type="pil", label="Upload Image")
                    clip_image_btn = gr.Button("🖼️ Get Image Embedding", variant="primary")
                    clip_image_output = gr.JSON(label="Image Embedding")
                
                with gr.Column(scale=1):
                    gr.Markdown("#### Text → Embedding")
                    clip_text = gr.Textbox(
                        label="Text Query",
                        placeholder="e.g., 'a famous temple in Tokyo'",
                        lines=3
                    )
                    clip_text_btn = gr.Button("📝 Get Text Embedding", variant="primary")
                    clip_text_output = gr.JSON(label="Text Embedding")
            
            clip_image_btn.click(
                fn=get_clip_embedding,
                inputs=[clip_image],
                outputs=[clip_image_output]
            )
            
            clip_text_btn.click(
                fn=search_by_text,
                inputs=[clip_text],
                outputs=[clip_text_output]
            )
        
        # ====================================================================
        # TAB 3: LLAVA ANALYSIS
        # ====================================================================
        with gr.Tab("🤖 LLaVA Analysis"):
            gr.Markdown("""
            ### Natural Language Scene Understanding
            Describe images or ask questions about them using LLaVA.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    llava_image = gr.Image(type="pil", label="Upload Image")
                    llava_prompt = gr.Textbox(
                        label="Custom Prompt (optional)",
                        placeholder="Leave empty for automatic description, or ask a question...",
                        lines=3
                    )
                    llava_btn = gr.Button("💬 Analyze Scene", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    llava_output = gr.JSON(label="Analysis Result")
            
            llava_btn.click(
                fn=analyze_with_llava,
                inputs=[llava_image, llava_prompt],
                outputs=[llava_output]
            )
        
        # ====================================================================
        # TAB 4: COMPLETE PIPELINE
        # ====================================================================
        with gr.Tab("🚀 Complete Pipeline"):
            gr.Markdown("""
            ### All Models at Once
            Run all three models simultaneously for comprehensive analysis.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    pipeline_image = gr.Image(type="pil", label="Upload Image")
                    pipeline_btn = gr.Button("⚡ Run Full Pipeline", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    pipeline_landmark = gr.JSON(label="Landmark Detection")
                    pipeline_clip = gr.JSON(label="CLIP Embedding")
                    pipeline_llava = gr.JSON(label="LLaVA Description")
            
            pipeline_btn.click(
                fn=complete_pipeline,
                inputs=[pipeline_image],
                outputs=[pipeline_landmark, pipeline_clip, pipeline_llava]
            )
    
    # ========================================================================
    # API DOCUMENTATION
    # ========================================================================
    
    gr.Markdown("---")
    gr.Markdown("""
    ## 📡 API Usage
    
    Call endpoints programmatically:
    
    ```python
    import requests
    from PIL import Image
    
    API_URL = "https://YOUR-USERNAME-cv-location-classifier.hf.space"
    
    # Landmark detection
    with open("image.jpg", "rb") as f:
        response = requests.post(f"{API_URL}/api/predict_landmark", files={"data": f})
        result = response.json()
    
    # CLIP embedding
    response = requests.post(f"{API_URL}/api/get_clip_embedding", files={"data": f})
    embedding = response.json()["embedding"]
    
    # LLaVA analysis
    response = requests.post(f"{API_URL}/api/analyze_with_llava", files={"data": f})
    description = response.json()["description"]
    ```
    
    ## 🔑 Response Formats
    
    ### Landmark Detection
    ```json
    {
      "success": true,
      "predictions": [{"class_id": 42, "confidence": 0.8567}, ...],
      "model": "EfficientNet-B3",
      "num_classes": 500
    }
    ```
    
    ### CLIP Embedding
    ```json
    {
      "success": true,
      "embedding": [0.123, -0.456, ...],  // 512-dim vector
      "dimension": 512,
      "model": "CLIP ViT-B/32"
    }
    ```
    
    ### LLaVA Analysis
    ```json
    {
      "success": true,
      "description": "This image shows a traditional Japanese temple...",
      "model": "LLaVA-1.5-7B"
    }
    ```
    
    ## ℹ️ Model Details
    
    | Model | Size | Purpose | Inference Time |
    |-------|------|---------|----------------|
    | EfficientNet-B3 | 132MB | Landmark classification | ~200ms |
    | CLIP ViT-B/32 | 338MB | Visual similarity | ~150ms |
    | LLaVA-1.5-7B | ~4GB (quantized) | Scene understanding | ~2-3s |
    
    Built by [Evan Petersen](https://github.com/evanpetersen919) | [GitHub Repo](https://github.com/evanpetersen919/CV-Location-Classifier)
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
