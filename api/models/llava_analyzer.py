"""
LLaVA Vision-Language Analyzer
================================

Uses LLaVA (Large Language and Vision Assistant) for image understanding
and location analysis. Provides natural language descriptions and can answer
questions about images.

Author: Evan Petersen
Date: November 2025
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import Optional, Dict, Any
from PIL import Image
import warnings


# ============================================================================
# LLAVA ANALYZER CLASS
# ============================================================================

class LLaVAAnalyzer:
    """
    Vision-Language analyzer using LLaVA model.
    
    Provides:
    - Natural language image descriptions
    - Location-specific analysis
    - Question answering about images
    
    Attributes:
        model: LLaVA model
        processor: Image and text processor
        device: Computation device (cuda or cpu)
    """
    
    def __init__(self,
                 model_name: str = "llava-hf/llava-1.5-7b-hf",
                 device: Optional[str] = None,
                 load_in_4bit: bool = True):
        """
        Initialize LLaVA analyzer.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda' or 'cpu')
            load_in_4bit: Use 4-bit quantization to save memory
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        print(f"Loading LLaVA model: {model_name}")
        print(f"Target device: {self.device} (CUDA available: {torch.cuda.is_available()})")
        print(f"PyTorch version: {torch.__version__}, CUDA build: {torch.version.cuda}")
        import sys
        print(f"Python: {sys.executable}")
        print("This may take a few minutes on first run (downloading ~13GB model)...")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load model with optional quantization
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
            )
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
            )
            print("✓ Model loaded with 4-bit quantization + CPU offloading")
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if not torch.cuda.is_available():
                self.model.to(self.device)
            print("✓ Model loaded")
        
        self.model.eval()
        
        # Check actual device after loading
        if hasattr(self.model, 'device'):
            actual_device = self.model.device
        elif hasattr(self.model, 'hf_device_map'):
            actual_device = f"distributed: {self.model.hf_device_map}"
        else:
            # Get device of first parameter
            actual_device = next(self.model.parameters()).device
        
        print(f"LLaVAAnalyzer initialized on {self.device}")
        print(f"Actual model device: {actual_device}")
    
    
    def analyze_location(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze visual features without guessing specific locations.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with feature description
        """
        # Focus on observable features, not location guessing
        question = "Describe the key visual features in this image: architectural style, colors, materials, landscape type, vegetation, weather, lighting, and any distinctive characteristics. Focus on what you see, not where it might be."
        
        description = self._generate_response(image, question)
        
        return {
            'description': description,
            'type': 'feature_analysis'
        }
    
    
    def describe_scene(self, image: Image.Image) -> Dict[str, Any]:
        """
        Generate a detailed description of the scene.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with scene description
        """
        prompt = "Describe this image in detail, including the setting, notable features, and overall atmosphere."
        
        description = self._generate_response(image, prompt)
        
        return {
            'description': description,
            'type': 'scene_description'
        }
    
    
    def ask_question(self, image: Image.Image, question: str) -> str:
        """
        Ask a specific question about the image.
        
        Args:
            image: PIL Image
            question: Question to ask
            
        Returns:
            Answer text
        """
        return self._generate_response(image, question)
    
    
    def _generate_response(self, image: Image.Image, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Generate response from LLaVA model.
        
        Args:
            image: PIL Image
            prompt: Text prompt/question
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # Format prompt for LLaVA
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
        
        # Move to device - use self.device instead of self.model.device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove prompt)
        # LLaVA includes the prompt in output, extract only the answer
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        elif prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    
    # ========================================================================
    # CONVENIENCE METHODS FOR PIPELINE
    # ========================================================================
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Main prediction method for pipeline integration.
        Returns location analysis by default.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with description and metadata
        """
        return self.analyze_location(image)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    print("=" * 80)
    print("LLAVA ANALYZER TEST")
    print("=" * 80)
    
    # Test with random image
    import numpy as np
    test_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_array)
    
    print("\nInitializing analyzer...")
    analyzer = LLaVAAnalyzer()
    
    print("\nAnalyzing test image...")
    result = analyzer.predict(test_image)
    
    print(f"\nResult:")
    print(f"Type: {result['type']}")
    print(f"Description: {result['description']}")
    
    print("\n" + "=" * 80)
