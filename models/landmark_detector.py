"""
Landmark Recognition using EfficientNet
========================================

Identifies famous landmarks and monuments using transfer learning.
Fine-tunable on custom landmark datasets.

Author: Evan Petersen
Date: November 2025
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from PIL import Image
import json


# ============================================================================
# LANDMARK CATEGORIES
# ============================================================================

# Sample landmark categories (expandable with training)
DEFAULT_LANDMARKS = [
    'eiffel_tower', 'statue_of_liberty', 'big_ben', 'colosseum', 'taj_mahal',
    'great_wall_china', 'machu_picchu', 'christ_redeemer', 'sagrada_familia', 'burj_khalifa',
    'golden_gate_bridge', 'sydney_opera_house', 'tower_bridge', 'arc_de_triomphe', 'pantheon',
    'notre_dame', 'tower_of_pisa', 'acropolis', 'stonehenge', 'petra',
    'angkor_wat', 'forbidden_city', 'kremlin', 'neuschwanstein_castle', 'mont_saint_michel',
    'alhambra', 'versailles', 'louvre', 'buckingham_palace', 'white_house',
    'empire_state_building', 'cn_tower', 'space_needle', 'gateway_arch', 'mount_rushmore',
    'grand_canyon', 'niagara_falls', 'yellowstone', 'mount_fuji', 'uluru',
    'victoria_falls', 'iguazu_falls', 'table_mountain', 'santorini', 'venice_canals',
    'brooklyn_bridge', 'palace_of_westminster', 'reichstag', 'brandenburg_gate', 'charles_bridge',
    'unknown'  # Fallback category
]


# ============================================================================
# LANDMARK DETECTOR CLASS
# ============================================================================

class LandmarkDetector:
    """
    Landmark detection using fine-tuned EfficientNet.
    
    Supports transfer learning for custom landmark datasets.
    
    Attributes:
        model: PyTorch model (EfficientNet)
        device: Computation device (cuda or cpu)
        landmarks: List of landmark names
        transform: Image preprocessing pipeline
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 num_classes: Optional[int] = None,
                 landmarks: Optional[List[str]] = None):
        """
        Initialize landmark detector.
        
        Args:
            model_path: Path to pretrained weights
            device: Device to run on ('cuda' or 'cpu')
            num_classes: Number of landmark classes
            landmarks: List of landmark names
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set landmarks
        self.landmarks = landmarks if landmarks else DEFAULT_LANDMARKS
        self.num_classes = num_classes if num_classes else len(self.landmarks)
        
        # Build model
        self.model = self._build_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessing
        self.transform = self._get_transform()
        
        print(f"LandmarkDetector initialized on {self.device}")
        print(f"Number of landmarks: {self.num_classes}")
    
    
    def _build_model(self, model_path: Optional[str] = None) -> nn.Module:
        """
        Build EfficientNet model.
        
        Args:
            model_path: Path to pretrained weights
            
        Returns:
            Loaded PyTorch model
        """
        # Use EfficientNet-B3 (good balance of speed/accuracy)
        model = models.efficientnet_b3(pretrained=True)
        
        # Modify classifier for landmarks
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)
        
        # Load custom weights if available
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print("Custom weights loaded")
        else:
            print("Warning: Using ImageNet initialization (not landmark-specific)")
            print("For best results, fine-tune on landmark dataset")
        
        return model
    
    
    def _get_transform(self) -> transforms.Compose:
        """
        Get preprocessing transform.
        
        Returns:
            Composed transforms
        """
        return transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    
    # ========================================================================
    # PREDICTION METHODS
    # ========================================================================
    
    def predict(self, 
                image: Image.Image, 
                top_k: int = 5) -> List[Dict[str, any]]:
        """
        Predict landmarks in an image.
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            List of dicts with 'landmark', 'confidence', and 'index'
        """
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Format results
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            predictions.append({
                'landmark': self.landmarks[idx.item()],
                'confidence': prob.item(),
                'index': idx.item()
            })
        
        return predictions
    
    
    def predict_batch(self,
                     images: List[Image.Image],
                     top_k: int = 5) -> List[List[Dict[str, any]]]:
        """
        Predict landmarks for multiple images.
        
        Args:
            images: List of PIL Images
            top_k: Number of top predictions per image
            
        Returns:
            List of prediction lists for each image
        """
        # Preprocess all images
        img_tensors = torch.stack([
            self.transform(img) for img in images
        ]).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensors)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k for each image
        results = []
        for probs in probabilities:
            top_probs, top_indices = torch.topk(probs, top_k)
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({
                    'landmark': self.landmarks[idx.item()],
                    'confidence': prob.item(),
                    'index': idx.item()
                })
            results.append(predictions)
        
        return results
    
    
    def get_top_landmark(self, image: Image.Image) -> str:
        """
        Get the single most likely landmark.
        
        Args:
            image: PIL Image
            
        Returns:
            Landmark name as string
        """
        predictions = self.predict(image, top_k=1)
        return predictions[0]['landmark']
    
    
    def is_known_landmark(self, 
                         image: Image.Image, 
                         threshold: float = 0.5) -> bool:
        """
        Check if image contains a known landmark with high confidence.
        
        Args:
            image: PIL Image
            threshold: Minimum confidence threshold
            
        Returns:
            True if known landmark detected above threshold
        """
        predictions = self.predict(image, top_k=1)
        top_pred = predictions[0]
        
        return (top_pred['landmark'] != 'unknown' and 
                top_pred['confidence'] >= threshold)
    
    
    # ========================================================================
    # TRAINING METHODS
    # ========================================================================
    
    def set_training_mode(self, freeze_backbone: bool = True):
        """
        Set model to training mode.
        
        Args:
            freeze_backbone: Whether to freeze EfficientNet backbone
        """
        self.model.train()
        
        if freeze_backbone:
            # Freeze all layers except classifier
            for param in self.model.features.parameters():
                param.requires_grad = False
            
            # Unfreeze classifier
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            
            print("Training mode: Classifier only")
        else:
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True
            
            print("Training mode: Full model")
    
    
    def set_eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
        print("Evaluation mode activated")
    
    
    def save_model(self, save_path: str):
        """
        Save model weights.
        
        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
        # Save landmark names
        landmarks_file = save_path.parent / "landmarks.json"
        with open(landmarks_file, 'w') as f:
            json.dump(self.landmarks, f, indent=2)
        print(f"Landmarks saved to {landmarks_file}")
    
    
    def load_model(self, model_path: str):
        """
        Load model weights.
        
        Args:
            model_path: Path to model weights
        """
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path}")
        
        # Load landmark names if available
        landmarks_file = Path(model_path).parent / "landmarks.json"
        if landmarks_file.exists():
            with open(landmarks_file, 'r') as f:
                self.landmarks = json.load(f)
            print(f"Landmarks loaded: {len(self.landmarks)} classes")
    
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_landmark_metadata(self, landmark_name: str) -> Dict:
        """
        Get metadata for a landmark (to be extended with database).
        
        Args:
            landmark_name: Name of the landmark
            
        Returns:
            Dictionary with landmark metadata
        """
        # TODO: Connect to database for real metadata
        return {
            'name': landmark_name.replace('_', ' ').title(),
            'category': 'landmark',
            'confidence_threshold': 0.5
        }
    
    
    def filter_by_region(self,
                        predictions: List[Dict[str, any]],
                        region: str) -> List[Dict[str, any]]:
        """
        Filter predictions by geographic region.
        
        Args:
            predictions: List of prediction dicts
            region: Region name (e.g., 'Europe', 'Asia')
            
        Returns:
            Filtered predictions
        """
        # TODO: Add region mapping for landmarks
        # For now, return all predictions
        return predictions
    
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"LandmarkDetector(device='{self.device}', "
                f"num_classes={self.num_classes}, "
                f"landmarks={len(self.landmarks)})")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_training_transforms(input_size: int = 300) -> transforms.Compose:
    """
    Get augmentation transforms for training.
    
    Args:
        input_size: Target image size
        
    Returns:
        Composed transforms with augmentation
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_validation_transforms(input_size: int = 300) -> transforms.Compose:
    """
    Get transforms for validation/inference.
    
    Args:
        input_size: Target image size
        
    Returns:
        Composed transforms without augmentation
    """
    return transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
