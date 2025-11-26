"""
Social Media Screenshot Preprocessor

Detects and crops UI elements from Instagram, TikTok, Facebook, Twitter screenshots
to isolate the actual landmark photo before classification.

Author: Evan Petersen
Date: November 2025
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import torch
from pathlib import Path


class SocialMediaPreprocessor:
    """Detect and remove social media UI elements from screenshots."""
    
    def __init__(self):
        # Common social media UI detection patterns
        self.platform_configs = {
            'instagram': {
                'top_bar_height': 0.08,  # Profile name, menu
                'bottom_bar_height': 0.12,  # Like, comment, share buttons
                'caption_height': 0.15,  # Caption area
                'min_content_ratio': 0.5  # Minimum content area
            },
            'tiktok': {
                'side_bar_width': 0.08,  # Right sidebar with buttons
                'top_bar_height': 0.10,  # Username, audio
                'bottom_bar_height': 0.15,  # Caption and hashtags
                'min_content_ratio': 0.5
            },
            'facebook': {
                'top_bar_height': 0.10,
                'bottom_bar_height': 0.12,
                'min_content_ratio': 0.6
            },
            'twitter': {
                'top_bar_height': 0.08,
                'bottom_bar_height': 0.10,
                'min_content_ratio': 0.6
            }
        }
    
    def detect_ui_elements(self, image: np.ndarray) -> dict:
        """
        Detect social media UI elements using edge detection and color analysis.
        
        Returns dict with detected regions: {'top', 'bottom', 'left', 'right'}
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect horizontal lines (top/bottom bars)
        top_region = edges[:int(h*0.15), :]
        bottom_region = edges[int(h*0.85):, :]
        
        top_density = np.sum(top_region) / (top_region.size + 1)
        bottom_density = np.sum(bottom_region) / (bottom_region.size + 1)
        
        # Detect vertical concentration (sidebars)
        left_region = edges[:, :int(w*0.1)]
        right_region = edges[:, int(w*0.9):]
        
        left_density = np.sum(left_region) / (left_region.size + 1)
        right_density = np.sum(right_region) / (right_region.size + 1)
        
        # Threshold for UI detection (high edge density = UI elements)
        ui_threshold = 20
        
        return {
            'top': int(h * 0.12) if top_density > ui_threshold else 0,
            'bottom': int(h * 0.15) if bottom_density > ui_threshold else 0,
            'left': int(w * 0.05) if left_density > ui_threshold else 0,
            'right': int(w * 0.08) if right_density > ui_threshold else 0
        }
    
    def detect_white_bars(self, image: np.ndarray, threshold: int = 240) -> dict:
        """Detect solid color bars (letterboxing, borders)."""
        h, w = image.shape[:2]
        
        # Check top
        top_crop = 0
        for y in range(int(h * 0.2)):
            row = image[y, :, :]
            if np.mean(row) > threshold or np.std(row) < 5:
                top_crop = y + 1
            else:
                break
        
        # Check bottom
        bottom_crop = 0
        for y in range(h-1, int(h*0.8), -1):
            row = image[y, :, :]
            if np.mean(row) > threshold or np.std(row) < 5:
                bottom_crop = h - y
            else:
                break
        
        # Check left
        left_crop = 0
        for x in range(int(w * 0.2)):
            col = image[:, x, :]
            if np.mean(col) > threshold or np.std(col) < 5:
                left_crop = x + 1
            else:
                break
        
        # Check right
        right_crop = 0
        for x in range(w-1, int(w*0.8), -1):
            col = image[:, x, :]
            if np.mean(col) > threshold or np.std(col) < 5:
                right_crop = w - x
            else:
                break
        
        return {
            'top': top_crop,
            'bottom': bottom_crop,
            'left': left_crop,
            'right': right_crop
        }
    
    def smart_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Intelligently crop image to remove UI elements while preserving content.
        
        Strategy:
        1. Detect solid color bars (letterboxing)
        2. Detect UI elements (high edge density regions)
        3. Apply conservative crop
        4. Validate content area is sufficient
        """
        original_h, original_w = image.shape[:2]
        
        # Detect both white bars and UI elements
        bars = self.detect_white_bars(image)
        ui = self.detect_ui_elements(image)
        
        # Combine detections (take maximum crop from either method)
        top = max(bars['top'], ui['top'])
        bottom = max(bars['bottom'], ui['bottom'])
        left = max(bars['left'], ui['left'])
        right = max(bars['right'], ui['right'])
        
        # Apply crop
        cropped = image[top:original_h-bottom, left:original_w-right]
        
        # Validate we didn't crop too much
        new_h, new_w = cropped.shape[:2]
        content_ratio = (new_h * new_w) / (original_h * original_w)
        
        if content_ratio < 0.4:  # Less than 40% remaining
            # Too aggressive, use fallback: just remove top 10% and bottom 15%
            fallback_top = int(original_h * 0.10)
            fallback_bottom = int(original_h * 0.15)
            cropped = image[fallback_top:original_h-fallback_bottom, :]
        
        return cropped
    
    def preprocess(self, image: Image.Image, detect_platform: bool = True) -> Image.Image:
        """
        Main preprocessing function.
        
        Args:
            image: PIL Image (could be screenshot or clean photo)
            detect_platform: Whether to detect platform-specific UI
        
        Returns:
            Cropped PIL Image with UI elements removed
        """
        # Convert to numpy
        img_array = np.array(image)
        
        # Apply smart cropping
        cropped = self.smart_crop(img_array)
        
        # Convert back to PIL
        return Image.fromarray(cropped)
    
    def batch_preprocess(self, images: list) -> list:
        """Preprocess multiple images."""
        return [self.preprocess(img) for img in images]


def test_preprocessor():
    """Test the preprocessor on sample images."""
    preprocessor = SocialMediaPreprocessor()
    
    # Test on a sample image
    sample_dir = Path("data/sample_images")
    if not sample_dir.exists():
        print("No sample images found")
        return
    
    for img_path in sample_dir.glob("*.jpg")[:5]:
        print(f"\nProcessing: {img_path.name}")
        
        # Load image
        img = Image.open(img_path)
        original_size = img.size
        
        # Preprocess
        processed = preprocessor.preprocess(img)
        new_size = processed.size
        
        # Calculate crop percentages
        width_crop = (1 - new_size[0]/original_size[0]) * 100
        height_crop = (1 - new_size[1]/original_size[1]) * 100
        
        print(f"  Original: {original_size}")
        print(f"  Cropped:  {new_size}")
        print(f"  Removed:  {width_crop:.1f}% width, {height_crop:.1f}% height")
        
        # Save comparison
        output_dir = Path("data/preprocessed_comparison")
        output_dir.mkdir(exist_ok=True)
        processed.save(output_dir / f"cropped_{img_path.name}")


if __name__ == "__main__":
    test_preprocessor()
