"""
Social Media Screenshot Augmentation

Adds realistic UI elements to clean training images to make model robust
to social media screenshots during inference.

Author: Evan Petersen
Date: November 2025
"""

import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from typing import Tuple


class SocialMediaAugmentation:
    """Add synthetic social media UI elements to training images."""
    
    def __init__(self, probability: float = 0.3):
        """
        Args:
            probability: Chance of applying social media augmentation
        """
        self.probability = probability
        
        # UI element colors (typical social media palettes)
        self.ui_colors = {
            'instagram': [(255, 255, 255), (240, 240, 240), (0, 0, 0)],
            'tiktok': [(0, 0, 0), (255, 255, 255), (254, 44, 85)],
            'dark_mode': [(18, 18, 18), (38, 38, 38), (255, 255, 255)]
        }
    
    def add_top_bar(self, image: Image.Image, height_ratio: float = 0.08) -> Image.Image:
        """Add top bar with profile name, time, menu."""
        w, h = image.size
        bar_height = int(h * height_ratio)
        
        # Create bar
        bar = Image.new('RGB', (w, bar_height), color=(250, 250, 250))
        draw = ImageDraw.Draw(bar)
        
        # Add some UI elements (circles for profile pic, lines for text)
        # Profile pic circle
        draw.ellipse([10, bar_height//4, 10+bar_height//2, 3*bar_height//4], 
                     fill=(200, 200, 200))
        
        # Username text simulation (gray rectangles)
        draw.rectangle([bar_height//2 + 20, bar_height//3, 
                       bar_height//2 + 120, bar_height//3 + 15], 
                      fill=(150, 150, 150))
        
        # Menu dots
        for i in range(3):
            x = w - 40 + i*10
            draw.ellipse([x, bar_height//2-2, x+4, bar_height//2+2], 
                        fill=(100, 100, 100))
        
        # Paste bar on top of image
        new_img = Image.new('RGB', (w, h + bar_height))
        new_img.paste(bar, (0, 0))
        new_img.paste(image, (0, bar_height))
        
        # Resize back to original size (squashes the image slightly)
        return new_img.resize((w, h), Image.LANCZOS)
    
    def add_bottom_bar(self, image: Image.Image, height_ratio: float = 0.12) -> Image.Image:
        """Add bottom bar with like, comment, share buttons."""
        w, h = image.size
        bar_height = int(h * height_ratio)
        
        # Create bar
        bar = Image.new('RGB', (w, bar_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(bar)
        
        # Action icons (heart, comment, share)
        icon_y = 10
        icon_spacing = 50
        
        # Heart icon (simple)
        draw.ellipse([15, icon_y, 30, icon_y+15], outline=(0, 0, 0), width=2)
        
        # Comment icon
        draw.rectangle([15+icon_spacing, icon_y, 30+icon_spacing, icon_y+15], 
                      outline=(0, 0, 0), width=2)
        
        # Share icon
        draw.polygon([(15+2*icon_spacing, icon_y), 
                     (30+2*icon_spacing, icon_y+7),
                     (15+2*icon_spacing, icon_y+15)], 
                    outline=(0, 0, 0))
        
        # Like count text simulation
        draw.rectangle([15, icon_y+25, 80, icon_y+35], fill=(180, 180, 180))
        
        # Paste bar on bottom of image
        new_img = Image.new('RGB', (w, h + bar_height))
        new_img.paste(image, (0, 0))
        new_img.paste(bar, (0, h))
        
        # Resize back to original size
        return new_img.resize((w, h), Image.LANCZOS)
    
    def add_caption_overlay(self, image: Image.Image, height_ratio: float = 0.15) -> Image.Image:
        """Add caption overlay at bottom."""
        w, h = image.size
        overlay_height = int(h * height_ratio)
        
        # Create semi-transparent overlay
        overlay = Image.new('RGBA', (w, overlay_height), color=(0, 0, 0, 180))
        draw = ImageDraw.Draw(overlay)
        
        # Simulate caption text
        for i in range(2):
            y = 10 + i*20
            width = random.randint(w//3, int(w*0.8))
            draw.rectangle([10, y, width, y+12], fill=(200, 200, 200, 255))
        
        # Convert image to RGBA
        img_rgba = image.convert('RGBA')
        
        # Paste overlay
        new_img = Image.new('RGBA', (w, h))
        new_img.paste(img_rgba, (0, 0))
        new_img.paste(overlay, (0, h - overlay_height), overlay)
        
        return new_img.convert('RGB')
    
    def add_side_bar(self, image: Image.Image, width_ratio: float = 0.08) -> Image.Image:
        """Add TikTok-style side bar with buttons."""
        w, h = image.size
        bar_width = int(w * width_ratio)
        
        # Create bar
        bar = Image.new('RGBA', (bar_width, h), color=(0, 0, 0, 100))
        draw = ImageDraw.Draw(bar)
        
        # Add button icons
        button_positions = [h//4, h//2, 3*h//4]
        for y in button_positions:
            # Circle button
            draw.ellipse([bar_width//4, y-20, 3*bar_width//4, y+20], 
                        fill=(255, 255, 255, 200))
        
        # Paste bar on right side
        img_rgba = image.convert('RGBA')
        new_img = Image.new('RGB', (w, h))
        new_img.paste(img_rgba.convert('RGB'), (0, 0))
        
        bar_rgb = bar.convert('RGB')
        # Blend the sidebar
        for x in range(bar_width):
            for y in range(h):
                old_pixel = new_img.getpixel((w-bar_width+x, y))
                new_pixel = bar_rgb.getpixel((x, y))
                blended = tuple(int(0.7*o + 0.3*n) for o, n in zip(old_pixel, new_pixel))
                new_img.putpixel((w-bar_width+x, y), blended)
        
        return new_img
    
    def add_letterbox(self, image: Image.Image, ratio: float = 0.1) -> Image.Image:
        """Add black/white bars (letterboxing)."""
        w, h = image.size
        bar_height = int(h * ratio)
        
        color = random.choice([(0, 0, 0), (255, 255, 255), (240, 240, 240)])
        
        # Create bars
        top_bar = Image.new('RGB', (w, bar_height), color=color)
        bottom_bar = Image.new('RGB', (w, bar_height), color=color)
        
        # Combine
        new_img = Image.new('RGB', (w, h + 2*bar_height))
        new_img.paste(top_bar, (0, 0))
        new_img.paste(image, (0, bar_height))
        new_img.paste(bottom_bar, (0, h + bar_height))
        
        # Resize back
        return new_img.resize((w, h), Image.LANCZOS)
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply random social media augmentation.
        
        Args:
            image: PIL Image (clean training image)
        
        Returns:
            Augmented image with social media UI elements
        """
        if random.random() > self.probability:
            return image
        
        # Randomly apply different UI elements
        augmentations = []
        
        if random.random() < 0.4:
            augmentations.append(('top_bar', self.add_top_bar))
        
        if random.random() < 0.5:
            augmentations.append(('bottom_bar', self.add_bottom_bar))
        
        if random.random() < 0.3:
            augmentations.append(('caption', self.add_caption_overlay))
        
        if random.random() < 0.2:
            augmentations.append(('side_bar', self.add_side_bar))
        
        if random.random() < 0.3:
            augmentations.append(('letterbox', self.add_letterbox))
        
        # Apply selected augmentations
        aug_image = image.copy()
        for name, aug_func in augmentations:
            try:
                aug_image = aug_func(aug_image)
            except Exception as e:
                print(f"Warning: Failed to apply {name}: {e}")
                continue
        
        return aug_image


class SocialMediaTransform:
    """Wrapper to use with PyTorch transforms."""
    
    def __init__(self, probability: float = 0.3):
        self.augmenter = SocialMediaAugmentation(probability)
    
    def __call__(self, img: Image.Image) -> Image.Image:
        return self.augmenter(img)


# Example usage with training pipeline
def get_training_transforms_with_social_media(input_size: int = 224):
    """Get training transforms including social media augmentation."""
    return transforms.Compose([
        # Apply social media augmentation FIRST (before other transforms)
        SocialMediaTransform(probability=0.3),
        
        # Then standard augmentations
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


if __name__ == "__main__":
    # Test the augmentation
    from pathlib import Path
    
    augmenter = SocialMediaAugmentation(probability=1.0)
    
    # Test on sample images
    sample_dir = Path("data/sample_images")
    output_dir = Path("data/augmented_samples")
    output_dir.mkdir(exist_ok=True)
    
    if sample_dir.exists():
        for img_path in list(sample_dir.glob("*.jpg"))[:5]:
            print(f"Augmenting: {img_path.name}")
            
            img = Image.open(img_path)
            
            # Apply augmentation
            aug_img = augmenter(img)
            
            # Save
            aug_img.save(output_dir / f"social_media_{img_path.name}")
            
        print(f"\nAugmented samples saved to: {output_dir}")
    else:
        print("No sample images found")
