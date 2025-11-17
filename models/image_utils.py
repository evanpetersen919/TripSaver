"""
Image Processing Utilities
==========================

Core utilities for image loading, preprocessing, augmentation, and visualization.
Provides unified interface for different model requirements.

Author: Evan Petersen
Date: November 2025
"""

import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional, List
import torch
from torchvision import transforms
import io


# ============================================================================
# IMAGE LOADING AND CONVERSION
# ============================================================================

def load_image(image_path: Union[str, io.BytesIO]) -> Image.Image:
    """
    Load an image from file path or bytes buffer.
    
    Args:
        image_path: Path to image file or BytesIO buffer
        
    Returns:
        PIL Image in RGB format
        
    Raises:
        ValueError: If image cannot be loaded or is corrupted
    """
    try:
        if isinstance(image_path, io.BytesIO):
            image = Image.open(image_path)
        else:
            image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
        
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")


def image_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """
    Convert PIL Image to PyTorch tensor.
    
    Args:
        image: PIL Image
        normalize: Whether to normalize to [0, 1] range
        
    Returns:
        Tensor of shape (C, H, W)
    """
    if normalize:
        transform = transforms.ToTensor()
    else:
        transform = transforms.Compose([
            transforms.PILToTensor(),
        ])
    
    return transform(image)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert PyTorch tensor to PIL Image.
    
    Args:
        tensor: Tensor of shape (C, H, W) or (H, W, C)
        
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    
    tensor = tensor.cpu().numpy()
    
    if tensor.max() <= 1.0:
        tensor = (tensor * 255).astype(np.uint8)
    else:
        tensor = tensor.astype(np.uint8)
    
    return Image.fromarray(tensor)


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.
    
    Args:
        image: PIL Image
        
    Returns:
        Numpy array of shape (H, W, C)
    """
    return np.array(image)


def numpy_to_image(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array of shape (H, W, C) or (H, W)
        
    Returns:
        PIL Image
    """
    if array.max() <= 1.0:
        array = (array * 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8)
    
    return Image.fromarray(array)


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def resize_image(image: Image.Image, size: Union[int, Tuple[int, int]], 
                 keep_aspect_ratio: bool = True) -> Image.Image:
    """
    Resize image to target size.
    
    Args:
        image: PIL Image
        size: Target size as (width, height) or single int for square
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if isinstance(size, int):
        size = (size, size)
    
    if keep_aspect_ratio:
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(size, Image.Resampling.LANCZOS)


def center_crop(image: Image.Image, size: Union[int, Tuple[int, int]]) -> Image.Image:
    """
    Center crop image to target size.
    
    Args:
        image: PIL Image
        size: Target size as (width, height) or single int for square
        
    Returns:
        Cropped PIL Image
    """
    if isinstance(size, int):
        size = (size, size)
    
    width, height = image.size
    target_width, target_height = size
    
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    return image.crop((left, top, right, bottom))


def normalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Normalize tensor with given mean and std.
    
    Args:
        tensor: Input tensor of shape (C, H, W) or (B, C, H, W)
        mean: Mean values for each channel
        std: Standard deviation values for each channel
        
    Returns:
        Normalized tensor
    """
    transform = transforms.Normalize(mean=mean, std=std)
    return transform(tensor)


# ============================================================================
# IMAGE AUGMENTATION
# ============================================================================

def get_training_transforms(input_size: int = 224) -> transforms.Compose:
    """
    Get standard training augmentation pipeline.
    
    Args:
        input_size: Target image size
        
    Returns:
        Composed transforms for training
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_validation_transforms(input_size: int = 224) -> transforms.Compose:
    """
    Get standard validation/inference pipeline.
    
    Args:
        input_size: Target image size
        
    Returns:
        Composed transforms for validation/inference
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_image_batch(image_paths: List[str], transform: transforms.Compose) -> torch.Tensor:
    """
    Process multiple images into a batch tensor.
    
    Args:
        image_paths: List of image file paths
        transform: Transform pipeline to apply
        
    Returns:
        Batch tensor of shape (B, C, H, W)
    """
    batch = []
    
    for path in image_paths:
        try:
            image = load_image(path)
            tensor = transform(image)
            batch.append(tensor)
        except Exception as e:
            print(f"Warning: Failed to process {path}: {str(e)}")
            continue
    
    if not batch:
        raise ValueError("No images could be processed")
    
    return torch.stack(batch)


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def show_image_grid(images: List[Image.Image], titles: Optional[List[str]] = None, 
                   rows: int = 1, cols: int = 4, figsize: Tuple[int, int] = (12, 4)):
    """
    Display multiple images in a grid.
    
    Args:
        images: List of PIL Images
        titles: Optional titles for each image
        rows: Number of rows in grid
        cols: Number of columns in grid
        figsize: Figure size as (width, height)
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis('off')
        
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
    
    plt.tight_layout()
    plt.show()


def save_image(image: Image.Image, path: str, quality: int = 95):
    """
    Save PIL Image to file.
    
    Args:
        image: PIL Image
        path: Output file path
        quality: JPEG quality (1-100)
    """
    image.save(path, quality=quality, optimize=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_image(image_path: str) -> bool:
    """
    Check if file is a valid image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False


def get_image_info(image: Union[str, Image.Image]) -> dict:
    """
    Get image metadata and properties.
    
    Args:
        image: Path to image or PIL Image
        
    Returns:
        Dictionary with image information
    """
    if isinstance(image, str):
        image = load_image(image)
    
    return {
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'format': image.format,
        'size_bytes': len(image.tobytes()) if hasattr(image, 'tobytes') else None
    }
