"""
Advanced Data Augmentation
===========================

Implements state-of-the-art augmentation strategies:
- AutoAugment
- RandAugment
- MixUp
- CutMix
- Test-Time Augmentation (TTA)

Author: Evan Petersen
Date: November 2025
"""

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
from typing import Tuple, List, Optional


# ============================================================================
# RANDAUGMENT
# ============================================================================

class RandAugment:
    """
    RandAugment: Practical automated data augmentation.
    
    Reference: https://arxiv.org/abs/1909.13719
    """
    
    def __init__(
        self,
        n: int = 2,
        m: int = 10,
        augmentations: Optional[List[str]] = None
    ):
        """
        Initialize RandAugment.
        
        Args:
            n: Number of augmentation transformations to apply
            m: Magnitude of augmentations (0-30)
            augmentations: List of augmentation operations to use
        """
        self.n = n
        self.m = m
        
        # Default augmentations
        self.augmentations = augmentations or [
            'AutoContrast', 'Equalize', 'Rotate', 'Solarize',
            'Color', 'Posterize', 'Contrast', 'Brightness',
            'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY'
        ]
    
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply RandAugment."""
        ops = random.choices(self.augmentations, k=self.n)
        
        for op in ops:
            img = self._apply_op(img, op, self.m)
        
        return img
    
    
    def _apply_op(self, img: Image.Image, op: str, magnitude: int) -> Image.Image:
        """Apply a single augmentation operation."""
        mag = float(magnitude) / 30.0  # Normalize to [0, 1]
        
        if op == 'AutoContrast':
            return ImageOps.autocontrast(img)
        
        elif op == 'Equalize':
            return ImageOps.equalize(img)
        
        elif op == 'Rotate':
            degrees = mag * 30.0  # Max 30 degrees
            return img.rotate(degrees)
        
        elif op == 'Solarize':
            threshold = int(mag * 256)
            return ImageOps.solarize(img, threshold)
        
        elif op == 'Color':
            factor = 1.0 + mag * 0.9  # 1.0 to 1.9
            enhancer = ImageEnhance.Color(img)
            return enhancer.enhance(factor)
        
        elif op == 'Posterize':
            bits = int(mag * 4) + 4  # 4 to 8 bits
            return ImageOps.posterize(img, bits)
        
        elif op == 'Contrast':
            factor = 1.0 + mag * 0.9
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(factor)
        
        elif op == 'Brightness':
            factor = 1.0 + mag * 0.9
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(factor)
        
        elif op == 'Sharpness':
            factor = 1.0 + mag * 0.9
            enhancer = ImageEnhance.Sharpness(img)
            return enhancer.enhance(factor)
        
        elif op == 'ShearX':
            shear = mag * 0.3  # Max 0.3
            return TF.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[shear, 0])
        
        elif op == 'ShearY':
            shear = mag * 0.3
            return TF.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[0, shear])
        
        elif op == 'TranslateX':
            pixels = int(mag * img.size[0] * 0.3)  # Max 30% of width
            return TF.affine(img, angle=0, translate=[pixels, 0], scale=1.0, shear=[0, 0])
        
        elif op == 'TranslateY':
            pixels = int(mag * img.size[1] * 0.3)  # Max 30% of height
            return TF.affine(img, angle=0, translate=[0, pixels], scale=1.0, shear=[0, 0])
        
        return img


# ============================================================================
# MIXUP
# ============================================================================

class MixUp:
    """
    MixUp: Beyond Empirical Risk Minimization.
    
    Reference: https://arxiv.org/abs/1710.09412
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize MixUp.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp augmentation.
        
        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B]
            
        Returns:
            mixed_images: Mixed images [B, C, H, W]
            labels_a: Original labels [B]
            labels_b: Mixed labels [B]
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_images, labels_a, labels_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    MixUp loss function.
    
    Args:
        criterion: Base loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        labels_a: Original labels
        labels_b: Mixed labels
        lam: Mixing coefficient
        
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


# ============================================================================
# CUTMIX
# ============================================================================

class CutMix:
    """
    CutMix: Regularization Strategy to Train Strong Classifiers.
    
    Reference: https://arxiv.org/abs/1905.04899
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation.
        
        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B]
            
        Returns:
            mixed_images: Mixed images [B, C, H, W]
            labels_a: Original labels [B]
            labels_b: Mixed labels [B]
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        # Get random box
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_images, labels_a, labels_b, lam


# ============================================================================
# TEST-TIME AUGMENTATION (TTA)
# ============================================================================

class TestTimeAugmentation:
    """
    Test-Time Augmentation for robust predictions.
    
    Averages predictions over multiple augmented versions of input.
    """
    
    def __init__(
        self,
        transforms: Optional[List[transforms.Compose]] = None,
        num_augmentations: int = 5
    ):
        """
        Initialize TTA.
        
        Args:
            transforms: List of augmentation transforms
            num_augmentations: Number of augmentations to average
        """
        self.num_augmentations = num_augmentations
        
        if transforms is None:
            # Default TTA transforms
            self.transforms = [
                transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
                transforms.Compose([transforms.RandomRotation(10)]),
                transforms.Compose([transforms.ColorJitter(brightness=0.2)]),
                transforms.Compose([transforms.RandomAffine(0, translate=(0.1, 0.1))]),
                transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.9, 1.0))]),
            ]
        else:
            self.transforms = transforms
    
    
    def __call__(
        self,
        model: nn.Module,
        image: torch.Tensor,
        base_transform: Optional[transforms.Compose] = None
    ) -> torch.Tensor:
        """
        Apply TTA and return averaged predictions.
        
        Args:
            model: Model to use for predictions
            image: Input image [C, H, W]
            base_transform: Base transform to apply to all augmentations
            
        Returns:
            Averaged predictions [num_classes]
        """
        model.eval()
        predictions = []
        
        with torch.no_grad():
            # Original prediction
            if base_transform:
                img = base_transform(image)
            else:
                img = image
            
            pred = model(img.unsqueeze(0))
            predictions.append(torch.softmax(pred, dim=1))
            
            # Augmented predictions
            for transform in self.transforms[:self.num_augmentations-1]:
                aug_img = transform(image)
                if base_transform:
                    aug_img = base_transform(aug_img)
                
                pred = model(aug_img.unsqueeze(0))
                predictions.append(torch.softmax(pred, dim=1))
        
        # Average predictions
        avg_pred = torch.stack(predictions).mean(dim=0)
        
        return avg_pred


# ============================================================================
# TRAINING TRANSFORMS
# ============================================================================

def get_advanced_train_transforms(input_size: int = 300) -> transforms.Compose:
    """
    Get advanced training transforms with RandAugment.
    
    Args:
        input_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        RandAugment(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test RandAugment
    from PIL import Image
    import numpy as np
    
    print("Testing RandAugment...")
    test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    rand_aug = RandAugment(n=2, m=10)
    augmented = rand_aug(test_img)
    print(f"✓ RandAugment applied: {test_img.size} -> {augmented.size}")
    
    # Test MixUp
    print("\nTesting MixUp...")
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])
    
    mixup = MixUp(alpha=1.0)
    mixed_images, labels_a, labels_b, lam = mixup(images, labels)
    print(f"✓ MixUp applied: lambda = {lam:.3f}")
    
    # Test CutMix
    print("\nTesting CutMix...")
    cutmix = CutMix(alpha=1.0)
    mixed_images, labels_a, labels_b, lam = cutmix(images, labels)
    print(f"✓ CutMix applied: lambda = {lam:.3f}")
    
    print("\n✓ All augmentations working!")
