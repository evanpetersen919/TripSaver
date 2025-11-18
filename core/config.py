"""
Configuration Management for CV Pipeline

This module provides centralized configuration for all computer vision models,
including model paths, hyperparameters, device settings, and performance tuning.

Author: Evan Petersen
Date: November 2025
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
for directory in [DATA_DIR, CHECKPOINTS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

@dataclass
class DeviceConfig:
    """Device and hardware acceleration settings"""
    
    # Primary device selection
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # GPU settings
    gpu_id: int = 0
    use_amp: bool = True  # Automatic Mixed Precision for faster inference
    
    # Multi-GPU settings
    use_data_parallel: bool = False
    gpu_ids: list = field(default_factory=lambda: [0])
    
    def get_device(self) -> torch.device:
        """Get the torch device object"""
        if self.device == "cuda":
            return torch.device(f"cuda:{self.gpu_id}")
        return torch.device("cpu")
    
    def __post_init__(self):
        """Validate device configuration"""
        if self.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            self.device = "cpu"


# ============================================================================
# IMAGE PROCESSING CONFIGURATION
# ============================================================================

@dataclass
class ImageConfig:
    """Image preprocessing and augmentation settings"""
    
    # Standard input sizes
    default_size: tuple = (224, 224)
    scene_classifier_size: tuple = (224, 224)
    clip_size: tuple = (224, 224)
    landmark_detector_size: tuple = (300, 300)  # EfficientNet-B3 optimal
    
    # Normalization (ImageNet statistics)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    
    # Data augmentation (training only)
    random_rotation: int = 15
    random_crop_scale: tuple = (0.8, 1.0)
    horizontal_flip_prob: float = 0.5
    color_jitter: Dict[str, float] = field(default_factory=lambda: {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1
    })
    
    # Quality settings
    jpeg_quality: int = 95
    interpolation: str = "bilinear"  # bilinear, bicubic, nearest


# ============================================================================
# SCENE CLASSIFIER CONFIGURATION
# ============================================================================

@dataclass
class SceneClassifierConfig:
    """Places365 scene classifier settings"""
    
    # Model architecture
    model_name: str = "resnet50"
    num_classes: int = 365
    pretrained: bool = True
    
    # Inference settings
    batch_size: int = 32
    num_workers: int = 4
    confidence_threshold: float = 0.3
    
    # Top-k predictions
    top_k: int = 5
    
    # Model checkpoint
    checkpoint_path: Optional[str] = None
    
    # Travel relevance filtering
    min_travel_confidence: float = 0.5
    
    def get_checkpoint_path(self) -> Path:
        """Get full path to model checkpoint"""
        if self.checkpoint_path:
            return Path(self.checkpoint_path)
        return CHECKPOINTS_DIR / "scene_classifier_best.pth"


# ============================================================================
# CLIP EMBEDDER CONFIGURATION
# ============================================================================

@dataclass
class CLIPConfig:
    """CLIP visual similarity search settings"""
    
    # Model variant
    model_name: str = "ViT-B/32"  # ViT-B/32, ViT-B/16, ViT-L/14
    embedding_dim: int = 512
    
    # FAISS index settings
    index_type: str = "IndexFlatIP"  # Flat inner product (cosine similarity)
    use_gpu_index: bool = False  # GPU index not available on Windows
    nprobe: int = 10  # Number of cells to visit (for IVF indexes)
    
    # Search settings
    top_k: int = 10
    similarity_threshold: float = 0.7
    
    # Batch processing
    batch_size: int = 64
    num_workers: int = 4
    
    # Index persistence
    index_save_path: Optional[str] = None
    metadata_save_path: Optional[str] = None
    
    def get_index_path(self) -> Path:
        """Get full path to FAISS index"""
        if self.index_save_path:
            return Path(self.index_save_path)
        return CACHE_DIR / "clip_index.faiss"
    
    def get_metadata_path(self) -> Path:
        """Get full path to index metadata"""
        if self.metadata_save_path:
            return Path(self.metadata_save_path)
        return CACHE_DIR / "clip_metadata.json"


# ============================================================================
# LANDMARK DETECTOR CONFIGURATION
# ============================================================================

@dataclass
class LandmarkDetectorConfig:
    """EfficientNet landmark recognition settings"""
    
    # Model architecture
    model_name: str = "efficientnet_b3"
    num_classes: int = 51  # Will increase with training data
    pretrained: bool = True
    
    # Transfer learning settings
    freeze_backbone: bool = True  # Freeze during initial training
    unfreeze_at_epoch: int = 10  # Unfreeze for fine-tuning
    
    # Training settings
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    dropout_rate: float = 0.3
    
    # Inference settings
    batch_size: int = 32
    num_workers: int = 4
    confidence_threshold: float = 0.5
    
    # Top-k predictions
    top_k: int = 3
    
    # Model checkpoint
    checkpoint_path: Optional[str] = None
    
    def get_checkpoint_path(self) -> Path:
        """Get full path to model checkpoint"""
        if self.checkpoint_path:
            return Path(self.checkpoint_path)
        return CHECKPOINTS_DIR / "landmark_detector_best.pth"
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration dictionary"""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout_rate": self.dropout_rate,
            "freeze_backbone": self.freeze_backbone,
            "unfreeze_at_epoch": self.unfreeze_at_epoch
        }


# ============================================================================
# PIPELINE ORCHESTRATION CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Settings for parallel model execution"""
    
    # Execution mode
    parallel_execution: bool = True
    use_asyncio: bool = True
    
    # Timeouts (seconds)
    model_timeout: float = 5.0
    total_timeout: float = 15.0
    
    # Performance settings
    warmup_iterations: int = 3  # Warmup runs for accurate benchmarking
    enable_profiling: bool = False
    
    # Result aggregation
    combine_results: bool = True
    min_confidence: float = 0.3
    
    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600  # Cache time-to-live in seconds


# ============================================================================
# GLOBAL CONFIGURATION INSTANCE
# ============================================================================

@dataclass
class GlobalConfig:
    """Master configuration for entire CV pipeline"""
    
    device: DeviceConfig = field(default_factory=DeviceConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    scene_classifier: SceneClassifierConfig = field(default_factory=SceneClassifierConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    landmark_detector: LandmarkDetectorConfig = field(default_factory=LandmarkDetectorConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = str(DATA_DIR / "pipeline.log")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "device": self.device.__dict__,
            "image": self.image.__dict__,
            "scene_classifier": self.scene_classifier.__dict__,
            "clip": self.clip.__dict__,
            "landmark_detector": self.landmark_detector.__dict__,
            "pipeline": self.pipeline.__dict__,
            "log_level": self.log_level,
            "log_to_file": self.log_to_file,
            "log_file": self.log_file
        }
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 80)
        print("CV PIPELINE CONFIGURATION")
        print("=" * 80)
        print(f"Device: {self.device.device}")
        if self.device.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(self.device.gpu_id)}")
            print(f"Mixed Precision: {self.device.use_amp}")
        print(f"\nScene Classifier: {self.scene_classifier.model_name} ({self.scene_classifier.num_classes} classes)")
        print(f"CLIP Model: {self.clip.model_name} ({self.clip.embedding_dim}D embeddings)")
        print(f"Landmark Detector: {self.landmark_detector.model_name} ({self.landmark_detector.num_classes} classes)")
        print(f"\nParallel Execution: {self.pipeline.parallel_execution}")
        print(f"Async Mode: {self.pipeline.use_asyncio}")
        print("=" * 80)


# Create default configuration instance
config = GlobalConfig()


# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

def get_config() -> GlobalConfig:
    """Get the global configuration instance"""
    return config


def update_config(**kwargs):
    """Update configuration parameters
    
    Args:
        **kwargs: Configuration parameters to update
        
    Example:
        update_config(device="cpu", log_level="DEBUG")
    """
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"WARNING: Unknown configuration parameter: {key}")


def reset_config():
    """Reset configuration to defaults"""
    global config
    config = GlobalConfig()


def validate_config() -> bool:
    """Validate configuration settings
    
    Returns:
        bool: True if configuration is valid
    """
    try:
        # Check device
        device = config.device.get_device()
        
        # Check paths exist
        assert DATA_DIR.exists(), f"Data directory not found: {DATA_DIR}"
        assert MODELS_DIR.exists(), f"Models directory not found: {MODELS_DIR}"
        
        # Check image sizes
        assert len(config.image.default_size) == 2, "Image size must be (height, width)"
        assert config.image.default_size[0] > 0 and config.image.default_size[1] > 0
        
        # Check batch sizes
        assert config.scene_classifier.batch_size > 0
        assert config.clip.batch_size > 0
        assert config.landmark_detector.batch_size > 0
        
        # Check thresholds
        assert 0 <= config.scene_classifier.confidence_threshold <= 1
        assert 0 <= config.clip.similarity_threshold <= 1
        assert 0 <= config.landmark_detector.confidence_threshold <= 1
        
        print("✓ Configuration validated successfully")
        return True
        
    except AssertionError as e:
        print(f"✗ Configuration validation failed: {e}")
        return False


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Setup environment variables and settings for optimal performance"""
    
    # PyTorch settings
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops for speed
    
    # Number of threads
    if config.device.device == "cpu":
        torch.set_num_threads(os.cpu_count())
    
    # Display settings
    os.environ["PYTHONUNBUFFERED"] = "1"  # Unbuffered output
    
    print(f"✓ Environment configured for {config.device.device}")


# ============================================================================
# QUICK START
# ============================================================================

if __name__ == "__main__":
    # Print configuration summary
    config.print_summary()
    
    # Validate configuration
    validate_config()
    
    # Setup environment
    setup_environment()
    
    # Print paths
    print("\nPaths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Models Directory: {MODELS_DIR}")
    print(f"  Checkpoints: {CHECKPOINTS_DIR}")
    print(f"  Cache: {CACHE_DIR}")
