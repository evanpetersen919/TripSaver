"""
CLIP Embeddings and FAISS Similarity Search
============================================

Visual similarity matching for niche locations using CLIP embeddings.
Converts images to high-dimensional vectors and finds similar images using FAISS.

Author: Evan Petersen
Date: November 2025
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import pickle
import json


# Try to import CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not installed. Install with: pip install faiss-gpu (or faiss-cpu)")


# ============================================================================
# CLIP EMBEDDER CLASS
# ============================================================================

class ClipEmbedder:
    """
    CLIP-based image embedding and similarity search.
    
    Uses OpenAI's CLIP model to convert images to embeddings, then FAISS
    for efficient similarity search across large image databases.
    
    Attributes:
        model: CLIP model
        preprocess: CLIP preprocessing function
        device: Computation device (cuda or cpu)
        index: FAISS index for similarity search
        metadata: Metadata for indexed images
    """
    
    def __init__(self, 
                 model_name: str = "ViT-B/32",
                 device: Optional[str] = None,
                 index_path: Optional[str] = None):
        """
        Initialize CLIP embedder.
        
        Args:
            model_name: CLIP model variant ('ViT-B/32', 'ViT-B/16', 'RN50')
            device: Device to run on ('cuda' or 'cpu')
            index_path: Path to load existing FAISS index
        """
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP is required. Install with: pip install git+https://github.com/openai/CLIP.git")
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required. Install with: pip install faiss-gpu")
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.visual.output_dim
        
        # Initialize FAISS index
        self.index = None
        self.metadata = []
        self.image_paths = []
        
        # Load existing index if provided
        if index_path:
            self.load_index(index_path)
        
        print(f"ClipEmbedder initialized on {self.device}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    
    # ========================================================================
    # EMBEDDING GENERATION
    # ========================================================================
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Convert image to CLIP embedding.
        
        Args:
            image: PIL Image
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        # Preprocess and convert to tensor
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # Normalize embedding
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        embedding = image_features.cpu().numpy().astype('float32')[0]
        return embedding
    
    
    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Convert multiple images to embeddings.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Array of embeddings (N x embedding_dim)
        """
        # Preprocess all images
        image_inputs = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        embeddings = image_features.cpu().numpy().astype('float32')
        return embeddings
    
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Convert text to CLIP embedding (for text-based search).
        
        Args:
            text: Text string or list of strings
            
        Returns:
            Normalized embedding vector(s)
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize text
        text_tokens = clip.tokenize(text).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        embeddings = text_features.cpu().numpy().astype('float32')
        return embeddings[0] if len(text) == 1 else embeddings
    
    
    # ========================================================================
    # FAISS INDEX MANAGEMENT
    # ========================================================================
    
    def build_index(self, use_gpu: bool = True):
        """
        Build FAISS index for similarity search.
        
        Args:
            use_gpu: Whether to use GPU for FAISS (if available)
        """
        # Check if GPU resources are available
        has_gpu = use_gpu and torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources')
        
        if has_gpu:
            try:
                # GPU index - faster search
                res = faiss.StandardGpuResources()
                index_flat = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
                self.index = faiss.index_cpu_to_gpu(res, 0, index_flat)
                print(f"FAISS index created: GPU")
            except Exception as e:
                print(f"GPU index failed, falling back to CPU: {e}")
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                print(f"FAISS index created: CPU")
        else:
            # CPU index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            print(f"FAISS index created: CPU")
    
    
    def add_image(self, image: Image.Image, metadata: Dict):
        """
        Add single image to index.
        
        Args:
            image: PIL Image
            metadata: Dictionary with image metadata (name, location, etc.)
        """
        if self.index is None:
            self.build_index()
        
        # Generate embedding
        embedding = self.encode_image(image)
        
        # Add to index
        self.index.add(embedding.reshape(1, -1))
        self.metadata.append(metadata)
    
    
    def add_images_batch(self, images: List[Image.Image], metadata_list: List[Dict]):
        """
        Add multiple images to index.
        
        Args:
            images: List of PIL Images
            metadata_list: List of metadata dicts
        """
        if self.index is None:
            self.build_index()
        
        # Generate embeddings
        embeddings = self.encode_batch(images)
        
        # Add to index
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)
    
    
    def add_from_directory(self, directory: str, 
                          metadata_file: Optional[str] = None,
                          extensions: Tuple[str] = ('.jpg', '.jpeg', '.png')):
        """
        Add all images from a directory to index.
        
        Args:
            directory: Path to directory containing images
            metadata_file: Optional JSON file with metadata
            extensions: Tuple of valid file extensions
        """
        directory_path = Path(directory)
        
        # Load metadata if available
        metadata_dict = {}
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(directory_path.glob(f"*{ext}"))
        
        print(f"Found {len(image_files)} images in {directory}")
        
        # Process in batches
        batch_size = 32
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            
            # Load images
            images = []
            batch_metadata = []
            
            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    
                    # Get metadata
                    img_name = img_path.name
                    meta = metadata_dict.get(img_name, {
                        'name': img_name,
                        'path': str(img_path)
                    })
                    batch_metadata.append(meta)
                    
                except Exception as e:
                    print(f"Failed to load {img_path}: {e}")
                    continue
            
            # Add batch to index
            if images:
                self.add_images_batch(images, batch_metadata)
                print(f"Processed {i+len(images)}/{len(image_files)} images")
        
        print(f"Index built with {self.index.ntotal} images")
    
    
    # ========================================================================
    # SIMILARITY SEARCH
    # ========================================================================
    
    def search(self, image: Image.Image, k: int = 5) -> List[Dict]:
        """
        Find k most similar images to query image.
        
        Args:
            image: Query image
            k: Number of results to return
            
        Returns:
            List of dicts with similarity scores and metadata
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty. Add images first.")
        
        # Generate query embedding
        query_embedding = self.encode_image(image).reshape(1, -1)
        
        # Search
        similarities, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity'] = float(sim)
                result['confidence'] = float((sim + 1) / 2)  # Convert to 0-1 range
                results.append(result)
        
        return results
    
    
    def search_by_text(self, text: str, k: int = 5) -> List[Dict]:
        """
        Find images matching text description.
        
        Args:
            text: Text query
            k: Number of results
            
        Returns:
            List of matching images with scores
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty. Add images first.")
        
        # Generate text embedding
        query_embedding = self.encode_text(text).reshape(1, -1)
        
        # Search
        similarities, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity'] = float(sim)
                result['confidence'] = float((sim + 1) / 2)
                results.append(result)
        
        return results
    
    
    # ========================================================================
    # SAVE/LOAD INDEX
    # ========================================================================
    
    def save_index(self, save_dir: str):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            save_dir: Directory to save index files
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            # Convert GPU index to CPU for saving
            if hasattr(faiss, 'index_gpu_to_cpu') and hasattr(self.index, 'getDevice'):
                try:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                except:
                    cpu_index = self.index
            else:
                cpu_index = self.index
            
            faiss.write_index(cpu_index, str(save_path / "index.faiss"))
        
        # Save metadata
        with open(save_path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'num_images': self.index.ntotal if self.index else 0
        }
        with open(save_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Index saved to {save_dir}")
    
    
    def load_index(self, load_dir: str):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            load_dir: Directory containing index files
        """
        load_path = Path(load_dir)
        
        # Load config
        with open(load_path / "config.json", 'r') as f:
            config = json.load(f)
        
        # Load FAISS index
        cpu_index = faiss.read_index(str(load_path / "index.faiss"))
        
        # Move to GPU if available
        if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            except:
                self.index = cpu_index
        else:
            self.index = cpu_index
        
        # Load metadata
        with open(load_path / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Index loaded: {self.index.ntotal} images")
    
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_index_size(self) -> int:
        """Get number of images in index."""
        return self.index.ntotal if self.index else 0
    
    
    def clear_index(self):
        """Clear all images from index."""
        self.index = None
        self.metadata = []
        print("Index cleared")
    
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ClipEmbedder(model='{self.model_name}', "
                f"device='{self.device}', "
                f"index_size={self.get_index_size()})")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Similarity score (0-1)
    """
    # Normalize embeddings
    emb1_norm = embedding1 / np.linalg.norm(embedding1)
    emb2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Cosine similarity
    similarity = np.dot(emb1_norm, emb2_norm)
    
    # Convert to 0-1 range
    return float((similarity + 1) / 2)
