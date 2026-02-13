"""
Base classes for patch embedding extractors.

This module defines the abstract interface that all patch embedding models
must implement to be compatible with the PROSPER pipeline.
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

# Lazy import openslide
openslide = None

def _ensure_openslide():
    """Lazily import openslide when needed."""
    global openslide
    if openslide is None:
        try:
            import openslide as _openslide
            openslide = _openslide
        except ImportError:
            raise ImportError(
                "openslide-python is required for patch embedding extraction. "
                "Install with: pip install openslide-python"
            )


class BasePatchEmbeddingExtractor(ABC):
    """
    Abstract base class for patch embedding extractors.
    
    All patch embedding models (Virchow2, UNI2, GigaPath, etc.) should inherit
    from this class and implement the required methods.
    """
    
    # Model metadata - subclasses should override these
    MODEL_NAME: str = "base"
    EMBEDDING_DIM: int = 0
    INPUT_SIZE: int = 224
    
    def __init__(
        self,
        device: str = 'cuda',
        batch_size: int = 64,
        num_workers: int = 8,
        verbose: bool = True
    ):
        """
        Initialize the base extractor.
        
        Args:
            device: Device for inference ('cuda' or 'cpu')
            batch_size: Batch size for processing
            num_workers: Number of workers for parallel data loading
            verbose: Whether to print progress
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        
        self.model = None
        self.transform = None
    
    @abstractmethod
    def load_model(self) -> Tuple[torch.nn.Module, Any]:
        """
        Load the model and preprocessing transform.
        
        Returns:
            Tuple of (model, transform)
        """
        pass
    
    @abstractmethod
    def extract_features(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract features from a batch of images.
        
        Args:
            batch_tensor: Batch of preprocessed images (B, C, H, W)
        
        Returns:
            Features array of shape (B, embedding_dim)
        """
        pass
    
    def _ensure_model_loaded(self):
        """Ensure the model is loaded (lazy loading)."""
        if self.model is None:
            self.model, self.transform = self.load_model()
            self.model = self.model.to(self.device)
            self.model.eval()
    
    def _read_and_transform_patch(
        self,
        slide,  # openslide.OpenSlide
        patch: Dict,
        base_patchsize: int
    ) -> torch.Tensor:
        """Read a patch from slide and apply transform."""
        w, h = patch['wsi_loc']
        patch_region = slide.read_region(
            (w, h), 0, (base_patchsize, base_patchsize)
        ).convert('RGB')
        return self.transform(patch_region).unsqueeze(0)
    
    def extract_slide_embeddings(
        self,
        slide_path: str,
        mosaic: List[Dict],
        base_patchsize: int
    ) -> Optional[np.ndarray]:
        """
        Extract embeddings for all patches in a slide's mosaic.
        
        Args:
            slide_path: Path to the slide file
            mosaic: List of mosaic patch dictionaries
            base_patchsize: Size of patches to extract
        
        Returns:
            Array of shape (n_patches, embedding_dim) or None if failed
        """
        _ensure_openslide()
        self._ensure_model_loaded()
        
        try:
            slide = openslide.OpenSlide(slide_path)
        except Exception as e:
            print(f"Failed to open slide {slide_path}: {e}")
            return None
        
        if mosaic is None or len(mosaic) == 0:
            print(f"No mosaic patches for slide {slide_path}")
            return None
        
        num_patches = len(mosaic)
        feature_queue = []
        batch_images = []
        
        with torch.no_grad():
            for idx, patch in enumerate(mosaic):
                w, h = patch['wsi_loc']
                patch_region = slide.read_region(
                    (w, h), 0, (base_patchsize, base_patchsize)
                ).convert('RGB')
                patch_tensor = self.transform(patch_region).to(self.device)
                batch_images.append(patch_tensor)
                
                if len(batch_images) == self.batch_size or idx == num_patches - 1:
                    batch_tensor = torch.stack(batch_images).to(self.device)
                    features = self.extract_features(batch_tensor)
                    feature_queue.extend(features)
                    batch_images.clear()
        
        if feature_queue:
            return np.array(feature_queue)
        return None
    
    def process_slides(
        self,
        slide_dir: str,
        mosaics: Dict[str, List[Dict]],
        output_dir: str,
        base_patchsize: int,
        force_recompute: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Process multiple slides to extract patch embeddings.
        
        Args:
            slide_dir: Directory containing slide files
            mosaics: Dictionary mapping slide IDs to mosaic patches
            output_dir: Directory to save embeddings
            base_patchsize: Size of patches
            force_recompute: Whether to recompute existing embeddings
        
        Returns:
            Dictionary mapping slide IDs to embedding arrays
        """
        tmpdir = os.path.join(output_dir, "tmp")
        os.makedirs(tmpdir, exist_ok=True)
        
        results = {}
        slides_to_process = []
        
        for slide_id, mosaic in mosaics.items():
            cache_path = os.path.join(tmpdir, f"{slide_id}.pkl")
            if not force_recompute and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        embeddings, _ = pickle.load(f)
                    results[slide_id] = embeddings
                    continue
                except Exception:
                    pass
            slides_to_process.append((slide_id, mosaic))
        
        if self.verbose:
            print(f"Model: {self.MODEL_NAME}")
            print(f"Total slides: {len(mosaics)}")
            print(f"Slides to process: {len(slides_to_process)}")
        
        iterator = slides_to_process
        if self.verbose:
            iterator = tqdm(iterator, desc=f"Extracting embeddings ({self.MODEL_NAME})")
        
        for slide_id, mosaic in iterator:
            slide_path = os.path.join(slide_dir, f"{slide_id}.svs")
            
            if not os.path.exists(slide_path):
                print(f"Slide not found: {slide_path}")
                continue
            
            embeddings = self.extract_slide_embeddings(slide_path, mosaic, base_patchsize)
            
            if embeddings is not None:
                results[slide_id] = embeddings
                
                cache_path = os.path.join(tmpdir, f"{slide_id}.pkl")
                with open(cache_path, 'wb') as f:
                    pickle.dump((embeddings, mosaic), f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return results
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], output_path: str):
        """Save embeddings to a pickle file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        if self.verbose:
            print(f"Saved embeddings for {len(embeddings)} slides to {output_path}")
    
    @staticmethod
    def load_embeddings(embeddings_path: str) -> Dict[str, np.ndarray]:
        """Load embeddings from a pickle file."""
        with open(embeddings_path, 'rb') as f:
            return pickle.load(f)


# Registry for patch embedding models
_PATCH_EMBEDDING_REGISTRY: Dict[str, type] = {}


def register_patch_embedding(name: str):
    """Decorator to register a patch embedding extractor class."""
    def decorator(cls):
        _PATCH_EMBEDDING_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_patch_embedding_extractor(
    model_name: str,
    device: str = 'cuda',
    batch_size: int = 64,
    num_workers: int = 8,
    verbose: bool = True,
    **kwargs
) -> BasePatchEmbeddingExtractor:
    """
    Factory function to get a patch embedding extractor by name.
    
    Args:
        model_name: Name of the model ('virchow2', 'uni2', 'gigapath')
        device: Device for inference
        batch_size: Batch size for processing
        num_workers: Number of workers
        verbose: Whether to print progress
        **kwargs: Additional model-specific arguments
    
    Returns:
        Instance of the requested patch embedding extractor
    
    Raises:
        ValueError: If model_name is not registered
    """
    name_lower = model_name.lower()
    if name_lower not in _PATCH_EMBEDDING_REGISTRY:
        available = list(_PATCH_EMBEDDING_REGISTRY.keys())
        raise ValueError(
            f"Unknown patch embedding model: '{model_name}'. "
            f"Available models: {available}"
        )
    
    cls = _PATCH_EMBEDDING_REGISTRY[name_lower]
    return cls(
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=verbose,
        **kwargs
    )


def list_patch_embedding_models() -> List[str]:
    """Return list of available patch embedding model names."""
    return list(_PATCH_EMBEDDING_REGISTRY.keys())
