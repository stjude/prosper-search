"""
Base classes for slide embedding generators.

This module defines the abstract interface that all slide embedding methods
must implement to be compatible with the COMET pipeline.
"""

import os
import pickle
from abc import ABC, abstractmethod
from glob import glob
from typing import Dict, List, Optional, Tuple, Any
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm


class BaseSlideEmbeddingGenerator(ABC):
    """
    Abstract base class for slide embedding generators.
    
    All slide embedding methods (Percentile, BoB, OT, etc.) should inherit
    from this class and implement the required methods.
    """
    
    # Method metadata - subclasses should override these
    METHOD_NAME: str = "base"
    PRODUCES_SIMILARITY: bool = False  # If True, generates pairwise similarities instead of embeddings
    
    def __init__(
        self,
        feature_dim: int = 1280,
        n_workers: int = 32,
        verbose: bool = True
    ):
        """
        Initialize the base generator.
        
        Args:
            feature_dim: Dimension of patch features to use
            n_workers: Number of parallel workers
            verbose: Whether to print progress
        """
        self.feature_dim = feature_dim
        self.n_workers = n_workers
        self.verbose = verbose
    
    @abstractmethod
    def process_single_slide(
        self,
        patch_embeddings: np.ndarray,
        slide_id: str,
        mosaic: Optional[List[Dict]] = None
    ) -> Any:
        """
        Process a single slide's patch embeddings.
        
        Args:
            patch_embeddings: Array of patch embeddings (N, D)
            slide_id: Identifier for the slide
            mosaic: Optional mosaic information for spatial context
        
        Returns:
            Processed representation (embedding or other)
        """
        pass
    
    def generate_embeddings(
        self,
        patch_embeddings_dir: str,
        output_dir: str,
        force_recompute: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate slide embeddings from patch embeddings.
        
        Args:
            patch_embeddings_dir: Directory containing patch embedding files
            output_dir: Directory to save slide embeddings
            force_recompute: Whether to recompute existing embeddings
        
        Returns:
            Dictionary mapping slide IDs to slide embeddings
        """
        os.makedirs(output_dir, exist_ok=True)
        
        input_files = glob(os.path.join(patch_embeddings_dir, '*.pkl'))
        
        args_list = []
        for input_path in input_files:
            slide_id = os.path.basename(input_path).replace('.pkl', '')
            output_path = os.path.join(output_dir, f"{slide_id}.pkl")
            
            if not force_recompute and os.path.exists(output_path):
                continue
            
            args_list.append((input_path, output_path))
        
        if self.verbose:
            print(f"Method: {self.METHOD_NAME}")
            print(f"Total slides: {len(input_files)}")
            print(f"Slides to process: {len(args_list)}")
        
        # Process slides
        results = {}
        if len(args_list) > 0:
            if self.n_workers > 1:
                with Pool(self.n_workers) as pool:
                    iterator = pool.imap_unordered(self._process_single_file, args_list)
                    if self.verbose:
                        iterator = tqdm(iterator, total=len(args_list),
                                      desc=f"Generating slide embeddings ({self.METHOD_NAME})")
                    for result in iterator:
                        if result is not None:
                            results[result[0]] = result[1]
            else:
                iterator = args_list
                if self.verbose:
                    iterator = tqdm(iterator, desc=f"Generating slide embeddings ({self.METHOD_NAME})")
                for args in iterator:
                    result = self._process_single_file(args)
                    if result is not None:
                        results[result[0]] = result[1]
        
        # Load all embeddings (including previously computed)
        all_results = self.load_all_embeddings(output_dir)
        all_results.update(results)
        
        return all_results
    
    def _process_single_file(self, args) -> Optional[Tuple[str, np.ndarray]]:
        """Process a single slide's patch embeddings file."""
        input_path, output_path = args
        
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different formats
            if isinstance(data, tuple):
                features = data[0]
                mosaic = data[1] if len(data) > 1 else None
            else:
                features = data
                mosaic = None
            
            # Truncate to feature dimension
            features = features[:, :self.feature_dim].astype(np.float32)
            
            slide_id = os.path.basename(input_path).replace('.pkl', '')
            slide_embedding = self.process_single_slide(features, slide_id, mosaic)
            
            # Save result
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(slide_embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            return (slide_id, slide_embedding)
        
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return None
    
    def load_all_embeddings(self, embeddings_dir: str) -> Dict[str, np.ndarray]:
        """Load all slide embeddings from a directory."""
        results = {}
        files = glob(os.path.join(embeddings_dir, '*.pkl'))
        
        for filepath in files:
            slide_id = os.path.basename(filepath).replace('.pkl', '')
            try:
                with open(filepath, 'rb') as f:
                    results[slide_id] = pickle.load(f)
            except Exception:
                pass
        
        return results
    
    def create_embedding_matrix(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """Create a matrix of slide embeddings for efficient search."""
        slide_ids = sorted(embeddings.keys())
        matrix = np.stack([embeddings[sid].flatten() for sid in slide_ids])
        return matrix, slide_ids
    
    def save_embedding_matrix(
        self,
        embeddings: Dict[str, np.ndarray],
        output_path: str
    ):
        """Save embedding matrix and slide IDs to file."""
        matrix, slide_ids = self.create_embedding_matrix(embeddings)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({'matrix': matrix, 'slide_ids': slide_ids}, f)
        
        if self.verbose:
            print(f"Saved embedding matrix of shape {matrix.shape} to {output_path}")


# Registry for slide embedding methods
_SLIDE_EMBEDDING_REGISTRY: Dict[str, type] = {}


def register_slide_embedding(name: str):
    """Decorator to register a slide embedding generator class."""
    def decorator(cls):
        _SLIDE_EMBEDDING_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_slide_embedding_generator(
    method_name: str,
    feature_dim: int = 1280,
    n_workers: int = 32,
    verbose: bool = True,
    **kwargs
) -> BaseSlideEmbeddingGenerator:
    """
    Factory function to get a slide embedding generator by name.
    
    Args:
        method_name: Name of the method ('percentile', 'bob', 'ot')
        feature_dim: Dimension of patch features to use
        n_workers: Number of parallel workers
        verbose: Whether to print progress
        **kwargs: Additional method-specific arguments
    
    Returns:
        Instance of the requested slide embedding generator
    
    Raises:
        ValueError: If method_name is not registered
    """
    name_lower = method_name.lower()
    if name_lower not in _SLIDE_EMBEDDING_REGISTRY:
        available = list(_SLIDE_EMBEDDING_REGISTRY.keys())
        raise ValueError(
            f"Unknown slide embedding method: '{method_name}'. "
            f"Available methods: {available}"
        )
    
    cls = _SLIDE_EMBEDDING_REGISTRY[name_lower]
    return cls(
        feature_dim=feature_dim,
        n_workers=n_workers,
        verbose=verbose,
        **kwargs
    )


def list_slide_embedding_methods() -> List[str]:
    """Return list of available slide embedding method names."""
    return list(_SLIDE_EMBEDDING_REGISTRY.keys())
