"""
Slide Embedding Generation Module.

This module provides a unified interface for generating slide-level embeddings
from patch embeddings using various aggregation methods:
- Percentile: Statistical percentile-based aggregation
- BoB (Bag of Barcodes): Binary barcode-based representation
- OT (Optimal Transport): Wasserstein distance-based similarity

Usage:
    from pepath.embeddings.slide_embedding import get_slide_embedding_generator
    
    # Get generator for a specific method
    generator = get_slide_embedding_generator("percentile", feature_dim=1280)
    
    # Generate slide embeddings
    embeddings = generator.generate_embeddings(patch_embeddings_dir, output_dir)
    
    # List available methods
    from pepath.embeddings.slide_embedding import list_slide_embedding_methods
    print(list_slide_embedding_methods())  # ['percentile', 'bob', 'ot']
"""

from .base import (
    BaseSlideEmbeddingGenerator,
    get_slide_embedding_generator,
    list_slide_embedding_methods,
    register_slide_embedding,
)

from .percentile_embedding import PercentileSlideEmbeddingGenerator
from .bob_embedding import BoBSlideEmbeddingGenerator
from .ot_embedding import OTSlideEmbeddingGenerator

# Backward compatibility alias
SlideEmbeddingGenerator = PercentileSlideEmbeddingGenerator


AVAILABLE_METHODS = {
    "percentile": PercentileSlideEmbeddingGenerator,
    "bob": BoBSlideEmbeddingGenerator,
    "ot": OTSlideEmbeddingGenerator,
}


__all__ = [
    # Factory and registry
    "get_slide_embedding_generator",
    "list_slide_embedding_methods",
    "register_slide_embedding",
    "AVAILABLE_METHODS",
    
    # Base class
    "BaseSlideEmbeddingGenerator",
    
    # Method implementations
    "PercentileSlideEmbeddingGenerator",
    "BoBSlideEmbeddingGenerator",
    "OTSlideEmbeddingGenerator",
    
    # Backward compatibility
    "SlideEmbeddingGenerator",
]
