"""
Embedding Extraction and Generation Module for PROSPER Pipeline.

This module provides unified interfaces for:
1. Patch-level embedding extraction using foundation models (Virchow2, UNI2, GigaPath)
2. Slide-level embedding generation using various aggregation methods (Percentile, BoB, OT)

Usage:
    # Patch embeddings
    from prosper.embeddings import get_patch_embedding_extractor
    extractor = get_patch_embedding_extractor("virchow2", device="cuda")
    
    # Slide embeddings
    from prosper.embeddings import get_slide_embedding_generator
    generator = get_slide_embedding_generator("percentile")
"""

# Patch embedding exports
from .patch_embedding import (
    # Factory functions
    get_patch_embedding_extractor,
    list_patch_embedding_models,
    register_patch_embedding,
    
    # Base class
    BasePatchEmbeddingExtractor,
    
    # Model implementations
    Virchow2EmbeddingExtractor,
    UNI2EmbeddingExtractor,
    UNIEmbeddingExtractor,
    GigaPathEmbeddingExtractor,
    
    # Backward compatibility
    PatchEmbeddingExtractor,
    load_virchow_model,
)

# Slide embedding exports
from .slide_embedding import (
    # Factory functions
    get_slide_embedding_generator,
    list_slide_embedding_methods,
    register_slide_embedding,
    
    # Base class
    BaseSlideEmbeddingGenerator,
    
    # Method implementations
    PercentileSlideEmbeddingGenerator,
    BoBSlideEmbeddingGenerator,
    OTSlideEmbeddingGenerator,
    
    # Backward compatibility
    SlideEmbeddingGenerator,
)


__all__ = [
    # Patch embedding factory
    "get_patch_embedding_extractor",
    "list_patch_embedding_models",
    "register_patch_embedding",
    "BasePatchEmbeddingExtractor",
    
    # Patch embedding models
    "Virchow2EmbeddingExtractor",
    "UNI2EmbeddingExtractor",
    "UNIEmbeddingExtractor",
    "GigaPathEmbeddingExtractor",
    
    # Slide embedding factory
    "get_slide_embedding_generator",
    "list_slide_embedding_methods",
    "register_slide_embedding",
    "BaseSlideEmbeddingGenerator",
    
    # Slide embedding methods
    "PercentileSlideEmbeddingGenerator",
    "BoBSlideEmbeddingGenerator",
    "OTSlideEmbeddingGenerator",
    
    # Backward compatibility
    "PatchEmbeddingExtractor",
    "SlideEmbeddingGenerator",
    "load_virchow_model",
]
