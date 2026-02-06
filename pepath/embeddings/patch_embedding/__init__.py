"""
Patch Embedding Extraction Module.

This module provides a unified interface for extracting patch-level embeddings
using various pathology foundation models (Virchow2, UNI2, GigaPath, etc.).

Usage:
    from pepath.embeddings.patch_embedding import get_patch_embedding_extractor
    
    # Get extractor for a specific model
    extractor = get_patch_embedding_extractor("virchow2", device="cuda")
    
    # Extract embeddings for a slide
    embeddings = extractor.extract_slide_embeddings(slide_path, mosaic, patch_size)
    
    # List available models
    from pepath.embeddings.patch_embedding import list_patch_embedding_models
    print(list_patch_embedding_models())  # ['virchow2', 'uni2', 'uni', 'gigapath']
"""

# Import base classes and registry functions first
from .base import (
    BasePatchEmbeddingExtractor,
    get_patch_embedding_extractor,
    list_patch_embedding_models,
    register_patch_embedding,
)

# Import model implementations (this registers them in the registry)
from .virchow2_embeddings import (
    Virchow2EmbeddingExtractor,
    PatchEmbeddingExtractor,  # Backward compatibility alias
    load_virchow_model,
)
from .uni2_embedding import (
    UNI2EmbeddingExtractor,
    UNIEmbeddingExtractor,
    load_uni_model,
    load_univ2_model,
)
from .gigapath_embedding import (
    GigaPathEmbeddingExtractor,
    load_gigapath_model,
)


# Default model mapping for convenience
AVAILABLE_MODELS = {
    "virchow2": Virchow2EmbeddingExtractor,
    "uni2": UNI2EmbeddingExtractor,
    "uni": UNIEmbeddingExtractor,
    "gigapath": GigaPathEmbeddingExtractor,
}


__all__ = [
    # Factory and registry
    "get_patch_embedding_extractor",
    "list_patch_embedding_models",
    "register_patch_embedding",
    "AVAILABLE_MODELS",
    
    # Base class
    "BasePatchEmbeddingExtractor",
    
    # Model implementations
    "Virchow2EmbeddingExtractor",
    "UNI2EmbeddingExtractor",
    "UNIEmbeddingExtractor",
    "GigaPathEmbeddingExtractor",
    
    # Backward compatibility
    "PatchEmbeddingExtractor",
    "load_virchow_model",
    "load_uni_model",
    "load_univ2_model",
    "load_gigapath_model",
]
