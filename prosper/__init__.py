"""
PROSPER: Pediatric Retrieval for Oncology Slides, Pathology Evidence, and Reporting

A production-grade pipeline for whole slide image (WSI) embedding and search,
specifically designed for pediatric pathology applications.

Features:
- Multiple patch embedding models: Virchow2, UNI2, UNI, GigaPath
- Multiple slide similarity formulations: Percentile, BoB (Bag of Barcodes), OT (Optimal Transport)
- Slide-level and ROI-level similarity search
- OT-based visualization of patch-level similarity

Usage:
    from prosper import ProsperPipeline, ProsperConfig
    
    config = ProsperConfig(
        slide_path="/path/to/slides",
        output_dir="./output",
        patch_model="virchow2",           # Options: virchow2, uni2, uni, gigapath
        slide_embedding_method="percentile"  # Options: percentile, bob, ot
    )
    
    pipeline = ProsperPipeline(config)
    results = pipeline.run()
    
    # List available models
    print(ProsperPipeline.list_available_models())

Author: Haoran Chen
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Haoran Chen"

from .pipeline import ProsperPipeline, ProsperResults
from .config import ProsperConfig

# Backward compatibility aliases (from previous names)
COMETPipeline = ProsperPipeline
COMETConfig = ProsperConfig
COMETResults = ProsperResults
PePathPipeline = ProsperPipeline
PePathConfig = ProsperConfig
PePathResults = ProsperResults

# Re-export embedding factory functions for convenience
from .embeddings import (
    get_patch_embedding_extractor,
    get_slide_embedding_generator,
    list_patch_embedding_models,
    list_slide_embedding_methods,
)

__all__ = [
    # Main classes
    "ProsperPipeline",
    "ProsperConfig",
    "ProsperResults",
    
    # Backward compatibility
    "COMETPipeline",
    "COMETConfig",
    "COMETResults",
    "PePathPipeline",
    "PePathConfig",
    "PePathResults",
    
    # Factory functions
    "get_patch_embedding_extractor",
    "get_slide_embedding_generator",
    "list_patch_embedding_models",
    "list_slide_embedding_methods",
    
    # Metadata
    "__version__",
    "__author__",
]
