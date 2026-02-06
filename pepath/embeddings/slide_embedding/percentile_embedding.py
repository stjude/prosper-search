"""
Percentile-based slide embedding generation.

This module aggregates patch-level features into slide-level representations
using percentile-based sampling for robust summarization.
"""

from typing import Dict, List, Optional, Any

import numpy as np

from .base import BaseSlideEmbeddingGenerator, register_slide_embedding


@register_slide_embedding("percentile")
class PercentileSlideEmbeddingGenerator(BaseSlideEmbeddingGenerator):
    """
    Generator for slide-level embeddings using percentile aggregation.
    
    Uses percentile-based aggregation to create robust slide representations
    that capture the distribution of morphological features across patches.
    
    The resulting embedding has shape (n_percentiles, feature_dim), which
    captures the feature distribution at multiple quantile levels.
    """
    
    METHOD_NAME = "percentile"
    PRODUCES_SIMILARITY = False
    
    def __init__(
        self,
        percentiles: Optional[List[int]] = None,
        feature_dim: int = 1280,
        n_workers: int = 32,
        verbose: bool = True
    ):
        """
        Initialize PercentileSlideEmbeddingGenerator.
        
        Args:
            percentiles: List of percentiles to compute (default: 10 evenly spaced)
            feature_dim: Dimension of patch features to use
            n_workers: Number of parallel workers
            verbose: Whether to print progress
        """
        super().__init__(feature_dim, n_workers, verbose)
        
        if percentiles is None:
            percentiles = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        self.percentiles = percentiles
    
    def process_single_slide(
        self,
        patch_embeddings: np.ndarray,
        slide_id: str,
        mosaic: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """
        Generate percentile-based slide embedding.
        
        Args:
            patch_embeddings: Array of patch embeddings (N, D)
            slide_id: Identifier for the slide
            mosaic: Not used for percentile method
        
        Returns:
            Slide embedding of shape (n_percentiles, feature_dim)
        """
        # Truncate to feature dimension if needed
        features = patch_embeddings[:, :self.feature_dim].astype(np.float16)
        
        # Compute percentiles across patches
        slide_embedding = np.percentile(features, self.percentiles, axis=0)
        
        return slide_embedding
    
    @property
    def embedding_dim(self) -> int:
        """Return the total dimension of the slide embedding."""
        return len(self.percentiles) * self.feature_dim
