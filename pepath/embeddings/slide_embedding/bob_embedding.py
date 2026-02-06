"""
Bag of Barcodes (BoB) slide embedding generation.

This module generates slide-level representations using binary barcodes
derived from patch embeddings. BoB enables efficient similarity computation
using Hamming distance.
"""

from typing import Dict, List, Optional, Any, Tuple

import numpy as np

try:
    import bitarray
    import bitarray.util as butil
    HAS_BITARRAY = True
except ImportError:
    HAS_BITARRAY = False

from .base import BaseSlideEmbeddingGenerator, register_slide_embedding


class BoB:
    """Bag of Barcodes representation for a slide."""
    
    def __init__(self, barcodes):
        """
        Initialize BoB from list of binary arrays.
        
        Args:
            barcodes: List of binary arrays (can be numpy or bitarray)
        """
        if HAS_BITARRAY:
            self.barcodes = [
                bitarray.bitarray(b.tolist()) if isinstance(b, np.ndarray) else b
                for b in barcodes
            ]
        else:
            self.barcodes = [np.array(b, dtype=np.uint8) for b in barcodes]
    
    def select_subset(self, n: int = 3) -> "BoB":
        """Select a random subset of n barcodes."""
        idx = np.arange(len(self.barcodes))
        np.random.shuffle(idx)
        idx = idx[:n]
        return BoB(barcodes=[self.barcodes[i] for i in idx])
    
    def distance(self, other: "BoB") -> float:
        """
        Compute BoB distance to another slide.
        
        Returns the median of minimum Hamming distances from each barcode
        in self to the closest barcode in other.
        """
        if HAS_BITARRAY:
            total_dist = []
            for feat in self.barcodes:
                distances = [butil.count_xor(feat, b) for b in other.barcodes]
                total_dist.append(np.min(distances))
            return np.median(total_dist)
        else:
            # Fallback using numpy
            total_dist = []
            for feat in self.barcodes:
                distances = [np.sum(feat != b) for b in other.barcodes]
                total_dist.append(np.min(distances))
            return np.median(total_dist)
    
    def to_int_array(self) -> np.ndarray:
        """Convert barcodes to integer array for storage."""
        if HAS_BITARRAY:
            return np.array([butil.ba2int(b) for b in self.barcodes])
        else:
            # Pack bits into integers
            return np.packbits(np.array(self.barcodes), axis=1).view(np.uint64)


def bob_from_int_array(int_array: np.ndarray, barcode_dim: int = 64) -> BoB:
    """Reconstruct BoB from integer array."""
    bob = BoB([])
    if HAS_BITARRAY:
        bob.barcodes = [butil.int2ba(int(x), length=barcode_dim) for x in int_array]
    else:
        # Unpack integers to bits
        bob.barcodes = [
            np.unpackbits(np.array([x], dtype=np.uint64).view(np.uint8))[-barcode_dim:]
            for x in int_array
        ]
    return bob


@register_slide_embedding("bob")
class BoBSlideEmbeddingGenerator(BaseSlideEmbeddingGenerator):
    """
    Generator for slide-level embeddings using Bag of Barcodes.
    
    Converts patch embeddings to binary barcodes using locality-sensitive
    hashing (LSH) for efficient similarity computation.
    """
    
    METHOD_NAME = "bob"
    PRODUCES_SIMILARITY = False
    
    def __init__(
        self,
        barcode_dim: int = 64,
        feature_dim: int = 1280,
        n_workers: int = 32,
        verbose: bool = True,
        projection_matrix: Optional[np.ndarray] = None
    ):
        """
        Initialize BoBSlideEmbeddingGenerator.
        
        Args:
            barcode_dim: Dimension of binary barcodes
            feature_dim: Dimension of patch features to use
            n_workers: Number of parallel workers
            verbose: Whether to print progress
            projection_matrix: Random projection matrix for LSH (optional)
        """
        super().__init__(feature_dim, n_workers, verbose)
        
        self.barcode_dim = barcode_dim
        
        if projection_matrix is None:
            # Create random projection matrix for LSH
            np.random.seed(42)  # Fixed seed for reproducibility
            self.projection_matrix = np.random.randn(feature_dim, barcode_dim)
            self.projection_matrix /= np.linalg.norm(
                self.projection_matrix, axis=0, keepdims=True
            )
        else:
            self.projection_matrix = projection_matrix
    
    def features_to_barcodes(self, features: np.ndarray) -> np.ndarray:
        """
        Convert feature vectors to binary barcodes using LSH.
        
        Args:
            features: Array of shape (N, feature_dim)
        
        Returns:
            Binary array of shape (N, barcode_dim)
        """
        # Random projection
        projected = features @ self.projection_matrix
        # Binarize
        barcodes = (projected > 0).astype(np.uint8)
        return barcodes
    
    def process_single_slide(
        self,
        patch_embeddings: np.ndarray,
        slide_id: str,
        mosaic: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate BoB slide representation.
        
        Args:
            patch_embeddings: Array of patch embeddings (N, D)
            slide_id: Identifier for the slide
            mosaic: Not used for BoB method
        
        Returns:
            Dictionary with BoB representation and metadata
        """
        # Truncate to feature dimension
        features = patch_embeddings[:, :self.feature_dim].astype(np.float32)
        
        # L2 normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features = features / norms
        
        # Generate barcodes
        barcodes = self.features_to_barcodes(features)
        
        # Create BoB object and convert to storable format
        bob = BoB(barcodes)
        
        return {
            'slide_id': slide_id,
            'barcodes': barcodes,
            'int_array': bob.to_int_array(),
            'n_patches': len(barcodes)
        }
    
    def compute_distance(
        self,
        bob_a: Dict[str, Any],
        bob_b: Dict[str, Any]
    ) -> float:
        """
        Compute BoB distance between two slides.
        
        Args:
            bob_a: BoB representation of first slide
            bob_b: BoB representation of second slide
        
        Returns:
            Distance value (median minimum Hamming distance)
        """
        barcodes_a = bob_a['barcodes']
        barcodes_b = bob_b['barcodes']
        
        # Compute XOR distance matrix
        # Shape: (n_a, n_b)
        xor_dists = np.zeros((len(barcodes_a), len(barcodes_b)))
        for i, ba in enumerate(barcodes_a):
            for j, bb in enumerate(barcodes_b):
                xor_dists[i, j] = np.sum(ba != bb)
        
        # Median of minimum distances
        min_dists = np.min(xor_dists, axis=1)
        return np.median(min_dists)
    
    def compute_distance_with_indices(
        self,
        bob_a: Dict[str, Any],
        bob_b: Dict[str, Any]
    ) -> Tuple[float, Any, Any]:
        """
        Compute BoB distance with representative patch indices.
        
        Args:
            bob_a: BoB representation of first slide
            bob_b: BoB representation of second slide
        
        Returns:
            Tuple of (distance, ref_patch_indices, query_patch_indices)
        """
        barcodes_a = bob_a['barcodes']
        barcodes_b = bob_b['barcodes']
        n_a, n_b = len(barcodes_a), len(barcodes_b)
        
        # Compute XOR distance matrix
        xor_dists = np.zeros((n_a, n_b))
        for i, ba in enumerate(barcodes_a):
            for j, bb in enumerate(barcodes_b):
                xor_dists[i, j] = np.sum(ba != bb)
        
        # Find minimum for each row
        min_sims = np.min(xor_dists, axis=1)
        argmins = np.argmin(xor_dists, axis=1)
        
        # Compute median and find representative indices
        median_value = np.median(min_sims)
        n = len(min_sims)
        order = np.argsort(min_sims)
        
        if n % 2 == 1:
            k = n // 2
            med_ref_idx = order[k]
            med_query_idx = argmins[med_ref_idx]
            return float(median_value), int(med_ref_idx), int(med_query_idx)
        else:
            k1, k2 = n // 2 - 1, n // 2
            r1, r2 = order[k1], order[k2]
            return float(median_value), [int(r1), int(r2)], [int(argmins[r1]), int(argmins[r2])]
