"""
Optimal Transport (OT) based slide similarity computation.

This module computes slide-level similarities using Optimal Transport
(Wasserstein distance) between patch embedding distributions.
"""

import os
import pickle
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False

from .base import BaseSlideEmbeddingGenerator, register_slide_embedding


@register_slide_embedding("ot")
class OTSlideEmbeddingGenerator(BaseSlideEmbeddingGenerator):
    """
    Generator for slide-level similarity using Optimal Transport.
    
    Instead of generating a fixed embedding, this method computes
    pairwise similarities between slides using the Wasserstein distance
    between their patch embedding distributions.
    
    This produces more accurate similarity measures at the cost of
    O(n^2) pairwise computations.
    """
    
    METHOD_NAME = "ot"
    PRODUCES_SIMILARITY = True
    
    def __init__(
        self,
        metric: str = "cosine",
        reg: float = 0.0,
        feature_dim: int = 1280,
        n_workers: int = 32,
        verbose: bool = True,
        max_iter: int = 1000000
    ):
        """
        Initialize OTSlideEmbeddingGenerator.
        
        Args:
            metric: Distance metric for OT cost matrix ('cosine', 'euclidean')
            reg: Regularization for entropic OT (0 = exact EMD)
            feature_dim: Dimension of patch features to use
            n_workers: Number of parallel workers
            verbose: Whether to print progress
            max_iter: Maximum iterations for OT solver
        """
        super().__init__(feature_dim, n_workers, verbose)
        
        if not HAS_POT:
            raise ImportError(
                "POT (Python Optimal Transport) library required for OT method. "
                "Install with: pip install POT"
            )
        
        self.metric = metric
        self.reg = reg
        self.max_iter = max_iter
    
    def process_single_slide(
        self,
        patch_embeddings: np.ndarray,
        slide_id: str,
        mosaic: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Prepare slide features for OT computation.
        
        For OT, we store the normalized features rather than a single embedding,
        as pairwise computation is needed.
        
        Args:
            patch_embeddings: Array of patch embeddings (N, D)
            slide_id: Identifier for the slide
            mosaic: Mosaic information (stored for visualization)
        
        Returns:
            Dictionary with normalized features and metadata
        """
        # Truncate to feature dimension
        features = patch_embeddings[:, :self.feature_dim].astype(np.float32)
        
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features_normalized = features / norms
        
        return {
            'slide_id': slide_id,
            'features': features_normalized,
            'features_raw': features,
            'n_patches': len(features),
            'mosaic': mosaic
        }
    
    def compute_ot_distance(
        self,
        feats_a: np.ndarray,
        feats_b: np.ndarray,
        return_transport: bool = False
    ) -> Any:
        """
        Compute OT distance between two sets of features.
        
        Args:
            feats_a: Features from first slide (N, D)
            feats_b: Features from second slide (M, D)
            return_transport: Whether to return transport matrix
        
        Returns:
            Distance value, or dict with distance, cost matrix, and transport matrix
        """
        # Uniform weights
        a = np.ones(len(feats_a)) / len(feats_a)
        b = np.ones(len(feats_b)) / len(feats_b)
        
        # Cost matrix
        M = ot.dist(feats_a, feats_b, metric=self.metric)
        
        if self.reg > 0:
            # Entropic regularization (faster but approximate)
            if return_transport:
                T = ot.sinkhorn(a, b, M, self.reg, numItermax=self.max_iter)
                dist = np.sum(T * M)
                return {'distance': dist, 'M': M, 'T': T}
            else:
                dist = ot.sinkhorn2(a, b, M, self.reg, numItermax=self.max_iter)
                return dist
        else:
            # Exact Earth Mover's Distance
            if return_transport:
                T = ot.emd(a, b, M, numItermax=self.max_iter)
                dist = np.sum(T * M)
                return {'distance': dist, 'M': M, 'T': T}
            else:
                dist = ot.emd2(a, b, M, numItermax=self.max_iter)
                return dist
    
    def compute_pairwise_similarities(
        self,
        slide_data: Dict[str, Dict],
        output_dir: str,
        slide_id: Optional[str] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute pairwise OT similarities between slides.
        
        Args:
            slide_data: Dictionary mapping slide IDs to processed data
            output_dir: Directory to save results
            slide_id: If provided, only compute similarities for this slide
        
        Returns:
            Dictionary mapping slide IDs to list of (other_id, distance) tuples
        """
        from tqdm import tqdm
        
        os.makedirs(output_dir, exist_ok=True)
        slide_ids = sorted(slide_data.keys())
        
        results = {}
        
        if slide_id is not None:
            # Compute for single slide vs all others
            query_ids = [slide_id]
        else:
            query_ids = slide_ids
        
        for qid in tqdm(query_ids, desc="Computing OT similarities", disable=not self.verbose):
            output_path = os.path.join(output_dir, f'similarity_{qid}.pkl')
            
            if os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    results[qid] = pickle.load(f)
                continue
            
            query_features = slide_data[qid]['features']
            similarities = []
            
            for oid in slide_ids:
                if oid == qid:
                    dist = 1e6  # Self-match placeholder
                else:
                    try:
                        other_features = slide_data[oid]['features']
                        dist = self.compute_ot_distance(query_features, other_features)
                    except Exception as e:
                        if self.verbose:
                            print(f"Failed comparing {qid} to {oid}: {e}")
                        dist = 1e6
                
                similarities.append((oid, float(dist)))
            
            results[qid] = similarities
            
            # Save intermediate result
            with open(output_path, 'wb') as f:
                pickle.dump(similarities, f)
        
        return results
    
    def compute_detailed_similarity(
        self,
        slide_a_data: Dict,
        slide_b_data: Dict,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute detailed OT similarity with transport matrix for visualization.
        
        Args:
            slide_a_data: Processed data for first slide
            slide_b_data: Processed data for second slide
            output_path: Optional path to save results
        
        Returns:
            Dictionary with distance, cost matrix M, and transport matrix T
        """
        feats_a = slide_a_data['features']
        feats_b = slide_b_data['features']
        
        result = self.compute_ot_distance(feats_a, feats_b, return_transport=True)
        result['slide_a_id'] = slide_a_data['slide_id']
        result['slide_b_id'] = slide_b_data['slide_id']
        
        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
        
        return result
