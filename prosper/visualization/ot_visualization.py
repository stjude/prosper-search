"""
Optimal Transport (OT) based visualization utilities.

This module provides functions for visualizing OT-based slide similarity,
including heatmap overlays showing patch-level contributions to similarity.
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from PIL import Image
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False


def transport_matrix_to_long_table(T: np.ndarray) -> pd.DataFrame:
    """
    Convert a transport matrix T into a long-format DataFrame.
    
    Args:
        T: Transport matrix of shape (n_a, n_b)
    
    Returns:
        DataFrame with columns: slide_a_patch_idx, slide_b_patch_idx, transport_value
        Only non-zero transport entries are kept.
    """
    i_indices, j_indices = np.nonzero(T)
    t_values = T[i_indices, j_indices]

    df = pd.DataFrame({
        'slide_a_patch_idx': i_indices,
        'slide_b_patch_idx': j_indices,
        'transport_value': t_values
    })

    df = df.sort_values(by='transport_value', ascending=False).reset_index(drop=True)
    return df


def compute_per_patch_weighted_distance_scores(
    T: np.ndarray,
    M: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-patch weighted average distance scores.
    
    Given transport matrix T and cost matrix M, compute weighted average
    distances for each patch in both slides.
    
    Args:
        T: Transport matrix of shape (n_a, n_b)
        M: Cost/distance matrix of shape (n_a, n_b)
    
    Returns:
        Tuple of (D_A, D_B) where:
        - D_A: Per-patch scores for slide A, shape (n_a,)
        - D_B: Per-patch scores for slide B, shape (n_b,)
    """
    D_A = np.sum(T * M, axis=1)  # shape (n_a,)
    D_B = np.sum(T * M, axis=0)  # shape (n_b,)
    return D_A, D_B


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] for heatmap visualization.
    
    Lower scores (stronger similarity) will be mapped to 0.
    
    Args:
        scores: Array of scores
    
    Returns:
        Normalized scores in [0, 1]
    """
    min_val = scores.min()
    max_val = scores.max()
    return (scores - min_val) / (max_val - min_val + 1e-8)


def scores_to_patch_dataframe(
    scores: np.ndarray,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Convert patch scores to a DataFrame for visualization.
    
    Args:
        scores: Per-patch scores
        normalize: Whether to normalize scores to [0, 1]
    
    Returns:
        DataFrame with patch_idx, weighted_distance, normalized_score columns
    """
    normalized = normalize_scores(scores) if normalize else scores
    
    return pd.DataFrame({
        'patch_idx': np.arange(len(scores)),
        'weighted_distance': scores,
        'normalized_score': normalized
    })


def generate_ot_heatmap_overlay(
    slide,
    slide_df: pd.DataFrame,
    mosaic: List[Dict],
    output_path: str,
    patch_size: int = 512,
    slide_level: int = 2,
    alpha: int = 80,
    cmap_continuous=None,
    cmap_discrete=None,
    invert_continuous: bool = True
) -> Optional[str]:
    """
    Create and save a heatmap overlay of OT-based patch similarity or cluster labels.
    
    Args:
        slide: OpenSlide object or path to slide file
        slide_df: DataFrame with 'patch_idx' and 'normalized_score' columns
        mosaic: Mosaic information with patch locations
        output_path: Path to save the overlay image
        patch_size: Size of patches in pixels
        slide_level: OpenSlide level to use for visualization
        alpha: Alpha value for heatmap overlay (0-255)
        cmap_continuous: Matplotlib colormap for continuous scores
        cmap_discrete: Matplotlib colormap for discrete labels
        invert_continuous: If True, invert continuous scores (lower = better)
    
    Returns:
        Path to saved image, or None if failed
    """
    if not HAS_MATPLOTLIB or not HAS_OPENSLIDE:
        print("matplotlib, PIL, and openslide required for visualization")
        return None
    
    if cmap_continuous is None:
        cmap_continuous = plt.cm.viridis
    if cmap_discrete is None:
        cmap_discrete = plt.cm.tab20
    
    # Handle slide path or OpenSlide object
    if isinstance(slide, str):
        slide = openslide.OpenSlide(slide)
    
    values = slide_df['normalized_score'].values
    
    # Check if discrete or continuous
    if np.issubdtype(values.dtype, np.integer) or np.allclose(values, np.round(values)):
        discrete_mode = True
        cmap = cmap_discrete
    else:
        discrete_mode = False
        cmap = cmap_continuous

    # Create heatmap grid
    level0_dims = slide.level_dimensions[0]
    heatmap_w = level0_dims[0] // patch_size + 1
    heatmap_h = level0_dims[1] // patch_size + 1
    heatmap = np.zeros((heatmap_w, heatmap_h), dtype=float)

    # Fill in heatmap values
    for idx, tile_info in enumerate(mosaic):
        if idx >= len(slide_df):
            continue
        score = values[idx]
        for x, y in tile_info.get('cluster_locs', [(tile_info['wsi_loc'][0], tile_info['wsi_loc'][1])]):
            xi = x // patch_size
            yi = y // patch_size
            if xi < heatmap_w and yi < heatmap_h:
                heatmap[xi, yi] = score

    mask = heatmap != 0

    if discrete_mode:
        # Build RGB image from discrete labels
        h_rgb = np.ones((heatmap.shape[0], heatmap.shape[1], 3), dtype=float)

        if hasattr(cmap, "colors"):
            palette = np.array(cmap.colors, dtype=float)
            if palette.shape[1] == 4:
                palette = palette[:, :3]
        else:
            slots = 20
            palette = np.array([cmap(i / max(slots - 1, 1))[:3] for i in range(slots)], dtype=float)

        labels = heatmap.astype(int)
        present = np.unique(labels[mask])

        for v in present:
            idx = (v - 1) % len(palette)
            color = palette[idx]
            h_rgb[labels == v] = color

        heatmap_rgb = (h_rgb * 255).astype(np.uint8)
    else:
        # Continuous mode with optional inversion
        normalized = np.zeros_like(heatmap, dtype=float)
        if np.any(mask):
            vals = heatmap[mask]
            normalized_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            if invert_continuous:
                normalized_vals = 1 - normalized_vals
            normalized[mask] = normalized_vals

        colored = cmap(normalized)[:, :, :3]
        colored[~mask] = [1, 1, 1]
        heatmap_rgb = (colored * 255).astype(np.uint8)

    heatmap_img = Image.fromarray(heatmap_rgb)

    # Resize heatmap to match slide level
    level_dims = slide.level_dimensions[slide_level]
    if slide_level < 3:
        resize_factor = int(patch_size / (4 ** slide_level))
    elif slide_level == 3:
        resize_factor = int(patch_size / (4 ** 2 * 2))
    else:
        resize_factor = int(patch_size / (4 ** 3))

    h_arr = np.array(heatmap_img)
    h_arr = np.repeat(np.repeat(h_arr, resize_factor, axis=0), resize_factor, axis=1)
    
    # Pad or crop to match slide dimensions
    pad_h = max(0, level_dims[1] - h_arr.shape[1])
    pad_w = max(0, level_dims[0] - h_arr.shape[0])
    if pad_h > 0 or pad_w > 0:
        h_arr = np.pad(h_arr, ((0, pad_w), (0, pad_h), (0, 0)), mode='constant', constant_values=255)
    h_arr = h_arr[:level_dims[0], :level_dims[1]]
    h_arr = np.transpose(h_arr, (1, 0, 2))

    # Create RGBA overlay
    rgba = np.zeros((h_arr.shape[0], h_arr.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = h_arr
    rgba[..., 3] = alpha
    heatmap_overlay = Image.fromarray(rgba)

    # Composite with slide image
    slide_img = slide.read_region((0, 0), slide_level, level_dims).convert("RGBA")
    composite = Image.alpha_composite(slide_img, heatmap_overlay)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    composite.save(output_path)
    
    return output_path


# Graph-based clustering for OT visualization

def sparsify_transport_matrix(
    T: np.ndarray,
    k: int = 4,
    method: str = "mutual_topk"
) -> np.ndarray:
    """
    Sparsify transport matrix by keeping only significant entries.
    
    Args:
        T: Transport matrix of shape (n_a, n_b)
        k: Number of top entries to keep per row/column
        method: Sparsification method ('topk', 'mutual_topk', 'threshold')
    
    Returns:
        Sparsified transport matrix
    """
    n_a, n_b = T.shape
    
    if method == "topk":
        T_sp = np.zeros_like(T)
        row_idx = np.arange(n_a)[:, None]
        topk_cols = np.argpartition(-T, kth=min(k, n_b-1), axis=1)[:, :k]
        T_sp[row_idx, topk_cols] = T[row_idx, topk_cols]
        return T_sp
    
    elif method == "mutual_topk":
        # Top-k per row
        row_topk = np.argpartition(-T, kth=min(k, n_b-1), axis=1)[:, :k]
        row_mask = np.zeros_like(T, dtype=bool)
        row_mask[np.arange(n_a)[:, None], row_topk] = True
        
        # Top-k per column
        col_topk = np.argpartition(-T, kth=min(k, n_a-1), axis=0)[:k, :]
        col_mask = np.zeros_like(T, dtype=bool)
        col_mask[col_topk, np.arange(n_b)[None, :]] = True
        
        # Mutual (intersection)
        mutual = row_mask & col_mask
        return np.where(mutual, T, 0.0)
    
    else:  # threshold
        threshold = np.percentile(T[T > 0], 90) if np.any(T > 0) else 0
        return np.where(T >= threshold, T, 0.0)


def cluster_patches_from_transport(
    T: np.ndarray,
    method: str = "connected_components",
    resolution: float = 1.0,
    sparsify_k: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster patches from both slides based on transport matrix.
    
    Args:
        T: Transport matrix of shape (n_a, n_b)
        method: Clustering method ('connected_components' or 'louvain')
        resolution: Resolution parameter for Louvain clustering
        sparsify_k: Number of neighbors for sparsification
    
    Returns:
        Tuple of (df_a, df_b) DataFrames with patch_idx and cluster_id columns
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required for graph-based clustering")
    
    # Sparsify transport matrix
    T_sp = sparsify_transport_matrix(T, k=sparsify_k, method="mutual_topk")
    
    n_a, n_b = T.shape
    
    # Build bipartite graph
    G = nx.Graph()
    G.add_nodes_from(range(n_a), bipartite=0, slide='A')
    G.add_nodes_from(range(n_a, n_a + n_b), bipartite=1, slide='B')
    
    i_idx, j_idx = np.nonzero(T_sp)
    for i, j in zip(i_idx, j_idx):
        G.add_edge(i, n_a + j, weight=float(T_sp[i, j]))
    
    if method == "connected_components":
        comps = list(nx.connected_components(G))
        partition = {}
        for cid, comp in enumerate(comps, start=1):
            for u in comp:
                partition[u] = cid
    else:  # louvain
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, weight="weight", resolution=resolution, seed=0)
            partition = {}
            for cid, nodes in enumerate(communities, start=1):
                for u in nodes:
                    partition[u] = cid
        except ImportError:
            try:
                import community.community_louvain as community_louvain
                partition = community_louvain.best_partition(G, weight="weight", resolution=resolution, random_state=0)
            except ImportError:
                raise ImportError("Louvain clustering requires python-louvain or networkx >= 2.7")
    
    # Convert to DataFrames
    a_records, b_records = [], []
    for node, cid in partition.items():
        if node < n_a:
            a_records.append((node, cid))
        else:
            b_records.append((node - n_a, cid))
    
    df_a = pd.DataFrame(a_records, columns=["patch_idx", "normalized_score"])
    df_b = pd.DataFrame(b_records, columns=["patch_idx", "normalized_score"])
    
    return df_a, df_b


class OTVisualizer:
    """
    Visualizer for Optimal Transport-based slide similarity.
    
    Provides methods for generating heatmap overlays and analyzing
    patch-level contributions to similarity.
    """
    
    def __init__(
        self,
        patch_size: int = 512,
        slide_level: int = 2,
        alpha: int = 100,
        output_dir: Optional[str] = None
    ):
        """
        Initialize OT visualizer.
        
        Args:
            patch_size: Size of patches in pixels
            slide_level: OpenSlide level for visualization
            alpha: Alpha value for heatmap overlay
            output_dir: Default output directory
        """
        self.patch_size = patch_size
        self.slide_level = slide_level
        self.alpha = alpha
        self.output_dir = output_dir or "./visualizations"
    
    def visualize_similarity(
        self,
        slide_a_path: str,
        slide_b_path: str,
        ot_result: Dict[str, Any],
        mosaic_a: List[Dict],
        mosaic_b: List[Dict],
        output_dir: Optional[str] = None,
        method: str = "weighted_distance"
    ) -> Dict[str, str]:
        """
        Generate visualization for OT similarity between two slides.
        
        Args:
            slide_a_path: Path to first slide
            slide_b_path: Path to second slide
            ot_result: OT computation result with 'M', 'T', 'distance' keys
            mosaic_a: Mosaic for first slide
            mosaic_b: Mosaic for second slide
            output_dir: Output directory (uses default if not provided)
            method: Scoring method ('weighted_distance' or 'cluster')
        
        Returns:
            Dictionary with paths to generated visualization files
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        T = ot_result['T']
        M = ot_result['M']
        slide_a_id = ot_result.get('slide_a_id', 'slide_a')
        slide_b_id = ot_result.get('slide_b_id', 'slide_b')
        
        results = {}
        
        if method == "weighted_distance":
            # Compute per-patch weighted distances
            D_A, D_B = compute_per_patch_weighted_distance_scores(T, M)
            df_a = scores_to_patch_dataframe(D_A)
            df_b = scores_to_patch_dataframe(D_B)
        else:  # cluster
            df_a, df_b = cluster_patches_from_transport(T)
        
        # Generate heatmaps
        path_a = os.path.join(output_dir, f"heatmap_{slide_a_id}.png")
        path_b = os.path.join(output_dir, f"heatmap_{slide_b_id}.png")
        
        generate_ot_heatmap_overlay(
            slide_a_path, df_a, mosaic_a, path_a,
            patch_size=self.patch_size,
            slide_level=self.slide_level,
            alpha=self.alpha
        )
        results['slide_a_heatmap'] = path_a
        
        generate_ot_heatmap_overlay(
            slide_b_path, df_b, mosaic_b, path_b,
            patch_size=self.patch_size,
            slide_level=self.slide_level,
            alpha=self.alpha
        )
        results['slide_b_heatmap'] = path_b
        
        return results
