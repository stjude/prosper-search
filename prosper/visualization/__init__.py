"""
Visualization Module for PROSPER Pipeline.

This module provides visualization utilities for search results and
similarity analysis, including OT-based heatmap overlays and ROI search heatmaps.

Usage:
    # OT Visualization
    from prosper.visualization import OTVisualizer
    
    visualizer = OTVisualizer(patch_size=512, slide_level=2)
    results = visualizer.visualize_similarity(
        slide_a_path, slide_b_path,
        ot_result, mosaic_a, mosaic_b
    )
    
    # ROI Search Heatmap
    from prosper.visualization import generate_roi_search_heatmap
    
    output_path = generate_roi_search_heatmap(
        embeddings_dir="data/patch_embeddings",
        query_slide_id="4261",
        query_patch_idx=860,
        hit_slide_id="4264",
        output_dir="output/heatmaps"
    )
"""

from .ot_visualization import (
    OTVisualizer,
    generate_ot_heatmap_overlay,
    transport_matrix_to_long_table,
    compute_per_patch_weighted_distance_scores,
    normalize_scores,
    scores_to_patch_dataframe,
    sparsify_transport_matrix,
    cluster_patches_from_transport,
)

from .roi_heatmap import (
    generate_roi_heatmap,
    generate_roi_heatmap_simple,
    generate_roi_search_heatmap,
    generate_heatmap_for_thumbnail,
    load_embedding_file,
    load_mosaic_metadata,
    load_thumbnail,
    load_slide_dimensions,
    get_slide_dimensions,
    overlay_heatmap_on_thumbnail,
    compute_patch_similarities,
)


__all__ = [
    # Main visualizer class
    "OTVisualizer",
    
    # OT Heatmap functions
    "generate_ot_heatmap_overlay",
    
    # ROI Search Heatmap functions
    "generate_roi_heatmap",
    "generate_roi_heatmap_simple",
    "generate_roi_search_heatmap",
    "generate_heatmap_for_thumbnail",
    "load_embedding_file",
    "load_mosaic_metadata",
    "load_thumbnail",
    "load_slide_dimensions",
    "get_slide_dimensions",
    "overlay_heatmap_on_thumbnail",
    "compute_patch_similarities",
    
    # Score computation
    "transport_matrix_to_long_table",
    "compute_per_patch_weighted_distance_scores",
    "normalize_scores",
    "scores_to_patch_dataframe",
    
    # Clustering
    "sparsify_transport_matrix",
    "cluster_patches_from_transport",
]
