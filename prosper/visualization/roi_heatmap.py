"""
ROI Search Heatmap Visualization.

Generates standalone heatmaps for ROI search results following the same
approach as the original demo notebook - using cluster_locs to fill all
tiles in each mosaic cluster with the same similarity value.

Supports overlay on slide thumbnails for publication-quality visualizations.
No SVS file required - works with embedding files, mosaic metadata, and thumbnails.

Uses slide_dimensions.csv for accurate coordinate mapping between slide and thumbnail.
"""

import os
import pickle
import argparse
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# Global cache for slide dimensions
_SLIDE_DIMENSIONS_CACHE = None
_SLIDE_DIMENSIONS_PATH = None


def load_slide_dimensions(csv_path: str) -> Dict[str, Tuple[int, int]]:
    """
    Load slide dimensions from CSV file.
    
    Args:
        csv_path: Path to slide_dimensions.csv with columns: slide_id, width, height
    
    Returns:
        Dictionary mapping slide_id (str) to (width, height) tuple
    """
    global _SLIDE_DIMENSIONS_CACHE, _SLIDE_DIMENSIONS_PATH
    
    # Return cached version if same path
    if _SLIDE_DIMENSIONS_CACHE is not None and _SLIDE_DIMENSIONS_PATH == csv_path:
        return _SLIDE_DIMENSIONS_CACHE
    
    df = pd.read_csv(csv_path)
    dimensions = {}
    for _, row in df.iterrows():
        slide_id = str(row['slide_id'])
        dimensions[slide_id] = (int(row['width']), int(row['height']))
    
    _SLIDE_DIMENSIONS_CACHE = dimensions
    _SLIDE_DIMENSIONS_PATH = csv_path
    
    return dimensions


def get_slide_dimensions(slide_id: str, dimensions_csv: str = None) -> Optional[Tuple[int, int]]:
    """
    Get slide dimensions for a specific slide.
    
    Args:
        slide_id: Slide identifier
        dimensions_csv: Path to slide_dimensions.csv
    
    Returns:
        (width, height) tuple or None if not found
    """
    if dimensions_csv is None:
        return None
    
    if not os.path.exists(dimensions_csv):
        print(f"Warning: Slide dimensions file not found: {dimensions_csv}")
        return None
    
    dimensions = load_slide_dimensions(dimensions_csv)
    return dimensions.get(str(slide_id))


def load_thumbnail(thumbnails_dir: str, slide_id: str) -> Optional[Image.Image]:
    """
    Load slide thumbnail image.
    
    Args:
        thumbnails_dir: Directory containing thumbnail images
        slide_id: Slide identifier
    
    Returns:
        PIL Image of the thumbnail, or None if not found
    """
    # Try common extensions
    for ext in ['.jpg', '.jpeg', '.png']:
        thumb_path = os.path.join(thumbnails_dir, f"{slide_id}{ext}")
        if os.path.exists(thumb_path):
            return Image.open(thumb_path).convert('RGBA')
    return None


def load_embedding_file(embeddings_dir: str, slide_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings and coordinates from .npz or .pkl file.
    
    Args:
        embeddings_dir: Directory containing embedding files
        slide_id: Slide identifier
    
    Returns:
        embeddings: Array of shape (N, embedding_dim)
        coords: Array of shape (N, 2) with [x, y] coordinates
    """
    npz_path = os.path.join(embeddings_dir, f"{slide_id}.npz")
    pkl_path = os.path.join(embeddings_dir, f"{slide_id}.pkl")
    
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return data['embeddings'], data['coords']
    elif os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            embeddings, coords_list = pickle.load(f)
        coords = np.array([c['wsi_loc'] for c in coords_list], dtype=np.int32)
        return embeddings, coords
    else:
        raise FileNotFoundError(f"No embedding file found for slide {slide_id}")


def load_mosaic_metadata(mosaic_path: str, slide_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load mosaic metadata for a specific slide from the mosaic pickle file.
    
    The mosaic file contains cluster_locs which lists all tiles belonging
    to each mosaic patch cluster.
    
    Args:
        mosaic_path: Path to the mosaic metadata pickle file
        slide_id: Slide identifier to find
    
    Returns:
        List of tile info dicts with 'cluster_locs', or None if not found
    """
    if not os.path.exists(mosaic_path):
        return None
    
    mosaic = pd.read_pickle(mosaic_path)
    
    # Search for the slide in the mosaic file
    for item in mosaic:
        slide_path = item[0]
        if slide_id in os.path.basename(slide_path):
            return item[1]  # Return the list of tile info dicts
    
    return None


def compute_patch_similarities(
    query_embedding: np.ndarray,
    hit_embeddings: np.ndarray,
    feature_dim: int = 1280
) -> np.ndarray:
    """
    Compute L1 distances from query patch to all hit slide patches.
    
    Args:
        query_embedding: Query patch embedding
        hit_embeddings: All embeddings from hit slide
        feature_dim: Number of features to use (class token dim)
    
    Returns:
        distances: L1 distances for each patch in hit slide
    """
    # Use first feature_dim dimensions (class token features)
    query_features = query_embedding[:feature_dim] if len(query_embedding) > feature_dim else query_embedding
    hit_features = hit_embeddings[:, :feature_dim] if hit_embeddings.shape[1] > feature_dim else hit_embeddings
    
    # Compute L1 distances
    distances = np.abs(hit_features.astype(np.float32) - query_features.astype(np.float32)).sum(axis=1)
    
    return distances


def generate_roi_heatmap(
    mosaic_tiles: List[Dict[str, Any]],
    distances: np.ndarray,
    patch_size: int = 512,
    tile_size: int = 16,
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Generate a standalone heatmap image following the original demo approach.
    
    Uses cluster_locs to fill ALL tiles in each cluster with the same 
    similarity value, exactly like the demo notebook.
    
    Args:
        mosaic_tiles: List of tile info dicts with 'cluster_locs' field
        distances: Array of shape (N,) with L1 distances (lower = more similar)
        patch_size: Size of each patch in pixels (used for grid calculation)
        tile_size: Size of each tile in the output heatmap (pixels per grid cell)
        output_path: Optional path to save the heatmap image
    
    Returns:
        PIL Image of the heatmap
    """
    # Determine grid dimensions from cluster_locs
    max_x = 0
    max_y = 0
    for tile_info in mosaic_tiles:
        cluster_locs = tile_info.get('cluster_locs', [tile_info.get('wsi_loc', [0, 0])])
        for loc in cluster_locs:
            max_x = max(max_x, loc[0])
            max_y = max(max_y, loc[1])
    
    grid_w = (max_x // patch_size) + 1
    grid_h = (max_y // patch_size) + 1
    
    # Create heatmap grids (indexed as [x, y] like the demo)
    # Use int for similarity accumulation, then convert to float
    heatmap_similarity = np.zeros((int(grid_w), int(grid_h)), dtype=float)
    heatmap_counts = np.zeros((int(grid_w), int(grid_h)), dtype=int)
    
    # Fill in ALL cluster_locs for each mosaic tile with its similarity
    for mosaic_idx, tile_info in enumerate(mosaic_tiles):
        # Get cluster_locs - all tile locations belonging to this mosaic
        cluster_locs = tile_info.get('cluster_locs', [tile_info.get('wsi_loc', [0, 0])])
        
        # Each tile in the cluster gets the same distance value
        tile_distance = distances[mosaic_idx]
        
        for cluster_loc in cluster_locs:
            grid_x = cluster_loc[0] // patch_size
            grid_y = cluster_loc[1] // patch_size
            
            if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
                heatmap_similarity[grid_x, grid_y] = tile_distance
                heatmap_counts[grid_x, grid_y] += 1
    
    # Average if there are overlapping tiles (handle gracefully)
    heatmap_final = np.zeros_like(heatmap_similarity, dtype=float)
    tissue_mask = heatmap_counts != 0
    heatmap_final[tissue_mask] = heatmap_similarity[tissue_mask] / heatmap_counts[tissue_mask]
    
    # Normalize and invert: lower distance = higher similarity = higher value
    # This matches the demo: normalized = 1 - (val - min) / (max - min)
    normalized_heatmap = np.zeros_like(heatmap_final, dtype=float)
    
    if np.any(tissue_mask):
        min_val = heatmap_final[tissue_mask].min()
        max_val = heatmap_final[tissue_mask].max()
        if max_val != min_val:
            # Invert so lower distance = higher normalized value (more similar = warmer color)
            normalized_heatmap[tissue_mask] = 1 - (heatmap_final[tissue_mask] - min_val) / (max_val - min_val)
        else:
            normalized_heatmap[tissue_mask] = 1.0
    
    # Apply rainbow colormap (exactly like the demo)
    colored_heatmap = plt.cm.rainbow(normalized_heatmap)[:, :, :3]  # RGB only, no alpha
    
    # Set background (non-tissue) to white
    background_color = [1.0, 1.0, 1.0]
    colored_heatmap[~tissue_mask] = background_color
    
    # Convert to uint8 image
    heatmap_rgb = (colored_heatmap * 255).astype(np.uint8)
    
    # Resize using np.repeat (exactly like the demo)
    # Each grid cell becomes tile_size x tile_size pixels
    resized_heatmap = np.repeat(np.repeat(heatmap_rgb, tile_size, axis=0), tile_size, axis=1)
    
    # Transpose to get correct orientation (demo uses [x, y] indexing then transposes)
    resized_heatmap = np.transpose(resized_heatmap, (1, 0, 2))
    
    # Create PIL image
    heatmap_image = Image.fromarray(resized_heatmap)
    
    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        heatmap_image.save(output_path)
        print(f"Saved heatmap to: {output_path}")
    
    return heatmap_image


def generate_heatmap_for_thumbnail(
    mosaic_tiles: List[Dict[str, Any]],
    distances: np.ndarray,
    thumbnail: Image.Image,
    patch_size: int = 512,
    slide_dimensions: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Generate a heatmap directly in thumbnail coordinate space for perfect alignment.
    
    This function maps patch coordinates directly to thumbnail pixels, ensuring
    the heatmap aligns perfectly with the thumbnail when overlaid.
    
    Uses actual slide dimensions from slide_dimensions.csv for accurate mapping.
    
    Args:
        mosaic_tiles: List of tile info dicts with 'cluster_locs' field
        distances: Array of shape (N,) with L1 distances (lower = more similar)
        thumbnail: PIL Image of the slide thumbnail
        patch_size: Size of each patch in original slide pixels
        slide_dimensions: Optional (width, height) from slide_dimensions.csv
        output_path: Optional path to save the heatmap image
    
    Returns:
        PIL Image of the heatmap (same size as thumbnail)
    """
    thumb_w, thumb_h = thumbnail.size
    
    # Use actual slide dimensions if provided, otherwise estimate from coordinates
    if slide_dimensions is not None:
        slide_w, slide_h = slide_dimensions
        print(f"Using slide dimensions from CSV: {slide_w} x {slide_h}")
    else:
        # Fallback: estimate from max coordinates
        max_x = 0
        max_y = 0
        for tile_info in mosaic_tiles:
            cluster_locs = tile_info.get('cluster_locs', [tile_info.get('wsi_loc', [0, 0])])
            for loc in cluster_locs:
                max_x = max(max_x, loc[0] + patch_size)
                max_y = max(max_y, loc[1] + patch_size)
        slide_w = max_x
        slide_h = max_y
        print(f"Estimated slide dimensions from coordinates: {slide_w} x {slide_h}")
    
    # Calculate scale factors from slide to thumbnail
    # The thumbnail was likely generated with a consistent downscale ratio
    scale_x = thumb_w / slide_w
    scale_y = thumb_h / slide_h
    
    print(f"Thumbnail dimensions: {thumb_w} x {thumb_h}")
    print(f"Scale factors: x={scale_x:.6f}, y={scale_y:.6f}")
    
    # Initialize heatmap arrays
    heatmap_similarity = np.zeros((thumb_h, thumb_w), dtype=float)
    heatmap_counts = np.zeros((thumb_h, thumb_w), dtype=int)
    
    # Fill in heatmap values by mapping patch coordinates to thumbnail pixels
    for mosaic_idx, tile_info in enumerate(mosaic_tiles):
        cluster_locs = tile_info.get('cluster_locs', [tile_info.get('wsi_loc', [0, 0])])
        tile_distance = distances[mosaic_idx]
        
        for cluster_loc in cluster_locs:
            # Map patch coordinates to thumbnail coordinates using actual scale factors
            # Use independent x and y scales for accurate mapping
            x_start = int(cluster_loc[0] * scale_x)
            y_start = int(cluster_loc[1] * scale_y)
            x_end = int((cluster_loc[0] + patch_size) * scale_x)
            y_end = int((cluster_loc[1] + patch_size) * scale_y)
            
            # Clamp to thumbnail bounds
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(thumb_w, x_end)
            y_end = min(thumb_h, y_end)
            
            if x_end > x_start and y_end > y_start:
                heatmap_similarity[y_start:y_end, x_start:x_end] += tile_distance
                heatmap_counts[y_start:y_end, x_start:x_end] += 1
    
    # Average where overlapping
    heatmap_final = np.zeros_like(heatmap_similarity, dtype=float)
    tissue_mask = heatmap_counts > 0
    heatmap_final[tissue_mask] = heatmap_similarity[tissue_mask] / heatmap_counts[tissue_mask]
    
    # Normalize and invert: lower distance = higher similarity = warmer color
    normalized_heatmap = np.zeros_like(heatmap_final, dtype=float)
    if np.any(tissue_mask):
        min_val = heatmap_final[tissue_mask].min()
        max_val = heatmap_final[tissue_mask].max()
        if max_val != min_val:
            normalized_heatmap[tissue_mask] = 1 - (heatmap_final[tissue_mask] - min_val) / (max_val - min_val)
        else:
            normalized_heatmap[tissue_mask] = 1.0
    
    # Apply rainbow colormap
    colored_heatmap = plt.cm.rainbow(normalized_heatmap)[:, :, :3]
    
    # Set background to white
    colored_heatmap[~tissue_mask] = [1.0, 1.0, 1.0]
    
    # Convert to uint8 image
    heatmap_rgb = (colored_heatmap * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_rgb)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        heatmap_image.save(output_path)
        print(f"Saved aligned heatmap to: {output_path}")
    
    return heatmap_image


def overlay_heatmap_on_thumbnail(
    heatmap_image: Image.Image,
    thumbnail: Image.Image,
    alpha: int = 120,
    output_path: Optional[str] = None,
    mosaic_tiles: Optional[List[Dict[str, Any]]] = None,
    distances: Optional[np.ndarray] = None,
    patch_size: int = 512,
    slide_dimensions: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """
    Overlay heatmap on slide thumbnail.
    
    If mosaic_tiles and distances are provided, generates an aligned heatmap
    directly in thumbnail coordinate space for perfect alignment.
    Otherwise, resizes the provided heatmap_image to match thumbnail.
    
    Args:
        heatmap_image: PIL Image of the heatmap (RGB) - used if mosaic_tiles not provided
        thumbnail: PIL Image of the slide thumbnail
        alpha: Alpha value for heatmap overlay (0-255, default 120)
        output_path: Optional path to save the overlaid image
        mosaic_tiles: Optional list of tile info dicts for aligned generation
        distances: Optional array of distances for aligned generation
        patch_size: Patch size in original slide pixels
        slide_dimensions: Optional (width, height) from slide_dimensions.csv
    
    Returns:
        PIL Image with heatmap overlaid on thumbnail
    """
    thumb_w, thumb_h = thumbnail.size
    
    # Generate aligned heatmap if we have the necessary data
    if mosaic_tiles is not None and distances is not None:
        heatmap_for_overlay = generate_heatmap_for_thumbnail(
            mosaic_tiles=mosaic_tiles,
            distances=distances,
            thumbnail=thumbnail,
            patch_size=patch_size,
            slide_dimensions=slide_dimensions
        )
    else:
        # Fallback: resize existing heatmap to match thumbnail
        heatmap_for_overlay = heatmap_image.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
    
    # Convert to RGBA
    heatmap_rgba = heatmap_for_overlay.convert('RGBA')
    heatmap_array = np.array(heatmap_rgba)
    
    # Detect white background pixels and make them transparent
    is_white = (
        (heatmap_array[:, :, 0] > 250) & 
        (heatmap_array[:, :, 1] > 250) & 
        (heatmap_array[:, :, 2] > 250)
    )
    
    # Set alpha: 0 for white (transparent), specified alpha for tissue
    heatmap_array[:, :, 3] = np.where(is_white, 0, alpha)
    
    heatmap_with_alpha = Image.fromarray(heatmap_array)
    
    # Ensure thumbnail is RGBA
    thumbnail_rgba = thumbnail.convert('RGBA') if thumbnail.mode != 'RGBA' else thumbnail
    
    # Composite heatmap over thumbnail
    composite = Image.alpha_composite(thumbnail_rgba, heatmap_with_alpha)
    
    # Convert to RGB for saving
    composite_rgb = composite.convert('RGB')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        composite_rgb.save(output_path)
        print(f"Saved overlay to: {output_path}")
    
    return composite_rgb


def generate_roi_heatmap_simple(
    coords: np.ndarray,
    distances: np.ndarray,
    patch_size: int = 512,
    tile_size: int = 16,
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Generate a heatmap using only coordinates (no cluster_locs).
    
    This is a fallback when mosaic metadata is not available.
    Only fills single tiles per mosaic patch.
    
    Args:
        coords: Array of shape (N, 2) with [x, y] patch coordinates
        distances: Array of shape (N,) with L1 distances
        patch_size: Size of each patch in pixels
        tile_size: Size of each tile in output heatmap
        output_path: Optional path to save the heatmap image
    
    Returns:
        PIL Image of the heatmap
    """
    # Create fake mosaic_tiles structure with just wsi_loc
    mosaic_tiles = [{'wsi_loc': coord.tolist(), 'cluster_locs': [coord.tolist()]} 
                    for coord in coords]
    
    return generate_roi_heatmap(mosaic_tiles, distances, patch_size, tile_size, output_path)


def generate_roi_search_heatmap(
    embeddings_dir: str,
    query_slide_id: str,
    query_patch_idx: int,
    hit_slide_id: str,
    output_dir: str,
    mosaic_path: str = None,
    thumbnails_dir: str = None,
    dimensions_csv: str = None,
    patch_size: int = 512,
    tile_size: int = 16,
    overlay_alpha: int = 120
) -> Dict[str, str]:
    """
    Generate a heatmap for an ROI search result.
    
    Args:
        embeddings_dir: Directory containing .npz embedding files
        query_slide_id: Query slide ID
        query_patch_idx: Query patch index
        hit_slide_id: Hit slide ID from ROI search
        output_dir: Output directory for heatmap
        mosaic_path: Path to mosaic metadata file (for cluster_locs)
        thumbnails_dir: Directory containing slide thumbnails (for overlay)
        dimensions_csv: Path to slide_dimensions.csv for accurate coordinate mapping
        patch_size: Patch size in pixels
        tile_size: Size of each tile in output heatmap
        overlay_alpha: Alpha value for overlay (0-255, default 120)
    
    Returns:
        Dictionary with paths to saved images:
        - 'heatmap': standalone heatmap path
        - 'overlay': overlay path (if thumbnails_dir provided)
    """
    # Load embeddings
    query_embeddings, query_coords = load_embedding_file(embeddings_dir, query_slide_id)
    hit_embeddings, hit_coords = load_embedding_file(embeddings_dir, hit_slide_id)
    
    print(f"Query slide {query_slide_id}: {query_embeddings.shape[0]} patches")
    print(f"Hit slide {hit_slide_id}: {hit_embeddings.shape[0]} patches")
    print(f"Query patch {query_patch_idx} location: {query_coords[query_patch_idx]}")
    
    # Get query patch embedding
    query_embedding = query_embeddings[query_patch_idx]
    
    # Compute distances
    distances = compute_patch_similarities(query_embedding, hit_embeddings)
    
    print(f"Distance range: {distances.min():.2f} - {distances.max():.2f}")
    
    # Generate output paths
    base_name = f"heatmap_query_{query_slide_id}_patch{query_patch_idx}_hit_{hit_slide_id}"
    output_path = os.path.join(output_dir, f"{base_name}.png")
    
    result_paths = {}
    
    # Try to load mosaic metadata for cluster_locs
    mosaic_tiles = None
    if mosaic_path:
        mosaic_tiles = load_mosaic_metadata(mosaic_path, hit_slide_id)
        if mosaic_tiles:
            print(f"Loaded mosaic metadata with {len(mosaic_tiles)} tiles (using cluster_locs)")
        else:
            print(f"Warning: Could not find slide {hit_slide_id} in mosaic metadata")
    
    # Generate heatmap
    if mosaic_tiles:
        heatmap_image = generate_roi_heatmap(
            mosaic_tiles=mosaic_tiles,
            distances=distances,
            patch_size=patch_size,
            tile_size=tile_size,
            output_path=output_path
        )
    else:
        print("Warning: Using simple mode (no cluster_locs) - heatmap will show only mosaic tile locations")
        heatmap_image = generate_roi_heatmap_simple(
            coords=hit_coords,
            distances=distances,
            patch_size=patch_size,
            tile_size=tile_size,
            output_path=output_path
        )
    
    result_paths['heatmap'] = output_path
    
    # Generate overlay if thumbnails_dir is provided
    if thumbnails_dir:
        thumbnail = load_thumbnail(thumbnails_dir, hit_slide_id)
        if thumbnail:
            # Load slide dimensions for accurate coordinate mapping
            slide_dims = None
            if dimensions_csv:
                slide_dims = get_slide_dimensions(hit_slide_id, dimensions_csv)
                if slide_dims:
                    print(f"Loaded slide dimensions: {slide_dims[0]} x {slide_dims[1]}")
                else:
                    print(f"Warning: Could not find dimensions for slide {hit_slide_id} in CSV")
            
            overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
            # Pass mosaic_tiles, distances, and slide_dimensions for aligned overlay generation
            overlay_heatmap_on_thumbnail(
                heatmap_image=heatmap_image,
                thumbnail=thumbnail,
                alpha=overlay_alpha,
                output_path=overlay_path,
                mosaic_tiles=mosaic_tiles,
                distances=distances,
                patch_size=patch_size,
                slide_dimensions=slide_dims
            )
            result_paths['overlay'] = overlay_path
            print(f"Generated overlay on thumbnail: {overlay_path}")
        else:
            print(f"Warning: Thumbnail not found for slide {hit_slide_id}")
    
    return result_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate ROI search heatmap visualization"
    )
    parser.add_argument(
        "--embeddings-dir", "-e",
        type=str,
        required=True,
        help="Directory containing patch embedding files (.npz or .pkl)"
    )
    parser.add_argument(
        "--query-slide", "-q",
        type=str,
        required=True,
        help="Query slide ID"
    )
    parser.add_argument(
        "--query-patch", "-p",
        type=int,
        required=True,
        help="Query patch index"
    )
    parser.add_argument(
        "--hit-slide", "-t",
        type=str,
        required=True,
        help="Hit slide ID from ROI search (target slide)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output/heatmaps",
        help="Output directory for heatmap images"
    )
    parser.add_argument(
        "--mosaic-path", "-m",
        type=str,
        default=None,
        help="Path to mosaic metadata pickle file (for cluster_locs)"
    )
    parser.add_argument(
        "--thumbnails-dir",
        type=str,
        default=None,
        help="Directory containing slide thumbnails for overlay"
    )
    parser.add_argument(
        "--dimensions-csv",
        type=str,
        default=None,
        help="Path to slide_dimensions.csv for accurate coordinate mapping"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Patch size in pixels (default: 512)"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
        help="Size of each tile in output heatmap in pixels (default: 16)"
    )
    parser.add_argument(
        "--overlay-alpha",
        type=int,
        default=120,
        help="Alpha value for overlay (0-255, default: 120)"
    )
    
    args = parser.parse_args()
    
    result_paths = generate_roi_search_heatmap(
        embeddings_dir=args.embeddings_dir,
        query_slide_id=args.query_slide,
        query_patch_idx=args.query_patch,
        hit_slide_id=args.hit_slide,
        output_dir=args.output_dir,
        mosaic_path=args.mosaic_path,
        thumbnails_dir=args.thumbnails_dir,
        dimensions_csv=args.dimensions_csv,
        patch_size=args.patch_size,
        tile_size=args.tile_size,
        overlay_alpha=args.overlay_alpha
    )
    
    print(f"\n✅ Heatmap generated: {result_paths['heatmap']}")
    if 'overlay' in result_paths:
        print(f"✅ Overlay generated: {result_paths['overlay']}")


if __name__ == "__main__":
    main()
