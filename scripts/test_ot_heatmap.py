#!/usr/bin/env python3
"""
Test script to generate OT slide similarity heatmaps.
Creates standalone heatmaps (no slide overlay) showing per-patch similarity scores.
"""

import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prosper.embeddings.slide_embedding import OTSlideEmbeddingGenerator
from prosper.visualization.ot_visualization import (
    compute_per_patch_weighted_distance_scores,
    normalize_scores
)


def load_embedding_file(filepath):
    """Load embeddings from .npz or .pkl file."""
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        embeddings = data['embeddings']
        coords = data['coords']
        # Handle potential quantization
        if 'scale' in data and 'zero_point' in data:
            scale = data['scale']
            zero_point = data['zero_point']
            embeddings = (embeddings.astype(np.float32) - zero_point) * scale
        return embeddings, coords
    else:
        with open(filepath, 'rb') as f:
            embeddings, coords_list = pickle.load(f)
        coords = np.array([c['wsi_loc'] for c in coords_list], dtype=np.int32)
        return embeddings, coords


def load_mosaic_metadata(mosaic_path):
    """Load mosaic metadata with cluster locations.
    
    The mosaic file contains a list of tuples: (slide_path, mosaic_tiles, None)
    Returns a dictionary mapping slide_id -> mosaic_tiles
    """
    with open(mosaic_path, 'rb') as f:
        data = pickle.load(f)
    
    # Build lookup by slide ID
    slide_lookup = {}
    for item in data:
        path = item[0]
        slide_id = path.split('/')[-1].replace('.svs', '')
        slide_lookup[slide_id] = item[1]  # mosaic_tiles (list of dicts) or None
    
    return slide_lookup


def load_slide_dimensions(csv_path):
    """Load slide dimensions from CSV file.
    
    Args:
        csv_path: Path to slide_dimensions.csv with columns: slide_id, width, height
    
    Returns:
        Dictionary mapping slide_id (str) to (width, height) tuple
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    dimensions = {}
    for _, row in df.iterrows():
        slide_id = str(row['slide_id'])
        dimensions[slide_id] = (int(row['width']), int(row['height']))
    return dimensions


def generate_ot_heatmap(
    coords: np.ndarray,
    scores: np.ndarray,
    mosaic_tiles: list = None,
    patch_size: int = 512,
    tile_size: int = 16,
    output_path: str = None,
    title: str = None,
    invert: bool = True
):
    """
    Generate a standalone OT similarity heatmap.
    
    Args:
        coords: Array of (x, y) patch coordinates
        scores: Per-patch similarity scores (lower = more similar)
        mosaic_tiles: Optional list of mosaic tile dicts with 'cluster_locs'
        patch_size: Size of patches in pixels
        tile_size: Size of output tiles in pixels
        output_path: Path to save the heatmap
        title: Optional title for the heatmap
        invert: If True, invert scores (so lower distance = warmer color)
    
    Returns:
        PIL Image of the heatmap
    """
    # Determine grid dimensions
    max_x = coords[:, 0].max()
    max_y = coords[:, 1].max()
    
    grid_w = (max_x // patch_size) + 1
    grid_h = (max_y // patch_size) + 1
    
    # Initialize heatmap grid with NaN for transparent background
    heatmap_grid = np.full((int(grid_w), int(grid_h)), np.nan, dtype=float)
    heatmap_counts = np.zeros((int(grid_w), int(grid_h)), dtype=float)
    
    if mosaic_tiles is not None:
        # Use cluster locations from mosaic metadata
        for idx, tile_info in enumerate(mosaic_tiles):
            if idx >= len(scores):
                continue
            score = scores[idx]
            cluster_locs = tile_info.get('cluster_locs', [tile_info.get('wsi_loc', [0, 0])])
            for loc in cluster_locs:
                x, y = loc[0], loc[1]
                grid_x = int(x // patch_size)
                grid_y = int(y // patch_size)
                if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
                    if np.isnan(heatmap_grid[grid_x, grid_y]):
                        heatmap_grid[grid_x, grid_y] = 0
                    heatmap_grid[grid_x, grid_y] += score
                    heatmap_counts[grid_x, grid_y] += 1
        
        # Average where multiple mosaics contribute
        valid = heatmap_counts > 0
        heatmap_grid[valid] /= heatmap_counts[valid]
    else:
        # Use raw coordinates
        for idx, (x, y) in enumerate(coords):
            grid_x = int(x // patch_size)
            grid_y = int(y // patch_size)
            if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
                heatmap_grid[grid_x, grid_y] = scores[idx]
    
    # Normalize for colormap
    mask = ~np.isnan(heatmap_grid)
    normalized_heatmap = np.zeros_like(heatmap_grid, dtype=float)
    if np.any(mask):
        min_val = np.nanmin(heatmap_grid)
        max_val = np.nanmax(heatmap_grid)
        if max_val != min_val:
            normalized_heatmap[mask] = (heatmap_grid[mask] - min_val) / (max_val - min_val)
            if invert:
                normalized_heatmap[mask] = 1 - normalized_heatmap[mask]
        else:
            normalized_heatmap[mask] = 0.5
    
    # Apply colormap (rainbow: red=high similarity, violet=low similarity)
    cmap = plt.cm.rainbow
    colored_heatmap = cmap(normalized_heatmap)[:, :, :3]
    
    # Set non-tissue background to white
    colored_heatmap[~mask] = [1, 1, 1]
    
    # Resize to desired output tile size
    resized_heatmap = np.repeat(np.repeat(colored_heatmap, tile_size, axis=0), tile_size, axis=1)
    
    # Transpose for correct orientation
    resized_heatmap = np.transpose(resized_heatmap, (1, 0, 2))
    
    # Convert to PIL Image
    final_image = Image.fromarray((resized_heatmap * 255).astype(np.uint8))
    
    # Save image
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_image.save(output_path)
        print(f"Saved heatmap to: {output_path}")
    
    return final_image


def generate_aligned_heatmap(
    mosaic_tiles: list,
    scores: np.ndarray,
    thumbnail: Image.Image,
    patch_size: int = 512,
    slide_dimensions: tuple = None,
    invert: bool = True
) -> Image.Image:
    """
    Generate a heatmap directly in thumbnail coordinate space for perfect alignment.
    
    Uses actual slide dimensions from CSV for accurate coordinate mapping.
    
    Args:
        mosaic_tiles: List of tile info dicts with 'cluster_locs' field
        scores: Per-mosaic similarity scores
        thumbnail: PIL Image of the slide thumbnail
        patch_size: Patch size in original slide pixels
        slide_dimensions: Optional (width, height) from slide_dimensions.csv
        invert: If True, invert scores (lower = warmer color)
    
    Returns:
        PIL Image of the heatmap (same size as thumbnail)
    """
    thumb_w, thumb_h = thumbnail.size
    
    # Use actual slide dimensions if provided, otherwise estimate from coordinates
    if slide_dimensions is not None:
        slide_w, slide_h = slide_dimensions
        print(f"  Using slide dimensions from CSV: {slide_w} x {slide_h}")
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
        print(f"  Estimated slide dimensions from coordinates: {slide_w} x {slide_h}")
    
    # Calculate scale factors from slide to thumbnail
    scale_x = thumb_w / slide_w
    scale_y = thumb_h / slide_h
    
    print(f"  Thumbnail dimensions: {thumb_w} x {thumb_h}")
    print(f"  Scale factors: x={scale_x:.6f}, y={scale_y:.6f}")
    
    # Initialize heatmap arrays
    heatmap_similarity = np.zeros((thumb_h, thumb_w), dtype=float)
    heatmap_counts = np.zeros((thumb_h, thumb_w), dtype=int)
    
    # Fill in heatmap values using independent x and y scale factors
    for mosaic_idx, tile_info in enumerate(mosaic_tiles):
        if mosaic_idx >= len(scores):
            continue
        cluster_locs = tile_info.get('cluster_locs', [tile_info.get('wsi_loc', [0, 0])])
        tile_score = scores[mosaic_idx]
        
        for cluster_loc in cluster_locs:
            # Map patch coordinates to thumbnail coordinates using actual scale factors
            x_start = int(cluster_loc[0] * scale_x)
            y_start = int(cluster_loc[1] * scale_y)
            x_end = int((cluster_loc[0] + patch_size) * scale_x)
            y_end = int((cluster_loc[1] + patch_size) * scale_y)
            
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(thumb_w, x_end)
            y_end = min(thumb_h, y_end)
            
            if x_end > x_start and y_end > y_start:
                heatmap_similarity[y_start:y_end, x_start:x_end] += tile_score
                heatmap_counts[y_start:y_end, x_start:x_end] += 1
    
    # Average where overlapping
    heatmap_final = np.zeros_like(heatmap_similarity, dtype=float)
    tissue_mask = heatmap_counts > 0
    heatmap_final[tissue_mask] = heatmap_similarity[tissue_mask] / heatmap_counts[tissue_mask]
    
    # Normalize
    normalized_heatmap = np.zeros_like(heatmap_final, dtype=float)
    if np.any(tissue_mask):
        min_val = heatmap_final[tissue_mask].min()
        max_val = heatmap_final[tissue_mask].max()
        if max_val != min_val:
            normalized_heatmap[tissue_mask] = (heatmap_final[tissue_mask] - min_val) / (max_val - min_val)
            if invert:
                normalized_heatmap[tissue_mask] = 1 - normalized_heatmap[tissue_mask]
        else:
            normalized_heatmap[tissue_mask] = 0.5
    
    # Apply rainbow colormap
    cmap = plt.cm.rainbow
    colored_heatmap = cmap(normalized_heatmap)[:, :, :3]
    colored_heatmap[~tissue_mask] = [1.0, 1.0, 1.0]
    
    heatmap_rgb = (colored_heatmap * 255).astype(np.uint8)
    return Image.fromarray(heatmap_rgb)


def overlay_heatmap_on_thumbnail(
    heatmap_image: Image.Image,
    thumbnail: Image.Image,
    alpha: int = 120,
    output_path: str = None,
    mosaic_tiles: list = None,
    scores: np.ndarray = None,
    patch_size: int = 512,
    slide_dimensions: tuple = None,
    invert: bool = True
) -> Image.Image:
    """
    Overlay heatmap on slide thumbnail with perfect alignment.
    
    If mosaic_tiles and scores are provided, generates an aligned heatmap.
    Otherwise, resizes the provided heatmap_image.
    
    Args:
        heatmap_image: PIL Image of the heatmap (used if mosaic_tiles not provided)
        thumbnail: PIL Image of the slide thumbnail
        alpha: Alpha value for overlay (0-255)
        output_path: Optional path to save
        mosaic_tiles: Optional list of tile info dicts for aligned generation
        scores: Optional array of scores for aligned generation
        patch_size: Patch size in original slide pixels
        slide_dimensions: Optional (width, height) from slide_dimensions.csv
        invert: If True, invert scores for coloring
    
    Returns:
        PIL Image with heatmap overlaid on thumbnail
    """
    thumb_w, thumb_h = thumbnail.size
    
    # Generate aligned heatmap if we have the necessary data
    if mosaic_tiles is not None and scores is not None:
        heatmap_for_overlay = generate_aligned_heatmap(
            mosaic_tiles=mosaic_tiles,
            scores=scores,
            thumbnail=thumbnail,
            patch_size=patch_size,
            slide_dimensions=slide_dimensions,
            invert=invert
        )
    else:
        heatmap_for_overlay = heatmap_image.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
    
    # Convert to RGBA
    heatmap_rgba = heatmap_for_overlay.convert('RGBA')
    heatmap_array = np.array(heatmap_rgba)
    
    # Make white pixels transparent
    is_white = (
        (heatmap_array[:, :, 0] > 250) & 
        (heatmap_array[:, :, 1] > 250) & 
        (heatmap_array[:, :, 2] > 250)
    )
    heatmap_array[:, :, 3] = np.where(is_white, 0, alpha)
    
    heatmap_with_alpha = Image.fromarray(heatmap_array)
    thumbnail_rgba = thumbnail.convert('RGBA')
    composite = Image.alpha_composite(thumbnail_rgba, heatmap_with_alpha)
    composite_rgb = composite.convert('RGB')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        composite_rgb.save(output_path)
        print(f"Saved overlay to: {output_path}")
    
    return composite_rgb


def combine_heatmaps_side_by_side(
    img1: Image.Image,
    img2: Image.Image,
    labels: list = None,
    output_path: str = None,
    padding: int = 20,
    bg_color: tuple = (255, 255, 255),
    label_height: int = 40
):
    """
    Combine two heatmap images side-by-side with padding.
    
    Args:
        img1: First PIL Image
        img2: Second PIL Image
        labels: Optional list of two labels for the images
        output_path: Path to save the combined image
        padding: Padding between images and around edges
        bg_color: Background color (RGB tuple)
        label_height: Height reserved for labels
    
    Returns:
        Combined PIL Image
    """
    from PIL import ImageDraw, ImageFont
    
    w1, h1 = img1.size
    w2, h2 = img2.size
    
    # Calculate target height (max of both) and pad accordingly
    max_h = max(h1, h2)
    
    # Calculate total dimensions
    total_width = w1 + w2 + 3 * padding  # left padding + middle padding + right padding
    total_height = max_h + 2 * padding + (label_height if labels else 0)
    
    # Create combined image with white background
    combined = Image.new('RGB', (total_width, total_height), bg_color)
    
    # Calculate vertical positions to center each image
    y1 = padding + (label_height if labels else 0) + (max_h - h1) // 2
    y2 = padding + (label_height if labels else 0) + (max_h - h2) // 2
    
    # Paste images
    combined.paste(img1, (padding, y1))
    combined.paste(img2, (padding + w1 + padding, y2))
    
    # Add labels if provided
    if labels and len(labels) >= 2:
        draw = ImageDraw.Draw(combined)
        
        # Try to use a nice font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
        
        # Draw labels centered above each image
        label1_bbox = draw.textbbox((0, 0), labels[0], font=font)
        label1_w = label1_bbox[2] - label1_bbox[0]
        label1_x = padding + (w1 - label1_w) // 2
        draw.text((label1_x, padding // 2), labels[0], fill=(0, 0, 0), font=font)
        
        label2_bbox = draw.textbbox((0, 0), labels[1], font=font)
        label2_w = label2_bbox[2] - label2_bbox[0]
        label2_x = padding + w1 + padding + (w2 - label2_w) // 2
        draw.text((label2_x, padding // 2), labels[1], fill=(0, 0, 0), font=font)
    
    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.save(output_path)
        print(f"Saved combined heatmap to: {output_path}")
    
    return combined


def main():
    print("=" * 60)
    print("Generating OT Slide Similarity Heatmaps")
    print("=" * 60)
    
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embeddings_dir = os.path.join(project_root, "data", "patch_embeddings")
    mosaic_path = os.path.join(project_root, "data", "mosaic_metadata.pkl")
    thumbnails_dir = os.path.join(project_root, "data", "thumbnails")
    dimensions_csv = os.path.join(project_root, "data", "slide_dimensions.csv")
    output_dir = os.path.join(project_root, "output", "ot_heatmaps")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mosaic metadata
    print("\nLoading mosaic metadata...")
    mosaic_data = load_mosaic_metadata(mosaic_path)
    print(f"Loaded mosaic metadata with {len(mosaic_data)} slides")
    
    # Load slide dimensions
    print("\nLoading slide dimensions...")
    slide_dims = load_slide_dimensions(dimensions_csv)
    print(f"Loaded dimensions for {len(slide_dims)} slides")
    
    # Select two slides for comparison
    slide1_id = "4261"  # Query slide
    slide2_id = "4264"  # Hit slide (from previous ROI search)
    
    # Check for embeddings
    slide1_file = os.path.join(embeddings_dir, f"{slide1_id}.npz")
    slide2_file = os.path.join(embeddings_dir, f"{slide2_id}.npz")
    
    if not os.path.exists(slide1_file):
        slide1_file = os.path.join(embeddings_dir, f"{slide1_id}.pkl")
    if not os.path.exists(slide2_file):
        slide2_file = os.path.join(embeddings_dir, f"{slide2_id}.pkl")
    
    print(f"\nLoading embeddings for slides {slide1_id} and {slide2_id}...")
    embeddings1, coords1 = load_embedding_file(slide1_file)
    embeddings2, coords2 = load_embedding_file(slide2_file)
    
    print(f"  Slide {slide1_id}: {embeddings1.shape} embeddings, {coords1.shape} coordinates")
    print(f"  Slide {slide2_id}: {embeddings2.shape} embeddings, {coords2.shape} coordinates")
    
    # Get mosaic metadata for both slides
    mosaic1 = mosaic_data.get(slide1_id) or []
    mosaic2 = mosaic_data.get(slide2_id) or []
    
    print(f"  Slide {slide1_id}: {len(mosaic1)} mosaic tiles")
    print(f"  Slide {slide2_id}: {len(mosaic2)} mosaic tiles")
    
    # Initialize OT generator
    print("\nInitializing OT generator...")
    ot_gen = OTSlideEmbeddingGenerator(
        metric="cosine",
        reg=0.0,
        feature_dim=1280,
        verbose=True
    )
    
    # Process slides
    print("\nProcessing slides for OT computation...")
    result1 = ot_gen.process_single_slide(embeddings1.astype(np.float32), slide1_id, mosaic1)
    result2 = ot_gen.process_single_slide(embeddings2.astype(np.float32), slide2_id, mosaic2)
    
    # Compute detailed OT similarity
    print("\nComputing OT distance with transport matrix...")
    ot_result = ot_gen.compute_detailed_similarity(result1, result2)
    print(f"  OT distance: {ot_result['distance']:.6f}")
    print(f"  Transport matrix shape: {ot_result['T'].shape}")
    print(f"  Cost matrix shape: {ot_result['M'].shape}")
    
    # Compute per-patch weighted distance scores
    print("\nComputing per-patch similarity scores...")
    T = ot_result['T']
    M = ot_result['M']
    D_A, D_B = compute_per_patch_weighted_distance_scores(T, M)
    
    print(f"  Slide {slide1_id} scores: min={D_A.min():.6f}, max={D_A.max():.6f}")
    print(f"  Slide {slide2_id} scores: min={D_B.min():.6f}, max={D_B.max():.6f}")
    
    # Generate heatmaps
    print("\nGenerating heatmaps...")
    
    # Heatmap for slide 1 (query)
    output_path1 = os.path.join(output_dir, f"ot_heatmap_{slide1_id}_vs_{slide2_id}.png")
    heatmap1 = generate_ot_heatmap(
        coords=coords1,
        scores=D_A,
        mosaic_tiles=mosaic1 if mosaic1 else None,
        patch_size=512,
        tile_size=16,
        output_path=output_path1,
        title=f"OT Similarity: Slide {slide1_id}",
        invert=True  # Lower distance = warmer color (more similar)
    )
    print(f"  Generated heatmap for slide {slide1_id}: {heatmap1.size}")
    
    # Heatmap for slide 2 (hit)
    output_path2 = os.path.join(output_dir, f"ot_heatmap_{slide2_id}_vs_{slide1_id}.png")
    heatmap2 = generate_ot_heatmap(
        coords=coords2,
        scores=D_B,
        mosaic_tiles=mosaic2 if mosaic2 else None,
        patch_size=512,
        tile_size=16,
        output_path=output_path2,
        title=f"OT Similarity: Slide {slide2_id}",
        invert=True
    )
    print(f"  Generated heatmap for slide {slide2_id}: {heatmap2.size}")
    
    # Generate overlays on thumbnails (with proper alignment using slide dimensions)
    print("\nGenerating aligned overlay on thumbnails...")
    
    # Get slide dimensions
    dims1 = slide_dims.get(slide1_id)
    dims2 = slide_dims.get(slide2_id)
    print(f"  Slide {slide1_id} dimensions: {dims1}")
    print(f"  Slide {slide2_id} dimensions: {dims2}")
    
    # Load thumbnails
    thumb1_path = os.path.join(thumbnails_dir, f"{slide1_id}.jpg")
    thumb2_path = os.path.join(thumbnails_dir, f"{slide2_id}.jpg")
    
    overlay1, overlay2 = None, None
    
    if os.path.exists(thumb1_path):
        thumb1 = Image.open(thumb1_path)
        overlay1_path = os.path.join(output_dir, f"ot_heatmap_{slide1_id}_vs_{slide2_id}_overlay.png")
        overlay1 = overlay_heatmap_on_thumbnail(
            heatmap1, thumb1, alpha=120, output_path=overlay1_path,
            mosaic_tiles=mosaic1, scores=D_A, patch_size=512, 
            slide_dimensions=dims1, invert=True
        )
        print(f"  Generated aligned overlay for slide {slide1_id}: {overlay1.size}")
    else:
        print(f"  Warning: Thumbnail not found for slide {slide1_id}")
    
    if os.path.exists(thumb2_path):
        thumb2 = Image.open(thumb2_path)
        overlay2_path = os.path.join(output_dir, f"ot_heatmap_{slide2_id}_vs_{slide1_id}_overlay.png")
        overlay2 = overlay_heatmap_on_thumbnail(
            heatmap2, thumb2, alpha=120, output_path=overlay2_path,
            mosaic_tiles=mosaic2, scores=D_B, patch_size=512, 
            slide_dimensions=dims2, invert=True
        )
        print(f"  Generated aligned overlay for slide {slide2_id}: {overlay2.size}")
    else:
        print(f"  Warning: Thumbnail not found for slide {slide2_id}")
    
    # Combine heatmaps side-by-side
    print("\nCombining heatmaps side-by-side...")
    combined_path = os.path.join(output_dir, f"ot_heatmap_combined_{slide1_id}_vs_{slide2_id}.png")
    combined_img = combine_heatmaps_side_by_side(
        heatmap1, heatmap2,
        labels=[f"Slide {slide1_id}", f"Slide {slide2_id}"],
        output_path=combined_path,
        padding=20,
        bg_color=(255, 255, 255)
    )
    print(f"  Combined heatmap: {combined_img.size}")
    
    # Combine overlays side-by-side
    if overlay1 is not None and overlay2 is not None:
        print("\nCombining overlays side-by-side...")
        combined_overlay_path = os.path.join(output_dir, f"ot_overlay_combined_{slide1_id}_vs_{slide2_id}.png")
        combined_overlay = combine_heatmaps_side_by_side(
            overlay1, overlay2,
            labels=[f"Slide {slide1_id}", f"Slide {slide2_id}"],
            output_path=combined_overlay_path,
            padding=20,
            bg_color=(255, 255, 255)
        )
        print(f"  Combined overlay: {combined_overlay.size}")
    
    print("\n" + "=" * 60)
    print("OT Heatmap Generation Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Standalone heatmaps:")
    print(f"    - {output_path1}")
    print(f"    - {output_path2}")
    print(f"    - {combined_path} (combined)")
    if overlay1 is not None:
        print(f"  Overlays on thumbnails:")
        print(f"    - {overlay1_path}")
    if overlay2 is not None:
        print(f"    - {overlay2_path}")
    if overlay1 is not None and overlay2 is not None:
        print(f"    - {combined_overlay_path} (combined)")


if __name__ == "__main__":
    main()
