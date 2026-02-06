"""
Mosaic generation module for whole slide images.

This module handles the extraction and clustering of tissue patches from WSIs
to create representative mosaics for downstream feature extraction.
"""

import os
import pickle
import warnings
from glob import glob
from typing import List, Dict, Optional, Tuple, Any
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm

import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans

# Lazy import openslide to avoid breaking other modules
openslide = None
OpenSlideError = None

def _ensure_openslide():
    """Lazily import openslide when needed."""
    global openslide, OpenSlideError
    if openslide is None:
        try:
            import openslide as _openslide
            from openslide import OpenSlideError as _OpenSlideError
            openslide = _openslide
            OpenSlideError = _OpenSlideError
        except ImportError:
            raise ImportError(
                "openslide-python is required for mosaic generation. "
                "Install with: pip install openslide-python"
            )

warnings.filterwarnings("ignore")


def rgb_to_hsd(X: np.ndarray) -> np.ndarray:
    """
    Convert RGB color space to HSD (Hue-Saturation-Density) for stain analysis.
    
    Args:
        X: Input RGB array with values in [0, 1]
    
    Returns:
        HSD transformed array
    """
    eps = np.finfo(float).eps
    X = np.clip(X, eps, 1.0)
    OD = -np.log(X / 1.0)
    D = np.mean(OD, 3)
    D = np.clip(D, eps, None)
    cx = OD[:, :, :, 0] / D - 1.0
    cy = (OD[:, :, :, 1] - OD[:, :, :, 2]) / (np.sqrt(3.0) * D)
    D = np.expand_dims(D, 3)
    cx = np.expand_dims(cx, 3)
    cy = np.expand_dims(cy, 3)
    X_HSD = np.concatenate((D, cx, cy), 3)
    return X_HSD


def clean_thumbnail(thumbnail: Image.Image) -> np.ndarray:
    """
    Clean thumbnail by removing low-variance (background) regions.
    
    Args:
        thumbnail: PIL Image of slide thumbnail
    
    Returns:
        Cleaned numpy array with background set to white
    """
    thumbnail_arr = np.asarray(thumbnail)
    wthumbnail = np.copy(thumbnail_arr)
    
    # Remove low-variance regions (likely background)
    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std < 5] = 255
    
    # Apply HSD-based cleaning for stain detection
    thumbnail_HSD = rgb_to_hsd(np.array([wthumbnail.astype('float32') / 255.]))[0]
    kernel = np.ones((30, 30), np.float32) / 900
    thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:, :, 2], -1, kernel)
    wthumbnail[thumbnail_HSD_mean < 0.05] = 255
    
    return wthumbnail


def process_slide(
    slide_fn: str,
    base_patchsize: int,
    percentage_selected: int = 15,
    tissue_threshold: float = 0.7,
    savejpg: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Process a single whole slide image to extract mosaic patches.
    
    Args:
        slide_fn: Path to the slide file
        base_patchsize: Size of patches to extract
        percentage_selected: Percentage of patches to select per cluster
        tissue_threshold: Minimum tissue coverage required for a patch
        savejpg: Optional directory to save patch images
    
    Returns:
        List of mosaic patch dictionaries, or None if processing fails
    """
    _ensure_openslide()
    try:
        slide = openslide.open_slide(slide_fn)
    except Exception as e:
        print(f"Error opening slide {slide_fn}: {e}")
        return None
    
    # Get slide dimensions
    w, h = slide.dimensions
    patch_size = base_patchsize
    
    # Get thumbnail for tissue detection
    try:
        thumbnail = slide.get_thumbnail((500, 500))
    except Exception as e:
        print(f"get_thumbnail failed for {slide_fn}: {e}")
        try:
            slide = openslide.open_slide(slide_fn)
            level_image = slide.read_region((0, 0), 2, slide.level_dimensions[2]).convert("RGB")
            thumbnail = level_image.resize((500, 500), Image.BILINEAR)
        except Exception as e2:
            print(f"Failed to get thumbnail: {e2}")
            return None
    
    # Clean thumbnail and create tissue mask
    cthumbnail = clean_thumbnail(thumbnail)
    tissue_mask = (cthumbnail.mean(axis=2) != 255).astype(float)
    
    mask_hratio = (tissue_mask.shape[0] / h) * patch_size
    mask_wratio = (tissue_mask.shape[1] / w) * patch_size
    
    # Extract all patches with tissue coverage information
    patches = []
    for i, hi in enumerate(range(0, h, int(patch_size))):
        for j, wi in enumerate(range(0, w, int(patch_size))):
            mi = int(i * mask_hratio)
            mj = int(j * mask_wratio)
            patch_mask = tissue_mask[mi:mi + int(mask_hratio), mj:mj + int(mask_wratio)]
            if patch_mask.size > 0:
                tissue_coverage = np.count_nonzero(patch_mask) / patch_mask.size
            else:
                tissue_coverage = 0
            patches.append({
                'loc': [i, j],
                'wsi_loc': [int(wi), int(hi)],
                'tissue_coverage': tissue_coverage
            })
    
    # Filter patches by tissue threshold
    selected_patches = [p for p in patches if p['tissue_coverage'] >= tissue_threshold]
    
    if len(selected_patches) == 0:
        print(f"No valid patches found for slide {slide_fn}")
        return None
    
    # Extract color histograms for clustering
    for patch in selected_patches:
        w_pos, h_pos = patch['wsi_loc']
        patch_size_5x = patch_size // 4
        try:
            patch_region = slide.read_region(
                (w_pos, h_pos), 1, (patch_size_5x, patch_size_5x)
            ).convert('RGB')
            if patch_region.size[0] != 250:
                patch_region = patch_region.resize((250, 250))
            histogram = (np.array(patch_region) / 255.).reshape((-1, 3)).mean(axis=0)
            patch['rgb_histogram'] = histogram
        except Exception:
            patch['rgb_histogram'] = np.array([0.5, 0.5, 0.5])
    
    # Cluster patches by color
    features = np.array([p['rgb_histogram'] for p in selected_patches])
    if len(features) == 0:
        print(f"Skipping slide {slide_fn}: no valid features")
        return None
    
    kmeans_clusters = min(9, len(features))
    kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42, n_init=10)
    kmeans.fit(features)
    
    for patch, label in zip(selected_patches, kmeans.labels_):
        patch['cluster_lbl'] = label
    
    # Select representative patches from each cluster
    mosaic = []
    for i in range(kmeans_clusters):
        cluster_patches = [p for p, l in zip(selected_patches, kmeans.labels_) if l == i]
        n_selected = max(1, int(len(cluster_patches) * percentage_selected / 100.))
        
        if len(cluster_patches) < n_selected:
            n_selected = len(cluster_patches)
        
        if n_selected == 0:
            continue
        
        # Sub-cluster by spatial location
        loc_features = np.array([p['wsi_loc'] for p in cluster_patches])
        km = KMeans(n_clusters=n_selected, random_state=42, n_init=10)
        ds = km.fit_transform(loc_features)
        ds_labels = km.labels_
        
        # Select patches closest to sub-cluster centers
        c_selected_idx = set()
        for idx in range(n_selected):
            sorted_idx = np.argsort(ds[:, idx])
            idx_loc_features = [loc_features[j].tolist() for j in range(len(loc_features)) if ds_labels[j] == idx]
            for sidx in sorted_idx:
                if sidx not in c_selected_idx:
                    c_selected_idx.add(sidx)
                    cluster_patches[sidx]['cluster_locs'] = idx_loc_features
                    mosaic.append(cluster_patches[sidx])
                    break
    
    # Optionally save patch images
    if savejpg and len(mosaic) > 0:
        os.makedirs(savejpg, exist_ok=True)
        slide_name = os.path.basename(slide_fn).replace('.svs', '')
        for i, patch in enumerate(mosaic):
            w_pos, h_pos = patch['wsi_loc']
            try:
                patch_img = slide.read_region((w_pos, h_pos), 0, (patch_size, patch_size)).convert('RGB')
                patch_img.save(os.path.join(savejpg, f"{slide_name}_patch_{i}.jpg"))
            except Exception:
                pass
    
    return mosaic


def _process_slide_wrapper(args: Tuple) -> Optional[Tuple[str, List[Dict]]]:
    """Wrapper function for parallel processing."""
    slide_path, tmpdir, base_patchsize, percentage_selected, tissue_threshold, savejpg = args
    slide_id = os.path.basename(slide_path).replace('.svs', '')
    cache_path = os.path.join(tmpdir, f"{slide_id}.pkl")
    
    # Check cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                mosaic = pickle.load(f)
            return (slide_id, mosaic)
        except Exception:
            pass
    
    # Process slide
    mosaic = process_slide(slide_path, base_patchsize, percentage_selected, tissue_threshold, savejpg)
    
    # Cache result
    if mosaic is not None:
        os.makedirs(tmpdir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(mosaic, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return (slide_id, mosaic)


class MosaicGenerator:
    """
    Generator for creating mosaics from whole slide images.
    
    This class handles batch processing of WSIs to extract representative
    tissue patches for downstream feature extraction.
    """
    
    def __init__(
        self,
        base_patchsize: int = 512,
        percentage_selected: int = 15,
        tissue_threshold: float = 0.7,
        n_workers: int = 32,
        verbose: bool = True
    ):
        """
        Initialize MosaicGenerator.
        
        Args:
            base_patchsize: Size of patches to extract
            percentage_selected: Percentage of patches to select per cluster
            tissue_threshold: Minimum tissue coverage required
            n_workers: Number of parallel workers
            verbose: Whether to print progress
        """
        self.base_patchsize = base_patchsize
        self.percentage_selected = percentage_selected
        self.tissue_threshold = tissue_threshold
        self.n_workers = n_workers
        self.verbose = verbose
    
    def process_slides(
        self,
        slide_paths: List[str],
        output_dir: str,
        force_recompute: bool = False,
        savejpg: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Process multiple slides to generate mosaics.
        
        Args:
            slide_paths: List of paths to slide files
            output_dir: Directory to save intermediate and final results
            force_recompute: Whether to recompute even if cached
            savejpg: Optional directory to save patch images
        
        Returns:
            Dictionary mapping slide IDs to their mosaic patches
        """
        tmpdir = os.path.join(output_dir, "tmp")
        os.makedirs(tmpdir, exist_ok=True)
        
        # Filter already processed unless force_recompute
        if not force_recompute:
            slides_to_process = []
            for s in slide_paths:
                slide_id = os.path.basename(s).replace('.svs', '')
                cache_path = os.path.join(tmpdir, f"{slide_id}.pkl")
                if not os.path.exists(cache_path):
                    slides_to_process.append(s)
        else:
            slides_to_process = slide_paths
        
        if self.verbose:
            print(f"Total slides: {len(slide_paths)}")
            print(f"Slides to process: {len(slides_to_process)}")
        
        # Prepare arguments for parallel processing
        args_list = [
            (s, tmpdir, self.base_patchsize, self.percentage_selected, 
             self.tissue_threshold, savejpg)
            for s in slides_to_process
        ]
        
        # Process slides
        results = {}
        if self.n_workers > 1 and len(slides_to_process) > 1:
            with Pool(self.n_workers) as pool:
                iterator = pool.imap_unordered(_process_slide_wrapper, args_list)
                if self.verbose:
                    iterator = tqdm(iterator, total=len(slides_to_process), desc="Processing slides")
                for slide_id, mosaic in iterator:
                    if mosaic is not None:
                        results[slide_id] = mosaic
        else:
            iterator = args_list
            if self.verbose:
                iterator = tqdm(iterator, desc="Processing slides")
            for args in iterator:
                result = _process_slide_wrapper(args)
                if result[1] is not None:
                    results[result[0]] = result[1]
        
        # Load cached results for previously processed slides
        for s in slide_paths:
            slide_id = os.path.basename(s).replace('.svs', '')
            if slide_id not in results:
                cache_path = os.path.join(tmpdir, f"{slide_id}.pkl")
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'rb') as f:
                            results[slide_id] = pickle.load(f)
                    except Exception:
                        pass
        
        return results
    
    def save_mosaics(
        self,
        mosaics: Dict[str, List[Dict]],
        output_path: str
    ):
        """
        Save mosaics to a pickle file.
        
        Args:
            mosaics: Dictionary of slide ID to mosaic patches
            output_path: Path to save the pickle file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_data = [(slide_id, patches) for slide_id, patches in mosaics.items()]
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        if self.verbose:
            print(f"Saved {len(save_data)} mosaics to {output_path}")
    
    @staticmethod
    def load_mosaics(mosaic_path: str) -> Dict[str, List[Dict]]:
        """
        Load mosaics from a pickle file.
        
        Args:
            mosaic_path: Path to the pickle file
        
        Returns:
            Dictionary mapping slide IDs to mosaic patches
        """
        with open(mosaic_path, 'rb') as f:
            data = pickle.load(f)
        return {slide_id: patches for slide_id, patches in data}
