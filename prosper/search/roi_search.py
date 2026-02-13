"""
ROI-level (patch-level) similarity search engine.

This module provides patch-level morphological similarity search
for finding similar regions of interest across whole slide images.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from multiprocessing import Pool
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class ROISearchResult:
    """Result of an ROI-level search."""
    query_slide_id: str
    query_patch_id: int
    query_location: Tuple[int, int]
    hit_slide_id: str
    hit_patch_id: int
    hit_location: Optional[Tuple[int, int]]
    distance: float
    rank: int
    
    def to_dict(self) -> Dict:
        return {
            'query_slide_id': self.query_slide_id,
            'query_patch_id': self.query_patch_id,
            'query_location_x': self.query_location[0] if self.query_location else None,
            'query_location_y': self.query_location[1] if self.query_location else None,
            'hit_slide_id': self.hit_slide_id,
            'hit_patch_id': self.hit_patch_id,
            'hit_location_x': self.hit_location[0] if self.hit_location else None,
            'hit_location_y': self.hit_location[1] if self.hit_location else None,
            'distance': self.distance,
            'rank': self.rank
        }


def load_patch_embedding_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load patch embeddings from file (supports both .npz and .pkl formats).
    
    Args:
        filepath: Path to embedding file (.npz or .pkl)
    
    Returns:
        embeddings: np.ndarray of shape (N, embedding_dim)
        coords: np.ndarray of shape (N, 2) - [[x, y], ...]
    """
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        embeddings = data['embeddings']
        coords = data['coords']
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        embeddings = data[0]
        # Convert list of dicts to numpy array
        if isinstance(data[1], list) and len(data[1]) > 0:
            if isinstance(data[1][0], dict):
                coords = np.array([d['wsi_loc'] for d in data[1]], dtype=np.int32)
            else:
                coords = np.array(data[1], dtype=np.int32)
        else:
            coords = np.array(data[1], dtype=np.int32)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    return embeddings, coords


class ROISearchEngine:
    """
    Engine for ROI-level (patch-level) morphological similarity search.
    
    Enables finding morphologically similar tissue regions across
    a database of whole slide images.
    """
    
    def __init__(
        self,
        top_k: int = 10,
        n_workers: int = 32,
        verbose: bool = True
    ):
        """
        Initialize ROISearchEngine.
        
        Args:
            top_k: Number of top results per patch
            n_workers: Number of parallel workers
            verbose: Whether to print progress
        """
        self.top_k = top_k
        self.n_workers = n_workers
        self.verbose = verbose
        
        self.reference_embeddings: Dict[str, np.ndarray] = {}
        self.reference_coords: Dict[str, np.ndarray] = {}  # Changed from mosaics
    
    def load_reference_database(
        self,
        embeddings_dir: str,
        mosaics_dir: str = None
    ):
        """
        Load reference database for ROI search.
        
        Args:
            embeddings_dir: Directory containing patch embedding files (.npz or .pkl)
            mosaics_dir: Optional directory containing mosaic files (legacy)
        """
        # Support both .npz and .pkl files
        files = glob(os.path.join(embeddings_dir, '*.npz')) + glob(os.path.join(embeddings_dir, '*.pkl'))
        
        for filepath in tqdm(files, desc="Loading embeddings", disable=not self.verbose):
            basename = os.path.basename(filepath)
            slide_id = basename.replace('.npz', '').replace('.pkl', '')
            
            try:
                embeddings, coords = load_patch_embedding_file(filepath)
                self.reference_embeddings[slide_id] = embeddings
                self.reference_coords[slide_id] = coords
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load {filepath}: {e}")
        
        if self.verbose:
            print(f"Loaded {len(self.reference_embeddings)} slides for ROI search")
    
    def _search_single_patch(
        self,
        query_patch: np.ndarray,
        exclude_slide_id: str = None
    ) -> List[Tuple[str, int, float]]:
        """Search for a single patch across all reference slides."""
        all_distances = []
        
        for slide_id, embeddings in self.reference_embeddings.items():
            if slide_id == exclude_slide_id:
                continue
            
            if embeddings is None or len(embeddings) == 0:
                continue
            
            # Use only class token features (first 1280 dims)
            ref_features = embeddings[:, :1280] if embeddings.shape[1] > 1280 else embeddings
            query_features = query_patch[:1280] if len(query_patch) > 1280 else query_patch
            
            # Compute L1 distances
            distances = np.abs(ref_features.astype(np.float32) - query_features.astype(np.float32)).sum(axis=1)
            
            # Find best patch in this slide
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            all_distances.append((slide_id, int(min_idx), float(min_dist)))
        
        # Sort by distance and return top-k
        all_distances.sort(key=lambda x: x[2])
        return all_distances[:self.top_k]
    
    def search(
        self,
        query_embeddings: np.ndarray,
        query_slide_id: str,
        query_coords: np.ndarray = None,
        patch_indices: List[int] = None
    ) -> List[List[ROISearchResult]]:
        """
        Search for similar ROIs given query patch embeddings.
        
        Args:
            query_embeddings: Array of shape (n_patches, embedding_dim)
            query_slide_id: ID of the query slide
            query_coords: Optional coords array of shape (n_patches, 2) for location data
            patch_indices: Optional list of specific patch indices to search (for subset search)
        
        Returns:
            List of lists of ROISearchResult (one list per query patch)
        """
        all_results = []
        
        # If patch_indices specified, only search those patches
        if patch_indices is not None:
            patches_to_search = [(idx, query_embeddings[idx]) for idx in patch_indices]
        else:
            patches_to_search = list(enumerate(query_embeddings))
        
        iterator = patches_to_search
        if self.verbose:
            iterator = tqdm(patches_to_search, desc=f"Searching patches for {query_slide_id}")
        
        for patch_idx, query_patch in iterator:
            # Get query location if available
            query_loc = None
            if query_coords is not None and patch_idx < len(query_coords):
                query_loc = tuple(query_coords[patch_idx].tolist())
            
            # Search for this patch
            top_hits = self._search_single_patch(query_patch, exclude_slide_id=query_slide_id)
            
            patch_results = []
            for rank, (hit_slide_id, hit_patch_id, distance) in enumerate(top_hits, 1):
                # Get hit location if available
                hit_loc = None
                if hit_slide_id in self.reference_coords:
                    coords = self.reference_coords[hit_slide_id]
                    if hit_patch_id < len(coords):
                        hit_loc = tuple(coords[hit_patch_id].tolist())
                
                patch_results.append(ROISearchResult(
                    query_slide_id=query_slide_id,
                    query_patch_id=patch_idx,
                    query_location=query_loc,
                    hit_slide_id=hit_slide_id,
                    hit_patch_id=hit_patch_id,
                    hit_location=hit_loc,
                    distance=float(distance),
                    rank=rank
                ))
            
            all_results.append(patch_results)
        
        return all_results
    
    def search_from_file(
        self,
        embedding_path: str,
        query_slide_id: str = None,
        patch_indices: List[int] = None
    ) -> List[List[ROISearchResult]]:
        """
        Search using patch embeddings from file.
        
        Args:
            embedding_path: Path to the embedding file (.npz or .pkl)
            query_slide_id: Optional ID for the query slide
            patch_indices: Optional list of specific patch indices to search
        
        Returns:
            List of search results per patch
        """
        embeddings, coords = load_patch_embedding_file(embedding_path)
        
        if query_slide_id is None:
            basename = os.path.basename(embedding_path)
            query_slide_id = basename.replace('.npz', '').replace('.pkl', '')
        
        return self.search(embeddings, query_slide_id, coords, patch_indices)
    
    def results_to_dataframe(
        self,
        results: List[List[ROISearchResult]]
    ) -> pd.DataFrame:
        """Convert search results to a DataFrame."""
        rows = []
        for patch_results in results:
            for result in patch_results:
                rows.append(result.to_dict())
        return pd.DataFrame(rows)
    
    def save_results(
        self,
        results: Dict[str, List[List[ROISearchResult]]],
        output_path: str
    ):
        """
        Save ROI search results to file.
        
        Args:
            results: Dictionary mapping slide IDs to ROI results
            output_path: Path to save results
        """
        serializable = {}
        for slide_id, slide_results in results.items():
            serializable[slide_id] = [
                [r.to_dict() for r in patch_results]
                for patch_results in slide_results
            ]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(serializable, f)
        
        if self.verbose:
            print(f"Saved ROI results for {len(results)} queries to {output_path}")
