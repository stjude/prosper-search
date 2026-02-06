"""
Utility functions for PePath.
"""

import os
import pickle
import numpy as np
from typing import Tuple, Dict, List, Optional
from glob import glob


def load_patch_embeddings(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load patch embeddings from file (supports both .npz and .pkl formats).
    
    Args:
        filepath: Path to embedding file (.npz or .pkl)
    
    Returns:
        embeddings: np.ndarray of shape (N, embedding_dim), dtype float16
        coords: np.ndarray of shape (N, 2), dtype int32 - [[x, y], ...]
    
    Example:
        >>> embeddings, coords = load_patch_embeddings('data/patch_embeddings/12345.npz')
        >>> print(embeddings.shape)  # (1500, 1280)
        >>> print(coords.shape)      # (1500, 2)
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
        if isinstance(data[1], list) and len(data[1]) > 0 and isinstance(data[1][0], dict):
            coords = np.array([d['wsi_loc'] for d in data[1]], dtype=np.int32)
        else:
            coords = np.array(data[1], dtype=np.int32)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    return embeddings, coords


def load_all_patch_embeddings(directory: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load all patch embeddings from a directory.
    
    Args:
        directory: Path to directory containing embedding files
    
    Returns:
        Dict mapping slide_id to (embeddings, coords) tuples
    
    Example:
        >>> all_embeddings = load_all_patch_embeddings('data/patch_embeddings')
        >>> for slide_id, (emb, coords) in all_embeddings.items():
        ...     print(f"{slide_id}: {emb.shape[0]} patches")
    """
    embeddings_dict = {}
    
    # Try both formats
    files = glob(os.path.join(directory, "*.npz")) + glob(os.path.join(directory, "*.pkl"))
    
    for filepath in sorted(files):
        basename = os.path.basename(filepath)
        slide_id = basename.replace('.npz', '').replace('.pkl', '')
        embeddings_dict[slide_id] = load_patch_embeddings(filepath)
    
    return embeddings_dict


def coords_to_dict_list(coords: np.ndarray) -> List[Dict]:
    """
    Convert coords array back to original dict format (for backward compatibility).
    
    Args:
        coords: np.ndarray of shape (N, 2)
    
    Returns:
        List of dicts: [{'wsi_loc': [x, y]}, ...]
    """
    return [{'wsi_loc': coord.tolist()} for coord in coords]


def get_slide_ids(directory: str) -> List[str]:
    """
    Get list of slide IDs from a directory of embedding files.
    
    Args:
        directory: Path to directory containing embedding files
    
    Returns:
        Sorted list of slide IDs
    """
    files = glob(os.path.join(directory, "*.npz")) + glob(os.path.join(directory, "*.pkl"))
    slide_ids = [os.path.basename(f).replace('.npz', '').replace('.pkl', '') for f in files]
    return sorted(set(slide_ids))
