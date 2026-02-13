"""
Slide-level similarity search engine.

This module provides efficient slide-level morphological similarity search
using pre-computed slide embeddings.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm


@dataclass
class SlideSearchResult:
    """Result of a slide-level search."""
    query_slide_id: str
    hit_slide_id: str
    distance: float
    rank: int
    percentile: Optional[float] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'query_slide_id': self.query_slide_id,
            'hit_slide_id': self.hit_slide_id,
            'distance': self.distance,
            'rank': self.rank,
            'percentile': self.percentile,
            'metadata': self.metadata
        }


class SlideSearchEngine:
    """
    Engine for slide-level morphological similarity search.
    
    Uses slide embeddings to find morphologically similar slides
    in a reference database.
    """
    
    def __init__(
        self,
        distance_metric: str = 'cityblock',
        top_k: int = 10,
        verbose: bool = True
    ):
        """
        Initialize SlideSearchEngine.
        
        Args:
            distance_metric: Distance metric ('cityblock' for L1, 'euclidean' for L2, 'cosine')
            top_k: Number of top results to return
            verbose: Whether to print progress
        """
        self.distance_metric = distance_metric
        self.top_k = top_k
        self.verbose = verbose
        
        self.reference_matrix: Optional[np.ndarray] = None
        self.reference_slide_ids: Optional[List[str]] = None
        self.background_distribution: Optional[np.ndarray] = None
        self.metadata: Optional[pd.DataFrame] = None
    
    def load_reference_database(
        self,
        embeddings_dir: str = None,
        matrix_path: str = None,
        metadata_path: str = None
    ):
        """
        Load reference database for search.
        
        Args:
            embeddings_dir: Directory containing individual embedding files,
                OR a single pickle/csv file containing an embedding matrix.
            matrix_path: Path to pre-computed embedding matrix
            metadata_path: Optional path to metadata CSV
        """
        # If embeddings_dir is actually a file, treat it as a matrix
        if embeddings_dir and os.path.isfile(embeddings_dir):
            matrix_path = embeddings_dir
            embeddings_dir = None

        if matrix_path and os.path.exists(matrix_path):
            # Load pre-computed matrix (supports dict or DataFrame)
            with open(matrix_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self.reference_matrix = data['matrix']
                self.reference_slide_ids = data['slide_ids']
            elif hasattr(data, 'values'):
                # pandas DataFrame — index = slide IDs, columns = features
                self.reference_slide_ids = [str(s) for s in data.index.tolist()]
                self.reference_matrix = data.values.astype(np.float64)
            else:
                raise ValueError(
                    f"Unsupported matrix format: {type(data)}. "
                    "Expected dict with 'matrix'/'slide_ids' or a DataFrame."
                )
        elif embeddings_dir:
            # Build matrix from individual files
            embeddings = {}
            from glob import glob
            files = glob(os.path.join(embeddings_dir, '*.pkl'))
            
            for filepath in tqdm(files, desc="Loading embeddings", disable=not self.verbose):
                slide_id = os.path.basename(filepath).replace('.pkl', '')
                try:
                    with open(filepath, 'rb') as f:
                        embeddings[slide_id] = pickle.load(f).flatten()
                except Exception:
                    pass
            
            self.reference_slide_ids = sorted(embeddings.keys())
            self.reference_matrix = np.stack(
                [embeddings[sid] for sid in self.reference_slide_ids]
            )
        else:
            raise ValueError("Must provide either embeddings_dir or matrix_path")
        
        if self.verbose:
            print(f"Loaded reference database: {len(self.reference_slide_ids)} slides, "
                  f"embedding dim: {self.reference_matrix.shape[1]}")
        
        # Load metadata if provided
        if metadata_path and os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
            if self.verbose:
                print(f"Loaded metadata with {len(self.metadata)} entries")
    
    def compute_background_distribution(
        self,
        similarity_matrix_path: str = None,
        sample_size: int = 10000
    ):
        """
        Compute or load background distribution for percentile calculation.
        
        Args:
            similarity_matrix_path: Path to pre-computed similarity matrix
            sample_size: Number of random pairs to sample if computing
        """
        if similarity_matrix_path and os.path.exists(similarity_matrix_path):
            with open(similarity_matrix_path, 'rb') as f:
                similarity_matrix = pickle.load(f)
            
            # Extract upper triangle
            if isinstance(similarity_matrix, pd.DataFrame):
                similarity_matrix = similarity_matrix.values
            
            background = []
            n = similarity_matrix.shape[0]
            for i in range(1, n):
                background.extend(similarity_matrix[i, :i])
            self.background_distribution = np.array(background)
        else:
            # Compute from reference matrix
            if self.reference_matrix is None:
                raise ValueError("Must load reference database first")
            
            n = len(self.reference_slide_ids)
            if n < 2:
                return
            
            # Sample random pairs
            n_pairs = min(sample_size, n * (n - 1) // 2)
            indices = np.random.choice(n, size=(n_pairs, 2), replace=True)
            indices = indices[indices[:, 0] != indices[:, 1]]
            
            distances = []
            for i, j in indices:
                dist = cdist(
                    self.reference_matrix[i:i+1],
                    self.reference_matrix[j:j+1],
                    metric=self.distance_metric
                )[0, 0]
                distances.append(dist)
            
            self.background_distribution = np.array(distances)
        
        if self.verbose:
            print(f"Background distribution: {len(self.background_distribution)} pairs")
    
    def search(
        self,
        query_embedding: np.ndarray,
        query_slide_id: str = "query",
        exclude_self: bool = True
    ) -> List[SlideSearchResult]:
        """
        Search for similar slides given a query embedding.
        
        Args:
            query_embedding: Query slide embedding
            query_slide_id: ID of the query slide (for exclusion)
            exclude_self: Whether to exclude the query slide from results
        
        Returns:
            List of SlideSearchResult objects
        """
        if self.reference_matrix is None:
            raise ValueError("Must load reference database first")
        
        # Reshape query if needed
        query = query_embedding.flatten().reshape(1, -1)
        
        # Compute distances
        distances = cdist(query, self.reference_matrix, metric=self.distance_metric)[0]
        
        # Get top-k indices
        sorted_indices = np.argsort(distances)
        
        results = []
        rank = 0
        for idx in sorted_indices:
            slide_id = self.reference_slide_ids[idx]
            
            # Skip self if requested
            if exclude_self and slide_id == query_slide_id:
                continue
            
            rank += 1
            if rank > self.top_k:
                break
            
            distance = distances[idx]
            
            # Compute percentile if background distribution available
            percentile = None
            if self.background_distribution is not None:
                percentile = np.sum(self.background_distribution < distance) / len(
                    self.background_distribution
                ) * 100
            
            # Get metadata if available
            meta = None
            if self.metadata is not None:
                slide_meta = self.metadata[
                    self.metadata.iloc[:, 0].astype(str) == str(slide_id)
                ]
                if len(slide_meta) > 0:
                    meta = slide_meta.iloc[0].to_dict()
            
            results.append(SlideSearchResult(
                query_slide_id=query_slide_id,
                hit_slide_id=slide_id,
                distance=float(distance),
                rank=rank,
                percentile=percentile,
                metadata=meta
            ))
        
        return results
    
    def search_from_file(
        self,
        embedding_path: str,
        query_slide_id: str = None
    ) -> List[SlideSearchResult]:
        """
        Search using a slide embedding from file.
        
        Args:
            embedding_path: Path to the embedding pickle file
            query_slide_id: Optional ID for the query slide
        
        Returns:
            List of search results
        """
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
        
        if query_slide_id is None:
            query_slide_id = os.path.basename(embedding_path).replace('.pkl', '')
        
        return self.search(embedding, query_slide_id)
    
    def batch_search(
        self,
        query_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, List[SlideSearchResult]]:
        """
        Perform batch search for multiple query slides.
        
        Args:
            query_embeddings: Dictionary mapping slide IDs to embeddings
        
        Returns:
            Dictionary mapping slide IDs to search results
        """
        results = {}
        
        iterator = query_embeddings.items()
        if self.verbose:
            iterator = tqdm(iterator, desc="Searching slides")
        
        for slide_id, embedding in iterator:
            results[slide_id] = self.search(embedding, slide_id)
        
        return results
    
    def results_to_dataframe(
        self,
        results: List[SlideSearchResult]
    ) -> pd.DataFrame:
        """Convert search results to a DataFrame."""
        return pd.DataFrame([r.to_dict() for r in results])
    
    def save_results(
        self,
        results: Dict[str, List[SlideSearchResult]],
        output_path: str
    ):
        """
        Save search results to file.
        
        Args:
            results: Dictionary of search results
            output_path: Path to save results
        """
        # Convert to serializable format
        serializable = {
            slide_id: [r.to_dict() for r in res]
            for slide_id, res in results.items()
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(serializable, f)
        
        if self.verbose:
            print(f"Saved results for {len(results)} queries to {output_path}")
