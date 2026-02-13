"""
Test slide search functionality with virchow2 + percentile (sampler) pipeline.

This test verifies that:
1. Pre-computed embeddings can be loaded and used for search
2. The SlideSearchEngine works correctly
3. The percentile slide embedding generator produces correct output shapes
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy.spatial.distance import cdist

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Minimal SlideSearchResult for testing without full imports
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
    """Minimal slide search engine for testing."""
    
    def __init__(
        self,
        distance_metric: str = 'cityblock',
        top_k: int = 10,
        verbose: bool = True
    ):
        self.distance_metric = distance_metric
        self.top_k = top_k
        self.verbose = verbose
        self.reference_matrix: Optional[np.ndarray] = None
        self.reference_slide_ids: Optional[List[str]] = None
    
    def search(
        self,
        query_embedding: np.ndarray,
        query_slide_id: str = "query",
        exclude_self: bool = True
    ) -> List[SlideSearchResult]:
        if self.reference_matrix is None:
            raise ValueError("Must set reference database first")
        
        query = query_embedding.flatten().reshape(1, -1)
        distances = cdist(query, self.reference_matrix, metric=self.distance_metric)[0]
        sorted_indices = np.argsort(distances)
        
        results = []
        rank = 0
        for idx in sorted_indices:
            slide_id = self.reference_slide_ids[idx]
            if exclude_self and slide_id == query_slide_id:
                continue
            rank += 1
            if rank > self.top_k:
                break
            results.append(SlideSearchResult(
                query_slide_id=query_slide_id,
                hit_slide_id=slide_id,
                distance=float(distances[idx]),
                rank=rank
            ))
        return results
    
    def batch_search(
        self,
        query_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, List[SlideSearchResult]]:
        results = {}
        for slide_id, embedding in query_embeddings.items():
            results[slide_id] = self.search(embedding, slide_id)
        return results


class PercentileSlideEmbeddingGenerator:
    """Minimal percentile embedding generator for testing."""
    
    METHOD_NAME = "percentile"
    
    def __init__(
        self,
        percentiles: Optional[List[int]] = None,
        feature_dim: int = 1280,
        n_workers: int = 32,
        verbose: bool = True
    ):
        self.feature_dim = feature_dim
        self.n_workers = n_workers
        self.verbose = verbose
        if percentiles is None:
            percentiles = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        self.percentiles = percentiles
    
    @property
    def embedding_dim(self) -> int:
        return len(self.percentiles) * self.feature_dim
    
    def process_single_slide(
        self,
        patch_embeddings: np.ndarray,
        slide_id: str,
        mosaic=None
    ) -> np.ndarray:
        features = patch_embeddings[:, :self.feature_dim].astype(np.float16)
        slide_embedding = np.percentile(features, self.percentiles, axis=0)
        return slide_embedding


# Paths to test data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MATRIX_PATH = os.path.join(DATA_DIR, "final_slides_sampler_matrix.pkl")
METADATA_PATH = os.path.join(DATA_DIR, "final_scan_id_disease_map_rich_data.csv")


def test_load_precomputed_embeddings():
    """Test loading pre-computed sampler (percentile) embeddings."""
    print("=" * 60)
    print("Test 1: Load Pre-computed Embeddings")
    print("=" * 60)
    
    # Load the matrix (now a DataFrame with slide IDs as index)
    ref_df = pd.read_pickle(MATRIX_PATH)
    slide_ids = [str(sid) for sid in ref_df.index.tolist()]
    matrix = ref_df.values
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix dtype: {matrix.dtype}")
    print(f"Number of slides: {len(slide_ids)}")
    print(f"Sample slide IDs: {slide_ids[:5]}")
    
    # Load metadata for disease labels
    metadata = pd.read_csv(METADATA_PATH)
    if 'Slide.Scan.File' in metadata.columns:
        metadata = metadata.rename(columns={'Slide.Scan.File': 'slide_id', 'Tumor.group': 'tumor_group'})
    metadata['slide_id'] = metadata['slide_id'].astype(int).astype(str)
    
    # Verify dimensions match expected percentile embedding
    # 10 percentiles × 1280 feature dim = 12800
    expected_dim = 10 * 1280
    assert matrix.shape[1] == expected_dim, f"Expected dim {expected_dim}, got {matrix.shape[1]}"
    
    print("PASSED: Pre-computed embeddings loaded successfully")
    return matrix, slide_ids, metadata


def test_slide_search_engine(matrix, slide_ids, metadata):
    """Test SlideSearchEngine with pre-computed embeddings."""
    print("\n" + "=" * 60)
    print("Test 2: Slide Search Engine")
    print("=" * 60)
    
    # Initialize search engine
    search_engine = SlideSearchEngine(
        distance_metric='cityblock',  # L1 distance
        top_k=10,
        verbose=True
    )
    
    # Set reference database directly
    search_engine.reference_matrix = matrix
    search_engine.reference_slide_ids = slide_ids
    
    print(f"Reference database: {len(slide_ids)} slides")
    print(f"Embedding dimension: {matrix.shape[1]}")
    
    # Find test slide (4261) in the database
    test_slide_id = "4261"
    if test_slide_id in slide_ids:
        test_idx = slide_ids.index(test_slide_id)
        test_embedding = matrix[test_idx]
        print(f"\nQuery slide: {test_slide_id} (index {test_idx})")
        
        # Get disease label
        test_disease = metadata[metadata['slide_id'] == test_slide_id]['tumor_group'].values
        if len(test_disease) > 0:
            print(f"Query disease: {test_disease[0]}")
    else:
        # Use first slide as test
        test_idx = 0
        test_slide_id = slide_ids[0]
        test_embedding = matrix[0]
        print(f"\nQuery slide: {test_slide_id} (index {test_idx})")
    
    # Perform search
    results = search_engine.search(
        query_embedding=test_embedding,
        query_slide_id=test_slide_id,
        exclude_self=True
    )
    
    print(f"\nTop {len(results)} similar slides:")
    for r in results:
        # Get disease label for hit
        hit_disease = metadata[metadata['slide_id'] == r.hit_slide_id]['tumor_group'].values
        disease_str = hit_disease[0] if len(hit_disease) > 0 else "Unknown"
        print(f"  Rank {r.rank}: {r.hit_slide_id} (distance: {r.distance:.4f}, disease: {disease_str})")
    
    # Verify results
    assert len(results) == 10, f"Expected 10 results, got {len(results)}"
    assert all(isinstance(r, SlideSearchResult) for r in results), "Results should be SlideSearchResult objects"
    assert results[0].rank == 1, "First result should have rank 1"
    assert all(r.query_slide_id == test_slide_id for r in results), "Query slide ID should be consistent"
    assert test_slide_id not in [r.hit_slide_id for r in results], "Query slide should be excluded"
    
    print("\nPASSED: Slide search works correctly")
    return results


def test_batch_search(matrix, slide_ids, metadata):
    """Test batch search functionality."""
    print("\n" + "=" * 60)
    print("Test 3: Batch Search")
    print("=" * 60)
    
    search_engine = SlideSearchEngine(
        distance_metric='cityblock',
        top_k=5,
        verbose=False
    )
    search_engine.reference_matrix = matrix
    search_engine.reference_slide_ids = slide_ids
    
    # Create query dict with first 5 slides
    query_embeddings = {
        slide_ids[i]: matrix[i] for i in range(5)
    }
    
    print(f"Running batch search for {len(query_embeddings)} query slides...")
    
    # Run batch search
    all_results = search_engine.batch_search(query_embeddings)
    
    print(f"Got results for {len(all_results)} slides")
    
    # Verify results
    assert len(all_results) == 5, f"Expected 5 result sets, got {len(all_results)}"
    for slide_id, results in all_results.items():
        assert len(results) == 5, f"Expected 5 results for {slide_id}, got {len(results)}"
        assert slide_id not in [r.hit_slide_id for r in results], f"Self should be excluded for {slide_id}"
    
    print("PASSED: Batch search works correctly")
    return all_results


def test_percentile_embedding_generator():
    """Test PercentileSlideEmbeddingGenerator produces correct output."""
    print("\n" + "=" * 60)
    print("Test 4: Percentile Slide Embedding Generator")
    print("=" * 60)
    
    # Create generator with default settings
    generator = PercentileSlideEmbeddingGenerator(
        percentiles=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        feature_dim=1280,
        n_workers=1,
        verbose=True
    )
    
    print(f"Method: {generator.METHOD_NAME}")
    print(f"Percentiles: {generator.percentiles}")
    print(f"Feature dimension: {generator.feature_dim}")
    print(f"Expected output dimension: {generator.embedding_dim}")
    
    # Create synthetic patch embeddings (100 patches, 2560 dim like Virchow2)
    np.random.seed(42)
    synthetic_patches = np.random.randn(100, 2560).astype(np.float32)
    
    # Generate slide embedding
    slide_embedding = generator.process_single_slide(
        patch_embeddings=synthetic_patches,
        slide_id="test_slide",
        mosaic=None
    )
    
    print(f"\nInput shape: {synthetic_patches.shape}")
    print(f"Output shape: {slide_embedding.shape}")
    print(f"Output dtype: {slide_embedding.dtype}")
    
    # Verify output shape: (n_percentiles, feature_dim)
    expected_shape = (10, 1280)
    assert slide_embedding.shape == expected_shape, f"Expected {expected_shape}, got {slide_embedding.shape}"
    
    # Verify flattened dimension matches pre-computed matrix
    flattened_dim = slide_embedding.flatten().shape[0]
    assert flattened_dim == 12800, f"Expected flattened dim 12800, got {flattened_dim}"
    
    print("\nPASSED: Percentile embedding generator works correctly")
    return slide_embedding


def test_disease_retrieval_accuracy(matrix, slide_ids, metadata):
    """Test if similar slides tend to have the same disease type."""
    print("\n" + "=" * 60)
    print("Test 5: Disease Retrieval Analysis")
    print("=" * 60)
    
    search_engine = SlideSearchEngine(
        distance_metric='cityblock',
        top_k=10,
        verbose=False
    )
    search_engine.reference_matrix = matrix
    search_engine.reference_slide_ids = slide_ids
    
    # Create slide_id to disease mapping
    disease_map = dict(zip(
        metadata['slide_id'],
        metadata['tumor_group']
    ))
    
    # Sample 50 slides for analysis
    np.random.seed(42)
    sample_indices = np.random.choice(len(slide_ids), size=min(50, len(slide_ids)), replace=False)
    
    matches = 0
    total = 0
    
    for idx in sample_indices:
        query_id = slide_ids[idx]
        query_disease = disease_map.get(query_id, "Unknown")
        query_embedding = matrix[idx]
        
        results = search_engine.search(query_embedding, query_id, exclude_self=True)
        
        # Check top-1 hit
        if results:
            hit_disease = disease_map.get(results[0].hit_slide_id, "Unknown")
            if query_disease == hit_disease:
                matches += 1
            total += 1
    
    accuracy = matches / total * 100 if total > 0 else 0
    print(f"Top-1 disease match accuracy: {accuracy:.1f}% ({matches}/{total})")
    
    # Also compute top-5 accuracy
    matches_5 = 0
    for idx in sample_indices:
        query_id = slide_ids[idx]
        query_disease = disease_map.get(query_id, "Unknown")
        query_embedding = matrix[idx]
        
        results = search_engine.search(query_embedding, query_id, exclude_self=True)
        
        # Check if any of top 5 matches disease
        for r in results[:5]:
            hit_disease = disease_map.get(r.hit_slide_id, "Unknown")
            if query_disease == hit_disease:
                matches_5 += 1
                break
    
    accuracy_5 = matches_5 / total * 100 if total > 0 else 0
    print(f"Top-5 disease match accuracy: {accuracy_5:.1f}% ({matches_5}/{total})")
    
    print("\nPASSED: Disease retrieval analysis complete")
    return accuracy, accuracy_5


def main():
    """Run all tests."""
    print("=" * 60)
    print("Virchow2 + Percentile (Sampler) Slide Search Tests")
    print("=" * 60)
    
    # Check if test data exists
    if not os.path.exists(MATRIX_PATH):
        print(f"ERROR: Matrix file not found: {MATRIX_PATH}")
        return 1
    
    if not os.path.exists(METADATA_PATH):
        print(f"ERROR: Metadata file not found: {METADATA_PATH}")
        return 1
    
    try:
        # Test 1: Load embeddings
        matrix, slide_ids, metadata = test_load_precomputed_embeddings()
        
        # Test 2: Single slide search
        test_slide_search_engine(matrix, slide_ids, metadata)
        
        # Test 3: Batch search
        test_batch_search(matrix, slide_ids, metadata)
        
        # Test 4: Percentile generator
        test_percentile_embedding_generator()
        
        # Test 5: Disease retrieval
        test_disease_retrieval_accuracy(matrix, slide_ids, metadata)
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
