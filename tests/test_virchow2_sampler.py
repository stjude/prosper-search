"""
Test virchow2 + sampler (percentile) slide search using actual pepath modules.

This test verifies the full pipeline integration without requiring openslide
by using pre-computed embeddings.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pepath.search.slide_search import SlideSearchEngine, SlideSearchResult
from pepath.embeddings.slide_embedding.percentile_embedding import PercentileSlideEmbeddingGenerator
from pepath.embeddings.slide_embedding import get_slide_embedding_generator, list_slide_embedding_methods
from pepath import PePathConfig

# Paths to test data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MATRIX_PATH = os.path.join(DATA_DIR, "final_slides_sampler_matrix.npy")
METADATA_PATH = os.path.join(DATA_DIR, "final_scan_id_disease_map.csv")


def test_pepath_config():
    """Test PePathConfig for virchow2 + percentile."""
    print("=" * 60)
    print("Test 1: PePathConfig for virchow2 + percentile")
    print("=" * 60)
    
    config = PePathConfig(
        slide_path=DATA_DIR,
        output_dir="./output/test",
        patch_model="virchow2",
        slide_embedding_method="percentile",
        device="cpu",
        verbose=True
    )
    
    print(f"Patch model: {config.patch_model}")
    print(f"Slide embedding method: {config.slide_embedding_method}")
    print(f"Feature dimension: {config.feature_dim}")
    print(f"Percentiles: {config.percentiles}")
    
    assert config.patch_model == "virchow2"
    assert config.slide_embedding_method == "percentile"
    assert config.feature_dim == 1280
    assert len(config.percentiles) == 10
    
    print("PASSED: PePathConfig works correctly")
    return config


def test_slide_embedding_factory():
    """Test slide embedding factory function."""
    print("\n" + "=" * 60)
    print("Test 2: Slide Embedding Factory")
    print("=" * 60)
    
    # List available methods
    methods = list_slide_embedding_methods()
    print(f"Available methods: {methods}")
    assert "percentile" in methods
    
    # Get percentile generator
    generator = get_slide_embedding_generator(
        method_name="percentile",
        feature_dim=1280,
        n_workers=1,
        verbose=True
    )
    
    print(f"Generator type: {type(generator).__name__}")
    print(f"Method name: {generator.METHOD_NAME}")
    
    assert isinstance(generator, PercentileSlideEmbeddingGenerator)
    assert generator.feature_dim == 1280
    
    print("PASSED: Slide embedding factory works correctly")
    return generator


def test_slide_search_with_precomputed():
    """Test slide search with pre-computed embeddings."""
    print("\n" + "=" * 60)
    print("Test 3: Slide Search with Pre-computed Embeddings")
    print("=" * 60)
    
    # Load pre-computed embeddings
    matrix = np.load(MATRIX_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    slide_ids = metadata['Slide.Scan.File'].astype(str).tolist()
    
    print(f"Loaded {len(slide_ids)} slides with embedding dim {matrix.shape[1]}")
    
    # Initialize search engine
    engine = SlideSearchEngine(
        distance_metric='cityblock',
        top_k=10,
        verbose=True
    )
    engine.reference_matrix = matrix
    engine.reference_slide_ids = slide_ids
    
    # Search for slide 4261 (NB)
    query_id = "4261"
    if query_id in slide_ids:
        idx = slide_ids.index(query_id)
        results = engine.search(matrix[idx], query_id, exclude_self=True)
        
        print(f"\nQuery: {query_id}")
        print(f"Top-5 results:")
        for r in results[:5]:
            disease = metadata[metadata['Slide.Scan.File'].astype(str) == r.hit_slide_id]['Tumor.group'].values
            disease_str = disease[0] if len(disease) > 0 else "?"
            print(f"  Rank {r.rank}: {r.hit_slide_id} ({disease_str}) - dist: {r.distance:.2f}")
        
        assert len(results) == 10
        assert all(isinstance(r, SlideSearchResult) for r in results)
        
        print("\nPASSED: Slide search works correctly")
    else:
        print(f"Slide {query_id} not found, skipping search test")


def test_percentile_generator():
    """Test percentile embedding generator."""
    print("\n" + "=" * 60)
    print("Test 4: Percentile Embedding Generator")
    print("=" * 60)
    
    generator = PercentileSlideEmbeddingGenerator(
        percentiles=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        feature_dim=1280,
        n_workers=1,
        verbose=True
    )
    
    # Create synthetic Virchow2-like embeddings (2560 dim)
    np.random.seed(42)
    patch_embeddings = np.random.randn(50, 2560).astype(np.float32)
    
    slide_embedding = generator.process_single_slide(
        patch_embeddings=patch_embeddings,
        slide_id="test",
        mosaic=None
    )
    
    print(f"Input: {patch_embeddings.shape}")
    print(f"Output: {slide_embedding.shape}")
    
    # Should be (10, 1280) - 10 percentiles, 1280 features
    assert slide_embedding.shape == (10, 1280)
    
    # When flattened, should match pre-computed matrix dim
    assert slide_embedding.flatten().shape[0] == 12800
    
    print("PASSED: Percentile generator works correctly")


def test_compare_with_precomputed():
    """Verify our generator produces compatible embeddings."""
    print("\n" + "=" * 60)
    print("Test 5: Embedding Compatibility Check")
    print("=" * 60)
    
    # Load pre-computed
    matrix = np.load(MATRIX_PATH)
    
    print(f"Pre-computed matrix shape: {matrix.shape}")
    print(f"Expected: (2921 slides, 12800 features)")
    print(f"This represents: 10 percentiles x 1280 Virchow2 features")
    
    # Verify structure matches our generator output
    generator = PercentileSlideEmbeddingGenerator(
        percentiles=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        feature_dim=1280
    )
    
    assert matrix.shape[1] == generator.embedding_dim
    print(f"Generator embedding_dim: {generator.embedding_dim}")
    print(f"Pre-computed dim: {matrix.shape[1]}")
    
    print("PASSED: Embeddings are compatible")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Virchow2 + Sampler (Percentile) Integration Tests")
    print("=" * 60)
    print()
    
    if not os.path.exists(MATRIX_PATH):
        print(f"ERROR: Matrix file not found: {MATRIX_PATH}")
        return 1
    
    try:
        test_pepath_config()
        test_slide_embedding_factory()
        test_slide_search_with_precomputed()
        test_percentile_generator()
        test_compare_with_precomputed()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
