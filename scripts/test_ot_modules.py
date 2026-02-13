#!/usr/bin/env python3
"""
Test script for OT embedding and OT visualization modules.
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ot_embedding():
    """Test the OT embedding module."""
    print("=" * 60)
    print("Testing OT Embedding Module")
    print("=" * 60)
    
    # Check if POT is installed
    try:
        import ot
        print(f"✓ POT (Python Optimal Transport) version: {ot.__version__}")
    except ImportError:
        print("✗ POT not installed. Install with: pip install POT")
        return False
    
    # Import OT embedding module
    try:
        from prosper.embeddings.slide_embedding import OTSlideEmbeddingGenerator
        print("✓ OTSlideEmbeddingGenerator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OTSlideEmbeddingGenerator: {e}")
        return False
    
    # Initialize generator
    try:
        ot_gen = OTSlideEmbeddingGenerator(
            metric="cosine",
            reg=0.0,  # Exact EMD
            feature_dim=1280,
            verbose=True
        )
        print("✓ OTSlideEmbeddingGenerator initialized")
    except Exception as e:
        print(f"✗ Failed to initialize OTSlideEmbeddingGenerator: {e}")
        return False
    
    # Load some patch embeddings
    embeddings_dir = "REDACTED_INTERNAL_PATH/ImageAnalysis/Core/Haoran/comet-github/data/patch_embeddings"
    npz_files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith('.npz')])[:2]
    
    if len(npz_files) < 2:
        print("✗ Need at least 2 .npz files for testing")
        return False
    
    print(f"\nLoading embeddings from: {npz_files[0]} and {npz_files[1]}")
    
    # Load first slide embeddings
    data1 = np.load(os.path.join(embeddings_dir, npz_files[0]))
    embeddings1 = data1['embeddings']
    coords1 = data1['coords']
    slide1_id = npz_files[0].replace('.npz', '')
    
    # Load second slide embeddings
    data2 = np.load(os.path.join(embeddings_dir, npz_files[1]))
    embeddings2 = data2['embeddings']
    coords2 = data2['coords']
    slide2_id = npz_files[1].replace('.npz', '')
    
    print(f"  Slide {slide1_id}: {embeddings1.shape} embeddings")
    print(f"  Slide {slide2_id}: {embeddings2.shape} embeddings")
    
    # Process slides
    try:
        result1 = ot_gen.process_single_slide(embeddings1.astype(np.float32), slide1_id)
        result2 = ot_gen.process_single_slide(embeddings2.astype(np.float32), slide2_id)
        print(f"✓ Processed slides successfully")
        print(f"  Slide {slide1_id}: {result1['n_patches']} patches, features shape: {result1['features'].shape}")
        print(f"  Slide {slide2_id}: {result2['n_patches']} patches, features shape: {result2['features'].shape}")
    except Exception as e:
        print(f"✗ Failed to process slides: {e}")
        return False
    
    # Compute OT distance
    print("\nComputing OT distance...")
    try:
        distance = ot_gen.compute_ot_distance(result1['features'], result2['features'])
        print(f"✓ OT distance computed: {distance:.6f}")
    except Exception as e:
        print(f"✗ Failed to compute OT distance: {e}")
        return False
    
    # Compute detailed similarity with transport matrix
    print("\nComputing detailed similarity with transport matrix...")
    try:
        detailed_result = ot_gen.compute_detailed_similarity(result1, result2)
        print(f"✓ Detailed similarity computed")
        print(f"  Distance: {detailed_result['distance']:.6f}")
        print(f"  Cost matrix M shape: {detailed_result['M'].shape}")
        print(f"  Transport matrix T shape: {detailed_result['T'].shape}")
        print(f"  T sum: {detailed_result['T'].sum():.6f} (should be ~1.0)")
    except Exception as e:
        print(f"✗ Failed to compute detailed similarity: {e}")
        return False
    
    print("\n✓ OT Embedding module tests PASSED!")
    return detailed_result


def test_ot_visualization(ot_result=None):
    """Test the OT visualization module."""
    print("\n" + "=" * 60)
    print("Testing OT Visualization Module")
    print("=" * 60)
    
    # Check dependencies
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib available")
    except ImportError:
        print("✗ matplotlib not installed")
        return False
    
    try:
        import networkx as nx
        print(f"✓ networkx version: {nx.__version__}")
    except ImportError:
        print("⚠ networkx not installed (graph clustering will be unavailable)")
    
    try:
        import openslide
        print(f"✓ openslide-python available")
        has_openslide = True
    except ImportError:
        print("⚠ openslide not installed (heatmap overlay on slides unavailable)")
        has_openslide = False
    
    # Import visualization functions
    try:
        from prosper.visualization.ot_visualization import (
            transport_matrix_to_long_table,
            compute_per_patch_weighted_distance_scores,
            normalize_scores,
            scores_to_patch_dataframe,
            sparsify_transport_matrix,
            cluster_patches_from_transport,
            OTVisualizer
        )
        print("✓ OT visualization functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OT visualization functions: {e}")
        return False
    
    # Use provided OT result or create synthetic one
    if ot_result is None:
        print("\nCreating synthetic OT result for testing...")
        n_a, n_b = 50, 60
        # Simulate transport matrix (should sum to 1)
        T = np.random.rand(n_a, n_b)
        T = T / T.sum()
        # Simulate cost matrix
        M = np.random.rand(n_a, n_b)
        ot_result = {'T': T, 'M': M, 'distance': 0.5}
    
    T = ot_result['T']
    M = ot_result['M']
    
    # Test transport_matrix_to_long_table
    print("\nTesting transport_matrix_to_long_table...")
    try:
        long_table = transport_matrix_to_long_table(T)
        print(f"✓ Transport matrix converted to long table: {len(long_table)} non-zero entries")
        print(f"  Top 3 transport values:\n{long_table.head(3)}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test compute_per_patch_weighted_distance_scores
    print("\nTesting compute_per_patch_weighted_distance_scores...")
    try:
        D_A, D_B = compute_per_patch_weighted_distance_scores(T, M)
        print(f"✓ Per-patch scores computed")
        print(f"  D_A shape: {D_A.shape}, range: [{D_A.min():.4f}, {D_A.max():.4f}]")
        print(f"  D_B shape: {D_B.shape}, range: [{D_B.min():.4f}, {D_B.max():.4f}]")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test normalize_scores
    print("\nTesting normalize_scores...")
    try:
        D_A_norm = normalize_scores(D_A)
        print(f"✓ Scores normalized: range [{D_A_norm.min():.4f}, {D_A_norm.max():.4f}]")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test scores_to_patch_dataframe
    print("\nTesting scores_to_patch_dataframe...")
    try:
        df_a = scores_to_patch_dataframe(D_A)
        print(f"✓ Scores converted to DataFrame: {len(df_a)} rows")
        print(f"  Columns: {list(df_a.columns)}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test sparsify_transport_matrix
    print("\nTesting sparsify_transport_matrix...")
    try:
        T_sparse = sparsify_transport_matrix(T, k=3, method="mutual_topk")
        nonzero = np.count_nonzero(T_sparse)
        print(f"✓ Transport matrix sparsified: {nonzero} non-zero entries (from {np.count_nonzero(T)})")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test cluster_patches_from_transport
    print("\nTesting cluster_patches_from_transport...")
    try:
        df_a_clusters, df_b_clusters = cluster_patches_from_transport(T, method="connected_components")
        n_clusters_a = df_a_clusters['normalized_score'].nunique()
        n_clusters_b = df_b_clusters['normalized_score'].nunique()
        print(f"✓ Patches clustered")
        print(f"  Slide A: {len(df_a_clusters)} patches in {n_clusters_a} clusters")
        print(f"  Slide B: {len(df_b_clusters)} patches in {n_clusters_b} clusters")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test OTVisualizer initialization
    print("\nTesting OTVisualizer...")
    try:
        visualizer = OTVisualizer(
            patch_size=512,
            slide_level=2,
            alpha=100,
            output_dir="REDACTED_INTERNAL_PATH/ImageAnalysis/Core/Haoran/comet-github/output/ot_visualizations"
        )
        print(f"✓ OTVisualizer initialized")
        print(f"  Output directory: {visualizer.output_dir}")
    except Exception as e:
        print(f"✗ Failed to initialize OTVisualizer: {e}")
        return False
    
    print("\n✓ OT Visualization module tests PASSED!")
    return True


def main():
    print("\n" + "=" * 60)
    print("PROSPER OT Module Tests")
    print("=" * 60 + "\n")
    
    # Test OT embedding
    ot_result = test_ot_embedding()
    
    if ot_result:
        # Test visualization with real OT result
        test_ot_visualization(ot_result)
    else:
        # Test visualization with synthetic data
        test_ot_visualization(None)
    
    print("\n" + "=" * 60)
    print("All Tests Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
