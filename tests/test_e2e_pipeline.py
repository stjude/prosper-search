"""
End-to-end test of the virchow2 + percentile (sampler) pipeline.

This script runs the complete pipeline:
1. Generate mosaics from the SVS slide
2. Extract patch embeddings using Virchow2
3. Generate slide embeddings using percentile method
4. Search against pre-computed reference database
5. Save results to output folder
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prosper import ProsperConfig, ProsperPipeline
from prosper.preprocessing.mosaic import process_slide, MosaicGenerator
from prosper.embeddings.patch_embedding import get_patch_embedding_extractor
from prosper.embeddings.slide_embedding import get_slide_embedding_generator
from prosper.search.slide_search import SlideSearchEngine

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SLIDE_PATH = os.path.join(DATA_DIR, "4261.svs")
MATRIX_PATH = os.path.join(DATA_DIR, "final_slides_sampler_matrix.pkl")  # Now a DataFrame with slide IDs as index
METADATA_PATH = os.path.join(DATA_DIR, "final_scan_id_disease_map_rich_data.csv")


def step1_generate_mosaic():
    """Step 1: Generate mosaic from the slide."""
    print("=" * 60)
    print("Step 1: Generate Mosaic")
    print("=" * 60)
    
    slide_id = os.path.basename(SLIDE_PATH).replace('.svs', '')
    mosaic_dir = os.path.join(OUTPUT_DIR, "mosaic")
    os.makedirs(mosaic_dir, exist_ok=True)
    
    cache_path = os.path.join(mosaic_dir, f"{slide_id}.pkl")
    
    # Check if already cached
    if os.path.exists(cache_path):
        print(f"Loading cached mosaic from {cache_path}")
        with open(cache_path, 'rb') as f:
            mosaic = pickle.load(f)
    else:
        print(f"Processing slide: {SLIDE_PATH}")
        mosaic = process_slide(
            slide_fn=SLIDE_PATH,
            base_patchsize=512,
            percentage_selected=15,
            tissue_threshold=0.7
        )
        
        if mosaic is not None:
            # Cache the mosaic
            with open(cache_path, 'wb') as f:
                pickle.dump(mosaic, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Mosaic saved to {cache_path}")
    
    if mosaic is None:
        raise RuntimeError("Failed to generate mosaic")
    
    print(f"Mosaic contains {len(mosaic)} patches")
    return mosaic, slide_id


def step2_extract_patch_embeddings(mosaic, slide_id):
    """Step 2: Extract patch embeddings using Virchow2."""
    print("\n" + "=" * 60)
    print("Step 2: Extract Patch Embeddings (Virchow2)")
    print("=" * 60)
    
    patch_emb_dir = os.path.join(OUTPUT_DIR, "patch_embeddings", "tmp")
    os.makedirs(patch_emb_dir, exist_ok=True)
    
    cache_path = os.path.join(patch_emb_dir, f"{slide_id}.pkl")
    
    # Check if already cached (from this run)
    if os.path.exists(cache_path):
        print(f"Loading cached patch embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, tuple):
                patch_embeddings = data[0]
            else:
                patch_embeddings = data
        print(f"Patch embeddings shape: {patch_embeddings.shape}")
        return patch_embeddings
    
    # Extract using Virchow2 model - NO FALLBACK
    print("Initializing Virchow2 extractor...")
    extractor = get_patch_embedding_extractor(
        model_name="virchow2",
        device="cuda",
        batch_size=64,
        num_workers=4,
        verbose=True
    )
    
    print(f"Extracting embeddings for {len(mosaic)} patches...")
    patch_embeddings = extractor.extract_slide_embeddings(
        slide_path=SLIDE_PATH,
        mosaic=mosaic,
        base_patchsize=512
    )
    
    if patch_embeddings is None:
        raise RuntimeError("Failed to extract patch embeddings")
    
    # Cache with mosaic
    with open(cache_path, 'wb') as f:
        pickle.dump((patch_embeddings, mosaic), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Patch embeddings saved to {cache_path}")
    print(f"Patch embeddings shape: {patch_embeddings.shape}")
    return patch_embeddings


def step3_generate_slide_embedding(patch_embeddings, slide_id):
    """Step 3: Generate slide embedding using percentile method."""
    print("\n" + "=" * 60)
    print("Step 3: Generate Slide Embedding (Percentile/Sampler)")
    print("=" * 60)
    
    slide_emb_dir = os.path.join(OUTPUT_DIR, "slide_embeddings")
    os.makedirs(slide_emb_dir, exist_ok=True)
    
    cache_path = os.path.join(slide_emb_dir, f"{slide_id}.pkl")
    
    # Check if already cached (from this run)
    if os.path.exists(cache_path):
        print(f"Loading cached slide embedding from {cache_path}")
        with open(cache_path, 'rb') as f:
            slide_embedding = pickle.load(f)
    else:
        generator = get_slide_embedding_generator(
            method_name="percentile",
            feature_dim=1280,  # Use first 1280 dims of Virchow2's 2560-dim output
            percentiles=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
            n_workers=1,
            verbose=True
        )
        
        print(f"Input patch embeddings: {patch_embeddings.shape}")
        slide_embedding = generator.process_single_slide(
            patch_embeddings=patch_embeddings,
            slide_id=slide_id,
            mosaic=None
        )
        
        # Save
        with open(cache_path, 'wb') as f:
            pickle.dump(slide_embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Slide embedding saved to {cache_path}")
    
    print(f"Slide embedding shape: {slide_embedding.shape}")
    print(f"Flattened dimension: {slide_embedding.flatten().shape[0]}")
    return slide_embedding


def step4_slide_search(slide_embedding, slide_id):
    """Step 4: Search against reference database."""
    print("\n" + "=" * 60)
    print("Step 4: Slide Search")
    print("=" * 60)
    
    # Load reference database (DataFrame with slide IDs as index)
    print(f"Loading reference database from {MATRIX_PATH}")
    with open(MATRIX_PATH, 'rb') as f:
        ref_df = pickle.load(f)
    
    # Extract matrix and slide IDs from DataFrame
    ref_slide_ids = [str(sid) for sid in ref_df.index.tolist()]
    ref_matrix = ref_df.values
    
    # Load metadata for disease labels
    metadata = pd.read_csv(METADATA_PATH)
    # Normalize column names (rich_data uses slide_id / tumor_group)
    if 'Slide.Scan.File' in metadata.columns:
        metadata = metadata.rename(columns={'Slide.Scan.File': 'slide_id', 'Tumor.group': 'tumor_group'})
    metadata['slide_id'] = metadata['slide_id'].astype(int).astype(str)
    
    print(f"Reference database: {len(ref_slide_ids)} slides, dim {ref_matrix.shape[1]}")
    
    # Initialize search engine
    search_engine = SlideSearchEngine(
        distance_metric='cityblock',  # L1 distance
        top_k=20,
        verbose=True
    )
    search_engine.reference_matrix = ref_matrix
    search_engine.reference_slide_ids = ref_slide_ids
    
    # Flatten query embedding
    query_embedding = slide_embedding.flatten()
    print(f"Query embedding dimension: {query_embedding.shape[0]}")
    
    # Verify dimensions match
    if query_embedding.shape[0] != ref_matrix.shape[1]:
        print(f"WARNING: Dimension mismatch! Query: {query_embedding.shape[0]}, Ref: {ref_matrix.shape[1]}")
        # Try to match by padding or truncating
        if query_embedding.shape[0] < ref_matrix.shape[1]:
            query_embedding = np.pad(query_embedding, (0, ref_matrix.shape[1] - query_embedding.shape[0]))
        else:
            query_embedding = query_embedding[:ref_matrix.shape[1]]
    
    # Search
    print(f"\nSearching for slide {slide_id}...")
    results = search_engine.search(
        query_embedding=query_embedding,
        query_slide_id=slide_id,
        exclude_self=True
    )
    
    # Display results with disease labels
    print(f"\nTop-20 Similar Slides:")
    print("-" * 70)
    print(f"{'Rank':<6}{'Slide ID':<12}{'Disease':<12}{'Distance':<12}")
    print("-" * 70)
    
    for r in results:
        hit_disease = metadata[metadata['slide_id'] == r.hit_slide_id]['tumor_group'].values
        disease_str = hit_disease[0] if len(hit_disease) > 0 else "Unknown"
        print(f"{r.rank:<6}{r.hit_slide_id:<12}{disease_str:<12}{r.distance:<12.2f}")
    
    return results


def step5_save_results(results, slide_id):
    """Step 5: Save search results to output folder."""
    print("\n" + "=" * 60)
    print("Step 5: Save Results")
    print("=" * 60)
    
    results_dir = os.path.join(OUTPUT_DIR, "search_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as pickle
    pkl_path = os.path.join(results_dir, f"{slide_id}_search_results.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump([r.to_dict() for r in results], f)
    print(f"Results saved to {pkl_path}")
    
    # Save as CSV for easy viewing
    csv_path = os.path.join(results_dir, f"{slide_id}_search_results.csv")
    df = pd.DataFrame([r.to_dict() for r in results])
    
    # Add disease labels
    metadata = pd.read_csv(METADATA_PATH)
    if 'Slide.Scan.File' in metadata.columns:
        metadata = metadata.rename(columns={'Slide.Scan.File': 'slide_id', 'Tumor.group': 'tumor_group'})
    metadata['slide_id'] = metadata['slide_id'].astype(int).astype(str)
    disease_map = dict(zip(metadata['slide_id'], metadata['tumor_group']))
    df['hit_disease'] = df['hit_slide_id'].map(disease_map)
    
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    return pkl_path, csv_path


def main():
    """Run the full end-to-end pipeline."""
    print("=" * 60)
    print("Virchow2 + Sampler (Percentile) End-to-End Pipeline")
    print("=" * 60)
    print()
    
    # Check that files exist
    if not os.path.exists(SLIDE_PATH):
        print(f"ERROR: Slide not found: {SLIDE_PATH}")
        return 1
    
    if not os.path.exists(MATRIX_PATH):
        print(f"ERROR: Reference matrix not found: {MATRIX_PATH}")
        return 1
    
    print(f"Slide: {SLIDE_PATH}")
    with open(MATRIX_PATH, 'rb') as f:
        ref_df = pickle.load(f)
    print(f"Reference DB: {MATRIX_PATH} ({len(ref_df)} slides)")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    try:
        # Run pipeline
        mosaic, slide_id = step1_generate_mosaic()
        patch_embeddings = step2_extract_patch_embeddings(mosaic, slide_id)
        slide_embedding = step3_generate_slide_embedding(patch_embeddings, slide_id)
        results = step4_slide_search(slide_embedding, slide_id)
        pkl_path, csv_path = step5_save_results(results, slide_id)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"\nOutput files:")
        print(f"  - Mosaic: {os.path.join(OUTPUT_DIR, 'mosaic', slide_id + '.pkl')}")
        print(f"  - Patch embeddings: {os.path.join(OUTPUT_DIR, 'patch_embeddings/tmp', slide_id + '.pkl')}")
        print(f"  - Slide embedding: {os.path.join(OUTPUT_DIR, 'slide_embeddings', slide_id + '.pkl')}")
        print(f"  - Search results (pkl): {pkl_path}")
        print(f"  - Search results (csv): {csv_path}")
        
        return 0
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
