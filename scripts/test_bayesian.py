#!/usr/bin/env python
"""
Test script for prosper.bayesian — compute disease probabilities for a query slide.

Usage:
    python scripts/test_bayesian.py --slide 4261

Outputs results to output/bayesian/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prosper.bayesian import compute_disease_probs


def main():
    parser = argparse.ArgumentParser(description="Test Bayesian disease probabilities")
    parser.add_argument("--slide", default="4261", help="Query slide ID (default: 4261)")
    parser.add_argument("--embedding", default="data/final_slides_sampler_matrix.pkl",
                        help="Embedding matrix pickle")
    parser.add_argument("--metadata", default="data/final_scan_id_disease_map.csv",
                        help="Metadata CSV with Tumor.group and Slide.Scan.File")
    parser.add_argument("--n-neighbors", type=int, default=20,
                        help="Number of nearest neighbors (default: 20)")
    parser.add_argument("--temperature", type=float, default=20.0,
                        help="Temperature parameter T (default: 20.0)")
    parser.add_argument("--outdir", default="output/bayesian",
                        help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load embeddings ---
    print(f"Loading embeddings from: {args.embedding}")
    emb = pd.read_pickle(args.embedding)
    emb.index = emb.index.astype(str)
    slide_ids = emb.index.tolist()
    print(f"  Embedding matrix: {emb.shape} ({len(slide_ids)} slides)")

    if args.slide not in slide_ids:
        print(f"ERROR: Slide {args.slide} not found in embedding matrix", file=sys.stderr)
        sys.exit(1)

    # --- Load metadata ---
    print(f"Loading metadata from: {args.metadata}")
    meta = pd.read_csv(args.metadata)
    # Standardize columns
    if 'Slide.Scan.File' in meta.columns:
        meta = meta.rename(columns={'Slide.Scan.File': 'slide_id', 'Tumor.group': 'tumor_group'})
    meta['slide_id'] = meta['slide_id'].astype(int).astype(str)
    meta = meta.dropna(subset=['tumor_group'])
    # Keep only slides present in embeddings
    meta = meta[meta['slide_id'].isin(slide_ids)].drop_duplicates(subset=['slide_id'])
    meta = meta.set_index('slide_id')
    print(f"  Metadata: {len(meta)} slides with labels")

    # --- Build reference set (exclude query slide) ---
    query_idx = slide_ids.index(args.slide)
    ref_ids = [s for s in slide_ids if s in meta.index and s != args.slide]
    ref_labels = [meta.loc[s, 'tumor_group'] for s in ref_ids]
    ref_indices = [slide_ids.index(s) for s in ref_ids]
    print(f"  Reference set: {len(ref_ids)} slides (query {args.slide} excluded)")
    print(f"  Unique diseases: {len(set(ref_labels))}")

    # --- Compute distances from query to all reference slides ---
    print("\nComputing L1 distances from query to reference slides...")
    query_emb = emb.values[query_idx:query_idx+1, :]   # (1, D)
    ref_emb = emb.values[ref_indices, :]                # (N_ref, D)
    dist_matrix = cdist(query_emb, ref_emb, metric='cityblock')  # (1, N_ref)
    print(f"  Distance matrix shape: {dist_matrix.shape}")

    # --- Compute disease probabilities ---
    print(f"\nComputing disease probabilities (k={args.n_neighbors}, T={args.temperature})...")
    list_of_dicts, df = compute_disease_probs(
        distance_matrix=dist_matrix,
        class_labels=ref_labels,
        n_neighbors=args.n_neighbors,
        T=args.temperature,
        test_slide_ids=[args.slide],
    )

    probs = list_of_dicts[0]

    # --- Sort by probability descending ---
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    # --- Display results ---
    query_label = meta.loc[args.slide, 'tumor_group'] if args.slide in meta.index else "Unknown"
    print(f"\n{'='*60}")
    print(f"Query slide: {args.slide}  (true label: {query_label})")
    print(f"{'='*60}")
    print(f"\n  Top disease probabilities:")
    for rank, (disease, prob) in enumerate(sorted_probs[:10], 1):
        marker = " <-- TRUE" if disease == query_label else ""
        print(f"    {rank:2d}. {disease:<25s}  {prob:.4f}{marker}")
    print(f"\n  Total diseases: {len(probs)}")
    print(f"  Sum of probabilities: {sum(probs.values()):.6f}")

    # --- Save outputs ---
    # 1. Full probability JSON
    output_json = {
        "query_slide": args.slide,
        "true_label": query_label,
        "n_neighbors": args.n_neighbors,
        "temperature": args.temperature,
        "n_reference_slides": len(ref_ids),
        "probabilities": {d: round(p, 6) for d, p in sorted_probs},
        "top_prediction": sorted_probs[0][0],
        "top_probability": round(sorted_probs[0][1], 6),
        "correct": sorted_probs[0][0] == query_label,
    }
    json_path = outdir / f"disease_probs_{args.slide}.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"\n  Saved probabilities JSON: {json_path}")

    # 2. Full probability DataFrame CSV
    csv_path = outdir / f"disease_probs_{args.slide}.csv"
    df.to_csv(csv_path)
    print(f"  Saved probabilities CSV:  {csv_path}")

    # --- Summary ---
    predicted = sorted_probs[0][0]
    correct = predicted == query_label
    print(f"\n  Prediction: {predicted} ({'CORRECT ✓' if correct else 'INCORRECT ✗'})")


if __name__ == "__main__":
    main()
