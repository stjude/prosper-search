"""CLI tool to build LLM-ready JSON for a single slide ID.

Generates a structured JSON file containing the query slide's metadata
and its top-k nearest neighbors with similarity scores. Optionally
includes Bayesian disease probabilities from the bayesian module output.

Usage:
    python -m prosper.llm.search.slide_to_json \
        --sim data/slide_embeddings/virchow2_sampler.pkl \
        --metadata data/metadata/final_scan_id_disease_map_rich_data.csv \
        --slide 4261 \
        --out output/llm/slide_4261.json

    # With Bayesian probabilities:
    python -m prosper.llm.search.slide_to_json \
        --sim data/slide_embeddings/virchow2_sampler.pkl \
        --metadata data/metadata/final_scan_id_disease_map_rich_data.csv \
        --slide 4261 \
        --out output/llm/slide_4261.json \
        --bayesian output/bayesian/disease_probs_4261.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis_lib import load_results_new, load_metadata, build_json_for_slide


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate LLM-ready neighbor JSON for one slide (PROSPER)"
    )
    p.add_argument(
        "--sim", required=True,
        help="Similarity/embedding matrix pickle or CSV. "
             "If non-square (embedding matrix), pairwise L1 distances will be computed."
    )
    p.add_argument(
        "--metadata", required=True,
        help="Metadata CSV with disease labels (auto-detects column names)"
    )
    p.add_argument(
        "--slide", required=True,
        help="Query slide ID (e.g. 4261)"
    )
    p.add_argument(
        "--out", required=True,
        help="Output JSON path"
    )
    p.add_argument(
        "--k", type=int, default=5,
        help="Number of neighbors (default: 5)"
    )
    p.add_argument(
        "--bayesian",
        type=str, default=None,
        help="Path to Bayesian disease probability JSON from prosper.bayesian "
             "(adds 'probabilities' key with non-zero entries to output)"
    )
    return p.parse_args()


def load_bayesian_probabilities(bayesian_path: str) -> dict:
    """Load Bayesian probabilities and return only non-zero entries.

    Args:
        bayesian_path: Path to disease_probs_<slide>.json from the bayesian module.

    Returns:
        Dict of {disease: probability} with only non-zero values, sorted descending.
    """
    with open(bayesian_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    probs = data.get("probabilities", {})
    # Keep only non-zero entries, sorted by probability descending
    nonzero = {d: p for d, p in probs.items() if p > 1e-8}
    return dict(sorted(nonzero.items(), key=lambda x: x[1], reverse=True))


def main() -> None:
    args = parse_args()

    print(f"Loading similarity matrix from: {args.sim}")
    sim = load_results_new(args.sim)

    print(f"Loading metadata from: {args.metadata}")
    metadata = load_metadata(args.metadata)

    print(f"\nBuilding JSON for slide {args.slide} with k={args.k} neighbors...")
    payload = build_json_for_slide(sim, metadata, slide_id=str(args.slide), k=args.k)

    # Attach Bayesian disease probabilities if provided
    if args.bayesian:
        print(f"\nLoading Bayesian probabilities from: {args.bayesian}")
        probs = load_bayesian_probabilities(args.bayesian)
        payload['probabilities'] = probs
        print(f"  Added {len(probs)} non-zero disease probabilities")
        # Show top 5
        for i, (disease, prob) in enumerate(list(probs.items())[:5], 1):
            print(f"    {i}. {disease}: {prob:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote JSON with {len(payload['neighbors'])} neighbors to {out_path}")
    print(f"\nQuery case: {payload['current_case']['case_id']}")
    print(f"  Disease: {payload['current_case'].get('tumor_group', 'Unknown')}")
    print(f"\nNeighbors:")
    for i, n in enumerate(payload['neighbors'], 1):
        print(f"  {i}. Slide {n['case_id']} ({n.get('tumor_group', '?')}) - distance: {n['similarity']:.2f}")


if __name__ == "__main__":  # pragma: no cover
    main()
