#!/usr/bin/env python
"""
PROSPER Command Line Interface

Run the PROSPER pipeline from the command line.

Usage:
    python -m prosper.cli --config configs/config.yaml
    python -m prosper.cli --slide-path /path/to/slides --output-dir /path/to/output
    python -m prosper.cli --slide-path /path/to/slides --patch-embedding uni2 --slide-embedding ot
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prosper import ProsperPipeline, ProsperConfig
from prosper.embeddings import list_patch_embedding_models, list_slide_embedding_methods


def parse_args():
    parser = argparse.ArgumentParser(
        description="PROSPER: Pediatric Retrieval for Oncology Slides, Pathology Evidence, and Reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python -m prosper.cli --config configs/config.example.yaml

  # Run with command line arguments (slide-level search, default models)
  python -m prosper.cli --slide-path /path/to/slides --output-dir ./output

  # Choose patch and slide embedding methods
  python -m prosper.cli --slide-path /path/to/slides \\
      --patch-embedding uni2 --slide-embedding ot

  # Run ROI search for a specific 512x512 region (upper-left corner at x=1024,y=2048)
  python -m prosper.cli --slide-path /path/to/slides --roi-search --coords 1024,2048

  # Enable Bayesian disease probability estimation
  python -m prosper.cli --slide-path /path/to/slides --proba

  # Enable LLM summarization (requires API credentials)
  python -m prosper.cli --slide-path /path/to/slides --llm

  # Full pipeline with all optional modules
  python -m prosper.cli --slide-path /path/to/slides \\
      --patch-embedding virchow2 --slide-embedding percentile \\
      --proba --llm

  # Run smoke test with GigaPath model
  python -m prosper.cli --slide-path /path/to/slides --patch-embedding gigapath --smoketest

  # Process single slide against reference database
  python -m prosper.cli --single-slide /path/to/slide.svs --reference-db /path/to/db

  # Generate OT visualizations
  python -m prosper.cli --slide-path /path/to/slides \\
      --slide-embedding ot --generate-visualizations

Available models and methods:
  Patch embedding models: virchow2 (default), uni2, uni, gigapath
  Slide embedding methods: percentile (default), bob, ot
        """
    )

    # Config file option
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file"
    )

    # Basic paths
    parser.add_argument(
        "--slide-path",
        type=str,
        help="Path to directory containing slide files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--reference-db",
        type=str,
        help="Path to reference database for search"
    )

    # Single slide mode
    parser.add_argument(
        "--single-slide",
        type=str,
        help="Process a single slide file"
    )

    # --- Embedding selection (renamed for clarity) ---
    parser.add_argument(
        "--patch-embedding",
        type=str,
        default="virchow2",
        choices=["virchow2", "uni2", "uni", "gigapath"],
        help="Patch embedding model (default: virchow2)"
    )
    # Keep old name as hidden alias for backward compatibility
    parser.add_argument(
        "--patch-model",
        type=str,
        dest="patch_embedding_compat",
        choices=["virchow2", "uni2", "uni", "gigapath"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--slide-embedding",
        type=str,
        default="percentile",
        choices=["percentile", "bob", "ot"],
        help="Slide embedding method (default: percentile)"
    )
    # Keep old name as hidden alias for backward compatibility
    parser.add_argument(
        "--slide-method",
        type=str,
        dest="slide_embedding_compat",
        choices=["percentile", "bob", "ot"],
        help=argparse.SUPPRESS,
    )

    # --- ROI search ---
    parser.add_argument(
        "--roi-search",
        action="store_true",
        help="Perform ROI-level (patch-level) similarity search instead of slide-level"
    )
    parser.add_argument(
        "--coords",
        type=str,
        default=None,
        help=(
            "Upper-left corner coordinates (x,y) of a 512x512 patch for ROI "
            "search. Implies --roi-search. Format: x,y (e.g. 1024,2048)"
        ),
    )

    # --- Optional analysis modules ---
    parser.add_argument(
        "--proba",
        action="store_true",
        help="Compute Bayesian disease probability estimation after search"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Generate LLM-powered narrative summary (requires API credentials)"
    )

    # Processing options
    parser.add_argument(
        "--base-patchsize",
        type=int,
        default=512,
        help="Patch size for tiling (default: 512)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for model inference (default: cuda)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for feature extraction (default: 64)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)"
    )
    parser.add_argument(
        "--n-cpus",
        type=int,
        default=32,
        help="Number of CPUs for parallel processing (default: 32)"
    )

    # Search options
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results to return (default: 10)"
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="l1",
        choices=["l1", "l2", "cosine"],
        help="Distance metric for search (default: l1)"
    )

    # Visualization options
    parser.add_argument(
        "--generate-visualizations",
        action="store_true",
        help="Generate OT-based visualization heatmaps (requires --slide-embedding ot)"
    )
    parser.add_argument(
        "--visualization-level",
        type=int,
        default=2,
        help="OpenSlide level for visualization (default: 2)"
    )

    # Runtime options
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="Run quick test with limited slides"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of all steps"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    # Bayesian options (fine-tuning; --proba enables the module)
    parser.add_argument(
        "--bayesian-neighbors",
        type=int,
        default=20,
        help="Number of nearest neighbors for Bayesian probability (default: 20)"
    )
    parser.add_argument(
        "--bayesian-temperature",
        type=float,
        default=20.0,
        help="Temperature for Bayesian exponential weighting (default: 20.0)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to metadata CSV for Bayesian disease probabilities and LLM"
    )

    # Individual steps (advanced)
    parser.add_argument(
        "--step",
        type=str,
        choices=["mosaic", "patch-embedding", "slide-embedding", "search", "bayesian", "llm", "all"],
        default="all",
        help="Run a specific pipeline step (default: all)"
    )

    # List available models
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available patch embedding models and slide embedding methods"
    )

    args = parser.parse_args()

    # Resolve backward-compat aliases: --patch-model -> --patch-embedding
    if args.patch_embedding_compat is not None:
        args.patch_embedding = args.patch_embedding_compat
    # --slide-method -> --slide-embedding
    if args.slide_embedding_compat is not None:
        args.slide_embedding = args.slide_embedding_compat

    # --coords implies --roi-search
    if args.coords is not None:
        args.roi_search = True

    # Derive the effective search_type from the flags
    args.search_type = "roi" if args.roi_search else "slide"

    return args


def _parse_coords(coords_str: str):
    """Parse 'x,y' coordinate string into (x, y) integers."""
    try:
        parts = coords_str.split(",")
        if len(parts) != 2:
            raise ValueError
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, AttributeError):
        print(f"Error: --coords must be in 'x,y' format (got '{coords_str}')")
        sys.exit(1)


def main():
    args = parse_args()

    # Handle --list-models
    if args.list_models:
        print("\nPROSPER - Available Models and Methods")
        print("=" * 50)
        print("\nPatch Embedding Models:")
        for model in list_patch_embedding_models():
            print(f"  - {model}")
        print("\nSlide Embedding Methods:")
        for method in list_slide_embedding_methods():
            print(f"  - {method}")
        print()
        return

    # Build configuration
    if args.config:
        config = ProsperConfig.from_yaml(args.config)
    else:
        if not args.slide_path and not args.single_slide:
            print("Error: Either --config or --slide-path/--single-slide is required")
            print("Use --list-models to see available models and methods")
            sys.exit(1)

        config = ProsperConfig(
            slide_path=args.slide_path or "",
            output_dir=args.output_dir,
            reference_db_path=args.reference_db,
            base_patchsize=args.base_patchsize,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_cpus=args.n_cpus,
            top_k=args.top_k,
            distance_metric=args.distance_metric,
            patch_model=args.patch_embedding,
            slide_embedding_method=args.slide_embedding,
            generate_visualizations=args.generate_visualizations,
            visualization_slide_level=args.visualization_level,
            smoketest=args.smoketest,
            force_recompute=args.force_recompute,
            verbose=not args.quiet,
        )

    # Override config with explicit command-line arguments
    if args.slide_path:
        config.slide_path = args.slide_path
    if args.output_dir != "./output":
        config.output_dir = args.output_dir
    if args.patch_embedding != "virchow2":
        config.patch_model = args.patch_embedding
    if args.slide_embedding != "percentile":
        config.slide_embedding_method = args.slide_embedding
    if args.smoketest:
        config.smoketest = True
    if args.force_recompute:
        config.force_recompute = True
    if args.quiet:
        config.verbose = False
    if args.generate_visualizations:
        config.generate_visualizations = True
    if args.bayesian_neighbors != 20:
        config.bayesian_n_neighbors = args.bayesian_neighbors
    if args.bayesian_temperature != 20.0:
        config.bayesian_temperature = args.bayesian_temperature

    # Validate visualization requirements
    if config.generate_visualizations and config.slide_embedding_method != 'ot':
        print("Warning: Visualizations are only generated for the OT slide embedding method.")
        print("         Setting slide_embedding_method to 'ot'")
        config.slide_embedding_method = 'ot'

    # Initialize pipeline
    pipeline = ProsperPipeline(config)

    # Handle single slide mode
    if args.single_slide:
        if not args.reference_db:
            print("Error: --reference-db is required for single slide mode")
            sys.exit(1)

        results = pipeline.process_single_slide(
            slide_path=args.single_slide,
            reference_db_path=args.reference_db,
            search_type=args.search_type,
        )

        # Print results
        df = results.to_dataframe(args.search_type)
        print("\nSearch Results:")
        print(df.to_string())
        return

    # --- Run pipeline steps ---
    if args.step == "all":
        # Full pipeline: mosaic -> patch-embed -> slide-embed -> search
        results = pipeline.run_core(search_type=args.search_type)

        # Optional: Bayesian disease probabilities (--proba)
        if args.proba:
            try:
                disease_probs = pipeline.compute_disease_probabilities(
                    metadata_path=args.metadata,
                    n_neighbors=args.bayesian_neighbors,
                    temperature=args.bayesian_temperature,
                )
                results.disease_probabilities = disease_probs
            except Exception as e:
                if config.verbose:
                    print(f"  Warning: Bayesian disease probabilities skipped: {e}")

        # Optional: LLM summarization (--llm)
        if args.llm:
            try:
                summaries = pipeline.generate_llm_summary(
                    disease_probs=results.disease_probabilities,
                )
                if summaries:
                    results.llm_summaries = summaries
            except Exception as e:
                if config.verbose:
                    print(f"  Warning: LLM summarization skipped: {e}")

    elif args.step == "mosaic":
        pipeline.generate_mosaics()
    elif args.step == "patch-embedding":
        pipeline.extract_patch_embeddings()
    elif args.step == "slide-embedding":
        pipeline.generate_slide_embeddings()
    elif args.step == "search":
        results = pipeline.search(search_type=args.search_type)
        df = results.to_dataframe(args.search_type)
        print("\nSearch Results:")
        print(df.head(20).to_string())
    elif args.step == "bayesian":
        disease_probs = pipeline.compute_disease_probabilities(
            metadata_path=args.metadata,
            n_neighbors=args.bayesian_neighbors,
            temperature=args.bayesian_temperature,
        )
        print(f"\nBayesian disease probabilities computed for {len(disease_probs)} slides")
    elif args.step == "llm":
        summaries = pipeline.generate_llm_summary()
        print(f"\nLLM summaries generated for {len(summaries)} slides")

    print("\n✓ Pipeline complete!")
    print(f"  Patch embedding model: {config.patch_model}")
    print(f"  Slide embedding method: {config.slide_embedding_method}")
    if args.proba:
        print(f"  Bayesian probability: enabled")
    if args.llm:
        print(f"  LLM summarization: enabled")
    if config.generate_visualizations:
        print(f"  Visualizations saved to: {config.visualizations_dir}")


if __name__ == "__main__":
    main()
