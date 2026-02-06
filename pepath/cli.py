#!/usr/bin/env python
"""
PePath Command Line Interface

Run the PePath pipeline from the command line.

Usage:
    python -m pepath.cli --config configs/config.yaml
    python -m pepath.cli --slide-path /path/to/slides --output-dir /path/to/output
    python -m pepath.cli --slide-path /path/to/slides --patch-model uni2 --slide-method ot
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pepath import PePathPipeline, PePathConfig
from pepath.embeddings import list_patch_embedding_models, list_slide_embedding_methods


def parse_args():
    parser = argparse.ArgumentParser(
        description="PePath: Pediatric Pathology Slide Search Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python -m pepath.cli --config configs/config.example.yaml

  # Run with command line arguments
  python -m pepath.cli --slide-path /path/to/slides --output-dir ./output

  # Use specific patch model and slide embedding method
  python -m pepath.cli --slide-path /path/to/slides --patch-model uni2 --slide-method ot

  # Run smoke test with GigaPath model
  python -m pepath.cli --slide-path /path/to/slides --patch-model gigapath --smoketest

  # Process single slide against reference database
  python -m pepath.cli --single-slide /path/to/slide.svs --reference-db /path/to/db

  # Generate OT visualizations
  python -m pepath.cli --slide-path /path/to/slides --slide-method ot --generate-visualizations

Available models and methods:
  Patch models: virchow2, uni2, uni, gigapath
  Slide embedding methods: percentile, bob, ot
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
    
    # Model selection
    parser.add_argument(
        "--patch-model",
        type=str,
        default="virchow2",
        choices=["virchow2", "uni2", "uni", "gigapath"],
        help="Patch embedding model to use (default: virchow2)"
    )
    parser.add_argument(
        "--slide-method",
        type=str,
        default="percentile",
        choices=["percentile", "bob", "ot"],
        help="Slide embedding method to use (default: percentile)"
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
        "--search-type",
        type=str,
        default="slide",
        choices=["slide", "roi"],
        help="Type of search: slide-level or ROI-level (default: slide)"
    )
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
        help="Generate OT-based visualization heatmaps (requires --slide-method ot)"
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
    
    # Individual steps
    parser.add_argument(
        "--step",
        type=str,
        choices=["mosaic", "patch-embedding", "slide-embedding", "search", "all"],
        default="all",
        help="Run specific pipeline step (default: all)"
    )
    
    # List available models
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available patch models and slide embedding methods"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle --list-models
    if args.list_models:
        print("\n🔬 PePath - Available Models and Methods")
        print("=" * 50)
        print("\nPatch Embedding Models:")
        for model in list_patch_embedding_models():
            print(f"  • {model}")
        print("\nSlide Embedding Methods:")
        for method in list_slide_embedding_methods():
            print(f"  • {method}")
        print()
        return
    
    # Build configuration
    if args.config:
        config = PePathConfig.from_yaml(args.config)
    else:
        if not args.slide_path and not args.single_slide:
            print("Error: Either --config or --slide-path/--single-slide is required")
            print("Use --list-models to see available models and methods")
            sys.exit(1)
        
        config = PePathConfig(
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
            patch_model=args.patch_model,
            slide_embedding_method=args.slide_method,
            generate_visualizations=args.generate_visualizations,
            visualization_slide_level=args.visualization_level,
            smoketest=args.smoketest,
            force_recompute=args.force_recompute,
            verbose=not args.quiet
        )
    
    # Override config with command line arguments
    if args.slide_path:
        config.slide_path = args.slide_path
    if args.output_dir != "./output":
        config.output_dir = args.output_dir
    if args.patch_model != "virchow2":
        config.patch_model = args.patch_model
    if args.slide_method != "percentile":
        config.slide_embedding_method = args.slide_method
    if args.smoketest:
        config.smoketest = True
    if args.force_recompute:
        config.force_recompute = True
    if args.quiet:
        config.verbose = False
    if args.generate_visualizations:
        config.generate_visualizations = True
    
    # Validate visualization requirements
    if config.generate_visualizations and config.slide_embedding_method != 'ot':
        print("Warning: Visualizations are only generated for OT slide embedding method.")
        print("         Setting slide_embedding_method to 'ot'")
        config.slide_embedding_method = 'ot'
    
    # Initialize pipeline
    pipeline = PePathPipeline(config)
    
    # Handle single slide mode
    if args.single_slide:
        if not args.reference_db:
            print("Error: --reference-db is required for single slide mode")
            sys.exit(1)
        
        results = pipeline.process_single_slide(
            slide_path=args.single_slide,
            reference_db_path=args.reference_db,
            search_type=args.search_type
        )
        
        # Print results
        df = results.to_dataframe(args.search_type)
        print("\nSearch Results:")
        print(df.to_string())
        return
    
    # Run pipeline steps
    if args.step == "all":
        results = pipeline.run(search_type=args.search_type)
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
    
    print("\n✓ Pipeline complete!")
    print(f"  Patch model used: {config.patch_model}")
    print(f"  Slide embedding method: {config.slide_embedding_method}")
    if config.generate_visualizations:
        print(f"  Visualizations saved to: {config.visualizations_dir}")


if __name__ == "__main__":
    main()
