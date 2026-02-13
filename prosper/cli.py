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
import logging
import sys
import os
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prosper import ProsperPipeline, ProsperConfig
from prosper.pipeline import ProsperResults
from prosper.embeddings import list_patch_embedding_models, list_slide_embedding_methods

# ── Logging setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger("prosper")


def _setup_logging(output_dir: str, quiet: bool = False):
    """Configure logging to both console and a timestamped log file.

    Log files are written to ``<output_dir>/logs/``.
    """
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"prosper_run_{timestamp}.log")

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler — always verbose
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — flush after every message so progress is visible
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING if quiet else logging.INFO)
    ch.setFormatter(fmt)
    ch.flush = sys.stdout.flush  # ensure immediate output

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return log_path


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


def _step(msg):
    """Log start of a pipeline step."""
    logger.info(f">> {msg} ...")


def _done(msg, elapsed=None):
    """Log completion of a pipeline step."""
    suffix = f" ({elapsed:.1f}s)" if elapsed is not None else ""
    logger.info(f"   {msg}{suffix}")


def _has_cached(path, label):
    """Check if a cached result exists. Log and return True/False."""
    if os.path.exists(path):
        logger.info(f"   Found cached {label}: {path} -- skipping")
        return True
    return False


def main():
    args = parse_args()

    # Handle --list-models (no logging needed)
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

    # ── Build configuration ───────────────────────────────────────────────
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

    # ── Set up logging ────────────────────────────────────────────────────
    log_path = _setup_logging(config.output_dir, quiet=args.quiet)
    t_total = time.time()
    force = config.force_recompute

    logger.info("=" * 60)
    logger.info("PROSPER Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Slide path         : {config.slide_path}")
    logger.info(f"  Output dir         : {config.output_dir}")
    logger.info(f"  Patch model        : {config.patch_model}")
    logger.info(f"  Slide embedding    : {config.slide_embedding_method}")
    logger.info(f"  Distance metric    : {config.distance_metric}")
    if args.reference_db:
        logger.info(f"  Reference DB       : {args.reference_db}")
    if args.metadata:
        logger.info(f"  Metadata           : {args.metadata}")
    logger.info(f"  Force recompute    : {force}")
    logger.info(f"  Steps              : {args.step}"
                + (" + bayesian" if args.proba else "")
                + (" + llm" if args.llm else ""))
    logger.info("=" * 60)

    # ── Initialize pipeline ───────────────────────────────────────────────
    _step("Initializing pipeline")
    t0 = time.time()
    pipeline = ProsperPipeline(config)
    _done("Pipeline initialized", time.time() - t0)

    # ── Single slide mode ─────────────────────────────────────────────────
    if args.single_slide:
        if not args.reference_db:
            print("Error: --reference-db is required for single slide mode")
            sys.exit(1)
        _step("Processing single slide")
        t0 = time.time()
        results = pipeline.process_single_slide(
            slide_path=args.single_slide,
            reference_db_path=args.reference_db,
            search_type=args.search_type,
        )
        _done("Single slide processed", time.time() - t0)
        df = results.to_dataframe(args.search_type)
        print("\nSearch Results:")
        print(df.to_string())
        return

    # ══════════════════════════════════════════════════════════════════════
    # Full pipeline — each step checks for cached results
    # ══════════════════════════════════════════════════════════════════════

    run_mosaic = args.step in ("all", "mosaic")
    run_patch  = args.step in ("all", "patch-embedding")
    run_slide  = args.step in ("all", "slide-embedding")
    run_search = args.step in ("all", "search")
    run_bayes  = args.step in ("all", "bayesian") or args.proba
    run_llm    = args.step in ("all", "llm") or args.llm

    # ── Step 1: Mosaic generation ─────────────────────────────────────────
    if run_mosaic:
        mosaic_file = config.mosaic_file
        if not force and _has_cached(mosaic_file, "mosaics"):
            from prosper.preprocessing import MosaicGenerator
            pipeline.mosaics = MosaicGenerator.load_mosaics(mosaic_file)
            _done(f"Loaded {len(pipeline.mosaics)} cached mosaics")
        else:
            _step("Generating mosaics from whole slide images")
            t0 = time.time()
            pipeline.generate_mosaics()
            n = len(pipeline.mosaics)
            _done(f"Mosaic generation complete — {n} slides", time.time() - t0)
        if args.step == "mosaic":
            _print_footer(t_total, log_path, config, args)
            return

    # ── Step 2: Patch embedding extraction ────────────────────────────────
    if run_patch:
        patch_tmp = os.path.join(config.patch_embeddings_dir, "tmp")
        # Check if any cached patch embeddings already exist
        cached_patches = [f for f in os.listdir(patch_tmp) if f.endswith('.pkl')] if os.path.isdir(patch_tmp) else []
        if not force and cached_patches:
            logger.info(f"   Found {len(cached_patches)} cached patch embeddings in {patch_tmp}")
            # Load them so downstream steps can use them
            if not pipeline.mosaics:
                from prosper.preprocessing import MosaicGenerator
                pipeline.mosaics = MosaicGenerator.load_mosaics(config.mosaic_file)
            # The patch extractor's process_slides will skip already-cached files
            _step(f"Extracting patch embeddings (model: {config.patch_model}) — {len(cached_patches)} already cached")
            t0 = time.time()
            pipeline.extract_patch_embeddings()
            _done(f"Patch embedding extraction complete — {len(pipeline.patch_embeddings)} slides", time.time() - t0)
        else:
            _step(f"Extracting patch embeddings (model: {config.patch_model})")
            t0 = time.time()
            pipeline.extract_patch_embeddings()
            _done(f"Patch embedding extraction complete — {len(pipeline.patch_embeddings)} slides", time.time() - t0)
        if args.step == "patch-embedding":
            _print_footer(t_total, log_path, config, args)
            return

    # ── Step 3: Slide embedding computation ───────────────────────────────
    if run_slide:
        slide_emb_dir = config.slide_embeddings_dir
        cached_slide = [f for f in os.listdir(slide_emb_dir) if f.endswith('.pkl')] if os.path.isdir(slide_emb_dir) else []
        if not force and cached_slide:
            logger.info(f"   Found {len(cached_slide)} cached slide embeddings in {slide_emb_dir}")
            _step(f"Computing slide embeddings (method: {config.slide_embedding_method}) — {len(cached_slide)} already cached")
        else:
            _step(f"Computing slide embeddings (method: {config.slide_embedding_method})")
        t0 = time.time()
        pipeline.generate_slide_embeddings()
        _done(f"Slide embedding computation complete — {len(pipeline.slide_embeddings)} slides", time.time() - t0)
        if args.step == "slide-embedding":
            _print_footer(t_total, log_path, config, args)
            return

    # ── Step 4: Slide search ──────────────────────────────────────────────
    if run_search:
        search_pkl = os.path.join(config.search_results_dir, "slide_search_results.pkl")
        if not force and _has_cached(search_pkl, "search results"):
            import pickle as _pkl
            with open(search_pkl, "rb") as f:
                cached_sr = _pkl.load(f)
            results = ProsperResults(
                query_slide_ids=list(cached_sr.keys()),
                slide_search_results=cached_sr,
            )
            _done(f"Loaded cached search results for {len(cached_sr)} slides")
        else:
            _step("Searching for similar slides")
            t0 = time.time()
            results = pipeline.search(search_type=args.search_type)
            n_queries = len(results.slide_search_results) if results.slide_search_results else 0
            _done(f"Slide search complete — {n_queries} query slides", time.time() - t0)
        if args.step == "search":
            df = results.to_dataframe(args.search_type)
            print("\nSearch Results:")
            print(df.head(20).to_string())
            _print_footer(t_total, log_path, config, args)
            return

    # ── Step 5: Bayesian disease probabilities ────────────────────────────
    if run_bayes:
        bayes_csv = os.path.join(config.bayesian_dir, "disease_probabilities.csv")
        if not force and _has_cached(bayes_csv, "Bayesian probabilities"):
            import pandas as _pd
            _df = _pd.read_csv(bayes_csv, index_col=0)
            _done(f"Loaded cached Bayesian probabilities: {_df.shape}")
        else:
            _step("Computing Bayesian disease probabilities")
            t0 = time.time()
            try:
                disease_probs = pipeline.compute_disease_probabilities(
                    metadata_path=args.metadata,
                    n_neighbors=args.bayesian_neighbors,
                    temperature=args.bayesian_temperature,
                )
                if 'results' in dir() and hasattr(results, 'disease_probabilities'):
                    results.disease_probabilities = disease_probs
                _done(f"Bayesian probabilities computed for {len(disease_probs)} slides", time.time() - t0)
            except Exception as e:
                logger.warning(f"Bayesian step failed: {e}")
        if args.step == "bayesian":
            _print_footer(t_total, log_path, config, args)
            return

    # ── Step 6: LLM summarization ─────────────────────────────────────────
    if run_llm:
        llm_dir = config.llm_dir
        cached_summaries = [f for f in os.listdir(llm_dir) if f.startswith("summary_")] if os.path.isdir(llm_dir) else []
        if not force and cached_summaries:
            logger.info(f"   Found {len(cached_summaries)} cached LLM summaries in {llm_dir} -- skipping")
        else:
            _step("Generating LLM summaries")
            t0 = time.time()
            try:
                sr = results.slide_search_results if 'results' in dir() else None
                dp = results.disease_probabilities if 'results' in dir() and hasattr(results, 'disease_probabilities') else None
                summaries = pipeline.generate_llm_summary(
                    disease_probs=dp,
                    search_results=sr,
                    metadata_path=args.metadata,
                )
                if 'results' in dir() and summaries:
                    results.llm_summaries = summaries
                n = len(summaries) if summaries else 0
                _done(f"LLM summaries generated for {n} slides", time.time() - t0)
            except Exception as e:
                logger.warning(f"LLM step failed: {e}")
        if args.step == "llm":
            _print_footer(t_total, log_path, config, args)
            return

    _print_footer(t_total, log_path, config, args)


def _print_footer(t_total, log_path, config, args):
    elapsed = time.time() - t_total
    logger.info("=" * 60)
    logger.info(f"Pipeline complete  ({elapsed:.1f}s total)")
    logger.info(f"  Patch model      : {config.patch_model}")
    logger.info(f"  Slide embedding  : {config.slide_embedding_method}")
    if args.proba:
        logger.info(f"  Bayesian         : enabled")
    if args.llm:
        logger.info(f"  LLM              : enabled")
    if config.generate_visualizations:
        logger.info(f"  Visualizations   : {config.visualizations_dir}")
    logger.info(f"  Log saved to     : {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
