# COMET Documentation

## Overview

COMET (Computational Morphology Embedding Tool) is a complete pipeline for:
1. Processing whole slide images (WSIs)
2. Extracting morphological embeddings using foundation models
3. Performing similarity search across slide databases

## Architecture

```
comet/
├── __init__.py           # Package exports
├── config.py             # Configuration management
├── pipeline.py           # Main pipeline orchestrator
├── cli.py                # Command-line interface
├── utils.py              # Utility functions
├── visualization.py      # Visualization tools
├── preprocessing/
│   ├── __init__.py
│   └── mosaic.py         # Mosaic generation
├── embeddings/
│   ├── __init__.py
│   ├── patch_embeddings.py   # Patch feature extraction
│   └── slide_embeddings.py   # Slide-level aggregation
└── search/
    ├── __init__.py
    ├── slide_search.py   # Slide-level search
    └── roi_search.py     # ROI-level search
```

## Pipeline Steps

### 1. Mosaic Generation
- Input: Whole slide images (.svs files)
- Process: 
  - Detect tissue regions using thumbnail analysis
  - Tile the slide into patches
  - Filter patches by tissue coverage
  - Cluster patches by color/staining
  - Select representative patches
- Output: Mosaic metadata (patch locations, cluster info)

### 2. Patch Embedding Extraction
- Input: Mosaic metadata, slide files
- Process:
  - Load Virchow2 foundation model
  - Extract patches at specified magnification
  - Generate 2560-dim embeddings per patch
- Output: Patch embedding arrays

### 3. Slide Embedding Generation
- Input: Patch embeddings
- Process:
  - Compute percentiles across patches (5th, 15th, ..., 95th)
  - Create 10 × 1280 slide signature
- Output: Slide-level embeddings

### 4. Similarity Search
- Input: Slide embeddings, reference database
- Process:
  - Compute L1 distances between embeddings
  - Rank by similarity
  - Compute statistical significance
- Output: Ranked list of similar slides

## API Reference

### COMETConfig
Configuration dataclass with all pipeline parameters.

### COMETPipeline
Main pipeline class with methods:
- `run()`: Run complete pipeline
- `generate_mosaics()`: Step 1
- `extract_patch_embeddings()`: Step 2
- `generate_slide_embeddings()`: Step 3
- `search()`: Step 4
- `process_single_slide()`: Process one slide

### Search Engines
- `SlideSearchEngine`: Slide-level similarity search
- `ROISearchEngine`: Patch-level similarity search

## Performance Tips

1. **GPU Memory**: Reduce `batch_size` if encountering OOM
2. **CPU Parallelization**: Increase `n_cpus` for faster mosaic generation
3. **Caching**: Intermediate results are cached; use `force_recompute=True` to regenerate
