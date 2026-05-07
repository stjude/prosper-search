# PROSPER Data Directory

This directory contains the data files required to run the PROSPER pipeline and demos.

## Data Files Overview

| File/Folder | Size | Required For | Included |
|-------------|------|--------------|----------|
| `final_scan_id_disease_map_rich_data.csv` | ~650 KB | Metadata, disease labels, clinical data | ✅ Yes |
| `slide_dimensions.csv` | ~128 KB | Heatmap coordinate mapping | ✅ Yes |
| `patch_embeddings/` | ~10 GB | ROI search, heatmaps | ❌ Download |
| `final_slides_sampler_matrix.pkl` | ~286 MB | Slide-level search | ❌ Download |
| `mosaic_metadata.pkl` | ~1.2 GB | Heatmap visualization | ❌ Download |
| `thumbnails/` | ~415 MB | Heatmap overlays | ❌ Download |
| `4261.svs` | ~2.3 GB | Demo (sample slide) | ❌ Download |

**Total download size: ~14 GB**

## Downloading Data

### Option 1: Download Script (Recommended)

```bash
# Download all data files
python scripts/download_data.py --all

# Download only essential files (for basic demo)
python scripts/download_data.py --essential

# Download specific components
python scripts/download_data.py --patch-embeddings
python scripts/download_data.py --thumbnails
python scripts/download_data.py --sample-slide
```

### Option 2: Git LFS (For Repository Clones)

Large files are tracked with Git LFS. After cloning:

```bash
# Install Git LFS if not already installed
git lfs install

# Pull large files
git lfs pull
```

### Option 3: Manual Download (Internal Mirror)

If your organization provides an internal mirror, set:

```bash
export PROSPER_INTERNAL_DATA_SOURCE=/path/to/internal/comet_portal_files
```

Copy files to this `data/` directory:

```bash
# From your internal mirror
cp -r "$PROSPER_INTERNAL_DATA_SOURCE/patch_embeddings" ./data/
cp "$PROSPER_INTERNAL_DATA_SOURCE/mosaic_metadata.pkl" ./data/
# ... etc
```

## File Descriptions

### `patch_embeddings/` (Required for ROI search)
Pre-computed Virchow2 patch embeddings for 2,921 slides in `.npz` format.
Each file contains:
- `embeddings`: (N, 2560) array of patch embeddings
- `coords`: (N, 2) array of patch coordinates

### `final_slides_sampler_matrix.pkl` (Required for slide search)
Pre-computed SAMPLER slide embeddings for all reference slides.
DataFrame with slide IDs as index and 12,800-dim embeddings.

### `mosaic_metadata.pkl` (Required for heatmap visualization)
Mosaic cluster information including `cluster_locs` for each slide.
Used to generate full-tissue heatmaps (not just mosaic tile locations).

### `thumbnails/` (Required for heatmap overlays)
JPEG thumbnails of all slides for visualization overlays.

### `slide_dimensions.csv` (Included)
Original slide dimensions (width, height) for coordinate mapping.

### `final_scan_id_disease_map_rich_data.csv` (Included)
Metadata for all slides including disease labels (`tumor_group`), demographics
(`Age.at.Collection`, `Sex`, `Site`), treatment status, diagnosis, and more.
Used for Bayesian disease probability estimation, LLM summarization, and
evaluating search accuracy.

### `4261.svs` (Sample slide for demo)
A sample whole slide image for running the demo notebook.

## Generating Your Own Data

If you want to process your own slides:

```python
from prosper import ProsperPipeline, ProsperConfig

config = ProsperConfig(
    slide_path="/path/to/your/slides",
    output_dir="./output",
    patch_model="virchow2",
    slide_embedding_method="percentile"
)

pipeline = ProsperPipeline(config)
results = pipeline.run()
```

The pipeline will generate all necessary embedding files in `output/`.
