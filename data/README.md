# PROSPER Data Directory

---

> **Data release and access — read first**  
> **Distribution of these datasets is not automatic with the public software repository.** Release, hosting, and sharing are subject to **St. Jude internal data-governance and approval processes**. External collaborators or other requesters may be able to obtain approved materials **only through those official channels** (e.g. data-use agreements, institutional request workflows), not by cloning this repo alone.  
> **Questions or formal access requests:** use your internal research-data / compliance route, or contact the maintainers listed in the root [`README.md`](../README.md).

---

This directory contains the data files required to run the PROSPER pipeline and demos.

## Data Files Overview

| File/Folder | Size | Required For |
|-------------|------|--------------|
| `final_scan_id_disease_map_rich_data.csv` | ~650 KB | Metadata, disease labels, clinical data |
| `slide_dimensions.csv` | ~128 KB | Heatmap coordinate mapping |
| `patch_embeddings/` | ~10 GB | ROI search, heatmaps |
| `final_slides_sampler_matrix.pkl` | ~286 MB | Slide-level search |
| `mosaic_metadata.pkl` | ~1.2 GB | Heatmap visualization |
| `thumbnails/` | ~415 MB | Heatmap overlays |
| `4261.svs` | ~2.3 GB | Demo (sample slide) |

The two small CSVs are **often** included when your institution ships an **approved** bundle; all larger rows follow the same rule as the banner above (**internal release / approved request** — not a public self-serve download).

**Approximate total size when you obtain the full bundle through an approved channel: ~14 GB**

## Obtaining data (after authorized access)

> **The steps below do not grant access by themselves.** They only apply **after** your team has gone through **internal data-release / request workflows** (or another approved path) and you have a supported source: e.g. internal mirror, org-managed Git LFS, or an approved data package. There is **no** implied public anonymous bulk download.

### Option 1: Download script (when your environment provides a source)

Use when `scripts/download_data.py` can reach an **authorized** internal mirror or other approved source configured for your site.

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

### Option 2: Git LFS (when large objects are released with the repo)

Large files may be tracked with Git LFS **for clones that include an approved LFS release**. After cloning:

```bash
# Install Git LFS if not already installed
git lfs install

# Pull large files
git lfs pull
```

### Option 3: Manual copy from an internal mirror (authorized sites only)

If your organization provides an **approved** internal mirror, set:

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

### `slide_dimensions.csv` (metadata)
Original slide dimensions (width, height) for coordinate mapping.

### `final_scan_id_disease_map_rich_data.csv` (metadata)
Metadata for all slides including disease labels (`tumor_group`), demographics
(`Age.at.Collection`, `Sex`, `Site`), treatment status, diagnosis, and more.
Used for Bayesian disease probability estimation, LLM summarization, and
evaluating search accuracy.

### `4261.svs` (sample slide for demo)
A sample whole slide image for running the demo notebook, when provided through an approved release.

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
