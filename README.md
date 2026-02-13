<div align="center">

# 🔬 PROSPER

### $\color{#E74C3C}{\textbf{P}}$ediatric $\color{#E67E22}{\textbf{R}}$etrieval for $\color{#F1C40F}{\textbf{O}}$ncology $\color{#2ECC71}{\textbf{S}}$lides, $\color{#3498DB}{\textbf{P}}$athology $\color{#9B59B6}{\textbf{E}}$vidence, and $\color{#1ABC9C}{\textbf{R}}$eporting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2024.XXXXX-b31b1b.svg)](https://arxiv.org/)

[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Models & Methods](#-models--methods) •
[Demo](#-demo) •
[Citation](#-citation)

</div>

---

## 📋 Overview

<div align="center">
<img src="data/fig/overview.png" alt="PROSPER Overview" width="100%">
</div>

<br>

**PROSPER** is a computational tool for extracting morphological embeddings from whole slide images (WSIs) and performing similarity-based search across pathology databases. Designed for **pediatric pathology** applications, PROSPER enables:

- 🔍 **Slide-level search**: Find morphologically similar whole slide images
- 🎯 **ROI-level search**: Identify similar tissue regions across slides
- 🧠 **Multiple foundation models**: Virchow2, UNI2, GigaPath
- 📊 **Multiple slide similarity formulations**: SAMPLER (percentile), Bag-of-Barcodes (BoB), Optimal Transport (OT)
- 🎨 **Interpretable visualization**: Heatmap overlays showing patch-level similarity contributions
- 📈 **Bayesian disease probabilities**: Exponential-weighted nearest-neighbor probability estimation
- 🤖 **LLM summarization**: Grounded, citation-backed narrative reports via GPT
- ⚡ **Efficient processing**: Optimized for large-scale datasets with GPU acceleration

## 🏗️ Pipeline Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐      ┌──────────────┐
│  Whole Slide    │─────▶│     Mosaic       │─────▶│     Patch       │─────▶│    Slide     │
│  Images (.svs)  │      │   Generation     │      │   Embeddings    │      │  Embeddings  │
└─────────────────┘      └──────────────────┘      │   (Virchow2/    │      │  (SAMPLER/   │
                                                   │  UNI2/GigaPath) │      │   BoB/OT)    │
                                                   └─────────────────┘      └──────────────┘
                                                                                    │
                                    ┌───────────────────────────────────────────────┘
                                    │
                                    ▼
                              ┌────────────┐      ┌─────────────────┐
                              │   Search   │─────▶│   Results &     │
                              │   Engine   │      │  Visualization  │
                              └────────────┘      └─────────────────┘
                                    │                      │
                                    ▼                      ▼
                              ┌────────────┐      ┌─────────────────┐
                              │  Bayesian  │─────▶│     LLM         │
                              │  Disease   │      │  Summarization  │
                              │  Probs     │      │  (Grounded)     │
                              └────────────┘      └─────────────────┘
```

## 🧠 Models & Methods

### Patch Embedding Models

| Model | Embedding Dim | Features Used | Description |
|-------|---------------|---------------|-------------|
| **Virchow2** | 2560 | First 1280 (CLS token) | Paige AI pathology foundation model (default) |
| **UNI2** | 1536 | 1536 | Mahmood Lab universal pathology model v2 |
| **GigaPath** | 1536 | 1536 | Providence large-scale pathology model |

### Slide Similarity Formulations

| Method | Speed | Accuracy | Interpretability | Description |
|--------|:-----:|:--------:|:----------------:|-------------|
| **SAMPLER** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | Percentile-based aggregation (5th-95th) |
| **BoB** | ★★★☆☆ | ★★★★☆ | ★★★★☆ | Bag-of-Barcodes binary hashing |
| **OT** | ★★☆☆☆ | ★★★★★ | ★★★★★ | Optimal Transport with visualization |

### Bayesian Disease Probability Estimation

After retrieval, PROSPER converts raw distances into calibrated disease probabilities using **exponential-weighted nearest neighbors**:

- Given a distance matrix and disease labels, the module selects the *k* closest reference slides
- Distances are mapped to weights via an exponential kernel controlled by a **temperature** parameter
- The resulting distribution answers: *"Given what the model retrieved, how likely is each disease?"*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neighbors` | 20 | Number of nearest neighbors to aggregate |
| `temperature` | 20.0 | Smoothing factor — larger = softer probabilities |

### Interpretability Visualization

PROSPER provides two complementary visualization modes that make search results **explainable at the patch level**:

| Visualization | Input | Output | Description |
|---------------|-------|--------|-------------|
| **OT Heatmap Overlay** | OT transport matrix + WSI | Whole-slide heatmap | Colors every patch by its transport-weighted similarity score, highlighting which regions drove the match |
| **ROI Search Heatmap** | Patch embeddings + query ROI | Thumbnail heatmap overlay | Computes cosine similarity of every tile to a query patch and overlays the result on a slide thumbnail |

**OT Heatmap Overlay** — After solving the optimal transport problem between two slides, the transport matrix \(T\) and cost matrix \(M\) are combined to produce per-patch weighted distance scores. These scores are normalized to \([0, 1]\) and rendered as a colormap overlay on the WSI, so clinicians can see *exactly* which tissue regions are most (or least) similar. An optional clustering step groups co-transported patches into discrete regions.

**ROI Search Heatmap** — Given a query patch (identified by slide ID and patch index), PROSPER computes cosine similarities between the query embedding and all patch embeddings of a hit slide. The resulting similarity map is overlaid on a slide thumbnail using accurate coordinate mapping from `slide_dimensions.csv`, producing a publication-ready figure without requiring the original `.svs` file.

```python
# OT Visualization
from prosper.visualization import OTVisualizer

visualizer = OTVisualizer(patch_size=512, slide_level=2)
results = visualizer.visualize_similarity(
    slide_a_path, slide_b_path,
    ot_result, mosaic_a, mosaic_b
)

# ROI Search Heatmap
from prosper.visualization import generate_roi_search_heatmap

output_path = generate_roi_search_heatmap(
    embeddings_dir="data/patch_embeddings",
    query_slide_id="4261",
    query_patch_idx=860,
    hit_slide_id="4264",
    output_dir="output/heatmaps"
)
```

### LLM Summarization

PROSPER includes a **grounded LLM summarization** module that generates structured narratives from search results:

1. **Payload construction** — Assembles a structured JSON containing the query slide's demographics (age, sex, site, diagnosis, treatment, discrepant status) and top-*k* neighbor metadata with similarity scores and Bayesian probabilities
2. **Grounded generation** — Sends the payload to a GPT model with a structured prompt requiring inline `[case_id]` citations and a `claims` array
3. **Post-hoc verification** — Every claim is checked against the source data: fabricated case IDs, unverified numbers, and mismatched field values are flagged

The output is a ≤200-word narrative that includes treatment status, concordance flags, dominant diagnosis with vote counts, and Bayesian probability estimates — all with traceable citations.

> **Requires:** Azure OpenAI or OpenAI API credentials set as environment variables.

## 🚀 Installation

### Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/stjude/prosper-search.git
cd prosper-search

# Run the installation script
chmod +x scripts/install.sh
./scripts/install.sh --conda
```

### Option 1: Conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml
conda activate prosper

# Install PROSPER
pip install -e .
```

### Option 2: Docker

```bash
# Build image
docker build -t prosper:latest .

# Run
docker run --gpus all -v /path/to/slides:/data -v /path/to/output:/output \
    prosper:latest python -m prosper.cli --slide-path /data --output-dir /output
```

### Option 3: Pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install OpenSlide (system dependency)
# Ubuntu: sudo apt-get install openslide-tools python3-openslide
# macOS: brew install openslide

# Install PROSPER
pip install -r requirements.txt
pip install -e .
```

### Setup Foundation Models

After installation, set up the foundation models:

```bash
# Login to Hugging Face
huggingface-cli login

# Download and cache models
python scripts/setup_models.py --model virchow2

# Or download all models
python scripts/setup_models.py --all
```

> **Note**: You need to accept model licenses on Hugging Face:
> - [Virchow2](https://huggingface.co/paige-ai/Virchow2) - Uses first 1280 features (CLS token)
> - [UNI2](https://huggingface.co/MahmoodLab/UNI2-h)
> - [GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath)

## 📦 Data Download

The demo and pre-computed search require data files (~14 GB total). Small metadata files are included; large files must be downloaded:

```bash
# List available files and status
python scripts/download_data.py --list

# Download all data (~14 GB)
python scripts/download_data.py --all

# Download essential files only (~10 GB) - for basic search
python scripts/download_data.py --essential

# Download specific components
python scripts/download_data.py --patch-embeddings  # For ROI search
python scripts/download_data.py --sample-slide      # For demo
```

**Data sources:**
- **St. Jude internal**: Auto-detected from `REDACTED_INTERNAL_PATH/DNB_Core_Public/`
- **Git LFS**: Large files tracked with Git LFS (run `git lfs pull` after cloning)

See [data/README.md](data/README.md) for detailed file descriptions.

## ⚡ Quick Start

### Python API

```python
from prosper import ProsperPipeline, ProsperConfig

config = ProsperConfig(
    slide_path="/path/to/slides",
    output_dir="./output",
    patch_model="virchow2",                # virchow2 | uni2 | gigapath
    slide_embedding_method="percentile",   # percentile | bob | ot
)
pipeline = ProsperPipeline(config)
results = pipeline.run()
```

### Command Line

```bash
# Default pipeline (Virchow2 + percentile)
python -m prosper.cli --slide-path /path/to/slides --output-dir ./output

# Switch models and methods
python -m prosper.cli --slide-path /path/to/slides \
    --patch-embedding uni2 --slide-embedding ot

# Full pipeline with Bayesian probabilities + LLM summarization
python -m prosper.cli --slide-path /path/to/slides --proba --llm

# ROI search for a specific region
python -m prosper.cli --slide-path /path/to/slides --roi-search --coords 1024,2048

# Use a YAML config file (CLI flags override config values)
python -m prosper.cli --config configs/config.example.yaml
```

<details>
<summary><b>Full CLI Reference</b></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--patch-embedding` | `virchow2` | Patch embedding model: `virchow2`, `uni2`, `uni`, `gigapath` |
| `--slide-embedding` | `percentile` | Slide embedding method: `percentile`, `bob`, `ot` |
| `--roi-search` | off | Enable ROI (patch-level) similarity search |
| `--coords X,Y` | — | Upper-left corner of 512×512 patch (implies `--roi-search`) |
| `--proba` | off | Compute Bayesian disease probability estimation |
| `--llm` | off | Generate LLM narrative summary (requires API key) |
| `--top-k` | `10` | Number of similar slides to return |
| `--distance-metric` | `l1` | Distance metric: `l1`, `l2`, `cosine` |
| `--device` | `cuda` | Compute device: `cuda`, `cpu` |
| `--batch-size` | `64` | Batch size for feature extraction |
| `--smoketest` | off | Quick test with limited slides |
| `--force-recompute` | off | Recompute all steps from scratch |
| `--config` | — | Path to YAML configuration file |
| `--single-slide` | — | Process one slide against a pre-built reference DB |
| `--step` | — | Run a single pipeline step: `mosaic`, `patch-embedding`, `slide-embedding`, `search`, `bayesian`, `llm` |

</details>

## 📓 Demo

We provide a comprehensive demo notebook that you can run on a small example:

```bash
# Start Jupyter
jupyter notebook examples/demo.ipynb
```

The demo includes:
- ✅ Environment verification
- ✅ Processing a sample slide end-to-end
- ✅ Generating patch and slide embeddings
- ✅ Performing slide-level and ROI-level similarity search
- ✅ OT (Optimal Transport) slide comparison with heatmaps
- ✅ Bayesian disease probability estimation with visualization
- ✅ LLM summarization with grounded citations

See [examples/demo.ipynb](examples/demo.ipynb) for the full walkthrough.

## 📁 Output Structure

```
output/
├── mosaic/                     # Tissue patches
├── patch_embeddings/
│   └── virchow2/              # Patch-level features (per model)
├── slide_embeddings/
│   └── percentile/            # Aggregated slide features (per method)
├── search_results/             # Search results
│   └── ot_similarities/        # OT pairwise distances
├── visualizations/             # OT heatmap overlays
├── bayesian/                   # Bayesian disease probabilities
│   ├── disease_probs_{id}.json # Per-slide probability distributions
│   └── disease_probs_{id}.csv  # Probability matrix
└── llm/                        # LLM summarization outputs
    ├── slide_{id}.json         # Structured payload with metadata + neighbors
    └── summary_{id}.txt        # Grounded narrative summary
```

## 📖 Configuration

```yaml
# config.yaml
slide_path: "/path/to/slides"
output_dir: "./output"

# Model selection
patch_model: "virchow2"           # virchow2, uni2, gigapath
slide_embedding_method: "percentile"  # percentile, bob, ot

# Processing
base_patchsize: 512
device: "cuda"
batch_size: 64
top_k: 10

# Bayesian disease probabilities
bayesian_n_neighbors: 20     # Number of nearest neighbors
bayesian_temperature: 20.0   # Exponential weighting temperature

# Visualization (for OT)
generate_visualizations: false
```

### LLM Environment Variables

```bash
# Azure OpenAI (recommended)
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_DEPLOYMENT="gpt-5"

# Or standard OpenAI
export OPENAI_API_KEY="your-key"
```

## 🎯 Use Cases

- **Diagnostic Support**: Find historically similar cases for differential diagnosis
- **Cohort Discovery**: Identify morphologically similar cases for research
- **Quality Control**: Detect potential mislabeling
- **Education**: Build teaching sets of related cases
- **Research**: Compare embedding models for your specific application

## 📚 Citation

If you use PROSPER in your research, please cite:

```bibtex
@article{chen2024prosper,
  title={PROSPER: Pediatric Retrieval for Oncology Slides, Pathology Evidence, and Reporting},
  author={Chen, Haoran and others},
  journal={arXiv preprint arXiv:2024.XXXXX},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Virchow2](https://huggingface.co/paige-ai/Virchow2) by Paige AI
- [UNI2](https://huggingface.co/MahmoodLab/UNI2-h) by Mahmood Lab
- [GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) by Providence
- [OpenSlide](https://openslide.org/)
- [POT](https://pythonot.github.io/)

---

<div align="center">

Made with ❤️ for the pediatric pathology community

</div>
