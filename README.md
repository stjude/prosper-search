<div align="center">

# 🔬 PePath

### Pediatric Pathology Slide Search Tool

**A Production-Grade Pipeline for Whole Slide Image Embedding and Morphological Similarity Search**

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

**PePath** is a computational tool for extracting morphological embeddings from whole slide images (WSIs) and performing similarity-based search across pathology databases. Designed for **pediatric pathology** applications, PePath enables:

- 🔍 **Slide-level search**: Find morphologically similar whole slide images
- 🎯 **ROI-level search**: Identify similar tissue regions across slides
- 🧠 **Multiple foundation models**: Virchow2, UNI2, UNI, GigaPath
- 📊 **Multiple embedding methods**: Percentile, Bag-of-Barcodes (BoB), Optimal Transport (OT)
- 🎨 **OT Visualization**: Heatmap overlays showing patch-level similarity contributions
- ⚡ **Efficient processing**: Optimized for large-scale datasets with GPU acceleration

## 🏗️ Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌────────────┐
│  Whole Slide    │────▶│  Mosaic          │────▶│  Patch          │────▶│  Slide     │
│  Images (.svs)  │     │  Generation      │     │  Embeddings     │     │  Embeddings│
└─────────────────┘     └──────────────────┘     │  (Virchow2/     │     │ (Percentile│
                                                 │   UNI2/GigaPath)│     │  /BoB/OT)  │
                                                 └─────────────────┘     └────────────┘
                                                                                │
                              ┌──────────────────────────────────────────────────┘
                              ▼
                        ┌────────────┐     ┌─────────────────┐
                        │  Search    │────▶│  Results &      │
                        │  Engine    │     │  Visualization  │
                        └────────────┘     └─────────────────┘
```

## 🧠 Models & Methods

### Patch Embedding Models

| Model | Embedding Dim | Description |
|-------|---------------|-------------|
| **Virchow2** | 2560 | Paige AI pathology foundation model (default) |
| **UNI2** | 1536 | Mahmood Lab universal pathology model |
| **UNI** | 1024 | Original UNI model |
| **GigaPath** | 1536 | Providence large-scale pathology model |

### Slide Embedding Methods

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **Percentile** | ⚡⚡⚡ | ★★★ | General purpose, fast search |
| **BoB** | ⚡⚡⚡⚡ | ★★ | Very large databases, binary hashing |
| **OT** | ⚡ | ★★★★★ | Highest accuracy, visualization |

## 🚀 Installation

### Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/pepath.git
cd pepath

# Run the installation script
chmod +x scripts/install.sh
./scripts/install.sh --conda
```

### Option 1: Conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml
conda activate pepath

# Install PePath
pip install -e .
```

### Option 2: Docker

```bash
# Build image
docker build -t pepath:latest .

# Run
docker run --gpus all -v /path/to/slides:/data -v /path/to/output:/output \
    pepath:latest python -m pepath.cli --slide-path /data --output-dir /output
```

### Option 3: Pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install OpenSlide (system dependency)
# Ubuntu: sudo apt-get install openslide-tools python3-openslide
# macOS: brew install openslide

# Install PePath
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
> - [Virchow2](https://huggingface.co/paige-ai/Virchow2)
> - [UNI2](https://huggingface.co/MahmoodLab/UNI2-h)
> - [GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath)

## ⚡ Quick Start

### Python API

```python
from pepath import PePathPipeline, PePathConfig

# Configure the pipeline
config = PePathConfig(
    slide_path="/path/to/your/slides",
    output_dir="./output",
    device="cuda",
    
    # Model selection
    patch_model="virchow2",           # Options: virchow2, uni2, uni, gigapath
    slide_embedding_method="percentile",  # Options: percentile, bob, ot
    
    top_k=10
)

# Initialize and run
pipeline = PePathPipeline(config)
results = pipeline.run()

# View results
df = results.to_dataframe()
print(df)

# List available models and methods
print(PePathPipeline.list_available_models())
```

### Command Line

```bash
# Run with default settings
python -m pepath.cli --slide-path /path/to/slides --output-dir ./output

# Use UNI2 model with OT embedding and visualization
python -m pepath.cli --slide-path /path/to/slides \
    --patch-model uni2 \
    --slide-method ot \
    --generate-visualizations

# Quick smoke test
python -m pepath.cli --slide-path /path/to/slides --smoketest

# List available models
python -m pepath.cli --list-models
```

## 📓 Demo

We provide a comprehensive demo notebook that you can run on a small example:

```bash
# Start Jupyter
jupyter notebook examples/demo.ipynb
```

The demo includes:
- ✅ Environment verification
- ✅ Processing 3 sample slides
- ✅ Generating embeddings
- ✅ Performing similarity search
- ✅ Visualizing results

See [examples/demo.ipynb](examples/demo.ipynb) for the full walkthrough.

## 📁 Output Structure

```
output/
├── mosaic/                     # Tissue patches
├── patch_embeddings/           # Patch-level features
├── slide_embeddings/           # Aggregated slide features
├── search_results/             # Search results
│   └── ot_similarities/        # OT pairwise distances
└── visualizations/             # OT heatmap overlays
```

## 📖 Configuration

```yaml
# config.yaml
slide_path: "/path/to/slides"
output_dir: "./output"

# Model selection
patch_model: "virchow2"           # virchow2, uni2, uni, gigapath
slide_embedding_method: "percentile"  # percentile, bob, ot

# Processing
base_patchsize: 512
device: "cuda"
batch_size: 64
top_k: 10

# Visualization (for OT)
generate_visualizations: false
```

## 🎯 Use Cases

- **Diagnostic Support**: Find historically similar cases for differential diagnosis
- **Cohort Discovery**: Identify morphologically similar cases for research
- **Quality Control**: Detect potential mislabeling
- **Education**: Build teaching sets of related cases
- **Research**: Compare embedding models for your specific application

## 📚 Citation

If you use PePath in your research, please cite:

```bibtex
@article{chen2024pepath,
  title={PePath: Pediatric Pathology Slide Search Tool for Whole Slide Image Analysis},
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
