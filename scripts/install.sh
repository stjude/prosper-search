#!/bin/bash
#
# PROSPER Installation Script
# 
# This script automates the installation of PROSPER and its dependencies.
# 
# Usage:
#   chmod +x scripts/install.sh
#   ./scripts/install.sh
#
# Options:
#   --conda     Use conda for installation (recommended)
#   --pip       Use pip only for installation
#   --docker    Build Docker image
#   --no-gpu    Install CPU-only version
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                                                            ║"
    echo "║   🔬 PROSPER - Pediatric Retrieval for Oncology Slides, Pathology Evidence, and Reporting        ║"
    echo "║                                                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Parse arguments
USE_CONDA=false
USE_PIP=false
USE_DOCKER=false
NO_GPU=false

for arg in "$@"; do
    case $arg in
        --conda)
            USE_CONDA=true
            ;;
        --pip)
            USE_PIP=true
            ;;
        --docker)
            USE_DOCKER=true
            ;;
        --no-gpu)
            NO_GPU=true
            ;;
        --help|-h)
            echo "Usage: ./install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --conda     Use conda for installation (recommended)"
            echo "  --pip       Use pip only for installation"
            echo "  --docker    Build Docker image"
            echo "  --no-gpu    Install CPU-only version"
            echo "  --help      Show this help message"
            exit 0
            ;;
    esac
done

# Default to conda if no option specified
if [ "$USE_CONDA" = false ] && [ "$USE_PIP" = false ] && [ "$USE_DOCKER" = false ]; then
    USE_CONDA=true
fi

print_header

# Check for Docker installation
if [ "$USE_DOCKER" = true ]; then
    print_step "Building Docker image..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first."
        exit 1
    fi
    
    docker build -t prosper:latest .
    print_success "Docker image built successfully!"
    echo ""
    echo "Usage:"
    echo "  docker run --gpus all -v /path/to/slides:/data -v /path/to/output:/output prosper:latest \\"
    echo "      python -m prosper.cli --slide-path /data --output-dir /output"
    exit 0
fi

# Check for conda
if [ "$USE_CONDA" = true ]; then
    print_step "Setting up with Conda..."
    
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found. Please install Miniconda or Anaconda first."
        echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    # Create conda environment
    print_step "Creating conda environment 'prosper'..."
    conda env create -f environment.yml || conda env update -f environment.yml
    
    print_success "Conda environment created!"
    echo ""
    echo "Activate with: conda activate prosper"
    
    # Activate and install package
    print_step "Installing PROSPER package..."
    eval "$(conda shell.bash hook)"
    conda activate prosper
    pip install -e .
    
    print_success "PROSPER installed successfully!"
fi

# Check for pip
if [ "$USE_PIP" = true ]; then
    print_step "Setting up with pip..."
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    print_step "Python version: $PYTHON_VERSION"
    
    # Create virtual environment
    print_step "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install OpenSlide (system package required)
    print_warning "OpenSlide requires system installation:"
    echo "  Ubuntu/Debian: sudo apt-get install openslide-tools python3-openslide"
    echo "  macOS: brew install openslide"
    echo "  Windows: Download from https://openslide.org/download/"
    echo ""
    
    # Install requirements
    print_step "Installing Python dependencies..."
    if [ "$NO_GPU" = true ]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    else
        pip install torch torchvision
    fi
    pip install -r requirements.txt
    
    # Install package
    print_step "Installing PROSPER..."
    pip install -e .
    
    print_success "PROSPER installed successfully!"
    echo ""
    echo "Activate with: source venv/bin/activate"
fi

# Hugging Face login instructions
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo ""
echo "1. Accept model licenses on Hugging Face:"
echo "   • Virchow2: https://huggingface.co/paige-ai/Virchow2"
echo "   • UNI2: https://huggingface.co/MahmoodLab/UNI2-h"
echo "   • GigaPath: https://huggingface.co/prov-gigapath/prov-gigapath"
echo ""
echo "2. Login to Hugging Face:"
echo "   huggingface-cli login"
echo ""
echo "3. Verify installation:"
echo "   python -c \"from prosper import ProsperPipeline; print('✓ PROSPER ready!')\""
echo ""
echo "4. Run demo:"
echo "   jupyter notebook examples/demo.ipynb"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
