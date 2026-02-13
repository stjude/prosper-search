#!/bin/bash
# PROSPER Quick Start Script
# This script helps set up and run PROSPER

set -e

echo "==========================================="
echo "  PROSPER - Pediatric Retrieval for Oncology"
echo "  Embedding Tool for WSI Analysis"
echo "==========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
echo "Python version: $PYTHON_VERSION"

# Function to show usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  install     Install PROSPER and dependencies"
    echo "  test        Run smoke test with sample data"
    echo "  run         Run the pipeline (requires config)"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install"
    echo "  $0 test"
    echo "  $0 run --config configs/config.yaml"
}

# Install command
install_prosper() {
    echo "Installing PROSPER..."
    echo ""
    
    # Check if venv exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    # Install PROSPER
    echo "Installing PROSPER..."
    pip install -e .
    
    echo ""
    echo "Installation complete!"
    echo "Activate the environment with: source venv/bin/activate"
    echo ""
    echo "Don't forget to log in to Hugging Face:"
    echo "  huggingface-cli login"
}

# Test command
run_test() {
    echo "Running smoke test..."
    
    if [ ! -d "venv" ]; then
        echo "Error: Virtual environment not found. Run '$0 install' first."
        exit 1
    fi
    
    source venv/bin/activate
    python run.py --slide-path ./test_data --output-dir ./test_output --smoketest
}

# Run command
run_pipeline() {
    if [ ! -d "venv" ]; then
        echo "Error: Virtual environment not found. Run '$0 install' first."
        exit 1
    fi
    
    source venv/bin/activate
    python run.py "$@"
}

# Main
case "${1:-help}" in
    install)
        install_prosper
        ;;
    test)
        run_test
        ;;
    run)
        shift
        run_pipeline "$@"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac
