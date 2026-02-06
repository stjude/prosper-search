#!/usr/bin/env python
"""
PePath - Pediatric Pathology Slide Search Tool

Quick-start script for running the PePath pipeline.

Usage:
    python run.py --config configs/config.example.yaml
    python run.py --slide-path /path/to/slides --output-dir ./output
    python run.py --slide-path /path/to/slides --patch-model uni2 --slide-method ot
"""

import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pepath.cli import main

if __name__ == "__main__":
    main()
