#!/usr/bin/env python
"""
PROSPER - Pediatric Retrieval for Oncology Slides, Pathology Evidence, and Reporting

Quick-start script for running the PROSPER pipeline.

Usage:
    python run.py --config configs/config.example.yaml
    python run.py --slide-path /path/to/slides --output-dir ./output
    python run.py --slide-path /path/to/slides --patch-model uni2 --slide-method ot
"""

import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prosper.cli import main

if __name__ == "__main__":
    main()
