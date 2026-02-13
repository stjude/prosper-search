"""
Search submodule for PROSPER LLM integration.

Provides utilities for generating LLM-ready JSON from slide search results,
including similarity matrix loading, metadata enrichment, and neighbor extraction.
"""

from .analysis_lib import (
    load_results_new,
    load_metadata,
    build_json_for_slide,
    top_k_neighbors,
    enrich_metadata,
    filter_data_new,
)
from .slide_to_json import main as slide_to_json_main

__all__ = [
    "load_results_new",
    "load_metadata",
    "build_json_for_slide",
    "top_k_neighbors",
    "enrich_metadata",
    "filter_data_new",
    "slide_to_json_main",
]
