"""
Search module for COMET pipeline.
Handles ROI-level and slide-level similarity search.
"""

from .slide_search import SlideSearchEngine
from .roi_search import ROISearchEngine

__all__ = ["SlideSearchEngine", "ROISearchEngine"]
