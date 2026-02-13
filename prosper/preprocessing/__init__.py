"""
Preprocessing module for PROSPER pipeline.
Handles mosaic generation from whole slide images.
"""

from .mosaic import MosaicGenerator, process_slide

__all__ = ["MosaicGenerator", "process_slide"]
