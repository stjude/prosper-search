"""Benchmark / evaluation layer for PROSPER.

Provenance
----------
``slide_eval.py``, ``analysis_lib.py``, ``benchmark.py`` and ``roi_eval.py`` are
copies of the scripts that produced the submitted manuscript's numbers, taken
2026-08-27 from:

    comet-feature-gen/comet-analysis/src/slide_eval.py               (byte-identical)
    comet-feature-gen/comet-analysis/src/analysis_lib.py             (byte-identical)
    comet-feature-gen/comet-analysis/src/benchmarking_slides_final.py (one import line)
    comet-new-scans/src/calculate_CI/roi_agg_eval.py                 (byte-identical)

They were copied rather than rewritten on purpose: a refactor cannot be
distinguished from a regression until the Table S3 reproduction gate passes.
Restructuring happens *after* that gate, not before.

The ``comet-*`` trees are read-only inputs. All modification happens here.
"""

from . import metrics, paths, slide_headers  # noqa: F401

__all__ = ["metrics", "paths", "slide_headers"]
