"""Make the revision analysis scripts importable as ``scripts.<name>``.

The revision tests were run from the analysis workspace root, where the scripts
lived under ``scripts/``; in this repository that directory is ``revision/scripts/``.
"""
import sys
from pathlib import Path

_REVISION = str(Path(__file__).resolve().parent)
if _REVISION not in sys.path:
    sys.path.insert(0, _REVISION)
