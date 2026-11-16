"""Stage a WSI to node-local storage before random-access tiling.

Why (C13, 2026-08-30): the KidRare files on ``/PROSPER_GROUP`` read
sequentially at ~59 MB/s but cost **~0.35 s per random 64 KB read**; a 15-Gpx
slide has ~58k JPEG tiles, so tile-wise access from the share is hours per
slide while a whole-file copy is ~80 s. ``stage_local`` copies the file once,
verifies the byte count, hands back the local path and deletes it afterwards.

Directory: ``$PROSPER_STAGE_DIR`` if set, else ``$TMPDIR`` (LSF sets one per
job), else ``/tmp``. Refuses to stage when free space is below 1.1x the file
size, so a full local disk fails loudly instead of producing a truncated copy.
"""
from __future__ import annotations

import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path


def stage_dir() -> Path:
    for var in ("PROSPER_STAGE_DIR", "TMPDIR"):
        v = os.environ.get(var)
        if v:
            return Path(v)
    return Path("/tmp")


@contextmanager
def stage_local(path: Path, enabled: bool = True, buf_mb: int = 16):
    """Yield a local copy of ``path`` (or ``path`` itself when disabled)."""
    path = Path(path)
    if not enabled:
        yield path, None
        return
    src = path.resolve()
    size = src.stat().st_size
    d = stage_dir() / f"prosper_stage_{os.environ.get('LSB_JOBID', os.getpid())}"
    d.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(d).free
    if free < size * 1.1:
        raise RuntimeError(f"stage_local: {free/1e9:.1f} GB free in {d} < 1.1 x {size/1e9:.1f} GB ({src.name})")
    dst = d / path.name
    t0 = time.time()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=buf_mb << 20)
    got = dst.stat().st_size
    if got != size:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"stage_local: copied {got} of {size} bytes for {src.name}")
    stats = {"staged_bytes": size, "stage_seconds": round(time.time() - t0, 1), "stage_dir": str(d)}
    try:
        yield dst, stats
    finally:
        dst.unlink(missing_ok=True)
