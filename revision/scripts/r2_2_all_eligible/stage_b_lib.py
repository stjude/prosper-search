"""Shared helpers for R2.2-ALL Stage B (gap-fill UNI2 extraction, fidelity probes, assembly).

Pure numpy/pandas/stdlib so it can be imported by the base python (planning, comparison)
and by the comet-uni2 python (extraction). Nothing here touches a comet-* tree on write.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import pickle
from pathlib import Path

import numpy as np

H = Path("/PROSPER_WORKSPACE")
MOSAIC_BANK = H / "comet-feature-gen/comet-uni2/data/tmp_final"
FULL_BANK_REL = "features/uni2_all_patches"
WSI_ROOT = H / "data/comet_final_HPC"
FALLBACK_SLIDES = ("9934", "11682", "14191")
DIM = 1536

COORD_COLUMNS = ("wsi_loc_x", "wsi_loc_y", "coverage")


def sha256_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_manifest(path):
    """Stage A per-slide manifest; coverage parsed exactly (round_trip)."""
    import pandas as pd
    return pd.read_csv(path, float_precision="round_trip",
                       dtype={"source": "string"}, keep_default_na=False)


def gap_mask(df):
    """D1 gap: generator-eligible (cluster_locs) and in neither existing bank."""
    return ((df["in_cluster_locs"].to_numpy() == 1)
            & (df["in_mosaic"].to_numpy() == 0)
            & (df["in_full_bank"].to_numpy() == 0))


def write_coords(path, xy, coverage):
    """Coordinate manifest consumed by stage_b_extract.py (gzip CSV, atomic)."""
    xy = np.asarray(xy, dtype=np.int64)
    coverage = np.asarray(coverage, dtype=np.float64)
    assert xy.ndim == 2 and xy.shape[1] == 2 and len(xy) == len(coverage)
    buf = io.StringIO()
    buf.write(",".join(COORD_COLUMNS) + "\n")
    for (x, y), c in zip(xy, coverage):
        buf.write(f"{int(x)},{int(y)},{float(c)!r}\n")
    data = gzip.compress(buf.getvalue().encode(), mtime=0)
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return sha256_bytes(data)


def read_coords(path):
    with gzip.open(path, "rt") as fh:
        header = fh.readline().strip().split(",")
        if tuple(header) != COORD_COLUMNS:
            raise ValueError(f"{path}: header {header} != {COORD_COLUMNS}")
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    xy = np.array([[int(r[0]), int(r[1])] for r in rows], dtype=np.int64).reshape(-1, 2)
    cov = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return xy, cov


def coord_key(xy):
    xy = np.asarray(xy, dtype=np.int64)
    return xy[:, 0] * (1 << 32) + xy[:, 1]


def match_rows(ref_xy, query_xy):
    """Row index in ``ref_xy`` of every query coordinate; raises on duplicate or absent."""
    rk = coord_key(ref_xy)
    qk = coord_key(query_xy)
    if len(np.unique(rk)) != len(rk):
        raise ValueError("duplicate coordinates in reference")
    if len(np.unique(qk)) != len(qk):
        raise ValueError("duplicate coordinates in query")
    order = np.argsort(rk, kind="stable")
    pos = np.searchsorted(rk[order], qk)
    pos = np.clip(pos, 0, len(rk) - 1)
    idx = order[pos]
    if len(rk) == 0 or not np.array_equal(rk[idx], qk):
        missing = int(np.sum(rk[idx] != qk)) if len(rk) else len(qk)
        raise KeyError(f"{missing} query coordinates absent from reference")
    return idx


def load_mosaic_bank(slide):
    with open(MOSAIC_BANK / f"{slide}.pkl", "rb") as fh:
        feats, mosaic = pickle.load(fh)
    xy = np.array([p["wsi_loc"] for p in mosaic], dtype=np.int64).reshape(-1, 2)
    return np.asarray(feats), xy


def compare_rows(a, b):
    """Per-row agreement of two (n, d) float32 arrays (new vs reference)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    assert a.shape == b.shape and a.ndim == 2
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    cos = (a64 * b64).sum(1) / (np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1))
    maxabs = np.abs(a64 - b64).max(1)
    bar = ((np.diff(a, axis=1) < 0) == (np.diff(b, axis=1) < 0)).mean(1)
    ident = np.all(a.view(np.uint32) == b.view(np.uint32), axis=1)
    return {"cos": cos, "maxabs": maxabs, "barcode": bar, "identical": ident}


def pack_tasks(slide_counts, target_tiles):
    """Greedy first-fit-decreasing packing of whole slides into tasks of ~target_tiles.

    ``slide_counts``: dict slide -> n tiles (>0). A slide larger than the target is its own
    task. Deterministic: ties broken by slide id. Returns a list of lists of slide ids.
    """
    items = sorted(((int(n), str(s)) for s, n in slide_counts.items() if int(n) > 0),
                   key=lambda t: (-t[0], t[1]))
    bins = []  # [load, [slides]]
    for n, s in items:
        for b in bins:
            if b[0] + n <= target_tiles:
                b[0] += n
                b[1].append(s)
                break
        else:
            bins.append([n, [s]])
    return [sorted(b[1], key=lambda x: (len(x), x)) for b in bins]


def dump_json(path, obj):
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False, default=_jdefault) + "\n")
    os.replace(tmp, path)


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
