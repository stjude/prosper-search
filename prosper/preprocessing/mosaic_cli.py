"""CLI: mosaic (15% tile selection) for a NEW cohort, as an LSF array.

Why a separate CLI rather than ``MosaicGenerator``: the class in ``mosaic.py``
is the *public-package* port and, for the colour-histogram step, reads a
``patch_size // 4`` region at pyramid **level 1** -- which on an Aperio SVS
(4x pyramid steps) covers the same footprint as the level-0 tile, but on a
generic TIFF with 2x pyramid steps (the CCDI-MCI cohort, and the
``solid_tumor_validation`` cohort) covers only a quarter of it. The pipeline
that actually produced the published external-cohort mosaics
(``comet-testsets/src/vertical_pipeline/mosaic/lib.py``) reads the **full
512 px tile at level 0**. This CLI reproduces that reference for TIFF cohorts:
same tissue mask, same grid, same threshold, same two-stage KMeans (colour then
spatial), same ``random_state=3``; ``n_init`` is pinned to 10 (the reference
inherits scikit-learn's default, which changed in 1.4 -- see ``mosaic.py``).

Output specification matches the published mosaic pkl exactly, so
``prosper.embeddings.extract_cli`` consumes it unchanged:
``combined_mosaics.pkl`` is a list of ``(filename_with_ext, [tile dicts])``
and every tile dict carries ``wsi_loc = [x, y]`` at level 0.

Usage (one array element handles a contiguous range of the sorted file list):
    python -m prosper.preprocessing.mosaic_cli extract --slides-dir D --ext .tiff \\
        --lo 1 --hi 2 --task mosaic/stv
    python -m prosper.preprocessing.mosaic_cli combine --task mosaic/stv --min-slides 1240
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..evaluation import paths
from .mosaic import clean_thumbnail


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def list_slides(slides_dir: Path, ext: str) -> List[Path]:
    """Deterministic (sorted) slide list -- the array index space."""
    return sorted(Path(p) for p in glob.glob(str(slides_dir / f"*{ext}")))


def slide_id(path: Path, ext: str) -> str:
    name = path.name
    return name[:-len(ext)] if name.endswith(ext) else name.rsplit(".", 1)[0]


# A thumbnail is one read_region at the coarsest level. 17 of the 1,247
# solid_tumor_validation TIFFs have <= 3 pyramid levels (11 have exactly one), so
# their coarsest level is still ~100k px wide and openslide.get_thumbnail() turns
# into a whole-slide level-0 read -- which is what killed the first mosaic fan-out
# elements on those files (TERM_MEMLIMIT at 1 GB). Above this pixel budget the
# thumbnail is assembled tile-wise instead: every 512-px level-0 tile is read and
# shrunk to a few pixels, which is the same I/O the colour-histogram step does
# anyway and needs no large allocation.
_THUMB_DIRECT_MAX_PX = 40_000_000
_THUMB_TILE_PX = 4


def _empty_tiles(slide_path: Path, patch_size: int):
    """Level-0 tiles with zero stored bytes, as a set of (col, row) grid cells.

    Found 2026-08-29 on the solid_tumor_validation cohort: the 11 single-level
    TIFFs have every level-0 tile at x >= 65,536 px stored with 0 bytes (a
    truncated export -- 12,420 of 33,660 tiles on one file), and OpenSlide
    raises ``Cannot read raw tile`` on them. Worse, an OpenSlide handle that has
    raised once fails every subsequent read, so a single bad tile silently
    blanked the tissue mask and turned the colour loop into a stream of
    caught exceptions. Reading the TIFF tag arrays first makes the empty set
    explicit, cheap, and reportable. Returns (set, n_tiles_level0, tile_width);
    an empty set if the file is not a tiled TIFF this can inspect.
    """
    try:
        import numpy as np
        import tifffile
        with tifffile.TiffFile(str(slide_path)) as tf:
            pg = tf.pages[0]
            if not pg.is_tiled:
                return set(), 0, 0
            bc = np.asarray(pg.databytecounts)
            ncols = (pg.imagewidth + pg.tilewidth - 1) // pg.tilewidth
            zero = np.where(bc == 0)[0]
            cells = set()
            # map TIFF tile (tw) cells onto the patch grid (patch_size) cells
            for idx in zero:
                r, c = divmod(int(idx), ncols)
                x0, y0 = c * pg.tilewidth, r * pg.tilelength
                for gx in range(x0 // patch_size, (x0 + pg.tilewidth - 1) // patch_size + 1):
                    for gy in range(y0 // patch_size, (y0 + pg.tilelength - 1) // patch_size + 1):
                        cells.add((gx, gy))
            return cells, int(bc.size), int(pg.tilewidth)
    except Exception:
        return set(), 0, 0


def _thumbnail(slide, patch_size: int, empty=frozenset(), slide_path=None,
               max_px: int = _THUMB_DIRECT_MAX_PX):
    from PIL import Image
    w, h = slide.dimensions
    best = slide.level_count - 1
    lw, lh = slide.level_dimensions[best]
    if lw * lh <= max_px:
        return slide.get_thumbnail((500, 500)), slide, 0
    nx = (w + patch_size - 1) // patch_size
    ny = (h + patch_size - 1) // patch_size
    t = _THUMB_TILE_PX
    canvas = Image.new("RGB", (nx * t, ny * t), (255, 255, 255))
    n_err = 0
    for j in range(ny):
        for i in range(nx):
            if (i, j) in empty:
                continue                      # known-empty: background, no read
            try:
                reg = slide.read_region((i * patch_size, j * patch_size), 0,
                                        (patch_size, patch_size)).convert("RGB")
                canvas.paste(reg.resize((t, t), Image.BILINEAR), (i * t, j * t))
            except Exception:
                # An errored OpenSlide handle fails every later read: reopen.
                n_err += 1
                if slide_path is not None:
                    import openslide
                    try:
                        slide.close()
                    except Exception:
                        pass
                    slide = openslide.open_slide(str(slide_path))
    canvas.thumbnail((500, 500), Image.BILINEAR)
    return canvas, slide, n_err


def process_slide_tiff(slide_path: Path, patch_size: int = 512,
                       percentage: int = 15, tissue_threshold: float = 0.7,
                       histogram_level: int = 0, seed: int = 3,
                       n_init: int = 10, thumb_max_px: int = _THUMB_DIRECT_MAX_PX) -> Optional[Dict]:
    """The reference TIFF-cohort mosaic (comet-testsets lib.py), one slide.

    Returns ``{"tiles": [...], "stats": {...}}`` or ``None`` on failure.
    """
    import openslide
    from sklearn.cluster import KMeans

    t0 = time.time()
    empty, n_tiff_tiles, tiff_tw = _empty_tiles(slide_path, patch_size)
    slide = openslide.open_slide(str(slide_path))
    w, h = slide.dimensions
    level_count = int(slide.level_count)
    thumbnail, slide, thumb_errors = _thumbnail(slide, patch_size, empty, slide_path,
                                                max_px=thumb_max_px)
    cthumb = clean_thumbnail(thumbnail)
    tissue_mask = (cthumb.mean(axis=2) != 255).astype(float)
    mask_hratio = (tissue_mask.shape[0] / h) * patch_size
    mask_wratio = (tissue_mask.shape[1] / w) * patch_size

    patches = []
    for i, hi in enumerate(range(0, h, int(patch_size))):
        for j, wi in enumerate(range(0, w, int(patch_size))):
            mi = int(i * mask_hratio)
            mj = int(j * mask_wratio)
            pm = tissue_mask[mi:mi + int(mask_hratio), mj:mj + int(mask_wratio)]
            cov = (np.count_nonzero(pm) / pm.size) if pm.size > 0 else 0.0
            if (j, i) in empty:
                cov = 0.0                     # zero-byte tile: nothing to read
            patches.append({"loc": [i, j], "wsi_loc": [int(wi), int(hi)],
                            "tissue_coverage": float(cov)})
    n_grid = len(patches)
    selected = [p for p in patches if p["tissue_coverage"] >= tissue_threshold]
    if not selected:
        return None

    # Colour feature: mean RGB of the tile. Reference reads the full tile at
    # level 0; --histogram-level > 0 reads the SAME footprint at a coarser
    # level (a box-averaged version of the same pixels), which is what the
    # public port should have done instead of a quarter-region at level 1.
    ds = float(slide.level_downsamples[histogram_level]) if histogram_level else 1.0
    size = max(1, int(round(patch_size / ds)))
    t1 = time.time()
    unreadable = 0
    kept = []
    for p in selected:
        x, y = p["wsi_loc"]
        try:
            region = slide.read_region((x, y), histogram_level, (size, size)).convert("RGB")
            p["rgb_histogram"] = (np.asarray(region) / 255.).reshape((-1, 3)).mean(axis=0)
            kept.append(p)
        except Exception:
            # Never substitute a grey placeholder (the earlier behaviour): a tile
            # that cannot be read cannot be a mosaic tile. Reopen the handle,
            # because one OpenSlideError poisons every later read.
            unreadable += 1
            try:
                slide.close()
            except Exception:
                pass
            slide = openslide.open_slide(str(slide_path))
    selected = kept
    if not selected:
        return None
    t_hist = time.time() - t1

    feats = np.array([p["rgb_histogram"] for p in selected])
    k_col = min(9, len(feats))
    km = KMeans(n_clusters=k_col, random_state=seed, n_init=n_init).fit(feats)
    for p, lab in zip(selected, km.labels_):
        p["cluster_lbl"] = int(lab)

    mosaic: List[Dict] = []
    for c in range(k_col):
        cp = [p for p, lab in zip(selected, km.labels_) if lab == c]
        n_sel = max(1, int(len(cp) * percentage / 100.))
        if len(cp) < n_sel:
            continue
        locs = np.array([p["wsi_loc"] for p in cp], dtype=float)
        km2 = KMeans(n_clusters=n_sel, random_state=seed, n_init=n_init)
        d = km2.fit_transform(locs)
        chosen = set()
        for idx in range(n_sel):
            for sidx in np.argsort(d[:, idx]):
                if sidx not in chosen:
                    chosen.add(int(sidx))
                    tile = {k: cp[sidx][k] for k in ("loc", "wsi_loc", "tissue_coverage", "cluster_lbl")}
                    tile["rgb_histogram"] = [float(v) for v in cp[sidx]["rgb_histogram"]]
                    mosaic.append(tile)
                    break
    slide.close()
    return {
        "tiles": mosaic,
        "stats": {
            "width": int(w), "height": int(h), "level_count": level_count,
            "n_grid": n_grid, "n_tissue": len(selected), "n_mosaic": len(mosaic),
            "tissue_fraction": len(selected) / max(1, n_grid),
            "mosaic_fraction_of_tissue": len(mosaic) / max(1, len(selected)),
            "n_zero_byte_tiff_tiles": len(empty), "n_tiff_tiles_level0": n_tiff_tiles,
            "n_unreadable_tissue_tiles": unreadable, "n_thumbnail_read_errors": thumb_errors,
            "histogram_seconds": round(t_hist, 1),
            "wall_seconds": round(time.time() - t0, 1),
        },
    }


def cmd_extract(a) -> int:
    out_root = paths.output_root()
    task_dir = out_root / a.task
    tmp = task_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    slides = list_slides(Path(a.slides_dir), a.ext)
    if not slides:
        raise SystemExit(f"no *{a.ext} under {a.slides_dir}")
    hi = a.hi or len(slides)
    sel = slides[max(1, a.lo) - 1:hi]
    print(f"[mosaic] {len(slides)} slides in {a.slides_dir}; this element "
          f"{a.lo}..{hi} ({len(sel)})", flush=True)
    done = skipped = failed = 0
    per_slide = {}
    t0 = time.time()
    for sp in sel:
        sid = slide_id(sp, a.ext)
        dest = tmp / f"{sid}.pkl"
        if dest.exists():
            try:
                with open(dest, "rb") as fh:
                    pickle.load(fh)
                skipped += 1
                continue
            except Exception:
                pass
        try:
            from .stage import stage_local
            with stage_local(sp, enabled=a.stage_local) as (sp_local, st):
                if st:
                    print(f"[mosaic] {sid}: staged {st['staged_bytes']/1e9:.2f} GB in "
                          f"{st['stage_seconds']}s to {st['stage_dir']}", flush=True)
                res = process_slide_tiff(sp_local, a.patch_size, a.percentage,
                                         a.tissue_threshold, a.histogram_level,
                                         thumb_max_px=a.thumb_direct_max_px)
                if res is not None and st:
                    res["stats"].update(st)
        except Exception as exc:                     # noqa: BLE001
            print(f"[mosaic] {sid}: FAILED {type(exc).__name__}: {exc}", flush=True)
            failed += 1
            per_slide[sid] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if res is None or not res["tiles"]:
            print(f"[mosaic] {sid}: no tissue tiles", flush=True)
            failed += 1
            per_slide[sid] = {"error": "no tissue tiles"}
            continue
        tmp_path = dest.with_suffix(".pkl.tmp")
        with open(tmp_path, "wb") as fh:
            pickle.dump(res["tiles"], fh, protocol=4)
        tmp_path.replace(dest)
        per_slide[sid] = res["stats"]
        done += 1
        s = res["stats"]
        print(f"[mosaic] {sid}: grid {s['n_grid']} tissue {s['n_tissue']} "
              f"mosaic {s['n_mosaic']} in {s['wall_seconds']}s", flush=True)
    prov = {
        "task": a.task, "slides_dir": str(a.slides_dir), "ext": a.ext,
        "patch_size": a.patch_size, "percentage_selected": a.percentage,
        "tissue_threshold": a.tissue_threshold, "histogram_level": a.histogram_level,
        "kmeans_seed": 3, "kmeans_n_init": 10,
        "reference": "comet-testsets/src/vertical_pipeline/mosaic/lib.py (TIFF variant: level-0 full-tile colour mean)",
        "range": [a.lo, hi], "n_selected": len(sel),
        "n_written": done, "n_skipped": skipped, "n_failed": failed,
        "per_slide": per_slide,
        "wall_seconds": round(time.time() - t0, 1),
        "git_commit": _git_commit(),
        "lsf_job": os.environ.get("LSB_JOBID"), "lsf_index": os.environ.get("LSB_JOBINDEX"),
    }
    pdir = out_root / "logs" / a.task.replace("/", "_")
    pdir.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("LSB_JOBINDEX") or f"{a.lo}_{hi}"
    (pdir / f"provenance_{tag}.json").write_text(json.dumps(prov, indent=2))
    print(f"[mosaic] done={done} skipped={skipped} failed={failed}", flush=True)
    return 1 if failed and not done else 0


def cmd_combine(a) -> int:
    out_root = paths.output_root()
    task_dir = out_root / a.task
    tmp = task_dir / "tmp"
    files = sorted(glob.glob(str(tmp / "*.pkl")))
    if a.min_slides and len(files) < a.min_slides:
        raise SystemExit(f"[mosaic-combine] REFUSING: {len(files)} mosaics in {tmp}, "
                         f"expected >= {a.min_slides}")
    entries = []
    counts = []
    bad = []
    for fp in files:
        sid = os.path.basename(fp)[:-4]
        try:
            with open(fp, "rb") as fh:
                tiles = pickle.load(fh)
        except Exception as exc:                     # noqa: BLE001
            bad.append(sid)
            continue
        if not tiles:
            bad.append(sid)
            continue
        entries.append((f"{sid}{a.ext}", tiles))
        counts.append(len(tiles))
    entries.sort()
    out = task_dir / "combined_mosaics.pkl"
    with open(out, "wb") as fh:
        pickle.dump(entries, fh, protocol=4)
    c = np.array(counts)
    summary = {
        "task": a.task, "n_slides": len(entries), "n_unreadable_or_empty": len(bad),
        "unreadable_or_empty": bad, "total_tiles": int(c.sum()) if c.size else 0,
        "tiles_per_slide": {"min": int(c.min()), "median": float(np.median(c)),
                            "mean": float(c.mean()), "max": int(c.max())} if c.size else None,
        "output": str(out), "git_commit": _git_commit(),
    }
    (task_dir / "combined_mosaics_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--slides-dir", required=True)
    ex.add_argument("--ext", default=".tiff")
    ex.add_argument("--lo", type=int, default=1)
    ex.add_argument("--hi", type=int, default=0)
    ex.add_argument("--task", required=True, help="e.g. mosaic/stv")
    ex.add_argument("--patch-size", type=int, default=512)
    ex.add_argument("--percentage", type=int, default=15)
    ex.add_argument("--stage-local", action="store_true",
                    help="copy each slide to node-local storage ($PROSPER_STAGE_DIR / $TMPDIR) before "
                         "tiling -- for shares with slow random reads (KidRare: 0.35 s per 64 KB read)")
    ex.add_argument("--tissue-threshold", type=float, default=0.7)
    ex.add_argument("--thumb-direct-max-px", type=int, default=_THUMB_DIRECT_MAX_PX,
                    help="read the thumbnail directly from the coarsest level when that level has at most "
                         "this many pixels; above it the thumbnail is assembled tile-wise (slow on NFS: "
                         "one read per 512-px level-0 tile). 40 Mpx fits a 1 GB job; ~120 Mpx (KidRare, "
                         "16x coarsest level of a 15 Gpx slide) needs ~1 GB of RGB and an 8 GB job")
    ex.add_argument("--histogram-level", type=int, default=0,
                    help="pyramid level for the colour mean; 0 = the reference")
    cb = sub.add_parser("combine")
    cb.add_argument("--task", required=True)
    cb.add_argument("--ext", default=".tiff")
    cb.add_argument("--min-slides", type=int, default=0)
    a = p.parse_args(argv)
    return cmd_extract(a) if a.cmd == "extract" else cmd_combine(a)


if __name__ == "__main__":
    raise SystemExit(main())
