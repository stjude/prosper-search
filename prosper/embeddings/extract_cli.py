"""CLI: extract tile features for a contiguous range of mosaic slides.

One LSF array element handles one range, so the array size is what we manage
and the per-slide count is the scheduler's problem. Idempotent: a slide whose
output pkl already loads is skipped, which makes a re-submission after a wall
-clock kill a resume rather than a restart (the published drivers had this
guard and it is worth keeping).

Provenance travels with the result (standing rule 1): every run writes a
sidecar JSON recording the encoder, the resolved mosaic path, tile counts,
throughput and the git commit.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..evaluation import paths
from . import tile_features as T


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", required=True, choices=sorted(T.REGISTRY))
    p.add_argument("--mosaic", default=None,
                   help="mosaic pkl; default is the published 15% mosaic")
    p.add_argument("--lo", type=int, default=1, help="1-based first slide index")
    p.add_argument("--hi", type=int, default=0, help="1-based last (0 = all)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--stage-local", action="store_true",
                   help="copy each slide to node-local storage before reading tiles (slow-random-read shares)")
    p.add_argument("--task", default=None,
                   help="output subdir; default features/<encoder>")
    p.add_argument("--limit-tiles", type=int, default=0,
                   help="debug: cap tiles per slide")
    p.add_argument("--tile-source", default="mosaic",
                   choices=["mosaic", "all_patches"],
                   help="mosaic = the published 15%% selection. all_patches = "
                        "the per-slide all-fully-tissue-tile CSVs, for the R2.2 "
                        "tile-fraction ladder. The SLIDE LIST always comes from "
                        "the mosaic pkl either way, so slide indices stay "
                        "comparable across arms and the 34 slides with no "
                        "all-patches list are reported rather than silently "
                        "dropped.")
    a = p.parse_args(argv)

    out_root = paths.output_root()
    task = a.task or f"features/{a.encoder}"
    feat_dir = out_root / task
    feat_dir.mkdir(parents=True, exist_ok=True)

    mosaic_path = Path(a.mosaic) if a.mosaic else (
        Path(os.environ.get(
            "COMET_SCANS_ROOT",
            "/PROSPER_WORKSPACE/comet-new-scans"))
        / "output" / "mosaic" / "mosaic512_final_simplified.pkl")
    paths.require(mosaic_path, "mosaic pkl")

    entries = T.load_mosaic(mosaic_path)
    hi = a.hi or len(entries)
    lo = max(1, a.lo)
    sel = entries[lo - 1:hi]
    print(f"[extract] encoder={a.encoder} slides {lo}..{hi} of {len(entries)} "
          f"({len(sel)} in this element)", flush=True)

    device = a.device if torch.cuda.is_available() else "cpu"
    if device != a.device:
        print(f"[extract] WARNING cuda unavailable, falling back to {device}",
              flush=True)
    model, transform = T.load_encoder(a.encoder, device=device)
    print(f"[extract] model loaded on {device}; "
          f"dim={T.REGISTRY[a.encoder].dim}", flush=True)

    done = skipped = failed = 0
    tiles_done = 0
    no_tile_list = []
    t0 = time.time()
    for entry in sel:
        sid = T.slide_id_of(entry)
        if a.tile_source == "all_patches":
            mosaic = T.load_all_patches(sid)
            if mosaic is None:
                # One of the 34 mosaic slides with no all-patches list. Counted
                # and named, never silently skipped: standing rule 4 turns on
                # this set difference, so it has to stay visible in the log.
                print(f"[extract] {sid}: NO all-patches list, skipping",
                      flush=True)
                no_tile_list.append(sid)
                continue
        else:
            mosaic = entry[1]
        if a.limit_tiles:
            mosaic = mosaic[:a.limit_tiles]
        dest = feat_dir / f"{sid}.pkl"
        if dest.exists():
            try:
                with open(dest, "rb") as fh:
                    pickle.load(fh)
                skipped += 1
                continue
            except Exception:
                print(f"[extract] {sid}: unreadable output, recomputing",
                      flush=True)
        slide_path = paths.slide_wsi_path(sid)
        if not slide_path.exists():
            print(f"[extract] {sid}: MISSING SLIDE {slide_path}", flush=True)
            failed += 1
            continue
        ts = time.time()
        try:
            from ..preprocessing.stage import stage_local
            with stage_local(slide_path, enabled=a.stage_local) as (local_path, st):
                if st:
                    print(f"[extract] {sid}: staged {st['staged_bytes']/1e9:.2f} GB in "
                          f"{st['stage_seconds']}s", flush=True)
                feats = T.extract_slide(
                    local_path, mosaic, model, transform,
                    patch_size=a.patch_size, batch_size=a.batch_size,
                    num_workers=a.num_workers, device=device, amp=not a.no_amp)
        except Exception as exc:                     # noqa: BLE001
            print(f"[extract] {sid}: FAILED {type(exc).__name__}: {exc}",
                  flush=True)
            failed += 1
            continue
        if feats is None:
            print(f"[extract] {sid}: no features", flush=True)
            failed += 1
            continue
        tmp = dest.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump((feats, mosaic), fh, protocol=4)
        tmp.replace(dest)
        done += 1
        tiles_done += len(mosaic)
        dt = time.time() - ts
        print(f"[extract] {sid}: {feats.shape} in {dt:.1f}s "
              f"({len(mosaic)/max(dt,1e-9):.1f} tiles/s)", flush=True)

    wall = time.time() - t0
    prov = {
        "task": task, "encoder": a.encoder,
        "encoder_dim": T.REGISTRY[a.encoder].dim,
        "mosaic_path": str(mosaic_path),
        "tile_source": a.tile_source,
        "all_patches_dir": (str(T.ALL_PATCHES_DIR)
                            if a.tile_source == "all_patches" else None),
        "n_no_tile_list": len(no_tile_list),
        "slides_no_tile_list": no_tile_list,
        "slide_range": [lo, hi], "n_selected": len(sel),
        "n_written": done, "n_skipped": skipped, "n_failed": failed,
        "tiles_written": tiles_done,
        "wall_seconds": round(wall, 1),
        "tiles_per_second": round(tiles_done / wall, 2) if wall > 0 else None,
        "batch_size": a.batch_size, "num_workers": a.num_workers,
        "patch_size": a.patch_size, "amp": not a.no_amp,
        "device": device, "torch": torch.__version__,
        "git_commit": _git_commit(),
        "lsf_job": os.environ.get("LSB_JOBID"),
        "lsf_index": os.environ.get("LSB_JOBINDEX"),
    }
    pdir = out_root / "logs" / f"extract_{a.encoder}"
    pdir.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("LSB_JOBINDEX") or f"{lo}_{hi}"
    (pdir / f"provenance_{tag}.json").write_text(json.dumps(prov, indent=2))
    print(f"[extract] done={done} skipped={skipped} failed={failed} "
          f"tiles={tiles_done} wall={wall:.1f}s "
          f"rate={prov['tiles_per_second']} tiles/s", flush=True)
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
