#!/usr/bin/env python3
"""Pack one per-slide feature bank (``<slide>.pkl`` = (features, [{'wsi_loc': [x, y]}, ...]))
into a contiguous float32 array for the R3.5-ROI6 GPU search ([analysis plan], 2026-11-11).

Outputs in ``--out-dir``: ``features.f32`` (raw row-major float32, rows in numeric slide
order, each slide's rows in pickle order), ``coords.i64`` (wsi_loc per row), ``slides.csv``
(row ranges, source dtype, source/coordinate SHA-256) and ``bank.json``; ``COMPLETE`` last.
Sources are read only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--src-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--expect-dim", required=True, type=int)
    ap.add_argument("--expect-slides", type=int, default=2921)
    a = ap.parse_args(argv)

    out = a.out_dir
    if (out / "COMPLETE").exists():
        print(f"{out} already COMPLETE", flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted((p.name[:-4] for p in a.src_dir.iterdir() if p.name.endswith(".pkl")), key=int)
    if len(ids) != a.expect_slides:
        raise SystemExit(f"{len(ids)} pickles in {a.src_dir}, expected {a.expect_slides}")

    t0 = time.time()
    fh = hashlib.sha256()
    rows, start = [], 0
    with open(out / "features.f32.tmp", "wb") as ff, open(out / "coords.i64.tmp", "wb") as fc:
        for i, s in enumerate(ids):
            p = a.src_dir / f"{s}.pkl"
            raw = p.read_bytes()
            obj = pickle.loads(raw)
            f, locs = obj[0], obj[1]
            if f.ndim != 2 or f.shape[1] != a.expect_dim or f.shape[0] != len(locs) or f.shape[0] == 0:
                raise SystemExit(f"{s}: features {f.shape}, {len(locs)} coordinates")
            c = np.asarray([d["wsi_loc"] for d in locs], dtype=np.int64)
            b = np.ascontiguousarray(f, dtype=np.float32).tobytes()
            ff.write(b)
            fh.update(b)
            fc.write(c.tobytes())
            n = f.shape[0]
            rows.append(dict(slide_id=s, row_start=start, row_stop=start + n, n_rows=n, src_dtype=str(f.dtype),
                             src_sha256=hashlib.sha256(raw).hexdigest(),
                             coords_sha256=hashlib.sha256(c.tobytes()).hexdigest()))
            start += n
            if (i + 1) % 250 == 0:
                print(f"[{a.name}] {i + 1}/{len(ids)} slides, {start} rows, {time.time() - t0:.0f}s", flush=True)
    os.replace(out / "features.f32.tmp", out / "features.f32")
    os.replace(out / "coords.i64.tmp", out / "coords.i64")
    pd.DataFrame(rows).to_csv(out / "slides.csv", index=False)
    dt = pd.Series([r["src_dtype"] for r in rows]).value_counts().to_dict()
    rec = dict(name=a.name, src_dir=str(a.src_dir), n_slides=len(ids), n_rows=start, dim=a.expect_dim,
               dtype="float32", order="numeric slide id; pickle row order within slide",
               features_sha256=fh.hexdigest(), coords_sha256=sha256_file(out / "coords.i64"),
               slides_csv_sha256=sha256_file(out / "slides.csv"), src_dtype_counts=dt,
               script=str(Path(__file__).resolve()), script_sha256=sha256_file(Path(__file__).resolve()),
               host=socket.gethostname(), lsb_jobid=os.environ.get("LSB_JOBID"),
               python=sys.version.split()[0], numpy=np.__version__,
               seconds=round(time.time() - t0, 1), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out / "bank.json").write_text(json.dumps(rec, indent=1))
    (out / "COMPLETE").write_text(rec["features_sha256"] + "\n")
    print(json.dumps({k: rec[k] for k in ("name", "n_slides", "n_rows", "src_dtype_counts", "seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
