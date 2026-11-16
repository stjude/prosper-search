#!/usr/bin/env python3
"""R2.2-ALL Stage C prep shard for the median-pooling arm (author amendment 2026-11-13 ~21:15 CST).

Per admitted bank slide, in one sha-verified read of its npz (all rows, codes 0/1/2):
  * production median-pooling vector ``aggregate_cli._pool(feats, "median_pooling")``
    (float16 cast of every row, then per-dimension median);
  * cross-check: production ``_pool(feats, "sampler")`` recomputed from the same read must equal,
    bit for bit, the vector already stored by the SAMPLER prep shards (stage_c/prep/sampler/).

Outputs under --out: median_pooling/<shard>.npz (ids, vecs, n_rows) and
records_median/<shard>.json. The shard split is the SAMPLER prep's (bank_slides()[shard::nshards])
only when --nshards matches; the cross-check looks the slide up by id, so any split works.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import stage_c_lib as L  # noqa: E402


def sampler_index(prep):
    out = {}
    for p in sorted((prep / "sampler").glob("*.npz")):
        with np.load(p) as z:
            for s, v in zip(z["ids"], z["vecs"]):
                out[str(s)] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sampler-prep", default=str(L.STAGE_C / "prep"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    for d in ("median_pooling", "records_median"):
        (out / d).mkdir(parents=True, exist_ok=True)
    agg, _, _ = L.production()
    ref = sampler_index(Path(a.sampler_prep))
    slides = L.bank_slides()[a.shard::a.nshards]
    vecs, nrows, recs = [], [], []
    for s in slides:
        t0 = time.time()
        feats = L.load_bank(s)["feats"]
        t1 = time.time()
        v = agg._pool(feats, "median_pooling")
        t2 = time.time()
        samp = agg._pool(feats, "sampler")
        same = bool(s in ref and samp.dtype == ref[s].dtype and np.array_equal(samp, ref[s]))
        if not same:
            raise ValueError(f"{s}: recomputed SAMPLER vector differs from stage_c/prep/sampler")
        if v.shape != (L.DIM,) or not np.isfinite(v).all():
            raise ValueError(f"{s}: median vector {v.dtype} {v.shape} not finite/1536")
        vecs.append(v)
        nrows.append(len(feats))
        recs.append({"slide": s, "n_rows": int(len(feats)), "t_load": t1 - t0, "t_median": t2 - t1,
                     "median_dtype": str(v.dtype), "sampler_equal_prep": same, "t_total": time.time() - t0})
        print(f"[prep-median] {s} n={len(feats)} {time.time() - t0:.1f}s", flush=True)
    L.save_npz(out / "median_pooling" / f"{a.shard:03d}.npz", ids=np.array(slides), vecs=np.vstack(vecs),
               n_rows=np.array(nrows, np.int64))
    L.dump_json(out / "records_median" / f"{a.shard:03d}.json",
                {"shard": a.shard, "nshards": a.nshards, "slides": recs, "vmhwm_gb": L.vmhwm_gb(),
                 "env": L.env_record(), "code": L.production_shas(),
                 "script_sha256": L.sha256_file(Path(__file__))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
