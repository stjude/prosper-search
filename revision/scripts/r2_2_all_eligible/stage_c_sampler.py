#!/usr/bin/env python3
"""R2.2-ALL Stage C pooling canary: SAMPLER (default) or, with ``--method median_pooling``, production
median pooling (``aggregate_cli._pool(feats, "median_pooling")``: float16 cast, per-dimension median);
both share the published cityblock path, so the description below applies to either method.

c0    production ``aggregate_cli._slide_vectors(tmp_final, None, None, "sampler")`` (the exact
      published path: float16 cast, percentiles 5..95, ravel), then float64 cityblock rows for the
      canary queries against all 2,921 mosaic slides (self excluded).
rows  the prep shards' production SAMPLER vectors for the 2,778 all-eligible bank slides, then
      float64 cityblock rows for the canary queries against the 2,778 library. Self and
      same-patient columns are kept in the saved row but flagged ``excluded`` (never a distance
      used for ranking). Also times one full 2,778 x 2,778 pdist for the production projection
      (the matrix itself is not saved: Stage D is not released).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy.spatial.distance import cdist, pdist  # noqa: E402

import stage_c_lib as L  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["c0", "rows"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep", default=str(L.STAGE_C / "prep"))
    ap.add_argument("--method", choices=["sampler", "median_pooling"], default="sampler")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    agg, _, _ = L.production()
    queries = [x["slide"] for x in L.selection()["queries"]]
    t0 = time.time()
    if a.mode == "c0":
        ids, mat = agg._slide_vectors(L.MOSAIC_BANK, None, None, a.method)
    else:
        ids, vecs = [], []
        for p in sorted((Path(a.prep) / a.method).glob("*.npz")):
            with np.load(p) as z:
                ids += [str(s) for s in z["ids"]]
                vecs.append(z["vecs"])
        mat = np.vstack(vecs)
        order = np.argsort(np.array(ids, dtype=np.int64), kind="stable")
        ids = [ids[i] for i in order]
        mat = mat[order]
        if ids != L.bank_slides():
            raise ValueError(f"prep vectors cover {len(ids)} slides, not the 2,778 bank slides")
    t_pool = time.time() - t0
    m64 = mat.astype(np.float64)
    pos = {s: i for i, s in enumerate(ids)}
    t1 = time.time()
    rows = cdist(m64[[pos[q] for q in queries]], m64, metric="cityblock")
    t_rows = time.time() - t1
    meta = L.meta()
    excl = np.array([[c == q or meta[c][0] == meta[q][0] for c in ids] for q in queries])
    rec = {"mode": a.mode, "method": a.method, "n_library": len(ids), "vector_dtype": str(mat.dtype),
           "vector_len": int(mat.shape[1]), "pool_or_load_s": t_pool, "rows_s": t_rows}
    if a.mode == "rows":
        t2 = time.time()
        pdist(m64, metric="cityblock")
        rec["full_pdist_s"] = time.time() - t2
    L.save_npz(out / f"{a.method}_{a.mode}.npz", queries=np.array(queries), cols=np.array(ids),
               d=rows, excluded=excl, vecs_sha=np.array(L.hashlib.sha256(mat.tobytes()).hexdigest()))
    rec.update(vmhwm_gb=L.vmhwm_gb(), env=L.env_record(), code=L.production_shas())
    L.dump_json(out / f"{a.method}_{a.mode}.json", rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
