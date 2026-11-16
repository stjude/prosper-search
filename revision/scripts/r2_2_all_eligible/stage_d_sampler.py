#!/usr/bin/env python3
"""R2.2-ALL Stage D SAMPLER production: the full 2,778 x 2,778 all-eligible distance matrix.

Inputs are the Stage C prep shards' production SAMPLER vectors (``aggregate_cli._pool``: float16
cast, percentiles 5..95, ravel; C0 reproduced the published mosaic matrix with relative
difference 0.0). The matrix is the production float64 cityblock distance between those vectors
(``scipy.spatial.distance.cdist``, the C1-verified route). Self and same-patient cells are kept in
the saved matrix and flagged in ``excluded`` (they are never used for ranking).

``--method median_pooling`` (author amendment 2026-11-13 ~21:15 CST) runs the identical path on the
median-pooling prep vectors (``aggregate_cli._pool(feats, "median_pooling")``) with the median C1
rows; outputs go to stage_d/median_pooling/median_pooling_matrix.{npz,json}. The default is unchanged.

Checks written to the record: ids equal the 2,778 bank slides; every cell finite; zero diagonal;
exact symmetry; the 23 canary rows equal the C1 rows bit for bit; a second, independent float64
route (pairwise ``np.abs(a - b).sum()`` for a seeded sample of 2,000 cells) agrees to 1e-9 relative.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

import stage_c_lib as L  # noqa: E402

STAGE_D = L.RUN / "stage_d"


def load_prep_vectors(prep, method="sampler"):
    ids, vecs, shas = [], [], {}
    for p in sorted((Path(prep) / method).glob("*.npz")):
        shas[p.name] = L.sha256_file(p)
        with np.load(p) as z:
            ids += [str(s) for s in z["ids"]]
            vecs.append(z["vecs"])
    mat = np.vstack(vecs)
    order = np.argsort(np.array(ids, dtype=np.int64), kind="stable")
    return [ids[i] for i in order], mat[order], shas


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sampler", "median_pooling"], default="sampler")
    ap.add_argument("--out")
    ap.add_argument("--prep", default=str(L.STAGE_C / "prep"))
    ap.add_argument("--c1")
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args(argv)
    a.out = a.out or str(STAGE_D / a.method)
    a.c1 = a.c1 or str(L.STAGE_C / "c1" / f"{a.method}_rows.npz")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ids, mat, shas = load_prep_vectors(a.prep, a.method)
    if ids != L.bank_slides():
        raise ValueError(f"prep vectors cover {len(ids)} slides, not the 2,778 bank slides")
    m64 = mat.astype(np.float64)
    t_load = time.time() - t0
    t1 = time.time()
    d = cdist(m64, m64, metric="cityblock")
    t_mat = time.time() - t1
    meta = L.meta()
    pat = np.array([meta[s][0] for s in ids], dtype=object)
    excluded = pat[:, None] == pat[None, :]
    checks = {"finite": bool(np.isfinite(d).all()), "zero_diagonal": bool((np.diag(d) == 0).all()),
              "exact_symmetry": bool((d == d.T).all()), "shape": list(d.shape),
              "self_excluded": bool(np.diag(excluded).all()),
              "excluded_cells_offdiag": int(excluded.sum() - len(ids))}
    with np.load(a.c1) as z:
        pos = {s: i for i, s in enumerate(ids)}
        if list(z["cols"]) != ids:
            raise ValueError("C1 columns differ from the bank order")
        rows = d[[pos[str(q)] for q in z["queries"]]]
        checks["c1_rows_equal"] = bool((rows == z["d"]).all())
        checks["c1_rows_n"] = int(len(z["queries"]))
        checks["c1_excluded_equal"] = bool((excluded[[pos[str(q)] for q in z["queries"]]] == z["excluded"]).all())
    rng = np.random.default_rng(a.seed)
    ii, jj = rng.integers(len(ids), size=2000), rng.integers(len(ids), size=2000)
    alt = np.array([np.abs(m64[i] - m64[j]).sum() for i, j in zip(ii, jj)])
    ref = d[ii, jj]
    rel = np.abs(alt - ref) / np.maximum(np.abs(ref), 1e-300)
    checks["independent_route_max_rel"] = float(rel[ref > 0].max()) if (ref > 0).any() else 0.0
    checks["independent_route_pass"] = bool(checks["independent_route_max_rel"] <= 1e-9)
    ok = all(checks[k] for k in ("finite", "zero_diagonal", "exact_symmetry", "self_excluded",
                                 "c1_rows_equal", "c1_excluded_equal", "independent_route_pass"))
    vec_sha = hashlib.sha256(mat.tobytes()).hexdigest()
    L.save_npz(out / f"{a.method}_matrix.npz", ids=np.array(ids), d=d, excluded=excluded,
               vecs_sha=np.array(vec_sha))
    rec = {"arm": a.method, "n": len(ids), "vector_dtype": str(mat.dtype), "vector_len": int(mat.shape[1]),
           "vecs_sha256": vec_sha, "prep_shard_sha256": shas, "load_s": t_load, "matrix_s": t_mat,
           "checks": checks, "pass": ok, "vmhwm_gb": L.vmhwm_gb(), "env": L.env_record(),
           "code": L.production_shas(),
           "script_sha256": L.sha256_file(Path(__file__)), "lib_sha256": L.sha256_file(Path(L.__file__)),
           "matrix_sha256": L.sha256_file(out / f"{a.method}_matrix.npz")}
    L.dump_json(out / f"{a.method}_matrix.json", rec)
    print(f"[{a.method}] pass={ok} {checks}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
