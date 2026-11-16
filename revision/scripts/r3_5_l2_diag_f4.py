#!/usr/bin/env python3
"""R3.5-L2 diagnostic of the F4 failure in l2/run_002 (label-shuffle control of the design-3a probe).

Diagnosis only: computes control quantities, never the observed-label probe.  Re-uses the run_002 code path
(scripts/r3_5_l2_alignment.py: MapSlide rows, knn_lopo, majority_vote, confusion, balanced_accuracy) and writes
l2/run_002/DIAG_F4/diag_f4.json:
  1. how many slides/rows have single-class annotation (a within-slide shuffle leaves them unchanged);
  2. the within-slide shuffle control exactly as run_002 (must reproduce run_002 F4 numbers);
  3. the same control scored only on rows of multi-class slides;
  4. a global (across-slide) label permutation, PCG64(123), which destroys slide composition.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r3_5_l2_alignment", REPO / "scripts" / "r3_5_l2_alignment.py")
L2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L2)
L1 = L2.L1


def main() -> int:
    run = L2.STAGE / "run_002"
    out = run / "DIAG_F4"
    out.mkdir(exist_ok=False)
    L1.assert_pinned_dispatch()
    LC, _ = L1.import_prosper()
    meta = L1.meta_cache_load(L2.BIND / "meta_tiles.pkl")
    pool = pd.read_csv(L1.bound_path(L2.BIND, "pool"), dtype=str)
    ann = pool[(pool.in_pool == "1") & (pool.in_final_library == "1")]
    ids = L1.numeric_sorted(ann.slide_id)
    pat = dict(zip(ann.slide_id, ann["UID.Subject"]))
    grp = dict(zip(ann.slide_id, ann.diagnostic_group))
    X, y, sl, pp = [], [], [], []
    for s in ids:
        ms = L1.MapSlide(s, meta[s], LC=LC)
        a = np.where(ms.row_label > 0)[0]
        X.append(ms.F[a]); y.append(ms.row_label[a].astype(np.int64)); sl += [s] * len(a); pp += [pat[s]] * len(a)
    X, y, sl, pp = np.concatenate(X), np.concatenate(y), np.array(sl), np.array(pp)
    t0 = time.time()
    nb = L2.knn_lopo(X, pp, L2.K_NN)
    secs = time.time() - t0
    rng = np.random.Generator(np.random.PCG64(L1.SEED))          # identical to run_probe
    ysh = y.copy()
    for s in ids:
        idx = np.where(sl == s)[0]
        if len(idx):
            ysh[idx] = y[idx][rng.permutation(len(idx))]
    yglob = y[np.random.Generator(np.random.PCG64(L1.SEED)).permutation(len(y))]
    res = {"script_sha256": L1.sha256_file(Path(__file__).resolve()),
           "l2_script_sha256": L1.sha256_file(REPO / "scripts" / "r3_5_l2_alignment.py"),
           "n_rows": int(len(y)), "n_slides": len(ids), "knn_seconds": round(secs, 1)}
    for gran, classes in (("5", L2.CLASSES5), ("bin", L2.CLASSES2)):
        cy = L1.class_of(y, gran == "bin")
        n_cls = pd.Series(cy).groupby(sl).nunique()
        single = set(n_cls[n_cls == 1].index)
        rows_single = np.isin(sl, list(single))
        comp = pd.DataFrame({"slide": sl, "c": cy}).groupby("slide").c.agg(lambda v: int(pd.Series(v).mode().iloc[0]))
        single_cls = pd.Series({s: comp[s] for s in single}).value_counts().to_dict() if single else {}
        single_grp = pd.Series([grp[s] for s in single]).value_counts().to_dict() if single else {}
        r = {"n_single_class_slides": len(single), "n_rows_in_single_class_slides": int(rows_single.sum()),
             "share_rows_in_single_class_slides": float(rows_single.mean()),
             "single_class_slides_by_class": {str(k): int(v) for k, v in single_cls.items()},
             "single_class_slides_by_group": {str(k): int(v) for k, v in single_grp.items()},
             "chance": 1.0 / len(classes)}
        for name, lab in (("within_slide_shuffle", ysh), ("global_permutation", yglob)):
            yt = L1.class_of(lab, gran == "bin")
            pred, tied = L2.majority_vote(yt[nb], classes)
            r[name] = {"balanced_accuracy_all_rows": L2.balanced_accuracy(L2.confusion(yt, pred, classes)),
                       "balanced_accuracy_multi_class_slide_rows": L2.balanced_accuracy(
                           L2.confusion(yt[~rows_single], pred[~rows_single], classes)),
                       "balanced_accuracy_single_class_slide_rows": L2.balanced_accuracy(
                           L2.confusion(yt[rows_single], pred[rows_single], classes)) if rows_single.any() else None,
                       "n_tied_votes": int(tied.sum())}
        res[gran] = r
    (out / "diag_f4.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
