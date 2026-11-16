#!/usr/bin/env python3
"""Bounded re-solve of the largest UNI2 30 % exact-OT slide pairs (R2.2, OT-30 §7).

Why: gate-4 review F1 found that some 30 % ``ot.emd2`` solves hit POT's default
``numItermax=1e6`` (result code 3) silently (process-wide warning filter in
``prosper/preprocessing/mosaic.py:60``). This script re-solves every directed pair
whose 30 % cost matrix has >= THRESHOLD cells, FIRST at production's
``numItermax=1e6`` under production's BLAS thread count (8 slots, no thread env
vars; the float32 cost matrix depends on the OpenBLAS thread count, review F1).
If that solve reproduces the stored value bit-exactly, its result code decides
CONVERGED (1) or CAPPED (3); capped directions are then re-solved at 1e8. A
direction whose 1e6 solve does not reproduce the stored value is NON_REPRODUCED
and is counted separately (no exact capped count is claimed while any exist).
Top-10 impact is checked on the saved 2,750-query ranked-list population only.

It deliberately does NOT import ``prosper``; the kernel below is the independent
re-implementation that reproduced stored values exactly in review script
``c5_recompute_pair.py``. Stored pickles/matrices are never modified.

Modes
  select     write pairs.csv (all unordered pairs with threshold <= n30_i*n30_j [< max-cells])
  solve      re-solve one chunk of pairs.csv (both directions) -> solves/part_K.csv
  summarize  aggregate parts -> summary.json, top10_impact.csv, COMPLETE
  selfcheck  synthetic consistency checks (no data)
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

FRACTION = 0.30
TRUNCATE = 1536
LADDER_SEED = 20260827
ITER_HIGH = 100_000_000
DEFAULT_THRESHOLD = 50_000_000
MASK_VALUE = 1_000_000.0
TOPK = 10
EXPECT_THREADS = 8
ITER_PROD = 1_000_000


def provenance(expect_threads: int) -> dict:
    import platform
    import scipy
    import ot
    from threadpoolctl import threadpool_info
    blas = [i for i in threadpool_info() if i.get("internal_api") == "openblas"]
    aff = len(os.sched_getaffinity(0))
    info = {"host": platform.node(), "cpu": _cpu_model(), "python": sys.version.split()[0],
            "numpy": np.__version__, "scipy": scipy.__version__, "pot": ot.__version__,
            "openblas": [{k: v for k, v in b.items() if k in ("version", "num_threads", "architecture", "threading_layer")} for b in blas],
            "affinity_cpus": aff, "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "lsb_jobid": os.environ.get("LSB_JOBID"), "lsb_jobindex": os.environ.get("LSB_JOBINDEX"),
            "script_sha256": sha256(Path(__file__)), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if expect_threads:
        if aff != expect_threads or not blas or any(b["num_threads"] != expect_threads for b in blas):
            raise SystemExit(f"thread setup mismatch: affinity={aff}, openblas={[b.get('num_threads') for b in blas]}, expected {expect_threads}")
        if any(v is not None for v in info["thread_env"].values()):
            raise SystemExit(f"thread env vars must be unset: {info['thread_env']}")
    return info


def _cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


# ----------------------------------------------------------------------------- kernel
def n_keep(n: int, fraction: float = FRACTION) -> int:
    return max(1, int(round(n * fraction)))


def subsample_idx(n: int, sid: str, fraction: float = FRACTION) -> np.ndarray:
    k = n_keep(n, fraction)
    if k >= n:
        return np.arange(n)
    seed = int(hashlib.sha256(str(sid).encode()).hexdigest()[:8], 16) ^ LADDER_SEED
    order = np.random.default_rng(seed).permutation(n)
    return np.sort(order[:k])


def load_features(feature_dir: Path, sid: str) -> np.ndarray:
    obj = pickle.load(open(feature_dir / f"{sid}.pkl", "rb"))
    feats = obj[0] if isinstance(obj, tuple) and len(obj) == 2 else obj
    return np.asarray(feats)[:, :TRUNCATE]


def prepare(feats: np.ndarray, sid: str) -> np.ndarray:
    sub = feats[subsample_idx(feats.shape[0], sid)].astype(np.float32)
    norms = np.maximum(np.linalg.norm(sub, axis=1, keepdims=True), 1e-8)
    return sub / norms


def emd_cost(fa: np.ndarray, fb: np.ndarray, num_iter: int):
    import ot  # POT; imported lazily so selfcheck/select do not need it
    a = np.ones(len(fa)) / len(fa)
    b = np.ones(len(fb)) / len(fb)
    M = ot.dist(fa, fb, metric="cosine")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        cost, log = ot.emd2(a, b, M, numItermax=num_iter, log=True)
        dt = time.time() - t0
    return float(cost), int(log["result_code"]), [str(w.message) for w in caught], dt


# ----------------------------------------------------------------------------- IO
def feature_count(feature_dir: Path, sid: str) -> int:
    obj = pickle.load(open(feature_dir / f"{sid}.pkl", "rb"))
    feats = obj[0] if isinstance(obj, tuple) and len(obj) == 2 else obj
    return int(np.asarray(feats).shape[0])


def stored_row(sim_dir: Path, sid: str) -> dict:
    return dict(pickle.load(open(sim_dir / f"similarity_{sid}.pkl", "rb")))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- modes
def mode_select(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    feature_dir = Path(a.feature_dir)
    sids = sorted(Path(p).stem for p in glob.glob(str(feature_dir / "*.pkl")))
    if a.slide_list:
        keep = set(l.strip() for l in open(a.slide_list) if l.strip())
        sids = [s for s in sids if s in keep]
    counts = {s: feature_count(feature_dir, s) for s in sids}
    n30 = {s: n_keep(counts[s]) for s in sids}
    rows = []
    for i, si in enumerate(sids):
        for sj in sids[i + 1:]:
            cells = n30[si] * n30[sj]
            if cells >= a.threshold and (a.max_cells is None or cells < a.max_cells):
                rows.append((si, sj, counts[si], counts[sj], n30[si], n30[sj], cells))
    rows.sort(key=lambda r: -r[6])
    with open(out / "pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slide_a", "slide_b", "n_a", "n_b", "n30_a", "n30_b", "cells"])
        w.writerows(rows)
    meta = {"threshold": a.threshold, "max_cells": a.max_cells, "n_slides": len(sids), "n_pairs": len(rows),
            "fraction": FRACTION, "truncate": TRUNCATE, "ladder_seed": LADDER_SEED,
            "feature_dir": str(feature_dir), "n30_min": min(n30.values()),
            "n30_max": max(n30.values()), "cells_max": rows[0][6] if rows else 0}
    json.dump(meta, open(out / "select_meta.json", "w"), indent=2)
    print(json.dumps(meta))


def mode_solve(a):
    out = Path(a.out)
    (out / "solves").mkdir(parents=True, exist_ok=True)
    prov = provenance(a.expect_threads)
    pairs_path = out / "pairs.csv"
    prov.update({"pairs_sha256": sha256(pairs_path), "n_tasks": a.n_tasks, "task_index": a.task_index,
                 "iter_prod": ITER_PROD, "iter_high": a.num_iter})
    pairs = list(csv.DictReader(open(pairs_path)))
    chunk = pairs[a.task_index::a.n_tasks]
    feature_dir, sim_dir = Path(a.feature_dir), Path(a.similarity_dir)
    cache: dict = {}

    def feats(sid):
        if sid not in cache:
            if len(cache) >= a.cache_size:
                cache.pop(next(iter(cache)))
            cache[sid] = prepare(load_features(feature_dir, sid), sid)
        return cache[sid]

    part = out / "solves" / f"part_{a.task_index:03d}.csv"
    tmp = part.with_suffix(".csv.partial")
    cols = ["query", "other", "n_q", "n_o", "cells", "stored", "stored_hex", "prod_cost", "prod_cost_hex",
            "prod_code", "bit_equal", "classification", "high_cost", "high_cost_hex", "high_code",
            "delta_stored_minus_high", "seconds_prod", "seconds_high", "warnings"]
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in chunk:
            fa, fb = feats(r["slide_a"]), feats(r["slide_b"])
            if fa.shape[0] != int(r["n30_a"]) or fb.shape[0] != int(r["n30_b"]):
                raise RuntimeError(f"n30 mismatch for pair {r['slide_a']},{r['slide_b']}")
            for q, o, fq, fo in ((r["slide_a"], r["slide_b"], fa, fb), (r["slide_b"], r["slide_a"], fb, fa)):
                stored = float(stored_row(sim_dir, q)[o])
                if not (stored < MASK_VALUE):
                    raise RuntimeError(f"stored sentinel/non-finite for {q}->{o}: {stored!r}")
                pc, pcode, pw, pdt = emd_cost(fq, fo, ITER_PROD)
                bit_equal = pc == stored
                if bit_equal and pcode == 1:
                    cls = "CONVERGED"
                elif bit_equal and pcode == 3:
                    cls = "CAPPED"
                elif bit_equal:
                    cls = f"BIT_EQUAL_CODE_{pcode}"
                else:
                    cls = "NON_REPRODUCED"
                hc = hcode = hdt = None
                hw: list = []
                if cls != "CONVERGED":
                    hc, hcode, hw, hdt = emd_cost(fq, fo, a.num_iter)
                w.writerow([q, o, fq.shape[0], fo.shape[0], fq.shape[0] * fo.shape[0],
                            repr(stored), stored.hex(), repr(pc), pc.hex(), pcode, int(bit_equal), cls,
                            "" if hc is None else repr(hc), "" if hc is None else hc.hex(),
                            "" if hcode is None else hcode,
                            "" if hc is None else repr(stored - hc),
                            f"{pdt:.2f}", "" if hdt is None else f"{hdt:.2f}", "|".join(pw + hw)])
                fh.flush()
    json.dump(prov, open(part.with_suffix(".json"), "w"), indent=2)
    os.replace(tmp, part)
    print(f"part {a.task_index}: {2 * len(chunk)} directed solves -> {part}")


def _topk(row: np.ndarray, ids: list, k: int = TOPK) -> list:
    valid = np.where(row < MASK_VALUE)[0]
    order = valid[np.lexsort((np.array(ids, dtype=object)[valid], row[valid]))]
    return [ids[i] for i in order[:k]]


def mode_summarize(a):
    import pandas as pd
    out = Path(a.out)
    pairs_sha = sha256(out / "pairs.csv")
    parts = sorted(glob.glob(str(out / "solves" / "part_*.csv")))
    if len(parts) != a.n_tasks:
        sys.exit(f"expected {a.n_tasks} parts, found {len(parts)}; no COMPLETE written")
    provs = []
    for p in parts:
        pr = json.load(open(Path(p).with_suffix(".json")))
        if pr["n_tasks"] != a.n_tasks or pr["pairs_sha256"] != pairs_sha:
            sys.exit(f"stale part {p}: n_tasks={pr['n_tasks']} pairs_sha={pr['pairs_sha256'][:12]}")
        provs.append(pr)
    if sorted(pr["task_index"] for pr in provs) != list(range(a.n_tasks)):
        sys.exit("task indices in part metadata are not 0..n_tasks-1")
    df = pd.concat([pd.read_csv(p, dtype={"query": str, "other": str}, float_precision="round_trip",
                                keep_default_na=False) for p in parts], ignore_index=True)
    pairs = list(csv.DictReader(open(out / "pairs.csv")))
    expected = {(r["slide_a"], r["slide_b"]) for r in pairs} | {(r["slide_b"], r["slide_a"]) for r in pairs}
    got = list(zip(df["query"], df["other"]))
    if len(got) != len(set(got)) or set(got) != expected:
        sys.exit(f"directed-pair set mismatch: {len(got)} rows, {len(set(got))} unique, {len(expected)} expected")
    cls = df["classification"].value_counts().to_dict()
    capped = df[df["classification"] == "CAPPED"].copy()
    capped["delta"] = capped["delta_stored_minus_high"].astype(float)
    nonrep = int((df["classification"] == "NON_REPRODUCED").sum())
    other = int(df["classification"].str.startswith("BIT_EQUAL_CODE_").sum())
    high_not_conv = int((capped["high_code"].astype(str) != "1").sum()) if len(capped) else 0
    # a cap can only leave the stored value too high; a non-positive delta is impossible (review R1)
    neg_delta = int((capped["delta"] <= 0).sum()) if len(capped) else 0
    pairs_affected = {tuple(sorted(t)) for t in zip(capped["query"], capped["other"])}
    bands = {}
    for lo in (40, 50, 60, 70, 80, 90):
        sel = capped[(capped["cells"] >= lo * 1_000_000) & (capped["cells"] < (lo + 10) * 1_000_000)]
        tot = df[(df["cells"] >= lo * 1_000_000) & (df["cells"] < (lo + 10) * 1_000_000)]
        bands[f"[{lo},{lo + 10})M"] = {"directed_solves": int(len(tot)), "capped": int(len(sel))}
    summary = {
        "directed_solves": int(len(df)), "pairs": len(pairs), "classification_counts": cls,
        "capped_directed": int(len(capped)), "capped_pairs": len(pairs_affected),
        "non_reproduced_directed": nonrep, "bit_equal_other_code_directed": other,
        "exact_count_valid": nonrep == 0 and other == 0 and high_not_conv == 0 and neg_delta == 0,
        "capped_high_solve_not_converged": high_not_conv,
        "capped_nonpositive_delta": neg_delta,
        "delta_stored_minus_high_over_capped": {
            "max": float(capped["delta"].max()), "min": float(capped["delta"].min()),
            "quantiles": {str(q): float(capped["delta"].quantile(q)) for q in (0.5, 0.9, 0.99)}}
        if len(capped) else None,
        "min_cells_capped": int(capped["cells"].min()) if len(capped) else None,
        "capped_by_cell_band": bands,
        "cpu_seconds_prod": float(df["seconds_prod"].astype(float).sum()),
        "cpu_seconds_high": float(pd.to_numeric(df["seconds_high"], errors="coerce").fillna(0).sum()),
        "warnings_seen": sorted({w for ws in df["warnings"] for w in str(ws).split("|") if w}),
        "iter_prod": ITER_PROD, "iter_high": a.num_iter, "pairs_sha256": pairs_sha,
        "script_sha256": sha256(Path(__file__)),
        "hosts": sorted({pr["host"] for pr in provs}), "cpus": sorted({pr["cpu"] for pr in provs}),
        "openblas": sorted({json.dumps(pr["openblas"], sort_keys=True) for pr in provs}),
    }
    capped.to_csv(out / "capped_solves.csv", index=False)
    df[df["classification"] != "CONVERGED"].to_csv(out / "non_converged_or_non_reproduced.csv", index=False)

    # --- top-10 impact on the saved ranked-list population only (review F2)
    saved = pd.read_csv(a.ranked_lists, dtype={"query": str, "neighbour": str})
    m = pickle.load(open(a.matrix, "rb"))
    all_ids = [str(i) for i in m.index]
    qset = set(saved["query"])
    ids = [i for i in all_ids if i in qset]
    if len(ids) != len(qset):
        sys.exit("saved queries not all present in matrix")
    sub = m.loc[ids, ids].to_numpy(dtype=np.float64)
    corr = sub.copy()
    pos = {s_: i for i, s_ in enumerate(ids)}
    n_replaced = 0
    for _, r in capped.iterrows():
        if r["query"] in pos and r["other"] in pos and sub[pos[r["query"]], pos[r["other"]]] < MASK_VALUE:
            corr[pos[r["query"]], pos[r["other"]]] = float(r["high_cost"]); n_replaced += 1
    saved_top = saved.sort_values(["query", "rank"]).groupby("query")["neighbour"].apply(list).to_dict()
    rows, mism = [], 0
    for i, q in enumerate(ids):
        t0, t1 = _topk(sub[i], ids), _topk(corr[i], ids)
        if t0 != saved_top[q][:TOPK]:
            mism += 1
        rows.append({"query": q, "same_set": set(t0) == set(t1), "same_order": t0 == t1,
                     "orig_top10": " ".join(t0), "corrected_top10": " ".join(t1)})
    if mism:
        sys.exit(f"reconstructed original top-10 differs from saved ranked list for {mism} queries; no COMPLETE")
    imp = pd.DataFrame(rows)
    imp.to_csv(out / "top10_impact.csv", index=False)
    # per capped direction: where the corrected neighbour sat before/after and its margin to the 10th (review R2)
    moves = []
    for _, r in capped.iterrows():
        if r["query"] not in pos or r["other"] not in pos:
            moves.append({"query": r["query"], "other": r["other"], "in_population": False}); continue
        i = pos[r["query"]]
        full0, full1 = _topk(sub[i], ids, len(ids)), _topk(corr[i], ids, len(ids))
        d10 = sub[i][pos[full0[TOPK - 1]]]
        moves.append({"query": r["query"], "other": r["other"], "in_population": True,
                      "orig_rank": full0.index(r["other"]) + 1, "corrected_rank": full1.index(r["other"]) + 1,
                      "orig_dist": repr(float(sub[i, pos[r["other"]]])), "corrected_dist": repr(float(corr[i, pos[r["other"]]])),
                      "orig_10th_dist": repr(float(d10)), "margin_to_10th": repr(float(sub[i, pos[r["other"]]] - d10))})
    mv = pd.DataFrame(moves)
    mv.to_csv(out / "capped_rank_moves.csv", index=False)
    enters = int(((mv.get("orig_rank", pd.Series(dtype=float)) > TOPK) & (mv.get("corrected_rank", pd.Series(dtype=float)) <= TOPK)).sum()) if len(mv) else 0
    summary["top10_impact"] = {
        "population": len(ids), "saved_top10_reconstructed": len(ids) - mism, "capped_entries_replaced": n_replaced,
        "queries_order_changed": int((~imp["same_order"]).sum()), "queries_set_changed": int((~imp["same_set"]).sum()),
        "capped_directions_in_population": int(mv["in_population"].sum()) if len(mv) else 0,
        "capped_neighbours_entering_top10": enters,
        "min_margin_to_10th_over_capped": (float(mv.loc[mv["in_population"], "margin_to_10th"].astype(float).min())
                                           if len(mv) and mv["in_population"].any() else None),
        "ranked_lists": {"path": a.ranked_lists, "sha256": sha256(Path(a.ranked_lists))},
        "matrix": {"path": a.matrix, "sha256": sha256(Path(a.matrix))}}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    (out / "COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    print(json.dumps(summary, indent=2))


def mode_selfcheck(a):
    rng = np.random.default_rng(0)
    # subsample: nested prefix property and k rule
    idx30 = subsample_idx(1000, "x"); idx15 = subsample_idx(1000, "x", 0.15)
    assert len(idx30) == 300 and len(idx15) == 150 and set(idx15) <= set(idx30)
    assert n_keep(1) == 1 and n_keep(3) == 1 and n_keep(5) == 2
    assert list(subsample_idx(1, "y")) == [0] and len(subsample_idx(2, "y")) == 1
    # kernel: symmetric and reproducible; capped run detected by inequality
    fa, fb = rng.normal(size=(40, 8)).astype(np.float32), rng.normal(size=(30, 8)).astype(np.float32)
    fa /= np.linalg.norm(fa, axis=1, keepdims=True); fb /= np.linalg.norm(fb, axis=1, keepdims=True)
    c1, k1, _, _ = emd_cost(fa, fb, ITER_HIGH); c2, k2, _, _ = emd_cost(fb, fa, ITER_HIGH)
    assert k1 == 1 and k2 == 1 and abs(c1 - c2) < 1e-8 and c1 == emd_cost(fa, fb, ITER_HIGH)[0]  # same-direction exact
    cc, kc, warns, _ = emd_cost(fa, fb, 3)
    assert kc != 1 and any("numItermax" in w for w in warns), (kc, warns)
    # topk: mask excluded, ties broken by id
    row = np.array([0.5, MASK_VALUE, 0.2, 0.2, 0.9]); ids = ["a", "b", "d", "c", "e"]
    assert _topk(row, ids, 3) == ["c", "d", "a"]
    # classification: a capped 1e6-style solve differs from converged
    assert cc != c1 and kc == 3
    # csv hex round trip
    x = 0.6266332594254216; assert float.fromhex(x.hex()) == x
    print("selfcheck OK")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["select", "solve", "summarize", "selfcheck"])
    p.add_argument("--out", default=os.path.join(os.environ.get("PROSPER_OUT", ""), "r2_2_ot30_resolve_20261007"))
    p.add_argument("--feature-dir", default=os.path.join(os.environ.get("PROSPER_OUT", ""), "features", "uni2_all_patches"))
    p.add_argument("--similarity-dir", default=os.path.join(os.environ.get("PROSPER_OUT", ""), "similarity", "uni2apf30_ot"))
    p.add_argument("--matrix", default=os.path.join(os.environ.get("PROSPER_OUT", ""), "similarity_matrix",
                                                      "combined_similarity_matrix_uni2apf30_ot_cosine_no_same_patient.pkl"))
    p.add_argument("--ranked-lists", default=os.path.join(os.environ.get("PROSPER_OUT", ""), "r22_metrics",
                                                            "ranked_lists_uni2apf30_ot_cosine.csv.gz"))
    p.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    p.add_argument("--max-cells", type=int, default=None,
                   help="exclusive upper cell bound for select (second tier: --threshold 40000000 --max-cells 50000000)")
    p.add_argument("--expect-threads", type=int, default=EXPECT_THREADS,
                   help="required CPU-affinity size and OpenBLAS thread count (production used 8); 0 disables the check")
    p.add_argument("--slide-list", default=None, help="optional file restricting slide ids (tests)")
    p.add_argument("--num-iter", type=int, default=ITER_HIGH)
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--n-tasks", type=int, default=1)
    p.add_argument("--cache-size", type=int, default=24)
    a = p.parse_args(argv)
    {"select": mode_select, "solve": mode_solve, "summarize": mode_summarize, "selfcheck": mode_selfcheck}[a.mode](a)


if __name__ == "__main__":
    main()
