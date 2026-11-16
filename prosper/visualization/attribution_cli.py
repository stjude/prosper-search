"""ST2.3 phase 1 (spec 0002, [analysis plan]): per-tile OT transport-mass attribution.

For each query in a manifest, recompute exact OT *with the coupling* against its
top-k retrieved neighbours (the production pipeline stores only the scalar
distance) and reduce each coupling to a per-query-tile weighted cost:

    D_A[i] = sum_j T[i, j] * M[i, j]        (ot_visualization, existing)
    per_tile_cost[i] = D_A[i] * n_q         (row mass is uniform 1/n_q, so this
                                             is the mass-weighted mean cosine
                                             distance of tile i's matches, in
                                             the same units as the cost matrix)

A tile that *pulls toward* a neighbour group is one whose per-tile cost against
that group is low relative to the other neighbours. Two rankings are derived:

    delta_os[i]   = mean cost over non-OS nbrs  - mean cost over OS nbrs
    delta_true[i] = mean cost over non-true nbrs - mean cost over true-group nbrs

(positive = tile i matches that group better than the rest of the top-k; NaN
when the top-k list has no neighbour of that group).

Fidelity gate per pair: the recomputed scalar distance must match the ranked
list's stored distance (the C3 production number) to --gate-rtol, and the
coupling's marginals must be uniform to 1e-8 -- a mismatch stops the task.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

OS_LABEL = "OS"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _load_pkl(path: Path):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    feats, coords = obj if isinstance(obj, tuple) else (obj, None)
    return np.asarray(feats), coords


def _one_query(qid: str, true_group: str, nbrs: pd.DataFrame, a) -> dict:
    """Compute all top-k couplings for one query. Runs in a worker process."""
    from ..embeddings.slide_embedding import get_slide_embedding_generator
    from .ot_visualization import compute_per_patch_weighted_distance_scores

    gen = get_slide_embedding_generator("ot", feature_dim=a.feature_dim,
                                        metric=a.metric, reg=0.0, verbose=False)
    qfeats, qcoords = _load_pkl(Path(a.query_features_dir) / f"{qid}.pkl")
    q = gen.process_single_slide(qfeats, qid)["features"]
    n_q = len(q)
    xy = np.asarray([c["wsi_loc"] for c in qcoords], dtype=np.int64)

    scores_f = Path(a.scores_dir) / f"{qid}.csv.gz" if a.scores_dir else None
    if scores_f and scores_f.exists():
        s = pd.read_csv(scores_f)
        if not (np.array_equal(xy[:, 0], s.wsi_loc_x.to_numpy())
                and np.array_equal(xy[:, 1], s.wsi_loc_y.to_numpy())):
            raise SystemExit(f"{qid}: five-class scores not aligned with features")
        pred, p_tumor = s.pred.to_numpy(), s.p_tumor.to_numpy()
    else:
        pred = np.zeros(n_q, dtype=int)
        p_tumor = np.full(n_q, np.nan)

    k = len(nbrs)
    costs = np.full((n_q, k), np.nan, dtype=np.float32)
    best_idx = np.full((n_q, k), -1, dtype=np.int32)
    best_mass = np.zeros((n_q, k), dtype=np.float32)
    gates, t0 = [], time.time()
    for r, row in enumerate(nbrs.itertuples()):
        nfeats, _ = _load_pkl(Path(a.library_features_dir) / f"{row.neighbour}.pkl")
        n = gen.process_single_slide(nfeats, row.neighbour)["features"]
        res = gen.compute_ot_distance(q, n, return_transport=True)
        T, M, dist = res["T"], res["M"], float(res["distance"])
        rel = abs(dist - row.distance) / max(abs(row.distance), 1e-12)
        marg = float(np.abs(T.sum(axis=1) - 1.0 / n_q).max())
        gates.append({"neighbour": row.neighbour, "rank": int(row.rank),
                      "stored": float(row.distance), "recomputed": dist,
                      "rel_diff": rel, "max_marginal_dev": marg})
        if rel > a.gate_rtol or marg > 1e-8:
            raise SystemExit(f"GATE FAILED {qid} vs {row.neighbour}: "
                             f"rel_diff={rel:.3e} marginal_dev={marg:.3e}")
        D_A, _ = compute_per_patch_weighted_distance_scores(T, M)
        costs[:, r] = D_A * n_q
        best_idx[:, r] = T.argmax(axis=1)
        best_mass[:, r] = T[np.arange(n_q), best_idx[:, r]] * n_q

    labels = nbrs.neighbour_label.to_numpy()
    is_os, is_true = labels == OS_LABEL, labels == true_group

    def _mean(mask):
        return costs[:, mask].mean(axis=1) if mask.any() else np.full(n_q, np.nan)

    cost_os, cost_true = _mean(is_os), _mean(is_true)
    cost_non_os, cost_non_true = _mean(~is_os), _mean(~is_true)
    df = pd.DataFrame({"tile_idx": np.arange(n_q), "x": xy[:, 0], "y": xy[:, 1],
                       "pred": pred, "p_tumor": p_tumor,
                       **{f"cost_r{r + 1}": costs[:, r] for r in range(k)},
                       "cost_os": cost_os, "cost_true": cost_true,
                       "delta_os": cost_non_os - cost_os,
                       "delta_true": cost_non_true - cost_true})
    out = Path(a.out_dir)
    df.to_csv(out / f"attr_{qid}.csv", index=False)
    np.savez_compressed(out / f"attr_{qid}.npz", costs=costs, best_idx=best_idx,
                        best_mass=best_mass, xy=xy, pred=pred,
                        neighbours=nbrs.neighbour.to_numpy().astype(str),
                        neighbour_labels=labels.astype(str),
                        distances=nbrs.distance.to_numpy())
    return {"query": qid, "true_group": true_group, "n_tiles": int(n_q),
            "n_os_top": int(is_os.sum()), "n_true_top": int(is_true.sum()),
            "gates": gates, "wall_s": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", required=True,
                   help="CSV with slide_id + tumor_group (manifest subset)")
    p.add_argument("--query-features-dir", required=True)
    p.add_argument("--library-features-dir", required=True)
    p.add_argument("--ranked-lists", required=True,
                   help="C3 ranked_lists_*.csv.gz (query,rank,neighbour,"
                        "neighbour_label,distance) -- also the F-gate reference")
    p.add_argument("--scores-dir", default=None,
                   help="five-class per-tile csv.gz dir (tumor_tiles_cli score)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--feature-dim", type=int, default=1536)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--gate-rtol", type=float, default=1e-6)
    p.add_argument("--num-workers", type=int, default=6,
                   help="process-parallel over queries")
    a = p.parse_args(argv)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(a.queries)
    ranked = pd.read_csv(a.ranked_lists)
    ranked = ranked[ranked["rank"] <= a.top_k]

    jobs = []
    for row in q.itertuples():
        nbrs = ranked[ranked["query"] == row.slide_id].sort_values("rank")
        if len(nbrs) != a.top_k:
            raise SystemExit(f"{row.slide_id}: {len(nbrs)} ranked rows != {a.top_k}")
        jobs.append((row.slide_id, row.tumor_group, nbrs))

    from joblib import Parallel, delayed
    t0 = time.time()
    recs = Parallel(n_jobs=a.num_workers)(
        delayed(_one_query)(qid, tg, nbrs, a) for qid, tg, nbrs in jobs)

    worst = max(g["rel_diff"] for r in recs for g in r["gates"])
    rec = {"config": {k: str(v) for k, v in vars(a).items()},
           "n_queries": len(recs), "worst_gate_rel_diff": worst,
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit(),
           "queries": recs}
    (out / "_record.json").write_text(json.dumps(rec, indent=1))
    pd.DataFrame([{k: r[k] for k in ("query", "true_group", "n_tiles",
                                     "n_os_top", "n_true_top", "wall_s")}
                  for r in recs]).to_csv(out / "summary.csv", index=False)
    print(json.dumps({k: rec[k] for k in ("n_queries", "worst_gate_rel_diff",
                                          "wall_s", "git_commit")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
