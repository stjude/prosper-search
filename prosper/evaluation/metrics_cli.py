"""Standard retrieval metric suite (task C2 -- the editor's second mandate).

R4.b asked why PROSPER reports a bespoke voting F1 instead of the metrics the
retrieval literature reports. This module answers that from the cached distance
matrices: no re-embedding, no GPU.

Three things it does that the published CI pipeline did not:

1. **Persists the ranked neighbour lists.** The published pipeline computed
   ``df_pred`` and threw it away, so every downstream analysis (C6 failure
   modes, C10 calibration) would have had to recompute ranks from scratch. They
   are written once, here.
2. **Uses a per-query relevant-set size.** The slides available to a query
   exclude that query's own patient, so the recall@k denominator is per-query,
   not per-class. See ``metrics.compute_all``.
3. **Puts a patient-clustered interval on every metric**, using the same PPS
   design as the published F1 bootstrap (``resample.sample_queries``), so the
   new numbers and the old ones carry comparable uncertainty.

The reporting rule this module enforces by naming, not by convention: textbook
``recall_at_k`` is bounded by roughly k/|class| and lands around 0.01 here. It is
a true number that reads as failure and says more about class size than about the
system. It is reported with its denominator, and ``hit_at_k`` is reported under
its own name. Presenting either as the other is the same error as the published
``any_hit`` tie-break, which is hit@k shown through an F1 wrapper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, paths, resample
from .cli import PUBLISHED, PUBLISHED_DROP_LABELS, _git_commit
from .slide_eval import filter_similarity_matrix, load_metadata, load_similarity_matrix

SENTINEL = 1.0e6

# A known-answer test for this whole layer, available for free and worth stating
# carefully because the obvious version of it is wrong.
#
# The published `any_hit` tie-break overwrites the prediction with the ground
# truth whenever the truth appears anywhere in the top-k. If the truth is NOT in
# the top-k then no neighbour carries it, so the plurality vote cannot equal the
# truth either. Predictions are therefore correct *exactly* when the truth is in
# the top-k -- which makes accuracy identical to micro hit@k, and for
# single-label multiclass, micro F1 == accuracy.
#
#   => archived any_hit MICRO F1 must equal micro hit@k, exactly.
#
# The macro values do NOT correspond: macro F1 mixes in precision, and the
# incorrect predictions distribute unevenly across classes. Asserting on macro
# would produce a confusing near-miss. Micro is the exact identity.
#
# Read from comet-uni2/confidence_interval/ (tie_break: any_hit) on 2026-08-27,
# all at n_slides = 2778, drop = ['MISC'].
ARCHIVED_ANY_HIT_MICRO_F1 = {1: 0.8358531317494601, 3: 0.9146868250539957,
                             5: 0.937365010799136, 10: 0.9654427645788337}


def ranked_neighbours(sim: pd.DataFrame, meta: pd.DataFrame, depth: int,
                      id_col: str, label_col: str, patient_col: str):
    """Top-``depth`` neighbour ids/labels per query, and per-query relevant counts.

    Sentinel-valued cells (same patient, and self) are dropped rather than
    trusted to sort last. They *do* sort last -- max real distance is 1.0088
    against a 1e6 sentinel -- but "cannot appear in a top-10" is an argument
    about the data, and an argument is a worse guarantee than a filter. The count
    of sentinels that would otherwise have entered a top-k is reported, so the
    claim is checked rather than assumed.
    """
    ids = meta[id_col].astype(str).tolist()
    labels = meta[label_col].astype(str).tolist()
    patients = meta[patient_col].astype(str).tolist()
    id_to_label = dict(zip(ids, labels))
    id_to_patient = dict(zip(ids, patients))

    arr = sim.loc[ids, ids].to_numpy(dtype=float)
    n = arr.shape[0]

    # Per-query relevant set: same class, different patient. Exact denominator.
    lab = np.array(labels, dtype=object)
    pat = np.array(patients, dtype=object)
    n_relevant = np.array([
        int(np.sum((lab == lab[i]) & (pat != pat[i]))) for i in range(n)
    ], dtype=float)

    nbr_ids = np.empty((n, depth), dtype=object)
    nbr_labels = np.empty((n, depth), dtype=object)
    nbr_dists = np.full((n, depth), np.nan, dtype=float)
    sentinel_in_topk = 0
    short_rows = 0

    for i in range(n):
        row = arr[i].copy()
        order = np.argsort(row, kind="mergesort")
        taken = 0
        for j in order:
            if j == i:
                continue
            if row[j] >= SENTINEL:
                sentinel_in_topk += 1 if taken < depth else 0
                continue
            if id_to_patient[ids[j]] == patients[i]:
                continue          # belt and braces: mask may be incomplete
            nbr_ids[i, taken] = ids[j]
            nbr_labels[i, taken] = id_to_label[ids[j]]
            nbr_dists[i, taken] = row[j]
            taken += 1
            if taken == depth:
                break
        if taken < depth:
            short_rows += 1
            for t in range(taken, depth):
                nbr_labels[i, t] = "__NONE__"

    diag = {"n_queries": int(n),
            "sentinels_encountered_before_depth": int(sentinel_in_topk),
            "queries_with_fewer_than_depth_neighbours": int(short_rows)}
    return nbr_ids, nbr_labels, nbr_dists, n_relevant, diag


def per_query_metric_vectors(nbr_labels, true_labels, n_relevant, k_values):
    """Per-query values for every metric, so the bootstrap is an index operation."""
    rel = metrics._relevance(nbr_labels, np.asarray(list(true_labels), dtype=object))
    out = {}
    for k in k_values:
        if k > rel.shape[1]:
            continue
        out[f"hit_at_{k}"] = metrics.hit_at_k(rel, k)
        out[f"precision_at_{k}"] = metrics.precision_at_k(rel, k)
        out[f"recall_at_{k}"] = metrics.recall_at_k(rel, k, n_relevant)
        out[f"ap_at_{k}"] = metrics.average_precision_at_k(rel, k, n_relevant)
        out[f"ndcg_at_{k}"] = metrics.ndcg_at_k(rel, k, n_relevant)
    out["mrr"] = metrics.reciprocal_rank(rel)
    return out


def bootstrap_metric_cis(per_query, query_ids, true_labels, meta, *, B, seed,
                         id_col, label_col, patient_col, mode, alpha=0.05):
    """Patient-clustered percentile CIs for each macro metric.

    Per-query values are computed once and the replicate only re-indexes them,
    which is what makes B=5000 across ~20 metrics cheap. Macro is the mean over
    classes of the mean over that class's sampled queries -- the same definition
    used for the point estimate, so the interval brackets the number reported.
    """
    idx_of = {q: i for i, q in enumerate(query_ids)}
    labels_order = sorted(set(map(str, true_labels)))
    index = resample.build_patient_index(meta, id_col, label_col, patient_col)
    tl = np.asarray([str(t) for t in true_labels], dtype=object)
    class_masks = {c: np.where(tl == c)[0] for c in labels_order}

    rng = np.random.default_rng(seed)
    draws = {name: np.empty(B, dtype=float) for name in per_query}
    for b in range(B):
        sampled = resample.sample_queries(index, labels_order, rng, mode=mode)
        rows = np.array([idx_of[s] for s in sampled if s in idx_of], dtype=int)
        if rows.size == 0:
            for name in per_query:
                draws[name][b] = np.nan
            continue
        rows_by_class = {}
        srl = tl[rows]
        for c in labels_order:
            sel = rows[srl == c]
            if sel.size:
                rows_by_class[c] = sel
        for name, vals in per_query.items():
            per_class = [float(np.mean(vals[sel])) for sel in rows_by_class.values()]
            draws[name][b] = float(np.mean(per_class)) if per_class else np.nan

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    cis = {}
    for name, vals in draws.items():
        v = vals[~np.isnan(vals)]
        point = float(np.mean([np.mean(per_query[name][m])
                               for m in class_masks.values() if m.size]))
        cis[f"macro_{name}"] = {
            "point": point,
            "lo": float(np.percentile(v, lo_q)) if v.size else None,
            "hi": float(np.percentile(v, hi_q)) if v.size else None,
            "B_effective": int(v.size),
        }
    return cis


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standard IR metric suite (C2/R4.b).")
    p.add_argument("--model", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--depth", type=int, default=10,
                   help="retrieval depth; must be >= max(--k-values)")
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--iter", type=int, default=PUBLISHED["bootstrap_B"],
                   dest="bootstrap_B")
    p.add_argument("--seed", type=int, default=PUBLISHED["seed"])
    p.add_argument("--tumor_only", action="store_true")
    p.add_argument("--bootstrap-mode", default=PUBLISHED["bootstrap_mode"],
                   choices=["pps_cluster_one_slide", "one_per_patient"])
    p.add_argument("--drop-labels", nargs="*", default=None)
    p.add_argument("--save-ranked-lists", action="store_true", default=True)
    p.add_argument("--task", default="c2_metric_suite")
    p.add_argument("--out", default=None)
    p.add_argument("--assert-hit-at-k", action="store_true",
                   help="C2 fidelity: require micro hit@k to equal the archived "
                        "any_hit micro F1 (uni2/ot_cosine, all-tissue) exactly")
    return p


def run(args: argparse.Namespace) -> dict:
    if args.drop_labels is None:
        args.drop_labels = list(PUBLISHED_DROP_LABELS[bool(args.tumor_only)])
    if args.depth < max(args.k_values):
        raise SystemExit(f"--depth {args.depth} < max k {max(args.k_values)}")

    sim_path = paths.require(
        paths.similarity_matrix(args.model, args.method,
                                tumor_only=args.tumor_only,
                                no_same_patient=True), "similarity matrix")
    meta_path = paths.require(paths.slide_metadata(), "slide metadata")

    sim = load_similarity_matrix(str(sim_path))
    meta = load_metadata(str(meta_path), label_col=PUBLISHED["label_col"],
                         id_col=PUBLISHED["id_col"],
                         patient_col=PUBLISHED["patient_col"],
                         drop_labels=list(args.drop_labels),
                         min_class_count=None)
    sim_f, meta_f = filter_similarity_matrix(sim, meta, id_col=PUBLISHED["id_col"])

    nbr_ids, nbr_labels, nbr_dists, n_relevant, diag = ranked_neighbours(
        sim_f, meta_f, args.depth, PUBLISHED["id_col"], PUBLISHED["label_col"],
        PUBLISHED["patient_col"])

    query_ids = meta_f[PUBLISHED["id_col"]].astype(str).tolist()
    true_labels = meta_f[PUBLISHED["label_col"]].astype(str).tolist()

    point = metrics.compute_all(nbr_labels, true_labels,
                                k_values=args.k_values, n_relevant=n_relevant)
    per_query = per_query_metric_vectors(nbr_labels, true_labels, n_relevant,
                                         args.k_values)
    cis = bootstrap_metric_cis(
        per_query, query_ids, true_labels, meta_f, B=args.bootstrap_B,
        seed=args.seed, id_col=PUBLISHED["id_col"],
        label_col=PUBLISHED["label_col"], patient_col=PUBLISHED["patient_col"],
        mode=args.bootstrap_mode)

    out_dir = paths.result_json(args.task, "x").parent
    ranked_path = None
    if args.save_ranked_lists:
        ranked_path = out_dir / (f"ranked_lists_{args.model}_{args.method}"
                                 f"{'_tumor_only' if args.tumor_only else ''}.parquet")
        rows = pd.DataFrame({
            "query": np.repeat(query_ids, args.depth),
            "true_label": np.repeat(true_labels, args.depth),
            "rank": np.tile(np.arange(1, args.depth + 1), len(query_ids)),
            "neighbour": nbr_ids.reshape(-1),
            "neighbour_label": nbr_labels.reshape(-1),
            "distance": nbr_dists.reshape(-1),
        })
        try:
            rows.to_parquet(ranked_path, index=False)
        except Exception:
            ranked_path = ranked_path.with_suffix(".csv.gz")
            rows.to_csv(ranked_path, index=False, compression="gzip")

    results = {
        "config": {
            "model": args.model, "method": args.method,
            "depth": args.depth, "k_values": list(args.k_values),
            "bootstrap_B": args.bootstrap_B, "seed": args.seed,
            "bootstrap_mode": args.bootstrap_mode,
            "dropped_labels": list(args.drop_labels),
            "tumor_only": args.tumor_only,
            "recall_denominator": "per-query: same class, different patient",
            "input_similarity_matrix": str(sim_path),
            "input_metadata": str(meta_path),
            "same_patient_masked_input": True,
            "protocol": "configs/config.paper.yaml @ 1.0-frozen-20260827",
            "prosper_commit": _git_commit(),
        },
        "n_slides": int(len(meta_f)),
        "neighbour_diagnostics": diag,
        "point_estimates": point,
        "confidence_intervals": cis,
        "ranked_lists": str(ranked_path) if ranked_path else None,
    }

    out = Path(args.out) if args.out else paths.result_json(
        args.task, f"metrics_{args.model}_{args.method}"
        f"{'_tumor_only' if args.tumor_only else ''}")
    out.write_text(json.dumps(results, indent=2))
    print(f"[c2] wrote {out}")
    for k in args.k_values:
        h = point.get(f"macro_hit_at_{k}")
        pr = point.get(f"macro_precision_at_{k}")
        rc = point.get(f"macro_recall_at_{k}")
        nd = point.get(f"macro_ndcg_at_{k}")
        if h is not None:
            print(f"[c2] k={k:<3} macro hit@k={h:.4f}  precision@k={pr:.4f}  "
                  f"recall@k={rc:.5f}  nDCG@k={nd:.4f}")
    print(f"[c2] macro MRR = {point.get('macro_mrr'):.4f}  n_slides = {len(meta_f)}")
    print(f"[c2] neighbour diagnostics: {diag}")
    return results


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    r = run(args)
    if not args.assert_hit_at_k:
        return 0

    if not (args.model == "uni2" and args.method == "ot_cosine"
            and not args.tumor_only):
        print("[C2-F] --assert-hit-at-k has known answers only for "
              "uni2/ot_cosine on the all-tissue population")
        return 2
    ok = True
    if r["n_slides"] != 2778:
        print(f"[C2-F] n_slides {r['n_slides']} != 2778 -- wrong population")
        ok = False
    for k, expected in ARCHIVED_ANY_HIT_MICRO_F1.items():
        got = r["point_estimates"].get(f"micro_hit_at_{k}")
        if got is None:
            continue
        good = abs(got - expected) < 5e-7
        ok &= good
        print(f"[C2-F] micro hit@{k:<2} : {got:.9f} vs archived any_hit micro F1 "
              f"{expected:.9f}  {'OK' if good else f'MISMATCH ({got-expected:+.2e})'}")
    print("[C2-F] PASS" if ok else
          "[C2-F] FAIL -- the ranked lists differ from the published top-k, so "
          "the new metrics are not computed over the same retrieval. Check "
          "sentinel handling, tie ordering, and the population before sweeping.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
