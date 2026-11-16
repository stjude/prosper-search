"""How much of a reported metric is an artifact of breaking distance ties?

Found 2026-08-27 while auditing the C2 voting sweep. `filter_similarity_matrix`
built the reference-library order with `list(<set of str>)`, whose iteration
order depends on PYTHONHASHSEED, and `_rank_neighbors` sorts with a *stable*
mergesort -- so equal distances were broken by array position and the position
changed every run. Two runs of identical code on an identical matrix disagreed
by up to 0.011 macro F1.

That is fixed (the library now takes the matrix's own row order), but fixing it
only makes the tie-break *deterministic*, not *correct*: when several reference
slides sit at exactly the same distance, which one is called "the" nearest
neighbour is genuinely undefined. The honest thing is to report the metric under
the declared order and, separately, the interval the metric spans across
tie-break orders. This module measures that interval.

Ties are not a rounding artifact here -- they are structural:

  * Yottixel/BoB distances are quantised to ~1e3 distinct values over ~3.4e6
    finite cells, giving 4.0-4.4% of queries a tied rank-1 neighbour.
  * TITAN: 0.9%.
  * The embedding branches (ot_cosine, median_pooling, sampler) have exactly
    ZERO rank-1 ties, so for them this analysis returns a degenerate interval of
    width 0 -- which is itself the result worth reporting for the declared
    primary configuration.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import paths
from .cli import PUBLISHED as P
from .slide_eval import (compute_metrics_from_predictions,
                         filter_similarity_matrix, load_metadata,
                         load_similarity_matrix, predict_labels_from_matrix)


def tie_stats(sim_f: pd.DataFrame) -> Dict:
    """Prevalence of tied neighbours at each depth, on the filtered matrix."""
    M = sim_f.to_numpy().astype(float).copy()
    np.fill_diagonal(M, np.inf)
    M = np.where(np.isnan(M), np.inf, M)
    n = M.shape[0]
    out: Dict[str, object] = {"n_queries": n}
    mins = M.min(axis=1)
    out["queries_with_tied_rank1"] = int(((M == mins[:, None]).sum(axis=1) > 1).sum())
    # A tie that straddles the top-k boundary is what can change a top-k set.
    for k in (3, 5, 10):
        part = np.partition(M, k - 1, axis=1)
        kth = part[:, k - 1]
        # more cells equal to the k-th value than the number of slots it fills
        n_eq = (M == kth[:, None]).sum(axis=1)
        n_at_or_before = (M < kth[:, None]).sum(axis=1)
        straddle = (n_at_or_before + n_eq) > k
        out[f"queries_with_tie_straddling_top{k}"] = int(straddle.sum())
    finite = M[np.isfinite(M) & (M < 1e6)]
    out["n_distinct_finite_distances"] = int(np.unique(finite).size)
    out["n_finite_cells"] = int(finite.size)
    return out


def run(model: str, method: str, topk: int, tie_break: str, tumor_only: bool,
        n_orders: int, seed: int, drop_labels: Optional[List[str]] = None) -> Dict:
    sim = load_similarity_matrix(str(paths.similarity_matrix(
        model, method, tumor_only=tumor_only, no_same_patient=True)))
    if drop_labels is None:
        drop_labels = ['MISC', 'NORMAL'] if tumor_only else ['MISC']
    meta = load_metadata(str(paths.slide_metadata()), label_col=P["label_col"],
                         id_col=P["id_col"], patient_col=P["patient_col"],
                         drop_labels=list(drop_labels), min_class_count=None)
    sim_f, meta_f = filter_similarity_matrix(sim, meta, id_col=P["id_col"])
    labels_order = sorted(meta_f[P["label_col"]].unique().tolist())

    def macro(s: pd.DataFrame, m: pd.DataFrame) -> float:
        dp = predict_labels_from_matrix(
            sim=s, meta=m, topk=topk, id_col=P["id_col"],
            label_col=P["label_col"], patient_col=P["patient_col"],
            larger_is_better=P["larger_is_better"], tie_break=tie_break,
            exclude_same_patient=False)
        return float(compute_metrics_from_predictions(
            dp, labels_order=labels_order)["macro_f1"])

    declared = macro(sim_f, meta_f)

    ids = list(sim_f.index)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_orders):
        order = [ids[j] for j in rng.permutation(len(ids))]
        vals.append(macro(sim_f.loc[order, order], meta_f))
    v = np.asarray(vals, dtype=float)

    return {
        "config": {
            "model": model, "method": method, "top_k": topk,
            "tie_break": tie_break, "tumor_only": tumor_only,
            "dropped_labels": list(drop_labels),
            "n_tie_orders": n_orders, "seed": seed,
            "input_similarity_matrix": str(paths.similarity_matrix(
                model, method, tumor_only=tumor_only, no_same_patient=True)),
            "lsf_job": os.environ.get("LSB_JOBID"),
        },
        "tie_prevalence": tie_stats(sim_f),
        "declared_order_macro_f1": declared,
        "tie_order_macro_f1": {
            "n": int(v.size),
            "min": float(v.min()), "max": float(v.max()),
            "range": float(v.max() - v.min()),
            "mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "p2.5": float(np.percentile(v, 2.5)),
            "p97.5": float(np.percentile(v, 97.5)),
        },
        # The number a reader should compare against a claimed difference: if a
        # reported gap between two configurations is smaller than this, the gap
        # is not resolvable by this evaluation.
        #
        # This is the 95% interval width, NOT max - min. `range` is an
        # extreme-value statistic: it grows without bound as `n_orders` rises
        # and converges to nothing, so quoting it would make the reported floor
        # an artifact of how many orders happened to be sampled. Measured on
        # this grid, median range/std = 5.13 against E[range/std] ~ 5.5 for a
        # normal sample of n=200 -- i.e. `range` was tracking n, not the
        # configuration. The 2.5-97.5 percentile width converges, so it is the
        # quantity that can be compared against a claimed gap. `range` is still
        # reported above, as a diagnostic only.
        "resolvable_difference_floor": float(
            np.percentile(v, 97.5) - np.percentile(v, 2.5)),
        "resolvable_difference_floor_statistic": "p97.5 - p2.5 over tie orders",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--tie-break", default="generous_majority")
    p.add_argument("--tumor_only", action="store_true")
    p.add_argument("--n-orders", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--task", default="c2_tie_sensitivity")
    a = p.parse_args(argv)
    res = run(a.model, a.method, a.topk, a.tie_break, a.tumor_only,
              a.n_orders, a.seed)
    name = (f"tie_{a.model}_{a.method}_top{a.topk}_{a.tie_break}"
            + ("_tumor_only" if a.tumor_only else ""))
    out = paths.result_json(a.task, name)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"[tie] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
