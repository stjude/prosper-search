"""Paired bootstrap comparisons between configurations (task C5 / R3.1, E4).

R3.1 asked for "the incremental contribution of each module". Answering that
needs *paired* intervals, and the distinction is not pedantic: the 33
configurations are evaluated on the **same 2,778 query slides**, so their errors
are strongly correlated. Independent per-configuration CIs -- the kind C2
reports, and the kind the submitted manuscript reports -- overlap heavily among
the top six configurations and therefore cannot separate them. The paired
interval on the *difference* is much tighter, because the shared
query-difficulty variance cancels.

Concretely, from the C2 grid: virchow2+yottixel hit@5 = 0.8976 [0.879, 0.915]
and uni2+ot_cosine = 0.8840 [0.864, 0.903]. Those intervals overlap almost
entirely, which invites the conclusion that the configurations are
indistinguishable. That conclusion does not follow from overlapping marginal
intervals, and asserting it would be as wrong as asserting the opposite.

Method. For configurations A and B, per-query metric values are computed once
from the persisted ranked lists. Each bootstrap replicate draws ONE
patient-clustered PPS sample of queries and evaluates *both* configurations on
that same sample, so the replicate difference retains the pairing. The reported
interval is the percentile interval of ``macro(A) - macro(B)``.

The reference is the protocol's declared primary configuration, so every Δ reads
as "this rung against PROSPER as declared" rather than against a moving target.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import metrics, paths, resample
from .cli import PUBLISHED, PUBLISHED_DROP_LABELS, _git_commit
from .slide_eval import load_metadata


def load_ranked(path: Path, depth: int = 10):
    """Ranked lists -> (query ids, true labels, neighbour-label matrix)."""
    df = (pd.read_parquet(path) if str(path).endswith(".parquet")
          else pd.read_csv(path))
    df = df.sort_values(["query", "rank"], kind="mergesort")
    queries = df["query"].astype(str).drop_duplicates().tolist()
    true = df.drop_duplicates("query").set_index(
        df.drop_duplicates("query")["query"].astype(str))["true_label"].astype(str)
    n = len(queries)
    labels = df["neighbour_label"].astype(str).to_numpy().reshape(n, -1)
    if labels.shape[1] < depth:
        raise ValueError(f"{path} has depth {labels.shape[1]} < {depth}")
    return queries, [true[q] for q in queries], labels[:, :depth]


def per_query(labels, true_labels, n_relevant, k: int) -> Dict[str, np.ndarray]:
    rel = metrics._relevance(labels, np.asarray(list(true_labels), dtype=object))
    return {
        f"hit_at_{k}": metrics.hit_at_k(rel, k),
        f"precision_at_{k}": metrics.precision_at_k(rel, k),
        f"ndcg_at_{k}": metrics.ndcg_at_k(rel, k, n_relevant),
        "mrr": metrics.reciprocal_rank(rel),
    }


def _macro(vals: np.ndarray, rows: np.ndarray, row_labels: np.ndarray,
           classes: List[str]) -> float:
    per_class = []
    for c in classes:
        sel = rows[row_labels == c]
        if sel.size:
            per_class.append(float(np.mean(vals[sel])))
    return float(np.mean(per_class)) if per_class else np.nan


def make_replicates(query_ids: List[str], true_labels: List[str],
                    meta: pd.DataFrame, *, B: int, seed: int, id_col: str,
                    label_col: str, patient_col: str, mode: str):
    """Draw the B patient-clustered query samples ONCE, as row indices.

    Two reasons this is precomputed rather than drawn inside each comparison.
    The cheap one: 33 configurations x B=5000 would be 165,000 draws instead of
    5,000. The important one: reusing the same replicates across every
    configuration is the method of **common random numbers**, so each Δ is
    measured against the reference on identical resamples and the Δs are
    mutually comparable. Drawing fresh samples per comparison would add
    independent sampling noise to each one and make "A beats B beats C"
    unorderable.
    """
    idx_of = {q: i for i, q in enumerate(query_ids)}
    tl = np.asarray([str(t) for t in true_labels], dtype=object)
    classes = sorted(set(tl.tolist()))
    index = resample.build_patient_index(meta, id_col, label_col, patient_col)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(B):
        sampled = resample.sample_queries(index, classes, rng, mode=mode)
        rows = np.array([idx_of[s] for s in sampled if s in idx_of], dtype=int)
        reps.append((rows, tl[rows] if rows.size else np.array([], dtype=object)))
    return reps, tl, classes


def macro_draws(vec: Dict[str, np.ndarray], reps, classes) -> Dict[str, np.ndarray]:
    """Macro value of each metric on every replicate, for one configuration."""
    out = {}
    for name, vals in vec.items():
        d = np.empty(len(reps), dtype=float)
        for b, (rows, rl) in enumerate(reps):
            d[b] = _macro(vals, rows, rl, classes) if rows.size else np.nan
        out[name] = d
    return out


def paired_ci(draws_a: Dict[str, np.ndarray], draws_b: Dict[str, np.ndarray],
              vec_a: Dict[str, np.ndarray], vec_b: Dict[str, np.ndarray],
              tl: np.ndarray, classes: List[str],
              alpha: float = 0.05) -> Dict[str, dict]:
    """Percentile CIs for macro(A) - macro(B) from paired replicate draws."""
    all_rows = np.arange(tl.shape[0])
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {}
    for name in (n for n in draws_a if n in draws_b):
        diff = draws_a[name] - draws_b[name]
        d = diff[~np.isnan(diff)]
        point = (_macro(vec_a[name], all_rows, tl, classes)
                 - _macro(vec_b[name], all_rows, tl, classes))
        lo = float(np.percentile(d, lo_q)) if d.size else None
        hi = float(np.percentile(d, hi_q)) if d.size else None
        # Two-sided percentile-bootstrap p-value (smallest alpha at which the
        # percentile interval excludes zero), floored at 1/B so it can be fed
        # to a Holm step across the comparison family.
        p2 = (float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))
              if d.size else None)
        if p2 is not None:
            p2 = max(p2, 1.0 / d.size)
        out[f"macro_{name}"] = {
            "delta": point, "lo": lo, "hi": hi,
            # A paired interval excluding zero is the claim R3.1 wants. An
            # interval CONTAINING zero is not evidence of equivalence, only
            # absence of evidence of a difference -- named explicitly so the
            # JSON cannot be misread as the former.
            "excludes_zero": bool(lo is not None and hi is not None
                                  and (lo > 0 or hi < 0)),
            "p_two_sided": p2,
            "B_effective": int(d.size),
        }
    return out


def holm_adjust(comparisons: Dict[str, dict], alpha: float = 0.05) -> None:
    """Holm step-down over the whole comparison family, per metric, in place.

    The family is every configuration paired against the reference for one
    metric (44 at k=5). Adds ``p_holm`` and ``holm_reject`` next to each
    ``p_two_sided``; the uncorrected interval is left untouched so the JSON
    reports both readings, as the plan's multiplicity sentence requires.
    """
    metric_names = sorted({m for v in comparisons.values() if isinstance(v, dict)
                           for m, r in v.items()
                           if isinstance(r, dict) and r.get("p_two_sided") is not None})
    for m in metric_names:
        items = [(name, v[m]["p_two_sided"]) for name, v in comparisons.items()
                 if isinstance(v, dict) and isinstance(v.get(m), dict)
                 and v[m].get("p_two_sided") is not None]
        n = len(items)
        order = sorted(range(n), key=lambda i: items[i][1])
        running = 0.0
        for rank, i in enumerate(order):
            adj = min(1.0, (n - rank) * items[i][1])
            running = max(running, adj)          # enforce monotonicity
            name = items[i][0]
            comparisons[name][m]["p_holm"] = running
            comparisons[name][m]["holm_reject"] = bool(running < alpha)
            comparisons[name][m]["holm_family_size"] = n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Paired bootstrap Δ between configurations (C5 / R3.1).")
    p.add_argument("--ranked-dir", default=None,
                   help="directory of C2 ranked-list files "
                        "(default $PROSPER_OUT/c2_metric_suite)")
    p.add_argument("--reference", default="uni2_ot_cosine",
                   help="baseline configuration, as <model>_<method>; default is "
                        "the protocol's declared primary")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--iter", type=int, default=PUBLISHED["bootstrap_B"],
                   dest="bootstrap_B")
    p.add_argument("--seed", type=int, default=PUBLISHED["seed"])
    p.add_argument("--bootstrap-mode", default=PUBLISHED["bootstrap_mode"])
    p.add_argument("--tumor_only", action="store_true")
    p.add_argument("--task", default="c5_paired")
    return p


def run(args) -> dict:
    ranked_dir = Path(args.ranked_dir) if args.ranked_dir else \
        paths.result_json("c2_metric_suite", "x").parent
    files = sorted(glob.glob(str(ranked_dir / "ranked_lists_*")))
    if not files:
        raise SystemExit(f"no ranked lists in {ranked_dir}; run C2 first")

    def tag(f: str) -> str:
        return (Path(f).name.split("ranked_lists_", 1)[1]
                .replace(".parquet", "").replace(".csv.gz", ""))

    ref_file = next((f for f in files if tag(f) == args.reference), None)
    if ref_file is None:
        raise SystemExit(f"reference {args.reference!r} not among "
                         f"{[tag(f) for f in files]}")

    meta_path = paths.require(paths.slide_metadata(), "slide metadata")
    meta = load_metadata(str(meta_path), label_col=PUBLISHED["label_col"],
                         id_col=PUBLISHED["id_col"],
                         patient_col=PUBLISHED["patient_col"],
                         drop_labels=list(PUBLISHED_DROP_LABELS[bool(args.tumor_only)]),
                         min_class_count=None)

    q_ref, t_ref, lab_ref = load_ranked(Path(ref_file), args.depth)
    meta = meta[meta[PUBLISHED["id_col"]].astype(str).isin(set(q_ref))]

    # Per-query relevant counts: same class, different patient.
    mp = dict(zip(meta[PUBLISHED["id_col"]].astype(str),
                  meta[PUBLISHED["patient_col"]].astype(str)))
    ml = dict(zip(meta[PUBLISHED["id_col"]].astype(str),
                  meta[PUBLISHED["label_col"]].astype(str)))
    lab_arr = np.array([ml.get(q, "?") for q in mp], dtype=object)
    pat_arr = np.array(list(mp.values()), dtype=object)
    n_rel_lookup = {}
    for q in q_ref:
        n_rel_lookup[q] = int(np.sum((lab_arr == ml.get(q)) & (pat_arr != mp.get(q))))
    n_relevant = np.array([n_rel_lookup[q] for q in q_ref], dtype=float)

    vec_ref = per_query(lab_ref, t_ref, n_relevant, args.k)
    reps, tl, classes = make_replicates(
        q_ref, t_ref, meta, B=args.bootstrap_B, seed=args.seed,
        id_col=PUBLISHED["id_col"], label_col=PUBLISHED["label_col"],
        patient_col=PUBLISHED["patient_col"], mode=args.bootstrap_mode)
    draws_ref = macro_draws(vec_ref, reps, classes)

    results = {"config": {
        "reference": args.reference, "k": args.k, "depth": args.depth,
        "bootstrap_B": args.bootstrap_B, "seed": args.seed,
        "bootstrap_mode": args.bootstrap_mode,
        "pairing": "same patient-clustered PPS query sample evaluated on both "
                   "configurations per replicate",
        "n_queries": len(q_ref), "prosper_commit": _git_commit(),
        "common_random_numbers": True,
    }, "comparisons": {}}

    for f in files:
        name = tag(f)
        if name == args.reference:
            continue
        q, t, lab = load_ranked(Path(f), args.depth)
        if q != q_ref:
            results["comparisons"][name] = {"error": "query set differs from reference"}
            continue
        vec = per_query(lab, t, n_relevant, args.k)
        results["comparisons"][name] = paired_ci(
            macro_draws(vec, reps, classes), draws_ref, vec, vec_ref, tl, classes)
        d = results["comparisons"][name][f"macro_hit_at_{args.k}"]
        print(f"[c5] {name:<28} Δ hit@{args.k} = {d['delta']:+.4f} "
              f"[{d['lo']:+.4f},{d['hi']:+.4f}]"
              f"{'  *' if d['excludes_zero'] else ''}")

    holm_adjust(results["comparisons"])
    results["config"]["multiplicity"] = (
        "p_two_sided = percentile-bootstrap two-sided p (floor 1/B); p_holm = Holm "
        "step-down over all comparisons against the reference for that metric; "
        "intervals are uncorrected")
    out = paths.result_json(args.task, f"paired_vs_{args.reference}_k{args.k}")
    out.write_text(json.dumps(results, indent=2))
    print(f"[c5] wrote {out}")
    hk = f"macro_hit_at_{args.k}"
    comps = [v for v in results["comparisons"].values() if isinstance(v, dict)]
    print(f"[c5] '*' marks a paired interval excluding zero "
          f"({sum(1 for v in comps if v.get(hk, {}).get('excludes_zero'))} of "
          f"{len(comps)}); Holm-significant on hit@{args.k}: "
          f"{sum(1 for v in comps if v.get(hk, {}).get('holm_reject'))}")
    return results


def main(argv=None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
