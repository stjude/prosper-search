"""C5 -- the five-rung module ablation ladder (R3.1, R2.1, E4).

R3.1 asked for "the incremental contribution of **each module**". An encoder x
aggregation grid does not answer that question because every cell in it has
every module switched on. This module answers it with a ladder where exactly
one module is added between consecutive rungs, and reports a **paired** delta
and CI for each transition, reusing ``prosper.evaluation.paired``'s replicate
machinery (common random numbers across the two configurations being compared)
rather than a new bootstrap.

The five rungs named in the revision plan ([internal record], task C5):

    1. raw tile features + mean pooling
    2. + mosaic selection (the 15% tile subsample)
    3. + tumour-only tile filtering
    4. + OT aggregation
    5. + Bayesian scoring

Runnability, verified against the artifact inventory on 2026-08-27 (not
assumed):

  * **Rung 1 -> 2 is BLOCKED.** Isolating the mosaic-selection effect needs
    features over ALL tiles, which exist only as coordinate lists
    (``comet-annotation/output/all_patches/``, ~41M tiles / 4,724 slides), not
    as extracted features for any of the 8 benchmark encoders. Every existing
    similarity matrix (median_pooling, ot_cosine, sampler, yottixel) is already
    computed over the 15%-mosaic subsample -- there is no "rung 1" matrix to
    diff it against. This is task C7's GPU item, not this one. Do not subsample
    the mosaic to fake it: the mosaic and the all-patches universe are only
    *nearly* nested (measured tile-id overlap 96.7% / 98.2% / 66.1% on three
    probe slides), so a subsample of the mosaic is not "all tiles" and a diff
    against it would not isolate the mosaic-selection effect.

  * **Rung 2 -> 3 is runnable, with a matrix-availability correction.** The
    task brief that motivated this module named ``median_pooling`` as the
    shared aggregation to hold fixed across this transition. That matrix does
    not exist: no encoder (checked all 8 under
    ``comet-<model>/output/similarity_matrix/``) has a
    ``*_median_pooling_tumor_only_no_same_patient.pkl``. Only ``ot_cosine`` and
    ``yottixel`` have a same-patient-masked tumour-only variant, and only for
    uni2 and virchow2. This transition is therefore run holding **ot_cosine**
    fixed (uni2), not median_pooling. See ``rung_2_to_3``.

  * **Rung 3 -> 4 is runnable, but not on the tumour-only population as
    literally specified.** The OT-aggregation increment (median_pooling ->
    ot_cosine) needs both matrices on the SAME population. Both exist for
    uni2 on **all-tissue** (n=2,778); neither the tumour-only nor any other
    population has a same-patient-masked median_pooling matrix (see above), so
    this transition is measured on the all-tissue population instead of after
    rung 3's tumour-only filtering. See ``rung_3_to_4``.

  * **Rung 4 -> 5 is runnable.** ``prosper.bayesian.disease_probs`` computes a
    posterior over disease classes from the same distance matrix; it has no
    notion of a top-5 *neighbour-slide* ranking, so ``macro hit@5`` does not
    apply to it directly (top-5-of-14-classes would be a different, much
    easier task, not a comparable one). The natural, already-precedented
    metric is single-label **accuracy** (Supplementary Tables S11/S13 report
    per-slide posteriors that reduce to an argmax prediction): rung 4's
    prediction is a plain majority vote over the top-5 OT-cosine neighbours,
    rung 5's is the Bayesian argmax at N=5 neighbours over the same
    distances. See ``rung_4_to_5`` and the docstring there for why
    "generous_majority" (the manuscript's published tie-break) is
    deliberately NOT used as rung 4's prediction rule here.

Population-mismatch handling (rung 2 -> 3). The all-tissue and tumour-only
arms have different query populations (2,778 vs 2,604), so a naive delta is
not paired and is partly a change of denominator. This module restricts BOTH
arms to the **intersection of queries** -- which, because the tumour-only
population is constructed by dropping exactly the 174 NORMAL slides from the
all-tissue population, equals the tumour-only population itself. Each arm
keeps its own candidate library (all-tissue's still contains NORMAL slides as
candidates; tumour-only's does not) -- only the query side is matched, because
matching candidates too would erase the very manipulation rung 3 is testing.
hit@k and precision@k do not depend on the recall-style ``n_relevant``
denominator, so this restriction does not need to reconcile the two arms'
differing candidate-library sizes for the primary metrics; ndcg/mrr are
reported too but inherit the tumour-only arm's n_relevant definition, noted in
the output.

Interpretation. A paired CI excluding zero supports "this module contributes".
A CI containing zero is NOT evidence the module does nothing -- it is absence
of evidence of a difference. Callers of this module's output must not collapse
that distinction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import paths
from .cli import PUBLISHED, PUBLISHED_DROP_LABELS, _git_commit
from .paired import load_ranked, macro_draws, make_replicates, paired_ci, per_query
from .slide_eval import _choose_label_from_topk, load_metadata, load_similarity_matrix

K_DEFAULT = 5
B_DEFAULT = PUBLISHED["bootstrap_B"]
SEED_DEFAULT = PUBLISHED["seed"]
MODE_DEFAULT = PUBLISHED["bootstrap_mode"]


def _ranked_path(ranked_dir: Path, model: str, method: str, tumor_only: bool) -> Path:
    tag = f"{model}_{method}{'_tumor_only' if tumor_only else ''}"
    for ext in (".parquet", ".csv.gz"):
        p = ranked_dir / f"ranked_lists_{tag}{ext}"
        if p.exists():
            return p
    raise SystemExit(f"no ranked list for {tag} under {ranked_dir}")


def _meta_for(tumor_only: bool) -> pd.DataFrame:
    meta_path = paths.require(paths.slide_metadata(), "slide metadata")
    return load_metadata(str(meta_path), label_col=PUBLISHED["label_col"],
                         id_col=PUBLISHED["id_col"], patient_col=PUBLISHED["patient_col"],
                         drop_labels=list(PUBLISHED_DROP_LABELS[bool(tumor_only)]),
                         min_class_count=None)


def _n_relevant_lookup(meta: pd.DataFrame, query_ids: List[str]) -> np.ndarray:
    mp = dict(zip(meta[PUBLISHED["id_col"]].astype(str), meta[PUBLISHED["patient_col"]].astype(str)))
    ml = dict(zip(meta[PUBLISHED["id_col"]].astype(str), meta[PUBLISHED["label_col"]].astype(str)))
    lab_arr = np.array([ml.get(q) for q in mp], dtype=object)
    pat_arr = np.array(list(mp.values()), dtype=object)
    out = {}
    for q in query_ids:
        out[q] = int(np.sum((lab_arr == ml.get(q)) & (pat_arr != mp.get(q))))
    return np.array([out[q] for q in query_ids], dtype=float)


def _config(task: str, extra: dict) -> dict:
    cfg = {
        "prosper_commit": _git_commit(),
        "protocol": "configs/config.paper.yaml @ 1.0-frozen-20260827",
        "bootstrap_B": B_DEFAULT, "seed": SEED_DEFAULT, "bootstrap_mode": MODE_DEFAULT,
        "pairing": "same patient-clustered PPS query sample evaluated on both "
                   "rungs per replicate (common random numbers)",
    }
    cfg.update(extra)
    cfg["task"] = task
    return cfg


# --------------------------------------------------------------------------
# Rung 2 -> 3: + tumour-only tile filtering (population-matched via intersection)
# --------------------------------------------------------------------------

def rung_2_to_3(ranked_dir: Path, *, model: str = "uni2", method: str = "ot_cosine",
                k: int = K_DEFAULT, depth: int = 10, B: int = B_DEFAULT,
                seed: int = SEED_DEFAULT, mode: str = MODE_DEFAULT) -> dict:
    late_path = _ranked_path(ranked_dir, model, method, tumor_only=True)   # rung 3
    early_path = _ranked_path(ranked_dir, model, method, tumor_only=False)  # rung 2

    q3, t3, lab3 = load_ranked(late_path, depth)
    q2, t2, lab2 = load_ranked(early_path, depth)

    missing = [q for q in q3 if q not in set(q2)]
    if missing:
        raise SystemExit(f"rung_2_to_3: {len(missing)} tumour-only queries are "
                         f"absent from the all-tissue ranked list -- the subset "
                         f"assumption (tumour-only = all-tissue minus NORMAL) "
                         f"does not hold; e.g. {missing[:5]}")

    meta3 = _meta_for(tumor_only=True)
    meta3 = meta3[meta3[PUBLISHED["id_col"]].astype(str).isin(set(q3))]
    n_relevant3 = _n_relevant_lookup(meta3, q3)

    idx2 = {q: i for i, q in enumerate(q2)}
    order = [idx2[q] for q in q3]
    lab2_sub = lab2[order]
    t2_sub = [t2[i] for i in order]
    mismatches = sum(1 for a, b in zip(t2_sub, t3) if a != b)
    if mismatches:
        raise SystemExit(f"rung_2_to_3: {mismatches} true-label mismatches between "
                         f"the all-tissue and tumour-only arms for the same query id "
                         f"-- metadata is not shared as assumed")

    vec3 = per_query(lab3, t3, n_relevant3, k)          # rung 3: tumour-only
    vec2 = per_query(lab2_sub, t2_sub, n_relevant3, k)  # rung 2, restricted to q3

    reps, tl, classes = make_replicates(
        q3, t3, meta3, B=B, seed=seed, id_col=PUBLISHED["id_col"],
        label_col=PUBLISHED["label_col"], patient_col=PUBLISHED["patient_col"], mode=mode)
    draws3 = macro_draws(vec3, reps, classes)
    draws2 = macro_draws(vec2, reps, classes)
    comparisons = paired_ci(draws3, draws2, vec3, vec2, tl, classes)

    return {
        "config": _config("c5_ladder_2to3", {
            "transition": "rung2(+mosaic, all-tissue) -> rung3(+tumour-only filtering)",
            "held_fixed": f"aggregation={method}, model={model}",
            "deviation_from_brief": "aggregation held fixed at ot_cosine, not "
                "median_pooling -- no encoder has a same-patient-masked "
                "median_pooling tumour-only matrix (verified across all 8 "
                "benchmark encoders on 2026-08-27)",
            "population_handling": "intersection of queries (2,604, == the "
                "tumour-only population, which is a subset of the 2,778 "
                "all-tissue population by construction); each arm keeps its "
                "own candidate library",
            "n_queries_rung2_full": len(q2), "n_queries_rung3_full": len(q3),
            "n_queries_compared": len(q3), "k": k, "depth": depth,
            "rung2_ranked_lists": str(early_path), "rung3_ranked_lists": str(late_path),
        }),
        "comparisons": comparisons,
    }


# --------------------------------------------------------------------------
# Rung 3 -> 4: + OT aggregation (matched population: all-tissue, uni2)
# --------------------------------------------------------------------------

def rung_3_to_4(ranked_dir: Path, *, model: str = "uni2", k: int = K_DEFAULT,
                depth: int = 10, B: int = B_DEFAULT, seed: int = SEED_DEFAULT,
                mode: str = MODE_DEFAULT) -> dict:
    late_path = _ranked_path(ranked_dir, model, "ot_cosine", tumor_only=False)      # rung 4
    early_path = _ranked_path(ranked_dir, model, "median_pooling", tumor_only=False)  # rung 3-proxy

    q4, t4, lab4 = load_ranked(late_path, depth)
    q3, t3, lab3 = load_ranked(early_path, depth)
    if q4 != q3:
        raise SystemExit("rung_3_to_4: median_pooling and ot_cosine ranked lists "
                         "do not share the same query order/population -- expected "
                         "identical construction (same metadata, all-tissue)")

    meta = _meta_for(tumor_only=False)
    meta = meta[meta[PUBLISHED["id_col"]].astype(str).isin(set(q4))]
    n_relevant = _n_relevant_lookup(meta, q4)

    vec4 = per_query(lab4, t4, n_relevant, k)
    vec3 = per_query(lab3, t3, n_relevant, k)

    reps, tl, classes = make_replicates(
        q4, t4, meta, B=B, seed=seed, id_col=PUBLISHED["id_col"],
        label_col=PUBLISHED["label_col"], patient_col=PUBLISHED["patient_col"], mode=mode)
    draws4 = macro_draws(vec4, reps, classes)
    draws3 = macro_draws(vec3, reps, classes)
    comparisons = paired_ci(draws4, draws3, vec4, vec3, tl, classes)

    return {
        "config": _config("c5_ladder_3to4", {
            "transition": "rung3-proxy(median_pooling) -> rung4(+OT aggregation, ot_cosine)",
            "held_fixed": f"population=all-tissue (n={len(q4)}), model={model}",
            "deviation_from_brief": "measured on the all-tissue population, not "
                "after rung 3's tumour-only filtering -- no encoder has a "
                "same-patient-masked median_pooling tumour-only matrix (verified "
                "2026-08-27), so the matched pair (median_pooling, ot_cosine) "
                "only coexists on all-tissue",
            "n_queries_compared": len(q4), "k": k, "depth": depth,
            "rung3proxy_ranked_lists": str(early_path), "rung4_ranked_lists": str(late_path),
        }),
        "comparisons": comparisons,
    }


def _load_ranked_with_dist(path: Path, depth: int):
    """Like ``paired.load_ranked``, but keeps the distance column.

    Needed for the Bayesian rung: the disease-probability kernel needs the
    (n_neighbors+1)-th smallest distance (``d_scale``), not just the labels.
    """
    df = (pd.read_parquet(path) if str(path).endswith(".parquet")
          else pd.read_csv(path))
    df = df.sort_values(["query", "rank"], kind="mergesort")
    queries = df["query"].astype(str).drop_duplicates().tolist()
    true = df.drop_duplicates("query").set_index(
        df.drop_duplicates("query")["query"].astype(str))["true_label"].astype(str)
    n = len(queries)
    labels = df["neighbour_label"].astype(str).to_numpy().reshape(n, -1)[:, :depth]
    dists = df["distance"].to_numpy(dtype=float).reshape(n, -1)[:, :depth]
    return queries, [true[q] for q in queries], labels, dists


def _bayesian_argmax_one(dists_row: np.ndarray, labels_row: np.ndarray,
                         n_neighbors: int, T: float) -> str:
    """One query's Bayesian argmax, from an already rank-sorted top-``depth`` slice.

    Reproduces ``prosper.bayesian.disease_probs._disease_probs_from_neighbors``
    exactly (same d_min-relative kernel, same tau, same equidistant-neighbour
    fallback), but reads ``d_scale`` off the persisted top-depth ranked list
    instead of a freshly sorted full distance row. Valid whenever
    ``depth > n_neighbors`` (10 > 5 here), because the top-depth slice already
    contains the n_neighbors+1 smallest distances in order.
    """
    d = dists_row[:n_neighbors + 1]
    labs = labels_row[:n_neighbors]
    d_min = d[0]
    gap = d[n_neighbors] - d_min
    tau = 0.0 if gap < 1e-10 else np.log(T) / gap
    weights: Dict[str, float] = {}
    for j in range(n_neighbors):
        w = float(np.exp(-tau * (d[j] - d_min)))
        weights[labs[j]] = weights.get(labs[j], 0.0) + w
    return max(weights, key=weights.get)  # dict preserves insertion (=rank) order


def _majority_vote_one(labels_row: np.ndarray, k: int) -> str:
    return _choose_label_from_topk(list(labels_row[:k]), tie_break="majority")


# --------------------------------------------------------------------------
# Rung 4 -> 5: + Bayesian scoring
# --------------------------------------------------------------------------

def rung_4_to_5(ranked_dir: Path, *, model: str = "uni2", method: str = "ot_cosine",
                n_neighbors: int = 5, temperature: float = 20.0,
                majority_k: Optional[int] = None, depth: int = 10,
                B: int = B_DEFAULT, seed: int = SEED_DEFAULT,
                mode: str = MODE_DEFAULT) -> dict:
    """Bayesian argmax (rung 5) vs plain majority vote (rung 4), paired.

    Metric substitution, stated once here rather than silently: hit@5/precision@5
    are top-5-*neighbour-slide* metrics and do not apply to a method whose
    output is a posterior over ~14 disease classes, not a ranking of candidate
    slides. The comparable, already-precedented metric (Supplementary Tables
    S11/S13 reduce to exactly this) is single-label accuracy.

    Rung 4's prediction is deliberately plain **majority** vote, not the
    manuscript's published "generous_majority" tie-break. generous_majority
    sets pred_label = true_label whenever the truth is anywhere in the top-k
    *tied* majority set -- which inflates accuracy by construction independent
    of retrieval quality, and would bias this specific comparison against
    Bayesian scoring for a reason that has nothing to do with Bayesian scoring.
    Plain majority vote (mode of the top-k neighbour labels, ties broken by
    rank) is the honest baseline that isolates what the Bayesian module is
    designed to fix -- see ``tests/test_disease_probs.py``'s "DSRCT case": one
    very near neighbour should be able to outweigh a plurality of distant ones,
    which a plain vote cannot express and a distance-weighted posterior can.
    """
    k = majority_k or n_neighbors
    path = _ranked_path(ranked_dir, model, method, tumor_only=True)
    q, t, lab, dist = _load_ranked_with_dist(path, depth)
    if depth <= n_neighbors:
        raise SystemExit(f"depth={depth} must exceed n_neighbors={n_neighbors} "
                         "for the Bayesian d_scale term")

    meta = _meta_for(tumor_only=True)
    meta = meta[meta[PUBLISHED["id_col"]].astype(str).isin(set(q))]

    bayes_pred = np.array([_bayesian_argmax_one(dist[i], lab[i], n_neighbors, temperature)
                           for i in range(len(q))], dtype=object)
    majority_pred = np.array([_majority_vote_one(lab[i], k) for i in range(len(q))], dtype=object)
    t_arr = np.array(t, dtype=object)
    bayes_correct = (bayes_pred == t_arr).astype(float)
    majority_correct = (majority_pred == t_arr).astype(float)

    reps, tl, classes = make_replicates(
        q, t, meta, B=B, seed=seed, id_col=PUBLISHED["id_col"],
        label_col=PUBLISHED["label_col"], patient_col=PUBLISHED["patient_col"], mode=mode)
    vec5 = {"accuracy": bayes_correct}
    vec4 = {"accuracy": majority_correct}
    draws5 = macro_draws(vec5, reps, classes)
    draws4 = macro_draws(vec4, reps, classes)
    comparisons = paired_ci(draws5, draws4, vec5, vec4, tl, classes)

    return {
        "config": _config("c5_ladder_4to5", {
            "transition": f"rung4(OT-cosine top-{k} plain majority vote) -> "
                          f"rung5(Bayesian argmax, N={n_neighbors}, T={temperature})",
            "metric_substitution": "macro accuracy (single-label argmax), not "
                "macro hit@5/precision@5 -- Bayesian scoring has no top-5 "
                "neighbour-slide ranking to score; see module docstring",
            "rung4_rule": "plain majority vote (mode of top-k neighbour labels, "
                "rank_sum tie-break), NOT the manuscript's published "
                "generous_majority, which inflates accuracy on ties independent "
                "of retrieval quality and would bias this comparison",
            "rung5_rule": "prosper.bayesian.disease_probs argmax "
                f"(n_neighbors={n_neighbors}, T={temperature}), computed from "
                "the same OT-cosine tumour-only distances",
            "n_queries": len(q), "n_neighbors": n_neighbors, "temperature": temperature,
            "majority_k": k, "ranked_lists": str(path),
        }),
        "comparisons": comparisons,
        "point_estimates": {
            "rung4_macro_accuracy": float(np.mean([
                np.mean(majority_correct[tl == c]) for c in classes if np.any(tl == c)])),
            "rung5_macro_accuracy": float(np.mean([
                np.mean(bayes_correct[tl == c]) for c in classes if np.any(tl == c)])),
        },
    }


def s11_s13_fidelity(*, model: str = "uni2", method: str = "ot_cosine",
                     temperature: float = 20.0,
                     s11_path: Optional[Path] = None, s13_path: Optional[Path] = None) -> dict:
    """Best-effort check against Supplementary Tables S11 (N=5) / S13 (N=18).

    This is a math-correctness diagnostic on the Bayesian kernel, not a gate on
    ``rung_4_to_5``: S11/S13 were built over the tumour-only distance matrix's
    full reference set (2,766 rows, NORMAL retained as a candidate class,
    dropping only MISC), which is a *different* candidate library from
    ``rung_4_to_5``'s (NORMAL dropped entirely, matching the rest of the C2/C5
    suite's tumour-only population). A mismatch here says the two analyses use
    different candidate libraries, which is documented, not a bug to chase
    silently. Returns a dict with the deviation regardless of outcome; never
    raises on a mismatch, only on missing inputs (caller decides whether that
    is fatal).
    """
    if s11_path is None or s13_path is None or not Path(s11_path).exists() or not Path(s13_path).exists():
        return {"skipped": True, "reason": "S11/S13 tables not reachable from this host"}

    sim_path = paths.require(
        paths.similarity_matrix(model, method, tumor_only=True, no_same_patient=True),
        "tumour-only similarity matrix")
    sim = load_similarity_matrix(str(sim_path))
    sim.index = sim.index.astype(str)
    sim.columns = sim.columns.astype(str)

    # Real labels, dropping only MISC (keeps NORMAL) -- matches S11/S13's own
    # column list, which includes NORMAL as a candidate class.
    meta_path = paths.require(paths.slide_metadata(), "slide metadata")
    meta_full = load_metadata(str(meta_path), label_col=PUBLISHED["label_col"],
                              id_col=PUBLISHED["id_col"], patient_col=PUBLISHED["patient_col"],
                              drop_labels=["MISC"], min_class_count=None)
    id_to_label = dict(zip(meta_full[PUBLISHED["id_col"]].astype(str),
                          meta_full[PUBLISHED["label_col"]].astype(str)))

    from ..bayesian.disease_probs import compute_disease_probs

    out = {}
    for name, path, n_neighbors in (("S11", s11_path, 5), ("S13", s13_path, 18)):
        tbl = pd.read_excel(path)
        tbl_ids = tbl["slides"].astype(str).str.replace(".svs", "", regex=False)
        class_cols = [c for c in tbl.columns if c not in ("Unnamed: 0", "slides")]
        ref_ids = [i for i in tbl_ids if i in sim.index and i in id_to_label]
        ref_labels = [id_to_label[i] for i in ref_ids]
        sub = sim.loc[ref_ids, ref_ids].to_numpy(dtype=float)
        np.fill_diagonal(sub, 1.0e6)  # exclude self, as every evaluation path does
        _, df = compute_disease_probs(sub, ref_labels, n_neighbors=n_neighbors,
                                       T=temperature, test_slide_ids=ref_ids)
        common_cols = [c for c in class_cols if c in df.columns]
        tbl_sub = tbl.set_index(tbl_ids).loc[ref_ids, common_cols]
        diff = (df[common_cols].to_numpy() - tbl_sub.to_numpy())
        argmax_match = (df[common_cols].to_numpy().argmax(axis=1)
                        == tbl_sub.to_numpy().argmax(axis=1))
        out[name] = {
            "n_rows_compared": len(ref_ids),
            "n_classes_compared": len(common_cols),
            "max_abs_diff": float(np.nanmax(np.abs(diff))),
            "mean_abs_diff": float(np.nanmean(np.abs(diff))),
            "argmax_agreement": float(np.mean(argmax_match)),
            "temperature_tried": temperature,
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="C5 module ablation ladder.")
    p.add_argument("--ranked-dir", default=None)
    p.add_argument("--rung", required=True, choices=["2to3", "3to4", "4to5"])
    p.add_argument("--model", default="uni2")
    p.add_argument("--method", default="ot_cosine")
    p.add_argument("--k", type=int, default=K_DEFAULT)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--iter", type=int, default=B_DEFAULT, dest="B")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--bootstrap-mode", default=MODE_DEFAULT)
    p.add_argument("--n-neighbors", type=int, default=5, help="rung 4->5 only")
    p.add_argument("--temperature", type=float, default=20.0, help="rung 4->5 only")
    p.add_argument("--assert-bayesian-fidelity", action="store_true",
                   help="rung 4->5 only: also check against Supplementary "
                        "Tables S11/S13 (best-effort, never fatal)")
    p.add_argument("--s11-path", default=None)
    p.add_argument("--s13-path", default=None)
    p.add_argument("--task", default="c5_ladder")
    a = p.parse_args(argv)

    ranked_dir = Path(a.ranked_dir) if a.ranked_dir else paths.result_json("c2_metric_suite", "x").parent
    if a.rung == "2to3":
        result = rung_2_to_3(ranked_dir, model=a.model, method=a.method, k=a.k,
                             depth=a.depth, B=a.B, seed=a.seed, mode=a.bootstrap_mode)
        out = paths.result_json(a.task, "rung_2to3")
        headline_metric = f"macro_hit_at_{a.k}"
    elif a.rung == "3to4":
        result = rung_3_to_4(ranked_dir, model=a.model, k=a.k, depth=a.depth,
                             B=a.B, seed=a.seed, mode=a.bootstrap_mode)
        out = paths.result_json(a.task, "rung_3to4")
        headline_metric = f"macro_hit_at_{a.k}"
    else:
        result = rung_4_to_5(ranked_dir, model=a.model, method=a.method,
                             n_neighbors=a.n_neighbors, temperature=a.temperature,
                             depth=a.depth, B=a.B, seed=a.seed, mode=a.bootstrap_mode)
        if a.assert_bayesian_fidelity:
            try:
                result["fidelity_vs_s11_s13"] = s11_s13_fidelity(
                    model=a.model, method=a.method, temperature=a.temperature,
                    s11_path=a.s11_path, s13_path=a.s13_path)
                print(f"[c5-ladder] fidelity vs S11/S13: "
                      f"{json.dumps(result['fidelity_vs_s11_s13'], indent=2)}")
            except Exception as exc:  # best-effort, never fatal to the ladder result
                result["fidelity_vs_s11_s13"] = {"skipped": True, "reason": str(exc)}
                print(f"[c5-ladder] fidelity check failed (non-fatal): {exc}")
        out = paths.result_json(a.task, "rung_4to5")
        headline_metric = "macro_accuracy"

    out.write_text(json.dumps(result, indent=2))
    print(f"[c5-ladder] wrote {out}")
    print(f"[c5-ladder] {result['config']['transition']}")
    d = result["comparisons"][headline_metric]
    print(f"[c5-ladder] Delta {headline_metric} = {d['delta']:+.4f} "
          f"[{d['lo']:+.4f},{d['hi']:+.4f}]{'  *' if d['excludes_zero'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
