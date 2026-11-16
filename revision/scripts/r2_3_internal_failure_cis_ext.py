#!/usr/bin/env python3
"""Whole-patient bootstrap 95% CIs for mAP@5/@10, nDCG@5/@10 (micro and macro) and macro-F1 of the depth-5 rules.

R4.b-vii extension of scripts/r2_3_internal_failure_cis.py (frozen; imported as a module, never edited). The
input is the same accepted Stream A per-query frame; no new retrieval. The frozen module supplies the loader,
make_patient_draws (PCG64, seed 123, B 5000), the patient-multiplicity weight matrix, the percentile interval,
the boundary flag and the record/reconciliation/COMPLETE-FAILED conventions, so every interval here comes from
the SAME patient resamples as the frozen intervals (checked at run time).

Definitions reproduce prosper-search prosper/evaluation/metrics.py (read, not imported) as used for
$PROSPER_OUT/c2_metric_suite/metrics_uni2_ot_cosine.json:
  relevant(rank i)  = neighbour label at rank i == query true label (ranks 1-5 from the frame's top5_labels;
                      ranks 1-10 from the c2 ranked lists, used only after they are verified against the frame)
  AP@k              = sum_{i<=k} rel_i * precision@i / min(k, R), R = library_support (same group, other patient)
  nDCG@k            = sum_{i<=k} rel_i / log2(i + 1)  /  sum_{i <= min(k, R)} 1 / log2(i + 1)   (binary gains)
  micro             = mean over queries; macro = unweighted mean over the diagnostic groups of per-group means
  macro-F1 (rule)   = unweighted mean over groups of 2TP / (2TP + FP + FN) for the rule's predicted label;
                      strict: ABSTAIN is a false negative for the true group and a false positive for no group.
The frame itself has no relevance for ranks 6-10. The @10 metrics therefore take rank 6-10 relevance from the
c2 ranked lists (sha-bound, read-only), after checking that the lists agree with the frame on query set, true
label, top-5 neighbours/labels, hit10, precision10 and first relevant rank; they use the identical patient draws.
"""
import argparse
import datetime
import hashlib
import importlib.util
import io
import json
import os
import platform
import socket
import sys

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_SCRIPT = os.path.join(R, "scripts", "r2_3_internal_failure_cis.py")
ACCEPTED_FROZEN_SCRIPT_SHA256 = "eb89305fce593efc79e7d70e241c04e8405363223f1c1f42512147682b3e5815"


def _load_frozen(path=FROZEN_SCRIPT):
    spec = importlib.util.spec_from_file_location("r2_3_internal_failure_cis_frozen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _load_frozen()
O = F.O
DEFAULT_INPUT = F.DEFAULT_INPUT
DEFAULT_METRICS_JSON = os.path.join(O, "c2_metric_suite", "metrics_uni2_ot_cosine.json")
DEFAULT_RANKED = os.path.join(O, "c2_metric_suite", "ranked_lists_uni2_ot_cosine.csv.gz")
ACCEPTED_METRICS_JSON_SHA256 = "895c8dc2d24f853155452bbf735d868f1e0c8b2a4f1b349ba35510d771bf5363"
ACCEPTED_RANKED_SHA256 = "f2733faa8a44463ed64f483594a2ae048ffbf38a14b39272f124a6abe91ca96b"
METRIC_SOURCE = ("prosper-search prosper/evaluation/metrics.py average_precision_at_k / ndcg_at_k / compute_all "
                 "(read for the definition, not imported; producer of c2_metric_suite metrics_*.json)")
ABSTAIN, TOL, K = F.ABSTAIN, F.TOL, 5
STATED_TOL = 5e-7  # author-stated values are given to 6 decimals
Q_LO, Q_HI = 2.5, 97.5  # 95% level; the reconciliation re-derives intervals with these literals

RULES = {"plurality": "plurality_k5", "generous": "legacy_generous_k5_pred", "strict": "strict_majority_k5"}
SUMMARY = {
    "map5_micro": "mean over queries of AP@5 (normaliser min(5, library_support))",
    "map5_macro": "unweighted mean over groups of the per-group mean AP@5",
    "ndcg5_micro": "mean over queries of nDCG@5 (binary gains, ideal DCG from min(5, library_support))",
    "ndcg5_macro": "unweighted mean over groups of the per-group mean nDCG@5",
    "plurality_macro_f1": "unweighted mean over groups of F1 of plurality_k5 (closest-ranked tied label)",
    "generous_macro_f1": "unweighted mean over groups of F1 of legacy_generous_k5_pred (ORACLE tie rule: the true "
                         "label whenever it is among the tied-max labels)",
    "strict_macro_f1": "unweighted mean over groups of F1 of strict_majority_k5 (>=3/5); ABSTAIN = false negative "
                       "for the true group, false positive for no group",
}
# @10 rows are appended after the @5/F1 rows so the pre-existing rows stay byte-identical
SUMMARY10 = {
    "map10_micro": "mean over queries of AP@10 (normaliser min(10, library_support); ranks 6-10 from c2 ranked lists)",
    "map10_macro": "unweighted mean over groups of the per-group mean AP@10",
    "ndcg10_micro": "mean over queries of nDCG@10 (binary gains, ideal DCG from min(10, library_support); ranks 6-10 "
                    "from c2 ranked lists)",
    "ndcg10_macro": "unweighted mean over groups of the per-group mean nDCG@10",
}
# author-stated point values (R4.b-vii) and the metrics JSON keys they correspond to
STATED = {"map5_micro": 0.748086, "ndcg5_macro": 0.708490, "plurality_macro_f1": 0.767912,
          "generous_macro_f1": 0.791156, "strict_macro_f1": 0.743908}
STATED_MAP10_MICRO = 0.701745
JSON_KEYS = {"map5_micro": "micro_ap_at_5", "map5_macro": "macro_ap_at_5", "ndcg5_micro": "micro_ndcg_at_5",
             "ndcg5_macro": "macro_ndcg_at_5", "map10_micro": "micro_ap_at_10", "map10_macro": "macro_ap_at_10",
             "ndcg10_micro": "micro_ndcg_at_10", "ndcg10_macro": "macro_ndcg_at_10"}
PER_GROUP = ["plurality_f1", "generous_f1", "strict_f1", "ndcg5", "ap5"]
PER_GROUP10 = ["ap10", "ndcg10"]
RESULT_FILES = ("summary_ext.csv", "per_group_ext.csv", "README.md")


# ---------------------------------------------------------------- per-query quantities and point estimates
def relevance(fr, k=K):
    labs = fr["top5_labels"].str.split("|")
    if not labs.map(len).eq(k).all():
        raise ValueError("top5_labels must hold exactly 5 labels per query")
    return np.array([[lab == t for lab in ls] for ls, t in zip(labs, fr["true_label"])], dtype=bool)


def support(fr):
    s = pd.to_numeric(fr["library_support"], errors="raise").to_numpy(float)
    if not (np.all(s == np.floor(s)) and np.all(s >= 1)):
        raise ValueError("library_support must be integers >= 1")
    return s


def ap_at_k(rel, n_rel, k=K):
    r = rel[:, :k].astype(float)
    prec = np.cumsum(r, axis=1) / np.arange(1, k + 1, dtype=float)[None, :]
    return (prec * r).sum(1) / np.minimum(float(k), n_rel)


def ndcg_at_k(rel, n_rel, k=K):
    disc = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    dcg = (rel[:, :k].astype(float) * disc[None, :]).sum(1)
    idcg = np.concatenate([[0.0], np.cumsum(disc)])[np.minimum(float(k), n_rel).astype(int)]
    return dcg / idcg


def per_query(fr, rel10=None):
    rel, s = relevance(fr), support(fr)
    V = dict(ap5=ap_at_k(rel, s), ndcg5=ndcg_at_k(rel, s))
    if rel10 is not None:
        V.update(ap10=ap_at_k(rel10, s, k=10), ndcg10=ndcg_at_k(rel10, s, k=10))
    return V


def f1_counts(pred, truth, g):
    """0/1 per-query indicators of TP, FP, FN for group g (ABSTAIN equals no group: FN only)."""
    p, t = (pred == g), (truth == g)
    return (p & t).astype(float), (p & ~t).astype(float), (~p & t).astype(float)


def f1(tp, fp, fn):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2 * tp / (2 * tp + fp + fn)


def point_estimates(fr, rel10=None):
    """Unweighted point estimates computed directly (no weights), as metrics.compute_all does."""
    V, t = per_query(fr, rel10), fr["true_label"].to_numpy()
    groups = sorted(set(t))
    gm = {m: {g: float(np.mean(V[m][t == g])) for g in groups} for m in ("ap5", "ndcg5", "ap10", "ndcg10") if m in V}
    gf = {}
    for rule, col in RULES.items():
        pred = fr[col].to_numpy()
        gf[rule] = {g: float(f1(*(c.sum() for c in f1_counts(pred, t, g)))) for g in groups}
    out = dict(map5_micro=float(np.mean(V["ap5"])), map5_macro=float(np.mean(list(gm["ap5"].values()))),
               ndcg5_micro=float(np.mean(V["ndcg5"])), ndcg5_macro=float(np.mean(list(gm["ndcg5"].values()))))
    for rule in RULES:
        out[f"{rule}_macro_f1"] = float(np.mean(list(gf[rule].values())))
    per_group = {(g, "ap5"): gm["ap5"][g] for g in groups}
    per_group.update({(g, "ndcg5"): gm["ndcg5"][g] for g in groups})
    per_group.update({(g, f"{rule}_f1"): gf[rule][g] for rule in RULES for g in groups})
    for m, a in (("map10", "ap10"), ("ndcg10", "ndcg10")):
        if a in V:
            out[f"{m}_micro"] = float(np.mean(V[a]))
            out[f"{m}_macro"] = float(np.mean(list(gm[a].values())))
            per_group.update({(g, a): gm[a][g] for g in groups})
    return out, per_group


# ---------------------------------------------------------------- bootstrap
def interval(v):
    """95% percentile interval via the frozen module (2.5/97.5, numpy linear, NaN if any replicate non-finite)."""
    return F.ci(v)


def compute(fr, B, seed, rel10=None):
    V, t = per_query(fr, rel10), fr["true_label"].to_numpy()
    patients, mult, W = F.weight_matrix(fr["patient"], B, seed)
    n, npat, groups = len(fr), len(patients), sorted(set(t))
    pts, gpts = point_estimates(fr, rel10)
    has10 = rel10 is not None
    summary, per_group_metrics = dict(SUMMARY, **(SUMMARY10 if has10 else {})), PER_GROUP + (PER_GROUP10 if has10 else [])
    absent = np.zeros((W.shape[0], len(groups)), bool)
    gboot = {}
    for j, g in enumerate(groups):
        mask = t == g
        absent[:, j] = W[:, mask].sum(1) == 0
        for m in ("ap5", "ndcg5") + (("ap10", "ndcg10") if has10 else ()):
            gboot[(g, m)] = F.boot_means(W, V[m], mask)
        for rule, col in RULES.items():
            tp, fp, fn = (W @ c for c in f1_counts(fr[col].to_numpy(), t, g))
            v = f1(tp, fp, fn)
            gboot[(g, f"{rule}_f1")] = np.where(absent[:, j], np.nan, v)

    def macro(m):
        G = np.vstack([gboot[(g, m)] for g in groups]).T
        with np.errstate(invalid="ignore"):
            return np.nanmean(np.where(absent, np.nan, G), axis=1)

    boots = dict(map5_micro=F.boot_means(W, V["ap5"]), map5_macro=macro("ap5"),
                 ndcg5_micro=F.boot_means(W, V["ndcg5"]), ndcg5_macro=macro("ndcg5"))
    for rule in RULES:
        boots[f"{rule}_macro_f1"] = macro(f"{rule}_f1")
    if has10:
        boots.update(map10_micro=F.boot_means(W, V["ap10"]), map10_macro=macro("ap10"),
                     ndcg10_micro=F.boot_means(W, V["ndcg10"]), ndcg10_macro=macro("ndcg10"))
    rows = []
    for k, desc in summary.items():
        lo, hi, bad = interval(boots[k])
        is_macro = not k.endswith("_micro")
        rows.append(dict(metric=k, point=pts[k], lo=lo, hi=hi, n_queries=n, n_patients=npat,
                         boundary_flag=F.boundary_flag(lo, hi), n_replicates=int(W.shape[0]), n_nonfinite=bad,
                         n_groups=len(groups) if is_macro else "",
                         n_replicates_with_absent_group=int(absent.any(1).sum()) if is_macro else "",
                         min_groups_present=int((~absent).sum(1).min()) if (is_macro and W.shape[0]) else "",
                         definition=desc))
    grows = []
    # group-major for the @5/F1 metrics, then group-major for @10 (appended so earlier rows stay byte-identical)
    for block in (PER_GROUP, PER_GROUP10 if has10 else []):
        for j, g in enumerate(groups):
            mask = t == g
            for m in block:
                lo, hi, bad = interval(gboot[(g, m)])
                grows.append(dict(true_label=g, metric=m, point=gpts[(g, m)], lo=lo, hi=hi,
                                  n_queries=int(mask.sum()), n_patients=int(fr.loc[mask, "patient"].nunique()),
                                  boundary_flag=F.boundary_flag(lo, hi), n_replicates=int(W.shape[0]),
                                  n_replicates_group_absent=int(absent[:, j].sum()), n_nonfinite=bad))
    return dict(V=V, W=W, mult=mult, patients=patients, groups=groups, points=pts, group_points=gpts, boots=boots,
                group_boots=gboot, absent=absent, summary=pd.DataFrame(rows), per_group=pd.DataFrame(grows),
                rel10=rel10, per_group_metrics=per_group_metrics)


# ---------------------------------------------------------------- checks
def spot_check(fr, res, B):
    """Materialise replicates {0, B//2, B-1} by repeating each drawn patient's rows by its multiplicity, recompute
    every summary metric with the unweighted point path (macro over groups present) and compare."""
    rows_of = {p: np.flatnonzero((fr["patient"] == p).to_numpy()) for p in res["patients"]}
    reps = sorted({0, B // 2, B - 1})
    worst, n = 0.0, 0
    for b in reps:
        idx = np.concatenate([np.repeat(rows_of[p], int(res["mult"][b, i])) for i, p in enumerate(res["patients"])])
        r10 = None if res["rel10"] is None else res["rel10"][idx]
        pts, gpts = point_estimates(fr.iloc[idx].reset_index(drop=True), r10)
        for k in res["boots"]:
            worst = max(worst, abs(pts[k] - float(res["boots"][k][b]))); n += 1
        for (g, m), bm in res["group_boots"].items():
            ref, got = gpts.get((g, m), np.nan), float(bm[b])
            worst = max(worst, (0.0 if np.isnan(ref) and np.isnan(got) else np.inf) if np.isnan(ref) or np.isnan(got)
                        else abs(ref - got)); n += 1
    return reps, worst, n


def load_ranked(fr, ranked_src, compression="infer"):
    """Read-only: depth-10 relevance (frame query order) from the c2 ranked lists (path or in-memory buffer),
    cross-checked against the frame. Returns (rel10 bool matrix, consistency dict); rel10 is None unless every
    consistency item holds."""
    rl = pd.read_csv(ranked_src, dtype={"query": str, "true_label": str, "neighbour": str, "neighbour_label": str},
                     keep_default_na=False, compression=compression)
    rl["rank"] = rl["rank"].astype(int)
    rl = rl.sort_values(["query", "rank"], kind="mergesort")
    lab = rl.groupby("query", sort=False)["neighbour_label"].apply(list)
    nbr = rl.groupby("query", sort=False)["neighbour"].apply(list)
    tru = rl.groupby("query", sort=False)["true_label"].first()
    ranks_ok = bool(rl.groupby("query", sort=False)["rank"].apply(lambda s: s.tolist() == list(range(1, 11))).all())
    q = fr["query"].tolist()
    same_set = set(lab.index) == set(q) and len(lab) == len(q)
    cons = dict(n_rows=len(rl), same_query_set=same_set, ranks_1_to_10=ranks_ok)
    if not (same_set and ranks_ok and len(rl) == 10 * len(q)):
        return None, cons
    labs = [lab[x] for x in q]
    rel = np.array([[lb == t for lb in ls] for ls, t in zip(labs, fr["true_label"])], dtype=bool)
    rank = F.rank_values(fr["first_relevant_rank"])
    first = np.where(rel.any(1), rel.argmax(1) + 1, 0)
    cons.update(
        true_label_matches=bool((tru.reindex(q).to_numpy() == fr["true_label"].to_numpy()).all()),
        top5_labels_match=all("|".join(ls[:5]) == s for ls, s in zip(labs, fr["top5_labels"])),
        top5_neighbours_match=("top5_neighbours" not in fr.columns) or
        all("|".join(nbr[x][:5]) == s for x, s in zip(q, fr["top5_neighbours"])),
        hit10_matches=bool((rel.any(1).astype(int) == fr["hit10"].astype(int).to_numpy()).all()),
        precision10_matches=bool(np.allclose(rel.sum(1) / 10, fr["precision10"].astype(float), atol=TOL, rtol=0)),
        first_relevant_rank_matches=bool((first == rank).all()))
    return (rel if all(cons[k] for k in cons if k != "n_rows") else None), cons


def map10_from_ranked(fr, ranked_path):
    """Micro AP@10 from the verified ranked lists (NaN if they are inconsistent with the frame)."""
    rel10, cons = load_ranked(fr, ranked_path)
    return (np.nan if rel10 is None else float(np.mean(ap_at_k(rel10, support(fr), k=10)))), cons


def reconcile(fr, res, B, seed, ctx):
    checks = []

    def add(name, ok, **d):
        checks.append(dict(name=name, status="PASS" if ok else "FAIL", **d))

    def waive(name, **d):
        checks.append(dict(name=name, status="WAIVED", note="--allow-any-n --no-reference (tests only)", **d))

    add("input_sha256_matches_expected", ctx["input_sha"] == ctx["expected_sha"], observed=ctx["input_sha"],
        expected=ctx["expected_sha"])
    if ctx["allow_any_n"]:
        waive("input_n_rows_equals_2778", observed=len(fr))
        waive("input_n_patients_equals_1876", observed=len(res["patients"]))
    else:
        add("input_n_rows_equals_2778", len(fr) == F.EXPECTED_N, observed=len(fr))
        add("input_n_patients_equals_1876", len(res["patients"]) == F.EXPECTED_PATIENTS, observed=len(res["patients"]))
    add("frozen_module_sha256_matches_accepted", ctx["frozen_sha"] == ACCEPTED_FROZEN_SCRIPT_SHA256,
        observed=ctx["frozen_sha"], expected=ACCEPTED_FROZEN_SCRIPT_SHA256)
    s = support(fr)
    add("library_support_integer_ge_1", True, min=float(s.min()), n_below_5=int((s < 5).sum()),
        n_below_10=int((s < 10).sum()), note="ideal DCG / AP normaliser use min(k, library_support)")
    groups = set(res["groups"])
    add("rule_predictions_in_group_set",
        set(fr["plurality_k5"]) <= groups and set(fr["legacy_generous_k5_pred"]) <= groups
        and set(fr["strict_majority_k5"]) <= groups | {ABSTAIN},
        note="F1 false positives are only counted for the true-label groups; strict may also be ABSTAIN")
    bad = F.rederive_from_top5(fr)
    add("rules_rederived_from_top5_labels", not bad, n_mismatch=len(bad), first=bad[:10],
        note="frozen-module re-derivation: plurality closest-ranked tie-break, strict >=3/5, oracle tied-max, "
             "hit/top1/precision5")
    V = res["V"]
    add("ap5_ndcg5_in_unit_interval", bool(((V["ap5"] >= 0) & (V["ap5"] <= 1 + TOL) & (V["ndcg5"] >= 0)
                                           & (V["ndcg5"] <= 1 + TOL)).all()))
    miss5 = fr["hit5"].astype(int).to_numpy() == 0
    add("ap5_ndcg5_zero_iff_hit5_zero", bool(((V["ap5"] == 0) == miss5).all() and ((V["ndcg5"] == 0) == miss5).all()))
    pts = res["points"]
    # ranked lists (always required: the @10 metrics take rank 6-10 relevance from them)
    rk = ctx["ranked"]
    add("ranked_lists_sha256_matches_accepted", rk["sha256"] == ctx["expected_ranked_sha"],
        observed=rk["sha256"], expected=ctx["expected_ranked_sha"])
    cons = rk.get("consistency") or {}
    add("ranked_lists_consistent_with_frame", bool(cons) and cons.get("n_rows") == 10 * len(fr)
        and all(v for k, v in cons.items() if k != "n_rows"), **cons)
    has10 = res["rel10"] is not None
    add("at10_metrics_computed_from_verified_ranked_lists",
        has10 and set(SUMMARY10) <= set(res["boots"]) and len(res["summary"]) == len(SUMMARY) + len(SUMMARY10),
        note="mAP@10 / nDCG@10 use rank 1-10 relevance from the ranked lists only after the consistency check")
    if has10:
        miss10 = fr["hit10"].astype(int).to_numpy() == 0
        add("ap10_ndcg10_in_unit_interval_and_zero_iff_hit10_zero",
            bool(((V["ap10"] >= 0) & (V["ap10"] <= 1 + TOL) & (V["ndcg10"] >= 0) & (V["ndcg10"] <= 1 + TOL)).all()
                 and ((V["ap10"] == 0) == miss10).all() and ((V["ndcg10"] == 0) == miss10).all()))
        add("ranked_top5_relevance_equals_frame", bool(np.array_equal(res["rel10"][:, :5], relevance(fr))))
    # reference reproduction
    if ctx["no_reference"]:
        for name in ("stated_point_values_reproduced", "metrics_json_sha256_matches_accepted",
                     "metrics_json_points_reproduced", "map10_micro_reproduced_from_ranked_lists"):
            waive(name)
    else:
        diffs = {k: pts[k] - v for k, v in STATED.items()}
        add("stated_point_values_reproduced", all(abs(d) <= STATED_TOL for d in diffs.values()), tol=STATED_TOL,
            stated=STATED, observed={k: pts[k] for k in STATED}, diff=diffs)
        mj = ctx["metrics_json"]
        add("metrics_json_sha256_matches_accepted", mj["sha256"] == ctx["expected_metrics_sha"],
            observed=mj["sha256"], expected=ctx["expected_metrics_sha"])
        pe = mj.get("point_estimates") or {}
        jd = {k: (pts[k] - pe[j]) if (j in pe and k in pts) else None for k, j in JSON_KEYS.items()}
        add("metrics_json_points_reproduced", all(d is not None and abs(d) <= TOL for d in jd.values()), tol=TOL,
            keys=JSON_KEYS, diff=jd, n_queries_json=pe.get("n_queries"))
        m10 = pts.get("map10_micro", np.nan)
        j10 = pe.get("micro_ap_at_10")
        add("map10_micro_reproduced_from_ranked_lists",
            bool(np.isfinite(m10) and abs(m10 - STATED_MAP10_MICRO) <= STATED_TOL and j10 is not None
                 and abs(m10 - j10) <= TOL), observed=m10, stated=STATED_MAP10_MICRO, json=j10,
            note="author-stated mAP@10 and c2 JSON micro_ap_at_10, both from the verified ranked lists")
    # pairing with the frozen intervals
    Wf = F.weight_matrix(fr["patient"], B, seed)[2]
    _, mult10 = F.make_patient_draws(list(fr["patient"]), count=min(10, B), seed=seed)
    fz = ctx["frozen_res"]
    hit5_b = F.boot_means(res["W"], relevance(fr).any(1).astype(float))
    add("draws_identical_to_frozen", np.array_equal(Wf, res["W"]) and np.array_equal(mult10, res["mult"][:10])
        and np.array_equal(fz["W"], res["W"]), compared="full weight matrix; first 10 draws of make_patient_draws")
    add("paired_with_frozen_bootstrap",
        float(np.max(np.abs(hit5_b - fz["boots"]["hit5"]))) <= TOL,
        max_abs_diff=float(np.max(np.abs(hit5_b - fz["boots"]["hit5"]))),
        note="per-replicate hit5 recomputed from top5_labels with this script's weights == frozen compute()")
    # interval method and level
    worst = 0.0
    tabs = [(res["summary"].set_index("metric"), res["boots"])]
    gtab = res["per_group"].set_index(["true_label", "metric"])
    for k, bv in res["boots"].items():
        if np.all(np.isfinite(bv)):
            lo, hi = np.percentile(bv, [Q_LO, Q_HI])
            worst = max(worst, abs(lo - tabs[0][0].loc[k, "lo"]), abs(hi - tabs[0][0].loc[k, "hi"]))
        elif np.isfinite(tabs[0][0].loc[k, "lo"]):
            worst = np.inf
    for key, bv in res["group_boots"].items():
        if np.all(np.isfinite(bv)):
            lo, hi = np.percentile(bv, [Q_LO, Q_HI])
            worst = max(worst, abs(lo - gtab.loc[key, "lo"]), abs(hi - gtab.loc[key, "hi"]))
        elif np.isfinite(gtab.loc[key, "lo"]):
            worst = np.inf
    add("ci_level_95_percentile_2_5_97_5", worst == 0.0, max_abs_diff=float(worst),
        note="independent np.percentile(boot, [2.5, 97.5]) equals every reported interval")
    add("replicate_count_equals_B", res["W"].shape[0] == B and all(len(v) == B for v in res["boots"].values()))
    macro_ok = all(abs(pts[f"{m}_macro"] - np.mean([res["group_points"][(g, a)] for g in res["groups"]])) <= TOL
                   for m, a in (("map5", "ap5"), ("ndcg5", "ndcg5"), ("map10", "ap10"), ("ndcg10", "ndcg10"))
                   if f"{m}_macro" in pts) and all(
        abs(pts[f"{r}_macro_f1"] - np.mean([res["group_points"][(g, f"{r}_f1")] for g in res["groups"]])) <= TOL
        for r in RULES)
    add("macro_point_equals_mean_of_group_points", macro_ok)
    add("per_group_n_sums_to_n", int(res["per_group"].query("metric == 'ap5'")["n_queries"].sum()) == len(fr))
    reps, sw, sn = spot_check(fr, res, B)
    add("brute_force_resample_spot_check", sw <= 1e-9, replicates=reps, n_compared=sn, max_abs_diff=sw,
        note="materialised patient resamples vs weight-matrix path (tolerance 1e-9: different summation order)")
    allrows = pd.concat([res["summary"][["lo", "hi", "point"]], res["per_group"][["lo", "hi", "point"]]])
    fin = allrows.dropna()
    out = fin[(fin["point"] < fin["lo"] - TOL) | (fin["point"] > fin["hi"] + TOL)]
    checks.append(dict(name="ci_contains_point", status="PASS" if out.empty else "WARN", n_outside=len(out),
                       note="WARN not FAIL: skewed small-group bootstrap distributions can exclude the point"))
    return checks


# ---------------------------------------------------------------- outputs
def write_readme(path, rec, res, checks):
    s = res["summary"]
    nw = sum(c["status"] == "WARN" for c in checks)
    lines = [f"STATUS: COMPLETE - 0 FAIL, {nw} WARN"
             + (f" ({', '.join(c['name'] for c in checks if c['status'] == 'WARN')}; see reconciliation.json)"
                if nw else "")
             + f", {sum(c['status'] == 'PASS' for c in checks)} PASS, "
             f"{sum(c['status'] == 'WAIVED' for c in checks)} WAIVED of {len(checks)} reconciliation checks", "",
             "# Whole-patient bootstrap CIs: mAP@5/@10, nDCG@5/@10, macro-F1 (R4.b-vii)", "",
             f"Input: `{rec['input']['path']}` (sha256 {rec['input']['sha256']}; {rec['input']['n_rows']} queries, "
             f"{rec['input']['n_patients']} patients, {rec['input']['n_groups']} diagnostic groups = 14 tumour + "
             "NORMAL; MISC dropped upstream).", "",
             "## Definitions (reproduce c2_metric_suite / prosper-search metrics.py)", "",
             "- Relevant neighbour: label of the neighbour at that rank equals the query's true group. Same-patient "
             "slides were already excluded from the neighbour lists. Ranks 1-5 come from the frame's `top5_labels`; "
             "ranks 6-10 are NOT in the frame and come from the c2 ranked lists (see below).",
             "- AP@k (k = 5, 10): sum over relevant ranks i<=k of precision@i, divided by min(k, R); R = "
             "`library_support`, the number of library slides of the same group from other patients (>= 27 here, "
             "so the normaliser is k).",
             "- nDCG@k: binary gains, log2(i+1) discount; ideal DCG places min(k, R) relevant items first.",
             "- micro = mean over queries; macro = unweighted mean over groups of per-group means (NORMAL counts as "
             "a group).",
             "- macro-F1: per group, F1 = 2TP/(2TP+FP+FN) for the rule's predicted label, then the unweighted mean "
             "over the 15 groups. plurality = most frequent top-5 label, ties to the closest-ranked tied label; "
             "generous = manuscript oracle tie rule (true label credited whenever it is among the tied-max labels; "
             "optimistic by construction); strict = label with >= 3 of 5 votes else ABSTAIN, and ABSTAIN is a false "
             "negative for the true group and a false positive for no group.", "",
             "## Patient bootstrap", "",
             f"B = {rec['B']}, seed = {rec['seed']}, {rec['rng']}; the patient draws are produced by the frozen "
             "module and are identical to those behind `r2_3_internal_failure_cis` (checked: weight matrix and "
             "per-replicate hit5). Each replicate resamples the patients with replacement and weights every query "
             "by its patient's multiplicity. F1 uses weighted TP/FP/FN counts. Macro values average over the groups "
             "with non-zero weight in that replicate (absent groups excluded; counts in the CSVs). Per-group "
             "intervals are unavailable if the group is absent from any replicate. "
             f"Interval: {rec['percentile_method']}.", "",
             "## Results", "", "| metric | point | 95% CI | flag |", "|---|---|---|---|"]
    for r in s.itertuples():
        lines.append(f"| {r.metric} | {r.point:.4f} | {r.lo:.4f}-{r.hi:.4f} | {r.boundary_flag} |")
    rk = rec.get("ranked_lists") or {}
    lines += ["", "Per-group F1 (three rules), nDCG@5, AP@5, AP@10 and nDCG@10 with intervals are in "
              "`per_group_ext.csv` (the @10 rows follow the @5/F1 rows). Percentile intervals in groups of only "
              "~20-40 patients can be too narrow; [0,0] or [x,1] bounds do not mean certainty.", "",
              "## @10 metrics and the ranked lists", "",
              f"mAP@10 and nDCG@10 take relevance for ranks 6-10 from the c2 ranked lists `{rk.get('path')}` "
              f"(sha256 {rk.get('sha256')}), read-only; the per-query frame has no rank 6-10 labels. They are used "
              "only after the lists are checked against the frame (same 2,778 queries, ranks 1-10 complete, true "
              "label, top-5 neighbour ids and labels, hit10, precision10 and first relevant rank all identical). "
              "The point estimates reproduce the c2 metrics JSON (micro/macro AP@10 and nDCG@10) and the stated "
              "mAP@10 = 0.701745; the intervals use the same patient draws as every other row.", "",
              "## Not computed / not comparable", "",
              "- Nothing beyond depth 10 (the ranked lists stop at rank 10).",
              "- The c2 metrics JSON intervals use a different resampling unit (pps_cluster_one_slide) and are not "
              "comparable to these whole-patient intervals; quote one interval family per estimand.", "",
              f"Command: `{rec['command']}`", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--expected-sha256", default=F.ACCEPTED_INPUT_SHA256)
    ap.add_argument("--metrics-json", default=DEFAULT_METRICS_JSON)
    ap.add_argument("--expected-metrics-sha256", default=ACCEPTED_METRICS_JSON_SHA256)
    ap.add_argument("--ranked-lists", default=DEFAULT_RANKED)
    ap.add_argument("--expected-ranked-sha256", default=ACCEPTED_RANKED_SHA256)
    ap.add_argument("--allow-any-n", action="store_true", help="tests only: do not require 2,778 rows/1,876 patients")
    ap.add_argument("--no-reference", action="store_true",
                    help="tests only (requires --allow-any-n): waive stated-value and metrics-JSON reproduction "
                         "checks (the ranked lists are still required and checked)")
    a = ap.parse_args(argv)
    if os.path.exists(a.outdir) and (not os.path.isdir(a.outdir) or os.listdir(a.outdir)):
        print(f"refusing to overwrite existing non-empty {a.outdir}", file=sys.stderr)
        return 2
    if a.B < 1:
        print("--B must be >= 1", file=sys.stderr)
        return 2
    if a.no_reference and not a.allow_any_n:
        print("--no-reference is only permitted together with --allow-any-n (tests)", file=sys.stderr)
        return 2
    fr, input_sha, n_bytes = F.read_input(a.input)
    if len(fr) != F.EXPECTED_N and not a.allow_any_n:
        print(f"input has {len(fr)} rows, expected {F.EXPECTED_N}; refusing (use --allow-any-n for tests)",
              file=sys.stderr)
        return 2
    ctx = dict(input_sha=input_sha, expected_sha=a.expected_sha256, allow_any_n=a.allow_any_n,
               no_reference=a.no_reference, frozen_sha=F.sha256(FROZEN_SCRIPT),
               expected_metrics_sha=a.expected_metrics_sha256, expected_ranked_sha=a.expected_ranked_sha256)
    if not a.no_reference:
        mj = dict(path=os.path.abspath(a.metrics_json), sha256=None)
        if os.path.isfile(a.metrics_json):
            with open(a.metrics_json, "rb") as fh:
                data = fh.read()
            mj.update(sha256=hashlib.sha256(data).hexdigest(), **json.loads(data))
        ctx.update(metrics_json=mj)
    rk, rel10 = dict(path=os.path.abspath(a.ranked_lists), sha256=None), None
    if os.path.isfile(a.ranked_lists):
        with open(a.ranked_lists, "rb") as fh:
            data = fh.read()
        rk["sha256"] = hashlib.sha256(data).hexdigest()
        rel10, rk["consistency"] = load_ranked(fr, io.BytesIO(data),
                                               "gzip" if a.ranked_lists.endswith(".gz") else None)
    ctx.update(ranked=rk)
    res = compute(fr, a.B, a.seed, rel10)
    ctx["frozen_res"] = F.compute(fr, a.B, a.seed)
    checks = reconcile(fr, res, a.B, a.seed, ctx)
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    os.makedirs(a.outdir, exist_ok=True)
    me = os.path.abspath(__file__)
    ref = lambda d: None if d is None else {k: d.get(k) for k in ("path", "sha256")}  # noqa: E731
    rec = dict(
        timestamp=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        command=" ".join([sys.executable] + (sys.argv if argv is None else [me] + list(argv))),
        cwd=os.getcwd(), prosper_out=O, hostname=socket.gethostname(),
        script=dict(path=me, sha256=F.sha256(me)),
        frozen_module=dict(path=FROZEN_SCRIPT, sha256=ctx["frozen_sha"], expected=ACCEPTED_FROZEN_SCRIPT_SHA256,
                           role="loader, make_patient_draws, weight_matrix, boot_means, ci, boundary_flag, "
                                "rederive_from_top5, compute (pairing check)"),
        input=dict(path=os.path.abspath(a.input), sha256=input_sha, n_bytes=n_bytes, n_rows=len(fr),
                   n_patients=len(res["patients"]), n_groups=len(res["groups"]),
                   hashing="input bytes read once by the frozen read_input; sha256 and parse from the same bytes"),
        metrics_json=ref(ctx.get("metrics_json")), ranked_lists=ref(ctx.get("ranked")),
        ranked_lists_role=("rank 1-10 relevance for mAP@10/nDCG@10, used only after the frame-consistency check; "
                           "bytes read once, sha256 and parse from the same bytes"),
        metric_definition_source=METRIC_SOURCE,
        B=a.B, seed=a.seed, rng="numpy.random.Generator(PCG64(seed)) via frozen make_patient_draws",
        interval="percentile 2.5/97.5 (95%)", percentile_method=F.PERCENTILE_METHOD,
        draws_function=F.DRAWS_COPIED_FROM, draws_source=F.DRAWS_SOURCE, draws_source_note=F.DRAWS_SOURCE_NOTE,
        macro_rule="unweighted mean over groups with non-zero weight in the replicate; absent groups excluded",
        f1_conventions=dict(strict="ABSTAIN = false negative for the true group, false positive for no group",
                            generous="legacy_generous_k5_pred (oracle tied-max rule)",
                            plurality="plurality_k5 (closest-ranked tied label)",
                            undefined="per-group F1 is NaN only when the group is absent from the replicate "
                                      "(TP+FN weight 0); then excluded from that replicate's macro mean"),
        allow_any_n=a.allow_any_n, no_reference=a.no_reference,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        reconciliation_all_pass=n_fail == 0, n_checks=len(checks))
    with open(os.path.join(a.outdir, "record.json"), "w") as fh:
        json.dump(rec, fh, indent=2, default=F._jsonable)
    with open(os.path.join(a.outdir, "reconciliation.json"), "w") as fh:
        json.dump(dict(all_pass=n_fail == 0, n_fail=n_fail, n_checks=len(checks), checks=checks), fh, indent=2,
                  default=F._jsonable)
    for c in checks:
        print(f"{c['status']:7s} {c['name']}")
    if n_fail:
        with open(os.path.join(a.outdir, "FAILED"), "w") as fh:
            fh.write("\n".join(c["name"] for c in checks if c["status"] == "FAIL") + "\n")
        print(f"RECONCILIATION FAILED: {n_fail} check(s); result CSVs/README not written", file=sys.stderr)
        return 1
    res["summary"].to_csv(os.path.join(a.outdir, "summary_ext.csv"), index=False)
    res["per_group"].to_csv(os.path.join(a.outdir, "per_group_ext.csv"), index=False)
    write_readme(os.path.join(a.outdir, "README.md"), rec, res, checks)
    with open(os.path.join(a.outdir, "COMPLETE"), "w") as fh:
        fh.write(f"{len(checks)} checks, 0 FAIL\n")
    print(f"all checks pass; outputs in {a.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
