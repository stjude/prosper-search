#!/usr/bin/env python3
"""R2.3 / R4.b / E2: whole-patient bootstrap 95% CIs for the accepted Stream A per-query frame.

Input is only the accepted all_query_frame.csv written by scripts/r2_3_internal_failure_tables.py
(UNI2 / OT cosine / unfiltered / 2,778 COMET slides / 15 groups incl. NORMAL / same-patient masked).
No retrieval is re-run. Each bootstrap replicate draws patients with replacement ([decision record]
make_patient_draws, PCG64) and weights every query by its patient's draw multiplicity; every
estimand is a weighted mean over queries; intervals are the 2.5/97.5 percentiles.

Encodings inherited from the frozen A script (verified at run time, see reconciliation.json):
  strict_majority_k5  a label, or 'ABSTAIN' when no label has >= 3 of the top-5.
  first_relevant_rank '1'..'10', or 'absent@10' when no top-10 neighbour has the true label.
  legacy_generous_k5_correct  the manuscript's ORACLE tie rule (correct if the truth is among the
                      tied-most-frequent top-5 labels); 0/1.
"""
import argparse
import datetime
import hashlib
import io
import json
import os
import platform
import socket
import sys
from collections import Counter

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = os.environ.get("PROSPER_OUT", os.path.join(R, "prosper_revision_output"))
DEFAULT_INPUT = os.path.join(O, "r2_3_internal_failure_20261003", "all_query_frame.csv")
ACCEPTED_INPUT_SHA256 = "1ab4e8441a4cd6210b3b114428414d572819cc91020f8ff9f40361a45a3c25c1"
ACCEPTED_FROZEN_PER_GROUP_SHA256 = "669eac6ede12f1dd50cf4b40b1d8778477fec04638f03aaa52a92759503d886f"
EXPECTED_N, EXPECTED_PATIENTS = 2778, 1876
ABSTAIN, ABSENT = "ABSTAIN", "absent@10"
TOL = 1e-12
DRAWS_SOURCE = ("make_patient_draws from prosper-search prosper/evaluation/r2_1_hoptimus_main.py, an UNTRACKED "
                "working-tree file (repository HEAD 0edc479 does not contain it); bound only by its sha256 "
                "dee7d68a8b9fcee61d2e845412afdaf20455388ae5c2e61844d4b21d73811faf")
DRAWS_COPIED_FROM = ("make_patient_draws copied verbatim from scripts/r3_7_fine_label_reevaluation.py "
                     "(sha256 bb7436efa7b18c617b970b0796ddf867be053b5f768b820d37cbe4902cc5e5f2), itself copied "
                     "from DRAWS_SOURCE")
DRAWS_SOURCE_NOTE = ("'repository HEAD 0edc479' in DRAWS_SOURCE refers to the prosper-search repository, "
                     "not this revision workspace")
A_SCRIPT = os.path.join(R, "scripts", "r2_3_internal_failure_tables.py")
PERCENTILE_METHOD = "numpy.percentile at 2.5 and 97.5, default method='linear', over all B replicates"
REQUIRED = ["query", "true_label", "patient", "top1", "plurality_k3", "plurality_k5", "plurality_k10",
            "strict_majority_k5", "legacy_generous_k5_correct", "legacy_generous_k5_pred", "hit1", "hit3", "hit5",
            "hit10", "precision5", "precision10", "first_relevant_rank", "top5_labels", "library_support"]
N_SPOT = 3

OVERALL = {
    "oracle_generous_k5": "legacy_generous_k5_correct (ORACLE tie rule: correct if truth is among the tied-max "
                          "top-5 labels); denominator all queries",
    "plurality_k5_accuracy": "plurality_k5 == true_label; denominator all queries",
    "strict_k5_accuracy": "strict_majority_k5 == true_label; denominator all queries (ABSTAIN counts as not correct)",
    "strict_k5_abstain_rate": "strict_majority_k5 == ABSTAIN; denominator all queries",
    "top1_accuracy": "hit1 (rank-1 neighbour label == truth); denominator all queries",
    "hit3": "any top-3 neighbour label == truth; denominator all queries",
    "hit5": "any top-5 neighbour label == truth; denominator all queries",
    "hit10": "any top-10 neighbour label == truth; denominator all queries",
    "precision5": "fraction of top-5 neighbours with the true label, averaged over all queries",
    "precision10": "fraction of top-10 neighbours with the true label, averaged over all queries",
    "mrr10": "1/first_relevant_rank, 0 when absent@10, averaged over all queries",
    "plurality_k3_accuracy": "plurality_k3 == true_label; denominator all queries",
    "plurality_k10_accuracy": "plurality_k10 == true_label; denominator all queries",
}
PAIRED = {
    "oracle_minus_plurality_k5": ("oracle_generous_k5", "plurality_k5_accuracy"),
    "oracle_minus_strict_k5": ("oracle_generous_k5", "strict_k5_accuracy"),
    "hit5_minus_plurality_k5": ("hit5", "plurality_k5_accuracy"),
}
PER_GROUP = ["plurality_k5_recall", "hit5", "hit10", "strict_k5_abstain_rate"]
MACRO = {"macro_plurality_k5_recall": "plurality_k5_recall", "macro_hit5": "hit5", "macro_hit10": "hit10"}
# frozen per_group_summary.csv column -> per-query vector name here
FROZEN_MAP = {"hit1": "top1_accuracy", "hit3": "hit3", "hit5": "hit5", "hit10": "hit10",
              "precision5": "precision5", "precision10": "precision10",
              "plurality5_recall": "plurality_k5_accuracy", "strict5_recall": "strict_k5_accuracy",
              "strict5_abstain_rate": "strict_k5_abstain_rate", "legacy_generous5_recall": "oracle_generous_k5"}


def make_patient_draws(patient_ids, count=5000, seed=123):
    """Copied from DRAWS_SOURCE (package import pulls heavy deps); identical sampling.
    Only the input validators (_integer/_ids) are reduced to int()/str checks."""
    count, seed = int(count), int(seed)
    if any(not isinstance(v, str) or not v.strip() for v in patient_ids):
        raise ValueError("patient_ids must contain nonempty string IDs")
    patients = tuple(sorted(set(patient_ids)))
    p = len(patients)
    multiplicities = np.zeros((count, p), dtype=np.int64)
    if p:
        rng = np.random.Generator(np.random.PCG64(seed))
        for b in range(count):
            multiplicities[b] = np.bincount(rng.integers(p, size=p), minlength=p)
    return patients, multiplicities


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_input(path):
    """Read the input bytes once; hash and parse the same bytes. Returns (frame, sha256, n_bytes)."""
    with open(path, "rb") as fh:
        data = fh.read()
    return load_frame(data), hashlib.sha256(data).hexdigest(), len(data)


def load_frame(data):
    fr = pd.read_csv(io.BytesIO(data), dtype={"query": str, "patient": str, "true_label": str, "first_relevant_rank": str,
                                  "strict_majority_k5": str, "plurality_k3": str, "plurality_k5": str,
                                  "plurality_k10": str, "top1": str, "top5_labels": str,
                                  "legacy_generous_k5_pred": str}, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in fr.columns]
    if missing:
        raise ValueError(f"input lacks columns {missing}")
    return fr


def rank_values(s):
    """first_relevant_rank -> int rank, 0 for absent@10; raises on any other encoding."""
    ok = set(map(str, range(1, 11))) | {ABSENT}
    bad = sorted(set(s) - ok)
    if bad:
        raise ValueError(f"unexpected first_relevant_rank values {bad[:5]}")
    return s.map(lambda x: 0 if x == ABSENT else int(x)).to_numpy()


def per_query_vectors(fr):
    """Per-query 0/1 or [0,1] values whose (weighted) means are the estimands."""
    t = fr["true_label"]
    rank = rank_values(fr["first_relevant_rank"])
    V = {
        "oracle_generous_k5": fr["legacy_generous_k5_correct"].astype(float).to_numpy(),
        "plurality_k5_accuracy": (fr["plurality_k5"] == t).astype(float).to_numpy(),
        "strict_k5_accuracy": (fr["strict_majority_k5"] == t).astype(float).to_numpy(),
        "strict_k5_abstain_rate": (fr["strict_majority_k5"] == ABSTAIN).astype(float).to_numpy(),
        "top1_accuracy": fr["hit1"].astype(float).to_numpy(),
        "hit3": fr["hit3"].astype(float).to_numpy(),
        "hit5": fr["hit5"].astype(float).to_numpy(),
        "hit10": fr["hit10"].astype(float).to_numpy(),
        "precision5": fr["precision5"].astype(float).to_numpy(),
        "precision10": fr["precision10"].astype(float).to_numpy(),
        "mrr10": np.where(rank > 0, 1.0 / np.where(rank > 0, rank, 1), 0.0),
        "plurality_k3_accuracy": (fr["plurality_k3"] == t).astype(float).to_numpy(),
        "plurality_k10_accuracy": (fr["plurality_k10"] == t).astype(float).to_numpy(),
    }
    V["plurality_k5_recall"] = V["plurality_k5_accuracy"]
    return V


def weight_matrix(patient_col, B, seed):
    """(B x n_queries) weights: each query gets its patient's multiplicity in replicate b."""
    patients, mult = make_patient_draws(list(patient_col), count=B, seed=seed)
    idx = pd.Index(patients).get_indexer(list(patient_col))
    return patients, mult, mult[:, idx].astype(float)


def boot_means(W, v, mask=None):
    """Weighted mean per replicate; NaN where the subset has zero weight."""
    if mask is not None:
        W, v = W[:, mask], v[mask]
    den = W.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ v) / np.where(den > 0, den, np.nan)


def ci(v):
    """Percentile 2.5/97.5 over ALL replicates; any non-finite replicate makes the interval unavailable
    (NaN) rather than being dropped ([decision record] / r3_7 convention). Returns (lo, hi, n_nonfinite)."""
    v = np.asarray(v, float)
    bad = int((~np.isfinite(v)).sum())
    if bad or not v.size:
        return np.nan, np.nan, bad
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), 0


def boundary_flag(lo, hi):
    """'' or '|'-joined: ci_unavailable, lower_eq_0, upper_eq_1, zero_width."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "ci_unavailable"
    f = [n for n, c in (("lower_eq_0", lo == 0), ("upper_eq_1", hi == 1), ("zero_width", lo == hi)) if c]
    return "|".join(f)


def _plurality(labs):
    c = Counter(labs)
    mx = max(c.values())
    return next(lab for lab in labs if c[lab] == mx)


def _strict(labs):
    lab, n = Counter(labs).most_common(1)[0]
    return lab if n > len(labs) / 2 else ABSTAIN


def _oracle(labs, truth):
    c = Counter(labs)
    mx = max(c.values())
    tied = {lab for lab, n in c.items() if n == mx}
    return (1, truth) if truth in tied else (0, next(lab for lab in labs if lab in tied))


def rederive_from_top5(fr):
    """Independently re-derive the depth-5 rule outputs from top5_labels; returns list of 'query:field' mismatches."""
    bad = []
    for r in fr.itertuples(index=False):
        labs, t = r.top5_labels.split("|"), r.true_label
        if len(labs) != 5:
            bad.append(f"{r.query}:top5_len"); continue
        rel = [lab == t for lab in labs]
        first = next((i + 1 for i, x in enumerate(rel) if x), None)
        ok, pred = _oracle(labs, t)
        exp = dict(top1=labs[0], plurality_k3=_plurality(labs[:3]), plurality_k5=_plurality(labs),
                   strict_majority_k5=_strict(labs), legacy_generous_k5_correct=ok, legacy_generous_k5_pred=pred,
                   hit1=int(rel[0]), hit3=int(any(rel[:3])), hit5=int(any(rel)))
        for k, v in exp.items():
            if getattr(r, k) != v:
                bad.append(f"{r.query}:{k}")
        if abs(float(r.precision5) - sum(rel) / 5) > TOL:
            bad.append(f"{r.query}:precision5")
        fr_rank = r.first_relevant_rank
        if first is not None and fr_rank != str(first):
            bad.append(f"{r.query}:first_relevant_rank")
        if first is None and fr_rank in {"1", "2", "3", "4", "5"}:
            bad.append(f"{r.query}:first_relevant_rank")
    return bad


def spot_check(fr, res, B):
    """Brute-force resample N_SPOT replicates from the explicit patient -> row mapping (repeat each drawn
    patient's rows by its multiplicity), recompute estimands on the materialised frame and compare with the
    weight-matrix path. Returns (replicates, max_abs_diff, n_compared)."""
    rows_of = {p: np.flatnonzero((fr["patient"] == p).to_numpy()) for p in res["patients"]}
    reps = sorted({0, B // 2, B - 1})[:N_SPOT]
    worst, n = 0.0, 0
    for b in reps:
        idx = np.concatenate([np.repeat(rows_of[p], int(res["mult"][b, i]))
                              for i, p in enumerate(res["patients"])])
        sub = fr.iloc[idx].reset_index(drop=True)
        rv = per_query_vectors(sub)
        for k in OVERALL:
            worst = max(worst, abs(float(rv[k].mean()) - float(res["boots"][k][b]))); n += 1
        for (g, m), bm in res["group_boots"].items():
            mk = (sub["true_label"] == g).to_numpy()
            ref = float(rv[m][mk].mean()) if mk.any() else np.nan
            if np.isnan(ref) or np.isnan(bm[b]):
                worst = max(worst, 0.0 if (np.isnan(ref) and np.isnan(bm[b])) else np.inf)
            else:
                worst = max(worst, abs(ref - float(bm[b])))
            n += 1
    return reps, worst, n


def compute(fr, B, seed):
    V = per_query_vectors(fr)
    patients, mult, W = weight_matrix(fr["patient"], B, seed)
    n, npat = len(fr), len(patients)
    overall, paired, group_rows, macro_rows = [], [], [], []
    boots = {}
    for k, desc in OVERALL.items():
        boots[k] = boot_means(W, V[k])
        lo, hi, bad = ci(boots[k])
        overall.append(dict(estimand=k, point=float(V[k].mean()), ci_lo=lo, ci_hi=hi, n_queries=n, n_patients=npat,
                            n_replicates=int(W.shape[0]), n_nonfinite=bad, definition=desc,
                            boundary_flag=boundary_flag(lo, hi)))
    for k, (a, b) in PAIRED.items():
        lo, hi, bad = ci(boots[a] - boots[b])
        paired.append(dict(estimand=k, minuend=a, subtrahend=b, point=float(V[a].mean() - V[b].mean()),
                           ci_lo=lo, ci_hi=hi, n_queries=n, n_patients=npat, n_replicates=int(W.shape[0]),
                           n_nonfinite=bad, definition=f"{a} - {b} on the same queries and the same patient draws"))
    groups = sorted(fr["true_label"].unique())
    gboot, gpoint = {m: [] for m in PER_GROUP}, {m: [] for m in PER_GROUP}
    group_boots = {}
    absent = np.zeros((W.shape[0], len(groups)), bool)
    for j, g in enumerate(groups):
        mask = (fr["true_label"] == g).to_numpy()
        absent[:, j] = W[:, mask].sum(1) == 0
        for m in PER_GROUP:
            bm = boot_means(W, V[m], mask)
            pt = float(V[m][mask].mean())
            gboot[m].append(bm); gpoint[m].append(pt); group_boots[(g, m)] = bm
            lo, hi, bad = ci(bm)
            group_rows.append(dict(true_label=g, metric=m, n_queries=int(mask.sum()),
                                   n_patients=int(fr.loc[mask, "patient"].nunique()), point=pt, ci_lo=lo, ci_hi=hi,
                                   n_replicates=int(W.shape[0]), n_replicates_group_absent=int(absent[:, j].sum()),
                                   n_nonfinite=bad, boundary_flag=boundary_flag(lo, hi)))
    for k, m in MACRO.items():
        G = np.vstack(gboot[m]).T  # B x groups, NaN where the group is absent from the replicate
        with np.errstate(invalid="ignore"):
            rep = np.nanmean(np.where(absent, np.nan, G), axis=1) if G.size else np.array([])
        lo, hi, bad = ci(rep)
        macro_rows.append(dict(estimand=k, metric=m, point=float(np.mean(gpoint[m])), ci_lo=lo, ci_hi=hi,
                               n_groups=len(groups), n_replicates=int(W.shape[0]),
                               n_replicates_with_absent_group=int(absent.any(1).sum()),
                               min_groups_present=int((~absent).sum(1).min()) if W.shape[0] else 0,
                               n_nonfinite=bad,
                               definition=f"unweighted mean of per-group {m} over groups present in the replicate "
                                          "(absent groups excluded from that replicate's mean)"))
    return dict(V=V, W=W, mult=mult, patients=patients, overall=pd.DataFrame(overall),
                paired=pd.DataFrame(paired), per_group=pd.DataFrame(group_rows), macro=pd.DataFrame(macro_rows),
                groups=groups, boots=boots, group_boots=group_boots)


def reconcile(fr, res, B, input_sha, expected_sha, frozen_path, expected_frozen_sha, allow_any_n):
    checks = []

    def add(name, ok, **detail):
        checks.append(dict(name=name, status="PASS" if ok else "FAIL", **detail))

    def add_warn(name, ok, **detail):
        checks.append(dict(name=name, status="PASS" if ok else "WARN", **detail))

    add("input_sha256_matches_expected", input_sha == expected_sha, observed=input_sha, expected=expected_sha)
    if allow_any_n:
        checks.append(dict(name="input_n_rows_equals_2778", status="WAIVED", observed=len(fr), expected=EXPECTED_N,
                           note="--allow-any-n passed (tests only); not enforced"))
        checks.append(dict(name="input_n_patients_equals_1876", status="WAIVED", observed=len(res["patients"]),
                           expected=EXPECTED_PATIENTS, note="--allow-any-n passed (tests only); not enforced"))
    else:
        add("input_n_rows_equals_2778", len(fr) == EXPECTED_N, observed=len(fr), expected=EXPECTED_N)
        add("input_n_patients_equals_1876", len(res["patients"]) == EXPECTED_PATIENTS,
            observed=len(res["patients"]), expected=EXPECTED_PATIENTS)
    add("queries_unique", not fr["query"].duplicated().any())
    add("patient_ids_nonempty", bool(fr["patient"].str.strip().ne("").all()))
    labels = set(fr["true_label"])
    strict_vals = set(fr["strict_majority_k5"])
    add("strict_k5_values_are_labels_or_ABSTAIN", strict_vals <= labels | {ABSTAIN},
        unexpected=sorted(strict_vals - labels - {ABSTAIN}))
    add("plurality_and_top1_never_ABSTAIN",
        not any((fr[c] == ABSTAIN).any() for c in ["top1", "plurality_k3", "plurality_k5", "plurality_k10"]))
    bin_cols = ["legacy_generous_k5_correct", "hit1", "hit3", "hit5", "hit10"]
    add("binary_columns_are_0_1", all(set(fr[c].unique()) <= {0, 1} for c in bin_cols))
    rank = rank_values(fr["first_relevant_rank"])
    cons = all(((rank > 0) & (rank <= k)).astype(int).tolist() == fr[f"hit{k}"].tolist() for k in (1, 3, 5, 10))
    add("first_relevant_rank_consistent_with_hit_k", cons, absent_at_10=int((rank == 0).sum()),
        note="absent@10 <=> hit10 == 0; mrr10 contribution 0 for absent@10")
    add("oracle_ge_plurality_k5_per_query",
        bool((res["V"]["oracle_generous_k5"] >= res["V"]["plurality_k5_accuracy"]).all()),
        note="the oracle tie rule credits every query that plurality@5 gets right")
    mism = rederive_from_top5(fr)
    add("rules_rederived_from_top5_labels", not mism, n_queries=len(fr), n_mismatches=len(mism),
        first_mismatches=mism[:20],
        note="top1, plurality@3/@5 (closest-rank tie-break), strict >=3/5 else ABSTAIN, oracle tied-max "
             "(correct and pred), hit1/3/5, precision5, first_relevant_rank<=5 re-derived independently")
    # overall point estimates vs plain column means computed independently
    direct = {"oracle_generous_k5": fr["legacy_generous_k5_correct"].mean(),
              "plurality_k5_accuracy": (fr["plurality_k5"] == fr["true_label"]).mean(),
              "strict_k5_accuracy": (fr["strict_majority_k5"] == fr["true_label"]).mean(),
              "strict_k5_abstain_rate": (fr["strict_majority_k5"] == ABSTAIN).mean(),
              "top1_accuracy": fr["hit1"].mean(), "hit3": fr["hit3"].mean(), "hit5": fr["hit5"].mean(),
              "hit10": fr["hit10"].mean(), "precision5": fr["precision5"].mean(),
              "precision10": fr["precision10"].mean(),
              "mrr10": fr["first_relevant_rank"].map(lambda x: 0.0 if x == ABSENT else 1.0 / int(x)).mean(),
              "plurality_k3_accuracy": (fr["plurality_k3"] == fr["true_label"]).mean(),
              "plurality_k10_accuracy": (fr["plurality_k10"] == fr["true_label"]).mean()}
    ov = res["overall"].set_index("estimand")["point"]
    diffs = {k: abs(float(ov[k]) - float(v)) for k, v in direct.items()}
    add("overall_points_equal_per_query_column_means", max(diffs.values()) <= TOL, max_abs_diff=max(diffs.values()))
    pd_ = res["paired"].set_index("estimand")["point"]
    pdiff = max(abs(pd_[k] - (ov[a] - ov[b])) for k, (a, b) in PAIRED.items())
    add("paired_points_equal_difference_of_overall_points", pdiff <= TOL, max_abs_diff=pdiff)
    # frozen per_group_summary.csv (A output)
    if not os.path.isfile(frozen_path):
        add("frozen_per_group_summary_present", False, path=frozen_path)
    else:
        fsha = sha256(frozen_path)
        add("frozen_per_group_summary_sha256_matches_expected", fsha == expected_frozen_sha, observed=fsha,
            expected=expected_frozen_sha, path=os.path.abspath(frozen_path))
        F = pd.read_csv(frozen_path, keep_default_na=False, na_values=[""]).set_index("true_label")
        worst, bad, n_cmp = 0.0, [], 0
        V = res["V"]
        for g in list(res["groups"]) + ["ALL"]:
            mask = np.ones(len(fr), bool) if g == "ALL" else (fr["true_label"] == g).to_numpy()
            if g not in F.index:
                bad.append(f"{g}:missing_in_frozen"); continue
            for col, vec in FROZEN_MAP.items():
                d = abs(float(V[vec][mask].mean()) - float(F.loc[g, col])); n_cmp += 1
                worst = max(worst, d)
                if not d <= TOL:
                    bad.append(f"{g}:{col}")
            for col, val in (("n_queries", int(mask.sum())), ("n_patients", int(fr.loc[mask, "patient"].nunique()))):
                n_cmp += 1
                if int(F.loc[g, col]) != val:
                    bad.append(f"{g}:{col}")
        extra = sorted(set(F.index) - set(res["groups"]) - {"ALL"})
        bad += [f"{g}:missing_in_input" for g in extra]
        add("points_equal_frozen_per_group_summary", not bad, n_compared=n_cmp, max_abs_diff=worst, mismatches=bad,
            tolerance=TOL)
        pg = res["per_group"].drop_duplicates("true_label").set_index("true_label")
        gd = []
        for m in PER_GROUP:
            sub = res["per_group"][res["per_group"]["metric"] == m].set_index("true_label")["point"]
            col = {"plurality_k5_recall": "plurality5_recall", "hit5": "hit5", "hit10": "hit10",
                   "strict_k5_abstain_rate": "strict5_abstain_rate"}[m]
            gd += [abs(sub[g] - float(F.loc[g, col])) for g in sub.index if g in F.index]
        add("per_group_output_points_equal_frozen", bool(gd) and max(gd) <= TOL and
            all(int(pg.loc[g, "n_queries"]) == int(F.loc[g, "n_queries"]) for g in pg.index if g in F.index),
            max_abs_diff=max(gd) if gd else None)
    # CI containment over every available interval
    tabs = [res["overall"], res["paired"], res["per_group"], res["macro"]]
    allci = pd.concat([t[["point", "ci_lo", "ci_hi"]] for t in tabs], ignore_index=True)
    avail = allci.dropna(subset=["ci_lo", "ci_hi"])
    inside = (avail["ci_lo"] <= avail["point"] + TOL) & (avail["point"] <= avail["ci_hi"] + TOL)
    add_warn("ci_contains_point_estimate_where_available", bool(inside.all()), n_available=len(avail),
             n_unavailable=int(len(allci) - len(avail)), n_not_containing=int((~inside).sum()),
             note="WARN, not FAIL: a percentile interval can legitimately exclude the point estimate in skewed "
                  "small groups; an unavailable (NaN) interval arises only when a replicate has zero weight in "
                  "the subset (ADR0021)")
    add("replicate_count_equals_B", res["mult"].shape[0] == B and all(int(t["n_replicates"].iloc[0]) == B
                                                                     for t in tabs if len(t)),
        observed=int(res["mult"].shape[0]), expected=B)
    add("each_replicate_draws_n_patients", bool((res["mult"].sum(1) == len(res["patients"])).all()))
    add("query_weights_equal_patient_multiplicity",
        bool(np.array_equal(res["W"].sum(1), (res["mult"] * fr.groupby("patient").size()
                                              .reindex(res["patients"]).to_numpy()).sum(1))))
    ng = res["per_group"].drop_duplicates("true_label")["n_queries"].sum()
    add("per_group_n_sums_to_n_rows", int(ng) == len(fr), observed=int(ng), expected=len(fr))
    if not allow_any_n:
        add("per_group_n_sums_to_2778", int(ng) == EXPECTED_N, observed=int(ng), expected=EXPECTED_N)
    reps, worst, n_cmp = spot_check(fr, res, B)
    add("brute_force_resample_spot_check", worst <= TOL, replicates=reps, n_compared=n_cmp, max_abs_diff=worst,
        tolerance=TOL, note="materialised resample from the explicit patient -> row mapping vs weight-matrix path; "
                            "all overall estimands and all per-group metrics")
    return checks


def write_readme(path, rec, res, fr, checks):
    ov = res["overall"]
    st = Counter(c["status"] for c in checks)
    pgp = res["per_group"].drop_duplicates("true_label")
    small = pgp[pgp["n_patients"] < 50].sort_values("n_patients")
    small_txt = ", ".join(f"{r.true_label} {r.n_patients}" for r in small.itertuples()) or "none"
    flagged = res["per_group"][res["per_group"]["boundary_flag"] != ""]
    ar = rec["frozen_a_reconciliation"]
    lines = [
        f"STATUS: COMPLETE - all {len(checks)} reconciliation checks: {st.get('PASS', 0)} PASS, "
        f"{st.get('WARN', 0)} WARN, {st.get('WAIVED', 0)} WAIVED, 0 FAIL (see reconciliation.json)",
        "",
        "# R2.3 / R4.b / E2: whole-patient bootstrap CIs for the Stream A per-query results",
        "",
        "Configuration (inherited, not re-run): UNI2 encoder / Optimal Transport cosine / unfiltered / "
        "2,778 COMET slides / 15 diagnostic groups (14 tumour + NORMAL; MISC dropped) / same-patient neighbours "
        "masked.",
        f"Input: `{rec['input']['path']}` (sha256 {rec['input']['sha256']}), {rec['input']['n_rows']} query slides "
        f"from {rec['input']['n_patients']} patients in {rec['input']['n_groups']} groups. No retrieval was re-run.",
        "",
        "## Unit of analysis and uncertainty",
        "- Every query slide is one row and contributes equally to point estimates (slide-level micro averages; "
        "they reproduce the accepted A per_group_summary.csv).",
        "- Several slides can come from one patient, so slides are not independent. Intervals therefore come from a "
        "**whole-patient bootstrap**: in each of B replicates the patients are drawn with replacement "
        f"(as many draws as patients; PCG64, B={rec['B']}, seed={rec['seed']}); every query slide is weighted by the "
        "number of times its patient was drawn; each estimand is recomputed as a weighted mean.",
        f"- Reported intervals are 95% percentile intervals: {PERCENTILE_METHOD}.",
        "- Paired differences are computed replicate-by-replicate on the same queries and the same draws.",
        "- A per-group interval is unavailable (NaN) if any replicate draws no patient of that group "
        "(ADR0021 convention, not dropped); `n_replicates_group_absent` counts such replicates.",
        f"- Macro averages are the unweighted mean over the {rec['input']['n_groups']} groups (for the accepted "
        "input: 15 groups including NORMAL); in a replicate where a group has zero queries, that group is excluded "
        "from that replicate's mean; `n_replicates_with_absent_group` counts how often.",
        f"- Draw function: {DRAWS_COPIED_FROM}. {DRAWS_SOURCE} ({DRAWS_SOURCE_NOTE}).",
        "",
        "## Caveat: small groups and boundary intervals",
        f"- Groups with fewer than 50 patients: {small_txt}. Percentile bootstrap intervals in groups of roughly "
        "23-41 patients can be too narrow (under-cover), so treat them as approximate.",
        "- `boundary_flag` marks intervals with lower bound 0 (`lower_eq_0`), upper bound 1 (`upper_eq_1`) or zero "
        "width (`zero_width`), or `ci_unavailable`. A bound at 0 or 1 means that at least 2.5% of resamples "
        "(all of them for [0, 0]) contained no counter-example; it reflects few patients/events, not certainty "
        "that the true rate is 0 or 1. "
        f"Flagged per-group rows in this run: {len(flagged)}"
        + (" (" + "; ".join(f"{r.true_label} {r.metric} {r.boundary_flag}" for r in flagged.itertuples()) + ")"
           if len(flagged) else "") + ".",
        "",
        "## Metric definitions (denominator: all query slides unless stated)",
    ]
    lines += [f"- `{k}`: {v}." for k, v in OVERALL.items()]
    lines += [f"- `{k}`: {a} minus {b} (paired)." for k, (a, b) in PAIRED.items()]
    lines += [
        "- Per group (true_label): `plurality_k5_recall` = share of that group's queries whose plurality@5 call is "
        "the true group; `hit5`/`hit10` = share with >= 1 same-group neighbour in the top 5/10; "
        "`strict_k5_abstain_rate` = share with no label holding >= 3 of the top 5. Denominator = that group's queries.",
        "- `macro_*`: unweighted mean of the per-group value over groups (each group, including NORMAL, counts once "
        "regardless of size).",
        "- Plurality@k: most frequent label among the top k, ties broken by the closest-ranked tied label; never "
        "uses the truth. Strict majority@5: a label with >= 3 of 5, else ABSTAIN (abstention is not correct).",
        "- Oracle generous@5 (`legacy_generous_k5_correct`): correct if the truth is among the tied-most-frequent "
        "top-5 labels, so it uses the truth to break ties; an upper bound, not a deployable decision rule. The "
        "accepted A record defines it as the legacy oracle `vote_k` rule of scripts/failure_modes_first_pass.py "
        f"(`{ar['path']}`, sha256 {ar['sha256']}, rules.legacy_generous_k5; all_match={ar['all_match']} over "
        f"{ar['n_reconciliations']} reconciliations against earlier first-pass summaries). Its identification with "
        "the manuscript's original success rule rests on that legacy lineage, not on a direct reconciliation with "
        "manuscript numbers in this script.",
        "- `first_relevant_rank` encodes no relevant neighbour in the top 10 as `absent@10`; such queries contribute 0 "
        "to `mrr10` (reciprocal rank truncated at 10).",
        "",
        "## Not computed (open scope decision)",
        "- mAP, nDCG and plurality macro-F1 are not reported here. The frame does contain the depth-5 relevance "
        "vector (`top5_labels` against `true_label`) and `library_support` (>= 5 for every query; minimum "
        f"{int(fr['library_support'].min())} here), so mAP@5, nDCG@5 and plurality@5 macro-F1 are computable from "
        "this frame alone. Only depth 6-10 versions (e.g. mAP@10, nDCG@10) need the upstream depth-10 rank file, "
        "because the frame does not keep the relevance of ranks 6-10 beyond the first relevant rank. Whether to "
        "report any of them is an open coordinator/author scope decision; they were deliberately not added.",
        "- Any patient-aggregated (one value per patient) estimand: estimands remain slide-level; only the "
        "uncertainty is patient-level.",
        "",
        "## Files",
        "- `summary_overall.csv`: overall estimands, point, ci_lo, ci_hi, boundary_flag.",
        "- `paired_differences.csv`: paired differences.",
        "- `per_group_summary.csv`: long format, one row per group x metric, with n_queries/n_patients, boundary_flag.",
        "- `macro_summary.csv`: macro averages over groups.",
        "- `record.json`: command, input sha256, counts, B, seed, percentile method, script and frozen A-script "
        "sha256, hostname, library versions, timestamp, reconciliation_all_pass.",
        "- `reconciliation.json`: named checks: PASS/FAIL; WARN only for CI containment; WAIVED only under "
        "--allow-any-n (tests only).",
        "- `COMPLETE`: written last on success. A failed run writes `FAILED`, record.json and reconciliation.json "
        "only (no result CSVs or README).",
        "",
        "## Overall results",
        "",
        "| estimand | point | 95% CI |",
        "|---|---|---|",
    ]
    lines += [f"| {r.estimand} | {r.point:.4f} | {r.ci_lo:.4f} - {r.ci_hi:.4f} |" for r in ov.itertuples()]
    lines += ["", f"Command: `{rec['command']}`", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--frozen-per-group", default=None,
                    help="frozen A per_group_summary.csv (default: next to --input)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--expected-sha256", default=ACCEPTED_INPUT_SHA256)
    ap.add_argument("--expected-frozen-sha256", default=ACCEPTED_FROZEN_PER_GROUP_SHA256)
    ap.add_argument("--allow-any-n", action="store_true", help="tests only: do not require 2,778 rows/1,876 patients")
    a = ap.parse_args(argv)
    if os.path.exists(a.outdir) and (not os.path.isdir(a.outdir) or os.listdir(a.outdir)):
        print(f"refusing to overwrite existing non-empty {a.outdir}", file=sys.stderr)
        return 2
    if a.B < 1:
        print("--B must be >= 1", file=sys.stderr)
        return 2
    fr, input_sha, n_bytes = read_input(a.input)
    if len(fr) != EXPECTED_N and not a.allow_any_n:
        print(f"input has {len(fr)} rows, expected {EXPECTED_N}; refusing (use --allow-any-n for tests)",
              file=sys.stderr)
        return 2
    in_dir = os.path.dirname(os.path.abspath(a.input))
    frozen = a.frozen_per_group or os.path.join(in_dir, "per_group_summary.csv")
    a_rec_path = os.path.join(in_dir, "reconciliation.json")
    a_rec = dict(path=a_rec_path, sha256=None, all_match=None, n_reconciliations=None)
    if os.path.isfile(a_rec_path):
        try:
            j = json.load(open(a_rec_path))
            a_rec.update(sha256=sha256(a_rec_path), all_match=j.get("all_match"),
                         n_reconciliations=len(j.get("reconciliation", [])))
        except (ValueError, AttributeError):
            a_rec.update(sha256=sha256(a_rec_path))
    res = compute(fr, a.B, a.seed)
    checks = reconcile(fr, res, a.B, input_sha, a.expected_sha256, frozen, a.expected_frozen_sha256, a.allow_any_n)
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    os.makedirs(a.outdir, exist_ok=True)
    me = os.path.abspath(__file__)
    rec = dict(
        timestamp=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        command=" ".join([sys.executable] + (sys.argv if argv is None else [me] + list(argv))),
        cwd=os.getcwd(), prosper_out=O, hostname=socket.gethostname(),
        script=dict(path=me, sha256=sha256(me)),
        frozen_a_script=dict(path=A_SCRIPT, sha256=sha256(A_SCRIPT) if os.path.isfile(A_SCRIPT) else None),
        input=dict(path=os.path.abspath(a.input), sha256=input_sha, n_bytes=n_bytes, n_rows=len(fr),
                   n_patients=len(res["patients"]), n_groups=len(res["groups"]),
                   hashing="input bytes read once; sha256 and CSV parse from the same in-memory bytes"),
        frozen_per_group=dict(path=os.path.abspath(frozen),
                              sha256=sha256(frozen) if os.path.isfile(frozen) else None),
        frozen_a_reconciliation=a_rec,
        B=a.B, seed=a.seed, rng="numpy.random.Generator(PCG64(seed))", interval="percentile 2.5/97.5",
        percentile_method=PERCENTILE_METHOD,
        allow_any_n=a.allow_any_n, draws_function=DRAWS_COPIED_FROM, draws_source=DRAWS_SOURCE,
        draws_source_note=DRAWS_SOURCE_NOTE,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        reconciliation_all_pass=n_fail == 0, n_checks=len(checks))
    with open(os.path.join(a.outdir, "record.json"), "w") as fh:
        json.dump(rec, fh, indent=2, default=_jsonable)
    with open(os.path.join(a.outdir, "reconciliation.json"), "w") as fh:
        json.dump(dict(all_pass=n_fail == 0, n_fail=n_fail, n_checks=len(checks), checks=checks), fh, indent=2,
                  default=_jsonable)
    for c in checks:
        print(f"{c['status']:7s} {c['name']}")
    if n_fail:
        with open(os.path.join(a.outdir, "FAILED"), "w") as fh:
            fh.write("\n".join(c["name"] for c in checks if c["status"] == "FAIL") + "\n")
        print(f"RECONCILIATION FAILED: {n_fail} check(s); result CSVs/README not written", file=sys.stderr)
        return 1
    res["overall"].to_csv(os.path.join(a.outdir, "summary_overall.csv"), index=False)
    res["paired"].to_csv(os.path.join(a.outdir, "paired_differences.csv"), index=False)
    res["per_group"].to_csv(os.path.join(a.outdir, "per_group_summary.csv"), index=False)
    res["macro"].to_csv(os.path.join(a.outdir, "macro_summary.csv"), index=False)
    write_readme(os.path.join(a.outdir, "README.md"), rec, res, fr, checks)
    with open(os.path.join(a.outdir, "COMPLETE"), "w") as fh:
        fh.write(f"{len(checks)} checks, 0 FAIL\n")
    print(f"all checks pass; outputs in {a.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
