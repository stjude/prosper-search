"""R4.a-vii: whole-patient bootstrap intervals for C1e, C1b and C1d.

The barred `pps_cluster_one_slide` intervals of three R4.a/E2 arms are replaced here
with whole-patient intervals. The specification is
`[internal record]`.

Arms:
- **C1e**: slide-level unmasked vs same-patient-masked retrieval, uni2 / ot_cosine.
- **C1b**: ROI-aggregated published (self-excluded) vs patient-excluded voting.
- **C1d**: same-patient consistency pairs vs class-matched null pairs.

Kernel: imported, not copied, from `scripts/r2_2_tile_ladder_whole_patient.py`, as
`scripts/r3_1_comparators_whole_patient.py` imports it. The functions used are
`make_patient_draws` (via `weight_matrix`), `boot_means`, `macro_reps`, `ci`, `plurality`
and `boundary_flag`.
- Patients are drawn with replacement: B = 5,000, PCG64 seed 123. Each query or pair is
  weighted by its patient's draw multiplicity.
- C1e and C1b share one set of draws over the 2,778 queries and 1,876 patients. The
  multiplicity array's sha256 is recorded per arm, so separate invocations can be shown
  to be identical.
- C1d draws over its own 463 multi-slide patients, the resampling unit of the saved
  producer.
- Paired differences are computed per replicate (rep_a - rep_b), never from the marginal
  intervals.
- Intervals are nominal 95 % percentile intervals, unadjusted. Including 0 is not
  evidence of equivalence.

Points and predictions come from the as-run prosper-search code, imported read-only:
- **C1e unmasked.** `slide_eval.predict_labels_from_matrix` and
  `compute_metrics_from_predictions` reproduce the eight saved C1e JSON points.
  Depth-10 lists are derived from the same matrix with the same self-exclusion and the
  stable mergesort.
- **C1e masked.** The masked side is the frozen `c2_metric_suite` ranked lists. Its k = 5
  predictions use the generous_majority and absolute rules, mirrored from
  `predict_labels_from_matrix`. The mirror is gated per query against the as-run function
  on the unmasked matrix, and its points are gated against the saved masked JSONs.
- **C1b.** `roi_eval.predict_slide_from_roi_neighbors`, the published arm, and
  `predict_slide_from_roi_neighbors_patient_excluded` are the per-slide functions behind
  `roi_cli`'s batch wrappers. They are called per slide, so per-slide predictions
  survive. Published-arm hit@5 comes from the patient-excluded function with an empty
  patient map. With an empty map that function is the published vote plus a hit@5
  column, which is gated per slide.
- **C1d.** The saved pair frame from `patient_consistency`.

Weighted F1 replicate:
- per class c, F1_c = 2 TP_c / (T_c + P_c), with weighted counts. An "Ambiguous"
  prediction (absolute rule) is a false negative and no false positive. A class with
  T_c + P_c = 0 gets 0, as sklearn does.
- macro = the mean over the 15 labels. The replicate is NaN if a label has zero true
  weight (kernel convention) and is counted in `n_nonfinite`.
- micro F1 (slide) = 2 sum TP / (sum T + sum P). Micro F1 (ROI) is the weighted accuracy,
  because `roi_eval` reports `accuracy_score`.
- At unit weights the replicate is gated to equal the as-run point to 1e-12.

Gates are per arm and independent. They run before that arm's bootstrap, except the
distinct-route C1e cross-check against the accepted r2_3 ext whole-patient intervals,
which needs the replicates.
- A failing arm contributes no interval rows. FAILED.json names the first failed gate and
  lists every gate record, and COMPLETE is not written.
- With `--c1b-limit N`, the C1b points cannot match the saved full-population values. The
  point/reconciliation gates are recorded as SKIPPED, the draws differ from production,
  and the C1b rows are labelled DEV and are not production numbers.
"""
import argparse
import datetime as dt
import gzip
import importlib.util
import io
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
OUT = os.environ.get("PROSPER_OUT", os.path.join(R, "prosper_revision_output"))
LADDER = os.path.join(HERE, "r2_2_tile_ladder_whole_patient.py")
PROSPER_SEARCH = os.path.realpath(os.path.join(R, "..", "prosper-search"))

spec = importlib.util.spec_from_file_location("ladder", LADDER)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

TAG = "r4a_vii_whole_patient_20261015"
COMET = "/PROSPER_WORKSPACE"
MATRIX_UNMASKED = f"{COMET}/comet-feature-gen/comet-uni2/output/similarity_matrix/combined_similarity_matrix_uni2_ot_cosine.pkl"
MATRIX_MASKED_CFG = f"{COMET}/comet-feature-gen/comet-uni2/output/similarity_matrix/combined_similarity_matrix_uni2_ot_cosine_no_same_patient.pkl"
META = L.META
LEGACY_META = f"{COMET}/comet-new-scans/data/final_slide_list/final_scan_id_disease_map.csv"
ROI_DIR = f"{COMET}/comet-new-scans/output/top_k_hit_slides_final_slides_final"
ID, LAB, PAT = "Slide.Scan.File", "Tumor.group", "UID.Subject"
AMBIG = "Ambiguous"
N_QUERIES, N_PATIENTS = 2778, 1876
POINT_TOL, CROSS_TOL, UNIT_TOL, C1D_TOL = 1e-6, 1e-9, 1e-12, 1e-9
TIE_STOP_FRACTION = 0.01
FAMILY = ["hit1", "hit3", "hit5", "hit10", "precision5", "precision10", "plurality5"]

C1E_JSON = {(k, rule): f"c1e_slide_unmasked/slide_ci_uni2_ot_cosine_top{k}_{rule}_SAME_PATIENT_ALLOWED.json"
            for k in (1, 3, 5, 10) for rule in ("generous_majority", "absolute")}
MASKED_RANKS = "c2_metric_suite/ranked_lists_uni2_ot_cosine.csv.gz"
MASKED_JSON = {"generous_majority": "g2_table_s3/slide_ci_uni2_ot_cosine_top5_generous_majority.json",
               "absolute": "c2_vote_rules_v2/slide_ci_uni2_ot_cosine_top5_absolute.json"}
R23_EXT = "r2_3_internal_failure_cis_ext_prod_20261005/summary_ext.csv"
R31_SUMMARY = "r3_1_comparators_whole_patient_20261008/summary.csv"
C1B_JSON = {(k, arm): f"c1b_roi_patient_excluded/roi_agg_k{k}_raw_hits{'_patient_excluded' if arm == 'patient_excluded' else ''}.json"
            for k in (5, 10) for arm in ("published", "patient_excluded")}
C1D_PAIRS = "c1d_patient_consistency/pairs_uni2_ot_cosine.csv"
C1D_JSON = "c1d_patient_consistency/consistency_uni2_ot_cosine.json"
C1D_METRICS = ["top1_label_agree", "top5_plurality_agree", "jaccard_top5_ids", "jaccard_top5_labels",
               "both_hit5", "split_hit5", "jsd_posterior_N5", "jsd_posterior_N18"]
C1D_EXTRA = ["both_generous5", "same_class"]
C1D_ROWS, C1D_PAIRS_EACH, C1D_PATIENTS = 3796, 1898, 463

# sha256 of every saved input as read by the analyst on 2026-10-15; a mismatch is a gate failure
EXPECTED_SHA256 = {
    MATRIX_UNMASKED: "f7e9b8a3584f2baaa9aca038d50b8cfc1bd186e764f92ecacd9d58c7e79bf41a",
    META: "45f1466c158f41ceea11abe06138356597a65ce1e97d2b072b780951864582b9",
    LEGACY_META: "210f8cf2f2e1ce3c10c848f2e336c98a5fdb69f2113aa83d3e4876d6bc8d95f8",
    MASKED_RANKS: "f2733faa8a44463ed64f483594a2ae048ffbf38a14b39272f124a6abe91ca96b",
    MASKED_JSON["generous_majority"]: "b5764bd4629fcfd5b427f4eb0950c347165a5d7961e1009afa638f4f4ef96416",
    MASKED_JSON["absolute"]: "a5f884edd2a43f9e02f00e311a21eac50be50b4b6c77716aa01a5754438f3140",
    R23_EXT: "65d86d1758ac2329640c2b604c2ee651b3f23c5d8446d1664342faccf3f5877a",
    R31_SUMMARY: "461bbb22560ed7c8b4575ff54caf8fe8f79889c1cce4afb0c8f9652de24b32cb",
    C1E_JSON[(10, "absolute")]: "9dd8df8887bce7d9489176f48024c2552041a27700b4aa60adfebd4913b8e43c",
    C1E_JSON[(10, "generous_majority")]: "9d839a2130a28e192c82c86989410cf4dee6aed7401970395d64064f9bac1170",
    C1E_JSON[(1, "absolute")]: "b542f5736e3309f5ec708a7607292bb4a85cf19a0c97b15e10bd4bfe11a5d403",
    C1E_JSON[(1, "generous_majority")]: "85ce4e2c906d7cb4485034af3b3555bd321e2b60fe998e95cdf05c19c8737432",
    C1E_JSON[(3, "absolute")]: "a116a9d94d7632dbbf648cd4fca4c5d10c20751901c4f51278b58f1475dbb863",
    C1E_JSON[(3, "generous_majority")]: "ccb5499748c45030f381fd529f1e66b4a9ae59802a5b1a22035d7fccecfaaec1",
    C1E_JSON[(5, "absolute")]: "1d1bf972a60a197c2306092bf6260939c37009ace3c72be939cd7cc92455cc8f",
    C1E_JSON[(5, "generous_majority")]: "095430dda84249c68f03fb2c8ad38d3111b78a310ba88e2f42480ae65caceb0c",
    C1B_JSON[(10, "published")]: "499b5b4efbbd3a66fbc370695c42e19f4cb68b4f081a18b2bc6cf13a9dcdac95",
    C1B_JSON[(10, "patient_excluded")]: "e34dac4b79ae770cb369703ebd42c48de24c5a58ac5ec1bdced2adf220e626e8",
    C1B_JSON[(5, "published")]: "f404c080b9de8d5c85d7cc79efa7536f57c8d1beb861d31b809bee87b8e4329f",
    C1B_JSON[(5, "patient_excluded")]: "9259360e591e577096f46e05b1a0cd39be218d4837909d52de51f8a46f7d517e",
    C1D_PAIRS: "808f8b74f9cd0c2708286b135411b57162123caa613661e8febf2a9769dc4c19",
    C1D_JSON: "34af92557d532cac7665f4fee74b31b4db30f7aa31fe24cb11a73c9433800360",
}
PROSPER_MODULES = ["prosper/evaluation/slide_eval.py", "prosper/evaluation/roi_eval.py", "prosper/evaluation/roi_cli.py",
                   "prosper/evaluation/patient_consistency.py", "prosper/evaluation/paths.py"]
PAIRED_DEF = ("minuend - subtrahend computed per replicate on the same units and the same patient draws; "
              "nominal 95% percentile, unadjusted; including 0 is not evidence of equivalence")


# ------------------------------------------------------------------ gates
class GateFailure(Exception):
    def __init__(self, arm, gate, detail):
        super().__init__(f"{arm}.{gate}: {detail}")
        self.arm, self.gate, self.detail = arm, gate, detail


class Gates:
    def __init__(self):
        self.records = []

    def check(self, arm, gate, ok, detail=None, **values):
        rec = dict(arm=arm, gate=gate, status="PASS" if ok else "FAIL", detail=detail, **values)
        self.records.append(rec)
        if not ok:
            raise GateFailure(arm, gate, detail)

    def skip(self, arm, gate, reason):
        self.records.append(dict(arm=arm, gate=gate, status="SKIPPED", detail=reason))

    def of(self, arm):
        return [r for r in self.records if r["arm"] == arm]


def resolve(rel):
    return rel if os.path.isabs(rel) else os.path.join(OUT, rel)


def bind(bindings, gates, arm, rel, role):
    path = resolve(rel)
    if not os.path.exists(path):
        gates.check(arm, f"input_present:{os.path.basename(path)}", False, f"missing input {path}")
    sha = L.sha256_bytes(L.read_bytes(path))
    exp = EXPECTED_SHA256.get(rel)
    bindings[path] = dict(role=role, arm=arm, sha256=sha, bytes=os.path.getsize(path), expected_sha256=exp,
                          matches_expected=(None if exp is None else sha == exp))
    if exp is not None:
        gates.check(arm, f"input_sha256:{os.path.basename(path)}", sha == exp, f"{path}: {sha} vs expected {exp}")
    return path, sha


def close(a, b, tol):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


# ------------------------------------------------------------------ weighted F1 / bootstrap helpers
def onehot(values, classes):
    v = np.asarray(values, dtype=object)
    return np.array([(v == c) for c in classes], dtype=float).T  # (n, C); labels outside classes -> zero row


def f1_reps(W, truth, pred, classes, micro="f1"):
    """(B,) macro and micro F1 replicates under weights W (B, n). micro='f1' (slide, sklearn micro over
    `classes`) or 'accuracy' (ROI evaluator). Macro replicate is NaN if a class has zero true weight."""
    T, P = onehot(truth, classes), onehot(pred, classes)
    TP, Tc, Pc = W @ (T * P), W @ T, W @ P
    den = Tc + Pc
    with np.errstate(invalid="ignore", divide="ignore"):
        f1c = np.where(den > 0, 2 * TP / np.where(den > 0, den, 1.0), 0.0)
    macro = f1c.mean(1)
    macro[(Tc == 0).any(1)] = np.nan
    if micro == "f1":
        d = Tc.sum(1) + Pc.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mic = 2 * TP.sum(1) / np.where(d > 0, d, np.nan)
    else:
        correct = (np.asarray(truth, dtype=object) == np.asarray(pred, dtype=object)).astype(float)
        mic = L.boot_means(W, correct)
    return macro, mic


def group_matrix(truth, classes):
    t = np.asarray(truth, dtype=object)
    return np.array([(t == c) for c in classes]).astype(float)  # (C, n)


def summary_row(arm, population, config, metric, average, point, reps, n_units, n_patients, n_groups, B, scope=True,
                role="", note=""):
    lo, hi, bad = L.ci(reps)
    return dict(arm=arm, population=population, config=config, metric=metric, average=average, point=point,
                ci_lo=lo, ci_hi=hi, n_queries=n_units, n_patients=n_patients, n_groups=n_groups, n_replicates=B,
                n_nonfinite=bad, flag=L.boundary_flag(lo, hi), in_contract_scope=scope, role=role, note=note)


def paired_row(arm, population, a, b, metric, average, point, reps, n_units, n_patients, B, scope=True, role="",
               note=""):
    lo, hi, bad = L.ci(reps)
    return dict(arm=arm, population=population, minuend=a, subtrahend=b, metric=metric, average=average,
                point=point, ci_lo=lo, ci_hi=hi, n_queries=n_units, n_patients=n_patients, n_replicates=B,
                n_nonfinite=bad, flag=L.boundary_flag(lo, hi),
                ci_includes_zero_nominal=bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0 <= hi),
                in_contract_scope=scope, role=role, definition=PAIRED_DEF, note=note)


def mult_sha(mult):
    return L.sha256_bytes(np.ascontiguousarray(mult).tobytes())


# ------------------------------------------------------------------ vote rules (mirrors of slide_eval)
def generous_majority(labels, truth):
    """Mirror of slide_eval.predict_labels_from_matrix tie_break='generous_majority' for a nonempty list."""
    c = Counter(labels)
    mx = max(c.values())
    maj = {lab for lab, v in c.items() if v == mx}
    if truth in maj:
        return truth
    return next(lab for lab in labels if lab in maj)


def absolute_majority(labels, k):
    """Mirror of tie_break='absolute': the unique label with >= floor(k/2)+1 votes, else Ambiguous."""
    need = k // 2 + 1
    reached = [lab for lab, v in Counter(labels).items() if v >= need]
    return reached[0] if len(reached) == 1 else AMBIG


def family_row(labs, t):
    return dict(hit1=int(t in labs[:1]), hit3=int(t in labs[:3]), hit5=int(t in labs[:5]), hit10=int(t in labs[:10]),
                precision5=sum(x == t for x in labs[:5]) / 5.0, precision10=sum(x == t for x in labs[:10]) / 10.0,
                plurality5=int(L.plurality(labs[:5]) == t), plurality5_pred=L.plurality(labs[:5]),
                generous5_pred=generous_majority(labs[:5], t), absolute5_pred=absolute_majority(labs[:5], 5))


def derive_depth(values, ids, depth=11):
    """Per row: self -> NaN, rank the finite cells ascending with the stable mergesort (as
    slide_eval._rank_neighbors with larger_is_better=False). Returns (index array (n, depth), distances)."""
    n = values.shape[0]
    idx = np.full((n, depth), -1, dtype=np.int64)
    dist = np.full((n, depth), np.nan)
    for i in range(n):
        s = values[i, :].astype(float).copy()
        s[i] = np.nan
        valid = np.where(~np.isnan(s))[0]
        order = valid[np.argsort(s[valid], kind="mergesort")][:depth]
        idx[i, :len(order)] = order
        dist[i, :len(order)] = s[order]
    return idx, dist


def tie_census(dist):
    d = dist
    return dict(n_queries=int(len(d)),
                tie_rank10_rank11=int((d[:, 9] == d[:, 10]).sum()),
                tie_rank5_rank6=int((d[:, 4] == d[:, 5]).sum()),
                any_tie_within_top10=int((np.diff(d[:, :10], axis=1) == 0).any(1).sum()),
                tie_rank1_rank2=int((d[:, 0] == d[:, 1]).sum()),
                tie_rule="stable mergesort on the matrix row order (filter_similarity_matrix keeps matrix order); "
                         "same as the as-run slide_eval ranking")


# ------------------------------------------------------------------ prosper import (lazy: ~1 min on this filesystem)
def prosper_modules():
    if PROSPER_SEARCH not in sys.path:
        sys.path.insert(0, PROSPER_SEARCH)
    from prosper.evaluation import slide_eval, roi_eval, roi_cli  # noqa: E402
    return slide_eval, roi_eval, roi_cli


def prosper_provenance():
    out = dict(path=PROSPER_SEARCH, modules={})
    for rel in PROSPER_MODULES:
        p = os.path.join(PROSPER_SEARCH, rel)
        out["modules"][rel] = L.sha256_bytes(L.read_bytes(p)) if os.path.exists(p) else None
    for key, cmd in (("head", ["git", "rev-parse", "HEAD"]),
                     ("status_of_modules", ["git", "status", "--porcelain", "--"] + PROSPER_MODULES)):
        try:
            out[key] = subprocess.check_output(cmd, cwd=PROSPER_SEARCH, stderr=subprocess.DEVNULL).decode().strip()
        except Exception as e:  # read-only provenance; absence is recorded, not fatal
            out[key] = f"unavailable: {e}"
    return out


# ------------------------------------------------------------------ C1e
def c1e_frames(ctx, gates, bindings):
    arm = "c1e"
    slide_eval, _, _ = prosper_modules()
    mpath, _ = bind(bindings, gates, arm, MATRIX_UNMASKED, "C1e unmasked distance matrix")
    meta_path, _ = bind(bindings, gates, arm, META, "slide metadata with UID.Subject")
    saved = {}
    for key, rel in C1E_JSON.items():
        p, _ = bind(bindings, gates, arm, rel, f"saved C1e unmasked JSON k={key[0]} {key[1]}")
        saved[key] = json.load(open(p))
    for rule, rel in MASKED_JSON.items():
        p, _ = bind(bindings, gates, arm, rel, f"saved masked k=5 {rule} JSON")
        saved[("masked", rule)] = json.load(open(p))
    rpath, _ = bind(bindings, gates, arm, MASKED_RANKS, "masked ranked lists (frozen c2_metric_suite)")
    xpath, _ = bind(bindings, gates, arm, R23_EXT, "accepted whole-patient masked intervals (distinct-route check)")
    x31, _ = bind(bindings, gates, arm, R31_SUMMARY, "accepted r3_1 whole-patient masked family intervals (distinct-route check)")

    bad_cfg = []
    for (k, rule), d in ((key, saved[key]) for key in C1E_JSON):
        c = d["config"]
        exp = dict(input_similarity_matrix=MATRIX_UNMASKED, input_metadata=META, larger_is_better=False, top_k=k,
                   tie_break=rule, dropped_labels=["MISC"], same_patient_masked_input=False, model="uni2",
                   method="ot_cosine", tumor_only=False)
        bad_cfg += [f"top{k}_{rule}.{f}={c.get(f)!r}" for f, v in exp.items() if c.get(f) != v]
    for rule in MASKED_JSON:
        c = saved[("masked", rule)]["config"]
        exp = dict(input_similarity_matrix=MATRIX_MASKED_CFG, input_metadata=META, larger_is_better=False, top_k=5,
                   tie_break=rule, dropped_labels=["MISC"], same_patient_masked_input=True)
        bad_cfg += [f"masked_{rule}.{f}={c.get(f)!r}" for f, v in exp.items() if c.get(f) != v]
    gates.check(arm, "saved_config_bindings", not bad_cfg, "; ".join(bad_cfg) or "all saved configs name the bound inputs")

    sim = slide_eval.load_similarity_matrix(mpath)
    meta = slide_eval.load_metadata(meta_path, label_col=LAB, id_col=ID, patient_col=PAT, drop_labels=["MISC"])
    sim_f, meta_f = slide_eval.filter_similarity_matrix(sim, meta, ID)
    gates.check(arm, "population", sim_f.shape == (N_QUERIES, N_QUERIES) and len(meta_f) == N_QUERIES,
                f"filtered matrix {sim_f.shape}, metadata {len(meta_f)}")
    labels_order = sorted(meta_f[LAB].unique().tolist())
    gates.check(arm, "labels", labels_order == sorted(L.CLASSES), f"{labels_order}")

    preds, mism = {}, []
    for (k, rule), d in ((key, saved[key]) for key in C1E_JSON):
        dfp = slide_eval.predict_labels_from_matrix(sim_f, meta_f, k, ID, LAB, PAT, larger_is_better=False,
                                                    tie_break=rule, exclude_same_patient=False)
        m = slide_eval.compute_metrics_from_predictions(dfp, labels_order)
        preds[(k, rule)] = dfp
        for met in ("macro_f1", "micro_f1"):
            ok = close(m[met], d["point_estimates"][met], POINT_TOL)
            if not ok:
                mism.append(f"top{k}_{rule}.{met} {m[met]} vs saved {d['point_estimates'][met]}")
        if d.get("labels_order") is not None and list(d["labels_order"]) != labels_order:
            mism.append(f"top{k}_{rule}.labels_order")
        if int(d.get("n_slides", m["n_slides"])) != m["n_slides"]:
            mism.append(f"top{k}_{rule}.n_slides {m['n_slides']} vs {d.get('n_slides')}")
    gates.check(arm, "fidelity_unmasked_saved_points", not mism, "; ".join(mism) or
                "8 saved unmasked JSONs: macro and micro F1 reproduced to <= 1e-6",
                unmasked_generous5_macro=float(slide_eval.compute_metrics_from_predictions(
                    preds[(5, "generous_majority")], labels_order)["macro_f1"]))

    ids = sim_f.index.tolist()
    lab = dict(zip(meta_f[ID], meta_f[LAB]))
    idx, dist = derive_depth(sim_f.values, ids, depth=11)
    tc = tie_census(dist)
    frac = tc["tie_rank10_rank11"] / tc["n_queries"]
    gates.check(arm, "tie_census_depth10", frac <= TIE_STOP_FRACTION,
                f"{tc['tie_rank10_rank11']} of {tc['n_queries']} queries tie at the rank-10/11 boundary "
                f"({frac:.4%}); stop threshold 1%", **tc)
    top = {q: [lab[ids[j]] for j in idx[i, :10]] for i, q in enumerate(ids)}
    bad = 0
    for k in (5, 10):
        dfp = preds[(k, "generous_majority")]
        for q, tl in zip(dfp[ID], dfp["topk_labels"]):
            bad += int(list(tl) != top[q][:k])
    gates.check(arm, "derived_lists_match_as_run_topk", bad == 0, f"{bad} query/k mismatches (k = 5, 10)")
    mis = 0
    for rule in ("generous_majority", "absolute"):
        dfp = preds[(5, rule)]
        for q, t, p in zip(dfp[ID], dfp["true_label"], dfp["pred_label"]):
            mine = generous_majority(top[q][:5], t) if rule == "generous_majority" else absolute_majority(top[q][:5], 5)
            mis += int(mine != p)
    gates.check(arm, "mirror_rules_match_as_run", mis == 0,
                f"{mis} per-query disagreements between the mirrored k=5 rules and predict_labels_from_matrix")

    pat = L.load_meta(meta_path)[0][PAT]
    rows = []
    for q in ids:
        t = lab[q]
        r = dict(query=q, true_label=t, patient=pat[q], top10="|".join(top[q]))
        r.update(family_row(top[q], t))
        rows.append(r)
    fu = pd.DataFrame(rows).set_index("query").sort_index()
    asrun = {rule: dict(zip(preds[(5, rule)][ID], preds[(5, rule)]["pred_label"])) for rule in ("generous_majority", "absolute")}
    fu["generous5_pred"] = [asrun["generous_majority"][q] for q in fu.index]
    fu["absolute5_pred"] = [asrun["absolute"][q] for q in fu.index]

    lmeta = L.load_meta(meta_path)[0]
    rk, rsha = L.load_ranks(rpath, lmeta)
    rows = []
    for q, g in rk.groupby("query", sort=True):
        labs, t = g["neighbour_label"].tolist(), g["true_label"].iloc[0]
        r = dict(query=q, true_label=t, patient=lmeta[PAT][q], top10="|".join(labs))
        r.update(family_row(labs, t))
        rows.append(r)
    fm = pd.DataFrame(rows).set_index("query").sort_index()
    ok = (fm.index.equals(fu.index) and (fm["true_label"] == fu["true_label"]).all()
          and (fm["patient"] == fu["patient"]).all() and fu["patient"].nunique() == N_PATIENTS)
    gates.check(arm, "arm_alignment", ok, f"masked {len(fm)} vs unmasked {len(fu)} queries; "
                f"patients {fu['patient'].nunique()}")
    mism, asrun_pts = [], {}
    for rule in ("generous_majority", "absolute"):
        m = slide_eval.compute_metrics_from_predictions(preds[(5, rule)], labels_order)
        asrun_pts[("unmasked", rule)] = {kk: m[kk] for kk in ("macro_f1", "micro_f1")}
    for rule, col in (("generous_majority", "generous5_pred"), ("absolute", "absolute5_pred")):
        dfp = pd.DataFrame({"true_label": fm["true_label"], "pred_label": fm[col]})
        m = slide_eval.compute_metrics_from_predictions(dfp, labels_order)
        asrun_pts[("masked", rule)] = {kk: m[kk] for kk in ("macro_f1", "micro_f1")}
        sp = saved[("masked", rule)]["point_estimates"]
        for met in ("macro_f1", "micro_f1"):
            if not close(m[met], sp[met], POINT_TOL):
                mism.append(f"masked_{rule}.{met} {m[met]} vs saved {sp[met]}")
    gates.check(arm, "fidelity_masked_saved_points", not mism, "; ".join(mism) or
                "masked k=5 generous (0.7912) and absolute (0.7439) macro/micro F1 reproduced to <= 1e-6")
    return dict(frames={"unmasked": fu, "masked": fm}, labels=labels_order, saved=saved, ties=tc, asrun=asrun_pts,
                masked_ranks_sha=rsha, xpath=xpath, x31=x31)


def c1e_boot(ctx, gates, d, B, seed):
    arm, pop = "c1e", "c1e_all_tissue_2778"
    fu, fm, classes = d["frames"]["unmasked"], d["frames"]["masked"], d["labels"]
    patients, mult, W = L.weight_matrix(fu["patient"], B, seed)
    n, npat = len(fu), len(patients)
    G = group_matrix(fu["true_label"], classes)
    one = np.ones((1, n))
    pts, reps, summ, paired = {}, {}, [], []
    for tag, fr in (("unmasked", fu), ("masked", fm)):
        for rule, col, role in (("generous5", "generous5_pred", "legacy estimand (primary)"),
                                ("absolute5", "absolute5_pred", "legacy estimand (secondary, declared before results)")):
            m_unit, mi_unit = f1_reps(one, fr["true_label"], fr[col], classes)
            mac, mic = f1_reps(W, fr["true_label"], fr[col], classes)
            for avg, u, rr in (("macro", m_unit, mac), ("micro", mi_unit, mic)):
                pts[(tag, f"f1_{rule}", avg)], reps[(tag, f"f1_{rule}", avg)] = float(u[0]), rr
                summ.append(summary_row(arm, pop, tag, f"f1_{rule}", avg, float(u[0]), rr, n, npat, len(classes), B,
                                        role=role))
        for m in FAMILY:
            v = fr[m].to_numpy(float)
            pts[(tag, m, "micro")] = float(v.mean())
            pts[(tag, m, "macro")] = float(np.mean([v[G[i] > 0].mean() for i in range(len(classes))]))
            reps[(tag, m, "micro")], reps[(tag, m, "macro")] = L.boot_means(W, v), L.macro_reps(W, G, v)
            for avg in ("micro", "macro"):
                summ.append(summary_row(arm, pop, tag, m, avg, pts[(tag, m, avg)], reps[(tag, m, avg)], n, npat,
                                        len(classes), B, role="batch-3 family"))
    # unit-weight gate: replicate machinery at unit weights equals the as-run points
    s = d["saved"]
    unit_bad = []
    for key, saved_val in ((("unmasked", "f1_generous5", "macro"), s[(5, "generous_majority")]["point_estimates"]["macro_f1"]),
                           (("unmasked", "f1_generous5", "micro"), s[(5, "generous_majority")]["point_estimates"]["micro_f1"]),
                           (("unmasked", "f1_absolute5", "macro"), s[(5, "absolute")]["point_estimates"]["macro_f1"]),
                           (("unmasked", "f1_absolute5", "micro"), s[(5, "absolute")]["point_estimates"]["micro_f1"]),
                           (("masked", "f1_generous5", "macro"), s[("masked", "generous_majority")]["point_estimates"]["macro_f1"]),
                           (("masked", "f1_generous5", "micro"), s[("masked", "generous_majority")]["point_estimates"]["micro_f1"]),
                           (("masked", "f1_absolute5", "macro"), s[("masked", "absolute")]["point_estimates"]["macro_f1"]),
                           (("masked", "f1_absolute5", "micro"), s[("masked", "absolute")]["point_estimates"]["micro_f1"])):
        if not close(pts[key], saved_val, POINT_TOL):
            unit_bad.append(f"{key}: {pts[key]} vs {saved_val}")
    for (tag, rule), pv in d["asrun"].items():
        short = "f1_generous5" if rule == "generous_majority" else "f1_absolute5"
        for kk, avg in (("macro_f1", "macro"), ("micro_f1", "micro")):
            if not close(pts[(tag, short, avg)], pv[kk], UNIT_TOL):
                unit_bad.append(f"{(tag, short, avg)}: {pts[(tag, short, avg)]} vs as-run {pv[kk]} (1e-12)")
    gates.check(arm, "unit_weight_replicate_equals_point", not unit_bad, "; ".join(unit_bad) or
                "weighted-F1 replicate at unit weights equals the as-run sklearn points to <= 1e-12 and the saved "
                "points to <= 1e-6")
    for m in ["f1_generous5", "f1_absolute5"] + FAMILY:
        for avg in ("macro", "micro"):
            dd = reps[("unmasked", m, avg)] - reps[("masked", m, avg)]
            paired.append(paired_row(arm, pop, "unmasked", "masked", m, avg,
                                     pts[("unmasked", m, avg)] - pts[("masked", m, avg)], dd, n, npat, B,
                                     role="legacy estimand" if m.startswith("f1_") else "batch-3 family"))
    # distinct-route check against the accepted r2_3 ext whole-patient intervals (same kernel draws expected)
    ext = pd.read_csv(d["xpath"]).set_index("metric")
    xc = {}
    for mine, theirs in ((("masked", "f1_generous5", "macro"), "generous_macro_f1"),
                         (("masked", "f1_absolute5", "macro"), "strict_macro_f1")):
        lo, hi, _ = L.ci(reps[mine])
        row = ext.loc[theirs]
        xc[theirs] = dict(recomputed=[lo, hi], recomputed_point=pts[mine], accepted=[float(row["lo"]), float(row["hi"])],
                          accepted_point=float(row["point"]),
                          max_abs_diff=float(max(abs(lo - row["lo"]), abs(hi - row["hi"]), abs(pts[mine] - row["point"]))))
    gates.check(arm, "cross_check_r2_3_ext", all(v["max_abs_diff"] <= CROSS_TOL for v in xc.values()),
                json.dumps({k: v["max_abs_diff"] for k, v in xc.items()}), results=xc)
    r31 = pd.read_csv(d["x31"])
    r31 = r31[(r31["population"] == "all_tissue_2778") & (r31["config"] == "primary")].set_index(["metric", "average"])
    diffs = {}
    for m in FAMILY:
        for avg in ("micro", "macro"):
            lo, hi, _ = L.ci(reps[("masked", m, avg)])
            row = r31.loc[(m, avg)]
            diffs[f"{m}_{avg}"] = float(max(abs(pts[("masked", m, avg)] - row["point"]), abs(lo - row["ci_lo"]),
                                            abs(hi - row["ci_hi"])))
    xc["r3_1_primary_masked_family_max_abs_diff"] = diffs
    gates.check(arm, "cross_check_r3_1_masked_family", len(diffs) == 14 and max(diffs.values()) <= CROSS_TOL,
                f"14 masked family point/interval rows vs r3_1 primary; max |diff| {max(diffs.values())}")
    legacy = {}
    for key, tag in (((5, "generous_majority"), "unmasked_generous5"), ((5, "absolute"), "unmasked_absolute5"),
                     (("masked", "generous_majority"), "masked_generous5"), (("masked", "absolute"), "masked_absolute5")):
        dj = s[key]
        cfg, ci_ = dj["config"], dj.get("confidence_intervals", {})
        legacy[tag] = {"label": "pps-legacy", "bootstrap_mode": cfg.get("bootstrap_mode"), "B": cfg.get("bootstrap_B"),
                       "seed": cfg.get("seed"), "point_estimates": dj["point_estimates"],
                       "macro_f1_ci": ci_.get("macro_f1_ci"), "micro_f1_ci": ci_.get("micro_f1_ci")}
        conf, rule = ("unmasked" if key[0] == 5 else "masked"), ("f1_generous5" if key[1] == "generous_majority" else "f1_absolute5")
        for avg in ("macro", "micro"):
            lo, hi, _ = L.ci(reps[(conf, rule, avg)])
            legacy[tag][f"new_whole_patient_{avg}"] = dict(point=pts[(conf, rule, avg)], ci=[lo, hi],
                                                          flag=L.boundary_flag(lo, hi))
    frame = pd.concat({"c1e_unmasked": fu, "c1e_masked": fm}, names=["config", "query"])
    return dict(summary=summ, paired=paired, frame=frame, n=n, npat=npat, mult_sha=mult_sha(mult),
                legacy=legacy, cross_check=xc, ties=d["ties"], patients=patients)


# ------------------------------------------------------------------ C1b
_C1B = {}


def _c1b_worker(sid):
    _, roi_eval, _ = prosper_modules()
    c = _C1B
    path = os.path.join(c["dir"], f"top_10_hit_slides_slide_{sid}.pkl")
    out = dict(slide=sid, pickle_sha256=L.sha256_bytes(L.read_bytes(path)) if os.path.exists(path) else None)
    for k in (5, 10):
        pub = roi_eval.predict_slide_from_roi_neighbors(sid, c["dir"], c["s2l_legacy"], k, "raw_hits")
        pubh = roi_eval.predict_slide_from_roi_neighbors_patient_excluded(sid, c["dir"], c["s2l_legacy"], {}, k, "raw_hits")
        exc = roi_eval.predict_slide_from_roi_neighbors_patient_excluded(sid, c["dir"], c["s2l_pat"], c["s2p"], k, "raw_hits")
        for tag, rec in (("pub", pub), ("pubh", pubh), ("exc", exc)):
            if rec is None:
                out[f"{tag}{k}"] = None
                continue
            out[f"{tag}{k}"] = {kk: rec.get(kk) for kk in ("slide", "true_label", "pred_label", "n_rois", "skipped_rois",
                                                            "total_hits_seen", "total_hits_counted", "missing_label_hits",
                                                            "same_patient_hits_dropped", "same_patient_hits_surviving",
                                                            "hit_at_5")}
    return out


def c1b_predict(ctx, gates, bindings, limit, n_workers):
    arm = "c1b"
    _, roi_eval, roi_cli = prosper_modules()
    meta_path, _ = bind(bindings, gates, arm, META, "slide metadata with UID.Subject (patient-excluded arm)")
    leg_path, _ = bind(bindings, gates, arm, LEGACY_META, "legacy patient-free ROI metadata (published arm)")
    saved = {}
    for key, rel in C1B_JSON.items():
        p, _ = bind(bindings, gates, arm, rel, f"saved C1b JSON k={key[0]} {key[1]}")
        saved[key] = json.load(open(p))
    if not os.path.isdir(ROI_DIR):
        gates.check(arm, "input_present:roi_dir", False, f"missing {ROI_DIR}")
    bad_cfg = []
    for (k, a_), dj in saved.items():
        c = dj["config"]
        exp = dict(top_k_per_roi=k, vote_mode="raw_hits", input_roi_hits_dir=ROI_DIR,
                   input_metadata=(META if a_ == "patient_excluded" else LEGACY_META), id_col=ID, label_col=LAB,
                   dropped_labels=["MISC"], exclude_same_patient=(a_ == "patient_excluded"))
        if a_ == "patient_excluded":
            exp["patient_col"] = PAT
        bad_cfg += [f"k{k}_{a_}.{f}={c.get(f)!r}" for f, v in exp.items() if c.get(f) != v]
    gates.check(arm, "saved_config_bindings", not bad_cfg, "; ".join(bad_cfg) or "all saved configs name the bound inputs")

    meta_pat = roi_eval.load_metadata(meta_path, ID, LAB, drop_labels=["MISC"], patient_col=PAT)
    meta_full = roi_eval.load_metadata(meta_path, ID, LAB, drop_labels=None, patient_col=PAT)
    meta_leg = roi_eval.load_metadata(leg_path, ID, LAB, drop_labels=["MISC"])
    rec = roi_cli.masked_pair_reconciliation(meta_full, meta_pat, ID, PAT)
    ev, fu = rec["roi_evaluated_population"], rec["full_population"]
    ok = (ev["n_slides"], ev["n_patients"], ev["n_multi_slide_patients"], ev["n_within_patient_pairs"]) == (N_QUERIES, N_PATIENTS, 463, 1898) \
        and (fu["n_slides"], fu["n_patients"], fu["n_multi_slide_patients"], fu["n_within_patient_pairs"]) == (2921, 1993, 477, 1949)
    gates.check(arm, "masked_pair_reconciliation", ok, json.dumps(rec, default=str), result=rec)
    gates.check(arm, "saved_masked_pair_reconciliation_matches",
                saved[(5, "patient_excluded")].get("masked_pair_reconciliation", {}).get("roi_evaluated_population") == ev,
                "saved k5 patient-excluded JSON roi_evaluated_population equals the recount")
    files = set(roi_eval.discover_slides_with_files(ROI_DIR))
    ids_all = sorted(set(meta_pat[ID]) & files)
    gates.check(arm, "population_files", len(ids_all) == N_QUERIES and set(meta_pat[ID]) <= files,
                f"{len(ids_all)} metadata slides with ROI pickles (metadata {len(meta_pat)})")
    leg_ids = sorted(set(meta_leg[ID]) & files)
    gates.check(arm, "published_population_equal", leg_ids == ids_all,
                f"legacy-metadata population {len(leg_ids)} vs patient-metadata population {len(ids_all)}")
    fb = roi_cli.count_glob_fallbacks(ids_all, ROI_DIR)
    gates.check(arm, "glob_fallbacks_zero", fb["n_would_use_glob_fallback"] == 0 and fb["n_no_file_at_all"] == 0,
                json.dumps(fb))
    s2l_pat = dict(zip(meta_pat[ID], meta_pat[LAB]))
    s2l_leg = dict(zip(meta_leg[ID], meta_leg[LAB]))
    lab_dis = [s for s in ids_all if s2l_pat.get(s) != s2l_leg.get(s)]
    gates.check(arm, "labels_agree_legacy_vs_patient_metadata", not lab_dis, f"{len(lab_dis)} disagreements")
    lmeta = L.load_meta(META)[0]
    s2p = dict(zip(meta_pat[ID], meta_pat[PAT]))
    pat_dis = [s for s in ids_all if s not in lmeta.index or lmeta.loc[s, PAT] != s2p[s]]
    gates.check(arm, "patients_agree_kernel_metadata", not pat_dis, f"{len(pat_dis)} disagreements; "
                f"queries without a patient: {sum(1 for s in ids_all if not s2p.get(s))}")
    slides = ids_all[:limit] if limit else ids_all
    _C1B.update(dir=ROI_DIR, s2l_legacy=s2l_leg, s2l_pat=s2l_pat, s2p=s2p)
    t0, c0 = time.time(), time.process_time()
    ctx_mp = mp.get_context("fork")
    with ctx_mp.Pool(processes=n_workers) as pool:
        res = list(pool.imap(_c1b_worker, slides, chunksize=1))
    wall, cpu_parent = time.time() - t0, time.process_time() - c0
    ru = __import__("resource").getrusage(__import__("resource").RUSAGE_CHILDREN)
    timing = dict(n_slides=len(slides), n_workers=n_workers, wall_seconds=wall, parent_cpu_seconds=cpu_parent,
                  children_cpu_seconds_cumulative=ru.ru_utime + ru.ru_stime)
    none = [r["slide"] for r in res if any(r[f"{t}{k}"] is None for t in ("pub", "pubh", "exc") for k in (5, 10))]
    gates.check(arm, "every_slide_predicted", not none, f"{len(none)} slides returned None")
    diff = []
    for r in res:
        for k in (5, 10):
            a, b = r[f"pub{k}"], r[f"pubh{k}"]
            for f in ("slide", "true_label", "pred_label", "n_rois", "skipped_rois", "total_hits_seen",
                      "total_hits_counted", "missing_label_hits"):
                if a[f] != b[f]:
                    diff.append(f"{r['slide']} k{k} {f}")
            if b["same_patient_hits_dropped"] != 0 or b["same_patient_hits_surviving"] != 0:
                diff.append(f"{r['slide']} k{k} empty-map drop counters nonzero")
    gates.check(arm, "published_hit5_route_identical", not diff,
                f"{len(diff)} differences between predict_slide_from_roi_neighbors and the empty-patient-map route")
    surv = sum(r[f"exc{k}"]["same_patient_hits_surviving"] for r in res for k in (5, 10))
    gates.check(arm, "same_patient_hits_surviving_zero", surv == 0, f"{surv} surviving same-patient hits")
    rows = []
    for r in res:
        for k in (5, 10):
            for tag, src in (("published", "pubh"), ("patient_excluded", "exc")):
                x = dict(r[f"{src}{k}"])
                x.update(config=f"{tag}_k{k}", query=r["slide"], patient=s2p[r["slide"]])
                if tag == "published":
                    x["pred_label_published_fn"] = r[f"pub{k}"]["pred_label"]
                rows.append(x)
    fr = pd.DataFrame(rows)
    fr["correct"] = (fr["true_label"] == fr["pred_label"]).astype(int)
    manifest = {r["slide"]: r["pickle_sha256"] for r in res}
    return dict(frame=fr, saved=saved, timing=timing, manifest=manifest, limit=limit, slides=slides,
                recon=rec, fallbacks=fb)


def c1b_boot(ctx, gates, d, B, seed):
    _, roi_eval, _ = prosper_modules()
    arm = "c1b"
    fr, saved, limit = d["frame"], d["saved"], d["limit"]
    pop = f"c1b_roi_DEV_first{limit}" if limit else "c1b_roi_2778"
    base = fr[fr["config"] == "published_k5"].set_index("query").sort_index()
    classes = sorted(base["true_label"].unique().tolist())
    patients, mult, W = L.weight_matrix(base["patient"], B, seed)
    n, npat = len(base), len(patients)
    G = group_matrix(base["true_label"], classes)
    one = np.ones((1, n))
    pts, reps, summ, paired, legacy, gate_bad = {}, {}, [], [], {}, []
    diag_bad = []
    for k in (5, 10):
        for tag in ("published", "patient_excluded"):
            cfg = f"{tag}_k{k}"
            f = fr[fr["config"] == cfg].set_index("query").sort_index()
            if not (f.index.equals(base.index) and (f["patient"] == base["patient"]).all()
                    and (f["true_label"] == base["true_label"]).all()):
                gate_bad.append(f"{cfg} not aligned with published_k5")
                continue
            m = roi_eval.compute_metrics_from_predictions(f[["true_label", "pred_label"]].reset_index(drop=True),
                                                          labels_order=classes)
            mac_u, mic_u = f1_reps(one, f["true_label"], f["pred_label"], classes, micro="accuracy")
            if not (abs(mac_u[0] - m["macro_f1"]) <= UNIT_TOL and abs(mic_u[0] - m["micro_f1"]) <= UNIT_TOL):
                gate_bad.append(f"{cfg} unit-weight replicate {mac_u[0]}/{mic_u[0]} vs as-run {m['macro_f1']}/{m['micro_f1']}")
            mac, mic = f1_reps(W, f["true_label"], f["pred_label"], classes, micro="accuracy")
            h = f["hit_at_5"].to_numpy(float)
            vals = {("f1", "macro"): (m["macro_f1"], mac), ("f1", "micro"): (m["micro_f1"], mic),
                    ("hit_at_5", "micro"): (float(h.mean()), L.boot_means(W, h)),
                    ("hit_at_5", "macro"): (float(np.mean([h[G[i] > 0].mean() for i in range(len(classes))])),
                                            L.macro_reps(W, G, h))}
            for (met, avg), (p, rr) in vals.items():
                pts[(cfg, met, avg)], reps[(cfg, met, avg)] = float(p), rr
                summ.append(summary_row(arm, pop, cfg, met, avg, float(p), rr, n, npat, len(classes), B,
                                        role=("primary" if k == 5 else "sensitivity") + (" (DEV subset)" if limit else ""),
                                        note="micro F1 = accuracy (roi_eval)" if (met, avg) == ("f1", "micro") else ""))
            sj = saved[(k, tag)]
            diag = {c: int(f[c].sum()) for c in ("n_rois", "skipped_rois", "total_hits_seen", "total_hits_counted",
                                                 "missing_label_hits", "same_patient_hits_dropped",
                                                 "same_patient_hits_surviving") if c in f.columns}
            if tag == "published":
                diag = {c: v for c, v in diag.items() if not c.startswith("same_patient")}
            legacy[cfg] = {"label": "pps-legacy" if tag == "patient_excluded" else "iid_within_class-legacy (as published; not pps)",
                           "bootstrap_mode": sj["config"].get("bootstrap_mode"), "B": sj["config"].get("bootstrap_B"),
                           "seed": sj["config"].get("seed"), "point_estimates": sj["point_estimates"],
                           "macro_f1_ci": sj["confidence_intervals"].get("macro_f1_ci"),
                           "micro_f1_ci": sj["confidence_intervals"].get("micro_f1_ci"),
                           "hit_at_5_rate": sj.get("hit_at_5_rate"), "saved_roi_diagnostics": sj.get("roi_diagnostics"),
                           "recomputed_roi_diagnostics": diag, "recomputed_point_estimates": {
                               kk: m[kk] for kk in ("macro_f1", "micro_f1", "macro_precision", "macro_recall")},
                           "recomputed_hit_at_5_rate": float(h.mean())}
            for met, avg in vals:
                lo, hi, _ = L.ci(reps[(cfg, met, avg)])
                legacy[cfg][f"new_whole_patient_{met}_{avg}"] = dict(point=pts[(cfg, met, avg)], ci=[lo, hi],
                                                                    flag=L.boundary_flag(lo, hi))
            if not limit:
                sp = sj["point_estimates"]
                for kk in ("macro_f1", "micro_f1", "macro_precision", "macro_recall"):
                    if not close(m[kk], sp[kk], POINT_TOL):
                        diag_bad.append(f"{cfg}.{kk} {m[kk]} vs saved {sp[kk]}")
                if sj.get("hit_at_5_rate") is not None and not close(h.mean(), sj["hit_at_5_rate"], POINT_TOL):
                    diag_bad.append(f"{cfg}.hit_at_5_rate {h.mean()} vs saved {sj['hit_at_5_rate']}")
                if diag != sj.get("roi_diagnostics"):
                    diag_bad.append(f"{cfg}.roi_diagnostics {diag} vs saved {sj.get('roi_diagnostics')}")
                if int(sj.get("n_slides", -1)) != n or list(sj.get("labels_order", [])) != classes:
                    diag_bad.append(f"{cfg}.n_slides/labels_order")
    gates.check(arm, "alignment_and_unit_weight_replicate", not gate_bad, "; ".join(gate_bad) or
                "four configurations aligned; unit-weight replicates equal as-run points to <= 1e-12")
    if limit:
        gates.skip(arm, "fidelity_saved_points_and_diagnostics",
                   f"--c1b-limit {limit}: subset cannot reproduce full-population saved points; NOT a production run")
        gates.skip(arm, "population_2778_shared_draws",
                   f"--c1b-limit {limit}: draws are over the subset's patients and differ from production")
    else:
        gates.check(arm, "fidelity_saved_points_and_diagnostics", not diag_bad, "; ".join(diag_bad) or
                    "published 0.7903/0.7549 and patient-excluded 0.7653/0.7371 macro F1, micro F1, macro P/R, "
                    "hit@5 rate and summed ROI diagnostics reproduced (points <= 1e-6, counts exact)")
        gates.check(arm, "population_2778_shared_draws", n == N_QUERIES and npat == N_PATIENTS,
                    f"{n} queries / {npat} patients")
    for k in (5, 10):
        a, b = f"patient_excluded_k{k}", f"published_k{k}"
        for met, avg in (("f1", "macro"), ("f1", "micro"), ("hit_at_5", "micro"), ("hit_at_5", "macro")):
            paired.append(paired_row(arm, pop, a, b, met, avg, pts[(a, met, avg)] - pts[(b, met, avg)],
                                     reps[(a, met, avg)] - reps[(b, met, avg)], n, npat, B,
                                     role=("primary" if k == 5 else "sensitivity") + (" (DEV subset)" if limit else "")))
    out = fr.rename(columns={"slide": "slide_returned"}).set_index(["config", "query"]).sort_index()
    return dict(summary=summ, paired=paired, frame=out, n=n, npat=npat, mult_sha=mult_sha(mult), legacy=legacy,
                timing=d["timing"], manifest=d["manifest"], limit=limit, recon=d["recon"], fallbacks=d["fallbacks"])


# ------------------------------------------------------------------ C1d
def c1d_load(ctx, gates, bindings):
    arm = "c1d"
    p, _ = bind(bindings, gates, arm, C1D_PAIRS, "C1d pair frame")
    j, _ = bind(bindings, gates, arm, C1D_JSON, "C1d saved consistency JSON (B = 2,000 patient resample)")
    df = pd.read_csv(p, dtype={"patient": str, "a": str, "b": str, "class_a": str, "class_b": str, "kind": str},
                     keep_default_na=False, na_values=[""], float_precision="round_trip")
    saved = json.load(open(j))
    same, null = df[df["kind"] == "same_patient"], df[df["kind"] == "class_matched_null"]
    ok = (len(df) == C1D_ROWS and len(same) == C1D_PAIRS_EACH and len(null) == C1D_PAIRS_EACH
          and set(df["kind"]) == {"same_patient", "class_matched_null"}
          and df["patient"].nunique() == C1D_PATIENTS and set(null["patient"]) <= set(same["patient"])
          and saved["n_same_patient_pairs"] == C1D_PAIRS_EACH and saved["n_null_pairs"] == C1D_PAIRS_EACH
          and saved["n_patients_multi_slide"] == C1D_PATIENTS)
    gates.check(arm, "frame_shape", ok, f"rows {len(df)}, same {len(same)}, null {len(null)}, "
                f"patients {df['patient'].nunique()}")
    cols = C1D_METRICS + C1D_EXTRA
    nonfinite = {c: int((~np.isfinite(df[c].to_numpy(float))).sum()) for c in cols}
    bad = []
    for c in cols:
        sp = saved["point"]
        s_, n_ = float(same[c].mean()), float(null[c].mean())
        for name, mine, theirs in (("same_patient", s_, sp["same_patient"][c]), ("class_matched_null", n_, sp["class_matched_null"][c]),
                                   ("difference", s_ - n_, sp["difference"][c])):
            if not close(mine, theirs, C1D_TOL):
                bad.append(f"{name}.{c} {mine} vs saved {theirs}")
    gates.check(arm, "fidelity_saved_points", not bad, "; ".join(bad) or
                "same-patient, null and difference means reproduce saved point.* to <= 1e-9 (10 columns)",
                nonfinite_cells=nonfinite)
    return dict(frame=df, saved=saved, nonfinite=nonfinite)


def c1d_boot(ctx, gates, d, B, seed):
    arm, pop = "c1d", "c1d_pairs_463_patients"
    df, saved = d["frame"], d["saved"]
    patients, mult, W = L.weight_matrix(df["patient"], B, seed)
    npat = len(patients)
    is_same, is_null = (df["kind"] == "same_patient").to_numpy(), (df["kind"] == "class_matched_null").to_numpy()
    summ, paired, legacy = [], [], {}
    unit_bad = []
    for c in C1D_METRICS + C1D_EXTRA:
        scope = c in C1D_METRICS
        v = df[c].to_numpy(float)
        fin = np.isfinite(v)
        vv = np.where(fin, v, 0.0)
        ms, mn = is_same & fin, is_null & fin
        rs, rn = L.boot_means(W, vv, ms), L.boot_means(W, vv, mn)
        ps, pn = float(v[ms].mean()), float(v[mn].mean())
        one = np.ones((1, len(df)))
        if not (abs(L.boot_means(one, vv, ms)[0] - ps) <= UNIT_TOL and abs(L.boot_means(one, vv, mn)[0] - pn) <= UNIT_TOL):
            unit_bad.append(c)
        note = "" if scope else "outside contract §3 metric list; reported for completeness"
        if c == "both_generous5":
            note += "; saved point difference is exactly 0.0 (same-patient and null means identical)"
        summ.append(summary_row(arm, pop, "same_patient", c, "pair_mean", ps, rs, int(ms.sum()), npat, "", B, scope, note=note))
        summ.append(summary_row(arm, pop, "class_matched_null", c, "pair_mean", pn, rn, int(mn.sum()), npat, "", B, scope, note=note))
        dd = rs - rn
        paired.append(paired_row(arm, pop, "same_patient", "class_matched_null", c, "pair_mean", ps - pn, dd,
                                 int(ms.sum() + mn.sum()), npat, B, scope,
                                 note=("null pairs share anchor slides with the same-patient pairs; the interval "
                                       "conditions on the saved pairing. " + note).strip()))
        lo, hi, _ = L.ci(dd)
        legacy[c] = {"label": "patient-resample-B2000-legacy (not pps)",
                     "saved_difference_ci_over_patients": saved["difference_ci_over_patients"].get(c),
                     "saved_point_difference": saved["point"]["difference"].get(c),
                     "new_whole_patient_difference": dict(point=ps - pn, ci=[lo, hi], flag=L.boundary_flag(lo, hi)),
                     "in_contract_scope": scope}
    gates.check(arm, "unit_weight_replicate_equals_point", not unit_bad, f"mismatch: {unit_bad}" if unit_bad else
                "weighted means at unit weights equal the point means to <= 1e-12")
    frame = df.copy()
    frame.index.name = "pair_row"
    return dict(summary=summ, paired=paired, frame=frame, npat=npat, mult_sha=mult_sha(mult), legacy=legacy,
                saved_config=saved["config"])


# ------------------------------------------------------------------ driver
def run_arm(name, fn, gates, failures):
    try:
        return fn()
    except GateFailure as e:
        failures.append(dict(arm=e.arm, gate=e.gate, detail=e.detail))
    except Exception as e:  # an unexpected error is a failure of the arm, recorded verbatim
        tb = traceback.format_exc()
        gates.records.append(dict(arm=name, gate="uncaught_exception", status="FAIL", detail=repr(e), traceback=tb))
        failures.append(dict(arm=name, gate="uncaught_exception", detail=repr(e)))
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=os.path.join(OUT, TAG, "production"))
    ap.add_argument("--arms", default="c1e,c1b,c1d")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--c1b-limit", type=int, default=None,
                    help="DEV/timing only: evaluate the first N C1b slides in sorted id order; fidelity gates skipped")
    ap.add_argument("--n-workers", type=int, default=8)
    a = ap.parse_args(argv)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if set(arms) - {"c1e", "c1b", "c1d"} or not arms:
        ap.error(f"--arms must be a subset of c1e,c1b,c1d; got {a.arms}")
    o = a.out_dir
    os.makedirs(o, exist_ok=False)
    t_start = time.time()
    gates, failures, bindings, res, timing = Gates(), [], {}, {}, {}
    ctx = {}
    if "c1e" in arms:
        t0 = time.time()
        d = run_arm("c1e", lambda: c1e_frames(ctx, gates, bindings), gates, failures)
        if d is not None:
            res["c1e"] = run_arm("c1e", lambda: c1e_boot(ctx, gates, d, a.B, a.seed), gates, failures)
        timing["c1e_wall_seconds"] = time.time() - t0
    if "c1b" in arms:
        t0 = time.time()
        d = run_arm("c1b", lambda: c1b_predict(ctx, gates, bindings, a.c1b_limit, a.n_workers), gates, failures)
        if d is not None:
            res["c1b"] = run_arm("c1b", lambda: c1b_boot(ctx, gates, d, a.B, a.seed), gates, failures)
        timing["c1b_wall_seconds"] = time.time() - t0
    if "c1d" in arms:
        t0 = time.time()
        d = run_arm("c1d", lambda: c1d_load(ctx, gates, bindings), gates, failures)
        if d is not None:
            res["c1d"] = run_arm("c1d", lambda: c1d_boot(ctx, gates, d, a.B, a.seed), gates, failures)
        timing["c1d_wall_seconds"] = time.time() - t0
    res = {k: v for k, v in res.items() if v is not None}
    failed_arms = sorted({f["arm"] for f in failures})
    if "c1e" in res and "c1b" in res and not a.c1b_limit:
        try:
            gates.check("shared", "c1e_c1b_identical_draws", res["c1e"]["mult_sha"] == res["c1b"]["mult_sha"]
                        and list(res["c1e"]["patients"]) == sorted(set(res["c1b"]["frame"].reset_index()["patient"])),
                        f"c1e {res['c1e']['mult_sha']} vs c1b {res['c1b']['mult_sha']}")
        except GateFailure as e:
            failures.append(dict(arm=e.arm, gate=e.gate, detail=e.detail))
            failed_arms.append("shared")

    summary = pd.DataFrame([r for k in ("c1e", "c1b", "c1d") if k in res for r in res[k]["summary"]])
    paired = pd.DataFrame([r for k in ("c1e", "c1b", "c1d") if k in res for r in res[k]["paired"]])
    summary.to_csv(os.path.join(o, "summary.csv"), index=False)
    paired.to_csv(os.path.join(o, "paired_differences.csv"), index=False)
    qf = [res[k]["frame"].reset_index() for k in ("c1e", "c1b") if k in res]
    if qf:
        pd.concat(qf, ignore_index=True, sort=False).to_csv(os.path.join(o, "per_query_frame.csv.gz"), index=False,
                                                            compression="gzip")
    if "c1d" in res:
        res["c1d"]["frame"].to_csv(os.path.join(o, "per_pair_frame.csv.gz"), compression="gzip")

    bindings[os.path.abspath(__file__)] = dict(role="this script", sha256=L.sha256_bytes(L.read_bytes(__file__)))
    bindings[LADDER] = dict(role="imported kernel (r2_2_tile_ladder_whole_patient.py)",
                            sha256=L.sha256_bytes(L.read_bytes(LADDER)))
    ib = dict(inputs=bindings, prosper_search=prosper_provenance(), python=sys.version, numpy=np.__version__,
              pandas=pd.__version__, host=platform.node())
    if "c1b" in res:
        man = res["c1b"]["manifest"]
        ib["c1b_roi_pickles"] = dict(directory=ROI_DIR, n=len(man),
                                     digest_of_sorted_manifest=L.sha256_bytes("\n".join(f"{k} {v}" for k, v in sorted(man.items())).encode()),
                                     per_slide_sha256=man)
    json.dump(ib, open(os.path.join(o, "input_bindings.json"), "w"), indent=2, default=str)

    generated = dt.datetime.now().isoformat(timespec="seconds")
    recon = dict(
        script=os.path.relpath(__file__, R), script_sha256=L.sha256_bytes(L.read_bytes(__file__)),
        imported_kernel=os.path.relpath(LADDER, R), imported_kernel_sha256=L.sha256_bytes(L.read_bytes(LADDER)),
        contract="docs/analysis/r4a_vii_uncertainty_inventory_contract_20261015.md",
        B=a.B, seed=a.seed, arms_requested=arms, c1b_limit=a.c1b_limit, n_workers=a.n_workers,
        production=(a.c1b_limit is None and not failures),
        bootstrap="whole-patient: patients with replacement (make_patient_draws, PCG64), every unit weighted by its "
                  "patient's multiplicity; paired differences per replicate",
        gates=gates.records, failures=failures, timing=timing, generated=generated,
        host=platform.node(),
    )
    if "c1e" in res:
        r = res["c1e"]
        recon["c1e"] = dict(n_queries=r["n"], n_patients=r["npat"], draws_multiplicity_sha256=r["mult_sha"],
                            tie_census_unmasked=r["ties"], legacy_vs_new=r["legacy"], cross_check_r2_3_ext=r["cross_check"],
                            masked_side_rules="masked k=5 predictions use generous_majority/absolute rules mirrored from "
                                              "slide_eval.predict_labels_from_matrix, gated per query on the unmasked matrix")
    if "c1b" in res:
        r = res["c1b"]
        recon["c1b"] = dict(n_queries=r["n"], n_patients=r["npat"], draws_multiplicity_sha256=r["mult_sha"],
                            dev_subset=bool(r["limit"]), legacy_vs_new=r["legacy"], timing=r["timing"],
                            masked_pair_reconciliation=r["recon"], glob_fallbacks=r["fallbacks"],
                            note=("DEV: first %d slides in sorted id order; intervals are NOT production numbers" % r["limit"])
                            if r["limit"] else "full 2,778-slide population")
    if "c1d" in res:
        r = res["c1d"]
        recon["c1d"] = dict(n_patients=r["npat"], draws_multiplicity_sha256=r["mult_sha"], legacy_vs_new=r["legacy"],
                            saved_config=r["saved_config"],
                            note="separate draws over the 463 multi-slide patients (the producer's resampling unit); "
                                 "null pairs carry their anchor's patient, so a drawn patient contributes all its "
                                 "same-patient pairs and the null pairs anchored on its slides; shared anchors mean "
                                 "the difference interval conditions on the saved pairing")
    json.dump(recon, open(os.path.join(o, "reconciliation.json"), "w"), indent=2, default=str)

    readme = [__doc__, "\n## Run\n",
              f"Generated {generated} on {platform.node()}; B = {a.B}, seed = {a.seed}; arms {arms}; "
              f"c1b_limit = {a.c1b_limit}; total wall {time.time() - t_start:.1f} s.",
              f"Status: {'FAILED (see FAILED.json)' if failures else ('DEV run (--c1b-limit): not production' if a.c1b_limit else 'COMPLETE')}",
              "\n## Gates\n"]
    for g in gates.records:
        readme.append(f"- [{g['status']}] {g['arm']}.{g['gate']}: {g.get('detail')}")
    if failures:
        readme.append("\nArms with a failed gate have no interval rows. Arms that passed were still written: "
                      f"{sorted(res)}.")
    if len(paired):
        readme.append("\n## Paired differences (counts only, nominal 95 %)\n")
        for _, row in paired[paired["in_contract_scope"]].iterrows():
            if row["arm"] == "c1e" and row["metric"] not in ("f1_generous5", "f1_absolute5", "hit5", "plurality5"):
                continue
            readme.append(f"- {row['arm']} {row['minuend']} - {row['subtrahend']} {row['metric']} {row['average']}: "
                          f"{row['point']:+.4f} [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}] {row['flag']}")
    open(os.path.join(o, "README.md"), "w").write("\n".join(readme) + "\n")
    if failures:
        json.dump(dict(first_failed_gate=failures[0], all_failures=failures, arms_written=sorted(res),
                       gates=gates.records, generated=generated), open(os.path.join(o, "FAILED.json"), "w"),
                  indent=2, default=str)
    with open(os.path.join(o, "sha256.txt"), "w") as f:
        for fn in sorted(os.listdir(o)):
            if fn in ("sha256.txt", "COMPLETE") or os.path.isdir(os.path.join(o, fn)):
                continue
            f.write(f"{L.sha256_bytes(L.read_bytes(os.path.join(o, fn)))}  {fn}\n")
    if not failures:
        open(os.path.join(o, "COMPLETE"), "w").write(
            generated + ("  DEV run (--c1b-limit %d): not production\n" % a.c1b_limit if a.c1b_limit else "\n"))
    print("\n".join(readme[readme.index("\n## Run\n"):]))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
