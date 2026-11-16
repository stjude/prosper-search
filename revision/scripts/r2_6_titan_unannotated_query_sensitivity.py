#!/usr/bin/env python3
"""R2.6 TITAN intersection bundles: query-subset sensitivity restricted to queries from
patients without tumour annotations.

The direct-1024 TITAN variants select patches with five fold-held-out CONCH tumour heads.
Reference slides from unannotated patients were filtered by the mean of all five heads, so the
reference-side filter is not independent of an annotated query patient's annotations. Queries
from patients with no annotations were never seen by any of the five CONCH heads (checked
against labels_1024.csv.gz). This does not remove the S2 arm's historical UNI2 classifier
exposure, nor re-filter the reference side. This script re-evaluates the
SAVED ranked lists of each frozen intersection bundle (no re-ranking; the 2,562-slide library is
unchanged) for (a) the full pool, reproduced against the bundle's summary to 1e-12 as a gate,
(b) queries from unannotated patients (has_annotations == 0; primary sensitivity) and (c)
queries from annotated patients (descriptive contrast).

Metrics, bootstrap and macro come from the frozen `scripts/r2_6_titan_common_pool.py`
(imported after a sha256 check). Run from the repository root with PROSPER_OUT exported.
Exit 0 = COMPLETE, 1 = a check failed (FAILED marker), 2 = refused (outdir not empty).
"""
import argparse, datetime, hashlib, importlib.util, json, os, platform, sys, time

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(HERE, "r2_6_titan_common_pool.py")
REF_SHA = "ebce93ce3a7cfd32636c4c51950b969b68dc25d16c74906a5daa9692f54547a2"
BUNDLES = ["b100_argmax_vs_arm2l1", "b100_argmax_vs_repaired", "b100_strict_vs_arm2l1",
           "b100_strict_vs_repaired", "b5_argmax_vs_arm2l1", "b5_argmax_vs_repaired", "repaired_vs_arm2l1"]
BUNDLE_FMT = "r2_6_titan_intersection_{}_20261004"
FOLDS_REL = "r2_1_cpu_20261002/filter_attempt_001/patient_folds.csv"
LABELS_REL = "r2_6_titan_direct1024_20261004/labels_1024/labels_1024.csv.gz"
CONCH_TRAIN_LABELS = (1, 2, 3, 4, 5)
OUT_NAME = "r2_6_titan_unannotated_query_sensitivity_20261004"
TOL = 1e-12
ENDPOINTS = ["hit@1", "hit@3", "hit@5", "hit@10", "precision@5", "plurality@5"]
POP_UNANN, POP_ANN = "unannotated_patient_queries", "annotated_patient_queries"
ROLE = {"full": "full_pool_reproduced", POP_UNANN: "primary_sensitivity", POP_ANN: "descriptive_contrast"}
CMP_NUM = ["estimate", "lo", "hi"]
CMP_EXACT = ["n_queries", "n_patients", "n_class_queries", "n_class_patients", "B_effective",
             "n_zero_support_draws", "ci_available", "ci_degenerate"]


class CheckError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference(path=REF_PATH, expected=REF_SHA):
    """Import the frozen common-pool module (no side effects at import) after a sha256 check."""
    got = sha256(path)
    if got != expected:
        raise CheckError(f"reference script sha256 {got} != frozen {expected}")
    spec = importlib.util.spec_from_file_location("r2_6_titan_common_pool_frozen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = load_reference()
B, SEED, KS = T.B, T.SEED, T.KS
assert (B, SEED) == (5000, 123)


def ci(point, draws):
    """[decision record] nominal 95% percentile interval (2.5/97.5, linear); delegates to the frozen module."""
    return T.ci(point, draws)


def weights(patients, draws):
    pts, mult = draws
    idx = {p: i for i, p in enumerate(pts)}
    return mult[:, [idx[p] for p in patients]].astype(float)


def evaluate(pq1, pq2, patients, classes, population, draws):
    """Micro/macro and per-group statistics for both arms and the paired delta (arm2 - arm1),
    computed within each whole-patient bootstrap replicate. `classes` is the fixed group set:
    a group with no query in this population makes the macro point estimate and interval
    unavailable (never averaged over surviving groups)."""
    q = pq1["query"].to_numpy()
    if not np.array_equal(q, pq2["query"].to_numpy()):
        raise CheckError("arms have different query order")
    patients = np.asarray(patients)
    W = weights(patients, draws)
    one = np.ones((1, len(q)))
    labels = pq1["true_label"].to_numpy()
    base = dict(population=population, n_queries=len(q), n_patients=len(set(patients)))
    rows, grows = [], []
    for col in ENDPOINTS:
        v1, v2 = pq1[col].to_numpy(dtype=float), pq2[col].to_numpy(dtype=float)
        for level, fn in (("micro", lambda v, W: T.micro(v, W)),
                          ("macro", lambda v, W: T.macro(v, W, labels, classes))):
            p1, p2, b1, b2 = fn(v1, one)[0], fn(v2, one)[0], fn(v1, W), fn(v2, W)
            for arm, p, b in (("arm1", p1, b1), ("arm2", p2, b2), ("delta", p2 - p1, b2 - b1)):
                rows.append({**base, "metric": f"{level}_{col}", "arm": arm, **ci(p, b)})
        for c in classes:
            sel = labels == c
            n = int(sel.sum())
            p1 = v1[sel].mean() if n else np.nan
            p2 = v2[sel].mean() if n else np.nan
            b1, b2 = T.class_rate(v1, W, sel), T.class_rate(v2, W, sel)
            for arm, p, b in (("arm1", p1, b1), ("arm2", p2, b2), ("delta", p2 - p1, b2 - b1)):
                grows.append({**base, "group": c, "metric": col, "arm": arm, "n_group_queries": n,
                              "n_group_patients": int(len(set(patients[sel]))), **ci(p, b)})
    return pd.DataFrame(rows), pd.DataFrame(grows)


def compare_frames(new, old, keys, label, extra_new_ok=False):
    """Row-by-row comparison on `keys`; numbers to TOL (NaN == NaN), counts/flags exactly. Every
    saved row must be matched; with `extra_new_ok`, recomputed rows absent from the saved file
    (metrics the saved file does not report) are ignored."""
    k_old = set(map(tuple, old[keys].astype(str).to_numpy()))
    k_new = set(map(tuple, new[keys].astype(str).to_numpy()))
    unmatched = len(k_old - k_new) + (0 if extra_new_ok else len(k_new - k_old))
    m = old.merge(new, on=keys, how="inner", suffixes=("_saved", "_new"))
    out = {"n_saved": len(old), "n_new": len(new), "n_compared": len(m), "unmatched_rows": unmatched}
    worst, bad = 0.0, []
    for c in CMP_NUM:
        a, b = m[f"{c}_saved"].to_numpy(float), m[f"{c}_new"].to_numpy(float)
        same_nan = np.isnan(a) == np.isnan(b)
        d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
        d = np.where(same_nan, d, np.inf)
        worst = max(worst, float(d.max()) if len(d) else 0.0)
        bad += m.loc[d > TOL, keys].assign(column=c).to_dict("records")[:5]
    for c in CMP_EXACT:
        if c not in old.columns:
            continue
        if c not in new.columns:
            bad.append({"column": c, "reason": "missing from recomputed rows"})
            continue
        a, b = m[f"{c}_saved"].astype(str).to_numpy(), m[f"{c}_new"].astype(str).to_numpy()
        bad += m.loc[a != b, keys].assign(column=c).to_dict("records")[:5]
    out.update(max_abs_diff=worst, mismatch_examples=bad[:10],
               passed=unmatched == 0 and len(m) == len(old) > 0 and not bad and worst <= TOL)
    out["label"] = label
    return out


def load_bundle(bdir):
    files = {"per_query": "per_query_common_pool.csv", "ranked_arm1": "ranked_lists_common_pool_arm1.csv.gz",
             "ranked_arm2": "ranked_lists_common_pool_arm2.csv.gz", "summary": "summary_common_pool.csv",
             "per_class": "per_class_common_pool.csv", "pool_queries": "pool_queries.txt",
             "bundle_reconciliation": "reconciliation.json", "bundle_readme": "README.md"}
    paths = {k: os.path.join(bdir, v) for k, v in files.items()}
    b = {"paths": paths,
         "per_query": pd.read_csv(paths["per_query"], dtype={"query": str, "patient": str, "true_label": str}),
         "rl1": T.load_ranked(paths["ranked_arm1"]), "rl2": T.load_ranked(paths["ranked_arm2"]),
         "summary": pd.read_csv(paths["summary"]), "per_class": pd.read_csv(paths["per_class"]),
         "pool": [l.strip() for l in open(paths["pool_queries"]) if l.strip()]}
    r = json.load(open(paths["bundle_reconciliation"]))
    am = r.get("arm_metrics", {})
    b["arm1_words"], b["arm2_words"] = am.get("arm1_words", "unrecorded"), am.get("arm2_words", "unrecorded")
    b["full_pop"] = r.get("pool", {}).get("label")
    return b


def canon(path):
    return os.path.realpath(path)


def conch_training_patients(labels_path):
    """(set of patients with CONCH training labels 1-5, number of label rows, sorted label values)."""
    lab = pd.read_csv(labels_path, usecols=["patient_id", "label_1024"], dtype={"patient_id": str})
    train = set(lab.loc[lab["label_1024"].isin(CONCH_TRAIN_LABELS), "patient_id"])
    return train, len(lab), sorted(map(int, lab["label_1024"].unique()))


def conch_premise(labels_path, folds):
    """The patients with CONCH training labels (1-5) must be exactly the has_annotations == 1 patients."""
    train, n_rows, values = conch_training_patients(labels_path)
    ann = set(folds.index[folds["has_annotations"] == "1"])
    info = dict(n_training_patients=len(train), n_has_annotations_1=len(ann),
                training_not_annotated=sorted(train - ann)[:20], annotated_not_training=sorted(ann - train)[:20],
                n_label_rows=n_rows, label_values=values)
    return train == ann and len(train) > 0, info


def load_folds(path):
    f = pd.read_csv(path, dtype=str)
    if f["patient_id"].duplicated().any():
        raise CheckError("duplicate patient_id in patient_folds.csv")
    bad = sorted(set(f["has_annotations"]) - {"0", "1"})
    if bad:
        raise CheckError(f"has_annotations values outside {{0,1}}: {bad}")
    return f.set_index("patient_id")


def per_query_checks(b):
    """Per-query metrics recomputed from the saved ranked lists must equal the saved per-query file."""
    pq1, pq2 = T.per_query(b["rl1"]), T.per_query(b["rl2"])
    sp = b["per_query"]
    ok = (np.array_equal(pq1["query"].to_numpy(), sp["query"].to_numpy())
          and np.array_equal(pq2["query"].to_numpy(), sp["query"].to_numpy())
          and np.array_equal(pq1["true_label"].to_numpy(), sp["true_label"].to_numpy()))
    diffs = {}
    if ok:
        for arm, t in (("arm1", pq1), ("arm2", pq2)):
            for c in t.columns[2:]:
                a, s = t[c].to_numpy(), sp[f"{arm}_{c}"].to_numpy()
                eq = (np.array_equal(a.astype(float), s.astype(float)) if a.dtype.kind == "f"
                      else np.array_equal(a.astype(str), s.astype(str)))
                if not eq:
                    diffs[f"{arm}_{c}"] = False
    depth = {arm: sorted(set(rl.groupby("query").size())) for arm, rl in (("arm1", b["rl1"]), ("arm2", b["rl2"]))}
    lib = set(b["pool"])
    nb_in_lib = all(set(rl["neighbour"]) <= lib for rl in (b["rl1"], b["rl2"]))
    info = dict(query_order_and_labels_match=bool(ok), mismatched_columns=sorted(diffs), depth=depth,
                n_queries=len(sp), queries_equal_pool=set(sp["query"]) == lib and len(sp) == len(lib),
                neighbours_in_pool_library=bool(nb_in_lib), library_size=len(lib))
    passed = (ok and not diffs and depth == {"arm1": [10], "arm2": [10]} and info["queries_equal_pool"]
              and nb_in_lib)
    return passed, info, pq1, pq2


def subset_counts(pqa, classes):
    rows = []
    for g in list(classes) + ["ALL"]:
        s = pqa if g == "ALL" else pqa[pqa["true_label"] == g]
        u, a = s[s["has_annotations"] == "0"], s[s["has_annotations"] == "1"]
        rows.append({"group": g, "n_queries_full": len(s), "n_patients_full": s["patient"].nunique(),
                     "n_queries_retained_unannotated": len(u), "n_patients_retained_unannotated": u["patient"].nunique(),
                     "n_queries_excluded_annotated": len(a), "n_patients_excluded_annotated": a["patient"].nunique()})
    return pd.DataFrame(rows)


def run_bundle(name, bdir, folds, outdir, log):
    """Returns (checks, summary, per_group, counts, subsets, meta); writes nothing."""
    b = load_bundle(bdir)
    checks = {}

    def check(k, ok, **info):
        checks[k] = {"passed": bool(ok), **info}
        log(f"[{'PASS' if ok else 'FAIL'}] {name} {k} {json.dumps(info, default=str)[:240]}")

    def gate(k, r):
        r = dict(r)
        check(k, r.pop("passed"), **r)

    ok, info, pq1, pq2 = per_query_checks(b)
    check("per_query_recomputed_from_ranked_lists", ok, **info)
    if not ok:
        return checks, None
    pq = b["per_query"]
    patients = pq["patient"].to_numpy()
    missing = sorted(set(patients) - set(folds.index))
    check("all_query_patients_in_patient_folds", not missing, n_missing=len(missing), missing=missing[:20])
    if missing:
        return checks, None
    classes = sorted(set(pq["true_label"]))
    full_pop = b["full_pop"]
    saved = b["summary"][b["summary"]["population"] == full_pop]

    # 1. reproduction gate: the frozen paired_summary and this script's evaluate() vs the saved summary
    draws_full = T.make_patient_draws(patients, count=B, seed=SEED)
    meta = pd.DataFrame({"patient": patients, "label": pq["true_label"].to_numpy()}, index=pq["query"].to_numpy())
    s_ref, c_ref = T.paired_summary(pq1, pq2, meta, full_pop, draws_full)
    gate("reproduce_saved_summary_frozen_function", compare_frames(
        s_ref, saved, ["population", "metric", "arm"], "frozen paired_summary vs summary_common_pool.csv"))
    s_full, g_full = evaluate(pq1, pq2, patients, classes, full_pop, draws_full)
    gate("reproduce_saved_summary_this_script", compare_frames(
        s_full, saved, ["population", "metric", "arm"], "evaluate() vs summary_common_pool.csv", extra_new_ok=True))
    pcs = b["per_class"][b["per_class"]["population"] == full_pop]
    g_cmp = g_full[g_full["metric"].isin(["hit@5", "plurality@5"])].rename(
        columns={"group": "class", "n_group_queries": "n_class_queries", "n_group_patients": "n_class_patients"}).copy()
    g_cmp["metric"] = g_cmp["metric"].replace({"plurality@5": "plurality@5_recall"})
    gate("reproduce_saved_per_class_this_script", compare_frames(
        g_cmp, pcs,
        ["population", "class", "metric", "arm"], "evaluate() per-group vs per_class_common_pool.csv"))
    if not all(c["passed"] for c in checks.values()):
        return checks, None

    # 2-5. subsets (library untouched: the saved ranked lists are only filtered by query)
    ann = folds.loc[patients, "has_annotations"].to_numpy()
    pqa = pq[["query", "true_label", "patient"]].assign(
        has_annotations=ann, fold=folds.loc[patients, "fold"].to_numpy(),
        subset=np.where(ann == "0", POP_UNANN, POP_ANN))
    counts = subset_counts(pqa, classes)
    s_full["population_role"] = ROLE["full"]
    g_full["population_role"] = ROLE["full"]
    S, G = [s_full], [g_full]
    sub_info = {}
    for pop, flag in ((POP_UNANN, "0"), (POP_ANN, "1")):
        m = ann == flag
        a1, a2 = pq1[m].reset_index(drop=True), pq2[m].reset_index(drop=True)
        d = T.make_patient_draws(patients[m], count=B, seed=SEED)
        s, g = evaluate(a1, a2, patients[m], classes, pop, d)
        s["population_role"], g["population_role"] = ROLE[pop], ROLE[pop]
        S.append(s)
        G.append(g)
        sub_info[pop] = {"n_queries": int(m.sum()), "n_patients": len(d[0]),
                         "groups_absent": sorted(set(classes) - set(pq["true_label"][m])),
                         "patient_list_sha256": hashlib.sha256("\n".join(d[0]).encode()).hexdigest()}
    nu, na = sub_info[POP_UNANN], sub_info[POP_ANN]
    check("subset_partition", nu["n_queries"] + na["n_queries"] == len(pq)
          and nu["n_patients"] + na["n_patients"] == len(set(patients))
          and set(pqa.loc[pqa.subset == POP_UNANN, "patient"]).isdisjoint(pqa.loc[pqa.subset == POP_ANN, "patient"]),
          **{k: {kk: vv for kk, vv in v.items() if kk != "patient_list_sha256"} for k, v in sub_info.items()})
    summary = pd.concat(S, ignore_index=True)
    per_group = pd.concat(G, ignore_index=True)
    meta = {"arm1_words": b["arm1_words"], "arm2_words": b["arm2_words"],
            "leak_free_scope": leak_free_scope(b["arm1_words"], b["arm2_words"]), "full_population": full_pop,
            "n_library": len(b["pool"]), "classes": classes, "subsets": sub_info,
            "n_full_queries": len(pq), "n_full_patients": len(draws_full[0]),
            "full_pool_patient_list_sha256": hashlib.sha256("\n".join(draws_full[0]).encode()).hexdigest(),
            "inputs": {k: {"path": canon(p), "sha256": sha256(p)} for k, p in b["paths"].items()}}
    return checks, (summary, per_group, counts, pqa, meta)


S2_WORDS = "S2 tumour-enriched"
CONCH_WORDS = "direct-1024 variant"
SCOPES = {
    "both": "both arms (CONCH-head arm vs unfiltered TITAN)",
    "conch_s2": "CONCH arm only; S2 arm exposure not removed",
    "none_s2": "population restriction only (neither arm uses CONCH heads); S2 arm exposure not removed",
    "none": "population restriction only (neither arm uses CONCH heads)",
}
S2_SENTENCE = ("The S2 arm used the historical tile-random UNI2 tumour classifier (trained without patient grouping "
               "on 231 annotated slides, 34 with unresolved patient identity); `has_annotations == 0` does not "
               "establish that this arm's queries were unseen by its classifier, so the subset does not remove S2's "
               "training exposure. (The 231/34 counts are quoted from the revision ledger, not recomputed by this script.)")


def arm_kind(words):
    if CONCH_WORDS in words:
        return "conch"
    if S2_WORDS in words:
        return "s2"
    if "all-tissue" in words:
        return "unfiltered"
    raise CheckError(f"cannot classify arm description {words!r}")


def leak_free_scope(arm1_words, arm2_words):
    kinds = {arm_kind(arm1_words), arm_kind(arm2_words)}
    if "conch" in kinds:
        return SCOPES["conch_s2"] if "s2" in kinds else SCOPES["both"]
    return SCOPES["none_s2"] if "s2" in kinds else SCOPES["none"]


def conch_arms(meta):
    return [a for a in ("arm1", "arm2") if arm_kind(meta[f"{a}_words"]) == "conch"]


def s2_arms(meta):
    return [a for a in ("arm1", "arm2") if arm_kind(meta[f"{a}_words"]) == "s2"]


def fmt_row(s, pop, metric, arm="delta"):
    r = s[(s.population == pop) & (s.metric == metric) & (s.arm == arm)].iloc[0]
    iv = f"[{r['lo']:+.4f}, {r['hi']:+.4f}]" if r["ci_available"] else "[interval unavailable]"
    return f"{r['estimate']:+.4f} {iv}"


def support_sentences(meta, per_group):
    """Group presence and zero-support statements derived from the outputs (not hard-coded)."""
    absent = {p: meta["subsets"][p]["groups_absent"] for p in (POP_UNANN, POP_ANN)}
    if not any(absent.values()):
        pres = "every group is present in both subsets"
    else:
        pres = "; ".join(f"`{p}` lacks {', '.join(g)} (its macro is unavailable)" if g else f"`{p}` has every group"
                         for p, g in absent.items())
    out = []
    for pop in (meta["full_population"], POP_UNANN, POP_ANN):
        g = per_group[(per_group.population == pop) & (per_group.metric == "hit@5") & (per_group.arm == "arm1")]
        z = g[g["n_zero_support_draws"] > 0].sort_values(["n_group_patients", "group"])
        if len(z):
            ex = ", ".join(f"{r.group} ({r.n_group_patients} patients)" for r in z.head(3).itertuples())
            out.append(f"`{pop}`: macro intervals unavailable; {len(z)} of {len(g)} groups have zero-support "
                       f"replicates (smallest: {ex}).")
        else:
            out.append(f"`{pop}`: no zero-support replicates; macro intervals available.")
    return pres, " ".join(out)


def training_library_sentence(meta):
    """Counts computed in main() from labels_1024.csv.gz and this bundle's query (= library) patients."""
    t = meta["conch_training_patients"]
    return (f"{t['n_in_library']} of the {t['n_training_patients']} CONCH training patients have a slide in the "
            f"{meta['n_library']:,}-slide library; {t['n_not_in_library']} do not.")


def exposure_paragraphs(meta, conch_present):
    """F2 (a)-(e) plus the fold-column note."""
    pre = "" if conch_present else ("This bundle has no CONCH-head arm, so (a)-(d) describe the CONCH pipeline "
                                    "generally and do not apply to either arm here. ")
    return [
        "**What this sensitivity does not change, and why the remainder is not query leakage.** " + pre +
        "(a) The reference-side CONCH patch selection is unchanged: reference slides keep the patches selected in "
        "the original run (five-head mean `mean5` for unannotated reference patients, their own fold-held-out head "
        "for annotated reference patients). Only the query set is restricted, so this is not a re-filtered, "
        "end-to-end patient-held-out pipeline. "
        "(b) The remaining reference-side exposure is not query leakage: for a retained query, every CONCH head "
        "that selected patches of any reference slide was trained only on annotated patients other than "
        "the query patient (the query patient has no annotation rows). " + training_library_sentence(meta) + " "
        "(c) The exposure is symmetric: each retained query's own patches were selected by `mean5`, and four of "
        "its five heads were trained on each annotated reference patient's patches. The comparison is therefore "
        "\"filter fitted on annotated patients\", not \"filter independent of the library\". "
        "(d) All retained queries were filtered by `mean5`, which has no direct held-out validation (the held-out "
        "sensitivity/specificity apply to the single fold heads, not to their mean). "
        "(e) Intervals are reseeded per population and are not paired with the full-pool intervals, so no "
        "full-minus-subset difference or \"leak effect size\" can be read from them. The annotated-subset "
        "intervals are wide; a null subset difference is not evidence of no bias.", "",
        "`fold` in `query_subsets.csv` is the patient's fold assignment in `patient_folds.csv`; unannotated "
        "patients' patches were selected by `mean5`, not by a fold head.", ""]


def bundle_readme(name, res):
    summary, per_group, counts, _, meta = res
    full, nu, na = meta["full_population"], meta["subsets"][POP_UNANN], meta["subsets"][POP_ANN]
    ca, sa = conch_arms(meta), s2_arms(meta)
    conch = (f"Arm(s) using the fold-held-out CONCH tumour heads in this bundle: {', '.join(ca)}." if ca else
             "Neither arm in this bundle uses the fold-held-out CONCH tumour heads; the subset is reported for "
             "completeness as a population restriction only and does not address the CONCH-head reference-side "
             "dependence.")
    s2 = f" S2 arm(s): {', '.join(sa)}. {S2_SENTENCE}" if sa else ""
    pres, zero = support_sentences(meta, per_group)
    lines = [f"# R2.6 TITAN unannotated-patient query-subset sensitivity: `{name}`", "",
             f"Produced by `scripts/r2_6_titan_unannotated_query_sensitivity.py` from the frozen bundle "
             f"`{BUNDLE_FMT.format(name)}`. Descriptive outputs; provenance in the top-level `record.json` and "
             "`reconciliation.json`.", "",
             f"**Arms.** Arm 1 = {meta['arm1_words']}; arm 2 = {meta['arm2_words']}. Delta = arm2 - arm1 "
             f"(arm 2 minus arm 1), computed within each bootstrap replicate. {conch}{s2}", "",
             f"**Leak-free scope (`leak_free_scope`): {meta['leak_free_scope']}.** This label applies only to the "
             f"`{POP_UNANN}` rows; in `summary_all_bundles.csv` it is set on those rows only and left empty on the "
             "full-pool and annotated-subset rows, which include annotated-patient queries and are not leak-free.", "",
             "**Why this subset.** The direct-1024 TITAN variants select tumour patches with five fold-held-out "
             "CONCH tumour heads. A query from an annotated patient was filtered by its own held-out head, but "
             "reference slides from unannotated patients were filtered by the mean of all five heads (and annotated "
             "references by their own fold head), so the reference-side filter is not independent of an annotated "
             "query patient's annotations. Queries from patients with no annotations were never seen by any of the "
             "five CONCH heads (check `conch_training_patients_equal_has_annotations`: the patients with CONCH "
             "training labels 1-5 in `labels_1024.csv.gz` are exactly the `has_annotations == 1` patients). This "
             "premise concerns the CONCH heads only.", "",
             "**Definition.** This is a **query-subset sensitivity**: the saved depth-10 ranked lists are filtered "
             "by query only. The reference library is unchanged (the full "
             f"{meta['n_library']:,}-slide library, including annotated patients' slides); nothing is re-ranked. "
             "Retained (`unannotated_patient_queries`, primary sensitivity): queries whose patient has "
             "`has_annotations == 0` in `r2_1_cpu_20261002/filter_attempt_001/patient_folds.csv`. Complement "
             "(`annotated_patient_queries`, `has_annotations == 1`): descriptive contrast only; its percentile "
             "intervals carry no inferential claim, and it is not a counterfactual for the unannotated subset "
             "(the two subsets differ in patients and group mix).", "",
             *exposure_paragraphs(meta, bool(ca)),
             f"Counts: full pool {meta['n_full_queries']} queries / {meta['n_full_patients']} patients; retained "
             f"{nu['n_queries']} queries / {nu['n_patients']} patients; excluded (annotated) {na['n_queries']} "
             f"queries / {na['n_patients']} patients. Per group: `subset_counts.csv`.", "",
             "**Statistics.** Per query: hit@k (k = 1, 3, 5, 10; a same-group slide among the top k), "
             "precision@5 (fraction of the top 5 in the query's group) and plurality@5 (the most frequent group "
             "among the top 5, ties to the tied group at the closest rank, equals the query's group). Micro = mean "
             "over queries; macro = unweighted mean over the groups present (the fixed set of "
             f"{len(meta['classes'])} groups of the full pool; {pres}; a population lacking a group would have "
             "its macro reported unavailable). Intervals: whole-patient percentile bootstrap "
             f"(B={B}, seed={SEED}, NumPy PCG64; patients resampled with replacement within each population, each "
             "query weighted by its patient's multiplicity; percentiles 2.5 and 97.5, i.e. nominal 95%). Each "
             "population has its own patient draws (sorted patient list, same B/seed); the same draws serve "
             "both arms and the delta (paired). A replicate with zero support for a required group makes that "
             f"interval unavailable (`ci_available` False); no replicate is dropped. {zero}", "",
             f"**Reproduction gate.** The full-pool rows (`population = {full}`) reproduce the bundle's "
             "`summary_common_pool.csv` (micro/macro hit@k, micro plurality@5, both arms and delta, estimates and "
             f"intervals) and `per_class_common_pool.csv` (hit@5, plurality@5 recall) to {TOL:g}, and the per-query "
             "metrics recomputed from the saved ranked lists equal `per_query_common_pool.csv`. precision@5 "
             "(micro/macro) and macro plurality@5 are not in the saved summary; they use the same verified code path.", "",
             "**Headline (micro hit@5 delta, arm2 - arm1).** "
             f"Full pool {fmt_row(summary, full, 'micro_hit@5')}; unannotated-patient queries "
             f"{fmt_row(summary, POP_UNANN, 'micro_hit@5')}; annotated-patient queries (descriptive) "
             f"{fmt_row(summary, POP_ANN, 'micro_hit@5')}.", "",
             "**Multiplicity.** Many metrics x populations x groups are reported with unadjusted 95% intervals; an "
             "interval excluding 0 is not a significance claim. Slide queries from one patient are not independent.", "",
             "Files:",
             "- `summary.csv`: population, population_role, metric (`micro_`/`macro_` + endpoint), arm "
             "(arm1/arm2/delta), estimate, lo, hi, B_effective, n_zero_support_draws, ci_available, ci_degenerate.",
             "- `per_group.csv`: the same per group (including LIVER) and endpoint, with group query/patient counts.",
             "- `subset_counts.csv`: per group, queries/patients in the full pool, retained (unannotated) and "
             "excluded (annotated).",
             "- `query_subsets.csv`: each query with patient, group, has_annotations, fold and subset.", ""]
    return "\n".join(lines)


def top_readme(results):
    tl = sorted({training_library_sentence(r[4]) for r in results.values()})
    tls = tl[0] if len(tl) == 1 else "Training-patient library counts differ by bundle; see each README."
    rows = "\n".join(f"| `{n}` | {r[4]['arm1_words']} | {r[4]['arm2_words']} | {r[4]['leak_free_scope']} |"
                     for n, r in results.items())
    return f"""# R2.6 TITAN unannotated-patient query-subset sensitivity (2026-10-04)

Produced by `scripts/r2_6_titan_unannotated_query_sensitivity.py`. One subdirectory per frozen
`r2_6_titan_intersection_*_20261004` bundle (see each README for the full statements). This is a
query-subset sensitivity: the saved ranked lists are restricted to queries whose patient has
`has_annotations == 0`; the reference library (2,562 slides) is unchanged and nothing is re-ranked.
Delta = arm2 - arm1. Whole-patient percentile bootstrap (B={B}, seed={SEED}, PCG64, percentiles
2.5/97.5). The full-pool rows of every bundle reproduce the bundle's saved summary to {TOL:g} (gate;
see `reconciliation.json`). Check `conch_training_patients_equal_has_annotations` confirms that the
patients with CONCH training labels are exactly the `has_annotations == 1` patients, so retained
queries were never seen by any of the five CONCH heads.

| bundle | arm 1 | arm 2 | `leak_free_scope` (unannotated-subset rows only) |
|---|---|---|---|
{rows}

- S2 arms: {S2_SENTENCE}
- (a) Reference-side CONCH patch selection is unchanged (`mean5` for unannotated reference patients,
  own fold head for annotated ones): this is not a re-filtered, end-to-end patient-held-out pipeline.
- (b) The remaining reference-side exposure is not query leakage: for a retained query, the heads that
  selected reference patches were trained only on annotated patients other than the query patient.
  {tls}
- (c) Symmetric exposure: retained queries' own selector (`mean5`) was trained on annotated reference
  patients' patches; read the comparison as "filter fitted on annotated patients", not
  "filter independent of the library".
- (d) All retained queries were filtered by `mean5`, which has no direct held-out validation.
- (e) Subset intervals are reseeded per population and not paired with the full-pool intervals: no
  full-minus-subset "leak effect size" can be read. Annotated-subset intervals are wide; a null subset
  difference is not evidence of no bias.

- `summary_all_bundles.csv`: all bundles' `summary.csv` rows with `bundle`, `arm1_words`, `arm2_words`,
  `leak_free_scope`. `leak_free_scope` applies only to the `unannotated_patient_queries` rows and is
  empty on the full-pool and annotated-subset rows (they include annotated-patient queries).
- `per_group_all_bundles.csv`: all bundles' `per_group.csv` rows with `bundle`, `arm1_words`, `arm2_words`.
- `record.json`: script/reference/input sha256s (including `labels_1024.csv.gz`), seed, B, patient-list
  hashes, command, versions.
- `reconciliation.json`: global and per-bundle checks. `COMPLETE` is written last and binds the sha256
  of every other output file; `FAILED` is written instead if any check or step fails.
"""


def write_failure(outdir, recon, rec, log_lines, msg):
    for fn, obj in (("reconciliation.json", recon), ("record.json", rec)):
        with open(os.path.join(outdir, fn), "w") as f:
            json.dump(obj, f, indent=1, default=str)
    with open(os.path.join(outdir, "run.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    with open(os.path.join(outdir, "FAILED"), "w") as f:
        f.write(msg + "\n")
    print(f"FAILED: {msg}", file=sys.stderr)
    return 1


def write_outputs(outdir, results, recon, rec, log_lines):
    allS, allG = [], []
    for name, (summary, per_group, counts, pqa, meta) in results.items():
        d = os.path.join(outdir, name)
        os.makedirs(d)
        summary.to_csv(os.path.join(d, "summary.csv"), index=False)
        per_group.to_csv(os.path.join(d, "per_group.csv"), index=False)
        counts.to_csv(os.path.join(d, "subset_counts.csv"), index=False)
        pqa.to_csv(os.path.join(d, "query_subsets.csv"), index=False)
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write(bundle_readme(name, (summary, per_group, counts, pqa, meta)))
        tag = dict(bundle=name, arm1_words=meta["arm1_words"], arm2_words=meta["arm2_words"])
        allS.append(summary.assign(**tag, leak_free_scope=np.where(
            summary["population"] == POP_UNANN, meta["leak_free_scope"], "")))
        allG.append(per_group.assign(**tag))
    for df, fn, first in (
            (pd.concat(allS, ignore_index=True), "summary_all_bundles.csv",
             ["bundle", "arm1_words", "arm2_words", "leak_free_scope"]),
            (pd.concat(allG, ignore_index=True), "per_group_all_bundles.csv", ["bundle", "arm1_words", "arm2_words"])):
        df[first + [c for c in df.columns if c not in first]].to_csv(os.path.join(outdir, fn), index=False)
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write(top_readme(results))
    with open(os.path.join(outdir, "run.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    with open(os.path.join(outdir, "reconciliation.json"), "w") as f:
        json.dump(recon, f, indent=1, default=str)
    with open(os.path.join(outdir, "record.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    outputs = {}
    for root, _, files in os.walk(outdir):
        for fn in files:
            p = os.path.join(root, fn)
            outputs[os.path.relpath(p, outdir)] = sha256(p)
    with open(os.path.join(outdir, "COMPLETE"), "w") as f:
        json.dump({"record_sha256": outputs["record.json"], "reconciliation_sha256": outputs["reconciliation.json"],
                   "bundles": list(results), "outputs_sha256": dict(sorted(outputs.items()))}, f, indent=1)


def main(argv=None):
    t0 = time.time()
    out_root = os.environ.get("PROSPER_OUT")
    if not out_root:
        print("export PROSPER_OUT first", file=sys.stderr)
        return 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(out_root, OUT_NAME))
    ap.add_argument("--bundle-root", default=out_root)
    ap.add_argument("--folds", default=os.path.join(out_root, FOLDS_REL))
    ap.add_argument("--labels", default=os.path.join(out_root, LABELS_REL))
    ap.add_argument("--bundles", nargs="+", default=BUNDLES)
    args = ap.parse_args(argv)
    outdir = args.outdir
    if os.path.exists(outdir) and (not os.path.isdir(outdir) or os.listdir(outdir)):
        print(f"REFUSING: {outdir} exists and is not an empty directory", file=sys.stderr)
        return 2
    os.makedirs(outdir, exist_ok=True)
    log_lines = []

    def log(s):
        print(s)
        log_lines.append(s)

    rec = {"timestamp": datetime.datetime.now().astimezone().isoformat(),
           "command": " ".join([sys.executable] + (sys.argv if argv is None else ["<main>"] + list(argv))),
           "script": {"path": canon(__file__), "sha256": sha256(os.path.abspath(__file__))},
           "reference_script": {"path": canon(REF_PATH), "sha256": sha256(REF_PATH), "expected_sha256": REF_SHA},
           "B": B, "seed": SEED, "rng": "numpy.random.Generator(PCG64(seed))", "percentiles": [2.5, 97.5],
           "tolerance": TOL, "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                          "pandas": pd.__version__}, "inputs": {}, "bundles": {}}
    recon = {"checks": {}}
    try:
        rec["inputs"]["patient_folds"] = {"path": canon(args.folds), "sha256": sha256(args.folds)}
        rec["inputs"]["conch_labels_1024"] = {"path": canon(args.labels), "sha256": sha256(args.labels)}
        folds = load_folds(args.folds)
        ok, info = conch_premise(args.labels, folds)
        train = conch_training_patients(args.labels)[0]
        rec["conch_training_patients"] = {"n": len(train), "patient_list_sha256": hashlib.sha256(
            "\n".join(sorted(train)).encode()).hexdigest()}
        recon["checks"]["global"] = {"conch_training_patients_equal_has_annotations": {"passed": ok, **info}}
        log(f"[{'PASS' if ok else 'FAIL'}] global conch_training_patients_equal_has_annotations "
            f"{json.dumps(info, default=str)[:240]}")
        if not ok:
            rec["runtime_s"] = round(time.time() - t0, 1)
            return write_failure(outdir, recon, rec, log_lines, "check failed: conch_training_patients_equal_has_annotations")
        results = {}
        for name in args.bundles:
            bdir = os.path.join(args.bundle_root, BUNDLE_FMT.format(name))
            checks, res = run_bundle(name, bdir, folds, outdir, log)
            recon["checks"][name] = checks
            if res is None or not all(c["passed"] for c in checks.values()):
                rec["runtime_s"] = round(time.time() - t0, 1)
                bad = [k for k, c in checks.items() if not c["passed"]]
                return write_failure(outdir, recon, rec, log_lines, f"check failed in {name}: {bad}")
            results[name] = res
            meta = res[4]
            lib_pat = set(res[3]["patient"])
            meta["conch_training_patients"] = {"n_training_patients": len(train),
                                               "n_in_library": len(train & lib_pat),
                                               "n_not_in_library": len(train - lib_pat)}
            rec["inputs"][name] = meta.pop("inputs")
            rec["bundles"][name] = meta
        rec["runtime_s"] = round(time.time() - t0, 1)
        write_outputs(outdir, results, recon, rec, log_lines)
    except Exception as e:  # any failure after outdir creation leaves an explicit FAILED marker
        recon["error"] = f"{type(e).__name__}: {e}"
        log(f"ERROR {recon['error']}")
        rec["runtime_s"] = round(time.time() - t0, 1)
        if os.path.exists(os.path.join(outdir, "COMPLETE")):
            os.remove(os.path.join(outdir, "COMPLETE"))
        return write_failure(outdir, recon, rec, log_lines, recon["error"])
    print(f"COMPLETE {outdir} ({rec['runtime_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
