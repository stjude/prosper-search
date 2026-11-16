#!/usr/bin/env python3
"""R3.7 supplementary inventory of ALL evaluated fine-label retrieval arms.

Read-only consumer of the frozen Stream D outputs
($PROSPER_OUT/r3_7_fine_label_20261003, produced by
scripts/r3_7_fine_label_reevaluation.py sha256 cc08ac94...). Every count and rate is
recomputed from per_query_<arm>.csv, required to equal confusion_k5_*_<arm>.csv
exactly, and cross-checked against summary_all_arms.csv, coverage.csv and the frozen
reconciliation.json. Every frozen input file must match its pinned sha256 (PINNED_SHA256);
copied intervals must satisfy lo <= estimate <= hi; coverage identities and the expected
broad-group name/count per arm are enforced. Any mismatch -> exit 1 with nothing written.

No new bootstrap is computed. A whole-patient interval is reported only when the
frozen summary row carries one (unit patient, ci_patient_level == "True",
ci_available == "True", finite bounds); slide-resampling intervals (KidRare) are never
reported as patient-level.

Label-source fields are read from the label manifests bound by sha256 in the frozen
reconciliation.json inputs (the CCDI manifest, not listed there, is pinned here).

Usage (repository root):
  export PROSPER_OUT="$PWD/prosper_revision_output"
  /HOME/miniconda3/bin/python3 scripts/r3_7_task_inventory.py \
      --out-dir "$PROSPER_OUT/r3_7_task_inventory_20261004b"
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
import sys

import pandas as pd

FROZEN_D_SCRIPT_SHA256 = "bb7436efa7b18c617b970b0796ddf867be053b5f768b820d37cbe4902cc5e5f2"
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
D_SCRIPT = os.path.join(REPO, "scripts", "r3_7_fine_label_reevaluation.py")
DEFAULT_IN = os.path.join(os.environ.get("PROSPER_OUT", os.path.join(REPO, "prosper_revision_output")),
                          "r3_7_fine_label_20261003")
TOL = 1e-12
ABSTAIN = "ABSTAIN"
NO_CI = "point only — no CI in frozen output"
NOT_PATIENT_CI = ("no whole-patient CI in frozen output (only a slide-resampling interval, "
                  "patient linkage unknown; not reported)")
BINARY_STRICT = "= plurality (binary)"

# Frozen D bundle (r3_7_fine_label_20261003) sha256 values: summary/coverage/reconciliation as
# recorded in [internal record], README and per-arm files as observed
# and independently re-hashed by the inventory review. Any difference -> refusal.
PINNED_SHA256 = {
    "README.md": "40691697f6acd767303bafd4b36dc3fdf014de8051bf1fceaf3e3899bf38ddc5",
    "summary_all_arms.csv": "cdc5f455c7217847a139b715d68c6f703fa8cbd537f580a296eae23520e4ed1e",
    "coverage.csv": "679b9fa96c1b5c6c115fa7c6c941fb81af6a4e25f83fb84a5c88bc95c284a694",
    "reconciliation.json": "03ab56e1789184ae9ef3b8f26518f3abac92a79d20cbfa074564f38444db99b9",
    "per_query_s61_int_2c.csv": "7efbc27b5828c9c3956c1ee2616fa1ca8233d2a5def5f0e4c06efd862f628715",
    "confusion_k5_plurality_s61_int_2c.csv": "f5b62c1ffb235bf255920b8145b7360ebce551955d439a805992e1be8da9c782",
    "confusion_k5_strict_s61_int_2c.csv": "95b8f917f3fc83e8e0268e36c954e50affe9206db93c064cf239636549094d8b",
    "per_query_s61_int_3c.csv": "5d9ab84af776e4c5407b5580d283bdc3fcdf86447d9630c56f72439b5ecc198c",
    "confusion_k5_plurality_s61_int_3c.csv": "72cb2fc9a7a1a767b3d5c42a17c79e677b519aca74516ad492c70bd701a1cb92",
    "confusion_k5_strict_s61_int_3c.csv": "7898dfdfa9238355cb413db849425b341cade43b1c4df1b006a93ba5c73485ba",
    "per_query_s61_int_2c_xeno.csv": "afb6f2cab9876e81212881e8f31886c4e75a1901c6699fc070a10c43847dd2c6",
    "confusion_k5_plurality_s61_int_2c_xeno.csv": "a8f95b13a37103d32c691e3e453ba64e854855d28091c903c6697b047ca81eee",
    "confusion_k5_strict_s61_int_2c_xeno.csv": "80b6270fb3d27416a3210373aa5e1209a1c1ce66714a7428287df4f20a38aa9d",
    "per_query_s61_fusion.csv": "2f5e079bf6bed17fde1d4d458ff2524b28ee72c90c6e681c050a58a72f0a054d",
    "confusion_k5_plurality_s61_fusion.csv": "c01588302de74ee07220ac23aca32bbe9061f416a6d79644c57b7321a2022f8d",
    "confusion_k5_strict_s61_fusion.csv": "88a14abfc54bff9018c2512b724fc364c7566b585250d810a100fa1006b9aaa9",
    "per_query_s61_ext_2c.csv": "d0fa1a215411f3cffb17767c01aaddacc50ff8e34595dda54cffc719b4466ff7",
    "confusion_k5_plurality_s61_ext_2c.csv": "c7e9bec39c3876cbbfdd76d60e94ee65003dc6c0a479fc38e274978d854147b5",
    "confusion_k5_strict_s61_ext_2c.csv": "ba4c322f92de5a14a3f47b825bdc8538f754b408c858716cb2f57f13f1e7113e",
    "per_query_s61_ext_3c.csv": "e95a28f1a0b1a64fc7ce21bd752ea8dd63405ab48eca9c5b6b16c1ee259b6cf3",
    "confusion_k5_plurality_s61_ext_3c.csv": "14f88ff3d52f043528e580973eb2d9c4494ac7358dc797ee69dd5894a2b364dc",
    "confusion_k5_strict_s61_ext_3c.csv": "a5fd9295ec33350883270eefaadbb837de972b87bd335caae4db2a5c38e9e372",
    "per_query_s62.csv": "1c5c3618634a0e01c2d82a7a09ea7698ee500604c87d5a7b60891f908ebaed23",
    "confusion_k5_plurality_s62.csv": "b32d520948ec7bf234b3cfcd6817f508b4b4119736a0c711bf4aca4e6a522ebf",
    "confusion_k5_strict_s62.csv": "3b4d2acf7b82e256fb6a653e2e28ce3c4e70f340fc30793589986b8cff0ba08c",
    "per_query_s62_xeno.csv": "b254e6048ce7544be74f117ca8c4962629d167ea5b1b1cec11219a31943c4e2e",
    "confusion_k5_plurality_s62_xeno.csv": "49b4dc9e722de9ae09312a2a57fb38251e89fca6aca62a89118f78a993e8c96a",
    "confusion_k5_strict_s62_xeno.csv": "1d6d773ae033dc82a9e307392dff3301f75435f95a9c718f4c44741ca9a0bc5d",
    "per_query_s64.csv": "ed4dc8f741ce6e92f9af3504fbef5a74e118fe7f6026cea84df159c0940d6a10",
    "confusion_k5_plurality_s64.csv": "e712f132184d472e1e6c02aa83e941b1a500b91157bc4af545173f2a70e1b7f0",
    "confusion_k5_strict_s64.csv": "c19eb3aaa407dea037d7d0aeaf9b1cf4660acbc4a7c9098b6038df4855cdb856",
    "per_query_s65_inpc.csv": "79f1df92b32b1f4c045349c10484d2f6495c42137887625b4fb5e57ab78a527d",
    "confusion_k5_plurality_s65_inpc.csv": "b7569b2de9d382fd93fd98f88b6dcc4f7389fdbbcfe38858af15eda4974a6406",
    "confusion_k5_strict_s65_inpc.csv": "02a48fcebab9a06354a90d96748f1719a577b1ee6b0f637372ce048e7174fa0c",
    "per_query_s65b_gnb.csv": "84fc4dc52a9c4935c1e981053a995858894fcb7f75cb2eb1eaa57c94e522d88e",
    "confusion_k5_plurality_s65b_gnb.csv": "d1031d8a98d6336800509545cf24a9107c2db5f3fe3db0fe2612af8220cb7a31",
    "confusion_k5_strict_s65b_gnb.csv": "8c3af640d19263344a7ff03dd3e2648d821290a01b294995ebd5097f2685c03f",
    "per_query_s65_ext_3c.csv": "fa2e9950b21b80a0e814af9142fb42ace6b5c1c122e3703c81ae04c7058da46c",
    "confusion_k5_plurality_s65_ext_3c.csv": "d3936645af115ddfa8850ff8abca2f09731cefe6e90f692d6379a24104546d33",
    "confusion_k5_strict_s65_ext_3c.csv": "25523e79e48f3253a4f35934cf5347cb43e86edc46b95f462ddb15949d1c9909",
    "per_query_s65_ext_binary.csv": "05cc3a05f088c006befd99dffa677a4905c2ca43387c16094b97bebaf2630208",
    "confusion_k5_plurality_s65_ext_binary.csv": "945b597ff047a5dc91c013427c8d0137c1ca5e9e2e78d90216fadae126962004",
    "confusion_k5_strict_s65_ext_binary.csv": "56e949df66f19a1799873ba43a52fe678690af2a9fd2dd642af63ca53b58e271",
}

# CCDI RMS label manifest: read by the frozen D script but not listed in its reconciliation
# inputs, so its observed sha256 is pinned here.
CCDI_MANIFEST_SHA256 = "68845b8b0e623015b50278804084a63bed115063d213975548584a7eb0c3340a"

# Expected broad-group (coverage denominator) name and slide count per arm.
EXPECTED_BROAD = {
    "s61_int_2c": ("COMET RMS (Tumor.group)", 471),
    "s61_int_3c": ("COMET RMS (Tumor.group)", 471),
    "s61_int_2c_xeno": ("COMET RMS (Tumor.group)", 471),
    "s61_fusion": ("COMET RMS (Tumor.group)", 471),
    "s61_ext_2c": ("CCDI RMS after SW1 exclusion", 439),
    "s61_ext_3c": ("CCDI RMS after SW1 exclusion", 439),
    "s62": ("COMET NB (Tumor.group)", 385),
    "s62_xeno": ("COMET NB (Tumor.group)", 385),
    "s64": ("COMET LIVER (Tumor.group)", 143),
    "s65_inpc": ("COMET NB (Tumor.group)", 385),
    "s65b_gnb": ("COMET NB (Tumor.group)", 385),
    "s65_ext_3c": ("KidRare NB", 304),
    "s65_ext_binary": ("KidRare NB", 304),
}

# Originating label field per arm, verified against the bound source file.
#   manifest: unique value of the manifest's label_source column
#   fusion:   column true_label of the saved S6 per_query file (assay-derived status)
#   kidrare:  column keep_subtype of the KidRare manifest (tumor_group NB)
LABEL_FIELD = {
    "s61_int_2c": ("manifest", "manifest_s61_rms_comet.csv", "ALIdf3Ali.Subtype"),
    "s61_int_3c": ("manifest", "manifest_s61_rms_comet.csv", "ALIdf3Ali.Subtype"),
    "s61_int_2c_xeno": ("manifest", "manifest_s61_rms_comet.csv", "ALIdf3Ali.Subtype"),
    "s61_fusion": ("fusion", "per_query_s61_fusion.csv", "true_label"),
    "s61_ext_2c": ("ccdi", "manifest_s61_rms_ccdi.csv", "CCDI Diagnosis (histologic)"),
    "s61_ext_3c": ("ccdi", "manifest_s61_rms_ccdi.csv", "CCDI Diagnosis (histologic)"),
    "s62": ("manifest", "manifest_s62_mycn_comet.csv", "ALIdf3Ali.NMYC.amplified..Y.N."),
    "s62_xeno": ("manifest", "manifest_s62_mycn_comet.csv", "ALIdf3Ali.NMYC.amplified..Y.N."),
    "s64": ("manifest", "manifest_s64_hbhcc_comet.csv", "rich.Diagnosis"),
    "s65_inpc": ("manifest", "manifest_s65_inpc_comet.csv", "ALIdf3Ali.INPC"),
    "s65b_gnb": ("manifest", "manifest_s65b_gnb_comet.csv", "rich.Diagnosis"),
    "s65_ext_3c": ("kidrare", "manifest_inscope.csv", "keep_subtype"),
    "s65_ext_binary": ("kidrare", "manifest_inscope.csv", "keep_subtype"),
}

# Author selection (docs/manuscript_supplement_edits.md R3.7-T4): main-text rows.
MAIN_TEXT = {
    "s61_int_2c": "main text",
    "s61_ext_2c": "main text",
    "s64": "main text",
    "s61_int_3c": "supplement (Spindle row: main-text footnote)",
    "s61_ext_3c": "supplement (Spindle row: main-text footnote)",
}
SUPPLEMENT_ONLY = "supplement only"

# Static descriptors; every arm present in the frozen outputs must appear here and vice versa.
ARM_META = {
    "s61_int_2c": ("RMS subtype (ERMS vs ARMS)", "histologic"),
    "s61_int_3c": ("RMS subtype (ERMS/ARMS/Spindle)", "histologic"),
    "s61_int_2c_xeno": ("RMS subtype (ERMS vs ARMS), xenografts retained", "histologic"),
    "s61_fusion": ("RMS fusion status", "molecular (assay-derived)"),
    "s61_ext_2c": ("RMS subtype (ERMS vs ARMS)", "histologic"),
    "s61_ext_3c": ("RMS subtype (ERMS/ARMS/Spindle)", "histologic"),
    "s62": ("NB MYCN amplification", "molecular (assay-derived)"),
    "s62_xeno": ("NB MYCN amplification, xenografts retained", "molecular (assay-derived)"),
    "s64": ("Liver tumour (HB vs HCC)", "histologic"),
    "s65_inpc": ("NB INPC favorable vs unfavorable", "histologic-prognostic (INPC)"),
    "s65b_gnb": ("NB vs GNB", "histologic"),
    "s65_ext_3c": ("NB differentiation (3 class)", "histologic"),
    "s65_ext_binary": ("NB differentiation (poorly differentiated vs differentiating)", "histologic"),
}
COHORT = {
    "internal": "internal COMET (queries and reference library)",
    "fusion": "internal COMET (queries and reference library)",
    "ccdi": "external CCDI queries vs internal COMET reference library",
    "kidrare": "external KidRare (queries and reference library within KidRare)",
}
# Independently stated reproduction targets ([assistant] spot values): (arm, kind, class, num, den)
SPOT_CHECKS = [
    ("s61_int_2c", "plurality_correct", None, 331, 373),
    ("s61_ext_2c", "plurality_correct", None, 221, 251),
    ("s64", "plurality_correct", None, 130, 139),
    ("s62", "class_recall", "amplified", 23, 66),
    ("s65b_gnb", "class_recall", "GNB", 14, 38),
]
DISPLAY_COLUMNS = [
    "arm", "main_text", "label_family", "label_type", "cohort", "label_source", "classes_n_slides", "unit",
    "n_queries", "n_patients", "n_reference_slides", "label_coverage", "exclusions",
    "plurality5_accuracy", "plurality5_accuracy_ci", "majority_class_rate",
    "mean_class_recall", "mean_class_recall_ci", "strict5_accuracy", "strict5_accuracy_ci",
    "per_class_plurality5_recall", "minority_class_recall", "minority_class_recall_ci",
    "uncertainty", "caveats",
]
NUMERIC_COLUMNS = [
    "num_plurality5_correct", "num_strict5_correct", "num_strict5_abstain",
    "num_majority_n", "num_mean_class_recall",
    "num_coverage_evaluated", "num_coverage_denominator", "num_fine_labelled_slides",
    "num_minority_correct", "num_minority_n",
]


class Mismatch(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def frac(n, d):
    return f"{n}/{d} ({n / d:.3f})"


def ci_text(lo, hi):
    return f"{lo:.3f}–{hi:.3f} (whole-patient 95% CI)"


def is_num(x):
    try:
        return x is not None and not (isinstance(x, float) and math.isnan(x)) and str(x) != ""
    except TypeError:
        return False


def as_int(x, what):
    if not is_num(x):
        return 0
    v = float(x)
    if v != int(v):
        raise Mismatch(f"{what}: non-integer count {x!r}")
    return int(v)


def close(a, b, what, ledger):
    diff = abs(float(a) - float(b))
    ledger.append({"check": what, "recomputed": a, "frozen": b, "abs_diff": diff})
    if not diff <= TOL:
        raise Mismatch(f"{what}: recomputed {float(a)!r} != frozen {float(b)!r}")


def equal(a, b, what, ledger):
    ledger.append({"check": what, "recomputed": a, "frozen": b})
    if a != b:
        raise Mismatch(f"{what}: recomputed {a!r} != frozen {b!r}")


def read_confusion(path, labels, strict):
    df = pd.read_csv(path, dtype=str)
    cols = ["truth"] + labels + ([ABSTAIN] if strict else [])
    if list(df.columns) != cols:
        raise Mismatch(f"{os.path.basename(path)}: columns {list(df.columns)} != {cols}")
    if list(df["truth"]) != labels:
        raise Mismatch(f"{os.path.basename(path)}: truth rows {list(df['truth'])} != {labels}")
    return {(t, c): int(df.loc[i, c]) for i, t in enumerate(labels) for c in cols[1:]}


def summary_row(summ, arm, cls):
    m = summ[(summ["arm"] == arm) & (summ["class"] == cls)]
    if len(m) != 1:
        raise Mismatch(f"summary_all_arms.csv: {len(m)} rows for arm={arm} class={cls}")
    idx = m.index[0]
    return m.iloc[0], f"summary_all_arms.csv line {idx + 2} (arm={arm}, class={cls})"


def flag(row, col, loc):
    """Frozen boolean flag read as text; only the exact strings True/False are accepted."""
    v = row[col]
    if v == "True":
        return True
    if v == "False":
        return False
    raise Mismatch(f"{loc}: {col} is {v!r}, expected exactly 'True' or 'False'")


def check_row_flags(row, unit, loc, led):
    """ci_patient_level must agree with the unit; both flags must be explicit booleans."""
    pl = flag(row, "ci_patient_level", loc)
    av = flag(row, "ci_available", loc)
    equal(pl, unit == "patient", f"{loc}: ci_patient_level consistent with unit {unit}", led)
    return pl, av


def ci_for(row, metric, unit, loc, led):
    """Whole-patient interval only if the frozen row carries a patient-level finite interval.
    Any finite interval read here must satisfy lo <= estimate <= hi, and lo == hi exactly when
    the metric is listed in ci_degenerate_metrics."""
    pl, av = check_row_flags(row, unit, loc, led)
    lo, hi = row.get(f"{metric}_lo"), row.get(f"{metric}_hi")
    finite = is_num(lo) and is_num(hi) and math.isfinite(float(lo)) and math.isfinite(float(hi))
    degenerate = metric in str(row.get("ci_degenerate_metrics") or "").split("|")
    if finite:
        lo, hi, est = float(lo), float(hi), float(row[metric])
        led.append({"check": f"{loc}: {metric} interval sanity lo <= estimate <= hi", "lo": lo,
                    "estimate": est, "hi": hi})
        if not (lo <= hi and lo - TOL <= est <= hi + TOL):
            raise Mismatch(f"{loc}: {metric} interval [{lo!r}, {hi!r}] impossible for estimate {est!r}")
        equal(lo == hi, degenerate, f"{loc}: {metric} lo == hi iff listed in ci_degenerate_metrics", led)
        if pl and av and unit == "patient":
            return ci_text(lo, hi), (lo, hi), degenerate
        if unit != "patient":
            return NOT_PATIENT_CI, None, False
        return NO_CI, None, False
    if degenerate:
        raise Mismatch(f"{loc}: {metric} listed as degenerate but bounds are not finite")
    if av:
        raise Mismatch(f"{loc}: {metric} bounds not finite but ci_available is True")
    return NO_CI, None, False


def bound_input(arm, recon, basename, led):
    """Path and sha256 of a file listed in the frozen reconciliation.json inputs of this arm;
    the file must still hash to the bound value."""
    inputs = recon["reconciliation"][arm]["inputs"]
    hits = [(p, v["sha256"]) for p, v in inputs.items() if os.path.basename(p) == basename]
    if len(hits) != 1:
        raise Mismatch(f"frozen reconciliation.json[{arm}].inputs: {len(hits)} entries named {basename}")
    p, want = hits[0]
    equal(sha256(p), want, f"{p} sha256 vs frozen reconciliation.json[{arm}].inputs", led)
    return p, want


def verify_label_source(arm, recon, truth, labels, n, broad, fine, manifest_rows, led):
    """Originating label field, verified in the bound source file; returns (display, binding)."""
    mode, base, field = LABEL_FIELD[arm]
    if mode in ("manifest", "ccdi"):
        if mode == "manifest":
            p, s = bound_input(arm, recon, base, led)
            how = "sha256 bound in frozen reconciliation.json inputs"
        else:
            comet, _ = bound_input(arm, recon, "manifest_s61_rms_comet.csv", led)
            p = os.path.join(os.path.dirname(comet), base)
            s = sha256(p)
            equal(s, CCDI_MANIFEST_SHA256, f"{p} sha256 vs pinned CCDI manifest hash", led)
            how = "sha256 pinned in this script; not listed in frozen reconciliation.json inputs"
        m = pd.read_csv(p, dtype=str, keep_default_na=False)
        equal(sorted(set(m["label_source"])), [field], f"{base} label_source column (unique value)", led)
        equal(len(m), manifest_rows, f"{base} rows vs coverage manifest_rows", led)
        text = f"{field} (label_source of {base})"
    elif mode == "fusion":
        p, s = bound_input(arm, recon, base, led)
        f = pd.read_csv(p, dtype=str)
        got = {t: int((f[field] == t).sum()) for t in labels}
        equal(len(f), n, f"saved S6 {base} rows vs evaluated queries", led)
        equal(got, {t: int((truth == t).sum()) for t in labels},
              f"saved S6 {base} {field} class counts vs evaluated truth", led)
        how = "sha256 bound in frozen reconciliation.json inputs"
        text = f"saved S6 s6_1_rms/{base} {field} (assay-derived fusion status; not a manifest field)"
    else:
        p, s = bound_input(arm, recon, base, led)
        k = pd.read_csv(p, dtype=str)
        nb = k[k["tumor_group"] == "NB"]
        equal(len(nb), broad, f"{base} tumor_group NB rows vs coverage broad_group_slides", led)
        equal(int(nb[field].notna().sum()), fine, f"{base} NB rows with {field} vs coverage fine_labelled_slides", led)
        how = "sha256 bound in frozen reconciliation.json inputs"
        text = f"KidRare {field} (external_kidrare/{base}, tumor_group NB)"
    return text, {"value": text, "field": field, "file": p, "sha256": s, "binding": how}


def build_arm(arm, in_dir, summ, cov, recon, ledger_root):
    meta = recon["arms"].get(arm)
    if meta is None:
        raise Mismatch(f"{arm}: absent from frozen reconciliation.json arms")
    labels = list(meta["labels"])
    kind = meta["kind"]
    binary = len(labels) == 2
    led = []
    bind = {}

    pq_name = f"per_query_{arm}.csv"
    pq = pd.read_csv(os.path.join(in_dir, pq_name), dtype=str)
    n = len(pq)
    truth, plur, strict = pq["truth"], pq["plurality5"], pq["strict5"]
    if not set(truth) <= set(labels):
        raise Mismatch(f"{pq_name}: truth labels {sorted(set(truth) - set(labels))} outside {labels}")
    if not set(plur) <= set(labels):
        raise Mismatch(f"{pq_name}: plurality5 predictions outside {labels}")
    if not set(strict) <= set(labels) | {ABSTAIN}:
        raise Mismatch(f"{pq_name}: strict5 predictions outside {labels}+ABSTAIN")
    if pq["query"].duplicated().any():
        raise Mismatch(f"{pq_name}: duplicate query rows")
    pcorr = (plur == truth)
    scorr = (strict == truth)
    sabst = (strict == ABSTAIN)
    for col, ind in (("plurality5_recall", pcorr), ("strict5_recall", scorr), ("strict5_abstain_rate", sabst)):
        if not (pq[col].astype(float) == ind.astype(float)).all():
            raise Mismatch(f"{pq_name}: per-row {col} disagrees with truth/prediction columns")
    if binary:
        equal(int((strict != plur).sum()), 0,
              f"{pq_name}: strict5 identical to plurality5 in a two-class arm (5 votes, 2 labels)", led)
    units = set(pq["unit"])
    if len(units) != 1:
        raise Mismatch(f"{pq_name}: mixed unit values {units}")
    pq_unit = units.pop()
    unit = unit_kind(pq_unit)

    # Confusions must equal the per-query crosstab exactly.
    conf = {}
    for rule, pred, is_strict in (("plurality", plur, False), ("strict", strict, True)):
        cname = f"confusion_k5_{rule}_{arm}.csv"
        c = read_confusion(os.path.join(in_dir, cname), labels, is_strict)
        cols = labels + ([ABSTAIN] if is_strict else [])
        for t in labels:
            for p in cols:
                equal(int(((truth == t) & (pred == p)).sum()), c[(t, p)],
                      f"{cname} cell truth={t} pred={p} vs {pq_name}", led)
        conf[rule] = (cname, c)

    p_correct = int(pcorr.sum())
    s_correct = int(scorr.sum())
    s_abst = int(sabst.sum())
    equal(p_correct, sum(conf["plurality"][1][(t, t)] for t in labels),
          f"plurality correct: {pq_name} vs {conf['plurality'][0]} diagonal", led)
    equal(s_correct, sum(conf["strict"][1][(t, t)] for t in labels),
          f"strict correct: {pq_name} vs {conf['strict'][0]} diagonal", led)
    equal(s_abst, sum(conf["strict"][1][(t, ABSTAIN)] for t in labels),
          f"strict abstain: {pq_name} vs {conf['strict'][0]} ABSTAIN column", led)

    all_row, all_loc = summary_row(summ, arm, "ALL")
    equal(unit_kind(all_row["unit"]), unit, f"unit kind: {pq_name} vs {all_loc}", led)
    equal(n, int(all_row["n_queries"]), f"n_queries: {pq_name} rows vs {all_loc}", led)
    close(p_correct / n, all_row["plurality5_recall"], f"plurality5 accuracy vs {all_loc}", led)
    close(s_correct / n, all_row["strict5_recall"], f"strict5 accuracy vs {all_loc}", led)
    close(s_abst / n, all_row["strict5_abstain_rate"], f"strict5 abstain rate vs {all_loc}", led)
    rec = recon["reconciliation"][arm]
    equal(list(rec["n_queries"]), [n, n], f"n_queries vs frozen reconciliation.json[{arm}]", led)
    if rec["reconciles"] is not True:
        raise Mismatch(f"frozen reconciliation.json[{arm}].reconciles is {rec['reconciles']!r}, not true")
    if not isinstance(rec["ci_patient_level"], bool):
        raise Mismatch(f"frozen reconciliation.json[{arm}].ci_patient_level is not a JSON boolean")
    equal(rec["ci_patient_level"], flag(all_row, "ci_patient_level", all_loc),
          f"ci_patient_level: frozen reconciliation.json[{arm}] vs {all_loc}", led)

    if unit == "patient":
        n_pat = int(pq["patient"].nunique())
        equal(n_pat, int(all_row["n_patients"]), f"n_patients: unique {pq_name} patient vs {all_loc}", led)
        n_pat_disp = str(n_pat)
    else:
        equal(str(all_row["n_patients"]), "unknown", f"n_patients unknown (slide unit) in {all_loc}", led)
        n_pat = None
        n_pat_disp = "unknown (no patient linkage)"

    cov_m = cov[cov["arm"] == arm]
    if len(cov_m) != 1:
        raise Mismatch(f"coverage.csv: {len(cov_m)} rows for {arm}")
    cv = cov_m.iloc[0]
    cov_loc = f"coverage.csv line {cov_m.index[0] + 2} (arm={arm})"
    n_ref = as_int(all_row["n_reference_slides"], "n_reference_slides")
    equal(as_int(cv["n_reference_slides"], "coverage n_reference_slides"), n_ref,
          f"n_reference_slides: {cov_loc} vs {all_loc}", led)
    equal(list(rec["n_reference"]), [n_ref, n_ref], f"n_reference vs frozen reconciliation.json[{arm}]", led)
    equal(n, as_int(cv["n_evaluated_queries"], "n_evaluated_queries"), f"n_evaluated_queries: {pq_name} vs {cov_loc}", led)
    drops = {k: as_int(cv[k], k) for k in ("dropped_xenograft", "dropped_discordant_patient",
                                            "dropped_no_fusion_status", "dropped_label_filter",
                                            "dropped_not_evaluated_residual")}
    manifest_rows = as_int(cv["manifest_rows"], "manifest_rows")
    equal(manifest_rows - sum(drops.values()), n,
          f"{cov_loc}: manifest_rows minus all dropped_* equals evaluated queries", led)
    broad = as_int(cv["broad_group_slides"], "broad_group_slides")
    fine = as_int(cv["fine_labelled_slides"], "fine_labelled_slides")
    nosub = as_int(cv["no_subtype"], "no_subtype")
    xeno_ex = as_int(cv["xenograft_excluded"], "xenograft_excluded")
    nofus = as_int(cv["no_fusion_status"], "no_fusion_status")
    broad_name = str(cv["broad_group"])
    exp_name, exp_broad = EXPECTED_BROAD[arm]
    equal(broad_name, exp_name, f"{cov_loc}: broad_group name (expected per arm)", led)
    equal(broad, exp_broad, f"{cov_loc}: broad_group_slides (expected per arm)", led)
    if kind == "fusion":
        equal(nosub + xeno_ex + nofus + fine, broad,
              f"{cov_loc}: broad = no_subtype + xenograft_excluded + no_fusion_status + fine_labelled", led)
        equal(fine, n, f"{cov_loc}: fusion fine_labelled (saved status) equals evaluated queries", led)
        equal(nofus, drops["dropped_no_fusion_status"], f"{cov_loc}: no_fusion_status equals dropped_no_fusion_status", led)
    else:
        equal(nosub + fine, broad, f"{cov_loc}: broad = no_subtype + fine_labelled", led)
        equal(manifest_rows, fine, f"{cov_loc}: manifest_rows equals fine_labelled", led)
        equal(nofus, 0, f"{cov_loc}: no_fusion_status is 0 outside the fusion arm", led)
    if kind in ("internal", "fusion"):
        equal(xeno_ex, drops["dropped_xenograft"], f"{cov_loc}: xenograft_excluded equals dropped_xenograft", led)
    else:
        equal(xeno_ex, 0, f"{cov_loc}: xenograft_excluded is 0 for external queries", led)
    if n > fine:
        raise Mismatch(f"{cov_loc}: more evaluated queries than fine-labelled slides")

    label_text, label_bind = verify_label_source(arm, recon, truth, labels, n, broad, fine, manifest_rows, led)

    # Per-class recall from confusion rows, cross-checked against summary class rows.
    per_class = []
    for t in labels:
        nt = int((truth == t).sum())
        ct = conf["plurality"][1][(t, t)]
        st = conf["strict"][1][(t, t)]
        row, loc = summary_row(summ, arm, t)
        equal(nt, int(row["n_queries"]), f"class {t} n: {pq_name} vs {loc}", led)
        if nt == 0:
            raise Mismatch(f"{arm}: class {t} has no queries")
        close(ct / nt, row["plurality5_recall"], f"class {t} plurality5 recall vs {loc}", led)
        close(st / nt, row["strict5_recall"], f"class {t} strict5 recall vs {loc}", led)
        if unit == "patient":
            equal(int(pq.loc[truth == t, "patient"].nunique()), int(row["n_patients"]),
                  f"class {t} n_patients vs {loc}", led)
        per_class.append((t, ct, nt, row, loc))
    macro, macro_loc = summary_row(summ, arm, "MACRO")
    mcr = sum(c / m for _, c, m, _, _ in per_class) / len(per_class)
    close(mcr, macro["plurality5_recall"], f"mean class plurality5 recall vs {macro_loc}", led)

    maj_n = max(m for _, _, m, _, _ in per_class)
    majors = [t for t, _, m, _, _ in per_class if m == maj_n]
    majority_text = "; ".join(f"{t} {frac(maj_n, n)}" for t in majors) + (" (tied)" if len(majors) > 1 else "")
    min_n = min(m for _, _, m, _, _ in per_class)
    minority = [x for x in per_class if x[2] == min_n]
    tie = " (tied minority)" if len(minority) > 1 else ""

    p_ci, p_ci_v, p_deg = ci_for(all_row, "plurality5_recall", unit, all_loc, led)
    s_ci, s_ci_v, s_deg = ci_for(all_row, "strict5_recall", unit, all_loc, led)
    mc_ci, mc_ci_v, mc_deg = ci_for(macro, "plurality5_recall", unit, macro_loc, led)
    cls_ci = {t: ci_for(row, "plurality5_recall", unit, loc, led) for t, _, _, row, loc in per_class}
    for t, _, _, row, loc in per_class:
        ci_for(row, "strict5_recall", unit, loc, led)
    m_cis = [(t,) + cls_ci[t] for t, _, _, _, _ in minority]
    if len(m_cis) == 1:
        min_ci = m_cis[0][1]
    else:
        min_ci = "; ".join(f"{t}: {c}" for t, c, _, _ in m_cis)

    if unit == "patient" and p_ci_v is not None:
        unc = (f"whole-patient bootstrap 95% percentile CI in frozen output (B={recon['B']}, seed {recon['seed']}); "
               "conditional on the fixed reference library")
    elif p_ci == NOT_PATIENT_CI:
        unc = NOT_PATIENT_CI
    else:
        unc = NO_CI

    # Exclusions: only counts/notes present in coverage.csv, README and reconciliation rules.
    ex = []
    if kind == "fusion":
        ex.append(f"xenografts not queried: {drops['dropped_xenograft']}")
        ex.append(f"fusion status not recorded in saved S6 output: {drops['dropped_no_fusion_status']}")
    elif drops["dropped_xenograft"]:
        ex.append(f"query xenografts excluded: {drops['dropped_xenograft']}")
    if meta.get("xeno"):
        ex.append("xenografts retained (sensitivity arm)")
    if drops["dropped_discordant_patient"]:
        ex.append(f"discordant-label patient slides excluded: {drops['dropped_discordant_patient']}"
                  + (f" ({', '.join(meta.get('drop_patients', []))})" if meta.get("drop_patients") else ""))
    if drops["dropped_label_filter"]:
        ex.append(f"labels outside {{{', '.join(labels)}}}: {drops['dropped_label_filter']}")
    if drops["dropped_not_evaluated_residual"]:
        note = str(cv["note"]) if is_num(cv["note"]) else "reason not recorded"
        ex.append(f"labelled but not evaluated: {drops['dropped_not_evaluated_residual']} ({note})")
    if nosub:
        ex.append(f"broad-group slides without a fine label in the family manifest: {nosub}")
    if kind == "ccdi":
        ex.append("query family defined after SW1 exclusion (duplicate _N-suffixed re-exported CCDI slides "
                  "removed; pre-exclusion count not in frozen output)")
        rrows = as_int(cv["reference_manifest_rows"], "reference_manifest_rows")
        rx = as_int(cv["reference_dropped_xenograft"], "reference_dropped_xenograft")
        rl = as_int(cv["reference_dropped_label_filter"], "reference_dropped_label_filter")
        equal(rrows - rx - rl, n_ref, f"{cov_loc}: reference manifest minus drops equals n_reference_slides", led)
        ex.append(f"reference: {cv['reference_manifest']} {rrows} rows minus {rx} xenografts "
                  f"minus {rl} outside-label = {n_ref} COMET slides")
    if kind in ("internal", "fusion"):
        ex.append("same-patient reference slides excluded per query")
    if kind == "ccdi":
        ex.append("no patient overlap between CCDI queries and the COMET reference (checked)")
    if kind == "kidrare":
        ex.append("query slide itself excluded; no same-patient exclusion possible (patient linkage unknown)")

    cav = []
    if label_is_molecular(arm):
        cav.append("assay-derived label: association between morphology retrieval and molecular status, "
                   "not morphology-visible truth")
    if arm == "s65_inpc":
        cav.append("INPC favorable/unfavorable is a composite prognostic category, not a single morphologic subtype")
    if "Spindle" in labels:
        sp = [x for x in per_class if x[0] == "Spindle"][0]
        sp_pat = f", {int(sp[3]['n_patients'])} patients" if unit == "patient" else ""
        cav.append(f"Spindle has {sp[2]} slides{sp_pat}: sparse, unstable estimate")
    if kind == "ccdi":
        cav.append(f"conditional labelled subset: of {broad} CCDI RMS slides (after SW1 exclusion), {fine} carry a "
                   f"CCDI histologic subtype label and {n} are {'/'.join(labels)}; not all-RMS performance")
    if kind == "kidrare":
        cav.append("no same-patient exclusion possible: slides of the same (unknown) patient may be retrieved as "
                   "neighbours, so point estimates may be optimistic and are not comparable with the "
                   "patient-excluded internal/CCDI arms (the query slide itself is excluded)")
        cav.append("slide units, patient linkage unknown; frozen slide-resampling intervals may be too narrow "
                   "and are not reported as patient-level")
    if arm == "s62_xeno":
        cav.append("differs from s62 by 30 slides: 28 xenografts plus 2 slides of discordant-label patient "
                   "SJHBB3FZN (README caveat 6)")
    if p_correct <= maj_n:
        cav.append(f"pooled plurality@5 accuracy ({p_correct}/{n}) does not exceed the majority-class rate "
                   f"({maj_n}/{n}); read mean class recall and minority-class recall")
    if unit == "patient" and n_pat is not None and n_pat < n:
        cav.append(f"{n} slides from {n_pat} patients pooled at slide level; only intervals account for clustering")
    if s_abst:
        cav.append(f"strict rule abstained on {s_abst}/{n} (counted not correct)")
    degen = []
    if p_deg:
        degen.append("ALL plurality5")
    if s_deg and not binary:
        degen.append("ALL strict5")
    if mc_deg:
        degen.append("MACRO plurality5")
    for t, _, _, d in m_cis:
        if d:
            degen.append(f"{t} plurality5")
    if degen:
        cav.append("degenerate interval (lo == hi) for " + ", ".join(degen)
                   + ": reflects a small/homogeneous set, not certainty")

    if kind == "fusion":
        coverage_text = (f"{n}/{broad} slides evaluated (denominator: {broad_name}); {n} non-xenograft slides "
                         f"with saved S6 fusion status; status not recorded for {nofus} others; xenografts not "
                         f"queried ({xeno_ex}); {nosub} without an RMS subtype in the family manifest")
    else:
        coverage_text = (f"{n}/{broad} slides evaluated (denominator: {broad_name}); "
                         f"{fine}/{broad} carry a fine label in the family manifest")

    fam, ltype = ARM_META[arm]
    row = {
        "arm": arm,
        "main_text": MAIN_TEXT.get(arm, SUPPLEMENT_ONLY),
        "label_family": fam,
        "label_type": ltype,
        "cohort": COHORT[kind],
        "label_source": label_text,
        "classes_n_slides": "; ".join(f"{t} {m}" for t, _, m, _, _ in per_class),
        "unit": unit,
        "n_queries": str(n),
        "n_patients": n_pat_disp,
        "n_reference_slides": str(n_ref),
        "label_coverage": coverage_text,
        "exclusions": "; ".join(ex) if ex else "none recorded",
        "plurality5_accuracy": frac(p_correct, n),
        "plurality5_accuracy_ci": p_ci,
        "majority_class_rate": majority_text,
        "mean_class_recall": f"{mcr:.3f}",
        "mean_class_recall_ci": mc_ci,
        "strict5_accuracy": BINARY_STRICT if binary else f"{frac(s_correct, n)}; abstained {s_abst}",
        "strict5_accuracy_ci": BINARY_STRICT if binary else s_ci,
        "per_class_plurality5_recall": "; ".join(f"{t} {frac(c, m)}" for t, c, m, _, _ in per_class),
        "minority_class_recall": "; ".join(f"{t} {frac(c, m)}" for t, c, m, _, _ in minority) + tie,
        "minority_class_recall_ci": min_ci,
        "uncertainty": unc,
        "caveats": "; ".join(cav) if cav else "none",
        "num_plurality5_correct": p_correct,
        "num_strict5_correct": s_correct,
        "num_strict5_abstain": s_abst,
        "num_majority_n": maj_n,
        "num_mean_class_recall": repr(mcr),
        "num_coverage_evaluated": n,
        "num_coverage_denominator": broad,
        "num_fine_labelled_slides": fine,
        "num_minority_correct": "; ".join(str(c) for _, c, _, _, _ in minority),
        "num_minority_n": "; ".join(str(m) for _, _, m, _, _ in minority),
    }
    for k in DISPLAY_COLUMNS:
        if "|" in row[k] or "\n" in row[k]:
            raise Mismatch(f"{arm}.{k}: display cell contains a pipe/newline")

    bind.update({
        "label_source": label_bind,
        "plurality5_correct": {"value": p_correct, "n": n,
                               "source": f"{pq_name} (all {n} rows; plurality5 == truth)",
                               "cross_checked": [f"{conf['plurality'][0]} diagonal",
                                                 f"{all_loc} plurality5_recall"]},
        "strict5_correct": {"value": s_correct, "abstain": s_abst, "n": n,
                            "displayed_as": BINARY_STRICT if binary else "count",
                            "source": f"{pq_name} (strict5 == truth; strict5 == ABSTAIN)",
                            "cross_checked": [f"{conf['strict'][0]} diagonal/ABSTAIN column",
                                              f"{all_loc} strict5_recall/strict5_abstain_rate"]},
        "majority_class": {"classes": majors, "n": maj_n, "of": n,
                           "source": f"{pq_name} truth counts", "cross_checked": "summary class rows n_queries"},
        "mean_class_recall": {"value": mcr, "source": "mean of per-class plurality5 recalls (confusion rows)",
                              "cross_checked": f"{macro_loc} plurality5_recall"},
        "per_class_plurality5": {t: {"correct": c, "n": m, "source": f"{conf['plurality'][0]} row truth={t}",
                                     "cross_checked": f"{loc} plurality5_recall/n_queries"}
                                 for t, c, m, _, loc in per_class},
        "minority_classes": [t for t, _, _, _, _ in minority],
        "n_patients": {"value": n_pat, "source": f"{pq_name} unique patient" if n_pat is not None
                       else f"{all_loc} n_patients == 'unknown'", "cross_checked": all_loc},
        "n_reference_slides": {"value": n_ref, "source": f"{all_loc} n_reference_slides",
                               "cross_checked": [cov_loc, f"frozen reconciliation.json reconciliation.{arm}.n_reference"]},
        "label_coverage": {"evaluated": n, "denominator": broad, "denominator_name": broad_name,
                           "denominator_source_named_in_coverage": str(cv["broad_source"]),
                           "fine_labelled": fine, "no_subtype": nosub, "xenograft_excluded": xeno_ex,
                           "no_fusion_status": nofus, "drops": drops, "manifest_rows": manifest_rows,
                           "expected_broad_group": [exp_name, exp_broad], "source": cov_loc},
        "ci": {"plurality5_accuracy": {"text": p_ci, "bounds": p_ci_v, "source": f"{all_loc} plurality5_recall_lo/hi"},
               "strict5_accuracy": {"text": BINARY_STRICT if binary else s_ci, "bounds": None if binary else s_ci_v,
                                    "source": f"{all_loc} strict5_recall_lo/hi"},
               "mean_class_recall": {"text": mc_ci, "bounds": mc_ci_v, "source": f"{macro_loc} plurality5_recall_lo/hi"},
               "minority": [{"class": t, "text": c, "bounds": v,
                             "source": f"{[x for x in minority if x[0] == t][0][4]} plurality5_recall_lo/hi"}
                            for t, c, v, _ in m_cis]},
    })
    ledger_root[arm] = {"bindings": bind, "checks": led}
    return row, {"correct": p_correct, "n": n, "class": {t: (c, m) for t, c, m, _, _ in per_class}}


def label_is_molecular(arm):
    return ARM_META[arm][1].startswith("molecular")


def unit_kind(u):
    u = str(u)
    if u == "patient":
        return "patient"
    if u.startswith("slide (patient linkage unknown"):
        return "slide"
    raise Mismatch(f"unrecognized unit {u!r}")


def run_spot_checks(stats):
    out = []
    for arm, what, cls, num, den in SPOT_CHECKS:
        if arm not in stats:
            raise Mismatch(f"spot check: arm {arm} missing")
        got = (stats[arm]["correct"], stats[arm]["n"]) if what == "plurality_correct" else stats[arm]["class"][cls]
        out.append({"arm": arm, "what": what, "class": cls, "expected": [num, den], "recomputed": list(got)})
        if tuple(got) != (num, den):
            raise Mismatch(f"spot check {arm} {what} {cls or ''}: expected {num}/{den}, got {got[0]}/{got[1]}")
    return out


def to_csv_text(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=DISPLAY_COLUMNS + NUMERIC_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def to_md_text(rows, recon, in_dir):
    lines = [
        "# Supplementary Table: inventory of all evaluated fine-grained retrieval tasks (R3.7)",
        "",
        "All thirteen arms evaluated in the frozen fine-label re-evaluation are listed so that the "
        "selection reported in the main text is visible (`main_text` column: RMS ERMS-vs-ARMS internal and "
        "external CCDI, and liver HB-vs-HCC; the Spindle class rows of the two 3-class RMS arms appear as a "
        "main-text footnote). Fusion, MYCN, INPC, NB-vs-GNB, xenograft-retained and KidRare arms are "
        "reported here only.",
        "",
        "**Legend.** *Conditional retrieval*: each query slide has a known broad diagnosis and is ranked "
        "only against the known-fine-label sub-library of the same diagnostic family "
        "(`n_reference_slides`), not against the whole library; results do not describe whole-library "
        "search. Saved S6 depth-10 rank lists were re-scored without re-ranking. The query patient's "
        "slides are excluded from the reference for internal COMET arms; external CCDI queries have no "
        "patient overlap with the COMET reference (checked); for KidRare only the query slide itself is "
        "excluded because patient linkage is unknown. k = 5. "
        f"*Plurality@5*: {recon['rules']['plurality5']}. *Strict@5*: {recon['rules']['strict5']}. "
        "In two-class arms strict@5 is identical to plurality@5 by construction (5 votes, 2 labels: no "
        f"abstention possible); those cells read '{BINARY_STRICT}' and are not independent confirmation. "
        "*Accuracy* = correct/evaluated query slides, pooled over slides. *Majority-class rate* = share of "
        "evaluated queries in the largest class (accuracy of always predicting it). *Mean class recall* "
        "(balanced accuracy) = unweighted mean of per-class plurality@5 recalls; its interval is the frozen "
        "MACRO row interval. Per-class values are plurality@5 recall (correct/class slides); minority class "
        "= class with fewest evaluated query slides. "
        "*Label coverage* = evaluated query slides / all slides of the named broad diagnostic group "
        "(denominator named in each cell), with the number that carry a fine label in the family "
        "manifest (not necessarily one of the arm's classes). *SW1*: a sweep that removed duplicate "
        "`_N`-suffixed re-exported CCDI slides; CCDI denominators are after this exclusion. "
        "*Label source* names the originating metadata field (ALIdf3Ali.NMYC.amplified..Y.N. = NMYC "
        "amplified Y/N). *Cohort*: internal COMET or external CCDI/KidRare. "
        "Intervals are copied from the frozen output only when it carries a whole-patient bootstrap "
        f"(B = {recon['B']}, percentile 2.5/97.5); they are conditional on the fixed reference library. "
        f"'{NO_CI}' marks absent intervals; KidRare slide-resampling intervals are not reported as "
        "patient-level, and KidRare point estimates may be optimistic because same-patient slides cannot "
        "be excluded. No new bootstrap was computed. Estimates are slide-level and slides of one patient "
        "are not independent. Machine-readable counts (num_* columns) are in task_inventory.csv.",
        "",
        "| " + " | ".join(DISPLAY_COLUMNS) + " |",
        "|" + "|".join("---" for _ in DISPLAY_COLUMNS) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r[c] for c in DISPLAY_COLUMNS) + " |")
    lines += ["", f"Source: `{in_dir}` (frozen; script sha256 {recon['script_sha256']}).", ""]
    return "\n".join(lines)


def verify_pins(in_dir):
    got = {}
    for f, want in PINNED_SHA256.items():
        h = sha256(os.path.join(in_dir, f))
        if h != want:
            raise Mismatch(f"pinned frozen input {f}: sha256 {h} != pinned {want}")
        got[f] = h
    return got


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in-dir", default=DEFAULT_IN)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    in_dir, out_dir = os.path.abspath(a.in_dir), os.path.abspath(a.out_dir)
    if os.path.realpath(out_dir).startswith(("/tmp", "/var/tmp")):
        print("REFUSED: output under /tmp", file=sys.stderr)
        return 1
    if os.path.exists(out_dir) and os.listdir(out_dir):
        print(f"REFUSED: output directory not empty: {out_dir}", file=sys.stderr)
        return 1
    try:
        inputs = verify_pins(in_dir)
        recon = json.load(open(os.path.join(in_dir, "reconciliation.json")))
        if recon.get("script_sha256") != FROZEN_D_SCRIPT_SHA256:
            raise Mismatch(f"frozen reconciliation.json script_sha256 {recon.get('script_sha256')} "
                           f"!= {FROZEN_D_SCRIPT_SHA256}")
        d_script_sha = sha256(D_SCRIPT)
        if d_script_sha != FROZEN_D_SCRIPT_SHA256:
            raise Mismatch(f"{D_SCRIPT} sha256 {d_script_sha} != frozen {FROZEN_D_SCRIPT_SHA256}")
        summ = pd.read_csv(os.path.join(in_dir, "summary_all_arms.csv"), dtype=str, keep_default_na=False)
        cov = pd.read_csv(os.path.join(in_dir, "coverage.csv"), dtype=str, keep_default_na=False)
        arms = list(dict.fromkeys(summ["arm"]))
        for src, s in (("summary_all_arms.csv", set(arms)), ("coverage.csv", set(cov["arm"])),
                       ("reconciliation.json arms", set(recon["arms"])),
                       ("reconciliation.json reconciliation", set(recon["reconciliation"]))):
            if s != set(ARM_META):
                raise Mismatch(f"{src} arm set {sorted(s)} != inventory arm set {sorted(ARM_META)}")
        expected_files = {"README.md", "summary_all_arms.csv", "coverage.csv", "reconciliation.json"}
        expected_files |= {f"{p}_{arm}.csv" for arm in arms
                           for p in ("per_query", "confusion_k5_plurality", "confusion_k5_strict")}
        if set(PINNED_SHA256) != expected_files:
            raise Mismatch("pinned file set differs from the frozen per-arm file set")
        ledger, rows, stats = {}, [], {}
        for arm in arms:
            r, st = build_arm(arm, in_dir, summ, cov, recon, ledger)
            rows.append(r)
            stats[arm] = st
        spots = run_spot_checks(stats)
        # Inputs must be unchanged after all reads.
        if verify_pins(in_dir) != inputs:
            raise Mismatch("frozen inputs changed during the run")
    except (Mismatch, KeyError, ValueError, TypeError, IndexError, FileNotFoundError) as e:
        print(f"REFUSED (no output written): {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    csv_text = to_csv_text(rows)
    md_text = to_md_text(rows, recon, in_dir)
    os.makedirs(out_dir, exist_ok=True)
    out = {}

    def write(name, text):
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        out[name] = sha256(p)

    write("task_inventory.csv", csv_text)
    write("task_inventory.md", md_text)
    write("reconciliation.json", json.dumps({
        "description": "Every inventory number bound to its frozen source file/row; each check recomputed "
                       "from per_query/confusion and compared to summary/coverage/frozen reconciliation "
                       f"(float tolerance {TOL}); every frozen input matched its pinned sha256.",
        "input_dir": in_dir, "spot_checks": spot_checks_doc(spots), "arms": ledger,
        "n_checks": sum(len(v["checks"]) for v in ledger.values())}, indent=1, default=str) + "\n")
    script = os.path.abspath(__file__)
    write("record.json", json.dumps({
        "script": script, "script_sha256": sha256(script),
        "interpreter": sys.executable,
        "command": " ".join([sys.executable, os.path.relpath(script, REPO)] + list(argv if argv is not None else sys.argv[1:])),
        "python": sys.version.split()[0], "pandas": pd.__version__,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_dir": in_dir, "input_sha256": inputs,
        "input_pins": "all frozen inputs matched PINNED_SHA256 before and after reading",
        "label_source_files": {arm: ledger[arm]["bindings"]["label_source"] for arm in arms},
        "main_text_selection": "docs/manuscript_supplement_edits.md R3.7-T4 (author decision), hardcoded in MAIN_TEXT",
        "frozen_d_script": D_SCRIPT, "frozen_d_script_sha256": d_script_sha,
        "n_arms": len(rows), "arms": arms,
        "new_bootstrap_computed": False,
        "outputs_sha256": dict(out)}, indent=1) + "\n")
    write("COMPLETE", json.dumps({"status": "COMPLETE", "outputs_sha256": out}, indent=1) + "\n")
    print(f"wrote {len(rows)} arms to {out_dir}")
    return 0


def spot_checks_doc(spots):
    return {"source": "Astra independently stated reproduction targets", "results": spots}


if __name__ == "__main__":
    sys.exit(main())
