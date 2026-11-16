#!/usr/bin/env python3
"""R3.5-L3: paired single-pair vs neighbour-averaged matched-cost map comparison from retained L1/L2 records.

Binding specification: $PROSPER_OUT/r3_5_localisation_maps_20261017/contract_005/SPECIFICATION.md
(sha256 6f387e56ff7a34da56443ab515338c50c2b9da2192c550aee47ec6e2b67e8b2d).

This stage recomputes no map.  It reads L1 family-B per-pair scores (submitted setting, design P2), the L2
per-reference yardstick and the L2 OT-row-level per-pair costs, and reports:
  primary (specification 3)   delta(r) = yard_auroc_A(r) - mean over r's 10 same-group P2 rows of auroc_A, over the
                         45 references with a yardstick; whole-patient percentile bootstrap (B = 5000, seed 123);
                         pooled single/avg means from the same resamples; count of delta > 0; the pre-registered
                         wording rule applied mechanically to the CI bounds; development/report halves.
  secondary (specification 4) within-reference SD of single-pair AUROC; other-group counterpart (46) and paired
                         same - other (45); OT-row-level k-curve from per_tile_cost_D2 (k in 1,2,3,5,10; 20
                         seeded subsets per (reference, k), k = 10 = the full set); Pearson r of row-level k = 10
                         vs yard_auroc_A; pointing/AP analogues.
  fidelity (specification 5)  F1 pair identity, F2 yardstick readback, F3 bootstrap determinism, F4 hash readback,
                         F5 unit tests (tests/test_r3_5_l3_pair_vs_average.py, run by this script).

Implementation choices (recorded here so the analyst can verify them):
 1. Bootstrap generator: numpy.random.default_rng(seed) (= Generator(PCG64(seed))); one multiplicity vector per
    replicate from rng.integers(p, size=p) over the population's sorted reference patients, exactly the
    L1.make_patient_draws convention used throughout R3.5.  Statistic = mean over references of the per-reference
    value, weighted by the patient's multiplicity.  One draw matrix per population (pooled 45, other-group 46,
    development, report, k-curve references); within a population every endpoint uses the same draws.
 2. single(r), its SD, AP and pointing analogues are means/SDs over the 10 same-group *pair rows* (duplicate
    (reference, query) plans kept with their multiplicity), the L1 per-reference convention and the one L2 used
    for the yardstick (np.mean(cs_same, axis=0) over the 10 pair rows, L2 IMPLEMENTATION_CHOICES item 4).
 3. k-curve "same-group queries" = the 10 same-group pair rows of the reference (multiset), so k = 10 is the
    exact row-level cost average underlying the L2 yardstick.  Subsets are drawn without replacement from the 10
    row positions with numpy.random.default_rng(123) per (reference, k); the number of distinct plans per
    reference is reported in per_reference_L3.csv (n_distinct_same_queries).
 4. Row-level AUROC: sklearn.metrics.roc_auc_score on rows with ref_label in {1,2,3,4,5}, positives ref_label == 1,
    score = -mean cost; NaN (reference skipped) when either class has no row.
 5. Wording rule: strict inequalities on the percentile bounds as written in specification 3; the verbatim branch
    text is stored, plus a copy with the numbers filled in.
 6. Gate order: bind (hash readback) -> F1 -> F2 -> F4 -> F5 -> estimates (F3 inside).  Any FAIL writes
    checks.json + FAILED.json and stops before results.json.
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CONTRACT_SHA = "6f387e56ff7a34da56443ab515338c50c2b9da2192c550aee47ec6e2b67e8b2d"
STUDY = "r3_5_localisation_maps_20261017"
INPUTS = {
    "per_pair_B": ("s2/run_003/per_pair_B.csv.gz",
                   "4e3eee1c47b6912bdfac7ddc1bc0d60ef070e2238d760468f113f0b2d2801cb6"),
    "per_reference_D2": ("l2/run_003/per_reference_D2.csv.gz",
                         "90cb7b826e32e40fe608ae9d1abc69a73fd472650a7b667d32ca38b098ed2e4b"),
    "per_tile_cost_D2": ("l2/run_003/per_tile_cost_D2.csv.gz",
                         "8a223931ec9873e274defab727d67ffda8fd7637d9416d987edb35b36a0b6717"),
    "report_estimates": ("l2/run_003/report_estimates.json",
                         "c49b2f1e9e11f8b3e37dedeb8c3a8c482f85bf259b8493c3e3001ed5b2c8bed4"),
    "per_pair_D2": ("l2/run_003/per_pair_D2.csv.gz", None),  # recorded at S0 (bind), read back at F4
}
SETTING = "B|meancost|none|None"
DESIGN = "P2"
F2_KEY = "D2|yard_auroc_A"
F2_TOL = 1e-9
KS = (1, 2, 3, 5, 10)
N_DRAWS = 20
N_SAME = 10
NEG_LABELS = (2, 3, 4, 5)
POS_LABEL = 1
STEP_LIMIT_S = 15 * 60

WORDING = {
    "ci_lo_gt_0": ("averaging the matched-cost map over a reference's same-diagnosis neighbours localised "
                   "annotated tumour better than a single pair's map (paired difference +\u0394 [lo, hi], "
                   "45 reference patients)"),
    "ci_spans_0": "no established difference between the single-pair and neighbour-averaged maps",
    "ci_hi_lt_0": "adverse: averaging does not help or hurts; report verbatim beside the 0.65",
}


# ---------------------------------------------------------------- utilities
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def binding(path, role, expected=None) -> dict:
    p = Path(path)
    st = p.stat()
    rec = {"role": role, "path": str(p), "resolved_path": str(p.resolve()), "size": st.st_size,
           "mtime_ns": st.st_mtime_ns, "sha256": sha256_file(p)}
    if expected is not None:
        rec["contract_sha256"] = expected
        rec["matches_contract"] = rec["sha256"] == expected
    return rec


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


class Stop(Exception):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def study_root() -> Path:
    out = os.environ.get("PROSPER_OUT")
    if not out:
        raise Stop("PROSPER_OUT is not exported; run `export PROSPER_OUT=\"$PWD/prosper_revision_output\"`")
    return Path(out) / STUDY


def snapshot_script(out_root: Path) -> dict:
    """Copy this script to $PROSPER_OUT/script_snapshots/r3_5_l3_pair_vs_average_v<N>_<sha8>.py (reuse if present)."""
    me = Path(__file__).resolve()
    sha = sha256_file(me)
    snap = out_root / "script_snapshots"
    snap.mkdir(parents=True, exist_ok=True)
    existing = sorted(snap.glob(f"r3_5_l3_pair_vs_average_v*_{sha[:8]}.py"))
    if existing:
        return {"snapshot": str(existing[0]), "sha256": sha, "created_now": False}
    n = len(list(snap.glob("r3_5_l3_pair_vs_average_v*_*.py"))) + 1
    p = snap / f"r3_5_l3_pair_vs_average_v{n}_{sha[:8]}.py"
    shutil.copyfile(me, p)
    return {"snapshot": str(p), "sha256": sha, "created_now": True}


# ---------------------------------------------------------------- data selection
def select_p2(per_pair_B: pd.DataFrame) -> pd.DataFrame:
    x = per_pair_B[(per_pair_B.setting == SETTING) & (per_pair_B.design == DESIGN)].copy()
    return x.reset_index(drop=True)


def per_reference_table(p2: pd.DataFrame, ref_d2: pd.DataFrame) -> pd.DataFrame:
    """One row per P2 reference (46): same-group means/SDs over pair rows, other-group means, L2 yardstick, deltas."""
    same = p2[p2.arm == "same_group"]
    other = p2[p2.arm == "other_group"]
    keys = ["reference", "ref_patient", "ref_group", "half", "ref_strict"]
    base = p2[keys].drop_duplicates("reference").set_index("reference")
    out = base.copy()
    gs = same.groupby("reference")
    out["n_same_rows"] = gs.size().reindex(out.index).fillna(0).astype(int)
    out["n_distinct_same_queries"] = gs["query"].nunique().reindex(out.index).fillna(0).astype(int)
    out["n_other_rows"] = other.groupby("reference").size().reindex(out.index).fillna(0).astype(int)
    out["n_annotated"] = p2.groupby("reference").n_annotated.first().reindex(out.index)
    out["single_mean"] = gs.auroc_A.mean().reindex(out.index)
    out["single_sd"] = gs.auroc_A.std(ddof=1).reindex(out.index)
    out["single_ap_mean"] = gs.ap_A.mean().reindex(out.index)
    out["single_pointing_mean"] = gs.pointing_A.mean().reindex(out.index)
    go = other.groupby("reference")
    out["other_mean"] = go.auroc_A.mean().reindex(out.index)
    out["other_ap_mean"] = go.ap_A.mean().reindex(out.index)
    out["other_pointing_mean"] = go.pointing_A.mean().reindex(out.index)
    y = ref_d2.set_index("reference")
    out["yard"] = y.yard_auroc_A.reindex(out.index)
    out["yard_ap"] = y.yard_ap_A.reindex(out.index)
    out["yard_pointing"] = y.yard_pointing_A.reindex(out.index)
    out["yard_n_annotated"] = y.yard_n_annotated.reindex(out.index)
    out["delta"] = out.yard - out.single_mean
    out["delta_ap"] = out.yard_ap - out.single_ap_mean
    out["delta_pointing"] = out.yard_pointing - out.single_pointing_mean
    out["single_minus_other"] = out.single_mean - out.other_mean
    return out.reset_index()


def primary_frame(tab: pd.DataFrame) -> pd.DataFrame:
    """The 45 references entering specification 3: yardstick non-null and exactly N_SAME same-group rows."""
    m = tab.yard.notna() & (tab.n_same_rows == N_SAME)
    return tab[m].sort_values("reference").reset_index(drop=True)


# ---------------------------------------------------------------- bootstrap (L1.make_patient_draws convention)
def make_draws(patient_ids, B: int, seed: int):
    patients = tuple(sorted(set(str(p) for p in patient_ids)))
    p = len(patients)
    mult = np.zeros((B, p), dtype=np.int64)
    if p:
        rng = np.random.default_rng(seed)
        for b in range(B):
            mult[b] = np.bincount(rng.integers(p, size=p), minlength=p)
    return patients, mult


def replicate_means(values, patients, pats, mult) -> np.ndarray:
    v = np.asarray(values, float)
    idx = pd.Index(pats).get_indexer([str(p) for p in patients])
    if (idx < 0).any():
        raise Stop("a reference patient is missing from the draw population")
    W = mult[:, idx].astype(float)
    den = W.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (W @ v) / np.where(den > 0, den, 1), np.nan)


def boot_estimate(values, patients, B: int, seed: int, draws=None) -> dict:
    """Mean over references; percentile 95 % interval over whole-patient resamples.

    `draws` = (pats, mult) lets several endpoints share one draw matrix; otherwise drawn here from `patients`."""
    v = np.asarray(values, float)
    pats_in = [str(p) for p in patients]
    if draws is None:
        draws = make_draws(pats_in, B, seed)
    pats, mult = draws
    reps = replicate_means(v, pats_in, pats, mult)
    fin = reps[np.isfinite(reps)]
    return {"point": float(v.mean()), "ci95_lo": float(np.percentile(fin, 2.5)),
            "ci95_hi": float(np.percentile(fin, 97.5)), "n_references": int(len(v)),
            "n_patients": int(len(set(pats_in))), "B": int(mult.shape[0]), "seed": int(seed),
            "n_replicates_finite": int(len(fin))}


def wording_rule(lo: float, hi: float) -> str:
    if lo > 0:
        return "ci_lo_gt_0"
    if hi < 0:
        return "ci_hi_lt_0"
    return "ci_spans_0"


def fill_wording(branch: str, point: float, lo: float, hi: float, n: int) -> str:
    text = WORDING[branch]
    if branch == "ci_lo_gt_0":
        text = text.replace("+\u0394 [lo, hi]", f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]").replace(
            "45 reference patients", f"{n} reference patients")
    return text


# ---------------------------------------------------------------- k-curve (OT-row level)
def kcurve_subsets(n: int, k: int, n_draws: int, seed: int) -> list:
    """Index subsets of size k from range(n) without replacement; k == n gives the single full set."""
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    if k == n:
        return [np.arange(n)]
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n, size=k, replace=False)) for _ in range(n_draws)]


def row_auroc(score: np.ndarray, label: np.ndarray) -> float:
    """AUROC of `score` for label == POS_LABEL vs label in NEG_LABELS; label 0 rows dropped; NaN if a class is empty."""
    lab = np.asarray(label)
    keep = (lab == POS_LABEL) | np.isin(lab, NEG_LABELS)
    y = (lab[keep] == POS_LABEL)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, float)[keep]))


def kcurve_per_reference(per_tile: pd.DataFrame, same_rows: pd.DataFrame, ks=KS, n_draws=N_DRAWS, seed=123) -> pd.DataFrame:
    """For each reference with same-group pair rows: mean over seeded subsets of the row-level AUROC of -mean cost.

    same_rows: the same-group P2 pair rows (reference, query), duplicates kept.  Returns one row per reference with
    columns k_<k> (mean over subsets), n_subsets_<k>, n_rows_pos, n_rows_neg, n_queries."""
    recs = []
    for r, g in same_rows.groupby("reference"):
        queries = g["query"].tolist()
        t = per_tile[per_tile.reference == r]
        if not len(t):
            continue
        lab_row = t.drop_duplicates("ref_row").set_index("ref_row").ref_label.sort_index()
        keep_rows = lab_row[(lab_row == POS_LABEL) | lab_row.isin(NEG_LABELS)].index.to_numpy()
        lab = lab_row.loc[keep_rows].to_numpy()
        n_pos, n_neg = int((lab == POS_LABEL).sum()), int(np.isin(lab, NEG_LABELS).sum())
        rec = {"reference": r, "n_queries": len(queries), "n_rows_pos": n_pos, "n_rows_neg": n_neg}
        if n_pos == 0 or n_neg == 0:
            for k in ks:
                rec[f"k_{k}"] = float("nan")
                rec[f"n_subsets_{k}"] = 0
            recs.append(rec)
            continue
        cost_by_q = {}
        for q, tq in t.groupby("query"):
            cost_by_q[q] = tq.set_index("ref_row").cost.reindex(keep_rows).to_numpy()
        missing = [q for q in queries if q not in cost_by_q]
        if missing:
            raise Stop(f"reference {r}: same-group queries {missing} absent from per_tile_cost_D2")
        C = np.vstack([cost_by_q[q] for q in queries])  # (n_queries, n_labelled_rows), duplicates kept
        for k in ks:
            subs = kcurve_subsets(len(queries), k, n_draws, seed)
            vals = [row_auroc(-C[s].mean(axis=0), lab) for s in subs]
            rec[f"k_{k}"] = float(np.mean(vals))
            rec[f"n_subsets_{k}"] = len(subs)
        recs.append(rec)
    return pd.DataFrame(recs)


# ---------------------------------------------------------------- fidelity checks
def check_F1(p2: pd.DataFrame, per_pair_D2: pd.DataFrame, tab: pd.DataFrame) -> dict:
    ids_b = set(p2.pair_id)
    ids_d2 = set(per_pair_D2.pair_id)
    with_yard = tab[tab.yard.notna()]
    no_yard = tab[tab.yard.isna()]
    rec = {
        "n_p2_rows_B": int(len(p2)), "n_pair_id_B": int(len(ids_b)), "n_pair_id_D2": int(len(ids_d2)),
        "pair_id_sets_equal": ids_b == ids_d2, "n_only_B": int(len(ids_b - ids_d2)), "n_only_D2": int(len(ids_d2 - ids_b)),
        "n_references_with_yard": int(len(with_yard)),
        "n_same_rows_per_yard_reference": sorted(set(with_yard.n_same_rows.astype(int))),
        "all_yard_references_have_10_same_rows": bool((with_yard.n_same_rows == N_SAME).all()),
        "references_without_yard": [int(r) for r in no_yard.reference],
        "no_yard_references_same_rows": [int(v) for v in no_yard.n_same_rows],
        "n_same_rows_total": int(tab.n_same_rows.sum()), "n_other_rows_total": int(tab.n_other_rows.sum()),
        "one_reference_per_patient": bool(tab.groupby("ref_patient").reference.nunique().max() == 1),
    }
    rec["pass"] = bool(rec["pair_id_sets_equal"] and rec["n_pair_id_B"] == 910 and rec["n_p2_rows_B"] == 910
                       and rec["n_references_with_yard"] == 45 and rec["all_yard_references_have_10_same_rows"]
                       and rec["references_without_yard"] == [13748] and rec["no_yard_references_same_rows"] == [0]
                       and rec["one_reference_per_patient"])
    return rec


def check_F2(prim: pd.DataFrame, report_estimates: dict) -> dict:
    target = report_estimates["pooled"][F2_KEY]
    mean_yard = float(prim.yard.mean())
    rec = {"mean_yard_auroc_A_45": mean_yard, "target_point": float(target["point"]),
           "target_n_references": int(target["n_references"]), "abs_diff": abs(mean_yard - float(target["point"])),
           "tolerance": F2_TOL, "n_references": int(len(prim))}
    rec["pass"] = bool(rec["abs_diff"] <= F2_TOL and rec["n_references"] == rec["target_n_references"] == 45)
    return rec


def check_F3(prim: pd.DataFrame, B: int, seed: int) -> tuple:
    a = boot_estimate(prim.delta, prim.ref_patient, B, seed)
    b = boot_estimate(prim.delta, prim.ref_patient, B, seed)
    c = boot_estimate(prim.delta, prim.ref_patient, B, seed + 1)
    rec = {"seed": seed, "run1": {k: a[k] for k in ("point", "ci95_lo", "ci95_hi")},
           "run2": {k: b[k] for k in ("point", "ci95_lo", "ci95_hi")},
           "alt_seed": seed + 1, "alt": {k: c[k] for k in ("point", "ci95_lo", "ci95_hi")},
           "identical_intervals_same_seed": (a["ci95_lo"] == b["ci95_lo"] and a["ci95_hi"] == b["ci95_hi"]),
           "same_point_alt_seed": a["point"] == c["point"],
           "different_interval_alt_seed": (a["ci95_lo"] != c["ci95_lo"] or a["ci95_hi"] != c["ci95_hi"])}
    rec["pass"] = bool(rec["identical_intervals_same_seed"] and rec["same_point_alt_seed"] and rec["different_interval_alt_seed"])
    return rec, a


def check_F4(bind_files: list) -> dict:
    items = []
    for f in bind_files:
        h = sha256_file(f["path"])
        exp = f.get("contract_sha256") or f["sha256"]
        items.append({"role": f["role"], "path": f["path"], "sha256_rerun": h, "expected": exp,
                      "source": "contract_table" if f.get("contract_sha256") else "recorded_at_S0_bindings", "match": h == exp})
    return {"items": items, "pass": bool(all(i["match"] for i in items))}


def check_F5(repo: Path, out_dir: Path) -> dict:
    test = repo / "tests" / "test_r3_5_l3_pair_vs_average.py"
    cmd = [sys.executable, "-m", "pytest", str(test.relative_to(repo)), "-q"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    (out_dir / "pytest_output.txt").write_text(f"$ {' '.join(cmd)}\n(cwd {repo}, {now()})\n\n{out}")
    summary = [ln for ln in proc.stdout.strip().splitlines() if "passed" in ln or "failed" in ln or "error" in ln]
    return {"command": " ".join(cmd), "cwd": str(repo), "returncode": proc.returncode, "wall_s": time.time() - t0,
            "summary": summary[-1] if summary else proc.stdout.strip()[-200:], "test_file_sha256": sha256_file(test),
            "pass": proc.returncode == 0}


# ---------------------------------------------------------------- estimates
def build_estimates(tab: pd.DataFrame, prim: pd.DataFrame, kc: pd.DataFrame, B: int, seed: int) -> tuple:
    """All specification 3/4 estimates.  Returns (results dict, F3 record)."""
    f3, _ = check_F3(prim, B, seed)
    pats45, mult45 = make_draws(prim.ref_patient, B, seed)
    d45 = (pats45, mult45)
    est = lambda s, d=d45, pats=prim.ref_patient: boot_estimate(s, pats, B, seed, draws=d)

    primary = {
        "delta_auroc_A": est(prim.delta), "single_auroc_A": est(prim.single_mean), "avg_auroc_A": est(prim.yard),
        "n_delta_positive": int((prim.delta > 0).sum()), "n_delta_zero": int((prim.delta == 0).sum()),
        "n_delta_negative": int((prim.delta < 0).sum()),
    }
    lo, hi = primary["delta_auroc_A"]["ci95_lo"], primary["delta_auroc_A"]["ci95_hi"]
    branch = wording_rule(lo, hi)
    primary["wording_rule_branch"] = branch
    primary["wording_rule_outcome"] = WORDING[branch]
    primary["wording_rule_outcome_filled"] = fill_wording(branch, primary["delta_auroc_A"]["point"], lo, hi,
                                                          primary["delta_auroc_A"]["n_references"])
    primary["wording_rule_inputs"] = {"ci95_lo": lo, "ci95_hi": hi, "rule": "lo > 0 -> ci_lo_gt_0; hi < 0 -> ci_hi_lt_0; else ci_spans_0"}

    halves = {}
    for h in ("development", "report"):
        ph = prim[prim.half == h]
        dh = make_draws(ph.ref_patient, B, seed)
        halves[h] = {"delta_auroc_A": boot_estimate(ph.delta, ph.ref_patient, B, seed, draws=dh),
                     "single_auroc_A": boot_estimate(ph.single_mean, ph.ref_patient, B, seed, draws=dh),
                     "avg_auroc_A": boot_estimate(ph.yard, ph.ref_patient, B, seed, draws=dh),
                     "n_delta_positive": int((ph.delta > 0).sum())}

    sd = prim.single_sd
    q = sd.quantile([0.25, 0.5, 0.75])
    other46 = tab[tab.other_mean.notna()].sort_values("reference")
    d46 = make_draws(other46.ref_patient, B, seed)
    secondary = {
        "single_sd_within_reference": {"median": float(q[0.5]), "iqr_lo": float(q[0.25]), "iqr_hi": float(q[0.75]),
                                       "mean": float(sd.mean()), "min": float(sd.min()), "max": float(sd.max()),
                                       "n_references": int(len(sd))},
        "other_group_auroc_A_46": boot_estimate(other46.other_mean, other46.ref_patient, B, seed, draws=d46),
        "same_minus_other_auroc_A_45": est(prim.single_minus_other),
        "delta_ap_A": est(prim.delta_ap), "single_ap_A": est(prim.single_ap_mean), "avg_ap_A": est(prim.yard_ap),
        "delta_pointing_A": est(prim.delta_pointing), "single_pointing_A": est(prim.single_pointing_mean),
        "avg_pointing_A": est(prim.yard_pointing),
        "n_delta_ap_positive": int((prim.delta_ap > 0).sum()),
        "n_delta_pointing_positive": int((prim.delta_pointing > 0).sum()),
    }

    kcv = kc.merge(prim[["reference", "ref_patient", "yard"]], on="reference", how="inner")
    kcv = kcv[kcv[[f"k_{k}" for k in KS]].notna().all(axis=1)].sort_values("reference")
    dk = make_draws(kcv.ref_patient, B, seed)
    kcurve = {f"k_{k}": boot_estimate(kcv[f"k_{k}"], kcv.ref_patient, B, seed, draws=dk) for k in KS}
    r = float(np.corrcoef(kcv["k_10"], kcv["yard"])[0, 1]) if len(kcv) > 2 else float("nan")
    secondary["kcurve_row_level"] = {
        "unit": "mean over 20 seeded subsets per (reference, k) of the row-level AUROC of -mean cost; then mean over references",
        "ks": list(KS), "n_draws": N_DRAWS, "subset_seed": seed, "estimates": kcurve,
        "n_references": int(len(kcv)), "n_references_skipped": int(len(kc) - len(kcv)),
        "k10_vs_yard_auroc_A_pearson_r": r, "k10_row_level_mean": float(kcv["k_10"].mean()),
        "yard_patch_level_mean_same_references": float(kcv["yard"].mean()),
        "note": "row-level (no patch projection); k = 10 is not expected to equal the patch-level yardstick",
    }
    kcurve_rows = [{"k": k, **{kk: kcurve[f"k_{k}"][kk] for kk in ("point", "ci95_lo", "ci95_hi", "n_references")}} for k in KS]
    return {"primary": primary, "halves": halves, "secondary": secondary}, f3, kcurve_rows, kcv


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="run directory, e.g. $PROSPER_OUT/r3_5_localisation_maps_20261017/l3/run_001")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--skip-tests", action="store_true", help="debugging only; F5 is then recorded as not run (FAIL)")
    args = ap.parse_args(argv)

    t_wall0, t_cpu0 = time.time(), time.process_time()
    timing = {}
    root = study_root()
    out_root = root.parent
    repo = repo_root()
    d = Path(args.out)
    if d.exists() and any(d.iterdir()):
        raise Stop(f"{d} exists and is not empty; refusing to overwrite")
    d.mkdir(parents=True, exist_ok=True)
    me = Path(__file__).resolve()
    script_sha = sha256_file(me)
    snap = snapshot_script(out_root)
    checks = {"created": now(), "script": str(me), "script_sha256": script_sha, "contract_sha256": CONTRACT_SHA}

    def fail(stage, msg):
        checks["all_pass"] = False
        checks["failed_stage"] = stage
        checks["failure"] = msg
        jdump(d / "checks.json", checks)
        jdump(d / "FAILED.json", {"created": now(), "stage": stage, "message": msg, "script": str(me),
                                  "script_sha256": script_sha, "checks": "checks.json"})
        print(f"FAIL at {stage}: {msg}", file=sys.stderr)
        return 2

    # ---- S0 bind
    t0 = time.time()
    contract = root / "contract_005" / "CONTRACT.md"
    files = [binding(contract, "contract_005", CONTRACT_SHA)]
    if not files[0]["matches_contract"]:
        return fail("bind", f"contract sha256 {files[0]['sha256']} != {CONTRACT_SHA}")
    for role, (rel, sha) in INPUTS.items():
        files.append(binding(root / rel, role, sha))
    bad = [f["role"] for f in files if f.get("matches_contract") is False]
    if bad:
        return fail("bind", f"input sha256 mismatch vs contract table: {bad}")
    bindings = {"created": now(), "script": str(me), "script_sha256": script_sha, "snapshot": snap,
                "contract_sha256": CONTRACT_SHA, "B": args.B, "seed": args.seed, "files": files,
                "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                "sklearn": __import__("sklearn").__version__}
    jdump(d / "bindings.json", bindings)
    back = json.loads((d / "bindings.json").read_text())
    rb = all(sha256_file(f["path"]) == f["sha256"] for f in back["files"])
    bindings["readback"] = {"all_match": rb, "n_files": len(back["files"])}
    jdump(d / "bindings.json", bindings)
    if not rb:
        return fail("bind", "bindings.json readback mismatch")
    timing["bind_s"] = time.time() - t0

    # ---- load
    t0 = time.time()
    P = lambda role: root / INPUTS[role][0]
    per_pair_B = pd.read_csv(P("per_pair_B"))
    ref_d2 = pd.read_csv(P("per_reference_D2"))
    per_pair_D2 = pd.read_csv(P("per_pair_D2"), usecols=["pair_id", "reference", "query", "arm"])
    report_estimates = json.loads(P("report_estimates").read_text())
    p2 = select_p2(per_pair_B)
    tab = per_reference_table(p2, ref_d2)
    prim = primary_frame(tab)
    timing["load_s"] = time.time() - t0

    # ---- F1, F2, F4, F5 (gates before any estimate)
    checks["F1_pair_identity"] = check_F1(p2, per_pair_D2, tab)
    if not checks["F1_pair_identity"]["pass"]:
        return fail("F1", "pair identity / row-count check failed")
    checks["F2_yardstick_readback"] = check_F2(prim, report_estimates)
    if not checks["F2_yardstick_readback"]["pass"]:
        return fail("F2", "yardstick mean does not reproduce report_estimates pooled D2|yard_auroc_A")
    checks["F4_hash_readback"] = check_F4(files)
    if not checks["F4_hash_readback"]["pass"]:
        return fail("F4", "run-time re-hash differs from the contract table / S0 record")
    if args.skip_tests:
        checks["F5_unit_tests"] = {"pass": False, "note": "skipped by --skip-tests"}
        return fail("F5", "unit tests skipped")
    checks["F5_unit_tests"] = check_F5(repo, d)
    if not checks["F5_unit_tests"]["pass"]:
        return fail("F5", f"pytest returncode {checks['F5_unit_tests']['returncode']}")

    # ---- k-curve (only non-trivial step)
    t0 = time.time()
    per_tile = pd.read_csv(P("per_tile_cost_D2"))
    same_rows = p2[p2.arm == "same_group"][["reference", "query"]].reset_index(drop=True)
    kc = kcurve_per_reference(per_tile, same_rows, KS, N_DRAWS, args.seed)
    timing["kcurve_s"] = time.time() - t0
    if timing["kcurve_s"] > STEP_LIMIT_S:
        return fail("kcurve", f"k-curve step took {timing['kcurve_s']:.0f}s > {STEP_LIMIT_S}s")
    checks["kcurve_inputs"] = {"n_per_tile_rows": int(len(per_tile)), "n_same_group_pair_rows": int(len(same_rows)),
                               "n_distinct_same_group_plans": int(same_rows.drop_duplicates().shape[0]),
                               "ref_label_counts": {int(k): int(v) for k, v in per_tile.ref_label.value_counts().sort_index().items()},
                               "n_references_scored": int(kc[[f"k_{k}" for k in KS]].notna().all(axis=1).sum())}
    del per_tile

    # ---- estimates (F3 inside)
    t0 = time.time()
    results, f3, kcurve_rows, kcv = build_estimates(tab, prim, kc, args.B, args.seed)
    checks["F3_bootstrap_determinism"] = f3
    if not f3["pass"]:
        return fail("F3", "bootstrap not deterministic / alt seed did not change the interval")
    timing["estimates_s"] = time.time() - t0

    # informational (not a gate): reproduce the L2 pooled yardstick interval with L2's 46-patient population
    # and NaN-dropping convention, to show the draw generator matches L1.make_patient_draws.
    pop46 = tab.sort_values("reference")
    pats46, mult46 = make_draws(pop46.ref_patient, args.B, args.seed)
    y46 = boot_estimate(prim.yard, prim.ref_patient, args.B, args.seed, draws=(pats46, mult46))
    tgt = report_estimates["pooled"][F2_KEY]
    checks["info_l2_yardstick_ci_reproduction"] = {
        "note": "informational, not a gate; 46-patient draw matrix with the missing reference dropped per replicate (L2 est)",
        "this_script": {k: y46[k] for k in ("point", "ci95_lo", "ci95_hi", "n_replicates_finite")},
        "l2_report_estimates": {k: tgt[k] for k in ("point", "ci95_lo", "ci95_hi", "n_replicates_finite")},
        "intervals_equal_1e-12": bool(abs(y46["ci95_lo"] - tgt["ci95_lo"]) < 1e-12 and abs(y46["ci95_hi"] - tgt["ci95_hi"]) < 1e-12)}
    checks["all_pass"] = True
    jdump(d / "checks.json", checks)

    # ---- result files
    cols = ["reference", "ref_patient", "ref_group", "half", "ref_strict", "n_same_rows", "n_distinct_same_queries",
            "n_other_rows", "n_annotated", "yard_n_annotated", "single_mean", "single_sd", "other_mean", "yard", "delta",
            "single_minus_other", "single_ap_mean", "other_ap_mean", "yard_ap", "delta_ap", "single_pointing_mean",
            "other_pointing_mean", "yard_pointing", "delta_pointing"]
    per_ref = tab[cols].merge(kc[["reference", "n_rows_pos", "n_rows_neg"] + [f"k_{k}" for k in KS]].rename(
        columns={f"k_{k}": f"kcurve_row_auroc_k{k}" for k in KS}), on="reference", how="left").sort_values("reference")
    per_ref.to_csv(d / "per_reference_L3.csv", index=False)
    pd.DataFrame(kcurve_rows).to_csv(d / "kcurve_L3.csv", index=False)
    timing["total_wall_s"] = time.time() - t_wall0
    timing["total_cpu_s"] = time.process_time() - t_cpu0
    results = {"created": now(), "script": str(me), "script_sha256": script_sha, "snapshot": snap["snapshot"],
               "contract_sha256": CONTRACT_SHA, "B": args.B, "seed": args.seed,
               "generator": "numpy.random.default_rng(seed) (PCG64); one multiplicity vector per replicate over sorted "
                            "reference patients (L1.make_patient_draws convention); one draw matrix per population",
               "unit": "per-reference value (pair statistics averaged over the reference's 10 same-group P2 rows), mean over "
                       "references; whole-patient percentile bootstrap, 95 %",
               "populations": {"pooled_45": "P2 references with a yardstick (45 references = 45 patients)",
                               "other_group_46": "all 46 P2 references", "development": "pooled_45 with half == development",
                               "report": "pooled_45 with half == report",
                               "kcurve": f"pooled_45 references with a finite row-level AUROC at every k ({len(kcv)})"},
               "setting": SETTING, "design": DESIGN, **results, "timing": timing}
    jdump(d / "results.json", results)
    outputs = {n: sha256_file(d / n) for n in ("bindings.json", "checks.json", "results.json", "per_reference_L3.csv",
                                                "kcurve_L3.csv", "pytest_output.txt")}
    jdump(d / "COMPLETE.json", {"created": now(), "stage": "R3.5-L3", "script": str(me), "script_sha256": script_sha,
                                "snapshot": snap, "contract_sha256": CONTRACT_SHA, "B": args.B, "seed": args.seed,
                                "outputs": outputs, "timing": timing, "all_checks_pass": True,
                                "argv": sys.argv[1:], "cwd": os.getcwd(), "python": sys.version})
    print(json.dumps({"out": str(d), "all_pass": True, "timing": timing}, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        sys.exit(3)
