#!/usr/bin/env python3
"""R3.5-L4: link-level concordance of transport cost with annotated tissue class on the saved P2 plans.

Binding specification: $PROSPER_OUT/r3_5_localisation_maps_20261017/contract_006/SPECIFICATION.md
(sha256 7e19b96b7359cb9ed1dd0790bca48f0f90ddfac92afb10372356c9dd04e258b9).

No OT plan is solved, no map is computed.  For every P2 pair row the saved exact plan (i = query row,
j = reference row, t = mass, m = cosine ground cost) is read, every link is typed from the existing tile labels
(binary TT / NN / MIX, unannotated excluded; 5-class concordant / discordant), and:
  primary (specification 4)   per-pair mass-weighted AUROC of m for discordant (y = 1) vs concordant (y = 0) annotated
                         links, eligible pairs >= 10 of each; per-reference mean over the same-group pair rows as
                         listed in pairs_P2.csv (upstream repeats kept); mean over references; whole-patient
                         percentile bootstrap (L1.make_patient_draws, B = 5000, seed 123); wording rule and rider
                         applied mechanically.
  secondary (specification 5) class-anchored AUROCs (MIX vs TT, MIX vs NN), five-class AUROC, other-group arm, mean link
                         cost by type and arm with paired same - other differences, sensitivities of the primary,
                         tile-level AUROC of the displayed map value c_j for f_j < 0.5 vs f_j >= 0.5.
  fidelity (specification 6)  F1 link/label join vs per_pair_D1, F2 plan identity, F3 c_j vs per_tile_cost_D2, F4
                         bootstrap determinism, F5 unit tests, F6 link counts vs the design-stage counts, F7 hash
                         readback.

Implementation choices (also written to [internal record]):
 1. Row/label binding exactly as contract_006/design_label_only_link_counts.py: L1.load_features(s) keys,
    L1.load_labels(s), label 0 where absent, L1.class_of(.., True / False), for query and reference alike.
 2. Each distinct (reference, query) plan is read once; its file bytes are hashed (F2) and the same bytes are
    parsed (np.load on the hashed buffer).  Per-plan statistics are copied to every pair row naming that plan.
 3. Per-pair AUROCs are stored whenever both classes have >= 1 link; eligibility (>= thr links of each class) is
    applied at aggregation time, so thresholds 5 / 10 / 20 share one per-pair table.
 4. AUROC = sklearn.metrics.roc_auc_score(y, m, sample_weight=t) (primary; unweighted = sensitivity), m cast
    float32 -> float64 exactly.  Positive class: discordant (primary, five-class), MIX (class-anchored),
    rows with f_j < 0.5 (tile level).  > 0.5 therefore always means the cross-class side costs more.
 5. Tile level (5.6) uses the binary classes (the primary's), f_j over annotated links at reference row j,
    c_j over all links at j; unweighted row AUROC (every reference row carries column mass 1 / n_ref).
 6. Bootstrap population of each estimate = the references with a finite per-reference value; draws from their
    sorted patients via L1.make_patient_draws (identical populations therefore share identical draws);
    L1.replicate_means / L1.summarise_reps.  n_pairs = eligible pair rows entering those references.
 7. Wording rule and rider: strict inequalities on the percentile bounds; the rider's restrictive clause
    (non-tumour-anchored CI spans 0.5 or < 15 references) takes precedence; any case the specification text does not
    cover is reported as such rather than mapped to a sentence.
 8. Gate order: bind -> F7 -> F5 -> plan pass (F2, F6, F1 inside) -> F3 -> F4 -> estimates.  Any FAIL writes
    checks.json + FAILED.json and stops before results.json and before any estimate is computed.
"""
import argparse
import datetime as _dt
import hashlib
import importlib.util
import io
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

CONTRACT_SHA = "7e19b96b7359cb9ed1dd0790bca48f0f90ddfac92afb10372356c9dd04e258b9"
STUDY = "r3_5_localisation_maps_20261017"
INPUTS = {
    "pairs_P2": ("s0_pairs/run_001/pairs_P2.csv",
                 "9b90ad629ee9e9b7bf04f43e913be5efa0000e9cc73180215260953ae187ffbe"),
    "plan_manifest": ("l2/run_003/plan_manifest.csv",
                      "7650333808829826dde73cc3112bca29015d185945b23e65d5ba8cab1b97d771"),
    "per_pair_D1": ("l2/run_003/per_pair_D1.csv.gz",
                    "8affa6412d89fd7f04f4c72df6db085e876059b9d96794273a7bbce05ab69d1f"),
    "per_tile_cost_D2": ("l2/run_003/per_tile_cost_D2.csv.gz",
                         "8a223931ec9873e274defab727d67ffda8fd7637d9416d987edb35b36a0b6717"),
    "design_counts": ("contract_006/design_label_only_link_counts.csv",
                      "6e5944181471e260d36bbf822953284175c0e00db0699a166c0c424d1b29dc0c"),
}
L1_SCRIPT_SHA = "b316497077dc03dd344918aeb8f47b441b85efc2ed45b71da65981d49650f47b"
TOL = 1e-9
THR = 10
THR_SENS = (5, 20)
MIN_REFS_RIDER = 15
STEP_LIMIT_S = 20 * 60
SAME, OTHER = "same_group", "other_group"

TYPE_UNANN, TYPE_TT, TYPE_NN, TYPE_MIX = 0, 1, 2, 3

WORDING = {
    "lo_gt_half_point_ge_070": ("Within a slide pair, transport links joining tiles of different annotated class carried "
                                "higher cost than links joining tiles of the same class (AUROC x, 95% CI lo–hi; "
                                "n reference patients)."),
    "lo_gt_half_point_lt_070": ("Within a slide pair, transport links joining tiles of different annotated class carried "
                                "modestly higher cost than links joining tiles of the same class (AUROC x, 95% CI "
                                "lo–hi; n reference patients)."),
    "ci_spans_half": "link cost did not distinguish same-class from cross-class links",
    "hi_lt_half": "adverse; reported verbatim in the main text beside the other map metrics",
}
WORDING_MAP_CLAUSE = {
    "lo_gt_half_point_ge_070": "The matched-cost map may then be described as marking tissue matched across annotated classes.",
    "lo_gt_half_point_lt_070": "The map may be described only as \"partly reflecting\" cross-class matching.",
    "ci_spans_half": ("Reported in the supplement; the Results reframing then rests on the class decomposition and the "
                      "exact additivity of the map only, and no cross-class description of the map is made."),
    "hi_lt_half": "Adverse result reported verbatim in the main text beside the other map metrics.",
}
RIDER = {
    "both": "the distinction held \"for both tumour and non-tumour tiles\"",
    "tumour_only": "the sentence must be limited to \"links from annotated tumour tiles\"",
    "not_covered": ("case not covered by the contract 4 rider text (the 'both' condition fails but the non-tumour-"
                    "anchored CI neither spans 0.5 nor is under-supported); the 'both' qualifier is not licensed"),
}


# ---------------------------------------------------------------- utilities
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _clean(o):
    """NaN -> None so results.json is strict JSON."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return None
    return o


def jdump(path, obj):
    Path(path).write_text(json.dumps(_clean(obj), indent=1, default=_json_default))


class Stop(Exception):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def study_root() -> Path:
    out = os.environ.get("PROSPER_OUT")
    if not out:
        raise Stop("PROSPER_OUT is not exported; run `export PROSPER_OUT=\"$PWD/prosper_revision_output\"`")
    return Path(out) / STUDY


def load_l1():
    path = repo_root() / "scripts" / "r3_5_l1_maps.py"
    spec = importlib.util.spec_from_file_location("r3_5_l1_maps", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def snapshot_script(out_root: Path) -> dict:
    me = Path(__file__).resolve()
    sha = sha256_file(me)
    snap = out_root / "script_snapshots"
    snap.mkdir(parents=True, exist_ok=True)
    existing = sorted(snap.glob(f"r3_5_l4_link_cost_concordance_v*_{sha[:8]}.py"))
    if existing:
        return {"snapshot": str(existing[0]), "sha256": sha, "created_now": False}
    n = len(list(snap.glob("r3_5_l4_link_cost_concordance_v*_*.py"))) + 1
    p = snap / f"r3_5_l4_link_cost_concordance_v{n}_{sha[:8]}.py"
    shutil.copyfile(me, p)
    return {"snapshot": str(p), "sha256": sha, "created_now": True}


# ---------------------------------------------------------------- pure kernels (unit-tested)
def link_types(c_ref_j, c_q_i) -> np.ndarray:
    """Binary link type per link: 0 unannotated (either end class 0), 1 TT, 2 NN, 3 MIX."""
    a, b = np.asarray(c_ref_j), np.asarray(c_q_i)
    out = np.zeros(len(a), dtype=np.int8)
    ann = (a > 0) & (b > 0)
    out[ann & (a == 1) & (b == 1)] = TYPE_TT
    out[ann & (a == 2) & (b == 2)] = TYPE_NN
    out[ann & (a != b)] = TYPE_MIX
    return out


def concordance5(c5_ref_j, c5_q_i) -> tuple:
    """(concordant, discordant) masks over 5-class annotated links."""
    a, b = np.asarray(c5_ref_j), np.asarray(c5_q_i)
    ann = (a > 0) & (b > 0)
    return ann & (a == b), ann & (a != b)


def auroc(pos_mask, neg_mask, score, weight=None) -> float:
    """AUROC of `score` for pos (y = 1) vs neg (y = 0); other entries dropped; NaN if either class is empty.

    weight given -> roc_auc_score(..., sample_weight=weight)."""
    pos, neg = np.asarray(pos_mask, bool), np.asarray(neg_mask, bool)
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    keep = pos | neg
    y = pos[keep].astype(np.int64)
    s = np.asarray(score, np.float64)[keep]
    w = None if weight is None else np.asarray(weight, np.float64)[keep]
    return float(roc_auc_score(y, s, sample_weight=w))


def eligible(n_pos, n_neg, thr) -> np.ndarray:
    return (np.asarray(n_pos) >= thr) & (np.asarray(n_neg) >= thr)


def weighted_mean(values, weights, mask) -> float:
    mask = np.asarray(mask, bool)
    w = np.asarray(weights, np.float64)[mask]
    if mask.sum() == 0 or w.sum() <= 0:
        return float("nan")
    return float((w * np.asarray(values, np.float64)[mask]).sum() / w.sum())


def tile_fc(j, t, m, ann_mask, conc_mask, n_ref, ref_annotated) -> tuple:
    """Per reference row: f_j = concordant annotated mass / annotated mass (NaN unless the row is annotated and has
    >= 1 annotated link); c_j = mass-weighted mean m over all links at j (NaN if unlinked)."""
    j = np.asarray(j, np.int64)
    t = np.asarray(t, np.float64)
    m = np.asarray(m, np.float64)
    ann, conc = np.asarray(ann_mask, bool), np.asarray(conc_mask, bool)
    mass = np.bincount(j, weights=t, minlength=n_ref)
    cm = np.bincount(j, weights=t * m, minlength=n_ref)
    c = np.full(n_ref, np.nan)
    c[mass > 0] = cm[mass > 0] / mass[mass > 0]
    amass = np.bincount(j[ann], weights=t[ann], minlength=n_ref)
    cmass = np.bincount(j[conc & ann], weights=t[conc & ann], minlength=n_ref)
    f = np.full(n_ref, np.nan)
    ok = np.asarray(ref_annotated, bool) & (amass > 0)
    f[ok] = cmass[ok] / amass[ok]
    return f, c


def tile_auroc(f, c) -> tuple:
    """(AUROC of c for f < 0.5 (positive) vs f >= 0.5, n_pos, n_neg); rows with NaN f dropped."""
    f = np.asarray(f, np.float64)
    fin = np.isfinite(f)
    pos = fin & (f < 0.5)
    neg = fin & (f >= 0.5)
    return auroc(pos, neg, c), int(pos.sum()), int(neg.sum())


def plan_stats(i, j, t, m, cq2, cr2, cq5, cr5) -> dict:
    """Every per-plan quantity of contracts 3-6 from the saved links and the row classes."""
    i, j = np.asarray(i, np.int64), np.asarray(j, np.int64)
    t, m = np.asarray(t, np.float64), np.asarray(m, np.float64)
    a2, b2 = np.asarray(cr2)[j], np.asarray(cq2)[i]
    ty = link_types(a2, b2)
    ann = ty > 0
    conc = (ty == TYPE_TT) | (ty == TYPE_NN)
    disc = ty == TYPE_MIX
    tt, nn = ty == TYPE_TT, ty == TYPE_NN
    c5, d5 = concordance5(np.asarray(cr5)[j], np.asarray(cq5)[i])
    rec = {"n_links": int(len(i)), "ann": int(ann.sum()), "TT": int(tt.sum()), "NN": int(nn.sum()),
           "MIX": int(disc.sum()), "conc": int(conc.sum()), "conc5": int(c5.sum()), "disc5": int(d5.sum()),
           "mass_total": float(t.sum()), "mass_ann_bin": float(t[ann].sum()), "mass_conc_bin": float(t[conc].sum()),
           "mass_ann_5": float(t[c5 | d5].sum()), "mass_conc_5": float(t[c5].sum())}
    rec["S_bin"] = rec["mass_conc_bin"] / rec["mass_ann_bin"] if rec["mass_ann_bin"] > 0 else float("nan")
    rec["S_5"] = rec["mass_conc_5"] / rec["mass_ann_5"] if rec["mass_ann_5"] > 0 else float("nan")
    rec["ann_share_bin"] = rec["mass_ann_bin"] / rec["mass_total"] if rec["mass_total"] > 0 else float("nan")
    rec["ann_share_5"] = rec["mass_ann_5"] / rec["mass_total"] if rec["mass_total"] > 0 else float("nan")
    rec["A_bin_w"] = auroc(disc, conc, m, t)
    rec["A_bin_u"] = auroc(disc, conc, m)
    rec["A_tanc_w"] = auroc(disc, tt, m, t)
    rec["A_ntanc_w"] = auroc(disc, nn, m, t)
    rec["A_5_w"] = auroc(d5, c5, m, t)
    rec["cost_TT"] = weighted_mean(m, t, tt)
    rec["cost_NN"] = weighted_mean(m, t, nn)
    rec["cost_MIX"] = weighted_mean(m, t, disc)
    n_ref = len(cr2)
    f, c = tile_fc(j, t, m, ann, conc, n_ref, np.asarray(cr2) > 0)
    rec["A_tile"], rec["tile_n_disc_rows"], rec["tile_n_conc_rows"] = tile_auroc(f, c)
    return rec, c


# metric -> (value column, positive-count column(s), negative-count column(s)); counts are summed when tuples
METRICS = {
    "A_bin_w": ("A_bin_w", ("MIX",), ("TT", "NN")),
    "A_bin_u": ("A_bin_u", ("MIX",), ("TT", "NN")),
    "A_tanc_w": ("A_tanc_w", ("MIX",), ("TT",)),
    "A_ntanc_w": ("A_ntanc_w", ("MIX",), ("NN",)),
    "A_5_w": ("A_5_w", ("disc5",), ("conc5",)),
    "A_tile": ("A_tile", ("tile_n_disc_rows",), ("tile_n_conc_rows",)),
    "cost_TT": ("cost_TT", ("TT",), ("TT",)),
    "cost_NN": ("cost_NN", ("NN",), ("NN",)),
    "cost_MIX": ("cost_MIX", ("MIX",), ("MIX",)),
}


def eligible_mask(pp: pd.DataFrame, metric: str, thr: int) -> np.ndarray:
    col, pos, neg = METRICS[metric]
    npos = pp[list(pos)].sum(axis=1).to_numpy()
    nneg = pp[list(neg)].sum(axis=1).to_numpy()
    return eligible(npos, nneg, thr) & np.isfinite(pp[col].to_numpy(dtype=float))


def per_reference_mean(pp: pd.DataFrame, metric: str, arm: str, thr: int = THR, distinct: bool = False) -> pd.DataFrame:
    """Mean of the metric over the reference's eligible pair rows of `arm` (rows as listed; upstream repeats kept
    unless distinct=True, which keeps the first row of each (reference, query) plan).  Returns reference, value,
    n_rows (eligible rows entering the mean)."""
    x = pp[pp.arm == arm]
    if distinct:
        x = x.drop_duplicates(["reference", "query"], keep="first")
    x = x[eligible_mask(x, metric, thr)]
    col = METRICS[metric][0]
    g = x.groupby("reference")[col]
    return pd.DataFrame({"value": g.mean(), "n_rows": g.size()}).reset_index()


def boot(L1, values, patients, n_rows, B: int, seed: int) -> dict:
    v = np.asarray(values, float)
    pats_in = [str(p) for p in patients]
    pats, mult = L1.make_patient_draws(pats_in, B, seed)
    reps = L1.replicate_means(v, pats_in, pats, mult)
    out = L1.summarise_reps(float(v.mean()) if len(v) else float("nan"), reps, len(v))
    out.update({"n_patients": int(len(set(pats_in))), "n_pairs": int(np.sum(n_rows)), "B": int(B), "seed": int(seed)})
    return out


def estimate(L1, pp, refinfo, metric, arm, B, seed, thr=THR, distinct=False, subset=None) -> tuple:
    pr = per_reference_mean(pp, metric, arm, thr, distinct).merge(refinfo, on="reference", how="left")
    if subset is not None:
        pr = pr[subset(pr)]
    pr = pr.sort_values("reference").reset_index(drop=True)
    e = boot(L1, pr.value, pr.ref_patient, pr.n_rows, B, seed)
    e.update({"metric": metric, "arm": arm, "eligibility_threshold": thr, "distinct_plans_only": distinct})
    return e, pr


def paired_difference(L1, pp, refinfo, metric, B, seed, thr=THR) -> dict:
    a = per_reference_mean(pp, metric, SAME, thr).rename(columns={"value": "same", "n_rows": "n_same"})
    b = per_reference_mean(pp, metric, OTHER, thr).rename(columns={"value": "other", "n_rows": "n_other"})
    x = a.merge(b, on="reference", how="inner").merge(refinfo, on="reference", how="left").sort_values("reference")
    e = boot(L1, x.same - x.other, x.ref_patient, x.n_same + x.n_other, B, seed)
    e.update({"metric": f"{metric} same_group - other_group (paired per reference)", "eligibility_threshold": thr,
              "n_pairs_note": "eligible pair rows of both arms in the paired references"})
    return e


def wording_rule(point: float, lo: float, hi: float) -> str:
    if lo > 0.5:
        return "lo_gt_half_point_ge_070" if point >= 0.70 else "lo_gt_half_point_lt_070"
    if hi < 0.5:
        return "hi_lt_half"
    return "ci_spans_half"


def fill_wording(branch: str, point: float, lo: float, hi: float, n_patients: int) -> str:
    text = WORDING[branch]
    if branch.startswith("lo_gt_half"):
        text = (text.replace("AUROC x", f"AUROC {point:.2f}").replace("lo–hi", f"{lo:.2f}–{hi:.2f}")
                .replace("n reference patients", f"{n_patients} reference patients"))
    return text


def rider_rule(t_lo: float, nt_lo: float, nt_hi: float, nt_n_refs: int, min_refs: int = MIN_REFS_RIDER) -> str:
    nt_estimable = nt_n_refs >= min_refs and nt_lo is not None and np.isfinite(nt_lo)
    if not nt_estimable or (nt_lo <= 0.5 <= nt_hi):
        return "tumour_only"
    if t_lo is not None and t_lo > 0.5 and nt_lo > 0.5:
        return "both"
    return "not_covered"


# ---------------------------------------------------------------- fidelity checks
def check_F5(repo: Path, out_dir: Path) -> dict:
    test = repo / "tests" / "test_r3_5_l4_link_cost_concordance.py"
    cmd = [sys.executable, "-m", "pytest", str(test.relative_to(repo)), "-q"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    (out_dir / "pytest_output.txt").write_text(f"$ {' '.join(cmd)}\n(cwd {repo}, {now()})\n\n{out}")
    summary = [ln for ln in proc.stdout.strip().splitlines() if "passed" in ln or "failed" in ln or "error" in ln]
    return {"command": " ".join(cmd), "cwd": str(repo), "returncode": proc.returncode, "wall_s": time.time() - t0,
            "summary": summary[-1] if summary else proc.stdout.strip()[-200:], "test_file_sha256": sha256_file(test),
            "pass": proc.returncode == 0}


def check_F7(files: list) -> dict:
    items = []
    for f in files:
        h = sha256_file(f["path"])
        exp = f.get("contract_sha256") or f["sha256"]
        items.append({"role": f["role"], "path": f["path"], "sha256_rerun": h, "expected": exp, "match": h == exp})
    return {"items": items, "n_files": len(items), "pass": bool(all(i["match"] for i in items))}


def close(a, b, tol=TOL) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    return both_nan | (np.abs(a - b) <= tol)


def check_F1(pp: pd.DataFrame, D1: pd.DataFrame) -> dict:
    d = D1.set_index("pair_id")
    x = pp.set_index("pair_id")
    rec = {"n_pairs_L4": int(len(x)), "n_pairs_D1": int(len(d)), "pair_id_sets_equal": set(x.index) == set(d.index)}
    if not rec["pair_id_sets_equal"]:
        rec["pass"] = False
        return rec
    d = d.loc[x.index]
    ok_all = True
    for mine, theirs in (("S_bin", "full_bin_S"), ("S_5", "full_5_S"), ("ann_share_bin", "full_bin_annotated_mass_share"),
                         ("ann_share_5", "full_5_annotated_mass_share")):
        a, b = x[mine].to_numpy(float), d[theirs].to_numpy(float)
        ok = close(a, b)
        diff = np.abs(a - b)
        rec[theirs] = {"max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else None,
                       "n_mismatch": int((~ok).sum()), "n_nan_L2": int(np.isnan(b).sum()), "n_nan_L4": int(np.isnan(a).sum())}
        ok_all &= bool(ok.all())
    rec["tolerance"] = TOL
    rec["pass"] = bool(ok_all)
    return rec


def check_F6(pp: pd.DataFrame, design: pd.DataFrame) -> dict:
    a = pp.reset_index(drop=True)
    b = design.reset_index(drop=True)
    rec = {"n_rows_L4": int(len(a)), "n_rows_design": int(len(b))}
    if len(a) != len(b):
        rec["pass"] = False
        return rec
    keys_ok = bool((a.reference.astype(str).values == b.reference.astype(str).values).all()
                   and (a["query"].astype(str).values == b["query"].astype(str).values).all()
                   and (a.arm.astype(str).values == b.arm.astype(str).values).all())
    rec["row_keys_equal"] = keys_ok
    ok = keys_ok
    for c in ("n_links", "ann", "TT", "NN", "MIX", "conc5", "disc5"):
        n_mis = int((a[c].astype(np.int64).values != b[c].astype(np.int64).values).sum())
        rec[f"n_mismatch_{c}"] = n_mis
        ok &= n_mis == 0
    rec["pass"] = bool(ok)
    return rec


def check_F3(cvals: dict, per_tile: pd.DataFrame) -> dict:
    max_diff, n_rows, n_mis, missing = 0.0, 0, 0, []
    for (r, q), g in per_tile.groupby(["reference", "query"], sort=False):
        c = cvals.get((r, q))
        if c is None:
            missing.append([r, q])
            continue
        rows = g.ref_row.to_numpy(np.int64)
        if rows.max() >= len(c):
            n_mis += len(rows)
            continue
        mine = c[rows]
        theirs = g.cost.to_numpy(float)
        ok = close(mine, theirs)
        n_mis += int((~ok).sum())
        dd = np.abs(mine - theirs)
        if np.isfinite(dd).any():
            max_diff = max(max_diff, float(np.nanmax(dd)))
        n_rows += len(rows)
    n_plans_l2 = int(per_tile.groupby(["reference", "query"]).ngroups)
    rec = {"n_rows_compared": n_rows, "n_plans_in_D2": n_plans_l2, "n_plans_L4": len(cvals),
           "n_plans_missing_in_L4": len(missing), "max_abs_diff": max_diff, "n_mismatch": n_mis,
           "n_per_tile_rows": int(len(per_tile)), "n_unlinked_rows_L4": int(sum(int(np.isnan(c).sum()) for c in cvals.values())),
           "tolerance": TOL}
    rec["pass"] = bool(n_mis == 0 and not missing and n_rows == len(per_tile) and n_plans_l2 == len(cvals))
    return rec


def check_F4(L1, pp, refinfo, B, seed) -> dict:
    a, _ = estimate(L1, pp, refinfo, "A_bin_w", SAME, B, seed)
    b, _ = estimate(L1, pp, refinfo, "A_bin_w", SAME, B, seed)
    c, _ = estimate(L1, pp, refinfo, "A_bin_w", SAME, B, seed + 1)
    k = ("ci95_lo", "ci95_hi")
    rec = {"endpoint": "primary (A_bin_w, same_group, pooled)", "seed": seed, "alt_seed": seed + 1,
           "identical_intervals_same_seed": all(a[x] == b[x] for x in k),
           "same_point_alt_seed": a["point"] == c["point"],
           "different_interval_alt_seed": any(a[x] != c[x] for x in k),
           "note": "estimate values are not recorded in this check (read only after all checks pass)"}
    rec["pass"] = bool(rec["identical_intervals_same_seed"] and rec["same_point_alt_seed"] and rec["different_interval_alt_seed"])
    return rec


# ---------------------------------------------------------------- plan pass
def plan_pass(L1, root: Path, p2: pd.DataFrame, manifest: pd.DataFrame, deadline: float) -> tuple:
    cache = {}

    def cls(s):
        if s not in cache:
            _, keys = L1.load_features(s)
            lab = L1.load_labels(s)
            labels = np.array([lab.get(k, 0) for k in keys], dtype=np.int8)
            cache[s] = (L1.class_of(labels, True), L1.class_of(labels, False))
        return cache[s]

    man = manifest.astype({"reference": str, "query": str}).set_index(["reference", "query"]).sha256.to_dict()
    per_plan, cvals, f2 = {}, {}, []
    for (r, q) in p2[["reference", "query"]].drop_duplicates().itertuples(index=False):
        if time.time() > deadline:
            raise Stop("plan pass exceeded the 20-minute step limit")
        path = root / "s1" / "run_001" / "refs" / r / "plans" / f"{q}.npz"
        raw = path.read_bytes()
        h = sha256_bytes(raw)
        z = np.load(io.BytesIO(raw))
        i, j, t, m = z["i"], z["j"], z["t"].astype(np.float64), z["m"].astype(np.float64)
        shape = tuple(int(v) for v in z["shape"])
        dist = float(z["distance"][0])
        cq2, cq5 = cls(q)
        cr2, cr5 = cls(r)
        f2.append({"reference": r, "query": q, "sha256": h, "manifest_sha256": man.get((r, q)),
                   "hash_ok": man.get((r, q)) == h, "shape_ok": shape == (len(cq2), len(cr2)),
                   "abs_diff_distance": abs(float((t * m).sum()) - dist)})
        if not f2[-1]["shape_ok"]:
            continue
        rec, c = plan_stats(i, j, t, m, cq2, cr2, cq5, cr5)
        per_plan[(r, q)] = rec
        cvals[(r, q)] = c
    f2 = pd.DataFrame(f2)
    rec_f2 = {"n_plans": int(len(f2)), "n_hash_mismatch": int((~f2.hash_ok).sum()),
              "n_shape_mismatch": int((~f2.shape_ok).sum()), "max_abs_diff_distance": float(f2.abs_diff_distance.max()),
              "n_distance_mismatch": int((f2.abs_diff_distance > TOL).sum()), "tolerance": TOL,
              "n_manifest_plans": int(len(manifest))}
    rec_f2["pass"] = bool(rec_f2["n_hash_mismatch"] == 0 and rec_f2["n_shape_mismatch"] == 0
                          and rec_f2["n_distance_mismatch"] == 0)
    rows = []
    for p in p2.itertuples(index=False):
        s = per_plan.get((p.reference, p.query))
        if s is None:
            continue
        rows.append({"pair_id": p.pair_id, "reference": p.reference, "query": p.query, "arm": p.arm,
                     "ref_patient": p.ref_patient, "ref_group": p.ref_group, "half": p.half, "ref_strict": p.ref_strict,
                     **s})
    return pd.DataFrame(rows), cvals, rec_f2, len(cache)


# ---------------------------------------------------------------- estimates
def build_estimates(L1, pp, refinfo, B, seed) -> tuple:
    E = lambda metric, arm=SAME, **kw: estimate(L1, pp, refinfo, metric, arm, B, seed, **kw)
    prim, prim_ref = E("A_bin_w")
    halves = {h: E("A_bin_w", subset=lambda d, h=h: d.half == h)[0] for h in ("development", "report")}
    branch = wording_rule(prim["point"], prim["ci95_lo"], prim["ci95_hi"])
    tanc, _ = E("A_tanc_w")
    ntanc, _ = E("A_ntanc_w")
    rb = rider_rule(tanc["ci95_lo"], ntanc["ci95_lo"], ntanc["ci95_hi"], ntanc["n_references"])
    primary = {
        "definition": "binary-class mass-weighted link AUROC (discordant = y 1), same-group arm, pooled references, "
                      f">= {THR} concordant and >= {THR} discordant annotated links",
        "estimate": prim, "halves_consistency_check": halves,
        "lead_prediction": "0.65-0.80 (stated in the contract before any link cost by type was read)",
        "wording_rule_branch": branch, "wording_rule_outcome": WORDING[branch],
        "wording_rule_outcome_filled": fill_wording(branch, prim["point"], prim["ci95_lo"], prim["ci95_hi"],
                                                    prim["n_patients"]),
        "wording_rule_map_clause": WORDING_MAP_CLAUSE[branch],
        "wording_rule_inputs": {"point": prim["point"], "ci95_lo": prim["ci95_lo"], "ci95_hi": prim["ci95_hi"],
                                "rule": "lo > 0.5 & point >= 0.70 -> lo_gt_half_point_ge_070; lo > 0.5 & point < 0.70 "
                                        "-> lo_gt_half_point_lt_070; hi < 0.5 -> hi_lt_half; else ci_spans_half"},
    }
    rider = {"branch": rb, "outcome": RIDER[rb],
             "inputs": {"tumour_anchored_ci95_lo": tanc["ci95_lo"], "nontumour_anchored_ci95_lo": ntanc["ci95_lo"],
                        "nontumour_anchored_ci95_hi": ntanc["ci95_hi"],
                        "nontumour_anchored_n_references": ntanc["n_references"], "min_references": MIN_REFS_RIDER},
             "rule": ("nontumour CI spans 0.5 (lo <= 0.5 <= hi) or < 15 references -> tumour_only; else both lo > 0.5 -> "
                      "both; else not_covered"),
             "note": "both anchored estimates go in the supplementary table whatever the outcome"}
    other = {m: E(m, OTHER)[0] for m in ("A_bin_w", "A_tanc_w", "A_ntanc_w", "A_5_w")}
    cost = {arm: {ty: E(f"cost_{ty}", arm)[0] for ty in ("TT", "NN", "MIX")} for arm in (SAME, OTHER)}
    cost["paired_same_minus_other"] = {ty: paired_difference(L1, pp, refinfo, f"cost_{ty}", B, seed) for ty in ("TT", "NN")}
    cost["lead_prediction"] = "TT lower in same-group than other-group pairs; NN difference near zero"
    sens = {
        "unweighted_auroc": E("A_bin_u")[0],
        "eligibility_threshold_5": E("A_bin_w", thr=5)[0],
        "eligibility_threshold_20": E("A_bin_w", thr=20)[0],
        "distinct_plans_only": E("A_bin_w", distinct=True)[0],
        "development_half": halves["development"],
        "report_half": halves["report"],
        "ref_strict_only": E("A_bin_w", subset=lambda d: d.ref_strict.astype(str) == "True")[0],
    }
    tile, _ = E("A_tile")
    tile["definition"] = ("per pair: unweighted AUROC of c_j (mass-weighted mean m over all links at reference row j) "
                          "for rows with f_j < 0.5 (positive) vs f_j >= 0.5, binary classes, >= 10 rows of each")
    secondary = {
        "1_class_anchored_same_group": {"tumour_anchored_MIX_vs_TT": tanc, "nontumour_anchored_MIX_vs_NN": ntanc},
        "2_five_class_same_group": E("A_5_w")[0],
        "3_other_group_arm": {"binary_primary": other["A_bin_w"], "tumour_anchored_MIX_vs_TT": other["A_tanc_w"],
                              "nontumour_anchored_MIX_vs_NN": other["A_ntanc_w"], "five_class": other["A_5_w"]},
        "4_mean_link_cost_by_type": cost,
        "5_sensitivities_of_primary": sens,
        "6_tile_level_displayed_map": tile,
    }
    return primary, rider, secondary


def per_reference_table(pp, refinfo) -> pd.DataFrame:
    out = refinfo.copy().set_index("reference")
    for arm, tag in ((SAME, "same"), (OTHER, "other")):
        x = pp[pp.arm == arm]
        out[f"n_rows_{tag}"] = x.groupby("reference").size().reindex(out.index).fillna(0).astype(int)
        out[f"n_distinct_plans_{tag}"] = x.groupby("reference")["query"].nunique().reindex(out.index).fillna(0).astype(int)
        for metric in METRICS:
            pr = per_reference_mean(pp, metric, arm).set_index("reference")
            out[f"{metric}_{tag}"] = pr.value.reindex(out.index)
            out[f"{metric}_{tag}_n_rows"] = pr.n_rows.reindex(out.index).fillna(0).astype(int)
    for thr in THR_SENS:
        pr = per_reference_mean(pp, "A_bin_w", SAME, thr).set_index("reference")
        out[f"A_bin_w_same_thr{thr}"] = pr.value.reindex(out.index)
        out[f"A_bin_w_same_thr{thr}_n_rows"] = pr.n_rows.reindex(out.index).fillna(0).astype(int)
    pr = per_reference_mean(pp, "A_bin_w", SAME, THR, distinct=True).set_index("reference")
    out["A_bin_w_same_distinct"] = pr.value.reindex(out.index)
    out["A_bin_w_same_distinct_n_rows"] = pr.n_rows.reindex(out.index).fillna(0).astype(int)
    for ty in ("TT", "NN"):
        out[f"cost_{ty}_same_minus_other"] = out[f"cost_{ty}_same"] - out[f"cost_{ty}_other"]
    return out.reset_index().sort_values("reference", key=lambda s: s.astype(int))


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="run directory, e.g. $PROSPER_OUT/r3_5_localisation_maps_20261017/l4/run_001")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
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

    # ---- bind
    t0 = time.time()
    L1, l1_path = load_l1()
    files = [binding(root / "contract_006" / "CONTRACT.md", "contract_006", CONTRACT_SHA),
             binding(l1_path, "l1_helpers", L1_SCRIPT_SHA)]
    for role, (rel, sha) in INPUTS.items():
        files.append(binding(root / rel, role, sha))
    bad = [f["role"] for f in files if f.get("matches_contract") is False]
    if bad:
        return fail("bind", f"sha256 mismatch vs contract table: {bad}")
    bindings = {"created": now(), "script": str(me), "script_sha256": script_sha, "snapshot": snap,
                "contract_sha256": CONTRACT_SHA, "B": args.B, "seed": args.seed, "files": files,
                "plans": "bound per plan against plan_manifest.csv (F2; per-plan hashes in checks.json summary)",
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

    # ---- F7, F5
    checks["F7_hash_readback"] = check_F7(files)
    if not checks["F7_hash_readback"]["pass"]:
        return fail("F7", "run-time re-hash differs from the contract table")
    checks["F5_unit_tests"] = check_F5(repo, d)
    if not checks["F5_unit_tests"]["pass"]:
        return fail("F5", f"pytest returncode {checks['F5_unit_tests']['returncode']}")

    # ---- plan pass (F2, F6, F1)
    t0 = time.time()
    P = lambda role: root / INPUTS[role][0]
    p2 = pd.read_csv(P("pairs_P2"), dtype=str)
    manifest = pd.read_csv(P("plan_manifest"), dtype={"reference": str, "query": str})
    pp, cvals, f2, n_slides = plan_pass(L1, root, p2, manifest, t0 + STEP_LIMIT_S)
    timing["plan_pass_s"] = time.time() - t0
    checks["F2_plan_identity"] = f2
    checks["F2_plan_identity"]["n_slides_labelled"] = n_slides
    if not f2["pass"] or len(pp) != len(p2):
        return fail("F2", f"plan identity failed or pair rows lost ({len(pp)} of {len(p2)})")
    design = pd.read_csv(P("design_counts"), dtype={"reference": str, "query": str})
    checks["F6_link_counts"] = check_F6(pp, design)
    if not checks["F6_link_counts"]["pass"]:
        return fail("F6", "link-type counts differ from the design-stage counts")
    D1 = pd.read_csv(P("per_pair_D1"), usecols=["pair_id", "full_bin_S", "full_5_S", "full_bin_annotated_mass_share",
                                                 "full_5_annotated_mass_share"], float_precision="round_trip")
    checks["F1_link_label_join"] = check_F1(pp, D1)
    if not checks["F1_link_label_join"]["pass"]:
        return fail("F1", "link/label join does not reproduce per_pair_D1")

    # ---- F3
    t0 = time.time()
    per_tile = pd.read_csv(P("per_tile_cost_D2"), dtype={"reference": str, "query": str}, float_precision="round_trip")
    checks["F3_map_quantity"] = check_F3(cvals, per_tile)
    del per_tile
    timing["F3_s"] = time.time() - t0
    if not checks["F3_map_quantity"]["pass"]:
        return fail("F3", "c_j does not reproduce per_tile_cost_D2")

    # ---- F4
    refinfo = p2[["reference", "ref_patient", "ref_group", "half", "ref_strict"]].drop_duplicates("reference")
    if refinfo.reference.duplicated().any() or refinfo.groupby("ref_patient").reference.nunique().max() != 1:
        return fail("F4", "reference / patient mapping is not one-to-one")
    checks["F4_bootstrap_determinism"] = check_F4(L1, pp, refinfo, args.B, args.seed)
    if not checks["F4_bootstrap_determinism"]["pass"]:
        return fail("F4", "bootstrap not deterministic / alt seed did not change the interval")
    checks["all_pass"] = True
    jdump(d / "checks.json", checks)

    # ---- estimates
    t0 = time.time()
    primary, rider, secondary = build_estimates(L1, pp, refinfo, args.B, args.seed)
    timing["estimates_s"] = time.time() - t0
    elig = {arm: {thr: {"n_pairs": int(eligible_mask(pp[pp.arm == arm], "A_bin_w", thr).sum()),
                        "n_references": int(pp[(pp.arm == arm)][eligible_mask(pp[pp.arm == arm], "A_bin_w", thr)].reference.nunique())}
                  for thr in (5, 10, 20)} for arm in (SAME, OTHER)}

    pp.to_csv(d / "per_pair_L4.csv.gz", index=False)
    per_reference_table(pp, refinfo).to_csv(d / "per_reference_L4.csv", index=False)
    timing["total_wall_s"] = time.time() - t_wall0
    timing["total_cpu_s"] = time.process_time() - t_cpu0
    results = {"created": now(), "stage": "R3.5-L4", "script": str(me), "script_sha256": script_sha,
               "snapshot": snap["snapshot"], "contract_sha256": CONTRACT_SHA, "B": args.B, "seed": args.seed,
               "generator": "L1.make_patient_draws (numpy Generator(PCG64(seed))), one multiplicity vector per replicate "
                            "over the sorted patients of the estimate's reference population; L1.replicate_means / "
                            "L1.summarise_reps; 95 % percentile interval",
               "unit": "per-pair statistic -> mean over the reference's eligible pair rows of the arm (rows as listed in "
                       "pairs_P2.csv) -> mean over references",
               "eligibility_counts_binary": elig,
               "primary": primary, "rider": rider, "secondary": secondary, "timing": timing}
    jdump(d / "results.json", results)
    outputs = {n: sha256_file(d / n) for n in ("bindings.json", "checks.json", "results.json", "per_pair_L4.csv.gz",
                                                "per_reference_L4.csv", "pytest_output.txt")}
    jdump(d / "COMPLETE.json", {"created": now(), "stage": "R3.5-L4", "script": str(me), "script_sha256": script_sha,
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
