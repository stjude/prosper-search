#!/usr/bin/env python3
"""R3.5-L5: all-links version of the R3.5-L4 link-cost analysis (tumour-tumour links vs every other link).

Binding specification: $PROSPER_OUT/r3_5_localisation_maps_20261017/contract_007/SPECIFICATION.md
(sha256 b7c243830294682742fa0e322db165d46d61752459abbeb6f8f5fcf12e428a79).  Post hoc, descriptive stage.

No OT plan is solved, no feature / head / mask is recomputed.  For every P2 pair row the saved exact plan
(i = query row, j = reference row, t = mass, m = cosine ground cost) is read and every link is typed TT / NN / MIX
under four tile-class versions (specification 1):
  ann   annotation (tumour / pooled non-tumour); unannotated excluded           (the L4 population)
  fill  annotation where annotated, else the patient-held-out tumour-filter prediction   (primary of this stage)
  lit   tumour if annotated tumour, else other
  pred  patient-held-out tumour-filter prediction for every tile
Held-out prediction of feature row k of slide s = fold{slide_fold(s)}_mask[tile_start(s) + k] (TF-2, UNI2, b100).

  primary (specification 4)   B = mass-weighted AUROC of m, y = 1 for every non-TT link (MIX or NN), y = 0 for TT;
                         eligible pairs >= 10 TT and >= 10 non-TT links; per-reference mean over the reference's
                         eligible same-group pair rows (as listed in pairs_P2.csv); mean over references; whole-patient
                         percentile bootstrap (L1.make_patient_draws, B = 5000, seed 123); rule branch mechanical.
  secondary (specification 5) concordance A and anchored AUROCs, other-group arm, mean cost by type, composition,
                         prediction / annotation agreement per slide, sensitivities of the primary.
  fidelity (specification 6)  F1 positive control vs L4, F2 plan identity, F3 mask binding, F4 typing completeness,
                         F5 bootstrap determinism, F6 unit tests, F7 hash readback.

Implementation choices (also in [internal record]):
 1. Link typing, weighted AUROC (sklearn roc_auc_score with sample_weight = t), eligibility, per-reference averaging
    and the bootstrap are the L4 functions (imported read-only by path) or line-for-line mirrors of them extended by a
    `version` filter.  Per-pair AUROCs are stored whenever both classes have >= 1 link; eligibility is applied at
    aggregation time (thresholds 5 / 10 / 20 share one table).
 2. Composition shares (5.4) are of the typed links / typed mass of the version (all links except under `ann`);
    `typed_mass_share` stores the typed share of total mass.  The `lit` unannotated-end share is over link ends
    (each link has a query end and a reference end): count-based (primary) and mass-weighted (each end carries t).
    Composition is summarised over every same-group pair row with a finite value (no link-count eligibility).
 3. Mean cost by type: mass-weighted mean m of a type, pairs with >= 10 links of that type (as L4).
 4. Prediction agreement (5.5): per slide on annotated tiles (binary classes); accuracy, sensitivity (predicted
    tumour | annotated tumour), specificity (predicted non-tumour | annotated non-tumour); summarised as the mean
    over slides with a finite value, whole-patient bootstrap over the slides' patients (slide_offsets patient_id),
    plus median / min / max and the pooled tile-level value.
 5. Rule branch: strict inequalities on the percentile bounds of B_fill (same-group, pooled): lo > 0.5 ->
    `fill_ci_lo_gt_half`; else hi < 0.5 -> `fill_ci_hi_lt_half`; else `fill_ci_spans_half`; the last two share the
    specification's outcome text.
 6. Gate order: bind -> F7 -> F6 -> F3 -> plan pass (F2, F4 inside) -> F1 -> F5 -> estimates.  Any FAIL writes
    checks.json + FAILED.json and stops before results.json.
"""
import argparse
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

CONTRACT_SHA = "b7c243830294682742fa0e322db165d46d61752459abbeb6f8f5fcf12e428a79"
STUDY = "r3_5_localisation_maps_20261017"
TF2 = "r2_1_filter_tf2_20261016/s3/uni2/b100"
INPUTS = {   # role -> (path relative to $PROSPER_OUT, specification sha256)
    "pairs_P2": (f"{STUDY}/s0_pairs/run_001/pairs_P2.csv",
                 "9b90ad629ee9e9b7bf04f43e913be5efa0000e9cc73180215260953ae187ffbe"),
    "plan_manifest": (f"{STUDY}/l2/run_003/plan_manifest.csv",
                      "7650333808829826dde73cc3112bca29015d185945b23e65d5ba8cab1b97d771"),
    "per_pair_L4": (f"{STUDY}/l4/run_001/per_pair_L4.csv.gz",
                    "c8d471fb82b070da70e45f72a93d64f8bb8fe1e1925d4bb7387b53785a15ea83"),
    "results_L4": (f"{STUDY}/l4/run_001/results.json",
                   "23acce30d949c356e766d955e3c42016b5bc38ec01955fb7f9a09ee32d6ee607"),
    "slide_offsets": (f"{TF2}/slide_offsets.csv",
                      "200637f151ac791ef0a38c0ae400a0324ab9d16ea89016cfd4616e3003fdee81"),
    "coordinates": (f"{TF2}/coordinates.npy",
                    "a893603e0acc212277cfaa58a7349fb2fa8820900796bf3c419454f7b7c3e573"),
    "fold0_mask": (f"{TF2}/fold0_mask.npy", "6c5a30df1aa8eb462c0684ccfdd71447f27b9a33ef88562a916cafdd334e0138"),
    "fold1_mask": (f"{TF2}/fold1_mask.npy", "0be876bb43eaf91ef1b1ce9ebfdf0cfa531a34972c16accd67a3d107097d8a73"),
    "fold2_mask": (f"{TF2}/fold2_mask.npy", "739c585449c9fc4ca0cb4fcc99d1bbc3186c2e779c3c72ded894af037f106663"),
    "fold3_mask": (f"{TF2}/fold3_mask.npy", "ca231e0a01ca1b285ede3eeab2f75b7abe06f71dd11de57876f3219cec59f9bc"),
    "fold4_mask": (f"{TF2}/fold4_mask.npy", "32779c5529aeb9437efcb2900a267c1bad0a08ac0bde6e7702a8928fd87cdfc7"),
}
L1_SCRIPT_SHA = "b316497077dc03dd344918aeb8f47b441b85efc2ed45b71da65981d49650f47b"
L4_SCRIPT_SHA = "a9e84b3db98dd2122b563fd4dcf8c3862a3cba352e6dc9169410f35aeb122632"
TOL = 1e-9
TOL_F1 = 1e-12
THR = 10
THR_SENS = (5, 20)
STEP_LIMIT_S = 20 * 60
SAME, OTHER = "same_group", "other_group"
VERSIONS = ("ann", "fill", "lit", "pred")
EXCL, T, N = 0, 1, 2

RULE_TEXT = {
    "fill_ci_lo_gt_half": (
        "the supplement and the letter may say \"the pattern held when all transport links were included, with "
        "unannotated tiles classified by the tumour filter's patient-held-out prediction (AUROC x, 95% CI lo–hi; "
        "n reference patients)\"; a main-text clause of the same content is offered to Haoran as optional."),
    "fill_ci_spans_half": (
        "reported verbatim in the supplement and letter; the main-text sentence on the L4 primary must then carry "
        "\"among annotated tiles\"."),
}
RULE_TEXT["fill_ci_hi_lt_half"] = RULE_TEXT["fill_ci_spans_half"]
RULE_FIXED = [
    "Whatever the outcome, the main-text link-cost estimate stays the annotation-only L4 primary (0.76, 0.70–0.81).",
    "fill, lit and pred are reported in the supplementary table as post hoc, with the statement that the tumour filter "
    "reads the same UNI2 features that set the link cost, so fill and pred are not independent of the cost.",
    "No version is chosen for the main text because it is larger.",
    "B_lit is reported in the supplement whatever its value, with the explanation that unannotated tiles are not "
    "non-tumour; it is never quoted in the main text as a property of the map.",
]
LEAD_PRED = {"fill": "0.70-0.85", "pred": "0.70-0.85", "ann": "0.72-0.80", "lit": "0.50-0.60"}


# ---------------------------------------------------------------- module loading
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L4_PATH = repo_root() / "scripts" / "r3_5_l4_link_cost_concordance.py"
L1_PATH = repo_root() / "scripts" / "r3_5_l1_maps.py"
L4 = _load(L4_PATH, "r3_5_l4_link_cost_concordance_ro")
Stop = L4.Stop
sha256_file, sha256_bytes, binding, jdump, now, close = (L4.sha256_file, L4.sha256_bytes, L4.binding, L4.jdump,
                                                         L4.now, L4.close)
auroc, eligible, weighted_mean, link_types = L4.auroc, L4.eligible, L4.weighted_mean, L4.link_types
TYPE_TT, TYPE_NN, TYPE_MIX = L4.TYPE_TT, L4.TYPE_NN, L4.TYPE_MIX


def out_root() -> Path:
    out = os.environ.get("PROSPER_OUT")
    if not out:
        raise Stop("PROSPER_OUT is not exported; run `export PROSPER_OUT=\"$PWD/prosper_revision_output\"`")
    return Path(out)


def snapshot_script(root_out: Path) -> dict:
    me = Path(__file__).resolve()
    sha = sha256_file(me)
    snap = root_out / "script_snapshots"
    snap.mkdir(parents=True, exist_ok=True)
    existing = sorted(snap.glob(f"r3_5_l5_all_links_v*_{sha[:8]}.py"))
    if existing:
        return {"snapshot": str(existing[0]), "sha256": sha, "created_now": False}
    n = len(list(snap.glob("r3_5_l5_all_links_v*_*.py"))) + 1
    p = snap / f"r3_5_l5_all_links_v{n}_{sha[:8]}.py"
    shutil.copyfile(me, p)
    return {"snapshot": str(p), "sha256": sha, "created_now": True}


# ---------------------------------------------------------------- pure kernels (unit-tested)
def class_versions(c2, pred) -> dict:
    """Per-tile classes (0 excluded, 1 tumour, 2 other) under the four versions of specification 1.

    c2: binary annotation class (0 unannotated, 1 tumour, 2 non-tumour; L1.class_of(labels, True));
    pred: boolean held-out predicted tumour per tile."""
    c2 = np.asarray(c2, np.int8)
    p = np.where(np.asarray(pred, bool), T, N).astype(np.int8)
    return {"ann": c2.copy(),
            "fill": np.where(c2 > 0, c2, p).astype(np.int8),
            "lit": np.where(c2 == T, T, N).astype(np.int8),
            "pred": p}


def held_out_prediction(masks: dict, slide_fold: int, tile_start: int, tile_stop: int) -> np.ndarray:
    """fold{slide_fold}_mask[tile_start:tile_stop] as a boolean array (masks: fold -> array-like)."""
    if slide_fold not in masks:
        raise Stop(f"slide_fold {slide_fold} has no fold mask")
    return np.asarray(masks[slide_fold][tile_start:tile_stop], dtype=bool)


def version_stats(i, j, t, m, cq, cr, cq_ann=None, cr_ann=None) -> dict:
    """Per-plan statistics of one class version.  cq / cr: version classes of query / reference rows;
    cq_ann / cr_ann: binary annotation classes (only used for the lit unannotated-end share)."""
    i, j = np.asarray(i, np.int64), np.asarray(j, np.int64)
    t, m = np.asarray(t, np.float64), np.asarray(m, np.float64)
    a, b = np.asarray(cr)[j], np.asarray(cq)[i]
    ty = link_types(a, b)
    typed = ty > 0
    tt, nn, mix = ty == TYPE_TT, ty == TYPE_NN, ty == TYPE_MIX
    ntt = nn | mix
    n_typed = int(typed.sum())
    mt = float(t[typed].sum())
    rec = {"n_links": int(len(i)), "n_typed": n_typed, "TT": int(tt.sum()), "NN": int(nn.sum()), "MIX": int(mix.sum()),
           "nonTT": int(ntt.sum()), "conc": int((tt | nn).sum()),
           "mass_total": float(t.sum()), "mass_typed": mt,
           "mass_TT": float(t[tt].sum()), "mass_NN": float(t[nn].sum()), "mass_MIX": float(t[mix].sum())}
    rec["typed_mass_share"] = mt / rec["mass_total"] if rec["mass_total"] > 0 else float("nan")
    for k in ("TT", "NN", "MIX"):
        rec[f"share_links_{k}"] = rec[k] / n_typed if n_typed > 0 else float("nan")
        rec[f"share_mass_{k}"] = rec[f"mass_{k}"] / mt if mt > 0 else float("nan")
    rec["B_w"] = auroc(ntt, tt, m, t)
    rec["B_u"] = auroc(ntt, tt, m)
    rec["A_w"] = auroc(mix, tt | nn, m, t)
    rec["A_tanc_w"] = auroc(mix, tt, m, t)
    rec["A_ntanc_w"] = auroc(mix, nn, m, t)
    rec["cost_TT"] = weighted_mean(m, t, tt)
    rec["cost_NN"] = weighted_mean(m, t, nn)
    rec["cost_MIX"] = weighted_mean(m, t, mix)
    if cq_ann is not None and cr_ann is not None:
        a_ann, b_ann = np.asarray(cr_ann)[j], np.asarray(cq_ann)[i]
        oth_r, oth_q = a == N, b == N
        un_r, un_q = oth_r & (a_ann == 0), oth_q & (b_ann == 0)
        n_oth = int(oth_r.sum() + oth_q.sum())
        m_oth = float(t[oth_r].sum() + t[oth_q].sum())
        rec["other_ends"] = n_oth
        rec["other_ends_unann"] = int(un_r.sum() + un_q.sum())
        rec["other_ends_unann_share"] = rec["other_ends_unann"] / n_oth if n_oth > 0 else float("nan")
        rec["other_ends_unann_mass_share"] = (float(t[un_r].sum() + t[un_q].sum()) / m_oth if m_oth > 0
                                              else float("nan"))
    else:
        rec["other_ends"] = rec["other_ends_unann"] = -1
        rec["other_ends_unann_share"] = rec["other_ends_unann_mass_share"] = float("nan")
    return rec


def prediction_agreement(c2, pred) -> dict:
    """Agreement of the held-out prediction with the binary annotation on annotated tiles of one slide."""
    c2, p = np.asarray(c2), np.asarray(pred, bool)
    at, an = c2 == T, c2 == N
    nt, nn_ = int(at.sum()), int(an.sum())
    tp, tn = int((at & p).sum()), int((an & ~p).sum())
    unann = c2 == 0
    return {"n_tiles": int(len(c2)), "n_annotated": nt + nn_, "n_ann_tumour": nt, "n_ann_nontumour": nn_,
            "n_unannotated": int(unann.sum()), "tp": tp, "tn": tn,
            "accuracy": (tp + tn) / (nt + nn_) if nt + nn_ > 0 else float("nan"),
            "sensitivity": tp / nt if nt > 0 else float("nan"),
            "specificity": tn / nn_ if nn_ > 0 else float("nan"),
            "pred_tumour_share_all": float(p.mean()) if len(p) else float("nan"),
            "pred_tumour_share_unannotated": float(p[unann].mean()) if unann.any() else float("nan")}


# metric -> (value column, positive-count column(s), negative-count column(s)); counts summed
METRICS = {
    "B_w": ("B_w", ("nonTT",), ("TT",)),
    "B_u": ("B_u", ("nonTT",), ("TT",)),
    "A_w": ("A_w", ("MIX",), ("TT", "NN")),
    "A_tanc_w": ("A_tanc_w", ("MIX",), ("TT",)),
    "A_ntanc_w": ("A_ntanc_w", ("MIX",), ("NN",)),
    "cost_TT": ("cost_TT", ("TT",), ("TT",)),
    "cost_NN": ("cost_NN", ("NN",), ("NN",)),
    "cost_MIX": ("cost_MIX", ("MIX",), ("MIX",)),
}
COMPOSITION = ("share_links_TT", "share_links_MIX", "share_links_NN", "share_mass_TT", "share_mass_MIX",
               "share_mass_NN", "typed_mass_share", "other_ends_unann_share", "other_ends_unann_mass_share")
for _c in COMPOSITION:
    METRICS[_c] = (_c, ("n_links",), ("n_links",))   # used with thr = 0 (finite value only)


def eligible_mask(pp: pd.DataFrame, metric: str, thr: int) -> np.ndarray:
    col, pos, neg = METRICS[metric]
    npos = pp[list(pos)].sum(axis=1).to_numpy()
    nneg = pp[list(neg)].sum(axis=1).to_numpy()
    return eligible(npos, nneg, thr) & np.isfinite(pp[col].to_numpy(dtype=float))


def per_reference_mean(pp, metric, arm, version, thr=THR, distinct=False) -> pd.DataFrame:
    """As L4.per_reference_mean, restricted to one class version."""
    x = pp[(pp.arm == arm) & (pp.version == version)]
    if distinct:
        x = x.drop_duplicates(["reference", "query"], keep="first")
    x = x[eligible_mask(x, metric, thr)]
    col = METRICS[metric][0]
    g = x.groupby("reference")[col]
    return pd.DataFrame({"value": g.mean(), "n_rows": g.size()}).reset_index()


def estimate(L1, pp, refinfo, metric, arm, version, B, seed, thr=THR, distinct=False, subset=None) -> tuple:
    """As L4.estimate (same bootstrap: L4.boot), restricted to one class version."""
    pr = per_reference_mean(pp, metric, arm, version, thr, distinct).merge(refinfo, on="reference", how="left")
    if subset is not None:
        pr = pr[subset(pr)]
    pr = pr.sort_values("reference").reset_index(drop=True)
    e = L4.boot(L1, pr.value, pr.ref_patient, pr.n_rows, B, seed)
    e.update({"metric": metric, "arm": arm, "version": version, "eligibility_threshold": thr,
              "distinct_plans_only": distinct})
    return e, pr


def paired_difference(L1, pp, refinfo, metric, version, B, seed, thr=THR) -> dict:
    a = per_reference_mean(pp, metric, SAME, version, thr).rename(columns={"value": "same", "n_rows": "n_same"})
    b = per_reference_mean(pp, metric, OTHER, version, thr).rename(columns={"value": "other", "n_rows": "n_other"})
    x = a.merge(b, on="reference", how="inner").merge(refinfo, on="reference", how="left").sort_values("reference")
    e = L4.boot(L1, x.same - x.other, x.ref_patient, x.n_same + x.n_other, B, seed)
    e.update({"metric": f"{metric} same_group - other_group (paired per reference)", "version": version,
              "eligibility_threshold": thr, "n_pairs_note": "eligible pair rows of both arms in the paired references"})
    return e


def rule_branch(lo, hi) -> str:
    if lo is not None and lo > 0.5:
        return "fill_ci_lo_gt_half"
    if hi is not None and hi < 0.5:
        return "fill_ci_hi_lt_half"
    return "fill_ci_spans_half"


def fill_rule_sentence(branch, point, lo, hi, n_patients) -> str:
    text = RULE_TEXT[branch]
    if branch == "fill_ci_lo_gt_half":
        text = (text.replace("AUROC x", f"AUROC {point:.2f}").replace("lo–hi", f"{lo:.2f}–{hi:.2f}")
                .replace("n reference patients", f"{n_patients} reference patients"))
    return text


# ---------------------------------------------------------------- fidelity checks
def check_F6(repo: Path, out_dir: Path) -> dict:
    test = repo / "tests" / "test_r3_5_l5_all_links.py"
    cmd = [sys.executable, "-m", "pytest", str(test.relative_to(repo)), "-q"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    (out_dir / "pytest_output.txt").write_text(f"$ {' '.join(cmd)}\n(cwd {repo}, {now()})\n\n{out}")
    summary = [ln for ln in proc.stdout.strip().splitlines() if "passed" in ln or "failed" in ln or "error" in ln]
    return {"command": " ".join(cmd), "cwd": str(repo), "returncode": proc.returncode, "wall_s": time.time() - t0,
            "summary": summary[-1] if summary else proc.stdout.strip()[-200:], "test_file_sha256": sha256_file(test),
            "pass": proc.returncode == 0}


def check_F1(pp, l4pp, est_ann, l4prim) -> dict:
    x = pp[pp.version == "ann"].set_index("pair_id")
    d = l4pp.set_index("pair_id")
    rec = {"n_pairs_L5_ann": int(len(x)), "n_pairs_L4": int(len(d)), "pair_id_sets_equal": set(x.index) == set(d.index)}
    if not rec["pair_id_sets_equal"]:
        rec["pass"] = False
        return rec
    d = d.loc[x.index]
    a, b = x.A_w.to_numpy(float), d.A_bin_w.to_numpy(float)
    ok = close(a, b, TOL_F1)
    diff = np.abs(a - b)
    rec["A_vs_L4_A_bin_w"] = {"max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else None,
                              "n_mismatch": int((~ok).sum()), "n_nan_L4": int(np.isnan(b).sum()),
                              "n_nan_L5": int(np.isnan(a).sum()), "tolerance": TOL_F1}
    ok_all = bool(ok.all())
    for c in ("TT", "NN", "MIX"):
        n = int((x[c].astype(np.int64).values != d[c].astype(np.int64).values).sum())
        rec[f"n_mismatch_{c}"] = n
        ok_all &= n == 0
    n = int((x.n_typed.astype(np.int64).values != d.ann.astype(np.int64).values).sum())
    rec["n_mismatch_typed_vs_L4_ann"] = n
    ok_all &= n == 0
    est = {}
    for k in ("point", "ci95_lo", "ci95_hi"):
        dd = abs(float(est_ann[k]) - float(l4prim[k]))
        est[k] = {"L5": est_ann[k], "L4": l4prim[k], "abs_diff": dd}
        ok_all &= dd <= TOL_F1
    for k in ("n_references", "n_pairs", "n_patients"):
        est[k] = {"L5": est_ann[k], "L4": l4prim[k], "equal": est_ann[k] == l4prim[k]}
        ok_all &= est_ann[k] == l4prim[k]
    rec["same_group_estimate_vs_L4_primary"] = est
    rec["pass"] = bool(ok_all)
    return rec


def check_F5(L1, pp, refinfo, B, seed) -> dict:
    a, _ = estimate(L1, pp, refinfo, "B_w", SAME, "fill", B, seed)
    b, _ = estimate(L1, pp, refinfo, "B_w", SAME, "fill", B, seed)
    c, _ = estimate(L1, pp, refinfo, "B_w", SAME, "fill", B, seed + 1)
    k = ("ci95_lo", "ci95_hi")
    rec = {"endpoint": "primary (B_w, fill, same_group, pooled)", "seed": seed, "alt_seed": seed + 1,
           "identical_intervals_same_seed": all(a[x] == b[x] for x in k),
           "same_point_alt_seed": a["point"] == c["point"],
           "different_interval_alt_seed": any(a[x] != c[x] for x in k),
           "note": "estimate values are not recorded in this check"}
    rec["pass"] = bool(rec["identical_intervals_same_seed"] and rec["same_point_alt_seed"]
                       and rec["different_interval_alt_seed"])
    return rec


# ---------------------------------------------------------------- slide pass (F3) and plan pass (F2, F4)
def slide_pass(L1, slides, offsets: pd.DataFrame, coords, masks, deadline) -> tuple:
    off = offsets.set_index("slide_id")
    cls, agree, f3 = {}, [], []
    for s in sorted(slides, key=int):
        if time.time() > deadline:
            raise Stop("slide pass exceeded the 20-minute step limit")
        _, keys = L1.load_features(s)
        lab = L1.load_labels(s)
        labels = np.array([lab.get(k, 0) for k in keys], dtype=np.int8)
        c2 = L1.class_of(labels, True)
        rec = {"slide_id": s, "in_offsets": s in off.index, "n_rows": len(keys)}
        if not rec["in_offsets"]:
            f3.append({**rec, "pass": False})
            continue
        o = off.loc[s]
        a, z, fold = int(o.tile_start), int(o.tile_stop), int(o.slide_fold)
        kk = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
        cc = np.asarray(coords[a:z])
        rec.update({"tile_start": a, "tile_stop": z, "slide_fold": fold, "patient_annotated": int(o.patient_annotated),
                    "patient_id": str(o.patient_id), "mask_used": f"fold{fold}_mask.npy",
                    "n_offset_rows": z - a, "rows_equal": (z - a) == len(keys),
                    "coords_equal": bool(cc.shape == kk.shape and np.array_equal(cc, kk)),
                    "fold_valid": fold in masks})
        rec["pass"] = bool(rec["rows_equal"] and rec["coords_equal"] and rec["patient_annotated"] == 1
                           and rec["fold_valid"])
        f3.append(rec)
        if not rec["pass"]:
            continue
        pred = held_out_prediction(masks, fold, a, z)
        cls[s] = {"c2": c2, "pred": pred, **class_versions(c2, pred)}
        agree.append({"slide_id": s, "patient_id": str(o.patient_id), "slide_fold": fold,
                      "mask_used": f"fold{fold}_mask.npy", **prediction_agreement(c2, pred)})
    f3 = pd.DataFrame(f3)
    rec = {"n_slides": int(len(f3)), "n_pass": int(f3["pass"].sum()),
           "n_not_in_offsets": int((~f3.in_offsets).sum()),
           "n_row_count_mismatch": int((f3.get("rows_equal", pd.Series(False, index=f3.index)) == False).sum()),
           "n_coord_mismatch": int((f3.get("coords_equal", pd.Series(False, index=f3.index)) == False).sum()),
           "n_patient_annotated_not_1": int((f3.get("patient_annotated", pd.Series(0, index=f3.index)) != 1).sum()),
           "n_invalid_fold": int((f3.get("fold_valid", pd.Series(False, index=f3.index)) == False).sum()),
           "slides_per_fold_mask": f3.mask_used.value_counts().sort_index().to_dict() if "mask_used" in f3 else {},
           "n_tiles_checked": int(f3.n_rows.sum()), "full_mask_used": False}
    rec["pass"] = bool(rec["n_pass"] == rec["n_slides"] == len(slides))
    return cls, pd.DataFrame(agree), rec


def plan_pass(root: Path, p2: pd.DataFrame, manifest: pd.DataFrame, cls: dict, deadline: float) -> tuple:
    man = manifest.astype({"reference": str, "query": str}).set_index(["reference", "query"]).sha256.to_dict()
    per_plan, f2, f4 = {}, [], []
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
        cq, cr = cls[q], cls[r]
        f2.append({"reference": r, "query": q, "sha256": h, "manifest_sha256": man.get((r, q)),
                   "hash_ok": man.get((r, q)) == h, "shape_ok": shape == (len(cq["c2"]), len(cr["c2"])),
                   "abs_diff_distance": abs(float((t * m).sum()) - dist)})
        if not f2[-1]["shape_ok"]:
            continue
        for v in VERSIONS:
            extra = (cq["c2"], cr["c2"]) if v == "lit" else (None, None)
            rec = version_stats(i, j, t, m, cq[v], cr[v], *extra)
            per_plan[(r, q, v)] = rec
            if v != "ann":
                f4.append({"reference": r, "query": q, "version": v, "untyped_links": rec["n_links"] - rec["n_typed"],
                           "abs_typed_mass_minus_1": abs(rec["mass_typed"] - 1.0),
                           "abs_typed_minus_total_mass": abs(rec["mass_typed"] - rec["mass_total"])})
    f2 = pd.DataFrame(f2)
    rec_f2 = {"n_plans": int(len(f2)), "n_hash_mismatch": int((~f2.hash_ok).sum()),
              "n_shape_mismatch": int((~f2.shape_ok).sum()), "max_abs_diff_distance": float(f2.abs_diff_distance.max()),
              "n_distance_mismatch": int((f2.abs_diff_distance > TOL).sum()), "tolerance": TOL,
              "n_manifest_plans": int(len(manifest))}
    rec_f2["pass"] = bool(rec_f2["n_hash_mismatch"] == 0 and rec_f2["n_shape_mismatch"] == 0
                          and rec_f2["n_distance_mismatch"] == 0)
    f4 = pd.DataFrame(f4)
    rec_f4 = {"n_plan_versions": int(len(f4)), "versions": ["fill", "lit", "pred"], "tolerance": TOL}
    if len(f4):
        rec_f4.update({"n_untyped_links": int(f4.untyped_links.sum()),
                       "max_abs_typed_mass_minus_1": float(f4.abs_typed_mass_minus_1.max()),
                       "max_abs_typed_minus_total_mass": float(f4.abs_typed_minus_total_mass.max()),
                       "n_mass_fail": int((f4.abs_typed_mass_minus_1 > TOL).sum())})
        rec_f4["pass"] = bool(rec_f4["n_untyped_links"] == 0 and rec_f4["n_mass_fail"] == 0
                              and len(f4) == 3 * int(f2.shape_ok.sum()))
    else:
        rec_f4["pass"] = False
    rows = []
    for p in p2.itertuples(index=False):
        for v in VERSIONS:
            s = per_plan.get((p.reference, p.query, v))
            if s is None:
                continue
            rows.append({"pair_id": p.pair_id, "reference": p.reference, "query": p.query, "arm": p.arm,
                         "ref_patient": p.ref_patient, "ref_group": p.ref_group, "half": p.half,
                         "ref_strict": p.ref_strict, "version": v, **s})
    return pd.DataFrame(rows), rec_f2, rec_f4, f2


# ---------------------------------------------------------------- estimates
def agreement_summary(L1, agree: pd.DataFrame, B, seed) -> dict:
    out = {}
    for k in ("accuracy", "sensitivity", "specificity", "pred_tumour_share_unannotated", "pred_tumour_share_all"):
        x = agree[np.isfinite(agree[k].to_numpy(float))].sort_values("slide_id", key=lambda s: s.astype(int))
        v = x[k].to_numpy(float)
        pats, mult = L1.make_patient_draws(list(x.patient_id), B, seed)
        reps = L1.replicate_means(v, list(x.patient_id), pats, mult)
        e = L1.summarise_reps(float(v.mean()) if len(v) else float("nan"), reps, len(v))
        e.pop("n_references")
        e.update({"n_slides": int(len(v)), "n_patients": int(len(pats)), "median": float(np.median(v)),
                  "min": float(v.min()), "max": float(v.max()), "B": B, "seed": seed,
                  "unit": "mean over slides with a finite value; whole-patient bootstrap over the slides' patients"})
        out[k] = e
    s = agree[["n_annotated", "n_ann_tumour", "n_ann_nontumour", "tp", "tn"]].sum()
    out["pooled_tiles"] = {"n_annotated": int(s.n_annotated), "accuracy": float((s.tp + s.tn) / s.n_annotated),
                           "sensitivity": float(s.tp / s.n_ann_tumour), "specificity": float(s.tn / s.n_ann_nontumour)}
    return out


def build_estimates(L1, pp, refinfo, B, seed) -> dict:
    E = lambda metric, arm=SAME, version="fill", **kw: estimate(L1, pp, refinfo, metric, arm, version, B, seed, **kw)[0]
    prim = E("B_w")
    branch = rule_branch(prim["ci95_lo"], prim["ci95_hi"])
    table_B = {arm: {v: E("B_w", arm, v) for v in VERSIONS} for arm in (SAME, OTHER)}
    for v in VERSIONS:
        table_B[SAME][v]["lead_prediction"] = LEAD_PRED[v]
    primary = {
        "definition": ("one-sided mass-weighted link AUROC B (y = 1 non-TT link (MIX or NN), y = 0 TT; score = m, "
                       f"weight = t), class version fill, same-group arm, pooled references, >= {THR} TT and >= {THR} "
                       "non-TT links"),
        "estimate": prim, "lead_prediction": LEAD_PRED["fill"],
        "rule_branch": branch, "rule_outcome": RULE_TEXT[branch],
        "rule_outcome_filled": fill_rule_sentence(branch, prim["point"], prim["ci95_lo"], prim["ci95_hi"],
                                                  prim["n_patients"]),
        "rule_inputs": {"point": prim["point"], "ci95_lo": prim["ci95_lo"], "ci95_hi": prim["ci95_hi"],
                        "rule": ("lo > 0.5 -> fill_ci_lo_gt_half; else hi < 0.5 -> fill_ci_hi_lt_half; else "
                                 "fill_ci_spans_half (the last two share the contract outcome text)")},
        "rule_fixed_whatever_the_outcome": RULE_FIXED,
        "status": "post hoc, descriptive (contract_007 header)",
    }
    s1 = {arm: {"A_concordance_fill": E("A_w", arm, "fill"), "A_concordance_pred": E("A_w", arm, "pred"),
                "tumour_anchored_MIX_vs_TT_fill": E("A_tanc_w", arm, "fill"),
                "other_anchored_MIX_vs_NN_fill": E("A_ntanc_w", arm, "fill")} for arm in (SAME, OTHER)}
    cost = {arm: {ty: E(f"cost_{ty}", arm, "fill") for ty in ("TT", "NN", "MIX")} for arm in (SAME, OTHER)}
    cost["paired_same_minus_other"] = {ty: paired_difference(L1, pp, refinfo, f"cost_{ty}", "fill", B, seed)
                                       for ty in ("TT", "NN")}
    comp = {}
    for v in VERSIONS:
        keys = [c for c in COMPOSITION if v == "lit" or not c.startswith("other_ends")]
        comp[v] = {c: E(c, SAME, v, thr=0) for c in keys}
    halves = {h: E("B_w", subset=lambda d, h=h: d.half == h) for h in ("development", "report")}
    sens = {"unweighted_auroc": E("B_u"), "eligibility_threshold_5": E("B_w", thr=5),
            "eligibility_threshold_20": E("B_w", thr=20), "distinct_plans_only": E("B_w", distinct=True),
            "development_half": halves["development"], "report_half": halves["report"]}
    secondary = {
        "1_concordance_and_anchored_same_group": s1[SAME],
        "2_other_group_arm": {"B": table_B[OTHER], **s1[OTHER]},
        "3_mean_link_cost_by_type_fill": cost,
        "4_composition_same_group": comp,
        "6_sensitivities_of_primary": sens,
    }
    return primary, table_B, secondary


def per_reference_table(pp, refinfo) -> pd.DataFrame:
    parts = []
    for v in VERSIONS:
        out = refinfo.copy().set_index("reference")
        out.insert(0, "version", v)
        x = pp[pp.version == v]
        for arm, tag in ((SAME, "same"), (OTHER, "other")):
            xa = x[x.arm == arm]
            out[f"n_rows_{tag}"] = xa.groupby("reference").size().reindex(out.index).fillna(0).astype(int)
            out[f"n_distinct_plans_{tag}"] = (xa.groupby("reference")["query"].nunique().reindex(out.index)
                                              .fillna(0).astype(int))
            for metric in METRICS:
                if metric.startswith("other_ends") and v != "lit":
                    continue
                thr = 0 if metric in COMPOSITION else THR
                pr = per_reference_mean(pp, metric, arm, v, thr).set_index("reference")
                out[f"{metric}_{tag}"] = pr.value.reindex(out.index)
                out[f"{metric}_{tag}_n_rows"] = pr.n_rows.reindex(out.index).fillna(0).astype(int)
        for thr in THR_SENS:
            pr = per_reference_mean(pp, "B_w", SAME, v, thr).set_index("reference")
            out[f"B_w_same_thr{thr}"] = pr.value.reindex(out.index)
            out[f"B_w_same_thr{thr}_n_rows"] = pr.n_rows.reindex(out.index).fillna(0).astype(int)
        pr = per_reference_mean(pp, "B_w", SAME, v, THR, distinct=True).set_index("reference")
        out["B_w_same_distinct"] = pr.value.reindex(out.index)
        out["B_w_same_distinct_n_rows"] = pr.n_rows.reindex(out.index).fillna(0).astype(int)
        for ty in ("TT", "NN"):
            out[f"cost_{ty}_same_minus_other"] = out[f"cost_{ty}_same"] - out[f"cost_{ty}_other"]
        parts.append(out.reset_index())
    res = pd.concat(parts, ignore_index=True)
    res["_r"] = res.reference.astype(int)
    res["_v"] = res.version.map({v: k for k, v in enumerate(VERSIONS)})
    return res.sort_values(["_v", "_r"]).drop(columns=["_r", "_v"])


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="run directory, e.g. $PROSPER_OUT/r3_5_localisation_maps_20261017/l5/run_001")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args(argv)

    t_wall0, t_cpu0 = time.time(), time.process_time()
    timing = {}
    po = out_root()
    root = po / STUDY
    repo = repo_root()
    d = Path(args.out)
    if d.exists():
        raise Stop(f"{d} already exists; refusing to overwrite")
    d.mkdir(parents=True)
    me = Path(__file__).resolve()
    script_sha = sha256_file(me)
    snap = snapshot_script(po)
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
    L1 = _load(L1_PATH, "r3_5_l1_maps_ro")
    files = [binding(root / "contract_007" / "CONTRACT.md", "contract_007", CONTRACT_SHA),
             binding(L1_PATH, "l1_helpers", L1_SCRIPT_SHA),
             binding(L4_PATH, "l4_script", L4_SCRIPT_SHA)]
    for role, (rel, sha) in INPUTS.items():
        files.append(binding(po / rel, role, sha))
    bad = [f["role"] for f in files if f.get("matches_contract") is False]
    if bad:
        return fail("bind", f"sha256 mismatch vs contract table: {bad}")
    bindings = {"created": now(), "script": str(me), "script_sha256": script_sha, "snapshot": snap,
                "contract_sha256": CONTRACT_SHA, "B": args.B, "seed": args.seed, "files": files,
                "plans": "bound per plan against plan_manifest.csv (F2; per-plan hashes in plan_hashes.csv)",
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

    # ---- F7, F6
    checks["F7_hash_readback"] = L4.check_F7(files)
    if not checks["F7_hash_readback"]["pass"]:
        return fail("F7", "run-time re-hash differs from the contract table")
    checks["F6_unit_tests"] = check_F6(repo, d)
    if not checks["F6_unit_tests"]["pass"]:
        return fail("F6", f"pytest returncode {checks['F6_unit_tests']['returncode']}")

    # ---- F3 (slide pass)
    t0 = time.time()
    P = lambda role: po / INPUTS[role][0]
    p2 = pd.read_csv(P("pairs_P2"), dtype=str)
    slides = sorted(set(p2.reference) | set(p2["query"]), key=int)
    offsets = pd.read_csv(P("slide_offsets"), dtype={"slide_id": str, "patient_id": str})
    coords = np.load(P("coordinates"), mmap_mode="r")
    masks = {k: np.load(P(f"fold{k}_mask"), mmap_mode="r") for k in range(5)}
    cls, agree, f3 = slide_pass(L1, slides, offsets, coords, masks, t0 + STEP_LIMIT_S)
    timing["slide_pass_s"] = time.time() - t0
    checks["F3_mask_binding"] = f3
    if not f3["pass"]:
        return fail("F3", "mask binding mismatch (coordinates / row count / patient_annotated / fold)")

    # ---- plan pass (F2, F4)
    t0 = time.time()
    manifest = pd.read_csv(P("plan_manifest"), dtype={"reference": str, "query": str})
    pp, f2, f4, plan_hashes = plan_pass(root, p2, manifest, cls, t0 + STEP_LIMIT_S)
    timing["plan_pass_s"] = time.time() - t0
    checks["F2_plan_identity"] = f2
    checks["F2_plan_identity"]["n_slides"] = len(cls)
    if not f2["pass"] or len(pp) != len(VERSIONS) * len(p2):
        return fail("F2", f"plan identity failed or pair rows lost ({len(pp)} of {len(VERSIONS) * len(p2)})")
    checks["F4_typing_completeness"] = f4
    if not f4["pass"]:
        return fail("F4", "untyped links or typed mass != 1 under fill / lit / pred")

    # ---- F1
    refinfo = p2[["reference", "ref_patient", "ref_group", "half", "ref_strict"]].drop_duplicates("reference")
    if refinfo.reference.duplicated().any() or refinfo.groupby("ref_patient").reference.nunique().max() != 1:
        return fail("F1", "reference / patient mapping is not one-to-one")
    l4pp = pd.read_csv(P("per_pair_L4"), dtype={"reference": str, "query": str}, float_precision="round_trip")
    l4prim = json.loads(P("results_L4").read_text())["primary"]["estimate"]
    est_ann, _ = estimate(L1, pp, refinfo, "A_w", SAME, "ann", args.B, args.seed)
    checks["F1_positive_control"] = check_F1(pp, l4pp, est_ann, l4prim)
    if not checks["F1_positive_control"]["pass"]:
        return fail("F1", "ann version does not reproduce L4")

    # ---- F5
    checks["F5_bootstrap_determinism"] = check_F5(L1, pp, refinfo, args.B, args.seed)
    if not checks["F5_bootstrap_determinism"]["pass"]:
        return fail("F5", "bootstrap not deterministic / alt seed did not change the interval")
    checks["all_pass"] = True
    jdump(d / "checks.json", checks)

    # ---- estimates
    t0 = time.time()
    primary, table_B, secondary = build_estimates(L1, pp, refinfo, args.B, args.seed)
    secondary["5_prediction_agreement_on_annotated_tiles"] = agreement_summary(L1, agree, args.B, args.seed)
    timing["estimates_s"] = time.time() - t0
    elig = {arm: {v: {thr: {"n_pairs": int(eligible_mask(pp[(pp.arm == arm) & (pp.version == v)], "B_w", thr).sum()),
                            "n_references": int(pp[(pp.arm == arm) & (pp.version == v)][
                                eligible_mask(pp[(pp.arm == arm) & (pp.version == v)], "B_w", thr)].reference.nunique())}
                      for thr in (5, 10, 20)} for v in VERSIONS} for arm in (SAME, OTHER)}

    pp.to_csv(d / "per_pair_L5.csv.gz", index=False)
    per_reference_table(pp, refinfo).to_csv(d / "per_reference_L5.csv", index=False)
    agree.to_csv(d / "per_slide_prediction_agreement.csv", index=False)
    plan_hashes.to_csv(d / "plan_hashes.csv", index=False)
    timing["total_wall_s"] = time.time() - t_wall0
    timing["total_cpu_s"] = time.process_time() - t_cpu0
    results = {"created": now(), "stage": "R3.5-L5", "status": "post hoc, descriptive", "script": str(me),
               "script_sha256": script_sha, "snapshot": snap["snapshot"], "contract_sha256": CONTRACT_SHA,
               "B": args.B, "seed": args.seed,
               "generator": "L1.make_patient_draws (numpy Generator(PCG64(seed))) via L4.boot; L1.replicate_means / "
                            "L1.summarise_reps; 95 % percentile interval",
               "unit": "per-pair statistic -> mean over the reference's eligible pair rows of the arm and version (rows "
                       "as listed in pairs_P2.csv) -> mean over references",
               "versions": {"ann": "annotation; unannotated excluded", "fill": "annotation, else held-out prediction",
                            "lit": "tumour if annotated tumour, else other", "pred": "held-out prediction"},
               "eligibility_counts_B": elig,
               "primary": primary, "table_B": table_B, "secondary": secondary, "timing": timing}
    jdump(d / "results.json", results)
    names = ("bindings.json", "checks.json", "results.json", "per_pair_L5.csv.gz", "per_reference_L5.csv",
             "per_slide_prediction_agreement.csv", "plan_hashes.csv", "pytest_output.txt")
    outputs = {n: sha256_file(d / n) for n in names}
    jdump(d / "COMPLETE.json", {"created": now(), "stage": "R3.5-L5", "script": str(me), "script_sha256": script_sha,
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
