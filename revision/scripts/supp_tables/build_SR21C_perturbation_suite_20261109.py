#!/usr/bin/env python3
"""Workspace Table S-R2.1-C (final S15 on 2026-11-09): shared-stimulus perturbation suite — identity values,
changes per condition x pipeline x metric x k, the pre-specified D contrasts, per-group changes at k = 5,
dose strata, null-stimulus-excluded sensitivity, identity-vs-main-grid agreement and the mMV@5 sheets.

Assembly only. Sources (accepted packages; read-only):
  per-pipeline estimates / paired deltas (all groups, k = 1/3/5/10):
    $PROSPER_OUT/r2_1_p2_eval_20261006/run_001/{estimates,paired_deltas}.csv              (H-Optimus-1 + SAMPLER)
    $PROSPER_OUT/r2_1_comparators_20261008/eval_{uni2,virchow2}/run_001/{estimates,paired_deltas}.csv
  D contrasts, per-group, dose-stratum and null-excluded blocks (5,824 rows) and patterns:
    $PROSPER_OUT/r2_1_p2_eval_20261006/n3_run_002/{d_contrasts,patterns}.csv, counts.json
    (= r2_1_fig4a_20261011/v002/supp_perturbation_tables.csv with a table_block column; asserted)
  dose and null-stimulus records: r2_1_p2_eval_20261006/dose_summary_001/, sensitivity_brightness_001/
  mMV@5: r2_1_fig4a_mmv5_20261021/run_001/{deltas,identity}.csv and the 10-21 sheets supp_tables_20261021/S-R2.1-C_mmv5_*.csv
  main-grid comparison: r2_1_comparators_20261008/n4_{uni2,virchow2}_sampler_main_grid_comparison.json
  figure: figures_20261021/figS_perturbation/source_data.csv (108 rows; asserted equal)
Output: .../B_external_perturbation_finelabel/S-R2.1-C_S15/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _supp_common_20261109 import OUT_ROOT, REPO, Build, ci, f3, sgn  # noqa: E402

P2 = OUT_ROOT / "r2_1_p2_eval_20261006"
CMP = OUT_ROOT / "r2_1_comparators_20261008"
MMV = OUT_ROOT / "r2_1_fig4a_mmv5_20261021/run_001"
V002 = OUT_ROOT / "r2_1_fig4a_20261011/v002"
FIG = OUT_ROOT / "figures_20261021/figS_perturbation"
ST1021 = OUT_ROOT / "supp_tables_20261021"
RUNS = {"hopt": P2 / "run_001", "uni2_sampler": CMP / "eval_uni2/run_001", "virchow2_sampler": CMP / "eval_virchow2/run_001"}
PIPE = {"hopt": "H-Optimus-1 + SAMPLER (fp16 autocast)", "uni2_sampler": "UNI2 + SAMPLER (fp32)", "virchow2_sampler": "Virchow2 + SAMPLER (fp32)"}
COMP = {"uni2_sampler": "UNI2 + SAMPLER", "virchow2_sampler": "Virchow2 + SAMPLER"}
FAMILIES = [("Geometric", [("rotate_90", "Rotation 90\u00b0"), ("rotate_180", "Rotation 180\u00b0"), ("rotate_270", "Rotation 270\u00b0"),
                           ("flip_horizontal", "Horizontal flip"), ("flip_vertical", "Vertical flip"), ("spatial_shift", "Spatial shift")]),
            ("Color and illumination", [("brightness", "Brightness"), ("saturation", "Saturation"), ("contrast", "Contrast")]),
            ("Image quality", [("gaussian_blur", "Blur (encoder input)"), ("gaussian_blur_image", "Blur (source tile)")]),
            ("Stain", [("reinhard", "Stain normalization (query-only Reinhard)")])]
ARM = {a: (fam, lab) for fam, arms in FAMILIES for a, lab in arms}
ARM_ORDER = {a: i for i, a in enumerate(ARM)}
GROUPS = ["ACT/ACC", "DSRCT", "EWS", "LIVER", "MEL", "MPNST", "MRT", "NB", "NF", "NORMAL", "OS", "RB", "RMS", "SS", "WT"]
METRIC = {"hit": "hit", "precision": "precision", "mmv": "majority-vote accuracy (MV)"}
TOL = 1e-12


def excl(lo, hi) -> bool:
    return bool((lo > 0) or (hi < 0))


def mlabel(metric: str, k: int, group: str = "MACRO") -> str:
    if metric == "mmv":
        return "mMV@5 (macro majority-vote accuracy)" if group == "MACRO" else "MV@5"
    return f"{'macro ' if group == 'MACRO' else ''}{metric}@{k}"


def main() -> None:
    b = Build("S-R2.1-C", "S-R2.1-C_S15", "scripts/supp_tables/build_SR21C_perturbation_suite_20261109.py", "### R2.1-T8",
              "R2.1-T8 'New Table S-R2.1-C' list (per arm x pipeline changes and D with CIs at k = 1/3/5/10, per-group rows, dose strata, "
              "null-stimulus-excluded sensitivity, secondary D rows, identity-vs-main-grid agreement, v002 legend disclosures) plus the "
              "2026-10-29 'Retain in S-R2.1-C notes/provenance' list and the 2026-10-21 \u224817:45 addendum (append the mMV@5 deltas and identity "
              "sheets; no D contrast on mMV@5).")

    # ---------------- per-pipeline estimates and paired deltas ----------------
    est, pdl = {}, {}
    for p, d in RUNS.items():
        b.bind(d / "COMPLETE", f"{p} run completion marker")
        est[p] = b.csv(d / "estimates.csv", f"{p} per-condition estimates (identity + 12 arms; all groups; k 1/3/5/10)").assign(pipeline=p)
        pdl[p] = b.csv(d / "paired_deltas.csv", f"{p} paired deltas (arm - identity; all groups; k 1/3/5/10)").assign(pipeline=p)
        e, q = est[p], pdl[p]
        b.check(f"{p}: estimates 13 conditions x 16 groups x 8 metric-k = 1664 rows, all estimable", len(e) == 1664 and (e.point_status == "estimable").all() and (e.ci_status == "estimable").all())
        b.check(f"{p}: paired deltas 12 arms x 16 groups x 8 = 1536 rows, all estimable", len(q) == 1536 and (q.point_status == "estimable").all() and (q.ci_status == "estimable").all())
        mac = q[q.group == "MACRO"]
        b.check(f"{p}: MACRO rows use 2,778 slide pairs / 1,876 patients", (mac.n_pairs == 2778).all() and (mac.n_patients == 1876).all())
        h1 = q[(q.metric == "hit") & (q.k == 1)].set_index(["condition", "group"]).sort_index()
        p1 = q[(q.metric == "precision") & (q.k == 1)].set_index(["condition", "group"]).sort_index()
        b.check(f"{p}: precision@1 equals hit@1 in every row (why k = 1 precision is not a separate D row)",
                np.allclose(h1[["delta", "ci_low", "ci_high"]].values, p1[["delta", "ci_low", "ci_high"]].values, atol=0, rtol=0))
        ident = e[e.condition == "identity"].set_index(["group", "metric", "k"])
        jj = q.join(ident[["estimate"]].rename(columns={"estimate": "_id"}), on=["group", "metric", "k"])
        b.check(f"{p}: paired-delta estimate_identity equals the identity estimate", (jj.estimate_identity - jj._id).abs().max() < TOL)
        b.check(f"{p}: delta = estimate_condition - estimate_identity", ((q.delta - (q.estimate_condition - q.estimate_identity)).abs() < 1e-12).all())
    E = pd.concat(est.values(), ignore_index=True)
    Q = pd.concat(pdl.values(), ignore_index=True)

    # ---------------- mMV@5 ----------------
    md = b.csv(MMV / "deltas.csv", "mMV@5 derivation: paired deltas (hit/precision/mmv at k = 5; 3 pipelines)")
    mi = b.csv(MMV / "identity.csv", "mMV@5 derivation: identity values")
    gates = b.json(MMV / "gates.json", "mMV@5 derivation gates")
    b.bind(MMV / "COMPLETE", "mMV@5 derivation completion marker")
    b.check("mMV@5 derivation gates G1-G4 all pass", gates.get("all_pass") is True)
    for metric in ("hit", "precision"):
        x = md[md.metric == metric].merge(Q[(Q.metric == metric) & (Q.k == 5)], on=["pipeline", "condition", "group"], suffixes=("_m", ""))
        b.check(f"mMV derivation {metric}@5 rows equal the accepted paired deltas (576 rows)", len(x) == 576 and
                max((x[f"{c}_m"] - x[c]).abs().max() for c in ("delta", "ci_low", "ci_high")) < 1e-9)
    mmv = md[md.metric == "mmv"].copy()
    b.check("mMV@5 deltas: 3 x 12 x 16 = 576 rows", len(mmv) == 576 and (mmv.ci_status == "estimable").all())
    old_d = b.csv(ST1021 / "S-R2.1-C_mmv5_deltas.csv", "10-21 Table S-R2.1-C mMV@5 deltas sheet (copied)")
    old_i = b.csv(ST1021 / "S-R2.1-C_mmv5_identity.csv", "10-21 Table S-R2.1-C mMV@5 identity sheet (copied)")
    b.bind(ST1021 / "Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx", "10-21 workbook holding the mMV@5 sheets")
    b.bind(REPO / "scripts/build_supp_tables_r2_1_20261021.py", "10-21 builder of the mMV@5 sheets")
    oo = old_d.merge(mmv, on=["pipeline", "condition", "group"], suffixes=("", "_r"))
    dd = max((oo.delta_perturbed_minus_identity - oo.delta).abs().max(), (oo.ci_low - oo.ci_low_r).abs().max(), (oo.ci_high - oo.ci_high_r).abs().max())
    b.check("10-21 mMV@5 deltas sheet equals the derivation (576 rows; delta and interval, |diff| < 1e-12)", len(old_d) == 576 and len(oo) == 576 and dd < 1e-12, f"max |diff| {dd:.3g}")
    b.check("10-21 mMV@5 identity sheet: 48 rows equal the derivation identity", len(old_i) == 48 and
            (old_i.merge(mi[mi.metric == "mmv"], on=["pipeline", "group"], suffixes=("", "_r")).pipe(lambda z: (z.estimate - z.estimate_r).abs().max()) < 1e-12))

    # ---------------- identity sheet ----------------
    idrows = []
    for p in RUNS:
        e = est[p][est[p].condition == "identity"]
        for r in e.itertuples():
            idrows.append(dict(pipeline=p, group=r.group, metric=r.metric, k=int(r.k), estimate=r.estimate, ci_low=r.ci_low, ci_high=r.ci_high,
                               n_slides=int(r.n_slides), n_patients=int(r.n_patients), value_source=f"{RUNS[p].relative_to(OUT_ROOT)}/estimates.csv"))
        m = mi[(mi.pipeline == p) & (mi.metric == "mmv")]
        for r in m.itertuples():
            idrows.append(dict(pipeline=p, group=r.group, metric="mmv", k=5, estimate=r.estimate, ci_low=r.ci_low, ci_high=r.ci_high,
                               n_slides=int(r.n_slides), n_patients=int(r.n_patients), value_source=f"{MMV.relative_to(OUT_ROOT)}/identity.csv"))
        for metric in ("hit", "precision"):
            a = mi[(mi.pipeline == p) & (mi.metric == metric)].set_index("group").estimate.sort_index()
            c = e[(e.metric == metric) & (e.k == 5)].set_index("group").estimate.sort_index()
            b.check(f"{p}: mMV-derivation identity {metric}@5 equals the run identity (16 rows)", (a - c).abs().max() < 1e-9)
    I = pd.DataFrame(idrows)
    exp_id = {("hopt", "hit"): 0.879, ("uni2_sampler", "hit"): 0.869, ("virchow2_sampler", "hit"): 0.875,
              ("hopt", "mmv"): 0.729, ("uni2_sampler", "mmv"): 0.707, ("virchow2_sampler", "mmv"): 0.723,
              ("hopt", "precision"): 0.666, ("uni2_sampler", "precision"): 0.647, ("virchow2_sampler", "precision"): 0.655}
    for (p, metric), v in exp_id.items():
        x = I[(I.pipeline == p) & (I.metric == metric) & (I.k == 5) & (I.group == "MACRO")].estimate.iloc[0]
        b.check(f"manuscript/legend identity: {PIPE[p]} macro {metric}@5 = {v}", round(float(x), 3) == v, f"{x:.6f}")
    I["_p"] = I.pipeline.map({p: i for i, p in enumerate(RUNS)})
    I["_m"] = I.metric.map({"hit": 0, "precision": 1, "mmv": 2})
    I["_g"] = I.group.map(lambda g: -1 if g == "MACRO" else GROUPS.index(g))
    I = I.sort_values(["_p", "_m", "k", "_g"]).drop(columns=["_p", "_m", "_g"]).reset_index(drop=True)
    Im = I[I.group == "MACRO"]
    id_disp = pd.DataFrame({"Pipeline": Im.pipeline.map(PIPE), "Metric": [mlabel(m, k) for m, k in zip(Im.metric, Im.k)],
                            "Unperturbed (identity) value [95% CI]": [ci(p, lo, hi) for p, lo, hi in zip(Im.estimate, Im.ci_low, Im.ci_high)],
                            "Query slides / patients": [f"{a:,} / {c:,}" for a, c in zip(Im.n_slides, Im.n_patients)]})

    # ---------------- changes (MACRO, all k) ----------------
    ch = Q[Q.group == "MACRO"][["pipeline", "condition", "metric", "k", "delta", "ci_low", "ci_high", "estimate_identity", "estimate_condition", "n_pairs", "n_patients"]].copy()
    ch["value_source"] = ch.pipeline.map(lambda p: f"{RUNS[p].relative_to(OUT_ROOT)}/paired_deltas.csv")
    mm = mmv[mmv.group == "MACRO"][["pipeline", "condition", "metric", "k", "delta", "ci_low", "ci_high", "estimate_identity", "estimate_condition", "n_pairs", "n_patients"]].copy()
    mm["value_source"] = f"{MMV.relative_to(OUT_ROOT)}/deltas.csv"
    ch = pd.concat([ch, mm], ignore_index=True)
    ch["excludes_zero"] = [excl(lo, hi) for lo, hi in zip(ch.ci_low, ch.ci_high)]
    ch["_p"] = ch.pipeline.map({p: i for i, p in enumerate(RUNS)})
    ch["_a"] = ch.condition.map(ARM_ORDER)
    ch["_m"] = ch.metric.map({"hit": 0, "mmv": 1, "precision": 2})
    ch = ch.sort_values(["_m", "k", "_a", "_p"]).drop(columns=["_p", "_a", "_m"]).reset_index(drop=True)
    b.check("changes: 3 pipelines x 12 arms x (hit, precision at k 1/3/5/10 + mMV@5) = 324 MACRO rows", len(ch) == 324)

    # figure equality: the 108 k = 5 rows
    fs = b.csv(FIG / "source_data.csv", "Supplementary Fig. S5 (workspace S-R2.1-F5) source data, 108 rows")
    b.bind(FIG / "PROVENANCE.md", "figure provenance")
    b.bind(REPO / "scripts/figures/figS_perturbation.py", "figure builder (file layout read)")
    b.bind(REPO / "scripts/figures/figS_perturbation_candidates.py", "builder of the adopted forest render (render_forest)")
    k5 = ch[ch.k == 5].copy()
    k5["fig_metric"] = k5.metric.map({"hit": "hit@5", "precision": "precision@5", "mmv": "mMV@5"})
    fsr = fs.rename(columns={c: f"{c}_fig" for c in fs.columns})
    j = k5.merge(fsr, left_on=["fig_metric", "pipeline", "condition"], right_on=["metric_fig", "pipeline_fig", "arm_fig"], validate="one_to_one")
    dmax = max((j[c] - j[f"{c}_fig"]).abs().max() for c in ("ci_low", "ci_high")) if len(j) else 1
    dmax = max(dmax, (j.delta - j.point_fig).abs().max())
    b.check("the 108 k = 5 rows equal figures_20261021/figS_perturbation/source_data.csv (point and interval)", len(fs) == 108 and len(j) == 108 and dmax == 0.0, f"max |diff| {dmax:.3g}")
    b.check("excludes-zero flags equal the figure source (108 rows)", (j.excludes_zero == j.excludes_zero_fig.astype(bool)).all())
    b.check("17 of the 108 figure intervals exclude zero", int(j.excludes_zero.sum()) == 17 and int(fs.excludes_zero.sum()) == 17, str(int(j.excludes_zero.sum())))
    v002 = b.csv(V002 / "fig4a_source_data.csv", "accepted v002 Fig. 4A source (hit@5/precision@5 Δ and D)")
    vv = v002[v002.panel.str.startswith("delta:")].assign(pipeline=lambda z: z.panel.str.split(":").str[1], fig_metric=lambda z: z.endpoint)
    jv = k5.merge(vv, left_on=["fig_metric", "pipeline", "condition"], right_on=["fig_metric", "pipeline", "arm"], suffixes=("", "_v"))
    b.check("v002 hit@5/precision@5 Δ rows (72) equal the paired deltas", len(jv) == 72 and (jv.delta - jv.point).abs().max() < TOL and (jv.ci_low - jv.ci_low_v).abs().max() < TOL)

    def g5(p, a, m):
        r = k5[(k5.pipeline == p) & (k5.condition == a) & (k5.metric == m)].iloc[0]
        return float(r.delta), float(r.ci_low), float(r.ci_high)
    d, lo, hi = g5("virchow2_sampler", "reinhard", "hit")
    b.check("manuscript: Virchow2 query-only Reinhard hit@5 \u22120.017 (\u22120.032 to \u22120.003)", (round(d, 3), round(lo, 3), round(hi, 3)) == (-0.017, -0.032, -0.003), f"{d:.5f} [{lo:.5f}, {hi:.5f}]")
    d, lo, hi = g5("virchow2_sampler", "reinhard", "mmv")
    b.check("manuscript: Virchow2 query-only Reinhard majority-vote accuracy \u22120.013 (\u22120.027 to \u22120.0004)", (round(d, 3), round(lo, 3), round(hi, 4)) == (-0.013, -0.027, -0.0004), f"{d:.5f} [{lo:.5f}, {hi:.5f}]")
    nr = k5[k5.condition != "reinhard"]
    mx = {m: nr[nr.metric == m].delta.abs().max() for m in ("hit", "mmv", "precision")}
    b.check("manuscript: outside stain normalization |Δ| < 0.005 (hit@5), < 0.008 (mMV@5), < 0.007 (precision@5)", mx["hit"] < 0.005 and mx["mmv"] < 0.008 and mx["precision"] < 0.007, str({k: round(v, 6) for k, v in mx.items()}))
    rp = k5[(k5.condition == "reinhard") & (k5.metric == "precision")]
    b.check("manuscript: Reinhard precision@5 decreased by 0.010\u20130.017 in all three pipelines, all intervals excluding zero",
            rp.delta.between(-0.0175, -0.0095).all() and rp.excludes_zero.all() and (rp.delta < 0).all(), str(rp.delta.round(5).tolist()))
    b.check("manuscript: all three k = 5 endpoints changed by at most 0.017 (max |Δ| 0.016984)", k5.delta.abs().max() <= 0.017 and round(k5.delta.abs().max(), 6) == 0.016984, f"{k5.delta.abs().max():.6f}")
    b.check("MS l.103 (edit sheet): mMV@5 changed by at most 0.014 in point estimate", k5[k5.metric == "mmv"].delta.abs().max() <= 0.014)

    ch_disp = pd.DataFrame({"Metric": [mlabel(m, k) for m, k in zip(ch.metric, ch.k)], "Family": ch.condition.map(lambda a: ARM[a][0]),
                            "Perturbation": ch.condition.map(lambda a: ARM[a][1]), "Pipeline": ch.pipeline.map(PIPE),
                            "Unperturbed": [f3(x) for x in ch.estimate_identity], "Perturbed": [f3(x) for x in ch.estimate_condition],
                            "Change (perturbed \u2212 unperturbed) [95% CI]": [ci(p, lo, hi, 4, True) for p, lo, hi in zip(ch.delta, ch.ci_low, ch.ci_high)],
                            "Interval excludes 0": ["yes" if x else "no" for x in ch.excludes_zero]})

    # ---------------- D contrasts ----------------
    dc = b.csv(P2 / "n3_run_002/d_contrasts.csv", "N3 D contrasts and blocks (5,824 rows)")
    b.bind(P2 / "n3_run_002/COMPLETE", "N3 run completion marker")
    b.bind(P2 / "n3_run_002/RECORD.json", "N3 run record")
    sp = b.csv(V002 / "supp_perturbation_tables.csv", "v002 re-export of d_contrasts with table_block (supplementary blocks)")
    spc = sp.drop(columns="table_block")[dc.columns]
    num = [c for c in dc.columns if dc[c].dtype.kind in "fi" and spc[c].dtype.kind in "fi"]
    dnum = max(float((spc[c] - dc[c]).abs().max()) if spc[c].notna().any() else 0.0 for c in num)
    same_na = all((spc[c].isna() == dc[c].isna()).all() for c in dc.columns)
    same_txt = all(spc[c].astype(str).equals(dc[c].astype(str)) for c in dc.columns if c not in num)
    b.check("v002 supp_perturbation_tables.csv = d_contrasts.csv + table_block (5,824 rows; same columns, text fields identical, numbers |diff| < 1e-15)",
            len(sp) == len(dc) == 5824 and set(sp.columns) == set(dc.columns) | {"table_block"} and same_na and same_txt and dnum < 1e-15, f"max numeric |diff| {dnum:.3g}")
    sp = pd.concat([dc, sp[["table_block"]]], axis=1)
    blocks = sp.table_block.value_counts().to_dict()
    b.check("blocks: primary 48, secondary_k 120, per_group 2520, stratum 2688, null_excluded 448", blocks == {"primary": 48, "secondary_k": 120, "per_group": 2520, "stratum": 2688, "null_excluded": 448}, str(blocks))
    sp["metric_k"] = sp.metric + "@" + sp.k.astype(str)
    full = sp[sp.population == "full"]
    for comp in COMP:
        for who, src, col in (("hopt", "hopt", "delta_hopt"), (comp, comp, "delta_c")):
            x = full[full.comparator == comp].merge(pdl[src], on=["condition", "group", "metric", "k"], suffixes=("", "_pd"))
            b.check(f"D rows (full population, {comp}): {col} and interval equal {src} paired deltas ({len(x)} rows)",
                    len(x) == len(full[full.comparator == comp]) and (x[col] - x.delta).abs().max() < TOL
                    and (x[f"{col}_ci_low"] - x.ci_low).abs().max() < TOL and (x[f"{col}_ci_high"] - x.ci_high).abs().max() < TOL)
        x = full[full.comparator == comp].merge(est["hopt"][est["hopt"].condition == "identity"], on=["group", "metric", "k"])
        b.check(f"D rows ({comp}): identity_abs_hopt equals H-Optimus-1 identity estimate", (x.identity_abs_hopt - x.estimate).abs().max() < TOL)
        x = full[full.comparator == comp].merge(est[comp][est[comp].condition == "identity"], on=["group", "metric", "k"])
        b.check(f"D rows ({comp}): identity_abs_c equals {comp} identity estimate", (x.identity_abs_c - x.estimate).abs().max() < TOL)
    b.check("D = delta_hopt - delta_c in every row (5,824)", (sp.D - (sp.delta_hopt - sp.delta_c)).abs().max() < 1e-12)
    b.check("no k = 1 precision rows in the D blocks (precision@1 = hit@1)", sp[(sp.metric == "precision") & (sp.k == 1)].empty)

    pr = sp[sp.table_block == "primary"].copy()
    b.check("primary D: 2 comparators x 12 arms x (hit@5, precision@5), MACRO, full population", len(pr) == 48 and (pr.group == "MACRO").all() and set(pr.k) == {5})
    pr_ex = pr[pr.D_excludes_zero.eq(True)]
    b.check("primary D: 4 of 48 intervals exclude zero, all precision@5 (3 UNI2, 1 Virchow2)", len(pr_ex) == 4 and (pr_ex.metric == "precision").all()
            and pr_ex.comparator.value_counts().to_dict() == {"uni2_sampler": 3, "virchow2_sampler": 1}, pr_ex[["comparator", "condition", "metric", "D"]].to_string())
    b.check("primary D excludes-zero flag equals the interval rule", (pr.D_excludes_zero.eq(True) == [excl(a, c) for a, c in zip(pr.D_ci_low, pr.D_ci_high)]).all())
    ps_ = b.csv(P2 / "n3_run_002/patterns.csv", "N3 pre-specified two-endpoint pattern per comparator x arm")
    cnt = b.json(P2 / "n3_run_002/counts.json", "N3 counts")
    b.check("patterns: 24 of 24 'not_established' (no same-sign exclusion at both endpoints)", len(ps_) == 24 and (ps_.pattern == "not_established").all())
    b.check("counts.json: 48 primary intervals, 4 excluding zero (hit@5 0, precision@5 4)",
            cnt["stage_totals"]["primary_intervals_computed"]["total"] == 48 and cnt["stage_totals"]["primary_intervals_excluding_zero"] == {"hit@5": 0, "precision@5": 4, "total": 4})
    for r in ps_.itertuples():
        for m, col in (("hit", "hit5"), ("precision", "prec5")):
            x = pr[(pr.comparator == r.comparator) & (pr.condition == r.condition) & (pr.metric == m)].iloc[0]
            b.close(f"pattern row {r.comparator}/{r.condition}: {m}@5 D equals primary D", getattr(r, f"{col}_D"), x.D, TOL)
    sk = sp[sp.table_block == "secondary_k"]
    nsk = int(sk.D_excludes_zero.eq(True).sum())
    b.check("secondary D (k 1/3/10, MACRO): 120 rows; 11 exclude zero (scientific review SPOT-C2)", len(sk) == 120 and nsk == 11, str(nsk))
    x = pr[(pr.comparator == "uni2_sampler") & (pr.condition == "rotate_90") & (pr.metric == "precision")].iloc[0]
    b.check("letter/MS text: UNI2 rotate 90\u00b0 precision@5 D = +0.007 [+0.00003, +0.014]", (round(x.D, 3), round(x.D_ci_low, 5), round(x.D_ci_high, 3)) == (0.007, 0.00003, 0.014), f"{x.D} [{x.D_ci_low}, {x.D_ci_high}]")

    def d_disp(df: pd.DataFrame, with_group: bool = False, with_pop: bool = False) -> pd.DataFrame:
        out = {}
        if with_pop:
            out["Population"] = df.population
        out["Comparator"] = df.comparator.map(COMP)
        out["Family"] = df.condition.map(lambda a: ARM[a][0])
        out["Perturbation"] = df.condition.map(lambda a: ARM[a][1])
        if with_group:
            out["Group"] = df.group
        out["Metric"] = [mlabel(m, k, g) for m, k, g in zip(df.metric, df.k, df.group)]
        out["Change, H-Optimus-1 [95% CI]"] = [ci(p, lo, hi, 4, True) for p, lo, hi in zip(df.delta_hopt, df.delta_hopt_ci_low, df.delta_hopt_ci_high)]
        out["Change, comparator [95% CI]"] = [ci(p, lo, hi, 4, True) for p, lo, hi in zip(df.delta_c, df.delta_c_ci_low, df.delta_c_ci_high)]
        out["D = change(H-Optimus-1) \u2212 change(comparator) [95% CI]"] = [ci(p, lo, hi, 4, True, unavailable="interval unavailable") for p, lo, hi in zip(df.D, df.D_ci_low, df.D_ci_high)]
        out["D interval excludes 0"] = ["\u2014" if pd.isna(e) else ("yes" if bool(e) else "no") for e in df.D_excludes_zero]
        out["Slide pairs / patients"] = [f"{int(a):,} / {int(c):,}" for a, c in zip(df.n_pairs, df.n_patients)]
        return pd.DataFrame(out)

    def order(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(_a=df.condition.map(ARM_ORDER), _g=df.group.map(lambda g: -1 if g == "MACRO" else GROUPS.index(g)),
                         _c=df.comparator.map({c: i for i, c in enumerate(COMP)})).sort_values(
            ["population", "metric", "k", "_c", "_a", "_g"]).drop(columns=["_a", "_g", "_c"]).reset_index(drop=True)
    pr, sk = order(pr), order(sk)
    pat = ps_.assign(_a=ps_.condition.map(ARM_ORDER)).sort_values(["comparator", "_a"]).drop(columns="_a")
    pat_disp = pd.DataFrame({"Comparator": pat.comparator.map(COMP), "Perturbation": pat.condition.map(lambda a: ARM[a][1]),
                             "D hit@5 [95% CI]": [ci(p, lo, hi, 4, True) for p, lo, hi in zip(pat.hit5_D, pat.hit5_ci_low, pat.hit5_ci_high)],
                             "D precision@5 [95% CI]": [ci(p, lo, hi, 4, True) for p, lo, hi in zip(pat.prec5_D, pat.prec5_ci_low, pat.prec5_ci_high)],
                             "Pre-specified pattern (same-sign exclusion of 0 at both endpoints)": pat.pattern.map({"not_established": "not established"}),
                             "Query-only stain-mismatch arm": ["yes" if x else "no" for x in pat.reinhard_flag]})

    # ---------------- per-group changes at k = 5 ----------------
    pg = Q[(Q.group != "MACRO") & (Q.k == 5)][["pipeline", "condition", "group", "metric", "k", "delta", "ci_low", "ci_high", "estimate_identity", "estimate_condition", "n_pairs", "n_patients"]].copy()
    pg["value_source"] = pg.pipeline.map(lambda p: f"{RUNS[p].relative_to(OUT_ROOT)}/paired_deltas.csv")
    pm = mmv[mmv.group != "MACRO"][pg.columns[:-1]].copy()
    pm["value_source"] = f"{MMV.relative_to(OUT_ROOT)}/deltas.csv"
    pg = pd.concat([pg, pm], ignore_index=True)
    pg["excludes_zero"] = [excl(lo, hi) for lo, hi in zip(pg.ci_low, pg.ci_high)]
    pg = pg.assign(_p=pg.pipeline.map({p: i for i, p in enumerate(RUNS)}), _a=pg.condition.map(ARM_ORDER), _g=pg.group.map(GROUPS.index),
                   _m=pg.metric.map({"hit": 0, "mmv": 1, "precision": 2})).sort_values(["_m", "_g", "_a", "_p"]).drop(columns=["_p", "_a", "_g", "_m"]).reset_index(drop=True)
    b.check("per-group k = 5 changes: 3 pipelines x 12 arms x 15 groups x 3 metrics = 1,620 rows", len(pg) == 1620)
    pg_disp = pd.DataFrame({"Group": pg.group, "Metric": [mlabel(m, k, g) for m, k, g in zip(pg.metric, pg.k, pg.group)],
                            "Perturbation": pg.condition.map(lambda a: ARM[a][1]), "Pipeline": pg.pipeline.map(PIPE),
                            "Query slides / patients": [f"{int(a):,} / {int(c):,}" for a, c in zip(pg.n_pairs, pg.n_patients)],
                            "Unperturbed": [f3(x) for x in pg.estimate_identity],
                            "Change [95% CI]": [ci(p, lo, hi, 4, True) for p, lo, hi in zip(pg.delta, pg.ci_low, pg.ci_high)],
                            "Interval excludes 0": ["yes" if x else "no" for x in pg.excludes_zero]})
    pgd = order(sp[(sp.table_block == "per_group") & (sp.k == 5)])
    b.check("per-group D at k = 5: 2 comparators x 12 arms x 15 groups x 2 metrics = 720 rows", len(pgd) == 720)
    pg_all = Q[Q.group != "MACRO"].copy()

    # ---------------- dose strata ----------------
    st = sp[sp.table_block == "stratum"].copy()
    st["stratum"] = st.population.str.split(":").str[2]
    sd = b.csv(P2 / "dose_summary_001/strata_deltas.csv", "H-Optimus-1 dose-stratum deltas (accepted dose summary)")
    dist = b.csv(P2 / "dose_summary_001/stimulus_distribution.csv", "realised stimulus distribution and tertile cut points")
    anyc = b.csv(P2 / "dose_summary_001/any_change_by_stratum.csv", "slides whose hit@k or precision@k changed, per stratum")
    b.bind(P2 / "dose_summary_001/SPEC.md", "dose summary specification (strength definitions, tertiles)")
    b.bind(P2 / "dose_summary_001/COMPLETE", "dose summary completion marker")
    x = st.merge(sd, on=["condition", "stratum", "group", "metric", "k"], suffixes=("", "_sd"))
    b.check("stratum block: delta_hopt and interval equal dose_summary_001/strata_deltas.csv (H-Optimus-1; 2,688 rows)",
            len(x) == len(st) == 2688 and (x.delta_hopt - x.delta).abs().max() < TOL and (x.delta_hopt_ci_low - x.ci_low).abs().max() < TOL
            and (x.delta_hopt_ci_high - x.ci_high).abs().max() < TOL)
    rng = sd.drop_duplicates(["condition", "stratum"])[["condition", "stratum", "n_stratum_slides", "strength_min", "strength_max"]]
    b.check("strata are tertiles: n per stratum sums to 2,778 for each arm", rng.groupby("condition").n_stratum_slides.sum().eq(2778).all())
    for r in dist.itertuples():
        nn = rng[rng.condition == r.arm].set_index("stratum").n_stratum_slides.to_dict()
        b.check(f"{r.arm}: stratum sizes equal stimulus_distribution ({r.n_T1}/{r.n_T2}/{r.n_T3})", nn == {"T1": r.n_T1, "T2": r.n_T2, "T3": r.n_T3})
    stm = order(st[st.group == "MACRO"]).merge(rng, on=["condition", "stratum"])
    st_disp = d_disp(stm)
    st_disp.insert(0, "Stratum (strength range)", [f"{s}: {f3(a, 4)}\u2013{f3(c, 4)} ({'px' if cond == 'spatial_shift' else '|factor \u2212 1|'}; {n} slides)"
                                                   for s, a, c, cond, n in zip(stm.stratum, stm.strength_min, stm.strength_max, stm.condition, stm.n_stratum_slides)])
    dist_disp = pd.DataFrame({"Perturbation": dist.arm.map(lambda a: ARM[a][1]), "Published draw": dist.published,
                              "Realised raw parameter min / median / max": [("\u2014" if pd.isna(a) else f"{a:.4f} / {m:.4f} / {c:.4f}") for a, m, c in zip(dist.raw_min, dist.raw_median, dist.raw_max)],
                              "Strength definition": ["|factor \u2212 1|" if a != "spatial_shift" else "shift length sqrt(x\u00b2 + y\u00b2), px" for a in dist.arm],
                              "Strength median / max": [f"{a:.4f} / {c:.4f}" for a, c in zip(dist.strength_median, dist.strength_max)],
                              "Tertile cut points T1|T2, T2|T3": [f"{a:.4f}, {c:.4f}" for a, c in zip(dist.cut_T1_T2, dist.cut_T2_T3)],
                              "Slides T1 / T2 / T3": [f"{a} / {m} / {c}" for a, m, c in zip(dist.n_T1, dist.n_T2, dist.n_T3)],
                              "Near-null slides (threshold)": [f"{n} (< {t:g})" for n, t in zip(dist.n_near_null, dist.near_null_threshold)],
                              "Exact-null slides": dist.n_exact_null.astype(int)})
    b.check("exact-null stimuli: 28 brightness, 4 spatial shift (others 0)", dist.set_index("arm").n_exact_null.to_dict() == {"brightness": 28, "saturation": 0, "contrast": 0, "spatial_shift": 4})
    anyc_disp = anyc.assign(Perturbation=anyc.arm.map(lambda a: ARM[a][1]))[["Perturbation", "stratum", "k", "n_slides", "n_changed", "frac_changed"]].rename(
        columns={"stratum": "Stratum", "n_slides": "Slides", "n_changed": "Slides whose hit@k or precision@k changed", "frac_changed": "Share changed (raw)"})

    # ---------------- null-excluded sensitivity ----------------
    ne = sp[sp.table_block == "null_excluded"].copy()
    sens = b.csv(P2 / "sensitivity_brightness_001/sensitivity_deltas.csv", "H-Optimus-1 null-stimulus-excluded deltas (accepted)")
    b.bind(P2 / "sensitivity_brightness_001/SPEC.md", "null-stimulus sensitivity specification")
    b.bind(P2 / "sensitivity_brightness_001/COMPLETE", "null-stimulus sensitivity completion marker")
    ne["_arm"] = ne.population.map({"brightness_null_excluded": "brightness", "shift_null_excluded": "spatial_shift"})
    b.check("null-excluded block: populations match their arms", (ne._arm == ne.condition).all())
    x = ne.merge(sens, on=["condition", "group", "metric", "k"], suffixes=("", "_s"))
    b.check("null-excluded block: delta_hopt and interval equal sensitivity_deltas.csv (448 rows)", len(x) == len(ne) == 448 and (x.delta_hopt - x.delta).abs().max() < TOL
            and (x.delta_hopt_ci_low - x.ci_low).abs().max() < TOL and (x.delta_hopt_ci_high - x.ci_high).abs().max() < TOL)
    exn = sens.drop_duplicates("condition").set_index("condition").excluded_null_slides.to_dict()
    b.check("null-excluded: 28 brightness and 4 spatial-shift slides removed (2,750 / 2,774 MACRO pairs)", exn == {"brightness": 28, "spatial_shift": 4}
            and set(ne[(ne.group == "MACRO") & (ne.condition == "brightness")].n_pairs) == {2750} and set(ne[(ne.group == "MACRO") & (ne.condition == "spatial_shift")].n_pairs) == {2774})
    ne_disp = d_disp(order(ne[ne.group == "MACRO"].drop(columns="_arm")), with_pop=True)
    ne_disp["Population"] = ne_disp.Population.map({"brightness_null_excluded": "All slides except the 28 exact-null brightness draws",
                                                     "shift_null_excluded": "All slides except the 4 zero spatial shifts"})

    # ---------------- identity vs main grid ----------------
    mg_rows = []
    for comp in COMP:
        jsn = b.json(CMP / f"n4_{comp}_main_grid_comparison.json", f"{comp} identity vs main-grid SAMPLER comparison (descriptive)")
        b.check(f"{comp}: 128 main-grid keys, all estimable, identity point inside the main-grid interval in all 128", jsn["summary"]["rows"] == 128 and jsn["summary"]["both_estimable"] == 128 and jsn["summary"]["identity_inside_main_grid_ci"] == 128)
        for r in jsn["rows"]:
            mg_rows.append(dict(pipeline=comp, group=r["group"], metric=r["endpoint"], k=r["k"], identity=r["identity"]["estimate"],
                                main_grid=r["main_grid"]["point"], delta_identity_minus_main_grid=r["delta_identity_minus_main_grid"],
                                identity_inside_main_grid_ci=r["identity_point_inside_main_grid_ci"]))
        idc = est[comp][est[comp].condition == "identity"].set_index(["group", "metric", "k"]).estimate
        z = pd.DataFrame([m for m in mg_rows if m["pipeline"] == comp])
        b.check(f"{comp}: main-grid comparison identity values equal the run identity estimates (128)",
                np.allclose(z.identity.values, idc.loc[list(zip(z.group, z.metric, z.k))].values, atol=1e-12, rtol=0))
    MG = pd.DataFrame(mg_rows)
    neq = MG.assign(eq=MG.delta_identity_minus_main_grid.abs() < 1e-12).groupby("pipeline").eq.sum().to_dict()
    mg_disp = pd.DataFrame([{"pipeline": pp, "Keys (group x metric x k)": len(z),
                             "Keys with identical values": int((z.delta_identity_minus_main_grid.abs() < 1e-12).sum()),
                             "Largest |identity \u2212 main grid|": f"{z.delta_identity_minus_main_grid.abs().max():.2g}",
                             "Identity inside main-grid 95% CI": int(z.identity_inside_main_grid_ci.sum())} for pp, z in MG.groupby("pipeline", sort=False)])
    mg_disp["pipeline"] = mg_disp.pipeline.map(PIPE)
    mg_disp = pd.concat([pd.DataFrame([{"pipeline": PIPE["hopt"], "Keys (group x metric x k)": 128, "Keys with identical values": 126,
                                        "Largest |identity \u2212 main grid|": "9.0e-04 (EWS precision@5); macro precision@5 6.0e-05",
                                        "Identity inside main-grid 95% CI": "not tabulated in a retained file"}]), mg_disp], ignore_index=True).rename(columns={"pipeline": "Pipeline"})
    b.bind(OUT_ROOT / "r2_1_synthesis_20261011/scientific_review_001/results.json", "scientific review 001 (SPOT-C1: H-Optimus-1 identity vs main grid 126/128)")
    spot = (OUT_ROOT / "r2_1_synthesis_20261011/scientific_review_001/results.json").read_text(encoding="utf-8")
    b.check("SPOT-C1 record states 126/128 keys identical for H-Optimus-1 (cited, not re-tabulated)", "126/128 keys identical" in spot)

    # ---------------- mMV@5 copied sheets ----------------
    mmv_d = old_d.copy()
    mmv_i = old_i.copy()

    # ---------------- caption ----------------
    title = "Shared-stimulus perturbation suite: changes, contrasts and dose strata"
    caption = ("Simulated query perturbations evaluated on one shared stimulus set. For 2,778 COMET query slides from 1,876 patients, each query slide's fixed tile mosaic "
               "was perturbed once per condition and searched against each pipeline's unaltered library (same-patient slides excluded) with three pipelines: "
               "H-Optimus-1, UNI2 and Virchow2 tile encoders, each with SAMPLER aggregation. The table gives the unperturbed (identity) values; the paired change "
               "(perturbed \u2212 unperturbed) in hit@k and precision@k at k = 1, 3, 5 and 10 and in majority-vote accuracy at k = 5; the pre-specified pipeline contrasts "
               "D = \u0394(H-Optimus-1) \u2212 \u0394(comparator) for hit@5 and precision@5 (primary) and at k = 1, 3 and 10 (secondary); per-group results at k = 5; results "
               "stratified by realised stimulus strength; and sensitivities excluding effectively null stimuli.")
    foot = [
        "Twelve perturbations (rotation 90\u00b0, 180\u00b0 and 270\u00b0; horizontal and vertical flip; spatial shift; brightness; saturation; contrast; blur of the normalized 224-pixel encoder input, 3 \u00d7 3 kernel, \u03c3 0.5; blur of the 512-pixel source tile before preprocessing, 7 \u00d7 7 kernel, \u03c3 8/7, reflect-101 padding; query-only Reinhard stain normalization) plus the unperturbed identity condition, each realised once per slide from a common perturbation manifest (PCG64 seed 1234). Neither blur simulates optical defocus.",
        "hit@k: at least one same-group other-patient slide among the k nearest; precision@k: share of the k nearest that are same-group (precision@1 equals hit@1 and is not repeated in the D sheets); majority-vote accuracy (MV@5): the plurality group of the five nearest slides equals the query's group, ties resolved toward the closest-ranked neighbour; it was derived later from the same ranked lists and is not a pre-specified primary endpoint, and no D contrast is defined on it. Macro values weight the 15 groups equally.",
        "Intervals: 95% paired whole-patient percentile bootstrap intervals (5,000 draws over 1,876 patients, identical draws for every pipeline and condition), unadjusted for multiplicity. Changes are printed to four decimals; 0.0000 is an exact zero (for an interval bound, a closed-interval bootstrap bound, so the interval covers zero), whereas +0.0000 or \u22120.0000 lies within 0.00005 of zero and carries the sign of the unrounded value in the raw sheets. An interval covering zero means a difference is not established, not that there is none.",
        "D > 0: the H-Optimus-1 pipeline's change was more favourable than the comparator's. A pattern required same-sign exclusion of zero at both hit@5 and precision@5 (pre-specified); none of the 24 condition \u00d7 comparator pairs met it. Of the 48 primary D intervals, 4 excluded zero, all for precision@5 (about 1.2 per endpoint expected by chance among correlated intervals): UNI2 under rotation 90\u00b0, rotation 180\u00b0 and vertical flip, arising from UNI2's own precision@5 decrease, and Virchow2 under saturation. Of the 120 secondary D intervals, 11 excluded zero.",
        "D compares pipelines, not isolated encoders: H-Optimus-1 ran with float16 autocast and the comparators in float32, and the software stacks differ. Comparator feature content was verified per slide against the stored banks and by the independent output review, not by independent re-extraction.",
        "Reinhard normalization (prospective pinned torchstain implementation; the historical implementation could not be recovered) was applied to queries only, creating a query\u2013library stain mismatch; three tiles were excluded by its colour-coverage rule without losing a slide. The query-only interpretation was decided after the H-Optimus-1 results and before the comparator results were seen.",
        "Dose strata: for brightness, saturation and contrast the strength is |factor \u2212 1|, for spatial shift the shift length in pixels; strata are tertiles of the realised strength over the 2,778 slides (sheet 'dose_distribution'). Strengths were drawn independently of slide content, so strata are near-randomised but smaller; descriptive.",
        "Null stimuli: 28 brightness draws (factor within (1, 1 + 1/254), no pixel change after uint8 truncation) and 4 zero spatial shifts are included in the main rows; sheet 'null_excluded' repeats both arms without them (2,750 and 2,774 slides).",
        "Identity rows are the perturbation suite's own unperturbed condition, not the main-grid rows of Supplementary Table S3: H-Optimus-1 identity SAMPLER features are non-cast whereas the main-grid features are float16-cast (126 of 128 group \u00d7 metric \u00d7 k values identical); for UNI2 and Virchow2 see sheet 'main_grid_agreement'.",
        "A software simulation, not a rescan or multi-scanner study; UNI2 with Optimal Transport and tumour filtering were not evaluated; CONCH + SAMPLER and CONCH + TITAN were not re-run (pipeline-specific 1,024-pixel tiling cannot share the pixel-level stimuli) and remain in the separate exploratory analysis (Supplementary Table S17 and the reproduced submitted Fig. 4). The k = 5 macro rows of sheet 'changes' are the source data of Supplementary Fig. S5.",
    ]
    sources = ["$PROSPER_OUT/r2_1_p2_eval_20261006/run_001 (H-Optimus-1), r2_1_comparators_20261008/eval_{uni2,virchow2}/run_001: estimates.csv, paired_deltas.csv",
               "$PROSPER_OUT/r2_1_p2_eval_20261006/n3_run_002: d_contrasts.csv (= r2_1_fig4a_20261011/v002/supp_perturbation_tables.csv), patterns.csv, counts.json",
               "$PROSPER_OUT/r2_1_p2_eval_20261006/dose_summary_001, sensitivity_brightness_001; r2_1_fig4a_mmv5_20261021/run_001 (mMV@5; supp_tables_20261021 sheets)",
               "$PROSPER_OUT/r2_1_comparators_20261008/n4_*_sampler_main_grid_comparison.json; r2_1_synthesis_20261011/scientific_review_001 (SPOT-C1)",
               "figure: $PROSPER_OUT/figures_20261021/figS_perturbation/source_data.csv (asserted equal)"]
    cap = b.caption_frame(title, caption, foot, sources, "v1 full workbook (2026-11-09); supersedes the partial 10-21 mMV@5-only sheets (copied unchanged)")

    def rawify(df: pd.DataFrame, src: str) -> pd.DataFrame:
        return df.assign(value_source=src) if "value_source" not in df.columns else df
    dsrc = "r2_1_p2_eval_20261006/n3_run_002/d_contrasts.csv (via v002 supp_perturbation_tables.csv table_block)"
    sheets = [
        ("identity", id_disp), ("identity_raw", I),
        ("changes", ch_disp), ("changes_raw", ch),
        ("D_primary", d_disp(pr)), ("D_patterns", pat_disp), ("D_secondary_k", d_disp(sk)),
        ("D_raw", rawify(pd.concat([pr, sk]).drop(columns="metric_k"), dsrc)),
        ("per_group_k5", pg_disp), ("per_group_k5_raw", pg),
        ("per_group_k5_D", d_disp(pgd, with_group=True)),
        ("dose_strata", st_disp), ("dose_distribution", dist_disp), ("dose_any_change", anyc_disp),
        ("null_excluded", ne_disp),
        ("main_grid_agreement", mg_disp), ("main_grid_agreement_raw", MG),
        ("mMV5_deltas", mmv_d), ("mMV5_identity", mmv_i),
    ]
    csv_only = [("per_group_all_k_raw", rawify(pg_all, "per-pipeline paired_deltas.csv")),
                ("per_group_D_all_k_raw", rawify(order(sp[sp.table_block == "per_group"]).drop(columns="metric_k"), dsrc)),
                ("dose_strata_raw", rawify(order(st).drop(columns="metric_k"), dsrc)),
                ("null_excluded_raw", rawify(order(ne.drop(columns="_arm")).drop(columns="metric_k"), dsrc))]
    b.write_workbook("Table_S-R2.1-C_perturbation_suite", cap, sheets)
    for name, df in csv_only:
        p = b.dir / f"Table_S-R2.1-C_perturbation_suite__{name}.csv"
        df.to_csv(p, index=False, lineterminator="\n")
        b.outputs.append(p)

    b.open_items += [
        "H-Optimus-1 identity vs main-grid agreement (126/128) is cited from scientific review 001 (SPOT-C1); no retained per-key comparison file for H-Optimus-1 was found, so that row is not tabulated key by key (UNI2/Virchow2 rows are from the retained n4 JSONs).",
        "Per-group rows for groups with few slides (e.g., NF, OS) carry wide intervals; zero-support bootstrap replicates are recorded in the raw columns (zero_support_replicates, ci_status) and some stratum/per-group D intervals are 'unavailable'.",
        "Caption is a draft mirror assembled from R2.1-T8 and the S-R2.1-F5 legend; the provisional numbering in the edit sheet (S34 / S18) is superseded by S15. The 'Supplementary Table S3' cross-reference for main-grid rows follows supp_table_numbering.json (workspace S-R2.1-E -> S3; submitted S3 removed 2026-11-04).",
        "The adopted Supplementary Fig. S5 render was produced by scripts/figures/figS_perturbation_candidates.py render_forest (PROVENANCE.md), not figS_perturbation.py; its source_data.csv layout (candidate, metric, pipeline, arm, point, ci_low, ci_high, excludes_zero, baseline_band_halfwidth) was used for the equality check.",
    ]
    b.finish(title, [
        "Identity (macro, k = 5): hit@5 0.879 / 0.869 / 0.875, mMV@5 0.729 / 0.707 / 0.723, precision@5 0.666 / 0.647 / 0.655 (H-Optimus-1 / UNI2 / Virchow2) \u2014 match.",
        "The 108 k = 5 macro change rows equal the Supplementary Fig. S5 source_data.csv exactly (max |diff| 0); 17 of 108 exclude zero.",
        "Virchow2 query-only Reinhard: hit@5 \u22120.017 (\u22120.032 to \u22120.003); majority-vote accuracy \u22120.013 (\u22120.027 to \u22120.0004) \u2014 match.",
        "D: 48 primary (4 exclude zero, all precision@5; 24/24 patterns not established), 120 secondary (11 exclude zero); every D-block Δ equals the per-pipeline paired deltas.",
        "Dose strata (2,688 rows) and null-excluded (448 rows) H-Optimus-1 values equal the accepted dose / sensitivity files; mMV@5 sheets copied unchanged from supp_tables_20261021.",
    ], cap, [("Neq (identity vs main grid, comparators)", [f"- {PIPE[k]}: {v}/128 keys identical" for k, v in neq.items()])])


if __name__ == "__main__":
    main()
