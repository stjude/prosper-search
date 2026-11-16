#!/usr/bin/env python3
"""R2.1-TF2 writing release 001: format Tables S-R2.1-B / S-R2.1-D, the Table S16
amendment and the Fig. 3 panel data from the accepted S2/S6 outputs.

Formatting only: values are selected and rounded, never recomputed (the per-fold
min/max of the classifier table is a selection over the fold rows of
tile_metrics.csv; the class/direction counts in prose_counts.csv are counts of rows).
Tables: differences and proportions to 4 decimals (design decision 2026-10-17).
Prose helpers (prose_counts.csv): 3 decimals, a value whose magnitude rounds to
0.000 is written "<0.001" with its sign (design decision 2026-10-17).

Run:  /HOME/miniconda3/bin/python3 -I make_tables.py
Writes, beside this script: tables_S_R2_1_B_D.md and the CSVs listed in OUTPUTS.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # $PROSPER_OUT/r2_1_filter_tf2_20261016
S6 = ROOT / "s6"
PRIMARY = S6 / "evaluation_primary_001"
S6C_P = S6 / "evaluation_s6c_primary_001"
S6C_S = S6 / "evaluation_s6c_secondary_001"
CLS = ROOT / "s2" / "run_002" / "classifier_report_001"
AUROC = ROOT / "s2" / "run_002" / "lead_auroc_recount_001" / "auroc_recount.csv"  # 229-slide heads
LINEAGE_RER = [
    ("r2_1_filter_rer_20261012/run_001", ROOT.parent / "r2_1_filter_rer_20261012" / "run_001" / "contrasts.csv"),
    ("r2_1_route_b_uni2_20261013/rer_run_002", ROOT.parent / "r2_1_route_b_uni2_20261013" / "rer_run_002" / "contrasts.csv"),
]

INPUTS = [
    PRIMARY / "contrasts.csv",
    PRIMARY / "wording_classes.csv",
    PRIMARY / "rer_contrasts.csv",
    PRIMARY / "coverage.csv",
    PRIMARY / "strata_contrasts.csv",
    PRIMARY / "tie_sensitivity.csv",
    PRIMARY / "claim_sentences.txt",
    S6C_P / "contrasts.csv",
    S6C_P / "wording_classes.csv",
    S6C_S / "contrasts.csv",
    S6C_S / "wording_classes.csv",
    CLS / "tile_metrics.csv",
    CLS / "per_label_pass_rates.csv",
    CLS / "s2b_full_noqc_vs_full.csv",
    AUROC,
] + [p for _, p in LINEAGE_RER]

ENC = ["uni2", "virchow2", "h_optimus_1", "titan"]
ENC_LABEL = {"uni2": "UNI2", "virchow2": "Virchow2", "h_optimus_1": "H-Optimus-1", "titan": "TITAN",
             "conch_v15": "CONCH v1.5 (TITAN arm)"}
METH = {"ot_cosine": "Optimal Transport", "yottixel": "Yottixel", "titan": "TITAN slide embedding (L1)"}
ARM = {"query_only": "query-only", "both_filtered": "both-sided"}
LABEL = {1: "Tumor", 2: "Immune cells", 3: "Necrosis", 4: "Non-tumor", 5: "Artifacts"}
MINUS = "−"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def d3(x) -> str:
    """Table format: signed difference, 4 decimals, Unicode minus."""
    if pd.isna(x):
        return "—"
    s = f"{x:+.4f}"
    return s.replace("-", MINUS)


def dprose(x) -> str:
    """Prose format: signed, 3 decimals; magnitude below 0.0005 written <0.001 with its sign."""
    if pd.isna(x):
        return "—"
    if abs(x) < 0.0005:
        return ("−" if x < 0 else "+") + "<0.001"
    return f"{x:+.3f}".replace("-", MINUS)


def ciprose(lo, hi) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return "not estimable"
    return f"[{dprose(lo)}, {dprose(hi)}]"


def p3(x) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.4f}".replace("-", MINUS)


def ci(lo, hi) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return "not estimable"
    return f"[{d3(lo)}, {d3(hi)}]"


def order_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_e"] = df["encoder"].map({e: i for i, e in enumerate(ENC)})
    df["_m"] = df["method"].map({"ot_cosine": 0, "yottixel": 1, "titan": 2})
    df["_a"] = df["arm"].map({"query_only": 0, "both_filtered": 1})
    df["_b"] = df["budget"].map({100: 0, 5: 1})
    sort = [c for c in ["_b", "_e", "_m", "_a", "k"] if c in df.columns]
    return df.sort_values(sort).drop(columns=["_e", "_m", "_a", "_b"])


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- family a/b/c
def family(c: pd.DataFrame, fam: str) -> pd.DataFrame:
    m = c[(c["family"] == fam) & (c["level"] == "macro")].copy()
    m["budget"] = m["budget"].astype(int)
    return order_frame(m)


def wide_endpoints(m: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in m.groupby(keys, sort=False):
        rec = dict(zip(keys, key))
        for ep in ["hit", "precision", "hit_yield"]:
            e = g[g["endpoint"] == ep]
            if len(e) == 0 and ep == "precision" and rec["k"] == 1:
                # precision@1 equals hit@1; the evaluator writes no separate row at k = 1
                for f in ["point", "ci_low", "ci_high", "n_queries", "population"]:
                    rec[f"{ep}_{f}"] = pd.NA
                continue
            if len(e) != 1:
                raise SystemExit(f"expected one {ep} row for {key}, got {len(e)}")
            e = e.iloc[0]
            rec[f"{ep}_point"] = e["point"]
            rec[f"{ep}_ci_low"] = e["ci_low"]
            rec[f"{ep}_ci_high"] = e["ci_high"]
            rec[f"{ep}_n_queries"] = int(e["n_queries"])
            rec[f"{ep}_population"] = e["population"]
        rec["n_patients"] = int(g["n_patients"].iloc[0])
        rows.append(rec)
    return order_frame(pd.DataFrame(rows))


def main() -> None:
    manifest = pd.DataFrame({"input": [str(p.relative_to(ROOT.parent)) for p in INPUTS],
                             "sha256": [sha256(p) for p in INPUTS]})
    manifest.to_csv(HERE / "inputs_sha256.csv", index=False)

    c = pd.read_csv(PRIMARY / "contrasts.csv")
    keys = ["encoder", "method", "arm", "budget", "k"]

    a = wide_endpoints(family(c, "a_filtered_minus_all_tissue"), keys)
    a.insert(0, "analysis", "primary")
    a.to_csv(HERE / "S_R2_1_B_part1_filtered_minus_all_tissue.csv", index=False)

    b = wide_endpoints(family(c, "b_budget5_minus_budget100"), keys)
    b.to_csv(HERE / "S_R2_1_B_part2_budget5_minus_budget100.csv", index=False)

    cc = wide_endpoints(family(c, "c_query_only_minus_both_sided"), keys)
    cc.to_csv(HERE / "S_R2_1_B_part3_query_only_minus_both_sided.csv", index=False)

    # sensitivity analyses at k = 5 (S6-c primary/secondary; strata)
    sens = []
    for name, d in [("s6c_primary_168_removed", S6C_P), ("s6c_secondary_355_removed", S6C_S)]:
        cs = pd.read_csv(d / "contrasts.csv")
        w = wide_endpoints(family(cs, "a_filtered_minus_all_tissue"), keys)
        w = w[w["k"] == 5]
        w.insert(0, "analysis", name)
        sens.append(w)
    st = pd.read_csv(PRIMARY / "strata_contrasts.csv")
    for stratum in ["deployment_unannotated", "annotated_patient"]:
        s = st[(st["stratum"] == stratum)]
        w = wide_endpoints(family(s, "a_filtered_minus_all_tissue"), keys)
        w = w[w["k"] == 5]
        w.insert(0, "analysis", f"stratum_{stratum}")
        sens.append(w)
    sens = pd.concat(sens, ignore_index=True)
    sens.to_csv(HERE / "S_R2_1_B_part4_sensitivity_k5.csv", index=False)

    # coverage
    cov = pd.read_csv(PRIMARY / "coverage.csv")
    cov = cov[(cov["level"] == "pooled_slides_NOT_macro") & (cov["stratum"] == "all")].copy()
    cov["budget"] = cov["budget"].fillna(0).astype(int)
    cov = cov[["encoder", "arm", "method", "budget", "intended", "returned", "non_returns",
               "non_return_slides", "coverage"]]
    cov.to_csv(HERE / "S_R2_1_B_part5_coverage.csv", index=False)

    # Yottixel tie sensitivity, macro k = 5, family a, conditional
    t = pd.read_csv(PRIMARY / "tie_sensitivity.csv")
    t = t[(t["row_kind"] == "contrast") & (t["family"] == "a_filtered_minus_all_tissue") &
          (t["level"] == "macro") & (t["k"] == 5) & (t["population"] == "both_returned") &
          (t["method"] == "yottixel")].copy()
    t["budget"] = t["budget"].astype(int)
    tw = t.pivot_table(index=["encoder", "method", "arm", "budget", "endpoint"], columns="tie_rule",
                       values=["point", "ci_low", "ci_high"], aggfunc="first")
    tw.columns = [f"{v}_{r}" for v, r in tw.columns]
    tw = order_frame(tw.reset_index())
    tw.to_csv(HERE / "S_R2_1_B_part6_yottixel_tie_sensitivity_k5.csv", index=False)

    # ------------------------------------------------------------ RER (S-R2.1-D)
    r = pd.read_csv(PRIMARY / "rer_contrasts.csv")
    rm = r[r["class"] == "MACRO"].copy()
    rm = order_frame(rm)
    rm = rm[["encoder", "method", "arm", "budget", "handling", "cell_role", "f1_all", "f1_filt",
             "delta_f1", "delta_ci_low", "delta_ci_high", "rer", "rer_ci_low", "rer_ci_high",
             "rer_ci_status", "n_undefined_reps", "support", "n_queries", "n_nonreturned_filt"]]
    rm.to_csv(HERE / "S_R2_1_D_macro_cells.csv", index=False)
    head = r[(r["encoder"] == "uni2") & (r["method"] == "ot_cosine") & (r["arm"] == "both_filtered") &
             (r["budget"] == 100) & (r["handling"] == "miss")].copy()
    head = head[["class", "cell_role", "f1_all", "f1_filt", "delta_f1", "delta_ci_low", "delta_ci_high",
                 "rer", "rer_ci_low", "rer_ci_high", "rer_ci_status", "n_undefined_reps",
                 "n_zero_support_reps", "support"]]
    head["_o"] = (head["class"] == "MACRO").astype(int)
    head = head.sort_values(["_o", "class"]).drop(columns="_o")
    head.to_csv(HERE / "S_R2_1_D_headline_cell_groups.csv", index=False)

    # ------------------------------------------------- classifier (S16, Fig. 3A)
    tm = pd.read_csv(CLS / "tile_metrics.csv")
    s2b = pd.read_csv(CLS / "s2b_full_noqc_vs_full.csv")
    import json
    crow = []
    for (enc, bud), g in tm.groupby(["encoder", "budget_percent"], sort=False):
        pooled = g[g["scope"] == "pooled_cross_fitted"].iloc[0]
        folds = g[g["scope"].str.startswith("fold")]
        ins = s2b[(s2b["encoder"] == enc) & (s2b["budget_percent"] == bud) &
                  (s2b["scope"] == "all_pool_tiles")].iloc[0]
        fi = json.loads(ins["full_in_sample"])
        crow.append({
            "encoder": enc, "budget_percent": int(bud), "n_tiles": int(pooled["n"]),
            "sens_pooled_cross_fitted": pooled["sensitivity"],
            "sens_fold_min": folds["sensitivity"].min(), "sens_fold_max": folds["sensitivity"].max(),
            "spec_pooled_cross_fitted": pooled["specificity"],
            "spec_fold_min": folds["specificity"].min(), "spec_fold_max": folds["specificity"].max(),
            "full_head_on_pool_tiles_sens": fi["sensitivity"],
            "full_head_on_pool_tiles_spec": fi["specificity"],
            "full_head_on_pool_tiles_note": ("in-sample: its training tiles" if int(bud) == 100 else
                                             "within-slide: pool tiles incl. its 5% training tiles"),
        })
    clsdf = pd.DataFrame(crow)
    clsdf["_e"] = clsdf["encoder"].map({"uni2": 0, "virchow2": 1, "h_optimus_1": 2, "conch_v15": 3})
    clsdf["_b"] = clsdf["budget_percent"].map({100: 0, 5: 1})
    clsdf = clsdf.sort_values(["_b", "_e"]).drop(columns=["_e", "_b"])
    clsdf.to_csv(HERE / "S16_amendment_tf2_cross_fitted.csv", index=False)
    pl = pd.read_csv(CLS / "per_label_pass_rates.csv")
    pl.to_csv(HERE / "S16_amendment_tf2_per_label.csv", index=False)
    au = pd.read_csv(AUROC)
    au.to_csv(HERE / "S16_auroc_recount_229pool.csv", index=False)
    au_cf = au[au["scope"] == "cross_fitted_patient_held_out"].set_index(["encoder", "budget"])
    au_in = au[au["scope"] == "full_head_in_sample(pool)"].set_index(["encoder", "budget"])

    # Markdown fragment for the Table S16 amendment
    frag = ["<!-- generated by make_tables.py; do not edit by hand -->", "",
            "**Sheet 2 (new): TF-2 tumor heads, cross-fitted over patients (each labeled tile scored once by the fold "
            "head that excluded its patient).**", ""]
    rows = []
    for _, x in clsdf.iterrows():
        rows.append({"Encoder (tiles)": ENC_LABEL[x.encoder] + (" (1,024-px patches)" if x.encoder == "conch_v15" else " (512-px tiles)"),
                     "Label budget": f"{x.budget_percent}%" + (" (primary)" if x.budget_percent == 100 else " (sensitivity)"),
                     "Labeled tiles": f"{x.n_tiles:,}",
                     "Tumor sensitivity, pooled [fold range]": f"{p3(x.sens_pooled_cross_fitted)} [{p3(x.sens_fold_min)}–{p3(x.sens_fold_max)}]",
                     "Tumor specificity, pooled [fold range]": f"{p3(x.spec_pooled_cross_fitted)} [{p3(x.spec_fold_min)}–{p3(x.spec_fold_max)}]",
                     "Tumor-vs-rest AUROC, cross-fitted": p3(au_cf.loc[(x.encoder, x.budget_percent), "auroc_tumour_vs_rest"]),
                     "Tumor-vs-rest AUROC, all-slide head in-sample": p3(au_in.loc[(x.encoder, x.budget_percent), "auroc_tumour_vs_rest"]),
                     "All-slide head on pool tiles: sensitivity / specificity": f"{p3(x.full_head_on_pool_tiles_sens)} / {p3(x.full_head_on_pool_tiles_spec)} ({x.full_head_on_pool_tiles_note})"})
    frag.append(md_table(pd.DataFrame(rows)))
    frag += ["", "**Sheet 2b (new): one-versus-rest AUROC by annotated class, cross-fitted over patients (point "
             "estimates; no bootstrap interval was computed for these heads; Immune cells and Artifacts are marked "
             "interval not estimable as in Sheet 1). Source: `s2/run_002/lead_auroc_recount_001/auroc_recount.csv`.**", ""]
    rows = []
    for _, x in clsdf.iterrows():
        r_ = au_cf.loc[(x.encoder, x.budget_percent)]
        rows.append({"Encoder": ENC_LABEL[x.encoder], "Budget": f"{x.budget_percent}%",
                     "Tumor": p3(r_["auroc_ovr_Tumor"]), "Necrosis": p3(r_["auroc_ovr_Necrosis"]),
                     "Non-tumor": p3(r_["auroc_ovr_Non-tumor"]),
                     "Immune cells": p3(r_["auroc_ovr_Immune"]) + "; interval not estimable",
                     "Artifacts": p3(r_["auroc_ovr_Artifacts"]) + "; interval not estimable",
                     "Macro (5 classes)": p3(r_["macro_ovr"]),
                     "Macro, all-slide head in-sample": p3(au_in.loc[(x.encoder, x.budget_percent), "macro_ovr"])})
    frag.append(md_table(pd.DataFrame(rows)))
    frag += ["", "**Sheet 3 (new): fraction of annotated tiles of each class kept by the tumor mask (cross-fitted), and "
             "own-class recall.**", ""]
    rows = []
    plo = pl.copy()
    plo["_e"] = plo["encoder"].map({"uni2": 0, "virchow2": 1, "h_optimus_1": 2, "conch_v15": 3})
    plo["_b"] = plo["budget_percent"].map({100: 0, 5: 1})
    plo = plo.sort_values(["_b", "_e", "true_label"])
    for _, x in plo.iterrows():
        rows.append({"Encoder": ENC_LABEL[x.encoder], "Budget": f"{int(x.budget_percent)}%",
                     "Annotated class": LABEL[int(x.true_label)], "Tiles": f"{int(x.tiles):,}",
                     "Kept as tumor": p3(x.pass_rate_predicted_tumour), "Own-class recall": p3(x.recall_own_class)})
    frag.append(md_table(pd.DataFrame(rows)))
    (HERE / "S16_amendment_tables.md").write_text("\n".join(frag) + "\n", encoding="utf-8")

    # Fig. 3 panel data
    clsdf[clsdf["budget_percent"] == 100].to_csv(HERE / "fig3_panel_A_data.csv", index=False)
    a5 = a[a["k"] == 5]
    a5[a5["budget"] == 100].to_csv(HERE / "fig3_panel_B_data.csv", index=False)
    a5[a5["budget"] == 5].to_csv(HERE / "fig3_panel_C_data.csv", index=False)

    # ------------------------------------------------------------- prose counts
    w = pd.read_csv(PRIMARY / "wording_classes.csv")
    counts = []
    for name, d in [("primary", PRIMARY), ("s6c_primary", S6C_P), ("s6c_secondary", S6C_S)]:
        ww = pd.read_csv(d / "wording_classes.csv")
        for bud in [100, 5]:
            vc = ww[ww["budget"] == bud]["wording_class"].value_counts().to_dict()
            for k_, v in sorted(vc.items()):
                counts.append({"item": f"wording_class_count:{name}:budget{bud}:class_{k_}", "value": v})
        m = ww.merge(w, on=["encoder", "method", "arm", "budget"], suffixes=("", "_primary"))
        ch = m[m["wording_class"] != m["wording_class_primary"]]
        for _, x in ch.iterrows():
            counts.append({"item": f"class_change_vs_primary:{name}:{x.encoder}:{x.method}:{x.arm}:{x.budget}",
                           "value": f"{x.wording_class_primary}->{x.wording_class}"})
    g = c[(c["family"] == "a_filtered_minus_all_tissue") & (c["level"] == "group") & (c["k"] == 5) &
          (c["population"] == "both_returned")]
    adv = order_frame(g[g["ci_high"] < 0].assign(budget=lambda d: d["budget"].astype(int)))
    for _, x in adv.iterrows():
        counts.append({"item": f"adverse_group_cell_k5:{x.encoder}:{x.method}:{x.arm}:{x.budget}:{x.endpoint}:{x.group}",
                       "value": f"{dprose(x.point)} {ciprose(x.ci_low, x.ci_high)} n={int(x.n_queries)}"})
    counts.append({"item": "group_cells_k5_conditional_total", "value": len(g)})
    counts.append({"item": "group_cells_k5_conditional_ci_excludes0_against", "value": int((g["ci_high"] < 0).sum())})
    counts.append({"item": "group_cells_k5_conditional_ci_excludes0_favor", "value": int((g["ci_low"] > 0).sum())})
    dep = sens[sens["analysis"] == "stratum_deployment_unannotated"]
    for bud in [100, 5]:
        x = dep[(dep["budget"] == bud) & (dep["method"] != "titan")]
        counts.append({"item": f"deployment_stratum_budget{bud}_OT_YX_cells", "value": len(x)})
        counts.append({"item": f"deployment_stratum_budget{bud}_OT_YX_hit_ci_low_gt0",
                       "value": int((x["hit_ci_low"] > 0).sum())})
        counts.append({"item": f"deployment_stratum_budget{bud}_OT_YX_precision_ci_low_gt0",
                       "value": int((x["precision_ci_low"] > 0).sum())})
        xt = dep[(dep["budget"] == bud) & (dep["method"] == "titan")]
        counts.append({"item": f"deployment_stratum_budget{bud}_TITAN_any_ci_excludes0",
                       "value": int(((xt["hit_ci_low"] > 0) | (xt["hit_ci_high"] < 0) |
                                     (xt["precision_ci_low"] > 0) | (xt["precision_ci_high"] < 0)).sum())})
    b5 = b[b["k"] == 5]
    counts.append({"item": "family_b_k5_point_min", "value": min(b5["hit_point"].min(), b5["precision_point"].min())})
    counts.append({"item": "family_b_k5_point_max", "value": max(b5["hit_point"].max(), b5["precision_point"].max())})
    for ep in ["hit", "precision"]:
        ex = b5[(b5[f"{ep}_ci_low"] > 0) | (b5[f"{ep}_ci_high"] < 0)]
        for _, x in ex.iterrows():
            counts.append({"item": f"family_b_k5_ci_excludes0:{ep}:{x.encoder}:{x.method}:{x.arm}",
                           "value": f"{dprose(x[f'{ep}_point'])} {ciprose(x[f'{ep}_ci_low'], x[f'{ep}_ci_high'])}"})
    c5 = cc[cc["k"] == 5]
    for ep in ["hit", "precision"]:
        ex = c5[(c5[f"{ep}_ci_low"] > 0) | (c5[f"{ep}_ci_high"] < 0)]
        for _, x in ex.iterrows():
            counts.append({"item": f"family_c_k5_ci_excludes0:{ep}:{x.encoder}:{x.method}:budget{x.budget}",
                           "value": f"{dprose(x[f'{ep}_point'])} {ciprose(x[f'{ep}_ci_low'], x[f'{ep}_ci_high'])}"})
    # conditional hit@5 versus full-target hit-yield@5: sign or exclusion disagreements (specification section 6)
    def _status(lo, hi):
        return "favor" if lo > 0 else ("against" if hi < 0 else "includes_0")
    for _, x in a[a["k"] == 5].iterrows():
        s1 = _status(x.hit_ci_low, x.hit_ci_high)
        s2 = _status(x.hit_yield_ci_low, x.hit_yield_ci_high)
        if s1 != s2 or (x.hit_point > 0) != (x.hit_yield_point > 0):
            counts.append({"item": f"hit_vs_hit_yield_disagreement_k5:{x.encoder}:{x.method}:{x.arm}:{x.budget}",
                           "value": f"hit {dprose(x.hit_point)} {ciprose(x.hit_ci_low, x.hit_ci_high)} ({s1}); "
                                    f"hit-yield {dprose(x.hit_yield_point)} {ciprose(x.hit_yield_ci_low, x.hit_yield_ci_high)} ({s2}); "
                                    f"unrounded lower bounds {x.hit_ci_low:.6f} / {x.hit_yield_ci_low:.6f}"})
    pd.DataFrame(counts).to_csv(HERE / "prose_counts.csv", index=False)

    # ------------------------------------------------------------- Markdown
    md = []
    md.append("# Supplementary Tables S-R2.1-B and S-R2.1-D — candidate content (R2.1-TF2, writing release 001)\n")
    md.append("*Generated by `make_tables.py` from the accepted S2/S6 outputs (input hashes in "
              "`inputs_sha256.csv`). Formatting only; no value recomputed. Differences and proportions to 4 decimals; "
              "U+2212 minus. The CSVs beside this file carry the unrounded values.*\n")
    md.append("Replaces the provisional content of supplementary-register rows **S32 (S-R2.1-B)** and "
              "**S33 (S-R2.1-D)** in `docs/manuscript_supplement_edits.md`; the header blocks, register "
              "wording and lineage notes are in `edit_sheet_r2_1_filtering.md` (blocks R2.1-T6-A and R2.1-T11-A).\n")

    md.append("## Table S-R2.1-B. Tumor-only filtering under query-patient-clean masks (TF-2): filtered minus all-tissue retrieval\n")
    md.append(CAPTION_B + "\n")
    md.append(ROUNDING_NOTE + "\n")
    md.append("### Part 1. Filtered − all-tissue, macro over 14 groups, k = 5 (all k in the CSV)\n")
    p1 = a[a["k"] == 5].copy()
    rows = []
    for _, x in p1.iterrows():
        rows.append({
            "Budget": f"{x.budget} %" + (" (primary)" if x.budget == 100 else " (sensitivity)"),
            "Encoder": ENC_LABEL[x.encoder], "Method": METH[x.method], "Arm": ARM[x.arm],
            "Δ hit@5 [95 % CI] (conditional)": f"{d3(x.hit_point)} {ci(x.hit_ci_low, x.hit_ci_high)}",
            "Δ precision@5 [95 % CI] (conditional)": f"{d3(x.precision_point)} {ci(x.precision_ci_low, x.precision_ci_high)}",
            "Δ hit-yield@5 [95 % CI] (full target)": f"{d3(x.hit_yield_point)} {ci(x.hit_yield_ci_low, x.hit_yield_ci_high)}",
            "Queries (cond. / target)": f"{x.hit_n_queries:,} / {x.hit_yield_n_queries:,}",
        })
    p1md = pd.DataFrame(rows)
    wc = w.copy()
    wc["key"] = wc["encoder"] + wc["method"] + wc["arm"] + wc["budget"].astype(str)
    p1["key"] = p1["encoder"] + p1["method"] + p1["arm"] + p1["budget"].astype(str)
    cls_map = dict(zip(wc["key"], "(" + wc["wording_class"] + ") " + wc["wording"]))
    p1md.insert(4, "Wording class (pre-registered rule)", [cls_map[k_] for k_ in p1["key"]])
    md.append(md_table(p1md) + "\n")

    md.append("### Part 2. Label budget: 5 % − 100 %, within encoder × method × arm, k = 5\n")
    rows = []
    for _, x in b[b["k"] == 5].iterrows():
        rows.append({"Encoder": ENC_LABEL[x.encoder], "Method": METH[x.method], "Arm": ARM[x.arm],
                     "Δ hit@5 [95 % CI]": f"{d3(x.hit_point)} {ci(x.hit_ci_low, x.hit_ci_high)}",
                     "Δ precision@5 [95 % CI]": f"{d3(x.precision_point)} {ci(x.precision_ci_low, x.precision_ci_high)}",
                     "Δ hit-yield@5 [95 % CI]": f"{d3(x.hit_yield_point)} {ci(x.hit_yield_ci_low, x.hit_yield_ci_high)}"})
    md.append(md_table(pd.DataFrame(rows)) + "\n")
    md.append("No equivalence margin was pre-registered; these differences are reported with their intervals "
              "and are not described as equivalence or similarity.\n")

    md.append("### Part 3. Arm: query-only − both-sided, within encoder × method × budget, k = 5\n")
    rows = []
    for _, x in cc[cc["k"] == 5].iterrows():
        rows.append({"Budget": f"{x.budget} %", "Encoder": ENC_LABEL[x.encoder], "Method": METH[x.method],
                     "Δ hit@5 [95 % CI]": f"{d3(x.hit_point)} {ci(x.hit_ci_low, x.hit_ci_high)}",
                     "Δ precision@5 [95 % CI]": f"{d3(x.precision_point)} {ci(x.precision_ci_low, x.precision_ci_high)}",
                     "Δ hit-yield@5 [95 % CI]": f"{d3(x.hit_yield_point)} {ci(x.hit_yield_ci_low, x.hit_yield_ci_high)}"})
    md.append(md_table(pd.DataFrame(rows)) + "\n")

    md.append("### Part 4. Sensitivity analyses (reported beside the primary; they do not replace or re-rank it), filtered − all-tissue, k = 5, conditional\n")
    md.append(PART4_NOTE + "\n")
    rows = []
    lab = {"s6c_primary_168_removed": "S6-c primary: 168 annotated library slides removed",
           "s6c_secondary_355_removed": "S6-c secondary: 355 slides of annotated patients removed",
           "stratum_deployment_unannotated": "Deployment stratum (queries from unannotated patients)",
           "stratum_annotated_patient": "Annotated-patient stratum (descriptive)"}
    for _, x in sens.iterrows():
        rows.append({"Analysis": lab[x.analysis], "Budget": f"{x.budget} %", "Encoder": ENC_LABEL[x.encoder],
                     "Method": METH[x.method], "Arm": ARM[x.arm],
                     "Δ hit@5 [95 % CI]": f"{d3(x.hit_point)} {ci(x.hit_ci_low, x.hit_ci_high)}",
                     "Δ precision@5 [95 % CI]": f"{d3(x.precision_point)} {ci(x.precision_ci_low, x.precision_ci_high)}",
                     "Queries / patients": f"{x.hit_n_queries:,} / {x.n_patients:,}"})
    sd = pd.DataFrame(rows)
    # wording classes for the S6-c rows (from the S6-c wording_classes.csv)
    wcs = {}
    for name, d in [("s6c_primary_168_removed", S6C_P), ("s6c_secondary_355_removed", S6C_S)]:
        ww = pd.read_csv(d / "wording_classes.csv")
        for _, y in ww.iterrows():
            wcs[(name, y.encoder, y.method, y.arm, int(y.budget))] = f"({y.wording_class})"
    sd.insert(5, "Class (S6-c only)", [wcs.get((x.analysis, x.encoder, x.method, x.arm, int(x.budget)), "descriptive")
                                       for _, x in sens.iterrows()])
    md.append(md_table(sd) + "\n")

    md.append("### Part 5. Coverage (pooled over slides; not a macro quantity)\n")
    rows = []
    cv = cov.copy()
    cv["_e"] = cv["encoder"].map({e: i for i, e in enumerate(ENC)})
    cv["_a"] = cv["arm"].map({"all_tissue": 0, "query_only": 1, "both_filtered": 2})
    cv["_m"] = cv["method"].map({"ot_cosine": 0, "yottixel": 1, "titan": 2})
    cv = cv.sort_values(["_e", "_m", "_a", "budget"])
    for _, x in cv.iterrows():
        rows.append({"Encoder": ENC_LABEL[x.encoder], "Method": METH[x.method],
                     "Arm": {"all_tissue": "all-tissue"}.get(x.arm, ARM.get(x.arm)),
                     "Budget": "—" if x.budget == 0 else f"{x.budget} %",
                     "Intended": f"{x.intended:,}", "Returned": f"{x.returned:,}",
                     "Non-returns": int(x.non_returns),
                     "Non-returned slide": "—" if pd.isna(x.non_return_slides) else str(int(x.non_return_slides))})
    md.append(md_table(pd.DataFrame(rows)) + "\n")
    md.append(PART5_NOTE + "\n")

    md.append("### Part 6. Yottixel k-boundary ties: fixed order (primary) versus tie-averaged, filtered − all-tissue, k = 5, conditional\n")
    rows = []
    for _, x in tw.iterrows():
        rows.append({"Budget": f"{x.budget} %", "Encoder": ENC_LABEL[x.encoder], "Arm": ARM[x.arm],
                     "Endpoint": {"hit": "hit@5", "precision": "precision@5"}[x.endpoint],
                     "Fixed order [95 % CI]": f"{d3(x.point_stable_reference_index)} {ci(x.ci_low_stable_reference_index, x.ci_high_stable_reference_index)}",
                     "Tie-averaged [95 % CI]": f"{d3(x.point_tie_averaged)} {ci(x.ci_low_tie_averaged, x.ci_high_tie_averaged)}"})
    md.append(md_table(pd.DataFrame(rows)) + "\n")
    md.append(PART6_NOTE + "\n")

    md.append("## Table S-R2.1-D. Relative error reduction of top-5 plurality-vote macro F1, filtered versus all-tissue, common 2,604-slide population (TF-2)\n")
    md.append(CAPTION_D + "\n")
    md.append(ROUNDING_NOTE + "\n")
    md.append("### Part 1. All 28 macro cells (non-returned queries scored as misses; primary handling)\n")
    rows = []
    for _, x in rm[rm["handling"] == "miss"].iterrows():
        role = {"headline_prenamed_cell": "pre-named headline cell", "macro_table": "table",
                "macro_table_sensitivity_budget5": "table (5 % sensitivity)"}[x.cell_role]
        rows.append({"Budget": f"{int(x.budget)} %", "Encoder": ENC_LABEL[x.encoder], "Method": METH[x.method],
                     "Arm": ARM[x.arm], "Role": role,
                     "F1 all-tissue": p3(x.f1_all), "F1 filtered": p3(x.f1_filt),
                     "ΔF1 [95 % CI]": f"{d3(x.delta_f1)} {ci(x.delta_ci_low, x.delta_ci_high)}",
                     "RER [95 % CI]": f"{d3(x.rer)} {ci(x.rer_ci_low, x.rer_ci_high)}",
                     "Support (queries)": f"{int(x.support):,}",
                     "Non-returned (filtered)": int(x.n_nonreturned_filt)})
    md.append(md_table(pd.DataFrame(rows)) + "\n")
    md.append("### Part 2. Same 28 cells, non-returned queries dropped (secondary handling)\n")
    rows = []
    for _, x in rm[rm["handling"] == "both_returned"].iterrows():
        rows.append({"Budget": f"{int(x.budget)} %", "Encoder": ENC_LABEL[x.encoder], "Method": METH[x.method],
                     "Arm": ARM[x.arm],
                     "ΔF1 [95 % CI]": f"{d3(x.delta_f1)} {ci(x.delta_ci_low, x.delta_ci_high)}",
                     "RER [95 % CI]": f"{d3(x.rer)} {ci(x.rer_ci_low, x.rer_ci_high)}",
                     "Support (queries)": f"{int(x.support):,}"})
    md.append(md_table(pd.DataFrame(rows)) + "\n")
    md.append("### Part 3. Per-group table for the pre-named headline cell (UNI2, Optimal Transport, both-sided, 100 %; misses)\n")
    rows = []
    for _, x in head.iterrows():
        rows.append({"Group": "Overall (macro)" if x["class"] == "MACRO" else x["class"],
                     "Support (queries)": f"{int(x.support):,}",
                     "F1 all-tissue": p3(x.f1_all), "F1 filtered": p3(x.f1_filt),
                     "ΔF1 [95 % CI]": f"{d3(x.delta_f1)} {ci(x.delta_ci_low, x.delta_ci_high)}",
                     "RER [95 % CI]": f"{d3(x.rer)} {ci(x.rer_ci_low, x.rer_ci_high)}",
                     "Undefined replicates (of 5,000)": int(x.n_undefined_reps)})
    md.append(md_table(pd.DataFrame(rows)) + "\n")
    md.append(PART_D3_NOTE + "\n")

    # Part 4: lineage sheet (design decision 2026-10-17): earlier designs, not used for any claim
    md.append("### Part 4. Lineage sheet: relative error reduction under the earlier, superseded filtering designs (not used for any claim)\n")
    md.append(LINEAGE_NOTE + "\n")
    lin = []
    for pkg, path in LINEAGE_RER:
        d = pd.read_csv(path)
        d = d[(d["class"].isin(["MACRO", "LIVER"])) & (d["handling"] == "miss")].copy()
        d.insert(0, "package", pkg)
        lin.append(d)
    lin = pd.concat(lin, ignore_index=True)
    keep = ["package", "encoder", "method", "design", "budget", "rule", "handling", "class", "f1_all", "f1_filt",
            "delta_f1", "delta_ci_low", "delta_ci_high", "rer", "rer_ci_low", "rer_ci_high", "rer_ci_status",
            "n_undefined_reps", "support", "n_queries", "n_nonreturned_filt", "submitted_delta"]
    lin = lin[keep]
    lin.to_csv(HERE / "S_R2_1_D_part4_lineage.csv", index=False)
    enc_l = {"uni2": "UNI2", "virchow2": "Virchow2", "hoptimus": "H-Optimus-1"}
    des_l = {"legacy_tumour_only": "submitted (legacy) classifier, re-scored", "query_only": "query-only",
             "both_filtered": "both-sided"}
    pk_l = {"r2_1_filter_rer_20261012/run_001": "10-12 fair-design RER",
            "r2_1_route_b_uni2_20261013/rer_run_002": "10-13 matched UNI2 heads"}
    rows = []
    for _, x in lin.iterrows():
        heads = ("legacy classifier" if x.design == "legacy_tumour_only" else
                 ("202-slide patient-fold heads" if x.encoder == "hoptimus" else "202-slide matched patient-fold heads"))
        rows.append({"Package": pk_l[x.package], "Encoder": enc_l[x.encoder], "Method": METH[x.method],
                     "Design / heads": des_l[x.design] if x.design == "legacy_tumour_only" else f"{des_l[x.design]}; {heads}",
                     "Budget": "—" if str(x.budget) == "legacy" else f"{int(float(x.budget))} %",
                     "Rule": {"plurality": "plurality", "legacy_generous_ORACLE": "submitted (oracle)"}[x.rule],
                     "Row": "Overall (macro)" if x["class"] == "MACRO" else "LIVER",
                     "RER [95 % CI]": f"{d3(x.rer)} {ci(x.rer_ci_low, x.rer_ci_high)}" + ("" if x.rer_ci_status == "reported" else " (not estimable)"),
                     "ΔF1 [95 % CI]": f"{d3(x.delta_f1)} {ci(x.delta_ci_low, x.delta_ci_high)}",
                     "Submitted Table S5 Delta": "—" if pd.isna(x.submitted_delta) else p3(x.submitted_delta),
                     "Queries (non-returned)": f"{int(x.n_queries):,} ({int(x.n_nonreturned_filt)})"})
    md.append(md_table(pd.DataFrame(rows)) + "\n")
    (HERE / "tables_S_R2_1_B_D.md").write_text("\n".join(md), encoding="utf-8")


CAPTION_B = (
    "**Caption (candidate).** Supplementary Table S-R2.1-B. Tumor-only filtering evaluated with patient-held-out, "
    "query-fold-consistent tumor masks (TF-2) for UNI2, Virchow2, H-Optimus-1 (Optimal Transport and Yottixel) and "
    "CONCH v1.5 → TITAN (slide embedding, L1 distance). Tumor heads (linear SGD, five classes, tumor = most probable "
    "class) were cross-fitted over five patient folds on the 229 annotated slides scanned at 40×, plus one head on all "
    "229 slides; each query was masked, together with every library slide it searched (both-sided arm), by the head of "
    "its own patient fold if its patient was annotated and otherwise by the all-slide head, so no mask a query "
    "encountered was trained on that patient's labels. Query-only: masked query against the unfiltered library. "
    "Population: 2,604 tumor query slides, 14 diagnostic groups, 1,800 patients; same-patient slides excluded from "
    "each query's candidates. Δ = filtered − all-tissue (Parts 1, 4, 6), 5 % − 100 % (Part 2) or query-only − "
    "both-sided (Part 3), macro-averaged over the 14 groups. Conditional estimand: queries returned in both arms; "
    "full-target estimand (hit-yield): all intended queries, a non-returned query counted as a miss. 95 % intervals: "
    "whole-patient percentile bootstrap, 5,000 replicates, seed 123, one draw matrix shared by every arm, method, "
    "encoder and budget, so differences are paired; unadjusted for multiplicity. The 100 % label budget is primary "
    "and 5 % the sensitivity analysis. Wording classes follow the rule registered before the outputs were read "
    "(classes (i)–(v), contract §6). Encoders are not compared; no cross-encoder contrast was pre-specified. "
    "Group-level rows (k = 1, 3, 5, 10) are in the CSV release `S_R2_1_B_*.csv` and the evaluator's `contrasts.csv`.")

LINEAGE_NOTE = (
    "Retained unmodified from the frozen packages `r2_1_filter_rer_20261012/run_001/contrasts.csv` and "
    "`r2_1_route_b_uni2_20261013/rer_run_002/contrasts.csv` (overall macro and LIVER rows; non-returns as misses; both "
    "tie rules). These designs differ from TF-2: the submitted UNI2/Virchow2 classifiers were not held out by patient "
    "(re-scored on the common 2,604-slide population); the H-Optimus-1 and matched-UNI2 heads were trained on 202 "
    "annotated slides with head f masking fold-f queries and the whole library, and unannotated queries were masked by "
    "fold heads rather than the deployable classifier (contract candidate 003 §0, §10). They are superseded for the "
    "filtering answer and are shown so that each earlier value can be traced; no value in this part is used as a "
    "result or headline. The submitted Table S5 Delta is repeated for the legacy rows.")

ROUNDING_NOTE = (
    "*Rounding.* Values are printed to 4 decimals; a bound printed as +0.0000 or \u22120.0000 lies within 0.00005 "
    "of zero, and its sign, and therefore whether the interval excludes zero, is that of the unrounded value in the "
    "CSV.")

PART4_NOTE = (
    "S6-c re-evaluates every arm after removing columns from the computed matrices (no new retrieval): primary, "
    "the 168 annotated slides that are members of the 2,604-slide reference population; secondary, the 355 "
    "reference slides of the 215 annotated patients. Classes for the S6-c rows are those of the same mechanical rule "
    "applied to the S6-c intervals. Strata are descriptive only (no class): the deployment stratum is the 2,249 "
    "queries from unannotated patients, masked with the all-slide head and searching the library masked by it; the "
    "annotated-patient stratum (355 queries, 173 patients) has no macro interval because some groups had no query in "
    "821 of the 5,000 replicates.")

PART5_NOTE = (
    "Slide 14601 (LIVER; patient unannotated) keeps no tile under the Virchow2 and H-Optimus-1 all-slide heads at "
    "either budget and is therefore a non-returned query in all 16 filtered arms of those two encoders (left out of "
    "the conditional estimand, counted as a miss in hit-yield and in Table S-R2.1-D) and absent as a reference from "
    "the affected both-sided library versions. Slides 15678 (OS) and 16738 (RB) are absent, as references only, from "
    "a few both-sided library versions (Virchow2 and H-Optimus-1: zero retained tiles; TITAN: fewer than two "
    "retained patches, the slide encoder's technical minimum); both are returned as queries in every arm. Slides "
    "with 1–7 retained tiles were kept (Optimal Transport, Yottixel).")

PART6_NOTE = (
    "Optimal Transport and TITAN rankings had no k-boundary ties. Yottixel had 367–456 boundary-tied queries per arm "
    "at k = 5. Tie-averaging moves the macro differences by at most 0.005 and changes no 100 % class; one 5 % "
    "sensitivity cell (Virchow2, Yottixel, both-sided, hit@5) would move from class (i) to class (ii).")

CAPTION_D = (
    "**Caption (candidate).** Supplementary Table S-R2.1-D. Relative error reduction (RER) of the macro F1 of a top-5 "
    "plurality vote (non-oracle; no use of the true label to break ties), filtered versus all-tissue, on the common "
    "2,604-slide, 14-group population with identical queries in both arms, under the TF-2 masks of Table S-R2.1-B. "
    "RER = (F1_filtered − F1_all-tissue) / (1 − F1_all-tissue). Primary handling: a query left without tumor tiles "
    "is scored as a miss; secondary: such queries are dropped. 95 % intervals: the shared whole-patient bootstrap of "
    "Table S-R2.1-B; group intervals are unadjusted and wide. The pre-named headline cell (registered before the "
    "outputs were read) is UNI2 + Optimal Transport, both-sided, 100 %, overall macro RER; no other cell, and no "
    "group cell, is used as a headline, and no \"up to\" value is drawn from this table. Replaces the provisional "
    "content of S33; the submitted Table S5 (relative error reduction per group, submitted classifier and design) "
    "is retained unchanged as the submitted analysis, and the 2026-10-12/13 fair-design RER rows (legacy UNI2/"
    "Virchow2 classifiers, H-Optimus-1 and matched-UNI2 heads under the earlier design) are retained as lineage "
    "in Part 4 (edit-sheet block R2.1-T11-A), not used for any claim.")

PART_D3_NOTE = (
    "Adverse cells are shown at the same prominence as favorable ones: DSRCT (31 queries) has the only group "
    "interval of this cell that excludes zero against filtering. RB: 32 of 5,000 replicates had an undefined RER "
    "(all-tissue error zero) and were dropped from the RER interval (below the 1 % reporting threshold); NF: 1. "
    "Cells with more than 1 % undefined replicates are marked not estimable in the CSV (`rer_ci_status`): "
    "H-Optimus-1 Optimal Transport NF and RB (all four arm × budget cells) and TITAN RB (all four).")

if __name__ == "__main__":
    main()
