#!/usr/bin/env python3
"""Workspace Table S16 (final number from scripts/figures/supp_table_numbering.json): tumour tile-classifier
performance, both evaluations in one workbook.

Assembly only (A_packaging, 2026-11-09). Spec: docs/manuscript_supplement_edits.md ### R3.5-T1 (Sheet 1, caption,
footnotes, P2 sensitivity paragraph, Do-not-add) as amended by ### R3.5-T1-A (Sheet 1 column rename; Sheets 2, 2b, 3;
caption append; Do-not-add). Every printed cell is generated from the bound sources and compared string-for-string
with the spec's tables; raw unrounded sources are copied verbatim under raw/.

Run from the repository root:
    export PROSPER_OUT="$PWD/prosper_revision_output"
    /HOME/miniconda3/bin/python3 scripts/supp_tables/build_S16_classifier_20261109.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from packaging_common_20261109 import (BUILD_DATE, EDIT_SHEET, NUMBERING, OUT_ROOT, PKG, REPO, SI_MD, Bindings,  # noqa: E402
                                       Report, find_line, md_table_after, now_utc, numbering, rel, save_xlsx,
                                       caption_fields, write_caption_sheet, write_rows_sheet, write_sha256sums,
                                       write_sheet_csv)

WID = "S16"
OUT = PKG / WID
R35 = OUT_ROOT / "r3_5_s16_whole_patient_ci_20261008"
TF2 = OUT_ROOT / "r2_1_filter_tf2_20261016"
W1 = TF2 / "writing_001"
CLS = TF2 / "s2/run_002/classifier_report_001"
AUC_RUN002 = TF2 / "s2/run_002/lead_auroc_recount_001/auroc_recount.csv"
SUBMITTED = REPO / "docs/prosper_submission/supplementary/supplementary_tables/Table_S16_tumor_classifier_performance.xlsx"
MS = REPO / "docs/prosper_revised/prosper_manuscript_revised.md"
FIGS4_PROV = OUT_ROOT / "figures_20261021/figS_r21_f4_classifier_budget/PROVENANCE.md"

P1, P2 = "P1_all237_unmapped_singletons", "P2_mapped202_192patients"
ENC_LABEL = {"uni2": "UNI2", "virchow2": "Virchow2", "h_optimus_1": "H-Optimus-1", "conch_v15": "CONCH v1.5 (TITAN arm)"}
LABEL = {1: "Tumor", 2: "Immune cells", 3: "Necrosis", 4: "Non-tumor", 5: "Artifacts"}
SUBMITTED_COL = "AUROC as submitted (in-sample: tiles from the classifier's training slides; test tiles)"


def p4(x: float) -> str:
    return f"{x:.4f}"


def quoted_block_after(needle: str, start: int = 1) -> tuple[int, str]:
    """The `> `-quoted paragraph following the needle line, joined (spec wording, verbatim)."""
    lines = EDIT_SHEET.read_text(encoding="utf-8").splitlines()
    i = find_line(EDIT_SHEET, needle, start) - 1
    while not lines[i].startswith(">"):
        i += 1
    first = i + 1
    out = []
    while i < len(lines) and lines[i].startswith(">"):
        out.append(lines[i][1:].strip())
        i += 1
    return first, " ".join(out)


def cell_addr(ws, row_match, col_name: str) -> tuple[str, object]:
    hdr = [c.value for c in ws[1]]
    j = hdr.index(col_name) + 1
    for r in range(2, ws.max_row + 1):
        if row_match([c.value for c in ws[r]]):
            return f"'{ws.title}'!{get_column_letter(j)}{r}", ws.cell(r, j).value
    raise KeyError(col_name)


def main() -> int:
    R = Report()
    B = Bindings()
    num = numbering(WID)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "raw"
    raw.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- inputs
    f_sum = B.bind(R35 / "summary.csv", "Sheet 1 slide-held-out column; P2 sensitivity paragraph")
    f_units = B.bind(R35 / "units.csv", "Sheet 1 units (resampling units)")
    f_rec = B.bind(R35 / "reconciliation.json", "r3_5_s16 run record (read; populations, fold-crossing)")
    f_readme = B.bind(R35 / "README.md", "r3_5_s16 README (read; evaluation design)")
    f_r35sha = B.bind(R35 / "sha256.txt", "r3_5_s16 own hash list")
    f_sub = B.bind(SUBMITTED, "Sheet 1 submitted column (submitted a8b14a04…)")
    f_cf = B.bind(W1 / "S16_amendment_tf2_cross_fitted.csv", "Sheet 2 sensitivity/specificity")
    f_au = B.bind(W1 / "S16_auroc_recount_229pool.csv", "Sheets 2 and 2b AUROC (4 decimals)")
    f_pl = B.bind(W1 / "S16_amendment_tf2_per_label.csv", "Sheet 3")
    f_mt = B.bind(W1 / "make_tables.py", "generator of the writing_001 S16 CSVs (read)")
    f_w1sha = B.bind(W1 / "inputs_sha256.csv", "writing_001 input hashes")
    f_tm = B.bind(CLS / "tile_metrics.csv", "TF-2 confusion counts per fold (Fig. S4 panel-a source)")
    f_s2b = B.bind(CLS / "s2b_full_noqc_vs_full.csv", "all-slide head on pool tiles (in-sample)")
    f_pl0 = B.bind(CLS / "per_label_pass_rates.csv", "Sheet 3 upstream")
    f_au0 = B.bind(AUC_RUN002, "AUROC recount run_002 (upstream of the writing_001 copy)")
    f_au0out = B.bind(AUC_RUN002.with_suffix(".out"), "AUROC recount stdout (4 decimals only)")
    f_figs4 = B.bind(FIGS4_PROV, "Fig. S4 (S-R2.1-F4) provenance: panel-a source")
    B.bind(EDIT_SHEET, "spec")
    B.bind(SI_MD, "SI title / notes (live file; hash at build)")
    B.bind(MS, "manuscript claim lookup (live file; hash at build)")
    B.bind(NUMBERING, "numbering")

    s = pd.read_csv(f_sum)
    units = pd.read_csv(f_units, dtype={"unit": str, "slides": str})
    rec = json.load(open(f_rec))
    cf = pd.read_csv(f_cf)
    au = pd.read_csv(f_au)
    pl = pd.read_csv(f_pl)
    tm = pd.read_csv(f_tm)
    s2b = pd.read_csv(f_s2b)
    sub_ws = openpyxl.load_workbook(f_sub)["Sheet1"]
    sub = {r[0]: (r[1], r[2]) for r in sub_ws.iter_rows(min_row=2, values_only=True)}

    # ---------------------------------------------------------------- provenance checks on the sources
    r35sha = dict(reversed(l.split()) for l in f_r35sha.read_text().splitlines() if l.strip())
    for f in (f_sum, f_units, f_rec, f_readme):
        R.check(r35sha.get(f.name) == B.sha(f), f"`{f.name}` sha256 {B.sha(f)[:12]}… equals r3_5_s16 sha256.txt")
    w1 = dict(pd.read_csv(f_w1sha).values.tolist())
    for f, key in ((f_tm, "r2_1_filter_tf2_20261016/s2/run_002/classifier_report_001/tile_metrics.csv"),
                   (f_s2b, "r2_1_filter_tf2_20261016/s2/run_002/classifier_report_001/s2b_full_noqc_vs_full.csv"),
                   (f_pl0, "r2_1_filter_tf2_20261016/s2/run_002/classifier_report_001/per_label_pass_rates.csv"),
                   (f_au0, "r2_1_filter_tf2_20261016/s2/run_002/lead_auroc_recount_001/auroc_recount.csv")):
        R.check(w1.get(key) == B.sha(f), f"`{f.name}` sha256 {B.sha(f)[:12]}… equals writing_001 inputs_sha256.csv")
    R.check(B.sha(f_au) == B.sha(f_au0),
            "writing_001 `S16_auroc_recount_229pool.csv` is byte-identical to `s2/run_002/lead_auroc_recount_001/"
            "auroc_recount.csv` (Do-not-add: no values from the run_002-less `s2/lead_auroc_recount_001/`)")
    pl0 = pd.read_csv(f_pl0)
    R.check(pl0.shape == pd.read_csv(f_pl).shape and bool((pl0.values == pd.read_csv(f_pl).values).all()),
            "writing_001 `S16_amendment_tf2_per_label.csv` equals `classifier_report_001/per_label_pass_rates.csv` value for "
            "value (bytes differ only by pandas re-serialisation, '0' → '0.0')")
    R.check("897dc45747c1e213eb05636ca1facb5b9a82fb3e703b627cd3fa2d76ab76ac3f" in FIGS4_PROV.read_text()
            and B.sha(f_tm).startswith("897dc457"),
            "Fig. S4 (S-R2.1-F4) panel a is drawn from the same `tile_metrics.csv` (sha256 897dc457…) as Sheet 2")
    R.check(B.sha(f_sub).startswith("a8b14a04"), f"submitted S16 workbook sha256 {B.sha(f_sub)[:12]}… (a8b14a04…)")

    # sensitivity/specificity re-derived from the confusion counts (exact)
    tm["sens_re"] = tm.tp / (tm.tp + tm.fn)
    tm["spec_re"] = tm.tn / (tm.tn + tm.fp)
    dmax = max((tm.sens_re - tm.sensitivity).abs().max(), (tm.spec_re - tm.specificity).abs().max())
    R.check(dmax <= 5e-10,
            f"tile_metrics sensitivity = tp/(tp+fn) and specificity = tn/(tn+fp) for all {len(tm)} rows to the stored "
            f"9 decimals (max |Δ| {dmax:.1e} ≤ 5e-10)")
    bad = 0
    for _, x in cf.iterrows():
        g = tm[(tm.encoder == x.encoder) & (tm.budget_percent == x.budget_percent)]
        pooled = g[g.scope == "pooled_cross_fitted"].iloc[0]
        folds = g[g.scope.str.startswith("fold")]
        ins = json.loads(s2b[(s2b.encoder == x.encoder) & (s2b.budget_percent == x.budget_percent)
                             & (s2b.scope == "all_pool_tiles")].iloc[0]["full_in_sample"])
        vals = [(x.sens_pooled_cross_fitted, pooled.sens_re), (x.spec_pooled_cross_fitted, pooled.spec_re),
                (x.sens_fold_min, folds.sens_re.min()), (x.sens_fold_max, folds.sens_re.max()),
                (x.spec_fold_min, folds.spec_re.min()), (x.spec_fold_max, folds.spec_re.max()),
                (x.full_head_on_pool_tiles_sens, ins["sensitivity"]), (x.full_head_on_pool_tiles_spec, ins["specificity"])]
        bad += sum(abs(a - b) > 5e-10 for a, b in vals) + (len(folds) != 5) + (int(x.n_tiles) != int(pooled.n))
    R.check(bad == 0, f"Sheet 2 source rows ({len(cf)} = 4 encoders × 2 budgets): pooled and fold-range sensitivity/"
                      f"specificity re-derived from the tp/fn/fp/tn counts, n tiles and all-slide head values from "
                      f"s2b; {bad} mismatches")

    # ---------------------------------------------------------------- Sheet 1 (R3.5-T1 table, T1-A rename)
    s1 = s[s.population == P1].set_index("statistic")
    s2_ = s[s.population == P2].set_index("statistic")
    order = [("Tumor", "Tumor"), ("Non-tumor", "Non-tumor"), ("Necrosis", "Necrosis"),
             ("Immune cells", "Immune cells"), ("Artifacts", "Artifacts")]
    sheet1 = []
    for cls, subname in order:
        r = s1.loc[f"auroc_ovr_{cls}"]
        a, n = sub[subname]
        mark = {"Immune cells": "ᵃ", "Artifacts": "ᵇ"}.get(cls, "")
        if cls == "Immune cells":
            held = f"{r.point:.2f}; interval not estimable{mark}"
        elif cls == "Artifacts":
            held = f"{r.point:.2f}; interval not estimable (too few tiles){mark}"
        else:
            held = f"{r.point:.3f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
        sheet1.append([cls, f"{int(r.n_pos_tiles):,}", f"{int(r.n_slides_with_class)} ({int(r.n_units_with_class)}){mark}",
                       f"{a} ({int(n):,})", held])
    a, _ = sub["Overall (macro / multiclass)"]
    sheet1.append(["Overall (macro)", f"{int(s1.loc['auroc_ovr_Tumor'].n_tiles):,}", str(rec["populations"][P1]["n_slides"]),
                   f"{a}", "—"])
    for lab, st in (("Tumour sensitivity", "tumour_sensitivity"), ("Tumour specificity", "tumour_specificity"),
                    ("Five-class tile accuracy", "tile_accuracy_5class")):
        r = s1.loc[st]
        sheet1.append([lab, "", "", "—", f"{r.point:.3f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]"])
    l_t1 = find_line(EDIT_SHEET, "### R3.5-T1 — Supplementary Table S16 rebuilt")
    l_hdr1, hdr_spec, rows_spec = md_table_after(EDIT_SHEET, "| Class (one-vs-rest) | Tiles | Slides (patient units)", l_t1)
    hdr1 = list(hdr_spec)
    R.check(hdr1[3] == "AUROC, within-slide (submitted; test tiles)", f"spec Sheet 1 header (l.{l_hdr1}) column 4 is "
                                                                       f"'{hdr1[3]}' (renamed per R3.5-T1-A (1))")
    hdr1[3] = SUBMITTED_COL
    diffs = [(i, j, rows_spec[i][j], sheet1[i][j]) for i in range(max(len(rows_spec), len(sheet1)))
             for j in range(5) if i >= len(rows_spec) or i >= len(sheet1) or rows_spec[i][j] != sheet1[i][j]]
    R.check(not diffs, f"Sheet 1: all {len(sheet1)} rows × 5 cells generated from summary.csv P1 (printing rule: 3-decimal "
                       f"point, 2-decimal bounds) and the submitted workbook equal the R3.5-T1 spec table "
                       f"(l.{l_hdr1}) string-for-string; differences {diffs[:4]}")

    # footnotes a/b and the P2 sensitivity paragraph (verbatim from the spec), with what the sources support
    lines = EDIT_SHEET.read_text(encoding="utf-8").splitlines()
    l_fa = find_line(EDIT_SHEET, "> ᵃ 3 slides:", l_hdr1)
    fn_a = lines[l_fa - 1][1:].strip()
    fn_b = lines[l_fa][1:].strip()
    p2_par = " ".join(l[1:].strip() for l in lines[l_fa + 1:l_fa + 4])
    R.check(p2_par.startswith("Sensitivity analysis restricted to the 202 slides"), f"P2 paragraph located (l.{l_fa + 2}–{l_fa + 4})")
    unm = set(rec["unmapped_slides"])
    imm1, imm2 = s1.loc["auroc_ovr_Immune cells"], s2_.loc["auroc_ovr_Immune cells"]
    art1, art2 = s1.loc["auroc_ovr_Artifacts"], s2_.loc["auroc_ovr_Artifacts"]
    R.check(imm1.n_slides_with_class == 3 and imm2.n_slides_with_class == 2 and imm2.n_units_with_class == 2,
            "footnote a '3 slides: 2 patients plus 1 slide without a patient mapping': P1 3 slides / 3 units, P2 (mapped "
            "only) 2 slides / 2 units → 1 unmapped slide. '93% of the tiles come from one slide' is carried from the spec "
            "(source F_DELTA_MAP §C, 3,421 of 3,686 = 92.8%; per-slide class tiles are not in units.csv) — not re-derived here")
    R.check(art1.n_units_with_class == 8 and art2.n_units_with_class == 7 and "4342" in unm,
            "footnote b '8 units (7 patients and one slide, 4342, without a mapped patient)': P1 8 units, P2 7, slide 4342 "
            "in reconciliation.json unmapped_slides. 'two contribute 3 and 7 tiles' is carried from the spec (author Q-A) — "
            "per-unit class tiles are not in these files")
    exp = {"tumour": s2_.loc["auroc_ovr_Tumor"], "non-tumour": s2_.loc["auroc_ovr_Non-tumor"],
           "necrosis": s2_.loc["auroc_ovr_Necrosis"], "sensitivity": s2_.loc["tumour_sensitivity"],
           "specificity": s2_.loc["tumour_specificity"], "accuracy": s2_.loc["tile_accuracy_5class"]}
    p2_bad = [k for k, r in exp.items() if f"{k} {r.point:.3f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]" not in p2_par]
    pops = rec["populations"][P2]
    R.check(not p2_bad and f"{pops['n_slides']} slides of {pops['n_patient_units']} mapped patients ({pops['n_tiles']:,} tiles)" in p2_par,
            f"P2 sensitivity paragraph: all six values (3-decimal point, 2-decimal bounds) and '202 slides of 192 mapped "
            f"patients (468,301 tiles)' equal summary.csv P2 / reconciliation.json; mismatched {p2_bad}")

    # units
    up = {p: g for p, g in units.groupby("population")}
    ok_u = True
    msg = []
    for p in (P1, P2):
        g = up[p]
        po = rec["populations"][p]
        okp = (len(g) == po["n_units"] and int(g.n_tiles.sum()) == po["n_tiles"] and int(g.n_slides.sum()) == po["n_slides"]
               and int((g.unit_type == "patient").sum()) == po["n_patient_units"])
        ok_u &= okp
        msg.append(f"{p}: {len(g)} units ({int((g.unit_type == 'patient').sum())} patient, {int((g.unit_type != 'patient').sum())} "
                   f"slide), {int(g.n_slides.sum())} slides, {int(g.n_tiles.sum()):,} tiles")
    R.check(ok_u, "units.csv reconciles with reconciliation.json populations: " + "; ".join(msg))
    R.check(rec["fold_crossing_patients"]["n_patients"] == 7 and rec["fold_crossing_patients"]["n_slides"] == 16,
            "fold-crossing statement '7 patients (16 slides)' = reconciliation.json fold_crossing_patients")

    # ---------------------------------------------------------------- Sheets 2, 2b, 3 (R3.5-T1-A)
    l_t1a = find_line(EDIT_SHEET, "### R3.5-T1-A — amendment to R3.5-T1")
    au_cf = au[au.scope == "cross_fitted_patient_held_out"].set_index(["encoder", "budget"])
    au_in = au[au.scope == "full_head_in_sample(pool)"].set_index(["encoder", "budget"])
    sheet2, sheet2b = [], []
    for _, x in cf.iterrows():
        k = (x.encoder, int(x.budget_percent))
        sheet2.append([ENC_LABEL[x.encoder] + (" (1,024-px patches)" if x.encoder == "conch_v15" else " (512-px tiles)"),
                       f"{int(x.budget_percent)}%" + (" (primary)" if x.budget_percent == 100 else " (sensitivity)"),
                       f"{int(x.n_tiles):,}",
                       f"{p4(x.sens_pooled_cross_fitted)} [{p4(x.sens_fold_min)}–{p4(x.sens_fold_max)}]",
                       f"{p4(x.spec_pooled_cross_fitted)} [{p4(x.spec_fold_min)}–{p4(x.spec_fold_max)}]",
                       p4(au_cf.loc[k, "auroc_tumour_vs_rest"]), p4(au_in.loc[k, "auroc_tumour_vs_rest"]),
                       f"{p4(x.full_head_on_pool_tiles_sens)} / {p4(x.full_head_on_pool_tiles_spec)} ({x.full_head_on_pool_tiles_note})"])
        r_ = au_cf.loc[k]
        sheet2b.append([ENC_LABEL[x.encoder], f"{int(x.budget_percent)}%", p4(r_["auroc_ovr_Tumor"]), p4(r_["auroc_ovr_Necrosis"]),
                        p4(r_["auroc_ovr_Non-tumor"]), p4(r_["auroc_ovr_Immune"]) + "; interval not estimable",
                        p4(r_["auroc_ovr_Artifacts"]) + "; interval not estimable", p4(r_["macro_ovr"]),
                        p4(au_in.loc[k, "macro_ovr"])])
    plo = pl.assign(_e=pl.encoder.map({"uni2": 0, "virchow2": 1, "h_optimus_1": 2, "conch_v15": 3}),
                    _b=pl.budget_percent.map({100: 0, 5: 1})).sort_values(["_b", "_e", "true_label"])
    sheet3 = [[ENC_LABEL[x.encoder], f"{int(x.budget_percent)}%", LABEL[int(x.true_label)], f"{int(x.tiles):,}",
               p4(x.pass_rate_predicted_tumour), p4(x.recall_own_class)] for _, x in plo.iterrows()]
    specs = {}
    for name, needle, built in (("Sheet2", "**Sheet 2 (new): TF-2 tumor heads", sheet2),
                                ("Sheet2b", "**Sheet 2b (new): one-versus-rest AUROC", sheet2b),
                                ("Sheet3", "**Sheet 3 (new): fraction of annotated tiles", sheet3)):
        lh, h, rows = md_table_after(EDIT_SHEET, needle, l_t1a)
        specs[name] = (lh, h)
        d = [(i, j, rows[i][j], built[i][j]) for i in range(max(len(rows), len(built))) for j in range(len(h))
             if i >= len(rows) or i >= len(built) or rows[i][j] != built[i][j]]
        R.check(not d, f"{name}: all {len(built)} rows × {len(h)} cells generated from the writing_001 CSVs equal the "
                       f"R3.5-T1-A spec table (l.{lh}) string-for-string; differences {d[:4]}")
    hdr2, hdr2b, hdr3 = specs["Sheet2"][1], specs["Sheet2b"][1], specs["Sheet3"][1]
    R.check(sorted(au.budget.unique().tolist()) == [5, 100] and len(au) == 16,
            "AUROC source: 16 rows = 4 encoders × 2 budgets × {cross_fitted_patient_held_out, full_head_in_sample(pool)}")

    # ---------------------------------------------------------------- workbook
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    names = {"Sheet1": "Sheet1_within_vs_slide_held", "Sheet1u": "Sheet1_units", "Sheet2": "Sheet2_TF2_cross_fitted",
             "Sheet2b": "Sheet2b_TF2_AUROC_by_class", "Sheet3": "Sheet3_TF2_kept_by_mask"}
    write_rows_sheet(wb, names["Sheet1"], hdr1, sheet1)
    write_rows_sheet(wb, names["Sheet1u"], list(units.columns), units.astype(object).values.tolist())
    write_rows_sheet(wb, names["Sheet2"], hdr2, sheet2)
    write_rows_sheet(wb, names["Sheet2b"], hdr2b, sheet2b)
    write_rows_sheet(wb, names["Sheet3"], hdr3, sheet3)

    # caption (spec wording verbatim)
    l_cap, cap = quoted_block_after("> Supplementary Table S16. Tumour-tissue tile classifier", l_t1)
    l_cap = l_cap
    l_app, app = quoted_block_after("**(3) Caption (SI L35, as replaced by R3.5-T1): append:**", l_t1a)
    cap_body = cap.split("Supplementary Table S16. ", 1)[1]
    cap_rows, title_line, title, si_cap = caption_fields(num["current"], cap_body + " " + app, B.sha(SI_MD))
    si_status = "written" if si_cap else "pending"
    write_caption_sheet(wb, [
        ("workspace_id", WID),
        ("final_number", f"Supplementary Table {num['current']} ({num['current_field']} of {rel(NUMBERING)}; earlier "
                         f"{num['history']})"),
        *cap_rows[:3],
        ("footnotes", f"{fn_a} {fn_b} {p2_par}"),
        ("sheets", f"{names['Sheet1']}: R3.5-T1 table (column 4 renamed per R3.5-T1-A (1)); {names['Sheet1u']}: the "
                   f"whole-patient resampling units of both populations (units.csv, verbatim); {names['Sheet2']}, "
                   f"{names['Sheet2b']}, {names['Sheet3']}: R3.5-T1-A Sheets 2, 2b, 3. Printed values are rounded as in "
                   f"the spec; unrounded values are in raw/ beside this workbook."),
        ("sources", "; ".join(f"{rel(p)} ({B.sha(p)[:12]}…)" for p in (f_sum, f_units, f_sub, f_cf, f_au, f_pl, f_tm, f_s2b))),
        ("spec", f"docs/manuscript_supplement_edits.md ### R3.5-T1 (l.{l_t1}; caption l.{l_cap}, table l.{l_hdr1}) + "
                 f"### R3.5-T1-A (l.{l_t1a}; caption append l.{l_app})"),
        cap_rows[3],
        ("built_by/date", f"scripts/supp_tables/build_S16_classifier_20261109.py, {BUILD_DATE}"),
    ])
    xlsx = OUT / f"Table_{WID}.xlsx"
    save_xlsx(wb, xlsx)
    back = openpyxl.load_workbook(xlsx)
    for k, n in names.items():
        write_sheet_csv(back[n], OUT / f"Table_{WID}__{n}.csv")
    for f in (f_sum, f_units, f_rec, f_cf, f_au, f_pl, f_tm, f_s2b):
        shutil.copyfile(f, raw / f.name)
    write_sheet_csv(sub_ws, raw / "submitted_Table_S16_Sheet1.csv")
    R.check(all((raw / f.name).read_bytes() == f.read_bytes() for f in (f_sum, f_units, f_rec, f_cf, f_au, f_pl, f_tm, f_s2b)),
            "raw/: verbatim byte copies of the 8 source files (unrounded values) plus the submitted Sheet1 as CSV")

    # ---------------------------------------------------------------- claim checks (named cells)
    ws2 = back[names["Sheet2"]]
    ws1 = back[names["Sheet1"]]
    col_sens, col_spec = hdr2[3], hdr2[4]
    is100 = lambda enc: (lambda r: r[0].startswith(enc) and r[1] == "100% (primary)")
    sens100 = cf[cf.budget_percent == 100].set_index("encoder")
    lo_s, hi_s = sens100.sens_pooled_cross_fitted.idxmin(), sens100.sens_pooled_cross_fitted.idxmax()
    lo_p, hi_p = sens100.spec_pooled_cross_fitted.idxmin(), sens100.spec_pooled_cross_fitted.idxmax()
    a_ls, v_ls = cell_addr(ws2, is100(ENC_LABEL[lo_s]), col_sens)
    a_hs, v_hs = cell_addr(ws2, is100(ENC_LABEL[hi_s]), col_sens)
    a_lp, v_lp = cell_addr(ws2, is100(ENC_LABEL[lo_p]), col_spec)
    a_hp, v_hp = cell_addr(ws2, is100(ENC_LABEL[hi_p]), col_spec)
    claim_si = "patient-held-out tumor-versus-rest sensitivity ranged from 0.943 to 0.968 and specificity from 0.667 to 0.863"
    rng_ok = (f"{sens100.sens_pooled_cross_fitted.min():.3f}", f"{sens100.sens_pooled_cross_fitted.max():.3f}",
              f"{sens100.spec_pooled_cross_fitted.min():.3f}", f"{sens100.spec_pooled_cross_fitted.max():.3f}") == \
             ("0.943", "0.968", "0.667", "0.863")
    R.check(rng_ok, f"claim '{claim_si}' (SI l.{find_line(SI_MD, 'Across the four encoders, patient-held-out')}; "
                    f"100% budget, pooled cross-fitted): sensitivity min {a_ls} = '{v_ls.split()[0]}' "
                    f"({ENC_LABEL[lo_s]}, raw {sens100.sens_pooled_cross_fitted.min():.6f}), max {a_hs} = '{v_hs.split()[0]}' "
                    f"({ENC_LABEL[hi_s]}, raw {sens100.sens_pooled_cross_fitted.max():.6f}); specificity min {a_lp} = "
                    f"'{v_lp.split()[0]}' ({ENC_LABEL[lo_p]}, raw {sens100.spec_pooled_cross_fitted.min():.6f}), max {a_hp} = "
                    f"'{v_hp.split()[0]}' ({ENC_LABEL[hi_p]}, raw {sens100.spec_pooled_cross_fitted.max():.6f}). Source: TF-2 "
                    f"`tile_metrics.csv` via writing_001 `S16_amendment_tf2_cross_fitted.csv`, not r3_5_s16")
    a_ov, v_ov = cell_addr(ws1, lambda r: r[0] == "Overall (macro)", SUBMITTED_COL)
    a_tu, v_tu = cell_addr(ws1, lambda r: r[0] == "Tumor", SUBMITTED_COL)
    R.check(f"{float(str(v_ov)):.3f}" == "0.998" and f"{float(str(v_tu).split()[0]):.3f}" == "0.998",
            f"claim 'within-slide AUROC of 0.998' (SI l.{find_line(SI_MD, 'The within-slide AUROC of 0.998')}): "
            f"{a_ov} = '{v_ov}' (Overall macro, submitted) and {a_tu} = '{v_tu}' (Tumor, submitted) both print as 0.998; "
            f"the submitted Overall cell 0.9983 is the likelier referent (the text does not name the class)")
    auc100 = au_cf.xs(100, level="budget")["auroc_tumour_vs_rest"]
    try:
        l_ms = find_line(MS, "patient-held-out tumor-versus-rest AUROC ranged from")
        ms_txt = MS.read_text(encoding="utf-8").splitlines()[l_ms - 1]
        ms_claim = ms_txt[ms_txt.index("patient-held-out tumor-versus-rest AUROC ranged from"):][:90]
    except KeyError:
        l_ms, ms_claim = None, "(sentence not found)"
    a_la, v_la = cell_addr(ws2, is100(ENC_LABEL[auc100.idxmin()]), hdr2[5])
    a_ha, v_ha = cell_addr(ws2, is100(ENC_LABEL[auc100.idxmax()]), hdr2[5])
    R.check(f"{auc100.min():.3f}" == "0.944" and f"{auc100.max():.3f}" == "0.965" and "0.944 to 0.965" in ms_claim,
            f"manuscript l.{l_ms} '{ms_claim.split(' (')[0]}': {a_la} = '{v_la}' ({ENC_LABEL[auc100.idxmin()]}) and {a_ha} = "
            f"'{v_ha}' ({ENC_LABEL[auc100.idxmax()]}); the source holds 4 decimals only, so 0.944/0.965 are rounded twice")

    # Do-not-add (R3.5-T1 and T1-A)
    s1_text = " ".join(str(v) for row in [hdr1] + sheet1 for v in row) + " " + cap + " " + fn_a + fn_b + p2_par
    R.check("patient-held-out" not in s1_text and "held-out patient" not in s1_text,
            "Do-not-add (R3.5-T1): 'patient-held-out' / 'held-out patient' absent from Sheet 1, its caption part, footnotes "
            "and P2 paragraph (the T1-A append uses it for the TF-2 heads only, as R2.1 audit F6 allows)")
    art = s1.loc["auroc_ovr_Artifacts"]
    R.check(f"{art.ci_lo:.2f}" not in sheet1[4][4] and f"{art.ci_hi:.2f}" not in sheet1[4][4],
            f"Do-not-add (R3.5-T1): the P1 Artifacts interval [{art.ci_lo:.2f}, {art.ci_hi:.2f}] present in summary.csv is not "
            f"printed (cell '{sheet1[4][4]}'); it survives only in raw/summary.csv (audit copy)")
    R.check("B = 2,000" not in s1_text and "0.925" not in s1_text, "Do-not-add (R3.5-T1): no slide-clustered B = 2,000 intervals")
    import re
    prose = " ".join([cap, app, fn_a, fn_b, p2_par])
    sent = [x for x in re.split(r"(?<=[.;])\s+", prose) if ("submitted" in x or "0.998" in x) and "held-out" in x]
    R.check(not sent and "in-sample" in hdr1[3] and "held-out" not in hdr1[3],
            "Do-not-add (R3.5-T1-A): no caption/footnote sentence pairs the submitted value with 'held-out' "
            f"({len(sent)} found{': ' + str(sent[:2]) if sent else ''}); Sheet 1 submitted column is labelled "
            f"'{hdr1[3]}'")

    # ---------------------------------------------------------------- the independent SI caption, claim by claim
    if si_cap:
        encs = sorted({ENC_LABEL[e] for e in cf.encoder})
        R.si_check("AUROC per encoder" not in si_cap,
                   "Block A '(…; AUROC per encoder)': the submitted evaluation is one classifier (UNI2 features) with "
                   f"one-versus-rest AUROC per class ({', '.join(c for c, _ in order)}, plus overall macro; "
                   f"{rel(SUBMITTED)}); there is no per-encoder submitted AUROC. Suggest 'AUROC per class'.")
        R.si_check("tiles held out from the classifier's own training slides" not in si_cap
                   or "test tiles" in hdr1[3],
                   "Block A 'tiles held out from the classifier's own training slides': consistent with the Sheet 1 "
                   f"label '{hdr1[3]}' (test tiles are the 95% not used for training, on the same slides)")
        R.si_check(encs == sorted(["UNI2", "Virchow2", "H-Optimus-1", "CONCH v1.5 (TITAN arm)"]),
                   f"Block B 'UNI2, Virchow2, H-Optimus-1 and CONCH v1.5': Sheet 2 encoders {encs}")
        nf = sorted(tm[tm.scope.str.startswith('fold')].scope.unique().tolist())
        R.si_check(len(nf) == 5, f"Block B 'five patient folds': tile_metrics folds {nf}")
        R.si_check("bootstrap intervals over patients" not in si_cap,
                   "Block B 'sensitivity and specificity pooled over folds with 95% bootstrap intervals over patients': "
                   "no interval exists for the TF-2 heads (writing_001 and tile_metrics.csv hold pooled values and the "
                   "range over the five folds; the T1-A caption says 'no interval was computed'). The only whole-patient "
                   "bootstrap intervals in this table are Sheet 1's slide-held-out UNI2 column (r3_5_s16). Sources "
                   "cannot support the claim; suggest '… with the range over the five folds'.")
        R.si_check(True, "Block B 'per-class recall': Sheet 3 'Own-class recall' (and 'Kept as tumor') for 5 classes")
        R.si_check(set(cf.budget_percent) == {5, 100},
                   "Block B 'and the same with 5% of the training labels': Sheets 2/2b/3 carry the 5% budget rows")
        R.si_check("slide-held-out" in si_cap or "slide-grouped" in si_cap,
                   "Not described by the SI caption: Sheet 1's slide-held-out column (UNI2, five slide-grouped folds, "
                   "whole-patient intervals; R3.5-T1) with its P2 sensitivity paragraph and footnotes a/b, and Sheet 2b "
                   "(cross-fitted AUROC by class, including the 0.944–0.965 cited in the manuscript). The workbook keeps "
                   "both per R3.5-T1-A (1)–(2); the lead to either describe them or decide to drop them.")

    # ---------------------------------------------------------------- open items
    R.open_items += [
        "**Brief vs source naming.** The brief calls `r3_5_s16_whole_patient_ci_20261008` the patient-held-out evaluation; "
        "its README and R3.5-T1 describe it as a *slide*-grouped out-of-fold evaluation (7 patients / 16 slides span "
        "folds) and R3.5-T1 forbids the label 'patient-held-out' for it. The patient-held-out values quoted in the SI "
        "(sensitivity 0.943–0.968, specificity 0.667–0.863) and the manuscript (AUROC 0.944–0.965) come from the TF-2 "
        "cross-fitted heads (Sheets 2/2b; `tile_metrics.csv`, the Fig. S4 panel-a source), bound here. Both evaluations "
        "are kept, as R3.5-T1-A (1)–(2) specify.",
        "**TF-2 AUROC precision.** `auroc_recount.csv` (and its `.out` log) hold 4-decimal AUROCs only; no unrounded TF-2 "
        "AUROC was retained, so raw/ carries the 4-decimal values and the 3-decimal manuscript range is a double rounding. "
        "Sensitivity/specificity are unrounded and re-derivable from the confusion counts.",
        "**No intervals for TF-2.** Sheets 2/2b are point estimates with fold ranges; no whole-patient interval exists for "
        "the TF-2 heads (as the T1-A caption states). Not computed here (no new computation).",
        "**Footnote facts not re-derivable from bound files:** footnote a '93% of the tiles come from one slide' (F_DELTA_MAP "
        "§C, 08b816ad…, not bound) and footnote b 'two contribute 3 and 7 tiles' (author Q-A) — carried from the spec.",
        "**Source errata owed (R3.5-T1, lead; unchanged here):** r3_5_s16 README says Artifacts is not estimable while P1 "
        "has an interval; P2 Artifacts `interval` field reads 'whole-patient B=5000' with blank bounds (4 non-finite "
        "replicates); the n_* columns of the sensitivity/specificity/accuracy rows hold tumour counts.",
        "**Restricted identifiers.** `Sheet1_units` (and raw/units.csv) list St. Jude patient IDs (SJ…) and slide scan "
        "numbers. Requested by the brief as part (b); keep the package controlled and decide at release whether the units "
        "sheet is published, replaced by counts, or withheld (packaging is not permission to publish).",
        "**Spelling/terminology harmonisation for the lead:** R3.5-T1 wording uses 'Tumour' and 'slide-held-out'; the T1-A "
        "append and SI title use 'tumor' and 'patient-held-out' (for TF-2). Kept verbatim; the SI Markdown entry is "
        f"currently {si_status}.",
        "**Submitted row label.** Submitted 'Overall (macro / multiclass)' prints as 'Overall (macro)' per the spec table; "
        "value 0.9983 unchanged.",
    ]

    # ---------------------------------------------------------------- outputs
    B.write(OUT, {"workspace_id": WID, "final_number": num["current"], "built": BUILD_DATE,
                  "spec_lines": {"R3.5-T1": l_t1, "R3.5-T1_table": l_hdr1, "R3.5-T1-A": l_t1a}})
    manifest_row = {"final_number": num["current"], "path": rel(xlsx)}
    rep = f"""# Workspace Table {WID} → Supplementary Table {num['current']} — build report

Script: `scripts/supp_tables/build_S16_classifier_20261109.py`; run {now_utc()}; output `{rel(OUT)}`.
Assembly only: no statistic is computed; printed cells are generated from the bound sources and compared
string-for-string with the spec tables; sensitivity/specificity are additionally re-derived from confusion counts.

## Contents

| Sheet | Content | Source |
|---|---|---|
| caption | field/text: ids, title (SI l.{title_line}), caption (R3.5-T1 + T1-A append), footnotes, sources, spec | spec |
| {names['Sheet1']} | R3.5-T1 table: submitted in-sample AUROC (renamed per T1-A) + slide-held-out [95% CI] (P1) | `summary.csv`, submitted xlsx |
| {names['Sheet1u']} | whole-patient resampling units, P1 and P2 ({len(units)} rows) | `units.csv` |
| {names['Sheet2']} | TF-2 cross-fitted (patient-held-out) tumor sensitivity/specificity + AUROC, 4 encoders × 2 budgets | writing_001 CSVs |
| {names['Sheet2b']} | TF-2 cross-fitted one-vs-rest AUROC by class | `S16_auroc_recount_229pool.csv` |
| {names['Sheet3']} | TF-2 fraction of each annotated class kept by the tumor mask | `S16_amendment_tf2_per_label.csv` |

Per-sheet CSVs `Table_{WID}__<sheet>.csv` mirror the workbook; `raw/` holds verbatim copies of the unrounded sources.

## Checks ({R.summary()})

{R.md_checks()}

## SI caption consistency (lead's caption vs this workbook; findings, not build failures)

{R.md_si(si_cap, title_line)}
## Open items

""" + "\n".join(f"- {o}" for o in R.open_items) + "\n"
    (OUT / "BUILD_REPORT.md").write_text(rep, encoding="utf-8")
    (OUT / "manifest_row.json").write_text(json.dumps({WID: manifest_row}, indent=1) + "\n")
    write_sha256sums(OUT)
    print(f"{WID}: {len(R.checks)} checks, {R.n_fail()} FAIL -> {OUT}")
    return 1 if R.n_fail() else 0


if __name__ == "__main__":
    sys.exit(main())
