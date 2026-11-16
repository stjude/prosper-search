#!/usr/bin/env python3
"""Workspace Table S7 (final S20 on 2026-11-09): top-5 plurality group (closest-rank tie rule) for the
89 CCDI-MCI neuroblastoma queries, by supplied diagnosis label. REPLACEMENT content ([internal record]
R1.2-T4); the submitted Table_S7_misclassification_matrix.csv (118 misclassified of 182) is not reused.

Source: $PROSPER_OUT/r1_2_scope_correction_20260930/main_nb_raw_diagnosis_closest_vote5_confusion.csv
(pivot of source diagnosis x pred_closest_k5, summing n), checked against the per-query file and the
Supplementary Fig. S6 panel-e source. Output: .../B_external_perturbation_finelabel/S7_S20/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _supp_common_20261109 import OUT_ROOT, REPO, Build, frac  # noqa: E402

R = OUT_ROOT / "r1_2_scope_correction_20260930"
FIG = OUT_ROOT / "figures_20261021/figS3_nb_analysis"


def main() -> None:
    b = Build("S7", "S7_S20", "scripts/supp_tables/build_S7_nb_confusion_20261109.py", "### R1.2-T4",
              "R1.2-T4 S7 bullet and 'Exact table bindings' S7 rule (pivot source diagnosis against pred_closest_k5, summing n; "
              "all 89 queries incl. the 74 assigned NB; non-NB EWS 6, RMS 5, WT 2, RB 1, NORMAL 1). No later amendment changes S7.")
    c = b.csv(R / "main_nb_raw_diagnosis_closest_vote5_confusion.csv", "S7 body: accepted NB89 diagnosis x closest-rank plurality table")
    d = b.csv(R / "decision_table_765.csv", "per-query eligibility/metadata table (765)", low_memory=False)
    q = b.csv(R / "whole_slide_ot_ffpe_scope_clear_per_query.csv", "per-query outcomes, UNI2 + OT, main population")
    fq = b.csv(FIG / "source_data_nb89_queries.csv", "Supplementary Fig. S6 panel b/e per-query source data")
    b.bind(REPO / "scripts/figures/figS3_nb_analysis.py", "Supplementary Fig. S6 builder (panel e rule)")

    b.check("no missing diagnosis or prediction", not c.source_diagnosis_missing.any() and not c.pred_closest_k5_missing.any())
    b.check("cells sum to 89 queries", c.n.sum() == 89, str(c.n.sum()))
    tot = c.groupby("pred_closest_k5").n.sum().to_dict()
    b.check("74 assigned NB; non-NB EWS 6, RMS 5, WT 2, RB 1, NORMAL 1 (15)", tot == {"NB": 74, "EWS": 6, "RMS": 5, "WT": 2, "RB": 1, "NORMAL": 1}, str(tot))
    b.check("cell hit@5 counts sum to 82; generous to 75", c.hit5_count.sum() == 82 and c.generous_vote5_count.sum() == 75)
    b.check("every NB-assigned query has hit@5", (c[c.pred_closest_k5 == "NB"].hit5_count == c[c.pred_closest_k5 == "NB"].n).all())
    nb = d[d.ffpe_scope_clear & d.tumor_group.eq("NB")].merge(q, left_on="slide_id", right_on="query", validate="one_to_one")
    rt = nb.groupby(["Diagnosis", "pred_closest_k5"]).agg(n=("query", "size"), hit5=("hit_k5", "sum"), gen=("generous_correct_k5", "sum")).reset_index()
    mm = c.merge(rt, on=["Diagnosis", "pred_closest_k5"], how="outer", indicator=True)
    b.check("re-tabulation from the per-query file reproduces all 8 cells (n, hit@5, generous)",
            (mm._merge == "both").all() and (mm.n_x == mm.n_y).all() and (mm.hit5_count == mm.hit5).all() and (mm.generous_vote5_count == mm.gen).all())
    fe = fq[fq.pred_closest_k5 != "NB"].pred_closest_k5.value_counts().to_dict()
    b.check("Supplementary Fig. S6 panel-e source gives the same 15 non-NB assignments", fe == {k: v for k, v in tot.items() if k != "NB"}, str(fe))
    ties = int(nb.tied_set_k5.astype(str).str.contains("|", regex=False).sum())
    nonnb_hit = int(c[c.pred_closest_k5 != "NB"].hit5_count.sum())
    b.check("non-NB-assigned queries with hit@5 = 82 - 74 = 8 (EWS 3, RMS 3, WT 2)", nonnb_hit == 8 and
            c[c.pred_closest_k5 != "NB"].groupby("pred_closest_k5").hit5_count.sum().to_dict() == {"EWS": 3.0, "NORMAL": 0.0, "RB": 0.0, "RMS": 3.0, "WT": 2.0})

    cols = ["NB"] + [k for k, _ in sorted(((k, v) for k, v in tot.items() if k != "NB"), key=lambda kv: (-kv[1], kv[0]))]
    rows_order = c.groupby("Diagnosis").n.sum().sort_values(ascending=False).index.tolist()
    pv = c.pivot_table(index="Diagnosis", columns="pred_closest_k5", values="n", aggfunc="sum", fill_value=0).reindex(index=rows_order, columns=cols, fill_value=0)
    hit = c.groupby("Diagnosis").hit5_count.sum().reindex(rows_order)
    pv_disp = pv.reset_index().rename(columns={"Diagnosis": "Supplied diagnosis"})
    pv_disp["Total"] = pv.sum(axis=1).values
    pv_disp["of which hit@5"] = hit.astype(int).values
    total = {"Supplied diagnosis": "Total", **{k: int(pv[k].sum()) for k in cols}, "Total": int(pv.values.sum()), "of which hit@5": int(hit.sum())}
    pv_disp = pd.concat([pv_disp, pd.DataFrame([total])], ignore_index=True)
    pv_disp = pv_disp.rename(columns={k: f"Top-5 plurality = {k}" for k in cols})

    long = c.copy()
    long["_o"] = long.Diagnosis.map({k: i for i, k in enumerate(rows_order)})
    long["_p"] = long.pred_closest_k5.map({k: i for i, k in enumerate(cols)})
    long = long.sort_values(["_o", "_p"]).drop(columns=["_o", "_p"]).reset_index(drop=True)
    cells = pd.DataFrame({
        "Supplied diagnosis": long.Diagnosis, "Top-5 plurality group (closest-rank tie rule)": long.pred_closest_k5,
        "Queries (n)": long.n.astype(int),
        "hit@5 among these queries": [frac(h, n) for h, n in zip(long.hit5_count, long.n)],
        "generous vote@5 correct among these queries [submitted rule]": [frac(g, n) for g, n in zip(long.generous_vote5_count, long.n)],
    })
    raw = long.copy()
    raw["value_source"] = "main_nb_raw_diagnosis_closest_vote5_confusion.csv (accepted, verbatim)"

    title = "Top-five plurality group for the CCDI-MCI neuroblastoma queries by supplied diagnosis label (revised FFPE, metadata-scope population)"
    caption = ("Retrieved plurality labels for revised FFPE, metadata-scope NB queries, stratified by supplied diagnostic label. "
               "Predictions use the top-five neighbours, with the nearest-ranked tied label selected.")
    foot = [
        "Population and search as in Supplementary Table S19: 89 CCDI-MCI NB queries (FFPE, metadata scope clear), UNI2 + exact Optimal Transport, fixed 2,778-slide COMET library.",
        "Prediction rule: the most frequent COMET group among the five nearest library slides; ties are resolved to the tied group with the nearest-ranked slide. 74 of 89 queries were assigned NB; the 15 others were assigned EWS (6), RMS (5), WT (2), RB (1) and NORMAL (1). Groups with no assignment are not shown as columns.",
        f"This plurality endpoint differs from hit@5 (at least one NB slide among the five nearest; 82/89): {nonnb_hit} of the 15 queries assigned a non-NB plurality group still had an NB slide in the top five (column 'of which hit@5' and sheet 'cells').",
        f"Ties: {ties} of the 89 queries had more than one group tied for the plurality at k = 5; the generous rule of the submitted analysis, which credits NB when it is among the tied groups, gives 75/89 (sheet 'cells').",
        "Supplied diagnosis labels are reported as recorded; no differentiation grade is inferred. The table describes retrieved labels, not diagnostic predictions or misclassification by NB subtype.",
        "Descriptive; no significance test. Same per-query source as Supplementary Fig. S6e.",
        "Replaces the submitted Table S7 (subtype-by-prediction matrix of 118 misclassified among 182 queries); no submitted value is carried into this table.",
    ]
    sources = [f"$PROSPER_OUT/r1_2_scope_correction_20260930/main_nb_raw_diagnosis_closest_vote5_confusion.csv (sha256 {b.inputs[str(R / 'main_nb_raw_diagnosis_closest_vote5_confusion.csv')]['sha256'][:16]}\u2026)",
               "checked against whole_slide_ot_ffpe_scope_clear_per_query.csv + decision_table_765.csv and figures_20261021/figS3_nb_analysis/source_data_nb89_queries.csv"]
    cap = b.caption_frame(title, caption, foot, sources, "v1 (2026-11-09): replacement content, revised NB89 population")
    b.write_workbook("Table_S7_NB_plurality_by_diagnosis", cap, [("matrix", pv_disp), ("cells", cells), ("cells_raw", raw)])
    b.open_items += [
        "Title: the legends numbering table lists S20 as 'Misclassification matrix for neuroblastoma subtypes in the external cohort' (submitted title). The new content is a plurality-label table by supplied diagnosis, not a subtype misclassification matrix; lead to align the title.",
    ]
    b.finish(title, [
        "Pivot of the accepted confusion file: 3 supplied diagnoses x 6 assigned groups, 89 queries; NB 74; EWS 6, RMS 5, WT 2, RB 1, NORMAL 1.",
        "Re-tabulated from the per-query file (all 8 cells equal); Fig. S6e source gives the same 15 non-NB assignments.",
        f"hit@5 82/89 shown alongside ({nonnb_hit} of the 15 non-NB plurality queries still had hit@5); {ties} of 89 queries had a tied plurality at k = 5.",
    ], cap)


if __name__ == "__main__":
    main()
