#!/usr/bin/env python3
"""Workspace Table S6 (final number from supp_table_numbering.json; S19 on 2026-11-09):
descriptive hit@5 by available metadata factor among the 89 CCDI-MCI neuroblastoma queries
of the revised FFPE, metadata-scope population. REPLACEMENT content ([internal record] R1.2-T4);
the submitted 182-query Table_S6_accuracy_by_factor.csv is not reused.

Sources ($PROSPER_OUT/r1_2_scope_correction_20260930/, accepted R1.2 package, LSF 324464955):
  main_nb_descriptive_factors.csv          the table body (bound verbatim)
  whole_slide_ot_ffpe_scope_clear_per_query.csv + decision_table_765.csv
                                           re-tabulation check of every row and the secondary
                                           closest-rank plurality column (counts only)
  COMPLETE.json, provenance.json           package identity
and the Supplementary Fig. S6 source ($PROSPER_OUT/figures_20261021/figS3_nb_analysis/) for the
one-source check. Output: $PROSPER_OUT/supp_tables_20261109/B_external_perturbation_finelabel/S6_S19/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _supp_common_20261109 import OUT_ROOT, REPO, Build, frac  # noqa: E402

R = OUT_ROOT / "r1_2_scope_correction_20260930"
FIG = OUT_ROOT / "figures_20261021/figS3_nb_analysis"
AGE_ORDER = ["<1 yr", "1-2 yr", "2-5 yr", "5-10 yr", "10-18 yr", ">=18 yr"]
AGE_LABEL = {"<1 yr": "<1", "1-2 yr": "1 to <2", "2-5 yr": "2 to <5", "5-10 yr": "5 to <10", "10-18 yr": "10 to <18", ">=18 yr": "\u226518"}
FACTOR_LABEL = {
    "Diagnosis": ("Supplied diagnosis", "CCDI 'Diagnosis'"),
    "source_site_region_heuristic": ("Site region (keyword grouping of recorded site text; not adjudicated)", "derived from CCDI 'Diagnosis Anatomic Site'"),
    "Diagnosis Anatomic Site": ("Recorded anatomic site", "CCDI 'Diagnosis Anatomic Site'"),
    "age_bucket": ("Age at diagnosis (years)", "CCDI 'Age at Diagnosis (days)' / 365.25"),
    "Sex at Birth": ("Sex at birth", "CCDI 'Sex at Birth'"),
    "Race": ("Race / ethnicity as recorded", "CCDI 'Race'"),
    "Diagnosis Category": ("Diagnosis category", "CCDI 'Diagnosis Category'"),
    "Treatment Type": ("Treatment type", "CCDI 'Treatment Type'"),
    "Last Known Survival Status": ("Last known survival status", "CCDI 'Last Known Survival Status'"),
    "prep_dicom": ("Specimen preparation", "DICOM series preparation (R1.2 preparation audit)"),
}


def main() -> None:
    b = Build("S6", "S6_S19", "scripts/supp_tables/build_S6_nb_factors_20261109.py", "### R1.2-T4",
              "R1.2-T4 S6 bullet and 'Exact table bindings' S6 rule (block as written; no later amendment changes "
              "the S6 table scope; the 2026-10-21 rendering note (edit sheet 'Rendered this pass') concerns Fig. S3 only).")
    comp = b.json(R / "COMPLETE.json", "R1.2 package completion record")
    prov = b.json(R / "provenance.json", "R1.2 package provenance")
    b.check("R1.2 package complete; main population 524 queries", comp["status"] == "complete_after_all_guards" and comp["main_queries"] == 524)
    b.check("COMPLETE.json binds provenance.json", comp["provenance_sha256"] == b.inputs[str(R / "provenance.json")]["sha256"])
    b.check("provenance: main population ffpe_scope_clear has 524 participants, 12 observed groups, NB 89",
            prov["populations"]["ffpe_scope_clear"]["n"] == 524 and len(prov["populations"]["ffpe_scope_clear"]["observed_groups"]) == 12
            and prov["populations"]["ffpe_scope_clear"]["observed_groups"]["NB"] == 89)

    f = b.csv(R / "main_nb_descriptive_factors.csv", "S6 body: accepted NB89 descriptive factor table")
    d = b.csv(R / "decision_table_765.csv", "per-query eligibility/metadata table (765)", low_memory=False)
    q = b.csv(R / "whole_slide_ot_ffpe_scope_clear_per_query.csv", "per-query outcomes, UNI2 + OT, main population")
    b.check("per-query file has the 524 main queries", len(q) == 524 and q["query"].is_unique)

    # ---- re-tabulation (distinct route for every accepted row; adds the closest-rank plurality count) ----
    sys.path.insert(0, str((REPO / "../comet-testsets/src/vertical_pipeline/search").resolve()))
    helper = b.bind(REPO / "../comet-testsets/src/vertical_pipeline/search/prediction_analysis_reasoning.py",
                    "frozen comet-testsets helper (age_bucket, extract_anatomic_region), read-only import")
    b.check("helper sha equals the R1.2 provenance pin", b.inputs[str(helper)]["sha256"] == prov["input_sha256"]["original_nb_helper"]["sha256"])
    from prediction_analysis_reasoning import age_bucket, extract_anatomic_region  # noqa: E402
    nb = d[d.ffpe_scope_clear & d.tumor_group.eq("NB")].merge(
        q[["query", "hit_k5", "generous_correct_k5", "pred_closest_k5"]], left_on="slide_id", right_on="query", validate="one_to_one")
    b.check("NB main population: 89 queries", len(nb) == 89, str(len(nb)))
    b.check("NB89: hit@5 82, no hit 7", int(nb.hit_k5.sum()) == 82 and int((nb.hit_k5 == 0).sum()) == 7, str(nb.hit_k5.sum()))
    nb["age_bucket"] = nb["Age at Diagnosis (days)"].map(age_bucket)
    nb["source_site_region_heuristic"] = nb["Diagnosis Anatomic Site"].map(extract_anatomic_region)
    nb["closest_correct"] = (nb.pred_closest_k5 == "NB").astype(int)
    re_rows = []
    for col in FACTOR_LABEL:
        src = {"age_bucket": "Age at Diagnosis (days)", "source_site_region_heuristic": "Diagnosis Anatomic Site"}.get(col, col)
        nb["_miss"] = nb[src].isna() | nb[src].eq("")
        g = nb.groupby([col, "_miss"], dropna=False).agg(n=("slide_id", "size"), hit5=("hit_k5", "sum"),
                                                         gen=("generous_correct_k5", "sum"), clo=("closest_correct", "sum")).reset_index()
        g.insert(0, "factor", col)
        re_rows.append(g.rename(columns={col: "source_value", "_miss": "source_value_missing"}))
    rt = pd.concat(re_rows, ignore_index=True)
    key = ["factor", "source_value", "source_value_missing"]
    m = f.merge(rt, on=key, how="outer", indicator=True, validate="one_to_one")
    b.check("re-tabulation covers exactly the 57 accepted rows", (m._merge == "both").all() and len(m) == len(f) == 57, str(m._merge.value_counts().to_dict()))
    b.check("re-tabulated n equals accepted n (all 57 rows)", (m.n_x == m.n_y).all())
    b.check("re-tabulated hit@5 counts equal accepted (all 57 rows)", (m.hit5_count == m.hit5).all())
    b.check("re-tabulated generous vote@5 counts equal accepted (all 57 rows)", (m.generous_vote5_count == m.gen).all())
    for fac, g in f.groupby("factor"):
        b.check(f"{fac}: rows sum to 89 queries / 82 hits / 75 generous", g.n.sum() == 89 and g.hit5_count.sum() == 82 and g.generous_vote5_count.sum() == 75,
                f"{g.n.sum()}/{g.hit5_count.sum()}/{g.generous_vote5_count.sum()}")
    b.check("closest-rank plurality correct (pred NB) = 74 of 89 in every factor block",
            all(m[m.factor == fac].clo.sum() == 74 for fac in FACTOR_LABEL))
    b.check("source diagnoses NB-NOS 86, GNB 2, GN 1",
            f[f.factor == "Diagnosis"].set_index("source_value").n.to_dict() == {"Ganglioneuroblastoma": 2, "Ganglioneuroma": 1, "Neuroblastoma, NOS": 86})
    b.check("rates equal count / n", ((f.hit5_rate - f.hit5_count / f.n).abs() < 1e-12).all() and ((f.generous_vote5_accuracy - f.generous_vote5_count / f.n).abs() < 1e-12).all())
    ages = nb["Age at Diagnosis (days)"].astype(float) / 365.25
    b.check("all 89 ages numeric; none >= 18 years (so no >=18 row)", ages.notna().all() and ages.max() < 18 and ">=18 yr" not in set(f[f.factor == 'age_bucket'].source_value),
            f"max {ages.max():.2f} y")
    miss = f[f.source_value_missing]
    b.check("explicit missingness: only Treatment Type and Last Known Survival Status have missing values (87 each)",
            set(miss.factor) == {"Treatment Type", "Last Known Survival Status"} and (miss.n == 87).all())

    # ---- eligibility flow (footnote values) ----
    nb178 = d[d.tumor_group == "NB"]
    withheld = nb178[(nb178.prep_dicom == "FFPE") & ~nb178.ffpe_scope_clear].scope_reason.value_counts().to_dict()
    b.check("flow 178 -> 103 FFPE -> 89 main", len(nb178) == 178 and int((nb178.prep_dicom == "FFPE").sum()) == 103 and int(nb178.ffpe_scope_clear.sum()) == 89)
    b.check("14 FFPE NB withheld: 4 known FOXR2, 7 site-only, 3 compound-diagnosis", withheld == {"site_only": 7, "known_foxr2": 4, "compound_support": 3}, str(withheld))
    prep = nb178.prep_dicom.value_counts(dropna=False).to_dict()
    b.check("non-FFPE NB: 71 frozen/OCT, 4 unclassified", prep == {"FFPE": 103, "FROZEN_OCT": 71, "NONE": 4}, str(prep))
    rec = b.csv(R / "nb_membership_reconciliation.csv", "submitted NB182 vs current NB178 vs main NB89 membership")
    b.check("reconciliation: 182 submitted, 178 current, 89 main; 173 common to 182 and 178",
            rec.old_nb182.sum() == 182 and rec.current_nb178.sum() == 178 and rec.main_nb89.sum() == 89 and int((rec.old_nb182 & rec.current_nb178).sum()) == 173)

    # ---- one source with Supplementary Fig. S6 ----
    fs = b.csv(FIG / "source_data_factors.csv", "Supplementary Fig. S6 (workspace Fig. S3) panel a/c/d source data")
    fq = b.csv(FIG / "source_data_nb89_queries.csv", "Supplementary Fig. S6 panel b/e per-query source data")
    b.bind(FIG / "PROVENANCE.md", "Supplementary Fig. S6 provenance")
    b.bind(REPO / "scripts/figures/figS3_nb_analysis.py", "Supplementary Fig. S6 builder (read for layout)")
    pd.testing.assert_frame_equal(f, fs, check_exact=True)
    b.check("Fig. S6 source_data_factors.csv equals the S19 body source (57 rows x 8 columns, exact)", f.shape == (57, 8) and f.equals(fs), str(f.shape))
    prov_txt = (FIG / "PROVENANCE.md").read_text(encoding="utf-8")
    b.check("Fig. S6 PROVENANCE pins the same main_nb_descriptive_factors.csv sha256", b.inputs[str(R / "main_nb_descriptive_factors.csv")]["sha256"] in prov_txt)
    b.check("Fig. S6 per-query source: 89 queries, 82 hit@5, 7 without", len(fq) == 89 and int(fq.hit_k5.sum()) == 82 and int((fq.hit_k5 == 0).sum()) == 7)
    b.check("Fig. S6 per-query hit@5 equals the per-query main file", fq.set_index("query").hit_k5.sort_index().equals(nb.set_index("query").hit_k5.astype(float).sort_index()))

    # ---- display ----
    order = {k: i for i, k in enumerate(FACTOR_LABEL)}
    m["_fo"] = m.factor.map(order)
    m["_vo"] = [AGE_ORDER.index(v) if fac == "age_bucket" else 0 for fac, v in zip(m.factor, m.source_value)]
    m = m.sort_values(["_fo", "_vo", "source_value_missing", "source_value"], kind="stable", na_position="last").reset_index(drop=True)
    disp = pd.DataFrame({
        "Factor": [FACTOR_LABEL[x][0] for x in m.factor],
        "Value as recorded": ["(not recorded)" if mi else (AGE_LABEL[v] if fac == "age_bucket" else v) for fac, v, mi in zip(m.factor, m.source_value, m.source_value_missing)],
        "Missing": ["yes" if x else "no" for x in m.source_value_missing],
        "Queries (n)": m.n_x.astype(int),
        "hit@5, hits/n (rate)": [frac(k, n) for k, n in zip(m.hit5_count, m.n_x)],
        "Top-5 plurality, closest-rank tie rule: NB/n (rate) [secondary]": [frac(k, n) for k, n in zip(m.clo, m.n_x)],
        "Top-5 vote, generous tie rule: correct/n (rate) [secondary, submitted rule]": [frac(k, n) for k, n in zip(m.generous_vote5_count, m.n_x)],
    })
    raw = pd.DataFrame({
        "factor": m.factor, "factor_label": [FACTOR_LABEL[x][0] for x in m.factor], "source_field": [FACTOR_LABEL[x][1] for x in m.factor],
        "source_value": m.source_value, "source_value_missing": m.source_value_missing, "n": m.n_x.astype(int),
        "hit5_count": m.hit5_count.astype(int), "hit5_rate": m.hit5_rate,
        "closest_plurality5_nb_count": m.clo.astype(int), "closest_plurality5_rate": m.clo / m.n_x,
        "generous_vote5_count": m.generous_vote5_count.astype(int), "generous_vote5_accuracy": m.generous_vote5_accuracy,
        "value_source": ["main_nb_descriptive_factors.csv (accepted); closest_plurality5_* re-tabulated from whole_slide_ot_ffpe_scope_clear_per_query.csv pred_closest_k5"] * len(m),
    })

    title = "Descriptive retrieval outcomes by available metadata factor among the CCDI-MCI neuroblastoma queries (revised FFPE, metadata-scope population)"
    caption = ("Descriptive retrieval outcomes by available metadata factors among the revised FFPE, metadata-scope NB queries. "
               "Outcomes and their denominators are specified separately; comparisons are descriptive.")
    foot = [
        "Population: the 89 CCDI-MCI queries in the COMET NB group within the main CCDI-MCI population (524 FFPE queries from 524 participants, 12 observed groups; verified FFPE preparation and metadata scope clear), searched with UNI2 + exact Optimal Transport against the fixed 2,778-slide COMET library. Each factor block sums to 89 queries.",
        "hit@5: at least one NB slide among the five nearest library slides (82/89, 0.921). This is the primary outcome and the outcome plotted in Supplementary Fig. S6a, c and d; both tables are built from the same source file.",
        "Secondary columns (not deployable diagnostic predictions): closest-rank plurality \u2014 the most frequent group among the five nearest slides, ties resolved to the nearest-ranked tied group (the rule of Supplementary Table S20; 74/89 assigned NB); generous vote \u2014 the submitted rule, which credits the query when NB is among the tied plurality groups (75/89).",
        "Supplied diagnosis labels are reported as recorded (Neuroblastoma, NOS 86; ganglioneuroblastoma 2; ganglioneuroma 1); no differentiation grade is inferred from 'Neuroblastoma, NOS'.",
        "Site region is a keyword grouping of the recorded site text, not a pathology-adjudicated primary site; the recorded site text is given separately.",
        "Age bins are lower-inclusive (years = days / 365.25). All 89 queries have a recorded age; none is 18 years or older (oldest 14.9 years).",
        "Treatment type and last known survival status are recorded for 2 of 89 queries; the other 87 are shown as '(not recorded)'.",
        "Eligibility flow: 178 NB queries among the 765 CCDI-MCI participants with a diagnosis in the COMET vocabulary; 103 on verified FFPE material (71 frozen/OCT and 4 of unclassified preparation not included); 89 with metadata scope clear. The 14 FFPE NB queries withheld comprise 4 known CNS neuroblastoma, FOXR2-activated, and 10 unresolved records (7 site-only, 3 compound-diagnosis), not 14 adjudicated CNS tumours.",
        "All comparisons are descriptive; no significance test was performed. The FFPE, metadata-scope population was defined after earlier results had been examined, and cells with few queries cannot support factor-level conclusions.",
        "Replaces the submitted Table S6 (182 queries, fine-label vote scoring, earlier eligibility); no submitted value is carried into this table. Of the submitted 182 NB queries, 173 are among the current 178.",
    ]
    sources = [f"$PROSPER_OUT/r1_2_scope_correction_20260930/main_nb_descriptive_factors.csv (sha256 {b.inputs[str(R / 'main_nb_descriptive_factors.csv')]['sha256'][:16]}\u2026)",
               "$PROSPER_OUT/r1_2_scope_correction_20260930/whole_slide_ot_ffpe_scope_clear_per_query.csv, decision_table_765.csv (re-tabulation; secondary closest-rank column)",
               "accepted R1.2 package LSF 324464955 (COMPLETE.json provenance 6b62801c\u2026); same source as Supplementary Fig. S6 (figures_20261021/figS3_nb_analysis)"]
    cap = b.caption_frame(title, caption, foot, sources, "v1 (2026-11-09): replacement content, revised NB89 population")
    b.write_workbook("Table_S6_NB_descriptive_factors", cap, [("factors", disp), ("factors_raw", raw)])

    b.open_items += [
        "Title: the legends numbering table still lists S19 under the submitted title 'Accuracy by clinicopathologic factor in the external neuroblastoma cohort'; the content is now hit@5 by metadata factor (not accuracy, not clinicopathologic inference). Lead to align the title in the Supplementary Information Markdown.",
        "Secondary vote columns: the spec allows the generous-vote column if named; the closest-rank plurality column was added (re-tabulated from the retained per-query file, counts only) so that S19 and S20 share one vote rule. Lead to keep both, one, or neither.",
        "Small cells: several rows have n = 1 (race categories, treatment type, survival status 'Dead'/'Alive', single anatomic sites). Small-cell disclosure review is needed before any public release of this restricted-data derivative.",
    ]
    b.finish(title, [
        f"Table body = accepted `main_nb_descriptive_factors.csv` (57 rows), every row re-tabulated from the per-query file (n, hit@5 and generous counts identical).",
        "89 queries; hit@5 82/89 (0.921), 7 without; closest-rank plurality NB 74/89; generous vote 75/89.",
        "Same source as Supplementary Fig. S6: figure `source_data_factors.csv` equal to the S19 body source (exact); figure per-query source 89 queries, 82/7.",
        "Explicit missingness (Treatment type, survival: 87 not recorded); no >=18-year row (oldest 14.9 years).",
        "No significance test; submitted Table S6 values not reused.",
    ], cap)


if __name__ == "__main__":
    main()
