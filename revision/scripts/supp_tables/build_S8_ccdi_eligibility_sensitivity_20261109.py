#!/usr/bin/env python3
"""Workspace Table S8 (final S21 on 2026-11-09): sensitivity of the CCDI-MCI retrieval estimates to
eligibility and preparation restrictions (six configurations x six populations), with the
all-preparation preparation strata and the common-12-group companion. REPLACEMENT content ([internal record]
R1.2-T4): the submitted Table_S8_statistical_tests.csv (p-values) is withdrawn and not reused.

Sources ($PROSPER_OUT/r1_2_scope_correction_20260930/): point_estimates_all_44_and_voting.csv (body),
<configuration>_<population>.json (36 per-cell reports: distinct-route check of every displayed point
and interval, population flags), provenance.json (observed groups per population),
allprep_clear_preparation_descriptive.csv (preparation strata), common_12_class_context.csv (companion).
Output: .../B_external_perturbation_finelabel/S8_S21/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _supp_common_20261109 import OUT_ROOT, Build, ci, f3  # noqa: E402

R = OUT_ROOT / "r1_2_scope_correction_20260930"
CONFIGS = [("whole_slide_ot", "UNI2 + Optimal Transport (main configuration)"),
           ("median_pooling", "UNI2 + median pooling"),
           ("sampler", "UNI2 + SAMPLER"),
           ("yottixel", "UNI2 + Yottixel"),
           ("query_tumor_ot", "UNI2 + Optimal Transport, tumour-filtered query mosaics"),
           ("both_tumor_ot", "UNI2 + Optimal Transport, tumour-filtered query and library mosaics (library 2,603 slides, NORMAL excluded)")]
POPS = [("ffpe_scope_clear", "Current main", "FFPE, metadata scope clear (main population)"),
        ("ffpe_add_site_only", "Sensitivity", "Main + 12 site-only unresolved records under provisional labels"),
        ("ffpe_add_all_unresolved", "Sensitivity", "Main + all 18 unresolved records under provisional labels"),
        ("allprep_scope_clear", "Sensitivity", "Metadata scope clear, all preparations (FFPE, frozen/OCT, mixed, unclassified)"),
        ("original_ffpe_546", "Historical context", "Earlier FFPE population (includes 4 known CNS neuroblastoma, FOXR2-activated, and 18 unresolved)"),
        ("legacy_765", "Historical context", "Earlier in-vocabulary population, all preparations (765)")]
PRIMARY = [("micro_hit_at_5", "micro hit@5"), ("macro_hit_at_5", "macro hit@5"),
           ("micro_precision_at_5", "micro precision@5"), ("macro_precision_at_5", "macro precision@5")]
JSON_KEY = {"micro_hit_at_5": "micro_hit_at_5", "macro_hit_at_5": "macro_hit_at_5",
            "micro_precision_at_5": "micro_precision_at_5", "macro_precision_at_5": "macro_precision_at_5"}
PREP = {"VERIFIED_FFPE": "verified FFPE", "FROZEN_OCT": "frozen / OCT", "MIXED": "mixed", "NONE_UNKNOWN": "unclassified"}


def main() -> None:
    b = Build("S8", "S8_S21", "scripts/supp_tables/build_S8_ccdi_eligibility_sensitivity_20261109.py", "### R1.2-T4",
              "R1.2-T4 S8 bullet and 'Exact table bindings' S8 rule (six configurations, six populations, explicit historical/current labels, "
              "n, observed groups from the population JSON, the four primary estimates with 95% intervals; historical 765/546 point-only; "
              "common_12_class_context.csv as descriptive companion; inferential table withdrawn). The all-preparation preparation strata "
              "(allprep_clear_preparation_descriptive.csv) are added as a descriptive sheet because the manuscript cites S19-S21 for the "
              "preparation comparison (MS Results 'CCDI-MCI retrieval depended on the query population analyzed').")
    pe = b.csv(R / "point_estimates_all_44_and_voting.csv", "S21 body: 36 configuration x population reports, all metrics")
    prov = b.json(R / "provenance.json", "R1.2 provenance (observed groups per population)")
    comp = b.json(R / "COMPLETE.json", "R1.2 package completion record")
    b.check("R1.2 package complete (6 configurations, 6 populations, B 5000)", comp["status"] == "complete_after_all_guards" and comp["n_configurations"] == 6 and comp["n_populations"] == 6 and comp["bootstrap_B"] == 5000)
    b.check("36 rows = 6 configurations x 6 populations", len(pe) == 36 and set(pe.configuration) == {c for c, _ in CONFIGS} and set(pe.population) == {p for p, *_ in POPS})

    disp, raw = [], []
    for pop, status, plabel in POPS:
        og = prov["populations"][pop]["observed_groups"]
        for cfg, clabel in CONFIGS:
            r = pe[(pe.configuration == cfg) & (pe.population == pop)].iloc[0]
            j = b.json(R / f"{cfg}_{pop}.json", f"per-cell report {cfg} x {pop}")
            hist = bool(j["historical_context_only"])
            b.check(f"{cfg} x {pop}: status flag consistent (historical_context_only = {hist})", hist == (status == "Historical context"))
            b.check(f"{cfg} x {pop}: n = {r.n}, participants = n, classes = observed groups", r.n == j["n_participants"] == prov["populations"][pop]["n"] and r.n_classes == len(og) == len(j["observed_classes"]))
            b.check(f"{cfg} x {pop}: query class counts equal provenance observed groups", j["query_class_counts"] == og)
            for col, _ in PRIMARY:
                b.close(f"{cfg} x {pop}: {col} point CSV = JSON", r[col], j["point_estimates"][JSON_KEY[col]], 1e-12)
                cij = j["confidence_intervals_95"]
                if hist:
                    b.check(f"{cfg} x {pop}: {col} interval not recomputed (historical)", pd.isna(r[f"{col}_ci_lo"]) and pd.isna(r[f"{col}_ci_hi"]) and col not in cij)
                else:
                    b.close(f"{cfg} x {pop}: {col} lower CSV = JSON", r[f"{col}_ci_lo"], cij[col]["lo"], 1e-12)
                    b.close(f"{cfg} x {pop}: {col} upper CSV = JSON", r[f"{col}_ci_hi"], cij[col]["hi"], 1e-12)
                    b.check(f"{cfg} x {pop}: {col} B_effective 5000", cij[col]["B_effective"] == 5000)
            row = {"Population status": status, "Population": plabel, "Configuration": clabel,
                   "Queries / participants": f"{int(r.n)} / {int(j['n_participants'])}",
                   "Observed groups (n)": int(r.n_classes),
                   "Observed groups (queries)": "; ".join(f"{k} {v}" for k, v in og.items()),
                   "Unresolved labels provisional": "yes" if j["unresolved_labels_provisional"] else "no"}
            for col, lab in PRIMARY:
                row[f"{lab} [95% CI]"] = (f"{f3(r[col])} [interval not recomputed]" if hist else ci(r[col], r[f'{col}_ci_lo'], r[f'{col}_ci_hi']))
            disp.append(row)
            raw.append({"population": pop, "population_status": status, "configuration": cfg, "n": int(r.n), "n_participants": int(j["n_participants"]),
                        "n_observed_groups": int(r.n_classes), "observed_groups": "; ".join(f"{k} {v}" for k, v in og.items()),
                        "historical_context_only": hist, "unresolved_labels_provisional": bool(j["unresolved_labels_provisional"]),
                        **{k: r[k] for c, _ in PRIMARY for k in (c, f"{c}_ci_lo", f"{c}_ci_hi")},
                        "value_source": f"point_estimates_all_44_and_voting.csv; checked = {cfg}_{pop}.json"})
    disp, raw = pd.DataFrame(disp), pd.DataFrame(raw)

    def get(cfg, pop, col):
        return pe[(pe.configuration == cfg) & (pe.population == pop)].iloc[0][col]
    # manuscript / record values
    for col, p, lo, hi in [("macro_hit_at_5", 0.707, 0.642, 0.773), ("micro_hit_at_5", 0.929, 0.910, 0.948),
                           ("macro_precision_at_5", 0.542, 0.491, 0.595), ("micro_precision_at_5", 0.738, 0.713, 0.761)]:
        v = [round(float(get("whole_slide_ot", "ffpe_scope_clear", c)), 3) for c in (col, f"{col}_ci_lo", f"{col}_ci_hi")]
        b.check(f"main OT FFPE {col} = {p} [{lo}, {hi}] (ledger R1.2 confirmed results; MS 0.707)", v == [p, lo, hi], str(v))
    v = [round(float(get("whole_slide_ot", "allprep_scope_clear", c)), 3) for c in ("macro_hit_at_5", "macro_hit_at_5_ci_lo", "macro_hit_at_5_ci_hi")]
    b.check("manuscript: all-preparation OT macro hit@5 0.570 (0.511-0.629), 735 participants", v == [0.570, 0.511, 0.629] and get("whole_slide_ot", "allprep_scope_clear", "n") == 735, str(v))
    npop = {p: prov["populations"][p]["n"] for p, *_ in POPS}
    b.check("population labels: site-only adds 12, all-unresolved adds 18, earlier FFPE 546 = 524 + 4 FOXR2 + 18 unresolved",
            npop["ffpe_add_site_only"] - 524 == 12 and npop["ffpe_add_all_unresolved"] - 524 == 18 and npop["original_ffpe_546"] == 524 + 4 + 18, str(npop))
    b.check("main population 524 participants, 12 observed groups", get("whole_slide_ot", "ffpe_scope_clear", "n") == 524 and get("whole_slide_ot", "ffpe_scope_clear", "n_classes") == 12)

    # preparation strata (all-preparation scope-clear population)
    ps = b.csv(R / "allprep_clear_preparation_descriptive.csv", "preparation strata within the 735-participant all-preparation population (descriptive)")
    b.check("preparation strata: 6 configurations x 4 strata, all descriptive_only", len(ps) == 24 and ps.descriptive_only.all())
    for cfg, _ in CONFIGS:
        s = ps[ps.configuration == cfg]
        b.check(f"{cfg}: strata sum to 735 (524 FFPE + 201 frozen/OCT + 6 mixed + 4 unclassified)", s.n.sum() == 735 and s.set_index("preparation").n.to_dict() == {"FROZEN_OCT": 201, "MIXED": 6, "NONE_UNKNOWN": 4, "VERIFIED_FFPE": 524})
        b.close(f"{cfg}: VERIFIED_FFPE stratum hit@5 = main-population micro hit@5", s[s.preparation == "VERIFIED_FFPE"].hit_at_5.iloc[0], get(cfg, "ffpe_scope_clear", "micro_hit_at_5"), 1e-12)
        b.close(f"{cfg}: VERIFIED_FFPE stratum generous vote accuracy = main-population generous_accuracy_k5", s[s.preparation == "VERIFIED_FFPE"].generous_vote_accuracy_k5.iloc[0], get(cfg, "ffpe_scope_clear", "generous_accuracy_k5"), 1e-12)
        wsum = (s.n * s.hit_at_5).sum() / s.n.sum()
        b.close(f"{cfg}: n-weighted strata hit@5 = all-preparation micro hit@5", wsum, get(cfg, "allprep_scope_clear", "micro_hit_at_5"), 1e-12)
    cmap = dict(CONFIGS)
    ps = ps.assign(_c=ps.configuration.map({c: i for i, (c, _) in enumerate(CONFIGS)}), _p=ps.preparation.map({k: i for i, k in enumerate(PREP)})).sort_values(["_c", "_p"]).drop(columns=["_c", "_p"]).reset_index(drop=True)
    ps_disp = pd.DataFrame({"Configuration": ps.configuration.map(cmap), "Preparation stratum": ps.preparation.map(PREP), "Queries (n)": ps.n.astype(int),
                            "Observed groups (n)": ps.n_classes.astype(int), "micro hit@5": [f3(x) for x in ps.hit_at_5],
                            "generous vote@5 accuracy (micro) [secondary, submitted rule]": [f3(x) for x in ps.generous_vote_accuracy_k5]})
    ps_raw = ps.assign(value_source="allprep_clear_preparation_descriptive.csv (accepted, verbatim)")

    # common 12-group companion
    cc = b.csv(R / "common_12_class_context.csv", "common-12-group descriptive companion (no intervals)")
    b.check("common-12 companion: 36 rows, descriptive_only, the 12 main-population groups", len(cc) == 36 and cc.descriptive_only.all()
            and set(cc.common_classes) == {"|".join(sorted(prov["populations"]["ffpe_scope_clear"]["observed_groups"]))})
    r0 = cc[(cc.configuration == "whole_slide_ot") & (cc.population == "ffpe_scope_clear")].iloc[0]
    b.check("common-12 companion equals the main table on the main population (no non-common queries)", r0.n_excluded_noncommon == 0
            and all(abs(r0[c] - get("whole_slide_ot", "ffpe_scope_clear", c)) < 1e-12 for c, _ in PRIMARY))
    cc = cc.assign(_c=cc.configuration.map({c: i for i, (c, _) in enumerate(CONFIGS)}), _p=cc.population.map({p: i for i, (p, *_) in enumerate(POPS)})).sort_values(["_p", "_c"]).drop(columns=["_c", "_p"]).reset_index(drop=True)
    plab = {p: (s, l) for p, s, l in POPS}
    cc_disp = pd.DataFrame({"Population status": cc.population.map(lambda p: plab[p][0]), "Population": cc.population.map(lambda p: plab[p][1]),
                            "Configuration": cc.configuration.map(cmap), "Queries in the 12 common groups": cc.n_common_class.astype(int),
                            "Queries outside the 12 groups (excluded here)": cc.n_excluded_noncommon.astype(int),
                            **{lab: [f3(x) for x in cc[c]] for c, lab in PRIMARY}})
    cc_raw = cc.assign(value_source="common_12_class_context.csv (accepted, verbatim)")

    title = "Sensitivity of the CCDI-MCI retrieval estimates to eligibility and preparation restrictions"
    caption = ("Sensitivity of corrected CCDI retrieval estimates to unresolved eligibility and preparation restrictions. "
               "Reintroduced unresolved labels are provisional assumptions, not adjudicated ground truth. "
               "Class support and common-class comparisons are reported explicitly.")
    foot = [
        "Six UNI2 retrieval configurations evaluated on six CCDI-MCI query populations against the fixed COMET library (2,778 slides; 2,603 for the tumour-filtered library configuration, which has no NORMAL slides). One query per participant.",
        "Populations: the current main population (524 FFPE queries, 12 observed groups); sensitivities restoring the 12 site-only (536) or all 18 unresolved records (542) under provisional labels, and the all-preparation scope-clear population (735; FFPE 524, frozen/OCT 201, mixed 6, unclassified 4). The new sensitivities exclude the 4 known CNS neuroblastoma, FOXR2-activated queries. Historical populations (earlier FFPE 546; earlier 765 all preparations) are shown for context only, as point estimates; their intervals were not recomputed.",
        "hit@5: at least one same-group library slide among the five nearest; precision@5: share of the five nearest slides in the query's group. Micro averages pool queries; macro averages weight each observed query group equally, so populations with different observed groups have different macro denominators. NF is not estimable in populations without NF queries.",
        "Intervals: 95% percentile intervals from 5,000 class-stratified patient-bootstrap resamples (seed 123; one slide per drawn participant; fixed library). Intervals for different populations are not paired comparisons and do not test a difference between populations.",
        "Sheet 'preparation_strata': micro hit@5 and the submitted generous vote@5 accuracy within each preparation stratum of the all-preparation population; descriptive only, no intervals. Strata differ in case mix (for example, 10 observed groups among frozen/OCT queries); they do not establish what the same case would achieve under another preparation.",
        "Sheet 'common_12_groups': the same estimates restricted to queries in the 12 groups observed in the main population, held constant across populations; descriptive only, without intervals or significance claims.",
        "The main CCDI-MCI population was restricted to FFPE material after examination of earlier results; the restriction changes the evaluated case mix.",
        "The submitted Table S8 (statistical tests of clinicopathologic factors: Mann\u2013Whitney U for continuous age, chi-squared tests for age group, anatomic region, subtype, treatment, diagnosis category, race, sex and survival) is withdrawn; no p-value is carried into this table and no new test was performed.",
    ]
    sources = [f"$PROSPER_OUT/r1_2_scope_correction_20260930/point_estimates_all_44_and_voting.csv (sha256 {b.inputs[str(R / 'point_estimates_all_44_and_voting.csv')]['sha256'][:16]}\u2026), checked against the 36 per-cell JSON reports",
               "$PROSPER_OUT/r1_2_scope_correction_20260930/provenance.json (observed groups), allprep_clear_preparation_descriptive.csv, common_12_class_context.csv",
               "accepted R1.2 package LSF 324464955 (COMPLETE.json provenance 6b62801c\u2026)"]
    cap = b.caption_frame(title, caption, foot, sources, "v1 (2026-11-09): replacement content (submitted inferential table withdrawn)")
    b.write_workbook("Table_S8_CCDI_eligibility_sensitivity", cap,
                     [("eligibility_sensitivity", disp), ("eligibility_sensitivity_raw", raw), ("preparation_strata", ps_disp), ("preparation_strata_raw", ps_raw),
                      ("common_12_groups", cc_disp), ("common_12_groups_raw", cc_raw)])
    b.open_items += [
        "Legends conflict (record, not changed here): `docs/prosper_revised/prosper_supplementary_legends.md` 'historical NB analysis' paragraph says 'Supplementary Table S21 associates p = 0.0026 with continuous age (Mann\u2013Whitney U) ... chi-squared p was 0.033'. The new S21 withdraws the inferential table and contains no p-value, so that sentence should cite the submitted Table S8 (historical) rather than Supplementary Table S21. Lead to edit the legends Markdown.",
        "Title: the legends numbering table lists S21 as 'Statistical tests of clinicopathologic factors in the external neuroblastoma cohort' (submitted title), which no longer describes the content. Lead to align.",
        "Spec vs manuscript framing: no conflict found on inference \u2014 the spec (R1.2-T4: 'Do not carry over old p-values'; new inference only via a separately agreed R3.2 analysis) and the manuscript (descriptive; populations differ in case mix; restriction made after earlier results) both exclude significance tests, so none was added. The preparation-strata sheet goes beyond the spec's named S8 binding (it is the accepted descriptive file the manuscript's preparation sentence relies on); lead may drop it.",
        "Spec names the output `Table_S8_CCDI_eligibility_sensitivity.csv`; this build provides it as `Table_S8_CCDI_eligibility_sensitivity__eligibility_sensitivity.csv` (and the _raw CSV) beside the workbook.",
    ]
    b.finish(title, [
        "36 configuration x population rows; every displayed point and interval equals its per-cell JSON report (distinct file route, |diff| <= 1e-12).",
        "Main OT FFPE (524, 12 groups): macro hit@5 0.707 [0.642, 0.773]; micro hit@5 0.929 [0.910, 0.948]; macro precision@5 0.542 [0.491, 0.595]; micro precision@5 0.738 [0.713, 0.761].",
        "All-preparation OT (735): macro hit@5 0.570 [0.511, 0.629] (manuscript value).",
        "Historical 765/546 rows point-only (intervals not recomputed); preparation strata and common-12 companion descriptive; no p-values.",
    ], cap)


if __name__ == "__main__":
    main()
