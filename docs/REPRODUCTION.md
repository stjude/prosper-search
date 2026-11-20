# How the code maps to the paper

This file links each display item in the revised manuscript (main Figs. 1–8,
Supplementary Figs. S1–S16 and Supplementary Tables S1–S49 in Supplementary
Data 1–6) to the code in this repository that produced it. For each item it
lists the inputs and how far the item can be reproduced.

## Conventions

**Paths.** Paths are relative to the repository root. `prosper/` is the
package. `revision/scripts/` holds the analyses, `revision/scripts/figures/`
the figure builders and `revision/scripts/supp_tables/` the table builders.
`revision/retained_executed/` holds code versions that produced a result but
were later edited. The revision scripts were run from a workspace whose root
corresponds to `revision/` here; see [PATHS.md](PATHS.md).

**File names.** Analysis files keep the internal work-package prefixes used
during the revision (`r1_`–`r4_`, `e1_`, `c3_`, `s5b_`, `s6_`, `s16_`, `st2_`).
They are only identifiers that match the execution records and have no meaning
for using the code. Dates in names such as `_20261109` give the build date of
a table builder.

**Inputs.**
- *COMET-R*: COMET/STV images, mosaics, tile features, labels and case
  metadata. Restricted; see [DATA_ACCESS.md](DATA_ACCESS.md).
- *EXT*: KidRare or CCDI-MCI. Third-party access conditions apply.
- *RS*: reader-study frozen export. Restricted, not released. The
  reader-study numbers in the manuscript correspond to the final frozen
  export of the study.
- *FID*: the local figure-identifier file `revision/config/figure_ids.local.json`
  (scan-to-slide identifiers shown in figure labels; format in
  `revision/config/README.md`). Not released. The shared figure helper loads
  it; without it the builders fall back to the shipped example with
  synthetic slide identifiers, so any slide-identifier label or check in a
  figure uses synthetic values.
- *LLM*: retained LLM payloads and responses. Case-level, not released.
- *INT*: intermediate run outputs under `$PROSPER_OUT` computed from the
  above, such as similarity matrices, saved ranked lists and per-query
  tables. Not released.
- *SUB*: the submitted manuscript's figure or table files, which are part of
  the journal record and are not redistributed here.

**Reproduction levels.**
- **R**: runnable with restricted data. The code in this repository produced
  the result, and it reruns given authorized access to the listed inputs and
  their intermediate outputs.
- **R\***: as R, but the display builder also reads an unreleased document
  (see the Notes column and [KNOWN_GAPS.md](KNOWN_GAPS.md)). The numbers are
  computed by released analysis code; the final rendering step cannot be run
  from this repository as released.
- **N**: not reproducible from this repository (reason given).
- **S**: static content carried over from the submitted manuscript; no
  computation.

No item is listed as runnable from released aggregates alone. The builders
read intermediate run outputs, not the published Supplementary Data
workbooks.

## Main figures

| Display | Analysis code (numbers) | Display builder | Inputs | Level | Notes |
|---|---|---|---|---|---|
| Fig. 1 Overview | — | `revision/scripts/figures/fig1_revise.py`, then `fig1_swap_bar_labels.py` | SUB (submitted Figure 1 image) | N | Edits the submitted Figure 1 image (label and bar-label changes). The submitted image and the internal source that both scripts read are not in the repository. Both files were renamed for the release; the executed names are in CODE_MANIFEST.tsv. |
| Fig. 2 Encoder and aggregation benchmark | `prosper/evaluation/` (`slide_eval.py`, `benchmark.py`, `metrics_cli.py`, `resample.py`), `prosper/embeddings/aggregate_cli.py`, `revision/scripts/r2_1_main_grid_plurality.py`, `r2_1_fig2_retrieval_metrics.py`, `r2_1_virchow2_fp32_*.py`, comparators `r3_1_comparators_whole_patient.py`, `r3_1_prism_*.py`, `r2_6_titan_*.py`, `prosper/baselines/sish_cli.py` | `revision/scripts/figures/fig2_benchmark.py` | COMET-R, INT; FID | R | Most grid similarity matrices were computed by the earlier COMET pipelines (see KNOWN_GAPS). |
| Fig. 3 Tumor filtering | `revision/scripts/r2_1_tf2_s0_pool_folds.py` … `r2_1_tf2_s5_jobspec.py`, `r2_1_tf2_titan_*.py`, `prosper/evaluation/r2_1_tf2_eval_cli.py`, `r2_1_filter_*_cli.py`, `revision/scripts/r2_1_filter_rer.py`, `r2_1_filter_rer_routeb.py`, `r3_5_fig3a_oof_panels.py`, `figures/derive_fig3_rer_precision.py`, `figures/check_fig3_panelC_groups.py` | `revision/scripts/figures/fig3_tumor_filtering.py` | COMET-R, INT; FID | R | The example patient label is read from the local figure-identifier file. |
| Fig. 4 Additional collections | `prosper/evaluation/external_cli.py`, `revision/scripts/r1_3_external_correction.py`, `r1_3_external_synthesis.py`, `r1_3_kidrare_filter_full.py`, `r1_3_kidrare_filter_extension.py`, `c3_ccdi_specimen_preparation.py`, `r3_2_s30_chance_baseline.py` | `revision/scripts/figures/fig4_additional_collections.py` | COMET-R, EXT, INT; FID | R | |
| Fig. 5 Optimal Transport correspondence | `prosper/embeddings/slide_embedding/ot_embedding.py`, `prosper/visualization/ot_visualization.py`, `revision/scripts/r2_5_fig5_submitted_recipe_composite.py`, `r2_5_fig5_he_overlays.py`, `r2_5_fig5_tissue_grid.py`, `r2_5_fig5_final_layout.py`, `r3_5_l1_maps.py`, `r3_5_l2_alignment.py`, `r3_5_tumour_correspondence_map.py` | `figures/fig5_ot_correspondence_absolute_candidate.py` → `fig5_ot_correspondence_option2_candidate.py` (released), which import `fig5_ot_correspondence.py` | COMET-R, INT; FID | R | The expected slide identifiers of the shown pairs are read from the local figure-identifier file. |
| Fig. 6 ROI search | `revision/scripts/roi_search_6cfg/`, `prosper/evaluation/roi_cli.py`, `roi_eval.py`, `prosper/search/roi_search.py`, `revision/scripts/r4a_c6_roi_per_group_whole_patient.py` | `revision/scripts/figures/fig6_roi_search.py` (panel A; default `--b-panel submitted`) | COMET-R, INT; FID; SUB for panel B | R | Panel B is the submitted panel. |
| Fig. 7 Case studies | `prosper/bayesian/disease_probs.py`, `revision/scripts/e1_bayes_heldout_primary.py`, `revision/scripts/fig7_roi_same_patient_readback.py` | — | COMET-R, INT; SUB | S | Submitted graphic, unchanged except that the delivered file is flattened onto a white background (no computation; the flattening step is not in the repository). The read-back script checks the case-study values against the retained outputs. |
| Fig. 8 Reader study | `revision/scripts/reader_study/` (`build_readings.py`, `derive.py`, `endpoints.py`, `paired.py`, `qc_sensitivity.py`, `analyze.py`), `revision/scripts/ilp_representatives/` and `slide_selector/` (case selection); panel B band and dotted line (retrieved labels without a reader, exploratory): `revision/scripts/reader_study/search_vs_reader.py` | `revision/scripts/figures/fig8_reader_study.py`, run with `--panels none --b-variants rows --e-variants v3 --compose --compose-supp --search-alone <output directory of search_vs_reader.py>` | RS; FID; reader configuration (see `revision/config/README.md`) | R | Reader logins are read from a local, git-ignored configuration. The delivered source-data files omit the `reader_id` column and the delivered provenance note omits the logins; that filtering was done by a wrapper that is not in the repository (the figure script itself writes both). |

## Supplementary figures

| Display | Analysis code (numbers) | Display builder | Inputs | Level | Notes |
|---|---|---|---|---|---|
| S1 Hit@k across neighborhood sizes | `revision/scripts/r4b_s1s2_neighbor_ladder.py` | `figures/figS1_S2_neighbor_ladder.py` | COMET-R, INT; FID | R | |
| S2 Neighbor diversity versus hit@k | `r4b_s1s2_neighbor_ladder.py` | `figures/figS1_S2_neighbor_ladder.py` | COMET-R, INT; FID | R | |
| S3 Subtype-level retrieval | `revision/scripts/s6_subtype_scoring.py`, `r3_7_fine_label_reevaluation.py`, `retained_executed/r3_7_all_tissue_ci_20261103/` | `figures/figS_subtype_retrieval.py` | COMET-R, INT; FID | R* | Also reads an unreleased internal document. |
| S4 Tumor calls on normal-tissue controls | `prosper/preprocessing/tumor_tiles_cli.py` | `revision/scripts/r2_4_supp_normal_fp_figure.py` | COMET-R, INT; FID | R | |
| S5 Tile classifier and filtering variants | `revision/scripts/s16_grouped_tile_classifier.py`, `r2_1_tf2_s2_heads.py`, `r2_6_conch_tumor_heads.py` | `figures/figS_r21_f4_classifier_budget.py`, then `figures/figS_lean_048.py letters --fig S5` | COMET-R, INT; FID | R | The final step re-renders the same values with upper-case panel letters. |
| S6 Retrieval under simulated perturbations | `prosper/evaluation/r2_1_perturb_*.py`, `r2_1_hoptimus_perturb*.py`; executed stage versions in `revision/retained_executed/r2_1_perturb_suite_20261005/`; `revision/scripts/r2_1_fig4a_mmv5.py` | `figures/figS_perturbation_candidates.py` (`render_forest(band=False)`, with `figS_perturbation.py`), then `figures/figS6_plurality_relabel.py --variant plurality` | COMET-R, INT; FID | R | The final step re-renders the same values with the corrected column header (plurality-vote accuracy). |
| S7 CCDI-MCI neuroblastoma by metadata factor | `revision/scripts/r1_2_scope_correction.py`, `s6_5_build_kidrare_nb_block.py` | `figures/figS3_nb_analysis.py`, then `figures/figS_lean_048.py letters --fig S7` | EXT, INT; FID | R | The final step re-renders the same values with upper-case panel letters. |
| S8 Effects of predicted tumor filtering in additional collections | `r1_3_kidrare_filter_full.py`, `r1_3_kidrare_filter_extension.py`, `r1_3_external_synthesis.py` | `figures/figS5_filtering_effects.py`, then `figures/figS_lean_048.py s8` | COMET-R, EXT, INT; FID | R | The final step keeps panels A and B only. |
| S9 Failure structure in additional collections | `r1_3_external_synthesis.py`, `r2_3_internal_failure_*.py` | `figures/figS6_S7_external_failure.py` | COMET-R, EXT, INT; FID | R* | Also reads an unreleased internal document. |
| S10 WT and preparation strata | `c3_ccdi_specimen_preparation.py`, `ccdi_frozen_hub_tabulation.py` | `figures/figS6_S7_external_failure.py` | COMET-R, EXT, INT; FID | R* | as S9 |
| S11 Transport-linked tissue clusters | `revision/scripts/r3_5_l1_maps.py` | `figures/figS_r35_l1_sources_rerender.py`, then the assembly `figS_r35_l1_l2_assembly.py`, then `figures/figS_lean_048.py s11` | COMET-R, INT; FID | R | Display slide identifiers are read from the local figure-identifier file. The final step keeps the three annotated same-group pairs, re-laid with larger captions; it reads the slide images to verify the copied tiles. |
| S12 Localization by matched transport-cost difference | `r3_5_l2_alignment.py`, `r3_5_l2_diag_f4.py`, `retained_executed/script_snapshots/r3_5_l2_alignment_v3_28b781b6.py` | `figS_r35_l2_sources_rerender.py` and `figS_r35_l1_l2_assembly.py`, then `figures/figS_lean_048.py s12` | COMET-R, INT; FID | R | As S11. The final step changes only the colour-bar label. |
| S13 Randomly sampled linked tile pairs | `revision/scripts/r3_5_fig5_ot_tile_montage.py` | `figures/figS13_tile_pairs_composite.py` | COMET-R, INT; FID | R* | Also reads an unreleased internal document. |
| S14 Reader study, secondary displays | `revision/scripts/reader_study/` | `figures/fig8_reader_study.py` (same run as Fig. 8, `--compose-supp`) | RS; FID | R | |
| S15 Tumor-group counts in COMET | — | — | SUB | S | Submitted Supplementary Fig. S4, unchanged |
| S16 Reader-study instrument | — | `figures/figS_instrument_screenshots.py` | instrument screenshots (practice case; not released) | R | Layout of screenshots, no computation; the screenshots are not released. |

## Supplementary tables (Supplementary Data 1–6)

Each table was first written as a per-table workbook by the builder listed
below. `revision/scripts/supp_tables/assemble_final_tables_20261110.py`
assembled those into the final per-table workbooks. The delivered workbooks
then went through text-only post-assembly edits that are **not** in this
repository (provenance labels, headers and captions, and the removal of
sheets that the Supplementary Information does not cite); see
[KNOWN_GAPS.md](KNOWN_GAPS.md). The reader-study tables (S31, S32, S39–S47)
were rebuilt for the final export by
`revision/scripts/supp_tables/build_reader_comment_coding.py` (coded
comments, S47), `build_reader_study_tables.py` (raw tables) and
`finalize_reader_study_tables.py` (presentation rules and the S31 case
composition). S49 was written by
`revision/retained_executed/collab_draft_20261102/checks_050/reproduction/make_s49_pass050.py`.
The six delivered Supplementary Data files were grouped by
`revision/retained_executed/collab_draft_20261102/checks_051/reproduction/build_supplementary_data_workbooks_051.py`
with the layout `revision/retained_executed/collab_draft_20261102/checks_050/rev3_lean_si_20261116/supplementary_data_layout_050_lean.json`;
these two scripts and their helpers under
`revision/retained_executed/e1_si_close_audit_20261115/scratch/si-tables/`
are kept under their executed names and output-folder layout. The earlier
grouping step, `revision/scripts/supp_tables/build_supplementary_data_workbooks.py`
with `supplementary_data_layout.json` (47 tables), is also released. All
builders read two unreleased documents, the Supplementary Information legend
source and an internal table specification, for captions and specification
checks. So every table builder is **R\*** at best. Each builder writes an
`input_bindings.json` listing every input it read, with hashes.

| Table | SD | Analysis code (numbers) | Table builder (`revision/scripts/supp_tables/`) | Inputs | Level |
|---|---|---|---|---|---|
| S1 Tile encoders | 1 | — | submitted Table S1 with text updates in `assemble_final_tables_20261110.py` | SUB | S |
| S2 Slide similarity strategies | 1 | — | submitted Table S2, as above | SUB | S |
| S3 Mosaic versus all eligible tiles | 1 | `revision/scripts/r2_2_all_eligible/` (stages A–E), `r2_2_tile_ladder_whole_patient.py`, `r2_2_ot30_resolve.py`, `r2_2_ot_combined_top10.py` | `build_A_ladder_failure_finelabel_20261022.py` | COMET-R, INT | R* |
| S4 Search metrics for every encoder and aggregation | 1 | `prosper/evaluation/`, `r2_1_main_grid_plurality.py`, `r2_1_fig2_retrieval_metrics.py`, `r2_1_virchow2_fp32_*.py`, `r4a_vii_whole_patient.py` (executed version in `retained_executed/provenance_retention_20261110/`), `r2_5_c4_whole_patient_ci.py` | `build_S-R2.1-E_merged_20261109.py`, `build_C_encoders_comparators_20261022.py` | COMET-R, INT | R* |
| S5 Comparator systems and module comparisons | 1 | `r3_1_comparators_whole_patient.py`, `r3_1_prism_*.py`, `r3_1_retccl_whole_patient.py`, `prosper/baselines/sish_cli.py`, `prosper/evaluation/module_ladder.py` | `build_C_encoders_comparators_20261022.py` | COMET-R, INT | R* |
| S6 Per-group retrieval and failure summary | 1 | `r2_3_internal_failure_cis.py`, `r2_3_internal_failure_cis_ext.py`, `r2_3_internal_failure_tables.py`, `r2_3_s1b_pergroup_metastasis.py` | `build_A_ladder_failure_finelabel_20261022.py` | COMET-R, INT | R* |
| S7 Plurality, strict-majority and top-1 confusion | 1 | as S6 | `build_A_ladder_failure_finelabel_20261022.py` | COMET-R, INT | R* |
| S8 Standard retrieval metrics | 1 | `prosper/evaluation/metrics_cli.py`, `metrics.py`, `revision/scripts/roi_search_6cfg/` | `build_C_encoders_comparators_20261022.py`, `build_S25_roi6_tcm_addendum_20261112.py` | COMET-R, INT | R* |
| S9 Subtype-level retrieval | 1 | `s6_subtype_scoring.py`, `r3_7_fine_label_reevaluation.py`, `retained_executed/r3_7_all_tissue_ci_20261103/` | `build_S26_v2_all_tissue_20261103.py` | COMET-R, INT | R* |
| S10 Fine-label task inventory | 1 | `r3_7_task_inventory.py`, `s6_0_label_inventory.py` | `build_S28_v2_finelabel_inventory_20261109.py` | COMET-R, INT | R* |
| S35 Virchow2 recomputation | 1 | `r2_1_virchow2_fp32_compare.py` … `_submit.py`, `prosper/evaluation/r2_1_virchow2_fp32_extract_cli.py`, `r3_1_virchow_identity_check.py` | `build_C_encoders_comparators_20261022.py` | COMET-R, INT | R* |
| S36 Same-patient slides excluded or left in | 1 | `prosper/evaluation/slide_eval.py`, `r4a_vii_whole_patient.py` | `build_B_external_leakage_titan_20261022.py` | COMET-R, INT | R* |
| S37 Retrieval consistency across a patient's slides | 1 | `prosper/evaluation/patient_consistency.py` | `build_B_external_leakage_titan_20261022.py` | COMET-R, INT | R* |
| S11 Tile-classifier performance | 2 | `s16_grouped_tile_classifier.py`, `r2_1_tf2_s2_heads.py`, `prosper/evaluation/r2_1_hoptimus_head*.py`, `r2_1_encoder_head_*.py` | `build_S16_classifier_20261109.py` | COMET-R, INT | R* |
| S12 Relative error reduction under held-out masks | 2 | `r2_1_filter_rer.py`, `r2_1_filter_rer_routeb.py`, `r2_1_tf2_*.py`, `retained_executed/r2_1_filter_tf2_20261016/writing_001/make_tables.py` | `build_C_encoders_comparators_20261022.py` | COMET-R, INT | R* |
| S13 Tumor-only filtering with held-out masks | 2 | as S12, `prosper/evaluation/r2_1_tf2_eval_cli.py` | `build_C_encoders_comparators_20261022.py` | COMET-R, INT | R* |
| S14 Perturbation suite | 2 | `prosper/evaluation/r2_1_perturb_*.py`, `r2_1_hoptimus_perturb*.py`, `revision/retained_executed/r2_1_perturb_suite_20261005/`, `r2_1_fig4a_mmv5.py` | `build_SR21C_perturbation_suite_20261109.py` | COMET-R, INT | R* |
| S34 TITAN on all-tissue and tumor-filtered embeddings | 2 | `r2_6_titan_arm1_regen.py`, `r2_6_titan_arm2_l1_matrix.py`, `r2_6_titan_common_pool.py`, `r2_6_titan_direct1024_subset.py`, `r2_6_titan_patch_labels_1024.py` | `build_B_external_leakage_titan_20261022.py` | COMET-R, INT | R* |
| S38 TITAN tumor-only, queries without annotations | 2 | `r2_6_titan_unannotated_query_sensitivity.py` | `build_B_external_leakage_titan_20261022.py` | COMET-R, INT | R* |
| S15 Cohort summary and accounting | 3 | `revision/scripts/build_table1_cohort_accounting_20261022.py`, `r3_3_cohort_accounting.py`, `prosper/preprocessing/stage.py`, `scripts/stage_{stv,kidrare,ccdi}.py` | `build_S14_cohort_accounting_20261109.py`, `finalize_A_packaging_20261109.py` | COMET-R, EXT, INT | R* |
| S16 Additional collections, per group | 3 | `prosper/evaluation/external_cli.py`, `r1_3_external_correction.py`, `r1_3_external_synthesis.py` | `build_B_external_leakage_titan_20261022.py` | COMET-R, EXT, INT | R* |
| S17 Per-group chance baseline | 3 | `r3_2_s30_chance_baseline.py` | `build_A_ladder_failure_finelabel_20261022.py` | INT | R* |
| S18 CCDI-MCI neuroblastoma by metadata factor | 3 | `r1_2_scope_correction.py` | `build_S6_nb_factors_20261109.py` | EXT, INT | R* |
| S19 CCDI-MCI neuroblastoma plurality group | 3 | `r1_2_scope_correction.py` | `build_S7_nb_confusion_20261109.py` | EXT, INT | R* |
| S20 CCDI-MCI eligibility sensitivity | 3 | `r1_3_external_correction.py`, `c3_ccdi_specimen_preparation.py` | `build_S8_ccdi_eligibility_sensitivity_20261109.py` | EXT, INT | R* |
| S21 Additional collections under predicted tumor filtering | 3 | `r1_3_kidrare_filter_full.py`, `r1_3_kidrare_filter_extension.py` | `build_B_external_leakage_titan_20261022.py` | COMET-R, EXT, INT | R* |
| S22 Additional collections, failure neighborhoods | 3 | `r1_3_external_synthesis.py` | `build_B_external_leakage_titan_20261022.py` | COMET-R, EXT, INT | R* |
| S23 Additional collections, descriptive strata | 3 | `c3_ccdi_specimen_preparation.py`, `ccdi_frozen_hub_tabulation.py` | `build_B_external_leakage_titan_20261022.py` | EXT, INT | R* |
| S33 CCDI and MCI diagnosis mapping | 3 | `r3_3_cohort_accounting.py` | `build_S15_ccdi_mapping_20261109.py` | EXT | R* |
| S24 OT distance by annotation class | 4 | `r3_5_l2_alignment.py`, `r3_5_l2_diag_f4.py`, `retained_executed/script_snapshots/r3_5_l2_alignment_v3_28b781b6.py` | `build_S-R3.5-L2_20261109.py`, `c_localization_common_20261109.py` | COMET-R, INT | R* |
| S25 Spatial interpretation: parameter studies, transport cost, ROI evaluation | 4 | `r3_5_l1_maps.py`, `r3_5_l3_pair_vs_average.py`, `r3_5_l4_link_cost_concordance.py`, `r3_5_l5_all_links.py`, `r3_5_h2_roi_class_specificity.py`, `r3_5_cosine_*.py`, `prosper/evaluation/localization_cli.py`, `revision/scripts/roi_search_6cfg/` | `build_S-R3.5-L1_20261109.py`, `build_S25_roi6_tcm_addendum_20261112.py`, `build_C_localization_index_20261109.py` | COMET-R, INT | R* |
| S26 Bayesian probability model across k | 5 | `prosper/bayesian/disease_probs.py`, `revision/scripts/e1_bayes_heldout_primary.py`, `e1_bayes_heldout_filter_sensitivity.py`, `prosper/evaluation/calibration_cli.py`, `retained_executed/knn_comparator_recalc_20261110/` | submitted Table S9 updated by `assemble_final_tables_20261110.py`; `audit_S26_v2_20261109.py` | COMET-R, INT | R* |
| S27–S30 Top-5/top-18 neighbor labels and probability matrices | 5 | `prosper/bayesian/disease_probs.py`, `e1_bayes_heldout_primary.py` | submitted Tables S10–S13 with additions in `assemble_final_tables_20261110.py` | COMET-R, INT | R* |
| S48 LLM summary checks | 5 | `revision/scripts/llm_pilot_round2/` (pilot generation harness `llm_grounding_cli.executed_fc88a2e7.py`, `run_round2.py`, `analyze_r2.py`), `revision/scripts/llm_reader80/`, `prosper/evaluation/st1_llm_summaries_cli.py` | `build_llm_pilot_r2_tables.py`; reader-summary rating sheets `reader80_sheets.py` | LLM, INT | R* (see note) |
| S31 Reader and case flow | 6 | `revision/scripts/reader_study/` | `build_reader_study_tables.py`, `finalize_reader_study_tables.py` | RS | R* |
| S32 Primary endpoints per reader | 6 | as S31 | `build_reader_study_tables.py`, `finalize_reader_study_tables.py` | RS | R* |
| S39–S47 Reader-study secondary tables | 6 | as S31; S47 coded comments: `build_reader_comment_coding.py` | `build_reader_study_tables.py`, `finalize_reader_study_tables.py` | RS; comment code sheet (reader-study material, not released) | R* |
| S49 Readers' answers and the retrieved labels without a reader (exploratory) | 6 | `revision/scripts/reader_study/search_vs_reader.py` | `revision/retained_executed/collab_draft_20261102/checks_050/reproduction/make_s49_pass050.py` | RS | R* |

Notes on specific tables:

- **S27–S30** keep the submitted tables' slots. The submitted Tables S10–S13
  contents are carried over, and the revision additions (patient-held-out
  recomputation, recomputed plurality columns) are written by
  `assemble_final_tables_20261110.py` from the outputs of
  `e1_bayes_heldout_primary.py`.
- **S48, LLM generation.** Re-running the summarizer calls a hosted model
  (GPT-5 via Azure OpenAI). Output is not guaranteed to match the retained
  responses. The executed instruction text contained two study identifiers
  as format examples; in this release they are replaced by synthetic
  identifiers of the same format, and nothing else in the prompt or logic
  differs. Pilot generation needs the restricted query payloads; analysis of
  the retained responses is runnable given the retained LLM outputs.
- **Reader-study tables** read a frozen, immutable export of the study
  database. The analysis never reads the reader-token table. Reader logins
  are mapped to "Reader 1–5" through a local configuration file that is not
  released. `search_vs_reader.py` takes the reader logins in Reader 1–5
  order on its command line (`--readers`).
- **S49** is an exploratory comparison added after the reader results were
  known. `search_vs_reader.py` applies fixed rules to the labels of the
  retrieved slides that readers were shown (for example the nearest slide, or
  the plurality vote over the five or ten nearest with count ties going to the
  nearest-ranked label) and compares them with each reader's favored
  diagnosis. `make_s49_pass050.py` writes four of its eight output tables as
  the S49 sheets, dropping the rows and the column listed in its docstring.

The upstream analysis lists name the principal producers. The complete input
set of each table is the `input_bindings.json` that its builder writes.

## Code manifest

[CODE_MANIFEST.tsv](CODE_MANIFEST.tsv) has one row per file in the repository
plus one row per executed file that is not released. Columns: `path`;
`source` (the executed file, with path placeholders as in
[PATHS.md](PATHS.md), or `public main` for files inherited unchanged from the
earlier public release); `sha256_executed` and `sha256_released`;
`change_class`; and `substitution_counts` per class. Change classes:

- `none`: released byte-identical to the executed file.
- `path-literal substitution`: workspace path prefixes, home directory and
  login name replaced by placeholders ([PATHS.md](PATHS.md)); `split literal`
  marks a prefix written across two string pieces.
- `identity scrub`: reader logins replaced by a lookup from a local
  configuration file, or by fake example logins in tests.
- `credential scrub`: a private service endpoint replaced by an empty value
  read from the environment; `host scrub`: an internal file-server name
  replaced by a placeholder.
- `identifier scrub`: study or slide identifiers replaced by synthetic
  identifiers of the same format (summarizer instruction examples and test
  fixtures), or by a lookup from the local figure-identifier file
  (`identifier scrub (local figure-ID lookup)`).
- `comment scrub`: internal work-tracking notes and personal names removed
  from comments and docstrings only; code is unchanged, which the release
  tooling verified token by token. String literals are released as executed.
- `checksum-pin update`: a sha256 pin to another released file replaced by
  that file's released checksum (see below).
- `held` / `not released`: see [KNOWN_GAPS.md](KNOWN_GAPS.md).

[INTEGRITY_PINS.tsv](INTEGRITY_PINS.tsv) lists every checksum pin inside a
released file whose target was edited for the release, with the executed
checksum (the value at execution time) and the released checksum (the value
now in the code).
