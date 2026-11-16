#!/usr/bin/env python3
"""Final numbered supplementary-table package (revision, 2026-11-10).

For every Supplementary Table S1–S47 of the revision numbering (scripts/figures/supp_table_numbering.json,
field `pass026`) this script takes the accepted source workbook (revision candidate build, earlier built
workbook, or the submitted file), applies the independent packaging decisions of 2026-11-10 (below), regenerates the
first sheet `caption` from the Supplementary Information Markdown (the single source of truth for caption
wording; decision D6 of [internal record]) and writes
`Supplementary_Table_S<n>.xlsx` with deterministic bytes under

    $PROSPER_OUT/supp_package_20261110/supplementary_tables/

plus manifest.json, SHA256SUMS.txt, BUILD_REPORT.md and README.md. Reader-study tables (S31, S32, S39–S47)
are taken from the directory given with `--reader-tables` (output of build_reader_study_tables.py on one
five-reader analysis directory); without the option they are listed as pending in the manifest.

Assembly only: no statistic is computed. The only new numbers are copied from
$PROSPER_OUT/c10_calibration/calibration_uni2_ot_cosine.json (already bound by the number manifests) into the
Table S26 calibration sheets, and the feature dimensions in Table S1 are read from the SI/Methods statements and
the feature banks (verified 2026-11-10: H-Optimus-1 1,536; RetCCL 2,048; CONCH v1.5 768 for TITAN).

Design decisions applied (2026-11-10; recorded in CHANGELOG_026.md):
  S1   revised encoder table: Virchow2 1,280 of 2,560 (Supplementary Note 4, item 21); H-Optimus-1 and RetCCL
       rows added; CONCH v1.5 described as the TITAN pipeline's tile encoder; placeholder and typos removed.
  S4   Sheet 2 keeps one flag column, `shown in Fig. 2C` (the pipelines of the current Fig. 2C); the RR-5 column
       is dropped.
  S11  Sheet1_units: St. Jude patient identifiers replaced by pseudonyms (patient_001…); the mapping is written
       to the controlled/ directory, which is not part of the release.
  S15  the CCDI-MCI 'Submitted -> current' cell and the COMET 'Slides evaluated' cell are shortened to pointers
       (R3.3-T1: 948 and 2,766 appear only in the footnotes).
  S26  calibration sheets added from c10_calibration (k = 5 and 18; 2,778 queries), as the SI caption states.
  S33  status value 'out of scope (CNS; corrected from NB — R1.2 scope decision)' -> 'out of scope (CNS
       neuroblastoma, FOXR2-activated; corrected from NB)'.
  S24, S25  per-reference rows (internal slide index and tumor group; no patient codes) and aggregate matrices
       are released as built.
  all  American spelling in prose cell text (whole-word replacements, counts reported; snake_case identifiers and
       source keys are left unchanged); the caption sheet is rebuilt.

revision packaging fixes (2026-11-10; audit findings C9–C12; function `pass028_fixes`, run after the handler):
  S10  24 '= plurality (binary)' text cells stored as strings (they were read as formulas); inventory Z/AA/AB 3 and 12
       point to Supplementary Table S9 (subtype all-tissue intervals), not S8.
  S7, S8, S9, S17  sheet names carrying pre-revision table numbers renamed (SHEET_RENAMES).
  S9   'S9 body' B14 and B18: tumor-only tile selection points to Supplementary Table S12 (submitted S5).
  S15  current_populations I5 points to Supplementary Table S33.
  S35  bare 'Table S3' (macro_F1 G1, other_metrics G1, raw_joined AE) qualified as 'submitted Table S3'.
  S26  reliability gap signed (mean confidence − accuracy); sheet plurality_recomputed from
       $PROSPER_OUT/knn_comparator_recalc_20261110/results.csv; population note below model_comparison_2766.
  S27, S29  Sheet2/Sheet3 header rows; Sheet1 columns G1_recomputed / G2_recomputed (plurality with closest-rank
       tie-break vs Bayesian argmax of S28/S30; asserted 72/89 and 109/46).
  S33  final row 'No diagnosis string recorded' (236 of 4,160; 231 of 4,154; r3_3 v3 counts.json NO_METADATA).

Run from the repository root:
    export PROSPER_OUT="$PWD/prosper_revision_output"
    /HOME/miniconda3/bin/python3 scripts/supp_tables/assemble_final_tables_20261110.py \
        [--reader-tables "$PROSPER_OUT/supp_tables_reader_study/<analysis-dir-name>"]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import packaging_common_20261109 as pc  # noqa: E402

pc.BUILD_DATE = "2026-11-10"
REPO, OUT = pc.REPO, pc.OUT_ROOT
SI_MD = pc.SI_MD
NUMBERING = pc.NUMBERING
SUB = REPO / "docs/prosper_submission/supplementary/supplementary_tables"
P26 = OUT / "supp_tables_20261109"
PKG = OUT / "supp_package_20261110"
TABLES = PKG / "supplementary_tables"
CONTROLLED = PKG / "controlled"
CALIB = OUT / "c10_calibration/calibration_uni2_ot_cosine.json"

# final number -> (workspace id, source path relative to $PROSPER_OUT or docs/, handler)
A, B, C = P26 / "A_packaging", P26 / "B_external_perturbation_finelabel", P26 / "C_localization"
A20 = OUT / "supp_tables_20261022/A_ladder_failure_finelabel"
A03 = OUT / "supp_tables_20261103/A_ladder_failure_finelabel"
B20 = OUT / "supp_tables_20261022/B_external_leakage_titan"
C20 = OUT / "supp_tables_20261022/C_encoders_comparators"
SOURCES = {
    "S1": ("S1", SUB / "Table_S1_encoder_details.xlsx", "s1"),
    "S2": ("S2", SUB / "Table_S2_aggregation_details.xlsx", None),
    "S3": ("S29", A20 / "R2.2-T4_S29.xlsx", None),
    "S4": ("S-R2.1-E", A / "S-R2.1-E/Table_S-R2.1-E.xlsx", "s4"),
    "S5": ("S-R3.1", C20 / "Table_S-R3.1.xlsx", None),
    "S6": ("S23", A20 / "R2.3-T2_S23_S-A1.xlsx", None),
    "S7": ("S24", A20 / "R2.3-T3_S24_S-A2.xlsx", None),
    "S8": ("S25", C20 / "Table_S25.xlsx", None),
    "S9": ("S26", A03 / "R3.7-T4_S26_S-D1_v2.xlsx", None),
    "S10": ("S28", B / "S28_S9_v2/Table_S28_finelabel_inventory_v2.xlsx", None),
    "S11": ("S16", A / "S16/Table_S16.xlsx", "s11"),
    "S12": ("S-R2.1-D", C20 / "Table_S-R2.1-D.xlsx", None),
    "S13": ("S-R2.1-B", C20 / "Table_S-R2.1-B.xlsx", None),
    "S14": ("S-R2.1-C", B / "S-R2.1-C_S15/Table_S-R2.1-C_perturbation_suite.xlsx", None),
    "S15": ("S14", A / "S14/Table_S14.xlsx", "s15"),
    "S16": ("S18", B20 / "Table_S18_primary_broad_group.xlsx", None),
    "S17": ("S30", A20 / "R3.2-T1_S30.xlsx", None),
    "S18": ("S6", B / "S6_S19/Table_S6_NB_descriptive_factors.xlsx", None),
    "S19": ("S7", B / "S7_S20/Table_S7_NB_plurality_by_diagnosis.xlsx", None),
    "S20": ("S8", B / "S8_S21/Table_S8_CCDI_eligibility_sensitivity.xlsx", None),
    "S21": ("S19", B20 / "Table_S19_configuration_filtering_coverage.xlsx", None),
    "S22": ("S20", B20 / "Table_S20_failures_neighbours_transitions.xlsx", None),
    "S23": ("S21", B20 / "Table_S21_descriptive_strata.xlsx", None),
    "S24": ("S-R3.5-L2", C / "S-R3.5-L2/Table_S-R3.5-L2.xlsx", None),
    "S25": ("S-R3.5-L1", C / "S-R3.5-L1/Table_S-R3.5-L1.xlsx", "s25"),
    "S26": ("S9", SUB / "Table_S9_bayesian_classification_stats.xlsx", "s26"),
    "S27": ("S10", SUB / "Table_S10_disease_of_top_hits_N5.xlsx", None),
    "S28": ("S11", SUB / "Table_S11_Interp_combined_similarity_matrix_uni2_ot_cosine_tumor_only_no_same_patient_probs_N5.xlsx", None),
    "S29": ("S12", SUB / "Table_S12_disease_of_top_hits_N18.xlsx", None),
    "S30": ("S13", SUB / "Table_S13_Interp_combined_similarity_matrix_uni2_ot_cosine_tumor_only_no_same_patient_probs_N18.xlsx", None),
    "S33": ("S15", A / "S15/Table_S15.xlsx", "s33"),
    "S34": ("S22", B20 / "Table_S22_TITAN_tumour_filtering.xlsx", None),
    "S35": ("S-R2.1-F", C20 / "Table_S-R2.1-F.xlsx", None),
    "S36": ("S-R4.a-1", B20 / "Table_S-R4a-1_masked_vs_unmasked.xlsx", None),
    "S37": ("S-R4.a-2", B20 / "Table_S-R4a-2_same_patient_consistency.xlsx", None),
    "S38": ("S27", B20 / "Table_S27_TITAN_unannotated_queries.xlsx", None),
    # revision (2026-11-10): LLM grounding pilot endpoints, built by build_llm_pilot_r2_tables.py
    "S48": ("S-LLM-PILOT", OUT / "supp_tables_20261110_llm_pilot/Table_S-LLM-PILOT.xlsx", None),
}
PENDING_FREEZE = {"S31": "S-RS-T1", "S32": "S-RS-T2", "S39": "S-RS-T3", "S40": "S-RS-T4", "S41": "S-RS-T5",
                  "S42": "S-RS-T6", "S43": "S-RS-T7", "S44": "S-RS-T8", "S45": "S-RS-T9", "S46": "S-RS-T10",
                  "S47": "S-RS-T11"}

# American spelling, whole words, case preserved on the first letter.
SPELLING = {
    "tumour": "tumor", "tumours": "tumors", "neighbour": "neighbor", "neighbours": "neighbors",
    "neighbourhood": "neighborhood", "neighbourhoods": "neighborhoods", "colour": "color", "colours": "colors",
    "labelled": "labeled", "labelling": "labeling", "normalisation": "normalization", "normalised": "normalized",
    "normalise": "normalize", "localisation": "localization", "localised": "localized", "centre": "center",
    "artefact": "artifact", "artefacts": "artifacts", "analysed": "analyzed", "behaviour": "behavior",
    "modelling": "modeling", "modelled": "modeled", "grey": "gray", "favoured": "favored", "favour": "favor",
    "summarised": "summarized", "standardised": "standardized", "optimised": "optimized", "randomised": "randomized",
    "recognised": "recognized", "characterised": "characterized", "minimised": "minimized", "maximised": "maximized",
    "visualisation": "visualization", "visualised": "visualized", "generalisation": "generalization",
    "initialised": "initialized", "harmonised": "harmonized", "categorised": "categorized",
    "nontumour": "nontumor", "non-tumour": "non-tumor",
}
# \b keeps snake_case / key-like tokens (e.g. 'tumour_only_2604', 'R_massmarg_nontumour_pooled') unchanged: they are
# identifiers that trace to source files and are not prose.
_SPELL_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, SPELLING), key=len, reverse=True)) + r")\b", re.I)


def americanize(s: str, counter: dict) -> str:
    def rep(m):
        w = m.group(0)
        r = SPELLING[w.lower()]
        counter[w.lower()] = counter.get(w.lower(), 0) + 1
        return r.capitalize() if w[0].isupper() else r
    return _SPELL_RE.sub(rep, s)


# ------------------------------------------------------------------------------------------------ SI parsing
def si_entries() -> dict[str, dict]:
    """Table entries of the SI Markdown: {final: {line, title, caption, paragraphs}}; shared 'Notes to
    Supplementary Tables Sa and Sb' paragraphs are attached to every listed table."""
    lines = SI_MD.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "## Supplementary Tables")
    ent: dict[str, dict] = {}
    cur = None
    shared: tuple[list[str], list[str]] | None = None  # (targets, paragraphs)
    head_re = re.compile(r"^Supplementary Table (S\d+) \| (.+)$")
    shared_re = re.compile(r"^(?:Notes|Footnotes) to Supplementary Tables ((?:S\d+(?:, | and )?)+)\.?$")
    for i in range(start + 1, len(lines)):
        l = lines[i].rstrip()
        if not l:
            continue
        if l.startswith("## "):
            break
        m = head_re.match(l)
        if m:
            if shared:
                for t in shared[0]:
                    ent.setdefault(t, {"line": None, "title": None, "caption": None, "paragraphs": []})
                    ent[t]["paragraphs"].extend(shared[1])
                shared = None
            num, rest = m.group(1), m.group(2)
            title, _, cap = rest.partition(". ")
            e = ent.setdefault(num, {"line": None, "title": None, "caption": None, "paragraphs": []})
            e.update(line=i + 1, title=title.rstrip("."), caption=cap.strip())
            cur = num
            continue
        ms = shared_re.match(l)
        if ms:
            targets = re.findall(r"S\d+", ms.group(1))
            shared = (targets, [l])
            continue
        if shared is not None:
            shared[1].append(l)
        elif cur and not (i == start + 2):  # skip the section intro sentence
            if l.startswith("Each table is provided as a separate workbook"):
                continue
            if l.startswith("Workbook: Supplementary Data "):
                continue  # file pointer written by build_supplementary_data_workbooks.py, not a note
            ent[cur]["paragraphs"].append(l)
    if shared:
        for t in shared[0]:
            ent.setdefault(t, {"line": None, "title": None, "caption": None, "paragraphs": []})
            ent[t]["paragraphs"].extend(shared[1])
    return ent


def caption_rows(final: str, e: dict, wid: str, src: Path, src_sha: str, si_sha: str) -> list[tuple[str, str]]:
    rows = [("table", f"Supplementary Table {final}"), ("title", e["title"]), ("caption", e["caption"])]
    k = 0
    for p in e["paragraphs"]:
        if re.match(r"^(Notes|Footnotes?) to Supplementary Tables? ", p) and p.rstrip(".").endswith(
                tuple(re.findall(r"S\d+", p))):
            rows.append(("notes_heading", p))
            continue
        if p.startswith("Footnote"):
            rows.append(("footnotes", p))
            continue
        k += 1
        rows.append((f"note_{k}", p))
    rows += [
        ("caption_source", f"docs/prosper_revised/prosper_supplementary_legends.md l.{e['line']} "
                           f"(sha256 {si_sha[:16]}); the Supplementary Information is the source of the caption wording."),
        ("source_workbook", f"{pc.rel(src)} (sha256 {src_sha[:16]}; workspace id {wid})"),
        ("assembled", f"scripts/supp_tables/assemble_final_tables_20261110.py, {pc.BUILD_DATE}"),
    ]
    return rows


# ------------------------------------------------------------------------------------------------ handlers
def h_s1(wb: openpyxl.Workbook, rep: pc.Report, ctx: dict) -> None:
    """Revised Table S1 (encoder table) in place of the submitted sheet."""
    ws_old = wb["Sheet1"]
    sub_rows = list(ws_old.iter_rows(values_only=True))
    rep.check(sub_rows[0][0] == "Encoder" and len(sub_rows) == 10, f"submitted S1 read: {len(sub_rows) - 1} encoder rows")
    sub = {r[0]: r for r in sub_rows[1:]}
    rep.check(sub["Virchow2"][4] == 2560, "submitted Virchow2 embedding dimension 2560 (corrected below to 1,280 used)")
    wb.remove(ws_old)
    ws = wb.create_sheet("encoders")
    header = ["Encoder", "Category", "Role in this study", "Input tile (pixels)", "Tiles encoded",
              "Embedding dimension used", "Model source", "Main-text reference", "Notes"]
    grid = "Tile-encoder grid (median pooling, SAMPLER, Optimal Transport, Yottixel)"
    mosaic = "512 × 512 mosaic tiles (15% of tissue tiles per slide)"
    inf = "Inference only, no fine-tuning"
    rows = [
        ["Virchow2", "Pathology foundation model", grid, "512 × 512", mosaic,
         "1,280 (the first 1,280 of the 2,560 output dimensions, the class token; the mean patch-token half was not used)",
         "https://huggingface.co/paige-ai/Virchow2", "20",
         f"{inf}. The benchmark feature bank was recomputed in float32 (Supplementary Table S35)."],
        ["UNI2", "Pathology foundation model", grid + "; main configuration with Optimal Transport", "512 × 512", mosaic,
         "1,536", "https://huggingface.co/MahmoodLab/UNI2-h", "19", inf + "."],
        ["GigaPath (tile encoder)", "Pathology foundation model", grid, "512 × 512", mosaic, "1,536",
         "https://huggingface.co/prov-gigapath/prov-gigapath", "21", inf + "; tile encoder only."],
        ["EXAONEPath", "Pathology foundation model", grid, "512 × 512", mosaic, "768",
         "https://huggingface.co/LGAI-EXAONE/EXAONEPath", "44", inf + "."],
        ["Phikon v2", "Pathology foundation model", grid, "512 × 512", mosaic, "1,024",
         "https://huggingface.co/owkin/phikon-v2", "45", inf + "."],
        ["CTransPath", "Pathology foundation model", grid, "512 × 512", mosaic, "768",
         "https://github.com/Xiyue-Wang/TransPath", "46", inf + "."],
        ["Hibou-L", "Pathology foundation model", grid, "512 × 512", mosaic, "1,024",
         "https://huggingface.co/histai/hibou-L", "47", inf + "."],
        ["H-Optimus-1", "Pathology foundation model", grid, "512 × 512 (resized to 224 × 224 for the encoder)", mosaic,
         "1,536", "https://huggingface.co/bioptimus/H-optimus-1", "54",
         f"{inf}; added after submission; encoder-specific normalization and fp16 autocast (Supplementary Methods)."],
        ["RetCCL", "Self-supervised pathology tile encoder (ResNet-50 backbone)", grid, "512 × 512", mosaic, "2,048",
         "https://github.com/Xiyue-Wang/RetCCL", "37",
         "Released tile encoder evaluated under each of the four aggregations; RetCCL releases no slide index of its own (Supplementary Table S5)."],
        ["SimCLR (COMET-trained)", "Self-supervised baseline trained in this study (ResNet-18)", grid, "512 × 512", mosaic,
         "512", "Trained in this study (no public checkpoint)", "14",
         "Trained from scratch on about 4 million COMET mosaic tiles with standard SimCLR augmentations; no diagnosis labels were used."],
        ["CONCH v1.5", "Vision–language pathology model (tile encoder of the CONCH–TITAN pipeline)",
         "Separate end-to-end pipeline: CONCH v1.5 tile embeddings aggregated by TITAN", "1,024 × 1,024 at 40×",
         "All tissue tiles (no mosaic selection; no overlap)", "768 (patch embedding consumed by TITAN)",
         "https://huggingface.co/MahmoodLab/conchv1_5", "28",
         "Inference only. Twenty-one all-zero TITAN slide embeddings were regenerated before benchmarking (Supplementary Table S34)."],
    ]
    ws.append(header)
    for r in rows:
        ws.append(r)
    for col, w in zip("ABCDEFGHI", (22, 44, 52, 30, 40, 48, 48, 12, 90)):
        ws.column_dimensions[col].width = w
    rep.check(len(rows) == 11, "11 encoder rows: eight foundation models, RetCCL, SimCLR (ten-encoder grid) and CONCH v1.5 (TITAN pipeline)")
    kept = {"UNI2": "1,536", "GigaPath": "1,536", "EXAONEPath": "768", "Phikon v2": "1,024", "CTransPath": "768",
            "Hibou L": "1,024", "CONCH v1.5": None, "SimCLR": "512"}
    for name, dim in kept.items():
        if dim:
            srow = sub[name]
            ok = f"{srow[4]:,}" == dim
            rep.check(ok, f"{name}: dimension {dim} equals the submitted value {srow[4]}")
    rep.notes.append("Dimensions verified 2026-11-10 against feature banks: H-Optimus-1 1,536 and RetCCL 2,048 "
                     "($PROSPER_OUT/features/{h_optimus_1,retccl}/10002.pkl, tile arrays of width 1,536 / 2,048); "
                     "Virchow2 first 1,280 = class token (docs/adr/0021, reviewer Q2); CONCH v1.5 768 "
                     "(comet-titan/trident/trident_processed/40x_1024px_0px_overlap/features_conch_v15/10002.h5, (4276, 768)).")
    rep.open_items.append("Main-text Methods state that CONCH features were extracted 'with the CLAM framework'; the TITAN "
                          "analyses of this revision read Trident-tiled CONCH v1.5 features (768-d, 1,024 px, 40×, no overlap). "
                          "Haoran to confirm how the submitted TITAN embeddings were produced and whether the Methods sentence "
                          "should name Trident (ref. 56) instead of CLAM (ref. 5).")
    rep.notes.append("Submitted SimCLR source note 'Trained from scratch on COMET using split XXX' (placeholder) replaced by "
                     "the Methods statement (about 4 million COMET mosaic tiles; self-supervised).")


def h_s4(wb, rep, ctx):
    ws = wb["Sheet2_per_group_k5"]
    hdr = [c.value for c in ws[1]]
    c_cur = hdr.index("current Fig. 2C pipeline (figures_20261021)") + 1
    c_rr5 = hdr.index("RR-5 Fig. 2C pipeline (fig2c_groups.csv)") + 1
    flagged = sorted({(ws.cell(r, hdr.index("Encoder") + 1).value, ws.cell(r, hdr.index("Aggregation") + 1).value)
                      for r in range(2, ws.max_row + 1) if ws.cell(r, c_cur).value == "yes"})
    rep.check(flagged == [("Conchv1.5", "TITAN"), ("H-Optimus-1", "Yottixel"), ("UNIv2", "Optimal Transport"),
                          ("Virchow2", "Yottixel")],
              f"current Fig. 2C flag marks exactly the four Fig. 2C pipelines: {flagged}")
    ws.delete_cols(c_rr5)
    ws.cell(1, c_cur).value = "shown in Fig. 2C"
    rep.notes.append("Sheet 2: RR-5 flag column dropped; the current-Fig. 2C flag column renamed 'shown in Fig. 2C'.")


def h_s11(wb, rep, ctx):
    ws = wb["Sheet1_units"]
    hdr = [c.value for c in ws[1]]
    cu, ct = hdr.index("unit") + 1, hdr.index("unit_type") + 1
    ids = sorted({str(ws.cell(r, cu).value) for r in range(2, ws.max_row + 1) if ws.cell(r, ct).value == "patient"})
    rep.check(all(i.startswith("SJ") for i in ids), f"{len(ids)} distinct patient identifiers in Sheet1_units, all SJ-prefixed")
    mapping = {i: f"patient_{k:03d}" for k, i in enumerate(ids, 1)}
    n = 0
    for r in range(2, ws.max_row + 1):
        if ws.cell(r, ct).value == "patient":
            ws.cell(r, cu).value = mapping[str(ws.cell(r, cu).value)]
            n += 1
    ws.cell(1, cu).value = "unit (patients pseudonymized; slide units keep the internal slide index)"
    CONTROLLED.mkdir(parents=True, exist_ok=True)
    p = CONTROLLED / "Supplementary_Table_S11_units_patient_map.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pseudonym", "patient_id"])
        for i, ps in mapping.items():
            w.writerow([ps, i])
    rep.check(n == sum(1 for r in range(2, ws.max_row + 1) if ws.cell(r, ct).value == "patient"),
              f"{n} patient-unit rows pseudonymized; mapping written to controlled/ (not released)")
    ctx["controlled"] = [p]


def h_s15(wb, rep, ctx):
    ws = wb["current_populations"]
    hdr = [c.value for c in ws[1]]
    col = {h: i + 1 for i, h in enumerate(hdr)}
    rows = {ws.cell(r, col["Collection"]).value: r for r in range(2, ws.max_row + 1)}
    r_comet, r_ccdi = rows["COMET (final library)"], rows["CCDI-MCI (external)"]
    old1 = ws.cell(r_comet, col["Slides evaluated"]).value
    old2 = ws.cell(r_ccdi, col["Submitted → current"]).value
    rep.check("2,766" in old1 and "948" in old2, "the two cells named in the S14 BUILD_REPORT open items located")
    ws.cell(r_comet, col["Slides evaluated"]).value = (
        "2,778 queries against 2,778 references (all-tissue); tumor-only: 2,604 queries against 2,603 non-empty "
        "references (one RB slide has no tumor tiles; the tumor-only distance matrix is described in footnote b)")
    ws.cell(r_ccdi, col["Submitted → current"]).value = (
        "Submitted row: slide count withdrawn (not reconstructable from retained records; see the notes), 21 entities, "
        "manual slide-level mapping cited as Table S10. The published Fig. 4C population (867 slides, 20 entities) is "
        "listed under historical collections; the mapping is regenerated as Supplementary Table S33; the 21 entities "
        "correspond to 12 observed COMET groups in the 524-slide main population.")
    allcells = [str(ws.cell(r, c).value) for r in range(2, ws.max_row + 1) for c in range(1, ws.max_column + 1)]
    rep.check(not any("948" in v for v in allcells), "948 absent from every current_populations cell")
    rep.check(not any("2,766" in v for v in allcells), "2,766 absent from every current_populations cell")
    rep.notes.append("S15 cells shortened per R3.3-T1 (948 and 2,766 only in the footnotes); previous texts kept in the "
                     "pass-026 candidate workbook.")


def h_s26(wb, rep, ctx):
    d = json.loads(CALIB.read_text())
    rep.check(d["config"]["n_queries"] == 2778 and d["config"]["tumor_only"] is False,
              "calibration source: 2,778 all-tissue queries, UNI2–Optimal Transport, T = 20")
    wb["Sheet1"].title = "model_comparison_2766"
    ws = wb.create_sheet("calibration_summary")
    fields = [("accuracy", "argmax accuracy"), ("mean_confidence", "mean maximum score"),
              ("ece_15_equal_width", "expected calibration error (15 equal-width bins)"),
              ("mce_15_equal_width", "maximum calibration error (15 equal-width bins)"),
              ("ece_15_equal_mass", "expected calibration error (15 equal-mass bins)"),
              ("brier_multiclass", "multiclass Brier score"),
              ("brier_prevalence_prior", "Brier score of the library-prevalence reference forecast"),
              ("brier_skill_score", "Brier skill score against the prevalence reference"),
              ("nll", "negative log-likelihood (nats)")]
    ws.append(["k", "n_queries", "quantity", "value", "95% CI low", "95% CI high", "note"])
    for k in ("5", "18"):
        b = d["by_N"][k]
        ci = b.get("ci", {})
        for key, name in fields:
            lo, hi = (ci[key] if key in ci else (None, None))
            ws.append([int(k), d["config"]["n_queries"], name, b[key], lo, hi,
                       "95% percentile bootstrap over queries (B = 2,000, seed 123)" if lo is not None else ""])
    ws2 = wb.create_sheet("calibration_reliability_bins")
    ws2.append(["k", "bin (equal width on [0, 1])", "n", "mean confidence", "accuracy", "gap (confidence − accuracy)"])
    for k in ("5", "18"):
        for bb in d["by_N"][k]["reliability_equal_width"]:
            # signed gap = mean confidence − accuracy (revision, audit C11: the JSON 'gap' field is |conf − acc|)
            conf, acc = bb.get("confidence"), bb.get("accuracy")
            gap = conf - acc if conf is not None and acc is not None else None
            if gap is not None and bb.get("gap") is not None:
                assert abs(abs(gap) - bb["gap"]) < 1e-12, (k, bb)
            ws2.append([int(k), bb["bin"], bb["n"], conf, acc, gap])
    e5, e18 = d["by_N"]["5"]["ece_15_equal_width"], d["by_N"]["18"]["ece_15_equal_width"]
    rep.check(round(e18, 3) == 0.030 and round(e5, 3) == 0.071, f"ECE k = 18 {e18:.4f} -> 0.030; k = 5 {e5:.4f} -> 0.071 (SI/Results)")
    rep.check(round(d["by_N"]["18"]["accuracy"], 3) == 0.846 and round(d["by_N"]["5"]["accuracy"], 3) == 0.838,
              "argmax accuracy 0.846 (k = 18) and 0.838 (k = 5) as in the SI note")
    rep.check(round(d["by_N"]["18"]["mean_confidence"], 3) == 0.827 and round(d["by_N"]["5"]["mean_confidence"], 3) == 0.909,
              "mean maximum score 0.827 (k = 18) and 0.909 (k = 5) as in the SI note")
    ctx["extra_inputs"] = [CALIB]
    rep.notes.append("Calibration sheets copied from c10_calibration/calibration_uni2_ot_cosine.json; the tumor-only "
                     "companion file (2,604 queries) is not published because the SI describes the 2,778-query assessment only.")


S25_ADDENDUM = OUT / "supp_tables_20261112_roi6_tcm/Table_S25_addendum_roi6_tcm.xlsx"


def h_s25(wb, rep, ctx):
    """revision draft (2026-11-12): sheets C3-C4, D5-D8 and E1-E2 from build_S25_roi6_tcm_addendum_20261112.py."""
    old = "all same-patient slides excluded (reported, Fig. 6A)"
    new = "all same-patient slides excluded (submitted float16 Virchow2 L1 hits; Fig. 6A: sheets D5–D8)"
    n = 0
    for name in ("D1_roi_overall", "D3_roi_per_group"):
        for row in wb[name].iter_rows():
            for c in row:
                if c.value == old:
                    c.value, n = new, n + 1
    rep.check(n == 2 + 15, f"block D label of the submitted patient-excluded k = 5 rows updated in D1/D3 ({n} cells)")
    add = openpyxl.load_workbook(S25_ADDENDUM)
    for ws_src in add.worksheets:
        ws = wb.create_sheet(ws_src.title)
        for row in ws_src.iter_rows(values_only=True):
            ws.append(list(row))
        for k, dim in ws_src.column_dimensions.items():
            ws.column_dimensions[k].width = dim.width
        ws.freeze_panes = "A2"
    rep.check(wb.sheetnames[-len(add.sheetnames):] == add.sheetnames, f"addendum sheets appended after D4: {add.sheetnames}")
    ctx["extra_inputs"] = [S25_ADDENDUM]
    rep.notes.append("S25 sheets C3-C4 (cosine heat maps, 44 references), D5-D8 (six ROI-search configurations) and E1-E2 "
                     "(tumour-correspondence map) appended from supp_tables_20261112_roi6_tcm (pass 036 draft).")


def h_s33(wb, rep, ctx):
    ws = wb["mapping"]
    hdr = [c.value for c in ws[1]]
    c = hdr.index("Eligibility status") + 1
    old = "out of scope (CNS; corrected from NB — R1.2 scope decision)"
    new = "out of scope (CNS neuroblastoma, FOXR2-activated; corrected from NB)"
    n = 0
    for r in range(2, ws.max_row + 1):
        if ws.cell(r, c).value == old:
            ws.cell(r, c).value = new
            n += 1
    rep.check(n == 3, f"{n} status values reworded (3 strings expected)")


# ------------------------------------------------------------------------------------------------ revision fixes
# Packaging fixes of 2026-11-10 (code/consistency audit $PROSPER_OUT/e1_code_audit_20261109/REVIEW.md, findings
# C9–C12; verification [internal record]). Applied by `pass028_fixes` after the per-table handler
# and before the spelling pass and the caption sheet. Every fix starts from the accepted source workbook on each
# run, so the step is idempotent; each fix asserts the exact text or count it changes and stops the build otherwise.
KNN_RECALC = OUT / "knn_comparator_recalc_20261110"
COHORT_COUNTS = OUT / "r3_3_cohort_accounting_20261008_v3/counts.json"  # source of build_S14_cohort_accounting_20261109.py
# old sheet name -> current sheet name (sheet names that carried a pre-revision table number)
SHEET_RENAMES = {
    "S7": {"S24 plurality@5": "S7 plurality@5", "S24b strict@5 + Abstain": "S7b strict@5 + Abstain",
           "S24c top-1 (optional)": "S7c top-1 (optional)"},
    "S8": {"S25": "S8", "S25_points_C2": "S8_points_C2", "S25b_paired": "S8b_paired", "S25_F1_k5": "S8_F1_k5"},
    "S9": {"S26 body": "S9 body"},
    "S17": {"S30 table": "S17 table", "long (table_s30.csv)": "S17 long"},
}
# (sheet, cell, old substring, new substring): stale cross-references in cell text
XREF_FIXES = {
    "S10": [("inventory", c, "Supplementary Table S8", "Supplementary Table S9")
            for c in ("Z3", "AA3", "AB3", "Z12", "AA12", "AB12")],
    "S15": [("current_populations", "I5", "(regenerated Table S15, v03 rule)",
             "(Supplementary Table S33, regenerated from the submitted Table S15; v03 rule)")],
    # the submitted Table S5 (Fig. 3 relative-error-reduction table) is now S12; B18 carries the same stale pointer
    # (matched before the spelling pass: the source cell reads 'tumour-only')
    "S9": [("S9 body", c, "tile selection, Supplementary Table S5)",
            "tile selection, Supplementary Table S12)") for c in ("B14", "B18")],
}
PLURALITY_NOTE = ("Plurality-vote kNN recomputed from the released neighbor labels of Supplementary Tables S27 (k = 5) "
                  "and S29 (k = 18); count ties broken toward the closest-ranked tied label; Bayesian argmax from "
                  "Supplementary Tables S28 and S30. The model_comparison_2766 kNN rows are carried from the submitted "
                  "workbook and could not be regenerated.")
MC2766_NOTE = ("Note: the population is the 2,766 query slides of the submitted tumor-only score matrices, NORMAL "
               "queries included; the \"no tie break\" rows are carried from the submitted analysis.")
G_EXPECTED = {"S27": (5, "S28", 72, 89), "S29": (18, "S30", 109, 46)}


def _plurality_closest_rank(labels: list) -> str:
    """Rule of knn_comparator_recalc.py: plurality over the k labels; count ties go to the closest-ranked tied label."""
    from collections import Counter
    c = Counter(labels)
    m = max(c.values())
    tied = [l for l in c if c[l] == m]
    if len(tied) == 1:
        return tied[0]
    return next(l for l in labels if l in tied)


def _fix_formulas(final, wb, rep):
    n = 0
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if cell.data_type == "f" and cell.value == "= plurality (binary)":
                    cell.value = "= plurality (binary)"
                    cell.data_type = "s"  # plain text: openpyxl infers a formula from the leading '='
                    n += 1
    if final == "S10":
        rep.check(n == 24, f"{n} '= plurality (binary)' text cells stored as strings, not formulas (24 expected)")
        if n != 24:
            raise SystemExit(f"S10: {n} '= plurality (binary)' formula cells, 24 expected")


def _fix_xrefs(final, wb, rep):
    for sheet, coord, old, new in XREF_FIXES.get(final, []):
        cell = wb[sheet][coord]
        v = cell.value
        if not (isinstance(v, str) and v.count(old) == 1):
            raise SystemExit(f"{final} {sheet}!{coord}: expected one '{old}' in {v!r}")
        cell.value = v.replace(old, new)
        rep.check(True, f"{sheet}!{coord}: '{old}' -> '{new}'")


_BARE_S3 = re.compile(r"(?<!submitted )\bTable S3\b")


def _s35_submitted_s3(wb, rep):
    """S35: the bare 'Table S3' means the submitted Table S3 (benchmark; now S4), not the current S3 (tile ladder)."""
    n = {}
    for ws in wb.worksheets:
        if ws.title == "caption":
            continue
        for row in ws.iter_rows():
            for cell in row:
                v = cell.value
                if isinstance(v, str) and _BARE_S3.search(v):
                    cell.value = _BARE_S3.sub("submitted Table S3", v)
                    key = (ws.title, cell.column_letter if ws.title == "raw_joined" else cell.coordinate)
                    n[key] = n.get(key, 0) + 1
    want = {("macro_F1", "G1"): 1, ("other_metrics", "G1"): 1, ("raw_joined", "AE"): 128}
    rep.check(n == want, f"bare 'Table S3' qualified as 'submitted Table S3': {n}")
    if n != want:
        raise SystemExit(f"S35: unexpected 'Table S3' cells {n}")


def _rename_sheets(final, wb, rep):
    for old, new in SHEET_RENAMES.get(final, {}).items():
        if old not in wb.sheetnames:
            raise SystemExit(f"{final}: sheet {old!r} not found ({wb.sheetnames})")
        wb[old].title = new
        rep.check(True, f"sheet '{old}' renamed '{new}'")


def _s26_fixes(wb, rep, ctx):
    import pandas as pd
    ws = wb["calibration_reliability_bins"]
    hdr = [c.value for c in ws[1]]
    ic, ia, ig = (hdr.index(h) + 1 for h in ("mean confidence", "accuracy", "gap (confidence − accuracy)"))
    ok, neg = True, 0
    for r in range(2, ws.max_row + 1):
        c, a, g = (ws.cell(r, i).value for i in (ic, ia, ig))
        if c is None or a is None:
            ok &= g is None
            continue
        ok &= abs(g - (c - a)) < 1e-15
        neg += g < 0
    rep.check(ok and neg > 0, f"calibration_reliability_bins: gap = mean confidence − accuracy (signed; {neg} negative bins)")
    if not (ok and neg > 0):
        raise SystemExit("S26: reliability gap column is not the signed difference")
    # model_comparison_2766: note below the table (one blank row), not inside it
    mc = wb["model_comparison_2766"]
    last = max(r for r in range(1, mc.max_row + 1) if any(c.value is not None for c in mc[r]))
    mc.cell(last + 2, 1).value = MC2766_NOTE
    rep.check(last == 7, f"model_comparison_2766: table rows 1–{last}; note written to A{last + 2}")
    # plurality_recomputed from the E1-KNN recalculation
    res = pd.read_csv(KNN_RECALC / "results.csv")
    want_rules = ["plurality_closest_rank", "plurality_ties_as_wrong", "bayesian_argmax", "historical_S26",
                  "historical_S26_bayesian"]
    cols = ["k", "rule", "accuracy", "macro_f1", "balanced_accuracy"]
    good = list(res.columns) == cols and list(res.rule) == want_rules * 2 and list(res.k) == [5] * 5 + [18] * 5
    rep.check(good, f"results.csv: columns {cols}, rules {want_rules} at k = 5 and 18")
    if not good:
        raise SystemExit("S26: unexpected results.csv layout")
    ws_p = wb.create_sheet("plurality_recomputed", wb.sheetnames.index("model_comparison_2766") + 1)
    ws_p.append([PLURALITY_NOTE])
    ws_p.append(cols)
    for rec in res.itertuples(index=False):
        ws_p.append([int(rec.k), rec.rule, float(rec.accuracy), float(rec.macro_f1), float(rec.balanced_accuracy)])
    ws_p.column_dimensions["B"].width = 26
    rep.check(True, "sheet plurality_recomputed: note row, header, 10 rows from knn_comparator_recalc_20261110/results.csv")
    ctx.setdefault("extra_inputs", []).append(KNN_RECALC / "results.csv")


def _s27_s29_fixes(final, wb, rep, ctx):
    import numpy as np
    import pandas as pd
    k, sc_final, g1_exp, g2_exp = G_EXPECTED[final]
    ws = wb["Sheet1"]
    hdr = [c.value for c in ws[1]]
    # Sheet2 / Sheet3: same header as Sheet1 (without Sheet1's unnamed index column, which they do not carry)
    for name in ("Sheet2", "Sheet3"):
        s = wb[name]
        if s.cell(1, 1).value != "slides":
            if s.max_column != len(hdr) - 1 or not str(s.cell(1, 1).value).endswith(".svs"):
                raise SystemExit(f"{final} {name}: unexpected layout ({s.max_column} columns)")
            s.insert_rows(1)
            for j, h in enumerate(hdr[1:], 1):
                s.cell(1, j).value = h
        rep.check([c.value for c in s[1]] == hdr[1:], f"{name}: header row = Sheet1 header ({len(hdr) - 1} columns, "
                                                      f"{s.max_row - 1} data rows)")
    col = {h: i for i, h in enumerate(hdr)}
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    slides = [r[col["slides"]] for r in rows]
    y = np.array([r[col["disease"]] for r in rows])
    knn = np.array([_plurality_closest_rank([r[col[f"N{i}"]] for i in range(k)]) for r in rows])
    sc_src = SOURCES[sc_final][1]
    sc = pd.read_excel(sc_src, sheet_name="Sheet1")
    if list(sc["slides"]) != slides:
        raise SystemExit(f"{final}: slide order differs from {sc_final} Sheet1")
    scols = [c for c in sc.columns if c not in ("Unnamed: 0", "slides")]
    bay = sc[scols].idxmax(axis=1).values
    g1 = (bay == y) & (knn != y)
    g2 = (knn == y) & (bay != y)
    ok = (int(g1.sum()), int(g2.sum())) == (g1_exp, g2_exp)
    rep.check(ok, f"G1_recomputed / G2_recomputed = {int(g1.sum())} / {int(g2.sum())} (expected {g1_exp} / {g2_exp}; "
                  f"plurality over N0..N{k - 1}, closest-rank tie-break; Bayesian argmax from {sc_final} Sheet1)")
    if not ok:
        raise SystemExit(f"{final}: G1/G2 = {int(g1.sum())}/{int(g2.sum())}, expected {g1_exp}/{g2_exp}")
    c1, c2 = len(hdr) + 1, len(hdr) + 2
    ws.cell(1, c1).value, ws.cell(1, c2).value = "G1_recomputed", "G2_recomputed"
    for i, (a, b) in enumerate(zip(g1, g2), 2):
        ws.cell(i, c1).value = "yes" if a else "no"
        ws.cell(i, c2).value = "yes" if b else "no"
    ctx.setdefault("extra_inputs", []).append(sc_src)


def _s33_fixes(wb, rep, ctx):
    ws = wb["mapping"]
    hdr = [c.value for c in ws[1]]
    cnt = json.loads(COHORT_COUNTS.read_text())["ccdi"]
    count_cols = list(range(6, 11))
    sums = [sum(int(ws.cell(r, c).value or 0) for r in range(2, ws.max_row + 1)) for c in count_cols]
    totals = [int(re.search(r"\(([\d,]+)\)$", hdr[c - 1]).group(1).replace(",", "")) for c in count_cols]
    diff = [t - s for t, s in zip(totals, sums)]
    want = [cnt["populations_4160"]["NO_METADATA"], cnt["populations_4154"]["NO_METADATA"], 0, 0, 0]
    ok = (totals == [cnt["staged_with_features"], cnt["evaluated_nonempty"], cnt["in_vocabulary"]["slides"],
                     cnt["ffpe_546"], cnt["final"]["slides"]]
          and diff == want == [236, 231, 0, 0, 0] and sums[0] == cnt["linked_to_participant"])
    rep.check(ok, f"mapping column sums {sums} vs header totals {totals}: differences {diff} equal the cohort-accounting "
                  f"NO_METADATA counts {want[:2]} (r3_3 v3 counts.json; linked to a participant {cnt['linked_to_participant']})")
    if not ok:
        raise SystemExit(f"S33: count verification failed (sums {sums}, totals {totals}, NO_METADATA {want})")
    r = ws.max_row + 1
    ws.cell(r, 1).value = "No diagnosis string recorded"
    for c, v in zip(count_cols, diff):
        ws.cell(r, c).value = v
    ctx.setdefault("extra_inputs", []).append(COHORT_COUNTS)


def pass028_fixes(final: str, wb, rep, ctx) -> None:
    _fix_formulas(final, wb, rep)
    _rename_sheets(final, wb, rep)
    _fix_xrefs(final, wb, rep)
    if final == "S26":
        _s26_fixes(wb, rep, ctx)
    if final in G_EXPECTED:
        _s27_s29_fixes(final, wb, rep, ctx)
    if final == "S33":
        _s33_fixes(wb, rep, ctx)
    if final == "S35":
        _s35_submitted_s3(wb, rep)
    left = [(ws.title, c.coordinate) for ws in wb.worksheets for row in ws.iter_rows() for c in row if c.data_type == "f"]
    rep.check(not left, f"no formula cells ({left[:5]})")
    old_names = [s for s in wb.sheetnames
                 if (m := re.match(r"^S(\d+)", s)) and f"S{m.group(1)}" != final or re.search(r"table_s\d+", s, re.I)]
    rep.check(not old_names, f"no sheet name carrying another table number ({old_names})")


HANDLERS = {"s1": h_s1, "s4": h_s4, "s11": h_s11, "s15": h_s15, "s25": h_s25, "s26": h_s26, "s33": h_s33}


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="final supplementary-table package")
    ap.add_argument("--reader-tables", type=Path, default=None,
                    help="directory written by build_reader_study_tables.py; its workbooks replace the pending slots")
    args = ap.parse_args()
    reader_manifest = None
    if args.reader_tables:
        reader_manifest = json.load(open(args.reader_tables / "manifest.json"))
    if TABLES.exists():
        shutil.rmtree(TABLES)
    TABLES.mkdir(parents=True)
    if CONTROLLED.exists():
        shutil.rmtree(CONTROLLED)
    si_sha = pc.sha256(SI_MD)
    ents = si_entries()
    numbering = json.load(open(NUMBERING))["map"]
    bind = pc.Bindings()
    bind.bind(SI_MD, "Supplementary Information Markdown (caption source)")
    bind.bind(NUMBERING, "numbering map")
    manifest, report_rows, controlled = {}, [], []
    sj_slide = re.compile(r"^SJ[A-Z]{2,6}\d{6}_[A-Z]\d+$")
    sj_any = re.compile(r"\bSJ[A-Z]{2,6}\d{3,6}[A-Z0-9_]*\b")
    for n in range(1, 49):
        final = f"S{n}"
        if final in PENDING_FREEZE and reader_manifest is None:
            wid = PENDING_FREEZE[final]
            assert numbering[wid]["pass026"] == final
            manifest[final] = {"workspace_id": wid, "status": "pending 13 Nov freeze (reader-study table)",
                               "si_line": ents[final]["line"], "title": ents[final]["title"]}
            report_rows.append((final, wid, "pending freeze", "", ""))
            continue
        if final in PENDING_FREEZE:
            wid = PENDING_FREEZE[final]
            assert reader_manifest["tables"][final]["workspace_id"] == wid, (final, wid)
            src, handler = args.reader_tables / reader_manifest["tables"][final]["file"], None
            reader_label = reader_manifest["label"]
        else:
            wid, src, handler = SOURCES[final]
            reader_label = None
        assert numbering[wid]["pass026"] == final, (final, wid, numbering[wid])
        e = ents[final]
        assert e["caption"], (final, "no SI caption")
        bind.bind(src, f"source workbook for {final} ({wid})")
        rep = pc.Report()
        ctx: dict = {}
        wb = openpyxl.load_workbook(src)
        for name in ("caption", "Notes"):
            if name in wb.sheetnames:
                wb.remove(wb[name])
        if handler:
            HANDLERS[handler](wb, rep, ctx)
        pass028_fixes(final, wb, rep, ctx)
        for p in ctx.get("extra_inputs", []):
            bind.bind(p, f"{final} extra input")
        counter: dict = {}
        sj_hits = []
        for ws in wb.worksheets:
            # verbatim reader comments are never edited (AGENTS.md): columns headed comment_* keep their spelling
            hdr0 = [c.value for c in ws[1]] if ws.max_row else []
            verbatim_cols = {i + 1 for i, h in enumerate(hdr0) if isinstance(h, str) and h.startswith("comment")}
            for row in ws.iter_rows():
                for cell in row:
                    v = cell.value
                    if isinstance(v, str):
                        nv = v if cell.column in verbatim_cols else americanize(v, counter)
                        if nv != v:
                            cell.value = nv
                        for tok in sj_any.findall(nv):
                            if not sj_slide.match(tok):
                                sj_hits.append((ws.title, tok))
        rows = caption_rows(final, e, wid, src, bind.sha(src), si_sha)
        pc.write_caption_sheet(wb, rows)
        out = TABLES / f"Supplementary_Table_{final}.xlsx"
        pc.save_xlsx(wb, out)
        sj_note = ""
        if sj_hits:
            uniq = sorted({t for _, t in sj_hits})
            sj_note = f"{len(uniq)} non-slide SJ tokens: {uniq[:5]}"
        rep.check(not sj_hits, f"no St. Jude patient-style identifiers in cell text ({sj_note or 'none'})")
        manifest[final] = {"workspace_id": wid, "file": out.name, "sha256": pc.sha256(out),
                           "source": pc.rel(src), "source_sha256": bind.sha(src), "si_line": e["line"],
                           **({"reader_study_build": reader_label} if reader_label else {}),
                           "title": e["title"], "sheets": wb.sheetnames, "spelling_replacements": counter,
                           "checks": rep.summary(), "open_items": rep.open_items}
        report_rows.append((final, wid, rep.summary(), "; ".join(f"{k} ×{v}" for k, v in sorted(counter.items())),
                            " | ".join(f"{s}: {t}" for s, t in rep.checks) + (" || notes: " + " ".join(rep.notes) if rep.notes else "")
                            + (" || OPEN: " + " ".join(rep.open_items) if rep.open_items else "")))
        controlled += ctx.get("controlled", [])
        print(final, wid, rep.summary(), sum(counter.values()), "spelling edits")
    bind.write(PKG, {"build_date": pc.BUILD_DATE, "script": "scripts/supp_tables/assemble_final_tables_20261110.py"})
    (PKG / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    lines = ["# Final supplementary-table package (pass 026b, 2026-11-10)", "",
             f"Caption source: `docs/prosper_revised/prosper_supplementary_legends.md` sha256 {si_sha}.", "",
             "| Table | Workspace id | Checks | Spelling edits | Detail |", "|---|---|---|---|---|"]
    for r in report_rows:
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    lines += ["", "## Controlled files (not part of the release)", ""] + [f"- `{pc.rel(p)}`" for p in controlled]
    (PKG / "BUILD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reader_note = (f"this build used: {reader_manifest['label']} (`{reader_manifest['analysis']}`)." if reader_manifest
                   else "this build has no reader-study directory; those tables are listed as pending in `manifest.json`.")
    readme = f"""# Supplementary tables, final numbering (pass 026b, assembled {pc.BUILD_DATE})

`supplementary_tables/Supplementary_Table_S<n>.xlsx`, one workbook per table, first sheet `caption` repeating
the Supplementary Information caption and notes. Numbering follows `scripts/figures/supp_table_numbering.json`
(field `pass026`). Tables S31, S32 and S39–S47 (reader study) come from `build_reader_study_tables.py` on one
five-reader analysis directory (`--reader-tables`); {reader_note}

`controlled/` holds the pseudonym map of Table S11 Sheet1_units (St. Jude patient identifiers); it is retained
for the authors and is not released. Released cells carry internal slide indices and de-identified slide names
only. Packaging is not permission to publish; see AGENTS.md.

Build: `scripts/supp_tables/assemble_final_tables_20261110.py`; inputs in `input_bindings.json`; per-table checks in
`BUILD_REPORT.md`; hashes in `SHA256SUMS.txt`.

Submission form (journal format, 2026-11-10): the per-table workbooks here are the source form; what is uploaded is
`supplementary_data/Supplementary_Data_1..6.xlsx` (the 47 tables grouped by theme, built from these workbooks by
`scripts/supp_tables/build_supplementary_data_workbooks.py`; see `supplementary_data/README.md` and
`FILE_DESCRIPTIONS.md`) plus the Supplementary Information PDF, which reproduces the display sheets of the tables
listed under `pdf_tables` in `scripts/supp_tables/supplementary_data_layout.json` and carries caption, notes and a
"Workbook:" pointer for every table.

The Supplementary Information document (`prosper_supplementary_information_<sha8>.docx` / `.pdf`, sha8 = first eight
hex digits of the SI Markdown hash; `render_record_supplement.json`) is written into this directory by
`scripts/render_supplement_docx.py --verify` after the tables; run order: this script, then
`build_supplementary_data_workbooks.py`, then the renderer, so that `SHA256SUMS.txt` lists everything.
"""
    (PKG / "README.md").write_text(readme, encoding="utf-8")
    pc.write_sha256sums(PKG)
    print("wrote", PKG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
