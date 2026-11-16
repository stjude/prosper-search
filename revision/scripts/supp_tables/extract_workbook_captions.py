#!/usr/bin/env python3
"""Collect the caption/legend text of the built supplementary-table workbooks.

Reads (read-only) the `caption` sheets of the built supplementary-table
workbooks, the `Notes` sheet of the 2026-10-21 S-R2.1-E/S-R2.1-C workbook and
the submitted captions in `supplementary_info_v04.docx`, and writes, under
`$PROSPER_OUT/supp_assembly_20261109/`:

  captions_from_workbooks.json   machine-readable, keyed by workspace id
  captions_from_workbooks.md     one section per table in final-number order
  EXTRACTION_REPORT.md           inputs, field choices, counts, unresolved items
  SHA256SUMS.txt                 every input read and every output written

Prose fields are copied verbatim except for (i) American spelling
normalisation, (ii) replacement of spaced em dashes outside quoted/code spans
and (iii) translation of cross-references from workspace ids to the current
final numbers (scripts/figures/supp_{table,figure}_numbering.json). The
as-written text is kept beside every transformed field.

Usage:
    export PROSPER_OUT="$PWD/prosper_revision_output"
    python3 scripts/supp_tables/extract_workbook_captions.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import openpyxl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import docx_to_markdown  # noqa: E402

OUT_NAME = "supp_assembly_20261109"
TABLE_MAP_PATH = REPO / "scripts/figures/supp_table_numbering.json"
FIGURE_MAP_PATH = REPO / "scripts/figures/supp_figure_numbering.json"
SUPP_INFO_DOCX = REPO / "docs/prosper_submission/supplementary/supplementary_info_v04.docx"

# Symbolic aliases used in October captions, mapped to the workspace id of the
# numbering map. Source: the 'provisional numbers' / 'Table ID' rows of the
# workbooks and [internal record].
TABLE_ALIASES = {
    "S-A1": "S23", "S-A2": "S24", "S-A3": "S25",
    "S-D1": "S26", "S-D2": "S28",
    "S-T1": "S22", "S-T2": "S27",
    "S-R4a-1": "S-R4.a-1", "S-R4a-2": "S-R4.a-2",
}

SUBMITTED_TABLES = ["S1", "S2", "S9", "S10", "S11", "S12", "S13"]
SUBMITTED_FIGURES = ["S4"]

# --------------------------------------------------------------------------
# Per-workbook field roles. Each source spec is one of
#   ("field", name)              the single row with this field name
#   ("lines", name, mode)        all rows with this field name, joined
#                                mode: "space" | "md" (strip '> ', keep lists)
#   ("line_n", prefix)           rows '<prefix> N', N ascending, space-joined
#   ("regex", spec, pattern)     group 1 of `pattern` applied to the joined spec
# --------------------------------------------------------------------------
A20 = "supp_tables_20261022/A_ladder_failure_finelabel"
A03 = "supp_tables_20261103/A_ladder_failure_finelabel"
B20 = "supp_tables_20261022/B_external_leakage_titan"
C20 = "supp_tables_20261022/C_encoders_comparators"
D21 = "supp_tables_20261021"

ENTRIES = [
    dict(
        wid="S29", files=[f"{A20}/R2.2-T4_S29.xlsx"],
        title=("field", "title"),
        caption=("field", "caption (verbatim, edit-sheet line 3461 ff.)"),
        footnotes=[("†", ("field", "footnote † (verbatim)")),
                   ("sheet e", ("field", "sheet e footnote (spec table row e)"))],
        choice="title = 'title'; caption = 'caption (verbatim, edit-sheet line 3461 ff.)' "
               "(identical in content to 'caption line 1-28'; lines 29-54 are the † footnote); "
               "footnotes = 'footnote † (verbatim)' and 'sheet e footnote (spec table row e)'. "
               "Not taken: 'sheet e OT 30 % solver qualification' and 'sheet e OT 100 % row' "
               "(spec text for table cells), build metadata.",
        checks=["The 'sheet e OT 30 % solver qualification' row (spec text for a sheet-e cell) "
                "is not caption prose and was not extracted; confirm it is printed in the sheet, not the legend."],
    ),
    dict(
        wid="S23", files=[f"{A20}/R2.3-T2_S23_S-A1.xlsx"],
        title="first_sentence",
        caption=("field", "title/legend (verbatim, joined)"),
        footnotes=[],
        choice="caption = 'title/legend (verbatim, joined)' (same text as 'legend line 1-15'); "
               "title = its first sentence. The '†' field is a builder note ('see legend'); the † "
               "definition is inside the legend, so no separate footnote is listed.",
        bare_refs=["S23"],
    ),
    dict(
        wid="S24", files=[f"{A20}/R2.3-T3_S24_S-A2.xlsx"],
        title="first_sentence",
        caption=("field", "legend (verbatim, joined)"),
        footnotes=[],
        choice="caption = 'legend (verbatim, joined)' (same text as 'legend line 1-5'); title = its first sentence.",
        checks=["Sheet 'S24c top-1 (optional)' has no legend text in the spec (workbook note: "
                "'author's assembly choice'); decide whether to keep the sheet and, if so, add a legend sentence."],
    ),
    dict(
        wid="S30", files=[f"{A20}/R3.2-T1_S30.xlsx"],
        title=("field", "title"),
        caption=("field", "caption (verbatim, joined)"),
        footnotes=[("footnote", ("field", "footnote (F1 APPLIED, clean)"))],
        choice="title = 'title'; caption = 'caption (verbatim, joined)'; footnote = 'footnote (F1 APPLIED, clean)'. "
               "Not taken: the two superseded footnote versions (edit-sheet markup; frozen pre-F1).",
        bare_refs=["S23"],
        checks=["The 'title' field is truncated in the workbook (ends '…random sampling of five library'); "
                "the remainder of the title sentence is not in the workbook.",
                "Optional R3.2 audit F2 (caption 'S18, S19 and S23' → 'S18 and S23'; 'resample within each group') "
                "is recorded as NOT applied (author option, undecided).",
                "After translation 'Supplementary Tables S18, S19 and S23' reads 'S17, S22 and S5' (source order kept; "
                "not ascending)."],
    ),
    dict(
        wid="S26", files=[f"{A03}/R3.7-T4_S26_S-D1_v2.xlsx"],
        superseded_files=[f"{A20}/R3.7-T4_S26_S-D1.xlsx"],
        title="first_sentence",
        caption=("field", "legend (APPLIED, v2)"),
        split_inline_footnote="Footnote: ",
        footnotes=[],
        choice="v2 workbook (2026-11-03) supersedes the 10-22 workbook; caption = 'legend (APPLIED, v2)'; "
               "title = its first sentence; the inline 'Footnote: …' tail is split into the footnote list.",
    ),
    dict(
        wid="S28", files=[f"{A20}/R3.7-T5_S28_S-D2.xlsx"],
        title=("field", "title"),
        caption_parts=[("field", "introductory sentence (verbatim, task_inventory.md line 3)"),
                       ("field", "legend (APPLIED: C1 = R3.7 F1 option (a) insertion)")],
        caption_additions=[("legend location note (pending: 'add to the legend', reviewer N3)",
                            ("field", "legend location note (add to the legend; reviewer N3; verbatim from edit sheet)"))],
        footnotes=[],
        choice="title = 'title'; caption = 'introductory sentence' + 'legend (APPLIED: C1 …)' (paragraphs; "
               "Markdown emphasis kept as in the source); the 'legend location note' (to be added to the "
               "legend per reviewer N3) is kept separately as a caption addition. Not taken: the pre-C1 "
               "legend, the option (a) sentence (already inside the applied legend), the source line.",
        checks=["Caption addition 'legend location note' still contains internal ids (R3.7-T1, R3.7-T4) and an "
                "instruction '(D-1 (c): do not give an assay- or morphology-based reason)'; reword before printing.",
                "Legend says 'Saved S6 depth-10 rank lists' and 'Machine-readable counts (num_* columns) are in "
                "task_inventory.csv'; 'S6' is an analysis-stage id, not a table (left as written)."],
        bare_non_refs={"S6": "analysis-stage id ('Saved S6 depth-10 rank lists')"},
    ),
    dict(
        wid="S-R4.a-1", files=[f"{B20}/Table_S-R4a-1_masked_vs_unmasked.xlsx"],
        title=("regex", ("lines", "table title + specification (verbatim)", "space"),
               r"\*\*New Supplementary Table S-R4\.a-1 — (.*?)\*\*"),
        caption=None,
        footnotes=[("footnotes", ("regex", ("lines", "table title + specification (verbatim)", "space"),
                                  r'Footnotes \(verbatim\): "(.*?)"'))],
        spec=("lines", "table title + specification (verbatim)", "md"),
        choice="No caption prose. title = the bold title in 'table title + specification (verbatim)' "
               "(text after the em dash); footnotes = the quoted 'Footnotes (verbatim)' text of the same field. "
               "The rest of that field (block specification) is kept as spec_text for reference, not as caption.",
        checks=["No caption prose in the workbook (title and footnotes only); the legend has to be written.",
                "Block B footnote/spec mismatch recorded by the builder: strict k = 3 left-in value spec 0.782 vs "
                "source 0.781 (unresolved).",
                "Edit-sheet block status recorded as 'review pending' (drafted 10-15)."],
    ),
    dict(
        wid="S-R4.a-2", files=[f"{B20}/Table_S-R4a-2_same_patient_consistency.xlsx"],
        title=("regex", ("lines", "table title + specification (verbatim)", "space"),
               r"\*\*New Supplementary Table S-R4\.a-2 — (.*?)\*\*"),
        caption=None,
        footnotes=[("footnotes", ("regex", ("lines", "table title + specification (verbatim)", "space"),
                                  r'Footnotes \(verbatim\): "(.*?)"'))],
        spec=("lines", "table title + specification (verbatim)", "md"),
        choice="No caption prose. title = the bold title in 'table title + specification (verbatim)'; "
               "footnotes = its quoted 'Footnotes (verbatim)' text; remainder kept as spec_text.",
        checks=["No caption prose in the workbook (title and footnotes only); the legend has to be written.",
                "Edit-sheet block status recorded as 'review pending' (drafted 10-15)."],
    ),
    dict(
        wid="S18", files=[f"{B20}/Table_S18_primary_broad_group.xlsx"],
        title="first_sentence",
        caption=("lines", "caption (verbatim)", "md"),
        footnotes=[],
        choice="caption = the 'caption (verbatim)' rows (Markdown block quote, '> ' stripped, soft wraps joined); "
               "title = its first sentence. Spec/'Where'/'Acceptance'/assembly rows not taken.",
        checks=["Missing space in the source caption: 'resampling units;95% intervals' (also in the edit sheet "
                "l.1535); left as written."],
    ),
    dict(
        wid="S19", files=[f"{B20}/Table_S19_configuration_filtering_coverage.xlsx"],
        title="first_sentence",
        caption=("lines", "caption (verbatim)", "md"),
        footnotes=[],
        choice="caption = 'caption (verbatim)' rows (block quote joined); title = first sentence.",
    ),
    dict(
        wid="S20", files=[f"{B20}/Table_S20_failures_neighbours_transitions.xlsx"],
        title="first_sentence",
        caption=("lines", "caption (verbatim)", "md"),
        footnotes=[],
        choice="caption = 'caption (verbatim)' rows (block quote joined); title = first sentence.",
    ),
    dict(
        wid="S21", files=[f"{B20}/Table_S21_descriptive_strata.xlsx"],
        title="first_sentence",
        caption=("lines", "caption (verbatim)", "md"),
        footnotes=[],
        choice="caption = 'caption (verbatim)' rows (block quote joined); title = first sentence.",
        checks=["Missing spaces before numbers in the source caption ('comprise59', 'and354', 'all413', "
                "'The735-query', 'comprises524', 'FFPE,201'; also in the edit sheet l.1558); left as written."],
    ),
    dict(
        wid="S22", files=[f"{B20}/Table_S22_TITAN_tumour_filtering.xlsx"],
        title="first_sentence",
        caption=("lines", "legend (verbatim)", "md"),
        panels=[("Panel A", ("lines", "Panel A heading (verbatim)", "space")),
                ("Panel B", ("lines", "Panel B heading (verbatim)", "space")),
                ("Panel C", ("lines", "Panel C heading (verbatim)", "space")),
                ("Panel D", ("lines", "Panel D heading (verbatim)", "space"))],
        footnotes_numbered=("lines", "table notes (verbatim; to print under S22)", "md"),
        footnotes=[],
        choice="caption = 'legend (verbatim)' rows (block quote stripped; bullet structure kept); title = first "
               "sentence; panel headings = 'Panel A-D heading (verbatim)'; footnotes = the five numbered "
               "'table notes (verbatim; to print under S22)'. Not taken: 'spec numbers source (not printed)', assembly notes.",
        bare_refs=["S22"],
        bare_non_refs={"S2": "filter arm name (mapped 512×512 tile predictions)"},
        checks=["Table note 4 cites 'the manuscript (MS L181)', an internal line reference; reword.",
                "Panel A and C headings contain build provenance (source CSV names, bundle checks, "
                "`common_pool_2562_intersection`); strip before printing.",
                "Builder note: several note 2/3/5 numbers and the panel D group assignment are 'not verified' "
                "against the bound sources (see workbook assembly notes)."],
    ),
    dict(
        wid="S27", files=[f"{B20}/Table_S27_TITAN_unannotated_queries.xlsx"],
        title="first_sentence",
        caption=("lines", "legend (verbatim)", "md"),
        panels=[("Panel A", ("lines", "Panel A heading (verbatim)", "space")),
                ("Panel A note", ("lines", "Panel A note (verbatim)", "space")),
                ("Panel B", ("lines", "Panel B heading (verbatim)", "space"))],
        footnotes=[("ᵃ", ("lines", "note a (verbatim)", "space"))],
        choice="caption = 'legend (verbatim)' rows; title = first sentence; panels = 'Panel A heading', "
               "'Panel A note', 'Panel B heading'; footnote ᵃ = 'note a (verbatim)'. Not taken: 'spec Source', "
               "'spec Population', 'spec Notes (provenance, not all for print)', assembly notes.",
        bare_refs=["S22", "S27"],
        bare_non_refs={"S2": "filter arm name"},
        checks=["Legend uses the code token `leak_free_scope` (a column name); confirm it is printed as a column name.",
                "Panel A/B headings and the Panel A note contain source-CSV provenance (`summary_all_bundles.csv`, "
                "`per_group_all_bundles.csv`, row filters); strip before printing."],
    ),
    dict(
        wid="S-R2.1-A", files=[f"{C20}/Table_S-R2.1-A.xlsx"],
        title=None,
        caption=None,
        footnotes=[("10-16 amendment", ("field", "Footnote (10-16 amendment, verbatim)"))],
        spec=("prefix_lines", "edit-sheet l.", "md"),
        choice="No caption prose: the caption sheet holds only the R2.1-T4 edit-sheet block (spec text and "
               "superseded operations, kept as spec_text) and one footnote ('Footnote (10-16 amendment, verbatim)').",
        checks=["No caption prose or title in the workbook; the legend has to be written from the edit-sheet "
                "specification (R2.1-T4, l.6149-6154).",
                "Edit-sheet text in this workbook (R2.1-T4 l.6136-6138) instructs appending a sentence to the "
                "submitted S1/S2 captions citing 'Table S3' (now removed); not applied to S1/S2 here.",
                "Number S48 is assigned because the table is cited only in the response letter (map lead note)."],
    ),
    dict(
        wid="S-R2.1-B", files=[f"{C20}/Table_S-R2.1-B.xlsx"],
        title="first_sentence",
        alt_title=("field", "tables_S_R2_1_B_D.md l.7"),
        caption=("regex", ("field", "tables_S_R2_1_B_D.md l.9"), r"^\*\*Caption \(candidate\)\.\*\* (.*)$"),
        footnotes=[("Rounding", ("regex", ("field", "tables_S_R2_1_B_D.md l.11"), r"^\*Rounding\.\* (.*)$")),
                   ("Part 2 note", ("field", "tables_S_R2_1_B_D.md l.65")),
                   ("Part 4 note", ("field", "tables_S_R2_1_B_D.md l.88")),
                   ("Part 5 note", ("field", "tables_S_R2_1_B_D.md l.245")),
                   ("Part 6 note", ("field", "tables_S_R2_1_B_D.md l.276"))],
        panels=[("Part 1", ("regex", ("field", "tables_S_R2_1_B_D.md l.13"), r"^### Part 1\. (.*)$")),
                ("Part 2", ("regex", ("field", "tables_S_R2_1_B_D.md l.46"), r"^### Part 2\. (.*)$")),
                ("Part 3", ("regex", ("field", "tables_S_R2_1_B_D.md l.67"), r"^### Part 3\. (.*)$")),
                ("Part 4", ("regex", ("field", "tables_S_R2_1_B_D.md l.86"), r"^### Part 4\. (.*)$")),
                ("Part 5", ("regex", ("field", "tables_S_R2_1_B_D.md l.205"), r"^### Part 5\. (.*)$")),
                ("Part 6", ("regex", ("field", "tables_S_R2_1_B_D.md l.247"), r"^### Part 6\. (.*)$"))],
        choice="caption = 'tables_S_R2_1_B_D.md l.9' with the '**Caption (candidate).**' label removed; title = its "
               "first sentence; alt_title = the l.7 Markdown heading; footnotes = rounding note (l.11) and the part "
               "notes (l.65, l.88, l.245, l.276); panels = part headings.",
        bare_non_refs={"S6": "analysis id ('S6-c' re-evaluation)"},
        checks=["Caption is labelled 'Caption (candidate)' in the source.",
                "Caption cites 'classes (i)–(v), contract §6' and the CSV release `S_R2_1_B_*.csv` / `contrasts.csv`; "
                "internal references to reword."],
    ),
    dict(
        wid="S-R2.1-D", files=[f"{C20}/Table_S-R2.1-D.xlsx"],
        title="first_sentence",
        alt_title=("field", "tables_S_R2_1_B_D.md l.278"),
        caption=("regex", ("field", "tables_S_R2_1_B_D.md l.280"), r"^\*\*Caption \(candidate\)\.\*\* (.*)$"),
        caption_additions=[("Block 2 caption addendum ((8) + (8′))",
                            ("field", "Block 2 caption addendum ((8) + (8′))"))],
        footnotes=[("Rounding", ("regex", ("field", "tables_S_R2_1_B_D.md l.282"), r"^\*Rounding\.\* (.*)$")),
                   ("Part 3 note", ("field", "tables_S_R2_1_B_D.md l.370")),
                   ("Part 4 note", ("field", "tables_S_R2_1_B_D.md l.374"))],
        panels=[("Part 1", ("regex", ("field", "tables_S_R2_1_B_D.md l.284"), r"^### Part 1\. (.*)$")),
                ("Part 2", ("regex", ("field", "tables_S_R2_1_B_D.md l.317"), r"^### Part 2\. (.*)$")),
                ("Part 3", ("regex", ("field", "tables_S_R2_1_B_D.md l.350"), r"^### Part 3\. (.*)$")),
                ("Part 4", ("regex", ("field", "tables_S_R2_1_B_D.md l.372"), r"^### Part 4\. (.*)$"))],
        choice="caption = 'tables_S_R2_1_B_D.md l.280' without the '**Caption (candidate).**' label; title = first "
               "sentence; alt_title = l.278 heading; caption addition = 'Block 2 caption addendum ((8) + (8′))'; "
               "footnotes = rounding note (l.282) and part notes (l.370, l.374); panels = part headings.",
        checks=["Caption is labelled 'Caption (candidate)' in the source.",
                "Caption says 'Replaces the provisional content of S33' — S33 is this table's own earlier provisional "
                "number (docs/analysis/supplementary_numbering_map_20261022.md); left as written, reword.",
                "Caption and Part 4 note say the submitted Table S5 is 'retained unchanged' / 'repeated'; submitted "
                "Table S5 was removed in pass 016, so these statements are no longer true; reword.",
                "Part 4 note cites 'contract candidate 003 §0, §10' and frozen package paths; internal references."],
        bare_non_refs={"S33": "earlier provisional number of this table (see LEAD CHECK)"},
    ),
    dict(
        wid="S-R2.1-E", files=[f"{C20}/Table_S-R2.1-E_sheets1-4.xlsx",
                               f"{D21}/Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx"],
        sheet_filter={f"{D21}/Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx": "S-R2.1-E"},
        title=None,
        caption=None,
        sheet_captions=[("Sheet 5", ("notes_row", f"{D21}/Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx", 1)),
                        ("Sheet 5 (systems)", ("notes_row", f"{D21}/Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx", 3)),
                        ("Sheet 6", ("notes_row", f"{D21}/Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx", 4))],
        footnotes_numbered=("range_lines", "edit-sheet l.", 6188, 6208, "md"),
        footnotes=[("6", ("regex", ("field", "edit-sheet l.7776"), r'Footnote 6: "(.*?)"')),
                   ("7", ("regex", ("field", "edit-sheet l.7776"), r'Footnote 7: "(.*?)"')),
                   ("7 (Notes sheet wording)",
                    ("regex", ("notes_row", f"{D21}/Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx", 2),
                     r"^Footnote 7: (.*)$"))],
        spec=("prefix_lines", "edit-sheet l.", "md"),
        bare_refs=["S-R2.1-E"],
        choice="No table-level caption prose: sheets 1-4 caption sheet holds the R2.1-T12 edit-sheet block "
               "(sheet specifications kept as spec_text). footnotes 1-5 = edit-sheet l.6188-6208 (numbered list); "
               "footnotes 6-7 = the quoted texts in edit-sheet l.7776; the 10-21 workbook 'Notes' rows 1, 3, 4 are "
               "sheet-5/6 captions and row 2 is footnote 7 in slightly different wording (both kept).",
        checks=["No table-level caption or title in either workbook; sheet 1-4 captions have to be written from "
                "the R2.1-T12 specification (spec_text).",
                "Edit-sheet l.7776 asks to move 'the TITAN macro-F1 \"best at k = 5 and 10\" statement from MS L41 "
                "into the Sheet 1 caption' — not present in the workbook.",
                "Edit-sheet l.7875 asks the sheet captions to state that mMV@5/MV@5 equal sheet 1's macro_recall at "
                "k = 5; the Notes-sheet captions do not say this.",
                "Footnote 7 exists in two wordings (edit sheet vs Notes sheet); choose one.",
                "Notes row 3 ('Sheet 5 (systems)') ends with build provenance ('Sources: … r2_1_fig2_retrieval_20261021/"
                "run_001 (gated against RR-5 and R3.1) …'); strip before printing.",
                "Footnote 1 cites 'SHA-256 in ADR0021', footnote 3 'ADR0010', footnote 2 'Methods as revised under "
                "R4.b'; internal references.",
                "Edit-sheet l.6210 'Numbering: provisional S35' (spec metadata, not caption).",
                "A merged S-R2.1-E workbook (supp_tables_20261109/A_packaging/S-R2.1-E/, "
                "scripts/supp_tables/build_S-R2.1-E_merged_20261109.py; footnote 3 open item) is not among this "
                "extraction's inputs; check whether its footnote 3 wording supersedes the one here."],
    ),
    dict(
        wid="S-R2.1-C", files=[f"{D21}/Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx"],
        sheet_filter={f"{D21}/Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx": "S-R2.1-C"},
        title=None,
        caption=None,
        sheet_captions=[("mMV@5 sheets", ("notes_row", f"{D21}/Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx", 5))],
        footnotes=[],
        bare_refs=["S-R2.1-C"],
        choice="Only the two mMV@5 sheets of S-R2.1-C are among the inputs (10-21 workbook); 'Notes' row 5 is their "
               "sheet caption. The main S-R2.1-C sheets and table caption are not in any input workbook.",
        checks=["Main S-R2.1-C table/caption is not in the input workbooks (only the mMV@5 sheets).",
                "The mMV@5 sheet caption contains build provenance ('Derived 2026-10-21 (scripts/r2_1_fig4a_mmv5.py, "
                "gates G1-G4 pass) …; lead distinct-route check …'); strip before printing."],
    ),
    dict(
        wid="S-R2.1-F", files=[f"{C20}/Table_S-R2.1-F.xlsx"],
        title=None,
        caption=None,
        footnotes_numbered=("regex", ("field", "Lineage footnotes (comparison_001, verbatim)"),
                            r"^# Footnotes to [^\n]*\n\n(.*)$"),
        footnotes=[("Abbreviations", ("field", "Abbreviations")),
                   ("Rounding", ("field", "Rounding"))],
        spec=("prefix_lines", "edit-sheet l.", "md"),
        choice="No caption prose: the edit-sheet R2.1-T12 amendment (l.6212) specifies what the caption must state "
               "(kept as spec_text). footnotes = 'Lineage footnotes (comparison_001, verbatim)' items 1-3, "
               "'Abbreviations', 'Rounding'. Not taken: materiality rule/disclosure, bootstrap rows (spec).",
        checks=["No caption prose or title in the workbook; the caption has to be written from the l.6212 amendment "
                "(median pooling, SAMPLER and OT unchanged at k = 5; Yottixel −0.003 [−0.009, +0.001] canonical, "
                "+0.009 reversed; materiality pre-registration not met).",
                "Lineage footnotes 1-3 cite 'author decision B (2026-10-13)' and 'contract M6'; footnote 2 says the "
                "tumour-only Virchow2 rows 'remain on the published bank until their separate package' — check current status."],
    ),
    dict(
        wid="S-R3.1", files=[f"{C20}/Table_S-R3.1.xlsx"],
        title="first_sentence_pipe",
        caption=("range_lines", "edit-sheet l.", 4320, 4331, "md"),
        footnotes_lettered=("range_lines", "edit-sheet l.", 4378, 4414, "md"),
        footnotes=[],
        choice="caption = edit-sheet l.4320-4331 (block quote joined); title = 'Table S-R3.1 | …' first segment; "
               "footnotes = l.4378-4414 (lettered a-k plus the unlettered neighbour-label item).",
        checks=["Title uses the pipe form 'Table S-R3.1 | …' rather than 'Supplementary Table Sxx. …'."],
    ),
    dict(
        wid="S25", files=[f"{C20}/Table_S25.xlsx"],
        title="first_sentence",
        caption=("range_lines", "edit-sheet l.", 2426, 2442, "md"),
        footnotes=[],
        choice="caption = edit-sheet l.2426-2442 (block quote joined); title = first sentence. 'Layout' row not taken.",
    ),
]

PROSE_KEYS = ("title", "caption", "caption_body", "alt_title")

# --------------------------------------------------------------------------
# Workbook reading
# --------------------------------------------------------------------------


class Workbook:
    def __init__(self, path: Path):
        self.path = path
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        self.sheet_names = list(wb.sheetnames)
        self.sheets = []
        self.rows = []
        self.notes = []
        for ws in wb.worksheets:
            vals = [r for r in ws.iter_rows(values_only=True)
                    if any(v is not None and str(v).strip() != "" for v in r)]
            self.sheets.append({"name": ws.title, "rows_including_header": len(vals)})
            if ws.title == "caption":
                self.rows = [(r[0], r[1] if len(r) > 1 else None) for r in vals[1:]]
            if ws.title == "Notes":
                self.notes = [r[0] for r in vals[1:]]
        wb.close()

    def get(self, field):
        hits = [t for f, t in self.rows if f == field]
        if len(hits) != 1:
            raise KeyError(f"{self.path.name}: field {field!r} found {len(hits)}x")
        return hits[0]

    def lines(self, field):
        hits = [t for f, t in self.rows if f == field]
        if not hits:
            raise KeyError(f"{self.path.name}: field {field!r} not found")
        return ["" if t is None else str(t) for t in hits]

    def prefix_lines(self, prefix, lo=None, hi=None):
        out = []
        for f, t in self.rows:
            if f is None or not str(f).startswith(prefix):
                continue
            m = re.match(re.escape(prefix) + r"(\d+)$", str(f))
            if not m:
                continue
            n = int(m.group(1))
            if (lo is None or n >= lo) and (hi is None or n <= hi):
                out.append("" if t is None else str(t))
        if not out:
            raise KeyError(f"{self.path.name}: no '{prefix}N' rows in [{lo}, {hi}]")
        return out


def join_md(lines):
    """Strip block-quote markers and join soft-wrapped lines; keep list items,
    blank-line paragraph breaks and headings on their own lines."""
    out = []
    for raw in lines:
        s = raw.rstrip()
        s = re.sub(r"^>\s?", "", s)
        stripped = s.strip()
        if stripped == "":
            if out and out[-1] != "":
                out.append("")
            continue
        is_item = bool(re.match(r"^(\s*)([-*]|\d+\.)\s", s)) or stripped.startswith("**") and not out
        indented_para = (bool(re.match(r"^\s{2,}[A-Z]", s)) and bool(out) and out[-1].rstrip().endswith((".", ":"))
                         and bool(re.match(r"^\s*([-*]|\d+\.)\s", out[-1])))
        if indented_para:
            out.append(s.rstrip())
        elif is_item or not out or out[-1] == "":
            out.append(s.rstrip() if is_item else stripped)
        else:
            out[-1] = out[-1] + " " + stripped
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def resolve(spec, wb: Workbook, books: dict):
    kind = spec[0]
    if kind == "field":
        v = wb.get(spec[1])
        return None if v is None else str(v).strip()
    if kind == "lines":
        ls = wb.lines(spec[1])
        return join_md(ls) if spec[2] == "md" else " ".join(x.strip() for x in ls if x.strip())
    if kind == "line_n":
        rows = [(int(m.group(1)), t) for f, t in wb.rows
                if f and (m := re.match(re.escape(spec[1]) + r" (\d+)$", str(f)))]
        return " ".join(str(t).strip() for _, t in sorted(rows))
    if kind == "prefix_lines":
        return join_md(wb.prefix_lines(spec[1]))
    if kind == "range_lines":
        ls = wb.prefix_lines(spec[1], spec[2], spec[3])
        return join_md(ls) if spec[4] == "md" else " ".join(x.strip() for x in ls)
    if kind == "notes_row":
        return str(books[spec[1]].notes[spec[2] - 1]).strip()
    if kind == "regex":
        base = resolve(spec[1], wb, books)
        m = re.search(spec[2], base, re.S)
        if not m:
            raise ValueError(f"{wb.path.name}: pattern {spec[2]!r} not found")
        return m.group(1).strip()
    raise ValueError(kind)


ABBREV = ("e.g.", "i.e.", "vs.", "Fig.", "Figs.", "al.", "No.", "approx.", "cf.")


def first_sentence(text):
    """First sentence at parenthesis depth 0, ignoring common abbreviations and
    the period that closes the 'Supplementary Table Sxx.' label."""
    depth = 0
    m0 = re.match(r"^(?:Supplementary\s+)?Table\s+S[\w.\-]*?\.\s", text)
    start = m0.end() if m0 else 0
    for i in range(start, len(text)):
        c = text[i]
        if c in "([":
            depth += 1
        elif c in ")]":
            depth = max(0, depth - 1)
        elif c == "." and depth == 0 and (i + 1 == len(text) or text[i + 1] in " \n"):
            if any(text[: i + 1].endswith(a) for a in ABBREV):
                continue
            return text[: i + 1], text[i + 1:].strip()
    return text, ""


def split_numbered(text, pattern=r"^(\d+)\.\s+"):
    items, cur = [], None
    for line in text.split("\n"):
        m = re.match(pattern, line)
        if m:
            cur = {"label": m.group(1), "text": line[m.end():].strip()}
            items.append(cur)
        elif cur is not None and line.strip():
            sep = "\n" if re.match(r"^\s*[-*]\s|^\s{2,}[A-Z]", line) else " "
            cur["text"] += sep + line.strip()
    return items


def split_lettered(text):
    items = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("**Footnotes"):
            continue
        m = re.match(r"^- ([a-k])\.\s+(.*)$", s)
        if m:
            items.append({"label": m.group(1), "text": m.group(2)})
        elif s.startswith("- "):
            items.append({"label": "(unlettered)", "text": s[2:]})
        elif items and s:
            items[-1]["text"] += " " + s
    return items


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

SPELLING = [
    (r"([Tt])umour", r"\1umor"),
    (r"([Cc])olour", r"\1olor"),
    (r"([Nn])eighbour", r"\1eighbor"),
    (r"([Ll])abell(ed|ing)", r"\1abel\2"),
    (r"([Ff])avour", r"\1avor"),
    (r"([Nn])ormalis(e|ed|es|ing|ation)\b", r"\1ormaliz\2"),
    (r"([Ss])ummaris(e|ed|es|ing|ation)\b", r"\1ummariz\2"),
    (r"\b([Aa])nalys(ed|ing|e)\b", r"\1nalyz\2"),
    (r"\b([Cc])entre(s?)\b", r"\1enter\2"),
]

PROTECT_RE = re.compile(r"`[^`]*`|'[^'\n]*—[^'\n]*'|\"[^\"\n]*—[^\"\n]*\"")


def _outside_protected(text, fn):
    out, pos = [], 0
    for m in PROTECT_RE.finditer(text):
        out.append(fn(text[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(fn(text[pos:]))
    return "".join(out)


def americanize(text):
    count = 0

    def fn(seg):
        nonlocal count
        for pat, rep in SPELLING:
            seg, n = re.subn(pat, rep, seg)
            count += n
        return seg

    return _outside_protected(text, fn), count


def replace_em_dashes(text):
    """' — ' outside code/quoted spans: a pair within one sentence is a
    parenthetical (commas); a single dash after a short leading label
    ('Panel A — x') or before an elaboration becomes '; '."""
    count = 0

    def fn(seg):
        nonlocal count
        parts = re.split(r"((?<=[.!?])\s+)", seg)
        for i in range(0, len(parts), 2):
            k = parts[i].count(" — ")
            if k == 2:
                parts[i] = parts[i].replace(" — ", ", ")
            elif k:
                head = parts[i].split(" — ", 1)[0].replace("*", "").strip()
                rep = ", " if len(head.split()) <= 3 else "; "
                parts[i] = parts[i].replace(" — ", rep)
            count += k
        return "".join(parts)

    if " — " not in text:
        return text, 0
    lines = text.split("\n")
    res = []
    for line in lines:
        if " — " in line:
            res.append(_outside_protected(line, fn))
        else:
            res.append(line)
    return "\n".join(res), count


# --------------------------------------------------------------------------
# Cross-references
# --------------------------------------------------------------------------

ID = r"S(?:-[A-Z][A-Za-z0-9]*(?:[.\-][A-Za-z0-9]+)*|\d+[a-z]?)"
ALIAS = r"(?:\s*\((?:S-[A-Z][A-Za-z0-9]*(?:[.\-][A-Za-z0-9]+)*)\))"
SEP = r"(?:\s*,\s*and\s+|\s*,\s*|\s+and\s+|\s*[–-]\s*|\s+to\s+)"
REF_RE = re.compile(
    r"\b(?P<prefix>(?:Supplementary\s+)?(?:Tables?|Figures?|Figs?\.)\s+)"
    r"(?P<ids>" + ID + ALIAS + r"?(?:" + SEP + ID + ALIAS + r"?)*)"
)
ID_RE = re.compile(r"(?P<sep>" + SEP + r")?(?P<id>" + ID + r")(?P<alias>" + ALIAS + r")?")


class Maps:
    def __init__(self, tmap, fmap):
        self.t = tmap["map"]
        self.f = fmap["map"]

    def lookup(self, kind, wid):
        base, suffix = wid, ""
        m = re.match(r"^(S\d+)([a-z])$", wid)
        if m:
            base, suffix = m.group(1), m.group(2)
        if kind == "table":
            base = TABLE_ALIASES.get(base, base)
            table = self.t
        else:
            table = self.f
        if base not in table:
            return None, base
        new = table[base]["new"]
        if new.startswith("removed"):
            return f"[REMOVED:{base}]", base
        mnum = re.match(r"^(S\d+)", new)
        if not mnum:
            return None, base
        return mnum.group(1) + suffix, base


def translate_refs(text, maps: Maps, self_wid, bare_refs=(), bare_non_refs=None, kind_default="table"):
    """Return (translated_text, refs, unresolved). Prefixed references are
    always translated; bare ids only if listed in `bare_refs`."""
    refs, unresolved = [], []
    spans = []

    def tr_ids(kind, ids_text):
        toks = list(ID_RE.finditer(ids_text))
        finals, ok = [], True
        i = 0
        while i < len(toks):
            t = toks[i]
            wid = t.group("id")
            if (i + 1 < len(toks) and toks[i + 1].group("sep")
                    and re.fullmatch(r"\s*[–-]\s*|\s+to\s+", toks[i + 1].group("sep"))
                    and re.fullmatch(r"S\d+", wid) and re.fullmatch(r"S\d+", toks[i + 1].group("id"))):
                a, b = int(wid[1:]), int(toks[i + 1].group("id")[1:])
                expanded = [f"S{n}" for n in range(a, b + 1)]
                i += 2
            else:
                expanded = [wid]
                i += 1
            for w in expanded:
                fin, canon = maps.lookup(kind, w)
                if fin is None:
                    ok = False
                    finals.append(w)
                    unresolved.append(f"{kind} {w}: not in the numbering map (left as written)")
                else:
                    finals.append(fin)
                    if fin.startswith("[REMOVED"):
                        unresolved.append(f"{kind} {w}: removed in pass 016 (marked {fin})")
        nums = [int(f[1:]) for f in finals if re.fullmatch(r"S\d+", f)]
        if len(finals) > 2 and len(nums) == len(finals) and nums == list(range(nums[0], nums[0] + len(nums))):
            out = f"S{nums[0]}–S{nums[-1]}"
        elif len(finals) == 1:
            out = finals[0]
        else:
            out = ", ".join(finals[:-1]) + " and " + finals[-1]
        return out, ok

    def sub_prefixed(m):
        prefix = m.group("prefix")
        kind = "figure" if prefix.strip().lower().startswith(("fig", "supplementary fig")) or "Fig" in prefix else "table"
        new_ids, _ = tr_ids(kind, m.group("ids"))
        new = prefix + new_ids
        refs.append({"as_written": m.group(0), "final": new, "kind": kind,
                     "self": any(maps.lookup(kind, t.group("id"))[1] == self_wid
                                 for t in ID_RE.finditer(m.group("ids")))})
        return new

    out, pos = [], 0
    for m in REF_RE.finditer(text):
        out.append(text[pos:m.start()])
        out.append(sub_prefixed(m))
        spans.append((m.start(), m.end()))
        pos = m.end()
    out.append(text[pos:])
    text1 = "".join(out)

    # bare ids (not preceded by a Table/Fig prefix)
    bare_non_refs = bare_non_refs or {}
    bare_seen = {}

    def sub_bare(m):
        wid = m.group("id")
        if wid in bare_refs:
            fin, _ = maps.lookup(kind_default, wid)
            new = fin if fin else wid
            refs.append({"as_written": m.group(0), "final": new, "kind": "table (bare)",
                         "self": TABLE_ALIASES.get(wid, wid) == self_wid})
            return new
        bare_seen[wid] = bare_seen.get(wid, 0) + 1
        return m.group(0)

    bare_re = re.compile(r"(?<![\w\-./`:])(?P<id>" + ID + r")(?P<alias>" + ALIAS + r")?(?![\w`])")
    pieces, pos = [], 0
    for m in REF_RE.finditer(text1):
        pieces.append(bare_re.sub(sub_bare, text1[pos:m.start()]))
        pieces.append(m.group(0))
        pos = m.end()
    pieces.append(bare_re.sub(sub_bare, text1[pos:]))
    text2 = "".join(pieces)
    for wid, n in sorted(bare_seen.items()):
        if wid in bare_non_refs:
            continue
        unresolved.append(f"bare token {wid} (x{n}) not preceded by 'Table'/'Fig.'; left as written")
    return text2, refs, unresolved


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

PENDING_RE = re.compile(r"provisional|at assembly|\[PENDING|\bpending\b|\bTODO\b|\(candidate\)", re.I)


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_text(text, maps, wid, cfg, stats):
    """Normalise and translate `text`; counts and refs go into `stats` (pass a
    throwaway dict for fields that are substrings of an already-counted one)."""
    if text is None:
        return None, []
    t, n_sp = americanize(text)
    t, n_em = replace_em_dashes(t)
    t, refs, unresolved = translate_refs(t, maps, wid, cfg.get("bare_refs", ()), cfg.get("bare_non_refs"))
    stats["spelling_changes"] += n_sp
    stats["em_dash_replacements"] += n_em
    stats["refs"].extend(refs)
    stats["unresolved"].extend(unresolved)
    return t, refs


def final_of(maps, wid):
    new = maps.t[wid]["new"]
    m = re.match(r"^S(\d+)", new)
    return (f"S{m.group(1)}" if m else new)


def build_entry(cfg, books, maps, out_root):
    wid = cfg["wid"]
    primary = books[cfg["files"][0]]
    stats = {"spelling_changes": 0, "em_dash_replacements": 0, "refs": [], "unresolved": []}
    raw = {}

    def res(spec, wb=primary):
        return resolve(spec, wb, books)

    # caption
    if cfg.get("caption_parts"):
        raw["caption"] = "\n\n".join(res(s) for s in cfg["caption_parts"])
    elif cfg.get("caption"):
        raw["caption"] = res(cfg["caption"])
    else:
        raw["caption"] = None
    footnotes_raw = []
    if raw["caption"] and cfg.get("split_inline_footnote"):
        key = cfg["split_inline_footnote"]
        if key in raw["caption"]:
            head, tail = raw["caption"].split(key, 1)
            raw["caption"] = head.rstrip()
            footnotes_raw.append({"label": "Footnote", "text": tail.strip()})
    # title
    t = cfg.get("title")
    raw["caption_body"] = None
    if t == "first_sentence":
        raw["title"], raw["caption_body"] = first_sentence(raw["caption"])
    elif t == "first_sentence_pipe":
        m = re.match(r"^(Table\s+\S+\s+\|\s+[^.]*\.)\s*(.*)$", raw["caption"], re.S)
        raw["title"], raw["caption_body"] = m.group(1), m.group(2)
    elif t:
        raw["title"] = res(t)
        if raw["caption"]:
            raw["caption_body"] = raw["caption"]
    else:
        raw["title"] = None
    raw["alt_title"] = res(cfg["alt_title"]) if cfg.get("alt_title") else None
    if raw["alt_title"]:
        raw["alt_title"] = re.sub(r"^#+\s*", "", raw["alt_title"])
    # footnotes
    if cfg.get("footnotes_numbered"):
        footnotes_raw += split_numbered(res(cfg["footnotes_numbered"]))
    if cfg.get("footnotes_lettered"):
        footnotes_raw += split_lettered(res(cfg["footnotes_lettered"]))
    for label, spec in cfg.get("footnotes", []):
        footnotes_raw.append({"label": label, "text": res(spec)})
    panels_raw = [{"label": lab, "text": res(s)} for lab, s in cfg.get("panels", [])]
    sheetcap_raw = [{"label": lab, "text": res(s)} for lab, s in cfg.get("sheet_captions", [])]
    additions_raw = [{"label": lab, "text": res(s)} for lab, s in cfg.get("caption_additions", [])]
    spec_text = res(cfg["spec"]) if cfg.get("spec") else None

    entry = {
        "workspace_id": wid,
        "final_number": final_of(maps, wid),
        "source": "built workbook caption sheet",
        "source_file": str(Path(cfg["files"][0]).name),
        "source_files": [str(out_root / f) for f in cfg["files"]],
        "superseded_files": [str(out_root / f) for f in cfg.get("superseded_files", [])],
        "field_choice": cfg["choice"],
    }
    title_in_caption = t in ("first_sentence", "first_sentence_pipe")
    for k in PROSE_KEYS:
        counted = not (k == "caption_body" or (k == "title" and title_in_caption))
        sink = stats if counted else {"spelling_changes": 0, "em_dash_replacements": 0, "refs": [], "unresolved": []}
        entry[k], _ = process_text(raw.get(k), maps, wid, cfg, sink)
        entry[k + "_as_written"] = raw.get(k)
    # title in final form: replace the leading table label with the final number
    if raw["title"]:
        body = re.sub(r"^(?:New\s+)?(?:Supplementary\s+)?Table\s+S[\w.\-]*?(?:\s*\|\s*|\.\s+)", "", entry["title"])
        entry["title_final_form"] = f"Supplementary Table {entry['final_number']}. {body}"
    else:
        entry["title_final_form"] = None

    def proc_list(items):
        out = []
        for it in items:
            txt, _ = process_text(it["text"], maps, wid, cfg, stats)
            out.append({"label": it["label"], "text": txt, "text_as_written": it["text"]})
        return out

    entry["footnotes"] = proc_list(footnotes_raw)
    entry["panel_headings"] = proc_list(panels_raw)
    entry["sheet_captions"] = proc_list(sheetcap_raw)
    entry["caption_additions"] = proc_list(additions_raw)
    entry["spec_text_not_caption"] = spec_text
    # sheets
    sheets = []
    filt = cfg.get("sheet_filter", {})
    for f in cfg["files"]:
        for s in books[f].sheets:
            if f in filt and not s["name"].startswith(filt[f]):
                continue
            sheets.append({"file": Path(f).name, **s})
    entry["sheets"] = sheets
    entry["has_caption_prose"] = bool(raw["caption"])
    entry["cross_refs"] = [{"as_written": r["as_written"], "final": r["final"], "kind": r["kind"], "self": r["self"]}
                           for r in stats["refs"]]
    entry["spelling_changes"] = stats["spelling_changes"]
    entry["em_dash_replacements"] = stats["em_dash_replacements"]
    # analyst checks
    checks = list(cfg.get("checks", []))
    seen = set()
    for u in stats["unresolved"]:
        if u not in seen:
            seen.add(u)
            checks.append("Cross-reference: " + u)
    texts = [raw.get(k) for k in PROSE_KEYS] + [x["text"] for x in footnotes_raw + panels_raw + sheetcap_raw + additions_raw]
    hits = sorted({m.group(0).lower() for tx in texts if tx for m in PENDING_RE.finditer(tx)})
    if hits:
        checks.append("Prose contains: " + ", ".join(repr(h) for h in hits) + ".")
    if not raw["caption"] and not any(c.startswith(("No caption prose", "No table-level", "Main ")) for c in checks):
        checks.append("No caption prose (see field_choice).")
    entry["lead_check"] = checks
    return entry


def submitted_entries(maps, fmaps_ok=True):
    md = docx_to_markdown.convert(str(SUPP_INFO_DOCX))
    tables, figures = {}, {}
    for line in md.split("\n"):
        m = re.match(r"^Supplementary Table (S\d+)\.\s", line)
        if m and m.group(1) in SUBMITTED_TABLES:
            tables[m.group(1)] = line.strip()
        m = re.match(r"^Fig\. (S\d+) \|", line)
        if m and m.group(1) in SUBMITTED_FIGURES:
            figures[m.group(1)] = line.strip()
    out_t, out_f = {}, {}
    for wid in SUBMITTED_TABLES:
        cfg = {"wid": wid}
        stats = {"spelling_changes": 0, "em_dash_replacements": 0, "refs": [], "unresolved": []}
        raw = tables[wid]
        title_raw, body_raw = first_sentence(raw)
        cap, _ = process_text(raw, maps, wid, cfg, stats)
        title, _ = process_text(title_raw, maps, wid, cfg, {"spelling_changes": 0, "em_dash_replacements": 0,
                                                            "refs": [], "unresolved": []})
        fin = final_of(maps, wid)
        checks = ["Submitted caption; the submitted title sentence is the whole caption (no legend body)."]
        if wid in ("S1", "S2"):
            checks.append("The S-R2.1-A workbook carries an edit-sheet operation (R2.1-T4, l.6136-6138) appending a "
                          "sentence citing 'Table S3' (removed) and S-R2.1-E to the S1/S2 captions; not applied here, "
                          "confirm the caption stays unchanged.")
        checks += ["Cross-reference: " + u for u in stats["unresolved"]]
        out_t[wid] = {
            "workspace_id": wid, "final_number": fin,
            "source": "submitted supplementary_info_v04", "source_file": SUPP_INFO_DOCX.name,
            "source_files": [str(SUPP_INFO_DOCX)], "superseded_files": [],
            "field_choice": "line 'Supplementary Table %s. …' of the docx_to_markdown conversion" % wid,
            "title": title, "title_as_written": title_raw,
            "title_final_form": re.sub(r"^Supplementary Table S\d+\.", f"Supplementary Table {fin}.", title),
            "caption": cap, "caption_as_written": raw,
            "caption_body": body_raw or None, "caption_body_as_written": body_raw or None,
            "alt_title": None, "alt_title_as_written": None,
            "footnotes": [], "panel_headings": [], "sheet_captions": [], "caption_additions": [],
            "spec_text_not_caption": None, "sheets": [], "has_caption_prose": True,
            "cross_refs": [{"as_written": r["as_written"], "final": r["final"], "kind": r["kind"], "self": r["self"]}
                           for r in stats["refs"]],
            "spelling_changes": stats["spelling_changes"], "em_dash_replacements": stats["em_dash_replacements"],
            "lead_check": checks,
        }
    for wid in SUBMITTED_FIGURES:
        raw = figures[wid]
        cap, n_sp = americanize(raw)
        new = maps.f[wid]["new"]
        fin = re.match(r"^S\d+", new).group(0)
        out_f[wid] = {
            "workspace_id": wid, "final_number": fin,
            "source": "submitted supplementary_info_v04", "source_file": SUPP_INFO_DOCX.name,
            "title_as_written": raw,
            "title_final_form": re.sub(r"^Fig\. S\d+ \|", f"Supplementary Figure {fin} |", cap),
            "caption": cap, "caption_as_written": raw,
            "render": maps.f[wid].get("render"),
            "spelling_changes": n_sp, "em_dash_replacements": 0, "cross_refs": [],
            "lead_check": ["Submitted caption is a title fragment without a final period and without legend body "
                           "('Count of tumor groups in the COMET dataset'); consider whether a legend sentence is needed.",
                           "Figure map: provisional S-SUB-F1 would take S3 at packaging and shift the current S3–S14 by "
                           "one, so this figure would become S12; S11 is the current pass-016 number."],
        }
    return out_t, out_f, md


def md_escape_cell(s):
    return s.replace("|", "\\|").replace("\n", " ")


def render_md(tables, figures, maps):
    by_final = {}
    for wid, e in tables.items():
        by_final[e["final_number"]] = e
    lines = ["# Supplementary-table captions collected for assembly (2026-11-09)", "",
             "Generated by `scripts/supp_tables/extract_workbook_captions.py`. Prose is copied from the built "
             "workbooks / submitted SI with American spelling, spaced em dashes replaced, and cross-references "
             "translated to the current final numbers (pass 016 maps). `[REMOVED:<id>]` marks a reference to an "
             "item removed on 2026-11-04. The JSON beside this file keeps every as-written text.", ""]
    finals = sorted({int(v["new"][1:]) for v in maps.t.values() if re.fullmatch(r"S\d+", v["new"])})
    inv = {}
    for k, v in maps.t.items():
        if re.fullmatch(r"S\d+", v["new"]):
            inv[v["new"]] = k
    for n in finals:
        fin = f"S{n}"
        e = by_final.get(fin)
        wid = inv[fin]
        if e is None:
            lines += [f"## Supplementary Table {fin} (workspace {wid}; not covered)", "",
                      "Not in this collection: no built workbook among the inputs and not one of the submitted "
                      "captions still in use unchanged (S1, S2, S9–S13).", ""]
            continue
        src = e["source_file"] if e["source"].startswith("built") else "submitted supplementary_info_v04"
        lines += [f"## Supplementary Table {fin} (workspace {wid}; file {src})", ""]
        if e["source"].startswith("submitted"):
            lines += ["*Source: submitted supplementary_info_v04.*", ""]
        lines += [f"**Title:** {e['title_final_form'] or '(no title in the workbook)'}", ""]
        if e.get("alt_title"):
            lines += [f"**Alternative short title (workbook heading):** {e['alt_title']}", ""]
        if e["caption"]:
            lines += ["**Caption (as in the source, after normalisation):**", "", e["caption"], ""]
        else:
            lines += ["**Caption:** none in the workbook (no caption prose).", ""]
        for key, head in (("caption_additions", "Caption additions"), ("sheet_captions", "Sheet captions"),
                          ("panel_headings", "Panel / part headings"), ("footnotes", "Footnotes")):
            if e[key]:
                lines += [f"**{head}:**", ""]
                for it in e[key]:
                    lines += [f"- **{it['label']}.** " + it["text"].replace("\n", "\n  ")]
                lines += [""]
        if e["cross_refs"]:
            lines += ["| as written | final | kind | self |", "|---|---|---|---|"]
            seen = set()
            for r in e["cross_refs"]:
                key = (r["as_written"], r["final"])
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"| {md_escape_cell(r['as_written'])} | {md_escape_cell(r['final'])} | {r['kind']} | "
                             f"{'yes' if r['self'] else ''} |")
            lines += [""]
        else:
            lines += ["Cross-references: none.", ""]
        lines += [f"Spelling changes: {e['spelling_changes']}; em-dash replacements: {e['em_dash_replacements']}.", ""]
        lines += ["LEAD CHECK: " + (" | ".join(e["lead_check"]) if e["lead_check"] else "nothing outstanding."), ""]
    lines += ["# Supplementary figure caption (submitted, still in use)", ""]
    for wid, e in figures.items():
        lines += [f"## Supplementary Figure {e['final_number']} (workspace {wid}; file {e['source_file']})", "",
                  "*Source: submitted supplementary_info_v04.*", "",
                  f"**Caption:** {e['title_final_form']}", "",
                  f"Render: `{e['render']}`", "",
                  f"Spelling changes: {e['spelling_changes']}; em-dash replacements: 0.", "",
                  "LEAD CHECK: " + " | ".join(e["lead_check"]), ""]
    return "\n".join(lines)


def render_report(tables, figures, inputs, maps, out_dir):
    wb = [e for e in tables.values() if e["source"].startswith("built")]
    sub = [e for e in tables.values() if e["source"].startswith("submitted")]
    no_prose = [e["workspace_id"] for e in wb if not e["has_caption_prose"]]
    L = ["# Extraction report: supplementary-table captions (2026-11-09)", "",
         "Script: `scripts/supp_tables/extract_workbook_captions.py` (deterministic; read-only on inputs).", "",
         "## Inputs", ""]
    L += [f"- `{p}`" for p in inputs]
    L += ["", "Workbook integrity against the builders' records (SHA256SUMS.txt, else BUILD_REPORT.md, in the "
          "workbook's directory):", ""]
    for p in inputs:
        if not p.endswith(".xlsx"):
            continue
        h, d = sha256(Path(p)), Path(p).parent
        recs = [r for r in (d / "SHA256SUMS.txt", d / "BUILD_REPORT.md") if r.exists()]
        hit = next((r.name for r in recs if h in r.read_text(errors="replace")), None)
        L.append(f"- `{Path(p).name}`: " + (f"matches {hit}" if hit else "NOT FOUND in builder records"))
    L += ["", "## Counts", "",
          f"- Tables covered: {len(tables)} ({len(wb)} from built workbooks, {len(sub)} submitted captions); "
          f"figures covered: {len(figures)} (submitted Fig. S4 → final {figures['S4']['final_number']}).",
          f"- Final table numbers in the map: S1–S{max(int(v['new'][1:]) for v in maps.t.values() if re.fullmatch(r'S\d+', v['new']))}; "
          f"not covered here: {', '.join(sorted(set(f'S{n}' for n in range(1, 49)) - {e['final_number'] for e in tables.values()}, key=lambda s: int(s[1:])))}.",
          f"- Workbook entries without caption prose: {', '.join(no_prose)}.",
          f"- Spelling changes: {sum(e['spelling_changes'] for e in tables.values()) + sum(e['spelling_changes'] for e in figures.values())}; "
          f"em-dash replacements: {sum(e['em_dash_replacements'] for e in tables.values())}; "
          f"cross-references translated: {sum(len(e['cross_refs']) for e in tables.values())} "
          f"(of which to removed items: {sum(1 for e in tables.values() for r in e['cross_refs'] if 'REMOVED' in r['final'])}).",
          "", "## Rules applied", "",
          "- Spelling: tumour→tumor, colour→color, neighbour→neighbor, labelled/labelling→labeled/labeling, "
          "favour→favor, normalis-→normaliz-, summaris-→summariz-, analysed/analyse/analysing→analyz-, centre→center; "
          "'grey' and 'analyses' unchanged; text inside backticks untouched.",
          "- Spaced em dash ' — ': two in one sentence → commas (parenthetical); otherwise '; '. Not applied inside "
          "backticks or inside quoted strings containing the dash (quoted table-cell literals such as "
          "'point only — no CI in frozen output').",
          "- Cross-references: 'Table(s)/Supplementary Table(s)/Fig(s)./Figure(s)' followed by workspace ids, lists, "
          "ranges (expanded, then re-collapsed only if the final numbers are contiguous) and parenthetical symbolic "
          "aliases (S-A1, S-T1, S-D1 …, dropped after translation). Bare ids are translated only where listed per "
          "workbook as table references; other bare S-tokens are left as written and reported. Panel suffixes are kept "
          "(S24b → S6b). The table's own number in its title/caption is also translated (marked self). "
          "Main-text figures (Fig. 3, Fig. 6) are untouched.",
          "- `title_final_form` replaces the leading table label with 'Supplementary Table S<final>.'; `title` keeps "
          "the as-written label translated in place.",
          "", "## Per-workbook field choices", ""]
    for e in sorted(wb, key=lambda e: int(e["final_number"][1:])):
        L += [f"- **{e['workspace_id']} → {e['final_number']}** (`{e['source_file']}`"
              + (f"; superseded, read but not used: `{Path(e['superseded_files'][0]).name}`" if e["superseded_files"] else "")
              + f"): {e['field_choice']}"]
    L += ["", "Submitted captions: the lines 'Supplementary Table S1/S2/S9–S13. …' and 'Fig. S4 | …' of the "
          "`docx_to_markdown.py` conversion of `supplementary_info_v04.docx`.", "",
          "## Unresolved items (LEAD CHECK)", ""]
    for e in sorted(tables.values(), key=lambda e: int(e["final_number"][1:])):
        for c in e["lead_check"]:
            L.append(f"- S{e['final_number'][1:]} ({e['workspace_id']}): {c}")
    for e in figures.values():
        for c in e["lead_check"]:
            L.append(f"- Fig. {e['final_number']} (workspace Fig. {e['workspace_id']}): {c}")
    L += ["", "## Outputs", "", f"- `{out_dir}/captions_from_workbooks.json`", f"- `{out_dir}/captions_from_workbooks.md`",
          f"- `{out_dir}/EXTRACTION_REPORT.md`", f"- `{out_dir}/SHA256SUMS.txt`", ""]
    return "\n".join(L)


def main():
    out_root = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output")).resolve()
    out_dir = out_root / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    maps = Maps(json.loads(TABLE_MAP_PATH.read_text()), json.loads(FIGURE_MAP_PATH.read_text()))

    rel_files = []
    for cfg in ENTRIES:
        for f in cfg["files"] + cfg.get("superseded_files", []):
            if f not in rel_files:
                rel_files.append(f)
    books = {f: Workbook(out_root / f) for f in rel_files}

    tables = {}
    for cfg in ENTRIES:
        tables[cfg["wid"]] = build_entry(cfg, books, maps, out_root)
    sub_t, sub_f, _ = submitted_entries(maps)
    tables.update(sub_t)
    tables = dict(sorted(tables.items(), key=lambda kv: int(kv[1]["final_number"][1:])))

    inputs = [str(out_root / f) for f in rel_files] + [str(SUPP_INFO_DOCX), str(TABLE_MAP_PATH), str(FIGURE_MAP_PATH),
                                                       str(REPO / "scripts/docx_to_markdown.py"), str(Path(__file__).resolve())]
    doc = {
        "meta": {
            "generator": "scripts/supp_tables/extract_workbook_captions.py",
            "numbering": "pass 016 maps (scripts/figures/supp_table_numbering.json, supp_figure_numbering.json)",
            "aliases": TABLE_ALIASES,
            "note": "prose fields are normalised (spelling, spaced em dashes) and cross-references translated; "
                    "*_as_written / text_as_written keep the source text",
        },
        "tables": tables,
        "figures": sub_f,
    }
    jpath = out_dir / "captions_from_workbooks.json"
    jpath.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    mpath = out_dir / "captions_from_workbooks.md"
    mpath.write_text(render_md(tables, sub_f, maps) + "\n")
    rpath = out_dir / "EXTRACTION_REPORT.md"
    rpath.write_text(render_report(tables, sub_f, inputs, maps, out_dir) + "\n")
    spath = out_dir / "SHA256SUMS.txt"
    lines = [f"{sha256(Path(p))}  {p}" for p in inputs + [str(jpath), str(mpath), str(rpath)]]
    spath.write_text("\n".join(lines) + "\n")
    print(f"tables: {len(tables)}  figures: {len(sub_f)}  -> {out_dir}")


if __name__ == "__main__":
    main()
