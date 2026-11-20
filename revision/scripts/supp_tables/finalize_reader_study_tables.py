#!/usr/bin/env python3
"""Final per-table reader-study workbooks (S31, S32, S39-S47) from the raw output of build_reader_study_tables.py.

Applies, as explicit rules, the presentation edits that the packaged workbooks carry on top of the raw analysis
tables (reader-facing headers in S31, endpoint prefixes dropped in S32, US spelling, cohort and tier labels in S40,
event names in S41, run identifiers in the `source` sheet replaced by a pointer to the controlled analysis record),
adds the S31 case-composition sheet and writes the `caption` sheet from the Supplementary Information Markdown.
No number is computed or changed; the case-composition counts are tallied from the analysis reading table.

`--check-against DIR` proves the rules: every produced data sheet and `source` sheet must equal, cell for cell and
type for type, the sheet of the same name in DIR (the `caption` sheet and the `source` label are excluded because
they follow the current legends and label). Run it on the 14 November raw workbooks against the revision per-table
workbooks before using the rules on a new export.

Usage:
  export PROSPER_OUT="$PWD/prosper_revision_output"
  A="$PROSPER_OUT/reader_study_suite/analysis/20261116_prod_export_20261116-085036_five_readers"
  python3 scripts/supp_tables/finalize_reader_study_tables.py \
      --raw "$PROSPER_OUT/supp_tables_reader_study/$(basename "$A")" --analysis "$A" --out <dir> \
      --label "analysis of the export of 16 November 2026 (all five readers at 80/80 cases)"
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import openpyxl
from openpyxl.styles import Font

sys.path.insert(0, str(Path(__file__).resolve().parent))
import packaging_common_20261109 as pc  # noqa: E402

SI_TOOLS = pc.OUT_ROOT / "e1_si_close_audit_20261115/scratch/si-tables"  # si_entries_044.py, apply_tables_046_save.py
sys.path.insert(0, str(SI_TOOLS))
import si_entries_044 as sie  # noqa: E402
import apply_tables_046_save  # noqa: E402,F401  (document author "PROSPER authors")

pc.BUILD_DATE = "2026-11-16"
TABLES = ["S31", "S32", "S39", "S40", "S41", "S42", "S43", "S44", "S45", "S46", "S47"]
READER_TEXT = {"S44", "S47"}  # sheets with readers' own words: spelling rules apply to the header row only
RETAINED = "retained in the controlled analysis record"
SPELLING = [("favoured", "favored"), ("Favoured", "Favored"), ("neighbour", "neighbor"), ("Neighbour", "Neighbor"),
            ("tumour", "tumor"), ("Tumour", "Tumor")]
S31_HEADERS = {"reader_id": "Reader", "expertise_tier": "Expertise tier (code)",
               "submissions_A_nonpilot": "Step 1 submissions (non-pilot)",
               "submissions_B_nonpilot": "Step 2 submissions (non-pilot)",
               "submissions_C_nonpilot": "Step 3 submissions (non-pilot)",
               "pilot_submissions_excluded": "Pilot submissions excluded",
               "readings_A_and_B": "Readings with Steps 1 and 2", "readings_with_C": "Readings with Step 3",
               "cases_CCDI": "CCDI-MCI cases", "cases_STV": "STV cases", "qc_gate": "Quality-check gate"}
S31_GATE = {"dropped (003 §A)": "not used to define the primary population (Supplementary Note 3)"}
S41_MEASURES = {"n_pecan_opened": "n_external_portal_opened",
                "n_pecan_opened (total events)": "n_external_portal_opened (total events)"}
TIER_RE = re.compile(r"^(CCDI|STV) \| Tier1 \((\w+)\)$")


def s40_level(v):
    if not isinstance(v, str):
        return v
    m = TIER_RE.match(v)
    if m:
        return f"{'CCDI-MCI' if m.group(1) == 'CCDI' else 'STV'} | first tier ({m.group(2).replace('_', ' ')})"
    v = v.replace("not_on_ali_list", "not_on_review_shortlist")
    return {"CCDI": "CCDI-MCI", "CCDI | not_on_review_shortlist": "CCDI-MCI | not_on_review_shortlist"}.get(v, v)


def col_of(ws, header: str) -> int:
    return [c.value for c in ws[1]].index(header) + 1


def spell(ws, header_only: bool) -> None:
    for row in ws.iter_rows(max_row=1 if header_only else ws.max_row):
        for c in row:
            if isinstance(c.value, str):
                for a, b in SPELLING:
                    c.value = c.value.replace(a, b)


def case_composition(wb, analysis: Path) -> None:
    seen: dict[str, tuple[str, str]] = {}
    with open(analysis / "readings.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            seen[r["case_id"]] = (r["reference"], r["cohort"])
    groups = sorted({g for g, _ in seen.values()})
    ws = wb.create_sheet("case_composition")
    ws.append(["Tumor group", "CCDI-MCI", "STV", "all"])
    tot = [0, 0]
    for g in groups:
        n = [sum(1 for x in seen.values() if x == (g, c)) for c in ("CCDI", "STV")]
        tot = [tot[0] + n[0], tot[1] + n[1]]
        ws.append([g, n[0], n[1], sum(n)])
    ws.append(["Total", tot[0], tot[1], sum(tot)])
    for c in ws[1]:
        c.font = Font(bold=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col in "ABCD":
        ws.column_dimensions[col].width = 25


def finalize(final: str, raw: Path, analysis: Path, label: str, entries: dict):
    wb = openpyxl.load_workbook(raw / f"Supplementary_Table_{final}_source.xlsx")
    src = wb["source"]
    for row in src.iter_rows(min_row=2):
        key = row[0].value
        if key == "label":
            row[1].value = label
        elif key in {"analysis_dir", "qc74_dir", "export", "analysis_commit", "readings_sha256"} or key.startswith("sheet:"):
            row[1].value = RETAINED
    for ws in wb.worksheets:
        if ws.title != "source":
            spell(ws, header_only=final in READER_TEXT)
    if final == "S31":
        ws = wb["reader_case_flow"]
        gate = col_of(ws, "qc_gate")
        for row in ws.iter_rows(min_row=2):
            row[gate - 1].value = S31_GATE.get(row[gate - 1].value, row[gate - 1].value)
        for c in ws[1]:
            c.value = S31_HEADERS[c.value]
        case_composition(wb, analysis)
    if final == "S32":
        ws = wb["primary_endpoints"]
        k = col_of(ws, "endpoint")
        for row in ws.iter_rows(min_row=2):
            row[k - 1].value = re.sub(r"^E\d ", "", row[k - 1].value)
    if final == "S40":
        ws = wb["strata"]
        k = col_of(ws, "level")
        for row in ws.iter_rows(min_row=2):
            row[k - 1].value = s40_level(row[k - 1].value)
    if final == "S41":
        ws = wb["engagement"]
        k = col_of(ws, "measure")
        for row in ws.iter_rows(min_row=2):
            row[k - 1].value = S41_MEASURES.get(row[k - 1].value, row[k - 1].value)
    pc.write_caption_sheet(wb, sie.caption_rows(final, entries[final]))
    return wb


def compare(final: str, wb, ref_dir: Path) -> list[str]:
    ref = openpyxl.load_workbook(ref_dir / f"Supplementary_Table_{final}.xlsx")
    bad = []
    if wb.sheetnames != ref.sheetnames:
        bad.append(f"{final}: sheets {wb.sheetnames} != {ref.sheetnames}")
    for name in ref.sheetnames:
        if name == "caption" or name not in wb.sheetnames:
            continue
        a, b = wb[name], ref[name]
        for r in range(1, max(a.max_row, b.max_row) + 1):
            for c in range(1, max(a.max_column, b.max_column) + 1):
                x, y = a.cell(r, c).value, b.cell(r, c).value
                if name == "source" and a.cell(r, 1).value == "label":
                    continue
                if x != y or type(x) is not type(y):
                    bad.append(f"{final}/{name}!R{r}C{c}: {x!r} != {y!r}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--analysis", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, default=None, help="omit with --check-against to test the rules only")
    ap.add_argument("--check-against", type=Path, default=None)
    a = ap.parse_args()
    entries = sie.si_entries()
    bad: list[str] = []
    manifest = {}
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
    for final in TABLES:
        wb = finalize(final, a.raw, a.analysis, a.label, entries)
        if a.check_against:
            bad += compare(final, wb, a.check_against)
        if a.out:
            path = a.out / f"Supplementary_Table_{final}.xlsx"
            pc.save_xlsx(wb, path)
            manifest[final] = {"file": path.name, "sha256": pc.sha256(path), "sheets": wb.sheetnames}
    if a.check_against:
        print(f"rule check against {a.check_against}: {len(bad)} differing cells")
        for line in bad[:40]:
            print("  ", line)
    if a.out and not bad:
        (a.out / "reader_tables_manifest.json").write_text(json.dumps(
            {"label": a.label, "raw": pc.rel(a.raw), "raw_manifest_sha256": pc.sha256(a.raw / "manifest.json"),
             "si_sha256": pc.sha256(pc.SI_MD), "tables": manifest}, indent=1) + "\n")
        print("wrote", a.out)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
