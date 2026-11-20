#!/usr/bin/env python3
"""revision third in-place revision (2026-11-16) copy of build_supplementary_data_workbooks_048.py: the Supplementary
Information prints no table, so the Index sheet has no "Previewed in the Supplementary Information" column, the SI has
no "Workbook:" pointer paragraphs (the script checks that none remains), and README.md does not mention previews.
Data and Captions sheets are written exactly as before.

Supplementary Data 1–6: the 48 tables grouped into six themed Excel files (2026-11-10; revision prep copy, 2026-11-16:
paths, layout, SI source, output folder and build date are command-line options (no default output folder: the
folder given by --out is deleted and rebuilt, so it must be named explicitly); every per-table workbook is sha-checked
against its manifest; sheet lists are taken from each workbook (tables may lose sheets or gain new ones; `source`
sheets are copied into a Provenance sheet only when present; S48 keeps its `source` sheet as data); optional
"sheet_overrides" in the layout JSON extend SHEET_OVERRIDES; README/Contents checks use the table and preview
counts of the layout instead of 48/19; README no longer prints "No workbook also have a Provenance sheet" when no
workbook has one; --pointers-out writes the expected SI "Workbook:" paragraphs and the differences as JSON;
--write-si-pointers is refused unless --si names a file other than the authoring SI. revision copy as before;
revision copy, 2026-11-16:
per-table workbooks read from scratch/si-tables/per_table_workbooks_046 (output of apply_tables_046.py), delivery in
e1_si_close_audit_20261115/tables_046, two sheet names shortened instead of truncated, docProps author "PROSPER authors",
README without paths/build history; revision copy, 2026-11-16: per-table workbooks read from checks_044/per_table_workbooks, delivery in
revision/supplementary, Title Case Methods headings synchronised from the SI; revision copy, 2026-11-15:
relabelled "Supplementary Data k, S#", per-table workbooks read from checks_043/per_table_workbooks, delivery in
revision/supplementary; "reproduced" -> "previewed" in the SI pointer paragraphs and Index).

npj Digital Medicine asks for every cited supplementary item in one merged Supplementary Information PDF and, for
tables too large for the PDF, separate Excel files labelled "Supplementary Data N" with title and legend in the PDF.
This script takes the 48 assembled per-table workbooks of
    $PROSPER_OUT/supp_package_20261110/supplementary_tables/Supplementary_Table_S<n>.xlsx
(output of assemble_final_tables_20261110.py, kept as the per-table source form) and writes
    $PROSPER_OUT/supp_package_20261110/supplementary_data/Supplementary_Data_<k>.xlsx      (k = 1..6)
following scripts/supp_tables/supplementary_data_layout.json. Each Supplementary Data file has
    Index        one row per table: number, title, sheets (new name <- original), whether the Supplementary
                 Information PDF reproduces the table, caption
    Captions     every row of every table's `caption` sheet (caption, footnotes, notes, caption source, source workbook)
    Provenance   every row of every table's `source` sheet (reader-study tables), when present
    S<n>_<sheet> the data sheets, values copied cell for cell (bold, wrap, column widths and frozen header kept);
                 sheet names are prefixed with the table number (`S8` -> `S8_table`, `S9 body` -> `S9_body`,
                 `S8b_paired` -> `S8_b_paired`), truncated to Excel's 31 characters where needed.
Nothing is computed. Values are checked cell for cell against the per-table workbooks after writing.

The Supplementary Information Markdown carries one pointer paragraph per table ("Workbook: Supplementary Data k, sheets
beginning S<n>_. ..."); `--write-si-pointers` inserts or updates those paragraphs from the layout, otherwise the script
only checks them and reports differences. Also written: supplementary_data/manifest.json, README.md and
FILE_DESCRIPTIONS.md (upload titles and <= 50-word descriptions); SHA256SUMS.txt of the package is refreshed.

Run from the repository root (after the assembler, before render_supplement_docx.py):
    export PROSPER_OUT="$PWD/prosper_revision_output"
    /HOME/miniconda3/bin/python3 scripts/supp_tables/build_supplementary_data_workbooks.py [--write-si-pointers]
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

sys.path.insert(0, "/PROSPER_WORKSPACE/prosper_revision/scripts/supp_tables")  # revision copy
import packaging_common_20261109 as pc  # noqa: E402
sys.path.insert(0, "/PROSPER_WORKSPACE/prosper_revision/prosper_revision_output/e1_si_close_audit_20261115/scratch/si-tables")
import si_entries_044  # noqa: E402
si_entries_044.HEADING = "## Supplementary Data 1–6: titles and legends"  # rev3: section heading without previews
from si_entries_044 import si_entries, caption_rows  # noqa: E402  (copied unchanged from checks_044/reproduction)

import apply_tables_046_save  # noqa: E402,F401  (A19: patches pc.save_xlsx -> creator "PROSPER authors")

pc.BUILD_DATE = "2026-11-16"  # overridden by --build-date
AUD = OUT_AUD = Path("/PROSPER_WORKSPACE/prosper_revision/prosper_revision_output/e1_si_close_audit_20261115")
REPO, OUT = pc.REPO, pc.OUT_ROOT
SI_MD = pc.SI_MD
LAYOUT = Path("/PROSPER_WORKSPACE/prosper_revision/prosper_revision_output/e1_si_close_audit_20261115/scratch/si-tables/supplementary_data_layout_044.json")  # --layout
PKG = AUD
TABLES = Path("/PROSPER_WORKSPACE/prosper_revision/prosper_revision_output/collab_draft_20261102/tables_048/per_table_workbooks")  # --tables
DATA = None  # --out (required)
TABLE_MANIFEST = TABLES / "manifest.json"  # revision: per-table manifest beside the per-table workbooks
POINTER_RE = re.compile(r"^Workbook: Supplementary Data (\d+), sheets beginning (S\d+)_\. .*$")
SHEET_RE = re.compile(r"^S(\d+)([a-z]?)(?:[ _](.*))?$")


DELIVERY = None  # revision: SHA256SUMS over the delivery folder only (set to --out)
SHEET_OVERRIDES = {("S26", "calibration_reliability_bins"): "S26_calibration_bins",
                   ("S48", "reader_study_summaries_checks"): "S48_reader_summaries_checks"}  # A13: no truncation


def fmt_runs(tables: list[str]) -> str:
    """S1..S10, S35..S37 -> 'S1–S10 and S35–S37'; S11..S14, S34, S38 -> 'S11–S14, S34 and S38'."""
    nums = sorted(int(t[1:]) for t in tables)
    runs, a = [], nums[0]
    for p, q in zip(nums, nums[1:] + [None]):
        if q != p + 1:
            runs += [f"S{a}"] if a == p else ([f"S{a}", f"S{p}"] if p == a + 1 else [f"S{a}–S{p}"])
            a = q
    return runs[0] if len(runs) == 1 else ", ".join(runs[:-1]) + " and " + runs[-1]


def load_layout() -> dict:
    lay = json.load(open(LAYOUT))
    seen: dict[str, int] = {}
    for f in lay["data_files"]:
        for t in f["tables"]:
            if t in seen:
                raise SystemExit(f"layout: {t} listed in Supplementary Data {seen[t]} and {f['number']}")
            seen[t] = f["number"]
    n_tables = max(int(t[1:]) for t in seen)  # revision: S48 added
    missing = [f"S{n}" for n in range(1, n_tables + 1) if f"S{n}" not in seen]
    extra = [t for t in seen if not re.match(r"^S[1-9]\d*$", t)]
    if missing or extra:
        raise SystemExit(f"layout: missing {missing}, unexpected {extra}")
    for t, specs in lay["pdf_tables"].items():
        if t not in seen:
            raise SystemExit(f"pdf_tables: unknown table {t}")
    lay["_file_of"] = seen
    for key, new in (lay.get("sheet_overrides") or {}).items():  # revision: "S41/sheet name": "S41_short"
        t, _, sheet = key.partition("/")
        SHEET_OVERRIDES[(t, sheet)] = new
    return lay


def new_sheet_name(final: str, name: str, used: set[str]) -> str:
    """`S8` -> `S8_table`; `S8b_paired` -> `S8_b_paired`; `S9 body` -> `S9_body`; `Part1` -> `S12_Part1`."""
    if (final, name) in SHEET_OVERRIDES:
        new = SHEET_OVERRIDES[(final, name)]
        if new in used or len(new) > 31:
            raise SystemExit(f"{final}: override {new!r}")
        used.add(new)
        return new
    m = SHEET_RE.match(name)
    if m and f"S{m.group(1)}" == final:
        letter, rest = m.group(2), m.group(3)
        if not letter and rest is None:
            new = f"{final}_table"
        else:
            new = f"{final}_{letter}{'_' if letter and rest else ''}{rest or ''}"
    else:
        new = f"{final}_{name}"
    new = re.sub(r"[\[\]:*?/\\]", "-", new)
    if len(new) > 31:
        raise SystemExit(f"{final}: sheet name {new!r} longer than 31 characters (add an override)")
        base = new[:29]
        k = 1
        new = f"{base}~{k}"
        while new in used:
            k += 1
            new = f"{base}~{k}"
    if new in used:
        raise SystemExit(f"{final}: duplicate sheet name {new!r}")
    used.add(new)
    return new


def copy_sheet(src, dst) -> int:
    n = 0
    for row in src.iter_rows():
        for c in row:
            if c.value is None:
                continue
            d = dst.cell(c.row, c.column, c.value)
            n += 1
            if c.data_type == "f":
                raise SystemExit(f"{src.title}!{c.coordinate}: formula cell")
            if c.number_format and c.number_format != "General":  # revision: p-value display format
                d.number_format = c.number_format
            if c.font is not None and c.font.bold:
                d.font = Font(bold=True)
            if c.alignment is not None and (c.alignment.wrap_text or c.alignment.vertical):
                d.alignment = Alignment(wrap_text=c.alignment.wrap_text, vertical=c.alignment.vertical)
    for key, dim in src.column_dimensions.items():
        if dim.width:
            dst.column_dimensions[key].width = dim.width
    if src.freeze_panes:
        dst.freeze_panes = src.freeze_panes
    return n


def pdf_status(final: str, lay: dict, sheet_map: dict[str, str], data_sheets: list[str]) -> tuple[str, str]:
    """(Index text, pointer paragraph) for the Supplementary Information."""
    k = lay["_file_of"][final]
    specs = lay["pdf_tables"].get(final, [])
    head = f"Workbook: Supplementary Data {k}, sheets beginning {final}_."
    if not specs:
        return "no (workbook only)", f"{head} Not previewed in this document."
    shown = [s["sheet"] for s in specs]
    for s in shown:
        if s not in sheet_map:
            raise SystemExit(f"{final}: pdf sheet {s!r} not in workbook ({list(sheet_map)})")
    subset = any(s.get("columns") for s in specs)
    cols = " (display columns; the source and unrounded columns are in the workbook only)" if subset else ""
    if set(shown) == set(data_sheets):
        return ("yes (all sheets)" if not subset else "yes (display columns)"), f"{head} Previewed in full above{cols}."
    names = [sheet_map[s] for s in shown]
    lst = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return (f"partly ({len(shown)} of {len(data_sheets)} sheets)",
            f"{head} {'Sheet' if len(names) == 1 else 'Sheets'} {lst} {'is' if len(names) == 1 else 'are'} previewed above"
            f"{cols}; the remaining sheets are in the workbook only.")


def si_pointer_lines() -> dict[str, tuple[int, str]]:
    """Existing pointer paragraphs in the SI Markdown: table -> (line index 0-based, text)."""
    out = {}
    for i, l in enumerate(SI_MD.read_text(encoding="utf-8").splitlines()):
        m = POINTER_RE.match(l.strip())
        if m:
            out[m.group(2)] = (i, l.strip())
    return out


def write_si_pointers(expected: dict[str, str], ents: dict) -> int:
    """Insert (after the caption paragraph) or replace the pointer paragraph of every table; returns changes."""
    lines = SI_MD.read_text(encoding="utf-8").splitlines()
    existing = si_pointer_lines()
    changes = 0
    # replace in place first (indices stable), then insert missing ones from the bottom up
    for t, (i, txt) in existing.items():
        if t in expected and txt != expected[t]:
            lines[i] = expected[t]
            changes += 1
    to_insert = sorted((ents[t]["line"], t) for t in expected if t not in existing)  # 1-based caption lines
    for cap_line, t in reversed(to_insert):
        i = cap_line - 1
        assert lines[i].startswith(f"Supplementary Data {ents[t]['file']}, {t} | "), (t, lines[i][:60])
        lines[i + 1:i + 1] = ["", expected[t]]
        changes += 1
    if changes:
        SI_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changes


def main() -> int:
    global TABLES, DATA, DELIVERY, LAYOUT, SI_MD, TABLE_MANIFEST
    ap = argparse.ArgumentParser(description="Supplementary Data 1–6 workbooks")
    ap.add_argument("--write-si-pointers", action="store_true",
                    help="insert/update the 'Workbook: Supplementary Data k ...' paragraph of every table in the SI Markdown")
    ap.add_argument("--tables", default=str(TABLES), help="per-table workbooks folder (with manifest.json)")
    ap.add_argument("--layout", default=str(LAYOUT), help="layout JSON (data_files, pdf_tables, optional sheet_overrides)")
    ap.add_argument("--si", default=str(SI_MD), help="Supplementary Information Markdown (pointer paragraphs, captions)")
    ap.add_argument("--out", required=True, help="output folder (deleted and rebuilt)")
    ap.add_argument("--build-date", default=pc.BUILD_DATE)
    ap.add_argument("--script-label", default="build_supplementary_data_workbooks_051.py")
    ap.add_argument("--skip-manifest-sha", action="store_true",
                    help="prep dry runs only: do not stop when per-table workbooks differ from their manifest (reported)")
    ap.add_argument("--pointers-out", default=None, help="write expected SI pointer paragraphs and differences (JSON)")
    args = ap.parse_args()
    TABLES, LAYOUT, SI_MD = Path(args.tables), Path(args.layout), Path(args.si)
    TABLE_MANIFEST = TABLES / "manifest.json"
    DATA = DELIVERY = Path(args.out)
    pc.BUILD_DATE = args.build_date
    if args.write_si_pointers and SI_MD.resolve() == pc.SI_MD.resolve():
        raise SystemExit("--write-si-pointers would rewrite the authoring SI; pass --si <copy> or use --pointers-out")
    if DATA.resolve() in (TABLES.resolve(), TABLES.resolve().parent) or any(
            p in DATA.resolve().parents for p in (OUT / "collab_draft_20261102" / f"pass_{n:03d}" for n in range(41, 48))):
        raise SystemExit(f"--out {DATA} would overwrite the per-table folder or a frozen pass")
    lay = load_layout()
    manifest = json.load(open(TABLE_MANIFEST))
    pending = [t for t, e in manifest.items() if "file" not in e]
    if pending:
        raise SystemExit(f"per-table package incomplete (pending: {pending}); run the assembler with --reader-tables first")
    bad_sha = [e["file"] for e in manifest.values() if pc.sha256(TABLES / e["file"]) != e["sha256"]]
    if bad_sha and args.skip_manifest_sha:
        print(f"WARNING (dry run): {len(bad_sha)} per-table workbooks differ from {TABLE_MANIFEST.name}")
    elif bad_sha:
        raise SystemExit(f"per-table workbooks differ from {TABLE_MANIFEST.name}: {bad_sha}")
    ents = si_entries(SI_MD)
    bind = pc.Bindings()
    bind.bind(LAYOUT, "layout")
    bind.bind(SI_MD, "Supplementary Information Markdown")
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    rep = pc.Report()
    caption_mismatch: dict = {}
    data_manifest: dict = {"build_date": pc.BUILD_DATE, "layout": pc.rel(LAYOUT), "layout_sha256": bind.sha(LAYOUT),
                           "script": args.script_label, "files": []}
    pointers: dict[str, str] = {}
    descriptions = []
    prov_files: dict[int, str] = {}
    for f in lay["data_files"]:
        k, title, tables = f["number"], f["title"], f["tables"]
        tables = sorted(tables, key=lambda s: int(s[1:]))
        wb = openpyxl.Workbook()
        wb.remove(wb.active)
        ws_idx = wb.create_sheet("Index")
        ws_cap = wb.create_sheet("Captions")
        ws_prov = None
        ws_idx.append([f"Supplementary Data {k}", title])
        ws_idx.append(["", "PROSPER: An Interpretable Whole Slide Image Search System for Pediatric Cancer Pathology. "
                           "Tables listed below; their titles and legends are in the Supplementary Information and in the sheet "
                           "Captions. Each table is a group of sheets whose names begin with its S-number."])
        ws_idx.append([])
        ws_idx.append(["Table (sheet group)", "Title", "Sheets in this workbook (sheet <- original name)",
                       "Legend"])
        for c in ws_idx[4]:
            c.font = Font(bold=True)
        ws_idx["A1"].font = Font(bold=True, size=13)
        ws_idx["B1"].font = Font(bold=True, size=13)
        ws_cap.append(["table", "field", "text"])
        for c in ws_cap[1]:
            c.font = Font(bold=True)
        used: set[str] = {"Index", "Captions", "Provenance"}
        file_rec = {"file": f"Supplementary_Data_{k}.xlsx", "title": title, "tables": {}}
        n_cells_total = 0
        for t in tables:
            src = TABLES / f"Supplementary_Table_{t}.xlsx"
            bind.bind(src, f"{t} per-table workbook")
            swb = openpyxl.load_workbook(src)
            if swb.sheetnames[0] != "caption":
                raise SystemExit(f"{t}: first sheet is {swb.sheetnames[0]!r}, expected 'caption'")
            data_sheets = [s for s in swb.sheetnames if s != "caption" and (s != "source" or t == "S48")]
            sheet_map: dict[str, str] = {}
            cells = 0
            for s in data_sheets:
                new = new_sheet_name(t, s, used)
                sheet_map[s] = new
            cap_sheet = []
            for row in swb["caption"].iter_rows(min_row=2, values_only=True):
                ws_cap.append([t, row[0], row[1]])
                if row[0] is not None or row[1] is not None:
                    cap_sheet.append((row[0], row[1]))
            # revision: the caption sheet (table, title, caption, notes) must equal the SI entry
            if cap_sheet != [tuple(r) for r in caption_rows(t, ents[t])]:
                caption_mismatch[t] = {"sheet": cap_sheet, "si": caption_rows(t, ents[t])}
            if "source" in swb.sheetnames and t != "S48":
                if ws_prov is None:
                    ws_prov = wb.create_sheet("Provenance", 2)
                    ws_prov.append(["table", "field", "value"])
                    for c in ws_prov[1]:
                        c.font = Font(bold=True)
                for row in swb["source"].iter_rows(min_row=2, values_only=True):
                    ws_prov.append([t, row[0], row[1]])
            for s in data_sheets:
                cells += copy_sheet(swb[s], wb.create_sheet(sheet_map[s]))
            status, pointer = pdf_status(t, lay, sheet_map, data_sheets)
            pointers[t] = pointer
            e = ents[t]
            ws_idx.append([f"Supplementary Data {k}, {t}", e["title"],
                           "; ".join(f"{sheet_map[s]} <- {s}" if sheet_map[s] != s else s for s in data_sheets),
                           e["caption"]])
            file_rec["tables"][t] = {"title": e["title"], "sheets": sheet_map, "source": src.name,
                                     "source_sha256": bind.sha(src),
                                     "cells": cells}
            n_cells_total += cells
        for col, w in zip("ABCD", (22, 60, 70, 120)):
            ws_idx.column_dimensions[col].width = w
        ws_idx.freeze_panes = "A5"
        for col, w in zip("ABC", (12, 22, 160)):
            ws_cap.column_dimensions[col].width = w
        if ws_prov is not None:
            for col, w in zip("ABC", (12, 40, 120)):
                ws_prov.column_dimensions[col].width = w
        prov_files[k] = "Provenance" if ws_prov is not None else ""
        out = DATA / file_rec["file"]
        pc.save_xlsx(wb, out)
        # read back: every data cell equals the per-table source (values), sheet count and names
        rb = openpyxl.load_workbook(out, read_only=True)
        n_back = 0
        for t, trec in file_rec["tables"].items():
            swb = openpyxl.load_workbook(TABLES / trec["source"], read_only=True)
            for s, new in trec["sheets"].items():
                a = [[v for v in r] for r in swb[s].iter_rows(values_only=True)]
                b = [[v for v in r] for r in rb[new].iter_rows(values_only=True)]
                # trailing empty rows/cols may differ in read-only mode; compare non-empty cells
                da = {(i, j): v for i, r in enumerate(a) for j, v in enumerate(r) if v is not None}
                db = {(i, j): v for i, r in enumerate(b) for j, v in enumerate(r) if v is not None}
                if da != db:
                    diff = [(key, da.get(key), db.get(key)) for key in set(da) | set(db) if da.get(key) != db.get(key)]
                    raise SystemExit(f"Supplementary Data {k}: {t} sheet {s!r} -> {new!r} differs: {diff[:3]}")
                n_back += len(db)
        rep.check(n_back == n_cells_total, f"Supplementary Data {k}: {len(tables)} tables, {len(rb.sheetnames)} sheets, "
                                           f"{n_back} non-empty cells read back equal to the per-table workbooks")
        file_rec.update(sheets=rb.sheetnames, cells=n_back, sha256=pc.sha256(out), bytes=out.stat().st_size)
        data_manifest["files"].append(file_rec)
        first, last = tables[0], tables[-1]
        descriptions.append((k, title, tables, f"Supplementary Data {k}: {title}. Excel workbook with the full content of "
                             f"tables {fmt_runs(tables)} ({len(rb.sheetnames) - (2 if ws_prov is None else 3)} data sheets named by table "
                             f"number), with an Index sheet and the titles and legends of every table."))
        print(f"Supplementary Data {k}: {len(tables)} tables, {len(rb.sheetnames)} sheets, {out.stat().st_size / 1e6:.2f} MB")
    n_tab = len(lay["_file_of"])
    rep.check(not caption_mismatch and len(ents) == n_tab,
              f"caption sheets (table, title, caption, notes) equal the SI entries for {n_tab - len(caption_mismatch)} of "
              f"{n_tab} tables (differences: {sorted(caption_mismatch, key=lambda x: int(x[1:]))})")
    if caption_mismatch and args.pointers_out:
        Path(str(args.pointers_out) + ".caption_mismatch.json").write_text(
            json.dumps(caption_mismatch, indent=1, ensure_ascii=False) + "\n")
    # rev3: no pointer paragraphs and no preview wording in the SI
    si_text = SI_MD.read_text(encoding="utf-8")
    rep.check(not si_pointer_lines() and "Workbook:" not in si_text,
              "SI Markdown has no 'Workbook:' pointer paragraph")
    rep.check(not lay["pdf_tables"] and "previewed" not in si_text.lower(),
              "layout prints no table in the SI and the SI does not mention previews")
    m = re.search(r"titles and legends of their (\d+) tables\)", si_text)
    rep.check(bool(m) and int(m.group(1)) == n_tab and f"carry all {n_tab} tables in full" in si_text,
              f"SI Contents line and table-section intro state {n_tab} tables")
    data_manifest["checks"] = rep.summary()
    data_manifest["check_list"] = rep.checks
    (DATA / "manifest.json").write_text(json.dumps(data_manifest, indent=1, ensure_ascii=False) + "\n")
    bind.write(DATA, {"build_date": pc.BUILD_DATE, "script": data_manifest["script"]})
    desc = ["# Supplementary Data files: titles and upload descriptions", "",
            "One line per file as entered in the submission system (title; description of at most 50 words).", ""]
    for k, title, tables, d in descriptions:
        desc += [f"## Supplementary Data {k}", "", f"Title: Supplementary Data {k}. {title}", "",
                 f"Description: {d}", "", f"Tables: {', '.join(tables)}", ""]
    (DATA / "FILE_DESCRIPTIONS.md").write_text("\n".join(desc), encoding="utf-8")
    has_prov = [f"Supplementary Data {k}" for k, _t, _tb, d in descriptions if "Provenance" in prov_files.get(k, "")]
    n_tables, n_prev = len(lay["_file_of"]), len(lay["pdf_tables"])
    prov_sentence = (f" {' and '.join(has_prov)} also {'has' if len(has_prov) == 1 else 'have'} a `Provenance` sheet for "
                     "the reader-study tables\n(analysis label, random seed, number of bootstrap draws, reader tier labels "
                     "and skipped-row notes).") if has_prov else ""
    readme = f"""# Supplementary Data 1–6

Six Excel workbooks that carry the full content of all {n_tables} tables of the Supplementary Information, grouped by theme.
Each table is cited as "Supplementary Data k, S#" and is a group of sheets whose names begin with its S-number. The
Supplementary Information gives the title and legend of every table. Where a table has both display and source forms, the display sheets or columns are rounded for
reading and the source or raw sheets and columns keep the unrounded values.

Each workbook has an `Index` sheet (one row per table, with the sheet names and the legend), a `Captions` sheet (the title, legend and notes of every table) and the data sheets, named
`S<n>_<sheet>`.{prov_sentence}

| File | Title | Tables |
|---|---|---|
""" + "\n".join(f"| Supplementary_Data_{k}.xlsx | {title} | {', '.join(tables)} |" for k, title, tables, _ in descriptions) + """

Upload titles and descriptions are listed in `FILE_DESCRIPTIONS.md`.
"""
    (DATA / "README.md").write_text(readme, encoding="utf-8")
    pc.write_sha256sums(DELIVERY)
    print(rep.summary())
    for s, t in rep.checks:
        print(f"  {s}: {t}")
    return 0 if rep.n_fail() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
