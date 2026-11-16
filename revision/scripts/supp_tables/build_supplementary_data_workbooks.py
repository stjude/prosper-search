#!/usr/bin/env python3
"""Supplementary Data 1–6: the 47 Supplementary Tables grouped into six themed Excel files (2026-11-10).

npj Digital Medicine asks for every cited supplementary item in one merged Supplementary Information PDF and, for
tables too large for the PDF, separate Excel files labelled "Supplementary Data N" with title and legend in the PDF.
This script takes the 47 assembled per-table workbooks of
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import packaging_common_20261109 as pc  # noqa: E402
from assemble_final_tables_20261110 import si_entries  # noqa: E402

pc.BUILD_DATE = "2026-11-10"
REPO, OUT = pc.REPO, pc.OUT_ROOT
SI_MD = pc.SI_MD
LAYOUT = REPO / "scripts/supp_tables/supplementary_data_layout.json"
PKG = OUT / "supp_package_20261110"
TABLES = PKG / "supplementary_tables"
DATA = PKG / "supplementary_data"
POINTER_RE = re.compile(r"^Workbook: Supplementary Data (\d+), sheets beginning (S\d+)_\. .*$")
SHEET_RE = re.compile(r"^S(\d+)([a-z]?)(?:[ _](.*))?$")


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
    return lay


def new_sheet_name(final: str, name: str, used: set[str]) -> str:
    """`S8` -> `S8_table`; `S8b_paired` -> `S8_b_paired`; `S9 body` -> `S9_body`; `Part1` -> `S12_Part1`."""
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
        return "no (workbook only)", f"{head} Not reproduced in this document."
    shown = [s["sheet"] for s in specs]
    for s in shown:
        if s not in sheet_map:
            raise SystemExit(f"{final}: pdf sheet {s!r} not in workbook ({list(sheet_map)})")
    subset = any(s.get("columns") for s in specs)
    cols = " (display columns; the source and unrounded columns are in the workbook only)" if subset else ""
    if set(shown) == set(data_sheets):
        return ("yes (all sheets)" if not subset else "yes (display columns)"), f"{head} Reproduced in full above{cols}."
    names = [sheet_map[s] for s in shown]
    lst = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return (f"partly ({len(shown)} of {len(data_sheets)} sheets)",
            f"{head} {'Sheet' if len(names) == 1 else 'Sheets'} {lst} {'is' if len(names) == 1 else 'are'} reproduced above"
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
        assert lines[i].startswith(f"Supplementary Table {t} | "), (t, lines[i][:60])
        lines[i + 1:i + 1] = ["", expected[t]]
        changes += 1
    if changes:
        SI_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changes


def main() -> int:
    ap = argparse.ArgumentParser(description="Supplementary Data 1–6 workbooks")
    ap.add_argument("--write-si-pointers", action="store_true",
                    help="insert/update the 'Workbook: Supplementary Data k ...' paragraph of every table in the SI Markdown")
    args = ap.parse_args()
    lay = load_layout()
    manifest = json.load(open(PKG / "manifest.json"))
    pending = [t for t, e in manifest.items() if "file" not in e]
    if pending:
        raise SystemExit(f"per-table package incomplete (pending: {pending}); run the assembler with --reader-tables first")
    ents = si_entries()
    bind = pc.Bindings()
    bind.bind(LAYOUT, "layout")
    bind.bind(SI_MD, "Supplementary Information Markdown")
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    rep = pc.Report()
    data_manifest: dict = {"build_date": pc.BUILD_DATE, "layout": pc.rel(LAYOUT), "layout_sha256": bind.sha(LAYOUT),
                           "script": "scripts/supp_tables/build_supplementary_data_workbooks.py", "files": []}
    pointers: dict[str, str] = {}
    descriptions = []
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
                           "Supplementary Tables listed below; captions and notes are in the Supplementary Information and in the "
                           "sheet Captions. Sheet names are prefixed with the table number."])
        ws_idx.append([])
        ws_idx.append(["Supplementary Table", "Title", "Sheets in this workbook (sheet <- original name)",
                       "Reproduced in the Supplementary Information", "Caption"])
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
            for row in swb["caption"].iter_rows(min_row=2, values_only=True):
                ws_cap.append([t, row[0], row[1]])
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
            ws_idx.append([f"Supplementary Table {t}", e["title"],
                           "; ".join(f"{sheet_map[s]} <- {s}" if sheet_map[s] != s else s for s in data_sheets),
                           status, e["caption"]])
            file_rec["tables"][t] = {"title": e["title"], "sheets": sheet_map, "source": src.name,
                                     "source_sha256": bind.sha(src), "reproduced_in_si": status,
                                     "cells": cells, "si_pointer": pointer}
            n_cells_total += cells
        for col, w in zip("ABCDE", (22, 60, 70, 30, 120)):
            ws_idx.column_dimensions[col].width = w
        ws_idx.freeze_panes = "A5"
        for col, w in zip("ABC", (12, 22, 160)):
            ws_cap.column_dimensions[col].width = w
        if ws_prov is not None:
            for col, w in zip("ABC", (12, 40, 120)):
                ws_prov.column_dimensions[col].width = w
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
                             f"Supplementary Tables {', '.join(tables[:-1])} and {last} of the Supplementary Information "
                             f"({len(rb.sheetnames) - (2 if ws_prov is None else 3)} data sheets, prefixed with the table number), "
                             f"an Index sheet and the captions and notes of every table."))
        print(f"Supplementary Data {k}: {len(tables)} tables, {len(rb.sheetnames)} sheets, {out.stat().st_size / 1e6:.2f} MB")
    # SI pointer paragraphs
    existing = si_pointer_lines()
    if args.write_si_pointers:
        n = write_si_pointers(pointers, ents)
        rep.check(True, f"SI Markdown pointer paragraphs written/updated: {n}")
        bind.bind(SI_MD, "Supplementary Information Markdown (after pointer update)")
        existing = si_pointer_lines()
    bad = [t for t, p in pointers.items() if existing.get(t, (None, None))[1] != p]
    rep.check(not bad, f"SI Markdown pointer paragraphs match the layout for all {len(lay['_file_of'])} tables (differences: {bad})")
    si_text = SI_MD.read_text(encoding="utf-8")
    m = re.search(r"captions and notes for all \d+; (\d+) reproduced in this document", si_text)
    n_pdf = len(lay["pdf_tables"])
    rep.check(bool(m) and int(m.group(1)) == n_pdf,
              f"SI Contents line states {m.group(1) if m else '?'} reproduced tables; layout has {n_pdf}")
    words = {19: "Nineteen", 20: "Twenty", 18: "Eighteen", 21: "Twenty-one", 17: "Seventeen"}
    rep.check(f"{words.get(n_pdf, str(n_pdf))} tables of moderate size are reproduced here" in si_text,
              f"SI table-section intro states {words.get(n_pdf, n_pdf)} reproduced tables")
    if bad and not args.write_si_pointers:
        for t in bad[:5]:
            print(f"  {t}: SI has {existing.get(t, (None, '<none>'))[1]!r}\n      expected {pointers[t]!r}")
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
    readme = f"""# Supplementary Data 1–6 (assembled {pc.BUILD_DATE})

Six Excel workbooks that carry the full content of the 47 Supplementary Tables, grouped by theme
(`scripts/supp_tables/supplementary_data_layout.json`). The journal's Supplementary Information PDF reproduces the
tables that fit (display sheets; see each table's "Workbook:" line in the Supplementary Information) and carries the
title, caption and notes of every table; these files are the complete, unrounded form and are labelled
"Supplementary Data k" as the journal requires for Excel files.

Each workbook: `Index` (one row per table; sheet names old -> new; whether the PDF reproduces the table), `Captions`
(every caption-sheet row of every table), `Provenance` (reader-study tables: analysis directory, commit, CSV hashes)
and the data sheets, named `S<n>_<sheet>`. Values were copied cell for cell from
`../supplementary_tables/Supplementary_Table_S<n>.xlsx` (the per-table workbooks, kept as the source form) and read
back for equality (`manifest.json`, field `check_list`). Inputs are bound in `input_bindings.json`.

| File | Title | Tables |
|---|---|---|
""" + "\n".join(f"| Supplementary_Data_{k}.xlsx | {title} | {', '.join(tables)} |" for k, title, tables, _ in descriptions) + """

Upload titles and descriptions: `FILE_DESCRIPTIONS.md`. Build:
`scripts/supp_tables/build_supplementary_data_workbooks.py` (after `assemble_final_tables_20261110.py --reader-tables ...`,
before `scripts/render_supplement_docx.py --verify`, which reproduces the display sheets in the PDF from the per-table
workbooks).
"""
    (DATA / "README.md").write_text(readme, encoding="utf-8")
    pc.write_sha256sums(PKG)
    print(rep.summary())
    for s, t in rep.checks:
        print(f"  {s}: {t}")
    return 0 if rep.n_fail() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
