#!/usr/bin/env python3
"""revision: per-table workbook Supplementary_Table_S49.xlsx (Supplementary Data 6, S49) from four analysis-layer CSVs
of the exploratory search-alone comparison (read-only; never re-run), plus tables_050/per_table_workbooks = byte copies
of tables_048/per_table_workbooks S1-S48 (and its reader_tables_manifest.json, unchanged) + S49 + manifest.json
(the tables_048 manifest with one S49 entry appended).

Form as the reader-study per-table workbooks of revision (tables_048/per_table_workbooks/S31, S32, S39-S47):
first sheet `caption` (field | text; column B width 160) written from the SI Markdown with si_entries_044.caption_rows,
then the data sheets, header row = CSV header, values converted as scripts/supp_tables/build_reader_study_tables.py
`coerce` (integers, decimals, empty -> blank; other text unchanged), no styles, no widths, no freeze panes.
Saved byte-deterministically with packaging_common.save_xlsx (build date 2026-11-16) patched by apply_tables_046_save
(document author "PROSPER authors"), as the grouped builder saves.

Sheets (brief of revision):
  search_alone        V1 without the row(s) whose search_criterion starts with "summary's stated leading group"
  who_corrected_whom  V8, all rows and columns
  by_vote_outcome     V7, all rows and columns
  search_vs_reader    V2 rows with reader exactly "Reader 1".."Reader 5", without column mcnemar_exact_p_unadjusted
Usage (from anywhere): make_s49_pass050.py"""
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import openpyxl

C = Path(__file__).resolve().parents[2]
OUTROOT = C.parent
REPO = OUTROOT.parent
sys.path.insert(0, str(REPO / "scripts/supp_tables"))
import packaging_common_20261109 as pc  # noqa: E402
sys.path.insert(0, str(OUTROOT / "e1_si_close_audit_20261115/scratch/si-tables"))
import si_entries_044 as sie  # noqa: E402
import apply_tables_046_save  # noqa: E402,F401

pc.BUILD_DATE = "2026-11-16"
A = OUTROOT / "reader_study_suite/analysis/20261116_prod_export_20261116-085036_five_readers_search_alone_vs_readers"
SI = REPO / "docs/prosper_revised/prosper_supplementary_legends.md"
SRC_T = C / "tables_048/per_table_workbooks"
DST_T = C / "tables_050/per_table_workbooks"
CHK = C / "checks_050"
HASH = {"V1_search_alone.csv": "0a85b3fbcef0e8c4", "V2_search_vs_reader_favoured.csv": "a19c053ba001069d",
        "V7_by_vote_outcome_per_reader.csv": "43edc34cc675ba2e", "V8_who_corrected_whom.csv": "483fe68ea460858d"}
READERS = [f"Reader {i}" for i in range(1, 6)]
LOGIN_RE = re.compile(r"\b[a-z]+\.[a-z]+\b")
NUM_RE = re.compile(r"^-?\d+$|^-?\d*\.\d+(e-?\d+)?$")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def coerce(v: str):  # = scripts/supp_tables/build_reader_study_tables.py::coerce
    if v == "":
        return None
    if NUM_RE.match(v):
        return int(v) if re.fullmatch(r"-?\d+", v) else float(v)
    return v


def read(name):
    p = A / name
    assert sha(p).startswith(HASH[name]), (name, sha(p))
    with open(p, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


assert sha(SI).startswith("704daa6d2d15d4fa"), sha(SI)
assert not DST_T.exists(), DST_T
rec = {"inputs": {n: sha(A / n) for n in HASH}, "si_sha256": sha(SI), "sheets": {}}

h1, b1 = read("V1_search_alone.csv")
i = h1.index("search_criterion")
drop1 = [r for r in b1 if r[i].startswith("summary's stated leading group")]
s_alone = (h1, [r for r in b1 if not r[i].startswith("summary's stated leading group")])
h8, b8 = read("V8_who_corrected_whom.csv")
h7, b7 = read("V7_by_vote_outcome_per_reader.csv")
h2, b2 = read("V2_search_vs_reader_favoured.csv")
ri, mi = h2.index("reader"), h2.index("mcnemar_exact_p_unadjusted")
keep2 = [r for r in b2 if r[ri] in READERS]
s_vs = ([h for j, h in enumerate(h2) if j != mi], [[v for j, v in enumerate(r) if j != mi] for r in keep2])
sheets = [("search_alone", s_alone), ("who_corrected_whom", (h8, b8)), ("by_vote_outcome", (h7, b7)),
          ("search_vs_reader", s_vs)]
rec["dropped"] = {"search_alone": {"rows_removed": len(drop1), "criterion": sorted({r[i] for r in drop1})},
                  "search_vs_reader": {"rows_removed": len(b2) - len(keep2),
                                       "readers_removed": sorted({r[ri] for r in b2 if r[ri] not in READERS}),
                                       "column_removed": "mcnemar_exact_p_unadjusted"}}

ents = sie.si_entries(SI)
assert "S49" in ents and ents["S49"]["file"] == 6, ents.get("S49")


def build_wb():
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, (hdr, body) in sheets:
        ws = wb.create_sheet(name)
        ws.append(hdr)
        for r in body:
            assert len(r) == len(hdr), (name, r)
            ws.append([coerce(v) for v in r])
        rec["sheets"][name] = {"rows": len(body), "columns": len(hdr), "header": hdr}
    ws = wb.create_sheet("caption", 0)  # as implement_data6.build_caption (revision): field | text, column B width 160
    ws.append(["field", "text"])
    for f, t in sie.caption_rows("S49", ents["S49"]):
        ws.append([f, t])
    ws.column_dimensions["B"].width = 160
    return wb


wb = build_wb()
rec["caption_rows"] = sie.caption_rows("S49", ents["S49"])

# identifier checks: no login-like token, no pooled/consensus row, only Reader 1-5 as readers, US spelling
bad = []
for w in wb.worksheets:
    for row in w.iter_rows(values_only=True):
        for v in row:
            if isinstance(v, str) and (LOGIN_RE.search(v) or re.search(r"\bpooled\b|favour|neighbour|tumour", v, re.I)):
                bad.append((w.title, v[:40]))
    if w.title != "caption":
        hdr = [c.value for c in w[1]]
        if "reader" in hdr:
            vals = {r[hdr.index("reader")] for r in w.iter_rows(min_row=2, values_only=True)}
            assert vals == set(READERS), (w.title, vals)
assert not bad, bad[:5]

DST_T.mkdir(parents=True)
for p in sorted(SRC_T.iterdir()):
    if p.name.startswith("Supplementary_Table_S") or p.name == "reader_tables_manifest.json":
        shutil.copyfile(p, DST_T / p.name)
        assert sha(p) == sha(DST_T / p.name), p.name
out = DST_T / "Supplementary_Table_S49.xlsx"
pc.save_xlsx(wb, out)
man = json.loads((SRC_T / "manifest.json").read_text())
assert list(man)[-1] == "S48" and "S49" not in man
for t, e in man.items():
    assert sha(DST_T / e["file"]) == e["sha256"], t
man["S49"] = {"workspace_id": "S-RS-V", "file": out.name, "sha256": sha(out),
              "source": "checks_050/reproduction/make_s49_pass050.py on $PROSPER_OUT/reader_study_suite/analysis/"
                        f"{A.name} (V1, V8, V7, V2)",
              "source_sha256": {n: sha(A / n) for n in HASH},
              "title": ents["S49"]["title"], "data_file": 6, "si_line": ents["S49"]["line"],
              "sheets": wb.sheetnames,
              "data_changes": "pass 050: new table (exploratory comparison of readers' answers with a plurality vote over "
                              "the retrieved labels); caption sheet from the SI",
              "reader_study_build": "analysis of the export of 16 November 2026 (all five readers at 80/80 cases); "
                                    "search_vs_reader.py commit dc62639"}
(DST_T / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
# second, independent build in a scratch directory: byte-determinism
pc.save_xlsx(build_wb(), CHK / "scratch/S49_second_save.xlsx")  # independent second build
rec["deterministic"] = sha(CHK / "scratch/S49_second_save.xlsx") == sha(out)
(CHK / "scratch/S49_second_save.xlsx").unlink()
rec["output"] = {"file": str(out.relative_to(C)), "sha256": sha(out), "sheets": wb.sheetnames}
rec["copied_byte_identical_from_tables_048"] = sorted(p.name for p in DST_T.iterdir()
                                                       if p.name not in ("Supplementary_Table_S49.xlsx", "manifest.json"))
rec["manifest_json"] = {"tables_048": sha(SRC_T / "manifest.json"), "tables_050": sha(DST_T / "manifest.json"),
                        "change": "one S49 entry appended; S1-S48 entries unchanged"}
(CHK / "tables_build").mkdir(exist_ok=True)
(CHK / "tables_build/s49_build_record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
print(json.dumps({k: rec[k] for k in ("sheets", "dropped", "deterministic", "output")}, indent=1)[:3000])
