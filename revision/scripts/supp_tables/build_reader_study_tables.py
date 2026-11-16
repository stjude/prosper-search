#!/usr/bin/env python3
"""Reader-study supplementary workbooks (final numbers S31, S32, S39–S47) from one five-reader analysis directory.

Input (read-only): the frozen analysis directory written by `scripts/reader_study/analyze.py`
(`tables/*.csv`, `analyze_run.json`) and its `_qc74` companion (`QC_sensitivity_trajectory.csv`). The reader-study
export, database and backups are never opened here (AGENTS.md, "Pathologist evaluation results are read-only");
this script only repackages analysis-layer CSVs.

Output: `$PROSPER_OUT/supp_tables_reader_study/<analysis-dir-name>/Supplementary_Table_S<n>_source.xlsx`, one
workbook per table with the sheets named below, reader identifiers replaced by the manuscript's "Reader n" labels
(order = `analyze_run.json` "readers", the order Fig. 8 and Supplementary Note 3 use), plus a `source` sheet with
the per-CSV hashes. `scripts/supp_tables/assemble_final_tables_20261110.py --reader-tables <this dir>` then adds
the caption sheet (from the SI Markdown), spelling normalization, the identifier scan and the final file names.

Checks (BUILD_REPORT.md): reader order equals the Fig. 8 default; readings per reader equal the counts stated in
Supplementary Note 3 ("Readers and expertise tier"); no reader login or excluded-reader name survives in any cell;
every sheet non-empty with the expected header.

All-reader rows (revision, finding I1): `analyze.py::reader_sets()` appends a combined reader set labelled
`pooled`. No published estimate pools readers (Methods; SI Note 3; locked plan candidate 003, amendment 004 item 2),
so every row whose reader column (READER_COLS) equals that label is skipped in every sheet; the per-sheet count of
skipped rows is written to the source sheet, manifest.json and BUILD_REPORT.md, and a global check asserts that the
word does not occur (case-insensitive, whole word) in any cell of any sheet. The analysis layer is unchanged.

Usage (interim refresh on the 14 Nov export; the final run is the same command with the new directories):
  export PROSPER_OUT="$PWD/prosper_revision_output"
  python3 scripts/supp_tables/build_reader_study_tables.py \
      --analysis "$PROSPER_OUT/reader_study_suite/analysis/20261114_prod_export_20261114-095616_five_readers" \
      --qc74 "$PROSPER_OUT/reader_study_suite/analysis/20261114_prod_export_20261114-095616_five_readers_qc74" \
      --label "interim refresh on the export of 14 November 2026 (Reader 3 at 73/80)"
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
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "reader_config", Path(__file__).resolve().parents[1] / "reader_study" / "reader_config.py")
_reader_config = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_reader_config)

FIG8_READER_ORDER = list(_reader_config.load_reader_config()["fig8_reader_order"])

# final number -> (workspace id, [(sheet name, csv relative to the analysis dir or 'QC74:<name>')])
TABLES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "S31": ("S-RS-T1", [("reader_case_flow", "tables/T1_reader_case_flow.csv")]),
    "S32": ("S-RS-T2", [("primary_endpoints", "tables/T2_primary_endpoints.csv"),
                        ("trajectory_three_steps", "tables/S1_trajectory_5_1.csv")]),
    "S39": ("S-RS-T3", [("outcome_classes", "tables/T3_outcome_classes.csv")]),
    "S40": ("S-RS-T4", [("strata", "tables/T4_strata_5_3.csv")]),
    "S41": ("S-RS-T5", [("engagement", "tables/T5a_engagement_5_4.csv"),
                        ("endpoints_by_engagement", "tables/T5b_by_engagement_5_4.csv")]),
    "S42": ("S-RS-T6", [("reading_time", "tables/T6_reading_time_5_6.csv"),
                        ("interrupted_readings", "tables/T6b_interrupted_readings.csv")]),
    "S43": ("S-RS-T7", [("step3_summary", "tables/T7_llm_summary_5_9.csv"),
                        ("summary_contradicts_reader", "tables/T7b_summary_contradicts_reader.csv")]),
    "S44": ("S-RS-T8", [("vocabulary", "tables/S3_vocabulary_5_8.csv"),
                        ("entity_map_applications", "tables/S4_entity_map_applications_5_8.csv"),
                        ("comment_named_entities", "tables/S4b_comment_named_entities_5_8.csv")]),
    "S45": ("S-RS-T9", [("qc74_sensitivity", "QC74:QC_sensitivity_trajectory.csv")]),
    "S46": ("S-RS-T10", [("between_reader", "tables/T10_between_reader_5_7.csv"),
                         ("paired_cases", "tables/T10b_paired_cases_5_7.csv")]),
    "S47": ("S-RS-T11", [("mislead", "tables/T8a_mislead_list.csv"), ("rescue", "tables/T8b_rescue_list.csv"),
                         ("misread", "tables/T8c_misread_list.csv"),
                         ("harmful_confidence", "tables/T8d_harmful_confidence_list.csv")]),
}
READER_COLS = {"reader_id", "reader1_id", "reader2_id"}
ALL_READER_SET = "pooled"  # label of the combined reader set in scripts/reader_study/analyze.py reader_sets()
ALL_READER_WORD_RE = re.compile(r"\bpooled\b", re.IGNORECASE)
LOGIN_RE = re.compile(r"\b[a-z]+\.[a-z]+\b")  # firstname.lastname logins (readers analysed or excluded)
NUM_RE = re.compile(r"^-?\d+$|^-?\d*\.\d+(e-?\d+)?$")


def coerce(v: str):
    if v == "":
        return None
    if NUM_RE.match(v):
        return int(v) if re.fullmatch(r"-?\d+", v) else float(v)
    return v


def note3_counts(si_text: str) -> dict[str, int]:
    """'Reader n (…; N cases' or '…; N STV cases' from the Note 3 paragraph 'Readers and expertise tier'."""
    m = re.search(r"\*\*Readers and expertise tier\.\*\*(.*?)\n", si_text)
    para = m.group(1) if m else ""
    out = {}
    for rm in re.finditer(r"Reader (\d) \(", para):
        tail = para[rm.end():]
        cm = re.search(r"(\d+) (?:STV |CCDI )?cases", tail)
        if cm:
            out[f"Reader {rm.group(1)}"] = int(cm.group(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True, type=Path)
    ap.add_argument("--qc74", required=True, type=Path)
    ap.add_argument("--label", required=True, help="provenance label written to every source sheet")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or (pc.OUT_ROOT / "supp_tables_reader_study" / a.analysis.name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    bind = pc.Bindings()
    run = json.load(open(bind.bind(a.analysis / "analyze_run.json", "analysis run record")))
    readers: list[str] = run["readers"]
    tiers: list[str] = run["tiers"]
    label_of = {r: f"Reader {i + 1}" for i, r in enumerate(readers)}
    rep_global = pc.Report()
    rep_global.check(readers == FIG8_READER_ORDER[: len(readers)],
                     f"reader order equals the Fig. 8 default order ({len(readers)} readers)")
    si_text = pc.SI_MD.read_text(encoding="utf-8")
    bind.bind(pc.SI_MD, "SI Markdown (Note 3 reader counts)")
    expected = note3_counts(si_text)

    manifest, report_rows = {}, []
    leaks_all: list[tuple[str, str, str]] = []
    word_hits: list[tuple[str, str, str]] = []
    dropped_all: dict[str, dict[str, int]] = {}
    for final, (wid, sheets) in TABLES.items():
        rep = pc.Report()
        wb = openpyxl.Workbook()
        wb.remove(wb.active)
        src_rows = [("label", a.label), ("analysis_dir", pc.rel(a.analysis)), ("qc74_dir", pc.rel(a.qc74)),
                    ("export", Path(run["command"].split("--export")[-1].strip()).name),
                    ("analysis_commit", run["commit"]), ("readings_sha256", run["readings_sha256"]),
                    ("seed", str(run["seed"])), ("n_draws", str(run["n_draws"])),
                    ("reader_labels", "; ".join(f"{label_of[r]} = {t} tier" for r, t in zip(readers, tiers)))]
        dropped_all[final] = {}
        for sheet, relp in sheets:
            p = (a.qc74 / relp[5:]) if relp.startswith("QC74:") else (a.analysis / relp)
            bind.bind(p, f"{final} sheet {sheet}")
            with open(p, newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            hdr, body = rows[0], rows[1:]
            ws = wb.create_sheet(sheet)
            ws.append(hdr)
            ridx = [i for i, h in enumerate(hdr) if h in READER_COLS]
            n_dropped = 0
            for r in body:
                if any(r[i] == ALL_READER_SET for i in ridx if i < len(r)):
                    n_dropped += 1          # all-reader row: not published (revision, I1)
                    continue
                vals = []
                for i, v in enumerate(r):
                    if i in ridx:
                        vals.append(label_of.get(v, v))
                    else:
                        for login, lab in label_of.items():
                            if login in v:
                                v = v.replace(login, lab)
                        vals.append(coerce(v))
                ws.append(vals)
            dropped_all[final][sheet] = n_dropped
            n_kept = len(body) - n_dropped
            empty_ok = n_kept == 0 and sheet in {"rescue", "misread", "mislead", "harmful_confidence",
                                                      "interrupted_readings", "comment_named_entities"}
            rep.check(len(hdr) > 1 and (n_kept > 0 or empty_ok),
                      f"sheet {sheet}: {n_kept} rows × {len(hdr)} columns from {p.name} "
                      f"({n_dropped} all-reader rows of {len(body)} skipped)"
                      + (" (listing empty: no readings in this class at this export)" if empty_ok else ""))
            src_rows.append((f"sheet:{sheet}", f"{relp} sha256 {bind.sha(p)}"
                             + ("; no readings in this class at this export (header only)" if empty_ok else "")))
            src_rows.append((f"all_reader_rows_skipped:{sheet}",
                             f"{n_dropped} of {len(body)} analysis rows skipped (combined reader set; no published "
                             "estimate combines readers)"))
            for row in ws.iter_rows(min_row=2, values_only=True):
                for v in row:
                    if isinstance(v, str):
                        for tok in LOGIN_RE.findall(v):
                            if tok in readers or tok.count(".") == 1 and tok not in {"e.g", "i.e", "vs.", "et.al"}:
                                leaks_all.append((final, sheet, tok))
        if final == "S31":
            ws = wb["reader_case_flow"]
            hdr = [c.value for c in ws[1]]
            got = {r[hdr.index("reader_id")]: r[hdr.index("readings_A_and_B")] for r in ws.iter_rows(min_row=2, values_only=True)}
            rep.check(got == expected, f"readings with Steps 1–2 per reader equal Note 3: table {got}; Note 3 {expected}")
        ws = wb.create_sheet("source", 0)
        ws.append(["field", "value"])
        for k, v in src_rows:
            ws.append([k, v])
        ws.column_dimensions["A"].width = 24
        ws.column_dimensions["B"].width = 140
        for wsx in wb.worksheets:
            for row in wsx.iter_rows(values_only=True):
                for v in row:
                    if isinstance(v, str) and ALL_READER_WORD_RE.search(v):
                        word_hits.append((final, wsx.title, v[:80]))
        path = out / f"Supplementary_Table_{final}_source.xlsx"
        pc.save_xlsx(wb, path)
        manifest[final] = {"workspace_id": wid, "file": path.name, "sha256": pc.sha256(path),
                           "sheets": wb.sheetnames, "all_reader_rows_skipped": dropped_all[final],
                           "checks": rep.summary()}
        report_rows.append((final, wid, rep.summary(), " / ".join(f"{s}: {t}" for s, t in rep.checks)))
    rep_global.check(not leaks_all, f"no reader login survives in any cell ({len(leaks_all)} hits: {leaks_all[:5]})")
    rep_global.check(not word_hits, "no all-reader row or label survives: the word '" + ALL_READER_SET + "' occurs in no "
                     f"cell of any sheet ({len(word_hits)} hits: {word_hits[:5]})")
    n_skip = {f"{t}/{s}": n for t, d in dropped_all.items() for s, n in d.items() if n}
    rep_global.check(True, f"all-reader rows skipped per sheet (nonzero only): {n_skip}")

    lines = [f"# Reader-study workbooks (build {pc.BUILD_DATE})", "", f"Label: {a.label}", "",
             f"Analysis `{pc.rel(a.analysis)}` (commit {run['commit'][:12]}, readings sha256 {run['readings_sha256'][:16]}…), "
             f"QC-74 `{pc.rel(a.qc74)}`.", "", "## Global checks", ""]
    lines += [f"- {s}: {t}" for s, t in rep_global.checks]
    lines += ["", "## Per table", "", "| Final | Workspace id | Checks | Detail |", "|---|---|---|---|"]
    lines += ["| " + " | ".join(str(x).replace("|", "/") for x in r) + " |" for r in report_rows]
    (out / "BUILD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"label": a.label, "analysis": pc.rel(a.analysis), "qc74": pc.rel(a.qc74),
                                                   "readers": {label_of[r]: t for r, t in zip(readers, tiers)},
                                                   "all_reader_rows_skipped": dropped_all,
                                                   "tables": manifest}, indent=1, ensure_ascii=False) + "\n")
    bind.write(out)
    pc.write_sha256sums(out)
    fails = [t for s, t in rep_global.checks if s != "PASS"] + [r[3] for r in report_rows if ", 0 FAIL" not in r[2]]
    print("wrote", out, "|", rep_global.summary(), "|", "FAIL" if fails else "all tables PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
