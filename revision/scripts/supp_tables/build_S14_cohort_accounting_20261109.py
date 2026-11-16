#!/usr/bin/env python3
"""Workspace Table S14 (final number from scripts/figures/supp_table_numbering.json): cohort summary and accounting.

Assembly only (A_packaging, 2026-11-09). Spec: docs/manuscript_supplement_edits.md ### R1.3-T7 (two sections, historical
fields, caption) as amended by ### R3.3-T1 (populate from the r3_3 v3 CSVs; strip provenance tags from published cells;
omit the Virchow2 column; historical count column; footnotes a–g verbatim; caption append; Do-not-add). Third sheet
`material_and_labels`: the SI "Material and label sources" note (moved from Table 1 in revision), one row per
collection, every cell a verbatim substring of that note. Cross-check against main Table 1.

Run from the repository root:
    export PROSPER_OUT="$PWD/prosper_revision_output"
    /HOME/miniconda3/bin/python3 scripts/supp_tables/build_S14_cohort_accounting_20261109.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from packaging_common_20261109 import (BUILD_DATE, EDIT_SHEET, NUMBERING, OUT_ROOT, PKG, REPO, SI_MD, Bindings,  # noqa: E402
                                       Report, caption_fields, find_line, md_table_after, now_utc, numbering, rel,
                                       save_xlsx, write_caption_sheet, write_rows_sheet, write_sha256sums,
                                       write_sheet_csv)

WID = "S14"
OUT = PKG / WID
V3 = OUT_ROOT / "r3_3_cohort_accounting_20261008_v3"
T1 = OUT_ROOT / "main_tables_20261022"
MS = REPO / "docs/prosper_revised/prosper_manuscript_revised.md"

# provenance codes used in the v3 CSV cells (README "Sources" + tagged constants)
CODE = re.compile(r"^(?:PM|RM|SM|ZT|C12|KE|SY|CM|CP|V03|CMD|S15|HX|T7|T1|U|R\d\.\d|ADR\d{4}|plan0002|MS L[\d–\-]+(?:/L[\d–\-]+)*)$")

MSREF = re.compile(r"^MS L[\d–\-]+(?:/L[\d–\-]+)*$")
CUR_COLS = [("cohort", "Collection"), ("role", "Role"), ("slides_delivered", "Slides delivered"),
            ("slides_evaluable", "Slides evaluable"), ("slides_evaluated", "Slides evaluated"), ("patients", "Patients"),
            ("groups", "Groups"), ("specimen_types", "Specimen types"), ("label_source", "Label source"),
            ("submitted_to_current", "Submitted → current"), ("exclusions", "Exclusions")]
HIST_TITLE = "With retained UNI2 retrieval output (submitted count in footnote)"


def strip_tags(text: str, log: list) -> str:
    """Remove bracketed provenance tags; a non-code part inside a tag (a qualifier) is kept in parentheses."""
    def rep(m):
        parts = [p.strip() for p in m.group(1).split(";")]
        codes = [p for p in parts if MSREF.match(p) or all(CODE.match(x) for x in p.split("/"))]
        keep = [p for p in parts if p not in codes]
        log.append((m.group(0), f"({'; '.join(keep)})" if keep else ""))
        return f" ({'; '.join(keep)})" if keep else ""
    out = re.sub(r"\s*\[([^\[\]]+)\]", rep, text)
    return out.strip()


def nums(s: str) -> set[str]:
    return {x.replace(",", "") for x in re.findall(r"\d[\d,]*(?:\.\d+)?", str(s))}


def main() -> int:
    R = Report()
    B = Bindings()
    num = numbering(WID)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit").mkdir(exist_ok=True)

    f_cur = B.bind(V3 / "table_s14_replacement_current_cohorts.csv", "current section")
    f_his = B.bind(V3 / "table_s14_replacement_historical_collections.csv", "historical section")
    f_md = B.bind(V3 / "table_s14_replacement.md", "rendered draft (read; cross-check)")
    f_cnt = B.bind(V3 / "counts.json", "counts / assertions; per-group main populations")
    f_rd = B.bind(V3 / "README.md", "r3_3 v3 README (read; tag legend)")
    f_sha = B.bind(V3 / "sha256.txt", "r3_3 v3 own hash list")
    f_t1 = B.bind(T1 / "Table_1_cohort_accounting.csv", "main Table 1 (cross-check)")
    B.bind(EDIT_SHEET, "spec")
    B.bind(SI_MD, "SI title/caption/notes (live file; hash at build)")
    B.bind(MS, "manuscript Table 1 (live file; cross-check)")
    B.bind(NUMBERING, "numbering")

    own = dict(reversed(l.split()) for l in f_sha.read_text().splitlines() if l.strip())
    for f in (f_cur, f_his, f_md, f_cnt, f_rd):
        R.check(own.get(f.name) == B.sha(f), f"`{f.name}` sha256 {B.sha(f)[:12]}… equals r3_3 v3 sha256.txt")
    for f, pre in ((f_cur, "6159bc05"), (f_his, "2f35d076"), (f_md, "f78b8d45"), (f_cnt, "3de05ea2")):
        R.check(B.sha(f).startswith(pre), f"`{f.name}` is the spec-cited version ({pre}…)")
    cur = pd.read_csv(f_cur, dtype=str).fillna("")
    his = pd.read_csv(f_his, dtype=str).fillna("")
    cnt = json.load(open(f_cnt))
    R.check(all(v["got"] == v["expected"] for v in cnt["assertions"].values()),
            f"counts.json: all {len(cnt['assertions'])} recorded assertions have got == expected")

    # ---------------------------------------------------------------- current section (tags stripped)
    log: list = []
    pub_cur = [[strip_tags(r[c], log) for c, _ in CUR_COLS] for _, r in cur.iterrows()]
    left = [c for row in pub_cur for c in row if "[" in c or "]" in c]
    R.check(not left, f"current section: {len(log)} bracketed tags removed from {len(pub_cur)} rows; no bracket left "
                      f"in published cells ({len(left)})")
    kept = sorted({(a, b) for a, b in log if b})
    R.check(True, "tags with a non-code qualifier kept in parentheses: " + "; ".join(f"`{a.strip()}` → `{b}`" for a, b in kept))
    raw_nums = [nums(" ".join(r[c] for c, _ in CUR_COLS)) for _, r in cur.iterrows()]
    pub_nums = [nums(" ".join(row)) for row in pub_cur]
    tag_nums = [nums(" ".join(re.findall(r"\[[^\[\]]+\]", " ".join(r[c] for c, _ in CUR_COLS)))) for _, r in cur.iterrows()]
    lost = [sorted(a - b) for a, b in zip(raw_nums, pub_nums)]
    R.check(all(set(l) <= t for l, t in zip(lost, tag_nums)),
            f"stripping removed only numbers that occur inside the tags themselves (manuscript line refs, plan0002, R2.1, "
            f"T7 …): per row {lost}; no count left the published cells")

    # spec current-section check values (R3.3-T1)
    l_t = find_line(EDIT_SHEET, "### R3.3-T1 — Supplementary Table S14: content replacement")
    lh, hdr_v, rows_v = md_table_after(EDIT_SHEET, "| Row | Delivered → evaluable → evaluated (slides) | Patients |", l_t)
    for rv, row in zip(rows_v, pub_cur):
        want = nums(rv[1] + " " + rv[2]) - {"16738"}
        have = nums(" ".join(row))
        miss = sorted(want - have, key=lambda x: float(x))
        R.check(not miss, f"current row '{row[0]}': every count of the R3.3-T1 check table (l.{lh}) occurs in the "
                          f"published row; missing {miss}")

    # ---------------------------------------------------------------- historical section
    pub_his, his_log = [], []
    for _, r in his.iterrows():
        name = r.collection
        if name == "COMET RMS arispe":
            his_log.append("collection 'COMET RMS arispe' printed as 'COMET RMS extension' (R1.3-T7 row name; R3.3-T1 "
                           "'COMET RMS extension (\"arispe\")'; SI notes)")
            name = "COMET RMS extension"
        unit = "cores" if "cores (TMA)" in r.note else "slides"
        note = "" if r.note.startswith("T7:") or r.note.startswith("cores (TMA)") else r.note
        if r.note.startswith("T7:"):
            his_log.append(f"{r.collection}: note '{r.note}' is an editing instruction; applied check below, not printed")
        pub_his.append([name, int(r.retained_uni2_output_rows), unit, r.material_stain_S14, r.labels_used_S14,
                        r.alignment_to_COMET_S14, r.source_S14, r.patients, r.current_comparison, "not harmonized", note])
    hist_hdr = ["Collection", HIST_TITLE, "Unit", "Material/stain", "Labels", "Alignment to COMET", "Source",
                "Patients", "Current-comparison inclusion", "Endpoint/run provenance", "Note"]
    printed = [r[1] for r in pub_his]
    R.check(printed == [99, 6, 55, 167, 515], f"historical printed counts {printed} = R3.3-T1 (99 cores / 6 / 55 / 167 / 515)")
    R.check([int(x) for x in his.submitted_count_S14] == [102, 6, 58, 167, 515],
            f"historical submitted counts {list(his.submitted_count_S14)} (footnote a: 102 cores, 58)")
    R.check(all(r[7] == "not established" for r in pub_his), "historical 'Patients' = 'not established' for all five (R1.3-T7 "
                                                              "patient overlap=not established)")
    R.check(all(r[8].startswith("no") for r in pub_his), "historical 'Current-comparison inclusion' = 'no …' for all five")
    R.check(all("14 Brain/CNS" not in " ".join(map(str, r)) for r in pub_his),
            "R1.3-T7: no '14 Brain/CNS-region slides' wording in the NB ADRN vs MES row")
    R.check("retained_virchow2_output_rows" not in hist_hdr, "R3.3-T1: Virchow2 output-row column omitted from the "
                                                              "published table (kept in audit/)")

    # ---------------------------------------------------------------- footnotes (verbatim) and caption
    lines = EDIT_SHEET.read_text(encoding="utf-8").splitlines()
    l_fn = find_line(EDIT_SHEET, "**Footnotes (insert verbatim below the table):**", l_t)
    i = l_fn
    while not lines[i].startswith(">"):
        i += 1
    blk = []
    while i < len(lines) and lines[i].startswith(">"):
        blk.append(lines[i][1:].strip())
        i += 1
    paras = [p for p in " \n".join(blk).split(" \n \n")]
    fnotes, cur_p = [], []
    for x in blk:
        if x == "":
            if cur_p:
                fnotes.append(" ".join(cur_p))
            cur_p = []
        else:
            cur_p.append(x)
    if cur_p:
        fnotes.append(" ".join(cur_p))
    R.check([f[:2] for f in fnotes[:7]] == ["a.", "b.", "c.", "d.", "e.", "f.", "g."] and len(fnotes) == 8,
            f"footnotes a–g plus the flow-figure sentence read verbatim from l.{l_fn + 2}–{i} ({len(fnotes)} paragraphs)")
    l_cap = find_line(EDIT_SHEET, "**Replacement S14 caption:**", find_line(EDIT_SHEET, "### R1.3-T7"))
    j = l_cap
    while not lines[j].startswith(">"):
        j += 1
    capl = []
    while lines[j].startswith(">"):
        capl.append(lines[j][1:].strip())
        j += 1
    spec_cap = " ".join(capl).split("Table S14. ", 1)[1]
    app = "Footnotes give the reconciliation with the submitted table. Slide and patient flow is shown in Supplementary Figure S-R3.3."
    R.check(app in " ".join(EDIT_SHEET.read_text(encoding="utf-8").split()), "R3.3-T1 caption append located verbatim "
                                                                              "(whitespace-normalised)")
    spec_cap = spec_cap + " " + app

    # footnote numbers against counts.json
    c, k, s = cnt["comet"], cnt["kidrare"], cnt["ccdi"]
    fn_checks = {
        "e: 4,307 − 4,160 = 147": s["downloaded_MS"] - s["staged_with_features"] == 147,
        "e: of 4,154 — 2,952 / 231 / 138 / 68": [s["populations_4154"][x] for x in ("OOS", "NO_METADATA", "UNMAPPED", "MISC")] == [2952, 231, 138, 68],
        "e: 546 FFPE — 4 FOXR2, 18 unresolved (12 site, 6 compound)": (s["scope_546"]["EXCLUDE_KNOWN_FOXR2"], s["scope_546"]["UNRESOLVED"], s["unresolved_546_reasons"]["site_only"], s["unresolved_546_reasons"]["compound_support"]) == (4, 18, 12, 6),
        "c: 867 = 765 + 68 + 34 (21 participants)": (s["fig4c_867"]["n"], s["fig4c_867"]["decomposition_on_CM"]["INSCOPE"], s["fig4c_867"]["decomposition_on_CM"]["MISC"], s["fig4c_867"]["decomposition_on_CM"]["NO_METADATA"], s["fig4c_867"]["suffixed_reexport_parent_participants"]) == (867, 765, 68, 34, 21),
        "c: 20 entities": s["fig4c_867"]["entities"] == 20,
        "f: 2,766 matrix rows; 12 slides (11 NORMAL + RB 16738)": "2,766" in c["matrix_rows_2766"] and "11 NORMAL + RB 16738" in c["matrix_rows_2766"],
        "f: 1,898 within-patient pairs": c["within_patient_pairs"] == 1898,
        "f: xenografts 250/198 library, 245/196 in 2,778": (c["xenograft_library"], c["xenograft_2778"]) == ({"slides": 250, "patients": 198}, {"slides": 245, "patients": 196}),
        "g: 15 labels in the 2,778": len(c["per_group_2778"]) == 15,
    }
    for t, ok in fn_checks.items():
        R.check(ok, f"footnote {t} = counts.json")
    R.check(True, "footnote d '24 of the 26 are among the 34, and 2 have no retrieval prediction' and footnote a "
                  "'3 TMA cores without retained output' are carried from the spec (review v3 §4; counts.json has the 26/15 "
                  f"and 99/102 but not the 24/2 split: ccdi.reexports_no_metadata = {json.dumps(s.get('reexports_no_metadata'))[:120]})")

    # ---------------------------------------------------------------- Do-not-add (R3.3-T1) on the published cells
    pub_text_cells = [(r[0], h, v) for r in pub_cur for h, v in zip([h for _, h in CUR_COLS], r)] + \
                     [(r[0], h, str(v)) for r in pub_his for h, v in zip(hist_hdr, r)]
    hits948 = [(a, b) for a, b, v in pub_text_cells if "948" in v]
    R.check_open(not hits948, f"Do-not-add '948 anywhere except footnote b': found in published cells {hits948}")
    hits2766 = [(a, b, v[max(0, v.find('2,766') - 40): v.find('2,766') + 60]) for a, b, v in pub_text_cells if "2,766" in v]
    R.check_open(not hits2766, f"Do-not-add '2,766 as an evaluation or interval population': occurs in {hits2766}")
    hits_ev = [(a, b) for a, b, v in pub_text_cells if re.search(r"4,154[^;.]{0,30}evaluated", v)]
    R.check(not hits_ev, f"Do-not-add 'evaluated' for the 4,154: {hits_ev}")
    stj = [r for r in pub_his if r[6] == "St. Jude"] + [pub_cur[0], pub_cur[1]]
    hits_ext = [(r[0], h) for r in stj for h, v in zip(hist_hdr if len(r) == len(hist_hdr) else [x for _, x in CUR_COLS], r)
                if re.search(r"external|multiple institutions", str(v), re.I)]
    R.check(not hits_ext, f"Do-not-add 'external'/'multiple institutions' for St. Jude collections (COMET, STV and the four "
                          f"St. Jude historical rows): {hits_ext}")
    R.check(not [x for x in pub_text_cells if "269" in nums(x[2])], "Do-not-add '269': absent")
    R.check(not [x for x in pub_text_cells if "earlier mapping version" in x[2]], "Do-not-add 'earlier mapping version': absent")
    R.check(not [x for x in pub_text_cells if re.search(r"discrepant[^;]{0,40}(2,572|173|split)", x[2])],
            "Do-not-add the discrepant/non-discrepant split: absent")
    R.check(not [r for r in pub_his if re.search(r"\d+(\.\d+)?\s*%|accuracy|F1", " ".join(map(str, r)))],
            "Do-not-add any historical rate: no percentage/accuracy/F1 in the historical rows")
    fn_text = " ".join(fnotes)
    R.check(fn_text.count("948") == 1 and "948" in fnotes[1], "948 occurs once in the footnotes, in footnote b")

    # ---------------------------------------------------------------- material_and_labels (SI note, verbatim)
    si_lines = SI_MD.read_text(encoding="utf-8").splitlines()
    l_mat = next(n for n, x in enumerate(si_lines, 1) if x.startswith("Material and label sources"))
    para = si_lines[l_mat - 1]
    lead_sentence, body = para.split(". ", 1)
    segs = re.split(r"\s*(?=\b(?:COMET|STV|KidRare|CCDI-MCI): )", body)
    segs = [x.strip() for x in segs if x.strip()]
    mat_rows = []
    for sgt in segs:
        coll, rest = sgt.split(": ", 1)
        m = re.match(r"^(.*?whole-slide images)(?: \((.*?)\))?; (.*?)\.?$", rest)
        material, paren, label_part = m.group(1), m.group(2) or "", m.group(3)
        prep = paren if re.search(r"preparation|frozen", paren) else ""
        notes = [] if prep else ([paren] if paren else [])
        lm = re.match(r"^(?:labels (?:are|from) the )?(.*?)(?: \(([^()]*;[^()]*)\))?$", label_part)
        label, lparen = lm.group(1), lm.group(2)
        if lparen:
            notes.append(lparen)
        mat_rows.append([coll, material, prep, label, "; ".join(notes), sgt])
    ok_sub = all(all(x in para for x in r[1:4]) and all(n in para for n in r[4].split("; ") if n) for r in mat_rows)
    R.check(len(mat_rows) == 4 and [r[0] for r in mat_rows] == ["COMET", "STV", "KidRare", "CCDI-MCI"] and ok_sub,
            f"material_and_labels: SI l.{l_mat} '{lead_sentence}.' parsed into {len(mat_rows)} collection rows; every "
            f"material / preparation / label / note cell is a verbatim substring of the note (no added fact); the full "
            f"collection sentence is kept in column 'SI text (verbatim)'")
    mat_hdr = ["Collection", "Material", "Preparation evidence", "Label source", "Notes", "SI text (verbatim)"]

    # ---------------------------------------------------------------- per-group counts of the main populations (counts.json)
    pg = []
    for g in c["per_group_2778"]:
        pg.append(["COMET (2,778 all-tissue)", g["Tumor.group"], g["slides"], g["patients"]])
    for g, n in sorted(cnt["stv"]["main"]["labels"].items()):
        pg.append(["STV (696 main)", g, n, ""])
    for g, n in sorted(k["amended"]["groups"].items()):
        pg.append(["KidRare (1,581 eligible)", g, n, ""])
    for g, n in sorted(s["final"]["groups"].items()):
        pg.append(["CCDI-MCI (524 main)", g, n, ""])
    tot = {}
    for r in pg:
        tot[r[0]] = tot.get(r[0], 0) + r[2]
    pat_sum = sum(r[3] for r in pg if r[3] != "")
    R.check(tot == {"COMET (2,778 all-tissue)": 2778, "STV (696 main)": 696, "KidRare (1,581 eligible)": 1581,
                    "CCDI-MCI (524 main)": 524},
            f"per_group_main (copied from counts.json): slide totals {tot} equal the main populations")
    R.check(True, f"COMET per-group patients sum to {pat_sum:,} against 1,876 distinct patients: {pat_sum - 1876} "
                  f"patient-group memberships beyond one (patients with slides in more than one group), so per-group "
                  f"patient counts are not additive (state in the sheet note)")
    pg_hdr = ["Population", "Group", "Slides", "Patients"]

    # ---------------------------------------------------------------- Table 1 cross-check
    t1 = pd.read_csv(f_t1, dtype=str).fillna("")
    xmap = {"COMET": 0, "STV": 1, "KidRare": 2, "CCDI-MCI": 3}
    agree, disagree = [], []
    for _, r in t1.iterrows():
        i_ = xmap[r.Cohort]
        s14_all = " ".join(pub_cur[i_]) + " " + " ".join(mat_rows[i_][1:5])
        for col in t1.columns[1:]:
            miss = sorted(nums(r[col]) - nums(s14_all), key=float)
            (disagree if miss else agree).append((r.Cohort, col, miss))
    for coh, col, _ in agree:
        pass
    R.check(True, f"main Table 1 (`{rel(f_t1)}`, {len(t1)} rows × {len(t1.columns) - 1} fields): every number agrees "
                  f"with S14 in {len(agree)} of {len(agree) + len(disagree)} cells")
    word = []
    for _, r in t1.iterrows():
        i_ = xmap[r.Cohort]
        mr = mat_rows[i_]
        norm = lambda x: x.lower().replace("tumour", "tumor")
        mat_ok = norm(mr[1].split(" (")[0]) in norm(r.Material) and (not mr[2] or norm(mr[2]) in norm(r.Material))
        lab_ok = norm(mr[3].split(" (")[0].split(",")[0]) in norm(r["Label source"])
        word.append((r.Cohort, mat_ok, lab_ok))
    # independent field-level comparisons (wording/meaning)
    pairs = {
        "KidRare source/access": (t1.loc[2, "Source"] + "; role: " + t1.loc[2, "Role in this study"], "provider-controlled release (R1.3-T7); SI Methods 'provider's access-controlled release'"),
        "KidRare role": (t1.loc[2, "Role in this study"], pub_cur[2][1]),
        "CCDI-MCI role": (t1.loc[3, "Role in this study"], pub_cur[3][1]),
        "STV label source": (t1.loc[1, "Label source"], pub_cur[1][8]),
        "CCDI-MCI label source": (t1.loc[3, "Label source"], pub_cur[3][8]),
    }

    # manuscript compact Table 1 (live)
    try:
        lt, mh, mrows = md_table_after(MS, "| Cohort | Role in this study | Slides received (patients) |")
        ms_agree, ms_dis = 0, []
        for mr_ in mrows:
            i_ = xmap[mr_[0]]
            s14_all = " ".join(pub_cur[i_])
            for h, v in zip(mh[1:], mr_[1:]):
                miss = sorted(nums(v) - nums(s14_all), key=float)
                if miss:
                    ms_dis.append((mr_[0], h, miss))
                else:
                    ms_agree += 1
    except KeyError:
        lt, ms_agree, ms_dis = None, 0, [("(table not found)", "", [])]

    # ---------------------------------------------------------------- workbook
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    write_rows_sheet(wb, "current_populations", [h for _, h in CUR_COLS], pub_cur)
    write_rows_sheet(wb, "historical_collections", hist_hdr, pub_his)
    write_rows_sheet(wb, "material_and_labels", mat_hdr, mat_rows)
    write_rows_sheet(wb, "per_group_main", pg_hdr, pg)
    cap_rows, title_line, title, si_cap = caption_fields(num["current"], spec_cap, B.sha(SI_MD))
    write_caption_sheet(wb, [
        ("workspace_id", WID),
        ("final_number", f"Supplementary Table {num['current']} ({num['current_field']} of {rel(NUMBERING)}; earlier {num['history']})"),
        *cap_rows[:3],
        ("footnotes", "\n".join(fnotes)),
        ("sheets", "current_populations: R1.3-T7 current section populated from the r3_3 v3 CSV (provenance tags "
                   "stripped; tagged copy in audit/). historical_collections: R1.3-T7 historical section, count column "
                   "per R3.3-T1. material_and_labels: SI note 'Material and label sources' (moved from Table 1, pass 019), "
                   "verbatim. per_group_main: per-group slide (and COMET patient) counts of the main populations, copied "
                   "from counts.json — not in the R3.3-T1 spec; added because the SI caption lists per-group counts; the "
                   f"lead to keep or drop. COMET per-group patient counts are not additive (sum {pat_sum:,} vs 1,876 "
                   "distinct patients: some patients have slides in more than one group)."),
        ("sources", "; ".join(f"{rel(p)} ({B.sha(p)[:12]}…)" for p in (f_cur, f_his, f_cnt, f_t1)) +
                    f"; SI note l.{l_mat}"),
        ("spec", f"docs/manuscript_supplement_edits.md ### R1.3-T7 (l.{find_line(EDIT_SHEET, '### R1.3-T7')}; caption "
                 f"l.{l_cap}) + ### R3.3-T1 (l.{l_t}; footnotes l.{l_fn})"),
        cap_rows[3],
        ("built_by/date", f"scripts/supp_tables/build_S14_cohort_accounting_20261109.py, {BUILD_DATE}"),
    ])
    xlsx = OUT / f"Table_{WID}.xlsx"
    save_xlsx(wb, xlsx)
    back = openpyxl.load_workbook(xlsx)
    for n in back.sheetnames[1:]:
        write_sheet_csv(back[n], OUT / f"Table_{WID}__{n}.csv")

    aud = openpyxl.Workbook()
    aud.remove(aud.active)
    write_rows_sheet(aud, "audit_current_tagged", list(cur.columns), cur.values.tolist())
    write_rows_sheet(aud, "audit_historical_full", list(his.columns), his.values.tolist())
    write_rows_sheet(aud, "tag_strip_log", ["tag", "replacement"], [[a.strip(), b] for a, b in log])
    write_rows_sheet(aud, "historical_relabel_log", ["action"], [[x] for x in his_log])
    save_xlsx(aud, OUT / "audit" / f"Table_{WID}_audit.xlsx")
    for f in (f_cur, f_his, f_md, f_cnt):
        shutil.copyfile(f, OUT / "audit" / f.name)

    # ---------------------------------------------------------------- SI caption claims
    if si_cap:
        R.si_check(True, "'Per collection (COMET, STV, KidRare, CCDI-MCI)': four current rows")
        R.si_check(False, "'source, material and label source': material and label source are in material_and_labels / "
                          "current_populations; there is no separate 'source' column in the accepted CSV (Table 1 has "
                          "one; R1.3-T7 also asks for 'Source/access'). Not added (no bound source cell).")
        R.si_check(True, "'slides and patients downloaded, with features, with retrieval output, and in each evaluation "
                         "population': current_populations Slides delivered / evaluable / evaluated and Patients (CCDI "
                         "4,307 / 4,160 / 4,154 / 765 / 546 / 524; KidRare patients not available)")
        R.si_check(True, "'exclusion counts by reason in order of application': Exclusions column (COMET MISC, NORMAL; "
                         "STV 6 / 80; KidRare 482 / 259; CCDI of 4,154 then of 546)")
        R.si_check(False, "'per-group counts of slides and patients in the main populations and the sensitivity "
                          "populations (CCDI-MCI 536 and 542 participants; 735 …)': main-population per-group slide "
                          "counts are copied into per_group_main (patients per group only for COMET); sensitivity-population "
                          "per-group counts are not in counts.json (only in r1_2_scope_correction per-query files) and "
                          "were not tallied (no new computation). counts.json gives the 735 only as ccdi.scope_765.CLEAR.")
        R.si_check(True, "'the historical 867-slide CCDI-MCI population for traceability': footnote c and the CCDI "
                         "'Submitted → current' cell")
        R.si_check("MYCN" in si_cap or "historical" in si_cap.replace("historical 867", ""),
                   "The five historical exploratory collections (historical_collections sheet, required by R1.3-T7/"
                   "R3.3-T1) are not mentioned in the SI caption (they are described in the S16 notes)")
        R.si_check(True, f"'Material and label notes for Table 1 are given in the notes below': material_and_labels "
                         f"mirrors SI l.{l_mat}")

    # ---------------------------------------------------------------- open items
    R.open_items += [
        f"**'948' in a published cell (Do-not-add).** The CCDI-MCI 'Submitted → current' cell of the accepted CSV contains "
        f"'948 is not reconstructable from any retained artefact'; R3.3-T1 allows 948 only in footnote b. Printed unchanged "
        f"(no value edited); the lead to shorten that cell (e.g. to a pointer to footnotes b, c and g) or accept.",
        "**'2,766' in 'Slides evaluated' (COMET).** The cell lists '2,766 tumour-only distance-matrix rows …' after the "
        "evaluated counts; R3.3-T1 forbids 2,766 as an evaluation population and footnote f already carries it. Printed "
        "unchanged; the lead to move it to footnote f only.",
        "**R1.3-T7 extra columns not in the accepted CSV:** query-only / both-sided target and returned (partly inside "
        "'Slides evaluated'), reference eligibility, specimen preparation evidence, acquisition evidence and analysis role. "
        "No bound source carries acquisition evidence; nothing invented ('not established' would need the lead's decision).",
        "**Historical relabel / constants applied by spec:** " + "; ".join(his_log) + "; 'Endpoint/run provenance = not "
        "harmonized' is the R1.3-T7 constant (not a CSV field); 'Unit' derived from the CSV note 'cores (TMA), not WSIs' "
        "(author Q-O).",
        "**Workspace numbers inside cells/footnotes:** 'Table S15' (→ final S25), 'Supplementary Figure S-R3.3' (flow "
        "figure), 'retained in S8' etc. are workspace numbers; renumber at assembly (not edited here).",
        "**KidRare source wording conflict:** main Table 1 says 'KidRare public release (multi-institutional)' / 'public "
        "release'; R1.3-T7 says 'provider-controlled release'; the SI Methods say 'the provider's access-controlled "
        "release'. The Table 1 build record already lists this as 'open at reading'.",
        "**Spelling:** CSV cells use 'tumour'; the SI uses 'tumor'. Kept verbatim.",
        "**SI per-group / sensitivity counts:** see SI caption consistency; per_group_main is an addition for the lead to "
        "keep or drop.",
    ]

    t1_lines = "\n".join(f"| {a} | {b} | {'agree' if not m else 'numbers not in S14: ' + ', '.join(m)} |"
                         for a, b, m in agree + disagree)
    pair_lines = "\n".join(f"| {k_} | {a} | {b} |" for k_, (a, b) in pairs.items())
    word_lines = "\n".join(f"| {a} | {'agree' if m else 'DIFFER'} | {'agree' if l_ else 'DIFFER'} |" for a, m, l_ in word)
    ms_lines = (f"Manuscript Table 1 (`docs/prosper_revised/prosper_manuscript_revised.md` l.{lt}): {ms_agree} cells with "
                f"every number in S14; {len(ms_dis)} with numbers not in S14: {ms_dis}")
    B.write(OUT, {"workspace_id": WID, "final_number": num["current"], "built": BUILD_DATE})
    rep = f"""# Workspace Table {WID} → Supplementary Table {num['current']} — build report

Script: `scripts/supp_tables/build_S14_cohort_accounting_20261109.py`; run {now_utc()}; output `{rel(OUT)}`.
Assembly only: cells are copied from the accepted r3_3 v3 outputs (tags stripped as R3.3-T1 directs) and from the SI
note; nothing is recomputed. `audit/` holds the tagged CSV cells, the full historical CSV (with the Virchow2 column),
the tag-strip log and verbatim copies of the sources.

## Checks ({R.summary()})

{R.md_checks()}

## Cross-check against main Table 1

Every number in each Table 1 cell is looked up in the S14 published row (current_populations + material_and_labels).

| Cohort | Table 1 field | Result |
|---|---|---|
{t1_lines}

Material / label-source wording (Table 1 vs material_and_labels):

| Cohort | Material | Label source |
|---|---|---|
{word_lines}

Field-level wording comparisons:

| Field | Table 1 | S14 / spec |
|---|---|---|
{pair_lines}

{ms_lines}

## SI caption consistency (lead's caption vs this workbook; findings, not build failures)

{R.md_si(si_cap, title_line)}
## Open items

""" + "\n".join(f"- {o}" for o in R.open_items) + "\n"
    (OUT / "BUILD_REPORT.md").write_text(rep, encoding="utf-8")
    (OUT / "manifest_row.json").write_text(json.dumps({WID: {"final_number": num["current"], "path": rel(xlsx)}}, indent=1) + "\n")
    write_sha256sums(OUT)
    print(f"{WID}: {len(R.checks)} checks, {R.n_fail()} FAIL; Table 1 cells agree {len(agree)}/{len(agree) + len(disagree)} -> {OUT}")
    return 1 if R.n_fail() else 0


if __name__ == "__main__":
    sys.exit(main())
