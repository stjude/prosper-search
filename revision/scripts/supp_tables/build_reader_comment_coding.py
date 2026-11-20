#!/usr/bin/env python3
"""Coded reader comments (plan 002 section 5.10) for one reader-study analysis directory.

Joins the independent code sheet (one row per non-empty comment; the nine pre-specified codes, no code added) to the
verbatim comments of the analysis table `tables/T9_comments_for_coding.csv`. Reads analysis-layer files only; the
reader-study export, database and backups are never opened. Comments are copied unedited.

Checks (exit 1 on failure): the code sheet covers exactly the non-empty comments; every code is one of the nine;
codes that refer to the search results or the summary are not applied to Step 1 comments, and the two summary codes
only to Step 3 comments.

Output (default `<analysis>_comment_coding/`): `comments_coded.csv` (one row per comment with its codes, the
usefulness rating given at that step and, for material or interface comments, the kind of issue; the coder's notes
stay in the code sheet), `comment_code_counts.csv` (comments per code, reader and step), `comment_issue_cases.csv`
(cases with a slide-or-tissue or displayed-site comment, the readers who wrote one, and the quality-check verdict
from `--qc74`), `comment_rating_context.csv` (median usefulness rating on readings with and without a comment at
that step), `RUN_RECORD.md` (input and output sha256).

Usage:
  python3 scripts/supp_tables/build_reader_comment_coding.py \
      --analysis "$PROSPER_OUT/reader_study_suite/analysis/20261116_prod_export_20261116-085036_five_readers" \
      --qc74 "$PROSPER_OUT/reader_study_suite/analysis/20261116_prod_export_20261116-085036_five_readers_qc74" \
      --codes [internal record]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

CODES = ["confirms", "rules_out", "retrieval_inconsistent", "names_off_vocabulary_entity", "needs_ancillary_test",
         "wants_more_clinical_info", "ui_or_material_issue", "summary_misleading", "summary_reframed"]
NOT_AT_STEP_A = {"confirms", "rules_out", "retrieval_inconsistent", "summary_misleading", "summary_reframed"}
ONLY_AT_STEP_C = {"summary_misleading", "summary_reframed"}
KEY = ["reader_id", "case_id", "phase"]
ISSUE_TYPES = {"slide_or_tissue", "displayed_site", "interface_or_entry"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(t9: pd.DataFrame, codes: pd.DataFrame, readers: list[str],
          readings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    errors: list[str] = []
    ne = t9[t9.comment.str.strip() != ""].copy()
    codes = codes.rename(columns={"step": "phase"})
    if codes.duplicated(KEY).any():
        errors.append("duplicate keys in the code sheet")
    have, want = set(map(tuple, codes[KEY].values)), set(map(tuple, ne[KEY].values))
    if have != want:
        errors.append(f"code sheet does not cover the comments: uncoded {sorted(want - have)}, extra {sorted(have - want)}")
    m = ne[KEY + ["expertise_tier", "truth_group", "eff_favoured", "outcome_class_transition", "outcome_class",
                  "comment"]].merge(codes[KEY + ["codes", "issue_type"]], on=KEY, how="left")
    use = readings.set_index(["reader_id", "case_id"])
    m["usefulness_rating_at_step"] = [
        "" if ph == "A" else int(use.loc[(r, c), f"use_{ph}"]) for r, c, ph in zip(m.reader_id, m.case_id, m.phase)]
    applied = m.codes.fillna("").map(lambda s: [c.strip() for c in s.split(";") if c.strip()])
    for (_, row), cs in zip(m.iterrows(), applied):
        where = f"{row.reader_id} {row.case_id} {row.phase}"
        for c in cs:
            if c not in CODES:
                errors.append(f"{where}: unknown code {c}")
            if row.phase == "A" and c in NOT_AT_STEP_A:
                errors.append(f"{where}: {c} applied to a Step 1 comment")
            if row.phase != "C" and c in ONLY_AT_STEP_C:
                errors.append(f"{where}: {c} applied outside Step 3")
    for c in CODES:
        m[c] = applied.map(lambda cs, c=c: c in cs)
    m["n_codes"] = applied.map(len)
    m["issue_type"] = m.issue_type.fillna("")
    for (_, row), cs in zip(m.iterrows(), applied):
        if ("ui_or_material_issue" in cs) != (row.issue_type in ISSUE_TYPES) or (row.issue_type and row.issue_type not in ISSUE_TYPES):
            errors.append(f"{row.reader_id} {row.case_id} {row.phase}: issue_type {row.issue_type!r} inconsistent with codes")
    order = {r: i for i, r in enumerate(readers)}
    m = m.assign(_o=m.reader_id.map(order)).sort_values(["_o", "case_id", "phase"]).drop(columns=["_o", "codes"])
    m = m.rename(columns={"phase": "step", "truth_group": "truth", "eff_favoured": "favoured_at_step"})
    rows = []
    for r in readers:
        g = m[m.reader_id == r]
        n_read = int((t9.reader_id == r).sum())
        for label, sub in [("comments (non-empty)", None), ("comments without a code", "none")] + [(c, c) for c in CODES]:
            sel = g if sub is None else g[g.n_codes == 0] if sub == "none" else g[g[sub]]
            rows.append({"reader_id": r, "measure": label, "step_A": int((sel.step == "A").sum()),
                         "step_B": int((sel.step == "B").sum()), "step_C": int((sel.step == "C").sum()),
                         "all_steps": len(sel), "step_submissions": n_read})
    return m, pd.DataFrame(rows), errors


def issue_cases(coded: pd.DataFrame, readers: list[str], qc_fail: set[str]) -> pd.DataFrame:
    label = {r: f"Reader {i + 1}" for i, r in enumerate(readers)}
    rows = []
    for (kind, case), g in coded[coded.issue_type.isin(["slide_or_tissue", "displayed_site"])].groupby(
            ["issue_type", "case_id"]):
        who = [r for r in readers if r in set(g.reader_id)]
        rows.append({"issue_type": kind, "case_id": case, "n_comments": len(g), "n_readers": len(who),
                     "readers": "; ".join(label[r] for r in who),
                     "quality_check": "fail" if case in qc_fail else "pass"})
    return pd.DataFrame(rows).sort_values(["issue_type", "case_id"], ascending=[False, True])


def rating_context(t9: pd.DataFrame, readings: pd.DataFrame, readers: list[str]) -> pd.DataFrame:
    has = t9.assign(has=t9.comment.str.strip() != "").set_index(KEY).has
    rows = []
    for r in readers:
        g = readings[readings.reader_id == r]
        for ph in ("B", "C"):
            flag = g.case_id.map(lambda c: bool(has.loc[(r, c, ph)]))
            use = pd.to_numeric(g[f"use_{ph}"])
            for lab, sel in (("with a comment at this step", flag), ("without a comment at this step", ~flag)):
                rows.append({"reader_id": r, "step": ph, "readings": lab, "n": int(sel.sum()),
                             "usefulness_median": float(use[sel].median()) if sel.any() else ""})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True, type=Path)
    ap.add_argument("--codes", required=True, type=Path)
    ap.add_argument("--qc74", required=True, type=Path, help="companion quality-check sensitivity directory")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.analysis.with_name(a.analysis.name + "_comment_coding")
    t9_path = a.analysis / "tables" / "T9_comments_for_coding.csv"
    run = json.load(open(a.analysis / "analyze_run.json"))
    t9 = pd.read_csv(t9_path, keep_default_na=False)
    readings = pd.read_csv(a.analysis / "readings.csv", keep_default_na=False)
    coded, counts, errors = build(t9, pd.read_csv(a.codes, keep_default_na=False), run["readers"], readings)
    qc_text = (a.qc74 / "RUN_RECORD.generated.md").read_text(encoding="utf-8")
    fm = re.search(r"failed cases \[([^\]]*)\]", qc_text)
    if not fm:
        errors.append("quality-check failed cases not found in the qc74 run record")
    qc_fail = set(re.findall(r"case\d+", fm.group(1))) if fm else set()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    coded.to_csv(out / "comments_coded.csv", index=False)
    counts.to_csv(out / "comment_code_counts.csv", index=False)
    issue_cases(coded, run["readers"], qc_fail).to_csv(out / "comment_issue_cases.csv", index=False)
    rating_context(t9, readings, run["readers"]).to_csv(out / "comment_rating_context.csv", index=False)
    (out / "RUN_RECORD.md").write_text(
        "# Coded reader comments\n\n"
        f"- run: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}\n"
        f"- analysis: `{a.analysis}` (readings.csv {run['readings_sha256']})\n"
        f"- T9_comments_for_coding.csv sha256 {sha(t9_path)}\n"
        f"- code sheet `{a.codes}` sha256 {sha(a.codes)}\n"
        f"- comments: {len(coded)} of {int(counts.groupby('reader_id').step_submissions.first().sum())} step submissions; "
        f"without a code: {int((coded.n_codes == 0).sum())}\n"
        "- coding: nine pre-specified codes (plan 002 section 5.10), applied by one author; not checked by a pathologist\n"
        f"- quality-check failures (from the qc74 run record): {sorted(qc_fail)}\n"
        "- output sha256: " + "; ".join(f"{n} {sha(out / n)}" for n in (
            "comments_coded.csv", "comment_code_counts.csv", "comment_issue_cases.csv",
            "comment_rating_context.csv")) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(coded)} comments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
