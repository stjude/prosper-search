#!/usr/bin/env python3
"""Workspace Table S15 (final number from scripts/figures/supp_table_numbering.json): CCDI/MCI diagnosis-string
mapping to COMET tumor groups.

Assembly only (A_packaging, 2026-11-09). Spec: docs/manuscript_supplement_edits.md ### R3.3-T2 (published columns,
caption, Do-not-add) and the R3.3 audit F1 candidate clause ("8 strings in the mapping tables occur on no staged
slide"). Source: r3_3_cohort_accounting_20261008_v3/table_s15_regenerated.csv (573fea91…), copied value for value;
the full 20-column CSV is kept in audit/.

Run from the repository root:
    export PROSPER_OUT="$PWD/prosper_revision_output"
    /HOME/miniconda3/bin/python3 scripts/supp_tables/build_S15_ccdi_mapping_20261109.py
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
from packaging_common_20261109 import (BUILD_DATE, EDIT_SHEET, NUMBERING, OUT_ROOT, PKG, SI_MD, Bindings, Report,  # noqa: E402
                                       caption_fields, find_line, now_utc, numbering, rel, save_xlsx,
                                       write_caption_sheet, write_rows_sheet, write_sha256sums, write_sheet_csv)

WID = "S15"
OUT = PKG / WID
V3 = OUT_ROOT / "r3_3_cohort_accounting_20261008_v3"

# (source column, published header); R3.3-T2 fixes the column set and the 4,154 header wording
PUB = [("ccdi_diagnosis_as_exported", "Diagnosis string as exported"),
       ("v03_comet_diagnosis", "COMET diagnosis (mapping rule used in the analysis code)"),
       ("v03_comet_tumor_group", "COMET tumor group (mapping rule used in the analysis code)"),
       ("submitted_S15_target", "Target in the submitted version of this table"),
       ("status_base", "Eligibility status"),
       ("n_slides_staged_4160", "Slides with features (4,160)"),
       ("n_slides_evaluated_4154", "Slides with historical retrieval output (4,154)"),
       ("n_slides_in_vocabulary_765", "Slides in the COMET vocabulary (765)"),
       ("n_slides_ffpe_546", "Verified FFPE slides (546)"),
       ("n_slides_final_524", "Slides in the main population (524)")]
CAP_STATUS = {"in scope": (35, 759),
              "out of scope (no COMET reference class)": (410, 2953),
              "unmapped (no mapping rule; slides excluded as unmapped)": (67, 105),
              "excluded (COMET MISC: not a diagnostic group)": (18, 68),
              "out of scope (CNS entity; no COMET reference class)": (4, 33),
              "out of scope (CNS; corrected from NB — R1.2 scope decision)": (3, 6)}


def main() -> int:
    R = Report()
    B = Bindings()
    num = numbering(WID)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit").mkdir(exist_ok=True)

    f_csv = B.bind(V3 / "table_s15_regenerated.csv", "S15 body")
    f_cnt = B.bind(V3 / "counts.json", "s15_summary, staged_by_status_base, identities")
    f_sha = B.bind(V3 / "sha256.txt", "r3_3 v3 own hash list")
    B.bind(EDIT_SHEET, "spec")
    B.bind(SI_MD, "SI title/caption (live file; hash at build)")
    B.bind(NUMBERING, "numbering")
    own = dict(reversed(l.split()) for l in f_sha.read_text().splitlines() if l.strip())
    for f in (f_csv, f_cnt):
        R.check(own.get(f.name) == B.sha(f), f"`{f.name}` sha256 {B.sha(f)[:12]}… equals r3_3 v3 sha256.txt")
    R.check(B.sha(f_csv).startswith("573fea91"), "S15 CSV is the spec-cited version (573fea91…)")

    d = pd.read_csv(f_csv, keep_default_na=False)
    cnt = json.load(open(f_cnt))
    s15 = cnt["s15_summary"]

    # ---------------------------------------------------------------- headline verification
    n_str, n_comp = len(d), int(d.compound_string.sum())
    R.check(n_str == 537 and d.ccdi_diagnosis_as_exported.is_unique and n_str == s15["n_strings"],
            f"537 strings: {n_str} rows, diagnosis strings unique ({d.ccdi_diagnosis_as_exported.is_unique}); counts.json "
            f"n_strings {s15['n_strings']}")
    has_sep = d.ccdi_diagnosis_as_exported.str.contains(r"[;#]")
    semi = int(d.ccdi_diagnosis_as_exported.str.contains(";").sum())
    hash_only = d[has_sep & ~d.ccdi_diagnosis_as_exported.str.contains(";")].ccdi_diagnosis_as_exported.tolist()
    R.check(n_comp == 338 and n_comp == s15["n_compound"] and bool((has_sep == d.compound_string).all()),
            f"338 compound strings: compound_string True {n_comp}; strings containing ';' or '#' {int(has_sep.sum())} "
            f"(';' {semi}; '#' only: {hash_only}), identical row set; counts.json n_compound {s15['n_compound']}")
    zero = d[d.n_slides_staged_4160 == 0]
    stage_cols = [c for c in d.columns if c.startswith("n_slides_")]
    R.check(len(zero) == 8 and bool((zero[stage_cols] == 0).all().all())
            and set(zero.manifest_population) == {"(no staged slide)"},
            f"8 strings on no staged slide: {len(zero)} rows with n_slides_staged_4160 = 0, all zero at every stage, "
            f"manifest_population '(no staged slide)'; in submitted S15 {int(zero.in_submitted_S15.sum())}, in v03 code "
            f"input {int(zero.in_v03_code_input.sum())} ({sorted(zero.ccdi_diagnosis_as_exported)[:3]} …)")
    R.check(len(d) - len(zero) == 529, f"{len(d) - len(zero)} distinct strings occur on staged slides (l.9220 basis: 529)")
    by = d.groupby("status_base").agg(strings=("ccdi_diagnosis_as_exported", "size"),
                                      staged=("n_slides_staged_4160", "sum"))
    for st, (ns, nsl) in CAP_STATUS.items():
        got = by.loc[st] if st in by.index else None
        R.check(got is not None and int(got.strings) == ns and int(got.staged) == nsl
                and s15["status_base_counts"][st] == ns and s15["staged_by_status_base"][st] == nsl,
                f"caption status '{st}': {ns} strings / {nsl:,} slides; table {int(got.strings) if got is not None else '—'} "
                f"/ {int(got.staged) if got is not None else '—'}; counts.json {s15['status_base_counts'].get(st)} / "
                f"{s15['staged_by_status_base'].get(st)}")
    R.check(set(by.index) == set(CAP_STATUS), f"status_base takes exactly six values ({len(by)})")
    R.check(int(d.n_slides_staged_4160.sum()) == 3924 == cnt["ccdi"]["linked_to_participant"],
            f"slide counts are of the 3,924 linked slides: sum n_slides_staged_4160 = {int(d.n_slides_staged_4160.sum()):,}; "
            f"counts.json ccdi.linked_to_participant {cnt['ccdi']['linked_to_participant']}")
    tot = {c: int(d[c].sum()) for c in stage_cols}
    R.check(tot["n_slides_in_vocabulary_765"] == 765 and tot["n_slides_ffpe_546"] == 546 and tot["n_slides_final_524"] == 524,
            f"stage column totals: {tot}")
    ev = d.groupby("status_base").n_slides_evaluated_4154.sum()
    ident1 = int(ev["unmapped (no mapping rule; slides excluded as unmapped)"] + ev["out of scope (CNS entity; no COMET reference class)"])
    voc = d.groupby("status_base").n_slides_in_vocabulary_765.sum()
    ident2 = int(voc["in scope"] + voc["out of scope (CNS; corrected from NB — R1.2 scope decision)"])
    R.check(ident1 == 138 and ident2 == 765,
            f"identities: unmapped {int(ev['unmapped (no mapping rule; slides excluded as unmapped)'])} + CNS entity "
            f"(ATRT) {int(ev['out of scope (CNS entity; no COMET reference class)'])} = {ident1} (138, of the 4,154); in "
            f"scope {int(voc['in scope'])} + corrected-from-NB {int(voc['out of scope (CNS; corrected from NB — R1.2 scope decision)'])} "
            f"= {ident2} (765)")
    unres = d[d.n_slides_scope_unresolved_546 > 0]
    site = int(unres[unres.revised_status.str.contains("site_only")].n_slides_scope_unresolved_546.sum())
    comp = int(unres[unres.revised_status.str.contains("compound_support")].n_slides_scope_unresolved_546.sum())
    R.check(len(unres) == 11 and int(unres.n_slides_scope_unresolved_546.sum()) == 18 and (site, comp) == (12, 6)
            and set(unres.status_base) == {"in scope"},
            f"caption 'Eleven in-scope strings carry 18 FFPE slides with unresolved scope (12 on anatomic site, 6 on a "
            f"compound diagnosis)': {len(unres)} strings, {int(unres.n_slides_scope_unresolved_546.sum())} slides "
            f"(site_only {site}, compound_support {comp}; from the audit-only columns)")
    normal = d[d.submitted_S15_target == "Normal"]
    R.check(len(normal) > 0 and set(normal.status_base) <= {"out of scope (no COMET reference class)",
                                                             "out of scope (CNS entity; no COMET reference class)"},
            f"caption 'Out of scope … replaces the submitted \"Normal\" target': {len(normal)} rows with submitted target "
            f"'Normal', status {sorted(set(normal.status_base))}")
    R.check(int((d.n_slides_final_524 > d.n_slides_ffpe_546).sum() + (d.n_slides_ffpe_546 > d.n_slides_in_vocabulary_765).sum()
                + (d.n_slides_in_vocabulary_765 > d.n_slides_evaluated_4154).sum() + (d.n_slides_evaluated_4154 > d.n_slides_staged_4160).sum()) == 0,
            "stage counts are monotone in every row (524 ≤ 546 ≤ 765 ≤ 4,154 ≤ 4,160 columns)")

    # ---------------------------------------------------------------- caption (spec) and Do-not-add
    l_t2 = find_line(EDIT_SHEET, "### R3.3-T2 — Supplementary Table S15: regenerated mapping table")
    lines = EDIT_SHEET.read_text(encoding="utf-8").splitlines()
    i = find_line(EDIT_SHEET, "**Replace the caption with:**", l_t2)
    while not lines[i].startswith(">"):
        i += 1
    cap = []
    while lines[i].startswith(">"):
        cap.append(lines[i][1:].strip())
        i += 1
    spec_cap = " ".join(cap).split("Supplementary Table S15. ", 1)[1]
    l_f1 = find_line(EDIT_SHEET, "- **R3.3-T2 caption and letter R3.3 (candidate clause, F1):**")
    clause = "(8 strings in the mapping tables occur on no staged slide)"
    spec_cap_f1 = spec_cap.replace("338 combine several diagnoses)", "338 combine several diagnoses; " + clause[1:])
    R.check(spec_cap_f1 != spec_cap, f"F1 candidate clause (l.{l_f1}) inserted after '338 combine several diagnoses' in the "
                                     f"spec draft caption (candidate; not applied to the SI)")
    pub_hdr = [h for _, h in PUB]
    pub_text = " ".join(pub_hdr) + " " + spec_cap_f1
    R.check("diagnostic entit" not in pub_text, "Do-not-add 'diagnostic entities' for the rows: absent from published "
                                                 "headers and the spec caption")
    R.check("485" not in pub_text, "Do-not-add '485 unique diagnostic entities': absent")
    R.check(set(d.status_base) == set(CAP_STATUS) and not any("FFPE slide(s) unresolved" in s for s in d.status_base),
            "Do-not-add the suffixed status strings as separate categories: published 'Eligibility status' is status_base "
            "(six values); revised_status stays in audit/")
    R.check("display_string" not in [c for c, _ in PUB] and "review_flags" not in [c for c, _ in PUB],
            "audit-only columns (revised_target, revised_status, display_string, manifest_population, membership flags, "
            "review_flags, unresolved/FOXR2 counts, compound_string) not published")

    # ---------------------------------------------------------------- workbook
    rows = d[[c for c, _ in PUB]].astype(object).values.tolist()
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    write_rows_sheet(wb, "mapping", pub_hdr, rows)
    cap_rows, title_line, title, si_cap = caption_fields(num["current"], spec_cap_f1, B.sha(SI_MD))
    write_caption_sheet(wb, [
        ("workspace_id", WID),
        ("final_number", f"Supplementary Table {num['current']} ({num['current_field']} of {rel(NUMBERING)}; earlier "
                         f"{num['history']})"),
        *cap_rows[:3],
        ("footnotes", ""),
        ("sheets", "mapping: one row per diagnosis string as exported (537), R3.3-T2 published columns; headers spelled "
                   "out (column ids: " + ", ".join(c for c, _ in PUB) + "). The full 20-column CSV is in audit/."),
        ("sources", f"{rel(f_csv)} ({B.sha(f_csv)[:12]}…); {rel(f_cnt)} ({B.sha(f_cnt)[:12]}…)"),
        ("spec", f"docs/manuscript_supplement_edits.md ### R3.3-T2 (l.{l_t2}); F1 clause l.{l_f1}"),
        cap_rows[3],
        ("built_by/date", f"scripts/supp_tables/build_S15_ccdi_mapping_20261109.py, {BUILD_DATE}"),
    ])
    xlsx = OUT / f"Table_{WID}.xlsx"
    save_xlsx(wb, xlsx)
    back = openpyxl.load_workbook(xlsx)
    write_sheet_csv(back["mapping"], OUT / f"Table_{WID}__mapping.csv")
    chk = pd.read_csv(OUT / f"Table_{WID}__mapping.csv", keep_default_na=False)
    src = d[[c for c, _ in PUB]].copy()
    src.columns = pub_hdr
    R.check(chk.shape == src.shape and bool((chk.astype(str).values == src.astype(str).values).all()),
            f"published sheet equals the source CSV columns value for value ({chk.shape[0]} × {chk.shape[1]})")
    shutil.copyfile(f_csv, OUT / "audit" / f_csv.name)
    R.check((OUT / "audit" / f_csv.name).read_bytes() == f_csv.read_bytes(), "audit/: verbatim copy of the 20-column CSV")

    # ---------------------------------------------------------------- SI caption claims
    if si_cap:
        R.si_check("diagnostic entit" not in title,
                   f"SI title '{title}' uses 'diagnostic entities'; R3.3-T2 Do-not-add: '\"diagnostic entities\" for the "
                   f"rows' (rows are diagnosis strings as exported; spec title 'Mapping of CCDI and MCI diagnosis strings "
                   f"to COMET tumour groups')")
        R.si_check(True, "'Each supplied diagnosis string of the CCDI-MCI download is listed': 537 strings (529 on staged "
                         "slides + 8 from the mapping tables only)")
        R.si_check("participant counts" not in si_cap,
                   "'with its slide and participant counts': the table has slide counts per stage only; no participant "
                   "counts per string exist in the source CSV (slides are of 3,924 participant-linked slides)")
        R.si_check(True, "'its COMET tumor-group assignment or its exclusion category (outside the 14-group vocabulary, "
                         "unmapped, MISC …)': Eligibility status (status_base)")
        R.si_check("unresolved scope" not in si_cap.split("exclusion category", 1)[-1].split(")")[0],
                   "'exclusion category (… unresolved scope …)': unresolved scope is not an Eligibility status value; it "
                   "is slide-level (18 FFPE slides on 11 in-scope strings) and appears only as 546 − 524 in the published "
                   "columns (the per-string count is audit-only per R3.3-T2)")
        foxr2 = d[d.status_base.str.contains("corrected from NB")]
        R.si_check(bool(foxr2.ccdi_diagnosis_as_exported.str.startswith("CNS neuroblastoma, FOXR2-activated;").all()),
                   f"'… FOXR2-activated CNS neuroblastoma)': the {len(foxr2)} strings with status 'out of scope (CNS; "
                   f"corrected from NB — R1.2 scope decision)' ({int(foxr2.n_slides_staged_4160.sum())} slides; "
                   f"{int(foxr2.n_slides_foxr2_excluded_546.sum())} of them among the 546 FFPE) all begin 'CNS "
                   f"neuroblastoma, FOXR2-activated;'; the status wording itself does not say FOXR2")
        R.si_check("sensitivity populations" not in si_cap,
                   "'the scope decision applied in the main and sensitivity populations': no sensitivity-population "
                   "membership column in the source (only the 524 main-population count); not added")
        R.si_check("provisional assignments" not in si_cap,
                   "'provisional assignments used only in the sensitivity populations are marked': no such marker in the "
                   "source CSV; not added")
        R.si_check("fixed before the corrected evaluation" not in si_cap,
                   "'The mapping was fixed before the corrected evaluation': not verifiable from the bound files; the "
                   "SI Methods also say the CCDI-MCI restrictions were introduced after earlier results were examined — "
                   "the lead to confirm the provenance statement")

    R.open_items += [
        "**Workspace reference inside a published value:** status 'out of scope (CNS; corrected from NB — R1.2 scope "
        "decision)' carries the workspace id 'R1.2'. Copied unchanged (value); the lead may want 'out of scope (CNS; "
        "corrected from NB)' as in the R3.3-T2 caption.",
        "**F1 clause** '(8 strings in the mapping tables occur on no staged slide)' is a candidate (l.9220); inserted in "
        "the spec-draft caption only.",
        "**Header wording:** R3.3-T2 fixes the column set and the 4,154 header ('with historical retrieval output'); the "
        "other headers are spelled out here from the spec caption's wording; column ids are listed in the caption sheet.",
        "**SI caption vs table:** participant counts, sensitivity-population scope decisions and 'provisional' markers "
        "are promised by the SI caption but absent from the source (see SI caption consistency).",
    ]

    B.write(OUT, {"workspace_id": WID, "final_number": num["current"], "built": BUILD_DATE,
                  "verified": {"strings": n_str, "compound": n_comp, "no_staged_slide": len(zero), "stage_totals": tot}})
    rep = f"""# Workspace Table {WID} → Supplementary Table {num['current']} — build report

Script: `scripts/supp_tables/build_S15_ccdi_mapping_20261109.py`; run {now_utc()}; output `{rel(OUT)}`.
Assembly only: the published sheet is the R3.3-T2 column subset of the accepted CSV, value for value.

## Checks ({R.summary()})

{R.md_checks()}

## SI caption consistency (lead's caption vs this workbook; findings, not build failures)

{R.md_si(si_cap, title_line)}
## Open items

""" + "\n".join(f"- {o}" for o in R.open_items) + "\n"
    (OUT / "BUILD_REPORT.md").write_text(rep, encoding="utf-8")
    (OUT / "manifest_row.json").write_text(json.dumps({WID: {"final_number": num["current"], "path": rel(xlsx)}}, indent=1) + "\n")
    write_sha256sums(OUT)
    print(f"{WID}: {len(R.checks)} checks, {R.n_fail()} FAIL -> {OUT}")
    return 1 if R.n_fail() else 0


if __name__ == "__main__":
    sys.exit(main())
