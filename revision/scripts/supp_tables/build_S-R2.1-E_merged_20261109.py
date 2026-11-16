#!/usr/bin/env python3
"""Table S-R2.1-E (final S3): merge sheets 1-4 (2026-10-22 C build) and sheets 5-6 (2026-10-21 analyst build) into one
workbook. Values are copied cell for cell and asserted equal to both source workbooks and to the source per-sheet CSVs;
the S-R2.1-C mMV@5 sheets of the 10-21 workbook belong to another table and are left out.

Footnote 3 open item: the spec footnote (R2.1-T12) was written on RR-5 run_001 (published Virchow2 bank). This script
re-derives the footnote's counts from the retained run_001 and run_003 table_s3_revised.csv by its own selection code
(not the 10-22 builder's), reports both, and proposes (does not apply) a run_003 wording.

Output: $PROSPER_OUT/supp_tables_20261109/A_packaging/S-R2.1-E/. Run from the repository root with PROSPER_OUT exported.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from packaging_common_20261109 import (BUILD_DATE, EDIT_SHEET, NUMBERING, OUT_ROOT, PKG, SI_MD, Bindings, Report, edit_sheet_lines,
                                       caption_fields, find_line, now_utc, numbering, rel, write_caption_sheet,
                                       save_xlsx, write_sha256sums, write_sheet_csv)

WID = "S-R2.1-E"
OUT = PKG / WID
SRC14 = OUT_ROOT / "supp_tables_20261022/C_encoders_comparators"
SRC56 = OUT_ROOT / "supp_tables_20261021"
X14 = SRC14 / "Table_S-R2.1-E_sheets1-4.xlsx"
X56 = SRC56 / "Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx"
RUN001 = OUT_ROOT / "r2_1_main_grid_plurality_20261012/run_001"
RUN003 = OUT_ROOT / "r2_1_virchow2_fp32_main_20261013/run_003/rr5_replacement_001"
# source sheet -> (merged sheet name, source CSV holding the same values)
SHEETS_14 = {s: (s, SRC14 / f"Table_S-R2.1-E_sheets1-4__{s}.csv") for s in
             ["Sheet1_main", "Sheet2_per_group_k5", "Sheet3_TITAN_repair", "Sheet4_winners",
              "Sheet4_tie_sensitivity", "Sheet4_changed_slots"]}
SHEETS_56 = {"S-R2.1-E s5 retrieval": ("Sheet5_retrieval_metrics", SRC56 / "S-R2.1-E_sheet5_retrieval_metrics.csv"),
             "S-R2.1-E s6 per-group k5": ("Sheet6_per_group_k5", SRC56 / "S-R2.1-E_sheet6_per_group_k5.csv")}
EXCLUDED_56 = ["Notes", "S-R2.1-C mMV5 deltas", "S-R2.1-C mMV5 identity"]


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b and type(a) is type(b)


def sheet_values(ws):
    return [tuple(r) for r in ws.iter_rows(values_only=True)]


def footnote3_counts(table: Path) -> dict:
    """Footnote-3 quantities from table_s3_revised.csv: generous (oracle) macro F1, canonical vs reversed order,
    against the submitted Table S3 value; reversed-order plurality winners from the same table (argmax over the four
    aggregations per encoder x k)."""
    t = pd.read_csv(table)
    g = t[(t.rule == "oracle") & (t.metric == "macro_f1")]
    can = g[g.tie_order == "canonical"].set_index(["cell_id", "k"])
    rev = g[g.tie_order == "reversed"].set_index(["cell_id", "k"])["value"]
    s = can[can.submitted_value.notna()].copy()
    s["rev"] = rev.loc[s.index].to_numpy()
    s["d"] = (s.value - s.submitted_value).abs()
    tie = s.n_order_sensitive_k > 0
    exact = s.d < 5e-7
    lo, hi = np.minimum(s.value, s.rev), np.maximum(s.value, s.rev)
    between = tie & (s.submitted_value >= lo - 5e-7) & (s.submitted_value <= hi + 5e-7)
    cid = s.index.get_level_values(0)
    v2 = cid.str.startswith("virchow2")
    out = dict(
        n_rows=len(s), n_cells=cid.nunique(), no_tie=int((~tie).sum()), no_tie_exact=int((~tie & exact).sum()),
        tie=int(tie.sum()), tie_yottixel=int((tie & cid.str.endswith("/yottixel")).sum()),
        tie_titan_submitted=int((tie & (cid == "titan/titan_published")).sum()),
        max_tie_diff=float(s.d[tie].max()),
        max_tie_diff_row="{} k = {}".format(*map(str, s.d[tie].idxmax())), between=int(between.sum()),
        between_excl_virchow2=int((between & ~v2).sum()), between_virchow2=int((between & v2).sum()),
        no_tie_not_exact=[dict(cell=i[0], k=int(i[1]), value=float(r.value), submitted=float(r.submitted_value))
                          for i, r in s[~tie & ~exact].iterrows()],
        virchow2_no_tie=int((~tie & v2).sum()), virchow2_no_tie_exact=int((~tie & exact & v2).sum()),
        submitted_by_row={f"{i[0].replace('virchow2_fp32', 'virchow2')}|{int(i[1])}": float(v)
                          for i, v in s.submitted_value.items()},
    )
    pl = t[(t.rule == "plurality") & (t.metric == "macro_f1") & ~t.cell_id.str.startswith("titan/")]
    for order in ("canonical", "reversed"):
        x = pl[pl.tie_order == order]
        mx = x.groupby(["display_encoder", "k"]).value.transform("max")
        w = x[x.value == mx]
        assert not w.duplicated(["display_encoder", "k"]).any(), "exact winner tie: counts would need weights"
        c32 = w[~w.display_encoder.str.contains("Optimus")]
        out[f"winners_{order}_C32"] = c32.aggregation.value_counts().to_dict()
        out[f"winners_{order}_H36"] = w.aggregation.value_counts().to_dict()
        out[f"n_slots_{order}"] = (len(c32), len(w))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    B, R = Bindings(), Report()
    for p, role in [(X14, "source workbook, sheets 1-4"), (X56, "source workbook, sheets 5-6 (+ S-R2.1-C, not copied)")]:
        B.bind(p, role)
    for _, (_, c) in {**SHEETS_14, **SHEETS_56}.items():
        B.bind(c, "source per-sheet CSV (equality check)")
    for p in [SRC14 / "BUILD_REPORT.md", SRC14 / "input_bindings.json", SRC56 / "input_bindings.json", SRC56 / "README.md"]:
        B.bind(p, "source build record")
    t001, t003 = B.bind(RUN001 / "table_s3_revised.csv", "footnote-3 check (run_001)"), \
        B.bind(RUN003 / "table_s3_revised.csv", "footnote-3 check (run_003, the table's source)")
    win003 = B.bind(RUN003 / "winners.csv", "footnote-3 winner counts (cross-check)")
    B.bind(EDIT_SHEET, "spec"); B.bind(SI_MD, "SI title"); B.bind(NUMBERING, "numbering")

    num = numbering(WID)
    # ---------------- copy sheets ----------------
    wb14 = openpyxl.load_workbook(X14)
    wb56 = openpyxl.load_workbook(X56)
    assert wb14.sheetnames == ["caption"] + list(SHEETS_14), wb14.sheetnames
    assert set(wb56.sheetnames) == set(SHEETS_56) | set(EXCLUDED_56), wb56.sheetnames
    new = openpyxl.Workbook()
    new.remove(new.active)
    order = [(wb14, s, n) for s, (n, _) in SHEETS_14.items()] + [(wb56, s, n) for s, (n, _) in SHEETS_56.items()]
    for wb, s, n in order:
        src, dst = wb[s], new.create_sheet(n)
        for row in src.iter_rows(values_only=True):
            dst.append(list(row))

    # ---------------- caption sheet ----------------
    l0 = find_line(EDIT_SHEET, "### R2.1-T12 — New Supplementary Table S-R2.1-E")
    spec = edit_sheet_lines(l0, l0 + 52)
    def block(a: str, b: str) -> str:
        i = next(j for j, x in enumerate(spec) if x.startswith(a))
        k = next(j for j, x in enumerate(spec) if j > i and x.startswith(b))
        return " ".join(x.strip() for x in spec[i:k] if x.strip())
    sheets_txt = " ".join([block("**Sheet 1 (main):**", "**Sheet 2"), block("**Sheet 2", "**Sheet 3"),
                           block("**Sheet 3", "**Sheet 4"), block("**Sheet 4", "**Footnotes:**")])
    fn_lines = spec[next(j for j, x in enumerate(spec) if x.startswith("**Footnotes:**")) + 1:
                    next(j for j, x in enumerate(spec) if x.startswith("**Numbering:**"))]
    fns, cur = [], ""
    for x in fn_lines:
        if x[:2] in {f"{d}." for d in range(1, 10)}:
            if cur:
                fns.append(cur)
            cur = x.strip()
        elif x.strip():
            cur += " " + x.strip()
    fns.append(cur)
    assert len(fns) == 5 and fns[2].startswith("3. Tie order"), fns
    fns[2] += (" [OPEN ITEM, not applied: counts are those of RR-5 run_001 (published Virchow2 bank); this workbook's "
               "Sheet 1 is run_003 (Virchow2 fp32). Proposed run_003 wording in BUILD_REPORT.md.]")
    notes56 = [r[0] for r in wb56["Notes"].iter_rows(values_only=True)][1:]
    l_add = find_line(EDIT_SHEET, "**Table S-R2.1-E additions (R2.1-T12 amendment):**")
    add_txt = edit_sheet_lines(l_add, l_add)[0]
    fn6 = "6. " + add_txt.split('Footnote 6: "')[1].split('" Footnote 7')[0]
    fn7 = "7. " + add_txt.split('Footnote 7: "')[1].split('" Move the TITAN')[0]
    l_asm = find_line(EDIT_SHEET, "- **R2.1-T12 (Table S-R2.1-E, provisional S35):** sheets 5 and 6 are the files above")
    l_mmv = find_line(EDIT_SHEET, "- Sheet captions should say that \"mMV@5 / MV@5 (majority-vote accuracy)\"")
    l_amd = find_line(EDIT_SHEET, "S-R2.1-E: regenerate from `run_003/rr5_replacement_001/table_s3_revised.csv`", l0)
    caption = (
        f"Sheets 1-4 (R2.1-T12, l.{l0}; amendment l.{l_amd}: 38 cells from RR-5 run_003/rr5_replacement_001, 34 "
        f"original cells plus four Virchow2 fp32 cells): {sheets_txt} "
        f"Sheets 5-6 (R2.1-T12 amendment l.{l_add}, assembly l.{l_asm}; Notes-sheet text of the 10-21 build, verbatim): "
        + notes56[0] + " " + notes56[2] + " " + notes56[3]
        + f" mMV@5 / MV@5 (majority-vote accuracy) are the macro recall / per-group recall under the plurality rule of "
          f"the RR-5 package, the same numbers as Sheet 1's macro_recall column at k = 5 (l.{l_mmv}).")
    cap_rows, title_line, title, si_cap = caption_fields(num["current"], caption, B.sha(SI_MD))
    rows = [
        ("workspace_id", WID), ("final_number", f"Supplementary Table {num['current']} ({num['current_field']} of {rel(NUMBERING)}; earlier {num['history']})"),
        *cap_rows[:3], ("footnotes", "\n".join(fns + [fn6, fn7])),
        ("sources", f"Sheets 1-4: {rel(X14)} (sha256 {B.sha(X14)[:16]}…; built 2026-10-22 from RR-5 run_003 "
                    f"rr5_replacement_001, see that directory's input_bindings.json). Sheets 5-6: {rel(X56)} "
                    f"(sha256 {B.sha(X56)[:16]}…; built 2026-10-21 from r2_1_fig2_retrieval_20261021/run_001, RR-5 "
                    "and R3.1 summaries). Values copied unchanged; raw values are those of the source sheets."),
        ("spec", f"docs/manuscript_supplement_edits.md ### R2.1-T12 (l.{l0}) + amendment l.{l_amd} + l.{l_add} + l.{l_asm} + l.{l_mmv}"),
        cap_rows[3],
        ("open_items", "Footnote 3 counts (run_001 vs run_003); Fig. 2C flag set in Sheet 2; Sheet 1 mMV@k column keep/drop; "
                       "TITAN 'best at k = 5 and 10' statement text (see BUILD_REPORT.md)"),
        ("built_by/date", f"scripts/supp_tables/build_S-R2.1-E_merged_20261109.py, {BUILD_DATE}"),
    ]
    write_caption_sheet(new, rows)
    xlsx = OUT / f"Table_{WID}.xlsx"
    save_xlsx(new, xlsx)

    # ---------------- equality checks ----------------
    back = openpyxl.load_workbook(xlsx)
    R.check(back.sheetnames == ["caption"] + [n for _, _, n in order],
            f"sheet order: {back.sheetnames}")
    total_cells = 0
    for wb, s, n in order:
        a, b = sheet_values(wb[s]), sheet_values(back[n])
        ncell = sum(len(r) for r in a)
        diff = sum(not same(x, y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
        ok = len(a) == len(b) and all(len(ra) == len(rb) for ra, rb in zip(a, b)) and diff == 0
        total_cells += ncell
        srcname = X14.name if wb is wb14 else X56.name
        R.check(ok, f"`{n}` == `{srcname}` sheet `{s}`: {len(a)} rows × {len(a[0])} cols, {ncell:,} cells, "
                    f"{diff} differing (value and type, cell for cell)")
    R.check(True, f"total cells compared: {total_cells:,}")
    # per-sheet CSVs (raw unrounded values) written from the merged workbook, compared with the source CSVs
    for wb, s, n in order:
        src_csv = (SHEETS_14.get(s) or SHEETS_56.get(s))[1]
        out_csv = OUT / f"Table_{WID}__{n}.csv"
        write_sheet_csv(back[n], out_csv)
        a, b = pd.read_csv(src_csv, low_memory=False), pd.read_csv(out_csv, low_memory=False)
        eq = a.shape == b.shape and list(a.columns) == list(b.columns)
        nd = 0
        if eq:
            for c in a.columns:
                x, y = a[c], b[c]
                if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                    nd += int((~((x == y) | (x.isna() & y.isna()))).sum())
                else:
                    nd += int((~((x.astype(str) == y.astype(str)) | (x.isna() & y.isna()))).sum())
        R.check(eq and nd == 0, f"`{out_csv.name}` == source CSV `{rel(src_csv)}` ({a.shape[0]} × {a.shape[1]}; "
                                f"{nd} differing values, exact float comparison)")
    R.check(not any(n.startswith("S-R2.1-C") for n in back.sheetnames),
            "S-R2.1-C mMV@5 sheets (deltas 576 rows, identity 48 rows) not copied")

    # ---------------- the independent SI caption, claim by claim ----------------
    if si_cap:
        d1 = pd.read_csv(OUT / f"Table_{WID}__Sheet1_main.csv", low_memory=False)
        d2 = pd.read_csv(OUT / f"Table_{WID}__Sheet2_per_group_k5.csv", low_memory=False)
        d3 = pd.read_csv(OUT / f"Table_{WID}__Sheet3_TITAN_repair.csv", low_memory=False)
        d4 = pd.read_csv(OUT / f"Table_{WID}__Sheet4_changed_slots.csv", low_memory=False)
        d5 = pd.read_csv(OUT / f"Table_{WID}__Sheet5_retrieval_metrics.csv", low_memory=False)
        d6 = pd.read_csv(OUT / f"Table_{WID}__Sheet6_per_group_k5.csv", low_memory=False)
        R.si_check(len(d1) == 152 and d1.cell_id.nunique() == 38 and sorted(d1.k.unique().tolist()) == [1, 3, 5, 10],
                   f"'38 cells, 152 rows', k = 1, 3, 5, 10: Sheet 1 has {len(d1)} rows, {d1.cell_id.nunique()} cell_id, "
                   f"k {sorted(d1.k.unique().tolist())}")
        enc = d1.groupby("Encoder").Aggregation.nunique()
        R.si_check(len(enc) == 10 and int((enc == 4).sum()) == 9,
                   f"'eight submitted encoders and H-Optimus-1, each with four aggregations; CONCH v1.5 with TITAN … "
                   f"regenerated and submitted': Sheet 1 aggregations per encoder {({k: int(v) for k, v in enc.items()})} "
                   f"(Conchv1.5/TITAN appears as 2 cells: {sorted(d1[d1.Encoder == 'Conchv1.5'].cell_id.unique())})")
        grid = d5[d5.kind == "grid"]
        nq, npat = sorted(grid.n_queries.dropna().unique().tolist()), sorted(grid.n_patients.dropna().unique().tolist())
        R.si_check(nq == [2778] and npat == [1876],
                   f"'2,778 query slides from 1,876 patients': Sheet 5 grid rows n_queries {nq}, n_patients {npat}")
        gcol = next(c for c in d6.columns if "group" in c.lower())
        ng = d6[gcol].nunique()
        R.si_check(ng == 15, f"'15 groups (14 tumor groups and NORMAL)': Sheet 6 `{gcol}` has {ng} distinct values")
        dcols = [c for c in d4.columns if c.startswith("differs")]
        per = {c: sorted(f"{e} k = {k}" for e, k in d4.loc[d4[c].astype(str) == "True", ["Encoder", "k"]].values)
               for c in dcols}
        anyd = sorted(f"{e} k = {k}" for e, k in d4.loc[d4[dcols].astype(str).eq("True").any(axis=1), ["Encoder", "k"]].values)
        want = sorted(["Virchow2 k = 5", "EXAONEPath k = 10", "Hibou-L k = 5"])
        R.si_check(anyd == want,
                   f"Sheet 4 'three slots whose winner differs from the submitted analysis (Virchow2 k = 5, EXAONEPath "
                   f"k = 10, Hibou-L k = 5)': the three named slots are exactly the generous-rule reversed-order set; "
                   f"the caption says 'under both rules and both tie orders', and the union over rule × tie order is "
                   f"{anyd} (by column {per}). Suggest naming the rule/order the three refer to.")
        sys5 = d5.drop_duplicates(["system", "aggregation"])
        n_enc_cells = int(((sys5.kind == "grid") & (sys5.system != "RetCCL")).sum() + (sys5.system == "TITAN").sum())
        n_ret = int((sys5.system == "RetCCL").sum())
        others = sorted(sys5[sys5.kind.isin(["e2e", "floor"]) & (sys5.system != "TITAN")].system.tolist())
        R.si_check(n_enc_cells == 37 and n_ret == 4 and len(others) == 3,
                   f"Sheet 5 '37 encoder–aggregation cells …, RetCCL with four aggregations, PRISM, SISH and the ImageNet "
                   f"ResNet-50 median-pooling floor': {n_enc_cells} cells (36 grid + TITAN), RetCCL {n_ret}, others {others}")
        v2 = sorted(d5[(d5.system == "Virchow2")].cell_id.unique().tolist())
        R.si_check(len(v2) == 4 and all(x.startswith("virchow2_fp32/") for x in v2),
                   f"Sheet 5 'Virchow2 from the fp32 feature bank': Virchow2 cell_id {v2}")
        reps = sorted(d5.n_replicates.dropna().unique().tolist())
        R.si_check(reps == [5000], f"'5,000 draws': Sheet 5 n_replicates {reps}")
        rules = sorted(d3.rule.dropna().unique().tolist())
        R.si_check(len(rules) >= 2, f"Sheet 3 'TITAN repair … under both rules': rules {rules}")
        f2c = [c for c in d2.columns if "2C" in c or "fig" in c.lower()]
        R.si_check(bool(f2c) and d2.cell_id.nunique() == 38,
                   f"Sheet 2 'per-group F1 … for all 38 cells, with the Fig. 2C pipelines flagged': {d2.cell_id.nunique()} "
                   f"cell_id; flag columns {f2c} (two flag sets; which one is 'the Fig. 2C pipelines' is the carried open "
                   f"item 'Fig. 2C flag set in Sheet 2')")

    # ---------------- footnote 3 ----------------
    f1, f3 = footnote3_counts(t001), footnote3_counts(t003)
    spec_vals = dict(no_tie=96, no_tie_exact=96, tie=36, tie_yottixel=32, tie_titan_submitted=4, between=23,
                     C32=(23, 9), H36=(24, 12))
    def summary(f):
        return (f"comparable rows {f['n_rows']} ({f['n_cells']} cells × 4 k); no-tie {f['no_tie']}, exact "
                f"{f['no_tie_exact']}; tie {f['tie']} (Yottixel {f['tie_yottixel']}, TITAN-submitted "
                f"{f['tie_titan_submitted']}); max |Δ| in tie rows {f['max_tie_diff']:.6f} ({f['max_tie_diff_row']}); "
                f"submitted between canonical and reversed {f['between']} (non-Virchow2 {f['between_excl_virchow2']}, "
                f"Virchow2 {f['between_virchow2']}); reversed-order plurality winners C32 "
                f"{f['winners_reversed_C32']}, H36 {f['winners_reversed_H36']}; canonical C32 "
                f"{f['winners_canonical_C32']}, H36 {f['winners_canonical_H36']}")
    r001 = (f1["no_tie"], f1["no_tie_exact"], f1["tie"], f1["tie_yottixel"], f1["tie_titan_submitted"], f1["between"],
            (f1["winners_reversed_C32"].get("Yottixel"), f1["winners_reversed_C32"].get("Optimal Transport")),
            (f1["winners_reversed_H36"].get("Yottixel"), f1["winners_reversed_H36"].get("Optimal Transport")))
    R.check(r001 == tuple(spec_vals.values()) and round(f1["max_tie_diff"], 3) <= 0.025,
            f"footnote 3 as printed reproduces from run_001 by this script's selection: {summary(f1)}")
    R.check(f1["submitted_by_row"] == f3["submitted_by_row"],
            "submitted Table S3 values identical in run_001 and run_003 for all 132 comparable rows "
            "(virchow2 ↔ virchow2_fp32 mapped)")
    R.check(True, f"run_003 (this table's source) by the same selection: {summary(f3)}")
    w = pd.read_csv(win003)
    cnt = w[(w.row_type == "count") & (w.rule == "plurality") & (w.tie_order == "reversed")]
    wc = {s_: cnt[cnt.slot_set == s_].set_index("winner").weight.to_dict() for s_ in ("C32", "H36")}
    R.check(wc["C32"] == {k: float(v) for k, v in f3["winners_reversed_C32"].items()}
            and wc["H36"] == {k: float(v) for k, v in f3["winners_reversed_H36"].items()},
            f"run_003 winners.csv count rows (plurality, reversed) agree with the table-derived argmax: {wc}")
    nte = f3["no_tie_not_exact"]
    assert len(nte) == 1
    x = nte[0]

    # ---------------- outputs ----------------
    B.write(OUT, {"footnote3": {"run_001": {k: v for k, v in f1.items() if k != "submitted_by_row"},
                                "run_003": {k: v for k, v in f3.items() if k != "submitted_by_row"}}})
    manifest_row = {"final_number": num["current"], "path": rel(xlsx)}
    proposed = (
        "3. Tie order: Yottixel distances are coarsely quantized, so the k-th neighbor is often tied. Values use the "
        "canonical order (ascending library index); the submitted analysis used an order that is not recoverable "
        f"(ADR0010). Of the {f3['n_rows']} rows with a submitted Table S3 value, the {f3['no_tie']} rows without ties "
        f"reproduce the submitted values exactly except one Virchow2 row, whose features were re-extracted in fp32 "
        f"(median pooling, k = {x['k']}: {x['value']:.3f} versus {x['submitted']:.3f} submitted); the {f3['tie']} rows "
        f"with ties ({f3['tie_yottixel']} Yottixel, {f3['tie_titan_submitted']} TITAN-submitted) differ from Table S3 "
        f"by < {math.ceil(f3['max_tie_diff'] * 1000) / 1000:.3f} macro F1, and for {f3['between']} of {f3['tie']} the "
        "submitted value lies between the canonical- and reversed-order values. Winner counts under the reversed order: "
        f"{f3['winners_reversed_C32']['Yottixel']} of 32 Yottixel / {f3['winners_reversed_C32']['Optimal Transport']} "
        f"Optimal Transport (plurality); with H-Optimus-1, {f3['winners_reversed_H36']['Yottixel']} / "
        f"{f3['winners_reversed_H36']['Optimal Transport']}. Virchow2 rows use the fp32 feature bank; the "
        f"submitted-bank values are in Supplementary Table {numbering('S-R2.1-F')['current']} (workspace S-R2.1-F).")
    old_fn3 = fns[2].split(" [OPEN ITEM")[0]
    rep = f"""# BUILD_REPORT — Table {WID} (final {num['current']}), merged workbook — {BUILD_DATE}

Script: `scripts/supp_tables/build_S-R2.1-E_merged_20261109.py`; run {now_utc()}; output `{rel(OUT)}`.
Assembly only: no statistic is computed; values are copied cell for cell from two accepted builds. The footnote-3
counts below are re-derived by selection/counting from retained RR-5 outputs (no new evaluation).

## Spec and version implemented

- `docs/manuscript_supplement_edits.md` (sha256 {B.sha(EDIT_SHEET)[:16]}…) ### R2.1-T12 (l.{l0}); amendment 2026-10-16
  (l.{l_amd}: regenerate from `run_003/rr5_replacement_001`, 38 cells incl. 4 `virchow2_fp32`); sheets 5-6 amendment
  (l.{l_add}) and the 2026-10-21 assembly addendum (l.{l_asm}, l.{l_mmv}). Not author-approved (G-8 reading pending).
- Final number {num['current']} from `scripts/figures/supp_table_numbering.json` ({num['current_field']}; earlier {num['history']}).

## Sources (sha256)

""" + "\n".join(f"- `{k}` {v['sha256']} — {v['role']}" for k, v in B.d.items()) + f"""

## Output

- `Table_{WID}.xlsx`: `caption` + {', '.join(f'`{n}`' for _, _, n in order)}. Sheet names 5-6 renamed from the source
  (`S-R2.1-E s5 retrieval` → `Sheet5_retrieval_metrics`, `S-R2.1-E s6 per-group k5` → `Sheet6_per_group_k5`); values unchanged.
- Per-sheet CSVs `Table_{WID}__<sheet>.csv` carry the full-precision values (the source sheets already hold the raw
  unrounded columns beside the formatted ones; no rounding applied here).
- Left out: the 10-21 workbook's `Notes` sheet (its footnote 6/7 and caption texts are carried into `caption`) and the
  two S-R2.1-C mMV@5 sheets (Table S-R2.1-C, final S15).

## Checks ({R.summary()})

{R.md_checks()}

## SI caption consistency (lead's caption vs this workbook; findings, not build failures)

{R.md_si(si_cap, title_line)}
## Footnote 3 — open item (not applied)

**What the footnote says (spec l.{l0 + 37}–{l0 + 42}, carried in the caption sheet with an OPEN ITEM flag):**

> {old_fn3}

**What run_003 is.** `r2_1_virchow2_fp32_main_20261013/run_003/rr5_replacement_001/` is the RR-5 replacement package
adopted on 2026-10-16 (edit sheet l.{l_amd}): the 34 original cells byte-identical to `r2_1_main_grid_plurality_20261012/run_001/`
plus the four Virchow2 cells recomputed from the fp32 feature bank (lineage `virchow2_fp32_replacement`). Sheet 1 of this
table is its `table_s3_revised.csv`. The footnote's numbers were computed on run_001, where Virchow2 used the published bank.

**Counts verified from retained outputs** (this script's own selection over `table_s3_revised.csv`; generous = `oracle`
rule, macro F1, canonical vs reversed tie order, `n_order_sensitive_k > 0` = tie row, exact = |Δ| < 5e-7):

| Quantity | Footnote (run_001) | run_001 re-derived | run_003 re-derived |
|---|---|---|---|
| no-tie rows reproducing submitted exactly | 96 of 96 | {f1['no_tie_exact']} of {f1['no_tie']} | {f3['no_tie_exact']} of {f3['no_tie']} |
| tie rows (Yottixel / TITAN-submitted) | 36 (32 / 4) | {f1['tie']} ({f1['tie_yottixel']} / {f1['tie_titan_submitted']}) | {f3['tie']} ({f3['tie_yottixel']} / {f3['tie_titan_submitted']}) |
| max |Δ| in tie rows | ≤ 0.025 | {f1['max_tie_diff']:.6f} | {f3['max_tie_diff']:.6f} |
| submitted between canonical and reversed | 23 of 36 | {f1['between']} | {f3['between']} (non-Virchow2 {f3['between_excl_virchow2']}, Virchow2 {f3['between_virchow2']}; run_001: {f1['between_excl_virchow2']} / {f1['between_virchow2']}) |
| reversed-order plurality winners C32 (Yottixel / OT) | 23 / 9 | {f1['winners_reversed_C32'].get('Yottixel')} / {f1['winners_reversed_C32'].get('Optimal Transport')} | {f3['winners_reversed_C32'].get('Yottixel')} / {f3['winners_reversed_C32'].get('Optimal Transport')} |
| with H-Optimus-1, H36 (Yottixel / OT) | 24 / 12 | {f1['winners_reversed_H36'].get('Yottixel')} / {f1['winners_reversed_H36'].get('Optimal Transport')} | {f3['winners_reversed_H36'].get('Yottixel')} / {f3['winners_reversed_H36'].get('Optimal Transport')} |

The one run_003 no-tie row that no longer reproduces is `{x['cell']}` k = {x['k']}: {x['value']:.6f} (fp32) vs
{x['submitted']:.6f} submitted (|Δ| {abs(x['value'] - x['submitted']):.6f}); the other
{f3['virchow2_no_tie_exact']} of {f3['virchow2_no_tie']} Virchow2 no-tie rows are identical. The between-count change
(23 → 22) is in the Virchow2 rows. Files: `{rel(t003)}` (sha256 {B.sha(t003)[:16]}…) for the reproduction counts and
the winners (argmax over aggregations per encoder × k), cross-checked against `{rel(win003)}` count rows
(sha256 {B.sha(win003)[:16]}…); run_001 from `{rel(t001)}` (sha256 {B.sha(t001)[:16]}…). These agree with the edit
sheet's 2026-10-22 record (l.≈9753) and the 10-22 C build's discrepancy 1-2.

**Proposed footnote 3 (run_003; for the lead/Haoran, not applied):**

> {proposed}

The "Supplementary Table {numbering('S-R2.1-F')['current']}" pointer is the current ({numbering('S-R2.1-F')['current_field']}) number of S-R2.1-F; check at renumbering.
The bound "≤ 0.025" of the printed footnote is true only after rounding (max |Δ| {f3['max_tie_diff']:.6f},
{f3['max_tie_diff_row']}, identical in run_001 and run_003); the proposal prints "< {math.ceil(f3['max_tie_diff'] * 1000) / 1000:.3f}"
instead. Keep "≤ 0.025" if the author prefers the 3-decimal convention.
Canonical-order winner counts are unchanged by the replacement (C32 Yottixel {f3['winners_canonical_C32'].get('Yottixel')} /
OT {f3['winners_canonical_C32'].get('Optimal Transport')}; H36 {f3['winners_canonical_H36'].get('Yottixel')} / {f3['winners_canonical_H36'].get('Optimal Transport')}),
consistent with the "26 of 32; 28 of 36" statement elsewhere (l.≈7560).

## Open items carried from the source builds (unchanged by this merge)

- Sheet 2 Fig. 2C flag: current Fig. 2C pipeline set (adds H-Optimus-1 + Yottixel) vs RR-5 `fig2c_groups.csv` set; both
  flags are in the sheet (10-22 C build discrepancy 3). Author choice.
- Sheet 1 `macro_recall` (mMV@k) column: builder inclusion per the 10-21 addendum; keep/drop is Haoran's decision.
- TITAN macro-F1 "best at k = 5 and 10" sentence: spec says move it into the Sheet 1 caption, but the text is elided in
  the spec; not composed (author wording owed).
- Sheet 4 caption: the three changed slots hold for the generous rule under the reversed tie order; canonical order
  reproduces every submitted winner (10-22 C build note).
- The `caption` sheet mirrors the lead's SI caption verbatim when written (`caption`), with the edit-sheet draft in
  `caption_spec_draft`; the SI Markdown is the source of truth for wording.
"""
    (OUT / "BUILD_REPORT.md").write_text(rep, encoding="utf-8")
    (OUT / "manifest_row.json").write_text(json.dumps({WID: manifest_row}, indent=1) + "\n")
    write_sha256sums(OUT)
    print(f"{WID}: {len(R.checks)} checks, {R.n_fail()} FAIL -> {OUT}")
    if R.n_fail():
        sys.exit(1)


if __name__ == "__main__":
    main()
