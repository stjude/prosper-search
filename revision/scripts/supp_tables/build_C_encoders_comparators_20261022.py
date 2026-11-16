#!/usr/bin/env python3
"""Assemble group C (encoders / comparators) supplementary tables, 2026-10-22.

Assembly only: every number is copied from an accepted source CSV/JSON and
formatted; no statistic is recomputed. Spec numbers quoted in
docs/manuscript_supplement_edits.md are checked against the sources at the
printed rounding; mismatches are logged (BUILD_REPORT.md), never "fixed".

Tables (spec heading in docs/manuscript_supplement_edits.md; line numbers are
resolved at run time and written to each caption sheet):
  S-R2.1-A   ### R2.1-T4 (+ 10-16 amendment inside ### R2.1-T12)
             src: r2_1_virchow2_fp32_main_20261013/run_003/rr5_replacement_001/
                  {hit_precision,tie_census}.csv; contrasts.csv from
                  r2_1_cpu_20261002/main_attempt_001 and
                  r2_1_main_grid_comparators_20261011/main_{uni2,virchow2}_attempt_001
  S-R2.1-E   ### R2.1-T12 (+ amendment; sheets 1-4 only; 5-6 in supp_tables_20261021)
             src: rr5_replacement_001/{table_s3_revised,rule_contrasts,hit_precision,
                  per_class,titan_contrast,winners,tie_sensitivity,tie_census}.csv
  S-R3.1     ### R3.1-T2
             src: r3_1_comparators_whole_patient_20261008, r3_1_retccl_whole_patient_20261016,
                  r3_1_prism_20261018/eval_{native,mosaic}_20261019
  S25        ### R4.b-T5 (R4.b-T9 is Methods wording only)
             src: c2_metric_suite/metrics_uni2_ot_cosine.json,
                  r2_3_internal_failure_cis_20261004/{summary_overall,macro_summary,paired_differences}.csv,
                  r2_3_internal_failure_cis_ext_prod_20261005/summary_ext.csv,
                  review_scratch_astra_session_20261004/recomputed.json
  S-R2.1-B   ### R2.1-T6 / ### R2.1-T6-A   src: r2_1_filter_tf2_20261016/writing_001
  S-R2.1-D   ### R2.1-T11 / ### R2.1-T11-A src: same + figures_20261021/fig3_tumor_filtering/
                  source_data_panelC_groups_all_arms.csv (block 2)
  S-R2.1-F   Methods note + T12 amendment; layout = builder's proposal
             src: r2_1_virchow2_fp32_main_20261013/run_003/comparison_001 and submitted
                  Table_S3_raw_search_metrics.xlsx

Read-only on inputs; idempotent; writes only to
$PROSPER_OUT/supp_tables_20261022/C_encoders_comparators/.
"""
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output"))
OUT = OUT_ROOT / "supp_tables_20261022" / "C_encoders_comparators"
EDITS = REPO / "docs" / "manuscript_supplement_edits.md"
SUBMITTED_S3 = (REPO / "docs/prosper_submission/supplementary/supplementary_tables/"
                "Table_S3_raw_search_metrics.xlsx")
PREV = OUT_ROOT / "supp_tables_20261021"

RR5 = OUT_ROOT / "r2_1_virchow2_fp32_main_20261013/run_003/rr5_replacement_001"
CMP = OUT_ROOT / "r2_1_virchow2_fp32_main_20261013/run_003/comparison_001"
R31C = OUT_ROOT / "r3_1_comparators_whole_patient_20261008"
R31R = OUT_ROOT / "r3_1_retccl_whole_patient_20261016"
R31N = OUT_ROOT / "r3_1_prism_20261018/eval_native_20261019"
R31M = OUT_ROOT / "r3_1_prism_20261018/eval_mosaic_20261019"
TF2W = OUT_ROOT / "r2_1_filter_tf2_20261016/writing_001"
FIG3 = OUT_ROOT / "figures_20261021/fig3_tumor_filtering/source_data_panelC_groups_all_arms.csv"
FIG2C = OUT_ROOT / "figures_20261021/fig2_benchmark/source_data_panelC.csv"
FIG2SUM = OUT_ROOT / "r2_1_fig2_retrieval_20261021/run_001/summary.csv"

MINUS = "\u2212"
INPUTS = {}          # path -> sha256 (filled by inp())
CHECKS = {}          # table -> list of passed assertion strings
ISSUES = {}          # table -> list of dicts
NOTES = {}           # table -> list of free-text report notes
BUILT = {}           # table -> dict(xlsx, sheets=[(name, csv, rows, cols)], spec)


# ---------------------------------------------------------------- helpers
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def inp(p):
    p = Path(p)
    assert p.exists(), f"missing input {p}"
    if str(p) not in INPUTS:
        INPUTS[str(p)] = sha256(p)
    return p


def rcsv(p, **kw):
    return pd.read_csv(inp(p), **kw)


def rjson(p):
    with open(inp(p)) as f:
        return json.load(f)


def ok(t, msg):
    CHECKS.setdefault(t, []).append(msg)


def issue(t, item, spec_line="", spec_value="", source_value="", note=""):
    ISSUES.setdefault(t, []).append(dict(item=item, spec_line=spec_line,
                                         spec_value=spec_value,
                                         source_value=source_value, note=note))


def note(t, msg):
    NOTES.setdefault(t, []).append(msg)


_EDIT_LINES = None


def edit_lines():
    global _EDIT_LINES
    if _EDIT_LINES is None:
        _EDIT_LINES = inp(EDITS).read_text().splitlines()
    return _EDIT_LINES


def spec_block(prefix):
    """Return (start_line_1based, [(lineno, text), ...]) for '### <prefix>' up to next '### '."""
    L = edit_lines()
    hits = [i for i, s in enumerate(L) if s.startswith("### " + prefix)
            and (len(s) == 4 + len(prefix) or not s[4 + len(prefix)].isalnum()
                 and s[4 + len(prefix)] != "-")]
    assert hits, f"spec heading not found: {prefix}"
    i = hits[0]
    j = i + 1
    while j < len(L) and not L[j].startswith("### "):
        j += 1
    return i + 1, [(k + 1, L[k]) for k in range(i, j)]


def find_line(substr, start=1):
    L = edit_lines()
    for k in range(start - 1, len(L)):
        if substr in L[k]:
            return k + 1
    return None


def neg(s):
    return s.replace("-", MINUS) if isinstance(s, str) else s


def f(x, d=3, signed=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)) or pd.isna(x):
        return "—"
    s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    if s.lstrip("+-").strip("0.") == "":           # rounds to zero: keep the true sign (TF-2 convention)
        s = (("-" if x < 0 else "+") if signed else "") + s.lstrip("+-")
    return s.replace("-", MINUS)


def fci(p, lo, hi, d=3, signed=False):
    if pd.isna(lo) or pd.isna(hi):
        return f(p, d, signed)
    return f"{f(p, d, signed)} [{f(lo, d, signed)}, {f(hi, d, signed)}]"


NUM_RE = re.compile(r"[+\-\u2212]?\d+\.\d+")


def nums(s):
    """Numbers printed in a spec cell, with their decimals."""
    out = []
    for m in NUM_RE.findall(str(s)):
        t = m.replace(MINUS, "-")
        out.append((float(t), len(t.split(".")[1])))
    return out


def same_at(spec_val, dec, src):
    return abs(round(float(src), dec) - spec_val) < 10 ** (-dec) / 2 + 1e-12


def check_cell(t, item, spec_line, spec_text, src_vals):
    """Compare the numbers printed in spec_text against src_vals at printed rounding."""
    sn = nums(spec_text)
    if len(sn) != len(src_vals):
        issue(t, item, spec_line, spec_text, " / ".join(f(v, 6) for v in src_vals),
              "count of printed numbers differs from source fields")
        return False
    bad = [(v, d, s) for (v, d), s in zip(sn, src_vals) if not same_at(v, d, s)]
    if bad:
        issue(t, item, spec_line, spec_text, " / ".join(f(v, 6) for v in src_vals),
              "[SPEC/SOURCE MISMATCH] printed value differs at its rounding")
        return False
    ok(t, f"{item}: spec '{spec_text.strip()}' (l.{spec_line}) == source")
    return True


def md_table(lines):
    """Parse consecutive markdown table lines -> (header, rows, linenos)."""
    rows, lns = [], []
    for ln, s in lines:
        s = s.strip()
        if s.startswith(">"):
            s = s.lstrip("> ").strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        rows.append(cells)
        lns.append(ln)
    return rows[0], rows[1:], lns[1:]


def md_tables_in(block):
    """Split a spec block into its markdown tables (lists of (ln, text))."""
    tabs, cur = [], []
    for ln, s in block:
        if s.strip().lstrip("> ").startswith("|"):
            cur.append((ln, s))
        elif cur:
            tabs.append(cur)
            cur = []
    if cur:
        tabs.append(cur)
    return tabs


def cited(t, p, prefix, where=""):
    h = INPUTS[str(inp(p))]
    if h.startswith(prefix):
        ok(t, f"source hash {Path(p).name} {prefix}… == spec-cited{(' ' + where) if where else ''}")
    else:
        issue(t, f"source hash {p}", where, prefix + "…", h[:8] + "…", "bound file differs from spec-cited hash")


def caption_sheet(table_id, spec_refs, caption_lines, extra=None):
    rows = [("Table ID", f"{table_id} (provisional; final numbering at assembly — "
             "see docs/analysis/supplementary_numbering_map_20261022.md)")]
    for heading, line in spec_refs:
        rows.append(("Spec", f"docs/manuscript_supplement_edits.md {heading} (l.{line})"))
    for ln, s in caption_lines:
        rows.append((f"edit-sheet l.{ln}", s))
    for k, v in (extra or []):
        rows.append((k, v))
    return pd.DataFrame(rows, columns=["field", "text"])


def write_table(t, fname, caption_df, sheets, spec):
    """sheets: list of (sheet_name, DataFrame). Writes xlsx + one CSV per data sheet."""
    OUT.mkdir(parents=True, exist_ok=True)
    xp = OUT / fname
    info = []
    with pd.ExcelWriter(xp, engine="openpyxl") as w:
        caption_df.to_excel(w, sheet_name="caption", index=False)
        for name, df in sheets:
            assert len(name) <= 31, name
            df.to_excel(w, sheet_name=name, index=False)
            cp = OUT / f"{xp.stem}__{name}.csv"
            df.to_csv(cp, index=False)
            info.append((name, cp.name, len(df), df.shape[1]))
    BUILT[t] = dict(xlsx=fname, sheets=info, spec=spec)
    ok(t, f"wrote {fname} with sheets caption + {[i[0] for i in info]}")


# ---------------------------------------------------------------- S-R3.1
R31_COLS = [("hit5", "macro"), ("hit5", "micro"), ("precision5", "macro"),
            ("precision5", "micro"), ("hit10", "macro")]
R31_HDR = ["macro hit@5", "micro hit@5", "macro precision@5", "micro precision@5", "macro hit@10"]


def _fmt_like(spec_cell, vals, signed):
    """Format source triplet with the decimals printed in the spec cell (3 dp default)."""
    sn = nums(spec_cell)
    decs = [d for _, d in sn] if len(sn) == 3 else [3, 3, 3]
    p, lo, hi = vals
    s = f"{f(p, decs[0], signed)} [{f(lo, decs[1], signed)}, {f(hi, decs[2], signed)}]"
    m = re.search(r"\]([^\d\]\s]+)\s*$", spec_cell)
    return s + (m.group(1) if m else "")


def build_r31():
    t = "S-R3.1"
    line, block = spec_block("R3.1-T2")
    summ = {"C": rcsv(R31C / "summary.csv"), "R": rcsv(R31R / "summary.csv"),
            "N": rcsv(R31N / "summary.csv"), "M": rcsv(R31M / "summary.csv")}
    pair = {"C": rcsv(R31C / "paired_differences.csv"), "R": rcsv(R31R / "paired_differences.csv"),
            "N": rcsv(R31N / "paired.csv"), "M": rcsv(R31M / "paired.csv")}
    srcname = {"C": "r3_1_comparators_whole_patient_20261008", "R": "r3_1_retccl_whole_patient_20261016",
               "N": "r3_1_prism_20261018/eval_native_20261019", "M": "r3_1_prism_20261018/eval_mosaic_20261019"}
    A = [("C", "primary"), ("C", "yottixel"), ("C", "median"), ("C", "sish"),
         ("R", "retccl_yottixel"), ("R", "retccl_ot"), ("R", "retccl_sampler"), ("R", "retccl_median"),
         ("C", "resnet50"), ("N", "prism_native"), ("M", "prism_mosaic")]
    B = [("C", "sish", "primary"), ("C", "resnet50", "primary"), ("C", "median", "primary"),
         ("C", "yottixel", "primary"), ("C", "resnet50", "median"),
         ("R", "retccl_yottixel", "primary"), ("R", "retccl_median", "primary"),
         ("R", "retccl_median", "median"), ("R", "retccl_yottixel", "yottixel"),
         ("R", "retccl_median", "resnet50"),
         ("N", "prism_native", "primary"), ("M", "prism_mosaic", "primary"),
         ("M", "prism_mosaic", "prism_native"), ("N", "prism_native", "titan_repaired"),
         ("M", "prism_mosaic", "titan_repaired")]
    Cb = [("C", "primary_sub", None), ("C", "primary_tumour", None), ("C", "primary_tumour", "primary_sub")]

    def getrow(src, pop, a, b, metric, avg):
        if b is None:
            d = summ[src]
            r = d[(d.population == pop) & (d.config == a) & (d.metric == metric) & (d.average == avg)]
        else:
            d = pair[src]
            r = d[(d.population == pop) & (d.minuend == a) & (d.subtrahend == b)
                  & (d.metric == metric) & (d.average == avg)]
        assert len(r) == 1, (src, pop, a, b, metric, avg, len(r))
        return r.iloc[0]

    tabs = md_tables_in(block)
    assert len(tabs) == 3, f"expected 3 md tables in R3.1-T2, got {len(tabs)}"
    blocks = [("A", "all_tissue_2778", A, tabs[0]), ("B", "all_tissue_2778", B, tabs[1]),
              ("C", "tumour_only_2604", Cb, tabs[2])]
    fmt_rows, raw_rows = [], []
    for bname, pop, spec_rows, tab in blocks:
        hdr, rows, lns = md_table(tab)
        assert len(rows) == len(spec_rows), (bname, len(rows), len(spec_rows))
        ok(t, f"Block {bname}: {len(rows)} spec rows == {len(spec_rows)} mapped source rows")
        for (srcrow, cells, ln) in zip(spec_rows, rows, lns):
            src, a, b = (srcrow + (None,))[:3] if len(srcrow) == 2 else srcrow
            signed = b is not None
            out = {"block": bname, "row label (spec)": cells[0]}
            for (metric, avg), h, spec_cell in zip(R31_COLS, R31_HDR, cells[1:6]):
                r = getrow(src, pop, a, b, metric, avg)
                vals = (r.point, r.ci_lo, r.ci_hi)
                assert all(np.isfinite(vals)), (a, b, metric)
                out[h] = _fmt_like(spec_cell, vals, signed)
                check_cell(t, f"Block {bname} '{cells[0][:40]}' {h}", ln, spec_cell, list(vals))
                if out[h].replace(" ", "") != spec_cell.replace(" ", ""):
                    issue(t, f"Block {bname} '{cells[0][:40]}' {h} string", ln, spec_cell, out[h],
                          "formatted string differs from spec cell text (values agree at rounding)")
                raw_rows.append(dict(block=bname, label=cells[0], source=srcname[src], population=pop,
                                     config_or_minuend=a, subtrahend=b or "", metric=metric, average=avg,
                                     point=r.point, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                                     n_queries=r.n_queries, n_patients=r.n_patients,
                                     n_replicates=r.n_replicates, n_nonfinite=r.n_nonfinite,
                                     formatted=out[h]))
            out["notes"] = cells[6] if len(cells) > 6 else ""
            out["source"] = f"{srcname[src]}/{'summary.csv' if b is None else ('paired.csv' if src in 'NM' else 'paired_differences.csv')}"
            out["config (source key)"] = a if b is None else f"{a} − {b}"
            fmt_rows.append(out)
    raw = pd.DataFrame(raw_rows)
    # population / count checks
    a_ = raw[raw.population == "all_tissue_2778"]
    c_ = raw[raw.population == "tumour_only_2604"]
    assert set(a_.n_queries) == {2778} and set(a_.n_patients) == {1876}, "Block A/B counts"
    assert set(c_.n_queries) == {2604} and set(c_.n_patients) == {1800}, "Block C counts"
    assert set(raw.n_replicates) == {5000} and set(raw.n_nonfinite) == {0}
    ok(t, "Blocks A/B: n_queries 2778, n_patients 1876; Block C: 2604/1800; B=5000, 0 non-finite replicates")
    for k, s in summ.items():
        g = s[s.population == "tumour_only_2604"]
        if len(g):
            assert set(g.n_groups) == {14}
            ok(t, "Block C n_groups 14 (summary.csv)")
        g = s[s.population == "all_tissue_2778"]
        assert set(g.n_groups) == {15}
    ok(t, "Blocks A/B n_groups 15 in all four summary.csv files")
    # footnotes
    yl = getrow("C", "all_tissue_2778", "yottixel", "primary", "precision5", "micro")
    fn = {L: (ln, s) for ln, s in block for L in "abcdefghijk" if s.startswith(f"- {L}. ")}
    if same_at(-0.00036, 5, yl.ci_hi):
        ok(t, f"footnote c: Yottixel−primary micro precision@5 upper bound {yl.ci_hi:.6f} → −0.00036")
    else:
        issue(t, "footnote c", fn["c"][0], "−0.00036", f(yl.ci_hi, 6))
    r = getrow("M", "all_tissue_2778", "prism_mosaic", "primary", "hit5", "macro")
    if same_at(-0.0003, 4, r.ci_hi):
        ok(t, f"footnote i: PRISM-mosaic−primary macro hit@5 upper {r.ci_hi:.6f} → −0.0003")
    else:
        issue(t, "footnote i", fn["i"][0], "−0.0003", f(r.ci_hi, 6))
    note(t, "footnote i: '+0.0007, +0.0003' under other seeds is not in the bound source files; not checked")
    r = getrow("M", "all_tissue_2778", "prism_mosaic", "titan_repaired", "hit5", "macro")
    if same_at(0.000009, 6, r.ci_hi):
        ok(t, f"footnote k: PRISM-mosaic−TITAN macro hit@5 upper {r.ci_hi:.7f} → +0.000009")
    else:
        issue(t, "footnote k", fn["k"][0], "+0.000009", f(r.ci_hi, 7))
    jrows = []
    for src in "NM":
        r = getrow(src, "all_tissue_2778", "titan_repaired", "primary", "hit5", "macro")
        jrows.append(r)
        check_cell(t, f"footnote j TITAN−primary macro hit@5 ({srcname[src]})", fn["j"][0],
                   "−0.0001 [−0.025, +0.024]", [r.point, r.ci_lo, r.ci_hi])
    rec = rjson(R31C / "reconciliation.json")
    sp = rec["sish_padding"]
    fb = (sp["queries_with_padding"]["sish"], sp["sish_rows_padded"], sp["sish_queries_zero_neighbours"])
    if fb == (35, 206, 1):
        ok(t, "footnote b: SISH padding 35 queries / 206 positions / 1 zero-neighbour query (reconciliation.json sish_padding)")
    else:
        issue(t, "footnote b", fn["b"][0], "35 / 206 / 1", str(fb))
    if sp["queries_with_padding"].get("primary_tumour") == 1:
        ok(t, "footnote d: one tumour-only query padded (reconciliation.json queries_with_padding.primary_tumour = 1); slide id 16738 not in this key, not checked")
    pe = rec["legacy_rung_4to5"]["point_estimates"]
    bl = [ln for ln, s in block if s.startswith("- Neighbour-label scores")][0]
    check_cell(t, "Bayesian line macro accuracy (rung4, rung5)", bl, "0.770 and 0.778",
               [pe["rung4_macro_accuracy"], pe["rung5_macro_accuracy"]])
    note(t, "Bayesian line wording: spec reads 'Neighbour-label scores (Bayesian formulation) versus plain majority "
            "vote … 0.770 and 0.778'; source keys are rung4 = plain majority 0.7697 and rung5 = Bayesian 0.7776, i.e. "
            "the values are printed in the order (majority, Bayesian) while the sentence names (Bayesian, majority) "
            "first. Reported for author check; values copied unchanged.")
    note(t, "footnote a (87 queries with ties spanning ranks 5/6) cites review §3; not re-derivable from the bound "
            "files and not checked. Cross-table: RR-5 tie_census boundary_tie_5 for uni2/yottixel is 404, a different "
            "tie definition/list (any tie at the k boundary vs the reviewer's 5/6 spanning ties); report only.")
    note(t, "footnotes g/h tile medians (13,890 / 1,299) come from tile_counts_prism_*.csv, not bound; not checked.")
    note(t, "PRISM paired files are named paired.csv (not paired_differences.csv); the spec's 'paired 8265a556… / "
            "e95102666…' refers to them. PRISM-mosaic rows (spec Block A/B) are included although the task brief named "
            "only PRISM native.")
    for k_, p in [("comparators summary", R31C / "summary.csv"), ("comparators paired", R31C / "paired_differences.csv"),
                  ("retccl summary", R31R / "summary.csv"), ("retccl paired", R31R / "paired_differences.csv"),
                  ("native summary", R31N / "summary.csv"), ("native paired", R31N / "paired.csv"),
                  ("mosaic summary", R31M / "summary.csv"), ("mosaic paired", R31M / "paired.csv"),
                  ("comparators reconciliation", R31C / "reconciliation.json")]:
        h = INPUTS[str(p)][:8]
        cited = {"comparators summary": "461bbb22", "comparators paired": "8c53bf54", "retccl summary": "39086996",
                 "retccl paired": "7f2df7b2", "native summary": "3cc04fe1", "native paired": "8265a556",
                 "mosaic summary": "220b662b", "mosaic paired": "e9510266", "comparators reconciliation": "19ed4d64"}[k_]
        if h == cited:
            ok(t, f"source hash {k_} {h}… == spec-cited")
        else:
            issue(t, f"source hash {k_}", line, cited + "…", h + "…", "bound file differs from spec-cited hash")
    cap_start = find_line("**Caption:**", line)
    cap = [(ln, s) for ln, s in block if ln > cap_start and s.startswith(">")]
    foot = [(ln, s) for ln, s in block if s.startswith("**Footnotes:**") or
            (s.startswith("- ") or s.startswith("  ")) and ln > cap_start and
            ln < find_line("**Sources:**", line)]
    capdf = caption_sheet("Table S-R3.1", [("### R3.1-T2", line)], cap + foot,
                          [("Cell format", "point [95% CI]; Block B and Block C difference row signed (row minus "
                            "column-header configuration); bounds printed to 4–6 decimals where the spec does so "
                            "(footnotes c, i, k)")])
    fmt = pd.DataFrame(fmt_rows)
    write_table(t, "Table_S-R3.1.xlsx", capdf,
                [("Block_A", fmt[fmt.block == "A"]), ("Block_B", fmt[fmt.block == "B"]),
                 ("Block_C", fmt[fmt.block == "C"]), ("raw_long", raw)], f"### R3.1-T2 (l.{line})")


# ---------------------------------------------------------------- S25
S25_C2 = OUT_ROOT / "c2_metric_suite/metrics_uni2_ot_cosine.json"
S25_SO = OUT_ROOT / "r2_3_internal_failure_cis_20261004/summary_overall.csv"
S25_MS = OUT_ROOT / "r2_3_internal_failure_cis_20261004/macro_summary.csv"
S25_PD = OUT_ROOT / "r2_3_internal_failure_cis_20261004/paired_differences.csv"
S25_EXT = OUT_ROOT / "r2_3_internal_failure_cis_ext_prod_20261005/summary_ext.csv"
S25_F1 = OUT_ROOT / "review_scratch_astra_session_20261004/recomputed.json"

S25_ROWS = [  # (key, micro source, macro source, fig2 cross-check (k, metric) or None)
    ("hit@1", ("so", "top1_accuracy"), ("c2", "macro_hit_at_1"), (1, "hit_at_k")),
    ("hit@3", ("so", "hit3"), ("c2", "macro_hit_at_3"), (3, "hit_at_k")),
    ("hit@5", ("so", "hit5"), ("ms", "macro_hit5"), (5, "hit_at_k")),
    ("hit@10", ("so", "hit10"), ("ms", "macro_hit10"), (10, "hit_at_k")),
    ("precision@5", ("so", "precision5"), ("c2", "macro_precision_at_5"), (5, "precision_at_k")),
    ("precision@10", ("so", "precision10"), ("c2", "macro_precision_at_10"), (10, "precision_at_k")),
    ("recall@5", ("c2", "micro_recall_at_5"), ("c2", "macro_recall_at_5"), None),
    ("recall@10", ("c2", "micro_recall_at_10"), ("c2", "macro_recall_at_10"), None),
    ("MRR", ("so", "mrr10"), ("c2", "macro_mrr"), None),
    ("mAP@5", ("ext", "map5_micro"), ("ext", "map5_macro"), (5, "map_at_k")),
    ("mAP@10", ("ext", "map10_micro"), ("ext", "map10_macro"), (10, "map_at_k")),
    ("nDCG@5", ("ext", "ndcg5_micro"), ("ext", "ndcg5_macro"), None),
    ("nDCG@10", ("ext", "ndcg10_micro"), ("ext", "ndcg10_macro"), None),
    ("Plurality@5", ("so", "plurality_k5_accuracy"), ("ms", "macro_plurality_k5_recall"), None),
    ("Plurality@3", ("so", "plurality_k3_accuracy"), None, None),
    ("Plurality@10", ("so", "plurality_k10_accuracy"), None, None),
    ("Strict@5 accuracy", ("so", "strict_k5_accuracy"), None, None),
    ("Strict@5 abstain", ("so", "strict_k5_abstain_rate"), None, None),
    ("Published generous@5", ("so", "oracle_generous_k5"), None, None),
]
C2_MICRO = {"top1_accuracy": "micro_hit_at_1", "hit3": "micro_hit_at_3", "hit5": "micro_hit_at_5",
            "hit10": "micro_hit_at_10", "precision5": "micro_precision_at_5",
            "precision10": "micro_precision_at_10", "mrr10": "micro_mrr", "macro_hit5": "macro_hit_at_5",
            "macro_hit10": "macro_hit_at_10", "map5_micro": "micro_ap_at_5", "map5_macro": "macro_ap_at_5",
            "map10_micro": "micro_ap_at_10", "map10_macro": "macro_ap_at_10",
            "ndcg5_micro": "micro_ndcg_at_5", "ndcg5_macro": "macro_ndcg_at_5",
            "ndcg10_micro": "micro_ndcg_at_10", "ndcg10_macro": "macro_ndcg_at_10"}


def build_s25():
    t = "S25"
    line, block = spec_block("R4.b-T5")
    c2 = rjson(S25_C2)["point_estimates"]
    so = rcsv(S25_SO).set_index("estimand")
    ms = rcsv(S25_MS).set_index("estimand")
    pdf = rcsv(S25_PD).set_index("estimand")
    ext = rcsv(S25_EXT).set_index("metric")
    f1 = rjson(S25_F1)["A"]["f1"]
    fig2 = rcsv(FIG2SUM)
    fig2 = fig2[fig2.cell_id == "uni2/ot_cosine"]

    def get(src):
        if src is None:
            return (np.nan, np.nan, np.nan, "")
        kind, key = src
        if kind == "c2":
            return (c2[key], np.nan, np.nan, f"C2 JSON point_estimates.{key}")
        if kind == "so":
            r = so.loc[key]
            assert r.n_queries == 2778 and r.n_patients == 1876
            return (r.point, r.ci_lo, r.ci_hi, f"summary_overall.csv {key}")
        if kind == "ms":
            r = ms.loc[key]
            assert r.n_groups == 15
            return (r.point, r.ci_lo, r.ci_hi, f"macro_summary.csv {key}")
        r = ext.loc[key]
        assert r.n_queries == 2778 and r.n_patients == 1876 and r.n_replicates == 5000 and r.n_nonfinite == 0
        return (r.point, r.lo, r.hi, f"summary_ext.csv {key}")

    tabs = md_tables_in(block)
    assert len(tabs) == 4, len(tabs)
    hdr, ci_rows, ci_lns = md_table(tabs[0])
    _, pd_rows, pd_lns = md_table(tabs[1])
    _, f1_rows, f1_lns = md_table(tabs[2])
    _, pt_rows, pt_lns = md_table(tabs[3])
    assert len(ci_rows) == len(S25_ROWS), (len(ci_rows), len(S25_ROWS))
    out = []
    for (key, mi, ma, fx), cells, ln in zip(S25_ROWS, ci_rows, ci_lns):
        assert cells[0].lower().startswith(key.lower().split()[0][:6]), (key, cells[0])
        mp, ml, mh, msrc = get(mi)
        ap, al, ah, asrc = get(ma)
        mcell = fci(mp, ml, mh) if not np.isnan(ml) else f(mp) + " (point estimate only)"
        acell = (fci(ap, al, ah) if not np.isnan(al) else f(ap) + " (point estimate only)") if ma else "—"
        for side, spec_cell, vals in (("micro", cells[1], (mp, ml, mh)), ("macro", cells[2], (ap, al, ah))):
            v = [x for x in vals if not np.isnan(x)]
            if nums(spec_cell):
                check_cell(t, f"{key} {side} (CI companion)", ln, spec_cell, v)
            elif v and not (side == "macro" and ma and ma[0] == "c2"):
                issue(t, f"{key} {side}", ln, spec_cell, f(v[0], 6), "source has a value; spec cell empty")
        if fx:
            for side, vals in (("micro", (mp, ml, mh)), ("macro", (ap, al, ah))):
                r = fig2[(fig2.k == fx[0]) & (fig2.metric == fx[1]) & (fig2.average == side)]
                if len(r) == 1:
                    r = r.iloc[0]
                    d = abs(r.point - vals[0])
                    if d < 5e-7 and (np.isnan(vals[1]) or (abs(r.ci_lo - vals[1]) < 5e-7 and abs(r.ci_hi - vals[2]) < 5e-7)):
                        ok(t, f"cross-check {key} {side} == Fig. 2 derivation summary.csv uni2/ot_cosine (≤5e-7)")
                    else:
                        issue(t, f"cross-check {key} {side} vs Fig. 2 summary", "", fci(*vals, d=6),
                              fci(r.point, r.ci_lo, r.ci_hi, d=6), "S25 spec sources vs Fig. 2 derivation differ")
        out.append({"Metric": cells[0], "Micro (slide queries) [95% CI]": mcell,
                    "Macro (15 labels) [95% CI]": acell,
                    "micro_point": mp, "micro_ci_lo": ml, "micro_ci_hi": mh, "micro_source": msrc,
                    "macro_point": ap, "macro_ci_lo": al, "macro_ci_hi": ah, "macro_source": asrc})
    main = pd.DataFrame(out)
    note(t, "Fig. 2 derivation summary.csv (r2_1_fig2_retrieval_20261021/run_001) also carries whole-patient "
            "intervals for macro hit@1/@3 and macro precision@5/@10, which the S25 spec keeps point-only; "
            "not added (spec followed). Built from the spec-named sources, not from the Fig. 2 summary.")
    # base point table: C2 JSON
    c2map = {"hit@1": ("micro_hit_at_1", "macro_hit_at_1"), "hit@3": ("micro_hit_at_3", "macro_hit_at_3"),
             "hit@5": ("micro_hit_at_5", "macro_hit_at_5"), "hit@10": ("micro_hit_at_10", "macro_hit_at_10"),
             "precision@5": ("micro_precision_at_5", "macro_precision_at_5"),
             "precision@10": ("micro_precision_at_10", "macro_precision_at_10"),
             "recall@5": ("micro_recall_at_5", "macro_recall_at_5"),
             "recall@10": ("micro_recall_at_10", "macro_recall_at_10"),
             "mAP@5": ("micro_ap_at_5", "macro_ap_at_5"), "mAP@10": ("micro_ap_at_10", "macro_ap_at_10"),
             "nDCG@5": ("micro_ndcg_at_5", "macro_ndcg_at_5"), "nDCG@10": ("micro_ndcg_at_10", "macro_ndcg_at_10"),
             "MRR (depth 10)": ("micro_mrr", "macro_mrr")}
    base = []
    for cells, ln in zip(pt_rows, pt_lns):
        k1, k2 = c2map[cells[0].split(" (")[0] if cells[0] != "MRR (depth 10)" else cells[0]]
        check_cell(t, f"base {cells[0]} micro (C2)", ln, cells[1], [c2[k1]])
        check_cell(t, f"base {cells[0]} macro (C2)", ln, cells[2], [c2[k2]])
        base.append({"Metric": cells[0], "Micro (slide queries)": f(c2[k1]), "Macro (15 labels)": f(c2[k2]),
                     "micro_point_C2": c2[k1], "macro_point_C2": c2[k2], "C2 keys": f"{k1}; {k2}"})
    base = pd.DataFrame(base)
    assert len(base) == 13
    # C2 vs CI-file agreement at 3 dp
    for (kind, key) in [s for r in S25_ROWS for s in (r[1], r[2]) if s and s[0] in ("so", "ms", "ext")]:
        if key in C2_MICRO:
            v = get((kind, key))[0]
            c = c2[C2_MICRO[key]]
            if round(v, 3) == round(c, 3):
                ok(t, f"CI-file point {key} agrees with C2 {C2_MICRO[key]} at 3 dp")
            else:
                issue(t, f"C2 vs CI-file {key}", "", f(c, 6), f(v, 6), "point estimates disagree at 3 dp")
    # paired differences
    pmap = ["oracle_minus_plurality_k5", "oracle_minus_strict_k5", "hit5_minus_plurality_k5"]
    prs = []
    for key, cells, ln in zip(pmap, pd_rows, pd_lns):
        r = pdf.loc[key]
        check_cell(t, f"S25b {cells[0]}", ln, cells[1], [r.point, r.ci_lo, r.ci_hi])
        prs.append({"Paired difference (same resampled patients)": cells[0],
                    "Estimate [95% CI]": fci(r.point, r.ci_lo, r.ci_hi, signed=True),
                    "point": r.point, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                    "source": f"paired_differences.csv {key}"})
    # F1
    fmap = [("generous_macro_f1", "oracle"), ("plurality_macro_f1", "plurality"),
            ("strict_macro_f1", "strict_15_labels")]
    frs = []
    for (ek, rk), cells, ln in zip(fmap, f1_rows, f1_lns):
        r = ext.loc[ek]
        g = f1[rk]
        check_cell(t, f"F1 {cells[0]} macro", ln, cells[1], [r.point, r.lo, r.hi])
        if rk == "strict_15_labels":
            mic = f"{f(g['micro_f1'], 4)} (P {f(g['covered_accuracy'], 4)}, R {f(g['accuracy'], 4)})"
            check_cell(t, f"F1 {cells[0]} micro/P/R", ln, cells[2],
                       [g["micro_f1"], g["covered_accuracy"], g["accuracy"]])
        else:
            mic = f(g["micro_f1"])
            check_cell(t, f"F1 {cells[0]} micro", ln, cells[2], [g["micro_f1"]])
        if abs(r.point - g["macro_f1"]) < 5e-7:
            ok(t, f"F1 {rk}: summary_ext macro F1 point == recomputed.json macro_f1")
        else:
            issue(t, f"F1 {rk} macro point", "", f(g["macro_f1"], 6), f(r.point, 6), "summary_ext vs recomputed.json")
        frs.append({"Rule": cells[0], "Macro F1 (15 labels)": fci(r.point, r.lo, r.hi), "Micro F1": mic,
                    "Note": cells[3], "macro_point": r.point, "macro_ci_lo": r.lo, "macro_ci_hi": r.hi,
                    "micro_f1": g["micro_f1"], "micro_precision_covered_accuracy": g["covered_accuracy"],
                    "micro_recall_accuracy": g["accuracy"],
                    "source": f"summary_ext.csv {ek}; recomputed.json A.f1.{rk}"})
    for p_, h_ in [(S25_C2, "895c8dc2"), (S25_SO, "9e701500"), (S25_MS, "e9d126fd"), (S25_PD, "7f066c2a"),
                   (S25_EXT, "65d86d17"), (S25_F1, "c3353df6")]:
        cited(t, p_, h_, f"(R4.b-T5 l.{line})")
    # strict note numbers (abstention 0.102) are in the abstain row; checked via the companion
    leg_start = find_line("**Legend:**", line)
    leg = [(ln, s) for ln, s in block if ln > leg_start and s.startswith(">")]
    note(t, "Legend recall sentence numbers (median 264; range 27–470; maxima 0.028/0.055, macro 0.055/0.110) "
            "are not in the bound sources; not checked.")
    capdf = caption_sheet("Supplementary Table S25 (S-A3)", [("### R4.b-T5", line),
                          ("### R4.b-T9 (Methods only; no table change)", find_line("### R4.b-T9"))], leg,
                          [("Layout", "Sheet S25: CI companion merged with the base point table (spec: 'replace "
                            "the matching rows' cells; append the vote-rule rows; recall stays point-only'); "
                            "S25_points_C2: the base point table; S25b: paired differences; S25_F1: F1 at k = 5.")])
    write_table(t, "Table_S25.xlsx", capdf, [("S25", main), ("S25_points_C2", base), ("S25b_paired", pd.DataFrame(prs)),
                                             ("S25_F1_k5", pd.DataFrame(frs))], f"### R4.b-T5 (l.{line})")


# ---------------------------------------------------------------- S-R2.1-A
MG = {"h_optimus_1": OUT_ROOT / "r2_1_cpu_20261002/main_attempt_001",
      "uni2": OUT_ROOT / "r2_1_main_grid_comparators_20261011/main_uni2_attempt_001",
      "virchow2": OUT_ROOT / "r2_1_main_grid_comparators_20261011/main_virchow2_attempt_001"}
AGG_ORDER = ["median_pooling", "sampler", "ot_cosine", "yottixel"]
AGG_LABEL = {"median_pooling": "Median Pooling", "sampler": "SAMPLER", "ot_cosine": "Optimal Transport",
             "yottixel": "Yottixel"}


def build_a():
    t = "S-R2.1-A"
    line, block = spec_block("R2.1-T4")
    hp = rcsv(RR5 / "hit_precision.csv")
    tc = rcsv(RR5 / "tie_census.csv")
    encs = [("h_optimus_1", "h_optimus_new"), ("uni2", "primary"), ("virchow2_fp32", "virchow2_fp32_replacement")]
    rows = []
    for enc, lin in encs:
        for agg in AGG_ORDER:
            cid = f"{enc}/{agg}"
            tie = tc[tc.cell_id == cid]
            assert len(tie) == 1, cid
            tie = tie.iloc[0]
            assert tie.lineage == lin, (cid, tie.lineage)
            for k in (1, 3, 5, 10):
                g = hp[(hp.cell_id == cid) & (hp.k == k) & (hp.tie_order == "canonical")]
                h = g[g.metric == "hit_macro"]
                p = g[g.metric == "precision_macro"]
                assert len(h) == 1 and len(p) == 1, (cid, k)
                h, p = h.iloc[0], p.iloc[0]
                assert h.lineage == lin and p.lineage == lin
                assert h.ci_status == "ok" and p.ci_status == "ok"
                assert int(h.n_order_sensitive_k) == int(tie[f"order_sensitive_{k}"])
                rows.append({"Encoder": h.display_encoder, "Aggregation": AGG_LABEL[agg], "k": k,
                             "macro hit@k [95% CI]": fci(h.value, h.ci_low, h.ci_high),
                             "macro precision@k [95% CI]": fci(p.value, p.ci_low, p.ci_high),
                             "k-boundary tied queries": int(tie[f"boundary_tie_{k}"]),
                             "order-sensitive queries at k": int(tie[f"order_sensitive_{k}"]),
                             "cell_id": cid, "lineage": lin, "matrix_sha256": h.matrix_sha256,
                             "hit_macro": h.value, "hit_ci_low": h.ci_low, "hit_ci_high": h.ci_high,
                             "precision_macro": p.value, "precision_ci_low": p.ci_low,
                             "precision_ci_high": p.ci_high, "ci_status": h.ci_status,
                             "n_zero_support": h.n_zero_support})
    cells = pd.DataFrame(rows)
    assert len(cells) == 48
    ok(t, "48 rows = 3 encoders × 4 aggregations × k 1/3/5/10; all intervals present (ci_status ok); "
          "n_order_sensitive_k in hit_precision == tie_census order_sensitive_k")
    ok(t, "Virchow2 cells from rr5_replacement_001/hit_precision.csv lineage virchow2_fp32_replacement (10-16 amendment)")
    # cross-check against main-grid estimates (published bank for Virchow2)
    maxd = {}
    for enc, src in MG.items():
        e = rcsv(src / "estimates.csv")
        e = e[(e.row_kind == "estimate") & (e.level == "macro")]
        renc = "virchow2_fp32" if enc == "virchow2" else enc
        d = 0.0
        for _, r in e.iterrows():
            c = cells[(cells.cell_id == f"{renc}/{r.method}") & (cells.k == r.k)].iloc[0]
            col = "hit_macro" if r.endpoint == "hit" else "precision_macro"
            d = max(d, abs(c[col] - r.point))
        maxd[enc] = d
        assert len(e) == 32, (enc, len(e))
    for enc in ("h_optimus_1", "uni2"):
        if maxd[enc] < 5e-7:
            ok(t, f"{enc}: RR-5 values == main-grid estimates.csv macro rows (max |Δ| {maxd[enc]:.1e})")
        else:
            issue(t, f"{enc} RR-5 vs main grid", "", "", f"max |Δ| {maxd[enc]:.2e}", "RR-5 vs main-grid differ")
    note(t, f"Virchow2 fp32 (RR-5) vs published-bank main grid estimates.csv: max |Δ| {maxd['virchow2']:.4f} "
            "(expected: different feature bank; submitted/published rows in S-R2.1-F).")
    # population/policy footnote
    for enc, src in MG.items():
        s = rjson(src / "summary.json")
        c, b = s["counts"], s["policy"]["bootstrap"]
        assert (c["target_slides"], c["patients"], c["groups"]) == (2778, 1876, 15), (enc, c)
        assert (b["count"], b["seed"], b["generator"]) == (5000, 123, "PCG64"), (enc, b)
    ok(t, "footnote population 2,778 / 1,876 / 15 and 5,000 draws (seed 123, PCG64) in all three main-grid summary.json")
    yt = tc[tc.cell_id.isin([f"{e}/yottixel" for e, _ in encs])][["cell_id", "boundary_tie_10"]]
    note(t, "Yottixel boundary_tie_10: " + ", ".join(f"{a} {b}" for a, b in yt.values) +
         " (Virchow2 fp32 737; published-bank value was 736).")
    # contrasts
    crs = []
    for enc, src in MG.items():
        c = rcsv(src / "contrasts.csv")
        c = c[(c.row_kind == "contrast") & (c.level == "macro")]
        assert len(c) == 12, (enc, len(c))
        for _, r in c.iterrows():
            assert r.n_queries == 2778 and r.n_patients == 1876
            crs.append({"Encoder": {"h_optimus_1": "H-Optimus-1", "uni2": "UNIv2", "virchow2": "Virchow2"}[enc],
                        "Contrast (row − column)": f"{AGG_LABEL[r.arm_a]} − {AGG_LABEL[r.arm_b]}",
                        "Endpoint": f"macro {r.endpoint}@{r.k}",
                        "Difference [95% CI]": fci(r.point, r.ci_low, r.ci_high, signed=True),
                        "lineage status": ("[SPEC/SOURCE MISMATCH] published-bank (mixed-precision) Virchow2 lineage; "
                                           "no fp32 contrast source exists" if enc == "virchow2" else "as specified"),
                        "source": str(src.relative_to(OUT_ROOT)) + "/contrasts.csv",
                        **{k: r[k] for k in ["arm_a", "arm_b", "endpoint", "k", "level", "point", "point_status",
                                             "ci_low", "ci_high", "ci_status", "n_queries", "n_patients",
                                             "zero_support_replicates"]}})
    con = pd.DataFrame(crs)
    ok(t, "contrasts: 12 macro rows (6 pre-specified pairs × hit/precision at k = 5) per encoder, n 2778/1876")
    issue(t, "Virchow2 contrast rows", line, "Virchow2 values from rr5_replacement_001 (fp32), amendment",
          "contrasts exist only in main_virchow2_attempt_001 (published bank)",
          "[SPEC/SOURCE MISMATCH] Virchow2 contrast rows included from the published-bank lineage and flagged; "
          "RR-5 carries no within-encoder contrasts for virchow2_fp32 (rr5 aggregation_contrasts.csv not "
          "bound because the spec names 'pre-specified within-encoder contrasts' from the main-grid package)")
    issue(t, "row count wording", line, "'all 8 aggregation cells × 3 encoders'", "4 aggregations per encoder",
          "spec wording: there are 4 aggregation methods (× 4 k = 16 rows per encoder); built 4")
    note(t, "Per-group contrast rows (90 × 2 per encoder in contrasts.csv) are not included; spec names the 12 "
            "within-encoder contrast rows.")
    note(t, "Caption: the spec gives a content description (footnote items) and the S3/S1/S2 caption sentences, "
            "not a drafted S-R2.1-A caption; the caption sheet carries the spec block verbatim. Caption prose owed.")
    capdf = caption_sheet("Table S-R2.1-A", [("### R2.1-T4", line)], block[1:],
                          [("Footnote (10-16 amendment, verbatim)",
                            "Virchow2 rows recomputed from the regenerated fp32 bank; submitted rows in Table S-R2.1-F.")])
    write_table(t, "Table_S-R2.1-A.xlsx", capdf, [("cells", cells), ("contrasts_k5", con)],
                f"### R2.1-T4 (l.{line}) + 10-16 amendment")


# ---------------------------------------------------------------- S-R2.1-E sheets 1-4
def _ci_or_nr(v, lo, hi, status, d=3, signed=False):
    if status == "not_reportable":
        return f"{f(v, d, signed)} [n.r.]"
    assert status == "ok" and np.isfinite(lo) and np.isfinite(hi), (v, lo, hi, status)
    return fci(v, lo, hi, d, signed)


def build_e():
    t = "S-R2.1-E"
    line, block = spec_block("R2.1-T12")
    s3 = rcsv(RR5 / "table_s3_revised.csv")
    rc = rcsv(RR5 / "rule_contrasts.csv")
    hp = rcsv(RR5 / "hit_precision.csv")
    pc = rcsv(RR5 / "per_class.csv")
    tc = rcsv(RR5 / "titan_contrast.csv")
    wn = rcsv(RR5 / "winners.csv")
    ts = rcsv(RR5 / "tie_sensitivity.csv")
    f2c = rcsv(FIG2C)
    rr5c = rcsv(RR5 / "fig2c_groups.csv")
    cells = list(dict.fromkeys(s3.cell_id))
    assert len(cells) == 38, len(cells)
    ok(t, "38 cells in table_s3_revised.csv (34 original + 4 virchow2_fp32; no published-bank virchow2 cell)")
    assert not any(c.startswith("virchow2/") for c in cells)
    can = s3[s3.tie_order == "canonical"]

    def g1(df, **kw):
        m = np.ones(len(df), bool)
        for k_, v in kw.items():
            m &= (df[k_] == v).values
        r = df[m]
        assert len(r) == 1, (kw, len(r))
        return r.iloc[0]

    rows = []
    for cid in cells:
        for k in (1, 3, 5, 10):
            pm = g1(can, cell_id=cid, k=k, rule="plurality", metric="macro_f1")
            pu = g1(can, cell_id=cid, k=k, rule="plurality", metric="micro_f1")
            om = g1(can, cell_id=cid, k=k, rule="oracle", metric="macro_f1")
            ou = g1(can, cell_id=cid, k=k, rule="oracle", metric="micro_f1")
            pr = g1(can, cell_id=cid, k=k, rule="plurality", metric="macro_recall")
            re_ = rc[(rc.cell_id == cid) & (rc.k == k) & (rc.metric == "macro_f1") & rc.group.isna()]
            assert len(re_) == 1
            re_ = re_.iloc[0]
            hh = {m: g1(hp, cell_id=cid, k=k, tie_order="canonical", metric=m)
                  for m in ("hit_micro", "hit_macro", "precision_micro")}
            sub, subsrc = om.submitted_value, "submitted_value"
            if pd.isna(sub) and pd.notna(om.submitted_value_from_published_matrix):
                sub, subsrc = om.submitted_value_from_published_matrix, "submitted_value_from_published_matrix"
            nos = {int(x) for x in [pm.n_order_sensitive_k, om.n_order_sensitive_k, hh["hit_macro"].n_order_sensitive_k]}
            assert len(nos) == 1, (cid, k, nos)
            nr = [n for n, r in [("plurality macro F1", pm), ("plurality micro F1", pu), ("generous macro F1", om),
                                 ("generous micro F1", ou), ("rule effect", re_), ("plurality macro recall", pr)]
                  + [(m, r) for m, r in hh.items()] if r.ci_status == "not_reportable"]
            rows.append({
                "Encoder": pm.display_encoder, "Aggregation": pm.aggregation, "k": k,
                "plurality macro F1 [95% CI]": _ci_or_nr(pm.value, pm.ci_low, pm.ci_high, pm.ci_status),
                "plurality micro F1 [95% CI]": _ci_or_nr(pu.value, pu.ci_low, pu.ci_high, pu.ci_status),
                "generous-rule macro F1 [95% CI]": _ci_or_nr(om.value, om.ci_low, om.ci_high, om.ci_status),
                "generous-rule micro F1 [95% CI]": _ci_or_nr(ou.value, ou.ci_low, ou.ci_high, ou.ci_status),
                "submitted Table S3 macro F1": f(sub) if pd.notna(sub) else "",
                "rule effect (plurality − generous macro F1) [95% CI]": _ci_or_nr(re_.value, re_.ci_low, re_.ci_high,
                                                                                   re_.ci_status, signed=True),
                "hit@k micro [95% CI]": _ci_or_nr(hh["hit_micro"].value, hh["hit_micro"].ci_low, hh["hit_micro"].ci_high, hh["hit_micro"].ci_status),
                "hit@k macro [95% CI]": _ci_or_nr(hh["hit_macro"].value, hh["hit_macro"].ci_low, hh["hit_macro"].ci_high, hh["hit_macro"].ci_status),
                "precision@k micro [95% CI]": _ci_or_nr(hh["precision_micro"].value, hh["precision_micro"].ci_low, hh["precision_micro"].ci_high, hh["precision_micro"].ci_status),
                "queries with tie-order-dependent top-k": nos.pop(),
                "CI status": "ok" if not nr else "n.r.: " + "; ".join(nr),
                "plurality macro recall (mMV@k) [95% CI]": _ci_or_nr(pr.value, pr.ci_low, pr.ci_high, pr.ci_status),
                "cell_id": cid, "lineage": pm.lineage, "matrix_sha256": pm.matrix_sha256,
                "plurality_macro_f1": pm.value, "plurality_macro_f1_ci_low": pm.ci_low, "plurality_macro_f1_ci_high": pm.ci_high, "plurality_macro_f1_ci_status": pm.ci_status,
                "plurality_micro_f1": pu.value, "plurality_micro_f1_ci_low": pu.ci_low, "plurality_micro_f1_ci_high": pu.ci_high,
                "oracle_macro_f1": om.value, "oracle_macro_f1_ci_low": om.ci_low, "oracle_macro_f1_ci_high": om.ci_high, "oracle_macro_f1_ci_status": om.ci_status,
                "oracle_micro_f1": ou.value, "oracle_micro_f1_ci_low": ou.ci_low, "oracle_micro_f1_ci_high": ou.ci_high,
                "submitted_macro_f1": sub, "submitted_macro_f1_source_column": subsrc if pd.notna(sub) else "",
                "submitted_source": om.submitted_source if pd.notna(om.submitted_source) else "",
                "rule_effect": re_.value, "rule_effect_ci_low": re_.ci_low, "rule_effect_ci_high": re_.ci_high, "rule_effect_ci_status": re_.ci_status,
                "hit_micro": hh["hit_micro"].value, "hit_micro_ci_low": hh["hit_micro"].ci_low, "hit_micro_ci_high": hh["hit_micro"].ci_high,
                "hit_macro": hh["hit_macro"].value, "hit_macro_ci_low": hh["hit_macro"].ci_low, "hit_macro_ci_high": hh["hit_macro"].ci_high,
                "precision_micro": hh["precision_micro"].value, "precision_micro_ci_low": hh["precision_micro"].ci_low, "precision_micro_ci_high": hh["precision_micro"].ci_high,
                "plurality_macro_recall": pr.value, "plurality_macro_recall_ci_low": pr.ci_low, "plurality_macro_recall_ci_high": pr.ci_high,
                "plurality_macro_recall_ci_status": pr.ci_status,
                "reversed_oracle_macro_f1": g1(s3, cell_id=cid, k=k, rule="oracle", metric="macro_f1", tie_order="reversed").value,
            })
    s1 = pd.DataFrame(rows)
    assert len(s1) == 152
    ok(t, "Sheet 1: 152 rows = 38 cells × k 1/3/5/10; canonical tie order")
    assert (s1.submitted_macro_f1.isna() == s1.cell_id.str.startswith("h_optimus_1/")).all()
    ok(t, "submitted Table S3 macro F1 blank exactly for the 4 H-Optimus-1 cells; TITAN-regenerated row uses "
          "submitted_value_from_published_matrix")
    note(t, "Sheet 1 also carries 'plurality macro recall (mMV@k)' (table_s3_revised.csv metric macro_recall): the "
            "R2.1-T12 column list omits it, but the 10-21 addendum (edit sheet l."
            f"{find_line('same numbers as Table S-R2.1-E sheet 1')}) states that sheet 1 has a macro_recall column "
            "equal to the sheet-5/6 mMV@5 values. Builder inclusion; drop if not wanted.")
    _e_footnotes(t, line, block, s1, rc, hp, wn)
    _e_rest(t, line, block, s1, s3, pc, tc, wn, ts, f2c, rr5c)


def _e_footnotes(t, line, block, s1, rc, hp, wn):
    fl = {s.split(".")[0].strip(): ln for ln, s in block if re.match(r"^\d\. ", s)}
    # footnote 2: rule-effect range at k = 5 and k = 3 main configuration
    r5 = rc[(rc.k == 5) & (rc.metric == "macro_f1") & rc.group.isna() & (rc.tie_order == "canonical")]
    assert len(r5) == 38
    lo, hi = r5.loc[r5.value.idxmin()], r5.loc[r5.value.idxmax()]
    for name, r, cid, txt in [("min", lo, "simclr/median_pooling", "−0.047"), ("max", hi, "h_optimus_1/ot_cosine", "−0.016")]:
        if r.cell_id == cid and same_at(float(txt.replace(MINUS, "-")), 3, r.value):
            ok(t, f"footnote 2: k = 5 rule effect {name} {r.cell_id} {r.value:.6f} → {txt}")
        else:
            issue(t, f"footnote 2 rule-effect {name}", fl["2"], f"{txt} ({cid})", f"{f(r.value, 6)} ({r.cell_id})")
    r3 = rc[(rc.k == 3) & (rc.metric == "macro_f1") & rc.group.isna() & (rc.cell_id == "uni2/ot_cosine")].iloc[0]
    check_cell(t, "footnote 2 k = 3 main configuration rule effect", fl["2"], "−0.048 [−0.063, −0.035]",
               [r3.value, r3.ci_low, r3.ci_high])
    # footnote 4: n.r. counts over the 304 macro-F1 values
    mf = pd.concat([s1[["cell_id", "k", "plurality_macro_f1_ci_status"]].rename(columns={"plurality_macro_f1_ci_status": "st"}),
                    s1[["cell_id", "k", "oracle_macro_f1_ci_status"]].rename(columns={"oracle_macro_f1_ci_status": "st"})])
    assert len(mf) == 304
    nr = mf[mf.st == "not_reportable"]
    got = (len(nr), int((nr.k == 10).sum()), int((nr.k == 5).sum()), int((nr.cell_id != "titan/titan_published").sum()))
    if got == (71, 61, 10, 67):
        ok(t, "footnote 4: n.r. 71 of 304 macro-F1 values (61 at k = 10, 10 at k = 5; 67 excl. TITAN submitted)")
    else:
        issue(t, "footnote 4 n.r. counts", fl["4"], "71 / 61 at k=10 / 10 at k=5 / 67 excl. TITAN-submitted",
              f"{got[0]} / {got[1]} / {got[2]} / {got[3]}", "[SPEC/SOURCE MISMATCH] counts in run_003 sheet 1")
    mc = s1[(s1.cell_id == "uni2/ot_cosine") & (s1.k == 10)].iloc[0]
    if mc.plurality_macro_f1_ci_status == "not_reportable" and same_at(0.746, 3, mc.plurality_macro_f1):
        ok(t, "footnote 4: main configuration plurality macro F1 at k = 10 = 0.746, n.r.")
    else:
        issue(t, "footnote 4 main config k=10", fl["4"], "0.746 n.r.", f"{mc.plurality_macro_f1:.6f} {mc.plurality_macro_f1_ci_status}")
    re_nr = s1.rule_effect_ci_status == "not_reportable"
    either = (s1.plurality_macro_f1_ci_status == "not_reportable") | (s1.oracle_macro_f1_ci_status == "not_reportable")
    note(t, f"footnote 4: rule-effect intervals n.r. in {int(re_nr.sum())} rows; rows with either macro F1 n.r.: "
            f"{int(either.sum())}; rule-effect n.r. ⊆ either-n.r.: {bool((~re_nr | either).all())}.")
    hpc = hp[hp.cell_id.isin(s1.cell_id.unique())]
    assert (hpc.ci_status == "ok").all()
    ok(t, "footnote 4: no hit@k or precision@k interval n.r. (hit_precision.csv, all 38 cells, both tie orders)")
    # footnote 3: reproduction of submitted Table S3 values (submitted cells = 33: excl. H-Optimus-1, TITAN-regenerated)
    sub = s1[~s1.cell_id.str.startswith("h_optimus_1/") & (s1.cell_id != "titan/titan_repaired")].copy()
    assert len(sub) == 132
    sub["d"] = (sub.oracle_macro_f1 - sub.submitted_macro_f1).abs()
    tie = sub["queries with tie-order-dependent top-k"] > 0
    exact = sub.d < 5e-7
    lo_ = np.minimum(sub.oracle_macro_f1, sub.reversed_oracle_macro_f1)
    hi_ = np.maximum(sub.oracle_macro_f1, sub.reversed_oracle_macro_f1)
    betw = tie & (sub.submitted_macro_f1 >= lo_ - 5e-7) & (sub.submitted_macro_f1 <= hi_ + 5e-7)
    nt_yot = int((tie & sub.cell_id.str.endswith("/yottixel")).sum())
    nt_tit = int((tie & (sub.cell_id == "titan/titan_published")).sum())
    got = dict(no_tie=int((~tie).sum()), no_tie_exact=int((~tie & exact).sum()), tie=int(tie.sum()),
               tie_yottixel=nt_yot, tie_titan=nt_tit, max_tie_diff=float(sub.d[tie].max()), between=int(betw.sum()))
    v2 = sub.cell_id.str.startswith("virchow2_fp32/")
    got_v2 = dict(no_tie_non_v2_exact=int((~tie & exact & ~v2).sum()), no_tie_non_v2=int((~tie & ~v2).sum()),
                  v2_no_tie_exact=int((~tie & exact & v2).sum()), v2_no_tie=int((~tie & v2).sum()),
                  v2_max_diff=float(sub.d[v2].max()))
    spec_ok = (got["no_tie"] == 96 and got["no_tie_exact"] == 96 and got["tie"] == 36 and nt_yot == 32
               and nt_tit == 4 and round(got["max_tie_diff"], 3) <= 0.025 and got["between"] == 23)
    if spec_ok:
        ok(t, f"footnote 3 reproduction counts {got}")
    else:
        issue(t, "footnote 3 reproduction counts", fl["3"],
              "96 no-tie rows exact; 36 tie rows (32 Yottixel, 4 TITAN-submitted) ≤ 0.025; 23 of 36 between",
              str({k_: (round(v, 6) if isinstance(v, float) else v) for k_, v in got.items()}) + "; " +
              str({k_: (round(v, 6) if isinstance(v, float) else v) for k_, v in got_v2.items()}),
              "[SPEC/SOURCE MISMATCH] footnote written on run_001 (published Virchow2 bank); with virchow2_fp32 "
              "cells the submitted values are no longer reproduced exactly for Virchow2 rows (counts by builder "
              "comparison of table_s3_revised columns; selection/counting only)")
    # winner counts under the reversed order
    cnt = wn[(wn.row_type == "count") & (wn.rule == "plurality") & (wn.tie_order == "reversed")]
    c32 = cnt[cnt.slot_set == "C32"].set_index("winner").weight.to_dict()
    h36 = cnt[cnt.slot_set == "H36"].set_index("winner").weight.to_dict()
    got = (c32.get("Yottixel"), c32.get("Optimal Transport"), h36.get("Yottixel"), h36.get("Optimal Transport"))
    if got == (23, 9, 24, 12):
        ok(t, "footnote 3 reversed-order winner counts 23/9 and 24/12")
    else:
        issue(t, "footnote 3 reversed-order winner counts (plurality)", fl["3"], "C32 Yottixel 23 / OT 9; H36 24 / 12",
              f"C32 Yottixel {got[0]:.0f} / OT {got[1]:.0f}; H36 {got[2]:.0f} / {got[3]:.0f} (winners.csv count rows)",
              "[SPEC/SOURCE MISMATCH] run_003 winners.csv (with virchow2_fp32) differs from the footnote")
    note(t, "Footnote 1 population (2,778 / 1,876 / 15) is not restated in the RR-5 files bound here; the same "
            "draws as S-R2.1-A, whose main-grid summary.json counts are asserted there.")
    note(t, "Footnote 5 (fp16 autocast for H-Optimus-1) is provenance text; not checkable from bound files.")


def _e_rest(t, line, block, s1, s3, pc, tc, wn, ts, f2c, rr5c):
    # Sheet 2
    p = pc[(pc.metric == "f1") & (pc.rule == "plurality") & (pc.tie_order == "canonical") & (pc.k == 5)].copy()
    assert len(p) == 38 * 15, len(p)
    cur = set(f2c.cell_id)
    rr5set = set(rr5c.pipeline)
    s2 = pd.DataFrame({
        "Encoder": p.display_encoder, "Aggregation": p.aggregation, "group": p.group,
        "plurality F1 at k = 5 [95% CI]": [_ci_or_nr(a, b, c, d) for a, b, c, d in zip(p.value, p.ci_low, p.ci_high, p.ci_status)],
        "current Fig. 2C pipeline (figures_20261021)": p.cell_id.isin(cur).map({True: "yes", False: ""}),
        "RR-5 Fig. 2C pipeline (fig2c_groups.csv)": p.cell_id.isin(rr5set).map({True: "yes", False: ""}),
        "cell_id": p.cell_id, "lineage": p.lineage, "value": p.value, "ci_low": p.ci_low, "ci_high": p.ci_high,
        "ci_status": p.ci_status, "n_zero_pred": p.n_zero_pred, "n_zero_support": p.n_zero_support})
    ok(t, f"Sheet 2: {len(s2)} rows = 38 cells × 15 groups (per_class.csv f1, plurality, canonical, k = 5)")
    issue(t, "Sheet 2 Fig. 2C flag", line, "'with the Fig. 2C pipelines flagged'",
          f"current Fig. 2C (figures_20261021 source_data_panelC.csv): {sorted(cur)}; RR-5 fig2c_groups.csv: {sorted(rr5set)}",
          "pipeline sets differ (current figure adds H-Optimus-1 + Yottixel per 10-21 decision (ii) and now shows "
          "hit@5, not F1); both flags provided, author to choose")
    # Sheet 3
    s3t = tc.copy()
    s3t.insert(0, "regenerated − submitted embeddings [95% CI]",
               [_ci_or_nr(a, b, c, d, signed=True) for a, b, c, d in zip(tc.value, tc.ci_low, tc.ci_high, tc.ci_status)])
    s3t.insert(0, "submitted embeddings", [f(x) for x in tc.published_value])
    s3t.insert(0, "regenerated embeddings", [f(x) for x in tc.repaired_value])
    assert len(s3t) == 48
    ok(t, "Sheet 3: titan_contrast.csv 48 rows (k × rule/metric × published tie order) copied with formatted columns")
    # Sheet 4
    w = wn.copy()
    w.insert(0, "winner value [95% CI]", [("" if pd.isna(a) else (fci(a, b, c) if not isinstance(d, str) or
                                          (d == "ok" and pd.isna(b)) else _ci_or_nr(a, b, c, d)))
                                         for a, b, c, d in zip(w.value, w.ci_low, w.ci_high, w.ci_status)])
    t4 = ts[ts.group.isna()].copy()
    ok(t, f"Sheet 4: winners.csv {len(w)} rows (slot/count/argmax_k); tie_sensitivity.csv {len(t4)} overall (group = NaN) rows")
    # builder selection: slots whose winner differs from the submitted-xlsx argmax (generous rule)
    c32 = s1[~s1.cell_id.str.startswith(("h_optimus_1/", "titan/"))]
    sel = []
    slots = wn[(wn.row_type == "slot") & (wn.slot_set == "C32")]
    for (enc, k), g in c32.groupby(["Encoder", "k"]):
        sw = g.loc[g.submitted_macro_f1.idxmax()]
        row = {"Encoder": enc, "k": int(k), "submitted-analysis winner (Table S3 argmax)": sw.Aggregation,
               "submitted winner value": sw.submitted_macro_f1}
        for rule in ("oracle", "plurality"):
            for to in ("canonical", "reversed"):
                r = slots[(slots.display_encoder == enc) & (slots.k == k) & (slots.rule == rule) & (slots.tie_order == to)]
                assert len(r) == 1, (enc, k, rule, to)
                r = r.iloc[0]
                row[f"{rule}/{to} winner"] = r.winner
                row[f"{rule}/{to} winner value"] = r.value
        for agg in g.Aggregation:
            gg = g[g.Aggregation == agg].iloc[0]
            row[f"submitted {agg}"] = gg.submitted_macro_f1
            row[f"generous canonical {agg}"] = gg.oracle_macro_f1
            row[f"plurality canonical {agg}"] = gg.plurality_macro_f1
        for rule in ("oracle", "plurality"):
            for to in ("canonical", "reversed"):
                row[f"differs ({rule}/{to} vs submitted)"] = row[f"{rule}/{to} winner"] != sw.Aggregation
        sel.append(row)
    sel = pd.DataFrame(sel)
    spec3 = [("EXAONEPath", 10), ("Hibou-L", 5), ("Virchow2", 5)]
    diffs = {c: sorted((a, int(b)) for a, b in sel[sel[f"differs ({c} vs submitted)"]][["Encoder", "k"]].values)
             for c in ("oracle/canonical", "oracle/reversed", "plurality/canonical", "plurality/reversed")}
    msg = "; ".join(f"{c}: {v}" for c, v in diffs.items())
    if diffs["oracle/reversed"] == spec3:
        ok(t, "Sheet 4: the spec's three slots (Virchow2 k = 5, EXAONEPath k = 10, Hibou-L k = 5) are exactly the "
              "slots whose generous-rule winner under the REVERSED tie order differs from the submitted Table S3 "
              f"argmax; generous canonical reproduces all 32 submitted winners ({msg})")
        note(t, "Sheet 4 caption wording: the spec says 'the three slots whose winner differs from the submitted "
                "analysis'; in run_003 this holds for the generous rule under the reversed tie order (canonical "
                "order reproduces every submitted winner). Author may want the caption to say so.")
    else:
        issue(t, "Sheet 4 slots differing from the submitted analysis", line,
              "Virchow2 k = 5, EXAONEPath k = 10, Hibou-L k = 5", msg,
              "builder-derived selection (argmax over submitted Table S3 values); not a new statistic")
    note(t, "Sheet 4 'changed_slots' is a builder-derived selection (argmax of submitted Table S3 macro F1 per "
            "encoder × k, compared with winners.csv slot winners); all 32 C32 slots listed with the flag columns.")
    note(t, "TITAN macro-F1 'best at k = 5 and 10' statement: l.7776 says move it into the Sheet 1 caption, l.7750 "
            "says Table S-R2.1-E footnote 6 (which l.7776 assigns to AP@k); the statement text is elided in the spec "
            "('…'), so it is not composed here. Owed: author wording.")
    # caption
    cap = block[1:]
    amend = [(ln, s) for ln, s in [(7776, None)]]
    extra_lines = []
    for sub_ in ("Move the TITAN macro-F1", "same numbers as Table S-R2.1-E sheet 1"):
        ln = find_line(sub_)
        if ln:
            extra_lines.append((f"edit-sheet l.{ln}", edit_lines()[ln - 1]))
    capdf = caption_sheet("Table S-R2.1-E (sheets 1–4)", [("### R2.1-T12", line)], cap,
                          extra_lines + [("Sheets 5–6", "already built: $PROSPER_OUT/supp_tables_20261021/"
                                          "Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx (merge at assembly)")])
    write_table(t, "Table_S-R2.1-E_sheets1-4.xlsx", capdf,
                [("Sheet1_main", s1), ("Sheet2_per_group_k5", s2), ("Sheet3_TITAN_repair", s3t),
                 ("Sheet4_winners", w), ("Sheet4_tie_sensitivity", t4), ("Sheet4_changed_slots", sel)],
                f"### R2.1-T12 (l.{line}) + 10-16 amendment")


# ---------------------------------------------------------------- S-R2.1-F (Virchow2 lineage)
F_RULE = {"plurality": "plurality (nearest-ranked tie-break)", "oracle": "generous (legacy 'oracle')",
          "rule_independent": "rule-independent"}
F_METRIC = {"macro_f1": "macro F1", "micro_f1": "micro F1", "hit_macro": "macro hit@k",
            "hit_micro": "micro hit@k", "precision_macro": "macro precision@k",
            "precision_micro": "micro precision@k"}
F_METHOD = {"median_pooling": "Median Pooling", "sampler": "SAMPLER", "ot_cosine": "Optimal Transport",
            "yottixel": "Yottixel"}


def build_f():
    t = "S-R2.1-F"
    line = find_line("Add **new Supplementary Table S-R2.1-F (lineage)", find_line("### R2.1-T12"))
    mline = find_line("The Virchow2 benchmark rows were recomputed from a regenerated fp32 feature bank")
    L = edit_lines()
    pe = rcsv(CMP / "point_estimates.csv")
    pdf = rcsv(CMP / "paired_differences.csv")
    lin = rcsv(CMP / "lineage_table_S3_virchow2.csv")
    mat = rjson(CMP / "materiality.json")
    fn = inp(CMP / "lineage_table_S3_virchow2.footnotes.md").read_text()
    sub = pd.read_excel(inp(SUBMITTED_S3), sheet_name="Overall Metrics")
    sub = sub[sub.Embedding == "Virchow2"]
    assert len(sub) == 16
    ok(t, "submitted Table_S3_raw_search_metrics.xlsx 'Overall Metrics' has 16 Virchow2 rows (4 aggregations × k 1/3/5/10)")
    key = ["method", "k", "rule", "metric", "tie_order"]
    assert len(pe) == 256 and not pe.duplicated(key).any()
    assert len(pdf) == 128 and set(pdf.tie_order) == {"canonical"} and not pdf.duplicated(key).any()
    m = pe.merge(pdf[key + ["D_point", "D_lo", "D_hi", "D_status", "n_invalid_replicates", "n_reps",
                            "flag_role", "material_flag", "flag_informational"]], on=key, how="left",
                 validate="one_to_one")
    lin_c = lin[key + ["submitted_value_xlsx", "submitted_ci_xlsx", "value_used_in_revised_manuscript",
                       "manuscript_location"]]
    m = m.merge(lin_c, on=key, how="left", validate="one_to_one")
    # consistency checks
    cn = m[m.tie_order == "canonical"]
    assert cn.D_point.notna().all() and m[m.tie_order == "reversed"].D_point.isna().all()
    d1 = (cn.D_point - cn.delta_fp32_minus_published_unrounded).abs().max()
    assert d1 < 5e-7, d1
    ok(t, f"paired_differences D_point == point_estimates delta (fp32 − published) for all 128 canonical rows (max |diff| {d1:.1e})")
    d2 = (m.delta_fp32_minus_published_unrounded - (m.fp32_value_unrounded - m.published_value_unrounded)).abs().max()
    assert d2 < 2e-6, d2
    ok(t, f"delta column == fp32 − published unrounded for all 256 rows (max |diff| {d2:.1e})")
    lc = lin.merge(pe, on=key, suffixes=("", "_pe"))
    assert len(lc) == 128
    for a, b in [("published_matrix_value", "published_value"), ("fp32_value", "fp32_value_pe"),
                 ("delta_fp32_minus_published", "delta_fp32_minus_published_unrounded")]:
        assert (lc[a] - lc[b]).abs().max() < 5e-7, a
    ok(t, "lineage_table_S3_virchow2.csv published/fp32/delta values == point_estimates.csv (128 canonical rows)")
    vu = (lc.value_used_in_revised_manuscript - lc.fp32_value).abs().max()
    assert vu < 5e-7
    ok(t, "lineage 'value_used_in_revised_manuscript' == fp32 value for all 128 rows (decision B: fp32 replaces submitted)")
    # submitted xlsx cross-check
    n_x, nrep = 0, []
    for _, s in sub.iterrows():
        meth = {v: k for k, v in F_METHOD.items()}[s.Aggregation]
        for met, col in (("macro_f1", "Macro F1"), ("micro_f1", "Micro F1")):
            r = m[(m.method == meth) & (m.k == s["Top K"]) & (m.rule == "oracle") & (m.metric == met)
                  & (m.tie_order == "canonical")].iloc[0]
            assert abs(r.submitted_value_xlsx - s[col]) < 5e-7, (meth, met)
            exp_ci = f"[{s[col + ' CI Low']:.6f}, {s[col + ' CI High']:.6f}]"
            assert r.submitted_ci_xlsx == exp_ci, (r.submitted_ci_xlsx, exp_ci)
            if abs(r.published_value_unrounded - s[col]) >= 5e-7:
                nrep.append(f"{F_METHOD[meth]} k={s['Top K']} {met}: submitted {s[col]:.6f} vs "
                            f"recomputed {r.published_value_unrounded:.6f}")
            n_x += 1
    ok(t, f"lineage submitted_value_xlsx/submitted_ci_xlsx == submitted Table S3 xlsx for all {n_x} generous-rule "
          f"macro/micro F1 cells; published-bank recomputation reproduces {n_x - len(nrep)}/{n_x} submitted point "
          "values (|Δ| < 5e-7)")
    if nrep:
        issue(t, "submitted value ≠ published-bank recomputation (generous rule, canonical)", line,
              "table pairs 'submitted value' with fp32 value and paired difference",
              f"{len(nrep)} cells differ: " + "; ".join(nrep),
              "Yottixel tie rows are not reproduced exactly by the recomputation (cf. S-R2.1-E footnote 3); the "
              "paired Δ is vs the recomputation, so Δ ≠ fp32 − submitted for these cells. Both columns shown.")
    # caption statements
    k5 = m[(m.k == 5) & (m.metric == "macro_f1")]
    unch = k5[k5.method != "yottixel"]
    assert (unch.delta_fp32_minus_published_unrounded == 0).all()
    ok(t, "caption: Median Pooling, SAMPLER and OT k = 5 macro F1 unchanged (Δ = 0 exactly) under both rules and both tie orders")
    y = lambda rule, to: k5[(k5.method == "yottixel") & (k5.rule == rule) & (k5.tie_order == to)].iloc[0]
    yc, yr = y("plurality", "canonical"), y("plurality", "reversed")
    check_cell(t, "caption Yottixel k = 5 plurality canonical Δ [95% CI]", line, "−0.003 [−0.009, +0.001]",
               [yc.D_point, yc.D_lo, yc.D_hi])
    check_cell(t, "caption Yottixel k = 5 plurality reversed Δ (point only)", line, "+0.009",
               [yr.delta_fp32_minus_published_unrounded])
    note(t, "Caption Yottixel numbers are the PLURALITY rule (package-flag cells). Under the generous rule, Yottixel "
            f"k = 5 Δ = {f(y('oracle','canonical').D_point, 6, True)} [{f(y('oracle','canonical').D_lo, 6, True)}, "
            f"{f(y('oracle','canonical').D_hi, 6, True)}] (canonical; interval excludes 0 but |Δ| < 0.005, "
            f"informational role, not flagged) and {f(y('oracle','reversed').delta_fp32_minus_published_unrounded, 6, True)} "
            "(reversed). Caption should name the rule.")
    # materiality rule re-applied (check, not a new statistic)
    pk = m[m.flag_role == "package_flag_cell"]
    assert len(pk) == 16
    def flagged(r):
        a = abs(r.D_point)
        return (r.D_status == "ok" and (r.D_lo > 0 or r.D_hi < 0) and a >= 0.005) or a >= 0.01
    nf = int(pk.apply(flagged, axis=1).sum())
    nall = int(m[m.D_point.notna()].apply(flagged, axis=1).sum())
    assert nf == 0 and mat["package_flag"] is False and mat["n_package_cells_flagged"] == 0
    assert (pk.material_flag == False).all()  # noqa: E712
    ok(t, "caption: materiality pre-registration not met — 0/16 package cells (plurality macro F1, canonical) "
          f"flagged on re-applying the materiality.json rule; materiality.json package_flag False; across all 128 "
          f"canonical paired intervals {nall} meet the rule")
    big = m[m.delta_fp32_minus_published_unrounded.abs() >= 0.005]
    note(t, "Rows with |Δ| ≥ 0.005 (any metric/rule/tie order; reversed rows are point-only): " +
         ("; ".join(f"{F_METHOD[r.method]} k={r.k} {r.rule} {r.metric} {r.tie_order} "
                    f"{f(r.delta_fp32_minus_published_unrounded, 6, True)}" for _, r in big.iterrows()) or "none"))
    exc = m[(m.D_lo > 0) | (m.D_hi < 0)]
    note(t, "Canonical paired intervals excluding 0 (all |Δ| < 0.005, so not flagged): " +
         ("; ".join(f"{F_METHOD[r.method]} k={r.k} {r.rule} {r.metric} {f(r.D_point, 6, True)} "
                    f"[{f(r.D_lo, 6, True)}, {f(r.D_hi, 6, True)}]" for _, r in exc.iterrows()) or "none"))
    # spec/source mismatches
    issue(t, "paired 95% interval 'under both tie orders'", line,
          "paired difference with 95% interval under both vote rules and both tie orders",
          "paired_differences.csv has canonical tie order only (128 rows); reversed rows have Δ as a point only",
          "reversed-order paired CI not in source; reported as point Δ with 'n.c.' interval — not computed here")
    issue(t, "'submitted value' under both vote rules", line,
          "submitted value ... under both vote rules",
          "submitted Table S3 contains the generous-rule F1 only (32 cells: macro/micro × 16); no submitted "
          "plurality, reversed-order or hit/precision values exist",
          "submitted column filled only for generous/canonical F1; 'published-bank recomputed' column (same bank, "
          "revised CI method) given for every row and is the Δ reference")
    issue(t, "Δ reference", mline, "Methods: table retains 'the submitted rows and the paired comparison'",
          "Δ = fp32 − published-bank recomputation (whole-patient bootstrap CI), not fp32 − submitted xlsx; point "
          "values agree with the submitted xlsx except the 8 Yottixel cells (item 1); CIs differ (submitted CI "
          "method pps_cluster_one_slide_class_stratified)",
          "layout states both; author to confirm wording")
    note(t, "Layout is the builder's proposal (spec gives contents, not column layout): sheet 'macro_F1' = the 16 "
            "cells × 2 rules × 2 tie orders (64 rows, headline); 'other_metrics' = micro F1 and rule-independent "
            "hit@k/precision@k (192 rows); 'raw_joined' = point_estimates ⋈ paired_differences ⋈ lineage (256 rows, "
            "unrounded).")
    note(t, "Median Pooling k = 10 changes under fp32 (generous canonical macro F1 published 0.726279 → fp32 "
            f"{f(m[(m.method=='median_pooling')&(m.k==10)&(m.rule=='oracle')&(m.metric=='macro_f1')&(m.tie_order=='canonical')].fp32_value_unrounded.iloc[0], 6)}).")
    pc = rcsv(CMP / "product_changes.csv")
    note(t, f"product_changes.csv ({len(pc)} rows) records Virchow2 slot-winner changes, e.g. " +
         "; ".join(f"{r.key}: {r.before} → {r.after}" for _, r in pc[pc.change_type == "slot_winner"].head(4).iterrows()))
    note(t, "comparison_001 lineage footnote 2 (carried verbatim in the caption sheet) says the Virchow2 tumour-only "
            "rows 'remain on the published bank until their separate package'. The later TF-2 package "
            "(r2_1_filter_tf2_20261016, source of S-R2.1-B/D) binds the Virchow2 fp32 similarity matrices "
            "(contract_candidate_002/CONTRACT.md), so that footnote appears superseded for the revised tumour-only "
            "rows; author to decide whether to keep it in the S-R2.1-F caption.")
    # build sheets
    order = {k: i for i, k in enumerate(AGG_ORDER)}
    m["_o"] = m.method.map(order)
    m["_r"] = m.rule.map({"plurality": 0, "oracle": 1, "rule_independent": 2})
    m["_m"] = m.metric.map({k: i for i, k in enumerate(F_METRIC)})
    m["_t"] = m.tie_order.map({"canonical": 0, "reversed": 1})
    m = m.sort_values(["_m", "_o", "k", "_r", "_t"]).reset_index(drop=True)

    def fmt(r):
        sub_ci = r.submitted_ci_xlsx if isinstance(r.submitted_ci_xlsx, str) else ""
        if sub_ci:
            lo, hi = [float(x) for x in sub_ci.strip("[]").split(",")]
            sv = fci(r.submitted_value_xlsx, lo, hi, 4)
        else:
            sv = "n.a."
        dci = (f"[{f(r.D_lo, 4, True)}, {f(r.D_hi, 4, True)}]" if pd.notna(r.D_lo) else "n.c.")
        return {"Encoder": "Virchow2", "Aggregation": F_METHOD[r.method], "k": int(r.k),
                "metric": F_METRIC[r.metric], "vote rule": F_RULE[r.rule], "tie order": r.tie_order,
                "submitted value [submitted 95% CI] (Table S3)": sv,
                "published bank, recomputed [95% CI]": fci(r.published_value, r.published_ci_low, r.published_ci_high, 4),
                "fp32 bank [95% CI]": fci(r.fp32_value, r.fp32_ci_low, r.fp32_ci_high, 4),
                "Δ fp32 − published": f(r.delta_fp32_minus_published_unrounded, 4, True),
                "paired 95% CI of Δ": dci,
                "materiality role": (r.flag_role if isinstance(r.flag_role, str) else "n.a. (reversed order)"),
                "flagged": ("no" if pd.notna(r.D_point) else "n.a."),
                "submitted_value_xlsx": r.submitted_value_xlsx, "submitted_ci_xlsx": r.submitted_ci_xlsx,
                "published_value_unrounded": r.published_value_unrounded,
                "published_ci_low": r.published_ci_low, "published_ci_high": r.published_ci_high,
                "fp32_value_unrounded": r.fp32_value_unrounded, "fp32_ci_low": r.fp32_ci_low,
                "fp32_ci_high": r.fp32_ci_high,
                "delta_fp32_minus_published_unrounded": r.delta_fp32_minus_published_unrounded,
                "D_lo": r.D_lo, "D_hi": r.D_hi, "D_status": r.D_status, "n_reps": r.n_reps}
    rows = pd.DataFrame([fmt(r) for _, r in m.iterrows()])
    mac = rows[rows.metric == "macro F1"].reset_index(drop=True)
    oth = rows[rows.metric != "macro F1"].reset_index(drop=True)
    assert len(mac) == 64 and len(oth) == 192
    ok(t, "macro_F1 sheet: 64 rows = 16 cells × 2 vote rules × 2 tie orders; other_metrics: 192 rows")
    raw = m.drop(columns=["_o", "_r", "_m", "_t"])
    cap = [(line, L[line - 1]), (mline, L[mline - 1])]
    capdf = caption_sheet("Table S-R2.1-F", [("### R2.1-T12 10-16 amendment", line),
                                             ("Methods note", mline)], cap,
                          [("Lineage footnotes (comparison_001, verbatim)", fn),
                           ("Materiality rule (materiality.json, verbatim)", mat.get("rule_verbatim", "")),
                           ("Materiality disclosure (materiality.json)", mat.get("disclosure", "")),
                           ("Bootstrap", mat.get("bootstrap_mode", "")),
                           ("Abbreviations", "n.a. = not applicable (no submitted value for this rule/order/metric); "
                            "n.c. = not computed in source (no reversed-order paired bootstrap)"),
                           ("Rounding", "4 decimals; a value printed +0.0000 or −0.0000 lies within 0.00005 of zero "
                            "and carries the sign of the unrounded value (raw columns)")])
    write_table(t, "Table_S-R2.1-F.xlsx", capdf,
                [("macro_F1", mac), ("other_metrics", oth), ("raw_joined", raw)],
                f"### R2.1-T12 10-16 amendment (l.{line}); Methods note l.{mline}")


# ---------------------------------------------------------------- S-R2.1-B / S-R2.1-D (TF-2 writing release 001)
TF2MD = TF2W / "tables_S_R2_1_B_D.md"
TRIP_RE = re.compile(r"([+\-\u2212]?\d+\.\d+) \[([+\-\u2212]?\d+\.\d+), ([+\-\u2212]?\d+\.\d+)\]")
ENC_LABEL = {"uni2": "UNI2", "virchow2": "Virchow2", "h_optimus_1": "H-Optimus-1", "titan": "TITAN"}
METH_LABEL = {"ot_cosine": "Optimal Transport", "yottixel": "Yottixel", "titan": "TITAN slide embedding (L1)"}
ARM_LABEL = {"query_only": "query-only", "both_filtered": "both-sided"}


def _tf2_sections():
    """Parse tables_S_R2_1_B_D.md -> {table: dict(heading, preface=[(ln, s)], parts=[dict])}."""
    L = inp(TF2MD).read_text().splitlines()
    out, cur, part = {}, None, None
    for i, s in enumerate(L, 1):
        if s.startswith("## Table "):
            tid = s.split()[2].rstrip(".")
            cur = out[tid] = dict(heading=(i, s), preface=[], parts=[])
            part = None
        elif cur is None:
            continue
        elif s.startswith("### Part "):
            part = dict(heading=(i, s), table=[], notes=[])
            cur["parts"].append(part)
        elif part is None:
            if s.strip():
                cur["preface"].append((i, s))
        elif s.strip().startswith("|"):
            part["table"].append((i, s))
        elif s.strip():
            part["notes"].append((i, s))
    for tid, sec in out.items():
        for p in sec["parts"]:
            h, rows, _ = md_table(p["table"])
            p["df"] = pd.DataFrame(rows, columns=h)
    return out


def _to_f(s):
    return float(s.replace(MINUS, "-"))


def _triplet_check(t, label, mdf, raw, prefixes, tol=0.00005 + 1e-9):
    """Every 'x [lo, hi]' printed in mdf must match some raw (point, lo, hi) triplet at 4 dp."""
    trips = []
    for pt, lo, hi in prefixes:
        trips.append(raw[[pt, lo, hi]].to_numpy(dtype=float))
    trips = np.vstack(trips)
    n, bad = 0, []
    for _, r in mdf.iterrows():
        for c in mdf.columns:
            for m_ in TRIP_RE.finditer(str(r[c])):
                v = np.array([_to_f(x) for x in m_.groups()])
                n += 1
                if not (np.abs(trips - v) <= tol).all(axis=1).any():
                    bad.append(f"{c}: {m_.group(0)}")
    if bad:
        issue(t, f"{label}: printed intervals not found in raw CSV", "", "", "; ".join(bad[:10]),
              f"{len(bad)} of {n} printed triplets have no raw match within 0.00005")
    else:
        ok(t, f"{label}: all {n} printed 'x [lo, hi]' values match a raw CSV triplet within 0.00005 (4-dp rounding)")
    return n


def _tf2_provenance(t):
    ih = rcsv(TF2W / "inputs_sha256.csv")
    bad, n = [], 0
    for _, r in ih.iterrows():
        p = OUT_ROOT / r.input if not str(r.input).startswith("/") else Path(r.input)
        if not p.exists():
            p = REPO / r.input
        if not p.exists():
            bad.append(f"{r.input}: missing")
            continue
        n += 1
        if sha256(p) != r.sha256:
            bad.append(f"{r.input}: hash changed")
    if bad:
        issue(t, "writing_001 inputs_sha256.csv provenance", "", "hashes as recorded", "; ".join(bad),
              "upstream S2/S6 inputs changed or missing since make_tables.py ran")
    else:
        ok(t, f"writing_001/inputs_sha256.csv: all {n} upstream inputs present with recorded sha256 (no drift since make_tables.py)")


def _parts_sheets(t, sec, raws, tag):
    sheets = []
    for p, (rawname, rdf, sel, prefixes) in zip(sec["parts"], raws):
        pn = re.match(r"### Part (\d+)", p["heading"][1]).group(1)
        mdf = p["df"]
        nsel = len(rdf if sel is None else rdf[sel(rdf)])
        if len(mdf) == nsel:
            ok(t, f"{tag} Part {pn}: md table {len(mdf)} rows == {rawname} rows selected for display ({nsel})")
        else:
            issue(t, f"{tag} Part {pn} row count", p["heading"][0], f"{len(mdf)} md rows", f"{nsel} CSV rows",
                  "md and CSV row counts differ")
        if prefixes:
            _triplet_check(t, f"{tag} Part {pn}", mdf, rdf, prefixes)
        sheets.append((f"Part{pn}", mdf))
        sheets.append((f"Part{pn}_raw", rdf))
    return sheets


def _tf2_caption(table_id, sec, specs, extra):
    lines = [sec["heading"]] + sec["preface"]
    for p in sec["parts"]:
        lines.append(p["heading"])
        lines += p["notes"]
    lines = [(f"tables_S_R2_1_B_D.md l.{ln}", s) for ln, s in lines]
    df = caption_sheet(table_id, specs, [], extra)
    src = pd.DataFrame(lines, columns=["field", "text"])
    return pd.concat([df.iloc[:1 + len(specs)], src, df.iloc[1 + len(specs):]], ignore_index=True)


def build_b():
    t = "S-R2.1-B"
    l6, _ = spec_block("R2.1-T6")
    l6a, blk = spec_block("R2.1-T6-A")
    l3 = next(ln for ln, s in blk if s.startswith("**(3) Table S-R2.1-B"))
    sec = _tf2_sections()["S-R2.1-B"]
    assert len(sec["parts"]) == 6
    ok(t, f"tables_S_R2_1_B_D.md § Table S-R2.1-B has caption and Parts 1–6 (spec T6-A item (3), l.{l3})")
    _tf2_provenance(t)
    T3 = [("hit_point", "hit_ci_low", "hit_ci_high"), ("precision_point", "precision_ci_low", "precision_ci_high"),
          ("hit_yield_point", "hit_yield_ci_low", "hit_yield_ci_high")]
    r = {n: rcsv(TF2W / f"S_R2_1_B_part{n}_{s}.csv") for n, s in
         [(1, "filtered_minus_all_tissue"), (2, "budget5_minus_budget100"), (3, "query_only_minus_both_sided"),
          (4, "sensitivity_k5"), (5, "coverage"), (6, "yottixel_tie_sensitivity_k5")]}
    k5 = lambda d: d.k == 5
    raws = [("part1 CSV (k = 5)", r[1], k5, T3), ("part2 CSV (k = 5)", r[2], k5, T3),
            ("part3 CSV (k = 5)", r[3], k5, T3), ("part4 CSV", r[4], None, T3[:2]),
            ("part5 CSV", r[5], None, None),
            ("part6 CSV", r[6], None, [("point_stable_reference_index", "ci_low_stable_reference_index",
                                        "ci_high_stable_reference_index"),
                                       ("point_tie_averaged", "ci_low_tie_averaged", "ci_high_tie_averaged")])]
    sheets = _parts_sheets(t, sec, raws, "B")
    # Part 5 integer check
    p5 = sec["parts"][4]["df"]
    got = sorted(tuple(int(str(x).replace(",", "")) for x in row) for row in p5[["Intended", "Returned", "Non-returns"]].values)
    exp = sorted(tuple(int(x) for x in row) for row in r[5][["intended", "returned", "non_returns"]].values)
    if got == exp:
        ok(t, "B Part 5: intended/returned/non-return counts in md == part5 CSV (35 rows, multiset)")
    else:
        issue(t, "B Part 5 counts", sec["parts"][4]["heading"][0], "md counts", "CSV counts", "multisets differ")
    note(t, "Part 1 display rows are k = 5; the Part1_raw sheet carries all k (1/3/5/10) per the md heading "
            "'(all k in the CSV)'. Parts 2–3 likewise.")
    note(t, "Per spec T6-A (3), the earlier H-Optimus-1 202-slide rows, route-(a) re-scoring rows and matched-UNI2 "
            "rows are not carried into S-R2.1-B (they remain as lineage in frozen packages).")
    note(t, "Panel-letter shift (edit sheet l.6818/6899: Fig. 3 A→B, B→C, C→D) may affect panel references inside "
            "the B/D captions; the md caption is carried verbatim — grep before Word incorporation.")
    capdf = _tf2_caption("Table S-R2.1-B", sec, [("### R2.1-T6", l6), ("### R2.1-T6-A item (3)", l3)],
                         [("Superseded", f"### R2.1-T6 (l.{l6}) 'New Table S-R2.1-B' content superseded by T6-A (l.{l6a})")])
    write_table(t, "Table_S-R2.1-B.xlsx", capdf, sheets, f"### R2.1-T6 (l.{l6}) + ### R2.1-T6-A item (3) (l.{l3})")


def build_d():
    t = "S-R2.1-D"
    l11, _ = spec_block("R2.1-T11")
    l11a, blka = spec_block("R2.1-T11-A")
    L = edit_lines()
    l8 = find_line("**(8) Table S-R2.1-D extension (new):**")
    l8p = find_line("**(8′) Table S-R2.1-D block 2:**")
    l8pp = find_line("**(8″) Table S-R2.1-D block 2:**")
    sec = _tf2_sections()["S-R2.1-D"]
    assert len(sec["parts"]) == 4
    ok(t, f"tables_S_R2_1_B_D.md § Table S-R2.1-D has caption and Parts 1–4 (spec T11-A l.{l11a})")
    mc = rcsv(TF2W / "S_R2_1_D_macro_cells.csv")
    hg = rcsv(TF2W / "S_R2_1_D_headline_cell_groups.csv")
    lg = rcsv(TF2W / "S_R2_1_D_part4_lineage.csv")
    TD = [("delta_f1", "delta_ci_low", "delta_ci_high"), ("rer", "rer_ci_low", "rer_ci_high")]
    raws = [("macro_cells (handling = miss)", mc, lambda d: d.handling == "miss", TD),
            ("macro_cells (handling = both_returned)", mc, lambda d: d.handling == "both_returned", TD),
            ("headline_cell_groups", hg, None, TD), ("part4_lineage", lg, None, TD)]
    sheets = []
    for i, (p, (nm, rdf, sel, pref)) in enumerate(zip(sec["parts"], raws), 1):
        mdf = p["df"]
        sub = rdf if sel is None else rdf[sel(rdf)].reset_index(drop=True)
        if len(mdf) == len(sub):
            ok(t, f"D Part {i}: md table {len(mdf)} rows == {nm} ({len(sub)})")
        else:
            issue(t, f"D Part {i} row count", p["heading"][0], f"{len(mdf)} md rows", f"{len(sub)} CSV rows", "")
        _triplet_check(t, f"D Part {i}", mdf, sub, pref)
        sheets += [(f"Part{i}", mdf), (f"Part{i}_raw", sub)]
    # headline / adverse cell / lineage values quoted in T11-A
    hl = mc[mc.cell_role == "headline_prenamed_cell"]
    assert len(hl) == 1
    hl = hl.iloc[0]
    assert (hl.encoder, hl.method, hl.arm, int(hl.budget), hl.handling) == ("uni2", "ot_cosine", "both_filtered", 100, "miss")
    check_cell(t, "T11-A headline RER (UNI2 OT both-sided 100 %, macro)", l11a, "+0.097 [+0.017, +0.175]",
               [hl.rer, hl.rer_ci_low, hl.rer_ci_high])
    ds = hg[hg["class"] == "DSRCT"].iloc[0]
    check_cell(t, "T11-A DSRCT RER (headline cell)", l11a, "−0.306 [−0.704, −0.067]",
               [ds.rer, ds.rer_ci_low, ds.rer_ci_high])
    hm = hg[hg["class"] == "MACRO"].iloc[0]
    assert max(abs(hm.rer - hl.rer), abs(hm.rer_ci_low - hl.rer_ci_low), abs(hm.rer_ci_high - hl.rer_ci_high)) < 5e-7
    ok(t, "headline_cell_groups MACRO row == macro_cells headline row (RER and interval)")
    liv = lg[(lg["class"] == "LIVER") & (lg.package == "r2_1_filter_rer_20261012/run_001") & (lg.encoder == "uni2")
             & (lg.method == "yottixel") & (lg.rule == "plurality")]
    hit = liv[(liv.rer.round(3) == 0.519)]
    if len(hit) >= 1:
        h0 = hit.iloc[0]
        check_cell(t, f"T11-A submitted Liver cell, common population (lineage row package={h0.package}, "
                      f"encoder={h0.encoder}, method={h0.method}, design={h0.design}, budget={h0.budget}, rule={h0.rule})",
                   l11a, "0.519 [0.307, 0.719]", [h0.rer, h0.rer_ci_low, h0.rer_ci_high])
    else:
        issue(t, "T11-A Liver 0.519 [0.307, 0.719]", l11a, "0.519 [0.307, 0.719]",
              "; ".join(f"{r.design}/{r.budget}: {r.rer:.4f}" for _, r in liv.iterrows()), "no matching lineage row")
    mu = lg[(lg["class"] == "LIVER") & (lg.package == "r2_1_route_b_uni2_20261013/rer_run_002")
            & (lg.rer.round(3) == 0.471)]
    if len(mu) >= 1:
        m0 = mu.iloc[0]
        check_cell(t, f"T11-A matched-UNI2 Liver value (lineage row method={m0.method}, design={m0.design}, "
                      f"budget={m0.budget}, rule={m0.rule})", l11a, "0.471 [0.226, 0.689]",
                   [m0.rer, m0.rer_ci_low, m0.rer_ci_high])
        if len(mu) > 1:
            note(t, f"T11-A matched-UNI2 0.471 matches {len(mu)} lineage rows; first reported.")
    else:
        issue(t, "T11-A matched-UNI2 0.471 [0.226, 0.689]", l11a, "0.471 [0.226, 0.689]", "no lineage row", "")
    # ---- block 2 (Fig. 3C per-group rows, both arms)
    b2 = rcsv(FIG3)
    cited(t, FIG3, "2eb5b20d", f"edit sheet (8″) rendered list l.{find_line('source_data_panelC_groups_all_arms.csv` (2eb5b20d')}")
    note(t, f"Block 2 source hash: (8)/(8′) cite 4f3d82ae… (earlier render, l.{l8}/{l8p}); the (8″) iteration "
            "re-rendered the file with RER columns and cites 2eb5b20d…, which is the bound file.")
    assert len(b2) == 196 and b2.groupby(["encoder", "method", "arm"]).size().eq(14).all() \
        and b2.groupby(["encoder", "method", "arm"]).ngroups == 14 and set(b2.budget) == {100} and set(b2.k) == {5}
    ok(t, "block 2: 196 rows = 14 groups × 14 settings (7 encoder/method × 2 arms), budget 100 %, k = 5")
    assert (b2.hit_n_queries == b2.precision_n_queries).all() and (b2.hit_n_queries == b2.rer_n_queries).all()
    ok(t, "block 2: hit, precision and RER query counts identical on every row")
    ref = b2[(b2.encoder == "uni2") & (b2.method == "ot_cosine") & (b2.arm == "query_only")].set_index("group").hit_n_queries
    dif = []
    for (e, mth, a), g in b2.groupby(["encoder", "method", "arm"]):
        d = g.set_index("group").hit_n_queries - ref
        for grp, v in d[d != 0].items():
            dif.append((e, mth, a, grp, int(v)))
    exp_ok = all(grp == "LIVER" and v == -1 and e in ("virchow2", "h_optimus_1") for e, _, _, grp, v in dif)
    (ok if exp_ok else lambda tt, mm: issue(tt, "block 2 query counts", l8p, "LIVER one fewer for Virchow2/H-Opt", mm))(
        t, f"block 2: query counts equal across settings except {sorted(set((e, grp, v) for e, _, _, grp, v in dif))} "
           "(slide 14601, LIVER, non-returned under Virchow2/H-Optimus-1 heads; B Part 5 note)")
    rd = b2.dropna(subset=["rer_point"])
    rerr = (rd.rer_point - rd.delta_f1 / (1 - rd.f1_all)).abs().max()
    assert rerr < 2e-6, rerr
    ok(t, f"block 2: rer_point == delta_f1 / (1 − f1_all) on all {len(rd)} rows with a point (max |diff| {rerr:.1e})")
    hc = b2[(b2.encoder == "uni2") & (b2.method == "ot_cosine") & (b2.arm == "both_filtered")].set_index("group")
    hgi = hg[hg["class"] != "MACRO"].set_index("class")
    cols = [("f1_all", "f1_all"), ("f1_filt", "f1_filt"), ("delta_f1", "delta_f1"), ("delta_ci_low", "delta_ci_low"),
            ("delta_ci_high", "delta_ci_high"), ("rer_point", "rer"), ("rer_ci_low", "rer_ci_low"),
            ("rer_ci_high", "rer_ci_high")]
    mx = max((hc.loc[hgi.index, a] - hgi[b]).abs().max() for a, b in cols)
    sup = (hc.loc[hgi.index, "rer_n_queries"] - hgi.support).abs().max()
    if mx < 5e-7 and sup == 0:
        ok(t, "block 2 UNI2/OT/both-sided per-group F1, ΔF1 [CI], RER [CI] and support == D Part 3 "
              "(headline_cell_groups.csv), 14 groups")
    else:
        issue(t, "block 2 vs D Part 3", l8pp, "same values", f"max |diff| {mx:.2e}, support diff {sup}",
              "block 2 headline-cell rows differ from Part 3")
    # macro consistency with B Part 1 (k = 5, budget 100)
    p1 = rcsv(TF2W / "S_R2_1_B_part1_filtered_minus_all_tissue.csv")
    p1 = p1[(p1.k == 5) & (p1.budget == 100)].set_index(["encoder", "method", "arm"])
    gm = b2.groupby(["encoder", "method", "arm"])[["hit_point", "precision_point"]].mean()
    mdh = (gm.hit_point - p1.loc[gm.index, "hit_point"]).abs().max()
    mdp = (gm.precision_point - p1.loc[gm.index, "precision_point"]).abs().max()
    if max(mdh, mdp) < 5e-6:
        ok(t, f"block 2 unweighted mean over 14 groups of Δhit@5/Δprecision@5 == B Part 1 macro (k = 5, 100 %) for "
              f"all 14 settings (max |diff| {max(mdh, mdp):.1e}; consistency check only, not reported)")
    else:
        issue(t, "block 2 vs B Part 1 macro", l8pp, "group mean == macro", f"max |diff| hit {mdh:.2e} prec {mdp:.2e}",
              "consistency check only")
    ns = b2[b2.rer_ci_status != "reported"]
    note(t, "block 2 RER not estimable (>1 % undefined replicates; interval suppressed, point kept): " +
         "; ".join(f"{ENC_LABEL[r.encoder]}/{METH_LABEL[r.method]}/{ARM_LABEL[r.arm]} {r.group} "
                   f"({int(r.rer_n_undefined_reps)} undefined)" for _, r in ns.iterrows()))
    eo = {k: i for i, k in enumerate(["uni2", "virchow2", "h_optimus_1", "titan"])}
    mo = {"ot_cosine": 0, "yottixel": 1, "titan": 2}
    ao = {"both_filtered": 0, "query_only": 1}
    b2s = b2.assign(_g=b2.group, _e=b2.encoder.map(eo), _m=b2.method.map(mo), _a=b2.arm.map(ao)) \
        .sort_values(["_g", "_e", "_m", "_a"]).reset_index(drop=True)
    rows = []
    for _, r in b2s.iterrows():
        est = r.rer_ci_status == "reported"
        rows.append({"Group": r.group, "Query slides (n)": int(r.hit_n_queries), "Patients (n)": int(r.n_patients),
                     "Encoder": ENC_LABEL[r.encoder], "Aggregation": METH_LABEL[r.method], "Arm": ARM_LABEL[r.arm],
                     "Δ hit@5 [95 % CI]": fci(r.hit_point, r.hit_ci_low, r.hit_ci_high, 4, True),
                     "Δ precision@5 [95 % CI]": fci(r.precision_point, r.precision_ci_low, r.precision_ci_high, 4, True),
                     "F1 all-tissue": f(r.f1_all, 4), "F1 filtered": f(r.f1_filt, 4),
                     "ΔF1 [95 % CI]": fci(r.delta_f1, r.delta_ci_low, r.delta_ci_high, 4, True),
                     "RER [95 % CI]": (fci(r.rer_point, r.rer_ci_low, r.rer_ci_high, 4, True) if est
                                       else f"{f(r.rer_point, 4, True)} [not estimable]"),
                     "RER interval status": ("reported" if est else
                                             f"not estimable ({int(r.rer_n_undefined_reps)} of 5,000 replicates undefined)")})
    blk2 = pd.DataFrame(rows)
    _triplet_check(t, "D block 2 (formatted vs source)", blk2, b2,
                   [("hit_point", "hit_ci_low", "hit_ci_high"), ("precision_point", "precision_ci_low", "precision_ci_high"),
                    ("delta_f1", "delta_ci_low", "delta_ci_high"), ("rer_point", "rer_ci_low", "rer_ci_high")])
    sheets += [("Block2_groups", blk2), ("Block2_raw", b2)]
    note(t, "Block 2 column order/labels follow (8)/(8″); 'Patients (n)' added from the source (n_patients). Rows "
            "sorted by group, then encoder (UNI2, Virchow2, H-Optimus-1, TITAN), aggregation, arm (both-sided first, "
            "the main-figure arm). Not-estimable RER intervals are printed as '[not estimable]' with the point kept "
            "(as in the source CSV); author may prefer to suppress the point.")
    note(t, "(8″) says the block-1 headline RER rows (Part 3) 'may be merged' into block 2 — not merged here; Part 3 "
            "kept as built by make_tables.py.")
    note(t, "RER on a precision@5 base (sixth iteration, decision pending with Haoran) is NOT added.")
    l8_txt = L[l8 - 1:l8 + 5]
    cap_add = ("Block 2: per-group change in hit@5 and precision@5 with tumor-only filtering for all 14 settings of "
               "Fig. 3B, queries returned in both arms, whole-patient bootstrap 95% intervals (Fig. 3C); the main "
               "figure shows the tumor-only query and library arm.")
    assert "Block 2: per-group change in hit@5 and precision@5" in " ".join(l8_txt)
    note(t, "Block 2 caption addendum assembled from (8) verbatim + the (8′) clause; (8″) adds RER columns but "
            "gives no caption wording for them — caption sentence for F1/RER columns owed (not drafted here). Panel "
            "letters (Fig. 3B/3C) may shift per the 10-21 A→B→C→D note.")
    capdf = _tf2_caption("Table S-R2.1-D", sec, [("### R2.1-T11", l11), ("### R2.1-T11-A", l11a),
                                                 ("(8) block 2", l8), ("(8′) block 2", l8p), ("(8″) block 2", l8pp)],
                         [("Block 2 caption addendum ((8) + (8′))", cap_add)])
    write_table(t, "Table_S-R2.1-D.xlsx", capdf, sheets,
                f"### R2.1-T11 (l.{l11}) + ### R2.1-T11-A (l.{l11a}); block 2 (8)/(8′)/(8″) l.{l8}/{l8p}/{l8pp}")


# ---------------------------------------------------------------- main / reports
TABLES = [("S-R2.1-A", "build_a", "Table S-R2.1-A — encoder–aggregation scope note table (Fig. 2 legend companion)"),
          ("S-R2.1-E", "build_e", "Table S-R2.1-E sheets 1–4 — 38 encoder–aggregation cells, plurality rule, hit@k"),
          ("S-R3.1", "build_r31", "Table S-R3.1 — comparators and module comparisons"),
          ("S25", "build_s25", "Table S25 — standard retrieval metrics, main configuration"),
          ("S-R2.1-B", "build_b", "Table S-R2.1-B — tumour-only filtering, TF-2 patient-held-out heads"),
          ("S-R2.1-D", "build_d", "Table S-R2.1-D — fair-design top-5 vote F1 RER (+ block 2, Fig. 3C groups)"),
          ("S-R2.1-F", "build_f", "Table S-R2.1-F — Virchow2 fp32 recomputation lineage")]
PREV_FILES = ["S-R2.1-E_sheet5_retrieval_metrics.csv", "S-R2.1-E_sheet6_per_group_k5.csv",
              "Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx"]
NUMBERING = ("Numbering: IDs are the edit-sheet working IDs. Per docs/analysis/supplementary_numbering_map_20261022.md "
             "(read-only; not applied here) the proposed final numbers are S19 = S-R2.1-E, S20 = S-R2.1-A, "
             "S21 = S-R3.1, S25 = S25 (S-A3 standard retrieval metrics), S29 = S-R2.1-B, S30 = S-R2.1-D, "
             "S41 = S-R2.1-F. Final numbers are applied at assembly, not here.")


def _short(p):
    s = re.sub(r"^.*/prosper_revision/prosper_revision_output", "$PROSPER_OUT", str(p))
    return re.sub(r"^.*/Haoran/prosper_revision/", "", s)


def _esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


SPEC_EQ = re.compile(r"spec '.*' \(l\.(\d+)\) == source$")


def _summarise_checks(cs):
    """Collapse per-cell 'spec == source' and cross-check lines into counts when a table has many."""
    if len(cs) <= 25:
        return [f"- {c}" for c in cs]
    eq = [int(m.group(1)) for c in cs for m in [SPEC_EQ.search(c)] if m]
    xc = [c for c in cs if c.startswith("cross-check ") or " agrees with " in c]
    rest = [c for c in cs if not SPEC_EQ.search(c) and c not in xc]
    out = []
    if eq:
        out.append(f"- {len(eq)} spec-printed values (edit-sheet l.{min(eq)}–{max(eq)}) found in the source at the "
                   "printed rounding (each listed in BUILD_CHECKS_FULL.md)")
    if xc:
        out.append(f"- {len(xc)} cross-source agreement checks passed (e.g. '{xc[0]}')")
    return out + [f"- {c}" for c in rest]


def write_reports(failed):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # README
    R = ["# Group C (encoders / comparators) supplementary tables — build 2026-10-22", "",
         f"Built by `scripts/supp_tables/build_C_encoders_comparators_20261022.py` at {now}. Assembly only: values "
         "are copied/formatted from accepted sources; no statistic recomputed. Each `.xlsx` has a first `caption` "
         "sheet (spec heading + line, candidate caption/footnotes verbatim from the edit sheet or the named "
         "writing release) followed by data sheets; every data sheet is also written as "
         "`<xlsx stem>__<sheet>.csv` (formatted display columns plus raw unrounded columns where applicable).", "",
         NUMBERING, ""]
    for t, _, title in TABLES:
        R.append(f"## {title}")
        if t in BUILT:
            b = BUILT[t]
            R.append(f"- File: `{b['xlsx']}`; spec: {b['spec']}")
            for name, csv, n, c in b["sheets"]:
                R.append(f"  - sheet `{name}` → `{csv}` ({n} rows × {c} cols)")
        else:
            R.append(f"- NOT BUILT: {failed.get(t, 'unknown')}")
        if t == "S-R2.1-E":
            R.append(f"- Sheets 5–6 are NOT rebuilt here; they already exist in `{_short(PREV)}/`: " +
                     ", ".join(f"`{p}`" + ("" if (PREV / p).exists() else " (MISSING)") for p in PREV_FILES) +
                     ". Merge into the final S-R2.1-E workbook at assembly.")
        R.append("")
    R += ["## Directory-level files",
          "- `input_bindings.json` — every input read: path → sha256, mtime (UTC ISO).",
          "- `SHA256SUMS.txt` — sha256 of every output file in this directory (except itself).",
          "- `BUILD_REPORT.md` — per-table spec location, sources, sheet shapes, passed assertions, "
          "discrepancies, notes and anything unbuilt.",
          "- `BUILD_CHECKS_FULL.md` — the full list of passed assertions (BUILD_REPORT collapses per-cell "
          "spec-value matches into counts for S-R3.1 and S25).", ""]
    (OUT / "README.md").write_text("\n".join(R))
    # input bindings
    ib = {p: {"sha256": h, "mtime_utc": datetime.fromtimestamp(os.path.getmtime(p), timezone.utc)
              .isoformat(timespec="seconds")} for p, h in sorted(INPUTS.items())}
    (OUT / "input_bindings.json").write_text(json.dumps(ib, indent=2) + "\n")
    # build report
    B = ["# BUILD_REPORT — group C_encoders_comparators (2026-10-22)", "",
         f"Script: `scripts/supp_tables/build_C_encoders_comparators_20261022.py`; output: `{OUT}`; run {now}.",
         f"Inputs bound: {len(INPUTS)} (see `input_bindings.json`). Spec: `docs/manuscript_supplement_edits.md` "
         f"(sha256 {INPUTS.get(str(EDITS), '?')[:16]}…). Assembly only — no statistic recomputed; checks compare "
         "printed spec numbers with sources at the printed rounding.", "",
         NUMBERING, "", "## Summary", "",
         "| Table | Built | Sheets (rows × cols) | Checks passed | Discrepancies | Notes |", "|---|---|---|---|---|---|"]
    for t, _, _ in TABLES:
        b = BUILT.get(t)
        sh = "; ".join(f"{n} {r}×{c}" for n, _, r, c in b["sheets"]) if b else "—"
        B.append(f"| {t} | {'yes' if b else 'NO'} | {sh} | {len(CHECKS.get(t, []))} | {len(ISSUES.get(t, []))} | "
                 f"{len(NOTES.get(t, []))} |")
    B.append("")
    for t, _, title in TABLES:
        B += [f"## {title}", ""]
        b = BUILT.get(t)
        if b:
            B.append(f"- **Spec implemented:** {b['spec']}")
            B.append(f"- **Output:** `{b['xlsx']}` — sheets: " +
                     "; ".join(f"`{n}` {r} × {c}" for n, _, r, c in b["sheets"]))
        else:
            B.append(f"- **NOT BUILT:** {failed.get(t)}")
        if t == "S-R2.1-E":
            B.append(f"- Sheets 5–6: not rebuilt; existing files in `{_short(PREV)}/` ({', '.join(PREV_FILES)}).")
        srcs = sorted({p for p in INPUTS if any(k in p for k in SRC_KEYS.get(t, []))})
        if srcs:
            B.append("- **Sources (sha256 prefix):** " + "; ".join(f"`{_short(p)}` "
                                                               f"{INPUTS[p][:12]}" for p in srcs))
        B += ["", f"**Checks passed ({len(CHECKS.get(t, []))}; full list in `BUILD_CHECKS_FULL.md`):**", ""]
        B += _summarise_checks(CHECKS.get(t, []))
        B += ["", f"**Discrepancies ({len(ISSUES.get(t, []))}):**", ""]
        if ISSUES.get(t):
            B += ["| # | Item | Spec line | Spec says | Source has | Note |", "|---|---|---|---|---|---|"]
            for i, d in enumerate(ISSUES[t], 1):
                B.append(f"| {i} | {_esc(d['item'])} | {d['spec_line']} | {_esc(d['spec_value'])} | "
                         f"{_esc(d['source_value'])} | {_esc(d['note'])} |")
        else:
            B.append("- none")
        B += ["", f"**Notes / left for the author ({len(NOTES.get(t, []))}):**", ""]
        B += [f"- {n}" for n in NOTES.get(t, [])] or ["- none"]
        B.append("")
    (OUT / "BUILD_REPORT.md").write_text("\n".join(B))
    F = ["# BUILD_CHECKS_FULL — every passed assertion, group C_encoders_comparators", ""]
    for t, _, title in TABLES:
        F += [f"## {title}", ""] + [f"- {c}" for c in CHECKS.get(t, [])] + [""]
    (OUT / "BUILD_CHECKS_FULL.md").write_text("\n".join(F))
    # checksums last
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))


SRC_KEYS = {
    "S-R2.1-A": ["rr5_replacement_001/hit_precision", "rr5_replacement_001/tie_census", "contrasts.csv",
                 "manuscript_supplement_edits"],
    "S-R2.1-E": ["rr5_replacement_001/", "fig2_benchmark/source_data_panelC", "manuscript_supplement_edits"],
    "S-R3.1": ["r3_1_", "manuscript_supplement_edits"],
    "S25": ["c2_metric_suite", "r2_3_internal_failure", "review_scratch_astra", "r2_1_fig2_retrieval",
            "manuscript_supplement_edits"],
    "S-R2.1-B": ["writing_001/S_R2_1_B", "writing_001/inputs_sha256", "writing_001/tables_S_R2_1_B_D",
                 "manuscript_supplement_edits"],
    "S-R2.1-D": ["writing_001/S_R2_1_D", "writing_001/S_R2_1_B_part1", "writing_001/tables_S_R2_1_B_D",
                 "fig3_tumor_filtering", "manuscript_supplement_edits"],
    "S-R2.1-F": ["comparison_001/", "Table_S3_raw_search_metrics", "manuscript_supplement_edits"],
}


def main():
    import shutil
    import traceback
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    failed = {}
    for t, fn, _ in TABLES:
        try:
            globals()[fn]()
        except Exception as e:  # report, never silently skip
            failed[t] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            BUILT.pop(t, None)
    write_reports(failed)
    for t, _, _ in TABLES:
        print(f"{t:9s} built={t in BUILT} checks={len(CHECKS.get(t, []))} issues={len(ISSUES.get(t, []))}"
              f"{' FAILED: ' + failed[t] if t in failed else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
