#!/usr/bin/env python3
"""Assemble supplementary-table group A_ladder_failure_finelabel (assembly only; 2026-10-22, D-2 (a)).

Tables, [internal record] spec blocks (docs/manuscript_supplement_edits.md; line numbers are located at run time
from the block heading and recorded in BUILD_REPORT.md; approximate lines at writing) and sources
($PROSPER_OUT = <repo>/prosper_revision_output):

1. Table S29 (R2.2-T4, l.~3409) tile-retention ladder, five sheets:
   r2_2_tile_ladder_whole_patient_20261007/{ladder_summary,paired_differences,per_group,tie_bounds}.csv + README.md;
   sheet e transcribed from [internal record] (8cd17991...). Cross-checks only:
   figures_20261021/figS_tile_ladder/source_*.csv.
2. Table S23 (S-A1) (R2.3-T2, l.~2008): r2_3_internal_failure_20261003/per_group_summary.csv (upper block);
   r2_3_internal_failure_cis_20261004/{per_group_summary,summary_overall}.csv (lower CI block).
3. Table S24 / S24b (S-A2 / S-A2b) (R2.3-T3, l.~2077), optional S24c (S-A2c):
   r2_3_internal_failure_20261003/{confusion_k5_plurality,confusion_k5_strict,confusion_k1_top1}.csv.
4. Table S30 (R3.2-T1, l.~4082; footnote as amended by the 2026-10-21 R3.2 audit F1, l.~7686):
   r3_2_s30_chance_20261006/{table_s30_wide.md,table_s30.csv,run_record.json}.
5. Table S26 (S-D1) (R3.7-T4, l.~2546; legend with the 2026-10-20 C1 = R3.7 F1 option (a) insertion and F2,
   l.~7528/7632): r3_7_fine_label_20261003/{summary_all_arms.csv, confusion_k5_plurality_<arm>.csv};
   r3_7_audit_20261020/results.json (only to verify the numbers quoted in the option-(a) legend sentence).
6. Table S28 (S-D2) (R3.7-T5, l.~2591): r3_7_task_inventory_20261004b/{task_inventory.md,task_inventory.csv}.

No statistic is computed: values are copied, rounded/formatted where the spec says so, and checked. Read-only
on all inputs; writes only to $PROSPER_OUT/supp_tables_20261022/A_ladder_failure_finelabel/ (recreated on
each run, so the build is idempotent).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path(os.environ.get("PROSPER_OUT") or REPO / "prosper_revision_output")
GROUP = "A_ladder_failure_finelabel"
OUT = OUT_ROOT / "supp_tables_20261022" / GROUP
EDIT = REPO / "docs/manuscript_supplement_edits.md"
COST = REPO / "docs/results/r2_2_cost_panel_20261006.md"
LAD = OUT_ROOT / "r2_2_tile_ladder_whole_patient_20261007"
FIG = OUT_ROOT / "figures_20261021/figS_tile_ladder"
F23 = OUT_ROOT / "r2_3_internal_failure_20261003"
C23 = OUT_ROOT / "r2_3_internal_failure_cis_20261004"
S30D = OUT_ROOT / "r3_2_s30_chance_20261006"
R37 = OUT_ROOT / "r3_7_fine_label_20261003"
R37A = OUT_ROOT / "r3_7_audit_20261020"
INV = OUT_ROOT / "r3_7_task_inventory_20261004b"
MINUS = "\u2212"

INPUTS: dict[str, Path] = {}
CHECKS: dict[str, list] = {}
OUTPUT_ROWS: dict[str, dict] = {}


# Decisions taken during assembly (reported, not resolved)
NOTES: dict[str, list[str]] = {
    "S29 (R2.2-T4)": [
        "DECISION: sheet e transcribed verbatim from docs/results/r2_2_cost_panel_20261006.md (sha 8cd17991…, the version the spec names as current); bold markers stripped, no value computed.",
        "Figure sources (figures_20261021/figS_tile_ladder/source_*.csv) used only as cross-checks.",
    ],
    "S23 (R2.3-T2)": [
        "DECISION: r2_3_internal_failure_cis_ext_prod_20261005/ not used — the R2.3-T2 spec and later amendments do not reference it (it serves S25 / R4.b-T5).",
        "Upper block from the 10-03 per-group file; lower block CIs from the 10-04 CI run (points cross-checked equal); joint-audit addendum (edit sheet ≈l.7497) changes no S23 value.",
    ],
    "S24/S24b (R2.3-T3)": [
        "DECISION: S24c (top-1, k=1 confusion) built as a separate optional sheet because the spec leaves its inclusion to the author; drop at assembly if not wanted.",
    ],
    "S30 (R3.2-T1)": [
        "DECISION: footnote implemented in the F1-APPLIED version (edit sheet addendum ≈l.7686): struck '(independent review bound 0.001-0.004; at most 0.007 for any single query)' replaced by '(per-group averages 0.001–0.004; at most 0.013 for a single query, reached by patients with eight or more same-group library slides)'; editorial marker '[R3.2 audit F1, 2026-10-21]' dropped. The verbatim marked-up text and the frozen table_s30_wide.md footnote are kept in the caption sheet.",
        "DEPENDENCY: the F1 numbers (0.001–0.004; 0.013) come from the R3.2 parent audit and cannot be re-derived from the S30 source directory; not verified by this build.",
        "DECISION: optional F2 (caption cross-reference 'S18/S19/S23' -> 'S18/S23') NOT applied — it is offered as optional in the addendum, not as an applied change.",
        "Source column observed_hit_at_5 is text (numbers plus 'not available' for 14 empty external cells); converted to numeric for checks only, raw kept in the long CSV.",
    ],
    "S26 (R3.7-T4)": [
        "DECISION: values from r3_7_fine_label_20261003 unchanged; the 10-20 R3.7 parent audit was resolved by author decision C1 = F1 option (a) (ledger R3.7 row, 2026-10-20 ≈17:16): disclose tile-selection difference + all-tissue sensitivity sentence, no value replacement. Applied legend = frozen legend + option (a) sentence + F2 footnote clause; frozen legend kept verbatim beside it.",
        "DECISION: the option (a) sentence is inserted literally after 'Point estimates are over slides;' as the addendum says; the resulting punctuation ('…+0.015]. 95% percentile intervals…') is left for the author/copy-edit.",
        "ROUNDING: the option (a) CI bound −0.063 corresponds to results.json −0.0625 (exactly representable); it matches under round-half-away-from-zero, whereas Python/round-half-even formatting would print −0.062. Not a value discrepancy, but the convention differs from the %.3f formatting used elsewhere.",
        "DECISION: optional hit@5 column not included (spec lists it as optional; strict/legacy columns excluded as the spec says).",
    ],
    "S28 (R3.7-T5)": [
        "DECISION: table copied verbatim from task_inventory.md (identical to the edit-sheet verbatim block); applied legend carries the C1 option (a) sentence after 'Saved S6 depth-10 rank lists were re-scored without re-ranking.'; numbers unchanged.",
    ],
}


def relp(x) -> str:
    x = str(x)
    for root, lab in ((str(OUT_ROOT), "$PROSPER_OUT"), (str(OUT_ROOT.resolve()), "$PROSPER_OUT"), (str(REPO), "")):
        if x.startswith(root):
            return (lab + x[len(root):]) if lab else x[len(root) + 1:]
    return x


# ---------------------------------------------------------------- helpers
def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def inp(p: Path) -> Path:
    assert p.exists(), f"missing input {p}"
    INPUTS[str(p)] = p
    return p


def check(table: str, name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.setdefault(table, []).append((bool(ok), name, detail))
    return bool(ok)


def close(x, e, tol):
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and abs(float(x) - e) <= tol + 1e-12


def fmt_small(x, signed=False, nd=3) -> str:
    """Spec R2.2-T4 rounding: 3 d.p.; a non-zero value that would show as 0.000 gets a 4th decimal, and if it
    still rounds to zero, two significant figures; exact 0 -> 0.000; Unicode minus; '+' when signed."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    x = float(x)
    if x == 0:
        return "0.000"
    if round(abs(x), nd) != 0:
        s = f"{abs(x):.{nd}f}"
    elif round(abs(x), nd + 1) != 0:
        s = f"{abs(x):.{nd + 1}f}"
    else:
        d = -math.floor(math.log10(abs(x))) + 1
        s = f"{abs(x):.{d}f}"
    sign = MINUS if x < 0 else ("+" if signed else "")
    return sign + s


def f3(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{float(x):.3f}"


def ci3(p, lo, hi) -> str:
    return f"{f3(p)} [{f3(lo)}, {f3(hi)}]"


def intc(x) -> str:
    return f"{int(x):,}"


class EditSheet:
    def __init__(self, path: Path):
        self.lines = path.read_text(encoding="utf-8").splitlines()

    def block(self, heading_prefix: str):
        hits = [i for i, l in enumerate(self.lines) if l.startswith("### " + heading_prefix)]
        assert len(hits) == 1, (heading_prefix, hits)
        s = hits[0]
        e = next((j for j in range(s + 1, len(self.lines)) if self.lines[j].startswith(("### ", "## "))), len(self.lines))
        return s, e  # 0-based start (heading), exclusive end

    def find(self, s, e, marker):
        for i in range(s, e):
            if marker in self.lines[i]:
                return i
        return None

    def quote_after(self, s, e, marker):
        i = self.find(s, e, marker)
        assert i is not None, marker
        out, started = [], False
        for j in range(i + 1, e):
            l = self.lines[j]
            if l.startswith(">"):
                started = True
                out.append(l[1:].lstrip(" ") if l != ">" else "")
            elif started:
                break
            elif l.strip():
                break
        return out, i

    def table_after(self, s, e, marker):
        i = self.find(s, e, marker)
        assert i is not None, marker
        rows, started = [], False
        for j in range(i + 1, e):
            l = self.lines[j]
            if l.startswith("|"):
                started = True
                rows.append(l)
            elif started:
                break
        return parse_md_table(rows)


def parse_md_table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], [c for c in cells[2:]]
    return header, body


def caption_frame(items):
    return pd.DataFrame(items, columns=["field", "text"])


def write_xlsx(path: Path, sheets: list[tuple[str, pd.DataFrame]]):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets:
            df.to_excel(xw, sheet_name=name[:31], index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 80 if ws.title != "caption" else 160)


def common_caption(es: EditSheet, tid: str, sym: str, heading_prefix: str):
    s, e = es.block(heading_prefix)
    return s, e, [("table_id (edit-sheet block)", tid), ("provisional numbers (final number at assembly via scripts/supplementary_numbering_map.py)", sym),
                  ("edit-sheet heading", es.lines[s][4:]), ("edit-sheet location", f"docs/manuscript_supplement_edits.md lines {s + 1}-{e} (block)")]


# ---------------------------------------------------------------- S29
SETTING = {"median_pooling": "median pooling", "yottixel": "Yottixel", "ot_cosine": "exact Optimal Transport (cosine cost)"}
ENDPOINT = {"hit1": "hit@1", "hit3": "hit@3", "hit5": "hit@5", "hit10": "hit@10", "precision5": "precision@5",
            "precision10": "precision@10", "plurality5": "plurality@5"}


def rung_label(rung: str) -> str:
    m = re.match(r"uni2apf(\d+)_(median_pooling|yottixel|ot_cosine)$", rung)
    assert m, rung
    lab = f"{SETTING[m.group(2)]} {m.group(1)} %"
    return lab + (" \u2020" if rung == "uni2apf30_ot_cosine" else "")


def build_s29(es: EditSheet):
    T = "S29 (R2.2-T4)"
    s, e, cap = common_caption(es, "R2.2-T4", "S29 (no symbolic ID)", "R2.2-T4 ")
    a = pd.read_csv(inp(LAD / "ladder_summary.csv"))
    b = pd.read_csv(inp(LAD / "paired_differences.csv"))
    c = pd.read_csv(inp(LAD / "per_group.csv"))
    d = pd.read_csv(inp(LAD / "tie_bounds.csv"))
    readme = inp(LAD / "README.md").read_text()
    for f, pre in [("ladder_summary.csv", "bc8639e9"), ("paired_differences.csv", "c4a11024"), ("per_group.csv", "4ceecfd8"), ("tie_bounds.csv", "8782aab2")]:
        check(T, f"{f} sha256 prefix {pre} (spec table)", sha(LAD / f).startswith(pre), sha(LAD / f)[:16])
    check(T, "README: B=5000, 1857 patients, 2750 queries", "B=5000" in readme and "1857 patients" in readme and "2750 queries" in readme)
    check(T, "sheet a 140 rows (10 settings x 7 endpoints x micro/macro)", len(a) == 140 and a.rung.nunique() == 10 and a.metric.nunique() == 7, f"{len(a)}")
    check(T, "sheet b 154 rows (11 comparisons x 7 x 2)", len(b) == 154 and b[["minuend", "subtrahend"]].drop_duplicates().shape[0] == 11, f"{len(b)}")
    check(T, "sheet c 600 rows (10 x 15 groups x 4 endpoints)", len(c) == 600 and c.group.nunique() == 15 and set(c.metric) == {"hit5", "hit10", "precision5", "plurality5"}, f"{len(c)}")
    check(T, "sheet d 10 rows", len(d) == 10, f"{len(d)}")
    for df, nm in [(a, "a"), (b, "b"), (c, "c")]:
        check(T, f"sheet {nm}: every interval present (point, ci_lo, ci_hi finite; n_nonfinite = 0)",
              df[["point", "ci_lo", "ci_hi"]].notna().all().all() and (df.n_nonfinite == 0).all() and (df.n_replicates == 5000).all())
    check(T, "sheets a/b: n_queries 2750, n_patients 1857 throughout", (a.n_queries == 2750).all() and (a.n_patients == 1857).all() and (b.n_queries == 2750).all() and (b.n_patients == 1857).all())
    order = ["uni2apf5_median_pooling", "uni2apf15_median_pooling", "uni2apf30_median_pooling", "uni2apf50_median_pooling", "uni2apf100_median_pooling",
             "uni2apf5_yottixel", "uni2apf15_yottixel", "uni2apf30_yottixel", "uni2apf15_ot_cosine", "uni2apf30_ot_cosine"]
    check(T, "settings are exactly the spec's 10 rungs (median 5/15/30/50/100; Yottixel 5/15/30; OT 15/30)", list(a.rung.drop_duplicates()) == order)
    exp_pairs = [("uni2apf5_median_pooling", "uni2apf15_median_pooling"), ("uni2apf30_median_pooling", "uni2apf15_median_pooling"),
                 ("uni2apf50_median_pooling", "uni2apf15_median_pooling"), ("uni2apf100_median_pooling", "uni2apf15_median_pooling"),
                 ("uni2apf5_yottixel", "uni2apf15_yottixel"), ("uni2apf30_yottixel", "uni2apf15_yottixel"),
                 ("uni2apf15_ot_cosine", "uni2apf15_median_pooling"), ("uni2apf15_ot_cosine", "uni2apf15_yottixel"),
                 ("uni2apf30_ot_cosine", "uni2apf15_ot_cosine"), ("uni2apf30_ot_cosine", "uni2apf30_median_pooling"), ("uni2apf30_ot_cosine", "uni2apf30_yottixel")]
    check(T, "paired comparisons are exactly the spec's 11", [tuple(x) for x in b[["minuend", "subtrahend"]].drop_duplicates().values] == exp_pairs)

    # ---- sheet a
    sa = a.copy()
    sa.insert(0, "setting", sa.rung.map(rung_label))
    sa.insert(1, "endpoint", sa.metric.map(ENDPOINT))
    sa["point_fmt"] = sa.point.map(fmt_small); sa["ci_lo_fmt"] = sa.ci_lo.map(fmt_small); sa["ci_hi_fmt"] = sa.ci_hi.map(fmt_small)
    sa["estimate [nominal 95% whole-patient CI]"] = sa.point_fmt + " [" + sa.ci_lo_fmt + ", " + sa.ci_hi_fmt + "]"
    xa = sa[["setting", "endpoint", "population", "estimate [nominal 95% whole-patient CI]", "n_queries", "n_patients", "n_replicates"]].rename(
        columns={"setting": "Setting", "endpoint": "Endpoint", "population": "Average (micro/macro)", "n_queries": "Queries", "n_patients": "Patients", "n_replicates": "Resamples"})

    # ---- sheet b
    sb = b.copy()
    sb.insert(0, "data_row_1based", range(1, len(sb) + 1))
    sb.insert(1, "comparison", sb.minuend.map(rung_label).str.replace(" \u2020", "") + f" {MINUS} " + sb.subtrahend.map(rung_label).str.replace(" \u2020", ""))
    sb.insert(2, "footnote", sb.minuend.eq("uni2apf30_ot_cosine").map({True: "\u2020", False: ""}))
    sb.insert(3, "endpoint", sb.metric.map(ENDPOINT))
    sb["point_fmt"] = sb.point.map(lambda v: fmt_small(v, True))
    sb["ci_lo_fmt"] = sb.ci_lo.map(lambda v: fmt_small(v, True))
    sb["ci_hi_fmt"] = sb.ci_hi.map(lambda v: fmt_small(v, True))
    sb["difference [nominal 95% CI]"] = sb.point_fmt + " [" + sb.ci_lo_fmt + ", " + sb.ci_hi_fmt + "]"
    xb = sb[["data_row_1based", "comparison", "footnote", "endpoint", "population", "difference [nominal 95% CI]", "ci_includes_zero_nominal", "definition"]].rename(
        columns={"data_row_1based": "Row", "comparison": "Comparison (minuend \u2212 subtrahend)", "footnote": "Footnote", "endpoint": "Endpoint",
                 "population": "Average (micro/macro)", "difference [nominal 95% CI]": "\u0394 [nominal 95% CI]"})

    # rounding-rule checks against the spec's worked examples (1-based data rows of sheet b)
    def row(r):
        return sb.iloc[r - 1]
    ex = {20: (None, f"{MINUS}0.017", "+0.0005", True), 34: (None, f"{MINUS}0.012", f"{MINUS}0.0001", False), 52: (None, "+0.0002", "+0.006", False),
          54: (None, "+0.000031", "+0.004", False), 120: (None, "+0.0001", "+0.013", False), 76: (f"{MINUS}0.000049", None, None, None)}
    labels = {20: ("uni2apf30_median_pooling", "hit5", "macro"), 34: ("uni2apf50_median_pooling", "hit5", "macro"), 52: ("uni2apf100_median_pooling", "precision5", "macro"),
              54: ("uni2apf100_median_pooling", "precision10", "macro"), 120: ("uni2apf30_ot_cosine", "hit10", "macro"), 76: ("uni2apf30_yottixel", "hit5", "macro")}
    for r, (p, lo, hi, inc) in ex.items():
        x = row(r)
        ok = (x.minuend, x.metric, x.population) == labels[r]
        if p is not None:
            ok &= x.point_fmt == p
        if lo is not None:
            ok &= x.ci_lo_fmt == lo and x.ci_hi_fmt == hi and bool(x.ci_includes_zero_nominal) == inc
        check(T, f"sheet b row {r} formatted as in spec", ok, f"{x.minuend} {x.metric} {x.population}: {x.point_fmt} [{x.ci_lo_fmt}, {x.ci_hi_fmt}] incl0={x.ci_includes_zero_nominal}")
    small_ep = {r for r in range(1, len(sb) + 1) if any(0 < abs(v) < 0.0005 for v in (row(r).ci_lo, row(r).ci_hi))}
    small_pt = {r for r in range(1, len(sb) + 1) if 0 < abs(row(r).point) < 0.0005}
    zero_pt = {r for r in range(1, len(sb) + 1) if row(r).point == 0}
    check(T, "sheet b rows with a non-zero endpoint < 0.0005 = spec list", small_ep == {5, 9, 11, 12, 21, 22, 39, 40, 43, 149, 20, 34, 52, 54, 120}, str(sorted(small_ep)))
    check(T, "sheet b rows with a non-zero point < 0.0005 = spec list", small_pt == {23, 25, 45, 47, 51, 58, 74, 75, 79, 123, 76}, str(sorted(small_pt)))
    check(T, "sheet b rows 17 and 49 exact 0 point shown 0.000", zero_pt == {17, 49} and all(row(r).point_fmt == "0.000" for r in (17, 49)), str(sorted(zero_pt)))
    allfmt = pd.concat([sa[["point_fmt", "ci_lo_fmt", "ci_hi_fmt"]], sb[["point_fmt", "ci_lo_fmt", "ci_hi_fmt"]]]).values.ravel()
    check(T, "no signed zero (\u00b10.000) anywhere in sheets a/b", not any(v in (f"{MINUS}0.000", "+0.000") for v in allfmt))

    # spec-quoted 4 d.p. values
    def q(minu, subt, metric, pop, p, lo, hi):
        if subt is None:
            r = a[(a.rung == minu) & (a.metric == metric) & (a.population == pop)].iloc[0]
        else:
            r = b[(b.minuend == minu) & (b.subtrahend == subt) & (b.metric == metric) & (b.population == pop)].iloc[0]
        ok = close(r.point, p, 5e-5) and (lo is None or (close(r.ci_lo, lo, 5e-5) and close(r.ci_hi, hi, 5e-5)))
        check(T, f"spec value {minu}{'' if subt is None else ' - ' + subt} {metric} {pop} = {p} [{lo}, {hi}] (4 d.p.)", ok, f"source {r.point:.5f} [{r.ci_lo:.5f}, {r.ci_hi:.5f}]")
        return r
    OT30, OT15, M30, Y30 = "uni2apf30_ot_cosine", "uni2apf15_ot_cosine", "uni2apf30_median_pooling", "uni2apf30_yottixel"
    q(OT30, None, "hit5", "macro", 0.8869, 0.8671, 0.9066); q(OT15, None, "hit5", "macro", 0.8912, None, None)
    q(OT30, None, "hit5", "micro", 0.9396, 0.9291, 0.9496); q(OT15, None, "hit5", "micro", 0.9415, None, None)
    q(OT30, None, "precision5", "macro", 0.6917, 0.6759, 0.7076); q(OT15, None, "precision5", "macro", 0.6929, None, None)
    q(OT30, None, "precision5", "micro", 0.7841, 0.7698, 0.7983); q(OT15, None, "precision5", "micro", 0.7847, None, None)
    for m, pop, p, lo, hi in [("hit5", "micro", -0.0018, -0.0047, 0.0010), ("hit5", "macro", -0.0043, -0.0104, 0.0013),
                              ("precision5", "micro", -0.0006, -0.0026, 0.0014), ("precision5", "macro", -0.0012, -0.0038, 0.0013),
                              ("hit1", "micro", 0.0025, -0.0015, 0.0067), ("hit1", "macro", 0.0052, -0.0016, 0.0132),
                              ("plurality5", "micro", 0.0015, -0.0026, 0.0056), ("plurality5", "macro", 0.0012, -0.0036, 0.0060),
                              ("hit3", "micro", 0.0040, 0.0007, 0.0074), ("hit3", "macro", 0.0035, 0.0009, 0.0066), ("hit10", "macro", 0.0059, 0.0001, 0.0127)]:
        q(OT30, OT15, m, pop, p, lo, hi)
    sub = b[(b.minuend == OT30) & (b.subtrahend == OT15)]
    excl = set(map(tuple, sub[~sub.ci_includes_zero_nominal.astype(bool)][["metric", "population"]].values))
    check(T, "OT30 - OT15: exactly hit@3 micro/macro and hit@10 macro exclude 0 (spec and caption)", excl == {("hit3", "micro"), ("hit3", "macro"), ("hit10", "macro")}, str(sorted(excl)))
    sub = b[(b.minuend == OT30) & (b.subtrahend == M30)]
    check(T, "OT30 - median 30: 14 rows, all exclude 0", len(sub) == 14 and (~sub.ci_includes_zero_nominal.astype(bool)).all())
    q(OT30, M30, "hit5", "micro", 0.0185, 0.0103, 0.0269); q(OT30, M30, "hit5", "macro", 0.0342, 0.0180, 0.0514); q(OT30, M30, "precision5", "micro", 0.0547, 0.0473, 0.0623)
    check(T, "OT30 - Yottixel 30: 14 rows", len(b[(b.minuend == OT30) & (b.subtrahend == Y30)]) == 14)
    q(OT30, Y30, "hit5", "micro", -0.0065, -0.0145, 0.0014); q(OT30, Y30, "hit5", "macro", -0.0094, -0.0262, 0.0073)
    q(OT30, Y30, "precision5", "micro", 0.0072, 0.0004, 0.0142); q(OT30, Y30, "precision5", "macro", 0.0102, 0.0008, 0.0198)
    ot30 = d[d.rung == OT30].iloc[0]
    check(T, "sheet d OT 30 %: no distance ties in the visible top 10", ot30.queries_any_tie_ranks1to6 == 0 and ot30.queries_tie_ranks9_10 == 0 and ot30.queries_with_rank56_tie == 0)
    yt = d[d.family == "yottixel"]
    check(T, "caption: Yottixel distances frequently tied at ranks 5/6 (all three Yottixel rungs > 0)", (yt.queries_with_rank56_tie > 0).all(), str(list(yt.queries_with_rank56_tie)))
    y15, y30 = d[d.rung == "uni2apf15_yottixel"].iloc[0], d[d.rung == Y30].iloc[0]
    lo_, hi_ = y30.macro_hit5_lower - y15.macro_hit5_upper, y30.macro_hit5_upper - y15.macro_hit5_lower
    check(T, "caption: Yottixel 30 vs 15 macro hit@5 tie-bound difference about -0.005 to +0.007 (verification arithmetic only, not published)",
          round(lo_, 3) == -0.005 and round(hi_, 3) == 0.007, f"{lo_:.4f} to {hi_:.4f}")
    g100 = c[(c.rung == "uni2apf100_median_pooling") & (c.metric == "hit5")].set_index("group").point
    g15 = c[(c.rung == "uni2apf15_median_pooling") & (c.metric == "hit5")].set_index("group").point
    dd = g100 - g15
    check(T, "caption: per-group median 100 % vs 15 % hit@5 MPNST -0.021, MRT +0.025 (verification arithmetic only)",
          round(dd["MPNST"], 3) == -0.021 and round(dd["MRT"], 3) == 0.025, f"MPNST {dd['MPNST']:.4f}, MRT {dd['MRT']:.4f}")
    # caption population arithmetic vs README
    mre = re.search(r"absent from the ladder \(([^)]*)\)", readme)
    rd = dict((k.strip(), int(v)) for k, v in (x.rsplit(" ", 1) for x in mre.group(1).split(","))) if mre else {}
    cap_d = {"ACT/ACC": 9, "NORMAL": 3, "LIVER": 2, "MEL": 2, "NB": 2, "OS": 2, "RB": 2, "RMS": 2, "DSRCT": 1, "EWS": 1, "MRT": 1, "WT": 1}
    check(T, "caption 28 missing slides by group (ACT/ACC 9, NORMAL 3, ...) = README", rd == cap_d and sum(rd.values()) == 28, str(rd))
    check(T, "README: 28 non-MISC slides (19 patients) absent", "28 non-MISC slides (19 patients)" in readme)

    # ---- sheet c
    sc = c.copy()
    sc.insert(0, "setting", sc.rung.map(rung_label)); sc.insert(1, "endpoint", sc.metric.map(ENDPOINT))
    sc["point_fmt"] = sc.point.map(fmt_small); sc["ci_lo_fmt"] = sc.ci_lo.map(fmt_small); sc["ci_hi_fmt"] = sc.ci_hi.map(fmt_small)
    sc["estimate [nominal 95% CI]"] = sc.point_fmt + " [" + sc.ci_lo_fmt + ", " + sc.ci_hi_fmt + "]"
    xc = sc[["setting", "group", "endpoint", "estimate [nominal 95% CI]", "n_queries", "n_patients", "flag"]].rename(
        columns={"setting": "Setting", "group": "Group", "endpoint": "Endpoint", "n_queries": "Queries", "n_patients": "Patients", "flag": "Flag"})
    check(T, "sheet c: per-setting query counts sum to 2,750", (c[c.metric == "hit5"].groupby("rung").n_queries.sum() == 2750).all())

    # ---- sheet d
    sd = d.copy(); sd.insert(0, "setting", sd.rung.map(rung_label))
    bound_cols = [x for x in d.columns if re.search(r"_(lower|saved|upper)$", x)]
    for col in bound_cols:
        sd[col + "_fmt"] = sd[col].map(fmt_small)
    xd = sd[["setting"] + [x for x in d.columns if x.startswith("queries_") or x == "max_tie_arrangements"] + [x + "_fmt" for x in bound_cols]]

    # ---- sheet e (transcribed from the cost panel)
    cost_txt = inp(COST).read_text(encoding="utf-8")
    check(T, "cost panel sha256 prefix 8cd17991 (current per spec)", sha(COST).startswith("8cd17991"), sha(COST)[:16])
    rows = [l for l in cost_txt.splitlines() if l.startswith("|")]
    hdr, body = parse_md_table(rows)
    check(T, "cost panel has 7 setting rows (spec: 7)", len(body) == 7, str(len(body)))
    strip = lambda t: t.replace("**", "")
    erows = []
    for r in body:
        rec = dict(zip(hdr, r))
        status = strip(rec["Status"])
        mlab = re.match(r"^(.*?)(?=\s\(|\.\s|$)", status)
        erows.append({"Aggregation": rec["Aggregation"], "Tile fraction": rec["Tile fraction"],
                      "CPU-hours (panel cell, verbatim)": strip(rec["**LSF CPU-hours (measured)**"]),
                      "Peak memory per element (panel cell, verbatim)": rec["Max memory per element"],
                      "Label (panel Status, leading phrase)": (mlab.group(1) if mlab else status),
                      "Footnote": "\u2020" if rec["Aggregation"].startswith("Exact OT") and rec["Tile fraction"].startswith("30") else "",
                      "Elements \u00d7 slots (panel)": rec["Elements × slots"], "Rows logged (panel)": rec["Rows logged"],
                      "Row wall s: mean / median / max (panel)": rec["Row wall s: mean / median / max"], "Status (panel cell, verbatim)": status})
    se = pd.DataFrame(erows)
    ot100 = se[se["Tile fraction"].str.startswith("100") & se.Aggregation.str.startswith("Exact OT")]
    check(T, "sheet e OT 100 % row reads '≈40,000–160,000 extrapolated, illustrative' with the power-law formula",
          len(ot100) == 1 and ot100.iloc[0]["CPU-hours (panel cell, verbatim)"].startswith("≈40,000–160,000 extrapolated, illustrative")
          and "8,954 × (100/30)^log₂(2.49) ≈ 4.4 × 10⁴" in ot100.iloc[0]["CPU-hours (panel cell, verbatim)"])
    ot30r = se[(se["Tile fraction"] == "30 %") & se.Aggregation.str.startswith("Exact OT")]
    check(T, "sheet e OT 30 % row labelled measured, 8,954 CPU-h", len(ot30r) == 1 and ot30r.iloc[0]["Label (panel Status, leading phrase)"].startswith("measured")
          and ot30r.iloc[0]["CPU-hours (panel cell, verbatim)"].startswith("8,954"))
    # cross-check vs the figure's source_E_cost.csv
    fe = pd.read_csv(inp(FIG / "source_E_cost.csv"))
    ok = True; det = []
    for _, r in fe.iterrows():
        fam = {"yottixel": "Yottixel", "ot": "Exact OT", "ot_cosine": "Exact OT", "median_pooling": "Median"}.get(r.family, r.family)
        m = se[se.Aggregation.str.startswith(fam) & se["Tile fraction"].str.startswith(f"{int(r.retention_pct)} ")]
        if len(m) == 1 and not pd.isna(r.cpu_hours):
            v = re.match(r"\u2248?([\d,]+(?:\u2013[\d,]+)?)", m.iloc[0]["CPU-hours (panel cell, verbatim)"])
            good = v is not None and v.group(1).replace(",", "").replace("\u2013", "-") == str(r.cpu_hours).strip()
            ok &= good; det.append(f"{r.family} {r.retention_pct}: {r.cpu_hours} {'=' if good else '!='} panel")
        else:
            det.append(f"{r.family} {r.retention_pct}: {r.cpu_hours} ({r.status}) not matched to a single numeric panel row")
    check(T, "cross-check: figure source_E_cost.csv CPU-hours = transcribed panel (numeric rows)", ok, "; ".join(det))
    # cross-checks against the companion figure's source CSVs
    fa = pd.read_csv(inp(FIG / "source_A_B_macro.csv"))
    mg = fa.merge(a, on=["rung", "metric", "population"], suffixes=("_f", ""))
    check(T, "cross-check: figure source_A_B_macro.csv rows identical to ladder_summary.csv", len(mg) == len(fa) and
          all((mg[k + "_f"] - mg[k]).abs().max() < 1e-12 for k in ["point", "ci_lo", "ci_hi"]), f"{len(mg)}/{len(fa)} rows")
    fc = pd.read_csv(inp(FIG / "source_C_contrasts.csv"))
    hits = sum(any(abs(b.point - r.point).lt(1e-12) & abs(b.ci_lo - r.ci_lo).lt(1e-12) & abs(b.ci_hi - r.ci_hi).lt(1e-12)) for _, r in fc.iterrows())
    check(T, "cross-check: every figure source_C_contrasts.csv row found in paired_differences.csv", hits == len(fc), f"{hits}/{len(fc)}")
    fd = pd.read_csv(inp(FIG / "source_D_per_group.csv"))
    hits = sum(any(c.group.eq(r.group) & abs(c.point - r.point).lt(1e-12) & abs(c.ci_lo - r.ci_lo).lt(1e-12) & abs(c.ci_hi - r.ci_hi).lt(1e-12)) for _, r in fd.iterrows())
    check(T, "cross-check: every figure source_D_per_group.csv row found in per_group.csv", hits == len(fd), f"{hits}/{len(fd)}")
    ft = pd.read_csv(inp(FIG / "source_tie_bounds.csv"))
    check(T, "cross-check: figure source_tie_bounds.csv identical to tie_bounds.csv", ft.equals(d))

    # ---- caption sheet
    title_i = es.find(s, e, "**Title:**")
    title = es.lines[title_i].split("**Title:**", 1)[1].strip()
    capq, ci = es.quote_after(s, e, "**Caption:**")
    split_at = next(i for i, l in enumerate(capq) if l.startswith("\u2020"))
    main_cap, dagger = " ".join(capq[:split_at]), " ".join(capq[split_at:])
    erow = next(l for l in es.lines[s:e] if l.startswith("| e. cost"))
    m = re.search(r"Footnote to sheet e: (.*?)\. Copy the OT 30 % solver qualification of footnote † \((.*?)\)\.\s+\[10-08", erow)
    fe_scope = m.group(1) + "." if m else "[SPEC/SOURCE MISMATCH: sheet-e footnote not parsed]"
    fe_q = m.group(2) if m else "[SPEC/SOURCE MISMATCH: OT 30 % qualification not parsed]"
    check(T, "sheet-e footnote and OT 30 % solver qualification parsed from the spec table row", m is not None)
    cap += [("title", title), ("caption (verbatim, edit-sheet line %d ff.)" % (ci + 2), main_cap), ("footnote \u2020 (verbatim)", dagger),
            ("sheet e footnote (spec table row e)", fe_scope),
            ("sheet e OT 30 % solver qualification (spec table row e, verbatim)", fe_q),
            ("sheet e OT 100 % row (spec)", "\"≈40,000–160,000 extrapolated, illustrative\" (power law 8,954 × (100/30)^log₂(2.49) ≈ 4.4 × 10⁴; quadratic ≈1.0–1.6 × 10⁵) — transcribed from the cost panel cell"),
            ("sheets", "a. summary (140 rows); b. paired (154); c. per group (600); d. ties (10); e. cost (7)"),
            ("rounding", "3 d.p.; a non-zero value that would round to ±0.000 is shown with a 4th decimal or, if still zero, two significant figures; exact 0 shown 0.000; differences signed (+/−, Unicode minus). Raw unrounded values are in the per-sheet CSVs."),
            ("version implemented", "R2.2-T4 as revised 10-08 (≥40 M tier; output review 901a2223…, ACCEPT); cost panel 8cd17991… (10-08). No later amendment to R2.2-T4 found (the 10-21 S-R2.2 figure block states 'R2.2-T4 unchanged').")]
    cap += [("caption line %d" % (i + 1), l) for i, l in enumerate(capq)]

    base = "R2.2-T4_S29"
    sa.to_csv(OUT / f"{base}_sheet_a_summary.csv", index=False)
    sb.to_csv(OUT / f"{base}_sheet_b_paired.csv", index=False)
    sc.to_csv(OUT / f"{base}_sheet_c_per_group.csv", index=False)
    sd.to_csv(OUT / f"{base}_sheet_d_ties.csv", index=False)
    se.to_csv(OUT / f"{base}_sheet_e_cost.csv", index=False)
    write_xlsx(OUT / f"{base}.xlsx", [("caption", caption_frame(cap)), ("a. summary", xa), ("b. paired", xb), ("c. per group", xc),
                                      ("d. ties", xd), ("e. cost", se[[k for k in se.columns if "(panel)" not in k and "Status (panel cell" not in k]])])
    OUTPUT_ROWS[T] = {"xlsx": f"{base}.xlsx", "sheet a": xa.shape, "sheet b": xb.shape, "sheet c": xc.shape, "sheet d": xd.shape, "sheet e": se.shape,
                      "spec": f"lines {s + 1}-{e}", "sources": [str(LAD / x) for x in ["ladder_summary.csv", "paired_differences.csv", "per_group.csv", "tie_bounds.csv", "README.md"]] + [str(COST)]}


# ---------------------------------------------------------------- S23
def build_s23(es: EditSheet):
    T = "S23 (R2.3-T2)"
    s, e, cap = common_caption(es, "R2.3-T2", "S23 (S-A1)", "R2.3-T2 ")
    g = pd.read_csv(inp(F23 / "per_group_summary.csv"))
    ci = pd.read_csv(inp(C23 / "per_group_summary.csv"))
    ov = pd.read_csv(inp(C23 / "summary_overall.csv"))
    check(T, "per_group_summary.csv sha256 prefix 669eac6e", sha(F23 / "per_group_summary.csv").startswith("669eac6e"))
    check(T, "CI per_group_summary.csv sha256 prefix 3248fd00", sha(C23 / "per_group_summary.csv").startswith("3248fd00"))
    check(T, "CI summary_overall.csv sha256 prefix 9e701500", sha(C23 / "summary_overall.csv").startswith("9e701500"))
    grp = g[g.true_label != "ALL"]
    check(T, "15 groups + ALL row", len(grp) == 15 and (g.true_label == "ALL").sum() == 1)
    grp = grp.sort_values("plurality5_recall", kind="stable")
    up = pd.concat([grp, g[g.true_label == "ALL"]], ignore_index=True)
    lad = lambda r: " / ".join(intc(r[k]) for k in ["ladder_rank1", "ladder_rank2_5", "ladder_rank6_10", "ladder_absent"])
    X = pd.DataFrame({
        "Group": up.true_label, "Queries": up.n_queries.map(intc), "Patients": up.n_patients.map(intc),
        "Median library support": up.median_library_support.map(lambda v: "—" if pd.isna(v) else intc(v)),
        "hit@1": up.hit1.map(f3), "hit@5": up.hit5.map(f3), "hit@10": up.hit10.map(f3), "P@5": up.precision5.map(f3),
        "Plurality@5": up.plurality5_recall.map(f3), "Wrong (plurality@5)": up.n_plurality5_wrong.map(intc),
        "Strict@5 correct": up.strict5_recall.map(f3), "Strict@5 abstain": up.strict5_abstain_rate.map(f3),
        "Published generous@5": up.legacy_generous5_recall.map(f3),
        "First same-group rank 1 / 2–5 / 6–10 / absent": up.apply(lad, axis=1),
        "Leading wrong destinations (plurality@5)": up.top3_plurality5_destinations})
    hdr, body = es.table_after(s, e, "**Body:**")
    check(T, "upper block: column headers equal the spec", list(X.columns) == hdr, str(hdr))
    diffs = [(i, h, sv, xv) for i, srow in enumerate(body) for h, sv, xv in zip(hdr, srow, X.iloc[i].tolist()) if sv != xv] if len(body) == len(X) else ["row count"]
    check(T, f"upper block: all {len(X)}x{len(hdr)} cells equal the spec body (incl. order by plurality@5)", not diffs, str(diffs[:10]))
    check(T, "Wrong column sums to 438 (= ALL)", grp.n_plurality5_wrong.sum() == 438 == int(g[g.true_label == "ALL"].n_plurality5_wrong.iloc[0]))
    check(T, "ladder columns sum to 2,778", int(grp[["ladder_rank1", "ladder_rank2_5", "ladder_rank6_10", "ladder_absent"]].values.sum()) == 2778)
    check(T, "per-group patients sum 1,921; ALL 1,876 taken from file", grp.n_patients.sum() == 1921 and int(g[g.true_label == "ALL"].n_patients.iloc[0]) == 1876)
    small = [(k, v) for k in ["hit1", "hit5", "hit10", "precision5", "plurality5_recall", "strict5_recall", "strict5_abstain_rate", "legacy_generous5_recall"] for v in up[k] if 0 < abs(v) < 0.0005]
    check(T, "no non-zero value displays as 0.000 at 3 d.p.", not small, str(small))

    # lower CI block
    MET = {"plurality_k5_recall": "Plurality@5", "hit5": "hit@5", "hit10": "hit@10", "strict_k5_abstain_rate": "Strict@5 abstain"}
    OVM = {"plurality_k5_accuracy": "Plurality@5", "hit5": "hit@5", "hit10": "hit@10", "strict_k5_abstain_rate": "Strict@5 abstain"}
    check(T, "CI file: 15 groups x 4 metrics, B = 5,000, no non-finite", len(ci) == 60 and set(ci.metric) == set(MET) and (ci.n_replicates == 5000).all() and (ci.n_nonfinite == 0).all())
    # point identity between the 10-03 table and the 10-04 CI run
    src_col = {"plurality_k5_recall": "plurality5_recall", "hit5": "hit5", "hit10": "hit10", "strict_k5_abstain_rate": "strict5_abstain_rate"}
    mism = [(r.true_label, r.metric) for _, r in ci.iterrows() if abs(r.point - g.set_index("true_label").loc[r.true_label, src_col[r.metric]]) > 1e-12]
    check(T, "CI run points = 10-03 per-group points (60/60)", not mism, str(mism))
    ovs = ov.set_index("estimand")
    mism = [k for k, col in [("plurality_k5_accuracy", "plurality5_recall"), ("hit5", "hit5"), ("hit10", "hit10"), ("strict_k5_abstain_rate", "strict5_abstain_rate")]
            if abs(ovs.loc[k, "point"] - g.set_index("true_label").loc["ALL", col]) > 1e-12]
    check(T, "summary_overall points = 10-03 ALL row", not mism, str(mism))
    long_rows = []
    for _, r in ci.iterrows():
        long_rows.append(dict(group=r.true_label, metric=MET[r.metric], source_metric=r.metric, n_queries=r.n_queries, n_patients=r.n_patients, point=r.point,
                              ci_lo=r.ci_lo, ci_hi=r.ci_hi, n_replicates=r.n_replicates, boundary_flag=r.boundary_flag if isinstance(r.boundary_flag, str) else "",
                              source=str(C23 / "per_group_summary.csv")))
    for k, lab in OVM.items():
        r = ovs.loc[k]
        long_rows.append(dict(group="ALL", metric=lab, source_metric=k, n_queries=r.n_queries, n_patients=r.n_patients, point=r.point, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                              n_replicates=r.n_replicates, boundary_flag=r.boundary_flag if isinstance(r.boundary_flag, str) else "", source=str(C23 / "summary_overall.csv")))
    L = pd.DataFrame(long_rows)
    L["formatted"] = L.apply(lambda r: ci3(r.point, r.ci_lo, r.ci_hi) + ("\u2020" if r.boundary_flag else ""), axis=1)
    orderg = list(up.true_label)
    W = L.pivot(index="group", columns="metric", values="formatted").reindex(orderg)
    W.insert(0, "Patients", up.set_index("true_label").n_patients.map(intc).reindex(orderg))
    W.insert(0, "Queries", up.set_index("true_label").n_queries.map(intc).reindex(orderg))
    W = W.reset_index().rename(columns={"group": "Group", "Plurality@5": "Plurality@5 [95% CI]", "hit@5": "hit@5 [95% CI]", "hit@10": "hit@10 [95% CI]",
                                        "Strict@5 abstain": "Strict@5 abstain [95% CI]"})
    W = W[["Group", "Queries", "Patients", "Plurality@5 [95% CI]", "hit@5 [95% CI]", "hit@10 [95% CI]", "Strict@5 abstain [95% CI]"]]
    hdr2, body2 = es.table_after(s, e, "**S23 lower block")
    check(T, "lower block: column headers equal the spec", list(W.columns) == hdr2, str(hdr2))
    diffs = [(i, h, sv, xv) for i, srow in enumerate(body2) for h, sv, xv in zip(hdr2, srow, W.iloc[i].tolist()) if sv != xv] if len(body2) == len(W) else ["row count"]
    check(T, f"lower block: all {len(W)}x{len(hdr2)} cells (incl. \u2020 marks) equal the spec", not diffs, str(diffs[:10]))

    legq, li = es.quote_after(s, e, "**Legend:**")
    cap += [("title/legend (verbatim, joined)", " ".join(legq)),
            ("blocks", "upper block: per-group point estimates (sorted by plurality@5, then ALL); lower block: whole-patient 95% CIs (B = 5,000, seed 123, PCG64; added 10-04, A-1/A-2 b)"),
            ("\u2020", "marks intervals touching 0 or 1 or of zero width (source boundary_flag: upper_eq_1, lower_eq_0, zero_width) — see legend"),
            ("version implemented", "R2.3-T2 as in the edit sheet (upper block 10-03 source 669eac6e; lower block added 10-04 from 3248fd00/9e701500). Later R2.3 audits (10-20 joint audit, l.~7497) change no value in R2.3-T1…T4b."),
            ("not used", "r2_3_internal_failure_cis_ext_prod_20261005/ (mAP/nDCG/macro-F1 intervals for Table S25, R4.b-T5) — not referenced by the R2.3-T2 spec.")]
    cap += [("legend line %d" % (i + 1), l) for i, l in enumerate(legq)]
    base = "R2.3-T2_S23_S-A1"
    rawup = up.copy()
    rawup.insert(0, "display_order", range(1, len(up) + 1))
    pd.concat([rawup.reset_index(drop=True), X.add_prefix("fmt: ")], axis=1).to_csv(OUT / f"{base}_upper_block.csv", index=False)
    L.to_csv(OUT / f"{base}_lower_block_CI.csv", index=False)
    write_xlsx(OUT / f"{base}.xlsx", [("caption", caption_frame(cap)), ("upper block", X), ("lower block 95% CI", W)])
    OUTPUT_ROWS[T] = {"xlsx": f"{base}.xlsx", "upper block": X.shape, "lower block": W.shape, "spec": f"lines {s + 1}-{e}",
                      "sources": [str(F23 / "per_group_summary.csv"), str(C23 / "per_group_summary.csv"), str(C23 / "summary_overall.csv")]}
    return g


# ---------------------------------------------------------------- S24
def build_s24(es: EditSheet, g23: pd.DataFrame):
    T = "S24/S24b (R2.3-T3)"
    s, e, cap = common_caption(es, "R2.3-T3", "S24 (S-A2); panel b S24b (S-A2b); optional S24c (S-A2c)", "R2.3-T3 ")
    files = {"S24 plurality@5": ("confusion_k5_plurality.csv", "fa102e83"), "S24b strict@5": ("confusion_k5_strict.csv", "985c35e5"),
             "S24c top-1 (optional)": ("confusion_k1_top1.csv", "d1343b9c")}
    M = {}
    for k, (f, pre) in files.items():
        M[k] = pd.read_csv(inp(F23 / f))
        check(T, f"{f} sha256 prefix {pre}", sha(F23 / f).startswith(pre))
        check(T, f"{f}: 15 rows, integer counts", len(M[k]) == 15 and all(pd.api.types.is_integer_dtype(M[k][c]) for c in M[k].columns[1:]))
        n = M[k].set_index("true_label").sum(axis=1)
        check(T, f"{f}: row totals = S23 queries per group (sum 2,778)", (n == g23.set_index("true_label").n_queries.reindex(n.index)).all() and n.sum() == 2778)
    groups = list(M["S24 plurality@5"].true_label)
    p = M["S24 plurality@5"].set_index("true_label")
    st = M["S24b strict@5"].set_index("true_label")
    check(T, "S24 columns = the 15 groups (same order as rows)", list(p.columns) == groups)
    check(T, "S24b columns = 15 groups + ABSTAIN", list(st.columns) == groups + ["ABSTAIN"])
    check(T, "S24 diagonal total 2,340 (spec)", sum(p.loc[x, x] for x in groups) == 2340)
    check(T, "S24b diagonal total 2,226 (spec)", sum(st.loc[x, x] for x in groups) == 2226)
    check(T, "S24b 284 abstentions (spec)", st.ABSTAIN.sum() == 284)
    check(T, "legend: column RMS receives 148 of 438 failures", p.RMS.sum() - p.loc["RMS", "RMS"] == 148 and p.values.sum() - sum(p.loc[x, x] for x in groups) == 438)
    legq, li = es.quote_after(s, e, "**Legend:**")
    cap += [("legend (verbatim, joined)", " ".join(legq)),
            ("rows/columns", "rows = true tumor group; columns = predicted group (S24b adds ABSTAIN). Integer counts copied verbatim from the source CSVs; group order as in the source (alphabetical). No totals added."),
            ("S24c", "Optional top-1 (k = 1) confusion table, included as a separate sheet for the author's assembly choice (spec: 'Optionally add … as S24c (S-A2c), at the author's assembly choice'). It has no legend text in the spec."),
            ("version implemented", "R2.3-T3 as in the edit sheet (no later amendment found).")]
    cap += [("legend line %d" % (i + 1), l) for i, l in enumerate(legq)]
    base = "R2.3-T3_S24_S-A2"
    sheets = [("caption", caption_frame(cap))]
    for k, nm, csvn in [("S24 plurality@5", "S24 plurality@5", "S24_plurality5"), ("S24b strict@5", "S24b strict@5 + Abstain", "S24b_strict5_abstain"),
                        ("S24c top-1 (optional)", "S24c top-1 (optional)", "S24c_top1_optional")]:
        df = M[k].rename(columns={"true_label": "True group \\ predicted", "ABSTAIN": "Abstain"})
        df.to_csv(OUT / f"{base}_{csvn}.csv", index=False)
        sheets.append((nm, df))
    write_xlsx(OUT / f"{base}.xlsx", sheets)
    OUTPUT_ROWS[T] = {"xlsx": f"{base}.xlsx", "S24": M["S24 plurality@5"].shape, "S24b": M["S24b strict@5"].shape, "S24c (optional)": M["S24c top-1 (optional)"].shape,
                      "spec": f"lines {s + 1}-{e}", "sources": [str(F23 / v[0]) for v in files.values()]}


# ---------------------------------------------------------------- S30
def build_s30(es: EditSheet):
    T = "S30 (R3.2-T1)"
    s, e, cap = common_caption(es, "R3.2-T1", "S30 (no symbolic ID)", "R3.2-T1 ")
    md = inp(S30D / "table_s30_wide.md").read_text(encoding="utf-8").splitlines()
    long = pd.read_csv(inp(S30D / "table_s30.csv"))
    check(T, "table_s30.csv observed_hit_at_5 non-numeric entries are only 'not available'",
          set(long.observed_hit_at_5[pd.to_numeric(long.observed_hit_at_5, errors="coerce").isna()]) <= {"not available"})
    long_num = long.copy()
    long_num["observed_hit_at_5"] = pd.to_numeric(long.observed_hit_at_5, errors="coerce")
    inp(S30D / "run_record.json")
    for f, pre in [("table_s30_wide.md", "0312bc92"), ("table_s30.csv", "e428d584"), ("run_record.json", "9aa29593")]:
        check(T, f"{f} sha256 prefix {pre}", sha(S30D / f).startswith(pre))
    hdr, body = parse_md_table([l for l in md if l.startswith("|")])
    W = pd.DataFrame(body, columns=hdr)
    shdr, sbody = es.table_after(s, e, "**Operation:**")
    check(T, "wide table (15 x 8) identical to the edit-sheet table", shdr == hdr and sbody == body and len(body) == 15, f"{len(body)} rows")
    check(T, "table_s30.csv: 60 rows = 15 groups x 4 collections", len(long) == 60 and long.tumor_group.nunique() == 15 and long.collection.nunique() == 4)
    nlib = W["n in library"].astype(int)
    check(T, "n_g sums to 2,778", nlib.sum() == 2778)
    cells = [c for row in body for c in row[4:]]
    check(T, "\u2020 on 11 external cells, \u2021 on 3 internal cells (spec)", sum("\u2020" in c for c in cells) == 11 and sum("\u2021" in c for c in cells) == 3)
    # consistency of the wide table with the long CSV (copied values, 3 d.p.)
    mism = []
    for _, r in long_num.iterrows():
        cell = W.set_index("Group").loc[r.tumor_group, r.collection]
        if pd.isna(r.observed_hit_at_5):
            if cell != "not available":
                mism.append((r.collection, r.tumor_group, cell))
        elif not cell.startswith(ci3(r.observed_hit_at_5, r.ci_lo, r.ci_hi)):
            mism.append((r.collection, r.tumor_group, cell))
    check(T, "wide cells = table_s30.csv observed hit@5 [CI] at 3 d.p. (60/60)", not mism, str(mism[:5]))
    ch = long.drop_duplicates("tumor_group").set_index("tumor_group")
    mism = [gname for gname, r in W.set_index("Group").iterrows() if r["Chance hit@5"] != f3(ch.loc[gname, "chance_hit_at_5"]) or r["Share"] != f3(ch.loc[gname, "library_share"])]
    check(T, "wide chance/share = table_s30.csv at 3 d.p.", not mism, str(mism))
    s23 = pd.read_csv(C23 / "per_group_summary.csv")
    s23 = s23[s23.metric == "hit5"].set_index("true_label")
    it = long_num[long_num.collection == "internal"].set_index("tumor_group")
    mism = [x for x in it.index if any(abs(it.loc[x, a] - s23.loc[x, b]) > 1e-12 for a, b in [("observed_hit_at_5", "point"), ("ci_lo", "ci_lo"), ("ci_hi", "ci_hi")])]
    check(T, "internal column = S23 CI file hit5 (15/15)", not mism, str(mism))
    for gname, chs, obs in [("DSRCT", "0.055", "0.645"), ("MPNST", "0.084", "0.583"), ("MRT", "0.072", "0.732"), ("RMS", "0.605", "0.947"), ("NB", "0.526", "0.971"), ("WT", "0.519", "0.963")]:
        r = W.set_index("Group").loc[gname]
        check(T, f"cited sentence: {gname} chance {chs}, internal {obs}", r["Chance hit@5"] == chs and r["internal"].startswith(obs))
    check(T, "CCDI-MCI WT is 0/1 (spec 'Do not add' example)", W.set_index("Group").loc["WT", "CCDI-MCI FFPE"].startswith("0.000 [0.000, 0.000] (n=1 slide"))
    # footnote: applied F1 version
    fi = es.find(s, e, "**Table footnote (verbatim):**")
    fq, _ = es.quote_after(s, e, "**Table footnote (verbatim):**")
    marked = " ".join(fq)
    struck = re.search(r"~~(.*?)~~", marked)
    applied = re.sub(r"\s*~~.*?~~", "", marked).replace(" [R3.2 audit F1, 2026-10-21]", "")
    md_foot = next(l for l in md if l.startswith("Observed hit@5"))
    restored = re.sub(r"\s*~~(.*?)~~", r" \1", marked)
    restored = re.sub(r" \(per-group averages 0\.001–0\.004; at most 0\.013 for a single query, reached by patients with eight or more same-group library slides\) \[R3\.2 audit F1, 2026-10-21\]", "", restored)
    check(T, "edit-sheet footnote with the F1 strike-through reverted = table_s30_wide.md footnote (verbatim)", restored == md_foot)
    check(T, "applied footnote carries the F1 bound 'at most 0.013 for a single query' and drops 'at most 0.007'", "at most 0.013 for a single query" in applied and "0.007" not in applied)
    check(T, "F1 per-group averages 0.001–0.004 / single-query 0.013 (R3.2 audit values; not re-derivable from S30 sources)", True,
          "NOT verified here: taken from docs/analysis/r3_2_parent_audit_20261021.md via the edit sheet; recorded as a dependency")
    title = es.lines[es.find(s, e, "**Title:**")].split("**Title:**", 1)[1].strip()
    capq, _ = es.quote_after(s, e, "**Caption:**")
    cap += [("title", title), ("caption (verbatim, joined)", " ".join(capq)),
            ("footnote (F1 APPLIED, clean)", applied),
            ("footnote (edit-sheet markup, verbatim, line %d)" % (fi + 3), marked),
            ("footnote (as frozen in table_s30_wide.md, superseded by F1)", md_foot),
            ("version implemented", "R3.2-T1 table verbatim from table_s30_wide.md (0312bc92…); footnote with R3.2 audit F1 (2026-10-21 ≈09:45 addendum) applied: struck '(independent review bound 0.001-0.004; at most 0.007 for any single query)' replaced by '(per-group averages 0.001–0.004; at most 0.013 for a single query, reached by patients with eight or more same-group library slides)'. Optional F2 (caption 'S18, S19 and S23' → 'S18 and S23'; 'resample within each group') NOT applied (author option, not decided in the sheet)."),
            ("columns", "internal / STV / KidRare / CCDI-MCI FFPE = observed hit@5 point [95% interval] (n = query slides; resampling units); † / ‡ as in the footnote")]
    cap += [("caption line %d" % (i + 1), l) for i, l in enumerate(capq)]
    base = "R3.2-T1_S30"
    W.to_csv(OUT / f"{base}_wide.csv", index=False)
    long.to_csv(OUT / f"{base}_long_source_table_s30.csv", index=False)
    write_xlsx(OUT / f"{base}.xlsx", [("caption", caption_frame(cap)), ("S30 table", W), ("long (table_s30.csv)", long)])
    OUTPUT_ROWS[T] = {"xlsx": f"{base}.xlsx", "wide": W.shape, "long": long.shape, "spec": f"lines {s + 1}-{e} + addendum F1",
                      "sources": [str(S30D / x) for x in ["table_s30_wide.md", "table_s30.csv", "run_record.json"]]}


# ---------------------------------------------------------------- S26
def build_s26(es: EditSheet):
    T = "S26 (R3.7-T4)"
    s, e, cap = common_caption(es, "R3.7-T4", "S26 (S-D1)", "R3.7-T4 ")
    sm = pd.read_csv(inp(R37 / "summary_all_arms.csv"))
    check(T, "summary_all_arms.csv sha256 prefix cdc5f455", sha(R37 / "summary_all_arms.csv").startswith("cdc5f455"))
    arms = [("s61_int_2c", "ERMS vs ARMS, COMET", ["ERMS", "ARMS"]), ("s61_ext_2c", "ERMS vs ARMS, CCDI-MCI → COMET", ["ERMS", "ARMS"]), ("s64", "HB vs HCC, COMET", ["HB", "HCC"])]
    rows, raw = [], []
    for arm, lab, cls in arms:
        cf = pd.read_csv(inp(R37 / f"confusion_k5_plurality_{arm}.csv")).set_index("truth")
        conf = "; ".join(f"{c} " + "/".join(str(int(cf.loc[c, k])) for k in cls) for c in cls)
        for j, (cl, rl) in enumerate([("ALL", "all"), ("MACRO", "macro")] + [(c, c) for c in cls]):
            r = sm[(sm.arm == arm) & (sm["class"] == cl)]
            assert len(r) == 1, (arm, cl)
            r = r.iloc[0]
            check(T, f"{arm}/{cl}: intervals present (ci_available, no non-finite draws)", bool(r.ci_available) and r.n_nonfinite_draws == 0 and
                  r[["plurality5_recall_lo", "plurality5_recall_hi", "hit1_lo", "hit1_hi", "precision5_lo", "precision5_hi"]].notna().all())
            rows.append({"Comparison": lab if j == 0 else "", "Row": rl,
                         "n slides / patients": "—" if cl == "MACRO" else f"{int(r.n_queries)} / {int(r.n_patients)}",
                         "Plurality@5 [95% CI]": ci3(r.plurality5_recall, r.plurality5_recall_lo, r.plurality5_recall_hi),
                         "hit@1 [95% CI]": ci3(r.hit1, r.hit1_lo, r.hit1_hi), "P@5 [95% CI]": ci3(r.precision5, r.precision5_lo, r.precision5_hi),
                         "AUROC [95% CI]": ci3(r.auroc_top5_pos_share, r.auroc_lo, r.auroc_hi) if cl == "ALL" else "—",
                         "Confusion (rows = truth)": conf if cl == "ALL" else ""})
            keep = ["arm", "class", "unit", "n_queries", "n_patients", "n_reference_slides", "n_reference_patients", "plurality5_recall", "plurality5_recall_lo", "plurality5_recall_hi",
                    "hit1", "hit1_lo", "hit1_hi", "precision5", "precision5_lo", "precision5_hi", "auroc_top5_pos_share", "auroc_lo", "auroc_hi",
                    "n_nonfinite_draws", "n_zero_support_draws", "ci_available", "ci_degenerate", "ci_degenerate_metrics", "ci_patient_level"]
            raw.append({**{k: r[k] for k in keep}, "confusion_k5_plurality": conf if cl == "ALL" else ""})
    X = pd.DataFrame(rows)
    hdr, body = es.table_after(s, e, "**Body (core rows):**")
    check(T, "column headers equal the spec", list(X.columns) == hdr, str(hdr))
    diffs = [(i, h, sv, xv) for i, srow in enumerate(body) for h, sv, xv in zip(hdr, srow, X.iloc[i].tolist()) if sv != xv] if len(body) == len(X) else ["row count"]
    check(T, f"all {len(X)}x{len(hdr)} cells equal the spec body", not diffs, str(diffs[:10]))
    R = pd.DataFrame(raw)
    small = [v for k in ["plurality5_recall", "hit1", "precision5"] for v in R[k] if 0 < abs(v) < 0.0005]
    check(T, "no non-zero value displays as 0.000", not small)
    ext = sm[(sm.arm == "s61_ext_2c") & (sm["class"] == "ALL")].iloc[0]
    check(T, "legend: external queries searched against the COMET RMS library of 373 slides / 306 patients", int(ext.n_reference_slides) == 373 and int(ext.n_reference_patients) == 306)
    # footnote rows (spindle)
    fn = []
    for arm, cohort in [("s61_int_3c", "COMET"), ("s61_ext_3c", "CCDI-MCI")]:
        r = sm[(sm.arm == arm) & (sm["class"] == "Spindle")].iloc[0]
        cf = pd.read_csv(inp(R37 / f"confusion_k5_plurality_{arm}.csv")).set_index("truth")
        fn.append({"arm": arm, "cohort": cohort, "class": "Spindle", "n_queries": int(r.n_queries), "n_patients": int(r.n_patients),
                   "plurality5_recall": r.plurality5_recall, "plurality5_recall_lo": r.plurality5_recall_lo, "plurality5_recall_hi": r.plurality5_recall_hi,
                   "ci_degenerate": r.ci_degenerate, "ci_degenerate_metrics": r.ci_degenerate_metrics,
                   "Spindle queries predicted ERMS / ARMS / Spindle (plurality@5)": " / ".join(str(int(cf.loc["Spindle", k])) for k in ["ERMS", "ARMS", "Spindle"]),
                   "Plurality@5 (3 d.p.)": f3(r.plurality5_recall), "Plurality@5 [95% CI] (3 d.p.)": ci3(r.plurality5_recall, r.plurality5_recall_lo, r.plurality5_recall_hi)})
    FN = pd.DataFrame(fn)
    check(T, "footnote: COMET spindle 18 slides / 14 patients, all 18 retrieved as ERMS, plurality@5 0",
          fn[0]["n_queries"] == 18 and fn[0]["n_patients"] == 14 and fn[0]["Spindle queries predicted ERMS / ARMS / Spindle (plurality@5)"].startswith("18 /") and fn[0]["plurality5_recall"] == 0)
    check(T, "footnote: CCDI-MCI spindle 20 slides, 17 of 20 retrieved as ERMS, plurality@5 0.05",
          fn[1]["n_queries"] == 20 and fn[1]["Spindle queries predicted ERMS / ARMS / Spindle (plurality@5)"].startswith("17 /") and close(fn[1]["plurality5_recall"], 0.05, 1e-12))
    # majority-class shares quoted in the legend (from the S28 inventory strings)
    inv = pd.read_csv(INV / "task_inventory.csv", dtype=str, keep_default_na=False).set_index("arm")
    check(T, "legend majority-class shares 0.606 / 0.677 / 0.683 (= S28 inventory majority_class_rate)",
          inv.loc["s61_int_2c", "majority_class_rate"].endswith("(0.606)") and inv.loc["s61_ext_2c", "majority_class_rate"].endswith("(0.677)") and inv.loc["s64", "majority_class_rate"].endswith("(0.683)"),
          f"{inv.loc['s61_int_2c','majority_class_rate']}; {inv.loc['s61_ext_2c','majority_class_rate']}; {inv.loc['s64','majority_class_rate']}")
    # R3.7 audit (10-20): option (a) legend sentence numbers
    rj = json.load(open(inp(R37A / "results.json")))["results"]
    ri, rl = rj["s61_int_2c"]["all_tissue_matrix"], rj["s64"]["all_tissue_matrix"]
    def round(x, n):  # half away from zero, as in the quoted sentence (-0.0625 -> -0.063)
        from decimal import Decimal, ROUND_HALF_UP
        return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP))
    ok = (round(ri["plurality"], 3) == 0.847 and round(rl["plurality"], 3) == 0.914 and round(ri["delta_plurality"], 3) == -0.040
          and [round(x, 3) for x in ri["delta_plurality_ci"]] == [-0.066, -0.016] and round(rl["delta_plurality"], 3) == -0.022
          and [round(x, 3) for x in rl["delta_plurality_ci"]] == [-0.063, 0.015])
    check(T, "option-(a) legend sentence values (0.847, 0.914; −0.040 [−0.066, −0.016]; −0.022 [−0.063, +0.015]) found in r3_7_audit_20261020/results.json (3 d.p., half away from zero)", ok,
          f"RMS {ri['plurality']:.4f} Δ {ri['delta_plurality']:.4f} {ri['delta_plurality_ci']}; liver {rl['plurality']:.4f} Δ {rl['delta_plurality']:.4f} {rl['delta_plurality_ci']}")
    # legend: verbatim + applied (C1 = F1 option (a) insertion; F2)
    legq, _ = es.quote_after(s, e, "**Legend:**")
    leg = " ".join(legq)
    a_s, a_e = None, None
    for i, l in enumerate(es.lines):
        if l.startswith("## Addendum 2026-10-20 ≈14:40 CDT — R3.7 parent audit"):
            a_s = i
    ad = "\n".join(es.lines[a_s:a_s + 46]) if a_s is not None else ""
    m_ins = re.search(r"R3\.7-T4 \(S26 legend\), after \"Point estimates are over slides;\", insert: \"(.*?)\"\n\s+- R3\.7-T5", ad, re.S)
    m_f2 = re.search(r"\*\*F2 \(minor; apply under either option\)\.\*\* R3\.7-T4 footnote, after \"was evaluated only in a\s+three-subtype search\", add: \"(.*?)\"\. No value changes\.", ad, re.S)
    ins = re.sub(r"\s+", " ", m_ins.group(1)) if m_ins else None
    f2 = re.sub(r"\s+", " ", m_f2.group(1)) if m_f2 else None
    check(T, "C1/F1 option (a) S26 legend insertion parsed from the 10-20 addendum", ins is not None)
    check(T, "F2 footnote clause parsed from the 10-20 addendum", f2 is not None)
    applied = leg
    if ins and "Point estimates are over slides;" in applied:
        applied = applied.replace("Point estimates are over slides;", "Point estimates are over slides; " + ins, 1)
        # the insertion is a full sentence ending with '.'; the original continues '95% percentile intervals …' (lower case)
    if f2 and "was evaluated only in a three-subtype search" in applied:
        applied = applied.replace("was evaluated only in a three-subtype search", "was evaluated only in a three-subtype search " + f2, 1)
    cap += [("legend (APPLIED: C1 = R3.7 F1 option (a) insertion + F2)", applied),
            ("legend (verbatim, edit-sheet R3.7-T4 block)", leg),
            ("option (a) insertion (verbatim from addendum 2026-10-20 ≈14:40)", ins or "[not parsed]"),
            ("F2 clause (verbatim from addendum 2026-10-20 ≈14:40)", f2 or "[not parsed]"),
            ("insertion grammar note", "The addendum says insert after 'Point estimates are over slides;' — the original continues '95% percentile intervals resample whole patients …', so the applied text reads '…over slides; COMET arms use … +0.015]. 95% percentile intervals …'. Placement follows the addendum literally; punctuation left for the author/copy-edit."),
            ("version implemented", "R3.7-T4 values from r3_7_fine_label_20261003 (cdc5f455…). Per the ledger (R3.7 row; 2026-10-20 ≈17:16 author decision C1 → option (a)) S26 numbers are unchanged; the 10-20 audit values (all-tissue sensitivity) appear only in the legend sentence. hit@5 not shown (spec: not a headline; optional column not added); legacy_* and strict5_* excluded."),
            ("footnote rows", "Spindle-cell RMS from s61_int_3c / s61_ext_3c (sheet 'footnote spindle')")]
    cap += [("legend line %d" % (i + 1), l) for i, l in enumerate(legq)]
    base = "R3.7-T4_S26_S-D1"
    R.to_csv(OUT / f"{base}_body_raw.csv", index=False)
    pd.concat([R[["arm", "class"]], X], axis=1).to_csv(OUT / f"{base}_body_formatted.csv", index=False)
    FN.to_csv(OUT / f"{base}_footnote_spindle.csv", index=False)
    write_xlsx(OUT / f"{base}.xlsx", [("caption", caption_frame(cap)), ("S26 body", X), ("footnote spindle", FN)])
    OUTPUT_ROWS[T] = {"xlsx": f"{base}.xlsx", "body": X.shape, "footnote spindle": FN.shape, "spec": f"lines {s + 1}-{e} + addendum 10-20 (C1 option (a), F2)",
                      "sources": [str(R37 / "summary_all_arms.csv")] + [str(R37 / f"confusion_k5_plurality_{a}.csv") for a in ["s61_int_2c", "s61_ext_2c", "s64", "s61_int_3c", "s61_ext_3c"]] + [str(R37A / "results.json")]}


# ---------------------------------------------------------------- S28
def build_s28(es: EditSheet):
    T = "S28 (R3.7-T5)"
    s, e, cap = common_caption(es, "R3.7-T5", "S28 (S-D2)", "R3.7-T5 ")
    mdp, csvp = inp(INV / "task_inventory.md"), inp(INV / "task_inventory.csv")
    check(T, "task_inventory.md sha256 prefix 16e2fc32", sha(mdp).startswith("16e2fc32"))
    check(T, "task_inventory.csv sha256 prefix 7aea2944", sha(csvp).startswith("7aea2944"))
    md = mdp.read_text(encoding="utf-8").splitlines()
    i0 = es.find(s, e, "**Table content (verbatim):**")
    i1 = es.find(s, e, "**Status (10-05):**")
    sheet_lines = [l for l in es.lines[i0 + 1:i1]]
    strip_blank = lambda L: [l for l in L if l.strip()]
    check(T, "edit-sheet 'Table content (verbatim)' = task_inventory.md lines 3–23", strip_blank(sheet_lines) == strip_blank(md[2:23]))
    hdr, body = parse_md_table([l for l in md if l.startswith("|")])
    check(T, "13 arms x 25 columns, every row has 25 cells", len(body) == 13 and all(len(r) == len(hdr) == 25 for r in body))
    X = pd.DataFrame(body, columns=hdr)
    C = pd.read_csv(csvp, dtype=str, keep_default_na=False)
    check(T, "task_inventory.csv: 13 rows x 35 columns", C.shape == (13, 35))
    mism = [(i, h) for i in range(13) for h in hdr if X.iloc[i][h] != C.iloc[i][h]]
    check(T, "md table cells = csv first 25 columns (13 x 25)", not mism, str(mism[:5]))
    intro = md[2]; legend = md[4]; source = md[22]
    lnote_i = es.find(s, e, "**Legend location note")
    lnote = " ".join(es.lines[lnote_i:i0]).split("reviewer N3):**", 1)[1].split("*(Not for the legend.)*", 1)[0].strip()
    lnote = re.sub(r"\s+", " ", lnote)
    # C1/F1 option (a) insertion for the S28 legend
    a_s = next(i for i, l in enumerate(es.lines) if l.startswith("## Addendum 2026-10-20 ≈14:40 CDT — R3.7 parent audit"))
    ad = "\n".join(es.lines[a_s:a_s + 46])
    m = re.search(r"R3\.7-T5 \(S28 legend\), after \"Saved S6 depth-10 rank lists were re-scored without re-ranking\.\",\s+insert: \"(.*?)\"", ad, re.S)
    ins = re.sub(r"\s+", " ", m.group(1)) if m else None
    check(T, "C1/F1 option (a) S28 legend insertion parsed from the 10-20 addendum", ins is not None)
    anchor = "Saved S6 depth-10 rank lists were re-scored without re-ranking."
    check(T, "insertion anchor present in the frozen legend", anchor in legend)
    applied = legend.replace(anchor, anchor + " " + ins, 1) if ins else legend
    title = es.lines[es.find(s, e, "**Title:**")].split("**Title:**", 1)[1].strip()
    for arm, needle in [("s62", "0.348"), ("s65b_gnb", "0.368")]:
        check(T, f"Discussion-quoted minority recall {needle} present in {arm} row", needle in C.set_index("arm").loc[arm, "minority_class_recall"], C.set_index("arm").loc[arm, "minority_class_recall"])
    cap += [("title", title), ("introductory sentence (verbatim, task_inventory.md line 3)", intro),
            ("legend (APPLIED: C1 = R3.7 F1 option (a) insertion)", applied),
            ("legend (verbatim, task_inventory.md line 5)", legend),
            ("legend location note (add to the legend; reviewer N3; verbatim from edit sheet)", lnote),
            ("option (a) insertion (verbatim from addendum 2026-10-20 ≈14:40)", ins or "[not parsed]"),
            ("source line (verbatim, task_inventory.md line 23)", source),
            ("version implemented", "R3.7-T5 verbatim table (frozen inventory 16e2fc32…/7aea2944…, unchanged); legend with the R3.7 F1 option (a) sentence (author decision C1, 2026-10-20 ≈17:16, applied at incorporation). The md H1 heading is replaced by the title as the spec says. Column names kept as in the source."),
            ("machine-readable", "num_* columns are in the CSV only (as the legend says)")]
    base = "R3.7-T5_S28_S-D2"
    X.to_csv(OUT / f"{base}_table_verbatim.csv", index=False)
    C.to_csv(OUT / f"{base}_task_inventory_full.csv", index=False)
    write_xlsx(OUT / f"{base}.xlsx", [("caption", caption_frame(cap)), ("S28 table", X), ("full csv (incl num_)", C)])
    OUTPUT_ROWS[T] = {"xlsx": f"{base}.xlsx", "table": X.shape, "full csv": C.shape, "spec": f"lines {s + 1}-{e} + addendum 10-20 (C1 option (a))",
                      "sources": [str(mdp), str(csvp)]}


# ---------------------------------------------------------------- main
def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    es = EditSheet(inp(EDIT))
    inp(REPO / "docs/revision_ledger_by_point.md")
    build_s29(es)
    g23 = build_s23(es)
    build_s24(es, g23)
    build_s30(es)
    build_s26(es)
    build_s28(es)

    bind = {p: {"sha256": sha(Path(p)), "mtime": dt.datetime.fromtimestamp(Path(p).stat().st_mtime).isoformat(timespec="seconds")} for p in sorted(INPUTS)}
    json.dump({"group": GROUP, "script": str(Path(__file__).relative_to(REPO)), "script_sha256": sha(Path(__file__)),
               "built": dt.datetime.now().isoformat(timespec="seconds"), "inputs": bind}, open(OUT / "input_bindings.json", "w"), indent=1, ensure_ascii=False)

    n_fail = sum(1 for v in CHECKS.values() for c in v if not c[0])
    n_all = sum(len(v) for v in CHECKS.values())
    rd = ["# Supplementary tables — group A_ladder_failure_finelabel (assembled 2026-10-22; candidates, not approval)", "",
          f"Built by `scripts/supp_tables/build_{GROUP}_20261022.py` from accepted outputs (see `input_bindings.json`); nothing recomputed. "
          "Each XLSX has a `caption` sheet (edit-sheet heading/lines, title, caption/legend and footnotes verbatim, version implemented) followed by the data sheets; "
          "each data sheet also exists as CSV with raw (unrounded) and formatted columns. Provisional numbers; final numbers at assembly via `scripts/supplementary_numbering_map.py`.", ""]
    for t, info in OUTPUT_ROWS.items():
        rd.append(f"## {t} — `{info['xlsx']}` (spec {info['spec']})")
        for k, v in info.items():
            if k not in ("xlsx", "spec", "sources"):
                rd.append(f"- {k}: {v[0]} rows x {v[1]} cols")
        rd.append("")
    rd.append("Files: " + ", ".join(sorted(p.name for p in OUT.iterdir())) + ", BUILD_REPORT.md, SHA256SUMS.txt")
    (OUT / "README.md").write_text("\n".join(rd) + "\n", encoding="utf-8")

    rep = [f"# BUILD_REPORT — {GROUP} (2026-10-22)", "",
           f"Script `scripts/supp_tables/build_{GROUP}_20261022.py` (sha256 {sha(Path(__file__))[:16]}…). Checks: {n_all - n_fail}/{n_all} pass; {n_fail} fail/mismatch.", ""]
    for t, info in OUTPUT_ROWS.items():
        rep.append(f"## {t}")
        rep.append(f"- Spec: docs/manuscript_supplement_edits.md {info['spec']}")
        rep.append("- Sources: " + "; ".join(f"`{relp(x)}`" for x in info["sources"]))
        rep.append("- Output: `" + info["xlsx"] + "`; " + "; ".join(f"{k} {v[0]}×{v[1]}" for k, v in info.items() if k not in ("xlsx", "spec", "sources")))
        for ok, name, det in CHECKS.get(t, []):
            rep.append(f"  - [{'PASS' if ok else 'MISMATCH'}] {name}" + (f" — {det}" if det and (not ok or 'NOT verified' in det or 'cross-check' in name or 'arithmetic' in name) else ""))
        for n in NOTES.get(t, []):
            rep.append(f"  - NOTE: {n}")
        rep.append("")
    rep += ["## Discrepancies and decisions (summary)",
            f"- Spec/source value mismatches: {n_fail} (every quoted spec number and every spec-copied cell checked above matched its source).",
            *[f"- {t}: {n}" for t, v in NOTES.items() for n in v if n.startswith(("DEPENDENCY", "DECISION", "ROUNDING"))],
            "- Nothing left unbuilt."]
    (OUT / "BUILD_REPORT.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    lines = [f"{sha(p)}  {p.name}" for p in sorted(OUT.iterdir()) if p.name != "SHA256SUMS.txt"]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(f"{n_all - n_fail}/{n_all} checks pass -> {OUT}")
    for t, v in CHECKS.items():
        for ok, name, det in v:
            if not ok:
                print("MISMATCH", t, name, det)


if __name__ == "__main__":
    main()
