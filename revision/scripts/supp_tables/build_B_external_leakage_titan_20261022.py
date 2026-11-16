#!/usr/bin/env python3
"""Assemble supplementary tables of group B_external_leakage_titan (assembly only; D-2 (a), 2026-10-22).

Tables, spec blocks (docs/manuscript_supplement_edits.md; line numbers are located at run time and
recorded in BUILD_REPORT.md) and sources (all under $PROSPER_OUT):

* Tables S18-S21 -- "### R1.3-T8 -- Tables S18-S21: exact source-to-publication transformations"
  (l.~1358); E20 synthesis `r1_3_external_synthesis_20260930/`: broad_group_primary,
  paired_primary_differences, coverage_and_hit_yield, expanded_secondary_retrieval_points,
  all_returned_label_slots, failure_conditioned_label_slots, first_relevant_rank_distribution,
  matched_endpoint_transitions, separate_coverage_losses, reference_recurrence, aggregate_context,
  descriptive_strata_points, wt_source_by_pattern_membership (.csv). Cross-check only: the revised
  Fig. 4 source data `figures_20261021/fig4_additional_collections/source_data.csv`.
* Table S-R4.a-1 -- "### R4.a-T4" (l.~5259); `r4a_vii_whole_patient_20261015/production/`
  {summary,paired_differences}.csv (arm c1e); Block B points: [internal record]
  (table), [internal record] (frozen masked generous values) and the unmasked
  C1e JSONs `c1e_slide_unmasked/`.
* Table S-R4.a-2 -- "### R4.a-T5" (l.~5315); `r4a_vii_whole_patient_20261015/production/` (arm c1d) and
  `production_c1d_same_class/` (arm c1d_same_class); per-pair frame used only to verify footnote counts.
* Table S22 (S-T1) -- "### R2.6-T7" (l.~2923); the seven `r2_6_titan_intersection_*_20261004/`
  {summary,per_class}_common_pool.csv; note checks from `r2_6_titan_common_pool_20261003/connectivity_per_group.csv`,
  `r2_6_titan_common_pool_l1_20261004/`, `r2_6_titan_common_pool_l1_repaired_arm1_20261004/`,
  `r2_6_titan_arm1_regen_20261004/regenerated_slide_summary.csv`,
  `r2_6_titan_direct1024_20261004/stage2/b100_strict/job_dir/40x_1024px_0px_overlap/`.
* Table S27 (S-T2) -- "### R2.6-T8" (l.~3046); `r2_6_titan_unannotated_query_sensitivity_20261005/`
  {summary,per_group}_all_bundles.csv.

No statistic is computed: values are copied, rounded for display and arranged. The only aggregations
are the row counts the R1.3-T8 spec itself defines for Table S20 (transitions, coverage losses,
reference exposure). Every spec-quoted number is compared with the source and the outcome recorded.
Read-only on all inputs; writes only $PROSPER_OUT/supp_tables_20261022/B_external_leakage_titan/.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path(os.environ.get("PROSPER_OUT") or REPO / "prosper_revision_output")
GROUP = "B_external_leakage_titan"
OUT = OUT_ROOT / "supp_tables_20261022" / GROUP
EDIT = REPO / "docs/manuscript_supplement_edits.md"
E20 = OUT_ROOT / "r1_3_external_synthesis_20260930"
FIG4 = OUT_ROOT / "figures_20261021/fig4_additional_collections/source_data.csv"
R4A = OUT_ROOT / "r4a_vii_whole_patient_20261015"
C1E_JSON = OUT_ROOT / "c1e_slide_unmasked"
C1E_NOTE = REPO / "docs/analysis/c1e_slide_level_masking_delta.md"
ADR9 = REPO / "docs/adr/0009-evaluation-protocol-freeze.md"
UNANN = OUT_ROOT / "r2_6_titan_unannotated_query_sensitivity_20261005"
MINUS = "\u2212"

INPUTS: dict[str, Path] = {}
CHECKS: list[dict] = []
TABLES: list[dict] = []


# ----------------------------------------------------------------------------- helpers
def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def use(p: Path) -> Path:
    assert p.exists(), f"missing input: {p}"
    INPUTS[str(p)] = p
    return p


def rd(p: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(use(p), **kw)


def check(table: str, what: str, ok: bool, expected="", observed="", hard=False) -> bool:
    CHECKS.append(dict(table=table, check=what, status="PASS" if ok else "MISMATCH",
                       expected=str(expected), observed=str(observed)))
    if hard:
        assert ok, f"{table}: {what}: expected {expected}, observed {observed}"
    return ok


def spec_num(s: str) -> float:
    return float(s.replace(MINUS, "-").replace("+", "").replace("**", "").strip())


def num_check(table: str, what: str, value: float, spec: str) -> bool:
    """Spec number `spec` (string as printed) must equal `value` at the spec's printed precision."""
    s = spec.replace(MINUS, "-").replace("+", "").strip()
    d = len(s.split(".")[1]) if "." in s else 0
    ok = value == value and abs(value - float(s)) <= 0.5 * 10 ** -d + 1e-9
    return check(table, what, ok, spec, f"{value:.6g}" if value == value else "NaN")


def fmt(x, d=3) -> str:
    if x is None or (isinstance(x, float) and x != x) or pd.isna(x):
        return "not estimable"
    return f"{float(x):.{d}f}"


def sfmt(x, d=4) -> str:
    if pd.isna(x):
        return "not estimable"
    s = f"{abs(float(x)):.{d}f}"
    return (MINUS if float(x) < 0 else "+") + s


def ci(lo, hi, d=3, signed=False) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return "not estimable"
    f = sfmt if signed else fmt
    return f"[{f(lo, d)}, {f(hi, d)}]"


def half_up(s: str, d: int) -> str:
    return str(Decimal(s).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


EDIT_LINES = EDIT.read_text(encoding="utf-8").splitlines()
use(EDIT)


def heading_line(prefix: str) -> int:
    hits = [i for i, l in enumerate(EDIT_LINES) if l.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def block(prefix: str) -> tuple[int, list[str]]:
    i = heading_line(prefix)
    j = i + 1
    while j < len(EDIT_LINES) and not re.match(r"^#{2,3} ", EDIT_LINES[j]):
        j += 1
    return i + 1, EDIT_LINES[i:j]


def take(lines: list[str], start_pat: str, stop=lambda l: l.strip() == "", include_start=True) -> list[str]:
    """Consecutive lines from the first line matching start_pat until stop(line) is true."""
    for i, l in enumerate(lines):
        if re.search(start_pat, l):
            out = [l] if include_start else []
            for m in lines[i + 1:]:
                if stop(m):
                    break
                out.append(m)
            return out
    raise AssertionError(f"pattern not found: {start_pat}")


def md_table(lines: list[str], header_pat: str) -> list[list[str]]:
    """Parse the Markdown table whose header row matches header_pat."""
    for i, l in enumerate(lines):
        if l.startswith("|") and re.search(header_pat, l):
            rows = []
            for m in lines[i + 2:]:
                if not m.startswith("|"):
                    break
                rows.append([c.strip() for c in m.strip().strip("|").split("|")])
            return rows
    raise AssertionError(f"table not found: {header_pat}")


CELL = re.compile(r"^\**\s*([+\u2212-]?\d+\.\d+)\s*\[([+\u2212-]?\d+\.\d+),\s*([+\u2212-]?\d+\.\d+)\]\**")


def write_table(tid: str, stem: str, heading: str, line: int, caption: list[tuple[str, str]], notes: list[str],
                sheets: list[tuple[str, pd.DataFrame, pd.DataFrame | None]]) -> None:
    """caption: (field label, verbatim [internal record] line); sheets: (name, XLSX frame, CSV frame or None = same)."""
    cap = [("Table ID (provisional)", tid), ("Edit-sheet block", heading),
           ("Edit-sheet line (at build time)", str(line)), ("", "")]
    cap += list(caption)
    cap += [("", "")] + [("assembly note (not caption text)", n) for n in notes]
    capdf = pd.DataFrame(cap, columns=["field", "text"])
    xlsx = OUT / f"{stem}.xlsx"
    files = [xlsx.name]
    rows = {}
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        capdf.to_excel(xw, sheet_name="caption", index=False)
        for name, disp, raw in sheets:
            assert len(name) <= 31, name
            disp.to_excel(xw, sheet_name=name, index=False)
            c = OUT / f"{stem}__{name}.csv"
            (raw if raw is not None else disp).to_csv(c, index=False)
            files.append(c.name)
            rows[name] = (len(disp), disp.shape[1], len(raw) if raw is not None else len(disp))
    TABLES.append(dict(tid=tid, stem=stem, heading=heading, line=line, files=files, rows=rows, notes=notes))


def V(label: str, lines: list[str]) -> list[tuple[str, str]]:
    return [(label, l) for l in lines]


# ----------------------------------------------------------------------------- S18-S21 (R1.3-T8)
COH = {"stv": "STV", "kidrare": "KidRare", "ccdi": "CCDI-MCI"}
COH_ORDER = {"stv": 0, "kidrare": 1, "ccdi": 2}
GROUPS = ["ACT/ACC", "DSRCT", "EWS", "LIVER", "MEL", "MPNST", "MRT", "NB", "NF", "NORMAL", "OS", "RB",
          "RMS", "SS", "WT"]
G_ORDER = {g: i for i, g in enumerate(GROUPS + ["micro", "macro"])}
HASH_COLS = ["population_order_sha256", "weights_sha256"]
OUTCOME = {"hit_miss": "hit@5 miss", "closest_mismatch": "closest-plurality mismatch",
           "generous_incorrect": "truth-aware generous-score failure"}


def flag(sp, dg) -> str:
    return "; ".join([t for t, v in (("sparse", sp), ("degenerate", dg)) if bool(v)])


def est_display(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "cohort": d.cohort.map(COH), "population": d.population, "configuration": d.configuration,
        "analysis_role": d.analysis_role, "group": d.tumor_group, "endpoint": d.metric,
        "n_slides": d.n_slides, "n_units": d.n_resampling_units, "unit_type": d.resampling_unit,
        "point": d.point.map(fmt), "95% CI": [ci(l, h) for l, h in zip(d.ci_lo, d.ci_hi)],
        "sparse/degenerate flag": [flag(a, b) for a, b in zip(d.sparse_units, d.degenerate_interval)],
    })
    if "n_returned" in d:
        out.insert(8, "n_returned", d.n_returned)
        out.insert(9, "n_confirmed_empty", d.n_confirmed_empty)
    if "comparison_id" in d:
        out["comparison_id"] = d.comparison_id
        out["alternative_configuration"] = d.alternative_configuration
        out["baseline_configuration"] = d.baseline_configuration
        out["baseline point"] = d.baseline_point.map(fmt)
        out["difference"] = d.difference.map(fmt)
        out["difference 95% CI"] = [ci(l, h) for l, h in zip(d.difference_ci_lo, d.difference_ci_hi)]
    raw = d.drop(columns=[c for c in HASH_COLS if c in d]).add_suffix("__raw")
    return pd.concat([out.reset_index(drop=True), raw.reset_index(drop=True)], axis=1)


def sort_est(d: pd.DataFrame) -> pd.DataFrame:
    return d.assign(_c=d.cohort.map(COH_ORDER), _g=d.tumor_group.map(G_ORDER)).sort_values(
        ["_c", "population", "analysis_role", "configuration", "metric", "_g"], kind="stable").drop(columns=["_c", "_g"])


def build_r13() -> None:
    T = "S18-S21"
    hl, blk = block("### R1.3-T8 ")
    heading = blk[0][4:]
    spec_tab = md_table(blk, r"New table / sheets")
    where = take(blk, r"^\*\*Where:\*\*")
    inv = take(blk, r"^The full per-query inventory")
    acc = take(blk, r"^\*\*Acceptance:\*\*")

    def spec_rows(prefix):
        return [("spec operation (verbatim table row)", " | ".join(r)) for r in spec_tab if r[0].startswith(prefix)]

    def capt(t):
        return V("caption (verbatim)", take(blk, rf"^> Table {t}\."))

    common = V("spec 'Where' (verbatim)", where) + V("spec (verbatim)", inv) + V("spec 'Acceptance' (verbatim)", acc)
    note_hash = ("Columns population_order_sha256 / weights_sha256 (hashes) are omitted from the public sheets per the "
                 "spec ('paths, hashes or raw case IDs belong in controlled provenance'); they remain in the bound source.")
    note_fmt = ("Display columns: points/CIs to three decimals, unavailable values 'not estimable' (spec 'Where'); every source "
                "column is kept at full precision with the suffix __raw.")

    b = rd(E20 / "broad_group_primary.csv")
    check(T, "broad_group_primary.csv rows = 744 (S19 absolute)", len(b) == 744, 744, len(b))
    # ---- S18 primary
    ids = {("stv", "main_696"), ("kidrare", "amended_1581"), ("ccdi", "ffpe_scope_clear")}
    in_ids = pd.Series([(c, p) in ids for c, p in zip(b.cohort, b.population)], index=b.index)
    s18 = b[(b.configuration == "whole_slide_ot") & in_ids]
    check("S18", "three main identities + whole_slide_ot = 74 long-format endpoint rows", len(s18) == 74, 74, len(s18))
    check("S18", "all S18 rows have analysis_role 'main'", (s18.analysis_role == "main").all(), "main",
          s18.analysis_role.unique().tolist())
    grp = s18[~s18.tumor_group.isin(["micro", "macro"])]
    check("S18", "observed group/endpoint rows = 62 (R1.3-T5)", len(grp) == 62, 62, len(grp))
    obs = [grp[grp.cohort == c].tumor_group.nunique() for c in ("stv", "kidrare", "ccdi")]
    check("S18", "observed groups STV/KidRare/CCDI = 15/4/12 (R1.3-T5)", obs == [15, 4, 12], "15/4/12", obs)
    check("S18", "CCDI NF absent (R1.3-T5)", not ((grp.cohort == "ccdi") & (grp.tumor_group == "NF")).any(), "absent", "")
    check("S18", "every S18 row has a point and a 95% interval", s18[["point", "ci_lo", "ci_hi"]].notna().all().all(),
          "no NaN", int(s18[["point", "ci_lo", "ci_hi"]].isna().sum().sum()))
    check("S18", "S18 comparison columns empty (absolute rows)", s18[["comparison_id", "difference"]].isna().all().all(), "NaN", "")
    s18 = sort_est(s18)
    d18 = est_display(s18.drop(columns=["comparison_id", "alternative_configuration", "baseline_configuration",
                                        "baseline_point", "difference", "difference_ci_lo", "difference_ci_hi"]))
    # Fig. 4 (2026-10-21 single grid) source data must be a subset of S18
    f4 = rd(FIG4)
    m = f4.merge(s18, on=["cohort", "population", "tumor_group", "metric"], suffixes=("_f", ""), how="left")
    ok = m.point.notna().all() and all((m[f"{c}_f"] - m[c]).abs().max() < 1e-12 for c in ("point", "ci_lo", "ci_hi")) \
        and (m.n_slides_f == m.n_slides).all() and (m.n_resampling_units_f == m.n_resampling_units).all()
    check("S18", "revised Fig. 4 source_data.csv (68 rows: 3 macro + 31 group x 2 metrics) equals S18 rows exactly",
          ok and len(f4) == 68, "68 rows, |diff| < 1e-12", f"{len(f4)} rows, matched {int(m.point.notna().sum())}")
    write_table("Table S18 (R1.3; provisional)", "Table_S18_primary_broad_group", heading, hl,
                capt("S18") + spec_rows("S18") + common,
                [note_fmt, note_hash,
                 "Scope implemented: original R1.3-T8 S18 spec (74 rows: STV main_696, KidRare amended_1581, CCDI-MCI "
                 "ffpe_scope_clear; whole_slide_ot; group + micro + macro). The 2026-10-21 Fig. 4 single-grid amendments "
                 "(edit sheet ~l.7858-7896) cite 'Source data in Supplementary Table S18' and do not change S18's scope; the "
                 "68-row figure source data were verified to be an exact subset of this sheet. The CCDI-MCI all-preparation "
                 "macro values quoted in the new Fig. 4 legend are in Table S19 'absolute' (preparation_sensitivity), not S18.",
                 "Rows ordered STV, KidRare, CCDI-MCI; endpoint; groups in the fixed R1.3-T5 order, then micro, macro. "
                 "Absent groups are not added as rows (the spec fixes 74 rows)."],
                [("primary", d18, None)])

    # ---- S19
    allprep = b[(b.population == "allprep_scope_clear") & (b.configuration == "whole_slide_ot") & (b.tumor_group == "macro")]
    h = allprep[allprep.metric == "hit_at_5"].iloc[0]
    p = allprep[allprep.metric == "precision_at_5"].iloc[0]
    num_check("S19", "Fig. 4 legend (10-21 15:45): CCDI all-prep macro hit@5 0.570", h.point, "0.570")
    num_check("S19", "Fig. 4 legend: CCDI all-prep macro hit@5 CI low 0.511", h.ci_lo, "0.511")
    num_check("S19", "Fig. 4 legend: CCDI all-prep macro hit@5 CI high 0.629", h.ci_hi, "0.629")
    num_check("S19", "Fig. 4 legend: CCDI all-prep macro precision@5 0.422", p.point, "0.422")
    d19a = est_display(sort_est(b))
    pr = rd(E20 / "paired_primary_differences.csv")
    check("S19", "paired_primary_differences.csv rows = 1128", len(pr) == 1128, 1128, len(pr))
    check("S19", "paired: 25 contrast identities (comparison_id)", pr.comparison_id.nunique() == 25, 25, pr.comparison_id.nunique())
    dd = (pr.point - pr.baseline_point - pr.difference).abs().max()
    check("S19", "paired: difference = alternative - baseline point (consistency only; CIs untouched)", dd < 1e-9, "<1e-9", f"{dd:.2e}")
    check("S19", "paired: every row has its own difference CI",
          pr[["difference_ci_lo", "difference_ci_hi"]].notna().all().all(), "no NaN",
          int(pr[["difference_ci_lo", "difference_ci_hi"]].isna().sum().sum()))
    d19p = est_display(sort_est(pr))
    cv = rd(E20 / "coverage_and_hit_yield.csv")
    check("S19", "coverage_and_hit_yield.csv rows = 650", len(cv) == 650, 650, len(cv))
    d19c = est_display(sort_est(cv))
    sec = rd(E20 / "expanded_secondary_retrieval_points.csv")
    check("S19", "expanded_secondary_retrieval_points.csv rows = 84", len(sec) == 84, 84, len(sec))
    check("S19", "secondary rows explicitly point-only", (sec.interval_status == "secondary_point_only_no_new_CI").all(),
          "secondary_point_only_no_new_CI", sec.interval_status.unique().tolist())
    d19s = pd.concat([pd.DataFrame({"cohort": sec.cohort.map(COH), "population": sec.population,
                                    "configuration": sec.configuration, "metric": sec.metric,
                                    "point (point only; no new CI)": sec.point.map(fmt), "n_queries": sec.n_queries,
                                    "retrieval_depth": sec.retrieval_depth, "estimand": sec.estimand,
                                    "interval_status": sec.interval_status}),
                      sec.add_suffix("__raw")], axis=1)
    write_table("Table S19 (R1.3; provisional)", "Table_S19_configuration_filtering_coverage", heading, hl,
                capt("S19") + spec_rows("S19") + common,
                [note_fmt, note_hash,
                 "Sheets: absolute = all 744 broad_group_primary rows (all roles); paired = all 1,128 rows, alternative point "
                 "('point'), baseline point, difference and the difference's own CI (CI endpoints never subtracted); coverage = "
                 "all 650 rows (macro coverage is class-weighted); secondary = all 84 point-only rows.",
                 "Rows ordered by cohort (STV, KidRare, CCDI-MCI), population, role, configuration, endpoint, group."],
                [("absolute", d19a, None), ("paired", d19p, None), ("coverage", d19c, None), ("secondary", d19s, None)])

    # ---- S20
    ls = rd(E20 / "all_returned_label_slots.csv")
    check("S20", "all_returned_label_slots.csv rows = 1580 (all kept)", len(ls) == 1580, 1580, len(ls))
    d20l = ls.assign(slot_fraction_fmt=ls.slot_fraction.map(fmt))
    fs = rd(E20 / "failure_conditioned_label_slots.csv")
    check("S20", "failure_conditioned_label_slots.csv rows = 4740 (all kept)", len(fs) == 4740, 4740, len(fs))
    d20f = fs.copy()
    d20f.insert(6, "outcome", fs.endpoint.map(OUTCOME))
    check("S20", "failure outcome label present on every row", d20f.outcome.notna().all(), "mapped", d20f.outcome.isna().sum())
    d20f["slot_fraction_fmt"] = fs.slot_fraction.map(fmt)
    fr = rd(E20 / "first_relevant_rank_distribution.csv")
    check("S20", "first_rank rows = 736 with four rank categories", len(fr) == 736 and fr.first_relevant_rank.nunique() == 4,
          "736 / 4", f"{len(fr)} / {fr.first_relevant_rank.nunique()}")
    k = ["cohort", "population", "configuration", "analysis_role", "tumor_group"]
    s = fr.groupby(k).agg(n=("n_queries", "sum"), d=("n_returned", "first"), u=("n_returned", "nunique"))
    check("S20", "first_rank: category counts sum to the returned-query denominator in every stratum",
          bool((s.n == s.d).all() and (s.u == 1).all()), "all strata", f"{int((s.n != s.d).sum())} strata differ")
    tr = rd(E20 / "matched_endpoint_transitions.csv")
    kt = ["cohort", "population", "configuration", "analysis_role", "baseline_configuration", "tumor_group", "endpoint"]
    cnt = tr.groupby(kt + ["transition"]).size().unstack("transition", fill_value=0)
    tcat = ["fixed", "broken", "unchanged_correct", "unchanged_incorrect"]
    check("S20", "transitions: four categories present in source", set(cnt.columns) == set(tcat), tcat, list(cnt.columns))
    cnt = cnt[tcat].stack().rename("n_query_rows").reset_index()
    check("S20", "transitions: counts sum to source rows (71,109)", int(cnt.n_query_rows.sum()) == len(tr) == 71109,
          71109, int(cnt.n_query_rows.sum()))
    cnt = cnt.assign(_c=cnt.cohort.map(COH_ORDER), _g=cnt.tumor_group.map(G_ORDER)).sort_values(
        ["_c", "population", "analysis_role", "configuration", "endpoint", "_g"], kind="stable").drop(columns=["_c", "_g"])
    cl = rd(E20 / "separate_coverage_losses.csv")
    check("S20", "separate_coverage_losses.csv: 4 rows, one query", len(cl) == 4 and cl["query"].nunique() == 1,
          "4 / 1", f"{len(cl)} / {cl['query'].nunique()}")
    kc = ["cohort", "population", "configuration", "analysis_role", "baseline_configuration", "tumor_group",
          "baseline_hit_at_5", "baseline_closest_correct", "baseline_generous_correct"]
    d20c = cl.groupby(kc, dropna=False).size().rename("n_records").reset_index()
    rr = rd(E20 / "reference_recurrence.csv")
    kr = ["cohort", "population", "configuration", "analysis_role", "endpoint", "neighbour_label",
          "all_query_exposure", "failure_exposure"]
    g = rr.groupby(kr, dropna=False)
    d20r = g.agg(n_references=("reference_id", "size"), n_all_returned_queries=("n_all_returned_queries", "first"),
                 n_failure_queries=("n_failure_queries", "first")).reset_index()
    u = g.agg(a=("n_all_returned_queries", "nunique"), f=("n_failure_queries", "nunique"))
    check("S20", "reference_exposure: denominators constant within each group (unchanged)",
          bool((u.a == 1).all() and (u.f == 1).all()), "constant", "")
    check("S20", "reference_exposure: references sum to source rows (44,730)", int(d20r.n_references.sum()) == len(rr),
          len(rr), int(d20r.n_references.sum()))
    d20r.insert(5, "outcome", d20r.endpoint.map(OUTCOME))
    ac = rd(E20 / "aggregate_context.csv")
    check("S20", "aggregate_context.csv: 66 rows, closest/generous outcomes only",
          len(ac) == 66 and set(ac.metric) == {"closest_correct", "generous_correct"}, 66, len(ac))
    d20o = ac.drop(columns=["weights_sha256"]).assign(
        conditional_point_fmt=ac.conditional_point.map(fmt), ci_fmt=[ci(l, h) for l, h in zip(ac.ci_lo, ac.ci_hi)],
        paired_baseline_point_fmt=ac.paired_baseline_point.map(fmt), paired_difference_fmt=ac.paired_difference.map(fmt),
        paired_difference_ci_fmt=[ci(l, h) for l, h in zip(ac.paired_difference_ci_lo, ac.paired_difference_ci_hi)])
    write_table("Table S20 (R1.3; provisional)", "Table_S20_failures_neighbours_transitions", heading, hl,
                capt("S20") + spec_rows("S20") + common,
                [note_fmt, note_hash,
                 "transitions / coverage_losses / reference_exposure are the row-count aggregations defined by the spec (group, "
                 "count); no statistic is estimated. transitions: 696 comparisons x 4 categories, absent categories filled with "
                 "0 only within existing comparisons. coverage_losses: the raw query identifier is dropped (one KidRare LIVER "
                 "query across four analyses). reference_exposure: individual reference identifiers dropped.",
                 "failure_slots and reference_exposure carry an 'outcome' label column (hit@5 miss / closest-plurality "
                 "mismatch / truth-aware generous-score failure) beside the source 'endpoint' key.",
                 "outcome_summary: absolute intervals absent in the source are left absent ('not estimable' in the "
                 "formatted column per the spec's global rule); they were not filled."],
                [("label_slots", d20l, None), ("failure_slots", d20f, None), ("first_rank", fr, None),
                 ("transitions", cnt, None), ("coverage_losses", d20c, None), ("reference_exposure", d20r, None),
                 ("outcome_summary", d20o, None)])

    # ---- S21
    st = rd(E20 / "descriptive_strata_points.csv")
    check("S21", "descriptive_strata_points.csv rows = 396", len(st) == 396, 396, len(st))
    check("S21", "strata are point-only (interval, test empty)", st[["interval", "test"]].isna().all().all(), "NaN", "")
    wt = rd(E20 / "wt_source_by_pattern_membership.csv")
    check("S21", "wt_source_by_pattern_membership.csv rows = 60", len(wt) == 60, 60, len(wt))
    cov = st[(st.endpoint == "coverage") & (st.configuration == "whole_slide_ot")]

    def nt(stype, stratum):
        r = cov[(cov.stratum_type == stype) & (cov.stratum == stratum)]
        return int(r.n_target.iloc[0]) if len(r) == 1 else None
    num_check("S21", "caption: 59 original WT-labelled KidRare queries", nt("source_group", "ORIGINAL_KEEP"), "59")
    num_check("S21", "caption: 354 added WT-labelled KidRare queries", nt("source_group", "ADDED_PATHPT"), "354")
    w = wt[(wt.population == "amended_1581") & (wt.configuration == "whole_slide_ot")]
    num_check("S21", "caption: 413 WT queries (source-by-pattern total)", w.n_queries.sum(), "413")
    num_check("S21", "caption: five source-labelled patterns", w.pathpt_pattern.nunique(), "5")
    prep = {s_: nt("preparation_status", s_) for s_ in ("VERIFIED_FFPE", "FROZEN_OCT", "MIXED", "NONE_UNKNOWN")}
    for s_, v in zip(prep, ("524", "201", "6", "4")):
        num_check("S21", f"caption: CCDI-MCI clear-scope {s_} queries", prep[s_], v)
    num_check("S21", "caption: 735-query CCDI-MCI clear-scope sensitivity", sum(prep.values()), "735")
    d21 = st.assign(cohort=st.cohort.map(COH), point_fmt=st.point.map(fmt))
    write_table("Table S21 (R1.3; provisional)", "Table_S21_descriptive_strata", heading, hl,
                capt("S21") + spec_rows("S21") + common,
                [note_fmt + " (S21: point only; no interval or test exists or is added.)",
                 "source_by_pattern counts are membership counts, not an interaction test; overlapping populations are not summed."],
                [("strata", d21, None), ("source_by_pattern", wt, None)])


# ----------------------------------------------------------------------------- S-R4.a-1 / S-R4.a-2
C1E_METRICS = [("f1_generous5", "Generous-majority F1 (submitted rule)"), ("f1_absolute5", "Strict-majority F1"),
               ("plurality5", "Plurality-vote recall (macro) / accuracy (micro); not F1"), ("hit1", "hit@1"),
               ("hit3", "hit@3"), ("hit5", "hit@5"), ("hit10", "hit@10"), ("precision5", "precision@5"),
               ("precision10", "precision@10")]
C1D_METRICS = [("top1_label_agree", "Top-1 label agreement"), ("top5_plurality_agree", "Top-5 plurality-label agreement"),
               ("jaccard_top5_ids", "Jaccard index of top-5 slide IDs"), ("jaccard_top5_labels", "Jaccard index of top-5 labels"),
               ("both_hit5", "Both slides hit@5"), ("split_hit5", "Split hit@5 (exactly one slide hits)"),
               ("jsd_posterior_N5", "Jensen-Shannon divergence of posteriors, N = 5 (nats)"),
               ("jsd_posterior_N18", "Jensen-Shannon divergence of posteriors, N = 18 (nats)")]


def one(d: pd.DataFrame, **kw) -> pd.Series:
    m = pd.Series(True, index=d.index)
    for k, v in kw.items():
        m &= d[k] == v
    r = d[m]
    assert len(r) == 1, (kw, len(r))
    return r.iloc[0]


def r4a_texts(prefix: str, title_pat: str) -> tuple[int, str, list[str]]:
    hl, blk = block(prefix)
    tab = take(blk, title_pat, stop=lambda l: l.startswith("**Sources:**"))
    src = take(blk, r"^\*\*Sources:\*\*")
    dna = take(blk, r"^\*\*Do not add:\*\*")
    return hl, blk[0][4:], tab, src, dna, blk


def build_r4a() -> None:
    T1, T2 = "S-R4.a-1", "S-R4.a-2"
    PROD, SC = R4A / "production", R4A / "production_c1d_same_class"
    expect = {PROD / "summary.csv": "8525fbea", PROD / "paired_differences.csv": "f9088a18",
              SC / "summary.csv": "1619d964", SC / "paired_differences.csv": "888537ac",
              SC / "per_pair_frame.csv.gz": "ebc4d7a7", C1E_NOTE: "c0fb9af2",
              C1E_JSON / "slide_ci_uni2_ot_cosine_top5_generous_majority_SAME_PATIENT_ALLOWED.json": "095430dd",
              C1E_JSON / "slide_ci_uni2_ot_cosine_top5_absolute_SAME_PATIENT_ALLOWED.json": "1d1bf972"}
    for p, pre in expect.items():
        check("S-R4.a", f"sha256 prefix of {p.relative_to(p.parents[1])} = {pre} (spec)", sha(use(p)).startswith(pre), pre,
              sha(p)[:8], hard=True)
    s, pdiff = rd(PROD / "summary.csv"), rd(PROD / "paired_differences.csv")

    # ---- S-R4.a-1 Block A
    hl, heading, tab, src, dna, blk = r4a_texts("### R4.a-T4 ", r"^\*\*New Supplementary Table S-R4\.a-1")
    ins = take(blk, r"^> All internal slide-search")
    e = s[(s.arm == "c1e") & (s.population == "c1e_all_tissue_2778")]
    ed = pdiff[(pdiff.arm == "c1e") & (pdiff.minuend == "unmasked") & (pdiff.subtrahend == "masked")]
    check(T1, "c1e: 36 config rows and 18 paired differences", len(e) == 36 and len(ed) == 18, "36/18", f"{len(e)}/{len(ed)}")
    check(T1, "c1e: n = 2,778 queries / 1,876 patients / 15 groups / 5,000 replicates on every row",
          bool((e.n_queries == 2778).all() and (e.n_patients == 1876).all() and (e.n_groups == 15).all()
               and (e.n_replicates == 5000).all() and (ed.n_queries == 2778).all() and (ed.n_patients == 1876).all()),
          "2778/1876/15/5000", "")
    check(T1, "c1e: no non-finite replicates", int(e.n_nonfinite.sum() + ed.n_nonfinite.sum()) == 0, 0,
          int(e.n_nonfinite.sum() + ed.n_nonfinite.sum()))
    rows, n_excl0 = [], 0
    for mk, ml in C1E_METRICS:
        for av in ("macro", "micro"):
            mas, unm, df = one(e, config="masked", metric=mk, average=av), one(e, config="unmasked", metric=mk, average=av), \
                one(ed, metric=mk, average=av)
            check(T1, f"{mk} {av}: difference point = left-in - excluded", abs(unm.point - mas.point - df.point) < 1e-9,
                  "<1e-9", f"{abs(unm.point - mas.point - df.point):.1e}")
            n_excl0 += int(df.ci_lo > 0 or df.ci_hi < 0)
            for lab, r in (("Excluded (published)", mas), ("Left in", unm), ("Difference (left in \u2212 excluded)", df)):
                rows.append({"metric": ml, "average": av, "row": lab, "point": fmt(r.point), "95% CI": ci(r.ci_lo, r.ci_hi),
                             "n_queries": int(r.n_queries), "n_patients": int(r.n_patients), "flag": r.flag if isinstance(r.flag, str) else "",
                             "metric_key__raw": mk, "source_config__raw": getattr(r, "config", "unmasked\u2212masked"),
                             "point__raw": r.point, "ci_lo__raw": r.ci_lo, "ci_hi__raw": r.ci_hi,
                             "n_replicates__raw": int(r.n_replicates), "role__raw": r.role})
    A = pd.DataFrame(rows)
    check(T1, "Block A: all 18 paired-difference intervals exclude zero (observation; Insert says 'same direction')",
          n_excl0 == 18, 18, n_excl0)

    def g(cfg, mk, av="macro", what="point"):
        if cfg == "diff":
            return one(ed, metric=mk, average=av)[what]
        return one(e, config=cfg, metric=mk, average=av)[what]
    for cfg, lab, vals in (("unmasked", "left in", ("0.802", "0.779", "0.820")), ("masked", "excluded", ("0.791", "0.768", "0.810")),
                           ("diff", "difference", ("0.011", "0.004", "0.018"))):
        for w, v in zip(("point", "ci_lo", "ci_hi"), vals):
            num_check(T1, f"Insert: generous macro F1 k=5 {lab} {w} {v}", g(cfg, "f1_generous5", what=w), v)
    for mk, lab, vals in (("plurality5", "plurality macro recall", ("0.008", "0.002", "0.013")),
                          ("f1_absolute5", "strict macro F1", ("0.018", "0.008", "0.029")),
                          ("hit1", "hit@1 macro", ("0.014", "0.007", "0.022")),
                          ("hit5", "hit@5 macro", ("0.017", "0.008", "0.026"))):
        for w, v in zip(("point", "ci_lo", "ci_hi"), vals):
            num_check(T1, f"Insert: difference {lab} {w} {v}", g("diff", mk, what=w), v)

    # ---- Block B (points only)
    adr = use(ADR9).read_text(encoding="utf-8")
    mm = re.search(r"\((0\.\d+) / (0\.\d+) / (0\.\d+) / (0\.\d+)\)", adr)
    adr_v = dict(zip((1, 3, 5, 10), mm.groups()))
    check(T1, "ADR 0009 frozen masked generous values found (0.769515 / 0.821829 / 0.765022 per spec)",
          [round(float(adr_v[k]), 6) for k in (1, 3, 10)] == [0.769515, 0.821829, 0.765022], "0.769515/0.821829/0.765022",
          "/".join(adr_v[k] for k in (1, 3, 10)))
    num_check(T1, "ADR 0009 k=5 masked generous = production masked f1_generous5 macro", g("masked", "f1_generous5"), f"{float(adr_v[5]):.6f}")
    note = use(C1E_NOTE).read_text(encoding="utf-8").splitlines()
    strict_masked, rule = {}, None
    for l in note:
        mt = re.match(r"^\|\s*(generous plurality|strict majority)?\s*\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*(0\.\d{4})\s*\|", l)
        if mt:
            rule = mt.group(1) or rule
            if rule == "strict majority":
                strict_masked[int(mt.group(2))] = mt.group(4)
    check(T1, "C1e note strict-majority masked rows parsed (k=1,3,5,10)", sorted(strict_masked) == [1, 3, 5, 10], "[1,3,5,10]",
          sorted(strict_masked))
    num_check(T1, "C1e note strict masked k=5 (4 dp) = production masked f1_absolute5 macro", g("masked", "f1_absolute5"),
              strict_masked[5])

    def jget(k, r):
        p = use(C1E_JSON / f"slide_ci_uni2_ot_cosine_top{k}_{r}_SAME_PATIENT_ALLOWED.json")
        return json.loads(p.read_text())["point_estimates"]["macro_f1"]
    num_check(T1, "unmasked JSON top5 generous macro F1 = production unmasked f1_generous5 macro (6 dp)",
              jget(5, "generous_majority"), f"{g('unmasked', 'f1_generous5'):.6f}")
    num_check(T1, "unmasked JSON top5 absolute macro F1 = production unmasked f1_absolute5 macro (6 dp)",
              jget(5, "absolute"), f"{g('unmasked', 'f1_absolute5'):.6f}")
    specB = {("generous", 1): ("0.770", "0.787"), ("generous", 3): ("0.822", "0.842"), ("generous", 10): ("0.765", "0.776"),
             ("strict", 1): ("0.770", "0.787"), ("strict", 3): ("0.761", "0.782"), ("strict", 10): ("0.691", "0.703")}
    rowsB, mismB = [], []
    for rl, jr in (("generous", "generous_majority"), ("strict", "absolute")):
        for k in (1, 3, 10):
            if rl == "generous":
                ex_raw, ex_src = adr_v[k], "ADR 0009 frozen masked value (full precision)"
                ex_disp = fmt(float(ex_raw))
            else:
                ex_raw, ex_src = strict_masked[k], "C1e note table (4 dp only)"
                ex_disp = half_up(ex_raw, 3)
            li = jget(k, jr)
            li_disp = fmt(li)
            sx, sl = specB[(rl, k)]
            okx = check(T1, f"Block B {rl} k={k} excluded display {ex_disp} = spec {sx}", ex_disp == sx, sx, ex_disp)
            okl = check(T1, f"Block B {rl} k={k} left-in display {li_disp} = spec {sl}", li_disp == sl, sl, f"{li_disp} ({li:.6f})")
            if not okl:
                mismB.append(f"{rl} k={k} left in: spec {sl}, source {li:.7f} -> {li_disp}")
            if not okx:
                mismB.append(f"{rl} k={k} excluded: spec {sx}, source {ex_raw} -> {ex_disp}")
            rowsB.append({"rule": "Generous-majority macro F1" if rl == "generous" else "Strict-majority macro F1", "k": k,
                          "Excluded (published)": ex_disp,
                          "Left in": li_disp if okl else f"{li_disp} [SPEC/SOURCE MISMATCH: spec {sl}]",
                          "interval": "whole-patient intervals not computed",
                          "excluded__raw": ex_raw, "excluded_source": ex_src,
                          "left_in__raw": li, "left_in_source": f"c1e_slide_unmasked top{k} {jr} point_estimates.macro_f1"})
    B = pd.DataFrame(rowsB)
    capA = [("table title + specification (verbatim)", l) for l in tab] + V("Insert (verbatim; text numbers checked)", ins) \
        + V("spec Sources (verbatim)", src) + V("spec Do not add (verbatim)", dna)
    write_table("Table S-R4.a-1 (provisional; not yet numbered)", "Table_S-R4a-1_masked_vs_unmasked", heading, hl, capA,
                ["Block A: arm c1e, population c1e_all_tissue_2778 (2,778 queries, 1,876 patients); excluded = config 'masked', "
                 "left in = 'unmasked', difference = paired_differences 'unmasked - masked' with its own whole-patient interval. "
                 "Points/CIs displayed to 3 dp; full precision in __raw columns.",
                 "Block B: point estimates only. Generous excluded = ADR 0009 frozen values; strict excluded = C1e note table, "
                 "which records only 4 dp (displayed 3 dp, ROUND_HALF_UP on the recorded string: 0.7695 -> 0.770); left in = "
                 "unmasked C1e JSONs (point_estimates.macro_f1). At k = 1 the strict and generous rules coincide.",
                 *(["Block B [SPEC/SOURCE MISMATCH]: " + m_ + " (spec value appears to be a double rounding of the note's 4-dp "
                    "0.7815; the cell shows the source value, flagged, not resolved)" for m_ in mismB])],
                [("block_A_k5", A, None), ("block_B_k1_3_10_points", B, None)])

    # ---- S-R4.a-2
    hl2, heading2, tab2, src2, dna2, blk2 = r4a_texts("### R4.a-T5 ", r"^\*\*New Supplementary Table S-R4\.a-2")
    ins2 = take(blk2, r"^> Different slides from the same patient")
    s2, pd2 = rd(SC / "summary.csv"), rd(SC / "paired_differences.csv")
    keys = ["config", "metric", "average", "point", "ci_lo", "ci_hi", "n_queries", "n_patients"]
    a = s[s.arm == "c1d"].sort_values(["config", "metric"])[keys].reset_index(drop=True)
    b_ = s2[s2.arm == "c1d"].sort_values(["config", "metric"])[keys].reset_index(drop=True)
    check(T2, "c1d rows in production_c1d_same_class/summary.csv identical to production/summary.csv", a.equals(b_), "identical",
          "identical" if a.equals(b_) else "differ")
    arms = {"(i) all within-patient pairs": (s, pdiff, "c1d", "c1d_pairs_463_patients", 1898, 463),
            "(ii) same-label within-patient pairs": (s2, pd2, "c1d_same_class", "c1d_same_class_pairs_446_patients", 1768, 446)}
    wide, longrows = [], []
    for mk, ml in C1D_METRICS:
        row = {"metric": ml}
        for an, (S_, P_, arm, pop, npairs, npat) in arms.items():
            sp = one(S_, arm=arm, population=pop, config="same_patient", metric=mk)
            cm = one(S_, arm=arm, population=pop, config="class_matched_null", metric=mk)
            df = one(P_, arm=arm, population=pop, minuend="same_patient", subtrahend="class_matched_null", metric=mk)
            check(T2, f"{arm} {mk}: n pairs/patients = {npairs}/{npat}; difference n_queries = {2 * npairs}",
                  sp.n_queries == npairs and cm.n_queries == npairs and sp.n_patients == npat and df.n_patients == npat
                  and df.n_queries == 2 * npairs, f"{npairs}/{npat}/{2 * npairs}", f"{sp.n_queries}/{sp.n_patients}/{df.n_queries}")
            check(T2, f"{arm} {mk}: difference point = within - comparison", abs(sp.point - cm.point - df.point) < 1e-9, "<1e-9", "")
            tag = an.split(" ")[0]
            for lab, r in (("within-patient", sp), ("comparison", cm), ("difference", df)):
                row[f"{tag} {lab}"] = fmt(r.point)
                row[f"{tag} {lab} 95% CI"] = ci(r.ci_lo, r.ci_hi)
                longrows.append({"column_group": an, "arm": arm, "population": pop, "metric": mk, "metric_label": ml,
                                 "row": lab, "point": r.point, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                                 "point_fmt": fmt(r.point), "ci_fmt": ci(r.ci_lo, r.ci_hi), "n_pairs_or_queries": int(r.n_queries),
                                 "n_patients": int(r.n_patients), "n_replicates": int(r.n_replicates),
                                 "flag": r.flag if isinstance(r.flag, str) else ""})
        wide.append(row)
    W, L = pd.DataFrame(wide), pd.DataFrame(longrows)

    def v(arm, mk, row, what="point"):
        return L[(L.arm == arm) & (L.metric == mk) & (L.row == row)][what].iloc[0]
    pc = lambda x: 100 * x
    for arm, mk, row, w, sv in [
        ("c1d", "jaccard_top5_ids", "within-patient", "point", "0.077"), ("c1d", "jaccard_top5_ids", "comparison", "point", "0.012"),
        ("c1d", "jaccard_top5_ids", "difference", "point", "0.065"), ("c1d", "jaccard_top5_ids", "difference", "ci_lo", "0.051"),
        ("c1d", "jaccard_top5_ids", "difference", "ci_hi", "0.079"),
        ("c1d_same_class", "jaccard_top5_ids", "difference", "point", "0.069"),
        ("c1d_same_class", "jaccard_top5_ids", "difference", "ci_lo", "0.055"),
        ("c1d_same_class", "jaccard_top5_ids", "difference", "ci_hi", "0.084"),
        ("c1d", "jaccard_top5_labels", "difference", "point", "0.022"), ("c1d", "jaccard_top5_labels", "difference", "ci_lo", "-0.001"),
        ("c1d", "jaccard_top5_labels", "difference", "ci_hi", "0.045"),
        ("c1d_same_class", "jaccard_top5_labels", "difference", "point", "0.039"),
        ("c1d_same_class", "jaccard_top5_labels", "difference", "ci_lo", "0.017"),
        ("c1d_same_class", "jaccard_top5_labels", "difference", "ci_hi", "0.061"),
        ("c1d", "jsd_posterior_N5", "within-patient", "point", "0.161"), ("c1d", "jsd_posterior_N5", "comparison", "point", "0.160"),
        ("c1d", "jsd_posterior_N5", "difference", "point", "0.001"), ("c1d", "jsd_posterior_N5", "difference", "ci_lo", "-0.016"),
        ("c1d", "jsd_posterior_N5", "difference", "ci_hi", "0.019"),
        ("c1d_same_class", "jsd_posterior_N5", "within-patient", "point", "0.139"),
        ("c1d_same_class", "jsd_posterior_N5", "comparison", "point", "0.159"),
        ("c1d_same_class", "jsd_posterior_N5", "difference", "point", "-0.020"),
        ("c1d_same_class", "jsd_posterior_N5", "difference", "ci_lo", "-0.034"),
        ("c1d_same_class", "jsd_posterior_N5", "difference", "ci_hi", "-0.005")]:
        num_check(T2, f"Insert: {arm} {mk} {row} {w} {sv}", v(arm, mk, row, w), sv)
    for arm, mk, row, w, sv in [
        ("c1d", "split_hit5", "within-patient", "point", "7.6"), ("c1d", "split_hit5", "comparison", "point", "10.2"),
        ("c1d", "split_hit5", "difference", "point", "-2.6"), ("c1d", "split_hit5", "difference", "ci_lo", "-4.3"),
        ("c1d", "split_hit5", "difference", "ci_hi", "-0.9"),
        ("c1d_same_class", "split_hit5", "difference", "point", "-2.8"), ("c1d_same_class", "split_hit5", "difference", "ci_lo", "-4.6"),
        ("c1d_same_class", "split_hit5", "difference", "ci_hi", "-1.0"),
        ("c1d", "top1_label_agree", "within-patient", "point", "73.7"), ("c1d", "top1_label_agree", "comparison", "point", "73.8"),
        ("c1d", "top1_label_agree", "difference", "point", "-0.1"), ("c1d", "top1_label_agree", "difference", "ci_lo", "-3.6"),
        ("c1d", "top1_label_agree", "difference", "ci_hi", "3.4"),
        ("c1d_same_class", "top1_label_agree", "within-patient", "point", "77.1"),
        ("c1d_same_class", "top1_label_agree", "comparison", "point", "74.4"),
        ("c1d_same_class", "top1_label_agree", "difference", "point", "2.7"),
        ("c1d_same_class", "top1_label_agree", "difference", "ci_lo", "-0.4"),
        ("c1d_same_class", "top1_label_agree", "difference", "ci_hi", "5.8"),
        ("c1d", "top5_plurality_agree", "difference", "point", "-3.4"), ("c1d", "top5_plurality_agree", "difference", "ci_lo", "-7.4"),
        ("c1d", "top5_plurality_agree", "difference", "ci_hi", "0.4"),
        ("c1d_same_class", "top5_plurality_agree", "difference", "point", "1.1"),
        ("c1d_same_class", "top5_plurality_agree", "difference", "ci_lo", "-2.2"),
        ("c1d_same_class", "top5_plurality_agree", "difference", "ci_hi", "4.2")]:
        num_check(T2, f"Insert (percent / percentage points): {arm} {mk} {row} {w} {sv}", pc(v(arm, mk, row, w)), sv)
    dints = L[L.row == "difference"]
    check(T2, "footnote: 16 differences (8 rows x 2 column groups)", len(dints) == 16, 16, len(dints))
    # footnote counts from the per-pair frame
    fr = pd.read_csv(use(SC / "per_pair_frame.csv.gz"))
    wp = fr[fr.kind == "same_patient"]
    diff = wp[wp.same_class == 0]
    num_check(T2, "footnote/Insert: 130 within-patient pairs with different labels", len(diff), "130")
    num_check(T2, "Insert: 130 pairs = 6.8% of 1,898", 100 * len(diff) / len(wp), "6.8")
    check(T2, "every comparison pair shares one label", bool((fr[fr.kind == "class_matched_null"].same_class == 1).all()), "all", "")
    num_check(T2, "footnote: 95 of the 130 involve normal tissue",
              int(((diff.class_a == "NORMAL") | (diff.class_b == "NORMAL")).sum()), "95")
    per = wp.groupby("patient").agg(n=("same_class", "size"), nd=("same_class", lambda x: int((x == 0).sum())))
    lost_all, lost_some = per[(per.nd > 0) & (per.nd == per.n)], per[(per.nd > 0) & (per.nd < per.n)]
    num_check(T2, "footnote: 17 patients have no remaining pair", len(lost_all), "17")
    num_check(T2, "footnote: 28 further patients lose some pairs", len(lost_some), "28")
    num_check(T2, "footnote: those 28 patients lose 113 pairs", int(lost_some.nd.sum()), "113")
    num_check(T2, "463 - 17 = 446 patients; 1,898 - 130 = 1,768 pairs",
              (len(per) - len(lost_all)) * 10000 + len(wp) - len(diff), str(446 * 10000 + 1768))
    capB = [("table title + specification (verbatim)", l) for l in tab2] + V("Insert (verbatim; text numbers checked)", ins2) \
        + V("spec Sources (verbatim)", src2) + V("spec Do not add (verbatim)", dna2)
    write_table("Table S-R4.a-2 (provisional; not yet numbered)", "Table_S-R4a-2_same_patient_consistency", heading2, hl2, capB,
                ["Column group (i) = arm c1d, population c1d_pairs_463_patients (production/); (ii) = arm c1d_same_class, "
                 "population c1d_same_class_pairs_446_patients (production_c1d_same_class/). within-patient = config "
                 "same_patient, comparison = class_matched_null, difference = paired_differences same_patient - "
                 "class_matched_null with its own whole-patient interval. average = pair_mean.",
                 "Values are proportions / Jaccard indices / divergences in nats to 3 dp (the Insert quotes agreement and split "
                 "hit@5 as percentages / percentage points = x100). both_generous5 and same_class rows excluded per 'Do not add'.",
                 "Footnote counts (130 / 95 / 17 / 28 / 113) were recounted from production_c1d_same_class/per_pair_frame.csv.gz."],
                [("consistency", W, L)])


# ----------------------------------------------------------------------------- S22 (R2.6-T7)
TITAN_METRICS = ["micro_hit@1", "micro_hit@3", "micro_hit@5", "micro_hit@10", "micro_plurality@5",
                 "macro_hit@1", "macro_hit@3", "macro_hit@5", "macro_hit@10"]
TITAN_COLS = [m.replace("_", " ") for m in TITAN_METRICS]
# bundle -> (arm1 key, arm2 key, contrast label); delta = arm2 - arm1
BUNDLES = {"repaired_vs_arm2l1": ("nf", "S2", "S2 \u2212 no filter"),
           "b100_argmax_vs_repaired": ("nf", "b100a", "b100a \u2212 no filter"),
           "b5_argmax_vs_repaired": ("nf", "b5a", "b5a \u2212 no filter"),
           "b100_strict_vs_repaired": ("nf", "strict", "strict \u2212 no filter"),
           "b100_argmax_vs_arm2l1": ("S2", "b100a", "b100a \u2212 S2"),
           "b5_argmax_vs_arm2l1": ("S2", "b5a", "b5a \u2212 S2"),
           "b100_strict_vs_arm2l1": ("S2", "strict", "strict \u2212 S2")}
ARM_LABEL = {"nf": "No filter (repaired all-tissue)", "S2": "S2 (mapped 512\u00d7512 predictions)",
             "b100a": "b100a (CONCH-native, argmax, 100 % labels)", "b5a": "b5a (CONCH-native, argmax, 5 % labels)",
             "strict": "strict (CONCH-native, p \u2265 0.9, 100 % labels)"}
POP22 = "common_pool_2562_intersection"


def sfmt0(x, d) -> str:
    """Signed display with Unicode minus; sign follows the stored value (exact zero shown '+0.000...')."""
    if pd.isna(x):
        return "not estimable"
    s = f"{abs(float(x)):.{d}f}"
    return (MINUS if float(x) < 0 else "+") + s


def sci(e, lo, hi, d) -> str:
    return f"{sfmt0(e, d)} [{sfmt0(lo, d)}, {sfmt0(hi, d)}]"


def bold_cells(xlsx: Path, sheet: str, cells: list[tuple[int, int]]) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Font
    wb = load_workbook(xlsx)
    ws = wb[sheet]
    for r, c in cells:
        ws.cell(row=r + 2, column=c + 1).font = Font(bold=True)
    wb.save(xlsx)


def build_s22() -> None:
    T = "S22"
    hl, blk = block("### R2.6-T7 ")
    heading = blk[0][4:]
    legend = take(blk, r"^> Supplementary Table S22\.")
    spec_A = md_table(blk, r"^\| Arm \| micro hit@1")
    spec_B = md_table(blk, r"^\| Difference \| micro hit@1")
    spec_C = md_table(blk, r"^\| Difference \| Group")
    spec_D_hdr = take(blk, r"^\| Arm \| Encoded")
    spec_D = md_table(blk, r"^\| Arm \| Encoded")
    notes_txt = take(blk, r"^\*\*Table notes", stop=lambda l: l.startswith("**Numbers source"))
    numsrc = take(blk, r"^\*\*Numbers source:\*\*")
    pa_intro = take(blk, r"^\*\*Panel A")
    pb_intro = take(blk, r"^\*\*Panel B")
    pc_intro = take(blk, r"^\*\*Panel C")
    pd_intro = take(blk, r"^\*\*Panel D")

    S, P = {}, {}
    for bn in BUNDLES:
        d = OUT_ROOT / f"r2_6_titan_intersection_{bn}_20261004"
        s = rd(d / "summary_common_pool.csv")
        S[bn] = s[s.population == POP22]
        p = rd(d / "per_class_common_pool.csv")
        P[bn] = p[p.population == POP22]
        check(T, f"{bn}: population {POP22} = 2,562 queries / 1,772 patients, L1 (cityblock) both arms",
              bool((S[bn].n_queries == 2562).all() and (S[bn].n_patients == 1772).all()
                   and (S[bn].arm1_metric == "cityblock").all() and (S[bn].arm2_metric == "cityblock").all()),
              "2562/1772/cityblock", f"{S[bn].n_queries.unique()}/{S[bn].n_patients.unique()}")
        check(T, f"{bn}: 9 metrics x 3 arms, per-class 14 groups x 2 metrics x 3 arms",
              len(S[bn]) == 27 and set(S[bn].metric) == set(TITAN_METRICS) and len(P[bn]) == 84, "27/84",
              f"{len(S[bn])}/{len(P[bn])}")
    # ---- Panel A
    vals: dict[str, dict[str, list[float]]] = {}
    for bn, (a1, a2, _) in BUNDLES.items():
        for key, arm in ((a1, "arm1"), (a2, "arm2")):
            for m in TITAN_METRICS:
                vals.setdefault(key, {}).setdefault(m, []).append(one(S[bn], metric=m, arm=arm).estimate)
    rowsA, rawA = [], []
    for key in ("nf", "S2", "b100a", "b5a", "strict"):
        same = all(max(v) - min(v) == 0 for v in vals[key].values())
        nb = len(next(iter(vals[key].values())))
        check(T, f"Panel A: {key} estimates identical in all {nb} bundles containing it", same, "identical", "identical" if same else "differ")
        rowsA.append({"Arm": ARM_LABEL[key], **{c: fmt(vals[key][m][0], 4) for c, m in zip(TITAN_COLS, TITAN_METRICS)}})
        rawA += [{"arm": key, "arm_label": ARM_LABEL[key], "metric": m, "estimate": vals[key][m][0],
                  "estimate_fmt": fmt(vals[key][m][0], 4), "n_bundles": nb} for m in TITAN_METRICS]
    A = pd.DataFrame(rowsA)
    lab2key = {v.split(" ")[0]: k for k, v in ARM_LABEL.items()}
    lab2key["No"] = "nf"
    for r in spec_A:
        key = lab2key[r[0].split(" ")[0]]
        for c, m, sv in zip(TITAN_COLS, TITAN_METRICS, r[1:]):
            num_check(T, f"Panel A {key} {c} = {sv}", vals[key][m][0], sv)
            check(T, f"Panel A {key} {c}: display identical to spec", fmt(vals[key][m][0], 4) == sv, sv, fmt(vals[key][m][0], 4))
    # ---- Panel B
    rowsB, rawB, boldB = [], [], []
    for i, (bn, (a1, a2, lab)) in enumerate(BUNDLES.items()):
        row = {"Difference": lab}
        for j, m in enumerate(TITAN_METRICS):
            r = one(S[bn], metric=m, arm="delta")
            e1, e2 = one(S[bn], metric=m, arm="arm1").estimate, one(S[bn], metric=m, arm="arm2").estimate
            assert abs(r.estimate - (e2 - e1)) < 1e-9, (bn, m)
            ex = bool(r.lo > 0 or r.hi < 0)
            row[TITAN_COLS[j]] = sci(r.estimate, r.lo, r.hi, 4)
            if ex:
                boldB.append((i, j + 1))
            rawB.append({"bundle": bn, "contrast": lab, "metric": m, "estimate": r.estimate, "lo": r.lo, "hi": r.hi,
                         "B_effective": r.B_effective, "n_zero_support_draws": r.n_zero_support_draws,
                         "ci_available": r.ci_available, "ci_degenerate": r.ci_degenerate,
                         "interval_excludes_zero_bold": ex, "display": row[TITAN_COLS[j]]})
        rowsB.append(row)
    B, RB = pd.DataFrame(rowsB), pd.DataFrame(rawB)
    check(T, "Panel B: 63 intervals, all 5,000 effective replicates", len(RB) == 63 and (RB.B_effective == 5000).all(), 63, len(RB))
    num_check(T, "Panel B: bold (interval excludes zero) = 2 of 63", len(boldB), "2")
    for r in spec_B:
        lab = r[0]
        for c, m, cell in zip(TITAN_COLS, TITAN_METRICS, r[1:]):
            mt = CELL.match(cell)
            src = RB[(RB.contrast == lab) & (RB.metric == m)].iloc[0]
            for w, sv in zip(("estimate", "lo", "hi"), mt.groups()):
                num_check(T, f"Panel B {lab} {c} {w} = {sv}", src[w], sv.replace(MINUS, "-"))
            check(T, f"Panel B {lab} {c}: display identical to spec cell", src.display == cell.strip("*"), cell.strip("*"),
                  src.display)
            check(T, f"Panel B {lab} {c}: bold in spec <=> interval excludes zero", cell.startswith("**") == bool(
                src.interval_excludes_zero_bold), cell.startswith("**"), bool(src.interval_excludes_zero_bold))
    # ---- Panel C
    pcs = []
    for bn, (a1, a2, lab) in BUNDLES.items():
        p = P[bn]
        dl = p[p.arm == "delta"]
        for _, r in dl.iterrows():
            pcs.append({"bundle": bn, "contrast": lab, "group": r["class"], "metric": r.metric,
                        "n_class_queries": r.n_class_queries, "n_class_patients": r.n_class_patients,
                        "estimate": r.estimate, "lo": r.lo, "hi": r.hi, "B_effective": r.B_effective,
                        "ci_available": r.ci_available, "ci_degenerate": r.ci_degenerate,
                        "excludes_zero": bool(r.lo > 0 or r.hi < 0), "bound_exactly_zero": bool(r.lo == 0 or r.hi == 0)})
    PC = pd.DataFrame(pcs)
    num_check(T, "Panel C: per-group delta intervals = 196", len(PC), "196")
    ex = PC[PC.excludes_zero].copy()
    num_check(T, "Panel C: per-group intervals excluding zero = 9", len(ex), "9")
    num_check(T, "Panel C: further intervals with a bound exactly at 0 = 72", int((PC.bound_exactly_zero & ~PC.excludes_zero).sum()), "72")
    mp = PC[(PC.contrast == "strict \u2212 S2") & (PC.group == "MPNST") & (PC.metric == "hit@5")].iloc[0]
    for w, sv in zip(("estimate", "lo", "hi"), ("-0.146", "-0.300", "0.000")):
        num_check(T, f"Panel C intro: MPNST strict - S2 hit@5 {w} = {sv}", mp[w], sv)
    num_check(T, "Panel C intro: MPNST 48 queries", mp.n_class_queries, "48")
    liv = PC[(PC.group == "LIVER") & (PC.metric == "plurality@5_recall") & PC.contrast.str.endswith("no filter")]
    check(T, "Legend: LIVER plurality@5 all four filters minus no filter in [-0.057, -0.050] (140 queries)",
          len(liv) == 4 and liv.estimate.round(3).between(-0.057, -0.050).all() and (liv.n_class_queries == 140).all(),
          "-0.050..-0.057, n=140", f"{liv.estimate.round(4).tolist()}")
    order = {lab: i for i, (_, _, lab) in enumerate(BUNDLES.values())}
    ex = ex.assign(_o=ex.contrast.map(order)).sort_values(["_o", "group"], kind="stable").drop(columns="_o")
    mlab = {"hit@5": "hit@5", "plurality@5_recall": "plurality@5 recall"}
    ex["Group (queries/patients)"] = [f"{g} ({q}/{n})" for g, q, n in zip(ex.group, ex.n_class_queries, ex.n_class_patients)]
    ex["Metric"] = ex.metric.map(mlab)
    ex["\u0394 [95 % CI]"] = [sci(e, l, h, 3) for e, l, h in zip(ex.estimate, ex.lo, ex.hi)]
    C = ex[["contrast", "Group (queries/patients)", "Metric", "\u0394 [95 % CI]"]].rename(columns={"contrast": "Difference"})
    spec_set = {(r[0], r[1], r[2]) for r in spec_C}
    src_set = set(zip(C.Difference, C["Group (queries/patients)"], C.Metric))
    check(T, "Panel C: spec rows = source excluding-zero rows (contrast, group (n/n), metric)", spec_set == src_set,
          sorted(spec_set - src_set), sorted(src_set - spec_set))
    for r in spec_C:
        sr = ex[(C.Difference == r[0]) & (C["Group (queries/patients)"] == r[1]) & (C.Metric == r[2])]
        if len(sr) == 1:
            mt = CELL.match(r[3])
            for w, sv in zip(("estimate", "lo", "hi"), mt.groups()):
                num_check(T, f"Panel C {r[0]} {r[1]} {r[2]} {w} = {sv}", sr.iloc[0][w], sv.replace(MINUS, "-"))
            disp = C.loc[sr.index[0], "\u0394 [95 % CI]"]
            check(T, f"Panel C {r[0]} {r[1]} {r[2]}: display identical to spec", disp == r[3], r[3], disp)
    # ---- Panel D (descriptive; static spec table verified against sources)
    con = rd(OUT_ROOT / "r2_6_titan_common_pool_20261003/connectivity_per_group.csv")
    kept, exc = con[con.section == "kept"], con[con.section == "excluded"]
    allk = kept[kept.label == "ALL"].iloc[0]
    num_check(T, "Panel D: S2 encoded 2,804", allk.n_slides, "2804")
    num_check(T, "Panel D: S2 not encoded 117", exc.n_slides.sum(), "117")
    num_check(T, "Panel D: 2,921 slides in total", allk.n_slides + exc.n_slides.sum(), "2921")
    f8 = exc[exc.reason_category == "fewer_than_8_tumour_patches"]
    num_check(T, "Panel D: fewer than 8 retained patches 83", f8.n_slides.sum(), "83")
    num_check(T, "Panel D: of which NORMAL 67", f8[f8.label == "NORMAL"].n_slides.sum(), "67")
    num_check(T, "Panel D: no tile scores 34", exc[exc.reason_category == "no_tile_scores"].n_slides.sum(), "34")
    num_check(T, "Panel D intro: S2 did not encode 40 of the 14-tumour-group slides",
              exc[~exc.label.isin(["NORMAL", "MISC"])].n_slides.sum(), "40")
    s2_tum = allk.n_slides - kept[kept.label.isin(["NORMAL", "MISC"])].n_slides.sum()
    num_check(T, "Legend: 2,564 encoded by S2 within the 14 tumour groups", s2_tum, "2564")
    num_check(T, "Legend: 2,604 slides in the 14 tumour groups", s2_tum + exc[~exc.label.isin(["NORMAL", "MISC"])].n_slides.sum(), "2604")
    jd = OUT_ROOT / "r2_6_titan_direct1024_20261004/stage2/b100_strict/job_dir/40x_1024px_0px_overlap"
    for fn, pre in (("_excluded_slides.csv", "c3a3bc4c"), ("_titan_infeasible_slides.csv", "2ddba0b7")):
        check(T, f"Panel D: sha256 {fn} starts {pre} (spec)", sha(use(jd / fn)).startswith(pre), pre, sha(jd / fn)[:8], hard=True)
    ze, inf = pd.read_csv(jd / "_excluded_slides.csv"), pd.read_csv(jd / "_titan_infeasible_slides.csv")
    num_check(T, "Panel D: strict zero retained patches 21", len(ze), "21")
    check(T, "Panel D: strict zero-patch list contains MEL 15110", 15110 in set(ze.slide), True, 15110 in set(ze.slide))
    check(T, "Panel D: strict single-patch slides = 12663, 12668, 12674, 12833, 12839, 12883, 6840",
          set(inf.slide) == {12663, 12668, 12674, 12833, 12839, 12883, 6840} and (inf.n_kept == 1).all(),
          "7 listed IDs, n_kept 1", sorted(inf.slide))
    num_check(T, "Panel D: strict not encoded 28 / encoded 2,893", (len(ze) + len(inf)) * 10000 + 2921 - len(ze) - len(inf), "282893")
    D = pd.DataFrame(spec_D, columns=[c.strip() for c in spec_D_hdr[0].strip().strip("|").split("|")])
    # ---- Notes 1 and 3
    num_check(T, "Note 1: median 1,475 retained patches", allk.n_kept_h5_median, "1475")
    num_check(T, "Note 1: IQR low 521", allk.n_kept_h5_q1, "521")
    num_check(T, "Note 1: IQR high 2,907", allk.n_kept_h5_q3, "2907")
    num_check(T, "Note 1: median 15 components", allk.n_components_median, "15")
    num_check(T, "Note 1: median largest-component share 0.66", allk.largest_component_frac_median, "0.66")
    for c_, v_ in (("n_largest_ge_8", "2727"), ("n_largest_ge_32", "2598"), ("n_largest_ge_128", "2289")):
        num_check(T, f"Note 1: {c_} = {v_}", allk[c_], v_)
    tg = kept[~kept.label.isin(["ALL", "NORMAL", "MISC"])].set_index("label").largest_component_frac_median
    check(T, "Note 1: tumour-group medians range EWS (min) to ACT/ACC (max)", tg.idxmin() == "EWS" and tg.idxmax() == "ACT/ACC",
          "EWS/ACT/ACC", f"{tg.idxmin()}/{tg.idxmax()}")
    num_check(T, "Note 1: EWS 0.50", tg["EWS"], "0.50")
    num_check(T, "Note 1: ACT/ACC 0.96", tg["ACT/ACC"], "0.96")
    num_check(T, "Note 1: NORMAL 0.22", kept[kept.label == "NORMAL"].largest_component_frac_median.iloc[0], "0.22")
    rg = rd(OUT_ROOT / "r2_6_titan_arm1_regen_20261004/regenerated_slide_summary.csv")
    num_check(T, "Note 3: 21 regenerated all-zero embeddings", len(rg), "21")
    gc = rg.group.value_counts().to_dict()
    expg = {"WT": 6, "OS": 3, "SS": 3, "RMS": 2, "LIVER": 2, "EWS": 1, "NB": 1, "RB": 1, "ACT/ACC": 1, "NORMAL": 1}
    check(T, "Note 3: group counts WT 6, OS 3, SS 3, RMS 2, LIVER 2, one each EWS/NB/RB/ACT/ACC/NORMAL", gc == expg, expg, gc)
    num_check(T, "Note 3: nearest neighbour in same group for 19 of 21", int(rg.nn_same_group.sum()), "19")
    for d_, lab, vals_ in (("r2_6_titan_common_pool_l1_20261004", "before repair", ("0.0074", "0.0008", "0.0143")),
                           ("r2_6_titan_common_pool_l1_repaired_arm1_20261004", "after repair", ("0.0023", "-0.0038", "0.0087"))):
        s_ = rd(OUT_ROOT / d_ / "summary_common_pool.csv")
        r_ = one(s_, population="common_pool_2564", metric="micro_hit@5", arm="delta")
        check(T, f"Note 3 ({lab}): population common_pool_2564 has 2,564 queries", r_.n_queries == 2564, 2564, r_.n_queries)
        for w, sv in zip(("estimate", "lo", "hi"), vals_):
            num_check(T, f"Note 3 ({lab}): S2 - no filter micro hit@5 {w} = {sv}", r_[w], sv)
    capt = V("legend (verbatim)", legend) + V("Panel A heading (verbatim)", pa_intro) + V("Panel B heading (verbatim)", pb_intro) \
        + V("Panel C heading (verbatim)", pc_intro) + V("Panel D heading (verbatim)", pd_intro) \
        + V("table notes (verbatim; to print under S22)", notes_txt) + V("spec numbers source (verbatim; not printed)", numsrc)
    write_table("Table S22 (S-T1; provisional)", "Table_S22_TITAN_tumour_filtering", heading, hl, capt,
                [f"Population {POP22} (2,562 queries / 1,772 patients) from the seven r2_6_titan_intersection_*_20261004 bundles; "
                 "delta = arm2 - arm1 of each bundle. Panel A 4 dp, panel B signed 4 dp, panel C signed 3 dp; signs follow the "
                 "stored value (Unicode minus; exact zero '+0.0000'), matching every spec cell. Bold cells (XLSX) = interval excludes zero; the CSVs carry the flag.",
                 "Panel D is copied from the spec and its counts verified against the S2 connectivity file and the strict "
                 "exclusion lists; the group assignment of strict exclusions (20 NORMAL, 1 MEL; 6 NORMAL, 1 OS) uses COMET slide "
                 "metadata outside the bound set and is NOT verified here (slide IDs and counts are).",
                 "Not verified (sources outside the bound set): note 2 classifier metrics and 339/164, 1,659/1,695/1,505 counts "
                 "(Astra session review); note 3 '20 patients'; note 5 (46-108 shift, 47-85 discordant; review 62c9df2e).",
                 "RR-5 (r2_1_virchow2_fp32_main_20261013/run_003/rr5_replacement_001/) is not cited by the R2.6-T7 spec and was not used."],
                [("panel_A_points", A, pd.DataFrame(rawA)), ("panel_B_differences", B, RB),
                 ("panel_C_groups_excl_zero", C, ex.drop(columns=["Group (queries/patients)", "Metric"])), ("panel_D_not_encoded", D, None)])
    bold_cells(OUT / "Table_S22_TITAN_tumour_filtering.xlsx", "panel_B_differences", boldB)
    S22_SHARED["PC"] = PC


S22_SHARED: dict = {}


# ----------------------------------------------------------------------------- S27 (R2.6-T8)
S27_ORDER = ["b100_argmax_vs_repaired", "b5_argmax_vs_repaired", "b100_strict_vs_repaired", "repaired_vs_arm2l1",
             "b100_argmax_vs_arm2l1", "b5_argmax_vs_arm2l1", "b100_strict_vs_arm2l1"]
S27_METRICS = ["micro_hit@1", "macro_hit@1", "micro_hit@3", "macro_hit@3", "micro_hit@5", "macro_hit@5", "micro_hit@10",
               "macro_hit@10", "micro_plurality@5", "macro_plurality@5", "micro_precision@5", "macro_precision@5"]
UN = "unannotated_patient_queries"


def build_s27() -> None:
    T = "S27"
    hl, blk = block("### R2.6-T8 ")
    heading = blk[0][4:]
    for fn, pre in (("summary_all_bundles.csv", "5c1d7c75"), ("per_group_all_bundles.csv", "e5b8b4a4"), ("README.md", "f863fa3f")):
        check(T, f"sha256 {fn} starts {pre} (spec)", sha(use(UNANN / fn)).startswith(pre), pre, sha(UNANN / fn)[:8], hard=True)
    for bn in S27_ORDER:
        use(UNANN / bn / "README.md")
    s, g = pd.read_csv(UNANN / "summary_all_bundles.csv"), pd.read_csv(UNANN / "per_group_all_bundles.csv")
    lab = {bn: BUNDLES[bn][2] for bn in S27_ORDER}
    s["contrast"], g["contrast"] = s.bundle.map(lab), g.bundle.map(lab)
    check(T, "every bundle maps to a contrast label", s.contrast.notna().all() and g.contrast.notna().all(), "mapped", "")
    sd, gd = s[s.arm == "delta"].copy(), g[g.arm == "delta"].copy()
    for d in (sd, gd):
        d["excludes_zero"] = (d.lo > 0) | (d.hi < 0)
    u, ug = sd[sd.population == UN], gd[gd.population == UN]
    check(T, "unannotated subset: 2,223 queries / 1,608 patients on every delta row",
          bool((u.n_queries == 2223).all() and (u.n_patients == 1608).all() and (ug.n_queries == 2223).all()), "2223/1608",
          f"{u.n_queries.unique()}/{u.n_patients.unique()}")
    check(T, "unannotated subset: ci_available True and B_effective 5,000 on every delta row",
          bool(u.ci_available.all() and ug.ci_available.all() and (u.B_effective == 5000).all()), True,
          f"{u.ci_available.all()}/{ug.ci_available.all()}")
    num_check(T, "unannotated summary delta intervals = 84 (7 x 12)", len(u), "84")
    num_check(T, "unannotated per-group delta intervals = 588 (7 x 14 x 6)", len(ug), "588")
    num_check(T, "Panel A note: 6 of 84 exclude zero", int(u.excludes_zero.sum()), "6")
    check(T, "Panel A note: all six positive", bool((u[u.excludes_zero].estimate > 0).all()), True, "")
    s22m = set(TITAN_METRICS)
    num_check(T, "Panel A note: 6 of 63 on the nine S22 summary metrics", int(u[u.metric.isin(s22m)].excludes_zero.sum()), "6")
    full = sd[sd.population == POP22]
    num_check(T, "Panel A note: full population 4 of 84", int(full.excludes_zero.sum()), "4")
    num_check(T, "Panel A note: full population 2 of 63 (= S22 panel B)", int(full[full.metric.isin(s22m)].excludes_zero.sum()), "2")
    fp = one(full, contrast="S2 \u2212 no filter", metric="macro_plurality@5")
    check(T, "Panel A note: S2 - no filter macro plurality@5 (full) upper bound -0.000002",
          bool(fp.excludes_zero) and round(fp.hi, 6) == -0.000002, -0.000002, f"{fp.hi:.3g}")
    # full-population rows reproduce S22 bundles
    worst = 0.0
    for bn in S27_ORDER:
        src = pd.read_csv(OUT_ROOT / f"r2_6_titan_intersection_{bn}_20261004/summary_common_pool.csv")
        src = src[(src.population == POP22) & (src.arm == "delta")].set_index("metric")
        f_ = full[full.bundle == bn].set_index("metric")
        for m in TITAN_METRICS:
            worst = max(worst, abs(f_.loc[m, "estimate"] - src.loc[m, "estimate"]))
    check(T, "full-population delta estimates equal the S22 intersection bundles (nine S22 metrics)", worst < 1e-12, "<1e-12",
          f"{worst:.1e}")
    ugf = gd[gd.population == POP22]
    num_check(T, "Panel B intro: 33 of 588 exclude zero", int(ug.excludes_zero.sum()), "33")
    s22g = ug[ug.metric.isin(["hit@5", "plurality@5"])]
    num_check(T, "Panel B intro: 9 of 196 on S22 per-group metrics", int(s22g.excludes_zero.sum()), "9")
    num_check(T, "Panel B intro / note: hit@5 2 of the 9", int(s22g[s22g.metric == "hit@5"].excludes_zero.sum()), "2")
    check(T, "Panel B intro: none of these nine has a bound at zero",
          not bool(((s22g.lo == 0) | (s22g.hi == 0))[s22g.excludes_zero].any()), True, "")
    num_check(T, "Panel B intro: full population 38 of 588", int(ugf.excludes_zero.sum()), "38")
    num_check(T, "Panel B intro: full population 9 of 196",
              int(ugf[ugf.metric.isin(["hit@5", "plurality@5"])].excludes_zero.sum()), "9")
    # ---- Panel A (3 metrics + scope)
    spec_A = md_table(blk, r"^\| Difference \| micro hit@1 \| macro hit@1")
    cols = ["micro_hit@1", "macro_hit@1", "macro_hit@5"]
    rowsA, boldA, rawA = [], [], []
    for i, bn in enumerate(S27_ORDER):
        row = {"Difference": lab[bn]}
        for j, m in enumerate(cols):
            r = one(u, bundle=bn, metric=m)
            row[m.replace("_", " ")] = sci(r.estimate, r.lo, r.hi, 4)
            if r.excludes_zero:
                boldA.append((i, j + 1))
        scope = u[u.bundle == bn].leak_free_scope.unique()
        assert len(scope) == 1, scope
        row["leak_free_scope"] = scope[0]
        rowsA.append(row)
    A = pd.DataFrame(rowsA)
    keep = ["bundle", "contrast", "arm1_words", "arm2_words", "leak_free_scope", "population", "population_role", "n_queries",
            "n_patients", "metric", "arm", "estimate", "lo", "hi", "B_effective", "n_zero_support_draws", "ci_available",
            "ci_degenerate", "excludes_zero"]
    rawA = u[u.metric.isin(cols)][keep].assign(display=lambda d: [sci(e, l, h, 4) for e, l, h in zip(d.estimate, d.lo, d.hi)])
    for r in spec_A:
        bn = [k for k, v in lab.items() if v == r[0]][0]
        for m, cell in zip(cols, r[1:4]):
            mt = CELL.match(cell)
            src = one(u, bundle=bn, metric=m)
            for w, sv in zip(("estimate", "lo", "hi"), mt.groups()):
                num_check(T, f"Panel A {r[0]} {m} {w} = {sv}", src[w], sv.replace(MINUS, "-"))
            check(T, f"Panel A {r[0]} {m}: display identical to spec", sci(src.estimate, src.lo, src.hi, 4) == cell.strip("*"),
                  cell.strip("*"), sci(src.estimate, src.lo, src.hi, 4))
            check(T, f"Panel A {r[0]} {m}: bold <=> excludes zero", cell.startswith("**") == bool(src.excludes_zero),
                  cell.startswith("**"), bool(src.excludes_zero))
        check(T, f"Panel A {r[0]}: leak_free_scope verbatim", r[4] == A.loc[S27_ORDER.index(bn), "leak_free_scope"], r[4],
              A.loc[S27_ORDER.index(bn), "leak_free_scope"])
    # Panel A note list (6 rows)
    spec_An = md_table(blk, r"^\| Difference \| Metric \| \u0394")
    ex = u[u.excludes_zero].assign(_o=lambda d: d.bundle.map(S27_ORDER.index)).sort_values(["_o", "metric"], kind="stable")
    An = pd.DataFrame({"Difference": ex.contrast, "Metric": ex.metric.str.replace("_", " "),
                       "\u0394 [95% CI]": [sci(e, l, h, 4) for e, l, h in zip(ex.estimate, ex.lo, ex.hi)]})
    check(T, "Panel A note list: spec rows = source excluding-zero rows",
          {(r[0], r[1]) for r in spec_An} == set(zip(An.Difference, An.Metric)), len(spec_An), len(An))
    for r in spec_An:
        src = ex[(ex.contrast == r[0]) & (ex.metric == r[1].replace(" ", "_"))]
        if len(src) == 1:
            for w, sv in zip(("estimate", "lo", "hi"), CELL.match(r[2]).groups()):
                num_check(T, f"Panel A note {r[0]} {r[1]} {w} = {sv}", src.iloc[0][w], sv.replace(MINUS, "-"))
    # full 12-metric unannotated summary (supports the legend's 12-metric set)
    U12 = u.assign(_o=u.bundle.map(S27_ORDER.index), _m=u.metric.map(S27_METRICS.index)).sort_values(["_o", "_m"])[keep]
    U12 = U12.assign(display=[sci(e, l, h, 4) for e, l, h in zip(U12.estimate, U12.lo, U12.hi)])
    # ---- Panel B
    spec_B = md_table(blk, r"^\| Difference \| Group \(queries/patients\)")
    eb = ug[ug.excludes_zero].copy()
    eb["Group (queries/patients)"] = [f"{a} ({q}/{n})" for a, q, n in zip(eb.group, eb.n_group_queries, eb.n_group_patients)]
    eb["Metric"] = eb.metric
    eb["\u0394 [95% CI]"] = [sci(e, l, h, 3) + ("\u1d43" if h < 0 and round(h, 3) == 0 else "") for e, l, h in
                             zip(eb.estimate, eb.lo, eb.hi)]
    mo = {"hit@1": 0, "hit@3": 1, "hit@5": 2, "hit@10": 3, "plurality@5": 4, "precision@5": 5}
    # spec order: group, metric (hit@10 before hit@5 in MPNST; follow the spec's printed order where it matches)
    spec_keys = [(r[0], r[1], r[2]) for r in spec_B]
    src_keys = list(zip(eb.contrast, eb["Group (queries/patients)"], eb.Metric))
    check(T, "Panel B: spec rows = source excluding-zero rows (contrast, group (n/n), metric)", set(spec_keys) == set(src_keys),
          sorted(set(spec_keys) - set(src_keys)), sorted(set(src_keys) - set(spec_keys)))
    pos = {k: i for i, k in enumerate(spec_keys)}
    eb["_o"] = [pos.get(k, 10 ** 6 + mo[k[2]]) for k in src_keys]
    eb = eb.sort_values("_o", kind="stable").drop(columns="_o")
    for r in spec_B:
        src = eb[(eb.contrast == r[0]) & (eb["Group (queries/patients)"] == r[1]) & (eb.Metric == r[2])]
        if len(src) == 1:
            for w, sv in zip(("estimate", "lo", "hi"), CELL.match(r[3]).groups()):
                num_check(T, f"Panel B {r[0]} {r[1]} {r[2]} {w} = {sv}", src.iloc[0][w], sv.replace(MINUS, "-"))
            check(T, f"Panel B {r[0]} {r[1]} {r[2]}: display identical to spec cell", src.iloc[0]["\u0394 [95% CI]"] == r[3],
                  r[3], src.iloc[0]["\u0394 [95% CI]"])
    liv = one(ug, contrast="S2 \u2212 no filter", group="LIVER", metric="precision@5")
    check(T, "note a: LIVER S2 - no filter precision@5 upper bound -1.08e-16", f"{liv.hi:.2e}" == "-1.08e-16", "-1.08e-16",
          f"{liv.hi:.3g}")
    B = eb[["contrast", "Group (queries/patients)", "Metric", "\u0394 [95% CI]"]].rename(columns={"contrast": "Difference"})
    rawB = eb[["bundle", "contrast", "population", "group", "metric", "n_group_queries", "n_group_patients", "estimate", "lo", "hi",
               "B_effective", "n_zero_support_draws", "ci_available", "ci_degenerate", "\u0394 [95% CI]"]]
    # ---- annotated-subset statements in the Notes
    an = sd[sd.population == "annotated_patient_queries"]
    num_check(T, "Population: annotated subset 339 queries", an.n_queries.iloc[0], "339")
    num_check(T, "Population: annotated subset 164 patients", an.n_patients.iloc[0], "164")
    mac = an[an.metric.str.startswith("macro")]
    check(T, "Notes: annotated macro intervals unavailable, B_effective 4,192 (808 lack support)",
          bool((~mac.ci_available).all() and (mac.B_effective == 4192).all() and (mac.n_zero_support_draws == 808).all()),
          "4192/808, ci_available False", f"{mac.B_effective.unique()}/{mac.n_zero_support_draws.unique()}/{mac.ci_available.unique()}")
    mic = an[an.metric.str.startswith("micro") & an.ci_available]
    num_check(T, "Notes: 42 available annotated micro intervals", len(mic), "42")
    mex = mic[(mic.lo > 0) | (mic.hi < 0)]
    num_check(T, "Notes: one annotated micro interval excludes zero", len(mex), "1")
    if len(mex) == 1:
        r = mex.iloc[0]
        check(T, "Notes: it is b5a - S2 micro hit@3", (r.contrast, r.metric) == ("b5a \u2212 S2", "micro_hit@3"),
              "b5a - S2 micro_hit@3", (r.contrast, r.metric))
        for w, sv in zip(("estimate", "lo", "hi"), ("-0.027", "-0.047", "-0.009")):
            num_check(T, f"Notes: annotated b5a - S2 micro hit@3 {w} = {sv}", r[w], sv)
    ga = gd[(gd.population == "annotated_patient_queries")]
    nog = sorted(ga.groupby("group").ci_available.any().loc[lambda x: ~x].index)
    check(T, "Notes: seven annotated groups lack per-group intervals (ACT/ACC, DSRCT, MEL, MPNST, MRT, NF, RB)",
          nog == ["ACT/ACC", "DSRCT", "MEL", "MPNST", "MRT", "NF", "RB"], "ACT/ACC, DSRCT, MEL, MPNST, MRT, NF, RB", nog)
    legend = take(blk, r"^> Supplementary Table S27\.")
    capt = V("legend (verbatim)", legend) + V("spec Source (verbatim; not printed)", take(blk, r"^\*\*Source:\*\*")) \
        + V("spec Population (verbatim)", take(blk, r"^\*\*Population:\*\*")) \
        + V("Panel A heading (verbatim)", take(blk, r"^\*\*Panel A —")) \
        + V("Panel A note (verbatim)", take(blk, r"^\*\*Panel A note\.\*\*")) \
        + V("Panel B heading (verbatim)", take(blk, r"^\*\*Panel B")) \
        + V("note a (verbatim)", take(blk, "^\u1d43 Upper bound")) \
        + V("spec Notes (verbatim; provenance, not all for print)", take(blk, r"^\*\*Notes:\*\*"))
    write_table("Table S27 (S-T2; provisional)", "Table_S27_TITAN_unannotated_queries", heading, hl, capt,
                ["Rows population == unannotated_patient_queries, arm == delta (CSV delta = arm2 - arm1; displayed as "
                 "first-named minus second-named arm). Signed 4 dp (panel A) / 3 dp (panel B); bold cells in the XLSX = interval "
                 "excludes zero; ᵃ marks an upper bound that is negative but rounds to zero.",
                 "Sheet summary_12_metrics_unann (84 rows) carries the full 12-metric unannotated-subset summary described in the "
                 "legend; panel A displays the spec's three columns. Annotated-subset intervals are not tabulated (spec).",
                 "Not verifiable from the bound sources: the S2 classifier counts (231 slides; 197 -> 187 patients; 34 unmapped) "
                 "and 164 of 192 CONCH training patients (ledger / production README)."],
                [("panel_A_unannotated", A, rawA), ("panel_A_note_excl_zero", An, ex.drop(columns="_o")[keep]),
                 ("panel_B_groups_excl_zero", B, rawB), ("summary_12_metrics_unann", U12, None)])
    bold_cells(OUT / "Table_S27_TITAN_unannotated_queries.xlsx", "panel_A_unannotated", boldA)


# ----------------------------------------------------------------------------- package files and report
SOURCES = {
    "Tables S18-S21": ["r1_3_external_synthesis_20260930/{broad_group_primary, paired_primary_differences, coverage_and_hit_yield, "
                       "expanded_secondary_retrieval_points, all_returned_label_slots, failure_conditioned_label_slots, "
                       "first_relevant_rank_distribution, matched_endpoint_transitions, separate_coverage_losses, "
                       "reference_recurrence, aggregate_context, descriptive_strata_points, wt_source_by_pattern_membership}.csv",
                       "cross-check only: figures_20261021/fig4_additional_collections/source_data.csv"],
    "Table S-R4.a-1": ["r4a_vii_whole_patient_20261015/production/{summary,paired_differences}.csv (arm c1e)",
                       "c1e_slide_unmasked/slide_ci_uni2_ot_cosine_top{1,3,5,10}_{generous_majority,absolute}_SAME_PATIENT_ALLOWED.json",
                       "docs/adr/0009-evaluation-protocol-freeze.md; docs/analysis/c1e_slide_level_masking_delta.md"],
    "Table S-R4.a-2": ["r4a_vii_whole_patient_20261015/production/{summary,paired_differences}.csv (arm c1d)",
                       "r4a_vii_whole_patient_20261015/production_c1d_same_class/{summary,paired_differences}.csv, per_pair_frame.csv.gz"],
    "Table S22": ["r2_6_titan_intersection_{repaired_vs_arm2l1, b100_argmax/b5_argmax/b100_strict _vs_repaired, "
                  "_vs_arm2l1}_20261004/{summary,per_class}_common_pool.csv",
                  "r2_6_titan_common_pool_20261003/connectivity_per_group.csv; r2_6_titan_common_pool_l1_20261004/ and "
                  "r2_6_titan_common_pool_l1_repaired_arm1_20261004/summary_common_pool.csv; "
                  "r2_6_titan_arm1_regen_20261004/regenerated_slide_summary.csv; "
                  "r2_6_titan_direct1024_20261004/stage2/b100_strict/job_dir/40x_1024px_0px_overlap/{_excluded_slides,_titan_infeasible_slides}.csv"],
    "Table S27": ["r2_6_titan_unannotated_query_sensitivity_20261005/{summary_all_bundles,per_group_all_bundles}.csv, README.md, "
                  "seven per-bundle README.md; cross-check: the S22 intersection summaries"],
}
PROPOSED = {"Table S18": "S33", "Table S22": "S22", "Table S-R4.a-1": "S27", "Table S-R4.a-2": "S28", "Table S27": "S31"}

FLAGS = [
    ("S18 / Fig. 4", "[SPEC/SOURCE MISMATCH] The revised Fig. 4 legend (edit sheet 10-21 ≈15:45/15:50 amendments) quotes the "
     "CCDI-MCI all-preparation macro hit@5 0.570 [0.511-0.629] and precision@5 0.422 and says 'Source data in Supplementary "
     "Table S18'. Those rows (population allprep_scope_clear, role preparation_sensitivity) are in Table S19 'absolute', not in "
     "S18 as specified by R1.3-T8 (main identities only). Values verified (0.570322 [0.510863, 0.629055]; 0.421692). Not "
     "resolved here: either the legend cites S18 + S19, or S18 gains the two all-prep macro rows (author decision)."),
    ("S-R4.a-1 Block B", "[SPEC/SOURCE MISMATCH] strict-majority macro F1, k = 3, left in: spec 0.782; source JSON "
     "point_estimates.macro_f1 = 0.7814532 -> 0.781 (spec appears to double-round the C1e note's 4-dp 0.7815). Cell shows 0.781 "
     "with the flag."),
    ("S-R4.a-1 Block B", "Strict-majority 'excluded' values (k = 1/3/10) exist only as 4-dp numbers in the C1e note table "
     "(0.7695 / 0.7614 / 0.6911); displayed at 3 dp by ROUND_HALF_UP on the recorded string (0.7695 -> 0.770 = spec). No "
     "full-precision masked strict source for k = 1/3/10 is in the bound set (k = 1 equals the generous value 0.769515 by "
     "definition)."),
    ("S22", "Not verified (outside bound sources): note 2 classifier metrics and 339/164, 1,659/1,695/1,505 counts; note 3 "
     "'20 patients'; note 5 (46-108 / 47-85, review 62c9df2e); Panel D group assignment of strict exclusions (20 NORMAL + MEL "
     "15110; 6 NORMAL + OS 6840) — slide IDs and counts verified, groups need COMET slide metadata."),
    ("S22", "RR-5 (r2_1_virchow2_fp32_main_20261013/run_003/rr5_replacement_001/) is not cited in the R2.6-T7 block (the numbering "
     "map lists 'r2_6_titan_* ; RR-5' for S22); not used."),
    ("S27", "Not verified (outside bound sources): S2 classifier counts 231 / 197 -> 187 / 34 and 164 of 192 CONCH training "
     "patients (ledger / production README)."),
]


def report() -> None:
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    ck = pd.DataFrame(CHECKS)
    ck.to_csv(OUT / "build_checks.csv", index=False)
    bind = {k: {"sha256": sha(p), "mtime": dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
                "bytes": p.stat().st_size} for k, p in sorted(INPUTS.items())}
    (OUT / "input_bindings.json").write_text(json.dumps({"script": str(Path(__file__).resolve().relative_to(REPO)),
                                                         "script_sha256": sha(Path(__file__).resolve()),
                                                         "built": now, "inputs": bind}, indent=2))
    (OUT / "README.md").write_text(
        f"# Supplementary tables, group {GROUP} (assembly only)\n\n"
        f"Built {now} by `scripts/supp_tables/build_B_external_leakage_titan_20261022.py` (read-only on inputs; idempotent; "
        "re-run with `PROSPER_OUT` exported and `/HOME/miniconda3/bin/python3`).\n\n"
        "Each `Table_*.xlsx` starts with a `caption` sheet (verbatim edit-sheet caption/legend/footnotes with block heading "
        "and line, followed by assembly notes), then the data sheets. Each data sheet has a CSV `Table_*__<sheet>.csv` with raw "
        "(full precision) and formatted columns. Table numbers are provisional (edit-sheet IDs); final numbers are assigned at "
        "assembly (see `docs/analysis/supplementary_numbering_map_20261021.md`).\n\n"
        "`build_checks.csv` lists every source/spec check; `input_bindings.json` the bound inputs (sha256, mtime); "
        "`SHA256SUMS.txt` the outputs; `BUILD_REPORT.md` the summary and discrepancies.\n\n"
        "Access: the sources include patient-level restricted COMET/KidRare/CCDI-MCI derivatives; these aggregate tables contain "
        "no raw case identifiers, paths or hashes, but are not authorised for public release until the author approves the "
        "supplement.\n", encoding="utf-8")
    L = [f"# BUILD_REPORT — {GROUP}", "", f"Built {now}; script `scripts/supp_tables/build_B_external_leakage_titan_20261022.py` "
         f"(sha256 {sha(Path(__file__).resolve())[:12]}); output `$PROSPER_OUT/supp_tables_20261022/{GROUP}/`.", "",
         f"Checks: {len(ck)} total, {int((ck.status == 'PASS').sum())} PASS, {int((ck.status != 'PASS').sum())} MISMATCH "
         "(full list `build_checks.csv`). No statistic was computed; values are copied/rounded; the only aggregations are the S20 "
         "row counts defined by the R1.3-T8 spec.", "",
         "## Amendment versions implemented", "",
         "- **S18-S21:** original R1.3-T8 spec (edit sheet heading line recorded below). The 2026-10-21 Fig. 4 single-grid "
         "amendments (≈15:10, ≈15:45, ≈15:50; STV, KidRare, CCDI-MCI FFPE in Fig. 4; five earlier collections to S-R2.1-F1 / "
         "Table S14) change the figure and text citations only (l.113 '(Fig. 4; Supplementary Table S18)'); no later amendment "
         "changes the S18-S21 table scope, and the ledger R1.3 row records no scope change. S18 therefore keeps all 74 rows "
         "(groups + micro + macro for the three main identities); the 68-row Fig. 4 source_data.csv was verified to be an exact "
         "subset of S18 (macro rows included).",
         "- **S-R4.a-1/2:** R4.a-T4/T5 as drafted 10-15; the 10-22 amendment 'A. Fig. 6 legend — supersedes \"Where (2)\" of "
         "R4.a-T4' changes the figure legend only and does not affect either table.",
         "- **S22:** R2.6-T7 as drafted; amendment l.~7523 (10-21) confirms panel A strict micro hit@5 0.9485 and changes no "
         "table value.",
         "- **S27:** R2.6-T8 as drafted (10-05 wording corrections included in the block).", "",
         "## Tables", ""]
    for t in TABLES:
        L += [f"### {t['tid']}", "", f"- Spec: edit sheet l.{t['line']} `### {t['heading']}`",
              f"- Proposed final number (numbering map 10-21, not applied): {PROPOSED.get(' '.join(t['tid'].split(' ')[:2]), 'see map')}",
              f"- Files: {', '.join(t['files'])}",
              "- Sheets (XLSX rows x cols; CSV rows): " + "; ".join(f"`{k}` {a} x {b}; CSV {c}" for k, (a, b, c) in t["rows"].items())]
        L += [f"- Note: {n}" for n in t["notes"]] + [""]
    for k, v in SOURCES.items():
        L.append(f"- Sources {k} (under $PROSPER_OUT unless docs/): " + " | ".join(v))
    L += ["", "## Checks by table", ""]
    for tb, d in ck.groupby("table", sort=False):
        L.append(f"- {tb}: {len(d)} checks, {int((d.status == 'PASS').sum())} PASS, {int((d.status != 'PASS').sum())} MISMATCH")
    L += ["", "## Discrepancies and flags (not resolved by this build)", ""]
    for i, (where, txt) in enumerate(FLAGS, 1):
        L.append(f"{i}. **{where}** — {txt}")
    mm = ck[ck.status != "PASS"]
    L += ["", "### Check mismatches (from build_checks.csv)", ""]
    L += [f"- [{r.table}] {r.check}: expected {r.expected}; observed {r.observed}" for r in mm.itertuples()] or ["- none"]
    L += ["", "## Not built", "", "- Nothing in scope left unbuilt. Final supplementary numbers, Word/rendered incorporation and "
          "the author's reading approval remain outstanding (protocol 8.3.3)."]
    (OUT / "BUILD_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    sums = [f"{sha(p)}  {p.name}" for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "SHA256SUMS.txt"]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    build_r13()
    build_r4a()
    build_s22()
    build_s27()
    report()
    ck = pd.DataFrame(CHECKS)
    print(f"{len(TABLES)} tables; {len(ck)} checks; {int((ck.status != 'PASS').sum())} mismatches -> {OUT}")


if __name__ == "__main__":
    main()
