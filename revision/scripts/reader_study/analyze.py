"""PROSPER reader study, handoff step 4: endpoints (plan 002 §4), secondary/exploratory analyses
(§5 as amended by 003), tables T1–T9 (+ S-tables for §5.1, §5.2, §5.5, §5.8), figures F1–F4.

    python3 scripts/reader_study/analyze.py --readers <reader-login>

Reads the step 1–3 outputs (`readings.csv`, `events_engagement.csv`) of the analysis directory,
plus case material (manifests, LLM text) and counts from the export. Descriptive only: no tests.
Only the readers named in --readers are loaded; rows of any other reader are dropped on read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive as d  # noqa: E402
import endpoints as ep  # noqa: E402
import paired  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXPORT_STAMP = "prod_export_20261027-101111"
OUT_ROOT = Path(os.environ.get("PROSPER_OUT", str(REPO / "prosper_revision_output")))
CDT = dt.timezone(dt.timedelta(hours=-5), "CDT")
ENTITY_MAP = REPO / "docs" / "pathologist_study" / "reader_study_misc_entity_map_20261027.csv"
SUBTYPE_GROUPS = ("RMS", "SS", "LIVER")
SUBTYPE_TERMS_RE = re.compile(r"alveolar|embryonal|hepatoblastoma", re.IGNORECASE)
RETRIEVAL_LIBRARY_N = 2604
POST_HOC = "[post-hoc-motivated]"
TIER_NOTE = "Every row carries the reader's expertise tier; pooled rows appear only with ≥2 readers, beside the per-reader rows."
CODES = ("confirms", "rules_out", "retrieval_inconsistent", "names_off_vocabulary_entity", "needs_ancillary_test",
         "wants_more_clinical_info", "ui_or_material_issue", "summary_misleading", "summary_reframed")
BOOL_COLS = [c for p in "ABC" for c in (f"fav_correct_{p}", f"in_diff_{p}", f"misc_in_{p}", f"normal_in_{p}",
                                         f"fav_correct_{p}_cmap", f"in_diff_{p}_cmap")] + [
    "has_C", "harmful_conf_B", "interrupted_A", "interrupted_B", "interrupted_C", "interrupted_between_phases",
    "engaged_B", "engaged_C_retrieval", "top1_correct", "hit5", "hit10", "majority10_correct", "meta_visible_A",
    "llm_top_correct", "llm_names_truth_leading", "llm_flags_discrepancy", "llm_wording_probability"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_bool(s: pd.Series) -> pd.Series:
    m = {True: True, False: False, "True": True, "False": False}
    out = s.map(m)
    if out.isna().any():
        raise ValueError(f"non-boolean values in {s.name}")
    return out.astype(bool)


# ---------------------------------------------------------------- loading

def load_readings(analysis: Path, readers: List[str]) -> pd.DataFrame:
    r = pd.read_csv(analysis / "readings.csv", keep_default_na=False, na_values=[""])
    r = r[r.reader_id.isin(readers)].copy()
    missing = [x for x in readers if x not in set(r.reader_id)]
    if missing:
        raise SystemExit(f"no readings for {missing}")
    for c in BOOL_COLS:
        r[c] = to_bool(r[c])
    for c in ("comments_A", "comments_B", "comments_C", "cmap_note_A", "cmap_note_B", "cmap_note_C",
              "confirm_sub_AB", "confirm_sub_BC", "confirm_sub_AC", "plurality10_label", "stv_subclass",
              "ccdi_diagnosis"):
        r[c] = r[c].fillna("")
    r["reader_order"] = r.reader_id.map({x: i for i, x in enumerate(readers)})
    return r.sort_values(["reader_order", "case_id"]).reset_index(drop=True)


def reader_sets(r: pd.DataFrame, readers: List[str]):
    sets = [(rid, d.EXPERTISE_TIER[rid], r[r.reader_id == rid]) for rid in readers]
    if len(sets) > 1:
        sets.append(("pooled", "+".join(sorted({t for _, t, _ in sets})), r))
    return sets


def per_reader_only(r: pd.DataFrame, readers: List[str]):
    return [(rid, d.EXPERTISE_TIER[rid], r[r.reader_id == rid]) for rid in readers]


def engagement_sets(sets):
    """002 §3.6: every §5 result for all readings and split by engaged_B (empty subsets omitted)."""
    for label, tier, g in sets:
        yield label, tier, "all", g
        for v in (True, False):
            sub = g[g.engaged_B == v]
            if len(sub):
                yield label, tier, f"engaged_B = {v}", sub


# ---------------------------------------------------------------- generic rows

def strat_row(sub: pd.DataFrame) -> Dict[str, object]:
    n = len(sub)
    row = {"n": n}
    for col, name in (("in_diff_A", "in_diff_A"), ("fav_correct_A", "fav_correct_A"), ("fav_correct_B", "fav_correct_B")):
        k = int(sub[col].sum())
        row[f"{name}_k"] = k
        row[f"{name}_p"] = k / n if n else None
    row["diff_B_minus_A"] = (sub.fav_correct_B.sum() - sub.fav_correct_A.sum()) / n if n else None
    row["A_wrong_B_right"] = int((~sub.fav_correct_A & sub.fav_correct_B).sum())
    row["A_right_B_wrong"] = int((sub.fav_correct_A & ~sub.fav_correct_B).sum())
    for c in ep.CLASS_ORDER:
        row[f"AB_{c.replace(' ', '_')}"] = int((sub.class_AB == c).sum())
    for s in ep.CONFIRM_SUBS:
        row[f"AB_{s}"] = int((sub.confirm_sub_AB == s).sum())
    lk = ep.likert_summary(sub.use_B.tolist())
    for lv in ep.LIKERT_LEVELS:
        row[f"use_B_{lv}"] = lk[f"n_{lv}"]
    row["use_B_median"] = lk["median"]
    return row


def class_rows(label, tier, sub, transition, cls_col, sub_col, changed_mask):
    rows = []
    for x in ep.class_table(sub[cls_col].tolist(), sub[sub_col].tolist()):
        rows.append({"reader_id": label, "expertise_tier": tier, "transition": transition, **x})
    rows.append({"reader_id": label, "expertise_tier": tier, "transition": transition, "class": "n (readings)",
                 "count": len(sub), "proportion": None})
    rows.append({"reader_id": label, "expertise_tier": tier, "transition": transition,
                 "class": "changed = yes (readings)", "count": int(changed_mask.sum()),
                 "proportion": changed_mask.sum() / len(sub) if len(sub) else None})
    return rows


def time_stats(sub: pd.DataFrame) -> Dict[str, object]:
    row = {"n": len(sub)}
    for p in "ABC":
        s = ep.median_iqr(sub[f"sec_{p}"].tolist())
        row.update({f"sec_{p}_median": s["median"], f"sec_{p}_q1": s["q1"], f"sec_{p}_q3": s["q3"]})
    return row


def top10_labels(data: Path, slide_id: str) -> str:
    m = json.loads((data / "manifests" / f"{slide_id}.json").read_text())
    return " ".join(n["label"] for n in sorted(m["slide_neighbours"], key=lambda n: n["rank"])[:10])


# ---------------------------------------------------------------- tables

def build_tables(r: pd.DataFrame, readers: List[str], export: Path, data: Path, events: pd.DataFrame,
                 entity_map_path: Path) -> Dict[str, pd.DataFrame]:
    T: Dict[str, pd.DataFrame] = {}
    llm_tog = events[(events.kind == "llm_panel_toggled") & ~events.auto].groupby(["reader_id", "case_id"]).size()
    r["n_llm_toggled"] = [int(llm_tog.get((a, b), 0)) for a, b in zip(r.reader_id, r.case_id)]
    sets = reader_sets(r, readers)

    # T1 flow (analysed readers only; counts from the export, filtered on read)
    subs = pd.read_csv(export / "submissions_export.csv", dtype=str, keep_default_na=False)
    subs = subs[subs.reader_id.isin(readers)]
    rows = []
    for rid, tier, g in per_reader_only(r, readers):
        s = subs[subs.reader_id == rid]
        pil = s[s.case_id.isin(d.PILOT_CASES)]
        np_ = s[~s.case_id.isin(d.PILOT_CASES)]
        rows.append({"reader_id": rid, "expertise_tier": tier,
                     "submissions_A_nonpilot": int((np_.phase == "A").sum()),
                     "submissions_B_nonpilot": int((np_.phase == "B").sum()),
                     "submissions_C_nonpilot": int((np_.phase == "C").sum()),
                     "pilot_submissions_excluded": len(pil),
                     "readings_A_and_B": len(g), "readings_with_C": int(g.has_C.sum()),
                     "cases_CCDI": int((g.cohort == "CCDI").sum()), "cases_STV": int((g.cohort == "STV").sum()),
                     "qc_gate": "dropped (003 §A)"})
    T["T1_reader_case_flow"] = pd.DataFrame(rows)

    # T2 primary endpoints
    rows = []
    for label, tier, g in sets:
        cases = sorted(g.case_id.unique())
        w = ep.case_weights(len(cases))
        base = {"reader_id": label, "expertise_tier": tier}
        for name, col in (("E1 baseline hit (in_diff_A)", "in_diff_A"), ("E2 fav_correct_A", "fav_correct_A"),
                          ("E2 fav_correct_B", "fav_correct_B")):
            rows.append({**base, "endpoint": name, **ep.proportion_endpoint(g.case_id, g[col], cases, w)})
        pdiff = ep.paired_difference(g.case_id, g.fav_correct_A, g.fav_correct_B, cases, w)
        rows.append({**base, "endpoint": "E2 paired difference fav_correct B − A", "n": pdiff["n"],
                     "estimate": pdiff["estimate"], "boot95_lo": pdiff["boot95_lo"], "boot95_hi": pdiff["boot95_hi"]})
        rows.append({**base, "endpoint": "E2 discordant A-wrong→B-right (count)", "k": pdiff["x_wrong_y_right"], "n": pdiff["n"]})
        rows.append({**base, "endpoint": "E2 discordant A-right→B-wrong (count)", "k": pdiff["x_right_y_wrong"], "n": pdiff["n"]})
        lk = ep.likert_summary(g.use_B.tolist())
        for lv, lab in zip(ep.LIKERT_LEVELS, ("misleading", "not useful", "marginal", "useful", "very useful")):
            rows.append({**base, "endpoint": f"E4 use_B = {lv} ({lab}) (count)", "k": lk[f"n_{lv}"], "n": lk["n"],
                         "estimate": lk[f"n_{lv}"] / lk["n"]})
        rows.append({**base, "endpoint": "E4 use_B median [q1, q3]", "n": lk["n"], "estimate": lk["median"],
                     "note": f"q1 {lk['q1']}, q3 {lk['q3']}"})
    cols = ["reader_id", "expertise_tier", "endpoint", "k", "n", "estimate", "boot95_lo", "boot95_hi",
            "wilson95_lo", "wilson95_hi", "note"]
    T["T2_primary_endpoints"] = pd.DataFrame(rows).reindex(columns=cols)

    # T3 outcome classes
    rows = []
    for label, tier, g in sets:
        chB = g.changed_B == "yes"
        chC = g.changed_C == "yes"
        rows += class_rows(label, tier, g, "A→B", "class_AB", "confirm_sub_AB", chB)
        rows += class_rows(label, tier, g, "B→C", "class_BC", "confirm_sub_BC", chC)
        rows += class_rows(label, tier, g, "A→C", "class_AC", "confirm_sub_AC", chB | chC)
    T["T3_outcome_classes"] = pd.DataFrame(rows)

    # S1 trajectory (§5.1)
    rows = []
    for label, tier, sl, g in engagement_sets(sets):
        cases = sorted(g.case_id.unique())
        w = ep.case_weights(len(cases))
        base = {"reader_id": label, "expertise_tier": tier, "engaged_B_subset": sl}
        for var in ("fav_correct", "in_diff"):
            for p in "ABC":
                rows.append({**base, "measure": f"{var}_{p}", **ep.proportion_endpoint(g.case_id, g[f"{var}_{p}"], cases, w)})
            for x, y in (("A", "B"), ("B", "C"), ("A", "C")):
                pd_ = ep.paired_difference(g.case_id, g[f"{var}_{x}"], g[f"{var}_{y}"], cases, w)
                rows.append({**base, "measure": f"{var} paired difference {y} − {x}", "n": pd_["n"],
                             "estimate": pd_["estimate"], "boot95_lo": pd_["boot95_lo"], "boot95_hi": pd_["boot95_hi"],
                             "note": f"{x}-wrong→{y}-right {pd_['x_wrong_y_right']}; {x}-right→{y}-wrong {pd_['x_right_y_wrong']}"})
        for p in "ABC":
            tr = g[f"truth_rank_{p}"]
            for v in sorted(tr.dropna().astype(int).unique()):
                rows.append({**base, "measure": f"truth_rank_{p} = {v} (count)", "k": int((tr == v).sum()), "n": len(g)})
            rows.append({**base, "measure": f"truth_rank_{p} = not in differential (count)", "k": int(tr.isna().sum()), "n": len(g)})
        for p in "ABC":
            nd = g[f"n_dx_{p}"].astype(int)
            for v in sorted(nd.unique()):
                rows.append({**base, "measure": f"n_dx_{p} = {v} (count)", "k": int((nd == v).sum()), "n": len(g)})
            s = ep.median_iqr(nd.tolist())
            rows.append({**base, "measure": f"n_dx_{p} median [q1, q3]", "n": s["n"], "estimate": s["median"],
                         "note": f"q1 {s['q1']}, q3 {s['q3']}"})
    T["S1_trajectory_5_1"] = pd.DataFrame(rows).reindex(
        columns=["reader_id", "expertise_tier", "engaged_B_subset", "measure", "k", "n", "estimate", "boot95_lo", "boot95_hi",
                 "wilson95_lo", "wilson95_hi", "note"])

    # S2 confidence (§5.2)
    rows = []
    for label, tier, sl, g in engagement_sets(sets):
        base = {"reader_id": label, "expertise_tier": tier, "engaged_B_subset": sl}
        for p in "ABC":
            lk = ep.likert_summary(g[f"conf_{p}"].tolist())
            rows.append({**base, "measure": f"conf_{p}", "subset": "all", **{f"n_{lv}": lk[f"n_{lv}"] for lv in ep.LIKERT_LEVELS},
                         "n": lk["n"], "median": lk["median"], "q1": lk["q1"], "q3": lk["q3"],
                         "mean": float(g[f"conf_{p}"].mean())})
        for dcol in ("dconf_BA", "dconf_CB"):
            for fa in (True, False):
                sub = g[g.fav_correct_A == fa][dcol]
                s = ep.median_iqr(sub.tolist())
                dist = "; ".join(f"{int(v):+d}: {int((sub == v).sum())}" for v in sorted(sub.unique()))
                rows.append({**base, "measure": dcol, "subset": f"fav_correct_A = {fa}", "n": s["n"], "median": s["median"],
                             "q1": s["q1"], "q3": s["q3"], "mean": float(sub.mean()) if len(sub) else None,
                             "distribution": dist})
        rows.append({**base, "measure": "harmful_conf_B (conf_B > conf_A and not fav_correct_B)", "subset": "all",
                     "n": len(g), "count": int(g.harmful_conf_B.sum())})
    T["S2_confidence_5_2"] = pd.DataFrame(rows)

    # T4 strata (§5.3)
    lib = pd.read_csv(export / "study_config" / "slides_index.csv", dtype=str, keep_default_na=False)
    group_counts = lib[lib.kind == "library"].label.value_counts().to_dict()
    tert = ep.group_tertiles(group_counts)
    r["library_support_tertile"] = r.reference.map(tert)
    r["cohort_tier"] = r.cohort + " | " + r.difficulty_tier
    rows = []
    strat_defs = [("hit5", "hit5"), ("majority10_correct", "majority10_correct"), ("top1_correct", "top1_correct"),
                  ("cohort", "cohort"), ("difficulty tier (per cohort, as stored)", "cohort_tier"),
                  ("library-support tertile of true group", "library_support_tertile")]
    for label, tier, sl, g in engagement_sets(reader_sets(r, readers)):
        for sname, col in strat_defs:
            for level, sub in g.groupby(col, sort=True):
                rows.append({"reader_id": label, "expertise_tier": tier, "engaged_B_subset": sl,
                             "stratifier": sname, "level": level, **strat_row(sub)})
    T["T4_strata_5_3"] = pd.DataFrame(rows)
    T["_library_groups"] = pd.DataFrame(
        [{"group": gname, "library_slides_in_index": n, "tertile": tert[gname]}
         for gname, n in sorted(group_counts.items(), key=lambda kv: kv[1])])

    # T5 engagement (§5.4)
    rows = []
    for label, tier, g in sets:
        base = {"reader_id": label, "expertise_tier": tier}
        n = len(g)
        rows.append({**base, "measure": "engaged_B", "value": "True", "count": int(g.engaged_B.sum()), "n": n,
                     "proportion": g.engaged_B.mean()})
        for col in ("n_hit_opened", "n_neighbour_opened", "n_roi_selected", "n_heatmap_toggled",
                    "n_search_mode_changed", "n_pecan_opened", "n_retrieval_events_C", "n_llm_toggled"):
            for v, k in g[col].value_counts().sort_index().items():
                rows.append({**base, "measure": col, "value": str(int(v)), "count": int(k), "n": n, "proportion": k / n})
            rows.append({**base, "measure": f"{col} (total events)", "value": "", "count": int(g[col].sum()), "n": n})
        dr = g.deepest_rank_opened
        for v, k in dr.dropna().astype(int).value_counts().sort_index().items():
            rows.append({**base, "measure": "deepest_rank_opened", "value": str(v), "count": int(k), "n": n, "proportion": k / n})
        rows.append({**base, "measure": "deepest_rank_opened", "value": "none opened", "count": int(dr.isna().sum()),
                     "n": n, "proportion": dr.isna().mean()})
        rows.append({**base, "measure": "readings with ≥1 heatmap toggle (Step 2)", "value": "",
                     "count": int((g.n_heatmap_toggled > 0).sum()), "n": n})
        rows.append({**base, "measure": "engaged_C_retrieval", "value": "True", "count": int(g.engaged_C_retrieval.sum()),
                     "n": n, "proportion": g.engaged_C_retrieval.mean()})
    T["T5a_engagement_5_4"] = pd.DataFrame(rows)
    rows = []
    for label, tier, g in sets:
        for level, sub in g.groupby("engaged_B", sort=True):
            s = ep.median_iqr(sub.sec_B.tolist())
            rows.append({"reader_id": label, "expertise_tier": tier, "engaged_B": level, **strat_row(sub),
                         "sec_B_median": s["median"], "sec_B_q1": s["q1"], "sec_B_q3": s["q3"]})
    T["T5b_by_engagement_5_4"] = pd.DataFrame(rows)

    # S5 usefulness vs need (§5.5)
    rows = []
    for label, tier, sl, g in engagement_sets(sets):
        for use, groupings in (("use_B", (("fav_correct_A", "fav_correct_A"), ("class A→B", "class_AB"))),
                               ("use_C", (("fav_correct_A", "fav_correct_A"), ("class B→C", "class_BC")))):
            for gname, col in groupings:
                for level, sub in g.groupby(col, sort=True):
                    lk = ep.likert_summary(sub[use].tolist())
                    rows.append({"reader_id": label, "expertise_tier": tier, "engaged_B_subset": sl,
                                 "rating": use, "by": gname, "level": level,
                                 **{f"n_{lv}": lk[f"n_{lv}"] for lv in ep.LIKERT_LEVELS}, "n": lk["n"],
                                 "median": lk["median"], "q1": lk["q1"], "q3": lk["q3"]})
    T["S5_usefulness_vs_need_5_5"] = pd.DataFrame(rows)

    # T6 time (§5.6)
    rows, irows = [], []
    for label, tier, g in sets:
        base = {"reader_id": label, "expertise_tier": tier}
        g = g.copy()
        g["chrono_q"] = [ep.chrono_quartile(int(rk), int((r.reader_id == rid).sum()))
                         for rk, rid in zip(g.chrono_rank, g.reader_id)]
        interrupted = g.interrupted_A | g.interrupted_B | g.interrupted_C | g.interrupted_between_phases
        rows.append({**base, "grouping": "all", "level": "", **time_stats(g)})
        rows.append({**base, "grouping": "sensitivity: excluding interrupted", "level": "", **time_stats(g[~interrupted])})
        for gname, col in (("cohort", "cohort"), ("engaged_B", "engaged_B"), ("session_id", "session_id")):
            for level, sub in g.groupby(col, sort=True):
                rows.append({**base, "grouping": gname, "level": level, **time_stats(sub)})
        for q in (1, 4):
            rows.append({**base, "grouping": "chronological quartile", "level": f"Q{q}" + (" (first)" if q == 1 else " (last)"),
                         **time_stats(g[g.chrono_q == q])})
        for _, x in g[interrupted].iterrows():
            irows.append({**base, "case_id": x.case_id, "interrupted_A": x.interrupted_A, "interrupted_B": x.interrupted_B,
                          "interrupted_C": x.interrupted_C, "interrupted_between_phases": x.interrupted_between_phases,
                          "sec_A": x.sec_A, "sec_B": x.sec_B, "sec_C": x.sec_C, "gap_AB_s": x.gap_AB_s, "gap_BC_s": x.gap_BC_s})
    T["T6_reading_time_5_6"] = pd.DataFrame(rows)
    T["T6b_interrupted_readings"] = pd.DataFrame(irows, columns=[
        "reader_id", "expertise_tier", "case_id", "interrupted_A", "interrupted_B", "interrupted_C",
        "interrupted_between_phases", "sec_A", "sec_B", "sec_C", "gap_AB_s", "gap_BC_s"])

    # T7 LLM (§5.9)
    neg_only = {}
    for sid in r.slide_id.unique():
        neg_only[sid] = ep.discrepancy_negation_only((data / "llm" / f"{sid}.txt").read_text())
    r["llm_discrepancy_negation_only"] = r.slide_id.map(neg_only)
    rows, crows = [], []
    for label, tier, sl, g in engagement_sets(reader_sets(r, readers)):
        base = {"reader_id": label, "expertise_tier": tier, "engaged_B_subset": sl}
        n = len(g)
        lk = ep.likert_summary(g.use_C.tolist())
        for lv in ep.LIKERT_LEVELS:
            rows.append({**base, "section": "use_C", "measure": f"use_C = {lv} (count)", "count": lk[f"n_{lv}"], "n": n})
        rows.append({**base, "section": "use_C", "measure": "use_C median [q1, q3]", "value": lk["median"],
                     "note": f"q1 {lk['q1']}, q3 {lk['q3']}", "n": n})
        chC = g.changed_C == "yes"
        rows.append({**base, "section": "change at C", "measure": "changed_C = yes", "count": int(chC.sum()), "n": n,
                     "value": chC.mean()})
        for a, b_, nm in ((False, True, "fav_correct B-wrong→C-right"), (True, False, "fav_correct B-right→C-wrong"),
                          (True, True, "fav_correct right at B and C"), (False, False, "fav_correct wrong at B and C")):
            rows.append({**base, "section": "change at C", "measure": nm,
                         "count": int(((g.fav_correct_B == a) & (g.fav_correct_C == b_)).sum()), "n": n})
        s = ep.median_iqr(g.dconf_CB.tolist())
        rows.append({**base, "section": "Δconf B→C", "measure": "dconf_CB median [q1, q3]", "value": s["median"],
                     "note": "; ".join(f"{int(v):+d}: {int((g.dconf_CB == v).sum())}" for v in sorted(g.dconf_CB.unique())),
                     "n": n})
        for col in ("llm_top_correct", "llm_names_truth_leading", "llm_wording_probability", "llm_flags_discrepancy"):
            rows.append({**base, "section": "summary properties (per case read)", "measure": col,
                         "count": int(g[col].sum()), "n": n, "value": g[col].mean()})
        rows.append({**base, "section": "summary properties (per case read)",
                     "measure": "llm_flags_discrepancy true only through a negation", "count": int(g.llm_discrepancy_negation_only.sum()), "n": n})
        for col in ("llm_top_correct", "llm_names_truth_leading"):
            for lv in (True, False):
                for oc in (True, False):
                    rows.append({**base, "section": "summary vs reader outcome at C", "measure": f"{col} = {lv} & fav_correct_C = {oc}",
                                 "count": int(((g[col] == lv) & (g.fav_correct_C == oc)).sum()), "n": n})
        for lv in (True, False):
            for c in ep.CLASS_ORDER:
                rows.append({**base, "section": "summary vs reader outcome at C", "measure": f"llm_top_correct = {lv} & class B→C = {c}",
                             "count": int(((g.llm_top_correct == lv) & (g.class_BC == c)).sum()), "n": n})
        contra = g[g.llm_top != g.favoured_B]
        rows.append({**base, "section": "summary top group ≠ reader's eff_B favoured", "measure": "readings",
                     "count": len(contra), "n": n})
        for nm, mask in (("summary right, reader wrong", contra.llm_top_correct & ~contra.fav_correct_B),
                         ("reader right, summary wrong", ~contra.llm_top_correct & contra.fav_correct_B),
                         ("both wrong", ~contra.llm_top_correct & ~contra.fav_correct_B)):
            rows.append({**base, "section": "summary top group ≠ reader's eff_B favoured", "measure": nm,
                         "count": int(mask.sum()), "n": len(contra)})
        rows.append({**base, "section": "summary top group ≠ reader's eff_B favoured", "measure": "reader changed at C",
                     "count": int((contra.changed_C == "yes").sum()), "n": len(contra)})
        rows.append({**base, "section": "summary top group ≠ reader's eff_B favoured", "measure": "reader's C favoured = summary top",
                     "count": int((contra.favoured_C == contra.llm_top).sum()), "n": len(contra)})
        if label != "pooled" and sl == "all":
            for _, x in contra.iterrows():
                who = ("summary" if x.llm_top_correct and not x.fav_correct_B else
                       "reader" if x.fav_correct_B and not x.llm_top_correct else "neither")
                crows.append({**base, "case_id": x.case_id, "truth": x.reference, "favoured_B": x.favoured_B,
                              "llm_top": x.llm_top, "who_was_right_at_B": who, "changed_C": x.changed_C,
                              "favoured_C": x.favoured_C, "fav_correct_C": x.fav_correct_C, "use_C": x.use_C,
                              "engaged_B": x.engaged_B})
    T["T7_llm_summary_5_9"] = pd.DataFrame(rows).reindex(
        columns=["reader_id", "expertise_tier", "engaged_B_subset", "section", "measure", "count", "n", "value", "note"])
    T["T7b_summary_contradicts_reader"] = pd.DataFrame(crows, columns=[
        "reader_id", "expertise_tier", "case_id", "truth", "favoured_B", "llm_top", "who_was_right_at_B",
        "changed_C", "favoured_C", "fav_correct_C", "use_C", "engaged_B"])

    # T8 lists
    top10 = {sid: top10_labels(data, sid) for sid in r.slide_id.unique()}

    def case_list(df, transition=None):
        out = []
        for _, x in df.iterrows():
            out.append({"reader_id": x.reader_id, "expertise_tier": x.expertise_tier, "case_id": x.case_id,
                        **({"transition": transition} if transition else {}),
                        "cohort": x.cohort, "truth": x.reference, "favoured_A": x.favoured_A, "favoured_B": x.favoured_B,
                        "favoured_C": x.favoured_C, "dx_A": x.dx_A, "dx_B": x.dx_B, "dx_C": x.dx_C,
                        "retrieval_top10_labels": top10[x.slide_id], "hit5": x.hit5, "top1_correct": x.top1_correct,
                        "majority10_correct": x.majority10_correct, "use_B": x.use_B, "use_C": x.use_C,
                        "engaged_B": x.engaged_B, "comment_A": x.comments_A, "comment_B": x.comments_B, "comment_C": x.comments_C})
        return out
    ml = []
    for tr, col in (("A→B", "class_AB"), ("B→C", "class_BC"), ("A→C", "class_AC")):
        ml += case_list(r[r[col] == "mislead"], tr)
    cols = ["reader_id", "expertise_tier", "case_id", "transition", "cohort", "truth", "favoured_A", "favoured_B",
            "favoured_C", "dx_A", "dx_B", "dx_C", "retrieval_top10_labels", "hit5", "top1_correct",
            "majority10_correct", "use_B", "use_C", "engaged_B", "comment_A", "comment_B", "comment_C"]
    T["T8a_mislead_list"] = pd.DataFrame(ml, columns=cols)
    T["T8b_rescue_list"] = pd.DataFrame(case_list(r[(r.class_AB == "correct") & ~r.hit5]),
                                        columns=[c for c in cols if c != "transition"])
    T["T8c_misread_list"] = pd.DataFrame(case_list(r[(r.class_AB == "mislead") & r.hit5]),
                                         columns=[c for c in cols if c != "transition"])
    hc = r[r.harmful_conf_B]
    T["T8d_harmful_confidence_list"] = pd.DataFrame(
        [{"reader_id": x.reader_id, "expertise_tier": x.expertise_tier, "case_id": x.case_id, "cohort": x.cohort,
          "truth": x.reference, "favoured_A": x.favoured_A, "favoured_B": x.favoured_B, "conf_A": x.conf_A,
          "conf_B": x.conf_B, "fav_correct_A": x.fav_correct_A, "class_AB": x.class_AB, "hit5": x.hit5,
          "use_B": x.use_B, "retrieval_top10_labels": top10[x.slide_id], "comment_B": x.comments_B}
         for _, x in hc.iterrows()],
        columns=["reader_id", "expertise_tier", "case_id", "cohort", "truth", "favoured_A", "favoured_B", "conf_A",
                 "conf_B", "fav_correct_A", "class_AB", "hit5", "use_B", "retrieval_top10_labels", "comment_B"])

    # S3 / S4 vocabulary (§5.8)
    rows = []
    for label, tier, sl, g in engagement_sets(sets):
        for p in "ABC":
            n = len(g)
            rows.append({"reader_id": label, "expertise_tier": tier, "engaged_B_subset": sl, "step": p, "n": n,
                         "readings_with_MISC": int(g[f"misc_in_{p}"].sum()),
                         "MISC_favoured": int((g[f"favoured_{p}"] == d.MISC).sum()),
                         "readings_with_NORMAL": int(g[f"normal_in_{p}"].sum()),
                         "NORMAL_favoured": int((g[f"favoured_{p}"] == "NORMAL").sum()),
                         "fav_correct_strict_k": int(g[f"fav_correct_{p}"].sum()),
                         "fav_correct_comment_informed_k": int(g[f"fav_correct_{p}_cmap"].sum()),
                         "in_diff_strict_k": int(g[f"in_diff_{p}"].sum()),
                         "in_diff_comment_informed_k": int(g[f"in_diff_{p}_cmap"].sum()),
                         "fav_correct_strict_p": g[f"fav_correct_{p}"].mean(),
                         "fav_correct_comment_informed_p": g[f"fav_correct_{p}_cmap"].mean(),
                         "in_diff_strict_p": g[f"in_diff_{p}"].mean(),
                         "in_diff_comment_informed_p": g[f"in_diff_{p}_cmap"].mean()})
    T["S3_vocabulary_5_8"] = pd.DataFrame(rows)
    rows = []
    for _, x in r.iterrows():
        for p in "ABC":
            note = x[f"cmap_note_{p}"]
            if not note or note == "misc_no_named_entity":
                continue
            status, _, rest = note.partition(":")
            entity, _, group = rest.partition("->")
            rows.append({"reader_id": x.reader_id, "expertise_tier": x.expertise_tier, "case_id": x.case_id,
                         "step": p, "entry_written_in_step": x[f"eff_source_{p}"],
                         "status": status.replace("misc_", ""), "entity": entity,
                         "mapped_group": group or ("outside_vocabulary" if "outside" in status else ""),
                         "MISC_was_favoured": x[f"favoured_{p}"] == d.MISC})
    T["S4_entity_map_applications_5_8"] = pd.DataFrame(rows, columns=[
        "reader_id", "expertise_tier", "case_id", "step", "entry_written_in_step", "status", "entity",
        "mapped_group", "MISC_was_favoured"])
    emap = d.load_entity_map(entity_map_path)
    rows = []
    for _, x in r.iterrows():
        for p in "ABC":
            for h in d.comment_entities(x[f"comments_{p}"], emap):
                rows.append({"reader_id": x.reader_id, "expertise_tier": x.expertise_tier, "case_id": x.case_id,
                             "comment_step": p, "entity": h.normalised_entity, "map_outcome": h.mapped_group,
                             "truth": x.reference, "favoured_in_step": x[f"favoured_{p}"],
                             "MISC_in_step": x[f"misc_in_{p}"]})
    T["S4b_comment_named_entities_5_8"] = pd.DataFrame(rows, columns=[
        "reader_id", "expertise_tier", "case_id", "comment_step", "entity", "map_outcome", "truth",
        "favoured_in_step", "MISC_in_step"])
    rows = []
    for _, x in r[r.reference.isin(SUBTYPE_GROUPS)].iterrows():
        for p in "ABC":
            com = x[f"comments_{p}"]
            if not com.strip():
                continue
            rows.append({"reader_id": x.reader_id, "expertise_tier": x.expertise_tier, "case_id": x.case_id,
                         "cohort": x.cohort, "truth": x.reference, "stv_subclass": x.stv_subclass,
                         "ccdi_diagnosis": x.ccdi_diagnosis, "comment_step": p,
                         "plan_subtype_terms_found": "; ".join(sorted({m.lower() for m in SUBTYPE_TERMS_RE.findall(com)})),
                         "comment": com})
    T["S6_subtype_mentions_for_coding_5_8"] = pd.DataFrame(rows, columns=[
        "reader_id", "expertise_tier", "case_id", "cohort", "truth", "stv_subclass", "ccdi_diagnosis",
        "comment_step", "plan_subtype_terms_found", "comment"])

    # T9 coding scaffold (§5.10)
    rows = []
    for _, x in r.iterrows():
        for p in "ABC":
            rows.append({"reader_id": x.reader_id, "expertise_tier": x.expertise_tier, "case_id": x.case_id, "phase": p,
                         "truth_group": x.reference, "eff_favoured": x[f"favoured_{p}"],
                         "outcome_class": {"A": "", "B": x.class_AB, "C": x.class_BC}[p],
                         "outcome_class_transition": {"A": "", "B": "A→B", "C": "B→C"}[p],
                         "comment": x[f"comments_{p}"], **{c: "" for c in CODES}})
    T["T9_comments_for_coding"] = pd.DataFrame(rows)

    # T10 between-reader, paired set (§5.7)
    if len(readers) >= 2:
        T["T10_between_reader_5_7"], T["T10b_paired_cases_5_7"] = paired.between_reader_tables(
            r, readers, d.EXPERTISE_TIER)
    return T


# ---------------------------------------------------------------- figures

def save(fig, figdir: Path, name: str) -> List[Path]:
    paths = []
    for ext in ("png", "pdf"):
        p = figdir / f"{name}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight", metadata=None if ext == "png" else {"CreationDate": None})
        paths.append(p)
    plt.close(fig)
    return paths


def figures(r: pd.DataFrame, readers: List[str], figdir: Path) -> List[Path]:
    out = []
    for rid in readers:
        g = r[r.reader_id == rid].sort_values(["cohort", "reference", "case_id"]).reset_index(drop=True)
        n = len(g)
        tier = d.EXPERTISE_TIER[rid]

        # F1 trajectory
        fig, ax = plt.subplots(figsize=(4.8, 0.16 * n + 1.4))
        ys = np.arange(n)[::-1]
        for y, (_, x) in zip(ys, g.iterrows()):
            ax.plot([1, 2, 3], [y, y, y], color="0.8", lw=0.8, zorder=1)
            for xi, p in zip((1, 2, 3), "ABC"):
                if x[f"fav_correct_{p}"]:
                    ax.scatter(xi, y, s=26, color="#1b7837", zorder=3)
                elif x[f"in_diff_{p}"]:
                    ax.scatter(xi, y, s=26, color="#fdb863", edgecolor="#b35806", zorder=3)
                else:
                    ax.scatter(xi, y, s=26, facecolor="white", edgecolor="0.4", zorder=3)
            ax.scatter(0.2, y, marker="s", s=18, facecolor="black" if x.hit5 else "white", edgecolor="black")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{c} {t}" for c, t in zip(g.case_id, g.reference)], fontsize=5)
        ax.set_xticks([0.2, 1, 2, 3])
        ax.set_xticklabels(["hit@5", "Step 1\n(A)", "Step 2\n(B)", "Step 3\n(C)"], fontsize=7)
        ax.set_xlim(-0.3, 3.5)
        ax.set_ylim(-1, n)
        cut = (g.cohort != g.cohort.shift()).to_numpy().nonzero()[0]
        for c in cut[1:]:
            ax.axhline(ys[c] + 0.5, color="0.3", lw=0.6, ls="--")
        for c0, c1 in zip(cut, list(cut[1:]) + [n]):
            ax.text(3.45, (ys[c0] + ys[c1 - 1]) / 2, g.cohort[c0], rotation=90, va="center", fontsize=7)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker="o", ls="", color="#1b7837", label="favoured = truth"),
                   Line2D([], [], marker="o", ls="", markerfacecolor="#fdb863", markeredgecolor="#b35806",
                          label="truth in differential, not favoured"),
                   Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="0.4", label="truth not in differential"),
                   Line2D([], [], marker="s", ls="", markerfacecolor="black", markeredgecolor="black", label="hit@5 (filled = yes)")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.6 / (0.16 * n)), fontsize=6, ncol=2, frameon=False)
        ax.set_title(f"F1 favoured-correct trajectory, {rid} ({tier}), n = {n}", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        out += save(fig, figdir, f"F1_trajectory_{rid}")

        # F2 usefulness by baseline correctness
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
        for ax, use, ttl in zip(axes, ("use_B", "use_C"), ("Step 2 search usefulness (use_B)", "Step 3 summary usefulness (use_C)")):
            for off, fa, col in ((-0.2, True, "#1b7837"), (0.2, False, "#b35806")):
                sub = g[g.fav_correct_A == fa][use]
                counts = [int((sub == lv).sum()) for lv in ep.LIKERT_LEVELS]
                ax.bar(np.array(ep.LIKERT_LEVELS) + off, counts, width=0.4, color=col,
                       label=f"Step 1 favoured {'correct' if fa else 'not correct'} (n = {len(sub)})")
                for lv, c in zip(ep.LIKERT_LEVELS, counts):
                    ax.text(lv + off, c + 0.3, str(c), ha="center", fontsize=6)
            ax.set_xticks(ep.LIKERT_LEVELS)
            ax.set_xticklabels(["1\nmisleading", "2\nnot useful", "3\nmarginal", "4\nuseful", "5\nvery useful"], fontsize=7)
            ax.set_title(ttl, fontsize=8)
        axes[0].set_ylabel("readings")
        axes[0].legend(fontsize=6, frameon=False)
        fig.suptitle(f"F2 usefulness by baseline correctness, {rid} ({tier}); Step 3 follows Step 2", fontsize=8)
        out += save(fig, figdir, f"F2_usefulness_{rid}")

        # F3 confidence trajectories by baseline correctness (distribution + median)
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
        cmap = plt.get_cmap("Blues")
        for ax, fa in zip(axes, (True, False)):
            sub = g[g.fav_correct_A == fa]
            bottoms = np.zeros(3)
            for lv in ep.LIKERT_LEVELS:
                fr = np.array([(sub[f"conf_{p}"] == lv).mean() if len(sub) else 0 for p in "ABC"])
                ax.bar([0, 1, 2], fr, bottom=bottoms, color=cmap(0.15 + 0.17 * lv), edgecolor="white", width=0.6,
                       label=str(lv))
                bottoms += fr
            meds = [sub[f"conf_{p}"].median() for p in "ABC"]
            ax2 = ax.twinx()
            ax2.plot([0, 1, 2], meds, color="black", marker="D", ms=4, lw=1)
            ax2.set_ylim(0.5, 5.5)
            ax2.set_yticks([1, 2, 3, 4, 5])
            ax2.tick_params(labelsize=6)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["Step 1", "Step 2", "Step 3"], fontsize=7)
            ax.set_title(f"Step 1 favoured {'correct' if fa else 'not correct'} (n = {len(sub)})", fontsize=8)
        axes[0].set_ylabel("fraction of readings", fontsize=7)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="confidence", fontsize=6, title_fontsize=6, frameon=False,
                   loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))
        fig.suptitle(f"F3 confidence by step (bars: distribution; ◆ median, right axis), {rid} ({tier})", fontsize=8)
        fig.tight_layout()
        out += save(fig, figdir, f"F3_confidence_{rid}")

        # F4 retrieval success vs outcome class
        fig, ax = plt.subplots(figsize=(7, 3))
        xs = np.arange(len(ep.CLASS_ORDER))
        for off, col, nm, colr in ((-0.25, "hit5", "hit@5", "#2166ac"), (0, "majority10_correct", "plurality of 10", "#67a9cf"),
                                   (0.25, "top1_correct", "top-1", "#d1e5f0")):
            vals = [g[g.class_AB == c][col].mean() if (g.class_AB == c).any() else 0 for c in ep.CLASS_ORDER]
            ax.bar(xs + off, vals, width=0.25, color=colr, edgecolor="0.3", label=nm)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{c}\n(n = {int((g.class_AB == c).sum())})" for c in ep.CLASS_ORDER], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fraction of readings with retrieval success")
        ax.legend(fontsize=6, frameon=False)
        ax.set_title(f"F4 slide-level retrieval success by Step 1→2 outcome class, {rid} ({tier})", fontsize=8)
        out += save(fig, figdir, f"F4_retrieval_vs_class_{rid}")
    return out


# ---------------------------------------------------------------- captions and markdown

def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-9 else str(int(round(v)))
    return str(v).replace("|", "\\|").replace("\n", " ")


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_\n"
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def captions(r: pd.DataFrame, readers: List[str], T: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    times = "; ".join(
        f"{rid}: median sec_A {r[r.reader_id == rid].sec_A.median():.0f} s, sec_B {r[r.reader_id == rid].sec_B.median():.0f} s, "
        f"sec_C {r[r.reader_id == rid].sec_C.median():.0f} s" for rid in readers)
    phase_c = (f"Phase C caveats: C follows B; the summary's effect is not isolated. Per-reader median reading time per "
               f"step (reading time within the instrument, not time-to-diagnosis): {times}.")
    lib = T["_library_groups"]
    tert = "; ".join(f"{t}: " + ", ".join(f"{g} ({n})" for g, n in zip(sub.group, sub.library_slides_in_index))
                     for t, sub in lib.groupby("tertile", sort=False))
    c = {
        "T1_reader_case_flow": f"T1. Reader and case flow (analysed readers only). Pilots case01/case02 excluded; QC gate dropped (003 §A: the key was not independently vetted). Readers excluded by name: {', '.join(sorted(d.EXCLUDED_READERS))}, demo-*. Readers with an expertise tier not analysed in this run (held): {', '.join(sorted(set(d.EXPERTISE_TIER) - set(readers))) or 'none'}. {TIER_NOTE}",
        "T2_primary_endpoints": f"T2. Primary endpoints (002 §4 / 001 §4) per reader. 95% intervals: whole-case bootstrap percentile (resample cases with replacement, 5,000 draws, seed 20261027; paired difference formed within each draw) with Wilson score intervals as a cross-check for single proportions. Discordant pairs are counts; no hypothesis test is performed. Endpoint 3 (outcome classes) is in T3. Usefulness 1 = misleading … 5 = very useful, reported as counts and median. {TIER_NOTE}",
        "T3_outcome_classes": f"T3. Outcome classes per transition (002 §3.3, rules evaluated in order); `confirm` split into confirm_favoured / confirm_differential_only (003 §B1; only the first supports the favoured diagnosis). A→C 'changed' = changed at B or at C (003 Amendment 001 item 7). A→B is primary. {phase_c} {TIER_NOTE}",
        "S1_trajectory_5_1": f"S1. Trajectory through the three steps (002 §5.1): favoured-correct and truth-in-differential per step with bootstrap and Wilson intervals; paired differences B−A, C−B, C−A (bootstrap); truth-rank and differential-size distributions per step. {phase_c} {TIER_NOTE}",
        "S2_confidence_5_2": f"S2. Confidence per step (1 = very low … 5 = very high) as counts, median, quartiles and mean beside the counts; Δconf A→B (dconf_BA) and B→C (dconf_CB) split by Step 1 favoured-correct; harmful-confidence count (conf_B > conf_A and Step 2 favoured not correct; readings listed in T8d). {phase_c} {TIER_NOTE}",
        "T4_strata_5_3": f"T4. Retrieval dependence (002 §5.3): endpoints 1–3 and use_B by slide-level hit@5 (primary stratifier), plurality-of-10 correct (single most frequent label = truth; tie = false; 003 §B3), top-1 correct, cohort, difficulty tier as stored per cohort, and library-support tertile of the true group. Difficulty tier correction (003 Amendment 001 item 4): 002 §3.7 said all STV cases are not_on_ali_list; the export carries Tier1 labels on 20 STV cases and not_on_ali_list on 20 CCDI cases, so the tier stratum is shown per cohort as stored. Library support (Amendment 001 item 5): the retrieval library is {RETRIEVAL_LIBRARY_N:,} slides; per-group support uses the 2,553-row slides_index (the library slides referenced by any of the 82 manifests), tertiles over the 14 groups at the 1/3 and 2/3 quantiles: {tert}. Descriptive counts; no intervals; {len(T['T4_strata_5_3'].stratifier.unique())} stratifiers. {TIER_NOTE}",
        "T5a_engagement_5_4": f"T5a. {POST_HOC} Engagement during Step 2 (002 §3.6/§5.4): events with auto:true excluded; Step 2 window t_open_B…t_submit_B plus the A→B gap; engaged_B = at least one reader-opened ROI hit, slide neighbour or ROI selection. engaged_C_retrieval = any reader-initiated retrieval event during Step 3 (003 §B2). Interpretation rule: a reading that is not engaged_B is a list-only condition — the reader evaluated the ranked label list, thumbnails and distances, not the image-retrieval interface; results must say which condition each reader's numbers describe. {TIER_NOTE}",
        "T5b_by_engagement_5_4": f"T5b. {POST_HOC} Endpoints 1–4 and Step 2 reading time by engaged_B. Interpretation rule as in T5a (engaged_B = False is the list-only condition). Strata with few readings are descriptive only. {TIER_NOTE}",
        "S5_usefulness_vs_need_5_5": f"S5. {POST_HOC} Usefulness versus need (002 §5.5): use_B by Step 1 favoured-correct and by A→B class; use_C by Step 1 favoured-correct and by B→C class; counts and median. {phase_c} {TIER_NOTE}",
        "T6_reading_time_5_6": f"T6. {POST_HOC} Reading time within the instrument (t_submit − t_open per step; never time-to-diagnosis), median and quartiles in seconds: all readings, sensitivity excluding interrupted readings (> 1800 s within a step or between steps), by cohort, by engaged_B, by session (gap > 30 min between consecutive submissions) and first vs last chronological quartile. Case order was fixed, not counterbalanced. {phase_c} {TIER_NOTE}",
        "T6b_interrupted_readings": f"T6b. {POST_HOC} Readings flagged interrupted (> 1800 s within a step) or interrupted_between_phases (> 1800 s between consecutive steps, 003 §B5). {TIER_NOTE}",
        "T7_llm_summary_5_9": f"T7. Step 3 LLM summary, descriptive (002 §5.9, 003 §A option (i)). Summary properties are per case read. llm_flags_discrepancy is the literal 002 §3.8 rule (text mentions 'discrepant', 'split' or 'uncertain'), reported with the count that matches only through a negation beside it (003 Amendment 001 item 6); it is not a measure of whether the summary called retrieval unreliable. llm_wording_probability: the summary presents the scores as probabilities. {phase_c} {TIER_NOTE}",
        "T7b_summary_contradicts_reader": f"T7b. Readings where the summary's top-scored group differs from the reader's effective Step 2 favoured diagnosis, who was right at Step 2, and what the reader did at Step 3. {phase_c} {TIER_NOTE}",
        "T8a_mislead_list": f"T8a. Safety table: every mislead reading (002 §3.3) for each transition, with Step 1/2/3 favoured and differentials, truth, slide-level top-10 retrieval labels (rank order), retrieval success, usefulness and verbatim comments. {phase_c} {TIER_NOTE}",
        "T8b_rescue_list": f"T8b. Rescue list (002 §5.3): A→B class correct when slide-level retrieval was wrong (hit@5 = False, the primary stratifier). {TIER_NOTE}",
        "T8c_misread_list": f"T8c. Misread list (002 §5.3): A→B class mislead when slide-level retrieval was right (hit@5 = True). {TIER_NOTE}",
        "T8d_harmful_confidence_list": f"T8d. Harmful-confidence readings (002 §3.4): confidence rose from Step 1 to Step 2 while the Step 2 favoured diagnosis was not the truth. {TIER_NOTE}",
        "S3_vocabulary_5_8": f"S3. Vocabulary (002 §5.8): readings whose effective differential contains MISC (other) or NORMAL, per step; strict (primary) favoured-correct and truth-in-differential beside the comment-informed sensitivity (003 §D entity map; applied to the comment of the step in which the MISC entry was written; case062 ruled not mapped, Amendment 001 item 2). The sensitivity never replaces the strict result. {phase_c} {TIER_NOTE}",
        "S4_entity_map_applications_5_8": f"S4. Every MISC (other) entry whose source comment names a map entity: re-mapped entries (status mapped) and entities outside the library vocabulary (status outside_vocabulary), per step in which the effective answer contains the entry. {phase_c} {TIER_NOTE}",
        "S4b_comment_named_entities_5_8": f"S4b. Every comment (any step, whether or not a MISC entry was written) that names an entity of the 003 §D entity map, with the map outcome (library group or outside_vocabulary). Entities not in the map are not detected; uncoded comments are in T9. {TIER_NOTE}",
        "S6_subtype_mentions_for_coding_5_8": f"S6. Subtype mentions (002 §5.8), listing for coding: every non-empty comment on readings whose truth group has subtypes named in the plan (RMS, SS, LIVER), with stv_subclass / ccdi_diagnosis beside it; `plan_subtype_terms_found` marks the literal plan terms (alveolar, embryonal, hepatoblastoma) only. Agreement with the recorded subclass is not scored here (pending lead coding). {TIER_NOTE}",
        "T9_comments_for_coding": f"T9. Comment-coding scaffold (002 §5.10): one row per reader × case × step with the verbatim comment and empty columns for the nine pre-specified codes. Coding is done by the lead and checked by one pathologist; pending. Outcome class: A→B on Step 2 rows, B→C on Step 3 rows. {phase_c} {TIER_NOTE}",
    }
    split = (" Following 002 §3.6, rows are given for all readings and split by engaged_B (column engaged_B_subset; "
             "empty subsets omitted; engaged_B = False is the list-only condition).")
    for k in ("S1_trajectory_5_1", "S2_confidence_5_2", "T4_strata_5_3", "S5_usefulness_vs_need_5_5",
              "T7_llm_summary_5_9", "S3_vocabulary_5_8"):
        c[k] += split
    c["T5a_engagement_5_4"] += " n_llm_toggled counts non-auto Step 3 summary-panel toggles (002 §3.6)."
    if "T10_between_reader_5_7" in T:
        ns = "; ".join(f"{a} vs {b}: n = {n}" for (a, b), n in T["T10_between_reader_5_7"].groupby(
            ["reader1_id", "reader2_id"], sort=False).paired_set_n.first().items())
        c["T10_between_reader_5_7"] = (
            f"T10. Between-reader agreement on the paired set (002 §5.7); descriptive; paired set n = {ns} "
            "(cases with A and B submitted by both readers); no test. Per step: raw agreement of the effective "
            "favoured diagnosis, unweighted Cohen's kappa over the 16 closed-list labels (NaN with a note when "
            "only one label is used), and favoured-correct counts for both / only reader 1 / only reader 2 / "
            "neither. A case enters a step only when both readers have an effective answer at that step. "
            f"{phase_c}")
        c["T10b_paired_cases_5_7"] = (
            f"T10b. Per-case table of the paired set (002 §5.7); descriptive; paired set n = {ns}; no test. "
            "Effective favoured diagnosis and favoured-correct at Steps 1–3 and the A→B class for each reader, "
            f"with truth and slide-level hit@5. {phase_c}")
    return c


FIG_CAPTIONS = {
    "F1": "F1. Per-case favoured-correct trajectory through Steps 1–3 for each reader (cases ordered by cohort, then truth group, then case id); filled square = slide-level hit@5. Step 3 follows Step 2; the summary's effect is not isolated.",
    "F2": "F2. Search (use_B) and summary (use_C) usefulness counts by Step 1 favoured-correct. [post-hoc-motivated] (002 §5.5). Step 3 follows Step 2; the summary's effect is not isolated.",
    "F3": "F3. Confidence distribution per step with the median, split by Step 1 favoured-correct (002 §5.2). Step 3 follows Step 2; the summary's effect is not isolated.",
    "F4": "F4. Fraction of readings with slide-level retrieval success (hit@5, plurality of 10, top-1) by A→B outcome class (002 §5.3).",
}


# ---------------------------------------------------------------- main

def git_state() -> Dict[str, str]:
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--", "scripts/reader_study")
    return {"commit": run("rev-parse", "HEAD"), "clean": "yes" if not dirty else f"NO: {dirty}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path,
                    default=OUT_ROOT / "reader_study_suite" / "analysis" / f"20261027_{EXPORT_STAMP}")
    ap.add_argument("--export", type=Path, default=OUT_ROOT / "reader_study_suite" / "exports" / EXPORT_STAMP)
    ap.add_argument("--data", type=Path, default=OUT_ROOT / "reader_study_suite" / "data")
    ap.add_argument("--entity-map", type=Path, default=ENTITY_MAP)
    ap.add_argument("--readers", required=True)
    args = ap.parse_args(argv)
    readers = [x.strip() for x in args.readers.split(",") if x.strip()]
    for rid in readers:
        if d.is_excluded_reader(rid) or rid not in d.EXPERTISE_TIER:
            raise SystemExit(f"{rid} is excluded or has no expertise tier")
    started = dt.datetime.now(CDT)
    git = git_state()

    rec = (args.analysis / "RUN_RECORD.generated.md").read_text()
    m = re.search(r"readings\.csv: ([0-9a-f]{64})", rec)
    actual = sha256_file(args.analysis / "readings.csv")
    if not m or m.group(1) != actual:
        raise SystemExit("readings.csv does not match the sha256 recorded by build_readings.py")

    r = load_readings(args.analysis, readers)
    ev = pd.read_csv(args.analysis / "events_engagement.csv", usecols=["reader_id", "case_id", "kind", "auto"])
    ev = ev[ev.reader_id.isin(readers)].copy()
    ev["auto"] = to_bool(ev.auto)
    T = build_tables(r, readers, args.export, args.data, ev, args.entity_map)
    tdir, fdir = args.analysis / "tables", args.analysis / "figures"
    tdir.mkdir(exist_ok=True)
    fdir.mkdir(exist_ok=True)
    caps = captions(r, readers, T)
    written: List[Path] = []
    for name, df in T.items():
        if name.startswith("_"):
            continue
        p = tdir / f"{name}.csv"
        df.to_csv(p, index=False)
        (tdir / f"{name}.md").write_text(md_table(df))
        written += [p, tdir / f"{name}.md"]
    T["_library_groups"].to_csv(tdir / "library_groups_tertiles.csv", index=False)
    written.append(tdir / "library_groups_tertiles.csv")
    cap_lines = ["# Table and figure captions", ""]
    for name in T:
        if name in caps:
            cap_lines += [f"## {name}", "", caps[name], ""]
    cap_lines += ["## Figures", ""] + [f"- {v}" for v in FIG_CAPTIONS.values()] + [""]
    (tdir / "CAPTIONS.md").write_text("\n".join(cap_lines))
    written.append(tdir / "CAPTIONS.md")
    written += figures(r, readers, fdir)

    finished = dt.datetime.now(CDT)
    meta = {"started": started.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"),
            "commit": git["commit"], "scripts_reader_study_clean": git["clean"], "readers": readers,
            "tiers": [d.EXPERTISE_TIER[x] for x in readers], "python": platform.python_version(),
            "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "seed": ep.SEED, "n_draws": ep.N_DRAWS, "readings_sha256": actual,
            "entity_map_sha256": sha256_file(args.entity_map),
            "command": " ".join([Path(sys.argv[0]).name] + (argv if argv is not None else sys.argv[1:]))}
    (args.analysis / "analyze_run.json").write_text(json.dumps(meta, indent=1) + "\n")
    skip = {"RUN_RECORD.md", "OUTPUTS_SHA256"}
    files = sorted(p for p in args.analysis.rglob("*") if p.is_file() and p.name not in skip)
    (args.analysis / "OUTPUTS_SHA256").write_text(
        "".join(f"{sha256_file(p)}  {p.relative_to(args.analysis)}\n" for p in files))
    print(f"wrote {len(written)} table/figure files under {args.analysis}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
