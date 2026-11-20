"""Fig. 8 (new main-text figure, v2): blinded before/after pathologist reader study.

Design agreed 2026-10-27 ≈16:15 CDT (candidate 003 Amendment 004;
spec `[internal record]`). Six panels, readers
as rows of small multiples labelled by expertise tier, per-reader estimates
only (no pooled row):

  A  study-design schematic (three steps per case, what is recorded, blinding,
     case set, readers by tier); abstract boxes, no patient images
  B  favored diagnosis = true group at Step 1 → 2 → 3 per reader, whole-case
     bootstrap 95 % intervals; truth-in-differential as a lighter marker;
     discordant counts (wrong→right / right→wrong) printed
  C  Step 1→2 outcome classes per reader (plan §3.3); the right→wrong class is
     labelled with its count even at zero (display labels avoid the word
     "mislead" 2026-10-28; internal class names unchanged)
  D  the same for Step 2→3 (after the LLM summary; follows Step 2, effect not
     isolated) — same layout and key as C (2026-10-28: no retrieval split
     in the main figure)
  S_hit10  supplementary (S-RS2): Step 1→2 outcome per reader by whether the true
     group was among the 10 slides shown (hit@10 from readings.csv; T4
     pre-specified hit@5). Not in the default render.
  E  usefulness ratings 1–5 (diverging), search results (Step 2) and summary
     (Step 3), per reader, median marked
  F  how the tool was used, per reader: share of readings with any retrieved
     image opened; ROI selections per case; median seconds per step with IQR
     ("reading time within the instrument")
  G  change in confidence (1–5) after the search results and after the LLM
     summary, split by whether the favored diagnosis after that stage was the
     true group ("more confident, favored wrong" = harmful confidence, plan 002
     §3.4); medians before → after printed (standalone candidate 2026-10-28)

Composite (2026-11-06, revision): main Fig. 8 = four panels A, B, C, E (B `rows`, E `v3`):
A|B on the top row, C|E on the bottom row, each with its own key underneath (`--compose`); face
letters A, B, C, D (working panel E is printed as D: usefulness is Fig. 8D).
Panels D, F, G move to one supplementary figure, provisional Supplementary Fig. S14 (workspace
ID S-RS-F1; `--compose-supp`): D|F on top, G across the full width below, lettered A, B, C on its
face (D→A, F→B, G→C). The seven-panel A–G layout of 2026-11-02 is still available with
`--compose-panels A,B,C,D,E,F,G` (archived render in v5_20261103_seven_panels_11-02_export/).

Stage wording on the image is "slide only / + search results / + LLM summary"
and transitions read "after seeing the search results" / "after reading the
LLM summary"; "Step 1/2/3" never appears on the figure (2026-10-28).

Sources (accepted, read-only; nothing recomputed here) — the five-reader run of the
pipeline (scripts/reader_study/analyze.py at HEAD c30e238; build_readings.py with the uncommitted
revision population-check fix; final run at HEAD 3e8b5c3) on export prod_export_20261116-085036, the
"reading stopped" export (readers <reader-login>, <reader-login>, <reader-login>, <reader-login>, <reader-login>;
80 readings each = 400; tiers expert, expert, general, general, general — Reader 5's tier assigned by
the author 2026-11-08, plan 003 amendment 009; final run registered in plan 003 amendment 018):
  $PROSPER_OUT/reader_study_suite/analysis/20261116_prod_export_20261116-085036_five_readers/
    readings.csv (sha256 asserted equal to analyze_run.json `readings_sha256`),
    analyze_run.json and every table below (sha256 asserted equal to the run's OUTPUTS_SHA256),
    tables/{T2_primary_endpoints,T3_outcome_classes,T4_strata_5_3,S1_trajectory_5_1,
            T6_reading_time_5_6,S2_confidence_5_2}.csv  — every drawn value is asserted equal to them.
(Earlier runs: 20261114_prod_export_20261114-095616_five_readers (80/80/73/6/80) — archived in
v9_20261114_five_readers_1114_export/; 20261027_prod_export_20261027-101111_two_readers, 20261102_prod_export_20261102-090640,
20261106_prod_export_20261106-090432 — its render archived in v6_20261106_four_readers_1106_export/;
20261108_prod_export_20261108-114240, the four-reader run of that export — its render archived in
v7_20261108_four_readers_1108_export/; 20261108_prod_export_20261108-114240_five_readers (80/80/45/6/15)
— archived in v8_20261108_five_readers_1108_export/.)
revision (five readers): E and G grow by two 0.1875 in rows per reader beyond four (paired_extra_in), the
composites and the standalone E/G slots grow by the same height; the paired-difference axis labels only
multiples of 0.4 when it carries more than four ticks.
revision (display labels only; no value, panel or tier code changed): tier "general" is shown as
"trainee-level pathologist(s)"; panel A says "same 80 cases for all readers" (readers could choose the order),
"+ age, site, presentation" and "differential ≤ 6"; the red end-of-bar label of C / S14A reads "away / dropped";
S14B's first heading reads "engaged readings" (engaged_B includes ROI selections). `--out-root` writes elsewhere.
revision (display labels only): panel A says "shared 80-case list for all readers (… STV)"; the trajectory
axes read "proportion of readings".

Outputs  $PROSPER_OUT/figures_20261021/fig8_reader_study/
         panels/<letter>_<variant>/panel_<letter>_<variant>.{png,pdf} + source_data.csv,
             legibility_check_8<letter>_<variant>.json, SHA256SUMS.txt, PROVENANCE.md
             (default run; one directory per panel candidate, drawn at the size the
             panel will have in the composite so fonts and spacing are final)
         Figure_8_reader_study.{png,pdf}, source_data_panel{B,C,D}.csv (D = working panel E, printed D),
             legibility_check_8.json                                  (with --compose)
         Figure_S14_reader_study_secondary.{png,pdf}, source_data_S14_panel{D,F,G}.csv
             (file names keep the Fig. 8 working letters; face letters A, B, C),
             legibility_check_S14.json                                (with --compose-supp)
         SHA256SUMS.txt, PROVENANCE.md — one pair for the directory, covering every composite
             written in the run; run --compose and --compose-supp together (the script refuses
             to rewrite them for one composite while the other's outputs sit beside it).
(Archived composites: v1_20261027_reader1_only/, v2_20261027_two_readers/,
v3_20261102_two_readers_7panel/, v4_20261102_four_readers_pre_wording/,
v5_20261103_seven_panels_11-02_export/ (the 11-02 seven-panel render plus the panels/ of 10-28).)

Workflow since 2026-10-28: iterate panels individually, compose once settled.
Panel-B variants   v2           per-reader horizontal offset (archived v2 design)
                   shapes       one marker shape per reader, same colour code; a small
                                offset (--dodge, default 0.05) only keeps the intervals apart
                   shapes0      shapes with no offset at all (intervals overlap)
                   shapes_diff  shapes, no offset, no interval bars on the trajectories;
                                right-hand axis = paired difference Step 2 − Step 1 (95 %
                                interval), same marker per reader
                   rows         readers as rows (as in C–F): Step 1 ○ → Step 2 ● → Step 3 ◆
                                per row, plus the paired-difference axis
                   small        small multiples: one trajectory axis per reader with its
                                own intervals (3 columns for six readers)
Panel-E variants   v2      red–blue Likert, "1 misleading"
                   v3      purple–orange Likert, "1 rated misleading", so red and the word
                           "mislead" are reserved for the objective outcome class in C–D

Reader labels: "Reader k" in --readers order; tiers from the analysis tables.
Later readers / a later export: --analysis <dir> --readers a,b,c.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "reader_config", Path(__file__).resolve().parents[1] / "reader_study" / "reader_config.py")
_reader_config = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_reader_config)
READER_CONFIG = _reader_config.load_reader_config()
from _common import GREY, out_dir, prosper_out, save, style, write_provenance  # noqa: E402
from figS1_S2_neighbor_ladder import legibility, visible_texts  # noqa: E402

FIGURE = "fig8_reader_study"
ANALYSIS_DEFAULT = "reader_study_suite/analysis/20261116_prod_export_20261116-085036_five_readers"
READERS_DEFAULT = ",".join(READER_CONFIG["fig8_reader_order"])  # Reader 1..5 order
TABLES = ("T2_primary_endpoints", "T3_outcome_classes", "T4_strata_5_3", "S1_trajectory_5_1", "T6_reading_time_5_6",
          "S2_confidence_5_2")
WIDTH_IN = 180 / 25.4
# revision ([internal record] 2026-11-14 15:23 CST): stored tier code "general" (plan 003 amendments 006/009)
# is displayed as "trainee-level pathologist"; the codes, row order and readings are unchanged.
TIER_LABEL = {"expert": "experienced pathologist", "general": "trainee-level pathologist", "trainee": "resident"}
TIER_PLURAL = {"expert": "experienced pathologists", "general": "trainee-level pathologists", "trainee": "residents"}
OUT_ROOT: Path | None = None  # --out-root: write composites/panels under <root>/<figure> instead of figures_20261021


def fig_dir(figure: str) -> Path:
    if OUT_ROOT is None:
        return out_dir(figure)
    d = OUT_ROOT / figure
    d.mkdir(parents=True, exist_ok=True)
    return d

COL_FAV = "#1b7837"
COL_DIFF = "#fdb863"
COL_DIFF_EDGE = "#b35806"
CLASS_ORDER = ["mislead", "correct", "expand", "confirm_favoured", "confirm_differential_only", "unhelpful", "other change"]
# Display labels (2026-10-28: do not say "mislead" in the figure). Internal class names (plan 002 §3.3)
# are unchanged in the tables and source data; the two directional classes are labelled symmetrically.
# Figure review 2026-11-03 (D5): the face says "reference label" (the legend's defined term), never "true group" /
# "truth"; stage names use the legend's "+" form; cohort names as in the legend.
CLASS_LABEL = {"mislead": "favored right \u2192 wrong, or ref. label dropped",
               "correct": "favored wrong \u2192 right", "expand": "ref. label added to differential",
               "confirm_favoured": "favored unchanged, correct", "confirm_differential_only": "unchanged, ref. label in differential",
               "unhelpful": "unchanged, ref. label absent", "other change": "other change"}
# printed with its count at the end of every bar, even when 0; it counts the whole red class (favored away from the
# reference label, or reference label dropped), hence "away / dropped" since revision (was "ref. label lost")
ZERO_LABEL = "away / dropped"
CLASS_COL = {"mislead": "#b2182b", "correct": "#1b7837", "expand": "#7fbc41", "confirm_favoured": "#a6dba0",
             "confirm_differential_only": "#e6f5d0", "unhelpful": "#bdbdbd", "other change": "#737373"}
LIKERT = [1, 2, 3, 4, 5]
# v2: red–blue; v3: purple–orange (PuOr) so that red stays reserved for the objective "mislead" class
LIK_COL = {"v2": {1: "#b2182b", 2: "#ef8a62", 3: "#d9d9d9", 4: "#67a9cf", 5: "#2166ac"},
           "v3": {1: "#b35806", 2: "#f1a340", 3: "#d9d9d9", 4: "#998ec3", 5: "#542788"}}
LIK_LAB = {"v2": {1: "1 misleading", 2: "2 not useful", 3: "3 marginal", 4: "4 useful", 5: "5 very useful"},
           "v3": {1: "1 rated misleading", 2: "2 not useful", 3: "3 marginal", 4: "4 useful", 5: "5 very useful"}}
# Stage wording on the figure (2026-10-28: never "Step 1/2/3" on the image; say what the reader saw).
# Internal A/B/C and the plan's Step 1/2/3 are unchanged in code, tables and source data.
STAGE = {"A": "slide only", "B": "+ search results", "C": "+ LLM summary"}
STEP_NAMES = ["slide\nonly", "+ search\nresults", "+ LLM\nsummary"]
AFTER = {"A→B": "after seeing the search results", "B→C": "after reading the LLM summary"}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]  # one per reader, in row order (shapes variant)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ data

def load(analysis: Path, readers: list[str]):
    run = json.loads((analysis / "analyze_run.json").read_text())
    assert sha256(analysis / "readings.csv") == run["readings_sha256"], "readings.csv differs from the analysed file"
    manifest = analysis / "OUTPUTS_SHA256"
    if manifest.exists():  # the pipeline's own output manifest: every input read here must match it
        listed = dict(reversed(ln.split(None, 1)) for ln in manifest.read_text().splitlines() if ln.strip())
        for rel in ["readings.csv", "analyze_run.json"] + [f"tables/{k}.csv" for k in TABLES]:
            assert listed.get(rel) == sha256(analysis / rel), f"{rel} differs from {manifest}"
    r = pd.read_csv(analysis / "readings.csv")
    r = r[r.reader_id.isin(readers)].copy()
    assert set(r.reader_id) == set(readers), f"missing readers: {set(readers) - set(r.reader_id)}"
    assert r.case_id.str.match(r"^case\d{3}$").all()
    T = {k: pd.read_csv(analysis / "tables" / f"{k}.csv") for k in TABLES}
    T = {k: v[v.reader_id.isin(readers)] for k, v in T.items()}  # never the pooled rows
    tiers = {rid: r[r.reader_id == rid].expertise_tier.iloc[0] for rid in readers}
    return r, T, run, tiers


def reader_rows(readers: list[str], tiers: dict, r: pd.DataFrame) -> list[dict]:
    """Row order: by tier (expert, general, trainee), then --readers order; label 'Reader k'."""
    order = {"expert": 0, "general": 1, "trainee": 2}
    rows = []
    for k, rid in enumerate(readers, start=1):
        n = int((r.reader_id == rid).sum())
        rows.append(dict(reader_id=rid, label=f"Reader {k}", tier=tiers[rid], n=n,
                         long=f"Reader {k} ({TIER_LABEL[tiers[rid]]}, n = {n})"))
    rows.sort(key=lambda d: (order[d["tier"]], int(d["label"].split()[1])))
    return rows


# ------------------------------------------------------------------ panel A: schematic

def panel_a(ax, rows: list[dict], r: pd.DataFrame):
    ax.set_xlim(0, 100)
    ax.set_ylim(-10, 100)
    ax.axis("off")
    n_cases = r.case_id.nunique()
    coh = r.drop_duplicates("case_id").cohort.value_counts().to_dict()

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec="0.3", lw=0.6))
        ax.text(x + w / 2, y + h - 3, title, ha="center", va="top", fontsize=7, fontweight="bold")
        ax.text(x + w / 2, y + h - 12, body, ha="center", va="top", fontsize=6, linespacing=1.3)

    rd = ", ".join(f"{d['label']} {d['n']}" for d in rows)
    # revision: one common case list; readers could choose the reading order, so no "fixed order" claim
    # revision: "shared 80-case list", "STV", "proportion of readings" (label-only); final export: every reader read all 80
    ax.text(0, 99, f"shared {n_cases}-case list for all readers ({coh.get('CCDI', 0)} CCDI-MCI, {coh.get('STV', 0)} STV), reference label hidden",
            fontsize=6, va="top")
    # reader numbers grouped by expertise tier (rows are already in tier order), e.g. with five readers
    # (revision) "Readers 1–2 pediatric pathologists; 3–5 general pathologists"; cases read on their own line
    # (four or more readers do not fit one line of the panel width)
    tiers = []
    for d in rows:
        k = int(d["label"].split()[1])
        if tiers and tiers[-1][0] == d["tier"]:
            tiers[-1][1].append(k)
        else:
            tiers.append((d["tier"], [k]))
    def nums(ks):
        return f"{ks[0]}\u2013{ks[-1]}" if len(ks) > 1 and ks == list(range(ks[0], ks[-1] + 1)) else ", ".join(map(str, ks))
    tier_txt = "; ".join(f"{nums(ks)} {TIER_PLURAL[t] if len(ks) > 1 else TIER_LABEL[t]}" for t, ks in tiers)
    ax.text(0, 92, f"Reader{'s' if len(rows) > 1 else ''} {tier_txt}", fontsize=6, va="top")
    ax.text(0, 85.5, f"cases read: {rd}", fontsize=6, va="top")
    xs = [0, 34, 68]
    w = 30
    titles = ["Slide only", "+ search results", "+ LLM summary"]
    sees = ["H&E slide\n(+ age, site,\npresentation)", "PROSPER: similar\nslides, ROI search,\nheat maps",
            "text summary of\nthe retrieved\ncases"]
    for x, t, s_ in zip(xs, titles, sees):
        box(x, 44, w, 34, t, s_, "#deebf7")
    for x0 in (30, 64):
        ax.add_patch(FancyArrowPatch((x0 + 0.8, 61), (x0 + 3.4, 61), arrowstyle="-|>", mutation_scale=6, lw=0.8, color="0.2"))
    rec = ["differential \u2264 6\nfavored\nconfidence", "changed?\nnew answer\nconfidence\nusefulness",
           "changed?\nnew answer\nconfidence\nusefulness"]
    for x, s_ in zip(xs, rec):
        box(x, 1, w, 39, "recorded", s_, "#fff7e6")
    ax.text(0, -6, "answers carry forward unless changed; every stage timed", fontsize=6, color="0.3", va="center")


# ------------------------------------------------------------------ panel B: before / after per reader

def b_values(d: dict, T, r) -> dict:
    """Per-reader Step 1/2/3 estimates (S1 table), discordant counts and paired difference (T2),
    each asserted against the readings. Shared by every panel-B variant."""
    s1 = T["S1_trajectory_5_1"]
    s1 = s1[s1.engaged_B_subset == "all"]
    g = r[r.reader_id == d["reader_id"]]
    e = s1[s1.reader_id == d["reader_id"]].set_index("measure")
    fav = [e.loc[f"fav_correct_{p}"] for p in "ABC"]
    ind = [e.loc[f"in_diff_{p}"] for p in "ABC"]
    for p, row in zip("ABC", fav):
        assert int(row.k) == int(g[f"fav_correct_{p}"].sum()) and int(row.n) == len(g)
    for p, row in zip("ABC", ind):
        assert int(row.k) == int(g[f"in_diff_{p}"].sum())
    e2 = T["T2_primary_endpoints"][T["T2_primary_endpoints"].reader_id == d["reader_id"]].set_index("endpoint")
    wr = int(e2.loc["E2 discordant A-wrong→B-right (count)", "k"])
    rw = int(e2.loc["E2 discordant A-right→B-wrong (count)", "k"])
    assert wr == int((~g.fav_correct_A & g.fav_correct_B).sum()) and rw == int((g.fav_correct_A & ~g.fav_correct_B).sum())
    diff = e2.loc["E2 paired difference fav_correct B − A"]
    assert abs(diff.estimate - (fav[1].estimate - fav[0].estimate)) < 1e-9
    src = []
    for p, fr, ir in zip("ABC", fav, ind):
        src.append(dict(reader_id=d["reader_id"], reader_label=d["label"], tier=d["tier"], step=p,
                        fav_correct_k=int(fr.k), n=int(fr.n), fav_correct=fr.estimate, boot95_lo=fr.boot95_lo, boot95_hi=fr.boot95_hi,
                        in_diff_k=int(ir.k), in_diff=ir.estimate, in_diff_boot95_lo=ir.boot95_lo, in_diff_boot95_hi=ir.boot95_hi))
    src.append(dict(reader_id=d["reader_id"], reader_label=d["label"], tier=d["tier"], step="B−A",
                    fav_correct=diff.estimate, boot95_lo=diff.boot95_lo, boot95_hi=diff.boot95_hi,
                    fav_correct_k=wr, n=rw, in_diff_k=np.nan, in_diff=np.nan, in_diff_boot95_lo=np.nan, in_diff_boot95_hi=np.nan))
    return dict(fav=fav, ind=ind, wr=wr, rw=rw, diff=diff, src=src)


def _b_axes_common(ax, xs):
    ax.set_xticks(xs)
    ax.set_xticklabels(STEP_NAMES, fontsize=6.5)
    ax.set_ylabel("proportion of readings")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)


def panel_b(ax, rows, T, r) -> pd.DataFrame:
    """v2: readers separated by a horizontal offset, labelled at Step 3 (archived design)."""
    src, disc = [], []
    nr = len(rows)
    xs = np.array([0, 1, 2])
    off = np.linspace(-0.16, 0.16, nr) if nr > 1 else [0.0]
    for i, d in enumerate(rows):
        v = b_values(d, T, r)
        y = [row.estimate for row in v["fav"]]
        lo = [row.estimate - row.boot95_lo for row in v["fav"]]
        hi = [row.boot95_hi - row.estimate for row in v["fav"]]
        ax.plot(xs + off[i], [row.estimate for row in v["ind"]], color=COL_DIFF_EDGE, lw=0.7, ls=":", marker="o", ms=2.5,
                mfc=COL_DIFF, mec=COL_DIFF_EDGE, zorder=2)
        ax.errorbar(xs + off[i], y, yerr=[lo, hi], color="black", lw=0.9, marker="o", ms=4,
                    mfc=COL_FAV, mec="black", mew=0.4, capsize=1.5, elinewidth=0.6, zorder=3)
        ax.text(2 + off[i] + 0.1, y[2], d["label"], fontsize=6, va="center", ha="left")
        disc.append(f"{d['label']}: {v['wr']} wrong\u2192right, {v['rw']} right\u2192wrong")
        src += v["src"]
    ax.text(0.02, 0.03, "favored diagnosis after the search results:\n" + "\n".join(disc), transform=ax.transAxes,
            fontsize=6, va="bottom", ha="left", color="0.2")
    _b_axes_common(ax, xs)
    ax.set_xlim(-0.4, 2.9)
    ax.set_ylim(0.3, 1.02)
    handles = [Line2D([], [], marker="o", color="black", mfc=COL_FAV, mec="black", ms=4, lw=0.9, label="favored diagnosis = reference label (95 % interval)"),
               Line2D([], [], marker="o", color=COL_DIFF_EDGE, mfc=COL_DIFF, mec=COL_DIFF_EDGE, ms=2.5, lw=0.7, ls=":", label="reference label in differential")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6, frameon=False, handlelength=1.8,
              borderaxespad=0, ncol=1)
    return pd.DataFrame(src)


def panel_b_shapes(ax, rows, T, r, dodge: float = 0.05, intervals: bool = True, key_counts: bool = True) -> pd.DataFrame:
    """shapes: one marker shape per reader (row order), colour code unchanged (filled green =
    favored correct, with 95 % interval when `intervals`; hollow orange dotted = truth in differential).
    `dodge` is a small per-reader offset that only keeps the interval bars from stacking;
    0 draws all readers at exactly the same x. `key_counts` prints the discordant counts in the reader key."""
    src, handles = [], []
    nr = len(rows)
    xs = np.array([0, 1, 2])
    off = (np.arange(nr) - (nr - 1) / 2) * dodge
    for i, d in enumerate(rows):
        mk = MARKERS[i % len(MARKERS)]
        v = b_values(d, T, r)
        y = [row.estimate for row in v["fav"]]
        ax.plot(xs + off[i], [row.estimate for row in v["ind"]], color=COL_DIFF_EDGE, lw=0.7, ls=":", marker=mk, ms=3.2,
                mfc="white", mec=COL_DIFF_EDGE, mew=0.7, zorder=2)
        if intervals:
            lo = [row.estimate - row.boot95_lo for row in v["fav"]]
            hi = [row.boot95_hi - row.estimate for row in v["fav"]]
            ax.errorbar(xs + off[i], y, yerr=[lo, hi], color="black", lw=0.9, marker=mk, ms=4.8,
                        mfc=COL_FAV, mec="black", mew=0.4, capsize=1.5, elinewidth=0.6, zorder=3)
        else:
            ax.plot(xs + off[i], y, color="black", lw=0.9, marker=mk, ms=4.8, mfc=COL_FAV, mec="black", mew=0.4, zorder=3)
        lab = f"{d['label']} (n = {d['n']})"
        if key_counts:
            lab += f": {v['wr']} wrong\u2192right, {v['rw']} right\u2192wrong"
        handles.append(Line2D([], [], marker=mk, ls="none", mfc="white", mec="black", mew=0.7, ms=4.8, label=lab))
        src += v["src"]
    _b_axes_common(ax, xs)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0.3, 1.02)
    fav_lab = "favored diagnosis = reference label" + (" (95 % interval)" if intervals else "")
    metric = [Line2D([], [], marker="o", color="black", mfc=COL_FAV, mec="black", ms=4.8, lw=0.9, label=fav_lab),
              Line2D([], [], marker="o", color=COL_DIFF_EDGE, mfc="white", mec=COL_DIFF_EDGE, mew=0.7, ms=3.2, lw=0.7, ls=":", label="reference label in differential")]
    leg1 = ax.legend(handles=metric, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6, frameon=False, handlelength=1.8,
                     borderaxespad=0, ncol=1)
    ax.add_artist(leg1)
    title = "marker = reader" + ("; favored diagnosis after the search results" if key_counts else "")
    ax.legend(handles=handles, loc="lower left", fontsize=6, frameon=False, handlelength=1.0, handletextpad=0.6,
              borderaxespad=0.3, title=title, title_fontsize=6, alignment="left")
    return pd.DataFrame(src)


def diff_axis(axd, rows, vals: list[dict], marker_by_reader: bool, labels: bool) -> None:
    """Right-hand axis: paired difference Step 2 − Step 1 per reader (one row each) with its 95 %
    interval, zero line, discordant counts printed to the right."""
    dlo, dhi = [], []
    for yi, (d, v) in enumerate(zip(rows, vals)):
        dd = v["diff"]
        mk = MARKERS[yi % len(MARKERS)] if marker_by_reader else "o"
        axd.errorbar(dd.estimate, yi, xerr=[[dd.estimate - dd.boot95_lo], [dd.boot95_hi - dd.estimate]], fmt=mk, color="black",
                     mfc=COL_FAV, mec="black", mew=0.4, ms=4.8, capsize=1.5, elinewidth=0.6, lw=0.9, zorder=3)
        dlo.append(dd.boot95_lo)
        dhi.append(dd.boot95_hi)
        axd.text(1.04, yi, f"{v['wr']} wrong\u2192right\n{v['rw']} right\u2192wrong", transform=axd.get_yaxis_transform(),
                 fontsize=6, va="center", ha="left", color="0.2")
    nr = len(rows)
    axd.set_yticks(range(nr))
    axd.set_ylim(-0.6, nr - 0.4)
    axd.invert_yaxis()
    axd.tick_params(axis="y", length=0)
    axd.spines["left"].set_visible(False)
    axd.grid(axis="x", linewidth=0.3, alpha=0.5)
    if labels:
        axd.set_yticklabels([f"{d['label']}\n(n = {d['n']})" for d in rows], fontsize=6.5)
    else:
        axd.tick_params(labelleft=False)
    axd.axvline(0, color="0.3", lw=0.6)
    xl = min(-0.05, np.floor((min(dlo) - 0.05) * 10) / 10)
    xh = np.ceil((max(dhi) + 0.05) * 10) / 10
    axd.set_xlim(xl, xh)
    axd.set_xticks([t for t in np.arange(np.ceil(xl / 0.2) * 0.2, xh + 1e-9, 0.2)])
    # revision: Reader 5's interval reaches +0.73, so the axis runs to +0.8 and five labels at 0.2 spacing
    # collide at the composite width; with more than four ticks only the multiples of 0.4 are labelled
    # (ticks and grid lines stay every 0.2). Four or fewer ticks: every tick labelled, as before.
    ticks = axd.get_xticks()
    sparse = len(ticks) > 4
    axd.set_xticklabels([("" if sparse and abs(round(t / 0.2)) % 2 else (f"{t:+.1f}" if abs(t) > 1e-9 else "0")) for t in ticks],
                        fontsize=6.5)
    axd.set_xlabel("change after the\nsearch results (95 % interval)", fontsize=6.5)


def panel_b_shapes_diff(axes, rows, T, r) -> pd.DataFrame:
    """shapes_diff: left = trajectories with one marker per reader, no offset and no interval bars
    (lines stay readable with six readers); right = the pre-specified paired difference Step 2 − Step 1
    with its 95 % interval, same marker per reader, discordant counts beside it."""
    ax, axd = axes
    src = panel_b_shapes(ax, rows, T, r, dodge=0.0, intervals=False, key_counts=False)
    vals = [b_values(d, T, r) for d in rows]
    diff_axis(axd, rows, vals, marker_by_reader=True, labels=False)
    return src


def panel_b_small(axes, rows, T, r) -> pd.DataFrame:
    """small: one small trajectory axis per reader (small multiples), each with its own 95 %
    intervals and discordant counts; shared y scale. Scales to six readers as a 3 × 2 grid."""
    src = []
    xs = np.array([0, 1, 2])
    for i, (ax, d) in enumerate(zip(axes, rows)):
        v = b_values(d, T, r)
        y = [row.estimate for row in v["fav"]]
        lo = [row.estimate - row.boot95_lo for row in v["fav"]]
        hi = [row.boot95_hi - row.estimate for row in v["fav"]]
        ax.plot(xs, [row.estimate for row in v["ind"]], color=COL_DIFF_EDGE, lw=0.7, ls=":", marker="o", ms=2.5,
                mfc=COL_DIFF, mec=COL_DIFF_EDGE, zorder=2)
        ax.errorbar(xs, y, yerr=[lo, hi], color="black", lw=0.9, marker="o", ms=4, mfc=COL_FAV, mec="black", mew=0.4,
                    capsize=1.5, elinewidth=0.6, zorder=3)
        ax.set_title(f"{d['label']} (n = {d['n']})", fontsize=6.5, loc="left", pad=2)
        ax.text(0.03, 0.04, f"{v['wr']} wrong\u2192right\n{v['rw']} right\u2192wrong", transform=ax.transAxes, fontsize=6,
                va="bottom", ha="left", color="0.2")
        ax.set_xticks(xs)
        ax.set_xticklabels(["slide\nonly", "+ search", "+ LLM\nsummary"], fontsize=6)
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(0.3, 1.02)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
        if i == 0:
            ax.set_ylabel("proportion of readings")
        else:
            ax.tick_params(labelleft=False)
        src += v["src"]
    handles = [Line2D([], [], marker="o", color="black", mfc=COL_FAV, mec="black", ms=4, lw=0.9, label="favored diagnosis = reference label (95 % interval)"),
               Line2D([], [], marker="o", color=COL_DIFF_EDGE, mfc=COL_DIFF, mec=COL_DIFF_EDGE, ms=2.5, lw=0.7, ls=":", label="reference label in differential")]
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.12), fontsize=6, frameon=False, handlelength=1.8,
                   borderaxespad=0, ncol=1)
    return pd.DataFrame(src)


STEP_MK = [("A", "o", False), ("B", "o", True), ("C", "D", True)]  # (step, marker, filled)

# Retrieved labels without a reader on the same cases (plan 003 amendment 021; set by main() from --search-alone).
# When set, panel B `rows` shows a band from the nearest-slide count to the plurality-of-five count, a dotted line
# at the hit@5 count, and labels its left axis in number of cases (plotted positions unchanged).
SEARCH_ALONE: dict | None = None
SEARCH_BAND_COL = "#d9f0d3"


def load_search_alone(search_dir: Path, r: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Counts from V1_search_alone.csv (scripts/reader_study/search_vs_reader.py), all 80 cases; the nearest-slide
    and hit@5 counts are asserted against the locked covariates of readings.csv."""
    v1 = pd.read_csv(search_dir / "V1_search_alone.csv")
    v1 = v1[v1.case_set == "all 80"].replace({"search_criterion": {"hit@5 (shortlist)": "hit5"}})
    v1 = v1[v1.search_criterion.isin(["top1", "plurality5", "hit5"])].set_index("search_criterion")
    cases = r.drop_duplicates("case_id")
    true = lambda s: int(s.astype(str).str.lower().eq("true").sum())
    assert len(cases) == int(v1.loc["top1", "n"]) == int(v1.loc["plurality5", "n"]) == int(v1.loc["hit5", "n"])
    assert true(cases.top1_correct) == int(v1.loc["top1", "k"]) and true(cases.hit5) == int(v1.loc["hit5", "k"])
    assert v1.loc["top1", "k"] <= v1.loc["plurality5", "k"] <= v1.loc["hit5", "k"]
    s = {k: int(v1.loc[k, "k"]) for k in ("top1", "plurality5", "hit5")}
    s["n"] = int(v1.loc["top1", "n"])
    return s, v1.loc[["top1", "plurality5", "hit5"], ["k", "n", "estimate", "boot95_lo", "boot95_hi"]].reset_index()


def panel_b_rows(axes, rows, T, r) -> pd.DataFrame:
    """rows: readers as rows (same order/labels as C–F). Left: proportion at Step 1 (hollow) →
    Step 2 (filled) → Step 3 (diamond), favored-correct in green, truth-in-differential in orange
    on a thinner line above. Right: pre-specified paired difference Step 2 − Step 1 with its 95 %
    interval and the discordant counts."""
    ax, axd = axes
    src = []
    allv, vals = [], []
    for yi, d in enumerate(rows):
        v = b_values(d, T, r)
        vals.append(v)
        yf, yd = yi + 0.14, yi - 0.2
        xi = [e.estimate for e in v["ind"]]
        xf = [e.estimate for e in v["fav"]]
        allv += xi + xf
        ax.plot(xi, [yd] * 3, color=COL_DIFF_EDGE, ls=":", lw=0.7, zorder=2)
        ax.plot(xf, [yf] * 3, color=COL_FAV, lw=1.0, zorder=2)
        for (p, mk, filled), e in zip(STEP_MK, v["ind"]):
            ax.plot(e.estimate, yd, marker=mk, ms=3.2 if mk == "o" else 2.8, mfc=COL_DIFF if filled else "white", mec=COL_DIFF_EDGE,
                    mew=0.7, ls="none", zorder=4 if mk == "o" else 3)
        for (p, mk, filled), e in zip(STEP_MK, v["fav"]):
            ax.plot(e.estimate, yf, marker=mk, ms=4.8 if mk == "o" else 5.2, mfc=COL_FAV if filled else "white", mec=COL_FAV,
                    mew=0.9, ls="none", zorder=4 if mk == "o" else 3)
        src += v["src"]
    nr = len(rows)
    ax.set_yticks(range(nr))
    ax.set_ylim(-0.6, nr - 0.4)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    ax.set_yticklabels([f"{d['label']}\n(n = {d['n']})" for d in rows], fontsize=6.5)
    diff_axis(axd, rows, vals, marker_by_reader=False, labels=False)
    lo = max(0.0, np.floor((min(allv) - 0.08) * 10) / 10)
    ax.set_xlim(lo, 1.02)
    ax.set_xlabel("proportion of readings", fontsize=6.5)
    handles = [Line2D([], [], marker="o", ls="none", mfc="white", mec=COL_FAV, mew=0.9, ms=4.8, label=STAGE["A"]),
               Line2D([], [], marker="o", ls="none", mfc=COL_FAV, mec=COL_FAV, ms=4.8, label=STAGE["B"]),
               Line2D([], [], marker="D", ls="none", mfc=COL_FAV, mec=COL_FAV, ms=5.2, label=STAGE["C"]),
               Line2D([], [], color=COL_FAV, lw=1.0, label="favored diagnosis = reference label"),
               Line2D([], [], color=COL_DIFF_EDGE, lw=0.7, ls=":", marker="o", mfc="white", mec=COL_DIFF_EDGE, mew=0.7, ms=3.2,
                      label="reference label in differential")]
    if SEARCH_ALONE:
        s, n = SEARCH_ALONE, SEARCH_ALONE["n"]
        assert all(int(v["fav"][0].n) == n for v in vals), "the case-count axis needs every reader on the same cases"
        ax.axvspan(s["top1"] / n, s["plurality5"] / n, color=SEARCH_BAND_COL, lw=0, zorder=0)
        ax.axvline(s["hit5"] / n, color=COL_DIFF_EDGE, lw=0.8, ls=":", zorder=1)
        ticks = [k for k in range(0, n + 1, 20) if k / n >= lo]
        ax.set_xticks([k / n for k in ticks])
        ax.set_xticklabels([str(k) for k in ticks])
        ax.set_xlabel(f"number of cases (of {n})", fontsize=6.5)
        # the three stage markers stay alone in the first key column (empty entry in its fourth row)
        handles = (handles[:3] + [Line2D([], [], ls="none", label="")] + handles[3:]
                   + [Patch(facecolor=SEARCH_BAND_COL, edgecolor="none", label="search alone: nearest slide to vote of 5"),
                      Line2D([], [], color=COL_DIFF_EDGE, lw=0.8, ls=":", label="search alone: reference label in 5 nearest")])
    ax.legend(handles=handles, labels=[h.get_label() for h in handles], loc="lower left", bbox_to_anchor=(0.0, 1.0),
              fontsize=6, frameon=False, handlelength=1.6, borderaxespad=0, ncol=2, columnspacing=1.0)
    return pd.DataFrame(src)


# ------------------------------------------------------------------ panel C: outcome classes per reader

def class_counts(T, rid, transition="A→B") -> dict:
    t3 = T["T3_outcome_classes"]
    c = t3[(t3.reader_id == rid) & (t3.transition == transition)].copy()
    c["class"] = c["class"].str.strip()  # the sub-counts are indented in the table
    c = c.set_index("class")["count"]
    out = {k: int(c.loc[k]) for k in CLASS_ORDER if k in c.index}
    # the two confirm sub-counts carry the class total in `confirm`
    assert out["confirm_favoured"] + out["confirm_differential_only"] == int(c.loc["confirm"])
    return out


def count_axis_max(n_max: int) -> int:
    step = 20 if n_max > 40 else 10
    return int(np.ceil(n_max / step) * step)


def place_counts(ax, items: list[tuple[float, float, int, float, str]], fontsize: float = 6) -> None:
    """Segment-count labels for stacked bars, placed after the axis limits are set (rows at integer y,
    axis inverted). items: (segment left, width, row, label value, inside colour), in segment order.
    A label goes inside its segment when the segment is wide enough for it at the drawn size; otherwise
    it goes in the white gap just above its own bar (each gap belongs to the bar below it, so a label is
    never read against the wrong row), centred on its segment and nudged right only as far as needed to
    clear the previous label there, so the labels keep the segment order. Widths are estimated in points
    at the drawn axes size, so this holds for any number of rows and any axes width."""
    fig = ax.figure
    pos = ax.get_position()
    w_pt = pos.width * fig.get_figwidth() * 72
    x0, x1 = ax.get_xlim()
    pt = w_pt / (x1 - x0)  # points per case on this axis
    pad = 1.5 / pt
    right_edge: dict[int, float] = {}  # row -> right end of the last label in the gap above that row
    for left, k, yi, val, inside_col in items:
        s = str(int(val))
        tw = 0.66 * fontsize * len(s) / pt  # label width in data units
        xc = left + k / 2
        if k - tw >= 1.0 / pt:
            ax.text(xc, yi, s, ha="center", va="center", fontsize=fontsize, color=inside_col)
            continue
        xc = max(xc, right_edge.get(yi, -np.inf) + pad + tw / 2)
        right_edge[yi] = xc + tw / 2
        ax.text(xc, yi - 0.5, s, ha="center", va="center", fontsize=fontsize, color="0.25")


def stacked_rows(ax, rows_spec: list[tuple[str, dict, int]], show_zero_mislead=True) -> None:
    """rows_spec: (label, counts dict, n). Bars in case counts (2026-10-28: not percentages, so
    the bar length shows each reader's actual number of cases); segment labels are the counts."""
    xmax = count_axis_max(max(n for _, _, n in rows_spec))
    items = []  # segment-count labels, placed by place_counts once the axis limits are set
    for yi, (lab, cnt, n) in enumerate(rows_spec):
        if n == 0:
            ax.text(xmax / 2, yi, "no readings", ha="center", va="center", fontsize=6, color="0.45", style="italic")
            continue
        left = 0.0
        for cls in CLASS_ORDER:
            k = cnt.get(cls, 0)
            if k:
                ax.barh(yi, k, left=left, color=CLASS_COL[cls], edgecolor="white", linewidth=0.4, height=0.62)
                items.append((left, k, yi, k, "white" if cls in ("mislead", "correct", "other change") else "black"))
                left += k
        if show_zero_mislead:
            ax.text(1.02 * xmax, yi, f"{ZERO_LABEL} {cnt.get('mislead', 0)}", ha="left", va="center", fontsize=6,
                    color=CLASS_COL["mislead"])
    ax.set_yticks(range(len(rows_spec)))
    ax.set_yticklabels([lab for lab, _, _ in rows_spec], fontsize=6.5)
    ax.set_ylim(-0.6, len(rows_spec) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0, xmax + 1, 20 if xmax > 40 else 10))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    place_counts(ax, items)


TRANSITION = {"A→B": ("class_AB", "confirm_sub_AB", AFTER["A→B"]),
              "B→C": ("class_BC", "confirm_sub_BC", AFTER["B→C"])}


def panel_c(ax, rows, T, r, transition: str = "A→B") -> pd.DataFrame:
    """Outcome classes per reader for one transition (C: Step 1→2; D: Step 2→3, the summary step,
    which follows Step 2 so its effect is not isolated). Counts asserted against the readings."""
    col, sub_col, step_lab = TRANSITION[transition]
    spec, src = [], []
    for d in rows:
        cnt = class_counts(T, d["reader_id"], transition)
        g = r[r.reader_id == d["reader_id"]]
        if transition == "B→C":
            assert g.has_C.astype(bool).all(), "a reading without Step 3 would need a separate denominator"
        for cls in ("mislead", "correct", "expand", "unhelpful", "other change"):
            assert cnt[cls] == int((g[col] == cls).sum()), cls
        assert cnt["confirm_favoured"] == int(((g[col] == "confirm") & (g[sub_col] == "confirm_favoured")).sum())
        assert sum(cnt.values()) == d["n"]
        spec.append((f"{d['label']}\n(n = {d['n']})", cnt, d["n"]))
        src += [dict(reader_id=d["reader_id"], reader_label=d["label"], tier=d["tier"], transition=transition, outcome_class=cls,
                     display_label=CLASS_LABEL[cls], count=cnt[cls], n=d["n"]) for cls in CLASS_ORDER]
    stacked_rows(ax, spec)
    ax.set_xlabel(f"number of cases: change in the answer {step_lab}", fontsize=6.5)
    return pd.DataFrame(src)


def legend_classes(ax):
    handles = [Patch(facecolor=CLASS_COL[c], edgecolor="0.4", linewidth=0.3, label=CLASS_LABEL[c]) for c in CLASS_ORDER]
    ax.legend(handles=handles, loc="center", fontsize=6, frameon=False, ncol=3, handlelength=1.0, columnspacing=1.0,
              borderaxespad=0)
    ax.axis("off")


# ------------------------------------------------------------------ panel D: outcome by hit@10 (10 slides shown)

def classes_from_readings(g: pd.DataFrame) -> dict:
    cnt = {cls: int((g.class_AB == cls).sum()) for cls in ("mislead", "correct", "expand", "unhelpful", "other change")}
    cnt["confirm_favoured"] = int(((g.class_AB == "confirm") & (g.confirm_sub_AB == "confirm_favoured")).sum())
    cnt["confirm_differential_only"] = int(((g.class_AB == "confirm") & (g.confirm_sub_AB == "confirm_differential_only")).sum())
    assert sum(cnt.values()) == len(g)
    return cnt


def panel_d(ax, rows, T, r) -> pd.DataFrame:
    """Outcome after the search results, split by whether the true group was among the 10 slides the
    reader was shown (`hit10`; the instrument displayed the 10 nearest slides — 2026-10-28).
    Plan 002 §5.3 pre-specified `hit5` for table T4, so the strata here come straight from the accepted
    readings.csv; per-stratum counts are asserted to add up to the T3 class totals."""
    assert (r.n_neighbours == 10).all(), "panel D assumes 10 slides shown per case"
    assert (r.hit10.astype(bool) == (r.n_truth_top10 > 0)).all()
    spec, src = [], []
    for d in rows:
        g = r[r.reader_id == d["reader_id"]]
        tot = class_counts(T, d["reader_id"])
        parts = []
        for level, lab in ((True, "reference label among\nthe 10 slides shown"), (False, "reference label not among\nthe 10 slides shown")):
            sub = g[g.hit10.astype(bool) == level]
            n = len(sub)
            cnt = classes_from_readings(sub) if n else {c: 0 for c in CLASS_ORDER}
            parts.append(cnt)
            spec.append((f"{d['label']}\n{lab}\n(n = {n})", cnt, n))
            src += [dict(reader_id=d["reader_id"], reader_label=d["label"], tier=d["tier"], hit10=level, outcome_class=cls,
                         display_label=CLASS_LABEL[cls], count=cnt[cls], n=n) for cls in CLASS_ORDER]
        for cls in CLASS_ORDER:
            assert parts[0][cls] + parts[1][cls] == tot[cls], (d["reader_id"], cls)
    stacked_rows(ax, spec)
    ax.set_xlabel("number of cases: change in the answer after seeing the search results", fontsize=6.5)
    return pd.DataFrame(src)


# ------------------------------------------------------------------ panel E: usefulness (diverging)

def panel_e(ax, rows, T, r, variant: str = "v3") -> pd.DataFrame:
    col = LIK_COL[variant]
    t2 = T["T2_primary_endpoints"]
    src, ylabels, ext, medians = [], [], [], []
    items = []  # segment-count labels, placed by place_counts once the axis limits are set
    yi = 0
    for d in rows:
        g = r[r.reader_id == d["reader_id"]]
        for use, lab in (("use_B", "search results"), ("use_C", "LLM summary")):
            counts = {lv: int((g[use] == lv).sum()) for lv in LIKERT}
            n = int(g[use].notna().sum())
            if use == "use_B":
                e2 = t2[t2.reader_id == d["reader_id"]].set_index("endpoint")
                for lv in LIKERT:
                    assert counts[lv] == int(e2["k"][e2.index.str.startswith(f"E4 use_B = {lv} ")].iloc[0])
            med = float(g[use].median())
            # diverging, in case counts: 1,2 to the left of zero, 3 centred, 4,5 to the right
            left = -(counts[1] + counts[2] + counts[3] / 2)
            ext.append((left, left + n))
            for lv in LIKERT:
                k = counts[lv]
                if k:
                    ax.barh(yi, k, left=left, color=col[lv], edgecolor="white", linewidth=0.4, height=PAIRED_BAR_H)
                    items.append((left, k, yi, k, "white" if lv in (1, 5) else "black"))
                left += k
            medians.append((yi, med))
            ylabels.append((d["label"], lab, n))
            src += [dict(reader_id=d["reader_id"], reader_label=d["label"], tier=d["tier"], rating=use, level=lv,
                         count=counts[lv], n=n, median=med) for lv in LIKERT]
            yi += 1
    diverging_count_axis(ax, ext, yi, ylabels)
    place_counts(ax, items)
    for y, med in medians:
        ax.text(1.02, y, f"median {med:g}", transform=ax.get_yaxis_transform(), fontsize=6, va="center", ha="left", color="0.2")
    ax.set_xlabel("number of cases: how useful was it for this case? (\u2190 not useful | useful \u2192)", fontsize=6.5)
    return pd.DataFrame(src)


# E and G draw two rows per reader; their bars are thinner than C/D's so that, with four readers (eight
# rows), the count labels of thin segments still fit in the white gap between bars at the composite size.
PAIRED_BAR_H = 0.45
# revision (five readers): E's and G's axes were tuned at 1.5 in for four readers (eight rows, 0.1875 in per
# row). Every further reader adds two rows at that height, so a thin segment's count keeps the same room;
# the composites and the standalone E/G slots grow by `paired_extra_in` and nothing else moves relative
# to its own row. With four or fewer readers the extra is 0 and the layout is the four-reader one.
PAIRED_ROW_IN = 1.5 / 8
PAIRED_BASE_READERS = 4


def paired_extra_in(n_readers: int) -> float:
    """Extra axes height (inches) the two-rows-per-reader panels E and G need beyond the four-reader layout."""
    return 2 * PAIRED_ROW_IN * max(0, n_readers - PAIRED_BASE_READERS)


def paired_row_labels(ax, ylabels: list[tuple[str, str, int]]) -> None:
    """Row labels for the two-rows-per-reader panels (E, G). Each row's tick label is one line (the stage);
    the reader label sits once per reader, left of the tick labels and centred on that reader's rows,
    with its n when all of that reader's rows share it (otherwise each row carries its own n)."""
    groups = []  # (reader label, first row, last row)
    for i, (lab, _, _) in enumerate(ylabels):
        if groups and groups[-1][0] == lab:
            groups[-1][2] = i
        else:
            groups.append([lab, i, i])
    ticks, group_text = [], []
    for lab, i0, i1 in groups:
        ns = {ylabels[i][2] for i in range(i0, i1 + 1)}
        same = len(ns) == 1
        ticks += [ylabels[i][1] if same else f"{ylabels[i][1]} (n = {ylabels[i][2]})" for i in range(i0, i1 + 1)]
        group_text.append((f"{lab}\n(n = {ns.pop()})" if same else lab, i0, i1))
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ticks, fontsize=6.5)
    renderer = ax.figure.canvas.get_renderer()
    w_pt = max(t.get_window_extent(renderer).width for t in ax.get_yticklabels()) * 72 / ax.figure.dpi
    off = w_pt + plt.rcParams["ytick.major.pad"] + 5
    for text, i0, i1 in group_text:
        ax.annotate(text, xy=(0, (i0 + i1) / 2), xycoords=("axes fraction", "data"), xytext=(-off, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=6.5)


def diverging_count_axis(ax, ext: list[tuple[float, float]], n_rows: int, ylabels: list[tuple[str, str, int]]) -> None:
    """Shared axis dressing for the diverging count panels (E, G): symmetric-looking count axis whose
    limits come from the data, ticks every 20 (or 10) cases labelled with absolute counts."""
    lo = min(a for a, _ in ext)
    hi = max(b for _, b in ext)
    step = 20 if (hi - lo) > 60 else 10
    xl = np.floor(lo / step) * step
    xh = np.ceil(hi / step) * step
    ax.axvline(0, color="0.3", lw=0.6)
    assert len(ylabels) == n_rows
    paired_row_labels(ax, ylabels)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(xl, xh)
    ticks = np.arange(xl, xh + 1e-9, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(int(t))}" for t in ticks], fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def legend_likert(ax, variant: str = "v3"):
    handles = [Patch(facecolor=LIK_COL[variant][lv], edgecolor="0.4", linewidth=0.3, label=LIK_LAB[variant][lv]) for lv in LIKERT]
    ax.legend(handles=handles, loc="center", fontsize=6, frameon=False, ncol=5, handlelength=1.0, columnspacing=1.0,
              borderaxespad=0)
    ax.axis("off")


# ------------------------------------------------------------------ panel G: confidence change by correctness

CONF_ORDER = ["down_wrong", "down_right", "same", "up_right", "up_wrong"]
CONF_COL = {"down_wrong": "#c6dbef", "down_right": "#fdae6b", "same": "#d9d9d9", "up_right": "#74c476", "up_wrong": "#b2182b"}
CONF_LAB = {"down_wrong": "less confident, favored wrong", "down_right": "less confident, favored correct",
            "same": "unchanged", "up_right": "more confident, favored correct", "up_wrong": "more confident, favored wrong"}


def panel_g(ax, rows, T, r) -> pd.DataFrame:
    """Change in the reader's confidence (1–5) after the search results and after the LLM summary,
    split by whether the favored diagnosis *after* that stage was the true group. "More confident,
    favored wrong" is the harmful-confidence count of plan 002 §3.4 (asserted against S2).
    Medians of the confidence before → after are printed at the right."""
    s2 = T["S2_confidence_5_2"]
    s2 = s2[s2.engaged_B_subset == "all"]
    src, ylabels, ext, medians = [], [], [], []
    items = []  # segment-count labels, placed by place_counts once the axis limits are set
    yi = 0
    for d in rows:
        g = r[r.reader_id == d["reader_id"]]
        e = s2[s2.reader_id == d["reader_id"]]
        for before, after, dcol, lab, tr in (("conf_A", "conf_B", "dconf_BA", "after the search results", "A→B"),
                                             ("conf_B", "conf_C", "dconf_CB", "after the LLM summary", "B→C")):
            correct_after = g["fav_correct_" + after[-1]].astype(bool)
            dc = g[dcol].astype(float)
            assert ((g[after] - g[before]) == dc).all()
            cnt = {"down_wrong": int(((dc < 0) & ~correct_after).sum()), "down_right": int(((dc < 0) & correct_after).sum()),
                   "same": int((dc == 0).sum()), "up_right": int(((dc > 0) & correct_after).sum()),
                   "up_wrong": int(((dc > 0) & ~correct_after).sum())}
            n = int(dc.notna().sum())
            assert sum(cnt.values()) == n == len(g)
            if tr == "A→B":
                harm = e[e.measure.str.startswith("harmful_conf_B")]["count"].iloc[0]
                assert cnt["up_wrong"] == int(harm) == int(g.harmful_conf_B.astype(bool).sum())
            for c in (before, after):
                row = e[(e.measure == c) & (e.subset == "all")].iloc[0]
                assert float(row["median"]) == float(g[c].median()) and int(row.n) == len(g)
                assert all(int(row[f"n_{lv}"]) == int((g[c] == lv).sum()) for lv in LIKERT)
            med_b, med_a = float(g[before].median()), float(g[after].median())
            left = -(cnt["down_wrong"] + cnt["down_right"] + cnt["same"] / 2)
            ext.append((left, left + n))
            for c in CONF_ORDER:
                k = cnt[c]
                if k:
                    ax.barh(yi, k, left=left, color=CONF_COL[c], edgecolor="white", linewidth=0.4, height=PAIRED_BAR_H)
                    items.append((left, k, yi, k, "white" if c == "up_wrong" else "black"))
                left += k
            medians.append((yi, med_b, med_a))
            ylabels.append((d["label"], lab, n))
            src += [dict(reader_id=d["reader_id"], reader_label=d["label"], tier=d["tier"], transition=tr, change=c,
                         display_label=CONF_LAB[c], count=cnt[c], n=n, median_before=med_b, median_after=med_a) for c in CONF_ORDER]
            yi += 1
    diverging_count_axis(ax, ext, yi, ylabels)
    place_counts(ax, items)
    for y, mb, ma in medians:
        ax.text(1.02, y, f"median {mb:g} \u2192 {ma:g}", transform=ax.get_yaxis_transform(), fontsize=6, va="center", ha="left",
                color="0.2")
    ax.set_xlabel("number of cases: change in confidence (\u2190 less | more \u2192)", fontsize=6.5)
    return pd.DataFrame(src)


def legend_conf(ax):
    handles = [Patch(facecolor=CONF_COL[c], edgecolor="0.4", linewidth=0.3, label=CONF_LAB[c]) for c in CONF_ORDER]
    ax.legend(handles=handles, loc="center", fontsize=6, frameon=False, ncol=3, handlelength=1.0, columnspacing=1.0,
              borderaxespad=0)
    ax.axis("off")


# ------------------------------------------------------------------ panel F: tool use and time

def panel_f(axes, rows, T, r) -> pd.DataFrame:
    ax_eng, ax_roi, ax_t = axes
    t6 = T["T6_reading_time_5_6"]
    t6 = t6[t6.grouping == "all"]
    src = []
    ylab = []
    for yi, d in enumerate(rows):
        g = r[r.reader_id == d["reader_id"]]
        n = len(g)
        eng = int(g.engaged_B.astype(bool).sum())
        xmax_eng = count_axis_max(max(dd["n"] for dd in rows))
        ax_eng.barh(yi, eng, color="#4d4d4d", height=0.6)
        ax_eng.text(eng + 0.03 * xmax_eng, yi, f"{eng}/{n}", va="center", fontsize=6)
        roi = g.n_roi_selected.astype(float)
        q1, med, q3 = roi.quantile([0.25, 0.5, 0.75])
        ax_roi.text(0.98, yi + 0.36, f"{int((roi > 0).sum())}/{n} readings > 0", transform=ax_roi.get_yaxis_transform(),
                    va="center", ha="right", fontsize=6, color="0.3")
        ax_roi.errorbar(med, yi, xerr=[[med - q1], [q3 - med]], fmt="o", color="#4d4d4d", ms=4, capsize=2, lw=0.8)
        row6 = t6[t6.reader_id == d["reader_id"]].iloc[0]
        for j, (p, col) in enumerate(zip("ABC", ("#9ecae1", "#4292c6", "#08519c"))):
            med_s, q1_s, q3_s = row6[f"sec_{p}_median"], row6[f"sec_{p}_q1"], row6[f"sec_{p}_q3"]
            assert abs(med_s - g[f"sec_{p}"].median()) < 1e-6
            yy = yi + (j - 1) * 0.24
            ax_t.errorbar(med_s, yy, xerr=[[med_s - q1_s], [q3_s - med_s]], fmt="o", color=col, ms=3.5, capsize=1.5, lw=0.8,
                          label=["slide only", "+ search", "+ summary"][j] if yi == 0 else None)
            src.append(dict(reader_id=d["reader_id"], reader_label=d["label"], tier=d["tier"], measure=f"sec_{p}",
                            median=med_s, q1=q1_s, q3=q3_s, n=n))
        src.append(dict(reader_id=d["reader_id"], reader_label=d["label"], tier=d["tier"], measure="engaged_B",
                        median=eng / n, q1=np.nan, q3=np.nan, n=n))
        src.append(dict(reader_id=d["reader_id"], reader_label=d["label"], tier=d["tier"], measure="n_roi_selected",
                        median=med, q1=q1, q3=q3, n=n))
        ylab.append(f"{d['label']}\n(n = {n})")
    for ax in axes:
        ax.set_yticks(range(len(rows)))
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    ax_eng.set_yticklabels(ylab, fontsize=6.5)
    for ax in (ax_roi, ax_t):
        ax.tick_params(labelleft=False)
    xmax_eng = count_axis_max(max(dd["n"] for dd in rows))
    ax_eng.set_xlim(0, xmax_eng)
    ax_eng.set_xticks([0, xmax_eng // 2, xmax_eng])
    # engaged_B also counts ROI selections ([internal record] Unit 17), so not "cases with any retrieved image opened"
    ax_eng.set_title("engaged\nreadings", fontsize=6.5, loc="left")
    ax_roi.set_xlim(-0.3, max(2.0, float(r.groupby("reader_id").n_roi_selected.quantile(0.75).max()) + 1.0))
    ax_roi.set_title("ROI searches\nper reading", fontsize=6.5, loc="left")
    ax_t.set_xscale("log")
    # limits from the data (revision: the fixed 4–400 s axis left the general pathologists' slide-only
    # medians, > 600 s, off the axis); lower limit kept at 4 s
    q3_max = max(float(t6[t6.reader_id == d["reader_id"]][[f"sec_{p}_q3" for p in "ABC"]].max(axis=1).iloc[0]) for d in rows)
    t_hi = max(400.0, 1.3 * q3_max)
    ticks = [5, 20, 60, 240] if t_hi <= 400 else [t for t in (10, 100, 1000, 10000) if t < t_hi]
    ax_t.set_xlim(4, t_hi)
    ax_t.set_xticks(ticks)
    ax_t.set_xticklabels([str(t) for t in ticks[:-1]] + [f"{ticks[-1]} s"], fontsize=6.5)
    ax_t.minorticks_off()
    ax_t.set_title("seconds per stage within\nthe instrument (median, IQR)", fontsize=6.5, loc="left")
    ax_t.legend(loc="center right", fontsize=6, frameon=False, ncol=1, handlelength=1.0, borderaxespad=0.2)
    return pd.DataFrame(src)


# ------------------------------------------------------------------ main

def draw_b(fig_slot, fig, rows, T, r, variant: str, dodge: float):
    """Add the panel-B axes into a gridspec slot (or a whole standalone figure) and draw the variant.
    Returns (axes_for_letter, source frame)."""
    if variant in ("rows", "shapes_diff"):
        gsB = fig_slot.subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.12 if variant == "rows" else 0.3)
        axes = [fig.add_subplot(gsB[0]), fig.add_subplot(gsB[1])]
        fn = panel_b_rows if variant == "rows" else panel_b_shapes_diff
        return axes[0], fn(axes, rows, T, r)
    if variant == "small":
        nr = len(rows)
        ncol = min(nr, 3)
        nrow = int(np.ceil(nr / ncol))
        gsB = fig_slot.subgridspec(nrow, ncol, wspace=0.12, hspace=0.6)
        axes = [fig.add_subplot(gsB[i // ncol, i % ncol]) for i in range(nr)]
        return axes[0], panel_b_small(axes, rows, T, r)
    ax = fig.add_subplot(fig_slot)
    if variant == "shapes":
        return ax, panel_b_shapes(ax, rows, T, r, dodge=dodge)
    if variant == "shapes0":
        return ax, panel_b_shapes(ax, rows, T, r, dodge=0.0)
    if variant == "v2":
        return ax, panel_b(ax, rows, T, r)
    raise SystemExit(f"unknown panel-B variant {variant}")


def draw_c(fig_slot, fig, rows, T, r, transition: str = "A→B"):
    gsC = fig_slot.subgridspec(2, 1, height_ratios=[1.0, 0.28], hspace=0.55)
    ax, axleg = fig.add_subplot(gsC[0]), fig.add_subplot(gsC[1])
    src = panel_c(ax, rows, T, r, transition)
    legend_classes(axleg)
    return ax, src


def draw_d(fig_slot, fig, rows, T, r):
    gsD = fig_slot.subgridspec(2, 1, height_ratios=[1.0, 0.2], hspace=0.5)
    ax, axleg = fig.add_subplot(gsD[0]), fig.add_subplot(gsD[1])
    src = panel_d(ax, rows, T, r)
    legend_classes(axleg)
    return ax, src


def paired_axes_ratio(slot_in: float, extra_in: float, key_ratio: float, hspace: float) -> float:
    """Axes height ratio for an E/G standalone slot that has grown by extra_in: the key row keeps the
    height it had in the four-reader slot (ratio 1 : key_ratio), the axes take the added height."""
    if not extra_in:
        return 1.0
    unit = slot_in / ((1 + key_ratio) * (1 + hspace / 2))  # four-reader axes height
    return (slot_in + extra_in) / (unit * (1 + hspace / 2)) - key_ratio


def draw_e(fig_slot, fig, rows, T, r, variant: str, axes_ratio: float = 1.0):
    gsE = fig_slot.subgridspec(2, 1, height_ratios=[axes_ratio, 0.22], hspace=0.55)
    ax, axleg = fig.add_subplot(gsE[0]), fig.add_subplot(gsE[1])
    src = panel_e(ax, rows, T, r, variant=variant)
    legend_likert(axleg, variant=variant)
    return ax, src


def draw_g(fig_slot, fig, rows, T, r, axes_ratio: float = 1.0):
    gsG = fig_slot.subgridspec(2, 1, height_ratios=[axes_ratio, 0.28], hspace=0.55)
    ax, axleg = fig.add_subplot(gsG[0]), fig.add_subplot(gsG[1])
    src = panel_g(ax, rows, T, r)
    legend_conf(axleg)
    return ax, src


def draw_f(fig_slot, fig, rows, T, r):
    gsF = fig_slot.subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.5], wspace=0.28)
    axes = [fig.add_subplot(gsF[i]) for i in range(3)]
    return axes[0], panel_f(axes, rows, T, r)


# Standalone panel figures are drawn at the physical size of the panel's slot in the composite
# (one column of a 180 mm, two-column grid ≈ 2.4 in of axes width; row ≈ 1.75 in of axes height),
# so font sizes, marker sizes and spacing are what the composite will show.
# (figsize_in, [left, bottom, right, top] of the slot as figure fractions)
PANEL_FIG = {
    "A": ((3.6, 2.6), (0.03, 0.08, 0.97, 0.95)),
    "B": ((3.6, 2.9), (0.17, 0.17, 0.97, 0.86)),
    "B_rows": ((4.4, 2.6), (0.15, 0.20, 0.86, 0.80)),
    "B_shapes_diff": ((4.4, 2.9), (0.12, 0.17, 0.86, 0.86)),
    "B_small": ((3.6, 2.9), (0.17, 0.17, 0.97, 0.80)),
    "C": ((3.6, 2.6), (0.17, 0.05, 0.86, 0.95)),
    "D": ((3.6, 2.6), (0.17, 0.05, 0.86, 0.95)),
    "S_hit10": ((3.6, 3.4), (0.17, 0.04, 0.86, 0.97)),
    "E": ((3.6, 3.0), (0.26, 0.05, 0.84, 0.95)),
    "F": ((3.6, 2.6), (0.12, 0.10, 0.97, 0.85)),
    "G": ((3.6, 3.0), (0.30, 0.05, 0.84, 0.95)),
}


def render_panel(letter: str, variant: str | None, rows, T, r, inputs, dodge: float, base_note: str) -> None:
    sub = f"{letter}_{variant}" if variant else letter
    key = f"{letter}_{variant}" if f"{letter}_{variant}" in PANEL_FIG else letter
    figsize, (l, b, rt, t) = PANEL_FIG[key]
    axes_ratio = 1.0
    if letter in ("E", "G") and paired_extra_in(len(rows)):  # revision: same inch margins and key, taller axes
        w0, h0 = figsize
        x = paired_extra_in(len(rows))
        axes_ratio = paired_axes_ratio(h0 * (t - b), x, 0.22 if letter == "E" else 0.28, 0.55)
        h1 = h0 + x
        b, t = b * h0 / h1, 1 - (1 - t) * h0 / h1
        figsize = (w0, h1)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 1, left=l, bottom=b, right=rt, top=t)
    slot = gs[0, 0]
    src = None
    if letter == "A":
        panel_a(fig.add_subplot(slot), rows, r)
    elif letter == "B":
        _, src = draw_b(slot, fig, rows, T, r, variant, dodge)
    elif letter == "C":
        _, src = draw_c(slot, fig, rows, T, r, "A→B")
    elif letter == "D":
        _, src = draw_c(slot, fig, rows, T, r, "B→C")
    elif letter == "S_hit10":
        _, src = draw_d(slot, fig, rows, T, r)
    elif letter == "E":
        _, src = draw_e(slot, fig, rows, T, r, variant, axes_ratio)
    elif letter == "F":
        _, src = draw_f(slot, fig, rows, T, r)
    elif letter == "G":
        _, src = draw_g(slot, fig, rows, T, r, axes_ratio)
    else:
        raise SystemExit(f"unknown panel {letter}")
    figure_key = f"{FIGURE}/panels/{sub}"
    d = fig_dir(figure_key)
    leg = legibility(fig, d, f"8{sub}")
    if not leg["pass"]:
        print(json.dumps({k: v for k, v in leg.items() if k != "texts"}, indent=1))
    assert leg["pass"], f"legibility check failed for panel {sub}"
    note = f"Standalone candidate for Fig. 8 panel {letter}" + (f", variant `{variant}`" if variant else "") + \
           f" (drawn at composite slot size; compose with --compose once settled). " + base_note
    save(fig, figure_key, f"panel_{sub}", inputs, note, {"source_data": src} if src is not None else None)
    with open(d / "SHA256SUMS.txt", "a") as fh:
        fh.write(f"{sha256(d / f'legibility_check_8{sub}.json')}  legibility_check_8{sub}.json\n")
    plt.close(fig)
    print(json.dumps({"panel": sub, **{k: v for k, v in leg.items() if k != "texts"}}))


def draw_g_wide(fig, rect, rows, T, r):
    """Panel G across the full figure width at reduced height: the diverging axis keeps about a
    one-column width (so its count scale reads like E's), the medians sit right of it and the key is
    one column at the far right instead of a three-column block underneath. rect = (l, b, r, t)."""
    l, b, rt, t = rect
    ax = fig.add_axes([l, b, rt - l, t - b])
    src = panel_g(ax, rows, T, r)
    axleg = fig.add_axes([rt + 0.14, b, 0.99 - (rt + 0.14), t - b])
    handles = [Patch(facecolor=CONF_COL[c], edgecolor="0.4", linewidth=0.3, label=CONF_LAB[c]) for c in CONF_ORDER]
    axleg.legend(handles=handles, loc="center left", fontsize=6, frameon=False, ncol=1, handlelength=1.0,
                 borderaxespad=0, labelspacing=0.6)
    axleg.axis("off")
    return ax, src


def _draw_slot(letter, slot, fig, rows, T, r, b_variant, e_variant, dodge):
    if letter == "A":
        ax = fig.add_subplot(slot)
        panel_a(ax, rows, r)
        return ax, None
    if letter == "B":
        return draw_b(slot, fig, rows, T, r, b_variant, dodge)
    if letter == "C":
        return draw_c(slot, fig, rows, T, r, "A→B")
    if letter == "D":
        return draw_c(slot, fig, rows, T, r, "B→C")
    if letter == "E":
        return draw_e(slot, fig, rows, T, r, e_variant)
    if letter == "F":
        return draw_f(slot, fig, rows, T, r)
    if letter == "G":
        return draw_g(slot, fig, rows, T, r)
    raise SystemExit(f"panel {letter} is not a composite panel")


# Seven-panel composite geometry (2026-11-02: A–G, B `rows`, E `v3`), in inches from the bottom of
# a 9.1 in (231 mm) tall, 180 mm wide figure. Every row label, end-of-bar label and key is kept inside
# the 180 mm canvas, so the tight-bbox crop on save does not widen the figure: each two-column row has its
# own left/right margin (room for its row labels and right-hand annotations) and a 1.2 in column gap
# (the left panel's right-hand labels plus the right panel's row labels). A (no axes, a schematic) takes
# the whole left column including the label margin, as its standalone slot did.
H7 = 9.10
ROWS7 = {  # (axes bottom, axes top) in inches
    # four readers (2026-11-02 export): E and G carry eight rows each, so their axes are 1.5 in tall
    # (about 13 pt per row: a thin segment's count in the gap between bars clears its neighbours' counts)
    "AB": (7.15, 8.60),    # B key above the axes, two-line x labels below
    "CD": (5.47, 6.45),    # C|D bars; one shared outcome-class key underneath (same key, same classes)
    "key": (4.80, 5.04),
    "EF": (2.74, 4.24),    # F's sub-axis titles sit above the axes top
    "Elegend": (2.18, 2.30),
    "G": (0.48, 1.98),
}
COLS7 = {  # (left margin, right margin, column gap, right-column share of the axes width) in inches
    "AB": (0.25, 0.66, 1.2, 0.5),   # only the right column is used (B); left column = A's own axes
    "CD": (0.66, 0.60, 1.2, 0.5),
    # E's row labels need 1.24 in on the left and the F time sub-axis title runs past its axes on the
    # right. E ≈ 2.03 in (narrower and E's thin-segment counts "3"/"3" of Reader 2 collide), F ≈ 2.4 in as
    # in the v2 composite (narrower and F's sub-axis titles run into each other); gap = E's "median" labels
    # plus F's row labels
    "EF": (1.24, 0.47, 0.95, 0.5425),
}


def _two_cols(fig, row: str, R: dict | None = None, H: float = H7):
    lm, rm, gap, share = COLS7[row]
    w = (WIDTH_IN - lm - rm - gap) / 2  # matplotlib's wspace is relative to the mean axes width
    b, t = (R or ROWS7)[row]
    return fig.add_gridspec(1, 2, left=lm / WIDTH_IN, right=1 - rm / WIDTH_IN, wspace=gap / w,
                            width_ratios=[1 - share, share], bottom=b / H, top=t / H)


def compose7(rows, T, r, b_variant: str, e_variant: str, dodge: float):
    # revision: G and E each grow by two 0.1875 in rows per reader beyond four; everything above moves up
    x = paired_extra_in(len(rows))
    up = {"AB": 2 * x, "CD": 2 * x, "key": 2 * x, "EF": (x, 2 * x), "Elegend": x, "G": (0, x)}
    R = {k: ((ROWS7[k][0] + v[0], ROWS7[k][1] + v[1]) if isinstance(v, tuple) else (ROWS7[k][0] + v, ROWS7[k][1] + v))
         for k, v in up.items()}
    H = H7 + 2 * x
    fig = plt.figure(figsize=(WIDTH_IN, H))
    f = lambda y: y / H  # noqa: E731
    gsAB, gsCD, gsEF = _two_cols(fig, "AB", R, H), _two_cols(fig, "CD", R, H), _two_cols(fig, "EF", R, H)
    axes, srcs = {}, {}
    a_lo, a_hi = R["AB"][0] - 0.38, R["AB"][1] + 0.15  # below the row letter, above the C letter
    axes["A"] = fig.add_axes([0.012, f(a_lo), 0.455, f(a_hi) - f(a_lo)])
    panel_a(axes["A"], rows, r)
    for patch in axes["A"].patches:  # the round-box pad reaches past x = 0; do not clip the first box's left edge
        patch.set_clip_on(False)
    axes["B"], srcs["B"] = draw_b(gsAB[0, 1], fig, rows, T, r, b_variant, dodge)
    axes["C"] = fig.add_subplot(gsCD[0, 0])
    srcs["C"] = panel_c(axes["C"], rows, T, r, "A→B")
    axes["D"] = fig.add_subplot(gsCD[0, 1])
    srcs["D"] = panel_c(axes["D"], rows, T, r, "B→C")
    axkey = fig.add_axes([0.03, f(R["key"][0]), 0.94, f(R["key"][1]) - f(R["key"][0])])
    handles = [Patch(facecolor=CLASS_COL[c], edgecolor="0.4", linewidth=0.3, label=CLASS_LABEL[c]) for c in CLASS_ORDER]
    axkey.legend(handles=handles, loc="center", fontsize=6, frameon=False, ncol=4, handlelength=1.0, columnspacing=1.4,
                 borderaxespad=0, title="key to C and D", title_fontsize=6)
    axkey.axis("off")
    axes["E"] = fig.add_subplot(gsEF[0, 0])
    srcs["E"] = panel_e(axes["E"], rows, T, r, variant=e_variant)
    pe = axes["E"].get_position()
    axleg = fig.add_axes([(pe.x0 + pe.x1) / 2 - 0.3, f(R["Elegend"][0]), 0.6, f(R["Elegend"][1]) - f(R["Elegend"][0])])
    legend_likert(axleg, variant=e_variant)
    axes["F"], srcs["F"] = draw_f(gsEF[0, 1], fig, rows, T, r)
    # F's stage key sits on top of the Reader 1/2 time markers at this row height; in the composite it is
    # moved under the three F sub-axes as one line (same handles and labels as the standalone panel).
    ax_t = [a for a in fig.axes if a.get_legend() is not None and a is not axes["F"]][-1]
    h, l = ax_t.get_legend_handles_labels()
    ax_t.get_legend().remove()
    pf = axes["F"].get_position()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=((pf.x0 + ax_t.get_position().x1) / 2, f(R["Elegend"][1] + 0.08)),
               ncol=len(l), fontsize=6, frameon=False, handlelength=1.6, columnspacing=1.4, borderaxespad=0)
    axes["G"], srcs["G"] = draw_g_wide(fig, (0.30, f(R["G"][0]), 0.60, f(R["G"][1])), rows, T, r)

    fig.canvas.draw()
    x_left, x_right = 0.004, 0.505  # letters in two fixed columns, left of each panel's row labels
    y_top = {"AB": f(R["AB"][1]) + 0.028, "CD": f(R["CD"][1]) + 0.018, "EF": f(R["EF"][1]) + 0.040,
             "G": f(R["G"][1]) + 0.010}
    for letter, x, y in (("A", x_left, y_top["AB"]), ("B", x_right, y_top["AB"]), ("C", x_left, y_top["CD"]),
                         ("D", x_right, y_top["CD"]), ("E", x_left, y_top["EF"]), ("F", x_right, y_top["EF"]),
                         ("G", x_left, y_top["G"])):
        fig.text(x, y, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")
    return fig, srcs, H


def _cols(fig, H: float, lm: float, rm: float, gap: float, share: float, b: float, t: float):
    """Two-column gridspec in inches (left/right margin, column gap, right-column share), rows b..t."""
    w = (WIDTH_IN - lm - rm - gap) / 2  # matplotlib's wspace is relative to the mean axes width
    return fig.add_gridspec(1, 2, left=lm / WIDTH_IN, right=1 - rm / WIDTH_IN, wspace=gap / w,
                            width_ratios=[1 - share, share], bottom=b / H, top=t / H)


def _wrap_xlabel(ax, at: str) -> None:
    """Break a long x label onto two lines at the first occurrence of `at` (same words)."""
    lab = ax.get_xlabel()
    if at in lab:
        k = lab.index(at)
        ax.set_xlabel(lab[:k].rstrip() + "\n" + lab[k:].lstrip(), fontsize=6.5)


# Four-panel main composite (2026-11-06, revision): A|B as in the seven-panel figure, C|E below,
# each with its own key underneath. Inches from the bottom of a 180 mm wide canvas.
H4 = 5.40
ROWS4 = {
    "AB": (3.50, 4.95),   # same axes height as the seven-panel A|B row; B's key above the axes
    "CE": (1.20, 2.70),   # E carries eight rows (two per reader): 1.5 in as in the seven-panel figure
    "key": (0.02, 0.50),  # C's class key under C, E's rating key under E
}
# CE: C's row labels need ≈0.55 in on the left; the gap holds C's "ref. label lost n" plus E's two-level
# row labels (≈1.24 in, as in the seven-panel E|F row); E's "median" labels sit in the right margin.
# Face letters (coordinator correction 2026-11-06): working panel E is printed as D, so usefulness is Fig. 8D;
# internal names, CLI and standalone panels keep A, B, C, E.
MAIN_LETTER = {"A": "A", "B": "B", "C": "C", "E": "D"}
COLS4 = {"AB": (0.25, 0.66, 1.2, 0.5), "CE": (0.55, 0.47, 2.02, 0.5)}


def compose4(rows, T, r, b_variant: str, e_variant: str, dodge: float):
    # revision: E's axes (and so the C|E row) grow by two 0.1875 in rows per reader beyond four; the A|B row
    # moves up by the same amount and the canvas grows by it (keys and all spacing unchanged)
    x = paired_extra_in(len(rows))
    H = H4 + x
    R = {"AB": (ROWS4["AB"][0] + x, ROWS4["AB"][1] + x), "CE": (ROWS4["CE"][0], ROWS4["CE"][1] + x), "key": ROWS4["key"]}
    fig = plt.figure(figsize=(WIDTH_IN, H))
    f = lambda y: y / H  # noqa: E731
    gsAB = _cols(fig, H, *COLS4["AB"], *R["AB"])
    gsCE = _cols(fig, H, *COLS4["CE"], *R["CE"])
    axes, srcs = {}, {}
    a_lo, a_hi = R["AB"][0] - 0.38, R["AB"][1] + 0.15
    axes["A"] = fig.add_axes([0.012, f(a_lo), 0.455, f(a_hi) - f(a_lo)])
    panel_a(axes["A"], rows, r)
    for patch in axes["A"].patches:  # the round-box pad reaches past x = 0
        patch.set_clip_on(False)
    axes["B"], srcs["B"] = draw_b(gsAB[0, 1], fig, rows, T, r, b_variant, dodge)
    axes["C"] = fig.add_subplot(gsCE[0, 0])
    srcs["C"] = panel_c(axes["C"], rows, T, r, "A→B")
    _wrap_xlabel(axes["C"], "after")
    axes["E"] = fig.add_subplot(gsCE[0, 1])
    srcs["E"] = panel_e(axes["E"], rows, T, r, variant=e_variant)
    _wrap_xlabel(axes["E"], "(")
    fig.canvas.draw()
    kb, kt = f(R["key"][0]), f(R["key"][1])
    pc, pe = axes["C"].get_position(), axes["E"].get_position()
    axk = fig.add_axes([0.01, kb, pc.x1 + 0.06 - 0.01, kt - kb])
    handles = [Patch(facecolor=CLASS_COL[c], edgecolor="0.4", linewidth=0.3, label=CLASS_LABEL[c]) for c in CLASS_ORDER]
    axk.legend(handles=handles, loc="center", fontsize=6, frameon=False, ncol=2, handlelength=1.0, columnspacing=1.0,
               borderaxespad=0, labelspacing=0.35)
    axk.axis("off")
    axl = fig.add_axes([pe.x0 - 0.12, kb, 0.99 - (pe.x0 - 0.12), kt - kb])
    hl = [Patch(facecolor=LIK_COL[e_variant][lv], edgecolor="0.4", linewidth=0.3, label=LIK_LAB[e_variant][lv]) for lv in LIKERT]
    axl.legend(handles=hl, loc="center", fontsize=6, frameon=False, ncol=3, handlelength=1.0, columnspacing=1.0,
               borderaxespad=0, labelspacing=0.35)
    axl.axis("off")
    x_left, x_right = 0.004, 0.505
    x_e = pe.x0 - 1.30 / WIDTH_IN  # left of E's two-level row labels
    y_ab, y_ce = f(R["AB"][1]) + 0.04, f(R["CE"][1]) + 0.035
    for letter, x, y in (("A", x_left, y_ab), ("B", x_right, y_ab), ("C", x_left, y_ce), ("E", x_e, y_ce)):
        fig.text(x, y, MAIN_LETTER[letter], fontsize=10, fontweight="bold", va="bottom", ha="left")
    layout = ("A|B on the top row, C|E on the bottom row; C's outcome-class key under C (two columns), "
              "E's rating key under E (three columns); face letters A, B, C, D (working panel E printed as D, "
              "source_data_panelD.csv)")
    return fig, srcs, layout, H


# Supplementary composite (provisional Supplementary Fig. S14, S-RS-F1): D|F on top, G across the full
# width below (as in the seven-panel figure); face letters A (= D), B (= F), C (= G).
HS = 5.40
ROWSS = {
    "DF": (3.30, 4.70),     # D's four bars / F's four rows at 1.4 in; F's sub-axis titles above the axes
    "Dkey": (2.22, 2.67),   # D's class key under D (two columns); F's stage key under F at the same height
    "G": (0.42, 1.92),
}
COLSS = {"DF": (0.66, 0.47, 1.3, 0.5425)}  # revision: gap 1.2 → 1.3 in so A's wider "away / dropped" end labels clear B's reader labels
SUPP_LETTER = {"D": "A", "F": "B", "G": "C"}


def compose_supp(rows, T, r):
    # revision: G's axes grow by two 0.1875 in rows per reader beyond four; D|F and its keys move up by the
    # same amount and the canvas grows by it
    x = paired_extra_in(len(rows))
    H = HS + x
    R = {"DF": (ROWSS["DF"][0] + x, ROWSS["DF"][1] + x), "Dkey": (ROWSS["Dkey"][0] + x, ROWSS["Dkey"][1] + x),
         "G": (ROWSS["G"][0], ROWSS["G"][1] + x)}
    fig = plt.figure(figsize=(WIDTH_IN, H))
    f = lambda y: y / H  # noqa: E731
    gs = _cols(fig, H, *COLSS["DF"], *R["DF"])
    axes, srcs = {}, {}
    axes["D"] = fig.add_subplot(gs[0, 0])
    srcs["D"] = panel_c(axes["D"], rows, T, r, "B→C")
    _wrap_xlabel(axes["D"], "after")
    axes["F"], srcs["F"] = draw_f(gs[0, 1], fig, rows, T, r)
    ax_t = [a for a in fig.axes if a.get_legend() is not None and a is not axes["F"]][-1]
    h, l = ax_t.get_legend_handles_labels()
    ax_t.get_legend().remove()
    pf, pd_ = axes["F"].get_position(), axes["D"].get_position()
    kb, kt = f(R["Dkey"][0]), f(R["Dkey"][1])
    fig.legend(h, l, loc="upper center", bbox_to_anchor=((pf.x0 + ax_t.get_position().x1) / 2, kt + 0.01),
               ncol=len(l), fontsize=6, frameon=False, handlelength=1.6, columnspacing=1.4, borderaxespad=0)
    axk = fig.add_axes([0.01, kb, pd_.x1 + 0.08 - 0.01, kt - kb])
    handles = [Patch(facecolor=CLASS_COL[c], edgecolor="0.4", linewidth=0.3, label=CLASS_LABEL[c]) for c in CLASS_ORDER]
    axk.legend(handles=handles, loc="center", fontsize=6, frameon=False, ncol=2, handlelength=1.0, columnspacing=1.0,
               borderaxespad=0, labelspacing=0.35)
    axk.axis("off")
    axes["G"], srcs["G"] = draw_g_wide(fig, (0.30, f(R["G"][0]), 0.60, f(R["G"][1])), rows, T, r)
    fig.canvas.draw()
    x_left, x_right = 0.004, 0.505
    y_top = {"DF": f(R["DF"][1]) + 0.075, "G": f(R["G"][1]) + 0.012}
    for letter, x, y in (("D", x_left, y_top["DF"]), ("F", x_right, y_top["DF"]), ("G", x_left, y_top["G"])):
        fig.text(x, y, SUPP_LETTER[letter], fontsize=10, fontweight="bold", va="bottom", ha="left")
    layout = ("D|F on the top row (D's outcome-class key under D, F's stage key under F), G across the full width "
              "below with its key at the right; face letters A (Fig. 8 working panel D), B (F), C (G)")
    return fig, srcs, layout, H


def compose(rows, T, r, b_variant: str, e_variant: str, dodge: float,
            panels: tuple[str, ...] = ("A", "B", "C", "E")):
    """Composite Figure 8. Default (2026-11-06): four panels A, B, C, E (compose4). The seven-panel
    A–G layout of 2026-11-02 (compose7) stays reachable; any other ordered list: ceil(n/2) two-column
    rows in reading order. Returns (fig, source frames, layout, canvas height in inches)."""
    panels = tuple(panels)
    if panels == ("A", "B", "C", "E"):
        return compose4(rows, T, r, b_variant, e_variant, dodge)
    if panels == ("A", "B", "C", "D", "E", "F", "G"):
        fig, srcs, H = compose7(rows, T, r, b_variant, e_variant, dodge)
        return fig, srcs, ("A|B, C|D with one shared outcome-class key, E|F, then G across the full width at reduced "
                           "height with its key at the right"), H
    nrow = int(np.ceil(len(panels) / 2))
    H = 8.2 if nrow <= 3 else 8.8
    fig = plt.figure(figsize=(WIDTH_IN, H))
    gs = fig.add_gridspec(nrow, 2, wspace=0.55, hspace=0.62, left=0.085, right=0.95, top=0.965, bottom=0.06)
    axes, srcs = {}, {}
    for i, letter in enumerate(panels):
        axes[letter], srcs[letter] = _draw_slot(letter, gs[i // 2, i % 2], fig, rows, T, r, b_variant, e_variant, dodge)
    fig.canvas.draw()
    for letter in panels:
        pos = axes[letter].get_position()
        fig.text(pos.x0 - 0.075, pos.y1 + 0.006, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")
    return fig, srcs, f"{nrow} two-column rows in reading order", H


def key_overlaps(fig) -> list[list[str]]:
    """The shared legibility check skips legend texts on axis-off key axes and figure legends; this adds
    them: every key label against every other drawn text (and the other keys)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    keys = [t for a in fig.axes if a.get_legend() is not None for t in a.get_legend().get_texts()]
    keys += [t for lg in fig.legends for t in lg.get_texts()]
    others = [t for t in visible_texts(fig) if t not in keys]
    def bb(t):
        b = t.get_window_extent(rend)
        return (b.x0, b.y0, b.x1, b.y1) if b.width > 0 and b.height > 0 else None
    out = []
    kb = [(t, bb(t)) for t in keys]
    ob = [(t, bb(t)) for t in others]
    for i, (t, a) in enumerate(kb):
        if a is None:
            continue
        for u, b in ob + kb[i + 1:]:
            if b is not None and u is not t and a[0] < b[2] - 0.5 and b[0] < a[2] - 0.5 and a[1] < b[3] - 0.5 and b[1] < a[3] - 0.5:
                out.append([t.get_text(), u.get_text()])
    return out


MAIN_STEM, SUPP_STEM = "Figure_8_reader_study", "Figure_S14_reader_study_secondary"


def write_composites(items: list[dict], inputs, note: str) -> None:
    """Save every composite of the run (PNG 300 dpi, PDF, source data, legibility check) into the fig8
    directory and write one SHA256SUMS.txt / PROVENANCE.md covering all of them."""
    d = fig_dir(FIGURE)
    stems = {it["stem"] for it in items}
    for other, pattern in ((MAIN_STEM, "source_data_panel*.csv"), (SUPP_STEM, "source_data_S14_panel*.csv")):
        if other not in stems and ((d / f"{other}.png").exists() or any(d.glob(pattern))):
            raise SystemExit(f"{other} outputs exist in {d}; run --compose and --compose-supp together so the "
                             "directory's SHA256SUMS.txt / PROVENANCE.md cover both")
    written, legs, notes = [], [], []
    for it in items:
        fig, stem, tag = it["fig"], it["stem"], it["tag"]
        leg = legibility(fig, d, tag)
        ko = key_overlaps(fig)
        if ko:  # recorded in the same JSON; a key collision fails the check like any other overlap
            leg["key_text_overlaps"], leg["pass"] = ko, False
        else:
            leg["key_text_overlaps"] = []
        (d / f"legibility_check_{tag}.json").write_text(json.dumps(leg, indent=1))
        if not leg["pass"]:
            print(json.dumps({k: v for k, v in leg.items() if k != "texts"}, indent=1))
        assert leg["pass"], f"legibility check failed for {stem}"
        png, pdf = d / f"{stem}.png", d / f"{stem}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf, bbox_inches="tight", facecolor="white", metadata={"CreationDate": None})
        written += [png, pdf]
        for k, v in it["srcs"].items():
            if v is not None:
                pth = d / f"{it['src_prefix']}{it.get('rename', {}).get(k, k)}.csv"
                v.to_csv(pth, index=False)
                written.append(pth)
        legs.append(d / f"legibility_check_{tag}.json")
        notes.append(f"**{stem}** — {it['what']}: {it['layout']}; canvas {WIDTH_IN * 25.4:.0f} mm x {it['H'] * 25.4:.1f} mm "
                     f"before tight-bbox cropping; legibility {leg['n_texts']} texts, pass.")
        print(f"wrote {png}")
        print(json.dumps({"stem": stem, **{k: v for k, v in leg.items() if k != "texts"}}))
        plt.close(fig)
    write_provenance(d, written, inputs, "\n\n".join(notes) + "\n\n" + note, Path(sys.argv[0]),
                     title=f"{FIGURE} / " + " + ".join(it["stem"] for it in items))
    with open(d / "SHA256SUMS.txt", "a") as fh:
        for p in legs:
            fh.write(f"{sha256(p)}  {p.name}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default=str(prosper_out() / ANALYSIS_DEFAULT))
    ap.add_argument("--readers", default=READERS_DEFAULT)
    ap.add_argument("--panels", default="A,B,C,D,E,F,G",
                    help="standalone panels to render (comma list, or 'none'); C/D = outcome classes after the search "
                         "results / after the LLM summary, G = confidence change by correctness, S_hit10 = supplementary "
                         "split by the 10 slides shown (not in the default)")
    ap.add_argument("--b-variants", default="shapes,shapes0,shapes_diff,rows,small",
                    help="panel-B variants to render standalone: v2,shapes,shapes0,shapes_diff,rows,small")
    ap.add_argument("--e-variants", default="v3", help="panel-E variants to render standalone: v2,v3")
    ap.add_argument("--dodge", type=float, default=0.05, help="per-reader x offset in the shapes variant (0 = none)")
    ap.add_argument("--compose", action="store_true", help="also write the composite Figure_8 (first listed B/E variant)")
    ap.add_argument("--compose-panels", default="A,B,C,E",
                    help="ordered panel list for the composite (Haoran 2026-11-06: A,B,C,E; seven-panel A,B,C,D,E,F,G)")
    ap.add_argument("--compose-supp", action="store_true",
                    help="also write the supplementary composite (provisional Supplementary Fig. S14, S-RS-F1): "
                         "working panels D, F, G lettered A, B, C")
    ap.add_argument("--out-root", default=None,
                    help="write under <out-root>/fig8_reader_study/ instead of $PROSPER_OUT/figures_20261021/ (pass 041)")
    ap.add_argument("--search-alone", default=None,
                    help="output directory of scripts/reader_study/search_vs_reader.py; adds the retrieved labels "
                         "without a reader to panel B `rows` (plan 003 amendment 021; pass 050)")
    args = ap.parse_args()
    global OUT_ROOT, SEARCH_ALONE
    OUT_ROOT = Path(args.out_root) if args.out_root else None
    analysis = Path(args.analysis)
    readers = [x.strip() for x in args.readers.split(",") if x.strip()]
    style()
    r, T, run, tiers = load(analysis, readers)
    rows = reader_rows(readers, tiers, r)
    band = None
    if args.search_alone:
        SEARCH_ALONE, band = load_search_alone(Path(args.search_alone), r)

    facts = []
    for dd in rows:
        g = r[r.reader_id == dd["reader_id"]]
        e2 = T["T2_primary_endpoints"][T["T2_primary_endpoints"].reader_id == dd["reader_id"]].set_index("endpoint")
        diff = e2.loc["E2 paired difference fav_correct B − A"]
        facts.append(
            f"{dd['label']} = {dd['reader_id']} ({TIER_LABEL[dd['tier']]}, stored tier code {dd['tier']}, n = {dd['n']}): favored correct "
            f"{int(g.fav_correct_A.sum())}/{int(g.fav_correct_B.sum())}/{int(g.fav_correct_C.sum())}; Step 2 − Step 1 "
            f"{diff.estimate:.4f} (bootstrap 95 % {diff.boot95_lo:.3f}–{diff.boot95_hi:.3f}); "
            f"{int(e2.loc['E2 discordant A-wrong→B-right (count)','k'])} wrong→right, "
            f"{int(e2.loc['E2 discordant A-right→B-wrong (count)','k'])} right→wrong; mislead {int((g.class_AB=='mislead').sum())}; "
            f"true group among the 10 shown {int(g.hit10.astype(bool).sum())}/{dd['n']} (hit@5 {int(g.hit5.astype(bool).sum())}); engaged readings (reader-initiated retrieved-slide or ROI-hit opening, or ROI selection) {int(g.engaged_B.astype(bool).sum())}/{dd['n']}; "
            f"median s per step {g.sec_A.median():.0f}/{g.sec_B.median():.0f}/{g.sec_C.median():.0f}.")
    n_cases = r.case_id.nunique()
    curator = next((d["label"] for d in rows if d["reader_id"] == READER_CONFIG["curator"]), "a non-reader co-author")
    note = (
        "Amendment 004 design (main figure A, B, C, E and supplementary D, F, G since 2026-11-06; A–G 2026-11-02), per-reader rows by expertise tier, no pooled estimate. All values from the "
        f"{len(rows)}-reader run of the {'committed ' if run.get('scripts_reader_study_clean') == 'yes' else ''}pipeline (analyze.py commit {run.get('commit', '?')[:7]}"
        f"{'' if run.get('scripts_reader_study_clean') == 'yes' else '; scripts/reader_study not clean at run time, see the run record'}) in analysis directory "
        f"{analysis.name} (readers with fewer than {n_cases} cases read the part of the fixed assignment reached, not a sample; "
        f"their numbers are not comparable to a complete reader's {n_cases}; the paired-set comparison is T10 / supplementary S-RS7). Intervals: "
        f"whole-case bootstrap percentile 95 %, {int(run['n_draws']):,} draws, seed {run['seed']}. Facts for the legend (no prose drafted): "
        + " ".join(facts)
        + f" Caveats to carry: common case list, reading order chosen by the reader (not counterbalanced); co-author readers; {curator} ({READER_CONFIG['curator']}) curated the STV cohort "
        "and supplied its metadata; Step 3 follows Step 2 (summary effect not isolated); reference key not independently "
        "vetted (QC gate dropped); summaries displayed scores as probabilities; reading time within the instrument is not "
        "time-to-diagnosis; outcome classes per plan 002 §3.3 with confirm split per 003 §B1. Descriptive; no test."
    )
    inputs = [analysis / "readings.csv", analysis / "analyze_run.json"] + [analysis / "tables" / f"{k}.csv" for k in TABLES]
    if SEARCH_ALONE:
        inputs.append(Path(args.search_alone) / "V1_search_alone.csv")
        note += (f" Panel B also shows the retrieved labels without a reader on the same {SEARCH_ALONE['n']} cases (exploratory, "
                 f"added after the reader results; plan 003 amendment 021): band from {SEARCH_ALONE['top1']} (nearest reference "
                 f"slide in the reference group) to {SEARCH_ALONE['plurality5']} (plurality vote over the five nearest, count ties "
                 f"to the nearest-ranked label), dotted line at {SEARCH_ALONE['hit5']} (reference group among the five nearest); "
                 "left axis labelled in number of cases; source_data_panelB_search_band.csv.")
    panels = [] if args.panels.lower() == "none" else [(p.strip().upper() if len(p.strip()) <= 2 else p.strip()) for p in args.panels.split(",") if p.strip()]
    b_variants = [v.strip() for v in args.b_variants.split(",") if v.strip()]
    e_variants = [v.strip() for v in args.e_variants.split(",") if v.strip()]
    for letter in panels:
        if letter == "B":
            for v in b_variants:
                render_panel("B", v, rows, T, r, inputs, args.dodge, note)
        elif letter == "E":
            for v in e_variants:
                render_panel("E", v, rows, T, r, inputs, args.dodge, note)
        else:
            render_panel(letter, None, rows, T, r, inputs, args.dodge, note)
    items = []
    if args.compose:
        cpanels = tuple(p.strip().upper() for p in args.compose_panels.split(",") if p.strip())
        fig, srcs, layout, H = compose(rows, T, r, b_variants[0], e_variants[0], args.dodge, cpanels)
        if band is not None and "B" in cpanels and b_variants[0] == "rows":
            srcs["B_search_band"] = band
        items.append(dict(fig=fig, srcs=srcs, layout=layout, H=H, stem=MAIN_STEM, tag="8", src_prefix="source_data_panel",
                          rename=MAIN_LETTER if cpanels == ("A", "B", "C", "E") else {},
                          what=f"Fig. 8 composite, panels {','.join(cpanels)} in that order (panel B `{b_variants[0]}`, "
                               f"panel E `{e_variants[0]}`)"))
    if args.compose_supp:
        fig, srcs, layout, H = compose_supp(rows, T, r)
        items.append(dict(fig=fig, srcs=srcs, layout=layout, H=H, stem=SUPP_STEM, tag="S14", src_prefix="source_data_S14_panel",
                          what="provisional Supplementary Fig. S14 (S-RS-F1), Fig. 8 working panels D, F, G "
                               "(source_data_S14_panel{D,F,G}.csv keep the working letters; face letters A, B, C)"))
    if items:
        write_composites(items, inputs, note)

if __name__ == "__main__":
    main()
