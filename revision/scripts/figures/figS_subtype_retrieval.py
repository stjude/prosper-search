"""Supplementary Fig. S-SUB-F1 (provisional ID; revision 2026-11-06): subtype-level retrieval
within a known tumor family.

Plots only numbers that already exist in Supplementary Table S8 (workspace ID S26, version 2 of
2026-11-03: all-tissue COMET arms reported, tumor-filtered COMET arms as a sensitivity block); nothing is
recomputed. Number assigned at packaging by first appearance (cited from the Results subtype paragraph).

Faces (house style of the revision figures: no internal prose, n beside rows, every point with its interval):
  A  one row per task and query cohort — RMS (embryonal vs alveolar) COMET queries, RMS CCDI-MCI queries
     against the COMET RMS library, liver tumors (hepatoblastoma vs hepatocellular carcinoma) COMET
     queries: top-5 subtype plurality agreement with the recorded label, point + 95 % interval;
     filled = all-tissue mosaics (the main-text values), open = tumor-filtered tiles (sensitivity, COMET
     arms only); grey dashed tick = majority-subtype share (share of the larger subtype among the
     queries, i.e. the agreement of always answering the larger subtype). Right-hand columns: slides /
     patients, agreement (95 % CI), majority share.
  B  the same agreement within each recorded subtype (per-subtype recall of the plurality vote), same
     marks, same arms.
Table S9 (workspace ID S28, inventory of the evaluated fine-label tasks) is not drawn: its internal COMET
rows are the tumor-filtered version-1 values (a different representation from panel A's all-tissue rows,
and version-1 bootstrap intervals that differ from Table S8's sensitivity block by up to 0.002), its
KidRare rows have no whole-patient interval, and its molecular/INPC tasks are outside this figure's
subject. It is read only for a distinct-route check of the majority shares and of the CCDI-MCI arm.

Sources (read-only; SHA-256 recorded in PROVENANCE.md):
  $PROSPER_OUT/supp_tables_20261103/A_ladder_failure_finelabel/
    R3.7-T4_S26_S-D1_v2_body_raw.csv        plotted values (hash asserted equal to BUILD_REPORT.md)
    R3.7-T4_S26_S-D1_v2_body_formatted.csv  check: every plotted value equals its 3-d.p. table cell
    R3.7-T4_S26_S-D1_v2.xlsx                check: same cells in the workbook sheet "S26 body"
    BUILD_REPORT.md                         check: no FAIL line; output hashes
  $PROSPER_OUT/supp_tables_20261022/A_ladder_failure_finelabel/R3.7-T5_S28_S-D2_table_verbatim.csv
                                            check: majority-class counts/shares; CCDI-MCI arm
  docs/prosper_revised/prosper_manuscript_revised.md, docs/prosper_revised/prosper_supplementary_legends.md
                                            recorded (not fatal) check: the quoted values appear verbatim

Outputs  $PROSPER_OUT/figures_20261021/figS_subtype_retrieval/
         Figure_S-SUB-F1_subtype_retrieval.{png,pdf}, source_data_panelA.csv, source_data_panelB.csv,
         value_check.json, legibility_check_S-SUB-F1.json, SHA256SUMS.txt, PROVENANCE.md

Run from the repository root:  python3 scripts/figures/figS_subtype_retrieval.py
Deterministic: re-running reproduces the PNG, PDF and CSV hashes (PROVENANCE.md carries a timestamp).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import transforms
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import AGG_COLORS, GREY, REPO, out_dir, prosper_out, sha256, style, write_provenance  # noqa: E402
from fig8_reader_study import key_overlaps  # noqa: E402
from figS1_S2_neighbor_ladder import legibility  # noqa: E402

FIGURE = "figS_subtype_retrieval"
FIG_ID = "S-SUB-F1"
STEM = "Figure_S-SUB-F1_subtype_retrieval"
S8_DIR = prosper_out() / "supp_tables_20261103/A_ladder_failure_finelabel"
S8_RAW = S8_DIR / "R3.7-T4_S26_S-D1_v2_body_raw.csv"
S8_FMT = S8_DIR / "R3.7-T4_S26_S-D1_v2_body_formatted.csv"
S8_XLSX = S8_DIR / "R3.7-T4_S26_S-D1_v2.xlsx"
S8_REPORT = S8_DIR / "BUILD_REPORT.md"
S9_TABLE = prosper_out() / "supp_tables_20261022/A_ladder_failure_finelabel/R3.7-T5_S28_S-D2_table_verbatim.csv"
MANUSCRIPT = REPO / "docs/prosper_revised/prosper_manuscript_revised.md"
LEGENDS = REPO / "docs/prosper_revised/prosper_supplementary_legends.md"

WIDTH_IN = 180 / 25.4
BLUE = AGG_COLORS["Optimal Transport"]  # the configuration is UNI2–Optimal Transport
INK, INK2 = "0.12", "0.42"  # primary text; secondary text (sensitivity values)
XLIM, XTICKS = (0.5, 1.015), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # a little room past 1.0 so an interval ending at 1.000 is not cut
DODGE = 0.2  # rows with two arms: all-tissue above, tumor-filtered below
MINUS, NDASH = "−", "–"

# arm -> (task label, query-cohort label, subtype classes in table order, majority class)
ARMS = {
    "s61_int_2c": ("RMS: embryonal vs alveolar", "COMET queries", ["ERMS", "ARMS"], "ERMS"),
    "s61_ext_2c": ("RMS: embryonal vs alveolar", "CCDI-MCI queries, COMET library", ["ERMS", "ARMS"], "ERMS"),
    "s64": ("Liver tumors: HB vs HCC", "COMET queries", ["HB", "HCC"], "HB"),
}
ARM_ORDER = ["s61_int_2c", "s61_ext_2c", "s64"]
GROUP_LABEL = {"s61_int_2c": "RMS\nCOMET", "s61_ext_2c": "RMS\nCCDI-MCI queries,\nCOMET library", "s64": "Liver tumors\nCOMET"}
SUBTYPE_LABEL = {"ERMS": "Embryonal", "ARMS": "Alveolar", "HB": "Hepatoblastoma", "HCC": "Hepatocellular carcinoma"}
TABLE_COMPARISON = {"s61_int_2c": "ERMS vs ARMS, COMET", "s61_ext_2c": "ERMS vs ARMS, CCDI-MCI → COMET", "s64": "HB vs HCC, COMET"}

# Named acceptance values (3 d.p.): main-text Results paragraph (all-tissue COMET arms), the Table S8/S9 notes in
# the supplementary legends (CCDI-MCI arm, tumor-filtered sensitivity) and the Table S8 v2 legend (majority shares).
EXPECT_ALL = {
    ("reported", "s61_int_2c"): (0.847, 0.807, 0.885, 373, 0.606),
    ("reported", "s61_ext_2c"): (0.880, 0.837, 0.920, 251, 0.677),
    ("reported", "s64"): (0.914, 0.855, 0.963, 139, 0.683),
    ("sensitivity", "s61_int_2c"): (0.887, 0.855, 0.920, 373, 0.606),
    ("sensitivity", "s64"): (0.935, 0.884, 0.977, 139, 0.683),
}

CHECKS: list[dict] = []


def check(name: str, ok: bool, detail: str = "", level: str = "fatal") -> None:
    """level fatal: assert; warn: recorded as WARN (prose the analyst may edit); info: recorded as INFO when the
    values differ for a documented reason (not an error)."""
    CHECKS.append(dict(check=name, result="PASS" if ok else {"fatal": "FAIL", "warn": "WARN", "info": "INFO"}[level],
                       detail=detail))
    if level == "fatal":
        assert ok, f"{name}: {detail}"
    elif not ok:
        print(f"{level.upper()} {name}: {detail}")


def r3(x) -> str:
    return f"{float(x):.3f}"


def load():
    raw = pd.read_csv(S8_RAW)
    fmt = pd.read_csv(S8_FMT, dtype=str, keep_default_na=False)
    xl = pd.read_excel(S8_XLSX, sheet_name="S26 body", dtype=str).fillna("")
    s9 = pd.read_csv(S9_TABLE, dtype=str)
    report = S8_REPORT.read_text(encoding="utf-8")

    # provenance of the table build
    check("BUILD_REPORT has no FAIL line", "- FAIL" not in report)
    for p in (S8_RAW, S8_FMT, S8_XLSX):
        m = re.search(rf"`{re.escape(p.name)}` sha256 ([0-9a-f]{{64}})", report)
        check(f"{p.name} sha256 = BUILD_REPORT", bool(m) and m.group(1) == sha256(p), sha256(p))

    # structure
    check("raw table: 20 rows", len(raw) == 20, str(len(raw)))
    rep = raw[raw.block == "reported"]
    sen = raw[raw.block == "sensitivity"]
    check("reported block = all-tissue, three arms x (ALL, MACRO, 2 subtypes)",
          set(rep.representation) == {"all_tissue"} and len(rep) == 12 and sorted(rep.arm.unique()) == sorted(ARM_ORDER))
    check("sensitivity block = tumour-only, COMET arms only",
          set(sen.representation) == {"tumour_only"} and len(sen) == 8 and sorted(sen.arm.unique()) == ["s61_int_2c", "s64"])
    check("unit = patient (whole-patient resampling) in every row", set(raw.unit) == {"patient"})
    for (blk, arm), g in raw.groupby(["block", "arm"]):
        _, _, cls, maj = ARMS[arm]
        a = g[g["class"] == "ALL"].iloc[0]
        sub = g[g["class"].isin(cls)]
        check(f"{blk}/{arm}: subtype slides sum to ALL", int(sub.n_queries.sum()) == int(a.n_queries))
        check(f"{blk}/{arm}: subtype patients sum to ALL (no patient carries both subtypes)",
              int(sub.n_patients.sum()) == int(a.n_patients))
        # distinct route for every point estimate: the confusion counts of the ALL row
        conf = {k.strip().split()[0]: tuple(int(v) for v in k.strip().split()[1].split("/"))
                for k in str(a.confusion).split(";")}
        check(f"{blk}/{arm}: confusion row sums = subtype n",
              all(sum(conf[c]) == int(g[g["class"] == c].n_queries.iloc[0]) for c in cls), str(conf))
        correct = {cls[0]: conf[cls[0]][0], cls[1]: conf[cls[1]][1]}
        check(f"{blk}/{arm}: ALL point = correct/n from confusion",
              abs(sum(correct.values()) / int(a.n_queries) - a.plurality5_recall) < 1e-12,
              f"{sum(correct.values())}/{int(a.n_queries)}")
        for c in cls:
            r = g[g["class"] == c].iloc[0]
            check(f"{blk}/{arm}/{c}: point = correct/n from confusion",
                  abs(correct[c] / int(r.n_queries) - r.plurality5_recall) < 1e-12, f"{correct[c]}/{int(r.n_queries)}")
        for _, r in g.iterrows():
            if not pd.isna(r.plurality5_recall_lo):
                check(f"{blk}/{arm}/{r['class']}: lo <= point <= hi",
                      r.plurality5_recall_lo <= r.plurality5_recall <= r.plurality5_recall_hi)
    return raw, fmt, xl, s9


def rows_for_panels(raw: pd.DataFrame):
    a_rows, b_rows = [], []
    for arm in ARM_ORDER:
        task, cohort, cls, maj = ARMS[arm]
        for blk in ("reported", "sensitivity"):
            g = raw[(raw.block == blk) & (raw.arm == arm)]
            if g.empty:
                continue
            a = g[g["class"] == "ALL"].iloc[0]
            m = g[g["class"] == maj].iloc[0]
            share = int(m.n_queries) / int(a.n_queries)
            a_rows.append(dict(panel="A", block=blk, representation=a.representation, arm=arm, task=task, cohort=cohort,
                               row="ALL", n_slides=int(a.n_queries), n_patients=int(a.n_patients),
                               n_reference_slides=int(a.n_reference_slides), n_reference_patients=int(a.n_reference_patients),
                               point=a.plurality5_recall, lo=a.plurality5_recall_lo, hi=a.plurality5_recall_hi,
                               majority_class=maj, majority_n=int(m.n_queries), majority_share=share,
                               source=f"{S8_RAW.name}: block={blk}, arm={arm}, class=ALL; majority = class {maj} n_queries / ALL n_queries"))
            for c in cls:
                r = g[g["class"] == c].iloc[0]
                b_rows.append(dict(panel="B", block=blk, representation=r.representation, arm=arm, task=task, cohort=cohort,
                                   row=c, subtype=SUBTYPE_LABEL[c], n_slides=int(r.n_queries), n_patients=int(r.n_patients),
                                   point=r.plurality5_recall, lo=r.plurality5_recall_lo, hi=r.plurality5_recall_hi,
                                   source=f"{S8_RAW.name}: block={blk}, arm={arm}, class={c}"))
    return pd.DataFrame(a_rows), pd.DataFrame(b_rows)


def value_checks(A: pd.DataFrame, B: pd.DataFrame, fmt: pd.DataFrame, xl: pd.DataFrame, s9: pd.DataFrame) -> None:
    # named acceptance values
    for (blk, arm), (p, lo, hi, n, share) in EXPECT_ALL.items():
        r = A[(A.block == blk) & (A.arm == arm)].iloc[0]
        got = (r3(r.point), r3(r.lo), r3(r.hi), int(r.n_slides), r3(r.majority_share))
        check(f"named value {blk}/{arm}", got == (r3(p), r3(lo), r3(hi), n, r3(share)), str(got))
    # every plotted value equals its formatted table cell (CSV and workbook)
    block_name = {"reported": "All-tissue mosaics (reported)", "sensitivity": "Sensitivity: tumour-only representations"}
    for name, tab in (("formatted CSV", fmt), ("xlsx sheet 'S26 body'", xl)):
        tab = tab.copy()
        tab["Block"] = tab["Block"].replace("", pd.NA).ffill()
        tab["Comparison"] = tab["Comparison"].replace("", pd.NA).ffill()
        for _, r in pd.concat([A, B]).iterrows():
            comp = TABLE_COMPARISON[r.arm]
            hit = tab[(tab.Block == block_name[r.block]) & tab.Comparison.str.startswith(comp)
                      & (tab.Row == ("all" if r.row == "ALL" else r.row))]
            want_ci = f"{r3(r.point)} [{r3(r.lo)}, {r3(r.hi)}]"
            want_n = f"{r.n_slides} / {r.n_patients}"
            ok = len(hit) == 1 and hit["Plurality@5 [95% CI]"].iloc[0] == want_ci and hit["n slides / patients"].iloc[0] == want_n
            check(f"{name}: {r.block}/{r.arm}/{r.row} = {want_ci}, {want_n}", ok,
                  "" if ok else str(hit[["Plurality@5 [95% CI]", "n slides / patients"]].values.tolist()))
    # Table S9 (S28 inventory): majority counts for every arm; the CCDI-MCI arm (all-tissue in both tables)
    for arm in ARM_ORDER:
        r = A[(A.block == "reported") & (A.arm == arm)].iloc[0]
        s = s9[s9.arm == arm].iloc[0]
        want = f"{r.majority_class} {r.majority_n}/{r.n_slides} ({r3(r.majority_share)})"
        check(f"Table S9 majority_class_rate {arm} = '{want}'", s.majority_class_rate == want, s.majority_class_rate)
    r = A[(A.block == "reported") & (A.arm == "s61_ext_2c")].iloc[0]
    s = s9[s9.arm == "s61_ext_2c"].iloc[0]
    check("Table S9 CCDI-MCI arm point", s.plurality5_accuracy.endswith(f"({r3(r.point)})"), s.plurality5_accuracy)
    check("Table S9 CCDI-MCI arm interval", s.plurality5_accuracy_ci.startswith(f"{r3(r.lo)}{NDASH}{r3(r.hi)}"), s.plurality5_accuracy_ci)
    for arm in ("s61_int_2c", "s64"):  # S9's internal rows are the tumour-only v1 values: points equal S8 sensitivity
        r = A[(A.block == "sensitivity") & (A.arm == arm)].iloc[0]
        s = s9[s9.arm == arm].iloc[0]
        check(f"Table S9 {arm} point = Table S8 tumour-only sensitivity point (S9 is tumour-only v1)",
              s.plurality5_accuracy.endswith(f"({r3(r.point)})"), s.plurality5_accuracy)
        check(f"Table S9 {arm} interval vs Table S8 v2 sensitivity interval (v1 vs v2 bootstrap route)",
              s.plurality5_accuracy_ci.startswith(f"{r3(r.lo)}{NDASH}{r3(r.hi)}"),
              f"S9 {s.plurality5_accuracy_ci.split(' ')[0]} vs S8 {r3(r.lo)}{NDASH}{r3(r.hi)}", level="info")
    # prose (recorded, not fatal: the analyst edits prose)
    ms, lg = MANUSCRIPT.read_text(encoding="utf-8"), LEGENDS.read_text(encoding="utf-8")
    para = next((ln for ln in ms.splitlines() if ln.startswith("Broad tumor groups can contain clinically distinct subtypes")), "")
    check("manuscript: Results subtype paragraph found", bool(para), level="warn")
    ra = {(r.block, r.arm): r for r in A.itertuples()}
    q = lambda r: f"{r3(r.point)} ({r3(r.lo)}{NDASH}{r3(r.hi)}"  # noqa: E731
    i, l_ = ra[("reported", "s61_int_2c")], ra[("reported", "s64")]
    for s in (f"{r3(i.point)} of RMS queries (95% CI {r3(i.lo)}{NDASH}{r3(i.hi)}; {i.n_slides} slides)",
              f"{r3(l_.point)} of liver tumor queries ({r3(l_.lo)}{NDASH}{r3(l_.hi)}; {l_.n_slides} slides)",
              f"majority-subtype shares of {r3(i.majority_share)} and {r3(l_.majority_share)}"):
        check(f"manuscript Results paragraph contains '{s}'", s in para, level="warn")
    e = ra[("reported", "s61_ext_2c")]
    for s in (f"gave {q(ra[('sensitivity', 's61_int_2c')])}) and {q(ra[('sensitivity', 's64')])})",
              f"against {r3(i.point)} and {r3(l_.point)} with the all-tissue representations",
              f"{q(e)}; {e.n_slides} slides; majority share {r3(e.majority_share)})"):
        check(f"supplementary legends (Table S8/S9 notes) contain '{s}'", s in lg, level="warn")


class HandlerVDash(HandlerBase):
    """Legend handle for the majority-share mark: a short vertical dashed segment, as drawn on the axes."""

    def create_artists(self, legend, orig, xdescent, ydescent, width, height, fontsize, trans):
        x = width / 2 - xdescent
        ln = Line2D([x, x], [-ydescent - 0.45 * height, height * 1.45 - ydescent], color=orig.get_color(),
                    lw=orig.get_linewidth(), linestyle=orig.get_linestyle(), transform=trans)
        return [ln]


# Layout, in inches from the left / bottom of a 180 mm wide canvas.
H = 5.30
LBL_X, SUB_X = 0.28, 1.12  # panel A row labels / panel B group labels; panel B subtype labels
AX_X0, AX_X1 = 2.45, 5.00
N_X, VAL_X, MAJ_X = 5.36, 5.72, 6.90  # slides/patients (centre), agreement text (left), majority share (centre)
A_Y = (3.42, 4.62)  # axes bottom/top: 3 rows x 0.40 in
B_Y = (0.46, 2.73)  # 6 rows + 2 group gaps x 0.32 in
B_POS = {"s61_int_2c": (0.0, 1.0), "s61_ext_2c": (2.5, 3.5), "s64": (5.0, 6.0)}
A_POS = {"s61_int_2c": 0.0, "s61_ext_2c": 1.0, "s64": 2.0}


def _axes(fig, y0, y1):
    ax = fig.add_axes([AX_X0 / WIDTH_IN, y0 / H, (AX_X1 - AX_X0) / WIDTH_IN, (y1 - y0) / H])
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{t:.1f}" for t in XTICKS])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", width=0.6, length=2.5, pad=2)
    for t in XTICKS:
        ax.axvline(t, color="0.90", lw=0.5, zorder=0)
    ax.patch.set_visible(False)  # transparent plot area: the panel-B group hairlines run through it
    return ax


def _mark(ax, y, r):
    filled = r["block"] == "reported"
    ax.plot([r["lo"], r["hi"]], [y, y], color=BLUE, lw=1.0, solid_capstyle="butt", zorder=2)
    ax.plot([r["point"]], [y], marker="o", ms=4.2, mec=BLUE, mew=0.9, mfc=BLUE if filled else "white", ls="none", zorder=3)


def _ytext(fig, ax):
    return transforms.blended_transform_factory(fig.transFigure, ax.transData)


def _ci_text(r) -> str:
    return f"{r3(r['point'])} ({r3(r['lo'])}{NDASH}{r3(r['hi'])})"


def compose(A: pd.DataFrame, B: pd.DataFrame):
    fig = plt.figure(figsize=(WIDTH_IN, H))
    fx = lambda x: x / WIDTH_IN  # noqa: E731

    # ---- panel A
    ax = _axes(fig, *A_Y)
    ax.set_ylim(2.55, -0.55)
    tr = _ytext(fig, ax)
    yA = []
    for arm in ARM_ORDER:
        y0 = A_POS[arm]
        g = A[A.arm == arm]
        two = len(g) == 2
        for _, r in g.iterrows():
            y = y0 + ((-DODGE if r["block"] == "reported" else DODGE) if two else 0.0)
            _mark(ax, y, r)
            fig.text(fx(VAL_X), y, _ci_text(r), transform=tr, ha="left", va="center", fontsize=6.5,
                     color=INK if r["block"] == "reported" else INK2)
            yA.append(y)
        r0 = g.iloc[0]
        ax.plot([r0["majority_share"]] * 2, [y0 - 0.40, y0 + 0.40], color=GREY, lw=1.0, ls=(0, (2.2, 1.4)), zorder=1)
        task, cohort = ARMS[arm][0], ARMS[arm][1]
        fig.text(fx(LBL_X), y0, f"{task}\n{cohort}", transform=tr, ha="left", va="center", fontsize=6.5,
                 color=INK, linespacing=1.25)
        fig.text(fx(N_X), y0, f"{r0['n_slides']} / {r0['n_patients']}", transform=tr, ha="center", va="center",
                 fontsize=6.5, color=INK)
        fig.text(fx(MAJ_X), y0, r3(r0["majority_share"]), transform=tr, ha="center", va="center", fontsize=6.5,
                 color=INK)
    ax.set_xlabel("Top-5 subtype plurality agreement with the recorded subtype", fontsize=7, labelpad=3)
    hy = (A_Y[1] + 0.06) / H
    fig.text(fx(LBL_X), hy, "Task and query cohort", ha="left", va="bottom", fontsize=6.5, color=INK2)
    fig.text(fx(N_X), hy, "Slides /\npatients", ha="center", va="bottom", fontsize=6.5, color=INK2, linespacing=1.15)
    fig.text(fx(VAL_X), hy, "Agreement\n(95% CI)", ha="left", va="bottom", fontsize=6.5, color=INK2, linespacing=1.15)
    fig.text(fx(MAJ_X), hy, "Majority\nshare", ha="center", va="bottom", fontsize=6.5, color=INK2, linespacing=1.15)

    # ---- panel B
    bx = _axes(fig, *B_Y)
    bx.set_ylim(6.55, -0.55)
    tr = _ytext(fig, bx)
    for arm in ARM_ORDER:
        g = B[B.arm == arm]
        cls = ARMS[arm][2]
        for c, y0 in zip(cls, B_POS[arm]):
            gc = g[g.row == c]
            two = len(gc) == 2
            for _, r in gc.iterrows():
                y = y0 + ((-DODGE if r["block"] == "reported" else DODGE) if two else 0.0)
                _mark(bx, y, r)
                fig.text(fx(VAL_X), y, _ci_text(r), transform=tr, ha="left", va="center", fontsize=6.5,
                         color=INK if r["block"] == "reported" else INK2)
            r0 = gc.iloc[0]
            fig.text(fx(SUB_X), y0, SUBTYPE_LABEL[c], transform=tr, ha="left", va="center", fontsize=6.5, color=INK)
            fig.text(fx(N_X), y0, f"{r0['n_slides']} / {r0['n_patients']}", transform=tr, ha="center", va="center",
                     fontsize=6.5, color=INK)
        ymid = sum(B_POS[arm]) / 2
        fig.text(fx(LBL_X), ymid, GROUP_LABEL[arm], transform=tr, ha="left", va="center", fontsize=6.5, color=INK,
                 linespacing=1.2)
    for ysep in (1.75, 4.25):  # hairlines between the three task/cohort groups, across labels and columns
        fig.add_artist(Line2D([fx(LBL_X), fx(WIDTH_IN - 0.05)], [ysep, ysep], transform=tr, color="0.85", lw=0.5,
                              zorder=0))
    bx.set_xlabel("Top-5 subtype plurality agreement within each recorded subtype", fontsize=7, labelpad=3)
    hy = (B_Y[1] + 0.06) / H
    fig.text(fx(LBL_X), hy, "Task, cohort", ha="left", va="bottom", fontsize=6.5, color=INK2)
    fig.text(fx(SUB_X), hy, "Recorded subtype", ha="left", va="bottom", fontsize=6.5, color=INK2)
    fig.text(fx(N_X), hy, "Slides /\npatients", ha="center", va="bottom", fontsize=6.5, color=INK2, linespacing=1.15)
    fig.text(fx(VAL_X), hy, "Agreement\n(95% CI)", ha="left", va="bottom", fontsize=6.5, color=INK2, linespacing=1.15)

    # ---- key and panel letters
    h_all = Line2D([], [], color=BLUE, lw=1.0, marker="o", ms=4.2, mfc=BLUE, mec=BLUE, mew=0.9)
    h_tum = Line2D([], [], color=BLUE, lw=1.0, marker="o", ms=4.2, mfc="white", mec=BLUE, mew=0.9)
    h_maj = Line2D([], [], color=GREY, lw=1.0, ls=(0, (2.2, 1.4)))
    fig.legend([h_all, h_tum, h_maj],
               ["All-tissue mosaics (95% CI)", "Tumor-filtered tiles, sensitivity (95% CI)", "Majority-subtype share"],
               handler_map={h_maj: HandlerVDash()}, loc="lower center",
               bbox_to_anchor=(fx((LBL_X + WIDTH_IN) / 2), (H - 0.30) / H), ncol=3, fontsize=6.5, frameon=False,
               handlelength=2.0, columnspacing=2.0, borderaxespad=0)
    fig.text(fx(0.02), (H - 0.28) / H, "A", fontsize=10, fontweight="bold", ha="left", va="bottom")
    fig.text(fx(0.02), (B_Y[1] + 0.30) / H, "B", fontsize=10, fontweight="bold", ha="left", va="bottom")
    return fig


def main():
    style()
    raw, fmt, xl, s9 = load()
    A, B = rows_for_panels(raw)
    check("panel A: 5 rows (3 all-tissue, 2 tumour-only)", len(A) == 5 and (A.block == "reported").sum() == 3)
    check("panel B: 10 rows (6 all-tissue, 4 tumour-only)", len(B) == 10 and (B.block == "reported").sum() == 6)
    check("all intervals inside the drawn x-range",
          bool((pd.concat([A.lo, B.lo, A.majority_share]) >= XLIM[0]).all() and (pd.concat([A.hi, B.hi]) <= 1.0).all()))
    value_checks(A, B, fmt, xl, s9)

    fig = compose(A, B)
    d = out_dir(FIGURE)
    leg = legibility(fig, d, FIG_ID)
    ko = key_overlaps(fig)
    leg["key_text_overlaps"] = ko
    if ko:
        leg["pass"] = False
    (d / f"legibility_check_{FIG_ID}.json").write_text(json.dumps(leg, indent=1))
    if not leg["pass"]:
        print(json.dumps({k: v for k, v in leg.items() if k != "texts"}, indent=1))
    assert leg["pass"], "legibility check failed"
    png, pdf = d / f"{STEM}.png", d / f"{STEM}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", metadata={"CreationDate": None})
    plt.close(fig)
    srcA, srcB = d / "source_data_panelA.csv", d / "source_data_panelB.csv"
    A.to_csv(srcA, index=False, float_format="%.16g")
    B.to_csv(srcB, index=False, float_format="%.16g")
    vc = d / "value_check.json"
    n_fail = sum(c["result"] == "FAIL" for c in CHECKS)
    n_warn = sum(c["result"] == "WARN" for c in CHECKS)
    n_info = sum(c["result"] == "INFO" for c in CHECKS)
    vc.write_text(json.dumps(dict(n_checks=len(CHECKS), n_pass=len(CHECKS) - n_fail - n_warn - n_info, n_warn=n_warn,
                                  n_info=n_info, n_fail=n_fail,
                                  checks=CHECKS), indent=1, ensure_ascii=False) + "\n")
    legj = d / f"legibility_check_{FIG_ID}.json"

    def line(r):
        return (f"{r.block}/{r.arm}/{r.row}: {r3(r.point)} ({r3(r.lo)}–{r3(r.hi)}), {r.n_slides} slides / {r.n_patients} patients"
                + (f", majority share {r.majority_class} {int(r.majority_n)}/{r.n_slides} = {r3(r.majority_share)}" if r.panel == "A" else ""))
    note = (
        f"Provisional Supplementary Fig. {FIG_ID} (pass 019, Haoran 2026-11-06). Plots only values of Supplementary Table S8 "
        "(workspace S26 v2, 2026-11-03; built by scripts/supp_tables/build_S26_v2_all_tissue_20261103.py from "
        "$PROSPER_OUT/r3_7_all_tissue_ci_20261103/ and, for the CCDI-MCI arm, the unchanged version-1 row); nothing recomputed. "
        "UNI2–Optimal Transport, each query ranked only against subtype-labelled slides of its family, same-patient slides "
        "excluded; CCDI-MCI queries against the COMET RMS library (373 slides, 306 patients). Intervals: percentile 95 %, "
        "5,000 whole-patient resamples, seed 123, reference library fixed (Table S8 legend). Filled marks: all-tissue mosaics; "
        "open marks: precomputed tumor-filtered tiles (tile classifier not held out by patient), COMET arms only. Majority share = "
        "larger subtype's slides / all slides of the arm (Table S8 n_queries; equals Table S9 majority_class_rate). Table S9 not "
        "drawn (tumour-only v1 values for the internal arms; no whole-patient interval for KidRare). "
        f"Value checks: {len(CHECKS)} ({n_fail} FAIL, {n_warn} WARN, {n_info} INFO) in value_check.json. Legibility: {leg['n_texts']} texts "
        f">= 6 pt, no overlaps (incl. key). Canvas 180 mm x {H * 25.4:.1f} mm before tight-bbox cropping; PNG 300 dpi.\n\n"
        "Plotted values:\n\n" + "\n".join(f"- {line(r)}" for r in pd.concat([A, B]).itertuples()) + "\n"
    )
    inputs = [S8_RAW, S8_FMT, S8_XLSX, S8_REPORT, S9_TABLE, MANUSCRIPT, LEGENDS]
    write_provenance(d, [png, pdf, srcA, srcB, vc, legj], inputs, note, Path(__file__), title=f"{FIGURE} / {STEM}")
    print(f"wrote {png}")
    print(json.dumps({"checks": len(CHECKS), "fail": n_fail, "warn": n_warn, "info": n_info, "legibility_texts": leg["n_texts"]}))
    for p in (png, pdf, srcA, srcB):
        print(sha256(p)[:16], p.name)


if __name__ == "__main__":
    main()
