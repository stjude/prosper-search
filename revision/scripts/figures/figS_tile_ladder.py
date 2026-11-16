"""Supplementary figure S-R2.2 — retrieval as a function of the fraction of tiles per slide
(the "mosaic-percentage ladder"; a review comment: all tiles vs a subset).

Reads only the accepted whole-patient ladder (`$PROSPER_OUT/r2_2_tile_ladder_whole_patient_20261007/`,
R2.2-T4 source, PASS 14/14) and the measured cost panel values transcribed from
`[internal record]` (checked against that file at run time). Nothing is recomputed.

Panels
  A  macro hit@5 vs tile fraction, three aggregation families, whole-patient 95 % intervals
  B  macro precision@5, same layout
  C  paired contrasts (macro hit@5 and precision@5): each fraction vs 15 % within a family and OT vs
     the other families at the same fraction, same queries and patient draws
  D  per-group hit@5 at the main configuration (OT 15 %), OT 30 % and median pooling 100 % (all tiles),
     with slides / patients per group
  E  measured distance-matrix CPU-hours per family and fraction (log scale); OT 100 % is an
     extrapolated range, drawn open
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GREY, GROUP_ORDER_ALPHA, REPO, prosper_out, save, style  # noqa: E402
from figS1_S2_neighbor_ladder import legibility  # noqa: E402

FIGURE = "figS_tile_ladder"
WIDTH_IN = 180 / 25.4
SRC = prosper_out() / "r2_2_tile_ladder_whole_patient_20261007"
COST_MD = REPO / "docs" / "results" / "r2_2_cost_panel_20261006.md"

FAMILY = {
    "ot_cosine": ("Optimal Transport (exact, cosine cost)", "#1f77b4", "o"),
    "yottixel": ("Yottixel", "#2ca02c", "s"),
    "median_pooling": ("Median pooling", "#8c564b", "D"),
}
FRACS = [5, 15, 30, 50, 100]
XPOS = {f: i for i, f in enumerate(FRACS)}

# Measured LSF CPU-hours, transcribed from the cost panel (bold values); checked at run time.
COST = [
    ("yottixel", 5, 22, "measured"),
    ("yottixel", 15, 174, "measured"),
    ("yottixel", 30, 640, "measured"),
    ("ot_cosine", 15, 3595, "measured"),
    ("ot_cosine", 30, 8954, "measured"),
]
OT100_RANGE = (40_000, 160_000)  # extrapolated, illustrative (cost panel 100 % row)


def check_cost_transcription():
    txt = COST_MD.read_text()
    for fam, frac, cpu, _ in COST:
        if not re.search(rf"\*\*{cpu:,}\*\*|\*\*{cpu}\*\*", txt):
            raise SystemExit(f"cost panel value {cpu} ({fam} {frac} %) not found in {COST_MD.name}")
    if "≈40,000–160,000 extrapolated" not in txt:
        raise SystemExit("OT 100 % extrapolated range not found in the cost panel")


def load():
    summ = pd.read_csv(SRC / "ladder_summary.csv")
    paired = pd.read_csv(SRC / "paired_differences.csv")
    grp = pd.read_csv(SRC / "per_group.csv")
    ties = pd.read_csv(SRC / "tie_bounds.csv")
    return summ, paired, grp, ties


def panel_metric(ax, summ, metric, ylabel, letter, show_legend):
    d = summ[(summ.metric == metric) & (summ.population == "macro")]
    for fam, (label, col, mk) in FAMILY.items():
        s = d[d.family == fam].sort_values("retention_pct")
        x = np.array([XPOS[f] for f in s.retention_pct]) + {"ot_cosine": 0.0, "yottixel": -0.09, "median_pooling": 0.09}[fam]
        ax.errorbar(x, s.point, yerr=[s.point - s.ci_lo, s.ci_hi - s.point], fmt=mk + "-", color=col, ms=4,
                    lw=1, elinewidth=0.8, capsize=2, label=label)
    ax.set_xticks(range(len(FRACS)))
    ax.set_xticklabels([f"{f} %" for f in FRACS])
    ax.set_xlim(-0.5, len(FRACS) - 0.5)
    ax.set_xlabel("Tiles retained per slide")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    ax.text(-0.22, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    return ax.get_legend_handles_labels()


CONTRASTS = [
    # (minuend family, pct, subtrahend family, pct, label)
    ("median_pooling", 100, "median_pooling", 15, "Median 100 % − 15 % (all tiles vs subset)"),
    ("median_pooling", 50, "median_pooling", 15, "Median 50 % − 15 %"),
    ("median_pooling", 30, "median_pooling", 15, "Median 30 % − 15 %"),
    ("median_pooling", 5, "median_pooling", 15, "Median 5 % − 15 %"),
    ("yottixel", 30, "yottixel", 15, "Yottixel 30 % − 15 %"),
    ("yottixel", 5, "yottixel", 15, "Yottixel 5 % − 15 %"),
    ("ot_cosine", 30, "ot_cosine", 15, "OT 30 % − 15 %"),
    ("ot_cosine", 15, "median_pooling", 15, "OT 15 % − Median 15 %"),
    ("ot_cosine", 15, "yottixel", 15, "OT 15 % − Yottixel 15 %"),
    ("ot_cosine", 30, "median_pooling", 30, "OT 30 % − Median 30 %"),
    ("ot_cosine", 30, "yottixel", 30, "OT 30 % − Yottixel 30 %"),
]


def panel_contrasts(ax, paired, letter):
    rows = []
    for i, (mf, mp, sf, sp, lab) in enumerate(CONTRASTS):
        for metric, col, mk, dy in (("hit5", "#1f77b4", "o", 0.17), ("precision5", "#e07b00", "D", -0.17)):
            r = paired[(paired.minuend_family == mf) & (paired.minuend_retention_pct == mp) &
                       (paired.subtrahend_family == sf) & (paired.subtrahend_retention_pct == sp) &
                       (paired.metric == metric) & (paired.population == "macro")]
            assert len(r) == 1, (lab, metric, len(r))
            r = r.iloc[0]
            y = len(CONTRASTS) - 1 - i + dy
            ax.errorbar(r.point, y, xerr=[[r.point - r.ci_lo], [r.ci_hi - r.point]], fmt=mk, color=col, ms=3.2,
                        elinewidth=0.8, capsize=1.6)
            rows.append(dict(contrast=lab, metric=metric, point=r.point, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                             ci_includes_zero_nominal=r.ci_includes_zero_nominal, n_queries=r.n_queries,
                             n_patients=r.n_patients))
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(CONTRASTS)))
    ax.set_yticklabels([c[4] for c in CONTRASTS][::-1])
    ax.set_ylim(-0.6, len(CONTRASTS) - 0.4)
    ax.set_xlabel("Paired difference, macro average\n(95 % whole-patient interval)")
    ax.grid(axis="x", lw=0.3, color="#dddddd")
    for y in range(len(CONTRASTS)):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f3f3f3", zorder=0, lw=0)
    ax.text(-0.66, 1.01, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.legend(handles=[Line2D([], [], marker="o", color="#1f77b4", ls="", ms=3.2, label="hit@5"),
                       Line2D([], [], marker="D", color="#e07b00", ls="", ms=3.2, label="precision@5")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False)
    return pd.DataFrame(rows)


GROUP_ARMS = [
    ("ot_cosine", 15, "OT 15 % (main configuration)", "#1f77b4", "o", -0.22),
    ("ot_cosine", 30, "OT 30 %", "#6baed6", "o", 0.0),
    ("median_pooling", 100, "Median 100 % (all tiles)", "#8c564b", "D", 0.22),
]


def panel_groups(ax, axn, grp, letter):
    g = grp[grp.metric == "hit5"]
    order = [x for x in GROUP_ORDER_ALPHA if x in set(g.group)]
    rows = []
    for fam, pct, lab, col, mk, dy in GROUP_ARMS:
        s = g[(g.family == fam) & (g.retention_pct == pct)].set_index("group").loc[order]
        y = np.arange(len(order)) + dy
        ax.errorbar(s.point, y, xerr=[s.point - s.ci_lo, s.ci_hi - s.point], fmt=mk, color=col, ms=3.2,
                    elinewidth=0.8, capsize=1.6, label=lab, ls="")
        for grp_name, r in s.iterrows():
            rows.append(dict(group=grp_name, arm=lab, point=r.point, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                             n_queries=r.n_queries, n_patients=r.n_patients, flag=r.flag))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0.25, 1.0)
    ax.set_xlabel("hit@5 per tumor group\n(95 % whole-patient interval)")
    ax.grid(axis="x", lw=0.3, color="#dddddd")
    for y in range(len(order)):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f3f3f3", zorder=0, lw=0)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.42, 1.0), ncol=1, frameon=False, handletextpad=0.3, labelspacing=0.2)
    ax.text(-0.3, 1.14, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    n = g[(g.family == "ot_cosine") & (g.retention_pct == 15)].set_index("group").loc[order]
    axn.set_xlim(0, 1)
    axn.set_ylim(len(order) - 0.5, -0.5)
    for i, grp_name in enumerate(order):
        axn.text(0.05, i, f"{int(n.n_queries[grp_name])} / {int(n.n_patients[grp_name])}", ha="left", va="center", fontsize=6.5)
    axn.text(0.05, -0.75, "slides /\npatients", ha="left", va="bottom", fontsize=6.5, color=GREY)
    axn.axis("off")
    return pd.DataFrame(rows)


def panel_cost(ax, letter):
    rows = []
    for fam, frac, cpu, status in COST:
        lab, col, mk = FAMILY[fam]
        ax.plot(XPOS[frac], cpu, marker=mk, color=col, ms=4.5, ls="")
        rows.append(dict(family=fam, retention_pct=frac, cpu_hours=cpu, status=status))
    for fam in ("yottixel", "ot_cosine"):
        pts = [(XPOS[f], c) for ff, f, c, _ in COST if ff == fam]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=FAMILY[fam][1], lw=1)
    lo, hi = OT100_RANGE
    ax.plot([XPOS[100]] * 2, [lo, hi], color=FAMILY["ot_cosine"][1], lw=1, ls=":")
    ax.plot(XPOS[100], lo, marker="o", mfc="white", color=FAMILY["ot_cosine"][1], ms=4.5, ls="")
    ax.plot(XPOS[100], hi, marker="o", mfc="white", color=FAMILY["ot_cosine"][1], ms=4.5, ls="")
    ax.plot([XPOS[30], XPOS[100]], [8954, lo], color=FAMILY["ot_cosine"][1], lw=1, ls=":")
    rows.append(dict(family="ot_cosine", retention_pct=100, cpu_hours=f"{lo}-{hi}", status="extrapolated, illustrative"))
    ax.set_yscale("log")
    ax.set_ylim(10, 3e5)
    ax.set_xticks(range(len(FRACS)))
    ax.set_xticklabels([f"{f} %" for f in FRACS])
    ax.set_xlim(-0.5, len(FRACS) - 0.5)
    ax.set_xlabel("Tiles retained per slide")
    ax.set_ylabel("Distance-matrix CPU-hours (log)")
    ax.grid(axis="y", lw=0.3, color="#dddddd", which="major")
    ax.text(-0.3, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.legend(handles=[Line2D([], [], marker="o", color=FAMILY["ot_cosine"][1], ls="", ms=4, label="measured"),
                       Line2D([], [], marker="o", color=FAMILY["ot_cosine"][1], mfc="white", ls=":", ms=4, label="extrapolated range")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.3, columnspacing=0.8)
    return pd.DataFrame(rows)


def main():
    style()
    check_cost_transcription()
    summ, paired, grp, ties = load()
    fig = plt.figure(figsize=(WIDTH_IN, 7.6))
    # top row (fractions of figure): A, B, E ; bottom row: C (wide labels), D, n column
    top_y, top_h = 0.665, 0.255
    axA = fig.add_axes([0.075, top_y, 0.22, top_h])
    axB = fig.add_axes([0.395, top_y, 0.22, top_h])
    axE = fig.add_axes([0.73, top_y, 0.25, top_h])
    bot_y, bot_h = 0.065, 0.46
    axC = fig.add_axes([0.235, bot_y, 0.33, bot_h])
    axD = fig.add_axes([0.70, bot_y, 0.20, bot_h])
    axDn = fig.add_axes([0.905, bot_y, 0.09, bot_h])
    hA, lA = panel_metric(axA, summ, "hit5", "Macro hit@5 (95 % whole-patient interval)", "A", show_legend=False)
    panel_metric(axB, summ, "precision5", "Macro precision@5 (95 % whole-patient interval)", "B", show_legend=False)
    fig.legend(hA, lA, loc="upper center", bbox_to_anchor=(0.36, 0.985), ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.2)
    cost_rows = panel_cost(axE, "E")
    contrast_rows = panel_contrasts(axC, paired, "C")
    group_rows = panel_groups(axD, axDn, grp, "D")
    macro = summ[(summ.population == "macro") & (summ.metric.isin(["hit5", "precision5"]))]
    d = save(fig, FIGURE, "Figure_S_tile_ladder",
             inputs=[SRC / "ladder_summary.csv", SRC / "paired_differences.csv", SRC / "per_group.csv",
                     SRC / "tie_bounds.csv", COST_MD],
             note=("R2.2 mosaic-percentage ladder figure (reviewer 2, bullet 2). Accepted whole-patient ladder "
                   "(2,750 queries / 1,857 patients; B = 5,000; nominal 95 %, unadjusted). Yottixel points and every "
                   "contrast using a Yottixel rung rest on the arbitrary saved order within distance ties "
                   "(tie_bounds.csv: macro hit@5 ranges 0.8865-0.8903 / 0.8917-0.8991 / 0.8940-0.8985 at 5/15/30 %). "
                   "Cost values transcribed from docs/results/r2_2_cost_panel_20261006.md and checked at run time; "
                   "median-pooling costs were not metered. OT 100 % is an extrapolated range, not a measurement."),
             source_frames={"source_A_B_macro": macro, "source_C_contrasts": contrast_rows,
                            "source_D_per_group": group_rows, "source_E_cost": cost_rows,
                            "source_tie_bounds": ties})
    chk = legibility(fig, d, "tile_ladder")
    print("legibility pass:", chk["pass"], "overlaps:", chk["overlaps"], "size mm:", chk["width_mm"], chk["height_mm"])
    (d / "run_summary.json").write_text(json.dumps(dict(legibility_pass=chk["pass"], overlaps=chk["overlaps"]), indent=1))


if __name__ == "__main__":
    main()
