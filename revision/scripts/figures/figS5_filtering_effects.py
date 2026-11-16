"""Fig. S5 (new): cohort-dependent effects of predicted tumor filtering.

[internal record] block R1.3-T9. Source (E20):
`$PROSPER_OUT/r1_3_external_synthesis_20260930/paired_primary_differences.csv`
using the six current filter identities named in the block (STV query-only
696 / both-sided 688; KidRare expanded query-only 1581 / both-sided 1531;
CCDI FFPE scope-clear query_tumor_ot / both_tumor_ot); baseline is the
all-tissue whole_slide_ot arm on identical returned queries.

A, B: macro hit@5 / macro precision@5 differences (filtered - all-tissue)
with 95 % paired bootstrap intervals and matched n.
C: group-level differences in cohort facets (fixed group order, paired-arm
shapes, intervals, n).
D (only with --with-hit-yield): full-target macro hit-yield differences with
returned/target counts. Omitted by default since 2026-11-04 (supplementary-figure
audit, analyst recommendation, reversible): returned coverage is >= 99.9 %
in every arm, so D repeats A to the third decimal; the values stay in the coverage
sheet of Supplementary Table S19.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GREY, GROUP_ORDER_ALPHA, HIT_COLOR, PREC_COLOR, prosper_out, save, style  # noqa: E402

SRC = prosper_out() / "r1_3_external_synthesis_20260930/paired_primary_differences.csv"
FIGURE = "figS5_filtering_effects"
IDS = [  # cohort, population, configuration, label, arm
    ("stv", "query_only_696", "query_only", "STV", "query-only"),
    ("stv", "both_sided_688", "both_sided", "STV", "both-sided"),
    ("kidrare", "expanded_query_only_1581", "query_only", "KidRare", "query-only"),
    ("kidrare", "expanded_both_sided_1531", "both_sided", "KidRare", "both-sided"),
    ("ccdi", "ffpe_scope_clear", "query_tumor_ot", "CCDI-MCI FFPE", "query-only"),
    ("ccdi", "ffpe_scope_clear", "both_tumor_ot", "CCDI-MCI FFPE", "both-sided"),
]
ARM_STYLE = {"query-only": ("o", "filled"), "both-sided": ("s", "open")}
ACCEPT = {  # R1.3-T9 acceptance values (macro, hit then precision)
    ("stv", "query_only"): (0.011663, 0.014091), ("stv", "both_sided"): (0.024322, 0.033461),
    ("kidrare", "query_only"): (-0.040104, -0.019747), ("kidrare", "both_sided"): (-0.003205, 0.032526),
    ("ccdi", "query_tumor_ot"): (0.016651, 0.008283), ("ccdi", "both_tumor_ot"): (-0.006203, 0.013708),
}


def load():
    p = pd.read_csv(SRC)
    parts = []
    for c, pop, cfg, lab, arm in IDS:
        s = p[(p.cohort == c) & (p.population == pop) & (p.configuration == cfg)].copy()
        assert len(s) > 0, (c, pop, cfg)
        s["label"] = lab; s["arm"] = arm
        parts.append(s)
    d = pd.concat(parts, ignore_index=True)
    for c, pop, cfg, lab, arm in IDS:
        m = d[(d.cohort == c) & (d.configuration == cfg) & (d.tumor_group == "macro") & (d.estimand == "conditional_returned")]
        h = m[m.metric == "hit_at_5"].difference.iloc[0]; pr = m[m.metric == "precision_at_5"].difference.iloc[0]
        eh, ep = ACCEPT[(c, cfg)]
        assert abs(h - eh) < 1e-6 and abs(pr - ep) < 1e-6, (c, cfg, h, pr)
    return d


def forest(ax, d, metric, estimand, title, xlab, show_counts=False):
    rows = []
    for c, pop, cfg, lab, arm in IDS:
        r = d[(d.cohort == c) & (d.configuration == cfg) & (d.tumor_group == "macro") & (d.metric == metric)
              & (d.estimand == estimand)].iloc[0]
        rows.append((lab, arm, r))
    y = np.arange(len(rows))[::-1]
    color = HIT_COLOR if "hit" in metric else PREC_COLOR
    ax.axvline(0, color="black", linewidth=0.6)
    for yi, (lab, arm, r) in zip(y, rows):
        marker, fill = ARM_STYLE[arm]
        ax.errorbar(r.difference, yi, xerr=[[r.difference - r.difference_ci_lo], [r.difference_ci_hi - r.difference]],
                    fmt=marker, color=color, mfc=color if fill == "filled" else "white", ms=4, capsize=2, elinewidth=0.8)
        if show_counts:
            txt = f"{int(r.n_returned)}/{int(r.n)} returned"
        else:
            txt = f"matched n = {int(r.n)}"
        ax.text(1.02, yi, f"{r.difference:+.3f} [{r.difference_ci_lo:+.3f}, {r.difference_ci_hi:+.3f}]  {txt}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=5.5)
    ax.set_yticks(y); ax.set_yticklabels([f"{lab}, {arm}" for lab, arm, _ in rows])
    ax.set_title(title, loc="left"); ax.set_xlabel(xlab)
    ax.set_ylim(-0.6, len(rows) - 0.4); ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    lim = max(abs(d[(d.tumor_group == "macro") & (d.metric == metric) & (d.estimand == estimand)][["difference_ci_lo", "difference_ci_hi"]].abs().max()))
    ax.set_xlim(-lim * 1.15, lim * 1.15)


def groups(axes, d):
    cohorts = [("stv", "STV"), ("kidrare", "KidRare"), ("ccdi", "CCDI-MCI: FFPE")]
    x = np.arange(len(GROUP_ORDER_ALPHA))
    for ax, (c, lab) in zip(axes, cohorts):
        sub = d[(d.cohort == c) & ~d.tumor_group.isin(["macro", "micro"]) & (d.estimand == "conditional_returned")]
        ax.axhline(0, color="black", linewidth=0.6)
        offsets = {("hit_at_5", "query-only"): -0.27, ("hit_at_5", "both-sided"): -0.09,
                   ("precision_at_5", "query-only"): 0.09, ("precision_at_5", "both-sided"): 0.27}
        hit_q = sub[(sub.metric == "hit_at_5") & (sub.arm == "query-only")].set_index("tumor_group").reindex(GROUP_ORDER_ALPHA)
        for (metric, arm), dx in offsets.items():
            m = sub[(sub.metric == metric) & (sub.arm == arm)].set_index("tumor_group").reindex(GROUP_ORDER_ALPHA)
            color = HIT_COLOR if metric == "hit_at_5" else PREC_COLOR
            marker, fill = ARM_STYLE[arm]
            for xi, g in zip(x, GROUP_ORDER_ALPHA):
                r = m.loc[g]
                if pd.isna(r.difference):
                    continue
                ax.errorbar(xi + dx, r.difference, yerr=[[r.difference - r.difference_ci_lo], [r.difference_ci_hi - r.difference]],
                            fmt=marker, color=color, mfc=color if fill == "filled" else "white", ms=2.8, capsize=1, elinewidth=0.6)
        labels = []
        for g in GROUP_ORDER_ALPHA:
            r = hit_q.loc[g]
            if pd.isna(r.n):
                labels.append(f"{g}  n = 0")
                ax.text(x[GROUP_ORDER_ALPHA.index(g)], 0, "n/a", rotation=90, ha="center", va="center", fontsize=4.5, color=GREY)
            else:
                labels.append(f"{g}  {int(r.n)}")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
        ax.set_title(lab); ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    axes[0].set_ylabel("difference (filtered − all-tissue)")
    h = [plt.Line2D([], [], marker="o", color=HIT_COLOR, linestyle="none", ms=4),
         plt.Line2D([], [], marker="s", color=HIT_COLOR, mfc="white", linestyle="none", ms=4),
         plt.Line2D([], [], marker="o", color=PREC_COLOR, linestyle="none", ms=4),
         plt.Line2D([], [], marker="s", color=PREC_COLOR, mfc="white", linestyle="none", ms=4)]
    axes[1].legend(h, ["hit@5, query-only", "hit@5, both-sided", "precision@5, query-only", "precision@5, both-sided"],
                   loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=4, frameon=False)
    axes[1].set_xlabel("tumor group   matched n (query-only arm)", labelpad=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--with-hit-yield", action="store_true",
                    help="restore panel D (full-target macro hit-yield); omitted by default since 2026-11-04")
    args = ap.parse_args()
    style()
    d = load()
    if args.with_hit_yield:
        fig = plt.figure(figsize=(8.5, 8.6))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.7, 1], hspace=0.75, left=0.2, right=0.98, top=0.96, bottom=0.08)
    else:
        fig = plt.figure(figsize=(8.5, 6.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.7], hspace=0.6, left=0.2, right=0.98, top=0.95, bottom=0.10)
    gsab = gs[0].subgridspec(1, 2, wspace=1.9)
    ax_a = fig.add_subplot(gsab[0, 0]); ax_b = fig.add_subplot(gsab[0, 1])
    forest(ax_a, d, "hit_at_5", "conditional_returned", "A  Macro hit@5", "Δ macro hit@5")
    forest(ax_b, d, "precision_at_5", "conditional_returned", "B  Macro precision@5", "Δ macro precision@5")
    ax_b.set_yticklabels([])
    gsc = gs[1].subgridspec(1, 3, wspace=0.15)
    axes_c = [fig.add_subplot(gsc[0, i]) for i in range(3)]
    groups(axes_c, d)
    axes_c[0].text(-0.3, 1.1, "C", transform=axes_c[0].transAxes, fontsize=11, fontweight="bold", va="top")
    if args.with_hit_yield:
        gsd = gs[2].subgridspec(1, 2, wspace=1.9)
        ax_d = fig.add_subplot(gsd[0, 0])
        forest(ax_d, d, "hit_yield", "full_target", "D  Full-target macro hit-yield",
               "Δ macro hit-yield (empty representation = no hit)", show_counts=True)
    note = ("Six current filter identities (R1.3-T9 table); baseline = all-tissue whole_slide_ot on identical returned "
            "queries. Macro differences in A/B equal the block's acceptance values (asserted in the script). "
            "Query-only filtering keeps the all-tissue library; both-sided filtering changes reference membership and "
            "removes NORMAL queries, so arm-level macro results refer to different class populations. "
            "Old KidRare 1226/1176 contrasts are not shown."
            + ("" if args.with_hit_yield else
               " Panel D (full-target macro hit-yield) omitted by default since 2026-11-04: returned coverage >= 99.9 % "
               "makes it repeat A to the third decimal; values remain in the source data and Supplementary Table S19; "
               "restore with --with-hit-yield."))
    save(fig, FIGURE, "Figure_S5_filtering_effects", [SRC], note, {"source_data": d})


if __name__ == "__main__":
    main()
