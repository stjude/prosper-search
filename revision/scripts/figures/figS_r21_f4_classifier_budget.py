"""Supplementary Fig. S-R2.1-F4: tumor tile-classifier quality and the
5 %-training-fraction sensitivity analysis (material removed from main Fig. 3 on 2026-10-21 ≈14:08: no review comment asked for it, and the numbers
are in Supplementary Tables S16 and S-R2.1-B).

Panel a  sensitivity and specificity for the tumor class of the UNI2,
         Virchow2, H-Optimus-1 and CONCH v1.5 (TITAN) tile classifiers trained
         on all annotated tiles, each labeled tile scored once by the fold
         classifier that had not seen its patient (bars, pooled over tiles;
         whiskers, range over the five patient folds). The in-sample
         (training-tile) bars of the earlier draft are not drawn.
Panel b  as main Fig. 3B (filtered minus all-tissue change in hit@5 and
         precision@5, classifiers trained on all annotated tiles) with both
         filtering arms: tumor-only query against the full library (open) and
         tumor-only query and library (filled; the arm shown in Fig. 3B).
Panel c  as b for classifiers trained on 5 % of the annotated tiles. Same
         y-scale as Fig. 3B throughout.

Loaders, panel functions, colours and y-limits are imported from
`fig3_tumor_filtering.py`, so a and b are drawn from exactly the rows that
module cross-checks (tile_metrics.csv; contrasts.csv / wording_classes.csv /
writing_001 panel extracts). Nothing is recomputed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import save, style  # noqa: E402
import fig3_tumor_filtering as f3  # noqa: E402

FIGURE = "figS_r21_f4_classifier_budget"
STEM = "FigS_R2.1-F4_classifier_and_5pct"


def main():
    style()
    a = f3.load_classifier()
    d = f3.load_contrasts()
    ylim = f3.shared_ylim(d)

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.15, 1.15], height_ratios=[1, 1], wspace=0.35, hspace=0.6,
                          left=0.08, right=0.99, top=0.92, bottom=0.14)
    ax_a = fig.add_subplot(gs[0, 0])
    axes_b = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    axes_c = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]

    f3.panel_classifier(ax_a, a, show_in_sample=False)
    # Display name per the Fig. 2 decision D3 (2026-11-03): TITAN first, its patch encoder in parentheses.
    # Since 2026-11-04 (afternoon) the label lives in fig3_tumor_filtering.ENCODERS; the earlier same-day
    # re-render overrode it here. Asserted so a regression in the constant cannot pass silently.
    ticks = [t.get_text() for t in ax_a.get_xticklabels()]
    assert "TITAN\n(CONCH v1.5)" in ticks and "CONCH v1.5\n(TITAN)" not in ticks, ticks
    ax_a.set_title("Tumor tile classifier", loc="left", fontsize=7.5)
    ax_a.get_legend().remove()
    h, l = ax_a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.19, 0.42), ncol=1, frameon=False, handlelength=1.2,
               fontsize=6.5, title="Panel a", title_fontsize=6.5)
    f3.panel_contrasts(axes_b, d[d.budget == 100], 100, ylim)
    axes_b[0].set_title("Classifier trained on all annotated tiles, both filtering arms", loc="left", fontsize=7.5)
    f3.panel_contrasts(axes_c, d[d.budget == 5], 5, ylim)
    axes_c[0].set_title("Classifier trained on 5% of annotated tiles, both filtering arms", loc="left", fontsize=7.5)
    f3.filtering_key(fig, 0.55, 0.005, arms=True)

    for ax, lab, dx in [(ax_a, "a", -0.30), (axes_b[0], "b", -0.36), (axes_c[0], "c", -0.36)]:
        ax.text(dx, 1.16, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    note = ("Panel a: TF-2 S2 classifier report, pooled_cross_fitted and fold0-4 rows at the 100 % budget "
            "(tile_metrics.csv); n = 544,182 tiles (UNI2, Virchow2, H-Optimus-1), 167,489 patches (CONCH v1.5); the "
            "full_in_sample grey bars of the earlier main-figure draft are not drawn (author, 2026-10-21 ≈14:08). "
            "Panels b-c: TF-2 S6 primary evaluation, family a_filtered_minus_all_tissue, macro, k = 5, population "
            "both_returned, both arms (query_only open, both_filtered filled), 100 % (b) and 5 % (c) budgets; n_queries 2,604 (UNI2, TITAN) or 2,603 (Virchow2, H-Optimus-1); 1,800 patients; "
            "whole-patient bootstrap 95 % intervals. Values cross-checked (tolerance 1e-9) against wording_classes.csv "
            f"and writing_001/fig3_panel_{{A,B,C}}_data.csv. Y-limits {ylim[0]:.2f} to {ylim[1]:.2f} shared with main "
            "Fig. 3B. 'Label budget', 'cross-fitted', 'in-sample' and all n counts are kept off the figure face and "
            "belong in the legend.")
    inputs = [f3.TILE_METRICS, f3.IN_SAMPLE, f3.CONTRASTS, f3.WORDING, f3.PANEL_DATA["A"], f3.PANEL_DATA["B"], f3.PANEL_DATA["C"]]
    save(fig, FIGURE, STEM, inputs, note, {"source_data_panel_a": a, "source_data_panel_b": d[d.budget == 100],
                                             "source_data_panel_c": d[d.budget == 5]})


if __name__ == "__main__":
    main()
