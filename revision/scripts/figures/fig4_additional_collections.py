"""Fig. 4 (revised, single grid; the design decisions of 2026-10-21 ≈15:32 CDT).

Fig. 4 shows the additional query collections only (Route 1; the simulated-
perturbation panel is Supplementary Fig. S-R2.1-F5, `figS_perturbation.py`).
One grid: top row = macro average over the collection's observed groups, then the
15 diagnostic groups ordered by their share of the reference library; columns =
STV | KidRare | CCDI-MCI FFPE. Each cell shows hit@5 (circle) and precision@5
(diamond) with 95 % intervals and the number of queries (n). decision 15:32: chance
shown as grey ticks only (no chance column); no open-marker rule — instead the
interval is omitted when fewer than 10 resampling units contributed (the point and
n are still shown); the CCDI-MCI all-preparation sensitivity moves to the legend /
Table S14. Public collection names only; no internal vocabulary.

Sources (accepted; read-only):
  E20  $PROSPER_OUT/r1_3_external_synthesis_20260930/broad_group_primary.csv
       configuration = whole_slide_ot, analysis_role = main; rows (stv, main_696),
       (kidrare, amended_1581), (ccdi, ffpe_scope_clear); macro + group rows for
       hit_at_5 and precision_at_5.
  S30  $PROSPER_OUT/r3_2_s30_chance_20261006/table_s30.csv
       chance_hit_at_5 and library_share per group (identical across collections;
       asserted); the group hit@5 points are asserted equal to S30 observed values.

Outputs  $PROSPER_OUT/figures_20261021/fig4_additional_collections/
         Figure_4_revised.{png,pdf}, source_data.csv, legibility_check.json,
         SHA256SUMS.txt, PROVENANCE.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GREY, HIT_COLOR, PREC_COLOR, out_dir, prosper_out, save, style  # noqa: E402

SRC = prosper_out() / "r1_3_external_synthesis_20260930/broad_group_primary.csv"
S30 = prosper_out() / "r3_2_s30_chance_20261006/table_s30.csv"
FIGURE = "fig4_additional_collections"
WIDTH_IN = 180 / 25.4
MIN_FONT_PT = 6.0

COLS = [("stv", "main_696", "STV"), ("kidrare", "amended_1581", "KidRare"),
        ("ccdi", "ffpe_scope_clear", "CCDI-MCI, FFPE")]
S30_NAME = {"stv": "STV", "kidrare": "KidRare", "ccdi": "CCDI-MCI FFPE"}
LIGHT_GREY = "#bdbdbd"
MIN_UNITS = 10
MACRO_LABEL = "All groups\n(macro average)"


def load():
    b = pd.read_csv(SRC)
    b = b[(b.configuration == "whole_slide_ot") & (b.analysis_role == "main")
          & (b.metric.isin(["hit_at_5", "precision_at_5"])) & (b.tumor_group != "micro")]
    g = pd.concat([b[(b.cohort == c) & (b.population == p)] for c, p, _ in COLS])
    assert len(g) == 2 * (31 + 3), len(g)  # (15 + 4 + 12 groups + 3 macro) x 2 metrics
    s30 = pd.read_csv(S30)
    lib = s30[["tumor_group", "n_library_slides", "library_share", "chance_hit_at_5"]].drop_duplicates()
    assert len(lib) == 15, "chance/library share must be identical across collections"
    for c, p, _ in COLS:
        sub = g[(g.cohort == c) & (g.metric == "hit_at_5") & (g.tumor_group != "macro")].set_index("tumor_group")
        ref = s30[s30.collection == S30_NAME[c]].set_index("tumor_group")
        for grp, r in sub.iterrows():
            assert abs(float(ref.loc[grp, "observed_hit_at_5"]) - r.point) < 1e-9, (c, grp)
    lib = lib.sort_values(["library_share", "tumor_group"], ascending=[False, True]).reset_index(drop=True)
    return g, lib


def grid(axes, g, lib):
    groups = lib.tumor_group.tolist()
    rows = ["macro"] + groups
    y = np.arange(len(rows))[::-1].astype(float)
    y[0] += 0.5  # separate the macro row
    chance = dict(zip(lib.tumor_group, lib.chance_hit_at_5))
    ypos = dict(zip(rows, y))
    for ax, (c, p, lab) in zip(axes, COLS):
        sub = g[g.cohort == c]
        for gr in rows:
            yi = ypos[gr]
            if gr != "macro":
                ax.plot([chance[gr]] * 2, [yi - 0.32, yi + 0.32], color=LIGHT_GREY, linewidth=1.2,
                        solid_capstyle="butt", zorder=1)
            cell = sub[sub.tumor_group == gr]
            if cell.empty:
                continue
            n_units = int(cell.n_resampling_units.iloc[0])
            for metric, color, marker, dy, ms in [("hit_at_5", HIT_COLOR, "o", 0.15, 3.8),
                                                  ("precision_at_5", PREC_COLOR, "D", -0.15, 3.2)]:
                r = cell[cell.metric == metric].iloc[0]
                if n_units >= MIN_UNITS:
                    ax.errorbar(r.point, yi + dy, xerr=[[r.point - r.ci_lo], [r.ci_hi - r.point]], fmt=marker,
                                color=color, ms=ms, capsize=1.4, elinewidth=0.7, mew=0.8, zorder=3)
                else:
                    ax.plot(r.point, yi + dy, marker=marker, color=color, ms=ms, linestyle="none", zorder=3)
            ax.text(1.06, yi, f"{int(cell.n_slides.iloc[0]):,}", fontsize=6, color="#555555", va="center",
                    ha="left", clip_on=False)
        ax.set_title(lab, fontsize=7.5)
        ax.text(1.06, y[0] + 0.78, "n", fontsize=6, color="#555555", va="center", ha="left", clip_on=False,
                style="italic")
        ax.set_yticks(y)
        if ax is not axes[0]:
            ax.tick_params(axis="y", length=0, labelleft=False)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_ylim(-0.6, y[0] + 0.6)
        ax.grid(axis="x", linewidth=0.3, alpha=0.5)
        ax.axhline(y[0] - 0.75, color="#999999", linewidth=0.6, zorder=0)
        for yi in y[2:]:
            ax.axhline(yi + 0.5, color="#eeeeee", linewidth=0.4, zorder=0)
    axes[0].set_yticklabels([MACRO_LABEL] + groups)
    axes[1].set_xlabel("Retrieval metric (95% CI)")
    axes[0].set_ylabel("Tumor group, ordered by share of the reference library")
    h = [plt.Line2D([], [], marker="o", color=HIT_COLOR, linestyle="none", ms=4),
         plt.Line2D([], [], marker="D", color=PREC_COLOR, linestyle="none", ms=3.5),
         plt.Line2D([], [], color=LIGHT_GREY, linewidth=1.5)]
    # revision wording (2026-11-10): "chance hit@5 (random retrieval)"
    axes[1].legend(h, ["hit@5", "precision@5", "chance hit@5 (random retrieval)"],
                   loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=3, frameon=False, handletextpad=0.4,
                   columnspacing=1.6)


def legibility(fig, d: Path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible()]
    boxes, rows = [], []
    for t in texts:
        bb = t.get_window_extent(renderer)
        if bb.width <= 0 or bb.height <= 0:
            continue
        rows.append(dict(text=t.get_text(), fontsize_pt=float(t.get_fontsize()),
                         font_ok=bool(t.get_fontsize() >= MIN_FONT_PT)))
        boxes.append((t.get_text(), bb.x0, bb.y0, bb.x1, bb.y1))
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[1] < b[3] - 0.5 and b[1] < a[3] - 0.5 and a[2] < b[4] - 0.5 and b[2] < a[4] - 0.5:
                overlaps.append([a[0], b[0]])
    w, h = fig.get_size_inches()
    check = dict(width_mm=round(w * 25.4, 1), height_mm=round(h * 25.4, 1), min_font_pt_required=MIN_FONT_PT,
                 n_texts=len(rows), all_fonts_ok=all(r["font_ok"] for r in rows), overlaps=overlaps, texts=rows)
    check["pass"] = bool(check["all_fonts_ok"] and not overlaps)
    (d / "legibility_check.json").write_text(json.dumps(check, indent=1))
    return check


def main():
    style()
    g, lib = load()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 5.6), sharey=True,
                             gridspec_kw=dict(wspace=0.28, left=0.17, right=0.965, top=0.955, bottom=0.11))
    grid(axes, g, lib)
    d = out_dir(FIGURE)
    check = legibility(fig, d)
    src = g.merge(lib, on="tumor_group", how="left")[
        ["tumor_group", "n_library_slides", "library_share", "chance_hit_at_5", "cohort", "population", "metric",
         "point", "ci_lo", "ci_hi", "n_slides", "n_resampling_units", "resampling_unit"]].copy()
    src["interval_drawn"] = src.n_resampling_units >= MIN_UNITS
    src = src.sort_values(["cohort", "library_share", "tumor_group", "metric"], ascending=[True, False, True, True])
    note = ("Primary all-tissue UNI2 + exact Optimal Transport against the fixed COMET library (E20 main rows). "
            "Intervals: percentile 95 % from 5,000 class-stratified resamples (complete STV patients, singleton CCDI "
            "participants, nominal KidRare slides); drawn only where n_resampling_units >= "
            f"{MIN_UNITS} (interval_drawn column). Rows ordered by library share (Table S30); chance tick = "
            "1 - (1 - share)^5 as tabulated in Table S30. n printed = number of queries (n_slides). The CCDI-MCI "
            "all-preparation sensitivity (E20 preparation_sensitivity row) is not drawn; it is quoted in the legend. "
            f"Legibility check pass = {check['pass']} (min font {MIN_FONT_PT} pt, no overlaps).")
    save(fig, FIGURE, "Figure_4_revised", [SRC, S30], note, {"source_data": src})
    print("legibility", check["pass"], "overlaps", check["overlaps"])


if __name__ == "__main__":
    main()
