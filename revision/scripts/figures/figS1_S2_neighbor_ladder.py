"""Supplementary Figs. S1 and S2, re-derived for the main configuration (D1/D5, 2026-10-21).

Replaces the submitted Virchow2/SAMPLER "disease-shortlist F1" figures (same-patient slides in
the library) with the main configuration (UNI2, Optimal Transport), every slide of the query's
patient excluded, 2,778 all-tissue COMET queries from 1,876 patients, 15 groups.

Source (read-only, gated derivation `scripts/r4b_s1s2_neighbor_ladder.py`, all gates pass):
  $PROSPER_OUT/figS1S2_neighbor_ladder_20261021/run_001/
    s1_hitk_per_group.csv      per-group hit@K (K = 1, 3, 5, 10, 15, 20) with whole-patient 95 % CIs
    s2_diversity_per_group.csv per-group mean number of distinct group labels among the K nearest
    s2_fit.csv                 OLS across the 15 groups per K (R^2, two-sided p, Spearman)

Faces (house style of Figs. 2-4: no internal prose, n beside rows, intervals or an explicit
statement in the legend):
  S1  heat-map, rows = groups ordered by hit@1 (descending), columns = K, cell = hit@K; a right-hand
      column gives n slides / n patients. Intervals are in the source table (too many cells to draw).
  S2  2 x 2 panels (K = 5, 10, 15, 20): x = mean distinct labels among the K nearest neighbors,
      y = hit@K with its whole-patient 95 % CI, dashed OLS line, points numbered in the Fig. S1 row
      order with a key column, R^2 in the corner; shared axis limits.

Outputs  $PROSPER_OUT/figures_20261021/figS1_S2_neighbor_ladder/
         Figure_S1_revised.{png,pdf}, Figure_S2_revised.{png,pdf}, source_s1.csv, source_s2.csv,
         source_s2_fit.csv, legibility_check_S1.json, legibility_check_S2.json, SHA256SUMS.txt, PROVENANCE.md
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
from _common import GREY, HIT_COLOR, out_dir, prosper_out, save, style, write_provenance  # noqa: E402

RUN = prosper_out() / "figS1S2_neighbor_ladder_20261021/run_001"
S1 = RUN / "s1_hitk_per_group.csv"
S2 = RUN / "s2_diversity_per_group.csv"
FIT = RUN / "s2_fit.csv"
GATES = RUN / "COMPLETE.json"
FIGURE = "figS1_S2_neighbor_ladder"
WIDTH_IN = 180 / 25.4
MIN_FONT_PT = 6.0
KS = [1, 3, 5, 10, 15, 20]
KS_S2 = [5, 10, 15, 20]


def load():
    if not GATES.exists():
        raise SystemExit(f"derivation did not COMPLETE: {RUN}")
    s1 = pd.read_csv(S1)
    s2 = pd.read_csv(S2)
    fit = pd.read_csv(FIT)
    assert set(s1.k) == set(KS) and set(s2.k) == set(KS_S2)
    assert s1[s1.average == "group"].group.nunique() == 15
    return s1, s2, fit


def visible_texts(fig):
    """Texts that are actually drawn: figure/axes texts, axis labels, and major tick labels whose
    tick lies inside the view limits on an axis whose labels are on (findobj also returns
    off-view and hidden tick labels)."""
    out = list(fig.texts)
    for ax in fig.axes:
        if not ax.axison:
            out += list(ax.texts)
            continue
        out += list(ax.texts) + [ax.xaxis.label, ax.yaxis.label] + list(ax.get_legend().get_texts() if ax.get_legend() else [])
        for axis, lim, side in ((ax.xaxis, sorted(ax.get_xlim()), ("labelbottom", "labeltop")),
                                (ax.yaxis, sorted(ax.get_ylim()), ("labelleft", "labelright"))):
            prm = axis.get_tick_params()
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                if lim[0] <= loc <= lim[1]:
                    if prm.get(side[0], True) and tick.label1.get_visible():
                        out.append(tick.label1)
                    if prm.get(side[1], False) and tick.label2.get_visible():
                        out.append(tick.label2)
    return [t for t in out if t.get_text().strip() and t.get_visible()]


def legibility(fig, d: Path, tag: str):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = visible_texts(fig)
    boxes, rows = [], []
    for t in texts:
        bb = t.get_window_extent(renderer)
        if bb.width <= 0 or bb.height <= 0:
            continue
        rows.append(dict(text=t.get_text(), fontsize_pt=float(t.get_fontsize()), font_ok=bool(t.get_fontsize() >= MIN_FONT_PT)))
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
    (d / f"legibility_check_{tag}.json").write_text(json.dumps(check, indent=1))
    return check


def draw_s1(s1: pd.DataFrame):
    g = s1[s1.average == "group"]
    order = g[g.k == 1].sort_values("hit", ascending=False).group.tolist()
    M = g.pivot(index="group", columns="k", values="hit").loc[order, KS]
    n_sl = g.groupby("group").n_slides.first().loc[order]
    n_pt = g.groupby("group").n_patients.first().loc[order]
    fig, (ax, axn) = plt.subplots(1, 2, figsize=(WIDTH_IN * 0.62, 4.0),
                                  gridspec_kw=dict(width_ratios=[6, 1.6], wspace=0.04, left=0.14, right=0.985,
                                                   top=0.92, bottom=0.08))
    im = ax.imshow(M.values, cmap="Blues", vmin=0.2, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.72 else "black")
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("K (nearest neighbors)")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    axn.set_xlim(0, 1)
    axn.set_ylim(len(order) - 0.5, -0.5)
    for i, grp in enumerate(order):
        axn.text(0.05, i, f"{int(n_sl[grp])} / {int(n_pt[grp])}", ha="left", va="center", fontsize=6.5)
    axn.text(0.05, -0.95, "slides / patients", ha="left", va="center", fontsize=6.5, color=GREY)
    axn.axis("off")
    cb = fig.colorbar(im, ax=[ax, axn], fraction=0.035, pad=0.02, shrink=0.6, location="bottom", anchor=(0.3, 1.0))
    cb.set_label("hit@K", fontsize=6.5)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    return fig, M.reset_index().melt(id_vars="group", var_name="k", value_name="hit").merge(
        g[["group", "k", "n_slides", "n_patients", "ci_lo", "ci_hi", "flag"]], on=["group", "k"])


def repel(ax, pts, labels, fontsize=6.0, pad=1.0, iters=200):
    """Place short labels right of each point; push vertically apart in display space."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [ax.annotate(lab, xy, xytext=(3.5, 0), textcoords="offset points", fontsize=fontsize, ha="left",
                         va="center", zorder=4) for xy, lab in zip(pts, labels)]
    for _ in range(iters):
        moved = False
        bbs = [t.get_window_extent(r) for t in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                a, b = bbs[i], bbs[j]
                if a.x0 < b.x1 + pad and b.x0 < a.x1 + pad and a.y0 < b.y1 + pad and b.y0 < a.y1 + pad:
                    ov = min(a.y1, b.y1) - max(a.y0, b.y0) + pad
                    up, dn = (i, j) if a.y0 + a.y1 >= b.y0 + b.y1 else (j, i)
                    for t, sgn in ((texts[up], 1), (texts[dn], -1)):
                        dx, dy = t.get_position()
                        t.set_position((dx, dy + sgn * ov / 2 * 72 / fig.dpi))
                    moved = True
        if not moved:
            break
        for t in texts:
            t.update_positions(r) if hasattr(t, "update_positions") else None
    return texts


def draw_s2(s1: pd.DataFrame, s2: pd.DataFrame, fit: pd.DataFrame):
    g = s1[s1.average == "group"]
    order = g[g.k == 1].sort_values("hit", ascending=False).group.tolist()  # same order as Fig. S1 rows
    num = {grp: i + 1 for i, grp in enumerate(order)}
    d = s2.merge(g[["group", "k", "ci_lo", "ci_hi"]], on=["group", "k"])
    d["label_no"] = d.group.map(num)
    fig = plt.figure(figsize=(WIDTH_IN, 4.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.34], wspace=0.12, hspace=0.3, left=0.065, right=0.99,
                          top=0.97, bottom=0.1)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    key = fig.add_subplot(gs[:, 2])
    xlim = (0.8, d.mean_distinct.max() + 0.4)
    ylim = (0.15, 1.03)
    for ax, k in zip(axes, KS_S2):
        dk = d[d.k == k]
        f = fit[fit.k == k].iloc[0]
        xs = np.linspace(*xlim, 50)
        ax.plot(xs, f.intercept + f.slope * xs, ls="--", lw=0.8, color=GREY, zorder=1)
        ax.errorbar(dk.mean_distinct, dk.hit, yerr=[dk.hit - dk.ci_lo, dk.ci_hi - dk.hit], fmt="o", ms=3.0,
                    color=HIT_COLOR, ecolor="#b0b0b0", elinewidth=0.7, capsize=0, zorder=3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axhline(1.0, color="#dddddd", lw=0.5, zorder=0)
        ax.text(0.03, 0.05, f"K = {k}\nR² = {f.r2:.2f}", transform=ax.transAxes, fontsize=6.5, ha="left", va="bottom")
        repel(ax, list(zip(dk.mean_distinct, dk.hit)), [str(n) for n in dk.label_no])
    for ax in axes[2:]:
        ax.set_xlabel("Mean distinct labels among K nearest")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("hit@K")
    for ax in (axes[1], axes[3]):
        ax.tick_params(labelleft=False)
    key.axis("off")
    key.set_xlim(0, 1)
    key.set_ylim(len(order) + 0.5, -0.5)
    for grp in order:
        key.text(0.05, num[grp], f"{num[grp]:>2}  {grp}", fontsize=6.5, ha="left", va="center")
    return fig, d


def main():
    style()
    s1, s2, fit = load()
    out = out_dir(FIGURE)
    written: list[Path] = []

    fig1, src1 = draw_s1(s1)
    c1 = legibility(fig1, out, "S1")
    for ext in ("png", "pdf"):
        p = out / f"Figure_S1_revised.{ext}"
        fig1.savefig(p, dpi=300, bbox_inches="tight", facecolor="white",
                     **({"metadata": {"CreationDate": None}} if ext == "pdf" else {}))
        written.append(p)
    src1.to_csv(out / "source_s1.csv", index=False)
    written.append(out / "source_s1.csv")

    fig2, src2 = draw_s2(s1, s2, fit)
    c2 = legibility(fig2, out, "S2")
    for ext in ("png", "pdf"):
        p = out / f"Figure_S2_revised.{ext}"
        fig2.savefig(p, dpi=300, bbox_inches="tight", facecolor="white",
                     **({"metadata": {"CreationDate": None}} if ext == "pdf" else {}))
        written.append(p)
    src2.to_csv(out / "source_s2.csv", index=False)
    fit.to_csv(out / "source_s2_fit.csv", index=False)
    written += [out / "source_s2.csv", out / "source_s2_fit.csv", out / "legibility_check_S1.json",
                out / "legibility_check_S2.json"]

    allrows = s1[s1.group == "ALL"]
    note = ("Main configuration (UNI2, Optimal Transport cosine), library excludes every slide of the query's patient; "
            "2,778 all-tissue COMET queries / 1,876 patients / 15 groups incl. NORMAL. hit@K = share of a group's queries "
            "whose K nearest other-patient slides include >= 1 slide of the query's group. Whole-patient bootstrap "
            "intervals (5,000 draws, seed 123, shared with R4.a-vii). Derivation gates: depth-10 lists identical to the "
            "frozen c2_metric_suite masked lists for all 2,778 queries; per-query hit@{1,3,5,10} identical to the accepted "
            "r4a_vii c1e_masked frame; population intervals identical to the accepted summary rows (1e-16); no rank-20/21 ties. "
            "Population hit@K (micro): " + "; ".join(f"K={int(r.k)} {r.hit:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]" for r in allrows[allrows.average == 'micro'].itertuples())
            + ". S2 fits (OLS over 15 groups, descriptive): " + "; ".join(f"K={int(r.k)} R2={r.r2:.2f} p={r.p_slope_two_sided:.3g} rho={r.spearman_rho:.2f}" for r in fit.itertuples())
            + ". hit@K approaches its ceiling at large K, which contributes to the weakening association. "
            f"Legibility S1 pass={c1['pass']} overlaps={c1['overlaps']}; S2 pass={c2['pass']} overlaps={c2['overlaps']}.")
    write_provenance(out, written, [S1, S2, FIT, GATES], note, Path(sys.argv[0]), title=FIGURE)
    print("S1 legibility", c1["pass"], c1["overlaps"])
    print("S2 legibility", c2["pass"], c2["overlaps"])
    for p in written[:2] + written[4:6]:
        print("wrote", p)


if __name__ == "__main__":
    main()
