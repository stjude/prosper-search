"""Supplementary Fig. S-R2.1-F5 (simulated query perturbations): redesign candidates, 2026-10-31.

the author found the adopted render (`figS_perturbation.py` -> figures_20261021/figS_perturbation/,
PNG 5c076b2e...) "not quite satisfactory". This script renders clearly different candidate
designs from the SAME retained source CSVs, one directory per candidate (panel-by-panel
convention of `fig8_reader_study.py`), plus a compare sheet. It never touches the adopted
render or its script; it imports `load()` and `legibility()` from `figS_perturbation.py` so the
108 macro rows and their v002 == mMV-run assertion are exactly the adopted figure's.

Candidates
  forest  Refined forest. Same estimand and 108 intervals as the adopted render, but: each arm
          carries its stimulus/dose on the face; a grey band marks the half-width of the
          unperturbed value's own 95 % interval (the sampling uncertainty of the baseline, a
          yardstick, not an equivalence margin); filled markers = interval excludes zero, open =
          covers zero; stain-mismatch row tinted; unperturbed values printed under each title.
  matrix  One-glance heat map: 12 arms x (3 metrics x 3 pipelines), diverging colour for the
          macro change; the value is printed only in cells whose interval excludes zero.
  detail  What the macro view cannot show. A: per-group change in hit@5 and precision@5
          (15 groups x 12 arms, one heat map per pipeline; dot = interval excludes zero).
          B: realised-dose strata (lowest / middle / highest third of the realised stimulus
          strength) for the four randomly dosed arms, macro hit@5 and precision@5 change;
          open markers where the stored stratum interval is unavailable.

Sources (read-only; nothing recomputed beyond display arithmetic: x100, CI half-width, sorting)
  $PROSPER_OUT/r2_1_fig4a_20261011/v002/fig4a_source_data.csv          (via figS_perturbation.load)
  $PROSPER_OUT/r2_1_fig4a_mmv5_20261021/run_001/{fig_source,identity}.csv (via load)
  $PROSPER_OUT/r2_1_fig4a_20261011/v002/supp_perturbation_tables.csv    (per_group, stratum blocks;
          the Table S-R2.1-C source)
  $PROSPER_OUT/r2_1_p2_eval_20261006/dose_summary_001/stimulus_distribution.csv (published ranges
          and tertile cut points, labels only; cross-checked against the stratum n per arm)

Outputs  $PROSPER_OUT/figures_20261021/figS_perturbation/candidates_20261031/<name>/
           Figure_S_perturbation_<name>.{png,pdf}, source_data.csv, legibility_check.json,
           SHA256SUMS.txt, PROVENANCE.md
         $PROSPER_OUT/figures_20261021/figS_perturbation/candidates_20261031/compare_20261031.png

Palette: categorical slots 1-3 of the dataviz reference palette (#2a78d6, #eb6834, #1baf7a;
validator 2026-10-31: all-pairs CVD dE >= 9.2, normal-vision >= 24.0; aqua < 3:1 contrast ->
relief by marker shape + legend + source_data.csv). Diverging: red (decrease) <-> grey <-> blue.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import out_dir, prosper_out, save, sha256, style  # noqa: E402
import figS_perturbation as base  # noqa: E402  (load(), legibility(), FAMILIES; no side effects on import)

SUPP = prosper_out() / "r2_1_fig4a_20261011/v002/supp_perturbation_tables.csv"
STIM = prosper_out() / "r2_1_p2_eval_20261006/dose_summary_001/stimulus_distribution.csv"
ADOPTED_PNG = prosper_out() / "figures_20261021/figS_perturbation/Figure_S_perturbation.png"
ROOT = "figS_perturbation/candidates_20261031"
WIDTH_IN = 180 / 25.4

PIPES = [("hopt", "delta:hopt", "H-Optimus-1 + SAMPLER", "o", "#2a78d6"),
         ("uni2_sampler", "delta:uni2_sampler", "UNI2 + SAMPLER", "s", "#eb6834"),
         ("virchow2_sampler", "delta:virchow2_sampler", "Virchow2 + SAMPLER", "^", "#1baf7a")]
PIPE_SHORT = {"hopt": "H-Optimus-1", "uni2_sampler": "UNI2", "virchow2_sampler": "Virchow2"}
METRICS = [("hit@5", "hit", "hit@5"), ("mMV@5", "mmv", "majority-vote accuracy"),
           ("precision@5", "precision", "precision@5")]
ARMS = [a for _, fam in base.FAMILIES for a, _ in fam]
LABEL = {a: lab for _, fam in base.FAMILIES for a, lab in fam}
GRADED = ["brightness", "saturation", "contrast", "spatial_shift"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
DIVERGING = LinearSegmentedColormap.from_list("red_grey_blue", ["#b42828", "#e34948", "#f0efec", "#2a78d6", "#184f95"])
STAIN_TINT = "#fbeaea"
SHORT = {"rotate_90": "Rotation 90°", "rotate_180": "Rotation 180°", "rotate_270": "Rotation 270°",
         "flip_horizontal": "Flip horiz.", "flip_vertical": "Flip vert.", "spatial_shift": "Shift",
         "brightness": "Brightness", "saturation": "Saturation", "contrast": "Contrast",
         "gaussian_blur": "Blur (input)", "gaussian_blur_image": "Blur (tile)", "reinhard": "Stain norm."}


# ----------------------------------------------------------------------------------------------- data
def dose_labels(stim: pd.DataFrame) -> dict[str, str]:
    s = stim.set_index("arm")
    for arm, lo, hi in (("brightness", 0.85, 1.15), ("saturation", 0.85, 1.15), ("contrast", 0.9, 1.1)):
        assert s.loc[arm, "published"] == f"uniform({lo},{hi})" or s.loc[arm, "published"] == f"uniform({lo:.2f},{hi:.2f})" \
            or s.loc[arm, "published"].replace(" ", "") == f"uniform({lo},{hi})", s.loc[arm, "published"]
    assert s.loc["spatial_shift", "published"].startswith("randint(-16,16)")
    return {
        "rotate_90": "exact", "rotate_180": "exact", "rotate_270": "exact",
        "flip_horizontal": "exact", "flip_vertical": "exact",
        "spatial_shift": "x, y offset −16 to 16 px",
        "brightness": "factor 0.85–1.15", "saturation": "factor 0.85–1.15", "contrast": "factor 0.90–1.10",
        "gaussian_blur": "3 × 3, σ 0.5, 224-px input", "gaussian_blur_image": "7 × 7, σ 8/7, 512-px tile",
        "reinhard": "query side only",
    }


def load_all():
    d, ident = base.load()  # asserts 108 rows and v002 == mMV-run hit/precision rows (1e-9)
    supp = pd.read_csv(SUPP)
    stim = pd.read_csv(STIM)
    # per-group block, k = 5; H-Optimus-1 columns are duplicated across the two comparator rows
    pg = supp[(supp.table_block == "per_group") & (supp.k == 5) & (supp.population == "full")]
    assert len(pg) == 2 * 12 * 15 * 2, len(pg)
    a = pg[pg.comparator == "uni2_sampler"].set_index(["condition", "group", "metric"])
    b = pg[pg.comparator == "virchow2_sampler"].set_index(["condition", "group", "metric"])
    for c in ("delta_hopt", "delta_hopt_ci_low", "delta_hopt_ci_high"):
        assert (a[c] - b[c]).abs().max() < 1e-12, c
    rows = []
    for pipe, src, cols in (("hopt", a, "hopt"), ("uni2_sampler", a, "c"), ("virchow2_sampler", b, "c")):
        for (cond, grp, met), r in src.iterrows():
            rows.append(dict(pipeline=pipe, arm=cond, group=grp, metric=met, point=r[f"delta_{cols}"],
                             ci_low=r[f"delta_{cols}_ci_low"], ci_high=r[f"delta_{cols}_ci_high"],
                             n_slides=int(r.n_pairs), n_patients=int(r.n_patients), ci_status=r.ci_status))
    groups = pd.DataFrame(rows)
    groups["excludes_zero"] = (groups.ci_low > 0) | (groups.ci_high < 0)
    # macro consistency: per-group block's macro (equal-weight mean of groups) == fig4a primary rows
    for pipe, panel, *_ in PIPES:
        for met, ep in (("hit", "hit@5"), ("precision", "precision@5")):
            g = groups[(groups.pipeline == pipe) & (groups.metric == met)].groupby("arm").point.mean()
            p = d[(d.panel == panel) & (d.endpoint == ep)].set_index("arm").point
            assert (g.reindex(p.index) - p).abs().max() < 1e-9, (pipe, met)
    # realised-dose strata, MACRO, k = 5
    st = supp[(supp.table_block == "stratum") & (supp.k == 5) & (supp.group == "MACRO")]
    assert len(st) == 2 * 4 * 3 * 2, len(st)
    a = st[st.comparator == "uni2_sampler"].set_index(["population", "metric"])
    b = st[st.comparator == "virchow2_sampler"].set_index(["population", "metric"])
    assert (a.delta_hopt - b.delta_hopt).abs().max() < 1e-12
    srows = []
    for pipe, src, cols in (("hopt", a, "hopt"), ("uni2_sampler", a, "c"), ("virchow2_sampler", b, "c")):
        for (pop, met), r in src.iterrows():
            _, arm, tert = pop.split(":")
            srows.append(dict(pipeline=pipe, arm=arm, tertile=tert, metric=met, point=r[f"delta_{cols}"],
                              ci_low=r[f"delta_{cols}_ci_low"], ci_high=r[f"delta_{cols}_ci_high"],
                              ci_status=r.ci_status, n_slides=int(r.n_pairs)))
    strata = pd.DataFrame(srows)
    sd = stim.set_index("arm")
    for arm in GRADED:  # stratum sizes equal the dose summary's tertile counts
        for t in ("T1", "T2", "T3"):
            n = strata[(strata.arm == arm) & (strata.tertile == t)].n_slides.unique()
            assert list(n) == [int(sd.loc[arm, f"n_{t}"])], (arm, t, n)
    return d, ident, groups, strata, stim


def ident_lookup(ident: pd.DataFrame) -> dict:
    out = {}
    for r in ident.itertuples():
        out[(r.pipeline, r.metric)] = (r.estimate, r.ci_low, r.ci_high)
    return out


# ------------------------------------------------------------------------------------------- forest
def render_forest(d, ident, dose, band: bool = True):
    """`band=False` (revision, design decision 2026-11-10): no grey baseline-uncertainty band. The band was a
    post-hoc yardstick (narrowest unperturbed CI half-width) that read as an equivalence margin; the
    filled/open markers already carry the inferential content. Source data then omit the band column."""
    il = ident_lookup(ident)
    n = len(ARMS)
    y_of = {a: float(n - 1 - i) for i, a in enumerate(ARMS)}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 5.4),
                             gridspec_kw=dict(wspace=0.10, left=0.30, right=0.99, top=0.875, bottom=0.14))
    src = []
    for ax, (ep, mkey, title) in zip(axes, METRICS):
        hw = min((il[(p, mkey)][2] - il[(p, mkey)][1]) / 2 for p, *_ in PIPES)
        if band:
            ax.axvspan(-hw, hw, color="#ecebe7", zorder=0, linewidth=0)
        ax.axvline(0, color=INK, linewidth=0.6, zorder=1)
        ax.axhspan(y_of["reinhard"] - 0.5, y_of["reinhard"] + 0.5, color=STAIN_TINT, zorder=0, linewidth=0)
        sub = d[d.endpoint == ep]
        for (pipe, panel, name, mk, col), dy in zip(PIPES, (0.25, 0.0, -0.25)):
            s = sub[sub.panel == panel].set_index("arm")
            for a in ARMS:
                r = s.loc[a]
                filled = bool(r.excludes_zero)
                ax.plot([r.ci_low, r.ci_high], [y_of[a] + dy] * 2, color=col, linewidth=0.8, zorder=2,
                        solid_capstyle="round")
                ax.plot(r.point, y_of[a] + dy, marker=mk, ms=3.6, mew=0.8, color=col,
                        mfc=col if filled else "white", zorder=3, linestyle="none")
                row = dict(candidate="forest", metric=ep, pipeline=pipe, arm=a, point=r.point, ci_low=r.ci_low,
                           ci_high=r.ci_high, excludes_zero=filled)
                if band:
                    row["baseline_band_halfwidth"] = hw
                src.append(row)
        ax.set_xlim(-0.035, 0.035)
        ax.set_xticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
        ax.set_xticklabels(["−0.03", "", "−0.01", "0", "0.01", "", "0.03"])
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks([y_of[a] for a in ARMS])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        k = 0
        for _, fam in base.FAMILIES[:-1]:
            k += len(fam)
            ax.axhline(n - k - 0.5, color="#d6d5d0", linewidth=0.5, zorder=0)
        base_txt = " · ".join(f"{il[(p, mkey)][0]:.3f}" for p, *_ in PIPES)
        ax.set_title(f"Δ {title}", fontsize=7.5, pad=14, color=INK)
        ax.text(0.5, 1.012, f"unperturbed {base_txt}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6.0, color=INK2)
    axes[0].set_yticklabels([])
    for a in ARMS:
        axes[0].text(-0.0375, y_of[a] + 0.17, LABEL[a], ha="right", va="center", fontsize=6.6, color=INK)
        axes[0].text(-0.0375, y_of[a] - 0.25, dose[a], ha="right", va="center", fontsize=6.0, color=MUTED)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fam_short = {"Geometric": "Geometric", "Color and\nillumination": "Color and\nillumination",
                 "Image quality": "Image\nquality", "Stain": "Stain"}
    for fam_name, fam in base.FAMILIES:
        ys = [y_of[a] for a, _ in fam]
        axes[0].text(-1.05, float(np.mean(ys)), fam_short[fam_name], transform=axes[0].get_yaxis_transform(),
                     ha="center", va="center", fontsize=6.3, fontstyle="italic", color=INK2, rotation=90,
                     linespacing=1.0)
    axes[1].set_xlabel("Change from the unperturbed query (perturbed − unperturbed), macro average")
    h = [plt.Line2D([], [], marker=mk, color=c, mfc=c, linestyle="-", linewidth=0.8, ms=4) for *_, mk, c in PIPES]
    h += [plt.Line2D([], [], marker="o", color=INK2, mfc="white", linestyle="none", ms=4)]
    labs = [p[2] for p in PIPES] + ["open = 95% CI covers 0"]
    if band:
        h.append(plt.Rectangle((0, 0), 1, 1, color="#ecebe7"))
        labs.append("± half-width of unperturbed 95% CI")
    fig.legend(h, labs, loc="lower center", ncol=5, frameon=False, fontsize=6.2, handletextpad=0.4,
               columnspacing=1.2, bbox_to_anchor=(0.6, 0.0))
    desc = ("Refined forest: adopted estimand and intervals; dose on the face; "
            + ("baseline-uncertainty band" if band else "no baseline band (pass 026b)"))
    return fig, pd.DataFrame(src), desc


# ------------------------------------------------------------------------------------------- matrix
def render_matrix(d, ident, dose):
    il = ident_lookup(ident)
    cols = [(ep, mkey, title, pipe, panel) for ep, mkey, title in METRICS for pipe, panel, *_ in PIPES]
    n = len(ARMS)
    M = np.zeros((n, len(cols)))
    E = np.zeros((n, len(cols)), dtype=bool)
    src = []
    for j, (ep, mkey, title, pipe, panel) in enumerate(cols):
        s = d[(d.endpoint == ep) & (d.panel == panel)].set_index("arm")
        for i, a in enumerate(ARMS):
            M[i, j] = s.loc[a, "point"]
            E[i, j] = bool(s.loc[a, "excludes_zero"])
            src.append(dict(candidate="matrix", metric=ep, pipeline=pipe, arm=a, point=M[i, j],
                            ci_low=s.loc[a, "ci_low"], ci_high=s.loc[a, "ci_high"], excludes_zero=E[i, j]))
    lim = 0.02
    assert np.abs(M).max() <= lim, np.abs(M).max()
    fig = plt.figure(figsize=(WIDTH_IN, 4.0))
    ax = fig.add_axes([0.27, 0.17, 0.62, 0.66])
    cax = fig.add_axes([0.915, 0.25, 0.012, 0.46])
    gap = 0.35
    xs = np.array([j + (j // 3) * gap for j in range(len(cols))])
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    for j, x in enumerate(xs):
        for i in range(n):
            ax.add_patch(plt.Rectangle((x - 0.47, n - 1 - i - 0.47), 0.94, 0.94, color=DIVERGING(norm(M[i, j])),
                                       linewidth=0))
            if E[i, j]:
                ax.text(x, n - 1 - i, (f"{M[i, j]:+.3f}" if abs(M[i, j]) >= 0.001 else f"{M[i, j]:+.4f}").replace("-", "−"), ha="center", va="center", fontsize=6.0,
                        color=INK, fontweight="bold")
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([PIPE_SHORT[c[3]] for c in cols], rotation=40, ha="right", rotation_mode="anchor", fontsize=6.2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([LABEL[a] if dose[a] == "exact" else f"{LABEL[a]}  ({dose[a]})" for a in ARMS][::-1], fontsize=6.4)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for b, (ep, mkey, title) in enumerate(METRICS):
        xm = xs[3 * b + 1]
        base_txt = " · ".join(f"{il[(p, mkey)][0]:.3f}" for p, *_ in PIPES)
        ax.text(xm, n - 0.25 + 0.85, f"Δ {title}", ha="center", va="bottom", fontsize=7.2, color=INK)
        ax.text(xm, n - 0.25 + 0.15, f"unperturbed {base_txt}", ha="center", va="bottom", fontsize=6.0, color=INK2)
    k = 0
    for _, fam in base.FAMILIES[:-1]:
        k += len(fam)
        ax.axhline(n - k - 0.5, color="white", linewidth=2.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
    cb = fig.colorbar(sm, cax=cax, ticks=[-0.02, -0.01, 0, 0.01, 0.02])
    cb.ax.set_yticklabels(["−0.02", "−0.01", "0", "0.01", "0.02"], fontsize=6.0)
    cb.outline.set_visible(False)
    cb.set_label("macro change", fontsize=6.2)
    fig.text(0.27, 0.01, "Values printed only where the 95% paired whole-patient interval excludes zero "
             "(unadjusted); all values and intervals in the source table.", fontsize=6.0, color=INK2)
    return fig, pd.DataFrame(src), "Matrix: 12 arms x 9 metric-pipeline cells, value printed where CI excludes 0"


# ------------------------------------------------------------------------------------------- detail
def render_detail(groups, strata, stim):
    order = (groups[(groups.pipeline == "hopt") & (groups.metric == "hit")].groupby("group").n_slides.first()
             .sort_values(ascending=False).index.tolist())
    nslides = groups[(groups.pipeline == "hopt") & (groups.metric == "hit")].groupby("group").n_slides.first()
    fig = plt.figure(figsize=(WIDTH_IN, 7.6))
    gs_top = fig.add_gridspec(2, 3, left=0.10, right=0.905, top=0.925, bottom=0.465, wspace=0.06, hspace=0.12)
    lims = {m: float(np.ceil(groups[groups.metric == m].point.abs().max() * 100) / 100) for m in ("hit", "precision")}
    src = []
    for r_i, (met, title) in enumerate((("hit", "hit@5"), ("precision", "precision@5"))):
        norm = TwoSlopeNorm(vmin=-lims[met], vcenter=0, vmax=lims[met])
        for c_i, (pipe, panel, name, mk, col) in enumerate(PIPES):
            ax = fig.add_subplot(gs_top[r_i, c_i])
            g = groups[(groups.pipeline == pipe) & (groups.metric == met)].set_index(["group", "arm"])
            M = np.array([[g.loc[(gr, a), "point"] for a in ARMS] for gr in order])
            E = np.array([[bool(g.loc[(gr, a), "excludes_zero"]) for a in ARMS] for gr in order])
            ax.imshow(M, cmap=DIVERGING, norm=norm, aspect="auto", interpolation="nearest")
            yy, xx = np.where(E)
            ax.scatter(xx, yy, s=4, color=INK, linewidths=0, zorder=3)
            for gr in order:
                for a in ARMS:
                    rr = g.loc[(gr, a)]
                    src.append(dict(candidate="detail", panel="A", metric=f"{title}", pipeline=pipe, group=gr, arm=a,
                                    point=rr.point, ci_low=rr.ci_low, ci_high=rr.ci_high,
                                    excludes_zero=bool(rr.excludes_zero), n_slides=int(rr.n_slides)))
            ax.set_xticks(range(len(ARMS)))
            ax.set_yticks(range(len(order)))
            ax.tick_params(length=0, pad=1.5)
            if r_i == 1:
                ax.set_xticklabels([SHORT[a] for a in ARMS], rotation=90, fontsize=6.0)
            else:
                ax.set_xticklabels([])
                ax.set_title(name, fontsize=7.0, color=INK, pad=3)
            if c_i == 0:
                ax.set_yticklabels([f"{gr} ({nslides[gr]})" for gr in order], fontsize=6.0)
                ax.set_ylabel(f"Δ {title}", fontsize=7.0)
            else:
                ax.set_yticklabels([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            for xb in (5.5, 8.5, 10.5):
                ax.axvline(xb, color="white", linewidth=1.2)
        cax = fig.add_axes([0.92, 0.735 - r_i * 0.235, 0.010, 0.15])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax,
                          ticks=[-lims[met], 0, lims[met]])
        cb.ax.set_yticklabels([f"−{lims[met]:.2f}", "0", f"{lims[met]:.2f}"], fontsize=6.0)
        cb.outline.set_visible(False)
    fig.text(0.015, 0.962, "A", fontsize=10, fontweight="bold")
    fig.text(0.10, 0.962, "Per-group change, k = 5 (rows: diagnostic group, slides in parentheses; "
             "dot = 95% CI excludes 0, unadjusted)", fontsize=6.5, color=INK2)
    # B: realised-dose strata
    gs_b = fig.add_gridspec(2, 4, left=0.10, right=0.985, top=0.30, bottom=0.07, wspace=0.12, hspace=0.25)
    sd = stim.set_index("arm")
    fig.text(0.015, 0.325, "B", fontsize=10, fontweight="bold")
    fig.text(0.10, 0.325, "Macro change by realised stimulus strength (thirds of the slides; open = interval "
             "not available, too few patients in some groups)", fontsize=6.5, color=INK2)
    unit = {"brightness": "|f − 1|", "saturation": "|f − 1|", "contrast": "|f − 1|", "spatial_shift": "px"}
    for r_i, (met, title) in enumerate((("hit", "hit@5"), ("precision", "precision@5"))):
        for c_i, arm in enumerate(GRADED):
            ax = fig.add_subplot(gs_b[r_i, c_i])
            ax.axhline(0, color=INK, linewidth=0.6)
            c1, c2, mx = sd.loc[arm, "cut_T1_T2"], sd.loc[arm, "cut_T2_T3"], sd.loc[arm, "strength_max"]
            fmt = (lambda v: f"{v:.0f}") if arm == "spatial_shift" else (lambda v: f"{v:.2f}")
            xl = [f"≤{fmt(c1)}", f"{fmt(c1)}–{fmt(c2)}", f"{fmt(c2)}–{fmt(mx)}"]
            for (pipe, panel, name, mk, col), dx in zip(PIPES, (-0.12, 0.0, 0.12)):
                s = strata[(strata.pipeline == pipe) & (strata.arm == arm) & (strata.metric == met)].set_index("tertile")
                xs = np.arange(3) + dx
                pts = [s.loc[t, "point"] for t in ("T1", "T2", "T3")]
                ax.plot(xs, pts, color=col, linewidth=0.7, zorder=2)
                for x, t in zip(xs, ("T1", "T2", "T3")):
                    rr = s.loc[t]
                    ok = rr.ci_status == "estimable"
                    if ok:
                        ax.plot([x, x], [rr.ci_low, rr.ci_high], color=col, linewidth=0.7, zorder=2)
                    ax.plot(x, rr.point, marker=mk, ms=3.2, color=col, mfc=col if ok else "white", mew=0.7, zorder=3)
                    src.append(dict(candidate="detail", panel="B", metric=title, pipeline=pipe, arm=arm, tertile=t,
                                    point=rr.point, ci_low=rr.ci_low, ci_high=rr.ci_high, ci_status=rr.ci_status,
                                    n_slides=int(rr.n_slides)))
            ax.set_ylim(-0.025, 0.025)
            ax.set_yticks([-0.02, 0, 0.02])
            ax.set_xlim(-0.4, 2.4)
            ax.set_xticks(range(3))
            if c_i == 0:
                ax.set_yticklabels(["−0.02", "0", "0.02"], fontsize=6.0)
                ax.set_ylabel(f"Δ {title}", fontsize=6.8)
            else:
                ax.set_yticklabels([])
            if r_i == 0:
                ax.set_title(f"{LABEL[arm]} ({unit[arm]})", fontsize=6.8, color=INK, pad=3)
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(xl, fontsize=6.0)
            ax.grid(axis="y", linewidth=0.3, color="#e4e3df")
    h = [plt.Line2D([], [], marker=mk, color=c, linestyle="-", linewidth=0.7, ms=4) for *_, mk, c in PIPES]
    fig.legend(h, [p[2] for p in PIPES], loc="lower center", ncol=3, frameon=False, fontsize=6.4,
               bbox_to_anchor=(0.55, 0.0))
    return fig, pd.DataFrame(src), "Detail: per-group heat maps (A) and realised-dose strata (B)"


# ------------------------------------------------------------------------------------------- driver
def render(name, fn, inputs):
    fig, src, desc = fn()
    key = f"{ROOT}/{name}"
    d = out_dir(key)
    leg = base.legibility(fig, d)
    note = (f"Candidate `{name}` for Supplementary Fig. S-R2.1-F5 (2026-10-31; not adopted). {desc}. Values copied "
            f"from the retained sources; display arithmetic only. Legibility pass = {leg['pass']}"
            f" (overlaps: {leg['overlaps']}).")
    save(fig, key, f"Figure_S_perturbation_{name}", inputs, note, {"source_data": src})
    with open(d / "SHA256SUMS.txt", "a") as fh:
        fh.write(f"{sha256(d / 'legibility_check.json')}  legibility_check.json\n")
    plt.close(fig)
    print(name, "legibility", leg["pass"], leg["overlaps"][:6])
    return d / f"Figure_S_perturbation_{name}.png"


def compare_sheet(pngs: list[tuple[str, Path]]):
    tiles = []
    W = 1500
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    for label, p in pngs:
        im = Image.open(p).convert("RGB")
        im = im.resize((W, int(im.height * W / im.width)), Image.LANCZOS)
        tile = Image.new("RGB", (W, im.height + 60), "white")
        ImageDraw.Draw(tile).text((10, 12), label, fill="black", font=font)
        tile.paste(im, (0, 60))
        tiles.append(tile)
    cols = 2
    rows = (len(tiles) + 1) // 2
    hs = [max(t.height for t in tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
    sheet = Image.new("RGB", (cols * W + 40, sum(hs) + 20 * (rows + 1)), "white")
    y = 20
    for r in range(rows):
        for c, t in enumerate(tiles[r * cols:(r + 1) * cols]):
            sheet.paste(t, (20 + c * W, y))
        y += hs[r] + 20
    out = out_dir(ROOT) / "compare_20261031.png"
    sheet.save(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="forest,matrix,detail")
    args = ap.parse_args()
    style()
    d, ident, groups, strata, stim = load_all()
    dose = dose_labels(stim)
    inputs = [base.V002, base.MMV, base.MMV_IDENT, SUPP, STIM]
    fns = {"forest": lambda: render_forest(d, ident, dose), "matrix": lambda: render_matrix(d, ident, dose),
           "detail": lambda: render_detail(groups, strata, stim)}
    pngs = [("adopted (5c076b2e, unchanged)", ADOPTED_PNG)]
    for name in [c.strip() for c in args.candidates.split(",") if c.strip()]:
        pngs.append((f"candidate: {name}", render(name, fns[name], inputs)))
    compare_sheet(pngs)


if __name__ == "__main__":
    main()
