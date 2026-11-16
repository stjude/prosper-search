"""Supplementary figure: simulated query perturbations (three retrieval metrics).

Route 1 of 2026-10-21: the perturbation panel leaves main Fig. 4 and becomes a
Supplementary figure showing, per perturbation arm and pipeline, the paired change in
macro hit@5, macro majority-vote accuracy (mMV@5) and macro precision@5.

Sources (read-only):
  v002  $PROSPER_OUT/r2_1_fig4a_20261011/v002/fig4a_source_data.csv
        accepted hit@5 / precision@5 deltas (panels delta:hopt, delta:uni2_sampler,
        delta:virchow2_sampler); the D contrasts of that file stay in Table S-R2.1-C.
  mmv   $PROSPER_OUT/r2_1_fig4a_mmv5_20261021/run_001/fig_source.csv
        mMV@5 deltas from the gated derivation (`scripts/r2_1_fig4a_mmv5.py`); the
        file's hit@5 / precision@5 rows must equal the v002 rows (asserted, 1e-9).

Face: three columns (hit@5 | mMV@5 | precision@5), twelve arms grouped by perturbation
family, three pipelines per arm (distinct marker and colour), 95 % paired whole-patient
bootstrap intervals, one shared x-scale, zero line. No internal vocabulary; the
unperturbed values and all caveats go in the legend.

Outputs  $PROSPER_OUT/figures_20261021/figS_perturbation/
         Figure_S_perturbation.{png,pdf}, source_data.csv, legibility_check.json,
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
from _common import out_dir, prosper_out, save, style  # noqa: E402

V002 = prosper_out() / "r2_1_fig4a_20261011/v002/fig4a_source_data.csv"
MMV = prosper_out() / "r2_1_fig4a_mmv5_20261021/run_001/fig_source.csv"
MMV_IDENT = prosper_out() / "r2_1_fig4a_mmv5_20261021/run_001/identity.csv"
FIGURE = "figS_perturbation"
WIDTH_IN = 180 / 25.4
MIN_FONT_PT = 6.0
XLIM = 0.035

PIPES = [("delta:hopt", "H-Optimus-1 + SAMPLER", "o", "#1f77b4", 0.26),
         ("delta:uni2_sampler", "UNI2 + SAMPLER", "s", "#2ca02c", 0.0),
         ("delta:virchow2_sampler", "Virchow2 + SAMPLER", "^", "#9467bd", -0.26)]
FAMILIES = [
    ("Geometric", [("rotate_90", "Rotation 90°"), ("rotate_180", "Rotation 180°"), ("rotate_270", "Rotation 270°"),
                   ("flip_horizontal", "Horizontal flip"), ("flip_vertical", "Vertical flip"),
                   ("spatial_shift", "Spatial shift")]),
    ("Color and\nillumination", [("brightness", "Brightness"), ("saturation", "Saturation"), ("contrast", "Contrast")]),
    ("Image quality", [("gaussian_blur", "Blur (encoder input)"), ("gaussian_blur_image", "Blur (source tile)")]),
    ("Stain", [("reinhard", "Stain normalization")]),
]
METRICS = [("hit@5", "Δ hit@5"), ("mMV@5", "Δ majority-vote accuracy"), ("precision@5", "Δ precision@5")]


def load():
    v = pd.read_csv(V002)
    v = v[v.panel.isin([p for p, *_ in PIPES])].copy()
    m = pd.read_csv(MMV)
    chk = v.merge(m, on=["endpoint", "panel", "arm"], suffixes=("", "_m"))
    assert len(chk) == 72, len(chk)
    for col in ("point", "ci_low", "ci_high"):
        assert (chk[col] - chk[f"{col}_m"]).abs().max() < 1e-9, col
    d = pd.concat([v[["endpoint", "panel", "arm", "point", "ci_low", "ci_high", "excludes_zero"]],
                   m[m.endpoint == "mMV@5"][["endpoint", "panel", "arm", "point", "ci_low", "ci_high", "excludes_zero"]]])
    d["source"] = np.where(d.endpoint == "mMV@5", "r2_1_fig4a_mmv5_20261021/run_001", "r2_1_fig4a_20261011/v002")
    assert len(d) == 108
    ident = pd.read_csv(MMV_IDENT)
    ident = ident[ident.group == "MACRO"]
    return d.reset_index(drop=True), ident


def draw(axes, d):
    arms = [a for _, fam in FAMILIES for a, _ in fam]
    labels = [lab for _, fam in FAMILIES for _, lab in fam]
    y_of = {a: float(len(arms) - 1 - i) for i, a in enumerate(arms)}
    for ax, (metric, title) in zip(axes, METRICS):
        sub = d[d.endpoint == metric]
        ax.axvline(0, color="black", linewidth=0.6, zorder=1)
        for panel, name, marker, color, dy in PIPES:
            s = sub[sub.panel == panel].set_index("arm")
            for a in arms:
                r = s.loc[a]
                ax.errorbar(r.point, y_of[a] + dy, xerr=[[r.point - r.ci_low], [r.ci_high - r.point]], fmt=marker,
                            color=color, ms=3.2, capsize=1.3, elinewidth=0.7, mew=0.7, zorder=3)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlim(-XLIM, XLIM)
        ax.set_xticks([-0.02, 0, 0.02])
        ax.set_xticklabels(["−0.02", "0", "0.02"])
        ax.set_ylim(-0.7, len(arms) - 0.3)
        ax.set_yticks([y_of[a] for a in arms])
        ax.grid(axis="x", linewidth=0.3, alpha=0.5)
        ax.tick_params(axis="y", length=0)
        # family separators
        k = 0
        for fam_name, fam in FAMILIES[:-1]:
            k += len(fam)
            ax.axhline(len(arms) - k - 0.5, color="#dddddd", linewidth=0.6, zorder=0)
    axes[0].set_yticklabels(labels)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    # family headers in the left margin
    k = 0
    for fam_name, fam in FAMILIES:
        ys = [y_of[a] for a, _ in fam]
        axes[0].text(-0.74, float(np.mean(ys)), fam_name, transform=axes[0].get_yaxis_transform(), ha="right",
                     va="center", fontsize=6.5, fontstyle="italic", color="#444444")
        k += len(fam)
    axes[1].set_xlabel("Change from the unperturbed query (perturbed − unperturbed), macro average with 95% CI")
    h = [plt.Line2D([], [], marker=mk, color=c, linestyle="none", ms=4) for _, _, mk, c, _ in PIPES]
    axes[1].legend(h, [n for _, n, *_ in PIPES], loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
                   frameon=False, handletextpad=0.4, columnspacing=1.8)


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
    d, ident = load()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 4.4), sharey=False,
                             gridspec_kw=dict(wspace=0.14, left=0.29, right=0.985, top=0.93, bottom=0.17))
    draw(list(axes), d)
    out = out_dir(FIGURE)
    check = legibility(fig, out)
    excl = d[d.excludes_zero.astype(bool)]
    note = ("Twelve single-dose perturbation arms applied to each query slide's fixed tile mosaic; 2,778 COMET query "
            "slides / 1,876 patients; unaltered library; paired whole-patient bootstrap (5,000 draws, shared across "
            "pipelines and arms). hit@5 / precision@5 rows are the accepted v002 source data (R2.1-T8); mMV@5 rows "
            "come from the gated derivation r2_1_fig4a_mmv5_20261021/run_001 (plurality of the five nearest "
            "other-patient slides, ties to the closest rank; gates G1-G4 pass, hit/precision replicated to 1e-16). "
            f"Intervals excluding zero: {len(excl)} of 108 ({', '.join(f'{r.endpoint} {r.panel.split(':')[1]} {r.arm} {r.point:+.3f}' for r in excl.itertuples())}); "
            "unadjusted for multiplicity. Unperturbed macro values (identity): "
            + "; ".join(f"{r.pipeline} {r.metric} {r.estimate:.3f}" for r in ident.itertuples())
            + f". Legibility check pass = {check['pass']}.")
    save(fig, FIGURE, "Figure_S_perturbation", [V002, MMV, MMV_IDENT], note, {"source_data": d})
    print("legibility", check["pass"], check["overlaps"])
    print(excl[["endpoint", "panel", "arm", "point", "ci_low", "ci_high"]].to_string())


if __name__ == "__main__":
    main()
