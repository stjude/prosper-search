"""Supplementary figures R2.4-T3 (i) and (ii): tile-classifier failure example
and per-slide out-of-fold summaries.

[internal record] block R2.4-T3 (final-assembly items from [internal record] §4 R1/R2:
legend moved off the highest right-panel point; bars labelled as medians).

(i)  `fig3a_supp_failure_WT_7968.png` from
     `$PROSPER_OUT/r3_5_fig3a_oof_panels_20261008_v2/` (pre-assembly render
     9032932a…) was copied unchanged until 2026-11-04. Since the 2026-11-04
     supplementary-figure audit it is re-drawn here by `render_failure()`, a
     port of `scripts/r3_5_fig3a_oof_panels.py::render` that plots the run's
     saved `tiles_7968.csv` and `metrics.json` over the read-only WSI, with
     American spelling and without the ", downsample 16" title suffix. The
     port's fidelity is checked by `--check-port`, which re-draws with the
     original strings and must reproduce 9032932a… pixel for pixel.
     `--accepted-failure-copy` restores the old byte copy.
(ii) the strip plot is redrawn here from the same run's two CSVs
     (`per_slide_oof_auroc_both_classes.csv`, `per_slide_necrosis_called_tumour_ge200.csv`)
     and `metrics.json` (pooled necrosis fraction); values are not recomputed.

Decision D1 (2026-11-04): faces show de-identified slide IDs (_common.SCAN_TO_SLIDE_ID):
(i) "H&E (<slide ID>)" instead of "H&E (scan 7968)" (TEXT_D1; TEXT_REVISED is the pre-D1
2026-11-04 text, kept for reference); (ii) legend entries "<slide ID> (Fig. 3A)" and
"<slide ID> (failure example)" instead of 12151 / 7968. File names keep the scan key.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GREY, out_dir, prosper_out, save, sha256, slide_id, style  # noqa: E402

SRC = prosper_out() / "r3_5_fig3a_oof_panels_20261008_v2"
FIGURE = "figS_r24_t3_tile_classifier"
MAIN_SLIDE, FAIL_SLIDE = 12151, 7968

# ---- (i) failure example: port of scripts/r3_5_fig3a_oof_panels.py::render (plotting only) ----
WSI = Path("/PROSPER_WORKSPACE/data/comet_final_HPC/7968.svs")
TILE, LEVEL = 512, 2
NAMES = {1: "Tumor", 2: "Immune cells", 3: "Necrosis", 4: "Non-tumor", 5: "Artifacts"}
ANN_RGB = {1: (0, 158, 115), 4: (0, 114, 178), 3: (230, 159, 0), 2: (128, 0, 128), 5: (140, 140, 140)}
TEXT_ORIGINAL = dict(
    he="H&E (scan {scan}, downsample {ds:.0f})",
    heat="Cross-validated probability of tumour class (annotated tiles only)",
    cbar="probability of tumour class (held-out fold)",
    line="black line = boundary of tumour calls within annotated tiles")
TEXT_REVISED = dict(
    he="H&E (scan {scan})",
    heat="Cross-validated probability of tumor class (annotated tiles only)",
    cbar="probability of tumor class (held-out fold)",
    line="black line = boundary of tumor calls within annotated tiles")
TEXT_D1 = dict(TEXT_REVISED, he="H&E ({sid})")


def render_failure(out_png: Path, text: dict) -> dict:
    import matplotlib
    import openslide
    from matplotlib import cm, colors
    from matplotlib.patches import Patch

    scan = str(FAIL_SLIDE)
    f = pd.read_csv(SRC / f"tiles_{scan}.csv", dtype={"slide_id": str})
    metrics = json.load(open(SRC / "metrics.json"))["scans"][scan]
    assert len(f) == metrics["n_tiles"] == 6339
    plt.rcParams.update(plt.rcParamsDefault)  # the original renderer ran with matplotlib defaults
    sl = openslide.OpenSlide(str(WSI))
    ds = float(sl.level_downsamples[LEVEL])
    W, Hh = sl.level_dimensions[LEVEL]
    assert abs(ds - metrics["render"]["actual_downsample"]) < 1e-9 and [W, Hh] == metrics["render"]["level_dims"]
    he_np = np.asarray(sl.read_region((0, 0), LEVEL, (W, Hh)).convert("RGB"))
    xs, ys = f.wsi_loc_x.to_numpy() / ds, f.wsi_loc_y.to_numpy() / ds
    tl = TILE / ds
    x0, x1 = max(0, int(xs.min() - 4 * tl)), min(W, int(xs.max() + 5 * tl))
    y0, y1 = max(0, int(ys.min() - 4 * tl)), min(Hh, int(ys.max() + 5 * tl))
    assert [x0, y0, x1, y1] == metrics["render"]["crop_level_px"]
    crop = he_np[y0:y1, x0:x1]
    ann = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    heat = np.full((y1 - y0, x1 - x0), np.nan, dtype=np.float32)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for x, y, lab, p, pr in zip(xs, ys, f.label, f.p_tumour, f.pred):
        a, b = int(round(x)) - x0, int(round(y)) - y0
        c, d = a + int(round(tl)), b + int(round(tl))
        r, g, bb = ANN_RGB[int(lab)]
        ann[b:d, a:c] = (r, g, bb, 170)
        heat[b:d, a:c] = p
        mask[b:d, a:c] = 1 if pr == 1 else 0
    cmap = matplotlib.colormaps["coolwarm"]
    heat_rgba = cmap(np.nan_to_num(heat, nan=0.0))
    heat_rgba[..., 3] = np.where(np.isnan(heat), 0.0, 0.85)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)
    for ax in axes:
        ax.imshow(crop); ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title(text["he"].format(scan=scan, sid=slide_id(scan), ds=ds))
    axes[1].imshow(ann); axes[1].set_title("Pathologist tile annotation")
    axes[2].imshow(heat_rgba); axes[2].set_title(text["heat"])
    annotated = ~np.isnan(heat)
    axes[2].contour(np.ma.masked_where(~annotated, mask.astype(float)), levels=[0.5], colors="black", linewidths=0.4)
    mpp = float(sl.properties.get("openslide.mpp-x")) * ds
    bar_px = 2000.0 / mpp
    for ax in axes:
        ax.plot([20, 20 + bar_px], [crop.shape[0] - 25] * 2, color="black", linewidth=3)
        ax.text(20, crop.shape[0] - 40, "2 mm", fontsize=8)
    axes[1].legend(handles=[Patch(color=np.array(ANN_RGB[k]) / 255, label=NAMES[k]) for k in (1, 4, 3, 2, 5)
                            if metrics["class_counts"][NAMES[k]] > 0] + [Patch(facecolor="white", edgecolor="grey", label="unannotated (no prediction)")],
                   loc="lower right", fontsize=8)
    sm = cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[2], fraction=0.03, pad=0.01); cb.set_label(text["cbar"])
    auc = metrics["oof_auroc"]
    comp = ", ".join(f"{v:,} {k.lower()}" for k, v in metrics["class_counts"].items() if v > 0)
    axes[2].text(0.01, 0.01, f"n={metrics['n_tiles']:,} tiles ({comp}); "
                 f"held-out AUROC {auc:.3f}; sens {metrics['sensitivity_argmax']:.3f} / spec {metrics['specificity_argmax']:.3f} (most-probable class); "
                 + text["line"], transform=axes[2].transAxes, fontsize=7,
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return dict(wsi=str(WSI), level=LEVEL, actual_downsample=ds, crop_level_px=[x0, y0, x1, y1])


def check_port(out_png: Path) -> bool:
    """Re-draw with the original strings; compare pixels with the accepted 9032932a render."""
    from PIL import Image, ImageChops
    render_failure(out_png, TEXT_ORIGINAL)
    a = Image.open(SRC / "fig3a_supp_failure_WT_7968.png").convert("RGBA")
    b = Image.open(out_png).convert("RGBA")
    same = a.size == b.size and ImageChops.difference(a, b).getbbox() is None
    print("port check:", "pixel-identical" if same else f"DIFFERENT sizes {a.size} {b.size}",
          "| byte-identical" if sha256(out_png) == sha256(SRC / "fig3a_supp_failure_WT_7968.png") else "| bytes differ")
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-port", metavar="PNG", help="write the original-string re-draw to PNG and compare; then exit")
    ap.add_argument("--accepted-failure-copy", action="store_true",
                    help="copy the accepted 9032932a render for (i) instead of re-drawing it (pre-2026-11-04 behaviour)")
    args = ap.parse_args()
    if args.check_port:
        sys.exit(0 if check_port(Path(args.check_port)) else 1)
    style()
    au = pd.read_csv(SRC / "per_slide_oof_auroc_both_classes.csv")
    ne = pd.read_csv(SRC / "per_slide_necrosis_called_tumour_ge200.csv")
    met = json.load(open(SRC / "metrics.json"))
    pooled = met["across_slide_context"]["pooled_necrosis_called_tumour"]
    assert len(au) == 57 and len(ne) == 21
    assert abs(pooled - 0.4594435) < 1e-6 and abs(ne.necrosis_called_tumour.median() - 0.2756892) < 1e-6
    au["imprecise"] = (au.n_tumour < 20) | (au.n_non < 20)
    assert au.imprecise.sum() == 6

    rng = np.random.RandomState(0)  # jitter only; deterministic
    fig, (l, r) = plt.subplots(1, 2, figsize=(6.4, 4.0))
    fig.subplots_adjust(wspace=0.45, left=0.1, right=0.98, top=0.92, bottom=0.26)
    # left: per-slide AUROC
    x = rng.uniform(-0.25, 0.25, len(au))
    l.scatter(x[~au.imprecise], au.auroc[~au.imprecise], s=12, color=GREY, label="slide (≥ 20 tiles in both classes)")
    l.scatter(x[au.imprecise], au.auroc[au.imprecise], s=12, facecolor="white", edgecolor=GREY, label="slide with < 20 tiles in one class (imprecise)")
    l.hlines(au.auroc.median(), -0.4, 0.4, color="black", linewidth=1.2, label=f"median {au.auroc.median():.3f}")
    for sid, mk, lab in [(MAIN_SLIDE, "D", f"{slide_id(MAIN_SLIDE)} (Fig. 3A)"),
                         (FAIL_SLIDE, "s", f"{slide_id(FAIL_SLIDE)} (failure example)")]:
        i = au.index[au.slide == sid][0]
        l.scatter(x[i], au.auroc[i], s=55, marker=mk, facecolor="none", edgecolor="red", linewidth=1.2, label=lab)
    l.set_xlim(-0.6, 0.6); l.set_xticks([]); l.set_ylabel("per-slide out-of-fold AUROC (tumor vs other classes)")
    l.set_title(f"{len(au)} slides annotated with tumor and non-tumor")
    l.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=5.5)
    # right: necrosis called tumor
    x2 = rng.uniform(-0.25, 0.25, len(ne))
    r.scatter(x2, ne.necrosis_called_tumour, s=12, color=GREY)
    r.hlines(ne.necrosis_called_tumour.median(), -0.4, 0.4, color="black", linewidth=1.2, label=f"median {ne.necrosis_called_tumour.median():.2f}")
    r.axhline(pooled, color="black", linestyle=":", linewidth=0.9, label=f"pooled over 30,550 necrosis tiles on 30 slides: {pooled:.2f}")
    i = ne.index[ne.slide == FAIL_SLIDE][0]
    r.scatter(x2[i], ne.necrosis_called_tumour[i], s=55, marker="s", facecolor="none", edgecolor="red", linewidth=1.2, label=f"{slide_id(FAIL_SLIDE)} (failure example)")
    r.set_xlim(-0.6, 0.6); r.set_xticks([]); r.set_ylim(-0.03, 1.03)
    r.set_ylabel("fraction of necrosis tiles called tumor"); r.set_title(f"{len(ne)} slides with ≥ 200 necrosis tiles")
    r.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=5.5)
    note = ("Strip plot redrawn from the v2 run's CSVs (no value recomputed); points jittered horizontally with a fixed "
            "seed; bars are medians; the dotted line is the pooled fraction from metrics.json. "
            + ("The 7968 three-panel failure example is copied unchanged from the same run." if args.accepted_failure_copy else
               "The 7968 three-panel failure example is re-drawn (render_failure, a port of r3_5_fig3a_oof_panels.render) "
               "from the run's tiles_7968.csv and metrics.json over the read-only WSI, with American spelling and without "
               "the downsample suffix; port fidelity: the original-string re-draw is pixel-identical to the accepted render "
               "9032932a (--check-port).")
            + f" Decision D1 (2026-11-04): the faces name the de-identified slide IDs {slide_id(12151)} (12151) and "
              f"{slide_id(7968)} (7968) instead of the scan keys.")
    outd = save(fig, FIGURE, "FigS_R2.4-T3_ii_per_slide_strip",
                [SRC / "per_slide_oof_auroc_both_classes.csv", SRC / "per_slide_necrosis_called_tumour_ge200.csv", SRC / "metrics.json",
                 SRC / "fig3a_supp_failure_WT_7968.png", SRC / f"tiles_{FAIL_SLIDE}.csv"], note,
                {"source_data_auroc": au, "source_data_necrosis": ne})
    dst = outd / "FigS_R2.4-T3_i_failure_WT_7968.png"
    if args.accepted_failure_copy:
        shutil.copyfile(SRC / "fig3a_supp_failure_WT_7968.png", dst)
        assert sha256(dst) == sha256(SRC / "fig3a_supp_failure_WT_7968.png")
    else:
        render_failure(dst, TEXT_D1)
    with open(outd / "SHA256SUMS.txt", "a") as fh:
        fh.write(f"{sha256(dst)}  {dst.name}\n")
    print("copied" if args.accepted_failure_copy else "re-drew", dst.name)


if __name__ == "__main__":
    main()
