"""Figure 6 (revised): ROI search under six search configurations; panel B as submitted.

Panel A (2026-11-12: "i want all three in both l1 and cosine on figure 6a. so 6 config"):
slide-level aggregation of mosaic-tile retrieval at k = 5 hits per ROI, all slides of the query's
patient excluded (C1b), for Virchow2 (first 1,280 dimensions), UNI2 and H-Optimus-1, each searched
with L1 and with cosine distance (R3.5-ROI6; float32 features, one A100 type). Per-group F1 and
overall macro / micro F1, each with a whole-patient 95 % interval (B = 5,000, seed 123, the R4.a-vii
draws). UNI2 + cosine is the pre-specified primary; the others are sensitivity settings. The
submitted float16 Virchow2 L1 hits are not drawn (gate rows only; kept in source_data_panelA.csv with
drawn = False). The single-arm script is kept at superseded/fig6_roi_search_single_arm_20261103.py.

Panel B (2026-11-12: panel B and Fig. 7 are A.F.'s and stay unless wrong): the submitted
panel, cropped from docs/prosper_submission/fig/Figure_6.png with the alpha flattened on white; its
hits and L1 maps come from the submitted ROI search (Virchow2, first 1,280 dimensions), and none of
its six displayed hits is from the query's patient. Candidates only (``--b-panel``): ``cosine``, the
same query ROIs re-drawn under UNI2 + cosine with the query's patient excluded and the map on ROI1
(candidates_20261112_cosine_panelB/); ``same_group``, the map on the first ROI from the query's group
(candidates_20261112_same_group_map/).

Sources (accepted; nothing recomputed here):
  * $PROSPER_OUT/roi_search_6cfg_20261111/eval/per_group_001/per_group_f1.csv (gates.json passed)
  * $PROSPER_OUT/roi_search_6cfg_20261111/eval/run_001/summary.csv (metric f1, configs *_k5)
  * docs/prosper_submission/fig/Figure_6.png (panel B)
  * $PROSPER_OUT/figures_20261021/fig6_roi_search/panelB_cosine/ (candidates; examples.json passed)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GREY, GROUP_ORDER_ALPHA, REPO, out_dir, prosper_out, style, write_provenance  # noqa: E402
import figS1_S2_neighbor_ladder as NL  # noqa: E402
from figS1_S2_neighbor_ladder import legibility  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
EVAL = prosper_out() / "roi_search_6cfg_20261111" / "eval"
PER_GROUP = EVAL / "per_group_001" / "per_group_f1.csv"
PER_GROUP_GATES = EVAL / "per_group_001" / "gates.json"
SUMMARY = EVAL / "run_001" / "summary.csv"
FIGURE = "fig6_roi_search"
ASSETS = out_dir(FIGURE) / "panelB_cosine"
EXAMPLES_JSON = ASSETS / "examples.json"
SUBMITTED = REPO / "docs" / "prosper_submission" / "fig" / "Figure_6.png"
PANEL_B_TOP = 1824  # first ink row of panel B in the submitted composite (blank band 1607-1823)
B_LETTER_BOX = (60, 1880, 150, 1980)  # the submitted "B" letter (ink rows 1892-1964, cols 77-132), masked and redrawn

CONFIGS = [  # (config, encoder, distance, role, colour)
    ("v2_l1", "Virchow2", "L1", "sensitivity", "#bcbddc"),
    ("v2_cos", "Virchow2", "cosine", "sensitivity", "#6a51a3"),
    ("uni2_l1", "UNI2", "L1", "sensitivity", "#9ecae1"),
    ("uni2_cos", "UNI2", "cosine", "primary", "#08519c"),
    ("hopt1_l1", "H-Optimus-1", "L1", "sensitivity", "#a1d99b"),
    ("hopt1_cos", "H-Optimus-1", "cosine", "sensitivity", "#238b45"),
]
INTERVAL = "whole-patient percentile 95 %, B = 5000, seed 123 (R4.a-vii draws)"
QUERY_COL, HIT_COL, ARROW_COL = "#1f77b4", "#4d4d4d", "#ffd400"
HEAT_CMAP, HEAT_ALPHA = "jet", 0.55
DPI = 600  # the 512-pixel ROI tiles are drawn about 1 inch wide


def load_a():
    assert json.loads(PER_GROUP_GATES.read_text())["passed"], "per_group_001 gates not passed"
    g = pd.read_csv(PER_GROUP)
    g = g[g.k == 5]
    s = pd.read_csv(SUMMARY)
    s = s[s.metric == "f1"]
    assert (g.n_replicates == 5000).all() and (s.n_replicates == 5000).all() and (s.n_patients == 1876).all()
    rows = []
    for cfg, enc, dist, role, _ in [("submitted", "Virchow2 (submitted float16 hits)", "L1", "gate only", None), *CONFIGS]:
        gc = g[g.config == cfg]
        assert len(gc) == 15, cfg
        for grp in GROUP_ORDER_ALPHA:
            r = gc[gc.group == grp].iloc[0]
            rows.append(dict(category=grp, kind="group", config=cfg, encoder=enc, distance=dist, role=role,
                             drawn=cfg != "submitted", n_slides=int(r.n_slides), n_patients=int(r.n_patients),
                             f1=r.f1, ci_lo=r.ci_lo, ci_hi=r.ci_hi))
        for avg, lab in [("macro", "Macro F1"), ("micro", "Micro F1")]:
            r = s[(s.config == f"{cfg}_k5") & (s.average == avg)].iloc[0]
            rows.append(dict(category=lab, kind="overall", config=cfg, encoder=enc, distance=dist, role=role,
                             drawn=cfg != "submitted", n_slides=int(r.n_queries), n_patients=int(r.n_patients),
                             f1=r.point, ci_lo=r.ci_lo, ci_hi=r.ci_hi))
    df = pd.DataFrame(rows)
    df["k"] = 5
    df["interval"] = INTERVAL
    n = df.groupby("category")[["n_slides", "n_patients"]].nunique()
    assert (n == 1).all().all(), "slide / patient counts differ between configurations"
    return df


def draw_a(ax, ax_n, df):
    cats = GROUP_ORDER_ALPHA + ["Macro F1", "Micro F1"]
    x = np.arange(len(cats), dtype=float)
    x[15:] += 0.6
    x[16:] += 0.2
    w = 0.13
    for i, (cfg, enc, dist, role, col) in enumerate(CONFIGS):
        d = df[df.config == cfg].set_index("category").loc[cats]
        xi = x + (i - 2.5) * w
        lab = f"{enc}, {dist}" + (" (primary)" if role == "primary" else "")
        ax.bar(xi, d.f1, w, color=col, edgecolor="black", linewidth=0.2, label=lab)
        ax.errorbar(xi, d.f1, yerr=[d.f1 - d.ci_lo, d.ci_hi - d.f1], fmt="none", ecolor="black",
                    elinewidth=0.35, capsize=0.6, capthick=0.35)
    ax.axvline(14.8, color=GREY, linewidth=0.6, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, x[-1] + 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 (k = 5 hits per ROI)")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, handlelength=1.2,
              columnspacing=1.2, borderaxespad=0.2)
    ref = df[df.config == "uni2_cos"].set_index("category").loc[cats]
    ax_n.set_xlim(ax.get_xlim()); ax_n.set_ylim(0, 1); ax_n.axis("off")
    for xi, r in zip(x, ref.itertuples()):
        ax_n.text(xi, 0.5, f"{r.n_slides:,}\n{r.n_patients:,}", ha="center", va="center", fontsize=6, color="black")
    ax_n.text(-0.65, 0.5, "slides\npatients", ha="right", va="center", fontsize=6, color=GREY)


NOTE_A = ("Panel A (six configurations, Haoran 2026-11-12): per-group F1 from R3.5-ROI6 eval/per_group_001 "
          "(C6 kernel on the R4.a-vii whole-patient draws; gates G1-G3 and identical draws passed) and overall "
          "macro / micro F1 from eval/run_001/summary.csv, k = 5, every slide of the query's patient excluded. "
          "UNI2 + cosine is the pre-specified primary; the other five are sensitivity settings. The submitted "
          "float16 Virchow2 L1 hits are not drawn (rows with drawn = False). All intervals " + INTERVAL + ".")
NOTE_B_SUBMITTED = ("Panel B: the submitted panel (rows 1824- of docs/prosper_submission/fig/Figure_6.png, alpha "
                    "flattened on white, the 'B' letter masked and redrawn), unchanged. Its hits and L1 maps come from "
                    "the submitted ROI search (Virchow2, first 1,280 dimensions; query slide excluded); none of the six "
                    "displayed hits is from the query's patient (checked against UID.Subject).")
NOTE_B = ("Panel B: the submitted EWS and NB query slides and query ROIs, re-drawn under UNI2 + cosine from "
          "panelB_cosine/ (fig6b_cosine_examples.py; gates in examples.json). ROI1-ROI3 are the query ROI's top "
          "three hits after excluding the query's patient; the map shows ROI1's slide (submitted rule), "
          "cosine distance to every mosaic tile assigned to the 512 x 512 patch locations it represents, "
          "min-max normalized within the pair and inverted, jet colour map at alpha 0.55.")


def write_fig(fig, d: Path, stem: str, frames: dict) -> list[Path]:
    png, pdf = d / f"{stem}.png", d / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", metadata={"CreationDate": None})
    out = [png, pdf]
    for name, frame in frames.items():
        frame.to_csv(d / f"{name}.csv", index=False)
        out.append(d / f"{name}.csv")
    print("wrote", png)
    return out


def render_a(d: Path):
    df = load_a()
    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.07], hspace=0.02, left=0.09, right=0.99, top=0.86, bottom=0.30)
    ax, ax_n = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    draw_a(ax, ax_n, df)
    pos = ax.get_position(); ax_n.set_position([pos.x0, 0.075, pos.width, 0.07])
    leg = legibility(fig, d, "6A")
    assert leg["pass"], {k: v for k, v in leg.items() if k != "texts"}
    written = write_fig(fig, d, "Figure_6A_revised", {"source_data_panelA": df})
    plt.close(fig)
    return written


def show_tile(ax, path, title, edge):
    ax.imshow(Image.open(path))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(edge); sp.set_linewidth(1.2)
    ax.set_title(title, fontsize=6.5, pad=2)


def trim_dark_edges(im, dark=120, bg=230, band=0.03):
    """Drop a dark scanner-edge band (border row / column mean < dark) up to the first background
    line, within 3 % of the border; returns (image, (dx, dy))."""
    a = np.asarray(im.convert("L"), dtype=float)

    def cut(prof):
        n = int(len(prof) * band)
        if prof[0] >= dark:
            return 0
        return next((i for i in range(n) if prof[i] >= bg), 0)

    r, c = a.mean(axis=1), a.mean(axis=0)
    y0, y1 = cut(r), len(r) - cut(r[::-1])
    x0, x1 = cut(c), len(c) - cut(c[::-1])
    return im.crop((x0, y0, x1, y1)), (x0, y0)


def mark(ax, xy, ds, col, size=512, off_xy=(0, 0)):
    x0, y0, s = xy[0] / ds - off_xy[0], xy[1] / ds - off_xy[1], size / ds
    ax.add_patch(Rectangle((x0, y0), s, s, fill=False, edgecolor="black", linewidth=0.6))
    span = max(ax.get_xlim()[1], ax.get_ylim()[0])
    off = 0.09 * span
    ax.annotate("", xy=(x0 + s / 2, y0 + s / 2), xytext=(x0 + s / 2 + off, y0 + s / 2 + off),
                arrowprops=dict(arrowstyle="-|>", color=col, linewidth=1.4, mutation_scale=7, shrinkA=0, shrinkB=1.5))


def heat_overlay(thumb, z, off_xy=(0, 0)):
    h, w = thumb.size[1], thumb.size[0]
    ov = np.zeros((h, w, 4), dtype=float)
    ds, p = float(z["downsample"]), int(z["patch"])
    rgba = plt.get_cmap(HEAT_CMAP)(z["heat"])
    for (x, y), c in zip(z["locs"], rgba):
        x, y = x - off_xy[0] * ds, y - off_xy[1] * ds
        x0, y0 = int(round(x / ds)), int(round(y / ds))
        x1, y1 = max(x0 + 1, int(round((x + p) / ds))), max(y0 + 1, int(round((y + p) / ds)))
        ov[y0:y1, x0:x1, :3] = c[:3]
        ov[y0:y1, x0:x1, 3] = HEAT_ALPHA
    return ov


def render_b(map_tag: str, d: Path):
    ex = json.loads(EXAMPLES_JSON.read_text())
    assert ex["passed"], "panel B assets failed their gates"
    fig = plt.figure(figsize=(7.2, 4.3))
    outer = fig.add_gridspec(2, 1, hspace=0.30, left=0.01, right=0.99, top=0.93, bottom=0.05)
    rows = []
    for k, e in enumerate(ex["examples"]):
        a = ASSETS / f"{e['group']}_{e['query']}"
        tag = map_tag if (a / f"{map_tag}_heat.npz").exists() else "roi1"
        z = np.load(a / f"{tag}_heat.npz")
        rank = int(z["rank"])
        h = e["hits"][rank - 1]
        inner = outer[k].subgridspec(2, 4, width_ratios=[2.3, 1.0, 1.0, 2.7], wspace=0.08, hspace=0.22)
        axq = fig.add_subplot(inner[:, 0])
        qt, qoff = trim_dark_edges(Image.open(a / "query_thumb.png"))
        axq.imshow(qt); axq.axis("off")
        axq.set_title(f"Query WSI:\n{e['group']} - {e['query_slide_id']}", fontsize=7, pad=3)
        mark(axq, e["query_xy"], e["query_thumb_downsample"], QUERY_COL, off_xy=qoff)
        show_tile(fig.add_subplot(inner[0, 1]), a / "query_roi.png", "Query ROI", QUERY_COL)
        for j, (ri, ci) in enumerate([(0, 2), (1, 1), (1, 2)], 1):
            hj = e["hits"][j - 1]
            show_tile(fig.add_subplot(inner[ri, ci]), a / f"roi{j}.png", f"ROI{j} - {hj['group']}", HIT_COL)
        axm = fig.add_subplot(inner[:, 3])
        mt, moff = trim_dark_edges(Image.open(a / f"{tag}_thumb.png"))
        axm.imshow(mt); axm.imshow(heat_overlay(mt, z, moff)); axm.axis("off")
        axm.set_title(f"Cosine similarity map, ROI{rank} slide:\n{h['group']} - {h['hit_slide_id']}", fontsize=7, pad=3)
        mark(axm, h["hit_xy"], float(z["downsample"]), ARROW_COL, off_xy=moff)
        cax = axm.inset_axes([0.05, -0.06, 0.5, 0.035])
        cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=HEAT_CMAP), cax=cax, orientation="horizontal")
        cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
        cb.outline.set_linewidth(0.3)
        rows.append(dict(group=e["group"], query_slide_id=e["query_slide_id"], query_row=e["query_row"],
                         map_rank=rank, map_slide_id=h["hit_slide_id"], map_slide_group=h["group"],
                         **{f"roi{j}_{f}": e["hits"][j - 1][v] for j in (1, 2, 3)
                            for f, v in [("slide_id", "hit_slide_id"), ("group", "group"),
                                         ("cosine_distance", "stored_distance")]}))
    leg = legibility_with_titles(fig, d, "6B")
    assert leg["pass"] and leg["n_texts"] == 8 * len(ex["examples"]), {k: v for k, v in leg.items() if k != "texts"}
    return fig, pd.DataFrame(rows)


def legibility_with_titles(fig, d: Path, tag: str):
    """The shared check also counting axes titles and inset (colour-bar) axes, which carry panel B's text."""
    base = NL.visible_texts

    def texts(f):
        out = base(f)
        for ax in f.axes:
            out += [ax.title] if ax.title.get_text().strip() else []
            for ch in ax.child_axes:
                out += [t.label1 for t, loc in zip(ch.xaxis.get_major_ticks(), ch.xaxis.get_majorticklocs())
                        if t.label1.get_visible() and t.label1.get_text().strip()]
        return out

    NL.visible_texts = texts
    try:
        return legibility(fig, d, tag)
    finally:
        NL.visible_texts = base


def submitted_panel_b() -> Image.Image:
    im = Image.open(SUBMITTED)
    assert im.size == (4354, 4464) and im.mode == "RGBA"
    bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[-1])
    bg.paste("white", B_LETTER_BOX)
    return bg.crop((0, PANEL_B_TOP, im.size[0], im.size[1]))


def compose(a_png: Path, b, out: Path):
    a = Image.open(a_png).convert("RGB")
    b = (b if isinstance(b, Image.Image) else Image.open(b)).convert("RGB")
    W = max(a.size[0], b.size[0])
    a = a if a.size[0] == W else a.resize((W, round(a.size[1] * W / a.size[0])), Image.LANCZOS)
    b = b if b.size[0] == W else b.resize((W, round(b.size[1] * W / b.size[0])), Image.LANCZOS)
    s = DPI / 300
    pad, letter_w = round(40 * s), round(110 * s)
    comp = Image.new("RGB", (W + letter_w, a.size[1] + b.size[1] + 3 * pad), "white")
    comp.paste(a, (letter_w, pad)); comp.paste(b, (letter_w, 2 * pad + a.size[1]))
    dr = ImageDraw.Draw(comp)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", round(80 * s))
    except OSError:
        font = ImageFont.load_default()
    dr.text((round(20 * s), pad), "A", fill="black", font=font)
    dr.text((round(20 * s), 2 * pad + a.size[1]), "B", fill="black", font=font)
    comp.save(out, dpi=(DPI, DPI))
    print("wrote", out, comp.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-panel", choices=["submitted", "cosine", "same_group"], default="submitted")
    args = ap.parse_args()
    style()
    d = out_dir(FIGURE)
    a_png = d / "Figure_6A_revised.png"
    inputs = [EXAMPLES_JSON, *sorted(p for p in ASSETS.rglob("*") if p.is_file() and p.name != "examples.json")]
    if args.b_panel == "submitted":
        written = render_a(d)
        comp = d / "Figure_6_revised_composite.png"
        compose(a_png, submitted_panel_b(), comp)
        written += [comp, d / "legibility_check_6A.json"]
        write_provenance(d, written, [PER_GROUP, PER_GROUP_GATES, SUMMARY, SUBMITTED],
                         NOTE_A + "\n\n" + NOTE_B_SUBMITTED, Path(sys.argv[0]), title=f"{FIGURE} / Figure_6_revised")
    elif args.b_panel == "cosine":
        assert a_png.exists(), "render the adopted figure first"
        c = d / "candidates_20261112_cosine_panelB"
        c.mkdir(exist_ok=True)
        fig, src = render_b("roi1", c)
        written = write_fig(fig, c, "Figure_6B_revised", {"source_data_panelB": src})
        plt.close(fig)
        comp = c / "Figure_6_revised_composite.png"
        compose(a_png, c / "Figure_6B_revised.png", comp)
        written += [comp, c / "legibility_check_6B.json"]
        write_provenance(c, written, [a_png, *inputs], "Candidate panel B, not the adopted figure. " + NOTE_B,
                         Path(sys.argv[0]), title="fig6 candidate / cosine panel B")
    else:
        assert a_png.exists(), "render the adopted figure first"
        c = d / "candidates_20261112_same_group_map"
        c.mkdir(exist_ok=True)
        fig, src = render_b("same_group", c)
        png = c / "Figure_6B_same_group_map.png"
        fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        src.to_csv(c / "source_data_panelB.csv", index=False)
        comp = c / "Figure_6_revised_composite_same_group_map.png"
        compose(a_png, png, comp)
        write_provenance(c, [png, comp, c / "source_data_panelB.csv", c / "legibility_check_6B.json"],
                         [a_png, *inputs], "Alternative panel B for the author's choice: where ROI1 is from another "
                         "tumour group (EWS example), the map shows the first ROI from the query's group (ROI2). "
                         "Not the adopted figure.", Path(sys.argv[0]), title="fig6 candidate / same-group map")


if __name__ == "__main__":
    main()
