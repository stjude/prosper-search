"""revision ("lean supplement", design decisions of 2026-11-16) image changes to Supplementary Figures.

Display-only re-renders from the retained figure scripts and their accepted inputs; no value is recomputed.
Each sub-command first re-renders the CURRENT figure ("original" variant), which must reproduce the adopted
PNG pixel for pixel, and then the changed variant, so the only difference is the requested one.

  letters --fig S5|S7   I1: panel letters upper case (S5: bold a/b/c and the legend title "Panel a";
                        S7: the "a  " ... "f  " prefixes of the six panel titles).
                        Scripts: figS_r21_f4_classifier_budget.py (S5), figS3_nb_analysis.py (S7).
  s12                   I2: overlay colour-bar label "Delta = ..." -> "Δ = ..." (figS_r35_l2_sources_rerender.py
                        ports, re-assembled with figS_r35_l1_l2_assembly.py functions).
  s11                   I3: only the three annotated same-group pairs (former D-F, now A-C), re-laid with larger
                        TrueType sub-captions; the 112-px tiles are copied from the accepted sources_20261104
                        montages (verified against fresh read-only WSI reads with --verify-tiles).
  s8                    I4: panels A and B only (panel C dropped), same forest code and axes geometry.

Outputs: $PROSPER_OUT/figures_20261021/figS_lean_048_20261116/<fig>/<variant>/.
The WSI / feature / label paths hard-coded under /PROSPER_DEPT/ in the analysis modules are
remapped to the equivalent /PROSPER_DEPT/ mount (same files) because the former is not mounted on every node.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import out_dir, prosper_out, sha256, write_provenance  # noqa: E402

ROOT_NAME = "figS_lean_048_20261116"
OLD_PREFIX = "/PROSPER_DEPT/"
NEW_PREFIX = "/PROSPER_DEPT/"
CURRENT = {  # adopted images in the revision Supplementary Information (render_record_supplement.json)
    "S5": prosper_out() / "figures_20261021/figS_r21_f4_classifier_budget/FigS_R2.1-F4_classifier_and_5pct.png",
    "S7": prosper_out() / "figures_20261021/figS3_nb_analysis/Figure_S3_NB_analysis_revised.png",
    "S8": prosper_out() / "figures_20261021/figS5_filtering_effects/Figure_S5_filtering_effects.png",
    "S11": prosper_out() / "collab_draft_20261102/pass_041/figures/figS_r35_L1_representative_tiles/"
                           "FigS_R3.5-L1_representative_tiles.png",
    "S12": prosper_out() / "figures_20261021/figS_r35_L2_discriminative_map/FigS_R3.5-L2_discriminative_map.png",
}


def remap(p) -> Path:
    s = str(p)
    return Path(NEW_PREFIX + s[len(OLD_PREFIX):]) if s.startswith(OLD_PREFIX) else Path(s)


def wsi_dir() -> Path:
    """Symlinks <scan>.svs -> the resolvable /PROSPER_DEPT copy of each read-only WSI."""
    d = prosper_out() / "collab_draft_20261102/checks_048/prep/scratch/wsi_links"
    d.mkdir(parents=True, exist_ok=True)
    return d


def wsi_template(scans) -> str:
    src = Path(NEW_PREFIX) / "core_operations/ImageAnalysis/Core/Haoran/data/comet_final_HPC"
    d = wsi_dir()
    for s in scans:
        target = remap(Path(src / f"{s}.svs").readlink())
        assert target.exists(), target
        link = d / f"{s}.svs"
        if not link.is_symlink():
            link.symlink_to(target)
    return str(d / "{}.svs")


def vdir(fig: str, variant: str) -> Path:
    return out_dir(f"{ROOT_NAME}/{fig}/{variant}")


# ------------------------------------------------------------------ I1: S5 / S7 panel letters
def run_letters(fig: str, variant: str):
    import matplotlib.axes
    import matplotlib.figure
    mod = importlib.import_module({"S5": "figS_r21_f4_classifier_budget", "S7": "figS3_nb_analysis"}[fig])
    mod.FIGURE = f"{ROOT_NAME}/{fig}/{variant}"
    changed = []
    orig_text, orig_title, orig_legend = matplotlib.axes.Axes.text, matplotlib.axes.Axes.set_title, \
        matplotlib.figure.Figure.legend

    def text(self, x, y, s, *a, **k):
        if fig == "S5" and s in ("a", "b", "c") and k.get("fontweight") == "bold":
            changed.append((s, s.upper())); s = s.upper()
        return orig_text(self, x, y, s, *a, **k)

    def set_title(self, label, *a, **k):
        if fig == "S7" and re.match(r"^[a-f]  ", label):
            changed.append((label[:1], label[:1].upper())); label = label[:1].upper() + label[1:]
        return orig_title(self, label, *a, **k)

    def legend(self, *a, **k):
        if fig == "S5" and k.get("title") == "Panel a":
            changed.append(("Panel a", "Panel A")); k["title"] = "Panel A"
        return orig_legend(self, *a, **k)

    if variant == "upper":
        matplotlib.axes.Axes.text, matplotlib.axes.Axes.set_title, matplotlib.figure.Figure.legend = \
            text, set_title, legend
    try:
        mod.main()
    finally:
        matplotlib.axes.Axes.text, matplotlib.axes.Axes.set_title, matplotlib.figure.Figure.legend = \
            orig_text, orig_title, orig_legend
    expect = {"S5": 4, "S7": 6}[fig] if variant == "upper" else 0
    assert len(changed) == expect, changed
    return {"changed": changed}


# ------------------------------------------------------------------ I4: S8 without panel C
def run_s8(variant: str):
    import matplotlib.pyplot as plt
    import numpy as np
    import figS5_filtering_effects as m
    from _common import style
    style()
    d = m.load()
    if variant == "original":  # the adopted default layout (no --with-hit-yield)
        sys.argv = [sys.argv[0]]
        m.FIGURE = f"{ROOT_NAME}/S8/original"
        m.main()
        return {}
    # same width, margins and A/B axes size/position (inches from the top) as the adopted 8.5 x 6.6 in layout
    from matplotlib.transforms import Bbox
    captured = {}
    m.save = lambda f, *a, **k: captured.setdefault("fig", f)   # the adopted figure, not saved
    sys.argv = [sys.argv[0]]
    m.main()
    fo = captured["fig"]
    fo.set_dpi(300); fo.canvas.draw()
    bo = fo.get_tightbbox(fo.canvas.get_renderer()).padded(0.1)  # what savefig(bbox_inches="tight") crops to
    ref = plt.figure(figsize=(8.5, 6.6))
    gs0 = ref.add_gridspec(2, 1, height_ratios=[1, 1.7], hspace=0.6, left=0.2, right=0.98, top=0.95, bottom=0.10)
    pos = gs0[0].get_position(ref)
    plt.close(ref)
    top_in, h_in = (1 - pos.y1) * 6.6, pos.height * 6.6
    H = 2.75
    fig = plt.figure(figsize=(8.5, H))
    gs = fig.add_gridspec(1, 1, left=0.2, right=0.98, top=1 - top_in / H, bottom=1 - (top_in + h_in) / H)
    gsab = gs[0].subgridspec(1, 2, wspace=1.9)
    ax_a = fig.add_subplot(gsab[0, 0]); ax_b = fig.add_subplot(gsab[0, 1])
    m.forest(ax_a, d, "hit_at_5", "conditional_returned", "A  Macro hit@5", "Δ macro hit@5")
    m.forest(ax_b, d, "precision_at_5", "conditional_returned", "B  Macro precision@5", "Δ macro precision@5")
    ax_b.set_yticklabels([])
    note = ("Pass 048 (author decision 2026-11-16): panels A and B of the adopted figS5_filtering_effects render only; "
            "panel C (group-level differences) dropped. Same load() with its asserted R1.3-T9 acceptance values, same "
            "forest() calls, same figure width, margins and A/B axes geometry (inches from the top) as the adopted "
            "8.5 x 6.6 in layout; figure height 2.75 in. Saved with the adopted render's tight-crop left/right/top "
            "edges and a bottom edge chosen so that the pixel grid falls identically (A/B pixels then equal the "
            "adopted render's). Source data = the same frame as the adopted render.")
    fig.set_dpi(300); fig.canvas.draw()
    bn = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
    y1 = bo.y1 - (6.6 - H)
    frac = lambda v: v - np.floor(v)  # noqa: E731
    e = frac(frac(bo.height * 300) - frac((y1 - bn.y0) * 300))
    box = Bbox([[bo.x0, bn.y0 - e / 300], [bo.x1, y1]])
    dd = vdir("S8", "ab")
    png, pdf = dd / "Figure_S5_filtering_effects.png", dd / "Figure_S5_filtering_effects.pdf"
    fig.savefig(png, dpi=300, bbox_inches=box, facecolor="white")
    fig.savefig(pdf, bbox_inches=box, facecolor="white", metadata={"CreationDate": None})
    d.to_csv(dd / "source_data.csv", index=False)
    write_provenance(dd, [png, pdf, dd / "source_data.csv"], [m.SRC], note, Path(__file__),
                     title=f"{ROOT_NAME}/S8/ab / Figure_S5_filtering_effects")
    return {"top_in": top_in, "axes_row_height_in": h_in, "fig_height_in": H, "bbox_in": box.bounds,
            "adopted_bbox_in": bo.bounds, "frac_adjust_px": e}


# ------------------------------------------------------------------ I2: S12 colour-bar label
def _l2_module():
    R = importlib.import_module("figS_r35_l2_sources_rerender")
    L1 = R.L1
    L1.DEV = remap(L1.DEV)
    L1.WSI = Path(wsi_template(R.REFS))
    init = L1.MapSlide.__init__
    init.__defaults__ = tuple(remap(v) if isinstance(v, Path) else v for v in init.__defaults__)
    return R


def run_s12(variant: str):
    import matplotlib.figure
    from PIL import Image
    A = importlib.import_module("figS_r35_l1_l2_assembly")
    R = _l2_module()
    d = vdir("S12", variant) / "sources"
    orig_cb = matplotlib.figure.Figure.colorbar
    changed = []

    def colorbar(self, *a, **k):
        lab = k.get("label", "")
        if lab.startswith("Delta = "):
            changed.append(lab); k["label"] = "Δ = " + lab[len("Delta = "):]
        return orig_cb(self, *a, **k)

    if variant == "delta":
        matplotlib.figure.Figure.colorbar = colorbar
    try:
        written = R.render_all(d, R.TEXT_D1)
    finally:
        matplotlib.figure.Figure.colorbar = orig_cb
    assert len(changed) == (2 if variant == "delta" else 0), changed
    panels = []
    for L, n, t, crop in A.L2_PANELS_D1:
        im = Image.open(d / n).convert("RGB")
        panels.append(A.labelled(im.crop((0, crop, im.width, im.height)), L, t))
    canvas = A.stack(panels)
    png = vdir("S12", variant) / "FigS_R3.5-L2_discriminative_map.png"
    canvas.save(png, dpi=(300, 300))
    canvas.convert("RGB").save(png.with_suffix(".pdf"), resolution=300)
    inputs = [R.RUN / n for n in ("per_tile_cost_D2.csv.gz", "per_pair_D2.csv.gz", "per_reference_D2.csv.gz",
                                  "matrices_D1.csv")] + [R.L2M.BIND / "meta_tiles.pkl"]
    write_provenance(vdir("S12", variant), written + [png, png.with_suffix(".pdf")], inputs,
                     ("Pass 048 I2: " if variant == "delta" else "Pass 048 reproduction of the adopted S12: ")
                     + "the three l2/run_003 panels re-drawn by figS_r35_l2_sources_rerender.render_all with TEXT_D1"
                     + (" and the overlay colour-bar label 'Delta = mean other-group cost - mean same-group cost' "
                        "changed to 'Δ = mean other-group cost - mean same-group cost' (nothing else)"
                        if variant == "delta" else "")
                     + "; assembled with figS_r35_l1_l2_assembly.labelled/stack, L2_PANELS_D1 and the same header crops.",
                     Path(__file__), title=f"S12 ({variant}), pass 048 prep")
    rel_sums(vdir("S12", variant))
    return {"colorbar_labels_changed": changed}


def rel_sums(d: Path) -> None:
    """Rewrite SHA256SUMS.txt with paths relative to `d` (_common.write_provenance records base names only)."""
    files = sorted(p for p in d.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS.txt", "PROVENANCE.md"))
    (d / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.relative_to(d).as_posix()}\n" for p in files))


# ------------------------------------------------------------------ I3: S11 annotated pairs only
FONT = "/HOME/miniconda3/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
KEPT = [  # new letter, accepted montage (former panel)
    ("A", "montage_annot_12151_vs_7443_p01.png", "D"),
    ("B", "montage_annot_16039_vs_16676_p01.png", "E"),
    ("C", "montage_annot_7292_vs_7347_p01.png", "F"),
]


def _s11_sources():
    S = importlib.import_module("figS_r35_l1_sources_rerender")
    import pandas as pd
    links = pd.read_csv(S.S3 / "representative_links.csv", dtype={"query": str, "reference": str,
                                                                   "query_class": str, "ref_class": str})
    links = links[links.kind.isin(S.KINDS)]
    return S, links


def _old_geometry(S, L):
    """Tile boxes of the accepted montage (port of figS_r35_l1_sources_rerender.render)."""
    cs = sorted(set(int(c) for c in L.cluster))[:20]
    out = []
    for ri, c in enumerate(cs):
        y = 30 + ri * (S.TS + 34)
        for ci, kind in enumerate(S.KINDS):
            l = L[(L.cluster == c) & (L.kind == kind)].iloc[0]
            x0 = 80 + ci * (2 * S.TS + 20)
            out.append(dict(ri=ri, c=c, ci=ci, kind=kind, link=l, x0=x0, y=y,
                            boxes=[(x0 + si * S.TS, y, x0 + si * S.TS + S.TS, y + S.TS) for si in (0, 1)]))
    return cs, out


def _reproduce_old(S, L, src_img):
    """Re-draw the accepted montage from its own tiles + cap_revised strings; must equal the accepted PNG."""
    from PIL import Image, ImageChops, ImageDraw
    cs, geo = _old_geometry(S, L)
    W = 80 + 3 * (2 * S.TS + 20); Hh = 30 + len(cs) * (S.TS + 34)
    img = Image.new("RGB", (W, Hh), "white"); dr = ImageDraw.Draw(img)
    for x0, title in ((80, "medoid"), (80 + 2 * S.TS + 20, "highest mass"), (80 + 2 * (2 * S.TS + 20), "lowest mass")):
        dr.text((x0, 8), f"{title} (query | reference)", fill="black")
    for g in geo:
        if g["ci"] == 0:
            dr.text((4, g["y"] + S.TS // 2), f"c{g['c']}", fill="black")
        for b in g["boxes"]:
            img.paste(src_img.crop(b), b[:2])
        t1, t2 = S.cap_revised(g["link"])
        dr.text((g["x0"], g["y"] + S.TS + 2), t1, fill="black")
        dr.text((g["x0"], g["y"] + S.TS + 16), t2, fill="black")
    return ImageChops.difference(img, src_img).getbbox() is None


def _new_montage(S, L, src_img, f: int):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(FONT, f)
    TS, KG, LW = S.TS, 40, 80   # 40-px gap between kind columns (was 20) so adjacent captions stay apart
    colw = 2 * TS + KG
    line = f + 5
    hdr = 2 * line + 8
    cap = 2 * line + 10
    cs, geo = _old_geometry(S, L)
    W = LW + 3 * colw
    Hh = hdr + len(cs) * (TS + cap)
    img = Image.new("RGB", (W, Hh), "white"); dr = ImageDraw.Draw(img)
    for ci, title in enumerate(("medoid", "highest mass", "lowest mass")):
        x0 = LW + ci * colw
        dr.text((x0, 4), title, fill="black", font=font)
        dr.text((x0, 4 + line), "(query | reference)", fill="black", font=font)
    strings, widths = [], []
    for g in geo:
        y = hdr + g["ri"] * (TS + cap)
        x0 = LW + g["ci"] * colw
        if g["ci"] == 0:
            dr.text((4, y + TS // 2), f"c{g['c']}", fill="black", font=font, anchor="lm")
        for si, b in enumerate(g["boxes"]):
            img.paste(src_img.crop(b), (x0 + si * TS, y))
        t1, t2 = S.cap_revised(g["link"])
        dr.text((x0, y + TS + 3), t1, fill="black", font=font)
        dr.text((x0, y + TS + 3 + line), t2, fill="black", font=font)
        strings.append(dict(cluster=g["c"], kind=g["kind"], caption_1=t1, caption_2=t2,
                            cosine_cost=float(g["link"].cosine_cost), mass=float(g["link"].mass),
                            query_class=g["link"].query_class, ref_class=g["link"].ref_class))
        widths += [dr.textlength(t1, font=font), dr.textlength(t2, font=font)]
    for t in ("(query | reference)", "highest mass"):
        widths.append(dr.textlength(t, font=font))
    return img, strings, max(widths), colw - 24   # >= 24 px clear space before the next column


def run_s11(variant: str, f: int, verify_tiles: bool):
    from PIL import Image, ImageChops, ImageFont
    A = importlib.import_module("figS_r35_l1_l2_assembly")
    S, links = _s11_sources()
    src = A.out_dir("figS_r35_L1_representative_tiles") / "sources_20261104"
    A.check_manifest(A.S3, ["LEGEND.txt", "representative_links.csv"])
    A.check_manifest(src, [n for _, n, _ in KEPT])
    titles = {L: t for L, _, t in A.L1_PANELS_D1}
    groups = {f"montage_{tag}_{q}_vs_{n}_p01.png": Lk for (tag, q, n), Lk in links.groupby(["pair", "query", "reference"],
                                                                                           sort=False)}
    rec = {"panels": []}
    panels = []
    for new, name, old in KEPT:
        L = groups[name]
        im = Image.open(src / name).convert("RGB")
        assert _reproduce_old(S, L, im), f"{name}: accepted montage not reproduced from its tiles + cap_revised"
        tiles_ok = None
        if verify_tiles:
            import openslide
            tmpl = wsi_template([L["query"].iloc[0], L["reference"].iloc[0]])
            wq, wr = (openslide.OpenSlide(tmpl.format(s)) for s in (L["query"].iloc[0], L["reference"].iloc[0]))
            _, geo = _old_geometry(S, L)
            tiles_ok = 0
            for g in geo:
                for (ws, xy), b in zip(((wq, g["link"].query_xy), (wr, g["link"].ref_xy)), g["boxes"]):
                    x, yy = (float(v) for v in xy.split(","))
                    t = ws.read_region((int(x), int(yy)), 0, (512, 512)).convert("RGB").resize((S.TS, S.TS))
                    assert ImageChops.difference(t, im.crop(b)).getbbox() is None, (name, g["c"], g["kind"])
                    tiles_ok += 1
        if variant == "original":
            continue
        mont, strings, maxw, avail = _new_montage(S, L, im, f)
        assert maxw <= avail, (name, maxw, avail)
        title = titles[old]
        panels.append(A.labelled(mont, new, title))
        rec["panels"].append(dict(letter=new, former_letter=old, montage=name, title=title, n_rows=len(set(s["cluster"] for s in strings)),
                                  tiles_verified_against_wsi=tiles_ok, max_text_px=maxw, column_px=avail,
                                  captions=strings))
    if variant == "original":
        # the adopted S11 (revision) re-assembled from the accepted sources must reproduce the adopted PNG
        return {"montages_reproduced_from_tiles_and_cap_revised": [n for _, n, _ in KEPT]}
    canvas = A.stack(panels)
    d = vdir("S11", variant)
    png = d / "FigS_R3.5-L1_representative_tiles.png"
    canvas.save(png, dpi=(300, 300))
    canvas.convert("RGB").save(png.with_suffix(".pdf"), resolution=300)
    (d / "caption_values.json").write_text(json.dumps(rec, indent=1, default=str))
    rec.update(font=FONT, caption_px=f, canvas_px=list(canvas.size))
    write_provenance(d, [png, png.with_suffix(".pdf"), d / "caption_values.json"],
                     [src / n for _, n, _ in KEPT] + [A.S3 / "representative_links.csv", A.S3 / "LEGEND.txt",
                                                       A.S3 / "COMPLETE.json"],
                     "Pass 048 I3 (author decision 2026-11-16): Supplementary Fig. S11 limited to the three annotated "
                     "same-group pairs (former panels D-F, relabelled A-C; the three Fig. 5 pairs, former A-C, dropped). "
                     "Tiles: the 112-px tiles of the accepted sources_20261104 montages, pasted unchanged (each accepted "
                     "montage was first re-drawn from these tiles and the cap_revised strings of representative_links.csv "
                     "and found pixel-identical; tiles additionally compared with fresh read-only WSI reads). Sub-captions, "
                     f"column headers and row labels: same strings, DejaVuSans {f} px (was the PIL default 10-px font); "
                     "column headers wrapped after the kind name; 40-px gap between the kind columns (was 20). Panel titles: the accepted D1 titles of former D-F. "
                     "Panels stacked in one column (figS_r35_l1_l2_assembly.labelled/stack).",
                     Path(__file__), title="S11 (annotated pairs only), pass 048 prep")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("letters"); p.add_argument("--fig", choices=["S5", "S7"], required=True)
    p.add_argument("--variant", choices=["original", "upper"], required=True)
    p = sub.add_parser("s8"); p.add_argument("--variant", choices=["original", "ab"], required=True)
    p = sub.add_parser("s12"); p.add_argument("--variant", choices=["original", "delta"], required=True)
    p = sub.add_parser("s11"); p.add_argument("--variant", choices=["original", "annot"], required=True)
    p.add_argument("--font-px", type=int, default=16); p.add_argument("--verify-tiles", action="store_true")
    a = ap.parse_args()
    if a.cmd == "letters":
        r = run_letters(a.fig, a.variant)
    elif a.cmd == "s8":
        r = run_s8(a.variant)
    elif a.cmd == "s12":
        r = run_s12(a.variant)
    else:
        r = run_s11(a.variant, a.font_px, a.verify_tiles)
    print(json.dumps({k: v for k, v in r.items() if k != "panels"}, default=str)[:2000])


if __name__ == "__main__":
    main()
