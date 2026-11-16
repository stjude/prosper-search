"""Fig. 5 (R2.5-T1): publication layout of the Optimal Transport correspondence communities with an H&E
reference column (A review comment: "Fig. 5 should include the original H&E image for reference").

Presentation work only. Uses exactly the six level-3 H&E renders and the six ORIGINAL notebook overlay renders
recorded in the accepted v2 recipe ($PROSPER_OUT/r2_5_fig5_submitted_recipe_20261006/recipe.json), each
authenticated by SHA-256 before use. No re-clustering, no overlay re-rendering, no WSI read, no new computation.

Differences from the 10-08 production_002 layout (same content): internal identifiers and audit wording are
removed from the artwork (scan numbers, "(P)"/"(M)" site tags, "COMET metadata", "(this pair)", "(H&E)", the
printed method note); the column meaning is given once by spanning headers (Query / Retrieved neighbour x
H&E / Correspondence communities); each row carries one short identity label per slide (group, site, slide ID,
and for the neighbour its rank and Optimal Transport distance); images are centred in their cells; every image
keeps its own 2-mm scale bar because panels are not at a common scale. Everything else (colour semantics, scale,
rank source) is stated in the legend ([internal record] R2.5-T2), not on the figure.

Writes to $PROSPER_OUT/figures_20261021/fig5_ot_correspondence/: Figure_5_revised.png (600 dpi), .pdf (TrueType),
panel_manifest.csv (every placed image with source path, hash, displayed size, bar length and label),
legibility_check.json (font sizes >= 6 pt, text within cells, no overlaps, bars measured in the PNG),
SHA256SUMS.txt and PROVENANCE.md.

Usage (from the repository root):
    export PROSPER_OUT="$PWD/prosper_revision_output"
    /HOME/miniconda3/bin/python3 scripts/figures/fig5_ot_correspondence.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import FIGURE_IDS, out_dir, prosper_out, sha256, write_provenance  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PngImagePlugin.MAX_TEXT_CHUNK = 1 << 28
Image.MAX_IMAGE_PIXELS = None

FIGURE = "fig5_ot_correspondence"
STEM = "Figure_5_revised"
V2_DIR_NAME = "r2_5_fig5_submitted_recipe_20261006"

# Accepted identities ([internal record] R2.5-T1/T2); the script fails if the recipe or metadata disagree.
EXPECTED = {
    ("A", 0): dict(query="16875", neighbour="16867", rank=1, distance=0.399, resolution=4e-05),
    ("A", 1): dict(query="16875", neighbour="15155", rank=10, distance=0.458, resolution=4e-05),
    ("B", 0): dict(query="7333", neighbour="7330", rank=1, distance=0.314, resolution=0.0005),
}
EXPECTED_IDS = dict(FIGURE_IDS["fig5_expected_ids"])

# ---- page geometry (inches) ----
WIDTH_MM = 180.0
WIDTH_IN = WIDTH_MM / 25.4
DPI = 600
MARGIN_L, MARGIN_R = 0.24, 0.04      # room for the panel letters on the left
GAP_X = 0.07                          # between the four image columns
GAP_PAIR = 0.16                       # extra gap between the query block and the neighbour block
HEADER_H = 0.36                       # spanning headers + column sub-headers
LABEL_H = 0.27                        # identity (and rank/distance) lines per slide, above its two images
ROW_H = {"A": 1.30, "B": 1.14}        # image cell height per panel (fits the tallest image in that panel)
ROW_GAP = 0.08
PANEL_GAP = 0.22
TOP_PAD, BOTTOM_PAD = 0.04, 0.05
SCALE_BAR_UM = 2000.0

FONT_HEADER = 8.0
FONT_SUBHEADER = 7.0
FONT_LABEL = 7.0
FONT_BAR = 6.5
FONT_PANEL = 10.0
MIN_FONT_PT = 6.0

ROWS = [("A", 0), ("A", 1), ("B", 0)]


def fail(msg: str) -> None:
    sys.exit(f"fig5_ot_correspondence: {msg}")


def check_sha(path: str, expected: str, what: str) -> str:
    got = sha256(Path(path))
    if got != expected:
        fail(f"{what}: sha256 {got} != recipe-recorded {expected} ({path})")
    return got


def resolve_he(v2_dir: Path, recorded: str) -> Path:
    # The recipe stores the absolute path at generation time (possibly via the other mount); use the basename.
    p = v2_dir / Path(recorded).name
    if not p.exists():
        fail(f"H&E render missing: {p}")
    return p


def slide_label(meta_row: pd.Series, rank: int | None = None, distance: float | None = None) -> str:
    group = str(meta_row["tumor_group"])
    # The metadata Site field carries a primary/metastatic tag, e.g. "Testicle (P)"; the legend states that all
    # displayed slides are primary-site, so the tag is not printed.
    site = re.sub(r"\s*\((P|M)\)\s*$", "", str(meta_row["Site"]).strip()).lower()
    sid = str(meta_row["Comp.Bio.ID"])
    s = f"{group} \u00b7 {site} \u00b7 {sid}"
    if rank is not None:
        s += f"\nRank {rank} \u00b7 Optimal Transport distance {distance:.3f}"
    return s


def main() -> None:
    root = prosper_out()
    v2_dir = root / V2_DIR_NAME
    recipe_path = v2_dir / "recipe.json"
    if not recipe_path.exists():
        fail(f"accepted v2 recipe not found: {recipe_path}")
    v2 = json.load(open(recipe_path))
    inputs = [recipe_path]

    meta_path = v2["metadata"]["path"]
    check_sha(meta_path, v2["metadata"]["sha256"], "identity metadata table")
    meta = pd.read_csv(meta_path, dtype=str)
    if len(meta) != int(v2["metadata"]["rows"]):
        fail("metadata row count differs from the v2 recipe")
    inputs.append(Path(meta_path))

    def meta_row(scan: str) -> pd.Series:
        rows = meta[meta["slide_id"].astype(str) == scan]
        if len(rows) != 1:
            fail(f"metadata: expected one row for scan {scan}, found {len(rows)}")
        r = rows.iloc[0]
        if str(r["Comp.Bio.ID"]) != EXPECTED_IDS[scan]:
            fail(f"scan {scan}: slide ID {r['Comp.Bio.ID']} != accepted {EXPECTED_IDS[scan]}")
        return r

    # ---- gather and authenticate every image ----
    rows = []
    for panel, idx in ROWS:
        e = v2["panels"][panel]["pairs"][idx]
        exp = EXPECTED[(panel, idx)]
        if (str(e["query"]), str(e["neighbour"]), int(e["rank"])) != (exp["query"], exp["neighbour"], exp["rank"]):
            fail(f"pair {panel}{idx}: recipe identities {e['query']}/{e['neighbour']}/{e['rank']} != accepted")
        if abs(float(e["distance"]) - exp["distance"]) > 5e-4 or float(e["resolution"]) != exp["resolution"]:
            fail(f"pair {panel}{idx}: distance/resolution differ from the accepted values")
        cells = []
        for sid, role in ((str(e["query"]), "query"), (str(e["neighbour"]), "neighbour")):
            s = e["slides"][sid]
            he_path = resolve_he(v2_dir, s["he_png"])
            check_sha(str(he_path), s["he_sha256"], f"H&E {sid}")
            check_sha(s["overlay_saved_png"], s["overlay_sha256"], f"overlay {sid}")
            inputs += [he_path, Path(s["overlay_saved_png"])]
            he = Image.open(he_path).convert("RGB")
            ov = Image.open(s["overlay_saved_png"]).convert("RGB")
            if he.size != tuple(s["level_dims"]) or ov.size != tuple(s["level_dims"]):
                fail(f"image size mismatch for {sid}: he {he.size} ov {ov.size} recipe {s['level_dims']}")
            r = meta_row(sid)
            label = slide_label(r, e["rank"] if role == "neighbour" else None,
                                float(e["distance"]) if role == "neighbour" else None)
            um_px = float(s["um_per_px_at_level"])
            cells.append(dict(im=he, kind="he", slide=sid, role=role, src=str(he_path), sha=s["he_sha256"],
                              um_px=um_px, label=label))
            cells.append(dict(im=ov, kind="overlay", slide=sid, role=role, src=s["overlay_saved_png"],
                              sha=s["overlay_sha256"], um_px=um_px, label=label))
        rows.append(dict(panel=panel, idx=idx, pair=e, cells=cells))

    # ---- geometry ----
    cell_w = (WIDTH_IN - MARGIN_L - MARGIN_R - 2 * GAP_X - GAP_PAIR) / 4.0
    col_x = [MARGIN_L, MARGIN_L + cell_w + GAP_X,
             MARGIN_L + 2 * cell_w + GAP_X + GAP_PAIR, MARGIN_L + 3 * cell_w + 2 * GAP_X + GAP_PAIR]
    fig_h = TOP_PAD + HEADER_H + BOTTOM_PAD
    prev = None
    for r in rows:
        if prev is not None:
            fig_h += PANEL_GAP if r["panel"] != prev else ROW_GAP
        fig_h += LABEL_H + ROW_H[r["panel"]]
        prev = r["panel"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(WIDTH_IN, fig_h))
    fx, fy = (lambda x: x / WIDTH_IN), (lambda y: y / fig_h)
    texts, bars, manifest = [], [], []

    # ---- spanning headers ----
    y = fig_h - TOP_PAD
    for (c0, c1, title) in ((0, 1, "Query slide"), (2, 3, "Retrieved neighbor")):
        xm = (col_x[c0] + col_x[c1] + cell_w) / 2
        t = fig.text(fx(xm), fy(y), title, fontsize=FONT_HEADER, fontweight="bold", ha="center", va="top")
        texts.append((t, "header", (col_x[c0], col_x[c1] + cell_w, y - 0.16, y)))
        ln = fig.add_axes([fx(col_x[c0]), fy(y - 0.17), fx(col_x[c1] + cell_w - col_x[c0]), 0.0005])
        ln.set_facecolor("black"); ln.set_xticks([]); ln.set_yticks([])
        for sp in ln.spines.values():
            sp.set_visible(False)
    for c, sub in enumerate(["H&E", "Correspondence communities", "H&E", "Correspondence communities"]):
        t = fig.text(fx(col_x[c] + cell_w / 2), fy(y - 0.20), sub, fontsize=FONT_SUBHEADER, ha="center", va="top")
        texts.append((t, "subheader", (col_x[c], col_x[c] + cell_w, y - HEADER_H, y - 0.18)))
    y -= HEADER_H

    # ---- rows ----
    prev = None
    for ri, r in enumerate(rows):
        panel = r["panel"]
        if prev is not None:
            y -= PANEL_GAP if panel != prev else ROW_GAP
        if panel != prev:
            t = fig.text(fx(0.03), fy(y + 0.01), panel, fontsize=FONT_PANEL, fontweight="bold", ha="left", va="top")
            texts.append((t, "panel_letter", None))
        prev = panel
        row_h = ROW_H[panel]
        # identity labels, one per slide, spanning that slide's two columns
        for (c0, cell) in ((0, r["cells"][0]), (2, r["cells"][2])):
            t = fig.text(fx(col_x[c0]), fy(y - 0.005), cell["label"], fontsize=FONT_LABEL, ha="left", va="top",
                         linespacing=1.25)
            texts.append((t, "label", (col_x[c0], col_x[c0 + 1] + cell_w, y - LABEL_H, y)))
        y_img_top = y - LABEL_H
        for ci, cell in enumerate(r["cells"]):
            im = cell["im"]
            full_w, full_h = im.size
            sc_in = min(cell_w / full_w, row_h / full_h)         # inches per level-3 pixel
            disp_w, disp_h = full_w * sc_in, full_h * sc_in
            px_w, px_h = int(round(disp_w * DPI)), int(round(disp_h * DPI))
            small = im.resize((px_w, px_h), Image.LANCZOS)
            x0 = col_x[ci] + (cell_w - disp_w) / 2
            y0 = y_img_top - row_h + (row_h - disp_h) / 2
            ax = fig.add_axes([fx(x0), fy(y0), fx(disp_w), fy(disp_h)])
            ax.imshow(small, interpolation="lanczos")
            ax.set_xlim(0, px_w); ax.set_ylim(px_h, 0); ax.axis("off")
            # 2-mm bar in displayed pixels
            bar_px = SCALE_BAR_UM / cell["um_px"] * (px_w / full_w)
            bx0, by0 = px_w * 0.04, px_h * 0.955
            ax.plot([bx0, bx0 + bar_px], [by0, by0], color="black", linewidth=1.3, solid_capstyle="butt")
            lt = ax.text(bx0, by0 - px_h * 0.015, "2 mm", fontsize=FONT_BAR, ha="left", va="bottom",
                         bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.5))
            texts.append((lt, "scale_label", (x0, x0 + disp_w, y0, y0 + disp_h)))
            bar_in = bar_px / px_w * disp_w
            bars.append(dict(row=ri, col=ci, slide=cell["slide"], kind=cell["kind"], ax=ax, bx0=bx0,
                             bx1=bx0 + bar_px, by0=by0, bar_mm=bar_in * 25.4))
            manifest.append(dict(panel=panel, row=ri, col=ci, role=cell["role"], kind=cell["kind"],
                                 slide_scan=cell["slide"], slide_id=EXPECTED_IDS[cell["slide"]],
                                 label=cell["label"], source=cell["src"], sha256=cell["sha"],
                                 level_px=f"{full_w}x{full_h}", um_per_px_level=cell["um_px"],
                                 displayed_mm=f"{disp_w*25.4:.1f}x{disp_h*25.4:.1f}",
                                 scale_bar_mm_on_page=round(bar_in * 25.4, 2),
                                 rank=r["pair"]["rank"] if cell["role"] == "neighbour" else "",
                                 ot_distance=round(float(r["pair"]["distance"]), 3) if cell["role"] == "neighbour" else "",
                                 louvain_resolution=r["pair"]["resolution"]))
        y = y_img_top - row_h

    # ---- save ----
    d = out_dir(FIGURE)
    png, pdf = d / f"{STEM}.png", d / f"{STEM}.pdf"
    fig.savefig(png, dpi=DPI, facecolor="white")
    fig.savefig(pdf, dpi=DPI, facecolor="white", metadata={"CreationDate": None})
    with Image.open(png) as im_png:
        rgb = Image.new("RGB", im_png.size, (255, 255, 255))
        rgba = im_png.convert("RGBA")
        rgb.paste(rgba, mask=rgba.split()[3])
        rgb.save(png, dpi=(DPI, DPI))

    # ---- legibility / geometry check on the rendered page ----
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fdpi = fig.dpi
    check = dict(width_mm=WIDTH_MM, height_mm=round(fig_h * 25.4, 1), dpi=DPI, min_font_pt_required=MIN_FONT_PT,
                 texts=[], scale_bars=[], overlaps=[], all_fonts_ok=True, all_texts_within_cells=True)
    boxes = []
    for art, kind, bounds in texts:
        bb = art.get_window_extent(renderer)
        x0, x1, y0, y1 = (float(v) for v in (bb.x0 / fdpi, bb.x1 / fdpi, bb.y0 / fdpi, bb.y1 / fdpi))
        ok_font = bool(art.get_fontsize() >= MIN_FONT_PT)
        within = True
        if bounds is not None:
            b0, b1, c0, c1 = bounds
            within = bool(x0 >= b0 - 0.01 and x1 <= b1 + 0.01 and y0 >= c0 - 0.01 and y1 <= c1 + 0.01)
        check["texts"].append(dict(kind=kind, fontsize_pt=float(art.get_fontsize()), text=art.get_text(),
                                   extent_in=[x0, y0, x1, y1], cell_bounds_in=bounds, font_ok=ok_font,
                                   within_cell=within))
        check["all_fonts_ok"] = bool(check["all_fonts_ok"] and ok_font)
        check["all_texts_within_cells"] = bool(check["all_texts_within_cells"] and within)
        boxes.append((kind, x0, y0, x1, y1))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]:
                check["overlaps"].append([a[0], b[0], list(a[1:]), list(b[1:])])
    arr = np.asarray(Image.open(png).convert("RGB"))
    H = arr.shape[0]
    sx = DPI / fdpi
    for b in bars:
        p0 = b["ax"].transData.transform((b["bx0"], b["by0"]))
        p1 = b["ax"].transData.transform((b["bx1"], b["by0"]))
        px0, px1, py = p0[0] * sx, p1[0] * sx, H - p0[1] * sx
        predicted = float(px1 - px0)
        xs = slice(max(0, int(px0) - 40), min(arr.shape[1], int(px1) + 40))
        best = 0
        for dy in range(-6, 7):
            yy = int(round(py)) + dy
            if 0 <= yy < H:
                dark = arr[yy, xs].max(axis=1) < 80
                run = cur = 0
                for v in dark:
                    cur = cur + 1 if v else 0
                    run = max(run, cur)
                best = max(best, run)
        check["scale_bars"].append(dict(row=b["row"], col=b["col"], slide=b["slide"], kind=b["kind"],
                                        bar_mm_on_page=round(b["bar_mm"], 3), predicted_px=predicted,
                                        measured_dark_run_px=int(best),
                                        measured_over_predicted=float(best / predicted)))
    check["scale_bars_ok"] = bool(all(0.97 <= s["measured_over_predicted"] <= 1.05 for s in check["scale_bars"]))
    check["pass"] = bool(check["all_fonts_ok"] and check["all_texts_within_cells"] and not check["overlaps"]
                         and check["scale_bars_ok"])
    plt.close(fig)
    (d / "legibility_check.json").write_text(json.dumps(check, indent=1))
    man = d / "panel_manifest.csv"
    pd.DataFrame(manifest).to_csv(man, index=False)

    note = ("Presentation-only rebuild of Fig. 5 for R2.5-T1 (H&E reference column beside each correspondence "
            "overlay). Source images: the six level-3 H&E renders and six original notebook overlay renders of the "
            f"accepted v2 recipe ({V2_DIR_NAME}/recipe.json), each SHA-256-authenticated; no re-clustering, "
            "re-rendering or new computation. Internal identifiers (scan numbers, site tags, metadata wording, "
            "method note) removed from the artwork; column semantics given by spanning headers; one identity label "
            "per slide; images centred; per-image 2-mm scale bars (panels are not at a common scale). Legend: edit "
            "sheet R2.5-T2 (proposed amendment: drop the '(P)' sentence, state that all slides are primary-site). "
            f"Legibility check pass = {check['pass']}.")
    write_provenance(d, [png, pdf, man, d / "legibility_check.json"], inputs, note, Path(__file__),
                     title=f"{FIGURE} / {STEM}")
    print(json.dumps(dict(png=str(png), sha256=sha256(png)[:16], size_px=list(Image.open(png).size),
                          page_mm=[WIDTH_MM, round(fig_h * 25.4, 1)], legibility_pass=check["pass"],
                          overlaps=len(check["overlaps"]), bars_ok=check["scale_bars_ok"]), indent=1))
    if not check["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
