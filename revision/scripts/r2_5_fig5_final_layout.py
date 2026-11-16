"""R2.5 Fig. 5 final-size layout rebuild ([assistant] R2.2/R2.5 challenge 80842afb..., finding 7; R2.5-T1).

Presentation work only. Takes the v2 composite directory written by scripts/r2_5_fig5_submitted_recipe_composite.py
(recipe.json + the saved level-3 H&E PNGs) and the ORIGINAL notebook overlay PNGs it references, authenticates every
image against the sha256 recorded in that recipe, and re-typesets the same content at the intended final width
(180 mm): one combined figure (panel A = two rows, panel B = one row; four columns per row: query H&E | query overlay
| neighbour H&E | neighbour overlay), all text >= 6 pt at print size, per-image 2-mm scale bars recomputed for the
final image scaling. No re-clustering, no overlay re-rendering, no WSI reads, no new computation.

Writes: Figure_5_revised.png (600 dpi) and Figure_5_revised.pdf, a layout_recipe.json with input/output hashes, and
legibility_check.json (font sizes, text extents vs cell bounds, scale-bar lengths predicted vs measured in the PNG).
Refuses a non-empty output directory.

Usage: python3 scripts/r2_5_fig5_final_layout.py V2_DIR OUT_DIR
"""
import hashlib
import json
import os
import platform
import socket
import sys
import textwrap
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 1 << 28
Image.MAX_IMAGE_PIXELS = None

WIDTH_MM = 180.0
WIDTH_IN = WIDTH_MM / 25.4
DPI = 600
FONT_TITLE = 6.0
FONT_LABEL = 6.0
FONT_PANEL = 9.0
FONT_NOTE = 6.0
MIN_FONT_PT = 6.0
SCALE_BAR_UM = 2000.0
# layout in inches
MARGIN_L, MARGIN_R = 0.22, 0.06
GAP_X = 0.08
TITLE_H = 0.46          # space reserved above each image for the wrapped title
IMG_H = 1.42            # image cell height
ROW_GAP = 0.12
TOP_PAD = 0.10
NOTE_H = 0.44
WRAP_CHARS = 33
THUMB = 1600

ROWS = [("A", 0), ("A", 1), ("B", 0)]  # (panel, pair index) in display order


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sha(path, expected, what):
    got = sha256_file(path)
    if got != expected:
        sys.exit(f"{what}: sha256 {got} != recipe-recorded {expected} ({path})")
    return got


def meta_row(meta, scan):
    rows = meta[meta["slide_id"].astype(str) == str(scan)]
    if len(rows) != 1:
        sys.exit(f"metadata: expected exactly one row for scan {scan}, found {len(rows)}")
    return rows.iloc[0]


def title_for(meta, scan, role, rank=None, distance=None):
    r = meta_row(meta, scan)
    s = f"{role}: {r['tumor_group']} | {r['Comp.Bio.ID']} | {r['Site']} | scan {scan}"
    if rank is not None:
        s += f" | rank {rank} (OT-cosine distance {distance:.3f})"
    return s


def wrap(s):
    return "\n".join(textwrap.wrap(s, WRAP_CHARS))


def main(v2_dir, out_dir):
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        sys.exit(f"refusing non-empty output directory {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    recipe_path = os.path.join(v2_dir, "recipe.json")
    v2 = json.load(open(recipe_path))
    meta_path = v2["metadata"]["path"]
    check_sha(meta_path, v2["metadata"]["sha256"], "metadata table")
    meta = pd.read_csv(meta_path, dtype=str)
    if int(len(meta)) != int(v2["metadata"]["rows"]):
        sys.exit("metadata row count differs from the v2 recipe")

    rec = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": os.path.relpath(__file__, os.getcwd()), "script_sha256": sha256_file(__file__),
        "command": " ".join(sys.argv), "cwd": os.getcwd(), "hostname": socket.gethostname(),
        "python": sys.version, "platform": platform.platform(),
        "purpose": "Astra 80842afb finding 7 / R2.5-T1: final-size layout of the v2 composites' content at 180 mm width, "
                   "text >= 6 pt; unchanged overlay and H&E renders; no re-clustering or new computation.",
        "v2_recipe": {"path": recipe_path, "sha256": sha256_file(recipe_path), "generated": v2["generated"],
                      "script_sha256": v2["script_sha256"]},
        "v2_composites": {p: v2["panels"][p]["composite_sha256"] for p in v2["panels"]},
        "metadata": v2["metadata"], "rank_source": v2["rank_source"],
        "width_mm": WIDTH_MM, "dpi": DPI, "fonts_pt": {"title": FONT_TITLE, "scale_label": FONT_LABEL,
                                                       "panel_letter": FONT_PANEL, "note": FONT_NOTE},
        "scale_bar_um": SCALE_BAR_UM, "level": v2["level"], "rows": [], "versions": {},
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec["versions"] = {"matplotlib": matplotlib.__version__, "PIL": Image.__version__, "pandas": pd.__version__,
                       "numpy": np.__version__}

    # ---- gather rows (authenticate every image) ----
    rows = []
    for panel, idx in ROWS:
        e = v2["panels"][panel]["pairs"][idx]
        q, n = e["query"], e["neighbour"]
        cells = []
        for sid, role in ((q, "Query"), (n, "Neighbour")):
            s = e["slides"][sid]
            he_path = s["he_png"]
            if not os.path.isabs(he_path):
                he_path = os.path.join(v2_dir, he_path)
            if not os.path.exists(he_path):  # the recipe stores the absolute path at generation; fall back to v2_dir
                he_path = os.path.join(v2_dir, os.path.basename(s["he_png"]))
            check_sha(he_path, s["he_sha256"], f"H&E {sid}")
            check_sha(s["overlay_saved_png"], s["overlay_sha256"], f"overlay {sid}")
            he = Image.open(he_path).convert("RGB")
            ov = Image.open(s["overlay_saved_png"]).convert("RGB")
            if he.size != tuple(s["level_dims"]) or ov.size != tuple(s["level_dims"]):
                sys.exit(f"image size mismatch for {sid}: he {he.size} ov {ov.size} recipe {s['level_dims']}")
            full_w = s["level_dims"][0]
            um_px = float(s["um_per_px_at_level"])
            t = title_for(meta, sid, role, e["rank"] if role == "Neighbour" else None,
                          e["distance"] if role == "Neighbour" else None)
            cells.append((he, t + "  (H&E)", full_w, um_px, sid, "he", he_path, s["he_sha256"]))
            cells.append((ov, f"{role}: OT correspondence communities overlaid on the H&E (this pair)", full_w, um_px,
                          sid, "overlay", s["overlay_saved_png"], s["overlay_sha256"]))
        rows.append((panel, q, n, e, cells))

    # ---- figure geometry ----
    n_rows = len(rows)
    cell_w = (WIDTH_IN - MARGIN_L - MARGIN_R - 3 * GAP_X) / 4.0
    fig_h = TOP_PAD + n_rows * (TITLE_H + IMG_H) + (n_rows - 1) * ROW_GAP + NOTE_H
    fig = plt.figure(figsize=(WIDTH_IN, fig_h))
    texts = []   # (artist, kind, cell bounds in figure inches or None)
    bars = []    # scale-bar records for the check
    y_top = fig_h - TOP_PAD
    prev_panel = None
    for ri, (panel, q, n, e, cells) in enumerate(rows):
        y_img_bottom = y_top - TITLE_H - IMG_H
        if panel != prev_panel:
            t = fig.text(0.02 / WIDTH_IN * 0 + 0.04 / WIDTH_IN, (y_top - 0.02) / fig_h, panel, fontsize=FONT_PANEL,
                         fontweight="bold", va="top", ha="left")
            texts.append((t, "panel_letter", None))
            prev_panel = panel
        row_rec = {"panel": panel, "query": q, "neighbour": n, "rank": e["rank"], "distance": e["distance"],
                   "resolution": e["resolution"], "plan_sha256": e["plan_sha256"], "cells": []}
        for ci, (im, title, full_w, um_px, sid, kind, src, sha) in enumerate(cells):
            x0 = MARGIN_L + ci * (cell_w + GAP_X)
            im = im.copy()
            im.thumbnail((THUMB, THUMB))
            # fit preserving aspect inside cell_w x IMG_H, anchored top-left
            sc = min(cell_w / im.width, IMG_H / im.height)
            disp_w, disp_h = im.width * sc, im.height * sc
            ax = fig.add_axes([x0 / WIDTH_IN, (y_top - TITLE_H - disp_h) / fig_h, disp_w / WIDTH_IN, disp_h / fig_h])
            ax.imshow(im, interpolation="lanczos")
            ax.set_xlim(0, im.width)
            ax.set_ylim(im.height, 0)
            ax.axis("off")
            tt = fig.text(x0 / WIDTH_IN, (y_top - 0.01) / fig_h, wrap(title), fontsize=FONT_TITLE, va="top", ha="left",
                          linespacing=1.15)
            texts.append((tt, "title", (x0, x0 + cell_w, y_top - TITLE_H, y_top)))
            # 2-mm bar in displayed-image data units: bar_px(level) * thumbnail scale
            thumb_scale = im.width / float(full_w)
            bar_data = SCALE_BAR_UM / um_px * thumb_scale
            bx0, by0 = im.width * 0.04, im.height * 0.95
            ax.plot([bx0, bx0 + bar_data], [by0, by0], color="black", linewidth=1.4, solid_capstyle="butt")
            lt = ax.text(bx0, by0 - im.height * 0.02, f"{SCALE_BAR_UM/1000:.0f} mm", fontsize=FONT_LABEL,
                         color="black", va="bottom", ha="left",
                         bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.6))
            texts.append((lt, "scale_label", (x0, x0 + cell_w, y_top - TITLE_H - IMG_H, y_top - TITLE_H)))
            bar_in = bar_data / im.width * disp_w   # inches on the page
            bars.append({"row": ri, "col": ci, "slide": sid, "kind": kind, "ax": ax, "bx0": bx0, "bx1": bx0 + bar_data,
                         "by0": by0, "bar_inches": bar_in, "bar_mm_on_page": bar_in * 25.4,
                         "um_per_px_level": um_px, "level_width_px": full_w, "displayed_width_in": disp_w})
            row_rec["cells"].append({"col": ci, "slide": sid, "kind": kind, "source": src, "sha256": sha,
                                     "title": title, "displayed_size_in": [float(disp_w), float(disp_h)],
                                     "scale_bar_mm_on_page": float(bar_in * 25.4)})
        rec["rows"].append(row_rec)
        y_top = y_img_bottom - ROW_GAP
    note = ("Colours are specific to each query\u2013neighbour pair (one clustering per row); do not compare colours "
            "across rows. Images are not at a common scale; each carries a 2-mm bar. Site tags as recorded in the "
            "COMET metadata: (P) primary, (M) metastatic. Ranks and distances: UNI2 features, exact Optimal "
            "Transport with cosine ground cost, leave-one-out over the 2,778-slide library, same-patient slides "
            "excluded.")
    nt = fig.text(MARGIN_L / WIDTH_IN, 0.08 / fig_h, wrap_note(note, WIDTH_IN - MARGIN_L - MARGIN_R), fontsize=FONT_NOTE,
                  va="bottom", ha="left", linespacing=1.15)
    texts.append((nt, "note", (MARGIN_L, WIDTH_IN - MARGIN_R, 0.0, NOTE_H)))

    out_png = os.path.join(out_dir, "Figure_5_revised.png")
    out_pdf = os.path.join(out_dir, "Figure_5_revised.pdf")
    # Production handoff ([assistant] layout review 1c0c99a7 findings 1-2): PDF raster images at full DPI, TrueType (Type 42)
    # fonts, and an opaque RGB PNG without an alpha channel.
    matplotlib.rcParams["pdf.fonttype"] = 42
    fig.savefig(out_png, dpi=DPI, facecolor="white")
    fig.savefig(out_pdf, dpi=DPI, facecolor="white")
    with Image.open(out_png) as im_png:
        rgb = Image.new("RGB", im_png.size, (255, 255, 255))
        rgb.paste(im_png.convert("RGBA"), mask=im_png.convert("RGBA").split()[3])
        rgb.save(out_png, dpi=(DPI, DPI))

    # ---- legibility / geometry check (on the figure as rendered) ----
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    check = {"width_mm": WIDTH_MM, "height_mm": fig_h * 25.4, "dpi": DPI, "min_font_pt_required": MIN_FONT_PT,
             "texts": [], "scale_bars": [], "all_fonts_ok": True, "all_texts_within_cells": True, "overlaps": []}
    fdpi = fig.dpi
    boxes = []
    for art, kind, bounds in texts:
        bb = art.get_window_extent(renderer)
        x0, x1 = bb.x0 / fdpi, bb.x1 / fdpi
        y0, y1 = bb.y0 / fdpi, bb.y1 / fdpi
        ok_font = bool(art.get_fontsize() >= MIN_FONT_PT)
        within = True
        if bounds is not None:
            bx0, bx1, by0, by1 = bounds
            within = bool(x0 >= bx0 - 0.01 and x1 <= bx1 + 0.01 and y0 >= by0 - 0.01 and y1 <= by1 + 0.01)
        x0, y0, x1, y1 = (float(v) for v in (x0, y0, x1, y1))
        check["texts"].append({"kind": kind, "fontsize_pt": float(art.get_fontsize()), "text": art.get_text(),
                               "extent_in": [x0, y0, x1, y1], "cell_bounds_in": bounds, "font_ok": ok_font,
                               "within_cell": within})
        check["all_fonts_ok"] = bool(check["all_fonts_ok"] and ok_font)
        check["all_texts_within_cells"] = bool(check["all_texts_within_cells"] and within)
        boxes.append((kind, x0, y0, x1, y1))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]:
                check["overlaps"].append([a[0], b[0], list(a[1:]), list(b[1:])])
    # measure the bars in the saved PNG: dark run along the bar row
    png = np.asarray(Image.open(out_png).convert("RGB"))
    Hpx = png.shape[0]
    for b in bars:
        p0 = b["ax"].transData.transform((b["bx0"], b["by0"]))
        p1 = b["ax"].transData.transform((b["bx1"], b["by0"]))
        sx = DPI / fdpi
        px0, px1 = p0[0] * sx, p1[0] * sx
        py = Hpx - p0[1] * sx
        predicted = px1 - px0
        yi = int(round(py))
        xs = slice(max(0, int(px0) - 40), min(png.shape[1], int(px1) + 40))
        best = 0
        for dy in range(-6, 7):
            yy = yi + dy
            if 0 <= yy < Hpx:
                dark = (png[yy, xs].max(axis=1) < 80)
                run, cur = 0, 0
                for v in dark:
                    cur = cur + 1 if v else 0
                    run = max(run, cur)
                best = max(best, run)
        ratio = float(best / predicted) if predicted > 0 else float("nan")
        predicted = float(predicted)
        check["scale_bars"].append({"row": b["row"], "col": b["col"], "slide": b["slide"], "kind": b["kind"],
                                    "bar_mm_on_page": float(b["bar_mm_on_page"]), "predicted_px": predicted,
                                    "measured_dark_run_px": int(best), "measured_over_predicted": ratio,
                                    "um_per_px_level": b["um_per_px_level"], "level_width_px": b["level_width_px"],
                                    "displayed_width_in": float(b["displayed_width_in"])})
    check["scale_bars_ok"] = bool(all(0.97 <= s["measured_over_predicted"] <= 1.05 for s in check["scale_bars"]))
    check["pass"] = bool(check["all_fonts_ok"] and check["all_texts_within_cells"] and not check["overlaps"]
                         and check["scale_bars_ok"])
    plt.close(fig)
    with open(os.path.join(out_dir, "legibility_check.json"), "w") as f:
        json.dump(check, f, indent=1)
    rec["outputs"] = {"Figure_5_revised.png": {"sha256": sha256_file(out_png), "pixels": list(Image.open(out_png).size)},
                      "Figure_5_revised.pdf": {"sha256": sha256_file(out_pdf)},
                      "legibility_check.json": {"pass": check["pass"]}}
    with open(os.path.join(out_dir, "layout_recipe.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({"png": rec["outputs"]["Figure_5_revised.png"], "legibility_pass": check["pass"],
                      "overlaps": len(check["overlaps"]), "fonts_ok": check["all_fonts_ok"],
                      "within": check["all_texts_within_cells"], "bars_ok": check["scale_bars_ok"]}, indent=1))
    if not check["pass"]:
        sys.exit(2)


def wrap_note(s, width_in):
    chars = int(width_in * 72 / (FONT_NOTE * 0.58))
    return "\n".join(textwrap.wrap(s, chars))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
