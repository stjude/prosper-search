#!/usr/bin/env python3
"""revision: minimal text-box edits to the submitted Fig. 1 schematic (E1-RU1).

Reads the preserved submitted raster (never modified) and writes revised
rasters of identical size/dpi. Each edit erases a label only where the
erase rectangle contains nothing but the flat panel/box fill and black
(or white-on-black) glyph pixels, verified before writing; new labels are
anti-aliased Noto Sans (closest available match to the original Myriad).

Edits (E1-RU1, adopted 2026-11-13, plus revision requests):
  1. LLM Summarization box -> verbatim cited excerpt of the retained
     SJRHB073134_D1 summary quoted in the main Results, truncated with an
     ellipsis at a sentence boundary (no query diagnosis, no "XXX").
  2. "Disease Probability" -> "Bayesian tumor-group probabilities".
  3. ROI branch "Search by Min L1" -> "Rank by closest ROI match".
     Pooled branch "Search by Min L1" -> "Nearest-neighbor search" (analyst
     revision rule; the strict-E1-RU1 variant keeps the submitted label).
  4. Bypass from tiled slide to feature extraction labelled
     "Full-tile input (specified comparators)".
  5. "Foundation Model" -> "Tile encoder".
  Request beyond E1-RU1: "Mosaic Tiles" -> "Mosaic Tiles (15%)".
  "LIV." is left unchanged (no legible space for "LIVER" without restyling
  the axis labels); bar heights are left unchanged (illustrative content).
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs/prosper_submission/fig/Figure_1.png"
MAIN = ROOT / "docs/prosper_revised/prosper_manuscript_revised.md"
FONT = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"

PINK = (187, 144, 156)
LAVENDER = (233, 232, 244)
PEACH = (254, 220, 196)
BLUE = (208, 231, 248)
GREEN = (214, 236, 223)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

LOG = []


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def font(size):
    return ImageFont.truetype(FONT, size)


def check_ring(a, rect, fill, tol=3, name=""):
    x0, y0, x1, y1 = rect
    ring = np.concatenate([
        a[y0 - 1, x0 - 1:x1 + 2], a[y1 + 1, x0 - 1:x1 + 2],
        a[y0 - 1:y1 + 2, x0 - 1], a[y0 - 1:y1 + 2, x1 + 1]])
    bad = int((np.abs(ring.astype(int) - fill).max(1) > tol).sum())
    assert bad == 0, f"{name}: erase ring not flat fill ({bad} px)"


def check_text_only(a, rect, fill, ink, tol=14, name=""):
    """Every pixel in rect must lie on the segment ink->fill (glyph AA)."""
    x0, y0, x1, y1 = rect
    r = a[y0:y1 + 1, x0:x1 + 1].astype(float)
    f = np.array(fill, float)
    k = np.array(ink, float)
    d = k - f
    t = np.clip(((r - f) @ d) / (d @ d), 0, 1)
    resid = np.abs(r - (f + t[..., None] * d)).max(2)
    bad = int((resid > tol).sum())
    assert bad == 0, f"{name}: non-text pixels inside erase rect ({bad} px)"
    return int((t > 0.5).sum())


def erase(a, rect, fill, ink=BLACK, name=""):
    check_ring(a, rect, fill, name=name)
    n = check_text_only(a, rect, fill, ink, name=name)
    x0, y0, x1, y1 = rect
    a[y0:y1 + 1, x0:x1 + 1] = fill
    LOG.append({"op": "erase", "name": name, "rect": rect, "fill": fill,
                "ink_pixels_removed": n})


def check_clear(a, rect, fill, tol=3, name=""):
    x0, y0, x1, y1 = rect
    r = a[y0:y1 + 1, x0:x1 + 1].astype(int)
    bad = int((np.abs(r - fill).max(2) > tol).sum())
    assert bad == 0, f"{name}: target area not clear ({bad} px)"


def text_mask(lines, size, pitch=None, align="center"):
    f = font(size)
    asc, desc = f.getmetrics()
    pitch = pitch or int(round(size * 1.2))
    widths = [f.getlength(t) for t in lines]
    W = int(max(widths)) + 20
    H = pitch * (len(lines) - 1) + asc + desc + 20
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for i, t in enumerate(lines):
        x = 10 if align == "left" else 10 + (W - 20 - widths[i]) / 2
        d.text((x, 10 + i * pitch), t, font=f, fill=255)
    return m


def place(a, mask, center=None, topleft=None, color=BLACK, fill=None,
          rotate=False, name="", inside=None):
    if rotate:
        mask = mask.rotate(90, expand=True)
    bb = mask.getbbox()
    mask = mask.crop(bb)
    w, h = mask.size
    if center is not None:
        x0 = int(round(center[0] - w / 2))
        y0 = int(round(center[1] - h / 2))
    else:
        x0, y0 = topleft
    rect = (x0, y0, x0 + w - 1, y0 + h - 1)
    if inside is not None:
        ix0, iy0, ix1, iy1 = inside
        assert ix0 <= rect[0] and iy0 <= rect[1] and rect[2] <= ix1 and rect[3] <= iy1, \
            f"{name}: text {rect} exceeds allowed area {inside}"
    if fill is not None:
        check_clear(a, (rect[0] - 3, rect[1] - 3, rect[2] + 3, rect[3] + 3), fill, name=name)
    al = np.asarray(mask, float)[..., None] / 255.0
    reg = a[y0:y0 + h, x0:x0 + w].astype(float)
    a[y0:y0 + h, x0:x0 + w] = np.round(reg * (1 - al) + np.array(color, float) * al).astype(np.uint8)
    LOG.append({"op": "draw", "name": name, "ink_bbox": rect})
    return rect


def llm_excerpt(width_px, size, max_lines):
    text = None
    for line in MAIN.read_text(encoding="utf-8").splitlines():
        if line.lstrip("> ").startswith("A 7-year-old male with a primary abdominal tumor, untreated."):
            text = line.lstrip("> ").strip()
            break
    assert text, "excerpt source not found in main manuscript"
    f = font(size)
    words = text.split(" ")

    def wrap(ws):
        lines, cur = [], ""
        for w_ in ws:
            cand = (cur + " " + w_).strip()
            if f.getlength(cand) <= width_px:
                cur = cand
            else:
                lines.append(cur)
                cur = w_
        lines.append(cur)
        return lines

    best = None
    for n in range(1, len(words) + 1):
        pref = " ".join(words[:n])
        if pref.count("[") != pref.count("]") or pref.count("(") != pref.count(")"):
            continue
        if not re.search(r"[.;]$", pref):
            continue
        lines = wrap((pref + " \u2026").split(" "))
        if len(lines) <= max_lines:
            best = (pref, lines)
    pref, lines = best
    assert text.startswith(pref)
    return text, pref, lines


def fit_box(a):
    """Fit the black rounded LLM box geometry to the submitted raster."""
    sub = a[1640:2170, 7550:8590].max(2) < 60
    best = None
    for r in range(76, 96, 2):
        for y0 in (1663, 1664, 1665):
            for y1 in (2148, 2149, 2150):
                for x0 in (7575, 7576, 7577):
                    for x1 in (8562, 8563, 8564):
                        m = Image.new("L", (1040 * 4, 530 * 4), 0)
                        ImageDraw.Draw(m).rounded_rectangle(
                            ((x0 - 7550) * 4, (y0 - 1640) * 4, (x1 + 1 - 7550) * 4 - 1, (y1 + 1 - 1640) * 4 - 1),
                            radius=r * 4, fill=255)
                        mm = np.asarray(m.resize((1040, 530), Image.LANCZOS)) > 127
                        # compare on corner/edge band only (text inside is white)
                        band = np.zeros_like(mm)
                        band[:, :] = True
                        band[30:-30, 40:-40] = False
                        err = int(((mm != sub) & band).sum())
                        if best is None or err < best[0]:
                            best = (err, r, x0, y0, x1, y1)
    return best


def build(strict):
    img = Image.open(SRC)
    dpi = img.info.get("dpi", (300, 300))
    a = np.array(img.convert("RGB"))
    H, W = a.shape[:2]
    LOG.clear()

    # 5. Foundation Model -> Tile encoder (two vertical lines, original size)
    erase(a, (3030, 948, 3171, 1273), PINK, name="Foundation Model")
    place(a, text_mask(["Tile", "encoder"], 58, pitch=80), center=(3101, 1111), rotate=True,
          fill=PINK, name="Tile encoder", inside=(3030, 948, 3171, 1273))

    # 3. ROI branch Search by Min L1 -> Rank by closest ROI match
    erase(a, (5068, 1421, 5143, 1887), PINK, name="ROI Search by Min L1")
    place(a, text_mask(["Rank by closest", "ROI match"], 46, pitch=50), center=(5097, 1649), rotate=True,
          fill=PINK, name="Rank by closest ROI match", inside=(5045, 1410, 5150, 1890))

    # Pooled branch (analyst rule; strict variant keeps submitted label)
    if not strict:
        erase(a, (5070, 342, 5145, 808), PINK, name="Slide Search by Min L1")
        place(a, text_mask(["Nearest-neighbor", "search"], 46, pitch=50), center=(5100, 570), rotate=True,
              fill=PINK, name="Nearest-neighbor search", inside=(5048, 330, 5153, 812))

    # 2. Disease Probability -> Bayesian tumor-group probabilities
    erase(a, (7786, 840, 8383, 915), PEACH, name="Disease Probability")
    place(a, text_mask(["Bayesian tumor-group probabilities"], 56), center=(8085, 880),
          fill=PEACH, name="Bayesian tumor-group probabilities", inside=(7560, 820, 8590, 935))

    # Mosaic Tiles -> Mosaic Tiles (15%)
    erase(a, (2239, 706, 2572, 765), LAVENDER, name="Mosaic Tiles")
    place(a, text_mask(["Mosaic Tiles (15%)"], 57), center=(2405, 736),
          fill=LAVENDER, name="Mosaic Tiles (15%)", inside=(2080, 690, 2760, 790))

    # 4. Bypass label above the horizontal connector
    place(a, text_mask(["Full-tile input (specified comparators)"], 52), center=(2095, 1496),
          fill=LAVENDER, name="Full-tile input (specified comparators)", inside=(1420, 1450, 2770, 1528))

    # 1. LLM box: redraw black rounded box, then verbatim excerpt in white
    err, r, x0, y0, x1, y1 = fit_box(a)
    LOG.append({"op": "fit_llm_box", "radius": r, "rect": [x0, y0, x1, y1], "edge_mismatch_px": err})
    er = (x0 - 6, y0 - 6, x1 + 6, y1 + 6)
    check_ring(a, er, PEACH, name="LLM box ring")
    a[er[1]:er[3] + 1, er[0]:er[2] + 1] = PEACH
    S = 4
    bw, bh = er[2] - er[0] + 1, er[3] - er[1] + 1
    m = Image.new("L", (bw * S, bh * S), 0)
    ImageDraw.Draw(m).rounded_rectangle(((x0 - er[0]) * S, (y0 - er[1]) * S, (x1 + 1 - er[0]) * S - 1,
                                         (y1 + 1 - er[1]) * S - 1), radius=r * S, fill=255)
    al = np.asarray(m.resize((bw, bh), Image.LANCZOS), float)[..., None] / 255.0
    reg = a[er[1]:er[3] + 1, er[0]:er[2] + 1].astype(float)
    a[er[1]:er[3] + 1, er[0]:er[2] + 1] = np.round(reg * (1 - al)).astype(np.uint8)
    LOG.append({"op": "redraw_llm_box", "erase_rect": er})
    size, pitch, margin = 44, 58, 52
    src_text, pref, lines = llm_excerpt(x1 - x0 - 2 * margin, size, 7)
    mask = text_mask(lines, size, pitch=pitch, align="left")
    mb = mask.getbbox()
    block_h = mb[3] - mb[1]
    ty = int(round((y0 + y1) / 2 - block_h / 2))
    place(a, mask, topleft=(x0 + margin, ty), color=WHITE, name="LLM excerpt",
          inside=(x0 + 40, y0 + 25, x1 - 40, y1 - 25))
    LOG.append({"op": "llm_excerpt", "verbatim_prefix": pref, "lines": lines,
                "is_prefix_of_main_line": src_text.startswith(pref)})

    out = Image.fromarray(a)
    assert out.size == (W, H) == (8610, 2197)
    return out, dpi, list(LOG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src_sha = sha256(SRC)
    rec = {"source": str(SRC.relative_to(ROOT)), "source_sha256": src_sha,
           "main_md_sha256": sha256(MAIN), "script_sha256": sha256(__file__), "font": FONT, "outputs": {}}
    for strict, name in ((False, "Figure_1_revised_pass042.png"),
                         (True, "Figure_1_revised_pass042_strictE1RU1.png")):
        img, dpi, log = build(strict)
        img.save(out / name, dpi=dpi, optimize=True)
        rec["outputs"][name] = {"sha256": sha256(out / name), "size": img.size, "dpi": dpi, "edits": log}
    assert sha256(SRC) == src_sha
    (out / "edit_log.json").write_text(json.dumps(rec, indent=1, default=str))
    sums = "".join(f"{v['sha256']}  {k}\n" for k, v in rec["outputs"].items())
    (out / "SHA256SUMS.txt").write_text(sums)
    print(json.dumps({k: v["sha256"] for k, v in rec["outputs"].items()}, indent=1))
    print(json.dumps(rec["outputs"]["Figure_1_revised_pass042.png"]["edits"][-1], indent=1))


if __name__ == "__main__":
    main()
