#!/usr/bin/env python3
"""Supplementary figure S-INST-F1 (provisional workspace ID; the analyst assigns the number at packaging):
the reader-study instrument, composed from the screenshots of its user manual (2026-11-06, revision).

Panels (reading order; face letters A-D in the house style of fig8_reader_study.py: 10 pt bold, top left):
  A  Step 1: whole-slide viewer of the query slide (03_step1_whole_slide.png, native size) with the
     completed Step 1 form beside it (05_step1_form.png, empty bottom band cropped, native size)
  B  Step 2: the whole-slide search results -- ranked result strip (thumbnail, recorded diagnosis,
     age/site/presentation, distance d) under the viewer, with the first-ranked slide open side by side
     (07_similar_slide_side_by_side.png, downscaled to 0.74)
  C  Step 2: region-of-interest search, ROI similarity heat map on the retrieved slide
     (11_roi_search_hit.png, downscaled to 0.74)
  D  Step 3: the LLM summary (13_step3_llm_summary.png, native size) with the Step 3 form beside it
     (14_step3_form.png, native size)
06_step2_search_unlocked.png was not used: its result strip is the same as in 07 (same seven visible
cards), and 07 adds the side-by-side comparison.

Sources (read-only; a separate repository, never modified here): prosper-reader-suite, docs/manual/img/,
screenshots of a practice ('demo') session of application version 1.0.2 (2026-10-05); the manual and the
suite are at version 1.0.6 (README.md). Every source's SHA-256 is asserted before anything is drawn.

The screenshots are not redrawn or edited. The only operations are the crop of the empty bottom band of
05 (rows CROP_05: are white apart from the sidebar's 1-px left border; asserted) and the Lanczos
downscale of B and C. A 2-px light-grey frame is drawn in the gap pixels around each screenshot, never on
it. After saving, the script asserts that every screenshot occupies its slot in the PNG pixel for pixel
(A and D equal the source files; B and C equal the in-script resample), that the panel letters touch only
the letter bands, and that the PDF carries the composed raster losslessly at full resolution.

Output  $PROSPER_OUT/figures_20261021/figS_instrument/
        Figure_S-INST-F1_instrument.png (2400 x 3000 px, 300 dpi) and .pdf (8 x 10 in; raster + vector letters)
        SHA256SUMS.txt (outputs, then the inputs by absolute path), PROVENANCE.md
Deterministic: re-running reproduces the PNG and PDF hashes.

Usage:  export PROSPER_OUT="$PWD/prosper_revision_output"
        /HOME/miniconda3/bin/python3 scripts/figures/figS_instrument_screenshots.py [--suite DIR]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, out_dir, sha256, style, write_provenance  # noqa: E402

FIGURE = "figS_instrument"
STEM = "Figure_S-INST-F1_instrument"
SUITE_DEFAULT = REPO.parent / "prosper-reader-suite"
IMG_REL = "docs/manual/img"
MANUAL_VERSION = "1.0.6"
SCREENSHOT_APP_VERSION = "1.0.2"
# SHA-256 of every source at prosper-reader-suite 1a1bc1c (v1.0.6, 2026-10-26)
SOURCES = {
    "03_step1_whole_slide.png": "8854c41747e2d3dc51ee5c5d5b4ebd38764418e3544b7818e1e2570dee856575",
    "05_step1_form.png": "b985825bcb3dd68c392606428f11fddaa4c09784cd6fd5bceed12216e223c303",
    "07_similar_slide_side_by_side.png": "a0bd7e0d08673fc61c8c1acd082afc4225525c34c51f3532ba5ca5b8fe61eed8",
    "11_roi_search_hit.png": "1f865c0422318cccc8065877fd174d5342455642a94aac54d6206ba8ede1ba58",
    "13_step3_llm_summary.png": "def23e3a03708b57037d65ed26733b8eeba6a93df26df70808a5ff731b512a03",
    "14_step3_form.png": "93068024059e932ee202e73a6b0f7c1051c1bddeb646b0ae5ea815f9b71d2343",
}
README_SHA = "3c7fe6a3d2647e98b6e3b314883c97786bd79d93d00dc88edcd071473d36f3d5"

DPI = 300
W, H = 2400, 3000          # 8 x 10 in at 300 dpi (both exact in binary, so the canvas maps 1:1 onto pixels)
M = 4                      # outer margin (room for the frame)
GAP = 24                   # between screenshots in a row
FRAME, FRAME_RGB = 2, (190, 190, 190)
BAND, ROWGAP = 58, 36      # letter band above each row; gap between rows
SCALE_BC = 0.74            # 1600 x 1000 -> 1184 x 740 (exact), two side by side fill the width
CROP_05 = 586              # rows kept of 05 (content rows 23-562; 23-px bottom padding kept, as at the top)
LETTER_DROP = 14           # letter baseline this many px above the top of its row


def load(suite: Path) -> dict[str, np.ndarray]:
    imgs = {}
    for name, want in SOURCES.items():
        p = suite / IMG_REL / name
        got = sha256(p)
        assert got == want, f"{p} changed: sha256 {got} != pinned {want}"
        im = Image.open(p)
        assert im.mode == "RGB", (name, im.mode)
        imgs[name] = np.asarray(im).copy()
    for name in ("03_step1_whole_slide.png", "07_similar_slide_side_by_side.png", "11_roi_search_hit.png",
                 "13_step3_llm_summary.png"):
        assert imgs[name].shape == (1000, 1600, 3), (name, imgs[name].shape)
    for name in ("05_step1_form.png", "14_step3_form.png"):
        assert imgs[name].shape == (922, 360, 3), (name, imgs[name].shape)
    return imgs


def crop_05(a: np.ndarray) -> np.ndarray:
    """Crop the empty bottom band; assert it is empty (white, apart from the sidebar's left border column)."""
    band = a[CROP_05:]
    assert (band[:, 1:] == 255).all(), "05: cropped band is not uniformly white"
    assert (band[:, 0] == band[0, 0]).all(), "05: left border column not uniform in the cropped band"
    nonwhite = np.where((a[:, 1:] != 255).any(axis=(1, 2)))[0]
    assert nonwhite.max() == 562 and nonwhite.min() == 23, (nonwhite.min(), nonwhite.max())
    return a[:CROP_05]


def downscale(a: np.ndarray) -> np.ndarray:
    w, h = round(a.shape[1] * SCALE_BC), round(a.shape[0] * SCALE_BC)
    assert (w, h) == (1184, 740)
    return np.asarray(Image.fromarray(a).resize((w, h), Image.LANCZOS))


def layout(imgs: dict[str, np.ndarray]):
    """Return the placed screenshots [(panel, name, array, x, y)], the letters [(letter, x, baseline_y)] and
    the row spans (y0, y1) that must be untouched by the letters."""
    a05 = crop_05(imgs["05_step1_form.png"])
    b07, c11 = downscale(imgs["07_similar_slide_side_by_side.png"]), downscale(imgs["11_roi_search_hit.png"])
    xs = M + 1600 + GAP  # form column in rows A and D
    yA = BAND
    yBC = yA + 1000 + ROWGAP + BAND
    yD = yBC + 740 + ROWGAP + BAND
    xC = M + 1184 + GAP
    placed = [
        ("A", "03_step1_whole_slide.png", imgs["03_step1_whole_slide.png"], M, yA),
        ("A", "05_step1_form.png", a05, xs, yA),
        ("B", "07_similar_slide_side_by_side.png", b07, M, yBC),
        ("C", "11_roi_search_hit.png", c11, xC, yBC),
        ("D", "13_step3_llm_summary.png", imgs["13_step3_llm_summary.png"], M, yD),
        ("D", "14_step3_form.png", imgs["14_step3_form.png"], xs, yD),
    ]
    assert xC + 1184 + M == W, "row B|C must fill the width"
    assert yD + 1000 + M <= H, (yD + 1000 + M, H)
    letters = [("A", M, yA - LETTER_DROP), ("B", M, yBC - LETTER_DROP), ("C", xC, yBC - LETTER_DROP),
               ("D", M, yD - LETTER_DROP)]
    rows = [(yA - FRAME, yA + 1000 + FRAME), (yBC - FRAME, yBC + 740 + FRAME), (yD - FRAME, yD + 1000 + FRAME)]
    return placed, letters, rows


def compose(placed) -> np.ndarray:
    canvas = np.full((H, W, 3), 255, np.uint8)
    for _, name, a, x, y in placed:
        h, w = a.shape[:2]
        assert x - FRAME >= 0 and y - FRAME >= 0 and x + w + FRAME <= W and y + h + FRAME <= H, name
        canvas[y - FRAME:y + h + FRAME, x - FRAME:x + w + FRAME] = FRAME_RGB
        canvas[y:y + h, x:x + w] = a
    # no two frames overlap (each screenshot's frame lies in white gap pixels only)
    boxes = [(x - FRAME, y - FRAME, x + a.shape[1] + FRAME, y + a.shape[0] + FRAME) for _, _, a, x, y in placed]
    for i, p in enumerate(boxes):
        for q in boxes[i + 1:]:
            assert p[2] <= q[0] or q[2] <= p[0] or p[3] <= q[1] or q[3] <= p[1], (p, q)
    return canvas


def render(canvas: np.ndarray, letters, png: Path, pdf: Path) -> None:
    style()
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(canvas, interpolation="none", extent=(0, W, H, 0), aspect="auto")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    for letter, x, y in letters:
        fig.text(x / W, 1 - y / H, letter, fontsize=10, fontweight="bold", va="baseline", ha="left")
    fig.savefig(png, dpi=DPI, facecolor="white")
    fig.savefig(pdf, dpi=DPI, facecolor="white", metadata={"CreationDate": None})
    plt.close(fig)


def verify(canvas: np.ndarray, placed, rows, png: Path, pdf: Path) -> list[str]:
    out = np.asarray(Image.open(png).convert("RGB"))
    assert out.shape == (H, W, 3), out.shape
    for _, name, a, x, y in placed:
        h, w = a.shape[:2]
        assert np.array_equal(out[y:y + h, x:x + w], a), f"PNG slot of {name} differs from its source/resample"
    for y0, y1 in rows:
        assert np.array_equal(out[y0:y1], canvas[y0:y1]), f"letters or rendering touched rows {y0}-{y1}"
    doc = pymupdf.open(pdf)
    assert doc.page_count == 1
    page = doc[0]
    assert abs(page.rect.width - W / DPI * 72) < 0.01 and abs(page.rect.height - H / DPI * 72) < 0.01, page.rect
    xrefs = page.get_images(full=True)
    assert len(xrefs) == 1, xrefs
    pix = pymupdf.Pixmap(doc, xrefs[0][0])
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3]
    (info,) = page.get_image_info(xrefs=True)
    t = info["transform"]  # matplotlib stores the rows bottom-up and flips them with the placement matrix
    stored = arr[::-1] if t[3] < 0 else arr
    assert (pix.w, pix.h) == (W, H) and np.array_equal(stored, canvas), "PDF raster is not the composed canvas"
    assert tuple(round(v, 3) for v in info["bbox"]) == (0, 0, round(W / DPI * 72, 3), round(H / DPI * 72, 3))
    r = page.get_pixmap(dpi=DPI)
    shown = np.frombuffer(r.samples, np.uint8).reshape(r.h, r.w, r.n)[..., :3]
    for y0, y1 in rows:
        assert np.array_equal(shown[y0:y1], canvas[y0:y1]), f"PDF page rendered at {DPI} dpi differs in rows {y0}-{y1}"
    text = page.get_text().split()
    assert text == ["A", "B", "C", "D"], text
    fonts = sorted({f[3] for f in page.get_fonts()})
    return [f"PNG {W} x {H}: all six screenshot slots pixel-identical to their sources (A, D) or the in-script "
            f"Lanczos resample (B, C); the three screenshot rows (frames included) identical to the composed canvas, "
            f"so the letters touch only the letter bands",
            f"PDF one page {W / DPI:.0f} x {H / DPI:.0f} in, one embedded raster {pix.w} x {pix.h} identical to the "
            f"composed canvas (lossless, Flate; stored bottom-up and flipped by its placement matrix) covering the page; "
            f"the page rasterised at {DPI} dpi by MuPDF reproduces the three screenshot rows exactly; "
            f"page text = A B C D (vector, font {', '.join(fonts)})"]


def suite_state(suite: Path) -> tuple[str, str]:
    head = subprocess.check_output(["git", "-C", str(suite), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(suite), "status", "--porcelain", "--", IMG_REL, "README.md"],
                                    text=True).strip()
    return head, dirty or "clean"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", type=Path, default=SUITE_DEFAULT, help="prosper-reader-suite checkout (read-only)")
    args = ap.parse_args()
    suite = args.suite.resolve()
    readme = suite / "README.md"
    assert sha256(readme) == README_SHA, f"{readme} changed"
    ver = re.search(r"\*\*Version (\d+\.\d+\.\d+)\*\*", readme.read_text(encoding="utf-8")).group(1)
    assert ver == MANUAL_VERSION, ver
    imgs = load(suite)
    placed, letters, rows = layout(imgs)
    canvas = compose(placed)
    d = out_dir(FIGURE)
    png, pdf = d / f"{STEM}.png", d / f"{STEM}.pdf"
    render(canvas, letters, png, pdf)
    checks = verify(canvas, placed, rows, png, pdf)
    head, dirty = suite_state(suite)
    inputs = [suite / IMG_REL / n for n in SOURCES] + [readme]
    slots = "\n".join(f"- {panel}: `{name}` {a.shape[1]} x {a.shape[0]} px at ({x}, {y})"
                      for panel, name, a, x, y in placed)
    note = f"""Supplementary figure S-INST-F1 (provisional ID; number assigned by the lead at packaging): the
reader-study instrument. Sources: prosper-reader-suite commit `{head}` (working tree for `{IMG_REL}` and
`README.md`: {dirty}); suite and manual version **{ver}** (`README.md`, "**Version {ver}**"); the screenshots
themselves were taken in a practice (`demo`) session of application version {SCREENSHOT_APP_VERSION}
(2026-10-05; stated in `docs/user_manual.md`).

Canvas {W} x {H} px, {DPI} dpi ({W / DPI:.0f} x {H / DPI:.0f} in). Rows: A = 03 + 05 at native size;
B | C = 07 and 11 downscaled to {SCALE_BC} (Lanczos, 1184 x 740); D = 13 + 14 at native size. 05 cropped to
rows 0-{CROP_05 - 1} (the removed band, rows {CROP_05}-921, is white apart from the sidebar's 1-px left border;
asserted). {FRAME}-px frame RGB{FRAME_RGB} in the gap pixels around each screenshot. Letters A-D: 10 pt bold
(house style), baseline {LETTER_DROP} px above each row. Slots (x, y of the top-left pixel):

{slots}

Checks (asserted in the script):

- {checks[0]}
- {checks[1]}

Compose script: `scripts/figures/figS_instrument_screenshots.py`.
"""
    write_provenance(d, [png, pdf], inputs, note, Path(__file__), title=f"{FIGURE} / {STEM}")
    with open(d / "SHA256SUMS.txt", "a") as fh:
        for p in inputs:
            fh.write(f"{sha256(p)}  {p}\n")
    print(f"wrote {png}\n{sha256(png)}  {png.name}\n{sha256(pdf)}  {pdf.name}")
    for c in checks:
        print("check:", c)


if __name__ == "__main__":
    main()
