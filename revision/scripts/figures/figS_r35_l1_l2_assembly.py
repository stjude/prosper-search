"""Assemble Supplementary Fig. S-R3.5-L1 and S-R3.5-L2 from the accepted
localisation-study renders ([internal record] blocks R3.5-T5 and R3.5-T7).

Pure assembly: the source PNGs are placed unchanged (pixel copy) except that
the one-line run-provenance header printed on each L2 panel by the analysis
script is cropped off (rows above a measured white gap; see HEADER_CROP) and
panel letters / titles are added. No value is recomputed; input hashes are
checked against the stage COMPLETE.json manifests.

S-R3.5-L1: six representative-tile montages, `s3/run_002/montage_*.png`,
  A-C the three Fig. 5 pairs on the published partitions (RMS 16875 vs 15155,
  RMS 16875 vs 16867, RB 7333 vs 7330); D-F three annotated same-group pairs
  drawn with seed 123 (EWS 12151 vs 7443, NB 16039 vs 16676, RB 7292 vs 7347).
  LEGEND.txt is the caption basis.
S-R3.5-L2: A/B `l2/run_003/overlay_D2_4288.png`, `overlay_D2_4298.png`;
  C `matrix_D1_bin_same_group.png`.

Since 2026-11-04 (supplementary-figure audit) the default sources are the plotting-only
re-draws with publication labels in `<figure dir>/sources_20261104/`, written by
`figS_r35_l1_sources_rerender.py` and `figS_r35_l2_sources_rerender.py` (both verified to
reproduce the run PNGs byte for byte with the original text); their hashes are checked
against that folder's SHA256SUMS.txt. `--accepted-sources` assembles from the run PNGs
as before.

Decision D1 (2026-11-04): panel titles show the de-identified slide IDs
(_common.SCAN_TO_SLIDE_ID) instead of the numeric scan keys, and L2 assembles from
`sources_20261104b/` (sub-panel titles with slide IDs; `figS_r35_l2_sources_rerender.py`). L1's
montage sources carry no scan key on the face and stay in `sources_20261104/`. `--pre-d1`
reproduces the pre-D1 assembly (numeric titles, L2 from `sources_20261104/`); combined with
`--accepted-sources` the titles are numeric as well (the original accepted assembly).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import out_dir, prosper_out, sha256, slide_id, write_provenance  # noqa: E402

ROOT = prosper_out() / "r3_5_localisation_maps_20261017"
S3 = ROOT / "s3" / "run_002"
L2 = ROOT / "l2" / "run_003"
FONT = "/HOME/miniconda3/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
FONT_BOLD = FONT.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")

L1_PANELS = [
    ("A", "montage_fig5_A1_16875_vs_15155_p01.png", "Fig. 5A pair 1: RMS query 16875 vs reference 15155"),
    ("B", "montage_fig5_A2_16875_vs_16867_p01.png", "Fig. 5A pair 2: RMS query 16875 vs reference 16867"),
    ("C", "montage_fig5_B_7333_vs_7330_p01.png", "Fig. 5B pair: RB query 7333 vs reference 7330"),
    ("D", "montage_annot_12151_vs_7443_p01.png", "annotated same-group pair: EWS 12151 vs 7443"),
    ("E", "montage_annot_16039_vs_16676_p01.png", "annotated same-group pair: NB 16039 vs 16676"),
    ("F", "montage_annot_7292_vs_7347_p01.png", "annotated same-group pair: RB 7292 vs 7347"),
]
L2_PANELS = [
    ("A", "overlay_D2_4288.png", "discriminative map, reference 4288 (NB)", 40),
    ("B", "overlay_D2_4298.png", "discriminative map, reference 4298 (NB)", 40),
    ("C", "matrix_D1_bin_same_group.png", "class-to-class transported mass, same-group pairs (observed / independence / ceiling)", 45),
]

def _d1(title: str) -> str:
    """Replace every numeric scan key in a panel title by its slide ID (decision D1)."""
    import re
    return re.sub(r"\b\d{4,5}\b", lambda m: slide_id(m.group(0)), title)


L1_PANELS_D1 = [(L, n, _d1(t)) for L, n, t in L1_PANELS]
L2_PANELS_D1 = [(L, n, _d1(t), c) for L, n, t, c in L2_PANELS]
HEADER_CROP = {"overlay_D2_4288.png": 40, "overlay_D2_4298.png": 40, "matrix_D1_bin_same_group.png": 45}


def check_manifest(stage_dir: Path, names):
    if (stage_dir / "SHA256SUMS.txt").exists() and not (stage_dir / "COMPLETE.json").exists():
        man = {ln.split()[1]: ln.split()[0] for ln in (stage_dir / "SHA256SUMS.txt").read_text().splitlines() if ln.strip()}
    else:
        man = json.load(open(stage_dir / "COMPLETE.json"))["outputs"]
    for n in names:
        actual = sha256(stage_dir / n)
        assert man[n] == actual, (n, man[n], actual)


def labelled(img: Image.Image, letter: str, title: str, pad_top=46) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + pad_top), "white")
    out.paste(img, (0, pad_top))
    d = ImageDraw.Draw(out)
    d.text((8, 6), letter, fill="black", font=ImageFont.truetype(FONT_BOLD, 30))
    d.text((48, 14), title, fill="black", font=ImageFont.truetype(FONT, 18))
    return out


def grid(panels, ncol, gap=30):
    rows = [panels[i:i + ncol] for i in range(0, len(panels), ncol)]
    widths = [max(p.width for p in [r[c] for r in rows if c < len(r)]) for c in range(ncol)]
    heights = [max(p.height for p in r) for r in rows]
    W = sum(widths) + gap * (ncol + 1)
    H = sum(heights) + gap * (len(rows) + 1)
    canvas = Image.new("RGB", (W, H), "white")
    y = gap
    for r, hrow in zip(rows, heights):
        x = gap
        for c, p in enumerate(r):
            canvas.paste(p, (x, y))
            x += widths[c] + gap
        y += hrow + gap
    return canvas


def stack(panels, gap=30):
    W = max(p.width for p in panels) + 2 * gap
    H = sum(p.height for p in panels) + gap * (len(panels) + 1)
    canvas = Image.new("RGB", (W, H), "white")
    y = gap
    for p in panels:
        canvas.paste(p, (gap, y))
        y += p.height + gap
    return canvas


def finish(figure, stem, canvas, inputs, note):
    d = out_dir(figure)
    png = d / f"{stem}.png"
    canvas.save(png, dpi=(300, 300))
    canvas.convert("RGB").save(d / f"{stem}.pdf", resolution=300)
    write_provenance(d, [png, d / f"{stem}.pdf"], inputs, note, Path(__file__))
    print("wrote", png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted-sources", action="store_true",
                    help="assemble from the accepted run PNGs (pre-2026-11-04) instead of the sources_20261104 re-draws")
    ap.add_argument("--pre-d1", action="store_true",
                    help="numeric scan keys in the titles and L2 from sources_20261104/ (pre-D1 assembly)")
    a = ap.parse_args()
    numeric = a.pre_d1 or a.accepted_sources
    l1_panels = L1_PANELS if numeric else L1_PANELS_D1
    l2_panels = L2_PANELS if numeric else L2_PANELS_D1
    src1 = S3 if a.accepted_sources else out_dir("figS_r35_L1_representative_tiles") / "sources_20261104"
    src2 = L2 if a.accepted_sources else out_dir("figS_r35_L2_discriminative_map") / (
        "sources_20261104" if a.pre_d1 else "sources_20261104b")
    # ---- L1
    check_manifest(S3, ["LEGEND.txt", "representative_links.csv"] + ([n for _, n, _ in l1_panels] if a.accepted_sources else []))
    if not a.accepted_sources:
        check_manifest(src1, [n for _, n, _ in l1_panels])
    panels = [labelled(Image.open(src1 / n).convert("RGB"), L, t) for L, n, t in l1_panels]
    canvas = grid(panels, ncol=3)
    legend = (S3 / "LEGEND.txt").read_text().strip()
    finish("figS_r35_L1_representative_tiles", "FigS_R3.5-L1_representative_tiles", canvas,
           [src1 / n for _, n, _ in l1_panels] + [S3 / "LEGEND.txt", S3 / "representative_links.csv", S3 / "COMPLETE.json"],
           ("Pixel-copy assembly of the six s3/run_002 montages (hashes checked against COMPLETE.json)" if a.accepted_sources else
            "Pixel-copy assembly of the six sources_20261104 montage re-draws (readable sub-captions; hashes checked against "
            "that folder's SHA256SUMS.txt; re-draw port verified byte-identical to s3/run_002 with the original text)")
           + "; panel letters and pair titles added"
           + ("" if numeric else " (titles name the de-identified slide IDs, decision D1 2026-11-04; mapping "
                                 "_common.SCAN_TO_SLIDE_ID)")
           + ". Caption basis (LEGEND.txt): " + legend)
    # ---- L2
    check_manifest(src2, [n for _, n, _, _ in l2_panels])
    panels = []
    for L, n, t, crop in l2_panels:
        im = Image.open(src2 / n).convert("RGB")
        im = im.crop((0, crop, im.width, im.height))  # drop the one-line provenance header
        panels.append(labelled(im, L, t))
    canvas = stack(panels)
    finish("figS_r35_L2_discriminative_map", "FigS_R3.5-L2_discriminative_map", canvas,
           [src2 / n for _, n, _, _ in l2_panels] + [L2 / "COMPLETE.json", L2 / "report_estimates.json"],
           ("Pixel-copy assembly of the l2/run_003 panels (hashes checked against COMPLETE.json)" if a.accepted_sources else
            "Pixel-copy assembly of the sources_20261104 re-draws of the l2/run_003 panels (tumor / non-tumor / artifact / "
            "non-tumor (pooled); class legend limited to n > 0; colour-bar label wrapped; hashes checked against that "
            "folder's SHA256SUMS.txt; port verified byte-identical to run_003 with the original text)"
            + ("" if numeric else "; D1 2026-11-04: sources_20261104b, sub-panel and panel titles name the slide IDs "
                                  f"{slide_id(4288)} / {slide_id(4298)}"))
           + "; the printed run-provenance "
           f"header was cropped (rows above {HEADER_CROP}) and panel letters/titles added. Rule text for the legend: per "
           "reference tile, cost_i = transport-weighted mean cosine cost of its exact-OT links (full plan); Delta_i = mean of "
           "cost_i over the reference's 10 other-group P2 queries minus mean over its 10 same-group P2 queries (positive = "
           "matched better by same-diagnosis slides); patch value = mean of its owning mosaic tiles. Reference choice: the "
           "Fig. 5 reference slides are not P2 references, so the two P2 references of lowest numeric id are shown. "
           "Panel C: row-normalised class-to-class transport (full exact plan, links with both tiles annotated), mean over "
           "same-group pairs; ceiling = min(p,q)/p on the diagonal.")


if __name__ == "__main__":
    main()
