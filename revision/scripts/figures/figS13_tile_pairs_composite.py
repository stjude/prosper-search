#!/usr/bin/env python3
"""Supplementary Fig. S13 (revision numbering; workspace id S-R3.5-T2): one composite from the three
accepted tile-pair montages of `scripts/r3_5_fig5_ot_tile_montage.py` (v2, 2026-10-08).

Assembly only: the three PNGs are placed unchanged (no resampling; same pixel width 3,124) one above the
other with a panel letter (A1, A2, B) in the top-left margin of each, matching the legend in
docs/prosper_revised/prosper_supplementary_legends.md. Nothing is recomputed.

Inputs   $PROSPER_OUT/r3_5_fig5_ot_tile_montage_20261008_v2/montage_{A1,A2,B}_*.png
Outputs  $PROSPER_OUT/figures_20261021/figS13_tile_pairs/Figure_S13_tile_pairs.png (+ PDF wrapper),
         SHA256SUMS.txt, PROVENANCE.md

Run from the repository root with PROSPER_OUT exported:
    /HOME/miniconda3/bin/python3 scripts/figures/figS13_tile_pairs_composite.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import out_dir, prosper_out, write_provenance  # noqa: E402

SRC = prosper_out() / "r3_5_fig5_ot_tile_montage_20261008_v2"
PANELS = [("A1", "montage_A1_16875_vs_15155.png"),
          ("A2", "montage_A2_16875_vs_16867.png"),
          ("B", "montage_B_7333_vs_7330.png")]
GAP = 70          # px between panels
LEFT = 150        # px margin for the panel letter
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def main() -> int:
    ims = [(lab, Image.open(SRC / f).convert("RGB")) for lab, f in PANELS]
    dpi = ims[0][1].info.get("dpi", (220, 220))
    w = max(im.width for _, im in ims)
    assert all(im.width == w for _, im in ims), [im.width for _, im in ims]
    h = sum(im.height for _, im in ims) + GAP * (len(ims) - 1)
    canvas = Image.new("RGB", (LEFT + w, h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 64)
    y = 0
    for lab, im in ims:
        canvas.paste(im, (LEFT, y))
        draw.text((20, y + 8), lab, fill="black", font=font)
        y += im.height + GAP
    d = out_dir("figS13_tile_pairs")
    png = d / "Figure_S13_tile_pairs.png"
    pdf = d / "Figure_S13_tile_pairs.pdf"
    canvas.save(png, dpi=dpi)
    canvas.save(pdf, "PDF", resolution=float(dpi[0]))
    note = ("Supplementary Fig. S13 composite assembled 2026-11-10 from the three accepted v2 montages "
            "(pixels unchanged; panel letters A1, A2, B added in a left margin). Source montages, labels and "
            "sampled_edges.csv remain in r3_5_fig5_ot_tile_montage_20261008_v2/. The in-image header text keeps "
            "the montage script's wording (internal slide indices, 'neighbour'); the legend carries the journal "
            f"wording. Canvas {canvas.width} x {canvas.height} px at {dpi[0]:.0f} dpi.")
    write_provenance(d, [png, pdf], [SRC / f for _, f in PANELS], note, Path(__file__),
                     title="figS13_tile_pairs / Figure_S13_tile_pairs")
    print("wrote", png, canvas.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
