#!/usr/bin/env python3
"""revision stage 2a: swap the "RMS" and "MRT" x-axis labels of the Fig. 1
"Bayesian tumor-group probabilities" mini bar chart (request 2026-11-15),
so that the tallest bar is labelled RMS and the bar to its left MRT,
consistent with the quoted SJRHB073134_D1 summary. Nothing else changes.

The two labels are the submitted artwork's own glyphs (same width, both
bottom-aligned at the same baseline), so the swap exchanges two equal pixel
rectangles; each rectangle is first verified to contain only the panel fill
on its border ring. Input: Figure_1_revised_pass042.png (kept unchanged).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

PEACH = (254, 220, 196)
# glyph columns: RMS 7936-7966 (under bar 7931-7971), MRT 8013-8043 (under bar 8002-8050);
# glyph rows 1379/1383-1467; the axis line ends at row 1367.
RMS_RECT = (7930, 1370, 7972, 1474)  # x0, y0, x1, y1 inclusive
MRT_RECT = (8007, 1370, 8049, 1474)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ring_ok(a, r):
    x0, y0, x1, y1 = r
    ring = np.concatenate([a[y0, x0:x1 + 1], a[y1, x0:x1 + 1], a[y0:y1 + 1, x0], a[y0:y1 + 1, x1]])
    return bool((np.abs(ring.astype(int) - PEACH).max(1) <= 3).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    src_sha = sha256(args.src)
    img = Image.open(args.src)
    dpi = img.info.get("dpi", (300, 300))
    a = np.array(img.convert("RGB"))
    for r in (RMS_RECT, MRT_RECT):
        assert ring_ok(a, r), r
    w = RMS_RECT[2] - RMS_RECT[0]
    assert w == MRT_RECT[2] - MRT_RECT[0] and RMS_RECT[1::2] == MRT_RECT[1::2]
    y0, y1 = RMS_RECT[1], RMS_RECT[3] + 1
    rms = a[y0:y1, RMS_RECT[0]:RMS_RECT[2] + 1].copy()
    mrt = a[y0:y1, MRT_RECT[0]:MRT_RECT[2] + 1].copy()
    a[y0:y1, RMS_RECT[0]:RMS_RECT[2] + 1] = mrt
    a[y0:y1, MRT_RECT[0]:MRT_RECT[2] + 1] = rms
    out = Image.fromarray(a)
    assert out.size == img.size
    out.save(args.out, dpi=dpi, optimize=True)
    assert sha256(args.src) == src_sha
    before = np.array(img.convert("RGB"))
    diff = np.argwhere((before != a).any(2))
    rec = {"op": "swap RMS/MRT x-axis labels of the Bayesian mini bar chart",
           "src": args.src.name, "src_sha256": src_sha, "out": args.out.name, "out_sha256": sha256(args.out),
           "script_sha256": sha256(__file__), "rects": {"RMS_glyphs_from": RMS_RECT, "MRT_glyphs_from": MRT_RECT},
           "changed_pixels": int(len(diff)),
           "changed_bbox": [int(diff[:, 1].min()), int(diff[:, 0].min()), int(diff[:, 1].max()), int(diff[:, 0].max())]}
    args.out.with_name(args.out.stem + "_edit_log.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
