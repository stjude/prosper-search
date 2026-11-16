"""Re-draw the six R3.5-L1 representative-tile montages used by Supplementary Fig.
S-R3.5-L1 with readable sub-captions (2026-11-04 supplementary-figure audit). Plotting only.

Source (read-only): `$PROSPER_OUT/r3_5_localisation_maps_20261017/s3/run_002/representative_links.csv`
(per link: pair, query, reference, kind, cluster, mass, cosine_cost, query/ref tile xy and class code), with
the tiles read from the read-only WSIs. The drawing is a port of the montage block of
`scripts/r3_5_l1_maps.py::stage_s3` (PIL, 112-px tiles, page 1 of <= 20 clusters); nothing is
re-clustered or re-transported.

Sub-caption changes (TEXT_REVISED vs TEXT_ORIGINAL):
  "c=0.402 m=3.9e-04"  ->  "cost 0.402, mass 3.9e-4"
  "T | NT", "- | -"    ->  "tumor | non-tumor", "unannotated | unannotated" (I immune, N necrosis, A artifact)
Scan IDs in the assembly titles are unchanged (author policy pending).

  --check-port DIR   re-draw with the original text into DIR; compare with run_002 PNGs (pixel-identical required)
  (default)          write to $PROSPER_OUT/figures_20261021/figS_r35_L1_representative_tiles/sources_20261104/
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import openslide
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import out_dir, prosper_out, sha256, write_provenance  # noqa: E402

S3 = prosper_out() / "r3_5_localisation_maps_20261017" / "s3" / "run_002"
WSI = "/PROSPER_WORKSPACE/data/comet_final_HPC/{}.svs"
TS = 112
KINDS = ("medoid", "highest_mass", "lowest_mass")
CLASS_TEXT = {"-": "unannotated", "T": "tumor", "I": "immune", "N": "necrosis", "NT": "non-tumor", "A": "artifact"}


def cap_original(l):
    return f"c={l.cosine_cost:.3f} m={l.mass:.1e}", f"{l.query_class} | {l.ref_class}"


def cap_revised(l):
    m = re.sub(r"e([+-])0*(\d)", lambda g: "e" + ("-" if g.group(1) == "-" else "") + g.group(2), f"{l.mass:.1e}")
    return (f"cost {l.cosine_cost:.3f}, mass {m}",
            f"{CLASS_TEXT.get(l.query_class, l.query_class)} | {CLASS_TEXT.get(l.ref_class, l.ref_class)}")


def render(d: Path, caption) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    links = pd.read_csv(S3 / "representative_links.csv", dtype={"query": str, "reference": str, "query_class": str,
                                                                  "ref_class": str})
    links = links[links.kind.isin(KINDS)]
    written = []
    for (tag, q, n), L in links.groupby(["pair", "query", "reference"], sort=False):
        wq, wr = openslide.OpenSlide(WSI.format(q)), openslide.OpenSlide(WSI.format(n))
        cs = sorted(set(int(c) for c in L.cluster))[:20]
        W = 80 + 3 * (2 * TS + 20)
        Hh = 30 + len(cs) * (TS + 34)
        img = Image.new("RGB", (W, Hh), "white")
        dr = ImageDraw.Draw(img)
        for x0, title in ((80, "medoid"), (80 + 2 * TS + 20, "highest mass"), (80 + 2 * (2 * TS + 20), "lowest mass")):
            dr.text((x0, 8), f"{title} (query | reference)", fill="black")
        for ri, c in enumerate(cs):
            y = 30 + ri * (TS + 34)
            dr.text((4, y + TS // 2), f"c{c}", fill="black")
            for ci, kind in enumerate(KINDS):
                l = L[(L.cluster == c) & (L.kind == kind)].iloc[0]
                x0 = 80 + ci * (2 * TS + 20)
                for si, (ws, xy) in enumerate(((wq, l.query_xy), (wr, l.ref_xy))):
                    x, yy = (float(v) for v in xy.split(","))
                    tile = ws.read_region((int(x), int(yy)), 0, (512, 512)).convert("RGB").resize((TS, TS))
                    img.paste(tile, (x0 + si * TS, y))
                t1, t2 = caption(l)
                dr.text((x0, y + TS + 2), t1, fill="black")
                dr.text((x0, y + TS + 16), t2, fill="black")
        p = d / f"montage_{tag}_{q}_vs_{n}_p01.png"
        img.save(p)
        written.append(p)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-port", metavar="DIR")
    a = ap.parse_args()
    if a.check_port:
        from PIL import ImageChops
        ok = True
        for p in render(Path(a.check_port), cap_original):
            x = Image.open(S3 / p.name).convert("RGB"); y = Image.open(p).convert("RGB")
            same = x.size == y.size and ImageChops.difference(x, y).getbbox() is None
            print(p.name, "pixel-identical" if same else f"DIFFERENT {x.size} {y.size}",
                  "| byte-identical" if sha256(p) == sha256(S3 / p.name) else "| bytes differ")
            ok &= same
        sys.exit(0 if ok else 1)
    d = out_dir("figS_r35_L1_representative_tiles") / "sources_20261104"
    written = render(d, cap_revised)
    write_provenance(d, written, [S3 / "representative_links.csv", S3 / "COMPLETE.json"] + [S3 / p.name for p in written],
                     "Plotting-only re-draw of the six s3/run_002 montages from representative_links.csv (tiles read from the "
                     "read-only WSIs) with readable sub-captions ('cost 0.402, mass 3.9e-4'; 'tumor | non-tumor'; "
                     "'unannotated'). Port of the r3_5_l1_maps.stage_s3 montage block; the original-text re-draw "
                     "(--check-port) is compared pixel-wise with the run_002 PNGs.", Path(__file__),
                     title="figS_r35_L1 sources (2026-11-04)")
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
