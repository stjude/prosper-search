"""ST1.5c: pyramidal viewing copies of the CCDI query slides for QuPath.

The packet's 22 query slides (BTIFF_ccdi_mci/<case>.tiff) are single-level
JPEG-tiled BigTIFFs: any zoomed-out render must decode the entire level-0
plane (a 1024-px overview of one case did not finish in 110 s), so the files
are unusable for the pathologist's open-and-look workflow. This script writes
pyramidal copies following the folder's existing precedent
(BTIFF_ccdi_mci_svs/PBBYHH.svs: BigTIFF, JPEG tiles, halving pyramid, named
.svs, opened by OpenSlide as generic-tiff) and restores the true pixel size
from the CCDI DICOM originals — the BTIFF resolution tags are a 10 px/cm
placeholder, so without this QuPath's scale bar is off by orders of magnitude.

Per case:
  1. mpp: scan the case's DICOM headers, take the largest instance carrying
     a PixelSpacing (mm, SharedFunctionalGroups PixelMeasures) and accept it
     if the BTIFF dimensions equal it up to conversion padding (the BTIFFs
     are the DICOM base level padded right/bottom by up to a few hundred
     pixels, e.g. PBCTPH 51456x48640 vs DICOM 51337x48488; padding does not
     change the spacing). No acceptable match -> no resolution is written
     (the copy is still viewable, just uncalibrated) and the log records why.
  2. vips sequential pass -> tiffsave: 512-px JPEG tiles, halving pyramid
     down to one tile, BigTIFF; written to <case>.svs.partial then renamed.
  3. QC with OpenSlide: opens, level-0 dims equal the source, pyramid depth
     as expected, mpp round-trips when embedded, and a 512-px centre crop
     stays within JPEG-re-encode distance of the source.

Outputs <out-dir>/<case>.svs and appends one JSON line per case to
<out-dir>/conversion_log.jsonl. Existing outputs are skipped unless
--overwrite.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import openslide
import pydicom
import pyvips


PAD_TOLERANCE = 1024  # px: BTIFF = DICOM base level + right/bottom padding


def mpp_from_dicom(case_dir: Path, width: int, height: int):
    """(mpp_x_um, mpp_y_um, detail) or (None, None, reason)."""
    best = None  # (cols, rows, row_mm, col_mm, filename)
    for f in sorted(case_dir.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        cols = getattr(ds, "TotalPixelMatrixColumns", None)
        rows = getattr(ds, "TotalPixelMatrixRows", None)
        try:
            ps = (ds.SharedFunctionalGroupsSequence[0]
                  .PixelMeasuresSequence[0].PixelSpacing)
        except Exception:
            continue
        if cols and rows and (best is None or cols > best[0]):
            best = (cols, rows, float(ps[0]), float(ps[1]), f.name)
    if best is None:
        return None, None, "no DICOM instance with dimensions and PixelSpacing"
    cols, rows, row_mm, col_mm, name = best
    if not (0 <= width - cols <= PAD_TOLERANCE and 0 <= height - rows <= PAD_TOLERANCE):
        return None, None, (f"largest DICOM instance {cols}x{rows} not within "
                            f"padding tolerance of BTIFF {width}x{height}")
    detail = f"{name}: {cols}x{rows}, pad +{width - cols}/+{height - rows}"
    return col_mm * 1000.0, row_mm * 1000.0, detail


def convert(case: str, src: Path, dst: Path, dicom_root: Path, quality: int):
    rec = {"case": case, "src": str(src), "dst": str(dst)}
    t0 = time.time()
    img = pyvips.Image.new_from_file(str(src), access="sequential")
    rec["dims"] = [img.width, img.height]

    mpp_x, mpp_y, detail = mpp_from_dicom(dicom_root / case, img.width, img.height)
    save_kw = dict(tile=True, tile_width=512, tile_height=512,
                   compression="jpeg", Q=quality, pyramid=True, bigtiff=True)
    if mpp_x:
        # vips resolution is px/mm; resunit converts on write
        save_kw.update(xres=1000.0 / mpp_x, yres=1000.0 / mpp_y, resunit="cm")
        rec["mpp"] = [round(mpp_x, 6), round(mpp_y, 6)]
        rec["mpp_source"] = detail
    else:
        rec["mpp"] = None
        rec["mpp_skipped_because"] = detail

    part = dst.with_suffix(".svs.partial")
    img.tiffsave(str(part), **save_kw)
    part.rename(dst)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["bytes"] = dst.stat().st_size

    # QC
    s_src = openslide.OpenSlide(str(src))
    s_out = openslide.OpenSlide(str(dst))
    assert s_out.dimensions == s_src.dimensions, \
        f"{case}: dims {s_out.dimensions} != source {s_src.dimensions}"
    expect_levels = 1 + max(0, int(np.ceil(np.log2(max(img.width, img.height) / 512))))
    assert s_out.level_count >= expect_levels - 1, \
        f"{case}: only {s_out.level_count} levels (expected ~{expect_levels})"
    rec["levels"] = s_out.level_count
    if mpp_x:
        # openslide 3.4 does not synthesise mpp for generic-tiff, so check the
        # written tags (QuPath additionally gets the calibration from the
        # project entry metadata, which the project builder fills from the log)
        xres = float(s_out.properties["tiff.XResolution"])  # px/cm
        assert abs(10000.0 / xres - mpp_x) / mpp_x < 0.01, \
            f"{case}: mpp round-trip {10000.0 / xres} != {mpp_x}"
    cx, cy = img.width // 2, img.height // 2
    a = np.asarray(s_src.read_region((cx, cy), 0, (512, 512)).convert("RGB"), float)
    b = np.asarray(s_out.read_region((cx, cy), 0, (512, 512)).convert("RGB"), float)
    rec["qc_center_mad"] = round(float(np.abs(a - b).mean()), 2)
    assert rec["qc_center_mad"] < 8.0, f"{case}: centre crop MAD {rec['qc_center_mad']}"
    s_src.close(); s_out.close()
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases-csv", required=True,
                   help="packet 01_cases.csv (case_id column, reading order)")
    p.add_argument("--btiff-dir", required=True)
    p.add_argument("--dicom-root", required=True,
                   help="ccdi_mci root: <case>/<study>/SM_<series>/*.dcm")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--quality", type=int, default=90)
    p.add_argument("--only", action="append", default=None,
                   help="restrict to these case ids (repeatable; canary)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)

    import pandas as pd
    cases = pd.read_csv(a.cases_csv)["case_id"].astype(str).tolist()
    if a.only:
        cases = [c for c in cases if c in set(a.only)]
    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    log = out / "conversion_log.jsonl"

    for case in cases:
        dst = out / f"{case}.svs"
        if dst.exists() and not a.overwrite:
            print(json.dumps({"case": case, "skipped": "exists"}), flush=True)
            continue
        rec = convert(case, Path(a.btiff_dir) / f"{case}.tiff", dst,
                      Path(a.dicom_root), a.quality)
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
