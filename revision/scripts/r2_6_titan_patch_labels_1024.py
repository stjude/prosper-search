#!/usr/bin/env python3
"""R2.6 direct-1024 route, stage 1: majority-pixel pathologist labels on the
TITAN/CONCH 1024-px level-0 patch grid for the 202 accepted R2.1 annotation slides.

Rule (same as comet-annotation create_tumor_nontumor_labels.py, applied to 1024
windows): the QuPath label raster (exported at downsample 4; 0 = unannotated/
background, 1 Tumor, 2 Immune cells, 3 Necrosis, 4 nonTumor, 5 Artifacts) is
nearest-neighbour resized to level-0 slide dimensions (cv2.INTER_NEAREST) and
each window [x, x+P) x [y, y+P), clipped to the slide, gets the most frequent
value; ties -> lowest label (np.unique + argmax in the legacy code).

The full-resolution resized mask is never materialised: the cv2 nearest-
neighbour source index map (src = min(floor(dst * (1 / (dst_size / src_size))),
src_size - 1); checked against cv2.resize in the test) is applied per axis, and
each window's histogram is the multiplicity-weighted histogram of the
corresponding raster block. This is exactly equal (not approximately) to
cv2.resize(INTER_NEAREST) + window majority under that index map.

Validation: the same function with P = 512 at every legacy tile coordinate
(comet-annotation/output/pathologist_annotated_labels/<slide>.csv) of all 202 slides
is compared with the legacy patch_label.

Run from the repository root with PROSPER_OUT exported. Does not import `prosper`.
Writes only under $PROSPER_OUT/r2_6_titan_direct1024_20261004/labels_1024/.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CORE = "/PROSPER_CORE"
HAORAN = CORE + "/Haoran"
RASTER_DIR = CORE + "/COMET/QuPathAnnotation_Tumor_NonTumor_processed_final/export"
QUPATH_EXPORT_SCRIPT = CORE + "/COMET/QuPathAnnotation_Tumor_NonTumor_processed_final/scripts/exportQupathMask.groovy"
TRIDENT_DIR = HAORAN + "/comet-feature-gen/comet-titan/trident/trident_processed/40x_1024px_0px_overlap"
WSI_DIR = HAORAN + "/data/comet_final_HPC"  # trident _config_coords.json wsi_source
METADATA = HAORAN + "/comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv"
LEGACY_DIR = HAORAN + "/comet-annotation/output/pathologist_annotated_labels"
LEGACY_SCRIPT = HAORAN + "/comet-annotation/src/create_tumor_nontumor_labels/create_tumor_nontumor_labels.py"
LEGACY_TRAINER = HAORAN + "/comet-annotation/src/train_and_evaluate_classifier/train_classifier_SGD_multiprocessing.py"
R21_FILTER_DIR = "r2_1_cpu_20261002/filter_attempt_001"
R21_STAGE1A_DIR = "r2_1_annotation_audit_20261001/stage1a_attempt_001"
OUT_NAME = "r2_6_titan_direct1024_20261004"
SUB = "labels_1024"
CLASS_NAMES = {0: "unannotated", 1: "Tumor", 2: "Immune cells", 3: "Necrosis",
               4: "nonTumor", 5: "Artifacts"}
PATCH = 1024
LEGACY_PATCH = 512
LABEL_COLUMNS = ["slide_id", "patient_id", "diagnostic_group", "held_out_fold", "x", "y",
                 "label_1024", "n_raster_pixels", "frac_label_pixels", "n512_available",
                 "n512_tumor", "feature_row"]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- core geometry
def nn_source_index(dst_size: int, src_size: int) -> np.ndarray:
    """cv2.resize INTER_NEAREST source index for every destination pixel (one axis).

    OpenCV resizeNN: inv_scale = dst/src; ifx = 1/inv_scale;
    sx = min(cvFloor(x * ifx), src - 1).
    """
    if dst_size <= 0 or src_size <= 0:
        raise ValueError("sizes must be positive")
    ifx = 1.0 / (float(dst_size) / float(src_size))
    idx = np.floor(np.arange(dst_size, dtype=np.float64) * ifx).astype(np.int64)
    return np.minimum(idx, src_size - 1)


def _runs(index_slice: np.ndarray):
    """(first, last, counts) for a nondecreasing contiguous index slice."""
    first, last = int(index_slice[0]), int(index_slice[-1])
    counts = np.bincount(index_slice - first, minlength=last - first + 1)
    return first, last, counts


def window_label(raster: np.ndarray, sx: np.ndarray, sy: np.ndarray, x: int, y: int,
                 patch: int, n_classes: int = 6):
    """Majority label of level-0 window [x,x+patch) x [y,y+patch) clipped to the slide.

    sx/sy are full-length nn_source_index arrays (len = level-0 width/height).
    Returns (label, n_raster_pixels, frac_label_pixels, n_level0_pixels, clipped).
    label is -1 for an empty (fully out-of-slide) window.
    """
    W, H = sx.shape[0], sy.shape[0]
    x1, y1 = min(x + patch, W), min(y + patch, H)
    clipped = (x1 - x) < patch or (y1 - y) < patch
    if x1 <= x or y1 <= y or x < 0 or y < 0:
        return -1, 0, float("nan"), 0, True
    c0, c1, wc = _runs(sx[x:x1])
    r0, r1, wr = _runs(sy[y:y1])
    block = raster[r0:r1 + 1, c0:c1 + 1]
    weights = np.multiply.outer(wr, wc).ravel().astype(np.float64)
    hist = np.bincount(block.ravel(), weights=weights, minlength=n_classes)
    label = int(np.argmax(hist))  # first maximum -> lowest label on ties
    total = (x1 - x) * (y1 - y)
    return label, int(block.size), float(hist[label] / total), int(total), bool(clipped)


def label_windows(raster: np.ndarray, level0_w: int, level0_h: int, coords: np.ndarray,
                  patch: int):
    sx = nn_source_index(level0_w, raster.shape[1])
    sy = nn_source_index(level0_h, raster.shape[0])
    n = coords.shape[0]
    lab = np.empty(n, np.int64)
    npx = np.empty(n, np.int64)
    frac = np.empty(n, np.float64)
    nl0 = np.empty(n, np.int64)
    clip = np.empty(n, bool)
    for i in range(n):
        lab[i], npx[i], frac[i], nl0[i], clip[i] = window_label(
            raster, sx, sy, int(coords[i, 0]), int(coords[i, 1]), patch)
    return lab, npx, frac, nl0, clip


def child_counts(coords: np.ndarray, legacy: dict, patch: int = PATCH):
    """For each 1024 patch, (n available 512 legacy tiles, n labelled Tumor)."""
    half = patch // 2
    n_av = np.zeros(coords.shape[0], np.int64)
    n_tu = np.zeros(coords.shape[0], np.int64)
    for i, (x, y) in enumerate(coords.tolist()):
        for dx in (0, half):
            for dy in (0, half):
                v = legacy.get((x + dx, y + dy))
                if v is not None:
                    n_av[i] += 1
                    n_tu[i] += int(v == 1)
    return n_av, n_tu


# ---------------------------------------------------------------- per-slide worker
def process_slide(task):
    import h5py
    import pandas as pd
    import tifffile

    slide, raster_dir, trident_dir, legacy_dir, r21_tiles = task
    t0 = time.time()
    rec = {"slide_id": slide}
    rpath = os.path.join(raster_dir, f"{slide}.tif")
    # Memory: the (uncompressed, single-strip ImageJ) raster is memory-mapped and
    # hashed by streaming; it is never decoded into a private copy or resized.
    rec["raster_sha256"] = sha256_file(rpath)
    rec["raster_bytes"] = os.path.getsize(rpath)
    raster = tifffile.memmap(rpath, mode="r")
    if raster.ndim != 2 or raster.dtype != np.uint8:
        raise ValueError(f"{slide}: raster shape/dtype {raster.shape} {raster.dtype}")
    pix = np.zeros(256, np.int64)
    for r0 in range(0, raster.shape[0], 1024):
        pix += np.bincount(np.asarray(raster[r0:r0 + 1024]).ravel(), minlength=256)
    vals = np.flatnonzero(pix)
    rec["raster_values"] = vals.tolist()
    rec["raster_pixel_counts"] = {str(int(v)): int(pix[v]) for v in vals}
    if vals.max() > 5:
        raise ValueError(f"{slide}: raster values outside 0..5: {vals.tolist()}")
    rec["raster_shape_hw"] = list(raster.shape)

    ppath = os.path.join(trident_dir, "patches", f"{slide}_patches.h5")
    fpath = os.path.join(trident_dir, "features_conch_v15", f"{slide}.h5")
    praw, fraw = Path(ppath).read_bytes(), Path(fpath).read_bytes()
    rec["patches_h5_sha256"] = hashlib.sha256(praw).hexdigest()
    rec["conch_h5_sha256"] = hashlib.sha256(fraw).hexdigest()
    with h5py.File(io.BytesIO(praw), "r") as h:
        coords = h["coords"][:]
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in h["coords"].attrs.items()}
    with h5py.File(io.BytesIO(fraw), "r") as h:
        fcoords = h["coords"][:]
        fshape = list(h["features"].shape)
    del praw, fraw
    if not np.array_equal(coords, fcoords):
        raise ValueError(f"{slide}: patch/feature coords differ")
    if fshape != [coords.shape[0], 768]:
        raise ValueError(f"{slide}: feature shape {fshape}")
    if int(attrs["patch_size"]) != PATCH or int(attrs["patch_size_level0"]) != PATCH \
            or int(attrs["overlap"]) != 0:
        raise ValueError(f"{slide}: unexpected patch attrs {attrs}")
    W, H = int(attrs["level0_width"]), int(attrs["level0_height"])
    # legacy code resized to openslide `slide.dimensions`; confirm they equal the trident attrs
    try:
        import openslide
        with openslide.OpenSlide(os.path.join(WSI_DIR, f"{slide}.svs")) as osl:
            rec["openslide_dims_eq_trident"] = tuple(osl.dimensions) == (W, H)
    except Exception as exc:  # recorded, not silently assumed
        rec["openslide_dims_eq_trident"] = f"unchecked: {type(exc).__name__}: {exc}"
    if rec["openslide_dims_eq_trident"] is False:
        raise ValueError(f"{slide}: openslide dimensions differ from trident level0 attrs")
    rec.update(level0_width=W, level0_height=H, n_patches=int(coords.shape[0]),
               coords_multiple_of_1024=bool((coords % PATCH == 0).all()),
               scale_x=W / raster.shape[1], scale_y=H / raster.shape[0],
               raster_w_eq_ceil_w_over_4=raster.shape[1] == -(-W // 4),
               raster_h_eq_ceil_h_over_4=raster.shape[0] == -(-H // 4),
               feature_shape=fshape)

    lab, npx, frac, nl0, clip = label_windows(raster, W, H, coords, PATCH)
    if (lab < 0).any():
        raise ValueError(f"{slide}: empty 1024 window(s)")
    rec["n_clipped_1024"] = int(clip.sum())

    # legacy 512 validation (same code path, P = 512)
    lpath = os.path.join(legacy_dir, f"{slide}.csv")
    rec["legacy_csv_sha256"] = sha256_file(lpath)
    leg = pd.read_csv(lpath)
    lc = leg[["wsi_loc_x", "wsi_loc_y"]].to_numpy(np.int64)
    if leg["slide_id"].astype(str).nunique() != 1 or str(leg["slide_id"].iloc[0]) != slide:
        raise ValueError(f"{slide}: legacy slide_id mismatch")
    llab, _, lfrac, _, lclip = label_windows(raster, W, H, lc, LEGACY_PATCH)
    legacy_label = leg["patch_label"].to_numpy(np.int64)
    agree = llab == legacy_label
    rec.update(n_legacy=int(len(leg)), n_legacy_agree=int(agree.sum()),
               n_legacy_clipped=int(lclip.sum()))
    dis = pd.DataFrame({"slide_id": slide, "x": lc[~agree, 0], "y": lc[~agree, 1],
                        "legacy_label": legacy_label[~agree], "recomputed_label": llab[~agree],
                        "recomputed_frac": lfrac[~agree]})
    legacy_map = {(int(a), int(b)): int(v) for (a, b), v in zip(lc.tolist(), legacy_label)}
    if len(legacy_map) != len(leg):
        raise ValueError(f"{slide}: duplicate legacy coordinates")
    n_av, n_tu = child_counts(coords, legacy_map, PATCH)

    # R2.1 accepted labels at their tiles vs legacy label and recomputed label
    r21_x, r21_y, r21_lab = r21_tiles
    leg_at = np.array([legacy_map.get((int(a), int(b)), -9) for a, b in zip(r21_x, r21_y)])
    rec["r21_tiles"] = int(len(r21_x))
    rec["r21_tiles_missing_in_legacy"] = int((leg_at == -9).sum())
    rec["r21_label_eq_legacy"] = int((leg_at == r21_lab).sum())
    if len(r21_x):
        rl, *_ = label_windows(raster, W, H, np.stack([r21_x, r21_y], 1), LEGACY_PATCH)
        rec["r21_label_eq_recomputed"] = int((rl == r21_lab).sum())
    else:
        rec["r21_label_eq_recomputed"] = 0
    rec["seconds"] = round(time.time() - t0, 2)
    out = {"x": coords[:, 0].astype(np.int64), "y": coords[:, 1].astype(np.int64),
           "label_1024": lab, "n_raster_pixels": npx, "frac_label_pixels": frac,
           "n512_available": n_av, "n512_tumor": n_tu,
           "feature_row": np.arange(coords.shape[0], dtype=np.int64)}
    return rec, out, dis


# ---------------------------------------------------------------- summaries
def summarise(labels, cross):
    import pandas as pd
    lab = labels
    s = {"class_names": {str(k): v for k, v in CLASS_NAMES.items()}}
    s["counts_overall"] = {str(k): int(v) for k, v in lab["label_1024"].value_counts().sort_index().items()}
    s["counts_by_fold"] = {str(f): {str(k): int(v) for k, v in g["label_1024"].value_counts().sort_index().items()}
                           for f, g in lab.groupby("held_out_fold")}
    s["counts_by_slide"] = {str(sl): {str(k): int(v) for k, v in g["label_1024"].value_counts().sort_index().items()}
                            for sl, g in lab.groupby("slide_id")}
    lab15 = lab[lab["label_1024"].between(1, 5)]
    s["patients_per_class"] = {str(k): int(g["patient_id"].nunique()) for k, g in lab15.groupby("label_1024")}
    s["slides_per_class"] = {str(k): int(g["slide_id"].nunique()) for k, g in lab15.groupby("label_1024")}
    s["patients_per_class_by_fold"] = {
        str(f): {str(k): int(gg["patient_id"].nunique()) for k, gg in g.groupby("label_1024")}
        for f, g in lab15.groupby("held_out_fold")}
    imm = lab[lab["label_1024"] == 2]
    s["immune"] = {
        "n_patches": int(len(imm)),
        "by_patient": {p: {"n": int(len(g)), "fold": int(g["held_out_fold"].iloc[0]),
                           "slides": sorted(g["slide_id"].unique().tolist())}
                       for p, g in imm.groupby("patient_id")},
        "folds_with_immune": sorted(int(f) for f in imm["held_out_fold"].unique()),
        "training_pool_immune_by_heldout_fold": {
            str(f): int((imm["held_out_fold"] != f).sum()) for f in range(5)},
    }
    s["training_pool_class_counts_by_heldout_fold"] = {
        str(f): {str(k): int(((lab15["held_out_fold"] != f) & (lab15["label_1024"] == k)).sum())
                 for k in range(1, 6)} for f in range(5)}
    s["n_slides"] = int(lab["slide_id"].nunique())
    s["n_patients"] = int(lab["patient_id"].nunique())
    s["n_patches"] = int(len(lab))
    s["n_labelled_1to5"] = int(len(lab15))
    s["frac_label_pixels_quantiles_labelled"] = {
        str(q): float(lab15["frac_label_pixels"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75)}
    s["cross_tab_label1024_vs_ge2of4_tumor"] = cross
    return s


def cross_tab(labels):
    import pandas as pd
    lab = labels
    out = {}
    for name, sub in (("all_patches", lab), ("all_four_512_available", lab[lab["n512_available"] == 4])):
        ge2 = np.where(sub["n512_tumor"] >= 2, "ge2of4_tumor", "lt2of4_tumor")
        ct = pd.crosstab(sub["label_1024"].map(lambda v: f"{v}:{CLASS_NAMES[v]}"), ge2)
        out[name] = {r: {c: int(ct.loc[r, c]) for c in ct.columns} for r in ct.index}
        t1024 = sub["label_1024"].to_numpy() == 1
        g = sub["n512_tumor"].to_numpy() >= 2
        out[name + "_tumor_agreement"] = {
            "n": int(len(sub)), "both_tumor": int((t1024 & g).sum()),
            "only_1024_tumor": int((t1024 & ~g).sum()), "only_ge2of4": int((~t1024 & g).sum()),
            "neither": int((~t1024 & ~g).sum())}
    out["n512_available_distribution"] = {str(k): int(v) for k, v in
                                          lab["n512_available"].value_counts().sort_index().items()}
    out["note"] = ("ge2of4 here uses the pathologist-derived legacy 512 labels (patch_label==1), "
                   "not classifier predictions; diagnostic comparison of aggregation rules only.")
    return out


# ---------------------------------------------------------------- main
def main(argv=None):
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6, help="<= 6 on the shared login node")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N slides only")
    args = ap.parse_args(argv)
    t_start = time.time()
    prosper_out = os.environ.get("PROSPER_OUT")
    if not prosper_out:
        sys.exit("PROSPER_OUT must be exported")
    sub = SUB if not args.limit else f"{SUB}_debug_limit{args.limit}"
    out_dir = Path(prosper_out) / OUT_NAME / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "reconciliation.json").exists():
        sys.exit(f"{out_dir}/reconciliation.json exists; refusing to overwrite")

    r21 = Path(prosper_out) / R21_FILTER_DIR
    s1a = Path(prosper_out) / R21_STAGE1A_DIR
    asg = pd.read_csv(r21 / "annotation_assignments.csv.gz", dtype={"slide_id": str, "patient_id": str})
    folds = pd.read_csv(r21 / "patient_folds.csv", dtype={"patient_id": str})
    if folds["patient_id"].duplicated().any():
        raise ValueError("duplicate patient in patient_folds.csv")
    fold_of = dict(zip(folds["patient_id"], folds["fold"].astype(int)))
    meta = asg.groupby("slide_id").agg(patient_id=("patient_id", "unique"),
                                       diagnostic_group=("diagnostic_group", "unique"),
                                       held_out_fold=("held_out_fold", "unique"))
    for c in meta.columns:
        if (meta[c].map(len) != 1).any():
            raise ValueError(f"slide with non-unique {c}")
        meta[c] = meta[c].map(lambda v: v[0])
    mismatch = [s for s, r in meta.iterrows() if fold_of.get(r["patient_id"]) != int(r["held_out_fold"])]
    if mismatch:
        raise ValueError(f"assignment held_out_fold disagrees with patient_folds for {mismatch[:5]}")
    slides = sorted(meta.index.tolist(), key=lambda s: (len(s), s))
    if len(slides) != 202 or meta["patient_id"].nunique() != 192 or len(asg) != 468301:
        raise ValueError("R2.1 population is not 202 slides / 192 patients / 468301 tiles")

    rasters = sorted(p.stem for p in Path(RASTER_DIR).glob("*.tif"))
    inv = pd.read_csv(s1a / "slide_inventory.csv", dtype={"slide_id": str, "patient_id": str})
    quar = pd.read_csv(s1a / "quarantine_tiles.csv", dtype={"slide_id": str}, usecols=["slide_id", "reason"])
    qreason = quar.groupby("slide_id")["reason"].unique().map(lambda v: ";".join(sorted(v))).to_dict()
    diag = json.loads((s1a / "diagnostics.json").read_text())
    extra = set()
    for e in diag.get("entries", []):
        if e.get("kind") == "additional_canonical_slides":
            extra |= set(map(str, e.get("examples", [])))
    inv_elig = dict(zip(inv["slide_id"], inv["eligibility"]))
    conch_ids = {p.name[:-3] for p in (Path(TRIDENT_DIR) / "features_conch_v15").glob("*.h5")}
    excluded = []
    for s in rasters:
        if s in meta.index:
            continue
        if s in inv_elig:
            reason = f"stage1a eligibility={inv_elig[s]}; reason={qreason.get(s, 'n/a')}"
        elif s in extra:
            reason = "not in the historical 237-slide annotation target (Stage1a diagnostics: additional_canonical_slides)"
        else:
            reason = "not in Stage1a historical target/inventory and not listed as additional canonical slide"
        excluded.append({"slide_id": s, "reason": reason, "has_conch_features": s in conch_ids})
    md = pd.read_csv(METADATA, dtype=str).set_index("Slide.Scan.File")
    md_mismatch = [s for s in slides if s not in md.index or md.loc[s, "UID.Subject"] != meta.loc[s, "patient_id"]
                   or md.loc[s, "Tumor.group"] != meta.loc[s, "diagnostic_group"]]
    if md_mismatch:
        raise ValueError(f"R2.1 patient/group differ from slide metadata for {md_mismatch[:5]}")
    for e in excluded:
        e["in_2921_slide_metadata"] = e["slide_id"] in md.index
    missing_raster = [s for s in slides if s not in set(rasters)]
    if missing_raster:
        raise ValueError(f"R2.1 slides without raster: {missing_raster}")
    if args.limit:
        slides = slides[:args.limit]

    asg_by = {s: g for s, g in asg.groupby("slide_id")}
    tasks = [(s, RASTER_DIR, TRIDENT_DIR, LEGACY_DIR,
              (asg_by[s]["x"].to_numpy(np.int64), asg_by[s]["y"].to_numpy(np.int64),
               asg_by[s]["tissue_label"].to_numpy(np.int64))) for s in slides]
    # legacy read rasters with skimage.io.imread; confirm identical arrays on the 3 smallest
    reader_check = {}
    import skimage.io
    import tifffile
    for s in sorted(slides, key=lambda s: os.path.getsize(os.path.join(RASTER_DIR, f"{s}.tif")))[:3]:
        p = os.path.join(RASTER_DIR, f"{s}.tif")
        reader_check[s] = bool(np.array_equal(skimage.io.imread(p), tifffile.memmap(p, mode="r")))
    if not all(reader_check.values()):
        raise ValueError(f"skimage/tifffile raster mismatch: {reader_check}")
    recs, frames, dis = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, (rec, out, d) in enumerate(ex.map(process_slide, tasks)):
            s = rec["slide_id"]
            df = pd.DataFrame(out)
            df.insert(0, "slide_id", s)
            df.insert(1, "patient_id", meta.loc[s, "patient_id"])
            df.insert(2, "diagnostic_group", meta.loc[s, "diagnostic_group"])
            df.insert(3, "held_out_fold", int(meta.loc[s, "held_out_fold"]))
            frames.append(df[LABEL_COLUMNS])
            recs.append(rec)
            if len(d):
                dis.append(d)
            if (k + 1) % 20 == 0:
                print(f"[{time.time() - t_start:7.1f}s] {k + 1}/{len(tasks)} slides", flush=True)
    labels = pd.concat(frames, ignore_index=True)
    labels.to_csv(out_dir / "labels_1024.csv.gz", index=False, float_format="%.6g")
    disdf = pd.concat(dis, ignore_index=True) if dis else pd.DataFrame(
        columns=["slide_id", "x", "y", "legacy_label", "recomputed_label", "recomputed_frac"])
    disdf.to_csv(out_dir / "validation_512_disagreements.csv.gz", index=False)
    per_slide = pd.DataFrame(recs)
    per_slide.drop(columns=["raster_values", "raster_shape_hw", "feature_shape", "raster_pixel_counts"]).to_csv(
        out_dir / "per_slide_geometry_validation.csv", index=False)

    cross = cross_tab(labels)
    summary = summarise(labels, cross)
    n_leg, n_agree = int(per_slide["n_legacy"].sum()), int(per_slide["n_legacy_agree"].sum())
    leg_ct = pd.crosstab(disdf["legacy_label"], disdf["recomputed_label"]) if len(disdf) else None
    validation = {
        "scope": "all legacy 512 tile coordinates of all 202 R2.1 slides (superset of the >=20-slide requirement)",
        "n_slides": int(len(per_slide)), "n_tiles": n_leg, "n_agree": n_agree,
        "agreement": n_agree / n_leg if n_leg else None,
        "min_slide_agreement": float((per_slide["n_legacy_agree"] / per_slide["n_legacy"]).min()),
        "slides_below_99pct": per_slide.loc[per_slide["n_legacy_agree"] / per_slide["n_legacy"] < 0.99,
                                            "slide_id"].tolist(),
        "disagreement_crosstab_legacy_rows_recomputed_cols": (
            {str(r): {str(c): int(leg_ct.loc[r, c]) for c in leg_ct.columns} for r in leg_ct.index}
            if leg_ct is not None else {}),
        "r21_tiles": int(per_slide["r21_tiles"].sum()),
        "r21_tiles_missing_in_legacy": int(per_slide["r21_tiles_missing_in_legacy"].sum()),
        "r21_label_eq_legacy": int(per_slide["r21_label_eq_legacy"].sum()),
        "r21_label_eq_recomputed": int(per_slide["r21_label_eq_recomputed"].sum()),
    }
    summary["validation_512"] = validation
    geo = {
        "scale_x_range": [float(per_slide["scale_x"].min()), float(per_slide["scale_x"].max())],
        "scale_y_range": [float(per_slide["scale_y"].min()), float(per_slide["scale_y"].max())],
        "raster_dims_eq_ceil_level0_over_4": int((per_slide["raster_w_eq_ceil_w_over_4"]
                                                  & per_slide["raster_h_eq_ceil_h_over_4"]).sum()),
        "coords_multiple_of_1024_slides": int(per_slide["coords_multiple_of_1024"].sum()),
        "n_clipped_1024_patches": int(per_slide["n_clipped_1024"].sum()),
        "n_clipped_512_legacy_tiles": int(per_slide["n_legacy_clipped"].sum()),
        "raster_value_sets": sorted({tuple(v) for v in per_slide["raster_values"]}),
        "raster_pixel_counts_total": {str(k): int(sum(int(d.get(str(k), 0)) for d in per_slide["raster_pixel_counts"]))
                                      for k in range(6)},
        "openslide_dims_eq_trident": {str(k): int(v) for k, v in
                                      per_slide["openslide_dims_eq_trident"].astype(str).value_counts().items()},
        "example": per_slide.loc[per_slide["slide_id"] == "10060",
                                 ["slide_id", "raster_shape_hw", "level0_width", "level0_height",
                                  "scale_x", "scale_y"]].to_dict("records"),
        "axis_order": "raster array is (rows=y, cols=x); trident coords are (x, y); level0_width pairs with raster columns",
    }
    summary["geometry"] = geo
    summary["excluded_raster_slides"] = excluded
    summary["definitions"] = {
        "label_1024": "majority value of the nearest-neighbour-resized raster in the clipped 1024 window; ties -> lowest label",
        "n_raster_pixels": "number of distinct raster (downsample-4) pixels the clipped window maps to",
        "frac_label_pixels": "fraction of the clipped window's level-0 pixels carrying label_1024 (after NN resize)",
        "n512_available": "number of the 4 constituent 512 tiles present in the legacy pathologist_annotated_labels CSV (legacy tissue tiles only)",
        "n512_tumor": "number of those constituent tiles with legacy patch_label == 1 (Tumor)",
        "feature_row": "row index in the CONCH v1.5 h5 (identical to the trident patches h5 coords order)",
        "held_out_fold": "R2.1 patient fold (patient_folds.csv), verified equal to annotation_assignments.held_out_fold",
    }
    (out_dir / "labels_summary.json").write_text(json.dumps(summary, indent=1, default=str))

    import cv2, h5py, pandas, sklearn, tifffile  # noqa: E401 (versions only)
    inputs = {
        "annotation_assignments": str(r21 / "annotation_assignments.csv.gz"),
        "patient_folds": str(r21 / "patient_folds.csv"),
        "stage1a_slide_inventory": str(s1a / "slide_inventory.csv"),
        "stage1a_quarantine_tiles": str(s1a / "quarantine_tiles.csv"),
        "stage1a_diagnostics": str(s1a / "diagnostics.json"),
        "legacy_label_script": LEGACY_SCRIPT,
        "legacy_trainer_class_names": LEGACY_TRAINER,
        "qupath_export_script": QUPATH_EXPORT_SCRIPT,
        "slide_metadata": METADATA,
        "this_script": os.path.abspath(__file__),
    }
    rec = {
        "run_id": f"{OUT_NAME}/{sub}", "created": datetime.datetime.now().astimezone().isoformat(),
        "argv": sys.argv, "seconds": round(time.time() - t_start, 1), "workers": args.workers,
        "limit": args.limit,
        "inputs": {k: {"path": v, "sha256": sha256_file(v), "bytes": os.path.getsize(v)} for k, v in inputs.items()},
        "per_slide_input_sha256": {r["slide_id"]: {k: r[k] for k in ("raster_sha256", "patches_h5_sha256",
                                                                     "conch_h5_sha256", "legacy_csv_sha256")}
                                   for r in recs},
        "outputs": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                    for p in sorted(out_dir.iterdir()) if p.is_file()},
        "checks": {
            "skimage_imread_eq_tifffile_memmap_smallest_rasters": reader_check,
            "r21_patient_and_group_eq_slide_metadata": True,
            "r21_population_202_slides_192_patients_468301_tiles": True,
            "assignment_fold_eq_patient_folds": True,
            "patch_coords_eq_conch_coords_all_slides": True,
            "raster_values_within_0_5": True,
            "n_raster_slides": len(rasters), "n_excluded_raster_slides": len(excluded),
            "labels_rows": int(len(labels)),
            "validation_512_agreement": validation["agreement"],
        },
        "class_meaning_source": "exportQupathMask.groovy: backgroundLabel 0, Tumor 1, Immune cells 2, Necrosis 3, nonTumor 4, Artifacts 5; downsample 4",
        "environment": {"python": sys.version, "platform": platform.platform(), "host": platform.node(),
                        "numpy": np.__version__, "pandas": pandas.__version__, "h5py": h5py.__version__,
                        "tifffile": tifffile.__version__, "cv2": cv2.__version__,
                        "sklearn": sklearn.__version__},
        "note": "cv2 is used only by the test (index-map equivalence); labels never call cv2.resize.",
    }
    (out_dir / "reconciliation.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({"validation_512": validation, "counts_overall": summary["counts_overall"],
                      "seconds": rec["seconds"]}, default=str))


if __name__ == "__main__":
    main()
