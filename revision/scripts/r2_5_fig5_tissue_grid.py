#!/usr/bin/env python3
"""Regenerate the 512-px tissue-tile grid of the Fig. 5 slides for the nearest-mosaic-tile fill rendering.

Why: the shipped cluster-map renderer (prosper-search/prosper/visualization/ot_visualization.py 1f664155...,
generate_ot_heatmap_overlay) paints each mosaic tile's label onto `tile_info['cluster_locs']`, the level-0
locations of every tissue tile in that representative's spatial sub-cluster (mosaic.py: within each colour
k-means cluster, spatial k-means into n_selected groups; the representative is the tile nearest each group
centre and `cluster_locs` = that group). The saved coordinate pickles (comet-uni2/data/tmp_final/<scan>.pkl)
hold only `wsi_loc`, so the published dense colouring cannot be reproduced from them directly.

This script re-implements prosper-search/prosper/preprocessing/mosaic.py (`clean_thumbnail`, `rgb_to_hsd`,
`process_slide`) verbatim in logic, without importing the package. Finding (analyst trial, 16875, sklearn 1.7.0):
the tissue grid reproduces (all 1,334 saved mosaic tiles lie inside the 8,935 regenerated tissue tiles) but
the k-means representative choice does not (531/1,334 identical; the reference mosaic.py docstring records that
the published mosaics predate seed pinning, so this is expected and not attributable to one library version),
so the ORIGINAL group memberships are not recoverable. Design decision (2026-10-06): colour every tissue tile
with the label of its nearest saved mosaic-tile centre. This is a reproducible rule using the same centres, but
it is NOT the original grouping: the original groups were spatial k-means cells within colour clusters, which
overlap spatially, and nearest-centre assignment agrees with them for only ~52-67 % of tissue tiles (independent
measurement on regenerated mosaics). This script therefore outputs the tissue
grid and PASSes when the saved mosaic is a subset of it; the representative agreement is recorded only as a
diagnostic. Nothing is silently substituted: missing saved tiles are listed and the slide FAILs.

Outputs per slide: `<scan>_tissue_grid.json` {status, tissue_tiles [[x,y],...], n_saved, n_saved_in_tissue,
saved_not_in_tissue, n_regenerated_representatives, n_representatives_matching, ...} and summary.json.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("PROSPER_OUT", os.path.join(R, "prosper_revision_output"))
H = "/PROSPER_WORKSPACE"
COORD_DIR = f"{H}/comet-feature-gen/comet-uni2/data/tmp_final"
WSI_DIR = f"{H}/data/comet_final_HPC"
SLIDES = ["16875", "16867", "15155", "7333", "7330"]


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def rgb_to_hsd(X):
    eps = np.finfo(float).eps
    X = np.clip(X, eps, 1.0)
    OD = -np.log(X / 1.0)
    D = np.mean(OD, 3)
    D = np.clip(D, eps, None)
    cx = OD[:, :, :, 0] / D - 1.0
    cy = (OD[:, :, :, 1] - OD[:, :, :, 2]) / (np.sqrt(3.0) * D)
    return np.concatenate((np.expand_dims(D, 3), np.expand_dims(cx, 3), np.expand_dims(cy, 3)), 3)


def clean_thumbnail(thumbnail):
    import cv2
    arr = np.asarray(thumbnail)
    w = np.copy(arr)
    w[np.std(w, axis=2) < 5] = 255
    hsd = rgb_to_hsd(np.array([w.astype("float32") / 255.]))[0]
    kernel = np.ones((30, 30), np.float32) / 900
    w[cv2.filter2D(hsd[:, :, 2], -1, kernel) < 0.05] = 255
    return w


def process_slide(slide_fn, patch_size=512, percentage_selected=15, tissue_threshold=0.7):
    import openslide
    from PIL import Image
    from sklearn.cluster import KMeans
    slide = openslide.open_slide(slide_fn)
    w, h = slide.dimensions
    try:
        thumbnail = slide.get_thumbnail((500, 500))
    except Exception:
        level_image = slide.read_region((0, 0), 2, slide.level_dimensions[2]).convert("RGB")
        thumbnail = level_image.resize((500, 500), Image.BILINEAR)
    cthumb = clean_thumbnail(thumbnail)
    tissue_mask = (cthumb.mean(axis=2) != 255).astype(float)
    mask_hratio = (tissue_mask.shape[0] / h) * patch_size
    mask_wratio = (tissue_mask.shape[1] / w) * patch_size
    patches = []
    for i, hi in enumerate(range(0, h, int(patch_size))):
        for j, wi in enumerate(range(0, w, int(patch_size))):
            mi, mj = int(i * mask_hratio), int(j * mask_wratio)
            pm = tissue_mask[mi:mi + int(mask_hratio), mj:mj + int(mask_wratio)]
            cov = np.count_nonzero(pm) / pm.size if pm.size > 0 else 0
            patches.append({"loc": [i, j], "wsi_loc": [int(wi), int(hi)], "tissue_coverage": cov})
    selected = [p for p in patches if p["tissue_coverage"] >= tissue_threshold]
    tissue = [p["wsi_loc"] for p in selected]
    info = dict(n_grid=len(patches), n_tissue=len(selected), level0_dims=[int(w), int(h)],
                thumbnail_size=list(thumbnail.size))
    if not selected:
        return None, tissue, info
    for p in selected:
        wp, hp = p["wsi_loc"]
        ps5 = patch_size // 4
        try:
            reg = slide.read_region((wp, hp), 1, (ps5, ps5)).convert("RGB")
            if reg.size[0] != 250:
                reg = reg.resize((250, 250))
            p["rgb_histogram"] = (np.array(reg) / 255.).reshape((-1, 3)).mean(axis=0)
        except Exception:
            p["rgb_histogram"] = np.array([0.5, 0.5, 0.5])
    feats = np.array([p["rgb_histogram"] for p in selected])
    kc = min(9, len(feats))
    km = KMeans(n_clusters=kc, random_state=3, n_init=10).fit(feats)
    mosaic = []
    for i in range(kc):
        cp = [p for p, l in zip(selected, km.labels_) if l == i]
        n_sel = max(1, int(len(cp) * percentage_selected / 100.))
        n_sel = min(n_sel, len(cp))
        if n_sel == 0:
            continue
        locf = np.array([p["wsi_loc"] for p in cp])
        km2 = KMeans(n_clusters=n_sel, random_state=3, n_init=10)
        ds = km2.fit_transform(locf)
        dl = km2.labels_
        chosen = set()
        for idx in range(n_sel):
            order = np.argsort(ds[:, idx])
            members = [locf[j].tolist() for j in range(len(locf)) if dl[j] == idx]
            for sidx in order:
                if sidx not in chosen:
                    chosen.add(sidx)
                    cp[sidx]["cluster_locs"] = members
                    mosaic.append(cp[sidx])
                    break
    return mosaic, tissue, dict(info, n_colour_clusters=kc)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(OUT, f"r2_5_fig5_tissue_grid_{dt.date.today():%Y%m%d}"))
    ap.add_argument("--slides", nargs="*", default=SLIDES)
    a = ap.parse_args(argv)
    if os.path.exists(a.out_dir) and os.listdir(a.out_dir):
        sys.exit(f"refusing to write into non-empty {a.out_dir}")
    os.makedirs(a.out_dir, exist_ok=True)
    import sklearn, openslide, cv2, PIL
    summary = dict(generated=dt.datetime.now().isoformat(), script=os.path.relpath(__file__, R),
                   script_sha256=sha256_file(__file__),
                   versions=dict(python=sys.version.split()[0], numpy=np.__version__, sklearn=sklearn.__version__,
                                 openslide=openslide.__version__, opencv=cv2.__version__, PIL=PIL.__version__),
                   params=dict(patch_size=512, percentage_selected=15, tissue_threshold=0.7, kmeans_random_state=3,
                               kmeans_n_init=10, max_colour_clusters=9),
                   pass_rule="saved mosaic tiles are a subset of the regenerated tissue grid; representative agreement is diagnostic only",
                   slides={})
    for scan in a.slides:
        cp = os.path.join(COORD_DIR, f"{scan}.pkl")
        wsi = os.path.join(WSI_DIR, f"{scan}.svs")
        with open(cp, "rb") as f:
            feats, locs = pickle.load(f)
        saved = {tuple(int(v) for v in l["wsi_loc"]) for l in locs}
        t0 = dt.datetime.now()
        mosaic, tissue, info = process_slide(wsi)
        secs = (dt.datetime.now() - t0).total_seconds()
        tissue_set = {tuple(t) for t in tissue}
        regen = {tuple(p["wsi_loc"]) for p in mosaic} if mosaic else set()
        not_in_tissue = sorted(saved - tissue_set)
        status = "PASS" if (not not_in_tissue and len(saved) == len(locs) and len(tissue_set) == len(tissue)) else "FAIL"
        rec = dict(status=status, scan=scan, wsi=wsi, wsi_target=os.path.realpath(wsi), wsi_sha256=sha256_file(wsi),
                   coords=cp, coords_sha256=sha256_file(cp), n_saved=len(saved), n_saved_rows=len(locs),
                   n_saved_in_tissue=len(saved & tissue_set), n_saved_not_in_tissue=len(not_in_tissue),
                   saved_not_in_tissue=not_in_tissue[:50], n_regenerated_representatives=len(regen),
                   n_representatives_matching=len(saved & regen), seconds=secs, **info)
        with open(os.path.join(a.out_dir, f"{scan}_tissue_grid.json"), "w") as f:
            json.dump(dict(rec, tissue_tiles=sorted(tissue_set)), f)
        summary["slides"][scan] = rec
        print(f"{scan}: {status} saved {len(saved)} in tissue {len(saved & tissue_set)} not in tissue {len(not_in_tissue)}; "
              f"tissue {info['n_tissue']} of {info['n_grid']} grid; representatives regenerated {len(regen)} "
              f"matching {len(saved & regen)} (diagnostic); {secs:.0f}s")
    summary["status"] = "PASS" if all(r["status"] == "PASS" for r in summary["slides"].values()) else "FAIL"
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(summary["status"])
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
