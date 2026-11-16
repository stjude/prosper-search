"""R2.2-ALL Stage A: faithful reconstruction of the mosaic tissue-eligibility stage.

`RGB2HSD` and `clean_thumbnail` are copied verbatim (whitespace aside) from
`comet-new-scans/src/process_mosaic/lib.py` (identical to
`comet-annotation/src/create_all_patches_for_all_slides/lib.py`); the thumbnail
route and grid/coverage loop follow `process_mosaic/lib.py::process_slide`
(lines ~40-77). No k-means or mosaic selection is performed. Nothing is
imported from the read-only comet-* trees.
"""
from __future__ import annotations

import numpy as np

TILE = 512
THUMB = 500
ELIGIBLE_THRESHOLD = 0.7


def RGB2HSD(X):
    eps = np.finfo(float).eps
    X[np.where(X == 0.0)] = eps
    OD = -np.log(X / 1.0)
    D = np.mean(OD, 3)
    D[np.where(D == 0.0)] = eps
    cx = OD[:, :, :, 0] / (D) - 1.0
    cy = (OD[:, :, :, 1] - OD[:, :, :, 2]) / (np.sqrt(3.0) * D)
    D = np.expand_dims(D, 3)
    cx = np.expand_dims(cx, 3)
    cy = np.expand_dims(cy, 3)
    X_HSD = np.concatenate((D, cx, cy), 3)
    return X_HSD


def clean_thumbnail(thumbnail):
    import cv2

    thumbnail_arr = np.asarray(thumbnail)
    wthumbnail = np.zeros_like(thumbnail_arr)
    wthumbnail[:, :, :] = thumbnail_arr[:, :, :]
    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std < 5] = (np.ones((1, 3), dtype="uint8") * 255)
    thumbnail_HSD = RGB2HSD(np.array([wthumbnail.astype('float32') / 255.]))[0]
    kernel = np.ones((30, 30), np.float32) / 900
    thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:, :, 2], -1, kernel)
    wthumbnail[thumbnail_HSD_mean < 0.05] = (np.ones((1, 3), dtype="uint8") * 255)
    return wthumbnail


def get_thumbnail(slide, slide_fn):
    """process_mosaic thumbnail route; returns (thumbnail, route).

    As in the original, the fallback re-opens the slide (an OpenSlide handle is
    unusable after an error) and resizes level 2 to 500x500 regardless of aspect.
    """
    import openslide
    from openslide import OpenSlideError
    from PIL import Image

    try:
        return slide.get_thumbnail((THUMB, THUMB)), "get_thumbnail"
    except OpenSlideError:
        slide = openslide.open_slide(str(slide_fn))
        level_image = slide.read_region((0, 0), 2, slide.level_dimensions[2]).convert("RGB")
        return level_image.resize((THUMB, THUMB), Image.BILINEAR), "level2_fallback"


def tissue_mask_from_thumbnail(thumbnail):
    cthumbnail = clean_thumbnail(thumbnail)
    return (cthumbnail.mean(axis=2) != 255).astype(float)


def grid_coverage(tissue_mask, w, h, patch_size=TILE):
    """Coverage of every level-0 grid cell, exactly as the original double loop.

    Returns int64 arrays i (row), j (col), wsi_x, wsi_y and float64 coverage in
    the original row-major loop order. A zero-size mask slice (where the
    original would raise ZeroDivisionError) raises ValueError.
    """
    mask_hratio = (tissue_mask.shape[0] / h) * patch_size
    mask_wratio = (tissue_mask.shape[1] / w) * patch_size
    ii, jj, xs, ys, cov = [], [], [], [], []
    for i, hi in enumerate(range(0, h, int(patch_size))):
        for j, wi in enumerate(range(0, w, int(patch_size))):
            mi = int(i * mask_hratio)
            mj = int(j * mask_wratio)
            patch_mask = tissue_mask[mi:mi + int(mask_hratio), mj:mj + int(mask_wratio)]
            if patch_mask.size == 0:
                raise ValueError(f"zero-size mask slice at cell ({i},{j})")
            tissue_coverage = np.count_nonzero(patch_mask) / patch_mask.size
            ii.append(i); jj.append(j); xs.append(int(wi)); ys.append(int(hi)); cov.append(tissue_coverage)
    return (np.asarray(ii, np.int64), np.asarray(jj, np.int64), np.asarray(xs, np.int64),
            np.asarray(ys, np.int64), np.asarray(cov, np.float64))


def coord_keys(xy):
    """Pack integer (x, y) pairs into int64 keys (0 <= x, y < 2**31)."""
    a = np.asarray(xy, dtype=np.int64).reshape(-1, 2)
    if a.size and (a.min() < 0 or a.max() >= 2 ** 31):
        raise ValueError("coordinate out of packable range")
    return (a[:, 0] << 32) | a[:, 1]


def n_duplicates(keys):
    keys = np.asarray(keys)
    return int(keys.size - np.unique(keys).size)


def containment(sub_keys, sup_keys):
    """Per element of sub: number of occurrences in sup."""
    sup_sorted = np.sort(np.asarray(sup_keys))
    sub = np.asarray(sub_keys)
    left = np.searchsorted(sup_sorted, sub, side="left")
    right = np.searchsorted(sup_sorted, sub, side="right")
    return right - left


def set_diff_counts(a_keys, b_keys):
    a = np.unique(np.asarray(a_keys)); b = np.unique(np.asarray(b_keys))
    return int(np.setdiff1d(a, b).size), int(np.setdiff1d(b, a).size)


def barcode(f):
    return np.diff(np.asarray(f, dtype=np.float32), axis=1) < 0


def row_compare(a, b):
    """Per-row cosine, max|d|, Yottixel barcode-bit agreement and exact equality."""
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    a64 = a.astype(np.float64); b64 = b.astype(np.float64)
    cos = (a64 * b64).sum(1) / (np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1))
    mad = np.abs(a64 - b64).max(1)
    agree = (barcode(a) == barcode(b)).mean(1)
    exact = np.all(a == b, axis=1)
    return cos, mad, agree, exact
