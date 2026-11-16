import os, sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import r2_6_titan_patch_labels_1024 as L  # noqa: E402


def legacy_reference(raster, W, H, coords, patch):
    """Literal legacy rule: cv2 nearest resize to slide dims, slice, np.unique + argmax."""
    full = cv2.resize(raster, (W, H), interpolation=cv2.INTER_NEAREST)
    out = []
    for x, y in coords:
        win = full[y:y + patch, x:x + patch]
        u, c = np.unique(win, return_counts=True)
        out.append(int(u[np.argmax(c)]))
    return np.array(out)


def test_nn_index_matches_cv2_resize():
    rng = np.random.default_rng(1)
    for _ in range(100):
        sh, sw = rng.integers(3, 120, 2)
        dh = int(sh * rng.uniform(1, 6)) + int(rng.integers(0, 4))
        dw = int(sw * rng.uniform(1, 6)) + int(rng.integers(0, 4))
        a = rng.integers(0, 6, (sh, sw), dtype=np.uint8)
        ref = cv2.resize(a, (dw, dh), interpolation=cv2.INTER_NEAREST)
        m = a[np.ix_(L.nn_source_index(dh, sh), L.nn_source_index(dw, sw))]
        assert np.array_equal(m, ref)


def test_nn_index_real_geometry():
    for src, dst in ((21912, 87647), (19201, 76804)):
        a = np.tile(np.arange(src, dtype=np.float32), (2, 1))
        ref = cv2.resize(a, (dst, 2), interpolation=cv2.INTER_NEAREST)[0].astype(np.int64)
        assert np.array_equal(L.nn_source_index(dst, src), ref)


def test_nn_index_real_geometry_uint8_both_axes():
    """uint8 (the raster dtype) through cv2 at the 10060 geometry, columns and rows."""
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (2, 21912), dtype=np.uint8)
    ref = cv2.resize(a, (87647, 2), interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(a[:, L.nn_source_index(87647, 21912)], ref)
    b = rng.integers(0, 256, (19201, 2), dtype=np.uint8)
    ref = cv2.resize(b, (2, 76804), interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(b[L.nn_source_index(76804, 19201), :], ref)


def test_window_label_on_memmap_equals_array(tmp_path):
    import tifffile
    rng = np.random.default_rng(4)
    a = np.kron(rng.integers(0, 6, (12, 15)), np.ones((4, 4), np.uint8)).astype(np.uint8)
    p = tmp_path / "r.tif"
    tifffile.imwrite(p, a, imagej=True)
    m = tifffile.memmap(p, mode="r")
    coords = np.array([(x, y) for x in range(0, 230, 32) for y in range(0, 190, 32)])
    r1 = L.label_windows(a, 230, 190, coords, 32)
    r2 = L.label_windows(m, 230, 190, coords, 32)
    for u, v in zip(r1, r2):
        assert np.array_equal(u, v)


def test_window_label_equals_legacy_rule_with_clipping():
    rng = np.random.default_rng(2)
    for trial in range(15):
        rh, rw = rng.integers(20, 60, 2)
        # blocky raster so majorities are not all ties
        a = np.kron(rng.integers(0, 6, (rh // 4 + 1, rw // 4 + 1)), np.ones((4, 4), np.uint8))[:rh, :rw].astype(np.uint8)
        H, W = int(rh * 4 - rng.integers(0, 3)), int(rw * 4 - rng.integers(0, 3))
        for patch in (16, 32):
            coords = np.array([(x, y) for x in range(0, W, patch) for y in range(0, H, patch)])
            ref = legacy_reference(a, W, H, coords, patch)
            lab, npx, frac, nl0, clip = L.label_windows(a, W, H, coords, patch)
            assert np.array_equal(lab, ref), (trial, patch)
            if W % patch or H % patch:
                assert clip.any()  # last row/col windows are clipped at the slide boundary
            assert (frac > 0).all() and (frac <= 1).all()
            assert (nl0 <= patch * patch).all()


def test_ties_resolve_to_lowest_label():
    a = np.zeros((4, 4), np.uint8)
    a[:, 2:] = 4  # half 0 / half 4 -> 0
    lab, *_ = L.label_windows(a, 4, 4, np.array([[0, 0]]), 4)
    assert lab[0] == 0
    a[:, :2] = 3  # half 3 / half 4 -> 3
    lab, npx, frac, nl0, clip = L.label_windows(a, 4, 4, np.array([[0, 0]]), 4)
    assert lab[0] == 3 and frac[0] == 0.5 and npx[0] == 16 and not clip[0]


def test_multiplicity_weights_matter():
    # raster 1x3 resized to width 10: idx = floor(x*0.3) -> [0,0,0,0,1,1,1,2,2,2]
    a = np.array([[1, 2, 2]], np.uint8)
    lab, npx, frac, *_ = L.label_windows(a, 10, 1, np.array([[0, 0]]), 10)
    assert lab[0] == 2 and npx[0] == 3 and abs(frac[0] - 0.6) < 1e-12
    # unweighted majority over distinct raster pixels would also say 2; make it differ
    lab, npx, frac, *_ = L.label_windows(a, 10, 1, np.array([[0, 0]]), 6)  # x 0..5 -> 0,0,0,0,1,1
    assert lab[0] == 1 and abs(frac[0] - 4 / 6) < 1e-12


def test_out_of_slide_window_is_flagged():
    a = np.ones((2, 2), np.uint8)
    lab, npx, *_ = L.label_windows(a, 8, 8, np.array([[8, 0]]), 4)
    assert lab[0] == -1 and npx[0] == 0


def test_child_counts():
    legacy = {(0, 0): 1, (512, 0): 1, (0, 512): 4, (1024, 0): 1, (1536, 512): 1}
    n_av, n_tu = L.child_counts(np.array([[0, 0], [1024, 0], [2048, 0]]), legacy)
    assert n_av.tolist() == [3, 2, 0] and n_tu.tolist() == [2, 2, 0]


def test_cross_tab_and_summary():
    lab = pd.DataFrame({
        "slide_id": ["a", "a", "b", "b", "c"], "patient_id": ["p", "p", "q", "q", "r"],
        "diagnostic_group": "G", "held_out_fold": [0, 0, 1, 1, 2],
        "x": 0, "y": 0, "label_1024": [1, 2, 0, 1, 4], "n_raster_pixels": 1,
        "frac_label_pixels": [1.0, 0.6, 0.9, 0.5, 0.7],
        "n512_available": [4, 4, 2, 4, 4], "n512_tumor": [2, 0, 0, 1, 3], "feature_row": 0})
    ct = L.cross_tab(lab)
    assert ct["all_patches_tumor_agreement"] == {"n": 5, "both_tumor": 1, "only_1024_tumor": 1,
                                                 "only_ge2of4": 1, "neither": 2}
    assert ct["all_four_512_available_tumor_agreement"]["n"] == 4
    s = L.summarise(lab, ct)
    assert s["counts_overall"] == {"0": 1, "1": 2, "2": 1, "4": 1}
    assert s["immune"]["by_patient"]["p"]["n"] == 1
    assert s["immune"]["training_pool_immune_by_heldout_fold"] == {"0": 0, "1": 1, "2": 1, "3": 1, "4": 1}
    assert s["training_pool_class_counts_by_heldout_fold"]["1"]["1"] == 1
    assert s["patients_per_class"] == {"1": 2, "2": 1, "4": 1}
