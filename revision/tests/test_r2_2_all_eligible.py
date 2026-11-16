import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "r2_2_all_eligible"))
import eligibility_lib as L  # noqa: E402


def _reference_loop(mask, w, h, ps=512):
    hr = (mask.shape[0] / h) * ps
    wr = (mask.shape[1] / w) * ps
    out = {}
    for i, hi in enumerate(range(0, h, ps)):
        for j, wi in enumerate(range(0, w, ps)):
            pm = mask[int(i * hr):int(i * hr) + int(hr), int(j * wr):int(j * wr) + int(wr)]
            out[(wi, hi)] = np.count_nonzero(pm) / pm.size
    return out


def test_grid_coverage_matches_reference_with_ragged_edges():
    rng = np.random.default_rng(0)
    w, h = 91631, 70411  # real slide dims: non-multiple of 512, ragged last row/col
    mask = (rng.random((384, 500)) > 0.4).astype(float)
    gi, gj, gx, gy, cov = L.grid_coverage(mask, w, h)
    ref = _reference_loop(mask, w, h)
    assert len(ref) == gi.size == (-(-w // 512)) * (-(-h // 512))
    assert all(ref[(x, y)] == c for x, y, c in zip(gx, gy, cov))
    assert np.array_equal(gx, gj * 512) and np.array_equal(gy, gi * 512)


def test_grid_coverage_thresholds_exact():
    w = h = 512 * 10
    mask = np.zeros((100, 100))
    mask[:10, :10] = 1.0          # cell (0,0) fully covered
    mask[10:20, :7] = 1.0         # cell (1,0) 70% covered
    mask[20:30, :6] = 1.0         # cell (2,0) 60% covered
    gi, gj, gx, gy, cov = L.grid_coverage(mask, w, h)
    c = {(i, j): v for i, j, v in zip(gi, gj, cov)}
    assert c[(0, 0)] == 1.0 and c[(1, 0)] == 0.7 and c[(2, 0)] == 0.6
    elig = cov >= L.ELIGIBLE_THRESHOLD
    assert set(zip(gi[elig], gj[elig])) == {(0, 0), (1, 0)}


def test_containment_and_duplicates():
    sup = L.coord_keys([[0, 0], [512, 0], [0, 512], [512, 512]])
    sub = L.coord_keys([[512, 0], [1024, 0]])
    assert list(L.containment(sub, sup)) == [1, 0]
    dup = L.coord_keys([[0, 0], [0, 0], [512, 0]])
    assert L.n_duplicates(dup) == 1
    assert list(L.containment(L.coord_keys([[0, 0]]), dup)) == [2]
    # x/y must not be confused
    assert L.coord_keys([[512, 0]])[0] != L.coord_keys([[0, 512]])[0]
    assert L.set_diff_counts(sup, sub) == (3, 1)


def test_row_compare_barcode_and_exact():
    a = np.array([[1.0, 2.0, 1.5, 3.0]], np.float32)
    b = a.copy()
    cos, mad, agr, ex = L.row_compare(a, b)
    assert ex[0] and mad[0] == 0 and agr[0] == 1.0 and abs(cos[0] - 1) < 1e-12
    b[0, 2] = 2.5  # flips only the 2nd diff sign
    cos, mad, agr, ex = L.row_compare(a, b)
    assert not ex[0] and mad[0] == 1.0 and agr[0] == 2 / 3


def test_generator_list_eligibility_overrides_threshold():
    import stage_a_slides as S

    gkeys = L.coord_keys([[0, 0], [512, 0], [1024, 0], [1536, 0]])
    cov = np.array([1.0, 0.6, 1.0, 0.8])
    ckeys = L.coord_keys([[0, 0], [512, 0], [1536, 0]])  # listed at recon 0.6; recon-full cell 2 not listed
    elig, recon, full = S.eligibility_flags(gkeys, cov, ckeys, "generator-list")
    assert elig.tolist() == [True, True, False, True]
    assert recon.tolist() == [True, False, True, True]
    assert full.tolist() == [True, False, False, False]  # full is restricted to eligible cells
    d_elig, d_recon, d_full = S.eligibility_flags(gkeys, cov, ckeys, "recon-threshold")
    assert d_elig.tolist() == d_recon.tolist() == [True, False, True, True]
    assert d_full.tolist() == [True, False, True, False]


import stage_b_lib as B  # noqa: E402


def test_stage_b_match_rows_maps_and_rejects():
    ref = np.array([[0, 0], [512, 0], [0, 512], [1024, 1536]])
    q = np.array([[1024, 1536], [0, 0], [0, 512]])
    assert B.match_rows(ref, q).tolist() == [3, 0, 2]
    import pytest
    with pytest.raises(KeyError):
        B.match_rows(ref, np.array([[512, 512]]))
    with pytest.raises(ValueError):
        B.match_rows(ref, np.array([[0, 0], [0, 0]]))
    # x/y must not be interchangeable
    with pytest.raises(KeyError):
        B.match_rows(np.array([[512, 0]]), np.array([[0, 512]]))


def test_stage_b_gap_mask_d1():
    import pandas as pd
    df = pd.DataFrame({"in_cluster_locs": [1, 1, 1, 0, 1],
                       "in_mosaic": [0, 1, 0, 0, 0],
                       "in_full_bank": [0, 0, 1, 0, 0]})
    assert B.gap_mask(df).tolist() == [True, False, False, False, True]


def test_stage_b_coords_roundtrip_exact(tmp_path):
    xy = np.array([[49152, 1536], [512, 99840]])
    cov = np.array([0.33333333333333331, 1.0])
    p = tmp_path / "s.coords.csv.gz"
    B.write_coords(p, xy, cov)
    rxy, rcov = B.read_coords(p)
    assert np.array_equal(rxy, xy) and np.array_equal(rcov, cov)


def test_stage_b_pack_tasks_covers_each_slide_once():
    counts = {"1": 9000, "2": 3000, "3": 2500, "4": 2400, "5": 100, "6": 0}
    tasks = B.pack_tasks(counts, 5000)
    flat = [s for t in tasks for s in t]
    assert sorted(flat) == ["1", "2", "3", "4", "5"]
    assert ["1"] in tasks
    assert all(sum(counts[s] for s in t) <= 5000 for t in tasks if len(t) > 1)


def _naive_min_median(a, b):
    m = np.array([min(int(np.sum(x != y)) for y in b) for x in a], dtype=np.float32)
    return float(np.median(m))


def test_stage_c_gpu_kernel_matches_naive_oracle_both_directions():
    import torch
    import stage_c_lib as CL
    import stage_c_yottixel as Y
    rng = np.random.default_rng(3)
    q = (rng.random((9, CL.NBITS)) > 0.5).astype(np.uint8)
    q[4] = q[1]                                      # duplicate query tiles
    lib = []
    for n in (1, 2, 5, 8, 13):                       # odd and even counts
        r = (rng.random((n, CL.NBITS)) > 0.5).astype(np.uint8)
        r[0] = q[0]                                  # zero distance
        if n > 2:
            r[1] = r[2]                              # duplicated reference rows (ties)
        lib.append(np.packbits(r, axis=1))
    for budget in (1 << 28, 9, 10, 27):              # block boundaries at 1, 1 and 3 ref rows
        fwd, rev, _ = Y.gpu_rows(q, lib, torch, budget=budget, dev="cpu")
        for p, f, r in zip(lib, fwd, rev):
            ub = CL.unpack_barcodes(p)
            assert f == _naive_min_median(q, ub)
            assert r == _naive_min_median(ub, q)


def test_stage_c_barcode_pack_roundtrip_and_production_bits():
    import stage_c_lib as CL
    _, bob, _ = CL.production()
    feats = np.random.default_rng(1).standard_normal((37, CL.DIM)).astype(np.float32)
    bits = bob.features_to_barcodes(feats)
    packed = CL.pack_barcodes(feats, bob)
    assert packed.shape == (37, 192) and np.array_equal(CL.unpack_barcodes(packed), bits)


def test_stage_c_ot_status_never_sentinel():
    import stage_c_ot as O
    rng = np.random.default_rng(2)
    fa = rng.standard_normal((40, 16)).astype(np.float32)
    fb = rng.standard_normal((55, 16)).astype(np.float32)
    ok = O.solve(fa, fb, 1e6)
    assert ok["status"] == "accepted" and ok["result_code"] == 1 and 0 < ok["cost"] < 2
    capped = O.solve(fa, fb, 3)
    assert capped["result_code"] == 3 and capped["status"] == "failed_max_iter_reached"
    assert O.code_from_warnings([]) == (1, "")


def test_stage_c_partial_match():
    import stage_c_lib as CL
    ref = np.array([[0, 0], [512, 0], [0, 512]])
    qry = np.array([[0, 512], [1024, 0], [0, 0]])
    qpos, rpos = CL.partial_match(ref, qry)
    assert qpos.tolist() == [0, 2] and rpos.tolist() == [2, 0]


def test_stage_c_hit5_tie_bounds():
    import stage_c_collect as CC
    d = [1, 2, 3, 4, 4, 4, 9]                       # 3 strictly inside, 3 tied for the last 2 places
    f = lambda c: CC.hit5_bounds(d, np.array(c, bool))  # noqa: E731
    assert f([0, 0, 0, 0, 1, 0, 0]) == (False, True)   # a tie may or may not admit the hit
    assert f([0, 0, 0, 0, 1, 1, 0]) == (True, True)    # only 1 incorrect tied < 2 places
    assert f([1, 0, 0, 0, 0, 0, 0]) == (True, True)
    assert f([0, 0, 0, 0, 0, 0, 1]) == (False, False)


import stage_e_eval as E  # noqa: E402


def test_stage_e_full_row_bounds_match_permutation_oracle():
    import itertools
    rng = np.random.default_rng(5)
    n = 9
    for _ in range(10):
        arr = rng.integers(0, 3, size=(n, n)).astype(float)
        labels = rng.choice(["a", "b"], size=n)
        patients = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7])
        got = E.full_row_bounds(arr, labels, patients, k=3)
        for i in range(n):
            keep = np.flatnonzero(patients != patients[i])
            hits, precs = set(), set()
            for perm in itertools.permutations(keep):
                order = sorted(perm, key=lambda j: arr[i, j])  # stable: every tie order reachable
                top = labels[order[:3]] == labels[i]
                hits.add(int(top.any()))
                precs.add(top.sum() / 3)
            assert (got.hit_lo[i], got.hit_hi[i]) == (min(hits), max(hits))
            assert np.isclose(got.prec_lo[i], min(precs)) and np.isclose(got.prec_hi[i], max(precs))
