"""Tests for scripts/r3_5_l1_maps.py (synthetic fixtures; no production input is read or written).

Kernels that mirror prosper-search code are checked against the imported originals
(localization_cli.Slide.heatmap / evaluate_pair, ot_visualization.sparsify_transport_matrix /
cluster_patches_from_transport) and the bootstrap draws against scripts/r2_5_c4_whole_patient_ci.py.
"""
import importlib.util
import os
import types

import numpy as np
import pandas as pd
import pytest

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(R, "scripts", "r3_5_l1_maps.py")
C4CI = os.path.join(R, "scripts", "r2_5_c4_whole_patient_ci.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load(SCRIPT, "r3_5_l1_maps")


@pytest.fixture(scope="module")
def prosper():
    return M.import_prosper()


def toy_plan(nq=7, nr=9, seed=0):
    import ot
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(nq, 6)).astype(np.float32)
    Y = rng.normal(size=(nr, 6)).astype(np.float32)
    Mc = ot.dist(X, Y, metric="cosine")
    T = ot.emd(np.ones(nq) / nq, np.ones(nr) / nr, Mc)
    return X, Y, np.asarray(T, float), np.asarray(Mc, float)


def fake_slide(n_tiles=12, dim=5, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_tiles, dim)).astype(np.float32)
    owners = [np.array(sorted(rng.choice(n_tiles, size=rng.integers(1, 4), replace=False))) for _ in range(30)]
    labels = rng.choice([0, 1, 2, 3, 4, 5], size=30).astype(np.int8)
    labels[:5] = 1
    labels[5:10] = 2
    return types.SimpleNamespace(X=X, _owners=owners, patch_label=labels)


# ------------------------------------------------------------ maps
def test_roi_l1_mean_equals_imported_heatmap(prosper):
    LC, _ = prosper
    sl = fake_slide()
    q = np.random.default_rng(3).normal(size=5).astype(np.float32)
    assert np.array_equal(M.roi_map(sl, q, "l1", "mean"), LC.Slide.heatmap(sl, q))


def test_score_continuous_equals_evaluate_pair(prosper):
    LC, _ = prosper
    sl = fake_slide()
    sl.heatmap = types.MethodType(LC.Slide.heatmap, sl)
    q = np.random.default_rng(4).normal(size=5).astype(np.float32)
    ref = LC.evaluate_pair(sl, q, min_neg=1)
    mine = M.score_continuous(M.roi_map(sl, q, "l1", "mean"), sl.patch_label, LC, min_neg=1)
    for k in ("auroc_A", "ap_A", "pointing_A", "tumour_fraction_A", "n_annotated"):
        assert mine[k] == ref[k]


def test_other_distances_and_min_aggregation():
    sl = fake_slide()
    q = sl.X[0]
    for dist in ("l2", "cosine"):
        h = M.roi_map(sl, q, dist, "min")
        assert h.shape == (30,) and h.min() >= 0 and h.max() <= 1
    d = M.roi_distances(sl.X, q, "cosine")
    assert abs(d[0]) < 1e-6
    v = M.project(np.arange(12, dtype=float), [np.array([1, 5]), np.array([3])], "min")
    assert list(v) == [1.0, 3.0]


# ------------------------------------------------------------ sparsification / clustering
def test_threshold_generalisation_equals_imported_at_090(prosper):
    _, OV = prosper
    _, _, T, _ = toy_plan()
    assert np.array_equal(M.threshold_sparsify(T, 0.90), OV.sparsify_transport_matrix(T, method="threshold"))


@pytest.mark.parametrize("method,res", [("connected_components", 1.0), ("louvain", 1.0), ("louvain", 0.5)])
def test_cluster_matches_imported_cluster_patches(prosper, method, res):
    _, OV = prosper
    _, _, T, _ = toy_plan(15, 20, seed=2)
    da, db = OV.cluster_patches_from_transport(T, method=method, resolution=res, sparsify_k=3)
    Tsp = M.sparsify(T, "mutual_topk", 3, OV)
    cq, cr = M.cluster_sparse(Tsp, method, res)
    assert np.array_equal(cq, da.sort_values("patch_idx").normalized_score.to_numpy())
    assert np.array_equal(cr, db.sort_values("patch_idx").normalized_score.to_numpy())
    gq, gr = M.cluster_graph(M.build_graph(Tsp), *T.shape, method, res)
    assert np.array_equal(cq, gq) and np.array_equal(cr, gr)


def test_grid_sizes():
    # amendment 003: 19 sparsifications x (connected components + 19 Louvain resolutions)
    assert len(M.grid_a()) == 6 and len(M.grid_b()) == 14 and len(M.grid_c()) == 380
    assert len({g["setting"] for g in M.grid_c()}) == 380 and len(M.LOUVAIN_RES) == 19
    assert set(M.LOUVAIN_RES_001) <= set(M.LOUVAIN_RES) and 4e-5 in M.LOUVAIN_RES
    s = {g["setting"] for g in M.grid_a()} | {g["setting"] for g in M.grid_b()} | {g["setting"] for g in M.grid_c()}
    assert set(M.SUBMITTED.values()) <= s


def test_same_partition():
    assert M.same_partition(np.array([1, 1, 2]), np.array([5, 5, 3]))
    assert not M.same_partition(np.array([1, 1, 2]), np.array([5, 3, 3]))


# ------------------------------------------------------------ family B
def test_b_tile_values():
    T = np.array([[0.2, 0.0, 0.0], [0.1, 0.2, 0.0]])
    C = np.array([[0.5, 0.9, 0.9], [0.2, 0.4, 0.9]])
    v, linked = M.b_tile_values(T, C, "meancost")
    assert list(linked) == [True, True, False]
    assert np.isclose(v[0], (0.2 * 0.5 + 0.1 * 0.2) / 0.3) and np.isclose(v[1], 0.4)
    assert np.isclose(v[2], max(v[0], v[1]) + 1.0)
    v, _ = M.b_tile_values(T, C, "mincost")
    assert np.isclose(v[0], 0.2) and np.isclose(v[1], 0.4) and np.isclose(v[2], 1.4)


def test_b_meancost_none_ranks_as_shipped_weighted_distance(prosper):
    _, OV = prosper
    _, _, T, C = toy_plan(8, 11, seed=5)
    v, _ = M.b_tile_values(T, C, "meancost")
    _, DB = OV.compute_per_patch_weighted_distance_scores(T, C)
    assert np.allclose(v, DB * 11)


# ------------------------------------------------------------ family C metrics
def test_cross_slide_agreement_and_chance():
    T = np.array([[0.3, 0.0], [0.0, 0.2], [0.1, 0.4]])
    lq = np.array([1, 2, 0])
    lr = np.array([1, 4])
    r = M.cross_slide_agreement(T, lq, lr, binary=True)
    assert np.isclose(r["agreement"], 1.0) and np.isclose(r["annotated_mass"], 0.5)
    assert np.isclose(r["chance"], 0.5 * 0.5 + 0.5 * 0.5)
    r5 = M.cross_slide_agreement(T, lq, lr, binary=False)
    assert np.isclose(r5["agreement"], 0.6)


def test_cluster_stats():
    T = np.array([[0.3, 0.0], [0.0, 0.2], [0.0, 0.1]])
    cq, cr = np.array([1, 2, 2]), np.array([1, 2])
    lq, lr = np.array([1, 2, 1]), np.array([1, 3])
    s = M.cluster_stats(T, cq, cr, lq, lr, binary=True)
    # cluster 1: tiles {q0:T, r0:T} purity 1, mass .3; cluster 2: {q1:NT, q2:T, r1:NT} purity 2/3, mass .3
    assert np.isclose(s["purity"], (0.3 * 1 + 0.3 * 2 / 3) / 0.6)
    assert s["n_clusters"] == 2 and s["frac_singleton_clusters"] == 0.0


# ------------------------------------------------------------ pairs
def _toy_population(n=30, seed=7):
    rng = np.random.default_rng(seed)
    ids = [str(100 + i) for i in range(n)]
    pats = [f"p{i // 2}" for i in range(n)]
    groups = [["A", "B", "C"][i % 3] for i in range(n)]
    D = rng.random((n, n))
    D = (D + D.T) / 2
    for i in range(n):
        for j in range(n):
            if pats[i] == pats[j]:
                D[i, j] = 1e6
    mat = pd.DataFrame(D, index=ids, columns=ids)
    pop = pd.DataFrame({"slide_id": ids, "patient": pats, "group": groups})
    return mat, pop


def test_population_topk_equals_independent_route():
    mat, pop = _toy_population()
    tk = M.population_topk(mat.values, list(mat.index), pop.patient.values, 10)
    pp = dict(zip(pop.slide_id, pop.patient))
    for q in mat.index:
        assert [n for n, _ in tk[q]] == M.independent_topk(mat, q, pp, 10)


def test_pairs_p1():
    mat, pop = _toy_population()
    tk = M.population_topk(mat.values, list(mat.index), pop.patient.values, 10)
    refs = ["100", "105", "111"]
    p1 = M.pairs_p1(refs, pop[pop.slide_id.isin(refs)], pop, tk)
    assert set(p1.reference) <= set(refs)
    assert (p1.ref_patient != p1.query_patient).all()
    for _, r in p1.iterrows():
        assert r.reference in [n for n, _ in tk[r["query"]]]
        assert (r.arm == "same_group") == (r.ref_group == r.query_group)
    expected = sum(r in [n for n, _ in tk[q]] for q in mat.index for r in refs)
    assert len(p1) == expected


def test_draw_rois_deterministic_without_replacement():
    lab = {"1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "2": []}
    pred = {"1": [3, 4], "2": list(range(20))}
    a = M.draw_rois(["2", "1", "1"], lab, pred)
    b = M.draw_rois(["1", "2"], lab, pred)
    assert a == b
    assert len(a["1"]["labelled"]) == 8 and len(set(a["1"]["labelled"])) == 8
    assert sorted(a["1"]["predicted"]) == [3, 4] and "labelled" not in a["2"]


def test_c4_draw_matches_localization_cli_loop():
    """Re-run the literal loop body of localization_cli.run on toy data and compare."""
    rng0 = np.random.default_rng(11)
    order = [str(s) for s in range(20, 32)]
    group = {s: ["RB", "NB", "NORMAL"][i % 3] for i, s in enumerate(order)}
    patient = {s: f"p{i // 2}" for i, s in enumerate(order)}
    tt = {s: (np.arange(rng0.integers(0, 12)) if group[s] != "NORMAL" else np.arange(0)) for s in order}
    ntum = {s: int(rng0.integers(0, 40)) for s in order}
    got = M.c4_draw(order, tt, ntum, group, patient)
    rng = np.random.default_rng(123)
    pool = {s: tt[s] for s in order}
    refs = [s for s in order if ntum[s] >= 20]
    rows = []
    for r in refs:
        g, pt = group[r], patient[r]
        same = [s for s in order if s != r and group[s] == g and patient[s] != pt and len(pool[s])]
        other = [s for s in order if group[s] != g and group[s] != "NORMAL" and len(pool[s])]
        for arm, cand in (("same_group", same), ("other_group", other)):
            if not cand:
                continue
            for _ in range(10):
                qs = cand[rng.integers(len(cand))]
                qidx = rng.choice(pool[qs], size=min(8, len(pool[qs])), replace=False)
                rows.append((r, arm, qs, ";".join(map(str, qidx.tolist()))))
    assert list(zip(got.reference, got.arm, got["query"], got.query_tiles)) == rows


def test_split_patients():
    df = pd.DataFrame({"patient": [f"p{i}" for i in range(11)], "group": list("AAAABBBCCCD")})
    s1 = M.split_patients(df)
    s2 = M.split_patients(df.sample(frac=1, random_state=3))
    assert s1.sort_values("patient").equals(s2.sort_values("patient")) or \
        s1.set_index("patient").half.sort_index().equals(s2.set_index("patient").half.sort_index())
    assert s1.patient.is_unique and set(s1.half) == {"development", "report"}
    for g, x in s1.groupby("group"):
        assert abs((x.half == "development").sum() - (x.half == "report").sum()) <= 1
    assert abs((s1.half == "development").sum() - (s1.half == "report").sum()) <= 1


# ------------------------------------------------------------ bootstrap
def test_patient_draws_match_c4_whole_patient_script():
    ref = _load(C4CI, "c4ci")
    ids = ["b", "a", "c", "a", "d"]
    p1, m1 = M.make_patient_draws(ids, 200, 123)
    p2, m2 = ref.make_patient_draws(ids, 200, 123)
    assert p1 == p2 and np.array_equal(m1, m2)
    v = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    reps = M.replicate_means(v, ids, p1, m1)
    W = m1[:, pd.Index(p1).get_indexer(ids)]
    assert np.allclose(reps, (W @ v) / W.sum(1))


# ------------------------------------------------------------ representative links
def test_representative_links():
    X, Y, T, _ = toy_plan(10, 12, seed=9)
    cq, cr = M.cluster_sparse(T, "connected_components", None)
    links = M.representative_links(T, cq, cr, X, Y)
    kinds = {(l["cluster"], l["kind"]) for l in links}
    for c in set(cq) | set(cr):
        assert (c, "medoid") in kinds
    for l in links:
        assert T[l["i"], l["j"]] > 0 and cq[l["i"]] == l["cluster"] == cr[l["j"]]
    hi = [l for l in links if l["kind"] == "highest_mass"][0]
    lo = [l for l in links if l["kind"] == "lowest_mass"][0]
    assert hi["mass"] >= lo["mass"]


def test_load_plan_roundtrip(tmp_path):
    _, _, T, C = toy_plan(6, 8, seed=12)
    ii, jj = np.nonzero(T)
    np.savez_compressed(tmp_path / "p.npz", i=ii.astype(np.int32), j=jj.astype(np.int32), t=T[ii, jj],
                        m=C[ii, jj].astype(np.float32), shape=np.array(T.shape), distance=np.array([0.0]))
    assert np.array_equal(M.load_plan(tmp_path / "p.npz"), T)


def test_score_c_records_nontermination(prosper, monkeypatch):
    import time as _t
    _, OV = prosper
    _, _, T, _ = toy_plan(6, 8, seed=13)
    lq, lr = np.array([1, 2, 1, 0, 2, 1]), np.array([1, 1, 2, 2, 0, 1, 3, 4])
    real = M.cluster_graph

    def slow(G, n_a, n_b, method, res):
        if method == "louvain" and res == 0.5:
            _t.sleep(5)
        return real(G, n_a, n_b, method, res)
    monkeypatch.setattr(M, "c_sparsifications", lambda: [("none", None), ("mutual_topk", 2)])
    monkeypatch.setattr(M, "cluster_graph", slow)
    rows = M.score_c(T, lq, lr, OV, timeout_s=1)
    bad = [r for r in rows if not r["terminated"]]
    assert len(rows) == 2 * 20 and len(bad) == 2
    assert all(r["setting"].endswith("|louvain|0.5") and r["degenerate_pair"] and np.isnan(r["purity_bin"]) for r in bad)
    ok = [r for r in rows if r["terminated"]]
    assert all(np.isfinite(r["n_clusters"]) for r in ok)


def test_load_plan_with_costs_gives_identical_b(prosper, tmp_path):
    LC, OV = prosper
    _, _, T, C = toy_plan(7, 9, seed=21)
    C = C.astype(np.float32)
    ii, jj = np.nonzero(T)
    np.savez_compressed(tmp_path / "p.npz", i=ii.astype(np.int32), j=jj.astype(np.int32), t=T[ii, jj],
                        m=C[ii, jj], shape=np.array(T.shape), distance=np.array([0.0]))
    T2, C2 = M.load_plan_with_costs(tmp_path / "p.npz")
    for stat in ("meancost", "mincost"):
        for sp, k in M.b_sparsifications():
            a = M.b_tile_values(M.sparsify(T, sp, k, OV), C, stat)[0]
            b = M.b_tile_values(M.sparsify(T2, sp, k, OV), C2, stat)[0]
            assert np.array_equal(a, b)


def test_pinned_dispatch_guard():
    from numpy._core._multiarray_umath import __cpu_features__ as cf
    if cf.get("AVX512_SKX"):
        with pytest.raises(RuntimeError):
            M.assert_pinned_dispatch()
    else:
        assert "avx2" in M.assert_pinned_dispatch()
