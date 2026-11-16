"""Tests for scripts/r3_5_l2_alignment.py (specification 004 F5; synthetic fixtures, no production input read or written).

F5 (specification 7): a synthetic 2-class plan with known composition gives S*, S0, NA and R_c computed by hand; a
synthetic reference with known same/other costs gives Delta_i and AUROC_c.  Further kernels: plan orientation
(i = query, j = reference), the class decomposition of D, unconstrained NN matching, the leave-one-patient-out
k-NN, the vote/confusion/balanced-accuracy bootstrap and the mechanical wording rules.
"""
import importlib.util
import os

import numpy as np
import pytest

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(R, "scripts", "r3_5_l2_alignment.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load(SCRIPT, "r3_5_l2_alignment")


@pytest.fixture(scope="module")
def LC():
    return M.L1.import_prosper()[0]


def two_class_plan():
    """Reference rows: classes 1,1,1,2 (+ one class-1 row reached only by an unannotated query row).
    Query rows: 1,1,2,2 and one unannotated (0).  Annotated links carry 0.8 of the mass, M (ref x query) after
    renormalisation = [[0.5, 0.2], [0.1, 0.2]]."""
    cr = np.array([1, 1, 1, 2, 1])
    cq = np.array([1, 1, 2, 2, 0])
    i = np.array([0, 2, 1, 3, 4])          # query rows
    j = np.array([0, 1, 3, 3, 4])          # reference rows
    w = np.array([0.40, 0.16, 0.08, 0.16, 0.20])
    return i, j, w, cq, cr


def test_f5_two_class_composition_by_hand():
    i, j, w, cq, cr = two_class_plan()
    Mx, mass = M.restricted_matrix(i, j, w, cq, cr, (1, 2))
    assert mass == pytest.approx(0.8)
    assert np.allclose(Mx, [[0.5, 0.2], [0.1, 0.2]])
    al = M.alignment(Mx)
    # by hand: p = (0.7, 0.3), q = (0.6, 0.4); S = 0.7; S0 = 0.42 + 0.12 = 0.54; S* = 0.6 + 0.3 = 0.9
    assert np.allclose(al["p"], [0.7, 0.3]) and np.allclose(al["q"], [0.6, 0.4])
    assert al["S"] == pytest.approx(0.7)
    assert al["S0"] == pytest.approx(0.54)
    assert al["Sstar"] == pytest.approx(0.9)
    assert al["NA"] == pytest.approx(0.16 / 0.36)
    assert al["kappa"] == pytest.approx(0.16 / 0.46)
    assert not al["no_headroom"]
    ret = M.retention(Mx, al["p"], al["q"], n_ref_rows=[4, 1], min_rows=1)
    assert ret[0]["R"] == pytest.approx(0.5 / 0.7) and ret[0]["chance"] == pytest.approx(0.6)
    assert ret[0]["ceiling"] == pytest.approx(0.6 / 0.7)
    assert ret[1]["R"] == pytest.approx(0.2 / 0.3) and ret[1]["chance"] == pytest.approx(0.4)
    assert ret[1]["ceiling"] == pytest.approx(1.0)
    # the >= 20-row rule: class 2 has one reference row -> not reported at the production threshold
    assert M.retention(Mx, al["p"], al["q"], n_ref_rows=[25, 1])[1] is None


def test_restricted_matrix_orientation_rows_are_reference():
    # one link: query row 0 (class 2) <- reference row 0 (class 1): M[ref=1, query=2]
    Mx, _ = M.restricted_matrix(np.array([0]), np.array([0]), np.array([1.0]), np.array([2]), np.array([1]), (1, 2))
    assert Mx[0, 1] == 1.0 and Mx.sum() == 1.0


def test_no_headroom_and_kappa_guard():
    al = M.alignment(np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert al["no_headroom"] and np.isnan(al["NA"]) and np.isnan(al["kappa"])
    al = M.alignment_from(0.7, [0.7, 0.3], [0.69, 0.31])        # S* - S0 = 0.99 - 0.576 >= 0.02
    assert not al["no_headroom"]
    al = M.alignment_from(0.9, [0.9, 0.1], [0.95, 0.05])        # S* - S0 = 0.95 - 0.86 = 0.09
    assert al["headroom"] == pytest.approx(0.09)
    al = M.alignment_from(0.98, [0.99, 0.01], [0.995, 0.005])   # headroom 0.0099 < 0.02
    assert al["no_headroom"] and np.isnan(al["NA"])


def test_no_annotated_links():
    Mx, mass = M.restricted_matrix(np.array([0]), np.array([0]), np.array([1.0]), np.array([0]), np.array([1]), (1, 2))
    assert mass == 0.0 and np.isnan(Mx).all()
    assert M.alignment(Mx)["no_headroom"]


def test_f5_discriminative_values_and_auroc_c(LC):
    same = [[0.1, 0.5, 0.3], [0.3, 0.5, 0.1]]       # mean (0.2, 0.5, 0.2)
    other = [[0.6, 0.5, 0.2], [0.8, 0.5, 0.4]]      # mean (0.7, 0.5, 0.3)
    assert np.allclose(M.discriminative_values(same, other), [0.5, 0.0, 0.1])
    # lower class cost = same group; pairs (s < o): 0.1 -> 3, 0.2 -> 3, 0.3 -> 2  => 8/9
    assert M.auroc_same_vs_other([0.1, 0.2, 0.3], [0.4, 0.25, 0.5], LC) == pytest.approx(8 / 9)
    assert M.auroc_same_vs_other([0.2], [0.2], LC) == pytest.approx(0.5)
    assert M.auroc_same_vs_other([0.1, 0.2], [0.3, 0.4], LC) == pytest.approx(1.0)
    assert M.auroc_same_vs_other([0.5, 0.6], [0.3, 0.4], LC) == pytest.approx(0.0)
    assert np.isnan(M.auroc_same_vs_other([], [0.3], LC))


def test_class_decomposition():
    out = M.class_decomposition(np.array([1, 1, 2]), np.array([0.25, 0.25, 0.5]), np.array([0.2, 0.4, 0.1]))
    assert out["D"] == pytest.approx(0.2)
    assert out["D_c1"] == pytest.approx(0.15) and out["k_c1"] == pytest.approx(0.3) and out["s_c1"] == pytest.approx(0.75)
    assert out["D_nt"] == pytest.approx(0.05) and out["k_nt"] == pytest.approx(0.1)
    assert out["mass_c0"] == 0 and np.isnan(out["k_c0"]) and out["s_c0"] == 0
    assert sum(out[f"s_c{c}"] for c in range(6)) == pytest.approx(1.0)


def test_tile_meancost_is_reference_side():
    """L1 b_tile_values (reused) aggregates over axis 0 = query rows, i.e. one value per reference column."""
    rng = np.random.default_rng(0)
    T = rng.random((4, 3))
    T /= T.sum()
    C = rng.random((4, 3)).astype(np.float32)
    val, linked = M.L1.b_tile_values(T, C, "meancost")
    assert val.shape == (3,) and linked.all()
    assert np.allclose(val, (T * C).sum(0) / T.sum(0))


def test_nn_match_and_count_alignment():
    Fr = np.array([[1, 0], [0, 1], [1, 0.1]], float)
    Fq = np.array([[0, 2], [3, 0]], float)
    nn = M.nn_match(Fr, Fq)
    assert nn.tolist() == [1, 0, 1]
    cref = np.array([1, 2, 1])
    cqa = np.array([2, 1])
    al = M.count_alignment(cref, cqa[nn], cqa, (1, 2))
    # matched classes 1, 2, 1 -> S = 1; p = (2/3, 1/3), q = (1/2, 1/2); S0 = 1/2; S* = 1/2 + 1/3
    assert al["S"] == pytest.approx(1.0)
    assert al["S0"] == pytest.approx(0.5)
    assert al["Sstar"] == pytest.approx(0.5 + 1 / 3)
    assert al["NA"] == pytest.approx(0.5 / (1 / 3))


def test_knn_lopo_excludes_same_patient():
    X = np.array([[1, 0], [1, 0.01], [0, 1], [0.01, 1], [1, 0.02]], np.float32)
    g = np.array(["a", "a", "b", "b", "c"])
    nb = M.knn_lopo(X, g, k=1)
    for r in range(len(X)):
        assert g[nb[r, 0]] != g[r]
    assert nb[0, 0] == 4 and nb[4, 0] == 1


def test_vote_confusion_balanced_accuracy_bootstrap():
    nl = np.array([[1, 1, 2], [2, 2, 1], [1, 2, 3]])
    pred, tied = M.majority_vote(nl, (1, 2, 3))
    assert pred.tolist() == [1, 2, 1] and tied.tolist() == [False, False, True]
    C = M.confusion(np.array([1, 2, 2]), pred, (1, 2, 3))
    assert C.tolist() == [[1, 0, 0], [1, 1, 0], [0, 0, 0]]
    assert M.balanced_accuracy(C) == pytest.approx((1 + 0.5) / 2)       # class 3 has no true row
    Cp = np.stack([np.array([[2, 0], [0, 0]]), np.array([[0, 0], [1, 1]])])
    mult = np.array([[1, 0], [0, 1], [1, 1]])
    ba, _ = M.ba_bootstrap(Cp, mult)
    assert ba[0] == pytest.approx(1.0) and ba[1] == pytest.approx(0.5)
    assert ba[2] == pytest.approx(M.balanced_accuracy(Cp.sum(0)))


def test_wording_rules():
    assert M.wording_map(0.71) == "localises annotated tumour"
    assert M.wording_map(0.70) == "modestly localises"
    assert M.wording_map(0.50) == "does not localise"
    assert M.wording_map(None) == "does not localise"
    assert "carried by tumour" in M.wording_class(0.01, 0.2)
    assert "not specific" in M.wording_class(-0.01, 0.2)
    assert "non-tumour" in M.wording_class(-0.3, -0.01)
    assert M.wording_d1(0.001).startswith("matched regions share")
    assert "do not" in M.wording_d1(0.0)
    assert M.wording_d3(0.80, 0.9, 0.01).startswith("transport")
    assert M.wording_d3(0.80, 0.9, 0.0).startswith("in between")
    assert M.wording_d3(0.5, 0.64, 0.5).startswith("embedding")
    assert M.wording_d3(0.6, 0.66, 0.5).startswith("in between")


def test_class_mapping_is_l1s():
    lab = np.array([0, 1, 2, 3, 4, 5])
    assert M.L1.class_of(lab, True).tolist() == [0, 1, 2, 2, 2, 2]
    assert M.L1.class_of(lab, False).tolist() == lab.tolist()


# ------------------------------------------------------------ amendment 001 (A1 tile-count marginals, A2 probe)
def test_f5_two_class_tile_count_marginals_by_hand():
    """A1: same plan, baseline/ceiling from annotated tile counts: reference rows 1,1,1,2,1 -> p = (0.8, 0.2);
    query rows 1,1,2,2,(0) -> q = (0.5, 0.5).  S = 0.7 (plan mass), S0 = 0.4 + 0.1 = 0.5, S* = 0.5 + 0.2 = 0.7."""
    i, j, w, cq, cr = two_class_plan()
    Mx, _ = M.restricted_matrix(i, j, w, cq, cr, (1, 2))
    p, q = M.count_marginals(cr, (1, 2)), M.count_marginals(cq, (1, 2))
    assert np.allclose(p, [0.8, 0.2]) and np.allclose(q, [0.5, 0.5])
    al = M.alignment_counts(Mx, p, q)
    assert al["S"] == pytest.approx(0.7) and al["S0"] == pytest.approx(0.5) and al["Sstar"] == pytest.approx(0.7)
    assert al["NA"] == pytest.approx(1.0) and al["kappa"] == pytest.approx(0.4)
    ret = M.retention(Mx, p, q, n_ref_rows=[4, 1], min_rows=1)
    assert ret[0]["R"] == pytest.approx(0.625) and ret[0]["chance"] == pytest.approx(0.5)
    assert ret[0]["ceiling"] == pytest.approx(0.625)
    assert ret[1]["R"] == pytest.approx(1.0) and ret[1]["ceiling"] == pytest.approx(1.0)


def test_tile_count_kappa_equals_l1_cross_slide_agreement():
    """With A1 marginals, kappa equals L1's (agreement - chance)/(1 - chance) on a dense plan (F2 identity)."""
    i, j, w, cq, cr = two_class_plan()
    T = np.zeros((5, 5))
    T[i, j] = w
    ag = M.L1.cross_slide_agreement(T, cq, cr, binary=True)
    Mx, _ = M.restricted_matrix(i, j, w, cq, cr, (1, 2))
    al = M.alignment_counts(Mx, M.count_marginals(cr, (1, 2)), M.count_marginals(cq, (1, 2)))
    assert al["kappa"] == pytest.approx((ag["agreement"] - ag["chance"]) / (1 - ag["chance"]), abs=1e-12)


def test_mixed_slide_balanced_accuracy():
    yt = np.r_[[1] * 20, [2] * 20, [1] * 30, [1] * 20, [2] * 25, [3] * 5]
    pred = np.r_[[1] * 20, [2] * 10, [1] * 10, [1] * 30, [2] * 20, [2] * 25, [1] * 5]
    sl = np.r_[["A"] * 40, ["B"] * 30, ["C"] * 50]
    pt = np.r_[["p1"] * 40, ["p2"] * 30, ["p1"] * 50]
    ms = M.mixed_slide_ba(yt, pred, sl, pt, (1, 2, 3))
    # A: (1 + 0.5)/2; B single-class -> excluded; C: class 3 has 5 rows -> excluded, (0 + 1)/2
    assert ms.slide_id.tolist() == ["A", "C"]
    assert ms.balanced_accuracy.tolist() == pytest.approx([0.75, 0.5])
    e = M.mixed_slide_estimate(ms, count=50, seed=1)
    assert e["point"] == pytest.approx(0.625) and e["n_patients"] == 1 and e["chance_per_slide"] == pytest.approx(0.5)
