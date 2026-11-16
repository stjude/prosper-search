"""Tests for scripts/r3_5_l4_link_cost_concordance.py (specification 006 F5; toy links only, no production input read).

F5: link typing (unannotated excluded; TT / NN / MIX), weighted AUROC direction (discordant costlier -> > 0.5) and
equality with a hand-computed value, eligibility threshold, per-reference averaging (repeats kept; distinct
sensitivity), f_j and c_j.  Also: plan_stats end to end on a toy plan, the wording rule and the rider.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(R, "scripts", "r3_5_l4_link_cost_concordance.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load(SCRIPT, "r3_5_l4_link_cost_concordance")
L1 = _load(os.path.join(R, "scripts", "r3_5_l1_maps.py"), "r3_5_l1_maps_test")


def test_link_typing():
    ref = np.array([1, 1, 2, 2, 0, 1, 2, 0])
    qry = np.array([1, 2, 2, 1, 1, 0, 0, 0])
    ty = M.link_types(ref, qry)
    assert ty.tolist() == [M.TYPE_TT, M.TYPE_MIX, M.TYPE_NN, M.TYPE_MIX, 0, 0, 0, 0]
    c5, d5 = M.concordance5(np.array([1, 3, 3, 0, 4]), np.array([1, 3, 4, 3, 0]))
    assert c5.tolist() == [True, True, False, False, False]
    assert d5.tolist() == [False, False, True, False, False]


def test_binary_classes_from_labels():
    lab = np.array([0, 1, 2, 3, 4, 5])
    assert L1.class_of(lab, True).tolist() == [0, 1, 2, 2, 2, 2]
    # 5-class labels 3 vs 4 are NN in binary but discordant in five-class
    ty = M.link_types(L1.class_of(np.array([3]), True), L1.class_of(np.array([4]), True))
    c5, d5 = M.concordance5(np.array([3]), np.array([4]))
    assert ty.tolist() == [M.TYPE_NN] and d5.tolist() == [True]


def test_weighted_auroc_direction_and_hand_value():
    # positives (discordant) cost more -> AUROC 1; reversed -> 0
    pos = np.array([1, 1, 0, 0], bool)
    m = np.array([0.9, 0.8, 0.2, 0.1])
    assert M.auroc(pos, ~pos, m, np.ones(4)) == pytest.approx(1.0)
    assert M.auroc(pos, ~pos, -m, np.ones(4)) == pytest.approx(0.0)
    # hand value: pos scores [0.5 (w 1), 0.2 (w 3)], neg scores [0.3 (w 2), 0.1 (w 1)]
    # weighted AUROC = sum_{p,n} w_p w_n [s_p > s_n] / (sum w_p * sum w_n)
    #   (0.5 > 0.3: 1*2) + (0.5 > 0.1: 1*1) + (0.2 > 0.1: 3*1) = 6 ; total = 4 * 3 = 12 -> 0.5
    pos = np.array([1, 1, 0, 0], bool)
    s = np.array([0.5, 0.2, 0.3, 0.1])
    w = np.array([1.0, 3.0, 2.0, 1.0])
    assert M.auroc(pos, ~pos, s, w) == pytest.approx(6 / 12)
    # change weights: w = [3, 1, 2, 1] -> (3*2) + (3*1) + (1*1) = 10 / (4*3) = 10/12
    assert M.auroc(pos, ~pos, s, np.array([3.0, 1.0, 2.0, 1.0])) == pytest.approx(10 / 12)
    # unweighted: 3 of 4 ordered pairs -> 0.75
    assert M.auroc(pos, ~pos, s) == pytest.approx(0.75)
    # ties count one half: pos [0.3], neg [0.3, 0.1] -> (0.5 + 1)/2
    assert M.auroc(np.array([1, 0, 0], bool), np.array([0, 1, 1], bool), np.array([0.3, 0.3, 0.1])) == pytest.approx(0.75)
    # entries in neither class are dropped
    assert M.auroc(np.array([1, 0, 0], bool), np.array([0, 0, 1], bool), np.array([0.5, 99.0, 0.1]),
                   np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)
    # empty class -> NaN
    assert np.isnan(M.auroc(np.array([1, 1], bool), np.array([0, 0], bool), np.array([0.1, 0.2])))


def test_eligibility_threshold():
    assert M.eligible([10, 9, 10, 20], [10, 10, 9, 20], 10).tolist() == [True, False, False, True]
    assert M.eligible([5], [5], 5).tolist() == [True]
    assert M.eligible([19], [25], 20).tolist() == [False]


def _toy_pp():
    rows = []
    # reference A: three same-group rows (one plan repeated twice), one other-group row; B: one ineligible row
    for pid, ref, q, arm, a, mix, tt, nn in (
            ("p1", "1", "10", "same_group", 0.6, 12, 6, 6),
            ("p2", "1", "10", "same_group", 0.6, 12, 6, 6),      # upstream repeat of the (1, 10) plan
            ("p3", "1", "11", "same_group", 0.9, 12, 5, 7),
            ("p4", "1", "20", "other_group", 0.5, 30, 30, 0),
            ("p5", "2", "12", "same_group", 0.99, 9, 30, 0),      # 9 discordant links -> ineligible at 10
            ("p6", "2", "13", "same_group", 0.7, 10, 10, 0)):
        rows.append(dict(pair_id=pid, reference=ref, query=q, arm=arm, A_bin_w=a, A_bin_u=a, MIX=mix, TT=tt, NN=nn,
                         A_tanc_w=a, A_ntanc_w=a, A_5_w=a, conc5=tt + nn, disc5=mix, A_tile=a, tile_n_disc_rows=20,
                         tile_n_conc_rows=20, cost_TT=0.1, cost_NN=0.2, cost_MIX=0.3))
    return pd.DataFrame(rows)


def test_per_reference_averaging_repeats_and_distinct():
    pp = _toy_pp()
    pr = M.per_reference_mean(pp, "A_bin_w", "same_group").set_index("reference")
    assert pr.loc["1", "value"] == pytest.approx((0.6 + 0.6 + 0.9) / 3)       # repeats kept
    assert pr.loc["1", "n_rows"] == 3
    assert pr.loc["2", "value"] == pytest.approx(0.7) and pr.loc["2", "n_rows"] == 1   # p5 ineligible
    prd = M.per_reference_mean(pp, "A_bin_w", "same_group", distinct=True).set_index("reference")
    assert prd.loc["1", "value"] == pytest.approx((0.6 + 0.9) / 2) and prd.loc["1", "n_rows"] == 2
    pr5 = M.per_reference_mean(pp, "A_bin_w", "same_group", thr=5).set_index("reference")
    assert pr5.loc["2", "value"] == pytest.approx((0.99 + 0.7) / 2)
    pro = M.per_reference_mean(pp, "A_bin_w", "other_group").set_index("reference")
    assert list(pro.index) == ["1"] and pro.loc["1", "value"] == pytest.approx(0.5)
    # class-anchored: non-tumour-anchored needs >= 10 NN links -> no eligible rows for reference 1 (6, 6, 7 NN)
    assert len(M.per_reference_mean(pp, "A_ntanc_w", "same_group")) == 0
    # mean cost by type: TT needs >= 10 TT links
    ct = M.per_reference_mean(pp, "cost_TT", "same_group").set_index("reference")
    assert list(ct.index) == ["2"] and ct.loc["2", "n_rows"] == 2


def test_estimate_and_bootstrap_conventions():
    pp = _toy_pp()
    refinfo = pd.DataFrame({"reference": ["1", "2"], "ref_patient": ["pa", "pb"], "ref_group": ["G", "G"],
                            "half": ["development", "report"], "ref_strict": ["True", "False"]})
    e, pr = M.estimate(L1, pp, refinfo, "A_bin_w", "same_group", B=200, seed=123)
    assert e["point"] == pytest.approx(((0.6 + 0.6 + 0.9) / 3 + 0.7) / 2)
    assert e["n_references"] == 2 and e["n_patients"] == 2 and e["n_pairs"] == 4 and e["B"] == 200 and e["seed"] == 123
    e2, _ = M.estimate(L1, pp, refinfo, "A_bin_w", "same_group", B=200, seed=123)
    e3, _ = M.estimate(L1, pp, refinfo, "A_bin_w", "same_group", B=200, seed=124)
    assert (e["ci95_lo"], e["ci95_hi"]) == (e2["ci95_lo"], e2["ci95_hi"]) and e3["point"] == e["point"]
    es, _ = M.estimate(L1, pp, refinfo, "A_bin_w", "same_group", B=50, seed=123,
                       subset=lambda d: d.ref_strict.astype(str) == "True")
    assert es["n_references"] == 1 and es["point"] == pytest.approx((0.6 + 0.6 + 0.9) / 3)
    assert es["ci95_lo"] == es["ci95_hi"] == pytest.approx(es["point"])
    # paired difference: only reference 1 has both arms for A_bin_w
    pp2 = pp.copy()
    pp2["cost_TT"] = np.where(pp2.arm == "same_group", 0.1, 0.4)
    pp2.loc[pp2.pair_id == "p4", "TT"] = 30
    pp2.loc[pp2.pair_id.isin(["p1", "p2", "p3"]), "TT"] = 10
    dpair = M.paired_difference(L1, pp2, refinfo, "cost_TT", B=50, seed=123)
    assert dpair["n_references"] == 1 and dpair["point"] == pytest.approx(0.1 - 0.4) and dpair["n_pairs"] == 4


def test_tile_fc():
    # reference rows: 0 tumour, 1 non-tumour, 2 unannotated, 3 tumour (only an unannotated query link)
    cr2 = np.array([1, 2, 0, 1])
    cq2 = np.array([1, 2, 1, 0])
    i = np.array([0, 1, 1, 0, 2, 3])
    j = np.array([0, 0, 1, 1, 2, 3])
    t = np.array([0.1, 0.3, 0.2, 0.2, 0.25, 0.25])
    m = np.array([0.2, 0.6, 0.4, 0.8, 0.5, 0.9])
    ty = M.link_types(cr2[j], cq2[i])
    ann, conc = ty > 0, (ty == M.TYPE_TT) | (ty == M.TYPE_NN)
    f, c = M.tile_fc(j, t, m, ann, conc, 4, cr2 > 0)
    # row 0: links TT (0.1, 0.2) and MIX (0.3, 0.6): f = 0.1/0.4, c = (0.02 + 0.18)/0.4
    assert f[0] == pytest.approx(0.25) and c[0] == pytest.approx(0.5)
    # row 1: NN (0.2, 0.4) and MIX (0.2, 0.8): f = 0.5, c = 0.6
    assert f[1] == pytest.approx(0.5) and c[1] == pytest.approx(0.6)
    # row 2 unannotated reference row -> f NaN, c still the mean over all links
    assert np.isnan(f[2]) and c[2] == pytest.approx(0.5)
    # row 3 annotated but no annotated link -> f NaN
    assert np.isnan(f[3]) and c[3] == pytest.approx(0.9)
    # c_j equals the L1 dense-plan per-tile mean cost
    T = np.zeros((3 + 1, 4))
    Mc = np.zeros((4, 4), dtype=np.float32)
    T[i, j] = t
    Mc[i, j] = m
    val, linked = L1.b_tile_values(T, Mc, "meancost")
    assert np.allclose(val, c, atol=1e-7)
    a, npos, nneg = M.tile_auroc(f, c)
    assert npos == 1 and nneg == 1 and a == pytest.approx(0.0)      # f < 0.5 row (cost 0.5) vs f = 0.5 row (0.6)


def test_plan_stats_end_to_end():
    # 2 query rows x 2 reference rows; full balanced plan of 4 links
    cr2, cq2 = np.array([1, 2]), np.array([1, 2])
    cr5, cq5 = np.array([1, 3]), np.array([1, 4])
    i = np.array([0, 0, 1, 1])
    j = np.array([0, 1, 0, 1])
    t = np.array([0.4, 0.1, 0.1, 0.4])
    m = np.array([0.1, 0.9, 0.8, 0.2])
    rec, c = M.plan_stats(i, j, t, m, cq2, cr2, cq5, cr5)
    assert (rec["TT"], rec["NN"], rec["MIX"], rec["ann"]) == (1, 1, 2, 4)
    assert (rec["conc5"], rec["disc5"]) == (1, 3)                    # 3 vs 4 is discordant in five-class
    assert rec["S_bin"] == pytest.approx(0.8) and rec["S_5"] == pytest.approx(0.4)
    assert rec["ann_share_bin"] == pytest.approx(1.0)
    assert rec["A_bin_w"] == pytest.approx(1.0)                      # MIX links cost more
    assert rec["cost_MIX"] == pytest.approx(0.85) and rec["cost_TT"] == pytest.approx(0.1)
    assert c.tolist() == pytest.approx([(0.04 + 0.08) / 0.5, (0.09 + 0.08) / 0.5])


def test_wording_rule_and_rider():
    assert M.wording_rule(0.72, 0.6, 0.8) == "lo_gt_half_point_ge_070"
    assert M.wording_rule(0.70, 0.51, 0.8) == "lo_gt_half_point_ge_070"
    assert M.wording_rule(0.69, 0.51, 0.8) == "lo_gt_half_point_lt_070"
    assert M.wording_rule(0.6, 0.5, 0.7) == "ci_spans_half"           # strict inequality
    assert M.wording_rule(0.4, 0.3, 0.49) == "hi_lt_half"
    assert M.wording_rule(0.45, 0.3, 0.5) == "ci_spans_half"
    s = M.fill_wording("lo_gt_half_point_ge_070", 0.7312, 0.701, 0.77, 44)
    assert "AUROC 0.73, 95% CI 0.70–0.77; 44 reference patients" in s
    assert M.fill_wording("ci_spans_half", 0.5, 0.4, 0.6, 44) == M.WORDING["ci_spans_half"]
    assert M.rider_rule(0.6, 0.55, 0.7, 27) == "both"
    assert M.rider_rule(0.6, 0.45, 0.7, 27) == "tumour_only"
    assert M.rider_rule(0.6, 0.55, 0.7, 14) == "tumour_only"
    assert M.rider_rule(0.6, None, None, 0) == "tumour_only"
    assert M.rider_rule(0.45, 0.55, 0.7, 27) == "not_covered"
    assert M.rider_rule(0.6, 0.3, 0.45, 27) == "not_covered"
