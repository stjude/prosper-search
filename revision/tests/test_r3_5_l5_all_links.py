"""Tests for scripts/r3_5_l5_all_links.py (contract_007 F6; toy data only, no production input read).

F6: the four class versions (annotation overrides prediction under fill; unannotated -> other under lit; excluded
under ann), held-out prediction slicing, B direction and a hand-computed value, eligibility, typing completeness;
also the lit unannotated-end share, prediction agreement, per-reference averaging by version and the rule branch.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load(os.path.join(R, "scripts", "r3_5_l5_all_links.py"), "r3_5_l5_all_links")
L1 = _load(os.path.join(R, "scripts", "r3_5_l1_maps.py"), "r3_5_l1_maps_test5")


def test_class_versions():
    # annotation classes: 0 unannotated, 1 tumour, 2 non-tumour; prediction: True = predicted tumour
    c2 = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([True, False, True, False, True, False])
    v = M.class_versions(c2, pred)
    assert v["ann"].tolist() == [0, 0, 1, 1, 2, 2]                 # unannotated excluded
    assert v["fill"].tolist() == [1, 2, 1, 1, 2, 2]                # annotation overrides prediction
    assert v["lit"].tolist() == [2, 2, 1, 1, 2, 2]              # unannotated -> other; annotated non-tumour -> other
    assert v["pred"].tolist() == [1, 2, 1, 2, 1, 2]                # prediction everywhere
    # from raw labels via L1.class_of (labels 3-5 pooled into non-tumour)
    c = L1.class_of(np.array([0, 1, 3, 5]), True)
    assert M.class_versions(c, np.array([True, True, True, False]))["fill"].tolist() == [1, 1, 2, 2]


def test_held_out_prediction_slice():
    masks = {0: np.array([1, 0, 0, 1, 1], bool), 2: np.array([0, 1, 1, 0, 0], bool)}
    assert M.held_out_prediction(masks, 2, 1, 4).tolist() == [True, True, False]
    assert M.held_out_prediction(masks, 0, 3, 5).tolist() == [True, True]
    with pytest.raises(M.Stop):
        M.held_out_prediction(masks, -1, 0, 2)


def _toy_plan():
    # 3 query rows x 3 reference rows, 6 links
    i = np.array([0, 0, 1, 1, 2, 2])
    j = np.array([0, 1, 1, 2, 0, 2])
    t = np.array([0.2, 0.1, 0.15, 0.15, 0.2, 0.2])
    m = np.array([0.1, 0.5, 0.4, 0.9, 0.3, 0.6])
    cq2, cr2 = np.array([1, 0, 2]), np.array([1, 0, 2])      # query/reference row 1 unannotated
    pq, pr = np.array([True, True, False]), np.array([True, False, True])
    return i, j, t, m, cq2, cr2, pq, pr


def test_version_stats_typing_and_completeness():
    i, j, t, m, cq2, cr2, pq, pr = _toy_plan()
    vq, vr = M.class_versions(cq2, pq), M.class_versions(cr2, pr)
    # ann: links (q,r) classes: (1,1) TT, (1,0) excl, (0,0) excl, (0,2) excl, (2,1) MIX, (2,2) NN
    a = M.version_stats(i, j, t, m, vq["ann"], vr["ann"])
    assert (a["TT"], a["NN"], a["MIX"], a["n_typed"], a["nonTT"]) == (1, 1, 1, 3, 2)
    assert a["mass_typed"] == pytest.approx(0.6) and a["typed_mass_share"] == pytest.approx(0.6)
    # fill: q classes [1,1,2], r classes [1,2,2] -> (1,1)TT (1,2)MIX (1,2)MIX (1,2)MIX (2,1)MIX (2,2)NN
    f = M.version_stats(i, j, t, m, vq["fill"], vr["fill"])
    assert (f["TT"], f["NN"], f["MIX"], f["n_typed"]) == (1, 1, 4, 6)
    assert f["mass_typed"] == pytest.approx(1.0) and f["n_links"] - f["n_typed"] == 0
    assert f["share_links_MIX"] == pytest.approx(4 / 6) and f["share_mass_TT"] == pytest.approx(0.2)
    # lit: q [1,2,2], r [1,2,2] -> TT, MIX, NN, NN, MIX, NN
    li = M.version_stats(i, j, t, m, vq["lit"], vr["lit"], cq2, cr2)
    assert (li["TT"], li["NN"], li["MIX"], li["n_typed"]) == (1, 3, 2, 6)
    # other ends: ref ends j=[0,1,1,2,0,2] classes [1,2,2,2,1,2] -> 4 other, unann at j=1 -> 2;
    #             query ends i=[0,0,1,1,2,2] classes [1,1,2,2,2,2] -> 4 other, unann at i=1 -> 2
    assert li["other_ends"] == 8 and li["other_ends_unann"] == 4 and li["other_ends_unann_share"] == pytest.approx(0.5)
    # mass-weighted: other-end mass ref (0.1+0.15+0.15+0.2) + query (0.15+0.15+0.2+0.2) = 1.3; unann 0.25 + 0.3
    assert li["other_ends_unann_mass_share"] == pytest.approx(0.55 / 1.3)
    # pred: q [1,1,2], r [1,2,1] -> (1,1)TT (1,2)MIX (1,2)MIX (1,1)TT (2,1)MIX (2,1)MIX
    p = M.version_stats(i, j, t, m, vq["pred"], vr["pred"])
    assert (p["TT"], p["NN"], p["MIX"], p["n_typed"]) == (2, 0, 4, 6)
    assert np.isnan(p["cost_NN"]) and np.isnan(p["A_ntanc_w"])


def test_B_direction_and_hand_value():
    i, j, t, m, cq2, cr2, pq, pr = _toy_plan()
    vq, vr = M.class_versions(cq2, pq), M.class_versions(cr2, pr)
    # lit: TT link 0 (m 0.1, w 0.2); non-TT links m [0.5,0.4,0.9,0.3,0.6] all above 0.1 -> B = 1
    li = M.version_stats(i, j, t, m, vq["lit"], vr["lit"], cq2, cr2)
    assert li["B_w"] == pytest.approx(1.0) and li["B_u"] == pytest.approx(1.0)
    # reversed cost -> 0
    assert M.version_stats(i, j, t, -m, vq["lit"], vr["lit"])["B_w"] == pytest.approx(0.0)
    # pred: TT links m [0.1 (w .2), 0.9 (w .15)]; non-TT m [0.5 (.1), 0.4 (.15), 0.3 (.2), 0.6 (.2)]
    # weighted B = sum_{p in nonTT, n in TT} w_p w_n [s_p > s_n] / (sum w_p * sum w_n)
    #   vs TT 0.1: all four greater -> 0.65 * 0.2 = 0.13; vs TT 0.9: none -> 0; total 0.65 * 0.35 = 0.2275
    p = M.version_stats(i, j, t, m, vq["pred"], vr["pred"])
    assert p["B_w"] == pytest.approx(0.13 / 0.2275)
    assert p["B_u"] == pytest.approx(4 / 8)
    # NN counts as y = 1 in B but y = 0 in A
    f = M.version_stats(i, j, t, m, vq["fill"], vr["fill"])
    # fill: TT m 0.1 (w .2); non-TT MIX [0.5,0.4,0.9,0.3] + NN 0.6 -> B = 1
    assert f["B_w"] == pytest.approx(1.0)
    # A: MIX [0.5 (.1),0.4 (.15),0.9 (.15),0.3 (.2)] vs TT|NN [0.1 (.2), 0.6 (.2)]
    #   vs 0.1: all -> 0.6*0.2 = 0.12; vs 0.6: only 0.9 -> 0.15*0.2 = 0.03; total 0.6*0.4 = 0.24
    assert f["A_w"] == pytest.approx(0.15 / 0.24)
    assert f["cost_MIX"] == pytest.approx((0.05 + 0.06 + 0.135 + 0.06) / 0.6)


def test_ann_matches_L4_plan_stats():
    rng = np.random.default_rng(0)
    nq, nr, nl = 30, 40, 400
    i, j = rng.integers(nq, size=nl), rng.integers(nr, size=nl)
    t = rng.random(nl)
    t /= t.sum()
    m = rng.random(nl)
    cq2, cr2 = rng.integers(0, 3, size=nq), rng.integers(0, 3, size=nr)
    a = M.version_stats(i, j, t, m, M.class_versions(cq2, cq2 > 0)["ann"], M.class_versions(cr2, cr2 > 0)["ann"])
    ref, _ = M.L4.plan_stats(i, j, t, m, cq2, cr2, cq2, cr2)
    assert a["A_w"] == ref["A_bin_w"] and a["A_tanc_w"] == ref["A_tanc_w"] and a["A_ntanc_w"] == ref["A_ntanc_w"]
    assert (a["TT"], a["NN"], a["MIX"], a["n_typed"]) == (ref["TT"], ref["NN"], ref["MIX"], ref["ann"])


def _toy_pp():
    rows = []
    for pid, ref, q, arm, ver, b, tt, ntt in (
            ("p1", "1", "10", "same_group", "fill", 0.6, 10, 10),
            ("p2", "1", "10", "same_group", "fill", 0.6, 10, 10),     # repeat of the (1, 10) plan
            ("p3", "1", "11", "same_group", "fill", 0.9, 30, 40),
            ("p4", "1", "11", "same_group", "lit", 0.5, 30, 40),      # other version
            ("p5", "2", "12", "same_group", "fill", 0.99, 9, 100),    # 9 TT links -> ineligible at 10
            ("p6", "2", "13", "same_group", "fill", 0.7, 100, 10),
            ("p7", "2", "13", "other_group", "fill", 0.4, 100, 10)):
        rows.append(dict(pair_id=pid, reference=ref, query=q, arm=arm, version=ver, B_w=b, B_u=b, TT=tt, nonTT=ntt,
                         MIX=ntt, NN=0, A_w=b, n_links=tt + ntt, share_links_TT=tt / (tt + ntt)))
    return pd.DataFrame(rows)


def test_eligibility_and_per_reference_by_version():
    assert M.eligible([10, 9, 10], [10, 10, 9], 10).tolist() == [True, False, False]
    pp = _toy_pp()
    pr = M.per_reference_mean(pp, "B_w", "same_group", "fill").set_index("reference")
    assert pr.loc["1", "value"] == pytest.approx((0.6 + 0.6 + 0.9) / 3) and pr.loc["1", "n_rows"] == 3
    assert pr.loc["2", "value"] == pytest.approx(0.7) and pr.loc["2", "n_rows"] == 1
    assert M.per_reference_mean(pp, "B_w", "same_group", "lit").set_index("reference").loc["1", "value"] == 0.5
    prd = M.per_reference_mean(pp, "B_w", "same_group", "fill", distinct=True).set_index("reference")
    assert prd.loc["1", "value"] == pytest.approx(0.75)
    pr5 = M.per_reference_mean(pp, "B_w", "same_group", "fill", thr=5).set_index("reference")
    assert pr5.loc["2", "value"] == pytest.approx((0.99 + 0.7) / 2)
    # composition: thr 0 -> every row with a finite value
    pc = M.per_reference_mean(pp, "share_links_TT", "same_group", "fill", thr=0).set_index("reference")
    assert pc.loc["2", "n_rows"] == 2
    refinfo = pd.DataFrame({"reference": ["1", "2"], "ref_patient": ["pa", "pb"], "ref_group": ["G", "G"],
                            "half": ["development", "report"], "ref_strict": ["True", "False"]})
    e, _ = M.estimate(L1, pp, refinfo, "B_w", "same_group", "fill", B=200, seed=123)
    assert e["point"] == pytest.approx(((0.6 + 0.6 + 0.9) / 3 + 0.7) / 2)
    assert e["n_references"] == 2 and e["n_pairs"] == 4 and e["version"] == "fill"
    e2, _ = M.estimate(L1, pp, refinfo, "B_w", "same_group", "fill", B=200, seed=123)
    e3, _ = M.estimate(L1, pp, refinfo, "B_w", "same_group", "fill", B=200, seed=124)
    assert (e["ci95_lo"], e["ci95_hi"]) == (e2["ci95_lo"], e2["ci95_hi"]) and e3["point"] == e["point"]
    d = M.paired_difference(L1, pp, refinfo, "B_w", "fill", B=50, seed=123)
    assert d["n_references"] == 1 and d["point"] == pytest.approx(0.7 - 0.4)


def test_prediction_agreement():
    c2 = np.array([1, 1, 1, 2, 2, 0, 0])
    pred = np.array([True, True, False, False, True, True, False])
    a = M.prediction_agreement(c2, pred)
    assert a["n_annotated"] == 5 and a["accuracy"] == pytest.approx(3 / 5)
    assert a["sensitivity"] == pytest.approx(2 / 3) and a["specificity"] == pytest.approx(1 / 2)
    assert a["pred_tumour_share_unannotated"] == pytest.approx(0.5)
    assert np.isnan(M.prediction_agreement(np.array([2, 0]), np.array([True, False]))["sensitivity"])


def test_rule_branch():
    assert M.rule_branch(0.51, 0.9) == "fill_ci_lo_gt_half"
    assert M.rule_branch(0.5, 0.9) == "fill_ci_spans_half"          # strict inequality
    assert M.rule_branch(0.3, 0.49) == "fill_ci_hi_lt_half"
    assert M.rule_branch(0.3, 0.5) == "fill_ci_spans_half"
    assert M.RULE_TEXT["fill_ci_hi_lt_half"] == M.RULE_TEXT["fill_ci_spans_half"]
    s = M.fill_rule_sentence("fill_ci_lo_gt_half", 0.7812, 0.702, 0.85, 44)
    assert "AUROC 0.78, 95% CI 0.70–0.85; 44 reference patients" in s
