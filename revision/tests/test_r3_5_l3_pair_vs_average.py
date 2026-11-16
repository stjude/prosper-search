"""Tests for scripts/r3_5_l3_pair_vs_average.py (specification 005 F5; toy tables only, no production input read or written).

F5 (specification 5): delta is computed per reference with the correct sign (yardstick minus single-pair mean);
references without a yardstick are excluded from the primary frame; the bootstrap returns the point estimate at
B = 1 with a single-reference resample; k-curve subset drawing honours k and the seed.  Also covered: the
row-level AUROC kernel (label 0 dropped, missing class -> NaN) and the mechanical wording rule.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(R, "scripts", "r3_5_l3_pair_vs_average.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load(SCRIPT, "r3_5_l3_pair_vs_average")


def toy_pairs():
    """Three references: A (10 same + 10 other rows, yardstick 0.8), B (10 same + 10 other, yardstick 0.5),
    C (0 same + 10 other, no yardstick).  Same-group auroc_A of A = 0.60..0.69, of B = 0.70..0.79."""
    rows = []
    for ref, pat, same_vals in (("1", "pA", np.arange(10) / 100 + 0.60), ("2", "pB", np.arange(10) / 100 + 0.70)):
        for i, v in enumerate(same_vals):
            rows.append(dict(reference=int(ref), query=100 + i, pair_id=f"P2:{ref}:same_group:{i}", arm="same_group",
                             ref_patient=pat, ref_group="G", half="development", ref_strict=True, n_annotated=50,
                             auroc_A=v, ap_A=v / 2, pointing_A=1.0 if i % 2 else 0.0,
                             setting=M.SETTING, design=M.DESIGN))
    for ref, pat in (("1", "pA"), ("2", "pB"), ("3", "pC")):
        for i in range(10):
            rows.append(dict(reference=int(ref), query=200 + i, pair_id=f"P2:{ref}:other_group:{i}", arm="other_group",
                             ref_patient=pat, ref_group="G", half="development", ref_strict=True, n_annotated=50,
                             auroc_A=0.4, ap_A=0.2, pointing_A=0.0, setting=M.SETTING, design=M.DESIGN))
    # a row from another setting / design that must be filtered out
    rows.append(dict(reference=1, query=999, pair_id="P1:1:same_group:0", arm="same_group", ref_patient="pA",
                     ref_group="G", half="development", ref_strict=True, n_annotated=50, auroc_A=0.99, ap_A=0.99,
                     pointing_A=1.0, setting=M.SETTING, design="P1"))
    rows.append(dict(reference=1, query=998, pair_id="P2:1:same_group:x", arm="same_group", ref_patient="pA",
                     ref_group="G", half="development", ref_strict=True, n_annotated=50, auroc_A=0.01, ap_A=0.01,
                     pointing_A=0.0, setting="B|other|none|None", design="P2"))
    pp = pd.DataFrame(rows)
    ref_d2 = pd.DataFrame({"reference": [1, 2, 3], "yard_auroc_A": [0.8, 0.5, np.nan], "yard_ap_A": [0.4, 0.3, np.nan],
                           "yard_pointing_A": [1.0, 0.0, np.nan], "yard_n_annotated": [50, 50, np.nan]})
    return pp, ref_d2


def test_delta_sign_and_exclusion():
    pp, ref_d2 = toy_pairs()
    p2 = M.select_p2(pp)
    assert len(p2) == 50                       # 20 same + 30 other; P1 row and other-setting row dropped
    tab = M.per_reference_table(p2, ref_d2)
    assert len(tab) == 3
    t = tab.set_index("reference")
    assert t.loc[1, "single_mean"] == pytest.approx(0.645)
    assert t.loc[2, "single_mean"] == pytest.approx(0.745)
    assert t.loc[1, "delta"] == pytest.approx(0.8 - 0.645)      # yardstick minus single: positive
    assert t.loc[2, "delta"] == pytest.approx(0.5 - 0.745)      # negative
    assert t.loc[1, "single_sd"] == pytest.approx(np.std(np.arange(10) / 100, ddof=1))
    assert t.loc[1, "other_mean"] == pytest.approx(0.4)
    assert t.loc[1, "single_minus_other"] == pytest.approx(0.645 - 0.4)
    assert t.loc[1, "delta_ap"] == pytest.approx(0.4 - 0.645 / 2)
    assert t.loc[1, "delta_pointing"] == pytest.approx(1.0 - 0.5)
    assert t.loc[3, "n_same_rows"] == 0 and np.isnan(t.loc[3, "yard"]) and np.isnan(t.loc[3, "delta"])
    assert t.loc[3, "other_mean"] == pytest.approx(0.4)        # other-group arm still present for ref 3
    prim = M.primary_frame(tab)
    assert prim.reference.tolist() == [1, 2]                     # ref 3 (no yardstick, 0 same rows) excluded


def test_duplicate_pair_rows_count_with_multiplicity():
    pp, ref_d2 = toy_pairs()
    pp2 = pp.copy()
    # make the query of ref 1's row with auroc 0.69 a duplicate plan of its 0.60 row: mean over rows is unchanged
    tab = M.per_reference_table(M.select_p2(pp2), ref_d2).set_index("reference")
    pp2.loc[(pp2.reference == 1) & (pp2.arm == "same_group") & (pp2["query"] == 109), "query"] = 100
    tab2 = M.per_reference_table(M.select_p2(pp2), ref_d2).set_index("reference")
    assert tab2.loc[1, "single_mean"] == pytest.approx(tab.loc[1, "single_mean"])
    assert tab2.loc[1, "n_same_rows"] == 10 and tab2.loc[1, "n_distinct_same_queries"] == 9


def test_bootstrap_single_reference_B1_returns_point():
    e = M.boot_estimate([0.7], ["p1"], B=1, seed=123)
    assert e["point"] == e["ci95_lo"] == e["ci95_hi"] == pytest.approx(0.7)
    assert e["n_references"] == 1 and e["n_patients"] == 1 and e["B"] == 1


def test_bootstrap_determinism_and_shared_draws():
    v = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    pats = ["a", "b", "c", "d", "e"]
    e1 = M.boot_estimate(v, pats, B=200, seed=123)
    e2 = M.boot_estimate(v, pats, B=200, seed=123)
    e3 = M.boot_estimate(v, pats, B=200, seed=124)
    assert (e1["ci95_lo"], e1["ci95_hi"]) == (e2["ci95_lo"], e2["ci95_hi"])
    assert e1["point"] == e3["point"] == pytest.approx(v.mean())
    assert (e1["ci95_lo"], e1["ci95_hi"]) != (e3["ci95_lo"], e3["ci95_hi"])
    assert e1["ci95_lo"] <= e1["point"] <= e1["ci95_hi"]
    # shared draws: the interval of a - b equals the interval of the paired difference computed from the same matrix
    draws = M.make_draws(pats, 200, 123)
    a = M.boot_estimate(v, pats, 200, 123, draws=draws)
    b = M.boot_estimate(v + 0.05, pats, 200, 123, draws=draws)
    assert b["ci95_lo"] == pytest.approx(a["ci95_lo"] + 0.05) and b["ci95_hi"] == pytest.approx(a["ci95_hi"] + 0.05)
    # the draw matrix follows the L1 convention: p integers in [0, p) per replicate, sorted patient order
    pats_s, mult = draws
    assert pats_s == ("a", "b", "c", "d", "e") and mult.shape == (200, 5) and (mult.sum(1) == 5).all()
    rng = np.random.default_rng(123)
    assert (mult[0] == np.bincount(rng.integers(5, size=5), minlength=5)).all()


def test_bootstrap_patient_sort_order_independent_of_input_order():
    v = np.array([0.1, 0.5, 0.9])
    pats = ["b", "a", "c"]
    e1 = M.boot_estimate(v, pats, B=100, seed=123)
    order = [1, 0, 2]
    e2 = M.boot_estimate(v[order], [pats[i] for i in order], B=100, seed=123)
    assert (e1["ci95_lo"], e1["ci95_hi"]) == (e2["ci95_lo"], e2["ci95_hi"])


def test_kcurve_subsets_honour_k_and_seed():
    s1 = M.kcurve_subsets(10, 3, 20, 123)
    s2 = M.kcurve_subsets(10, 3, 20, 123)
    s3 = M.kcurve_subsets(10, 3, 20, 124)
    assert len(s1) == 20 and all(len(s) == 3 and len(set(s)) == 3 and s.min() >= 0 and s.max() < 10 for s in s1)
    assert all((a == b).all() for a, b in zip(s1, s2))
    assert any(not (a == c).all() for a, c in zip(s1, s3))
    full = M.kcurve_subsets(10, 10, 20, 123)
    assert len(full) == 1 and (full[0] == np.arange(10)).all()
    for k in (1, 2, 5):
        assert all(len(s) == k for s in M.kcurve_subsets(10, k, 20, 123))
    with pytest.raises(ValueError):
        M.kcurve_subsets(5, 6, 20, 123)


def test_row_auroc_kernel():
    lab = np.array([1, 1, 3, 4, 0, 0, 5])
    score = np.array([0.9, 0.8, 0.1, 0.2, 5.0, -5.0, 0.3])    # label-0 rows carry extreme scores and must be ignored
    assert M.row_auroc(score, lab) == pytest.approx(1.0)
    assert M.row_auroc(-score, lab) == pytest.approx(0.0)
    assert np.isnan(M.row_auroc(score, np.array([1, 1, 0, 0])))          # no negatives
    assert np.isnan(M.row_auroc(score[:3], np.array([3, 4, 5])))         # no positives
    assert M.row_auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 1, 3, 4])) == pytest.approx(0.5)


def test_kcurve_per_reference_toy():
    # reference 7: 4 rows (labels 1,1,3,4), three same-group pair rows (one plan duplicated): costs chosen so that
    # the mean over all three plans separates tumour (low cost) from non-tumour perfectly, while one single plan
    # alone is anti-aligned.
    cost = {11: [0.1, 0.2, 0.9, 0.8], 12: [0.9, 0.8, 0.1, 0.2], 13: [0.0, 0.1, 0.95, 0.9]}
    rows = []
    for q, c in cost.items():
        for i, v in enumerate(c):
            rows.append(dict(reference=7, query=q, ref_row=i, cost=v, ref_label=[1, 1, 3, 4][i]))
    per_tile = pd.DataFrame(rows)
    same_rows = pd.DataFrame({"reference": [7, 7, 7], "query": [11, 12, 11]})     # query 11 appears twice
    kc = M.kcurve_per_reference(per_tile, same_rows, ks=(1, 3), n_draws=5, seed=123)
    assert kc.shape[0] == 1 and kc.n_queries.iloc[0] == 3 and kc.n_rows_pos.iloc[0] == 2 and kc.n_rows_neg.iloc[0] == 2
    # k = 3 = full multiset {11, 12, 11}: mean cost = (2*c11 + c12)/3 -> tumour rows lower -> AUROC 1
    assert kc.k_3.iloc[0] == pytest.approx(1.0) and kc.n_subsets_3.iloc[0] == 1
    # k = 1: each draw is plan 11 (AUROC 1) or plan 12 (AUROC 0); the mean over 5 draws lies in [0, 1]
    assert 0.0 <= kc.k_1.iloc[0] <= 1.0 and kc.n_subsets_1.iloc[0] == 5
    # a reference with no negative labelled rows is scored NaN (skipped), not an error
    per_tile2 = per_tile.copy()
    per_tile2["ref_label"] = 1
    kc2 = M.kcurve_per_reference(per_tile2, same_rows, ks=(1, 3), n_draws=5, seed=123)
    assert np.isnan(kc2.k_3.iloc[0]) and kc2.n_rows_neg.iloc[0] == 0


def test_wording_rule_branches():
    assert M.wording_rule(0.01, 0.2) == "ci_lo_gt_0"
    assert M.wording_rule(-0.01, 0.2) == "ci_spans_0"
    assert M.wording_rule(0.0, 0.2) == "ci_spans_0"           # strict inequality
    assert M.wording_rule(-0.2, -0.01) == "ci_hi_lt_0"
    assert M.wording_rule(-0.2, 0.0) == "ci_spans_0"
    assert M.WORDING["ci_lo_gt_0"].startswith("averaging the matched-cost map")
    assert M.WORDING["ci_spans_0"] == "no established difference between the single-pair and neighbour-averaged maps"
    filled = M.fill_wording("ci_lo_gt_0", 0.1234, 0.05, 0.2, 45)
    assert "+0.123 [+0.050, +0.200]" in filled and "45 reference patients" in filled
    assert M.fill_wording("ci_spans_0", 0.0, -0.1, 0.1, 45) == M.WORDING["ci_spans_0"]
