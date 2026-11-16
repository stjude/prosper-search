"""Tests for scripts/r3_5_h2_roi_class_specificity.py (specification 008 G4; toy slides only, no production input read).

G4: ROI draw rule, class-c target labelling and eligibility, M / C / D on a toy case where the map must follow the
ROI (M > 0.5 > C), both aggregation directions, patient-distinctness assertion.  Also: the AUROC_1 path equals the
L1 score_continuous AUROC (G1 route), the rule branches and source eligibility.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(R, "scripts", "r3_5_h2_roi_class_specificity.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load(SCRIPT, "r3_5_h2_roi_class_specificity")
L1 = _load(os.path.join(R, "scripts", "r3_5_l1_maps.py"), "r3_5_l1_maps_test_h2")


class _LC:
    """Stand-in exposing localization_cli._auroc's mid-rank definition (no prosper import in the unit tests)."""

    @staticmethod
    def _auroc(score, y):
        from scipy.stats import rankdata
        pos, neg = score[y == 1], score[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        r = rankdata(np.concatenate([pos, neg]))
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    @staticmethod
    def _ap(score, y):                                       # needed only by L1.score_continuous; not tested here
        return float("nan")


class ToySlide:
    """Slide-like object: X (tiles x dim), tile_label, patch_label and _owners (one owner per patch here)."""

    def __init__(self, tile_label, rng, centres, noise=0.05):
        self.tile_label = np.asarray(tile_label, dtype=np.int8)
        self.X = np.stack([centres[int(c)] + noise * rng.standard_normal(len(centres[0]))
                           for c in self.tile_label]).astype(np.float32)
        # two patches per tile, both carrying the tile's label
        self._owners = [np.array([j]) for j in range(len(self.tile_label)) for _ in range(2)]
        self.patch_label = np.repeat(self.tile_label, 2)


def _centres(dim=6):
    e = np.eye(dim)
    # 0 unannotated, 1 tumour, 2 immune, 3 necrosis, 4 nonTumor, 5 artifact: distinct directions (+ offset)
    return {c: e[c] + 0.2 for c in range(6)}


def test_roi_draw_rule():
    tl = np.array([0, 3, 1, 3, 3, 1, 3, 3, 0, 3, 3, 3, 1, 4])
    idx = M.draw_roi(tl, 3, "1234")
    rows3 = np.flatnonzero(tl == 3)
    expect = np.sort(np.random.default_rng([123, 3, 1234]).choice(rows3, size=8, replace=False))
    assert np.array_equal(idx, expect)
    assert np.all(np.diff(idx) > 0) and set(idx) <= set(rows3)
    assert M.draw_roi(tl, 4, "1234") is None                 # 1 tile of class 4 < 8
    assert np.array_equal(M.draw_roi(tl, 3, "1234"), idx)     # deterministic
    rows = M.roi_rows(tl, 3)
    assert rows.tolist() == sorted(rows.tolist())
    X = np.arange(len(tl) * 2, dtype=float).reshape(len(tl), 2)
    assert np.allclose(M.roi_vector(X, idx), X[idx].mean(axis=0))
    # alternative seed draws through default_rng([seed, c, sid])
    alt = M.draw_roi(np.array([3] * 30), 3, "77", seed=124)
    assert np.array_equal(alt, np.sort(np.random.default_rng([124, 3, 77]).choice(np.arange(30), 8, replace=False)))


def test_source_eligibility():
    assert M.is_source(np.array([3] * 8 + [1] * 8), 3)
    assert not M.is_source(np.array([3] * 8 + [1] * 7), 3)
    assert not M.is_source(np.array([3] * 7 + [1] * 20), 3)
    assert not M.is_source(np.array([4] * 20 + [1] * 20), 3)


def test_class_labelling_and_target_eligibility():
    pl = np.array([0, 0, 1, 2, 3, 3, 4, 5, 3])
    ann, y = M.class_labels(pl, 3)
    assert ann.tolist() == [False, False, True, True, True, True, True, True, True]
    assert y.tolist() == [0, 0, 1, 1, 0, 0, 1]               # unannotated (0) excluded
    assert M.class_counts(pl, 3) == (3, 4)
    assert M.class_counts(pl, 1) == (1, 6)
    pl2 = np.array([3] * 20 + [1] * 10 + [4] * 10 + [0] * 50)
    assert M.target_eligible(pl2, 3, 20)                     # 20 class-3, 20 other annotated
    assert not M.target_eligible(pl2, 3, 21)
    assert not M.target_eligible(pl2, 1, 20)                 # only 10 tumour patches
    assert M.target_eligible(pl2, 1, 10)


def test_auroc_class_orientation_and_g1_route_identity():
    pl = np.array([1, 1, 3, 4, 0, 1, 2])
    h = np.array([0.9, 0.8, 0.1, 0.2, 0.99, 0.7, 0.3])
    assert M.auroc_class(h, pl, 1, _LC) == pytest.approx(1.0)
    assert M.auroc_class(-h, pl, 1, _LC) == pytest.approx(0.0)
    assert M.auroc_class(h, pl, 3, _LC) == pytest.approx(0.0)   # class 3 ranked lowest
    assert np.isnan(M.auroc_class(h, np.array([1, 1, 0, 0, 0, 1, 1]), 1, _LC))
    assert np.isnan(M.auroc_class(h, pl, 3, _LC, min_pos=2))
    # AUROC_1 path equals L1.score_continuous's variant-A AUROC (the G1 route), bit for bit
    rng = np.random.default_rng(0)
    pl = rng.integers(0, 6, 300)
    h = np.round(rng.random(300), 2)                          # ties present
    a = M.auroc_class(h, pl, 1, _LC)
    b = L1.score_continuous(h, pl, _LC, min_neg=1)["auroc_A"]
    assert a == b


def test_toy_map_follows_roi():
    rng = np.random.default_rng(1)
    cen = _centres()
    src = ToySlide([1] * 12 + [3] * 12 + [4] * 12 + [0] * 5, rng, cen)
    tgt = ToySlide([1] * 15 + [3] * 15 + [4] * 15 + [2] * 5 + [0] * 5, rng, cen)
    qn = M.roi_vector(src.X, M.draw_roi(src.tile_label, 3, "11"))
    qt = M.roi_vector(src.X, M.draw_roi(src.tile_label, 1, "11"))
    hn = L1.roi_map(tgt, qn, "cosine", "mean")
    ht = L1.roi_map(tgt, qt, "cosine", "mean")
    Mv = M.auroc_class(hn, tgt.patch_label, 3, _LC)
    Cv = M.auroc_class(ht, tgt.patch_label, 3, _LC)
    TM = M.auroc_class(ht, tgt.patch_label, 1, _LC)
    TC = M.auroc_class(hn, tgt.patch_label, 1, _LC)
    assert Mv > 0.5 > Cv
    assert Mv - Cv > 0
    assert TM > 0.5 > TC
    # score_map returns AUROC_1, AUROC_3, AUROC_4 of one map
    a1, a3, a4 = M.score_map(L1, _LC, tgt, qn, "cosine")
    assert (a1, a3) == (TC, Mv) and a4 < 0.5


def _toy_pairs():
    # 3 sources (patients A, B, C), 3 targets (X, Y, Z), class 3; M values chosen by hand
    rows = [("s1", "A", "t1", "X", 0.9, 0.3), ("s1", "A", "t2", "Y", 0.7, 0.4),
            ("s2", "B", "t1", "X", 0.8, 0.2), ("s2", "B", "t3", "Z", 0.6, 0.6),
            ("s3", "C", "t3", "Z", 0.5, 0.1)]
    pp = pd.DataFrame(rows, columns=["source", "source_patient", "target", "target_patient", "M", "C"])
    pp["D"] = pp.M - pp.C
    return pp


def test_both_aggregation_directions():
    pp = _toy_pairs()
    mask = np.ones(len(pp), bool)
    bt = M.aggregate(pp, "M", mask, "target").set_index("unit")
    assert bt.loc["t1", "value"] == pytest.approx(0.85) and bt.loc["t1", "n_rows"] == 2
    assert bt.loc["t2", "value"] == pytest.approx(0.7)
    assert bt.loc["t3", "value"] == pytest.approx(0.55)
    bs = M.aggregate(pp, "M", mask, "source").set_index("unit")
    assert bs.loc["s1", "value"] == pytest.approx(0.8)
    assert bs.loc["s2", "value"] == pytest.approx(0.7)
    assert bs.loc["s3", "value"] == pytest.approx(0.5)
    assert bt.loc["t1", "patient"] == "X" and bs.loc["s2", "patient"] == "B"
    e = M.estimate_both(L1, pp, "M", mask, 200, 123)
    assert e["by_target"]["point"] == pytest.approx((0.85 + 0.7 + 0.55) / 3)
    assert e["by_source"]["point"] == pytest.approx((0.8 + 0.7 + 0.5) / 3)
    assert e["n_targets"] == 3 and e["n_sources"] == 3 and e["n_pairs"] == 5
    assert e["by_target"]["n_pairs"] == 5 and e["by_source"]["n_patients"] == 3
    lo, hi = e["by_target"]["ci95_lo"], e["by_target"]["ci95_hi"]
    assert 0.55 - 1e-12 <= lo <= e["by_target"]["point"] <= hi <= 0.85 + 1e-12
    # D per target = mean M - mean C over the same rows
    d = M.aggregate(pp, "D", mask, "target").set_index("unit")
    assert d.loc["t1", "value"] == pytest.approx(0.85 - 0.25)
    # masked rows and NaN values are excluded
    pp.loc[0, "M"] = np.nan
    bt = M.aggregate(pp, "M", mask, "target").set_index("unit")
    assert bt.loc["t1", "value"] == pytest.approx(0.8) and bt.loc["t1", "n_rows"] == 1
    bt = M.aggregate(pp, "M", np.array([False, True, True, True, True]), "target")
    assert set(bt.unit) == {"t1", "t2", "t3"}


def test_bootstrap_determinism_and_seed():
    pp = _toy_pairs()
    mask = np.ones(len(pp), bool)
    a = M.estimate_both(L1, pp, "D", mask, 500, 123)
    b = M.estimate_both(L1, pp, "D", mask, 500, 123)
    c = M.estimate_both(L1, pp, "D", mask, 500, 124)
    for k in ("by_target", "by_source"):
        assert a[k] == b[k]
        assert a[k]["point"] == c[k]["point"]


def test_patient_distinctness():
    ok = {"1": "P1", "2": "P2", "3": "P3"}
    rec = M.assert_patients_distinct(ok, [("1", "2"), ("2", "3")])
    assert rec["pass"]
    with pytest.raises(AssertionError):
        M.assert_patients_distinct({"1": "P1", "2": "P1", "3": "P3"})
    with pytest.raises(AssertionError):
        M.assert_patients_distinct({"1": "P1", "2": "P2"}, [("1", "1")])
    assert not M.check_patients_distinct({"1": "P1", "2": "P1"}, [("1", "2")])["pass"]


def _e(lo, hi, point=None):
    return {"point": (lo + hi) / 2 if point is None else point, "ci95_lo": lo, "ci95_hi": hi, "n_patients": 15}


def test_rule_branches():
    both = lambda lo, hi: {"by_target": _e(lo, hi), "by_source": _e(lo, hi)}
    assert M.rule_branch(both(0.6, 0.8), both(0.1, 0.3), 15, 15) == "branch_1_follows"
    assert M.rule_branch(both(0.45, 0.7), both(0.05, 0.2), 15, 15) == "branch_2_shifted"
    assert M.rule_branch(both(0.6, 0.8), both(-0.05, 0.2), 15, 15) == "branch_3_not_distinguished"
    assert M.rule_branch(both(0.6, 0.8), both(-0.3, -0.1), 15, 15) == "branch_3_not_distinguished"
    assert M.rule_branch(both(0.3, 0.45), both(0.05, 0.2), 15, 15) == "not_covered"
    assert M.rule_branch(both(0.6, 0.8), both(0.1, 0.3), 11, 15) == "descriptive_only"
    assert M.rule_branch(both(0.6, 0.8), both(0.1, 0.3), 15, 11) == "descriptive_only"
    # both directions required: the source-patient interval alone failing moves the branch
    mixed_D = {"by_target": _e(0.1, 0.3), "by_source": _e(0.0, 0.3)}
    assert M.rule_branch(both(0.6, 0.8), mixed_D, 15, 15) == "branch_3_not_distinguished"
    mixed_M = {"by_target": _e(0.6, 0.8), "by_source": _e(0.5, 0.8)}
    assert M.rule_branch(mixed_M, both(0.1, 0.3), 15, 15) == "branch_2_shifted"
    s = M.fill_rule("branch_1_follows", 3, both(0.6, 0.8), both(0.3, 0.4), both(0.1, 0.3))
    assert s.startswith("A necrosis ROI ranked annotated necrosis") and "AUROC 0.70, 95% CI 0.60–0.80; 15" in s
    g = M.general_statement({3: "branch_1_follows", 4: "branch_2_shifted"})
    assert not g["general_non_tumor_statement_licensed"] and g["classes_passing_branch_1"] == ["necrosis"]
    assert M.general_statement({3: "branch_1_follows", 4: "branch_1_follows"})["general_non_tumor_statement_licensed"]
