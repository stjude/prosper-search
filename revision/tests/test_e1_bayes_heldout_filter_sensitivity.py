"""Tests for scripts/e1_bayes_heldout_filter_sensitivity.py ([analysis plan] E1-BAYHO gate F5; toy data only).

Kernel weights and the T = 20 scale, probabilities from the k nearest references with the (k + 1)st as scale,
exclusion of same-patient and sentinel cells, the closest-ranked plurality tie rule, metric definitions against
scikit-learn (with bootstrap multiplicities), corrected/introduced counts and the pre-registered case checks.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "e1_bayes_heldout_filter_sensitivity", os.path.join(R, "scripts", "e1_bayes_heldout_filter_sensitivity.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_kernel_weights_scale_and_degenerate_span():
    d = np.array([0.1, 0.2, 0.3, 0.4])
    w = M.kernel_weights(d, 3)
    assert w[0] == 1.0
    assert w[1] == pytest.approx(20 ** (-1 / 3))
    assert w[2] == pytest.approx(20 ** (-2 / 3))
    # a reference at the (k + 1)st distance would weigh 1/T
    assert np.exp(-np.log(20) / (d[3] - d[0]) * (d[3] - d[0])) == pytest.approx(1 / 20)
    assert np.array_equal(M.kernel_weights(np.array([0.5, 0.5, 0.5, 0.5]), 3), np.ones(3))


def _toy():
    ids = ["q", "a1", "b1", "a2", "c1", "qsib"]
    labels = ["a", "a", "b", "a", "c", "b"]
    patients = ["P0", "P1", "P2", "P3", "P4", "P0"]  # qsib is the query's patient
    x = np.full((6, 6), 0.9)
    x[0] = [1e6, 0.1, 0.2, 0.3, 0.4, 0.01]  # the same-patient slide is the closest and must be ignored
    np.fill_diagonal(x, 1e6)
    return pd.DataFrame(x, index=ids, columns=ids), ids, labels, patients


def test_probabilities_use_k_neighbors_and_exclude_same_patient():
    m, ids, labels, patients = _toy()
    conf = M.prepare(m, ids, labels, patients, ["a", "b", "c"], kmax=3)
    assert conf["dist"][0].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert conf["cells_masked_by_patient_rule_only"] == 2  # q-qsib and qsib-q
    p, ok = M.bayes(conf, 3)
    assert ok[0]
    w = np.array([1.0, 20 ** (-1 / 3), 20 ** (-2 / 3)])
    expect = np.array([w[0] + w[2], w[1], 0.0]) / w.sum()  # class c (the 4th neighbor) only sets the scale
    assert p[0] == pytest.approx(expect)
    assert M.argmax_pred(p, ok)[0] == 0


def test_query_without_enough_neighbors_is_not_returned():
    m, ids, labels, patients = _toy()
    m.loc["q", ["a2", "c1"]] = np.nan
    conf = M.prepare(m, ids, labels, patients, ["a", "b", "c"], kmax=3)
    p, ok = M.bayes(conf, 3)
    assert not ok[0] and np.isnan(p[0]).all()
    assert M.argmax_pred(p, ok)[0] == -1


def test_plurality_ties_go_to_closest_ranked_label():
    assert M.plurality_closest(np.array([2, 1, 1, 2, 0])) == 2
    assert M.plurality_closest(np.array([0, 1, 1])) == 1
    assert M.plurality_closest(np.array([3, 4, 5])) == 3


def test_metrics_match_sklearn_with_multiplicities():
    rng = np.random.default_rng(0)
    y, pred = rng.integers(0, 5, 300), rng.integers(0, 6, 300)  # class 5 predicted but never true
    w = rng.integers(0, 4, 300)
    yy, pp = np.repeat(y, w), np.repeat(pred, w)
    got = M.metrics(y, pred, 6, w.astype(float))
    assert got["accuracy"] == pytest.approx(accuracy_score(yy, pp))
    assert got["macro_f1"] == pytest.approx(f1_score(yy, pp, average="macro"))
    assert got["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(yy, pp))
    plain = M.metrics(y, pred, 6)
    assert plain["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))


def test_corrected_introduced_and_ratio():
    y = np.array([0, 0, 1, 1])
    bayes_pred = np.array([0, 1, 1, 0])
    vote = np.array([1, 0, 1, 1])
    p = np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.55, 0.45]])
    g = M.g_counts(y, bayes_pred, vote, p)
    assert (g["corrected"], g["introduced"]) == (1, 2)
    assert M.top_two_ratio(np.array([[1.0, 0.0]]))[0] == pytest.approx(1001.0)


def test_bootstrap_is_deterministic_and_whole_patient():
    y = np.array([0, 0, 1, 1, 1])
    preds = {"a": np.array([0, 1, 1, 1, 0])}
    patients = np.array(["p", "p", "q", "r", "r"])
    b1 = M.patient_bootstrap(y, preds, patients, 2, b=50, seed=7)
    b2 = M.patient_bootstrap(y, preds, patients, 2, b=50, seed=7)
    assert np.array_equal(b1["a"], b2["a"])
    # accuracies reachable only when patients are drawn whole: weights (2a, b, 2c) with a + b + c = 3
    reachable = {(a * 1 + b * 1 + c * 1) / (2 * a + b + 2 * c) for a in range(4) for b in range(4) for c in range(4) if a + b + c == 3}
    assert all(any(abs(v - r) < 1e-12 for r in reachable) for v in b1["a"][:, 0])


def test_case_reading_checks():
    k5 = {"probs": {"MPNST": 0.51, "RMS": 0.49}, "counts": {"MPNST": 3, "RMS": 2}}
    assert all(M.case_reading("SJMPNST076015_D1", k5, k5).values())
    wide = {"probs": {"RMS": 0.72, "MPNST": 0.25, "OS": 0.03}, "counts": {"RMS": 3, "MPNST": 2}}
    r = M.case_reading("SJMPNST076015_D1", wide, wide)
    assert r["MPNST and RMS top two at k=5"] and not r["top-two gap <= 0.10 at k=5"]
    k18 = {"probs": {"LIVER": 0.53, "MRT": 0.24, "RMS": 0.20}, "counts": {"LIVER": 8, "MRT": 2, "RMS": 6}}
    assert all(M.case_reading("SJHB075481_D1", k18, k18).values())
    d5 = {"probs": {"DSRCT": 0.7, "RMS": 0.16}, "counts": {"DSRCT": 1, "RMS": 2, "WT": 1, "MPNST": 1}}
    assert all(M.case_reading("SJDSRCT073174_D1", d5, d5).values())


def test_primary_prevalence_tie_rule():
    from scripts.e1_bayes_heldout_primary import plurality_prevalence

    frequencies = {"A": 10, "B": 30, "C": 100}
    assert plurality_prevalence(["A", "B", "B", "A", "C"], frequencies) == "B"
    assert plurality_prevalence(["A", "A", "A", "B", "C"], frequencies) == "A"
    assert plurality_prevalence(["B", "A"], {"A": 10, "B": 10}) == "A"


def test_primary_scalar_probability_route():
    from scripts.e1_bayes_heldout_primary import scalar_probabilities

    p = scalar_probabilities([0.1, 0.2, 0.3, 0.4], ["A", "B", "A", "C"], ["A", "B", "C"], 3)
    weights = np.array([1.0, 20 ** (-1 / 3), 20 ** (-2 / 3)])
    assert p == pytest.approx([(weights[0] + weights[2]) / weights.sum(), weights[1] / weights.sum(), 0])
    assert scalar_probabilities([0.4] * 4, ["A", "B", "A", "C"], ["A", "B", "C"], 3) == pytest.approx([2 / 3, 1 / 3, 0])
