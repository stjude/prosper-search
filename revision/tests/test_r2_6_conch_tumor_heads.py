import ast
import hashlib
import inspect
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import r2_6_conch_tumor_heads as M  # noqa: E402


def _source(path):
    with open(path) as fh:
        return fh.read()


def test_copied_objects_are_verbatim():
    for path, names in M.COPIED.items():
        src = _source(path)
        for name in names:
            assert inspect.getsource(getattr(M, name)) in src, name


def test_parameters_equal_r2_1_except_verbose():
    tree = ast.parse(_source(M.FIT_SRC))
    params = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "_PARAMETERS":
            params = ast.literal_eval(node.value.args[0])
    assert params is not None
    assert {**params, "verbose": 0} == M.PARAMETERS
    assert params["verbose"] == 1


def test_training_patch_hash_is_r2_1_analogue():
    payload = json.dumps(["r2_6_training_patch_1024_v1", 1234, "P1", "100", 0, 1024, 2048, 1024, 1],
                         ensure_ascii=True, separators=(",", ":")).encode()
    assert M.training_patch_hash("P1", "100", np.int64(1024), np.int64(2048), np.int64(1)) == \
        hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("n,k", [(1, 1), (19, 1), (20, 1), (23, 1), (30, 2), (50, 2), (70, 4), (853, 43)])
def test_five_percent_count(n, k):
    assert M.five_percent_count(n) == k


def _rows(seed=0, n=600):
    rng = np.random.default_rng(seed)
    pats = [f"P{i}" for i in range(20)]
    fold = {p: i % 5 for i, p in enumerate(pats)}
    p = rng.choice(pats, n)
    df = pd.DataFrame({"patient_id": p, "slide_id": [f"S{q[1:]}" for q in p],
                       "x": rng.integers(0, 50, n) * 1024, "y": rng.integers(0, 50, n) * 1024,
                       "label": rng.choice([1, 1, 1, 2, 3, 4, 5], n)})
    df = df.drop_duplicates(["slide_id", "x", "y"]).reset_index(drop=True)
    df["held_out_fold"] = df["patient_id"].map(fold)
    return df


def test_manifests_hash_order_nesting_and_weights():
    rows = _rows()
    order, hashes = M.hash_order(rows)
    assert hashes == sorted(hashes)
    man = M.build_manifests(rows, order)
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    for f in range(5):
        full, five = man[(f, 100)], man[(f, 5)]
        assert (rows["held_out_fold"].to_numpy()[full["idx"]] != f).all()
        assert set(five["idx"]) <= set(full["idx"])
        assert (np.diff(rank[full["idx"]]) > 0).all() and (np.diff(rank[five["idx"]]) > 0).all()
        lab = rows["label"].to_numpy()
        for cs in five["class_stats"]:
            c = cs["tissue_label"]
            pool_c = full["idx"][lab[full["idx"]] == c]
            sel_c = five["idx"][lab[five["idx"]] == c]
            assert len(sel_c) == M.five_percent_count(len(pool_c))
            assert np.array_equal(sel_c, pool_c[:len(sel_c)])  # first in hash order
            assert cs["balanced_weight"] == len(five["idx"]) / (5 * len(sel_c))
        assert full["selected_count"] == full["pool_count"]


def test_absent_class_raises():
    rows = _rows()
    rows = rows[~((rows["label"] == 2) & (rows["held_out_fold"] != 0))].reset_index(drop=True)
    order, _ = M.hash_order(rows)
    with pytest.raises(M.ManifestValidationError):
        M.build_manifests(rows, order)


def test_fit_one_and_fingerprint_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    y = np.repeat(np.array(M.CLASSES), 40)
    centres = rng.normal(0, 3, (5, M.WIDTH)).astype(np.float32)
    X = (centres[y - 1] + rng.normal(0, 1, (len(y), M.WIDTH))).astype(np.float32)
    p = tmp_path / "f.npy"
    np.save(p, X)
    idx = np.arange(len(y))
    w = {c: len(y) / (5 * 40) for c in M.CLASSES}
    f, b, blob, d = M.fit_one((0, 100, str(p), idx, y, w))
    est = pickle.loads(blob)
    assert est.classes_.tolist() == [1, 2, 3, 4, 5] and est.coef_.dtype == np.float32
    assert M._fingerprint(est) == d["fitted_state_sha256"]
    assert est.get_params()["class_weight"] == w and est.get_params()["random_state"] == 1234
    _, _, blob2, d2 = M.fit_one((0, 100, str(p), idx, y, w))
    assert d2["fitted_state_sha256"] == d["fitted_state_sha256"]  # deterministic
    assert (est.predict(X) == y).mean() > 0.95


def test_heldout_metrics_small_example():
    truth = np.array([1, 1, 1, 4, 4, 3])
    proba = np.zeros((6, 5))
    for i, c in enumerate([1, 1, 4, 4, 1, 3]):
        proba[i, c - 1] = 0.95 if i != 1 else 0.6
        proba[i, (c % 5)] += 1 - proba[i, c - 1]
    pats = np.array(["a", "a", "a", "b", "b", "b"])
    m, pp = M.heldout_metrics(truth, proba, pats)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["tumor_argmax_tp"] == 2 and m["tumor_argmax_fn"] == 1 and m["tumor_argmax_fp"] == 1
    assert m["tumor_argmax_sensitivity"] == pytest.approx(2 / 3)
    assert m["tumor_argmax_specificity"] == pytest.approx(2 / 3)
    assert m["tumor_strict0.9_tp"] == 1  # row 1 has p_tumor 0.6
    assert m["recall_2"] != m["recall_2"]  # NaN: no Immune support
    cm = json.loads(m["confusion_rows_truth_cols_pred_1to5"])
    assert cm[0] == [2, 0, 0, 1, 0] and cm[3] == [1, 0, 0, 1, 0]
    assert [r["tumor_vs_rest_accuracy"] for r in pp] == [pytest.approx(2 / 3), pytest.approx(2 / 3)]


class _Const:
    def __init__(self, v):
        self.v = np.asarray(v, np.float32)

    def predict_proba(self, X):
        return np.tile(self.v, (len(X), 1))


def test_routing_oof_and_mean():
    heads = {(f, b): _Const(np.eye(5)[f] * 0.6 + 0.08) for f in range(5) for b in M.BUDGETS}
    X = np.zeros((3, M.WIDTH), np.float32)
    p, src = M.route_probabilities(heads, 100, X, 2)
    assert src == "oof_fold2" and np.allclose(p[0], heads[(2, 100)].v)
    p, src = M.route_probabilities(heads, 5, X, None)
    assert src == "mean5" and np.allclose(p[0], 0.2)
    assert M.argmax_label(p).tolist() == [1, 1, 1]  # exact tie -> lowest label


def test_selection_tables():
    scores = {"s1": {"x": np.array([0, 1024, 2048]), "y": np.zeros(3, int),
                     "p_tumor_b100": np.array([0.95, 0.6, 0.1]), "pred_b100": np.array([1, 1, 4]),
                     "p_tumor_b5": np.array([0.5, 0.2, 0.1]), "pred_b5": np.array([1, 3, 4]),
                     "head_source": "mean5"}}
    kept, summ = M.selection_tables(scores, ["s1"])
    assert kept["b100_argmax"]["x"].tolist() == [0, 1024]
    assert kept["b5_argmax"]["x"].tolist() == [0]
    assert kept["b100_strict"]["x"].tolist() == [0]
    s = {r["variant"]: r for r in summ}
    assert s["b100_argmax"]["n_kept"] == 2 and s["b100_argmax"]["frac_kept"] == pytest.approx(2 / 3)
    assert not s["b100_argmax"]["ge8"]
