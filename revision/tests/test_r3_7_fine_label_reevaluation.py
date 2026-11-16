"""Synthetic checks for scripts/r3_7_fine_label_reevaluation.py."""
import importlib.util
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "r37", os.path.join(HERE, "..", "scripts", "r3_7_fine_label_reevaluation.py"))
r37 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r37)


def fixture():
    """16 reference slides (8 patients x 2); queries s0..s7 = 4 patients, 2 classes."""
    ref = pd.DataFrame({"slide_id": [f"s{i}" for i in range(16)],
                        "patient": [f"P{i // 2}" for i in range(16)],
                        "label": ["A" if (i // 2) % 2 == 0 else "B" for i in range(16)]})
    queries = ref.iloc[:8]
    rows = []
    for _, q in queries.iterrows():
        others = ref[ref["patient"] != q["patient"]]
        order = pd.concat([others[others["label"] == q["label"]], others[others["label"] != q["label"]]])
        for rank, (_, n) in enumerate(order.head(10).iterrows(), start=1):
            rows.append({"query": q["slide_id"], "rank": rank, "neighbour": n["slide_id"],
                         "neighbour_label": n["label"], "distance": rank / 10})
    rl = pd.DataFrame(rows)
    truth = dict(zip(queries["slide_id"], queries["label"]))
    qpat = dict(zip(queries["slide_id"], queries["patient"]))
    return rl, truth, qpat, ref


def test_plurality_tie_break_closest_rank():
    assert r37.plurality(["A", "B", "B", "A", "C"]) == "A"
    assert r37.plurality(["B", "A", "A", "B", "C"]) == "B"
    assert r37.plurality(["C", "A", "A", "B", "B"]) == "A"


def test_strict_majority_abstain():
    assert r37.strict_majority(["A", "B", "A", "B", "C"]) == r37.ABSTAIN
    assert r37.strict_majority(["A", "B", "A", "B", "A"]) == "A"


def test_checks_pass_and_same_patient_raises():
    rl, truth, qpat, ref = fixture()
    lab, pat = dict(zip(ref.slide_id, ref.label)), dict(zip(ref.slide_id, ref.patient))
    assert r37.check_rank_list(rl, qpat, lab, pat, ["A", "B"]) == 80
    bad = rl.copy()
    bad.loc[0, ["neighbour", "neighbour_label"]] = ["s1", "A"]  # s0 and s1 are both P0
    with pytest.raises(ValueError, match="same-patient"):
        r37.check_rank_list(bad, qpat, lab, pat, ["A", "B"])
    short = rl[~((rl["query"] == "s3") & (rl["rank"] == 10))]
    with pytest.raises(ValueError, match="ranks"):
        r37.check_rank_list(short, qpat, lab, pat, ["A", "B"])


def test_library_support_and_rules():
    rl, truth, qpat, ref = fixture()
    df = r37.per_query_table(rl, truth, qpat, ref, pos="B").set_index("query")
    # A slides: P0,P2,P4,P6 x2 = 8; query s0 (P0) -> 6 with a different patient
    assert df.loc["s0", "library_support"] == 6
    assert df.loc["s2", "library_support"] == 6
    assert (df["hit1"] == 1).all() and (df["precision5"] == 1).all()
    assert (df["plurality5"] == df["truth"]).all() and (df["strict5"] == df["truth"]).all()
    assert (df["first_relevant_rank"] == 1).all()
    assert df.loc["s0", "pos_share5"] == 0 and df.loc["s2", "pos_share5"] == 1


def test_bootstrap_reproducible_and_auroc_separable():
    rl, truth, qpat, ref = fixture()
    df = r37.per_query_table(rl, truth, qpat, ref, pos="B")
    p1, m1 = r37.make_patient_draws(df["patient"].tolist(), count=200, seed=123)
    p2, m2 = r37.make_patient_draws(df["patient"].tolist(), count=200, seed=123)
    assert p1 == p2 == ("P0", "P1", "P2", "P3")
    assert np.array_equal(m1, m2) and (m1.sum(1) == 4).all()
    _, m3 = r37.make_patient_draws(df["patient"].tolist(), count=200, seed=124)
    assert not np.array_equal(m1, m3)
    W = m1[:, pd.Index(p1).get_indexer(df["patient"])].astype(float)
    y, s = (df["truth"] == "B").to_numpy(int), df["pos_share5"].to_numpy(float)
    assert roc_auc_score(y, s) == 1.0
    boot = r37.boot_auroc(W, y, s)
    assert np.all(boot[np.isfinite(boot)] == 1.0) and np.isfinite(boot).sum() > 100
    v = (df["truth"] == "A").to_numpy(float)
    lo_hi = r37.ci(r37.boot_means(W, v, np.ones(len(df), bool)))
    assert lo_hi == r37.ci(r37.boot_means(W, v, np.ones(len(df), bool)))
    assert 0 <= lo_hi[0] <= 0.5 <= lo_hi[1] <= 1 and lo_hi[2] == 0


def test_no_truth_leakage_into_predictions():
    rl, truth, qpat, ref = fixture()
    base = r37.per_query_table(rl, truth, qpat, ref).set_index("query")
    keys = sorted(truth)
    perm = dict(zip(keys, np.random.default_rng(0).permutation([truth[k] for k in keys])))
    flip = {k: "B" if v == "A" else "A" for k, v in truth.items()}
    for t in (perm, flip):
        alt = r37.per_query_table(rl, t, qpat, ref).set_index("query")
        for c in ("top1", "plurality5", "strict5"):
            assert alt[c].equals(base[c])


def test_ci_nonfinite_replicates_give_nan_and_count():
    lo, hi, bad = r37.ci(np.array([0.1, np.nan, 0.3, np.inf, 0.2]))
    assert np.isnan(lo) and np.isnan(hi) and bad == 2
    W = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 3.0]])  # draw 2 has zero support for the masked class
    lo, hi, bad = r37.ci(r37.boot_means(W, np.array([1.0, 0.0, 1.0]), np.array([True, True, False])))
    assert np.isnan(lo) and bad == 1


@pytest.fixture
def tiny_root(monkeypatch):
    root = os.path.join(HERE, "_r3_7_tiny_fixture")  # repository-local scratch, removed afterwards
    os.makedirs(os.path.join(root, "inv"), exist_ok=True)
    os.makedirs(os.path.join(root, "arm"), exist_ok=True)
    pd.DataFrame({"slide_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
                  "patient_id": ["P1", "P2", "P3", "PD", "PD", "P4"],
                  "label": ["A", "B", "A", "A", "B", "C"],
                  "sample_type": ["Patient tumor"] * 2 + ["Xenograft"] + ["Patient tumor"] * 3}
                 ).to_csv(os.path.join(root, "inv", "manifest_tiny.csv"), index=False)
    pd.DataFrame({"Tumor.group": ["GRP"] * 8 + ["OTHER"]}).to_csv(os.path.join(root, "meta.csv"), index=False)
    pd.DataFrame({"query": ["s1", "s2"], "rank": [1, 1], "neighbour": ["s2", "s1"], "neighbour_label": ["B", "A"],
                  "distance": [0.1, 0.1]}).to_csv(os.path.join(root, "arm", "ranked_lists_tiny.csv.gz"), index=False)
    monkeypatch.setattr(r37, "O", root)
    monkeypatch.setattr(r37, "INV", os.path.join(root, "inv"))
    monkeypatch.setattr(r37, "FROZEN_META", os.path.join(root, "meta.csv"))
    monkeypatch.setitem(r37.BROAD, "tiny", "GRP")
    yield root
    shutil.rmtree(root)


def test_load_populations_xenograft_and_discordant_patient(tiny_root):
    cfg = dict(d="arm", kind="internal", man="tiny", labels=["A", "B"], drop_patients=["PD"])
    pq = os.path.join(tiny_root, "arm", "per_query_tiny.csv")
    pd.DataFrame({"query": ["s1", "s2"], "true_label": ["A", "B"]}).to_csv(pq, index=False)
    _, _, qpop, ref, cov = r37.load_populations("tiny", cfg)
    assert sorted(ref["slide_id"]) == ["s1", "s2"] and qpop is ref
    assert (cov["manifest_rows"], cov["dropped_xenograft"], cov["dropped_discordant_patient"],
            cov["dropped_label_filter"]) == (6, 1, 2, 1)
    assert (cov["no_subtype"], cov["xenograft_excluded"], cov["no_fusion_status"]) == (2, 1, 0)
    assert "PD" in cov["note"]
    pd.DataFrame({"query": ["s1", "s3"], "true_label": ["A", "A"]}).to_csv(pq, index=False)  # s3 is a xenograft
    with pytest.raises(ValueError, match="not eligible"):
        r37.load_populations("tiny", cfg)
