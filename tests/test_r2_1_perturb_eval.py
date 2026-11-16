"""Synthetic fixtures for the R2.1 perturbation evaluator (SAMPLER -> L1 -> hit@k/P@k)."""
from __future__ import annotations

import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from prosper.evaluation import r2_1_hoptimus_main as K
from prosper.evaluation import r2_1_perturb_eval_cli as E
from prosper.evaluation import r2_1_perturb_sampler as S

D = 3  # tiny feature dimension, patched into the encoder table
P = list(S.PERCENTILES)

# 6 slides, 5 patients, 2 groups; s0/s1 share patient P0. Every tile of a slide carries the
# same constant value v, so SAMPLER = v everywhere and L1 = 10 * D * |v_i - v_j|.
TINY = pd.DataFrame({
    "population_index": range(6),
    "slide_id": [f"s{i}" for i in range(6)],
    "patient_id": ["P0", "P0", "P1", "P2", "P3", "P4"],
    "diagnostic_group": ["A", "A", "A", "B", "B", "A"],
})
IDENT = [0.0, 0.1, 5.0, 1.0, 6.0, 5.5]
SHIFTED = [5.2, 0.1, 5.0, 1.0, 6.0, 5.5]   # rotate_90: only s0 moves (next to s2/s5)


def _write_slide(root, sid, values: dict, n=4, listed=(), nan_rows=None):
    d = root / "uni2"
    d.mkdir(parents=True, exist_ok=True)
    feats = {}
    for c in E.CONDITIONS:
        x = np.full((n, D), values.get(c, values["identity"]), dtype=np.float32)
        if nan_rows and c in nan_rows:
            x[nan_rows[c]] = np.nan
        feats[c] = x
    data = pickle.dumps((feats, [(i, i) for i in range(n)]))
    (d / f"{sid}.pkl").write_bytes(data)
    rec = {"slide_id": sid, "encoder": "uni2", "conditions": list(E.CONDITIONS), "dim": D,
           "n_tiles": n, "pickle_sha256": hashlib.sha256(data).hexdigest(),
           "pickle_bytes": len(data), "reinhard_nonfinite_tile_indices": list(listed)}
    (d / f"{sid}.record.json").write_text(json.dumps(rec))


def _setup(tmp_path, monkeypatch, *, listed=None, nan_rows=None, pop=TINY, ident=IDENT,
           shifted=SHIFTED):
    monkeypatch.setitem(E.ENCODER_DIM, "uni2", D)
    feat = tmp_path / "features"
    for i, sid in enumerate(pop.slide_id):
        vals = {"identity": ident[i], "rotate_90": shifted[i]}
        _write_slide(feat, sid, vals, listed=(listed or {}).get(sid, ()),
                     nan_rows=(nan_rows or {}).get(sid))
    pop_p = tmp_path / "pop.csv"
    pop.to_csv(pop_p, index=False)
    return feat, pop_p


def _run(tmp_path, feat, pop_p, out="out", extra=()):
    return E.main(["run", "--encoders", "uni2", "--population", str(pop_p), "--features-root",
                   str(feat), "--out-root", str(tmp_path / out), "--draw-count", "200",
                   *extra])


# (i) SAMPLER ---------------------------------------------------------------------------------

def test_sampler_equals_np_percentile():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(257, 17)).astype(np.float32)
    r = S.sampler(x)
    ref = np.percentile(x, P, axis=0)
    assert r.embedding.dtype == np.float64 and r.embedding.shape == (10, 17)
    assert np.array_equal(r.embedding, ref)
    assert (r.n_tiles, r.n_excluded_listed, r.n_excluded_nonfinite, r.n_used) == (257, 0, 0, 257)


def test_sampler_excludes_listed_and_nonfinite_rows():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 6)).astype(np.float32)
    x[3, 2] = np.nan
    x[7, 0] = np.inf
    x[11] = np.nan          # listed and non-finite: counted as listed only
    r = S.sampler(x, exclude=[11, 20, 20])
    keep = np.setdiff1d(np.arange(50), [3, 7, 11, 20])
    assert np.array_equal(r.embedding, np.percentile(x[keep], P, axis=0))
    assert (r.n_excluded_listed, r.n_excluded_nonfinite, r.n_used) == (2, 2, 46)
    none = S.sampler(x[:2], exclude=[0, 1])
    assert none.embedding is None and none.n_used == 0
    with pytest.raises(ValueError):
        S.sampler(x, exclude=[50])


def test_blocked_l1_is_bitwise_cdist():
    rng = np.random.default_rng(2)
    q, r = rng.normal(size=(150, 40)), rng.normal(size=(90, 40))
    ref = cdist(q, r, metric="cityblock")
    assert np.array_equal(S.l1_distances(q, r, threads=1, block=7), ref)
    assert np.array_equal(S.l1_distances(q, r, threads=4, block=16), ref)


# (ii) sentinels ------------------------------------------------------------------------------

def test_distance_sentinels_self_and_same_patient(tmp_path, monkeypatch):
    feat, pop_p = _setup(tmp_path, monkeypatch)
    assert _run(tmp_path, feat, pop_p) == 0
    out = tmp_path / "out" / "uni2"
    for c in E.CONDITIONS:
        d = np.load(out / "distances" / f"{c}.npy")
        assert d.dtype == np.float32 and d.shape == (6, 6)
        assert np.all(np.diag(d) == S.SENTINEL)
        assert d[0, 1] == S.SENTINEL and d[1, 0] == S.SENTINEL
        other = ~np.eye(6, dtype=bool)
        other[0, 1] = other[1, 0] = False
        assert np.all(d[other] < S.SENTINEL)
    ident = np.load(out / "distances" / "identity.npy")
    assert ident[2, 5] == pytest.approx(10 * D * 0.5, rel=1e-6)
    rec = json.loads((out / "run_record.json").read_text())
    # identity-vs-identity diagonal is exactly 0 before the sentinel is applied
    assert rec["diagnostics"]["identity"]["diagonal_before_sentinel_max_abs"] == 0.0
    assert rec["diagnostics"]["identity"]["sentinel_cells"] == 8   # 6 self + 2 same-patient
    s = np.load(out / "sampler" / "identity.npy")
    assert s.dtype == np.float32 and s.shape == (6, 10, D)


def test_apply_sentinels_counts():
    d = np.zeros((3, 3))
    info = S.apply_sentinels(d, ["a", "a", "b"])
    assert info == {"self_cells": 3, "same_patient_offdiag_cells": 2}
    assert d[0, 1] == d[1, 0] == d[2, 2] == S.SENTINEL and d[0, 2] == 0


# (iii) hand-computed metrics, paired delta, bootstrap ------------------------------------------

def _est(out, cond, group, metric, k):
    e = pd.read_csv(out / "estimates.csv")
    row = e[(e.condition == cond) & (e.group == group) & (e.metric == metric) & (e.k == k)]
    assert len(row) == 1
    return row.iloc[0]


def test_hand_computed_hit_precision_and_paired_delta(tmp_path, monkeypatch):
    feat, pop_p = _setup(tmp_path, monkeypatch)
    assert _run(tmp_path, feat, pop_p) == 0
    out = tmp_path / "out" / "uni2"
    # identity top-1 (self + same patient excluded; s5 ties s2/s4 at 0.5 -> s2 by population order):
    # s0->s3 B miss, s1->s3 miss, s2->s5 hit, s3->s1 miss, s4->s5 miss, s5->s2 hit
    assert _est(out, "identity", "A", "hit", 1).estimate == pytest.approx(0.5)
    assert _est(out, "identity", "B", "hit", 1).estimate == pytest.approx(0.0)
    assert _est(out, "identity", "MACRO", "hit", 1).estimate == pytest.approx(0.25)
    # s2 top-3: s5(A) 0.5, s4(B) 1.0, s3(B) 4.0 -> P@3 = 1/3; s0 top-3: s3,s2,s5 -> 2/3
    q = pd.read_csv(out / "query_metrics.csv.gz")
    q3 = q[(q.condition == "identity") & (q.k == 3)].set_index("query")
    assert q3.loc["s2", "conditional_precision"] == pytest.approx(1 / 3)
    assert q3.loc["s0", "conditional_precision"] == pytest.approx(2 / 3)
    assert q3.loc["s0", "conditional_hit"] == 1
    # rotate_90 moves only s0 next to s2 -> s0 top-1 becomes a hit
    assert _est(out, "rotate_90", "A", "hit", 1).estimate == pytest.approx(0.75)
    assert _est(out, "rotate_90", "MACRO", "hit", 1).estimate == pytest.approx(0.375)
    dl = pd.read_csv(out / "paired_deltas.csv")
    r = dl[(dl.condition == "rotate_90") & (dl.metric == "hit") & (dl.k == 1)].set_index("group")
    assert r.loc["A", "delta"] == pytest.approx(0.25)
    assert r.loc["B", "delta"] == pytest.approx(0.0)
    assert r.loc["MACRO", "delta"] == pytest.approx(0.125)
    assert r.loc["MACRO", "n_pairs"] == 6
    assert set(dl.condition) == set(E.CONDITIONS) - {"identity"}
    # ties at the s5 top-1 boundary are reported
    assert q[(q.condition == "identity") & (q.k == 1) & (q["query"] == "s5")].boundary_tie.item() == 1
    rk = pd.read_csv(out / "ranked_lists.csv.gz")
    top = rk[(rk.condition == "identity") & (rk["query"] == "s5") & (rk["rank"] == 1)]
    assert top.neighbour.item() == "s2"


def test_bootstrap_matches_independent_patient_resampling():
    """CI of the evaluator == independent numpy replicate computation on the kernel's draws."""
    rng = np.random.default_rng(5)
    n_pat = 30
    rows = []
    for p in range(n_pat):
        for j in range(rng.integers(1, 4)):
            rows.append((f"P{p:02d}", ["A", "B", "C"][(p + j) % 3]))
    pop = pd.DataFrame({"population_index": range(len(rows)),
                        "slide_id": [f"s{i}" for i in range(len(rows))],
                        "patient_id": [r[0] for r in rows],
                        "diagnostic_group": [r[1] for r in rows]})
    n = len(pop)
    centers = {"A": 0.0, "B": 1.0, "C": 2.0}
    base = np.array([centers[g] for g in pop.diagnostic_group])[:, None, None]
    emb = {"identity": np.broadcast_to(base + rng.normal(0, 0.8, size=(n, 1, 1)),
                                       (n, 10, 2)).copy()}
    emb["rotate_90"] = emb["identity"] + rng.normal(0, 0.5, size=(n, 1, 1))
    res = E.evaluate(pop, emb, draw_count=300, seed=123)
    # independent: per-query hit@1 from the stored float64 ranking path
    dist = cdist(emb["identity"].reshape(n, -1), emb["identity"].reshape(n, -1), "cityblock")
    pat = pop.patient_id.to_numpy()
    hit = np.zeros(n)
    for i in range(n):
        cand = np.flatnonzero(pat != pat[i])
        j = cand[np.argsort(dist[i, cand], kind="mergesort")[0]]
        hit[i] = float(pop.diagnostic_group[j] == pop.diagnostic_group[i])
    draws = K.make_patient_draws(pop.patient_id.tolist(), count=300, seed=123)
    pidx = {p: i for i, p in enumerate(draws.patient_ids)}
    w = np.asarray(draws.multiplicities)[:, [pidx[p] for p in pat]]   # (B, n) slide weights
    groups = sorted(set(pop.diagnostic_group))
    reps = []
    for g in groups:
        m = (pop.diagnostic_group == g).to_numpy()
        reps.append((w[:, m] * hit[m]).sum(1) / w[:, m].sum(1))
    macro_rep = np.mean(reps, axis=0)
    lo, hi = np.quantile(macro_rep, [0.025, 0.975], method="linear")
    point = np.mean([hit[(pop.diagnostic_group == g).to_numpy()].mean() for g in groups])
    est = [r for r in res["estimates"]
           if r[0] == "identity" and r[1] == "MACRO" and r[2] == "hit" and r[3] == 1][0]
    assert est[4] == pytest.approx(point, abs=1e-12)
    assert est[5] == pytest.approx(lo, abs=1e-12) and est[6] == pytest.approx(hi, abs=1e-12)
    assert est[7] == n and est[8] == n_pat


# (iv) non-returned queries and coverage --------------------------------------------------------

def test_non_returned_query_and_coverage(tmp_path, monkeypatch):
    # every s3 tile is listed for reinhard -> s3 has no reinhard SAMPLER (non-returned query);
    # one non-finite brightness row in s4 is excluded and counted
    feat, pop_p = _setup(tmp_path, monkeypatch, listed={"s3": range(4), "s2": [1]},
                         nan_rows={"s4": {"brightness": [2]}})
    assert _run(tmp_path, feat, pop_p) == 0
    out = tmp_path / "out" / "uni2"
    cov = pd.read_csv(out / "coverage.csv", keep_default_na=False)
    a = cov[(cov.condition == "reinhard") & (cov.group == "ALL")].iloc[0]
    assert (a.intended, a.returned, a.non_returns, a.non_return_slides) == (6, 5, 1, "s3")
    assert a.tiles_excluded_listed == 5
    b = cov[(cov.condition == "reinhard") & (cov.group == "B")].iloc[0]
    assert b.coverage == pytest.approx(0.5)
    br = cov[(cov.condition == "brightness") & (cov.group == "ALL")].iloc[0]
    assert (br.returned, br.tiles_excluded_nonfinite) == (6, 1)
    idx = pd.read_csv(out / "sampler_index.csv")
    r = idx[(idx.condition == "reinhard") & (idx.slide_id == "s3")].iloc[0]
    assert (r.n_excluded_listed, r.n_used, r.returned) == (4, 0, 0)
    assert np.isnan(np.load(out / "sampler" / "reinhard.npy")[3]).all()
    # listed tiles affect the reinhard arm only
    r2 = idx[(idx.condition == "identity") & (idx.slide_id == "s3")].iloc[0]
    assert (r2.n_excluded_listed, r2.returned) == (0, 1)
    q = pd.read_csv(out / "query_metrics.csv.gz")
    s3 = q[(q.condition == "reinhard") & (q["query"] == "s3")]
    assert set(s3.status) == {"empty_query"}
    d = np.load(out / "distances" / "reinhard.npy")
    assert np.isnan(d[3]).all() and np.isfinite(d[[0, 1, 2, 4, 5]]).all()
    est = _est(out, "reinhard", "B", "hit", 1)
    assert (est.n_slides, est.population) == (1, "returned")
    dl = pd.read_csv(out / "paired_deltas.csv")
    m = dl[(dl.condition == "reinhard") & (dl.group == "MACRO") & (dl.metric == "hit")
           & (dl.k == 1)].iloc[0]
    assert (m.n_pairs, m.population) == (5, "both_returned")
    full = dl[(dl.condition == "rotate_90") & (dl.group == "MACRO")].iloc[0]
    assert full.n_pairs == 6


def test_non_returned_identity_leaves_reference_library(tmp_path, monkeypatch):
    pop = TINY.copy()
    monkeypatch.setitem(E.ENCODER_DIM, "uni2", D)
    feat = tmp_path / "features"
    for i, sid in enumerate(pop.slide_id):
        _write_slide(feat, sid, {"identity": IDENT[i]},
                     nan_rows={"identity": list(range(4))} if sid == "s5" else None)
    pop.to_csv(tmp_path / "pop.csv", index=False)
    assert _run(tmp_path, feat, tmp_path / "pop.csv") == 0
    out = tmp_path / "out" / "uni2"
    rk = pd.read_csv(out / "ranked_lists.csv.gz")
    assert "s5" not in set(rk.neighbour)
    # s2's nearest is now s4 (B): miss; s5 identity query is non-returned
    q = pd.read_csv(out / "query_metrics.csv.gz")
    assert q[(q.condition == "identity") & (q["query"] == "s5")].status.iloc[0] == "empty_query"


# (v) refusal on incomplete inputs ----------------------------------------------------------

def test_refuses_incomplete_inputs_without_allow_partial(tmp_path, monkeypatch, capsys):
    feat, pop_p = _setup(tmp_path, monkeypatch)
    (feat / "uni2" / "s4.record.json").unlink()      # pickle without record = incomplete
    assert _run(tmp_path, feat, pop_p) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    assert _run(tmp_path, feat, pop_p, extra=("--allow-partial",)) == 0
    rec = json.loads((tmp_path / "out" / "uni2" / "run_record.json").read_text())
    assert rec["partial"] is True and rec["allow_partial"] is True
    assert rec["missing_inputs"] == ["s4"]
    assert rec["population"]["intended_slides"] == 6
    assert rec["population"]["evaluated_slides"] == 5
    assert (tmp_path / "out" / "uni2" / "COMPLETE").is_file()


def test_rejects_tampered_pickle(tmp_path, monkeypatch):
    feat, pop_p = _setup(tmp_path, monkeypatch)
    p = feat / "uni2" / "s2.pkl"
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(E.EvalError, match="sha256"):
        _run(tmp_path, feat, pop_p)
    assert not (tmp_path / "out" / "uni2" / "COMPLETE").exists()


def test_record_with_wrong_conditions_counts_as_missing(tmp_path, monkeypatch):
    feat, pop_p = _setup(tmp_path, monkeypatch)
    r = feat / "uni2" / "s1.record.json"
    rec = json.loads(r.read_text())
    rec["conditions"] = rec["conditions"][:-1]
    r.write_text(json.dumps(rec))
    assert _run(tmp_path, feat, pop_p) == 2


# (vi) no overwrite --------------------------------------------------------------------------

def test_never_overwrites_existing_outputs(tmp_path, monkeypatch):
    feat, pop_p = _setup(tmp_path, monkeypatch)
    assert _run(tmp_path, feat, pop_p) == 0
    out = tmp_path / "out" / "uni2"
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    with pytest.raises(E.EvalError, match="refusing to overwrite"):
        _run(tmp_path, feat, pop_p)
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before
    w = E.Writer(tmp_path / "w")
    w.put("a.txt", b"1")
    with pytest.raises(FileExistsError):
        w.put("a.txt", b"2")
    assert (tmp_path / "w" / "a.txt").read_bytes() == b"1"


def test_complete_lists_every_output(tmp_path, monkeypatch):
    feat, pop_p = _setup(tmp_path, monkeypatch)
    assert _run(tmp_path, feat, pop_p, extra=("--conditions", "identity,rotate_90")) == 0
    out = tmp_path / "out" / "uni2"
    files = json.loads((out / "COMPLETE").read_text())["files"]
    on_disk = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()} - {"COMPLETE"}
    assert set(files) == on_disk
    for name, meta in files.items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == meta["sha256"]
    for need in ("estimates.csv", "paired_deltas.csv", "coverage.csv",
                 "historical_topk_accuracy.csv", "sampler_index.csv", "run_record.json",
                 "sampler/identity.npy", "distances/rotate_90.npy"):
        assert need in files


def test_conditions_must_include_identity(tmp_path, monkeypatch):
    feat, pop_p = _setup(tmp_path, monkeypatch)
    with pytest.raises(E.EvalError, match="identity"):
        _run(tmp_path, feat, pop_p, extra=("--conditions", "rotate_90"))


# historical secondary metric -----------------------------------------------------------------

def test_historical_predict_generous_rule_and_counter_order():
    # neighbours by argpartition(k=3): labels B, A, B for an A query -> modal B (miss)
    d = np.array([[0.1, 0.2, 0.3, 9.0]])
    preds, gen = S.historical_predict(d, ["A"], ["B", "A", "B", "C"], 3)
    assert preds == ["B"] and gen == [0]
    # neighbours A, B, C (one vote each) -> generous rule: A is among the modal classes
    preds, gen = S.historical_predict(d, ["A"], ["C", "B", "A", "D"], 3)
    assert preds == ["A"] and gen == [1]
    d2 = np.array([[0.1, 0.2, 0.3, 9.0]])
    preds, _ = S.historical_predict(d2, ["Z"], ["C", "D", "E", "F"], 3)
    assert preds[0] in {"C", "D", "E"}


def test_historical_keep_min_count_and_misc():
    labels = ["A"] * 40 + ["B"] * 39 + ["MISC"] * 50
    keep = S.historical_keep(labels)
    assert keep[:40].all() and not keep[40:].any()
    sub = S.historical_keep(["A", "B"], counts_from=labels)
    assert sub.tolist() == [True, False]


def test_historical_table_is_labelled_secondary(tmp_path, monkeypatch):
    feat, pop_p = _setup(tmp_path, monkeypatch)
    assert _run(tmp_path, feat, pop_p) == 0
    h = pd.read_csv(tmp_path / "out" / "uni2" / "historical_topk_accuracy.csv")
    assert h.label.str.startswith("SECONDARY").all()
    # groups have < 40 slides in this fixture -> no historical queries; table still written
    assert (h.n_queries == 0).all()


def test_worker_pool_and_threads_match_serial(tmp_path, monkeypatch):
    feat, pop_p = _setup(tmp_path, monkeypatch, listed={"s2": [1]})
    assert _run(tmp_path, feat, pop_p, out="serial") == 0
    assert _run(tmp_path, feat, pop_p, out="pool", extra=("--workers", "2", "--threads", "2")) == 0
    a, b = tmp_path / "serial" / "uni2", tmp_path / "pool" / "uni2"
    for name in ("estimates.csv", "paired_deltas.csv", "coverage.csv", "sampler_index.csv",
                 "historical_topk_accuracy.csv", "query_metrics.csv.gz", "ranked_lists.csv.gz"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    for c in E.CONDITIONS:
        assert (a / "sampler" / f"{c}.npy").read_bytes() == (b / "sampler" / f"{c}.npy").read_bytes()
        assert (a / "distances" / f"{c}.npy").read_bytes() == (b / "distances" / f"{c}.npy").read_bytes()
