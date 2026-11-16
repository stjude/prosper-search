"""Synthetic fixtures for the R2.1 F2 filtered-retrieval evaluator."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from prosper.evaluation import r2_1_filter_eval_cli as E

# 8 slides, 6 patients, 3 groups. P1 carries two labels (slides 0:A and 3:B).
POP = pd.DataFrame({
    "query_index": range(8), "reference_index": range(8), "metadata_index": range(8),
    "slide_id": [f"s{i}" for i in range(8)],
    "patient_id": ["P1", "P2", "P3", "P1", "P4", "P5", "P6", "P6"],
    "diagnostic_group": ["A", "A", "B", "B", "C", "C", "A", "B"],
    "fold": [0] * 8,
})


def _matrix(seed=0, ids=None):
    ids = ids or POP.slide_id.tolist()
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.1, 0.9, size=(len(ids), len(ids)))
    np.fill_diagonal(d, 1e6)
    pat = POP.set_index("slide_id").patient_id.reindex(ids).to_numpy()
    same = (pat[:, None] == pat[None, :]) & ~np.eye(len(ids), dtype=bool)
    d[same] = 1e6
    return pd.DataFrame(d, index=ids, columns=ids)


def _setup(tmp_path, base=None, filt=None, written=None):
    pop_p = tmp_path / "pop.csv"
    POP.to_csv(pop_p, index=False)
    base = _matrix(0) if base is None else base
    bp = tmp_path / "base.pkl"
    pickle.dump(base, open(bp, "wb"))
    arms = [E.Arm("all_tissue", "ot_cosine", "", bp, None)]
    if written is not None:
        ids = [s for s, w in zip(POP.slide_id, written) if w]
        filt = _matrix(1, ids) if filt is None else filt
        fp = tmp_path / "filt.pkl"
        pickle.dump(filt, open(fp, "wb"))
        sp = tmp_path / "summary.csv"
        pd.DataFrame({"slide_id": POP.slide_id, "written": [int(w) for w in written]}).to_csv(sp, index=False)
        arms.append(E.Arm("both_filtered", "ot_cosine", "5", fp, sp))
    return E.load_population(pop_p), arms


def _est(res, **kw):
    df = pd.DataFrame(res["estimates"] + res["contrasts"], columns=E.ESTIMATE_HEADER)
    for k, v in kw.items():
        df = df[df[k] == v]
    return df


def test_hand_ranking_same_patient_exclusion_and_cross_label_patient(tmp_path):
    base = _matrix(0)
    # s0 (P1, A): nearest is its own patient's s3 -- must be excluded -- then s1 (A).
    base.loc["s0"] = [1e6, 0.30, 0.50, 1e6, 0.60, 0.70, 0.80, 0.90]
    pop, arms = _setup(tmp_path, base=base)
    res = E.evaluate(pop, arms, draw_count=50)
    ranks = [r for r in res["rank_rows"] if r[3] == "s0"]
    assert [r[5] for r in ranks] == ["s1", "s2", "s4", "s5", "s6", "s7"]   # 6 eligible, s3 excluded
    q = pd.DataFrame(res["query_rows"], columns=E.QUERY_HEADER)
    r0 = q[(q["query"] == "s0") & (q.k == 10)].iloc[0]
    assert r0.short_list == 1 and r0.returned_depth == 6 and r0.conditional_precision == 2 / 10
    assert res["matrix_diagnostics"][arms[0].key]["used_1e6_entries"] == 0
    assert res["matrix_diagnostics"][arms[0].key]["same_patient_cells_not_1e6"] == 0
    # the two-label patient P1 is one bootstrap unit: draws have 6 patient columns
    assert res["full_draws"].multiplicities.shape == (50, 6)
    # group A point of hit@1 = mean over its 3 queries of their own hits
    hit1 = q[q.k == 1].groupby("diagnostic_group").conditional_hit.mean()
    g = _est(res, row_kind="estimate", endpoint="hit", k=1, level="group").set_index("group").point
    assert np.allclose(g.loc[list(hit1.index)].to_numpy(), hit1.to_numpy())
    macro = _est(res, row_kind="estimate", endpoint="hit", k=1, level="macro").point.iloc[0]
    assert macro == pytest.approx(hit1.mean())


def test_non_return_counts_in_denominator_only(tmp_path):
    written = [True, True, True, True, True, True, False, True]          # s6 (A, P6) not written
    pop, arms = _setup(tmp_path, written=written)
    res = E.evaluate(pop, arms, draw_count=50)
    q = pd.DataFrame(res["query_rows"], columns=E.QUERY_HEADER)
    s6 = q[(q.arm == "both_filtered") & (q["query"] == "s6")]
    assert set(s6.status) == {"empty_query"} and s6.coverage.sum() == 0 and s6.hit_yield.sum() == 0
    # s6 never appears as a reference in the filtered arm
    assert not any(r[0] == "both_filtered" and r[5] == "s6" for r in res["rank_rows"])
    hit = _est(res, row_kind="estimate", arm="both_filtered", endpoint="hit", k=5, level="group", group="A")
    assert hit.n_queries.iloc[0] == 2 and hit.population.iloc[0] == "returned"
    yld = _est(res, row_kind="estimate", arm="both_filtered", endpoint="hit_yield", k=5, level="group",
               group="A")
    assert yld.n_queries.iloc[0] == 3 and yld.population.iloc[0] == "intended"
    cov = pd.DataFrame(res["coverage"], columns=E.COVERAGE_HEADER)
    a = cov[(cov.arm == "both_filtered") & (cov.group == "A")].iloc[0]
    assert (a.intended, a.returned, a.non_returns, a.non_return_slides) == (3, 2, 1, "s6")
    con = _est(res, row_kind="contrast", endpoint="hit", k=5, level="group", group="A")
    assert con.n_queries.iloc[0] == 2 and con.population.iloc[0] == "both_returned"
    # group A both-returned queries s0 (P1) and s1 (P2); P6 stays in the draw table through s7 (B)
    assert con.n_patients.iloc[0] == 2
    macro = _est(res, row_kind="contrast", endpoint="hit", k=5, level="macro")
    assert macro.n_patients.iloc[0] == 6 and macro.n_queries.iloc[0] == 7
    # filtered-arm yield in group A <= coverage-limited maximum 2/3
    assert yld.point.iloc[0] <= 2 / 3 + 1e-12


def test_baseline_vs_baseline_contrast_is_exactly_zero(tmp_path):
    pop, arms = _setup(tmp_path)
    copy = E.Arm("baseline_copy", "ot_cosine", "", arms[0].matrix, None)
    res = E.evaluate(pop, [arms[0], copy], draw_count=100)
    con = _est(res, row_kind="contrast")
    est = con[con.point_status == "estimable"]
    assert len(con) == len(E.CUTOFFS) * 3 * 4 and (est.point == 0).all()
    ci = con[con.ci_status == "estimable"]
    assert (ci.ci_low == 0).all() and (ci.ci_high == 0).all()


def test_used_sentinel_and_axis_mismatch_rejected(tmp_path):
    base = _matrix(0)
    base.loc["s1", "s2"] = 1e6                   # a failed pair on a different-patient cell
    pop, arms = _setup(tmp_path, base=base)
    with pytest.raises(E.EvalError, match="sentinel"):
        E.evaluate(pop, arms, draw_count=20)
    res = E.evaluate(pop, arms, draw_count=20, allow_used_sentinel=True)
    assert res["matrix_diagnostics"][arms[0].key]["used_1e6_entries"] == 1
    written = [True] * 7 + [False]
    pop, arms = _setup(tmp_path, written=written, filt=_matrix(1))   # filtered axes include s7
    with pytest.raises(E.EvalError, match="differ from the"):
        E.evaluate(pop, arms, draw_count=20)


def test_ties_break_by_population_order(tmp_path):
    base = _matrix(0)
    base.loc["s1"] = [0.5, 1e6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    pop, arms = _setup(tmp_path, base=base)
    res = E.evaluate(pop, arms, draw_count=20)
    ranks = [r[5] for r in res["rank_rows"] if r[3] == "s1"]
    assert ranks == ["s0", "s2", "s3", "s4", "s5", "s6", "s7"]
    q = pd.DataFrame(res["query_rows"], columns=E.QUERY_HEADER)
    assert q[(q["query"] == "s1") & (q.k == 1)].boundary_tie.iloc[0] == 1


def test_write_outputs_never_overwrites(tmp_path):
    pop, arms = _setup(tmp_path, written=[True] * 8)
    res = E.evaluate(pop, arms, draw_count=20)
    out = tmp_path / "eval"
    files = E.write_outputs(res, out, {"note": "fixture"})
    assert {"estimates.csv", "contrasts.csv", "coverage.csv", "query_metrics.csv.gz",
            "ranked_lists.csv.gz", "patient_draws.npy", "summary.json"}.issubset(files)
    est = pd.read_csv(out / "estimates.csv")
    assert tuple(est.columns) == E.ESTIMATE_HEADER
    reps = np.load(out / "replicates.npy")
    assert reps.shape == (len(res["estimates"]) + len(res["contrasts"]), 20)
    with pytest.raises(E.EvalError, match="not empty"):
        E.write_outputs(res, out, {})


# ---------------------------------------------------------------- query-fold design ([decision record] B)
import itertools  # noqa: E402

# 10 slides, two folds; patients are fold-pure; P1 carries two labels (s0:A, s1:B).
POPQ = pd.DataFrame({
    "query_index": range(10), "reference_index": range(10), "metadata_index": range(10),
    "slide_id": [f"s{i}" for i in range(10)],
    "patient_id": ["P1", "P1", "P2", "P3", "P4", "P5", "P6", "P6", "P7", "P8"],
    "diagnostic_group": ["A", "B", "A", "B", "C", "A", "B", "C", "C", "A"],
    "fold": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
})
IDSQ = POPQ.slide_id.tolist()


def _full(seed):
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.1, 0.9, size=(10, 10))
    np.fill_diagonal(d, 1e6)
    return d


def _masked_frame(d):
    pat = POPQ.patient_id.to_numpy()
    d = d.copy()
    d[(pat[:, None] == pat[None, :]) & ~np.eye(10, dtype=bool)] = 1e6
    return pd.DataFrame(d, index=IDSQ, columns=IDSQ)


def _qf_fixture(tmp_path, written, dist=None, arms=("both", "query"), methods=("ot",), budgets=("5",)):
    """written[budget][fold] -> 10 bools (head `fold` keeps >= min tiles for each slide)."""
    pop_p = tmp_path / "popq.csv"
    POPQ.to_csv(pop_p, index=False)
    pop = E.load_population(pop_p)
    dist = _full(7) if dist is None else dist
    banks, rows = tmp_path / "banks", tmp_path / "rows"
    for b in budgets:
        for f in (0, 1):
            w = np.asarray(written[b][f], dtype=bool)
            d = banks / f"budget{b}" / f"fold{f}"
            d.mkdir(parents=True)
            pd.DataFrame({"slide_id": IDSQ, "head_fold": f, "slide_fold": POPQ.fold,
                          "written": w.astype(int), "own_fold_query": (POPQ.fold == f).astype(int)}
                         ).to_csv(d / "_filter_summary.csv", index=False)
            for arm in arms:
                lib = w if arm == "both" else np.ones(10, dtype=bool)
                for m in methods:
                    rd = rows / f"budget{b}" / f"fold{f}" / f"{arm}_{m}"
                    rd.mkdir(parents=True)
                    for i in np.flatnonzero(w & (POPQ.fold.to_numpy() == f)):
                        row = [(IDSQ[j], float(dist[i, j])) for j in np.flatnonzero(lib)]
                        pickle.dump(row, open(rd / f"similarity_{IDSQ[i]}.pkl", "wb"))
    return pop, banks, rows


ALL = {"5": {0: [1] * 10, 1: [1] * 10}, "100": {0: [1] * 10, 1: [1] * 10}}


def _combine(pop, banks, rows, out, arm="both_filtered", method="ot", budget="5"):
    return E.combine_qf(pop=pop, arm=arm, method=method, budget=budget, rows_root=rows,
                        banks_root=banks, out_dir=out)


def test_combine_qf_per_fold_libraries_nan_outside_and_same_patient_mask(tmp_path):
    # head 0 drops s3 (own fold 0 -> non-returned query) and s8 (fold-1 slide, absent from the
    # fold-0 library only); head 1 drops s6 (own fold 1 -> non-returned) but keeps s3 and s8.
    w0 = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    w1 = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    d = _full(3)
    pop, banks, rows = _qf_fixture(tmp_path, {"5": {0: w0, 1: w1}}, dist=d)
    rec = _combine(pop, banks, rows, tmp_path / "m")
    raw = pickle.load(open(tmp_path / "m" / E.qf_matrix_name("5", "both_filtered", "ot_cosine", False), "rb"))
    msk = pickle.load(open(tmp_path / "m" / E.qf_matrix_name("5", "both_filtered", "ot_cosine", True), "rb"))
    assert raw.index.tolist() == ["s0", "s1", "s2", "s4", "s5", "s7", "s8", "s9"]   # s3, s6 non-returned
    assert raw.columns.tolist() == IDSQ
    assert np.isnan(raw.loc["s0", "s3"]) and np.isnan(raw.loc["s0", "s8"])          # fold-0 library
    assert raw.loc["s5", "s3"] == d[5, 3] and raw.loc["s5", "s8"] == d[5, 8]       # fold-1 library
    assert np.isnan(raw.loc["s5", "s6"]) and raw.loc["s0", "s6"] == d[0, 6]
    assert raw.loc["s0", "s0"] == 1e6 and raw.loc["s0", "s1"] == d[0, 1]
    assert msk.loc["s0", "s1"] == 1e6 and msk.loc["s7", "s6"] != 1e6 and np.isnan(msk.loc["s7", "s6"])
    side = pd.read_csv(tmp_path / "m" / E.qf_rows_name("5", "both_filtered", "ot_cosine"))
    assert side.query_fold.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert side.library_size.tolist() == [8] * 4 + [9] * 4
    assert side.library_id.tolist() == ["bank_b5_fold0"] * 4 + ["bank_b5_fold1"] * 4
    assert rec["folds"]["0"]["non_returned"] == ["s3"] and rec["folds"]["1"]["excluded"] == ["s6"]
    assert rec["same_patient_cells_masked"] == 2          # s0->s1, s1->s0; s7->s6 is outside the library
    q = _combine(pop, banks, rows, tmp_path / "mq", arm="query_only")
    rawq = pickle.load(open(tmp_path / "mq" / E.qf_matrix_name("5", "query_only", "ot_cosine", False), "rb"))
    assert not np.isnan(rawq.to_numpy()).any() and q["folds"]["0"]["size"] == 10
    with pytest.raises(FileExistsError):                  # never overwrites
        _combine(pop, banks, rows, tmp_path / "m")


def test_combine_qf_fails_loudly_on_gaps_extras_and_library_mismatch(tmp_path):
    pop, banks, rows = _qf_fixture(tmp_path, {"5": {0: [1] * 10, 1: [1] * 10}}, arms=("both",))
    rd = rows / "budget5" / "fold1" / "both_ot"
    (rd / "similarity_s9.pkl").rename(tmp_path / "s9.pkl")
    with pytest.raises(E.EvalError, match="1 missing"):
        _combine(pop, banks, rows, tmp_path / "a")
    (tmp_path / "s9.pkl").rename(rd / "similarity_s9.pkl")
    pickle.dump([], open(rd / "similarity_s0.pkl", "wb"))            # fold-0 query in fold-1 dir
    with pytest.raises(E.EvalError, match="unexpected"):
        _combine(pop, banks, rows, tmp_path / "b")
    (rd / "similarity_s0.pkl").unlink()
    row = pickle.load(open(rd / "similarity_s5.pkl", "rb"))
    pickle.dump(row[:-1], open(rd / "similarity_s5.pkl", "wb"))
    with pytest.raises(E.EvalError, match="not exactly the fold-1 library"):
        _combine(pop, banks, rows, tmp_path / "c")
    pickle.dump([(o, 0.5 if o == "s5" else v) for o, v in row], open(rd / "similarity_s5.pkl", "wb"))
    with pytest.raises(E.EvalError, match="self distance"):
        _combine(pop, banks, rows, tmp_path / "d")
    assert not (tmp_path / "a").exists() or not any((tmp_path / "a").iterdir())


def _qf_arms(tmp_path, pop, banks, rows, d_base, budgets=("5",), methods=("ot",)):
    bp = tmp_path / "base.pkl"
    pickle.dump(_masked_frame(d_base), open(bp, "wb"))
    arms = [E.QfArm("all_tissue", E.QF_METHOD_TAG[m], "", bp, None) for m in methods]
    for b in budgets:
        for name in ("query_only", "both_filtered"):
            for m in methods:
                out = tmp_path / "mat"
                _combine(pop, banks, rows, out, arm=name, method=m, budget=b)
                tag = E.QF_METHOD_TAG[m]
                arms.append(E.QfArm(name, tag, b, out / E.qf_matrix_name(b, name, tag, True),
                                    out / E.qf_rows_name(b, name, tag)))
    return arms


def test_qf_all_returned_same_distances_give_zero_contrasts_and_baseline_estimates(tmp_path):
    d = _full(5)
    pop, banks, rows = _qf_fixture(tmp_path, ALL, dist=d, budgets=("5", "100"))
    arms = _qf_arms(tmp_path, pop, banks, rows, d, budgets=("5", "100"))
    res = E.evaluate_qf(pop, arms, banks_root=banks, draw_count=40)
    con = pd.DataFrame(res["contrasts"], columns=E.ESTIMATE_HEADER)
    assert len(res["contrast_pairs"]) == 8                # 2x(q-all, both-all, q-both) + 2 budget pairs
    assert np.all(con.point.to_numpy() == 0.0)
    assert np.all(np.nan_to_num(con.ci_low.to_numpy()) == 0.0)
    est = pd.DataFrame(res["estimates"], columns=E.ESTIMATE_HEADER)
    ref = est[est.arm == "all_tissue"].reset_index(drop=True)
    for name in ("query_only", "both_filtered"):
        x = est[(est.arm == name) & (est.budget == "5")].reset_index(drop=True)
        assert np.array_equal(x.point.to_numpy(), ref.point.to_numpy())
        assert np.array_equal(x.ci_low.to_numpy(), ref.ci_low.to_numpy())
    assert not ((est.endpoint == "precision") & (est.k == 1)).any()
    assert ((est.endpoint == "hit") & (est.k == 1)).any()


def test_qf_estimator_matches_the_existing_evaluator_on_the_baseline(tmp_path):
    d = _full(9)
    pop_p = tmp_path / "popq.csv"
    POPQ.to_csv(pop_p, index=False)
    pop = E.load_population(pop_p)
    bp = tmp_path / "b.pkl"
    pickle.dump(_masked_frame(d), open(bp, "wb"))
    old = E.evaluate(pop, [E.Arm("all_tissue", "ot_cosine", "", bp, None)], draw_count=60)
    new = E.evaluate_qf(pop, [E.QfArm("all_tissue", "ot_cosine", "", bp, None)], banks_root=None,
                        draw_count=60)
    key = ["endpoint", "k", "level", "group", "population"]
    o = pd.DataFrame(old["estimates"], columns=E.ESTIMATE_HEADER).set_index(key)
    n = pd.DataFrame(new["estimates"], columns=E.ESTIMATE_HEADER).set_index(key)
    assert len(n) == len(o) - 4                          # precision@1 (3 groups + macro) omitted
    for c in ("point", "ci_low", "ci_high", "n_queries", "n_patients"):
        assert n[c].equals(o.loc[n.index, c])
    assert old["rank_rows"] == new["rank_rows"]


def test_qf_per_row_library_and_own_fold_non_return(tmp_path):
    w0 = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1]       # head 0 drops s3 (own fold) and s8 (other fold)
    w1 = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]       # head 1 drops s6 (own fold)
    d = _full(11)
    d[0, 8] = d[0, 3] = 0.01                  # closest to s0 -- but outside fold 0's library
    d[5, 3] = 0.01                            # closest to s5 -- inside fold 1's library
    pop, banks, rows = _qf_fixture(tmp_path, {"5": {0: w0, 1: w1}}, dist=d)
    arms = _qf_arms(tmp_path, pop, banks, rows, d)
    res = E.evaluate_qf(pop, arms, banks_root=banks, draw_count=30)
    rk = pd.DataFrame(res["rank_rows"], columns=E.RANK_HEADER)
    both = rk[rk.arm == "both_filtered"]
    assert not both[both["query"].isin(["s0", "s1", "s2", "s4"])].neighbour.isin(["s3", "s8"]).any()
    assert both[(both["query"] == "s5") & (both["rank"] == 1)].neighbour.iloc[0] == "s3"
    qo = rk[(rk.arm == "query_only") & (rk["query"] == "s0") & (rk["rank"] == 1)]
    assert qo.neighbour.iloc[0] in ("s3", "s8")                # unfiltered library
    q = pd.DataFrame(res["query_rows"], columns=E.QUERY_HEADER_QF)
    st = q[(q.k == 1) & (q.arm == "both_filtered")].set_index("query").status
    assert st["s3"] == "empty_query" and st["s6"] == "empty_query" and st["s8"] == "returned"
    lib = q[(q.k == 1) & (q.arm == "both_filtered") & (q.status == "returned")].set_index("query").library_size
    assert lib["s0"] == 8 and lib["s5"] == 9
    libs = pd.DataFrame(res["libraries"], columns=E.LIBRARY_HEADER)
    b = libs[libs.arm == "both_filtered"].set_index("query_fold")
    assert b.library_size.to_dict() == {0: 8, 1: 9} and b.library_excluded_slides[0] == "s3;s8"
    cov = pd.DataFrame(res["coverage"], columns=E.COVERAGE_HEADER_QF)
    allrow = cov[(cov.arm == "both_filtered") & (cov.group == "ALL")].iloc[0]
    assert allrow.level == "pooled_slides_NOT_macro" and allrow.returned == 8
    # a matrix whose NaN pattern disagrees with the declared library is rejected
    m = pickle.load(open(arms[2].matrix, "rb"))
    m.loc["s0", "s8"] = 0.5
    bad = tmp_path / "bad.pkl"
    pickle.dump(m, open(bad, "wb"))
    with pytest.raises(E.EvalError, match="non-NaN cells differ"):
        E.evaluate_qf(pop, [E.QfArm("both_filtered", "ot_cosine", "5", bad, arms[2].rows)],
                      banks_root=banks, draw_count=5)


def test_budget_contrast_uses_common_returned_population(tmp_path):
    w = {"5": {0: [1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 1: [1] * 10},
         "100": {0: [1] * 10, 1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]}}
    d = _full(13)
    pop, banks, rows = _qf_fixture(tmp_path, w, dist=d, budgets=("5", "100"))
    arms = _qf_arms(tmp_path, pop, banks, rows, d, budgets=("5", "100"))
    res = E.evaluate_qf(pop, arms, banks_root=banks, draw_count=30)
    con = pd.DataFrame(res["contrasts"], columns=E.ESTIMATE_HEADER)
    bc = con[(con.arm_a == "both_filtered|ot_cosine|5") & (con.arm_b == "both_filtered|ot_cosine|100")]
    hit = bc[(bc.endpoint == "hit") & (bc.k == 1) & (bc.level == "macro")].iloc[0]
    assert hit.population == "both_returned" and hit.n_queries == 8          # s3 and s9 excluded
    hy = bc[(bc.endpoint == "hit_yield") & (bc.k == 1) & (bc.level == "macro")].iloc[0]
    assert hy.population == "intended" and hy.n_queries == 10
    pairs = {(p["arm_a"], p["arm_b"]) for p in res["contrast_pairs"]}
    assert ("query_only|ot_cosine|5", "both_filtered|ot_cosine|5") in pairs
    assert ("query_only|ot_cosine|5", "query_only|ot_cosine|100") in pairs


def _brute_tie(scores, rel, k):
    """Average the stable-rule metric over every ordering of each tied block."""
    order = np.argsort(scores, kind="mergesort")
    s, r = scores[order], rel[order]
    blocks = [list(g) for _, g in itertools.groupby(range(len(s)), key=lambda i: s[i])]
    hits, precs = [], []
    for perm in itertools.product(*(itertools.permutations(b) for b in blocks)):
        seq = [r[i] for blk in perm for i in blk]
        hits.append(float(any(seq[:k])))
        precs.append(sum(seq[:k]) / k)
    return np.mean(hits), np.mean(precs)


def test_tie_averaged_matches_brute_force_enumeration():
    assert E.tie_averaged(np.array([.1, .2, .2, .2, .5]), np.array([0, 0, 1, 0, 0], bool), 2) == \
        pytest.approx((1 / 3, 1 / 6))
    rng = np.random.default_rng(0)
    for _ in range(60):
        n = int(rng.integers(1, 8))
        scores = np.sort(rng.integers(0, 3, size=n).astype(float))
        rel = rng.random(n) < 0.4
        for k in (1, 2, 3, 5):
            got = E.tie_averaged(scores, rel, k)
            exp = _brute_tie(scores, rel, k)
            assert got[0] == pytest.approx(exp[0], abs=1e-12) and got[1] == pytest.approx(exp[1], abs=1e-12)
    # no ties -> identical to the stable rule
    assert E.tie_averaged(np.array([.1, .2, .3]), np.array([0, 1, 0], bool), 1) == (0.0, 0.0)


def test_tie_sensitivity_outputs_and_csv_clamp(tmp_path):
    d = _full(5)
    d[0, 2] = d[0, 4] = d[0, 3] = 0.05                   # s0: three-way tie at rank 1 (A, C, B)
    pop, banks, rows = _qf_fixture(tmp_path, ALL, dist=d)
    arms = _qf_arms(tmp_path, pop, banks, rows, d)
    res = E.evaluate_qf(pop, arms, banks_root=banks, draw_count=20)
    tc = pd.DataFrame(res["tie_counts"], columns=E.TIE_COUNT_HEADER)
    t1 = tc[(tc.arm == "all_tissue") & (tc.k == 1)].iloc[0]
    assert t1.boundary_tie_queries >= 1 and t1.tie_dependent_hit_queries >= 1
    q = pd.DataFrame(res["query_rows"], columns=E.QUERY_HEADER_QF)
    r0 = q[(q.arm == "all_tissue") & (q["query"] == "s0") & (q.k == 1)].iloc[0]
    assert r0.conditional_hit == 1.0 and r0.tie_averaged_hit == pytest.approx(1 / 3)
    rows_, n = E.clamp_rows([(0, "c", "a", "m", "", "", "", "hit", 1, "macro", "", "returned",
                              -5.55e-17, "ok", None, 1e-11, -0.0, "ok", None, 3, 3, 0)])
    assert n == 2 and rows_[0][12] == 0.0 and not np.signbit(rows_[0][12]) and rows_[0][15] == 1e-11
    assert rows_[0][16] == 0.0 and not np.signbit(rows_[0][16])
    out = tmp_path / "ev"
    meta = {"note": "fixture"}
    files = E.write_outputs_qf(res, out, meta)
    for f in ("estimates.csv", "contrasts.csv", "tie_sensitivity.csv", "tie_counts.csv", "coverage.csv",
              "libraries.csv", "tie_averaged_replicates.npy", "replicates.npy", "summary.json"):
        assert f in files
    ts = pd.read_csv(out / "tie_sensitivity.csv")
    assert set(ts.tie_rule) == {"stable_reference_index", "tie_averaged"}
    assert len(np.load(out / "tie_averaged_replicates.npy")) == (ts.tie_rule == "tie_averaged").sum()
    with pytest.raises(E.EvalError, match="not empty"):
        E.write_outputs_qf(res, out, meta)


def test_qf_cli_combine_and_evaluate_end_to_end(tmp_path, monkeypatch):
    monkeypatch.delenv("PROSPER_OUT", raising=False)
    d = _full(5)
    pop, banks, rows = _qf_fixture(tmp_path, ALL, dist=d)
    bp = tmp_path / "base.pkl"
    pickle.dump(_masked_frame(d), open(bp, "wb"))
    mats = tmp_path / "mats"
    for arm in ("both_filtered", "query_only"):
        assert E.main(["combine-qf", "--arm", arm, "--method", "ot", "--budget", "5",
                       "--population", str(tmp_path / "popq.csv"), "--rows-root", str(rows),
                       "--banks-root", str(banks), "--out-dir", str(mats), "--no-metadata-check",
                       "--folds", "0", "1"]) == 0
    argv = ["evaluate-qf", "--population", str(tmp_path / "popq.csv"), "--banks-root", str(banks),
            "--out-dir", str(tmp_path / "ev"), "--draw-count", "10",
            "--arm", "all_tissue", "ot_cosine", "-", str(bp), "-"]
    for arm in ("both_filtered", "query_only"):
        m = mats / E.qf_matrix_name("5", arm, "ot_cosine", True)
        argv += ["--arm", arm, "ot_cosine", "5", str(m), str(mats / E.qf_rows_name("5", arm, "ot_cosine"))]
    assert E.main(argv) == 0
    s = __import__("json").loads((tmp_path / "ev" / "summary.json").read_text())
    assert s["policy"]["primary_tie_rule"] == "stable_reference_index"
    assert len(s["contrast_pairs"]) == 3


def test_old_mode_main_is_unchanged_by_dispatch(tmp_path, monkeypatch):
    monkeypatch.delenv("PROSPER_OUT", raising=False)
    pop, arms = _setup(tmp_path)
    assert E.main(["--population", str(tmp_path / "pop.csv"), "--out-dir", str(tmp_path / "ev"),
                   "--draw-count", "10", "--arm", "all_tissue", "ot_cosine", "-", str(arms[0].matrix), "-"]) == 0
    hdr = (tmp_path / "ev" / "estimates.csv").read_text().splitlines()[0]
    assert hdr == ",".join(E.ESTIMATE_HEADER)


def _load_submit(monkeypatch, out):
    import importlib.util
    from pathlib import Path as _P
    monkeypatch.setenv("PROSPER_OUT", str(out))
    spec = importlib.util.spec_from_file_location(
        "_submit_r2_1_filter_qf_test", _P(__file__).resolve().parents[1] / "scripts/lsf/submit_r2_1_filter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_submit_qf_dry_run_layout_matches_combine_qf(tmp_path, monkeypatch, capsys):
    out = tmp_path / "OUT"
    Sm = _load_submit(monkeypatch, out)
    R = out / "r2_1_filter_20261003"
    for arm_tok, arm in Sm.QF_ARM_NAME.items():
        assert E.QF_ARM_TOKEN[arm] == arm_tok
        # rows directory written by similarity_cli (OUT/<task>) == directory combine-qf reads
        assert out / Sm.qf_task(5, 3, arm_tok, "ot") == R / "similarity_qf" / "budget5" / "fold3" / f"{arm_tok}_ot"
    assert f"--library-features {R / 'features_qf' / 'budget5' / 'fold3'} " in Sm.qf_sim_line(5, 3, "both", "ot", 1, 1)
    assert f"--library-features {R / 'similarity_inputs_qf' / 'library_unfiltered_pop2604'} " in \
        Sm.qf_sim_line(5, 3, "query", "ot", 1, 1)
    assert f"--features {R / 'similarity_inputs_qf' / 'queries' / 'budget5' / 'fold3'} " in Sm.qf_sim_line(5, 3, "both", "bob", 1, 1)
    with pytest.raises(SystemExit):        # no views yet: loud
        Sm.qf_sim(False, [5], [0], ["both"], ["ot"], {("both", "ot"): Sm.S.Resources(1, 1024, 10)}, 1, 50,
                  False, "", Sm._Queues(["qa"]))
    for f in (0, 1):
        (R / "similarity_inputs_qf" / "queries" / "budget5" / f"fold{f}").mkdir(parents=True)
        (R / "features_qf" / "budget5" / f"fold{f}").mkdir(parents=True)
        for q in ("a", "b", "c"):
            (R / "similarity_inputs_qf" / "queries" / "budget5" / f"fold{f}" / f"{f}{q}.pkl").write_bytes(b"")
    (R / "similarity_inputs_qf" / "library_unfiltered_pop2604").mkdir(parents=True)
    res = {(a, m): Sm.S.Resources(1, 1024, 10) for a in ("both", "query") for m in ("ot", "bob")}
    si = Sm.qf_sim(False, [5], [0, 1], ["both", "query"], ["ot", "bob"], res, 1, 2, False, "", Sm._Queues(["qa", "qb"]))
    assert si["budget5_both_ot"] == ["<r21qf_sim_b5_f0_both_ot>", "<r21qf_sim_b5_f1_both_ot>"]
    p = (R / "params_qf" / "sim_budget5_fold1_query_bob.txt").read_text().splitlines()
    assert [x.split("--lo ")[1].split(" --task")[0] for x in p] == ["1 --hi 2", "3 --hi 3"]
    co = Sm.qf_combine(False, [5], ["both"], ["ot"], si, Sm._Queues(["qa"]))
    ev = Sm.qf_evaluate(False, list(co.values()), Sm._Queues(["qa"]))
    txt = capsys.readouterr().out
    assert "bsub" in txt and "ended(<r21qf_sim_b5_f0_both_ot>) && ended(<r21qf_sim_b5_f1_both_ot>)" in txt
    assert "combine-qf --arm both_filtered --method ot --budget 5" in txt
    assert ev == "<r21qf_evaluate>" and "evaluate-qf --standard" in txt and "ended(<r21qf_combine_b5_both_ot>)" in txt
    assert " -q qa " in txt and " -q qb " in txt
    assert not (R / "similarity_qf").exists()          # dry run computes nothing
