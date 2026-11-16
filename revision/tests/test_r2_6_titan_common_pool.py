import json, os, sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import r2_6_titan_common_pool as T  # noqa: E402

# 8 slides, 2 labels, P1/P2 have two slides each, P3..P6 one each
IDS = [str(i) for i in range(1, 9)]
META = pd.DataFrame({"label": ["A", "A", "B", "B", "A", "B", "A", "B"],
                     "patient": ["P1", "P1", "P2", "P2", "P3", "P4", "P5", "P6"]}, index=IDS)


def matrix():
    rng = np.random.default_rng(0)
    x = rng.random((8, 3))
    d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    np.fill_diagonal(d, 1e6)
    d[0, 1] = d[1, 0] = 1e6            # P1 pair masked by sentinel
    d[2, 3] = d[3, 2] = 0.0            # P2 pair NOT masked in matrix: must still be excluded
    return pd.DataFrame(d, index=IDS, columns=IDS)


def test_ranking_excludes_self_and_same_patient():
    df, diag = T.rank(matrix(), IDS, IDS, META, depth=5)
    for q, g in df.groupby("query"):
        nb = set(g["neighbour"])
        assert q not in nb
        assert not (META.loc[list(nb), "patient"] == META.at[q, "patient"]).any()
        assert (np.diff(g["distance"].to_numpy()) >= 0).all()
    assert diag["same_patient_unmasked_in_matrix"] == 2
    assert T.same_patient_guard(df, META) == len(df)
    with pytest.raises(T.CheckError):
        T.rank(matrix(), IDS, IDS, META, depth=7)   # P1/P2 queries have only 6 eligible


def test_plurality_tie_break_closest_rank():
    assert T.plurality(["NB", "RMS", "EWS", "RMS", "EWS"]) == "RMS"
    assert T.plurality(["Z", "A", "A", "Z", "B"]) == "Z"
    assert T.plurality(["B", "A", "A", "C", "D"]) == "A"


def ranked(rows):
    """rows: {query: (true_label, [10 neighbour labels])}."""
    return pd.DataFrame([dict(query=q, true_label=t, rank=r + 1, neighbour=f"n{r}", neighbour_label=l,
                              distance=float(r)) for q, (t, ls) in rows.items() for r, l in enumerate(ls)])


def test_macro_vs_micro():
    hit = ["A"] + ["B"] * 9
    pq = T.per_query(ranked({"q1": ("A", hit), "q2": ("A", hit), "q3": ("A", hit), "q4": ("B", ["A"] * 10)}))
    v = pq["hit@1"].to_numpy()
    one = np.ones((1, 4))
    labels = pq["true_label"].to_numpy()
    assert T.micro(v, one)[0] == pytest.approx(0.75)
    assert T.macro(v, one, labels, ["A", "B"])[0] == pytest.approx(0.5)
    W = np.array([[2.0, 0, 0, 1]])                    # weighted: A rate 1, B rate 0
    assert T.micro(v, W)[0] == pytest.approx(2 / 3)
    assert T.macro(v, W, labels, ["A", "B"])[0] == pytest.approx(0.5)
    W0 = np.array([[1.0, 1, 1, 0]])                   # class B absent -> undefined, not dropped
    assert np.isnan(T.macro(v, W0, labels, ["A", "B"])[0])
    assert pq.loc[pq["query"] == "q4", "hit@10"].item() == 0


def test_patient_draws_and_crn():
    pats = ["P2", "P1", "P1", "P3"]
    p1, m1 = T.make_patient_draws(pats, count=50, seed=7)
    p2, m2 = T.make_patient_draws(pats, count=50, seed=7)
    assert p1 == ("P1", "P2", "P3") and np.array_equal(m1, m2)
    assert (m1.sum(axis=1) == 3).all()
    rng = np.random.Generator(np.random.PCG64(7))
    assert np.array_equal(m1[0], np.bincount(rng.integers(3, size=3), minlength=3))
    # slide weights = multiplicity of its patient; identical for both arms
    meta = pd.DataFrame({"label": ["A", "A", "B", "B"], "patient": pats}, index=list("abcd"))
    pq1 = pd.DataFrame({"query": list("abcd"), "true_label": ["A", "A", "B", "B"],
                        **{f"hit@{k}": [1.0, 0, 1, 0] for k in T.KS}, "plurality@5": [1.0, 1, 0, 0]})
    pq2 = pq1.copy()
    s, c = T.paired_summary(pq1, pq2, meta, "x", (p1, m1))
    d = s[s["arm"] == "delta"]
    mic, mac = d[d["metric"].str.startswith("micro")], d[d["metric"].str.startswith("macro")]
    assert np.allclose(mic[["estimate", "lo", "hi"]].to_numpy(), 0)   # CRN: identical arms -> zero delta
    assert mic["ci_available"].all() and mic["ci_degenerate"].all()
    # tiny fixture: some draws miss a class -> macro interval unavailable, never silently dropped
    assert (mac["n_zero_support_draws"] > 0).all() and not mac["ci_available"].any()
    assert mac["lo"].isna().all() and (mac["estimate"] == 0).all()
    assert s["n_patients"].iloc[0] == 3
    W = m1[:, [p1.index(p) for p in pats]]
    expect = (W * np.array([1.0, 0, 1, 0])).sum(1) / W.sum(1)
    est = s[(s["metric"] == "micro_hit@1") & (s["arm"] == "arm1")].iloc[0]
    assert est["lo"] == pytest.approx(np.percentile(expect, 2.5))


def test_connected_components():
    blob1 = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)]
    blob2 = [(5, 5), (5, 6), (6, 6)]
    diag_single = [(3, 3)]                          # touches nothing 4-connected
    xy = np.array(blob1 + blob2 + diag_single) * 1024 + np.array([2048, 3072])
    r = T.components(xy)
    assert r["n_components"] == 3 and r["largest_component"] == 5 and r["n_singletons"] == 1
    assert r["n_off_grid"] == 0 and r["n_duplicate_coords"] == 0
    r2 = T.components(np.array([[0, 0], [1024, 1024]]))   # diagonal neighbours are not 4-connected
    assert r2["n_components"] == 2


def test_same_patient_guard_raises():
    df = pd.DataFrame({"query": ["1"], "neighbour": ["2"]})
    with pytest.raises(T.CheckError):
        T.same_patient_guard(df, META)


def test_compare_allows_only_exact_ties():
    M = matrix()
    new, _ = T.rank(M, IDS, IDS, META, depth=3)
    assert T.compare(new, new.copy(), M, IDS)["exact_match_queries"] == 8
    bad = new.copy()
    bad.loc[0, "neighbour"], bad.loc[1, "neighbour"] = new.loc[1, "neighbour"], new.loc[0, "neighbour"]
    res = T.compare(new, bad, M, IDS)
    assert not res["passed"] and res["mismatch_queries"] == 1


def test_ci_zero_support_and_degenerate():
    r = T.ci(0.5, np.array([0.4, np.nan, 0.6]))
    assert np.isnan(r["lo"]) and np.isnan(r["hi"]) and not r["ci_available"]
    assert r["n_zero_support_draws"] == 1 and r["B_effective"] == 2 and not r["ci_degenerate"]
    r = T.ci(0.0, np.zeros(10))
    assert r["ci_available"] and r["ci_degenerate"] and r["lo"] == r["hi"] == 0.0
    r = T.ci(0.5, np.linspace(0, 1, 101))
    assert r["ci_available"] and not r["ci_degenerate"] and r["lo"] == pytest.approx(0.025)


def test_compare_rejects_non_library_tied_neighbour():
    ids = ["1", "3", "5", "6", "7"]
    M = pd.DataFrame(1e6, index=ids, columns=ids)
    for a, b, d in (("5", "6", 0.1), ("5", "7", 0.2), ("5", "1", 0.2), ("5", "3", 0.2)):
        M.loc[a, b] = M.loc[b, a] = d
    new, _ = T.rank(M, ["5"], ["6", "7", "1"], META, depth=2)
    assert new["neighbour"].tolist() == ["6", "7"]
    swap = new.copy()
    swap.loc[1, "neighbour"] = "1"                    # tied, in library -> tie permutation
    assert T.compare(new, swap, M, ["6", "7", "1"])["tie_permutation_queries"] == 1
    swap.loc[1, "neighbour"] = "3"                    # tied but not a library member
    assert not T.compare(new, swap, M, ["6", "7", "1"])["passed"]


def test_tie_diagnostics_and_clique():
    ids = ["1", "3", "5", "6", "7", "8"]
    M = pd.DataFrame(0.5, index=ids, columns=ids)
    np.fill_diagonal(M.values, 1e6)
    for a in ("3", "5", "6"):                          # identical embeddings: B, A, B
        for b in ("3", "5", "6"):
            if a != b:
                M.loc[a, b] = 0.0
    M.loc["1", "7"] = M.loc["7", "1"] = 0.1
    members, info = T.identical_clique(M)
    assert members == ["3", "5", "6"] and info["n_zero_pairs"] == 3 and info["rows_identical_outside_clique"]
    d, hmin, hmax = T.tie_diagnostics(M, ["5", "1"], ids, META, depth=2, k=1)
    # query 5 (A): two tied at 0 (both B) -> hit@1 decided 0; query 1 (A): 7 is A at 0.1 -> 1
    assert hmin.tolist() == [0, 1] and hmax.tolist() == [0, 1]
    assert d["queries_with_ties_in_top_or_at_cutoff"] == 2 and d["queries_hit_at_5_undecided"] == 0
    d, hmin, hmax = T.tie_diagnostics(M, ["3"], ids, META, depth=1, k=1)
    assert hmin.tolist() == [0] and hmax.tolist() == [1]  # tied 5 (A) and 6 (B) for query 3 (B)
    assert d["queries_hit_at_5_undecided"] == 1


def test_outdir_guard(tmp_path):
    d = tmp_path / "frozen"
    assert T.resolve_outdir(None, str(d)) == str(d)              # absent: allowed
    d.mkdir()
    assert T.resolve_outdir(None, str(d)) == str(d)              # exists but no results: allowed
    (d / "summary_common_pool.csv").write_text("x")
    with pytest.raises(SystemExit, match="REFUSING"):
        T.resolve_outdir(None, str(d))
    assert T.resolve_outdir(str(tmp_path / "other"), str(d)) == str(tmp_path / "other")


def test_override_skip_plan():
    assert T.legacy_skips({}) == {}
    s2 = T.legacy_skips({"arm2_matrix": "x", "arm2_record": "y"})
    assert set(s2) == {"reconcile_a_arm2_legacy"} | {f"c_legacy_macro_delta_hit@{k}" for k in T.KS}
    assert s2["reconcile_a_arm2_legacy"] == "arm-2 matrix overridden (metric/selection differs from legacy run)"
    s1 = T.legacy_skips({"arm1_matrix": "x"})
    assert set(s1) == {"reconcile_b_arm1_legacy", "reconcile_b_arm1_reference_2604"} | {
        f"c_legacy_macro_delta_hit@{k}" for k in T.KS}
    assert T.legacy_skips({"arm2_record": "y"}) == {}             # record alone skips nothing


def override_summary():
    rows = []
    for pop, e in (("common_pool_2564", 0.0074), (T.LEGACY_POP, -0.001), ("common_pool_excl_clique", 0.0031)):
        for m in ("micro_hit@5", "macro_hit@5"):
            for arm, v in (("arm1", 0.9), ("arm2", 0.9 + e), ("delta", e)):
                rows.append(dict(population=pop, metric=m, arm=arm, estimate=v, lo=v - .006, hi=v + .006))
    return pd.DataFrame(rows)


def test_annotate_metrics_and_readme(tmp_path):
    s = T.annotate_metrics(override_summary(), "cityblock", "cityblock")
    leg = s["population"] == T.LEGACY_POP
    assert (s.loc[leg, "arm2_metric"] == "cosine").all() and (s.loc[leg, "arm1_metric"] == "cityblock").all()
    assert (s.loc[leg, "note"] == "legacy ranked lists; arm 2 cosine; not an L1 result").all()
    assert (s.loc[~leg, "arm2_metric"] == "cityblock").all() and (s.loc[~leg, "note"] == "").all()
    # arm-1 metric only from a matching passing bitwise recompute
    rp = tmp_path / "_rec.json"
    rp.write_text("{}")
    (tmp_path / "reconciliation.json").write_text(json.dumps({
        "inputs": {"arm1_matrix": {"sha256": "abc"}},
        "checks": {T.ARM1_L1_CHECK: {"result": "PASS", "bitwise_equal_full_matrix": True}}}))
    assert T.arm1_metric_evidence(str(rp), "abc", "no record file; ...")[0] == "cityblock"
    assert T.arm1_metric_evidence(str(rp), "zzz", "no record file; ...") == ("unrecorded", None)
    rec = {"timestamp": "2026-10-04T12:00:00", "checks_skipped": {"reconcile_a_arm2_legacy": {}},
           "arm_metrics": {"arm1": "cityblock", "arm1_evidence": "E", "arm2": "cityblock"}}
    fmt = dict(n_pat=10, B=5000, seed=123, n_cl=21, n_zero=209, n_cl_pool=20, n_x=8, n_pat_x=7, n_tie=31,
               n_und=21, dmin=0.0031, dmax=0.0113, max1=630.2)
    txt = T.override_readme(fmt, s, rec, {"arm2_matrix": "m", "arm2_record": "r"})
    assert "override run (2026-10-04)" in txt and "Both arms use L1" in txt
    assert "~+0.0043 of the +0.0074" in txt and "+0.0031 to +0.0113" in txt
    assert "2 metrics x 3 populations" in txt and "not a significance claim" in txt
    assert "all-zero" in txt and "`reconcile_a_arm2_legacy`" in txt


# ---- CP-2: intersection pool, variant-dir connectivity, arm-1 record metric ----

def test_intersection_pool_order_dropped_and_reasons():
    meta = META.assign(label=["A", "A", "B", "B", "A", "B", "A", "B"])
    qc = ["2", "1", "4", "3", "6", "5", "8", "7"]                     # metadata-order input is preserved as given
    idx = [("arm1.pkl", IDS), ("arm2.pkl", ["1", "2", "3", "4", "6", "7", "8"]),
           ("extra.pkl", ["1", "2", "3", "4", "5", "6", "7"])]
    pool, dropped = T.intersection_pool(qc, idx)
    assert pool == ["2", "1", "4", "3", "6", "7"]
    t = T.pool_dropped_table(dropped, meta)
    assert t.columns.tolist() == ["slide", "label", "patient", "reason"]
    assert t["slide"].tolist() == ["5", "8"]
    assert t["label"].tolist() == ["A", "B"] and t["patient"].tolist() == ["P3", "P6"]
    assert t["reason"].tolist() == ["absent_from:arm2.pkl", "absent_from:extra.pkl"]
    pool, dropped = T.intersection_pool(qc, [("a", IDS), ("b", ["1", "2", "3", "4", "6", "7"])])
    assert [s for s, _ in dropped] == ["5", "8"] and pool == ["2", "1", "4", "3", "6", "7"]
    assert T.pool_dropped_table([], meta).columns.tolist() == ["slide", "label", "patient", "reason"]


def test_h5_vs_matrix_with_infeasible():
    ok, info = T.h5_vs_matrix(["1", "2", "3"], ["1", "2"], ["3"])
    assert ok and info == dict(n_h5=3, n_infeasible=1, n_infeasible_in_h5=1, n_infeasible_in_arm2_matrix=0)
    assert not T.h5_vs_matrix(["1", "2", "3"], ["1", "2"], [])[0]          # infeasible list missing
    assert not T.h5_vs_matrix(["1", "2"], ["1", "2"], ["3"])[0]            # infeasible slide has no h5
    assert not T.h5_vs_matrix(["1", "2", "3"], ["1", "2", "3"], ["3"])[0]  # infeasible slide embedded
    assert T.h5_vs_matrix(["1", "2"], ["1", "2"], [])[0]


def write_h5(path, coords):
    import h5py
    with h5py.File(path, "w") as h:
        h["coords"] = np.asarray(coords, dtype=np.int64)


def test_variant_paths_and_connectivity(tmp_path):
    vd = tmp_path / "stage2" / "V"
    d = vd / "job_dir" / "40x_1024px_0px_overlap"
    (d / "features_conch_v15").mkdir(parents=True)
    p, h5dir, sfd = T.variant_paths(str(vd))
    assert set(p) == {"patch_summary", "excluded", "subset_record"} and h5dir == str(d / "features_conch_v15")
    assert sfd == str(d / "slide_features_titan")
    (d / T.INFEASIBLE_CSV).write_text("slide,n_kept,reason,lsf_job\n3,1,titan_single_patch_alibi_indexerror,1\n")
    p, _, _ = T.variant_paths(str(vd))
    assert p["titan_infeasible"] == str(d / T.INFEASIBLE_CSV) and T.read_infeasible(p["titan_infeasible"]) == ["3"]
    assert T.read_infeasible(None) == []
    write_h5(d / "features_conch_v15" / "1.h5", [[0, 0], [1024, 0], [4096, 4096]])
    write_h5(d / "features_conch_v15" / "2.h5", [[2048, 1024]])
    write_h5(d / "features_conch_v15" / "3.h5", [2048, 1024])               # trident-squeezed one-patch coords
    summ = pd.DataFrame({"slide": ["1", "2", "3", "4"], "n_patches": [9, 9, 9, 9], "n_kept": [3, 1, 1, 0],
                         "frac_kept": [1 / 3, 1 / 9, 1 / 9, 0]})            # no mean_tiles_scored_per_patch
    excl = pd.DataFrame({"slide": pd.Series([], dtype=str), "reason": pd.Series([], dtype=str)})
    conn, grp, h = T.connectivity(h5dir, summ, excl, META)
    assert conn["n_kept_h5"].tolist() == [3, 1, 1] and conn["n_components"].tolist() == [2, 1, 1]
    assert conn["unscored_tile_fraction_all_patches"].isna().all()
    assert (grp["section"] == "kept").all() and len(h) == 64
    excl = pd.DataFrame({"slide": ["4"], "reason": ["0 kept patches in selection_b100_strict"]})
    _, grp, h2 = T.connectivity(h5dir, summ, excl, META)
    assert h2 == h and grp.loc[grp["section"] == "excluded", "reason_category"].tolist() == [
        "0_kept_patches_in_selection_b100_strict"]


def test_matrix_records_prefers_matching_out(tmp_path):
    m = tmp_path / "combined_similarity_matrix_titan_titan_tumor_only_l1_no_same_patient.pkl"
    m.write_bytes(b"x")
    assert T.matrix_records(str(m)) == ([], None, [])
    r1 = tmp_path / "_combined_similarity_matrix_titan_titan_tumor_only_l1_record.json"
    r1.write_text(json.dumps({"metric": "cityblock", "tumor_only": True, "out": str(m)}))
    other = tmp_path / "_combined_similarity_matrix_titan_titan_x_record.json"
    other.write_text(json.dumps({"metric": "cosine", "out": str(tmp_path / "missing.pkl")}))
    assert T.matrix_records(str(m)) == ([str(r1)], "cityblock", [str(other)])   # record of this matrix wins
    r1.write_text(json.dumps({"metric": "cityblock"}))                       # no record names this matrix: never borrowed
    assert T.matrix_records(str(m)) == ([], "unverified", sorted([str(r1), str(other)]))
    r1.write_text(json.dumps({"metric": "cityblock", "out": str(m)}))
    other.write_text(json.dumps({"metric": "cosine", "out": str(m)}))       # two records of this matrix disagree
    with pytest.raises(T.CheckError):
        T.matrix_records(str(m))


def test_arm1_tumour_only_matrix_not_misreported(tmp_path):
    """Arm 1 = the arm-2 L1 matrix: the arm-2 dir's bitwise check concerns the published matrix (other
    sha), so no bitwise evidence, and the README must not describe arm 1 as the published matrix."""
    rp = tmp_path / "_rec.json"
    rp.write_text("{}")
    (tmp_path / "reconciliation.json").write_text(json.dumps({
        "inputs": {"arm1_matrix": {"sha256": "published"}},
        "checks": {T.ARM1_L1_CHECK: {"result": "PASS", "bitwise_equal_full_matrix": True}}}))
    m1, ev = T.arm1_metric_evidence(str(rp), "tumour_only_l1_sha", "cityblock")
    assert (m1, ev) == ("cityblock", None)
    s = T.annotate_metrics(override_summary(), m1, "cityblock").replace(
        {"population": {"common_pool_2564": "common_pool_2562_intersection"}})
    rec = {"timestamp": "2026-10-04T15:00:00", "checks_skipped": {},
           "arm_metrics": {"arm1": m1, "arm1_evidence": ev, "arm2": "cityblock", "arm1_matrix": "A1.pkl",
                           "arm1_record": ["A1_record.json"], "arm1_tumor_only": True, "arm2_matrix": "V.pkl",
                           "arm2_slide_features_dir": "stage2/V/slide_features_titan",
                           "arm1_words": T.S2_WORDS, "arm2_words": "direct-1024 variant V",
                           "legacy_note": T.LEGACY_ARMS_NOTE},
           "pool": {"label": "common_pool_2562_intersection", "n_common": 2564, "n_pool": 2562, "n_dropped": 2,
                    "matrices": [{"role": "arm1_matrix", "path": "A1.pkl"}, {"role": "pool_matrix", "path": "P.pkl"}],
                    "dropped": [{"slide": "6840", "label": "OS"}, {"slide": "15110", "label": "MEL"}]},
           "variant": {"dir": "stage2/V", "h5dir": "h", "slide_features_dir": "s", "n_infeasible": 7,
                       "mean_tiles_scored_available": False}}
    fmt = dict(n_pat=10, B=5000, seed=123, n_cl=0, n_zero=0, n_cl_pool=0, n_x=2562, n_pat_x=7, n_tie=3,
               n_und=1, dmin=-0.001, dmax=0.002, max1=630.2)
    txt = T.override_readme(fmt, s, rec, {"arm1_matrix": "A1.pkl", "arm2_matrix": "V.pkl", "arm2_record": "r"})
    assert "Headline population: `common_pool_2562_intersection`" in txt
    for bad in ("Both arms use L1", "Arm 1 = published", "Arm 1 (published", "all-zero", "artefact-affected",
                "tumour-enriched minus all-tissue", "2026-10-03 run only by metric"):
        assert bad not in txt, bad
    assert "Arm 1 = `A1.pkl`, metric cityblock (per record `A1_record.json`)" in txt
    assert "6840 (OS); 15110 (MEL)" in txt and "**Variant.**" in txt and "7 slide(s)" in txt
    assert "`--pool-matrices` = `P.pkl`" in txt and "`--variant-dir` = `stage2/V`" in txt
    assert "`unscored_tile_fraction_all_patches` is empty" in txt and "arms as defined above" not in txt
    assert f"Arm 1 = {T.S2_WORDS}; arm 2 = direct-1024 variant V." in txt
    assert f"arm2 - arm1 (direct-1024 variant V minus {T.S2_WORDS})" in txt
    assert f"are {T.LEGACY_ARMS_NOTE}." in txt and f'`note`: "{T.LEGACY_ARMS_NOTE}"' in txt


def build_fixture(root, n=20):
    """Synthetic workspace for main(): slides 1..n-2 tumour (A/B/C), n-1/n NORMAL; arm 1 has all slides,
    the variant arm 2 lacks infeasible slide 5 and zero-kept slide n-2; an extra pool matrix lacks 5 and 7."""
    P, R, OUT = root, root / "repo", root / "out"
    R.mkdir()
    ids = [str(i) for i in range(1, n + 1)]
    labels = [["A", "B", "C"][i % 3] for i in range(n - 2)] + ["NORMAL", "NORMAL"]
    pats = [f"P{(i + 1) // 2}" if i < 6 else f"P{i}" for i in range(n)]   # slides 2/3, 4/5 share a patient
    meta_df = pd.DataFrame({"Slide.Scan.File": ids, "Tumor.group": labels, "UID.Subject": pats,
                            "Sample.ID": [f"S{i}" for i in ids]})
    mp = P / "comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv"
    mp.parent.mkdir(parents=True)
    meta_df.to_csv(mp, index=False)
    meta = T.load_meta(str(mp))
    rng = np.random.default_rng(1)
    emb = pd.DataFrame(rng.normal(size=(n, 4)), index=ids)

    def l1(sl):
        x = emb.loc[sl].to_numpy()
        d = np.abs(x[:, None] - x[None]).sum(-1)
        p = meta.loc[sl, "patient"].to_numpy()
        d[(p[:, None] == p[None]) | np.eye(len(sl), dtype=bool)] = 1e6
        return pd.DataFrame(d, index=sl, columns=sl)

    def save(dirpath, name, sl, extra):
        dirpath.mkdir(parents=True, exist_ok=True)
        M = l1(sl)
        M.to_pickle(dirpath / f"{name}.pkl")
        a = M.to_numpy()
        rec = {"metric": "cityblock", "out": str(dirpath / f"{name}.pkl"), "max_real_distance": float(a[a < 1e5].max()),
               "same_patient_pairs_masked": int(((a >= 1e5) & ~np.eye(len(a), dtype=bool)).sum() // 2), **extra}
        rp = dirpath / f"_combined_similarity_matrix_titan_titan_{name}_record.json"
        rp.write_text(json.dumps(rec))
        return M, str(dirpath / f"{name}.pkl"), str(rp)

    vd = OUT / "stage2" / "V"
    jd = vd / "job_dir" / "40x_1024px_0px_overlap"
    (jd / "slide_features_titan").mkdir(parents=True)
    (jd / "features_conch_v15").mkdir()
    zero_kept = str(n - 2)
    M1, a1, _ = save(OUT / "arm1", "arm1", ids, {"tumor_only": False})
    v_ids = [s for s in ids[:n - 2] if s not in ("5", zero_kept)]
    _, a2, r2 = save(OUT / "l1_V", "arm2", v_ids, {"tumor_only": True, "slide_features_dir": str(jd / "slide_features_titan")})
    _, ax, _ = save(OUT / "extra", "extra", [s for s in ids if s not in ("5", "7")], {})
    common = ids[:n - 2]
    (OUT / "s2_paired_in").mkdir(parents=True)
    (OUT / "s2_paired_in" / "_common_queries.txt").write_text("".join(s + "\n" for s in common))
    gz = {"method": "gzip", "mtime": 0}
    rl, _ = T.rank(M1, common, common, meta)
    for f in ("s2_paired_in/ranked_lists_titan_titan.csv.gz", "s2_paired_in/ranked_lists_titan_titan_tumor_only.csv.gz",
              "s2_tumor_only_reference/ranked_lists_titan_titan.csv.gz"):
        (OUT / f).parent.mkdir(parents=True, exist_ok=True)
        rl.to_csv(OUT / f, index=False, compression=gz)
    (OUT / "s2_paired").mkdir()
    for k in T.KS:
        (OUT / "s2_paired" / f"paired_vs_titan_titan_tumor_only_k{k}.json").write_text("{}")
    h5_ids = ids[:n - 2]
    h5_ids.remove(zero_kept)
    rows = []
    for i, s in enumerate(ids):
        nk = 0
        if s in h5_ids:
            nk = 1 if s == "5" else 2 + i % 3
            write_h5(jd / "features_conch_v15" / f"{s}.h5", [[1024 * j, 0] for j in range(nk)])
        rows.append({"slide": s, "n_patches": 10, "n_kept": nk, "frac_kept": nk / 10})
    pd.DataFrame(rows).to_csv(jd / "_tumour_patch_summary.csv", index=False)
    pd.DataFrame({"slide": [zero_kept, str(n - 1), str(n)], "reason": ["0 kept patches in selection_V"] * 3}).to_csv(
        jd / "_excluded_slides.csv", index=False)
    (jd / "_subset_record.json").write_text("{}")
    (jd / T.INFEASIBLE_CSV).write_text("slide,n_kept,reason,lsf_job\n5,1,titan_single_patch_alibi_indexerror,1\n")
    draws_src = root / "draws_src.py"
    draws_src.write_text("# stand-in for the package source hashed by main()\n")
    return dict(R=R, OUT=OUT, a1=a1, a2=a2, r2=r2, ax=ax, vd=str(vd), jd=jd, draws_src=str(draws_src), ids=ids)


def run_main(monkeypatch, fx, argv):
    monkeypatch.chdir(fx["R"])
    monkeypatch.setenv("PROSPER_OUT", str(fx["OUT"]))
    monkeypatch.setattr(T, "DRAWS_SRC", fx["draws_src"])
    monkeypatch.setattr(T, "B", 200)
    monkeypatch.setattr(sys, "argv", ["r2_6_titan_common_pool.py"] + argv)
    return T.main()


def test_main_pool_and_variant_end_to_end(tmp_path, monkeypatch):
    fx = build_fixture(tmp_path)
    od = tmp_path / "res"
    code = run_main(monkeypatch, fx, ["--arm1-matrix", fx["a1"], "--arm2-matrix", fx["a2"], "--arm2-record", fx["r2"],
                                      "--pool-matrices", fx["ax"], fx["a1"], "--variant-dir", fx["vd"],
                                      "--outdir", str(od)])
    rec = json.load(open(od / "reconciliation.json"))
    assert code == 0, {k: v for k, v in rec["checks"].items() if not v["passed"]}
    assert "population_sizes" not in rec["checks"] and rec["checks"]["population_sizes_pool"]["passed"]
    assert rec["checks"]["variant_arm2_record_features_dir"]["passed"]
    assert rec["checks"]["i_h5_slides_equal_arm2_matrix"]["n_infeasible"] == 1
    assert set(rec["checks_skipped"]) == {"reconcile_a_arm2_legacy", "reconcile_b_arm1_legacy",
                                          "reconcile_b_arm1_reference_2604"} | {f"c_legacy_macro_delta_hit@{k}" for k in T.KS}
    expect_pool = [s for s in fx["ids"][:18] if s not in ("5", "7", "18")]
    assert (od / "pool_queries.txt").read_text().split() == expect_pool
    dropped = pd.read_csv(od / "pool_dropped.csv", dtype=str)
    assert dropped.to_dict("list") == {"slide": ["5", "7", "18"], "label": ["B", "A", "C"], "patient": ["P2", "P6", "P17"],
                                       "reason": ["absent_from:arm2.pkl;extra.pkl", "absent_from:extra.pkl",
                                                  "absent_from:arm2.pkl"]}
    assert [m["role"] for m in rec["pool"]["matrices"]] == ["arm1_matrix", "arm2_matrix", "pool_matrix"]  # a1 deduplicated
    assert rec["pool"]["n_pool"] == 15 and rec["pool"]["n_common"] == 18
    s = pd.read_csv(od / "summary_common_pool.csv")
    assert set(s["population"]) == {"common_pool_15_intersection", T.LEGACY_POP, "common_pool_excl_clique"}
    assert (s["n_queries"] == 15).all()                                   # legacy and clique rows restricted to the pool
    assert (s.loc[s["population"] != T.LEGACY_POP, ["arm1_metric", "arm2_metric"]] == "cityblock").all().all()
    pq = pd.read_csv(od / "per_query_common_pool.csv", dtype={"query": str})
    assert sorted(pq["query"]) == sorted(expect_pool) and pq["query"].is_unique
    rl = pd.read_csv(od / "ranked_lists_common_pool_arm2.csv.gz", dtype={"neighbour": str})
    assert set(rl["neighbour"]) <= set(expect_pool)
    conn = pd.read_csv(od / "connectivity_per_slide.csv", dtype={"slide": str})
    assert len(conn) == 17 and conn.loc[conn["titan_infeasible"], "slide"].tolist() == ["5"]
    am = rec["arm_metrics"]
    assert am["arm1"] == "cityblock" and am["arm1_evidence"] is None and am["arm1_tumor_only"] is False
    assert am["arm1_record"] == [str(fx["OUT"] / "arm1" / "_combined_similarity_matrix_titan_titan_arm1_record.json")]
    txt = (od / "README.md").read_text()
    assert "Headline population: `common_pool_15_intersection`" in txt and "**Pool.**" in txt
    assert "5 (B); 7 (A); 18 (C)" in txt and "1 slide(s)" in txt and "Both arms use L1" not in txt
    assert "arm2 - arm1 (direct-1024 variant V minus all-tissue TITAN (`arm1.pkl`))" in txt
    assert (s.loc[s["population"] == T.LEGACY_POP, "note"] == T.LEGACY_ARMS_NOTE).all()       # R1
    assert (s.loc[s["population"] != T.LEGACY_POP, "note"].isna()).all()
    assert T.LEGACY_ARMS_NOTE in txt
    assert rec["pool"]["dropped_slides"] == ["5", "7", "18"]                                   # R3
    assert rec["checks"]["arm2_record_out_is_arm2_matrix"]["passed"]
    assert rec["checks"]["arm1_override_no_zero_distance_slides"]["passed"]


def test_main_variant_record_mismatch_fails(tmp_path, monkeypatch):
    fx = build_fixture(tmp_path)
    r = json.load(open(fx["r2"]))
    r["slide_features_dir"] = str(tmp_path)                               # embeddings of some other run
    open(fx["r2"], "w").write(json.dumps(r))
    od = tmp_path / "res"
    assert run_main(monkeypatch, fx, ["--arm1-matrix", fx["a1"], "--arm2-matrix", fx["a2"], "--arm2-record", fx["r2"],
                                      "--pool-matrices", fx["ax"], "--variant-dir", fx["vd"], "--outdir", str(od)]) == 1
    rec = json.load(open(od / "reconciliation.json"))
    assert not rec["checks"]["variant_arm2_record_features_dir"]["passed"]
    assert not (od / "summary_common_pool.csv").exists()


def test_main_variant_without_infeasible_csv_fails_h5_check(tmp_path, monkeypatch):
    fx = build_fixture(tmp_path)
    os.remove(fx["jd"] / T.INFEASIBLE_CSV)
    od = tmp_path / "res"
    assert run_main(monkeypatch, fx, ["--arm1-matrix", fx["a1"], "--arm2-matrix", fx["a2"], "--arm2-record", fx["r2"],
                                      "--pool-matrices", fx["ax"], "--variant-dir", fx["vd"], "--outdir", str(od)]) == 1
    c = json.load(open(od / "reconciliation.json"))["checks"]["i_h5_slides_equal_arm2_matrix"]
    assert not c["passed"] and c["n_infeasible"] == 0


def test_override_readme_refuses_clique_text_for_other_arm1():
    s = T.annotate_metrics(override_summary(), "cityblock", "cityblock")
    rec = {"timestamp": "2026-10-04T15:00:00", "checks_skipped": {},
           "arm_metrics": {"arm1": "cityblock", "arm1_evidence": None, "arm2": "cityblock", "arm1_matrix": "A1.pkl"}}
    fmt = dict(n_pat=10, B=5000, seed=123, n_cl=3, n_zero=3, n_cl_pool=3, n_x=8, n_pat_x=7, n_tie=0,
               n_und=0, dmin=0.0, dmax=0.0, max1=630.2)
    with pytest.raises(T.CheckError):
        T.override_readme(fmt, s, rec, {"arm1_matrix": "A1.pkl", "arm2_matrix": "m", "arm2_record": "r"})


# ---- CP-3: legacy-arm note, arm words, expected pool, early clique refusal, record binding ----

def test_annotate_metrics_legacy_arms_note():
    s = T.annotate_metrics(override_summary(), "cityblock", "cityblock", T.LEGACY_ARMS_NOTE)
    leg = s["population"] == T.LEGACY_POP
    assert (s.loc[leg, "note"] == "legacy arms: arm1 = published all-tissue TITAN (cityblock), arm2 = S2 "
            "tumour-enriched TITAN (cosine); not this run's arms").all()
    assert (s.loc[leg, ["arm1_metric", "arm2_metric"]] == ["cityblock", "cosine"]).all().all()


def test_arm_words():
    W = T.arm_words
    assert W(None, False, "published all-tissue TITAN", "x.pkl") == "published all-tissue TITAN"
    assert W({"tumor_only": False, "slide_features_dir": "a (2,900 non-zero) + b (21 regenerated)"}, True, "-",
             "combined_similarity_matrix_titan_titan_repaired_no_same_patient.pkl") == "repaired all-tissue TITAN"
    assert W({"tumor_only": True, "slide_features_dir": "/o/s2_titan_tumor_only/40x/slide_features_titan"}, True, "-",
             "m.pkl") == T.S2_WORDS
    assert W({"tumor_only": True, "slide_features_dir": "/o/d/stage2/b5_argmax/job_dir/x/slide_features_titan"}, True,
             "-", "m.pkl") == "direct-1024 variant b5_argmax"
    assert W({"tumor_only": True}, True, "-", "m.pkl", "/o/d/stage2/b100_strict/") == "direct-1024 variant b100_strict"
    assert W(None, True, "-", "/a/m.pkl") == "unidentified matrix `m.pkl` (no matching record)"
    assert W({"metric": "cityblock"}, True, "-", "/a/m.pkl") == "unidentified matrix `m.pkl` (record lacks `tumor_only`)"


def test_main_expect_pool(tmp_path, monkeypatch):
    fx = build_fixture(tmp_path)
    base = ["--arm1-matrix", fx["a1"], "--arm2-matrix", fx["a2"], "--arm2-record", fx["r2"],
            "--pool-matrices", fx["ax"], "--variant-dir", fx["vd"]]
    pool = [s for s in fx["ids"][:18] if s not in ("5", "7", "18")]
    ok = tmp_path / "ok.txt"
    ok.write_text("".join(s + "\n" for s in pool))
    assert run_main(monkeypatch, fx, base + ["--expect-pool", str(ok), "--outdir", str(tmp_path / "r_ok")]) == 0
    rec = json.load(open(tmp_path / "r_ok" / "reconciliation.json"))
    assert rec["checks"]["pool_matches_expected"]["passed"]
    assert rec["inputs"]["expect_pool"]["sha256"] == T.sha256(str(ok))
    for name, ids in (("order", pool[1::-1] + pool[2:]), ("set", pool[:-1] + ["7"])):
        f = tmp_path / f"{name}.txt"
        f.write_text("".join(s + "\n" for s in ids))
        od = tmp_path / f"r_{name}"
        assert run_main(monkeypatch, fx, base + ["--expect-pool", str(f), "--outdir", str(od)]) == 1
        c = json.load(open(od / "reconciliation.json"))["checks"]["pool_matches_expected"]
        assert not c["passed"] and c["same_set"] == (name == "order") and not c["same_order"]
        assert not (od / "summary_common_pool.csv").exists()


def test_main_arm2_record_out_must_name_arm2_matrix(tmp_path, monkeypatch):
    fx = build_fixture(tmp_path)
    r = json.load(open(fx["r2"]))
    r["out"] = fx["ax"]                                                    # record of another matrix
    open(fx["r2"], "w").write(json.dumps(r))
    od = tmp_path / "res"
    assert run_main(monkeypatch, fx, ["--arm1-matrix", fx["a1"], "--arm2-matrix", fx["a2"], "--arm2-record", fx["r2"],
                                      "--pool-matrices", fx["ax"], "--variant-dir", fx["vd"], "--outdir", str(od)]) == 1
    c = json.load(open(od / "reconciliation.json"))["checks"]["arm2_record_out_is_arm2_matrix"]
    assert not c["passed"] and c["arm2_record_out"] == fx["ax"]


def test_main_overridden_arm1_with_zero_slides_fails_early(tmp_path, monkeypatch):
    fx = build_fixture(tmp_path)
    M = pd.read_pickle(fx["a1"])
    M.loc["1", "6"] = M.loc["6", "1"] = 0.0                                 # two indistinguishable arm-1 slides
    M.to_pickle(fx["a1"])
    od = tmp_path / "res"
    assert run_main(monkeypatch, fx, ["--arm1-matrix", fx["a1"], "--arm2-matrix", fx["a2"], "--arm2-record", fx["r2"],
                                      "--pool-matrices", fx["ax"], "--variant-dir", fx["vd"],
                                      "--outdir", str(od)]) == 1
    c = json.load(open(od / "reconciliation.json"))["checks"]["arm1_override_no_zero_distance_slides"]
    assert not c["passed"] and c["n_slides"] == 2 and not (od / "README.md").exists()


def test_override_detection_by_same_file(tmp_path):
    a = tmp_path / "a.pkl"
    a.write_bytes(b"x")
    (tmp_path / "alias").symlink_to(tmp_path)
    assert T.same_path(str(tmp_path / "alias" / "a.pkl"), str(a))           # mount alias is not an override
    assert not T.same_path(str(tmp_path / "b.pkl"), str(a))
    assert T.same_path(str(tmp_path / "x" / ".." / "b.pkl"), str(tmp_path / "b.pkl"))
