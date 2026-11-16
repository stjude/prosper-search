"""Synthetic-only tests for scripts/r2_3_internal_failure_cis.py (no real data, no package import)."""
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(R, "scripts", "r2_3_internal_failure_cis.py")
A_SCRIPT = os.path.join(R, "scripts", "r2_3_internal_failure_tables.py")
R37 = os.path.join(R, "scripts", "r3_7_fine_label_reevaluation.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load(SCRIPT, "r2_3_cis")
A = _load(A_SCRIPT, "r2_3_tables_frozen")  # frozen A rules, used to build encodings exactly as A does

LABELS = ["G1", "G2", "G3"]


def _row(q, truth, patient, labs):
    """One per-query row encoded with the frozen A script's rules."""
    rel = [lab == truth for lab in labs]
    first = next((i + 1 for i, r in enumerate(rel) if r), None)
    ok, pred = A.legacy_generous(labs, truth)
    return dict(query=q, true_label=truth, patient=patient, library_support=5, top1=labs[0],
                plurality_k3=A.plurality(labs, 3), plurality_k5=A.plurality(labs, 5),
                plurality_k10=A.plurality(labs, 10), strict_majority_k5=A.strict_majority(labs),
                legacy_generous_k5_correct=int(ok), legacy_generous_k5_pred=pred,
                hit1=int(any(rel[:1])), hit3=int(any(rel[:3])), hit5=int(any(rel[:5])), hit10=int(any(rel[:10])),
                precision5=sum(rel[:5]) / 5, precision10=sum(rel[:10]) / 10,
                first_relevant_rank=str(first) if first else "absent@10", n_patients_top5=5,
                top5_labels="|".join(labs[:5]), top5_neighbours="|".join(f"n{i}" for i in range(5)),
                top1_distance=0.1)


def synth_frame(n_patients=12, seed=0, labels=LABELS):
    rng = np.random.default_rng(seed)
    rows, q = [], 0
    for p in range(n_patients):
        truth = labels[p % len(labels)]
        for _ in range(1 + p % 3):
            labs = list(rng.choice(labels, size=10))
            rows.append(_row(f"q{q}", truth, f"P{p:03d}", labs)); q += 1
    # guaranteed edge cases: abstention, absent@10, oracle tie credit
    rows.append(_row(f"q{q}", "G1", "P900", ["G1", "G2", "G3", "G2", "G1"] + ["G3"] * 5)); q += 1
    rows.append(_row(f"q{q}", "G2", "P901", ["G1"] * 10)); q += 1
    rows.append(_row(f"q{q}", "G3", "P902", ["G1", "G3", "G1", "G3", "G2"] + ["G2"] * 5))
    return pd.DataFrame(rows)


def frozen_summary(fr):
    """per_group_summary.csv as the frozen A script writes it."""
    return pd.DataFrame([dict(true_label=t, **A.summarise(g)) for t, g in fr.groupby("true_label")]
                        + [{"true_label": "ALL", **A.summarise(fr), "median_library_support": float("nan")}])


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


RESULT_FILES = ("summary_overall.csv", "paired_differences.csv", "per_group_summary.csv", "macro_summary.csv",
                "README.md", "COMPLETE")


def assert_failed_layout(out):
    assert (out / "FAILED").is_file() and (out / "record.json").is_file() and (out / "reconciliation.json").is_file()
    assert not any((out / f).exists() for f in RESULT_FILES)
    fails = [c["name"] for c in json.load(open(out / "reconciliation.json"))["checks"] if c["status"] == "FAIL"]
    assert fails and (out / "FAILED").read_text().split() == fails


@pytest.fixture
def inputs(tmp_path):
    fr = synth_frame()
    d = tmp_path / "in"; d.mkdir()
    fr.to_csv(d / "all_query_frame.csv", index=False)
    frozen_summary(fr).to_csv(d / "per_group_summary.csv", index=False)
    return d, fr


def args_for(d, out, B=200, extra=()):
    inp, frz = str(d / "all_query_frame.csv"), str(d / "per_group_summary.csv")
    return ["--input", inp, "--outdir", str(out), "--B", str(B), "--seed", "7", "--allow-any-n",
            "--expected-sha256", sha(inp), "--expected-frozen-sha256", sha(frz), *extra]


def test_draws_function_is_verbatim_copy():
    def seg(path, name):
        src = open(path).read()
        for n in ast.parse(src).body:
            if isinstance(n, ast.FunctionDef) and n.name == name:
                return ast.get_source_segment(src, n)
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == name:
                return ast.get_source_segment(src, n)
    for name in ("make_patient_draws", "DRAWS_SOURCE"):
        assert seg(R37, name) is not None and seg(R37, name) == seg(SCRIPT, name)


def test_draws_reproducible_with_seed():
    ids = [f"P{i}" for i in range(30)] * 2
    p1, m1 = M.make_patient_draws(ids, count=50, seed=123)
    p2, m2 = M.make_patient_draws(ids, count=50, seed=123)
    _, m3 = M.make_patient_draws(ids, count=50, seed=124)
    assert p1 == p2 == tuple(sorted(set(ids)))
    assert np.array_equal(m1, m2) and not np.array_equal(m1, m3)
    assert (m1.sum(1) == 30).all()


def test_weights_match_brute_force_resample():
    fr = synth_frame(n_patients=6, seed=3)
    V = M.per_query_vectors(fr)
    B, seed = 25, 11
    patients, mult, W = M.weight_matrix(fr["patient"], B, seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    for b in range(B):
        drawn = [patients[i] for i in rng.integers(len(patients), size=len(patients))]
        res = pd.concat([fr[fr["patient"] == p] for p in drawn], ignore_index=True)
        rv = M.per_query_vectors(res)
        for k in ("hit5", "plurality_k5_accuracy", "mrr10", "strict_k5_abstain_rate"):
            assert M.boot_means(W[b:b + 1], V[k])[0] == pytest.approx(rv[k].mean(), abs=1e-12)
        g1 = (fr["true_label"] == "G1").to_numpy()
        sub = res[res["true_label"] == "G1"]
        bm = M.boot_means(W[b:b + 1], V["hit10"], g1)[0]
        if len(sub):
            assert bm == pytest.approx(sub["hit10"].mean(), abs=1e-12)
        else:
            assert np.isnan(bm)


def test_strict_abstention_and_rank_encodings():
    fr = synth_frame()
    assert (fr["strict_majority_k5"] == "ABSTAIN").any() and (fr["first_relevant_rank"] == "absent@10").any()
    V = M.per_query_vectors(fr)
    ab = (fr["strict_majority_k5"] == "ABSTAIN").to_numpy()
    assert (V["strict_k5_accuracy"][ab] == 0).all()
    assert V["strict_k5_abstain_rate"].mean() == pytest.approx(ab.mean())
    absent = (fr["first_relevant_rank"] == "absent@10").to_numpy()
    assert (V["mrr10"][absent] == 0).all() and (fr.loc[absent, "hit10"] == 0).all()
    r = fr.loc[~absent, "first_relevant_rank"].astype(int).to_numpy()
    assert np.allclose(V["mrr10"][~absent], 1.0 / r)
    tie = fr.iloc[-3]  # truth G1; G1 and G2 tied 2-2 in the top 5
    assert tie["strict_majority_k5"] == "ABSTAIN" and tie["legacy_generous_k5_correct"] == 1
    with pytest.raises(ValueError):
        M.rank_values(pd.Series(["1", "0"]))
    with pytest.raises(ValueError):
        M.rank_values(pd.Series(["11"]))


def test_per_group_zero_count_handling():
    fr = synth_frame(n_patients=9, seed=5)
    fr = pd.concat([fr, pd.DataFrame([_row("solo", "G4", "PSOLO", ["G4"] + ["G1"] * 9)])], ignore_index=True)
    B = 300
    res = M.compute(fr, B, seed=1)
    pg = res["per_group"]
    solo = pg[pg["true_label"] == "G4"]
    n_abs = int(solo["n_replicates_group_absent"].iloc[0])
    assert 0 < n_abs < B
    assert solo["ci_lo"].isna().all() and solo["ci_hi"].isna().all()
    assert (solo["n_nonfinite"] == n_abs).all()
    mac = res["macro"].set_index("estimand").loc["macro_hit5"]
    V, W = res["V"], res["W"]
    absent = np.column_stack([W[:, (fr["true_label"] == g).to_numpy()].sum(1) == 0 for g in res["groups"]])
    assert int(absent[:, res["groups"].index("G4")].sum()) == n_abs
    assert int(mac["n_replicates_with_absent_group"]) == int(absent.any(1).sum()) >= n_abs
    assert int(mac["min_groups_present"]) == int((~absent).sum(1).min()) <= 3
    assert np.isfinite(mac["ci_lo"]) and np.isfinite(mac["ci_hi"])
    # manual replicate-level macro: mean over groups present
    reps = []
    for b in range(B):
        vals = [M.boot_means(W[b:b + 1], V["hit5"], (fr["true_label"] == g).to_numpy())[0] for g in res["groups"]]
        reps.append(np.mean([v for v in vals if np.isfinite(v)]))
    lo, hi, _ = M.ci(reps)
    assert mac["ci_lo"] == pytest.approx(lo, abs=1e-12) and mac["ci_hi"] == pytest.approx(hi, abs=1e-12)
    assert mac["point"] == pytest.approx(np.mean([fr.loc[fr["true_label"] == g, "hit5"].mean()
                                                  for g in res["groups"]]))


def test_end_to_end_pass_and_containment(inputs, tmp_path):
    d, fr = inputs
    out = tmp_path / "out"
    assert M.main(args_for(d, out)) == 0
    for f in ("summary_overall.csv", "paired_differences.csv", "per_group_summary.csv", "macro_summary.csv",
              "README.md", "record.json", "reconciliation.json"):
        assert (out / f).is_file()
    rec = json.load(open(out / "reconciliation.json"))
    st = {c["name"]: c["status"] for c in rec["checks"]}
    assert rec["all_pass"] and "FAIL" not in st.values()
    assert st["ci_contains_point_estimate_where_available"] == "PASS"
    assert st["points_equal_frozen_per_group_summary"] == "PASS"
    assert st["input_n_rows_equals_2778"] == "WAIVED"
    ov = pd.read_csv(out / "summary_overall.csv")
    assert ((ov["ci_lo"] <= ov["point"]) & (ov["point"] <= ov["ci_hi"])).all()
    assert (ov["n_replicates"] == 200).all() and set(ov["estimand"]) == set(M.OVERALL)
    r = json.load(open(out / "record.json"))
    assert r["B"] == 200 and r["seed"] == 7 and r["input"]["n_rows"] == len(fr)
    assert r["input"]["sha256"] == sha(str(d / "all_query_frame.csv"))
    readme = open(out / "README.md").read()
    assert readme.startswith("STATUS: COMPLETE") and "nDCG@5" in readme and "open coordinator/author scope" in readme
    assert (out / "COMPLETE").is_file() and not (out / "FAILED").exists()
    assert r["reconciliation_all_pass"] is True and r["percentile_method"].startswith("numpy.percentile")
    assert r["frozen_a_script"]["sha256"] == sha(A_SCRIPT) and r["hostname"] and "prosper-search" in r["draws_source_note"]
    assert st["rules_rederived_from_top5_labels"] == "PASS" and st["brute_force_resample_spot_check"] == "PASS"
    pg = pd.read_csv(out / "per_group_summary.csv", keep_default_na=False)
    assert "boundary_flag" in pg.columns and "boundary_flag" in ov.columns


def test_refuses_to_overwrite(inputs, tmp_path):
    d, _ = inputs
    out = tmp_path / "out"; out.mkdir(); (out / "keep.txt").write_text("x")
    assert M.main(args_for(d, out)) == 2
    assert os.listdir(out) == ["keep.txt"] and (out / "keep.txt").read_text() == "x"


def test_allow_any_n_gate(inputs, tmp_path):
    d, _ = inputs
    out = tmp_path / "out"
    a = [x for x in args_for(d, out) if x != "--allow-any-n"]
    assert M.main(a) == 2 and not out.exists()
    assert M.main(a + ["--allow-any-n"]) == 0


def test_reconciliation_fail_exits_nonzero(inputs, tmp_path):
    d, _ = inputs
    frz = d / "per_group_summary.csv"
    F = pd.read_csv(frz); F.loc[0, "hit5"] = F.loc[0, "hit5"] + 1e-9; F.to_csv(frz, index=False)
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, "-B", SCRIPT] + args_for(d, out), capture_output=True, text=True)
    assert p.returncode == 1 and "RECONCILIATION FAILED" in p.stderr
    assert_failed_layout(out)
    assert json.load(open(out / "record.json"))["reconciliation_all_pass"] is False
    st = {c["name"]: c["status"] for c in json.load(open(out / "reconciliation.json"))["checks"]}
    assert st["points_equal_frozen_per_group_summary"] == "FAIL"
    out2 = tmp_path / "out2"
    assert M.main(args_for(d, out2, extra=["--expected-sha256", "0" * 64])) == 1
    assert_failed_layout(out2)
    st2 = {c["name"]: c["status"] for c in json.load(open(out2 / "reconciliation.json"))["checks"]}
    assert st2["input_sha256_matches_expected"] == "FAIL"


def test_invalid_strict_encoding_fails(tmp_path):
    fr = synth_frame(); fr.loc[0, "strict_majority_k5"] = "NOVOTE"
    d = tmp_path / "in"; d.mkdir()
    fr.to_csv(d / "all_query_frame.csv", index=False)
    frozen_summary(synth_frame()).to_csv(d / "per_group_summary.csv", index=False)
    out = tmp_path / "out"
    assert M.main(args_for(d, out)) == 1
    st = {c["name"]: c["status"] for c in json.load(open(out / "reconciliation.json"))["checks"]}
    assert st["strict_k5_values_are_labels_or_ABSTAIN"] == "FAIL"
    assert st["rules_rederived_from_top5_labels"] == "FAIL"
    assert_failed_layout(out)


def test_ci_equals_numpy_linear_percentile():
    v = np.random.default_rng(2).normal(size=1001)
    lo, hi, bad = M.ci(v)
    assert bad == 0 and (lo, hi) == tuple(np.percentile(v, [2.5, 97.5]))
    assert (lo, hi) == tuple(np.percentile(v, [2.5, 97.5], method="linear"))
    lo2, hi2, bad2 = M.ci(np.r_[v, np.nan])
    assert np.isnan(lo2) and np.isnan(hi2) and bad2 == 1


def test_paired_row_equals_ci_of_within_replicate_differences():
    fr = synth_frame(n_patients=20, seed=9)
    res = M.compute(fr, 400, seed=3)
    pr = res["paired"].set_index("estimand")
    for k, (a, b) in M.PAIRED.items():
        lo, hi, _ = M.ci(res["boots"][a] - res["boots"][b])
        assert pr.loc[k, "ci_lo"] == lo and pr.loc[k, "ci_hi"] == hi
        # within-replicate differences, recomputed from per-query differences under the same weights
        dv = M.boot_means(res["W"], res["V"][a] - res["V"][b])
        lo3, hi3, _ = M.ci(dv)
        assert pr.loc[k, "ci_lo"] == pytest.approx(lo3, abs=1e-12) and pr.loc[k, "ci_hi"] == pytest.approx(hi3, abs=1e-12)


def test_spot_check_detects_weight_corruption():
    fr = synth_frame(n_patients=10, seed=4)
    res = M.compute(fr, 50, seed=8)
    reps, worst, n = M.spot_check(fr, res, 50)
    assert reps == [0, 25, 49] and worst <= 1e-12 and n > len(M.OVERALL)
    res["boots"]["hit5"] = res["boots"]["hit5"].copy(); res["boots"]["hit5"][25] += 1e-6
    assert M.spot_check(fr, res, 50)[1] > 1e-12


def test_boundary_flags_and_containment_is_warn_only(inputs):
    assert M.boundary_flag(0.0, 0.0) == "lower_eq_0|zero_width"
    assert M.boundary_flag(0.4, 1.0) == "upper_eq_1"
    assert M.boundary_flag(0.2, 0.9) == "" and M.boundary_flag(np.nan, np.nan) == "ci_unavailable"
    d, fr = inputs
    res = M.compute(fr, 100, seed=1)
    res["overall"].loc[0, "point"] = res["overall"].loc[0, "ci_hi"] + 0.5
    chk = M.reconcile(fr, res, 100, "x", "x", str(d / "per_group_summary.csv"), sha(str(d / "per_group_summary.csv")),
                      True)
    st = {c["name"]: c["status"] for c in chk}
    assert st["ci_contains_point_estimate_where_available"] == "WARN"


def test_rederivation_catches_tie_break_tampering():
    fr = synth_frame()
    i = fr.index[fr["top5_labels"] == "G1|G2|G3|G2|G1"][0]
    assert fr.loc[i, "plurality_k5"] == "G1"  # closest-rank tie-break between G1 and G2
    assert M.rederive_from_top5(fr) == []
    fr.loc[i, "plurality_k5"] = "G2"
    assert M.rederive_from_top5(fr) == [f"{fr.loc[i, 'query']}:plurality_k5"]
