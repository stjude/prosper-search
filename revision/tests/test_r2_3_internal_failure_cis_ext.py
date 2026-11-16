"""Tests for scripts/r2_3_internal_failure_cis_ext.py (synthetic frames; real files read-only and skipped if absent)."""
import hashlib
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(R, "scripts", "r2_3_internal_failure_cis_ext.py")
FROZEN = os.path.join(R, "scripts", "r2_3_internal_failure_cis.py")
A_SCRIPT = os.path.join(R, "scripts", "r2_3_internal_failure_tables.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load(SCRIPT, "r2_3_cis_ext")
FZ = _load(FROZEN, "r2_3_cis_frozen_for_ext_test")  # independent load of the frozen module
A = _load(A_SCRIPT, "r2_3_tables_frozen_for_ext_test")  # frozen A rules, to encode synthetic rows exactly as A does
LABELS = ["G1", "G2", "G3"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _row(q, truth, patient, labs, support=27):
    rel = [lab == truth for lab in labs]
    first = next((i + 1 for i, r in enumerate(rel) if r), None)
    ok, pred = A.legacy_generous(labs, truth)
    return dict(query=q, true_label=truth, patient=patient, library_support=support, top1=labs[0],
                plurality_k3=A.plurality(labs, 3), plurality_k5=A.plurality(labs, 5),
                plurality_k10=A.plurality(labs, 10), strict_majority_k5=A.strict_majority(labs),
                legacy_generous_k5_correct=int(ok), legacy_generous_k5_pred=pred,
                hit1=int(rel[0]), hit3=int(any(rel[:3])), hit5=int(any(rel[:5])), hit10=int(any(rel[:10])),
                precision5=sum(rel[:5]) / 5, precision10=sum(rel[:10]) / 10,
                first_relevant_rank=str(first) if first else "absent@10", n_patients_top5=5,
                top5_labels="|".join(labs[:5]), top5_neighbours="|".join(f"n{i}" for i in range(5)),
                top1_distance=0.1, labs10="|".join(labs[:10]))


def synth_frame(n_patients=12, seed=0, labels=LABELS):
    rng = np.random.default_rng(seed)
    rows, q = [], 0
    for p in range(n_patients):
        truth = labels[p % len(labels)]
        for _ in range(1 + p % 3):
            rows.append(_row(f"q{q}", truth, f"P{p:03d}", list(rng.choice(labels, size=10)))); q += 1
    rows.append(_row(f"q{q}", "G1", "P900", ["G1", "G2", "G3", "G2", "G1"] + ["G3"] * 5)); q += 1  # tie G1/G2
    rows.append(_row(f"q{q}", "G2", "P901", ["G1"] * 10)); q += 1                                 # absent@10
    rows.append(_row(f"q{q}", "G3", "P902", ["G1", "G3", "G1", "G3", "G2"] + ["G2"] * 5))         # ABSTAIN
    return pd.DataFrame(rows)


def ranked_of(fr):
    """c2-style ranked lists (10 rows per query) from the synthetic labs10 column."""
    return pd.DataFrame([dict(query=r.query, true_label=r.true_label, rank=i + 1, neighbour=f"n{i}",
                              neighbour_label=lab, distance=0.1 + i)
                         for r in fr.itertuples() for i, lab in enumerate(r.labs10.split("|"))])


def rel10_of(fr):
    return np.array([[lab == t for lab in ls.split("|")] for ls, t in zip(fr["labs10"], fr["true_label"])], bool)


def write_input(tmp_path, fr, ranked=None):
    d = tmp_path / "in"; d.mkdir()
    fr.to_csv(d / "all_query_frame.csv", index=False)
    (ranked if ranked is not None else ranked_of(fr)).to_csv(d / "ranked.csv.gz", index=False, compression="gzip")
    return str(d / "all_query_frame.csv")


def args_for(inp, out, B=200, extra=()):
    rk = os.path.join(os.path.dirname(inp), "ranked.csv.gz")
    return ["--input", inp, "--outdir", str(out), "--B", str(B), "--seed", "7", "--allow-any-n", "--no-reference",
            "--expected-sha256", sha(inp), "--ranked-lists", rk, "--expected-ranked-sha256", sha(rk), *extra]


def checks_of(out):
    return {c["name"]: c for c in json.load(open(out / "reconciliation.json"))["checks"]}


def assert_failed_layout(out, must_include):
    assert (out / "FAILED").is_file() and (out / "record.json").is_file() and (out / "reconciliation.json").is_file()
    assert not any((out / f).exists() for f in E.RESULT_FILES + ("COMPLETE",))
    fails = [n for n, c in checks_of(out).items() if c["status"] == "FAIL"]
    assert must_include in fails and (out / "FAILED").read_text().split() == fails
    assert json.load(open(out / "record.json"))["reconciliation_all_pass"] is False


# ------------------------------------------------------------------ binding to the frozen module and its draws
def test_frozen_module_bound_and_draws_identical():
    assert sha(FROZEN) == E.ACCEPTED_FROZEN_SCRIPT_SHA256
    assert os.path.abspath(E.F.__file__) == os.path.abspath(FROZEN)
    fr = synth_frame()
    res = E.compute(fr, 50, 123)
    _, first10 = FZ.make_patient_draws(list(fr["patient"]), count=10, seed=123)
    assert np.array_equal(res["mult"][:10], first10)
    assert np.array_equal(res["W"], FZ.compute(fr, 50, 123)["W"])
    # a different seed gives different draws, so the comparison is not vacuous
    assert not np.array_equal(FZ.make_patient_draws(list(fr["patient"]), count=10, seed=124)[1], first10)


def test_defaults_are_B5000_seed123():
    src = open(SCRIPT).read()
    assert 'add_argument("--B", type=int, default=5000)' in src and 'add_argument("--seed", type=int, default=123)' in src
    assert (E.Q_LO, E.Q_HI) == (2.5, 97.5)


# ------------------------------------------------------------------ definitions
def test_ap_ndcg_hand_values_at10():
    rel = np.zeros((2, 10), bool); rel[0, [1, 6]] = True; rel[1, [0, 1, 2]] = True
    R_ = np.array([27.0, 3.0])
    d = 1 / np.log2(np.arange(2, 12))
    assert np.allclose(E.ap_at_k(rel, R_, k=10), [(1 / 2 + 2 / 7) / 10, 1.0], atol=1e-15)
    assert np.allclose(E.ndcg_at_k(rel, R_, k=10), [(d[1] + d[6]) / d.sum(), 1.0], atol=1e-15)


def test_ap_ndcg_hand_values():
    rel = np.array([[0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], bool)
    R_ = np.array([27.0, 2.0, 27.0, 5.0])
    d = 1 / np.log2(np.arange(2, 7))
    assert np.allclose(E.ap_at_k(rel, R_), [(1 / 2 + 2 / 4) / 5, (1 / 2 + 2 / 4) / 2, 0.0, 1.0], atol=1e-15)
    assert np.allclose(E.ndcg_at_k(rel, R_), [(d[1] + d[3]) / d.sum(), (d[1] + d[3]) / d[:2].sum(), 0.0, 1.0],
                       atol=1e-15)


def test_plurality_tie_break_closest_ranked_and_rederivation_guard(tmp_path):
    fr = synth_frame()
    tie = fr.index[fr["query"] == fr["query"].iloc[-3]][0]
    assert fr.at[tie, "top5_labels"] == "G1|G2|G3|G2|G1" and fr.at[tie, "plurality_k5"] == "G1"  # G1 ranked first
    bad = fr.copy(); bad.at[tie, "plurality_k5"] = "G2"   # the farther tied label
    out = tmp_path / "out"
    assert E.main(args_for(write_input(tmp_path, bad), out)) == 1
    assert_failed_layout(out, "rules_rederived_from_top5_labels")


def test_strict_convention_abstain_is_fn_only_and_f1_hand():
    rows = [_row("a", "G1", "P1", ["G1"] * 10), _row("b", "G1", "P2", ["G1", "G2", "G3", "G2", "G1"] + ["G3"] * 5),
            _row("c", "G2", "P3", ["G2"] * 10), _row("d", "G2", "P4", ["G1", "G1", "G1", "G2", "G2"] + ["G2"] * 5)]
    fr = pd.DataFrame(rows)
    assert list(fr["strict_majority_k5"]) == ["G1", "ABSTAIN", "G2", "G1"]
    pts, gp = E.point_estimates(fr)
    # strict: G1 TP1 FP1 FN1 -> 2/4; G2 TP1 FP0 FN1 -> 2/3 (ABSTAIN on b adds FN to G1, FP to nobody)
    assert gp[("G1", "strict_f1")] == pytest.approx(0.5) and gp[("G2", "strict_f1")] == pytest.approx(2 / 3)
    assert pts["strict_macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)
    # plurality: b -> G1 (tie, closest), d -> G1: G1 TP2 FP1 -> 4/5; G2 TP1 FN1 -> 2/3
    assert pts["plurality_macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    # an ABSTAIN-counted-as-FP mutant would differ
    assert pts["strict_macro_f1"] != pytest.approx(pts["plurality_macro_f1"])


# ------------------------------------------------------------------ bootstrap
def test_weighted_f1_and_macro_match_brute_force_all_replicates():
    fr = synth_frame(n_patients=7, seed=3)
    fr.loc[fr["patient"] == "P900", "true_label"] = "G4"  # a one-patient group, absent from some replicates
    fr.loc[fr["patient"] == "P900", "legacy_generous_k5_pred"] = "G1"
    fr.loc[fr["patient"] == "P900", "top5_labels"] = "G1|G2|G3|G2|G1"
    fr.loc[fr["patient"] == "P900", "labs10"] = "G1|G2|G3|G2|G1|G4|G3|G3|G3|G3"
    B = 60
    rel10 = rel10_of(fr)
    res = E.compute(fr, B, 11, rel10)
    assert set(E.SUMMARY10) <= set(res["boots"])
    assert res["absent"][:, res["groups"].index("G4")].any()
    rows_of = {p: np.flatnonzero((fr["patient"] == p).to_numpy()) for p in res["patients"]}
    for b in range(B):
        idx = np.concatenate([np.repeat(rows_of[p], res["mult"][b, i]) for i, p in enumerate(res["patients"])])
        pts, gp = E.point_estimates(fr.iloc[idx].reset_index(drop=True), rel10[idx])
        for k in res["boots"]:
            assert res["boots"][k][b] == pytest.approx(pts[k], abs=1e-12)
        for (g, m), bv in res["group_boots"].items():
            assert (np.isnan(bv[b]) and (g, m) not in gp) or bv[b] == pytest.approx(gp[(g, m)], abs=1e-12)
    row = res["per_group"].query("true_label == 'G4' and metric == 'plurality_f1'").iloc[0]
    assert row["n_replicates_group_absent"] > 0 and np.isnan(row["lo"]) and row["boundary_flag"] == "ci_unavailable"
    s = res["summary"].set_index("metric")
    assert s.loc["plurality_macro_f1", "n_replicates_with_absent_group"] == res["absent"].any(1).sum() > 0


def test_intervals_are_95_percentile():
    res = E.compute(synth_frame(), 300, 5)
    s = res["summary"].set_index("metric")
    for k, bv in res["boots"].items():
        lo, hi = np.percentile(bv, [2.5, 97.5])
        assert (s.loc[k, "lo"], s.loc[k, "hi"]) == (lo, hi)
        assert (s.loc[k, "lo"], s.loc[k, "hi"]) != tuple(np.percentile(bv, [10, 90]))


def test_80_percent_mutant_fails_reconciliation(tmp_path, monkeypatch):
    inp = write_input(tmp_path, synth_frame())
    ok = tmp_path / "ok"
    assert E.main(args_for(inp, ok)) == 0 and (ok / "COMPLETE").is_file()
    assert checks_of(ok)["ci_level_95_percentile_2_5_97_5"]["status"] == "PASS"

    def eighty(v):
        v = np.asarray(v, float)
        return (np.nan, np.nan, 1) if not np.isfinite(v).all() else (*np.percentile(v, [10, 90]), 0)
    monkeypatch.setattr(E, "interval", eighty)
    out = tmp_path / "mut"
    assert E.main(args_for(inp, out)) == 1
    assert_failed_layout(out, "ci_level_95_percentile_2_5_97_5")


# ------------------------------------------------------------------ CLI conventions
def test_complete_layout_and_records(tmp_path):
    inp = write_input(tmp_path, synth_frame())
    out = tmp_path / "out"
    assert E.main(args_for(inp, out, B=100)) == 0
    for f in E.RESULT_FILES + ("COMPLETE", "record.json", "reconciliation.json"):
        assert (out / f).is_file()
    assert not (out / "FAILED").exists()
    rec = json.load(open(out / "record.json"))
    assert rec["B"] == 100 and rec["seed"] == 7 and rec["input"]["sha256"] == sha(inp)
    assert rec["script"]["sha256"] == sha(SCRIPT) and rec["frozen_module"]["sha256"] == sha(FROZEN)
    assert rec["reconciliation_all_pass"] is True
    st = {c["status"] for c in checks_of(out).values()}
    assert st <= {"PASS", "WARN", "WAIVED"}
    c = checks_of(out)
    for name in ("draws_identical_to_frozen", "paired_with_frozen_bootstrap", "brute_force_resample_spot_check",
                 "replicate_count_equals_B", "per_group_n_sums_to_n"):
        assert c[name]["status"] == "PASS"
    s = pd.read_csv(out / "summary_ext.csv")
    assert list(s["metric"]) == list(E.SUMMARY) + list(E.SUMMARY10) and {"point", "lo", "hi", "n_queries", "n_patients",
                                                     "boundary_flag"} <= set(s.columns)
    first = open(out / "README.md").readline()
    nw = sum(v["status"] == "WARN" for v in c.values())
    assert first.startswith(f"STATUS: COMPLETE - 0 FAIL, {nw} WARN")
    g = pd.read_csv(out / "per_group_ext.csv")
    ng = g["true_label"].nunique()
    assert list(g["metric"][:5 * ng]) == E.PER_GROUP * ng and list(g["metric"][5 * ng:]) == E.PER_GROUP10 * ng
    assert c["at10_metrics_computed_from_verified_ranked_lists"]["status"] == "PASS"


def test_inconsistent_ranked_lists_fail(tmp_path):
    fr = synth_frame()
    rk = ranked_of(fr)
    rk.loc[(rk["query"] == "q0") & (rk["rank"] == 2), "neighbour_label"] = "G9"  # disagrees with top5_labels
    out = tmp_path / "out"
    assert E.main(args_for(write_input(tmp_path, fr, rk), out)) == 1
    assert_failed_layout(out, "ranked_lists_consistent_with_frame")
    assert "at10_metrics_computed_from_verified_ranked_lists" in (out / "FAILED").read_text().split()


@pytest.mark.parametrize("ranked_case", ["inconsistent", "missing"])
def test_reference_mode_bad_ranked_lists_fail_cleanly(tmp_path, ranked_case):
    """Regression (re-review R1): in reference mode (no --no-reference) pts lacks the @10 keys when the ranked
    lists are missing/inconsistent; reconcile must still record FAIL rather than raise KeyError."""
    fr = synth_frame()
    rk = ranked_of(fr)
    rk.loc[(rk["query"] == "q0") & (rk["rank"] == 2), "neighbour_label"] = "G9"
    inp = write_input(tmp_path, fr, rk)
    rpath = os.path.join(os.path.dirname(inp), "ranked.csv.gz")
    if ranked_case == "missing":
        os.remove(rpath)
    mjson = tmp_path / "in" / "metrics.json"
    mjson.write_text(json.dumps({"point_estimates": {j: 0.5 for j in E.JSON_KEYS.values()}, "n_queries": len(fr)}))
    out = tmp_path / "out"
    argv = ["--input", inp, "--outdir", str(out), "--B", "50", "--seed", "7", "--allow-any-n",
            "--expected-sha256", sha(inp), "--metrics-json", str(mjson), "--expected-metrics-sha256", sha(str(mjson)),
            "--ranked-lists", rpath, "--expected-ranked-sha256", "0" * 64]
    assert E.main(argv) == 1
    assert_failed_layout(out, "ranked_lists_consistent_with_frame")
    failed = (out / "FAILED").read_text().split()
    assert "at10_metrics_computed_from_verified_ranked_lists" in failed
    c = checks_of(out)
    assert c["metrics_json_sha256_matches_accepted"]["status"] == "PASS"
    assert c["metrics_json_points_reproduced"]["status"] == "FAIL"
    assert all(c["metrics_json_points_reproduced"]["diff"][k] is None for k in E.SUMMARY10)
    assert all(c[n]["status"] != "WAIVED" for n in ("stated_point_values_reproduced", "metrics_json_points_reproduced",
                                                    "map10_micro_reproduced_from_ranked_lists"))  # reference mode


def test_plurality_not_top1_and_tie_not_at_rank1(tmp_path):
    rows = [_row("a", "G1", "P1", ["G2", "G1", "G1", "G1", "G3"] + ["G3"] * 5),   # top1 G2, plurality G1 (no tie)
            _row("b", "G3", "P2", ["G2", "G3", "G1", "G3", "G1"] + ["G2"] * 5),   # tie G3/G1; closest G3 (rank 2)
            _row("c", "G2", "P3", ["G2", "G1", "G1", "G3", "G3"] + ["G2"] * 5),   # tie G1/G3; closest G1 (rank 2)
            _row("d", "G2", "P4", ["G2"] * 10)]
    fr = pd.DataFrame(rows)
    assert list(fr["top1"]) == ["G2", "G2", "G2", "G2"]
    assert list(fr["plurality_k5"]) == ["G1", "G3", "G1", "G2"]
    pts, gp = E.point_estimates(fr)
    # plurality: G1 TP1 FP1 -> 2/3; G2 TP1 FN1 -> 2/3; G3 TP1 -> 1
    assert pts["plurality_macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)
    top1 = E.point_estimates(fr.assign(plurality_k5=fr["top1"]))[0]["plurality_macro_f1"]
    farthest = E.point_estimates(fr.assign(plurality_k5=["G1", "G1", "G3", "G2"]))[0]["plurality_macro_f1"]
    assert top1 != pytest.approx(pts["plurality_macro_f1"]) and farthest != pytest.approx(pts["plurality_macro_f1"])
    for i, wrong in ((0, "G2"), (1, "G1"), (2, "G3")):   # top1 instead of plurality; farther tied label
        bad = fr.copy(); bad.at[i, "plurality_k5"] = wrong
        d = tmp_path / f"case{i}"; d.mkdir()
        out = d / "out"
        assert E.main(args_for(write_input(d, bad), out, B=50)) == 1
        assert_failed_layout(out, "rules_rederived_from_top5_labels")


def test_known_boundary_flags(tmp_path):
    # G1 always correct with no false positives -> F1 = 1 in every replicate; G3 never predicted -> F1 = 0.
    # 20 patients per group so no group is absent from any replicate (absence gives ci_unavailable, tested above).
    rows = [_row(f"a{p}", "G1", f"P{p}", ["G1"] * 10) for p in range(20)]
    rows += [_row(f"b{p}", "G2", f"Q{p}", ["G2"] * 10) for p in range(20)]
    rows += [_row(f"c{p}", "G3", f"S{p}", ["G2"] * 10) for p in range(20)]
    fr = pd.DataFrame(rows)
    res = E.compute(fr, 200, 3, rel10_of(fr))
    g = res["per_group"].set_index(["true_label", "metric"])["boundary_flag"]
    assert g[("G1", "plurality_f1")] == "upper_eq_1|zero_width"
    assert g[("G3", "plurality_f1")] == "lower_eq_0|zero_width"
    assert g[("G3", "ap10")] == "lower_eq_0|zero_width"
    assert g[("G2", "plurality_f1")] == ""  # G2 F1 varies with the G3 false positives
    s = res["summary"].set_index("metric")["boundary_flag"]
    assert s["map5_micro"] == "" and s["plurality_macro_f1"] == ""


def test_refusals(tmp_path):
    inp = write_input(tmp_path, synth_frame())
    out = tmp_path / "busy"; out.mkdir(); (out / "x").write_text("x")
    assert E.main(args_for(inp, out)) == 2 and sorted(os.listdir(out)) == ["x"]
    o2 = tmp_path / "o2"
    a = [x for x in args_for(inp, o2) if x not in ("--allow-any-n", "--no-reference")]
    assert E.main(a) == 2 and not o2.exists()  # wrong n without --allow-any-n
    a = [x for x in args_for(inp, o2) if x != "--allow-any-n"]
    assert E.main(a) == 2 and not o2.exists()  # --no-reference needs --allow-any-n


def test_fail_path_writes_only_failed_record_reconciliation(tmp_path):
    inp = write_input(tmp_path, synth_frame())
    out = tmp_path / "out"
    assert E.main(args_for(inp, out, extra=("--expected-sha256", "0" * 64))) == 1
    assert_failed_layout(out, "input_sha256_matches_expected")


# ------------------------------------------------------------------ real point estimates (read-only)
REAL = os.path.isfile(E.DEFAULT_INPUT) and os.path.isfile(E.DEFAULT_METRICS_JSON) and os.path.isfile(E.DEFAULT_RANKED)


@pytest.mark.skipif(not REAL, reason="accepted input / c2 metrics not available")
def test_real_point_estimates_reproduce_stated_and_json():
    fr, input_sha, _ = E.F.read_input(E.DEFAULT_INPUT)
    assert input_sha == E.F.ACCEPTED_INPUT_SHA256 and len(fr) == 2778
    pts, _ = E.point_estimates(fr)
    stated = {"map5_micro": 0.748086, "ndcg5_macro": 0.708490, "plurality_macro_f1": 0.767912,
              "generous_macro_f1": 0.791156, "strict_macro_f1": 0.743908}
    for k, v in stated.items():
        assert abs(pts[k] - v) <= 5e-7, k
    pe = json.load(open(E.DEFAULT_METRICS_JSON))["point_estimates"]
    for k, j in E.JSON_KEYS.items():
        if k in pts:  # @5 keys; @10 keys are checked below with the ranked-list relevance
            assert abs(pts[k] - pe[j]) <= 1e-12, k
    m10, cons = E.map10_from_ranked(fr, E.DEFAULT_RANKED)
    assert abs(m10 - 0.701745) <= 5e-7 and abs(m10 - pe["micro_ap_at_10"]) <= 1e-12
    assert all(v for k, v in cons.items() if k != "n_rows") and cons["n_rows"] == 27780
    rel10, _ = E.load_ranked(fr, E.DEFAULT_RANKED)
    pts10, _ = E.point_estimates(fr, rel10)
    for k, j in E.JSON_KEYS.items():
        assert abs(pts10[k] - pe[j]) <= 1e-12, k
    assert all(pts10[k] == pts[k] for k in pts)  # @5/F1 values unchanged by adding @10
