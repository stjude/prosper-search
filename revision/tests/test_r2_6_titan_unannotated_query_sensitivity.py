import hashlib, json, os, sys

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import r2_6_titan_unannotated_query_sensitivity as U  # noqa: E402

T = U.T
NAME = "synth"
POP = "common_pool_60_intersection"


def synth_meta(n_pat=40, seed=1):
    """60 slides, 3 groups; 20 patients have two slides; patients P00-P09 are annotated."""
    rng = np.random.default_rng(seed)
    pats = [f"P{i:02d}" for i in range(n_pat)] + [f"P{i:02d}" for i in range(20)]
    grp = {p: "ABC"[i % 3] for i, p in enumerate(sorted(set(pats)))}
    ids = [str(1000 + i) for i in range(len(pats))]
    meta = pd.DataFrame({"label": [grp[p] for p in pats], "patient": pats, "sample": ids}, index=ids)
    x = rng.normal(size=(len(ids), 4)) + np.array([{"A": 0, "B": 1.5, "C": 3}[g] for g in meta["label"]])[:, None]
    return meta, x


def matrix(x, ids, noise):
    d = np.abs(x[:, None] - x[None]).sum(-1) + noise
    d = (d + d.T) / 2
    np.fill_diagonal(d, 1e6)
    return pd.DataFrame(d, index=ids, columns=ids)


def write_bundle(root, meta, x):
    d = os.path.join(root, U.BUNDLE_FMT.format(NAME))
    os.makedirs(d)
    ids = list(meta.index)
    rng = np.random.default_rng(7)
    rl1, _ = T.rank(matrix(x, ids, rng.random((len(ids),) * 2) * 3), ids, ids, meta)
    rl2, _ = T.rank(matrix(x, ids, rng.random((len(ids),) * 2) * 1), ids, ids, meta)
    gz = {"method": "gzip", "mtime": 0}
    rl1.to_csv(os.path.join(d, "ranked_lists_common_pool_arm1.csv.gz"), index=False, compression=gz)
    rl2.to_csv(os.path.join(d, "ranked_lists_common_pool_arm2.csv.gz"), index=False, compression=gz)
    pq1, pq2 = T.per_query(rl1), T.per_query(rl2)
    pq = pq1[["query", "true_label"]].copy()
    pq["patient"] = meta.loc[pq["query"], "patient"].to_numpy()
    for arm, t in (("arm1", pq1), ("arm2", pq2)):
        for c in t.columns[2:]:
            pq[f"{arm}_{c}"] = t[c].to_numpy()
    pq.to_csv(os.path.join(d, "per_query_common_pool.csv"), index=False)
    draws = T.make_patient_draws(meta.loc[ids, "patient"], count=T.B, seed=T.SEED)
    s, c = T.paired_summary(pq1, pq2, meta, POP, draws)
    s.to_csv(os.path.join(d, "summary_common_pool.csv"), index=False)
    c.to_csv(os.path.join(d, "per_class_common_pool.csv"), index=False)
    with open(os.path.join(d, "pool_queries.txt"), "w") as f:
        f.write("".join(i + "\n" for i in ids))
    with open(os.path.join(d, "reconciliation.json"), "w") as f:
        json.dump({"pool": {"label": POP}, "arm_metrics": {"arm1_words": "repaired all-tissue TITAN",
                                                           "arm2_words": "direct-1024 variant synth"}}, f)
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write("synthetic\n")
    return d


ANNOTATED = [f"P{i:02d}" for i in range(10)]


def write_folds(root, meta, annotated=ANNOTATED):
    pats = sorted(set(meta["patient"]))
    f = pd.DataFrame({"patient_id": pats, "fold": [i % 5 for i in range(len(pats))],
                      "has_annotations": [1 if p in annotated else 0 for p in pats], "has_retrieval": 1})
    p = os.path.join(root, "patient_folds.csv")
    f.to_csv(p, index=False)
    return p


def write_labels(root, meta, trained=ANNOTATED):
    """CONCH label file: training labels 1-5 only for `trained` patients, 0 (unlabelled) elsewhere."""
    rows = [{"slide_id": s, "patient_id": p, "label_1024": (1 + i % 5) if p in trained else 0}
            for i, (s, p) in enumerate(meta["patient"].items())]
    p = os.path.join(root, "labels_1024.csv.gz")
    pd.DataFrame(rows).to_csv(p, index=False, compression={"method": "gzip", "mtime": 0})
    return p


@pytest.fixture
def synth(tmp_path, monkeypatch):
    monkeypatch.setenv("PROSPER_OUT", str(tmp_path))
    meta, x = synth_meta()
    write_bundle(str(tmp_path), meta, x)
    folds = write_folds(str(tmp_path), meta)
    labels = write_labels(str(tmp_path), meta)
    args = ["--bundle-root", str(tmp_path), "--folds", folds, "--labels", labels, "--bundles", NAME]
    return tmp_path, meta, args


def run(tmp_path, args, out="out"):
    o = os.path.join(str(tmp_path), out)
    return U.main(args + ["--outdir", o]), o


def test_full_pool_reproduction_gate_passes_and_completes(synth):
    tmp_path, meta, args = synth
    code, o = run(tmp_path, args)
    assert code == 0 and os.path.exists(os.path.join(o, "COMPLETE"))
    rec = json.load(open(os.path.join(o, "reconciliation.json")))["checks"][NAME]
    for k in ("reproduce_saved_summary_frozen_function", "reproduce_saved_summary_this_script",
              "reproduce_saved_per_class_this_script", "per_query_recomputed_from_ranked_lists"):
        assert rec[k]["passed"], k
    assert rec["reproduce_saved_summary_this_script"]["max_abs_diff"] <= 1e-12
    r = json.load(open(os.path.join(o, "record.json")))
    assert r["B"] == 5000 and r["seed"] == 123 and r["reference_script"]["sha256"] == U.REF_SHA
    assert r["inputs"][NAME]["summary"]["sha256"] == U.sha256(r["inputs"][NAME]["summary"]["path"])


def test_reproduction_gate_refuses_tampered_summary(synth):
    tmp_path, meta, args = synth
    p = os.path.join(str(tmp_path), U.BUNDLE_FMT.format(NAME), "summary_common_pool.csv")
    s = pd.read_csv(p)
    s.loc[5, "hi"] += 1e-9
    s.to_csv(p, index=False)
    code, o = run(tmp_path, args)
    assert code == 1
    assert os.path.exists(os.path.join(o, "FAILED")) and not os.path.exists(os.path.join(o, "COMPLETE"))
    assert not os.path.exists(os.path.join(o, NAME))


def test_95_percent_level_mutant_caught(synth, monkeypatch):
    tmp_path, meta, args = synth

    def ci90(point, draws):
        r = T.ci(point, draws)
        if r["ci_available"]:
            r["lo"], r["hi"] = float(np.percentile(draws, 5)), float(np.percentile(draws, 95))
        return r

    monkeypatch.setattr(U, "ci", ci90)
    code, o = run(tmp_path, args)
    assert code == 1
    rec = json.load(open(os.path.join(o, "reconciliation.json")))["checks"][NAME]
    assert not rec["reproduce_saved_summary_this_script"]["passed"]
    assert rec["reproduce_saved_summary_frozen_function"]["passed"]


def test_patient_subset_membership(synth):
    tmp_path, meta, args = synth
    code, o = run(tmp_path, args)
    assert code == 0
    qs = pd.read_csv(os.path.join(o, NAME, "query_subsets.csv"), dtype=str)
    ann = {p: p < "P10" for p in set(meta["patient"])}
    exp = np.where([ann[p] for p in qs["patient"]], U.POP_ANN, U.POP_UNANN)
    assert (qs["subset"].to_numpy() == exp).all()
    assert (qs["has_annotations"] == np.where(exp == U.POP_ANN, "1", "0")).all()
    assert set(qs.loc[qs.subset == U.POP_ANN, "patient"]).isdisjoint(qs.loc[qs.subset == U.POP_UNANN, "patient"])
    s = pd.read_csv(os.path.join(o, NAME, "summary.csv"))
    n = s.groupby("population")[["n_queries", "n_patients"]].first()
    assert n.loc[U.POP_ANN].tolist() == [int((exp == U.POP_ANN).sum()), 10]
    assert n.loc[U.POP_UNANN].tolist() == [int((exp == U.POP_UNANN).sum()), 30]
    assert n.loc[POP].tolist() == [60, 40]
    c = pd.read_csv(os.path.join(o, NAME, "subset_counts.csv")).set_index("group")
    assert (c["n_queries_retained_unannotated"] + c["n_queries_excluded_annotated"] == c["n_queries_full"]).all()
    assert c.loc["ALL", "n_patients_excluded_annotated"] == 10


def test_subset_metrics_use_only_subset_queries_and_own_draws(synth):
    tmp_path, meta, args = synth
    code, o = run(tmp_path, args)
    pq = pd.read_csv(os.path.join(str(tmp_path), U.BUNDLE_FMT.format(NAME), "per_query_common_pool.csv"),
                     dtype={"patient": str})
    keep = ~pq["patient"].lt("P10")
    s = pd.read_csv(os.path.join(o, NAME, "summary.csv")).set_index(["population", "metric", "arm"])
    assert s.loc[(U.POP_UNANN, "micro_hit@5", "arm1"), "estimate"] == pytest.approx(pq.loc[keep, "arm1_hit@5"].mean(), abs=1e-15)
    sub = pq[keep]
    draws = T.make_patient_draws(sub["patient"], count=5000, seed=123)
    W = U.weights(sub["patient"].to_numpy(), draws)
    b = T.micro(sub["arm2_precision@5"].to_numpy(float), W) - T.micro(sub["arm1_precision@5"].to_numpy(float), W)
    r = s.loc[(U.POP_UNANN, "micro_precision@5", "delta")]
    assert r["lo"] == pytest.approx(np.percentile(b, 2.5), abs=1e-15)
    assert r["hi"] == pytest.approx(np.percentile(b, 97.5), abs=1e-15)


def test_paired_difference_sign_arm2_minus_arm1():
    q = [str(i) for i in range(12)]
    lab = ["A", "B", "C"] * 4
    pats = [f"P{i // 2}" for i in range(12)]
    base = pd.DataFrame({"query": q, "true_label": lab})
    arm1 = base.assign(**{c: 0.0 for c in U.ENDPOINTS})
    arm2 = base.assign(**{c: 1.0 for c in U.ENDPOINTS})
    arm1.loc[[0, 1], U.ENDPOINTS] = 1.0
    draws = T.make_patient_draws(pats, count=200, seed=123)
    s, g = U.evaluate(arm1, arm2, pats, ["A", "B", "C"], "x", draws)
    for df, keys in ((s, ["metric"]), (g, ["group", "metric"])):
        t = df.pivot_table(index=keys, columns="arm", values="estimate")
        assert np.allclose(t["delta"], t["arm2"] - t["arm1"])
        assert (t["delta"] > 0).all()
    assert s.set_index(["metric", "arm"]).loc[("micro_hit@5", "delta"), "estimate"] == pytest.approx(10 / 12)


def test_macro_unavailable_when_group_absent():
    q = [str(i) for i in range(6)]
    pq = pd.DataFrame({"query": q, "true_label": ["A", "B"] * 3, **{c: 1.0 for c in U.ENDPOINTS}})
    pats = [f"P{i}" for i in range(6)]
    s, _ = U.evaluate(pq, pq, pats, ["A", "B", "C"], "x", T.make_patient_draws(pats, count=50, seed=123))
    m = s[s.metric.str.startswith("macro_")]
    assert m["estimate"].isna().all() and (~m["ci_available"]).all()
    assert s[s.metric.str.startswith("micro_")]["ci_available"].all()


def test_readme_states_definitions(synth):
    tmp_path, meta, args = synth
    code, o = run(tmp_path, args)
    txt = open(os.path.join(o, NAME, "README.md")).read()
    for s in ("query-subset sensitivity", "reference library is unchanged", "has_annotations == 0",
              "arm2 - arm1", "B=5000", "seed=123", "PCG64", "2.5 and 97.5", "nominal 95%",
              "whole-patient percentile bootstrap", "multiplicity", "unweighted mean over the groups present",
              "descriptive contrast", "fold-held-out CONCH", "not a counterfactual", "1e-12", "arm2"):
        assert s in txt, s
    for s in ("never seen by any of the five CONCH heads", "Leak-free scope (`leak_free_scope`): "
              "both arms (CONCH-head arm vs unfiltered TITAN)", "patch selection is unchanged",
              "not a re-filtered, end-to-end patient-held-out pipeline", "not query leakage",
              "trained only on annotated patients other than the query patient",
              "10 of the 10 CONCH training patients have a slide in the 60-slide library; 0 do not.",
              "applies only to the `unannotated_patient_queries` rows", "filter fitted on annotated patients\"", "not \"filter independent of the library\"",
              "no direct held-out validation", "reseeded per population", "not paired with the full-pool intervals",
              "a null subset difference is not evidence of no bias", "not by a fold head",
              "every group is present in both subsets", "conch_training_patients_equal_has_annotations"):
        assert s in txt, s
    assert "never seen by any head" not in txt and U.S2_SENTENCE not in txt and "library patients" not in txt
    top = open(os.path.join(o, "README.md")).read()
    for s in ("query-subset sensitivity", "unchanged", "five CONCH heads", U.S2_SENTENCE, "leak_free_scope",
              "not a re-filtered, end-to-end patient-held-out pipeline", "not query leakage",
              "filter fitted on annotated patients", "no direct held-out validation", "unannotated-subset rows only",
              "10 of the 10 CONCH training patients have a slide in the 60-slide library; 0 do not.",
              "reseeded per population", "not evidence of no bias", U.SCOPES["both"]):
        assert s in top, s
    assert "library patients" not in top
    s = pd.read_csv(os.path.join(o, "summary_all_bundles.csv"))
    assert list(s.columns[:4]) == ["bundle", "arm1_words", "arm2_words", "leak_free_scope"]


def test_readme_flags_bundle_without_conch_arm(synth):
    tmp_path, meta, args = synth
    p = os.path.join(str(tmp_path), U.BUNDLE_FMT.format(NAME), "reconciliation.json")
    r = json.load(open(p))
    r["arm_metrics"]["arm2_words"] = "S2 tumour-enriched TITAN (UNI2 tile-random selection)"
    json.dump(r, open(p, "w"))
    code, o = run(tmp_path, args)
    assert code == 0
    txt = open(os.path.join(o, NAME, "README.md")).read()
    assert "Neither arm in this bundle uses the fold-held-out CONCH" in txt
    assert U.S2_SENTENCE in txt and U.SCOPES["none_s2"] in txt
    assert "population restriction only (neither arm uses CONCH heads)" in txt
    s = pd.read_csv(os.path.join(o, "summary_all_bundles.csv"), keep_default_na=False)
    assert set(s.loc[s.population == U.POP_UNANN, "leak_free_scope"]) == {U.SCOPES["none_s2"]}
    assert set(s.loc[s.population != U.POP_UNANN, "leak_free_scope"]) == {""}
    assert "quoted from the revision ledger, not recomputed" in txt


def test_refuses_nonempty_outdir(synth):
    tmp_path, meta, args = synth
    o = os.path.join(str(tmp_path), "out")
    os.makedirs(o)
    open(os.path.join(o, "x"), "w").close()
    assert U.main(args + ["--outdir", o]) == 2
    assert os.listdir(o) == ["x"]


def test_reference_sha_guard():
    with pytest.raises(U.CheckError):
        U.load_reference(expected="0" * 64)


REAL = os.path.join(os.environ.get("PROSPER_OUT", ""), U.BUNDLE_FMT.format("b100_argmax_vs_repaired"))


@pytest.mark.skipif(not os.path.isdir(REAL), reason="frozen bundle not available")
def test_real_bundle_counts_and_gate():
    folds = U.load_folds(os.path.join(os.environ["PROSPER_OUT"], U.FOLDS_REL))
    checks, res = U.run_bundle("b100_argmax_vs_repaired", REAL, folds, None, lambda s: None)
    assert all(c["passed"] for c in checks.values())
    sub = res[4]["subsets"]
    assert (sub[U.POP_ANN]["n_queries"], sub[U.POP_ANN]["n_patients"]) == (339, 164)
    assert (sub[U.POP_UNANN]["n_queries"], sub[U.POP_UNANN]["n_patients"]) == (2223, 1608)
    assert "LIVER" in set(res[1]["group"])


def test_leak_free_scope_mapping_for_bundle_arm_types():
    s2 = "S2 tumour-enriched TITAN (UNI2 tile-random selection)"
    rep = "repaired all-tissue TITAN"
    var = "direct-1024 variant b100_argmax"
    assert U.leak_free_scope(s2, var) == "CONCH arm only; S2 arm exposure not removed"
    assert U.leak_free_scope(rep, var) == "both arms (CONCH-head arm vs unfiltered TITAN)"
    assert U.leak_free_scope(rep, s2).startswith("population restriction only (neither arm uses CONCH heads)")
    with pytest.raises(U.CheckError):
        U.leak_free_scope("unrecorded", var)


def test_conch_training_premise_mismatch_fails(synth):
    tmp_path, meta, args = synth
    write_labels(str(tmp_path), meta, trained=ANNOTATED + ["P25"])  # an unannotated patient with labels
    code, o = run(tmp_path, args)
    assert code == 1 and os.path.exists(os.path.join(o, "FAILED")) and not os.path.exists(os.path.join(o, "COMPLETE"))
    g = json.load(open(os.path.join(o, "reconciliation.json")))["checks"]["global"]
    c = g["conch_training_patients_equal_has_annotations"]
    assert not c["passed"] and c["training_not_annotated"] == ["P25"]
    assert NAME not in json.load(open(os.path.join(o, "reconciliation.json")))["checks"]


def test_conch_training_premise_passes_and_is_recorded(synth):
    tmp_path, meta, args = synth
    code, o = run(tmp_path, args)
    assert code == 0
    c = json.load(open(os.path.join(o, "reconciliation.json")))["checks"]["global"][
        "conch_training_patients_equal_has_annotations"]
    assert c["passed"] and c["n_training_patients"] == c["n_has_annotations_1"] == 10
    r = json.load(open(os.path.join(o, "record.json")))["inputs"]["conch_labels_1024"]
    assert r["sha256"] == U.sha256(os.path.join(str(tmp_path), "labels_1024.csv.gz"))


def test_tampered_per_query_value_fails(synth):
    tmp_path, meta, args = synth
    p = os.path.join(str(tmp_path), U.BUNDLE_FMT.format(NAME), "per_query_common_pool.csv")
    pq = pd.read_csv(p, dtype=str)
    pq.loc[3, "arm1_hit@5"] = "0" if pq.loc[3, "arm1_hit@5"] in ("1", "1.0", "True") else "1"
    pq.to_csv(p, index=False)
    code, o = run(tmp_path, args)
    assert code == 1 and os.path.exists(os.path.join(o, "FAILED"))
    c = json.load(open(os.path.join(o, "reconciliation.json")))["checks"][NAME]
    assert not c["per_query_recomputed_from_ranked_lists"]["passed"]
    assert "arm1_hit@5" in c["per_query_recomputed_from_ranked_lists"]["mismatched_columns"]


def test_tampered_per_class_interval_fails(synth):
    tmp_path, meta, args = synth
    p = os.path.join(str(tmp_path), U.BUNDLE_FMT.format(NAME), "per_class_common_pool.csv")
    c = pd.read_csv(p)
    c.loc[c["lo"].notna().idxmax(), "lo"] -= 1e-9
    c.to_csv(p, index=False)
    code, o = run(tmp_path, args)
    assert code == 1 and os.path.exists(os.path.join(o, "FAILED"))
    r = json.load(open(os.path.join(o, "reconciliation.json")))["checks"][NAME]
    assert not r["reproduce_saved_per_class_this_script"]["passed"]
    assert r["reproduce_saved_summary_this_script"]["passed"]


def test_query_patient_missing_from_folds_fails_cleanly(synth):
    tmp_path, meta, args = synth
    fp = os.path.join(str(tmp_path), "patient_folds.csv")
    f = pd.read_csv(fp, dtype=str)
    f[f.patient_id != "P39"].to_csv(fp, index=False)
    code, o = run(tmp_path, args)
    assert code == 1 and os.path.exists(os.path.join(o, "FAILED"))
    r = json.load(open(os.path.join(o, "reconciliation.json")))
    assert "error" not in r  # a gate, not an exception
    c = r["checks"][NAME]["all_query_patients_in_patient_folds"]
    assert not c["passed"] and c["missing"] == ["P39"]


def test_subset_with_absent_group_reports_macro_unavailable(synth):
    tmp_path, meta, args = synth
    # sorted patients cycle A, B, C: keep only A/B patients among the annotated, so group C is absent there
    grp = meta.drop_duplicates("patient").set_index("patient")["label"]
    ann = [p for p in ANNOTATED if grp[p] != "C"]
    write_folds(str(tmp_path), meta, annotated=ann)
    write_labels(str(tmp_path), meta, trained=ann)
    code, o = run(tmp_path, args)
    assert code == 0
    s = pd.read_csv(os.path.join(o, NAME, "summary.csv"))
    m = s[(s.population == U.POP_ANN) & s.metric.str.startswith("macro_")]
    assert len(m) and m["estimate"].isna().all() and (~m["ci_available"]).all()
    u = s[(s.population == U.POP_UNANN) & s.metric.str.startswith("macro_")]
    assert u["estimate"].notna().all()
    rec = json.load(open(os.path.join(o, "record.json")))["bundles"][NAME]["subsets"]
    assert rec[U.POP_ANN]["groups_absent"] == ["C"] and rec[U.POP_UNANN]["groups_absent"] == []
    txt = open(os.path.join(o, NAME, "README.md")).read()
    assert f"`{U.POP_ANN}` lacks C (its macro is unavailable)" in txt
    assert "every group is present in both subsets" not in txt


def test_any_exception_writes_failed_marker(synth):
    tmp_path, meta, args = synth
    code, o = run(tmp_path, args[:-1] + ["does_not_exist"])
    assert code == 1
    assert os.path.exists(os.path.join(o, "FAILED")) and not os.path.exists(os.path.join(o, "COMPLETE"))
    r = json.load(open(os.path.join(o, "reconciliation.json")))
    assert r["error"].startswith("FileNotFoundError")
    assert os.path.exists(os.path.join(o, "record.json")) and os.path.exists(os.path.join(o, "run.log"))


def test_complete_binds_every_output_and_full_patient_list(synth):
    tmp_path, meta, args = synth
    code, o = run(tmp_path, args)
    assert code == 0
    c = json.load(open(os.path.join(o, "COMPLETE")))
    files = {os.path.relpath(os.path.join(r, f), o) for r, _, fs in os.walk(o) for f in fs} - {"COMPLETE"}
    assert set(c["outputs_sha256"]) == files and len(files) == 11
    for rel, h in c["outputs_sha256"].items():
        assert U.sha256(os.path.join(o, rel)) == h, rel
    b = json.load(open(os.path.join(o, "record.json")))["bundles"][NAME]
    assert b["full_pool_patient_list_sha256"] == hashlib.sha256(
        "\n".join(sorted(set(meta["patient"]))).encode()).hexdigest()
    assert b["subsets"][U.POP_ANN]["patient_list_sha256"] == hashlib.sha256("\n".join(ANNOTATED).encode()).hexdigest()


@pytest.mark.skipif(not os.path.isdir(REAL), reason="frozen bundle not available")
def test_real_conch_training_premise():
    folds = U.load_folds(os.path.join(os.environ["PROSPER_OUT"], U.FOLDS_REL))
    ok, info = U.conch_premise(os.path.join(os.environ["PROSPER_OUT"], U.LABELS_REL), folds)
    assert ok and info["n_training_patients"] == info["n_has_annotations_1"] == 192


def test_leak_free_scope_only_on_unannotated_rows(synth):
    tmp_path, meta, args = synth
    code, o = run(tmp_path, args)
    assert code == 0
    s = pd.read_csv(os.path.join(o, "summary_all_bundles.csv"), dtype=str, keep_default_na=False)
    u = s.population == U.POP_UNANN
    assert u.sum() > 0 and (~u).sum() > 0
    assert set(s.loc[~u, "population"]) == {POP, U.POP_ANN}
    assert (s.loc[u, "leak_free_scope"] == "both arms (CONCH-head arm vs unfiltered TITAN)").all()
    assert (s.loc[~u, "leak_free_scope"] == "").all()


def test_annotated_patient_without_training_labels_fails(synth):
    tmp_path, meta, args = synth
    write_labels(str(tmp_path), meta, trained=ANNOTATED[1:])  # P00 annotated but has no label 1-5 rows
    code, o = run(tmp_path, args)
    assert code == 1 and os.path.exists(os.path.join(o, "FAILED")) and not os.path.exists(os.path.join(o, "COMPLETE"))
    c = json.load(open(os.path.join(o, "reconciliation.json")))["checks"]["global"][
        "conch_training_patients_equal_has_annotations"]
    assert not c["passed"] and c["annotated_not_training"] == ["P00"] and c["training_not_annotated"] == []


def test_training_patients_outside_library_counted(synth):
    tmp_path, meta, args = synth
    fp = os.path.join(str(tmp_path), "patient_folds.csv")
    f = pd.read_csv(fp, dtype=str)
    extra = pd.DataFrame({"patient_id": ["Q90", "Q91"], "fold": "0", "has_annotations": "1", "has_retrieval": "0"})
    pd.concat([f, extra]).to_csv(fp, index=False)
    lp = os.path.join(str(tmp_path), "labels_1024.csv.gz")
    lab = pd.read_csv(lp, dtype=str)
    lab = pd.concat([lab, pd.DataFrame({"slide_id": ["9990", "9991"], "patient_id": ["Q90", "Q91"],
                                        "label_1024": ["3", "1"]})])
    lab.to_csv(lp, index=False, compression={"method": "gzip", "mtime": 0})
    code, o = run(tmp_path, args)
    assert code == 0
    b = json.load(open(os.path.join(o, "record.json")))["bundles"][NAME]["conch_training_patients"]
    assert b == {"n_training_patients": 12, "n_in_library": 10, "n_not_in_library": 2}
    want = "10 of the 12 CONCH training patients have a slide in the 60-slide library; 2 do not."
    assert want in open(os.path.join(o, NAME, "README.md")).read()
    assert want in open(os.path.join(o, "README.md")).read()
