"""Tests for scripts/r4a_vii_whole_patient.py (synthetic fixtures; nothing under /tmp).

Scratch files go to $PROSPER_OUT/r4a_vii_whole_patient_20261015/scratch/ and are removed afterwards.
The test that cross-checks the mirrored vote rules against prosper-search's
slide_eval.predict_labels_from_matrix imports prosper (about a minute on this filesystem);
it is skipped if prosper-search is not importable.
"""
import importlib.util
import json
import os
import shutil
import uuid

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(R, "scripts", "r4a_vii_whole_patient.py")
KERNEL = os.path.join(R, "scripts", "r2_2_tile_ladder_whole_patient.py")
OUT = os.environ.get("PROSPER_OUT", os.path.join(R, "prosper_revision_output"))
SCRATCH = os.path.join(OUT, "r4a_vii_whole_patient_20261015", "scratch")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load(SCRIPT, "r4a_vii")
K = _load(KERNEL, "ladder_independent_for_r4a_vii_test")  # independent load of the accepted kernel


@pytest.fixture
def scratch():
    d = os.path.join(SCRATCH, f"pytest_{uuid.uuid4().hex[:10]}")
    os.makedirs(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    try:
        os.rmdir(SCRATCH)
    except OSError:
        pass


def _multi_slide_frame(n_pat=60, seed=0):
    """Patients with 1-6 slides each; a patient-level latent makes slides of one patient agree."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        k = int(rng.integers(1, 7))
        good = rng.random() < 0.6
        lab = ["A", "B", "C"][p % 3]
        for s in range(k):
            rows.append(dict(query=f"q{p:03d}_{s}", patient=f"P{p:03d}", true_label=lab,
                             v=float(good if rng.random() < 0.95 else not good)))
    return pd.DataFrame(rows).set_index("query")


# ------------------------------------------------------------------ kernel identity
def test_kernel_is_imported_not_copied():
    for fn in ("make_patient_draws", "weight_matrix", "boot_means", "macro_reps", "ci", "plurality", "boundary_flag"):
        assert not hasattr(S, fn), f"{fn} must be used via the imported kernel module, not defined in the script"
        assert getattr(S.L, fn).__code__.co_code == getattr(K, fn).__code__.co_code
    assert os.path.samefile(S.LADDER, KERNEL)


# ------------------------------------------------------------------ shared draws => exact paired difference
def test_shared_draws_give_exact_paired_difference():
    fr = _multi_slide_frame()
    rng = np.random.default_rng(1)
    a = fr["v"].to_numpy()
    b = np.clip(a - (rng.random(len(a)) < 0.2), 0, 1)
    B = 400
    _, mult, W = S.L.weight_matrix(fr["patient"], B, 123)
    ra, rb = S.L.boot_means(W, a), S.L.boot_means(W, b)
    row = S.paired_row("t", "pop", "a", "b", "m", "micro", a.mean() - b.mean(), ra - rb, len(a), 60, B)
    # same draws, recomputed independently with the accepted kernel loaded separately
    pats, mult2 = K.make_patient_draws(list(fr["patient"]), count=B, seed=123)
    np.testing.assert_array_equal(mult, mult2)
    idx = pd.Index(pats).get_indexer(fr["patient"])
    W2 = mult2[:, idx].astype(float)
    d2 = (W2 @ a) / W2.sum(1) - (W2 @ b) / W2.sum(1)
    lo, hi, _ = K.ci(d2)
    assert row["ci_lo"] == pytest.approx(lo, abs=1e-15) and row["ci_hi"] == pytest.approx(hi, abs=1e-15)
    # per-replicate difference is not the difference of marginal endpoints
    la, ha, _ = K.ci(ra)
    lb, hb, _ = K.ci(rb)
    assert (row["ci_lo"], row["ci_hi"]) != pytest.approx((la - hb, ha - lb))
    assert row["ci_hi"] - row["ci_lo"] < (ha - lb) - (la - hb)


def test_c1d_paired_uses_same_rows_draws(monkeypatch):
    df = _c1d_frame()
    d = dict(frame=df, saved=_c1d_saved(df))
    g = S.Gates()
    out = S.c1d_boot({}, g, d, 300, 123)
    pats, mult = K.make_patient_draws(list(df["patient"]), count=300, seed=123)
    W = mult[:, pd.Index(pats).get_indexer(df["patient"])].astype(float)
    same = (df["kind"] == "same_patient").to_numpy()
    for m in S.C1D_METRICS:
        v = df[m].to_numpy(float)
        dd = (W[:, same] @ v[same]) / W[:, same].sum(1) - (W[:, ~same] @ v[~same]) / W[:, ~same].sum(1)
        lo, hi, _ = K.ci(dd)
        row = next(r for r in out["paired"] if r["metric"] == m)
        assert row["ci_lo"] == pytest.approx(lo, abs=1e-12) and row["ci_hi"] == pytest.approx(hi, abs=1e-12)
        assert row["point"] == pytest.approx(v[same].mean() - v[~same].mean(), abs=1e-15)


# ------------------------------------------------------------------ whole-patient vs independent-slide
def test_whole_patient_differs_from_independent_slide_when_patients_have_multiple_slides():
    fr = _multi_slide_frame(n_pat=80, seed=3)
    assert fr.groupby("patient").size().max() > 1
    v = fr["v"].to_numpy()
    _, _, Wp = S.L.weight_matrix(fr["patient"], 2000, 123)
    _, _, Ws = S.L.weight_matrix(pd.Series(fr.index, index=fr.index), 2000, 123)  # each slide its own unit
    lp, hp, _ = S.L.ci(S.L.boot_means(Wp, v))
    ls, hs, _ = S.L.ci(S.L.boot_means(Ws, v))
    assert (lp, hp) != (ls, hs)
    assert (hp - lp) > 1.2 * (hs - ls)  # within-patient correlation widens the whole-patient interval
    # singleton patients: patient and slide resampling have the same unit count and structure
    single = fr.groupby("patient").head(1)
    _, _, W1 = S.L.weight_matrix(single["patient"], 500, 123)
    _, _, W2 = S.L.weight_matrix(pd.Series(single["patient"].to_numpy(), index=single.index), 500, 123)
    np.testing.assert_array_equal(W1, W2)


# ------------------------------------------------------------------ A-3 plurality tie-break
@pytest.mark.parametrize("labs,expected", [
    (["A", "B", "A", "B", "C"], "A"),
    (["B", "A", "A", "B", "C"], "B"),
    (["C", "A", "B", "D", "E"], "C"),
    (["A", "B", "B", "C", "C"], "B"),
    (["D", "C", "C", "D", "A"], "D"),
    (["A", "A", "B", "B", "B"], "B"),
])
def test_a3_plurality_matches_kernel(labs, expected):
    labs10 = labs + ["Z"] * 5
    row = S.family_row(labs10, "A")
    assert row["plurality5_pred"] == K.plurality(labs) == expected
    assert row["plurality5"] == int(expected == "A")


def test_a3_plurality_random_lists_match_kernel():
    rng = np.random.default_rng(7)
    for _ in range(500):
        labs = list(rng.choice(list("ABCD"), size=10))
        assert S.family_row(labs, "A")["plurality5_pred"] == K.plurality(labs[:5])


# ------------------------------------------------------------------ weighted F1 replicate
@pytest.mark.parametrize("micro", ["f1", "accuracy"])
def test_weighted_f1_matches_sklearn_sample_weight(micro):
    rng = np.random.default_rng(11)
    classes = ["A", "B", "C", "D"]
    n = 300
    truth = rng.choice(classes, size=n)
    pred = np.where(rng.random(n) < 0.6, truth, rng.choice(classes + [S.AMBIG], size=n))
    W = rng.integers(0, 4, size=(25, n)).astype(float)
    W[0] = 1.0
    mac, mic = S.f1_reps(W, truth, pred, classes, micro=micro)
    for b in range(W.shape[0]):
        w = W[b]
        if any(w[truth == c].sum() == 0 for c in classes):
            assert np.isnan(mac[b])
            continue
        exp_mac = f1_score(truth, pred, labels=classes, average="macro", sample_weight=w, zero_division=0)
        assert mac[b] == pytest.approx(exp_mac, abs=1e-12)
        if micro == "f1":
            exp_mic = f1_score(truth, pred, labels=classes, average="micro", sample_weight=w, zero_division=0)
        else:
            exp_mic = float(np.sum(w * (truth == pred)) / w.sum())
        assert mic[b] == pytest.approx(exp_mic, abs=1e-12)


def test_weighted_f1_absent_class_replicate_is_nan():
    truth = np.array(["A", "A", "B"])
    pred = np.array(["A", "B", "B"])
    W = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    mac, _ = S.f1_reps(W, truth, pred, ["A", "B"])
    assert np.isnan(mac[0]) and np.isfinite(mac[1])


# ------------------------------------------------------------------ depth derivation and tie census
def test_derive_depth_self_excluded_stable_and_tie_census():
    vals = np.array([[0.0, 0.2, 0.2, 0.5, 0.1],
                     [0.2, 0.0, 0.3, 0.3, 0.3],
                     [0.2, 0.3, 0.0, 0.1, 0.4],
                     [0.5, 0.3, 0.1, 0.0, 0.2],
                     [0.1, 0.3, 0.4, 0.2, 0.0]])
    idx, dist = S.derive_depth(vals, list("abcde"), depth=4)
    assert idx[0].tolist() == [4, 1, 2, 3]  # stable among the 0.2 tie (column order)
    assert idx[1].tolist() == [0, 2, 3, 4]
    assert not any(i in row for i, row in enumerate(idx.tolist()))
    big = np.zeros((3, 11))
    big[:, :] = np.arange(11)
    big[1, 10] = big[1, 9]
    tc = S.tie_census(big)
    assert tc["tie_rank10_rank11"] == 1 and tc["n_queries"] == 3


# ------------------------------------------------------------------ mirrored vote rules (pure)
def test_mirror_rules_semantics():
    assert S.generous_majority(["B", "A", "B", "A", "C"], "A") == "A"  # truth in tied max set
    assert S.generous_majority(["B", "C", "C", "B", "A"], "A") == "B"  # first max label in rank order
    assert S.absolute_majority(["A", "A", "A", "B", "C"], 5) == "A"
    assert S.absolute_majority(["A", "A", "B", "B", "C"], 5) == S.AMBIG
    assert S.absolute_majority(["A", "A", "B", "B", "C", "C", "C", "C", "C", "C"], 10) == "C"


def _prosper_or_skip():
    try:
        return S.prosper_modules()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"prosper-search not importable: {e}")


def test_mirror_rules_and_depth_match_prosper_predict_labels_from_matrix():
    slide_eval, _, _ = _prosper_or_skip()
    rng = np.random.default_rng(5)
    n = 60
    ids = [str(1000 + i) for i in range(n)]
    labs = rng.choice(list("ABCD"), size=n)
    pats = [f"P{i // 3}" for i in range(n)]
    x = np.round(rng.random((n, n)), 1)  # coarse values => many ties, exercising the stable sort
    x = (x + x.T) / 2
    np.fill_diagonal(x, 0.0)
    sim = pd.DataFrame(x, index=ids, columns=ids)
    meta = pd.DataFrame({S.ID: ids, S.LAB: labs, S.PAT: pats})
    idx, _ = S.derive_depth(sim.values, ids, depth=11)
    lab = dict(zip(ids, labs))
    for k in (5, 10):
        for rule in ("generous_majority", "absolute"):
            dfp = slide_eval.predict_labels_from_matrix(sim, meta, k, S.ID, S.LAB, S.PAT, larger_is_better=False,
                                                        tie_break=rule, exclude_same_patient=False)
            for i, (q, t, p, tl) in enumerate(zip(dfp[S.ID], dfp["true_label"], dfp["pred_label"], dfp["topk_labels"])):
                mine_top = [lab[ids[j]] for j in idx[i, :k]]
                assert list(tl) == mine_top
                mine = S.generous_majority(mine_top, t) if rule == "generous_majority" else S.absolute_majority(mine_top, k)
                assert mine == p, (q, rule, k)


# ------------------------------------------------------------------ gate failure -> FAILED.json
def _c1d_frame(n_pat=12, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        n_pairs = int(rng.integers(1, 4))
        for j in range(n_pairs):
            for kind in ("same_patient", "class_matched_null"):
                r = {m: float(rng.random()) for m in S.C1D_METRICS + S.C1D_EXTRA}
                r.update(patient=f"P{p:02d}", a=f"{p}{j}a", b=f"{p}{j}b", class_a="X", class_b="X", kind=kind)
                rows.append(r)
    return pd.DataFrame(rows)


def _c1d_saved(df):
    same, null = df[df.kind == "same_patient"], df[df.kind == "class_matched_null"]
    cols = S.C1D_METRICS + S.C1D_EXTRA
    return dict(config=dict(bootstrap_B=2000, seed=123), n_same_patient_pairs=len(same), n_null_pairs=len(null),
                n_patients_multi_slide=int(df.patient.nunique()),
                point=dict(same_patient={c: float(same[c].mean()) for c in cols},
                           class_matched_null={c: float(null[c].mean()) for c in cols},
                           difference={c: float(same[c].mean() - null[c].mean()) for c in cols}),
                difference_ci_over_patients={c: [0.0, 0.0] for c in cols})


def _write_c1d(scratch, df, saved, monkeypatch):
    pp, jp = os.path.join(scratch, "pairs.csv"), os.path.join(scratch, "consistency.json")
    df.to_csv(pp, index=False)
    json.dump(saved, open(jp, "w"))
    same = int((df.kind == "same_patient").sum())
    monkeypatch.setattr(S, "C1D_PAIRS", pp)
    monkeypatch.setattr(S, "C1D_JSON", jp)
    monkeypatch.setattr(S, "C1D_ROWS", len(df))
    monkeypatch.setattr(S, "C1D_PAIRS_EACH", same)
    monkeypatch.setattr(S, "C1D_PATIENTS", int(df.patient.nunique()))
    return pp, jp


def test_c1d_synthetic_passes_and_writes_complete(scratch, monkeypatch):
    df = _c1d_frame()
    _write_c1d(scratch, df, _c1d_saved(df), monkeypatch)
    o = os.path.join(scratch, "run_ok")
    rc = S.main(["--arms", "c1d", "--B", "200", "--out-dir", o])
    assert rc == 0
    assert os.path.exists(os.path.join(o, "COMPLETE")) and not os.path.exists(os.path.join(o, "FAILED.json"))
    for f in ("summary.csv", "paired_differences.csv", "per_pair_frame.csv.gz", "reconciliation.json",
              "input_bindings.json", "README.md", "sha256.txt"):
        assert os.path.exists(os.path.join(o, f)), f
    rec = json.load(open(os.path.join(o, "reconciliation.json")))
    assert all(g["status"] == "PASS" for g in rec["gates"])
    assert rec["c1d"]["legacy_vs_new"]["jaccard_top5_ids"]["label"].startswith("patient-resample-B2000-legacy")


def test_fidelity_gate_failure_writes_failed_json(scratch, monkeypatch):
    df = _c1d_frame()
    saved = _c1d_saved(df)
    saved["point"]["same_patient"]["both_hit5"] += 1e-6  # outside the 1e-9 gate
    _write_c1d(scratch, df, saved, monkeypatch)
    o = os.path.join(scratch, "run_fail")
    rc = S.main(["--arms", "c1d", "--B", "200", "--out-dir", o])
    assert rc == 1
    assert not os.path.exists(os.path.join(o, "COMPLETE"))
    f = json.load(open(os.path.join(o, "FAILED.json")))
    assert f["first_failed_gate"]["arm"] == "c1d" and f["first_failed_gate"]["gate"] == "fidelity_saved_points"
    assert "both_hit5" in f["first_failed_gate"]["detail"]
    summ = pd.read_csv(os.path.join(o, "summary.csv")) if os.path.getsize(os.path.join(o, "summary.csv")) > 1 else pd.DataFrame()
    assert summ.empty  # no intervals from a failed arm


def test_hash_mismatch_is_a_gate_failure(scratch, monkeypatch):
    df = _c1d_frame()
    pp, _ = _write_c1d(scratch, df, _c1d_saved(df), monkeypatch)
    monkeypatch.setitem(S.EXPECTED_SHA256, pp, "0" * 64)
    o = os.path.join(scratch, "run_hash")
    assert S.main(["--arms", "c1d", "--B", "50", "--out-dir", o]) == 1
    f = json.load(open(os.path.join(o, "FAILED.json")))
    assert f["first_failed_gate"]["gate"].startswith("input_sha256:")


def test_out_dir_must_not_exist(scratch):
    with pytest.raises(FileExistsError):
        S.main(["--arms", "c1d", "--out-dir", scratch])


# ------------------------------------------------------------------ amendment 1: same-class C1d arm
def _c1d_producer_frame(n_pat=14, seed=5):
    """Frame laid out as patient_consistency.py writes it: per patient, combinations(sorted(slides), 2), each
    same-patient row followed by its own class-matched null (anchor a, partner from another patient).
    Some patients mix classes, so some same-patient pairs are cross-class; `marker` identifies each pair."""
    from itertools import combinations
    rng = np.random.default_rng(seed)
    slides = {}
    for p in range(n_pat):
        k = int(rng.integers(2, 5))
        slides[f"P{p:02d}"] = {f"s{p:02d}_{j}": ("X" if (j == 0 or rng.random() < 0.6) else "Y") for j in range(k)}
    by_class = {}
    for p, ss in slides.items():
        for s, c in ss.items():
            by_class.setdefault(c, []).append((s, p))
    rows, marker = [], 0
    for p, ss in slides.items():
        for x, y in combinations(sorted(ss), 2):
            base = {m: float(rng.random()) for m in S.C1D_METRICS + ["both_generous5"]}
            rows.append(dict(base, same_class=float(ss[x] == ss[y]), patient=p, a=x, b=y, class_a=ss[x],
                             class_b=ss[y], kind="same_patient", marker=marker))
            cands = [s for s, q in by_class[ss[x]] if q != p]
            c = cands[int(rng.integers(len(cands)))]
            nb = {m: float(rng.random()) for m in S.C1D_METRICS + ["both_generous5"]}
            rows.append(dict(nb, same_class=1.0, patient=p, a=x, b=c, class_a=ss[x], class_b=ss[x],
                             kind="class_matched_null", marker=marker))
            marker += 1
    return pd.DataFrame(rows)


def test_same_class_selection_drops_cross_class_pairs_and_their_own_nulls(monkeypatch):
    df = _c1d_producer_frame()
    same = df[df.kind == "same_patient"]
    keep_markers = set(same.loc[same.same_class == 1.0, "marker"])
    drop_markers = set(same.loc[same.same_class == 0.0, "marker"])
    assert drop_markers and keep_markers  # the fixture has cross-class pairs
    # an anchor with both a same-class and a cross-class pair: the anchor-only key would be ambiguous
    g = same.groupby(["patient", "a"]).same_class.nunique()
    assert (g > 1).any()
    monkeypatch.setattr(S, "C1D_SC_EACH", len(keep_markers))
    out = S.c1d_same_class_select(S.Gates(), dict(frame=df, saved=None, nonfinite=None))
    r = out["frame"]
    assert set(r.marker) == keep_markers and not (set(r.marker) & drop_markers)
    assert (r.kind == "same_patient").sum() == (r.kind == "class_matched_null").sum() == len(keep_markers)
    for m, grp in r.groupby("marker"):  # each kept pair keeps exactly its own null row
        assert sorted(grp.kind) == ["class_matched_null", "same_patient"]
        assert grp.pair_id.nunique() == 1
    assert (r.loc[r.kind == "same_patient", "same_class"] == 1.0).all()
    sel = out["selection"]
    assert sel["n_pairs_dropped"] == len(drop_markers)
    assert sel["n_patients_remaining"] == r.patient.nunique()
    assert set(sel["patients_dropped"]) == set(df.patient) - set(r.patient)
    assert sel["ambiguity"]["n_anchor_keys_with_mixed_same_class"] == int((g > 1).sum())


def test_same_class_count_mismatch_and_broken_pairing_fail():
    df = _c1d_producer_frame()
    with pytest.raises(S.GateFailure) as e:  # C1D_SC_EACH is the production 1,768
        S.c1d_same_class_select(S.Gates(), dict(frame=df, saved=None, nonfinite=None))
    assert e.value.gate == "restricted_counts"
    swapped = df.copy()
    swapped.iloc[[2, 3]] = swapped.iloc[[3, 2]].to_numpy()  # a null before its same-patient row
    with pytest.raises(S.GateFailure) as e:
        S.c1d_same_class_select(S.Gates(), dict(frame=swapped, saved=None, nonfinite=None))
    assert e.value.gate == "pairing_key_positional_unambiguous"
    wrong_anchor = df.copy()
    wrong_anchor.loc[1, "a"] = "elsewhere"
    with pytest.raises(S.GateFailure) as e:
        S.c1d_same_class_select(S.Gates(), dict(frame=wrong_anchor, saved=None, nonfinite=None))
    assert e.value.gate == "pairing_key_positional_unambiguous"


def _sc_setup(scratch, monkeypatch):
    df = _c1d_producer_frame().drop(columns="marker")
    _write_c1d(scratch, df, _c1d_saved(df), monkeypatch)
    same = df[df.kind == "same_patient"]
    monkeypatch.setattr(S, "C1D_SC_EACH", int((same.same_class == 1.0).sum()))
    ref = os.path.join(scratch, "reference")
    assert S.main(["--arms", "c1d", "--B", "200", "--out-dir", ref]) == 0
    return df, ref


def test_same_class_arm_end_to_end_complete_and_all_pairs_identical(scratch, monkeypatch):
    df, ref = _sc_setup(scratch, monkeypatch)
    o = os.path.join(scratch, "run_sc")
    assert S.main(["--arms", "c1d_same_class", "--B", "200", "--out-dir", o, "--c1d-reference-dir", ref]) == 0
    assert os.path.exists(os.path.join(o, "COMPLETE"))
    rec = json.load(open(os.path.join(o, "reconciliation.json")))
    assert all(g["status"] == "PASS" for g in rec["gates"])
    assert any(g["gate"] == "all_pairs_c1d_identical_to_reference" for g in rec["gates"])
    sc = rec["c1d_same_class"]
    assert set(sc["all_pairs_vs_same_class"]) == set(S.C1D_METRICS + S.C1D_EXTRA)
    assert sc["disclosure"] == S.C1D_SC_DISCLOSURE
    summ = pd.read_csv(os.path.join(o, "summary.csv"))
    same_rows = summ[(summ.arm == "c1d_same_class") & (summ.config == "same_patient")]
    assert (same_rows.n_queries == S.C1D_SC_EACH).all()
    keep = df[(df.kind == "same_patient") & (df.same_class == 1.0)]
    m = same_rows.set_index("metric").point
    for c in S.C1D_METRICS:
        assert abs(m[c] - keep[c].mean()) < 1e-12
    assert "Binding disclosure" in open(os.path.join(o, "README.md")).read()
    assert os.path.exists(os.path.join(o, "per_pair_frame_same_class.csv.gz"))


def test_same_class_arm_reference_mismatch_writes_failed_json(scratch, monkeypatch):
    df, ref = _sc_setup(scratch, monkeypatch)
    p = os.path.join(ref, "summary.csv")
    lines = open(p).read().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("c1d,"))
    lines[i] = lines[i].replace("c1d_pairs", "c1d_pairz", 1)
    open(p, "w").write("\n".join(lines) + "\n")
    o = os.path.join(scratch, "run_sc_bad")
    assert S.main(["--arms", "c1d,c1d_same_class", "--B", "200", "--out-dir", o, "--c1d-reference-dir", ref]) == 1
    f = json.load(open(os.path.join(o, "FAILED.json")))
    assert f["first_failed_gate"]["gate"] == "all_pairs_c1d_identical_to_reference"
    assert not os.path.exists(os.path.join(o, "COMPLETE"))
