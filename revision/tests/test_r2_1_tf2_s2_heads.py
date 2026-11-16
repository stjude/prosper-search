"""Synthetic-fixture tests for scripts/r2_1_tf2_s2_heads.py (R2.1-TF2 S2) and the S1 canary rule.

Mocked banks only (the S0 fixture builder from test_r2_1_tf2_s0_pool_folds.py plus synthetic S1
outputs). Covers: ordered-id digest identity, the width wrapper, budget/manifest rules, fit_generic ==
fit_head on the same inputs, width generalisation (1,280 / 768), the dry-run head plan (56 heads),
per-head fits, F7 checkpoint (ii) pass and leak detection, the classifier report and the R3 canary
rule. Run with
    python3 -m pytest -p no:cacheprovider --basetemp=<tmp> tests/test_r2_1_tf2_s2_heads.py
"""
import gzip
import hashlib
import importlib.util
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S2 = _load("r2_1_tf2_s2_heads", REPO / "scripts" / "r2_1_tf2_s2_heads.py")
S1 = _load("r2_1_tf2_s1_jobspec", REPO / "scripts" / "r2_1_tf2_s1_jobspec.py")
S0 = S2.S0
kernel, fitter = S2.kernel, S2.fitter
import test_r2_1_tf2_s0_pool_folds as T0  # noqa: E402  (fixture builder)
from prosper.evaluation import r2_1_hoptimus_filter_cli as filter_cli  # noqa: E402


@pytest.fixture(autouse=True)
def single_thread_pools():
    with threadpool_limits(limits=1):
        yield


# ---------------------------------------------------------------- unit tests
def _records(n=40, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        s, x, y = f"s{i % 4}", 512 * (i // 4), 512 * (i % 3)
        recs.append(kernel.AnnotationRecord(kernel.physical_tile_id(s, x, y), s, f"p{i % 8}", "G", 0, x, y,
                                            512, int(rng.integers(1, 6)) if i >= 5 else i + 1, i // 4))
    return recs


def test_ordered_ids_sha_matches_accepted_convention():
    recs = _records()
    assert S2.ordered_ids_sha([r.tile_id for r in recs]) == filter_cli._ordered_ids_sha(recs)
    assert S2.ordered_ids_sha([r.tile_id for r in recs]) == fitter._order_digest(tuple(recs))


def test_fit_width_restores_constant():
    assert fitter._WIDTH == 1536
    with pytest.raises(RuntimeError):
        with S2.fit_width(768):
            assert fitter._WIDTH == 768
            raise RuntimeError("x")
    assert fitter._WIDTH == 1536


def test_select_budget_five_percent_rule():
    labels = np.array([1] * 45 + [2] * 3 + [3] * 30 + [4] * 21 + [5] * 1)
    keep = S2.select_budget(labels, 5)
    got = np.bincount(labels[keep], minlength=6)[1:]
    assert got.tolist() == [kernel.five_percent_count(n) for n in (45, 3, 30, 21, 1)] == [2, 1, 2, 1, 1]
    assert S2.select_budget(labels, 100).tolist() == list(range(len(labels)))
    with pytest.raises(S2.S2Error):
        S2.select_budget(np.array([1, 1, 2, 3, 4]), 5)  # class 5 absent


def _tile_frame(n=300, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        s = f"{100 + i % 12}"
        p = f"P{(i % 12) // 2}"
        x, y = 512 * (i // 12), 0
        lab = int(rng.integers(1, 6))
        rec = kernel.AnnotationRecord(kernel.physical_tile_id(s, x, y), s, p, "G", 0, x, y, 512, lab, i // 12)
        rows.append({"tile_id": rec.tile_id, "slide_id": s, "patient_id": p, "tissue_label": lab, "x": x, "y": y,
                     "feature_row": i // 12, "training_hash": kernel.training_tile_hash(rec),
                     "qc_removed_slide": int(s == "111")})
    return pd.DataFrame(rows)


def test_build_manifests_rules():
    t = _tile_frame()
    fold = {f"P{k}": k % 5 for k in range(6)}
    mans, held = S2.build_manifests(t, 512, fold, prefix="px512")
    assert len(mans) == len(S2.SLOTS) * 2 and len(held) == 6
    for slot in S2.SLOTS:
        a, b = mans[f"px512_b5_{slot}"], mans[f"px512_b100_{slot}"]
        assert set(a["frame"].tile_id) <= set(b["frame"].tile_id)
        for m in (a, b):
            st = m["class_stats"]
            assert [s["balanced_weight"] for s in st] == [m["selected_count"] / (5 * s["selected_count"]) for s in st]
        if slot.startswith("fold"):
            f = int(slot[4:])
            assert not any(fold[p] == f for p in b["frame"].patient_id)
            assert b["pool_count"] == int((t.patient_id.map(fold) != f).sum())
    assert mans["px512_b100_full"]["pool_count"] == len(t)
    assert set(mans["px512_b100_full_noqc"]["frame"].slide_id) == set(t.slide_id) - {"111"}
    # hash order identical to the kernel's (fit_head's validator accepts the fold manifest)
    m = mans["px512_b5_fold1"]
    recs = S2.records_512(m["frame"])
    head = {k: m[k] for k in ("class_stats", "pool_count", "selected_count", "selected_fraction")}
    head.update(held_out_fold=1, budget_percent=5)
    full_fold = {**fold}
    fitter._validate_manifest(S2.manifest_object(head, m["frame"], recs), full_fold)
    # held-out sets partition the pool in physical order
    assert sum(h["rows"] for k, h in held.items() if "fold" in k) == len(t)


def _synthetic_fit_inputs(width, n=200, seed=5):
    t = _tile_frame(n, seed)
    fold = {f"P{k}": k % 5 for k in range(6)}
    mans, _ = S2.build_manifests(t, 512, fold, prefix="px512")
    m = mans["px512_b100_fold0"]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((len(m["frame"]), width)).astype(np.float32)
    X += m["frame"]["tissue_label"].to_numpy()[:, None].astype(np.float32)
    return m, fold, X


def test_fit_generic_equals_fit_head_and_width_generalises():
    for width in (1536, 1280, 768):
        m, fold, X = _synthetic_fit_inputs(width)
        recs = S2.records_512(m["frame"])
        head = {k: m[k] for k in ("class_stats", "pool_count", "selected_count", "selected_fraction")}
        head.update(held_out_fold=0, budget_percent=100)
        with S2.fit_width(width):
            a = fitter.fit_head(X, S2.manifest_object(head, m["frame"], recs), fold)
        weights = {s["tissue_label"]: s["balanced_weight"] for s in m["class_stats"]}
        b = S2.fit_generic(X, m["frame"]["tissue_label"].to_numpy(np.int64), weights, width=width, context="t",
                           held_out_fold=0, budget=100, frame=m["frame"], class_stats_rows=m["class_stats"],
                           pool_count=m["pool_count"], selected_fraction=m["selected_fraction"])
        assert a.diagnostics["fitted_state_sha256"] == b.diagnostics["fitted_state_sha256"]
        assert a.diagnostics["ordered_tile_ids_sha256"] == b.diagnostics["ordered_tile_ids_sha256"]
        assert b.estimator.coef_.shape == (5, width)
        pa, la = S2.predict(a, X[:50], width)
        pb, lb = S2.predict(b, X[:50], width)
        assert pa.tobytes() == pb.tobytes() and la.tobytes() == lb.tobytes()
    assert fitter._WIDTH == 1536
    with pytest.raises(fitter.HeadValidationError):  # wrong width outside the wrapper is refused
        fitter.predict_head(b, X[:5])


def test_s1_canary_rule():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((200, 16)).astype(np.float32)
    new = ref.copy()
    new[0] += np.float32(1e-3) * np.abs(ref[0]).max()     # rel 1e-3 on 1 row: bitwise 99.5 %
    assert S1.compare_rows(ref, new)["pass"]
    bad = ref.copy()
    bad[0, 0] += np.float32(3e-3) * np.abs(ref[0]).max()  # rel 3e-3 > 2e-3
    r = S1.compare_rows(ref, bad)
    assert not r["pass"] and r["bitwise_fraction"] == 0.995
    many = ref.copy()
    many[:3] *= np.float32(1.0001)                          # 98.5 % bitwise < 99 %
    assert not S1.compare_rows(ref, many)["pass"]


# ---------------------------------------------------------------- pipeline fixture
def _make_s1_outputs(po: Path, inputs: dict, s0_dir: Path):
    """Synthetic S1a / S1c outputs for the non-library pool slides (N1, N2) + 1024 supplement."""
    import h5py
    s1 = po / S0.RUN_NAME / "s1"
    ho = s1 / "h_optimus_1_all_patches_tf2"
    ho.mkdir(parents=True)
    rng = np.random.default_rng(9)
    cells = {}
    for s in ("N1", "N2"):
        lab = pd.read_csv(Path(inputs["label_dir"]) / f"{s}.csv")
        tiles = [{"wsi_loc": [int(x), int(y)]} for x, y in zip(lab.wsi_loc_x, lab.wsi_loc_y)]
        cells[s] = {(int(x) // 512, int(y) // 512): int(v) for x, y, v in zip(lab.wsi_loc_x, lab.wsi_loc_y, lab.patch_label)}
        pickle.dump((T0.informative(lab.patch_label.tolist(), 1536, rng), tiles), open(ho / f"{s}.pkl", "wb"))
    tr = s1 / "conch_v15_tf2" / "chunk1" / "job_dir" / "40x_1024px_0px_overlap"
    (tr / "patches").mkdir(parents=True)
    (tr / "features_conch_v15").mkdir()
    g = np.array([(0, 0), (1024, 0), (0, 1024), (1024, 1024)], np.int64)
    for s in ("N1", "N2"):
        with h5py.File(tr / "patches" / f"{s}_patches.h5", "w") as h:
            ds = h.create_dataset("coords", data=g)
            for k, v in dict(level0_width=2048, level0_height=2048, patch_size=1024,
                             patch_size_level0=1024, overlap=0).items():
                ds.attrs[k] = v
        with h5py.File(tr / "features_conch_v15" / f"{s}.h5", "w") as h:
            h.create_dataset("coords", data=g)
            h.create_dataset("features", data=T0.informative(T0.window_labels_1024(cells[s], g.tolist()), 768, rng))
    sup = S0.check_out_dir(s1 / "labels_1024_nonlibrary", str(po), "s1")
    assert S0.run_supplement(s0_dir, tr, sup, inputs, workers=1, log=lambda *a: None)["status"] == "COMPLETE"
    return sup / "labels_1024_supplement.csv.gz"


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    import types
    mod = types.ModuleType("openslide")
    mod.OpenSlide = T0._FakeSlide
    monkeypatch.setitem(sys.modules, "openslide", mod)
    po, inputs, expected = T0.build_fixture(tmp_path, T0._FakeSlide)
    s0_dir = S0.check_out_dir(po / S0.RUN_NAME / "s0", str(po), "s0")
    assert S0.run_s0(s0_dir, inputs, expected, workers=1, hash_workers=2, log=lambda *a: None)["status"] == "COMPLETE"
    return po, inputs, s0_dir


def test_dry_run_blocks_without_s1(pipeline):
    po, inputs, s0_dir = pipeline
    plan_dir = S0.check_out_dir(po / S0.RUN_NAME / "s2" / "plan_001", str(po), "s2")
    plan = S2.run_dry(s0_dir, plan_dir, str(po), inputs=inputs, log=lambda *a: None)
    heads = plan["heads"]
    assert len(heads) == 56 and sum(h["kind"] == "primary" for h in heads) == 48
    st = {h["head_id"]: h["status"] for h in heads}
    assert all(st[f"uni2/b{b}/{s}"] == "ready" for b in (5, 100) for s in S2.SLOTS)
    assert all(st[f"virchow2/b{b}/{s}"] == "ready" for b in (5, 100) for s in S2.SLOTS)
    assert all(st[f"h_optimus_1/b{b}/{s}"] == "blocked" for b in (5, 100) for s in S2.SLOTS)
    assert all(st[f"conch_v15/b{b}/{s}"] == "blocked" for b in (5, 100) for s in S2.SLOTS)
    assert plan["labels_1024_missing_slides"] == ["N1", "N2"]
    # identical tile identities for the three 512 encoders: same manifest id
    mids = {h["manifest_id"] for h in heads if h["encoder"] != "conch_v15" and h["slot"] == "fold0" and h["budget_percent"] == 5}
    assert mids == {"px512_b5_fold0"}
    assert "bsub " not in plan["lsf"]["script"].replace("#BSUB", "")
    assert plan["lsf"]["ready_indices"] == "1-28"
    # fitting a blocked head is refused
    with pytest.raises(S2.S2Error, match="blocked"):
        S2.run_fit(plan_dir, po / S0.RUN_NAME / "s2", str(po), head_id="h_optimus_1/b5/fold0", inputs=inputs)


def test_full_pipeline_fit_f7_report(pipeline):
    po, inputs, s0_dir = pipeline
    sup = _make_s1_outputs(po, inputs, s0_dir)
    s2 = po / S0.RUN_NAME / "s2"
    plan_dir = S0.check_out_dir(s2 / "plan_002", str(po), "s2")
    plan = S2.run_dry(s0_dir, plan_dir, str(po), supplements=[sup], inputs=inputs, log=lambda *a: None)
    assert all(h["status"] == "ready" for h in plan["heads"]), [h["head_id"] for h in plan["heads"] if h["status"] != "ready"]
    for h in plan["heads"]:
        m = S2.run_fit(plan_dir, s2, str(po), head_index=h["index"], inputs=inputs, log=lambda *a: None)
        assert m["status"] == "COMPLETE"
    d = s2 / "heads/virchow2/b100/fold1"
    diag = json.loads((d / "fit_diagnostics.json").read_text())
    assert diag["coefficient_shape"] == [5, 1280] and diag["n_features_in"] == 1280
    dc = json.loads((s2 / "heads/conch_v15/b5/full/fit_diagnostics.json").read_text())
    assert dc["coefficient_shape"] == [5, 768] and dc["held_out_fold"] is None
    counts = json.loads((d / "counts.json").read_text())
    folds = pd.read_csv(s0_dir / "patient_folds_tf2.csv")
    fold = dict(zip(folds.patient_id, folds.fold))
    assert not any(fold[p] == 1 for p in counts["training_patients"])
    assert (d / "probabilities.npy").exists() and (s2 / "heads/uni2/b5/full/pool_probabilities.npy").exists()
    # reloaded head predicts identically under the width wrapper
    head = pickle.loads((d / "head.pkl").read_bytes())
    assert fitter._fingerprint(head.estimator) == diag["fitted_state_sha256"]
    # refuse to refit into an existing head directory
    with pytest.raises(S0.GateError, match="non-empty"):
        S2.run_fit(plan_dir, s2, str(po), head_id="uni2/b5/fold0", inputs=inputs)
    # F7 checkpoint (ii)
    f7_out = S0.check_out_dir(s2 / "f7_checkpoint_ii", str(po), "s2")
    r = S2.run_f7([plan_dir], s2, s0_dir, f7_out, log=lambda *a: None)
    assert r["pass"] and r["heads_checked"] == 48, r["violations"][:3]
    # classifier report
    rep_out = S0.check_out_dir(s2 / "classifier_report", str(po), "s2")
    S2.run_report([plan_dir], s2, rep_out, log=lambda *a: None)
    tm = pd.read_csv(rep_out / "tile_metrics.csv")
    pooled = tm[tm.scope == "pooled_cross_fitted"]
    assert len(pooled) == 8
    labelled = pd.read_csv(s0_dir / "annotation_tiles.csv.gz")
    assert set(pooled[pooled.encoder == "uni2"].n) == {len(labelled)}  # each labelled tile once per budget
    s2b = pd.read_csv(rep_out / "s2b_full_noqc_vs_full.csv")
    assert set(s2b.scope) == {"all_pool_tiles", "qc_removed_slide_tiles", "other_tiles"}


def test_f7_ii_detects_leak(pipeline):
    po, inputs, s0_dir = pipeline
    sup = _make_s1_outputs(po, inputs, s0_dir)
    s2 = po / S0.RUN_NAME / "s2"
    plan_dir = S0.check_out_dir(s2 / "plan_003", str(po), "s2")
    plan = S2.run_dry(s0_dir, plan_dir, str(po), supplements=[sup], inputs=inputs, log=lambda *a: None)
    for h in plan["heads"]:
        if h["kind"] == "primary":
            S2.run_fit(plan_dir, s2, str(po), head_index=h["index"], inputs=inputs, log=lambda *a: None)
    # corrupt the saved training set of uni2/b100/fold0 with a fold-0 patient's tile (P1 owns query L1)
    d = s2 / "heads/uni2/b100/fold0"
    tt = pd.read_csv(d / "training_tiles.csv.gz", dtype={"slide_id": str})
    tiles = pd.read_csv(s0_dir / "annotation_tiles.csv.gz", dtype={"slide_id": str})
    extra = tiles[tiles.slide_id == "L1"].iloc[:1]
    tt = pd.concat([tt, pd.DataFrame({"position": [len(tt)], "tile_id": extra.tile_id.values,
                                      "slide_id": ["L1"], "patient_id": ["P1"],
                                      "tissue_label": extra.tissue_label.values})], ignore_index=True)
    S0.write_csv(tt, d / "training_tiles.csv.gz")
    c = json.loads((d / "COMPLETE.json").read_text())
    c["outputs"]["training_tiles.csv.gz"]["sha256"] = S0.sha256_file(d / "training_tiles.csv.gz")
    (d / "COMPLETE.json").write_text(json.dumps(c))
    f7_out = S0.check_out_dir(s2 / "f7_leak", str(po), "s2")
    r = S2.run_f7([plan_dir], s2, s0_dir, f7_out, log=lambda *a: None)
    kinds = {v["kind"] for v in r["violations"] if v["head_id"] == "uni2/b100/fold0"}
    assert not r["pass"]
    assert {"query_patient_in_saved_training", "saved_set_differs_from_plan_manifest",
            "saved_set_outside_planned_set"} <= kinds
    assert (f7_out / "f7_checkpoint_ii_FAILED.json").exists()


def test_s1_write_specs_does_not_submit(pipeline):
    po, inputs, s0_dir = pipeline
    out = S0.check_out_dir(po / S0.RUN_NAME / "s1" / "jobspec", str(po), "s1")
    spec = S1.write_specs(s0_dir, out, str(po), inputs=inputs)
    assert spec["s1a_h_optimus_1"]["slides"] == ["N1", "N2"]
    assert spec["s1c_conch_v15"]["slides"] == ["N1", "N2"]
    assert set(spec["canary_rule"]["canary_slides"]) <= {"L1", "L2", "L3", "L4", "L5"}
    assert len(spec["canary_rule"]["canary_slides"]) == 5
    for f in ("s1a_canary.lsf", "s1a_production.lsf", "s1c_canary.lsf", "s1c_production.lsf"):
        body = "\n".join(l for l in (out / f).read_text().splitlines() if not l.startswith("#"))
        assert "bsub" not in body
    assert (out / "wsi_nonlibrary" / "N1.svs").is_symlink()
    mos = pickle.load(open(out / "mosaic_h_optimus_1_nonlibrary.pkl", "rb"))
    assert mos == [("N1.svs", []), ("N2.svs", [])]


# ---------------------------------------------------------------- amendment P1: plan_002 on the excluded pool
def test_dry_run_four_encoders_mixed_roots_after_exclusion(tmp_path, monkeypatch):
    import types
    mod = types.ModuleType("openslide")
    mod.OpenSlide = T0._FakeSlide
    monkeypatch.setitem(sys.modules, "openslide", mod)
    po, inputs, expected = T0.build_fixture(tmp_path, T0._FakeSlide)
    ex = T0.write_exclusion(tmp_path, ["N2"])
    s0_dir = S0.check_out_dir(po / S0.RUN_NAME / "s0" / "run_002", str(po), "s0")
    m0 = S0.run_s0(s0_dir, inputs, T0.exclusion_expected(expected, ["N2"]), workers=1, hash_workers=2,
                   exclusion={"path": str(ex), "reason": "scan_magnification_20x"}, log=lambda *a: None)
    assert m0["status"] == "COMPLETE"
    sup = _make_s1_outputs(po, inputs, s0_dir)        # S1 roots hold N1 and N2; the supplement labels N1 only
    s2 = po / S0.RUN_NAME / "s2"
    heads_root = s2 / "run_002"
    plan_dir = S0.check_out_dir(s2 / "plan_002", str(po), "s2")
    plan = S2.run_dry(s0_dir, plan_dir, str(po), supplements=[sup], inputs=inputs, heads_root=heads_root,
                      log=lambda *a: None)
    heads = plan["heads"]
    assert len(heads) == 56 and all(h["status"] == "ready" for h in heads)
    assert {h["encoder"] for h in heads} == set(S2.ENCODERS)
    assert collections_count(h["encoder"] for h in heads) == {e: 14 for e in S2.ENCODERS}
    assert plan["heads_root"] == str(heads_root) and plan["s0"]["exclusion"]["applied"]
    # mixed bank roots: library slides from the accepted bank, the non-library slide from S1
    av = plan["feature_availability"]
    assert av["h_optimus_1"]["roots"] == [inputs["ann_h_optimus_1_dir"], str(po / S0.RUN_NAME / "s1" / "h_optimus_1_all_patches_tf2")]
    assert len(av["conch_v15"]["roots"]) == 2 and all(av[e]["present"] == 6 and not av[e]["missing"] for e in av)
    ho_roots, cr = av["h_optimus_1"]["roots"], av["conch_v15"]["roots"]
    assert S2.feature_file("h_optimus_1", "N1", ho_roots).parent == Path(ho_roots[1])
    assert S2.feature_file("h_optimus_1", "L1", ho_roots).parent == Path(ho_roots[0])
    assert S2.feature_file("conch_v15", "N1", cr).parents[1] == Path(cr[1])
    assert S2.feature_file("conch_v15", "L1", cr).parents[1] == Path(cr[0])
    # no manifest contains a tile of the excluded slide
    for mid, e in plan["manifests"].items():
        f = pd.read_csv(plan_dir / e["file"], dtype={"slide_id": str})
        assert "N2" not in set(f.slide_id), mid
    # cmds.txt: one fit command per head, in index order, pointing at the plan and heads root
    lines = (plan_dir / "cmds.txt").read_text().splitlines()
    assert len(lines) == 56 and plan["cmds"]["lines"] == 56
    for i, line in enumerate(lines, 1):
        assert f"--head-index {i} " in line and f"--plan {plan_dir} " in line
        assert line.endswith(f"--heads-root {heads_root}") and "bsub" not in line
    assert f"{heads_root}/logs" in plan["lsf"]["script"] and plan["lsf"]["ready_indices"] == "1-56"
    # the plan's heads root is enforced
    with pytest.raises(S2.S2Error, match="differs from the plan"):
        S2.run_fit(plan_dir, s2, str(po), head_id="uni2/b5/fold0", inputs=inputs)
    # one H-Optimus-1 and one CONCH head read the non-library slide from the S1 roots
    for hid in ("h_optimus_1/b100/full", "conch_v15/b100/full"):
        assert S2.run_fit(plan_dir, heads_root, str(po), head_id=hid, inputs=inputs,
                          log=lambda *a: None)["status"] == "COMPLETE"
        b = json.loads((heads_root / "heads" / hid / "input_bindings.json").read_text())
        paths = {r["slide_id"]: r["path"] for r in b["feature_reads"]}
        assert "N2" not in paths and str(po / S0.RUN_NAME / "s1") in paths["N1"]
    # a non-empty heads root is refused for a new plan
    plan_dir2 = S0.check_out_dir(s2 / "plan_003", str(po), "s2")
    with pytest.raises(S2.S2Error, match="not empty"):
        S2.run_dry(s0_dir, plan_dir2, str(po), supplements=[sup], inputs=inputs, heads_root=heads_root,
                   log=lambda *a: None)


def collections_count(it):
    import collections
    return dict(collections.Counter(it))
