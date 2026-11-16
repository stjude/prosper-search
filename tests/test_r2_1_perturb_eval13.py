"""Tests of the R2.1 P2 13-arm H-Optimus-1 perturbation evaluation adapter ([decision record] P2 specification
001+002+003). Synthetic fixtures only: first-run-layout 12-arm pickles+records, suite-layout
pickles+records (2/3/13-arm), a synthetic main bank whose identity matches, PLAN.json and a
population CSV are generated in every test."""
from __future__ import annotations

import ast
import copy
import csv
import dataclasses
import gzip
import hashlib
import importlib
import io
import json
import pickle
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prosper.evaluation import r2_1_hoptimus_main as K
from prosper.evaluation import r2_1_perturb_eval13_cli as A
from prosper.evaluation import r2_1_perturb_eval_cli as E
from prosper.evaluation import r2_1_perturb_sampler as S

DIM = A.DIM
DRAWS = 100
BLUR = A.BLUR_IMAGE
FR = A.FIRST_RUN_CONDITIONS
# >= 2 patients per group (P2-A4 analyst-gate advisory); s01/s02 share patient p1
SLIDES = [("s01", "p1", "GA"), ("s02", "p1", "GA"), ("s03", "p2", "GA"),
          ("s04", "p3", "GB"), ("s05", "p4", "GB"), ("s06", "p5", "GC"), ("s07", "p6", "GC")]
POP_COLUMNS = ["population_index", "metadata_index", "slide_id", "patient_id",
               "diagnostic_group", "feature_rows", "feature_path", "resolved_path",
               "feature_size", "feature_sha256"]
OUT_TABLES = ("estimates.csv", "paired_deltas.csv", "coverage.csv",
              "historical_topk_accuracy.csv", "sampler_index.csv", "query_metrics.csv.gz",
              "ranked_lists.csv.gz")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _adapter_sha() -> str:
    return E.sha256_file(Path(A.__file__))


# ---------------------------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------------------------

def build(root: Path, *, classes=None, fr_listed=None, s_listed=None, zero=("s03",),
          n_tiles=6, values=None, dup_rows=None, fr_rec_hook=None, s_rec_hook=None,
          plan_hook=None, fr_arrays_hook=None, s_arrays_hook=None, mb_hook=None,
          fr_coords_hook=None, s_coords_hook=None, pop_hook=None, slides=SLIDES, seed=0):
    """Write a complete synthetic input set. ``classes``: slide -> pinned|direct|absent (default
    pinned). ``fr_listed``/``s_listed``: listed non-finite reinhard tiles (NaN rows written in
    the reinhard array of that source). Hooks mutate objects before they are hashed."""
    rng = np.random.default_rng(seed)
    classes = {**{s: "pinned" for s, _, _ in slides}, **(classes or {})}
    fr_listed = {"s02": [1]} if fr_listed is None else fr_listed
    s_listed = s_listed or {}
    first_dir = root / "first" / "features" / A.ENCODER
    suite_dir = root / "suite" / "features" / A.ENCODER
    mb_dir = root / "main" / "features" / A.ENCODER
    for d in (first_dir, suite_dir, mb_dir, root / "suite" / "extension"):
        d.mkdir(parents=True, exist_ok=True)
    pins = A.Pins()
    fx = types.SimpleNamespace(root=root, first_dir=first_dir, suite_dir=suite_dir,
                               classes=classes, fr_feats={}, s_feats={}, arrs={}, rows={},
                               fr_listed=fr_listed, s_listed=s_listed, zero=tuple(zero))
    rows = []
    coords = {}
    for i, (sid, pid, grp) in enumerate(slides):
        n = n_tiles
        if values and sid in values:
            base = np.full((n, DIM), values[sid], dtype=np.float32)
        else:
            base = rng.normal(size=(n, DIM)).astype(np.float32)
        for a, b in (dup_rows or {}).get(sid, []):
            base[b] = base[a]
        arrs = {"identity": base}
        for c in A.ARMS13[1:]:
            arrs[c] = (base + rng.normal(scale=0.3, size=(n, DIM))).astype(np.float32)
        if sid in zero:
            arrs["spatial_shift"] = base.copy()
        fx.arrs[sid] = arrs
        coords[sid] = [{"wsi_loc": [224 * k, 17 * k]} for k in range(n)]
        mb_obj = (base.copy(), copy.deepcopy(coords[sid]))
        if mb_hook:
            mb_obj = mb_hook(sid, mb_obj) or mb_obj
        data = pickle.dumps(mb_obj, protocol=4)
        mb_path = mb_dir / f"{sid}.pkl"
        mb_path.write_bytes(data)
        rows.append({"population_index": i, "metadata_index": 100 + i, "slide_id": sid,
                     "patient_id": pid, "diagnostic_group": grp, "feature_rows": n,
                     "feature_path": str(mb_path), "resolved_path": str(mb_path),
                     "feature_size": len(data), "feature_sha256": _sha(data)})
    pop = pd.DataFrame(rows, columns=POP_COLUMNS)
    if pop_hook:
        new = pop_hook(pop)
        pop = pop if new is None else new
    pop_path = root / "queries.csv"
    pop.to_csv(pop_path, index=False)
    pop_sha = E.sha256_file(pop_path)
    fx.pop_path, fx.pop_sha = pop_path, pop_sha
    fx.rows = {r["slide_id"]: r for r in rows}

    def stimuli(sid, conds):
        out = []
        for c in conds:
            p = {"level": 1}
            if c == "spatial_shift":
                p = {"x_shift": 0, "y_shift": 0} if sid in zero else {"x_shift": 5, "y_shift": -3}
            out.append({"condition": c, "level": None, "parameters": p, "patch_size": 224,
                        "slide_id": sid})
        return out

    fr_info = {}
    for i, (sid, pid, grp) in enumerate(slides):
        cls = classes[sid]
        if cls == "absent":
            continue
        arrs = fx.arrs[sid]
        feats = {c: arrs[c].copy() for c in FR}
        if cls == "direct":
            feats["reinhard"] = (arrs["reinhard"] + 0.25).astype(np.float32)  # direct fit
        for t in fr_listed.get(sid, []):
            feats["reinhard"][t] = np.nan
        if fr_arrays_hook:
            fr_arrays_hook(sid, feats)
        crd = copy.deepcopy(coords[sid])
        if fr_coords_hook:
            crd = fr_coords_hook(sid, crd)
        data = pickle.dumps((feats, crd), protocol=4)
        pkl = first_dir / f"{sid}.pkl"
        pkl.write_bytes(data)
        fx.fr_feats[sid] = feats
        row = fx.rows[sid]
        direct = cls == "direct"
        rec = {"schema": A.FIRST_RUN_SCHEMA, "slide_id": sid,
               "code_sha256": pins.extractor_code_sha256,
               "manifest_sha256": pins.manifest_v1_sha256, "queries_sha256": pop_sha,
               "encoder": A.ENCODER, "dim": DIM, "n_tiles": n_tiles, "conditions": list(FR),
               "pickle_sha256": _sha(data), "pickle_bytes": len(data),
               "query_row": {k: str(v) for k, v in row.items()},
               "coords_source": row["resolved_path"], "coords_source_sha256":
                   row["feature_sha256"], "host": f"node{i:02d}",
               "gpu": "NVIDIA A100-SXM4-40GB", "lsf_job": None if direct else 9000 + i,
               "torch_backend_flags": {"main_process_num_threads": 121 if direct else 1},
               "reinhard": {"target_means": list(A.DIRECT_MEANS if direct else A.PINNED_MEANS),
                            "target_stds": list(A.PINNED_STDS)},
               "reinhard_nonfinite_tile_indices": list(fr_listed.get(sid, [])),
               "stimuli": stimuli(sid, FR)}
        if fr_rec_hook:
            fr_rec_hook(sid, rec)
        rb = json.dumps(rec, indent=1).encode()
        rp = first_dir / f"{sid}.record.json"
        rp.write_bytes(rb)
        fr_info[sid] = {"record_path": str(rp), "record_sha256": _sha(rb),
                        "pickle_sha256": _sha(data), "pickle_bytes": len(data)}

    counts = {"absent": 0, "direct": 0, "pinned": 0, "slides": len(slides)}
    entries = []
    for sid, _, _ in slides:
        cls = classes[sid]
        counts[cls] += 1
        if cls == "absent":
            entries.append({"slide_id": sid, "first_run_record": "absent",
                            "first_run_class": None, "predicate": None, "fit": None,
                            "conditions": list(A.ARMS13)})
        else:
            d = cls == "direct"
            entries.append({"slide_id": sid, "first_run_record": "present",
                            "first_run_class": cls,
                            "predicate": {"lsf_job_null": d, "means_direct": d,
                                          "threads_121": d},
                            "fit": "direct_121thread" if d else "pinned_1thread",
                            **fr_info[sid], "conditions": A.SUITE_CONDITIONS[cls]})
    plan = {"counts": dict(counts), "expected_direct_count": counts["direct"],
            "encoder": A.ENCODER, "slides": entries}
    if plan_hook:
        plan_hook(plan)
    plan_path = root / "suite" / "extension" / "PLAN.json"
    plan_path.write_bytes(json.dumps(plan, indent=1).encode())
    plan_sha = E.sha256_file(plan_path)
    fx.plan_path, fx.plan_sha = plan_path, plan_sha

    for sid, _, _ in slides:
        cls = classes[sid]
        arrs = fx.arrs[sid]
        keys = A.SUITE_CONDITIONS[cls]
        feats = {c: arrs[c].copy() for c in keys}
        if "reinhard" in feats:
            for t in s_listed.get(sid, []):
                feats["reinhard"][t] = np.nan
        if s_arrays_hook:
            s_arrays_hook(sid, feats)
        crd = copy.deepcopy(coords[sid])
        if s_coords_hook:
            crd = s_coords_hook(sid, crd)
        data = pickle.dumps((feats, crd), protocol=4)
        (suite_dir / f"{sid}.pkl").write_bytes(data)
        fx.s_feats[sid] = feats
        if cls == "absent":
            first_run = {"class": None, "status": "absent"}
        else:
            first_run = {"class": cls, "identity_equal": True,
                         "pickle_sha256": fr_info[sid]["pickle_sha256"],
                         "record_path": fr_info[sid]["record_path"],
                         "record_sha256": fr_info[sid]["record_sha256"], "status": "present"}
        rec = {"schema": A.SUITE_SCHEMA, "slide_id": sid, "encoder": A.ENCODER, "dim": DIM,
               "n_tiles": n_tiles, "conditions_encoded": list(keys),
               "pickle_sha256": _sha(data), "pickle_bytes": len(data),
               "plan_sha256": plan_sha, "manifest_v2_sha256": pins.manifest_v2_sha256,
               "manifest_v1_sha256": pins.manifest_v1_sha256, "queries_sha256": pop_sha,
               "extractor_code_sha256": pins.extractor_code_sha256,
               "suite_code_sha256": pins.suite_code_sha256,
               "fingerprint_verification": {"mismatches": 0, "status": A.FINGERPRINT_STATUS},
               "first_run": first_run,
               "first_run_record": "absent" if cls == "absent" else "present",
               "reinhard": {"target_means": list(A.PINNED_MEANS),
                            "target_stds": list(A.PINNED_STDS), "info": {}},
               "reinhard_nonfinite_tile_indices": list(s_listed.get(sid, [])),
               "stimuli": stimuli(sid, keys),
               "gpu": {"gpu_model": "NVIDIA A100 80GB PCIe", "driver": "550.127.05"}}
        if s_rec_hook:
            s_rec_hook(sid, rec)
        (suite_dir / f"{sid}.record.json").write_bytes(json.dumps(rec, indent=1).encode())

    fx.pins = dataclasses.replace(
        pins, queries_sha256=pop_sha, plan_sha256=plan_sha,
        expected_counts=(("direct", counts["direct"]), ("pinned", counts["pinned"]),
                         ("absent", counts["absent"])),
        expected_direct_count=counts["direct"], zero_shift_slides=tuple(zero))
    return fx


def run13(fx, out="run_001", *, workers=1, **kw):
    return A.run(population=fx.pop_path, population_sha256=kw.pop("population_sha256", fx.pop_sha),
                 first_root=fx.root / "first", suite_root=fx.root / "suite",
                 plan=kw.pop("plan", fx.plan_path), out_dir=fx.root / out,
                 adapter_sha256=kw.pop("adapter_sha256", _adapter_sha()), workers=workers,
                 threads=1, draw_count=DRAWS, seed=123, argv=["pytest"],
                 pins=kw.pop("pins", fx.pins))


def failures(fx, out="run_001"):
    root = fx.root / out
    assert [p.name for p in root.iterdir()] == ["VERIFICATION_FAILED.json"]
    doc = json.loads((root / "VERIFICATION_FAILED.json").read_text())
    assert doc["schema"] == A.FAILED_SCHEMA
    glob = [f["check"] for f in doc["global_failures"]]
    per = {sid: [f["check"] for f in fs] for sid, fs in doc["slide_failures"].items()}
    return glob, per, doc


def _plan_entry(fx, sid):
    return next(e for e in json.loads(fx.plan_path.read_text())["slides"]
                if e["slide_id"] == sid)


def _rows(path: Path):
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return list(csv.reader(io.StringIO(data.decode())))


def _drop_blur(path: Path) -> bytes:
    """Raw bytes with rows whose ``condition`` is gaussian_blur_image removed (order kept)."""
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    lines = data.decode().split("\n")
    col = next(csv.reader([lines[0]])).index("condition")
    kept = [lines[0]] + [ln for ln in lines[1:]
                         if not ln or next(csv.reader([ln]))[col] != BLUR]
    return "\n".join(kept).encode()


def _ordered(col):
    seen = []
    for v in col:
        if not seen or seen[-1] != v:
            assert v not in seen, f"non-contiguous arm {v}"
            seen.append(v)
    return seen


# ---------------------------------------------------------------------------------------------
# Constants / routing / predicate
# ---------------------------------------------------------------------------------------------

def test_fixed_13_arm_tuple_matches_frozen_evaluator():
    assert A.FIRST_RUN_CONDITIONS == E.CONDITIONS
    assert A.ARMS13 == E.CONDITIONS + ("gaussian_blur_image",)
    assert len(A.ARMS13) == 13 and A.ARMS13[0] == "identity"


def test_arm_sources_routing_table():
    p, d, a = A.arm_sources("pinned"), A.arm_sources("direct"), A.arm_sources("absent")
    assert list(p) == list(d) == list(a) == list(A.ARMS13)
    assert [c for c in A.ARMS13 if p[c] == "suite"] == [BLUR]
    assert [c for c in A.ARMS13 if d[c] == "suite"] == ["reinhard", BLUR]
    assert set(a.values()) == {"suite"}


def _rec(lsf, means, threads, stds=A.PINNED_STDS):
    return {"lsf_job": lsf, "torch_backend_flags": {"main_process_num_threads": threads},
            "reinhard": {"target_means": list(means), "target_stds": list(stds)}}


def test_classify_record_follows_a13():
    assert A.classify_record(_rec(None, A.DIRECT_MEANS, 121))[0] == "direct"
    assert A.classify_record(_rec(123, A.PINNED_MEANS, 1))[0] == "pinned"
    # criteria disagree
    assert A.classify_record(_rec(None, A.PINNED_MEANS, 1))[0] is None
    assert A.classify_record(_rec(123, A.DIRECT_MEANS, 121))[0] is None
    assert A.classify_record(_rec(None, A.DIRECT_MEANS, 1))[0] is None
    assert A.classify_record(_rec(123, A.PINNED_MEANS, 121))[0] is None
    # third fit / non-pinned stds
    assert A.classify_record(_rec(123, (60.0, 20.0, -10.0), 1))[0] is None
    assert A.classify_record(_rec(None, A.DIRECT_MEANS, 121, stds=(18.0, 10.0, 11.0)))[0] is None
    # missing lsf_job key is not null; threads must be the int 121
    r = _rec(None, A.DIRECT_MEANS, 121)
    del r["lsf_job"]
    assert A.classify_record(r)[0] is None
    assert A.classify_record(_rec(None, A.DIRECT_MEANS, 121.0))[0] is None
    # float32-byte comparison: the float64 neighbour of a pinned mean is not float32 exact
    near = list(A.PINNED_MEANS)
    near[0] = near[0] + 1e-12
    assert A.classify_record(_rec(123, near, 1))[0] is None


# ---------------------------------------------------------------------------------------------
# P2-A4 bit equality with E on an all-pinned fixture
# ---------------------------------------------------------------------------------------------

def _assert_a4_byte_equal(tmp_path, fx):
    """Run the adapter (normal 13-arm mode) and E on the first-run tree; assert every P2-A4-named
    file is byte-identical (tables after removing gaussian_blur_image rows, gz decompressed)."""
    assert run13(fx) == 0
    a_out = tmp_path / "run_001"
    e_out = tmp_path / "e_run"
    recE = E.run_encoder("h_optimus_1", pop_full=E.load_population(fx.pop_path),
                         population_path=fx.pop_path, features_root=tmp_path / "first" / "features",
                         out_dir=e_out, conditions=E.CONDITIONS, subset=None, allow_partial=False,
                         cls_only=False, workers=1, threads=1, draw_count=DRAWS, seed=123,
                         argv=[])
    for c in E.CONDITIONS:
        for d in ("sampler", "distances"):
            assert (a_out / d / f"{c}.npy").read_bytes() == (e_out / d / f"{c}.npy").read_bytes(), (d, c)
    assert (a_out / "patient_draws.npy").read_bytes() == (e_out / "patient_draws.npy").read_bytes()
    for name in OUT_TABLES:
        ea = _drop_blur(a_out / name)
        ee = (gzip.decompress((e_out / name).read_bytes()) if name.endswith(".gz")
              else (e_out / name).read_bytes())
        assert ea == ee, name
        # the adapter rows really include the 13th arm before removal
        rows = _rows(a_out / name)
        col = rows[0].index("condition")
        assert BLUR in {r[col] for r in rows[1:]}, name
    return a_out, recE


def test_bit_equality_with_frozen_evaluator_all_pinned(tmp_path):
    fx = build(tmp_path, fr_listed={"s02": [1], "s06": [0, 4]})
    a_out, recE = _assert_a4_byte_equal(tmp_path, fx)
    # the listed non-finite reinhard tiles were really excluded in both
    idx = pd.read_csv(a_out / "sampler_index.csv", dtype=str)
    rr = idx[(idx.condition == "reinhard") & (idx.slide_id == "s06")].iloc[0]
    assert rr.n_excluded_listed == "2" and rr.n_used == "4"
    # E policy strings copied verbatim; design decisions and manifest_v2 bound
    rec13 = json.loads((a_out / "run_record.json").read_text())
    for k, v in recE["policy"].items():
        assert rec13["policy"][k] == v, k
    assert set(rec13["policy"]["author_decisions"]) == {"A", "B", "C", "D"}
    assert rec13["manifest_v2_sha256"] == fx.pins.manifest_v2_sha256
    assert recE["draw_tables"] == rec13["draw_tables"]


# 24 slides / 2 groups / 23 patients (s01, s02 share p1): enough patients per group that the
# bootstrap CI path (np.quantile) produces estimable cells (review 001 L5)
SLIDES_CI = [(f"s{i:02d}", f"p{1 if i == 2 else i}", "GA" if i <= 12 else "GB")
             for i in range(1, 25)]


def test_bit_equality_with_frozen_evaluator_estimable_cis(tmp_path):
    """P2-A4 second configuration (review 001 L5): estimable CIs are byte-compared too."""
    fx = build(tmp_path, slides=SLIDES_CI, fr_listed={"s02": [1], "s14": [0, 4]}, zero=("s03",))
    a_out, _ = _assert_a4_byte_equal(tmp_path, fx)
    est = pd.read_csv(a_out / "estimates.csv", dtype=str, keep_default_na=False)
    dl = pd.read_csv(a_out / "paired_deltas.csv", dtype=str, keep_default_na=False)
    assert (est.ci_status == "estimable").any()
    assert (dl.ci_status == "estimable").any()
    # the estimable cells carry numeric CI bounds that were part of the byte comparison
    e1 = est[est.ci_status == "estimable"].iloc[0]
    assert e1.ci_low != "" and e1.ci_high != ""
    assert float(e1.ci_low) <= float(e1.ci_high)


# ---------------------------------------------------------------------------------------------
# A3 / A11(i): 13 names everywhere; A10/A12 code_frozen first; run_record
# ---------------------------------------------------------------------------------------------

EXPECTED_RECORD_KEYS = [
    "schema", "created", "host", "lsf_job", "argv", "encoder", "feature_dim_used", "conditions",
    "arm_labels", "population", "inputs", "manifest_v2_sha256", "input_manifest", "code_sha256",
    "code_frozen", "versions", "policy", "routing", "per_arm_source", "null_stimuli",
    "reinhard_excluded_cells", "absent_branch_count", "identity_equals_main_bank",
    "first_run_provenance", "suite_gpu_provenance", "coverage_summary", "disclosures",
    "draw_tables", "diagnostics", "timings_seconds", "workers", "threads",
    "peak_rss_mb_main_process", "outputs"]


def test_every_output_has_exactly_the_13_arms_in_order(tmp_path):
    fx = build(tmp_path, classes={"s04": "direct", "s07": "absent"}, s_listed={"s04": [2]})
    assert run13(fx) == 0
    out = tmp_path / "run_001"
    want = list(A.ARMS13)
    for name in OUT_TABLES:
        rows = _rows(out / name)
        col = rows[0].index("condition")
        got = _ordered([r[col] for r in rows[1:]])
        assert got == (want[1:] if name == "paired_deltas.csv" else want), name
    for d in ("sampler", "distances"):
        assert sorted(p.stem for p in (out / d).iterdir()) == sorted(want)
    rec = json.loads((out / "run_record.json").read_text())
    assert rec["conditions"] == want
    assert [k[len("sampler/"):-4] for k in rec["outputs"] if k.startswith("sampler/")] == want
    assert [k[len("distances/"):-4] for k in rec["outputs"] if k.startswith("distances/")] == want
    man = pd.read_csv(out / "input_manifest.csv", dtype=str)
    assert list(man.columns) == list(A.MANIFEST_HEADER)
    assert list(rec["per_arm_source"]) == want


def test_assert_arms_rejects_missing_extra_or_reordered():
    want = list(A.ARMS13)
    files = [f"sampler/{c}.npy" for c in want] + [f"distances/{c}.npy" for c in want]
    good = {"estimates.csv": want, "paired_deltas.csv": want[1:]}
    A.assert_arms(good, files, want)
    with pytest.raises(E.EvalError):
        A.assert_arms({"estimates.csv": want[:-1]}, files, want)
    with pytest.raises(E.EvalError):
        A.assert_arms({"estimates.csv": [want[1], want[0], *want[2:]]}, files, want)
    with pytest.raises(E.EvalError):
        A.assert_arms({"estimates.csv": want + ["extra"]}, files, want)
    with pytest.raises(E.EvalError):
        A.assert_arms(good, files[:-1], want)
    with pytest.raises(E.EvalError):
        A.assert_arms(good, files, want[:12])
    with pytest.raises(E.EvalError):   # non-contiguous
        A.assert_arms({"estimates.csv": want + [want[0]]}, files, want)


def test_code_frozen_first_puts_and_run_record(tmp_path):
    fx = build(tmp_path)
    assert run13(fx) == 0
    out = tmp_path / "run_001"
    rec = json.loads((out / "run_record.json").read_text())
    assert list(rec) == EXPECTED_RECORD_KEYS
    assert rec["schema"] == "r2_1_perturb_eval13_run_v1"
    outs = list(rec["outputs"])
    live = A._live_code()
    names = [f"code_frozen/{live[k].name}" for k in ("adapter", "E", "S", "kernel")]
    assert outs[:4] == names
    pinned = [_adapter_sha(), A.E_SHA256, A.S_SHA256, A.KERNEL_SHA256]
    for nm, p, sha in zip(names, (live[k] for k in ("adapter", "E", "S", "kernel")), pinned):
        assert (out / nm).read_bytes() == p.read_bytes()
        assert rec["code_frozen"][nm] == sha == rec["code_sha256"][str(p)]
    assert set(rec["code_sha256"]) == {str(p) for p in live.values()}
    complete = json.loads((out / "COMPLETE").read_text())
    assert set(complete["files"]) == set(rec["outputs"]) | {"run_record.json"}
    for nm, meta in complete["files"].items():
        assert _sha((out / nm).read_bytes()) == meta["sha256"]
    # labels (P2.2), disclosures (V11, decision A), no claim, HIST label (P2-A8)
    assert rec["arm_labels"]["gaussian_blur"].startswith("post-preprocessing")
    assert rec["arm_labels"][BLUR].startswith("source-pixel blur")
    assert rec["disclosures"]["V11"] == A.V11_DISCLOSURE
    assert "established, not proved" in A.V11_DISCLOSURE
    assert "no float16" in rec["disclosures"]["decision_A"].lower()
    assert rec["policy"]["historical_secondary"] == E.HIST_LABEL
    hist = pd.read_csv(out / "historical_topk_accuracy.csv", dtype=str)
    assert (hist.label == E.HIST_LABEL).all()
    est = pd.read_csv(out / "estimates.csv", dtype=str)
    for c in ("point_status", "ci_status", "zero_support_replicates"):
        assert c in est.columns
    assert rec["identity_equals_main_bank"]["byte_equal"] == len(SLIDES)
    assert rec["absent_branch_count"] == 0
    assert rec["coverage_summary"]["non_returns"] == {c: 0 for c in A.ARMS13}


def test_refuses_nonempty_output_directory_before_reading_inputs(tmp_path):
    fx = build(tmp_path)
    (tmp_path / "run_001").mkdir()
    (tmp_path / "run_001" / "x").write_text("x")
    with pytest.raises(E.EvalError):
        run13(fx)


# ---------------------------------------------------------------------------------------------
# A11(ii): E/S/kernel module attributes unchanged after import and after a run
# ---------------------------------------------------------------------------------------------

_SIMPLE = (tuple, list, dict, set, frozenset, str, bytes, int, float, bool, type(None))


def _snapshot(mod):
    snap = {}
    for k, v in vars(mod).items():
        val = None
        if isinstance(v, np.ndarray):
            val = (v.dtype.str, v.shape, v.tobytes())
        elif isinstance(v, _SIMPLE) and not k.startswith("__"):
            val = repr(v)
        snap[k] = (v, val)
    return snap


def _assert_same(mod, snap):
    now = vars(mod)
    assert set(now) == set(snap), (mod.__name__, set(now) ^ set(snap))
    for k, (obj, val) in snap.items():
        assert now[k] is obj, (mod.__name__, k)
        if val is not None:
            cur = now[k]
            cur_val = ((cur.dtype.str, cur.shape, cur.tobytes()) if isinstance(cur, np.ndarray)
                       else repr(cur))
            assert cur_val == val, (mod.__name__, k)


def test_frozen_module_attributes_unchanged_after_import_and_run(tmp_path):
    snaps = {m: _snapshot(m) for m in (E, S, K)}
    conds = E.CONDITIONS
    mod = importlib.reload(A)
    for m, s in snaps.items():
        _assert_same(m, s)
    fx = build(tmp_path, classes={"s04": "direct", "s07": "absent"}, s_listed={"s07": [3]})
    assert mod.run(population=fx.pop_path, population_sha256=fx.pop_sha,
                   first_root=tmp_path / "first", suite_root=tmp_path / "suite",
                   plan=fx.plan_path, out_dir=tmp_path / "run_001", adapter_sha256=_adapter_sha(),
                   workers=1, threads=1, draw_count=DRAWS, seed=123, argv=[],
                   pins=fx.pins) == 0
    # a failing run must not mutate anything either
    assert mod.run(population=fx.pop_path, population_sha256="0" * 64,
                   first_root=tmp_path / "first", suite_root=tmp_path / "suite",
                   plan=fx.plan_path, out_dir=tmp_path / "run_002", adapter_sha256=_adapter_sha(),
                   workers=1, threads=1, draw_count=DRAWS, seed=123, argv=[],
                   pins=fx.pins) == 2
    for m, s in snaps.items():
        _assert_same(m, s)
    assert E.CONDITIONS is conds and len(E.CONDITIONS) == 12
    assert E.ENCODER_DIM == {"h_optimus_1": 1536, "uni2": 1536, "virchow2": 2560}


def test_frozen_module_attributes_unchanged_by_fresh_import_subprocess():
    code = (
        "import numpy as np\n"
        "from prosper.evaluation import r2_1_perturb_eval_cli as E, r2_1_perturb_sampler as S, "
        "r2_1_hoptimus_main as K\n"
        "def snap(m):\n"
        "    return {k: (id(v), repr(v) if isinstance(v, (tuple, list, dict, str, int, float))"
        " and not k.startswith('__') else None) for k, v in vars(m).items()}\n"
        "before = {m.__name__: snap(m) for m in (E, S, K)}\n"
        "import prosper.evaluation.r2_1_perturb_eval13_cli as A\n"
        "after = {m.__name__: snap(m) for m in (E, S, K)}\n"
        "assert before == after, 'module attributes changed'\n"
        "print('OK', len(A.ARMS13))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "OK 13"


def test_adapter_uses_only_whitelisted_e_names():
    tree = ast.parse(Path(A.__file__).read_text())
    used = {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "E"}
    blacklist = {"check_inputs", "_slide_task", "compute_samplers", "_compute_samplers",
                 "evaluate", "run_encoder", "code_hashes", "CONDITIONS"}
    blacklist |= {n for n in vars(E) if n.startswith("cmd_")}
    assert not used & blacklist
    whitelist = {"rank_condition", "DrawCache", "query_vectors", "_est_cells", "_status_cells",
                 "historical_table", "coverage_rows", "Writer", "_csv_bytes", "_npy_bytes",
                 "load_population", "sha256_file", "EvalError", "peak_rss_mb", "default_out",
                 "__file__", "CUTOFFS", "ENDPOINTS", "MACRO", "HIST_LABEL", "ESTIMATE_HEADER",
                 "DELTA_HEADER", "COVERAGE_HEADER", "HIST_HEADER", "SAMPLER_INDEX_HEADER",
                 "QUERY_HEADER", "RANK_HEADER"}
    assert used <= whitelist, used - whitelist


# ---------------------------------------------------------------------------------------------
# A11(iii): no fp16 cast (design decision A)
# ---------------------------------------------------------------------------------------------

def test_sampler_has_no_float16_cast(tmp_path):
    v = 1 + 2 ** -20
    assert np.float16(v) == 1.0 and np.float32(v) != 1.0
    fx = build(tmp_path, values={"s05": v})
    assert run13(fx) == 0
    emb = np.load(tmp_path / "run_001" / "sampler" / "identity.npy")
    row = emb[4]
    assert row.dtype == np.float32
    assert np.all(row == np.float32(v))
    for c in A.ARMS13:
        got = np.load(tmp_path / "run_001" / "sampler" / f"{c}.npy")
        src = fx.s_feats if A.arm_sources("pinned")[c] == "suite" else fx.fr_feats
        for i, (sid, _, _) in enumerate(SLIDES):
            r = S.sampler(src[sid][c], fx.fr_listed.get(sid) if c == "reinhard" else None)
            assert got[i].tobytes() == r.embedding.astype(np.float32).tobytes(), (c, sid)


# ---------------------------------------------------------------------------------------------
# Routing (P2.3(4), P2-A2): pinned / direct / absent
# ---------------------------------------------------------------------------------------------

def test_routing_pinned_direct_absent_and_suite_listed_indices(tmp_path):
    # A11(vi): direct slide s04 has deliberately different first-run and suite lists
    fx = build(tmp_path, classes={"s04": "direct", "s05": "direct", "s07": "absent"},
               fr_listed={"s02": [1], "s04": [0]}, s_listed={"s04": [2], "s07": [5]})
    assert run13(fx) == 0
    out = tmp_path / "run_001"
    man = pd.read_csv(out / "input_manifest.csv", dtype=str, keep_default_na=False)
    by = man.set_index("slide_id")
    for sid, cls in fx.classes.items():
        src = A.arm_sources(cls)
        assert by.loc[sid, "class"] == cls
        assert [by.loc[sid, f"source_{c}"] for c in A.ARMS13] == [src[c] for c in A.ARMS13]
    assert by.loc["s07", "first_run_pickle"] == "" and by.loc["s07", "first_run_record"] == ""
    assert by.loc["s04", "reinhard_nonfinite_source"] == "suite"
    assert by.loc["s04", "reinhard_nonfinite_listed"] == "[2]"
    assert by.loc["s02", "reinhard_nonfinite_source"] == "first_run"
    assert by.loc["s07", "reinhard_nonfinite_listed"] == "[5]"
    for c in A.ARMS13:
        got = np.load(out / "sampler" / f"{c}.npy")
        for i, (sid, _, _) in enumerate(SLIDES):
            src = A.arm_sources(fx.classes[sid])[c]
            feats = fx.s_feats[sid] if src == "suite" else fx.fr_feats[sid]
            lst = (fx.s_listed if A.arm_sources(fx.classes[sid])["reinhard"] == "suite"
                   else fx.fr_listed).get(sid) if c == "reinhard" else None
            exp = S.sampler(feats[c], lst).embedding.astype(np.float32)
            assert got[i].tobytes() == exp.tobytes(), (c, sid)
    # the direct-fit first-run reinhard array is not used
    direct_fit = S.sampler(fx.fr_feats["s04"]["reinhard"], [0]).embedding.astype(np.float32)
    assert np.load(out / "sampler" / "reinhard.npy")[3].tobytes() != direct_fit.tobytes()
    idx = pd.read_csv(out / "sampler_index.csv", dtype=str)
    r = idx[(idx.condition == "reinhard") & (idx.slide_id == "s04")].iloc[0]
    assert (r.n_excluded_listed, r.n_used) == ("1", "5")
    rec = json.loads((out / "run_record.json").read_text())
    assert rec["routing"]["class_counts"] == {"direct": 2, "pinned": 4, "absent": 1}
    assert rec["absent_branch_count"] == 1
    assert rec["per_arm_source"]["reinhard"] == {"first_run": 4, "suite": 3}
    assert rec["per_arm_source"][BLUR] == {"first_run": 0, "suite": 7}
    assert rec["per_arm_source"]["identity"] == {"first_run": 6, "suite": 1}
    cells = {(c["slide_id"], c["tile_index"], c["source"]) for c in rec["reinhard_excluded_cells"]}
    assert cells == {("s02", 1, "first_run"), ("s04", 2, "suite"), ("s07", 5, "suite")}
    assert rec["first_run_provenance"]["summary"] == {
        "NVIDIA A100-SXM4-40GB | lsf_job null": 2, "NVIDIA A100-SXM4-40GB | lsf_job set": 4}
    assert rec["suite_gpu_provenance"] == {"NVIDIA A100 80GB PCIe | driver 550.127.05": 7}
    assert "first_run_host" in man.columns and by.loc["s01", "first_run_lsf_job"] == "9000"


def test_listed_duplicate_identity_rows_are_noted(tmp_path):
    fx = build(tmp_path, fr_listed={"s02": [1, 3]}, dup_rows={"s02": [(1, 3)]})
    assert run13(fx) == 0
    rec = json.loads((tmp_path / "run_001" / "run_record.json").read_text())
    cells = {c["tile_index"]: c["identity_row_byte_equal_to_listed"]
             for c in rec["reinhard_excluded_cells"] if c["slide_id"] == "s02"}
    assert cells == {1: [3], 3: [1]}


def test_workers_spawn_pool_matches_serial(tmp_path):
    fx = build(tmp_path, classes={"s04": "direct", "s07": "absent"})
    assert run13(fx, "run_001") == 0
    assert run13(fx, "run_002", workers=2) == 0
    a, b = tmp_path / "run_001", tmp_path / "run_002"
    for p in sorted(a.rglob("*")):
        if p.is_file() and p.name not in ("run_record.json", "COMPLETE"):
            assert p.read_bytes() == (b / p.relative_to(a)).read_bytes(), p


# ---------------------------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------------------------

def _set(path, value):
    def hook(sid, rec, path=path, value=value):
        if sid != "s03":
            return
        d = rec
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return hook


@pytest.mark.parametrize("path,value,check", [
    (("schema",), "r2_1_perturb_record_v0", "first_run.schema"),
    (("code_sha256",), "1" * 64, "first_run.code_sha256"),
    (("manifest_sha256",), "2" * 64, "first_run.manifest_sha256"),
    (("queries_sha256",), "3" * 64, "first_run.queries_sha256"),
    (("query_row", "resolved_path"), "/elsewhere.pkl", "first_run.query_row.resolved_path"),
    (("query_row", "feature_sha256"), "4" * 64, "first_run.query_row.feature_sha256"),
    (("query_row", "population_index"), "7", "first_run.query_row.population_index"),
    (("encoder",), "uni2", "first_run.encoder"),
    (("dim",), 1280, "first_run.dim"),
    (("n_tiles",), 5, "first_run.n_tiles"),
    (("conditions",), list(FR[:-1]), "first_run.conditions"),
    (("coords_source",), "/elsewhere.pkl", "first_run.coords_source"),
    (("coords_source_sha256",), "5" * 64, "first_run.coords_source_sha256"),
    (("pickle_sha256",), "6" * 64, "first_run.pickle_sha256"),
    (("pickle_bytes",), 1, "first_run.pickle_bytes"),
    (("reinhard_nonfinite_tile_indices",), [99], "first_run.reinhard_nonfinite_tile_indices"),
])
def test_first_run_record_binding_refusals(tmp_path, path, value, check):
    fx = build(tmp_path, fr_rec_hook=_set(path, value))
    assert run13(fx) == 2
    glob, per, _ = failures(fx)
    assert check in per["s03"], per
    assert set(per) == {"s03"}


@pytest.mark.parametrize("path,value,check", [
    (("schema",), "x", "suite.schema"),
    (("slide_id",), "s99", "suite.slide_id"),
    (("encoder",), "uni2", "suite.encoder"),
    (("dim",), 1280, "suite.dim"),
    (("n_tiles",), 7, "suite.n_tiles"),
    (("queries_sha256",), "1" * 64, "suite.queries_sha256"),
    (("plan_sha256",), "2" * 64, "suite.plan_sha256"),
    (("manifest_v2_sha256",), "3" * 64, "suite.manifest_v2_sha256"),
    (("manifest_v1_sha256",), "4" * 64, "suite.manifest_v1_sha256"),
    (("extractor_code_sha256",), "5" * 64, "suite.extractor_code_sha256"),
    (("suite_code_sha256",), "6" * 64, "suite.suite_code_sha256"),
    (("fingerprint_verification", "mismatches"), 1, "suite.fingerprint_verification.mismatches"),
    (("fingerprint_verification", "status"), "partial", "suite.fingerprint_verification.status"),
    (("first_run_record",), "absent", "suite.first_run_record"),
    (("first_run", "status"), "absent", "suite.first_run.status"),
    (("first_run", "record_path"), "/elsewhere/s03.record.json", "suite.first_run.record_path"),
    (("first_run", "record_sha256"), "7" * 64, "suite.first_run.record_sha256"),
    (("first_run", "pickle_sha256"), "8" * 64, "suite.first_run.pickle_sha256"),
    (("first_run", "class"), "direct", "suite.first_run.class"),
    (("conditions_encoded",), ["identity", "reinhard", BLUR], "suite.conditions_encoded"),
    (("reinhard", "target_means"), list(A.DIRECT_MEANS), "suite.reinhard_fit"),
    (("reinhard", "target_stds"), [18.0, 10.0, 11.0], "suite.reinhard_fit"),
    (("pickle_sha256",), "9" * 64, "suite.pickle_sha256"),
    (("pickle_bytes",), 3, "suite.pickle_bytes"),
])
def test_suite_record_refusals(tmp_path, path, value, check):
    fx = build(tmp_path, s_rec_hook=_set(path, value))
    assert run13(fx) == 2
    glob, per, _ = failures(fx)
    assert check in per["s03"], per
    assert set(per) == {"s03"}


def test_suite_record_path_accepts_realpath_alias(tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)

    def hook(sid, rec):
        if rec.get("first_run", {}).get("record_path"):
            rec["first_run"]["record_path"] = rec["first_run"]["record_path"].replace(
                str(tmp_path), str(alias), 1)
    fx = build(tmp_path, s_rec_hook=hook)
    assert run13(fx) == 0


def test_suite_pickle_key_set_and_absent_branch_checks(tmp_path):
    def s_arr(sid, feats):
        if sid == "s03":
            feats["reinhard"] = feats["identity"].copy()
    fx = build(tmp_path, s_arrays_hook=s_arr)
    assert run13(fx) == 2
    _, per, _ = failures(fx)
    assert per == {"s03": ["suite.pickle_keys"]}


def test_absent_branch_requires_suite_absent_status(tmp_path):
    fx = build(tmp_path, classes={"s07": "absent"}, s_rec_hook=lambda sid, rec: rec.update(
        first_run_record="present") if sid == "s07" else None)
    assert run13(fx) == 2
    _, per, _ = failures(fx)
    assert per == {"s07": ["suite.first_run_record"]}


def test_absent_branch_requires_13_arm_suite_pickle(tmp_path):
    def hook(sid, rec):
        if sid == "s07":
            rec["conditions_encoded"] = ["identity", BLUR]
    fx = build(tmp_path, classes={"s07": "absent"}, s_rec_hook=hook)
    assert run13(fx) == 2
    _, per, _ = failures(fx)
    assert "suite.conditions_encoded" in per["s07"]


@pytest.mark.parametrize("where", ["first_run", "suite"])
def test_array_dtype_and_shape_refusals(tmp_path, where):
    def hook(sid, feats):
        if sid == "s03":
            feats[BLUR if where == "suite" else "brightness"] = feats["identity"].astype(np.float64)
            if where == "first_run":
                feats["contrast"] = feats["contrast"][:-1]
    fx = build(tmp_path, **{("s_arrays_hook" if where == "suite" else "fr_arrays_hook"): hook})
    assert run13(fx) == 2
    _, per, _ = failures(fx)
    if where == "suite":
        assert per == {"s03": [f"suite.array[{BLUR}]"]}
    else:
        assert per == {"s03": ["first_run.array[brightness]", "first_run.array[contrast]"]}


def test_tampered_pickle_bytes_refused(tmp_path):
    fx = build(tmp_path)
    p = fx.first_dir / "s05.pkl"
    data = bytearray(p.read_bytes())
    data[-20] ^= 1
    p.write_bytes(bytes(data))
    assert run13(fx) == 2
    _, per, _ = failures(fx)
    assert per["s05"][0] == "first_run.pickle_sha256"


def test_predicate_disagreement_is_a_verification_failure(tmp_path):
    def hook(sid, rec):
        if sid == "s03":
            rec["lsf_job"] = None        # lsf null but pinned means/1 thread
    fx = build(tmp_path, fr_rec_hook=hook)
    assert run13(fx) == 2
    glob, per, _ = failures(fx)
    assert "class.predicate" in per["s03"]
    assert "class.recomputed_counts" in glob


def test_class_must_match_plan_entry(tmp_path):
    def hook(plan):
        for e in plan["slides"]:
            if e["slide_id"] == "s03":
                e["first_run_class"] = "direct"
            if e["slide_id"] == "s05":
                e["predicate"]["threads_121"] = True
    fx = build(tmp_path, plan_hook=hook)
    assert run13(fx) == 2
    _, per, _ = failures(fx)
    assert per == {"s03": ["class.plan_first_run_class"], "s05": ["class.plan_predicate"]}


def test_identity_differs_from_main_bank_reports_rows_delta_dtype_shape(tmp_path):
    def mb(sid, obj):
        if sid == "s06":
            f = obj[0].copy()
            f[2, 7] += 0.5
            return (f, obj[1])
    fx = build(tmp_path, mb_hook=mb)
    assert run13(fx) == 2
    _, per, doc = failures(fx)
    # the (equal) suite identity of this present-branch slide differs from the main bank too
    assert per == {"s06": ["identity_equals_main_bank", "suite_identity_equals_main_bank"]}
    for f in doc["slide_failures"]["s06"]:
        assert f["differing_rows"] == 1 and f["max_abs_delta"] == pytest.approx(0.5)
        assert f["dtype"] == ["float32", "float32"] and f["shape"] == [[6, DIM], [6, DIM]]
        assert f["array_equal"] is False


@pytest.mark.parametrize("sid,cls", [("s05", "pinned"), ("s04", "direct")])
def test_suite_identity_must_equal_main_bank_on_present_branch(tmp_path, sid, cls):
    """Review 001 L4: on present-branch slides the suite identity (paired partner of
    gaussian_blur_image) must be byte-identical to the main bank (hence the first-run identity),
    even though the first-run identity is the arm used for the 12 non-blur conditions."""
    def hook(s, feats):
        if s == sid:
            feats["identity"][2, 7] += 0.5
    fx = build(tmp_path, classes={"s04": "direct"}, s_arrays_hook=hook)
    assert run13(fx) == 2
    glob, per, doc = failures(fx)
    assert glob == [] and per == {sid: ["suite_identity_equals_main_bank"]}
    f = doc["slide_failures"][sid][0]
    assert (f["expected"], f["observed"]) == ("byte-identical", "differs")
    assert f["differing_rows"] == 1 and f["max_abs_delta"] == pytest.approx(0.5)
    assert f["dtype"] == ["float32", "float32"] and f["shape"] == [[6, DIM], [6, DIM]]
    assert f["array_equal"] is False
    assert not (tmp_path / "run_001" / "COMPLETE").exists()


def test_suite_identity_equal_to_main_bank_passes_and_absent_branch_not_double_counted(tmp_path):
    """Review 001 L4 positive: equal suite identities pass on pinned/direct present-branch
    slides; on the absent branch the suite identity IS the used identity, so a drift there is
    reported once, as identity_equals_main_bank."""
    fx = build(tmp_path, classes={"s04": "direct", "s07": "absent"})
    assert run13(fx) == 0
    assert (tmp_path / "run_001" / "COMPLETE").exists()
    for sid in ("s01", "s04"):
        res = A._verify_slide((sid, {c: str(v) for c, v in fx.rows[sid].items()},
                               _plan_entry(fx, sid), str(fx.first_dir), str(fx.suite_dir),
                               fx.pins))
        assert res[1] == [] and res[3]["suite_identity_equals_main_bank"] is True
    def hook(s, feats):
        if s == "s07":
            feats["identity"][0, 0] += 1.0
    fx2 = build(tmp_path / "b", classes={"s04": "direct", "s07": "absent"}, s_arrays_hook=hook)
    assert run13(fx2) == 2
    _, per, _ = failures(fx2)
    assert per == {"s07": ["identity_equals_main_bank"]}


def test_main_bank_float64_with_equal_values_fails(tmp_path):
    """A11(iv)."""
    fx = build(tmp_path, mb_hook=lambda sid, obj: (obj[0].astype(np.float64), obj[1])
               if sid == "s01" else None)
    assert run13(fx) == 2
    _, per, _ = failures(fx)
    assert per == {"s01": ["main_bank.array"]}


def test_main_bank_size_and_sha_bound_to_population(tmp_path):
    def pop_hook(pop):
        pop.loc[pop.slide_id == "s02", "feature_size"] = 1
        pop.loc[pop.slide_id == "s04", "feature_sha256"] = "0" * 64
        return pop
    fx = build(tmp_path, pop_hook=pop_hook)
    assert run13(fx) == 2
    _, per, _ = failures(fx)
    # query_row/coords_source bindings in the first-run record also differ from the edited row
    assert "main_bank.st_size" in per["s02"]
    assert "main_bank.sha256" in per["s04"]


@pytest.mark.parametrize("which", ["first_run", "suite", "suite_type", "main_bank_type"])
def test_coords_equality_and_types(tmp_path, which):
    def change(sid, crd):
        if sid != "s05":
            return crd
        crd = copy.deepcopy(crd)
        if which == "suite_type":
            crd[0]["wsi_loc"] = tuple(crd[0]["wsi_loc"])
        else:
            crd[1]["wsi_loc"][0] += 1
        return crd
    kw = {}
    if which == "first_run":
        kw["fr_coords_hook"] = change
    elif which in ("suite", "suite_type"):
        kw["s_coords_hook"] = change
    else:
        kw["mb_hook"] = lambda sid, obj: (obj[0], tuple(obj[1])) if sid == "s05" else None
    fx = build(tmp_path, **kw)
    assert run13(fx) == 2
    _, per, _ = failures(fx)
    exp = {"first_run": ["coords.first_run_equals_main_bank"],
           "suite": ["coords.suite_equals_main_bank"],
           "suite_type": ["coords.suite_equals_main_bank"],
           "main_bank_type": ["coords.main_bank_type", "coords.first_run_equals_main_bank",
                              "coords.suite_equals_main_bank"]}[which]
    assert per == {"s05": exp}


# ---------------------------------------------------------------------------------------------
# A11(vii) zero-shift; A11(viii) PLAN counts; A11(ix) all failures listed
# ---------------------------------------------------------------------------------------------

def test_zero_shift_null_stimuli_flagged_and_asserted(tmp_path):
    fx = build(tmp_path, zero=("s03", "s06"), classes={"s06": "direct"})
    assert run13(fx) == 0
    out = tmp_path / "run_001"
    rec = json.loads((out / "run_record.json").read_text())
    assert rec["null_stimuli"]["slides"] == ["s03", "s06"]
    assert rec["null_stimuli"]["n"] == 2 and rec["null_stimuli"]["tiles"] == 12
    assert rec["null_stimuli"]["spatial_shift_bit_equal_identity"] is True
    man = pd.read_csv(out / "input_manifest.csv", dtype=str).set_index("slide_id")
    assert man.null_stimulus_spatial_shift.to_dict() == {
        s: ("1" if s in ("s03", "s06") else "0") for s, _, _ in SLIDES}
    ss = np.load(out / "sampler" / "spatial_shift.npy")
    ii = np.load(out / "sampler" / "identity.npy")
    assert ss[2].tobytes() == ii[2].tobytes()


def test_zero_shift_slide_whose_spatial_shift_differs_stops(tmp_path):
    def hook(sid, feats):
        if sid == "s03":
            feats["spatial_shift"][0, 0] += 1e-3
    fx = build(tmp_path, fr_arrays_hook=hook)
    assert run13(fx) == 2
    _, per, _ = failures(fx)
    assert per == {"s03": ["zero_shift.spatial_shift_equals_identity"]}


def test_zero_shift_set_must_equal_pinned_set(tmp_path):
    fx = build(tmp_path)
    pins = dataclasses.replace(fx.pins, zero_shift_slides=("s03", "s04"))
    assert run13(fx, pins=pins) == 2
    glob, per, _ = failures(fx)
    assert glob == ["zero_shift.slide_set"] and per == {}


def test_zero_shift_absent_branch_reads_suite_stimuli(tmp_path):
    fx = build(tmp_path, zero=("s03", "s07"), classes={"s07": "absent"})
    assert run13(fx) == 0
    rec = json.loads((tmp_path / "run_001" / "run_record.json").read_text())
    assert rec["null_stimuli"]["slides"] == ["s03", "s07"]


def test_spatial_shift_stimulus_must_be_unique(tmp_path):
    def hook(sid, rec):
        if sid == "s04":
            rec["stimuli"].append(dict(rec["stimuli"][9]))
    fx = build(tmp_path, fr_rec_hook=hook)
    assert run13(fx) == 2
    _, per, _ = failures(fx)
    assert per == {"s04": ["first_run.stimuli.spatial_shift_count"]}


@pytest.mark.parametrize("mutation,check", [
    (lambda p: p["counts"].update(direct=1), "plan.counts"),
    (lambda p: p.update(expected_direct_count=654), "plan.expected_direct_count"),
])
def test_plan_count_mismatch_stops(tmp_path, mutation, check):
    fx = build(tmp_path, plan_hook=mutation)
    assert run13(fx) == 2
    glob, per, _ = failures(fx)
    assert check in glob and per == {}


def test_production_pins_require_655_direct(tmp_path):
    """A11(viii) against the production pins: a PLAN with expected_direct_count != 655 stops."""
    fx = build(tmp_path)
    pins = dataclasses.replace(A.Pins(), queries_sha256=fx.pop_sha, plan_sha256=fx.plan_sha,
                               zero_shift_slides=("s03",))
    assert A.Pins().expected_direct_count == 655
    assert dict(A.Pins().expected_counts) == {"direct": 655, "pinned": 2123, "absent": 0}
    assert run13(fx, pins=pins) == 2
    glob, _, _ = failures(fx)
    assert {"plan.counts", "plan.expected_direct_count", "class.recomputed_counts"} <= set(glob)


def test_recomputed_class_counts_must_match_plan_counts(tmp_path):
    fx = build(tmp_path)
    pins = dataclasses.replace(fx.pins, expected_counts=(("direct", 1), ("pinned", 6),
                                                         ("absent", 0)))
    assert run13(fx, pins=pins) == 2
    glob, _, _ = failures(fx)
    assert "class.recomputed_counts" in glob and "plan.counts" in glob


def test_verification_failure_lists_all_failures_and_no_complete(tmp_path):
    """A11(ix): failures on several slides are all collected; only VERIFICATION_FAILED.json."""
    def fr_hook(sid, rec):
        if sid == "s02":
            rec["code_sha256"] = "1" * 64
            rec["schema"] = "bad"
    def s_hook(sid, rec):
        if sid == "s06":
            rec["fingerprint_verification"]["mismatches"] = 2
    fx = build(tmp_path, fr_rec_hook=fr_hook, s_rec_hook=s_hook,
               mb_hook=lambda sid, obj: (obj[0] + np.float32(1), obj[1]) if sid == "s07" else None)
    assert run13(fx) == 2
    glob, per, doc = failures(fx)
    assert per == {"s02": ["first_run.schema", "first_run.code_sha256"],
                   "s06": ["suite.fingerprint_verification.mismatches"],
                   "s07": ["identity_equals_main_bank", "suite_identity_equals_main_bank"]}
    assert doc["n_slides_checked"] == 7 and doc["n_slides_failed"] == 3
    assert doc["n_failures"] == 5 and glob == []
    assert not (tmp_path / "run_001" / "COMPLETE").exists()
    f = doc["slide_failures"]["s02"][1]
    assert f["expected"] == A.Pins().extractor_code_sha256 and f["observed"] == "1" * 64


def test_code_and_population_sha_bindings(tmp_path):
    fx = build(tmp_path)
    assert run13(fx, adapter_sha256="0" * 64) == 2
    glob, _, _ = failures(fx)
    assert glob == ["code_sha256.adapter"]
    pins = dataclasses.replace(fx.pins, e_sha256="0" * 64)
    assert run13(fx, "run_002", pins=pins) == 2
    glob, _, _ = failures(fx, "run_002")
    assert glob == ["code_sha256.E"]
    assert run13(fx, "run_003", population_sha256="0" * 64) == 2
    glob, _, _ = failures(fx, "run_003")
    assert glob == ["population.sha256_flag"]


def test_unreadable_plan_is_a_verification_failure(tmp_path):
    fx = build(tmp_path)
    assert run13(fx, plan=tmp_path / "missing.json") == 2
    glob, _, doc = failures(fx)
    assert glob == ["inputs.readable"]


@pytest.mark.parametrize("bad", [["s01"], {"id": "s01"}, 1, None])
def test_non_str_plan_slide_id_is_a_verification_failure(tmp_path, bad):
    """Review 001 L1: a non-str (incl. unhashable) PLAN slide_id is collected, never raised."""
    def hook(plan):
        plan["slides"][0]["slide_id"] = bad
    fx = build(tmp_path, plan_hook=hook)
    assert run13(fx) == 2
    glob, per, doc = failures(fx)
    assert "plan.slide_id" in glob
    f = next(f for f in doc["global_failures"] if f["check"] == "plan.slide_id")
    assert (f["expected"], f["observed"]) == ("str", repr(bad))
    assert per == {"s01": ["plan.slide_entry"]}
    assert not (tmp_path / "run_001" / "COMPLETE").exists()


def test_unexpected_exception_after_writer_is_collected(tmp_path, monkeypatch):
    """Review 001 L1: any other exception after the Writer exists (here: raised from the
    per-slide loop) leaves only VERIFICATION_FAILED.json, exit 2."""
    fx = build(tmp_path)
    def boom(task):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(A, "_verify_slide", boom)
    assert run13(fx) == 2
    glob, per, doc = failures(fx)
    assert glob == ["inputs.exception"] and per == {}
    f = doc["global_failures"][0]
    assert f["observed"] == "RuntimeError: synthetic failure" and "traceback" in f


def test_non_dict_plan_counts_and_slides_are_collected(tmp_path):
    def hook(plan):
        plan["counts"] = ["not", "a", "dict"]
        plan["slides"] = "nope"
    fx = build(tmp_path, plan_hook=hook)
    assert run13(fx) == 2
    glob, per, _ = failures(fx)
    assert "plan.counts" in glob and set(per) == {s for s, _, _ in SLIDES}


@pytest.mark.parametrize("cls,hook_kw,check", [
    ("pinned", "fr_rec_hook", "first_run.reinhard_nonfinite_tile_indices"),
    ("direct", "s_rec_hook", "suite.reinhard_nonfinite_tile_indices"),
])
def test_missing_reinhard_nonfinite_key_on_read_source_fails(tmp_path, cls, hook_kw, check):
    """Review 001 L2: a missing key on the record actually read for reinhard is a failure,
    not an empty list."""
    def hook(sid, rec):
        if sid == "s04":
            del rec["reinhard_nonfinite_tile_indices"]
    fx = build(tmp_path, classes={"s04": cls}, **{hook_kw: hook})
    assert run13(fx) == 2
    glob, per, doc = failures(fx)
    assert glob == [] and per == {"s04": [check]}
    f = doc["slide_failures"]["s04"][0]
    assert (f["expected"], f["observed"]) == ("present", "missing")


@pytest.mark.parametrize("cls,hook_kw", [("direct", "fr_rec_hook"), ("pinned", "s_rec_hook")])
def test_missing_reinhard_nonfinite_key_on_unread_source_is_ignored(tmp_path, cls, hook_kw):
    """The key is only required on the record of the source read for reinhard (P2-A13)."""
    def hook(sid, rec):
        if sid == "s04":
            del rec["reinhard_nonfinite_tile_indices"]
    fx = build(tmp_path, classes={"s04": cls}, **{hook_kw: hook})
    assert run13(fx) == 0


# P2-A1 sentence, verbatim from [decision record] (line breaks joined with single spaces)
P2_A1_SENTENCE = (
    "the first-run 12-arm tree is lead-reconciled; its **independent content evidence** is (i) "
    "`SUITE/review/pilot_output_review_001/REVIEW.md` — ACCEPT WITH AMENDMENTS: 13 stratified "
    "slides (all 13 first-run groups, 3 CPU models, one slide per first-run host) × 12 arms "
    "re-encoded with frozen X on two A100 hosts, whole pickles byte-identical for 12/13 slides "
    "and 11/12 arms for 10057 (only `reinhard`, = G3, expected); and (ii) features output "
    "review 001 — identity arm bit-exact 2,778/2,778 and direct/pinned class agreement "
    "2,778/2,778. P2.3 adds per-slide integrity, record binding, routing and identity == "
    "main-bank checks; it does **not** re-encode or otherwise verify the content of the ten "
    "non-identity arms on the remaining 2,765 slides. The reuse chain remains \"established, "
    "not proved\" (item 8); the P2 result package must disclose this (V11).")


def test_v11_disclosure_is_verbatim_p2_a1():
    """Review 001 L3."""
    assert A.V11_DISCLOSURE == P2_A1_SENTENCE
    # cross-check against the ADR itself when the revision workspace is mounted alongside
    adr = (Path(__file__).resolve().parents[2] / "prosper_revision" / "docs" / "adr"
           / "0021-h-optimus-comparable-benchmarks.md")
    if adr.is_file():
        joined = " ".join(ln.strip() for ln in adr.read_text().split("\n"))
        assert P2_A1_SENTENCE in joined


def test_zero_shift_requires_python_int_zero(tmp_path):
    """Review 001 N2: False / 0.0 shifts are not the zero-shift null stimulus."""
    def hook(sid, rec):
        if sid == "s03":
            for st in rec["stimuli"]:
                if st["condition"] == "spatial_shift":
                    st["parameters"] = {"x_shift": False, "y_shift": 0.0}
    fx = build(tmp_path, fr_rec_hook=hook)
    assert run13(fx) == 2
    glob, per, _ = failures(fx)
    assert glob == ["zero_shift.slide_set"] and per == {}


def test_missing_input_file_is_collected_not_raised(tmp_path):
    fx = build(tmp_path)
    (fx.suite_dir / "s04.pkl").unlink()
    assert run13(fx) == 2
    _, per, _ = failures(fx)
    assert per == {"s04": ["exception"]}


def test_main_cli_uses_production_pins_and_exit_codes(tmp_path, capsys):
    fx = build(tmp_path)
    argv = ["--population", str(fx.pop_path), "--population-sha256", fx.pop_sha,
            "--first-root", str(tmp_path / "first"), "--suite-root", str(tmp_path / "suite"),
            "--out", str(tmp_path / "run_cli"), "--adapter-sha256", _adapter_sha(),
            "--draw-count", "10"]
    assert A.main(argv) == 2           # synthetic inputs do not match the pinned production shas
    assert "VERIFICATION FAILED" in capsys.readouterr().err
    glob, _, doc = failures(fx, "run_cli")
    assert "population.sha256_pinned" in glob and "plan.sha256" in glob
    assert doc["argv"][1:] == argv
    with pytest.raises(SystemExit) as ei:
        A.main(argv[:-4])              # --adapter-sha256 is required
    assert ei.value.code == 2
    with pytest.raises(SystemExit):
        A.main(argv + ["--waive", "s01"])   # no waiver flag exists (P2-A6)
