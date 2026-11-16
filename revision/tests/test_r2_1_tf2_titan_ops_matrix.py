"""Synthetic tests for scripts/r2_1_tf2_titan_ops.py and scripts/r2_1_tf2_titan_matrix.py
(R2.1-TF2 operations release 006: TITAN staging/verify/delete and the TITAN filtered-arm matrices).

Covers: set planning (queries masked by their own S0 head, versions by their slot, mask != prediction
STOPs), verify PASS/FAIL and delete-only-staged-copies, the eligibility relabel (encoder column only),
the L1 rule (cdist == the R2.6 pdist rows), and build(): NaN outside the row's version, self sentinel,
same-patient twin, F7 (iii) refusal, and acceptance of the matrices + sidecar by the S6 evaluator.
Run with  python3 -m pytest -p no:cacheprovider --basetemp=<tmp> tests/test_r2_1_tf2_titan_ops_matrix.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OPS = _load("r2_1_tf2_titan_ops", REPO / "scripts" / "r2_1_tf2_titan_ops.py")
MX = _load("r2_1_tf2_titan_matrix", REPO / "scripts" / "r2_1_tf2_titan_matrix.py")
from prosper.evaluation import r2_1_tf2_eval_cli as T  # noqa: E402
from prosper.evaluation import r2_1_filter_eval_cli as fe  # noqa: E402

QUIET = dict(log=lambda *_: None)
IDS = ["11", "12", "13", "14", "15", "16"]
PAT = ["P1", "P1", "P2", "P3", "P4", "P4"]
GRP = ["A", "A", "B", "B", "C", "C"]
SLOT = ["fold0", "fold0", "fold1", "full", "full", "full"]   # S0 head slot per query
GPU_TXT = ("nodegpuX\nGPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-1)\nNVIDIA A100-SXM4-80GB, 595.58.03\n")


def pop_frame():
    return pd.DataFrame({"query_index": range(6), "reference_index": range(6), "slide_id": IDS,
                         "patient_id": PAT, "diagnostic_group": GRP})


def qa_frame():
    return pd.DataFrame({"query_index": range(6), "slide_id": IDS, "patient_id": PAT, "diagnostic_group": GRP,
                         "patient_annotated": [1, 1, 1, 0, 0, 0], "patient_fold": [0, 0, 1, 2, 3, 3],
                         "stratum": ["a"] * 3 + ["u"] * 3, "head_slot": SLOT,
                         "library_version_both_sided": ["v_" + s for s in SLOT],
                         "library_version_query_only": ["unfiltered"] * 6})


# ---------------------------------------------------------------- set planning
def _offsets_and_masks(n_tiles=4):
    off = pd.DataFrame({"slide_index": range(6), "slide_id": IDS, "patient_id": PAT, "in_population": 1,
                        "query_index": range(6), "tile_start": [i * n_tiles for i in range(6)],
                        "tile_stop": [(i + 1) * n_tiles for i in range(6)], "feature_sha256": ["x"] * 6,
                        "bank_file": [f"/bank/{s}.h5" for s in IDS]})
    rng = np.random.default_rng(0)
    preds = {s: rng.integers(1, 3, size=6 * n_tiles) for s in OPS.SLOTS}
    masks = {s: preds[s] == 1 for s in OPS.SLOTS}
    coords = np.arange(6 * n_tiles * 2).reshape(-1, 2)
    return off, masks, preds, coords


def test_set_plan_routes_queries_to_their_own_head():
    off, masks, preds, coords = _offsets_and_masks()
    m, keep, rows, _ = OPS.set_plan("queries", 100, pop_frame(), qa_frame(), off, masks, preds, coords)
    assert m == IDS
    for i, s in enumerate(IDS):
        assert rows[s]["head_id"] == f"conch_v15/b100/{SLOT[i]}" and rows[s]["library_version"] == "v_" + SLOT[i]
        assert np.array_equal(keep[s], masks[SLOT[i]][i * 4:(i + 1) * 4])
    m, keep, rows, _ = OPS.set_plan("v_fold1", 5, pop_frame(), qa_frame(), off, masks, preds, coords)
    assert {r["head_id"] for r in rows.values()} == {"conch_v15/b5/fold1"}
    assert np.array_equal(keep["11"], masks["fold1"][0:4])


def test_set_plan_stops_when_mask_is_not_prediction_eq_1():
    off, masks, preds, coords = _offsets_and_masks()
    masks["full"] = ~masks["full"]
    with pytest.raises(OPS.OpsError, match="mask != "):
        OPS.set_plan("v_full", 100, pop_frame(), qa_frame(), off, masks, preds, coords)


# ---------------------------------------------------------------- verify / delete
def _fake_set(tmp, f6_ok=True, drop_embedding=None):
    src, wsi = tmp / "src", tmp / "wsi"
    src.mkdir()
    wsi.mkdir()
    keep = {}
    for k, s in enumerate(["201", "202", "203"]):
        with h5py.File(src / f"{s}.h5", "w") as h:
            c = h.create_dataset("coords", data=np.arange(10).reshape(5, 2) + 100 * k)
            c.attrs["patch_size_level0"] = 1024
            h.create_dataset("features", data=np.random.default_rng(k).normal(size=(5, 768)).astype("float32"))
        (wsi / f"{s}.svs").write_text("x")
        keep[s] = np.array([1, 0, 1, 1, k > 0], dtype=bool) if s != "203" else np.zeros(5, dtype=bool)
    out = tmp / "stage"
    out.mkdir()
    rec = OPS.S4.titan_stage("v_full", ["201", "202", "203"], keep, src, wsi, out)
    rows = {s: {"slide_id": s, "head_slot": "full", "head_id": "conch_v15/b100/full", "library_version": "v_full",
                "original_tiles": 5, "retained_tiles": int(keep[s].sum()),
                "bank_feature_sha256": OPS.sha256_file(src / f"{s}.h5")} for s in keep}
    OPS.write_manifest(out, "set", 100, "v_full", rec, rows, {})
    job = out / "job_dir"
    (job / "gpu_info.txt").write_text(GPU_TXT)
    cdir = job / OPS.REGEN.COORDS
    (cdir / OPS.REGEN.SLIDE_CONFIG).write_text(json.dumps({"skip_errors": False}))
    emb = cdir / "slide_features_titan"
    emb.mkdir()
    for x in rec["staged"]:
        s = x["slide_id"]
        if s == drop_embedding:
            continue
        with h5py.File(cdir / "features_conch_v15" / f"{s}.h5", "r") as st, h5py.File(emb / f"{s}.h5", "w") as h:
            h.create_dataset("features", data=np.random.default_rng(int(s)).normal(size=768).astype("float32"))
            c = h.create_dataset("coords", data=st["coords"][:])
            for k, v in st["coords"].attrs.items():
                c.attrs[k] = v
    f6 = tmp / "f6_record.json"
    f6.write_text(json.dumps({"status": "PASS" if f6_ok else "FAIL", "gpu_model": "NVIDIA A100-SXM4-80GB"}))
    return out, f6, rec


def test_verify_pass_then_delete_only_staged_copies(tmp_path):
    out, f6, rec = _fake_set(tmp_path)
    assert rec["not_staged_zero_tiles"] == ["203"]
    v = OPS.verify(out, f6, **QUIET)
    assert v["status"] == "PASS", v["checks"]
    done = OPS.delete_staged(out, "test-operator", **QUIET)
    feat = out / "job_dir" / OPS.REGEN.COORDS / "features_conch_v15"
    assert done["deleted"] == 2 and list(feat.iterdir()) == []
    assert sorted(p.name for p in (out / "job_dir" / OPS.REGEN.COORDS / "slide_features_titan").iterdir()) == \
        ["201.h5", "202.h5"]
    dr = json.loads((out / "deletion_record.json").read_text())
    assert dr["operator"] == "test-operator" and {f["sha256"] for f in dr["files"]} == \
        {x["staged_sha256"] for x in rec["staged"]}
    assert (out / "stage_record.json").exists() and (out / "wsi_dir" / "201.svs").is_symlink()


def test_verify_fails_on_missing_embedding_or_f6_and_delete_refuses(tmp_path):
    out, f6, _ = _fake_set(tmp_path, drop_embedding="202")
    v = OPS.verify(out, f6, **QUIET)
    assert v["status"] == "FAIL" and v["checks"]["every_staged_slide_has_embedding"]["missing"] == ["202"]
    with pytest.raises(OPS.OpsError, match="not PASS"):
        OPS.delete_staged(out, "op", **QUIET)
    (tmp_path / "b").mkdir()
    out2, f62, _ = _fake_set(tmp_path / "b", f6_ok=False)
    assert OPS.verify(out2, f62, **QUIET)["checks"]["f6_passed_beforehand"]["result"] == "FAIL"


# ---------------------------------------------------------------- matrix
def test_l1_rows_equal_r26_pdist_rows():
    from scipy.spatial.distance import pdist, squareform
    X = np.random.default_rng(3).normal(size=(7, 768)).astype("float32").astype(np.float64)
    full = squareform(pdist(X, metric="cityblock"))
    assert np.array_equal(MX.l1_rows(X[2:4], X), full[2:4])


def _elig_rows(budget, versions):
    rows = []
    for arm, vers in (("query_only", ["unfiltered"] * 6), ("both_sided", ["v_" + s for s in SLOT])):
        for i, s in enumerate(IDS):
            mem = IDS if vers[i] == "unfiltered" else versions[vers[i]]
            rows.append({"encoder": "titan", "budget_percent": str(budget), "query": s, "query_index": i,
                         "arm": arm, "head_slot": SLOT[i], "head_id": f"conch_v15/b{budget}/{SLOT[i]}",
                         "library_version": vers[i], "query_returned": 0 if s == "16" else 1,
                         "n_eligible": len(mem), "eligible_ids_sha256": T.ids_sha(mem),
                         "eligible_reference_ids": ";".join(mem)})
    return rows


def _emb_set(root, name, budget, vecs, heads):
    d = root / f"b{budget}" / name
    emb = d / "job_dir" / "e"
    emb.mkdir(parents=True)
    sha = {}
    for s, v in vecs.items():
        with h5py.File(emb / f"{s}.h5", "w") as h:
            h.create_dataset("features", data=v.astype("float32"))
        sha[s] = OPS.sha256_file(emb / f"{s}.h5")
    man = {"set": name, "budget_percent": budget,
           "slides": [{"slide_id": s, "head_id": heads[s]} for s in vecs]}
    (d / "stage_manifest.json").write_text(json.dumps(man))
    (d / "verify_record.json").write_text(json.dumps({"status": "PASS", "embeddings_dir": str(emb),
                                                      "embedding_sha256": sha,
                                                      "stage_manifest_sha256": OPS.sha256_file(d / "stage_manifest.json")}))


@pytest.fixture
def mworld(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    budget = 100
    s0 = tmp_path / "s0"
    s0.mkdir()
    qa_frame().to_csv(s0 / "query_assignment.csv", index=False)
    pop = pop_frame()
    monkeypatch.setattr(OPS, "population", lambda *_: fe.load_population(_pop_csv(tmp_path, pop)))
    versions = {"v_fold0": ["11", "12", "13", "15", "16"], "v_fold1": IDS, "v_full": ["11", "13", "14", "15"]}
    elig = tmp_path / "elig" / "b100.csv.gz"
    T.write_eligibility_table(pd.DataFrame(_elig_rows(budget, versions)), elig)
    troot = tmp_path / "titan"
    full = {s: rng.normal(size=768) for s in IDS}          # accepted unfiltered embeddings
    q = {s: rng.normal(size=768).astype("float32").astype(np.float64) for s in IDS if s != "16"}
    _emb_set(troot, "queries", budget, q, {s: f"conch_v15/b{budget}/{SLOT[IDS.index(s)]}" for s in q})
    vv = {}
    for v, mem in versions.items():
        vecs = {s: (q[s] if s in q and SLOT[IDS.index(s)] == v[2:] else rng.normal(size=768)) for s in mem}
        vv[v] = {s: x.astype("float32").astype(np.float64) for s, x in vecs.items()}
        _emb_set(troot, v, budget, vecs, {s: f"conch_v15/b{budget}/{v[2:]}" for s in mem})
    acc = (IDS, np.stack([full[s] for s in IDS]), {"baseline_matrix": {"sha256": "x"}})
    return dict(tmp=tmp_path, s0=s0, elig=elig, troot=troot, full=full, q=q, vv=vv, versions=versions,
                acc=acc, budget=budget)


def _pop_csv(tmp, pop):
    p = tmp / "pop.csv"
    if not p.exists():
        pop.to_csv(p, index=False)
    return p


@pytest.mark.parametrize("arm", ["query_only", "both_sided"])
def test_build_layout_and_evaluator_acceptance(mworld, arm):
    w = mworld
    out = w["tmp"] / "out" / arm
    rec = MX.build(w["budget"], arm, w["s0"], w["troot"], w["elig"], out, "/unused", accepted=w["acc"], **QUIET)
    base = MX.matrix_name(100, arm, False)
    M = pd.read_pickle(out / base)
    Mm = pd.read_pickle(out / MX.matrix_name(100, arm, True))
    assert list(M.index) == IDS[:5] and list(M.columns) == IDS and rec["rows"] == 5
    for q in IDS[:5]:
        v = "unfiltered" if arm == "query_only" else "v_" + SLOT[IDS.index(q)]
        lib = IDS if v == "unfiltered" else w["versions"][v]
        for s in IDS:
            x = M.loc[q, s]
            if s not in lib:
                assert np.isnan(x)
            elif s == q:
                assert x == MX.SENTINEL
            else:
                ref = w["full"][s] if v == "unfiltered" else w["vv"][v][s]
                assert x == pytest.approx(float(np.abs(w["q"][q] - ref).sum()), rel=1e-12)
    assert Mm.loc["11", "12"] == MX.SENTINEL and M.loc["11", "12"] != MX.SENTINEL      # same patient P1
    if arm == "both_sided":
        assert np.isnan(Mm.loc["14", "12"]) and rec["query_embedding_equals_own_version_embedding"] == 5
    # the S6 evaluator accepts matrix + sidecar
    pop = fe.load_population(_pop_csv(w["tmp"], pop_frame()))
    qa = T.load_query_assignment(w["s0"] / "query_assignment.csv", pop)
    E = T.read_eligibility(w["elig"], pop, "titan", "100", qa)["query_only" if arm == "query_only" else "both_filtered"]
    mp, sp = out / MX.matrix_name(100, arm, True), out / f"{base[:-4]}_rows.json"
    spec = T.ArmSpec("titan", "query_only" if arm == "query_only" else "both_filtered", "titan", "100", mp,
                     fe.sha256_file(mp), sp, fe.sha256_file(sp))
    dist, diag = T.load_arm_distances(spec, pop, E.returned, E.lib, strict_outside=True)
    assert diag["non_nan_cells_outside_eligible"] == 0
    assert T.check_rows_sidecar(spec, pop, E)["fields_checked"] == ["library_version", "library_ids_sha256",
                                                                     "library_size", "head_id"]


def test_build_refuses_head_mismatch(mworld):
    w = mworld
    d = w["troot"] / "b100" / "queries"
    man = json.loads((d / "stage_manifest.json").read_text())
    man["slides"][0]["head_id"] = "conch_v15/b100/full"
    (d / "stage_manifest.json").write_text(json.dumps(man))
    v = json.loads((d / "verify_record.json").read_text())
    v["stage_manifest_sha256"] = OPS.sha256_file(d / "stage_manifest.json")
    (d / "verify_record.json").write_text(json.dumps(v))
    with pytest.raises(OPS.OpsError, match="F7"):
        MX.build(100, "both_sided", w["s0"], w["troot"], w["elig"], w["tmp"] / "o", "/unused", **QUIET)


def test_titan_min_tiles_rule_excludes_one_tile_slides():
    keep = {"a": np.array([1, 0, 0], bool), "b": np.array([1, 1, 0], bool), "c": np.zeros(3, bool),
            "d": np.ones(3, bool)}
    kept, below = OPS.apply_min_tiles(["a", "b", "c", "d"], keep)
    assert OPS.TITAN_MIN_TILES == 2
    assert kept == ["b", "c", "d"] and below == [{"slide_id": "a", "tiles": 1}]   # zero-tile c left to titan_stage


def test_titan_stage_record_lists_below_min_tiles(tmp_path, monkeypatch):
    """stage(): a 1-tile slide is not staged and is listed in stage_record.json with its tile count."""
    calls = {}

    def fake_stage(set_name, members, keep_of, conch_dir, wsi, out):
        calls["members"] = list(members)
        rec = {"set": set_name, "job_dir": str(out / "job_dir"), "wsi_dir": str(out / "wsi_dir"), "staged": [],
               "not_staged_zero_tiles": [], "trident_command": "CMD", "executed_by_this_script": False}
        OPS.write_json(rec, out / "stage_record.json")
        return rec
    off, masks, preds, coords = _offsets_and_masks()
    masks["full"][:] = False
    preds["full"][:] = 2
    preds["full"][0] = 1
    masks["full"][0] = True                                  # slide 11: exactly 1 tile under full
    monkeypatch.setattr(OPS, "population", lambda *_: pop_frame())
    monkeypatch.setattr(OPS, "query_assignment", lambda *_: qa_frame())
    monkeypatch.setattr(OPS, "load_s3_unit", lambda *_: (tmp_path, {}, off, masks, preds, coords))
    monkeypatch.setattr(OPS, "_check_h5_coords", lambda *a: None)
    monkeypatch.setattr(OPS.S4, "titan_stage", fake_stage)
    monkeypatch.setattr(OPS, "write_manifest", lambda *a, **k: {"n_staged": 0, "staged_tiles": 0, "staged_bytes": 0,
                                                                 "seconds": 0})
    monkeypatch.setattr(OPS, "_check_under", lambda p, po: Path(p))
    monkeypatch.setattr(OPS, "sha256_file", lambda p: "x")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"sets": [{"budget_percent": 100, "set": "v_full", "stage_dir": str(tmp_path / "st"),
                                          "trident_command": "CMD"}]}))
    OPS.stage(100, "v_full", tmp_path, tmp_path, spec, "/unused", **QUIET)
    rec = json.loads((tmp_path / "st" / "stage_record.json").read_text())
    assert rec["not_staged_below_min_tiles"] == [{"slide_id": "11", "tiles": 1}] and rec["min_tiles"] == 2
    assert "11" not in calls["members"] and len(calls["members"]) == 5


def test_titan_eligibility_min_tiles_2_and_relabel(tmp_path):
    src_dir = tmp_path / "conch_v15"
    versions1 = {"v_fold0": IDS, "v_fold1": IDS, "v_full": IDS}
    rows = pd.DataFrame(_elig_rows(5, versions1))
    rows["encoder"] = "conch_v15"
    src_dir.mkdir()
    OPS.S0.write_csv(rows[OPS.S4.ELIGIBILITY_COLUMNS], src_dir / "b5.csv.gz")
    (src_dir / "b5_record.json").write_text(json.dumps({"sha256": OPS.sha256_file(src_dir / "b5.csv.gz"),
                                                        "encoder": "conch_v15", "budget_percent": 5, "min_tiles": 1,
                                                        "s3_unit": str(tmp_path / "s3unit")}))
    seen = {}

    def fake_elig(s3_unit, s0, enc, budget, prosper_out=None, min_tiles=1):
        seen.update(min_tiles=min_tiles, enc=enc, s3=str(s3_unit))
        v2 = {**versions1, "v_fold1": [s for s in IDS if s != "13"]}     # "13" has 1 tile under fold1
        df = pd.DataFrame(_elig_rows(5, v2))
        df.loc[df["query"] == "13", "query_returned"] = 0                # its own head is fold1
        df["encoder"] = "conch_v15"
        return df[OPS.S4.ELIGIBILITY_COLUMNS]
    rec = MX.titan_eligibility(src_dir / "b5.csv.gz", tmp_path / "titan", 5, tmp_path, "/unused",
                               eligibility_fn=fake_elig, **QUIET)
    assert seen == {"min_tiles": 2, "enc": "conch_v15", "s3": str(tmp_path / "s3unit")}
    b = pd.read_csv(tmp_path / "titan" / "b5.csv.gz", dtype=str, keep_default_na=False)
    assert set(b.encoder) == {"titan"} and rec["rows"] == 12 and rec["min_tiles"] == 2
    assert rec["removed_ids_by_version_vs_min_tiles_1"] == {"v_fold1": ["13"]}
    both13 = b[(b["query"] == "13") & (b["arm"] == "both_sided")].iloc[0]
    assert both13.query_returned == "0" and both13.n_eligible == "5" and "13" not in both13.eligible_reference_ids.split(";")
    assert both13.eligible_ids_sha256 == T.ids_sha([s for s in IDS if s != "13"])
