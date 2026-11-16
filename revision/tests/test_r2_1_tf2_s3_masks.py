"""Synthetic-fixture tests for scripts/r2_1_tf2_s3_masks.py (R2.1-TF2 S3).

Mocked banks and heads only: a 12-slide library (9 population slides) with UNI2 (1,536-d) and
Virchow2 (1,280-d) pickles on one coordinate grid, real FittedHead objects fitted with the S2 recipe
path on synthetic data, a mocked S0 directory, F7 (ii) records and a route-b-style mosaic bundle.
`build_world` is reused by the S4 and S5 tests. Run with
    python3 -m pytest -p no:cacheprovider --basetemp=<tmp> tests/test_r2_1_tf2_s3_masks.py
"""
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


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S3 = _load("r2_1_tf2_s3_masks", REPO / "scripts" / "r2_1_tf2_s3_masks.py")
S2, S0 = S3.S2, S3.S0

LIB = [str(100 + i) for i in range(1, 13)]                 # 101..112
POP = LIB[:9]                                              # population 101..109
GROUP = {"101": "A", "102": "A", "103": "B", "104": "B", "105": "C", "106": "C", "107": "A", "108": "B",
         "109": "C", "110": "NORMAL", "111": "MISC", "112": "NORMAL"}
PATIENT = {"101": "P1", "102": "P1", "103": "P2", "104": "P3", "105": "P4", "106": "P5", "107": "P6",
           "108": "P7", "109": "P8", "110": "P9", "111": "P10", "112": "P11"}
FOLD = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "P5": 4, "P6": 0, "P7": 1, "P8": 2, "P9": 3, "P10": 4, "P11": 0}
ANNOTATED = {"P1", "P2"}                                   # P1 in fold 0, P2 in fold 1
TILES = {"101": 30, "102": 12, "103": 25, "104": 6, "105": 20, "106": 18, "107": 9, "108": 40, "109": 14,
         "110": 10, "111": 16, "112": 22}
WIDTH = {"uni2": 1536, "virchow2": 1280}
SEED = 7


def _centers(width):
    rng = np.random.default_rng(99)
    return rng.normal(0, 1, (5, width)).astype(np.float32) * 3


def _tile_classes(slide):
    rng = np.random.default_rng(int(slide))
    return rng.choice(np.arange(1, 6), size=TILES[slide], p=[0.6, 0.1, 0.1, 0.1, 0.1])


def _features(slide, width):
    rng = np.random.default_rng(int(slide) * 10 + width)
    c = _centers(width)
    cls = _tile_classes(slide)
    return (c[cls - 1] + rng.normal(0, 1.2, (len(cls), width))).astype(np.float32)


def _coords(slide):
    n = TILES[slide]
    return [{"wsi_loc": [512 * (i % 7), 512 * (i // 7) + 1024 * int(slide[-1])]} for i in range(n)]


def _write(path, df):
    df.to_csv(path, index=False, lineterminator="\n")


def fit_head(width, seed, budget):
    """A real FittedHead through S2.fit_generic on synthetic class-centre data."""
    rng = np.random.default_rng(seed)
    c = _centers(width)
    n = 60 if budget == 100 else 25
    y = np.repeat(np.arange(1, 6), n)
    X = np.ascontiguousarray((c[y - 1] + rng.normal(0, 1.5, (len(y), width))).astype(np.float32))
    stats = [{"tissue_label": k, "pool_count": n, "selected_count": n, "fraction": 1.0,
              "balanced_weight": len(y) / (5 * n)} for k in range(1, 6)]
    frame = pd.DataFrame({"slide_id": ["s"] * len(y), "patient_id": ["p"] * len(y), "tissue_label": y,
                          "tile_id": [f"t{i}" for i in range(len(y))]})
    with threadpool_limits(limits=1):
        return S2.fit_generic(X, y, {k: s["balanced_weight"] for k, s in zip(range(1, 6), stats)}, width=width,
                              context="test", held_out_fold=None, budget=budget, frame=frame,
                              class_stats_rows=stats, pool_count=len(y), selected_fraction=1.0)


def build_world(root: Path, *, encoders=("uni2", "virchow2")) -> dict:
    root = Path(root)
    prosper_out = root / "prosper_out"
    run = prosper_out / S0.RUN_NAME
    s0 = run / "s0"
    for d in (s0, run / "s2" / "f7"):
        d.mkdir(parents=True)
    banks = {}
    bank_role = {"uni2": "lib_uni2_dir", "virchow2": "lib_virchow2_fp32_dir"}
    for enc in encoders:
        d = root / "banks" / enc
        d.mkdir(parents=True)
        for s in LIB:
            with open(d / f"{s}.pkl", "wb") as fh:
                pickle.dump((_features(s, WIDTH[enc]), _coords(s)), fh, protocol=4)
        banks[enc] = d
    for extra in ("h_optimus_1", "trident"):
        (root / "banks" / extra).mkdir(parents=True)
    inputs = {"lib_uni2_dir": str(banks["uni2"]), "lib_virchow2_fp32_dir": str(banks.get("virchow2", root / "banks" / "virchow2")),
              "lib_h_optimus_1_dir": str(root / "banks" / "h_optimus_1"), "trident_root": str(root / "banks" / "trident")}
    lib_map = pd.DataFrame({"Tumor.group": [GROUP[s] for s in LIB], "Slide.Scan.File": LIB,
                            "Sample.ID": [f"S{s}" for s in LIB], "UID.Subject": [PATIENT[s] for s in LIB]})
    _write(root / "library_map.csv", lib_map)
    inputs["library_map"] = str(root / "library_map.csv")
    pop = pd.DataFrame({"query_index": range(9), "reference_index": range(9), "metadata_index": range(9),
                        "slide_id": POP, "patient_id": [PATIENT[s] for s in POP],
                        "diagnostic_group": [GROUP[s] for s in POP], "fold": [FOLD[PATIENT[s]] for s in POP]})
    _write(root / "retrieval_population.csv", pop)
    inputs["retrieval_population"] = str(root / "retrieval_population.csv")
    rows = []
    for i, s in enumerate(POP):
        p = PATIENT[s]
        ann = int(p in ANNOTATED)
        slot = f"fold{FOLD[p]}" if ann else "full"
        rows.append({"query_index": i, "slide_id": s, "patient_id": p, "diagnostic_group": GROUP[s],
                     "patient_annotated": ann, "patient_fold": FOLD[p],
                     "stratum": slot if ann else "unannotated", "head_slot": slot,
                     "library_version_both_sided": "v_" + slot, "library_version_query_only": "unfiltered",
                     "f7_i_query_patient_in_planned_training": 0})
    _write(s0 / "query_assignment.csv", pd.DataFrame(rows))
    _write(s0 / "patient_folds_tf2.csv", pd.DataFrame({"patient_id": list(FOLD), "fold": list(FOLD.values())}))
    _write(s0 / "pool.csv", pd.DataFrame({"slide_id": ["101", "103"], "labelled_tiles": [30, 25], "in_pool": [1, 1],
                                          "in_final_library": [1, 1], "provenance_class": ["library", "library"],
                                          "UID.Subject": ["P1", "P2"], "patient_source": ["library_map"] * 2,
                                          "qc_bad_slide": [0, 0], "diagnostic_group": ["A", "B"]}))
    files = []
    for enc in encoders:
        for s in LIB:
            p = banks[enc] / f"{s}.pkl"
            files.append({"role": bank_role[enc], "path": str(p), "sha256": S0.sha256_file(p)})
    files.append({"role": "library_map", "path": inputs["library_map"], "sha256": S0.sha256_file(inputs["library_map"])})
    files.append({"role": "retrieval_population", "path": inputs["retrieval_population"],
                  "sha256": S0.sha256_file(inputs["retrieval_population"])})
    dirs = [{"role": r, "path": inputs[r]} for r in ("lib_uni2_dir", "lib_virchow2_fp32_dir", "lib_h_optimus_1_dir",
                                                       "trident_root")]
    S0.write_json({"files": files, "directories": dirs}, s0 / "input_bindings.json")
    finalize_s0(s0)
    heads_root = run / "s2"
    f7rows = []
    for enc in encoders:
        for b in (100, 5):
            for k, slot in enumerate(S3.SLOTS):
                d = heads_root / "heads" / enc / f"b{b}" / slot
                d.mkdir(parents=True)
                h = fit_head(WIDTH[enc], SEED + k + (0 if b == 100 else 50), b)
                (d / "head.pkl").write_bytes(pickle.dumps(h, protocol=4))
                S0.write_json({"status": "COMPLETE", "head_id": f"{enc}/b{b}/{slot}",
                               "fitted_state_sha256": h.diagnostics["fitted_state_sha256"],
                               "outputs": {"head.pkl": {"sha256": S0.sha256_file(d / "head.pkl")}}}, d / "COMPLETE.json")
                f7rows.append({"head_id": f"{enc}/b{b}/{slot}", "status": "checked", "saved_training_patients": 2,
                               "queries_assigned": 1, "query_patients_in_saved_training": 0,
                               "saved_subset_of_planned": True, "saved_equals_plan_manifest": True,
                               "library_version": "v_" + slot, "library_version_head_matches": True})
    f7 = heads_root / "f7"
    _write(f7 / "f7_checkpoint_ii.csv", pd.DataFrame(f7rows))
    S0.write_json({"violations": [], "pass": True}, f7 / "f7_checkpoint_ii.json")
    # route-b-style mosaic bundle for the population slides (UNI2)
    mos = root / "routeb_mosaic"
    mos.mkdir()
    off, coords, start = [], [], 0
    for i, s in enumerate(POP):
        n = TILES[s]
        off.append({"slide_index": i, "slide_id": s, "patient_id": PATIENT[s], "diagnostic_group": GROUP[s],
                    "slide_fold": FOLD[PATIENT[s]], "tile_start": start, "tile_stop": start + n, "feature_rows": n,
                    "feature_sha256": S0.sha256_file(banks["uni2"] / f"{s}.pkl")})
        coords.append(np.array([c["wsi_loc"] for c in _coords(s)], np.int64))
        start += n
    _write(mos / "slide_offsets.csv", pd.DataFrame(off))
    np.save(mos / "coordinates.npy", np.concatenate(coords))
    return {"root": root, "prosper_out": str(prosper_out), "run": run, "s0": s0, "heads_root": heads_root,
            "f7": f7, "inputs": inputs, "routeb_mosaic": mos, "banks": banks}


def finalize_s0(s0: Path):
    outs = {p.name: {"sha256": S0.sha256_file(p)} for p in sorted(Path(s0).iterdir())
            if p.is_file() and p.name != "COMPLETE.json"}
    S0.write_json({"status": "COMPLETE", "outputs": outs}, Path(s0) / "COMPLETE.json")


def run_s3(world, enc, b, root=None):
    out = Path(root or world["run"] / "s3") / enc / f"b{b}"
    return S3.run_unit(enc, b, world["s0"], world["heads_root"], world["f7"], out, world["prosper_out"],
                       inputs=world["inputs"], routeb_mosaic=world["routeb_mosaic"], expected=(12, 9),
                       log=lambda *_: None), out


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = build_world(tmp_path_factory.mktemp("s3world"))
    S3.AGREEMENT_SAMPLE = 3
    return w


# ---------------------------------------------------------------- tests
def test_tumour_mask_rule_and_validator():
    p = np.array([[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1]], np.float32)
    y = np.array([1, 2], np.int64)
    assert S3.tumour_mask(p, y).tolist() == [True, False]
    with pytest.raises(S3.mosaic_rule.MosaicError, match="argmax"):
        S3.tumour_mask(p, np.array([2, 2], np.int64))          # label != argmax + 1


def test_readiness_ready_and_blocked(world, tmp_path):
    rd = S3.unit_readiness("uni2", 100, world["heads_root"], world["f7"])
    assert rd["ready"] and set(rd["heads"]) == set(S3.SLOTS)
    # F7 (ii) row with a leak blocks the unit
    f7 = tmp_path / "f7"
    shutil.copytree(world["f7"], f7)
    rows = pd.read_csv(f7 / "f7_checkpoint_ii.csv")
    rows.loc[rows["head_id"] == "uni2/b100/fold0", "query_patients_in_saved_training"] = 1
    rows.to_csv(f7 / "f7_checkpoint_ii.csv", index=False)
    rd2 = S3.unit_readiness("uni2", 100, world["heads_root"], f7)
    assert not rd2["ready"] and any("fold0" in r for r in rd2["reasons"])
    # an encoder without heads is blocked
    assert not S3.unit_readiness("h_optimus_1", 100, world["heads_root"], world["f7"])["ready"]


def test_run_unit_outputs_and_verify(world):
    m, out = run_s3(world, "uni2", 100)
    assert m["status"] == "COMPLETE" and m["slides"] == 12 and m["population_slides"] == 9
    off = pd.read_csv(out / "slide_offsets.csv", dtype={"slide_id": str})
    assert off["slide_id"].tolist()[:9] == POP and set(off["slide_id"][9:]) == {"110", "111", "112"}
    coords = np.load(out / "coordinates.npy")
    for r in off.itertuples():
        assert np.array_equal(coords[r.tile_start:r.tile_stop], np.array([c["wsi_loc"] for c in _coords(r.slide_id)]))
    for slot in S3.SLOTS:
        M, Y = np.load(out / f"{slot}_mask.npy"), np.load(out / f"{slot}_predictions.npy")
        assert M.dtype == bool and np.array_equal(M, Y == 1) and len(M) == len(coords)
        # predictions are the S2 head's own predictions on the bank features
    head = pickle.loads((world["heads_root"] / "heads/uni2/b100/full/head.pkl").read_bytes())
    p, y = S2.predict(head, _features("105", 1536), 1536)
    r = off.set_index("slide_id").loc["105"]
    assert np.array_equal(np.load(out / "full_predictions.npy")[r.tile_start:r.tile_stop], y)
    counts = pd.read_csv(out / "mask_counts.csv.gz", dtype={"slide_id": str})
    assert len(counts) == 6 * 12 and set(counts["library_version"]) == {f"v_{s}" for s in S3.SLOTS}
    cov = pd.read_csv(out / "coverage.csv")
    assert set(cov["scope"]) == {"library_all", "population"}
    share = pd.read_csv(out / "query_head_share.csv").set_index("diagnostic_group")
    assert share.loc["ALL", "queries"] == 9 and share.loc["ALL", "fold_head_queries"] == 3   # 101,102 (P1), 103 (P2)
    agr = pd.read_csv(out / "mask_agreement_per_slide.csv", dtype={"slide_id": str})
    assert set(agr["slide_id"]).isdisjoint({"101", "102", "103"}) and agr["slide_id"].nunique() == 3
    assert S3.verify_unit(out, log=lambda *_: None)["pass"]


def test_virchow2_coordinates_checked_against_uni2(world, tmp_path):
    m, out = run_s3(world, "virchow2", 5)
    assert m["status"] == "COMPLETE"
    ib = json.loads((out / "input_bindings.json").read_text())
    assert ib["coordinate_reference"]["slides_checked"] == 12
    u, uout = run_s3(world, "uni2", 5)
    assert S3.verify_unit(out, reference_unit=uout, log=lambda *_: None)["pass"]
    # a Virchow2 file whose coordinates differ from UNI2 -> STOP
    bad = tmp_path / "v2bad"
    shutil.copytree(world["banks"]["virchow2"], bad)
    f, c = pickle.loads((bad / "106.pkl").read_bytes())
    c[3] = {"wsi_loc": [99999, 99999]}
    (bad / "106.pkl").write_bytes(pickle.dumps((f, c), protocol=4))
    s0b = tmp_path / "s0b"
    shutil.copytree(world["s0"], s0b)
    ib0 = json.loads((s0b / "input_bindings.json").read_text())
    for rec in ib0["files"]:
        if rec["role"] == "lib_virchow2_fp32_dir":
            rec["path"] = str(bad / Path(rec["path"]).name)
            rec["sha256"] = S0.sha256_file(rec["path"])
    for d in ib0["directories"]:
        if d["role"] == "lib_virchow2_fp32_dir":
            d["path"] = str(bad)
    S0.write_json(ib0, s0b / "input_bindings.json")
    finalize_s0(s0b)
    inputs = dict(world["inputs"], lib_virchow2_fp32_dir=str(bad))
    with pytest.raises(S3.S3Error, match="coordinates differ"):
        S3.run_unit("virchow2", 100, s0b, world["heads_root"], world["f7"], tmp_path / "out", world["prosper_out"],
                    inputs=inputs, expected=(12, 9), log=lambda *_: None)
    assert (tmp_path / "out" / "FAILED.json").exists()


def test_bank_sha_and_routeb_mosaic_gates(world, tmp_path):
    mos = tmp_path / "mos"
    shutil.copytree(world["routeb_mosaic"], mos)
    c = np.load(mos / "coordinates.npy")
    c[5] = [1, 1]
    np.save(mos / "coordinates.npy", c)
    with pytest.raises(S3.S3Error, match="route-b mosaic"):
        S3.run_unit("uni2", 100, world["s0"], world["heads_root"], world["f7"], tmp_path / "o1", world["prosper_out"],
                    inputs=world["inputs"], routeb_mosaic=mos, expected=(12, 9), log=lambda *_: None)


def test_agreement_sample_deterministic_unannotated_only():
    t = pd.DataFrame({"slide_id": [str(i) for i in range(500)], "patient_annotated": [int(i % 5 == 0) for i in range(500)]})
    a = S3.agreement_sample(t, n=200)
    assert a == S3.agreement_sample(t, n=200) and len(a) == 200 == len(set(a))
    assert all(int(s) % 5 for s in a)
    assert a != S3.agreement_sample(t, n=200, seed=124)


def test_jaccard_both_empty_is_undefined():
    j, i, u = S3.jaccard(np.zeros(4, bool), np.zeros(4, bool))
    assert np.isnan(j) and i == u == 0
    assert S3.jaccard(np.array([1, 1, 0], bool), np.array([1, 0, 0], bool))[0] == 0.5
