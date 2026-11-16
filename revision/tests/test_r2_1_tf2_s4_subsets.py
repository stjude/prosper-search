"""Synthetic-fixture tests for scripts/r2_1_tf2_s4_subsets.py (R2.1-TF2 S4, index-based design).

Reuses the S3 fixture world (mocked banks/heads). Covers: Option C version routing in the
eligibility table (analyst addendum schema; queries of different folds get different versions;
unannotated -> v_full), the min-tiles parameter (production 1 keeps 1..7-tile masks with a warning;
8 reproduces route-b), population-only index lists, query arrays equal their own version's arrays,
the load-time subset is byte-identical to the written subset of r2_1_filter_features_cli, the
storage gate (STOP before writing) and TITAN staging. Run with
    python3 -m pytest -p no:cacheprovider --basetemp=<tmp> tests/test_r2_1_tf2_s4_subsets.py
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_r2_1_tf2_s3_masks as T3  # noqa: E402

S3 = T3.S3
S4 = T3._load("r2_1_tf2_s4_subsets", T3.REPO / "scripts" / "r2_1_tf2_s4_subsets.py")
S0 = S3.S0
QUIET = dict(log=lambda *_: None)
PASS = {"status": "PASS", "projected_total_gb": 0.0}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = T3.build_world(tmp_path_factory.mktemp("s4world"))
    S3.AGREEMENT_SAMPLE = 3
    for enc in ("uni2", "virchow2"):
        for b in (100, 5):
            T3.run_s3(w, enc, b)
    w["s3_root"] = w["run"] / "s3"
    return w


def _counts(world, enc="uni2", b=100):
    return pd.read_csv(S3.unit_dir(world["s3_root"], enc, b) / "mask_counts.csv.gz", dtype={"slide_id": str})


@pytest.mark.parametrize("min_tiles", [1, 8])
def test_eligibility_option_c_routing_and_min_tiles(world, min_tiles):
    el = S4.eligibility(S3.unit_dir(world["s3_root"], "uni2", 100), world["s0"], "uni2", 100,
                        inputs=world["inputs"], min_tiles=min_tiles)
    assert list(el.columns) == S4.ELIGIBILITY_COLUMNS and len(el) == 2 * 9
    both = el[el["arm"] == "both_sided"].set_index("query")
    qo = el[el["arm"] == "query_only"].set_index("query")
    assert both.loc["101", "library_version"] == "v_fold0" and both.loc["103", "library_version"] == "v_fold1"
    assert both.loc["105", "library_version"] == "v_full" and both.loc["105", "head_id"] == "uni2/b100/full"
    assert (qo["library_version"] == "unfiltered").all() and (qo["n_eligible"] == 9).all()
    counts = _counts(world)
    for q, ver in (("101", "fold0"), ("103", "fold1"), ("105", "full")):
        c = counts[(counts["head_slot"] == ver) & counts["slide_id"].isin(T3.POP)].set_index("slide_id")
        expect = [s for s in T3.POP if c.loc[s, "retained_tiles"] >= min_tiles]
        assert both.loc[q, "eligible_reference_ids"].split(";") == expect
        assert both.loc[q, "query_returned"] == int(c.loc[q, "retained_tiles"] >= min_tiles)
    c = counts[(counts["head_slot"] == "full")].set_index("slide_id")
    n104 = int(c.loc["104", "retained_tiles"])          # slide 104 has 6 tiles in total
    assert (("104" in both.loc["105", "eligible_reference_ids"].split(";")) == (n104 >= min_tiles))
    qa = pd.read_csv(world["s0"] / "query_assignment.csv", dtype={"slide_id": str}).set_index("slide_id")
    assert (both["head_slot"] == qa.loc[both.index, "head_slot"]).all()
    assert (both["library_version"] == qa.loc[both.index, "library_version_both_sided"]).all()
    assert (qo["library_version"] == qa.loc[qo.index, "library_version_query_only"]).all()
    assert (el.groupby(["arm", "library_version"])["eligible_reference_ids"].nunique() == 1).all()
    for r in el.itertuples():
        ids = r.eligible_reference_ids.split(";")
        assert r.eligible_ids_sha256 == hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
        assert r.n_eligible == len(ids) and set(ids) <= set(T3.POP)


def test_run_unit_index_files_population_only(world, tmp_path):
    out = tmp_path / "index" / "uni2" / "b100"
    m = S4.run_unit("uni2", 100, world["s3_root"], world["s0"], out, world["prosper_out"], inputs=world["inputs"],
                    projection=PASS, **QUIET)
    assert m["status"] == "COMPLETE" and set(m["sets"]) == {f"v_{s}" for s in S3.SLOTS} | {"queries"}
    assert not list(out.glob("**/*.pkl"))                 # no feature copies
    udir = S3.unit_dir(world["s3_root"], "uni2", 100)
    off = pd.read_csv(udir / "slide_offsets.csv", dtype={"slide_id": str}).set_index("slide_id")
    warned = 0
    for slot in S3.SLOTS:
        arr = S4.load_index(out / f"v_{slot}.npz", m["sets"][f"v_{slot}"]["npz_sha256"])
        assert set(arr) <= set(T3.POP)                    # population-only libraries (Q3)
        mask = np.load(udir / f"{slot}_mask.npy")
        summ = pd.read_csv(out / f"_summary_v_{slot}.csv", dtype={"slide_id": str}, keep_default_na=False).set_index("slide_id")
        for s in T3.POP:
            keep = mask[off.loc[s, "tile_start"]:off.loc[s, "tile_stop"]]
            if keep.sum() > 0:                            # min_tiles 1: every nonempty mask written
                assert np.array_equal(arr[s], np.flatnonzero(keep))
                warned += int(summ.loc[s, "warning"] != "")
                assert (summ.loc[s, "warning"] != "") == (keep.sum() < 8)
            else:
                assert s not in arr and summ.loc[s, "reason"] == "zero_retained_tiles"
    assert warned > 0                                     # 104 (6 tiles) is kept with a warning
    q = S4.load_index(out / "queries.npz")
    qa = pd.read_csv(world["s0"] / "query_assignment.csv", dtype={"slide_id": str}).set_index("slide_id")
    for s, p in q.items():
        assert np.array_equal(p, S4.load_index(out / f"v_{qa.loc[s, 'head_slot']}.npz")[s])
    assert S4.verify_unit(out, **QUIET)["pass"]


def test_load_time_subset_byte_identical_to_written_subset(world):
    ff = S4.ff
    bank = world["banks"]["uni2"]
    for s in ("101", "104", "108"):
        feats, coords, _ = ff.load_feature_pickle(bank / f"{s}.pkl")
        rng = np.random.default_rng(int(s))
        keep = rng.random(len(coords)) < 0.6
        keep[0] = True
        written = ff.filtered_bytes(feats, coords, keep)
        sub = S4.load_time_subset(bank / f"{s}.pkl", np.flatnonzero(keep), expected_rows=len(coords))
        assert pickle.dumps(sub, protocol=ff.PICKLE_PROTOCOL) == written
    with pytest.raises(S4.S4Error):
        S4.load_time_subset(bank / "101.pkl", np.array([3, 2]))      # unsorted positions refused


def test_storage_gate_stops_before_writing(world, tmp_path):
    proj = S4.project(world["s3_root"], world["s0"], world["prosper_out"], inputs=world["inputs"], bound=100, **QUIET)
    assert proj["status"] == "STOP"
    assert {u["basis"].split(":")[0].split(" ")[0] for u in proj["units"]} >= {"exact_from_s3_mask_counts", "estimate"}
    out = tmp_path / "i" / "uni2" / "b5"
    with pytest.raises(S4.S4Error, match="STOP: projected incremental storage"):
        S4.run_unit("uni2", 5, world["s3_root"], world["s0"], out, world["prosper_out"], inputs=world["inputs"],
                    bound=100, **QUIET)
    assert (out / "FAILED.json").exists() and not list(out.glob("*.npz"))
    assert S4.project(world["s3_root"], world["s0"], world["prosper_out"], inputs=world["inputs"], **QUIET)["status"] == "PASS"


def test_titan_stage_masks_h5(tmp_path):
    import h5py
    src = tmp_path / "conch"
    src.mkdir()
    with h5py.File(src / "201.h5", "w") as h:
        d = h.create_dataset("coords", data=np.arange(20, dtype=np.int64).reshape(10, 2))
        d.attrs["patch_size_level0"] = 1024
        h.create_dataset("features", data=np.arange(7680, dtype=np.float32).reshape(10, 768))
    with h5py.File(src / "202.h5", "w") as h:
        h.create_dataset("coords", data=np.zeros((3, 2), np.int64))
        h.create_dataset("features", data=np.zeros((3, 768), np.float32))
    keep = {"201": np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], bool), "202": np.zeros(3, bool)}
    rec = S4.titan_stage("v_full", ["201", "202"], keep, src, tmp_path / "wsi", tmp_path / "stage")
    assert rec["not_staged_zero_tiles"] == ["202"] and rec["staged"][0]["tiles"] == 3
    T = S4.titan_pins()
    with h5py.File(tmp_path / "stage" / "job_dir" / T.COORDS / "features_conch_v15" / "201.h5", "r") as h:
        assert h["coords"][:].tolist() == [[0, 1], [4, 5], [18, 19]] and h["coords"].attrs["patch_size_level0"] == 1024
    assert "--slide_encoder titan" in rec["trident_command"] and "--skip_errors" not in rec["trident_command"]
