import json, os, sys

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import r2_6_titan_direct1024_subset as T  # noqa: E402

V = T.VARIANTS
COORD_ATTRS = {"patch_size": 1024, "patch_size_level0": 1024, "name": "x", "level0_width": 9000}


def _coords(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.choice(400, size=n, replace=False)
    return np.stack([xy // 20 * 1024, xy % 20 * 1024], axis=1).astype(np.int64)


def make_fixture(tmp_path, kept, n_patches=None, seed=0, extra_manifest=None):
    """kept: {slide: {variant: row indices kept}}; manifests list (x, y) in reversed (non-file) order."""
    conch, sel, wsi = tmp_path / "conch", tmp_path / "sel", tmp_path / "wsi"
    for d in (conch, sel, wsi):
        d.mkdir()
    n_patches = n_patches or {s: 12 for s in kept}
    coords, summ, man = {}, [], {v: [] for v in V}
    for i, s in enumerate(sorted(kept)):
        c = _coords(n_patches[s], seed + i)
        f = np.random.default_rng(100 + i).standard_normal((len(c), T.FEAT_DIM)).astype(np.float32)
        with h5py.File(conch / f"{s}.h5", "w") as h:
            d = h.create_dataset("coords", data=c)
            d.attrs.update(COORD_ATTRS)
            d = h.create_dataset("features", data=f)
            d.attrs.update({"encoder": "conch_v15"})
        coords[s] = (c, f)
        (wsi / f"{s}.svs").write_bytes(b"svs")
        for v in V:
            idx = list(kept[s].get(v, []))
            summ.append({"slide_id": s, "variant": v, "n_patches": len(c), "n_kept": len(idx)})
            for j in reversed(idx):
                man[v].append((s, int(c[j, 0]), int(c[j, 1])))
    for v, rows in (extra_manifest or {}).items():
        man[v].extend(rows)
    pd.DataFrame(summ).to_csv(sel / "selection_summary.csv", index=False)
    for v in V:
        pd.DataFrame(man[v], columns=["slide_id", "x", "y"]).to_csv(sel / f"selection_{v}.csv.gz", index=False)
    rc = tmp_path / "regen_command_record.json"
    rc.write_text(json.dumps({"main_command": T.trident_command("/a/regen/wsi_dir", "/a/regen/job_dir")}))
    src = tmp_path / "src.py"
    src.write_text("# copied source\n")
    p = {"conch_dir": str(conch), "wsi_src": str(wsi), "selection_dir": str(sel), "stage2": str(tmp_path / "stage2"),
         "regen_command_record": str(rc), "regen_script": str(src), "subset_src": str(src)}
    return p, coords


KEPT = {"10": {V[0]: [0, 3, 5], V[1]: [0, 1, 2, 3, 4, 5, 6, 7, 8], V[2]: [5]},
        "11": {V[0]: [11], V[1]: [2, 11], V[2]: []},
        "12": {V[0]: list(range(12)), V[1]: list(range(10)), V[2]: [1, 2, 3, 4, 5, 6, 7, 8]}}


def test_subset_rule_records_and_symlinks(tmp_path):
    p, _ = make_fixture(tmp_path, KEPT)
    rec = T.run_subset(p, workers=2, n_expected=3)
    st = rec["variants"]
    assert st[V[0]]["n_written"] == 3 and st[V[0]]["n_excluded"] == 0
    assert st[V[0]]["ids_with_1_to_7"] == ["10", "11"] and st[V[0]]["n_with_1_to_7"] == 2
    assert st[V[1]]["ids_with_1_to_7"] == ["11"]
    assert st[V[2]]["excluded_ids"] == ["11"] and st[V[2]]["n_written"] == 2
    assert st[V[2]]["ids_with_1_to_7"] == ["10"]                       # 1 kept patch: included
    for v in V:
        D = T.variant_dirs(p["stage2"], v)
        r = json.load(open(D["coords"] / "_subset_record.json"))
        written = sorted(f.stem for f in D["feat"].glob("*.h5"))
        links = sorted(f.stem for f in D["wsi"].iterdir())
        assert written == links == sorted(s for s in KEPT if KEPT[s][v])
        assert all(os.path.islink(D["wsi"] / f"{s}.svs") for s in links)
        assert os.readlink(D["wsi"] / f"{links[0]}.svs") == str(tmp_path / "wsi" / f"{links[0]}.svs")
        ex = pd.read_csv(D["coords"] / "_excluded_slides.csv", dtype={"slide": str})
        assert list(ex.slide) == [s for s in sorted(KEPT) if not KEPT[s][v]]
        sm = pd.read_csv(D["coords"] / "_tumour_patch_summary.csv", dtype={"slide": str})
        assert list(sm.columns) == ["slide", "n_patches", "n_kept", "frac_kept"] and len(sm) == 3
        assert r["n_slides_in"] == 3 and r["min_patches"] == 1
        assert r["n_rows_written"] == sum(len(KEPT[s][v]) for s in KEPT)
        cmd = (D["root"] / "command.txt").read_text().strip()
        assert cmd == T.trident_command(str(D["wsi"]), str(D["job"])) and "--skip_errors" not in cmd
        assert json.load(open(D["root"] / "command_record.json"))["trident_command"] == cmd
        assert not T.variant_dirs(p["stage2"], v, partial=True)["root"].exists()
    assert (tmp_path / "stage2" / "stage2_record.json").exists()


def test_order_preserved_and_attrs_copied(tmp_path):
    p, src = make_fixture(tmp_path, KEPT)
    T.run_subset(p, workers=1, n_expected=3)
    for v in V:
        for s in KEPT:
            if not KEPT[s][v]:
                continue
            c, f = src[s]
            idx = sorted(KEPT[s][v])
            with h5py.File(T.variant_dirs(p["stage2"], v)["feat"] / f"{s}.h5", "r") as h:
                assert np.array_equal(h["coords"][:], c[idx]) and h["coords"].dtype == np.int64
                assert np.array_equal(h["features"][:], f[idx]) and h["features"].dtype == np.float32
                assert dict(h["coords"].attrs)["patch_size_level0"] == 1024
                assert h["features"].attrs["encoder"] == "conch_v15"


def _nothing_written(p):
    st = tmp = p["stage2"]
    return not os.path.exists(st) or not any(os.scandir(tmp))


def test_missing_coordinate_refused_before_writing(tmp_path):
    p, _ = make_fixture(tmp_path, {"10": {V[0]: [0], V[1]: [0], V[2]: [0]}},
                        extra_manifest={V[1]: [("10", 999 * 1024, 0)]})
    sm = pd.read_csv(tmp_path / "sel" / "selection_summary.csv", dtype={"slide_id": str})
    sm.loc[sm.variant == V[1], "n_kept"] = 2
    sm.to_csv(tmp_path / "sel" / "selection_summary.csv", index=False)
    with pytest.raises(T.Refuse, match="exactly once"):
        T.run_subset(p, workers=2, n_expected=1)
    assert _nothing_written(p)


def test_duplicate_coord_in_conch_refused(tmp_path):
    p, (src) = make_fixture(tmp_path, {"10": {V[0]: [0], V[1]: [0], V[2]: [0]}})
    f = tmp_path / "conch" / "10.h5"
    with h5py.File(f, "r+") as h:
        c = h["coords"][:]
        c[1] = c[0]
        h["coords"][:] = c
    with pytest.raises(T.Refuse, match="occurs 2 times"):
        T.run_subset(p, workers=1, n_expected=1)
    assert _nothing_written(p)


def test_count_mismatch_with_summary_refused(tmp_path):
    p, _ = make_fixture(tmp_path, KEPT)
    sm = pd.read_csv(tmp_path / "sel" / "selection_summary.csv", dtype={"slide_id": str})
    sm.loc[(sm.variant == V[0]) & (sm.slide_id == "12"), "n_kept"] = 11
    sm.to_csv(tmp_path / "sel" / "selection_summary.csv", index=False)
    with pytest.raises(T.Refuse, match="n_kept"):
        T.run_subset(p, workers=1, n_expected=3)
    assert _nothing_written(p)


def test_wrong_slide_universe_refused(tmp_path):
    p, _ = make_fixture(tmp_path, KEPT)
    with pytest.raises(T.Refuse, match="expected 4"):
        T.run_subset(p, workers=1, n_expected=4)


def test_zero_kept_everywhere_excluded_with_reason(tmp_path):
    p, _ = make_fixture(tmp_path, {"10": {V[0]: [1], V[1]: [1], V[2]: [1]}, "11": {}})
    rec = T.run_subset(p, workers=1, n_expected=2)
    for v in V:
        D = T.variant_dirs(p["stage2"], v)
        assert rec["variants"][v]["excluded_ids"] == ["11"]
        assert not (D["feat"] / "11.h5").exists() and not (D["wsi"] / "11.svs").exists()
        ex = pd.read_csv(D["coords"] / "_excluded_slides.csv", dtype={"slide": str})
        assert ex.reason.tolist() == [f"0 kept patches in selection_{v}"]


@pytest.mark.parametrize("which", ["final", "partial", "record"])
def test_refuse_if_exists(tmp_path, which):
    p, _ = make_fixture(tmp_path, KEPT)
    st = tmp_path / "stage2"
    st.mkdir()
    target = {"final": st / V[2], "partial": st / f".{V[1]}.partial", "record": st / "stage2_record.json"}[which]
    target.mkdir() if which != "record" else target.write_text("{}")
    with pytest.raises(SystemExit, match="REFUSING"):
        T.run_subset(p, workers=1, n_expected=3)
    assert sorted(os.listdir(st)) == [target.name]


def test_missing_wsi_refused(tmp_path):
    p, _ = make_fixture(tmp_path, KEPT)
    os.remove(tmp_path / "wsi" / "11.svs")
    with pytest.raises(T.Refuse, match="WSI"):
        T.run_subset(p, workers=1, n_expected=3)


def test_regen_command_pattern_must_match(tmp_path):
    p, _ = make_fixture(tmp_path, KEPT)
    rc = tmp_path / "regen_command_record.json"
    rc.write_text(json.dumps({"main_command": T.trident_command("/w", "/j") + " --skip_errors"}))
    with pytest.raises(SystemExit, match="pattern"):
        T.run_subset(p, workers=1, n_expected=3)


GPU_INFO = ("node1\nGPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-x)\nNVIDIA A100-SXM4-40GB, 550.1\n")


def fake_gpu_run(p, v, shape=(T.FEAT_DIM,), skip_errors=False, squeeze=False, skip=()):
    """squeeze=True emulates trident's coords.cpu().numpy().squeeze() (WSI.py:950): one-patch slide -> (2,).
    skip: slides not encoded (pruned as infeasible)."""
    D = T.variant_dirs(p["stage2"], v)
    D["slide_out"].mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for f in sorted(D["feat"].glob("*.h5")):
        if f.stem in skip:
            continue
        with h5py.File(f, "r") as c, h5py.File(D["slide_out"] / f.name, "w") as o:
            xy = c["coords"][:]
            d = o.create_dataset("coords", data=xy.squeeze() if squeeze else xy)
            d.attrs.update(dict(c["coords"].attrs))
            o["features"] = rng.standard_normal(shape).astype(np.float32)
    (D["coords"] / T.SLIDE_CONFIG).write_text(json.dumps({"skip_errors": skip_errors}))
    (D["job"] / "gpu_info.txt").write_text(GPU_INFO)
    return D


@pytest.mark.parametrize("shape", [(T.FEAT_DIM,), (1, T.FEAT_DIM)])
def test_verify_pass(tmp_path, shape):
    p, _ = make_fixture(tmp_path, KEPT)
    T.run_subset(p, workers=1, n_expected=3)
    D = fake_gpu_run(p, V[2], shape=shape)
    assert T.run_verify(p, V[2]) == 0
    r = json.load(open(D["root"] / "verify_record.json"))
    assert r["all_passed"] and r["n_expected"] == r["n_found"] == 2 and r["gpu_info"]["model"] == "NVIDIA A100-SXM4-40GB"
    with pytest.raises(SystemExit, match="never overwrites"):
        T.run_verify(p, V[2])


def test_verify_pass_one_patch_slide_squeezed_coords(tmp_path):
    p, _ = make_fixture(tmp_path, KEPT)
    T.run_subset(p, workers=1, n_expected=3)
    D = fake_gpu_run(p, V[2], squeeze=True)
    with h5py.File(D["slide_out"] / "10.h5", "r") as h:
        assert h["coords"].shape == (2,)                     # one kept patch, stored squeezed as trident does
    with h5py.File(D["slide_out"] / "12.h5", "r") as h:
        assert h["coords"].shape == (8, 2)                   # multi-patch slide unaffected by squeeze
    assert T.run_verify(p, V[2]) == 0
    assert json.load(open(D["root"] / "verify_record.json"))["all_passed"]


def _rewrite_coords(path, new):
    with h5py.File(path, "r+") as h:
        attrs = dict(h["coords"].attrs)
        del h["coords"]
        d = h.create_dataset("coords", data=new)
        d.attrs.update(attrs)


@pytest.mark.parametrize("how", ["one_patch_value", "multi_patch_flattened", "multi_patch_value"])
def test_verify_fail_squeezed_coords_still_checked(tmp_path, how):
    p, _ = make_fixture(tmp_path, KEPT)
    T.run_subset(p, workers=1, n_expected=3)
    D = fake_gpu_run(p, V[2], squeeze=True)
    if how == "one_patch_value":                             # (2,) but a different coordinate
        with h5py.File(D["slide_out"] / "10.h5", "r") as h:
            c = h["coords"][:]
        _rewrite_coords(D["slide_out"] / "10.h5", c + np.array([1024, 0]))
    else:
        with h5py.File(D["slide_out"] / "12.h5", "r") as h:
            c = h["coords"][:]
        if how == "multi_patch_flattened":                   # only the exact (1, 2) -> (2,) case is tolerated
            _rewrite_coords(D["slide_out"] / "12.h5", c.reshape(-1))
        else:
            c[-1, 1] += 1024
            _rewrite_coords(D["slide_out"] / "12.h5", c)
    assert T.run_verify(p, V[2]) == 1
    assert not (D["root"] / "verify_record.json").exists()
    failed = list(D["root"].glob("verify_failed_*.json"))
    assert len(failed) == 1
    assert any("coords differ" in x for x in json.load(open(failed[0]))["violations"])


def _break(D, how):
    out = D["slide_out"]
    first = sorted(out.glob("*.h5"))[0]
    if how == "missing":
        os.remove(first)
    elif how == "extra":
        with h5py.File(out / "99.h5", "w") as h:
            h["features"] = np.ones(T.FEAT_DIM, np.float32)
    elif how == "lock":
        (out / "10.lock").write_text("")
    elif how in ("zero", "nan", "lowstd", "shape"):
        val = {"zero": np.zeros(T.FEAT_DIM), "nan": np.full(T.FEAT_DIM, np.nan),
               "lowstd": np.full(T.FEAT_DIM, 0.5) + np.arange(T.FEAT_DIM) * 1e-5, "shape": np.ones(512)}[how]
        with h5py.File(first, "r+") as h:
            del h["features"]
            h["features"] = val.astype(np.float32)
    elif how == "coords":
        with h5py.File(first, "r+") as h:
            c = h["coords"][:]
            c[0, 0] += 1024
            h["coords"][:] = c
    elif how == "skip_errors":
        (D["coords"] / T.SLIDE_CONFIG).write_text(json.dumps({"skip_errors": True}))
    elif how == "no_config":
        os.remove(D["coords"] / T.SLIDE_CONFIG)
    elif how == "gpu_info":
        os.remove(D["job"] / "gpu_info.txt")


@pytest.mark.parametrize("how", ["missing", "extra", "lock", "zero", "nan", "lowstd", "shape", "coords",
                                 "skip_errors", "no_config", "gpu_info"])
def test_verify_fail(tmp_path, how):
    p, _ = make_fixture(tmp_path, KEPT)
    T.run_subset(p, workers=1, n_expected=3)
    D = fake_gpu_run(p, V[0])
    _break(D, how)
    assert T.run_verify(p, V[0]) == 1
    assert not (D["root"] / "verify_record.json").exists()
    failed = list(D["root"].glob("verify_failed_*.json"))
    assert len(failed) == 1 and json.load(open(failed[0]))["violations"]


def test_verify_without_gpu_output_fails(tmp_path):
    p, _ = make_fixture(tmp_path, KEPT)
    T.run_subset(p, workers=1, n_expected=3)
    assert T.run_verify(p, V[1]) == 1
    assert not (T.variant_dirs(p["stage2"], V[1])["root"] / "verify_record.json").exists()


# ---- prune-infeasible (TITAN cannot encode one-patch slides; LSF 327638230) ----
# In KEPT, b100_strict (V[2]) has slide 10 with exactly one kept patch, 11 with none (excluded), 12 with 8.

def _subset_with_stale_lock(tmp_path):
    p, _ = make_fixture(tmp_path, KEPT)
    T.run_subset(p, workers=1, n_expected=3)
    D = T.variant_dirs(p["stage2"], V[2])
    D["slide_out"].mkdir(parents=True)
    (D["slide_out"] / "10.h5.lock").write_text("")          # left behind by the crashed run
    return p, D


def test_prune_moves_symlink_and_lock_writes_records_refuses_rerun(tmp_path):
    p, D = _subset_with_stale_lock(tmp_path)
    cmd_sha = T.sha256(D["root"] / "command.txt")
    target = os.readlink(D["wsi"] / "10.svs")
    rec = T.run_prune(p, V[2])
    assert not os.path.lexists(D["wsi"] / "10.svs") and os.path.lexists(D["wsi"] / "12.svs")
    moved = D["root"] / "wsi_dir_infeasible" / "10.svs"
    assert moved.is_symlink() and os.readlink(moved) == target
    assert not (D["slide_out"] / "10.h5.lock").exists() and not T.lock_files(D["job"])
    stale = list((D["job"] / "_stale_locks").iterdir())
    assert len(stale) == 1 and stale[0].name.startswith("10.h5.lock.")
    assert (D["feat"] / "10.h5").is_file()                  # subset CONCH input untouched
    assert T.sha256(D["root"] / "command.txt") == cmd_sha
    df = pd.read_csv(D["coords"] / T.INFEASIBLE_CSV, dtype={"slide": str})
    assert df.to_dict("records") == [{"slide": "10", "n_kept": 1, "reason": "titan_single_patch_alibi_indexerror",
                                      "lsf_job": 327638230}]
    r = json.load(open(D["coords"] / T.PRUNE_RECORD))
    assert r == rec and r["infeasible_slides"] == ["10"] and r["n_moved_wsi_symlinks"] == 1
    assert r["n_moved_locks"] == 1 and r["n_written"] == 2 and r["n_titan_expected"] == 1
    assert r["tumour_patch_summary_sha256"] == T.sha256(D["coords"] / "_tumour_patch_summary.csv")
    assert r["script"]["sha256"] == T.sha256(T.__file__)
    with pytest.raises(SystemExit, match="already run"):
        T.run_prune(p, V[2])
    assert pd.read_csv(D["coords"] / T.INFEASIBLE_CSV).shape[0] == 1


def _untouched(D):
    return (os.path.lexists(D["wsi"] / "10.svs") and not (D["root"] / "wsi_dir_infeasible").exists()
            and not (D["job"] / "_stale_locks").exists() and not (D["coords"] / T.INFEASIBLE_CSV).exists()
            and not (D["coords"] / T.PRUNE_RECORD).exists())


@pytest.mark.parametrize("locked", [False, True])
def test_prune_refuses_when_output_h5_exists(tmp_path, locked):
    p, D = _subset_with_stale_lock(tmp_path)
    if not locked:
        os.remove(D["slide_out"] / "10.h5.lock")
    with h5py.File(D["slide_out"] / "10.h5", "w") as h:
        h["features"] = np.ones(T.FEAT_DIM, np.float32)
    with pytest.raises(SystemExit, match="locked partial" if locked else "finished"):
        T.run_prune(p, V[2])
    assert _untouched(D)


def test_prune_refuses_when_other_slide_locked(tmp_path):
    p, D = _subset_with_stale_lock(tmp_path)
    (D["slide_out"] / "12.h5.lock").write_text("")
    with pytest.raises(SystemExit, match="may be active"):
        T.run_prune(p, V[2])
    assert _untouched(D)


def test_prune_without_lock_and_zero_infeasible_variant(tmp_path):
    p, _ = make_fixture(tmp_path, KEPT)
    T.run_subset(p, workers=1, n_expected=3)
    D0 = T.variant_dirs(p["stage2"], V[0])                   # b100_argmax: slide 11 has one kept patch
    D1 = T.variant_dirs(p["stage2"], V[1])                   # b5: n_kept 9, 2, 10 -> nothing to prune
    before = sorted(str(x) for x in D1["root"].rglob("*"))
    assert T.run_prune(p, V[1]) is None
    assert sorted(str(x) for x in D1["root"].rglob("*")) == before
    rec = T.run_prune(p, V[0])                               # V[0] slide 11 has one kept patch, no lock
    assert rec["infeasible_slides"] == ["11"] and rec["moved_locks"] == [] and not (D0["job"] / "_stale_locks").exists()


def _pruned_run(tmp_path):
    p, D = _subset_with_stale_lock(tmp_path)
    T.run_prune(p, V[2])
    fake_gpu_run(p, V[2], squeeze=True, skip=("10",))
    return p, D


def test_verify_passes_with_pruned_set(tmp_path):
    p, D = _pruned_run(tmp_path)
    assert T.run_verify(p, V[2]) == 0
    r = json.load(open(D["root"] / "verify_record.json"))
    assert r["all_passed"] and r["n_expected"] == r["n_found"] == 1
    assert r["infeasible_slides"] == ["10"] and r["n_infeasible"] == 1
    assert r["infeasible_csv_sha256"] == T.sha256(D["coords"] / T.INFEASIBLE_CSV)
    assert r["infeasible_checks"] and all(c["ok"] for c in r["infeasible_checks"].values())
    assert set(r["infeasible_checks"]) == {"prune_record_present", "prune_record_infeasible_csv_sha256_matches",
                                           "prune_record_tumour_patch_summary_sha256_matches",
                                           "csv_set_equals_summary_n_kept_1", "csv_n_kept_all_1",
                                           "wsi_dir_equals_expected", "wsi_dir_infeasible_equals_infeasible"}


@pytest.mark.parametrize("how", ["h5", "lock"])
def test_verify_fails_if_infeasible_output_appears(tmp_path, how):
    p, D = _pruned_run(tmp_path)
    if how == "h5":
        with h5py.File(D["slide_out"] / "10.h5", "w") as h:
            h["features"] = np.random.default_rng(1).standard_normal(T.FEAT_DIM).astype(np.float32)
    else:
        (D["slide_out"] / "10.h5.lock").write_text("")
    assert T.run_verify(p, V[2]) == 1
    assert not (D["root"] / "verify_record.json").exists()
    v = json.load(open(next(D["root"].glob("verify_failed_*.json"))))["violations"]
    msg = {"h5": "h5 output exists", "lock": "lock exists"}[how] + " for an infeasible slide"
    assert any(msg in x for x in v)


def test_verify_without_infeasible_file_unchanged(tmp_path):
    p, _ = make_fixture(tmp_path, KEPT)
    T.run_subset(p, workers=1, n_expected=3)
    D = fake_gpu_run(p, V[2], squeeze=True, skip=("10",))   # one-patch slide not encoded, not pruned
    assert T.run_verify(p, V[2]) == 1
    v = json.load(open(next(D["root"].glob("verify_failed_*.json"))))
    assert v["missing"] == ["10"] and v["n_expected"] == 2 and v["n_infeasible"] == 0


def _resign(D, **kv):
    rf = D["coords"] / T.PRUNE_RECORD
    r = json.load(open(rf))
    r.update(kv)
    rf.write_text(json.dumps(r, indent=1))


@pytest.mark.parametrize("how,failed", [
    ("csv_extra_slide", {"csv_set_equals_summary_n_kept_1", "csv_n_kept_all_1",
                         "prune_record_infeasible_csv_sha256_matches", "wsi_dir_equals_expected",
                         "wsi_dir_infeasible_equals_infeasible"}),
    ("csv_extra_slide_resigned_and_moved", {"csv_set_equals_summary_n_kept_1", "csv_n_kept_all_1"}),
    ("missing_prune_record", {"prune_record_present"}),
    ("stale_csv_sha", {"prune_record_infeasible_csv_sha256_matches"}),
    ("stale_summary_sha", {"prune_record_tumour_patch_summary_sha256_matches"}),
    ("infeasible_back_in_wsi_dir", {"wsi_dir_equals_expected", "wsi_dir_infeasible_equals_infeasible"}),
    ("expected_missing_from_wsi_dir", {"wsi_dir_equals_expected"}),
])
def test_verify_rejects_unauthenticated_infeasible_list(tmp_path, how, failed):
    p, D = _subset_with_stale_lock(tmp_path)
    T.run_prune(p, V[2])
    csv_f = D["coords"] / T.INFEASIBLE_CSV
    # probe: an 8-patch slide (12) never encoded and appended to the CSV must not shrink the expected set
    fake_gpu_run(p, V[2], squeeze=True, skip=("10", "12") if how.startswith("csv_extra") else ("10",))
    if how.startswith("csv_extra"):
        with open(csv_f, "a") as fh:
            fh.write("12,8,titan_single_patch_alibi_indexerror,327638230\n")
        if how == "csv_extra_slide_resigned_and_moved":
            os.rename(D["wsi"] / "12.svs", D["root"] / "wsi_dir_infeasible" / "12.svs")
            _resign(D, infeasible_csv_sha256=T.sha256(csv_f))
    elif how == "missing_prune_record":
        os.remove(D["coords"] / T.PRUNE_RECORD)
    elif how == "stale_csv_sha":
        _resign(D, infeasible_csv_sha256="0" * 64)
    elif how == "stale_summary_sha":
        _resign(D, tumour_patch_summary_sha256="0" * 64)
    elif how == "infeasible_back_in_wsi_dir":
        os.rename(D["root"] / "wsi_dir_infeasible" / "10.svs", D["wsi"] / "10.svs")
    elif how == "expected_missing_from_wsi_dir":
        os.rename(D["wsi"] / "12.svs", tmp_path / "12.svs")
    assert T.run_verify(p, V[2]) == 1
    assert not (D["root"] / "verify_record.json").exists()
    r = json.load(open(next(D["root"].glob("verify_failed_*.json"))))
    assert {k for k, c in r["infeasible_checks"].items() if not c["ok"]} == failed
    assert any(x.startswith("infeasible authentication:") for x in r["violations"])
