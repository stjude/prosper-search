import json, os, pickle, sys

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import r2_6_titan_arm1_regen as T  # noqa: E402

N_OTHER = 9                                   # synthetic: 21 zero slides + 9 normal slides
# patch counts: the 21 have 5..25 (min 5, median 15, max 25); published non-zero slides below
OTHER_COUNTS = {"2000": 4, "2001": 15, "2002": 15, "2003": 26, "2004": 30, "2005": 6, "2006": 9, "2007": 12, "2008": 20}
CIDS = ["2000", "2001", "2003"]
A100 = "NVIDIA A100-SXM4-80GB"
SW = {"titan_hf_snapshot": {"refs_main": "abc123", "model_safetensors_blob": "blob456"},
      "trident_git": {"path": "/x/trident", "head": "deadbeef", "diff_sha256": "d1ff", "diff_bytes": 10}}


def gpu_txt(host="noderome300", model=A100, n=1, driver="550.127.05"):
    return "\n".join([host] + [f"GPU {i}: {model} (UUID: GPU-{i:04d})" for i in range(n)]
                     + [f"{model}, {driver}"] * n) + "\n"


def write_gpu(job, **kw):
    os.makedirs(job, exist_ok=True)
    with open(os.path.join(job, "gpu_info.txt"), "w") as f:
        f.write(gpu_txt(**kw))


def wh5(path, feats, coords=None):
    with h5py.File(path, "w") as h:
        h["features"] = np.asarray(feats, dtype=np.float32)
        if coords is not None:
            h["coords"] = coords
            h["coords"].attrs["patch_size_level0"] = 1024


def world(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    zero = [str(1000 + i) for i in range(T.N_REGEN)]
    other = [str(2000 + i) for i in range(N_OTHER)]
    ids = sorted(zero + other)
    pub = tmp_path / "pub"
    sf, pf, wsi = pub / "slide_features_titan", pub / "features_conch_v15", tmp_path / "wsi"
    for d in (sf, pf, wsi):
        d.mkdir(parents=True)
    X = {}
    for s in ids:
        X[s] = np.zeros(768) if s in zero else rng.normal(size=768).astype(np.float32)
        if s == "2003":                           # a close (different-patient) published pair -> small published NN
            X[s] = (X["2002"] + 0.3 * rng.normal(size=768)).astype(np.float32)
        wh5(sf / f"{s}.h5", X[s])
        n = 5 + zero.index(s) if s in zero else OTHER_COUNTS[s]
        wh5(pf / f"{s}.h5", rng.normal(size=(n, 768)), np.arange(2 * n).reshape(n, 2) * 1024)
        (wsi / f"{s}.svs").write_bytes(b"svs")
    # patient: 1000/1001 share a patient, 2000/2001 share a patient
    pat = {s: ("PA" if s in ("1000", "1001") else "PB" if s in ("2000", "2001") else "P" + s) for s in ids}
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"Tumor.group": ["G" + s[-1] for s in ids], "Slide.Scan.File": [int(s) for s in ids],
                  "Sample.ID": ids, "UID.Subject": [pat[s] for s in ids]}).to_csv(meta, index=False)
    M, n = T._mask_same_patient(T._distance_frame(ids, np.stack([X[s].astype(np.float64) for s in ids]),
                                                  "cityblock"), meta)
    mpath = tmp_path / "pub_matrix.pkl"
    with open(mpath, "wb") as f:
        pickle.dump(M, f)
    clique = tmp_path / "clique.csv"
    pd.DataFrame({"slide": zero}).to_csv(clique, index=False)
    monkeypatch.setattr(T, "N_ALL", len(ids))
    monkeypatch.setattr(T, "N_MASKED", n)
    monkeypatch.setattr(T, "software_provenance", lambda: json.loads(json.dumps(SW)))
    p = {"published_slide_features": str(sf), "published_patch_features": str(pf), "published_matrix": str(mpath),
         "metadata": str(meta), "clique_csv": str(clique), "wsi_src": str(wsi), "outdir": str(tmp_path / "out")}
    return p, zero, ids, M


def shuffled_other(p):
    """Regenerated stand-in: a permuted published non-zero vector (norm inside the published range)."""
    others = sorted(f for f in os.listdir(p["published_slide_features"]) if f.startswith("2"))
    return lambda s, rng: rng.permutation(T.read_vec(os.path.join(p["published_slide_features"],
                                                                   others[int(s) % len(others)])))


def write_regen(p, zero, fn, job=None):
    """Regenerated h5 in trident layout: features + coords/attrs copied from the published CONCH h5."""
    job = job or os.path.join(p["outdir"], "job_dir")
    d = os.path.join(job, T.COORDS, "slide_features_titan")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(7)
    for s in zero:
        v = fn(s, rng)
        if v is not None:
            with h5py.File(os.path.join(p["published_patch_features"], f"{s}.h5"), "r") as c:
                coords, attrs = c["coords"][:], dict(c["coords"].attrs)
            with h5py.File(os.path.join(d, f"{s}.h5"), "w") as h:
                h["features"] = np.asarray(v, dtype=np.float32)
                h["coords"] = coords
                h["coords"].attrs.update(attrs)


def write_config(job, skip_errors=False):
    os.makedirs(os.path.join(job, T.COORDS), exist_ok=True)
    with open(os.path.join(job, T.COORDS, T.SLIDE_CONFIG), "w") as f:
        json.dump({"skip_errors": skip_errors, "coords_dir": T.COORDS}, f)


def ready(p, control_record="default"):
    """prepare + trident config + main gpu_info + (by default) a consistent controls manifest/gpu_info/record,
    i.e. the state expected before rebuild."""
    assert T.prepare(p) == 0
    write_config(os.path.join(p["outdir"], "job_dir"))
    write_gpu(os.path.join(p["outdir"], "job_dir"))
    C = T.ctrl_paths(p["outdir"])
    if control_record == "default":
        write_gpu(str(C["job"]), host="noderome301")
        with open(C["manifest"], "w") as f:
            json.dump({"slide_ids": CIDS}, f)
        control_record = {"all_passed": True, "control_slide_ids": CIDS,
                          "controls_manifest_sha256": T.sha256(C["manifest"]),
                          "gpu_info": T.parse_gpu_info(C["job"] / "gpu_info.txt"), "provenance": T.software_provenance()}
    if control_record is not None:
        os.makedirs(C["root"], exist_ok=True)
        with open(C["record"], "w") as f:
            json.dump(control_record, f)


def no_rebuild_outputs(p):
    return not any(f.startswith(("_" + T.BASE, T.BASE, "reconciliation", "regenerated_slide_summary"))
                   for f in outputs(p))


def outputs(p):
    return sorted(os.listdir(p["outdir"]))


def test_prepare_copies_and_manifest(tmp_path, monkeypatch):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    assert T.prepare(p) == 0
    out = tmp_path / "out"
    fd = out / "job_dir" / T.COORDS / "features_conch_v15"
    assert sorted(f.stem for f in fd.glob("*.h5")) == zero
    assert not any(f.is_symlink() for f in fd.iterdir())
    assert all((out / "wsi_dir" / f"{s}.svs").is_symlink() for s in zero)
    m = json.load(open(out / "manifest.json"))
    assert m["slide_ids"] == zero and not m["trident_command_executed_by_this_script"]
    assert f"--wsi_dir {out / 'wsi_dir'} --job_dir {out / 'job_dir'}" in m["trident_command"]
    assert m["slides"][zero[0]]["patch_features_sha256"] == T.sha256(fd / f"{zero[0]}.h5")
    assert json.load(open(out / "prepare_record.json"))["n_slides"] == 21
    with pytest.raises(SystemExit):
        T.prepare(p)                                              # never overwrites


def test_prepare_refuses_nonzero_published(tmp_path, monkeypatch):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    wh5(os.path.join(p["published_slide_features"], f"{zero[3]}.h5"), np.ones(768))
    with pytest.raises(SystemExit, match="not an all-zero"):
        T.prepare(p)
    assert not os.path.exists(p["outdir"])


def test_rebuild_passes_and_preserves_published_block(tmp_path, monkeypatch):
    p, zero, ids, M = world(tmp_path, monkeypatch)
    ready(p)
    write_regen(p, zero, shuffled_other(p))
    assert T.rebuild(p) == 0
    rec = json.load(open(os.path.join(p["outdir"], "reconciliation.json")))
    assert rec["all_passed"] and len(rec["per_slide"]) == 21
    new = pd.read_pickle(os.path.join(p["outdir"], f"{T.BASE}_no_same_patient.pkl"))
    keep = [s for s in ids if s not in zero]
    assert np.array_equal(new.loc[keep, keep].to_numpy(), M.loc[keep, keep].to_numpy())
    assert new.at["1000", "1001"] == T.SENTINEL                  # same-patient regenerated pair masked
    assert (new.loc[zero, zero].to_numpy()[~np.eye(21, dtype=bool) & (new.loc[zero, zero].to_numpy() < 1e5)] > 0).all()


@pytest.mark.parametrize("fault", ["missing", "zero", "nan", "shape"])
def test_rebuild_refuses_bad_regenerated(tmp_path, monkeypatch, fault):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    ready(p)
    bad = zero[5]

    def fn(s, rng):
        v = rng.normal(size=768)
        if s != bad:
            return v
        return {"missing": None, "zero": np.zeros(768), "nan": np.full(768, np.nan), "shape": np.ones(512)}[fault]
    write_regen(p, zero, fn)
    with pytest.raises(SystemExit, match=bad):
        T.rebuild(p)
    assert not os.path.exists(os.path.join(p["outdir"], f"{T.BASE}_no_same_patient.pkl"))


def test_rebuild_dry_run(tmp_path, monkeypatch):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    ready(p)
    write_regen(p, zero[:-1], lambda s, rng: rng.normal(size=768))
    assert T.rebuild(p, dry_run=True) == 1
    write_regen(p, zero[-1:], lambda s, rng: rng.normal(size=768))
    assert T.rebuild(p, dry_run=True) == 0
    assert not os.path.exists(os.path.join(p["outdir"], "reconciliation.json"))


# ---------------- review fixes 10-04 ----------------
def test_published_world_has_small_nn(tmp_path, monkeypatch):
    p, zero, ids, M = world(tmp_path, monkeypatch)
    keep = np.array([s not in zero for s in ids])
    nn = T.published_nn(M.to_numpy(), keep)
    assert nn.min() < 300 < np.percentile(nn, 50)


def test_rebuild_failed_check_candidate_then_FAILED_naming(tmp_path, monkeypatch):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    ready(p)
    sh = shuffled_other(p)
    write_regen(p, zero, lambda s, rng: np.full(768, 0.8) if s == zero[2] else sh(s, rng))   # constant vector
    assert T.rebuild(p) == 1
    names = outputs(p)
    for f in (f"{T.BASE}_no_same_patient.pkl", f"_{T.BASE}_record.json", "regenerated_slide_summary.csv",
              "reconciliation.json"):
        assert f not in names and f + ".candidate" not in names and f + ".FAILED" in names
    rec = json.load(open(os.path.join(p["outdir"], "reconciliation.json.FAILED")))
    assert not rec["all_passed"]
    assert rec["checks"][f"regenerated_std_ge_{T.MIN_STD}"]["result"] == "FAIL"
    assert rec["checks"][f"regenerated_std_ge_{T.MIN_STD}"]["failing"] == [zero[2]]
    with pytest.raises(SystemExit, match="FAILED exists"):                     # never overwrites failure evidence
        T.rebuild(p)


def test_rebuild_rejects_duplicate_regenerations(tmp_path, monkeypatch):
    p, zero, ids, M = world(tmp_path, monkeypatch)
    ready(p)
    sh = shuffled_other(p)
    dup = {}

    def fn(s, rng):
        v = sh(s, rng)
        if s in (zero[3], zero[4]):                                            # different patients
            dup.setdefault("v", v)
            return dup["v"] + (1e-3 if s == zero[4] else 0)
        return v
    write_regen(p, zero, fn)
    assert T.rebuild(p) == 1
    rec = json.load(open(os.path.join(p["outdir"], "reconciliation.json.FAILED")))
    c = rec["checks"]["no_near_duplicate_regenerated_pairs"]
    assert c["result"] == "FAIL" and c["duplicates"][0][:2] == [zero[3], zero[4]]
    assert rec["checks"]["regenerated_nn_ge_published_min_nn"]["result"] == "FAIL"
    nn = T.published_nn(M.to_numpy(), np.array([s not in zero for s in ids]))
    assert rec["published_nn_stats"]["min"] == nn.min() and rec["published_nn_stats"]["p5"] == np.percentile(nn, 5)
    assert not os.path.exists(os.path.join(p["outdir"], f"{T.BASE}_no_same_patient.pkl"))


def test_rebuild_rejects_all_identical_regenerations(tmp_path, monkeypatch):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    ready(p)
    v = shuffled_other(p)("0", np.random.default_rng(3))
    write_regen(p, zero, lambda s, rng: v)
    assert T.rebuild(p) == 1
    rec = json.load(open(os.path.join(p["outdir"], "reconciliation.json.FAILED")))
    assert rec["checks"]["regenerated_median_pairwise_ge_published_p5_nn"]["result"] == "FAIL"


@pytest.mark.parametrize("what", ["coords", "attr"])
def test_rebuild_rejects_coords_mismatch(tmp_path, monkeypatch, what):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    ready(p)
    write_regen(p, zero, shuffled_other(p))
    with h5py.File(os.path.join(p["outdir"], "job_dir", T.COORDS, "slide_features_titan", f"{zero[7]}.h5"), "a") as h:
        if what == "coords":
            h["coords"][0, 0] += 1024
        else:
            h["coords"].attrs["patch_size_level0"] = 512
    with pytest.raises(SystemExit, match=zero[7]):
        T.rebuild(p)
    assert outputs(p) == ["controls", "job_dir", "manifest.json", "prepare_record.json", "wsi_dir"]


def test_rebuild_rejects_lock_and_skip_errors_and_staged_change(tmp_path, monkeypatch):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    ready(p)
    write_regen(p, zero, shuffled_other(p))
    job = os.path.join(p["outdir"], "job_dir")
    lock = os.path.join(job, T.COORDS, "slide_features_titan", f"{zero[1]}.h5.lock")
    open(lock, "w").close()
    with pytest.raises(SystemExit, match="lock"):
        T.rebuild(p)
    os.remove(lock)
    write_config(job, skip_errors=True)
    with pytest.raises(SystemExit, match="skip_errors"):
        T.rebuild(p)
    write_config(job)
    with h5py.File(os.path.join(job, T.COORDS, "features_conch_v15", f"{zero[0]}.h5"), "a") as h:
        h.attrs["tamper"] = 1
    with pytest.raises(SystemExit, match="staged CONCH copies differ"):
        T.rebuild(p)
    assert outputs(p) == ["controls", "job_dir", "manifest.json", "prepare_record.json", "wsi_dir"]


@pytest.mark.parametrize("crec", [None, {"all_passed": False, "control_slide_ids": ["2000"]},
                                  {"control_slide_ids": ["2000"]}, {"all_passed": "true"},
                                  {"all_passed": True, "control_slide_ids": ["1003"]}])
def test_rebuild_blocked_by_control_record(tmp_path, monkeypatch, crec):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    ready(p, control_record=crec)
    write_regen(p, zero, shuffled_other(p))
    with pytest.raises(SystemExit, match="control"):
        T.rebuild(p)
    assert not any(f.startswith(("_" + T.BASE, T.BASE, "reconciliation")) for f in outputs(p))


def test_rebuild_refuses_extra_embedding_in_main_job(tmp_path, monkeypatch):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    ready(p)
    write_regen(p, zero + ["2001"], shuffled_other(p))                         # a control vector in the main dir
    with pytest.raises(SystemExit, match="unexpected"):
        T.rebuild(p)


def test_select_controls_rule():
    pub = {"a1": 4, "a0": 6, "b": 15, "b0": 15, "c": 26, "d": 24, "e": 9}
    ch = T.select_controls(pub, [5, 25, 15, 7, 20])
    assert [(c["target"], c["slide"]) for c in ch] == [("min", "a0"), ("median", "b"), ("max", "c")]
    assert ch[1]["target_patch_count"] == 15
    ch = T.select_controls({"x": 10, "y": 11, "z": 30}, [10, 10, 10])         # each slide chosen once
    assert [c["slide"] for c in ch] == ["x", "y", "z"]


def stage(p):
    ready(p, control_record=None)
    assert T.stage_controls(p) == 0
    return json.load(open(os.path.join(p["outdir"], "controls", "manifest.json")))


def test_stage_controls_and_command(tmp_path, monkeypatch, capsys):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    m = stage(p)
    assert m["slide_ids"] == ["2000", "2001", "2003"]                         # ties -> lowest lexical id
    assert [m["slides"][s]["n_patches"] for s in m["slide_ids"]] == [4, 15, 26]
    C = T.ctrl_paths(p["outdir"])
    assert sorted(f.stem for f in C["feat"].glob("*.h5")) == m["slide_ids"]
    assert not any(f.is_symlink() for f in C["feat"].iterdir())
    assert all((C["wsi"] / f"{s}.svs").is_symlink() for s in m["slide_ids"])
    with pytest.raises(SystemExit):
        T.stage_controls(p)                                                   # never overwrites
    # frozen main manifest carries the legacy command text; `command` prints the corrected one
    mp = os.path.join(p["outdir"], "manifest.json")
    mm = json.load(open(mp))
    mm["trident_command"] = T.legacy_trident_command(mm["wsi_dir"], mm["job_dir"])
    json.dump(mm, open(mp, "w"), indent=1)
    capsys.readouterr()
    assert T.command(p) == 0
    printed = capsys.readouterr().out.splitlines()
    rec = json.load(open(os.path.join(p["outdir"], "command_record.json")))
    assert printed == ["# main (21 slides)", rec["main_command"], "# controls (3 slides)", rec["controls_command"]]
    for c in (rec["main_command"], rec["controls_command"]):
        assert "--skip_errors" not in c and "HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 " in c and c.endswith("--gpu 0")
    assert f"--job_dir {C['job']}" in rec["controls_command"]
    assert rec["manifest_sha256"] == T.sha256(mp) and "--skip_errors" in rec["frozen_manifest_command"]
    assert T.command(p) == 0                                                  # idempotent
    mm["trident_command"] = "something else"
    json.dump(mm, open(mp, "w"), indent=1)
    with pytest.raises(SystemExit):
        T.command(p)


@pytest.mark.parametrize("case,ok", [("same", True), ("tiny", True), ("drift", False), ("missing", False),
                                     ("lock", False)])
def test_check_controls_rule(tmp_path, monkeypatch, case, ok):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    m = stage(p)
    C = T.ctrl_paths(p["outdir"])
    pubv = lambda s: T.read_vec(os.path.join(p["published_slide_features"], f"{s}.h5"))

    def fn(s, rng):
        if case == "missing" and s == "2003":
            return None
        v = pubv(s)
        return {"tiny": v + 1e-4 * rng.normal(size=768), "drift": v + 0.1 * rng.normal(size=768)}.get(case, v)
    write_regen(p, m["slide_ids"], fn, job=str(C["job"]))
    write_config(str(C["job"]))
    write_gpu(str(C["job"]), host="noderome301")
    if case == "lock":
        open(C["regen"] / "2001.h5.lock", "w").close()
    assert T.check_controls(p) == (0 if ok else 1)
    rec = json.load(open(C["record"]))
    assert rec["all_passed"] is ok and rec["control_slide_ids"] == m["slide_ids"]
    assert rec["gpu_model"] == A100 and rec["gpu_hostname"] == "noderome301"
    assert rec["integrity"]["staged_and_published_sha256_match_manifest"] is True
    if case in ("same", "tiny", "drift"):
        for r in rec["per_control"]:
            assert r["passed"] is ok and r["cosine"] > 0.99
            assert (r["l1_diff"] <= 0.01 * r["published_l1_norm"] and r["l1_diff"] < 0.05 * r["published_nn_distance"]) is ok
    with pytest.raises(SystemExit):
        T.check_controls(p)                                                   # never overwrites
    # rebuild gate consumes this record
    write_regen(p, zero, shuffled_other(p))
    if ok:
        assert T.rebuild(p) == 0
        rr = json.load(open(os.path.join(p["outdir"], "reconciliation.json")))
        assert rr["all_passed"] and rr["environment_consistency"]["gpu_model"] == A100
        assert rr["environment_consistency"]["software_equal_controls_and_rebuild_time"]["trident_git.head"] == "deadbeef"
    else:
        with pytest.raises(SystemExit, match="all_passed is not true"):
            T.rebuild(p)


# ---------------- review fixes D1-D3 (10-04) ----------------
def test_parse_gpu_info_rule(tmp_path):
    f = tmp_path / "gpu_info.txt"
    f.write_text(gpu_txt(n=2))
    g = T.parse_gpu_info(f)
    assert g["hostname"] == "noderome300" and g["model"] == A100 and g["models"] == [A100, A100]
    assert g["driver_versions"] == ["550.127.05"] and g["file_sha256"] == T.sha256(f)
    lines = gpu_txt().splitlines()
    for bad in ([], lines[1:], lines[:2], [lines[0], lines[2]], lines + ["junk"], [lines[0], lines[2], lines[1]],
                [lines[0], lines[1], "NVIDIA H100, 550"],
                [lines[0], lines[1], "GPU 1: NVIDIA H100 (UUID: GPU-1)", lines[2], "NVIDIA H100, 550"]):
        f.write_text("\n".join(bad))
        with pytest.raises(SystemExit, match="malformed"):
            T.parse_gpu_info(f)
    with pytest.raises(SystemExit, match="missing"):
        T.parse_gpu_info(tmp_path / "nope.txt")


@pytest.mark.parametrize("which", ["main", "controls"])
def test_rebuild_refuses_missing_gpu_info(tmp_path, monkeypatch, which):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    ready(p)
    write_regen(p, zero, shuffled_other(p))
    job = os.path.join(p["outdir"], "job_dir") if which == "main" else str(T.ctrl_paths(p["outdir"])["job"])
    os.remove(os.path.join(job, "gpu_info.txt"))
    with pytest.raises(SystemExit, match="gpu_info.txt missing"):
        T.rebuild(p)
    assert no_rebuild_outputs(p)


def test_rebuild_refuses_gpu_model_mismatch(tmp_path, monkeypatch):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    ready(p)
    write_regen(p, zero, shuffled_other(p))
    write_gpu(os.path.join(p["outdir"], "job_dir"), model="NVIDIA H100 80GB HBM3")
    with pytest.raises(SystemExit, match="GPU model differs"):
        T.rebuild(p)
    assert no_rebuild_outputs(p)


@pytest.mark.parametrize("key", ["trident_git.head", "trident_git.diff_sha256", "titan_hf_snapshot.refs_main",
                                 "titan_hf_snapshot.model_safetensors_blob", "missing_in_record", "controls_gpu_file"])
def test_rebuild_refuses_environment_provenance_mismatch(tmp_path, monkeypatch, key):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    ready(p)
    write_regen(p, zero, shuffled_other(p))
    C = T.ctrl_paths(p["outdir"])
    if key == "missing_in_record":
        rec = json.load(open(C["record"]))
        del rec["provenance"]["trident_git"]["head"]
        json.dump(rec, open(C["record"], "w"))
    elif key == "controls_gpu_file":                     # same model, but not the file check-controls recorded
        write_gpu(str(C["job"]), host="other-node")
        key = "gpu_info"
    else:
        a, b = key.split(".")
        sw = json.loads(json.dumps(SW))
        sw[a][b] = "changed"
        monkeypatch.setattr(T, "software_provenance", lambda: sw)
    with pytest.raises(SystemExit, match=key.replace("missing_in_record", "trident_git.head")):
        T.rebuild(p)
    assert no_rebuild_outputs(p)


def test_rebuild_refuses_control_manifest_hash_mismatch(tmp_path, monkeypatch):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    ready(p)
    write_regen(p, zero, shuffled_other(p))
    C = T.ctrl_paths(p["outdir"])
    with open(C["manifest"], "a") as f:
        f.write(" ")
    with pytest.raises(SystemExit, match="controls_manifest_sha256"):
        T.rebuild(p)
    os.remove(C["manifest"])
    with pytest.raises(SystemExit, match="manifest.json missing"):
        T.rebuild(p)
    assert no_rebuild_outputs(p)


def test_rebuild_exception_after_candidate_write_leaves_FAILED(tmp_path, monkeypatch):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    ready(p)
    write_regen(p, zero, shuffled_other(p))

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(T, "published_nn", boom)       # called after pkl/record/summary candidates are written
    with pytest.raises(RuntimeError, match="boom"):
        T.rebuild(p)
    names = outputs(p)
    finals = [f"{T.BASE}_no_same_patient.pkl", f"_{T.BASE}_record.json", "regenerated_slide_summary.csv",
              "reconciliation.json"]
    assert not any(f in names or f + ".candidate" in names for f in finals)
    assert all(f + ".FAILED" in names for f in finals[:3])
    with pytest.raises(SystemExit, match="FAILED exists"):
        T.rebuild(p)


@pytest.mark.parametrize("what", ["staged", "published", "gpu_missing"])
def test_check_controls_refuses_input_hash_mismatch(tmp_path, monkeypatch, what):
    p, zero, _, _ = world(tmp_path, monkeypatch)
    m = stage(p)
    C = T.ctrl_paths(p["outdir"])
    write_regen(p, m["slide_ids"], lambda s, rng: T.read_vec(os.path.join(p["published_slide_features"], f"{s}.h5")),
                job=str(C["job"]))
    write_config(str(C["job"]))
    write_gpu(str(C["job"]))
    if what == "staged":
        with h5py.File(C["feat"] / "2001.h5", "a") as h:
            h.attrs["tamper"] = 1
    elif what == "published":
        with h5py.File(os.path.join(p["published_slide_features"], "2003.h5"), "a") as h:
            h.attrs["tamper"] = 1
    else:
        os.remove(C["job"] / "gpu_info.txt")
    with pytest.raises(SystemExit, match={"staged": "2001: staged CONCH copy",
                                          "published": "2003: published slide embedding",
                                          "gpu_missing": "gpu_info.txt missing"}[what]):
        T.check_controls(p)
    assert not C["record"].exists()
