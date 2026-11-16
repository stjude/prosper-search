#!/usr/bin/env python3
"""R2.1-TF2 S4-TITAN operations: stage / F6 control / verify / delete for the TITAN slide-embedding
sets of s4/jobspec_titan_003/titan_spec.json (operations release 006, written by the operator because
the spec names `titan-stage` but the S4 script exposes only the `titan_stage` function).

Binding: specification candidate 003 §3.5, §4 S4, §7 F6/F7(iii) + 004 R1 + 006 P1; titan_spec.json
"staging_rule" (Q-T, [internal record] of release 002); operations_release_006/BRIEF.md.
No scientific choice is made here: masks come from the COMPLETE S3 conch_v15 units (mask == prediction
== 1, re-checked), min-tiles = 1 (zero retained tiles = not staged), the F6 slides are drawn with the
release's pilot sampler (`r2_1_tf2_s5_jobspec.pilot_queries`, PCG64 seed 123, n = 10), the F6
tolerance is the specification's (max rel 1e-6), and the trident command is the pinned
`r2_6_titan_arm1_regen.trident_command`, checked equal to the spec's text.

Subcommands (all write only under $PROSPER_OUT/r2_1_filter_tf2_20261016/s4/titan/)
  stage      one set (budget x {queries, v_<slot>}): S4.titan_stage + stage_manifest.json
             (per slide: head slot/id, library version, source/staged sha256; F7 (iii) fields)
  f6-stage   the F6 control: 10 unmasked population slides (all tiles) staged by the same function
  f6-check   after the F6 GPU job: regenerated vectors and their L1 rows vs the accepted R2.6
             embeddings / bound baseline matrix (max rel <= 1e-6) -> f6_record.json (PASS/FAIL)
  verify     after a set's GPU job: F6 PASS beforehand, same GPU model as F6, every staged slide has
             a finite non-zero 768-d embedding whose coords equal the staged copy, staged copies re-hash
             to stage_record.json, no locks/extras, skip_errors false -> verify_record.json
  delete     requires verify PASS: deletion_record.json (paths + sha256, time, operator) BEFORE
             deleting ONLY job_dir/<COORDS>/features_conch_v15/*.h5; then deletion_done.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib.util
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S0 = _load("r2_1_tf2_s0_pool_folds", SCRIPTS / "r2_1_tf2_s0_pool_folds.py")
S3 = _load("r2_1_tf2_s3_masks", SCRIPTS / "r2_1_tf2_s3_masks.py")
S4 = _load("r2_1_tf2_s4_subsets", SCRIPTS / "r2_1_tf2_s4_subsets.py")
S5 = _load("r2_1_tf2_s5_jobspec", SCRIPTS / "r2_1_tf2_s5_jobspec.py")
REGEN = S4.titan_pins()                                  # scripts/r2_6_titan_arm1_regen.py

STAGE = "s4"
ENCODER = "conch_v15"
SLOTS = S3.SLOTS
BUDGETS = S3.BUDGETS
SET_NAMES = ("queries",) + tuple(f"v_{s}" for s in SLOTS)
F6_N, F6_SEED, F6_REL = 10, S5.PILOT_SEED, 1e-6          # specification §7 F6; release pilot sampler
EMB_DIM = 768
TITAN_MIN_TILES = 2        # design decision T1 (contract_candidate_006, 2026-10-17): TITAN cannot embed a 1-tile
                           # slide; < 2 retained tiles = no representation (not staged); OT/BoB keep 1..7
SENTINEL = REGEN.SENTINEL
SENTINEL_MIN = REGEN.SENTINEL_MIN
R26 = "r2_6_titan_arm1_regen_20261004"
BASELINE_ROLE = "baseline:titan/titan"

sha256_file = S0.sha256_file
write_json = S0.write_json
now = S0.now


class OpsError(RuntimeError):
    """A gate or binding failed: STOP."""


def require(cond, message):
    if not cond:
        raise OpsError(message)


def code_record():
    paths = {"titan_ops": Path(__file__).resolve(), "s4_script": Path(S4.__file__).resolve(),
             "s3_script": Path(S3.__file__).resolve(), "s5_script": Path(S5.__file__).resolve(),
             "titan_regen_pins": Path(REGEN.__file__).resolve()}
    return {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in paths.items()}


def titan_root(prosper_out) -> Path:
    return S0.stage_root(prosper_out) / STAGE / "titan"


def _check_under(path: Path, prosper_out) -> Path:
    root = titan_root(prosper_out).resolve()
    p = Path(path).absolute()
    require(root == p.resolve() or root in p.resolve().parents, f"{p} is not under {root}")
    return p


# ---------------------------------------------------------------- inputs
def load_s3_unit(s3_root: Path, budget: int):
    """COMPLETE S3 conch_v15 unit (re-hashed), offsets of population slides, masks/predictions."""
    import pandas as pd
    udir = S3.unit_dir(Path(s3_root), ENCODER, budget)
    comp = S4.verify_s3_unit(udir)
    off = pd.read_csv(udir / "slide_offsets.csv", dtype={"slide_id": str, "patient_id": str})
    masks = {s: np.load(udir / f"{s}_mask.npy", mmap_mode="r") for s in SLOTS}
    preds = {s: np.load(udir / f"{s}_predictions.npy", mmap_mode="r") for s in SLOTS}
    coords = np.load(udir / "coordinates.npy", mmap_mode="r")
    return udir, comp, off, masks, preds, coords


def population(s0_dir: Path, prosper_out):
    inputs = S0.default_inputs(S0.HAORAN, prosper_out)
    return S4.load_population(Path(s0_dir), inputs)


def query_assignment(s0_dir: Path):
    import pandas as pd
    return pd.read_csv(Path(s0_dir) / "query_assignment.csv", dtype={"slide_id": str, "patient_id": str})


def wsi_src_dir() -> Path:
    return Path(REGEN.WSI_SRC)


def spec_set(spec: dict, budget, set_name) -> dict:
    hits = [x for x in spec["sets"] if x.get("budget_percent") == budget and x.get("set") == set_name]
    require(len(hits) == 1, f"titan_spec: {len(hits)} entries for b{budget}/{set_name}")
    return hits[0]


# ---------------------------------------------------------------- stage
def apply_min_tiles(members, keep_of, min_tiles=TITAN_MIN_TILES):
    """Design decision T1: slides with 1 .. min_tiles-1 retained tiles are not staged (listed with their
    tile counts); zero-tile slides stay with titan_stage's own zero rule."""
    excluded = [{"slide_id": s, "tiles": int(keep_of[s].sum())} for s in members
                if 0 < int(keep_of[s].sum()) < min_tiles]
    drop = {e["slide_id"] for e in excluded}
    return [s for s in members if s not in drop], excluded


def set_plan(set_name, budget, pop, qa, off, masks, preds, coords):
    """members (population order), keep_of, per-slide head/version rows. Mask == (prediction == 1)
    re-checked; S3 coordinates kept for the h5 coordinate check."""
    require(set_name in SET_NAMES, f"unknown set {set_name}")
    offp = off[off["in_population"] == 1].set_index("slide_id")
    require(set(offp.index) == set(pop["slide_id"]), "S3 population slides != retrieval population")
    qa = qa.set_index("slide_id")
    members, keep_of, rows, s3_coords = [], {}, {}, {}
    for sid in pop["slide_id"]:
        r = offp.loc[sid]
        if set_name == "queries":
            slot = qa.loc[sid, "head_slot"]
            require(qa.loc[sid, "library_version_both_sided"] == "v_" + slot, f"{sid}: S0 version/head mismatch")
            version = qa.loc[sid, "library_version_both_sided"]
        else:
            slot, version = set_name[2:], set_name
        a, z = int(r["tile_start"]), int(r["tile_stop"])
        keep = np.asarray(masks[slot][a:z], dtype=bool)
        require(np.array_equal(keep, np.asarray(preds[slot][a:z]) == S3.TUMOUR_CLASS),
                f"{sid}: mask != (prediction == 1) for {slot}")
        members.append(sid)
        keep_of[sid] = keep
        s3_coords[sid] = np.asarray(coords[a:z])
        rows[sid] = {"slide_id": sid, "slide_index": int(r["slide_index"]), "query_index": int(r["query_index"]),
                     "patient_id": str(r["patient_id"]), "head_slot": slot,
                     "head_id": S3.head_id(ENCODER, budget, slot), "library_version": version,
                     "original_tiles": z - a, "retained_tiles": int(keep.sum()),
                     "bank_feature_sha256": str(r["feature_sha256"]), "bank_file": str(r["bank_file"])}
    return members, keep_of, rows, s3_coords


def _check_h5_coords(members, keep_of, s3_coords, conch_dir: Path):
    import h5py
    bad = []
    for s in members:
        if not keep_of[s].any():
            continue
        with h5py.File(Path(conch_dir) / f"{s}.h5", "r") as h:
            c = np.asarray(h["coords"][:])
        if c.shape != s3_coords[s].shape or not np.array_equal(c, s3_coords[s]):
            bad.append(s)
    require(not bad, f"h5 coords differ from S3 coordinates for {len(bad)} slides, e.g. {bad[:5]}")


def write_manifest(out: Path, kind: str, budget, set_name, rec: dict, rows: dict, extra: dict):
    staged = {x["slide_id"]: x for x in rec["staged"]}
    per = []
    for sid, r in rows.items():
        x = staged.get(sid)
        if x is not None:
            require(x["tiles"] == r["retained_tiles"], f"{sid}: staged tiles != retained tiles")
            require(x["source_sha256"] == r["bank_feature_sha256"], f"{sid}: bank h5 sha256 != S3/S0 binding")
        per.append({**r, "staged": int(x is not None), "staged_sha256": x["staged_sha256"] if x else "",
                    "source_sha256": x["source_sha256"] if x else ""})
    man = {"schema_version": 1, "kind": kind, "encoder_id_in_tables": "titan", "mask_encoder": ENCODER,
           "budget_percent": budget, "set": set_name, "created": now(), "host": socket.gethostname(),
           "lsf_job": os.environ.get("LSB_JOBID"), "stage_dir": str(out),
           "stage_record": {"path": str(out / "stage_record.json"), "sha256": sha256_file(out / "stage_record.json")},
           "n_members": len(rows), "n_staged": len(staged), "not_staged_zero_tiles": rec["not_staged_zero_tiles"],
           "staged_tiles": int(sum(x["tiles"] for x in rec["staged"])),
           "staged_bytes": int(sum((Path(rec["job_dir"]) / REGEN.COORDS / "features_conch_v15" / f"{s}.h5").stat().st_size
                                   for s in staged)),
           "trident_command": rec["trident_command"], "slides": per, **extra, "code": code_record()}
    write_json(man, out / "stage_manifest.json")
    return man


def stage(budget, set_name, s0_dir: Path, s3_root: Path, spec_path: Path, prosper_out, *, out=None, log=print):
    t0 = time.time()
    spec = json.loads(Path(spec_path).read_text())
    ent = spec_set(spec, budget, set_name)
    out = Path(out or ent["stage_dir"])
    _check_under(out, prosper_out)
    require(not out.exists(), f"refusing existing stage dir {out}")
    pop, qa = population(s0_dir, prosper_out), query_assignment(s0_dir)
    udir, comp, off, masks, preds, coords = load_s3_unit(s3_root, budget)
    members, keep_of, rows, s3c = set_plan(set_name, budget, pop, qa, off, masks, preds, coords)
    conch_dirs = {str(Path(r["bank_file"]).parent) for r in rows.values()}
    require(len(conch_dirs) == 1, f"population bank files in {len(conch_dirs)} directories")
    conch_dir = Path(conch_dirs.pop())
    _check_h5_coords(members, keep_of, s3c, conch_dir)
    out.mkdir(parents=True)
    staged_members, below = apply_min_tiles(members, keep_of)
    rec = S4.titan_stage(set_name, staged_members, keep_of, conch_dir, wsi_src_dir(), out)
    require(rec["trident_command"] == ent["trident_command"], "trident command differs from titan_spec.json")
    rec["min_tiles"] = TITAN_MIN_TILES
    rec["not_staged_below_min_tiles"] = below
    rec["min_tiles_rule"] = ("lead decision T1 (contract_candidate_006/CONTRACT_AMENDMENT_POOL_40X.md, 2026-10-17): "
                             "TITAN arm only, < 2 retained tiles = no representation; zero-tile slides are listed "
                             "in not_staged_zero_tiles")
    write_json(rec, out / "stage_record.json")
    man = write_manifest(out, "set", budget, set_name, rec, rows, {
        "s3_unit": {"dir": str(udir), "complete_sha256": sha256_file(udir / "COMPLETE.json")},
        "s0": {"dir": str(s0_dir), "query_assignment_sha256": sha256_file(Path(s0_dir) / "query_assignment.csv")},
        "titan_spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "conch_dir": str(conch_dir), "wsi_src": str(wsi_src_dir()), "min_tiles": TITAN_MIN_TILES,
        "not_staged_below_min_tiles": below,
        "seconds": round(time.time() - t0, 1)})
    log(json.dumps({"set": f"b{budget}/{set_name}", "staged": man["n_staged"], "tiles": man["staged_tiles"],
                    "bytes": man["staged_bytes"], "seconds": man["seconds"]}))
    return man


# ---------------------------------------------------------------- F6
def accepted_paths(prosper_out) -> dict:
    p = REGEN.default_paths(str(S0.REPO), str(prosper_out))
    regen = Path(prosper_out) / R26 / "job_dir" / REGEN.COORDS / "slide_features_titan"
    return {"published": Path(p["published_slide_features"]), "regenerated": regen,
            "metadata": Path(p["metadata"]), "matrix": Path(prosper_out) / R26 /
            f"{REGEN.BASE}_no_same_patient.pkl", "patch_features": Path(p["published_patch_features"])}


def accepted_embedding_file(sid, acc: dict) -> Path:
    f = acc["regenerated"] / f"{sid}.h5"
    return f if f.exists() else acc["published"] / f"{sid}.h5"


def baseline_binding(s0_dir: Path) -> dict:
    b = json.loads((Path(s0_dir) / "input_bindings.json").read_text())
    hits = [f for f in b["files"] if f.get("role") == BASELINE_ROLE]
    require(len(hits) == 1, f"S0 bindings: {len(hits)} entries for {BASELINE_ROLE}")
    return hits[0]


def accepted_embeddings(prosper_out, s0_dir: Path, *, check_matrix=True) -> tuple:
    """The accepted unfiltered TITAN embeddings (2,900 published + 21 regenerated) and the gate that
    they rebuild the bound baseline matrix bitwise (R2.6 rebuild rule: pdist cityblock, same-patient
    mask by the metadata UID.Subject, 1e6 sentinel)."""
    import pandas as pd
    acc = accepted_paths(prosper_out)
    ids, X = REGEN.load_slide_embeddings(acc["published"])
    regen_ids = sorted(f.stem for f in acc["regenerated"].glob("*.h5"))
    require(len(regen_ids) == REGEN.N_REGEN, f"{len(regen_ids)} regenerated embeddings, expected {REGEN.N_REGEN}")
    pos = {s: i for i, s in enumerate(ids)}
    for s in regen_ids:
        X[pos[s]] = REGEN.read_vec(acc["regenerated"] / f"{s}.h5")
    info = {"published_dir": str(acc["published"]), "regenerated_dir": str(acc["regenerated"]),
            "n": len(ids), "regenerated": regen_ids}
    if check_matrix:
        bind = baseline_binding(s0_dir)
        mpath = Path(bind.get("resolved_path") or bind["path"])
        require(sha256_file(mpath) == bind["sha256"], "baseline TITAN matrix sha256 != S0 binding")
        M, _ = REGEN._mask_same_patient(REGEN._distance_frame(ids, X, "cityblock"), acc["metadata"])
        B = pd.read_pickle(mpath)
        same = [str(i) for i in B.index] == ids and [str(c) for c in B.columns] == ids
        eq = bool(same and np.array_equal(M.to_numpy(), B.to_numpy()))
        info["baseline_matrix"] = {"path": str(mpath), "sha256": bind["sha256"], "same_layout": same,
                                   "rebuilt_bitwise_equal": eq}
        require(eq, "accepted TITAN embeddings do not rebuild the bound baseline matrix bitwise: STOP")
    return ids, X, info


def f6_stage(s0_dir: Path, prosper_out, out: Path, *, log=print):
    import pandas as pd
    out = _check_under(Path(out), prosper_out)
    require(not out.exists(), f"refusing existing {out}")
    pop = population(s0_dir, prosper_out)
    ids = S5.pilot_queries(pop["slide_id"].tolist(), n=F6_N, seed=F6_SEED)
    acc = accepted_paths(prosper_out)
    conch = acc["patch_features"]
    import h5py
    keep_of, rows = {}, {}
    for s in ids:
        with h5py.File(conch / f"{s}.h5", "r") as h:
            n = len(h["coords"])
        keep_of[s] = np.ones(n, dtype=bool)
        e = accepted_embedding_file(s, acc)
        rows[s] = {"slide_id": s, "head_slot": "", "head_id": "unmasked", "library_version": "unfiltered",
                   "original_tiles": n, "retained_tiles": n, "bank_feature_sha256": sha256_file(conch / f"{s}.h5"),
                   "accepted_embedding": str(e), "accepted_embedding_sha256": sha256_file(e)}
    out.mkdir(parents=True)
    rec = S4.titan_stage("f6_control_unmasked", ids, keep_of, conch, wsi_src_dir(), out)
    man = write_manifest(out, "f6_control", None, "f6_control_unmasked", rec, rows, {
        "selection": f"r2_1_tf2_s5_jobspec.pilot_queries(population ids in query_index order, n={F6_N}, "
                     f"seed={F6_SEED}) (the release's seeded sampler)", "slide_ids": ids,
        "rule": f"F6: regenerated embedding and its L1 row reproduce the accepted R2.6 values, max rel <= {F6_REL}"})
    log(json.dumps({"f6_staged": ids}))
    return man


def _gpu(job_dir: Path) -> dict:
    return REGEN.parse_gpu_info(Path(job_dir) / "gpu_info.txt")


def f6_check(stage_dir: Path, s0_dir: Path, prosper_out, *, log=print) -> dict:
    stage_dir = Path(stage_dir)
    man = json.loads((stage_dir / "stage_manifest.json").read_text())
    rec_path = stage_dir / "f6_record.json"
    require(not rec_path.exists(), f"refusing existing {rec_path}")
    job = stage_dir / "job_dir"
    out_dir = job / REGEN.COORDS / "slide_features_titan"
    gpu = _gpu(job)
    ids, X, info = accepted_embeddings(prosper_out, s0_dir)
    pos = {s: i for i, s in enumerate(ids)}
    import pandas as pd
    B = pd.read_pickle(info["baseline_matrix"]["path"]).to_numpy()
    per, ok = [], True
    for s in man["slide_ids"]:
        f = out_dir / f"{s}.h5"
        if not f.exists():
            per.append({"slide": s, "passed": False, "error": "missing"})
            ok = False
            continue
        v = REGEN.read_vec(f)
        a = X[pos[s]]
        vec_rel = float(np.abs(v - a).max() / np.abs(a).max())
        Y = X.copy()
        Y[pos[s]] = v
        new_row = np.abs(Y - v[None, :]).sum(axis=1)
        brow = B[pos[s]]
        use = (brow < SENTINEL_MIN) & (np.arange(len(ids)) != pos[s])
        old_row = np.abs(X - a[None, :]).sum(axis=1)
        row_rel = float(np.max(np.abs(new_row[use] - brow[use]) / np.abs(brow[use])))
        own_rel = float(np.max(np.abs(old_row[use] - brow[use]) / np.abs(brow[use])))
        passed = bool(vec_rel <= F6_REL and row_rel <= F6_REL)
        ok &= passed
        per.append({"slide": s, "embedding_sha256": sha256_file(f), "accepted_embedding": str(accepted_embedding_file(
            s, accepted_paths(prosper_out))), "vector_max_abs_diff": float(np.abs(v - a).max()),
            "vector_max_rel": vec_rel, "bitwise_equal": bool(np.array_equal(v, a)), "row_cells": int(use.sum()),
            "row_max_rel_vs_bound_matrix": row_rel, "accepted_row_max_rel_vs_bound_matrix": own_rel, "passed": passed})
    try:
        REGEN.check_coords(out_dir, job / REGEN.COORDS / "features_conch_v15", man["slide_ids"])
        coords_ok = True
    except SystemExit as e:
        coords_ok, per = False, per + [{"coords_error": str(e)}]
    REGEN.check_trident_config(job)
    staged_ok = all(sha256_file(Path(job) / REGEN.COORDS / "features_conch_v15" / f"{x['slide_id']}.h5")
                    == x["staged_sha256"] for x in man["slides"] if x["staged"])
    locks = REGEN.lock_files(job)
    status = "PASS" if (ok and coords_ok and staged_ok and not locks) else "FAIL"
    rec = {"check": "F6 (contract §7): TITAN embedding of unmasked tile sets for 10 slides reproduces the accepted "
                    "R2.6 embeddings and regenerated matrix rows, max rel <= 1e-6", "status": status,
           "created": now(), "host": socket.gethostname(), "gpu_info": gpu, "gpu_model": gpu["model"],
           "per_slide": per, "coords_ok": coords_ok, "staged_copies_rehash_ok": staged_ok, "locks": locks,
           "accepted": info, "stage_manifest_sha256": sha256_file(stage_dir / "stage_manifest.json"),
           "provenance": REGEN.provenance(job), "code": code_record()}
    write_json(rec, rec_path)
    log(json.dumps({"F6": status, "gpu": gpu["model"],
                    "max_vector_rel": max((p.get("vector_max_rel", np.inf) for p in per if "slide" in p), default=None),
                    "max_row_rel": max((p.get("row_max_rel_vs_bound_matrix", np.inf) for p in per if "slide" in p),
                                       default=None)}))
    return rec


# ---------------------------------------------------------------- verify / delete
def verify(stage_dir: Path, f6_record: Path, *, log=print) -> dict:
    stage_dir = Path(stage_dir)
    out_path = stage_dir / "verify_record.json"
    require(not out_path.exists(), f"refusing existing {out_path}")
    man = json.loads((stage_dir / "stage_manifest.json").read_text())
    srec = json.loads((stage_dir / "stage_record.json").read_text())
    job = stage_dir / "job_dir"
    feat = job / REGEN.COORDS / "features_conch_v15"
    emb = job / REGEN.COORDS / "slide_features_titan"
    checks = {}

    def check(name, ok, **info):
        checks[name] = {"result": "PASS" if ok else "FAIL", **info}

    f6 = json.loads(Path(f6_record).read_text())
    check("f6_passed_beforehand", f6.get("status") == "PASS", f6_record=str(f6_record),
          f6_sha256=sha256_file(f6_record))
    check("stage_record_matches_manifest", sha256_file(stage_dir / "stage_record.json") == man["stage_record"]["sha256"])
    try:
        gpu = _gpu(job)
        check("gpu_model_equals_f6", gpu["model"] == f6["gpu_model"], gpu_model=gpu["model"], f6_model=f6["gpu_model"],
              hostname=gpu["hostname"])
    except SystemExit as e:
        gpu = {"error": str(e)}
        check("gpu_model_equals_f6", False, error=str(e))
    staged = [x["slide_id"] for x in srec["staged"]]
    bad_hash = [x["slide_id"] for x in srec["staged"] if sha256_file(feat / f"{x['slide_id']}.h5") != x["staged_sha256"]]
    check("staged_copies_rehash_to_stage_record", not bad_hash, n=len(staged), mismatched=bad_hash[:20])
    have = sorted(f.stem for f in emb.glob("*.h5")) if emb.is_dir() else []
    missing = sorted(set(staged) - set(have))
    extra = sorted(set(have) - set(staged))
    check("every_staged_slide_has_embedding", not missing and not extra, n_embeddings=len(have),
          missing=missing[:20], extra=extra[:20])
    bad, embs = [], {}
    for s in sorted(set(staged) & set(have)):
        v = REGEN.read_vec(emb / f"{s}.h5")
        if v.shape != (EMB_DIM,) or not np.isfinite(v).all() or not v.any():
            bad.append(s)
        embs[s] = sha256_file(emb / f"{s}.h5")
    check("embeddings_finite_nonzero_768", not bad, failing=bad[:20])
    try:
        REGEN.check_coords(emb, feat, sorted(set(staged) & set(have)))
        check("embedding_coords_equal_staged_copies", True)
    except SystemExit as e:
        check("embedding_coords_equal_staged_copies", False, error=str(e)[:2000])
    try:
        REGEN.check_trident_config(job)
        check("trident_config_skip_errors_false", True)
    except SystemExit as e:
        check("trident_config_skip_errors_false", False, error=str(e))
    locks = REGEN.lock_files(job)
    check("no_lock_files", not locks, locks=locks[:20])
    status = "PASS" if all(c["result"] == "PASS" for c in checks.values()) else "FAIL"
    rec = {"schema_version": 1, "status": status, "set": man["set"], "budget_percent": man["budget_percent"],
           "created": now(), "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"),
           "checks": checks, "gpu_info": gpu, "embeddings_dir": str(emb), "embedding_sha256": embs,
           "stage_manifest_sha256": sha256_file(stage_dir / "stage_manifest.json"),
           "provenance": REGEN.provenance(job), "code": code_record()}
    write_json(rec, out_path)
    log(json.dumps({"verify": status, "set": f"b{man['budget_percent']}/{man['set']}", "embeddings": len(have),
                    "failed": [k for k, c in checks.items() if c["result"] != "PASS"]}))
    return rec


def delete_staged(stage_dir: Path, operator: str, *, log=print) -> dict:
    """Delete ONLY job_dir/<COORDS>/features_conch_v15/*.h5 after a PASS verify; record first."""
    stage_dir = Path(stage_dir)
    v = json.loads((stage_dir / "verify_record.json").read_text())
    require(v["status"] == "PASS", f"{stage_dir}: verify_record is not PASS; nothing deleted")
    dr = stage_dir / "deletion_record.json"
    require(not dr.exists(), f"refusing existing {dr}")
    srec = json.loads((stage_dir / "stage_record.json").read_text())
    feat = (stage_dir / "job_dir" / REGEN.COORDS / "features_conch_v15").resolve()
    files = []
    for x in srec["staged"]:
        f = feat / f"{x['slide_id']}.h5"
        require(f.parent == feat and f.suffix == ".h5" and f.is_file(), f"{f}: not a staged copy")
        files.append({"path": str(f), "sha256": x["staged_sha256"], "bytes": f.stat().st_size})
    others = sorted(str(p) for p in feat.iterdir() if p.name not in {Path(x["path"]).name for x in files})
    require(not others, f"unexpected files among the staged copies: {others[:5]}")
    rec = {"schema_version": 1, "set": srec["set"], "stage_dir": str(stage_dir), "created": now(),
           "operator": operator, "host": socket.gethostname(),
           "rule": "titan_spec.json staging_rule 4-5: delete ONLY the staged byte-copies of inputs "
                   "(job_dir/<COORDS>/features_conch_v15/*.h5); outputs, manifests, records, configs, logs and "
                   "wsi_dir symlinks are kept",
           "verify_record_sha256": sha256_file(stage_dir / "verify_record.json"),
           "stage_record_sha256": sha256_file(stage_dir / "stage_record.json"),
           "stage_manifest_sha256": sha256_file(stage_dir / "stage_manifest.json"),
           "sha256_source": "stage_record.json staged_sha256, re-hashed and matched by verify_record.json",
           "n_files": len(files), "bytes": int(sum(f["bytes"] for f in files)), "files": files}
    write_json(rec, dr)
    for f in files:
        os.remove(f["path"])
    left = sorted(p.name for p in feat.iterdir())
    done = {"deleted": len(files), "bytes": rec["bytes"], "remaining_in_features_dir": left,
            "deletion_record_sha256": sha256_file(dr), "completed": now()}
    write_json(done, stage_dir / "deletion_done.json")
    log(json.dumps({"deleted": len(files), "bytes": rec["bytes"], "set": srec["set"]}))
    return done


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--budget", type=int, required=True, choices=BUDGETS)
    s.add_argument("--set", required=True, choices=SET_NAMES)
    s.add_argument("--s0", required=True)
    s.add_argument("--s3-root", required=True)
    s.add_argument("--spec", required=True)
    f = sub.add_parser("f6-stage")
    f.add_argument("--s0", required=True)
    f.add_argument("--out", required=True)
    c = sub.add_parser("f6-check")
    c.add_argument("--stage-dir", required=True)
    c.add_argument("--s0", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--stage-dir", required=True)
    v.add_argument("--f6-record", required=True)
    d = sub.add_parser("delete")
    d.add_argument("--stage-dir", required=True)
    d.add_argument("--operator", required=True)
    a = ap.parse_args(argv)
    prosper_out = os.environ.get("PROSPER_OUT")
    if not prosper_out:
        sys.exit("PROSPER_OUT must be exported")
    if a.cmd == "stage":
        stage(a.budget, a.set, Path(a.s0), Path(a.s3_root), Path(a.spec), prosper_out)
        return 0
    if a.cmd == "f6-stage":
        f6_stage(Path(a.s0), prosper_out, Path(a.out))
        return 0
    if a.cmd == "f6-check":
        return 0 if f6_check(Path(a.stage_dir), Path(a.s0), prosper_out)["status"] == "PASS" else 1
    if a.cmd == "verify":
        return 0 if verify(Path(a.stage_dir), Path(a.f6_record))["status"] == "PASS" else 1
    delete_staged(Path(a.stage_dir), a.operator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
