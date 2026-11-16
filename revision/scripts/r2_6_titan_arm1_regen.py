#!/usr/bin/env python3
"""R2.6 / Fig. 3B TITAN comparator repair: regenerate the 21 all-zero published
(arm-1, all-tissue) TITAN slide embeddings and rebuild the arm-1 L1 matrix.

  prepare        stage a trident job dir with byte-copies of the 21 published CONCH v1.5
                 patch-feature h5 files and a wsi_dir of symlinks to the 21 WSIs; write
                 manifest.json (incl. the trident command, NOT run) and prepare_record.json.
  stage-controls stage 3 reproducibility controls (non-zero published slides whose patch counts are
                 nearest to the min/median/max patch count of the 21; ties -> lowest lexical id) under
                 controls/ (job_dir copies, wsi_dir symlinks, controls/manifest.json).
  command        print the corrected trident commands (main 21-slide run and controls run;
                 no --skip_errors, HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1, --gpu 0) and write
                 command_record.json. The frozen manifest.json is not edited.
  check-controls after the GPU controls run: refuse unless controls/job_dir/gpu_info.txt parses and the staged
                 control CONCH copies and published control embeddings re-hash to controls/manifest.json;
                 compare each regenerated control vector with its published vector; write
                 controls/control_record.json (incl. parsed GPU model/hostname).
  rebuild        after the GPU slide-encoder-only job: verify the 21 regenerated embeddings against the
                 staged inputs, combine with the 2,900 published non-zero embeddings, build the cityblock
                 matrix with the published same-patient mask convention, and reconcile. Outputs are
                 written as *.candidate and renamed to final names only if ALL checks pass, otherwise
                 to *.FAILED (exit 1); an unhandled exception also renames partial outputs to *.FAILED.
                 Requires controls/control_record.json with all_passed true, its controls_manifest_sha256
                 equal to controls/manifest.json, and the same environment for both runs (job_dir/ and
                 controls/job_dir/gpu_info.txt with the same GPU model; trident HEAD/diff sha256 and TITAN
                 refs/main/blob in control_record equal to rebuild-time values);
                 control vectors are never included in the matrix. `--dry-run` checks input presence only.

Interpretation: controls reproduce + 21 non-zero -> transient fault in the Nov-2025 run, mixing
published and regenerated vectors is defensible; controls deviate -> environment drift, do not mix;
21 zero again -> deterministic fault (rebuild refuses all-zero regenerations).

Writes only under $PROSPER_OUT/r2_6_titan_arm1_regen_20261004/. Does not import
`prosper`; distance/mask code is copied from scripts/r2_6_titan_arm2_l1_matrix.py
(itself verbatim from prosper-search aggregate_cli.py; source hashes recorded).
"""
import argparse, datetime, hashlib, json, os, pickle, re, shutil, socket, subprocess, sys, time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

OUT_NAME = "r2_6_titan_arm1_regen_20261004"
COORDS = "40x_1024px_0px_overlap"
N_REGEN, N_ALL, N_MASKED = 21, 2921, 1949
BASE = "combined_similarity_matrix_titan_titan_repaired"
SENTINEL_MIN = 1.0e5
PKG = "/PROSPER_WORKSPACE/prosper-search"
AGG_SRC = PKG + "/prosper/embeddings/aggregate_cli.py"
TRIDENT = "/PROSPER_WORKSPACE/comet-feature-gen/comet-titan/trident"
TRIDENT_PY = "/HOME/miniconda3/envs/trident/bin/python"
WSI_SRC = "/PROSPER_WORKSPACE/data/comet_final_HPC"
HF_TITAN = os.path.expanduser("~/.cache/huggingface/hub/models--MahmoodLab--TITAN")
SLIDE_CONFIG = "_config_slide_features_titan.json"
PATCH_ATTRS = ("patch_size", "patch_size_level0")
MIN_STD = 0.1                 # regenerated vector population std must be >= this (published min 0.234)
DUP_REL = 0.01                # near-duplicate: pairwise L1 among the 21 < DUP_REL * mean published NN distance
CTRL_L1_REL, CTRL_NN_REL = 0.01, 0.05
INTERPRETATION = {
    "controls_reproduce_and_21_nonzero": "transient fault in the Nov-2025 run; mixing 2,900 published and "
                                         "21 regenerated vectors is defensible",
    "controls_deviate": "environment drift since the Nov-2025 run; do not mix regenerated with published vectors",
    "21_zero_again": "deterministic fault; rebuild refuses all-zero regenerations",
}

# ---- copied from scripts/r2_6_titan_arm2_l1_matrix.py (verbatim from AGG_SRC L44, L123-148,
# metadata path parameterised instead of paths.slide_metadata()) ----
SENTINEL = 1e6


def _distance_frame(ids, mat, metric):
    d = squareform(pdist(mat, metric=metric))
    np.fill_diagonal(d, SENTINEL)
    return pd.DataFrame(d, index=ids, columns=ids)


def _mask_same_patient(df, metadata_csv, patient_col="UID.Subject"):
    meta = pd.read_csv(metadata_csv)
    id_col = "Slide.Scan.File"
    meta[id_col] = meta[id_col].astype(str)
    lookup = (meta.drop_duplicates(id_col)
                  .set_index(id_col)[patient_col].to_dict())
    ids = [str(i) for i in df.index]
    pat = np.array([str(lookup.get(i, f"__unknown_{i}")) for i in ids])
    same = pat[:, None] == pat[None, :]
    np.fill_diagonal(same, False)
    out = df.copy()
    vals = out.to_numpy()
    n_masked = int(same.sum())
    vals[same] = SENTINEL
    out.iloc[:, :] = vals
    return out, n_masked // 2


def load_slide_embeddings(src):
    ids, vecs = [], []
    for h5 in sorted(Path(src).glob("*.h5")):
        with h5py.File(h5, "r") as h:
            v = np.asarray(h["features"][:], dtype=np.float64).reshape(-1)
        ids.append(h5.stem)
        vecs.append(v)
    return ids, np.stack(vecs)


def all_zero_ids(ids, mat):
    return [s for s, z in zip(ids, ~mat.any(axis=1)) if z]


def masked_offdiag(a):
    return (a >= SENTINEL_MIN) & ~np.eye(len(a), dtype=bool)
# ---- end of copy ----


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def default_paths(repo, out_root):
    P = os.path.dirname(repo)
    pub = P + f"/comet-feature-gen/comet-titan/trident/trident_processed/{COORDS}"
    return {
        "published_slide_features": pub + "/slide_features_titan",
        "published_patch_features": pub + "/features_conch_v15",
        "published_matrix": P + "/comet-feature-gen/comet-titan/output/similarity_matrix/combined_similarity_matrix_titan_titan_no_same_patient.pkl",
        "metadata": P + "/comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv",
        "clique_csv": out_root + "/r2_6_titan_common_pool_20261003/arm1_identical_embedding_clique.csv",
        "wsi_src": WSI_SRC,
        "outdir": os.path.join(out_root, OUT_NAME),
    }


def read_vec(path):
    with h5py.File(path, "r") as h:
        return np.asarray(h["features"][:], dtype=np.float64).reshape(-1)


def clique_ids(path):
    ids = sorted(pd.read_csv(path, dtype={"slide": str})["slide"])
    if len(ids) != N_REGEN or len(set(ids)) != N_REGEN:
        raise SystemExit(f"REFUSING: expected {N_REGEN} unique clique slides, got {len(ids)}")
    return ids


def legacy_trident_command(wsi_dir, job_dir):
    """Command text frozen in the 10-04 manifest.json (superseded: used --skip_errors)."""
    return (f"cd {TRIDENT} && {TRIDENT_PY} run_batch_of_slides.py --task feat --wsi_dir {wsi_dir} "
            f"--job_dir {job_dir} --patch_encoder conch_v15 --slide_encoder titan --mag 40 "
            f"--patch_size 1024 --coords_dir {COORDS} --skip_errors")


def trident_command(wsi_dir, job_dir):
    return (f"cd {TRIDENT} && HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 {TRIDENT_PY} run_batch_of_slides.py "
            f"--task feat --wsi_dir {wsi_dir} --job_dir {job_dir} --patch_encoder conch_v15 --slide_encoder titan "
            f"--mag 40 --patch_size 1024 --coords_dir {COORDS} --gpu 0")


def corrected_command(manifest):
    new = trident_command(manifest["wsi_dir"], manifest["job_dir"])
    if manifest["trident_command"] not in (new, legacy_trident_command(manifest["wsi_dir"], manifest["job_dir"])):
        raise SystemExit("REFUSING: manifest trident_command is neither the legacy nor the corrected text")
    return new


def ctrl_paths(out):
    c = Path(out) / "controls"
    job = c / "job_dir"
    return {"root": c, "job": job, "feat": job / COORDS / "features_conch_v15", "wsi": c / "wsi_dir",
            "regen": job / COORDS / "slide_features_titan", "manifest": c / "manifest.json",
            "record": c / "control_record.json"}


def iso_mtime(f):
    return datetime.datetime.fromtimestamp(os.stat(f).st_mtime).astimezone().isoformat()


GPU_L_RE = re.compile(r"^GPU\s+\d+:\s*(.+?)\s*\(UUID:")
SW_KEYS = (("trident_git", "head"), ("trident_git", "diff_sha256"),
           ("titan_hf_snapshot", "refs_main"), ("titan_hf_snapshot", "model_safetensors_blob"))


def parse_gpu_info(path):
    """Parse the independently written gpu_info.txt; raise SystemExit if missing or malformed.

    Rule (blank lines ignored, each line stripped): line 1 = hostname (must not look like an nvidia-smi line);
    then >=1 `nvidia-smi -L` lines `GPU <i>: <model> (UUID: ...)` (`MIG ...` sub-device lines tolerated and
    recorded); then >=1 `--query-gpu=name,driver_version --format=csv,noheader` lines `<model>, <driver>` (split at
    the last comma). The query names must equal the -L models in order and all listed GPUs must be one model, so
    the device used by `--gpu 0` is unambiguous."""
    f = Path(path)
    if not f.is_file():
        raise SystemExit(f"REFUSING: {f} missing (lead must write hostname + nvidia-smi output)")
    lines = [ln.strip() for ln in f.read_text(errors="replace").splitlines() if ln.strip()]
    bad = lambda why: SystemExit(f"REFUSING: {f} malformed: {why}")
    if not lines or GPU_L_RE.match(lines[0]) or "," in lines[0]:
        raise bad("line 1 must be the hostname")
    l_models, mig, rows = [], [], []
    for ln in lines[1:]:
        m = GPU_L_RE.match(ln)
        if m or ln.startswith("MIG "):
            if rows:
                raise bad(f"nvidia-smi -L line after query lines: {ln!r}")
            if m:
                l_models.append(m.group(1))
            else:
                mig.append(ln)
        elif "," in ln:
            name, drv = (x.strip() for x in ln.rsplit(",", 1))
            if not name or not drv:
                raise bad(f"query line {ln!r}")
            rows.append([name, drv])
        else:
            raise bad(f"unrecognised line {ln!r}")
    if not l_models or not rows:
        raise bad("needs >=1 `nvidia-smi -L` line and >=1 name,driver_version line")
    if [r[0] for r in rows] != l_models:
        raise bad(f"-L models {l_models} != query names {[r[0] for r in rows]}")
    if len(set(l_models)) != 1:
        raise bad(f"heterogeneous GPU models {sorted(set(l_models))}")
    return {"file_sha256": sha256(f), "hostname": lines[0], "model": l_models[0], "models": l_models,
            "driver_versions": sorted({r[1] for r in rows}), "query_rows": rows, "mig_lines": mig,
            "raw": "\n".join(lines)[:4000]}


def software_provenance():
    """TITAN HF snapshot ref/blob, trident git HEAD and sha256 of `git diff`. Never fails."""
    try:
        ref = (Path(HF_TITAN) / "refs" / "main").read_text().strip()
        titan = {"refs_main": ref, "model_safetensors_blob": os.path.basename(
            os.path.realpath(Path(HF_TITAN) / "snapshots" / ref / "model.safetensors"))}
    except OSError as e:
        titan = {"error": repr(e)}
    try:
        run = lambda *a: subprocess.run(["git", "-C", TRIDENT, *a], capture_output=True, check=True).stdout
        diff = run("diff")
        git = {"path": TRIDENT, "head": run("rev-parse", "HEAD").decode().strip(),
               "diff_sha256": hashlib.sha256(diff).hexdigest(), "diff_bytes": len(diff)}
    except (OSError, subprocess.CalledProcessError) as e:
        git = {"error": repr(e)}
    return {"titan_hf_snapshot": titan, "trident_git": git}


def provenance(job_dir):
    """Hashes/mtimes of trident _config*/_logs* files, parsed gpu_info.txt (or its error), plus
    software_provenance(). Never fails."""
    job = Path(job_dir)
    files = lambda pat: {str(f.relative_to(job)): {"sha256": sha256(f), "mtime": iso_mtime(f)}
                         for f in sorted(job.rglob(pat)) if f.is_file()}
    try:
        gpu = parse_gpu_info(job / "gpu_info.txt")
    except SystemExit as e:
        gpu = {"error": str(e)}
    return {"hostname_of_this_check": socket.gethostname(), "config_files": files("_config*"),
            "log_files": files("_logs*"), "gpu_info": gpu, **software_provenance()}


def lock_files(job_dir):
    return sorted(str(f) for f in Path(job_dir).rglob("*.lock"))


def check_trident_config(job_dir):
    """Raise SystemExit unless the slide-encoder _config exists and no _config* json used skip_errors."""
    f = Path(job_dir) / COORDS / SLIDE_CONFIG
    if not f.exists():
        raise SystemExit(f"REFUSING: trident config {f} missing")
    if json.load(open(f)).get("skip_errors") is not False:
        raise SystemExit(f"REFUSING: {f} does not record skip_errors=false")
    for c in sorted(Path(job_dir).rglob("_config*.json")):
        v = json.load(open(c)).get("skip_errors", False)
        if v is not False:
            raise SystemExit(f"REFUSING: {c} has skip_errors={v!r}")


def check_coords(regen_dir, staged_dir, ids):
    """Raise SystemExit unless each regenerated h5's coords (and patch-size attrs present in the staged copy)
    equal the staged CONCH copy."""
    bad = []
    for s in ids:
        with h5py.File(Path(regen_dir) / f"{s}.h5", "r") as r, h5py.File(Path(staged_dir) / f"{s}.h5", "r") as c:
            if "coords" not in r:
                bad.append(f"{s}: no coords")
                continue
            if r["coords"].shape != c["coords"].shape or not np.array_equal(r["coords"][:], c["coords"][:]):
                bad.append(f"{s}: coords differ")
            for k in PATCH_ATTRS:
                if k in c["coords"].attrs and (k not in r["coords"].attrs
                                               or r["coords"].attrs[k] != c["coords"].attrs[k]):
                    bad.append(f"{s}: coords attr {k} differs")
    if bad:
        raise SystemExit("REFUSING: regenerated outputs not tied to staged inputs: " + "; ".join(bad))


def select_controls(pub_counts, regen_counts):
    """Nearest published (non-zero) slide by patch count to min/median/max of the 21; ties -> lowest lexical id;
    a slide is chosen at most once."""
    r = sorted(regen_counts)
    targets = {"min": r[0], "median": float(np.median(r)), "max": r[-1]}
    chosen = []
    for lab, t in targets.items():
        s = min((x for x in pub_counts if x not in [c["slide"] for c in chosen]),
                key=lambda x: (abs(pub_counts[x] - t), str(x)))
        chosen.append({"target": lab, "target_patch_count": t, "slide": s, "n_patches": pub_counts[s]})
    return chosen


def stage_controls(p):
    t0 = time.time()
    out = Path(p["outdir"])
    C = ctrl_paths(out)
    if not (out / "manifest.json").exists():
        sys.exit(f"REFUSING: {out / 'manifest.json'} missing; run prepare first")
    if C["root"].exists():
        sys.exit(f"REFUSING: {C['root']} exists; stage-controls never overwrites")
    ids = clique_ids(p["clique_csv"])
    m = json.load(open(out / "manifest.json"))
    if m["slide_ids"] != ids:
        raise SystemExit("REFUSING: manifest slide_ids differ from clique")
    src_feat, src_slide = Path(p["published_patch_features"]), Path(p["published_slide_features"])
    pub_counts = {}
    for f in sorted(src_slide.glob("*.h5")):
        if f.stem in ids:
            continue
        with h5py.File(src_feat / f"{f.stem}.h5", "r") as h:
            pub_counts[f.stem] = int(h["features"].shape[0])
    chosen = select_controls(pub_counts, [m["slides"][s]["n_patches"] for s in ids])
    cids = [c["slide"] for c in chosen]
    slides = {}
    for c in chosen:
        s = c["slide"]
        v = read_vec(src_slide / f"{s}.h5")
        if v.shape != (768,) or not np.isfinite(v).all() or not v.any():
            raise SystemExit(f"REFUSING: published control embedding {s} is not a finite non-zero 768-d vector")
        w = Path(p["wsi_src"]) / f"{s}.svs"
        if not w.exists():
            raise SystemExit(f"REFUSING: WSI {w} missing")
        slides[s] = {**c, "published_slide_embedding": str(src_slide / f"{s}.h5"),
                     "published_slide_embedding_sha256": sha256(src_slide / f"{s}.h5"),
                     "patch_features_source": str(src_feat / f"{s}.h5"), "wsi": str(w),
                     "wsi_resolved": os.path.realpath(w)}
    C["feat"].mkdir(parents=True)
    C["wsi"].mkdir()
    for s in cids:
        shutil.copyfile(slides[s]["patch_features_source"], C["feat"] / f"{s}.h5")
        os.symlink(slides[s]["wsi"], C["wsi"] / f"{s}.svs")
        a, b = sha256(slides[s]["patch_features_source"]), sha256(C["feat"] / f"{s}.h5")
        if a != b:
            raise SystemExit(f"copy mismatch for {s}")
        slides[s].update(patch_features_copy=str(C["feat"] / f"{s}.h5"), patch_features_sha256=b,
                         wsi_symlink=str(C["wsi"] / f"{s}.svs"))
    cmd = trident_command(str(C["wsi"]), str(C["job"]))
    man = {"purpose": "reproducibility controls: regenerate published non-zero TITAN embeddings in the same "
                      "environment as the 21-slide repair; never included in the repaired matrix",
           "selection_rule": "for each of min/median/max patch count of the 21 regenerated slides, the published "
                             "non-zero slide with nearest patch count (features_conch_v15 rows); ties -> lowest "
                             "lexical slide id; each slide chosen once",
           "n_published_candidates": len(pub_counts), "slide_ids": cids, "n_slides": len(cids),
           "job_dir": str(C["job"]), "wsi_dir": str(C["wsi"]), "expected_output_dir": str(C["regen"]),
           "trident_command": cmd, "trident_command_executed_by_this_script": False, "slides": slides,
           "pass_rule": f"L1(new-published) <= {CTRL_L1_REL} * published L1 norm AND < {CTRL_NN_REL} * the slide's "
                        "published nearest-neighbour L1 distance (published matrix, non-zero columns)",
           "interpretation": INTERPRETATION,
           "timestamp": datetime.datetime.now().astimezone().isoformat(),
           "command": " ".join([sys.executable] + sys.argv),
           "script": {"path": os.path.abspath(__file__), "sha256": sha256(__file__)},
           "main_manifest_sha256": sha256(out / "manifest.json"), "runtime_s": round(time.time() - t0, 1)}
    with open(C["manifest"], "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps({s: {k: slides[s][k] for k in ("target", "target_patch_count", "n_patches",
                                                    "patch_features_sha256", "published_slide_embedding_sha256")}
                      for s in cids}, indent=1))
    print("controls_manifest_sha256", sha256(C["manifest"]))
    return 0


def command(p):
    out = Path(p["outdir"])
    C = ctrl_paths(out)
    for f in (out / "manifest.json", C["manifest"]):
        if not f.exists():
            sys.exit(f"REFUSING: {f} missing (run prepare and stage-controls first)")
    m, cm = json.load(open(out / "manifest.json")), json.load(open(C["manifest"]))
    main_cmd, ctrl_cmd = corrected_command(m), corrected_command(cm)
    rec = {"main_command": main_cmd, "controls_command": ctrl_cmd,
           "manifest_sha256": sha256(out / "manifest.json"), "controls_manifest_sha256": sha256(C["manifest"]),
           "frozen_manifest_command": m["trident_command"],
           "changes_vs_frozen_manifest": ["removed --skip_errors", "prefixed HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1",
                                          "added --gpu 0"],
           "executed_by_this_script": False}
    print("# main (21 slides)\n" + main_cmd + "\n# controls (3 slides)\n" + ctrl_cmd)
    path = out / "command_record.json"
    if path.exists():
        old = json.load(open(path))
        if any(old.get(k) != rec[k] for k in ("main_command", "controls_command", "manifest_sha256",
                                              "controls_manifest_sha256")):
            sys.exit(f"REFUSING: {path} exists with different content")
        return 0
    rec.update(timestamp=datetime.datetime.now().astimezone().isoformat(),
               command=" ".join([sys.executable] + sys.argv),
               script={"path": os.path.abspath(__file__), "sha256": sha256(__file__)})
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    return 0


def check_controls(p):
    t0 = time.time()
    out = Path(p["outdir"])
    C = ctrl_paths(out)
    if not C["manifest"].exists():
        sys.exit(f"REFUSING: {C['manifest']} missing")
    if C["record"].exists():
        sys.exit(f"REFUSING: {C['record']} exists")
    cm = json.load(open(C["manifest"]))
    cids, ids = cm["slide_ids"], clique_ids(p["clique_csv"])
    gpu = parse_gpu_info(C["job"] / "gpu_info.txt")
    bad = []
    for s in cids:
        st, pe = C["feat"] / f"{s}.h5", Path(p["published_slide_features"]) / f"{s}.h5"
        if not st.is_file() or sha256(st) != cm["slides"][s]["patch_features_sha256"]:
            bad.append(f"{s}: staged CONCH copy")
        if not pe.is_file() or sha256(pe) != cm["slides"][s]["published_slide_embedding_sha256"]:
            bad.append(f"{s}: published slide embedding")
    if bad:
        raise SystemExit(f"REFUSING: control inputs differ from {C['manifest']}: {bad}")
    integrity = {"locks": lock_files(C["job"]), "staged_and_published_sha256_match_manifest": True}
    try:
        check_trident_config(C["job"])
        integrity["config_ok"] = True
    except SystemExit as e:
        integrity["config_ok"], integrity["config_error"] = False, str(e)
    pub = pd.read_pickle(p["published_matrix"])
    pidx = [str(x) for x in pub.index]
    nonzero_cols = np.array([s not in set(ids) for s in pidx])
    per = []
    for s in cids:
        r = {"slide": s}
        f = C["regen"] / f"{s}.h5"
        pv = read_vec(Path(p["published_slide_features"]) / f"{s}.h5")
        if not f.exists():
            per.append({**r, "passed": False, "error": "missing"})
            continue
        nv = read_vec(f)
        if nv.shape != pv.shape or not np.isfinite(nv).all():
            per.append({**r, "passed": False, "error": f"shape {nv.shape} or non-finite"})
            continue
        try:
            check_coords(C["regen"], C["feat"], [s])
            coords_ok = True
        except SystemExit as e:
            coords_ok, r["coords_error"] = False, str(e)
        row = pub.to_numpy()[pidx.index(s)]
        row = row[nonzero_cols & (row < SENTINEL_MIN)]
        l1, norm, nn = float(np.abs(nv - pv).sum()), float(np.abs(pv).sum()), float(row.min())
        den = np.linalg.norm(nv) * np.linalg.norm(pv)
        r.update(sha256=sha256(f), mtime=iso_mtime(f), l1_diff=l1, published_l1_norm=norm,
                 published_nn_distance=nn, l1_rel_norm=l1 / norm, l1_rel_nn=l1 / nn,
                 cosine=float(nv @ pv / den) if den > 0 else None, coords_ok=coords_ok,
                 passed=bool(coords_ok and l1 <= CTRL_L1_REL * norm and l1 < CTRL_NN_REL * nn))
        per.append(r)
    all_passed = bool(len(per) == len(cids) == 3 and all(r["passed"] for r in per)
                      and not integrity["locks"] and integrity["config_ok"])
    rec = {"action": "check-controls", "timestamp": datetime.datetime.now().astimezone().isoformat(),
           "command": " ".join([sys.executable] + sys.argv),
           "script": {"path": os.path.abspath(__file__), "sha256": sha256(__file__)},
           "controls_manifest_sha256": sha256(C["manifest"]), "control_slide_ids": cids,
           "gpu_info": gpu, "gpu_hostname": gpu["hostname"], "gpu_model": gpu["model"],
           "pass_rule": cm.get("pass_rule"), "per_control": per, "integrity": integrity,
           "provenance": provenance(C["job"]), "all_passed": all_passed,
           "verdict": "controls reproduce" if all_passed else "controls deviate (or incomplete): do not mix",
           "interpretation": INTERPRETATION, "runtime_s": round(time.time() - t0, 1)}
    with open(C["record"], "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(json.dumps({"all_passed": all_passed, "per_control": per}, indent=1, default=str))
    return 0 if all_passed else 1


def prepare(p):
    t0 = time.time()
    out = Path(p["outdir"])
    if out.exists():
        sys.exit(f"REFUSING: {out} exists; prepare never overwrites")
    ids = clique_ids(p["clique_csv"])
    src_feat, src_slide = Path(p["published_patch_features"]), Path(p["published_slide_features"])
    pre = {}
    for s in ids:
        v = read_vec(src_slide / f"{s}.h5")
        if v.shape != (768,) or v.any():
            raise SystemExit(f"REFUSING: published embedding {s} is not an all-zero 768-d vector")
        f = src_feat / f"{s}.h5"
        with h5py.File(f, "r") as h:
            feats, coords = h["features"], h["coords"]
            info = {"n_patches": int(feats.shape[0]), "feature_dim": int(feats.shape[1]),
                    "coords_shape": list(coords.shape),
                    "patch_size_level0": int(coords.attrs.get("patch_size_level0", -1)),
                    "features_nonzero": bool(np.asarray(feats[:]).any()),
                    "features_finite": bool(np.isfinite(feats[:]).all())}
        if not (info["features_nonzero"] and info["features_finite"] and info["feature_dim"] == 768
                and info["coords_shape"] == [info["n_patches"], 2]):
            raise SystemExit(f"REFUSING: published CONCH features for {s} not usable: {info}")
        w = Path(p["wsi_src"]) / f"{s}.svs"
        if not w.exists():
            raise SystemExit(f"REFUSING: WSI {w} missing")
        pre[s] = {"published_slide_embedding": str(src_slide / f"{s}.h5"),
                  "published_slide_embedding_sha256": sha256(src_slide / f"{s}.h5"),
                  "patch_features_source": str(f), **info, "wsi": str(w), "wsi_resolved": os.path.realpath(w)}
    job = out / "job_dir"
    feat_dir, wsi_dir = job / COORDS / "features_conch_v15", out / "wsi_dir"
    feat_dir.mkdir(parents=True)
    wsi_dir.mkdir()
    for s in ids:
        shutil.copyfile(pre[s]["patch_features_source"], feat_dir / f"{s}.h5")
        os.symlink(pre[s]["wsi"], wsi_dir / f"{s}.svs")
        a, b = sha256(pre[s]["patch_features_source"]), sha256(feat_dir / f"{s}.h5")
        if a != b:
            raise SystemExit(f"copy mismatch for {s}")
        pre[s].update(patch_features_copy=str(feat_dir / f"{s}.h5"), patch_features_sha256=b,
                      wsi_symlink=str(wsi_dir / f"{s}.svs"))
    cmd = trident_command(str(wsi_dir), str(job))
    manifest = {"slide_ids": ids, "n_slides": len(ids), "job_dir": str(job), "wsi_dir": str(wsi_dir),
                "expected_output_dir": str(job / COORDS / "slide_features_titan"),
                "trident_command": cmd, "trident_command_executed_by_this_script": False,
                "slides": pre,
                "notes": ["patches/<slide>_patches.h5 not staged: with all 21 features_conch_v15/<slide>.h5 present "
                          "the slide-encoder path does not run patch extraction; coords are read from the feature h5.",
                          "wsi_dir holds symlinks only; trident enumerates it (os.listdir, extension filter) and "
                          "constructs lazy WSI objects; the slide-encoder path does not read pixels."]}
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    rec = {"action": "prepare", "timestamp": datetime.datetime.now().astimezone().isoformat(),
           "command": " ".join([sys.executable] + sys.argv),
           "script": {"path": os.path.abspath(__file__), "sha256": sha256(__file__)},
           "inputs": {k: p[k] for k in ("published_slide_features", "published_patch_features", "clique_csv", "wsi_src")},
           "clique_csv_sha256": sha256(p["clique_csv"]),
           "n_slides": len(ids), "all_published_embeddings_zero": True,
           "n_patches_total": sum(v["n_patches"] for v in pre.values()),
           "n_patches_min_max": [min(v["n_patches"] for v in pre.values()), max(v["n_patches"] for v in pre.values())],
           "patch_size_level0_values": sorted({v["patch_size_level0"] for v in pre.values()}),
           "n_feature_copies_sha_verified": len(ids), "n_wsi_symlinks": len(ids),
           "manifest_sha256": sha256(out / "manifest.json"), "runtime_s": round(time.time() - t0, 1)}
    with open(out / "prepare_record.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in ("n_slides", "n_patches_total", "n_patches_min_max",
                                          "patch_size_level0_values", "manifest_sha256")}))
    print(cmd)
    return 0


def check_regenerated(regen_dir, ids):
    """Raise SystemExit unless all regenerated embeddings exist and are 768-d, finite, non-zero."""
    bad, vecs = [], {}
    for s in ids:
        f = Path(regen_dir) / f"{s}.h5"
        if not f.exists():
            bad.append(f"{s}: missing")
            continue
        v = read_vec(f)
        if v.shape != (768,):
            bad.append(f"{s}: shape {v.shape}")
        elif not np.isfinite(v).all():
            bad.append(f"{s}: non-finite")
        elif not v.any():
            bad.append(f"{s}: all-zero")
        else:
            vecs[s] = v
    if bad:
        raise SystemExit("REFUSING: regenerated embeddings not usable: " + "; ".join(bad))
    return vecs


def published_nn(b, keep):
    """Per-slide nearest-neighbour L1 distance within the published non-zero block (self and same-patient masked)."""
    blk = b[np.ix_(keep, keep)]
    return np.nanmin(np.where(blk < SENTINEL_MIN, blk, np.nan), axis=1)


def check_same_environment(job, C, crec):
    """Raise SystemExit unless the controls run (control_record.json + controls/manifest.json + controls gpu_info)
    and the main run share GPU model, trident HEAD/diff sha256 and TITAN refs/main (+ blob). Returns the record."""
    if not C["manifest"].is_file():
        raise SystemExit(f"REFUSING: {C['manifest']} missing")
    cman_sha = sha256(C["manifest"])
    if crec.get("controls_manifest_sha256") != cman_sha:
        raise SystemExit(f"REFUSING: control_record controls_manifest_sha256 "
                         f"{crec.get('controls_manifest_sha256')} != actual {C['manifest']} sha256 {cman_sha}")
    if [str(x) for x in json.load(open(C["manifest"]))["slide_ids"]] != [str(x) for x in crec["control_slide_ids"]]:
        raise SystemExit("REFUSING: control_record control_slide_ids differ from controls/manifest.json")
    g_main, g_ctrl = parse_gpu_info(Path(job) / "gpu_info.txt"), parse_gpu_info(C["job"] / "gpu_info.txt")
    if g_main["model"] != g_ctrl["model"]:
        raise SystemExit(f"REFUSING: GPU model differs: main {g_main['model']!r} vs controls {g_ctrl['model']!r}")
    rg = crec.get("gpu_info") or {}
    if rg.get("file_sha256") != g_ctrl["file_sha256"] or rg.get("model") != g_ctrl["model"]:
        raise SystemExit("REFUSING: control_record gpu_info does not match controls/job_dir/gpu_info.txt")
    now, rec_sw = software_provenance(), crec.get("provenance") or {}
    sw = {}
    for a, b in SW_KEYS:
        r, n = (rec_sw.get(a) or {}).get(b), (now.get(a) or {}).get(b)
        if r is None or n is None or r != n:
            raise SystemExit(f"REFUSING: {a}.{b} differs between control_record ({r!r}) and rebuild time ({n!r})")
        sw[f"{a}.{b}"] = n
    return {"gpu_model": g_main["model"], "main_gpu_info": g_main, "controls_gpu_info": g_ctrl,
            "driver_versions_equal": g_main["driver_versions"] == g_ctrl["driver_versions"],
            "software_equal_controls_and_rebuild_time": sw, "controls_manifest_sha256": cman_sha}


def rebuild(p, dry_run=False):
    t0 = time.time()
    out = Path(p["outdir"])
    ids = clique_ids(p["clique_csv"])
    job = out / "job_dir"
    regen_dir, staged_dir = job / COORDS / "slide_features_titan", job / COORDS / "features_conch_v15"
    C = ctrl_paths(out)
    if dry_run:
        need = [regen_dir / f"{s}.h5" for s in ids] + [Path(p[k]) for k in
                ("published_slide_features", "published_matrix", "metadata")] + [
                out / "manifest.json", job / COORDS / SLIDE_CONFIG, C["record"], C["manifest"],
                job / "gpu_info.txt", C["job"] / "gpu_info.txt"]
        missing = [str(f) for f in need if not f.exists()]
        print(json.dumps({"dry_run": True, "n_required": len(need), "missing": missing}, indent=1))
        return 1 if missing else 0
    out_pkl, out_rec = out / f"{BASE}_no_same_patient.pkl", out / f"_{BASE}_record.json"
    out_sum, out_recon = out / "regenerated_slide_summary.csv", out / "reconciliation.json"
    finals = (out_pkl, out_rec, out_sum, out_recon)
    cand = {f: f.with_name(f.name + ".candidate") for f in finals}
    for f in finals:
        for g in (f, cand[f], f.with_name(f.name + ".FAILED")):
            if g.exists():
                sys.exit(f"REFUSING: {g} exists")

    # ---- input gates (SystemExit before anything is written) ----
    if not C["record"].exists():
        raise SystemExit(f"REFUSING: {C['record']} missing; run the controls job and check-controls first")
    crec = json.load(open(C["record"]))
    if crec.get("all_passed") is not True:
        raise SystemExit(f"REFUSING: {C['record']} all_passed is not true; controls deviate -> do not mix")
    ctrl_ids = [str(x) for x in crec.get("control_slide_ids", [])]
    if set(ctrl_ids) & set(ids):
        raise SystemExit(f"REFUSING: control ids overlap the 21: {sorted(set(ctrl_ids) & set(ids))}")
    env = check_same_environment(job, C, crec)
    locks = lock_files(job)
    if locks:
        raise SystemExit(f"REFUSING: leftover trident lock files: {locks}")
    man_path = out / "manifest.json"
    man = json.load(open(man_path))
    if man["slide_ids"] != ids:
        raise SystemExit("REFUSING: manifest slide_ids differ from clique")
    prep = out / "prepare_record.json"
    if prep.exists() and json.load(open(prep)).get("manifest_sha256") != sha256(man_path):
        raise SystemExit("REFUSING: manifest.json sha256 differs from prepare_record.json")
    staged = {s: sha256(staged_dir / f"{s}.h5") for s in ids}
    bad = [s for s in ids if staged[s] != man["slides"][s]["patch_features_sha256"]]
    if bad:
        raise SystemExit(f"REFUSING: staged CONCH copies differ from manifest.json: {bad}")
    check_trident_config(job)
    new = check_regenerated(regen_dir, ids)
    extra = sorted({f.stem for f in regen_dir.glob("*.h5")} - set(ids))
    if extra:
        raise SystemExit(f"REFUSING: unexpected slide embeddings in {regen_dir}: {extra}")
    check_coords(regen_dir, staged_dir, ids)

    pub_ids, X = load_slide_embeddings(p["published_slide_features"])
    zero = all_zero_ids(pub_ids, X)
    if sorted(zero) != ids or len(pub_ids) != N_ALL:
        raise SystemExit(f"REFUSING: published set has {len(pub_ids)} slides, zero ids {zero} != clique")
    pos = {s: i for i, s in enumerate(pub_ids)}
    for s in ids:
        X[pos[s]] = new[s]
    raw = _distance_frame(pub_ids, X, "cityblock")
    M, n_pairs = _mask_same_patient(raw, p["metadata"])
    try:
        with open(cand[out_pkl], "wb") as fh:
            pickle.dump(M, fh)
        finite = raw.to_numpy()[raw.to_numpy() < SENTINEL]
        record = {"action": "matrix", "slide_features_dir": f"{p['published_slide_features']} (2,900 non-zero) + "
                  f"{regen_dir} (21 regenerated)", "n_slides": len(pub_ids), "feature_dim": int(X.shape[1]),
                  "metric": "cityblock", "model": "titan", "method": "titan", "tumor_only": False,
                  "same_patient_pairs_masked": n_pairs, "max_real_distance": float(finite.max()), "out": str(out_pkl)}
        with open(cand[out_rec], "w") as f:
            json.dump(record, f, indent=1)

        checks = {}

        def check(name, ok, **info):
            checks[name] = {"result": "PASS" if ok else "FAIL", **info}
            print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(info, default=str)[:300]}")

        pub = pd.read_pickle(p["published_matrix"])
        a, b = M.to_numpy(), pub.to_numpy()
        lay = list(M.index) == list(pub.index) and list(M.columns) == list(pub.columns)
        keep = np.array([s not in set(ids) for s in pub_ids])
        check("published_2900_block_bitwise_identical", lay and np.array_equal(a[np.ix_(keep, keep)], b[np.ix_(keep, keep)]),
              same_layout=lay, n_kept=int(keep.sum()))
        check("masked_pairs_equal_published", n_pairs == int(masked_offdiag(b).sum() // 2) == N_MASKED
              and np.array_equal(masked_offdiag(a), masked_offdiag(b)), masked_pairs=n_pairs)
        off = ~np.eye(len(a), dtype=bool)
        check("symmetric_diag_finite", bool(np.array_equal(a, a.T) and (np.diag(a) == SENTINEL).all()
                                            and np.isfinite(a).all()))
        zpairs = int((a[off] == 0).sum() // 2)
        check("no_zero_distance_pairs", zpairs == 0, zero_offdiag_pairs=zpairs)
        check("controls_not_in_matrix", not (set(ctrl_ids) & set(ids)) and len(new) == N_REGEN,
              control_slide_ids=ctrl_ids, note="controls run in controls/job_dir; only the 21 clique rows are replaced")

        meta = pd.read_csv(p["metadata"], dtype=str).drop_duplicates("Slide.Scan.File").set_index("Slide.Scan.File")
        per = []
        for s in ids:
            i = pos[s]
            row = np.where(a[i] < SENTINEL_MIN, a[i], np.nan)
            j = int(np.nanargmin(row))
            per.append({"slide": s, "group": meta.at[s, "Tumor.group"], "l1_norm": float(np.abs(X[i]).sum()),
                        "std": float(np.std(X[i])),
                        "nn_slide": pub_ids[j], "nn_group": meta.at[pub_ids[j], "Tumor.group"],
                        "nn_same_group": meta.at[pub_ids[j], "Tumor.group"] == meta.at[s, "Tumor.group"],
                        "nn_is_regenerated": pub_ids[j] in set(ids),
                        "min_distance": float(np.nanmin(row)), "median_distance": float(np.nanmedian(row))})
        per = pd.DataFrame(per)
        per.to_csv(cand[out_sum], index=False)
        norms_old = np.abs(X[keep]).sum(axis=1)
        q = lambda v: {k: float(np.percentile(v, k)) for k in (0, 5, 25, 50, 75, 95, 100)}
        norm_cmp = {"regenerated_21": q(per["l1_norm"].to_numpy()), "published_2900": q(norms_old),
                    "regenerated_percentile_ranks_in_published": [float((norms_old < x).mean()) for x in per["l1_norm"]]}
        check("regenerated_norms_within_published_range",
              bool(per["l1_norm"].between(norms_old.min(), norms_old.max()).all()), **norm_cmp)

        # ---- degeneracy checks (thresholds computed from the published block, not hard-coded) ----
        nn = published_nn(b, keep)
        nn_stats = {"min": float(nn.min()), "p5": float(np.percentile(nn, 5)), "mean": float(nn.mean()),
                    "median": float(np.median(nn)), "n": int(len(nn))}
        check(f"regenerated_std_ge_{MIN_STD}", bool((per["std"] >= MIN_STD).all()),
              min_std=float(per["std"].min()), failing=per.loc[per["std"] < MIN_STD, "slide"].tolist())
        check("regenerated_nn_ge_published_min_nn", bool((per["min_distance"] >= nn_stats["min"]).all()),
              published_min_nn=nn_stats["min"], min_regenerated_nn=float(per["min_distance"].min()),
              failing=per.loc[per["min_distance"] < nn_stats["min"], "slide"].tolist(),
              rule="NN over all 2,921 slides excluding self and same-patient")
        ri = [pos[s] for s in ids]
        rr = raw.to_numpy()[np.ix_(ri, ri)][np.triu_indices(len(ri), 1)]
        med = float(np.median(rr))
        check("regenerated_median_pairwise_ge_published_p5_nn", med >= nn_stats["p5"], median_pairwise_l1=med,
              published_p5_nn=nn_stats["p5"], rule="median of unmasked L1 over all 210 pairs among the 21")
        thr = DUP_REL * nn_stats["mean"]
        tri = np.triu_indices(len(ri), 1)
        dups = [[ids[x], ids[y], float(d)] for x, y, d in zip(*tri, rr) if d < thr]
        check("no_near_duplicate_regenerated_pairs", not dups, threshold=thr, min_pairwise_l1=float(rr.min()),
              duplicates=dups, rule=f"unmasked L1 between any two of the 21 < {DUP_REL} x mean published NN distance")

        rec = {"action": "rebuild", "timestamp": datetime.datetime.now().astimezone().isoformat(),
               "command": " ".join([sys.executable] + sys.argv),
               "script": {"path": os.path.abspath(__file__), "sha256": sha256(__file__)},
               "inputs": {"published_matrix": {"path": p["published_matrix"], "sha256": sha256(p["published_matrix"])},
                          "metadata": {"path": p["metadata"], "sha256": sha256(p["metadata"])},
                          "manifest_sha256": sha256(man_path), "staged_conch_copies_sha256": staged,
                          "control_record": {"path": str(C["record"]), "sha256": sha256(C["record"])},
                          "command_record_sha256": sha256(out / "command_record.json")
                          if (out / "command_record.json").exists() else None,
                          "regenerated": {s: {"sha256": sha256(regen_dir / f"{s}.h5"),
                                              "mtime": iso_mtime(regen_dir / f"{s}.h5")} for s in ids}},
               "provenance": provenance(job), "environment_consistency": env, "published_nn_stats": nn_stats,
               "checks": checks, "per_slide": per.to_dict(orient="records"),
               "outputs": {f.name: sha256(cand[f]) for f in (out_pkl, out_rec, out_sum)}}
        rec["all_passed"] = all(c["result"] == "PASS" for c in checks.values())
        rec["interpretation"] = (INTERPRETATION["controls_reproduce_and_21_nonzero"] if rec["all_passed"] else
                                 "rebuild checks failed; outputs renamed *.FAILED and must not be used")
        rec["runtime_s"] = round(time.time() - t0, 1)
        with open(cand[out_recon], "w") as f:
            json.dump(rec, f, indent=1, default=str)
        for f in finals:
            os.replace(cand[f], f if rec["all_passed"] else f.with_name(f.name + ".FAILED"))
    except BaseException:
        for f in finals:                     # any partial output becomes failure evidence, never a final name
            for g in (cand[f], f):
                if g.exists():
                    os.replace(g, f.with_name(f.name + ".FAILED"))
        print("REBUILD ABORTED: partial outputs renamed *.FAILED", file=sys.stderr)
        raise
    if not rec["all_passed"]:
        print("REBUILD FAILED: outputs renamed *.FAILED")
    return 0 if rec["all_passed"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prepare", "stage-controls", "command", "check-controls"):
        sub.add_parser(c)
    rb = sub.add_parser("rebuild")
    rb.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    out_root = os.environ.get("PROSPER_OUT")
    if not out_root:
        sys.exit("export PROSPER_OUT first")
    p = default_paths(os.getcwd(), out_root)
    if args.cmd == "rebuild":
        return rebuild(p, args.dry_run)
    return {"prepare": prepare, "stage-controls": stage_controls, "command": command,
            "check-controls": check_controls}[args.cmd](p)


if __name__ == "__main__":
    sys.exit(main())
