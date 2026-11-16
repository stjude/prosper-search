#!/usr/bin/env python3
"""R2.1-TF2 stage S1: GPU job SPECIFICATIONS (no submission) and the canary / output checks.

Binding specification: r2_1_filter_tf2_20261016/contract_candidate_003/SPECIFICATION.md §4 S1 (S1a H-Optimus-1
all-patches features of the labelled non-library slides via the accepted port; S1c CONCH v1.5
1024-px features for pool slides absent from the trident root; no Virchow2 re-extraction [F4]) +
contract_candidate_004 R3 (canary rule: per overlap slide >= 99 % of feature rows bitwise identical
to the existing bank, and for the remaining rows max |delta| <= 2e-3 x max |value| of that row;
report the bitwise fraction and per-row relative maxima; STOP only above the rule).

Subcommands
  write          write job specifications (LSF scripts with a queue placeholder, WSI symlink
                 directories, slide lists / mosaic pickles, jobspec.json). The analyst submits them
                 through the lsf-max-throughput skill; this script never calls bsub.
  canary-check   apply the R3 rule to a canary output directory against the existing bank.
  verify-outputs check that every expected slide was written with the expected grid/rows.

Writes only under $PROSPER_OUT/r2_1_filter_tf2_20261016/s1 (destinations must be new or empty).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import pickle
import sys
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
STAGE = "s1"
CANARY_N = 5
CANARY_NAMESPACE = "r2_1_tf2_s1_canary_v1"
BITWISE_FRACTION = 0.99
REL_LIMIT = 2e-3
PYTHON = "/HOME/miniconda3/bin/python3"
TRIDENT_PYTHON = "/HOME/miniconda3/envs/trident/bin/python"
CONCH_CHUNKS = 5
require = S0.require
now = S0.now


def canary_slides(library_pool_slides, n=CANARY_N):
    """Deterministic canary choice: first n library pool slides in sha256(namespace|slide) order.
    (Not specified by the specification; the analyst may override with --canary-slides.)"""
    return sorted(library_pool_slides,
                  key=lambda s: hashlib.sha256(f"{CANARY_NAMESPACE}|{s}".encode()).hexdigest())[:n]


def chunks(seq, k):
    seq = list(seq)
    size = -(-len(seq) // k)
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def symlink_dir(d: Path, slides, src_dir: Path):
    d.mkdir(parents=True)
    out = []
    for s in slides:
        src = src_dir / f"{s}.svs"
        require(src.exists(), f"WSI missing: {src}")
        target = src.resolve(strict=True)
        os.symlink(target, d / f"{s}.svs")
        out.append({"slide_id": s, "symlink": str(d / f"{s}.svs"), "target": str(target),
                    "size": target.stat().st_size})
    return out


def lsf_gpu(name, body, logs, *, slots=16, mem_per_slot_mb=2048, walltime=60, array=None):
    jname = f"{name}[{array}]" if array else name
    return f"""#!/bin/bash
# R2.1-TF2 S1 GPU job: {name}. NOT SUBMITTED BY THE IMPLEMENTER. The lead submits through the
# lsf-max-throughput skill (queue/GPU type chosen and probed there; GPU type recorded per output).
# Bounds (contract §4 S1): <= 1 GPU / 16 slots / 60 min per element. rusage mem is per slot here.
#BSUB -J {jname}
#BSUB -q <GPU_QUEUE_FROM_lsf-max-throughput>
#BSUB -n {slots}
#BSUB -gpu "num=1/host"
#BSUB -R "select[mem>={mem_per_slot_mb}] rusage[mem={mem_per_slot_mb}] span[hosts=1]"
#BSUB -M {mem_per_slot_mb}
#BSUB -W {walltime}
#BSUB -oo {logs}/{name}_%I.%J.out
#BSUB -eo {logs}/{name}_%I.%J.err
set -euo pipefail
export LSB_JOB_REPORT_MAIL=N
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee {logs}/{name}_${{LSB_JOBINDEX:-0}}.${{LSB_JOBID}}.gpu.txt
{body}
"""


def write_specs(s0_dir: Path, out: Path, prosper_out: str, canary=None, inputs=None) -> dict:
    import pandas as pd
    complete = json.loads((s0_dir / "COMPLETE.json").read_text())
    require(complete["status"] == "COMPLETE", "S0 is not COMPLETE")
    for name, e in complete["outputs"].items():
        require(S0.sha256_file(s0_dir / name) == e["sha256"], f"S0 output {name} changed")
    inputs = inputs or S0.default_inputs(S0.HAORAN, prosper_out)
    pool = pd.read_csv(s0_dir / "pool.csv", dtype={"slide_id": str})
    pooled = pool[pool["in_pool"] == 1]
    nonlib = sorted(pooled.loc[pooled["in_final_library"] == 0, "slide_id"], key=S0.slide_sort_key)
    libpool = sorted(pooled.loc[pooled["in_final_library"] == 1, "slide_id"], key=S0.slide_sort_key)
    s1 = S0.stage_root(prosper_out) / STAGE
    # what is actually missing
    ho_missing = [s for s in pooled["slide_id"] if not (Path(inputs["ann_h_optimus_1_dir"]) / f"{s}.pkl").exists()]
    tr = Path(inputs["trident_root"])
    conch_missing = [s for s in pooled["slide_id"]
                     if not ((tr / "features_conch_v15" / f"{s}.h5").exists() and (tr / "patches" / f"{s}_patches.h5").exists())]
    ho_missing = sorted(ho_missing, key=S0.slide_sort_key)
    conch_missing = sorted(conch_missing, key=S0.slide_sort_key)
    canary = list(canary or canary_slides(libpool))
    require(set(canary) <= set(libpool), "canary slides must be library pool slides present in both banks")
    logs = s1 / "logs"
    out.mkdir(parents=True, exist_ok=True)
    tiles = {s: int(len(pd.read_csv(Path(inputs["patch_dir"]) / f"{s}.csv", usecols=["wsi_loc_x"])))
             for s in set(ho_missing) | set(canary)}
    # ---- S1a H-Optimus-1 (accepted port: prosper.embeddings.extract_cli, all_patches, fp16 autocast, batch 256)
    wsi_non = symlink_dir(out / "wsi_nonlibrary", ho_missing, Path(inputs["wsi_nonlibrary_dir"]))
    wsi_can = symlink_dir(out / "wsi_canary", canary, Path(inputs["wsi_library_dir"]))
    for name, slides in (("mosaic_h_optimus_1_nonlibrary.pkl", ho_missing),
                         ("mosaic_h_optimus_1_canary.pkl", canary)):
        with open(out / name, "wb") as fh:
            pickle.dump([(f"{s}.svs", []) for s in sorted(slides)], fh, protocol=4)

    def ho_body(mosaic, wsi_dir, task_root, task):
        return (f"cd {S0.PROSPER_SEARCH}\n"
                f"# PROSPER_OUT is overridden for this process only so that extract_cli's provenance\n"
                f"# (<PROSPER_OUT>/logs/extract_h_optimus_1/provenance_<tag>.json) and features stay inside s1/.\n"
                f"env -u LSB_JOBINDEX PROSPER_OUT={task_root} COMET_SLIDES_ROOT={wsi_dir} "
                f"{PYTHON} -m prosper.embeddings.extract_cli --encoder h_optimus_1 --mosaic {mosaic} "
                f"--tile-source all_patches --task {task} --batch-size 256 --num-workers 14")
    s1a_canary = lsf_gpu("tf2_s1a_canary", ho_body(out / "mosaic_h_optimus_1_canary.pkl", out / "wsi_canary",
                                                   s1 / "canary_h_optimus_1", "h_optimus_1_all_patches"), logs)
    s1a_prod = lsf_gpu("tf2_s1a_production", ho_body(out / "mosaic_h_optimus_1_nonlibrary.pkl", out / "wsi_nonlibrary",
                                                     s1, "h_optimus_1_all_patches_tf2"), logs)
    # ---- S1c CONCH v1.5 via trident (pinned config of the trident root: 40x, 1024 px, overlap 0,
    # min_tissue_proportion 0, hest segmenter defaults, holes kept); seg -> coords -> feat
    conch_chunks = chunks(conch_missing, CONCH_CHUNKS) if conch_missing else []
    chunk_dirs = []
    for k, ch in enumerate(conch_chunks, 1):
        symlink_dir(out / f"wsi_conch_chunk{k}", ch, Path(inputs["wsi_nonlibrary_dir"]))
        chunk_dirs.append({"chunk": k, "slides": ch, "wsi_dir": str(out / f"wsi_conch_chunk{k}"),
                           "job_dir": str(s1 / "conch_v15_tf2" / f"chunk{k}" / "job_dir")})
    symlink_dir(out / "wsi_conch_canary", canary, Path(inputs["wsi_library_dir"]))
    trident_repo = Path(inputs["trident_root"]).parents[1]

    def conch_body(wsi_dir, job_dir):
        return (f"cd {trident_repo}\n"
                f"HF_HUB_OFFLINE=1 {TRIDENT_PYTHON} run_batch_of_slides.py --task all "
                f"--wsi_dir {wsi_dir} --job_dir {job_dir} --patch_encoder conch_v15 --mag 40 "
                f"--patch_size 1024 --overlap 0 --min_tissue_proportion 0.0 --segmenter hest "
                f"--seg_conf_thresh 0.5 --coords_dir 40x_1024px_0px_overlap --batch_size 64 --gpu 0")
    s1c_canary = lsf_gpu("tf2_s1c_canary", conch_body(out / "wsi_conch_canary", s1 / "canary_conch_v15" / "job_dir"),
                         logs, slots=8)
    arr_body = ("CHUNK=${LSB_JOBINDEX}\n" + conch_body(f"{out}/wsi_conch_chunk${{CHUNK}}",
                                                       f"{s1}/conch_v15_tf2/chunk${{CHUNK}}/job_dir"))
    s1c_prod = lsf_gpu("tf2_s1c_production", arr_body, logs, slots=8,
                       array=f"1-{len(conch_chunks)}" if conch_chunks else None)
    for name, text in (("s1a_canary.lsf", s1a_canary), ("s1a_production.lsf", s1a_prod),
                       ("s1c_canary.lsf", s1c_canary), ("s1c_production.lsf", s1c_prod)):
        (out / name).write_text(text)
    spec = {
        "schema_version": 1, "created": now(), "s0": {"dir": str(s0_dir),
                                                     "complete_sha256": S0.sha256_file(s0_dir / "COMPLETE.json")},
        "contract": S0.CONTRACT_SHA,
        "submitted_by_implementer": False,
        "order": ["mkdir -p " + str(logs),
                  "S1a canary -> canary-check --bank h_optimus_1 (STOP if FAIL) -> S1a production -> verify-outputs",
                  "S1c canary -> canary-check --bank conch_v15 (STOP if FAIL) -> S1c production array -> verify-outputs",
                  "then (S0 script) labels-1024-supplement per CONCH chunk trident root; then S2 dry-run into a new plan dir"],
        "canary_rule": {"source": "contract_candidate_004 R3",
                        "per_slide_pass": f">= {BITWISE_FRACTION:.0%} of feature rows bitwise identical to the existing bank "
                                          f"AND for every remaining row max|delta| <= {REL_LIMIT} x max|value| of that row "
                                          "(row maximum of the existing bank's row)",
                        "reported": ["fraction of bitwise rows", "per-row relative maxima of non-bitwise rows"],
                        "grid": "the canary's coordinate list must equal the existing bank's (HO: all_patches order; CONCH: trident coords) "
                                "-- stated here because a different grid makes row comparison undefined",
                        "stop": "any canary slide failing -> STOP, no production submission",
                        "canary_slides": canary,
                        "canary_selection": f"first {CANARY_N} library pool slides by sha256('{CANARY_NAMESPACE}|'+slide); "
                                            "not specified by the contract (lead may override)",
                        "canary_tiles": {s: tiles[s] for s in canary}},
        "s1a_h_optimus_1": {
            "slides": ho_missing, "n_slides": len(ho_missing), "n_tiles": int(sum(tiles[s] for s in ho_missing)),
            "port": "prosper.embeddings.extract_cli (accepted all-patches bank command: --encoder h_optimus_1 "
                    "--tile-source all_patches --batch-size 256 --num-workers 14, fp16 autocast default, 16 slots)",
            "output_dir": str(s1 / "h_optimus_1_all_patches_tf2"),
            "canary_output_dir": str(s1 / "canary_h_optimus_1" / "h_optimus_1_all_patches"),
            "reference_bank": inputs["ann_h_optimus_1_dir"],
            "wsi_symlinks": wsi_non, "canary_wsi_symlinks": wsi_can,
            "scripts": ["s1a_canary.lsf", "s1a_production.lsf"],
            "resources": {"gpu": 1, "slots": 16, "mem_per_slot_mb": 2048, "walltime_min": 60, "elements": 1},
            "throughput_reference": "accepted all-patches run ~190 tiles/s on 16 slots",
            "notes": ["PROSPER_OUT is overridden for the extraction subprocess only (provenance JSON lands in s1/logs/, "
                      "not in the shared $PROSPER_OUT/logs/extract_h_optimus_1/ that holds accepted provenance_<n>.json)",
                      "LSB_JOBINDEX is unset for the subprocess so the provenance tag is '<lo>_<hi>' (no collision)"]},
        "s1c_conch_v15": {
            "slides": conch_missing, "n_slides": len(conch_missing), "chunks": chunk_dirs,
            "trident_repo": str(trident_repo), "python": TRIDENT_PYTHON,
            "pinned_config": {"source": [str(Path(inputs["trident_root"]) / "_config_coords.json"),
                                         str(Path(inputs["trident_root"]) / "_config_feats_conch_v15.json"),
                                         str(Path(inputs["trident_root"]).parent / "_config_segmentation.json")],
                              "target_magnification": 40, "patch_size": 1024, "overlap": 0,
                              "min_tissue_proportion": 0.0, "seg_mag": 10, "holes_are_tissue": True,
                              "segmenter": "hest (default)", "seg_conf_thresh": 0.5, "batch_size": 64},
            "canary_output_dir": str(s1 / "canary_conch_v15" / "job_dir" / "40x_1024px_0px_overlap"),
            "reference_root": inputs["trident_root"],
            "scripts": ["s1c_canary.lsf", "s1c_production.lsf"],
            "resources": {"gpu": 1, "slots": 8, "mem_per_slot_mb": 2048, "walltime_min": 60,
                          "elements": len(conch_chunks)},
            "notes": ["trident working tree is modified relative to its commit (recorded in the TITAN regen LEAD_ADDENDUM); "
                      "the canary is the fidelity check for this code state",
                      "segmentation is re-run for the canary slides; a different tissue grid fails the canary"]},
        "virchow2": "no re-extraction (F4); historical annotation bank used directly (F8 at S0)",
        "code": {k: {"path": str(v), "sha256": S0.sha256_file(v)} for k, v in
                 (("s1_script", Path(__file__).resolve()), ("s0_script", Path(S0.__file__).resolve()))},
    }
    S0.write_json(spec, out / "jobspec.json")
    return spec


# ---------------------------------------------------------------- canary rule (R3)
def compare_rows(ref: np.ndarray, new: np.ndarray):
    """R3: per-row bitwise identity and relative maxima (relative to the reference row max)."""
    require(ref.shape == new.shape, f"shape {ref.shape} != {new.shape}")
    bitwise = (ref == new).all(axis=1)
    d = np.abs(ref.astype(np.float64) - new.astype(np.float64)).max(axis=1)
    rowmax = np.abs(ref.astype(np.float64)).max(axis=1)
    rel = np.where(rowmax > 0, d / np.where(rowmax > 0, rowmax, 1), np.where(d > 0, np.inf, 0))
    frac = float(bitwise.mean()) if len(bitwise) else 0.0
    worst = float(rel[~bitwise].max()) if (~bitwise).any() else 0.0
    return {"rows": int(len(bitwise)), "bitwise_rows": int(bitwise.sum()), "bitwise_fraction": frac,
            "max_rel_non_bitwise": worst, "max_abs": float(d.max()) if len(d) else 0.0,
            "per_row_rel_non_bitwise": [float(x) for x in rel[~bitwise]],
            "pass": bool(len(bitwise) and frac >= BITWISE_FRACTION and worst <= REL_LIMIT)}


def load_ho(path):
    with open(path, "rb") as fh:
        X, tiles = pickle.load(fh)
    return np.asarray(X), [tuple(int(v) for v in t["wsi_loc"]) for t in tiles]


def load_conch(feat_path):
    import h5py
    with h5py.File(feat_path, "r") as h:
        return np.asarray(h["features"][:]), [tuple(int(v) for v in c) for c in h["coords"][:]]


def canary_check(bank, canary_dir: Path, reference: Path, slides, out: Path) -> dict:
    rows = []
    for s in slides:
        if bank == "h_optimus_1":
            n, nc = load_ho(canary_dir / f"{s}.pkl")
            r, rc = load_ho(reference / f"{s}.pkl")
        else:
            n, nc = load_conch(canary_dir / "features_conch_v15" / f"{s}.h5")
            r, rc = load_conch(reference / "features_conch_v15" / f"{s}.h5")
        rec = {"slide_id": s, "grid_identical": nc == rc, "canary_rows": len(nc), "reference_rows": len(rc)}
        if nc == rc:
            rec.update(compare_rows(r, n))
        else:
            rec.update({"pass": False, "coincident_coords": len(set(nc) & set(rc))})
        rows.append(rec)
    res = {"bank": bank, "rule": "contract_candidate_004 R3", "created": now(),
           "canary_dir": str(canary_dir), "reference": str(reference), "slides": rows,
           "pass": bool(rows) and all(r["pass"] for r in rows)}
    out.mkdir(parents=True, exist_ok=True)
    S0.write_json(res, out / (f"canary_{bank}.json" if res["pass"] else f"canary_{bank}_FAILED.json"))
    return res


def verify_outputs(bank, out_dirs, slides, inputs, out: Path) -> dict:
    import pandas as pd
    rows = []
    for s in slides:
        if bank == "h_optimus_1":
            hits = [Path(d) / f"{s}.pkl" for d in out_dirs if (Path(d) / f"{s}.pkl").exists()]
            rec = {"slide_id": s, "files": [str(h) for h in hits]}
            if len(hits) == 1:
                X, coords = load_ho(hits[0])
                ap = pd.read_csv(Path(inputs["patch_dir"]) / f"{s}.csv", usecols=["wsi_loc_x", "wsi_loc_y"])
                rec.update(rows=int(X.shape[0]), width=int(X.shape[1]), dtype=str(X.dtype),
                           order_equals_all_patches=coords == list(zip(ap["wsi_loc_x"], ap["wsi_loc_y"])),
                           finite=bool(np.isfinite(X).all()), sha256=S0.sha256_file(hits[0]))
                rec["pass"] = rec["order_equals_all_patches"] and rec["width"] == 1536 and rec["finite"] \
                    and rec["dtype"] == "float32"
            else:
                rec["pass"] = False
        else:
            hits = [Path(d) for d in out_dirs if (Path(d) / "features_conch_v15" / f"{s}.h5").exists()]
            rec = {"slide_id": s, "roots": [str(h) for h in hits]}
            if len(hits) == 1:
                import h5py
                X, coords = load_conch(hits[0] / "features_conch_v15" / f"{s}.h5")
                with h5py.File(hits[0] / "patches" / f"{s}_patches.h5", "r") as h:
                    pc = [tuple(int(v) for v in c) for c in h["coords"][:]]
                    attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in h["coords"].attrs.items()}
                rec.update(rows=int(X.shape[0]), width=int(X.shape[1]), coords_equal_patches=coords == pc,
                           patch_size_level0=int(attrs.get("patch_size_level0", -1)),
                           finite=bool(np.isfinite(X).all()),
                           sha256=S0.sha256_file(hits[0] / "features_conch_v15" / f"{s}.h5"))
                rec["pass"] = rec["coords_equal_patches"] and rec["width"] == 768 and rec["finite"] \
                    and rec["patch_size_level0"] == 1024
            else:
                rec["pass"] = False
        rows.append(rec)
    res = {"bank": bank, "created": now(), "slides": rows, "pass": all(r["pass"] for r in rows)}
    out.mkdir(parents=True, exist_ok=True)
    S0.write_json(res, out / (f"verify_{bank}.json" if res["pass"] else f"verify_{bank}_FAILED.json"))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--s0", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--canary-slides", nargs="*")
    c = sub.add_parser("canary-check")
    c.add_argument("--bank", choices=["h_optimus_1", "conch_v15"], required=True)
    c.add_argument("--canary-dir", required=True)
    c.add_argument("--reference", required=True)
    c.add_argument("--slides", nargs="+", required=True)
    c.add_argument("--out", required=True)
    v = sub.add_parser("verify-outputs")
    v.add_argument("--bank", choices=["h_optimus_1", "conch_v15"], required=True)
    v.add_argument("--dirs", nargs="+", required=True)
    v.add_argument("--slides", nargs="+", required=True)
    v.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prosper_out = os.environ.get("PROSPER_OUT")
    if not prosper_out:
        sys.exit("PROSPER_OUT must be exported")
    out = S0.check_out_dir(Path(a.out), prosper_out, STAGE)
    if a.cmd == "write":
        spec = write_specs(Path(a.s0), out, prosper_out, a.canary_slides)
        print(json.dumps({"s1a_slides": spec["s1a_h_optimus_1"]["n_slides"],
                          "s1c_slides": spec["s1c_conch_v15"]["n_slides"],
                          "canary": spec["canary_rule"]["canary_slides"]}))
        return 0
    if a.cmd == "canary-check":
        return 0 if canary_check(a.bank, Path(a.canary_dir), Path(a.reference), a.slides, out)["pass"] else 1
    inputs = S0.default_inputs(S0.HAORAN, prosper_out)
    return 0 if verify_outputs(a.bank, a.dirs, a.slides, inputs, out)["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
