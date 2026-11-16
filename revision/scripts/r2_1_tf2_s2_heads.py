#!/usr/bin/env python3
"""R2.1-TF2 stage S2: head plan (dry run), per-head fit driver, F7 checkpoint (ii), classifier
report and the F2 head-recipe regression.

Binding specification: r2_1_filter_tf2_20261016/contract_candidate_003/SPECIFICATION.md §3.3, §4 S2, §5.1,
§5.2, §5.4, §7 F2/F7 + contract_candidate_004 R1 (F7 checkpoint (ii)); release BRIEF.md.

Heads: 4 encoders (uni2, virchow2, h_optimus_1, conch_v15) x 2 budgets (100, 5) x
{fold0..fold4, full} = 48 primary + 8 sensitivity full heads without the 14 QC-removed slides
(slot full_noqc, §5.4 / S2-b) = 56.

Recipe: prosper.evaluation.r2_1_hoptimus_fit (imported, not copied or edited). That module
hard-codes _WIDTH = 1536; the input width is generalised by a thin wrapper (`fit_width`) that sets
the module constant for the duration of a call and restores it (Virchow2 1,280, CONCH v1.5 768;
UNI2 and H-Optimus-1 are 1,536, so their path is the unmodified module). Fold heads on 512-px
banks call `fit_head` itself; full heads and CONCH heads (no held-out fold / 1024-px records,
which `fit_head`'s manifest validator cannot represent) use `fit_generic`, which calls the same
module's `_parameters`, `_runtime_profile`, `_warnings_report`, `_state`, `_fingerprint` and
`FittedHead`; tests show it yields the identical fitted state on the same inputs.

Subcommands
  dry-run        validate S0 and inputs, write head_plan.json + tile-hash manifests + LSF array
                 spec + cmds.txt (one fit command per ready head, for the lsf-max-throughput pool
                 runner); --heads-root fixes where the fits will write (recorded in the plan and
                 enforced by `fit`); no fitting
  fit            fit ONE head (LSF array element); writes s2/heads/<encoder>/b<budget>/<slot>/
  f7-check       F7 checkpoint (ii) on the saved training-patient sets of every head (R1)
  report         classifier report §5.1, §5.2, §5.4 from completed heads
  f2-regression  F2: reproduce the accepted H-Optimus-1 fold-0 5 % head and the route-b UNI2
                 fold-0 5 % head (202-slide pool, accepted folds) through this file's fit path

Writes only under $PROSPER_OUT/r2_1_filter_tf2_20261016/s2 (destinations must be new or empty).
Never submits jobs, never runs model inference or retrieval.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"  # fit_head requires native pools of one; set before numpy is imported

import argparse
import collections
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import pickle
import socket
import sys
import time
import traceback
import warnings
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

from prosper.evaluation import r2_1_hoptimus_filter as kernel  # noqa: E402
from prosper.evaluation import r2_1_hoptimus_fit as fitter  # noqa: E402
from prosper.evaluation import feature_bank_audit as legacy  # noqa: E402
from sklearn.linear_model import SGDClassifier  # noqa: E402

RUN_NAME = S0.RUN_NAME
STAGE = "s2"
CLASSES = (1, 2, 3, 4, 5)
BUDGETS = (100, 5)
FOLDS = (0, 1, 2, 3, 4)
PRIMARY_SLOTS = ("fold0", "fold1", "fold2", "fold3", "fold4", "full")
SENSITIVITY_SLOTS = ("full_noqc",)
SLOTS = PRIMARY_SLOTS + SENSITIVITY_SLOTS
PRED_BATCH = 4096
CONCH_HASH_NAMESPACE = "r2_6_training_patch_1024_v1"
ENCODERS = {
    "uni2": {"width": 1536, "resolution": 512, "column_stop": 1536, "format": "pickle",
             "roots": ["ann_uni2_dir"]},
    "virchow2": {"width": 1280, "resolution": 512, "column_stop": 1280, "format": "pickle",
                 "roots": ["ann_virchow2_dir"]},
    "h_optimus_1": {"width": 1536, "resolution": 512, "column_stop": 1536, "format": "pickle",
                    "roots": ["ann_h_optimus_1_dir", "s1:h_optimus_1_all_patches_tf2"]},
    "conch_v15": {"width": 768, "resolution": 1024, "column_stop": 768, "format": "h5",
                  "roots": ["trident_root", "s1glob:conch_v15_tf2/chunk*/job_dir/40x_1024px_0px_overlap"]},
}
MANIFEST_COLUMNS = ["position", "tile_id", "slide_id", "patient_id", "tissue_label", "x", "y",
                    "feature_row", "training_hash"]
LIMITATIONS = [
    "conditional_on_prescribed_recipe_and_existing_annotation_banks",
    "full_heads_have_no_held_out_tiles; their accuracy is the cross-fitted estimate (contract 5.2)",
    "pool_predictions of full heads are in-sample and only feed the descriptive S2-b comparison",
    "trusted_local_pickle_not_untrusted_model_ingestion",
    "fit_width wrapper sets r2_1_hoptimus_fit._WIDTH during the call; reload heads with fit_width(width)",
]


class S2Error(RuntimeError):
    """Plan, input, fit or gate requirement failed: STOP."""


def require(cond, message):
    if not cond:
        raise S2Error(message)


sha256_file = S0.sha256_file
json_sha = S0.json_sha
write_csv = S0.write_csv
write_json = S0.write_json
now = S0.now


def ordered_ids_sha(ids) -> str:
    """Byte-identical to r2_1_hoptimus_filter_cli._ordered_ids_sha / fit._order_digest."""
    digest = hashlib.sha256(b"[")
    for i, t in enumerate(ids):
        if i:
            digest.update(b",")
        digest.update(json.dumps(t, ensure_ascii=True).encode("ascii"))
    digest.update(b"]")
    return digest.hexdigest()


@contextlib.contextmanager
def fit_width(width: int):
    """Thin width wrapper around r2_1_hoptimus_fit (module constant _WIDTH = 1536)."""
    old = fitter._WIDTH
    fitter._WIDTH = int(width)
    try:
        yield
    finally:
        fitter._WIDTH = old


def code_record():
    paths = {"s2_script": Path(__file__).resolve(), "s0_script": Path(S0.__file__).resolve(),
             "fit": Path(fitter.__file__).resolve(), "filter_kernel": Path(kernel.__file__).resolve(),
             "legacy_io": Path(legacy.__file__).resolve()}
    return {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in paths.items()}


def conch_hash_fn():
    mod = _load("r2_6_conch_tumor_heads_tf2", SCRIPTS / "r2_6_conch_tumor_heads.py")
    require(mod.NAMESPACE == CONCH_HASH_NAMESPACE, "CONCH namespace changed")
    return mod.training_patch_hash


def patch_id_1024(slide, x, y) -> str:
    return kernel._json_sha([slide, 0, int(x), int(y), 1024])


# ---------------------------------------------------------------- manifests (§3.3)
def select_budget(order_labels: np.ndarray, budget: int) -> np.ndarray:
    """Positions (into an already hash-ordered pool) kept by the budget: 100 % = all; 5 % = per
    class the first five_percent_count(n_class) in hash order (nested in 100 %)."""
    if budget == 100:
        return np.arange(len(order_labels))
    keep = np.zeros(len(order_labels), bool)
    for c in CLASSES:
        pos = np.flatnonzero(order_labels == c)
        require(len(pos) > 0, f"training pool lacks class {c}")
        keep[pos[:kernel.five_percent_count(int(len(pos)))]] = True
    return np.flatnonzero(keep)


def class_stats(pool_labels: np.ndarray, sel_labels: np.ndarray):
    n_sel = len(sel_labels)
    out = []
    for c in CLASSES:
        pc, sc = int((pool_labels == c).sum()), int((sel_labels == c).sum())
        require(pc > 0 and sc > 0, f"class {c} absent from training pool/selection")
        out.append({"tissue_label": c, "pool_count": pc, "selected_count": sc,
                    "fraction": sc / pc, "balanced_weight": n_sel / (5 * sc)})
    return out


def tiles_512(s0_dir: Path):
    """Pool tiles (512 px) from S0 with recomputed kernel identity and training hash."""
    import pandas as pd
    t = pd.read_csv(s0_dir / "annotation_tiles.csv.gz",
                    dtype={"slide_id": str, "patient_id": str, "tile_id": str, "training_hash": str})
    ids, hashes = [], []
    for r in t.itertuples(index=False):
        rec = kernel.AnnotationRecord(r.tile_id, r.slide_id, r.patient_id, r.diagnostic_group, 0,
                                      int(r.x), int(r.y), 512, int(r.tissue_label), int(r.label_csv_row))
        hashes.append(kernel.training_tile_hash(rec))
    require(list(t["training_hash"]) == hashes, "S0 training hashes do not reproduce")
    t = t.rename(columns={"label_csv_row": "feature_row"})
    return t


def tiles_1024(label_files, pool_slides, fold, qc_slides, patient_of):
    """Labelled (1..5) 1024-px patches of the pool slides; hash namespace r2_6_training_patch_1024_v1."""
    import pandas as pd
    frames = [pd.read_csv(f, dtype={"slide_id": str, "patient_id": str}) for f in label_files]
    lab = pd.concat(frames, ignore_index=True)
    require(not lab.duplicated(["slide_id", "x", "y"]).any(), "duplicate 1024 patches across label files")
    covered = set(lab["slide_id"])
    missing = sorted(set(pool_slides) - covered, key=S0.slide_sort_key)
    lab = lab[lab["slide_id"].isin(set(pool_slides)) & lab["label_1024"].between(1, 5)].copy()
    require(bool((lab["patient_id"] == lab["slide_id"].map(patient_of)).all()),
            "1024 label patient differs from the S0 pool table")
    require(bool((lab["held_out_fold"] == lab["patient_id"].map(fold)).all()),
            "1024 label fold differs from the S0 fold table")
    hfn = conch_hash_fn()
    lab["tile_id"] = [patch_id_1024(s, x, y) for s, x, y in zip(lab["slide_id"], lab["x"], lab["y"])]
    lab["training_hash"] = [hfn(p, s, x, y, l) for p, s, x, y, l in
                            zip(lab["patient_id"], lab["slide_id"], lab["x"], lab["y"], lab["label_1024"])]
    lab = lab.rename(columns={"label_1024": "tissue_label"})
    lab["qc_removed_slide"] = lab["slide_id"].isin(qc_slides).astype(int)
    return lab, missing


def hash_order(t, resolution):
    """Hash order: (training hash, physical key (slide_id, 0, x, y, patch))."""
    keys = sorted(zip(t["training_hash"].tolist(), t["slide_id"].tolist(), t["x"].astype(int).tolist(),
                      t["y"].astype(int).tolist(), range(len(t))),
                  key=lambda k: (k[0], (k[1], 0, k[2], k[3], resolution)))
    return np.array([k[4] for k in keys], dtype=np.int64)


def physical_order(t):
    return t.sort_values(["slide_id", "x", "y"], kind="mergesort").index.to_numpy()


def build_manifests(t, resolution, fold, *, prefix):
    """All (slot, budget) training manifests + per-fold held-out sets + the full pool set."""
    order = hash_order(t, resolution)
    ordered = t.iloc[order].reset_index(drop=True)
    ofold = ordered["patient_id"].map(fold).to_numpy()
    olab = ordered["tissue_label"].to_numpy()
    manifests, heldout = {}, {}
    for slot in SLOTS:
        if slot == "full":
            mask = np.ones(len(ordered), bool)
        elif slot == "full_noqc":
            mask = ordered["qc_removed_slide"].to_numpy() == 0
        else:
            mask = ofold != int(slot[4:])
        pool = np.flatnonzero(mask)
        for b in BUDGETS:
            sel = pool[select_budget(olab[pool], b)]
            rows = ordered.iloc[sel].copy()
            rows.insert(0, "position", np.arange(len(rows)))
            mid = f"{prefix}_b{b}_{slot}"
            pats = sorted(set(rows["patient_id"]))
            manifests[mid] = {
                "frame": rows[MANIFEST_COLUMNS], "slot": slot, "budget_percent": b,
                "held_out_fold": int(slot[4:]) if slot.startswith("fold") else None,
                "pool_count": int(len(pool)), "selected_count": int(len(sel)),
                "selected_fraction": len(sel) / len(pool),
                "class_stats": class_stats(olab[pool], olab[sel]),
                "training_tile_ids_sha256": ordered_ids_sha(rows["tile_id"]),
                "training_patients": len(pats), "training_patients_sha256": json_sha(pats),
                "training_slides": int(rows["slide_id"].nunique()), "patients": pats}
    phys = t.loc[physical_order(t)].reset_index(drop=True)
    for f in FOLDS:
        rows = phys[phys["patient_id"].map(fold) == f].copy()
        rows.insert(0, "position", np.arange(len(rows)))
        heldout[f"{prefix}_heldout_fold{f}"] = {
            "frame": rows[MANIFEST_COLUMNS], "held_out_fold": f,
            "held_out_tile_ids_sha256": ordered_ids_sha(rows["tile_id"]), "rows": int(len(rows))}
    rows = phys.copy()
    rows.insert(0, "position", np.arange(len(rows)))
    heldout[f"{prefix}_pool_all"] = {"frame": rows[MANIFEST_COLUMNS], "held_out_fold": None,
                                     "held_out_tile_ids_sha256": ordered_ids_sha(rows["tile_id"]),
                                     "rows": int(len(rows)),
                                     "qc_flags": phys["qc_removed_slide"].to_numpy()}
    return manifests, heldout


# ---------------------------------------------------------------- dry run / plan
def s1_root(prosper_out):
    return S0.stage_root(prosper_out) / "s1"


def bank_roots(encoder, inputs, prosper_out):
    roots = []
    for r in ENCODERS[encoder]["roots"]:
        if r.startswith("s1:"):
            roots.append(str(s1_root(prosper_out) / r[3:]))
        elif r.startswith("s1glob:"):
            roots.extend(str(p) for p in sorted(s1_root(prosper_out).glob(r[7:])))
        else:
            roots.append(inputs[r])
    return roots


def feature_file(encoder, slide, roots):
    """Exactly one bank root must hold the slide's file (ambiguity -> STOP)."""
    if ENCODERS[encoder]["format"] == "pickle":
        cands = [Path(r) / f"{slide}.pkl" for r in roots]
    else:
        cands = [Path(r) / "features_conch_v15" / f"{slide}.h5" for r in roots]
    found = [c for c in cands if c.exists()]
    require(len(found) <= 1, f"{encoder} {slide}: file present in more than one bank root {found}")
    return found[0] if found else None


def verify_s0(s0_dir: Path):
    complete = json.loads((s0_dir / "COMPLETE.json").read_text())
    require(complete.get("status") == "COMPLETE", "S0 COMPLETE.json status is not COMPLETE")
    for name, entry in complete["outputs"].items():
        require(sha256_file(s0_dir / name) == entry["sha256"], f"S0 output {name} changed since S0")
    return complete


def run_dry(s0_dir: Path, out: Path, prosper_out: str, *, supplements=(), encoders=tuple(ENCODERS),
            inputs=None, heads_root=None, log=print) -> dict:
    import pandas as pd
    t0 = time.time()
    heads_root = Path(heads_root) if heads_root is not None else S0.stage_root(prosper_out) / "s2"
    hr = heads_root.absolute().resolve()
    s2_root = (S0.stage_root(prosper_out) / "s2").resolve()
    require(hr == s2_root or s2_root in hr.parents, f"--heads-root {heads_root} is not under {s2_root}")
    if heads_root.exists() and hr != s2_root:
        require(heads_root.is_dir() and not any(heads_root.iterdir()),
                f"--heads-root {heads_root} exists and is not empty")
    complete = verify_s0(s0_dir)
    inputs = inputs or S0.default_inputs(S0.HAORAN, prosper_out)
    pool = pd.read_csv(s0_dir / "pool.csv", dtype={"slide_id": str, "UID.Subject": str})
    pooled = pool[pool["in_pool"] == 1]
    pool_slides = pooled["slide_id"].tolist()
    patient_of = dict(zip(pooled["slide_id"], pooled["UID.Subject"]))
    qc_slides = set(pooled.loc[pooled["provenance_class"] == "qc_removed", "slide_id"])
    folds = pd.read_csv(s0_dir / "patient_folds_tf2.csv", dtype={"patient_id": str})
    fold = dict(zip(folds["patient_id"], folds["fold"].astype(int)))
    planned = pd.read_csv(s0_dir / "planned_head_patients.csv", dtype={"patient_id": str})
    planned = {k: set(g["patient_id"]) for k, g in planned.groupby("head_slot")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir()
    sets, checks = {}, {}
    t512 = tiles_512(s0_dir)
    require(bool((t512["patient_id"] == t512["slide_id"].map(patient_of)).all()), "tile patient != pool patient")
    m512, h512 = build_manifests(t512, 512, fold, prefix="px512")
    sets[512] = (m512, h512, [])
    label_files = [s0_dir / "labels_1024.csv.gz"] + [Path(p) for p in supplements]
    t1024, missing1024 = tiles_1024(label_files, pool_slides, fold, qc_slides, patient_of)
    if not missing1024:
        m1024, h1024 = build_manifests(t1024, 1024, fold, prefix="px1024")
    else:
        m1024, h1024 = {}, {}
    sets[1024] = (m1024, h1024, missing1024)
    files = {}
    for res, (mans, helds, _) in sets.items():
        for mid, m in list(mans.items()) + list(helds.items()):
            path = out / "manifests" / f"{mid}.csv.gz"
            write_csv(m["frame"], path)
            files[mid] = {"file": f"manifests/{path.name}", "sha256": sha256_file(path),
                          "rows": int(len(m["frame"]))}
    pats = [{"manifest_id": mid, "patient_id": p} for res in sets for mid, m in sets[res][0].items()
            for p in m["patients"]]
    write_csv(pd.DataFrame(pats), out / "manifest_patients.csv")
    # cross-encoder identity of 512 tile selections
    checks["identical_512_tile_identities_across_uni2_virchow2_h_optimus_1"] = (
        "one manifest per (budget, slot) at 512 px is shared by the three 512-px encoders")
    for slot in SLOTS:
        a = set(m512[f"px512_b5_{slot}"]["frame"]["tile_id"])
        b = set(m512[f"px512_b100_{slot}"]["frame"]["tile_id"])
        require(a <= b, f"5 % not nested in 100 % for {slot}")
    checks["five_percent_nested_in_hundred"] = True
    heads, availability = [], {}
    index = 0
    for enc in ENCODERS:
        cfg = ENCODERS[enc]
        roots = bank_roots(enc, inputs, prosper_out)
        present, missing = {}, []
        for s in pool_slides:
            f = feature_file(enc, s, roots)
            if f is None:
                missing.append(s)
            else:
                present[s] = {"path": str(f), "size": f.stat().st_size}
        availability[enc] = {"roots": roots, "present": len(present),
                             "missing": sorted(missing, key=S0.slide_sort_key)}
        mans, helds, miss_lab = sets[cfg["resolution"]]
        for b in BUDGETS:
            for slot in SLOTS:
                index += 1
                hid = f"{enc}/b{b}/{slot}"
                mid = f"px{cfg['resolution']}_b{b}_{slot}"
                blocked = []
                if enc not in encoders:
                    blocked.append("encoder not requested in this plan")
                if missing:
                    blocked.append(f"{len(missing)} pool slides lack {enc} feature files (S1 pending)")
                if miss_lab and cfg["resolution"] == 1024:
                    blocked.append(f"{len(miss_lab)} pool slides lack 1024-px labels (S1c grid + labels-1024-supplement pending)")
                head = {"index": index, "head_id": hid, "encoder": enc, "budget_percent": b, "slot": slot,
                        "kind": "primary" if slot in PRIMARY_SLOTS else "sensitivity_s2b",
                        "held_out_fold": int(slot[4:]) if slot.startswith("fold") else None,
                        "width": cfg["width"], "resolution": cfg["resolution"],
                        "manifest_id": mid if mid in mans else None,
                        "held_out_id": (f"px{cfg['resolution']}_heldout_{slot}" if slot.startswith("fold") else
                                        f"px{cfg['resolution']}_pool_all"),
                        "status": "ready" if not blocked else "blocked", "blocked_reasons": blocked,
                        "out_subdir": f"heads/{enc}/b{b}/{slot}"}
                if mid in mans:
                    m = mans[mid]
                    head.update({k: m[k] for k in ("pool_count", "selected_count", "selected_fraction",
                                                   "class_stats", "training_tile_ids_sha256",
                                                   "training_patients", "training_patients_sha256",
                                                   "training_slides")})
                    # planned-set consistency with S0 (slot full_noqc subset of full)
                    ref = planned["full" if slot == "full_noqc" else slot]
                    require(set(m["patients"]) <= ref, f"{hid}: training patients outside the S0 planned set")
                    if slot.startswith("fold"):
                        require(not any(fold[p] == int(slot[4:]) for p in m["patients"]),
                                f"{hid}: held-out fold patient in training")
                else:
                    ref = planned["full" if slot == "full_noqc" else slot]
                    head.update({"planned_training_patients_upper_bound": len(ref),
                                 "planned_training_patients_upper_bound_sha256": json_sha(sorted(ref))})
                heads.append(head)
    # support gate (STOP if a training set lacks a class): class_stats raises if absent
    checks["every_planned_training_set_has_all_five_classes"] = True
    plan = {
        "schema_version": 1, "run": f"{RUN_NAME}/{STAGE}", "created": now(), "dry_run": True,
        "contract": S0.CONTRACT_SHA,
        "s0": {"dir": str(s0_dir), "complete_sha256": sha256_file(s0_dir / "COMPLETE.json"),
               "exclusion": complete.get("records", {}).get("exclusion")},
        "heads_root": str(heads_root.absolute()),
        "labels_1024_files": [{"path": str(p), "sha256": sha256_file(p)} for p in label_files],
        "recipe": {"module": str(Path(fitter.__file__).resolve()), "sha256": sha256_file(fitter.__file__),
                   "parameters": dict(fitter._PARAMETERS), "classes": list(CLASSES),
                   "class_weight": "n_selected / (5 * n_selected_c)",
                   "decision": "argmax over five classes; tumour = class 1",
                   "width_wrapper": "fit_width(width) sets r2_1_hoptimus_fit._WIDTH during the call"},
        "encoders": {e: {k: v for k, v in c.items() if k != "roots"} | {"roots": availability[e]["roots"]}
                     for e, c in ENCODERS.items()},
        "hash_namespaces": {"512": "r2_1_training_tile_v1 (kernel.training_tile_hash)",
                            "1024": CONCH_HASH_NAMESPACE + " (r2_6_conch_tumor_heads.training_patch_hash)"},
        "manifests": {mid: {**files[mid], **{k: v for k, v in m.items()
                                             if k not in ("frame", "patients", "qc_flags")}}
                      for res in sets for mid, m in list(sets[res][0].items()) + list(sets[res][1].items())},
        "labels_1024_missing_slides": missing1024,
        "feature_availability": availability,
        "heads": heads,
        "checks": checks,
        "code": code_record(),
    }
    plan["lsf"] = lsf_spec(out, heads, prosper_out, heads_root)
    (out / "lsf").mkdir()
    (out / "lsf" / "s2_heads_array.lsf").write_text(plan["lsf"]["script"])
    cmds = fit_commands(out, heads, prosper_out, heads_root)
    (out / "cmds.txt").write_text("".join(c + "\n" for c in cmds))
    plan["cmds"] = {"file": "cmds.txt", "sha256": sha256_file(out / "cmds.txt"), "lines": len(cmds),
                    "one_line_per_ready_head_in_index_order": True,
                    "resources_per_task": {"cpu": 1, "mem": "16 GiB", "walltime_min": 60, "gpu": 0},
                    "submission": "by the lead through the lsf-max-throughput pool runner; not submitted here"}
    write_csv(pd.DataFrame([{"index": h["index"], "head_id": h["head_id"], "status": h["status"],
                             "blocked_reasons": "; ".join(h["blocked_reasons"])} for h in heads]),
              out / "lsf" / "head_index.csv")
    plan["seconds"] = round(time.time() - t0, 1)
    write_json(plan, out / "head_plan.json")
    log(json.dumps({"heads": len(heads), "ready": sum(h["status"] == "ready" for h in heads),
                    "blocked": [h["head_id"] for h in heads if h["status"] != "ready"][:4]}))
    return plan


def ranges(idx):
    idx = sorted(idx)
    out, start, prev = [], None, None
    for i in idx:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append((start, prev)); start = prev = i
    if start is not None:
        out.append((start, prev))
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in out)


FIT_ENV = ("OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
           "BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 LSB_JOB_REPORT_MAIL=N")
PYTHON = "/HOME/miniconda3/bin/python3"


def fit_commands(plan_dir: Path, heads, prosper_out, heads_root):
    """One shell command per ready head (index order) for the lsf-max-throughput pool runner."""
    script = S0.REPO / "scripts" / "r2_1_tf2_s2_heads.py"
    return [f"export PROSPER_OUT={prosper_out} {FIT_ENV}; cd {S0.REPO}; {PYTHON} {script} fit "
            f"--plan {Path(plan_dir).absolute()} --head-index {h['index']} --heads-root {Path(heads_root).absolute()}"
            for h in sorted(heads, key=lambda h: h["index"]) if h["status"] == "ready"]


def lsf_spec(plan_dir: Path, heads, prosper_out, heads_root=None):
    """LSF array specification for the analyst (submitted through the lsf-max-throughput skill)."""
    repo = str(S0.REPO)
    ready = [h["index"] for h in heads if h["status"] == "ready"]
    heads_root = Path(heads_root).absolute() if heads_root is not None else S0.stage_root(prosper_out) / "s2"
    logs = heads_root / "logs"
    script = f"""#!/bin/bash
# R2.1-TF2 S2 head fits: ONE head per array element. NOT SUBMITTED BY THE IMPLEMENTER.
# The lead submits through the lsf-max-throughput skill (queue chosen there; probe first).
# Resources per element (contract §4 S2): 1 CPU / 16 GiB / 60 min.
#BSUB -J tf2_s2_heads[{ranges(ready) or '<indices>'}]
#BSUB -q <QUEUE_FROM_lsf-max-throughput>
#BSUB -n 1
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -M 16GB
#BSUB -W 60
#BSUB -oo {logs}/head_%I.%J.out
#BSUB -eo {logs}/head_%I.%J.err
set -euo pipefail
export LSB_JOB_REPORT_MAIL=N
export PROSPER_OUT={prosper_out}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd {repo}
/HOME/miniconda3/bin/python3 scripts/r2_1_tf2_s2_heads.py fit \\
    --plan {plan_dir} --head-index ${{LSB_JOBINDEX}} --heads-root {heads_root}
"""
    return {"script": script, "ready_indices": ranges(ready),
            "blocked_indices": ranges([h["index"] for h in heads if h["status"] != "ready"]),
            "resources": {"cpu": 1, "mem": "16 GiB", "walltime_min": 60, "gpu": 0},
            "logs_dir": str(logs),
            "notes": ["mkdir -p the logs dir before submission",
                      "blocked heads (H-Optimus-1 / CONCH) need S1 outputs (and for CONCH the 1024-px "
                      "label supplement and a CONCH plan) before submission",
                      "after all heads: f7-check (F7 checkpoint ii) then report; no retries without diagnosis"]}


# ---------------------------------------------------------------- features
def read_bytes_hashed(path: Path):
    data = Path(path).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def pickle_block(path: Path, slide: str, column_stop: int):
    data, digest = read_bytes_hashed(path)
    payload = legacy._pickle(data, f"feature {slide}")
    del data
    require(type(payload) is tuple and len(payload) == 2, f"{slide}: (features, tiles) tuple required")
    raw, tiles = payload
    require(type(raw) is np.ndarray and raw.ndim == 2 and raw.dtype == np.float32,
            f"{slide}: float32 2-D feature matrix required")
    require(raw.shape[1] >= column_stop, f"{slide}: feature width {raw.shape[1]} < {column_stop}")
    coords = np.array(legacy._coordinates(tiles, f"feature {slide}"), dtype=np.int64)
    require(len(coords) == raw.shape[0], f"{slide}: coordinate/feature row count")
    if raw.shape[1] != column_stop:
        raw = np.ascontiguousarray(raw[:, :column_stop])
    require(bool(np.isfinite(raw).all()), f"{slide}: non-finite features")
    return raw, coords, {"slide_id": slide, "path": str(Path(path).resolve()), "sha256": digest,
                         "rows": int(raw.shape[0]), "raw_width": int(payload[0].shape[1])}


def h5_block(path: Path, slide: str):
    import h5py
    data, digest = read_bytes_hashed(path)
    with h5py.File(io.BytesIO(data), "r") as h:
        coords = h["coords"][:].astype(np.int64)
        raw = np.ascontiguousarray(h["features"][:], dtype=np.float32)
    require(raw.ndim == 2 and raw.shape == (len(coords), 768), f"{slide}: CONCH shape {raw.shape}")
    require(bool(np.isfinite(raw).all()), f"{slide}: non-finite CONCH features")
    return raw, coords, {"slide_id": slide, "path": str(Path(path).resolve()), "sha256": digest,
                         "rows": int(raw.shape[0]), "raw_width": 768}


def records_512(frame):
    return tuple(kernel.AnnotationRecord(r.tile_id, r.slide_id, r.patient_id, "NA", 0, int(r.x), int(r.y),
                                         512, int(r.tissue_label), int(r.feature_row))
                 for r in frame.itertuples(index=False))


def assemble_512(records, encoder, files_of, opened):
    """fitter.assemble_selected_features over per-slide pickle blocks (coordinate-checked rows)."""
    cfg = ENCODERS[encoder]

    def blocks():
        for slide in sorted({r.slide_id for r in records}):
            raw, coords, rec = pickle_block(files_of[slide], slide, cfg["column_stop"])
            opened.append(rec)
            yield slide, raw, coords

    with fit_width(cfg["width"]):
        return fitter.assemble_selected_features(records, blocks())


def assemble_1024(frame, files_of, opened):
    X = np.empty((len(frame), 768), np.float32)
    labels = frame["tissue_label"].to_numpy(np.int64)
    for slide, g in frame.groupby("slide_id", sort=True):
        raw, coords, rec = h5_block(files_of[slide], slide)
        opened.append(rec)
        fr = g["feature_row"].to_numpy(np.int64)
        require(bool((fr < len(coords)).all()), f"{slide}: feature_row out of range")
        require(np.array_equal(coords[fr], g[["x", "y"]].to_numpy(np.int64)),
                f"{slide}: feature_row/coordinate mismatch")
        X[g.index.to_numpy()] = raw[fr]
    return X, labels


# ---------------------------------------------------------------- fitting
def manifest_object(head, frame, records):
    stats = tuple(kernel.ClassSelection(**s) for s in head["class_stats"])
    return kernel.TrainingManifest(head["held_out_fold"], head["budget_percent"], records, stats,
                                   head["pool_count"], head["selected_count"], head["selected_fraction"])


def fit_generic(features, labels, weights, *, width, context, held_out_fold, budget, frame,
                class_stats_rows, pool_count, selected_fraction):
    """Same recipe as fitter.fit_head, without its 512-px/held-out-fold manifest validator."""
    with fit_width(width):
        fitter._features(features, rows=len(labels))
        runtime = fitter._runtime_profile()
        input_digest = hashlib.sha256(memoryview(features).cast("B")).hexdigest()
        estimator = SGDClassifier(**fitter._parameters(weights))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(features, labels, sample_weight=None)
        fit_warnings = fitter._warnings_report(caught, context, retain_convergence=True)
        fitter._state(estimator, weights)
        diagnostics = {
            "schema_version": 1, "fit_path": "fit_generic (r2_1_hoptimus_fit primitives)",
            "held_out_fold": held_out_fold, "budget_percent": budget,
            "selected_tiles": int(len(labels)),
            "selected_slides": int(frame["slide_id"].nunique()),
            "selected_patients": int(frame["patient_id"].nunique()),
            "pool_count": pool_count, "selected_fraction": selected_fraction,
            "class_support": [{
                "tissue_label": s["tissue_label"], "pool_tiles": s["pool_count"],
                "selected_tiles": s["selected_count"],
                "selected_slides": int(frame.loc[frame["tissue_label"] == s["tissue_label"], "slide_id"].nunique()),
                "selected_patients": int(frame.loc[frame["tissue_label"] == s["tissue_label"], "patient_id"].nunique()),
                "selected_fraction": s["fraction"], "balanced_weight": s["balanced_weight"]}
                for s in class_stats_rows],
            "ordered_tile_ids_sha256": ordered_ids_sha(frame["tile_id"]),
            "feature_sha256": input_digest, "feature_shape": list(features.shape),
            "feature_dtype": features.dtype.str, "parameters": estimator.get_params(deep=False),
            "classes": estimator.classes_.tolist(), "finite_state": True,
            "coefficient_shape": list(estimator.coef_.shape), "coefficient_dtype": estimator.coef_.dtype.str,
            "intercept_shape": list(estimator.intercept_.shape), "intercept_dtype": estimator.intercept_.dtype.str,
            "n_features_in": estimator.n_features_in_, "n_iter": estimator.n_iter_, "t": estimator.t_,
            "n_iter_semantics": "maximum_across_five_one_vs_rest_fits", "warnings": fit_warnings,
            "iteration_limit_warning": bool(fit_warnings),
            "convergence_status": ("iteration_limit_warning_retained" if fit_warnings
                                   else "no_warning_not_optimality_claim"),
            "runtime": runtime, "fitted_state_sha256": fitter._fingerprint(estimator),
        }
        return fitter.FittedHead(estimator, diagnostics)


def predict(head, X, width):
    probs = np.empty((len(X), 5), np.float32)
    labels = np.empty(len(X), np.int64)
    with fit_width(width):
        for s in range(0, len(X), PRED_BATCH):
            b = fitter.predict_head(head, np.ascontiguousarray(X[s:s + PRED_BATCH]))
            probs[s:s + PRED_BATCH], labels[s:s + PRED_BATCH] = b.probabilities, b.labels
    require(np.array_equal(labels, np.array(CLASSES)[probs.argmax(axis=1)]), "prediction/argmax mismatch")
    return probs, labels


def confusion(truth, pred):
    cm = np.zeros((5, 5), np.int64)
    for a, b in zip(truth, pred):
        cm[int(a) - 1, int(b) - 1] += 1
    return cm


def fit_core(head, train_frame, other_frame, files_of, fold_map, *, other_is_pool):
    """Fit one head; return (FittedHead, other-set probabilities/labels, opened feature records).

    other_frame: held-out tiles (fold heads) or all pool tiles (full heads, in-sample)."""
    enc = ENCODERS[head["encoder"]]
    width, opened = enc["width"], []
    weights = {s["tissue_label"]: s["balanced_weight"] for s in head["class_stats"]}
    context = f"{head['head_id']}"
    if enc["resolution"] == 512:
        if not other_is_pool:  # fold head: training + held-out disjoint; fit_head itself
            train_rec, held_rec = records_512(train_frame), records_512(other_frame)
            X, y = assemble_512(train_rec + held_rec, head["encoder"], files_of, opened)
            n = len(train_rec)
            manifest = manifest_object(head, train_frame, train_rec)
            with fit_width(width):
                fitted = fitter.fit_head(X[:n], manifest, fold_map)
            Xo = X[n:]
        else:  # full heads: assemble the pool once, training rows gathered by tile id
            pool_rec = records_512(other_frame)
            Xo, _ = assemble_512(pool_rec, head["encoder"], files_of, opened)
            pos = {t: i for i, t in enumerate(other_frame["tile_id"])}
            idx = np.array([pos[t] for t in train_frame["tile_id"]], np.int64)
            Xt = np.ascontiguousarray(Xo[idx])
            yt = train_frame["tissue_label"].to_numpy(np.int64)
            fitted = fit_generic(Xt, yt, weights, width=width, context=context, held_out_fold=None,
                                 budget=head["budget_percent"], frame=train_frame,
                                 class_stats_rows=head["class_stats"], pool_count=head["pool_count"],
                                 selected_fraction=head["selected_fraction"])
    else:
        if not other_is_pool:
            both = pd_concat(train_frame, other_frame)
            X, _ = assemble_1024(both, files_of, opened)
            n = len(train_frame)
            Xt, Xo = X[:n], X[n:]
        else:
            Xo, _ = assemble_1024(other_frame.reset_index(drop=True), files_of, opened)
            pos = {t: i for i, t in enumerate(other_frame["tile_id"])}
            Xt = np.ascontiguousarray(Xo[np.array([pos[t] for t in train_frame["tile_id"]], np.int64)])
        yt = train_frame["tissue_label"].to_numpy(np.int64)
        fitted = fit_generic(np.ascontiguousarray(Xt), yt, weights, width=width, context=context,
                             held_out_fold=head["held_out_fold"], budget=head["budget_percent"],
                             frame=train_frame, class_stats_rows=head["class_stats"],
                             pool_count=head["pool_count"], selected_fraction=head["selected_fraction"])
    probs, preds = predict(fitted, Xo, width)
    sample = np.ascontiguousarray(Xo[:PRED_BATCH])
    return fitted, probs, preds, opened, sample


def pd_concat(a, b):
    import pandas as pd
    return pd.concat([a, b], ignore_index=True)


def load_plan(plan_dir: Path):
    plan = json.loads((plan_dir / "head_plan.json").read_text())
    return plan


def read_manifest(plan_dir, plan, mid):
    import pandas as pd
    entry = plan["manifests"][mid]
    path = plan_dir / entry["file"]
    require(sha256_file(path) == entry["sha256"], f"manifest {mid} changed since the plan")
    return pd.read_csv(path, dtype={"tile_id": str, "slide_id": str, "patient_id": str, "training_hash": str})


def run_fit(plan_dir: Path, heads_root: Path, prosper_out: str, *, head_id=None, head_index=None,
            inputs=None, log=print) -> dict:
    t0 = time.time()
    plan = load_plan(plan_dir)
    heads = {h["head_id"]: h for h in plan["heads"]}
    if head_index is not None:
        matches = [h for h in plan["heads"] if h["index"] == int(head_index)]
        require(len(matches) == 1, f"no head with index {head_index}")
        head = matches[0]
    else:
        require(head_id in heads, f"unknown head {head_id}")
        head = heads[head_id]
    require(head["status"] == "ready", f"{head['head_id']} is blocked: {head['blocked_reasons']}")
    if plan.get("heads_root"):
        require(Path(heads_root).absolute().resolve() == Path(plan["heads_root"]).resolve(),
                f"--heads-root {heads_root} differs from the plan's heads root {plan['heads_root']}")
    out = S0.check_out_dir(heads_root / head["out_subdir"], prosper_out, STAGE)
    out.mkdir(parents=True, exist_ok=True)
    started = now()
    try:
        s0_dir = Path(plan["s0"]["dir"])
        require(sha256_file(s0_dir / "COMPLETE.json") == plan["s0"]["complete_sha256"], "S0 changed")
        import pandas as pd
        folds = pd.read_csv(s0_dir / "patient_folds_tf2.csv", dtype={"patient_id": str})
        fold_map = dict(zip(folds["patient_id"], folds["fold"].astype(int)))
        train = read_manifest(plan_dir, plan, head["manifest_id"])
        other = read_manifest(plan_dir, plan, head["held_out_id"])
        require(ordered_ids_sha(train["tile_id"]) == head["training_tile_ids_sha256"], "training order sha")
        other_is_pool = not head["slot"].startswith("fold")
        if not other_is_pool:
            require(not (set(train["patient_id"]) & set(other["patient_id"])), "train/held-out patient overlap")
        inputs = inputs or S0.default_inputs(S0.HAORAN, prosper_out)
        roots = bank_roots(head["encoder"], inputs, prosper_out)
        require([str(Path(r).resolve()) for r in roots] ==
                [str(Path(r).resolve()) for r in plan["feature_availability"][head["encoder"]]["roots"]],
                f"{head['encoder']}: bank roots differ from the plan")
        slides = sorted(set(train["slide_id"]) | set(other["slide_id"]))
        files_of = {}
        for s in slides:
            f = feature_file(head["encoder"], s, roots)
            require(f is not None, f"{head['encoder']} {s}: feature file missing")
            files_of[s] = f
        fitted, probs, preds, opened, sample = fit_core(head, train, other, files_of, fold_map,
                                                        other_is_pool=other_is_pool)
        diag = fitted.diagnostics
        require(diag["ordered_tile_ids_sha256"] == head["training_tile_ids_sha256"], "fitted training order")
        # artifacts
        blob = pickle.dumps(fitted, protocol=4)
        (out / "head.pkl").write_bytes(blob)
        restored = pickle.loads((out / "head.pkl").read_bytes())
        p2, l2 = predict(restored, sample, head["width"])
        require(p2.tobytes() == probs[:len(sample)].tobytes() and l2.tobytes() == preds[:len(sample)].tobytes(),
                "reloaded head.pkl predictions differ")
        require(fitter._fingerprint(restored.estimator) == diag["fitted_state_sha256"], "reload fingerprint")
        write_json(diag, out / "fit_diagnostics.json")
        tt = train[["position", "tile_id", "slide_id", "patient_id", "tissue_label"]]
        write_csv(tt, out / "training_tiles.csv.gz")
        truth = other["tissue_label"].to_numpy(np.int64)
        prefix = "pool_" if other_is_pool else ""
        np.save(out / f"{prefix}probabilities.npy", probs, allow_pickle=False)
        np.save(out / f"{prefix}predictions.npy", preds, allow_pickle=False)
        write_csv(other[["position", "tile_id", "slide_id", "patient_id", "tissue_label", "x", "y"]],
                  out / f"{prefix}rows.csv.gz" if other_is_pool else out / "held_out_rows.csv.gz")
        train_patients = sorted(set(train["patient_id"]))
        counts = {
            "schema_version": 1, "head_id": head["head_id"], "encoder": head["encoder"],
            "budget_percent": head["budget_percent"], "slot": head["slot"],
            "held_out_fold": head["held_out_fold"],
            "training_tiles": int(len(train)), "training_slides": int(train["slide_id"].nunique()),
            "training_patients": train_patients, "training_patients_sha256": json_sha(train_patients),
            "training_tile_ids_sha256": head["training_tile_ids_sha256"],
            "training_class_counts": {str(c): int((train["tissue_label"] == c).sum()) for c in CLASSES},
            "evaluation_set": "pool_in_sample" if other_is_pool else "held_out_fold",
            "evaluation_tile_ids_sha256": ordered_ids_sha(other["tile_id"]),
            "tumour_counts": fitter.tumour_counts(truth, preds),
            "tissue_confusion": confusion(truth, preds).tolist(),
            "evaluation_support": [{"tissue_label": c, "tiles": int((truth == c).sum()),
                                    "patients": int(other.loc[other["tissue_label"] == c, "patient_id"].nunique())}
                                   for c in CLASSES],
        }
        write_json(counts, out / "counts.json")
        write_json({"schema_version": 1, "head_id": head["head_id"],
                    "plan": {"path": str(plan_dir / "head_plan.json"),
                             "sha256": sha256_file(plan_dir / "head_plan.json")},
                    "manifests": {head["manifest_id"]: plan["manifests"][head["manifest_id"]]["sha256"],
                                  head["held_out_id"]: plan["manifests"][head["held_out_id"]]["sha256"]},
                    "feature_reads": opened, "code": code_record(),
                    "environment": {"python": sys.version, "host": socket.gethostname(),
                                    "lsf_job": os.environ.get("LSB_JOBID"),
                                    "lsf_index": os.environ.get("LSB_JOBINDEX"),
                                    "fit_runtime": diag["runtime"]},
                    "started": started, "finished": now()}, out / "input_bindings.json")
        outputs = {p.name: {"sha256": sha256_file(p), "size": p.stat().st_size}
                   for p in sorted(out.iterdir()) if p.is_file()}
        marker = {"schema_version": 1, "status": "COMPLETE", "head_id": head["head_id"],
                  "fitted_state_sha256": diag["fitted_state_sha256"],
                  "training_tile_ids_sha256": head["training_tile_ids_sha256"],
                  "outputs": outputs, "limitations": LIMITATIONS,
                  "seconds": round(time.time() - t0, 1), "created": now()}
        write_json(marker, out / "COMPLETE.json")
        log(json.dumps({"head": head["head_id"], "status": "COMPLETE", "seconds": marker["seconds"],
                        "n_iter": diag["n_iter"], "fingerprint": diag["fitted_state_sha256"]}))
        return marker
    except Exception as exc:
        write_json({"status": "FAILED", "head_id": head["head_id"], "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(), "created": now()}, out / "FAILED.json")
        raise


# ---------------------------------------------------------------- F7 checkpoint (ii)
def run_f7(plan_dirs, heads_root: Path, s0_dir: Path, out: Path, *, log=print) -> dict:
    """R1 (ii): against each head's SAVED training set (patients via the S0 pool table): every query
    assigned to the head has no training tile from its patient; every library version's head is
    the head of that version's query fold; saved sets equal the planned manifests."""
    import pandas as pd
    verify_s0(s0_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(s0_dir / "pool.csv", dtype={"slide_id": str, "UID.Subject": str})
    patient_of = dict(zip(pool["slide_id"], pool["UID.Subject"]))
    q = pd.read_csv(s0_dir / "query_assignment.csv", dtype={"slide_id": str, "patient_id": str})
    ma = pd.read_csv(s0_dir / "mask_assignment.csv")
    planned = pd.read_csv(s0_dir / "planned_head_patients.csv", dtype={"patient_id": str})
    planned = {k: set(g["patient_id"]) for k, g in planned.groupby("head_slot")}
    heads = {}
    for pd_ in plan_dirs:
        for h in load_plan(Path(pd_))["heads"]:
            if h["status"] == "ready":
                heads[h["head_id"]] = (Path(pd_), h)
    rows, violations = [], []
    for enc in S0.ENCODERS:
        for b in BUDGETS:
            for slot in PRIMARY_SLOTS:
                hid = f"{enc}/b{b}/{slot}"
                d = heads_root / f"heads/{enc}/b{b}/{slot}"
                row = {"head_id": hid, "status": None, "saved_training_patients": None,
                       "queries_assigned": None, "query_patients_in_saved_training": None,
                       "saved_subset_of_planned": None, "saved_equals_plan_manifest": None,
                       "library_version": "v_" + slot, "library_version_head_matches": None}
                if hid not in heads or not (d / "COMPLETE.json").exists():
                    row["status"] = "missing"
                    violations.append({"head_id": hid, "kind": "head_not_complete_or_not_planned"})
                    rows.append(row)
                    continue
                complete = json.loads((d / "COMPLETE.json").read_text())
                for name, e in complete["outputs"].items():
                    require(sha256_file(d / name) == e["sha256"], f"{hid}: {name} changed after fit")
                tt = pd.read_csv(d / "training_tiles.csv.gz", dtype={"slide_id": str, "patient_id": str,
                                                                    "tile_id": str})
                via_pool = tt["slide_id"].map(patient_of)
                require(not via_pool.isna().any(), f"{hid}: training slide outside the S0 pool table")
                if not (via_pool == tt["patient_id"]).all():
                    violations.append({"head_id": hid, "kind": "saved_patient_column_differs_from_pool_table"})
                saved = set(via_pool)
                assigned = q[q["head_slot"] == slot]
                leak = assigned[assigned["patient_id"].isin(saved)]
                plan_head = heads[hid][1]
                same_plan = ordered_ids_sha(tt["tile_id"]) == plan_head["training_tile_ids_sha256"]
                vrow = ma[(ma["encoder"] == enc) & (ma["budget_percent"] == b) &
                          (ma["entry_kind"] == "library_version") & (ma["entry_id"] == "v_" + slot)]
                qrow = ma[(ma["encoder"] == enc) & (ma["budget_percent"] == b) &
                          (ma["entry_kind"] == "query_stratum") & (ma["head_slot"] == slot)]
                version_ok = (len(vrow) == 1 and len(qrow) == 1 and vrow["head_id"].iat[0] == hid ==
                              qrow["head_id"].iat[0] and qrow["both_sided_library_version"].iat[0] == "v_" + slot)
                row.update(status="checked", saved_training_patients=len(saved),
                           queries_assigned=int(len(assigned)), query_patients_in_saved_training=int(len(leak)),
                           saved_subset_of_planned=saved <= planned[slot], saved_equals_plan_manifest=same_plan,
                           library_version_head_matches=version_ok)
                for r in leak.itertuples(index=False):
                    violations.append({"head_id": hid, "kind": "query_patient_in_saved_training",
                                       "query_index": int(r.query_index), "patient_id": r.patient_id})
                if not saved <= planned[slot]:
                    violations.append({"head_id": hid, "kind": "saved_set_outside_planned_set"})
                if not same_plan:
                    violations.append({"head_id": hid, "kind": "saved_set_differs_from_plan_manifest"})
                if not version_ok:
                    violations.append({"head_id": hid, "kind": "library_version_head_mismatch"})
                rows.append(row)
    result = {"checkpoint": "F7 (ii) after S2 on saved training-patient sets (R1)", "created": now(),
              "heads_checked": sum(r["status"] == "checked" for r in rows), "heads_expected": len(rows),
              "violations": violations, "pass": not violations, "code": code_record()}
    write_csv(pd.DataFrame(rows), out / "f7_checkpoint_ii.csv")
    write_json(result, out / ("f7_checkpoint_ii.json" if result["pass"] else "f7_checkpoint_ii_FAILED.json"))
    log(json.dumps({"pass": result["pass"], "violations": len(violations)}))
    return result


# ---------------------------------------------------------------- classifier report
def tumour_metrics(truth, pred):
    t, p = truth == 1, pred == 1
    tp, fn, fp, tn = int((t & p).sum()), int((t & ~p).sum()), int((~t & p).sum()), int((~t & ~p).sum())
    return {"n": int(len(truth)), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None}


def run_report(plan_dirs, heads_root: Path, out: Path, *, log=print) -> dict:
    """§5.1 cross-fitted tile metrics from fold heads (each labelled tile once per budget);
    §5.2 H_all reported as the cross-fitted estimate; §5.4 S2-b full_noqc vs full (descriptive)."""
    import pandas as pd
    out.mkdir(parents=True, exist_ok=True)
    metric_rows, label_rows, conf_rows, s2b_rows = [], [], [], []
    statements = []
    for enc in S0.ENCODERS:
        for b in BUDGETS:
            truth_all, pred_all, ok = [], [], True
            for f in FOLDS:
                d = heads_root / f"heads/{enc}/b{b}/fold{f}"
                if not (d / "COMPLETE.json").exists():
                    ok = False
                    continue
                rows = pd.read_csv(d / "held_out_rows.csv.gz")
                pred = np.load(d / "predictions.npy")
                truth = rows["tissue_label"].to_numpy(np.int64)
                require(len(truth) == len(pred), f"{enc} b{b} fold{f}: rows/predictions")
                metric_rows.append({"encoder": enc, "budget_percent": b, "scope": f"fold{f}",
                                    **tumour_metrics(truth, pred)})
                truth_all.append(truth); pred_all.append(pred)
            if not truth_all:
                continue
            truth, pred = np.concatenate(truth_all), np.concatenate(pred_all)
            metric_rows.append({"encoder": enc, "budget_percent": b,
                                "scope": "pooled_cross_fitted" + ("" if ok else "_INCOMPLETE"),
                                **tumour_metrics(truth, pred)})
            cm = confusion(truth, pred)
            for i, c in enumerate(CLASSES):
                n = int((truth == c).sum())
                label_rows.append({"encoder": enc, "budget_percent": b, "true_label": c, "tiles": n,
                                   "pass_rate_predicted_tumour": float((pred[truth == c] == 1).mean()) if n else None,
                                   "recall_own_class": float(cm[i, i] / n) if n else None})
                for j, c2 in enumerate(CLASSES):
                    conf_rows.append({"encoder": enc, "budget_percent": b, "true_label": c,
                                      "predicted_label": c2, "tiles": int(cm[i, j])})
            statements.append({"encoder": enc, "budget_percent": b, "head": f"{enc}/b{b}/full",
                               "statement": "H_all has no held-out tiles; its accuracy is reported as the "
                                            "cross-fitted (fold-head) estimate above (contract 5.2)",
                               "cross_fitted_complete": ok})
            # S2-b: full_noqc vs full on all pool tiles (in-sample for full; QC tiles out-of-sample for noqc)
            dfull, dnoqc = heads_root / f"heads/{enc}/b{b}/full", heads_root / f"heads/{enc}/b{b}/full_noqc"
            if (dfull / "COMPLETE.json").exists() and (dnoqc / "COMPLETE.json").exists():
                ra = pd.read_csv(dfull / "pool_rows.csv.gz", dtype={"slide_id": str, "tile_id": str})
                rb = pd.read_csv(dnoqc / "pool_rows.csv.gz", dtype={"slide_id": str, "tile_id": str})
                require(list(ra["tile_id"]) == list(rb["tile_id"]), f"{enc} b{b}: S2-b pool rows differ")
                pa, pb = np.load(dfull / "pool_predictions.npy"), np.load(dnoqc / "pool_predictions.npy")
                ca = json.loads((dnoqc / "counts.json").read_text())
                qc_slides = set()
                for pdir in plan_dirs:
                    s0_dir = Path(load_plan(Path(pdir))["s0"]["dir"])
                    pool = pd.read_csv(s0_dir / "pool.csv", dtype={"slide_id": str})
                    qc_slides = set(pool.loc[pool["provenance_class"] == "qc_removed", "slide_id"])
                    break
                isqc = ra["slide_id"].isin(qc_slides).to_numpy()
                truth = ra["tissue_label"].to_numpy(np.int64)
                for scope, m in (("all_pool_tiles", np.ones(len(ra), bool)), ("qc_removed_slide_tiles", isqc),
                                 ("other_tiles", ~isqc)):
                    s2b_rows.append({"encoder": enc, "budget_percent": b, "scope": scope, "tiles": int(m.sum()),
                                     "tumour_call_agreement": float(((pa[m] == 1) == (pb[m] == 1)).mean()) if m.any() else None,
                                     "five_class_agreement": float((pa[m] == pb[m]).mean()) if m.any() else None,
                                     "full_in_sample": json.dumps(tumour_metrics(truth[m], pa[m])),
                                     "full_noqc": json.dumps(tumour_metrics(truth[m], pb[m])),
                                     "full_noqc_training_tiles": ca["training_tiles"]})
    write_csv(pd.DataFrame(metric_rows), out / "tile_metrics.csv")
    write_csv(pd.DataFrame(label_rows), out / "per_label_pass_rates.csv")
    write_csv(pd.DataFrame(conf_rows), out / "confusion.csv")
    write_csv(pd.DataFrame(s2b_rows), out / "s2b_full_noqc_vs_full.csv")
    rep = {"created": now(), "sections": {"5.1": "tile_metrics.csv, per_label_pass_rates.csv, confusion.csv",
                                          "5.2": statements, "5.4": "s2b_full_noqc_vs_full.csv (descriptive)"},
           "definitions": {"pass_rate_predicted_tumour": "fraction of tiles with this true label predicted class 1 (kept by the tumour mask)",
                           "qc_removed_slide_tiles": "out-of-sample for full_noqc, in-sample for full"},
           "code": code_record()}
    write_json(rep, out / "report.json")
    return rep


# ---------------------------------------------------------------- F2 regression
F2_REFERENCES = {
    "h_optimus_1": "r2_1_head_pilot_20261002/fold0_budget5_attempt001",
    "uni2": "r2_1_route_b_uni2_20261013/heads/fold0_budget5_attempt001",
}


def run_f2(out: Path, prosper_out: str, which=("h_optimus_1", "uni2"), *, log=print) -> dict:
    """Rebuild the accepted fold-0 5 % manifest (202-slide Stage1a pool, accepted folds) with this
    file's manifest builder and fit it with this file's fit path; compare bitwise."""
    import pandas as pd
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(prosper_out)
    inputs = S0.default_inputs(S0.HAORAN, prosper_out)
    el = pd.read_csv(inputs["ref:stage1a_eligible"], dtype={"slide_id": str, "patient_id": str, "tile_id": str})
    acc = S0.read_accepted_folds(inputs["accepted_folds"])
    t = pd.DataFrame({"tile_id": el["tile_id"], "slide_id": el["slide_id"], "patient_id": el["patient_id"],
                      "tissue_label": el["tissue_label"], "x": el["x"], "y": el["y"],
                      "feature_row": el["h_optimus_1_feature_row"], "qc_removed_slide": 0})
    t["training_hash"] = [kernel.training_tile_hash(r) for r in records_512(t)]
    order = hash_order(t, 512)
    ordered = t.iloc[order].reset_index(drop=True)
    pool = np.flatnonzero(ordered["patient_id"].map(acc).to_numpy() != 0)
    sel = pool[select_budget(ordered["tissue_label"].to_numpy()[pool], 5)]
    train = ordered.iloc[sel].copy()
    train.insert(0, "position", np.arange(len(train)))
    phys = t.loc[physical_order(t)].reset_index(drop=True)
    held = phys[phys["patient_id"].map(acc) == 0].copy()
    held.insert(0, "position", np.arange(len(held)))
    accepted_plan = json.loads((root / "r2_1_cpu_20261002/filter_attempt_001/head_plan.json").read_text())
    ref_head = [h for h in accepted_plan["heads"] if h["held_out_fold"] == 0 and h["budget_percent"] == 5][0]
    stats = class_stats(ordered["tissue_label"].to_numpy()[pool], train["tissue_label"].to_numpy())
    head = {"head_id": "f2/{enc}/b5/fold0", "encoder": None, "budget_percent": 5, "slot": "fold0",
            "held_out_fold": 0, "class_stats": stats, "pool_count": int(len(pool)),
            "selected_count": int(len(train)), "selected_fraction": len(train) / len(pool)}
    manifest_checks = {
        "training_tile_ids_sha256": ordered_ids_sha(train["tile_id"]),
        "accepted_head_plan_training_tile_ids_sha256": ref_head["training_tile_ids_sha256"],
        "class_stats_equal_accepted": stats == ref_head["class_stats"],
        "pool_count_equal": int(len(pool)) == ref_head["pool_count"],
        "held_out_tile_ids_sha256": ordered_ids_sha(held["tile_id"]),
    }
    report = {"created": now(), "manifest": manifest_checks, "banks": {}}
    for enc in which:
        ref = root / F2_REFERENCES[enc]
        rsum = json.loads((ref / "summary.json").read_text())
        rdiag = json.loads((ref / "fit_diagnostics.json").read_text())
        h = dict(head, encoder=enc, head_id=f"f2/{enc}/b5/fold0")
        roots = [inputs["ann_h_optimus_1_dir"]] if enc == "h_optimus_1" else [inputs["ann_uni2_dir"]]
        files_of = {s: Path(roots[0]) / f"{s}.pkl" for s in set(train["slide_id"]) | set(held["slide_id"])}
        te = time.time()
        fitted, probs, preds, opened, _ = fit_core(h, train, held, files_of, acc, other_is_pool=False)
        d = fitted.diagnostics
        rprob = np.load(ref / "probabilities.npy")
        rpred = np.load(ref / "predictions.npy")
        res = {
            "reference": str(ref),
            "training_tile_ids_sha256_equal": manifest_checks["training_tile_ids_sha256"] == rsum["details"]["training_tile_ids_sha256"],
            "held_out_tile_ids_sha256_equal": manifest_checks["held_out_tile_ids_sha256"] == rsum["details"]["held_out_tile_ids_sha256"],
            "feature_sha256_equal": d["feature_sha256"] == rdiag["feature_sha256"],
            "fitted_state_sha256": d["fitted_state_sha256"], "reference_fitted_state_sha256": rdiag["fitted_state_sha256"],
            "fitted_state_equal": d["fitted_state_sha256"] == rdiag["fitted_state_sha256"],
            "n_iter_equal": d["n_iter"] == rdiag["n_iter"],
            "probabilities_bitwise_equal": probs.shape == rprob.shape and probs.tobytes() == rprob.tobytes(),
            "predictions_bitwise_equal": preds.shape == rpred.shape and preds.tobytes() == rpred.tobytes(),
            "feature_files_read": len(opened), "seconds": round(time.time() - te, 1),
        }
        res["pass"] = all(v for k, v in res.items() if k.endswith("_equal"))
        report["banks"][enc] = res
        log(json.dumps({enc: res}))
    report["pass"] = (manifest_checks["training_tile_ids_sha256"] ==
                      manifest_checks["accepted_head_plan_training_tile_ids_sha256"]
                      and manifest_checks["class_stats_equal_accepted"] and manifest_checks["pool_count_equal"]
                      and all(r["pass"] for r in report["banks"].values()))
    report["seconds"] = round(time.time() - t0, 1)
    report["code"] = code_record()
    write_json(report, out / ("f2_regression.json" if report["pass"] else "f2_regression_FAILED.json"))
    return report


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run", help="validate inputs, write the head plan; no fitting")
    d.add_argument("--s0", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--labels-1024-supplement", nargs="*", default=[])
    d.add_argument("--heads-root", default=None,
                   help="where the planned fits write (under s2, new or empty); default s2/")
    f = sub.add_parser("fit")
    f.add_argument("--plan", required=True)
    f.add_argument("--heads-root", required=True)
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--head-id")
    g.add_argument("--head-index", type=int)
    c = sub.add_parser("f7-check")
    c.add_argument("--plan", nargs="+", required=True)
    c.add_argument("--heads-root", required=True)
    c.add_argument("--s0", required=True)
    c.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--plan", nargs="+", required=True)
    r.add_argument("--heads-root", required=True)
    r.add_argument("--out", required=True)
    x = sub.add_parser("f2-regression")
    x.add_argument("--out", required=True)
    x.add_argument("--which", nargs="+", default=["h_optimus_1", "uni2"], choices=list(F2_REFERENCES))
    a = ap.parse_args(argv)
    prosper_out = os.environ.get("PROSPER_OUT")
    if not prosper_out:
        sys.exit("PROSPER_OUT must be exported")
    if a.cmd == "dry-run":
        out = S0.check_out_dir(Path(a.out), prosper_out, STAGE)
        run_dry(Path(a.s0), out, prosper_out, supplements=a.labels_1024_supplement,
                heads_root=a.heads_root)
        return 0
        return 0
    if a.cmd == "fit":
        run_fit(Path(a.plan), Path(a.heads_root), prosper_out, head_id=a.head_id, head_index=a.head_index)
        return 0
    if a.cmd == "f7-check":
        out = S0.check_out_dir(Path(a.out), prosper_out, STAGE)
        return 0 if run_f7(a.plan, Path(a.heads_root), Path(a.s0), out)["pass"] else 1
    if a.cmd == "report":
        out = S0.check_out_dir(Path(a.out), prosper_out, STAGE)
        run_report(a.plan, Path(a.heads_root), out)
        return 0
    out = S0.check_out_dir(Path(a.out), prosper_out, STAGE)
    return 0 if run_f2(out, prosper_out, tuple(a.which))["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
