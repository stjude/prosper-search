#!/usr/bin/env python3
"""R2.1-TF2 stage S3: tumour masks of the six primary heads on the all-tissue library mosaic.

Binding specification: r2_1_filter_tf2_20261016/contract_candidate_003/SPECIFICATION.md (3a444fc714c87d70...)
§3.3, §3.4 (Option C), §4 S3, §5.3, §5.5 + contract_candidate_004 amendments (R1); release brief
implementation_release_002/BRIEF.md (e8cc9bca...).

One unit = one encoder x one budget. For every one of the 2,921 library slides the unit reads the
encoder's all-tissue bank file (sha256 bound to S0's input_bindings.json), verifies tile order and
coordinates (UNI2: against the accepted route-b mosaic bundle for the 2,604 population slides;
every other 512-px encoder: against the UNI2 bank, slide by slide, STOP on mismatch), predicts with
the six primary heads (fold0..fold4, full) saved by S2 and writes per-head masks/predictions/
probabilities in one packed tile order.

Tumour call: predicted label == class 1 after the five-class argmax (specification §3.3), validated
with the accepted route-b mosaic validator `r2_1_encoder_mosaic_cli.validate_array_slice`
(imported). NOTE (open question Q1 of the release report): the frozen kernel
`r2_1_hoptimus_filter` contains no mask function; the brief's stop condition applies and real S3
runs are on HOLD until the analyst confirms this binding.

Subcommands
  dry-run   readiness (heads COMPLETE + F7 (ii) passed), bank presence/bindings, tile and byte
            projection per unit; writes plan.json; reads no feature payloads
  run       one unit (LSF job): writes s3/<encoder>/b<budget>/
  verify    re-check one unit's outputs (hashes, recounts, coordinate identity, zero-tile lists)
  jobspec   one shell command per line for the analyst (lsfq pool format) + jobspec.json

Writes only under $PROSPER_OUT/r2_1_filter_tf2_20261016/s3 (destinations must be new or empty).
Never submits jobs, never runs retrieval.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import gzip
import importlib.util
import json
import pickle
import socket
import sys
import time
import traceback
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
S2 = _load("r2_1_tf2_s2_heads", SCRIPTS / "r2_1_tf2_s2_heads.py")

from prosper.evaluation import r2_1_encoder_mosaic_cli as mosaic_rule  # noqa: E402
from prosper.evaluation import feature_bank_audit as legacy  # noqa: E402

RUN_NAME = S0.RUN_NAME
STAGE = "s3"
SLOTS = S2.PRIMARY_SLOTS                      # fold0..fold4, full
BUDGETS = S2.BUDGETS
ENCODERS = tuple(S2.ENCODERS)
TUMOUR_CLASS = 1                              # specification §3.3 "tumour = class 1" (see Q1)
MIN_TILES = 8                                 # accepted --min-tiles rule (applied at S4; flagged here)
AGREEMENT_SAMPLE = 200                        # §5.3
AGREEMENT_SEED = 123
EXPECTED_LIBRARY = 2921
EXPECTED_POPULATION = 2604
BANK_ROLE = {"uni2": "lib_uni2_dir", "virchow2": "lib_virchow2_fp32_dir",
             "h_optimus_1": "lib_h_optimus_1_dir", "conch_v15": "trident_root"}
COORD_REFERENCE = {"virchow2": "uni2", "h_optimus_1": "uni2"}   # 512-px encoders checked vs UNI2
ROUTEB_MOSAIC = "r2_1_route_b_uni2_20261013/mosaic/attempt_001"
UNIT_RESOURCES = {"cpu": 1, "mem_gib": 16, "walltime_min": 120, "gpu": 0}
OFFSET_COLUMNS = ["slide_index", "slide_id", "patient_id", "diagnostic_group", "slide_fold",
                  "tile_start", "tile_stop", "feature_rows", "feature_sha256", "in_population",
                  "query_index", "patient_annotated", "bank_file"]
COUNT_COLUMNS = ["head_slot", "head_id", "library_version", "budget_percent", "slide_index", "slide_id",
                 "patient_id", "diagnostic_group", "in_population", "original_tiles",
                 "predicted_class_1", "predicted_class_2", "predicted_class_3", "predicted_class_4",
                 "predicted_class_5", "retained_tiles", "discarded_tiles", "status"]


class S3Error(RuntimeError):
    """A gate, binding or consistency requirement failed: STOP."""


def require(cond, message):
    if not cond:
        raise S3Error(message)


sha256_file = S0.sha256_file
json_sha = S0.json_sha
write_csv = S0.write_csv
write_json = S0.write_json
now = S0.now


def head_id(encoder, budget, slot):
    return S0.head_id(encoder, budget, slot)


def unit_dir(stage_root: Path, encoder, budget) -> Path:
    return Path(stage_root) / encoder / f"b{budget}"


def code_record():
    paths = {"s3_script": Path(__file__).resolve(), "s2_script": Path(S2.__file__).resolve(),
             "s0_script": Path(S0.__file__).resolve(), "fit": Path(S2.fitter.__file__).resolve(),
             "mosaic_rule": Path(mosaic_rule.__file__).resolve(),
             "legacy_io": Path(legacy.__file__).resolve()}
    return {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in paths.items()}


# ---------------------------------------------------------------- tumour rule (Q1)
def tumour_mask(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Tumour call = predicted label 1 (argmax of five classes); checked with the accepted
    route-b mosaic validator (probability validity, label == argmax + 1, mask == (label == 1))."""
    mask = np.ascontiguousarray(labels == TUMOUR_CLASS)
    mosaic_rule.validate_array_slice(probabilities, labels, mask)
    return mask


# ---------------------------------------------------------------- readiness (heads + F7 ii)
def _f7_files(f7_dir: Path):
    f7_dir = Path(f7_dir)
    csv = f7_dir / "f7_checkpoint_ii.csv"
    js = [p for p in (f7_dir / "f7_checkpoint_ii.json", f7_dir / "f7_checkpoint_ii_FAILED.json") if p.exists()]
    require(csv.exists() and len(js) == 1, f"{f7_dir}: F7 (ii) csv + exactly one json required")
    return csv, js[0]


def unit_readiness(encoder, budget, heads_root: Path, f7_dir: Path) -> dict:
    """A unit is ready iff its six primary heads are COMPLETE (outputs re-hash) and F7 (ii)
    checked all six with no violation naming them. Returns {"ready": bool, "reasons": [...]}."""
    import pandas as pd
    reasons, heads = [], {}
    for slot in SLOTS:
        hid = head_id(encoder, budget, slot)
        d = Path(heads_root) / "heads" / encoder / f"b{budget}" / slot
        c = d / "COMPLETE.json"
        if not c.exists():
            reasons.append(f"{hid}: no COMPLETE.json")
            continue
        comp = json.loads(c.read_text())
        if comp.get("status") != "COMPLETE" or comp.get("head_id") != hid:
            reasons.append(f"{hid}: COMPLETE.json status/head_id")
            continue
        heads[slot] = {"dir": str(d), "complete_sha256": sha256_file(c),
                       "head_pkl_sha256": comp["outputs"]["head.pkl"]["sha256"],
                       "fitted_state_sha256": comp["fitted_state_sha256"]}
    try:
        csv, js = _f7_files(f7_dir)
        rows = pd.read_csv(csv)
        rec = json.loads(js.read_text())
        bad_ids = {v.get("head_id") for v in rec.get("violations", [])}
        for slot in SLOTS:
            hid = head_id(encoder, budget, slot)
            r = rows[rows["head_id"] == hid]
            ok = (len(r) == 1 and r["status"].iat[0] == "checked"
                  and int(r["query_patients_in_saved_training"].iat[0]) == 0
                  and str(r["saved_subset_of_planned"].iat[0]) == "True"
                  and str(r["saved_equals_plan_manifest"].iat[0]) == "True"
                  and str(r["library_version_head_matches"].iat[0]) == "True"
                  and r["library_version"].iat[0] == "v_" + slot)
            if not ok:
                reasons.append(f"{hid}: F7 (ii) row missing or not passed")
            if hid in bad_ids:
                reasons.append(f"{hid}: named in an F7 (ii) violation")
        f7 = {"csv": str(csv), "csv_sha256": sha256_file(csv), "json": str(js), "json_sha256": sha256_file(js)}
    except S3Error as exc:
        reasons.append(str(exc))
        f7 = None
    return {"encoder": encoder, "budget_percent": budget, "ready": not reasons, "reasons": reasons,
            "heads": heads, "f7": f7}


def load_heads(readiness: dict) -> dict:
    out = {}
    for slot in SLOTS:
        h = readiness["heads"][slot]
        d = Path(h["dir"])
        comp = json.loads((d / "COMPLETE.json").read_text())
        for name, e in comp["outputs"].items():
            require(sha256_file(d / name) == e["sha256"], f"{d}/{name} changed after S2")
        data = (d / "head.pkl").read_bytes()
        require(S0.sha256_bytes(data) == h["head_pkl_sha256"], f"{d}/head.pkl sha256")
        head = pickle.loads(data)
        require(S2.fitter._fingerprint(head.estimator) == h["fitted_state_sha256"], f"{d}: fingerprint")
        out[slot] = head
    return out


# ---------------------------------------------------------------- library table
def s0_bound(s0_dir: Path) -> dict:
    """role -> {relative file name -> sha256} from S0 input_bindings.json (per-slide roles)."""
    b = json.loads((Path(s0_dir) / "input_bindings.json").read_text())
    dirs = {d["role"]: d["path"] for d in b["directories"]}
    out = {}
    for f in b["files"]:
        role = f["role"]
        if role in dirs:
            rel = os.path.relpath(f["path"], dirs[role])
            out.setdefault(role, {})[rel] = f["sha256"]
        else:
            out.setdefault("_files", {})[role] = f
    return out


def library_table(s0_dir: Path, inputs: dict, bound: dict):
    """2,921 library slides: population slides first in query_index order, then the others
    in (len, id) order. Columns: OFFSET_COLUMNS minus tile ranges/sha/bank."""
    import pandas as pd
    lib_path = inputs["library_map"]
    require(sha256_file(lib_path) == bound["_files"]["library_map"]["sha256"], "library map changed since S0")
    lib = S0.read_library_map(lib_path)
    q = pd.read_csv(Path(s0_dir) / "query_assignment.csv", dtype={"slide_id": str, "patient_id": str})
    folds = pd.read_csv(Path(s0_dir) / "patient_folds_tf2.csv", dtype={"patient_id": str})
    fold = dict(zip(folds["patient_id"], folds["fold"].astype(int)))
    pool = pd.read_csv(Path(s0_dir) / "pool.csv", dtype={"slide_id": str, "UID.Subject": str})
    annotated = set(pool.loc[pool["in_pool"] == 1, "UID.Subject"])
    L = lib.set_index("Slide.Scan.File")
    require(set(q["slide_id"]) <= set(L.index), "population slide outside the library map")
    q = q.sort_values("query_index", kind="mergesort")
    require((q["patient_id"].to_numpy() == L.loc[q["slide_id"], "UID.Subject"].to_numpy()).all(),
            "population patient differs from the library map")
    pop_ids = q["slide_id"].tolist()
    rest = sorted(set(L.index) - set(pop_ids), key=S0.slide_sort_key)
    rows = []
    qi = dict(zip(q["slide_id"], q["query_index"].astype(int)))
    for s in pop_ids + rest:
        p = L.loc[s, "UID.Subject"]
        rows.append({"slide_id": s, "patient_id": p, "diagnostic_group": L.loc[s, "Tumor.group"],
                     "slide_fold": int(fold.get(p, -1)), "in_population": int(s in qi),
                     "query_index": int(qi.get(s, -1)), "patient_annotated": int(p in annotated)})
    t = pd.DataFrame(rows)
    t.insert(0, "slide_index", np.arange(len(t)))
    return t, q


def bank_file(encoder, inputs, slide) -> tuple[Path, str]:
    root = Path(inputs[BANK_ROLE[encoder]])
    if encoder == "conch_v15":
        rel = f"features_conch_v15/{slide}.h5"
    else:
        rel = f"{slide}.pkl"
    return root / rel, rel


def read_block(encoder, path: Path, slide):
    cfg = S2.ENCODERS[encoder]
    if cfg["format"] == "pickle":
        return S2.pickle_block(path, slide, cfg["column_stop"])
    return S2.h5_block(path, slide)


# ---------------------------------------------------------------- §5.3 / §5.5
def agreement_sample(table, n=None, seed=AGREEMENT_SEED) -> list:
    """Fixed sample of n library slides of unannotated patients (PCG64 seed 123, draw without
    replacement from the (len, id)-sorted eligible list; returned sorted)."""
    n = AGREEMENT_SAMPLE if n is None else n
    elig = sorted(table.loc[table["patient_annotated"] == 0, "slide_id"], key=S0.slide_sort_key)
    require(len(elig) >= n, "fewer eligible slides than the agreement sample size")
    rng = np.random.Generator(np.random.PCG64(seed))
    pick = rng.choice(len(elig), size=n, replace=False)
    return sorted((elig[i] for i in pick), key=S0.slide_sort_key)


def jaccard(a: np.ndarray, b: np.ndarray):
    inter, union = int((a & b).sum()), int((a | b).sum())
    return (inter / union if union else float("nan")), inter, union


def agreement_tables(offsets, masks: dict, sample: list):
    import pandas as pd
    pos = offsets.set_index("slide_id")
    rows = []
    for s in sample:
        a, z = int(pos.loc[s, "tile_start"]), int(pos.loc[s, "tile_stop"])
        full = masks["full"][a:z]
        for slot in SLOTS[:-1]:
            j, inter, union = jaccard(full, masks[slot][a:z])
            rows.append({"slide_id": s, "fold_head_slot": slot, "tiles": z - a,
                         "full_tumour_tiles": int(full.sum()), "fold_tumour_tiles": int(masks[slot][a:z].sum()),
                         "intersection": inter, "union": union, "jaccard": j})
    per = pd.DataFrame(rows)
    summ = []
    for slot, g in per.groupby("fold_head_slot", sort=True):
        j = g["jaccard"].dropna()
        summ.append({"fold_head_slot": slot, "slides": int(len(g)), "jaccard_defined": int(len(j)),
                     "jaccard_undefined_both_empty": int(g["jaccard"].isna().sum()),
                     "jaccard_mean": float(j.mean()) if len(j) else None,
                     "jaccard_median": float(j.median()) if len(j) else None,
                     "jaccard_min": float(j.min()) if len(j) else None,
                     "pooled_tile_jaccard": (float(g["intersection"].sum() / g["union"].sum())
                                             if g["union"].sum() else None)})
    return per, pd.DataFrame(summ)


def query_head_share(q):
    """§5.5: per diagnostic group, share of population queries masked by fold heads vs the full head."""
    import pandas as pd
    rows = []
    for grp, g in list(q.groupby("diagnostic_group", sort=True)) + [("ALL", q)]:
        n = len(g)
        nf = int(g["head_slot"].str.startswith("fold").sum())
        rows.append({"diagnostic_group": grp, "queries": n, "fold_head_queries": nf,
                     "full_head_queries": n - nf, "fold_head_share": nf / n if n else None,
                     "full_head_share": (n - nf) / n if n else None})
    return pd.DataFrame(rows)


def counts_frame(encoder, budget, offsets, preds: dict, masks: dict):
    import pandas as pd
    rows = []
    for slot in SLOTS:
        p, m = preds[slot], masks[slot]
        for r in offsets.itertuples(index=False):
            a, z = int(r.tile_start), int(r.tile_stop)
            c = np.bincount(p[a:z], minlength=6)[1:6]
            kept = int(m[a:z].sum())
            rows.append((slot, head_id(encoder, budget, slot), "v_" + slot, budget, int(r.slide_index),
                         r.slide_id, r.patient_id, r.diagnostic_group, int(r.in_population), z - a,
                         *(int(v) for v in c), kept, z - a - kept,
                         "nonempty" if kept else "zero_predicted_tumour"))
    return pd.DataFrame(rows, columns=COUNT_COLUMNS)


def library_versions_frame(counts):
    lv = counts[["library_version", "head_id", "slide_index", "slide_id", "diagnostic_group",
                 "in_population", "original_tiles", "retained_tiles"]].copy()
    lv["zero_tile"] = (lv["retained_tiles"] == 0).astype(int)
    lv[f"below_min_tiles_{MIN_TILES}"] = (lv["retained_tiles"] < MIN_TILES).astype(int)
    return lv


def coverage_frame(lv):
    import pandas as pd
    rows = []
    for scope, sub in (("library_all", lv), ("population", lv[lv["in_population"] == 1])):
        for (ver, hid), g in sub.groupby(["library_version", "head_id"], sort=True):
            for grp, gg in list(g.groupby("diagnostic_group", sort=True)) + [("ALL", g)]:
                ot, rt = int(gg["original_tiles"].sum()), int(gg["retained_tiles"].sum())
                rows.append({"scope": scope, "library_version": ver, "head_id": hid, "diagnostic_group": grp,
                             "slides": int(len(gg)), "zero_tile_slides": int(gg["zero_tile"].sum()),
                             f"below_min_tiles_{MIN_TILES}_slides": int(gg[f"below_min_tiles_{MIN_TILES}"].sum()),
                             f"slides_with_ge_{MIN_TILES}_tiles": int(len(gg) - gg[f"below_min_tiles_{MIN_TILES}"].sum()),
                             "original_tiles": ot, "retained_tiles": rt,
                             "retained_fraction": rt / ot if ot else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- dry run
def dry_run(s0_dir: Path, heads_root: Path, f7_dir: Path, out: Path, prosper_out: str, *,
            encoders=ENCODERS, inputs=None, log=print) -> dict:
    import pandas as pd
    t0 = time.time()
    S2.verify_s0(Path(s0_dir))
    inputs = inputs or S0.default_inputs(S0.HAORAN, prosper_out)
    bound = s0_bound(s0_dir)
    table, q = library_table(s0_dir, inputs, bound)
    require(len(table) == EXPECTED_LIBRARY and int(table["in_population"].sum()) == EXPECTED_POPULATION,
            f"library {len(table)} / population {int(table['in_population'].sum())} != 2921 / 2604")
    rb = Path(prosper_out) / ROUTEB_MOSAIC
    rb_off = pd.read_csv(rb / "slide_offsets.csv", dtype={"slide_id": str}) if (rb / "slide_offsets.csv").exists() else None
    units = []
    for enc in ENCODERS:
        cfg = S2.ENCODERS[enc]
        role = BANK_ROLE[enc]
        files, missing, unbound, size = {}, [], [], 0
        for s in table["slide_id"]:
            p, rel = bank_file(enc, inputs, s)
            if not p.exists():
                missing.append(s)
                continue
            files[s] = p.stat().st_size
            size += files[s]
            if rel not in bound.get(role, {}):
                unbound.append(s)
        # tile projection: exact for population slides (route-b UNI2 mosaic offsets, same grid),
        # bytes-per-tile ratio for the rest (estimate)
        tiles_est = None
        if enc != "conch_v15" and rb_off is not None and not missing:
            known = dict(zip(rb_off["slide_id"], rb_off["feature_rows"].astype(int)))
            bpt = sum(files[s] for s in known) / sum(known.values())
            tiles_est = int(sum(known.values()) + sum(round(files[s] / bpt) for s in files if s not in known))
        for b in BUDGETS:
            rd = unit_readiness(enc, b, heads_root, f7_dir)
            reasons = list(rd["reasons"])
            if missing:
                reasons.append(f"{len(missing)} library slides lack {enc} bank files")
            if unbound:
                reasons.append(f"{len(unbound)} bank files not bound at S0")
            proj = None
            if tiles_est is not None:
                per_head = tiles_est * (5 * 4 + 8 + 1)
                proj = {"tiles_estimate": tiles_est,
                        "bytes_estimate": int(per_head * len(SLOTS) + tiles_est * 16)}
            units.append({"encoder": enc, "budget_percent": b, "status": "ready" if not reasons else "blocked",
                          "reasons": reasons, "bank_role": role, "bank_dir": inputs[role],
                          "bank_files": len(files), "bank_bytes": size, "missing": missing[:20],
                          "unbound": unbound[:20], "width": cfg["width"], "format": cfg["format"],
                          "coordinate_check": ("route-b accepted mosaic (2,604 population slides) + S0 sha256"
                                               if enc == "uni2" else
                                               f"slide-by-slide equality with the {COORD_REFERENCE[enc]} bank"
                                               if enc in COORD_REFERENCE else "own 1024-px trident grid (h5)"),
                          "heads": rd["heads"], "f7": rd["f7"], "projection": proj,
                          "out_dir": str(unit_dir(S0.stage_root(prosper_out) / STAGE, enc, b))})
    plan = {"schema_version": 1, "run": f"{RUN_NAME}/{STAGE}", "created": now(), "dry_run": True,
            "contract": S0.CONTRACT_SHA, "s0": {"dir": str(s0_dir), "complete_sha256": sha256_file(Path(s0_dir) / "COMPLETE.json")},
            "heads_root": str(heads_root), "f7_dir": str(f7_dir),
            "library": {"slides": len(table), "population": int(table["in_population"].sum()),
                        "non_population": int((table["in_population"] == 0).sum()),
                        "non_population_groups": table.loc[table["in_population"] == 0, "diagnostic_group"].value_counts().to_dict(),
                        "unannotated_patient_slides": int((table["patient_annotated"] == 0).sum())},
            "agreement_sample": {"n": AGREEMENT_SAMPLE, "seed": AGREEMENT_SEED,
                                 "universe": "library slides whose patient is not annotated (no labelled pool tile)",
                                 "slides": agreement_sample(table)},
            "tumour_rule": {"tumour_class": TUMOUR_CLASS, "validator": "r2_1_encoder_mosaic_cli.validate_array_slice",
                            "open_question": "Q1: frozen kernel r2_1_hoptimus_filter has no mask function"},
            "units": units, "resources_per_unit": UNIT_RESOURCES, "code": code_record(),
            "seconds": round(time.time() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    write_json(plan, out / "plan.json")
    write_csv(table, out / "library_table.csv")
    log(json.dumps({"units": [(u["encoder"], u["budget_percent"], u["status"],
                               (u["projection"] or {}).get("bytes_estimate")) for u in units]}))
    return plan


# ---------------------------------------------------------------- run one unit
def run_unit(encoder, budget, s0_dir: Path, heads_root: Path, f7_dir: Path, out: Path, prosper_out: str, *,
             inputs=None, routeb_mosaic: Path | None = None, expected=(EXPECTED_LIBRARY, EXPECTED_POPULATION),
             log=print) -> dict:
    import pandas as pd
    t0 = time.time()
    started = now()
    out.mkdir(parents=True, exist_ok=True)
    try:
        rd = unit_readiness(encoder, budget, heads_root, f7_dir)
        require(rd["ready"], f"{encoder}/b{budget} not ready: {rd['reasons']}")
        complete_s0 = S2.verify_s0(Path(s0_dir))
        inputs = inputs or S0.default_inputs(S0.HAORAN, prosper_out)
        bound = s0_bound(s0_dir)
        table, q = library_table(s0_dir, inputs, bound)
        require((len(table), int(table["in_population"].sum())) == tuple(expected),
                f"library/population sizes {len(table)}/{int(table['in_population'].sum())} != {expected}")
        heads = load_heads(rd)
        cfg = S2.ENCODERS[encoder]
        width = cfg["width"]
        role = BANK_ROLE[encoder]
        rb = None
        if encoder == "uni2":
            rbdir = Path(routeb_mosaic) if routeb_mosaic else Path(prosper_out) / ROUTEB_MOSAIC
            rb_off = pd.read_csv(rbdir / "slide_offsets.csv", dtype={"slide_id": str, "feature_sha256": str})
            rb = {"dir": str(rbdir), "offsets": rb_off.set_index("slide_id"),
                  "coordinates": np.load(rbdir / "coordinates.npy", mmap_mode="r"),
                  "sha256": {n: sha256_file(rbdir / n) for n in ("slide_offsets.csv", "coordinates.npy")}}
        ref_enc = COORD_REFERENCE.get(encoder)
        coords_all, probs = [], {s: [] for s in SLOTS}
        preds, masks = {s: [] for s in SLOTS}, {s: [] for s in SLOTS}
        offsets, start, reads, ref_reads, rb_checked = [], 0, [], [], 0
        for r in table.itertuples(index=False):
            path, rel = bank_file(encoder, inputs, r.slide_id)
            require(rel in bound.get(role, {}), f"{encoder} {r.slide_id}: bank file not bound at S0")
            raw, xy, rec = read_block(encoder, path, r.slide_id)
            require(rec["sha256"] == bound[role][rel], f"{encoder} {r.slide_id}: bank sha256 differs from S0")
            reads.append({"slide_id": r.slide_id, "sha256": rec["sha256"], "rows": rec["rows"]})
            if rb is not None and r.slide_id in rb["offsets"].index:
                o = rb["offsets"].loc[r.slide_id]
                require(o["feature_sha256"] == rec["sha256"], f"{r.slide_id}: sha256 differs from route-b mosaic")
                ref = np.asarray(rb["coordinates"][int(o["tile_start"]):int(o["tile_stop"])])
                require(ref.shape == xy.shape and np.array_equal(ref, xy),
                        f"{r.slide_id}: coordinates differ from the accepted route-b mosaic")
                rb_checked += 1
            if ref_enc is not None:
                rpath, rrel = bank_file(ref_enc, inputs, r.slide_id)
                rrole = BANK_ROLE[ref_enc]
                require(rrel in bound.get(rrole, {}), f"{ref_enc} {r.slide_id}: reference bank not bound")
                _, rxy, rrec = read_block(ref_enc, rpath, r.slide_id)
                require(rrec["sha256"] == bound[rrole][rrel], f"{ref_enc} {r.slide_id}: reference sha256")
                require(rxy.shape == xy.shape and np.array_equal(rxy, xy),
                        f"STOP: {encoder} {r.slide_id} coordinates differ from the {ref_enc} bank")
                ref_reads.append(r.slide_id)
            n = len(xy)
            for slot in SLOTS:
                p, lab = S2.predict(heads[slot], raw, width)
                m = tumour_mask(p, lab)
                probs[slot].append(p); preds[slot].append(lab); masks[slot].append(m)
            coords_all.append(xy)
            offsets.append({"slide_index": int(r.slide_index), "slide_id": r.slide_id, "patient_id": r.patient_id,
                            "diagnostic_group": r.diagnostic_group, "slide_fold": int(r.slide_fold),
                            "tile_start": start, "tile_stop": start + n, "feature_rows": n,
                            "feature_sha256": rec["sha256"], "in_population": int(r.in_population),
                            "query_index": int(r.query_index), "patient_annotated": int(r.patient_annotated),
                            "bank_file": str(path)})
            start += n
            del raw
        if rb is not None:
            require(rb_checked == len(rb["offsets"]), f"route-b mosaic slides checked {rb_checked} != {len(rb['offsets'])}")
        off = pd.DataFrame(offsets, columns=OFFSET_COLUMNS)
        coordinates = np.ascontiguousarray(np.concatenate(coords_all).astype(np.int64))
        np.save(out / "coordinates.npy", coordinates, allow_pickle=False)
        write_csv(off, out / "slide_offsets.csv")
        cat = {}
        for slot in SLOTS:
            P = np.ascontiguousarray(np.concatenate(probs[slot]).astype(np.float32))
            Y = np.ascontiguousarray(np.concatenate(preds[slot]).astype(np.int64))
            M = np.ascontiguousarray(np.concatenate(masks[slot]).astype(bool))
            require(len(P) == len(Y) == len(M) == len(coordinates), f"{slot}: packed length")
            mosaic_rule.validate_array_slice(P, Y, M)
            np.save(out / f"{slot}_mask.npy", M, allow_pickle=False)
            np.save(out / f"{slot}_predictions.npy", Y, allow_pickle=False)
            np.save(out / f"{slot}_probabilities.npy", P, allow_pickle=False)
            cat[slot] = (Y, M)
            probs[slot] = preds[slot] = masks[slot] = None
        counts = counts_frame(encoder, budget, off, {s: cat[s][0] for s in SLOTS}, {s: cat[s][1] for s in SLOTS})
        write_csv(counts, out / "mask_counts.csv.gz")
        lv = library_versions_frame(counts)
        write_csv(lv, out / "library_versions.csv")
        write_csv(coverage_frame(lv), out / "coverage.csv")
        sample = agreement_sample(table)
        per, summ = agreement_tables(off, {s: cat[s][1] for s in SLOTS}, sample)
        write_csv(per, out / "mask_agreement_per_slide.csv")
        write_csv(summ, out / "mask_agreement.csv")
        write_csv(query_head_share(q), out / "query_head_share.csv")
        write_json({"schema_version": 1, "unit": f"{encoder}/b{budget}", "created": now(),
                    "contract": S0.CONTRACT_SHA,
                    "s0": {"dir": str(s0_dir), "complete_sha256": sha256_file(Path(s0_dir) / "COMPLETE.json"),
                           "input_bindings_sha256": sha256_file(Path(s0_dir) / "input_bindings.json")},
                    "heads": rd["heads"], "f7_checkpoint_ii": rd["f7"],
                    "bank": {"role": role, "dir": inputs[role], "files": len(reads),
                             "aggregate_sha256": json_sha([[x["slide_id"], x["sha256"]] for x in reads])},
                    "coordinate_reference": ({"kind": "route-b mosaic", **{k: v for k, v in rb.items() if k in ("dir", "sha256")},
                                              "slides_checked": rb_checked} if rb is not None else
                                             {"kind": f"{ref_enc} bank (S0-bound sha256)", "slides_checked": len(ref_reads)}
                                             if ref_enc else {"kind": "none (own grid)"}),
                    "library_map": bound["_files"]["library_map"],
                    "code": code_record(),
                    "environment": {"python": sys.version, "host": socket.gethostname(),
                                    "lsf_job": os.environ.get("LSB_JOBID")},
                    "started": started, "finished": now()}, out / "input_bindings.json")
        outputs = {p.name: {"sha256": sha256_file(p), "size": p.stat().st_size}
                   for p in sorted(out.iterdir()) if p.is_file()}
        zero = {("v_" + s): sorted(counts.loc[(counts["head_slot"] == s) & (counts["retained_tiles"] == 0), "slide_id"],
                                   key=S0.slide_sort_key) for s in SLOTS}
        marker = {"schema_version": 1, "status": "COMPLETE", "unit": f"{encoder}/b{budget}",
                  "encoder": encoder, "budget_percent": budget, "tiles": int(len(coordinates)),
                  "slides": int(len(off)), "population_slides": int(off["in_population"].sum()),
                  "retained_tiles": {"v_" + s: int(cat[s][1].sum()) for s in SLOTS},
                  "zero_tile_slides": zero, "outputs": outputs,
                  "tumour_rule": {"tumour_class": TUMOUR_CLASS,
                                  "validator": "r2_1_encoder_mosaic_cli.validate_array_slice"},
                  "seconds": round(time.time() - t0, 1), "created": now()}
        write_json(marker, out / "COMPLETE.json")
        log(json.dumps({"unit": marker["unit"], "status": "COMPLETE", "tiles": marker["tiles"],
                        "seconds": marker["seconds"]}))
        return marker
    except Exception as exc:
        write_json({"schema_version": 1, "status": "FAILED", "unit": f"{encoder}/b{budget}",
                    "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                    "created": now(), "seconds": round(time.time() - t0, 1)}, out / "FAILED.json")
        raise


# ---------------------------------------------------------------- verify
def verify_unit(udir: Path, *, reference_unit: Path | None = None, log=print) -> dict:
    """Re-hash every output, recount masks vs mask_counts, recompute zero-tile lists, and (if given)
    check coordinate identity against a reference unit (e.g. UNI2 for Virchow2)."""
    import pandas as pd
    udir = Path(udir)
    comp = json.loads((udir / "COMPLETE.json").read_text())
    require(comp["status"] == "COMPLETE", f"{udir}: not COMPLETE")
    checks = {}
    for name, e in comp["outputs"].items():
        require(sha256_file(udir / name) == e["sha256"], f"{udir}/{name} changed")
    checks["outputs_rehash"] = len(comp["outputs"])
    off = pd.read_csv(udir / "slide_offsets.csv", dtype={"slide_id": str, "feature_sha256": str})
    coords = np.load(udir / "coordinates.npy", mmap_mode="r")
    require(int(off["tile_stop"].iat[-1]) == len(coords) == comp["tiles"], "tile count")
    require((off["tile_stop"] - off["tile_start"] == off["feature_rows"]).all(), "offset ranges")
    require((off["tile_start"].iloc[1:].to_numpy() == off["tile_stop"].iloc[:-1].to_numpy()).all(), "contiguity")
    require((off["feature_rows"] > 0).all(), "slide with zero bank tiles")
    counts = pd.read_csv(udir / "mask_counts.csv.gz", dtype={"slide_id": str})
    zero = {}
    for slot in SLOTS:
        M = np.load(udir / f"{slot}_mask.npy")
        Y = np.load(udir / f"{slot}_predictions.npy")
        require(np.array_equal(M, Y == TUMOUR_CLASS), f"{slot}: mask != (prediction == 1)")
        kept = np.add.reduceat(M.astype(np.int64), off["tile_start"].to_numpy())
        c = counts[counts["head_slot"] == slot].sort_values("slide_index")
        require(np.array_equal(c["retained_tiles"].to_numpy(), kept), f"{slot}: retained counts differ")
        require(int(M.sum()) == comp["retained_tiles"]["v_" + slot], f"{slot}: retained total")
        zero["v_" + slot] = sorted(off["slide_id"][kept == 0], key=S0.slide_sort_key)
        require(zero["v_" + slot] == comp["zero_tile_slides"]["v_" + slot], f"{slot}: zero-tile list")
    checks["masks_recounted"] = list(SLOTS)
    if reference_unit is not None:
        ref = Path(reference_unit)
        roff = pd.read_csv(ref / "slide_offsets.csv", dtype={"slide_id": str})
        rc = np.load(ref / "coordinates.npy", mmap_mode="r")
        require(list(roff["slide_id"]) == list(off["slide_id"]) and
                (roff["feature_rows"].to_numpy() == off["feature_rows"].to_numpy()).all(),
                "slide order / tile counts differ from the reference unit")
        require(np.array_equal(np.asarray(rc), np.asarray(coords)), "coordinates differ from the reference unit")
        checks["coordinates_equal_reference_unit"] = str(ref)
    res = {"unit": comp["unit"], "checks": checks, "zero_tile_counts": {k: len(v) for k, v in zero.items()},
           "pass": True, "created": now()}
    log(json.dumps(res))
    return res


# ---------------------------------------------------------------- job spec
def jobspec(plan_path: Path, out: Path, prosper_out: str, *, log=print) -> dict:
    plan = json.loads(Path(plan_path).read_text())
    repo = str(S0.REPO)
    py = "/HOME/miniconda3/bin/python3"
    env = (f"export PROSPER_OUT={prosper_out} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
           "NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "
           "LSB_JOB_REPORT_MAIL=N")
    lines, units = [], []
    for u in plan["units"]:
        cmd = (f"{env}; cd {repo} && {py} scripts/r2_1_tf2_s3_masks.py run --encoder {u['encoder']} "
               f"--budget {u['budget_percent']} --s0 {plan['s0']['dir']} --heads-root {plan['heads_root']} "
               f"--f7 {plan['f7_dir']} --out {u['out_dir']}")
        units.append({"encoder": u["encoder"], "budget_percent": u["budget_percent"], "status": u["status"],
                      "reasons": u["reasons"], "command": cmd})
        if u["status"] == "ready":
            lines.append(cmd)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cmds.txt").write_text("".join(x + "\n" for x in lines))
    spec = {"schema_version": 1, "stage": "S3", "created": now(), "plan": str(plan_path),
            "plan_sha256": sha256_file(plan_path), "cmds": str(out / "cmds.txt"),
            "cmds_sha256": sha256_file(out / "cmds.txt"), "lines": len(lines),
            "shape_per_line": UNIT_RESOURCES, "lsfq_shape_args": "-n 1 -m 16 -W 2h",
            "cwd": repo, "units": units,
            "order": "independent units; Virchow2/H-Optimus-1 read the UNI2 bank for the coordinate check "
                     "(no dependency on the UNI2 S3 job)",
            "after_completion": "verify each unit (Virchow2 with --reference-unit s3/uni2/b<budget>)",
            "hold": "HOLD until the lead answers Q1 (tumour-rule binding; brief stop condition)",
            "notes": ["one command per line; lsfq pool create NAME --tasks cmds.txt --cwd <repo> -n 1 -m 16 -W 2h",
                      "no automatic retries (contract §9): use --max-attempts 1 or equivalent",
                      "NOT SUBMITTED BY THE IMPLEMENTER"]}
    write_json(spec, out / "jobspec.json")
    log(json.dumps({"lines": len(lines), "cmds": str(out / "cmds.txt")}))
    return spec


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run")
    for p in (d,):
        p.add_argument("--s0", required=True)
        p.add_argument("--heads-root", required=True, help="the S2 stage dir holding heads/<enc>/b<b>/<slot>")
        p.add_argument("--f7", required=True, help="S2 F7 checkpoint (ii) output dir")
        p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--encoder", required=True, choices=ENCODERS)
    r.add_argument("--budget", required=True, type=int, choices=BUDGETS)
    r.add_argument("--s0", required=True)
    r.add_argument("--heads-root", required=True)
    r.add_argument("--f7", required=True)
    r.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--unit", required=True)
    v.add_argument("--reference-unit", default=None)
    v.add_argument("--out", required=True)
    j = sub.add_parser("jobspec")
    j.add_argument("--plan", required=True)
    j.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prosper_out = os.environ.get("PROSPER_OUT")
    if not prosper_out:
        sys.exit("PROSPER_OUT must be exported")
    out = S0.check_out_dir(Path(a.out), prosper_out, STAGE)
    if a.cmd == "dry-run":
        dry_run(Path(a.s0), Path(a.heads_root), Path(a.f7), out, prosper_out)
        return 0
    if a.cmd == "run":
        m = run_unit(a.encoder, a.budget, Path(a.s0), Path(a.heads_root), Path(a.f7), out, prosper_out)
        return 0 if m["status"] == "COMPLETE" else 1
    if a.cmd == "verify":
        res = verify_unit(Path(a.unit), reference_unit=Path(a.reference_unit) if a.reference_unit else None)
        out.mkdir(parents=True, exist_ok=True)
        write_json(res, out / "verify.json")
        return 0
    jobspec(Path(a.plan), out, prosper_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
