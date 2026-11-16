#!/usr/bin/env python3
"""R2.1-TF2 stage S4: tumour-filtered tile sets BY INDEX (no re-extraction, no filtered feature
copies), the per-query eligibility table, the storage gate and the TITAN (GPU) job specification.

Binding specification: r2_1_filter_tf2_20261016/contract_candidate_003/SPECIFICATION.md §3.3, §3.4 (Option C),
§3.5, §4 S4, §7 F6/F7(iii), §9 (storage <= 200 GB incremental, index-based subsets) +
contract_candidate_004 R1/R2; release brief implementation_release_002/BRIEF.md (e8cc9bca...);
analyst addendum (eligibility schema) and LEAD_DECISIONS_001.md (Q2 masks applied at load time in
S5; Q3 population-only libraries; Q4 min-tiles parameter, production = 1).

Per encoder x budget (from a COMPLETE S3 unit), s4/index/<enc>/b<b>/:
  * v_<slot>.npz        one int32 array per population slide with >= min_tiles retained tiles:
                        the retained tile positions (rows of the slide's all-tissue bank pickle)
                        under head <slot>; the six library versions (Option C)
  * queries.npz         one array per returned population query: its positions under the head of
                        s0/query_assignment.csv:head_slot (= its slide's array in its own version)
  * _summary_<set>.csv  per slide: retained tiles, written, reason, warning (1..7 tiles), index sha256
  * COMPLETE.json       sha256 of every file
The S5 element loads the all-tissue bank pickle and applies the positions in memory
(`load_time_subset`, identical to r2_1_filter_features_cli.filtered_bytes' subset). The query-only
arm's library is the unfiltered bank restricted to the 2,604 population (no index).
Zero retained tiles = not written (non-return / absent from that version); 1..min_tiles-1 retained
tiles are not written when min_tiles > 1 (F3b replay: 8); production min_tiles = 1 keeps every
nonempty mask and flags 1..7-tile slides with a warning (specification §3.3).

Subcommands
  project      projected incremental bytes for the whole release -> PASS/STOP against 200 GB
  eligibility  s4/eligibility/<enc>/b<b>.csv.gz (analyst addendum schema)
  run          one encoder x budget: the index files above (LSF job; reads S3 outputs only)
  verify       re-check one unit's S4 outputs
  jobspec      one shell command per line for the analyst + jobspec.json
  titan-spec   TITAN slide-embedding job specification for masked CONCH 1024-px tile sets (spec only)
Writes only under $PROSPER_OUT/r2_1_filter_tf2_20261016/s4. Never submits jobs or runs retrieval/TITAN.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import importlib.util
import json
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
S3 = _load("r2_1_tf2_s3_masks", SCRIPTS / "r2_1_tf2_s3_masks.py")

from prosper.evaluation import r2_1_filter_features_cli as ff  # noqa: E402
from prosper.evaluation import r2_1_filter_eval_cli as ev  # noqa: E402

RUN_NAME = S0.RUN_NAME
STAGE = "s4"
SLOTS = S3.SLOTS
BUDGETS = S3.BUDGETS
ENCODERS = S3.ENCODERS
SIM_ENCODERS = ("uni2", "virchow2", "h_optimus_1")     # mosaic pickles for OT / BoB
MIN_TILES = 1                                         # LEAD_DECISIONS_001 Q4: production; F3b replay uses 8
WARN_BELOW = 8                                        # 1..7 retained tiles -> written with a warning
STORAGE_BOUND_BYTES = 200 * 10**9                     # specification §9
INDEX_BYTES_PER_TILE = 4                              # int32 positions
NPZ_ENTRY_OVERHEAD = 200                              # zip member + npy header, per array (estimate)
H5_TILE_BYTES = 768 * 4 + 16                          # CONCH features + int64 coords (TITAN staging)
ROUTEB_FEATURES = "r2_1_route_b_uni2_20261013/features_qf"
UNIT_RESOURCES = {"cpu": 1, "mem_gib": 8, "walltime_min": 30, "gpu": 0}
SUMMARY_COLUMNS = ["slide_index", "slide_id", "patient_id", "diagnostic_group", "set", "head_slot", "head_id",
                   "budget_percent", "original_tiles", "retained_tiles", "written", "reason", "warning",
                   "feature_sha256", "index_sha256"]
# analyst addendum (2026-10-16, release 002): the schema the S6 evaluator (release 003) consumes
ELIGIBILITY_COLUMNS = ["encoder", "budget_percent", "query", "query_index", "arm", "head_slot", "head_id",
                       "library_version", "query_returned", "n_eligible", "eligible_ids_sha256",
                       "eligible_reference_ids"]
ELIGIBILITY_SEPARATOR = ";"
ARMS = ("query_only", "both_sided")


class S4Error(RuntimeError):
    """A gate, binding or consistency requirement failed: STOP."""


def require(cond, message):
    if not cond:
        raise S4Error(message)


sha256_file = S0.sha256_file
json_sha = S0.json_sha
write_csv = S0.write_csv
write_json = S0.write_json
now = S0.now
head_id = S3.head_id


def code_record():
    paths = {"s4_script": Path(__file__).resolve(), "s3_script": Path(S3.__file__).resolve(),
             "s2_script": Path(S2.__file__).resolve(), "s0_script": Path(S0.__file__).resolve(),
             "filter_features_cli": Path(ff.__file__).resolve(), "filter_eval_cli": Path(ev.__file__).resolve()}
    return {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in paths.items()}


def index_root(stage_root: Path, encoder, budget) -> Path:
    return Path(stage_root) / "index" / encoder / f"b{budget}"


def verify_s3_unit(udir: Path) -> dict:
    comp = json.loads((Path(udir) / "COMPLETE.json").read_text())
    require(comp.get("status") == "COMPLETE", f"{udir}: S3 unit not COMPLETE")
    for name, e in comp["outputs"].items():
        require(sha256_file(Path(udir) / name) == e["sha256"], f"{udir}/{name} changed since S3")
    return comp


def index_sha(positions: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(positions, dtype="<i4").tobytes()).hexdigest()


# ---------------------------------------------------------------- the load-time subset (S5 uses it)
def load_time_subset(bank_pickle: Path, positions, *, expected_rows=None):
    """(features[positions], [coords[i] for i in positions]) from an all-tissue bank pickle, the
    same subset r2_1_filter_features_cli.filtered_bytes writes (byte-identical when pickled with
    ff.PICKLE_PROTOCOL; tested). positions=None -> all tiles."""
    feats, coords, _ = ff.load_feature_pickle(Path(bank_pickle))
    require(expected_rows is None or feats.shape[0] == int(expected_rows),
            f"{bank_pickle}: {feats.shape[0]} rows != expected {expected_rows}")
    if positions is None:
        return feats, coords
    idx = np.asarray(positions, dtype=np.int64)
    require(idx.ndim == 1 and len(idx) > 0 and bool((np.diff(idx) > 0).all()) and idx[0] >= 0
            and idx[-1] < feats.shape[0], f"{bank_pickle}: invalid tile positions")
    return feats[idx], [coords[i] for i in idx]


def load_index(npz_path: Path, expected_sha256: str | None = None) -> dict:
    if expected_sha256 is not None:
        require(sha256_file(npz_path) == expected_sha256, f"{npz_path}: sha256 differs from its binding")
    with np.load(npz_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- storage projection
def routeb_retention(prosper_out) -> dict:
    import pandas as pd
    out = {}
    root = Path(prosper_out) / ROUTEB_FEATURES
    for b in BUDGETS:
        fr = []
        for f in range(5):
            p = root / f"budget{b}" / f"fold{f}" / "_filter_summary.csv"
            if p.exists():
                s = pd.read_csv(p)
                fr.append(float(s["retained_tiles"].sum() / s["original_tiles"].sum()))
        out[b] = {"fraction": float(np.mean(fr)) if fr else 1.0, "banks": len(fr),
                  "source": str(root) if fr else "none (upper bound used)"}
    return out


def population_tiles(encoder, inputs, slides) -> tuple[int, int]:
    """(bank bytes of the population slides, estimated tiles) from file sizes (no payload read)."""
    size = 0
    for s in slides:
        p, _ = S3.bank_file(encoder, inputs, s)
        if p.exists():
            size += p.stat().st_size
    w = S2.ENCODERS[encoder]["width"]
    per = H5_TILE_BYTES if encoder == "conch_v15" else 4 * w + 17
    return size, int(size // per)


def project(s3_root: Path, s0_dir: Path, prosper_out: str, *, inputs=None, min_tiles=MIN_TILES,
            bound=STORAGE_BOUND_BYTES, log=print) -> dict:
    """Persistent incremental bytes: index files (six versions + queries) per encoder x budget,
    exact from S3 mask counts where S3 is COMPLETE, else estimated from bank sizes x route-b
    retention. TITAN staging h5 (CONCH) is reported separately as transient per-set scratch."""
    import pandas as pd
    inputs = inputs or S0.default_inputs(S0.HAORAN, prosper_out)
    pop = load_population(s0_dir, inputs)
    pop_ids = pop["slide_id"].tolist()
    ret = routeb_retention(prosper_out)
    units, total = [], 0
    for enc in ENCODERS:
        size, tiles = population_tiles(enc, inputs, pop_ids)
        for b in BUDGETS:
            udir = S3.unit_dir(s3_root, enc, b)
            if (udir / "COMPLETE.json").exists():
                verify_s3_unit(udir)
                c = pd.read_csv(udir / "mask_counts.csv.gz", dtype={"slide_id": str})
                c = c[c["in_population"] == 1]
                per = {}
                for slot in SLOTS:
                    g = c[(c["head_slot"] == slot) & (c["retained_tiles"] >= min_tiles)]
                    per["v_" + slot] = int(g["retained_tiles"].sum() * INDEX_BYTES_PER_TILE + len(g) * NPZ_ENTRY_OVERHEAD)
                per["queries"] = int(np.mean(list(per.values())))      # one array per query ~ one version
                basis = "exact_from_s3_mask_counts (index files)"
            else:
                one = int(tiles * ret[b]["fraction"] * INDEX_BYTES_PER_TILE + len(pop_ids) * NPZ_ENTRY_OVERHEAD)
                per = {**{"v_" + s: one for s in SLOTS}, "queries": one}
                basis = f"estimate: population tiles x route-b retention {ret[b]['fraction']:.4f} x 4 B (S3 not run)"
            ub = int(sum(per.values()))
            total += ub
            titan = None
            if enc == "conch_v15":
                titan = {"transient_per_set_bytes_estimate": int(size * ret[b]["fraction"]),
                         "sets": 1 + len(SLOTS),
                         "note": "trident needs staged h5 files; staged per GPU set and removable after the set's "
                                 "embeddings are verified (see report Q-T); not counted as persistent"}
            units.append({"encoder": enc, "budget_percent": b, "basis": basis, "population_bank_bytes": size,
                          "population_tiles_estimate": tiles, "files_bytes": per, "unit_bytes": ub,
                          "titan_staging": titan})
    rec = {"schema_version": 2, "created": now(), "bound_bytes": bound, "min_tiles": min_tiles,
           "design": "LEAD_DECISIONS_001 Q2: index lists only; masks applied at load time in S5",
           "projected_total_bytes": int(total), "projected_total_gb": round(total / 1e9, 4),
           "projected_total_gib": round(total / 2**30, 4),
           "status": "PASS" if total <= bound else "STOP",
           "rule": "STOP if the projected incremental bytes for the whole release exceed 200 GB (contract §9)",
           "routeb_retention": ret, "units": units, "code": code_record()}
    log(json.dumps({"projected_total_gb": rec["projected_total_gb"], "status": rec["status"]}))
    return rec


# ---------------------------------------------------------------- eligibility (R2 input)
def load_population(s0_dir: Path, inputs: dict):
    b0 = S3.s0_bound(s0_dir)
    p = inputs["retrieval_population"]
    require(sha256_file(p) == b0["_files"]["retrieval_population"]["sha256"], "retrieval population changed since S0")
    return ev.load_population(Path(p))


def eligibility(s3_unit: Path, s0_dir: Path, encoder, budget, *, inputs=None, prosper_out=None,
                min_tiles=MIN_TILES):
    """One row per (query, arm), analyst addendum schema. Eligible references = population slides in
    the arm's library for the query (unfiltered: all 2,604; both-sided: population slides of the
    query's version with >= min_tiles retained tiles), the query itself included; ';'-joined in
    reference_index order; sha256 of the newline-joined sorted ids; same-patient exclusion NOT applied."""
    import pandas as pd
    verify_s3_unit(s3_unit)
    inputs = inputs or S0.default_inputs(S0.HAORAN, prosper_out)
    pop = load_population(s0_dir, inputs)
    q = pd.read_csv(Path(s0_dir) / "query_assignment.csv", dtype={"slide_id": str, "patient_id": str})
    q = q.set_index("slide_id").reindex(pop["slide_id"]).reset_index()
    require(not q["head_slot"].isna().any(), "population query without an S0 assignment")
    c = pd.read_csv(Path(s3_unit) / "mask_counts.csv.gz", dtype={"slide_id": str})
    ids = pop["slide_id"].tolist()
    keep, retained = {}, {}
    for slot in SLOTS:
        g = c[c["head_slot"] == slot].set_index("slide_id")["retained_tiles"]
        require(set(ids) <= set(g.index), f"{slot}: population slides missing from S3 mask counts")
        retained[slot] = g
        keep[slot] = g.reindex(ids).to_numpy() >= min_tiles
    sep = ELIGIBILITY_SEPARATOR
    all_ids, all_sha = sep.join(ids), ev._ids_sha(ids)
    ver_ids = {s: [i for i, k in zip(ids, keep[s]) if k] for s in SLOTS}
    ver_str = {s: sep.join(v) for s, v in ver_ids.items()}
    ver_sha = {s: ev._ids_sha(v) for s, v in ver_ids.items()}
    rows = []
    for r in q.itertuples(index=False):
        slot = r.head_slot
        require(r.library_version_both_sided == "v_" + slot, f"{r.slide_id}: S0 version/head mismatch")
        hid = head_id(encoder, budget, slot)
        returned = int(int(retained[slot].loc[r.slide_id]) >= min_tiles)
        rows.append((encoder, budget, r.slide_id, int(r.query_index), "query_only", slot, hid,
                     r.library_version_query_only, returned, len(ids), all_sha, all_ids))
        rows.append((encoder, budget, r.slide_id, int(r.query_index), "both_sided", slot, hid,
                     r.library_version_both_sided, returned, len(ver_ids[slot]), ver_sha[slot], ver_str[slot]))
    df = pd.DataFrame(rows, columns=ELIGIBILITY_COLUMNS)
    require(len(df) == 2 * len(pop), "eligibility row count")
    return df


def write_eligibility(encoder, budget, s3_root: Path, s0_dir: Path, out_dir: Path, prosper_out: str, *,
                      inputs=None, min_tiles=MIN_TILES, log=print) -> int:
    """s4/eligibility/<enc>/b<b>.csv.gz + b<b>_record.json; out_dir = s4/eligibility/<enc>; the two
    target files must not exist (never overwritten; the other budget may already be present)."""
    root = (S0.stage_root(prosper_out) / STAGE).resolve()
    out_dir = Path(out_dir).absolute()
    require(root in out_dir.resolve().parents, f"--out {out_dir} is not under {root}")
    path, rec = out_dir / f"b{budget}.csv.gz", out_dir / f"b{budget}_record.json"
    require(not path.exists() and not rec.exists(), f"refusing existing {path} / {rec}")
    udir = S3.unit_dir(s3_root, encoder, budget)
    df = eligibility(udir, s0_dir, encoder, budget, inputs=inputs, prosper_out=prosper_out, min_tiles=min_tiles)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(df, path)
    write_json({"created": now(), "encoder": encoder, "budget_percent": budget, "rows": len(df),
                "sha256": sha256_file(path), "columns": ELIGIBILITY_COLUMNS, "min_tiles": min_tiles,
                "s3_unit": str(udir), "s3_complete_sha256": sha256_file(udir / "COMPLETE.json"),
                "semantics": "lead addendum schema; population-only libraries (LEAD_DECISIONS_001 Q3)",
                "code": code_record()}, rec)
    log(json.dumps({"eligibility": str(path), "rows": len(df)}))
    return 0


# ---------------------------------------------------------------- run one unit (index files)
def _save_npz(path: Path, arrays: dict):
    require(not path.exists(), f"refusing existing {path}")
    with open(path, "xb") as fh:
        np.savez(fh, **arrays)


def run_unit(encoder, budget, s3_root: Path, s0_dir: Path, out_root: Path, prosper_out: str, *,
             inputs=None, min_tiles=MIN_TILES, bound=STORAGE_BOUND_BYTES, projection=None, log=print) -> dict:
    """Index files of one unit into out_root = s4/index/<enc>/b<b> (absent or empty)."""
    import pandas as pd
    t0, started = time.time(), now()
    require(encoder in SIM_ENCODERS, f"{encoder}: S4 index files are for {SIM_ENCODERS} (CONCH: TITAN staging)")
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        inputs = inputs or S0.default_inputs(S0.HAORAN, prosper_out)
        proj = projection or project(s3_root, s0_dir, prosper_out, inputs=inputs, min_tiles=min_tiles,
                                     bound=bound, log=log)
        write_json(proj, out_root / "_projection_at_run.json")
        require(proj["status"] == "PASS", f"STOP: projected incremental storage {proj['projected_total_gb']} GB "
                                          f"exceeds the {bound / 1e9:.0f} GB bound (contract §9)")
        udir = S3.unit_dir(s3_root, encoder, budget)
        comp3 = verify_s3_unit(udir)
        S2.verify_s0(Path(s0_dir))
        pop = load_population(s0_dir, inputs)
        off = ff.load_offsets(udir)
        offp = off[off["in_population"] == 1].set_index("slide_id")
        require(set(offp.index) == set(pop["slide_id"]), "S3 population slides != retrieval population")
        masks = {s: np.load(udir / f"{s}_mask.npy", mmap_mode="r") for s in SLOTS}
        preds = {s: np.load(udir / f"{s}_predictions.npy", mmap_mode="r") for s in SLOTS}
        pos = {s: {} for s in SLOTS}
        rows = []
        for sid in pop["slide_id"]:
            r = offp.loc[sid]
            a, z = int(r["tile_start"]), int(r["tile_stop"])
            for slot in SLOTS:
                keep = np.asarray(masks[slot][a:z], dtype=bool)
                require(np.array_equal(keep, np.asarray(preds[slot][a:z]) == S3.TUMOUR_CLASS),
                        f"{sid}: mask != (prediction == 1) for {slot}")
                n = int(keep.sum())
                p = np.flatnonzero(keep).astype("<i4")
                if n >= min_tiles and n > 0:
                    pos[slot][sid] = p
                    rec = (1, "", "retained_tiles<8 (kept, contract §3.3)" if n < WARN_BELOW else "", index_sha(p))
                else:
                    rec = (0, "zero_retained_tiles" if n == 0 else f"retained_tiles<{min_tiles}", "", "")
                rows.append((int(r["slide_index"]), sid, r["patient_id"], r["diagnostic_group"], "v_" + slot, slot,
                             head_id(encoder, budget, slot), budget, z - a, n, rec[0], rec[1], rec[2],
                             r["feature_sha256"], rec[3]))
        summ = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
        files = {}
        for slot in SLOTS:
            name = f"v_{slot}.npz"
            _save_npz(out_root / name, pos[slot])
            sp = out_root / f"_summary_v_{slot}.csv"
            write_csv(summ[summ["set"] == "v_" + slot], sp)
            files["v_" + slot] = {"npz": name, "npz_sha256": sha256_file(out_root / name),
                                  "summary": sp.name, "summary_sha256": sha256_file(sp),
                                  "head_id": head_id(encoder, budget, slot), "slides": len(pos[slot]),
                                  "library_ids_sha256": ev._ids_sha(list(pos[slot]))}
        qa = pd.read_csv(Path(s0_dir) / "query_assignment.csv", dtype={"slide_id": str, "patient_id": str})
        qpos, qrows = {}, []
        for r in qa.sort_values("query_index").itertuples(index=False):
            require(r.library_version_both_sided == "v_" + r.head_slot, f"{r.slide_id}: S0 version/head mismatch")
            s = summ[(summ["set"] == "v_" + r.head_slot) & (summ["slide_id"] == r.slide_id)].iloc[0]
            if int(s["written"]) == 1:
                qpos[r.slide_id] = pos[r.head_slot][r.slide_id]
            qrows.append((int(s["slide_index"]), r.slide_id, r.patient_id, r.diagnostic_group, "queries", r.head_slot,
                          head_id(encoder, budget, r.head_slot), budget, int(s["original_tiles"]),
                          int(s["retained_tiles"]), int(s["written"]), s["reason"], s["warning"],
                          s["feature_sha256"], s["index_sha256"]))
        _save_npz(out_root / "queries.npz", qpos)
        qdf = pd.DataFrame(qrows, columns=SUMMARY_COLUMNS)
        write_csv(qdf, out_root / "_summary_queries.csv")
        files["queries"] = {"npz": "queries.npz", "npz_sha256": sha256_file(out_root / "queries.npz"),
                            "summary": "_summary_queries.csv", "summary_sha256": sha256_file(out_root / "_summary_queries.csv"),
                            "returned": len(qpos)}
        unfiltered = Path(off["bank_file"].iat[0]).parent
        marker = {"schema_version": 2, "status": "COMPLETE", "unit": f"{encoder}/b{budget}", "created": now(),
                  "started": started, "seconds": round(time.time() - t0, 1), "min_tiles": min_tiles,
                  "design": "index lists only; S5 applies them to the all-tissue bank at load time; "
                            "population-only libraries (2,604)", "sets": files,
                  "bank": {"dir": str(unfiltered), "s3_slide_offsets_sha256": comp3["outputs"]["slide_offsets.csv"]["sha256"]},
                  "unfiltered_library": {"dir": str(unfiltered), "slides": "the 2,604 population, all tiles"},
                  "warnings_1_to_7_tiles": {k: int((summ[summ["set"] == k]["warning"] != "").sum())
                                            for k in summ["set"].unique()},
                  "s3_unit": {"dir": str(udir), "complete_sha256": sha256_file(udir / "COMPLETE.json")},
                  "projection_sha256": sha256_file(out_root / "_projection_at_run.json"),
                  "outputs": {p.name: {"sha256": sha256_file(p), "size": p.stat().st_size}
                              for p in sorted(out_root.iterdir()) if p.is_file()},
                  "code": code_record(), "environment": {"host": socket.gethostname(),
                                                         "lsf_job": os.environ.get("LSB_JOBID")}}
        write_json(marker, out_root / "COMPLETE.json")
        log(json.dumps({"unit": marker["unit"], "status": "COMPLETE", "seconds": marker["seconds"]}))
        return marker
    except Exception as exc:
        write_json({"schema_version": 2, "status": "FAILED", "unit": f"{encoder}/b{budget}",
                    "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                    "created": now()}, out_root / "FAILED.json")
        raise


# ---------------------------------------------------------------- verify
def verify_unit(out_root: Path, *, log=print) -> dict:
    """Re-hash outputs; every array equals its summary sha256 and positions; queries equal their
    own version's arrays; summaries agree with S3 mask counts."""
    import pandas as pd
    out_root = Path(out_root)
    comp = json.loads((out_root / "COMPLETE.json").read_text())
    require(comp["status"] == "COMPLETE", f"{out_root}: not COMPLETE")
    for name, e in comp["outputs"].items():
        require(sha256_file(out_root / name) == e["sha256"], f"{out_root}/{name} changed")
    udir = Path(comp["s3_unit"]["dir"])
    counts = pd.read_csv(udir / "mask_counts.csv.gz", dtype={"slide_id": str})
    res = {"unit": comp["unit"], "sets": {}}
    arrays = {}
    for key, v in comp["sets"].items():
        arr = load_index(out_root / v["npz"])
        arrays[key] = arr
        s = pd.read_csv(out_root / v["summary"], dtype={"slide_id": str}, keep_default_na=False)
        w = s[s["written"] == 1]
        require(sorted(arr) == sorted(w["slide_id"]), f"{key}: arrays != summary written set")
        for r in w.itertuples(index=False):
            require(index_sha(arr[r.slide_id]) == r.index_sha256 and len(arr[r.slide_id]) == r.retained_tiles,
                    f"{key}/{r.slide_id}: index sha/length")
        if key != "queries":
            c = counts[counts["head_slot"] == key[2:]].set_index("slide_id")["retained_tiles"]
            require((s.set_index("slide_id")["retained_tiles"] == c.reindex(s["slide_id"]).to_numpy()).all(),
                    f"{key}: retained counts differ from S3")
        res["sets"][key] = {"written": len(arr), "warnings": int((s["warning"] != "").sum())}
    qs = pd.read_csv(out_root / "_summary_queries.csv", dtype={"slide_id": str}, keep_default_na=False)
    for r in qs[qs["written"] == 1].itertuples(index=False):
        require(np.array_equal(arrays["queries"][r.slide_id], arrays["v_" + r.head_slot][r.slide_id]),
                f"query {r.slide_id}: not its own version's array")
    res["pass"] = True
    log(json.dumps(res))
    return res


# ---------------------------------------------------------------- job spec
def jobspec(s3_root: Path, s0_dir: Path, out: Path, prosper_out: str, *, encoders=("uni2", "virchow2"),
            log=print) -> dict:
    repo, py = str(S0.REPO), "/HOME/miniconda3/bin/python3"
    stage = S0.stage_root(prosper_out) / STAGE
    env = (f"export PROSPER_OUT={prosper_out} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
           "NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "
           "LSB_JOB_REPORT_MAIL=N")
    lines = []
    for enc in encoders:
        for b in BUDGETS:
            lines.append(f"{env}; cd {repo} && {py} scripts/r2_1_tf2_s4_subsets.py run --encoder {enc} --budget {b} "
                         f"--s3-root {s3_root} --s0 {s0_dir} --out {index_root(stage, enc, b)} && "
                         f"{py} scripts/r2_1_tf2_s4_subsets.py eligibility --encoder {enc} --budget {b} "
                         f"--s3-root {s3_root} --s0 {s0_dir} --out {stage / 'eligibility' / enc}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "cmds.txt").write_text("".join(x + "\n" for x in lines))
    spec = {"schema_version": 2, "stage": "S4", "created": now(), "cmds": str(out / "cmds.txt"),
            "cmds_sha256": sha256_file(out / "cmds.txt"), "lines": len(lines),
            "per_line": "index files (s4/index/<enc>/b<b>/) then the eligibility table (s4/eligibility/<enc>/b<b>.csv.gz)",
            "shape_per_line": UNIT_RESOURCES, "lsfq_shape_args": "-n 1 -m 8 -W 30", "cwd": repo,
            "min_tiles": MIN_TILES, "depends_on": "the matching S3 unit COMPLETE and verified",
            "gate": "each line recomputes the release storage projection and STOPs above 200 GB",
            "hold": "S3 is on HOLD (author decision on the eight 20x pool slides; LEAD_DECISIONS_001)",
            "after_completion": "verify --unit s4/index/<enc>/b<b>",
            "notes": ["NOT SUBMITTED BY THE IMPLEMENTER", "no automatic retries (--max-attempts 1)"]}
    write_json(spec, out / "jobspec.json")
    log(json.dumps({"lines": len(lines)}))
    return spec


# ---------------------------------------------------------------- TITAN (spec only)
def titan_pins():
    return _load("r2_6_titan_arm1_regen_pins", SCRIPTS / "r2_6_titan_arm1_regen.py")


def titan_stage(set_name, members: list, keep_of: dict, conch_dir: Path, wsi_dir_src: Path, out: Path) -> dict:
    """Stage one masked CONCH set: job_dir/<COORDS>/features_conch_v15/<slide>.h5 with
    features[keep]/coords[keep] (attributes copied), wsi_dir symlinks. Zero retained tiles = not staged."""
    import h5py
    T = titan_pins()
    job = Path(out) / "job_dir"
    feat = job / T.COORDS / "features_conch_v15"
    wsi = Path(out) / "wsi_dir"
    feat.mkdir(parents=True, exist_ok=False)
    wsi.mkdir(parents=True, exist_ok=False)
    staged, skipped = [], []
    for s in members:
        keep = np.asarray(keep_of[s], dtype=bool)
        if not keep.any():
            skipped.append(s)
            continue
        src = Path(conch_dir) / f"{s}.h5"
        with h5py.File(src, "r") as h, h5py.File(feat / f"{s}.h5", "x") as g:
            require(len(h["coords"]) == len(keep), f"{s}: mask length != h5 rows")
            for name in ("coords", "features"):
                dset = g.create_dataset(name, data=np.asarray(h[name][:])[keep])
                for k, v in h[name].attrs.items():
                    dset.attrs[k] = v
        os.symlink(str(Path(wsi_dir_src) / f"{s}.svs"), wsi / f"{s}.svs")
        staged.append({"slide_id": s, "tiles": int(keep.sum()), "source_sha256": sha256_file(src),
                       "staged_sha256": sha256_file(feat / f"{s}.h5")})
    rec = {"set": set_name, "job_dir": str(job), "wsi_dir": str(wsi), "staged": staged, "not_staged_zero_tiles": skipped,
           "trident_command": T.trident_command(str(wsi), str(job)), "executed_by_this_script": False}
    write_json(rec, Path(out) / "stage_record.json")
    return rec


def titan_spec(out: Path, prosper_out: str, s3_root: Path, s0_dir: Path, *, log=print) -> dict:
    T = titan_pins()
    stage = S0.stage_root(prosper_out) / STAGE
    sets = []
    for b in BUDGETS:
        udir = S3.unit_dir(s3_root, "conch_v15", b)
        for name in ["queries"] + [f"v_{s}" for s in SLOTS]:
            d = stage / "titan" / f"b{b}" / name
            sets.append({"budget_percent": b, "set": name,
                         "members": ("2,604 population queries, each masked by its S0 head_slot head"
                                     if name == "queries" else f"the 2,604 population slides masked by conch_v15/b{b}/{name[2:]}"),
                         "head_ids": ("per query: conch_v15/b%d/<head_slot>" % b if name == "queries"
                                      else head_id("conch_v15", b, name[2:])),
                         "stage_dir": str(d), "job_dir": str(d / "job_dir"), "wsi_dir": str(d / "wsi_dir"),
                         "trident_command": T.trident_command(str(d / "wsi_dir"), str(d / "job_dir")),
                         "expected_output": str(d / "job_dir" / T.COORDS / "slide_features_titan"),
                         "requires": [f"S3 unit {udir} COMPLETE (CONCH heads blocked on S1c/S2)",
                                      "titan-stage of this set (transient; see report Q-T)"]})
    unf = {"set": "unfiltered_library", "note": "query-only arm: regenerated unfiltered TITAN library embeddings "
                                                "(r2_6_titan_arm1_regen_20261004, bound at S0 as baseline:titan/titan)"}
    spec = {"schema_version": 2, "stage": "S4-TITAN", "status": "SPEC_ONLY_BLOCKED",
            "encoder_id_in_tables": "titan", "mask_head_id_format": "conch_v15/b<budget>/<slot>",
            "blocked_on": ["S1c CONCH features for the 35 non-library pool slides", "S2 CONCH heads",
                           "S3 conch_v15 units", "F6 (TITAN embedding of unmasked tile sets for 10 slides reproduces "
                                                 "regenerated matrix rows, max rel 1e-6) before any masked set",
                           "S3 HOLD (author decision on the eight 20x pool slides)"],
            "pins": {"source": str(SCRIPTS / "r2_6_titan_arm1_regen.py"),
                     "source_sha256": sha256_file(SCRIPTS / "r2_6_titan_arm1_regen.py"),
                     "TRIDENT": T.TRIDENT, "TRIDENT_PY": T.TRIDENT_PY, "HF_TITAN": T.HF_TITAN, "COORDS": T.COORDS,
                     "SLIDE_CONFIG": T.SLIDE_CONFIG, "patch_encoder": "conch_v15", "slide_encoder": "titan",
                     "mag": 40, "patch_size": 1024, "skip_errors": False, "env": "HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1",
                     "software_provenance": "record r2_6_titan_arm1_regen.software_provenance() + gpu_info.txt per run"},
            "resources_per_set": {"gpu": 1, "cpu": 4, "mem_gib": 32, "walltime_min": 60,
                                  "note": "contract §4: TITAN ~ 1-2 GPU-h total; GPU type recorded per output"},
            "sets": sets + [unf], "distance": "L1 (cityblock) between slide embeddings, accepted R2.6 rule",
            "min_tiles": "zero retained tiles = not staged; 1..7 kept (contract §3.3)",
            "f7_iii": "every TITAN embedding record names the set's library version and head id (stage_record.json)",
            "storage": "staged h5 sets are transient (about 0.8 x the CONCH population bank per set)",
            "staging_rule": ["Q-T authorized (LEAD_CODE_REVIEW_001): one set at a time",
                             "1. titan-stage the set (stage_record.json lists the sha256 of every staged copy)",
                             "2. run the set's trident command (GPU, through lsf-max-throughput)",
                             "3. verify: F6 passed beforehand; every staged copy re-hashes to its stage_record sha256; "
                             "every staged slide has its embedding",
                             "4. write deletion_record.json (set, paths and sha256 of the copies to delete, time, operator) "
                             "BEFORE deleting",
                             "5. delete ONLY the staged byte-copies of inputs (job_dir/<COORDS>/features_conch_v15/*.h5); "
                             "never outputs (slide_features_titan), manifests, records, configs or logs; wsi_dir "
                             "symlinks are kept",
                             "6. stage the next set"],
            "created": now(), "notes": ["NOT RUN; GPU submissions go through lsf-max-throughput by the lead"]}
    out.mkdir(parents=True, exist_ok=True)
    write_json(spec, out / "titan_spec.json")
    log(json.dumps({"titan_sets": len(sets), "status": spec["status"]}))
    return spec


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("project")
    p.add_argument("--s3-root", required=True); p.add_argument("--s0", required=True); p.add_argument("--out", required=True)
    e = sub.add_parser("eligibility")
    r = sub.add_parser("run")
    for x in (e, r):
        x.add_argument("--encoder", required=True, choices=ENCODERS)
        x.add_argument("--budget", required=True, type=int, choices=BUDGETS)
        x.add_argument("--s3-root", required=True)
        x.add_argument("--s0", required=True)
        x.add_argument("--out", required=True)
        x.add_argument("--min-tiles", type=int, default=MIN_TILES)
    v = sub.add_parser("verify"); v.add_argument("--unit", required=True); v.add_argument("--out", required=True)
    j = sub.add_parser("jobspec")
    j.add_argument("--s3-root", required=True); j.add_argument("--s0", required=True); j.add_argument("--out", required=True)
    j.add_argument("--encoders", nargs="+", default=["uni2", "virchow2"], choices=list(SIM_ENCODERS))
    t = sub.add_parser("titan-spec")
    t.add_argument("--s3-root", required=True); t.add_argument("--s0", required=True); t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prosper_out = os.environ.get("PROSPER_OUT")
    if not prosper_out:
        sys.exit("PROSPER_OUT must be exported")
    if a.cmd == "eligibility":
        return write_eligibility(a.encoder, a.budget, Path(a.s3_root), Path(a.s0), Path(a.out), prosper_out,
                                 min_tiles=a.min_tiles)
    out = S0.check_out_dir(Path(a.out), prosper_out, STAGE)
    if a.cmd == "project":
        rec = project(Path(a.s3_root), Path(a.s0), prosper_out)
        out.mkdir(parents=True, exist_ok=True)
        write_json(rec, out / "projection.json")
        return 0 if rec["status"] == "PASS" else 2
    if a.cmd == "run":
        run_unit(a.encoder, a.budget, Path(a.s3_root), Path(a.s0), out, prosper_out, min_tiles=a.min_tiles)
        return 0
    if a.cmd == "verify":
        res = verify_unit(Path(a.unit))
        out.mkdir(parents=True, exist_ok=True)
        write_json(res, out / "verify.json")
        return 0
    if a.cmd == "jobspec":
        jobspec(Path(a.s3_root), Path(a.s0), out, prosper_out, encoders=tuple(a.encoders))
        return 0
    titan_spec(out, prosper_out, Path(a.s3_root), Path(a.s0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
