#!/usr/bin/env python3
"""R2.1-TF2 stage S5: retrieval elements with load-time masking (OT / BoB), pilot, CPU projection and
journal, receipts (F7 iii), Option-C combine, and the F3 / F3b controls.

Binding specification: r2_1_filter_tf2_20261016/contract_candidate_003/SPECIFICATION.md §3.4, §3.5, §4 S5,
§7 F3/F7, §9 + contract_candidate_004 R1 (F7 iii), R7 (pilot / projection / journal), R8 (F3b);
release brief implementation_release_002/BRIEF.md (e8cc9bca...); analyst addendum (matrix sidecar);
LEAD_DECISIONS_001.md (Q2 masks applied at load time; Q3 population-only libraries; Q5 pilot per
encoder x arm x budget).

Element = one query (OT) or a chunk of 25 queries of one head slot (BoB), 1 CPU. The element
(`element` subcommand) loads the library from the all-tissue bank pickles, applies the S4 index
list of the row's library version in memory (r2_1_tf2_s4_subsets.load_time_subset), applies the same
per-slide preparation as prosper.embeddings.similarity_cli._load_slide_data (truncate to the native
width for OT, generator.process_single_slide, drop 'features_raw') and computes each row with the
unchanged prosper.embeddings.similarity_cli._row_vs_library (external mode, num_workers 1). Rows are
written as similarity_cli writes them (pickle of [(reference_id, distance), ...], self = 1e6), with a
provenance record and the F7 (iii) receipt. Thread pins, PYTHONHASHSEED=0, TQDM_DISABLE=1 and host
records follow the accepted route-b packaging; commands are lsfq pool lines.

Option C: a query's both-sided row is computed against its own library version
(s0/query_assignment.csv:library_version_both_sided = v_<head_slot>, population slides with >=
min_tiles retained tiles); its query-only row against the unfiltered bank restricted to the 2,604
population. The query's own tile set is its head's index list (s4 queries.npz).

Subcommands
  pilot           200 population queries, seeded random sampling (PCG64 seed 123); pilot_queries.csv
  specs           elements.csv.gz, library id lists, cmds_pilot.txt / cmds_production.txt (OT),
                  cmds_production_bob.txt (BoB) and jobspec.json for one encoder x budget
  element         (LSF task) compute one element's rows, provenance and receipt
  f7-iii          check every receipt of a jobspec against S0 (version + head id per query)
  combine         Option-C combine of one encoder x budget x arm x method with same-patient exclusion
  project         pilot epilogues -> mean CPU-h per row -> projected total -> PASS/STOP vs 16,000
  journal         consumed CPU-h from epilogues of every finished element -> consumed.json
  controls        element specs for F3 (UNI2 legacy rows, Virchow2-fp32 row vs run_003, BoB bitwise)
                  and F3b (route-b filtered rows through the load-time path), F3b index files
  f3b-prepare     (LSF task) byte-identity of every load-time subset vs the accepted route-b banks
  check-controls  compare control outputs (max rel < 1e-6 + exact eligible top-10; BoB bitwise)
Writes only under $PROSPER_OUT/r2_1_filter_tf2_20261016/s5. Never submits jobs. Pilot rows are not
read here except for their LSF CPU time.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import hashlib
import importlib.util
import json
import pickle
import re
import socket
import subprocess
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
S2 = _load("r2_1_tf2_s2_heads", SCRIPTS / "r2_1_tf2_s2_heads.py")
S3 = _load("r2_1_tf2_s3_masks", SCRIPTS / "r2_1_tf2_s3_masks.py")
S4 = _load("r2_1_tf2_s4_subsets", SCRIPTS / "r2_1_tf2_s4_subsets.py")

from prosper.evaluation import r2_1_filter_eval_cli as ev  # noqa: E402
from prosper.evaluation import r2_1_filter_features_cli as ff  # noqa: E402
from prosper.embeddings import similarity_cli as sc  # noqa: E402

RUN_NAME = S0.RUN_NAME
STAGE = "s5"
SLOTS = S3.SLOTS
BUDGETS = S3.BUDGETS
SIM_ENCODERS = S4.SIM_ENCODERS
ARMS = ("query_only", "both_sided")
METHODS = ("ot", "bob")
METHOD_TAG = {"ot": "ot_cosine", "bob": "yottixel"}
BOB_CHUNK = 25                                   # route-b `--hi 25` chunks
PILOT_N, PILOT_SEED = 200, 123                   # R7
PILOT_BUDGETS = (100, 5)                         # LEAD_DECISIONS_001 Q5: per encoder x arm x budget
CPU_GATE_H = 16000.0                             # §9 / R7
BOB_FALLBACK_CPU_H_PER_ENCODER = 450.0           # specification §9 (used only if no BoB epilogues)
FIXED_OTHER_CPU_H = 150.0                        # specification §9 "heads/masks/evaluation ~ 150"
TILE_DEVIATION_LIMIT = 0.10                      # R7 size-adjustment trigger
SENTINEL = ev.SENTINEL
SHAPES = {"ot": {"cpu": 1, "mem_gib": 32, "walltime_h": 6, "lsfq": "-n 1 -m 32 -W 6h"},
          "bob": {"cpu": 1, "mem_gib": 16, "walltime_h": 8, "lsfq": "-n 1 -m 16 -W 8h"}}
PY = "/HOME/miniconda3/bin/python3"
PROSPER_SEARCH = S0.PROSPER_SEARCH
THREAD_ENV = ("OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
              "BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 "
              "PYTHONDONTWRITEBYTECODE=1 LSB_JOB_REPORT_MAIL=N")
ELEMENT_COLUMNS = ["element_id", "encoder", "budget_percent", "arm", "method", "head_slot", "library_version",
                   "query_head_id", "library_head_id", "line", "n_queries", "query_ids", "pilot", "width",
                   "query_bank_dir", "query_index_file", "query_index_sha256", "library_bank_dir",
                   "library_ids_file", "library_ids_sha256", "library_index_file", "library_index_sha256",
                   "bank_offsets", "rows_dir", "provenance_file", "receipt_file"]
ROUTEB = "r2_1_route_b_uni2_20261013"
LEGACY = {"query_dir": "st2p2/kidrare/features_in/tumor_only", "query": "NULL-20241227-133138",
          "library_filtered": "st2p2/library/features_in/tumor_only",
          "ref_all_tissue": "similarity_st2p2_kidrare_tumor_only_ot/similarity_NULL-20241227-133138.pkl",
          "ref_filtered": "similarity_st2p2_kidrare_tumor_only_both_ot/similarity_NULL-20241227-133138.pkl"}
F3B_N, F3B_SEED, F3B_BUDGET, F3B_MIN_TILES = 3, 123, 5, 8
REL_LIMIT, TOPK = 1e-6, 10


class S5Error(RuntimeError):
    """A gate, binding or consistency requirement failed: STOP."""


def require(cond, message):
    if not cond:
        raise S5Error(message)


sha256_file = S0.sha256_file
json_sha = S0.json_sha
write_csv = S0.write_csv
write_json = S0.write_json
now = S0.now
head_id = S3.head_id


def code_record():
    paths = {"s5_script": Path(__file__).resolve(), "s4_script": Path(S4.__file__).resolve(),
             "s3_script": Path(S3.__file__).resolve(), "filter_eval_cli": Path(ev.__file__).resolve(),
             "filter_features_cli": Path(ff.__file__).resolve(), "similarity_cli": Path(sc.__file__).resolve()}
    return {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in paths.items()}


def git_head():
    try:
        return subprocess.run(["git", "-C", PROSPER_SEARCH, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def stage_dir(prosper_out) -> Path:
    return S0.stage_root(prosper_out) / STAGE


def lib_order(ids) -> list:
    """similarity_cli's reference order: sorted(glob(dir/*.pkl)) == sorted '<id>.pkl' names."""
    return [n[:-4] for n in sorted(f"{i}.pkl" for i in ids)]


def write_ids(path: Path, ids) -> str:
    data = "".join(f"{i}\n" for i in ids).encode()
    ff.write_once(Path(path), data)
    return hashlib.sha256(data).hexdigest()


def read_ids(path: Path, sha: str) -> list:
    require(sha256_file(path) == sha, f"{path}: library id list sha256 differs from its binding")
    return Path(path).read_text().split()


# ---------------------------------------------------------------- pilot (R7)
def pilot_queries(population_ids: list, n=None, seed=PILOT_SEED) -> list:
    """n queries drawn without replacement (PCG64 seed) from the population in query_index order;
    returned in query_index order. The same draw serves every encoder x arm x budget."""
    n = PILOT_N if n is None else n
    require(len(population_ids) >= n, "population smaller than the pilot")
    rng = np.random.Generator(np.random.PCG64(seed))
    pick = np.sort(rng.choice(len(population_ids), size=n, replace=False))
    return [population_ids[i] for i in pick]


def run_pilot(s0_dir: Path, out: Path, prosper_out: str, *, inputs=None) -> dict:
    import pandas as pd
    inputs = inputs or S0.default_inputs(S0.HAORAN, prosper_out)
    pop = S4.load_population(s0_dir, inputs)
    pick = pilot_queries(pop["slide_id"].tolist())
    q = pd.read_csv(Path(s0_dir) / "query_assignment.csv", dtype={"slide_id": str, "patient_id": str}).set_index("slide_id")
    df = q.loc[pick].reset_index()[["query_index", "slide_id", "patient_id", "diagnostic_group", "stratum", "head_slot",
                                    "library_version_both_sided"]]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(df, out / "pilot_queries.csv")
    rec = {"created": now(), "n": PILOT_N, "seed": PILOT_SEED, "generator": "numpy PCG64, choice without replacement "
           "over the 2,604 population in query_index order", "budgets": list(PILOT_BUDGETS),
           "scope": "OT rows; every encoder x arm x budget (LEAD_DECISIONS_001 Q5)",
           "strata": df["stratum"].value_counts().to_dict(), "sha256": sha256_file(out / "pilot_queries.csv"),
           "rule": "reused in production; production commands exclude them; not evaluated or read before S6"}
    write_json(rec, out / "pilot_record.json")
    return rec


# ---------------------------------------------------------------- the element (load-time masking)
def _prepare(generator, feats, sid, truncate):
    """The per-slide steps of similarity_cli._load_slide_data, verbatim in effect."""
    feats = np.asarray(feats)
    if truncate:
        feats = feats[:, :truncate]
    processed = generator.process_single_slide(feats, sid)
    processed.pop("features_raw", None)
    return processed


def _expected_rows(offsets_csv):
    if not offsets_csv:
        return {}
    import pandas as pd
    o = pd.read_csv(offsets_csv, dtype={"slide_id": str})
    return dict(zip(o["slide_id"], o["feature_rows"].astype(int)))


def _index(path, sha):
    return S4.load_index(Path(path), sha) if path else None


def run_element(jobspec_path: Path, element_id: str, prosper_out: str, *, log=print) -> dict:
    t0 = time.time()
    spec, el = _jobspec_elements(jobspec_path)
    row = el[el["element_id"] == element_id]
    require(len(row) == 1, f"unknown element {element_id}")
    e = row.iloc[0].to_dict()
    method, width = e["method"], int(e["width"])
    gen_kwargs = dict(feature_dim=width, verbose=False)
    if method == "ot":
        gen_kwargs.update(metric="cosine", reg=0.0)
    generator = sc.get_slide_embedding_generator(method, **gen_kwargs)
    trunc = width if method == "ot" else None
    rows_expected = _expected_rows(e["bank_offsets"])
    lib_ids = read_ids(Path(e["library_ids_file"]), e["library_ids_sha256"])
    require(lib_ids == lib_order(lib_ids), "library id list is not in similarity_cli order")
    lidx = _index(e["library_index_file"], e["library_index_sha256"])
    qidx = _index(e["query_index_file"], e["query_index_sha256"])
    library, tiles = {}, {}
    for sid in lib_ids:
        pos = None if lidx is None else lidx[sid]
        feats, _ = S4.load_time_subset(Path(e["library_bank_dir"]) / f"{sid}.pkl", pos,
                                       expected_rows=rows_expected.get(sid))
        tiles[sid] = int(feats.shape[0])
        library[sid] = _prepare(generator, feats, sid, trunc)
    load_s = time.time() - t0
    rows_dir = Path(e["rows_dir"])
    rows_dir.mkdir(parents=True, exist_ok=True)
    qids = e["query_ids"].split()
    written = []
    t1 = time.time()
    for q in qids:
        pos = None if qidx is None else qidx[q]
        feats, _ = S4.load_time_subset(Path(e["query_bank_dir"]) / f"{q}.pkl", pos, expected_rows=rows_expected.get(q))
        qdata = _prepare(generator, feats, q, trunc)
        out = sc._row_vs_library(generator, method, q, qdata, library, lib_ids, 1)
        data = pickle.dumps(out)
        ff.write_once(rows_dir / f"similarity_{q}.pkl", data)
        written.append({"query": q, "query_tiles": int(feats.shape[0]), "row_file": str(rows_dir / f"similarity_{q}.pkl"),
                        "sha256": hashlib.sha256(data).hexdigest()})
    prov = {"element_id": element_id, "encoder": e["encoder"], "method": method, "feature_dim": width,
            "metric": "cosine" if method == "ot" else None, "reg": 0.0 if method == "ot" else None,
            "external": True, "load_time_masking": True,
            "query_bank_dir": e["query_bank_dir"], "query_index_file": e["query_index_file"],
            "library_bank_dir": e["library_bank_dir"], "library_ids_file": e["library_ids_file"],
            "library_index_file": e["library_index_file"], "n_library": len(lib_ids),
            "library_tiles_total": int(sum(tiles.values())), "queries": qids, "rows": written,
            "load_seconds": round(load_s, 1), "wall_seconds": round(time.time() - t1, 1), "num_workers": 1,
            "prosper_search_git_head": git_head(), "code": code_record(),
            "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"), "lsf_queue": os.environ.get("LSB_QUEUE"),
            "created": now()}
    Path(e["provenance_file"]).parent.mkdir(parents=True, exist_ok=True)
    ff.write_once(Path(e["provenance_file"]), json.dumps(prov, indent=1).encode())
    rec = receipt(spec, e, written, prosper_out, log=log)
    log(json.dumps({"element": element_id, "rows": len(written), "seconds": round(time.time() - t0, 1)}))
    return rec


# ---------------------------------------------------------------- specs
def element_command(e: dict, jobspec_path: Path, s5: Path, prosper_out: str) -> str:
    host_dir = s5 / "host_records" / f"{e['encoder']}_b{e['budget_percent']}_{e['arm']}_{e['method']}"
    host = (f"mkdir -p {host_dir} && {{ echo \"host=$(hostname)\"; grep '^PRETTY_NAME=' /etc/os-release; "
            f"grep -m1 'model name' /proc/cpuinfo; echo \"lsb_jobid=$LSB_JOBID lsb_queue=$LSB_QUEUE element={e['element_id']}\"; "
            f"env | grep -E '^(OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|NUMEXPR_NUM_THREADS|BLIS_NUM_THREADS|"
            f"VECLIB_MAXIMUM_THREADS|PYTHONHASHSEED|TQDM_DISABLE|LSB_JOB_REPORT_MAIL|PROSPER_OUT)=' | sort | sed 's/^/env:/'; "
            f"date -Is; }} > {host_dir}/{e['element_id']}_${{LSB_JOBID}}.txt")
    run = (f"cd {PROSPER_SEARCH} && {PY} {SCRIPTS / 'r2_1_tf2_s5_jobspec.py'} element --jobspec {jobspec_path} "
           f"--element {e['element_id']}")
    return f"export {THREAD_ENV} PROSPER_OUT={prosper_out}; {host} && {run}"


def build_specs(encoder, budget, s3_unit: Path, s4_unit: Path, s0_dir: Path, pilot_csv: Path, out: Path,
                prosper_out: str, *, inputs=None, log=print) -> dict:
    """All elements of one encoder x budget. `out` = s5/jobspec/<enc>/b<b> (new or empty)."""
    import pandas as pd
    require(encoder in SIM_ENCODERS, f"{encoder}: OT/BoB specs are for {SIM_ENCODERS}")
    inputs = inputs or S0.default_inputs(S0.HAORAN, prosper_out)
    comp3 = S4.verify_s3_unit(s3_unit)
    S4.verify_unit(s4_unit, log=lambda *_: None)
    comp4 = json.loads((Path(s4_unit) / "COMPLETE.json").read_text())
    require(comp4["unit"] == f"{encoder}/b{budget}" == comp3["unit"], "unit identity")
    s5 = stage_dir(prosper_out)
    pop = S4.load_population(s0_dir, inputs)
    pop_ids = pop["slide_id"].tolist()
    qa = pd.read_csv(Path(s0_dir) / "query_assignment.csv", dtype={"slide_id": str, "patient_id": str})
    qs = pd.read_csv(Path(s4_unit) / "_summary_queries.csv", dtype={"slide_id": str}, keep_default_na=False)
    require(list(qs["slide_id"]) == list(qa.sort_values("query_index")["slide_id"]), "S4 query summary order")
    require((qs["head_slot"].to_numpy() == qa.sort_values("query_index")["head_slot"].to_numpy()).all(),
            "S4 query head_slot differs from S0")
    pilot = set(pd.read_csv(pilot_csv, dtype={"slide_id": str})["slide_id"])
    bank = comp4["bank"]["dir"]
    width = S2.ENCODERS[encoder]["width"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "libraries").mkdir()
    lib_files = {"unfiltered": (out / "libraries" / "unfiltered.txt", None, "")}
    lib_files["unfiltered"] = (lib_files["unfiltered"][0], write_ids(lib_files["unfiltered"][0], lib_order(pop_ids)), "")
    for slot in SLOTS:
        v = comp4["sets"]["v_" + slot]
        ids = lib_order(S4.load_index(Path(s4_unit) / v["npz"], v["npz_sha256"]).keys())
        p = out / "libraries" / f"v_{slot}.txt"
        lib_files["v_" + slot] = (p, write_ids(p, ids), str(Path(s4_unit) / v["npz"]))
    qnpz = str(Path(s4_unit) / comp4["sets"]["queries"]["npz"])
    qsha = comp4["sets"]["queries"]["npz_sha256"]
    jobspec_path = out / "jobspec.json"
    elements, lines = [], {"pilot": [], "production": [], "production_bob": []}
    offsets = str(Path(s3_unit) / "slide_offsets.csv")
    for slot in SLOTS:
        members = qs[(qs["head_slot"] == slot) & (qs["written"] == 1)]
        order = members.sort_values("slide_index")["slide_id"].tolist()
        for arm in ARMS:
            ver = "v_" + slot if arm == "both_sided" else "unfiltered"
            lfile, lsha, lnpz = lib_files[ver]
            for method in METHODS:
                set_name = f"{arm}_{method}_{slot}"
                chunks = ([[q] for q in order] if method == "ot" else
                          [order[i:i + BOB_CHUNK] for i in range(0, len(order), BOB_CHUNK)])
                for k, qids in enumerate(chunks, 1):
                    eid = f"{encoder}_b{budget}_{set_name}_{k:05d}"
                    task = f"rows/{encoder}/b{budget}/{arm}_{method}/{slot}"
                    e = {"element_id": eid, "encoder": encoder, "budget_percent": budget, "arm": arm, "method": method,
                         "head_slot": slot, "library_version": ver, "query_head_id": head_id(encoder, budget, slot),
                         "library_head_id": head_id(encoder, budget, slot) if arm == "both_sided" else "",
                         "line": k, "n_queries": len(qids), "query_ids": " ".join(qids),
                         "pilot": int(method == "ot" and budget in PILOT_BUDGETS and qids[0] in pilot),
                         "width": width, "query_bank_dir": bank, "query_index_file": qnpz, "query_index_sha256": qsha,
                         "library_bank_dir": bank, "library_ids_file": str(lfile), "library_ids_sha256": lsha,
                         "library_index_file": lnpz,
                         "library_index_sha256": (comp4["sets"][ver]["npz_sha256"] if lnpz else ""),
                         "bank_offsets": offsets, "rows_dir": str(s5 / task),
                         "provenance_file": str(s5 / "provenance" / task / f"{eid}.json"),
                         "receipt_file": str(s5 / "receipts" / encoder / f"b{budget}" / f"{arm}_{method}" / f"{eid}.json")}
                    elements.append(e)
                    cmd = element_command(e, jobspec_path, s5, prosper_out)
                    key = "production_bob" if method == "bob" else ("pilot" if e["pilot"] else "production")
                    lines[key].append((arm, cmd))
    el = pd.DataFrame(elements, columns=ELEMENT_COLUMNS)
    write_csv(el, out / "elements.csv.gz")
    files = {}
    for arm in ARMS:
        (out / arm).mkdir()
        for key, name in (("pilot", "cmds_pilot.txt"), ("production", "cmds_production.txt"),
                          ("production_bob", "cmds_production_bob.txt")):
            p = out / arm / name
            p.write_text("".join(c + "\n" for a, c in lines[key] if a == arm))
            files[f"{arm}/{name}"] = {"sha256": sha256_file(p), "lines": sum(1 for a, _ in lines[key] if a == arm)}
    spec = {"schema_version": 2, "stage": "S5", "unit": f"{encoder}/b{budget}", "created": now(),
            "contract": S0.CONTRACT_SHA,
            "s3_unit": {"dir": str(s3_unit), "complete_sha256": sha256_file(Path(s3_unit) / "COMPLETE.json")},
            "s4_unit": {"dir": str(s4_unit), "complete_sha256": sha256_file(Path(s4_unit) / "COMPLETE.json")},
            "s0": {"dir": str(s0_dir), "query_assignment_sha256": sha256_file(Path(s0_dir) / "query_assignment.csv")},
            "pilot_csv": {"path": str(pilot_csv), "sha256": sha256_file(pilot_csv)},
            "elements_csv": {"path": str(out / "elements.csv.gz"), "sha256": sha256_file(out / "elements.csv.gz"),
                             "rows": len(el)},
            "libraries": {k: {"ids_file": str(v[0]), "sha256": v[1], "index_file": v[2]} for k, v in lib_files.items()},
            "cmds": files, "shapes": {"ot": SHAPES["ot"], "bob": SHAPES["bob"]},
            "per_element": {"ot": "one query", "bob": f"chunks of {BOB_CHUNK} queries of one head slot"},
            "design": {"both_sided": "the query's own version v_<head_slot>: population slides with >= min_tiles "
                                     "retained tiles, masked at load time by the S4 index list",
                       "query_only": "the unfiltered bank restricted to the 2,604 population",
                       "query": "the query's head index list (s4 queries.npz) applied at load time"},
            "run_order": ["1. controls: cmds_prepare (F3b byte identity) + cmds_ot + cmds_bob, then check-controls PASS",
                          "2. cmds_pilot.txt of both arms for every encoder x budget (OT)",
                          "3. project -> PASS (<= 16,000 CPU-h) and tile-deviation check",
                          "4. cmds_production.txt (OT) and cmds_production_bob.txt (BoB); journal after each batch, "
                          "hard STOP at 16,000 consumed CPU-h",
                          "5. f7-iii then combine per arm x method"],
            "packaging_differences_vs_route_b": [
                "load-time masking element (`element`) instead of prosper_job.sh + similarity_cli --features dirs; "
                "the row computation is similarity_cli._row_vs_library unchanged",
                "lsfq pool lines instead of LSF arrays", "receipt (F7 iii) written by the element itself",
                "route-b requested -M 28672 / -W 600 for OT; this spec uses the brief's OT 32 GiB / 6 h"],
            "notes": ["NOT SUBMITTED BY THE IMPLEMENTER", "no automatic retries (--max-attempts 1)",
                      "pilot rows are not read or evaluated before S6 (timings only)"],
            "code": code_record()}
    write_json(spec, jobspec_path)
    log(json.dumps({"unit": spec["unit"], "elements": len(el), "cmds": files}))
    return spec


# ---------------------------------------------------------------- receipts / F7 (iii)
def _jobspec_elements(jobspec_path: Path):
    import pandas as pd
    spec = json.loads(Path(jobspec_path).read_text())
    p = Path(spec["elements_csv"]["path"])
    require(sha256_file(p) == spec["elements_csv"]["sha256"], "elements.csv.gz changed since the spec")
    el = pd.read_csv(p, dtype=str, keep_default_na=False)
    for c in ("budget_percent", "line", "n_queries", "pilot", "width"):
        el[c] = el[c].astype(int)
    return spec, el


def s0_assignment(s0_dir: Path):
    import pandas as pd
    q = pd.read_csv(Path(s0_dir) / "query_assignment.csv", dtype={"slide_id": str, "patient_id": str})
    return q.set_index("slide_id")


def receipt(spec: dict, e: dict, rows: list, prosper_out: str, *, log=print) -> dict:
    """F7 (iii) receipt: the element's library version and head ids must equal those S0 assigns to
    each of its queries (TF-2 elements); controls carry no S0 assignment."""
    violations = []
    if e["arm"] in ARMS:
        s0 = Path(spec["s0"]["dir"])
        require(sha256_file(s0 / "query_assignment.csv") == spec["s0"]["query_assignment_sha256"],
                "S0 query assignment changed")
        qa = s0_assignment(s0)
        for q in e["query_ids"].split():
            a = qa.loc[q]
            ver = a["library_version_both_sided"] if e["arm"] == "both_sided" else a["library_version_query_only"]
            hid = head_id(e["encoder"], int(e["budget_percent"]), a["head_slot"])
            if ver != e["library_version"] or hid != e["query_head_id"] or (
                    e["arm"] == "both_sided" and e["library_head_id"] != hid):
                violations.append({"query": q, "s0_version": ver, "s0_head": hid,
                                   "element_version": e["library_version"], "element_head": e["query_head_id"]})
    rec = {"schema_version": 2, "element_id": e["element_id"], "created": now(), "host": socket.gethostname(),
           "lsf_job": os.environ.get("LSB_JOBID"), "lsf_queue": os.environ.get("LSB_QUEUE"),
           "encoder": e["encoder"], "budget_percent": int(e["budget_percent"]), "arm": e["arm"], "method": e["method"],
           "library_version": e["library_version"], "query_head_id": e["query_head_id"],
           "library_head_id": e["library_head_id"], "library_ids_sha256": e["library_ids_sha256"],
           "library_index_sha256": e["library_index_sha256"], "query_index_sha256": e["query_index_sha256"],
           "queries": e["query_ids"].split(), "rows": rows,
           "provenance": {"path": e["provenance_file"], "sha256": sha256_file(e["provenance_file"])},
           "f7_iii": {"pass": not violations, "violations": violations,
                      "applies": e["arm"] in ARMS}}
    Path(e["receipt_file"]).parent.mkdir(parents=True, exist_ok=True)
    ff.write_once(Path(e["receipt_file"]), json.dumps(rec, indent=1).encode())
    require(not violations, f"F7 (iii) violation in {e['element_id']}: STOP")
    return rec


def f7_iii(jobspec_path: Path, prosper_out: str, *, arms=ARMS, methods=METHODS) -> dict:
    spec, el = _jobspec_elements(jobspec_path)
    qa = s0_assignment(Path(spec["s0"]["dir"]))
    missing, bad = [], []
    for e in el.itertuples(index=False):
        if e.arm not in arms or e.method not in methods:
            continue
        p = Path(e.receipt_file)
        if not p.exists():
            missing.append(e.element_id)
            continue
        r = json.loads(p.read_text())
        for q in r["queries"]:
            a = qa.loc[q]
            ver = a["library_version_both_sided"] if e.arm == "both_sided" else a["library_version_query_only"]
            hid = head_id(e.encoder, int(e.budget_percent), a["head_slot"])
            if r["library_version"] != ver or r["query_head_id"] != hid or not r["f7_iii"]["pass"]:
                bad.append({"element": e.element_id, "query": q})
    return {"checkpoint": "F7 (iii) after S5 (R1)", "elements": int(len(el)), "missing_receipts": missing[:50],
            "n_missing": len(missing), "violations": bad, "pass": not missing and not bad, "created": now()}


# ---------------------------------------------------------------- combine (Option C)
def matrix_name(encoder, budget, arm, method, masked):
    return (f"combined_similarity_matrix_tf2_{encoder}_b{budget}_{arm}_{METHOD_TAG[method]}"
            f"{'_no_same_patient' if masked else ''}.pkl")


def combine(jobspec_path: Path, arm: str, method: str, out: Path, prosper_out: str, *, inputs=None, log=print) -> dict:
    """Rows of every returned query (each against its own library) -> one query x population matrix
    (NaN outside the row's library) + its _no_same_patient twin (route-b combine-qf rule: population
    patient_id, self keeps the 1e6 sentinel) + rows csv + per-row sidecar JSON (analyst addendum) + record."""
    import pandas as pd
    t0 = time.time()
    spec, el = _jobspec_elements(jobspec_path)
    enc, b = spec["unit"].split("/")
    budget = int(b[1:])
    inputs = inputs or S0.default_inputs(S0.HAORAN, prosper_out)
    pop = S4.load_population(Path(spec["s0"]["dir"]), inputs)
    chk = f7_iii(jobspec_path, prosper_out, arms=(arm,), methods=(method,))
    require(chk["pass"], f"F7 (iii) not passed for {arm}/{method}: {chk['n_missing']} missing, {len(chk['violations'])} violations")
    ids = pop["slide_id"].tolist()
    col = {s: j for j, s in enumerate(ids)}
    patients = pop["patient_id"].to_numpy(dtype=object)
    sub = el[(el["arm"] == arm) & (el["method"] == method)]
    qrows = [(q, e) for e in sub.itertuples(index=False) for q in e.query_ids.split()]
    qset = [q for q, _ in qrows]
    require(len(set(qset)) == len(qset), "a query appears in more than one element")
    q_idx = np.array(sorted(col[q] for q in qset), dtype=np.int64)
    mat = np.full((len(q_idx), len(ids)), np.nan)
    rows_out, per_version, libs = [], {}, {}
    for q, e in qrows:
        if e.library_ids_file not in libs:
            lids = read_ids(Path(e.library_ids_file), e.library_ids_sha256)
            require(set(lids) <= set(ids), "library slide outside the population")
            libs[e.library_ids_file] = (set(lids), ev._ids_sha(lids), len(lids))
        lset, lsha, lsize = libs[e.library_ids_file]
        fp = Path(e.rows_dir) / f"similarity_{q}.pkl"
        raw = fp.read_bytes()
        row = pickle.loads(raw)
        oids = [str(o) for o, _ in row]
        require(len(set(oids)) == len(oids) and set(oids) == lset, f"{fp}: reference ids are not exactly its library")
        vals = {o: float(v) for o, v in row}
        require(all(np.isfinite(v) for v in vals.values()), f"{fp}: nonfinite distance")
        require(vals.get(q) == SENTINEL, f"{fp}: self distance is not the 1e6 sentinel")
        n_failed = sum(1 for o, v in vals.items() if v == SENTINEL and o != q)
        r = int(np.searchsorted(q_idx, col[q]))
        for o, v in vals.items():
            mat[r, col[o]] = v
        i = col[q]
        rows_out.append((q, i, patients[i], pop["diagnostic_group"].iat[i], e.head_slot, e.library_version,
                         e.query_head_id, e.library_head_id, lsize, lsha, n_failed, e.element_id, str(fp),
                         hashlib.sha256(raw).hexdigest(), e.receipt_file))
        pv = per_version.setdefault(e.library_version, {"rows": 0, "library_size": lsize, "library_ids_sha256": lsha})
        pv["rows"] += 1
    rows_out.sort(key=lambda t: t[1])
    q_ids = [ids[i] for i in q_idx]
    masked = mat.copy()
    same = patients[q_idx][:, None] == patients[None, :]
    same[np.arange(len(q_idx)), q_idx] = False
    used = ~np.isnan(mat)
    n_masked = int((same & used).sum())
    masked[same & used] = SENTINEL
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, arr in ((matrix_name(enc, budget, arm, method, False), mat),
                      (matrix_name(enc, budget, arm, method, True), masked)):
        data = pickle.dumps(pd.DataFrame(arr, index=q_ids, columns=ids), protocol=pickle.HIGHEST_PROTOCOL)
        with open(out / name, "xb") as fh:
            fh.write(data)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    base = matrix_name(enc, budget, arm, method, False)[:-4]
    hdr = ("query", "query_index", "patient_id", "diagnostic_group", "head_slot", "library_version", "query_head_id",
           "library_head_id", "library_size", "library_ids_sha256", "n_failed_pairs", "element_id", "row_file", "row_sha256")
    data = ev._csv_bytes(hdr, [r[:14] for r in rows_out])
    with open(out / f"{base}_rows.csv", "xb") as fh:
        fh.write(data)
    files[f"{base}_rows.csv"] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    side = {"unit": spec["unit"], "encoder": enc, "budget_percent": budget, "arm": arm, "method": method,
            "f7_iii": "every row's library_version / head_id equal s0/query_assignment.csv (checked)",
            "rows": [{"query": r[0], "query_index": int(r[1]), "head_slot": r[4], "library_version": r[5],
                      "head_id": r[6], "library_head_id": r[7], "library_size": int(r[8]), "library_ids_sha256": r[9],
                      "element_id": r[11], "element_receipt": r[14], "element_receipt_sha256": sha256_file(r[14])}
                     for r in rows_out]}
    data = json.dumps(side, indent=1).encode()
    with open(out / f"{base}_rows.json", "xb") as fh:
        fh.write(data)
    files[f"{base}_rows.json"] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    rec = {"created": now(), "host": socket.gethostname(), "argv": sys.argv,
           "design": "Option C, population-only libraries, load-time masking", "unit": spec["unit"], "arm": arm,
           "method": method, "method_tag": METHOD_TAG[method], "rows": len(q_ids), "columns": len(ids),
           "cells_outside_row_library": "NaN", "same_patient_cells_masked": n_masked,
           "same_patient_rule": "population patient_id (all slides of the query's patient); self keeps the 1e6 sentinel",
           "failed_pairs_total": int(sum(r[10] for r in rows_out)), "per_library_version": per_version,
           "f7_iii": {k: chk[k] for k in ("pass", "n_missing", "violations")},
           "jobspec": {"path": str(jobspec_path), "sha256": sha256_file(jobspec_path)},
           "outputs": files, "code": code_record(), "wall_seconds": round(time.time() - t0, 1)}
    with open(out / f"{base}_combine_record.json", "xb") as fh:
        fh.write(json.dumps(rec, indent=1, default=str).encode())
    log(json.dumps({"rows": len(q_ids), "masked": n_masked}))
    return rec


# ---------------------------------------------------------------- epilogues / projection / journal
CPU_RE = re.compile(r"CPU time\s*:\s*([0-9.]+)\s*sec")
ELEMENT_RE = re.compile(r"--element\s+(\S+)")


def parse_cpu_seconds(text: str):
    """CPU time from the LAST LSF resource-usage epilogue in an -oo log (None if absent)."""
    m = CPU_RE.findall(text)
    return float(m[-1]) if m else None


def pool_epilogues(pool_dir: Path) -> list:
    """lsfq pool dir (~/.lsfq/pools/NAME): task -> command -> element id; logs/<task>.out epilogue;
    markers/<task>.rc exit code."""
    st = json.loads((Path(pool_dir) / "state.json").read_text())
    out = []
    for tid, t in st["tasks"].items():
        m = ELEMENT_RE.search(t["cmd"])
        if not m:
            continue
        log = Path(pool_dir) / "logs" / f"{tid}.out"
        rcf = Path(pool_dir) / "markers" / f"{tid}.rc"
        cpu = parse_cpu_seconds(log.read_text(errors="replace")) if log.exists() else None
        rc = int(rcf.read_text().strip()) if rcf.exists() else None
        out.append({"pool": st["name"], "task": tid, "element_id": m.group(1), "cpu_seconds": cpu, "exit_code": rc,
                    "attempts": t.get("attempts"), "state": t.get("state"), "log": str(log)})
    return out


def _all_elements(jobspec_root: Path):
    import pandas as pd
    frames = [_jobspec_elements(js)[1] for js in sorted(Path(jobspec_root).glob("*/b*/jobspec.json"))]
    return pd.concat(frames, ignore_index=True) if frames else None


def journal(pool_dirs, jobspec_root: Path, *, gate=CPU_GATE_H) -> dict:
    eps = [e for p in pool_dirs for e in pool_epilogues(Path(p))]
    fin = [e for e in eps if e["cpu_seconds"] is not None]
    total_h = sum(e["cpu_seconds"] for e in fin) / 3600.0
    return {"created": now(), "pools": [str(p) for p in pool_dirs], "elements_seen": len(eps),
            "elements_with_epilogue": len(fin), "consumed_cpu_h": round(total_h, 3),
            "hard_stop_cpu_h": gate, "status": "STOP" if total_h >= gate else "CONTINUE",
            "rule": "R7: hard STOP if consumed CPU-h reaches 16,000 (the lead applies it)",
            "missing_epilogue": [e["element_id"] for e in eps if e["cpu_seconds"] is None][:50]}


def project_cpu(pilot_pools, jobspec_root: Path, *, done_pools=(), population=2604, s4_root: Path | None = None,
                encoders=SIM_ENCODERS, gate=CPU_GATE_H) -> dict:
    """Mean CPU-h per OT row per (encoder, budget, arm) from the pilot epilogues; remaining OT rows of
    every encoder x budget x arm (units without a jobspec at `population` rows) x that mean (missing
    cells: mean over measured encoders for the same budget x arm, else the arm mean); BoB from
    measured BoB epilogues if any, else the specification figure; plus consumed CPU-h and the fixed term.
    R7 tile check: pilot vs all returned queries' retained tiles (> 10 % -> SIZE_ADJUSTMENT_REQUIRED)."""
    el = _all_elements(jobspec_root)
    require(el is not None, f"no jobspec.json under {jobspec_root}")
    by_id = el.set_index("element_id")
    pilot_eps = [e for p in pilot_pools for e in pool_epilogues(Path(p)) if e["cpu_seconds"] is not None]
    done_eps = [e for p in list(pilot_pools) + list(done_pools) for e in pool_epilogues(Path(p))
                if e["cpu_seconds"] is not None and e["exit_code"] == 0]
    done_ids = {e["element_id"] for e in done_eps}
    per = {}
    for e in pilot_eps:
        require(e["element_id"] in by_id.index, f"pilot element {e['element_id']} not in any jobspec")
        r = by_id.loc[e["element_id"]]
        if int(r["pilot"]) != 1:
            continue
        per.setdefault((r["encoder"], int(r["budget_percent"]), r["arm"]), []).append(
            e["cpu_seconds"] / 3600.0 / int(r["n_queries"]))
    means = {f"{k[0]}|b{k[1]}|{k[2]}": {"mean_cpu_h_per_row": float(np.mean(v)), "rows": len(v)}
             for k, v in sorted(per.items())}
    arm_mean = {a: float(np.mean([m for k, v in per.items() if k[2] == a for m in v]))
                for a in ARMS if any(k[2] == a for k in per)}
    require(set(arm_mean) == set(ARMS), f"pilot epilogues missing for arms {sorted(set(ARMS) - set(arm_mean))}")
    remaining_ot, detail = 0.0, []
    for enc in encoders:
        for b in BUDGETS:
            for arm in ARMS:
                own = per.get((enc, b, arm))
                same_cell = [m for k, v in per.items() if k[1] == b and k[2] == arm for m in v]
                if own:
                    m, src = float(np.mean(own)), "own pilot"
                elif same_cell:
                    m, src = float(np.mean(same_cell)), "mean over measured encoders (same budget x arm)"
                else:
                    m, src = arm_mean[arm], "arm mean"
                sub = el[(el["encoder"] == enc) & (el["budget_percent"] == b) & (el["arm"] == arm) & (el["method"] == "ot")]
                if len(sub):
                    rows_left = int(sub.loc[~sub["element_id"].isin(done_ids), "n_queries"].sum())
                    basis = "specified elements not yet finished"
                else:
                    rows_left, basis = population, f"unit not yet specified: {population} rows assumed"
                remaining_ot += rows_left * m
                detail.append({"encoder": enc, "budget_percent": b, "arm": arm, "rows_remaining": rows_left,
                               "cpu_h_per_row": m, "per_row_source": src, "basis": basis, "cpu_h": rows_left * m})
    bob_eps = [e for e in done_eps if by_id.loc[e["element_id"], "method"] == "bob"]
    if bob_eps:
        per_row = float(np.mean([e["cpu_seconds"] / 3600.0 / int(by_id.loc[e["element_id"], "n_queries"]) for e in bob_eps]))
        left = 0
        for enc in encoders:
            for b in BUDGETS:
                for arm in ARMS:
                    sub = el[(el["encoder"] == enc) & (el["budget_percent"] == b) & (el["arm"] == arm) & (el["method"] == "bob")]
                    left += int(sub.loc[~sub["element_id"].isin(done_ids), "n_queries"].sum()) if len(sub) else population
        bob = {"basis": "measured BoB epilogues", "cpu_h_per_row": per_row, "rows_remaining": left, "cpu_h": per_row * left}
    else:
        bob = {"basis": "contract §9 figure (no BoB epilogues yet)", "cpu_h": BOB_FALLBACK_CPU_H_PER_ENCODER * len(encoders)}
    consumed = sum(e["cpu_seconds"] for e in done_eps) / 3600.0
    total = consumed + remaining_ot + bob["cpu_h"] + FIXED_OTHER_CPU_H
    tile = tile_deviation(el, s4_root) if s4_root is not None else {"checked": False}
    status = "STOP" if total > gate else ("SIZE_ADJUSTMENT_REQUIRED" if tile.get("exceeds") else "PASS")
    return {"created": now(), "gate_cpu_h": gate, "projected_total_cpu_h": round(total, 2), "status": status,
            "components": {"consumed_cpu_h": round(consumed, 3), "remaining_ot_cpu_h": round(remaining_ot, 2),
                           "bob": bob, "fixed_other_cpu_h": FIXED_OTHER_CPU_H},
            "pilot_means": means, "pilot_arm_means": arm_mean, "detail": detail, "tile_check": tile,
            "rule": "R7: STOP if the projection exceeds 16,000 CPU-h; SIZE_ADJUSTMENT_REQUIRED (stop and ask the "
                    "lead) if pilot tile counts differ from the returned-query mean by > 10 %"}


def tile_deviation(el, s4_root: Path) -> dict:
    import pandas as pd
    out, exceeds = {}, False
    pilot_ids = set(" ".join(el.loc[el["pilot"] == 1, "query_ids"]).split())
    for p in sorted(Path(s4_root).glob("index/*/b*/_summary_queries.csv")):
        s = pd.read_csv(p, dtype={"slide_id": str}, keep_default_na=False)
        s = s[s["written"] == 1]
        mp, ma = s.loc[s["slide_id"].isin(pilot_ids), "retained_tiles"].mean(), s["retained_tiles"].mean()
        dev = float(abs(mp - ma) / ma) if ma else None
        out[f"{p.parents[1].name}/{p.parent.name}"] = {"pilot_mean": float(mp), "returned_mean": float(ma),
                                                       "relative_deviation": dev}
        exceeds |= dev is not None and dev > TILE_DEVIATION_LIMIT
    return {"checked": True, "units": out, "limit": TILE_DEVIATION_LIMIT, "exceeds": exceeds}


# ---------------------------------------------------------------- controls (F3, F3b)
def _control_element(eid, method, width, qids, qbank, qidx, lids_file, lbank, lidx, out: Path, offsets=""):
    def sha(p):
        return sha256_file(p) if p else ""
    return {"element_id": eid, "encoder": "uni2" if width == 1536 else "virchow2", "budget_percent": 0,
            "arm": "control", "method": method, "head_slot": "", "library_version": "", "query_head_id": "",
            "library_head_id": "", "line": 1, "n_queries": len(qids), "query_ids": " ".join(qids), "pilot": 0,
            "width": width, "query_bank_dir": str(qbank), "query_index_file": str(qidx or ""),
            "query_index_sha256": sha(qidx), "library_bank_dir": str(lbank), "library_ids_file": str(lids_file),
            "library_ids_sha256": sha256_file(lids_file), "library_index_file": str(lidx or ""),
            "library_index_sha256": sha(lidx), "bank_offsets": offsets, "rows_dir": str(out / "rows" / eid),
            "provenance_file": str(out / "provenance" / f"{eid}.json"), "receipt_file": str(out / "receipts" / f"{eid}.json")}


def routeb_positions(prosper_out, fold, budget=F3B_BUDGET, min_tiles=F3B_MIN_TILES):
    """slide -> positions of the accepted route-b fold-<f> mask (written slides only, min_tiles 8)."""
    import pandas as pd
    mdir = Path(prosper_out) / ROUTEB / "mosaic" / "attempt_001"
    off = ff.load_offsets(mdir)
    m = np.load(mdir / f"fold{fold}_budget{budget}_mask.npy", mmap_mode="r")
    out = {}
    for r in off.itertuples(index=False):
        p = np.flatnonzero(np.asarray(m[int(r.tile_start):int(r.tile_stop)], bool)).astype("<i4")
        if len(p) >= min_tiles:
            out[r.slide_id] = p
    return out


def write_controls(out: Path, s0_dir: Path, pilot_csv: Path, prosper_out: str, *, inputs=None, log=print) -> dict:
    """Controls as load-time elements: F3 (a) UNI2 legacy row ([decision record]) vs the all-tissue and the legacy
    filtered library, (b) one Virchow2-fp32 OT row vs the bound run_003 matrix, (c) one UNI2 BoB row vs
    the bound UNI2 Yottixel matrix (bitwise); F3b: F3B_N route-b UNI2 5 % queries, query-only and
    both-sided, with query tile sets and both-sided libraries given by index lists derived from the
    accepted route-b masks (min_tiles 8) and applied at load time to the all-tissue bank."""
    import pandas as pd
    inputs = inputs or S0.default_inputs(S0.HAORAN, prosper_out)
    P = Path(prosper_out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "libraries").mkdir()
    (out / "index").mkdir()
    pilot = pd.read_csv(pilot_csv, dtype={"slide_id": str})
    uni2, v2 = Path(inputs["lib_uni2_dir"]), Path(inputs["lib_virchow2_fp32_dir"])
    elements, ctl = [], []

    def ids_of(d):
        return lib_order(p.name[:-4] for p in Path(d).glob("*.pkl"))

    all_uni2 = out / "libraries" / "uni2_all_tissue.txt"
    write_ids(all_uni2, ids_of(uni2))
    legacy_f = out / "libraries" / "legacy_filtered.txt"
    write_ids(legacy_f, ids_of(P / LEGACY["library_filtered"]))
    all_v2 = out / "libraries" / "virchow2_all_tissue.txt"
    write_ids(all_v2, ids_of(v2))
    lq = LEGACY["query"]
    for arm, lfile, lbank, ref in (("all_tissue", all_uni2, uni2, P / LEGACY["ref_all_tissue"]),
                                   ("filtered", legacy_f, P / LEGACY["library_filtered"], P / LEGACY["ref_filtered"])):
        eid = f"f3_uni2_legacy_{arm}"
        elements.append(_control_element(eid, "ot", 1536, [lq], P / LEGACY["query_dir"], None, lfile, lbank, None, out))
        ctl.append({"control": eid, "kind": "F3 legacy (ADR 0020)", "method": "ot", "query": lq, "element": eid,
                    "reference_row": str(ref), "reference_sha256": sha256_file(ref),
                    "rule": "max rel < 1e-6 and exact top-10; reference ids equal"})
    q0 = pilot["slide_id"].iat[0]
    for eid, method, width, bank, lfile, mrole in (
            ("f3_virchow2_fp32_ot", "ot", 1280, v2, all_v2, "baseline:virchow2_fp32/ot_cosine"),
            ("f3_uni2_bob_bitwise", "bob", 1536, uni2, all_uni2, "baseline:uni2/yottixel")):
        elements.append(_control_element(eid, method, width, [q0], bank, None, lfile, bank, None, out))
        ctl.append({"control": eid, "kind": "F3 accepted matrix row", "method": method, "query": q0, "element": eid,
                    "reference_matrix": inputs[mrole], "reference_role": mrole,
                    "rule": "bitwise on reference cells" if method == "bob" else "max rel < 1e-6 and exact eligible top-10"})
    # F3b
    rb = P / ROUTEB
    rows_csv = {arm: pd.read_csv(rb / "similarity_matrix_qf" /
                                 f"combined_similarity_matrix_r2_1_hopt_filter_qf_budget{F3B_BUDGET}_{tok}_ot_cosine_rows.csv",
                                 dtype={"query": str})
                for arm, tok in (("both_sided", "both"), ("query_only", "query"))}
    both_idx = rows_csv["both_sided"].set_index("query")
    common = sorted(set(both_idx.index) & set(rows_csv["query_only"]["query"]),
                    key=lambda s: int(both_idx.loc[s, "query_index"]))
    rng = np.random.Generator(np.random.PCG64(F3B_SEED))
    pick = [common[i] for i in np.sort(rng.choice(len(common), size=F3B_N, replace=False))]
    fold_of = both_idx["query_fold"].astype(int).to_dict()
    pos = {f: routeb_positions(prosper_out, f) for f in sorted({fold_of[q] for q in pick})}
    qnpz = out / "index" / "f3b_queries.npz"
    S4._save_npz(qnpz, {q: pos[fold_of[q]][q] for q in pick})
    pop = S4.load_population(s0_dir, inputs)
    popf = out / "libraries" / "population.txt"
    write_ids(popf, lib_order(pop["slide_id"]))
    f3b_q = []
    for f, pf in pos.items():
        lnpz = out / "index" / f"f3b_library_fold{f}.npz"
        S4._save_npz(lnpz, pf)
        lf = out / "libraries" / f"f3b_routeb_bank_b{F3B_BUDGET}_fold{f}.txt"
        write_ids(lf, lib_order(pf))
        for q in [q for q in pick if fold_of[q] == f]:
            for arm, lfile, lidx in (("both_sided", lf, lnpz), ("query_only", popf, None)):
                eid = f"f3b_{arm}_{q}"
                elements.append(_control_element(eid, "ot", 1536, [q], uni2, qnpz, lfile, uni2, lidx, out))
                ctl.append({"control": eid, "kind": "F3b filtered-path regression (R8)", "method": "ot", "query": q,
                            "element": eid, "fold": f,
                            "reference_matrix": str(rb / "similarity_matrix_qf" /
                                                    f"combined_similarity_matrix_r2_1_hopt_filter_qf_budget{F3B_BUDGET}_"
                                                    f"{'both' if arm == 'both_sided' else 'query'}_ot_cosine.pkl"),
                            "rule": "max rel < 1e-6 and exact eligible top-10 (same-patient excluded); library ids equal"})
            f3b_q.append({"query": q, "fold": f})
    el = pd.DataFrame(elements, columns=ELEMENT_COLUMNS)
    write_csv(el, out / "elements.csv.gz")
    jobspec_path = out / "jobspec.json"
    s5 = stage_dir(prosper_out)
    cmds = {"ot": [], "bob": []}
    for e in elements:
        cmds[e["method"]].append(element_command(e, jobspec_path, s5, prosper_out).replace(
            f"{s5}/host_records/{e['encoder']}_b0_control_{e['method']}", f"{out}/host_records"))
    prep = (f"export {THREAD_ENV} PROSPER_OUT={prosper_out}; cd {PROSPER_SEARCH} && {PY} "
            f"{SCRIPTS / 'r2_1_tf2_s5_jobspec.py'} f3b-prepare --controls {out}")
    (out / "cmds_prepare.txt").write_text(prep + "\n")
    (out / "cmds_ot.txt").write_text("".join(c + "\n" for c in cmds["ot"]))
    (out / "cmds_bob.txt").write_text("".join(c + "\n" for c in cmds["bob"]))
    spec = {"schema_version": 2, "stage": "S5-controls", "created": now(), "controls": ctl,
            "elements_csv": {"path": str(out / "elements.csv.gz"), "sha256": sha256_file(out / "elements.csv.gz"),
                             "rows": len(el)},
            "s0": {"dir": str(s0_dir), "query_assignment_sha256": sha256_file(Path(s0_dir) / "query_assignment.csv")},
            "f3b": {"budget_percent": F3B_BUDGET, "n": F3B_N, "seed": F3B_SEED, "min_tiles": F3B_MIN_TILES,
                    "queries": f3b_q,
                    "selection": "PCG64 seed 123 draw from queries with both accepted route-b 5 % OT rows",
                    "path": "load-time masking (index lists from the accepted route-b masks) on the all-tissue UNI2 bank",
                    "prepare": "re-derive every file of the five route-b budget-5 banks from the load-time subset "
                               "and compare sha256 with their _filter_summary.csv (byte identity)"},
            "cmds": {"prepare": str(out / "cmds_prepare.txt"), "ot": str(out / "cmds_ot.txt"), "bob": str(out / "cmds_bob.txt")},
            "cmds_sha256": {k: sha256_file(out / f"cmds_{k}.txt") for k in ("prepare", "ot", "bob")},
            "shapes": {"prepare": {"cpu": 1, "mem_gib": 16, "walltime_h": 2, "lsfq": "-n 1 -m 16 -W 2h"},
                       "ot": SHAPES["ot"], "bob": SHAPES["bob"]},
            "order": "prepare -> ot + bob -> check-controls; all before any pilot/production element",
            "code": code_record()}
    write_json(spec, jobspec_path)
    log(json.dumps({"controls": len(ctl), "ot_lines": len(cmds["ot"]), "bob_lines": len(cmds["bob"])}))
    return spec


def f3b_prepare(controls: Path, prosper_out: str, *, log=print) -> dict:
    """Inside the prepare job: for every slide of the accepted route-b mosaic and each of the five
    fold heads (5 %), pickle the load-time subset with ff.PICKLE_PROTOCOL and compare its sha256 with
    the accepted route-b bank file (written slides) - byte identity of the load-time path."""
    import pandas as pd
    P = Path(prosper_out)
    mdir = P / ROUTEB / "mosaic" / "attempt_001"
    off = ff.load_offsets(mdir)
    coordinates = np.load(mdir / "coordinates.npy", mmap_mode="r")
    feats_dir = Path(json.loads((P / ROUTEB / "features_qf" / f"budget{F3B_BUDGET}" / "fold0" / "_filter_record.json")
                                .read_text())["inputs"]["features_dir"])
    pos = {f: routeb_positions(prosper_out, f) for f in range(5)}
    summ = {f: pd.read_csv(P / ROUTEB / "features_qf" / f"budget{F3B_BUDGET}" / f"fold{f}" / "_filter_summary.csv",
                           dtype={"slide_id": str, "sha256": str}).set_index("slide_id") for f in range(5)}
    mism, checked = [], 0
    for r in off.itertuples(index=False):
        feats, coords, data = ff.load_feature_pickle(feats_dir / f"{r.slide_id}.pkl")
        ff.verify_slide(r, feats, coords, data, coordinates, width=1536)
        for f in range(5):
            s = summ[f].loc[r.slide_id]
            if r.slide_id in pos[f]:
                sub = S4.load_time_subset(feats_dir / f"{r.slide_id}.pkl", pos[f][r.slide_id])
                payload = pickle.dumps(sub, protocol=ff.PICKLE_PROTOCOL)
                checked += 1
                if int(s["written"]) != 1 or hashlib.sha256(payload).hexdigest() != s["sha256"]:
                    mism.append({"slide_id": r.slide_id, "fold": f})
            elif int(s["written"]) != 0:
                mism.append({"slide_id": r.slide_id, "fold": f, "kind": "min_tiles"})
    rec = {"created": now(), "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"),
           "slides": int(len(off)), "subsets_checked": checked, "byte_mismatches": mism,
           "pass": not mism, "code": code_record()}
    ff.write_once(Path(controls) / "f3b_prepare_record.json", json.dumps(rec, indent=1).encode())
    log(json.dumps({k: rec[k] for k in ("subsets_checked", "pass")}))
    require(rec["pass"], "F3b prepare: the load-time subsets do not reproduce the accepted route-b bank bytes")
    return rec


def _rank_topk(cols, vals, order_index, k=TOPK):
    idx = np.lexsort((np.asarray(order_index), np.asarray(vals)))
    return [cols[i] for i in idx[:k]]


def compare_row(recomputed: dict, reference: dict, order: list, *, exclude=(), bitwise=False) -> dict:
    """Compare on the reference's eligible cells (finite, < 1e5, not excluded)."""
    cols = [c for c in order if c in reference and c not in set(exclude)
            and np.isfinite(reference[c]) and reference[c] < 1e5]
    missing = [c for c in cols if c not in recomputed]
    if missing:
        return {"pass": False, "missing": missing[:10]}
    a = np.array([recomputed[c] for c in cols], dtype=np.float64)
    b = np.array([reference[c] for c in cols], dtype=np.float64)
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
    pos = {c: i for i, c in enumerate(order)}
    ta = _rank_topk(cols, a, [pos[c] for c in cols])
    tb = _rank_topk(cols, b, [pos[c] for c in cols])
    res = {"cells": len(cols), "max_abs": float(np.max(np.abs(a - b))) if len(cols) else 0.0,
           "max_rel": float(rel.max()) if len(cols) else 0.0, "top10_equal": ta == tb,
           "top10_recomputed": ta, "top10_reference": tb}
    if bitwise:
        res["bitwise_equal"] = bool(np.array_equal(a, b))
        res["pass"] = res["bitwise_equal"]
    else:
        res["pass"] = bool(res["max_rel"] < REL_LIMIT and res["top10_equal"])
    return res


def check_controls(controls: Path, prosper_out: str, *, inputs=None, log=print) -> dict:
    import pandas as pd
    inputs = inputs or S0.default_inputs(S0.HAORAN, prosper_out)
    spec = json.loads((Path(controls) / "jobspec.json").read_text())
    _, el = _jobspec_elements(Path(controls) / "jobspec.json")
    rows_dir = el.set_index("element_id")["rows_dir"].to_dict()
    pop = pd.read_csv(inputs["retrieval_population"], dtype={"slide_id": str, "patient_id": str})
    patient = dict(zip(pop["slide_id"], pop["patient_id"]))
    results = []
    for c in spec["controls"]:
        q = c["query"]
        row = dict((str(o), float(v)) for o, v in pickle.loads((Path(rows_dir[c["element"]]) / f"similarity_{q}.pkl").read_bytes()))
        if c["kind"].startswith("F3 legacy"):
            ref = dict((str(o), float(v)) for o, v in pickle.loads(Path(c["reference_row"]).read_bytes()))
            r = compare_row(row, ref, list(ref), exclude=[q])
            r["reference_ids_equal"] = set(row) == set(ref)
            r["pass"] = r["pass"] and r["reference_ids_equal"]
        elif c["kind"].startswith("F3 accepted"):
            m = pickle.loads(Path(c["reference_matrix"]).read_bytes())
            ref = {str(k): float(v) for k, v in m.loc[q].items()}
            r = compare_row(row, ref, [str(x) for x in m.columns], exclude=[q], bitwise=c["method"] == "bob")
        else:
            m = pickle.loads(Path(c["reference_matrix"]).read_bytes())
            ref = {str(k): float(v) for k, v in m.loc[q].items()}
            excl = [s for s in m.columns if patient.get(str(s)) == patient.get(q)]
            r = compare_row(row, ref, [str(x) for x in m.columns], exclude=excl)
            r["library_ids_equal"] = set(row) == {k for k, v in ref.items() if np.isfinite(v)}
            r["pass"] = r["pass"] and r["library_ids_equal"]
        results.append({"control": c["control"], **r})
    prep = Path(controls) / "f3b_prepare_record.json"
    f3b_prep = json.loads(prep.read_text())["pass"] if prep.exists() else False
    res = {"created": now(), "results": results, "f3b_prepare_pass": f3b_prep,
           "pass": all(r["pass"] for r in results) and f3b_prep, "rules": {"rel_limit": REL_LIMIT, "top_k": TOPK}}
    log(json.dumps({"pass": res["pass"], "n": len(results)}))
    return res


# ---------------------------------------------------------------- CLI
def _out(path, prosper_out):
    return S0.check_out_dir(Path(path), prosper_out, STAGE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot"); p.add_argument("--s0", required=True); p.add_argument("--out", required=True)
    s = sub.add_parser("specs")
    s.add_argument("--encoder", required=True, choices=SIM_ENCODERS)
    s.add_argument("--budget", required=True, type=int, choices=BUDGETS)
    s.add_argument("--s3-unit", required=True); s.add_argument("--s4-unit", required=True)
    s.add_argument("--s0", required=True); s.add_argument("--pilot", required=True); s.add_argument("--out", required=True)
    r = sub.add_parser("element"); r.add_argument("--jobspec", required=True); r.add_argument("--element", required=True)
    f = sub.add_parser("f7-iii"); f.add_argument("--jobspec", required=True); f.add_argument("--out", required=True)
    c = sub.add_parser("combine")
    c.add_argument("--jobspec", required=True); c.add_argument("--arm", required=True, choices=ARMS)
    c.add_argument("--method", required=True, choices=METHODS); c.add_argument("--out", required=True)
    pr = sub.add_parser("project")
    pr.add_argument("--jobspec-root", required=True); pr.add_argument("--pilot-pool", nargs="+", required=True)
    pr.add_argument("--done-pool", nargs="*", default=[]); pr.add_argument("--s4-root", default=None)
    pr.add_argument("--out", required=True)
    j = sub.add_parser("journal")
    j.add_argument("--jobspec-root", required=True); j.add_argument("--pool", nargs="+", required=True)
    j.add_argument("--out", required=True)
    k = sub.add_parser("controls")
    k.add_argument("--s0", required=True); k.add_argument("--pilot", required=True); k.add_argument("--out", required=True)
    kp = sub.add_parser("f3b-prepare"); kp.add_argument("--controls", required=True)
    kc = sub.add_parser("check-controls"); kc.add_argument("--controls", required=True); kc.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prosper_out = os.environ.get("PROSPER_OUT")
    if not prosper_out:
        sys.exit("PROSPER_OUT must be exported")
    if a.cmd == "pilot":
        run_pilot(Path(a.s0), _out(a.out, prosper_out), prosper_out)
    elif a.cmd == "specs":
        build_specs(a.encoder, a.budget, Path(a.s3_unit), Path(a.s4_unit), Path(a.s0), Path(a.pilot),
                    _out(a.out, prosper_out), prosper_out)
    elif a.cmd == "element":
        run_element(Path(a.jobspec), a.element, prosper_out)
    elif a.cmd == "f7-iii":
        o = _out(a.out, prosper_out); o.mkdir(parents=True, exist_ok=True)
        res = f7_iii(Path(a.jobspec), prosper_out)
        write_json(res, o / ("f7_checkpoint_iii.json" if res["pass"] else "f7_checkpoint_iii_FAILED.json"))
        return 0 if res["pass"] else 1
    elif a.cmd == "combine":
        combine(Path(a.jobspec), a.arm, a.method, _out(a.out, prosper_out), prosper_out)
    elif a.cmd == "project":
        o = _out(a.out, prosper_out); o.mkdir(parents=True, exist_ok=True)
        res = project_cpu(a.pilot_pool, Path(a.jobspec_root), done_pools=a.done_pool,
                          s4_root=Path(a.s4_root) if a.s4_root else None)
        write_json(res, o / "projection.json")
        print(json.dumps({"projected_total_cpu_h": res["projected_total_cpu_h"], "status": res["status"]}))
        return 0 if res["status"] == "PASS" else 2
    elif a.cmd == "journal":
        o = _out(a.out, prosper_out); o.mkdir(parents=True, exist_ok=True)
        res = journal(a.pool, Path(a.jobspec_root))
        write_json(res, o / "consumed.json")
        print(json.dumps({"consumed_cpu_h": res["consumed_cpu_h"], "status": res["status"]}))
        return 0 if res["status"] == "CONTINUE" else 2
    elif a.cmd == "controls":
        write_controls(_out(a.out, prosper_out), Path(a.s0), Path(a.pilot), prosper_out)
    elif a.cmd == "f3b-prepare":
        f3b_prepare(Path(a.controls), prosper_out)
    else:
        o = _out(a.out, prosper_out); o.mkdir(parents=True, exist_ok=True)
        res = check_controls(Path(a.controls), prosper_out)
        write_json(res, o / ("check_controls.json" if res["pass"] else "check_controls_FAILED.json"))
        return 0 if res["pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
