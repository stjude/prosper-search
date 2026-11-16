#!/usr/bin/env python3
"""R2.1-TF2 TITAN filtered-arm matrices (operations release 006, step 2).

Mirrors `r2_1_tf2_s5_jobspec.py combine` (layout: rows = returned queries, columns = the 2,604
population in reference order, NaN outside the row's library version, self = 1e6 sentinel, a
`_no_same_patient` twin with the population-patient_id same-patient rule, `_rows.csv`, `_rows.json`
sidecar with library_version / head_id / receipts, `_combine_record.json`) and the accepted R2.6 L1
rule of `scripts/r2_6_titan_arm2_l1_matrix.py` / `r2_6_titan_arm1_regen.py` (float64 slide
embeddings, scipy cityblock).

  eligibility  s4/eligibility/titan/b<b>.csv.gz: the S4 `eligibility` function on the conch_v15 S3 unit
               with min_tiles = 2 (design decision T1, TITAN only) and the `encoder` column relabelled `titan`
               (LEAD_DECISIONS_001 of release 003, Q3: encoder id `titan`, head id conch_v15/b<b>/<slot>);
               differences from the min_tiles = 1 conch_v15 table are listed in the record
  build        one budget x arm:
                 query_only  masked query embedding (s4/titan/b<b>/queries) vs the accepted unfiltered
                             library embeddings (2,900 published + 21 regenerated R2.6; gate: they
                             rebuild the S0-bound baseline:titan/titan matrix bitwise)
                 both_sided  masked query embedding vs the embeddings of its own library version
                             (s4/titan/b<b>/v_<slot>), eligible ids from the eligibility table
               Every TITAN set used must carry verify_record.json PASS; F7 (iii): every row's stage
               manifests name the head id and library version S0 assigns to the query.
Writes only under $PROSPER_OUT/r2_1_filter_tf2_20261016/{s4/eligibility/titan, s5/combined_titan_001}.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import importlib.util
import json
import pickle
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


OPS = _load("r2_1_tf2_titan_ops", SCRIPTS / "r2_1_tf2_titan_ops.py")
S0, S3, S4 = OPS.S0, OPS.S3, OPS.S4
REGEN = OPS.REGEN
from prosper.evaluation import r2_1_filter_eval_cli as ev  # noqa: E402

ENCODER_ID = "titan"
METHOD_TAG = "titan"
ARMS = ("query_only", "both_sided")
SENTINEL = ev.SENTINEL
OUT_STAGE = "s5"
OUT_NAME = "combined_titan_001"
ROWS_HEADER = ("query", "query_index", "patient_id", "diagnostic_group", "head_slot", "library_version",
               "query_head_id", "library_head_id", "library_size", "library_ids_sha256", "n_failed_pairs",
               "element_id", "row_file", "row_sha256")

sha256_file = S0.sha256_file
write_json = S0.write_json
now = S0.now
require = OPS.require


def code_record():
    return {**OPS.code_record(), "titan_matrix": {"path": str(Path(__file__).resolve()),
                                                  "sha256": sha256_file(Path(__file__).resolve())},
            "filter_eval_cli": {"path": str(Path(ev.__file__).resolve()), "sha256": sha256_file(Path(ev.__file__))}}


def matrix_name(budget, arm, masked):
    return (f"combined_similarity_matrix_tf2_{ENCODER_ID}_b{budget}_{arm}_{METHOD_TAG}"
            f"{'_no_same_patient' if masked else ''}.pkl")


def l1_rows(Q: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Accepted R2.6 L1 rule: float64 embeddings, scipy cityblock."""
    from scipy.spatial.distance import cdist
    return cdist(np.asarray(Q, dtype=np.float64), np.asarray(L, dtype=np.float64), metric="cityblock")


# ---------------------------------------------------------------- eligibility (relabel only)
def titan_eligibility(source: Path, out_dir: Path, budget, s0_dir: Path, prosper_out, *,
                      min_tiles=OPS.TITAN_MIN_TILES, eligibility_fn=None, log=print) -> dict:
    """TITAN table = S4 `eligibility` of the conch_v15 S3 unit recorded in the source table's record,
    recomputed with min_tiles = 2 (design decision T1: < 2 retained tiles = no representation; the slide
    leaves that version and is a non-return as a query), `encoder` relabelled `titan`. Differences from
    the min_tiles = 1 source table are listed in the record."""
    import pandas as pd
    eligibility_fn = eligibility_fn or S4.eligibility
    source, out_dir = Path(source), Path(out_dir)
    src_rec = source.with_name(f"b{budget}_record.json")
    require(src_rec.exists(), f"{src_rec} missing (run s4 eligibility --encoder conch_v15 first)")
    r = json.loads(src_rec.read_text())
    require(r["sha256"] == sha256_file(source) and r["encoder"] == OPS.ENCODER and r["budget_percent"] == budget,
            f"{source}: record mismatch")
    path, rec_path = out_dir / f"b{budget}.csv.gz", out_dir / f"b{budget}_record.json"
    require(not path.exists() and not rec_path.exists(), f"refusing existing {path}")
    src = pd.read_csv(source, dtype=str, keep_default_na=False)
    require(list(src.columns) == S4.ELIGIBILITY_COLUMNS, "unexpected eligibility columns")
    require(set(src["encoder"]) == {OPS.ENCODER}, "source encoder is not conch_v15")
    df = eligibility_fn(Path(r["s3_unit"]), Path(s0_dir), OPS.ENCODER, budget, prosper_out=str(prosper_out),
                        min_tiles=min_tiles)
    require(list(df.columns) == S4.ELIGIBILITY_COLUMNS, "unexpected eligibility columns")
    require(df["head_id"].astype(str).str.startswith(f"{OPS.ENCODER}/b{budget}/").all(),
            "head ids are not conch_v15/b<b>/...")
    out = df.copy()
    out["encoder"] = ENCODER_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    S0.write_csv(out, path)
    back = pd.read_csv(path, dtype=str, keep_default_na=False)
    require(set(back["encoder"]) == {ENCODER_ID} and len(back) == len(src)
            and back[["query", "arm"]].equals(src[["query", "arm"]]), "row layout differs from the source table")
    changed = []
    for c in ("query_returned", "n_eligible", "eligible_ids_sha256"):
        d = back[c] != src[c]
        for i in d[d].index:
            changed.append({"query": back.at[i, "query"], "arm": back.at[i, "arm"],
                            "library_version": back.at[i, "library_version"], "column": c,
                            "min_tiles_1": src.at[i, c], "min_tiles_2": back.at[i, c]})
    removed = {}
    for v in sorted(set(back["library_version"])):
        a_ids = set(";".join(src[src.library_version == v].eligible_reference_ids.unique()).split(";")) - {""}
        b_ids = set(";".join(back[back.library_version == v].eligible_reference_ids.unique()).split(";")) - {""}
        require(b_ids <= a_ids, f"{v}: min_tiles 2 added ids")
        if a_ids - b_ids:
            removed[v] = sorted(a_ids - b_ids)
    same = [c for c in S4.ELIGIBILITY_COLUMNS if c not in ("encoder",) and back[c].equals(src[c])]
    rec = {"created": now(), "encoder": ENCODER_ID, "budget_percent": budget, "rows": len(out),
           "sha256": sha256_file(path), "columns": S4.ELIGIBILITY_COLUMNS, "min_tiles": min_tiles,
           "rule": "lead decision T1 (contract_candidate_006/CONTRACT_AMENDMENT_POOL_40X.md, 2026-10-17): TITAN arm "
                   "only, < 2 retained tiles = no representation (leaves the version; non-return as a query); "
                   "OT/BoB tables keep min_tiles = 1",
           "s3_unit": r["s3_unit"], "s0": str(s0_dir),
           "derived_from": {"path": str(source), "sha256": r["sha256"], "record": str(src_rec),
                            "record_sha256": sha256_file(src_rec), "min_tiles": r.get("min_tiles")},
           "transformation": "S4.eligibility(conch_v15, min_tiles=2); encoder column -> titan; head_id keeps the mask "
                             "head conch_v15/b<b>/<slot>",
           "removed_ids_by_version_vs_min_tiles_1": removed, "changed_cells_vs_min_tiles_1": changed,
           "columns_identical_to_source": same, "code": code_record()}
    write_json(rec, rec_path)
    log(json.dumps({"titan_eligibility": str(path), "rows": len(out), "removed": removed}))
    return rec


# ---------------------------------------------------------------- embeddings of one verified set
def load_set(stage_dir: Path) -> dict:
    stage_dir = Path(stage_dir)
    v = json.loads((stage_dir / "verify_record.json").read_text())
    require(v["status"] == "PASS", f"{stage_dir}: verify_record is not PASS")
    man = json.loads((stage_dir / "stage_manifest.json").read_text())
    require(sha256_file(stage_dir / "stage_manifest.json") == v["stage_manifest_sha256"],
            f"{stage_dir}: stage manifest changed since verify")
    emb = Path(v["embeddings_dir"])
    vec, sha = {}, {}
    for s, h in v["embedding_sha256"].items():
        f = emb / f"{s}.h5"
        require(sha256_file(f) == h, f"{f}: embedding changed since verify")
        vec[s] = REGEN.read_vec(f)
        sha[s] = h
    slides = {x["slide_id"]: x for x in man["slides"]}
    return {"dir": str(stage_dir), "manifest": man, "vec": vec, "sha": sha, "slides": slides, "emb_dir": emb,
            "receipt": {"stage_manifest": str(stage_dir / "stage_manifest.json"),
                        "stage_manifest_sha256": v["stage_manifest_sha256"],
                        "verify_record": str(stage_dir / "verify_record.json"),
                        "verify_record_sha256": sha256_file(stage_dir / "verify_record.json")}}


# ---------------------------------------------------------------- build
def build(budget, arm, s0_dir: Path, titan_root: Path, eligibility: Path, out: Path, prosper_out, *,
          accepted=None, log=print) -> dict:
    import pandas as pd
    from prosper.evaluation import r2_1_tf2_eval_cli as T
    t0 = time.time()
    require(arm in ARMS, f"arm {arm}")
    out = Path(out)
    require(not out.exists() or not any(out.iterdir()), f"refusing non-empty {out}")
    pop = OPS.population(s0_dir, prosper_out)
    qa_raw = OPS.query_assignment(s0_dir)
    qa = T.load_query_assignment(Path(s0_dir) / "query_assignment.csv", pop)
    E = T.read_eligibility(Path(eligibility), pop, ENCODER_ID, str(budget), qa)
    E = E["query_only" if arm == "query_only" else "both_filtered"]
    ids = pop["slide_id"].tolist()
    col = {s: j for j, s in enumerate(ids)}
    patients = pop["patient_id"].to_numpy(dtype=object)
    qa_i = qa_raw.set_index("slide_id")
    qset = load_set(Path(titan_root) / f"b{budget}" / "queries")
    require(qset["manifest"]["set"] == "queries" and qset["manifest"]["budget_percent"] == budget, "query set label")
    returned_ids = [ids[i] for i in np.flatnonzero(E.returned)]
    require(sorted(qset["vec"]) == sorted(returned_ids),
            f"embedded queries ({len(qset['vec'])}) != returned queries of the eligibility table ({len(returned_ids)})")
    violations = []
    for q in returned_ids:
        a, x = qa_i.loc[q], qset["slides"][q]
        hid = S3.head_id(OPS.ENCODER, budget, a["head_slot"])
        if x["head_id"] != hid or x["head_id"] != E.head_id[col[q]]:
            violations.append({"query": q, "s0_head": hid, "stage_head": x["head_id"]})
    lib_sets, lib_vec, lib_receipt, lib_head = {}, {}, {}, {}
    acc_info = None
    if arm == "query_only":
        a_ids, X, acc_info = accepted if accepted is not None else OPS.accepted_embeddings(prosper_out, s0_dir)
        apos = {s: i for i, s in enumerate(a_ids)}
        require(set(ids) <= set(apos), "population slides without an accepted TITAN embedding")
        lib_vec["unfiltered"] = {s: X[apos[s]] for s in ids}
        lib_receipt["unfiltered"] = {"accepted_unfiltered": acc_info}
        lib_head["unfiltered"] = ""
    else:
        for v in sorted(set(E.version[i] for i in np.flatnonzero(E.returned))):
            st = load_set(Path(titan_root) / f"b{budget}" / v)
            require(st["manifest"]["set"] == v and st["manifest"]["budget_percent"] == budget, f"{v}: set label")
            elig_ids = [ids[j] for j in np.flatnonzero(E.versions[v]["mask"])]
            require(sorted(st["vec"]) == sorted(elig_ids),
                    f"{v}: embedded slides ({len(st['vec'])}) != eligible ids ({len(elig_ids)})")
            heads = {x["head_id"] for x in st["slides"].values()}
            require(heads == {S3.head_id(OPS.ENCODER, budget, v[2:])}, f"{v}: stage head ids {heads}")
            lib_vec[v], lib_receipt[v], lib_head[v] = st["vec"], st["receipt"], heads.pop()
    returned_idx = np.array(sorted(col[q] for q in returned_ids), dtype=np.int64)
    mat = np.full((len(returned_idx), len(ids)), np.nan)
    rows_out, per_version, equal_own = [], {}, 0
    for r, i in enumerate(returned_idx):
        q = ids[i]
        v = E.version[i]
        if arm == "both_sided" and v != qa_i.loc[q, "library_version_both_sided"]:
            violations.append({"query": q, "s0_version": qa_i.loc[q, "library_version_both_sided"], "row_version": v})
        if arm == "query_only" and v != qa_i.loc[q, "library_version_query_only"]:
            violations.append({"query": q, "s0_version": qa_i.loc[q, "library_version_query_only"], "row_version": v})
        lids = [ids[j] for j in np.flatnonzero(E.versions[v]["mask"])]
        require(q in lids, f"{q}: query not in its own library {v}")
        L = np.stack([lib_vec[v][s] for s in lids])
        d = l1_rows(qset["vec"][q][None, :], L)[0]
        require(np.isfinite(d).all(), f"{q}: nonfinite distance")
        jj = np.array([col[s] for s in lids])
        mat[r, jj] = d
        mat[r, i] = SENTINEL
        if arm == "both_sided" and np.array_equal(qset["vec"][q], lib_vec[v][q]):
            equal_own += 1
        rows_out.append((q, int(i), patients[i], pop["diagnostic_group"].iat[i], E.head_slot[i], v, E.head_id[i],
                         lib_head[v], len(lids), E.versions[v]["ids_sha256"], 0,
                         f"titan_b{budget}_{arm}_{v}", str(qset["emb_dir"] / f"{q}.h5"), qset["sha"][q]))
        pv = per_version.setdefault(v, {"rows": 0, "library_size": len(lids),
                                        "library_ids_sha256": E.versions[v]["ids_sha256"]})
        pv["rows"] += 1
    require(not violations, f"F7 (iii) violations in TITAN {arm} b{budget}: {violations[:5]} STOP")
    q_ids = [ids[i] for i in returned_idx]
    masked = mat.copy()
    same = patients[returned_idx][:, None] == patients[None, :]
    same[np.arange(len(returned_idx)), returned_idx] = False
    used = ~np.isnan(mat)
    n_masked = int((same & used).sum())
    masked[same & used] = SENTINEL
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, arr in ((matrix_name(budget, arm, False), mat), (matrix_name(budget, arm, True), masked)):
        data = pickle.dumps(pd.DataFrame(arr, index=q_ids, columns=ids), protocol=pickle.HIGHEST_PROTOCOL)
        with open(out / name, "xb") as fh:
            fh.write(data)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    base = matrix_name(budget, arm, False)[:-4]
    data = ev._csv_bytes(ROWS_HEADER, rows_out)
    with open(out / f"{base}_rows.csv", "xb") as fh:
        fh.write(data)
    files[f"{base}_rows.csv"] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    side = {"unit": f"{ENCODER_ID}/b{budget}", "encoder": ENCODER_ID, "mask_encoder": OPS.ENCODER,
            "budget_percent": budget, "arm": arm, "method": METHOD_TAG,
            "f7_iii": "every row's library_version / head_id equal s0/query_assignment.csv and the TITAN stage "
                      "manifests (checked)",
            "rows": [{"query": r[0], "query_index": int(r[1]), "head_slot": r[4], "library_version": r[5],
                      "head_id": r[6], "library_head_id": r[7], "library_size": int(r[8]), "library_ids_sha256": r[9],
                      "element_id": r[11], "query_embedding": r[12], "query_embedding_sha256": r[13],
                      "query_receipt": qset["receipt"]["stage_manifest"],
                      "query_receipt_sha256": qset["receipt"]["stage_manifest_sha256"],
                      "library_receipt": lib_receipt[r[5]].get("stage_manifest", "accepted_unfiltered_r2_6"),
                      "library_receipt_sha256": lib_receipt[r[5]].get(
                          "stage_manifest_sha256", (acc_info or {}).get("baseline_matrix", {}).get("sha256", ""))}
                     for r in rows_out]}
    data = json.dumps(side, indent=1).encode()
    with open(out / f"{base}_rows.json", "xb") as fh:
        fh.write(data)
    files[f"{base}_rows.json"] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    rec = {"created": now(), "host": socket.gethostname(), "argv": sys.argv, "lsf_job": os.environ.get("LSB_JOBID"),
           "design": "Option C, population-only libraries; TITAN slide embeddings of masked CONCH 1024-px tile sets; "
                     "L1 (cityblock) on float64 embeddings (accepted R2.6 rule)",
           "unit": f"{ENCODER_ID}/b{budget}", "arm": arm, "method": METHOD_TAG, "rows": len(q_ids), "columns": len(ids),
           "cells_outside_row_library": "NaN", "same_patient_cells_masked": n_masked,
           "same_patient_rule": "population patient_id (all slides of the query's patient); self keeps the 1e6 sentinel",
           "failed_pairs_total": 0, "per_library_version": per_version,
           "query_embedding_equals_own_version_embedding": (equal_own if arm == "both_sided" else None),
           "f7_iii": {"pass": True, "violations": []},
           "eligibility": {"path": str(eligibility), "sha256": sha256_file(eligibility)},
           "query_set": qset["receipt"], "library_sets": lib_receipt,
           "outputs": files, "code": code_record(), "wall_seconds": round(time.time() - t0, 1)}
    with open(out / f"{base}_combine_record.json", "xb") as fh:
        fh.write(json.dumps(rec, indent=1, default=str).encode())
    log(json.dumps({"unit": rec["unit"], "arm": arm, "rows": len(q_ids), "masked": n_masked,
                    "equal_own": rec["query_embedding_equals_own_version_embedding"]}))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eligibility")
    e.add_argument("--budget", type=int, required=True, choices=S3.BUDGETS)
    e.add_argument("--source", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--s0", required=True)
    b = sub.add_parser("build")
    b.add_argument("--budget", type=int, required=True, choices=S3.BUDGETS)
    b.add_argument("--arm", required=True, choices=ARMS)
    b.add_argument("--s0", required=True)
    b.add_argument("--titan-root", required=True)
    b.add_argument("--eligibility", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prosper_out = os.environ.get("PROSPER_OUT")
    if not prosper_out:
        sys.exit("PROSPER_OUT must be exported")
    root = S0.stage_root(prosper_out)
    if a.cmd == "eligibility":
        out = Path(a.out).absolute()
        require((root / "s4" / "eligibility").resolve() in out.resolve().parents or
                out.resolve() == (root / "s4" / "eligibility" / ENCODER_ID).resolve(), f"--out {out} outside s4/eligibility")
        titan_eligibility(Path(a.source), out, a.budget, Path(a.s0), prosper_out)
        return 0
    out = S0.check_out_dir(Path(a.out), prosper_out, OUT_STAGE)
    build(a.budget, a.arm, Path(a.s0), Path(a.titan_root), Path(a.eligibility), out, prosper_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
