"""N3: cross-encoder perturbation contrast (difference-in-differences) CLI.

Implements ``r2_1_p2_eval_20261006/n3_spec_001/SPEC.md`` (sections 1-8 plus spec
amendments 001, 002, 003 and 004) under [decision record] P3 amendments 001-003.  For every accepted
comparator pipeline c, perturbation arm a, population P, metric and cutoff k::

    D = Delta_HOpt(a) - Delta_c(a)

Each Delta is computed by the accepted, unmodified kernel ``summarize_paired``
(arm minus identity) on the full population vectors with the same inclusion array
object and the same patient-draw table object.  D is composed from the two kernel
summaries (the kernel rejects signed vectors).

Blinding (B5, N3-03): stdout, stderr and FAILED.json carry only check ids,
pipeline ids, row keys, counts, paths and sha256 values -- never a D, Delta or
interval value, nor a tolerance excess.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gzip
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import platform
import re
import shutil
import socket
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

KERNEL_MODULE = "prosper.evaluation.r2_1_hoptimus_main"
KERNEL_SHA256 = "dee7d68a8b9fcee61d2e845412afdaf20455388ae5c2e61844d4b21d73811faf"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _locate_kernel() -> Tuple[Optional[str], Optional[str]]:
    """N3-11: sha256 of the kernel file that will be imported, computed before import."""
    try:
        spec = importlib.util.find_spec(KERNEL_MODULE)
    except (ImportError, ValueError):
        return None, None
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        return None, None
    path = str(Path(spec.origin).resolve())
    return path, sha256_file(Path(path))


KERNEL_PRE_IMPORT_PATH, KERNEL_PRE_IMPORT_SHA256 = _locate_kernel()
kernel = (importlib.import_module(KERNEL_MODULE)
          if KERNEL_PRE_IMPORT_SHA256 == KERNEL_SHA256 else None)

# ---- fixed constants (N3-18: no CLI override) ----------------------------------------------
HOPT_RUN_RECORD_SHA256 = "9064605b80fb86a90d0bc821878aab809a6e61090410a6fdcf43fcceec621cb5"
HOPT_PIPELINE_ID = "hoptimus_1"
# amendment 004 item 1 (N3II-01): non-replay stages run only against the canonical P2ROOT.
CANONICAL_P2ROOT = ("/PROSPER_WORKSPACE/"
                    "prosper_revision/prosper_revision_output/r2_1_p2_eval_20261006")
POPULATION_SHA256 = "5ff0ea3bd2999a4dbebb4c73e679d15870450d38f5bb6fc91be071ee72cbe471"
POPULATION_N_SLIDES = 2778
POPULATION_N_PATIENTS = 1876
ENUM_PIPELINES = ("uni2_sampler", "virchow2_sampler", "conch_sampler", "conch_titan")
SYNTHETIC_PREFIX = "synthetic_"
REQUIRED_SETS = {"stage1_512px": ("uni2_sampler", "virchow2_sampler"),
                 "stage2_conch": ("conch_sampler", "conch_titan"),
                 "single": ENUM_PIPELINES}
STAGES = ("replay", "stage1_512px", "stage2_conch", "single")
DROPPED_DISPOSITIONS = ("dropped_fidelity", "dropped_n4", "nonreturn")
IDENTITY = "identity"
N_CONDITIONS = 13
CUTOFFS = (1, 3, 5, 10)
METRICS = ("hit", "precision")
D_CUTOFFS = {"hit": (1, 3, 5, 10), "precision": (3, 5, 10)}
PRIMARY_K = 5
MACRO = "MACRO"
DRAW_COUNT = 5000
DRAW_SEED = 123
QUANTILES = (0.025, 0.975)
QUANTILE_METHOD = "linear"
TOL = 1e-12
NOMINAL_PRIMARY_PER_COMPARATOR = 24
STRATIFIED_ARMS = ("brightness", "saturation", "contrast", "spatial_shift")
STRATA = ("T1", "T2", "T3")
NULL_POPULATIONS = (("brightness", "brightness_null_excluded"),
                    ("spatial_shift", "shift_null_excluded"))
RUN_RECORD_OUTPUTS = ("query_metrics.csv.gz", "paired_deltas.csv")
COMPARATOR_READ = ("run_record.json", "coverage.csv", "query_metrics.csv.gz",
                   "paired_deltas.csv")
# B5 whitelist (N3-04): files that need not be bound by the blinded set.
WHITELIST_EXACT = ("COMPLETE", "VERIFICATION_FAILED.json", "coverage.csv",
                   "input_manifest.csv", "run_record.json")
WHITELIST_PREFIXES = ("logs/", "lsf_logs/", "fidelity/", "n4/")
WHITELIST_BASENAME_PREFIXES = ("stimulus_manifest",)
WHITELIST_SUFFIXES = (".log",)
WHITELIST_SUFFIX_DIRS = ("logs/", "lsf_logs/")     # amendment 003 item 11 (A4)
SPEC_RELPATH = ("n3_spec_001", "SPEC.md")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
NONREPLAY_STAGES = ("stage1_512px", "stage2_conch", "single")
PATTERN_LABELS = ("both_exclude_same_sign_positive", "both_exclude_same_sign_negative",
                  "mixed", "not_established", "unavailable")
QUERY_HEADER = ("condition", "population_index", "query", "patient_id", "diagnostic_group",
                "k", "status", "returned_depth", "conditional_hit", "conditional_precision",
                "n_candidates", "boundary_tie")
D_HEADER = (
    "comparator", "condition", "population", "group", "metric", "k", "endpoint_role",
    "delta_hopt", "delta_hopt_ci_low", "delta_hopt_ci_high",
    "delta_c", "delta_c_ci_low", "delta_c_ci_high",
    "D", "D_ci_low", "D_ci_high", "D_excludes_zero",
    "identity_abs_hopt", "identity_abs_c", "arm_abs_hopt", "arm_abs_c",
    "identity_abs_hopt_ci_low", "identity_abs_hopt_ci_high",
    "identity_abs_c_ci_low", "identity_abs_c_ci_high",
    "arm_abs_hopt_ci_low", "arm_abs_hopt_ci_high",
    "arm_abs_c_ci_low", "arm_abs_c_ci_high",
    "n_pairs", "n_patients", "point_status", "point_reason", "ci_status", "ci_reason",
    "zero_support_replicates", "zero_width_both", "zero_width_cause",
    "draws_multiplicities_sha256", "draws_patients_sha256")
PATTERN_HEADER = ("comparator", "pipeline_id", "condition",
                  "hit5_D", "hit5_ci_low", "hit5_ci_high", "hit5_excludes_zero",
                  "prec5_D", "prec5_ci_low", "prec5_ci_high", "prec5_excludes_zero",
                  "pattern", "pattern_reason", "reinhard_flag")
INDEX_HEADER = ("row", "comparator", "condition", "population", "group", "metric", "k",
                "endpoint_role")
REPLAY_NUMERIC = ("delta", "ci_low", "ci_high", "estimate_condition", "estimate_identity")
REPLAY_INT = ("n_pairs", "n_patients", "zero_support_replicates")
REPLAY_STATUS = ("point_status", "point_reason", "ci_status", "ci_reason")
# ``entries`` carries attempt-entry paths; ``state`` one token of STAGE1_STATES (amendment 004).
CONTEXT_KEYS = ("pipeline_id", "row_key", "file", "count", "columns", "entries", "state")
STAGE1_STATES = ("complete", "failed", "not_complete", "superseded", "ambiguous", "none")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
LIMITATION_TEXT = ("The acceptance record and COMPLETE chain bind integrity, not access; "
                   "opening comparator blinded files before A7 step 7 remains a "
                   "procedural deviation that N3 cannot detect.")


class N3Error(RuntimeError):
    """Hard stop. Carries a check id and allow-listed context only (N3-03)."""

    def __init__(self, check: str, **context: Any):
        unknown = set(context) - set(CONTEXT_KEYS)
        if unknown:
            raise ValueError("N3Error context keys must be allow-listed")
        super().__init__(check)
        self.check = check
        self.context = context


# ---------------------------------------------------------------------------------------------
# Serialisation (N3-18)
# ---------------------------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def draws_hashes(draws: Any) -> Tuple[str, str]:
    mult = sha256_bytes(np.ascontiguousarray(draws.multiplicities, dtype=np.int64).tobytes())
    pats = sha256_bytes("\n".join(draws.patient_ids).encode())
    return mult, pats


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return str(v)


def csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow([_cell(v) for v in r])
    return buf.getvalue().encode("utf-8")


def json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=1) + "\n").encode("utf-8")


def npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a, dtype=np.float64), allow_pickle=False)
    return buf.getvalue()


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------

def _read_csv(path: Path, check: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """All columns are read as str (csv module; N3-12)."""
    with open(path, "rt", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise N3Error(f"{check}.empty_csv", file=str(path))
        rows = []
        for raw in reader:
            if len(raw) != len(header):
                raise N3Error(f"{check}.ragged_csv", file=str(path))
            rows.append(dict(zip(header, raw)))
    return header, rows


def _read_json(path: Path, check: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise N3Error(f"{check}.unreadable_json", file=str(path))


def _opt_float(s: str) -> Optional[float]:
    return None if s == "" else float(s)


def _opt_str(s: str) -> Optional[str]:
    return None if s == "" else s


def _int(s: str, check: str) -> int:
    try:
        return int(s)
    except (TypeError, ValueError):
        raise N3Error(check)


def _iso(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        _dt.datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------------------------
# Composition (A1/B1/N3-07/N3-09)
# ---------------------------------------------------------------------------------------------

def compose(e_h: Any, e_c: Any, identical_nonzero: bool = False) -> Dict[str, Any]:
    """D = Delta_H - Delta_c for one row from two kernel difference Estimates.

    ``identical_nonzero`` is the N3-09 exact exception (bitwise-identical nonzero
    per-slide changes in both pipelines): D = 0.0 and D_b = 0 on valid replicates.
    """
    if not np.array_equal(e_h.valid, e_c.valid):
        raise N3Error("composition.validity_differs")
    if e_h.n_queries != e_c.n_queries or e_h.n_patients != e_c.n_patients:
        raise N3Error("composition.counts_differ")
    valid = np.asarray(e_h.valid & e_c.valid, dtype=bool)
    if not (np.array_equal(valid, e_h.valid) and np.array_equal(valid, e_c.valid)):
        raise N3Error("composition.validity_differs")
    reps = np.full(valid.shape, np.nan, dtype=np.float64)
    if identical_nonzero:
        reps[valid] = 0.0
    else:
        reps[valid] = e_h.replicates[valid] - e_c.replicates[valid]
    point_reason = e_h.point_reason or e_c.point_reason
    if e_h.point is None or e_c.point is None:
        point = None
    else:
        point = 0.0 if identical_nonzero else float(e_h.point - e_c.point)
    invalid = int(np.count_nonzero(~valid))
    lo = hi = None
    if point is None:
        ci_status, ci_reason = "unavailable", point_reason
    elif bool(valid.all()):
        lo, hi = (float(x) for x in np.quantile(reps, list(QUANTILES), method=QUANTILE_METHOD))
        ci_status, ci_reason = "estimable", None
    else:
        ci_status, ci_reason = "unavailable", "zero_support_replicates"
    excludes = None if ci_status != "estimable" else bool(lo > 0 or hi < 0)
    return dict(point=point, point_status="estimable" if point is not None else "unavailable",
                point_reason=point_reason, ci_low=lo, ci_high=hi, ci_status=ci_status,
                ci_reason=ci_reason, zero_support_replicates=invalid, excludes_zero=excludes,
                replicates=reps, valid=valid, n_pairs=int(e_h.n_queries),
                n_patients=int(e_h.n_patients))


def _zero_interval(e: Any) -> bool:
    return e.ci_low == 0.0 and e.ci_high == 0.0


def zero_width_both(e_h: Any, e_c: Any) -> Optional[bool]:
    if e_h.ci_status != "estimable" or e_c.ci_status != "estimable":
        return None
    return _zero_interval(e_h) and _zero_interval(e_c)


def pattern_of(no_d_reason: Optional[str], hit: Optional[dict],
               prec: Optional[dict]) -> Tuple[str, Optional[str]]:
    """N3-08 evaluation order; sign from the interval, not the point."""
    if no_d_reason is not None:
        return "unavailable", no_d_reason
    un = [lab for lab, d in (("hit5", hit), ("prec5", prec)) if d["ci_status"] == "unavailable"]
    if un:
        return "unavailable", ("both_ci_unavailable" if len(un) == 2 else
                               f"{un[0]}_ci_unavailable")
    if not hit["excludes_zero"] or not prec["excludes_zero"]:
        return "not_established", None
    if hit["ci_low"] > 0 and prec["ci_low"] > 0:
        return "both_exclude_same_sign_positive", None
    if hit["ci_high"] < 0 and prec["ci_high"] < 0:
        return "both_exclude_same_sign_negative", None
    return "mixed", None


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------

class Population:
    def __init__(self, name: str, mask: np.ndarray, draws: Any, matched: str):
        mask = np.array(mask, dtype=bool, copy=True)
        mask.setflags(write=False)
        self.name, self.mask, self.draws, self.matched = name, mask, draws, matched
        self.mult_sha, self.pat_sha = draws_hashes(draws)


class Pipeline:
    """Per-(condition, metric, k) value vectors in population_index order."""

    def __init__(self, pid: str):
        self.pid = pid
        self.vec: Dict[Tuple[str, str, int], np.ndarray] = {}
        self.nonreturn_rows = 0
        self.nonreturn_by_condition: Dict[str, int] = {}


def load_queries(path: Path) -> Tuple[List[str], List[str], List[str]]:
    header, rows = _read_csv(path, "population")
    for col in ("population_index", "slide_id", "patient_id", "diagnostic_group"):
        if col not in header:
            raise N3Error("population.queries_column_missing", columns=[col])
    idx = [_int(r["population_index"], "population.population_index") for r in rows]
    if sorted(idx) != list(range(len(rows))):
        raise N3Error("population.population_index_not_dense")
    order = sorted(range(len(rows)), key=lambda i: idx[i])
    qids = [rows[i]["slide_id"] for i in order]
    pids = [rows[i]["patient_id"] for i in order]
    labels = [rows[i]["diagnostic_group"] for i in order]
    if len(set(qids)) != len(qids):
        raise N3Error("population.slide_id_not_unique")
    if any(not x.strip() for x in (*qids, *pids, *labels)):
        raise N3Error("population.empty_identifier")
    if len(qids) != POPULATION_N_SLIDES or len(set(pids)) != POPULATION_N_PATIENTS:
        raise N3Error("population.size", count=len(qids))
    return qids, pids, labels


def load_query_metrics(path: Path, pid: str, conditions: Sequence[str],
                       ref: Tuple[List[str], List[str], List[str]],
                       require_returned: bool) -> Pipeline:
    """Checks 1-3 and N3-16: exact header; join by (condition, k, population_index)."""
    qids, pids, labels = ref
    n = len(qids)
    cond_set = set(conditions)
    values: Dict[Tuple[str, int, int], Tuple[str, Optional[float], Optional[float]]] = {}
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = tuple(next(reader, ()))
        if header != QUERY_HEADER:
            raise N3Error("check1.query_metrics_header", pipeline_id=pid)
        for raw in reader:
            if len(raw) != len(QUERY_HEADER):
                raise N3Error("check2.ragged_row", pipeline_id=pid)
            cond, pidx, q, patient, grp, k, status = raw[:7]
            if cond not in cond_set:
                raise N3Error("check2.undeclared_condition", pipeline_id=pid)
            i = _int(pidx, "check2.population_index")
            kk = _int(k, "check2.k")
            if kk not in CUTOFFS or not 0 <= i < n:
                raise N3Error("check2.key_outside_grid", pipeline_id=pid)
            if q != qids[i] or patient != pids[i] or grp != labels[i]:
                raise N3Error("check2.joined_key_identity_mismatch", pipeline_id=pid,
                              row_key=[cond, i, kk])
            key = (cond, kk, i)
            if key in values:
                raise N3Error("check2.duplicate_key", pipeline_id=pid, row_key=[cond, i, kk])
            hit = prec = None
            if status == "returned":
                try:
                    hit, prec = float(raw[8]), float(raw[9])
                except ValueError:
                    hit = prec = None
                if hit is not None and not (math.isfinite(hit) and math.isfinite(prec)):
                    hit = prec = None
                if hit is not None and not (0.0 <= hit <= 1.0 and 0.0 <= prec <= 1.0):
                    raise N3Error("check3.value_outside_unit_interval", pipeline_id=pid,
                                  row_key=[cond, i, kk])
            values[key] = (status, hit, prec)
    if len(values) != len(conditions) * len(CUTOFFS) * n:
        raise N3Error("check2.incomplete_grid", pipeline_id=pid, count=len(values))
    pipe = Pipeline(pid)
    for cond in conditions:
        bad = set()
        for k in CUTOFFS:
            hit = np.zeros(n, dtype=np.float64)
            prec = np.zeros(n, dtype=np.float64)
            for i in range(n):
                status, h, p = values[(cond, k, i)]
                if status != "returned" or h is None:
                    bad.add(i)
                    pipe.nonreturn_rows += 1
                    continue
                hit[i], prec[i] = h, p
            hit.setflags(write=False)
            prec.setflags(write=False)
            pipe.vec[(cond, "hit", k)] = hit
            pipe.vec[(cond, "precision", k)] = prec
        pipe.nonreturn_by_condition[cond] = len(bad)
        if not np.array_equal(pipe.vec[(cond, "hit", 1)], pipe.vec[(cond, "precision", 1)]):
            raise N3Error("n3_16.precision1_differs_from_hit1", pipeline_id=pid,
                          row_key=[cond, 1])
    if require_returned and pipe.nonreturn_rows:
        raise N3Error("check3.reference_nonreturn", pipeline_id=pid, count=pipe.nonreturn_rows)
    return pipe


def check_run_record(rec: Any, pid: str, conditions: Optional[Sequence[str]],
                     groups: Optional[Sequence[str]]) -> None:
    """Check 1 + N3-10 on a run_record (run_001 or comparator)."""
    if not isinstance(rec, dict):
        raise N3Error("check1.run_record_not_object", pipeline_id=pid)
    conds = rec.get("conditions")
    if not (isinstance(conds, list) and all(isinstance(c, str) for c in conds)):
        raise N3Error("check1.conditions_missing", pipeline_id=pid)
    if len(conds) != N_CONDITIONS or conds[0] != IDENTITY or len(set(conds)) != len(conds):
        raise N3Error("check1.conditions_not_13_identity_first", pipeline_id=pid,
                      count=len(conds))
    if conditions is not None and list(conds) != list(conditions):
        raise N3Error("check1.conditions_differ", pipeline_id=pid)
    pop = rec.get("population")
    if not isinstance(pop, dict) or not isinstance(pop.get("groups"), list):
        raise N3Error("check1.groups_missing", pipeline_id=pid)
    if groups is not None and list(pop["groups"]) != list(groups):
        raise N3Error("check1.groups_differ", pipeline_id=pid)
    if (pop.get("sha256") != POPULATION_SHA256
            or pop.get("evaluated_slides") != POPULATION_N_SLIDES
            or pop.get("evaluated_patients") != POPULATION_N_PATIENTS):
        raise N3Error("n3_10.population_invariant", pipeline_id=pid)


def _table_present(tables: Any, entry: dict) -> bool:
    if not isinstance(tables, list):
        return False
    return any(isinstance(t, dict) and t.get("n_patients") == entry.get("n_patients")
               and t.get("patients_sha256") == entry.get("patients_sha256")
               and t.get("multiplicities_sha256") == entry.get("multiplicities_sha256")
               for t in tables)


def _match_recorded(tables: Any, draws: Any, pop_name: str, source: str) -> str:
    """N3-13: exactly one entry by patients_sha256, then n_patients and multiplicities."""
    mult, pats = draws_hashes(draws)
    if not isinstance(tables, list):
        raise N3Error("check4.draw_tables_missing", file=source, row_key=[pop_name])
    cands = [t for t in tables if isinstance(t, dict) and t.get("patients_sha256") == pats]
    if len(cands) != 1:
        raise N3Error("check4.patients_sha256_match_count", file=source, row_key=[pop_name],
                      count=len(cands))
    if cands[0].get("n_patients") != len(draws.patient_ids):
        raise N3Error("check4.n_patients_differs", file=source, row_key=[pop_name])
    if cands[0].get("multiplicities_sha256") != mult:
        raise N3Error("check4.multiplicities_sha256_differs", file=source, row_key=[pop_name])
    return source


# ---------------------------------------------------------------------------------------------
# Comparator admission (sha only for blinded files; N3-01/N3-04)
# ---------------------------------------------------------------------------------------------

def _safe_rel(rel: Any) -> bool:
    if not isinstance(rel, str) or not rel or rel.startswith("/") or "\\" in rel:
        return False
    return ".." not in Path(rel).parts


def whitelisted(rel: str) -> bool:
    base = rel.rsplit("/", 1)[-1]
    return (rel in WHITELIST_EXACT or rel.startswith(WHITELIST_PREFIXES)
            or base.startswith(WHITELIST_BASENAME_PREFIXES)
            or (rel.endswith(WHITELIST_SUFFIXES) and rel.startswith(WHITELIST_SUFFIX_DIRS)))


def check_outputs_in_run_record(rec: Dict[str, Any], shas: Dict[str, str],
                                pid: Optional[str]) -> None:
    """Amendment 004 item 6 (N3II-06/07): the query_metrics/paired_deltas actually used must be
    the outputs recorded (with sha256) in the pinned run_record."""
    outputs = rec.get("outputs")
    for fname, sha in shas.items():
        entry = outputs.get(fname) if isinstance(outputs, dict) else None
        if not isinstance(entry, dict) or entry.get("sha256") != sha:
            raise N3Error("n3_07.hopt_output_sha_not_in_run_record", pipeline_id=pid,
                          file=fname)


def admit_comparator(name: str, cdir: Path, acc_path: Path, acc_pin: str) -> Dict[str, Any]:
    if not cdir.is_dir():
        raise N3Error("acceptance.directory_missing", pipeline_id=name, file=str(cdir))
    if not acc_path.is_file() or sha256_file(acc_path) != acc_pin:
        raise N3Error("acceptance.record_sha256_differs_from_pin", pipeline_id=name,
                      file=str(acc_path))
    acc = _read_json(acc_path, "acceptance")
    if not isinstance(acc, dict) or acc.get("comparator") != name:
        raise N3Error("acceptance.comparator_name", pipeline_id=name)
    if acc.get("disposition") != "accepted":
        raise N3Error("acceptance.disposition_not_accepted", pipeline_id=name)
    if acc.get("pipeline_id") != name:
        raise N3Error("acceptance.pipeline_id_differs_from_name", pipeline_id=name)
    if not _iso(acc.get("recorded_at")):
        raise N3Error("acceptance.recorded_at_missing", pipeline_id=name)
    comp_path = cdir / "COMPLETE"
    if not comp_path.is_file():
        raise N3Error("complete_chain.complete_missing", pipeline_id=name)
    complete_sha = sha256_file(comp_path)
    if acc.get("complete_sha256") != complete_sha:
        raise N3Error("acceptance.complete_sha256_differs", pipeline_id=name)
    complete = _read_json(comp_path, "complete_chain")
    files = complete.get("files") if isinstance(complete, dict) else None
    if not isinstance(files, dict) or not all(
            _safe_rel(r) and isinstance(e, dict) and isinstance(e.get("sha256"), str)
            for r, e in files.items()):
        raise N3Error("complete_chain.malformed", pipeline_id=name)
    blinded = acc.get("blinded_sha256")
    if not isinstance(blinded, dict):
        raise N3Error("acceptance.blinded_sha256_missing", pipeline_id=name)
    expected = {rel for rel in files if not whitelisted(rel)}
    if set(blinded) != expected:
        raise N3Error("acceptance.blinded_set_differs_from_complete_complement",
                      pipeline_id=name, count=len(set(blinded) ^ expected))
    for rel, sha in blinded.items():
        if not isinstance(sha, str) or not SHA_RE.match(sha):
            raise N3Error("acceptance.malformed_sha", pipeline_id=name, file=rel)
        if files[rel]["sha256"] != sha:
            raise N3Error("acceptance.sha_differs_from_complete", pipeline_id=name, file=rel)
        if not (cdir / rel).is_file():
            raise N3Error("acceptance.blinded_file_missing", pipeline_id=name, file=rel)
        # N3I-01: binary stream hash only; the content is never parsed (B5).
        if sha256_file(cdir / rel) != sha:
            raise N3Error("acceptance.blinded_file_differs", pipeline_id=name, file=rel)
    actual: Dict[str, str] = {}
    for rel in COMPARATOR_READ:
        p = cdir / rel
        if rel not in files or not p.is_file():
            raise N3Error("complete_chain.read_file_not_bound", pipeline_id=name, file=rel)
        sha = sha256_file(p)
        if files[rel]["sha256"] != sha:
            raise N3Error("complete_chain.read_file_differs", pipeline_id=name, file=rel)
        actual[rel] = sha
    rec = _read_json(cdir / "run_record.json", "run_record")
    if not isinstance(rec, dict):
        raise N3Error("check1.run_record_not_object", pipeline_id=name)
    if actual["run_record.json"] == HOPT_RUN_RECORD_SHA256:
        source = "constant: frozen run_001 run_record sha256"
        pid = HOPT_PIPELINE_ID
        if "pipeline_id" in rec and rec["pipeline_id"] != pid:
            raise N3Error("n3_01.pipeline_id_differs", pipeline_id=name)
        check_outputs_in_run_record(rec, {f: actual[f] for f in RUN_RECORD_OUTPUTS}, name)
    else:
        source = "run_record.json[pipeline_id]"
        pid = rec.get("pipeline_id")
        if not isinstance(pid, str) or not pid:
            raise N3Error("n3_01.pipeline_id_missing", pipeline_id=name)
        if pid == HOPT_PIPELINE_ID:
            # N3I-08: hoptimus_1 only for the frozen run_001 run_record, in every stage.
            raise N3Error("n3_01.hoptimus_1_requires_run001_run_record", pipeline_id=name)
    if pid != name:
        raise N3Error("n3_01.pipeline_id_differs", pipeline_id=name)
    return dict(name=name, dir=cdir, acceptance_path=acc_path, acceptance_sha256=acc_pin,
                acceptance_recorded_at=acc["recorded_at"], pipeline_id=pid,
                pipeline_id_source=source, run_record=rec, files_read_sha256=actual,
                blinded_sha256=dict(blinded), complete_sha256=complete_sha,
                disposition=None)


def read_dropped(name: str, path: Path, pin: str) -> Dict[str, Any]:
    if not path.is_file():
        raise N3Error("dropped.record_missing", pipeline_id=name, file=str(path))
    sha = sha256_file(path)
    if sha != pin:
        raise N3Error("dropped.record_sha256_differs_from_pin", pipeline_id=name,
                      file=str(path))
    rec = _read_json(path, "dropped")
    if not isinstance(rec, dict) or rec.get("pipeline_id") != name:
        raise N3Error("dropped.pipeline_id_differs", pipeline_id=name)
    if rec.get("disposition") not in DROPPED_DISPOSITIONS:
        raise N3Error("dropped.disposition", pipeline_id=name)
    if not isinstance(rec.get("reason"), str) or not rec["reason"].strip():
        raise N3Error("dropped.reason_missing", pipeline_id=name)
    if not _iso(rec.get("timestamp")):
        raise N3Error("dropped.timestamp_missing", pipeline_id=name)
    return dict(name=name, pipeline_id=name, path=path, sha256=sha,
                disposition=rec["disposition"])


# ---------------------------------------------------------------------------------------------
# Stage / attempt (N3-02, N3-14)
# ---------------------------------------------------------------------------------------------

def _pairs(values: Sequence[str], flag: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for v in values or ():
        if "=" not in v:
            raise N3Error("arguments.name_value_form", columns=[flag])
        k, val = v.split("=", 1)
        if not NAME_RE.match(k) or k in out:
            raise N3Error("arguments.invalid_or_duplicate_name", columns=[flag])
        out[k] = val
    return out


def pipeline_order(names) -> List[str]:
    def key(n: str):
        if n in ENUM_PIPELINES:
            return (0, ENUM_PIPELINES.index(n), n)
        if n == HOPT_PIPELINE_ID:
            return (1, 0, n)
        return (2, 0, n)
    return sorted(names, key=key)


def check_stage_names(stage: str, names: Sequence[str]) -> List[str]:
    if stage == "replay":
        for n in names:
            if n != HOPT_PIPELINE_ID and not n.startswith(SYNTHETIC_PREFIX):
                raise N3Error("n3_02.replay_refuses_pipeline", pipeline_id=n)
        return pipeline_order(names)
    for n in names:
        if n not in ENUM_PIPELINES:
            raise N3Error("n3_02.pipeline_not_in_enum_for_stage", pipeline_id=n)
    required = REQUIRED_SETS[stage]
    missing = [n for n in required if n not in names]
    if missing:
        raise N3Error("n3_02.required_member_omitted", pipeline_id=missing[0],
                      count=len(missing))
    extra = [n for n in names if n not in required]
    if extra:
        raise N3Error("n3_02.member_outside_required_set", pipeline_id=extra[0])
    return list(required)


def _read_entry(d: Path) -> Optional[Dict[str, Any]]:
    """RECORD.json (else FAILED.json) of an attempt entry; None if absent/unparsable."""
    if not d.is_dir() or d.name.endswith(".partial"):
        return None
    for f in ("RECORD.json", "FAILED.json"):
        p = d / f
        if p.is_file():
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            return rec if isinstance(rec, dict) else None
    return None


def _as_list(value: Any) -> List[str]:
    """``--supersedes`` values: a list (candidate 003) or a single string (earlier records)."""
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v]
    return []


def _resolved(paths: Any) -> Optional[List[str]]:
    vals = _as_list(paths)
    return sorted({str(Path(v).resolve()) for v in vals}) if vals else None


def _is_attempt_number(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _scan_attempts(parent: Path, prefix: str, out: Optional[Path]) -> List[Dict[str, Any]]:
    """Every ``prefix*`` entry of ``parent`` except ``out`` itself (if given).

    ``stage``: the entry's label; ``.partial`` directories, non-directories and entries whose
    RECORD/FAILED JSON is missing, unparsable or stage-less are ``unknown`` (amendment 003 item
    2c), and so -- for ``n3_run_*`` accounting -- is a ``replay`` label (amendment 004 item 3).
    ``valid``: the entry passed ``derive_attempt`` (integer ``attempt``) and has a non-replay
    stage; only such entries' ``supersedes_resolved`` is honoured (amendment 004 item 2i)."""
    found = []
    for d in sorted(parent.glob(prefix + "*")):
        if out is not None and d.resolve() == out.resolve():
            continue
        rec = _read_entry(d)
        stage = rec.get("stage") if rec else None
        if stage not in STAGES or (prefix == "n3_run_" and stage == "replay"):
            stage = "unknown"
        attempt = rec.get("attempt") if rec else None
        valid = stage in NONREPLAY_STAGES and _is_attempt_number(attempt)
        found.append(dict(path=str(d.resolve()), stage=stage,
                          attempt=attempt if valid else None, valid=valid,
                          complete=valid and (d / "COMPLETE").is_file()
                          and (d / "RECORD.json").is_file(),
                          supersedes_resolved=(_as_list(rec.get("supersedes_resolved"))
                                               if valid else [])))
    return found


def _superseded(entries: List[Dict[str, Any]]) -> set:
    """Paths named by ``supersedes_resolved`` of a valid entry (amendment 004 item 2i)."""
    return {p for e in entries if e["valid"] for p in e["supersedes_resolved"]}


def _public(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(path=e["path"], stage=e["stage"]) for e in entries]


def _conflicts(stage: str, prior_stage: str) -> bool:
    """Amendment 003 item 2b: same stage, an unknown entry, or single <-> stage1/stage2."""
    if prior_stage == "unknown" or prior_stage == stage:
        return True
    return (stage == "single") != (prior_stage == "single")


def derive_attempt(out: Path, stage: str, supersedes: Any, defect: Optional[str],
                   p2root: Path) -> Tuple[int, List[Dict[str, Any]]]:
    """N3-14 as amended by amendment 003 item 2 and amendment 004 items 2-3."""
    if stage != "replay" and not out.name.startswith("n3_run_"):
        raise N3Error("n3_14.real_stage_out_must_be_n3_run", file=str(out))
    if stage == "replay" and out.name.startswith("n3_run_"):
        raise N3Error("n3_14.replay_must_not_write_n3_run", file=str(out))
    if stage == "replay":
        priors = [e for e in _scan_attempts(out.parent, "n3_replay_", out)
                  if e["stage"] == "replay"]
        conflicting = priors
    else:
        if out.parent.resolve() != p2root.resolve():
            raise N3Error("n3_14.out_parent_not_p2root", file=str(out))
        priors = _scan_attempts(p2root, "n3_run_", out)
        conflicting = [e for e in priors if _conflicts(stage, e["stage"])]
    named = set(_resolved(supersedes) or [])
    if conflicting:
        if not named or not named <= {e["path"] for e in conflicting}:
            raise N3Error("n3_14.supersedes_required", count=len(conflicting))
        if stage == "stage1_512px":
            # amendment 004 item 2(iii): every un-superseded prior stage-1 or unknown entry.
            done = _superseded(priors)
            required = {e["path"] for e in priors if e["stage"] in ("stage1_512px", "unknown")
                        and e["path"] not in done}
            unnamed = sorted(required - named)
            if unnamed:
                raise N3Error("n3_14.supersedes_incomplete", entries=unnamed,
                              count=len(unnamed))
        if not defect or not defect.strip():
            raise N3Error("n3_14.defect_required", count=len(conflicting))
    elif named or defect:
        raise N3Error("n3_14.no_prior_attempt_to_supersede", count=len(priors))
    return len(priors) + 1, _public(priors)


def preflight(args: argparse.Namespace) -> None:
    """Amendment 004 items 1, 4 and 5: refusals before any output (no FAILED.json)."""
    if args.defect is not None and re.search(r"\d", args.defect):
        raise N3Error("n3_14.defect_contains_number")
    p2root = Path(args.hopt_dir).resolve().parent
    if args.stage in NONREPLAY_STAGES:
        if p2root != Path(CANONICAL_P2ROOT).resolve():
            raise N3Error("n3_14.p2root_not_canonical", file=str(p2root))
    spec_path = p2root.joinpath(*SPEC_RELPATH)
    if args.stage in NONREPLAY_STAGES or args.spec_sha256 is not None:
        if not spec_path.is_file():
            raise N3Error("n3_09.spec_missing", file=str(spec_path))
        if args.spec_sha256 != sha256_file(spec_path):
            raise N3Error("n3_14.spec_sha_mismatch", file=str(spec_path))
    if args.stage in NONREPLAY_STAGES:
        bad = [k for k in THREAD_ENV if os.environ.get(k) != "1"]
        if bad:
            raise N3Error("n3_09.thread_env_not_1", columns=bad)


# ---------------------------------------------------------------------------------------------
# Kernel calls and replay comparison
# ---------------------------------------------------------------------------------------------

def _paired(pipe: Pipeline, cond: str, metric: str, k: int, ctx: dict, pop: Population) -> Any:
    try:
        return kernel.summarize_paired(
            ctx["qids"], ctx["pids"], ctx["labels"], ctx["groups"],
            pipe.vec[(cond, metric, k)], pipe.vec[(IDENTITY, metric, k)], pop.mask, pop.draws)
    except (kernel.ValidationError, kernel.ComputationError):
        raise N3Error("kernel_call", pipeline_id=pipe.pid, row_key=[cond, pop.name, metric, k])


def _rows_of(ps: Any, groups: Sequence[str]):
    return list(zip((*groups, MACRO),
                    (*ps.difference.group_estimates, ps.difference.macro),
                    (*ps.arm_a.group_estimates, ps.arm_a.macro),
                    (*ps.arm_b.group_estimates, ps.arm_b.macro)))


_LIGHT_FIELDS = ("point", "ci_low", "ci_high", "n_queries", "n_patients",
                 "zero_support_replicates", "point_status", "point_reason", "ci_status",
                 "ci_reason")


def _light(e: Any) -> SimpleNamespace:
    return SimpleNamespace(**{f: getattr(e, f) for f in _LIGHT_FIELDS})


def _light_rows(ps: Any, groups: Sequence[str]) -> list:
    return [(g, _light(e), _light(a), _light(b)) for g, e, a, b in _rows_of(ps, groups)]


def compare_target(target_rows: List[Dict[str, str]], computed: Dict[tuple, tuple], keyf,
                   check: str, pid: Optional[str], extra_ok=None) -> Dict[str, Any]:
    """Row-by-row comparison (<= TOL, identical null pattern; exact otherwise)."""
    tgt: Dict[tuple, Dict[str, str]] = {}
    for r in target_rows:
        key = keyf(r)
        if key in tgt:
            raise N3Error(f"{check}.duplicate_target_key", pipeline_id=pid, row_key=list(key))
        tgt[key] = r
    if set(tgt) != set(computed):
        raise N3Error(f"{check}.key_sets_differ", pipeline_id=pid,
                      count=len(set(tgt) ^ set(computed)))
    for key in sorted(tgt, key=lambda x: tuple(map(str, x))):
        r = tgt[key]
        e, a, b = computed[key]
        bad = []
        got = dict(delta=e.point, ci_low=e.ci_low, ci_high=e.ci_high,
                   estimate_condition=a.point, estimate_identity=b.point)
        for c in REPLAY_NUMERIC:
            try:
                t = _opt_float(r[c])
            except ValueError:
                t = math.nan
            g = got[c]
            if (t is None) != (g is None):
                bad.append(c)
            elif t is not None and not abs(t - g) <= TOL:
                bad.append(c)
        ints = dict(n_pairs=e.n_queries, n_patients=e.n_patients,
                    zero_support_replicates=e.zero_support_replicates)
        for c in REPLAY_INT:
            try:
                if int(r[c]) != ints[c]:
                    bad.append(c)
            except ValueError:
                bad.append(c)
        for c in REPLAY_STATUS:
            if _opt_str(r[c]) != getattr(e, c):
                bad.append(c)
        if r.get("population") != "both_returned":
            bad.append("population")
        if extra_ok is not None and not extra_ok(key, r):
            bad.append("population_descriptor")
        if bad:
            raise N3Error(f"{check}.mismatch", pipeline_id=pid, row_key=list(key), columns=bad)
    return dict(rows_checked=len(tgt), tolerance=TOL, passed=True)


# ---------------------------------------------------------------------------------------------
# Counts (N3-05)
# ---------------------------------------------------------------------------------------------

def _empty_tally(nominal: int) -> Dict[str, Any]:
    return {"primary_intervals_nominal": nominal,
            "primary_intervals_computed": {"hit@5": 0, "precision@5": 0, "total": 0},
            "primary_intervals_excluding_zero": {"hit@5": 0, "precision@5": 0, "total": 0},
            "patterns": {lab: 0 for lab in PATTERN_LABELS}}


def _add_tally(acc: Dict[str, Any], t: Dict[str, Any]) -> None:
    acc["primary_intervals_nominal"] += t["primary_intervals_nominal"]
    for f in ("primary_intervals_computed", "primary_intervals_excluding_zero"):
        for k in acc[f]:
            acc[f][k] += t[f][k]
    for lab in PATTERN_LABELS:
        acc["patterns"][lab] += t["patterns"][lab]


def _valid_tally(t: Any) -> bool:
    try:
        ref = _empty_tally(0)
        return (isinstance(t["primary_intervals_nominal"], int)
                and all(set(t[f]) == set(ref[f]) and all(isinstance(v, int) for v in t[f].values())
                        for f in ("primary_intervals_computed",
                                  "primary_intervals_excluding_zero", "patterns")))
    except (KeyError, TypeError, AttributeError):
        return False


def check_stage1_record(path: Path, own: Dict[str, Any], p2root: Path) -> Dict[str, Any]:
    """N3-02 / amendment 003 item 5 / amendment 004 item 2(ii): stage-1 RECORD.json (+ sibling
    counts.json), bound by its COMPLETE, located in ``P2ROOT/n3_run_*`` (never ``.partial``)
    and belonging to the latest stage-1 entry: the valid ``stage1_512px`` entry with the
    highest integer attempt, which must be COMPLETE and not superseded by a valid entry."""
    d = path.parent
    if (d.parent.resolve() != p2root.resolve() or not d.name.startswith("n3_run_")
            or d.name.endswith(".partial")):
        raise N3Error("n3_02.stage1_record_location", file=str(path))
    entries = _scan_attempts(p2root, "n3_run_", None)
    stage1 = [e for e in entries if e["valid"] and e["stage"] == "stage1_512px"]
    if not stage1:
        raise N3Error("n3_02.stage1_record_not_latest", file=str(path), count=0, state="none")
    top = max(e["attempt"] for e in stage1)
    tops = [e for e in stage1 if e["attempt"] == top]
    latest = tops[0]
    if len(tops) > 1:
        state = "ambiguous"
    elif latest["path"] in _superseded(entries):
        state = "superseded"
    elif not latest["complete"]:
        state = "failed" if (Path(latest["path"]) / "FAILED.json").is_file() else "not_complete"
    else:
        state = "complete"
    if state != "complete" or latest["path"] != str(d.resolve()):
        raise N3Error("n3_02.stage1_record_not_latest", file=latest["path"], count=top,
                      state=state)
    if path.name != "RECORD.json" or not path.is_file() or not (d / "COMPLETE").is_file():
        raise N3Error("n3_02.stage1_record_missing", file=str(path))
    complete_sha = sha256_file(d / "COMPLETE")
    complete = _read_json(d / "COMPLETE", "n3_02.stage1")
    files = complete.get("files", {}) if isinstance(complete, dict) else {}
    shas = {}
    for f in ("RECORD.json", "counts.json"):
        p = d / f
        if f not in files or not p.is_file() or files[f].get("sha256") != sha256_file(p):
            raise N3Error("n3_02.stage1_complete_chain", file=f)
        shas[f] = files[f]["sha256"]
    rec = _read_json(path, "n3_02.stage1")
    counts = _read_json(d / "counts.json", "n3_02.stage1")
    if not isinstance(rec, dict) or rec.get("stage") != "stage1_512px":
        raise N3Error("n3_02.stage1_record_stage")
    code = rec.get("code", {})
    if (code.get("n3_sha256") != own["n3_sha256"]
            or code.get("kernel_sha256") != own["kernel_sha256"]):
        raise N3Error("n3_02.stage1_code_or_kernel_sha_differs")
    s1_inputs = rec.get("inputs", {})
    for key, v in own["inputs"].items():
        if not isinstance(s1_inputs.get(key), dict) or s1_inputs[key].get("sha256") != v["sha256"]:
            raise N3Error("n3_02.stage1_input_sha_differs", file=key)
    if not isinstance(counts, dict) or not _valid_tally(counts.get("stage_totals")):
        raise N3Error("n3_02.stage1_counts_malformed")
    return dict(path=str(path.resolve()), complete_sha256=complete_sha,
                record_sha256=shas["RECORD.json"], counts_sha256=shas["counts.json"],
                stage_totals=counts["stage_totals"])


B4_DOCUMENTS = (("adapter_contract_review_path", "adapter_contract_review_sha256"),
                ("conch_inclusion_rule_path", "conch_inclusion_rule_sha256"))


def check_b4_gate(path: Path) -> Dict[str, Any]:
    """N3-02 / amendment 003 item 3: both B4 documents exist and hash to the stated shas."""
    if not path.is_file():
        raise N3Error("n3_02.b4_gate_missing", file=str(path))
    gate_sha = sha256_file(path)
    rec = _read_json(path, "n3_02.b4_gate")
    if not isinstance(rec, dict) or not all(
            isinstance(rec.get(s), str) and SHA_RE.match(rec[s])
            and isinstance(rec.get(p), str) and rec[p] for p, s in B4_DOCUMENTS):
        raise N3Error("n3_02.b4_gate_malformed", file=str(path))
    out: Dict[str, Any] = dict(path=str(path.resolve()), sha256=gate_sha)
    for pkey, skey in B4_DOCUMENTS:
        doc = Path(rec[pkey])
        if not doc.is_file():
            raise N3Error("n3_02.b4_gate_document_missing", file=str(doc))
        if sha256_file(doc) != rec[skey]:
            raise N3Error("n3_02.b4_gate_document_sha_differs", file=str(doc))
        out[pkey] = str(doc.resolve())
        out[skey] = rec[skey]
    return out


# ---------------------------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------------------------

def run(args: argparse.Namespace, argv: Sequence[str], started: str, tmp: Path,
        attempt: int, priors: List[Dict[str, Any]]) -> Dict[str, Any]:
    if kernel is None:
        raise N3Error("n3_11.kernel_sha256_differs_before_import",
                      file=str(KERNEL_PRE_IMPORT_PATH))
    kernel_path = str(Path(kernel.__file__).resolve())
    kernel_sha = sha256_file(Path(kernel_path))
    if (kernel_sha != KERNEL_SHA256 or kernel_path != KERNEL_PRE_IMPORT_PATH
            or KERNEL_PRE_IMPORT_SHA256 != KERNEL_SHA256):
        raise N3Error("n3_11.kernel_sha256_differs", file=kernel_path)
    n3_path = Path(__file__).resolve()
    n3_sha = sha256_file(n3_path)
    stage = args.stage
    comps = _pairs(args.comparator, "--comparator")
    accs = _pairs(args.acceptance, "--acceptance")
    acc_pins = _pairs(args.acceptance_sha256, "--acceptance-sha256")
    dropped_args = _pairs(args.dropped, "--dropped")
    dropped_pins = _pairs(args.dropped_sha256, "--dropped-sha256")
    if set(comps) & set(dropped_args):
        raise N3Error("arguments.comparator_also_dropped")
    for name in dropped_args:
        if name not in dropped_pins or not SHA_RE.match(dropped_pins[name]):
            raise N3Error("dropped.record_pin_missing", pipeline_id=name)
    if set(dropped_pins) - set(dropped_args):
        raise N3Error("arguments.dropped_pin_for_undeclared_pipeline")
    for name in comps:
        if name not in accs:
            raise N3Error("acceptance.record_missing", pipeline_id=name)
        if name not in acc_pins or not SHA_RE.match(acc_pins[name]):
            raise N3Error("acceptance.record_pin_missing", pipeline_id=name)
    if set(accs) - set(comps) or set(acc_pins) - set(comps):
        raise N3Error("arguments.acceptance_for_undeclared_comparator")
    order = check_stage_names(stage, list(comps) + list(dropped_args))
    if stage != "stage1_512px" and args.b4_gate:
        raise N3Error("n3_02.b4_gate_only_for_stage1")
    if stage != "stage2_conch" and args.stage1_record:
        raise N3Error("n3_02.stage1_record_only_for_stage2")
    if stage == "stage1_512px" and not args.b4_gate:
        raise N3Error("n3_02.b4_gate_required")
    if stage == "stage2_conch" and not args.stage1_record:
        raise N3Error("n3_02.stage1_record_required")
    if stage in NONREPLAY_STAGES and args.hopt_run_record_sha256 != HOPT_RUN_RECORD_SHA256:
        # amendment 003 item 7 (N3I-07): real stages contrast against run_001 only.
        raise N3Error("n3_07.hopt_reference_not_run001")
    p2root = Path(args.hopt_dir).resolve().parent
    spec_path = p2root.joinpath(*SPEC_RELPATH)
    if spec_path.is_file():
        spec_rec = dict(path=str(spec_path), sha256=sha256_file(spec_path))
    elif stage in NONREPLAY_STAGES:
        raise N3Error("n3_09.spec_missing", file=str(spec_path))
    else:
        spec_rec = dict(path=str(spec_path), sha256=None)

    hdir, ddir, sdir = Path(args.hopt_dir), Path(args.dose_dir), Path(args.sensitivity_dir)
    inputs = {
        "hopt/query_metrics.csv.gz": (hdir / "query_metrics.csv.gz", args.hopt_query_metrics_sha256),
        "hopt/run_record.json": (hdir / "run_record.json", args.hopt_run_record_sha256),
        "hopt/paired_deltas.csv": (hdir / "paired_deltas.csv", args.hopt_paired_deltas_sha256),
        "hopt/patient_draws.npy": (hdir / "patient_draws.npy", args.hopt_patient_draws_sha256),
        "queries.csv": (Path(args.queries), args.queries_sha256),
        "dose/realised_stimuli.csv": (ddir / "realised_stimuli.csv", args.realised_stimuli_sha256),
        "dose/RECORD.json": (ddir / "RECORD.json", args.dose_record_sha256),
        "dose/strata_deltas.csv": (ddir / "strata_deltas.csv", args.strata_deltas_sha256),
        "sensitivity/null_slides.csv": (sdir / "null_slides.csv", args.null_slides_sha256),
        "sensitivity/RECORD.json": (sdir / "RECORD.json", args.sensitivity_record_sha256),
        "sensitivity/sensitivity_deltas.csv": (sdir / "sensitivity_deltas.csv",
                                               args.sensitivity_deltas_sha256),
    }
    record_inputs: Dict[str, Dict[str, Any]] = {}
    for key, (p, pin) in inputs.items():
        if not SHA_RE.match(pin or ""):
            raise N3Error("pins.malformed", file=key)
        if not p.is_file():
            raise N3Error("pins.input_missing", file=key)
        if sha256_file(p) != pin:
            raise N3Error("pins.sha256_differs", file=key)
        record_inputs[key] = dict(path=str(p.resolve()), sha256=pin)
    if args.queries_sha256 != POPULATION_SHA256:
        raise N3Error("n3_10.queries_sha256_differs")

    b4 = check_b4_gate(Path(args.b4_gate)) if stage == "stage1_512px" else None
    own = dict(n3_sha256=n3_sha, kernel_sha256=kernel_sha, inputs=record_inputs)
    stage1 = (check_stage1_record(Path(args.stage1_record), own, p2root)
              if stage == "stage2_conch" else None)

    admitted = {name: admit_comparator(name, Path(comps[name]), Path(accs[name]), acc_pins[name])
                for name in pipeline_order(comps)}
    dropped = {name: read_dropped(name, Path(p), dropped_pins[name])
               for name, p in dropped_args.items()}

    # ---- reference population and H-Optimus run_001 (checks 1-2) -------------------------
    qids, pids, labels = load_queries(inputs["queries.csv"][0])
    n = len(qids)
    hrec = _read_json(inputs["hopt/run_record.json"][0], "run_record")
    check_run_record(hrec, "reference", None, None)
    if stage in NONREPLAY_STAGES:
        check_outputs_in_run_record(hrec, {
            "query_metrics.csv.gz": args.hopt_query_metrics_sha256,
            "paired_deltas.csv": args.hopt_paired_deltas_sha256}, None)
    conditions = list(hrec["conditions"])
    groups = tuple(hrec["population"]["groups"])
    if list(groups) != sorted(set(groups)) or set(groups) != set(labels):
        raise N3Error("population.groups_not_sorted_label_set")
    manifest = hrec.get("manifest_v2_sha256")
    if not isinstance(manifest, str) or not manifest:
        raise N3Error("run_record.reference_manifest_v2_missing")
    for arm in (*STRATIFIED_ARMS, *(a for a, _ in NULL_POPULATIONS)):
        if arm not in conditions:
            raise N3Error("run_record.reference_condition_missing", row_key=[arm])
    full_entry = (hrec.get("draw_tables") or [None])[0]
    if not isinstance(full_entry, dict):
        raise N3Error("check4.reference_full_table_missing")
    ctx = dict(qids=tuple(qids), pids=tuple(pids), labels=tuple(labels), groups=groups)
    hopt = load_query_metrics(inputs["hopt/query_metrics.csv.gz"][0], "reference", conditions,
                              (qids, pids, labels), require_returned=True)

    # ---- comparators: checks 1-3 + amendment 001 run_record checks ------------------------
    comp_pipes: Dict[str, Pipeline] = {}
    for name, adm in admitted.items():
        crec = adm["run_record"]
        check_run_record(crec, name, conditions, groups)
        if crec.get("manifest_v2_sha256") != manifest:
            raise N3Error("c2.manifest_v2_sha256_differs", pipeline_id=name)
        if not _table_present(crec.get("draw_tables"), full_entry):
            raise N3Error("c2.full_draw_table_missing", pipeline_id=name)
        cov_header, cov_rows = _read_csv(adm["dir"] / "coverage.csv", "check3")
        if "non_returns" not in cov_header:
            raise N3Error("check3.coverage_non_returns_missing", pipeline_id=name)
        cov_nonreturns = sum(_int(r["non_returns"], "check3.coverage_non_returns")
                             for r in cov_rows)
        pipe = load_query_metrics(adm["dir"] / "query_metrics.csv.gz", name, conditions,
                                  (qids, pids, labels), require_returned=False)
        adm["coverage_non_returns_total"] = cov_nonreturns
        adm["query_metrics_nonreturn_rows"] = pipe.nonreturn_rows
        if pipe.nonreturn_rows or cov_nonreturns:
            adm["disposition"] = "nonreturn"
            adm["nonreturn_slides_by_condition"] = pipe.nonreturn_by_condition
        else:
            adm["disposition"] = "contrasted"
            comp_pipes[name] = pipe

    # ---- populations and draw tables (check 4, B1, N3-13) ---------------------------------
    draw_cache: Dict[Tuple[str, ...], Any] = {}

    def draws_for(mask: np.ndarray) -> Any:
        key = tuple(sorted({p for p, m in zip(pids, mask) if m}))
        if key not in draw_cache:
            draw_cache[key] = kernel.make_patient_draws(list(key), count=DRAW_COUNT, seed=DRAW_SEED)
        return draw_cache[key]

    pops: Dict[str, Population] = {}
    full_draws = draws_for(np.ones(n, dtype=bool))
    fm, fp = draws_hashes(full_draws)
    if (full_entry.get("n_patients") != len(full_draws.patient_ids)
            or full_entry.get("patients_sha256") != fp
            or full_entry.get("multiplicities_sha256") != fm):
        raise N3Error("check4.full_table_differs", row_key=["full"])
    try:
        saved = np.load(inputs["hopt/patient_draws.npy"][0], allow_pickle=False)
    except (OSError, ValueError):
        raise N3Error("check4.patient_draws_unreadable")
    if saved.dtype != np.int64 or saved.shape != full_draws.multiplicities.shape:
        raise N3Error("check4.patient_draws_dtype_shape")
    saved_sha = sha256_bytes(np.ascontiguousarray(saved, dtype=np.int64).tobytes())
    if saved_sha != fm:
        raise N3Error("check4.patient_draws_array_differs")
    record_inputs["hopt/patient_draws.npy"]["array_sha256"] = saved_sha
    pops["full"] = Population("full", np.ones(n, dtype=bool), full_draws,
                              "run_001 run_record draw_tables[0]")

    rs_header, rs_rows = _read_csv(inputs["dose/realised_stimuli.csv"][0], "population")
    for col in ("slide_id", "arm", "strength", "stratum"):
        if col not in rs_header:
            raise N3Error("population.realised_stimuli_column_missing", columns=[col])
    dose_rec = _read_json(inputs["dose/RECORD.json"][0], "population")
    cuts = dose_rec.get("tertile_cuts") if isinstance(dose_rec, dict) else None
    if not isinstance(cuts, dict) or not all(
            isinstance(cuts.get(a), list) and len(cuts[a]) == 2 for a in STRATIFIED_ARMS):
        raise N3Error("population.tertile_cuts_missing")
    qpos = {q: i for i, q in enumerate(qids)}
    strata_lab: Dict[str, List[Optional[str]]] = {a: [None] * n for a in STRATIFIED_ARMS}
    for r in rs_rows:
        a, sid = r["arm"], r["slide_id"]
        if a not in strata_lab or sid not in qpos:
            raise N3Error("population.realised_stimuli_unknown_arm_or_slide")
        i = qpos[sid]
        if strata_lab[a][i] is not None:
            raise N3Error("population.realised_stimuli_duplicate", row_key=[a, sid])
        c1, c2 = (float(x) for x in cuts[a])
        s = float(r["strength"])
        if r["stratum"] != ("T1" if s <= c1 else "T2" if s <= c2 else "T3"):
            raise N3Error("population.stratum_inconsistent_with_cuts", row_key=[a, sid])
        strata_lab[a][i] = r["stratum"]
    for a in STRATIFIED_ARMS:
        if any(x is None for x in strata_lab[a]):
            raise N3Error("population.realised_stimuli_incomplete", row_key=[a])
        lab = np.array(strata_lab[a], dtype=object)
        for t in STRATA:
            mask = lab == t
            if not mask.any():
                raise N3Error("population.empty_stratum", row_key=[a, t])
            pname = f"stratum:{a}:{t}"
            d = draws_for(mask)
            pops[pname] = Population(pname, mask, d, _match_recorded(
                dose_rec.get("draw_tables"), d, pname, "dose_summary RECORD.json draw_tables"))

    ns_header, ns_rows = _read_csv(inputs["sensitivity/null_slides.csv"][0], "population")
    if tuple(ns_header) != ("slide_id", "arm", "factor", "reason"):
        raise N3Error("population.null_slides_header")
    if any(r["arm"] not in ("brightness", "spatial_shift") for r in ns_rows):
        raise N3Error("population.null_slides_unknown_arm")
    sens_rec = _read_json(inputs["sensitivity/RECORD.json"][0], "population")
    nulls = hrec.get("null_stimuli")
    if not (isinstance(nulls, dict) and nulls.get("arm") == "spatial_shift"
            and isinstance(nulls.get("slides"), list)):
        raise N3Error("population.null_stimuli_missing")
    null_lists = {"brightness": [r["slide_id"] for r in ns_rows if r["arm"] == "brightness"],
                  "spatial_shift": [r["slide_id"] for r in ns_rows if r["arm"] == "spatial_shift"]}
    if sorted(null_lists["spatial_shift"]) != sorted(str(s) for s in nulls["slides"]):
        raise N3Error("n3_13.shift_null_rows_differ_from_run_record")
    null_sets = {}
    for arm, pname in NULL_POPULATIONS:
        lst = null_lists[arm]
        s = set(lst)
        if not s or len(s) != len(lst) or not s <= set(qids):
            raise N3Error("population.null_set_invalid", row_key=[arm], count=len(lst))
        null_sets[arm] = s
        mask = np.array([q not in s for q in qids], dtype=bool)
        d = draws_for(mask)
        pops[pname] = Population(pname, mask, d, _match_recorded(
            sens_rec.get("draw_tables") if isinstance(sens_rec, dict) else None, d, pname,
            "sensitivity RECORD.json draw_tables"))

    def populations_for(cond: str) -> List[Population]:
        out = [pops["full"]]
        if cond in STRATIFIED_ARMS:
            out += [pops[f"stratum:{cond}:{t}"] for t in STRATA]
        out += [pops[p] for a, p in NULL_POPULATIONS if a == cond]
        return out

    arms = [c for c in conditions if c != IDENTITY]
    base_key = lambda r: (r["condition"], r["group"], r["metric"], _int(r["k"], "replay.k"))  # noqa: E731

    def computed_for(light, popname_of, conds) -> Dict[tuple, tuple]:
        out = {}
        for cond in conds:
            for pname, extra in popname_of(cond):
                for metric in METRICS:
                    for k in CUTOFFS:
                        for g, e, a, b in light[(pname, cond, metric, k)]:
                            out[(cond, g, metric, k, *extra)] = (e, a, b)
        return out

    # ---- H-Optimus Delta and replay (section 6.1) ----------------------------------------
    sh: Dict[Tuple[str, str, str, int], Any] = {}
    h_light: Dict[Tuple[str, str, str, int], list] = {}
    for cond in arms:
        for pop in populations_for(cond):
            for metric in METRICS:
                for k in CUTOFFS:
                    key = (pop.name, cond, metric, k)
                    s = _paired(hopt, cond, metric, k, ctx, pop)
                    h_light[key] = _light_rows(s, groups)
                    if k in D_CUTOFFS[metric]:
                        sh[key] = s
    _, pd_rows = _read_csv(inputs["hopt/paired_deltas.csv"][0], "replay")
    _, st_rows = _read_csv(inputs["dose/strata_deltas.csv"][0], "replay")
    _, se_rows = _read_csv(inputs["sensitivity/sensitivity_deltas.csv"][0], "replay")
    replay = {}
    replay["paired_deltas.csv"] = compare_target(
        pd_rows, computed_for(h_light, lambda c: [("full", ())], arms), base_key,
        "replay.paired_deltas", None)
    sizes = {p.name: int(p.mask.sum()) for p in pops.values()}
    replay["strata_deltas.csv"] = compare_target(
        st_rows, computed_for(h_light, lambda c: [(f"stratum:{c}:{t}", (t,)) for t in STRATA],
                              STRATIFIED_ARMS),
        lambda r: (*base_key(r), r["stratum"]), "replay.strata_deltas", None,
        lambda key, r: r.get("n_stratum_slides") == str(sizes[f"stratum:{key[0]}:{key[4]}"]))
    if any(r.get("analysis") != "exclude_null_stimuli" for r in se_rows):
        raise N3Error("replay.sensitivity_analysis_label")
    replay["sensitivity_deltas.csv"] = compare_target(
        se_rows, computed_for(h_light, lambda c: [(p, ()) for a, p in NULL_POPULATIONS if a == c],
                              [a for a, _ in NULL_POPULATIONS]),
        base_key, "replay.sensitivity_deltas", None,
        lambda key, r: r.get("excluded_null_slides") == str(len(null_sets[key[0]])))
    del h_light

    # ---- check 6 for every contrasted comparator before any D (N3-03/N3-06) ---------------
    for name, cp in comp_pipes.items():
        c_light = {}
        for cond in arms:
            for metric in METRICS:
                for k in CUTOFFS:
                    c_light[("full", cond, metric, k)] = _light_rows(
                        _paired(cp, cond, metric, k, ctx, pops["full"]), groups)
        _, cpd_rows = _read_csv(admitted[name]["dir"] / "paired_deltas.csv", "check6")
        admitted[name]["check6"] = compare_target(
            cpd_rows, computed_for(c_light, lambda c: [("full", ())], arms), base_key,
            "check6", name)

    # ---- D composition ---------------------------------------------------------------------
    labels_arr = np.array(labels, dtype=object)
    d_rows: List[tuple] = []
    pattern_rows: List[tuple] = []
    rep_index: List[tuple] = []
    rep_rows: List[np.ndarray] = []
    per_comp: Dict[str, Any] = {}
    for name in order:
        tally = _empty_tally(NOMINAL_PRIMARY_PER_COMPARATOR)
        if name in dropped or admitted[name]["disposition"] != "contrasted":
            disp = dropped[name]["disposition"] if name in dropped else admitted[name]["disposition"]
            for cond in arms:
                pat, reason = pattern_of(disp, None, None)
                pattern_rows.append((name, name, cond, None, None, None, None, None, None, None,
                                     None, pat, reason, cond == "reinhard"))
                tally["patterns"][pat] += 1
            per_comp[name] = dict(disposition=disp, **tally)
            continue
        cp = comp_pipes[name]
        prim: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for cond in arms:
            for pop in populations_for(cond):
                for metric in METRICS:
                    for k in D_CUTOFFS[metric]:
                        key = (pop.name, cond, metric, k)
                        s_h = sh[key]
                        s_c = _paired(cp, cond, metric, k, ctx, pop)
                        if (s_h.difference.included_query_ids != s_c.difference.included_query_ids
                                or s_h.arm_a.included_query_ids != s_c.arm_a.included_query_ids
                                or not np.array_equal(s_h.difference.replicate_group_support,
                                                      s_c.difference.replicate_group_support)):
                            raise N3Error("composition.membership_or_support_differs",
                                          pipeline_id=name, row_key=[cond, pop.name, metric, k])
                        diff_h = hopt.vec[(cond, metric, k)] - hopt.vec[(IDENTITY, metric, k)]
                        diff_c = cp.vec[(cond, metric, k)] - cp.vec[(IDENTITY, metric, k)]
                        for (g, eh, ah, bh), (_, ec, ac, bc) in zip(_rows_of(s_h, groups),
                                                                    _rows_of(s_c, groups)):
                            rowmask = pop.mask if g == MACRO else pop.mask & (labels_arr == g)
                            dh, dc = diff_h[rowmask], diff_c[rowmask]
                            if not np.any(dh != 0) and not np.any(dc != 0):
                                cause = "no_change_either"
                                dd = compose(eh, ec)
                                if ((dd["point"] is not None and dd["point"] != 0.0)
                                        or np.any(dd["replicates"][dd["valid"]] != 0.0)):
                                    raise N3Error("n3_09.no_change_row_not_exactly_zero",
                                                  pipeline_id=name,
                                                  row_key=[cond, pop.name, g, metric, k])
                            elif dh.tobytes() == dc.tobytes():
                                cause = "identical_nonzero_changes"
                                dd = compose(eh, ec, identical_nonzero=True)
                            else:
                                cause = None
                                dd = compose(eh, ec)
                            primary = g == MACRO and pop.name == "full"
                            role = ("primary" if primary and k == PRIMARY_K else
                                    "secondary" if primary else "descriptive")
                            d_rows.append((
                                name, cond, pop.name, g, metric, k, role,
                                eh.point, eh.ci_low, eh.ci_high, ec.point, ec.ci_low, ec.ci_high,
                                dd["point"], dd["ci_low"], dd["ci_high"], dd["excludes_zero"],
                                bh.point, bc.point, ah.point, ac.point,
                                bh.ci_low, bh.ci_high, bc.ci_low, bc.ci_high,
                                ah.ci_low, ah.ci_high, ac.ci_low, ac.ci_high,
                                dd["n_pairs"], dd["n_patients"], dd["point_status"],
                                dd["point_reason"], dd["ci_status"], dd["ci_reason"],
                                dd["zero_support_replicates"], zero_width_both(eh, ec), cause,
                                pop.mult_sha, pop.pat_sha))
                            if primary:
                                rep_index.append((len(rep_index), name, cond, pop.name, g,
                                                  metric, k, role))
                                rep_rows.append(dd["replicates"])
                            if role == "primary":
                                prim[(cond, metric)] = dd
                                ep = "hit@5" if metric == "hit" else "precision@5"
                                if dd["ci_status"] == "estimable":
                                    tally["primary_intervals_computed"][ep] += 1
                                    tally["primary_intervals_computed"]["total"] += 1
                                    if dd["excludes_zero"]:
                                        tally["primary_intervals_excluding_zero"][ep] += 1
                                        tally["primary_intervals_excluding_zero"]["total"] += 1
        for cond in arms:
            h5, p5 = prim[(cond, "hit")], prim[(cond, "precision")]
            pat, reason = pattern_of(None, h5, p5)
            tally["patterns"][pat] += 1
            pattern_rows.append((name, name, cond,
                                 h5["point"], h5["ci_low"], h5["ci_high"], h5["excludes_zero"],
                                 p5["point"], p5["ci_low"], p5["ci_high"], p5["excludes_zero"],
                                 pat, reason, cond == "reinhard"))
        per_comp[name] = dict(disposition="contrasted", **tally)

    # ---- outputs ---------------------------------------------------------------------------
    stage_totals = _empty_tally(0)
    for name in order:
        _add_tally(stage_totals, per_comp[name])
    stage_totals["required_set_size"] = len(order)
    counts: Dict[str, Any] = {"stage": stage, "pipeline_ids": list(order),
                              "per_comparator": per_comp, "stage_totals": stage_totals}
    if stage1 is not None:
        cum = _empty_tally(0)
        _add_tally(cum, stage1["stage_totals"])
        _add_tally(cum, stage_totals)
        cum["stages"] = ["stage1_512px", "stage2_conch"]
        counts["cumulative_totals"] = cum
    reps = (np.vstack(rep_rows).astype(np.float64) if rep_rows
            else np.zeros((0, DRAW_COUNT), dtype=np.float64))
    reps = np.ascontiguousarray(reps)
    outputs: Dict[str, bytes] = {
        "d_contrasts.csv": csv_bytes(D_HEADER, d_rows),
        "patterns.csv": csv_bytes(PATTERN_HEADER, pattern_rows),
        "counts.json": json_bytes(counts),
        "d_replicates.npy": npy_bytes(reps),
        "d_replicates_index.csv": csv_bytes(INDEX_HEADER, rep_index),
        "replay_check.json": json_bytes({
            "tolerance": TOL, "targets": replay,
            "note": "Delta_HOpt via the N3 path for every (metric, k) incl. precision@1"}),
    }
    for fname, data in outputs.items():
        (tmp / fname).write_bytes(data)

    record = dict(
        schema="r2_1_n3_perturb_contrast_v1",
        spec="r2_1_p2_eval_20261006/n3_spec_001/SPEC.md (sections 1-8 + amendments 001-004)",
        spec_path=spec_rec["path"], spec_sha256=spec_rec["sha256"],
        stage=stage, attempt=attempt, prior_attempts=priors,
        supersedes=_as_list(args.supersedes) or None,
        supersedes_resolved=_resolved(args.supersedes),
        defect=args.defect,
        argv=list(argv), host=socket.gethostname(), start=started, end=_now(),
        code={"n3_path": str(n3_path), "n3_sha256": n3_sha, "kernel_path": kernel_path,
              "kernel_sha256": kernel_sha, "kernel_pre_import_path": KERNEL_PRE_IMPORT_PATH,
              "kernel_pre_import_sha256": KERNEL_PRE_IMPORT_SHA256},
        versions={"python": platform.python_version(), "numpy": np.__version__,
                  "platform": platform.platform()},
        thread_env={k: os.environ.get(k) for k in THREAD_ENV},
        threadpool_info=_threadpool_info(),
        constants={"count": DRAW_COUNT, "seed": DRAW_SEED, "quantiles": list(QUANTILES),
                   "quantile_method": QUANTILE_METHOD, "tolerance": TOL,
                   "population_sha256": POPULATION_SHA256,
                   "population_n_slides": POPULATION_N_SLIDES,
                   "population_n_patients": POPULATION_N_PATIENTS,
                   "hopt_run_record_sha256": HOPT_RUN_RECORD_SHA256},
        inputs=record_inputs,
        population={"n_slides": n, "n_patients": len(set(pids)), "groups": list(groups),
                    "conditions": conditions, "manifest_v2_sha256": manifest},
        precision_at_1_note=("conditional_precision == conditional_hit at k = 1 asserted for "
                             "every row of every pipeline (N3-16); hit@1 absolute values stand "
                             "for precision@1"),
        draw_tables=[{"population": p.name, "n_slides": int(p.mask.sum()),
                      "n_patients": len(p.draws.patient_ids),
                      "patients_sha256": p.pat_sha, "multiplicities_sha256": p.mult_sha,
                      "matched": p.matched} for p in pops.values()],
        null_sets={a: sorted(s) for a, s in null_sets.items()},
        comparators={name: ({
            "dir": str(admitted[name]["dir"].resolve()), "pipeline_id": name,
            "pipeline_id_source": admitted[name]["pipeline_id_source"],
            "disposition": admitted[name]["disposition"],
            "acceptance_path": str(admitted[name]["acceptance_path"].resolve()),
            "acceptance_sha256": admitted[name]["acceptance_sha256"],
            "acceptance_recorded_at": admitted[name]["acceptance_recorded_at"],
            "complete_sha256": admitted[name]["complete_sha256"],
            "blinded_sha256": admitted[name]["blinded_sha256"],
            "files_read_sha256": admitted[name]["files_read_sha256"],
            "coverage_non_returns_total": admitted[name]["coverage_non_returns_total"],
            "query_metrics_nonreturn_rows": admitted[name]["query_metrics_nonreturn_rows"],
            "nonreturn_slides_by_condition": admitted[name].get("nonreturn_slides_by_condition"),
            "check6": admitted[name].get("check6")} if name in admitted else {
            "pipeline_id": name, "disposition": dropped[name]["disposition"],
            "dropped_record_path": str(dropped[name]["path"].resolve()),
            "dropped_record_sha256": dropped[name]["sha256"]}) for name in order},
        b4_gate=b4, stage1_record=stage1,
        limitation=LIMITATION_TEXT,
        arrays={"d_replicates_array_sha256": sha256_bytes(reps.tobytes()),
                "d_replicates_shape": list(reps.shape),
                "d_replicates_scope": "every MACRO row of the full population (N3-17)"},
        outputs={f: sha256_bytes(b) for f, b in outputs.items()},
    )
    return dict(record=record, processed=len(comp_pipes), omitted=len(order) - len(comp_pipes))


def _threadpool_info() -> Optional[List[Dict[str, Any]]]:
    """Amendment 003 item 8: BLAS/OpenMP pools actually loaded (recorded, not enforced)."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    keys = ("user_api", "internal_api", "prefix", "filepath", "version", "num_threads",
            "threading_layer", "architecture")
    return [{k: i.get(k) for k in keys if k in i} for i in threadpool_info()]


def _write_complete(tmp: Path) -> None:
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(tmp.iterdir())}
    (tmp / "COMPLETE").write_bytes(json_bytes({"files": files}))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m prosper.evaluation.r2_1_perturb_contrast_cli",
        description="N3 cross-encoder perturbation contrast D = Delta_HOpt - Delta_c.")
    ap.add_argument("--out", required=True, help="new output directory (must not exist)")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--supersedes", action="append", default=None, metavar="PATH",
                    help="prior attempt entry superseded by this run (repeatable)")
    ap.add_argument("--defect", default=None)
    ap.add_argument("--hopt-dir", required=True, help="H-Optimus run_001 directory")
    ap.add_argument("--hopt-query-metrics-sha256", required=True)
    ap.add_argument("--hopt-run-record-sha256", required=True)
    ap.add_argument("--hopt-paired-deltas-sha256", required=True)
    ap.add_argument("--hopt-patient-draws-sha256", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--queries-sha256", required=True)
    ap.add_argument("--dose-dir", required=True)
    ap.add_argument("--realised-stimuli-sha256", required=True)
    ap.add_argument("--dose-record-sha256", required=True)
    ap.add_argument("--strata-deltas-sha256", required=True)
    ap.add_argument("--sensitivity-dir", required=True)
    ap.add_argument("--null-slides-sha256", required=True)
    ap.add_argument("--sensitivity-record-sha256", required=True)
    ap.add_argument("--sensitivity-deltas-sha256", required=True)
    ap.add_argument("--comparator", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--acceptance", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--acceptance-sha256", action="append", default=[], metavar="NAME=SHA256")
    ap.add_argument("--dropped", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--dropped-sha256", action="append", default=[], metavar="NAME=SHA256")
    ap.add_argument("--b4-gate", default=None, metavar="PATH")
    ap.add_argument("--stage1-record", default=None, metavar="PATH")
    ap.add_argument("--spec-sha256", default=None, metavar="SHA256",
                    help="sha256 of P2ROOT/n3_spec_001/SPEC.md (required in non-replay stages)")
    return ap


def _warning_counts(caught) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for w in caught:
        out[w.category.__name__] = out.get(w.category.__name__, 0) + 1
    return out


def _publish_failed(out: Path, partial: Path, failed: Dict[str, Any]) -> None:
    """Amendment 003 item 6: always attempt a FAILED.json.  Normal path: remove the partial
    and publish ``<out>/FAILED.json`` alone by rename.  If the partial cannot be removed, it is
    left in place (it counts as an attempt) and FAILED.json is written directly into ``<out>``.
    Amendment 004 item 7 (B5): FAILED.json records ``partial_left``."""
    shutil.rmtree(partial, ignore_errors=True)
    try:
        if not os.path.lexists(partial):
            failed["partial_left"] = False
            partial.mkdir()
            (partial / "FAILED.json").write_bytes(json_bytes(failed))
            os.rename(partial, out)
            return
    except OSError:
        pass
    try:
        failed["partial_left"] = os.path.lexists(partial)
        out.mkdir(exist_ok=True)
        (out / "FAILED.json").write_bytes(json_bytes(failed))
    except OSError:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    partial = out.parent / f"{out.name}.partial"
    if os.path.lexists(out):
        print(f"REFUSED: --out exists: {out}", file=sys.stderr)
        return 2
    if not out.parent.is_dir():
        print(f"REFUSED: --out parent missing: {out.parent}", file=sys.stderr)
        return 2
    if os.path.lexists(partial):
        print(f"REFUSED: partial directory exists: {partial}", file=sys.stderr)
        return 2
    try:
        preflight(args)
    except N3Error as exc:
        # amendment 004 items 1, 4, 5: refused before any output; nothing is written.
        print(f"REFUSED: {out} check={exc.check}", file=sys.stderr)
        return 2
    started = _now()
    print(f"N3 out: {out}")
    attempt: Optional[int] = None
    caught: list = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p2root = Path(args.hopt_dir).resolve().parent
            attempt, priors = derive_attempt(out, args.stage, args.supersedes, args.defect,
                                             p2root)
            partial.mkdir()
            result = run(args, argv, started, partial, attempt, priors)
        record = result["record"]
        record["warnings"] = _warning_counts(caught)
        (partial / "RECORD.json").write_bytes(json_bytes(record))
        _write_complete(partial)
        os.rename(partial, out)
    except BaseException as exc:  # noqa: BLE001 -- default-deny handler (N3-03, N3I-06)
        interrupted = isinstance(exc, KeyboardInterrupt)
        check = (exc.check if isinstance(exc, N3Error)
                 else "interrupted" if interrupted else "unexpected")
        context = exc.context if isinstance(exc, N3Error) else {}
        # N3-03 allow-list only: check id, exception class, pipeline_id/row keys/counts/
        # paths/shas (context, argv), plus the stage label and attempt count (N3-14).
        failed = {"status": "FAILED", "check": check, "exception_class": type(exc).__name__,
                  "context": context, "stage": args.stage, "attempt": attempt,
                  "argv": argv, "warnings": _warning_counts(caught or [])}
        if attempt is None:
            # amendment 004 item 2(i): a run refused before passing derive_attempt supersedes
            # nothing; its raw --supersedes is kept only as ``supersedes_ignored``.
            failed.update(supersedes=None, supersedes_resolved=None,
                          supersedes_ignored=_as_list(args.supersedes) or None)
        else:
            failed.update(supersedes=_as_list(args.supersedes) or None,
                          supersedes_resolved=_resolved(args.supersedes),
                          supersedes_ignored=None)
        _publish_failed(out, partial, failed)
        print(f"FAILED: {out} check={check}", file=sys.stderr)
        return 130 if interrupted else 1
    print(f"N3 comparators processed: {result['processed']} omitted: {result['omitted']}")
    print(f"COMPLETE: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
