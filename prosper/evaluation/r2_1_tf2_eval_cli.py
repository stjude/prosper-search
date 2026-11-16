"""R2.1-TF2 stage S6 evaluator (specification candidate 003 + candidate-004 amendments R2, R4, R5).

Additive module: nothing in ``r2_1_filter_eval_cli`` (``fe``) or the frozen kernel
``r2_1_hoptimus_main`` (``kernel``) is edited; both are imported.

What this module adds to ``fe.evaluate_qf``:

* **Per-query reference eligibility (R2).** Per encoder x budget, an S4 eligibility table
  (``ELIGIBILITY_COLUMNS``) gives, for every population query and filtered arm, the query's
  head, its library version and the reference slide ids eligible for it (before same-patient
  exclusion). Top-k, hit, precision, coverage and hit-yield are computed against that query's own
  eligible set (``fe.rank_arm_qf`` with a per-row library mask); same-patient exclusion follows by
  population ``patient_id``; used 1e6 sentinel cells are counted and are an error unless allowed.
  The all-tissue baseline is the special case "every population slide eligible".
* **One external whole-patient draw matrix (F10).** ``make-draws`` writes
  ``kernel.make_patient_draws(population patients, count=5000, seed=123)`` once; ``evaluate``
  reuses it for every arm, method, encoder, budget and stratum. Draw mode ``shared`` (the TF-2
  rule): every summarized population (intended, returned, both-returned, a stratum) is weighted by
  the columns of that one matrix for its included patients, so all paired differences are taken
  within the same replicate. Draw mode ``per_population`` reproduces the accepted route-b rule
  (``fe.DrawCache``: fresh seed-123 draws per summarized patient set) and exists only for the F4
  regression.
* **Methods** ``ot_cosine``, ``yottixel`` and ``titan`` (TITAN: L1 distances between slide
  embeddings, consumed in the same pickled ``_no_same_patient`` DataFrame layout).
* **Bindings from a config file** (``CONFIG_SCHEMA``), baselines resolved by role from
  ``s0/input_bindings.json``; every bound file is re-hashed and must match.
* **Families** (section 3.6): (a) filtered - all-tissue, (b) 5 % - 100 %, (c) query-only -
  both-sided, all endpoints/k; (d) strata (deployment = unannotated-patient queries; annotated-
  patient queries) reported beside the primary, descriptive only (no between-stratum contrast).
* **Wording classes** (section 6 + R5) from the conditional k = 5 macro family-(a) intervals,
  100 % primary and 5 % labelled sensitivity; ``claim_sentences.txt`` one sentence per encoder;
  RER headline only for the pre-named cell UNI2 + OT both-sided 100 % (overall macro, plurality,
  non-returns as misses).
* **RER** via the sha-verified bytes of the frozen ``prosper_revision/scripts/r2_1_filter_rer.py``
  (and its pinned vote module).
* **S6-c (R4)** ``--remove-reference-slides``: the listed slides are removed from every arm's
  eligible reference set (baseline included); queries are unchanged; zero new retrieval.
  ``make-s6c-lists`` derives the 168 / 364 lists from S0 and stops on any other count.
* **F4** ``f4-regression``: route-b inputs through the new code path in both draw modes.

No p-values, no winner statements, no cross-encoder contrasts.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
import pickle
import socket
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import r2_1_filter_eval_cli as fe
from . import r2_1_hoptimus_main as kernel

EvalError = fe.EvalError
CUTOFFS = fe.CUTOFFS
SENTINEL = fe.SENTINEL
BASELINE_ARM = fe.BASELINE_ARM                      # "all_tissue"
FILTERED_ARMS = fe.QF_ARMS                          # ("query_only", "both_filtered")
ARM_NAMES = (BASELINE_ARM,) + FILTERED_ARMS
ELIGIBILITY_ARM = {"query_only": "query_only", "both_filtered": "both_sided"}
METHODS = ("ot_cosine", "yottixel", "titan")
BUDGETS = ("5", "100")
PRIMARY_BUDGET, SENSITIVITY_BUDGET = "100", "5"
PRIMARY_K = 5
DRAW_COUNT, DRAW_SEED = 5000, 123
DRAW_MODES = ("shared", "per_population")
STRATUM_ALL = "all"
STRATUM_DEPLOYMENT = "deployment_unannotated"
STRATUM_ANNOTATED = "annotated_patient"
CONFIG_SCHEMA = "r2_1_tf2_eval_config/1"
DRAWS_SCHEMA = "r2_1_tf2_shared_draws/1"

FROZEN_RER_SHA256 = "b292ca107e5841793b6182aebc4f8d9db6c0730d24962a8bea6f4339eb404367"
RER_RULE = "plurality"
RER_HANDLINGS = ("miss", "both_returned")           # frozen docstring: miss primary
HEADLINE_RER_CELL = ("uni2", "ot_cosine", "both_filtered", PRIMARY_BUDGET, "miss")
S6C_EXPECTED = {"primary": 168, "secondary": 364}

ELIGIBILITY_COLUMNS = (
    "encoder", "budget_percent", "query", "query_index", "arm", "head_slot", "head_id",
    "library_version", "query_returned", "n_eligible", "eligible_ids_sha256",
    "eligible_reference_ids")

EST_HEADER = ("column", "row_kind", "family", "encoder", "stratum") + fe.ESTIMATE_HEADER[2:]
_CLAMP_OFFSET = len(EST_HEADER) - len(fe.ESTIMATE_HEADER)          # 3
TIE_HEADER = ("tie_rule",) + EST_HEADER
QUERY_HEADER = (
    "encoder", "arm", "method", "budget", "query_index", "query", "patient_id", "diagnostic_group",
    "stratum", "k", "status", "returned_depth", "short_list", "conditional_hit",
    "conditional_precision", "conditional_reason", "coverage", "hit_yield", "n_candidates",
    "boundary_tie", "head_slot", "library_version", "library_size", "removed_eligible",
    "tie_averaged_hit", "tie_averaged_precision")
RANK_HEADER = ("encoder",) + fe.RANK_HEADER
COVERAGE_HEADER = ("encoder", "stratum") + fe.COVERAGE_HEADER_QF
TIE_COUNT_HEADER = ("encoder",) + fe.TIE_COUNT_HEADER
LIBRARY_HEADER = ("encoder", "arm", "method", "budget", "library_version", "head_ids", "n_eligible",
                  "n_eligible_after_removal", "removed_eligible", "queries_assigned",
                  "queries_returned", "non_returned_queries", "eligible_ids_sha256")
RER_HEADER = ("encoder", "method", "arm", "budget", "rule", "handling", "contrast_id", "class",
              "cell_role", "f1_all", "f1_all_ci_low", "f1_all_ci_high", "f1_filt", "f1_filt_ci_low",
              "f1_filt_ci_high", "delta_f1", "delta_ci_low", "delta_ci_high", "rer", "rer_ci_low",
              "rer_ci_high", "rer_ci_status", "n_undefined_reps", "n_zero_support_reps", "support",
              "n_queries", "n_nonreturned_filt", "n_nonreturned_all", "quantile_method")
WORDING_HEADER = ("analysis", "encoder", "method", "arm", "budget", "budget_role", "k", "estimand",
                  "hit_point", "hit_ci_low", "hit_ci_high", "hit_ci_status", "hit_direction",
                  "precision_point", "precision_ci_low", "precision_ci_high", "precision_ci_status",
                  "precision_direction", "wording_class", "wording")

DEFAULT_DISPLAY = {
    "encoder": {"uni2": "UNI2", "virchow2_fp32": "Virchow2", "h_optimus_1": "H-Optimus-1",
                "titan": "TITAN"},
    "method": {"ot_cosine": "Optimal Transport", "yottixel": "Yottixel",
               "titan": "TITAN slide-embedding L1"},
    "arm": {"query_only": "query-only", "both_filtered": "both-sided"},
}


# =====================================================================================
# Small helpers
# =====================================================================================
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_sha(path: Path, expected: Optional[str], what: str) -> str:
    if not expected:
        raise EvalError(f"{what}: no sha256 bound for {path}")
    got = fe.sha256_file(Path(path))
    if got != expected:
        raise EvalError(f"{what}: {path} sha256 {got} != bound {expected}")
    return got


def ids_sha(ids: Sequence[str]) -> str:
    """sha256 of the newline-joined sorted id strings (as ``fe._ids_sha``)."""
    return fe._ids_sha(list(ids))


def refuse_nonempty(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise EvalError(f"{out_dir} is not empty; use a new output directory (never overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _put(out_dir: Path, files: dict, name: str, data: bytes) -> None:
    p = Path(out_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "xb") as fh:
        fh.write(data)
    files[name] = {"sha256": sha256_bytes(data), "bytes": len(data)}


def _npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a)
    return buf.getvalue()


class NpyRowSink:
    """Stream float64 rows of fixed width to ``<name>`` in exactly ``np.save`` format.

    Rows are written to ``<name>.partial`` as they are produced (bounded memory) and the final
    file is header + data; the partial file is removed. Output bytes equal
    ``np.save(np.vstack(rows).astype(float64))`` for a nonempty set of rows."""

    def __init__(self, out_dir: Path, name: str):
        self.out_dir, self.name = Path(out_dir), name
        self.partial = self.out_dir / (name + ".partial")
        self.fh = open(self.partial, "xb")
        self.rows, self.width = 0, None

    def add(self, row: np.ndarray) -> None:
        row = np.ascontiguousarray(np.asarray(row, dtype=np.float64).reshape(-1))
        if self.width is None:
            self.width = row.size
        elif row.size != self.width:
            raise EvalError(f"{self.name}: replicate width {row.size} != {self.width}")
        self.fh.write(row.tobytes())
        self.rows += 1

    def close(self, files: dict) -> None:
        self.fh.close()
        shape = (self.rows, self.width) if self.rows else (0, 0)
        head = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            head, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.float64)),
                   "fortran_order": False, "shape": shape})
        h = hashlib.sha256()
        final = self.out_dir / self.name
        total = 0
        with open(final, "xb") as out, open(self.partial, "rb") as src:
            hb = head.getvalue()
            out.write(hb)
            h.update(hb)
            total += len(hb)
            for chunk in iter(lambda: src.read(1 << 24), b""):
                out.write(chunk)
                h.update(chunk)
                total += len(chunk)
        self.partial.unlink()
        files[self.name] = {"sha256": h.hexdigest(), "bytes": total, "shape": list(shape)}


# =====================================================================================
# Frozen RER core (sha-verified bytes, as scripts/r2_1_filter_rer_routeb.py does)
# =====================================================================================
class _BytesLoader(importlib.abc.Loader):
    def __init__(self, source: bytes, origin: str):
        self.source, self.origin = source, origin

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__dict__.setdefault("__file__", self.origin)
        exec(compile(self.source, self.origin, "exec"), module.__dict__)


def load_frozen_rer(path: Path, expected_sha: str = FROZEN_RER_SHA256):
    """Read the frozen B(c) RER script once, verify its sha256, execute those bytes privately."""
    source = Path(path).read_bytes()
    got = sha256_bytes(source)
    if got != expected_sha:
        raise EvalError(f"frozen RER {path} sha256 {got} != pinned {expected_sha}")
    loader = _BytesLoader(source, str(path))
    spec = importlib.util.spec_from_loader("r2_1_filter_rer_frozen_for_tf2", loader, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module, {"path": str(path), "sha256": got, "size": len(source)}


@dataclass
class RerCore:
    M: object
    plurality: object
    legacy_generous: object
    info: dict


def load_rer_core(frozen_rer: Path, vote_module: Optional[Path] = None) -> RerCore:
    M, info = load_frozen_rer(frozen_rer)
    vote_path = Path(vote_module) if vote_module else Path(frozen_rer).parent / "r2_3_internal_failure_tables.py"
    plurality, legacy, vinfo = M.load_vote_functions(str(vote_path))   # pinned sha inside
    return RerCore(M, plurality, legacy, {"frozen_rer": info, "vote_module": vinfo})


def default_frozen_rer() -> Optional[Path]:
    out = os.environ.get("PROSPER_OUT")
    return Path(out).parent / "scripts" / "r2_1_filter_rer.py" if out else None


# =====================================================================================
# Population, query assignment, shared draws
# =====================================================================================
def load_query_assignment(path: Path, pop: pd.DataFrame) -> pd.DataFrame:
    """S0 ``query_assignment.csv`` aligned to population order, cross-checked against it."""
    qa = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"slide_id", "patient_id", "diagnostic_group", "patient_annotated", "stratum",
            "head_slot", "library_version_both_sided", "library_version_query_only"}
    if not need.issubset(qa.columns):
        raise EvalError(f"{path}: lacks {sorted(need - set(qa.columns))}")
    if qa.slide_id.duplicated().any() or set(qa.slide_id) != set(pop.slide_id):
        raise EvalError(f"{path}: slides differ from the population")
    qa = qa.set_index("slide_id").reindex(pop.slide_id).reset_index()
    for c in ("patient_id", "diagnostic_group"):
        bad = int((qa[c].to_numpy() != pop[c].to_numpy()).sum())
        if bad:
            raise EvalError(f"{path}: {c} disagrees with the population for {bad} slides")
    if not set(qa.patient_annotated).issubset({"0", "1"}):
        raise EvalError(f"{path}: patient_annotated must be 0/1")
    if (qa.groupby("patient_id").patient_annotated.nunique() > 1).any():
        raise EvalError(f"{path}: patient_annotated is not constant within a patient")
    return qa


def make_shared_draws(pop: pd.DataFrame, out_dir: Path, *, population_path: Optional[Path] = None,
                      count: int = DRAW_COUNT, seed: int = DRAW_SEED) -> dict:
    """Write the one external draw matrix for the population (refuses a non-empty directory)."""
    out_dir = refuse_nonempty(out_dir)
    pids = pop.patient_id.tolist()
    d = kernel.make_patient_draws(pids, count=count, seed=seed)
    files: dict = {}
    _put(out_dir, files, "patient_draws.npy",
         _npy_bytes(np.ascontiguousarray(d.multiplicities, dtype=np.int64)))
    _put(out_dir, files, "patient_draws_patients.txt", ("\n".join(d.patient_ids) + "\n").encode())
    rec = {"schema": DRAWS_SCHEMA,
           "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "host": socket.gethostname(), "argv": sys.argv,
           "generator": "kernel.make_patient_draws (numpy PCG64)", "count": count, "seed": seed,
           "n_patients": len(d.patient_ids), "n_queries": len(pop),
           "columns": "sorted unique population patient_id (patient_draws_patients.txt)",
           "patients_sha256": hashlib.sha256("\n".join(d.patient_ids).encode()).hexdigest(),
           "multiplicities_sha256": hashlib.sha256(
               np.ascontiguousarray(d.multiplicities).tobytes()).hexdigest(),
           "population": ({"path": str(population_path), "sha256": fe.sha256_file(Path(population_path))}
                          if population_path else None),
           "kernel_sha256": fe.sha256_file(Path(kernel.__file__)),
           "outputs": dict(files)}
    _put(out_dir, files, "draws.json", json.dumps(rec, indent=1).encode())
    return {"out_dir": str(out_dir), "files": files, "record": rec}


def load_shared_draws(draws_dir: Path, pop: pd.DataFrame, *, verify_regeneration: bool = True
                      ) -> Tuple[kernel.PatientDraws, dict]:
    draws_dir = Path(draws_dir)
    rec_bytes = (draws_dir / "draws.json").read_bytes()
    rec = json.loads(rec_bytes)
    if rec.get("schema") != DRAWS_SCHEMA:
        raise EvalError(f"{draws_dir}/draws.json: schema {rec.get('schema')!r} != {DRAWS_SCHEMA}")
    for name in ("patient_draws.npy", "patient_draws_patients.txt"):
        verify_sha(draws_dir / name, rec["outputs"][name]["sha256"], "shared draws")
    patients = tuple((draws_dir / "patient_draws_patients.txt").read_text().split("\n")[:-1])
    want = tuple(sorted(set(pop.patient_id.tolist())))
    if patients != want:
        raise EvalError(f"{draws_dir}: draw columns are not the sorted population patients")
    w = np.load(draws_dir / "patient_draws.npy", allow_pickle=False)
    if w.dtype != np.int64 or w.ndim != 2 or w.shape[1] != len(patients) or w.shape[0] != rec["count"]:
        raise EvalError(f"{draws_dir}: draw matrix shape/dtype {w.shape}/{w.dtype} unexpected")
    if np.any(w.sum(axis=1) != len(patients)):
        raise EvalError(f"{draws_dir}: draw rows do not sum to P")
    if verify_regeneration:
        regen = kernel.make_patient_draws(list(patients), count=int(rec["count"]), seed=int(rec["seed"]))
        if regen.patient_ids != patients or not np.array_equal(regen.multiplicities, w):
            raise EvalError(f"{draws_dir}: draw matrix differs from make_patient_draws(seed={rec['seed']})")
    d = kernel.PatientDraws(patients, kernel._readonly(w))
    info = {"dir": str(draws_dir), "draws_json_sha256": sha256_bytes(rec_bytes),
            "patient_draws_sha256": rec["outputs"]["patient_draws.npy"]["sha256"],
            "count": int(rec["count"]), "seed": int(rec["seed"]), "n_patients": len(patients),
            "regeneration_verified": bool(verify_regeneration)}
    return d, info


def _difference(x: kernel.Estimate, y: kernel.Estimate) -> kernel.Estimate:
    """Exactly the kernel's ``summarize_paired`` within-replicate difference."""
    valid = x.valid & y.valid
    replicates = np.full(x.replicates.shape, np.nan, dtype=np.float64)
    replicates[valid] = x.replicates[valid] - y.replicates[valid]
    point = x.point - y.point if x.point is not None and y.point is not None else None
    return kernel._estimate(point, x.point_reason or y.point_reason, x.n_queries, x.n_patients,
                            replicates, valid)


class SharedDrawPolicy:
    """One external draw matrix; a population's weights are its included patients' columns."""
    mode = "shared"

    def __init__(self, pop: pd.DataFrame, draws: kernel.PatientDraws, groups: Sequence[str]):
        self.qids = tuple(pop.slide_id.tolist())
        self.pids = tuple(pop.patient_id.tolist())
        self.labels = tuple(pop.diagnostic_group.tolist())
        self.groups = tuple(groups)
        if tuple(draws.patient_ids) != tuple(sorted(set(self.pids))):
            raise EvalError("shared draws must cover exactly the population patients")
        self.draws = draws
        self.W = np.asarray(draws.multiplicities)
        self.pcol = {p: i for i, p in enumerate(draws.patient_ids)}
        self.qcol = np.array([self.pcol[p] for p in self.pids])
        self._cache: "OrderedDict[bytes, tuple]" = OrderedDict()
        self.populations: Dict[str, int] = {}

    def _weights(self, inc: np.ndarray):
        key = np.packbits(inc).tobytes()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        cols = np.unique(self.qcol[inc])
        order = tuple(self.draws.patient_ids[c] for c in cols)
        w = np.ascontiguousarray(self.W[:, cols], dtype=np.int64)
        self._cache[key] = (order, w)
        if len(self._cache) > 6:
            self._cache.popitem(last=False)
        self.populations[hashlib.sha256(key).hexdigest()[:16]] = len(order)
        return order, w

    def summarize(self, values, inclusion) -> kernel.MetricSummary:
        inc = kernel._booleans(np.asarray(inclusion, dtype=bool), "inclusion")
        vals = kernel._values(values, len(self.qids), "values")
        order, w = self._weights(inc)
        return kernel._summarize(vals, self.qids, self.pids, self.labels, self.groups, inc, w, order)

    def paired(self, values_a, values_b, inclusion) -> kernel.MetricSummary:
        a = self.summarize(values_a, inclusion)
        b = self.summarize(values_b, inclusion)
        if (a.included_query_ids != b.included_query_ids
                or not np.array_equal(a.replicate_group_support, b.replicate_group_support)):
            raise EvalError("paired arms must have identical membership and denominator tensors")
        return kernel.MetricSummary(
            a.groups, tuple(_difference(x, y) for x, y in zip(a.group_estimates, b.group_estimates)),
            _difference(a.macro, b.macro), a.included_query_ids, a.patient_ids,
            a.replicate_group_support)

    def describe(self) -> dict:
        return {"mode": self.mode, "matrix_patients": len(self.draws.patient_ids),
                "rule": "weights of a summarized population = columns of the one external draw "
                        "matrix for its included patients (no regeneration per population)"}


class PerPopulationDrawPolicy:
    """Accepted route-b rule (``fe.DrawCache``): seed-123 draws generated per patient set. F4 only."""
    mode = "per_population"

    def __init__(self, pop: pd.DataFrame, draws: kernel.PatientDraws, groups: Sequence[str],
                 seed: int = DRAW_SEED):
        self.qids = pop.slide_id.tolist()
        self.pids = pop.patient_id.tolist()
        self.labels = pop.diagnostic_group.tolist()
        self.groups = tuple(groups)
        self.cache = fe.DrawCache(draws.multiplicities.shape[0], seed)
        full = self.cache.get(self.pids)
        if full.patient_ids != draws.patient_ids or not np.array_equal(full.multiplicities,
                                                                      draws.multiplicities):
            raise EvalError("per-population draws for the intended population differ from the "
                            "external draw matrix")

    def _d(self, inc):
        return self.cache.get([p for p, r in zip(self.pids, inc) if r])

    def summarize(self, values, inclusion):
        inc = np.asarray(inclusion, dtype=bool)
        return kernel.summarize_metric(self.qids, self.pids, self.labels, self.groups, values, inc,
                                       self._d(inc))

    def paired(self, values_a, values_b, inclusion):
        inc = np.asarray(inclusion, dtype=bool)
        return kernel.summarize_paired(self.qids, self.pids, self.labels, self.groups, values_a,
                                       values_b, inc, self._d(inc)).difference

    def describe(self) -> dict:
        return {"mode": self.mode, "draw_tables": self.cache.describe(),
                "rule": "make_patient_draws(sorted included patients, seed) per summarized population "
                        "(accepted route-b rule; F4 regression only)"}


# =====================================================================================
# Eligibility tables (S4 -> S6 interface, amendment R2)
# =====================================================================================
@dataclass
class Eligibility:
    """One encoder x budget x filtered arm: per-query returned flag, library row mask, provenance."""
    returned: np.ndarray            # (n,) bool
    lib: np.ndarray                 # (n, n) bool, rows queries / columns references (population order)
    version: List[str]
    head_slot: List[str]
    head_id: List[str]
    versions: Dict[str, dict]       # version -> {mask, n_eligible, ids_sha256}
    source: str


def write_eligibility_table(df: pd.DataFrame, path: Path) -> dict:
    missing = [c for c in ELIGIBILITY_COLUMNS if c not in df.columns]
    if missing:
        raise EvalError(f"eligibility rows lack {missing}")
    buf = io.StringIO()
    df.loc[:, list(ELIGIBILITY_COLUMNS)].to_csv(buf, index=False, lineterminator="\n")
    data = gzip.compress(buf.getvalue().encode(), mtime=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:
        fh.write(data)
    return {"path": str(path), "sha256": sha256_bytes(data), "bytes": len(data), "rows": len(df)}


def read_eligibility(path: Path, pop: pd.DataFrame, encoder: str, budget: str,
                     qa: Optional[pd.DataFrame] = None) -> Dict[str, Eligibility]:
    """Parse and validate one ``s4/eligibility/<encoder>/b<budget>.csv.gz``.

    Returns {"query_only": Eligibility, "both_filtered": Eligibility}."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ELIGIBILITY_COLUMNS if c not in df.columns]
    if missing:
        raise EvalError(f"{path}: lacks columns {missing}")
    if set(df.encoder) != {encoder}:
        raise EvalError(f"{path}: encoder values {sorted(set(df.encoder))} != [{encoder!r}]")
    if set(df.budget_percent) != {str(budget)}:
        raise EvalError(f"{path}: budget_percent {sorted(set(df.budget_percent))} != [{budget!r}]")
    want_arms = set(ELIGIBILITY_ARM.values())
    if set(df.arm) != want_arms:
        raise EvalError(f"{path}: arms {sorted(set(df.arm))} != {sorted(want_arms)}")
    ids = pop.slide_id.tolist()
    col = {s: j for j, s in enumerate(ids)}
    n = len(ids)
    out: Dict[str, Eligibility] = {}
    returned_by_arm = {}
    for arm_name, earm in ELIGIBILITY_ARM.items():
        sub = df[df.arm == earm]
        if sub["query"].duplicated().any() or set(sub["query"]) != set(ids):
            raise EvalError(f"{path}: arm {earm}: queries are not exactly one row per population slide")
        sub = sub.set_index("query").reindex(ids)
        qi = sub.query_index.astype(int).to_numpy()
        if not np.array_equal(qi, pop.query_index.to_numpy()):
            raise EvalError(f"{path}: arm {earm}: query_index disagrees with the population")
        if not set(sub.query_returned).issubset({"0", "1"}):
            raise EvalError(f"{path}: arm {earm}: query_returned must be 0/1")
        returned = sub.query_returned.to_numpy() == "1"
        versions: Dict[str, dict] = {}
        raw_of: Dict[str, str] = {}
        lib = np.zeros((n, n), dtype=bool)
        for i, (v, raw, ne, sh) in enumerate(zip(sub.library_version, sub.eligible_reference_ids,
                                                 sub.n_eligible, sub.eligible_ids_sha256)):
            if not v:
                raise EvalError(f"{path}: arm {earm}: empty library_version for {ids[i]}")
            if v not in versions:
                members = [m for m in raw.split(";") if m] if raw else []
                if len(set(members)) != len(members):
                    raise EvalError(f"{path}: version {v}: duplicate eligible ids")
                outside = [m for m in members if m not in col]
                if outside:
                    raise EvalError(f"{path}: version {v}: {len(outside)} eligible ids outside the "
                                    f"population, e.g. {outside[:5]}")
                if int(ne) != len(members):
                    raise EvalError(f"{path}: version {v}: n_eligible {ne} != {len(members)} ids")
                if sh != ids_sha(members):
                    raise EvalError(f"{path}: version {v}: eligible_ids_sha256 mismatch")
                mask = np.zeros(n, dtype=bool)
                mask[[col[m] for m in members]] = True
                versions[v] = {"mask": mask, "n_eligible": len(members), "ids_sha256": sh}
                raw_of[v] = raw
            elif raw != raw_of[v]:
                members = [m for m in raw.split(";") if m] if raw else []
                if sh != versions[v]["ids_sha256"] or ids_sha(members) != sh:
                    raise EvalError(f"{path}: arm {earm}: library_version {v} lists different eligible "
                                    f"sets for different queries")
            elif sh != versions[v]["ids_sha256"] or int(ne) != versions[v]["n_eligible"]:
                raise EvalError(f"{path}: arm {earm}: version {v} metadata differs between rows")
            lib[i] = versions[v]["mask"]
        heads = sub.head_slot.tolist()
        if qa is not None:
            want_v = qa["library_version_both_sided" if earm == "both_sided"
                        else "library_version_query_only"].tolist()
            bad = [ids[i] for i in range(n) if sub.library_version.iat[i] != want_v[i]]
            if bad:
                raise EvalError(f"{path}: arm {earm}: library_version disagrees with S0 query_assignment "
                                f"for {len(bad)} queries, e.g. {bad[:5]}")
            bad = [ids[i] for i in range(n) if heads[i] != qa.head_slot.iat[i]]
            if bad:
                raise EvalError(f"{path}: arm {earm}: head_slot disagrees with S0 query_assignment for "
                                f"{len(bad)} queries, e.g. {bad[:5]}")
        returned_by_arm[earm] = returned
        out[arm_name] = Eligibility(returned, lib, sub.library_version.tolist(), heads,
                                    sub.head_id.tolist(), versions, str(path))
    if not np.array_equal(returned_by_arm["query_only"], returned_by_arm["both_sided"]):
        raise EvalError(f"{path}: query_returned differs between arms (one query mask per query)")
    return out


def eligibility_from_route_b(pop: pd.DataFrame, banks_root: Path, budget: str, encoder: str = "uni2"
                             ) -> pd.DataFrame:
    """Route-b design (head f filters fold-f queries and their library) in the S4 table schema."""
    rows = []
    ids = pop.slide_id.to_numpy(dtype=object)
    for arm_name, earm in ELIGIBILITY_ARM.items():
        des = fe.qf_design(pop, banks_root, budget, arm_name)
        cache = {}
        for i in range(len(pop)):
            f = int(des["folds"][i])
            L = des["libraries"][f]
            if L["library_id"] not in cache:
                members = sorted(ids[L["mask"]].tolist())
                cache[L["library_id"]] = (";".join(members), len(members), ids_sha(members))
            raw, ne, sh = cache[L["library_id"]]
            if sh != L["ids_sha256"]:
                raise EvalError("route-b library sha disagrees with qf_design")
            rows.append({"encoder": encoder, "budget_percent": str(budget), "query": ids[i],
                         "query_index": int(pop.query_index.iat[i]), "arm": earm,
                         "head_slot": f"fold{f}", "head_id": f"route_b_{encoder}/b{budget}/fold{f}",
                         "library_version": L["library_id"],
                         "query_returned": int(des["returned"][i]), "n_eligible": ne,
                         "eligible_ids_sha256": sh, "eligible_reference_ids": raw})
    return pd.DataFrame(rows, columns=list(ELIGIBILITY_COLUMNS))


# =====================================================================================
# Config
# =====================================================================================
@dataclass(frozen=True)
class ArmSpec:
    encoder: str
    name: str
    method: str
    budget: str
    matrix: Path
    matrix_sha256: str
    rows: Optional[Path] = None
    rows_sha256: Optional[str] = None
    binding_role: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.encoder}|{self.name}|{self.method}|{self.budget}"

    @property
    def legacy_key(self) -> str:
        return f"{self.name}|{self.method}|{self.budget}"


@dataclass
class EvalConfig:
    path: Path
    sha256: str
    raw: dict
    population: Path
    population_sha256: str
    query_assignment: Optional[Path]
    query_assignment_sha256: Optional[str]
    draws_dir: Path
    encoders: List[dict]
    arms: List[ArmSpec]
    eligibility: Dict[Tuple[str, str], Tuple[Path, str]]
    bindings: Optional[dict]


def _resolve_role(bindings: dict, role: str) -> Tuple[Path, str]:
    hits = [f for f in bindings["files"] if f.get("role") == role]
    if len(hits) != 1:
        raise EvalError(f"binding role {role!r}: {len(hits)} entries in the S0 bindings")
    f = hits[0]
    return Path(f.get("resolved_path") or f["path"]), f["sha256"]


def load_config(path: Path, *, require_complete: bool = True) -> EvalConfig:
    raw_bytes = Path(path).read_bytes()
    cfg = json.loads(raw_bytes)
    if cfg.get("schema") != CONFIG_SCHEMA:
        raise EvalError(f"{path}: schema {cfg.get('schema')!r} != {CONFIG_SCHEMA}")
    bindings = None
    if cfg.get("baseline_bindings"):
        bp = Path(cfg["baseline_bindings"]["path"])
        verify_sha(bp, cfg["baseline_bindings"].get("sha256"), "baseline bindings")
        bindings = json.loads(bp.read_bytes())
    arms: List[ArmSpec] = []
    elig: Dict[Tuple[str, str], Tuple[Path, str]] = {}
    encs = []
    for e in cfg["encoders"]:
        enc = e["encoder"]
        if enc in [x["encoder"] for x in encs]:
            raise EvalError(f"{path}: encoder {enc} listed twice")
        encs.append({"encoder": enc, "display_name": e.get("display_name")
                     or DEFAULT_DISPLAY["encoder"].get(enc, enc)})
        for b, ent in (e.get("eligibility") or {}).items():
            if str(b) not in BUDGETS:
                raise EvalError(f"{path}: {enc}: eligibility budget {b!r} not in {BUDGETS}")
            elig[(enc, str(b))] = (Path(ent["path"]), ent["sha256"])
        for a in e["arms"]:
            name, method = a["arm"], a["method"]
            budget = "" if a.get("budget") in (None, "", "-") else str(a["budget"])
            if name not in ARM_NAMES:
                raise EvalError(f"{path}: arm {name!r} not in {ARM_NAMES}")
            if method not in METHODS:
                raise EvalError(f"{path}: method {method!r} not in {METHODS}")
            if (name == BASELINE_ARM) != (budget == ""):
                raise EvalError(f"{path}: {enc}/{name}/{method}: budget must be empty only for "
                                f"{BASELINE_ARM}")
            if budget and budget not in BUDGETS:
                raise EvalError(f"{path}: budget {budget!r} not in {BUDGETS}")
            role = a.get("binding_role")
            if role:
                if bindings is None:
                    raise EvalError(f"{path}: binding_role {role} needs baseline_bindings")
                mpath, msha = _resolve_role(bindings, role)
                if a.get("matrix") or a.get("matrix_sha256"):
                    raise EvalError(f"{path}: {role}: give a binding_role or a matrix, not both")
            else:
                mpath, msha = Path(a["matrix"]), a.get("matrix_sha256")
            if not msha:
                raise EvalError(f"{path}: {enc}/{name}/{method}/{budget}: matrix sha256 required")
            rows = Path(a["rows"]) if a.get("rows") else None
            if rows is not None and not a.get("rows_sha256"):
                raise EvalError(f"{path}: rows sidecar {rows} needs rows_sha256")
            arms.append(ArmSpec(enc, name, method, budget, mpath, msha, rows, a.get("rows_sha256"), role))
    keys = [a.key for a in arms]
    if len(set(keys)) != len(keys):
        raise EvalError(f"{path}: duplicate arm keys")
    for a in arms:
        if a.name != BASELINE_ARM and (a.encoder, a.budget) not in elig:
            raise EvalError(f"{path}: {a.key} needs eligibility for budget {a.budget}")
    if require_complete:
        for e in encs:
            enc = e["encoder"]
            methods = list(dict.fromkeys(a.method for a in arms if a.encoder == enc))
            for m in methods:
                need = [f"{enc}|{BASELINE_ARM}|{m}|"] + [f"{enc}|{n}|{m}|{b}" for n in FILTERED_ARMS
                                                          for b in BUDGETS]
                miss = [k for k in need if k not in keys]
                if miss:
                    raise EvalError(f"{path}: incomplete design for {enc}/{m}: missing {miss} "
                                    f"(pass --allow-partial for fixtures)")
    qa = cfg.get("query_assignment")
    return EvalConfig(
        Path(path), sha256_bytes(raw_bytes), cfg, Path(cfg["population"]["path"]),
        cfg["population"]["sha256"], Path(qa["path"]) if qa else None, qa["sha256"] if qa else None,
        Path(cfg["draws_dir"]), encs, arms, elig, bindings)


# =====================================================================================
# Arm loading and ranking
# =====================================================================================
def load_arm_distances(spec: ArmSpec, pop: pd.DataFrame, returned: np.ndarray, lib: np.ndarray,
                       *, strict_outside: bool) -> Tuple[np.ndarray, dict]:
    """Population x population float64 distances (rows queries, columns references).

    Rows of non-returned queries are NaN and never read. Every eligible cell of a returned row must
    be present and non-NaN; with ``strict_outside`` every in-population cell outside the row's
    eligible set must be NaN (the per-version layout of combine)."""
    with open(spec.matrix, "rb") as fh:
        df = pickle.load(fh)
    if not isinstance(df, pd.DataFrame):
        raise EvalError(f"{spec.matrix}: not a DataFrame")
    rows = [str(i) for i in df.index]
    cols = [str(c) for c in df.columns]
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
        raise EvalError(f"{spec.matrix}: duplicate axis ids")
    values = df.to_numpy()
    if values.dtype.kind != "f":
        raise EvalError(f"{spec.matrix}: matrix must be floating point")
    ids = pop.slide_id.tolist()
    n = len(ids)
    ri = {s: i for i, s in enumerate(rows)}
    ci = {s: i for i, s in enumerate(cols)}
    rsel = np.array([ri.get(s, -1) for s in ids])
    csel = np.array([ci.get(s, -1) for s in ids])
    missing_rows = [ids[i] for i in np.flatnonzero(returned & (rsel < 0))]
    if missing_rows:
        raise EvalError(f"{spec.matrix}: {len(missing_rows)} returned queries have no row, "
                        f"e.g. {missing_rows[:5]}")
    if spec.name != BASELINE_ARM:
        extra = [ids[i] for i in np.flatnonzero(~returned & (rsel >= 0))]
        if extra:
            raise EvalError(f"{spec.matrix}: rows for {len(extra)} non-returned queries, e.g. {extra[:5]}")
    need_cols = lib[returned].any(axis=0)
    miss_cols = [ids[j] for j in np.flatnonzero(need_cols & (csel < 0))]
    if miss_cols:
        raise EvalError(f"{spec.matrix}: eligible references missing from the columns, e.g. {miss_cols[:5]}")
    dist = np.full((n, n), np.nan)
    rq = np.flatnonzero(returned)
    pc = np.flatnonzero(csel >= 0)
    dist[np.ix_(rq, pc)] = values[np.ix_(rsel[rq], csel[pc])].astype(np.float64)
    present = ~np.isnan(dist)
    nan_eligible = int((lib[rq] & ~present[rq]).sum())
    if nan_eligible:
        raise EvalError(f"{spec.matrix}: {nan_eligible} eligible cells are NaN or absent")
    outside = int((~lib[rq] & present[rq]).sum())
    if strict_outside and outside:
        raise EvalError(f"{spec.matrix}: {outside} non-NaN cells outside the rows' eligible sets "
                        f"(row library differs from the eligibility table)")
    diag = {"matrix_shape": list(values.shape),
            "extra_axis_ids_outside_population": len(set(cols) - set(ids)),
            "non_nan_cells_outside_eligible": outside}
    return dist, diag


def read_rows_sidecar(path: Path) -> pd.DataFrame:
    """Per-row provenance sidecar: route-b CSV, or JSON (a list of row objects or {"rows": [...]})."""
    path = Path(path)
    if path.suffix == ".json":
        obj = json.loads(path.read_bytes())
        rows = obj["rows"] if isinstance(obj, dict) else obj
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise EvalError(f"{path}: JSON sidecar must be a list of row objects or {{'rows': [...]}}")
        df = pd.DataFrame(rows)
        if "query" not in df.columns:
            raise EvalError(f"{path}: sidecar rows lack 'query'")
        return df.astype(str)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def check_rows_sidecar(spec: ArmSpec, pop: pd.DataFrame, E: Eligibility) -> dict:
    """Per-row provenance of the combined matrix agrees with the eligibility table."""
    with open(spec.matrix, "rb") as fh:
        df = pickle.load(fh)
    mrows = [str(i) for i in df.index]
    side = read_rows_sidecar(spec.rows)
    if side["query"].tolist() != mrows:
        raise EvalError(f"{spec.rows}: sidecar queries differ from the matrix rows")
    idx = {s: i for i, s in enumerate(pop.slide_id)}
    vcol = "library_version" if "library_version" in side.columns else (
        "library_id" if "library_id" in side.columns else None)
    checked = []
    for r in side.itertuples(index=False):
        i = idx[r.query]
        v = E.version[i]
        if vcol is not None and getattr(r, vcol) != v:
            raise EvalError(f"{spec.rows}: {r.query}: {vcol} {getattr(r, vcol)} != eligibility {v}")
        if "library_ids_sha256" in side.columns and r.library_ids_sha256 != E.versions[v]["ids_sha256"]:
            raise EvalError(f"{spec.rows}: {r.query}: library_ids_sha256 differs from the eligibility table")
        if "library_size" in side.columns and int(r.library_size) != E.versions[v]["n_eligible"]:
            raise EvalError(f"{spec.rows}: {r.query}: library_size differs from the eligibility table")
        if "head_id" in side.columns and r.head_id != E.head_id[i]:
            raise EvalError(f"{spec.rows}: {r.query}: head_id differs from the eligibility table")
    for c in (vcol, "library_ids_sha256", "library_size", "head_id"):
        if c and c in side.columns:
            checked.append(c)
    return {"rows": len(side), "fields_checked": checked}


def contrast_specs(arms: Sequence[ArmSpec]) -> List[Tuple[str, str, str]]:
    """(family, A key, B key), A minus B. Order generalises ``fe.qf_contrast_specs``."""
    have = {a.key for a in arms}
    out = []
    for enc in dict.fromkeys(a.encoder for a in arms):
        for m in dict.fromkeys(a.method for a in arms if a.encoder == enc):
            base = f"{enc}|{BASELINE_ARM}|{m}|"
            for b in BUDGETS:
                q, bo = f"{enc}|query_only|{m}|{b}", f"{enc}|both_filtered|{m}|{b}"
                for a in (q, bo):
                    if a in have and base in have:
                        out.append(("a_filtered_minus_all_tissue", a, base))
                if q in have and bo in have:
                    out.append(("c_query_only_minus_both_sided", q, bo))
            for a in FILTERED_ARMS:
                x, y = f"{enc}|{a}|{m}|{SENSITIVITY_BUDGET}", f"{enc}|{a}|{m}|{PRIMARY_BUDGET}"
                if x in have and y in have:
                    out.append(("b_budget5_minus_budget100", x, y))
    return out


def _legacy(key: str) -> str:
    return key.split("|", 1)[1]


# =====================================================================================
# Evaluation
# =====================================================================================
def load_removed(path: Optional[Path], pop: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    if path is None:
        return None, None
    raw = Path(path).read_bytes()
    ids = [x.strip() for x in raw.decode().splitlines() if x.strip()]
    if len(set(ids)) != len(ids):
        raise EvalError(f"{path}: duplicate slide ids")
    col = {s: j for j, s in enumerate(pop.slide_id)}
    bad = [s for s in ids if s not in col]
    if bad:
        raise EvalError(f"{path}: {len(bad)} ids outside the population, e.g. {bad[:5]}")
    mask = np.zeros(len(pop), dtype=bool)
    mask[[col[s] for s in ids]] = True
    return mask, {"path": str(path), "sha256": sha256_bytes(raw), "n_slides": len(ids)}


def evaluate(cfg: EvalConfig, *, draw_mode: str = "shared", removed_path: Optional[Path] = None,
             allow_used_sentinel: bool = False, rer: Optional[RerCore] = None,
             rer_budgets: Sequence[str] = BUDGETS, analysis: str = "primary",
             out_dir: Path, verify_draws: bool = True) -> dict:
    """Run the S6 evaluation for every arm of ``cfg`` and write all outputs into ``out_dir``."""
    if draw_mode not in DRAW_MODES:
        raise EvalError(f"draw_mode must be one of {DRAW_MODES}")
    t0 = time.time()
    out_dir = refuse_nonempty(out_dir)
    files: dict = {}
    bindings = {"config": {"path": str(cfg.path), "sha256": cfg.sha256}}
    verify_sha(cfg.population, cfg.population_sha256, "population")
    bindings["population"] = {"path": str(cfg.population), "sha256": cfg.population_sha256}
    pop = fe.load_population(cfg.population)
    qa = None
    if cfg.query_assignment is not None:
        verify_sha(cfg.query_assignment, cfg.query_assignment_sha256, "query assignment")
        bindings["query_assignment"] = {"path": str(cfg.query_assignment),
                                        "sha256": cfg.query_assignment_sha256}
        qa = load_query_assignment(cfg.query_assignment, pop)
    draws, dinfo = load_shared_draws(cfg.draws_dir, pop, verify_regeneration=verify_draws)
    bindings["draws"] = dinfo
    groups = tuple(sorted(pop.diagnostic_group.unique()))
    policy = (SharedDrawPolicy(pop, draws, groups) if draw_mode == "shared"
              else PerPopulationDrawPolicy(pop, draws, groups, seed=dinfo["seed"]))
    removed, rinfo = load_removed(removed_path, pop)
    bindings["removed_reference_slides"] = rinfo
    n = len(pop)
    qids, pids, labels = pop.slide_id.tolist(), pop.patient_id.tolist(), pop.diagnostic_group.tolist()
    strata = OrderedDict([(STRATUM_ALL, np.ones(n, dtype=bool))])
    if qa is not None:
        ann = qa.patient_annotated.to_numpy() == "1"
        strata[STRATUM_DEPLOYMENT] = ~ann
        strata[STRATUM_ANNOTATED] = ann
    stratum_of = (np.where(qa.patient_annotated.to_numpy() == "1", STRATUM_ANNOTATED, STRATUM_DEPLOYMENT)
                  if qa is not None else np.array([STRATUM_ALL] * n))

    elig_cache: Dict[Tuple[str, str], Dict[str, Eligibility]] = {}
    bindings["eligibility"] = {}
    for (enc, b), (p, sha) in cfg.eligibility.items():
        verify_sha(p, sha, f"eligibility {enc}/b{b}")
        elig_cache[(enc, b)] = read_eligibility(p, pop, enc, b, qa)
        bindings["eligibility"][f"{enc}|{b}"] = {"path": str(p), "sha256": sha}

    per_arm: "OrderedDict[str, dict]" = OrderedDict()
    matrix_diag, lib_rows, tie_counts = {}, [], []
    query_rows, rank_rows = [], []
    bindings["arms"] = []
    for spec in cfg.arms:
        verify_sha(spec.matrix, spec.matrix_sha256, spec.key)
        brec = {"key": spec.key, "matrix": str(spec.matrix), "matrix_sha256": spec.matrix_sha256,
                "binding_role": spec.binding_role, "rows": str(spec.rows) if spec.rows else None,
                "rows_sha256": spec.rows_sha256}
        if spec.name == BASELINE_ARM:
            returned_in = np.ones(n, dtype=bool)
            lib = np.ones((n, n), dtype=bool)
            version = ["all_tissue_population"] * n
            head_slot, head_id = [""] * n, [""] * n
            versions = {"all_tissue_population": {"mask": np.ones(n, dtype=bool), "n_eligible": n,
                                                  "ids_sha256": ids_sha(qids)}}
            E = None
        else:
            E = elig_cache[(spec.encoder, spec.budget)][spec.name]
            returned_in, lib, version = E.returned, E.lib, E.version
            head_slot, head_id, versions = E.head_slot, E.head_id, E.versions
        dist, mdiag = load_arm_distances(spec, pop, returned_in, lib,
                                         strict_outside=spec.name != BASELINE_ARM)
        if spec.rows is not None:
            verify_sha(spec.rows, spec.rows_sha256, f"{spec.key} rows")
            if E is None:
                raise EvalError(f"{spec.key}: a rows sidecar is only defined for filtered arms")
            mdiag["rows_sidecar"] = check_rows_sidecar(spec, pop, E)
        bindings["arms"].append(brec)
        lib_eff = lib & ~removed[None, :] if removed is not None else lib
        removed_eligible = (lib & removed[None, :]).sum(axis=1) if removed is not None else np.zeros(n, int)
        statuses, rel, ranks, n_cand, ties, ta, rdiag = fe.rank_arm_qf(
            pop, dist, returned_in, lib_eff, allow_used_sentinel)
        del dist
        matrix_diag[spec.key] = {**mdiag, **rdiag, "matrix": str(spec.matrix),
                                 "returned": int(returned_in.sum()), "intended": n,
                                 "removed_eligible_cells": int(removed_eligible[returned_in].sum()),
                                 "boundary_ties": {str(k): int(sum(t[k] for t in ties)) for k in CUTOFFS}}
        vers_of = np.array(version, dtype=object)
        for v, info in versions.items():
            assigned = vers_of == v
            mask_eff = info["mask"] & ~removed if removed is not None else info["mask"]
            nr = sorted(np.asarray(qids, dtype=object)[assigned & ~returned_in].tolist())
            lib_rows.append((spec.encoder, spec.name, spec.method, spec.budget, v,
                             ";".join(sorted(set(np.asarray(head_id, dtype=object)[assigned].tolist()))),
                             info["n_eligible"], int(mask_eff.sum()),
                             int(info["n_eligible"] - mask_eff.sum()), int(assigned.sum()),
                             int((assigned & returned_in).sum()), ";".join(nr), info["ids_sha256"]))
        lib_size = lib_eff.sum(axis=1)
        vec = {}
        for k in CUTOFFS:
            v = {x: np.zeros(n) for x in ("hit", "precision", "hit_yield", "hit_ta", "precision_ta",
                                           "hit_yield_ta", "coverage")}
            for i in range(n):
                m = kernel.query_metrics(rel[i], k, statuses[i])
                v["hit"][i] = m.conditional_hit or 0.0
                v["precision"][i] = m.conditional_precision or 0.0
                v["hit_yield"][i] = m.hit_yield
                v["coverage"][i] = m.coverage
                if statuses[i] == "returned":
                    v["hit_ta"][i], v["precision_ta"][i] = ta[i][k]
                    v["hit_yield_ta"][i] = v["hit_ta"][i]
                query_rows.append((
                    spec.encoder, spec.name, spec.method, spec.budget, i, qids[i], pids[i], labels[i],
                    stratum_of[i], k, m.status, m.returned_depth, int(m.short_list), m.conditional_hit,
                    m.conditional_precision, m.conditional_reason, m.coverage, m.hit_yield, n_cand[i],
                    int(ties[i][k]), head_slot[i], version[i], int(lib_size[i]) if returned_in[i] else 0,
                    int(removed_eligible[i]),
                    ta[i][k][0] if statuses[i] == "returned" else None,
                    ta[i][k][1] if statuses[i] == "returned" else None))
            for x in ("hit", "precision", "hit_yield", "hit_ta", "precision_ta", "hit_yield_ta"):
                vec[(x, k)] = v[x]
            vec[("coverage", 0)] = v["coverage"]
            r = np.array([s == "returned" for s in statuses])
            tie_counts.append((spec.encoder, spec.name, spec.method, spec.budget, k, int(r.sum()),
                               int(sum(t[k] for t in ties)),
                               int(np.count_nonzero(r & (v["hit"] != v["hit_ta"]))),
                               int(np.count_nonzero(r & (v["precision"] != v["precision_ta"])))))
        returned = np.array([s == "returned" for s in statuses])
        top5 = [[labels[j] for j, _ in ranks[i][:PRIMARY_K]] for i in range(n)]
        per_arm[spec.key] = dict(spec=spec, vec=vec, returned=returned, top5=top5)
        for i in range(n):
            for rr, (j, d) in enumerate(ranks[i], start=1):
                rank_rows.append((spec.encoder, spec.name, spec.method, spec.budget, qids[i], rr, qids[j],
                                  labels[j], int(labels[j] == labels[i]), d))

    specs = contrast_specs([pa["spec"] for pa in per_arm.values()])
    all_in = np.ones(n, dtype=bool)

    def arm_estimates(col, sink, rows_out, stratum, smask, suffix=""):
        for key, pa in per_arm.items():
            s_, vec = pa["spec"], pa["vec"]
            ret = pa["returned"] & smask
            pre = ("estimate", "", s_.encoder, stratum, s_.name, s_.method, s_.budget, "", "")
            for ep, k in fe._endpoint_list():
                sm = policy.summarize(vec[(ep + suffix, k)], ret)
                a, reps = fe._rows_from(col, (*pre, ep, k), "returned", sm, groups)
                rows_out += a; col += len(a)
                for r_ in reps:
                    sink.add(r_)
            for k in CUTOFFS:
                sm = policy.summarize(vec[("hit_yield" + suffix, k)], smask)
                a, reps = fe._rows_from(col, (*pre, "hit_yield", k), "intended", sm, groups)
                rows_out += a; col += len(a)
                for r_ in reps:
                    sink.add(r_)
            if not suffix:
                sm = policy.summarize(vec[("coverage", 0)], smask)
                a, reps = fe._rows_from(col, (*pre, "coverage", ""), "intended", sm, groups)
                rows_out += a; col += len(a)
                for r_ in reps:
                    sink.add(r_)
        return col

    def contrasts(col, sink, rows_out, stratum, smask, suffix=""):
        for fam, ka, kb in specs:
            A, B = per_arm[ka], per_arm[kb]
            both = A["returned"] & B["returned"] & smask
            s_ = A["spec"]
            pre = ("contrast", fam, s_.encoder, stratum, s_.name, s_.method, s_.budget, _legacy(ka), _legacy(kb))
            for ep, k in fe._endpoint_list():
                sm = policy.paired(A["vec"][(ep + suffix, k)], B["vec"][(ep + suffix, k)], both)
                a, reps = fe._rows_from(col, (*pre, ep, k), "both_returned", sm, groups)
                rows_out += a; col += len(a)
                for r_ in reps:
                    sink.add(r_)
            for k in CUTOFFS:
                sm = policy.paired(A["vec"][("hit_yield" + suffix, k)], B["vec"][("hit_yield" + suffix, k)],
                                   smask)
                a, reps = fe._rows_from(col, (*pre, "hit_yield", k), "intended", sm, groups)
                rows_out += a; col += len(a)
                for r_ in reps:
                    sink.add(r_)
        return col

    try:
        est_rows, con_rows = [], []
        sink = NpyRowSink(out_dir, "replicates.npy")
        col = arm_estimates(0, sink, est_rows, STRATUM_ALL, all_in)
        contrasts(col, sink, con_rows, STRATUM_ALL, all_in)
        sink.close(files)
        ta_est, ta_con = [], []
        sink = NpyRowSink(out_dir, "tie_averaged_replicates.npy")
        c2 = arm_estimates(0, sink, ta_est, STRATUM_ALL, all_in, "_ta")
        contrasts(c2, sink, ta_con, STRATUM_ALL, all_in, "_ta")
        sink.close(files)
        st_est, st_con = [], []
        if len(strata) > 1:
            sink = NpyRowSink(out_dir, "strata_replicates.npy")
            c3 = 0
            for sname, smask in strata.items():
                if sname == STRATUM_ALL:
                    continue
                c3 = arm_estimates(c3, sink, st_est, sname, smask)
            for sname, smask in strata.items():
                if sname == STRATUM_ALL:
                    continue
                c3 = contrasts(c3, sink, st_con, sname, smask)
            sink.close(files)
    except BaseException:
        _put(out_dir, files, "FAILED.json", json.dumps(
            {"status": "FAILED", "traceback": traceback.format_exc()}, indent=1).encode())
        raise

    cov_rows = []
    gcol = pop.diagnostic_group.to_numpy()
    for key, pa in per_arm.items():
        s_, vec, ret = pa["spec"], pa["vec"], pa["returned"]
        for sname, smask in strata.items():
            for g in (*groups, "ALL"):
                sel = smask & (np.ones(n, dtype=bool) if g == "ALL" else gcol == g)
                nr = [qids[i] for i in np.flatnonzero(sel & ~ret)]
                tot = int(sel.sum())
                cov_rows.append((s_.encoder, sname, s_.name, s_.method, s_.budget,
                                 "pooled_slides_NOT_macro" if g == "ALL" else "group", g, tot,
                                 int((sel & ret).sum()), len(nr), ";".join(nr),
                                 float((sel & ret).sum() / tot) if tot else None,
                                 *((float(vec[("hit_yield", k)][sel].mean()) if tot else None)
                                   for k in CUTOFFS)))

    rer_rows, rer_meta = [], None
    if rer is not None:
        rer_rows, rer_meta = rer_tables(rer, pop, groups, per_arm, specs, draws, rer_budgets)

    est, n1 = fe.clamp_rows(est_rows, _CLAMP_OFFSET)
    con, n2 = fe.clamp_rows(con_rows, _CLAMP_OFFSET)
    st_e, n4 = fe.clamp_rows(st_est, _CLAMP_OFFSET)
    st_c, n5 = fe.clamp_rows(st_con, _CLAMP_OFFSET)
    ep_i = EST_HEADER.index("endpoint")
    tie_rows = [("stable_reference_index", *r) for r in (*est_rows, *con_rows)
                if r[ep_i] in ("hit", "precision", "hit_yield")]
    tie_rows += [("tie_averaged", *r) for r in (*ta_est, *ta_con)]
    tie_rows, n3 = fe.clamp_rows(tie_rows, _CLAMP_OFFSET + 1)
    wording_rows, sentences = wording_tables(con, cfg, analysis, rer_rows)

    _put(out_dir, files, "estimates.csv", fe._csv_bytes(EST_HEADER, est))
    _put(out_dir, files, "contrasts.csv", fe._csv_bytes(EST_HEADER, con))
    if len(strata) > 1:
        _put(out_dir, files, "strata_estimates.csv", fe._csv_bytes(EST_HEADER, st_e))
        _put(out_dir, files, "strata_contrasts.csv", fe._csv_bytes(EST_HEADER, st_c))
    _put(out_dir, files, "tie_sensitivity.csv", fe._csv_bytes(TIE_HEADER, tie_rows))
    _put(out_dir, files, "tie_counts.csv", fe._csv_bytes(TIE_COUNT_HEADER, tie_counts))
    _put(out_dir, files, "coverage.csv", fe._csv_bytes(COVERAGE_HEADER, cov_rows))
    _put(out_dir, files, "libraries.csv", fe._csv_bytes(LIBRARY_HEADER, lib_rows))
    _put(out_dir, files, "query_metrics.csv.gz", fe._csv_bytes(QUERY_HEADER, query_rows, compressed=True))
    _put(out_dir, files, "ranked_lists.csv.gz", fe._csv_bytes(RANK_HEADER, rank_rows, compressed=True))
    if rer is not None:
        _put(out_dir, files, "rer_contrasts.csv", fe._csv_bytes(RER_HEADER, rer_rows))
    _put(out_dir, files, "wording_classes.csv", fe._csv_bytes(WORDING_HEADER, wording_rows))
    _put(out_dir, files, "claim_sentences.txt", ("\n".join(sentences) + "\n").encode())

    code = {str(p): fe.sha256_file(p) for p in (Path(__file__).resolve(), Path(fe.__file__).resolve(),
                                                Path(kernel.__file__).resolve())}
    bindings["code_sha256"] = code
    if rer is not None:
        bindings["rer_core"] = rer.info
    bindings["versions"] = {"numpy": np.__version__, "pandas": pd.__version__,
                            "python": sys.version.split()[0]}
    _put(out_dir, files, "input_bindings.json", json.dumps(bindings, indent=1, default=str).encode())
    summary = {
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"), "argv": sys.argv,
        "analysis": analysis, "draw_mode": draw_mode, "draw_policy": policy.describe(),
        "design": "R2.1-TF2 Option C: per-query reference eligibility from S4 tables; same-patient "
                  "exclusion after eligibility by population patient_id",
        "population": {"slides": n, "patients": int(pop.patient_id.nunique()), "groups": len(groups)},
        "groups": list(groups), "strata": {k: int(v.sum()) for k, v in strata.items()},
        "contrast_pairs": [dict(family=f, arm_a=a, arm_b=b) for f, a, b in specs],
        "matrix_diagnostics": matrix_diag,
        "clamped_csv_cells": {"estimates": n1, "contrasts": n2, "tie_sensitivity": n3,
                              "strata_estimates": n4, "strata_contrasts": n5,
                              "rule": f"|x| < {fe.CLAMP_ABS} written as 0.0 in CSV only"},
        "replicates_rows": "replicates.npy rows align with `column` of estimates.csv then contrasts.csv; "
                           "tie_averaged_replicates.npy with the tie_averaged rows of tie_sensitivity.csv; "
                           "strata_replicates.npy with strata_estimates.csv then strata_contrasts.csv",
        "rer": rer_meta,
        "policy": {
            "cutoffs": list(CUTOFFS), "precision_cutoffs_reported": list(fe.QF_PRECISION_CUTOFFS),
            "precision_denominator": "k (short lists keep k)",
            "estimands": "conditional (returned / both-returned queries) and hit-yield (intended)",
            "families": {"a": "filtered minus all-tissue", "b": "5 % minus 100 % within arm",
                         "c": "query-only minus both-sided within budget",
                         "d": "strata tables (deployment_unannotated, annotated_patient) beside the "
                              "primary; descriptive; no between-stratum contrast is computed"},
            "tie_rules": fe.TIE_RULES, "primary_tie_rule": "stable_reference_index",
            "s6c": "listed slides removed from every arm's eligible reference set (baseline included); "
                   "queries unchanged" if rinfo else None,
            "wording": "section 6 classes from the conditional k=5 macro family-(a) intervals as written "
                       "to contrasts.csv: excludes 0 in favour iff ci_low > 0; against iff ci_high < 0",
            "no_p_values": True, "no_cross_encoder_contrasts": True},
        "counts": {"estimates": len(est), "contrasts": len(con), "strata_estimates": len(st_e),
                   "strata_contrasts": len(st_c), "tie_sensitivity": len(tie_rows),
                   "query_rows": len(query_rows), "rank_rows": len(rank_rows), "rer_rows": len(rer_rows)},
        "wall_seconds": round(time.time() - t0, 1),
        "outputs": dict(files)}
    _put(out_dir, files, "summary.json", json.dumps(summary, indent=1, default=str).encode())
    sums = "".join(f"{v['sha256']}  {k}\n" for k, v in sorted(files.items()))
    with open(out_dir / "SHA256SUMS.txt", "x") as fh:
        fh.write(sums)
    complete = {"status": "COMPLETE", "analysis": analysis, "draw_mode": draw_mode,
                "created": summary["created"], "sha256sums": sha256_bytes(sums.encode()),
                "summary_sha256": files["summary.json"]["sha256"]}
    with open(out_dir / "COMPLETE.json", "x") as fh:
        fh.write(json.dumps(complete, indent=1))
    return {"out_dir": str(out_dir), "files": files, "summary": summary,
            "_est_rows": est_rows, "_con_rows": con_rows, "_wording": wording_rows,
            "_sentences": sentences, "_rer": rer_rows, "_per_arm_returned":
            {k: v["returned"] for k, v in per_arm.items()}}


# =====================================================================================
# RER (frozen core)
# =====================================================================================
def rer_tables(rer: RerCore, pop: pd.DataFrame, groups, per_arm, specs, draws: kernel.PatientDraws,
               budgets: Sequence[str]):
    M = rer.M
    truth = pop.diagnostic_group.to_numpy()
    pcol = {p: i for i, p in enumerate(draws.patient_ids)}
    W = np.asarray(draws.multiplicities)[:, [pcol[p] for p in pop.patient_id]].astype(np.float64)
    gi = {g: i for i, g in enumerate(groups)}
    T = np.zeros((len(pop), len(groups)))
    T[np.arange(len(pop)), [gi[t] for t in truth]] = 1.0
    preds_cache = {}

    def preds(key):
        if key not in preds_cache:
            pa = per_arm[key]
            p = []
            for i in range(len(pop)):
                if not pa["returned"][i]:
                    p.append(M.NONE_LABEL)
                else:
                    p.append(M.vote(pa["top5"][i], truth[i], RER_RULE, rer.plurality, rer.legacy_generous))
            preds_cache[key] = (p, M.prediction_matrix(p, list(groups)))
        return preds_cache[key]

    rows = []
    n_reps = W.shape[0]
    for fam, ka, kb in specs:
        if not fam.startswith("a_"):
            continue
        A = per_arm[ka]["spec"]
        if A.budget not in budgets:
            continue
        nonret_f = ~per_arm[ka]["returned"]
        nonret_a = ~per_arm[kb]["returned"]
        _, PF = preds(ka)
        _, PA = preds(kb)
        cid = f"{A.encoder}:{A.method}:{A.name}:{A.budget}-vs-all_tissue"
        for handling in RER_HANDLINGS:
            inc = M.inclusion_mask(handling, nonret_a, nonret_f)
            res = M.evaluate_contrast(W, T, PA, PF, inc)
            out, _ = M.summarise_contrast(res, list(groups), n_reps)
            incb = inc > 0
            for r in out:
                is_macro = r["class"] == M.MACRO
                if (A.encoder, A.method, A.name, A.budget, handling) == HEADLINE_RER_CELL and is_macro:
                    role = "headline_prenamed_cell"
                elif not is_macro:
                    role = "group_table_only"
                elif A.budget == PRIMARY_BUDGET:
                    role = "macro_table" if handling == "miss" else "macro_table_secondary_handling"
                else:
                    role = "macro_table_sensitivity_budget5"
                rows.append((A.encoder, A.method, A.name, A.budget, RER_RULE, handling, cid, r["class"], role,
                             r["f1_all"], *r["f1_all_ci"], r["f1_filt"], *r["f1_filt_ci"], r["delta_f1"],
                             *r["delta_ci"], r["rer"], r["rer_ci_low"], r["rer_ci_high"], r["rer_ci_status"],
                             r["n_undefined_reps"], r["n_zero_support_reps"], r["support"], int(inc.sum()),
                             int((nonret_f & incb).sum()), int((nonret_a & incb).sum()), M.QUANTILE_METHOD))
    meta = {"rule": RER_RULE, "handlings": list(RER_HANDLINGS), "primary_handling": "miss",
            "budgets": list(budgets), "headline_cell": list(HEADLINE_RER_CELL),
            "weights": "query weight = its patient's multiplicity in the shared draw matrix (all modes)",
            "undefined_threshold": M.UNDEFINED_THRESHOLD, "frozen": rer.info}
    return rows, meta


# =====================================================================================
# Wording classes and claim sentences (section 6, R5)
# =====================================================================================
def interval_direction(ci_low, ci_high, ci_status) -> str:
    if ci_status != "estimable" or ci_low is None or ci_high is None:
        return "not_estimable"
    if ci_low > 0:
        return "favour"
    if ci_high < 0:
        return "against"
    return "includes_0"


def wording_class(hit_dir: str, prec_dir: str, k: int = PRIMARY_K) -> Tuple[str, str]:
    h, p = f"hit@{k}", f"precision@{k}"
    if "not_estimable" in (hit_dir, prec_dir):
        return "not_estimable", "an interval was not estimable; no wording class assigned"
    if "against" in (hit_dir, prec_dir):
        if hit_dir == "against" and prec_dir == "against":
            return "v", f"decreased {h} and {p}"
        if hit_dir == "against" and prec_dir == "favour":
            return "v_mixed", f"increased {p}; decreased {h}"
        if prec_dir == "against" and hit_dir == "favour":
            return "v_mixed", f"increased {h}; decreased {p}"
        if hit_dir == "against":
            return "v", f"decreased {h}; a change in {p} was not established"
        return "v", f"decreased {p}; a change in {h} was not established"
    if hit_dir == "favour" and prec_dir == "favour":
        return "i", f"increased {h} and {p}"
    if prec_dir == "favour":
        return "ii", f"increased {p}; a change in {h} was not established"
    if hit_dir == "favour":
        return "iii", f"increased {h}; a change in {p} was not established"
    return "iv", "no change established"


def _f(x) -> str:
    return "NA" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.4f}"


def wording_tables(con_rows, cfg: EvalConfig, analysis: str, rer_rows) -> Tuple[list, List[str]]:
    H = {c: i for i, c in enumerate(EST_HEADER)}
    sel = {}
    for r in con_rows:
        if (r[H["family"]] == "a_filtered_minus_all_tissue" and r[H["level"]] == "macro"
                and r[H["population"]] == "both_returned" and r[H["k"]] == PRIMARY_K
                and r[H["endpoint"]] in ("hit", "precision") and r[H["stratum"]] == STRATUM_ALL):
            sel[(r[H["encoder"]], r[H["method"]], r[H["arm"]], r[H["budget"]], r[H["endpoint"]])] = r
    rows, by_enc = [], OrderedDict()
    disp = {e["encoder"]: e["display_name"] for e in cfg.encoders}
    order = [(a.encoder, a.method, a.name) for a in cfg.arms if a.name != BASELINE_ARM]
    for budget, role in ((PRIMARY_BUDGET, "primary"), (SENSITIVITY_BUDGET, "sensitivity")):
        for enc, m, arm in dict.fromkeys(order):
            hr, pr = sel.get((enc, m, arm, budget, "hit")), sel.get((enc, m, arm, budget, "precision"))
            if hr is None or pr is None:
                continue
            hd = interval_direction(hr[H["ci_low"]], hr[H["ci_high"]], hr[H["ci_status"]])
            pd_ = interval_direction(pr[H["ci_low"]], pr[H["ci_high"]], pr[H["ci_status"]])
            cls, text = wording_class(hd, pd_)
            rows.append((analysis, enc, m, arm, budget, role, PRIMARY_K, "conditional_macro_both_returned",
                         hr[H["point"]], hr[H["ci_low"]], hr[H["ci_high"]], hr[H["ci_status"]], hd,
                         pr[H["point"]], pr[H["ci_low"]], pr[H["ci_high"]], pr[H["ci_status"]], pd_,
                         cls, text))
            by_enc.setdefault((budget, role, enc), []).append(
                f"{DEFAULT_DISPLAY['method'].get(m, m)} {DEFAULT_DISPLAY['arm'].get(arm, arm)}, class ({cls}): "
                f"{text} [hit@{PRIMARY_K} {_f(hr[H['point']])} ({_f(hr[H['ci_low']])}, {_f(hr[H['ci_high']])}); "
                f"precision@{PRIMARY_K} {_f(pr[H['point']])} ({_f(pr[H['ci_low']])}, {_f(pr[H['ci_high']])})]")
    sentences = [f"# analysis: {analysis}. One sentence per encoder; classes are mechanical (contract "
                 f"section 6, amendment R5); encoders are not compared and no cell is selected."]
    for (budget, role, enc), parts in by_enc.items():
        lead = "" if role == "primary" else "Sensitivity (5 % label budget, not the primary result): "
        sentences.append(
            f"{lead}{disp.get(enc, enc)}, {budget} % label budget, tumour-filtered minus all-tissue, "
            f"conditional macro over the diagnostic groups at k = {PRIMARY_K} with whole-patient 95 % "
            f"intervals: " + "; ".join(parts) + ".")
    R = {c: i for i, c in enumerate(RER_HEADER)}
    head = [r for r in rer_rows if r[R["cell_role"]] == "headline_prenamed_cell"]
    if head:
        r = head[0]
        if r[R["rer_ci_status"]] == "reported":
            val = (f"RER {r[R['rer']]:+.4f} (95 % interval {r[R['rer_ci_low']]:+.4f} to "
                   f"{r[R['rer_ci_high']]:+.4f})")
        else:
            val = f"RER {r[R['rer']]!r}; interval status {r[R['rer_ci_status']]}"
        sentences.append(
            f"RER headline (the only RER headline permitted; pre-named cell UNI2 + Optimal Transport, "
            f"both-sided, 100 % budget, overall macro top-{PRIMARY_K} plurality-vote F1, non-returns scored "
            f"as misses): {val}. The other macro cells are reported in rer_contrasts.csv only.")
    return rows, sentences


# =====================================================================================
# S6-c lists (R4)
# =====================================================================================
def s0_provenance(s0_dir: Path, pool_path: Path, qa_path: Path, population_path: Path) -> dict:
    """Bind the S6-c inputs to S0: pool/query_assignment sha == S0 COMPLETE outputs; population sha ==
    the S0 ``retrieval_population`` binding."""
    s0_dir = Path(s0_dir)
    comp_b = (s0_dir / "COMPLETE.json").read_bytes()
    comp = json.loads(comp_b)
    if comp.get("status") != "COMPLETE":
        raise EvalError(f"{s0_dir}/COMPLETE.json status {comp.get('status')!r}")
    out = comp["outputs"]
    for name, path in (("pool.csv", pool_path), ("query_assignment.csv", qa_path)):
        verify_sha(path, out[name]["sha256"], f"S0 {name}")
    verify_sha(s0_dir / "input_bindings.json", out["input_bindings.json"]["sha256"], "S0 input_bindings.json")
    bind = json.loads((s0_dir / "input_bindings.json").read_bytes())
    pop_sha = [f["sha256"] for f in bind["files"] if f.get("role") == "retrieval_population"]
    if len(pop_sha) != 1:
        raise EvalError("S0 bindings lack a unique retrieval_population")
    verify_sha(population_path, pop_sha[0], "retrieval population (S0 binding)")
    return {"s0_dir": str(s0_dir), "COMPLETE.json_sha256": sha256_bytes(comp_b),
            "input_bindings.json_sha256": out["input_bindings.json"]["sha256"],
            "pool.csv_sha256": out["pool.csv"]["sha256"],
            "query_assignment.csv_sha256": out["query_assignment.csv"]["sha256"],
            "retrieval_population_sha256": pop_sha[0]}


def make_s6c_lists(pool_path: Path, qa_path: Path, population_path: Path, out_dir: Optional[Path],
                   expect: Dict[str, int] = S6C_EXPECTED, s0_dir: Optional[Path] = None) -> dict:
    prov = s0_provenance(s0_dir, pool_path, qa_path, population_path) if s0_dir else None
    pop = fe.load_population(population_path)
    pool = pd.read_csv(pool_path, dtype=str, keep_default_na=False)
    need = {"slide_id", "in_pool", "in_final_library", "UID.Subject"}
    if not need.issubset(pool.columns):
        raise EvalError(f"{pool_path}: lacks {sorted(need - set(pool.columns))}")
    qa = load_query_assignment(qa_path, pop)
    popset = set(pop.slide_id)
    labelled = pool[pool.in_pool == "1"]
    primary = sorted(set(labelled.slide_id) & popset, key=lambda s: (len(s), s))
    lib_labelled = sorted(set(labelled[labelled.in_final_library == "1"].slide_id) & popset,
                          key=lambda s: (len(s), s))
    if primary != lib_labelled:
        raise EvalError("labelled population slides differ from labelled final-library population slides")
    ann_patients = set(labelled["UID.Subject"])
    secondary = pop.slide_id[pop.patient_id.isin(ann_patients)].tolist()
    sec_qa = qa.slide_id[qa.patient_annotated == "1"].tolist()
    if sorted(secondary) != sorted(sec_qa):
        raise EvalError("slides of annotated patients (pool UID.Subject) differ from query_assignment "
                        "patient_annotated == 1")
    counts = {"primary": len(primary), "secondary": len(secondary)}
    rec = {"counts": counts, "expected": dict(expect),
           "definitions": {
               "primary": "pool.csv in_pool == 1 (labelled) slides that are members of the population "
                          "(identical to labelled in_final_library == 1 members)",
               "secondary": "population slides whose patient_id is the UID.Subject of a labelled pool slide "
                            "(identical to query_assignment patient_annotated == 1)"},
           "inputs": {str(p): fe.sha256_file(Path(p)) for p in (pool_path, qa_path, population_path)},
           "s0_provenance": prov,
           "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "argv": sys.argv,
           "code_sha256": {str(Path(__file__).resolve()): fe.sha256_file(Path(__file__).resolve())}}
    if counts != dict(expect):
        rec["status"] = "STOP"
        return rec
    rec["status"] = "OK"
    if out_dir is not None:
        out_dir = refuse_nonempty(out_dir)
        files: dict = {}
        _put(out_dir, files, "s6c_primary_annotated_population_slides.txt", ("\n".join(primary) + "\n").encode())
        _put(out_dir, files, "s6c_secondary_annotated_patient_slides.txt", ("\n".join(secondary) + "\n").encode())
        rec["outputs"] = dict(files)
        _put(out_dir, files, "s6c_lists.json", json.dumps(rec, indent=1).encode())
    return rec


# =====================================================================================
# F4 regression on the accepted route-b inputs
# =====================================================================================
_FLOATS = ("point", "ci_low", "ci_high")


def _compare_frames(new: pd.DataFrame, old: pd.DataFrame, float_cols: Sequence[str], tol: float) -> dict:
    """Positional comparison of two string-typed frames on the old frame's columns."""
    res = {"rows_new": len(new), "rows_old": len(old), "columns": list(old.columns)}
    if len(new) != len(old):
        res.update(ok=False, reason="row count differs")
        return res
    missing = [c for c in old.columns if c not in new.columns]
    if missing:
        res.update(ok=False, reason=f"columns missing in new: {missing}")
        return res
    new = new[list(old.columns)].reset_index(drop=True)
    old = old.reset_index(drop=True)
    str_cols = [c for c in old.columns if c not in float_cols]
    mism = {c: int((new[c] != old[c]).sum()) for c in str_cols}
    maxd, nan_mis = {}, {}
    for c in float_cols:
        a = pd.to_numeric(new[c].replace("", np.nan), errors="raise").to_numpy(dtype=float)
        b = pd.to_numeric(old[c].replace("", np.nan), errors="raise").to_numpy(dtype=float)
        nan_mis[c] = int((np.isnan(a) != np.isnan(b)).sum())
        ok = ~np.isnan(a) & ~np.isnan(b)
        maxd[c] = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0
    res.update(string_mismatches={k: v for k, v in mism.items() if v},
               nan_pattern_mismatches={k: v for k, v in nan_mis.items() if v}, max_abs_diff=maxd,
               ok=not any(mism.values()) and not any(nan_mis.values()) and all(v <= tol for v in maxd.values()))
    return res


def _read_str(path: Path, compression=None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, compression=compression)


def _compare_npy(new: Path, old: Path, tol: float) -> dict:
    a = np.load(new, allow_pickle=False)
    b = np.load(old, allow_pickle=False)
    if a.shape != b.shape:
        return {"ok": False, "shape_new": list(a.shape), "shape_old": list(b.shape)}
    nan_mis = int((np.isnan(a) != np.isnan(b)).sum())
    ok = ~np.isnan(a) & ~np.isnan(b)
    md = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0
    return {"ok": nan_mis == 0 and md <= tol, "shape": list(a.shape), "nan_pattern_mismatches": nan_mis,
            "max_abs_diff": md}


def _project_estimates(path: Path) -> pd.DataFrame:
    df = _read_str(path)
    if "stratum" in df.columns:
        df = df[df.stratum == STRATUM_ALL]
    return df


def _shared_mode_comparison(new: pd.DataFrame, old: pd.DataFrame, full_patients: int, tol: float) -> dict:
    """Shared-draw output vs route-b: points everywhere; intervals where the population is complete."""
    keys = ["row_kind", "arm", "method", "budget", "arm_a", "arm_b", "endpoint", "k", "population"]
    new = new[list(old.columns)].reset_index(drop=True)
    old = old.reset_index(drop=True)
    same_keys = all((new[c] == old[c]).all() for c in keys + ["level", "group", "column"])
    if not same_keys or len(new) != len(old):
        return {"ok": False, "reason": "row keys differ"}
    macro = old[old.level == "macro"].set_index(keys).n_patients
    pop_pat = old.set_index(keys).index.map(lambda k: int(macro.loc[k]))
    full = np.asarray(pop_pat) == full_patients
    out = {"rows": len(old), "rows_full_population": int(full.sum()),
           "rows_subset_population": int((~full).sum()),
           "subset_population_patient_counts": sorted(set(np.asarray(pop_pat)[~full].tolist()))}
    pt = _compare_frames(new, old, list(_FLOATS), tol)
    a = pd.to_numeric(new.point.replace("", np.nan)).to_numpy(float)
    b = pd.to_numeric(old.point.replace("", np.nan)).to_numpy(float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    out["point_max_abs_diff_all_rows"] = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0
    out["point_nan_pattern_mismatches"] = int((np.isnan(a) != np.isnan(b)).sum())
    fullc = _compare_frames(new[full], old[full], list(_FLOATS), tol)
    out["full_population_rows"] = {k: fullc[k] for k in ("ok", "max_abs_diff", "string_mismatches",
                                                         "nan_pattern_mismatches")}
    sub_n, sub_o = new[~full], old[~full]
    dif = {}
    for c in ("ci_low", "ci_high"):
        x = pd.to_numeric(sub_n[c].replace("", np.nan)).to_numpy(float)
        y = pd.to_numeric(sub_o[c].replace("", np.nan)).to_numpy(float)
        okc = ~np.isnan(x) & ~np.isnan(y)
        d = np.abs(x[okc] - y[okc])
        dif[c] = {"max_abs_diff": float(d.max()) if d.size else 0.0,
                  "n_differ_gt_tol": int((d > tol).sum())}
    out["subset_population_interval_differences"] = dif
    other = [c for c in old.columns if c not in ("ci_low", "ci_high", "point", "ci_status", "ci_reason",
                                                 "zero_support_replicates")]
    out["subset_rows_non_interval_fields_identical"] = bool(all((sub_n[c] == sub_o[c]).all() for c in other))
    out["ok_points_and_full_population_rows"] = bool(
        out["point_max_abs_diff_all_rows"] <= tol and out["point_nan_pattern_mismatches"] == 0
        and fullc["ok"] and out["subset_rows_non_interval_fields_identical"])
    out["string_mismatches_all_rows"] = pt.get("string_mismatches")
    return out


def f4_regression(route_b: Path, out_dir: Path, *, frozen_rer: Optional[Path], tol: float = 1e-12,
                  rer_run: Optional[Path] = None) -> dict:
    t0 = time.time()
    route_b = Path(route_b)
    ev = route_b / "evaluation_qf"
    out_dir = refuse_nonempty(out_dir)
    sums = {}
    for line in (ev / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split(None, 1)
        sums[name.strip()] = h
    checks = {}
    bad = {n: h for n, h in sums.items() if fe.sha256_file(ev / n) != h}
    checks["route_b_inputs_rehash"] = {"n": len(sums), "mismatches": bad, "ok": not bad}
    if bad:
        raise EvalError(f"route-b evaluation files differ from SHA256SUMS.txt: {sorted(bad)}")
    summ = json.loads((ev / "summary.json").read_bytes())
    pop_path = Path(summ["population"]["path"])
    verify_sha(pop_path, summ["population"]["sha256"], "route-b population")
    pop = fe.load_population(pop_path)
    banks_root = None
    for i, a in enumerate(summ["argv"]):
        if a == "--banks-root":
            banks_root = Path(summ["argv"][i + 1])
    if banks_root is None:
        raise EvalError("route-b summary argv lacks --banks-root")
    bank_bad = {p: h for p, h in summ["bank_summaries_sha256"].items() if fe.sha256_file(Path(p)) != h}
    checks["route_b_bank_summaries_rehash"] = {"n": len(summ["bank_summaries_sha256"]),
                                               "mismatches": bank_bad, "ok": not bank_bad}
    if bank_bad:
        raise EvalError("route-b bank summaries differ from summary.json")
    # 1. one external draw matrix
    dres = make_shared_draws(pop, out_dir / "draws", population_path=pop_path)
    new_d = np.load(out_dir / "draws" / "patient_draws.npy", allow_pickle=False)
    old_d = np.load(ev / "patient_draws.npy", allow_pickle=False)
    checks["shared_draws_equal_route_b_patient_draws"] = {
        "ok": bool(new_d.dtype == old_d.dtype and np.array_equal(new_d, old_d)
                   and (out_dir / "draws" / "patient_draws.npy").read_bytes() == (ev / "patient_draws.npy").read_bytes()),
        "shape": list(new_d.shape), "sha256": dres["files"]["patient_draws.npy"]["sha256"]}
    # 2. route-b design as S4 eligibility tables
    elig = {}
    for b in BUDGETS:
        df = eligibility_from_route_b(pop, banks_root, b)
        elig[b] = write_eligibility_table(df, out_dir / "eligibility" / "uni2" / f"b{b}.csv.gz")
    lib_versions = {b: sorted(set(pd.read_csv(elig[b]["path"], dtype=str, keep_default_na=False)
                                  .loc[lambda d: d.arm == "both_sided", "library_version"])) for b in BUDGETS}
    # 3. config
    arms = []
    for a in summ["arms"]:
        name, method, budget = a["key"].split("|")
        ent = {"arm": name, "method": method, "budget": budget, "matrix": a["matrix"],
               "matrix_sha256": a["matrix_sha256"]}
        if a.get("rows"):
            ent.update(rows=a["rows"], rows_sha256=a["rows_sha256"])
        arms.append(ent)
    cfg_obj = {"schema": CONFIG_SCHEMA, "note": "F4: accepted route-b inputs in the TF-2 evaluator layout",
               "population": {"path": str(pop_path), "sha256": summ["population"]["sha256"]},
               "query_assignment": None, "draws_dir": str(out_dir / "draws"),
               "encoders": [{"encoder": "uni2", "display_name": "UNI2",
                             "eligibility": {b: {"path": elig[b]["path"], "sha256": elig[b]["sha256"]}
                                             for b in BUDGETS},
                             "arms": arms}]}
    cfg_path = out_dir / "config.json"
    with open(cfg_path, "x") as fh:
        fh.write(json.dumps(cfg_obj, indent=1))
    cfg = load_config(cfg_path)
    rer = load_rer_core(frozen_rer) if frozen_rer else None
    runs = {}
    for mode in DRAW_MODES[::-1]:            # per_population first, then shared
        runs[mode] = evaluate(cfg, draw_mode=mode, rer=rer, analysis=f"f4_{mode}",
                              out_dir=out_dir / f"eval_{mode}")
    pp = out_dir / "eval_per_population"
    comp = {}
    comp["estimates.csv"] = _compare_frames(_project_estimates(pp / "estimates.csv"), _read_str(ev / "estimates.csv"),
                                            _FLOATS, tol)
    comp["contrasts.csv"] = _compare_frames(_project_estimates(pp / "contrasts.csv"), _read_str(ev / "contrasts.csv"),
                                            _FLOATS, tol)
    comp["tie_sensitivity.csv"] = _compare_frames(_project_estimates(pp / "tie_sensitivity.csv"),
                                                  _read_str(ev / "tie_sensitivity.csv"), _FLOATS, tol)
    comp["tie_counts.csv"] = _compare_frames(_read_str(pp / "tie_counts.csv"), _read_str(ev / "tie_counts.csv"), (), tol)
    cov_new = _read_str(pp / "coverage.csv")
    comp["coverage.csv"] = _compare_frames(cov_new[cov_new.stratum == STRATUM_ALL], _read_str(ev / "coverage.csv"),
                                           ("coverage",) + tuple(f"slide_mean_hit_yield_at_{k}" for k in CUTOFFS), tol)
    qn = _read_str(pp / "query_metrics.csv.gz", "gzip")
    qo = _read_str(ev / "query_metrics.csv.gz", "gzip")
    qcols = [c for c in qo.columns if c not in ("query_fold", "library_id")]
    comp["query_metrics.csv.gz"] = _compare_frames(qn, qo[qcols], ("conditional_hit", "conditional_precision",
                                                                    "tie_averaged_hit", "tie_averaged_precision"), tol)
    comp["query_metrics.csv.gz"]["columns_not_compared"] = ["query_fold", "library_id (new: library_version)"]
    comp["ranked_lists.csv.gz"] = _compare_frames(_read_str(pp / "ranked_lists.csv.gz", "gzip"),
                                                  _read_str(ev / "ranked_lists.csv.gz", "gzip"), ("distance",), tol)
    comp["replicates.npy"] = _compare_npy(pp / "replicates.npy", ev / "replicates.npy", tol)
    comp["tie_averaged_replicates.npy"] = _compare_npy(pp / "tie_averaged_replicates.npy",
                                                       ev / "tie_averaged_replicates.npy", tol)
    sh = out_dir / "eval_shared"
    shared = {
        "estimates.csv": _shared_mode_comparison(_project_estimates(sh / "estimates.csv"),
                                                 _read_str(ev / "estimates.csv"), len(new_d[0]), tol),
        "contrasts.csv": _shared_mode_comparison(_project_estimates(sh / "contrasts.csv"),
                                                 _read_str(ev / "contrasts.csv"), len(new_d[0]), tol)}
    rer_cmp = None
    if rer is not None and rer_run is not None:
        old = pd.read_csv(Path(rer_run) / "contrasts.csv", dtype=str, keep_default_na=False)
        old = old[old.rule == RER_RULE].copy()
        old["budget"] = old.budget.map(lambda s: str(int(float(s))))
        rer_cmp = {}
        for mode in DRAW_MODES:
            new = _read_str(out_dir / f"eval_{mode}" / "rer_contrasts.csv")
            k = ["method", "arm", "budget", "handling", "class"]
            m = new.merge(old, on=k, suffixes=("", "_old"), how="outer", indicator=True)
            unmatched = int((m["_merge"] != "both").sum())
            m = m[m["_merge"] == "both"]
            fl = ["f1_all", "f1_filt", "delta_f1", "delta_ci_low", "delta_ci_high", "rer", "rer_ci_low",
                  "rer_ci_high", "support"]
            md = {}
            for c in fl:
                a = pd.to_numeric(m[c].replace("", np.nan)).to_numpy(float)
                b = pd.to_numeric(m[c + "_old"].replace("", np.nan)).to_numpy(float)
                okc = ~np.isnan(a) & ~np.isnan(b)
                md[c] = float(np.max(np.abs(a[okc] - b[okc]))) if okc.any() else 0.0
                md[c + "_nan_mismatch"] = int((np.isnan(a) != np.isnan(b)).sum())
            ints = ["rer_ci_status", "n_undefined_reps", "n_zero_support_reps", "n_queries", "n_nonreturned_filt"]
            im = {c: int((m[c].map(lambda s: s.replace(".0", "")) != m[c + "_old"].map(
                lambda s: s.replace(".0", ""))).sum()) for c in ints}
            rer_cmp[mode] = {"rows_new": len(new), "rows_old_plurality": len(old), "unmatched": unmatched,
                             "max_abs_diff": md, "field_mismatches": im,
                             "ok": unmatched == 0 and all(v <= tol for kk, v in md.items() if not kk.endswith("mismatch"))
                             and not any(v for kk, v in md.items() if kk.endswith("mismatch")) and not any(im.values())}
    primary_ok = (checks["route_b_inputs_rehash"]["ok"] and checks["shared_draws_equal_route_b_patient_draws"]["ok"]
                  and all(v["ok"] for v in comp.values()))
    shared_ok = all(v["ok_points_and_full_population_rows"] for v in shared.values())
    rer_ok = None if rer_cmp is None else all(v["ok"] for v in rer_cmp.values())
    status = "PASS" if primary_ok and shared_ok and rer_ok in (None, True) else "FAIL"
    res = {"status": status, "tolerance": tol,
           "definition": "F4 (contract section 7 + amendment R2): the new TF-2 evaluator code path (per-query "
                         "eligibility tables, config bindings, external draw matrix) on the accepted route-b inputs "
                         "reproduces the accepted route-b estimates and contrasts to 1e-12. Primary check = "
                         "draw_mode per_population (the draw rule the accepted outputs were produced under). "
                         "draw_mode shared (the TF-2 F10 rule) must reproduce every point estimate and every row "
                         "whose summarized population covers all patients; its intervals for populations missing "
                         "patients (non-returned queries) differ by construction and are reported, not gated.",
           "route_b_root": str(route_b),
           "library_versions_both_sided": lib_versions,
           "single_shared_version_arms": "all_tissue and query_only (one version each); both_filtered uses the "
                                         "five route-b fold versions per budget",
           "checks": checks, "per_population_mode": comp, "shared_mode": shared, "rer_vs_rer_run_002": rer_cmp,
           "primary_ok": primary_ok, "shared_ok": shared_ok, "rer_ok": rer_ok,
           "eligibility_tables": elig, "config_sha256": fe.sha256_file(cfg_path),
           "eval_dirs": {m: runs[m]["out_dir"] for m in runs},
           "code_sha256": {str(p): fe.sha256_file(p) for p in (Path(__file__).resolve(), Path(fe.__file__).resolve(),
                                                               Path(kernel.__file__).resolve())},
           "rer_core": rer.info if rer else None,
           "host": socket.gethostname(), "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "wall_seconds": round(time.time() - t0, 1), "cpu_seconds_process": round(time.process_time(), 1)}
    with open(out_dir / "f4_regression.json", "x") as fh:
        fh.write(json.dumps(res, indent=1, default=str))
    return res


# =====================================================================================
# F4b (non-gating control): TITAN all-tissue arm vs the accepted R2.6 common-pool estimates
# =====================================================================================
R26_DEFAULT = "r2_6_titan_common_pool_l1_repaired_arm1_20261004"


def f4b_titan(r26_dir: Path, out_dir: Path, *, tol: float = 1e-12) -> dict:
    """Run the accepted R2.6 all-tissue TITAN arm (arm 1, repaired L1 matrix) through this evaluator.

    Population = the R2.6 common pool (queries = library, in the R2.6 library order, so stable
    tie-breaking is the same); draws = the external matrix for its patients (R2.6 used a verbatim
    copy of ``kernel.make_patient_draws``, seed 123, B = 5000). Compared: ranked lists, per-query
    hit@1/3/5/10 and precision@5, macro hit@k (point and interval), micro hit@k (point; this
    evaluator reports the slide-pooled mean in coverage.csv, without an interval) and per-class
    hit@5 (point and interval). Non-gating control (design decision Q7)."""
    t0 = time.time()
    r26_dir = Path(r26_dir)
    out_dir = refuse_nonempty(out_dir)
    rec = json.loads((r26_dir / "reconciliation.json").read_bytes())
    m = rec["inputs"]["arm1_matrix"]
    matrix = Path(m["path"])
    verify_sha(matrix, m["sha256"], "R2.6 arm-1 matrix")
    rl_old = _read_str(r26_dir / "ranked_lists_common_pool_arm1.csv.gz", "gzip")
    pq_old = _read_str(r26_dir / "per_query_common_pool.csv")
    queries = list(dict.fromkeys(rl_old["query"].tolist()))
    first = rl_old.drop_duplicates("query").set_index("query")
    pat = pq_old.set_index("query").patient
    pop_df = pd.DataFrame({"query_index": range(len(queries)), "reference_index": range(len(queries)),
                           "slide_id": queries, "patient_id": [pat[q] for q in queries],
                           "diagnostic_group": [first.true_label[q] for q in queries]})
    pop_path = out_dir / "population_r26_common_pool.csv"
    with open(pop_path, "x") as fh:
        pop_df.to_csv(fh, index=False, lineterminator="\n")
    pop = fe.load_population(pop_path)
    make_shared_draws(pop, out_dir / "draws", population_path=pop_path)
    cfg_obj = {"schema": CONFIG_SCHEMA, "note": "F4b: accepted R2.6 TITAN arm 1 (all-tissue, L1, repaired)",
               "population": {"path": str(pop_path), "sha256": fe.sha256_file(pop_path)},
               "query_assignment": None, "draws_dir": str(out_dir / "draws"),
               "encoders": [{"encoder": "titan", "arms": [
                   {"arm": BASELINE_ARM, "method": "titan", "budget": "", "matrix": str(matrix),
                    "matrix_sha256": m["sha256"]}]}]}
    cfg_path = out_dir / "config.json"
    with open(cfg_path, "x") as fh:
        fh.write(json.dumps(cfg_obj, indent=1))
    cfg = load_config(cfg_path, require_complete=False)
    ev_dir = out_dir / "eval_shared"
    evaluate(cfg, draw_mode="shared", rer=None, analysis="f4b_titan", out_dir=ev_dir)
    comp = {}
    rl_new = _read_str(ev_dir / "ranked_lists.csv.gz", "gzip")
    rl_new = rl_new.rename(columns={"neighbour_group": "neighbour_label"})
    rl_new["true_label"] = rl_new["query"].map(dict(zip(pop.slide_id, pop.diagnostic_group)))
    comp["ranked_lists"] = _compare_frames(rl_new, rl_old[["query", "true_label", "rank", "neighbour",
                                                           "neighbour_label", "distance"]], ("distance",), tol)
    qm = _read_str(ev_dir / "query_metrics.csv.gz", "gzip")
    per_q = {}
    for k in CUTOFFS:
        a = qm[qm.k == str(k)].set_index("query").conditional_hit.astype(float).reindex(queries).to_numpy()
        b = pq_old.set_index("query")[f"arm1_hit@{k}"].astype(float).reindex(queries).to_numpy()
        per_q[f"hit@{k}"] = float(np.max(np.abs(a - b)))
    a = qm[qm.k == "5"].set_index("query").conditional_precision.astype(float).reindex(queries).to_numpy()
    b = pq_old.set_index("query")["arm1_precision@5"].astype(float).reindex(queries).to_numpy()
    per_q["precision@5"] = float(np.max(np.abs(a - b)))
    comp["per_query"] = {"max_abs_diff": per_q, "ok": all(v <= tol for v in per_q.values())}
    est = _read_str(ev_dir / "estimates.csv")
    cov = _read_str(ev_dir / "coverage.csv")
    sm = _read_str(r26_dir / "summary_common_pool.csv")
    pop_name = sorted(set(sm.population) - {"legacy_population_lib1_2604_lib2_2564"})
    main_pop = "common_pool_2564" if "common_pool_2564" in pop_name else pop_name[0]
    sm = sm[(sm.population == main_pop) & (sm.arm == "arm1")]
    rows = []
    for k in CUTOFFS:
        o = sm[sm.metric == f"macro_hit@{k}"].iloc[0]
        n = est[(est.endpoint == "hit") & (est.k == str(k)) & (est.level == "macro")
                & (est.population == "returned")].iloc[0]
        rows.append({"metric": f"macro_hit@{k}", "new": [n.point, n.ci_low, n.ci_high, n.ci_status],
                     "old": [o.estimate, o.lo, o.hi, o.ci_available],
                     "max_abs_diff": max(abs(float(n.point) - float(o.estimate)), abs(float(n.ci_low) - float(o.lo)),
                                         abs(float(n.ci_high) - float(o.hi)))})
        o = sm[sm.metric == f"micro_hit@{k}"].iloc[0]
        c = cov[(cov.arm == BASELINE_ARM) & (cov.group == "ALL") & (cov.stratum == STRATUM_ALL)].iloc[0]
        rows.append({"metric": f"micro_hit@{k} (point only)", "new": [c[f"slide_mean_hit_yield_at_{k}"]],
                     "old": [o.estimate], "max_abs_diff": abs(float(c[f"slide_mean_hit_yield_at_{k}"]) - float(o.estimate))})
    comp["summary_arm1"] = {"rows": rows, "ok": all(r["max_abs_diff"] <= tol for r in rows)}
    pc = _read_str(r26_dir / "per_class_common_pool.csv")
    pc = pc[(pc.population == main_pop) & (pc.arm == "arm1") & (pc.metric == "hit@5")]
    crow, cmax = [], 0.0
    for r in pc.itertuples(index=False):
        n = est[(est.endpoint == "hit") & (est.k == "5") & (est.level == "group") & (est.group == r[pc.columns.get_loc("class")])
                & (est.population == "returned")].iloc[0]
        avail_new, avail_old = n.ci_status == "estimable", str(r.ci_available) == "True"
        d = abs(float(n.point) - float(r.estimate))
        if avail_new and avail_old:
            d = max(d, abs(float(n.ci_low) - float(r.lo)), abs(float(n.ci_high) - float(r.hi)))
        same_n = (int(n.n_queries) == int(r.n_class_queries) and int(n.n_patients) == int(r.n_class_patients)
                  and avail_new == avail_old)
        crow.append({"class": r[pc.columns.get_loc("class")], "max_abs_diff": d, "support_equal": same_n,
                     "ci_available": avail_new})
        cmax = max(cmax, d)
    comp["per_class_hit@5_arm1"] = {"n_classes": len(crow), "max_abs_diff": cmax, "rows": crow,
                                    "ok": cmax <= tol and all(x["support_equal"] for x in crow)}
    status = "PASS" if all(v["ok"] for v in comp.values()) else "FAIL"
    res = {"status": status, "gating": False, "tolerance": tol,
           "definition": "F4b (lead decision Q7, non-gating): the TITAN all-tissue arm (accepted R2.6 repaired L1 "
                         "matrix) through the TF-2 evaluator on the accepted R2.6 common-pool population reproduces "
                         "the accepted R2.6 arm-1 ranked lists, per-query metrics, macro hit@k (point + interval), "
                         "micro hit@k (point) and per-class hit@5 (point + interval).",
           "r26_dir": str(r26_dir), "r26_population": main_pop,
           "population": {"queries": len(pop), "patients": int(pop.patient_id.nunique()),
                          "groups": int(pop.diagnostic_group.nunique()), "sha256": fe.sha256_file(pop_path)},
           "matrix": {"path": str(matrix), "sha256": m["sha256"]},
           "r26_inputs_sha256": {n: fe.sha256_file(r26_dir / n) for n in (
               "reconciliation.json", "ranked_lists_common_pool_arm1.csv.gz", "per_query_common_pool.csv",
               "summary_common_pool.csv", "per_class_common_pool.csv")},
           "comparisons": comp, "config_sha256": fe.sha256_file(cfg_path),
           "code_sha256": {str(p): fe.sha256_file(p) for p in (Path(__file__).resolve(), Path(fe.__file__).resolve(),
                                                               Path(kernel.__file__).resolve())},
           "host": socket.gethostname(), "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "wall_seconds": round(time.time() - t0, 1), "cpu_seconds_process": round(time.process_time(), 1)}
    with open(out_dir / "f4b_titan.json", "x") as fh:
        fh.write(json.dumps(res, indent=1, default=str))
    return res


# =====================================================================================
# CLI
# =====================================================================================
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="r2_1_tf2_eval_cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("make-draws", help="write the one external whole-patient draw matrix")
    d.add_argument("--population", type=Path, required=True)
    d.add_argument("--out-dir", type=Path, required=True)
    d.add_argument("--count", type=int, default=DRAW_COUNT)
    d.add_argument("--seed", type=int, default=DRAW_SEED)
    s = sub.add_parser("make-s6c-lists", help="derive the S6-c removal lists from S0 (stops unless 168/364)")
    s.add_argument("--pool", type=Path, required=True)
    s.add_argument("--query-assignment", type=Path, required=True)
    s.add_argument("--population", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--s0-dir", type=Path, default=None, help="bind inputs to S0 COMPLETE/input_bindings")
    e = sub.add_parser("evaluate", help="S6 evaluation of every configured arm")
    e.add_argument("--config", type=Path, required=True)
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--draw-mode", choices=DRAW_MODES, default="shared")
    e.add_argument("--remove-reference-slides", type=Path, default=None,
                   help="S6-c: one slide id per line, removed from every arm's eligible references")
    e.add_argument("--analysis", default=None, help="label (default primary, or s6c:<file name>)")
    e.add_argument("--frozen-rer", type=Path, default=default_frozen_rer())
    e.add_argument("--vote-module", type=Path, default=None)
    e.add_argument("--no-rer", action="store_true")
    e.add_argument("--rer-budgets", nargs="+", default=list(BUDGETS), choices=list(BUDGETS))
    e.add_argument("--allow-used-sentinel", action="store_true")
    e.add_argument("--allow-partial", action="store_true", help="fixtures only: incomplete arm design")
    f = sub.add_parser("f4-regression", help="F4: route-b inputs through this evaluator")
    f.add_argument("--route-b-root", type=Path, required=True)
    f.add_argument("--out-dir", type=Path, required=True)
    f.add_argument("--frozen-rer", type=Path, default=default_frozen_rer())
    f.add_argument("--rer-run", type=Path, default=None, help="route-b rer_run_002 for the RER cross-check")
    f.add_argument("--tolerance", type=float, default=1e-12)
    g = sub.add_parser("f4b-titan", help="F4b (non-gating): TITAN all-tissue vs accepted R2.6 estimates")
    g.add_argument("--r26-dir", type=Path, default=(Path(os.environ["PROSPER_OUT"]) / R26_DEFAULT)
                   if os.environ.get("PROSPER_OUT") else None)
    g.add_argument("--out-dir", type=Path, required=True)
    g.add_argument("--tolerance", type=float, default=1e-12)
    a = p.parse_args(argv)
    if a.cmd == "f4b-titan":
        if a.r26_dir is None:
            p.error("--r26-dir required (or export PROSPER_OUT)")
        r = f4b_titan(a.r26_dir, a.out_dir, tol=a.tolerance)
        print(json.dumps({k: r[k] for k in ("status", "gating", "population", "wall_seconds")}, indent=1))
        return 0 if r["status"] == "PASS" else 1
    if a.cmd == "make-draws":
        pop = fe.load_population(a.population)
        r = make_shared_draws(pop, a.out_dir, population_path=a.population, count=a.count, seed=a.seed)
        print(json.dumps(r["record"], indent=1))
        return 0
    if a.cmd == "make-s6c-lists":
        r = make_s6c_lists(a.pool, a.query_assignment, a.population, a.out_dir, s0_dir=a.s0_dir)
        print(json.dumps(r, indent=1))
        return 0 if r["status"] == "OK" else 1
    if a.cmd == "evaluate":
        cfg = load_config(a.config, require_complete=not a.allow_partial)
        rer = None
        if not a.no_rer:
            if a.frozen_rer is None:
                p.error("--frozen-rer required (or export PROSPER_OUT), or pass --no-rer")
            rer = load_rer_core(a.frozen_rer, a.vote_module)
        label = a.analysis or (f"s6c:{a.remove_reference_slides.name}" if a.remove_reference_slides else "primary")
        r = evaluate(cfg, draw_mode=a.draw_mode, removed_path=a.remove_reference_slides,
                     allow_used_sentinel=a.allow_used_sentinel, rer=rer, rer_budgets=a.rer_budgets,
                     analysis=label, out_dir=a.out_dir)
        print(json.dumps({"out_dir": r["out_dir"], "files": r["files"]}, indent=1, default=str))
        return 0
    if a.cmd == "f4-regression":
        r = f4_regression(a.route_b_root, a.out_dir, frozen_rer=a.frozen_rer, tol=a.tolerance, rer_run=a.rer_run)
        print(json.dumps({k: r[k] for k in ("status", "primary_ok", "shared_ok", "rer_ok", "wall_seconds")}, indent=1))
        return 0 if r["status"] == "PASS" else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
