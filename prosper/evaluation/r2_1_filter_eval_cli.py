"""R2.1 F2: evaluate H-Optimus tumour-filtered (both-filtered) retrieval against all-tissue.

Population: the accepted 2,604-slide / 1,800-patient / 14-group ``retrieval_population.csv``;
queries and references are the same population. For every arm (a distance matrix plus an
optional F1a ``_filter_summary.csv`` saying which slides were written):

* a query whose slide was not written is a NON-RETURN (status ``empty_query``): kept in the
  coverage / hit-yield denominators, excluded from conditional ranking metrics;
* the reference library of an arm is the written population slides only;
* same-patient references are excluded by population ``patient_id`` identity (as the main
  R2.1 evaluator does), applied to matrices that are themselves in ``_no_same_patient`` form;
  a used cell equal to the 1e6 sentinel is counted and is an error unless explicitly allowed;
* ranking is ascending distance, ties broken stably by population (``reference_index``)
  order, which preserves the metadata order the main evaluator used.

Inference reuses the frozen kernel ``r2_1_hoptimus_main`` unchanged (imported):
``query_metrics`` per query and cutoff, ``make_patient_draws`` (PCG64 seed 123, 5,000 whole-
patient draws, one multiplicity per patient across all of that patient's slides),
``summarize_metric`` (slide-weighted group means, equal-weight macro, linear 2.5/97.5%
percentiles) and ``summarize_paired`` (A-minus-B on one shared membership and draw table).
As the kernel requires, every summarized population gets draws generated for exactly its
included patients (sorted), with the same seed: intended (all 2,604), an arm's returned
subset, or a contrast's both-returned intersection.

No p-values or winner statements are produced.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gzip
import hashlib
import io
import json
import os
import pickle
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import r2_1_hoptimus_main as kernel

CUTOFFS = (1, 3, 5, 10)
ENDPOINTS = ("hit", "precision")
SENTINEL = 1e6
BASELINE_ARM = "all_tissue"
ESTIMATE_HEADER = (
    "column", "row_kind", "arm", "method", "budget", "arm_a", "arm_b", "endpoint", "k",
    "level", "group", "population", "point", "point_status", "point_reason", "ci_low",
    "ci_high", "ci_status", "ci_reason", "n_queries", "n_patients", "zero_support_replicates")
QUERY_HEADER = (
    "arm", "method", "budget", "query_index", "query", "patient_id", "diagnostic_group", "k",
    "status", "returned_depth", "short_list", "conditional_hit", "conditional_precision",
    "conditional_reason", "coverage", "hit_yield", "n_candidates", "boundary_tie")
RANK_HEADER = ("arm", "method", "budget", "query", "rank", "neighbour", "neighbour_group",
               "relevant", "distance")
COVERAGE_HEADER = ("arm", "method", "budget", "level", "group", "intended", "returned",
                   "non_returns", "non_return_slides", "coverage",
                   *(f"hit_yield_at_{k}" for k in CUTOFFS))


class EvalError(RuntimeError):
    """An input violates the declared population/matrix specification."""


@dataclass(frozen=True)
class Arm:
    name: str
    method: str
    budget: str            # "" for the all-tissue baseline
    matrix: Path
    summary: Optional[Path]

    @property
    def key(self) -> str:
        return f"{self.name}|{self.method}|{self.budget}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_population(path: Path) -> pd.DataFrame:
    pop = pd.read_csv(path, dtype={"slide_id": str, "patient_id": str, "diagnostic_group": str})
    need = {"query_index", "reference_index", "slide_id", "patient_id", "diagnostic_group"}
    if not need.issubset(pop.columns):
        raise EvalError(f"population lacks {sorted(need - set(pop.columns))}")
    pop = pop.sort_values("reference_index", kind="mergesort").reset_index(drop=True)
    if not (pop.reference_index.to_numpy() == np.arange(len(pop))).all() \
            or not (pop.query_index.to_numpy() == pop.reference_index.to_numpy()).all():
        raise EvalError("population query/reference indices must both be 0..n-1 and equal")
    if pop.slide_id.duplicated().any():
        raise EvalError("duplicate population slide_id")
    for c in ("slide_id", "patient_id", "diagnostic_group"):
        if pop[c].isna().any() or (pop[c].str.strip() == "").any():
            raise EvalError(f"empty {c} in population")
    return pop


def returned_mask(arm: Arm, pop: pd.DataFrame) -> np.ndarray:
    if arm.summary is None:
        return np.ones(len(pop), dtype=bool)
    s = pd.read_csv(arm.summary, dtype={"slide_id": str})
    if s.slide_id.duplicated().any() or set(s.slide_id) != set(pop.slide_id):
        raise EvalError(f"{arm.summary}: slides differ from the population")
    if not set(s.written.unique()).issubset({0, 1}):
        raise EvalError(f"{arm.summary}: written must be 0/1")
    w = s.set_index("slide_id").written.reindex(pop.slide_id).to_numpy()
    return w.astype(bool)


def load_matrix(arm: Arm, pop: pd.DataFrame, returned: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Population x population float64 array (rows queries, columns references).

    Rows/columns of slides that are not returned are NaN and never read."""
    with open(arm.matrix, "rb") as fh:
        df = pickle.load(fh)
    if not isinstance(df, pd.DataFrame):
        raise EvalError(f"{arm.matrix}: not a DataFrame")
    rows = [str(i) for i in df.index]
    cols = [str(c) for c in df.columns]
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
        raise EvalError(f"{arm.matrix}: duplicate axis ids")
    want = set(pop.slide_id[returned])
    if arm.summary is not None:
        if set(rows) != want or set(cols) != want:
            raise EvalError(f"{arm.matrix}: axes ({len(rows)}x{len(cols)}) differ from the "
                            f"{len(want)} written slides of {arm.summary}")
    elif not (want.issubset(rows) and want.issubset(cols)):
        raise EvalError(f"{arm.matrix}: population slides missing from the axes")
    values = df.to_numpy()
    if values.dtype.kind != "f":
        raise EvalError(f"{arm.matrix}: matrix must be floating point")
    ri = {s: i for i, s in enumerate(rows)}
    ci = {s: i for i, s in enumerate(cols)}
    ids = pop.slide_id.tolist()
    rsel = np.array([ri.get(s, -1) for s in ids])
    csel = np.array([ci.get(s, -1) for s in ids])
    out = np.full((len(ids), len(ids)), np.nan)
    ok = returned
    out[np.ix_(ok, ok)] = values[np.ix_(rsel[ok], csel[ok])]
    diag = {
        "matrix_shape": list(values.shape),
        "extra_axis_ids_outside_population": len(set(rows) - set(ids)),
    }
    return out, diag


def rank_arm(pop: pd.DataFrame, dist: np.ndarray, returned: np.ndarray,
             allow_used_sentinel: bool = False):
    """Per-query relevance of the top-10 eligible references and the diagnostics."""
    groups = pop.diagnostic_group.to_numpy(dtype=object)
    patients = pop.patient_id.to_numpy(dtype=object)
    ids = pop.slide_id.to_numpy(dtype=object)
    statuses, relevance, ranks, n_cand, ties = [], [], [], [], []
    diag = {"used_pairs": 0, "excluded_same_patient_pairs": 0, "used_1e6_entries": 0,
            "same_patient_cells_not_1e6": 0}
    for i in range(len(pop)):
        if not returned[i]:
            statuses.append("empty_query")
            relevance.append(np.zeros(0, dtype=bool))
            ranks.append([])
            n_cand.append(0)
            ties.append({k: False for k in CUTOFFS})
            continue
        same = (patients == patients[i]) & returned
        same[i] = False
        cand = np.flatnonzero(returned & (patients != patients[i]))
        diag["same_patient_cells_not_1e6"] += int(np.count_nonzero(dist[i, same] != SENTINEL))
        diag["excluded_same_patient_pairs"] += int(np.count_nonzero(same))
        if cand.size == 0:
            statuses.append("empty_reference_library")
            relevance.append(np.zeros(0, dtype=bool))
            ranks.append([])
            n_cand.append(0)
            ties.append({k: False for k in CUTOFFS})
            continue
        scores = dist[i, cand]
        if not np.all(np.isfinite(scores)):
            raise EvalError(f"nonfinite distance on a used pair for query {ids[i]}")
        n_sent = int(np.count_nonzero(scores == SENTINEL))
        diag["used_1e6_entries"] += n_sent
        if n_sent and not allow_used_sentinel:
            raise EvalError(f"query {ids[i]}: {n_sent} used 1e6 sentinel distances "
                            f"(failed pair or patient-identity disagreement)")
        diag["used_pairs"] += int(cand.size)
        order = np.argsort(scores, kind="mergesort")
        top = cand[order[:10]]
        relevance.append(groups[top] == groups[i])
        ranks.append(list(zip(top.tolist(), scores[order[:10]].tolist())))
        n_cand.append(int(cand.size))
        srt = scores[order]
        ties.append({k: bool(srt.size > k and srt[k - 1] == srt[k]) for k in CUTOFFS})
        statuses.append("returned")
    return statuses, relevance, ranks, n_cand, ties, diag


class DrawCache:
    def __init__(self, count: int, seed: int):
        self.count, self.seed, self._d = count, seed, {}

    def get(self, patients: Sequence[str]) -> kernel.PatientDraws:
        key = tuple(sorted(set(patients)))
        if key not in self._d:
            self._d[key] = kernel.make_patient_draws(list(key), count=self.count, seed=self.seed)
        return self._d[key]

    def describe(self) -> List[dict]:
        out = []
        for key, d in self._d.items():
            out.append({"n_patients": len(key),
                        "patients_sha256": hashlib.sha256("\n".join(key).encode()).hexdigest(),
                        "multiplicities_sha256": hashlib.sha256(
                            np.ascontiguousarray(d.multiplicities).tobytes()).hexdigest()})
        return out


def _rows_from(column0: int, prefix: tuple, population: str, summary: kernel.MetricSummary,
               groups: Sequence[str]):
    rows, reps = [], []
    for j, (g, e) in enumerate(zip((*groups, ""), (*summary.group_estimates, summary.macro))):
        rows.append((column0 + j, *prefix, "group" if g else "macro", g, population,
                     e.point, e.point_status, e.point_reason, e.ci_low, e.ci_high, e.ci_status,
                     e.ci_reason, e.n_queries, e.n_patients, e.zero_support_replicates))
        reps.append(e.replicates)
    return rows, reps


def _csv_bytes(header, rows, compressed=False) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(["" if v is None else (repr(v) if isinstance(v, float) else v) for v in r])
    data = buf.getvalue().encode()
    return gzip.compress(data, mtime=0) if compressed else data


def evaluate(pop: pd.DataFrame, arms: Sequence[Arm], *, draw_count: int = 5000, seed: int = 123,
             allow_used_sentinel: bool = False) -> dict:
    keys = [a.key for a in arms]
    if len(set(keys)) != len(keys):
        raise EvalError("duplicate arm keys")
    groups = tuple(sorted(pop.diagnostic_group.unique()))
    qids = pop.slide_id.tolist()
    pids = pop.patient_id.tolist()
    labels = pop.diagnostic_group.tolist()
    n = len(pop)
    all_in = np.ones(n, dtype=bool)
    draws = DrawCache(draw_count, seed)
    per_arm, matrix_diag = {}, {}
    query_rows, rank_rows = [], []
    for arm in arms:
        ret = returned_mask(arm, pop)
        dist, mdiag = load_matrix(arm, pop, ret)
        statuses, rel, ranks, n_cand, ties, rdiag = rank_arm(pop, dist, ret, allow_used_sentinel)
        matrix_diag[arm.key] = {**mdiag, **rdiag, "matrix": str(arm.matrix),
                                "returned": int(ret.sum()), "intended": n,
                                "boundary_ties": {str(k): int(sum(t[k] for t in ties)) for k in CUTOFFS}}
        vec = {}
        for k in CUTOFFS:
            hit = np.zeros(n)
            prec = np.zeros(n)
            yld = np.zeros(n)
            cov = np.zeros(n)
            for i in range(n):
                m = kernel.query_metrics(rel[i], k, statuses[i])
                hit[i] = m.conditional_hit or 0.0
                prec[i] = m.conditional_precision or 0.0
                yld[i] = m.hit_yield
                cov[i] = m.coverage
                query_rows.append((arm.name, arm.method, arm.budget, i, qids[i], pids[i], labels[i], k,
                                   m.status, m.returned_depth, int(m.short_list), m.conditional_hit,
                                   m.conditional_precision, m.conditional_reason, m.coverage,
                                   m.hit_yield, n_cand[i], int(ties[i][k])))
            vec[("hit", k)], vec[("precision", k)], vec[("hit_yield", k)] = hit, prec, yld
            vec[("coverage", 0)] = cov
        returned = np.array([s == "returned" for s in statuses])
        per_arm[arm.key] = dict(arm=arm, vec=vec, returned=returned)
        for i in range(n):
            for r, (j, d) in enumerate(ranks[i], start=1):
                rank_rows.append((arm.name, arm.method, arm.budget, qids[i], r, qids[j], labels[j],
                                  int(labels[j] == labels[i]), d))

    est_rows, est_reps, con_rows, con_reps = [], [], [], []
    col = 0
    for key, pa in per_arm.items():
        arm, vec, ret = pa["arm"], pa["vec"], pa["returned"]
        prefix_base = (arm.name, arm.method, arm.budget, "", "")
        d_ret = draws.get([p for p, r in zip(pids, ret) if r])
        d_all = draws.get(pids)
        for k in CUTOFFS:
            for ep in ENDPOINTS:
                s = kernel.summarize_metric(qids, pids, labels, groups, vec[(ep, k)], ret, d_ret)
                rows, reps = _rows_from(col, ("estimate", *prefix_base, ep, k), "returned", s, groups)
                est_rows += rows; est_reps += reps; col += len(rows)
            s = kernel.summarize_metric(qids, pids, labels, groups, vec[("hit_yield", k)], all_in, d_all)
            rows, reps = _rows_from(col, ("estimate", *prefix_base, "hit_yield", k), "intended", s, groups)
            est_rows += rows; est_reps += reps; col += len(rows)
        s = kernel.summarize_metric(qids, pids, labels, groups, vec[("coverage", 0)], all_in, d_all)
        rows, reps = _rows_from(col, ("estimate", *prefix_base, "coverage", ""), "intended", s, groups)
        est_rows += rows; est_reps += reps; col += len(rows)

    contrast_pairs = []
    for key, pa in per_arm.items():
        arm = pa["arm"]
        if arm.name == BASELINE_ARM:
            continue
        base = [b for b in per_arm.values()
                if b["arm"].name == BASELINE_ARM and b["arm"].method == arm.method]
        if len(base) != 1:
            raise EvalError(f"{key}: needs exactly one {BASELINE_ARM} arm with method {arm.method}")
        b = base[0]
        contrast_pairs.append(dict(arm_a=key, arm_b=b["arm"].key))
        both = pa["returned"] & b["returned"]
        d_both = draws.get([p for p, r in zip(pids, both) if r])
        d_all = draws.get(pids)
        prefix = (arm.name, arm.method, arm.budget, key, b["arm"].key)
        for k in CUTOFFS:
            for ep in ENDPOINTS:
                s = kernel.summarize_paired(qids, pids, labels, groups, pa["vec"][(ep, k)],
                                            b["vec"][(ep, k)], both, d_both).difference
                rows, reps = _rows_from(col, ("contrast", *prefix, ep, k), "both_returned", s, groups)
                con_rows += rows; con_reps += reps; col += len(rows)
            s = kernel.summarize_paired(qids, pids, labels, groups, pa["vec"][("hit_yield", k)],
                                        b["vec"][("hit_yield", k)], all_in, d_all).difference
            rows, reps = _rows_from(col, ("contrast", *prefix, "hit_yield", k), "intended", s, groups)
            con_rows += rows; con_reps += reps; col += len(rows)

    cov_rows = []
    for key, pa in per_arm.items():
        arm, vec, ret = pa["arm"], pa["vec"], pa["returned"]
        for g in (*groups, "ALL"):
            sel = np.ones(n, dtype=bool) if g == "ALL" else (pop.diagnostic_group.to_numpy() == g)
            nr = [qids[i] for i in np.flatnonzero(sel & ~ret)]
            cov_rows.append((arm.name, arm.method, arm.budget, "pooled" if g == "ALL" else "group", g,
                             int(sel.sum()), int((sel & ret).sum()), len(nr), ";".join(nr),
                             float((sel & ret).sum() / sel.sum()),
                             *(float(vec[("hit_yield", k)][sel].mean()) for k in CUTOFFS)))

    return dict(groups=groups, estimates=est_rows, estimate_replicates=est_reps,
                contrasts=con_rows, contrast_replicates=con_reps, coverage=cov_rows,
                query_rows=query_rows, rank_rows=rank_rows, matrix_diagnostics=matrix_diag,
                contrast_pairs=contrast_pairs, draws=draws, full_draws=draws.get(pids))


def write_outputs(result: dict, out_dir: Path, meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise EvalError(f"{out_dir} is not empty; use a new output directory (never overwrite)")
    files = {}

    def put(name: str, data: bytes):
        p = out_dir / name
        with open(p, "xb") as fh:
            fh.write(data)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    put("estimates.csv", _csv_bytes(ESTIMATE_HEADER, result["estimates"]))
    put("contrasts.csv", _csv_bytes(ESTIMATE_HEADER, result["contrasts"]))
    put("coverage.csv", _csv_bytes(COVERAGE_HEADER, result["coverage"]))
    put("query_metrics.csv.gz", _csv_bytes(QUERY_HEADER, result["query_rows"], compressed=True))
    put("ranked_lists.csv.gz", _csv_bytes(RANK_HEADER, result["rank_rows"], compressed=True))
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(result["full_draws"].multiplicities, dtype=np.int64))
    put("patient_draws.npy", buf.getvalue())
    reps = np.vstack([*result["estimate_replicates"], *result["contrast_replicates"]]) \
        if result["estimate_replicates"] else np.zeros((0, 0))
    buf = io.BytesIO()
    np.save(buf, reps.astype(np.float64))
    put("replicates.npy", buf.getvalue())
    summary = {**meta, "groups": list(result["groups"]),
               "patient_draws_order": "sorted unique patient_id of the intended population "
                                      "(columns of patient_draws.npy)",
               "patient_draws_patients": list(result["full_draws"].patient_ids),
               "draw_tables": result["draws"].describe(),
               "replicates_rows": "aligned with the `column` field of estimates.csv then contrasts.csv; "
                                  "replicates of different populations use different draw tables",
               "contrast_pairs": result["contrast_pairs"],
               "matrix_diagnostics": result["matrix_diagnostics"],
               "counts": {"estimates": len(result["estimates"]), "contrasts": len(result["contrasts"]),
                          "query_rows": len(result["query_rows"]), "rank_rows": len(result["rank_rows"])},
               "outputs": dict(files)}
    put("summary.json", json.dumps(summary, indent=1, default=str).encode())
    return files


def standard_arms(out: Path) -> List[Arm]:
    sm = out / "similarity_matrix"
    root = out / "r2_1_filter_20261003" / "features"
    arms = []
    for m in ("ot_cosine", "yottixel"):
        arms.append(Arm(BASELINE_ARM, m, "", sm / f"combined_similarity_matrix_h_optimus_1_{m}_no_same_patient.pkl", None))
    for b in (5, 100):
        for m in ("ot_cosine", "yottixel"):
            arms.append(Arm("both_filtered", m, str(b),
                            sm / f"combined_similarity_matrix_r2_1_hopt_filter_budget{b}_{m}_no_same_patient.pkl",
                            root / f"budget{b}" / "_filter_summary.csv"))
    return arms


def code_paths() -> List[Path]:
    return [Path(__file__).resolve(), Path(kernel.__file__).resolve()]


# =====================================================================================
# Query-fold design ([decision record] section B; repair 2026-10-04). Additive: nothing above changed.
#
# A query of fold f is filtered by head f, and so is the reference library it searches:
#   both_filtered -- library = every population slide returned (>= min tiles) by head f
#   query_only    -- library = the unfiltered 2,604-slide population (saved bank)
# One similarity row per returned query (similarity_cli external mode), assembled across
# folds by ``combine_qf``; a row's library is its query fold's. Non-returned queries are
# those with fewer than the minimum retained tiles under the head of their own fold.
# =====================================================================================
QF_ARMS = ("query_only", "both_filtered")
QF_ARM_TOKEN = {"query_only": "query", "both_filtered": "both"}
QF_METHOD_TAG = {"ot": "ot_cosine", "bob": "yottixel"}
QF_FOLDS = (0, 1, 2, 3, 4)
UNFILTERED_LIBRARY_ID = "unfiltered_pop2604"
QF_ROWS_HEADER = ("query", "query_index", "patient_id", "diagnostic_group", "query_fold",
                  "library_id", "library_size", "library_ids_sha256", "n_failed_pairs",
                  "row_file", "row_sha256")
QF_PRECISION_CUTOFFS = (3, 5, 10)      # precision@1 == hit@1 exactly; not repeated
CLAMP_ABS = 1e-12
QUERY_HEADER_QF = QUERY_HEADER + ("query_fold", "library_id", "library_size",
                                  "tie_averaged_hit", "tie_averaged_precision")
COVERAGE_HEADER_QF = ("arm", "method", "budget", "level", "group", "intended", "returned",
                      "non_returns", "non_return_slides", "coverage",
                      *(f"slide_mean_hit_yield_at_{k}" for k in CUTOFFS))
TIE_HEADER = ("tie_rule",) + ESTIMATE_HEADER
TIE_COUNT_HEADER = ("arm", "method", "budget", "k", "returned_queries", "boundary_tie_queries",
                    "tie_dependent_hit_queries", "tie_dependent_precision_queries")
LIBRARY_HEADER = ("arm", "method", "budget", "query_fold", "library_id", "library_size",
                  "library_excluded_slides", "queries_intended", "queries_returned",
                  "non_returned_queries", "library_ids_sha256")
TIE_RULES = {
    "stable_reference_index": "primary: ascending distance, ties broken stably by population "
                              "reference_index order (the rule of the existing evaluator)",
    "tie_averaged": "prespecified alternative: expected value of the endpoint when the order "
                    "within the tied-distance block at the k boundary is uniformly random; "
                    "hit = 1 if a relevant reference is strictly closer than the k-th distance, "
                    "else 1 - C(t-r, s)/C(t, s); precision = (relevant strictly closer + s*r/t)/k, "
                    "with t the tied-block size, r its relevant count, s = k - (strictly closer)",
}


def qf_matrix_name(budget, arm: str, method_tag: str, masked: bool = True) -> str:
    return (f"combined_similarity_matrix_r2_1_hopt_filter_qf_budget{budget}_{QF_ARM_TOKEN[arm]}_"
            f"{method_tag}{'_no_same_patient' if masked else ''}.pkl")


def qf_rows_name(budget, arm: str, method_tag: str) -> str:
    """Per-row provenance sidecar written beside the two matrices by ``combine_qf``."""
    return qf_matrix_name(budget, arm, method_tag, False)[:-len(".pkl")] + "_rows.csv"


def _ids_sha(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _bank_summary(banks_root: Path, budget, fold: int, pop: pd.DataFrame) -> Tuple[pd.DataFrame, Path]:
    p = Path(banks_root) / f"budget{budget}" / f"fold{fold}" / "_filter_summary.csv"
    s = pd.read_csv(p, dtype={"slide_id": str})
    if s.slide_id.duplicated().any() or set(s.slide_id) != set(pop.slide_id):
        raise EvalError(f"{p}: slides differ from the population")
    if "head_fold" not in s.columns or not (s.head_fold == fold).all():
        raise EvalError(f"{p}: not a head-fold-{fold} bank (head_fold column)")
    if not set(s.written.unique()).issubset({0, 1}):
        raise EvalError(f"{p}: written must be 0/1")
    s = s.set_index("slide_id").reindex(pop.slide_id)
    own = pop.fold.astype(int).to_numpy() == fold
    if not (s.slide_fold.astype(int).to_numpy() == pop.fold.astype(int).to_numpy()).all():
        raise EvalError(f"{p}: slide_fold disagrees with the population fold")
    if "own_fold_query" in s.columns and not (s.own_fold_query.astype(int).to_numpy() == own).all():
        raise EvalError(f"{p}: own_fold_query disagrees with the population fold")
    return s, p


def qf_design(pop: pd.DataFrame, banks_root: Path, budget, arm: str,
              folds: Optional[Sequence[int]] = None) -> dict:
    """Declared returned queries and per-fold libraries of one query-fold arm/budget.

    ``folds`` defaults to the population's folds (0..4 for the accepted population)."""
    if arm not in QF_ARMS:
        raise EvalError(f"unknown query-fold arm {arm!r}")
    if "fold" not in pop.columns:
        raise EvalError("population lacks the fold column")
    pf = pop.fold.astype(int).to_numpy()
    if folds is None:
        folds = sorted(set(pf.tolist()))
    if sorted(set(pf.tolist())) != sorted(folds):
        raise EvalError(f"population folds {sorted(set(pf.tolist()))} != {sorted(folds)}")
    ids = pop.slide_id.to_numpy(dtype=object)
    returned = np.zeros(len(pop), dtype=bool)
    libs = {}
    for f in folds:
        s, p = _bank_summary(banks_root, budget, f, pop)
        w = s.written.astype(int).to_numpy().astype(bool)
        own = pf == f
        returned[own] = w[own]
        lib = w if arm == "both_filtered" else np.ones(len(pop), dtype=bool)
        libs[f] = dict(
            library_id=(f"bank_b{budget}_fold{f}" if arm == "both_filtered" else UNFILTERED_LIBRARY_ID),
            mask=lib, size=int(lib.sum()), excluded=sorted(ids[~lib].tolist()),
            ids_sha256=_ids_sha(ids[lib].tolist()), summary=str(p), summary_sha256=sha256_file(p),
            intended=int(own.sum()), returned=int((own & w).sum()),
            non_returned=sorted(ids[own & ~w].tolist()))
    return dict(returned=returned, libraries=libs, folds=pf)


def combine_qf(*, pop: pd.DataFrame, arm: str, method: str, budget, rows_root: Path,
               banks_root: Path, out_dir: Path, metadata: Optional[Path] = None,
               folds: Optional[Sequence[int]] = None, code: Sequence[Path] = ()) -> dict:
    """Assemble per-query external rows into one query x population matrix (NaN outside a row's
    library) plus its ``_no_same_patient`` twin, a per-row provenance sidecar and a record.

    Fails loudly unless every fold's row files are exactly its declared returned queries and
    every row's reference ids are exactly that fold's declared library."""
    t0 = time.time()
    if method not in QF_METHOD_TAG:
        raise EvalError(f"method must be one of {sorted(QF_METHOD_TAG)}")
    tag = QF_METHOD_TAG[method]
    des = qf_design(pop, banks_root, budget, arm, folds)
    folds = sorted(des["libraries"])
    ids = pop.slide_id.tolist()
    col = {s: j for j, s in enumerate(ids)}
    patients = pop.patient_id.to_numpy(dtype=object)
    meta_check = None
    if metadata is not None:
        m = pd.read_csv(metadata, dtype=str)
        lk = m.drop_duplicates("Slide.Scan.File").set_index("Slide.Scan.File")["UID.Subject"]
        got = pop.slide_id.map(lk)
        bad = int((got.isna() | (got != pop.patient_id)).sum())
        if bad:
            raise EvalError(f"{metadata}: UID.Subject disagrees with population patient_id for {bad} slides")
        meta_check = {"path": str(metadata), "sha256": sha256_file(metadata), "slides_checked": len(pop),
                      "disagreements": 0}
    q_idx = np.flatnonzero(des["returned"])
    mat = np.full((len(q_idx), len(ids)), np.nan)
    rows_out, gaps, per_fold = [], [], {}
    for f in folds:
        lib = des["libraries"][f]
        d = Path(rows_root) / f"budget{budget}" / f"fold{f}" / f"{QF_ARM_TOKEN[arm]}_{method}"
        expect = sorted(pop.slide_id[(des["folds"] == f) & des["returned"]])
        have = sorted(p.name[len("similarity_"):-len(".pkl")] for p in d.glob("similarity_*.pkl")) \
            if d.is_dir() else []
        if have != expect:
            miss, extra = sorted(set(expect) - set(have)), sorted(set(have) - set(expect))
            gaps.append(f"fold {f} ({d}): {len(miss)} missing {miss[:10]}, {len(extra)} unexpected {extra[:10]}")
            continue
        lib_ids = set(np.asarray(ids, dtype=object)[lib["mask"]].tolist())
        n_failed_fold = 0
        for qid in expect:
            fp = d / f"similarity_{qid}.pkl"
            raw = fp.read_bytes()
            row = pickle.loads(raw)
            oids = [str(o) for o, _ in row]
            if len(set(oids)) != len(oids) or set(oids) != lib_ids:
                raise EvalError(f"{fp}: reference ids ({len(oids)}) are not exactly the fold-{f} "
                                f"library {lib['library_id']} ({len(lib_ids)})")
            r = int(np.searchsorted(q_idx, col[qid]))
            vals = {o: float(v) for o, v in row}
            if not all(np.isfinite(v) for v in vals.values()):
                raise EvalError(f"{fp}: nonfinite distance")
            if vals[qid] != SENTINEL:
                raise EvalError(f"{fp}: self distance {vals[qid]!r} is not the 1e6 sentinel")
            n_failed = sum(1 for o, v in vals.items() if v == SENTINEL and o != qid)
            n_failed_fold += n_failed
            for o, v in vals.items():
                mat[r, col[o]] = v
            i = col[qid]
            rows_out.append((qid, i, patients[i], pop.diagnostic_group[i], f, lib["library_id"],
                             lib["size"], lib["ids_sha256"], n_failed, str(fp),
                             hashlib.sha256(raw).hexdigest()))
        per_fold[str(f)] = {k: v for k, v in lib.items() if k != "mask"}
        per_fold[str(f)].update(rows_dir=str(d), rows=len(expect), failed_pairs=n_failed_fold)
    if gaps:
        raise EvalError("row files differ from the declared returned queries: " + "; ".join(gaps))
    rows_out.sort(key=lambda t: t[1])
    q_ids = [ids[i] for i in q_idx]
    raw_df = pd.DataFrame(mat, index=q_ids, columns=ids)
    masked = mat.copy()
    same = patients[q_idx][:, None] == patients[None, :]
    same[np.arange(len(q_idx)), q_idx] = False
    lib_used = ~np.isnan(mat)
    n_masked = int((same & lib_used).sum())
    masked[same & lib_used] = SENTINEL
    masked_df = pd.DataFrame(masked, index=q_ids, columns=ids)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, obj in ((qf_matrix_name(budget, arm, tag, False), raw_df),
                      (qf_matrix_name(budget, arm, tag, True), masked_df)):
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        with open(out_dir / name, "xb") as fh:
            fh.write(data)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    base = qf_matrix_name(budget, arm, tag, False)[:-len(".pkl")]
    data = _csv_bytes(QF_ROWS_HEADER, rows_out)
    with open(out_dir / qf_rows_name(budget, arm, tag), "xb") as fh:
        fh.write(data)
    files[qf_rows_name(budget, arm, tag)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    rec = {"created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"), "argv": sys.argv,
           "design": "query-fold (ADR0021 B): rows of fold f use head f's library", "arm": arm,
           "method": method, "method_tag": tag, "budget": str(budget), "rows": len(q_ids),
           "columns": len(ids), "cells_outside_row_library": "NaN",
           "same_patient_cells_masked": n_masked,
           "same_patient_rule": "population patient_id (all slides of the query's patient, "
                                "cross-label included); self cell keeps the 1e6 sentinel",
           "metadata_patient_check": meta_check, "failed_pairs_total": sum(r[8] for r in rows_out),
           "max_real_distance": float(np.nanmax(np.where(mat < SENTINEL, mat, np.nan))) if len(q_ids) else None,
           "folds": per_fold, "banks_root": str(banks_root), "rows_root": str(rows_root),
           "code_sha256": {str(c): sha256_file(Path(c)) for c in code},
           "outputs": files, "wall_seconds": round(time.time() - t0, 1)}
    data = json.dumps(rec, indent=1, default=str).encode()
    with open(out_dir / f"{base}_combine_record.json", "xb") as fh:
        fh.write(data)
    return rec


@dataclass(frozen=True)
class QfArm:
    name: str              # all_tissue | query_only | both_filtered
    method: str            # ot_cosine | yottixel
    budget: str            # "" for all_tissue
    matrix: Path
    rows: Optional[Path]   # combine_qf sidecar; None for all_tissue

    @property
    def key(self) -> str:
        return f"{self.name}|{self.method}|{self.budget}"


def load_qf_arm(arm: QfArm, pop: pd.DataFrame, banks_root: Optional[Path]):
    """Distance array (population rows/columns), returned mask, per-row library mask, info.

    The all-tissue baseline uses the existing ``load_matrix`` path (library = population)."""
    n = len(pop)
    if arm.name == BASELINE_ARM:
        base = Arm(BASELINE_ARM, arm.method, "", arm.matrix, None)
        ret = np.ones(n, dtype=bool)
        dist, diag = load_matrix(base, pop, ret)
        lib = np.ones((n, n), dtype=bool)
        fold = pop.fold.astype(int).to_numpy() if "fold" in pop.columns else np.zeros(n, dtype=int)
        info = {f: dict(library_id="all_tissue_pop2604", size=n, excluded=[], intended=int((fold == f).sum()),
                        returned=int((fold == f).sum()), non_returned=[],
                        ids_sha256=_ids_sha(pop.slide_id.tolist())) for f in sorted(set(fold.tolist()))}
        return dist, ret, lib, ["all_tissue_pop2604"] * n, {**diag, "libraries": info}
    if banks_root is None or arm.rows is None:
        raise EvalError(f"{arm.key}: query-fold arms need --banks-root and a rows sidecar")
    des = qf_design(pop, banks_root, arm.budget, arm.name)
    with open(arm.matrix, "rb") as fh:
        df = pickle.load(fh)
    if not isinstance(df, pd.DataFrame) or df.to_numpy().dtype.kind != "f":
        raise EvalError(f"{arm.matrix}: not a floating DataFrame")
    rows = [str(i) for i in df.index]
    cols = [str(c) for c in df.columns]
    ids = pop.slide_id.tolist()
    if cols != ids:
        raise EvalError(f"{arm.matrix}: columns are not the population in reference_index order")
    want = pop.slide_id[des["returned"]].tolist()
    if rows != want:
        raise EvalError(f"{arm.matrix}: rows ({len(rows)}) are not the declared returned queries ({len(want)})")
    side = pd.read_csv(arm.rows, dtype={"query": str, "library_id": str, "patient_id": str})
    if side["query"].tolist() != rows:
        raise EvalError(f"{arm.rows}: sidecar queries differ from the matrix rows")
    values = df.to_numpy().astype(np.float64)
    q_idx = np.flatnonzero(des["returned"])
    dist = np.full((n, n), np.nan)
    dist[q_idx] = values
    lib = np.zeros((n, n), dtype=bool)
    lib_id = [""] * n
    for r, i in enumerate(q_idx):
        f = int(des["folds"][i])
        L = des["libraries"][f]
        got = ~np.isnan(values[r])
        if not np.array_equal(got, L["mask"]):
            raise EvalError(f"{arm.matrix}: row {ids[i]} non-NaN cells differ from the fold-{f} library")
        s = side.iloc[r]
        if (int(s.query_fold) != f or s.library_id != L["library_id"] or int(s.library_size) != L["size"]
                or s.library_ids_sha256 != L["ids_sha256"]):
            raise EvalError(f"{arm.rows}: provenance of {ids[i]} disagrees with the declared library")
        lib[i] = L["mask"]
        lib_id[i] = L["library_id"]
    info = {f: {k: v for k, v in L.items() if k != "mask"} for f, L in des["libraries"].items()}
    return dist, des["returned"], lib, lib_id, {"matrix_shape": list(values.shape), "libraries": info}


def tie_averaged(scores_sorted: np.ndarray, rel_sorted: np.ndarray, k: int) -> Tuple[float, float]:
    """Expected hit@k / precision@k under a uniformly random order within distance ties."""
    from fractions import Fraction
    from math import comb
    n = scores_sorted.size
    if n <= k:
        c = int(np.count_nonzero(rel_sorted))
        return float(c > 0), c / k
    d = scores_sorted[k - 1]
    a = int(np.searchsorted(scores_sorted, d, side="left"))
    e = int(np.searchsorted(scores_sorted, d, side="right"))
    t, s = e - a, k - a
    r = int(np.count_nonzero(rel_sorted[a:e]))
    rb = int(np.count_nonzero(rel_sorted[:a]))
    hit = 1.0 if rb > 0 else float(1 - Fraction(comb(t - r, s), comb(t, s)))
    prec = float(Fraction(rb * t + s * r, t * k))
    return hit, prec


def rank_arm_qf(pop: pd.DataFrame, dist: np.ndarray, returned: np.ndarray, lib: np.ndarray,
                allow_used_sentinel: bool = False):
    """``rank_arm`` with a per-row library (``lib[i]``) and tie-averaged metrics."""
    groups = pop.diagnostic_group.to_numpy(dtype=object)
    patients = pop.patient_id.to_numpy(dtype=object)
    ids = pop.slide_id.to_numpy(dtype=object)
    statuses, relevance, ranks, n_cand, ties, ta = [], [], [], [], [], []
    diag = {"used_pairs": 0, "excluded_same_patient_pairs": 0, "used_1e6_entries": 0,
            "same_patient_cells_not_1e6": 0}
    nothing = {k: (0.0, 0.0) for k in CUTOFFS}
    for i in range(len(pop)):
        if not returned[i]:
            statuses.append("empty_query"); relevance.append(np.zeros(0, dtype=bool)); ranks.append([])
            n_cand.append(0); ties.append({k: False for k in CUTOFFS}); ta.append(nothing)
            continue
        same = (patients == patients[i]) & lib[i]
        same[i] = False
        cand = np.flatnonzero(lib[i] & (patients != patients[i]))
        diag["same_patient_cells_not_1e6"] += int(np.count_nonzero(dist[i, same] != SENTINEL))
        diag["excluded_same_patient_pairs"] += int(np.count_nonzero(same))
        if cand.size == 0:
            statuses.append("empty_reference_library"); relevance.append(np.zeros(0, dtype=bool))
            ranks.append([]); n_cand.append(0); ties.append({k: False for k in CUTOFFS}); ta.append(nothing)
            continue
        scores = dist[i, cand]
        if not np.all(np.isfinite(scores)):
            raise EvalError(f"nonfinite distance on a used pair for query {ids[i]}")
        n_sent = int(np.count_nonzero(scores == SENTINEL))
        diag["used_1e6_entries"] += n_sent
        if n_sent and not allow_used_sentinel:
            raise EvalError(f"query {ids[i]}: {n_sent} used 1e6 sentinel distances "
                            f"(failed pair or patient-identity disagreement)")
        diag["used_pairs"] += int(cand.size)
        order = np.argsort(scores, kind="mergesort")
        top = cand[order[:10]]
        relevance.append(groups[top] == groups[i])
        ranks.append(list(zip(top.tolist(), scores[order[:10]].tolist())))
        n_cand.append(int(cand.size))
        srt = scores[order]
        rel_all = groups[cand[order]] == groups[i]
        ties.append({k: bool(srt.size > k and srt[k - 1] == srt[k]) for k in CUTOFFS})
        ta.append({k: tie_averaged(srt, rel_all, k) for k in CUTOFFS})
        statuses.append("returned")
    return statuses, relevance, ranks, n_cand, ties, ta, diag


def qf_contrast_specs(keys: Sequence[str]) -> List[Tuple[str, str]]:
    """Prespecified contrasts (A minus B) among the arms present."""
    have = set(keys)
    out = []
    for m in ("ot_cosine", "yottixel"):
        base = f"{BASELINE_ARM}|{m}|"
        for b in ("5", "100"):
            q, bo = f"query_only|{m}|{b}", f"both_filtered|{m}|{b}"
            for a in (q, bo):
                if a in have and base in have:
                    out.append((a, base))
            if q in have and bo in have:
                out.append((q, bo))
        for a in QF_ARMS:
            x, y = f"{a}|{m}|5", f"{a}|{m}|100"
            if x in have and y in have:
                out.append((x, y))
    return out


def _endpoint_list():
    eps = []
    for k in CUTOFFS:
        eps.append(("hit", k))
        if k in QF_PRECISION_CUTOFFS:
            eps.append(("precision", k))
    return eps


def evaluate_qf(pop: pd.DataFrame, arms: Sequence[QfArm], *, banks_root: Optional[Path],
                draw_count: int = 5000, seed: int = 123, allow_used_sentinel: bool = False) -> dict:
    keys = [a.key for a in arms]
    if len(set(keys)) != len(keys):
        raise EvalError("duplicate arm keys")
    groups = tuple(sorted(pop.diagnostic_group.unique()))
    qids, pids, labels = pop.slide_id.tolist(), pop.patient_id.tolist(), pop.diagnostic_group.tolist()
    n = len(pop)
    fold = pop.fold.astype(int).to_numpy()
    all_in = np.ones(n, dtype=bool)
    draws = DrawCache(draw_count, seed)
    per_arm, matrix_diag, query_rows, rank_rows, lib_rows, tie_counts = {}, {}, [], [], [], []
    for arm in arms:
        dist, ret, lib, lib_id, mdiag = load_qf_arm(arm, pop, banks_root)
        statuses, rel, ranks, n_cand, ties, ta, rdiag = rank_arm_qf(pop, dist, ret, lib, allow_used_sentinel)
        matrix_diag[arm.key] = {**{k: v for k, v in mdiag.items() if k != "libraries"}, **rdiag,
                                "matrix": str(arm.matrix), "returned": int(ret.sum()), "intended": n,
                                "boundary_ties": {str(k): int(sum(t[k] for t in ties)) for k in CUTOFFS}}
        for f, L in mdiag["libraries"].items():
            lib_rows.append((arm.name, arm.method, arm.budget, f,
                             L["library_id"], L["size"], ";".join(L["excluded"]), L["intended"],
                             L["returned"], ";".join(L["non_returned"]), L["ids_sha256"]))
        vec = {}
        lib_size = lib.sum(axis=1)
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
                query_rows.append((arm.name, arm.method, arm.budget, i, qids[i], pids[i], labels[i], k,
                                   m.status, m.returned_depth, int(m.short_list), m.conditional_hit,
                                   m.conditional_precision, m.conditional_reason, m.coverage,
                                   m.hit_yield, n_cand[i], int(ties[i][k]), int(fold[i]), lib_id[i],
                                   int(lib_size[i]) if ret[i] else 0,
                                   ta[i][k][0] if statuses[i] == "returned" else None,
                                   ta[i][k][1] if statuses[i] == "returned" else None))
            for x in ("hit", "precision", "hit_yield", "hit_ta", "precision_ta", "hit_yield_ta"):
                vec[(x, k)] = v[x]
            vec[("coverage", 0)] = v["coverage"]
            r = np.array([s == "returned" for s in statuses])
            tie_counts.append((arm.name, arm.method, arm.budget, k, int(r.sum()),
                               int(sum(t[k] for t in ties)),
                               int(np.count_nonzero(r & (v["hit"] != v["hit_ta"]))),
                               int(np.count_nonzero(r & (v["precision"] != v["precision_ta"])))))
        returned = np.array([s == "returned" for s in statuses])
        per_arm[arm.key] = dict(arm=arm, vec=vec, returned=returned)
        for i in range(n):
            for rr, (j, d) in enumerate(ranks[i], start=1):
                rank_rows.append((arm.name, arm.method, arm.budget, qids[i], rr, qids[j], labels[j],
                                  int(labels[j] == labels[i]), d))

    def arm_estimates(col, suffix=""):
        rows_, reps_ = [], []
        for key, pa in per_arm.items():
            arm, vec, ret = pa["arm"], pa["vec"], pa["returned"]
            pre = (arm.name, arm.method, arm.budget, "", "")
            d_ret = draws.get([p for p, r in zip(pids, ret) if r])
            d_all = draws.get(pids)
            for ep, k in _endpoint_list():
                s = kernel.summarize_metric(qids, pids, labels, groups, vec[(ep + suffix, k)], ret, d_ret)
                a, b = _rows_from(col, ("estimate", *pre, ep, k), "returned", s, groups)
                rows_ += a; reps_ += b; col += len(a)
            for k in CUTOFFS:
                s = kernel.summarize_metric(qids, pids, labels, groups, vec[("hit_yield" + suffix, k)], all_in, d_all)
                a, b = _rows_from(col, ("estimate", *pre, "hit_yield", k), "intended", s, groups)
                rows_ += a; reps_ += b; col += len(a)
            if not suffix:
                s = kernel.summarize_metric(qids, pids, labels, groups, vec[("coverage", 0)], all_in, d_all)
                a, b = _rows_from(col, ("estimate", *pre, "coverage", ""), "intended", s, groups)
                rows_ += a; reps_ += b; col += len(a)
        return rows_, reps_, col

    specs = qf_contrast_specs(list(per_arm))

    def contrasts(col, suffix=""):
        rows_, reps_ = [], []
        for ka, kb in specs:
            A, B = per_arm[ka], per_arm[kb]
            both = A["returned"] & B["returned"]
            d_both = draws.get([p for p, r in zip(pids, both) if r])
            d_all = draws.get(pids)
            arm = A["arm"]
            pre = (arm.name, arm.method, arm.budget, ka, kb)
            for ep, k in _endpoint_list():
                s = kernel.summarize_paired(qids, pids, labels, groups, A["vec"][(ep + suffix, k)],
                                            B["vec"][(ep + suffix, k)], both, d_both).difference
                a, b = _rows_from(col, ("contrast", *pre, ep, k), "both_returned", s, groups)
                rows_ += a; reps_ += b; col += len(a)
            for k in CUTOFFS:
                s = kernel.summarize_paired(qids, pids, labels, groups, A["vec"][("hit_yield" + suffix, k)],
                                            B["vec"][("hit_yield" + suffix, k)], all_in, d_all).difference
                a, b = _rows_from(col, ("contrast", *pre, "hit_yield", k), "intended", s, groups)
                rows_ += a; reps_ += b; col += len(a)
        return rows_, reps_, col

    est_rows, est_reps, col = arm_estimates(0)
    con_rows, con_reps, col = contrasts(col)
    ta_est, ta_est_reps, c2 = arm_estimates(0, "_ta")
    ta_con, ta_con_reps, _ = contrasts(c2, "_ta")

    cov_rows = []
    for key, pa in per_arm.items():
        arm, vec, ret = pa["arm"], pa["vec"], pa["returned"]
        for g in (*groups, "ALL"):
            sel = np.ones(n, dtype=bool) if g == "ALL" else (pop.diagnostic_group.to_numpy() == g)
            nr = [qids[i] for i in np.flatnonzero(sel & ~ret)]
            cov_rows.append((arm.name, arm.method, arm.budget,
                             "pooled_slides_NOT_macro" if g == "ALL" else "group", g,
                             int(sel.sum()), int((sel & ret).sum()), len(nr), ";".join(nr),
                             float((sel & ret).sum() / sel.sum()),
                             *(float(vec[("hit_yield", k)][sel].mean()) for k in CUTOFFS)))
    return dict(groups=groups, estimates=est_rows, estimate_replicates=est_reps, contrasts=con_rows,
                contrast_replicates=con_reps, tie_estimates=ta_est, tie_estimate_replicates=ta_est_reps,
                tie_contrasts=ta_con, tie_contrast_replicates=ta_con_reps, tie_counts=tie_counts,
                coverage=cov_rows, query_rows=query_rows, rank_rows=rank_rows, libraries=lib_rows,
                matrix_diagnostics=matrix_diag, contrast_pairs=[dict(arm_a=a, arm_b=b) for a, b in specs],
                draws=draws, full_draws=draws.get(pids))


_POINT_COLS = tuple(ESTIMATE_HEADER.index(c) for c in ("point", "ci_low", "ci_high"))


def clamp_rows(rows, offset: int = 0) -> Tuple[list, int]:
    """CSV-only: |x| < 1e-12 at point/ci_low/ci_high becomes exactly 0.0 (replicates untouched)."""
    out, n = [], 0
    for r in rows:
        r = list(r)
        for c in _POINT_COLS:
            v = r[c + offset]
            if isinstance(v, float) and np.isfinite(v) and abs(v) < CLAMP_ABS and (v != 0.0 or
                                                                                  np.signbit(v)):
                r[c + offset] = 0.0
                n += 1
        out.append(tuple(r))
    return out, n


def write_outputs_qf(result: dict, out_dir: Path, meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise EvalError(f"{out_dir} is not empty; use a new output directory (never overwrite)")
    files = {}

    def put(name: str, data: bytes):
        with open(out_dir / name, "xb") as fh:
            fh.write(data)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    def npy(a) -> bytes:
        buf = io.BytesIO()
        np.save(buf, a)
        return buf.getvalue()

    est, n1 = clamp_rows(result["estimates"])
    con, n2 = clamp_rows(result["contrasts"])
    tie_rows = [("stable_reference_index", *r) for r in (*result["estimates"], *result["contrasts"])
                if r[ESTIMATE_HEADER.index("endpoint")] in ("hit", "precision", "hit_yield")]
    tie_rows += [("tie_averaged", *r) for r in (*result["tie_estimates"], *result["tie_contrasts"])]
    tie_rows, n3 = clamp_rows(tie_rows, offset=1)
    put("estimates.csv", _csv_bytes(ESTIMATE_HEADER, est))
    put("contrasts.csv", _csv_bytes(ESTIMATE_HEADER, con))
    put("tie_sensitivity.csv", _csv_bytes(TIE_HEADER, tie_rows))
    put("tie_counts.csv", _csv_bytes(TIE_COUNT_HEADER, result["tie_counts"]))
    put("coverage.csv", _csv_bytes(COVERAGE_HEADER_QF, result["coverage"]))
    put("libraries.csv", _csv_bytes(LIBRARY_HEADER, result["libraries"]))
    put("query_metrics.csv.gz", _csv_bytes(QUERY_HEADER_QF, result["query_rows"], compressed=True))
    put("ranked_lists.csv.gz", _csv_bytes(RANK_HEADER, result["rank_rows"], compressed=True))
    put("patient_draws.npy", npy(np.ascontiguousarray(result["full_draws"].multiplicities, dtype=np.int64)))
    reps = [*result["estimate_replicates"], *result["contrast_replicates"]]
    put("replicates.npy", npy(np.vstack(reps).astype(np.float64) if reps else np.zeros((0, 0))))
    treps = [*result["tie_estimate_replicates"], *result["tie_contrast_replicates"]]
    put("tie_averaged_replicates.npy", npy(np.vstack(treps).astype(np.float64) if treps else np.zeros((0, 0))))
    summary = {**meta, "groups": list(result["groups"]),
               "patient_draws_order": "sorted unique patient_id of the intended population "
                                      "(columns of patient_draws.npy)",
               "patient_draws_patients": list(result["full_draws"].patient_ids),
               "draw_tables": result["draws"].describe(),
               "replicates_rows": "replicates.npy rows align with the `column` field of estimates.csv "
                                  "then contrasts.csv (stable rule); tie_averaged_replicates.npy rows align "
                                  "with the `column` field of the tie_averaged rows of tie_sensitivity.csv; "
                                  "stable rows of tie_sensitivity.csv keep their estimates/contrasts column",
               "clamped_csv_cells": {"estimates": n1, "contrasts": n2, "tie_sensitivity": n3,
                                     "rule": f"|x| < {CLAMP_ABS} written as 0.0 in CSV only"},
               "contrast_pairs": result["contrast_pairs"],
               "matrix_diagnostics": result["matrix_diagnostics"],
               "counts": {"estimates": len(result["estimates"]), "contrasts": len(result["contrasts"]),
                          "tie_sensitivity": len(tie_rows), "query_rows": len(result["query_rows"]),
                          "rank_rows": len(result["rank_rows"])},
               "outputs": dict(files)}
    put("summary.json", json.dumps(summary, indent=1, default=str).encode())
    return files


def standard_qf_arms(out: Path, matrix_dir: Optional[Path] = None) -> List[QfArm]:
    sm = out / "similarity_matrix"
    qd = matrix_dir or (out / "r2_1_filter_20261003" / "similarity_matrix_qf")
    arms = [QfArm(BASELINE_ARM, m, "", sm / f"combined_similarity_matrix_h_optimus_1_{m}_no_same_patient.pkl", None)
            for m in ("ot_cosine", "yottixel")]
    for b in (5, 100):
        for name in QF_ARMS:
            for m in ("ot_cosine", "yottixel"):
                mp = qd / qf_matrix_name(b, name, m, True)
                arms.append(QfArm(name, m, str(b), mp, qd / qf_rows_name(b, name, m)))
    return arms


def main_combine_qf(argv) -> int:
    env_out = os.environ.get("PROSPER_OUT")
    out = Path(env_out) if env_out else None
    R = out / "r2_1_filter_20261003" if out else None
    p = argparse.ArgumentParser(prog="r2_1_filter_eval_cli combine-qf", description=combine_qf.__doc__)
    p.add_argument("--arm", required=True, choices=QF_ARMS)
    p.add_argument("--method", required=True, choices=sorted(QF_METHOD_TAG))
    p.add_argument("--budget", required=True, choices=["5", "100"])
    p.add_argument("--population", type=Path, default=(out / "r2_1_mosaic_cpu_20261003" / "attempt_001"
                                                        / "retrieval_population.csv") if out else None)
    p.add_argument("--rows-root", type=Path, default=R / "similarity_qf" if R else None)
    p.add_argument("--banks-root", type=Path, default=R / "features_qf" if R else None)
    p.add_argument("--out-dir", type=Path, default=R / "similarity_matrix_qf" if R else None)
    p.add_argument("--slide-metadata", type=Path, default=None,
                   help="cross-check population patient_id against UID.Subject (default: prosper paths)")
    p.add_argument("--no-metadata-check", action="store_true")
    p.add_argument("--folds", type=int, nargs="+", default=list(QF_FOLDS),
                   help="population folds that must all be present (default 0..4)")
    a = p.parse_args(argv)
    if None in (a.population, a.rows_root, a.banks_root, a.out_dir):
        p.error("export PROSPER_OUT or pass all paths")
    md = None
    if not a.no_metadata_check:
        if a.slide_metadata is None:
            from . import paths as _paths
            md = _paths.slide_metadata()
        else:
            md = a.slide_metadata
    pop = load_population(a.population)
    rec = combine_qf(pop=pop, arm=a.arm, method=a.method, budget=a.budget, rows_root=a.rows_root,
                     banks_root=a.banks_root, out_dir=a.out_dir, metadata=md, folds=a.folds,
                     code=code_paths())
    print(json.dumps(rec, indent=1, default=str))
    return 0


def main_evaluate_qf(argv) -> int:
    env_out = os.environ.get("PROSPER_OUT")
    out = Path(env_out) if env_out else None
    R = out / "r2_1_filter_20261003" if out else None
    p = argparse.ArgumentParser(prog="r2_1_filter_eval_cli evaluate-qf",
                                description="Query-fold design: all_tissue / query_only / both_filtered")
    p.add_argument("--population", type=Path, default=(out / "r2_1_mosaic_cpu_20261003" / "attempt_001"
                                                        / "retrieval_population.csv") if out else None)
    p.add_argument("--standard", action="store_true",
                   help="all-tissue baselines + query_only/both_filtered x budgets 5/100 x OT/Yottixel")
    p.add_argument("--matrix-dir", type=Path, default=None, help="default r2_1_filter_20261003/similarity_matrix_qf")
    p.add_argument("--arm", nargs=5, action="append", default=[],
                   metavar=("NAME", "METHOD", "BUDGET", "MATRIX", "ROWS"),
                   help="arm; BUDGET/ROWS '-' for the all_tissue baseline")
    p.add_argument("--banks-root", type=Path, default=R / "features_qf" if R else None)
    p.add_argument("--out-dir", type=Path, default=R / "evaluation_qf" if R else None)
    p.add_argument("--draw-count", type=int, default=5000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--allow-used-sentinel", action="store_true")
    a = p.parse_args(argv)
    if a.population is None or a.out_dir is None:
        p.error("export PROSPER_OUT or pass --population/--out-dir")
    arms = standard_qf_arms(out, a.matrix_dir) if a.standard else []
    for name, method, budget, matrix, rows in a.arm:
        arms.append(QfArm(name, method, "" if budget == "-" else budget, Path(matrix),
                          None if rows == "-" else Path(rows)))
    if not arms:
        p.error("no arms: pass --standard and/or --arm")
    t0 = time.time()
    pop = load_population(a.population)
    result = evaluate_qf(pop, arms, banks_root=a.banks_root, draw_count=a.draw_count, seed=a.seed,
                         allow_used_sentinel=a.allow_used_sentinel)
    banks = {}
    if a.banks_root is not None:
        for sp in sorted(Path(a.banks_root).glob("budget*/fold*/_filter_summary.csv")):
            banks[str(sp)] = sha256_file(sp)
    meta = {
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"), "argv": sys.argv,
        "design": "query-fold (ADR0021 section B): head f filters fold-f queries and their library",
        "population": {"path": str(a.population), "sha256": sha256_file(a.population),
                       "slides": len(pop), "patients": int(pop.patient_id.nunique()),
                       "groups": int(pop.diagnostic_group.nunique()),
                       "cross_label_patients": int((pop.groupby("patient_id").diagnostic_group.nunique() > 1).sum())},
        "arms": [{"key": x.key, "matrix": str(x.matrix), "matrix_sha256": sha256_file(x.matrix),
                  "rows": str(x.rows) if x.rows else None,
                  "rows_sha256": sha256_file(x.rows) if x.rows else None} for x in arms],
        "bank_summaries_sha256": banks,
        "code_sha256": {str(c): sha256_file(c) for c in code_paths()},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]},
        "policy": {
            "cutoffs": list(CUTOFFS), "endpoints": ["hit", "precision", "hit_yield", "coverage"],
            "precision_cutoffs_reported": list(QF_PRECISION_CUTOFFS),
            "precision_at_1": "omitted from estimates/contrasts: identical to hit@1 by definition "
                              "(still in query_metrics.csv.gz)",
            "precision_denominator": "k (short lists keep k)",
            "bootstrap": {"count": a.draw_count, "seed": a.seed, "generator": "PCG64",
                          "unit": "patient, one multiplicity across all of the patient's slides",
                          "quantiles": [0.025, 0.975], "quantile_method": "linear",
                          "draws_per_population": "make_patient_draws(sorted included patients, seed)"},
            "weighting": "slide-weighted group means; equal-weight macro over groups",
            "same_patient": "excluded by population patient_id identity (all slides of the query's patient, "
                            "cross-label included); matrices are _no_same_patient form",
            "tie_rules": TIE_RULES, "primary_tie_rule": "stable_reference_index",
            "non_return": "fewer than the minimum retained tiles under the head of the query's own fold: "
                          "empty_query; in coverage/hit-yield denominators only",
            "library": {"all_tissue": "the 2,604-slide population, unfiltered",
                        "query_only": "the 2,604-slide population, unfiltered (saved bank)",
                        "both_filtered": "population slides returned by the query fold's head (per fold)"},
            "contrasts": "A minus B; conditional endpoints on the both-returned population (own draw table); "
                         "hit_yield on the intended 2,604. Pairs: each filtered arm minus all_tissue, "
                         "query_only minus both_filtered, budget 5 minus budget 100 within each filtered arm",
            "coverage_csv": "group rows are slide means within a group; the ALL row is a slide-pooled mean "
                            "(level pooled_slides_NOT_macro), not the equal-weight macro estimate",
            "csv_clamp": f"|x| < {CLAMP_ABS} written as 0.0 in CSV point/ci columns only",
            "no_p_values": True,
        },
        "wall_seconds_before_write": round(time.time() - t0, 1),
    }
    files = write_outputs_qf(result, a.out_dir, meta)
    print(json.dumps({"out_dir": str(a.out_dir), "files": files,
                      "matrix_diagnostics": result["matrix_diagnostics"]}, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "combine-qf":
        return main_combine_qf(args[1:])
    if args and args[0] == "evaluate-qf":
        return main_evaluate_qf(args[1:])
    env_out = os.environ.get("PROSPER_OUT")
    out = Path(env_out) if env_out else None
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--population", type=Path, default=(out / "r2_1_mosaic_cpu_20261003" / "attempt_001"
                                                        / "retrieval_population.csv") if out else None)
    p.add_argument("--standard", action="store_true",
                   help="all-tissue OT/Yottixel baselines + both-filtered budgets 5/100 (default paths)")
    p.add_argument("--arm", nargs=5, action="append", default=[],
                   metavar=("NAME", "METHOD", "BUDGET", "MATRIX", "SUMMARY"),
                   help=f"arm; BUDGET/SUMMARY '-' for none. NAME {BASELINE_ARM} is the contrast baseline")
    p.add_argument("--out-dir", type=Path,
                   default=out / "r2_1_filter_20261003" / "evaluation" if out else None)
    p.add_argument("--draw-count", type=int, default=5000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--allow-used-sentinel", action="store_true")
    a = p.parse_args(argv)
    if a.population is None or a.out_dir is None:
        p.error("export PROSPER_OUT or pass --population/--out-dir")
    arms = standard_arms(out) if a.standard else []
    for name, method, budget, matrix, summary in a.arm:
        arms.append(Arm(name, method, "" if budget == "-" else budget, Path(matrix),
                        None if summary == "-" else Path(summary)))
    if not arms:
        p.error("no arms: pass --standard and/or --arm")
    t0 = time.time()
    pop = load_population(a.population)
    result = evaluate(pop, arms, draw_count=a.draw_count, seed=a.seed,
                      allow_used_sentinel=a.allow_used_sentinel)
    meta = {
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"), "argv": sys.argv,
        "population": {"path": str(a.population), "sha256": sha256_file(a.population),
                       "slides": len(pop), "patients": int(pop.patient_id.nunique()),
                       "groups": int(pop.diagnostic_group.nunique()),
                       "cross_label_patients": int((pop.groupby("patient_id").diagnostic_group.nunique() > 1).sum())},
        "arms": [{"key": x.key, "matrix": str(x.matrix), "matrix_sha256": sha256_file(x.matrix),
                  "summary": str(x.summary) if x.summary else None,
                  "summary_sha256": sha256_file(x.summary) if x.summary else None} for x in arms],
        "code_sha256": {str(c): sha256_file(c) for c in code_paths()},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]},
        "policy": {
            "cutoffs": list(CUTOFFS), "endpoints": ["hit", "precision", "hit_yield", "coverage"],
            "precision_denominator": "k (short lists keep k)",
            "bootstrap": {"count": a.draw_count, "seed": a.seed, "generator": "PCG64",
                          "unit": "patient, one multiplicity across all of the patient's slides",
                          "quantiles": [0.025, 0.975], "quantile_method": "linear",
                          "draws_per_population": "make_patient_draws(sorted included patients, seed)"},
            "weighting": "slide-weighted group means; equal-weight macro over groups",
            "same_patient": "excluded by population patient_id identity; matrices are _no_same_patient form",
            "tie_break": "stable ascending distance, population reference_index order",
            "non_return": "slide not written by F1a: empty_query; in coverage/hit-yield denominators only",
            "library": "returned (written) population slides of the arm",
            "contrasts": "arm minus all_tissue of the same method; conditional endpoints on both-returned; "
                         "hit_yield on intended",
            "no_p_values": True,
        },
        "wall_seconds_before_write": round(time.time() - t0, 1),
    }
    files = write_outputs(result, a.out_dir, meta)
    print(json.dumps({"out_dir": str(a.out_dir), "files": files,
                      "matrix_diagnostics": result["matrix_diagnostics"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
