"""R2.1 P2: perturbation SAMPLER aggregation, identity references, L1 retrieval, paired evaluation.

Inputs (per encoder): the P1 extractor outputs ``features/<encoder>/<slide>.pkl`` =
``(dict condition -> float32 (n_tiles, D), coords)`` plus ``<slide>.record.json`` (pickle sha256,
``reinhard_nonfinite_tile_indices``). Population: ``perturb_attempt_001/queries.csv`` (2,778
slides, 15 diagnostic groups; identical to references.csv).

Design (agreed; see the implementation report):

1. SAMPLER per encoder/slide/condition = ``np.percentile(X, [5..95], axis=0)`` -> (10, D),
   computed in float64 exactly as the historical harness, stored as float32. Tiles listed in
   ``reinhard_nonfinite_tile_indices`` are excluded from the ``reinhard`` arm only; any row with
   a non-finite value is excluded from its arm. Counts are recorded. A slide/condition with no
   usable tile is a NON-RETURNED query for that condition (kept in coverage denominators).
2. References = identity-arm SAMPLER of every population slide (same encoder, same recipe).
   Queries = every condition's SAMPLER (identity included as the control arm).
3. Distance = float64 ``cdist(..., 'cityblock')`` between flattened SAMPLER vectors; self and
   same-patient (population ``patient_id``) cells set to the 1e6 sentinel; ranking excludes them
   by patient identity (never by threshold), ascending distance, ties broken stably in population
   order. Ranking uses the float64 matrix; the stored ``distances/<condition>.npy`` is its float32
   copy (rounding diagnostics in run_record.json).
4. Metrics: hit@k and precision@k (c/k), k = 1,3,5,10, slide-weighted group means (sorted
   groups) and equal-weight macro, with the frozen whole-patient bootstrap of
   ``r2_1_hoptimus_main`` (PCG64 seed 123, 5,000 draws, one multiplicity per patient, linear
   2.5/97.5% percentiles); draws are generated for exactly the included patients of each
   summarized population. Paired slide-matched deltas (condition minus identity) use the
   both-returned population and one shared draw table. No p-values, no winner statements.
5. Secondary, clearly labelled: the historical ``benchmarking_perturbations.eval_new_model``
   top-k majority-vote accuracy and macro F1/precision/recall on the same distance matrices,
   after its metadata filter (groups with < 40 slides and MISC removed).

Sub-commands:
  run              production / pilot evaluation (refuses incomplete inputs unless --allow-partial)
  hist-sampler     fidelity stage 1: SAMPLER of the historical perturbed pickles (one shard)
  hist-l1          fidelity stage 2: L1 to the historical references for one arm
  hist-eval        fidelity stage 3: compare with stored historical matrices/metrics and Fig. 4A
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import pickle
import resource
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import r2_1_hoptimus_main as kernel
from . import r2_1_perturb_sampler as S

CONDITIONS: Tuple[str, ...] = (
    "identity", "rotate_90", "rotate_180", "rotate_270", "flip_horizontal", "flip_vertical",
    "brightness", "saturation", "contrast", "spatial_shift", "reinhard", "gaussian_blur")
IDENTITY = "identity"
ENCODER_DIM = {"h_optimus_1": 1536, "uni2": 1536, "virchow2": 2560}
VIRCHOW2_CLS_DIM = 1280
CUTOFFS = (1, 3, 5, 10)
ENDPOINTS = ("hit", "precision")
DEPTH = 10
MACRO = "MACRO"

ESTIMATE_HEADER = ("condition", "group", "metric", "k", "estimate", "ci_low", "ci_high",
                   "n_slides", "n_patients", "population", "point_status", "point_reason",
                   "ci_status", "ci_reason", "zero_support_replicates")
DELTA_HEADER = ("condition", "group", "metric", "k", "delta", "ci_low", "ci_high", "n_pairs",
                "n_patients", "estimate_condition", "estimate_identity", "population",
                "point_status", "point_reason", "ci_status", "ci_reason",
                "zero_support_replicates")
COVERAGE_HEADER = ("condition", "group", "intended", "returned", "non_returns",
                   "non_return_slides", "coverage", "tiles_total", "tiles_excluded_listed",
                   "tiles_excluded_nonfinite")
HIST_HEADER = ("label", "condition", "k", "n_queries", "n_references", "accuracy", "macro_f1",
               "macro_precision", "macro_recall", "delta_accuracy_vs_identity",
               "delta_macro_f1_vs_identity", "delta_macro_precision_vs_identity",
               "delta_macro_recall_vs_identity", "delta_population")
SAMPLER_INDEX_HEADER = ("condition", "row", "population_index", "slide_id", "patient_id",
                        "diagnostic_group", "n_tiles", "n_excluded_listed",
                        "n_excluded_nonfinite", "n_used", "returned")
QUERY_HEADER = ("condition", "population_index", "query", "patient_id", "diagnostic_group", "k",
                "status", "returned_depth", "conditional_hit", "conditional_precision",
                "n_candidates", "boundary_tie")
RANK_HEADER = ("condition", "query", "rank", "neighbour", "neighbour_group", "relevant",
               "distance")
HIST_LABEL = ("SECONDARY historical metric: benchmarking_perturbations.eval_new_model semantics "
              "(argpartition top-k, generous majority vote, sklearn macro scores, groups with "
              "<40 slides and MISC removed); NOT the primary hit@k/precision@k endpoint")


class EvalError(RuntimeError):
    """An input violates the declared population / feature / output specification."""


class IncompleteInputs(EvalError):
    """Some population slide x encoder outputs are missing and --allow-partial was not given."""


# ---------------------------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_bytes(header, rows, compressed: bool = False) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(["" if v is None else repr(float(v)) if isinstance(v, (float, np.floating))
                    else int(v) if isinstance(v, (np.integer, np.bool_)) else v for v in r])
    data = buf.getvalue().encode()
    return gzip.compress(data, mtime=0) if compressed else data


def _npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a))
    return buf.getvalue()


class Writer:
    """Exclusive-create writer: refuses a non-empty directory and never replaces a file."""

    def __init__(self, out_dir: Path):
        if out_dir.exists() and any(out_dir.iterdir()):
            raise EvalError(f"{out_dir} exists and is not empty; refusing to overwrite "
                            f"(choose a new output directory)")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.root = out_dir
        self.files: Dict[str, dict] = {}

    def put(self, name: str, data: bytes) -> None:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        self.files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def default_out() -> Optional[Path]:
    v = os.environ.get("PROSPER_OUT")
    return Path(v) if v else None


# ---------------------------------------------------------------------------------------------
# Population and inputs
# ---------------------------------------------------------------------------------------------

def load_population(path: Path) -> pd.DataFrame:
    pop = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"population_index", "slide_id", "patient_id", "diagnostic_group"}
    if not need.issubset(pop.columns):
        raise EvalError(f"population lacks {sorted(need - set(pop.columns))}")
    pop["population_index"] = pop["population_index"].astype(int)
    pop = pop.sort_values("population_index", kind="mergesort").reset_index(drop=True)
    if not (pop.population_index.to_numpy() == np.arange(len(pop))).all():
        raise EvalError("population_index must be 0..n-1")
    if pop.slide_id.duplicated().any():
        raise EvalError("duplicate population slide_id")
    for c in ("slide_id", "patient_id", "diagnostic_group"):
        if (pop[c].str.strip() == "").any():
            raise EvalError(f"empty {c} in population")
    return pop


def parse_subset(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    p = Path(value)
    if p.is_file():
        items = [x.strip() for x in p.read_text().replace(",", "\n").splitlines()]
    else:
        items = [x.strip() for x in value.split(",")]
    items = [x for x in items if x]
    if len(set(items)) != len(items):
        raise EvalError("--slides-subset has duplicates")
    return items


def input_paths(features_root: Path, encoder: str, sid: str) -> Tuple[Path, Path]:
    d = features_root / encoder
    return d / f"{sid}.pkl", d / f"{sid}.record.json"


def check_inputs(pop: pd.DataFrame, features_root: Path, encoder: str) -> List[str]:
    """Slides whose pickle+record pair is not both present (a pickle without its record is
    treated as missing: the extractor publishes the record last)."""
    missing = []
    for sid in pop.slide_id:
        pkl, rec = input_paths(features_root, encoder, sid)
        if not (pkl.is_file() and rec.is_file()):
            missing.append(sid)
            continue
        try:
            r = json.loads(rec.read_text())
        except (OSError, ValueError):
            missing.append(sid)
            continue
        if list(r.get("conditions", [])) != list(CONDITIONS) or r.get("slide_id") != sid:
            missing.append(sid)
    return missing


def _slide_task(args):
    """Worker: verify one slide's pickle against its record and compute all SAMPLER arms."""
    (sid, pkl, rec_path, encoder, conditions, cls_only, expected_dim) = args
    rec_bytes = Path(rec_path).read_bytes()
    rec = json.loads(rec_bytes)
    if rec.get("slide_id") != sid or rec.get("encoder") != encoder:
        raise EvalError(f"{rec_path}: slide/encoder mismatch")
    if list(rec.get("conditions", [])) != list(CONDITIONS):
        raise EvalError(f"{rec_path}: conditions differ from the 12 declared")
    if int(rec.get("dim", -1)) != expected_dim:
        raise EvalError(f"{rec_path}: dim {rec.get('dim')} != {expected_dim}")
    data = Path(pkl).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != rec.get("pickle_sha256") or len(data) != rec.get("pickle_bytes"):
        raise EvalError(f"{pkl}: sha256/size differ from its record")
    obj = pickle.loads(data)
    del data
    if not (isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], dict)):
        raise EvalError(f"{pkl}: not (dict, coords)")
    feats = obj[0]
    if set(feats) != set(CONDITIONS):
        raise EvalError(f"{pkl}: condition keys differ")
    n = int(rec["n_tiles"])
    listed = [int(i) for i in rec.get("reinhard_nonfinite_tile_indices", [])]
    out = {}
    for c in conditions:
        x = feats[c]
        if not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.shape != (n, expected_dim):
            raise EvalError(f"{pkl}[{c}]: expected float32 ({n}, {expected_dim}), "
                            f"got {getattr(x, 'dtype', None)} {getattr(x, 'shape', None)}")
        if cls_only:
            x = x[:, :VIRCHOW2_CLS_DIM]
        r = S.sampler(x, listed if c == "reinhard" else None)
        out[c] = r
    meta = {"slide_id": sid, "pickle_sha256": digest, "record_sha256":
            hashlib.sha256(rec_bytes).hexdigest(), "n_tiles": n,
            "reinhard_nonfinite_listed": len(listed)}
    return sid, out, meta


def compute_samplers(pop: pd.DataFrame, features_root: Path, encoder: str,
                     conditions: Sequence[str], cls_only: bool, workers: int):
    """Returns ({condition: (N, 10, D) float64 with NaN rows for non-returns},
    sampler-index rows, per-slide input metadata)."""
    # Spawned (not forked) workers: no inherited threads/locks and no mapping of the parent's
    # large (N, 10, D) arrays. Task arguments carry everything a worker needs.
    pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None
    try:
        return _compute_samplers(pop, features_root, encoder, conditions, cls_only, pool)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


def _compute_samplers(pop, features_root, encoder, conditions, cls_only, pool):
    dim = VIRCHOW2_CLS_DIM if cls_only else ENCODER_DIM[encoder]
    n = len(pop)
    emb = {c: np.full((n, len(S.PERCENTILES), dim), np.nan, dtype=np.float64)
           for c in conditions}
    counts = {c: np.zeros((n, 4), dtype=np.int64) for c in conditions}  # tiles, listed, nonfin, used
    tasks = []
    for sid in pop.slide_id:
        pkl, rec = input_paths(features_root, encoder, sid)
        tasks.append((sid, str(pkl), str(rec), encoder, tuple(conditions), cls_only,
                      ENCODER_DIM[encoder]))
    pos = {s: i for i, s in enumerate(pop.slide_id)}
    metas = [None] * n

    def absorb(res):
        sid, out, meta = res
        i = pos[sid]
        metas[i] = meta
        for c, r in out.items():
            counts[c][i] = (r.n_tiles, r.n_excluded_listed, r.n_excluded_nonfinite, r.n_used)
            if r.embedding is not None:
                emb[c][i] = r.embedding

    if pool is None:
        for t in tasks:
            absorb(_slide_task(t))
    else:
        for res in pool.imap_unordered(_slide_task, tasks, chunksize=1):
            absorb(res)
    return emb, counts, metas


# ---------------------------------------------------------------------------------------------
# Ranking and inference
# ---------------------------------------------------------------------------------------------

def rank_condition(pop: pd.DataFrame, dist: np.ndarray, returned: np.ndarray,
                   library: np.ndarray, dist32: Optional[np.ndarray] = None):
    """Per-query top-10 relevance over eligible references (library, other patients)."""
    groups = pop.diagnostic_group.to_numpy(dtype=object)
    patients = pop.patient_id.to_numpy(dtype=object)
    statuses, relevance, ranks, n_cand, ties = [], [], [], [], []
    diag = {"used_pairs": 0, "used_1e6_entries": 0, "excluded_same_patient_cells_not_1e6": 0,
            "float32_top10_order_differs": 0}
    for i in range(len(pop)):
        if not returned[i]:
            statuses.append("empty_query")
            relevance.append(np.zeros(0, dtype=bool))
            ranks.append([])
            n_cand.append(0)
            ties.append({k: False for k in CUTOFFS})
            continue
        same = library & (patients == patients[i])
        diag["excluded_same_patient_cells_not_1e6"] += int(
            np.count_nonzero(dist[i, same] != S.SENTINEL))
        cand = np.flatnonzero(library & (patients != patients[i]))
        if cand.size == 0:
            statuses.append("empty_reference_library")
            relevance.append(np.zeros(0, dtype=bool))
            ranks.append([])
            n_cand.append(0)
            ties.append({k: False for k in CUTOFFS})
            continue
        scores = dist[i, cand]
        if not np.all(np.isfinite(scores)):
            raise EvalError(f"nonfinite distance on a used pair for query {pop.slide_id[i]}")
        diag["used_1e6_entries"] += int(np.count_nonzero(scores == S.SENTINEL))
        diag["used_pairs"] += int(cand.size)
        top, top_d, srt = S.rank_query(scores, cand, DEPTH)
        if dist32 is not None:
            top32, _, _ = S.rank_query(dist32[i, cand].astype(np.float64), cand, DEPTH)
            if not np.array_equal(top, top32):
                diag["float32_top10_order_differs"] += 1
        relevance.append(groups[top] == groups[i])
        ranks.append(list(zip(top.tolist(), top_d.tolist())))
        n_cand.append(int(cand.size))
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
        return [{"n_patients": len(k),
                 "patients_sha256": hashlib.sha256("\n".join(k).encode()).hexdigest(),
                 "multiplicities_sha256": hashlib.sha256(
                     np.ascontiguousarray(d.multiplicities).tobytes()).hexdigest()}
                for k, d in self._d.items()]


def _est_cells(e: kernel.Estimate):
    return (e.point, e.ci_low, e.ci_high, e.n_queries, e.n_patients)


def _status_cells(e: kernel.Estimate):
    return (e.point_status, e.point_reason, e.ci_status, e.ci_reason, e.zero_support_replicates)


def query_vectors(statuses, relevance):
    n = len(statuses)
    vec = {}
    for k in CUTOFFS:
        hit = np.zeros(n)
        prec = np.zeros(n)
        depth = [0] * n
        for i in range(n):
            m = kernel.query_metrics(relevance[i], k, statuses[i])
            hit[i] = m.conditional_hit or 0.0
            prec[i] = m.conditional_precision or 0.0
            depth[i] = m.returned_depth
        vec[("hit", k)], vec[("precision", k)], vec[("depth", k)] = hit, prec, depth
    return vec


def evaluate(pop: pd.DataFrame, emb: Dict[str, np.ndarray], *, draw_count: int = 5000,
             seed: int = 123, threads: int = 1, keep_float32: bool = True,
             hist_counts_from: Optional[Sequence[str]] = None) -> dict:
    """Distances, rankings, bootstrap estimates, paired deltas, coverage and the historical
    secondary table for all conditions in ``emb`` (which must include identity)."""
    if IDENTITY not in emb:
        raise EvalError("the identity condition is required (it defines the references)")
    conds = [c for c in CONDITIONS if c in emb]
    n = len(pop)
    qids = pop.slide_id.tolist()
    pids = pop.patient_id.tolist()
    labels = pop.diagnostic_group.tolist()
    groups = tuple(sorted(set(labels)))
    flat = {c: emb[c].reshape(n, -1) for c in conds}
    returned = {c: np.isfinite(flat[c]).all(axis=1) for c in conds}
    library = returned[IDENTITY]
    lib_idx = np.flatnonzero(library)
    draws = DrawCache(draw_count, seed)
    # Group sizes for the historical min-40 filter come from the full intended population, so a
    # --slides-subset pilot keeps the production filter.
    hist_keep = S.historical_keep(labels, counts_from=hist_counts_from)
    hist_labels = [S.historical_label(x) for x in labels]

    dist64, dist32, per, diags, timings = {}, {}, {}, {}, {}
    query_rows, rank_rows = [], []
    for c in conds:
        t0 = time.time()
        d = np.full((n, n), np.nan, dtype=np.float64)
        rows = np.flatnonzero(returned[c])
        if rows.size and lib_idx.size:
            d[np.ix_(rows, lib_idx)] = S.l1_distances(flat[c][rows], flat[IDENTITY][lib_idx],
                                                      threads=threads)
        t_l1 = time.time() - t0
        both = returned[c] & library
        diag_vals = d[np.flatnonzero(both), np.flatnonzero(both)]
        dd = {"diagonal_before_sentinel_max_abs": float(np.abs(diag_vals).max())
              if diag_vals.size else None,
              "diagonal_before_sentinel_n_zero": int(np.count_nonzero(diag_vals == 0))}
        p = np.asarray(pids, dtype=object)
        same = p[rows][:, None] == p[lib_idx][None, :]
        if rows.size and lib_idx.size:
            block = d[np.ix_(rows, lib_idx)]
            block[same] = S.SENTINEL
            d[np.ix_(rows, lib_idx)] = block
        dd["sentinel_cells"] = int(np.count_nonzero(same))
        d32 = d.astype(np.float32)
        fin = np.isfinite(d) & (d != S.SENTINEL)
        dd["float32_max_abs_rounding"] = float(np.abs(d32[fin].astype(np.float64) - d[fin]).max()) \
            if fin.any() else None
        st, rel, rk, ncand, ties, rdiag = rank_condition(pop, d, returned[c], library, d32)
        dd.update(rdiag)
        dd["boundary_ties"] = {str(k): int(sum(t[k] for t in ties)) for k in CUTOFFS}
        dd["returned"] = int(returned[c].sum())
        dd["intended"] = n
        diags[c] = dd
        vec = query_vectors(st, rel)
        per[c] = dict(statuses=st, vec=vec, returned=np.array([s == "returned" for s in st]))
        for i in range(n):
            for k in CUTOFFS:
                query_rows.append((c, i, qids[i], pids[i], labels[i], k, st[i],
                                   vec[("depth", k)][i],
                                   vec[("hit", k)][i] if st[i] == "returned" else None,
                                   vec[("precision", k)][i] if st[i] == "returned" else None,
                                   ncand[i], int(ties[i][k])))
            for r, (j, dv) in enumerate(rk[i], start=1):
                rank_rows.append((c, qids[i], r, qids[j], labels[j], int(labels[j] == labels[i]),
                                  dv))
        dist64[c] = d
        if keep_float32:
            dist32[c] = d32
        timings[c] = {"l1_seconds": round(t_l1, 2), "total_seconds": round(time.time() - t0, 2)}

    est_rows, delta_rows = [], []
    for c in conds:
        ret = per[c]["returned"]
        d_ret = draws.get([pp for pp, r in zip(pids, ret) if r])
        for ep in ENDPOINTS:
            for k in CUTOFFS:
                s = kernel.summarize_metric(qids, pids, labels, groups, per[c]["vec"][(ep, k)],
                                            ret, d_ret)
                for g, e in zip((*groups, MACRO), (*s.group_estimates, s.macro)):
                    est_rows.append((c, g, ep, k, *_est_cells(e), "returned", *_status_cells(e)))
    for c in conds:
        if c == IDENTITY:
            continue
        both = per[c]["returned"] & per[IDENTITY]["returned"]
        d_both = draws.get([pp for pp, r in zip(pids, both) if r])
        for ep in ENDPOINTS:
            for k in CUTOFFS:
                ps = kernel.summarize_paired(qids, pids, labels, groups, per[c]["vec"][(ep, k)],
                                             per[IDENTITY]["vec"][(ep, k)], both, d_both)
                for g, e, a, b in zip((*groups, MACRO),
                                      (*ps.difference.group_estimates, ps.difference.macro),
                                      (*ps.arm_a.group_estimates, ps.arm_a.macro),
                                      (*ps.arm_b.group_estimates, ps.arm_b.macro)):
                    delta_rows.append((c, g, ep, k, e.point, e.ci_low, e.ci_high, e.n_queries,
                                       e.n_patients, a.point, b.point, "both_returned",
                                       *_status_cells(e)))

    hist_rows = historical_table(pop, dist64, per, library, hist_keep, hist_labels, conds)
    return dict(groups=groups, conds=conds, dist32=dist32, per=per, returned=returned,
                library=library, estimates=est_rows, deltas=delta_rows, historical=hist_rows,
                query_rows=query_rows, rank_rows=rank_rows, diagnostics=diags, timings=timings,
                draws=draws, full_draws=draws.get(pids))


def historical_table(pop, dist64, per, library, keep, hist_labels, conds):
    rows = []
    hl = np.asarray(hist_labels, dtype=object)
    scores = {}
    for c in conds:
        q = np.flatnonzero(per[c]["returned"] & keep)
        r = np.flatnonzero(library & keep)
        for k in CUTOFFS:
            if r.size <= k or q.size == 0:
                # argpartition(row, k) needs more than k references (always true in production)
                scores[(c, k)] = None
                continue
            preds, _ = S.historical_predict(dist64[c][np.ix_(q, r)], hl[q].tolist(),
                                            hl[r].tolist(), k)
            scores[(c, k)] = (q, dict(zip(q.tolist(), preds)))
    nan4 = [float("nan")] * 4
    for c in conds:
        for k in CUTOFFS:
            if scores[(c, k)] is None or (c != IDENTITY and scores[(IDENTITY, k)] is None):
                rows.append((HIST_LABEL, c, k, int(np.count_nonzero(per[c]["returned"] & keep)),
                             int(np.count_nonzero(library & keep)), *nan4, None, None, None,
                             None, "not_estimable_references_le_k"))
                continue
            q, pred = scores[(c, k)]
            sc = S.historical_scores(hl[q].tolist(), [pred[i] for i in q.tolist()])
            delta = [None] * 4
            dpop = None
            if c != IDENTITY:
                qi, predi = scores[(IDENTITY, k)]
                common = sorted(set(q.tolist()) & set(qi.tolist()))
                if common == q.tolist() == qi.tolist():
                    base = S.historical_scores(hl[qi].tolist(), [predi[i] for i in qi.tolist()])
                    a, dpop = sc, "identical_query_sets"
                else:
                    a = S.historical_scores(hl[common].tolist(), [pred[i] for i in common])
                    base = S.historical_scores(hl[common].tolist(), [predi[i] for i in common])
                    dpop = f"both_returned_{len(common)}"
                delta = [a[m] - base[m] for m in ("accuracy", "f1", "precision", "recall")]
            rows.append((HIST_LABEL, c, k, int(q.size), int(np.count_nonzero(library & keep)),
                         sc["accuracy"], sc["f1"], sc["precision"], sc["recall"], *delta, dpop))
    return rows


def coverage_rows(pop, conds, returned, counts):
    rows = []
    g_arr = pop.diagnostic_group.to_numpy()
    for c in conds:
        for g in (*sorted(set(g_arr)), "ALL"):
            sel = np.ones(len(pop), bool) if g == "ALL" else g_arr == g
            ret = returned[c]
            nr = [pop.slide_id[i] for i in np.flatnonzero(sel & ~ret)]
            cnt = counts[c][sel]
            rows.append((c, g, int(sel.sum()), int((sel & ret).sum()), len(nr), ";".join(nr),
                         float((sel & ret).sum() / sel.sum()), int(cnt[:, 0].sum()),
                         int(cnt[:, 1].sum()), int(cnt[:, 2].sum())))
    return rows


# ---------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------

def code_hashes() -> Dict[str, str]:
    return {str(p): sha256_file(p) for p in (Path(__file__).resolve(), Path(S.__file__).resolve(),
                                             Path(kernel.__file__).resolve())}


def run_encoder(encoder: str, *, pop_full: pd.DataFrame, population_path: Path,
                features_root: Path, out_dir: Path, conditions: Sequence[str],
                subset: Optional[List[str]], allow_partial: bool, cls_only: bool, workers: int,
                threads: int, draw_count: int, seed: int, argv: Sequence[str]) -> dict:
    t_start = time.time()
    if cls_only and encoder != "virchow2":
        raise EvalError("--virchow2-cls-only applies to virchow2 only")
    pop = pop_full
    if subset is not None:
        unknown = sorted(set(subset) - set(pop_full.slide_id))
        if unknown:
            raise EvalError(f"--slides-subset slides not in the population: {unknown[:10]}")
        pop = pop_full[pop_full.slide_id.isin(subset)].reset_index(drop=True)
    missing = check_inputs(pop, features_root, encoder)
    if missing and not allow_partial:
        raise IncompleteInputs(
            f"{encoder}: {len(missing)} of {len(pop)} slides lack a complete pickle+record pair "
            f"under {features_root / encoder} (first: {missing[:5]}); refusing to start without "
            f"--allow-partial")
    if missing:
        pop = pop[~pop.slide_id.isin(missing)].reset_index(drop=True)
    if len(pop) == 0:
        raise EvalError("no evaluable slides")
    writer = Writer(out_dir)
    t0 = time.time()
    emb, counts, metas = compute_samplers(pop, features_root, encoder, conditions, cls_only,
                                          workers)
    t_sampler = time.time() - t0
    n = len(pop)
    idx_rows = []
    for c in conditions:
        writer.put(f"sampler/{c}.npy", _npy_bytes(emb[c].astype(np.float32)))
        fin = np.isfinite(emb[c].reshape(n, -1)).all(axis=1)
        for i in range(n):
            t, li, nf, u = counts[c][i].tolist()
            idx_rows.append((c, i, int(pop.population_index[i]), pop.slide_id[i],
                             pop.patient_id[i], pop.diagnostic_group[i], t, li, nf, u,
                             int(fin[i])))
    writer.put("sampler_index.csv", _csv_bytes(SAMPLER_INDEX_HEADER, idx_rows))
    t1 = time.time()
    res = evaluate(pop, emb, draw_count=draw_count, seed=seed, threads=threads,
                   hist_counts_from=pop_full.diagnostic_group.tolist())
    t_eval = time.time() - t1
    for c in res["conds"]:
        writer.put(f"distances/{c}.npy", _npy_bytes(res["dist32"][c]))
    writer.put("estimates.csv", _csv_bytes(ESTIMATE_HEADER, res["estimates"]))
    writer.put("paired_deltas.csv", _csv_bytes(DELTA_HEADER, res["deltas"]))
    writer.put("coverage.csv", _csv_bytes(COVERAGE_HEADER, coverage_rows(
        pop, res["conds"], res["returned"], counts)))
    writer.put("historical_topk_accuracy.csv", _csv_bytes(HIST_HEADER, res["historical"]))
    writer.put("query_metrics.csv.gz", _csv_bytes(QUERY_HEADER, res["query_rows"], True))
    writer.put("ranked_lists.csv.gz", _csv_bytes(RANK_HEADER, res["rank_rows"], True))
    writer.put("patient_draws.npy", _npy_bytes(
        np.asarray(res["full_draws"].multiplicities, dtype=np.int64)))
    writer.put("input_manifest.csv", _csv_bytes(
        ("population_index", "slide_id", "pickle", "pickle_sha256_verified", "record",
         "record_sha256", "n_tiles", "reinhard_nonfinite_listed"),
        [(int(pop.population_index[i]), m["slide_id"],
          str(input_paths(features_root, encoder, m["slide_id"])[0]), m["pickle_sha256"],
          str(input_paths(features_root, encoder, m["slide_id"])[1]), m["record_sha256"],
          m["n_tiles"], m["reinhard_nonfinite_listed"]) for i, m in enumerate(metas)]))
    record = {
        "schema": "r2_1_perturb_eval_run_v1",
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"),
        "argv": list(argv), "encoder": encoder, "virchow2_cls_only": cls_only,
        "feature_dim_used": VIRCHOW2_CLS_DIM if cls_only else ENCODER_DIM[encoder],
        "conditions": list(res["conds"]),
        "population": {"path": str(population_path), "sha256": sha256_file(population_path),
                       "intended_slides": len(pop_full),
                       "evaluated_slides": n, "evaluated_patients": int(pop.patient_id.nunique()),
                       "groups": list(res["groups"])},
        "slides_subset": subset, "partial": bool(missing), "allow_partial": allow_partial,
        "missing_inputs": missing,
        "features_root": str(features_root / encoder),
        "input_manifest": "input_manifest.csv (every pickle sha256 verified against its record)",
        "code_sha256": code_hashes(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": __import__("scipy").__version__,
                     "sklearn": __import__("sklearn").__version__,
                     "python": sys.version.split()[0]},
        "policy": {
            "sampler": "np.percentile(X, [5,15,...,95], axis=0) in float64 on kept rows; stored "
                       "float32; no float16 cast",
            "tile_exclusion": "reinhard: tiles in record reinhard_nonfinite_tile_indices; every "
                              "condition: rows with any non-finite value",
            "non_return": "a slide/condition with zero usable tiles; empty_query status, kept in "
                          "coverage denominators",
            "references": "identity-arm SAMPLER of every evaluated slide with identity returned",
            "distance": "scipy cdist cityblock, float64, on flattened (10*D) vectors",
            "exclusion": "self and same population patient_id excluded by identity; cells set to "
                         "1e6 in the stored matrices",
            "tie_break": "stable ascending distance, population order",
            "precision_denominator": "k",
            "bootstrap": {"count": draw_count, "seed": seed, "generator": "PCG64",
                          "unit": "patient (one multiplicity across all of the patient's slides)",
                          "quantiles": [0.025, 0.975], "quantile_method": "linear",
                          "draws_per_population": "make_patient_draws(sorted included patients)"},
            "weighting": "slide-weighted group means (sorted groups); equal-weight macro",
            "paired": "condition minus identity on both-returned slides, one shared draw table",
            "historical_secondary": HIST_LABEL,
            "no_p_values": True},
        "draw_tables": res["draws"].describe(),
        "diagnostics": res["diagnostics"],
        "timings_seconds": {"sampler_stage": round(t_sampler, 1),
                            "evaluate_stage": round(t_eval, 1),
                            "per_condition": res["timings"],
                            "total_before_record": round(time.time() - t_start, 1)},
        "workers": workers, "threads": threads,
        "peak_rss_mb_main_process": round(peak_rss_mb(), 1),
        "outputs": dict(writer.files),
    }
    writer.put("run_record.json", json.dumps(record, indent=1, default=str).encode())
    writer.put("COMPLETE", json.dumps({"files": writer.files}, indent=1, sort_keys=True).encode())
    return record


def cmd_run(a) -> int:
    pop_full = load_population(a.population)
    conds = list(CONDITIONS) if not a.conditions else a.conditions.split(",")
    bad = set(conds) - set(CONDITIONS)
    if bad:
        raise EvalError(f"unknown conditions {sorted(bad)}")
    if IDENTITY not in conds:
        raise EvalError("--conditions must include identity (the reference arm)")
    conds = [c for c in CONDITIONS if c in conds]
    subset = parse_subset(a.slides_subset)
    status = 0
    for enc in a.encoders.split(","):
        if enc not in ENCODER_DIM:
            raise EvalError(f"unknown encoder {enc}")
        name = f"{enc}_cls_only" if a.virchow2_cls_only else enc
        out_dir = a.out_root / name
        print(f"[perturb-eval] {enc} -> {out_dir}", flush=True)
        rec = run_encoder(enc, pop_full=pop_full, population_path=a.population,
                          features_root=a.features_root, out_dir=out_dir, conditions=conds,
                          subset=subset, allow_partial=a.allow_partial,
                          cls_only=a.virchow2_cls_only, workers=a.workers, threads=a.threads,
                          draw_count=a.draw_count, seed=a.seed,
                          argv=getattr(a, "argv_full", sys.argv))
        print(json.dumps({"encoder": enc, "out_dir": str(out_dir),
                          "evaluated_slides": rec["population"]["evaluated_slides"],
                          "partial": rec["partial"], "timings": rec["timings_seconds"],
                          "peak_rss_mb": rec["peak_rss_mb_main_process"]}, indent=1), flush=True)
    return status


# ---------------------------------------------------------------------------------------------
# Historical fidelity (stages hist-sampler / hist-l1 / hist-eval)
# ---------------------------------------------------------------------------------------------

HIST_ROOT = Path("/PROSPER_WORKSPACE")
HIST_CONDITIONS = ("rotate_90", "rotate_180", "rotate_270", "reinhard", "flip_h", "flip_v",
                   "brightness", "saturation", "contrast", "blur", "spatial_shift")
HIST_EVAL_ORDER = ("rotate_90", "rotate_180", "rotate_270", "flip_h", "flip_v", "reinhard",
                   "brightness", "saturation", "contrast", "blur", "spatial_shift")
HIST_TO_NEW = {"flip_h": "flip_horizontal", "flip_v": "flip_vertical", "blur": "gaussian_blur"}
# Published Fig. 4A annotations (3 d.p., transcribed from docs/prosper_submission/fig/Figure_4.png),
# rows in HIST_EVAL_ORDER, columns Top-1/3/5/10: macro F1 (perturbed) minus macro F1 (original).
FIG4A = {
    "virchow2": [[-0.012, -0.017, -0.014, -0.017], [-0.012, -0.010, -0.012, -0.018],
                 [-0.015, -0.015, -0.013, -0.017], [-0.020, -0.017, -0.014, -0.016],
                 [-0.011, -0.010, -0.012, -0.017], [-0.040, -0.025, -0.031, -0.029],
                 [-0.016, -0.016, -0.013, -0.016], [-0.016, -0.018, -0.008, -0.013],
                 [-0.019, -0.023, -0.010, -0.014], [-0.018, -0.021, -0.010, -0.018],
                 [-0.021, -0.019, -0.013, -0.016]],
    "uni2": [[-0.035, -0.027, -0.031, -0.009], [-0.028, -0.023, -0.024, -0.009],
             [-0.037, -0.026, -0.027, -0.005], [-0.025, -0.018, -0.019, -0.014],
             [-0.026, -0.024, -0.023, -0.008], [-0.055, -0.027, -0.011, -0.020],
             [-0.031, -0.020, -0.014, -0.010], [-0.021, -0.020, -0.022, -0.012],
             [-0.026, -0.018, -0.020, -0.013], [-0.024, -0.015, -0.024, -0.016],
             [-0.024, -0.020, -0.018, -0.013]],
}


def hist_paths(encoder: str) -> dict:
    r, fg = HIST_ROOT / "comet-robustness", HIST_ROOT / "comet-feature-gen"
    base = {"uni2": "all_tiles", "virchow2": "all_tiles_old_good"}[encoder]
    return {
        "raw": r / "data" / f"tmp_final_{encoder}",
        "raw_reinhard": (r / "data" / "tmp_final_stain_normalization_only"
                         if encoder == "virchow2" else None),
        "sampled": r / "feature" / f"tmp_sampled_features_perturbations_{encoder}",
        "refs": fg / f"comet-{encoder}" / "output" / "sampler",
        "main_raw": fg / f"comet-{encoder}" / "data" / "tmp_final",
        "stored_matrix": r / "output" / f"similarity_matrix_sampler_perturbation_{encoder}.pkl",
        "metrics": r / "fig" / f"benchmarking_results_sampler_{encoder}",
        "baseline": fg / f"comet-{encoder}" / "benchmark_final" / "sampler" / "all_slides" / base,
        "baseline_matrix": (fg / f"comet-{encoder}" / "output" / "similarity_matrix" /
                            f"combined_similarity_matrix_{encoder}_sampler_no_same_patient.pkl"),
        "metadata": HIST_ROOT / "comet-new-scans" / "data" / "final_slide_list" /
        "final_scan_id_disease_map.csv",
        "patient_map": fg / "comet-analysis" / "data" /
        "final_scan_id_disease_map_with_sample_id_patient_id.csv",
    }


def hist_slide_ids(encoder: str) -> List[str]:
    """Sorted glob order of the historical sampled-feature files (the notebook's row order)."""
    files = sorted(hist_paths(encoder)["sampled"].glob("*.pkl"), key=lambda p: str(p))
    return [p.name.split(".")[0] for p in files]


def _hist_slide(args):
    encoder, sid = args
    P = hist_paths(encoder)
    raw = pickle.load(open(P["raw"] / f"{sid}.pkl", "rb"))[0]
    if P["raw_reinhard"] is not None:
        rr = pickle.load(open(P["raw_reinhard"] / f"{sid}.pkl", "rb"))[0]
    stored = pickle.load(open(P["sampled"] / f"{sid}.pkl", "rb"))
    out, cmp, nonfin = {}, {}, {}
    for c in HIST_CONDITIONS:
        x = np.asarray(raw[c])
        if x.size == 0 and P["raw_reinhard"] is not None and c == "reinhard":
            x = np.asarray(rr[c])
        if encoder == "virchow2":
            x = x[:, :VIRCHOW2_CLS_DIM]
        nonfin[c] = int((~np.isfinite(x)).any(axis=1).sum())
        r = S.sampler(x)
        out[c] = r.embedding
        st = np.asarray(stored[c])
        cmp[c] = {"equal": bool(r.embedding is not None and np.array_equal(r.embedding, st)),
                  "max_abs": float(np.nanmax(np.abs(r.embedding - st)))
                  if r.embedding is not None else None,
                  "used": r.n_used, "n": r.n_tiles}
    m = pickle.load(open(P["main_raw"] / f"{sid}.pkl", "rb"))
    mx = np.asarray(m[0] if isinstance(m, tuple) else m)
    ref = np.asarray(pickle.load(open(P["refs"] / f"{sid}.pkl", "rb")))
    out["identity_fp32"] = S.sampler(mx.astype(np.float32)).embedding
    fp16 = np.percentile(mx.astype(np.float16), S.PERCENTILES, axis=0)
    cmp["main_fp16_equals_reference"] = bool(np.array_equal(fp16, ref))
    cmp["main_dtype"] = str(mx.dtype)
    return sid, out, cmp, nonfin


def cmd_hist_sampler(a) -> int:
    ids = hist_slide_ids(a.encoder)
    shard = ids[a.shard::a.n_shards]
    t0 = time.time()
    res = [_hist_slide((a.encoder, s)) for s in shard]
    w = Writer(a.out_dir / a.encoder / f"shard_{a.shard:03d}")
    for c in (*HIST_CONDITIONS, "identity_fp32"):
        w.put(f"{c}.npy", _npy_bytes(np.stack([r[1][c] for r in res])))
    rec = {"encoder": a.encoder, "shard": a.shard, "n_shards": a.n_shards,
           "slides": [r[0] for r in res], "comparisons": {r[0]: r[2] for r in res},
           "raw_nonfinite_rows": {r[0]: r[3] for r in res},
           "seconds": round(time.time() - t0, 1), "peak_rss_mb": round(peak_rss_mb(), 1),
           "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"),
           "code_sha256": code_hashes()}
    w.put("shard.json", json.dumps(rec, indent=1).encode())
    w.put("COMPLETE", json.dumps(w.files, indent=1).encode())
    return 0


def _load_hist_arm(root: Path, encoder: str, arm: str, n_shards: int):
    ids = hist_slide_ids(encoder)
    parts, order = [], []
    for s in range(n_shards):
        d = root / encoder / f"shard_{s:03d}"
        if not (d / "COMPLETE").is_file():
            raise EvalError(f"{d}: shard incomplete")
        parts.append(np.load(d / f"{arm}.npy"))
        order += json.loads((d / "shard.json").read_text())["slides"]
    pos = {s: i for i, s in enumerate(order)}
    arr = np.concatenate(parts)[[pos[s] for s in ids]]
    return ids, arr.reshape(len(ids), -1)


def _load_hist_refs(encoder: str, ids: Sequence[str]) -> np.ndarray:
    P = hist_paths(encoder)
    return np.stack([np.asarray(pickle.load(open(P["refs"] / f"{s}.pkl", "rb"))).flatten()
                     for s in ids])


def cmd_hist_l1(a) -> int:
    t0 = time.time()
    ids, q = _load_hist_arm(a.sampler_dir, a.encoder, a.arm, a.n_shards) \
        if a.arm != "identity_fp16_reference" else (hist_slide_ids(a.encoder), None)
    refs = _load_hist_refs(a.encoder, ids)
    if q is None:
        q = refs
    d = S.l1_distances(q, refs, threads=a.threads)
    w = Writer(a.out_dir / a.encoder / a.arm)
    w.put("l1.npy", _npy_bytes(d))
    w.put("record.json", json.dumps({"encoder": a.encoder, "arm": a.arm, "ids": ids,
                                     "seconds": round(time.time() - t0, 1),
                                     "peak_rss_mb": round(peak_rss_mb(), 1),
                                     "host": socket.gethostname(),
                                     "lsf_job": os.environ.get("LSB_JOBID"),
                                     "code_sha256": code_hashes()}).encode())
    w.put("COMPLETE", json.dumps(w.files, indent=1).encode())
    return 0


def _hist_metadata(encoder: str):
    P = hist_paths(encoder)
    md = pd.read_csv(P["metadata"]).dropna(subset=["Slide.Scan.File"])
    md["Tumor.group"] = md["Tumor.group"].apply(S.historical_label)
    md["Slide.Scan.File"] = md["Slide.Scan.File"].astype(int).astype(str)
    return md


def _historical_sentinels(ids: Sequence[str], pmap: pd.DataFrame) -> np.ndarray:
    """Cells set to 100000 by the notebook: update_same_slide_1e6 (row label startswith column
    id) then update_same_patient_1e6 (patients with >1 slide; rows startswith any patient slide,
    columns that patient's slides). Row labels are '<id>_<perturbation>'."""
    ids = list(ids)
    col = {s: j for j, s in enumerate(ids)}
    mask = np.zeros((len(ids), len(ids)), bool)
    rows = [f"{s}_x" for s in ids]  # the suffix never starts a slide id
    arr = np.array(rows, dtype=object)

    def starts(prefix):
        return np.array([r.startswith(prefix) for r in arr])

    cache = {}
    for s in ids:
        cache[s] = starts(s)
        mask[cache[s], col[s]] = True
    m = pmap[["Slide.Scan.File", "UID.Subject"]].dropna().copy()
    m["Slide.Scan.File"] = m["Slide.Scan.File"].astype(str)
    for _, sl in m.groupby("UID.Subject")["Slide.Scan.File"]:
        sl = [str(x) for x in sl]
        if len(sl) < 2:
            continue
        for a_ in sl:
            if a_ not in col:
                continue
            for b_ in sl:
                rws = cache[b_] if b_ in cache else starts(b_)
                mask[rws, col[a_]] = True
    return mask


def _first_class(md: pd.DataFrame) -> Dict[str, str]:
    """get_class: the first metadata row of a slide."""
    cls: Dict[str, str] = {}
    for s, g in zip(md["Slide.Scan.File"], md["Tumor.group"]):
        cls.setdefault(s, g)
    return cls


def _hist_eval_matrix(dist: np.ndarray, ids: Sequence[str], md: pd.DataFrame, k: int):
    """eval_new_model on a square (ids x ids) matrix: filter_data_new + make_nn_predictions_new."""
    cls = _first_class(md)
    keep = [i for i, s in enumerate(ids) if s in cls and cls[s] != "MISC"]
    sub = dist[np.ix_(keep, keep)]
    lab = [cls[ids[i]] for i in keep]
    preds, gen = S.historical_predict(sub, lab, lab, k)
    return [ids[i] for i in keep], lab, preds, gen


def cmd_hist_eval(a) -> int:
    t0 = time.time()
    enc = a.encoder
    P = hist_paths(enc)
    md_all = _hist_metadata(enc)
    vc = md_all["Tumor.group"].value_counts()
    md40 = md_all[md_all["Tumor.group"].isin(list(vc[vc >= 40].index))]
    pmap = pd.read_csv(P["patient_map"])
    ids = hist_slide_ids(enc)
    report: dict = {"encoder": enc, "paths": {k: str(v) for k, v in P.items()},
                    "n_slides": len(ids)}

    # Stage-1 SAMPLER comparison (all shards)
    shards = sorted((a.sampler_dir / enc).glob("shard_*/shard.json"))
    eq = tot = 0
    maxabs = 0.0
    fp16_ok = 0
    nonfin = 0
    for sj in shards:
        r = json.loads(sj.read_text())
        for sid, cm in r["comparisons"].items():
            for c in HIST_CONDITIONS:
                tot += 1
                eq += int(cm[c]["equal"])
                if cm[c]["max_abs"] is not None:
                    maxabs = max(maxabs, cm[c]["max_abs"])
            fp16_ok += int(cm["main_fp16_equals_reference"])
        nonfin += sum(sum(v.values()) for v in r["raw_nonfinite_rows"].values())
    report["sampler_vs_stored"] = {"slide_conditions": tot, "bitwise_equal": eq,
                                   "max_abs_diff": maxabs, "shards": len(shards)}
    report["main_raw_fp16_sampler_equals_historical_reference"] = {"slides": fp16_ok,
                                                                   "of": len(ids)}
    report["historical_raw_nonfinite_rows_total"] = nonfin

    # Stored historical matrix and stored metrics
    stored = pd.read_pickle(P["stored_matrix"])
    stored.index = stored.index.astype(str)
    stored.columns = stored.columns.astype(str)
    if list(stored.columns) != ids:
        raise EvalError("stored matrix columns differ from the sorted historical slide order")
    hs = _historical_sentinels(ids, pmap)
    base_f1 = {k: json.loads((P["baseline"] / f"top_{k}" / "generous" /
                              "evaluation_metrics.json").read_text()) for k in CUTOFFS}
    l1cmp, fig_rows, pred_cmp = {}, [], {}
    mine_mats = {}
    for c in HIST_EVAL_ORDER:
        rows = [f"{s}_{c}" for s in ids]
        st = stored.loc[rows, ids].to_numpy(dtype=np.float64)
        mine = np.load(a.l1_dir / enc / c / "l1.npy")
        sent = st == 100000
        nonsent = ~hs
        l1cmp[c] = {
            "stored_100000_cells": int(sent.sum()),
            "replicated_sentinel_cells": int(hs.sum()),
            "sentinel_sets_identical": bool(np.array_equal(sent, hs)),
            "non_sentinel_cells": int(nonsent.sum()),
            "non_sentinel_bitwise_equal": int(np.count_nonzero(st[nonsent] == mine[nonsent])),
            "non_sentinel_max_abs_diff": float(np.abs(st[nonsent] - mine[nonsent]).max()),
            "non_sentinel_max_rel_diff": float((np.abs(st[nonsent] - mine[nonsent])
                                                / np.abs(st[nonsent])).max()),
        }
        mh = mine.copy()
        mh[hs] = 100000.0
        mine_mats[c] = mine
        for kk, k in enumerate(CUTOFFS):
            js = json.loads((P["metrics"] / f"top_{k}" / c / "evaluation_metrics.json").read_text())
            q_s, lab_s, pr_s, _ = _hist_eval_matrix(st, ids, md40, k)
            q_m, lab_m, pr_m, _ = _hist_eval_matrix(mh, ids, md40, k)
            sc_s = S.historical_scores(lab_s, pr_s)
            sc_m = S.historical_scores(lab_m, pr_m)
            pr_hist = pd.read_csv(P["metrics"] / f"top_{k}" / c / "pred_results.csv", dtype=str)
            hp = dict(zip(pr_hist["query"], pr_hist["prediction"]))
            pred_cmp[f"{c}|{k}"] = {
                "n_hist": len(pr_hist), "n_stored_replay": len(q_s),
                "stored_replay_pred_equal": sum(hp.get(q) == p for q, p in zip(q_s, pr_s)),
                "pipeline_pred_equal": sum(hp.get(q) == p for q, p in zip(q_m, pr_m))}
            fig_rows.append({
                "condition": c, "k": k, "n_queries": len(q_m),
                "json_accuracy": js["accuracy"], "json_f1": js["f1"],
                "stored_replay_accuracy": sc_s["accuracy"], "stored_replay_f1": sc_s["f1"],
                "pipeline_accuracy": sc_m["accuracy"], "pipeline_f1": sc_m["f1"],
                "pipeline_precision": sc_m["precision"], "pipeline_recall": sc_m["recall"],
                "json_precision": js["precision"], "json_recall": js["recall"],
                "baseline_f1_json": base_f1[k]["f1"],
                "fig4a_published": FIG4A[enc][HIST_EVAL_ORDER.index(c)][kk],
                "pipeline_delta_f1_vs_baseline_json": sc_m["f1"] - base_f1[k]["f1"],
            })
    del stored
    fr = pd.DataFrame(fig_rows)
    fr["pipeline_delta_f1_3dp"] = fr.pipeline_delta_f1_vs_baseline_json.round(3)
    fr["fig4a_cell_reproduced"] = np.isclose(fr.pipeline_delta_f1_3dp, fr.fig4a_published,
                                             atol=1e-9)
    fr["accuracy_exact"] = fr.pipeline_accuracy == fr.json_accuracy
    fr["f1_exact"] = fr.pipeline_f1 == fr.json_f1
    fr["stored_replay_f1_exact"] = fr.stored_replay_f1 == fr.json_f1
    report["l1_vs_stored_matrix"] = l1cmp
    report["prediction_agreement"] = pred_cmp
    report["fig4a_cells_reproduced"] = int(fr.fig4a_cell_reproduced.sum())
    report["fig4a_cells"] = int(len(fr))
    report["exact_f1_cells"] = int(fr.f1_exact.sum())
    report["exact_accuracy_cells"] = int(fr.accuracy_exact.sum())

    # Baseline ("original") replication from the main-grid no_same_patient matrix
    base_rows = []
    try:
        bm = pd.read_pickle(P["baseline_matrix"])
        bm.index = bm.index.astype(str)
        bm.columns = bm.columns.astype(str)
        sl = bm.index[np.isin(bm.index, md40["Slide.Scan.File"])]
        bsub = bm.loc[sl, sl]
        misc = md40.loc[md40["Tumor.group"] == "MISC", "Slide.Scan.File"].tolist()
        bsub = bsub.drop(index=misc, columns=misc, errors="ignore")
        bsub = bsub.dropna(axis=0, how="all").dropna(axis=1, how="all")
        cls = _first_class(md40)
        bl = list(bsub.index)
        lab = [cls[s] for s in bl]
        vals = bsub.to_numpy(dtype=np.float64)
        labarr = np.asarray(lab, dtype=object)
        for k in CUTOFFS:
            preds = []
            for i in range(len(bl)):
                idx = np.argsort(vals[i])[:k]
                cnt = __import__("collections").Counter(labarr[idx].tolist())
                top = max(cnt.values())
                modal = [x for x, v in cnt.items() if v == top]
                preds.append(lab[i] if lab[i] in modal else modal[0])
            sc = S.historical_scores(lab, preds)
            base_rows.append({"k": k, "n_queries": len(bl), "replay_f1": sc["f1"],
                              "json_f1": base_f1[k]["f1"], "replay_accuracy": sc["accuracy"],
                              "json_accuracy": base_f1[k]["accuracy"],
                              "f1_exact": sc["f1"] == base_f1[k]["f1"]})
        report["baseline_slides_absent_vs_perturbed"] = sorted(
            set(_hist_eval_matrix(mine_mats[HIST_EVAL_ORDER[0]], ids, md40, 1)[0]) - set(bl))
    except Exception as exc:  # noqa: BLE001 - report, do not hide
        report["baseline_replay_error"] = f"{type(exc).__name__}: {exc}"
    report["baseline_replay"] = base_rows

    # F2: population-restricted, population patient exclusion, identity pseudo-arms
    pop = load_population(a.population)
    pos = {s: i for i, s in enumerate(ids)}
    pidx = np.array([pos[s] for s in pop.slide_id])
    um = dict(zip(pmap["Slide.Scan.File"].astype(str), pmap["UID.Subject"].astype(str)))
    report["population_patient_vs_UID_Subject_agree"] = int(sum(
        um.get(s) == p for s, p in zip(pop.slide_id, pop.patient_id)))
    hl = [S.historical_label(x) for x in pop.diagnostic_group]
    keep = S.historical_keep(pop.diagnostic_group.tolist())
    p = pop.patient_id.to_numpy(dtype=object)
    same = p[:, None] == p[None, :]
    arms = {c: mine_mats[c] for c in HIST_EVAL_ORDER}
    for extra in ("identity_fp16_reference", "identity_fp32"):
        f = a.l1_dir / enc / extra / "l1.npy"
        if f.is_file():
            arms[extra] = np.load(f)
    f2, f2_scores = [], {}
    qk = np.flatnonzero(keep)
    groups = sorted(set(pop.diagnostic_group))
    for arm, full in arms.items():
        d = full[np.ix_(pidx, pidx)].copy()
        d[same] = S.SENTINEL
        for k in CUTOFFS:
            preds, _ = S.historical_predict(d[np.ix_(qk, qk)], [hl[i] for i in qk],
                                            [hl[i] for i in qk], k)
            sc = S.historical_scores([hl[i] for i in qk], preds)
            f2_scores[(arm, k)] = sc
        # primary-style point estimates (no bootstrap) over all 2,778 slides
        ret = np.ones(len(pop), bool)
        st, rel, _, _, _, _ = rank_condition(pop, d, ret, ret)
        vec = query_vectors(st, rel)
        for k in CUTOFFS:
            for ep in ENDPOINTS:
                gm = [float(np.mean(vec[(ep, k)][pop.diagnostic_group.to_numpy() == g]))
                      for g in groups]
                f2_scores[(arm, k, ep)] = float(np.mean(gm))
    for arm in arms:
        for kk, k in enumerate(CUTOFFS):
            sc = f2_scores[(arm, k)]
            row = {"arm": arm, "k": k, "n_queries": int(qk.size), "f1": sc["f1"],
                   "accuracy": sc["accuracy"],
                   "macro_hit": f2_scores[(arm, k, "hit")],
                   "macro_precision_at_k": f2_scores[(arm, k, "precision")],
                   "delta_f1_vs_baseline_json": sc["f1"] - base_f1[k]["f1"]}
            for ref in ("identity_fp16_reference", "identity_fp32"):
                if (ref, k) in f2_scores:
                    row[f"delta_f1_vs_{ref}"] = sc["f1"] - f2_scores[(ref, k)]["f1"]
                    row[f"delta_macro_hit_vs_{ref}"] = (f2_scores[(arm, k, "hit")]
                                                        - f2_scores[(ref, k, "hit")])
            if arm in HIST_EVAL_ORDER:
                row["fig4a_published"] = FIG4A[enc][HIST_EVAL_ORDER.index(arm)][kk]
            f2.append(row)
    report["population_restricted"] = {
        "population": str(a.population), "slides": len(pop), "historical_filter_queries":
        int(qk.size), "exclusion": "self + same population patient_id -> 1e6",
        "order": "population order"}
    report["seconds"] = round(time.time() - t0, 1)
    report["peak_rss_mb"] = round(peak_rss_mb(), 1)
    report["code_sha256"] = code_hashes()
    report["versions"] = {"numpy": np.__version__, "pandas": pd.__version__,
                          "scipy": __import__("scipy").__version__,
                          "sklearn": __import__("sklearn").__version__}
    w = Writer(a.out_dir / enc)
    w.put("fig4a_comparison.csv", fr.to_csv(index=False).encode())
    w.put("population_restricted.csv", pd.DataFrame(f2).to_csv(index=False).encode())
    w.put("baseline_replay.csv", pd.DataFrame(base_rows).to_csv(index=False).encode())
    w.put("fidelity_report.json", json.dumps(report, indent=1, default=str).encode())
    w.put("COMPLETE", json.dumps(w.files, indent=1).encode())
    print(json.dumps({k: report[k] for k in ("sampler_vs_stored", "fig4a_cells_reproduced",
                                             "fig4a_cells", "exact_f1_cells")}, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    out = default_out()
    root = out / "r2_1_perturb_20261003" if out else None
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate encoders (production or pilot)")
    r.add_argument("--encoders", default="h_optimus_1,uni2,virchow2")
    r.add_argument("--conditions", default="")
    r.add_argument("--slides-subset", default="",
                   help="comma-separated slide ids or a file of ids (pilot/test only)")
    r.add_argument("--allow-partial", action="store_true")
    r.add_argument("--virchow2-cls-only", action="store_true",
                   help="sensitivity: slice virchow2 features [:, :1280] before SAMPLER")
    r.add_argument("--population", type=Path, default=(
        out / "r2_1_cpu_20261002" / "perturb_attempt_001" / "queries.csv") if out else None)
    r.add_argument("--features-root", type=Path, default=root / "features" if root else None)
    r.add_argument("--out-root", type=Path, default=root / "eval" if root else None)
    r.add_argument("--workers", type=int, default=1, help="processes for the SAMPLER stage")
    r.add_argument("--threads", type=int, default=1, help="threads for the L1 stage")
    r.add_argument("--draw-count", type=int, default=5000)
    r.add_argument("--seed", type=int, default=123)
    hs = sub.add_parser("hist-sampler")
    hs.add_argument("--encoder", choices=("uni2", "virchow2"), required=True)
    hs.add_argument("--shard", type=int, required=True)
    hs.add_argument("--n-shards", type=int, required=True)
    hs.add_argument("--out-dir", type=Path, required=True)
    hl = sub.add_parser("hist-l1")
    hl.add_argument("--encoder", choices=("uni2", "virchow2"), required=True)
    hl.add_argument("--arm", required=True,
                    choices=(*HIST_CONDITIONS, "identity_fp32", "identity_fp16_reference"))
    hl.add_argument("--sampler-dir", type=Path, required=True)
    hl.add_argument("--n-shards", type=int, required=True)
    hl.add_argument("--out-dir", type=Path, required=True)
    hl.add_argument("--threads", type=int, default=1)
    he = sub.add_parser("hist-eval")
    he.add_argument("--encoder", choices=("uni2", "virchow2"), required=True)
    he.add_argument("--sampler-dir", type=Path, required=True)
    he.add_argument("--l1-dir", type=Path, required=True)
    he.add_argument("--out-dir", type=Path, required=True)
    he.add_argument("--population", type=Path, default=(
        out / "r2_1_cpu_20261002" / "perturb_attempt_001" / "queries.csv") if out else None)
    a = ap.parse_args(argv)
    a.argv_full = [sys.argv[0], *argv] if argv is not None else list(sys.argv)
    try:
        if a.cmd == "run":
            if a.population is None or a.features_root is None or a.out_root is None:
                ap.error("export PROSPER_OUT or pass --population/--features-root/--out-root")
            return cmd_run(a)
        if a.cmd == "hist-sampler":
            return cmd_hist_sampler(a)
        if a.cmd == "hist-l1":
            return cmd_hist_l1(a)
        return cmd_hist_eval(a)
    except IncompleteInputs as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
