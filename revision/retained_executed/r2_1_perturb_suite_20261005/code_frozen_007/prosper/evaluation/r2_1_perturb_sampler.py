"""Pure SAMPLER / L1 / ranking / historical top-k kernels for the R2.1 perturbation evaluation.

SAMPLER (``comet-robustness/src/sampler_gen_multiprocess_perturbation.py``): a slide is the
per-dimension 5/15/.../95th percentiles of its tile features, ``np.percentile(X, P, axis=0)``
-> (10, D), flattened row-major for distances. No float16 cast (that cast belongs to the
main-grid SAMPLER of ``aggregate_cli``, not to the perturbation harness).

Historical secondary metric (``comet-robustness/src/benchmarking_perturbations.py``
``eval_new_model`` with ``analysis_lib.make_nn_predictions_new`` / ``load_metadata`` /
``filter_data_new``): ``np.argpartition(row, k)[:k]`` neighbours, majority vote with the
"generous" tie rule (correct if the query class is among the modal classes, otherwise the first
modal class in ``Counter`` insertion order), then sklearn accuracy and macro F1/precision/recall.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.distance import cdist

PERCENTILES = (5, 15, 25, 35, 45, 55, 65, 75, 85, 95)
SENTINEL = 1e6
HISTORICAL_MIN_SITE_COUNT = 40
HISTORICAL_DROP_GROUP = "MISC"


@dataclass(frozen=True)
class SamplerResult:
    embedding: Optional[np.ndarray]     # (10, D) float64, or None when no usable tile remains
    n_tiles: int
    n_excluded_listed: int              # explicitly listed (e.g. Reinhard non-finite Lab) tiles
    n_excluded_nonfinite: int           # remaining rows containing any non-finite value
    n_used: int


def sampler(features: np.ndarray, exclude: Optional[Sequence[int]] = None) -> SamplerResult:
    """``np.percentile(X[kept], PERCENTILES, axis=0)`` with explicit row exclusions.

    Rows listed in ``exclude`` are dropped first; then any row with a non-finite value. The
    percentile is evaluated on a transposed, row-sorted copy, which is numerically identical to
    ``np.percentile(X, P, axis=0)`` (same order statistics and the same linear interpolation) but
    about twice as fast for large slides (asserted in the tests).
    """
    x = np.asarray(features)
    if x.ndim != 2:
        raise ValueError(f"features must be (n, D), got {x.shape}")
    n = x.shape[0]
    keep = np.ones(n, dtype=bool)
    listed = 0
    if exclude is not None and len(exclude):
        idx = np.asarray(sorted(set(int(i) for i in exclude)), dtype=np.int64)
        if idx.min() < 0 or idx.max() >= n:
            raise ValueError(f"excluded tile index outside 0..{n - 1}")
        keep[idx] = False
        listed = int(idx.size)
    finite = np.isfinite(x).all(axis=1)
    nonfinite = int(np.count_nonzero(keep & ~finite))
    keep &= finite
    used = int(np.count_nonzero(keep))
    if used == 0:
        return SamplerResult(None, n, listed, nonfinite, 0)
    xt = np.ascontiguousarray(x[keep].T)
    xt.sort(axis=1)
    emb = np.percentile(xt, PERCENTILES, axis=1)
    return SamplerResult(emb, n, listed, nonfinite, used)


def l1_distances(queries: np.ndarray, references: np.ndarray, threads: int = 1,
                 block: int = 64) -> np.ndarray:
    """float64 ``cdist(Q, R, 'cityblock')`` computed in independent row blocks.

    Each cell is computed by the same scipy kernel regardless of blocking, so the result is
    bitwise identical to a single ``cdist`` call (asserted in the tests)."""
    q = np.ascontiguousarray(queries, dtype=np.float64)
    r = np.ascontiguousarray(references, dtype=np.float64)
    out = np.empty((q.shape[0], r.shape[0]), dtype=np.float64)
    if q.shape[0] == 0 or r.shape[0] == 0:
        return out
    starts = list(range(0, q.shape[0], block))

    def run(s: int) -> None:
        out[s:s + block] = cdist(q[s:s + block], r, metric="cityblock")

    if threads <= 1:
        for s in starts:
            run(s)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(run, starts))
    return out


def apply_sentinels(dist: np.ndarray, patients: Sequence[str]) -> dict:
    """Set self and same-patient cells of a square population matrix to ``SENTINEL`` in place."""
    p = np.asarray(patients, dtype=object)
    if dist.shape != (p.size, p.size):
        raise ValueError("distance matrix must be square over the population")
    same = p[:, None] == p[None, :]
    n_self = int(p.size)
    n_same = int(np.count_nonzero(same)) - n_self
    dist[same] = SENTINEL
    return {"self_cells": n_self, "same_patient_offdiag_cells": n_same}


def rank_query(scores: np.ndarray, candidates: np.ndarray, depth: int = 10):
    """Stable ascending order of ``scores`` (aligned with population-ordered ``candidates``)."""
    order = np.argsort(scores, kind="mergesort")
    return candidates[order[:depth]], scores[order[:depth]], scores[order]


# ---------------------------------------------------------------------------------------------
# Historical secondary metric
# ---------------------------------------------------------------------------------------------

def historical_label(label: str) -> str:
    """``load_metadata``: ``str(x).upper().replace(' ', '_')``."""
    return str(label).upper().replace(" ", "_")


def historical_keep(labels: Sequence[str], min_site_count: int = HISTORICAL_MIN_SITE_COUNT,
                    counts_from: Optional[Sequence[str]] = None) -> np.ndarray:
    """Slides kept by ``load_metadata`` (group count >= min_site_count) then ``filter_data_new``
    (MISC dropped). Counts are taken over ``counts_from`` (default: ``labels``)."""
    lab = [historical_label(x) for x in labels]
    src = [historical_label(x) for x in (counts_from if counts_from is not None else labels)]
    c = Counter(src)
    return np.array([c[x] >= min_site_count and x != HISTORICAL_DROP_GROUP for x in lab], bool)


def historical_predict(dist: np.ndarray, query_labels: Sequence[str],
                       ref_labels: Sequence[str], k: int):
    """``make_nn_predictions_new`` on a (n_queries x n_refs) float64 matrix.

    Returns (predictions, generous) per query, replicating ``np.argpartition(row, k)[:k]``,
    ``Counter`` insertion order and the generous tie rule."""
    ref = np.asarray(ref_labels, dtype=object)
    preds, gen = [], []
    for i, qc in enumerate(query_labels):
        row = np.asarray(dist[i], dtype=np.float64)
        idx = np.argpartition(row, k)[:k]
        counts = Counter(ref[idx].tolist())
        top = max(counts.values())
        modal = [c for c, v in counts.items() if v == top]
        if qc in modal:
            preds.append(qc)
            gen.append(1)
        else:
            preds.append(modal[0])
            gen.append(0)
    return preds, gen


def historical_scores(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """``eval_new_model`` scores: accuracy and sklearn macro F1/precision/recall."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
