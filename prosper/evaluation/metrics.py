"""Standard information-retrieval metrics for slide/ROI search.

A review comment asked for the metrics the retrieval literature actually reports. None
of them existed anywhere in the codebase -- a grep across ``comet-analysis``,
``comet-visualization`` and the ``prosper`` package finds no implementation of
MRR, mAP@k, nDCG, recall@k or precision@k. They are cheap: ranked neighbour lists
are fully recoverable from the cached distance matrices, so this is a new function
over an existing artifact with no re-embedding.

Setup. Each query slide has one true label. Every library slide has a label. A
retrieved neighbour is *relevant* iff its label equals the query's true label.
Relevance is therefore binary and the relevant set is the query's whole class.

A definitional warning that matters for how these get reported
--------------------------------------------------------------
``recall@k`` is ambiguous in the WSI-retrieval literature and the two readings
differ by orders of magnitude here:

* **Textbook recall@k** = (relevant retrieved in top-k) / (all relevant in
  library). With class sizes in the hundreds and k <= 10, this is bounded above by
  roughly k/|class| and so is ~0.01 -- a true number that reads as failure and
  says more about class size than about the system.
* **"recall@k" as commonly used in pathology retrieval papers** = the fraction of
  queries with at least one relevant neighbour in the top-k. That is
  ``hit@k``/``success@k``.

Both are computed. Report ``hit@k`` under that name, report textbook recall@k as
``recall_at_k`` with its denominator stated, and never present one as the other.
This is the same class of problem as the published ``any_hit`` tie-break, which is
hit@k reported through an F1 wrapper.

All functions take ``neighbor_labels``: an ``(n_queries, k)`` array of retrieved
neighbour labels, already ordered best-first and already excluding self and
same-patient slides. Doing the exclusion upstream keeps this module free of any
opinion about masking.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def _relevance(neighbor_labels: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """Binary relevance matrix, shape ``(n_queries, k)``."""
    nl = np.asarray(neighbor_labels, dtype=object)
    tl = np.asarray(true_labels, dtype=object).reshape(-1, 1)
    if nl.ndim != 2:
        raise ValueError(f"neighbor_labels must be 2-D, got shape {nl.shape}")
    if nl.shape[0] != tl.shape[0]:
        raise ValueError(
            f"n_queries mismatch: {nl.shape[0]} neighbour rows vs {tl.shape[0]} labels")
    return (nl == tl)


def hit_at_k(rel: np.ndarray, k: int) -> np.ndarray:
    """Per-query 1 if any of the top-k is relevant. Monotone non-decreasing in k."""
    return rel[:, :k].any(axis=1).astype(float)


def precision_at_k(rel: np.ndarray, k: int) -> np.ndarray:
    """Per-query fraction of the top-k that is relevant."""
    return rel[:, :k].sum(axis=1) / float(k)


def recall_at_k(rel: np.ndarray, k: int, n_relevant: np.ndarray) -> np.ndarray:
    """Textbook recall@k. ``n_relevant`` is the query's class size in the library
    (excluding the query itself and any masked slides)."""
    n_rel = np.asarray(n_relevant, dtype=float)
    out = np.zeros(rel.shape[0], dtype=float)
    nz = n_rel > 0
    out[nz] = rel[:nz.shape[0]][nz, :k].sum(axis=1) / n_rel[nz]
    return out


def reciprocal_rank(rel: np.ndarray) -> np.ndarray:
    """Per-query 1/rank of the first relevant neighbour, 0 if none in the list.

    Note this is truncated at the retrieved depth: a query whose first relevant
    neighbour sits at rank 50 scores 0 if only 10 were retrieved. Report the
    retrieval depth alongside MRR.
    """
    n, k = rel.shape
    first = np.argmax(rel, axis=1)
    found = rel.any(axis=1)
    rr = np.zeros(n, dtype=float)
    rr[found] = 1.0 / (first[found] + 1.0)
    return rr


def average_precision_at_k(rel: np.ndarray, k: int,
                           n_relevant: np.ndarray) -> np.ndarray:
    """Per-query AP@k with the standard ``min(k, R)`` normaliser.

    Dividing by ``min(k, R)`` rather than ``R`` keeps AP@k in [0, 1] and
    attainable: with R >> k, normalising by R would cap the best possible score at
    k/R and make the metric unreadable for exactly the same reason textbook
    recall@k is unreadable here.
    """
    n_rel = np.asarray(n_relevant, dtype=float)
    r = rel[:, :k].astype(float)
    csum = np.cumsum(r, axis=1)
    ranks = np.arange(1, k + 1, dtype=float)[None, :]
    prec_at_i = csum / ranks
    numer = (prec_at_i * r).sum(axis=1)
    denom = np.minimum(float(k), n_rel)
    out = np.zeros(rel.shape[0], dtype=float)
    nz = denom > 0
    out[nz] = numer[nz] / denom[nz]
    return out


def ndcg_at_k(rel: np.ndarray, k: int, n_relevant: np.ndarray) -> np.ndarray:
    """Per-query nDCG@k with binary gains and log2 discount.

    The ideal ranking places ``min(k, R)`` relevant items first.
    """
    n_rel = np.asarray(n_relevant, dtype=float)
    disc = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    dcg = (rel[:, :k].astype(float) * disc[None, :]).sum(axis=1)
    ideal_n = np.minimum(float(k), n_rel).astype(int)
    cum_disc = np.concatenate([[0.0], np.cumsum(disc)])
    idcg = cum_disc[np.clip(ideal_n, 0, k)]
    out = np.zeros(rel.shape[0], dtype=float)
    nz = idcg > 0
    out[nz] = dcg[nz] / idcg[nz]
    return out


def compute_all(
    neighbor_labels: np.ndarray,
    true_labels: Sequence,
    k_values: Sequence[int] = (1, 3, 5, 10),
    library_class_counts: Optional[Dict] = None,
    n_relevant: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """All metrics at all k, micro- and macro-averaged.

    ``library_class_counts`` maps label -> number of library slides of that label,
    used as the recall/AP/nDCG denominator. It must already reflect whatever
    exclusion was applied (self, same-patient, dropped classes); if omitted, the
    metrics that need a denominator are skipped rather than silently computed
    against a wrong one.

    Macro averages weight each *class* equally, which is the honest summary given
    a class distribution spanning 1 to several hundred slides. Micro averages
    weight each query equally and are dominated by the largest classes. Both are
    returned; the manuscript should lead with macro.
    """
    true_labels = np.asarray(list(true_labels), dtype=object)
    rel = _relevance(neighbor_labels, true_labels)
    depth = rel.shape[1]

    # A per-query count is preferred over a per-class one, and the difference is
    # not cosmetic. The relevant set available to a query excludes that query's
    # own patient's slides, so for a 30-slide class where the patient holds 3,
    # the honest denominator is 27 and not 30 -- a ~10% error in recall@k, on
    # exactly the metric a reader checking the definition would recompute by hand.
    if n_relevant is not None:
        n_relevant = np.asarray(n_relevant, dtype=float)
        if n_relevant.shape[0] != true_labels.shape[0]:
            raise ValueError(
                f"n_relevant has {n_relevant.shape[0]} entries for "
                f"{true_labels.shape[0]} queries")
    elif library_class_counts is not None:
        n_relevant = np.array(
            [float(library_class_counts.get(t, 0)) for t in true_labels])

    classes = sorted(set(true_labels.tolist()), key=str)
    out: Dict[str, float] = {"n_queries": int(rel.shape[0]),
                             "retrieval_depth": int(depth)}

    def _add(name: str, per_query: np.ndarray) -> None:
        out[f"micro_{name}"] = float(np.mean(per_query))
        per_class = [float(np.mean(per_query[true_labels == c])) for c in classes]
        out[f"macro_{name}"] = float(np.mean(per_class))

    for k in k_values:
        if k > depth:
            continue
        _add(f"hit_at_{k}", hit_at_k(rel, k))
        _add(f"precision_at_{k}", precision_at_k(rel, k))
        if n_relevant is not None:
            _add(f"recall_at_{k}", recall_at_k(rel, k, n_relevant))
            _add(f"ap_at_{k}", average_precision_at_k(rel, k, n_relevant))
            _add(f"ndcg_at_{k}", ndcg_at_k(rel, k, n_relevant))

    _add("mrr", reciprocal_rank(rel))
    return out
