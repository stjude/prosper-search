# roi_agg_eval.py
import os
import glob
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)

# -------------------------
# I/O helpers
# -------------------------

def load_metadata(
    csv_path: str,
    id_col: str = "Slide.Scan.File",
    label_col: str = "Tumor.group",
    drop_labels: Optional[List[str]] = None,
    min_class_count: Optional[int] = None,
    patient_col: Optional[str] = None,
) -> pd.DataFrame:
    """Load the slide->label map used by the ROI evaluator.

    ``patient_col`` is additive and defaults to ``None`` so every existing call
    site (G3's fidelity gate, which must load the legacy patient-free map)
    behaves exactly as before. Passing it (C1b's patient-excluded path) loads
    the patient-carrying map and carries the column through instead of
    dropping it, which is what ``paths.slide_metadata()`` provides and
    ``paths.roi_metadata_legacy()`` does not.
    """
    df = pd.read_csv(csv_path)
    subset = [id_col, label_col] + ([patient_col] if patient_col else [])
    df = df.dropna(subset=subset).copy()
    df[id_col] = df[id_col].astype(int).astype(str)
    df[label_col] = df[label_col].astype(str)
    if patient_col:
        df[patient_col] = df[patient_col].astype(str)
    if drop_labels:
        df = df[~df[label_col].isin(drop_labels)]
    if min_class_count is not None:
        counts = df[label_col].value_counts()
        keep = counts[counts >= min_class_count].index
        df = df[df[label_col].isin(keep)]
    cols = [id_col, label_col] + ([patient_col] if patient_col else [])
    return df[cols].drop_duplicates().reset_index(drop=True)


def find_roi_file_for_slide(slide_id: str, roi_hits_dir: str) -> Optional[str]:
    pats = [
        os.path.join(roi_hits_dir, f"top_10_hit_slides_slide_{slide_id}.pkl"),
        os.path.join(roi_hits_dir, f"*{slide_id}*.pkl"),
    ]
    for pat in pats:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    return None


def discover_slides_with_files(roi_hits_dir: str) -> List[str]:
    files = glob.glob(os.path.join(roi_hits_dir, "*.pkl"))
    ids = []
    for fp in files:
        base = os.path.basename(fp)
        nums = [s for s in base.replace(".", "_").split("_") if s.isdigit()]
        if nums:
            ids.append(nums[-1])
    return sorted(list(set(ids)))

# -------------------------
# Aggregation utilities
# -------------------------

def _first_present(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None

def _dedup_preserve_order(seq: List[int]) -> List[int]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _tie_break_by_counts_lex(class_counts: Dict[str, int]) -> str:
    max_cnt = max(class_counts.values())
    tied = [c for c, v in class_counts.items() if v == max_cnt]
    return tied[0] if len(tied) == 1 else sorted(tied)[0]

# -------------------------
# Core: per-slide prediction from ROI neighbors
# -------------------------

def predict_slide_from_roi_neighbors(
    slide_id: str,
    roi_hits_dir: str,
    slide_to_label: Dict[str, str],
    top_k_per_roi: int = 10,
    vote_mode: str = "raw_hits",  # "raw_hits" | "unique_per_roi" | "unique_library"
    fallback_chunk_size: int = 10,  # used if ROI id column is absent
) -> Optional[Dict]:
    """
    Aggregate ALL ROIs for a slide into one prediction.

    vote_mode:
      - "raw_hits":        count EVERY hit row across all ROIs (no dedup, no weights).
      - "unique_per_roi":  dedup by slide within each ROI (so the same library slide can vote
                           again if it appears for another ROI).
      - "unique_library":  one vote per library slide across all ROIs.

    Returns dict with: slide, true_label, pred_label, counts and diagnostics.
    """
    fpath = find_roi_file_for_slide(slide_id, roi_hits_dir)
    if fpath is None:
        return None

    with open(fpath, "rb") as f:
        df = pickle.load(f)

    roi_col = _first_present(df, ["roi_id", "query_roi_id", "roi_idx", "roi_index"])
    hit_slide_col = _first_present(df, ["hit_slide_id", "neighbor_slide_id", "hit_id", "hit_slide"])
    query_slide_col = _first_present(df, ["query_slide_id", "slide_id", "scan_id", "query_slide"])
    if hit_slide_col is None:
        return None

    qid = str(int(df[query_slide_col].unique()[0])) if query_slide_col in df.columns else slide_id
    true_label = slide_to_label.get(qid, None)
    if true_label is None:
        return None

    # We IGNORE rank/sim columns (no weighting, no sorting)
    roi_groups: List[pd.DataFrame] = []
    if roi_col in df.columns:
        for _, g in df.groupby(roi_col, sort=False):
            roi_groups.append(g)
    else:
        n = len(df)
        for i in range(0, n, fallback_chunk_size):
            roi_groups.append(df.iloc[i:i + fallback_chunk_size])

    class_counts = Counter()
    used_library_slides_global: set = set()
    n_rois = 0
    skipped_rois = 0
    total_hits_seen = 0
    total_hits_counted = 0
    missing_label_hits = 0

    for g in roi_groups:
        n_rois += 1
        ordered_hits = g[hit_slide_col].astype(int).astype(str).tolist()
        ordered_hits = [h for h in ordered_hits if h != qid]  # drop self
        total_hits_seen += len(ordered_hits)

        if vote_mode == "raw_hits":
            hits = ordered_hits[:top_k_per_roi]
        elif vote_mode == "unique_per_roi":
            hits = _dedup_preserve_order(ordered_hits)[:top_k_per_roi]
        elif vote_mode == "unique_library":
            # we'll just collect unique globally later; here keep per-ROI unique to reduce noise
            hits = _dedup_preserve_order(ordered_hits)[:top_k_per_roi]
        else:
            raise ValueError(f"Unknown vote_mode: {vote_mode}")

        if len(hits) == 0:
            skipped_rois += 1
            continue

        if vote_mode in ("raw_hits", "unique_per_roi"):
            for lib_id in hits:
                lbl = slide_to_label.get(lib_id, None)
                if lbl is None or lbl == "NA":
                    missing_label_hits += 1
                    continue
                class_counts[lbl] += 1
                total_hits_counted += 1

        elif vote_mode == "unique_library":
            for lib_id in hits:
                used_library_slides_global.add(lib_id)

    if vote_mode == "unique_library":
        for lib_id in used_library_slides_global:
            lbl = slide_to_label.get(lib_id, None)
            if lbl is None or lbl == "NA":
                missing_label_hits += 1
                continue
            class_counts[lbl] += 1
            total_hits_counted += 1

    if len(class_counts) == 0:
        return None

    pred_label = _tie_break_by_counts_lex(class_counts)

    return {
        "slide": qid,
        "true_label": true_label,
        "pred_label": pred_label,
        "vote_mode": vote_mode,
        "n_rois": n_rois,
        "skipped_rois": skipped_rois,
        "total_hits_seen": total_hits_seen,
        "total_hits_counted": total_hits_counted,
        "missing_label_hits": missing_label_hits,
    }

# -------------------------
# C1b: patient-excluded per-slide prediction from ROI neighbors
# -------------------------

def predict_slide_from_roi_neighbors_patient_excluded(
    slide_id: str,
    roi_hits_dir: str,
    slide_to_label: Dict[str, str],
    slide_to_patient: Dict[str, str],
    top_k_per_roi: int = 5,
    vote_mode: str = "raw_hits",
    fallback_chunk_size: int = 10,
) -> Optional[Dict]:
    """``predict_slide_from_roi_neighbors`` with same-patient hits removed from
    the vote pool -- the one confirmed patient-leakage exposure in the paper.

    Only the *pool* changes. Voting (``_dedup_preserve_order``) and the
    tie-break (``_tie_break_by_counts_lex``) are the exact same functions the
    published path calls, unmodified, so a change in the outcome is
    attributable to which rows are eligible to vote, not to a reimplemented
    voting rule.

    Same-patient hits are dropped from ``ordered_hits`` *before* the
    ``top_k_per_roi`` slice, mirroring how the slide-level matrix demotes
    same-patient cells before ``argsort`` rather than after the top-k is
    already chosen ([decision record]). Self-exclusion (``h != qid``) is kept explicit
    even though it is subsumed by patient exclusion (a slide always shares its
    own patient), because it is the published behaviour and keeps this
    function auditable line-for-line against the original.

    Every vote that is actually counted is checked against the query's patient
    at the point it is counted (not just at the filtering step) and tallied in
    ``same_patient_hits_surviving`` -- a direct count over the full vote pool,
    not an assumption that the upstream filter worked. It must be exactly 0
    over the whole run (see G3/C1b fidelity check (ii) in ``roi_cli.py``).
    """
    fpath = find_roi_file_for_slide(slide_id, roi_hits_dir)
    if fpath is None:
        return None

    with open(fpath, "rb") as f:
        df = pickle.load(f)

    roi_col = _first_present(df, ["roi_id", "query_roi_id", "roi_idx", "roi_index"])
    hit_slide_col = _first_present(df, ["hit_slide_id", "neighbor_slide_id", "hit_id", "hit_slide"])
    query_slide_col = _first_present(df, ["query_slide_id", "slide_id", "scan_id", "query_slide"])
    if hit_slide_col is None:
        return None

    qid = str(int(df[query_slide_col].unique()[0])) if query_slide_col in df.columns else slide_id
    true_label = slide_to_label.get(qid, None)
    if true_label is None:
        return None
    query_patient = slide_to_patient.get(qid, None)

    roi_groups: List[pd.DataFrame] = []
    if roi_col in df.columns:
        for _, g in df.groupby(roi_col, sort=False):
            roi_groups.append(g)
    else:
        n = len(df)
        for i in range(0, n, fallback_chunk_size):
            roi_groups.append(df.iloc[i:i + fallback_chunk_size])

    class_counts = Counter()
    used_library_slides_global: set = set()
    n_rois = 0
    skipped_rois = 0
    total_hits_seen = 0
    total_hits_counted = 0
    missing_label_hits = 0
    same_patient_hits_dropped = 0
    same_patient_hits_surviving = 0  # must end at 0 -- see docstring

    for g in roi_groups:
        n_rois += 1
        ordered_hits = g[hit_slide_col].astype(int).astype(str).tolist()
        ordered_hits = [h for h in ordered_hits if h != qid]  # drop self (as published)
        total_hits_seen += len(ordered_hits)

        if query_patient is not None:
            kept = []
            for h in ordered_hits:
                if slide_to_patient.get(h, None) == query_patient:
                    same_patient_hits_dropped += 1
                else:
                    kept.append(h)
            ordered_hits = kept

        if vote_mode == "raw_hits":
            hits = ordered_hits[:top_k_per_roi]
        elif vote_mode == "unique_per_roi":
            hits = _dedup_preserve_order(ordered_hits)[:top_k_per_roi]
        elif vote_mode == "unique_library":
            hits = _dedup_preserve_order(ordered_hits)[:top_k_per_roi]
        else:
            raise ValueError(f"Unknown vote_mode: {vote_mode}")

        if len(hits) == 0:
            skipped_rois += 1
            continue

        if vote_mode in ("raw_hits", "unique_per_roi"):
            for lib_id in hits:
                if query_patient is not None and slide_to_patient.get(lib_id, None) == query_patient:
                    same_patient_hits_surviving += 1
                lbl = slide_to_label.get(lib_id, None)
                if lbl is None or lbl == "NA":
                    missing_label_hits += 1
                    continue
                class_counts[lbl] += 1
                total_hits_counted += 1

        elif vote_mode == "unique_library":
            for lib_id in hits:
                used_library_slides_global.add(lib_id)

    if vote_mode == "unique_library":
        for lib_id in used_library_slides_global:
            if query_patient is not None and slide_to_patient.get(lib_id, None) == query_patient:
                same_patient_hits_surviving += 1
            lbl = slide_to_label.get(lib_id, None)
            if lbl is None or lbl == "NA":
                missing_label_hits += 1
                continue
            class_counts[lbl] += 1
            total_hits_counted += 1

    if len(class_counts) == 0:
        return None

    pred_label = _tie_break_by_counts_lex(class_counts)

    # hit@5: is the true label among the top-5 classes by vote count? Not part
    # of the published voting/tie-break (which only needs the single argmax),
    # so this is purely additional reporting -- it does not affect pred_label.
    ranked = sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top5_labels = [lbl for lbl, _ in ranked[:5]]
    hit_at_5 = int(true_label in top5_labels)

    return {
        "slide": qid,
        "true_label": true_label,
        "pred_label": pred_label,
        "vote_mode": vote_mode,
        "n_rois": n_rois,
        "skipped_rois": skipped_rois,
        "total_hits_seen": total_hits_seen,
        "total_hits_counted": total_hits_counted,
        "missing_label_hits": missing_label_hits,
        "same_patient_hits_dropped": same_patient_hits_dropped,
        "same_patient_hits_surviving": same_patient_hits_surviving,
        "hit_at_5": hit_at_5,
    }

# -------------------------
# Batch prediction
# -------------------------

def predict_all_slides_agg(
    roi_hits_dir: str,
    meta_df: pd.DataFrame,
    id_col: str = "Slide.Scan.File",
    label_col: str = "Tumor.group",
    top_k_per_roi: int = 10,
    vote_mode: str = "raw_hits",
    n_workers: int = 32,
) -> pd.DataFrame:
    """
    Predict slide-level labels for all slides with ROI hit files.
    Uses ALL ROIs per slide; aggregation per 'vote_mode'.
    """
    slide_ids_all = meta_df[id_col].astype(str).tolist()
    files_ids = set(discover_slides_with_files(roi_hits_dir))
    slide_ids = [sid for sid in slide_ids_all if sid in files_ids]

    slide_to_label = dict(zip(meta_df[id_col].astype(str), meta_df[label_col].astype(str)))

    import multiprocessing as mp
    from functools import partial
    worker = partial(
        predict_slide_from_roi_neighbors,
        roi_hits_dir=roi_hits_dir,
        slide_to_label=slide_to_label,
        top_k_per_roi=top_k_per_roi,
        vote_mode=vote_mode,
    )

    results = []
    with mp.Pool(processes=n_workers) as pool:
        for rec in pool.imap_unordered(worker, slide_ids):
            if rec is not None:
                results.append(rec)

    if len(results) == 0:
        return pd.DataFrame(columns=[
            "slide", "true_label", "pred_label", "vote_mode",
            "n_rois", "skipped_rois", "total_hits_seen", "total_hits_counted", "missing_label_hits"
        ])

    return pd.DataFrame(results)


def predict_all_slides_agg_patient_excluded(
    roi_hits_dir: str,
    meta_df: pd.DataFrame,
    id_col: str = "Slide.Scan.File",
    label_col: str = "Tumor.group",
    patient_col: str = "UID.Subject",
    top_k_per_roi: int = 5,
    vote_mode: str = "raw_hits",
    n_workers: int = 32,
) -> pd.DataFrame:
    """C1b: :func:`predict_all_slides_agg` with same-patient library slides
    removed from every query's vote pool. ``meta_df`` must carry
    ``patient_col`` (i.e. come from ``paths.slide_metadata()``, not the legacy
    map) -- that is the entire fix, per [decision record] gap 1.
    """
    slide_ids_all = meta_df[id_col].astype(str).tolist()
    files_ids = set(discover_slides_with_files(roi_hits_dir))
    slide_ids = [sid for sid in slide_ids_all if sid in files_ids]

    slide_to_label = dict(zip(meta_df[id_col].astype(str), meta_df[label_col].astype(str)))
    slide_to_patient = dict(zip(meta_df[id_col].astype(str), meta_df[patient_col].astype(str)))

    import multiprocessing as mp
    from functools import partial
    worker = partial(
        predict_slide_from_roi_neighbors_patient_excluded,
        roi_hits_dir=roi_hits_dir,
        slide_to_label=slide_to_label,
        slide_to_patient=slide_to_patient,
        top_k_per_roi=top_k_per_roi,
        vote_mode=vote_mode,
    )

    results = []
    with mp.Pool(processes=n_workers) as pool:
        for rec in pool.imap_unordered(worker, slide_ids):
            if rec is not None:
                results.append(rec)

    if len(results) == 0:
        return pd.DataFrame(columns=[
            "slide", "true_label", "pred_label", "vote_mode",
            "n_rois", "skipped_rois", "total_hits_seen", "total_hits_counted",
            "missing_label_hits", "same_patient_hits_dropped",
            "same_patient_hits_surviving", "hit_at_5",
        ])

    return pd.DataFrame(results)

# -------------------------
# Metrics & class-stratified bootstrap (slides)
# -------------------------

def compute_metrics_from_predictions(
    df_pred: pd.DataFrame,
    labels_order: Optional[List[str]] = None,
) -> Dict:
    y_true = df_pred["true_label"].values
    y_pred = df_pred["pred_label"].values

    if labels_order is None:
        labels_order = sorted(pd.unique(y_true))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)
    macro_pr = precision_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)
    macro_re = recall_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)

    f1_per = f1_score(y_true, y_pred, labels=labels_order, average=None, zero_division=0)
    pr_per = precision_score(y_true, y_pred, labels=labels_order, average=None, zero_division=0)
    re_per = recall_score(y_true, y_pred, labels=labels_order, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    per_class = {
        lbl: {
            "support": int((y_true == lbl).sum()),
            "f1": float(f1_per[i]),
            "precision": float(pr_per[i]),
            "recall": float(re_per[i]),
        }
        for i, lbl in enumerate(labels_order)
    }

    return {
        "micro_f1": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_pr),
        "macro_recall": float(macro_re),
        "per_class": per_class,
        "labels_order": list(labels_order),
        "confusion_matrix": cm.astype(int).tolist(),
        "n_slides": int(len(y_true))
    }


def stratified_bootstrap_cis(
    df_pred: pd.DataFrame,
    B: int = 4000,
    seed: int = 17,
    labels_order: Optional[List[str]] = None,
) -> Dict:
    rng = np.random.default_rng(seed)
    y_true = df_pred["true_label"].values
    y_pred = df_pred["pred_label"].values

    if labels_order is None:
        labels_order = sorted(pd.unique(y_true))

    idx_by_class = {c: np.where(y_true == c)[0] for c in labels_order}
    n_by_class = {c: len(idx_by_class[c]) for c in labels_order}

    micro_vals = np.empty(B, dtype=float)
    macro_f1_vals = np.empty(B, dtype=float)
    macro_pr_vals = np.empty(B, dtype=float)
    macro_re_vals = np.empty(B, dtype=float)
    per_class_f1_vals = {c: np.empty(B, dtype=float) for c in labels_order}

    for b in range(B):
        sample_idx = []
        for c in labels_order:
            n = n_by_class[c]
            draws = rng.integers(0, n, size=n)
            sample_idx.append(idx_by_class[c][draws])
        sample_idx = np.concatenate(sample_idx, axis=0)

        yt = y_true[sample_idx]
        yp = y_pred[sample_idx]

        micro_vals[b] = accuracy_score(yt, yp)
        macro_f1_vals[b] = f1_score(yt, yp, labels=labels_order, average="macro", zero_division=0)
        macro_pr_vals[b] = precision_score(yt, yp, labels=labels_order, average="macro", zero_division=0)
        macro_re_vals[b] = recall_score(yt, yp, labels=labels_order, average="macro", zero_division=0)

        f1_per = f1_score(yt, yp, labels=labels_order, average=None, zero_division=0)
        for i, c in enumerate(labels_order):
            per_class_f1_vals[c][b] = f1_per[i]

    def _ci(arr: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> Tuple[float, float]:
        q = np.percentile(arr, [lo, hi])
        return float(q[0]), float(q[1])

    return {
        "micro_f1_ci": _ci(micro_vals),
        "macro_f1_ci": _ci(macro_f1_vals),
        "macro_precision_ci": _ci(macro_pr_vals),
        "macro_recall_ci": _ci(macro_re_vals),
        "per_class_f1_ci": {c: _ci(per_class_f1_vals[c]) for c in labels_order},
        "B": B,
        "seed": seed
    }
