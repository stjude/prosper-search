# slide_eval.py
from concurrent import futures
import json
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from concurrent.futures import ProcessPoolExecutor, as_completed  # <-- new
from tqdm import tqdm

# -------------------------
# I/O
# -------------------------

def load_similarity_matrix(path: str) -> pd.DataFrame:
    """
    Load a square (n x n) matrix with both row/column index = slide IDs.
    Accepts .pkl (pickle of DataFrame) or .csv (with index col).
    """
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            sim = pickle.load(f)
        if not isinstance(sim, pd.DataFrame):
            raise ValueError("Pickle must contain a pandas DataFrame for the similarity/distance matrix.")
        return sim
    elif path.endswith(".csv"):
        sim = pd.read_csv(path, index_col=0)
        return sim
    else:
        raise ValueError("Unsupported matrix file type. Use .pkl or .csv")


def load_metadata(
    path: str,
    label_col: str,
    id_col: str,
    patient_col: Optional[str] = None,
    drop_labels: Optional[List[str]] = None,
    min_class_count: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load metadata CSV, keep relevant columns, drop unwanted labels,
    and optionally filter out rare classes.
    """
    meta = pd.read_csv(path)
    cols = [id_col, label_col]
    if patient_col:
        cols.append(patient_col)
    meta = meta[cols].dropna(subset=[id_col, label_col])

    if drop_labels:
        meta = meta[~meta[label_col].isin(drop_labels)].copy()

    if min_class_count is not None:
        counts = meta[label_col].value_counts()
        keep = counts[counts >= min_class_count].index
        meta = meta[meta[label_col].isin(keep)].copy()

    # Ensure IDs are strings (match matrix indices)
    meta[id_col] = meta[id_col].astype(str)
    return meta


def filter_similarity_matrix(sim: pd.DataFrame, meta: pd.DataFrame, id_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Intersect matrix with metadata IDs and ensure same order.
    """
    ids = set(meta[id_col].astype(str).unique().tolist())
    sim = sim.copy()
    sim.index = sim.index.astype(str)
    sim.columns = sim.columns.astype(str)

    # DETERMINISM FIX (2026-08-27). The published line here was
    #     keep = list(ids.intersection(sim.index).intersection(sim.columns))
    # and `list(<set of str>)` iterates in an order that depends on
    # PYTHONHASHSEED, which CPython randomises per process and which no job
    # script in this project pins. That reorders the reference library on every
    # run. Ordinarily harmless -- except that `_rank_neighbors` sorts with
    # `kind="mergesort"`, which is *stable*, so equal distances are broken by
    # array position. Yottixel/BoB distances are quantised to ~1e3 distinct
    # values over ~3.4e6 finite cells, so 4.0-4.4% of queries have a TIED
    # rank-1 neighbour, and TITAN 0.9%. For those queries the retrieved top-1
    # -- and therefore the k=1 prediction -- was arbitrary and changed between
    # runs. Measured impact: macro F1 moved by up to 0.011 (hibou/yottixel)
    # between two runs of identical code on an identical matrix.
    #
    # Taking the similarity matrix's own row order makes the library order a
    # property of the stored artifact rather than of the process, so a rerun is
    # bit-identical. Ties still have to be broken somehow and any rule is
    # arbitrary; what matters is that it is fixed, declared, and that the
    # residual ambiguity is quantified separately (see `tie_sensitivity`).
    #
    # The embedding branches (ot_cosine, median_pooling, sampler) have ZERO
    # rank-1 ties, so their numbers -- including the declared primary
    # uni2/ot_cosine -- are unaffected by this fix.
    keep = [i for i in sim.index if i in ids]
    sim_f = sim.loc[keep, keep]
    meta_f = meta[meta[id_col].isin(keep)].copy()

    return sim_f, meta_f

# -------------------------
# Prediction
# -------------------------

def _rank_neighbors(arr: np.ndarray, larger_is_better: bool) -> np.ndarray:
    """
    Return indices that sort arr properly for neighbor ranking.
    """
    if larger_is_better:
        return np.argsort(-arr, kind="mergesort")
    else:
        return np.argsort(arr, kind="mergesort")


# Emitted by the strict-majority (`absolute`) rule when no label clears
# floor(k/2)+1 of the top-k. Never a member of `labels_order`, so it scores as
# a miss rather than as a wrong class. See the `absolute` branch below.
AMBIGUOUS_LABEL = "Ambiguous"


def _choose_label_from_topk(
    topk_labels: List[str],
    tie_break: str = "rank_sum",
) -> Optional[str]:
    """
    Decide predicted label given ordered top-k labels.
    tie_break:
      - "rank_sum": minimize sum of ranks
      - "majority": majority vote (fall back to rank_sum if tie)
    """
    if len(topk_labels) == 0:
        return None

    if tie_break not in {"rank_sum", "majority"}:
        raise ValueError("tie_break must be 'rank_sum' or 'majority'.")

    # rank positions: 0..k-1
    if tie_break == "majority":
        counts = pd.Series(topk_labels).value_counts()
        max_count = counts.max()
        candidates = counts[counts == max_count].index.tolist()
        if len(candidates) == 1:
            return candidates[0]
        # fall back to rank_sum among tied labels
        tie_break = "rank_sum"

    # rank_sum
    rank_sum = {}
    for r, lab in enumerate(topk_labels):
        rank_sum[lab] = rank_sum.get(lab, 0) + (r + 1)  # 1-based ranks
    best = sorted(rank_sum.items(), key=lambda x: (x[1], x[0]))[0][0]
    return best


# def predict_labels_from_matrix(
#     sim: pd.DataFrame,
#     meta: pd.DataFrame,
#     topk: int,
#     id_col: str,
#     label_col: str,
#     patient_col: Optional[str],
#     larger_is_better: bool,
#     tie_break: str = "rank_sum",
#     exclude_same_patient: bool = False,
# ) -> pd.DataFrame:
#     """
#     For each query slide, rank library slides, take top-k neighbors (excluding self),
#     optionally exclude same-patient slides from library, and produce a predicted label.

#     tie_break options:
#       "rank_sum": rank-sum over top-k labels
#       "majority": majority vote over top-k labels; if tie, fall back to rank_sum over all labels
#       "any_hit": if true_label is in top-k, set pred_label = true_label, else fall back to rank_sum
#       "tie_majority": prediction still returns a single label, but we compute and return the majority set;
#                       correct-at-eval time if true_label is in that majority, even when it is a tie

#     Returns a DataFrame with columns:
#       [id_col, "true_label", "pred_label", "topk_labels", "topk_hit", "majority_set", "majority_hit"]
#     """
#     ids = sim.index.tolist()
#     id_to_label = dict(zip(meta[id_col], meta[label_col]))
#     id_to_patient = dict(zip(meta[id_col], meta[patient_col])) if patient_col else {}
#     labels_order_global = sorted(meta[label_col].unique().tolist())

#     records = []
#     sim_values = sim.values
#     n = sim.shape[0]

#     for i, qid in enumerate(ids):
#         scores = sim_values[i, :].copy()
#         # Exclude self
#         scores[i] = np.nan

#         # Optional: exclude same patient
#         if exclude_same_patient and patient_col:
#             qpat = id_to_patient.get(qid, None)
#             if qpat is not None:
#                 for j, sid in enumerate(ids):
#                     if id_to_patient.get(sid, None) == qpat:
#                         scores[j] = np.nan
#                 scores[i] = np.nan

#         # Rank neighbors (ignoring NaNs)
#         valid = ~np.isnan(scores)
#         valid_idx = np.where(valid)[0]

#         if valid_idx.size > 0:
#             ordering = _rank_neighbors(scores[valid_idx], larger_is_better)
#             ranked_idx = valid_idx[ordering]
#             topk_idx = ranked_idx[:topk]
#             topk_ids = [ids[j] for j in topk_idx]
#             topk_labels = [id_to_label[s] for s in topk_ids]
#         else:
#             topk_ids = []
#             topk_labels = []

#         true_lab = id_to_label[qid]
#         topk_hit = true_lab in topk_labels

#         # Majority set over top-k
#         if len(topk_labels) > 0:
#             counts = pd.Series(topk_labels).value_counts()
#             max_count = counts.max()
#             majority = counts[counts == max_count].index.tolist()
#         else:
#             majority = []
#         majority_hit = true_lab in majority

#         # Determine pred_label
#         if tie_break == "any_hit":
#             if topk_hit:
#                 pred_label = true_lab
#             else:
#                 if len(topk_labels) > 0:
#                     pred_label = _choose_label_from_topk(topk_labels, tie_break="rank_sum")
#                 else:
#                     fallback = next((lab for lab in labels_order_global if lab != true_lab), true_lab)
#                     pred_label = fallback

#         elif tie_break in {"rank_sum", "majority"}:
#             pred_label = _choose_label_from_topk(topk_labels, tie_break=tie_break)

#         elif tie_break == "tie_majority":
#             if len(majority) == 1:
#                 pred_label = majority[0]
#             elif len(majority) > 1:
#                 # rank-sum restricted to majority candidates
#                 if len(topk_labels) == 0:
#                     pred_label = None
#                 else:
#                     rank_sum = {}
#                     for r, lab in enumerate(topk_labels):
#                         if lab in majority:
#                             rank_sum[lab] = rank_sum.get(lab, 0) + (r + 1)  # 1-based
#                     pred_label = sorted(rank_sum.items(), key=lambda x: (x[1], x[0]))[0][0]
#             else:
#                 # No neighbors: deterministic fallback
#                 fallback = next((lab for lab in labels_order_global if lab != true_lab), true_lab)
#                 pred_label = fallback
#         else:
#             raise ValueError("tie_break must be 'rank_sum', 'majority', 'any_hit', or 'tie_majority'.")

#         records.append({
#             id_col: qid,
#             "true_label": true_lab,
#             "pred_label": pred_label,
#             "topk_labels": topk_labels,
#             "topk_hit": bool(topk_hit),
#             "majority_set": majority,
#             "majority_hit": bool(majority_hit),
#         })

#     df_pred = pd.DataFrame.from_records(records)
#     return df_pred


# # -------------------------
# # Metrics
# # -------------------------

# def compute_metrics_from_predictions(
#     df_pred: pd.DataFrame,
#     labels_order: Optional[List[str]] = None,
#     correctness: str = "strict",   # "strict" or "tie_majority"
# ) -> Dict:
#     """
#     Compute micro-F1 (accuracy), macro-F1, macro-precision, macro-recall,
#     per-class metrics, and confusion matrix.

#     correctness:
#       "strict": use pred_label as-is
#       "tie_majority": count a sample as correct if true_label is in the top-k majority set
#                       including ties; implemented by substituting pred_for_scoring=true_label
#                       whenever majority_hit is True
#     """
#     if "true_label" not in df_pred.columns or "pred_label" not in df_pred.columns:
#         raise ValueError("df_pred must contain 'true_label' and 'pred_label'.")

#     df = df_pred.dropna(subset=["true_label", "pred_label"]).copy()

#     if correctness == "tie_majority":
#         if "majority_hit" not in df.columns:
#             raise ValueError("majority_hit column missing. Run prediction with majority tracking or tie_break='tie_majority'.")
#         df["pred_for_scoring"] = np.where(df["majority_hit"], df["true_label"], df["pred_label"])
#     else:
#         df["pred_for_scoring"] = df["pred_label"]

#     y_true = df["true_label"].tolist()
#     y_pred = df["pred_for_scoring"].tolist()

#     if labels_order is None:
#         labels_order = sorted(list(set(y_true) | set(y_pred)))

#     cm = confusion_matrix(y_true, y_pred, labels=labels_order).tolist()
#     micro = f1_score(y_true, y_pred, average="micro", labels=labels_order)
#     macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels_order)
#     macro_prec = precision_score(y_true, y_pred, average="macro", labels=labels_order, zero_division=0)
#     macro_rec = recall_score(y_true, y_pred, average="macro", labels=labels_order, zero_division=0)

#     per_class = {}
#     df_support = pd.Series(y_true).value_counts()
#     for c in labels_order:
#         y_true_bin = [1 if t == c else 0 for t in y_true]
#         y_pred_bin = [1 if p == c else 0 for p in y_pred]
#         f1_c = f1_score(y_true_bin, y_pred_bin, zero_division=0)
#         prec_c = precision_score(y_true_bin, y_pred_bin, zero_division=0)
#         rec_c = recall_score(y_true_bin, y_pred_bin, zero_division=0)
#         per_class[c] = {
#             "support": int(df_support.get(c, 0)),
#             "f1": float(f1_c),
#             "precision": float(prec_c),
#             "recall": float(rec_c),
#         }

#     return {
#         "n_slides": int(len(df_pred)),
#         "labels_order": labels_order,
#         "confusion_matrix": cm,
#         "micro_f1": float(micro),
#         "macro_f1": float(macro_f1),
#         "macro_precision": float(macro_prec),
#         "macro_recall": float(macro_rec),
#         "per_class": per_class,
#         "correctness_mode": correctness,
#     }

def predict_labels_from_matrix(
    sim: pd.DataFrame,
    meta: pd.DataFrame,
    topk: int,
    id_col: str,
    label_col: str,
    patient_col: Optional[str],
    larger_is_better: bool,
    tie_break: str = "rank_sum",   # also supports "generous_majority"
    exclude_same_patient: bool = False,
) -> pd.DataFrame:
    """
    tie_break options:
      rank_sum: rank-sum over top-k labels
      majority: majority vote; if tie, fall back to rank_sum
      any_hit: if true_label is in top-k, set pred_label = true_label; else fall back to rank_sum
      generous_majority: if true_label is in the top-k majority set including ties, set pred_label = true_label
                         and mark generous_tie=1; otherwise choose the first majority label in top-k order
                         and set generous_tie=0. No rank-sum inside tied majorities.
    """
    ids = sim.index.tolist()
    id_to_label = dict(zip(meta[id_col], meta[label_col]))
    id_to_patient = dict(zip(meta[id_col], meta[patient_col])) if patient_col else {}
    labels_order_global = sorted(meta[label_col].unique().tolist())

    records = []
    sim_values = sim.values

    for i, qid in enumerate(ids):
        scores = sim_values[i, :].copy()
        scores[i] = np.nan  # exclude self

        if exclude_same_patient and patient_col:
            qpat = id_to_patient.get(qid, None)
            if qpat is not None:
                for j, sid in enumerate(ids):
                    if id_to_patient.get(sid, None) == qpat:
                        scores[j] = np.nan
                scores[i] = np.nan

        valid = ~np.isnan(scores)
        valid_idx = np.where(valid)[0]

        if valid_idx.size > 0:
            ordering = _rank_neighbors(scores[valid_idx], larger_is_better)
            ranked_idx = valid_idx[ordering]
            topk_idx = ranked_idx[:topk]
            topk_ids = [ids[j] for j in topk_idx]
            topk_labels = [id_to_label[s] for s in topk_ids]
        else:
            topk_ids = []
            topk_labels = []

        true_lab = id_to_label[qid]
        topk_hit = true_lab in topk_labels

        if len(topk_labels) > 0:
            counts = pd.Series(topk_labels).value_counts()
            max_count = counts.max()
            majority = counts[counts == max_count].index.tolist()
        else:
            majority = []

        generous_tie = False

        if tie_break == "any_hit":
            if topk_hit:
                pred_label = true_lab
            else:
                pred_label = _choose_label_from_topk(topk_labels, tie_break="rank_sum") if topk_labels else \
                             next((lab for lab in labels_order_global if lab != true_lab), true_lab)

        elif tie_break in {"rank_sum", "majority"}:
            pred_label = _choose_label_from_topk(topk_labels, tie_break=tie_break)

        elif tie_break == "generous_majority":
            if len(majority) == 0:
                pred_label = next((lab for lab in labels_order_global if lab != true_lab), true_lab)
            elif true_lab in majority:
                pred_label = true_lab
                generous_tie = True
            else:
                first_majority = next((lab for lab in topk_labels if lab in majority), None)
                pred_label = first_majority if first_majority is not None else majority[0]
                generous_tie = False
        elif tie_break == "absolute":
            # Genuine strict majority, ported from analysis_lib's
            # `make_nn_predictions_tie_breaker_absolute`: need = floor(k/2)+1,
            # so at most one label can ever reach it and the "unique" test is
            # belt-and-braces. When nothing reaches it the retrieval did not
            # commit, and the honest prediction is AMBIGUOUS.
            #
            # AMBIGUOUS is deliberately NOT a class: `labels_order` is built
            # from the metadata's truth labels (cli.run), so an ambiguous
            # answer contributes a false negative to the true class and no
            # false positive anywhere -- i.e. it is scored as a miss, not as a
            # wrong class. If `labels_order` were ever allowed to default to
            # `set(y_true) | set(y_pred)`, AMBIGUOUS would silently become a
            # 16th zero-support class and deflate macro F1 by ~15/16.
            need = topk // 2 + 1
            if len(topk_labels) == 0:
                pred_label = AMBIGUOUS_LABEL
            else:
                reached = [lab for lab in counts.index if counts[lab] >= need]
                pred_label = reached[0] if len(reached) == 1 else AMBIGUOUS_LABEL

        elif tie_break == "closest":
            # Plurality with a nearest-neighbour tie-break: among the tied
            # argmax labels take whichever appears earliest in the ranked
            # top-k. Differs from generous_majority ONLY in not crediting the
            # truth for merely being present in the tied set.
            if len(majority) == 0:
                pred_label = next((lab for lab in labels_order_global if lab != true_lab), true_lab)
            else:
                pred_label = next((lab for lab in topk_labels if lab in majority), majority[0])

        else:
            raise ValueError(
                "tie_break must be 'rank_sum', 'majority', 'any_hit', "
                "'generous_majority', 'absolute', or 'closest'.")

        records.append({
            id_col: qid,
            "true_label": true_lab,
            "pred_label": pred_label,
            "topk_labels": topk_labels,
            "topk_hit": bool(topk_hit),
            "majority_set": majority,
            "generous_tie": int(generous_tie),
        })

    return pd.DataFrame.from_records(records)


def compute_metrics_from_predictions(
    df_pred: pd.DataFrame,
    labels_order: Optional[List[str]] = None,
) -> Dict:
    df = df_pred.dropna(subset=["true_label", "pred_label"]).copy()
    y_true = df["true_label"].tolist()
    y_pred = df["pred_label"].tolist()

    if labels_order is None:
        labels_order = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels_order).tolist()
    micro = f1_score(y_true, y_pred, average="micro", labels=labels_order)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels_order)
    macro_prec = precision_score(y_true, y_pred, average="macro", labels=labels_order, zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", labels=labels_order, zero_division=0)

    per_class = {}
    df_support = pd.Series(y_true).value_counts()
    for c in labels_order:
        y_true_bin = [1 if t == c else 0 for t in y_true]
        y_pred_bin = [1 if p == c else 0 for p in y_pred]
        f1_c = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        prec_c = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec_c = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        per_class[c] = {
            "support": int(df_support.get(c, 0)),
            "f1": float(f1_c),
            "precision": float(prec_c),
            "recall": float(rec_c),
        }

    return {
        "n_slides": int(len(df_pred)),
        "labels_order": labels_order,
        "confusion_matrix": cm,
        "micro_f1": float(micro),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "per_class": per_class,
    }


# -------------------------
# Cluster-stratified bootstrap over queries
# -------------------------

# slide_eval.py
import json
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from concurrent.futures import ProcessPoolExecutor, as_completed  # <-- new

# ... (unchanged load_* / filter_* / prediction / metrics code above) ...


# -------------------------
# Worker for parallel bootstrap
# -------------------------

def _bootstrap_worker(
    b_seed: int,
    df_pred_thin: pd.DataFrame,
    id_col: str,
    label_col: str,
    labels_order: List[str],
    mode: str,
    class_patient_to_slides: Dict[str, Dict[str, List[str]]],
    class_patient_weights: Dict[str, Dict[str, int]],
    class_S: Dict[str, int],
    class_P: Dict[str, int],
) -> Optional[Tuple[float, float, float, float, Dict[str, float]]]:
    """
    One bootstrap replicate, returns (micro, macro_f1, macro_prec, macro_rec, per_class_f1_dict)
    """
    rng = np.random.default_rng(b_seed)

    def sample_queries_for_class(c: str) -> List[str]:
        patients = list(class_patient_to_slides[c].keys())
        if len(patients) == 0:
            return []
        slides_sampled: List[str] = []
        if mode == "pps_cluster_one_slide":
            S = class_S[c]
            w = np.array([class_patient_weights[c][p] for p in patients], dtype=float)
            w = np.ones_like(w) / len(w) if w.sum() == 0 else (w / w.sum())
            chosen_pat_idx = rng.choice(len(patients), size=S, replace=True, p=w)
            for idx in chosen_pat_idx:
                p = patients[idx]
                slides = class_patient_to_slides[c][p]
                s = slides[rng.integers(len(slides))]
                slides_sampled.append(s)
        elif mode == "one_per_patient":
            P = class_P[c]
            if P == 0:
                return []
            chosen_pat_idx = rng.choice(len(patients), size=P, replace=True)
            for idx in chosen_pat_idx:
                p = patients[idx]
                slides = class_patient_to_slides[c][p]
                s = slides[rng.integers(len(slides))]
                slides_sampled.append(s)
        else:
            raise ValueError("mode must be 'pps_cluster_one_slide' or 'one_per_patient'")
        return slides_sampled

    sample_ids: List[str] = []
    for c in labels_order:
        sample_ids.extend(sample_queries_for_class(c))

    if len(sample_ids) == 0:
        return None

    counts = pd.Series(sample_ids).value_counts()
    df_sub = df_pred_thin[df_pred_thin[id_col].isin(counts.index)].copy()
    if df_sub.empty:
        return None
    df_sub = df_sub.loc[df_sub.index.repeat(df_sub[id_col].map(counts).astype(int))].copy()

    metrics_b = compute_metrics_from_predictions(df_sub, labels_order=labels_order)
    per_class_f1 = {c: metrics_b["per_class"][c]["f1"] for c in labels_order}
    return (
        metrics_b["micro_f1"],
        metrics_b["macro_f1"],
        metrics_b["macro_precision"],
        metrics_b["macro_recall"],
        per_class_f1,
    )


# -------------------------
# Cluster-stratified bootstrap (optionally parallel)
# -------------------------

def stratified_bootstrap_cis(
    df_pred: pd.DataFrame,
    meta: pd.DataFrame,
    B: int,
    seed: int,
    id_col: str,
    label_col: str,
    patient_col: str,
    labels_order: List[str],
    mode: str = "pps_cluster_one_slide",
    alpha: float = 0.05,
    n_jobs: int = 1,                           # <-- new
) -> Dict:
    """
    Cluster bootstrap with strata = label_col, cluster = patient_col.
    Optionally parallelized across bootstrap replicates via ProcessPoolExecutor.

    Modes:
      - "pps_cluster_one_slide" (default): preserve slides per class S_c using PPS patient draws.
      - "one_per_patient": one slide per sampled patient; class sizes change to ≈ #patients.
    """
    # Thin df_pred before shipping to workers (minimize pickling/memory)
    df_pred_thin = df_pred[[id_col, "true_label", "pred_label"]].copy()

    # Join prediction rows to meta to ensure we have class/patient for each query id
    m = meta[[id_col, label_col, patient_col]].copy()
    df = df_pred.merge(m, how="left", on=id_col, validate="one_to_one")
    if df[patient_col].isna().any():
        missing = df[df[patient_col].isna()][id_col].tolist()[:5]
        raise ValueError(f"Missing {patient_col} for some slides (e.g., {missing}).")

    # Build strata → patient → slide lists and counts/weights
    class_to_df = {c: df[df[label_col] == c].copy() for c in labels_order}
    class_patient_to_slides: Dict[str, Dict[str, List[str]]] = {}
    class_patient_weights: Dict[str, Dict[str, int]] = {}
    class_S: Dict[str, int] = {}
    class_P: Dict[str, int] = {}

    for c, cdf in class_to_df.items():
        group = cdf.groupby(patient_col)[id_col].agg(list)
        class_patient_to_slides[c] = group.to_dict()
        class_patient_weights[c] = {p: len(slides) for p, slides in class_patient_to_slides[c].items()}
        class_S[c] = int(len(cdf))
        class_P[c] = int(len(group))

    # Prepare seeds per replicate to ensure reproducibility across processes
    base_rng = np.random.default_rng(seed)
    rep_seeds = base_rng.integers(0, 2**63 - 1, size=B, dtype=np.int64).tolist()

    # Run bootstrap replicates
    micro_vals: List[float] = []
    macro_f1_vals: List[float] = []
    macro_prec_vals: List[float] = []
    macro_rec_vals: List[float] = []
    per_class_f1_list: Dict[str, List[float]] = {c: [] for c in labels_order}

    if n_jobs is None or n_jobs <= 1:
        # Serial
        for b in range(B):
            out = _bootstrap_worker(
                rep_seeds[b],
                df_pred_thin,
                id_col,
                label_col,
                labels_order,
                mode,
                class_patient_to_slides,
                class_patient_weights,
                class_S,
                class_P,
            )
            if out is None:
                continue
            mi, ma_f1, ma_p, ma_r, pc = out
            micro_vals.append(mi); macro_f1_vals.append(ma_f1); macro_prec_vals.append(ma_p); macro_rec_vals.append(ma_r)
            for c in labels_order:
                per_class_f1_list[c].append(pc[c])
    else:
        # Parallel
        # Cap workers to B to avoid spinning idle processes
        workers = min(int(n_jobs), int(B)) if B > 0 else 1
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = []
            for b in range(B):
                futures.append(ex.submit(
                    _bootstrap_worker,
                    rep_seeds[b],
                    df_pred_thin,
                    id_col,
                    label_col,
                    labels_order,
                    mode,
                    class_patient_to_slides,
                    class_patient_weights,
                    class_S,
                    class_P,
                ))
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Bootstrap (parallel)"):
                out = fut.result()
                if out is None:
                    continue
                mi, ma_f1, ma_p, ma_r, pc = out
                micro_vals.append(mi); macro_f1_vals.append(ma_f1); macro_prec_vals.append(ma_p); macro_rec_vals.append(ma_r)
                for c in labels_order:
                    per_class_f1_list[c].append(pc[c])

    def pct_ci(x: List[float], a: float) -> Tuple[float, float]:
        if len(x) == 0:
            return (float("nan"), float("nan"))
        lo = float(np.nanpercentile(x, 100 * (a / 2)))
        hi = float(np.nanpercentile(x, 100 * (1 - a / 2)))
        return (lo, hi)

    micro_ci = pct_ci(micro_vals, alpha)
    macro_f1_ci = pct_ci(macro_f1_vals, alpha)
    macro_prec_ci = pct_ci(macro_prec_vals, alpha)
    macro_rec_ci = pct_ci(macro_rec_vals, alpha)
    per_class_f1_ci = {c: pct_ci(per_class_f1_list[c], alpha) for c in labels_order}

    return {
        "micro_f1_ci": micro_ci,
        "macro_f1_ci": macro_f1_ci,
        "macro_precision_ci": macro_prec_ci,
        "macro_recall_ci": macro_rec_ci,
        "per_class_f1_ci": per_class_f1_ci,
    }

