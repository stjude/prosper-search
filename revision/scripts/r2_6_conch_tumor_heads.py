#!/usr/bin/env python3
"""R2.6 direct-1024 route, stage 2: patient-held-out CONCH v1.5 tumour heads,
held-out report, scoring of all 2,921 slides and TITAN patch-selection manifests.

Inputs: labels_1024/labels_1024.csv.gz (stage 1), the accepted R2.1
patient_folds.csv, the 2,921-slide metadata map and the existing CONCH v1.5
trident features (40x, 1024 px, 0 overlap).

Recipe (R2.1 analogue, copied, not imported; `prosper` is never imported):
- pool for held-out fold f = patches labelled 1..5 from patients whose fold != f;
- budget 100 % = whole pool; budget 5 % = per class, the first
  five_percent_count(n_class) patches in training-hash order, where the hash is
  the R2.1 training_tile_hash analogue
  sha256(json([NAMESPACE, 1234, patient, slide, 0, x, y, 1024, label])) with
  NAMESPACE = "r2_6_training_patch_1024_v1" (R2.1 used "r2_1_training_tile_v1"
  and 512); the selected matrix keeps the pool's (hash, physical key) order;
- SGDClassifier with R2.1 _PARAMETERS (verbose 0 instead of 1; logging only),
  class_weight[c] = n_selected / (5 * n_selected_c), FP32 features, native
  thread pools of one; no retries, convergence warnings retained.
Scoring: patients with annotation (patient_folds has_annotations == 1) use their
held-out-fold head for every one of their slides; all other slides use the mean of
the five heads' probabilities of the same budget. pred_argmax = class with the
largest (mean) probability, ties -> lowest label.
Selections: b100_argmax (pred_argmax == Tumor, 100 % heads), b5_argmax (5 %
heads), b100_strict (p_tumor >= 0.9, 100 % heads).

Stages: --stage fit | score | all. Writes only under
$PROSPER_OUT/r2_6_titan_direct1024_20261004/conch_heads[...]/.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"  # R2.1 fitted with native pools of one; set before numpy import

import argparse
import datetime
import hashlib
import io
import json
import math
import pickle
import platform
import struct
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HAORAN = "/PROSPER_WORKSPACE"
K = HAORAN + "/prosper-search"
FIT_SRC = K + "/prosper/evaluation/r2_1_hoptimus_fit.py"
FILTER_SRC = K + "/prosper/evaluation/r2_1_hoptimus_filter.py"
TRIDENT_DIR = HAORAN + "/comet-feature-gen/comet-titan/trident/trident_processed/40x_1024px_0px_overlap"
METADATA = HAORAN + "/comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv"
OUT_NAME = "r2_6_titan_direct1024_20261004"
LABELS_SUB = "labels_1024"
SUB = "conch_heads"
R21_FOLDS = "r2_1_cpu_20261002/filter_attempt_001/patient_folds.csv"
AUGUST_SUMMARY = "s2_titan_tumor_only/40x_1024px_0px_overlap/_tumour_patch_summary.csv"
NAMESPACE = "r2_6_training_patch_1024_v1"
PATCH = 1024
WIDTH = 768
CLASSES = (1, 2, 3, 4, 5)
BUDGETS = (100, 5)
STRICT_THRESHOLD = 0.9
MIN_KEPT = 8
R21_VERSIONS = {"scikit-learn": "1.7.0", "numpy": "2.2.4", "scipy": "1.16.0"}
# R2.1 _PARAMETERS (r2_1_hoptimus_fit.py L49-57) with verbose 1 -> 0 (stdout logging only).
PARAMETERS = {
    "loss": "log_loss", "penalty": "l2", "alpha": 0.001,
    "max_iter": 1000, "tol": 0.001, "random_state": 1234,
    "fit_intercept": True, "shuffle": True, "learning_rate": "optimal",
    "early_stopping": False, "n_iter_no_change": 5,
    "warm_start": False, "average": False, "n_jobs": 1, "verbose": 0,
    "l1_ratio": 0.15, "epsilon": 0.1, "eta0": 0.0, "power_t": 0.5,
    "validation_fraction": 0.1,
}
COPIED = {
    FILTER_SRC: ["ManifestValidationError", "_integer", "_json_sha", "five_percent_count"],
    FIT_SRC: ["HeadValidationError", "_require", "_float", "_json_bytes", "_typed", "_fingerprint"],
}


# ---------------------------------------------------------------- copied verbatim
# from prosper/evaluation/r2_1_hoptimus_filter.py (sha256 recorded in reconciliation)
class ManifestValidationError(ValueError):
    """An identity, population or training-support requirement is violated."""


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ManifestValidationError(f"{name} must be an integer >= {minimum}")


def _json_sha(values: list) -> str:
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def five_percent_count(class_count: int) -> int:
    """Round n/20 to nearest, ties to even, with the agreed minimum of one."""
    _integer(class_count, "class_count", minimum=1)
    quotient, remainder = divmod(class_count, 20)
    rounded = quotient + int(remainder > 10 or (remainder == 10 and quotient % 2 == 1))
    return max(1, rounded)


# from prosper/evaluation/r2_1_hoptimus_fit.py (sha256 recorded in reconciliation)
class HeadValidationError(ValueError):
    """A fitting, identity, support, state or prediction requirement failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeadValidationError(message)


def _float(value: Any, name: str) -> None:
    _require(type(value) is float and math.isfinite(value),
             f"{name} must be a finite float")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _typed(value: Any) -> Any:
    """Canonical metadata retains scalar and dictionary-key types."""
    if value is None:
        return ["none"]
    if type(value) in (str, bool, int):
        return [type(value).__name__, value]
    if type(value) is float:
        _float(value, "state metadata")
        return ["float", value]
    if type(value) is dict:
        items = [[_typed(key), _typed(item)] for key, item in value.items()]
        return ["dict", sorted(items, key=lambda item: _json_bytes(item[0]))]
    if type(value) in (list, tuple):
        return [type(value).__name__, [_typed(item) for item in value]]
    raise HeadValidationError(f"unsupported state metadata type {type(value).__name__}")


def _fingerprint(estimator: SGDClassifier) -> str:
    digest = hashlib.sha256()
    metadata = {
        "parameters": estimator.get_params(deep=False),
        "n_features_in_": estimator.n_features_in_,
        "n_iter_": estimator.n_iter_, "t_": estimator.t_,
    }
    pieces = [("metadata", _json_bytes(_typed(metadata)))]
    for name in ("classes_", "coef_", "intercept_"):
        array = getattr(estimator, name)
        pieces.append((name + "_metadata",
                       _json_bytes(_typed((array.shape, array.dtype.str)))))
        pieces.append((name + "_values", array.tobytes(order="C")))
    for name, data in pieces:
        tag = name.encode("ascii")
        digest.update(struct.pack(">Q", len(tag)))
        digest.update(tag)
        digest.update(struct.pack(">Q", len(data)))
        digest.update(data)
    return digest.hexdigest()
# ---------------------------------------------------------------- end of copies


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def training_patch_hash(patient_id: str, slide_id: str, x: int, y: int, label: int) -> str:
    """R2.1 training_tile_hash analogue for level-0 1024-px patches."""
    return _json_sha([NAMESPACE, 1234, str(patient_id), str(slide_id), 0, int(x), int(y),
                      PATCH, int(label)])


def hash_order(rows) -> np.ndarray:
    """Positions of `rows` (DataFrame: patient_id, slide_id, x, y, label) sorted by
    (training hash, physical key (slide_id, 0, x, y, 1024)) as in R2.1."""
    keys = [(training_patch_hash(p, s, x, y, lab), (str(s), 0, int(x), int(y), PATCH), i)
            for i, (p, s, x, y, lab) in enumerate(zip(rows["patient_id"], rows["slide_id"],
                                                      rows["x"], rows["y"], rows["label"]))]
    keys.sort()
    return np.array([k[2] for k in keys], dtype=np.int64), [k[0] for k in keys]


def build_manifests(rows, order: np.ndarray) -> dict:
    """{(fold, budget): dict(idx=positions in hash order, class_stats, weights, ...)}."""
    labels = rows["label"].to_numpy()
    folds = rows["held_out_fold"].to_numpy()
    out = {}
    for f in range(5):
        pool = order[folds[order] != f]
        by_class = {c: pool[labels[pool] == c] for c in CLASSES}
        absent = [c for c, v in by_class.items() if len(v) == 0]
        if absent:
            raise ManifestValidationError(f"held-out fold {f}: training pool lacks classes {absent}")
        chosen = np.zeros(len(rows), bool)
        for c, v in by_class.items():
            chosen[v[:five_percent_count(int(len(v)))]] = True
        primary = pool[chosen[pool]]
        for budget, sel in ((5, primary), (100, pool)):
            counts = {c: int((labels[sel] == c).sum()) for c in CLASSES}
            weights = {c: len(sel) / (5 * counts[c]) for c in CLASSES}
            out[(f, budget)] = {
                "idx": sel, "pool_count": int(len(pool)), "selected_count": int(len(sel)),
                "class_stats": [{"tissue_label": c, "pool_count": int(len(by_class[c])),
                                 "selected_count": counts[c],
                                 "fraction": counts[c] / len(by_class[c]),
                                 "balanced_weight": weights[c]} for c in CLASSES],
                "weights": weights,
            }
    return out


def make_estimator(weights: dict):
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(**{**PARAMETERS, "class_weight": {int(c): float(w) for c, w in weights.items()}})


def fit_one(task):
    """Fit one head in a worker. features_path is an .npy of labelled FP32 rows."""
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_info
    f, budget, features_path, idx, labels, weights = task
    t0 = time.time()
    feats = np.load(features_path, mmap_mode="r")
    X = np.ascontiguousarray(feats[idx], dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    est = make_estimator(weights)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(X, y, sample_weight=None)
    unexpected = [w for w in caught if not issubclass(w.category, ConvergenceWarning)]
    if unexpected:
        raise HeadValidationError(f"fold {f} budget {budget}: unexpected warnings "
                                  + "; ".join(str(w.message) for w in unexpected))
    _require(est.classes_.tolist() == list(CLASSES), "fitted classes must be [1..5]")
    _require(est.coef_.dtype == np.float32 and est.coef_.shape == (5, WIDTH)
             and bool(np.isfinite(est.coef_).all()), "coef_ must be finite FP32 (5,768)")
    conv = [str(w.message) for w in caught]
    diag = {
        "held_out_fold": f, "budget_percent": budget,
        "n_iter": int(est.n_iter_), "t": float(est.t_),
        "n_iter_semantics": "maximum_across_five_one_vs_rest_fits",
        "convergence_warnings": conv,
        "converged": (not conv) and int(est.n_iter_) < PARAMETERS["max_iter"],
        "feature_sha256": hashlib.sha256(memoryview(X).cast("B")).hexdigest(),
        "feature_shape": list(X.shape),
        "fitted_state_sha256": _fingerprint(est),
        "seconds": round(time.time() - t0, 1),
        "native_pools": [{k: p.get(k) for k in ("internal_api", "num_threads", "version")}
                         for p in threadpool_info()],
    }
    return f, budget, pickle.dumps(est, protocol=5), diag


# ---------------------------------------------------------------- held-out metrics
def heldout_metrics(truth: np.ndarray, proba: np.ndarray, patients: np.ndarray) -> tuple[dict, list]:
    pred = np.array(CLASSES)[np.argmax(proba, axis=1)]
    cm = np.zeros((5, 5), np.int64)
    for t, p in zip(truth, pred):
        cm[t - 1, p - 1] += 1
    m = {"n": int(len(truth)), "accuracy": float((pred == truth).mean()) if len(truth) else float("nan")}
    for i, c in enumerate(CLASSES):
        m[f"support_{c}"] = int(cm[i].sum())
        m[f"recall_{c}"] = float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan")
        m[f"precision_{c}"] = float(cm[i, i] / cm[:, i].sum()) if cm[:, i].sum() else float("nan")
    rec = [m[f"recall_{c}"] for c in CLASSES if m[f"support_{c}"]]
    m["balanced_accuracy_present_classes"] = float(np.mean(rec)) if rec else float("nan")
    tt = truth == 1
    for name, pt in (("argmax", pred == 1), ("strict0.9", proba[:, 0] >= STRICT_THRESHOLD)):
        tp, fn = int((tt & pt).sum()), int((tt & ~pt).sum())
        tn, fp = int((~tt & ~pt).sum()), int((~tt & pt).sum())
        m[f"tumor_{name}_tp"], m[f"tumor_{name}_fn"], m[f"tumor_{name}_tn"], m[f"tumor_{name}_fp"] = tp, fn, tn, fp
        m[f"tumor_{name}_sensitivity"] = tp / (tp + fn) if tp + fn else float("nan")
        m[f"tumor_{name}_specificity"] = tn / (tn + fp) if tn + fp else float("nan")
        m[f"tumor_{name}_ppv"] = tp / (tp + fp) if tp + fp else float("nan")
    per_patient = []
    pt = pred == 1
    for p in sorted(set(patients.tolist())):
        sel = patients == p
        per_patient.append({"patient_id": p, "n": int(sel.sum()), "n_tumor_truth": int(tt[sel].sum()),
                            "tumor_vs_rest_accuracy": float((pt[sel] == tt[sel]).mean())})
    acc = np.array([r["tumor_vs_rest_accuracy"] for r in per_patient])
    m["n_patients"] = len(per_patient)
    if len(acc):
        for q, v in zip(("min", "q25", "median", "q75", "max"), np.quantile(acc, [0, .25, .5, .75, 1])):
            m[f"patient_tumor_vs_rest_accuracy_{q}"] = float(v)
        m["patient_tumor_vs_rest_accuracy_mean"] = float(acc.mean())
        m["patients_tumor_vs_rest_accuracy_lt_0.8"] = int((acc < 0.8).sum())
    m["confusion_rows_truth_cols_pred_1to5"] = json.dumps(cm.tolist())
    return m, per_patient


# ---------------------------------------------------------------- scoring
def route_probabilities(heads: dict, budget: int, features: np.ndarray, fold) -> tuple[np.ndarray, str]:
    """Held-out-fold head for annotated patients (fold is int), else mean of the 5 heads."""
    if fold is not None:
        return heads[(int(fold), budget)].predict_proba(features), f"oof_fold{int(fold)}"
    stack = np.stack([heads[(f, budget)].predict_proba(features) for f in range(5)])
    return stack.mean(axis=0, dtype=np.float64), "mean5"


def argmax_label(proba: np.ndarray) -> np.ndarray:
    return np.array(CLASSES, dtype=np.int64)[np.argmax(proba, axis=1)]  # first max -> lowest label


_HEADS = None


def _init_scorer(head_paths):
    global _HEADS
    _HEADS = {}
    for key, path in head_paths.items():
        with open(path, "rb") as fh:
            _HEADS[key] = pickle.load(fh)


def score_slide(task):
    import h5py
    slide, fold = task
    fpath = os.path.join(TRIDENT_DIR, "features_conch_v15", f"{slide}.h5")
    ppath = os.path.join(TRIDENT_DIR, "patches", f"{slide}_patches.h5")
    raw = Path(fpath).read_bytes()
    rec = {"slide_id": slide, "conch_h5_sha256": hashlib.sha256(raw).hexdigest()}
    with h5py.File(io.BytesIO(raw), "r") as h:
        coords = h["coords"][:]
        feats = np.ascontiguousarray(h["features"][:], dtype=np.float32)
    del raw
    with h5py.File(ppath, "r") as h:
        pcoords = h["coords"][:]
    if not np.array_equal(coords, pcoords):
        raise ValueError(f"{slide}: CONCH coords differ from patches h5 coords")
    if feats.shape != (coords.shape[0], WIDTH) or not np.isfinite(feats).all():
        raise ValueError(f"{slide}: bad features {feats.shape}")
    out = {"x": coords[:, 0].astype(np.int64), "y": coords[:, 1].astype(np.int64)}
    for b in BUDGETS:
        proba, src = route_probabilities(_HEADS, b, feats, fold)
        out[f"p_tumor_b{b}"] = proba[:, 0].astype(np.float64)
        out[f"pred_b{b}"] = argmax_label(proba)
        out["head_source"] = src
    rec["n_patches"] = int(coords.shape[0])
    return rec, out


def selection_tables(scores: dict, slides: list) -> tuple[dict, list]:
    """scores: slide -> out dict. Returns kept-coordinate frames and per-slide summary rows."""
    import pandas as pd
    variants = {
        "b100_argmax": lambda o: o["pred_b100"] == 1,
        "b5_argmax": lambda o: o["pred_b5"] == 1,
        "b100_strict": lambda o: o["p_tumor_b100"] >= STRICT_THRESHOLD,
    }
    kept, summary = {v: [] for v in variants}, []
    for s in slides:
        o = scores[s]
        n = len(o["x"])
        for v, rule in variants.items():
            k = rule(o)
            kept[v].append(pd.DataFrame({"slide_id": s, "x": o["x"][k], "y": o["y"][k]}))
            nk = int(k.sum())
            summary.append({"slide_id": s, "variant": v, "head_source": o["head_source"],
                            "n_patches": n, "n_kept": nk, "frac_kept": nk / n if n else float("nan"),
                            "ge8": nk >= MIN_KEPT})
    return {v: pd.concat(fr, ignore_index=True) for v, fr in kept.items()}, summary


# ---------------------------------------------------------------- stages
def load_labelled(out_root: Path):
    import pandas as pd
    lab = pd.read_csv(out_root / LABELS_SUB / "labels_1024.csv.gz",
                      dtype={"slide_id": str, "patient_id": str})
    lab = lab[lab["label_1024"].between(1, 5)].rename(columns={"label_1024": "label"}).reset_index(drop=True)
    return lab


def stage_fit(args, out_root: Path, out_dir: Path) -> dict:
    import h5py
    import pandas as pd
    t0 = time.time()
    folds = pd.read_csv(Path(os.environ["PROSPER_OUT"]) / R21_FOLDS, dtype={"patient_id": str})
    fold_of = dict(zip(folds["patient_id"], folds["fold"].astype(int)))
    rows = load_labelled(out_root)
    bad = rows[rows["patient_id"].map(fold_of) != rows["held_out_fold"]]
    if len(bad):
        raise ValueError("labels held_out_fold disagrees with patient_folds.csv")
    labels_rec = json.loads((out_root / LABELS_SUB / "reconciliation.json").read_text())
    conch_sha = {s: d["conch_h5_sha256"] for s, d in labels_rec["per_slide_input_sha256"].items()}

    # labelled FP32 features in label-row order, coordinates checked at feature_row
    feats = np.empty((len(rows), WIDTH), np.float32)
    for s, g in rows.groupby("slide_id", sort=True):
        fpath = os.path.join(TRIDENT_DIR, "features_conch_v15", f"{s}.h5")
        raw = Path(fpath).read_bytes()
        if hashlib.sha256(raw).hexdigest() != conch_sha[s]:
            raise ValueError(f"{s}: CONCH h5 changed since the labels stage")
        with h5py.File(io.BytesIO(raw), "r") as h:
            c = h["coords"][:]
            fr = g["feature_row"].to_numpy()
            if not np.array_equal(c[fr], g[["x", "y"]].to_numpy()):
                raise ValueError(f"{s}: feature_row/coordinate mismatch")
            feats[g.index.to_numpy()] = h["features"][:][fr]
    if not np.isfinite(feats).all():
        raise ValueError("non-finite labelled features")
    scratch = out_root / "scratch"
    scratch.mkdir(exist_ok=True)
    fpath = scratch / f"labelled_conch_features_{os.getpid()}.npy"
    np.save(fpath, feats)

    order, hashes = hash_order(rows)
    manifests = build_manifests(rows, order)
    y_all = rows["label"].to_numpy(np.int64)
    tasks = [(f, b, str(fpath), m["idx"], y_all[m["idx"]], m["weights"])
             for (f, b), m in sorted(manifests.items(), key=lambda kv: (-kv[0][1], kv[0][0]))]
    heads_dir = out_dir / "heads"
    heads_dir.mkdir()
    heads, diags = {}, []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
        for f, b, blob, d in ex.map(fit_one, tasks):
            p = heads_dir / f"conch_head_fold{f}_b{b}.pkl"
            p.write_bytes(blob)
            heads[(f, b)] = pickle.loads(blob)
            _require(_fingerprint(heads[(f, b)]) == d["fitted_state_sha256"], "pickle round-trip changed state")
            m = manifests[(f, b)]
            sel = rows.iloc[m["idx"]]
            d.update(pickle_file=p.name, pickle_sha256=sha256_file(p),
                     pool_count=m["pool_count"], selected_count=m["selected_count"],
                     selected_slides=int(sel["slide_id"].nunique()),
                     selected_patients=int(sel["patient_id"].nunique()),
                     class_stats=[{**cs, "selected_patients": int(sel.loc[sel["label"] == cs["tissue_label"],
                                                                              "patient_id"].nunique())}
                                  for cs in m["class_stats"]],
                     class_weight={str(c): w for c, w in m["weights"].items()},
                     ordered_patch_keys_sha256=_json_sha([[str(s), int(x), int(yy)] for s, x, yy in
                                                          zip(sel["slide_id"], sel["x"], sel["y"])]),
                     parameters={**PARAMETERS, "class_weight": {str(c): w for c, w in m["weights"].items()}})
            diags.append(d)
            print(f"[{time.time() - t0:7.1f}s] fit fold {f} b{b}: n={m['selected_count']} n_iter={d['n_iter']} "
                  f"converged={d['converged']} {d['seconds']}s", flush=True)
    fpath.unlink()
    diags.sort(key=lambda d: (-d["budget_percent"], d["held_out_fold"]))
    (out_dir / "fit_diagnostics.json").write_text(json.dumps({
        "namespace": NAMESPACE, "classes": list(CLASSES),
        "selection_rule": "5%: per class first five_percent_count(n) in (hash, physical key) order; 100%: whole pool",
        "heads": diags}, indent=1))

    # held-out evaluation
    metrics, per_patient_all = [], []
    pred_dir = out_dir / "held_out_predictions"
    pred_dir.mkdir()
    for (f, b) in sorted(heads, key=lambda k: (-k[1], k[0])):
        te = np.flatnonzero(rows["held_out_fold"].to_numpy() == f)
        proba = heads[(f, b)].predict_proba(feats[te])
        truth = y_all[te]
        m, pp = heldout_metrics(truth, proba, rows["patient_id"].to_numpy()[te])
        metrics.append({"held_out_fold": f, "budget_percent": b, **m})
        per_patient_all += [{"held_out_fold": f, "budget_percent": b, **r} for r in pp]
        sub = rows.iloc[te][["slide_id", "patient_id", "x", "y", "label"]].copy()
        for i, c in enumerate(CLASSES):
            sub[f"p{c}"] = proba[:, i]
        sub["pred_argmax"] = argmax_label(proba)
        sub.to_csv(pred_dir / f"held_out_predictions_fold{f}_b{b}.csv.gz", index=False, float_format="%.7g")
    mdf = pd.DataFrame(metrics)
    mdf.to_csv(out_dir / "held_out_metrics.csv", index=False)
    pd.DataFrame(per_patient_all).to_csv(out_dir / "held_out_per_patient.csv", index=False)
    pooled = {}
    for b in BUDGETS:
        g = mdf[mdf["budget_percent"] == b]
        tp, fn = g["tumor_argmax_tp"].sum(), g["tumor_argmax_fn"].sum()
        tn, fp = g["tumor_argmax_tn"].sum(), g["tumor_argmax_fp"].sum()
        pooled[str(b)] = {"pooled_oof_tumor_argmax_sensitivity": float(tp / (tp + fn)),
                          "pooled_oof_tumor_argmax_specificity": float(tn / (tn + fp)),
                          "pooled_oof_accuracy": float((g["accuracy"] * g["n"]).sum() / g["n"].sum())}
    return {"seconds_fit_stage": round(time.time() - t0, 1), "n_labelled_patches": int(len(rows)),
            "label_counts": {str(c): int((y_all == c).sum()) for c in CLASSES},
            "pooled_oof": pooled, "ordered_training_hash_count": len(hashes)}


def load_heads(out_dir: Path) -> dict:
    diag = json.loads((out_dir / "fit_diagnostics.json").read_text())
    paths = {}
    for d in diag["heads"]:
        p = out_dir / "heads" / d["pickle_file"]
        if sha256_file(p) != d["pickle_sha256"]:
            raise ValueError(f"{p} changed since fit")
        with open(p, "rb") as fh:
            if _fingerprint(pickle.load(fh)) != d["fitted_state_sha256"]:
                raise ValueError(f"{p}: fitted-state fingerprint mismatch")
        paths[(d["held_out_fold"], d["budget_percent"])] = str(p)
    if sorted(paths) != sorted((f, b) for f in range(5) for b in BUDGETS):
        raise ValueError("expected 10 heads")
    return paths


def stage_score(args, out_root: Path, fit_dir: Path, out_dir: Path) -> dict:
    import pandas as pd
    t0 = time.time()
    head_paths = load_heads(fit_dir)
    md = pd.read_csv(METADATA, dtype=str)
    if md["Slide.Scan.File"].duplicated().any():
        raise ValueError("duplicate slide in metadata")
    folds = pd.read_csv(Path(os.environ["PROSPER_OUT"]) / R21_FOLDS, dtype={"patient_id": str})
    ann_fold = dict(zip(folds.loc[folds["has_annotations"] == 1, "patient_id"],
                        folds.loc[folds["has_annotations"] == 1, "fold"].astype(int)))
    conch = sorted(p.name[:-3] for p in (Path(TRIDENT_DIR) / "features_conch_v15").glob("*.h5"))
    if set(conch) != set(md["Slide.Scan.File"]) or len(conch) != 2921:
        raise ValueError("CONCH feature set differs from the 2,921-slide metadata")
    patient = dict(zip(md["Slide.Scan.File"], md["UID.Subject"]))
    slides = sorted(conch, key=lambda s: (len(s), s))
    labelled_slides = set(load_labelled(out_root)["slide_id"])
    if args.score_sample:
        ranked = sorted(slides, key=lambda s: hashlib.sha256(f"r2_6_score_sample_v1|{s}".encode()).hexdigest())
        keep = set(ranked[:args.score_sample])
        slides = [s for s in slides if s in keep]
    tasks = [(s, ann_fold.get(patient[s])) for s in slides]
    scores, recs = {}, []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_scorer,
                             initargs=(head_paths,)) as ex:
        for k, (rec, out) in enumerate(ex.map(score_slide, tasks, chunksize=4)):
            scores[rec["slide_id"]] = out
            recs.append(rec)
            if (k + 1) % 200 == 0:
                print(f"[{time.time() - t0:7.1f}s] scored {k + 1}/{len(tasks)}", flush=True)
    t_score = time.time() - t0

    # OOF consistency: labelled patches scored here == held-out predictions of the same head
    oof_check = {}
    for f in range(5):
        for b in BUDGETS:
            hp = pd.read_csv(fit_dir / "held_out_predictions" / f"held_out_predictions_fold{f}_b{b}.csv.gz",
                             dtype={"slide_id": str})
            hp = hp[hp["slide_id"].isin(scores)]
            n_bad, n = 0, 0
            for s, g in hp.groupby("slide_id"):
                o = scores[s]
                pos = pd.Series(np.arange(len(o["x"])), index=pd.MultiIndex.from_arrays([o["x"], o["y"]]))
                i = pos.loc[list(zip(g["x"], g["y"]))].to_numpy()
                n += len(i)
                n_bad += int((np.abs(o[f"p_tumor_b{b}"][i] - g["p1"].to_numpy()) > 1e-6).sum()
                             + (o[f"pred_b{b}"][i] != g["pred_argmax"].to_numpy()).sum())
            oof_check[f"fold{f}_b{b}"] = {"n": n, "n_mismatch": n_bad}
    if any(v["n_mismatch"] for v in oof_check.values()):
        raise ValueError(f"OOF scoring inconsistent with held-out predictions: {oof_check}")

    for b in BUDGETS:
        frames = [pd.DataFrame({"slide_id": s, "x": scores[s]["x"], "y": scores[s]["y"],
                                "p_tumor": scores[s][f"p_tumor_b{b}"], "pred_argmax": scores[s][f"pred_b{b}"],
                                "head_source": scores[s]["head_source"]}) for s in slides]
        pd.concat(frames, ignore_index=True).to_csv(out_dir / f"scores_b{b}.csv.gz", index=False,
                                                    float_format="%.7g")
    kept, summ = selection_tables(scores, slides)
    for v, df in kept.items():
        df.to_csv(out_dir / f"selection_{v}.csv.gz", index=False)
    sdf = pd.DataFrame(summ)
    aug = pd.read_csv(Path(os.environ["PROSPER_OUT"]) / AUGUST_SUMMARY, dtype={"slide": str})
    aug = aug.rename(columns={"slide": "slide_id", "n_patches": "aug_n_patches", "n_kept": "aug_n_kept",
                              "frac_kept": "aug_frac_kept"})[["slide_id", "aug_n_patches", "aug_n_kept",
                                                               "aug_frac_kept"]]
    sdf = sdf.merge(aug, on="slide_id", how="left")
    sdf["annotation_slide"] = sdf["slide_id"].isin(labelled_slides)
    sdf.to_csv(out_dir / "selection_summary.csv", index=False)

    comp = {}
    for v, g in sdf.groupby("variant"):
        both = g.dropna(subset=["aug_frac_kept"])
        d = both["frac_kept"] - both["aug_frac_kept"]
        comp[v] = {
            "n_slides": int(len(g)), "head_source_counts": g["head_source"].map(
                lambda s: "oof" if s.startswith("oof") else s).value_counts().to_dict(),
            "patches_total": int(g["n_patches"].sum()), "kept_total": int(g["n_kept"].sum()),
            "frac_kept_pooled": float(g["n_kept"].sum() / g["n_patches"].sum()),
            "frac_kept_slide_mean": float(g["frac_kept"].mean()),
            "frac_kept_slide_median": float(g["frac_kept"].median()),
            "slides_lt8_kept": int((~g["ge8"]).sum()),
            "slides_zero_kept": int((g["n_kept"] == 0).sum()),
            "august_overlap_slides": int(len(both)),
            "august_n_patches_equal": int((both["n_patches"] == both["aug_n_patches"]).sum()),
            "pearson_frac_kept_vs_august": float(np.corrcoef(both["frac_kept"], both["aug_frac_kept"])[0, 1])
            if len(both) > 2 else None,
            "spearman_frac_kept_vs_august": float(both["frac_kept"].rank().corr(both["aug_frac_kept"].rank()))
            if len(both) > 2 else None,
            "mean_diff_frac_kept_minus_august": float(d.mean()) if len(both) else None,
            "median_abs_diff": float(d.abs().median()) if len(both) else None,
            "august_slides_lt8_in_overlap": int((both["aug_n_kept"] < MIN_KEPT).sum()),
            "this_slides_lt8_in_overlap": int((both["n_kept"] < MIN_KEPT).sum()),
        }
    (out_dir / "selection_comparison.json").write_text(json.dumps(comp, indent=1))
    return {"seconds_scoring": round(t_score, 1), "seconds_score_stage": round(time.time() - t0, 1),
            "n_slides_scored": len(slides), "score_sample": args.score_sample,
            "n_patches_scored": int(sum(r["n_patches"] for r in recs)),
            "annotated_patients_oof": len(ann_fold),
            "oof_consistency": oof_check, "selection_comparison": comp,
            "per_slide_conch_h5_sha256": {r["slide_id"]: r["conch_h5_sha256"] for r in recs},
            "head_pickles": {f"fold{k[0]}_b{k[1]}": {"path": v, "sha256": sha256_file(v)}
                             for k, v in sorted(head_paths.items())}}


def environment() -> dict:
    import importlib.metadata
    import h5py
    import pandas
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info
    vers = {n: importlib.metadata.version(n) for n in R21_VERSIONS}
    return {"python": sys.version, "platform": platform.platform(), "host": platform.node(),
            "versions": vers, "r2_1_pins": R21_VERSIONS, "versions_equal_r2_1_pins": vers == R21_VERSIONS,
            "pandas": pandas.__version__, "h5py": h5py.__version__, "sklearn_path": sklearn.__file__,
            "numpy_path": np.__file__, "scipy": scipy.__version__,
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                          "MKL_NUM_THREADS")},
            "native_pools_main": [{k: p.get(k) for k in ("internal_api", "num_threads", "version")}
                                  for p in threadpool_info()]}


def write_reconciliation(path: Path, out_dir: Path, args, t_start, payload: dict, inputs: dict):
    rec = {
        "run_id": f"{OUT_NAME}/{out_dir.name}", "created": datetime.datetime.now().astimezone().isoformat(),
        "argv": sys.argv, "seconds": round(time.time() - t_start, 1), "workers": args.workers,
        "inputs": {k: {"path": v, "sha256": sha256_file(v), "bytes": os.path.getsize(v)} for k, v in inputs.items()},
        "copied_source": {p: {"sha256": sha256_file(p), "objects": names} for p, names in COPIED.items()},
        "parameters": PARAMETERS, "namespace": NAMESPACE, "environment": environment(), **payload,
    }
    rec["outputs"] = {str(p.relative_to(out_dir)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                      for p in sorted(out_dir.rglob("*")) if p.is_file() and p != path}
    path.write_text(json.dumps(rec, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("fit", "score", "all"), default="all")
    ap.add_argument("--workers", type=int, default=6, help="<= 6 on the shared login node")
    ap.add_argument("--score-sample", type=int, default=0,
                    help="score only a deterministic hash-ordered sample of N slides (0 = all 2,921)")
    args = ap.parse_args(argv)
    if args.workers > 6:
        sys.exit("--workers must be <= 6 on the shared node")
    t_start = time.time()
    if not os.environ.get("PROSPER_OUT"):
        sys.exit("PROSPER_OUT must be exported")
    out_root = Path(os.environ["PROSPER_OUT"]) / OUT_NAME
    fit_dir = out_root / SUB
    inputs = {"labels_1024": str(out_root / LABELS_SUB / "labels_1024.csv.gz"),
              "labels_reconciliation": str(out_root / LABELS_SUB / "reconciliation.json"),
              "patient_folds": str(Path(os.environ["PROSPER_OUT"]) / R21_FOLDS),
              "slide_metadata": METADATA, "this_script": os.path.abspath(__file__)}
    if args.stage in ("fit", "all"):
        if (fit_dir / "reconciliation_fit.json").exists() or (fit_dir / "heads").exists():
            sys.exit(f"{fit_dir} already holds fitted heads; refusing to overwrite")
        fit_dir.mkdir(parents=True, exist_ok=True)
        payload = stage_fit(args, out_root, fit_dir)
        write_reconciliation(fit_dir / "reconciliation_fit.json", fit_dir, args, t_start, payload, inputs)
        print(json.dumps({k: payload[k] for k in ("seconds_fit_stage", "label_counts", "pooled_oof")}))
    if args.stage in ("score", "all"):
        score_dir = fit_dir if not args.score_sample else out_root / f"{SUB}_score_sample{args.score_sample}"
        score_dir.mkdir(parents=True, exist_ok=True)
        if (score_dir / "reconciliation_score.json").exists():
            sys.exit(f"{score_dir}/reconciliation_score.json exists; refusing to overwrite")
        inputs_s = {**inputs, "august_selection_summary": str(Path(os.environ["PROSPER_OUT"]) / AUGUST_SUMMARY),
                    "fit_diagnostics": str(fit_dir / "fit_diagnostics.json")}
        payload = stage_score(args, out_root, fit_dir, score_dir)
        write_reconciliation(score_dir / "reconciliation_score.json", score_dir, args, t_start, payload, inputs_s)
        print(json.dumps({k: payload[k] for k in ("seconds_score_stage", "n_slides_scored")}))


if __name__ == "__main__":
    main()
