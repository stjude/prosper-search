"""Synthetic-release array primitives; no asset IO or model serialization.

Assembly preserves accepted manifest order. A single selected manifest cannot
prove unseen pool membership, and a bare matrix cannot prove row identity.
Trusted callers must later bind the accepted manifests and physical sources.
Fitted-state hashes observe current state, not mutate-and-restore history.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import inspect
import json
import math
import struct
from types import MappingProxyType
from typing import Any
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from threadpoolctl import threadpool_info

from .r2_1_hoptimus_filter import (
    AnnotationRecord,
    ClassSelection,
    ManifestValidationError,
    TrainingManifest,
    five_percent_count,
    training_tile_hash,
    validate_manifests,
)


class HeadValidationError(ValueError):
    """A fitting, identity, support, state or prediction requirement failed."""


_WIDTH = 1536
_CLASSES = (1, 2, 3, 4, 5)
_VERSIONS = MappingProxyType({
    "scikit-learn": "1.7.0", "numpy": "2.2.4", "scipy": "1.16.0",
})
_PARAMETERS = MappingProxyType({
    "loss": "log_loss", "penalty": "l2", "alpha": 0.001,
    "max_iter": 1000, "tol": 0.001, "random_state": 1234,
    "fit_intercept": True, "shuffle": True, "learning_rate": "optimal",
    "early_stopping": False, "n_iter_no_change": 5,
    "warm_start": False, "average": False, "n_jobs": 1, "verbose": 1,
    "l1_ratio": 0.15, "epsilon": 0.1, "eta0": 0.0, "power_t": 0.5,
    "validation_fraction": 0.1,
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeadValidationError(message)


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    _require(type(value) is int and value >= minimum,
             f"{name} must be an exact integer >= {minimum}")


def _float(value: Any, name: str) -> None:
    _require(type(value) is float and math.isfinite(value),
             f"{name} must be a finite float")


def _identifier(value: Any, name: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()),
             f"{name} must be a nonblank exact identifier")


def _records(records: tuple[AnnotationRecord, ...]) -> None:
    _require(type(records) is tuple and bool(records),
             "records must be an ordered nonempty tuple")
    try:
        validate_manifests(records, ())
    except ManifestValidationError as exc:
        raise HeadValidationError(f"annotation records: {exc}") from exc


def _features(array: np.ndarray, *, rows: int | None = None,
              contiguous: bool = True) -> None:
    _require(type(array) is np.ndarray, "features must be a base ndarray")
    _require(array.dtype == np.dtype(np.float32), "features must be native FP32")
    _require(array.ndim == 2 and array.shape[1] == _WIDTH,
             "features must have shape (N,1536)")
    _require(rows is None or array.shape[0] == rows,
             "feature/record count mismatch")
    _require(not contiguous or array.flags.c_contiguous,
             "features must be C-contiguous")
    _require(bool(np.isfinite(array).all()), "features must be finite")


def _coordinates(array: np.ndarray, rows: int) -> None:
    _require(type(array) is np.ndarray, "coordinates must be a base ndarray")
    _require(array.dtype.kind in ("i", "u"),
             "coordinates must have a signed/unsigned integer dtype")
    _require(array.ndim == 2 and array.shape == (rows, 2),
             "coordinates must have shape (raw_rows,2)")
    _require(bool((array >= 0).all()), "coordinates must be nonnegative")
    keys = {(int(x), int(y)) for x, y in array}
    _require(len(keys) == rows, "duplicate raw coordinates")


def _scatter(block: tuple[str, np.ndarray, np.ndarray],
             positions: dict[str, list[tuple[int, AnnotationRecord]]],
             seen: set[str], output: np.ndarray, placed: np.ndarray) -> None:
    _require(isinstance(block, tuple) and len(block) == 3,
             "each block must be a (slide_id,features,coordinates) tuple")
    slide, raw, coordinates = block
    _identifier(slide, "block slide_id")
    _require(slide in positions, f"unknown block slide {slide!r}")
    _require(slide not in seen, f"duplicate block slide {slide!r}")
    _features(raw, contiguous=False)
    _coordinates(coordinates, raw.shape[0])
    for position, record in positions[slide]:
        row = record.h_optimus_1_feature_row
        _require(row < raw.shape[0], f"{slide!r}: feature index out of bounds")
        _require((int(coordinates[row, 0]), int(coordinates[row, 1])) ==
                 (record.x, record.y),
                 f"{slide!r}: source index/physical coordinate mismatch")
        _require(not placed[position], "duplicate selected feature placement")
        output[position] = raw[row]
        placed[position] = True
    seen.add(slide)


def assemble_selected_features(
    records: tuple[AnnotationRecord, ...],
    blocks: Iterable[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter sequential raw blocks into a fresh FP32 matrix in record order."""
    _records(records)
    positions: dict[str, list[tuple[int, AnnotationRecord]]] = {}
    for position, record in enumerate(records):
        positions.setdefault(record.slide_id, []).append((position, record))
    output = np.empty((len(records), _WIDTH), dtype=np.float32, order="C")
    placed = np.zeros(len(records), dtype=bool)
    seen: set[str] = set()
    for block in blocks:
        _scatter(block, positions, seen, output, placed)
        del block
    _require(seen == set(positions) and bool(placed.all()),
             "missing blocks or incomplete selected feature placement")
    labels = np.array([record.tissue_label for record in records], dtype=np.int64)
    return output, labels


def _validate_manifest(manifest: TrainingManifest,
                       patient_to_fold: Mapping[str, int]) -> dict[int, float]:
    _require(type(manifest) is TrainingManifest, "expected TrainingManifest")
    _integer(manifest.held_out_fold, "held_out_fold")
    _require(manifest.held_out_fold < 5, "held_out_fold must be 0..4")
    _integer(manifest.budget_percent, "budget_percent")
    _require(manifest.budget_percent in (5, 100), "budget must be 5 or 100")
    _records(manifest.tiles)
    _require(isinstance(patient_to_fold, Mapping) and bool(patient_to_fold),
             "patient_to_fold must be a nonempty mapping")
    folds = dict(patient_to_fold)
    for patient, fold in folds.items():
        _identifier(patient, "patient_id")
        _integer(fold, "patient fold")
        _require(fold < 5, "patient fold must be 0..4")
    for record in manifest.tiles:
        _require(record.patient_id in folds, "selected patient is not mapped")
        _require(folds[record.patient_id] != manifest.held_out_fold,
                 "selected patient leaks from the held-out fold")
    ordered = tuple(sorted(manifest.tiles,
                           key=lambda row: (training_tile_hash(row), row.physical_key)))
    _require(manifest.tiles == ordered, "manifest does not preserve training hash order")
    _integer(manifest.pool_count, "pool_count", 1)
    _integer(manifest.selected_count, "selected_count", 1)
    _require(manifest.selected_count == len(manifest.tiles),
             "manifest selected total disagrees with records")
    _float(manifest.selected_fraction, "selected_fraction")
    _require(manifest.selected_fraction ==
             manifest.selected_count / manifest.pool_count,
             "manifest selected fraction disagrees with counts")
    _require(type(manifest.class_stats) is tuple and len(manifest.class_stats) == 5,
             "class_stats must contain all five ordered classes")
    weights = {}
    for label, stat in zip(_CLASSES, manifest.class_stats):
        _require(type(stat) is ClassSelection, "expected ClassSelection")
        _integer(stat.tissue_label, "class label", 1)
        _require(stat.tissue_label == label, "class order must be [1,2,3,4,5]")
        _integer(stat.pool_count, "class pool_count", 1)
        _integer(stat.selected_count, "class selected_count", 1)
        selected = sum(record.tissue_label == label for record in manifest.tiles)
        _require(selected == stat.selected_count, "selected class count mismatch")
        expected = (stat.pool_count if manifest.budget_percent == 100
                    else five_percent_count(stat.pool_count))
        _require(selected == expected and selected <= stat.pool_count,
                 "class rounding/nesting count mismatch")
        _float(stat.fraction, "class fraction")
        _float(stat.balanced_weight, "class balanced_weight")
        _require(stat.fraction == selected / stat.pool_count,
                 "class fraction mismatch")
        _require(stat.balanced_weight == manifest.selected_count / (5 * selected),
                 "selected-label class weight mismatch")
        weights[label] = stat.balanced_weight
    _require(sum(stat.pool_count for stat in manifest.class_stats) == manifest.pool_count,
             "manifest training pool total mismatch")
    _require(sum(stat.selected_count for stat in manifest.class_stats) ==
             manifest.selected_count, "manifest class selected total mismatch")
    return weights


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


def _parameters(weights: dict[int, float]) -> dict[str, Any]:
    return {**_PARAMETERS, "class_weight": dict(weights)}


def _state(estimator: SGDClassifier, weights: dict[int, float]) -> None:
    _require(type(estimator) is SGDClassifier, "expected exact SGDClassifier type")
    _require(_typed(estimator.get_params(deep=False)) == _typed(_parameters(weights)),
             "estimator full fixed parameters/class weights changed")
    for name in ("classes_", "coef_", "intercept_", "n_features_in_", "n_iter_", "t_"):
        _require(hasattr(estimator, name), f"missing fitted attribute {name}")
    classes = estimator.classes_
    _require(type(classes) is np.ndarray and classes.dtype.kind in ("i", "u") and
             classes.shape == (5,) and classes.tolist() == list(_CLASSES),
             "fitted class order must be [1,2,3,4,5]")
    for name, shape in (("coef_", (5, _WIDTH)), ("intercept_", (5,))):
        array = getattr(estimator, name)
        _require(type(array) is np.ndarray and array.dtype == np.dtype(np.float32)
                 and array.shape == shape and bool(np.isfinite(array).all()),
                 f"{name} must have finite FP32 shape {shape}")
    _integer(estimator.n_features_in_, "n_features_in_", 1)
    _require(estimator.n_features_in_ == _WIDTH, "n_features_in_ must be 1536")
    _integer(estimator.n_iter_, "n_iter_", 1)
    _require(estimator.n_iter_ <= _PARAMETERS["max_iter"], "n_iter_ exceeds max_iter")
    _float(estimator.t_, "t_")
    _require(estimator.t_ > 0.0, "t_ must be positive")


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


@dataclass(frozen=True, init=False)
class FittedHead:
    estimator: SGDClassifier
    _diagnostic_snapshot: bytes = field(repr=False)

    def __init__(self, estimator: SGDClassifier, diagnostics: dict[str, Any]):
        object.__setattr__(self, "estimator", estimator)
        object.__setattr__(self, "_diagnostic_snapshot", _json_bytes(diagnostics))

    @property
    def diagnostics(self) -> dict[str, Any]:
        """A detached finite-JSON report, never a mutable alias to fit evidence."""
        return json.loads(self._diagnostic_snapshot)


@dataclass(frozen=True)
class HeadPredictions:
    probabilities: np.ndarray
    labels: np.ndarray


def _runtime_profile() -> dict[str, Any]:
    versions = {name: importlib.metadata.version(name) for name in _VERSIONS}
    _require(versions == dict(_VERSIONS), f"unsupported fitting versions: {versions}")
    pools = threadpool_info()
    _require(all(type(pool["num_threads"]) is int and pool["num_threads"] == 1
                 for pool in pools), "fitting requires native pools of one")
    return {
        "distribution_versions": versions,
        "implementation": {
            "sklearn": {"version": sklearn.__version__, "path": sklearn.__file__},
            "numpy": {"version": np.__version__, "path": np.__file__},
            "scipy": {"version": scipy.__version__, "path": scipy.__file__},
            "SGDClassifier": {"path": inspect.getfile(SGDClassifier)},
        },
        "native_pools": pools,
    }


def _warning_records(caught: list[warnings.WarningMessage]) -> list[dict[str, Any]]:
    return [{
        "category": f"{item.category.__module__}.{item.category.__qualname__}",
        "message": str(item.message), "filename": item.filename, "lineno": item.lineno,
    } for item in caught]


def _warnings_report(caught: list[warnings.WarningMessage],
                     context: str, *, retain_convergence: bool) -> list[dict[str, Any]]:
    report = _warning_records(caught)
    unexpected = [item for item in caught
                  if not (retain_convergence and
                          issubclass(item.category, ConvergenceWarning))]
    if unexpected:
        error = HeadValidationError(
            f"{context}: unexpected warnings: " +
            "; ".join(f"{item.category.__name__}: {item.message}" for item in unexpected))
        error.add_note("All captured warnings: " + _json_bytes(report).decode("utf-8"))
        raise error
    return report


def _order_digest(records: tuple[AnnotationRecord, ...]) -> str:
    digest = hashlib.sha256(b"[")
    for position, record in enumerate(records):
        if position:
            digest.update(b",")
        digest.update(_json_bytes(record.tile_id))
    digest.update(b"]")
    return digest.hexdigest()


def fit_head(features: np.ndarray, manifest: TrainingManifest,
             patient_to_fold: Mapping[str, int]) -> FittedHead:
    """One fresh prescribed five-class fit; no retries or warning suppression."""
    weights = _validate_manifest(manifest, patient_to_fold)
    _features(features, rows=manifest.selected_count)
    runtime = _runtime_profile()
    labels = np.array([record.tissue_label for record in manifest.tiles], dtype=np.int64)
    input_digest = hashlib.sha256(memoryview(features).cast("B")).hexdigest()
    estimator = SGDClassifier(**_parameters(weights))
    context = f"fold={manifest.held_out_fold}, budget={manifest.budget_percent}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            estimator.fit(features, labels, sample_weight=None)
        except (ValueError, TypeError, FloatingPointError, OverflowError) as exc:
            exc.add_note(f"{context}: prescribed fit failed; no retry")
            exc.add_note("All captured fit warnings: " +
                         _json_bytes(_warning_records(caught)).decode("utf-8"))
            raise
    fit_warnings = _warnings_report(caught, context, retain_convergence=True)
    try:
        _state(estimator, weights)
    except HeadValidationError as exc:
        exc.add_note(f"{context}: invalid fitted state; no retry")
        exc.add_note("All captured fit warnings: " +
                     _json_bytes(fit_warnings).decode("utf-8"))
        raise
    diagnostics = {
        "schema_version": 1,
        "held_out_fold": manifest.held_out_fold,
        "budget_percent": manifest.budget_percent,
        "selected_tiles": len(manifest.tiles),
        "selected_slides": len({record.slide_id for record in manifest.tiles}),
        "selected_patients": len({record.patient_id for record in manifest.tiles}),
        "pool_count": manifest.pool_count,
        "selected_fraction": manifest.selected_fraction,
        "class_support": [{
            "tissue_label": stat.tissue_label, "pool_tiles": stat.pool_count,
            "selected_tiles": stat.selected_count,
            "selected_slides": len({row.slide_id for row in manifest.tiles
                                    if row.tissue_label == stat.tissue_label}),
            "selected_patients": len({row.patient_id for row in manifest.tiles
                                      if row.tissue_label == stat.tissue_label}),
            "selected_fraction": stat.fraction, "balanced_weight": stat.balanced_weight,
        } for stat in manifest.class_stats],
        "ordered_tile_ids_sha256": _order_digest(manifest.tiles),
        "feature_sha256": input_digest,
        "feature_shape": list(features.shape), "feature_dtype": features.dtype.str,
        "parameters": estimator.get_params(deep=False),
        "classes": estimator.classes_.tolist(),
        "finite_state": True,
        "coefficient_shape": list(estimator.coef_.shape),
        "coefficient_dtype": estimator.coef_.dtype.str,
        "intercept_shape": list(estimator.intercept_.shape),
        "intercept_dtype": estimator.intercept_.dtype.str,
        "n_features_in": estimator.n_features_in_,
        "n_iter": estimator.n_iter_, "t": estimator.t_,
        "n_iter_semantics": "maximum_across_five_one_vs_rest_fits",
        "warnings": fit_warnings,
        "iteration_limit_warning": bool(fit_warnings),
        "convergence_status": ("iteration_limit_warning_retained" if fit_warnings
                               else "no_warning_not_optimality_claim"),
        "runtime": runtime,
        "fitted_state_sha256": _fingerprint(estimator),
        "limits": [
            "selected_manifest_cannot_prove_unseen_pool_membership",
            "bare_matrix_cannot_prove_row_identity_or_file_provenance",
            "state_fingerprint_is_not_a_model_serializer_or_history_guard",
        ],
    }
    return FittedHead(estimator, diagnostics)


def _validate_head(head: FittedHead) -> None:
    _require(type(head) is FittedHead, "expected FittedHead")
    report = head.diagnostics
    _require(type(report) is dict and
             {"parameters", "fitted_state_sha256", "classes", "n_features_in",
              "n_iter", "t"}.issubset(report),
             "invalid fitted diagnostics snapshot")
    _require(type(report["parameters"]) is dict and
             "class_weight" in report["parameters"], "missing recorded parameters")
    recorded = report["parameters"]["class_weight"]
    _require(type(recorded) is dict and set(recorded) == {str(x) for x in _CLASSES},
             "invalid recorded class weights")
    weights = {label: recorded[str(label)] for label in _CLASSES}
    for weight in weights.values():
        _float(weight, "recorded class weight")
        _require(weight > 0.0, "recorded class weights must be positive")
    recorded_parameters = {**report["parameters"], "class_weight": weights}
    _require(_typed(recorded_parameters) == _typed(_parameters(weights)),
             "recorded full parameters disagree with the fixed recipe")
    _state(head.estimator, weights)
    for recorded_name, actual in (
        ("classes", head.estimator.classes_.tolist()),
        ("n_features_in", head.estimator.n_features_in_),
        ("n_iter", head.estimator.n_iter_), ("t", head.estimator.t_),
    ):
        _require(_typed(report[recorded_name]) == _typed(actual),
                 f"recorded {recorded_name} disagrees with fitted state")
    _require(_fingerprint(head.estimator) == report["fitted_state_sha256"],
             "fitted state fingerprint changed; fit diagnostics no longer apply")


def predict_head(fitted_head: FittedHead, features: np.ndarray) -> HeadPredictions:
    """Native five-class PROBA argmax, including validated-state empty batches."""
    _validate_head(fitted_head)
    _features(features)
    rows = features.shape[0]
    if not rows:
        return HeadPredictions(np.empty((0, 5), dtype=np.float32),
                               np.empty(0, dtype=np.int64))
    estimator = fitted_head.estimator
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = estimator.decision_function(features)
        _require(type(scores) is np.ndarray and scores.shape == (rows, 5)
                 and scores.dtype.kind == "f" and bool(np.isfinite(scores).all()),
                 "prediction decision scores must be finite with shape (N,5)")
        probabilities = estimator.predict_proba(features)
    _warnings_report(caught, "prediction", retain_convergence=False)
    _require(type(probabilities) is np.ndarray and
             probabilities.dtype == np.dtype(np.float32) and
             probabilities.shape == (rows, 5) and
             bool(np.isfinite(probabilities).all()),
             "predict_proba must return native finite FP32 shape (N,5)")
    _require(bool(((probabilities >= 0) & (probabilities <= 1)).all()),
             "probabilities must lie in [0,1]")
    sums = probabilities.sum(axis=1, dtype=np.float64)
    _require(bool((sums > 0).all()) and
             bool((np.abs(sums - 1.0) <= 8 * np.finfo(np.float32).eps).all()),
             "probability normalization outside declared FP32 roundoff bound")
    labels = estimator.classes_[probabilities.argmax(axis=1)].astype(np.int64)
    _validate_head(fitted_head)
    return HeadPredictions(probabilities, labels)


def _count_labels(array: np.ndarray, name: str) -> None:
    _require(type(array) is np.ndarray, f"{name} must be a base ndarray")
    _require(array.ndim == 1 and array.dtype.kind in ("i", "u"),
             f"{name} must be a one-dimensional integer label array")
    _require(bool(((array >= 1) & (array <= 5)).all()), f"{name} labels must be 1..5")


def tumour_counts(truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Pooled class-1-versus-rest counts; no fold average or inference."""
    _count_labels(truth, "truth")
    _count_labels(predictions, "predictions")
    _require(truth.shape == predictions.shape, "truth/prediction shape mismatch")
    positive, called = truth == 1, predictions == 1
    tp = int(np.count_nonzero(positive & called))
    fn = int(np.count_nonzero(positive & ~called))
    fp = int(np.count_nonzero(~positive & called))
    tn = int(np.count_nonzero(~positive & ~called))
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn, "total": int(truth.size),
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "sensitivity_status": "estimable" if tp + fn else "unavailable",
        "sensitivity_reason": None if tp + fn else "no_truth_tumour_tiles",
        "specificity": tn / (tn + fp) if tn + fp else None,
        "specificity_status": "estimable" if tn + fp else "unavailable",
        "specificity_reason": None if tn + fp else "no_truth_non_tumour_tiles",
    }
