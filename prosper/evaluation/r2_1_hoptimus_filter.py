"""Pure patient-fold and nested training-label manifests for R2.1.

No feature payloads, fitted heads, masks, retrieval or statistical inference
are handled here. The two-Immune-patient fold constraint is prespecified from
annotation composition; selection and weights subsequently use training only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Tuple


class ManifestValidationError(ValueError):
    """An identity, population or training-support requirement is violated."""


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or not value.strip():
        raise ManifestValidationError(f"{name} must be a nonempty exact string")


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ManifestValidationError(f"{name} must be an integer >= {minimum}")


def _json_sha(values: list) -> str:
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def physical_tile_id(slide_id: str, x: int, y: int) -> str:
    """Frozen Stage1a identity for a level-0, 512-pixel tile."""
    _identifier(slide_id, "slide_id")
    _integer(x, "x")
    _integer(y, "y")
    return _json_sha([slide_id, 0, x, y, 512])


@dataclass(frozen=True)
class AnnotationRecord:
    tile_id: str
    slide_id: str
    patient_id: str
    diagnostic_group: str
    level: int
    x: int
    y: int
    patch_size: int
    tissue_label: int
    h_optimus_1_feature_row: int

    def __post_init__(self) -> None:
        _validate_annotation(self)

    @property
    def physical_key(self) -> Tuple[str, int, int, int, int]:
        return self.slide_id, self.level, self.x, self.y, self.patch_size


def _validate_annotation(record: AnnotationRecord) -> None:
    for name in ("tile_id", "slide_id", "patient_id", "diagnostic_group"):
        _identifier(getattr(record, name), name)
    for name in ("level", "x", "y", "patch_size", "tissue_label", "h_optimus_1_feature_row"):
        _integer(getattr(record, name), name)
    if record.level != 0 or record.patch_size != 512:
        raise ManifestValidationError("annotation geometry must be level 0 / patch size 512")
    if record.tissue_label not in (1, 2, 3, 4, 5):
        raise ManifestValidationError("tissue_label must be 1..5, not unannotated/quarantined")
    if record.tile_id != physical_tile_id(record.slide_id, record.x, record.y):
        raise ManifestValidationError("tile_id disagrees with its physical key")


@dataclass(frozen=True)
class RetrievalRecord:
    slide_id: str
    patient_id: str
    diagnostic_group: str

    def __post_init__(self) -> None:
        _validate_retrieval(self)


def _validate_retrieval(record: RetrievalRecord) -> None:
    for name in ("slide_id", "patient_id", "diagnostic_group"):
        _identifier(getattr(record, name), name)
    if record.diagnostic_group in ("MISC", "NORMAL"):
        raise ManifestValidationError("retrieval excludes MISC and NORMAL")


def validate_manifests(
    annotations: Iterable[AnnotationRecord],
    retrieval: Iterable[RetrievalRecord],
) -> Tuple[Tuple[AnnotationRecord, ...], Tuple[RetrievalRecord, ...]]:
    """Validate and return deterministic immutable copies of both populations."""
    try:
        annotation_rows = tuple(annotations)
        retrieval_rows = tuple(retrieval)
    except TypeError as exc:
        raise ManifestValidationError(
            "annotation and retrieval populations must be iterable"
        ) from exc
    metadata = {}
    physical_keys = set()
    tile_ids = set()
    source_indices = set()

    def metadata_for(slide: str, patient: str, group: str) -> None:
        pair = patient, group
        if slide in metadata and metadata[slide] != pair:
            raise ManifestValidationError(
                f"inconsistent patient/group metadata for slide {slide!r}"
            )
        metadata[slide] = pair

    for record in annotation_rows:
        if not isinstance(record, AnnotationRecord):
            raise ManifestValidationError("annotations must contain AnnotationRecord values")
        _validate_annotation(record)
        metadata_for(record.slide_id, record.patient_id, record.diagnostic_group)
        if record.physical_key in physical_keys or record.tile_id in tile_ids:
            raise ManifestValidationError("duplicate annotation physical key/tile ID")
        source = record.slide_id, record.h_optimus_1_feature_row
        if source in source_indices:
            raise ManifestValidationError("duplicate per-slide H-Optimus source index")
        physical_keys.add(record.physical_key)
        tile_ids.add(record.tile_id)
        source_indices.add(source)

    retrieval_slides = set()
    for record in retrieval_rows:
        if not isinstance(record, RetrievalRecord):
            raise ManifestValidationError("retrieval must contain RetrievalRecord values")
        _validate_retrieval(record)
        if record.slide_id in retrieval_slides:
            raise ManifestValidationError("duplicate retrieval slide ID")
        retrieval_slides.add(record.slide_id)
        metadata_for(record.slide_id, record.patient_id, record.diagnostic_group)
    return (
        tuple(sorted(annotation_rows, key=lambda row: row.physical_key)),
        tuple(sorted(retrieval_rows, key=lambda row: row.slide_id)),
    )


def patient_fold_hash(patient_id: str) -> str:
    _identifier(patient_id, "patient_id")
    return _json_sha(["r2_1_patient_fold_v1", 1234, patient_id])


def training_tile_hash(record: AnnotationRecord) -> str:
    if not isinstance(record, AnnotationRecord):
        raise ManifestValidationError("training hash requires an AnnotationRecord")
    _validate_annotation(record)
    return _json_sha([
        "r2_1_training_tile_v1", 1234, record.patient_id, record.slide_id,
        0, record.x, record.y, 512, record.tissue_label,
    ])


@dataclass(frozen=True)
class PatientFold:
    patient_id: str
    fold: int

    def __post_init__(self) -> None:
        _identifier(self.patient_id, "patient_id")
        _integer(self.fold, "fold")
        if self.fold >= 5:
            raise ManifestValidationError("fold must be 0..4")


@dataclass(frozen=True)
class ClassCount:
    tissue_label: int
    count: int


@dataclass(frozen=True)
class FoldSupport:
    fold: int
    patient_count: int
    annotation_patient_count: int
    retrieval_patient_count: int
    annotation_slide_count: int
    retrieval_slide_count: int
    annotation_tile_count: int
    class_counts: Tuple[ClassCount, ...]


@dataclass(frozen=True)
class FoldManifest:
    patients: Tuple[PatientFold, ...]
    support: Tuple[FoldSupport, ...]


def build_patient_folds(
    annotations: Iterable[AnnotationRecord],
    retrieval: Iterable[RetrievalRecord],
) -> FoldManifest:
    """Assign the annotation/retrieval patient union under the fixed ADR recipe."""
    annotation_rows, retrieval_rows = validate_manifests(annotations, retrieval)
    patients = {row.patient_id for row in annotation_rows + retrieval_rows}
    ordered = sorted(patients, key=lambda patient: (patient_fold_hash(patient), patient))
    immune = {row.patient_id for row in annotation_rows if row.tissue_label == 2}
    if len(immune) != 2:
        raise ManifestValidationError(
            "exactly two distinct Immune-supporting patients are required"
        )
    immune_order = [patient for patient in ordered if patient in immune]
    assignment = {patient: fold for fold, patient in enumerate(immune_order)}
    sizes = [1, 1, 0, 0, 0]
    for patient in ordered:
        if patient not in immune:
            fold = min(range(5), key=lambda index: (sizes[index], index))
            assignment[patient] = fold
            sizes[fold] += 1

    support = []
    for fold in range(5):
        ann = tuple(row for row in annotation_rows if assignment[row.patient_id] == fold)
        ret = tuple(row for row in retrieval_rows if assignment[row.patient_id] == fold)
        support.append(FoldSupport(
            fold=fold,
            patient_count=sizes[fold],
            annotation_patient_count=len({row.patient_id for row in ann}),
            retrieval_patient_count=len({row.patient_id for row in ret}),
            annotation_slide_count=len({row.slide_id for row in ann}),
            retrieval_slide_count=len(ret),
            annotation_tile_count=len(ann),
            class_counts=tuple(
                ClassCount(label, sum(row.tissue_label == label for row in ann))
                for label in range(1, 6)
            ),
        ))
    return FoldManifest(
        patients=tuple(PatientFold(patient, assignment[patient]) for patient in ordered),
        support=tuple(support),
    )


def five_percent_count(class_count: int) -> int:
    """Round n/20 to nearest, ties to even, with the agreed minimum of one."""
    _integer(class_count, "class_count", minimum=1)
    quotient, remainder = divmod(class_count, 20)
    rounded = quotient + int(remainder > 10 or (remainder == 10 and quotient % 2 == 1))
    return max(1, rounded)


@dataclass(frozen=True)
class ClassSelection:
    tissue_label: int
    pool_count: int
    selected_count: int
    fraction: float
    balanced_weight: float


@dataclass(frozen=True)
class TrainingManifest:
    held_out_fold: int
    budget_percent: int
    tiles: Tuple[AnnotationRecord, ...]
    class_stats: Tuple[ClassSelection, ...]
    pool_count: int
    selected_count: int
    selected_fraction: float


def build_training_manifests(
    annotations: Iterable[AnnotationRecord],
    retrieval: Iterable[RetrievalRecord],
    folds: FoldManifest,
) -> Tuple[TrainingManifest, ...]:
    """Return five-fold x two-budget manifests, rejecting altered common folds."""
    annotation_rows, retrieval_rows = validate_manifests(annotations, retrieval)
    expected = build_patient_folds(annotation_rows, retrieval_rows)
    if not isinstance(folds, FoldManifest) or folds != expected:
        raise ManifestValidationError(
            "fold manifest does not match the prescribed common patient map"
        )
    assignment = {row.patient_id: row.fold for row in folds.patients}
    ordered = tuple(sorted(
        annotation_rows, key=lambda row: (training_tile_hash(row), row.physical_key)
    ))
    manifests = []
    for held_out in range(5):
        pool = tuple(row for row in ordered if assignment[row.patient_id] != held_out)
        by_class = {
            label: tuple(row for row in pool if row.tissue_label == label)
            for label in range(1, 6)
        }
        absent = [label for label, rows in by_class.items() if not rows]
        if absent:
            raise ManifestValidationError(
                f"held-out fold {held_out}: training pool lacks tissue classes {absent}"
            )
        selected_ids = {
            row.tile_id
            for rows in by_class.values()
            for row in rows[:five_percent_count(len(rows))]
        }
        primary = tuple(row for row in pool if row.tile_id in selected_ids)
        for budget, selected in ((5, primary), (100, pool)):
            selected_counts = {
                label: sum(row.tissue_label == label for row in selected)
                for label in range(1, 6)
            }
            stats = tuple(
                ClassSelection(
                    tissue_label=label,
                    pool_count=len(by_class[label]),
                    selected_count=selected_counts[label],
                    fraction=selected_counts[label] / len(by_class[label]),
                    balanced_weight=len(selected) / (5 * selected_counts[label]),
                )
                for label in range(1, 6)
            )
            manifests.append(TrainingManifest(
                held_out_fold=held_out,
                budget_percent=budget,
                tiles=selected,
                class_stats=stats,
                pool_count=len(pool),
                selected_count=len(selected),
                selected_fraction=len(selected) / len(pool),
            ))
    return tuple(manifests)
