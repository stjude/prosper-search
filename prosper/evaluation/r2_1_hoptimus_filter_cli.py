"""Bound CPU manifests only: no feature loading, fitting, masks or retrieval."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import csv
import gzip
import hashlib
import io
from itertools import zip_longest
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable, Iterator, Mapping, Sequence
import zlib

from . import r2_1_hoptimus_filter as kernel


TASK = "hoptimus_filter_manifest"
ARTIFACT_NAMES = frozenset({
    "retrieval_population.csv", "patient_folds.csv",
    "annotation_assignments.csv.gz", "training_rows.csv.gz",
    "head_plan.json", "summary.json",
})
SOURCE_NAMES = frozenset({
    "metadata", "stage0_review", "stage0_complete", "stage0_inventory",
    "stage0_summary", "kernel_review", "stage1a_review", "stage1a_complete",
    "stage1a_eligible", "stage1a_feature_inventory", "stage1a_summary",
})
EXPECTED_KEYS = frozenset({
    "raw_slides", "target_slides", "patients", "groups", "cross_label_patients",
    "annotation_tiles", "annotation_slides", "annotation_patients",
    "quarantine_tiles", "quarantine_slides",
})
METADATA_COLUMNS = {
    "id_col": "Slide.Scan.File",
    "patient_col": "UID.Subject",
    "label_col": "Tumor.group",
}
ELIGIBLE_HEADER = (
    "tile_id", "slide_id", "patient_id", "diagnostic_group", "level", "x", "y",
    "patch_size", "tissue_label", "uni2_legacy_row", "virchow2_legacy_row",
    "uni2_feature_row", "virchow2_feature_row", "h_optimus_1_feature_row",
)
FEATURE_HEADER = (
    "bank", "slide_id", "feature_dim", "projection_start", "projection_stop",
    "feature_rows", "dtype", "target_rows", "unused_unannotated",
    "unused_labelled_nontarget", "unused_outside_declared_sources",
    "label_csv_order_equal", "label_csv_position_mismatches", "feature_path",
    "resolved_path", "size", "sha256",
)
RETRIEVAL_HEADER = (
    "query_index", "reference_index", "metadata_index", "slide_id", "patient_id",
    "diagnostic_group", "fold",
)
FOLD_HEADER = (
    "patient_id", "fold", "kernel_order", "patient_hash",
    "has_annotations", "has_retrieval",
)
ANNOTATION_HEADER = (
    "tile_id", "slide_id", "patient_id", "diagnostic_group", "level", "x", "y",
    "patch_size", "tissue_label", "h_optimus_1_feature_row", "held_out_fold",
)
TRAINING_HEADER = (
    "held_out_fold", "budget_percent", "training_position", "tile_id",
    "slide_id", "patient_id", "tissue_label", "h_optimus_1_feature_row",
)
POLICY = {
    "fold_seed": 1234, "folds": 5, "budgets": [5, 100], "classes": [1, 2, 3, 4, 5],
    "fold_policy": "r2_1_patient_fold_v1", "training_policy": "r2_1_training_tile_v1",
}
QUARANTINE_SOURCE = "accepted_stage1a_review_and_summary_not_recounted"
LIMITATIONS = (
    "Quarantine counts are inherited from accepted Stage1a controls, not freshly recounted.",
    "Source indices are checked against accepted inventory counts; feature payloads are not read.",
    "No fitting, predictions, masks, retrieval or model-compatibility acceptance is performed.",
)


class FilterManifestError(ValueError):
    """Invalid bound source semantics or serialized manifest content."""


def _require(condition: bool, context: str) -> None:
    if not condition:
        raise FilterManifestError(context)


def _identity(value: object, context: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{context}: invalid exact ID")
    return value


def _integer(value: object, context: str) -> int:
    _require(type(value) is int and value >= 0, f"{context}: invalid nonnegative integer")
    return value


def _decimal(value: str, context: str) -> int:
    _require(bool(re.fullmatch(r"[0-9]+", value)), f"{context}: invalid decimal integer")
    return int(value)


def _pairs(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"JSON duplicate key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise FilterManifestError(f"nonfinite JSON constant: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    _require(math.isfinite(parsed), "nonfinite JSON number")
    return parsed


def _json(data: bytes, context: str) -> dict:
    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs,
                            parse_constant=_constant, parse_float=_finite_float)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FilterManifestError(f"{context}: malformed JSON") from exc
    _require(isinstance(result, dict), f"{context}: expected JSON object")
    return result


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def _csv_rows(
    data: bytes, header: Sequence[str], context: str, *, compressed: bool = False,
) -> Iterator[list[str]]:
    if compressed:
        _gzip_framing(data, context)
    raw = gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") if compressed else io.BytesIO(data)
    try:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
            reader = csv.reader(text, strict=True)
            _require(
                next(reader, None) == list(header), f"{context}: wrong/duplicate/unknown header"
            )
            for number, row in enumerate(reader, 2):
                _require(len(row) == len(header), f"{context} row {number}: wrong CSV width")
                yield row
    except (UnicodeError, csv.Error, OSError, EOFError) as exc:
        raise FilterManifestError(f"{context}: malformed CSV/gzip") from exc


def _gzip_framing(data: bytes, context: str) -> None:
    _require(len(data) >= 18 and data[:4] == b"\x1f\x8b\x08\x00"
             and data[4:8] == b"\x00" * 4, f"{context}: invalid gzip header/mtime")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        for offset in range(0, len(data), 16384):
            chunk = data[offset:offset + 16384]
            while chunk:
                decoder.decompress(chunk, 65536)
                _require(not decoder.unused_data, f"{context}: trailing gzip bytes/member")
                chunk = decoder.unconsumed_tail
            if decoder.eof:
                _require(offset + 16384 >= len(data), f"{context}: trailing gzip bytes/member")
                break
        _require(decoder.eof, f"{context}: truncated gzip")
    except zlib.error as exc:
        raise FilterManifestError(f"{context}: invalid gzip stream") from exc


def _csv_bytes(header: Sequence[str], rows: Iterable[Sequence], *, compressed: bool) -> bytes:
    buffer = io.BytesIO()
    raw = gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) if compressed else buffer
    text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    text.flush()
    text.detach()
    if compressed:
        raw.close()
    return buffer.getvalue()


@dataclass(frozen=True)
class MetadataRow:
    metadata_index: int
    slide_id: str
    patient_id: str
    diagnostic_group: str


def _metadata(data: bytes) -> tuple[MetadataRow, ...]:
    header = ("Tumor.group", "Slide.Scan.File", "Sample.ID", "UID.Subject")
    rows = []
    slides = set()
    for index, cells in enumerate(_csv_rows(data, header, "metadata")):
        group, slide, _, patient = cells
        for name, value in (("group", group), ("slide", slide), ("patient", patient)):
            _identity(value, f"metadata {name}")
        _require(slide not in slides, "metadata: duplicate slide ID")
        slides.add(slide)
        rows.append(MetadataRow(index, slide, patient, group))
    _require(bool(rows), "metadata: empty population")
    return tuple(rows)


def _inventory(data: bytes, metadata: Mapping[str, MetadataRow]) -> dict[str, dict]:
    hopt = {}
    seen = set()
    for cells in _csv_rows(data, FEATURE_HEADER, "stage1a_feature_inventory"):
        row = dict(zip(FEATURE_HEADER, cells))
        key = row["bank"], row["slide_id"]
        _require(row["bank"] in ("uni2", "virchow2", "h_optimus_1"), "inventory: unknown bank")
        _require(key not in seen, "inventory: duplicate bank/slide")
        seen.add(key)
        for name in FEATURE_HEADER[2:6] + FEATURE_HEADER[7:11] + (
            "label_csv_position_mismatches", "size",
        ):
            row[name] = _decimal(row[name], f"inventory {name}")
        _require(row["slide_id"] in metadata, "inventory: slide absent from metadata")
        _require(row["label_csv_order_equal"] in ("true", "false"),
                 "inventory: invalid order flag")
        _require(bool(re.fullmatch(r"[0-9a-f]{64}", row["sha256"])),
                 "inventory: invalid feature hash")
        for name in ("feature_path", "resolved_path", "dtype"):
            _identity(row[name], f"inventory {name}")
        _require(row["feature_rows"] == sum(row[name] for name in (
            "target_rows", "unused_unannotated", "unused_labelled_nontarget",
            "unused_outside_declared_sources",
        )), "inventory: unreconciled source rows")
        if row["bank"] == "h_optimus_1":
            _require((row["feature_dim"], row["projection_start"], row["projection_stop"])
                     == (1536, 0, 1536), "inventory: wrong H-Optimus width/projection")
            _require(row["dtype"] in ("float16", "float32"), "inventory: wrong H-Optimus dtype")
            hopt[row["slide_id"]] = row
    _require(bool(hopt), "inventory: no H-Optimus rows")
    return hopt


def _annotations(
    data: bytes, metadata: Mapping[str, MetadataRow], inventory: Mapping[str, dict],
) -> tuple[kernel.AnnotationRecord, ...]:
    rows = []
    for cells in _csv_rows(data, ELIGIBLE_HEADER, "stage1a_eligible"):
        row = dict(zip(ELIGIBLE_HEADER, cells))
        for name in ELIGIBLE_HEADER[4:]:
            row[name] = _decimal(row[name], f"eligible {name}")
        slide = row["slide_id"]
        _require(slide in metadata and slide in inventory, "eligible: unmapped/unknown slide")
        mapped = metadata[slide]
        _require((row["patient_id"], row["diagnostic_group"])
                 == (mapped.patient_id, mapped.diagnostic_group),
                 "eligible: patient/group differs from bound metadata")
        index = row["h_optimus_1_feature_row"]
        _require(index < inventory[slide]["feature_rows"],
                 "eligible: H-Optimus feature-row index outside accepted inventory")
        rows.append(kernel.AnnotationRecord(
            row["tile_id"], slide, row["patient_id"], row["diagnostic_group"],
            row["level"], row["x"], row["y"], row["patch_size"], row["tissue_label"], index,
        ))
    _require(bool(rows), "eligible: empty annotation population")
    return tuple(rows)


def _same(actual: object, expected: object, context: str) -> None:
    _require(type(actual) is type(expected), f"{context}: wrong value type")
    if isinstance(expected, dict):
        _require(set(actual) == set(expected), f"{context}: missing/extra keys")
        for key in expected:
            _same(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, list):
        _require(len(actual) == len(expected), f"{context}: wrong list length")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _same(left, right, f"{context}[{index}]")
    else:
        _require(actual == expected, f"{context}: mismatched value")


def _controls(
    sources: Mapping[str, bytes], annotations: Sequence[kernel.AnnotationRecord],
    inventory: Mapping[str, dict],
) -> tuple[int, int]:
    summary = _json(sources["stage1a_summary"], "stage1a_summary")
    review = _json(sources["stage1a_review"], "stage1a_review")
    _require(review.get("disposition") == "accepted"
             and review.get("scope") == "Stage1a historical-annotation target keyed binding only",
             "Stage1a review: unaccepted/wrong scope")
    eligible = summary["eligible"]
    observed = {
        "rows": len(annotations),
        "slides": len({row.slide_id for row in annotations}),
        "patients": len({row.patient_id for row in annotations}),
    }
    for name, count in observed.items():
        _same(eligible[name], count, f"Stage1a eligible {name}")
        _same(review["reconciled"][f"eligible_{name}"], count, f"Stage1a review eligible {name}")
    for field, identity in (("class_rows", None), ("class_slides", "slide_id"),
                            ("class_patients", "patient_id")):
        counts = {
            str(label): (
                sum(row.tissue_label == label for row in annotations) if identity is None else
                len({getattr(row, identity) for row in annotations if row.tissue_label == label})
            ) for label in range(1, 6)
        }
        _same(eligible[field], counts, f"Stage1a eligible {field}")
    slide_counts = Counter(row.slide_id for row in annotations)
    _require(set(slide_counts) == set(inventory), "inventory: H-Optimus slide population mismatch")
    for slide, count in slide_counts.items():
        _same(inventory[slide]["target_rows"], count, "inventory target rows")
    totals = summary["feature_inventory_totals"]["h_optimus_1"]
    for name in ("feature_rows", "target_rows", "unused_unannotated",
                 "unused_labelled_nontarget", "unused_outside_declared_sources"):
        _same(totals[name], sum(row[name] for row in inventory.values()),
              f"Stage1a H-Optimus total {name}")
    banks = [row for row in summary["declarations"]["banks"] if row["name"] == "h_optimus_1"]
    _require(len(banks) == 1, "Stage1a declaration: missing/duplicate H-Optimus bank")
    _same(banks[0]["feature_dim"], 1536, "Stage1a declared H-Optimus width")
    _same(banks[0]["projection"], [0, 1536], "Stage1a declared H-Optimus projection")
    quarantine = tuple(
        _integer(summary["quarantine"][name], f"inherited quarantine {name}")
        for name in ("rows", "slides")
    )
    for name, count in zip(("rows", "slides"), quarantine):
        _same(review["reconciled"][f"quarantine_{name}"], count, f"review quarantine {name}")
    return quarantine


def _ordered_ids_sha(tiles: Iterable[kernel.AnnotationRecord]) -> str:
    digest = hashlib.sha256(b"[")
    for index, tile in enumerate(tiles):
        if index:
            digest.update(b",")
        digest.update(json.dumps(tile.tile_id, ensure_ascii=True).encode("ascii"))
    digest.update(b"]")
    return digest.hexdigest()


def _head(manifest: kernel.TrainingManifest) -> dict:
    return {
        "held_out_fold": manifest.held_out_fold, "budget_percent": manifest.budget_percent,
        "pool_count": manifest.pool_count, "selected_count": manifest.selected_count,
        "selected_fraction": manifest.selected_fraction,
        "training_tile_ids_sha256": _ordered_ids_sha(manifest.tiles),
        "class_stats": [asdict(stat) for stat in manifest.class_stats],
    }


def _support(support: kernel.FoldSupport) -> dict:
    value = asdict(support)
    value["class_counts"] = list(value["class_counts"])
    return value


def _retrieval_rows(
    retrieval: Sequence[MetadataRow], mapping: Mapping[str, int],
) -> Iterator[tuple]:
    for index, row in enumerate(retrieval):
        yield (
            index, index, row.metadata_index, row.slide_id, row.patient_id,
            row.diagnostic_group, mapping[row.patient_id],
        )


def _fold_rows(
    folds: kernel.FoldManifest, annotations: Sequence[kernel.AnnotationRecord],
    retrieval: Sequence[MetadataRow],
) -> Iterator[tuple]:
    has_annotations = {row.patient_id for row in annotations}
    has_retrieval = {row.patient_id for row in retrieval}
    indexed = sorted(enumerate(folds.patients), key=lambda item: item[1].patient_id)
    for index, row in indexed:
        yield (
            row.patient_id, row.fold, index, kernel.patient_fold_hash(row.patient_id),
            int(row.patient_id in has_annotations), int(row.patient_id in has_retrieval),
        )


def _annotation_rows(
    annotations: Sequence[kernel.AnnotationRecord], mapping: Mapping[str, int],
) -> Iterator[tuple]:
    for row in annotations:
        yield (
            row.tile_id, row.slide_id, row.patient_id, row.diagnostic_group,
            row.level, row.x, row.y, row.patch_size, row.tissue_label,
            row.h_optimus_1_feature_row, mapping[row.patient_id],
        )


def _training_rows(manifests: Sequence[kernel.TrainingManifest]) -> Iterator[tuple]:
    for manifest in manifests:
        for position, row in enumerate(manifest.tiles):
            yield (
                manifest.held_out_fold, manifest.budget_percent, position, row.tile_id,
                row.slide_id, row.patient_id, row.tissue_label, row.h_optimus_1_feature_row,
            )


def _check_rows(
    data: bytes, header: Sequence[str], expected: Iterable[Sequence], context: str,
    *, compressed: bool = False,
) -> Iterator[list[str]]:
    for number, (actual, wanted) in enumerate(zip_longest(
        _csv_rows(data, header, context, compressed=compressed), expected,
    ), 1):
        _require(actual is not None and wanted is not None,
                 f"{context}: extra/missing serialized row {number}")
        _require(actual == [str(value) for value in wanted],
                 f"{context}: mismatched serialized row {number}")
        yield actual


def validate_artifacts(
    artifacts: Mapping[str, bytes], spec: dict,
    annotations: Sequence[kernel.AnnotationRecord], retrieval: Sequence[MetadataRow],
    folds: kernel.FoldManifest, manifests: Sequence[kernel.TrainingManifest],
    summary: dict,
) -> None:
    """Stream semantic readback without reconstructing annotation/training objects."""
    _require(set(artifacts) == ARTIFACT_NAMES, "artifacts: incorrect branch artifact set")
    mapping = {row.patient_id: row.fold for row in folds.patients}
    for _ in _check_rows(artifacts["retrieval_population.csv"], RETRIEVAL_HEADER,
                         _retrieval_rows(retrieval, mapping), "retrieval_population"):
        pass
    rebuilt = {}
    for cells in _check_rows(artifacts["patient_folds.csv"], FOLD_HEADER,
                             _fold_rows(folds, annotations, retrieval), "patient_folds"):
        patient, fold, order, _, _, _ = cells
        index = _decimal(order, "kernel_order")
        _require(index not in rebuilt, "patient_folds: duplicate kernel_order")
        rebuilt[index] = kernel.PatientFold(patient, _decimal(fold, "fold"))
    _require(tuple(rebuilt[index] for index in range(len(rebuilt))) == folds.patients,
             "patient_folds: failed hash-order reconstruction")
    partitions = Counter()
    for cells in _check_rows(artifacts["annotation_assignments.csv.gz"], ANNOTATION_HEADER,
                             _annotation_rows(annotations, mapping), "annotation_assignments",
                             compressed=True):
        partitions[_decimal(cells[-1], "held_out_fold")] += 1
    for support in folds.support:
        _require(partitions[support.fold] == support.annotation_tile_count,
                 "annotation_assignments: held-out partition mismatch")
    by_id = {row.tile_id: row for row in annotations}
    selected = defaultdict(Counter)
    digests = {}
    row_counts = Counter()
    for cells in _check_rows(artifacts["training_rows.csv.gz"], TRAINING_HEADER,
                             _training_rows(manifests), "training_rows", compressed=True):
        fold, budget, position = (_decimal(cells[index], "training position") for index in range(3))
        key = fold, budget
        _require(position == row_counts[key], "training_rows: noncontiguous fitting position")
        row_counts[key] += 1
        tile = by_id[cells[3]]
        _require(mapping[tile.patient_id] != fold, "training_rows: held-out patient leakage")
        _require(_decimal(cells[7], "source index") == tile.h_optimus_1_feature_row,
                 "training_rows: wrong dedicated source index")
        selected[key][tile.tissue_label] += 1
        if key not in digests:
            digests[key] = hashlib.sha256(b"[")
        else:
            digests[key].update(b",")
        digests[key].update(json.dumps(tile.tile_id, ensure_ascii=True).encode("ascii"))
    head_plan = _json(artifacts["head_plan.json"], "head_plan")
    expected_heads = [_head(manifest) for manifest in manifests]
    _same(head_plan, {
        "schema_version": 1, "run_id": spec["run_id"], "task": TASK,
        "classes": [1, 2, 3, 4, 5], "heads": expected_heads,
    }, "head_plan")
    for head in head_plan["heads"]:
        key = head["held_out_fold"], head["budget_percent"]
        digests[key].update(b"]")
        _same(digests[key].hexdigest(), head["training_tile_ids_sha256"], "head manifest digest")
        _same(row_counts[key], head["selected_count"], "head selected count")
        for stat in head["class_stats"]:
            count = selected[key][stat["tissue_label"]]
            _same(count, stat["selected_count"], "serialized selected-class count")
            _same(row_counts[key] / (5 * count), stat["balanced_weight"],
                  "serialized selected-training weight")
    _same(_json(artifacts["summary.json"], "summary"), summary, "summary")


def execute(spec: dict, sources: Mapping[str, bytes]) -> dict[str, bytes]:
    """Consume only bound source bytes and return semantically verified artifacts."""
    try:
        _require(spec["task"] == TASK, "wrong filter task")
        _same(spec["parameters"], {"metadata": METADATA_COLUMNS}, "parameters")
        _require(set(sources) == SOURCE_NAMES and set(spec["sources"]) == SOURCE_NAMES,
                 "filter: wrong source roles")
        _require(set(spec["expected"]) == EXPECTED_KEYS, "filter: wrong expected-count keys")
        for name, count in spec["expected"].items():
            _integer(count, f"expected {name}")
        metadata = _metadata(sources["metadata"])
        lookup = {row.slide_id: row for row in metadata}
        retrieval = tuple(row for row in metadata if row.diagnostic_group not in ("MISC", "NORMAL"))
        inventory = _inventory(sources["stage1a_feature_inventory"], lookup)
        annotations = _annotations(sources["stage1a_eligible"], lookup, inventory)
        records = tuple(kernel.RetrievalRecord(
            row.slide_id, row.patient_id, row.diagnostic_group,
        ) for row in retrieval)
        annotations, records = kernel.validate_manifests(annotations, records)
        quarantine_tiles, quarantine_slides = _controls(sources, annotations, inventory)
        patient_groups = defaultdict(set)
        for row in retrieval:
            patient_groups[row.patient_id].add(row.diagnostic_group)
        counts = {
            "raw_slides": len(metadata), "target_slides": len(retrieval),
            "patients": len(patient_groups),
            "groups": len({row.diagnostic_group for row in retrieval}),
            "cross_label_patients": sum(len(groups) > 1 for groups in patient_groups.values()),
            "annotation_tiles": len(annotations),
            "annotation_slides": len({row.slide_id for row in annotations}),
            "annotation_patients": len({row.patient_id for row in annotations}),
            "quarantine_tiles": quarantine_tiles, "quarantine_slides": quarantine_slides,
        }
        _same(counts, spec["expected"], "observed counts")
        _require(all(counts[name] > 0 for name in (
            "target_slides", "patients", "groups", "annotation_tiles",
            "annotation_slides", "annotation_patients",
        )), "filter: required population is empty")
        folds = kernel.build_patient_folds(annotations, records)
        manifests = kernel.build_training_manifests(annotations, records, folds)
        mapping = {row.patient_id: row.fold for row in folds.patients}
        summary = {
            "schema_version": 1, "run_id": spec["run_id"], "task": TASK,
            "counts": counts, "policy": POLICY,
            "details": {
                "fold_support": [_support(support) for support in folds.support],
                "quarantine_count_source": QUARANTINE_SOURCE,
            },
            "limitations": list(LIMITATIONS),
        }
        artifacts = {
            "retrieval_population.csv": _csv_bytes(
                RETRIEVAL_HEADER, _retrieval_rows(retrieval, mapping), compressed=False,
            ),
            "patient_folds.csv": _csv_bytes(
                FOLD_HEADER, _fold_rows(folds, annotations, retrieval), compressed=False,
            ),
            "annotation_assignments.csv.gz": _csv_bytes(
                ANNOTATION_HEADER, _annotation_rows(annotations, mapping), compressed=True,
            ),
            "training_rows.csv.gz": _csv_bytes(
                TRAINING_HEADER, _training_rows(manifests), compressed=True,
            ),
            "head_plan.json": _json_bytes({
                "schema_version": 1, "run_id": spec["run_id"], "task": TASK,
                "classes": [1, 2, 3, 4, 5], "heads": [_head(manifest) for manifest in manifests],
            }),
            "summary.json": _json_bytes(summary),
        }
        validate_artifacts(artifacts, spec, annotations, retrieval, folds, manifests, summary)
        return artifacts
    except (KeyError, TypeError, kernel.ManifestValidationError) as exc:
        raise FilterManifestError(f"invalid filter source/declaration: {exc}") from exc


def run(spec_path: Path, spec_sha256: str, out: Path, out_root: Path):
    from . import feature_bank_audit as legacy_io
    from . import r2_1_hoptimus_io as shared_io
    import prosper
    from prosper import evaluation

    return shared_io.run_bound(
        spec_path, spec_sha256, out, out_root, task=TASK,
        code_paths={
            "runner": Path(__file__), "kernel": Path(kernel.__file__),
            "io": Path(shared_io.__file__), "legacy_io": Path(legacy_io.__file__),
            "package_init": Path(prosper.__file__), "evaluation_init": Path(evaluation.__file__),
        },
        execute=execute,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(args.spec, args.spec_sha256, args.out, args.out_root)
    except Exception as exc:
        print(f"FILTER MANIFEST FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
