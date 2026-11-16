"""Accepted mosaic inputs for the direct-source filtering pilot.

``load_mosaic_bundle`` authenticates an explicitly trusted review/COMPLETE
chain and only the declared controls, coordinates and selected masks.
``load_raw_slide`` reads a declared slide on demand and uses the frozen
mosaic decoder without selecting, casting or canonicalizing its raw array.

The caller constructs matching bytes-access AssetGuard policies, owns its
active context and mandatory final_bindings sweep, and records runtime/output
custody. This reader uses only read_verified: no leases, private guard state,
head loading, probability reads, normalization or whole-library raw cache.
Bundle arrays have immutable byte backing. Raw arrays are the exact decoder
return and only cooperatively read-only. Native imports are intentional;
this is not the stdlib perturbation admission boundary.
Referenced ancestry and unselected arrays are not current-file checks.
Feature pickle decoding is trusted-local ingestion, not safe untrusted input.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, fields
import hashlib
import io
from pathlib import Path

import numpy as np

from . import r2_1_hoptimus_asset_io as assets
from . import r2_1_hoptimus_io as cpu
from . import r2_1_hoptimus_mosaic_cli as mosaic
from . import r2_1_hoptimus_head_cli as producer


MIB = 1024 * 1024
POPULATION_HEADER = (
    "query_index", "reference_index", "metadata_index", "slide_id", "patient_id",
    "diagnostic_group", "fold",
)
OFFSET_HEADER = mosaic.OFFSET_HEADER
COUNT_HEADER = mosaic.COUNT_HEADER
VIEW_ORDER = mosaic.TUPLES
OFFSET_EXPANSION_LIMIT = 22011136
COUNT_EXPANSION_LIMIT = 219998464
REVIEW_STAGE = "actual_hoptimus_mosaic_outputs"
REVIEW_DISPOSITION = "accepted_for_preservation_and_later_separately_specified_filtering_inputs"
_CONTROLS = (
    ("review", "mosaic:review105", "mosaic_review", 2 * MIB, "actual_outputs_independent_review.json"),
    ("complete", "mosaic:complete", "mosaic_complete", 2 * MIB, "COMPLETE.json"),
    ("spec", "mosaic:spec", "mosaic_spec", 8 * MIB, "mosaic_spec.json"),
    ("summary", "mosaic:summary", "mosaic_summary", 16 * MIB, "summary.json"),
    ("input_bindings", "mosaic:input_bindings", "mosaic_input_bindings", 16 * MIB, "input_bindings.json"),
    ("population", "mosaic:population", "mosaic_population", 16 * MIB, "retrieval_population.csv"),
    ("offsets", "mosaic:offsets", "mosaic_offsets", 16 * MIB, "slide_offsets.csv"),
    ("mask_counts", "mosaic:mask_counts", "mosaic_mask_counts", 16 * MIB, "mask_counts.csv.gz"),
)
_POLICY = {
    "budgets": [5, 100], "classes": [1, 2, 3, 4, 5],
    "coordinates": "accepted_stage0_order_reuse_unchanged_feature_bytes",
    "decision": "probability_argmax_class1", "folds": [0, 1, 2, 3, 4],
    "no_new_canonical_comparison": True, "population_order": "accepted_cpu_filter_order",
    "prediction_batch_rows": 4096, "scope": mosaic.SCOPE,
}


class MosaicReaderError(ValueError):
    """Invalid accepted-bundle declaration, data or source join."""


def _require(condition, message):
    if not condition:
        raise MosaicReaderError(message)


def _integer(value, context, minimum=0):
    _require(type(value) is int and value >= minimum, context + ": strict integer required")
    return value


def _text(value, context):
    _require(type(value) is str and bool(value.strip()) and "\0" not in value,
             context + ": nonempty text required")
    return value


def _digest(value):
    producer._digest(value, "SHA256")
    return value


def _tuple(value, cls, context):
    _require(type(value) is tuple and all(type(item) is cls for item in value),
             context + ": frozen record tuple required")


def _json(data, context):
    value = cpu.strict_json(data)
    _require(type(value) is dict, context + ": object required")
    return value


def _keys(value, names, context):
    _require(type(value) is dict and set(value) == set(names), context + ": exact keys required")


def _same(a, b, context):
    _require(cpu.json_bytes(a) == cpu.json_bytes(b), context + ": mismatch")


def _descriptor(value, context, limit=None):
    _keys(value, ("path", "size", "sha256"), context)
    _text(value["path"], context)
    _require(Path(value["path"]).is_absolute(), context + ": explicit absolute path required")
    _integer(value["size"], context)
    _digest(value["sha256"])
    if limit is not None:
        _require(value["size"] <= limit, context + ": encoded ceiling")


def _asset(value, identity, role, limit):
    _require(type(value) is assets.AssetSpec, identity + ": AssetSpec required")
    _require(value.asset_id == identity and value.role == role, identity + ": ID/role mismatch")
    _require(Path(value.path).is_absolute(), identity + ": explicit current absolute path required")
    _integer(value.size, identity)
    _digest(value.sha256)
    _require(value.size <= limit, identity + ": encoded ceiling")


@dataclass(frozen=True, slots=True)
class HeadView:
    held_out_fold: int
    budget_percent: int

    def __post_init__(self):
        _integer(self.held_out_fold, "fold")
        _integer(self.budget_percent, "budget")
        _require(self.held_out_fold < 5 and self.budget_percent in (5, 100), "invalid HeadView")


@dataclass(frozen=True, slots=True)
class MosaicCounts:
    slides: int
    patients: int
    diagnostic_groups: int
    union_patients: int
    feature_files: int
    original_tiles: int
    heads: int
    slide_head_cells: int
    tile_head_predictions: int

    def __post_init__(self):
        for field in fields(self):
            _integer(getattr(self, field.name), field.name, 1)
        _require(self.heads == 10 and self.feature_files == self.slides
                 and self.original_tiles >= self.slides
                 and self.slide_head_cells == 10 * self.slides
                 and self.tile_head_predictions == 10 * self.original_tiles
                 and self.patients <= self.slides and self.diagnostic_groups <= self.slides
                 and self.union_patients >= self.patients, "incoherent MosaicCounts")


@dataclass(frozen=True, slots=True)
class MosaicExpected:
    review_sha256: str
    complete_sha256: str
    spec_sha256: str
    run_id: str
    counts: MosaicCounts

    def __post_init__(self):
        for value in (self.review_sha256, self.complete_sha256, self.spec_sha256):
            _digest(value)
        _text(self.run_id, "run_id")
        _require(type(self.counts) is MosaicCounts, "MosaicCounts required")


@dataclass(frozen=True, slots=True)
class MaskAsset:
    view: HeadView
    asset: assets.AssetSpec

    def __post_init__(self):
        _require(type(self.view) is HeadView, "MaskAsset HeadView required")
        _asset(self.asset, f"mosaic:mask:fold{self.view.held_out_fold}:budget{self.view.budget_percent}",
               "mosaic_mask", 8 * MIB)


@dataclass(frozen=True, slots=True)
class FeatureAsset:
    slide_index: int
    asset: assets.AssetSpec

    def __post_init__(self):
        _integer(self.slide_index, "feature slide_index")
        _require(type(self.asset) is assets.AssetSpec, "feature AssetSpec required")
        _require(self.asset.asset_id.startswith("feature:") and bool(self.asset.asset_id[8:]),
                 "feature AssetID namespace")
        _asset(self.asset, self.asset.asset_id, "mosaic_feature", 64 * MIB)


@dataclass(frozen=True, slots=True)
class MosaicBundleAssets:
    review: assets.AssetSpec
    complete: assets.AssetSpec
    spec: assets.AssetSpec
    summary: assets.AssetSpec
    input_bindings: assets.AssetSpec
    population: assets.AssetSpec
    offsets: assets.AssetSpec
    mask_counts: assets.AssetSpec
    coordinates: assets.AssetSpec
    masks: tuple[MaskAsset, ...]
    features: tuple[FeatureAsset, ...]

    def __post_init__(self):
        for name, identity, role, limit, _ in _CONTROLS:
            _asset(getattr(self, name), identity, role, limit)
        _asset(self.coordinates, "mosaic:coordinates", "mosaic_coordinates", 64 * MIB)
        _tuple(self.masks, MaskAsset, "mask assets")
        _tuple(self.features, FeatureAsset, "feature assets")
        keys = [(item.view.held_out_fold, item.view.budget_percent) for item in self.masks]
        _require(len(set(keys)) == len(keys), "duplicate selected views")
        indices = [item.slide_index for item in self.features]
        _require(len(set(indices)) == len(indices), "duplicate requested feature index")
        for item in self.features:
            _asset(item.asset, item.asset.asset_id, "mosaic_feature", 64 * MIB)
        ids = [getattr(self, n).asset_id for n, *_ in _CONTROLS] + [self.coordinates.asset_id]
        ids += [item.asset.asset_id for item in (*self.masks, *self.features)]
        _require(len(set(ids)) == len(ids), "duplicate consumer AssetID")


@dataclass(frozen=True, slots=True)
class SlideRecord:
    slide_index: int
    metadata_index: int
    slide_id: str
    patient_id: str
    diagnostic_group: str
    slide_fold: int
    tile_start: int
    tile_stop: int
    feature_rows: int

    def __post_init__(self):
        for name in ("slide_index", "metadata_index", "slide_fold", "tile_start", "tile_stop", "feature_rows"):
            _integer(getattr(self, name), name)
        for name in ("slide_id", "patient_id", "diagnostic_group"):
            _text(getattr(self, name), name)
        _require(self.slide_fold < 5 and self.feature_rows > 0
                 and self.tile_stop - self.tile_start == self.feature_rows, "invalid slide range/fold")


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    slide_index: int
    slide_id: str
    producer_path: str
    producer_resolved_path: str
    size: int
    sha256: str
    rows: int
    dtype: str
    columns: int
    consumer_asset: assets.AssetSpec | None

    def __post_init__(self):
        _integer(self.slide_index, "feature index")
        _integer(self.rows, "feature rows", 1)
        _integer(self.columns, "feature columns", 1)
        _text(self.slide_id, "feature slide")
        _descriptor(dict(path=self.producer_path, size=self.size, sha256=self.sha256),
                    "feature producer", 64 * MIB)
        _require(Path(_text(self.producer_resolved_path, "feature resolved path")).is_absolute(),
                 "feature resolved path must be absolute")
        _require(self.dtype == "float32" and self.columns == 1536, "feature FP32/1536 required")
        if self.consumer_asset is not None:
            _asset(self.consumer_asset, "feature:" + self.slide_id, "mosaic_feature", 64 * MIB)
            _require((self.consumer_asset.size, self.consumer_asset.sha256) == (self.size, self.sha256),
                     "consumer feature descriptor differs from accepted feature")


def _npy_header(data):
    _require(type(data) is bytes, "NPY immutable bytes required")
    # NPY1 has a two-byte header length: never copy its potentially huge body.
    handle = io.BytesIO(data[:65545])
    try:
        _require(np.lib.format.read_magic(handle) == (1, 0), "NPY1.0 required")
        _require(len(data) >= 10, "NPY1 header length missing")
        header_size = int.from_bytes(data[8:10], "little")
        _require(10 + header_size <= len(data), "NPY1 truncated header")
        header_bytes = data[10:10 + header_size]
        _require(header_bytes.endswith(b"\n") and (10 + header_size) % 16 == 0,
                 "NPY1 header framing")
        parsed_header = ast.parse(header_bytes.decode("latin1").strip(), mode="eval")
        _require(type(parsed_header.body) is ast.Dict and len(parsed_header.body.keys) == 3,
                 "NPY header exact dictionary fields (no duplicate keys)")
        header = ast.literal_eval(parsed_header)
        _keys(header, ("descr", "fortran_order", "shape"), "NPY header")
        _require(type(header["descr"]) is str and header["descr"] in ("<i8", "|b1"),
                 "NPY literal dtype descriptor required")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle, max_header_size=65535)
    except (ValueError, EOFError, TypeError, SyntaxError) as error:
        raise MosaicReaderError("invalid NPY header: " + str(error)) from error
    _require(type(shape) is tuple and bool(shape)
             and all(type(n) is int and n > 0 for n in shape), "NPY strict positive integer shape")
    _require(type(fortran) is bool and not fortran, "NPY C-order required")
    _require(dtype.fields is None and dtype.subdtype is None and dtype.str in ("<i8", "|b1"),
             "NPY coordinate/Boolean dtype required")
    count = 1
    for n in shape:
        count *= n
    offset = handle.tell()
    _require(offset + count * dtype.itemsize == len(data), "NPY exact extent/EOF required")
    return offset, dtype.str, shape, fortran


@dataclass(frozen=True, slots=True)
class NpyBacking:
    encoded: bytes
    data_offset: int
    dtype_str: str
    shape: tuple[int, ...]
    fortran_order: bool

    def __post_init__(self):
        _integer(self.data_offset, "NPY data offset")
        _require(type(self.dtype_str) is str and type(self.shape) is tuple
                 and all(type(n) is int for n in self.shape)
                 and type(self.fortran_order) is bool, "NPY metadata types")
        _require(_npy_header(self.encoded) ==
                 (self.data_offset, self.dtype_str, self.shape, self.fortran_order),
                 "NPY backing metadata mismatch")

    def array_view(self) -> np.ndarray:
        """Fresh view: immutable backing, no cached mutable shape metadata."""
        return np.ndarray(self.shape, dtype=self.dtype_str, buffer=self.encoded,
                          offset=self.data_offset, order="C")


@dataclass(frozen=True, slots=True)
class MaskCountRecord:
    held_out_fold: int
    budget_percent: int
    slide_index: int
    slide_id: str
    patient_id: str
    slide_fold: int
    query_fold_eligible: int
    original_tiles: int
    predicted_class_1: int
    predicted_class_2: int
    predicted_class_3: int
    predicted_class_4: int
    predicted_class_5: int
    retained_tiles: int
    discarded_tiles: int
    status: str

    def __post_init__(self):
        for name in COUNT_HEADER:
            if name in ("slide_id", "patient_id", "status"):
                _text(getattr(self, name), name)
            else:
                _integer(getattr(self, name), name)
        HeadView(self.held_out_fold, self.budget_percent)
        _require(self.slide_fold < 5 and self.query_fold_eligible == int(self.slide_fold == self.held_out_fold)
                 and self.original_tiles > 0
                 and sum(getattr(self, "predicted_class_" + str(i)) for i in range(1, 6)) == self.original_tiles
                 and self.retained_tiles == self.predicted_class_1
                 and self.discarded_tiles == self.original_tiles - self.retained_tiles
                 and self.status == ("nonempty" if self.retained_tiles else "zero_predicted_tumor"),
                 "mask count/fold/status mismatch")


@dataclass(frozen=True, slots=True)
class LoadedMask:
    view: HeadView
    backing: NpyBacking
    counts: tuple[MaskCountRecord, ...]

    def __post_init__(self):
        _require(type(self.view) is HeadView and type(self.backing) is NpyBacking, "LoadedMask records")
        _tuple(self.counts, MaskCountRecord, "LoadedMask.counts")
        _require(all((r.held_out_fold, r.budget_percent) ==
                     (self.view.held_out_fold, self.view.budget_percent) for r in self.counts),
                 "LoadedMask counts/view mismatch")


@dataclass(frozen=True, slots=True)
class ArrayLayout:
    dtype_str: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    c_contiguous: bool
    f_contiguous: bool

    def __post_init__(self):
        _text(self.dtype_str, "layout dtype")
        _require(type(self.shape) is tuple and type(self.strides) is tuple
                 and len(self.shape) == len(self.strides)
                 and all(type(n) is int and n >= 0 for n in self.shape)
                 and all(type(n) is int for n in self.strides)
                 and type(self.c_contiguous) is bool and type(self.f_contiguous) is bool,
                 "layout strict types")


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    name: str
    size: int
    sha256: str

    def __post_init__(self):
        _text(self.name, "artifact name")
        _require(Path(self.name).name == self.name and self.name not in (".", "..")
                 and "\\" not in self.name, "artifact leaf required")
        _integer(self.size, "artifact size")
        _digest(self.sha256)


@dataclass(frozen=True, slots=True)
class ConsumptionRecord:
    asset_id: str
    logical_name: str
    consumer_path: str
    producer_size: int
    producer_sha256: str
    semantic_checks: tuple[str, ...]

    def __post_init__(self):
        for name in ("asset_id", "logical_name", "consumer_path"):
            _text(getattr(self, name), name)
        _integer(self.producer_size, "consumed size")
        _digest(self.producer_sha256)
        _require(type(self.semantic_checks) is tuple
                 and all(type(v) is str and bool(v) for v in self.semantic_checks),
                 "consumption frozen check names")


@dataclass(frozen=True, slots=True)
class ReferenceRecord:
    kind: str
    logical_name: str
    details_json: str

    def __post_init__(self):
        _text(self.kind, "reference kind")
        _text(self.logical_name, "reference logical name")
        _text(self.details_json, "reference JSON")
        value = cpu.strict_json(self.details_json.encode("utf-8"))
        _require(cpu.json_bytes(value).decode("utf-8") == self.details_json, "canonical finite reference JSON")


@dataclass(frozen=True, slots=True)
class MosaicBundle:
    expected: MosaicExpected
    slides: tuple[SlideRecord, ...]
    feature_records: tuple[FeatureRecord, ...]
    coordinates: NpyBacking
    masks: tuple[LoadedMask, ...]
    output_registry: tuple[ArtifactDescriptor, ...]
    consumed_controls: tuple[ConsumptionRecord, ...]
    consumed_arrays: tuple[ConsumptionRecord, ...]
    referenced_only: tuple[ReferenceRecord, ...]

    def __post_init__(self):
        _require(type(self.expected) is MosaicExpected and type(self.coordinates) is NpyBacking,
                 "MosaicBundle records")
        for name, cls in (("slides", SlideRecord), ("feature_records", FeatureRecord),
                          ("masks", LoadedMask), ("output_registry", ArtifactDescriptor),
                          ("consumed_controls", ConsumptionRecord), ("consumed_arrays", ConsumptionRecord),
                          ("referenced_only", ReferenceRecord)):
            _tuple(getattr(self, name), cls, name)


@dataclass(frozen=True, slots=True)
class RawSlide:
    slide: SlideRecord
    raw: np.ndarray
    coordinates: np.ndarray
    layout: ArrayLayout
    consumption: ConsumptionRecord

    def __post_init__(self):
        _require(type(self.slide) is SlideRecord and type(self.raw) is np.ndarray
                 and type(self.coordinates) is np.ndarray and type(self.layout) is ArrayLayout
                 and type(self.consumption) is ConsumptionRecord, "RawSlide record types")


def _read(guard, asset, context, expected=None):
    if expected is not None:
        _keys(expected, ("size", "sha256"), context)
        _require((asset.size, asset.sha256) == (expected["size"], expected["sha256"]),
                 context + ": consumer/producer descriptor mismatch")
    data = guard.read_verified(asset.asset_id)
    _require(type(data) is bytes and len(data) == asset.size
             and hashlib.sha256(data).hexdigest() == asset.sha256,
             context + ": returned bytes/descriptor mismatch")
    return data


def _consumed(asset, logical, checks):
    return ConsumptionRecord(asset.asset_id, logical, asset.path, asset.size, asset.sha256, checks)


def _version_run(value, names, expected, context):
    _keys(value, names, context)
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1
             and value["run_id"] == expected.run_id and value["task"] == mosaic.TASK,
             context + ": schema/run/task mismatch")


def _binding_entry(entry, context):
    _keys(entry, ("asset_id", "role", "path", "resolved_path", "size", "sha256", "device", "inode"), context)
    _text(entry["asset_id"], context)
    _text(entry["role"], context)
    _descriptor({k: entry[k] for k in ("path", "size", "sha256")}, context)
    _require(Path(_text(entry["resolved_path"], context)).is_absolute(), context + ": resolved path")
    _integer(entry["device"], context)
    _integer(entry["inode"], context)


def _producer_graph(spec, summary, inputs, expected):
    _version_run(spec, ("schema_version", "run_id", "task", "sources", "code_files",
                       "features", "heads", "parameters", "expected"), expected, "producer spec")
    _same(spec["expected"], asdict(expected.counts), "producer expected counts")
    _same(spec["parameters"], {"prediction_batch_rows": 4096}, "producer parameters")
    _keys(spec["sources"], mosaic.SOURCES, "producer source roles")
    code_roles = set(mosaic.code_paths(mosaic._reader()))
    _keys(spec["code_files"], code_roles, "producer code roles")
    _require(type(spec["features"]) is list and len(spec["features"]) == expected.counts.slides
             and type(spec["heads"]) is list and len(spec["heads"]) == 10, "producer feature/head counts")
    declarations = {"spec": ("spec", None)}
    for kind, entries in (("source", spec["sources"]), ("code", spec["code_files"])):
        for role, entry in entries.items():
            _descriptor(entry, "producer " + kind, mosaic.source_limit(role) if kind == "source" else 2 * MIB)
            declarations[kind + ":" + role] = (role if kind == "source" else "code", entry)
    slides = set()
    for feature in spec["features"]:
        _keys(feature, ("slide_id", "path", "size", "sha256", "rows", "dtype", "columns"), "producer feature")
        slide = _text(feature["slide_id"], "producer feature ID")
        _require(slide not in slides, "duplicate producer feature")
        slides.add(slide)
        _integer(feature["rows"], "producer feature rows", 1)
        _integer(feature["columns"], "producer feature columns", 1)
        _require(feature["dtype"] == "float32" and feature["columns"] == 1536, "producer FP32/1536")
        _descriptor({k: feature[k] for k in ("path", "size", "sha256")}, "producer feature", 64 * MIB)
        declarations["feature:" + slide] = ("feature", feature)
    heads = {}
    for head in spec["heads"]:
        _keys(head, ("held_out_fold", "budget_percent", "assets", "registry_keys", "producer_spec_path"),
              "producer head declaration")
        view = HeadView(head["held_out_fold"], head["budget_percent"])
        key = view.held_out_fold, view.budget_percent
        _require(key not in heads, "duplicate producer head tuple")
        heads[key] = head
        _keys(head["assets"], mosaic.HEAD_ASSETS, "producer head assets")
        _keys(head["registry_keys"], ("complete", *mosaic.REGISTRY_NAMES), "producer registry keys")
        for path in (*head["registry_keys"].values(), head["producer_spec_path"]):
            _require(Path(_text(path, "historical head reference")).is_absolute(), "historical head path")
        for role, entry in head["assets"].items():
            _descriptor(entry, "producer head " + role, 2 * MIB)
            declarations[f"head:fold{key[0]}_budget{key[1]}:{role}"] = (
                "head_model" if role == "model" else "head_control", entry)
    _require(set(heads) == set(VIEW_ORDER), "producer ten-view membership")
    _version_run(summary, ("schema_version", "run_id", "task", "counts", "policy", "details", "limitations"),
                 expected, "summary")
    _same(summary["policy"], _POLICY, "summary policy")
    counts = summary["counts"]
    _keys(counts, (*mosaic.EXPECTED, "zero_mask_cells", "nonzero_mask_cells"), "summary counts")
    _same({k: counts[k] for k in mosaic.EXPECTED}, asdict(expected.counts), "summary expected counts")
    _integer(counts["zero_mask_cells"], "zero mask cells")
    _integer(counts["nonzero_mask_cells"], "nonzero mask cells")
    _require(counts["zero_mask_cells"] + counts["nonzero_mask_cells"] == expected.counts.slide_head_cells,
             "summary mask-cell coverage")
    details = summary["details"]
    _keys(details, ("head_provenance", "tuple_order", "feature_columns", "coordinate_columns",
                   "packed_array_data_bytes", "array_count", "decoded_feature_files", "consumed_feature_bytes",
                   "minimum_feature_bytes_with_final_sweep", "maximum_encoded_feature_bytes",
                   "maximum_dense_feature_bytes", "maximum_contiguous_feature_batch_bytes",
                   "coverage_accounting_bytes", "written_slide_head_cells", "written_tile_head_predictions",
                   "consumed_source_roles", "semantic_readback"), "summary details")
    _same(details["tuple_order"], [list(k) for k in VIEW_ORDER], "summary tuple order")
    _same(details["consumed_source_roles"], sorted(mosaic.SOURCES), "summary consumed source roles")
    _require(details["semantic_readback"] == "all_arrays_offsets_counts_masks_and_coverage_recounted",
             "summary semantic readback")
    numbers = {
        "feature_columns": 1536, "coordinate_columns": 2,
        "packed_array_data_bytes": expected.counts.original_tiles * 306,
        "array_count": 31, "decoded_feature_files": expected.counts.slides,
        "consumed_feature_bytes": sum(f["size"] for f in spec["features"]),
        "maximum_encoded_feature_bytes": max(f["size"] for f in spec["features"]),
        "maximum_dense_feature_bytes": max(f["rows"] * 6144 for f in spec["features"]),
        "maximum_contiguous_feature_batch_bytes": min(4096, max(f["rows"] for f in spec["features"])) * 6144,
        "coverage_accounting_bytes": expected.counts.slides * 90,
        "written_slide_head_cells": expected.counts.slide_head_cells,
        "written_tile_head_predictions": expected.counts.tile_head_predictions,
    }
    numbers["minimum_feature_bytes_with_final_sweep"] = numbers["consumed_feature_bytes"] * 2
    _same({k: details[k] for k in numbers}, numbers, "summary byte/count accounting")
    _version_run(inputs, ("schema_version", "run_id", "task", "spec_sha256", "bindings",
                         "executing_code_paths", "consumed_cpu_artifacts", "head_provenance",
                         "producer_artifact_links", "feature_consumption", "current_environment",
                         "started_at", "captured_at", "limitations"), expected, "input bindings")
    _require(inputs["spec_sha256"] == expected.spec_sha256, "input bindings spec hash")
    for name in ("started_at", "captured_at"):
        _text(inputs[name], name)
    _require(type(inputs["bindings"]) is list, "producer bindings list")
    binding_map = {}
    for entry in inputs["bindings"]:
        _binding_entry(entry, "producer binding")
        identity = entry["asset_id"]
        _require(identity not in binding_map, "duplicate producer binding AssetID")
        binding_map[identity] = entry
    _require(set(binding_map) == set(declarations), "producer binding membership mismatch")
    for identity, (role, entry) in declarations.items():
        binding = binding_map[identity]
        _require(binding["role"] == role, "producer binding role mismatch")
        if identity == "spec":
            _require(binding["sha256"] == expected.spec_sha256, "producer spec binding hash")
        else:
            _same({k: binding[k] for k in ("path", "size", "sha256")},
                  {k: entry[k] for k in ("path", "size", "sha256")}, "producer declaration/binding join")
    _same(inputs["executing_code_paths"],
          {role: binding_map["code:" + role]["resolved_path"] for role in code_roles}, "producer code paths")
    cpu_roles = {"head_plan.json": "head_plan", "input_bindings.json": "cpu_input_bindings",
                 "patient_folds.csv": "patient_folds", "retrieval_population.csv": "retrieval_population",
                 "summary.json": "cpu_summary"}
    _keys(inputs["consumed_cpu_artifacts"], cpu_roles, "producer consumed CPU artifacts")
    for name, role in cpu_roles.items():
        _same(inputs["consumed_cpu_artifacts"][name],
              {k: spec["sources"][role][k] for k in ("size", "sha256")}, "producer consumed CPU descriptor")
    _same(inputs["feature_consumption"],
          [dict(asset_id="feature:" + f["slide_id"], slide_id=f["slide_id"], status="decoded_once")
           for f in spec["features"]], "producer feature consumption/order")
    _same(inputs["head_provenance"], details["head_provenance"], "producer head provenance")
    for name in ("head_provenance", "producer_artifact_links"):
        records = inputs[name]
        _require(type(records) is list and len(records) == 10, name + ": ten records")
        for key, record in zip(VIEW_ORDER, records):
            _require(type(record) is dict and type(record.get("held_out_fold")) is int
                     and type(record.get("budget_percent")) is int
                     and (record["held_out_fold"], record["budget_percent"]) == key,
                     name + ": tuple order/types")
    for key, provenance, link in zip(VIEW_ORDER, inputs["head_provenance"], inputs["producer_artifact_links"]):
        _head_references(heads[key], provenance, link, spec["sources"])
    _keys(inputs["current_environment"], ("python", "hostname", "native_environment", "head_consumer_runtime"),
          "producer environment")
    for name in ("python", "hostname"):
        _text(inputs["current_environment"][name], "producer " + name)
    native = inputs["current_environment"]["native_environment"]
    _keys(native, producer.NATIVE_VARIABLES, "producer native environment")
    _require(all(v is None or type(v) is str for v in native.values()), "producer native declaration types")
    _same(inputs["current_environment"]["head_consumer_runtime"],
          [p["consumer_runtime_json"] for p in inputs["head_provenance"]], "producer runtime references")
    for record in (summary, inputs):
        _same(record["limitations"], list(mosaic.LIMITATIONS), "producer limitations")
    return binding_map


def _head_references(head, provenance, link, sources):
    _keys(provenance, ("held_out_fold", "budget_percent", "run_id", "producer_spec_path",
                      "registry_complete_key", "registry_artifact_keys", "artifact_sha256",
                      "fitted_state_sha256", "diagnostics_json", "consumer_runtime_json"),
          "head provenance reference")
    _text(provenance["run_id"], "head run")
    _digest(provenance["fitted_state_sha256"])
    _same(provenance["producer_spec_path"], head["producer_spec_path"], "head historical spec")
    _same(provenance["registry_complete_key"], head["registry_keys"]["complete"], "head complete key")
    _same(provenance["registry_artifact_keys"],
          [[name, head["registry_keys"][name]] for name in mosaic.REGISTRY_NAMES], "head artifact keys")
    common = {"acceptance": "head_acceptance", "registry": "head_registry",
              "pilot_acceptance": "pilot_acceptance", "head_plan": "head_plan"}
    roles = ("acceptance", "registry", "pilot_acceptance", *mosaic.HEAD_ASSETS, "head_plan")
    _same(provenance["artifact_sha256"],
          [[role, (sources[common[role]] if role in common else head["assets"][role])["sha256"]]
           for role in roles], "head artifact hash references")
    for name in ("diagnostics_json", "consumer_runtime_json"):
        text = _text(provenance[name], name)
        parsed = _json(text.encode("utf-8"), name)
        _require(cpu.json_bytes(parsed).decode() == text, "head canonical JSON reference")
    diagnostic = _json(provenance["diagnostics_json"].encode(), "head diagnostic reference")
    _same({k: diagnostic.get(k) for k in ("held_out_fold", "budget_percent", "fitted_state_sha256")},
          {k: provenance[k] for k in ("held_out_fold", "budget_percent", "fitted_state_sha256")},
          "head diagnostic identity reference")
    _keys(link, ("held_out_fold", "budget_percent", "complete_current_path", "complete_historical_registry_key",
                 "producer_spec_current_path", "producer_spec_historical_path", "consumed_artifacts",
                 "referenced_not_reread"), "head producer links")
    for key, value in (
        ("complete_current_path", head["assets"]["complete"]["path"]),
        ("complete_historical_registry_key", head["registry_keys"]["complete"]),
        ("producer_spec_current_path", head["assets"]["spec"]["path"]),
        ("producer_spec_historical_path", head["producer_spec_path"]),
    ):
        _same(link[key], value, "head producer path link")
    file_roles = {"head.pkl": "model", "fit_diagnostics.json": "diagnostics",
                  "input_bindings.json": "input_bindings", "summary.json": "summary"}
    _keys(link["consumed_artifacts"], file_roles, "head consumed artifact references")
    for name, role in file_roles.items():
        entry = link["consumed_artifacts"][name]
        _keys(entry, ("asset_id", "current_path", "historical_registry_key", "size", "sha256"),
              "head consumed reference")
        descriptor = head["assets"][role]
        _same(entry, dict(asset_id=f"head:fold{head['held_out_fold']}_budget{head['budget_percent']}:{role}",
                          current_path=descriptor["path"], historical_registry_key=head["registry_keys"][name],
                          size=descriptor["size"], sha256=descriptor["sha256"]), "head consumption reference")
    _keys(link["referenced_not_reread"], ("counts.json", "held_out_rows.csv.gz", "predictions.npy",
                                        "probabilities.npy"), "head unconsumed references")
    for name, entry in link["referenced_not_reread"].items():
        _keys(entry, ("size", "sha256"), "head unconsumed " + name)
        _integer(entry["size"], "head reference size")
        _digest(entry["sha256"])


def _tables(data, spec, binding_map, bundle_assets, expected):
    n = expected.counts.slides
    population = tuple(producer.csv_rows(data["population"], POPULATION_HEADER, "population",
                       expanded_limit=16 * MIB, row_limit=n, record_limit=8448))
    offsets = tuple(producer.csv_rows(data["offsets"], OFFSET_HEADER, "offsets",
                    expanded_limit=OFFSET_EXPANSION_LIMIT, row_limit=n, record_limit=8448))
    _require(len(population) == len(offsets) == n, "population/offset record count")
    consumers = {f.slide_index: f.asset for f in bundle_assets.features}
    _require(all(index < n for index in consumers), "requested feature index outside population")
    slides, features = [], []
    seen, metadata, patients, groups, folds = set(), set(), set(), set(), {}
    end = 0
    for i, (p, o, f) in enumerate(zip(population, offsets, spec["features"])):
        p = dict(p)
        o = dict(o)
        for name in ("query_index", "reference_index", "metadata_index", "fold"):
            p[name] = producer._decimal(p[name], name)
        for name in ("slide_index", "slide_fold", "tile_start", "tile_stop", "feature_rows"):
            o[name] = producer._decimal(o[name], name)
        _require(p["query_index"] == p["reference_index"] == o["slide_index"] == i,
                 "population/offset index order")
        _require(p["slide_id"] == o["slide_id"] == f["slide_id"] and p["slide_id"] not in seen
                 and p["metadata_index"] not in metadata, "population feature identity/order/duplicates")
        _require(p["patient_id"] == o["patient_id"] and p["diagnostic_group"] == o["diagnostic_group"]
                 and p["fold"] == o["slide_fold"] and p["diagnostic_group"] not in ("MISC", "NORMAL"),
                 "population/offset patient/group/fold join")
        _require(p["patient_id"] not in folds or folds[p["patient_id"]] == p["fold"],
                 "patient assigned multiple folds")
        _require(o["tile_start"] == end and o["tile_stop"] - end == o["feature_rows"] == f["rows"]
                 and o["feature_sha256"] == f["sha256"], "offset range/feature descriptor join")
        s = SlideRecord(i, p["metadata_index"], p["slide_id"], p["patient_id"], p["diagnostic_group"],
                        p["fold"], o["tile_start"], o["tile_stop"], o["feature_rows"])
        b = binding_map["feature:" + s.slide_id]
        slides.append(s)
        features.append(FeatureRecord(i, s.slide_id, b["path"], b["resolved_path"], b["size"], b["sha256"],
                                     f["rows"], f["dtype"], f["columns"], consumers.get(i)))
        seen.add(s.slide_id)
        metadata.add(s.metadata_index)
        patients.add(s.patient_id)
        groups.add(s.diagnostic_group)
        folds[s.patient_id] = s.slide_fold
        end = s.tile_stop
    _require(end == expected.counts.original_tiles and len(patients) == expected.counts.patients
             and len(groups) == expected.counts.diagnostic_groups, "population/offset totals")
    _same(dict(size=len(data["population"]), sha256=hashlib.sha256(data["population"]).hexdigest()),
          {k: spec["sources"]["retrieval_population"][k] for k in ("size", "sha256")},
          "accepted population byte-copy/source join")
    counts = {key: [] for key in VIEW_ORDER}
    rows = producer.csv_rows(data["mask_counts"], COUNT_HEADER, "mask counts", compressed=True,
                            expanded_limit=COUNT_EXPANSION_LIMIT, row_limit=10 * n, record_limit=8448)
    total = 0
    for j, row in enumerate(rows):
        key, i = VIEW_ORDER[j // n], j % n
        values = {name: row[name] if name in ("slide_id", "patient_id", "status") else
                  producer._decimal(row[name], name) for name in COUNT_HEADER}
        r = MaskCountRecord(**values)
        s = slides[i]
        _require((r.held_out_fold, r.budget_percent) == key and r.slide_index == i
                 and r.slide_id == s.slide_id and r.patient_id == s.patient_id
                 and r.slide_fold == s.slide_fold and r.original_tiles == s.feature_rows,
                 "mask count population/tuple order join")
        counts[key].append(r)
        total += 1
    _require(total == 10 * n, "missing mask count rows")
    return tuple(slides), tuple(features), {key: tuple(value) for key, value in counts.items()}


def load_mosaic_bundle(guard, *, bundle_assets: MosaicBundleAssets, expected: MosaicExpected) -> MosaicBundle:
    """Read declared subset bytes; return immutable metadata, never a task success."""
    _require(type(bundle_assets) is MosaicBundleAssets and type(expected) is MosaicExpected,
             "MosaicBundleAssets/MosaicExpected required")
    a = bundle_assets
    _require(all(f.slide_index < expected.counts.slides for f in a.features),
             "requested feature index outside expected population")
    for asset, digest in ((a.review, expected.review_sha256), (a.complete, expected.complete_sha256),
                          (a.spec, expected.spec_sha256)):
        _require(asset.sha256 == digest, "expected trust-anchor descriptor mismatch")
    review = _json(_read(guard, a.review, "review"), "review")
    _require(set(review) <= {"schema_version", "stage", "disposition", "bindings", "review_date", "reviewer",
                            "job_id", "independent_evidence", "query_fold_eligible_mask_counts", "resources", "limits"},
             "unknown review105 fields (not a CPU branches/output-registry review)")
    _require(type(review.get("schema_version")) is int and review["schema_version"] == 1
             and review.get("stage") == REVIEW_STAGE and review.get("disposition") == REVIEW_DISPOSITION,
             "review105 schema/stage/disposition")
    links = review.get("bindings")
    _keys(links, ("complete_sha256", "input_bindings_sha256", "spec_sha256", "summary_sha256",
                  "lead_reconciliation_sha256", "runtime_acceptance_sha256", "cpu_preflight_receipt_sha256"),
          "review105 bindings")
    for value in links.values():
        _digest(value)
    _require(links["complete_sha256"] == expected.complete_sha256
             and links["spec_sha256"] == expected.spec_sha256
             and links["summary_sha256"] == a.summary.sha256
             and links["input_bindings_sha256"] == a.input_bindings.sha256, "review105 linked descriptors")
    complete = _json(_read(guard, a.complete, "COMPLETE"), "COMPLETE")
    _version_run(complete, ("schema_version", "run_id", "task", "scope", "spec_sha256",
                           "outputs", "verified_at", "limitations"), expected, "COMPLETE")
    _require(complete["scope"] == mosaic.SCOPE and complete["spec_sha256"] == expected.spec_sha256,
             "COMPLETE scope/spec")
    _text(complete["verified_at"], "COMPLETE time")
    _same(complete["limitations"], list(mosaic.LIMITATIONS), "COMPLETE limitations")
    registry = complete["outputs"]
    _keys(registry, mosaic.artifact_names(), "full36 artifact registry")
    descriptors = []
    for name, entry in sorted(registry.items()):
        _keys(entry, ("size", "sha256"), name)
        descriptors.append(ArtifactDescriptor(name, entry["size"], entry["sha256"]))
    raw = {"spec": _read(guard, a.spec, "producer spec")}
    for name, _, _, _, logical in _CONTROLS[3:]:
        raw[name] = _read(guard, getattr(a, name), logical, registry[logical])
    spec, summary, inputs = (_json(raw[name], name) for name in ("spec", "summary", "input_bindings"))
    binding_map = _producer_graph(spec, summary, inputs, expected)
    _require(binding_map["spec"]["size"] == a.spec.size, "producer spec binding size")
    slides, features, counts = _tables(raw, spec, binding_map, a, expected)
    zero_cells = sum(r.retained_tiles == 0 for records in counts.values() for r in records)
    _require(zero_cells == summary["counts"]["zero_mask_cells"], "summary/count zero-cell totals")
    data = _read(guard, a.coordinates, "coordinates.npy", registry["coordinates.npy"])
    meta = _npy_header(data)
    _require(meta[1:3] == ("<i8", (expected.counts.original_tiles, 2)), "coordinate NPY dtype/shape")
    coordinates = NpyBacking(data, *meta)
    xy = coordinates.array_view()
    for slide in slides:
        part = xy[slide.tile_start:slide.tile_stop]
        _require(bool((part >= 0).all()) and len({(int(x), int(y)) for x, y in part}) == slide.feature_rows,
                 "coordinate nonnegative/per-slide unique")
    masks = []
    consumed_arrays = [_consumed(a.coordinates, "coordinates.npy",
                                ("NPY1_C_i8_shape_EOF", "all_slide_ranges_nonnegative_unique"))]
    for declaration in sorted(a.masks, key=lambda m: (m.view.held_out_fold, m.view.budget_percent)):
        view = declaration.view
        name = f"fold{view.held_out_fold}_budget{view.budget_percent}_mask.npy"
        data = _read(guard, declaration.asset, name, registry[name])
        meta = _npy_header(data)
        _require(meta[1:3] == ("|b1", (expected.counts.original_tiles,)), "mask NPY dtype/shape")
        backing = NpyBacking(data, *meta)
        for start in range(meta[0], len(data), MIB):
            values = np.frombuffer(data, dtype=np.uint8, offset=start, count=min(MIB, len(data) - start))
            _require(bool((values <= 1).all()), "noncanonical Boolean payload")
        values = backing.array_view()
        records = counts[view.held_out_fold, view.budget_percent]
        for slide, row in zip(slides, records):
            _require(int(values[slide.tile_start:slide.tile_stop].sum()) == row.retained_tiles,
                     "selected mask/count slice mismatch")
        masks.append(LoadedMask(view, backing, records))
        consumed_arrays.append(_consumed(declaration.asset, name,
                                        ("NPY1_C_bool_shape_EOF", "canonical_bool_bytes",
                                         "named_view_population_counts")))
    consumed_controls = tuple(_consumed(getattr(a, name), logical,
                              ("authenticated_descriptor", "accepted105_producer_graph_and_table_joins"))
                              for name, _, _, _, logical in _CONTROLS)
    consumed_names = {r.logical_name for r in (*consumed_controls, *consumed_arrays)}
    references = [ReferenceRecord("output", name, cpu.json_bytes(entry).decode())
                  for name, entry in sorted(registry.items()) if name not in consumed_names]
    for kind, mapping in (("source", spec["sources"]), ("code", spec["code_files"])):
        references.extend(ReferenceRecord(kind, role, cpu.json_bytes(entry).decode())
                          for role, entry in sorted(mapping.items()))
    ordered_heads = sorted(spec["heads"], key=lambda h: (h["held_out_fold"], h["budget_percent"]))
    for kind, records in (("head", ordered_heads), ("head_provenance", inputs["head_provenance"]),
                          ("head_producer_links", inputs["producer_artifact_links"])):
        references.extend(ReferenceRecord(kind, f"fold{k[0]}_budget{k[1]}", cpu.json_bytes(r).decode())
                          for k, r in zip(VIEW_ORDER, records))
    return MosaicBundle(expected, slides, features, coordinates, tuple(masks), tuple(descriptors),
                        consumed_controls, tuple(consumed_arrays), tuple(references))


def load_raw_slide(guard, bundle: MosaicBundle, slide_index: int) -> RawSlide:
    """Authenticate one declared original raw slide and its complete ordered XY."""
    _require(type(bundle) is MosaicBundle, "MosaicBundle required")
    _integer(slide_index, "raw slide_index")
    _require(slide_index < len(bundle.slides), "raw slide_index outside population")
    slide, feature = bundle.slides[slide_index], bundle.feature_records[slide_index]
    _require(feature.consumer_asset is not None, "raw feature not declared for consumption")
    data = _read(guard, feature.consumer_asset, "raw " + slide.slide_id,
                 dict(size=feature.size, sha256=feature.sha256))
    declared = dict(slide_id=feature.slide_id, path=feature.producer_path, size=feature.size,
                    sha256=feature.sha256, rows=feature.rows, dtype=feature.dtype, columns=feature.columns)
    raw, original_xy = mosaic.decode_feature(data, declared)
    xy = bundle.coordinates.array_view()[slide.tile_start:slide.tile_stop]
    _require(np.array_equal(original_xy, xy), "raw ordered coordinate slice mismatch")
    layout = ArrayLayout(raw.dtype.str, raw.shape, raw.strides,
                         bool(raw.flags.c_contiguous), bool(raw.flags.f_contiguous))
    raw.flags.writeable = False
    consumption = _consumed(feature.consumer_asset, "feature:" + slide.slide_id,
                            ("accepted_feature_bytes", "frozen_tuple_FP32_1536_decoder",
                             "complete_ordered_coordinate_slice_match", "decoded_layout_preserved"))
    return RawSlide(slide, raw, xy, layout, consumption)
