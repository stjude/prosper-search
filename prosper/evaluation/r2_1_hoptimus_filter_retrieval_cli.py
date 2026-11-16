"""Bounded direct-source filtering pilot; no cohort inference or production runner."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import csv
import hashlib
import importlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import sys
import warnings
import weakref

import numpy as np
import ot
import ot.lp._network_simplex as simplex
import ot.lp.emd_wrap as emd_wrap
import pandas as pd
import threadpoolctl

from prosper.embeddings.slide_embedding import bob_embedding as bob
from prosper.embeddings.slide_embedding import ot_embedding as transport
from . import feature_bank_audit as legacy
from . import r2_1_hoptimus_asset_io as assets
from . import r2_1_hoptimus_fit as fit
from . import r2_1_hoptimus_head_cli as producer
from . import r2_1_hoptimus_io as cpu
from . import r2_1_hoptimus_main as kernel
from . import r2_1_hoptimus_main_cli as main_cli
from . import r2_1_hoptimus_mosaic_cli as mosaic

MIB = 1024**2
_DIAGNOSTIC_INTEGER_BITS = 256
_DIAGNOSTIC_INTEGER_TYPES = (
    int, np.int8, np.int16, np.int32, np.int64, np.longlong,
    np.uint8, np.uint16, np.uint32, np.uint64, np.ulonglong,
)
_SOLVER_LOG_KEYS = ("u", "v", "warning", "result_code")
TASK = "hoptimus_filter_retrieval_pilot"
SCOPE = "direct_source_filtering_pilot_only"
METHODS = ("ot_cosine", "yottixel")
TOP_KEYS = (
    "schema_version", "task", "run_id", "code_files", "mosaic_bundle_assets",
    "mosaic_expected", "baseline_assets", "baseline_expected", "runtime_assets",
    "query_indices", "pair_requests", "row_requests", "limits",
)
LIMITS = {
    "max_queries": 2, "max_pair_requests": 32, "max_row_requests": 12,
    "max_population_slides": 4096, "max_raw_rows_per_slide": 8192,
    "max_pair_cost_cells": 32000000, "max_accounted_working_bytes": 2147483648,
}
LIMITATIONS = [
    "structural_and_unchanged_controls_only_not_cohort_inference",
    "fixed_reference_folds_heads_masks_and_saved_baselines",
    "no_historical_encoder_or_whole_matrix_lineage_certification",
    "exact_control_equality_not_equivalence_margin",
    "cooperative_file_custody_not_atomic_snapshot",
    "accounted_buffers_not_rss_or_solver_workspace_bound",
    "runtime_observation_not_full_executable_mapping",
    "no_next_stage_or_job_authorization",
]
PREPARATION_HEADER = (
    "preparation_id", "slide_index", "slide_id", "method", "selection",
    "budget_percent", "query_fold", "raw_rows", "selected_rows",
    "raw_dtype", "raw_shape", "raw_strides", "raw_c_contiguous", "raw_f_contiguous",
    "selected_dtype", "selected_shape", "selected_strides",
    "selected_c_contiguous", "selected_f_contiguous",
    "prepared_dtype", "prepared_shape", "prepared_strides",
    "prepared_c_contiguous", "prepared_f_contiguous",
)
PAIR_HEADER = (
    "request_id", "query_index", "reference_index", "query_preparation_id",
    "reference_preparation_id", "same_patient_excluded", "zero_query",
    "zero_reference", "observation_status", "computed_valid", "distance_hex",
    "cosine_min_hex", "cosine_max_hex", "cosine_negative_count",
    "solver_result_code", "solver_warning_is_none", "captured_warning_count",
    "scalar_distance_hex", "scalar_log_comparison", "baseline_distance_hex",
    "baseline_comparison", "literal_comparison",
)
ROW_HEADER = (
    "request_id", "reference_index", "reference_id", "query_preparation_id",
    "reference_preparation_id", "same_patient_excluded", "zero_query",
    "zero_reference", "observation_status", "computed_valid", "distance_hex",
    "cosine_min_hex", "cosine_max_hex", "cosine_negative_count",
    "solver_result_code", "solver_warning_is_none", "captured_warning_count",
    "candidate_rank", "baseline_distance_hex", "baseline_rank",
    "baseline_distance_comparison", "baseline_rank_comparison",
)
RANK_HEADER = (
    "request_id", "rank", "reference_index", "reference_id", "reference_group",
    "relevant", "distance_hex",
)
HEX_COLUMNS = frozenset({
    "distance_hex", "cosine_min_hex", "cosine_max_hex",
    "scalar_distance_hex", "baseline_distance_hex",
})
ARTIFACTS = (
    "requests.json", "preparations.csv", "pair_probes.csv", "row_cells.csv",
    "rank_probes.csv", "summary.json", "runtime.json", "input_bindings.json",
)
MAIN_OUTPUTS = {
    "contrasts.csv", "estimates.csv", "input_bindings.json", "patient_draws.npy",
    "populations.csv", "query_metrics.csv.gz", "replicate_index.csv",
    "replicate_valid.npy", "replicates.npy", "summary.json", "tie_sensitivity.csv.gz",
}
MOSAIC_CONTROLS = {
    "review": ("mosaic:review105", "mosaic_review", 2*MIB),
    "complete": ("mosaic:complete", "mosaic_complete", 2*MIB),
    "spec": ("mosaic:spec", "mosaic_spec", 8*MIB),
    "summary": ("mosaic:summary", "mosaic_summary", 16*MIB),
    "input_bindings": ("mosaic:input_bindings", "mosaic_input_bindings", 16*MIB),
    "population": ("mosaic:population", "mosaic_population", 16*MIB),
    "offsets": ("mosaic:offsets", "mosaic_offsets", 16*MIB),
    "mask_counts": ("mosaic:mask_counts", "mosaic_mask_counts", 16*MIB),
    "coordinates": ("mosaic:coordinates", "mosaic_coordinates", 64*MIB),
}


class PilotError(ValueError):
    """A declaration, numerical interface, control or custody requirement failed."""


class PilotPromotionHeld(PilotError):
    """Retained raw evidence is not eligible for compatibility promotion."""


def require(condition, message):
    if not condition:
        raise PilotError(message)


def keys(value, names, context):
    require(type(value) is dict and set(value) == set(names), f"{context}: exact keys required")


def integer(value, context, minimum=0):
    require(type(value) is int and value >= minimum, f"{context}: strict integer required")
    return value


def identifier(value, context):
    require(type(value) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value),
            f"{context}: invalid identifier")
    return value


def binding(data):
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def asset(value, asset_id, role, ceiling):
    keys(value, ("asset_id", "role", "path", "sha256", "size"), asset_id)
    require(value["asset_id"] == asset_id and value["role"] == role,
            f"{asset_id}: wrong identity/role")
    integer(value["size"], asset_id)
    require(value["size"] <= ceiling, f"{asset_id}: pre-read ceiling")
    return assets.AssetSpec(**value)


def _reader():
    return importlib.import_module("prosper.evaluation.r2_1_hoptimus_mosaic_reader")


def code_paths():
    paths = mosaic.code_paths(mosaic._reader())
    paths.update(runner=Path(__file__), mosaic_runner=Path(mosaic.__file__),
                 mosaic_reader=Path(_reader().__file__), main_cli=Path(main_cli.__file__),
                 main_kernel=Path(kernel.__file__))
    return paths


def _request(value, *, pair):
    fields = ("request_id", "query_index", "query_fold", "method", "selection", "budget_percent")
    keys(value, fields + (("reference_index",) if pair else ()), "request")
    identifier(value["request_id"], "request ID")
    integer(value["query_index"], "query index")
    require(integer(value["query_fold"], "query fold") < 5, "query fold out of range")
    if pair:
        integer(value["reference_index"], "reference index")
    require(value["method"] in METHODS, "unknown method")
    choices = ("filtered", "literal_all_tissue", "all_true") if pair else (
        "filtered", "literal_all_tissue")
    require(value["selection"] in choices, "unknown selection")
    if value["selection"] == "filtered":
        require(type(value["budget_percent"]) is int and value["budget_percent"] in (5, 100),
                "filtered budget must be 5 or 100")
    else:
        require(value["budget_percent"] is None, "control budget must be null")
    return dict(value)


def validate_spec(spec):
    keys(spec, TOP_KEYS, "pilot spec")
    require(type(spec["schema_version"]) is int and spec["schema_version"] == 1
            and spec["task"] == TASK, "wrong pilot schema/task")
    identifier(spec["run_id"], "run ID")
    keys(spec["limits"], LIMITS, "limits")
    for name, maximum in LIMITS.items():
        require(integer(spec["limits"][name], name, 1) <= maximum, f"{name}: upper bound")
    for name in ("query_indices", "pair_requests", "row_requests"):
        require(type(spec[name]) is list, f"{name}: list required")
    pairs = [_request(row, pair=True) for row in spec["pair_requests"]]
    rows = [_request(row, pair=False) for row in spec["row_requests"]]
    require(pairs or rows, "empty request plan")
    require(len(pairs) <= spec["limits"]["max_pair_requests"] and
            len(rows) <= spec["limits"]["max_row_requests"], "request safety cap")
    seen_ids, seen_queries, pair_keys, row_keys = set(), [], set(), set()
    literal_pairs = set()
    for request in pairs + rows:
        require(request["request_id"] not in seen_ids, "duplicate request ID")
        seen_ids.add(request["request_id"])
        if request["query_index"] not in seen_queries:
            seen_queries.append(request["query_index"])
    for query in spec["query_indices"]:
        integer(query, "declared query")
    require(spec["query_indices"] == seen_queries and
            len(seen_queries) <= spec["limits"]["max_queries"], "query union/order/cap")
    require(len({r["query_fold"] for r in pairs + rows}) == 1, "mixed query folds")
    for request in pairs:
        identity = recipe(request) + (request["query_index"], request["reference_index"])
        require(identity not in pair_keys, "duplicate pair recipe")
        pair_keys.add(identity)
        literal_key = (request["method"], request["query_index"], request["reference_index"])
        if request["selection"] == "literal_all_tissue":
            literal_pairs.add(literal_key)
        if request["selection"] == "all_true":
            require(literal_key in literal_pairs, "missing/inverted literal pair partner")
    literal_rows = {(r["method"], r["query_index"]) for r in rows
                    if r["selection"] == "literal_all_tissue"}
    for request in rows:
        identity = recipe(request) + (request["query_index"],)
        require(identity not in row_keys, "duplicate row recipe")
        row_keys.add(identity)
        require(request["selection"] != "filtered" or
                (request["method"], request["query_index"]) in literal_rows,
                "missing literal row control")
    methods = tuple(m for m in METHODS if any(
        r["method"] == m and (r in rows or r["selection"] != "filtered") for r in pairs + rows))
    require((spec["baseline_assets"] is not None) == bool(methods) and
            (spec["baseline_expected"] is not None) == bool(methods), "baseline presence mismatch")
    return pairs, rows, methods


def recipe(request):
    return (request["method"], request["selection"], request["budget_percent"], request["query_fold"])


def declarations(spec, spec_path, spec_data, paths, reader):
    declared, policies = [], {}

    def add(value, aid, role, cap, access="bytes"):
        entry = asset(value, aid, role, cap)
        declared.append(entry)
        policies[role] = assets.AssetPolicy(access, cap)
        return entry

    declared.append(assets.AssetSpec("pilot:spec", "pilot_spec", spec_path,
                                     binding(spec_data)["sha256"], len(spec_data)))
    policies["pilot_spec"] = assets.AssetPolicy("bytes", 8*MIB)
    keys(spec["code_files"], paths, "code role graph")
    for role, actual in paths.items():
        entry = add(spec["code_files"][role], "code:"+role, "executing_code", 64*MIB, "path")
        require(Path(entry.path).resolve(strict=True) == Path(actual).resolve(strict=True),
                f"{role}: executing code path mismatch")
    bundle = spec["mosaic_bundle_assets"]
    keys(bundle, tuple(MOSAIC_CONTROLS) + ("masks", "features"), "mosaic assets")
    controls = {name: add(bundle[name], *description)
                for name, description in MOSAIC_CONTROLS.items()}
    require(type(bundle["masks"]) is list and type(bundle["features"]) is list,
            "mosaic masks/features must be lists")
    mask_records, feature_records = [], []
    for item in bundle["masks"]:
        keys(item, ("view", "asset"), "mask asset")
        keys(item["view"], ("held_out_fold", "budget_percent"), "head view")
        f, b = item["view"]["held_out_fold"], item["view"]["budget_percent"]
        require(type(f) is int and f in range(5) and type(b) is int and b in (5, 100),
                "invalid head view")
        entry = add(item["asset"], f"mosaic:mask:fold{f}:budget{b}", "mosaic_mask", 8*MIB)
        mask_records.append(reader.MaskAsset(reader.HeadView(f, b), entry))
    for item in bundle["features"]:
        keys(item, ("slide_index", "asset"), "feature asset")
        index = integer(item["slide_index"], "feature index")
        value = item["asset"]
        require(type(value) is dict and type(value.get("asset_id")) is str and
                value["asset_id"].startswith("feature:"), "feature asset ID")
        entry = add(value, value["asset_id"], "mosaic_feature", 64*MIB)
        feature_records.append(reader.FeatureAsset(index, entry))
    typed_assets = reader.MosaicBundleAssets(**controls, masks=tuple(mask_records),
                                            features=tuple(feature_records))
    expected = spec["mosaic_expected"]
    keys(expected, ("review_sha256", "complete_sha256", "spec_sha256", "run_id", "counts"),
         "mosaic expected")
    count_fields = ("slides", "patients", "diagnostic_groups", "union_patients", "feature_files",
                    "original_tiles", "heads", "slide_head_cells", "tile_head_predictions")
    keys(expected["counts"], count_fields, "mosaic counts")
    for name in count_fields:
        integer(expected["counts"][name], name, 1)
    for name in ("review_sha256", "complete_sha256", "spec_sha256"):
        cpu.digest(expected[name], name)
    typed_expected = reader.MosaicExpected(**{k: v for k, v in expected.items() if k != "counts"},
                                          counts=reader.MosaicCounts(**expected["counts"]))
    require(typed_expected.counts.slides <= spec["limits"]["max_population_slides"],
            "pre-read population safety cap")
    runtime = spec["runtime_assets"]
    keys(runtime, ("expected_profile", "implementations"), "runtime assets")
    add(runtime["expected_profile"], "runtime:expected", "runtime_profile", 8*MIB)
    require(type(runtime["implementations"]) is list, "runtime implementation list")
    for value in runtime["implementations"]:
        require(type(value) is dict and type(value.get("asset_id")) is str and
                value["asset_id"].startswith("runtime:impl:"), "runtime implementation ID")
        add(value, value["asset_id"], "runtime_implementation", 2048*MIB, "path")
    if spec["baseline_assets"] is not None:
        baseline = spec["baseline_assets"]
        keys(baseline, ("review", "complete", "spec", "input_bindings", "metadata",
                        "matrices", "historical_review_complete_key"), "baseline assets")
        for name, suffix in (("review", "review57"), ("complete", "complete"), ("spec", "spec"),
                             ("input_bindings", "input_bindings"), ("metadata", "metadata")):
            add(baseline[name], "baseline:"+suffix, "baseline_"+name, 8*MIB)
        _, _, methods = validate_spec(spec)
        keys(baseline["matrices"], methods, "baseline methods")
        for method in methods:
            add(baseline["matrices"][method], "baseline:matrix:"+method,
                "baseline_matrix", 128*MIB)
        keys(spec["baseline_expected"], ("review_sha256", "complete_sha256", "spec_sha256",
                                        "input_bindings_sha256", "run_id"), "baseline expected")
        for name, value in spec["baseline_expected"].items():
            if name.endswith("sha256"):
                cpu.digest(value, name)
    require(len({entry.asset_id for entry in declared}) == len(declared), "duplicate asset IDs")
    return declared, policies, typed_assets, typed_expected


class WorkingMemory:
    def __init__(self, limit):
        self.limit, self.live, self.peak = limit, 0, 0
        self.retained = {}

    def hold(self, name, size):
        integer(size, "buffer reservation")
        require(name not in self.retained and self.live + size <= self.limit,
                f"pre-allocation working limit: {name}")
        self.retained[name] = size
        self.live += size
        self.peak = max(self.peak, self.live)

    def release(self, name):
        self.live -= self.retained.pop(name)

    @contextmanager
    def temporary(self, size):
        token = object()
        self.hold(token, size)
        try:
            yield
        finally:
            self.release(token)

    def summary(self):
        return {"peak_accounted_working_bytes": self.peak, "limit_bytes": self.limit,
                "basis": "conservative_buffers_and_known_reservations_not_rss"}


def pot_interface():
    signature = inspect.signature(ot.emd2)
    defaults = {name: parameter.default for name, parameter in signature.parameters.items()
                if parameter.default is not inspect.Parameter.empty}
    cpu.json_bytes(defaults)
    for name, value in (("center_dual", True), ("check_marginals", True),
                        ("processes", 1), ("numThreads", 1)):
        require(name in defaults and type(defaults[name]) is type(value) and defaults[name] == value,
                f"POT scalar/log default changed: {name}")
    return {
        "signature": str(signature), "defaults": defaults,
        "call_arguments": {"dist": {"metric": "cosine"}, "emd2": {
            "numItermax": 1000000, "numThreads": 1, "log": True, "return_matrix": False}},
        "optimal_status_source": {"package": "POT", "release": "0.9.6.post1", "optimal_code": 1,
            "emd_h_git_blob": "b56f0601bbcc8f10a6671153ded6a23c408428e8",
            "emd_wrap_git_blob": "53df54fc36b804ff2eb7e179ae25d35071f7194b"},
        "numeric_dtype_recipe": {"status": "declared", "raw": "float32", "prepared_ot": "float32",
            "cost": "float32", "masses": "float64", "lp_cost": "float64", "distance": "float64",
            "barcodes": "uint8", "barcode_width": 1535},
    }


def runtime_snapshot(implementation_assets):
    base = fit._runtime_profile()
    extension = np._core._multiarray_umath
    logical = {"python": sys.executable, "pandas": pd.__file__,
               "threadpoolctl": threadpoolctl.__file__, "POT": ot.__file__,
               "POT.dist": inspect.getfile(ot.dist), "POT.network_simplex": simplex.__file__,
               "POT.emd_wrap": emd_wrap.__file__, "numpy_cpu_extension": extension.__file__}
    for name, value in base["implementation"].items():
        logical["base:"+name] = value["path"]
    for index, pool in enumerate(base["native_pools"]):
        logical[f"native_pool:{index}"] = pool["filepath"]
    by_path = {}
    for value in implementation_assets:
        entry = assets.AssetSpec(**value)
        path = str(Path(entry.path).resolve(strict=True))
        require(path not in by_path, "runtime duplicate canonical implementation")
        by_path[path] = entry
    require(set(by_path) == {str(Path(path).resolve(strict=True)) for path in logical.values()},
            "runtime missing/extra implementations")
    observed = {}
    for name, path in logical.items():
        entry = by_path[str(Path(path).resolve(strict=True))]
        observed[name] = {"path": str(path), "asset_id": entry.asset_id}
    normalized = cpu.strict_json(cpu.json_bytes(base))
    for name, value in normalized["implementation"].items():
        value["path"] = observed["base:"+name]["asset_id"]
    for index, pool in enumerate(normalized["native_pools"]):
        pool["filepath"] = observed[f"native_pool:{index}"]["asset_id"]
    normalized["native_pools"].sort(key=cpu.json_bytes)
    environment = {name: os.environ.get(name) for name in producer.NATIVE_VARIABLES}
    environment["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(all(environment[name] == "1" for name in producer.NATIVE_VARIABLES) and
            environment["CUDA_VISIBLE_DEVICES"] == "", "native/CUDA declaration mismatch")
    versions = {name: importlib.metadata.version(name) for name in ("POT", "pandas", "threadpoolctl")}
    versions["pot_module_version"] = ot.__version__
    require(versions["POT"] == versions["pot_module_version"] == "0.9.6.post1",
            "unsupported POT version")
    require(all(hasattr(extension, name) for name in (
        "__cpu_baseline__", "__cpu_dispatch__", "__cpu_features__")), "CPU dispatch attributes missing")
    profile = {
        "schema_version": 1, "runtime_scope": TASK+"_environment_only",
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version(),
                   "byteorder": sys.byteorder, "executable_asset_id": observed["python"]["asset_id"]},
        "base_profile": normalized, "distribution_versions": versions,
        "implementation_assets": sorted(
            [{"asset_id": e.asset_id, "size": e.size, "sha256": e.sha256} for e in by_path.values()],
            key=lambda e: e["asset_id"]),
        "native_environment": environment,
        "numpy_numerics": {"error_policy": np.geterr(), "baseline": list(extension.__cpu_baseline__),
            "dispatch": list(extension.__cpu_dispatch__),
            "features": {k: bool(v) for k, v in extension.__cpu_features__.items()},
            "extension_asset_id": observed["numpy_cpu_extension"]["asset_id"]},
        "pot_interface": pot_interface(), "limitations": list(LIMITATIONS),
    }
    return {"profile": profile, "observed_paths": observed}


def double(value, context):
    require(not isinstance(value, (bool, np.bool_)), context+": Boolean scalar")
    if type(value) is float:
        result = value
    elif isinstance(value, (np.floating, np.ndarray)):
        require(np.asarray(value).shape == () and np.asarray(value).dtype == np.dtype("<f8"),
                context+": real FP64 scalar required")
        result = float(value)
    else:
        raise PilotError(context+": real FP64 scalar required")
    require(math.isfinite(result), context+": nonfinite scalar: "+(
        number_tag(result) if not math.isfinite(result) else "none"))
    return result


def same_double(a, b):
    return struct.pack("<d", a) == struct.pack("<d", b)


def number_tag(value):
    if math.isnan(value):
        return "nan"
    return "positive_infinity" if value > 0 else "negative_infinity"


@contextmanager
def numerical(state, request, stage, observation=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(divide="warn", invalid="warn", over="warn", under="ignore"):
            primary = None
            try:
                yield
            except BaseException as exc:
                primary = exc
                raise
            finally:
                records = [{"request_id": request["request_id"], "stage": stage,
                    "category": item.category.__module__+"."+item.category.__qualname__,
                    "message": str(item.message), "filename": item.filename, "lineno": item.lineno}
                    for item in caught]
                state.warnings.extend(records)
                if observation is not None:
                    observation["captured_warning_count"] = len(records)
                if records:
                    if primary is not None:
                        primary.add_note("Captured retrieval warnings: "+cpu.json_bytes(records).decode())
                    else:
                        raise PilotError(stage+": captured numerical warnings")


@dataclass(frozen=True)
class Prepared:
    preparation_id: str
    slide_index: int
    recipe: tuple
    array: np.ndarray | None
    rows: int
    reservation: object


def _layout(array):
    if array is None:
        return [None]*5
    return [array.dtype.str, json.dumps(list(array.shape), separators=(",", ":")),
            json.dumps(list(array.strides), separators=(",", ":")),
            bool(array.flags.c_contiguous), bool(array.flags.f_contiguous)]


def prepare(state, raw_slide, request):
    raw = raw_slide.raw
    index = raw_slide.slide.slide_index
    count = selected_count(state.bundle, request, index)
    n = len(raw)
    selected_bytes = n*1536*4
    prepared_bytes = count*(1536*4 if request["method"] == "ot_cosine" else 1535)
    token = object()
    state.memory.hold(token, selected_bytes + prepared_bytes)
    pid = f"p{len(state.preparations):010d}"
    values = [pid, index, raw_slide.slide.slide_id, request["method"], request["selection"],
              request["budget_percent"], request["query_fold"], n, count]
    values += _layout(raw) + _layout(None) + _layout(None)
    record = dict(zip(PREPARATION_HEADER, values))
    state.preparations.append(record)
    # Selection, norm/difference/finiteness buffers coexist during preparation.
    with state.memory.temporary(24*n*1536 + 64*n + 4096):
        with numerical(state, request, "preparation"):
            require(raw.dtype == np.dtype("<f4") and raw.shape == (n, 1536) and
                    np.isfinite(raw).all(), "invalid raw feature matrix")
            if request["selection"] == "literal_all_tissue":
                selected = raw
            elif request["selection"] == "all_true":
                selected = raw[np.ones(n, dtype=bool)]
            else:
                loaded = mask_for(state.bundle, request)
                s = raw_slide.slide
                selected = raw[loaded.backing.array_view()[s.tile_start:s.tile_stop]]
            require(len(selected) == count, "selected count/mask mismatch")
            record.update(dict(zip(PREPARATION_HEADER[14:19], _layout(selected))))
            prepared = None
            if count:
                if request["method"] == "ot_cosine":
                    payload = state.generator.process_single_slide(selected, raw_slide.slide.slide_id)
                    prepared = payload["features"]
                    del payload
                    require(prepared.dtype == np.dtype("<f4") and prepared.shape == (count, 1536)
                            and np.isfinite(prepared).all(), "OT prepared dtype/shape/finiteness")
                    norms = np.linalg.norm(prepared, axis=1)
                    require(np.isfinite(norms).all() and (norms > 0).all(),
                            "invalid prepared cosine norms")
                else:
                    prepared = bob.features_to_barcodes(selected)
                    require(prepared.dtype == np.dtype("|u1") and prepared.shape == (count, 1535)
                            and ((prepared == 0) | (prepared == 1)).all(), "barcode dtype/shape/binary")
            record.update(dict(zip(PREPARATION_HEADER[19:24], _layout(prepared))))
    event = Prepared(pid, index, recipe(request), prepared, count, token)
    state.prepared_events[pid] = event
    return event


def mask_for(bundle, request):
    matches = [m for m in bundle.masks if
               (m.view.held_out_fold, m.view.budget_percent) ==
               (request["query_fold"], request["budget_percent"])]
    require(len(matches) == 1, "missing/duplicate requested mask")
    return matches[0]


def selected_count(bundle, request, index):
    if request["selection"] != "filtered":
        return bundle.slides[index].feature_rows
    return mask_for(bundle, request).counts[index].retained_tiles


def validate_preparation(state, event, request, index):
    require(type(event) is Prepared and event.slide_index == index and event.recipe == recipe(request),
            "preparation association does not match request/view/slide")
    require(state.prepared_events.get(event.preparation_id) is event,
            "preparation event/actual array association changed")
    record = state.preparations[int(event.preparation_id[1:])]
    require(record["preparation_id"] == event.preparation_id and record["slide_index"] == index
            and (record["method"], record["selection"], record["budget_percent"],
                 record["query_fold"]) == recipe(request), "preparation event identity mismatch")


def _diagnostic_integer(value):
    if any(type(value) is scalar for scalar in _DIAGNOSTIC_INTEGER_TYPES):
        return int(value)
    return None


def _solver_log_fields(log):
    if type(log) is not dict or len(log) > len(_SOLVER_LOG_KEYS):
        return None
    fields = {}
    for name, value in log.items():
        if type(name) is not str:
            return None
        fields[name] = value
    return fields


def _returned_solver_note(error, request, reference_index, log):
    def text(value, limit):
        if type(value) is not str:
            return "unsupported text [detail omitted]"
        return value[:limit]+(" [detail omitted]" if len(value) > limit else "")

    def type_name(value):
        cls = type(value)
        try:
            module = type.__getattribute__(cls, "__module__")
            name = type.__getattribute__(cls, "__name__")
            if type(module) is not str or type(name) is not str:
                return "type metadata unavailable [detail omitted]"
            return text(module, 32)+"."+text(name, 64)
        except BaseException:
            return "type metadata unavailable [detail omitted]"

    def representation(value):
        kind = type(value)
        if value is None:
            detail = "None"
        elif kind is bool or kind is np.bool_:
            detail = "True" if bool(value) else "False"
        elif (number := _diagnostic_integer(value)) is not None:
            bits = number.bit_length()
            if bits > _DIAGNOSTIC_INTEGER_BITS:
                sign = "negative" if number < 0 else "positive"
                detail = f"integer detail omitted; sign={sign}; bit_length={bits}"
            else:
                detail = format(number, "d")
        elif any(kind is scalar for scalar in (float, np.float16, np.float32, np.float64)):
            detail = float(value).hex()
        elif kind is str:
            detail = text(value, 256)
        else:
            detail = "unsupported container/type [detail omitted]"
        return {"type": type_name(value), "representation": detail}

    def returned_type(name):
        if fields is not None and name in fields:
            return type_name(fields[name])
        return type_name(log)+" [returned field unavailable]"

    def returned_representation(name):
        if fields is not None and name in fields:
            return representation(fields[name])
        return {"type": type_name(log), "representation": "returned field unavailable [detail omitted]"}

    def add_bounded(note):
        encoder = json.JSONEncoder(ensure_ascii=True)

        def size(value):
            return len(encoder.encode(value).encode("utf8"))

        if size(note) > 4096:
            suffix = " [detail truncated to 4096-byte JSON-string ceiling]"
            low, high = 0, len(note)
            while low < high:
                middle = (low+high+1)//2
                if size(note[:middle]+suffix) <= 4096:
                    low = middle
                else:
                    high = middle-1
            note = note[:low]+suffix
        BaseException.add_note(error, note)

    context = {"request_id": "request field unavailable [detail omitted]",
               "result_code_type": "returned field unavailable [detail omitted]",
               "warning_type": "returned field unavailable [detail omitted]"}
    try:
        context["request_id"] = "request field unavailable: "+type_name(request)
        fields = _solver_log_fields(log)
        context["result_code_type"] = returned_type("result_code")
        context["warning_type"] = returned_type("warning")
        request_fields = None
        if type(request) is dict and len(request) <= 7:
            request_fields = {}
            for name, value in request.items():
                if type(name) is not str:
                    request_fields = None
                    break
                request_fields[name] = value
        if request_fields is not None and "request_id" in request_fields:
            context["request_id"] = text(request_fields["request_id"], 64)
        if fields is not None and "result_code" in fields:
            number = _diagnostic_integer(fields["result_code"])
            if number is not None and number.bit_length() > _DIAGNOSTIC_INTEGER_BITS:
                sign = "negative" if number < 0 else "positive"
                context["solver_result_code_export"] = (
                    f"not exported; numeric cell=null; sign={sign}; bit_length={number.bit_length()}; "
                    f"diagnostic domain={_DIAGNOSTIC_INTEGER_BITS} bits")
        require(request_fields is not None, "request diagnostic fields unavailable")
        evidence = {name: representation(request_fields[name]) for name in
                    ("query_index", "query_fold", "method", "selection", "budget_percent")}
        evidence.update(reference_index=representation(reference_index),
                        result_code=returned_representation("result_code"),
                        warning=returned_representation("warning"))
        note = ("Bounded returned OT interface evidence: "+
                cpu.json_bytes(context).decode("ascii").rstrip("\n")+
                "; details: "+cpu.json_bytes(evidence).decode("ascii").rstrip("\n"))
        add_bounded(note)
    except BaseException as additional:
        def ascii_context(value):
            value = text(value, 128)
            clean = "".join(c if 32 <= ord(c) <= 126 else "?" for c in value)
            return clean+(" [detail omitted]" if clean != value else "")

        note = ("Bounded returned OT interface evidence: formatting unavailable; "+
                "; ".join(name+"="+ascii_context(value) for name, value in context.items())+
                "; formatting_error_type="+ascii_context(type_name(additional)))
        # Printable ASCII needs only quote/backslash expansion and the outer quotes.
        require(2+len(note)+note.count('"')+note.count("\\") <= 4096, "fallback note byte ceiling")
        BaseException.add_note(error, note)


def compute_pair(state, request, query, reference, observation, *, scalar_control):
    qi, ri = request["query_index"], observation["reference_index"]
    validate_preparation(state, query, request, qi)
    validate_preparation(state, reference, request, ri)
    observation.update(query_preparation_id=query.preparation_id,
                       reference_preparation_id=reference.preparation_id)
    require(query.array is not None and reference.array is not None, "empty prepared pair")
    nq, nr = query.rows, reference.rows
    cells = nq*nr
    require(cells <= state.spec["limits"]["max_pair_cost_cells"], "pair cell safety cap")
    if request["method"] == "ot_cosine":
        reserve = 20*cells + 128*(nq+nr) + 4096
    else:
        block = min(nq, max(1, bob._MAX_BLOCK//nr))*nr
        reserve = 4*1535*(nq+nr) + 12*block + 4*nq + 4096
    warning_start = len(state.warnings)
    with state.memory.temporary(reserve):
        with numerical(state, request, "pair", observation):
            if request["method"] == "ot_cosine":
                pot_interface()
                a, b = np.ones(nq)/nq, np.ones(nr)/nr
                require(a.dtype == b.dtype == np.dtype("<f8"), "mass dtype")
                require(query.array.dtype == reference.array.dtype == np.dtype("<f4"),
                        "OT input dtype")
                for matrix in (query.array, reference.array):
                    norms = np.linalg.norm(matrix, axis=1)
                    require(np.isfinite(matrix).all() and np.isfinite(norms).all()
                            and (norms > 0).all(), "invalid prepared cosine norm")
                costs = ot.dist(query.array, reference.array, metric="cosine")
                require(type(costs) is np.ndarray and costs.shape == (nq, nr) and
                        costs.dtype == np.dtype("<f4"), "cosine cost dtype/shape")
                if not np.isfinite(costs).all():
                    observation["nonfinite_cost_tags"] = sorted({
                        number_tag(float(v)) for v in costs[~np.isfinite(costs)]})
                    raise PilotError("nonfinite cosine costs: "+",".join(observation["nonfinite_cost_tags"]))
                observation.update(cosine_min_hex=float(costs.min()).hex(),
                                   cosine_max_hex=float(costs.max()).hex(),
                                   cosine_negative_count=int(np.count_nonzero(costs < 0)))
                result = ot.emd2(a, b, costs, numItermax=1000000, numThreads=1,
                                 log=True, return_matrix=False)
                require(isinstance(result, (list, tuple)) and len(result) == 2, "OT log return length")
                distance, log = result
                try:
                    fields = _solver_log_fields(log)
                    require(fields is not None, "OT log: exact keys required")
                    keys(fields, _SOLVER_LOG_KEYS, "OT log")
                    code = fields["result_code"]
                    observation["solver_warning_is_none"] = fields["warning"] is None
                    observation["solver_warning"] = fields["warning"]
                    value = double(distance, "OT distance")
                    observation["distance_hex"] = value.hex()
                    number = _diagnostic_integer(code)
                    require(number is not None, "nonintegral/Boolean solver status")
                    observation["solver_result_code"] = (
                        number if number.bit_length() <= _DIAGNOSTIC_INTEGER_BITS else None)
                    for name, length in (("u", nq), ("v", nr)):
                        vector = fields[name]
                        require(type(vector) is np.ndarray and vector.shape == (length,) and
                                vector.dtype.kind == "f" and np.isfinite(vector).all(), "invalid dual vector")
                    require(number == 1 and fields["warning"] is None, "unsuccessful solver status/warning")
                except BaseException as exc:
                    _returned_solver_note(exc, request, ri, log)
                    raise
                del result, log, fields, costs
                if scalar_control:
                    scalar = double(state.generator.compute_ot_distance(query.array, reference.array),
                                    "scalar OT control")
                    observation["scalar_distance_hex"] = scalar.hex()
                    observation["scalar_log_comparison"] = (
                        "equal" if same_double(value, scalar) else "different")
                    state.compare(request, "scalar_log", value, scalar)
            else:
                value = double(bob.hamming_min_median(query.array, reference.array), "Yottixel distance")
                observation["distance_hex"] = value.hex()
    observation["captured_warning_count"] = len(state.warnings)-warning_start
    if value < 0 or (request["method"] == "ot_cosine" and observation["cosine_negative_count"] > 0):
        observation["observation_status"] = "held"
        raise PilotPromotionHeld("signed cost/distance anomaly")
    if observation.get("scalar_log_comparison") == "different":
        observation["observation_status"] = "held"
        raise PilotPromotionHeld("scalar/log control discrepancy")
    observation["computed_valid"] = True
    observation["observation_status"] = "valid_distance"
    return value


def _descriptor(value, context):
    require(type(value) is dict and set(value) >= {"path", "size", "sha256"},
            context+": descriptor required")
    integer(value["size"], context)
    cpu.digest(value["sha256"], context)
    require(type(value["path"]) is str and value["path"], context+": path required")
    return value


def baseline_controls(guard, spec, bundle):
    if spec["baseline_assets"] is None:
        return None
    declaration, expected = spec["baseline_assets"], spec["baseline_expected"]
    payloads = {name: guard.read_verified(declaration[name]["asset_id"])
                for name in ("review", "complete", "spec", "input_bindings", "metadata")}
    for name in ("review", "complete", "spec", "input_bindings"):
        require(binding(payloads[name])["sha256"] == expected[name+"_sha256"],
                "baseline expected "+name+" digest")
    review, complete, source_spec, inputs = [
        cpu.strict_json(payloads[name]) for name in ("review", "complete", "spec", "input_bindings")]
    require(type(review.get("schema_version")) is int and review["schema_version"] == 1
            and review.get("disposition") == "accepted"
            and review.get("scope") ==
            "Actual main saved-source evaluation and filter/perturbation preparation bundles only"
            and review.get("branches", {}).get("main", {}).get("disposition") == "accepted",
            "baseline review/main not accepted")
    historical_key = declaration["historical_review_complete_key"]
    require(type(historical_key) is str and
            review.get("artifacts", {}).get(historical_key) == expected["complete_sha256"],
            "baseline literal historical COMPLETE key")
    for record in (complete, source_spec, inputs):
        require(type(record.get("schema_version")) is int and record["schema_version"] == 1
                and record.get("task") == "hoptimus_main" and record.get("run_id") == expected["run_id"],
                "baseline schema/task/run mismatch")
    require(complete.get("scope") == "hoptimus_main_saved_rank_evaluation_only"
            and complete.get("spec_sha256") == inputs.get("spec_sha256") == expected["spec_sha256"],
            "baseline scope/spec mismatch")
    outputs = complete.get("outputs")
    keys(outputs, MAIN_OUTPUTS, "baseline successful registry")
    for name, value in outputs.items():
        keys(value, ("size", "sha256"), name)
        integer(value["size"], name)
        cpu.digest(value["sha256"], name)
    require(outputs["input_bindings.json"] == binding(payloads["input_bindings"]),
            "baseline input bindings not COMPLETE-bound")
    require(type(inputs.get("bindings")) is list, "baseline binding list")
    records = {}
    for record in inputs["bindings"]:
        require(type(record) is dict and "kind" in record and "name" in record,
                "baseline source binding shape")
        key = (record["kind"], record["name"])
        require(key not in records, "duplicate baseline binding")
        records[key] = record
    require(sum(row.get("kind") == "spec" for row in inputs["bindings"]) == 1,
            "baseline unique spec binding")
    original_spec = records.get(("spec", "spec"))
    require(original_spec is not None and original_spec["sha256"] == expected["spec_sha256"]
            and original_spec["size"] == len(payloads["spec"]), "baseline original spec binding")
    sources = source_spec.get("sources")
    require(type(sources) is dict, "baseline source descriptors")
    consumed = {}
    for name, current in [("metadata", declaration["metadata"])] + [
            ("matrix_"+m, value) for m, value in declaration["matrices"].items()]:
        original = _descriptor(sources.get(name), "baseline "+name)
        record = records.get(("source", name))
        require(record is not None and record.get("path") == original["path"]
                and record.get("size") == original["size"]
                and record.get("sha256") == original["sha256"], "baseline source record mismatch")
        require(current["size"] == original["size"] and current["sha256"] == original["sha256"],
                "baseline current alias content differs")
        consumed[name] = dict(original)
    raw = []
    header = ("Tumor.group", "Slide.Scan.File", "Sample.ID", "UID.Subject")
    for row in cpu.csv_rows(payloads["metadata"], header):
        require(all(type(v) is str and v for v in row.values()), "empty baseline metadata cell")
        raw.append(row)
    ids = [r["Slide.Scan.File"] for r in raw]
    require(len(ids) == len(set(ids)) == source_spec["expected"]["raw_slides"],
            "baseline unique complete raw population")
    matched = [(i, r) for i, r in enumerate(raw) if r["Tumor.group"] not in ("MISC", "NORMAL")]
    require(len(matched) == len(bundle.slides), "baseline matched population count")
    for slide, (index, row) in zip(bundle.slides, matched):
        require((slide.metadata_index, slide.slide_id, slide.patient_id, slide.diagnostic_group) ==
                (index, row["Slide.Scan.File"], row["UID.Subject"], row["Tumor.group"]),
                "baseline/mosaic ordered population join")
    referenced = [{"name": name, "descriptor": value} for name, value in sources.items()
                  if name not in consumed]
    referenced.extend({"name": name, "descriptor": value} for name, value in outputs.items()
                      if name != "input_bindings.json")
    return {"raw_ids": ids, "sources": consumed, "provenance": {
        **expected, "historical_review_complete_key": historical_key,
        "consumed_methods": [{"method": m, "asset_id": value["asset_id"],
            "current_path": value["path"], "producer_descriptor": consumed["matrix_"+m]}
            for m, value in declaration["matrices"].items()],
        "referenced_only": referenced,
    }}


def load_baseline(state, method):
    entry = state.spec["baseline_assets"]["matrices"][method]
    n = len(state.baseline["raw_ids"])
    token = ("baseline", method)
    state.memory.hold(token, entry["size"] + 24*n*n + 4096)
    data = state.guard.read_verified(entry["asset_id"])
    matrix = legacy._pickle(data, "trusted declared synthetic/local baseline matrix")
    del data
    require(type(matrix) is pd.DataFrame, "baseline must be a DataFrame")
    rows, cols = main_cli._axis(matrix.index, "baseline rows"), main_cli._axis(
        matrix.columns, "baseline columns")
    require(set(rows) == set(cols) == set(state.baseline["raw_ids"]) and
            matrix.shape == (n, n), "baseline complete raw axes")
    values = matrix.to_numpy(copy=False)
    require(values.dtype.kind == "f" and values.ndim == 2, "baseline floating values")
    return matrix, {s: i for i, s in enumerate(rows)}, {s: i for i, s in enumerate(cols)}, token


def validate_bundle(spec, bundle):
    pairs, rows, methods = validate_spec(spec)
    n = len(bundle.slides)
    require(n <= spec["limits"]["max_population_slides"], "population safety cap")
    for request in pairs + rows:
        qi = request["query_index"]
        require(qi < n and request["query_fold"] == bundle.slides[qi].slide_fold,
                "request query/fold does not match population")
        if "reference_index" in request:
            require(request["reference_index"] < n, "pair reference outside population")
    required_views = sorted({(r["query_fold"], r["budget_percent"])
                             for r in pairs + rows if r["selection"] == "filtered"})
    declared_views = [(m.view.held_out_fold, m.view.budget_percent) for m in bundle.masks]
    require(declared_views == required_views, "missing/extra/order mask declarations")
    feature_indices = sorted(range(n) if rows else {
        r[k] for r in pairs for k in ("query_index", "reference_index")})
    actual_indices = [r.slide_index for r in bundle.feature_records if r.consumer_asset is not None]
    require(actual_indices == feature_indices, "missing/extra feature declarations")
    for index in feature_indices:
        feature = bundle.feature_records[index]
        require(feature.rows <= spec["limits"]["max_raw_rows_per_slide"],
                "original row safety cap")
        require(feature.consumer_asset.asset_id == "feature:"+bundle.slides[index].slide_id,
                "feature ID/slide join")
    return {"mask_views": [list(v) for v in required_views], "feature_indices": feature_indices,
            "baseline_methods": list(methods)}


def _control_kinds(request, *, row):
    kinds = []
    if not row and request["method"] == "ot_cosine":
        kinds.append("scalar_log")
    if not row and request["selection"] == "all_true":
        kinds.append("literal_all_true")
    if request["selection"] != "filtered":
        kinds.append("baseline_distance")
        if row:
            kinds.append("baseline_rank")
    return kinds


class PilotState:
    def __init__(self, spec, memory):
        self.spec, self.memory = spec, memory
        self.guard = self.bundle = self.baseline = None
        self.generator = transport.OTSlideEmbeddingGenerator(
            metric="cosine", reg=0, feature_dim=1536, n_workers=1, verbose=False, max_iter=1000000)
        self.preparations, self.pair_probes, self.row_cells, self.ranks, self.warnings = [], [], {}, [], []
        self.controls, self.query_raw, self.query_prepared = {}, {}, {}
        self.prepared_events = weakref.WeakValueDictionary()
        self.decode_counts, self.consumptions = {}, {}
        self.current_request, self.stage, self.reason_codes = None, "declaration", []
        self.results = {r["request_id"]: {"status": "not_attempted", "reason": None,
                                        "controls_passed": False} for r in
                        spec["pair_requests"] + spec["row_requests"]}
        self.final = None
        self.before = self.after = self.expected_profile = None
        self.runtime_binding = None
        self.started = producer._time()
        for is_row, requests in ((False, spec["pair_requests"]), (True, spec["row_requests"])):
            for request in requests:
                for kind in _control_kinds(request, row=is_row):
                    self.controls[request["request_id"], kind] = {
                        "request_id": request["request_id"], "kind": kind,
                        "status": "unavailable_due_failure", "comparison_count": 0, "difference_count": 0}

    def compare(self, request, kind, a, b):
        record = self.controls[request["request_id"], kind]
        equal = same_double(a, b)
        record["comparison_count"] += 1
        record["difference_count"] += not equal
        record["status"] = "different" if record["difference_count"] else "equal"
        return equal

    def raw(self, index, *, query):
        if index in self.query_raw:
            return self.query_raw[index], None
        record = self.bundle.feature_records[index]
        token = ("query_raw", index) if query else object()
        # Encoded pickle, decoded features, finiteness and ordered-coordinate work.
        self.memory.hold(token, record.size + record.rows*(1536*5 + 96) + 4096)
        raw = _reader().load_raw_slide(self.guard, self.bundle, index)
        self.decode_counts[index] = self.decode_counts.get(index, 0) + 1
        self.consumptions.setdefault(index, []).append(asdict(raw.consumption))
        if query:
            self.query_raw[index] = raw
        return raw, None if query else token

    def query_event(self, request):
        index = request["query_index"]
        key = (index, recipe(request))
        if key not in self.query_prepared:
            raw, _ = self.raw(index, query=True)
            self.query_prepared[key] = prepare(self, raw, request)
        return self.query_prepared[key]

    def observation(self, request, ri):
        q, r = self.bundle.slides[request["query_index"]], self.bundle.slides[ri]
        return {"request_id": request["request_id"], "query_index": q.slide_index,
            "reference_index": ri, "reference_id": r.slide_id,
            "query_preparation_id": None, "reference_preparation_id": None,
            "same_patient_excluded": q.patient_id == r.patient_id,
            "zero_query": selected_count(self.bundle, request, q.slide_index) == 0,
            "zero_reference": selected_count(self.bundle, request, ri) == 0,
            "observation_status": "structural_skip", "computed_valid": False,
            "distance_hex": None, "cosine_min_hex": None, "cosine_max_hex": None,
            "cosine_negative_count": None, "solver_result_code": None,
            "solver_warning_is_none": None, "captured_warning_count": 0,
            "scalar_distance_hex": None, "scalar_log_comparison": "not_applicable",
            "baseline_distance_hex": None, "baseline_comparison": "not_applicable",
            "literal_comparison": "not_applicable", "candidate_rank": None,
            "baseline_rank": None, "baseline_distance_comparison": "not_applicable",
            "baseline_rank_comparison": "not_applicable"}


def _baseline_value(matrix_info, state, request, ri):
    matrix, rows, cols, _ = matrix_info
    qid = state.bundle.slides[request["query_index"]].slide_id
    rid = state.bundle.slides[ri].slide_id
    value = matrix.iat[rows[qid], cols[rid]]
    require(isinstance(value, (float, np.floating)) and math.isfinite(float(value)),
            "nonfinite/nonreal eligible baseline value")
    return float(value)


def _skip(observation):
    return observation["same_patient_excluded"] or observation["zero_query"] or observation["zero_reference"]


def explicit_pairs(state):
    literal = {}
    for request in state.spec["pair_requests"]:
        state.current_request, state.stage = request, "pair_probe"
        observation = state.observation(request, request["reference_index"])
        state.pair_probes.append(observation)
        result = state.results[request["request_id"]]
        result["status"] = "attempted"
        if _skip(observation):
            result.update(status="structural_skip", controls_passed=True)
            for kind in _control_kinds(request, row=False):
                state.controls[request["request_id"], kind]["status"] = "not_applicable"
            if request["selection"] == "literal_all_tissue":
                literal[request["method"], request["query_index"], request["reference_index"]] = observation
            continue
        ri = request["reference_index"]
        raw = reference = None
        raw_token, reference_owned = None, False
        try:
            query = state.query_event(request)
            observation["query_preparation_id"] = query.preparation_id
            raw, raw_token = state.raw(ri, query=ri in state.spec["query_indices"])
            ref_key = (ri, recipe(request))
            reference = state.query_prepared.get(ref_key) if ri in state.query_raw else None
            reference_owned = reference is None
            if reference is None:
                reference = prepare(state, raw, request)
                if ri in state.query_raw:
                    state.query_prepared[ref_key] = reference
                    reference_owned = False
            observation["reference_preparation_id"] = reference.preparation_id
            value = compute_pair(state, request, query, reference, observation, scalar_control=True)
            if request["selection"] != "filtered":
                info = load_baseline(state, request["method"])
                try:
                    previous = _baseline_value(info, state, request, ri)
                    observation["baseline_distance_hex"] = previous.hex()
                    equal = state.compare(request, "baseline_distance", value, previous)
                    observation["baseline_comparison"] = "equal" if equal else "different"
                    if not equal:
                        raise PilotPromotionHeld("literal/all-true baseline distance discrepancy")
                finally:
                    state.memory.release(info[3])
                    info = None
            key = (request["method"], request["query_index"], ri)
            if request["selection"] == "literal_all_tissue":
                literal[key] = observation
            if request["selection"] == "all_true":
                previous = literal[key]
                require(previous["computed_valid"], "literal partner has no valid distance")
                equal = state.compare(request, "literal_all_true", value,
                                      float.fromhex(previous["distance_hex"]))
                observation["literal_comparison"] = "equal" if equal else "different"
                if not equal:
                    raise PilotPromotionHeld("all-true/literal discrepancy")
            result.update(status="valid_distance", controls_passed=True)
        except BaseException as exc:
            observation["computed_valid"] = False
            observation["observation_status"] = "held" if isinstance(exc, PilotPromotionHeld) else "failed"
            result.update(status=observation["observation_status"], reason=str(exc))
            raise
        finally:
            if reference_owned and reference is not None:
                state.memory.release(reference.reservation)
            if raw_token is not None:
                state.memory.release(raw_token)
            raw = reference = None


def complete_rows(state):
    for request in state.spec["row_requests"]:
        state.row_cells[request["request_id"]] = []
    for method in METHODS:
        requests = [r for r in state.spec["row_requests"] if r["method"] == method]
        if not requests:
            continue
        info = load_baseline(state, method)
        try:
            for ri in range(len(state.bundle.slides)):
                raw, raw_token, reference_events = None, None, {}
                try:
                    for request in requests:
                        state.current_request, state.stage = request, "row_pair"
                        cell = state.observation(request, ri)
                        state.row_cells[request["request_id"]].append(cell)
                        state.results[request["request_id"]]["status"] = "attempted"
                        if _skip(cell):
                            continue
                        try:
                            query = state.query_event(request)
                            cell["query_preparation_id"] = query.preparation_id
                            if raw is None:
                                raw, raw_token = state.raw(ri, query=ri in state.spec["query_indices"])
                            key = (ri, recipe(request))
                            reference = state.query_prepared.get(key)
                            if reference is None:
                                reference = reference_events.get(key)
                            if reference is None:
                                reference = prepare(state, raw, request)
                                if ri in state.query_raw:
                                    state.query_prepared[key] = reference
                                else:
                                    reference_events[key] = reference
                            cell["reference_preparation_id"] = reference.preparation_id
                            value = compute_pair(state, request, query, reference, cell, scalar_control=False)
                            if request["selection"] == "literal_all_tissue":
                                previous = _baseline_value(info, state, request, ri)
                                cell["baseline_distance_hex"] = previous.hex()
                                equal = state.compare(request, "baseline_distance", value, previous)
                                cell["baseline_distance_comparison"] = "equal" if equal else "different"
                                if not equal:
                                    raise PilotPromotionHeld("full-row baseline distance discrepancy")
                        except BaseException as exc:
                            cell["computed_valid"] = False
                            cell["observation_status"] = (
                                "held" if isinstance(exc, PilotPromotionHeld) else "failed")
                            state.results[request["request_id"]].update(
                                status=cell["observation_status"], reason=str(exc))
                            raise
                finally:
                    reference_tokens = [item.reservation for item in reference_events.values()]
                    reference_events.clear()
                    raw = reference = None
                    for token in reference_tokens:
                        state.memory.release(token)
                    if raw_token is not None:
                        state.memory.release(raw_token)
            # All literal controls are accepted before conditional diagnostics.
            for request in sorted(requests, key=lambda r: r["selection"] != "literal_all_tissue"):
                state.current_request, state.stage = request, "row_controls_and_ranking"
                finish_row(state, request, info)
        finally:
            state.memory.release(info[3])
            info = None


def finish_row(state, request, matrix_info):
    cells = state.row_cells[request["request_id"]]
    require(len(cells) == len(state.bundle.slides) and
            [c["reference_index"] for c in cells] == list(range(len(cells))), "row coverage incomplete")
    for cell in cells:
        require(cell["computed_valid"] == (not _skip(cell)), "invalid/held/missing eligible pair")
    eligible = [c for c in cells if c["computed_valid"]]
    ordered = sorted(eligible, key=lambda c: float.fromhex(c["distance_hex"]))
    for rank, cell in enumerate(ordered, 1):
        cell["candidate_rank"] = rank
    ids = [state.bundle.slides[c["reference_index"]].slide_id for c in ordered]
    baseline_hash = None
    if request["selection"] == "literal_all_tissue":
        historical = sorted(eligible, key=lambda c: _baseline_value(
            matrix_info, state, request, c["reference_index"]))
        baseline_ids = [state.bundle.slides[c["reference_index"]].slide_id for c in historical]
        baseline_hash = binding(cpu.json_bytes(baseline_ids))["sha256"]
        ranks = {c["reference_index"]: rank for rank, c in enumerate(historical, 1)}
        control = state.controls[request["request_id"], "baseline_rank"]
        for cell in eligible:
            cell["baseline_rank"] = ranks[cell["reference_index"]]
            same = cell["baseline_rank"] == cell["candidate_rank"]
            cell["baseline_rank_comparison"] = "equal" if same else "different"
            control["comparison_count"] += 1
            control["difference_count"] += not same
        control["status"] = "different" if ids != baseline_ids else "equal"
        if not eligible:
            state.controls[request["request_id"], "baseline_distance"]["status"] = "not_applicable"
        if ids != baseline_ids:
            raise PilotPromotionHeld("full eligible baseline ranking discrepancy")
    else:
        partner = next(r for r in state.spec["row_requests"] if r["method"] == request["method"]
                       and r["query_index"] == request["query_index"]
                       and r["selection"] == "literal_all_tissue")
        require(state.results[partner["request_id"]]["controls_passed"], "literal row not accepted")
    zero_query = selected_count(state.bundle, request, request["query_index"]) == 0
    status = "empty_query" if zero_query else "returned" if eligible else "empty_reference_library"
    query = state.bundle.slides[request["query_index"]]
    relevance = [state.bundle.slides[c["reference_index"]].diagnostic_group ==
                 query.diagnostic_group for c in ordered]
    metrics = [{"cutoff": k, **asdict(kernel.query_metrics(relevance, k, status))} for k in (5, 10)]
    boundaries = [boundary(state, ordered, relevance, k) for k in (5, 10)]
    state.results[request["request_id"]].update(
        status=status, reason=None, controls_passed=True, metrics=metrics, boundaries=boundaries,
        candidate_ranking_sha256=binding(cpu.json_bytes(ids))["sha256"],
        baseline_ranking_sha256=baseline_hash)
    for rank, cell in enumerate(ordered[:10], 1):
        ref = state.bundle.slides[cell["reference_index"]]
        state.ranks.append({"request_id": request["request_id"], "rank": rank,
            "reference_index": ref.slide_index, "reference_id": ref.slide_id,
            "reference_group": ref.diagnostic_group, "relevant": relevance[rank-1],
            "distance_hex": cell["distance_hex"]})


def boundary(state, ordered, relevance, cutoff):
    record = {"cutoff": cutoff, "status": "no_kth_boundary", "distance_hex": None,
        "strict_count": None, "equal_count": None, "strict_relevant_count": None,
        "equal_relevant_count": None, "slots": None, "selected_ids": []}
    if len(ordered) < cutoff:
        return record
    threshold = float.fromhex(ordered[cutoff-1]["distance_hex"])
    strict = [i for i, c in enumerate(ordered) if float.fromhex(c["distance_hex"]) < threshold]
    equal = [i for i, c in enumerate(ordered) if float.fromhex(c["distance_hex"]) == threshold]
    record.update(status="boundary", distance_hex=threshold.hex(), strict_count=len(strict),
        equal_count=len(equal), strict_relevant_count=sum(relevance[i] for i in strict),
        equal_relevant_count=sum(relevance[i] for i in equal), slots=cutoff-len(strict),
        selected_ids=[state.bundle.slides[ordered[i]["reference_index"]].slide_id
                      for i in equal if i < cutoff])
    return record


def csv_cell(value, *, numeric=False):
    if value is None:
        return ""
    if numeric:
        require(type(value) is str, "FP64 column must be hex string/null")
        decoded = float.fromhex(value)
        require(math.isfinite(decoded) and decoded.hex() == value, "noncanonical FP64 hex")
        return value
    if type(value) is bool:
        return "1" if value else "0"
    require(type(value) in (str, int), "CSV field must be string/integer/Boolean/null")
    return str(value)


def json_output(output, name, value, bindings, memory):
    with memory.temporary(16*MIB):
        encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        size = 1
        for chunk in encoder.iterencode(value):
            size += len(chunk.encode("ascii"))
            require(size <= 8*MIB, name+": JSON ceiling")
        data = cpu.json_bytes(value)
        original = binding(data)
        output.write(name, lambda handle: handle.write(data))
        actual, _ = output.readback(name, 8*MIB, lambda handle: require(
            cpu.strict_json(handle.read()) == value, name+": JSON semantic readback"),
            expected=original)
        bindings[name] = actual


def csv_output(output, name, header, records, bindings, memory, maximum):
    require(len(records) <= maximum, name+": row ceiling")
    digest, total = hashlib.sha256(), 0

    def encoded(row, *, is_header=False):
        require(set(row) >= set(header), name+": record fields")
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerow([
            csv_cell(row[k], numeric=not is_header and k in HEX_COLUMNS) for k in header])
        data = buffer.getvalue().encode("utf-8")
        require(len(data) <= 8448, name+": record ceiling")
        return data

    def emit(handle):
        nonlocal total
        for index, row in enumerate([dict(zip(header, header))] + records):
            data = encoded(row, is_header=index == 0)
            total += len(data)
            require(total <= 64*MIB, name+": CSV ceiling")
            digest.update(data)
            handle.write(data)

    with memory.temporary(32768):
        output.write(name, emit)
        original = {"size": total, "sha256": digest.hexdigest()}

        def check(handle):
            parsed = producer.csv_rows(handle, header, name, compressed=False,
                expanded_limit=64*MIB, row_limit=maximum, record_limit=8448)
            count = 0
            for actual, expected in zip(parsed, records, strict=True):
                require(tuple(actual.values()) == tuple(
                    csv_cell(expected[k], numeric=k in HEX_COLUMNS) for k in header),
                        name+": semantic/FP64 record mismatch")
                count += 1
            require(count == len(records) and handle.read(1) == b"", name+": incomplete/trailing CSV")

        actual, _ = output.readback(name, 64*MIB, check, expected=original)
        bindings[name] = actual


def _counts(state, planned):
    n = len(state.bundle.slides) if state.bundle is not None else state.spec["mosaic_expected"]["counts"]["slides"]
    if planned:
        return {"pair_requests": len(state.spec["pair_requests"]), "row_requests": len(state.spec["row_requests"]),
                "reference_cells": n*len(state.spec["row_requests"]),
                "preparations": 2*len(state.spec["pair_requests"])+(n+1)*len(state.spec["row_requests"])}
    return {"pair_requests": len(state.pair_probes), "row_requests": sum(
        state.results[r["request_id"]]["status"] != "not_attempted" for r in state.spec["row_requests"]),
        "reference_cells": sum(map(len, state.row_cells.values())), "preparations": len(state.preparations)}


def prefix(state, digest):
    return {"schema_version": 1, "task": TASK, "run_id": state.spec["run_id"], "spec_sha256": digest}


def summary(state, digest, disposition):
    pairs = [{"request_id": r["request_id"],
              **{k: state.results[r["request_id"]][k] for k in ("status", "reason", "controls_passed")}}
             for r in state.spec["pair_requests"]]
    rows = []
    for r in state.spec["row_requests"]:
        result = state.results[r["request_id"]]
        cells = state.row_cells.get(r["request_id"], [])
        successful = result["status"] in ("returned", "empty_query", "empty_reference_library")
        rows.append({"request_id": r["request_id"], "status": result["status"], "reason": result["reason"],
            "reference_cells": len(cells),
            "patient_excluded_count": sum(c["same_patient_excluded"] for c in cells),
            "zero_reference_count": sum(c["zero_reference"] for c in cells),
            "computed_valid_count": sum(c["computed_valid"] for c in cells),
            "candidate_ranking_sha256": result.get("candidate_ranking_sha256") if successful else None,
            "baseline_ranking_sha256": result.get("baseline_ranking_sha256") if successful else None,
            "controls_passed": result["controls_passed"],
            "metrics": result.get("metrics", []) if successful else [],
            "boundaries": result.get("boundaries", []) if successful else []})
    controls = []
    for r in state.spec["pair_requests"] + state.spec["row_requests"]:
        for kind in ("scalar_log", "literal_all_true", "baseline_distance", "baseline_rank"):
            if (r["request_id"], kind) in state.controls:
                controls.append(state.controls[r["request_id"], kind])
    return {**prefix(state, digest), "disposition": disposition,
        "planned_counts": _counts(state, True), "completed_counts": _counts(state, False),
        "pair_results": pairs, "row_results": rows, "control_results": controls,
        "promotion": {"eligible": disposition == "success", "reason_codes": state.reason_codes},
        "working_memory": state.memory.summary(), "limitations": list(LIMITATIONS)}


def requests_record(state, digest, derived):
    return {**prefix(state, digest), "population": [asdict(s) for s in state.bundle.slides],
        "query_indices": state.spec["query_indices"], "pair_requests": state.spec["pair_requests"],
        "row_requests": state.spec["row_requests"], "derived_declarations": derived,
        "limits": state.spec["limits"]}


def runtime_record(state, digest):
    return {**prefix(state, digest), "expected_profile_sha256": state.runtime_binding["sha256"],
        "expected_profile": state.expected_profile, "before": state.before, "after": state.after,
        "comparison": {"before_expected": state.before["profile"] == state.expected_profile,
                       "after_expected": state.after["profile"] == state.expected_profile,
                       "before_after": state.before["profile"] == state.after["profile"]},
        "limitations": list(LIMITATIONS)}


def _consumption_json(record):
    return {"asset_id": record.asset_id, "logical_name": record.logical_name,
        "consumer_path": record.consumer_path, "producer_size": record.producer_size,
        "producer_sha256": record.producer_sha256, "semantic_checks": list(record.semantic_checks)}


def inputs_record(state, digest, paths):
    bundle = state.bundle
    features = []
    for record in bundle.feature_records:
        count = state.decode_counts.get(record.slide_index, 0)
        declared = record.consumer_asset is not None
        features.append({"slide_index": record.slide_index, "slide_id": record.slide_id,
            "asset_id": record.consumer_asset.asset_id if declared else None,
            "status": "decoded" if count else "hashed_only" if declared else "referenced_only",
            "decode_count": count, "coordinate_match": True if count else None})
    provenance = {"review_sha256": bundle.expected.review_sha256,
        "complete_sha256": bundle.expected.complete_sha256, "spec_sha256": bundle.expected.spec_sha256,
        "consumed_controls": [_consumption_json(v) for v in bundle.consumed_controls],
        "consumed_arrays": [_consumption_json(v) for v in bundle.consumed_arrays],
        "referenced_only": [asdict(v) for v in bundle.referenced_only], "limitations": list(LIMITATIONS)}
    return {**prefix(state, digest), "bindings": list(state.final),
        "executing_code_paths": {role: str(path.resolve(strict=True)) for role, path in paths.items()},
        "mosaic_provenance": provenance,
        "baseline_provenance": state.baseline["provenance"] if state.baseline else None,
        "feature_consumption": features,
        "runtime_implementation_bindings": [v for v in state.final if v["role"] == "runtime_implementation"],
        "started_at": state.started, "captured_at": producer._time(), "limitations": list(LIMITATIONS)}


def write_results(output, state, digest, derived, paths, bindings, disposition):
    n = len(state.bundle.slides)
    # Reserve the fixed writer upper envelope before recording the peak.
    with state.memory.temporary(16*MIB + 32768):
        pass
    records = {
        "preparations.csv": (PREPARATION_HEADER, state.preparations, _counts(state, True)["preparations"]),
        "pair_probes.csv": (PAIR_HEADER, state.pair_probes, len(state.spec["pair_requests"])),
        "row_cells.csv": (ROW_HEADER, [c for r in state.spec["row_requests"]
                         for c in state.row_cells.get(r["request_id"], [])], n*len(state.spec["row_requests"])),
        "rank_probes.csv": (RANK_HEADER, sorted(state.ranks, key=lambda v: (
            next(i for i, r in enumerate(state.spec["row_requests"]) if r["request_id"] == v["request_id"]),
            v["rank"])), 10*len(state.spec["row_requests"])),
    }
    for name, (header, values, ceiling) in records.items():
        if name not in output.files:
            csv_output(output, name, header, values, bindings, state.memory, ceiling)
    jsons = {"requests.json": requests_record(state, digest, derived),
             "summary.json": summary(state, digest, disposition)}
    if state.before is not None and state.after is not None:
        jsons["runtime.json"] = runtime_record(state, digest)
    if state.final is not None:
        jsons["input_bindings.json"] = inputs_record(state, digest, paths)
    for name, value in jsons.items():
        if name not in output.files:
            json_output(output, name, value, bindings, state.memory)


def run(spec_path: str | Path, spec_sha256: str, out: str | Path, out_root: str | Path) -> dict:
    cpu.digest(spec_sha256, "spec SHA256")
    spec_path = Path(spec_path).absolute()
    data, initial = cpu._scan_file(spec_path, "spec", "spec", 8*MIB)
    require(initial["sha256"] == spec_sha256, "pilot spec digest mismatch")
    spec = cpu.strict_json(data)
    validate_spec(spec)
    reader, paths = _reader(), code_paths()
    declared, policies, bundle_assets, expected = declarations(spec, spec_path, data, paths, reader)
    memory = WorkingMemory(spec["limits"]["max_accounted_working_bytes"])
    memory.hold("pilot_spec", len(data)*2 + 65536)
    state = PilotState(spec, memory)
    output = producer._Output(out, out_root, [Path(e.path) for e in declared],
                              [Path(v["asset"]["path"]).parent for v in spec["mosaic_bundle_assets"]["features"]])
    primary, bindings, derived = None, {}, None
    try:
        with assets.AssetGuard(declared, base_dir=spec_path.parent, policies=policies) as guard:
            state.guard = guard
            require(guard.read_verified("pilot:spec") == data, "pilot spec changed on acquisition")
            observed = spec_path.stat()
            require((observed.st_dev, observed.st_ino, observed.st_size) ==
                    (initial["device"], initial["inode"], initial["size"]), "pilot spec acquisition changed")
            for role in paths:
                with guard.lease_verified_path("code:"+role):
                    pass
            # Fixed runtime objects determine membership before implementation hashing.
            state.before = runtime_snapshot(spec["runtime_assets"]["implementations"])
            for item in spec["runtime_assets"]["implementations"]:
                with guard.lease_verified_path(item["asset_id"]):
                    pass
            memory.hold("runtime_profile", spec["runtime_assets"]["expected_profile"]["size"]*2 + 65536)
            profile_data = guard.read_verified("runtime:expected")
            state.runtime_binding = binding(profile_data)
            state.expected_profile = cpu.strict_json(profile_data)
            require(state.before["profile"] == state.expected_profile, "runtime before/expected differs")
            controls = spec["mosaic_bundle_assets"]
            bundle_bytes = sum(controls[k]["size"] for k in MOSAIC_CONTROLS)
            bundle_bytes += sum(v["asset"]["size"] for v in controls["masks"])
            # Coordinate predicates and per-slide ordered-coordinate checks.
            bundle_bytes += 2*expected.counts.original_tiles + 96*expected.counts.slides
            memory.hold("bundle", bundle_bytes)
            state.bundle = reader.load_mosaic_bundle(guard, bundle_assets=bundle_assets, expected=expected)
            derived = validate_bundle(spec, state.bundle)
            baseline_bytes = (sum(spec["baseline_assets"][k]["size"] for k in
                ("review", "complete", "spec", "input_bindings", "metadata"))
                if spec["baseline_assets"] else 0)
            with memory.temporary(baseline_bytes*2 + 4096):
                state.baseline = baseline_controls(guard, spec, state.bundle)
            memory.hold("results", _counts(state, True)["reference_cells"]*256
                        + _counts(state, True)["preparations"]*256 + 4096)
            output.claim()
            try:
                explicit_pairs(state)
                complete_rows(state)
                state.stage = "runtime_after"
                state.after = runtime_snapshot(spec["runtime_assets"]["implementations"])
                require(state.after["profile"] == state.expected_profile == state.before["profile"],
                        "runtime after/profile drift")
                state.stage = "final_input_sweep"
                state.final = guard.final_bindings()
            except BaseException as exc:
                if isinstance(exc, PilotPromotionHeld) and guard.state == "active":
                    try:
                        state.after = runtime_snapshot(spec["runtime_assets"]["implementations"])
                        state.final = guard.final_bindings()
                    except BaseException as additional:
                        producer._note(exc, additional, "held diagnostic final custody")
                raise
        state.stage = "output_readback"
        write_results(output, state, spec_sha256, derived, paths, bindings, "success")
        require(set(bindings) == set(ARTIFACTS), "exact eight-artifact registry")
        output.check(bindings)
        manifest_sha = bindings["requests.json"]["sha256"]
        marker = {**prefix(state, spec_sha256), "scope": SCOPE, "request_manifest_sha256": manifest_sha,
                  "disposition": "success", "outputs": dict(bindings),
                  "verified_at": producer._time(), "limitations": list(LIMITATIONS)}
        json_output(output, "COMPLETE.json", marker, bindings, memory)
        output.check(bindings, marker=True)
        state.stage = "directory_release"
        output.close()
        return marker
    except BaseException as exc:
        primary = exc
        disposition = "held" if isinstance(exc, PilotPromotionHeld) else "failed"
        state.reason_codes.append(str(exc))
        if state.current_request is not None:
            result = state.results[state.current_request["request_id"]]
            result.update(status=disposition, reason=str(exc), controls_passed=False)
        for result in state.results.values():
            if result["status"] == "attempted":
                result.update(status=disposition, reason="incomplete after "+str(exc), controls_passed=False)
        secondary = []
        if output.owned:
            if state.bundle is not None and derived is not None:
                try:
                    write_results(output, state, spec_sha256, derived, paths, bindings, disposition)
                except BaseException as additional:
                    producer._note(exc, additional, "partial pilot artifacts")
                    secondary.append(type(additional).__name__+": "+str(additional))
            try:
                output.fail(exc, [])
            except BaseException as additional:
                producer._note(exc, additional, "pilot FAILED/owned marker cleanup")
                secondary.append(type(additional).__name__+": "+str(additional))
            diagnostic = {**prefix(state, spec_sha256), "disposition": disposition,
                "failed_request_id": state.current_request["request_id"] if state.current_request else None,
                "stage": state.stage, "reason_codes": state.reason_codes, "warnings": state.warnings,
                "error": {"type": type(exc).__name__, "message": str(exc),
                          "notes": list(getattr(exc, "__notes__", ()))},
                "planned_counts": _counts(state, True), "attempted_counts": _counts(state, False),
                "custody": {"final_sweep_completed": state.final is not None,
                            "output_custody_verified": False, "secondary_errors": secondary},
                "retained_artifacts": [{"name": name, **value} for name, value in bindings.items()
                                       if name != "COMPLETE.json"]}
            try:
                json_output(output, "DIAGNOSTIC.json", diagnostic, bindings, memory)
            except BaseException as additional:
                producer._note(exc, additional, "pilot diagnostic persistence")
        raise
    finally:
        output.close(primary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("spec", "spec-sha256", "out", "out-root"):
        parser.add_argument("--"+option, required=True)
    args = parser.parse_args(argv)
    marker = run(args.spec, args.spec_sha256, args.out, args.out_root)
    print(cpu.json_bytes(marker).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
