"""One bound annotation head; trusted local inputs and cooperative custody only."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import errno
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import socket
import stat
import sys
import warnings
import zlib

import numpy as np
import prosper
import prosper.evaluation

from . import feature_bank_audit as legacy
from . import r2_1_hoptimus_asset_io as assets
from . import r2_1_hoptimus_filter as kernel
from . import r2_1_hoptimus_filter_cli as manifests
from . import r2_1_hoptimus_fit as fitter
from . import r2_1_hoptimus_io as cpu


TASK = "hoptimus_annotation_head"
SCOPE = "annotation_patient_held_out_head_only"
PRIMITIVE_CONTRACT = "7954b19445d1587b3604e277c302b1b09b60fe5a24d1253c05ab1846551ef663"
MIB = 1024 * 1024
SOURCE_LIMITS = {
    "cpu_review": 8 * MIB, "cpu_complete": 8 * MIB,
    "cpu_input_bindings": 8 * MIB, "patient_folds": 4 * MIB,
    "annotation_assignments": 32 * MIB, "training_rows": 128 * MIB,
    "head_plan": 8 * MIB, "cpu_summary": 8 * MIB,
    "feature_inventory": 2 * MIB, "asset_fit_review": 8 * MIB,
    "primitive_integration_review": 8 * MIB,
}
EXPECTED = {
    "annotation_tiles", "annotation_slides", "annotation_patients", "union_patients",
    "feature_files", "training_tiles", "training_slides", "training_patients",
    "held_out_tiles", "held_out_slides", "held_out_patients",
}
ARTIFACTS = {
    "head.pkl", "fit_diagnostics.json", "held_out_rows.csv.gz", "probabilities.npy",
    "predictions.npy", "counts.json", "summary.json", "input_bindings.json",
}
CONSUMED = {
    "cpu_input_bindings": "input_bindings.json", "patient_folds": "patient_folds.csv",
    "annotation_assignments": "annotation_assignments.csv.gz",
    "training_rows": "training_rows.csv.gz", "head_plan": "head_plan.json",
    "cpu_summary": "summary.json",
}
HELD_HEADER = ("prediction_index",) + manifests.ANNOTATION_HEADER
LIMITATIONS = [
    "conditional_on_prescribed_head_and_existing_annotation_bank",
    "sparse_tissue_patient_support_not_extra_independence",
    "annotation_to_mosaic_transfer_not_assessed",
    "filtering_retrieval_not_performed",
    "historical_encoder_identity_not_established",
    "no_author_scientific_or_r2_1_closure",
    "trusted_local_pickle_not_untrusted_model_ingestion",
    "cooperative_inputs_not_atomic_snapshot",
]
NATIVE_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


class HeadTaskError(ValueError):
    """A declaration, manifest, payload, output or fidelity check failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HeadTaskError(message)


def _text(value, name):
    require(type(value) is str and bool(value.strip()) and "\x00" not in value,
            f"{name}: nonblank exact string required")
    return value


def _int(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name}: exact integer required")
    return value


def _decimal(value, name):
    require(type(value) is str and re.fullmatch(r"0|[1-9][0-9]*", value) is not None,
            f"{name}: canonical nonnegative integer text required")
    return int(value)


def _digest(value, name):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{name}: lowercase SHA256 required")
    return value


def _float(value, name):
    require(type(value) is float and math.isfinite(value), f"{name}: finite float required")
    return value


def _same(actual, expected, name):
    require(cpu.json_bytes(actual) == cpu.json_bytes(expected), f"{name}: mismatch")


def _json(data, name):
    value = cpu.strict_json(data)
    require(type(value) is dict, f"{name}: JSON object required")
    return value


def _version(value, name):
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1,
            f"{name}: integer schema_version 1 required")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _time():
    return datetime.now(timezone.utc).isoformat()


def code_paths():
    return {
        "runner": Path(__file__), "fit": Path(fitter.__file__),
        "asset_io": Path(assets.__file__), "filter_kernel": Path(kernel.__file__),
        "filter_cli": Path(manifests.__file__), "cpu_io": Path(cpu.__file__),
        "legacy_io": Path(legacy.__file__), "package_init": Path(prosper.__file__),
        "evaluation_init": Path(prosper.evaluation.__file__),
    }


def _binding(entry, name, limit):
    cpu.keys(entry, {"path", "sha256", "size"}, name)
    _text(entry["path"], f"{name} path")
    _digest(entry["sha256"], name)
    _int(entry["size"], f"{name} size")
    require(entry["size"] <= limit, f"{name}: encoded limit exceeded")


def _schema(spec, base):
    cpu.keys(spec, {"schema_version", "run_id", "task", "sources", "code_files",
                    "features", "parameters", "expected"}, "spec")
    _version(spec, "spec")
    _text(spec["run_id"], "run_id")
    require(spec["task"] == TASK, "wrong task")
    cpu.keys(spec["sources"], SOURCE_LIMITS, "source roles")
    cpu.keys(spec["code_files"], code_paths(), "code roles")
    for name, entry in spec["sources"].items():
        _binding(entry, name, SOURCE_LIMITS[name])
    for name, entry in spec["code_files"].items():
        _binding(entry, name, 2 * MIB)
        path = Path(entry["path"])
        if not path.is_absolute():
            path = base / path
        require(path.resolve(strict=True) == code_paths()[name].resolve(strict=True),
                f"{name}: declaration is not the actual executing file")
    cpu.keys(spec["parameters"], {"held_out_fold", "budget_percent",
                                 "prediction_batch_rows"}, "parameters")
    p = spec["parameters"]
    require(_int(p["held_out_fold"], "held_out_fold") < 5, "fold must be 0..4")
    require(_int(p["budget_percent"], "budget_percent") in (5, 100), "budget must be 5/100")
    require(_int(p["prediction_batch_rows"], "batch rows") == 4096, "batch rows must be 4096")
    cpu.keys(spec["expected"], EXPECTED, "expected")
    for name, value in spec["expected"].items():
        _int(value, name, 1)
    require(type(spec["features"]) is list and
            len(spec["features"]) == spec["expected"]["feature_files"],
            "feature declaration count mismatch")
    slides, resolved, identities = set(), set(), set()
    for entry in spec["features"]:
        cpu.keys(entry, {"slide_id", "path", "sha256", "size", "feature_rows",
                         "feature_dim", "dtype"}, "feature declaration")
        slide = _text(entry["slide_id"], "feature slide")
        require(slide not in slides, "duplicate feature slide")
        slides.add(slide)
        _binding({key: entry[key] for key in ("path", "sha256", "size")}, slide, 192 * MIB)
        _int(entry["feature_rows"], "feature_rows", 1)
        require(_int(entry["feature_dim"], "feature_dim") == 1536 and
                entry["dtype"] == "float32", "feature profile must be float32/1536")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = base / path
        target = path.resolve(strict=True)
        observed = path.stat()
        identity = observed.st_dev, observed.st_ino
        require(str(target) not in resolved and identity not in identities,
                "duplicate resolved/physical feature file")
        resolved.add(str(target))
        identities.add(identity)


def _chunks(data, *, compressed, limit, context):
    encoded = (data[offset:offset + 16384] for offset in range(0, len(data), 16384)) \
        if type(data) is bytes else iter(lambda: data.read(16384), b"")
    if not compressed:
        total = 0
        for chunk in encoded:
            total += len(chunk)
            require(total <= limit, f"{context}: expanded limit")
            yield chunk
        return
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    total = 0
    try:
        for chunk in encoded:
            require(not decoder.eof, f"{context}: trailing bytes/gzip member")
            while chunk:
                expanded = decoder.decompress(chunk, 65536)
                total += len(expanded)
                require(total <= limit, f"{context}: expanded limit")
                require(not decoder.unused_data, f"{context}: trailing bytes/gzip member")
                if expanded:
                    yield expanded
                chunk = decoder.unconsumed_tail
        require(decoder.eof, f"{context}: truncated gzip")
    except zlib.error as exc:
        raise HeadTaskError(f"{context}: malformed gzip") from exc


class _Lines:
    def __init__(self, chunks, context, record_limit):
        self.chunks = iter(chunks)
        self.buffer = bytearray()
        self.context, self.limit, self.record_bytes = context, record_limit, 0
        self.eof = False

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            end = self.buffer.find(b"\n")
            if end >= 0 or self.eof:
                if not self.buffer:
                    raise StopIteration
                end = end + 1 if end >= 0 else len(self.buffer)
                line = bytes(self.buffer[:end])
                del self.buffer[:end]
                require(self.record_bytes != 0 or line not in (b"\n", b"\r\n", b"\r"),
                        f"{self.context}: blank CSV record")
                self.record_bytes += len(line)
                require(self.record_bytes <= self.limit,
                        f"{self.context}: per-record byte limit")
                return line.decode("utf-8")
            require(self.record_bytes + len(self.buffer) <= self.limit,
                    f"{self.context}: per-record byte limit")
            try:
                self.buffer.extend(next(self.chunks))
            except StopIteration:
                self.eof = True


def csv_rows(data, header, context, *, compressed=False, expanded_limit,
             row_limit, record_limit=8192):
    lines = _Lines(_chunks(data, compressed=compressed, limit=expanded_limit,
                           context=context), context, record_limit)
    reader = csv.reader(lines, strict=True)
    try:
        require(next(reader, None) == list(header), f"{context}: wrong CSV header")
        lines.record_bytes = 0
        for index, row in enumerate(reader):
            require(index < row_limit, f"{context}: declared row limit exceeded")
            require(len(row) == len(header), f"{context}: malformed CSV width")
            lines.record_bytes = 0
            yield dict(zip(header, row))
    except (UnicodeError, csv.Error) as exc:
        raise HeadTaskError(f"{context}: malformed CSV") from exc


def _controls(spec, sources):
    consumed = {name: sources[role] for role, name in CONSUMED.items()}
    assets.verify_cpu_review_link(
        sources["cpu_review"], sources["cpu_complete"], branch="filter",
        receipt_key="filter_attempt_001/COMPLETE.json", consumed=consumed)
    bindings = _json(sources["cpu_input_bindings"], "CPU bindings")
    entries = {(entry["kind"], entry["name"]): entry for entry in bindings["bindings"]}
    original_inventory = entries.get(("source", "stage1a_feature_inventory"))
    require(original_inventory is not None, "missing source/stage1a_feature_inventory binding")
    _same(original_inventory["size"], len(sources["feature_inventory"]), "inventory size")
    _same(original_inventory["sha256"], _sha(sources["feature_inventory"]), "inventory hash")
    old_roles = {"filter_kernel": "kernel", "filter_cli": "runner", "cpu_io": "io",
                 "legacy_io": "legacy_io", "package_init": "package_init",
                 "evaluation_init": "evaluation_init"}
    for role, old_role in old_roles.items():
        entry = entries.get(("code", old_role))
        require(entry is not None, f"CPU did not bind code/{old_role}")
        _same(entry["sha256"], spec["code_files"][role]["sha256"], f"{role} CPU digest")
        _same(_int(entry["size"], "CPU code size"), spec["code_files"][role]["size"],
              f"{role} CPU size")
    for role, kind in (("asset_fit_review", "implementation"),
                       ("primitive_integration_review", "synthetic_integration")):
        review = _json(sources[role], role)
        _version(review, role)
        require(review.get("disposition") == "accepted" and review.get("review_kind") == kind
                and review.get("contract_sha256") == PRIMITIVE_CONTRACT,
                f"{role}: wrong primitive acceptance")
        require(type(review.get("source_files")) is dict, f"{role}: source_files required")
        for code_role, leaf in (("fit", "r2_1_hoptimus_fit.py"),
                                ("asset_io", "r2_1_hoptimus_asset_io.py")):
            require(review["source_files"].get("prosper/evaluation/" + leaf) ==
                    spec["code_files"][code_role]["sha256"],
                    f"{role}: executing {code_role} not accepted")
    return bindings, {
        name: {"size": len(data), "sha256": _sha(data)} for name, data in consumed.items()
    }


def _inventory(spec, data):
    declarations = {entry["slide_id"]: entry for entry in spec["features"]}
    inventory, seen = {}, set()
    for row in csv_rows(data, manifests.FEATURE_HEADER, "feature inventory",
                        expanded_limit=2 * MIB, row_limit=2 * MIB):
        key = row["bank"], _text(row["slide_id"], "inventory slide")
        require(row["bank"] in ("uni2", "virchow2", "h_optimus_1") and key not in seen,
                "unknown/duplicate inventory bank/slide")
        seen.add(key)
        for name in manifests.FEATURE_HEADER[2:6] + manifests.FEATURE_HEADER[7:11] + (
            "label_csv_position_mismatches", "size",
        ):
            row[name] = _decimal(row[name], f"inventory {name}")
        _digest(row["sha256"], "inventory hash")
        for name in ("feature_path", "resolved_path", "dtype"):
            _text(row[name], f"inventory {name}")
        require(row["label_csv_order_equal"] in ("true", "false"), "inventory order flag")
        require(row["feature_rows"] == sum(row[name] for name in (
            "target_rows", "unused_unannotated", "unused_labelled_nontarget",
            "unused_outside_declared_sources")), "inventory row totals")
        if row["bank"] != "h_optimus_1":
            continue
        require(row["slide_id"] in declarations, "undeclared H-Optimus inventory slide")
        entry = declarations[row["slide_id"]]
        for declared, field in (("path", "feature_path"), ("sha256", "sha256"),
                                ("size", "size"), ("feature_rows", "feature_rows"),
                                ("feature_dim", "feature_dim"), ("dtype", "dtype")):
            _same(entry[declared], row[field], f"feature/inventory {declared}")
        require((row["projection_start"], row["projection_stop"]) == (0, 1536),
                "wrong H-Optimus projection")
        inventory[row["slide_id"]] = row
    require(set(inventory) == set(declarations), "missing H-Optimus inventory slides")
    return inventory


def _folds(spec, data):
    mapping, order, flags, patients = {}, {}, {}, []
    for row in csv_rows(data, manifests.FOLD_HEADER, "patient folds",
                        expanded_limit=4 * MIB, row_limit=spec["expected"]["union_patients"]):
        patient = _text(row["patient_id"], "patient ID")
        fold, index = _decimal(row["fold"], "fold"), _decimal(row["kernel_order"], "kernel order")
        require(patient not in mapping and fold < 5 and index not in order,
                "duplicate patient/order or invalid fold")
        require(row["patient_hash"] == kernel.patient_fold_hash(patient), "wrong patient hash")
        require(row["has_annotations"] in ("0", "1") and row["has_retrieval"] in ("0", "1"),
                "invalid patient membership flags")
        mapping[patient], order[index] = fold, patient
        flags[patient] = row["has_annotations"] == "1"
        patients.append(patient)
    require(len(mapping) == spec["expected"]["union_patients"] and
            set(order) == set(range(len(mapping))), "patient kernel_order permutation/count")
    require(patients == sorted(patients), "patient presentation order mismatch")
    require([order[i] for i in range(len(order))] == sorted(
        mapping, key=lambda patient: (kernel.patient_fold_hash(patient), patient)),
        "patient hash-order evidence mismatch")
    return mapping, flags


def _annotations(spec, data, inventory, mapping, flags):
    records, physical, sources, metadata = {}, set(), set(), {}
    held_out, patients, by_slide = [], set(), Counter()
    fold = spec["parameters"]["held_out_fold"]
    for row in csv_rows(data, manifests.ANNOTATION_HEADER, "annotations", compressed=True,
                        expanded_limit=128 * MIB, row_limit=spec["expected"]["annotation_tiles"]):
        values = {name: _decimal(row[name], name) for name in (
            "level", "x", "y", "patch_size", "tissue_label", "h_optimus_1_feature_row",
            "held_out_fold")}
        record = kernel.AnnotationRecord(**{
            **{name: row[name] for name in ("tile_id", "slide_id", "patient_id", "diagnostic_group")},
            **{name: value for name, value in values.items() if name != "held_out_fold"},
        })
        require(record.patient_id in mapping and values["held_out_fold"] == mapping[record.patient_id],
                "annotation patient fold mismatch")
        require(record.slide_id in inventory and record.h_optimus_1_feature_row <
                inventory[record.slide_id]["feature_rows"], "annotation slide/source index")
        require(record.tile_id not in records and record.physical_key not in physical and
                (record.slide_id, record.h_optimus_1_feature_row) not in sources,
                "duplicate annotation physical identity/source index")
        pair = record.patient_id, record.diagnostic_group
        require(record.slide_id not in metadata or metadata[record.slide_id] == pair,
                "inconsistent joint slide metadata")
        metadata[record.slide_id] = pair
        records[record.tile_id] = record
        physical.add(record.physical_key)
        sources.add((record.slide_id, record.h_optimus_1_feature_row))
        patients.add(record.patient_id)
        by_slide[record.slide_id] += 1
        if values["held_out_fold"] == fold:
            held_out.append(record)
    require(len(records) == spec["expected"]["annotation_tiles"] and
            len(by_slide) == spec["expected"]["annotation_slides"] and
            len(patients) == spec["expected"]["annotation_patients"],
            "annotation population counts")
    require(set(by_slide) == set(inventory), "annotation/inventory slide set mismatch")
    require(patients == {patient for patient, annotated in flags.items() if annotated},
            "patient annotation membership mismatch")
    for slide, count in by_slide.items():
        require(count == inventory[slide]["target_rows"], "inventory labelled-row count mismatch")
    return records, tuple(held_out)


def _plans(data, run_id):
    plan = _json(data, "head plan")
    cpu.keys(plan, {"schema_version", "run_id", "task", "classes", "heads"}, "head plan")
    _version(plan, "head plan")
    require(plan["run_id"] == run_id and plan["task"] == manifests.TASK, "head plan run/task")
    _same(plan["classes"], [1, 2, 3, 4, 5], "head plan classes")
    require(type(plan["heads"]) is list and len(plan["heads"]) == 10, "ten heads required")
    result = {}
    for head in plan["heads"]:
        cpu.keys(head, {"held_out_fold", "budget_percent", "pool_count", "selected_count",
                        "selected_fraction", "class_stats", "training_tile_ids_sha256"},
                 "planned head")
        key = (_int(head["held_out_fold"], "planned fold"), _int(head["budget_percent"], "budget"))
        require(key[0] < 5 and key[1] in (5, 100) and key not in result, "duplicate/invalid head")
        for name in ("pool_count", "selected_count"):
            _int(head[name], name, 1)
        _float(head["selected_fraction"], "head fraction")
        require(head["selected_fraction"] == head["selected_count"] / head["pool_count"],
                "head fraction/count mismatch")
        _digest(head["training_tile_ids_sha256"], "ordered training digest")
        require(type(head["class_stats"]) is list and len(head["class_stats"]) == 5,
                "five ordered class-stat rows required")
        for label, entry in enumerate(head["class_stats"], 1):
            cpu.keys(entry, {"tissue_label", "pool_count", "selected_count", "fraction",
                             "balanced_weight"}, "class stat")
            require(_int(entry["tissue_label"], "class label", 1) == label, "class stat order")
            pool = _int(entry["pool_count"], "class pool", 1)
            selected = _int(entry["selected_count"], "class selected", 1)
            require(selected == (pool if key[1] == 100 else kernel.five_percent_count(pool)),
                    "planned class rounding/count mismatch")
            require(_float(entry["fraction"], "class fraction") == selected / pool and
                    _float(entry["balanced_weight"], "class weight") ==
                    head["selected_count"] / (5 * selected), "planned class fraction/weight")
        require(sum(row["pool_count"] for row in head["class_stats"]) == head["pool_count"] and
                sum(row["selected_count"] for row in head["class_stats"]) ==
                head["selected_count"], "planned head totals")
        result[key] = head
    return result


def _training(spec, data, plans, annotations, mapping):
    requested = spec["parameters"]["held_out_fold"], spec["parameters"]["budget_percent"]
    retained, finalized, current, seen, counts = [], set(), None, set(), Counter()
    position, digest = 0, None

    def finalize():
        require(current is not None, "empty training stream")
        plan = plans[current]
        digest.update(b"]")
        require(position == plan["selected_count"] and
                digest.hexdigest() == plan["training_tile_ids_sha256"],
                "streamed head count/order digest mismatch")
        require(all(counts[label] == row["selected_count"]
                    for label, row in enumerate(plan["class_stats"], 1)),
                "streamed head class counts")
        finalized.add(current)

    for row in csv_rows(data, manifests.TRAINING_HEADER, "training", compressed=True,
                        expanded_limit=512 * MIB,
                        row_limit=sum(head["selected_count"] for head in plans.values())):
        key = _decimal(row["held_out_fold"], "training fold"), _decimal(row["budget_percent"], "budget")
        require(key in plans, "unknown training head")
        if key != current:
            if current is not None:
                finalize()
            require(key not in finalized, "training head reappeared after finalization")
            require(key == (len(finalized) // 2, 5 if len(finalized) % 2 == 0 else 100),
                    "training head presentation order")
            current, seen, counts, position = key, set(), Counter(), 0
            digest = hashlib.sha256(b"[")
        require(_decimal(row["training_position"], "training_position") == position,
                "noncontiguous training_position")
        tile = row["tile_id"]
        require(tile in annotations and tile not in seen, "unknown/duplicate training tile")
        record = annotations[tile]
        require(row["slide_id"] == record.slide_id and row["patient_id"] == record.patient_id and
                _decimal(row["tissue_label"], "training label") == record.tissue_label and
                _decimal(row["h_optimus_1_feature_row"], "training feature row") ==
                record.h_optimus_1_feature_row, "training/annotation identity mismatch")
        require(mapping[record.patient_id] != key[0], "same-patient training leakage")
        require(position < plans[key]["selected_count"], "head selected row limit")
        if position:
            digest.update(b",")
        digest.update(json.dumps(tile, ensure_ascii=True).encode("ascii"))
        seen.add(tile)
        counts[record.tissue_label] += 1
        if key == requested:
            retained.append(record)
        position += 1
    finalize()
    require(set(plans) == finalized, "missing training heads")
    plan = plans[requested]
    pool = Counter(record.tissue_label for record in annotations.values()
                   if mapping[record.patient_id] != requested[0])
    require(all(pool[label] == row["pool_count"] for label, row in
                enumerate(plan["class_stats"], 1)), "requested head full training pool counts")
    return kernel.TrainingManifest(
        requested[0], requested[1], tuple(retained),
        tuple(kernel.ClassSelection(**row) for row in plan["class_stats"]),
        plan["pool_count"], plan["selected_count"], plan["selected_fraction"])


def _population_counts(spec, annotations, mapping, manifest, held, sources, cpu_run):
    summary = _json(sources["cpu_summary"], "CPU summary")
    _version(summary, "CPU summary")
    require(summary.get("run_id") == cpu_run and summary.get("task") == manifests.TASK,
            "CPU summary run/task")
    _same(summary["policy"], manifests.POLICY, "CPU manifest policy")
    for name in ("annotation_tiles", "annotation_slides", "annotation_patients"):
        _same(summary["counts"][name], spec["expected"][name], f"CPU {name}")
    observed = {
        "annotation_tiles": len(annotations),
        "annotation_slides": len({row.slide_id for row in annotations.values()}),
        "annotation_patients": len({row.patient_id for row in annotations.values()}),
        "union_patients": len(mapping), "feature_files": len(spec["features"]),
        "training_tiles": len(manifest.tiles),
        "training_slides": len({row.slide_id for row in manifest.tiles}),
        "training_patients": len({row.patient_id for row in manifest.tiles}),
        "held_out_tiles": len(held), "held_out_slides": len({row.slide_id for row in held}),
        "held_out_patients": len({row.patient_id for row in held}),
    }
    _same(observed, spec["expected"], "pre-fit counts")
    require(not ({row.patient_id for row in manifest.tiles} & {row.patient_id for row in held}),
            "train/held-out patient overlap")


def _blocks(guard, inventory, records, consumption):
    for slide in sorted({row.slide_id for row in records}):
        encoded = guard.read_verified("feature:" + slide)
        payload = legacy._pickle(encoded, f"feature {slide}")
        del encoded
        require(type(payload) is tuple and len(payload) == 2, "feature (features,tiles) tuple required")
        raw, tiles = payload
        require(type(raw) is np.ndarray and raw.dtype == np.dtype(np.float32) and
                raw.shape == (inventory[slide]["feature_rows"], 1536) and
                bool(np.isfinite(raw).all()), "feature raw shape/dtype/finiteness")
        coordinates = legacy._coordinates(tiles, f"feature {slide}")
        require(len(coordinates) == raw.shape[0], "feature coordinate row count")
        require(all(0 <= value <= np.iinfo(np.int64).max for pair in coordinates for value in pair),
                "feature coordinates outside nonnegative int64")
        coordinates = np.array(coordinates, dtype=np.int64)
        consumption[slide] = "decoded"
        yield slide, raw, coordinates
        del payload, raw, tiles, coordinates


def _note(primary, additional, context):
    primary.add_note(f"{context}: {type(additional).__name__}: {additional}")


def _close(fd, primary=None):
    try:
        os.close(fd)
    except BaseException as exc:
        if primary is None:
            raise
        _note(primary, exc, "descriptor close")


def _identity(observed):
    return observed.st_dev, observed.st_ino


@contextmanager
def _fd_file(fd, mode):
    try:
        handle = os.fdopen(fd, mode)
    except BaseException as exc:
        _close(fd, exc)
        raise
    primary = None
    try:
        yield handle
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            handle.close()
        except BaseException as exc:
            if primary is None:
                raise
            _note(primary, exc, "file close")


class _Output:
    """Local single-stage ownership and streamed readback, not a shared runner."""
    def __init__(self, out, root, input_paths, feature_roots):
        self.out, self.root = Path(out).absolute(), Path(root).absolute()
        require("PROSPER_OUT" in os.environ, "explicit PROSPER_OUT required")
        require(Path(os.environ["PROSPER_OUT"]).resolve(strict=True) == self.root.resolve(strict=True),
                "PROSPER_OUT/out-root mismatch")
        self.target = legacy._output_path(self.out, self.root, input_paths, feature_roots)
        self.root_target = self.root.resolve(strict=True)
        self.root_identity = _identity(self.root.stat())
        self.fd, self.owned, self.files = None, False, {}
        self.directory_identity = None
        self.release_state = "unclaimed"
        self._uncertain_fd = None
        self._cleanup_recovery_attempted = False

    def claim(self):
        parent_fd = os.open(self.out.parent, os.O_RDONLY | os.O_DIRECTORY)
        descriptor, primary = None, None
        try:
            require(self.root.resolve(strict=True) == self.root_target and
                    _identity(self.root.stat()) == self.root_identity,
                    "output root retargeted before claim")
            require(_identity(os.fstat(parent_fd)) == _identity(self.target.parent.stat()),
                    "output parent retargeted before claim")
            os.mkdir(self.out.name, dir_fd=parent_fd)
            observed = os.stat(self.out.name, dir_fd=parent_fd, follow_symlinks=False)
            descriptor = os.open(self.out.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                 dir_fd=parent_fd)
            require(stat.S_ISDIR(observed.st_mode) and
                    _identity(observed) == _identity(os.fstat(descriptor)) ==
                    _identity(os.stat(self.out.name, dir_fd=parent_fd, follow_symlinks=False)),
                    "output acquisition handoff changed identity")
            require(not os.listdir(descriptor), "new output directory not empty")
            require(self.out.resolve(strict=True) == self.target, "output retargeted during claim")
            require(stat.S_ISDIR(self.out.lstat().st_mode) and
                    self.root.resolve(strict=True) == self.root_target and
                    _identity(self.root.stat()) == self.root_identity,
                    "output/root acquisition substitution")
            legacy._claimed_path(descriptor, self.target)
            self.directory_identity = _identity(observed)
            self.fd, descriptor = descriptor, None
            self.owned = True
            self.release_state = "active"
            self.check_path()
        except BaseException as exc:
            primary = exc
            raise
        finally:
            if descriptor is not None:
                _close(descriptor, primary)
            _close(parent_fd, primary)

    def check_path(self):
        require(self.owned and self.fd is not None, "output not owned")
        require(self._owns_descriptor(self.fd), "output descriptor is no longer owned")
        require(self.root.resolve(strict=True) == self.root_target and
                _identity(self.root.stat()) == self.root_identity and
                self.out.resolve(strict=True) == self.target,
                "output/root path retargeted")
        require(stat.S_ISDIR(self.out.lstat().st_mode), "output replaced by symlink")
        legacy._claimed_path(self.fd, self.target)

    def _owns_descriptor(self, descriptor):
        observed = os.fstat(descriptor)
        return stat.S_ISDIR(observed.st_mode) and _identity(observed) == self.directory_identity

    def _cleanup_access(self, error):
        candidate = self.fd if self.fd is not None else self._uncertain_fd
        self.fd, self._uncertain_fd = None, None
        if candidate is not None:
            try:
                available = self._owns_descriptor(candidate)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise
                available = False
            if available:
                self.fd = candidate
                self.release_state = "active"
                return
            error.add_note("Cleanup access lost: stale/reused descriptor not used or closed")
        self.release_state = "access_lost"
        require(not self._cleanup_recovery_attempted, "cleanup recovery already attempted")
        self._cleanup_recovery_attempted = True
        require(self.root.resolve(strict=True) == self.root_target and
                _identity(self.root.stat()) == self.root_identity and
                self.out.resolve(strict=True) == self.target and
                stat.S_ISDIR(self.out.lstat().st_mode) and
                _identity(self.out.stat()) == self.directory_identity,
                "cleanup recovery refused: directory/root identity or path changed")
        descriptor = os.open(self.out, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        primary = None
        try:
            require(self._owns_descriptor(descriptor) and
                    self.root.resolve(strict=True) == self.root_target and
                    _identity(self.root.stat()) == self.root_identity and
                    self.out.resolve(strict=True) == self.target and
                    stat.S_ISDIR(self.out.lstat().st_mode) and
                    _identity(self.out.stat()) == self.directory_identity,
                    "cleanup recovery refused: acquired directory changed")
            try:
                marker = os.stat("COMPLETE.json", dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                require(self.files.get("COMPLETE.json") == _identity(marker),
                        "cleanup recovery refused: foreign COMPLETE")
            self.fd, descriptor = descriptor, None
            self.release_state = "active"
        except BaseException as exc:
            primary = exc
            raise
        finally:
            if descriptor is not None:
                _close(descriptor, primary)

    def write(self, name, emit):
        self.check_path()
        require(name not in os.listdir(self.fd), f"{name}: refusing existing artifact")
        part = "." + name + ".part"
        descriptor = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=self.fd)
        with _fd_file(descriptor, "wb") as handle:
            self.files[name] = _identity(os.fstat(handle.fileno()))
            emit(handle)
            handle.flush()
            os.fsync(handle.fileno())
        require(name not in os.listdir(self.fd), f"{name}: final leaf appeared during write")
        os.replace(part, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
        os.fsync(self.fd)

    def _scan(self, handle, name, limit):
        before = os.fstat(handle.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= limit and
                _identity(before) == self.files[name] ==
                _identity(os.stat(name, dir_fd=self.fd, follow_symlinks=False)),
                f"{name}: invalid output acquisition/size")
        handle.seek(0)
        digest, total = hashlib.sha256(), 0
        while True:
            data = handle.read(min(MIB, before.st_size - total + 1))
            if not data:
                break
            total += len(data)
            require(total <= before.st_size, f"{name}: output grew during read")
            digest.update(data)
        after = os.fstat(handle.fileno())
        path = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        require(total == before.st_size == after.st_size == path.st_size and
                _identity(before) == _identity(after) == _identity(path),
                f"{name}: output changed while reading")
        return {"size": total, "sha256": digest.hexdigest()}

    def readback(self, name, limit, semantic, *, expected=None):
        self.check_path()
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
        with _fd_file(descriptor, "rb") as handle:
            binding = self._scan(handle, name, limit)
            if expected is not None:
                _same(binding, expected, f"{name} original serialized size/hash")
            handle.seek(0)
            result = semantic(handle)
            _same(self._scan(handle, name, limit), binding, f"{name} semantic custody")
        self.check_path()
        return binding, result

    def check(self, bindings, *, marker=False):
        self.check_path()
        require(set(os.listdir(self.fd)) == set(bindings), "output allowlist mismatch")
        for name, expected in bindings.items():
            actual, _ = self.readback(name, expected["size"], lambda handle: None)
            _same(actual, expected, f"{name} final output hash")
        self.check_path()

    def fail(self, error, fit_warnings):
        if not self.owned:
            return
        self._cleanup_access(error)
        try:
            observed = os.stat("COMPLETE.json", dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if self.files.get("COMPLETE.json") == _identity(observed):
                os.unlink("COMPLETE.json", dir_fd=self.fd)
            else:
                error.add_note("Marker cleanup: refusing to remove foreign COMPLETE")
        data = cpu.json_bytes({
            "error_type": type(error).__name__, "message": str(error),
            "notes": list(getattr(error, "__notes__", ())), "fit_warnings": fit_warnings,
        })
        # Failure evidence follows the acquired descriptor even if the public
        # path was retargeted. Never touch the substituted directory.
        descriptor = os.open("FAILED", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=self.fd)
        with _fd_file(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(self.fd)

    def close(self, primary=None):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            self.release_state = "releasing"
            try:
                require(self._owns_descriptor(fd), "refusing to close an unowned/reused descriptor")
                os.close(fd)
            except BaseException as exc:
                self._uncertain_fd = fd
                self.release_state = "release_uncertain"
                if primary is None:
                    raise
                _note(primary, exc, "owned directory release")
            else:
                self._uncertain_fd = None
                self.release_state = "released"


def _json_output(output, name, value, bindings):
    data = cpu.json_bytes(value)
    require(len(data) <= 8 * MIB, f"{name}: JSON limit")
    output.write(name, lambda handle: handle.write(data))
    binding, _ = output.readback(
        name, 8 * MIB, lambda handle: _same(cpu.strict_json(handle.read()), value, name))
    bindings[name] = binding


def _array_same(actual, expected, context):
    require(type(actual) is np.ndarray and actual.shape == expected.shape and
            actual.dtype.str == expected.dtype.str and actual.flags.c_contiguous,
            f"{context}: array shape/dtype/layout mismatch")
    for offset in range(0, len(expected), 4096):
        require(actual[offset:offset + 4096].tobytes(order="C") ==
                expected[offset:offset + 4096].tobytes(order="C"),
                f"{context}: byte-exact array fidelity mismatch")


def _array_output(output, name, array, bindings):
    require(array.dtype.str in ("<f4", "<i8") and array.flags.c_contiguous,
            "NPY output must be little-endian/C-order")
    output.write(name, lambda handle: np.lib.format.write_array(
        handle, array, version=(1, 0), allow_pickle=False))

    def check(handle):
        require(np.lib.format.read_magic(handle) == (1, 0), f"{name}: NPY version")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        require(shape == array.shape and not fortran and dtype.str == array.dtype.str,
                f"{name}: NPY header mismatch")
        require(os.fstat(handle.fileno()).st_size == handle.tell() + array.nbytes,
                f"{name}: trailing/truncated NPY payload")
        handle.seek(0)
        restored = np.load(handle, allow_pickle=False)
        require(handle.read(1) == b"", f"{name}: trailing NPY array")
        _array_same(restored, array, name)

    binding, _ = output.readback(name, array.nbytes + 65545, check)
    bindings[name] = binding


def _held_cells(index, record, fold):
    return (index, record.tile_id, record.slide_id, record.patient_id,
            record.diagnostic_group, record.level, record.x, record.y, record.patch_size,
            record.tissue_label, record.h_optimus_1_feature_row, fold)


def _held_output(output, records, fold, bindings):
    def write(handle):
        with gzip.GzipFile(fileobj=handle, mode="wb", filename="", mtime=0) as compressed:
            text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
            try:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(HELD_HEADER)
                for index, record in enumerate(records):
                    writer.writerow(_held_cells(index, record, fold))
                text.flush()
            finally:
                text.detach()

    output.write("held_out_rows.csv.gz", write)
    limit = (len(records) + 1) * (8192 + 64) + 4096

    def check(handle):
        require(handle.read(8) == b"\x1f\x8b\x08\x00" + b"\x00" * 4,
                "held-out gzip header")
        handle.seek(0)
        count = 0
        for index, row in enumerate(csv_rows(
            handle, HELD_HEADER, "held-out output", compressed=True,
            expanded_limit=limit, row_limit=len(records), record_limit=8192 + 64,
        )):
            require(tuple(row.values()) == tuple(str(value) for value in
                    _held_cells(index, records[index], fold)), "held-out CSV truth/order mismatch")
            count += 1
        require(count == len(records), "held-out CSV row count mismatch")

    binding, _ = output.readback("held_out_rows.csv.gz", limit, check)
    bindings["held_out_rows.csv.gz"] = binding


def _predict(head, features):
    probabilities = np.empty((len(features), 5), dtype=np.float32)
    labels = np.empty(len(features), dtype=np.int64)
    for start in range(0, len(features), 4096):
        batch = fitter.predict_head(head, features[start:start + 4096])
        probabilities[start:start + 4096] = batch.probabilities
        labels[start:start + 4096] = batch.labels
    return probabilities, labels


def _model_output(output, head, features, probabilities, labels, bindings):
    data = pickle.dumps(head, protocol=4)
    require(len(data) <= 2 * MIB and data[:2] == b"\x80\x04", "protocol4 model limit")
    original = {"size": len(data), "sha256": _sha(data)}
    output.write("head.pkl", lambda handle: handle.write(data))
    del data

    def check(handle):
        data = handle.read(2 * MIB + 1)
        require(len(data) <= 2 * MIB and data[:2] == b"\x80\x04", "model readback framing/limit")
        _same({"size": len(data), "sha256": _sha(data)}, original,
              "head.pkl bytes before deserialization")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                restored = legacy._pickle(data, "just-written head")
            except BaseException as exc:
                if caught:
                    exc.add_note("Model-loader warnings (not fit warnings): " +
                                 cpu.json_bytes(fitter._warning_records(caught)).decode("utf-8").rstrip())
                raise
        if caught:
            error = HeadTaskError("head.pkl deserialization: unexpected model-loader warnings")
            error.add_note("Model-loader warnings (not fit warnings): " +
                           cpu.json_bytes(fitter._warning_records(caught)).decode("utf-8").rstrip())
            raise error
        require(type(restored) is fitter.FittedHead, "wrong restored head type")
        _same(restored.diagnostics, head.diagnostics, "restored immutable diagnostics")
        fitter.predict_head(restored, np.empty((0, 1536), np.float32))
        for start in range(0, len(features), 4096):
            batch = fitter.predict_head(restored, features[start:start + 4096])
            _array_same(batch.probabilities, probabilities[start:start + 4096],
                        "restored probabilities")
            _array_same(batch.labels, labels[start:start + 4096], "restored labels")

    binding, _ = output.readback("head.pkl", 2 * MIB, check, expected=original)
    bindings["head.pkl"] = binding


def run(spec_path, spec_sha256, out, out_root):
    started = _time()
    _digest(spec_sha256, "spec digest")
    spec_path = Path(spec_path).absolute()
    data, initial = cpu._scan_file(spec_path, "spec", "spec", 4 * MIB)
    require(initial["sha256"] == spec_sha256, "spec digest mismatch")
    spec, base = _json(data, "spec"), spec_path.parent
    _schema(spec, base)
    declarations = [assets.AssetSpec("spec", "spec", spec_path, spec_sha256, len(data))]
    policies = {"spec": assets.AssetPolicy("bytes", 4 * MIB),
                "code": assets.AssetPolicy("bytes", 2 * MIB),
                "feature": assets.AssetPolicy("bytes", 192 * MIB)}
    for role, entry in spec["sources"].items():
        declarations.append(assets.AssetSpec("source:" + role, role, **entry))
        policies[role] = assets.AssetPolicy("bytes", SOURCE_LIMITS[role])
    for role, entry in spec["code_files"].items():
        declarations.append(assets.AssetSpec("code:" + role, "code", **entry))
    for entry in spec["features"]:
        declarations.append(assets.AssetSpec("feature:" + entry["slide_id"], "feature",
                                            **{key: entry[key] for key in ("path", "sha256", "size")}))
    input_paths = [Path(entry.path) if Path(entry.path).is_absolute() else base / entry.path
                   for entry in declarations]
    roots = [path.parent for path, entry in zip(input_paths, declarations) if entry.role == "feature"]
    output = _Output(out, out_root, input_paths, roots)
    primary, fit_warnings = None, []
    try:
        with assets.AssetGuard(declarations, base_dir=base, policies=policies) as guard:
            require(guard.read_verified("spec") == data, "spec reacquisition mismatch")
            observed_spec = spec_path.stat()
            require((observed_spec.st_dev, observed_spec.st_ino, observed_spec.st_size) ==
                    (initial["device"], initial["inode"], initial["size"]) and
                    str(spec_path.resolve(strict=True)) == initial["resolved_path"],
                    "spec bootstrap/reacquisition identity changed")
            for role in spec["code_files"]:
                guard.read_verified("code:" + role)
            sources = {role: guard.read_verified("source:" + role) for role in SOURCE_LIMITS}
            cpu_bindings, consumed = _controls(spec, sources)
            inventory = _inventory(spec, sources["feature_inventory"])
            mapping, flags = _folds(spec, sources["patient_folds"])
            annotations, held = _annotations(
                spec, sources["annotation_assignments"], inventory, mapping, flags)
            plans = _plans(sources["head_plan"], cpu_bindings["run_id"])
            manifest = _training(spec, sources["training_rows"], plans, annotations, mapping)
            _population_counts(spec, annotations, mapping, manifest, held, sources,
                               cpu_bindings["run_id"])
            sources.clear()
            del data
            output.claim()
            ordered = manifest.tiles + held
            consumption = {entry["slide_id"]: "hash_only" for entry in spec["features"]}
            matrix, labels = fitter.assemble_selected_features(
                ordered, _blocks(guard, inventory, ordered, consumption))
            count = len(manifest.tiles)
            _same(labels[:count].tolist(), [row.tissue_label for row in manifest.tiles],
                  "assembled training labels")
            truth = labels[count:]
            _same(truth.tolist(), [row.tissue_label for row in held], "assembled held-out truth")
            head = fitter.fit_head(matrix[:count], manifest, mapping)
            diagnostics = head.diagnostics
            fit_warnings = diagnostics["warnings"]
            probabilities, predictions = _predict(head, matrix[count:])
            require(np.array_equal(predictions,
                    head.estimator.classes_[probabilities.argmax(axis=1)]),
                    "prediction/probability argmax mismatch")
            tumour = fitter.tumour_counts(truth, predictions)
            confusion = np.zeros((5, 5), np.int64)
            for actual, predicted in zip(truth, predictions):
                confusion[int(actual) - 1, int(predicted) - 1] += 1
            support = [{
                "tissue_label": label, "tiles": sum(row.tissue_label == label for row in held),
                "slides": len({row.slide_id for row in held if row.tissue_label == label}),
                "patients": len({row.patient_id for row in held if row.tissue_label == label}),
            } for label in range(1, 6)]
            counts = dict(schema_version=1, run_id=spec["run_id"], task=TASK,
                          held_out_fold=manifest.held_out_fold, budget_percent=manifest.budget_percent,
                          tumour_counts=tumour, tissue_confusion=confusion.tolist(),
                          held_out_support=support)
            summary = {
                "schema_version": 1, "run_id": spec["run_id"], "task": TASK,
                "counts": spec["expected"],
                "policy": {
                    "held_out_fold": manifest.held_out_fold, "budget_percent": manifest.budget_percent,
                    "classes": [1, 2, 3, 4, 5], "recipe": "r2_1_fixed_sgd_ovr_v1",
                    "feature_dtype": "float32", "prediction_batch_rows": 4096,
                    "decision": "probability_argmax_class1", "scope": SCOPE,
                    "tuning": "none", "encoder_replay": "not_performed_existing_annotation_bank",
                },
                "details": {
                    "fitted_state_sha256": diagnostics["fitted_state_sha256"],
                    "training_tile_ids_sha256": manifests._ordered_ids_sha(manifest.tiles),
                    "held_out_tile_ids_sha256": manifests._ordered_ids_sha(held),
                    "fit_warnings": fit_warnings, "convergence_status": diagnostics["convergence_status"],
                    "serialization_protocol": 4, "features_total": len(consumption),
                    "features_decoded": sum(status == "decoded" for status in consumption.values()),
                    "features_hash_only": sum(status == "hash_only" for status in consumption.values()),
                    "raw_rows_decoded": sum(inventory[slide]["feature_rows"] for slide, status
                                            in consumption.items() if status == "decoded"),
                    "held_out_prediction_rows": len(held), "scientific_assessment": "pending",
                },
                "limitations": LIMITATIONS,
            }
            bindings = {}
            _model_output(output, head, matrix[count:], probabilities, predictions, bindings)
            _json_output(output, "fit_diagnostics.json", diagnostics, bindings)
            _held_output(output, held, manifest.held_out_fold, bindings)
            _array_output(output, "probabilities.npy", probabilities, bindings)
            _array_output(output, "predictions.npy", predictions, bindings)
            _json_output(output, "counts.json", counts, bindings)
            _json_output(output, "summary.json", summary, bindings)
            output.check(bindings)
            final_inputs = guard.final_bindings()
            input_record = {
                "schema_version": 1, "run_id": spec["run_id"], "task": TASK,
                "spec_sha256": spec_sha256, "bindings": list(final_inputs),
                "executing_code_paths": {role: str(path.resolve(strict=True))
                                         for role, path in code_paths().items()},
                "environment": {"python": sys.version, "hostname": socket.gethostname(),
                                "fit_runtime": diagnostics["runtime"],
                                "native_environment": {name: os.environ.get(name) for name in NATIVE_VARIABLES}},
                "started_at": started, "captured_at": _time(),
                "head": {"held_out_fold": manifest.held_out_fold, "budget_percent": manifest.budget_percent},
                "consumed_cpu_artifacts": consumed,
                "feature_consumption": [{"slide_id": slide, "asset_id": "feature:" + slide,
                                         "status": consumption[slide]} for slide in sorted(consumption)],
                "limitations": LIMITATIONS,
            }
            _json_output(output, "input_bindings.json", input_record, bindings)
            require(set(bindings) == ARTIFACTS, "final artifact set mismatch")
            output.check(bindings)
            marker = dict(schema_version=1, run_id=spec["run_id"], task=TASK, scope=SCOPE,
                          spec_sha256=spec_sha256, outputs=dict(bindings),
                          verified_at=_time(), limitations=LIMITATIONS)
            _json_output(output, "COMPLETE.json", marker, bindings)
            output.check(bindings, marker=True)
        output.close()
        return marker
    except BaseException as exc:
        primary = exc
        if not fit_warnings:
            for note in tuple(getattr(exc, "__notes__", ())):
                for prefix in ("All captured fit warnings: ", "All captured warnings: "):
                    if note.startswith(prefix):
                        try:
                            reported = cpu.strict_json(note[len(prefix):].encode("utf-8"))
                            require(type(reported) is list, "warning note must contain a list")
                            fit_warnings = reported
                        except (ValueError, UnicodeError) as additional:
                            _note(exc, additional, "fit-warning note capture")
        try:
            output.fail(exc, fit_warnings)
        except BaseException as additional:
            _note(exc, additional, "failure record/marker cleanup")
        raise
    finally:
        output.close(primary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args(argv)
    marker = run(args.spec, args.spec_sha256, args.out, args.out_root)
    print(cpu.json_bytes(marker).decode("utf-8").rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
