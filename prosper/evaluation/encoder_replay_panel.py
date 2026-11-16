"""Prepare and bind a fixed coordinate/vector replay panel; no model forwards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import annotation_bank_audit as annotation
from . import feature_bank_audit as bank_audit


AuditError = bank_audit.AuditError
require = bank_audit._require
keys = bank_audit._keys
path_of = bank_audit._path
now = bank_audit._time
sha_text = annotation._sha
safe_id = annotation._safe_id
decimal = annotation._decimal
compact = lambda value: json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")

BANKS = ("uni2", "virchow2", "h_optimus_1")
ROOTS = ("O", "F3", "F4", "P3", "P4", "C", "T")
CONTROL_TAGS = ("annotation_array", "auxiliary_canary", "main_array",
                "main_canary", "recovery_a", "recovery_b")
PANEL_PRODUCTION = ("7483", "8874", "10003", "5605", "15825", "6798",
                    "16973", "6677", "17289", "15841", "19162", "19125")
PANEL_DIGEST = "bc0277bbf2f55939cac662d436a44d0dbb320f2ac9ad6ea86e59935b127b70d9"
MAIN_DIM = {"uni2": 1536, "virchow2": 1280, "h_optimus_1": 1536}
ANNOTATION_DIM = {"uni2": 1536, "virchow2": 2560, "h_optimus_1": 1536}
ARRAYS = ("uni2_main.npy", "uni2_annotation.npy",
          "virchow2_main_f16.npy", "virchow2_main_f32.npy",
          "virchow2_annotation_full.npy", "h_optimus_1_main.npy",
          "h_optimus_1_annotation.npy", "h_optimus_1_auxiliary.npy")
PANEL_HEADER = ("universe", "slide_id", "patient_id", "group", "selection_index",
                "level", "x", "y", "patch_size", "tile_id", "tissue_label",
                "selection_role")
FEATURE_HEADER = ("bank", "universe", "slide_id", "tile_id", "source_id",
                  "source_row", "source_dimension", "source_dtype", "array_name", "array_row")
SOURCE_HEADER = ("source_id", "kind", "bank", "universe", "slide_id", "path",
                 "resolved_path", "expected_presence", "observed_presence",
                 "expected_size_bytes", "observed_size_bytes",
                 "expected_sha256", "observed_sha256", "digest_basis")
PAIR_KEYS = ("pair_id", "left_source_id", "right_source_id", "relation", "status",
             "reason", "left_rows", "right_rows", "left_dimension", "right_dimension",
             "left_dtype", "right_dtype", "common_keys", "left_only_keys",
             "right_only_keys", "order_equal", "exact_value_equal", "bitwise_equal",
             "destination_cast_equal", "exact_value_mismatch_rows",
             "bitwise_mismatch_rows", "cast_mismatch_rows")
PAIR_COUNT_KEYS = ("pairs", "comparable", "non_comparable", "exact_value_different",
                   "bitwise_different", "destination_cast_different")
CSV_OUTPUTS = {"panel_tiles.csv": PANEL_HEADER, "feature_rows.csv": FEATURE_HEADER,
               "source_inventory.csv": SOURCE_HEADER}
OTHER_CAP = 128 * 1024 * 1024
FEATURE_CAP = 512 * 1024 * 1024
OUTPUT_NPY_CAP = 16 * 1024 * 1024
HASH_BLOCK = 1024 * 1024
SEED = 1234
SCOPE = "encoder_replay_panel_only"
WRITE_LINE = re.compile(
    r"\[extract\] (?P<slide>[^:\s]+): \((?P<rows>[1-9][0-9]*), 1536\) "
    r"in (?P<seconds>[0-9]+(?:\.[0-9]+)?)s "
    r"\((?P<rate>[0-9]+(?:\.[0-9]+)?) tiles/s\)",
    re.ASCII,
)


def _hex(value, context):
    require(type(value) is str and bool(re.fullmatch("[0-9a-f]{64}", value)),
            f"{context}: expected lower-case SHA256")
    return value


def _int(value, context, minimum=0):
    require(type(value) is int and value >= minimum,
            f"{context}: expected integer >= {minimum}, not bool")


def _string(value, context):
    require(type(value) is str and bool(value.strip()) and value == value.strip(),
            f"{context}: expected nonempty unpadded string")


def _list(value, context):
    require(type(value) is list, f"{context}: expected list")


def _declared_path(value, base, context):
    _string(value, context)
    return path_of(value, base)


def _schema(spec):
    keys(spec, ("schema_version", "run_id", "stage0", "stage1a", "metadata",
                "panel", "banks", "auxiliary", "wsi", "models", "virchow_candidates",
                "lineage_pairs", "log_controls", "code_files", "expected"), "spec")
    require(type(spec["schema_version"]) is int and spec["schema_version"] == 1,
            "schema_version: expected integer 1")
    safe_id(spec["run_id"], "run_id")
    for stage in ("stage0", "stage1a"):
        item = spec[stage]
        keys(item, ("review", "review_sha256", "complete"), stage)
        for name in ("review", "complete"):
            _string(item[name], f"{stage}.{name}")
        _hex(item["review_sha256"], f"{stage}.review_sha256")
    metadata = spec["metadata"]
    keys(metadata, ("path", "id_col", "patient_col", "label_col"), "metadata")
    for name in metadata:
        _string(metadata[name], f"metadata.{name}")
    require(len({metadata[name] for name in ("id_col", "patient_col", "label_col")}) == 3,
            "metadata columns must be distinct")
    _list(spec["panel"], "panel")
    require(bool(spec["panel"]), "panel must be nonempty")
    slides, patients = set(), set()
    for item in spec["panel"]:
        keys(item, ("slide_id", "patient_id", "group", "virchow_stratum",
                    "annotation_classes", "control_tags"), "panel entry")
        sid = safe_id(item["slide_id"], "panel.slide_id")
        _string(item["patient_id"], f"panel {sid} patient_id")
        _string(item["group"], f"panel {sid} group")
        _string(item["virchow_stratum"], f"panel {sid} virchow_stratum")
        require(sid not in slides and item["patient_id"] not in patients,
                f"panel {sid}: duplicate slide or patient")
        slides.add(sid)
        patients.add(item["patient_id"])
        for field in ("annotation_classes", "control_tags"):
            _list(item[field], f"panel {sid} {field}")
        require(all(type(v) is int and 1 <= v <= 5 for v in item["annotation_classes"]),
                f"panel {sid}: invalid class")
        require(all(type(v) is str and v in CONTROL_TAGS for v in item["control_tags"]),
                f"panel {sid}: unknown control tag")
        for field in ("annotation_classes", "control_tags"):
            require(item[field] == sorted(set(item[field])),
                    f"panel {sid}: {field} must be sorted and unique")
    panel_ids = [p["slide_id"] for p in spec["panel"]]
    expected = spec["expected"]
    keys(expected, ("panel_sha256", "slides", "patients", "universes", "tile_rows",
                    "feature_rows", "feature_files", "array_files",
                    "candidate_files_present", "candidate_files_absent",
                    "lineage_pairs", "array_payload_bytes"), "expected")
    for name in expected:
        if name != "panel_sha256":
            _int(expected[name], f"expected.{name}")
    _hex(expected["panel_sha256"], "expected.panel_sha256")
    require(hashlib.sha256(compact(panel_ids)).hexdigest() == expected["panel_sha256"]
            and expected["slides"] == len(panel_ids)
            and expected["patients"] == len(patients)
            and expected["array_files"] == 8,
            "panel digest/count or array count mismatch")
    if expected["slides"] == 12:
        require(tuple(panel_ids) == PANEL_PRODUCTION
                and expected["panel_sha256"] == PANEL_DIGEST
                and (expected["universes"], expected["tile_rows"],
                     expected["feature_rows"], expected["feature_files"],
                     expected["candidate_files_present"], expected["candidate_files_absent"],
                     expected["lineage_pairs"], expected["array_payload_bytes"])
                == (19, 152, 440, 55, 30, 54, 42, 2678784),
                "production panel membership/counts disagree with frozen contract")
    _list(spec["banks"], "banks")
    require(len(spec["banks"]) == 6, "banks: require six encoder/universe roles")
    roles = set()
    for item in spec["banks"]:
        keys(item, ("name", "universe", "path", "feature_dim", "dtypes"), "bank")
        role = item["universe"], item["name"]
        require(role[0] in ("main", "annotation") and role[1] in BANKS and role not in roles,
                f"bank: unknown/duplicate role {role}")
        roles.add(role)
        _string(item["path"], "bank.path")
        require(type(item["feature_dim"]) is int
                and item["feature_dim"] == (MAIN_DIM if role[0] == "main"
                                           else ANNOTATION_DIM)[role[1]],
                f"bank {role}: wrong dimension")
        _list(item["dtypes"], f"bank {role} dtypes")
        require(bool(item["dtypes"]) and len(item["dtypes"]) == len(set(item["dtypes"]))
                and all(type(d) is str and d in ("float16", "float32")
                        for d in item["dtypes"]), f"bank {role}: unsupported dtype")
    require(roles == {(u, b) for u in ("main", "annotation") for b in BANKS},
            "banks: missing role")
    aux = spec["auxiliary"]
    keys(aux, ("slide_id", "path", "csv", "csv_sha256", "feature_dim", "dtype"), "auxiliary")
    require(aux["slide_id"] in slides and aux["feature_dim"] == 1536
            and type(aux["feature_dim"]) is int and aux["dtype"] == "float32",
            "auxiliary: invalid slide, dimension or dtype")
    for key in ("path", "csv"):
        _string(aux[key], f"auxiliary.{key}")
    _hex(aux["csv_sha256"], "auxiliary.csv_sha256")
    _list(spec["wsi"], "wsi")
    require(all(type(w) is dict for w in spec["wsi"]), "wsi: expected objects")
    require([w.get("slide_id") for w in spec["wsi"]] == panel_ids, "wsi order mismatch")
    for item in spec["wsi"]:
        keys(item, ("slide_id", "path", "resolved_path", "size_bytes"), "wsi")
        for key in ("path", "resolved_path"):
            _string(item[key], f"wsi.{key}")
        _int(item["size_bytes"], "wsi.size_bytes", 1)
    _list(spec["models"], "models")
    require(all(type(m) is dict for m in spec["models"]), "models: expected objects")
    require([m.get("name") for m in spec["models"]] == list(BANKS),
            "model role/order mismatch")
    for item in spec["models"]:
        keys(item, ("name", "model_id", "revision", "config_path", "config_sha256",
                    "weight_path", "resolved_weight_path", "weight_size_bytes",
                    "expected_weight_sha256", "digest_basis"), "model")
        for key in ("model_id", "revision", "config_path", "weight_path",
                    "resolved_weight_path"):
            _string(item[key], f"model.{key}")
        _hex(item["config_sha256"], "model.config_sha256")
        _int(item["weight_size_bytes"], "model.weight_size_bytes", 1)
        require(item["digest_basis"] in ("cache_blob", "declaration", "observed_only"),
                "model digest_basis invalid")
        if item["digest_basis"] == "observed_only":
            require(item["expected_weight_sha256"] is None,
                    "observed-only weight cannot claim prior digest")
        else:
            _hex(item["expected_weight_sha256"], "model.expected_weight_sha256")
    _list(spec["virchow_candidates"], "virchow_candidates")
    require(all(type(r) is dict for r in spec["virchow_candidates"]),
            "virchow_candidates: expected objects")
    require([r.get("name") for r in spec["virchow_candidates"]] == list(ROOTS),
            "candidate root order mismatch")
    for root in spec["virchow_candidates"]:
        keys(root, ("name", "path", "expected_root_presence", "files"), "candidate root")
        _string(root["path"], "candidate root.path")
        require(type(root["expected_root_presence"]) is bool,
                "candidate root expected_root_presence must be bool")
        _list(root["files"], f"candidate {root['name']} files")
        require([f.get("slide_id") for f in root["files"]] == panel_ids,
                f"candidate {root['name']}: file declarations not panel-ordered")
        for file in root["files"]:
            keys(file, ("slide_id", "present", "size_bytes", "expected_sha256"),
                 "candidate file")
            require(type(file["present"]) is bool, "candidate present must be bool")
            if file["present"]:
                require(root["expected_root_presence"], "present file in absent root")
                _int(file["size_bytes"], "candidate.size_bytes", 1)
                if file["expected_sha256"] is not None:
                    _hex(file["expected_sha256"], "candidate.expected_sha256")
            else:
                require(file["size_bytes"] is None and file["expected_sha256"] is None,
                        "absent candidate must have null size/hash")
    _list(spec["lineage_pairs"], "lineage_pairs")
    pair_ids = set()
    for item in spec["lineage_pairs"]:
        keys(item, ("pair_id", "left_source_id", "right_source_id", "relation"),
             "lineage_pair")
        for key in item:
            _string(item[key], f"lineage_pair.{key}")
        require(item["pair_id"] not in pair_ids, "duplicate pair ID")
        pair_ids.add(item["pair_id"])
    require(len(pair_ids) == expected["lineage_pairs"], "pair count mismatch")
    _list(spec["log_controls"], "log_controls")
    tags = set()
    for item in spec["log_controls"]:
        keys(item, ("tag", "slide_id", "path", "sha256", "write_line"), "log_control")
        require(item["tag"] in CONTROL_TAGS and item["tag"] not in tags
                and item["slide_id"] in slides, "duplicate/unknown control tag or slide")
        tags.add(item["tag"])
        _string(item["path"], "control.path")
        _hex(item["sha256"], "control.sha256")
        _int(item["write_line"], "control.write_line", 1)
    require({tag for item in spec["panel"] for tag in item["control_tags"]} == tags,
            "control tag coverage mismatch")
    _list(spec["code_files"], "code_files")
    source_names = set()
    for item in spec["code_files"]:
        keys(item, ("name", "path", "sha256"), "code_file")
        safe_id(item["name"], "code_file.name")
        _string(item["path"], "code_file.path")
        _hex(item["sha256"], "code_file.sha256")
        require(item["name"] not in source_names and not item["name"].startswith("auto/"),
                "duplicate/reserved code name")
        source_names.add(item["name"])


def _bounded(path, limit):
    return annotation._read_capped(path, limit)


def _hash_stream(path, expected_size):
    _int(expected_size, f"{path} declared size", 1)
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while total < expected_size:
            block = handle.read(min(HASH_BLOCK, expected_size - total))
            require(block, f"{path}: early EOF after {total}/{expected_size} bytes")
            total += len(block)
            digest.update(block)
        require(not handle.read(1), f"{path}: exceeds declared {expected_size} bytes")
    return digest.hexdigest(), total


def _entry_exists(path):
    return os.path.lexists(path)


def _source(source_id, kind, path, *, bank="", universe="", slide_id="",
            expected_present=True, expected_size=None, expected_sha=None,
            basis="declaration", cap=OTHER_CAP, stream=False):
    path = Path(path)
    require(_entry_exists(path) == expected_present,
            f"{source_id}: declared presence differs from directory entry")
    observed_path, observed_size, observed_sha, data = "", None, "", None
    if expected_present:
        require(path.is_file(), f"{source_id}: declared present input not a regular file")
        observed_path = str(path.resolve(strict=True))
        if stream:
            require(expected_size is not None, f"{source_id}: streaming size required")
            observed_sha, observed_size = _hash_stream(path, expected_size)
        else:
            data = _bounded(path, cap)
            observed_size = len(data)
            observed_sha = hashlib.sha256(data).hexdigest()
        if expected_size is not None:
            require(observed_size == expected_size, f"{source_id}: size mismatch")
        if expected_sha is not None:
            require(observed_sha == expected_sha, f"{source_id}: SHA256 mismatch")
    return {
        "source_id": source_id, "kind": kind, "bank": bank, "universe": universe,
        "slide_id": slide_id, "path": str(path), "resolved_path": observed_path,
        "expected_presence": expected_present, "observed_presence": expected_present,
        "expected_size_bytes": expected_size, "observed_size_bytes": observed_size,
        "expected_sha256": expected_sha, "observed_sha256": observed_sha,
        "digest_basis": basis if expected_present else "absent",
    }, data


def _array(data, context, *, width=None, dtypes=("float16", "float32")):
    obj = bank_audit._pickle(data, context)
    require(type(obj) is tuple and len(obj) == 2,
            f"{context}: expected (features, ordered tiles) tuple")
    array, tiles = obj
    require(isinstance(array, np.ndarray) and array.ndim == 2
            and array.shape[1] in ((width,) if width is not None else (1280, 2560))
            and array.dtype.kind == "f" and array.dtype.name in dtypes
            and bool(np.isfinite(array).all()),
            f"{context}: invalid raw shape/dtype/nonfinite features")
    coords = bank_audit._coordinates(tiles, context)
    require(bool(coords) and len(set(coords)) == len(coords)
            and all(x >= 0 and y >= 0 for x, y in coords)
            and len(coords) == array.shape[0], f"{context}: coordinates/rows invalid")
    return array, coords


def _physical(sid, xy):
    return sid, 0, xy[0], xy[1], 512


def _rank(key):
    return hashlib.sha256(compact(["r2_1_identity_tile_v1", SEED, *key])).hexdigest(), key


def _choose(coords, sid, *, annotations=None, main_coords=(), reserved=()):
    """Return (coordinate, insertion reason) in fixed selection order."""
    require(len(coords) == len(set(coords)), "duplicate physical coordinate")
    available = set(coords)
    chosen = []
    selected = set()

    def add(xy, reason):
        if xy in available and xy not in selected and len(chosen) < 8:
            selected.add(xy)
            chosen.append((xy, reason))

    if annotations is not None:
        labels, classes = annotations
        for cls in classes:
            possibilities = [xy for xy, lab in labels.items() if lab == cls]
            require(possibilities, f"class {cls}: no eligible coordinate")
            for xy in sorted(possibilities, key=lambda p:
                             (p not in main_coords, _rank(_physical(sid, p)))):
                add(xy, "class_reservation")
                if xy in selected:
                    break
        for xy in sorted(available - selected, key=lambda p:
                         (p not in main_coords, _rank(_physical(sid, p)))):
            add(xy, "hash_fill")
    else:
        for xy in reserved:
            add(xy, "intersection")
        for xy in (coords[0], coords[len(coords) // 2], coords[-1]):
            add(xy, "landmark")
        for xy in sorted(available - selected, key=lambda p: _rank(_physical(sid, p))):
            add(xy, "hash_fill")
    require(len(chosen) == 8, f"panel universe {sid}: insufficient distinct coordinates")
    return chosen


def _numeric_npy(data, expected_shape=None, expected_dtype=None):
    require(len(data) <= OUTPUT_NPY_CAP, "numeric NPY: exceeds 16MiB")
    stream = io.BytesIO(data)
    try:
        require(np.lib.format.read_magic(stream) == (1, 0), "numeric NPY: require version 1.0")
        begin = stream.tell()
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(
            stream, max_header_size=4096)
        require(stream.tell() - begin <= 4096 and len(shape) == 2
                and all(type(n) is int and n > 0 for n in shape) and not fortran,
                "numeric NPY: invalid shape, header or layout")
        require(dtype.str in ("<f2", "<f4") and not dtype.hasobject,
                "numeric NPY: require little-endian float16/float32")
        require(expected_shape is None or tuple(shape) == tuple(expected_shape),
                "numeric NPY: wrong shape")
        require(expected_dtype is None or dtype.str == expected_dtype,
                "numeric NPY: wrong dtype")
        require(len(data) - stream.tell() == shape[0] * shape[1] * dtype.itemsize,
                "numeric NPY: truncated or trailing payload")
        values = np.frombuffer(data, dtype=dtype, count=shape[0] * shape[1],
                               offset=stream.tell()).reshape(shape)
        require(bool(np.isfinite(values).all()), "numeric NPY: nonfinite values")
        return values
    except AuditError:
        raise
    except (ValueError, EOFError, SyntaxError, TypeError) as exc:
        raise AuditError(f"numeric NPY: malformed header: {exc}") from exc


def _upstream(stage, declared, base, review_data, entries):
    review_path = path_of(declared["review"], base)
    review = bank_audit._json(review_data)
    scope = ("Stage0 feature-bank binding only" if stage == "stage0" else
             "Stage1a historical-annotation target keyed binding only")
    complete_scope = ("feature-bank binding only" if stage == "stage0" else
                      "historical-annotation target keyed binding only")
    require(review.get("schema_version") == 1 and review.get("disposition") == "accepted"
            and review.get("scope") == scope and type(review.get("artifacts")) is dict,
            f"{stage}: review disposition/scope/artifacts invalid")
    review_id = f"upstream/{stage}/review"
    review_entry, _ = _source(review_id, "upstream", review_path,
                              expected_sha=declared["review_sha256"])
    entries[review_id] = review_entry
    artifacts = {}
    for rel, digest in sorted(review["artifacts"].items()):
        _hex(digest, f"{stage} artifact {rel}")
        require(type(rel) is str and not Path(rel).is_absolute()
                and all(part != ".." for part in Path(rel).parts),
                f"{stage}: unsafe artifact path {rel}")
        artifact_path = path_of(rel, review_path.parent)
        require(bank_audit._inside(artifact_path.resolve(strict=True),
                                   review_path.parent.resolve(strict=True)),
                f"{stage}: artifact escapes reviewed root")
        source_id = f"upstream/{stage}/{rel}"
        data, observed = None, None
        # The eligible manifest is checked through the bounded raw-record stream later.
        if stage == "stage1a" and rel.endswith("/eligible_tiles.csv"):
            require(artifact_path.is_file(), "Stage1a eligible manifest absent")
            observed = {"source_id": source_id, "kind": "upstream", "path": str(artifact_path),
                        "resolved_path": str(artifact_path.resolve(strict=True)),
                        "expected_presence": True, "observed_presence": True,
                        "expected_sha256": digest, "observed_sha256": "",
                        "expected_size_bytes": None, "observed_size_bytes": None,
                        "digest_basis": stage, "bank": "", "universe": "", "slide_id": ""}
        else:
            observed, data = _source(source_id, "upstream", artifact_path,
                                     expected_sha=digest, basis=stage)
        entries[source_id] = observed
        artifacts[rel] = (observed, data)
    complete_path = path_of(declared["complete"], base)
    complete_match = [(name, row) for name, row in artifacts.items()
                      if row[0]["resolved_path"] == str(complete_path.resolve(strict=True))]
    require(len(complete_match) == 1 and complete_match[0][0].endswith("/COMPLETE.json"),
            f"{stage}: selected COMPLETE not reviewed")
    complete = bank_audit._json(complete_match[0][1][1])
    require(complete.get("schema_version") == 1 and complete.get("scope") == complete_scope
            and type(complete.get("outputs")) is dict,
            f"{stage}: invalid COMPLETE scope/outputs")
    require(set(complete["outputs"]) == {
        rel.rsplit("/", 1)[-1] for rel in artifacts
        if rel.startswith(complete_match[0][0].rsplit("/", 1)[0] + "/")
        and not rel.endswith("/COMPLETE.json")},
        f"{stage}: reviewed COMPLETE does not cover exact output set")
    for filename, identity in complete["outputs"].items():
        found = [(rel, row) for rel, row in artifacts.items()
                 if row[0]["resolved_path"] == str((complete_path.parent / filename).resolve(strict=True))]
        require(len(found) == 1 and set(identity) == {"sha256", "size"}
                and identity["sha256"] == found[0][1][0]["expected_sha256"]
                and (filename == "eligible_tiles.csv" and stage == "stage1a"
                     or identity["size"] == found[0][1][0]["observed_size_bytes"]),
                f"{stage}: COMPLETE output link mismatch {filename}")
        if filename == "eligible_tiles.csv" and stage == "stage1a":
            found[0][1][0]["expected_size_bytes"] = identity["size"]
    summaries = [data for rel, (_, data) in artifacts.items()
                 if rel.endswith("/summary.json")]
    require(len(summaries) == 1, f"{stage}: missing reviewed summary")
    summary = bank_audit._json(summaries[0])
    require(summary.get("run_id") == complete.get("run_id"),
            f"{stage}: reviewed summary/COMPLETE run mismatch")
    return review, complete, summary, artifacts


def _stream_eligible(path, expected_digest, expected_count, panel):
    rows = defaultdict(dict)
    selected = {item["slide_id"]: item for item in panel}
    digest, consumed = hashlib.sha256(), 0

    class Lines:
        def __init__(self, handle):
            self.handle = handle

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal consumed
            line = self.handle.readline(1024 * 1024 + 1)
            if not line:
                raise StopIteration
            require(len(line) <= 1024 * 1024, "eligible CSV: oversized record line")
            consumed += len(line)
            require(consumed <= OTHER_CAP, "eligible CSV: raw bytes exceed 128MiB")
            digest.update(line)
            return line.decode("utf-8-sig" if consumed == len(line) else "utf-8")

    with path.open("rb") as handle:
        reader = csv.reader(Lines(handle))
        require(tuple(next(reader, ())) == annotation.ELIGIBLE_HEADER,
                "Stage1a eligible manifest: wrong header")
        count = 0
        for values in reader:
            require(len(values) == len(annotation.ELIGIBLE_HEADER),
                    "Stage1a eligible manifest: malformed record")
            count += 1
            row = dict(zip(annotation.ELIGIBLE_HEADER, values))
            sid = row["slide_id"]
            if sid not in selected:
                continue
            group = selected[sid]
            require((row["patient_id"], row["diagnostic_group"])
                    == (group["patient_id"], group["group"]),
                    f"Stage1a eligible {sid}: patient/group mismatch")
            require(decimal(row["level"], "eligible level") == 0
                    and decimal(row["patch_size"], "eligible patch_size") == 512,
                    f"Stage1a eligible {sid}: physical scale mismatch")
            xy = (decimal(row["x"], "eligible x"), decimal(row["y"], "eligible y"))
            key = _physical(sid, xy)
            require(row["tile_id"] == annotation._tile_id(key),
                    f"Stage1a eligible {sid}: tile-ID mismatch")
            label = decimal(row["tissue_label"], "eligible label", 5)
            require(label >= 1 and xy not in rows[sid],
                    f"Stage1a eligible {sid}: invalid/duplicate tile")
            indices = {
                bank: decimal(row[f"{bank}_feature_row"],
                              f"Stage1a eligible {sid} {bank} feature row")
                for bank in BANKS}
            rows[sid][xy] = (label, indices)
    require(count == expected_count and digest.hexdigest() == expected_digest,
            "Stage1a eligible manifest: row count or raw-byte SHA256 mismatch")
    return rows, consumed


def _root_entry(source_id, root, present):
    require(_entry_exists(root) == present, f"{source_id}: root presence changed")
    if present:
        require(root.is_dir(), f"{source_id}: root is not a directory")
    return {
        "source_id": source_id, "kind": "root", "bank": "", "universe": "",
        "slide_id": "", "path": str(root),
        "resolved_path": str(root.resolve(strict=True)) if present else "",
        "expected_presence": present, "observed_presence": present,
        "expected_size_bytes": None, "observed_size_bytes": None,
        "expected_sha256": None, "observed_sha256": "",
        "digest_basis": "observed_only" if present else "absent",
    }


def _csv_bytes(header, rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _inventory_rows(entries):
    result = []
    for source_id in sorted(entries):
        entry = entries[source_id]
        result.append(tuple(
            ("true" if value else "false") if type(value) is bool
            else "" if value is None else str(value)
            for value in (entry[k] for k in SOURCE_HEADER)))
    return result


def _array_name(universe, bank, dtype):
    if bank == "virchow2":
        return ("virchow2_annotation_full.npy" if universe == "annotation"
                else f"virchow2_main_{'f16' if dtype == 'float16' else 'f32'}.npy")
    return f"{bank}_{universe}.npy"


def _expected_pairs(spec, candidate):
    panel = [item["slide_id"] for item in spec["panel"]]
    present = {name: {p["slide_id"] for p in item["files"] if p["present"]}
               for name, item in candidate.items()}
    groups = (
        ("C", "main", panel, "candidate_copy"),
        ("O", "C", present["O"], "candidate_copy"),
        ("F3", "C", present["F3"], "cls_projection"),
        ("F4", "P4", present["F4"], "cls_projection"),
        ("P4", "C", present["P4"], "candidate_copy"),
        ("O", "F4", present["O"] & present["F4"], "old_vs_new_diagnostic"),
        ("F3", "P3", present["F3"], "cls_projection"),
        ("P3", "C", present["F3"], "candidate_copy"),
        ("T", "F3", present["F3"], "candidate_generation_stage"),
        ("T", "F4", present["F4"], "candidate_generation_stage"),
    )
    return [
        {"pair_id": f"{left}__{right}__{sid}",
         "left_source_id": f"candidate/{left}/{sid}",
         "right_source_id": (f"feature/main/virchow2/{sid}" if right == "main"
                             else f"candidate/{right}/{sid}"),
         "relation": relation}
        for left, right, membership, relation in groups
        for sid in panel if sid in membership
    ]


def _pair_report(pair, entries):
    left, right = (entries[pair[field]] for field in
                   ("left_source_id", "right_source_id"))
    result = dict(pair)
    result.update({key: None for key in PAIR_KEYS if key not in result})
    left_ok, right_ok = left["observed_presence"], right["observed_presence"]
    arrays = []
    for source in (left, right):
        if not source["observed_presence"]:
            arrays.append(None)
            continue
        blob = _bounded(Path(source["path"]), FEATURE_CAP)
        require(hashlib.sha256(blob).hexdigest() == source["observed_sha256"],
                f"{source['source_id']}: changed during pair read")
        arrays.append(_array(blob, source["source_id"]))
        del blob
    for side, observed in zip(("left", "right"), arrays):
        if observed is not None:
            arr, coords = observed
            result.update({f"{side}_rows": len(coords),
                           f"{side}_dimension": arr.shape[1],
                           f"{side}_dtype": arr.dtype.str})
    if not left_ok or not right_ok:
        result.update(status="non_comparable", reason=(
            "declared_missing_both" if not left_ok and not right_ok
            else "declared_missing_left" if not left_ok else "declared_missing_right"))
        return result
    (la, lc), (ra, rc) = arrays
    common = set(lc) & set(rc)
    result.update(common_keys=len(common), left_only_keys=len(set(lc) - common),
                  right_only_keys=len(set(rc) - common), order_equal=(lc == rc))
    if not common or min(la.shape[1], ra.shape[1]) < 1280:
        result.update(status="non_comparable",
                      reason="no_common_keys" if not common else "projection_unavailable")
        return result
    result.update(status="comparable", reason=None)
    li, ri = {xy: i for i, xy in enumerate(lc)}, {xy: i for i, xy in enumerate(rc)}
    exact_diff = bits_diff = cast_diff = 0
    for xy in sorted(common):
        l, r = la[li[xy], :1280], ra[ri[xy], :1280]
        exact_diff += not bool(np.array_equal(l.astype(np.float64), r.astype(np.float64)))
        left_bytes = l.astype(l.dtype.newbyteorder("<"), copy=False).tobytes()
        right_bytes = r.astype(r.dtype.newbyteorder("<"), copy=False).tobytes()
        bits_diff += l.dtype.itemsize != r.dtype.itemsize or left_bytes != right_bytes
        cast_diff += (l.astype(r.dtype.newbyteorder("<")).tobytes() != right_bytes)
    result.update(exact_value_equal=not exact_diff, bitwise_equal=not bits_diff,
                  destination_cast_equal=not cast_diff,
                  exact_value_mismatch_rows=exact_diff,
                  bitwise_mismatch_rows=bits_diff, cast_mismatch_rows=cast_diff)
    return result


def _production_contract(spec):
    if spec["expected"]["slides"] != 12:
        return
    classes = {"7483": [2, 3, 4], "8874": [1, 5], "5605": [1],
               "15825": [1, 3], "6798": [1], "16973": [4]}
    groups = ({"7483", "8874", "10003", "5605", "15825", "6798"},
              {"6677", "17289"}, {"16973", "15841"}, {"19162", "19125"})
    declared = defaultdict(set)
    for item in spec["panel"]:
        declared[item["virchow_stratum"]].add(item["slide_id"])
    require(len(declared) == 4 and {frozenset(g) for g in declared.values()}
            == {frozenset(g) for g in groups},
            "production candidate-origin stratum memberships differ")
    for panel in spec["panel"]:
        require(panel["annotation_classes"] == classes.get(panel["slide_id"], []),
                f"production annotation classes {panel['slide_id']}: wrong coverage")
    tags = {"main_canary": "10003", "auxiliary_canary": "10003",
            "recovery_a": "15825", "recovery_b": "6798",
            "main_array": "7483", "annotation_array": "7483"}
    require({item["tag"]: item["slide_id"] for item in spec["log_controls"]} == tags,
            "production log-control tag/slide mismatch")
    declared = {root["name"]: {f["slide_id"] for f in root["files"] if f["present"]}
                for root in spec["virchow_candidates"]}
    require(declared == {
        "O": {"7483", "8874", "10003", "5605", "15825", "6798", "16973", "15841"},
        "F3": {"6677", "17289"},
        "F4": {"16973", "15841", "19162", "19125"},
        "P3": set(), "P4": {"16973", "15841", "19162", "19125"},
        "C": set(PANEL_PRODUCTION), "T": set()},
        "production candidate membership differs from measured declaration")
    require({r["name"] for r in spec["virchow_candidates"]
             if not r["expected_root_presence"]} == {"P3", "T"},
            "production absent roots differ from measured declaration")
    require(spec["auxiliary"]["slide_id"] == "10003"
            and spec["auxiliary"]["csv_sha256"]
            == "ac1fefd2b0fc9172d0a497ccfcdb74dc8c7a6f68311d16de1f74f2a5847f840c",
            "production auxiliary control declaration differs")


def _read_metadata(data, spec, panel):
    metadata = annotation._metadata(data, spec["metadata"])
    for item in panel:
        require(item["slide_id"] in metadata
                and metadata[item["slide_id"]] == (item["patient_id"], item["group"]),
                f"metadata: mismatched panel patient/group {item['slide_id']}")
    return metadata


def _reference(root, universe, bank, sid, accepted, entries):
    source_id = f"feature/{universe}/{bank}/{sid}"
    expected = accepted[(bank, sid)]
    path = root / f"{sid}.pkl"
    require(str(path.resolve(strict=True)) == expected["resolved_path"]
            and str(path) == expected["path"],
            f"{source_id}: accepted lexical/resolved path mismatch")
    row, blob = _source(source_id, "feature", path, bank=bank, universe=universe,
                        slide_id=sid, expected_size=int(expected["size"]),
                        expected_sha=expected["sha256"], basis="stage0" if universe == "main"
                        else "stage1a", cap=FEATURE_CAP)
    entries[source_id] = row
    width = MAIN_DIM[bank] if universe == "main" else ANNOTATION_DIM[bank]
    array, coords = _array(blob, source_id, width=width)
    require(array.dtype.name == expected["dtype"], f"{source_id}: accepted dtype mismatch")
    if "source_row_count" in expected:
        require(len(coords) == int(expected["source_row_count"]),
                f"{source_id}: accepted row count mismatch")
    del blob
    return array, coords


def _emit_vectors(panel, universe, bank_roots, accepted, entries, bank_declarations,
                  selected_arrays, references, annotation_rows=None):
    for sid, chosen in panel.items():
        for bank in BANKS:
            root = bank_roots[(universe, bank)]
            array, coords = _reference(root, universe, bank, sid,
                                        accepted[universe], entries)
            config = next(r for r in bank_declarations
                          if (r["universe"], r["name"]) == (universe, bank))
            require(array.dtype.name in config["dtypes"],
                    f"{universe}/{bank}/{sid}: precision not declared")
            require((universe == "main" and bank == "virchow2")
                    or array.dtype.name == "float32",
                    f"{universe}/{bank}/{sid}: source precision cannot be promoted")
            lookup = {xy: index for index, xy in enumerate(coords)}
            require(len(lookup) == len(coords), f"{universe}/{bank}/{sid}: duplicate tile")
            for xy, _ in chosen:
                require(xy in lookup, f"{universe}/{bank}/{sid}: selected tile missing {xy}")
                arrname = _array_name(universe, bank, array.dtype.name)
                source_row = lookup[xy]
                if annotation_rows is not None:
                    require(source_row == annotation_rows[sid][xy][1][bank],
                            f"annotation/{bank}/{sid}/{xy}: Stage1a feature row "
                            "disagrees with physical-coordinate lookup")
                arrrow = len(selected_arrays[arrname])
                dtype = "<f2" if array.dtype.itemsize == 2 else "<f4"
                selected_arrays[arrname].append(
                    array[source_row].astype(dtype).tobytes())
                references.append((bank, universe, sid, annotation._tile_id(_physical(sid, xy)),
                                   f"feature/{universe}/{bank}/{sid}", source_row,
                                   array.shape[1], array.dtype.str, arrname, arrrow))
            del array, coords, lookup


def _selected_pairs(spec, entries):
    candidate = {c["name"]: c for c in spec["virchow_candidates"]}
    require(spec["lineage_pairs"] == _expected_pairs(spec, candidate),
            "lineage_pairs: expected group order/roles/membership mismatch")
    records = []
    for pair in spec["lineage_pairs"]:
        require(pair["left_source_id"] in entries and pair["right_source_id"] in entries,
                f"pair {pair['pair_id']}: undeclared source role")
        records.append(_pair_report(pair, entries))
    counts = {"pairs": len(records),
              "comparable": sum(p["status"] == "comparable" for p in records),
              "non_comparable": sum(p["status"] == "non_comparable" for p in records)}
    for name, key in (("exact_value_different", "exact_value_equal"),
                      ("bitwise_different", "bitwise_equal"),
                      ("destination_cast_different", "destination_cast_equal")):
        counts[name] = sum(p["status"] == "comparable" and not p[key] for p in records)
    require(tuple(counts) == PAIR_COUNT_KEYS, "lineage count keys invalid")
    return {"schema_version": 1, "run_id": spec["run_id"],
            "scope": "selected_files_all_common_coordinates",
            "pairs": records, "counts": counts}


def _parse_write_line(line, sid, source_id):
    match = WRITE_LINE.fullmatch(line)
    require(match is not None and match["slide"] == sid
            and all(math.isfinite(float(match[field]))
                    and float(match[field]) > 0 for field in ("seconds", "rate")),
            f"{source_id}: declared line is not a successful structured "
            f"[extract] write for {sid}")


def _bind_write_controls(spec, base, entries):
    verified = {}
    for item in spec["log_controls"]:
        source_id = f"control/{item['tag']}"
        row, data = _source(source_id, "control", path_of(item["path"], base),
                            slide_id=item["slide_id"], expected_sha=item["sha256"])
        entries[source_id] = row
        lines = data.decode("utf-8").splitlines()
        require(len(lines) >= item["write_line"],
                f"{source_id}: declared write line out of range")
        _parse_write_line(lines[item["write_line"] - 1], item["slide_id"], source_id)
        verified[item["tag"]] = item["slide_id"]
        del data
    declared = [(tag, p["slide_id"]) for p in spec["panel"] for tag in p["control_tags"]]
    require(len(declared) == len(verified)
            and len({tag for tag, _ in declared}) == len(declared)
            and dict(declared) == verified,
            "panel control tag/slide assignment differs from verified write controls")
    return verified


def _prepare(spec, base, entries):
    reviewed = {}
    accepted = {}
    metadata_shas = []
    metadata_bindings = []
    for stage in ("stage0", "stage1a"):
        config = spec[stage]
        review_path = path_of(config["review"], base)
        review_bytes = _bounded(review_path, OTHER_CAP)
        require(hashlib.sha256(review_bytes).hexdigest() == config["review_sha256"],
                f"{stage}: review SHA256 mismatch")
        review, complete, summary, artifacts = _upstream(
            stage, config, base, review_bytes, entries)
        reviewed[stage] = {"review_sha256": config["review_sha256"],
                           "complete_sha256": hashlib.sha256(
                               _bounded(path_of(config["complete"], base), OTHER_CAP)).hexdigest()}
        metadata_shas.append(review["source_sha256"]["metadata"])
        metadata_bindings.append(next(b for b in summary["input_bindings"]
                                      if b["kind"] == "metadata"
                                      and b["name"] == "metadata"))
        if stage == "stage0":
            require(review["source_sha256"]["auditor_module"]
                    == hashlib.sha256(Path(bank_audit.__file__).read_bytes()).hexdigest(),
                    "Stage0 frozen helper differs from accepted code")
            inventory = next(blob for name, (_, blob) in artifacts.items()
                             if name.endswith("/inventory.csv"))
            records = csv.DictReader(io.StringIO(inventory.decode("utf-8")))
            require(tuple(records.fieldnames or ()) == (
                "bank", "slide_id", "canonical_tiles", "feature_rows", "dtype",
                "feature_path", "resolved_path", "size", "sha256"), "Stage0 inventory header")
            accepted["main"] = {(r["bank"], r["slide_id"]):
                                {"path": r["feature_path"], "resolved_path": r["resolved_path"],
                                 "size": r["size"], "sha256": r["sha256"], "dtype": r["dtype"]}
                                for r in records}
        else:
            require(review["source_sha256"]["auditor_module"]
                    == hashlib.sha256(Path(annotation.__file__).read_bytes()).hexdigest()
                    and review["source_sha256"]["stage0_helper_module"]
                    == hashlib.sha256(Path(bank_audit.__file__).read_bytes()).hexdigest()
                    and review["source_sha256"]["stage0_independent_review"]
                    == reviewed["stage0"]["review_sha256"],
                    "Stage1a frozen helpers or Stage0 review link differ")
            inventory = next(blob for name, (_, blob) in artifacts.items()
                             if name.endswith("/feature_inventory.csv"))
            records = csv.DictReader(io.StringIO(inventory.decode("utf-8")))
            require(tuple(records.fieldnames or ()) == annotation.FEATURE_HEADER,
                    "Stage1a feature inventory header")
            accepted["annotation"] = {
                (r["bank"], r["slide_id"]):
                {"path": r["feature_path"], "resolved_path": r["resolved_path"],
                 "size": r["size"], "sha256": r["sha256"], "dtype": r["dtype"],
                 "source_row_count": r["feature_rows"]}
                for r in records}
            eligible_file = next(row for name, row in artifacts.items()
                                 if name.endswith("/eligible_tiles.csv"))
            eligible_binding = eligible_file[0]
            expected_eligible_rows = summary["eligible"]["rows"]
        require(review["source_sha256"]["metadata"]
                == next(b["sha256"] for b in summary["input_bindings"]
                        if b["kind"] == "metadata" and b["name"] == "metadata"),
                f"{stage}: metadata binding mismatch")
        require({k: summary["declarations"]["metadata"][k]
                 for k in ("id_col", "patient_col", "label_col")}
                == {k: spec["metadata"][k] for k in
                    ("id_col", "patient_col", "label_col")},
                f"{stage}: metadata selector mismatch")
    require(reviewed["stage0"] and reviewed["stage1a"], "both accepted reviews required")
    metadata_path = path_of(spec["metadata"]["path"], base)
    require(metadata_shas[0] == metadata_shas[1]
            and all(binding["sha256"] == metadata_shas[0]
                    and binding["path"] == str(metadata_path)
                    and binding["resolved_path"] == str(metadata_path.resolve(strict=True))
                    for binding in metadata_bindings),
            "Stage0/Stage1a metadata path or bytes differ")
    meta_source, meta_bytes = _source(
        "metadata", "metadata", metadata_path,
        expected_sha=metadata_shas[0], basis="stage0")
    entries["metadata"] = meta_source
    _read_metadata(meta_bytes, spec, spec["panel"])
    del meta_bytes
    panel = spec["panel"]
    panel_ids = [p["slide_id"] for p in panel]
    chosen_by_slide = {p["slide_id"]: p for p in panel}
    annotation_rows, raw_bytes = _stream_eligible(
        Path(eligible_binding["path"]), eligible_binding["expected_sha256"],
        expected_eligible_rows, panel)
    eligible_binding["observed_sha256"] = eligible_binding["expected_sha256"]
    eligible_binding["observed_size_bytes"] = raw_bytes
    require(raw_bytes == eligible_binding["expected_size_bytes"],
            "Stage1a eligible manifest: accepted size mismatch")
    for item in panel:
        sid = item["slide_id"]
        if item["annotation_classes"]:
            labels = {xy: code for xy, (code, _) in annotation_rows[sid].items()}
            require(sorted(set(labels.values())) == item["annotation_classes"],
                    f"panel {sid}: declared annotation class presence mismatch")
        else:
            require(sid not in annotation_rows, f"panel {sid}: unexpected annotation rows")

    bank_roots = {(r["universe"], r["name"]): path_of(r["path"], base)
                  for r in spec["banks"]}
    by_root = defaultdict(set)
    for (universe, bank), root in bank_roots.items():
        by_root[str(root.resolve(strict=True))].add(bank)
    require(all(len(banks) == 1 for banks in by_root.values()),
            "bank roots for different encoders alias")
    for (universe, bank), root in bank_roots.items():
        root_id = f"root/{universe}/{bank}"
        entries[root_id] = _root_entry(root_id, root, True)
    main_coords = {}
    for sid in panel_ids:
        array, coords = _reference(bank_roots[("main", "uni2")], "main", "uni2", sid,
                                    accepted["main"], entries)
        main_coords[sid] = coords
        del array
    aux = spec["auxiliary"]
    aux_id = aux["slide_id"]
    csv_path = path_of(aux["csv"], base)
    csv_entry, csv_data = _source("control/auxiliary_csv", "control", csv_path,
                                  slide_id=aux_id, expected_sha=aux["csv_sha256"])
    entries[csv_entry["source_id"]] = csv_entry
    aux_order, _ = annotation._source_csv(csv_data, aux_id, False, "auxiliary CSV")
    del csv_data
    aux_path = path_of(aux["path"], base)
    aux_entry, aux_blob = _source("feature/auxiliary/h_optimus_1/" + aux_id,
                                  "feature", aux_path, bank="h_optimus_1",
                                  universe="auxiliary", slide_id=aux_id, cap=FEATURE_CAP,
                                  basis="observed_only")
    entries[aux_entry["source_id"]] = aux_entry
    aux_array, aux_coords = _array(aux_blob, "auxiliary", width=1536, dtypes=("float32",))
    require(aux_array.dtype.name == "float32" and aux_order == aux_coords,
            "auxiliary tuple/CSV full ordered-coordinate mismatch")
    del aux_blob
    annotated = {}
    for item in panel:
        sid = item["slide_id"]
        if item["annotation_classes"]:
            labels = {xy: value[0] for xy, value in annotation_rows[sid].items()}
            annotated[sid] = _choose(list(labels), sid,
                                     annotations=(labels, item["annotation_classes"]),
                                     main_coords=set(main_coords[sid]))
    main = {sid: _choose(main_coords[sid], sid,
                          reserved=[xy for xy, _ in annotated.get(sid, ())
                                    if xy in set(main_coords[sid])])
            for sid in panel_ids}
    aux_selected = _choose(aux_coords, aux_id,
                           reserved=[xy for xy, _ in main[aux_id] if xy in set(aux_coords)])
    universes = ([("main", sid, main[sid]) for sid in panel_ids]
                 + [("annotation", sid, annotated[sid]) for sid in panel_ids if sid in annotated]
                 + [("auxiliary", aux_id, aux_selected)])
    panel_rows, unique_keys = [], set()
    for universe, sid, chosen in universes:
        item = chosen_by_slide[sid]
        for index, (xy, reason) in enumerate(chosen):
            key = _physical(sid, xy)
            unique_keys.add(key)
            tissue = annotation_rows[sid][xy][0] if universe == "annotation" else ""
            panel_rows.append((universe, sid, item["patient_id"], item["group"],
                               index, 0, *xy, 512, annotation._tile_id(key), tissue, reason))
    require(len(universes) == spec["expected"]["universes"]
            and len(panel_rows) == spec["expected"]["tile_rows"],
            "declared universe/coordinate counts do not match fixed selection")
    selected_arrays = {name: [] for name in ARRAYS}
    references = []
    aux_lookup = {xy: i for i, xy in enumerate(aux_coords)}
    for xy, _ in aux_selected:
        row = aux_lookup[xy]
        name = "h_optimus_1_auxiliary.npy"
        selected_arrays[name].append(aux_array[row].astype("<f4").tobytes())
        references.append(("h_optimus_1", "auxiliary", aux_id,
                           annotation._tile_id(_physical(aux_id, xy)), aux_entry["source_id"],
                           row, 1536, aux_array.dtype.str, name,
                           len(selected_arrays[name]) - 1))
    del aux_array, aux_coords, aux_lookup
    _emit_vectors(main, "main", bank_roots, accepted, entries, spec["banks"],
                  selected_arrays, references)
    _emit_vectors(annotated, "annotation", bank_roots, accepted, entries, spec["banks"],
                  selected_arrays, references, annotation_rows)
    del main_coords, annotation_rows
    require(len(references) == spec["expected"]["feature_rows"]
            and len({row[4] for row in references}) == spec["expected"]["feature_files"],
            "selected feature reference/file counts mismatch")
    reference_index = {(r[1], r[2], r[3], r[0]): r for r in references}
    require(len(reference_index) == len(references), "duplicate feature reference")
    references = [reference_index[(row[0], row[1], row[9], bank)]
                  for row in panel_rows
                  for bank in (BANKS if row[0] != "auxiliary" else ("h_optimus_1",))]
    return entries, reviewed, panel_rows, references, selected_arrays, unique_keys, universes


def _remaining_sources(spec, base, entries):
    for name, root in ((r["name"], r) for r in spec["virchow_candidates"]):
        root_path = path_of(root["path"], base)
        root_id = f"root/candidate/{name}"
        entries[root_id] = _root_entry(root_id, root_path, root["expected_root_presence"])
        for file in root["files"]:
            sid = file["slide_id"]
            source_id = f"candidate/{name}/{sid}"
            row, _ = _source(source_id, "candidate", root_path / f"{sid}.pkl",
                             bank="virchow2", slide_id=sid,
                             expected_present=file["present"],
                             expected_size=file["size_bytes"],
                             expected_sha=file["expected_sha256"],
                             basis="declaration" if file["expected_sha256"]
                             else "observed_only", cap=FEATURE_CAP)
            entries[source_id] = row
            if file["present"]:
                # A declared present malformed candidate is a hard error,
                # including candidates that have no selected panel coordinate.
                blob = _bounded(Path(row["path"]), FEATURE_CAP)
                require(hashlib.sha256(blob).hexdigest() == row["observed_sha256"],
                        f"{source_id}: changed before parse")
                arr, coords = _array(blob, source_id)
                del blob, arr, coords
    for item in spec["wsi"]:
        sid = item["slide_id"]
        path = path_of(item["path"], base)
        require(str(path.resolve(strict=True)) == item["resolved_path"],
                f"WSI {sid}: resolved target differs from declaration")
        entry, _ = _source(f"wsi/{sid}", "wsi", path, slide_id=sid,
                            expected_size=item["size_bytes"],
                            basis="observed_only", stream=True)
        entries[entry["source_id"]] = entry
        root_id = f"root/wsi/{sid}"
        entries[root_id] = _root_entry(root_id, path.parent, True)
    for item in spec["models"]:
        name = item["name"]
        config_path = path_of(item["config_path"], base)
        config, _ = _source(f"model_config/{name}", "model_config", config_path,
                            bank=name, expected_sha=item["config_sha256"])
        entries[config["source_id"]] = config
        weight_path = path_of(item["weight_path"], base)
        require(str(weight_path.resolve(strict=True)) == item["resolved_weight_path"],
                f"weight/{name}: resolved path disagrees with declaration")
        weight, _ = _source(f"weight/{name}", "weight", weight_path, bank=name,
                            expected_size=item["weight_size_bytes"],
                            expected_sha=item["expected_weight_sha256"],
                            basis=item["digest_basis"], stream=True)
        entries[weight["source_id"]] = weight
        root_id = f"root/weight/{name}"
        entries[root_id] = _root_entry(root_id, weight_path.parent, True)
    root_id = "root/auxiliary"
    aux = path_of(spec["auxiliary"]["path"], base)
    entries[root_id] = _root_entry(root_id, aux.parent, True)
    roles = {}
    for source in entries.values():
        if source["kind"] not in ("candidate", "feature") or not source["observed_presence"]:
            continue
        device = os.stat(source["path"])
        physical = device.st_dev, device.st_ino
        earlier = roles.get(physical)
        require(earlier is None or (
            earlier["kind"] == source["kind"] == "feature"
            and earlier["bank"] == source["bank"]
            and earlier["slide_id"] == source["slide_id"]),
                f"unexpected feature/candidate alias: {source['source_id']} / "
                f"{earlier['source_id'] if earlier else ''}")
        roles[physical] = source
    return _selected_pairs(spec, entries)


def _code_sources(spec, base, entries):
    repo = Path(__file__).resolve().parents[2]
    stage0_review = bank_audit._json(_bounded(
        path_of(spec["stage0"]["review"], base), OTHER_CAP))
    stage1a_review = bank_audit._json(_bounded(
        path_of(spec["stage1a"]["review"], base), OTHER_CAP))
    automatic = [
        ("auto/panel", Path(__file__), None),
        ("auto/panel_tests", repo / "tests/test_encoder_replay_panel.py", None),
        ("auto/stage0_helper", Path(bank_audit.__file__),
         stage0_review["source_sha256"]["auditor_module"]),
        ("auto/stage0_tests", repo / "tests/test_feature_bank_audit.py",
         stage0_review["source_sha256"]["auditor_tests"]),
        ("auto/stage1a_helper", Path(annotation.__file__),
         stage1a_review["source_sha256"]["auditor_module"]),
        ("auto/stage1a_tests", repo / "tests/test_annotation_bank_audit.py",
         stage1a_review["source_sha256"]["auditor_tests"]),
    ]
    for source_id, path, expected in automatic:
        entry, _ = _source(source_id, "code", path, expected_sha=expected,
                            basis="stage0" if source_id.startswith("auto/stage0")
                            else "stage1a" if source_id.startswith("auto/stage1a")
                            else "observed_only")
        entries[source_id] = entry
    for item in spec["code_files"]:
        source_id = f"code/{item['name']}"
        entry, _ = _source(source_id, "code", path_of(item["path"], base),
                            expected_sha=item["sha256"])
        entries[source_id] = entry
    return [{"source_id": name, "sha256": entries[name]["observed_sha256"]}
            for name in sorted(entries) if entries[name]["kind"] == "code"]


def _output_vectors(selected):
    widths = {"uni2_main.npy": 1536, "uni2_annotation.npy": 1536,
              "virchow2_main_f16.npy": 1280, "virchow2_main_f32.npy": 1280,
              "virchow2_annotation_full.npy": 2560, "h_optimus_1_main.npy": 1536,
              "h_optimus_1_annotation.npy": 1536, "h_optimus_1_auxiliary.npy": 1536}
    result = {}
    for name in ARRAYS:
        rows = selected[name]
        require(rows, f"{name}: zero selected rows cannot form a valid NPY")
        dtype = "<f2" if name == "virchow2_main_f16.npy" else "<f4"
        payload = b"".join(rows)
        require(len(payload) == len(rows) * widths[name] * np.dtype(dtype).itemsize,
                f"{name}: selected source width/precision mismatch")
        values = np.frombuffer(payload, dtype=dtype).reshape(len(rows), widths[name])
        output = io.BytesIO()
        np.lib.format.write_array(output, values, version=(1, 0), allow_pickle=False)
        blob = output.getvalue()
        require(np.array_equal(_numeric_npy(blob, values.shape, dtype).view("u1"),
                               values.view("u1")), f"{name}: serialized bit mismatch")
        result[name] = blob
    return result


def _reports(spec, start, entries, reviewed, panel_rows, references, unique_keys,
             universes, lineage, code, payload_bytes, verified_controls):
    counts = {
        "slides": len(spec["panel"]), "patients": len({p["patient_id"] for p in spec["panel"]}),
        "universes": len(universes), "tile_rows": len(panel_rows),
        "feature_rows": len(references),
        "feature_files": len({ref[4] for ref in references}),
        "array_files": len(ARRAYS),
        "candidate_files_present": sum(e["observed_presence"] for e in entries.values()
                                       if e["kind"] == "candidate"),
        "candidate_files_absent": sum(not e["observed_presence"] for e in entries.values()
                                      if e["kind"] == "candidate"),
        "lineage_pairs": len(lineage["pairs"]),
        "array_payload_bytes": payload_bytes,
        "unique_physical_tiles": len(unique_keys),
    }
    coverage = {
        "virchow_strata": {group: [p["slide_id"] for p in spec["panel"]
                                    if p["virchow_stratum"] == group]
                            for group in dict.fromkeys(p["virchow_stratum"] for p in spec["panel"])},
        "annotation_classes": {str(c): [row[9] for row in panel_rows
                                         if row[0] == "annotation" and row[10] == c]
                               for c in range(1, 6)},
        "annotation_group_sources": {
            name: [p["slide_id"] for p in spec["panel"]
                   if p["annotation_classes"] and p["group"] == name]
            for name in ("NORMAL", "MISC")},
        "control_tags": verified_controls,
    }
    limits = [
        "This is a sampled coordinate/reference panel, not whole-bank coverage.",
        "Current image and weight bytes cannot establish historical image/weight identity.",
        "Filename or producer-log lineage does not establish historical encoder compatibility.",
        "No pixel, MPP, image bounds, forward, training or performance validation.",
        "Only six declared write-control lines are checked, not all producer logs.",
        "Source hashes do not establish the identity of already loaded bytecode.",
        "Cooperative output ownership and two non-atomic input hash sweeps require an input freeze.",
    ]
    summary = {
        "schema_version": 1, "run_id": spec["run_id"], "scope": SCOPE,
        "started_at": start, "prepared_at": now(),
        "panel_sha256": spec["expected"]["panel_sha256"], "counts": counts,
        "coverage": coverage,
        "models": [{"name": m["name"], "config_source_id": f"model_config/{m['name']}",
                    "weight_source_id": f"weight/{m['name']}",
                    "identity_scope": "pinned_current_cache_candidate"}
                   for m in spec["models"]],
        "wsi": [{"slide_id": w["slide_id"], "source_id": f"wsi/{w['slide_id']}",
                 "identity_scope": "current_bytes_only"} for w in spec["wsi"]],
        "upstream": reviewed, "code": code,
        "diagnostic_counts": lineage["counts"], "limits": limits,
    }
    return summary


def _read_written(fd, name, cap=OTHER_CAP):
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    with os.fdopen(descriptor, "rb") as handle:
        require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode),
                f"written {name}: not a regular file")
        data = handle.read(cap + 1)
    require(len(data) <= cap, f"written {name}: exceeds bound")
    return data


def _csv_readback(data, header, expected, name):
    try:
        records = list(csv.reader(io.StringIO(data.decode("utf-8"), newline=""),
                                  strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise AuditError(f"written {name}: malformed CSV: {exc}") from exc
    require(records and tuple(records[0]) == header and
            all(len(row) == len(header) for row in records),
            f"written {name}: missing/extra columns")
    require(data == _csv_bytes(header, records[1:]),
            f"written {name}: CSV encoding/records differ")
    require(records[1:] == [[str(v) if v is not None else "" for v in row]
                            for row in expected],
            f"written {name}: row-level bindings differ from validated sources")
    return records[1:]


def _verify_written(fd, expected, spec, selected):
    retained = {}
    parsed = {}
    names = set(CSV_OUTPUTS) | set(ARRAYS) | {"lineage_diagnostics.json", "summary.json"}
    require(set(os.listdir(fd)) == names, "written output set contains missing/extra entries")
    for name, header in CSV_OUTPUTS.items():
        data = _read_written(fd, name)
        parsed[name] = _csv_readback(data, header, expected[name], name)
        retained[name] = {"sha256": hashlib.sha256(data).hexdigest(),
                          "size_bytes": len(data)}
    parsed_panel = parsed["panel_tiles.csv"]
    require(len({(r[0], r[1], r[4]) for r in parsed_panel}) == len(parsed_panel)
            and len({(r[0], r[1], r[9]) for r in parsed_panel}) == len(parsed_panel),
            "written panel: duplicate universe/slide/index or tile")
    for row in parsed_panel:
        sid = row[1]
        key = (sid, decimal(row[5], "written level"),
               decimal(row[6], "written x"), decimal(row[7], "written y"),
               decimal(row[8], "written patch size"))
        require(key[1] == 0 and key[4] == 512
                and annotation._tile_id(key) == row[9]
                and row[11] in ("class_reservation", "intersection", "landmark",
                                "hash_fill")
                and (row[10] == "" if row[0] != "annotation"
                     else 1 <= decimal(row[10], "written label", 5) <= 5),
                f"written panel {sid}: invalid physical key, label or role")
    ref_rows = parsed["feature_rows.csv"]
    require(len({(r[0], r[1], r[2], r[3]) for r in ref_rows}) == len(ref_rows)
            and len({(r[8], r[9]) for r in ref_rows}) == len(ref_rows),
            "written feature rows: duplicate key or array row")
    observed_arrays = Counter((r[8] for r in ref_rows))
    for name in ARRAYS:
        data = _read_written(fd, name, OUTPUT_NPY_CAP)
        rows = selected[name]
        dtype = "<f2" if name == "virchow2_main_f16.npy" else "<f4"
        width = (2560 if name == "virchow2_annotation_full.npy" else
                 1280 if name.startswith("virchow2_") else 1536)
        array = _numeric_npy(data, (len(rows), width), dtype)
        require(array.tobytes(order="C") == b"".join(rows),
                f"written {name}: selected source component bits/row order differ")
        require(observed_arrays[name] == len(array)
                and {int(r[9]) for r in ref_rows if r[8] == name}
                == set(range(len(array))),
                f"written {name}: reference coverage not contiguous")
        retained[name] = {"sha256": hashlib.sha256(data).hexdigest(),
                          "size_bytes": len(data)}
    for name in ("lineage_diagnostics.json", "summary.json"):
        data = _read_written(fd, name)
        obj = bank_audit._json(data)
        require(type(obj) is dict
                and _json_bytes(obj) == _json_bytes(expected[name])
                and data == _json_bytes(obj),
                f"written {name}: semantic report differs from source-derived report")
        parsed[name] = obj
        retained[name] = {"sha256": hashlib.sha256(data).hexdigest(),
                          "size_bytes": len(data)}
    summary, lineage = parsed["summary.json"], parsed["lineage_diagnostics.json"]
    require(set(summary) == {"schema_version", "run_id", "scope", "started_at",
                             "prepared_at", "panel_sha256", "counts", "coverage",
                             "models", "wsi", "upstream", "code",
                             "diagnostic_counts", "limits"}
            and set(lineage) == {"schema_version", "run_id", "scope", "pairs", "counts"}
            and all(set(pair) == set(PAIR_KEYS) for pair in lineage["pairs"])
            and set(lineage["counts"]) == set(PAIR_COUNT_KEYS)
            and lineage["counts"] == summary["diagnostic_counts"],
            "written reports: schema or diagnostics mismatch")
    counts = summary["counts"]
    require(set(counts) == set(spec["expected"]) - {"panel_sha256"} | {
        "unique_physical_tiles"}
            and all(type(counts[k]) is int and counts[k] == spec["expected"][k]
                    for k in spec["expected"] if k != "panel_sha256")
            and counts["tile_rows"] == len(parsed_panel)
            and counts["feature_rows"] == len(ref_rows)
            and counts["unique_physical_tiles"] == len({
                (r[1], r[5], r[6], r[7], r[8]) for r in parsed_panel}),
            "written summary counts do not reconcile to parsed output rows")
    return retained


def _recheck_inputs(entries):
    for source_id in sorted(entries):
        entry = entries[source_id]
        path = Path(entry["path"])
        require(_entry_exists(path) == entry["expected_presence"],
                f"{source_id}: late presence/absence change")
        if not entry["expected_presence"]:
            continue
        require(str(path.resolve(strict=True)) == entry["resolved_path"],
                f"{source_id}: late lexical/resolved retarget")
        if entry["kind"] == "root":
            require(path.is_dir(), f"{source_id}: root no longer a directory")
            continue
        require(path.is_file(), f"{source_id}: input no longer a regular file")
        digest, size = _hash_stream(path, entry["observed_size_bytes"])
        require(digest == entry["observed_sha256"]
                and size == entry["observed_size_bytes"],
                f"{source_id}: late byte mutation")


def _finish(spec, base, directory_fd, spec_path, spec_sha, out, out_root, target, start,
            preflight_paths, preflight_roots):
    entries = {}
    verified_controls = _bind_write_controls(spec, base, entries)
    (entries, reviewed, panel_rows, references, selected,
     unique_keys, universes) = _prepare(spec, base, entries)
    lineage = _remaining_sources(spec, base, entries)
    code = _code_sources(spec, base, entries)
    spec_entry, _ = _source("spec", "control", spec_path, expected_sha=spec_sha)
    entries["spec"] = spec_entry
    require(sum(e["observed_presence"] for e in entries.values()
                if e["kind"] == "candidate") ==
            spec["expected"]["candidate_files_present"]
            and sum(not e["observed_presence"] for e in entries.values()
                    if e["kind"] == "candidate") ==
            spec["expected"]["candidate_files_absent"],
            "candidate present/absent counts differ")
    array_bytes = _output_vectors(selected)
    payload_bytes = sum(sum(len(row) for row in selected[name]) for name in ARRAYS)
    require(payload_bytes == spec["expected"]["array_payload_bytes"],
            "selected array payload bytes mismatch")
    summary = _reports(spec, start, entries, reviewed, panel_rows, references,
                       unique_keys, universes, lineage, code, payload_bytes,
                       verified_controls)
    require(all(summary["counts"][k] == v for k, v in spec["expected"].items()
                if k != "panel_sha256"), "selected output counts disagree with declaration")
    expected = {"panel_tiles.csv": panel_rows, "feature_rows.csv": references,
                "source_inventory.csv": _inventory_rows(entries),
                "lineage_diagnostics.json": lineage, "summary.json": summary}
    for name, header in CSV_OUTPUTS.items():
        bank_audit._atomic_bytes(directory_fd, name, _csv_bytes(header, expected[name]))
    for name in ARRAYS:
        bank_audit._atomic_bytes(directory_fd, name, array_bytes[name])
    for name in ("lineage_diagnostics.json", "summary.json"):
        bank_audit._atomic_bytes(directory_fd, name, _json_bytes(expected[name]))
    del array_bytes
    retained = _verify_written(directory_fd, expected, spec, selected)
    _recheck_inputs(entries)
    for name, identity in retained.items():
        data = _read_written(directory_fd, name,
                             OUTPUT_NPY_CAP if name in ARRAYS else OTHER_CAP)
        require(len(data) == identity["size_bytes"]
                and hashlib.sha256(data).hexdigest() == identity["sha256"],
                f"written {name}: changed after semantic readback")
    require(set(os.listdir(directory_fd)) == set(retained),
            "written output set changed after semantic readback")
    bank_audit._claimed_path(directory_fd, target)
    require(bank_audit._output_path(
        out, out_root, preflight_paths,
        [root for root in preflight_roots if _entry_exists(root)]) == target,
        "output root or lexical destination retargeted before completion")
    receipt = {
        "schema_version": 1, "run_id": spec["run_id"], "scope": SCOPE,
        "disposition": "prepared", "completed_at": now(),
        "spec": {"path": str(spec_path), "sha256": spec_sha},
        "upstream": reviewed, "code": code,
        "input_inventory_sha256": retained["source_inventory.csv"]["sha256"],
        "outputs": retained,
    }
    bank_audit._atomic_bytes(directory_fd, "COMPLETE.json", _json_bytes(receipt))
    if lineage["counts"]["non_comparable"] or any(
        lineage["counts"][k] for k in ("exact_value_different", "bitwise_different",
                                       "destination_cast_different")):
        print("WARNING: lineage diagnostics contain non-comparable/different pairs; "
              "inspect lineage_diagnostics.json before replay", file=sys.stderr)
    return receipt


def audit(spec_path, spec_sha, out_path):
    _hex(spec_sha, "spec SHA256")
    require(os.environ.get("PROSPER_OUT", "").strip(),
            "PROSPER_OUT must be explicitly nonempty")
    spec_path = path_of(spec_path, Path.cwd())
    spec_bytes = _bounded(spec_path, OTHER_CAP)
    require(hashlib.sha256(spec_bytes).hexdigest() == spec_sha,
            "spec SHA256 does not match parsed bytes")
    spec = bank_audit._json(spec_bytes)
    _schema(spec)
    _production_contract(spec)
    base = spec_path.parent
    out_root = path_of(os.environ["PROSPER_OUT"], Path.cwd())
    out = path_of(out_path, Path.cwd())
    present = [spec_path, path_of(spec["metadata"]["path"], base)]
    for stage in ("stage0", "stage1a"):
        present += [path_of(spec[stage][key], base) for key in ("review", "complete")]
    present += [path_of(item["path"], base) for item in spec["wsi"]]
    present += [path_of(item[key], base) for item in spec["models"]
                for key in ("config_path", "weight_path")]
    present += [path_of(item["path"], base) for item in spec["log_controls"]]
    present += [path_of(item["path"], base) for item in spec["code_files"]]
    present += [path_of(spec["auxiliary"][key], base) for key in ("path", "csv")]
    roots = [path_of(item["path"], base) for item in spec["banks"]]
    roots += [path_of(item["path"], base) for item in spec["virchow_candidates"]]
    require(all(not bank_audit._inside(out.resolve(), root.resolve())
                and not bank_audit._inside(root.resolve(), out.resolve()) for root in roots),
            "output overlaps declared bank/candidate root, including absent roots")
    target = bank_audit._output_path(out, out_root, present,
                                    [root for root in roots if _entry_exists(root)])
    fd = bank_audit._claim_output(out, target)
    owned = False
    try:
        bank_audit._claimed_path(fd, target)
        require(bank_audit._output_path(out, out_root, present,
                                        [root for root in roots if _entry_exists(root)])
                == target, "output retargeted before ownership established")
        owned = True
        return _finish(spec, base, fd, spec_path, spec_sha, out, out_root, target, now(),
                       present, roots)
    except Exception as exc:
        if owned:
            try:
                bank_audit._atomic_bytes(fd, "FAILED.json", _json_bytes({
                    "failed_at": now(), "type": type(exc).__name__, "error": str(exc)}))
            except OSError as failure:
                print(f"WARNING: unable to write owned FAILED.json: {failure}",
                      file=sys.stderr)
        raise
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        audit(args.spec, args.spec_sha256, args.out)
    except (AuditError, OSError, ValueError, KeyError, TypeError, UnicodeError,
            json.JSONDecodeError) as exc:
        print(f"encoder replay panel FAILED: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
