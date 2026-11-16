"""Bind historical annotation coordinates to declared encoder feature rows.

Trusted local pickles only. This is a keyed input audit, not training, legacy-X
validation, model identity, or scientific acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import socket
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import feature_bank_audit as stage0


AuditError = stage0.AuditError
_require = stage0._require
_keys = stage0._keys
_text = stage0._text
_integer = stage0._integer
_json = stage0._json
_path = stage0._path
_binding = stage0._binding
_pickle = stage0._pickle
_coordinates = stage0._coordinates
_time = stage0._time
_output_path = stage0._output_path
_claim_output = stage0._claim_output
_claimed_path = stage0._claimed_path
_atomic_bytes = stage0._atomic_bytes
_read_output = stage0._read_output

OTHER_LIMIT = 128 * 1024 * 1024
FEATURE_LIMIT = 512 * 1024 * 1024
CLASSES = tuple(str(i) for i in range(1, 6))
LINEAGES = ("uni2", "virchow2")
BANKS = ("uni2", "virchow2", "h_optimus_1")
DIMENSIONS = {"uni2": (1536, (0, 1536)), "virchow2": (2560, (0, 1280)),
              "h_optimus_1": (1536, (0, 1536))}
ELIGIBLE_HEADER = ("tile_id", "slide_id", "patient_id", "diagnostic_group",
                   "level", "x", "y", "patch_size", "tissue_label",
                   "uni2_legacy_row", "virchow2_legacy_row",
                   "uni2_feature_row", "virchow2_feature_row", "h_optimus_1_feature_row")
QUARANTINE_HEADER = ("tile_id", "slide_id", "level", "x", "y", "patch_size",
                     "tissue_label", "uni2_legacy_row", "virchow2_legacy_row", "reason")
SLIDE_HEADER = ("slide_id", "eligibility", "patient_id", "diagnostic_group",
                "target_rows", "patch_rows", "label_rows", "canonical_rows",
                "extra_labelled_rows", "patch_not_label", "label_not_patch",
                "patch_not_canonical", "canonical_not_patch",
                "label_not_canonical", "canonical_not_label",
                "uni2_csv_target_order_equal", "virchow2_csv_target_order_equal")
FEATURE_HEADER = ("bank", "slide_id", "feature_dim", "projection_start",
                  "projection_stop", "feature_rows", "dtype", "target_rows",
                  "unused_unannotated", "unused_labelled_nontarget",
                  "unused_outside_declared_sources", "label_csv_order_equal",
                  "label_csv_position_mismatches", "feature_path", "resolved_path",
                  "size", "sha256")
OUTPUT_HEADERS = {"eligible_tiles.csv": ELIGIBLE_HEADER,
                  "quarantine_tiles.csv": QUARANTINE_HEADER,
                  "slide_inventory.csv": SLIDE_HEADER,
                  "feature_inventory.csv": FEATURE_HEADER}
RESERVED = {"spec", "metadata", "stage0_review", "stage0_complete",
            "stage0_helper", "running_auditor", "canonical"}


def _sha(value, context):
    _require(isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", value)),
             f"{context}: expected 64 hex SHA256")
    return value.lower()


def _safe_id(value, context):
    _text(value, context)
    _require(value == value.strip() and value not in (".", "..")
             and not any(c in value for c in "/\\")
             and not any(ord(c) < 32 or ord(c) == 127 for c in value)
             and Path(value).name == value,
             f"{context}: unsafe or whitespace-padded slide identity {value!r}")
    return value


def _schema(spec):
    _keys(spec, ("schema_version", "run_id", "stage0", "metadata", "lineages",
                 "annotations", "banks", "expected", "code_files"), "spec")
    _require(type(spec["schema_version"]) is int and spec["schema_version"] == 1,
             "schema_version: expected integer 1")
    _text(spec["run_id"], "run_id")
    review = spec["stage0"]
    _keys(review, ("review", "review_sha256", "complete"), "stage0")
    _text(review["review"], "stage0.review")
    _text(review["complete"], "stage0.complete")
    _sha(review["review_sha256"], "stage0.review_sha256")
    meta = spec["metadata"]
    _keys(meta, ("path", "id_col", "patient_col", "label_col"), "metadata")
    for key, value in meta.items():
        _text(value, f"metadata.{key}")
    _require(len({meta[k] for k in ("id_col", "patient_col", "label_col")}) == 3,
             "metadata: duplicate column selectors")
    annotations = spec["annotations"]
    _keys(annotations, ("patch_csv_dir", "label_csv_dir", "canonical_path",
                        "level", "patch_size"), "annotations")
    for key in ("patch_csv_dir", "label_csv_dir", "canonical_path"):
        _text(annotations[key], f"annotations.{key}")
    _require(type(annotations["level"]) is int and annotations["level"] == 0,
             "annotations.level: expected integer 0")
    _require(type(annotations["patch_size"]) is int and annotations["patch_size"] == 512,
             "annotations.patch_size: expected integer 512")
    _require(isinstance(spec["lineages"], list) and len(spec["lineages"]) == 2,
             "lineages: expected exactly two entries")
    names = set()
    for item in spec["lineages"]:
        _keys(item, ("name", "slide_info", "train_indices", "test_indices",
                     "train_labels", "test_labels"), "lineage")
        for key, value in item.items():
            _text(value, f"lineage.{key}")
        _require(item["name"] not in names, f"duplicate lineage: {item['name']}")
        names.add(item["name"])
    _require(names == set(LINEAGES), "lineages: expected uni2 and virchow2")
    _require(isinstance(spec["banks"], list) and len(spec["banks"]) == 3,
             "banks: expected exactly three entries")
    names = set()
    roots = set()
    for bank in spec["banks"]:
        _keys(bank, ("name", "path", "feature_dim", "projection", "dtypes"), "bank")
        _text(bank["name"], "bank.name")
        _text(bank["path"], "bank.path")
        _require(bank["name"] in DIMENSIONS and bank["name"] not in names,
                 f"unknown or duplicate bank: {bank['name']}")
        _require(bank["path"] not in roots, f"duplicate bank path: {bank['path']}")
        roots.add(bank["path"])
        names.add(bank["name"])
        dim, projection = DIMENSIONS[bank["name"]]
        _require(type(bank["feature_dim"]) is int and bank["feature_dim"] == dim,
                 f"bank {bank['name']}: wrong raw feature_dim")
        _require(isinstance(bank["projection"], list)
                 and len(bank["projection"]) == 2
                 and all(type(v) is int for v in bank["projection"])
                 and tuple(bank["projection"]) == projection,
                 f"bank {bank['name']}: wrong projection")
        dtypes = bank["dtypes"]
        _require(isinstance(dtypes, list) and bool(dtypes)
                 and all(type(v) is str and v in ("float16", "float32") for v in dtypes)
                 and len(dtypes) == len(set(dtypes)),
                 f"bank {bank['name']}: unsupported or duplicate dtype")
    _require(names == set(BANKS), "banks: require uni2, virchow2, h_optimus_1")
    expected = spec["expected"]
    _keys(expected, ("historical_rows", "historical_slides", "train_rows", "test_rows",
                     "mapped_rows", "mapped_slides", "quarantined_rows",
                     "quarantined_slides", "mapped_class_counts",
                     "quarantined_class_counts"), "expected")
    for name in ("historical_rows", "historical_slides", "train_rows", "test_rows",
                 "mapped_rows", "mapped_slides", "quarantined_rows",
                 "quarantined_slides"):
        _integer(expected[name], f"expected.{name}",
                 1 if name in ("historical_rows", "historical_slides",
                               "mapped_rows", "mapped_slides") else 0)
    for name in ("mapped_class_counts", "quarantined_class_counts"):
        counts = expected[name]
        _keys(counts, CLASSES, f"expected.{name}")
        for label, count in counts.items():
            _integer(count, f"expected.{name}.{label}")
        _require(sum(counts.values()) == expected[name.split("_class")[0] + "_rows"],
                 f"expected.{name}: class sum mismatch")
    for total, left, right in (("historical_rows", "train_rows", "test_rows"),
                               ("historical_rows", "mapped_rows", "quarantined_rows"),
                               ("historical_slides", "mapped_slides", "quarantined_slides")):
        _require(expected[total] == expected[left] + expected[right],
                 f"expected: {total} != {left} + {right}")
    _require(isinstance(spec["code_files"], list) and bool(spec["code_files"]),
             "code_files: expected nonempty list")
    code_names = set()
    for item in spec["code_files"]:
        _keys(item, ("name", "path"), "code_file")
        for key, value in item.items():
            _text(value, f"code_file.{key}")
        _require(item["name"] not in code_names | RESERVED | set(LINEAGES) | set(BANKS)
                 and not item["name"].startswith(("feature:", "lineage:", "patch:",
                                                   "labels:", "stage0_artifact:")),
                 f"code_file: duplicate or reserved binding name {item['name']}")
        code_names.add(item["name"])


def _read_capped(path, limit):
    chunks = []
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(min(1024 * 1024, limit - size + 1))
            if not chunk:
                break
            size += len(chunk)
            _require(size <= limit, f"{path}: input exceeds {limit}-byte materialized-read cap")
            chunks.append(chunk)
    return b"".join(chunks)


def _input(path, kind, name, bindings, aliases, identities, limit=OTHER_LIMIT):
    data = _read_capped(path, limit)
    item = _binding(path, kind, name, data=data)
    ident = (item["device"], item["inode"])
    if ident in identities:
        aliases.append((identities[ident], f"{kind}:{name}"))
    else:
        identities[ident] = f"{kind}:{name}"
    bindings.append(item)
    return data, item


def _csv(data, required, context):
    reader = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    headers = next(reader, None)
    _require(headers is not None and len(headers) == len(set(headers))
             and all(h.strip() for h in headers) and set(required) <= set(headers),
             f"{context}: duplicate/blank headers or missing {sorted(set(required))}")
    rows = []
    for number, cells in enumerate(reader, 2):
        _require(len(cells) == len(headers), f"{context} row {number}: wrong CSV width")
        rows.append(dict(zip(headers, cells)))
    return rows


def _decimal(value, context, maximum=None):
    _require(isinstance(value, str) and bool(re.fullmatch(r"[0-9]+", value, re.ASCII)),
             f"{context}: expected unsigned ASCII decimal integer")
    result = int(value)
    _require(maximum is None or result <= maximum,
             f"{context}: integer exceeds {maximum}")
    return result


def _metadata(data, config):
    rows = _csv(data, (config["id_col"], config["patient_col"], config["label_col"]),
                "metadata")
    result = {}
    for i, row in enumerate(rows, 2):
        sid = _safe_id(row[config["id_col"]], f"metadata row {i} scan ID")
        patient = row[config["patient_col"]]
        label = row[config["label_col"]]
        for value, field in ((patient, "patient"), (label, "diagnostic group")):
            _require(bool(value) and value == value.strip(),
                     f"metadata row {i}: blank or padded {field}")
        _require(sid not in result, f"metadata: duplicate slide {sid}")
        result[sid] = (patient, label)
    return result


def _coords(tiles, context):
    coords = _coordinates(tiles, context)
    _require(all(x >= 0 and y >= 0 for x, y in coords),
             f"{context}: negative physical coordinate")
    return coords


def _key(sid, xy):
    return sid, 0, xy[0], xy[1], 512


def _tile_id(key):
    blob = json.dumps(key, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _npy(data, context):
    stream = io.BytesIO(data)
    try:
        major, minor = np.lib.format.read_magic(stream)
        _require((major, minor) in ((1, 0), (2, 0)),
                 f"{context}: unsupported NPY version")
        read_header = (np.lib.format.read_array_header_1_0 if major == 1
                       else np.lib.format.read_array_header_2_0)
        shape, _, dtype = read_header(stream, max_header_size=65536)
        _require(len(shape) == 1 and type(shape[0]) is int and shape[0] >= 0
                 and dtype.kind == "i" and dtype.itemsize == 8 and not dtype.hasobject,
                 f"{context}: expected one-dimensional signed int64 NPY")
        offset = stream.tell()
        _require(len(data) - offset == shape[0] * 8,
                 f"{context}: truncated or trailing NPY payload")
        return np.frombuffer(data, dtype=dtype, count=shape[0], offset=offset)
    except AuditError:
        raise
    except (ValueError, EOFError, SyntaxError, TypeError) as exc:
        raise AuditError(f"{context}: invalid NPY header or payload: {exc}") from exc


def _source_csv(data, sid, label_source, context):
    required = ("slide_id", "wsi_loc_x", "wsi_loc_y")
    rows = _csv(data, required + (("patch_label",) if label_source else ()), context)
    coordinates = []
    labels = {}
    for number, row in enumerate(rows, 2):
        _require(_safe_id(row["slide_id"], f"{context} row {number} slide_id") == sid,
                 f"{context} row {number}: slide_id disagrees with filename {sid}")
        xy = (_decimal(row["wsi_loc_x"], f"{context} row {number} x"),
              _decimal(row["wsi_loc_y"], f"{context} row {number} y"))
        _require(xy not in labels, f"{context}: duplicate physical key {xy}")
        labels[xy] = (_decimal(row["patch_label"], f"{context} row {number} label", 5)
                      if label_source else None)
        coordinates.append(xy)
    return coordinates, labels


def _diagnostic(entries, kind, count, examples, *, slide_id=None, bank=None, lineage=None):
    if count:
        entry = {"kind": kind, "count": count, "examples": examples[:5]}
        for key, value in (("slide_id", slide_id), ("bank", bank), ("lineage", lineage)):
            if value is not None:
                entry[key] = value
        entries.append(entry)


def _position_diff(left, right):
    positions = [i for i in range(max(len(left), len(right)))
                 if i >= len(left) or i >= len(right) or left[i] != right[i]]
    return len(positions), positions[:5]


def _write_csv(directory_fd, name, header, rows):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    _atomic_bytes(directory_fd, name, data)
    return data


def _read_csv_output(data, name, header):
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    _require(tuple(next(reader, ())) == header, f"written {name}: wrong header")
    rows = []
    for row in reader:
        _require(len(row) == len(header), f"written {name}: malformed row")
        rows.append(tuple(row))
    return rows


def _verify_outputs(directory_fd, expected_rows, expected_bytes, summary, diagnostics,
                    metadata, historical, lineages, features, source_rows, expected,
                    extra_canonical, aliases, validated_features):
    data = {name: _read_output(directory_fd, name)
            for name in (*OUTPUT_HEADERS, "diagnostics.json", "summary.json")}
    for name, expected_data in expected_bytes.items():
        _require(data[name] == expected_data, f"written {name}: validated bytes changed")
    rows = {name: _read_csv_output(data[name], name, header)
            for name, header in OUTPUT_HEADERS.items()}
    _require(rows == expected_rows, "written CSV: rows, indices, labels or inventories changed")
    eligible_keys, quarantine_keys = set(), set()
    for name, mapped, seen in (("eligible_tiles.csv", True, eligible_keys),
                               ("quarantine_tiles.csv", False, quarantine_keys)):
        header = OUTPUT_HEADERS[name]
        for row_values in rows[name]:
            row = dict(zip(header, row_values))
            level = _decimal(row["level"], f"written {name} level")
            patch_size = _decimal(row["patch_size"], f"written {name} patch_size")
            _require(level == 0 and patch_size == 512,
                     f"written {name}: wrong physical level or patch size")
            key = (row["slide_id"], level,
                   _decimal(row["x"], f"written {name} x"),
                   _decimal(row["y"], f"written {name} y"), patch_size)
            _require(key not in seen and row["tile_id"] == _tile_id(key)
                     and key in historical, f"written {name}: wrong or duplicate physical key")
            seen.add(key)
            sid, xy = key[0], (key[2], key[3])
            _require(int(row["tissue_label"]) == historical[key]
                     and int(row["uni2_legacy_row"]) == lineages["uni2"]["indices"][key]
                     and int(row["virchow2_legacy_row"]) == lineages["virchow2"]["indices"][key],
                     f"written {name}: label or lineage index mismatch {key}")
            _require((sid in metadata) == mapped, f"written {name}: wrong metadata partition")
            if mapped:
                _require((row["patient_id"], row["diagnostic_group"]) == metadata[sid]
                         and all(int(row[f"{bank}_feature_row"]) == features[bank][key]
                                 for bank in BANKS),
                         f"written {name}: patient or feature index mismatch {key}")
            else:
                _require(row["reason"] == "slide_absent_from_bound_metadata",
                         f"written {name}: invalid quarantine reason")
    _require(eligible_keys | quarantine_keys == set(historical)
             and not (eligible_keys & quarantine_keys)
             and len(eligible_keys) == expected["mapped_rows"]
             and len(quarantine_keys) == expected["quarantined_rows"],
             "written tiles: incomplete eligible/quarantine partition")
    _require(len(rows["slide_inventory.csv"]) == expected["historical_slides"]
             and {r[0] for r in rows["slide_inventory.csv"]}
             == {key[0] for key in historical}
             and len(rows["feature_inventory.csv"]) == 3 * expected["mapped_slides"]
             and {(r[0], r[1]) for r in rows["feature_inventory.csv"]}
             == {(bank, sid) for bank in BANKS for sid in metadata if sid in source_rows},
             "written inventories: missing or duplicate required coverage")
    for feature_row in rows["feature_inventory.csv"]:
        row = dict(zip(FEATURE_HEADER, feature_row))
        _require(feature_row == validated_features[(row["bank"], row["slide_id"])],
                 f"written feature inventory: differs from validated bank {row['bank']}/{row['slide_id']}")
        total = (int(row["target_rows"]) + int(row["unused_unannotated"])
                 + int(row["unused_labelled_nontarget"])
                 + int(row["unused_outside_declared_sources"]))
        _require(int(row["feature_rows"]) == total,
                 f"written feature inventory: unreconciled rows {row['bank']}/{row['slide_id']}")
    parsed = _json(data["summary.json"])
    reported = _json(data["diagnostics.json"])
    _require(isinstance(parsed, dict) and isinstance(reported, dict)
             and isinstance(reported.get("entries"), list),
             "written JSON: malformed summary or diagnostics")
    for name, keys in (("eligible", eligible_keys), ("quarantine", quarantine_keys)):
        support = {
            "rows": len(keys), "slides": len({key[0] for key in keys}),
            "class_rows": {c: sum(historical[key] == int(c) for key in keys)
                           for c in CLASSES},
            "class_slides": {c: len({key[0] for key in keys
                                     if historical[key] == int(c)}) for c in CLASSES},
        }
        if name == "eligible":
            support["patients"] = len({metadata[key[0]][0] for key in keys})
            support["class_patients"] = {
                c: len({metadata[key[0]][0] for key in keys
                        if historical[key] == int(c)}) for c in CLASSES}
        _require(parsed.get(name) == support, f"written summary: {name} support mismatch")
    source_totals = {
        field: sum(int(row[SLIDE_HEADER.index(field)])
                   for row in rows["slide_inventory.csv"]) for field in SLIDE_HEADER[4:15]}
    feature_totals = {
        bank: {field: sum(int(row[FEATURE_HEADER.index(field)])
                          for row in rows["feature_inventory.csv"] if row[0] == bank)
               for field in ("feature_rows", "target_rows", "unused_unannotated",
                             "unused_labelled_nontarget", "unused_outside_declared_sources")}
        for bank in BANKS}
    expected_global_count, _ = _position_diff(
        lineages["uni2"]["order"], lineages["virchow2"]["order"])
    required = Counter()

    def require_diagnostic(kind, count, slide_id=None, bank=None, lineage=None):
        if count:
            required[(kind, slide_id, bank, lineage, count)] += 1

    require_diagnostic("lineage_global_order", expected_global_count)
    require_diagnostic("additional_canonical_slides", len(extra_canonical))
    for _ in aliases:
        require_diagnostic("declared_file_alias", 1)
    lineage_csv_equal = {name: True for name in LINEAGES}
    source_sets_equal = not extra_canonical
    for values in rows["slide_inventory.csv"]:
        row = dict(zip(SLIDE_HEADER, values))
        sid = row["slide_id"]
        patch_order, patch, label_order, labels, canonical_order = source_rows[sid]
        mapped = sid in metadata
        _require(row["eligibility"] == ("mapped" if mapped else "quarantined")
                 and (row["patient_id"], row["diagnostic_group"])
                 == (metadata[sid] if mapped else ("", "")),
                 f"written slide inventory: eligibility or metadata mismatch {sid}")
        target = set(lineages["uni2"]["by_slide"][sid])
        difference_sets = {
            "patch_not_label": set(patch) - set(labels),
            "label_not_patch": set(labels) - set(patch),
            "patch_not_canonical": set(patch) - set(canonical_order),
            "canonical_not_patch": set(canonical_order) - set(patch),
            "label_not_canonical": set(labels) - set(canonical_order),
            "canonical_not_label": set(canonical_order) - set(labels),
        }
        _require(int(row["target_rows"]) == len(target)
                 and int(row["patch_rows"]) == len(patch_order)
                 and int(row["label_rows"]) == len(label_order)
                 and int(row["canonical_rows"]) == len(canonical_order),
                 f"written slide inventory: source/target mismatch {sid}")
        extra = sum(code > 0 and xy not in target for xy, code in labels.items())
        _require(int(row["extra_labelled_rows"]) == extra,
                 f"written slide inventory: extra labelled count mismatch {sid}")
        require_diagnostic("additional_labelled_nontarget", extra, slide_id=sid)
        for kind, difference in difference_sets.items():
            _require(int(row[kind]) == len(difference),
                     f"written slide inventory: source difference mismatch {sid}/{kind}")
            require_diagnostic("nontarget_source_difference", len(difference),
                               slide_id=sid, lineage=kind)
            source_sets_equal &= not difference
        for name in LINEAGES:
            order = [xy for xy in label_order if xy in target]
            count, _ = _position_diff(order, lineages[name]["by_slide"][sid])
            flag = row[f"{name}_csv_target_order_equal"]
            _require(flag == ("true" if count == 0 else "false"),
                     f"written slide inventory: lineage order mismatch {sid}/{name}")
            lineage_csv_equal[name] &= count == 0
            require_diagnostic("lineage_target_order", count,
                               slide_id=sid, lineage=name)
    feature_order_equal = {name: True for name in BANKS}
    for values in rows["feature_inventory.csv"]:
        row = dict(zip(FEATURE_HEADER, values))
        bank, sid = row["bank"], row["slide_id"]
        mismatch = int(row["label_csv_position_mismatches"])
        _require(row["label_csv_order_equal"]
                 == ("true" if mismatch == 0 else "false"),
                 f"written feature inventory: order status mismatch {bank}/{sid}")
        feature_order_equal[bank] &= mismatch == 0
        require_diagnostic("feature_csv_order", mismatch, slide_id=sid, bank=bank)
        require_diagnostic("unused_outside_declared_sources",
                           int(row["unused_outside_declared_sources"]),
                           slide_id=sid, bank=bank)
    observed = Counter()
    for entry in reported["entries"]:
        _require(isinstance(entry, dict) and type(entry.get("count")) is int
                 and entry["count"] > 0 and isinstance(entry.get("examples"), list)
                 and len(entry["examples"]) <= 5,
                 "written diagnostics: invalid count or examples")
        observed[(entry.get("kind"), entry.get("slide_id"), entry.get("bank"),
                  entry.get("lineage"), entry["count"])] += 1
    _require(observed == required,
             f"written diagnostics: missing={required - observed}, "
             f"spurious={observed - required}")
    _require(reported == {"schema_version": 1, "run_id": summary["run_id"],
                          "entries": json.loads(json.dumps(diagnostics["entries"]))},
             "written diagnostics: altered examples or entry metadata")
    _require(parsed.get("schema_version") == 1
             and parsed.get("run_id") == summary["run_id"]
             and parsed.get("scope") == "historical-annotation target keyed binding only"
             and parsed.get("historical") == {
                 "rows": len(historical), "slides": len(source_rows)}
             and parsed.get("keyed_join_status") == "complete"
             and parsed.get("lineage_order_equal") == (expected_global_count == 0)
             and parsed.get("lineage_csv_target_order_equal") == lineage_csv_equal
             and parsed.get("mapped_feature_csv_order_equal") == feature_order_equal
             and parsed.get("nontarget_source_sets_equal") == source_sets_equal
             and parsed.get("legacy_X_validation") == "not_performed"
             and parsed.get("encoder_identity") == "declared_unverified"
             and parsed.get("source_inventory_totals") == source_totals
             and parsed.get("feature_inventory_totals") == feature_totals
             and parsed.get("additional_canonical_slides") == len(extra_canonical)
             and parsed.get("diagnostic_entries") == len(reported["entries"]),
             "written summary: statuses, source, feature or diagnostic totals mismatch")
    for name in ("declarations", "input_bindings", "input_roots", "environment",
                 "verification_started_at", "source_recheck_completed_at"):
        _require(parsed.get(name) == summary[name],
                 f"written summary: {name} binding or provenance mismatch")
    hashes = {}
    for name, validated in data.items():
        current = _read_output(directory_fd, name)
        _require(current == validated, f"written {name}: late output change")
        hashes[name] = {"size": len(current), "sha256": hashlib.sha256(current).hexdigest()}
    return hashes


def _upstream(spec, base, read, bindings):
    config = spec["stage0"]
    review_path = _path(config["review"], base)
    review_data, review_binding = read(review_path, "stage0", "stage0_review")
    _require(review_binding["sha256"] == _sha(config["review_sha256"], "review SHA256"),
             "Stage0 review SHA256 mismatch")
    review = _json(review_data)
    _require(isinstance(review, dict)
             and {"disposition", "scope", "schema_version", "artifacts",
                  "source_sha256"} <= set(review),
             "Stage0 review: missing disposition/scope/artifact/source bindings")
    _require(review["disposition"] == "accepted"
             and review["scope"] == "Stage0 feature-bank binding only"
             and review["schema_version"] == 1,
             "Stage0 review: unaccepted or wrong binding scope")
    links = review["artifacts"]
    _require(isinstance(links, dict) and bool(links), "Stage0 review: missing artifacts")
    artifact_dir = review_path.parent.resolve(strict=True)
    artifacts = {}
    for rel, digest in sorted(links.items()):
        _sha(digest, f"Stage0 artifact {rel}")
        _require(isinstance(rel, str) and rel and not Path(rel).is_absolute()
                 and all(p not in (".", "..") for p in Path(rel).parts),
                 f"Stage0 artifact: unsafe path {rel!r}")
        path = _path(rel, review_path.parent)
        _require(stage0._inside(path.resolve(strict=True), artifact_dir),
                 f"Stage0 artifact: escapes review directory {rel}")
        data, item = read(path, "stage0_artifact", f"stage0_artifact:{rel}")
        _require(item["sha256"] == digest.lower(), f"Stage0 artifact hash mismatch: {rel}")
        artifacts[rel] = (data, item)
    complete_path = _path(config["complete"], base)
    complete_key = str(complete_path.resolve(strict=True))
    matching = [(rel, pair) for rel, pair in artifacts.items()
                if pair[1]["resolved_path"] == complete_key]
    _require(len(matching) == 1 and matching[0][0].endswith("/COMPLETE.json"),
             "Stage0 selected COMPLETE not uniquely linked by review")
    selected_data = _read_capped(complete_path, OTHER_LIMIT)
    selected_binding = _binding(complete_path, "stage0", "stage0_complete",
                                data=selected_data)
    _require(selected_binding["sha256"] == matching[0][1][1]["sha256"],
             "Stage0 selected COMPLETE bytes disagree with review link")
    bindings.append(selected_binding)
    complete = _json(selected_data)
    _require(complete["schema_version"] == 1
             and complete["scope"] == "feature-bank binding only"
             and set(complete["outputs"]) == {"inventory.csv", "summary.json"},
             "Stage0 selected COMPLETE: wrong scope or output set")
    for name in ("inventory.csv", "summary.json"):
        path = complete_path.parent / name
        match = [(rel, pair) for rel, pair in artifacts.items()
                 if pair[1]["resolved_path"] == str(path.resolve(strict=True))]
        _require(len(match) == 1, f"Stage0 review: missing {name} artifact link")
        item = match[0][1][1]
        _require(complete["outputs"][name] == {
            "sha256": item["sha256"], "size": item["size"]},
            f"Stage0 COMPLETE: {name} output hash/size disagreement")
    summary_path = complete_path.parent / "summary.json"
    summary_data = next(pair[0] for pair in artifacts.values()
                        if pair[1]["resolved_path"] == str(summary_path.resolve(strict=True)))
    summary = _json(summary_data)
    _require(summary["schema_version"] == 1
             and summary["run_id"] == complete["run_id"]
             and summary["scope"].startswith("feature-bank binding only"),
             "Stage0 summary/COMPLETE scope or run mismatch")
    source_hashes = review["source_sha256"]
    source_names = {"auditor_module": ("code", "running_auditor"),
                    "auditor_tests": ("code", "stage0_tests"),
                    "metadata": ("metadata", "metadata"),
                    "canonical_mosaic": ("mosaic", "canonical")}
    for source, (kind, name) in source_names.items():
        matches = [item for item in summary["input_bindings"]
                   if item["kind"] == kind and item["name"] == name]
        _require(len(matches) == 1
                 and matches[0]["sha256"] == _sha(source_hashes[source], source),
                 f"Stage0 review/summary source disagreement: {source}")
    spec_matches = [item for item in summary["input_bindings"]
                    if item["kind"] == "spec" and item["name"] == "spec"]
    stage0_specs = [item for rel, (_, item) in artifacts.items() if rel == "stage0_spec.json"]
    _require(len(spec_matches) == len(stage0_specs) == 1
             and spec_matches[0]["sha256"] == stage0_specs[0]["sha256"],
             "Stage0 review/spec binding disagreement")
    launch = artifacts.get("stage0_launch.sha256")
    _require(launch is not None, "Stage0 review: missing launch hash manifest")
    launch_lines = launch[0].decode("utf-8").splitlines()
    launch_hashes = set()
    for line in launch_lines:
        digest, separator, path = line.partition("  ")
        _require(separator and path and bool(re.fullmatch("[0-9a-f]{64}", digest)),
                 "Stage0 launch hash manifest: invalid entry")
        launch_hashes.add(digest)
    expected_launch = {
        _sha(source_hashes[k], k) for k in
        ("auditor_module", "auditor_tests", "metadata", "canonical_mosaic",
         "worker_wrapper")
    } | {stage0_specs[0]["sha256"]}
    _require(len(launch_hashes) == len(launch_lines)
             and launch_hashes == expected_launch,
             "Stage0 review/launch source or spec hashes disagree")
    meta0 = spec["metadata"]
    upstream_meta = summary["declarations"]["metadata"]
    _require(all(meta0[col] == upstream_meta[col]
                 for col in ("id_col", "patient_col", "label_col")),
             "Stage1 metadata column selectors disagree with accepted Stage0")
    metadata_binding = next(item for item in summary["input_bindings"]
                            if item["kind"] == "metadata" and item["name"] == "metadata")
    helper_path = _path(Path(stage0.__file__), Path.cwd())
    _, helper = read(helper_path, "code", "stage0_helper")
    _require(helper["sha256"] == _sha(source_hashes["auditor_module"], "Stage0 helper"),
             "imported Stage0 helper hash differs from accepted review")
    return metadata_binding["sha256"]


def audit(spec_path, spec_sha256, out, out_root):
    _sha(spec_sha256, "spec-sha256")
    _require(out_root and str(out_root).strip(), "PROSPER_OUT must be explicitly nonempty")
    spec_path = _path(spec_path, Path.cwd())
    spec_data = _read_capped(spec_path, OTHER_LIMIT)
    _require(hashlib.sha256(spec_data).hexdigest() == spec_sha256.lower(),
             "spec SHA256 mismatch")
    spec = _json(spec_data)
    _schema(spec)
    base = spec_path.parent
    bindings, aliases, identities = [], [], {}

    def read(path, kind, name, limit=OTHER_LIMIT):
        return _input(path, kind, name, bindings, aliases, identities, limit)

    spec_binding = _binding(spec_path, "spec", "spec", data=spec_data)
    bindings.append(spec_binding)
    identities[(spec_binding["device"], spec_binding["inode"])] = "spec:spec"
    upstream_meta_sha = _upstream(spec, base, read, bindings)
    metadata_path = _path(spec["metadata"]["path"], base)
    metadata_data, metadata_binding = read(metadata_path, "metadata", "metadata")
    _require(metadata_binding["sha256"] == upstream_meta_sha,
             "Stage1 metadata bytes differ from accepted Stage0")
    metadata = _metadata(metadata_data, spec["metadata"])
    read(_path(Path(__file__), Path.cwd()), "code", "running_auditor")
    for item in spec["code_files"]:
        read(_path(item["path"], base), "code", item["name"])
    annotations = spec["annotations"]
    canonical_path = _path(annotations["canonical_path"], base)
    patch_root = _path(annotations["patch_csv_dir"], base)
    label_root = _path(annotations["label_csv_dir"], base)
    bank_roots = {item["name"]: _path(item["path"], base) for item in spec["banks"]}
    roots = {"patch_csv": patch_root, "label_csv": label_root, **bank_roots}
    resolved = [root.resolve(strict=True) for root in roots.values()]
    _require(len(set(resolved)) == len(roots) and all(root.is_dir() for root in roots.values()),
             "annotation/feature input roots must be distinct directories")
    root_bindings = {name: {"path": str(root), "resolved_path": str(root.resolve(strict=True))}
                     for name, root in roots.items()}
    lineage_paths = [_path(item[field], base) for item in spec["lineages"]
                     for field in ("slide_info", "train_indices", "test_indices",
                                   "train_labels", "test_labels")]
    paths = [Path(item["path"]) for item in bindings] + [canonical_path, *lineage_paths]
    out = _path(out, Path.cwd())
    out_root = _path(out_root, Path.cwd())
    claimed_target = _output_path(out, out_root, paths, list(roots.values()))
    directory_fd = _claim_output(out, claimed_target)
    owned = False
    try:
        _claimed_path(directory_fd, claimed_target)
        _require(_output_path(out, out_root, paths, list(roots.values()))
                 == claimed_target, "output retargeted before ownership established")
        owned = True
        return _audit_claimed(spec, base, read, bindings, aliases, metadata, canonical_path,
                              patch_root, label_root, bank_roots, roots, root_bindings,
                              directory_fd, out, out_root, claimed_target, _time())
    except Exception as exc:
        if owned:
            try:
                _atomic_bytes(directory_fd, "FAILED.json", stage0._encoded({
                    "failed_at": _time(), "type": type(exc).__name__, "error": str(exc)}))
            except Exception:
                pass
        raise
    finally:
        os.close(directory_fd)


def _audit_claimed(spec, base, read, bindings, aliases, metadata, canonical_path,
                   patch_root, label_root, bank_roots, roots, root_bindings,
                   directory_fd, out, out_root, claimed_target, start):
    canonical_data, _ = read(canonical_path, "canonical", "canonical")
    canonical_obj = _pickle(canonical_data, "annotation canonical")
    _require(isinstance(canonical_obj, list), "annotation canonical: expected list of tuples")
    canonical = {}
    for position, entry in enumerate(canonical_obj):
        _require(isinstance(entry, tuple) and len(entry) == 2,
                 f"annotation canonical entry {position}: expected tuple")
        sid, tiles = entry
        sid = _safe_id(sid, f"annotation canonical entry {position}")
        _require(not sid.endswith(".svs"),
                 f"annotation canonical entry {position}: expected bare slide ID")
        _require(sid not in canonical, f"annotation canonical: duplicate slide {sid}")
        canonical[sid] = _coords(tiles, f"annotation canonical {sid}")
    del canonical_obj, canonical_data
    expected = spec["expected"]
    lineage_data = {}
    lineage_config = {entry["name"]: entry for entry in spec["lineages"]}
    for name in LINEAGES:
        config = lineage_config[name]
        csv_data, _ = read(_path(config["slide_info"], base),
                           "lineage", f"lineage:{name}:slide_info")
        rows = _csv(csv_data, ("slide_id", "wsi_loc_x", "wsi_loc_y"),
                    f"lineage {name}")
        ordered, indices, by_slide = [], {}, defaultdict(list)
        for number, row in enumerate(rows, 2):
            sid = _safe_id(row["slide_id"], f"lineage {name} row {number} slide_id")
            xy = (_decimal(row["wsi_loc_x"], f"lineage {name} row {number} x"),
                  _decimal(row["wsi_loc_y"], f"lineage {name} row {number} y"))
            key = _key(sid, xy)
            _require(key not in indices, f"lineage {name}: duplicate physical key {key}")
            indices[key] = len(ordered)
            ordered.append(key)
            by_slide[sid].append(xy)
        _require(len(ordered) == expected["historical_rows"]
                 and len({key[0] for key in ordered}) == expected["historical_slides"],
                 f"lineage {name}: historical row/slide count mismatch")
        controls = {}
        for part in ("train", "test"):
            for field in ("indices", "labels"):
                data, _ = read(_path(config[f"{part}_{field}"], base),
                               "lineage", f"lineage:{name}:{part}_{field}")
                controls[f"{part}_{field}"] = _npy(data, f"lineage {name} {part}_{field}")
        labels = {}
        used = set()
        for part in ("train", "test"):
            positions = controls[f"{part}_indices"]
            values = controls[f"{part}_labels"]
            _require(len(positions) == len(values) == expected[f"{part}_rows"],
                     f"lineage {name}: {part} index/label count mismatch")
            for pos, value in zip(positions, values):
                row_index, code = int(pos), int(value)
                _require(0 <= row_index < len(ordered) and row_index not in used,
                         f"lineage {name}: duplicate/out-of-range {part} index {row_index}")
                _require(1 <= code <= 5, f"lineage {name}: invalid tissue label {code}")
                used.add(row_index)
                labels[ordered[row_index]] = code
        _require(len(used) == len(ordered), f"lineage {name}: incomplete index partition")
        lineage_data[name] = {"order": ordered, "by_slide": by_slide,
                              "indices": indices, "labels": labels}
        del controls, rows, csv_data, data
    historical = lineage_data["uni2"]["labels"]
    _require(historical == lineage_data["virchow2"]["labels"],
             "lineages: missing/conflicting physical keys or tissue labels")
    global_mismatches, global_positions = _position_diff(
        lineage_data["uni2"]["order"], lineage_data["virchow2"]["order"])
    historical_slides = sorted({key[0] for key in historical})
    mapped = [sid for sid in historical_slides if sid in metadata]
    quarantine = [sid for sid in historical_slides if sid not in metadata]
    _require(len(mapped) == expected["mapped_slides"]
             and len(quarantine) == expected["quarantined_slides"],
             "historical metadata eligibility slide counts mismatch")
    targets = defaultdict(dict)
    for key, label in historical.items():
        targets[key[0]][(key[2], key[3])] = label
    _require(sum(map(len, (targets[s] for s in mapped))) == expected["mapped_rows"]
             and sum(map(len, (targets[s] for s in quarantine))) == expected["quarantined_rows"],
             "historical metadata eligibility row counts mismatch")

    sources = {}
    slide_rows = []
    diagnostics = []
    nontarget_equal = True
    lineage_csv_equal = {name: True for name in LINEAGES}
    _diagnostic(diagnostics, "lineage_global_order", global_mismatches,
                [{"position": i, "uni2": lineage_data["uni2"]["order"][i],
                  "virchow2": lineage_data["virchow2"]["order"][i]}
                 for i in global_positions])
    for sid in historical_slides:
        path_data, _ = read(patch_root / f"{sid}.csv", "patch", f"patch:{sid}")
        label_data, _ = read(label_root / f"{sid}.csv", "labels", f"labels:{sid}")
        patch_order, patch = _source_csv(path_data, sid, False, f"patch {sid}")
        label_order, label = _source_csv(label_data, sid, True, f"labels {sid}")
        _require(sid in canonical, f"annotation canonical: missing historical slide {sid}")
        canonical_order = canonical[sid]
        mosaic = set(canonical_order)
        for xy, tissue in targets[sid].items():
            _require(xy in patch and xy in label and xy in mosaic and label[xy] == tissue,
                     f"slide {sid}: missing/conflicting target {xy} tissue label {tissue}")
        differences = (("patch_not_label", set(patch) - set(label)),
                       ("label_not_patch", set(label) - set(patch)),
                       ("patch_not_canonical", set(patch) - mosaic),
                       ("canonical_not_patch", mosaic - set(patch)),
                       ("label_not_canonical", set(label) - mosaic),
                       ("canonical_not_label", mosaic - set(label)))
        extra_labelled = sorted(xy for xy, code in label.items()
                                if code > 0 and xy not in targets[sid])
        _diagnostic(diagnostics, "additional_labelled_nontarget", len(extra_labelled),
                    extra_labelled, slide_id=sid)
        for kind, difference in differences:
            ordered = sorted(difference)
            _diagnostic(diagnostics, "nontarget_source_difference", len(ordered),
                        ordered, slide_id=sid, lineage=kind)
            nontarget_equal &= not difference
        flags = []
        for name in LINEAGES:
            lineage_order = lineage_data[name]["by_slide"][sid]
            target_csv_order = [xy for xy in label_order if xy in targets[sid]]
            count, examples = _position_diff(target_csv_order, lineage_order)
            _diagnostic(diagnostics, "lineage_target_order", count, examples,
                        slide_id=sid, lineage=name)
            flags.append(count == 0)
            lineage_csv_equal[name] &= count == 0
        eligible = sid in metadata
        patient, group = metadata[sid] if eligible else ("", "")
        slide_rows.append((sid, "mapped" if eligible else "quarantined", patient, group,
                           len(targets[sid]), len(patch), len(label), len(mosaic),
                           len(extra_labelled), *(len(part) for _, part in differences),
                           *("true" if flag else "false" for flag in flags)))
        sources[sid] = (patch_order, patch, label_order, label, canonical_order)
    extra_canonical = sorted(set(canonical) - set(historical_slides))
    _diagnostic(diagnostics, "additional_canonical_slides", len(extra_canonical),
                extra_canonical)
    nontarget_equal &= not extra_canonical
    del canonical

    features = {name: {} for name in BANKS}
    feature_rows = []
    feature_order_equal = {name: True for name in BANKS}
    bank_config = {item["name"]: item for item in spec["banks"]}
    for bank in BANKS:
        config = bank_config[bank]
        for sid in mapped:
            path = bank_roots[bank] / f"{sid}.pkl"
            data, item = read(path, "feature", f"feature:{bank}/{sid}", FEATURE_LIMIT)
            obj = _pickle(data, f"bank {bank}/{sid}")
            del data
            _require(isinstance(obj, tuple) and len(obj) == 2,
                     f"bank {bank}/{sid}: expected (features, mosaic) tuple")
            array, tiles = obj
            _require(isinstance(array, np.ndarray) and array.ndim == 2
                     and array.shape[1] == config["feature_dim"]
                     and array.dtype.kind == "f"
                     and array.dtype.name in config["dtypes"]
                     and array.dtype.isnative
                     and bool(np.isfinite(array).all()),
                     f"bank {bank}/{sid}: invalid raw shape, dtype or nonfinite values")
            coords = _coords(tiles, f"bank {bank}/{sid}")
            _require(len(coords) == array.shape[0],
                     f"bank {bank}/{sid}: feature row/coordinate count mismatch")
            indices = {xy: i for i, xy in enumerate(coords)}
            _require(set(targets[sid]) <= set(indices),
                     f"bank {bank}/{sid}: missing target feature coordinates")
            features[bank].update({_key(sid, xy): indices[xy] for xy in targets[sid]})
            patch_order, patch, label_order, labels, canonical_order = sources[sid]
            common = set(patch) & set(labels) & set(canonical_order)
            unannotated = sum(xy not in targets[sid] and xy in common
                              and labels[xy] == 0 for xy in coords)
            labelled = sum(xy not in targets[sid] and xy in common
                           and labels[xy] > 0 for xy in coords)
            outside = len(coords) - len(targets[sid]) - unannotated - labelled
            _require(outside >= 0 and len(coords) == len(targets[sid]) + unannotated
                     + labelled + outside, f"bank {bank}/{sid}: unused row partition error")
            outside_coords = sorted(xy for xy in coords if xy not in targets[sid]
                                    and xy not in common)
            _diagnostic(diagnostics, "unused_outside_declared_sources", outside,
                        outside_coords, slide_id=sid, bank=bank)
            mismatches, examples = _position_diff(coords, label_order)
            _diagnostic(diagnostics, "feature_csv_order", mismatches, examples,
                        slide_id=sid, bank=bank)
            feature_order_equal[bank] &= mismatches == 0
            feature_rows.append((bank, sid, config["feature_dim"], *config["projection"],
                                 len(coords), array.dtype.name, len(targets[sid]),
                                 unannotated, labelled, outside,
                                 "true" if mismatches == 0 else "false", mismatches,
                                 item["path"], item["resolved_path"], item["size"],
                                 item["sha256"]))
            del obj, array, tiles
    for first, second in aliases:
        _diagnostic(diagnostics, "declared_file_alias", 1, [[first, second]])

    eligible_rows, quarantine_rows = [], []
    for sid in historical_slides:
        for xy, tissue in sorted(targets[sid].items()):
            key = _key(sid, xy)
            prefix = (_tile_id(key), sid)
            lineage_indices = tuple(lineage_data[name]["indices"][key] for name in LINEAGES)
            if sid in metadata:
                patient, group = metadata[sid]
                eligible_rows.append((*prefix, patient, group, 0, *xy, 512, tissue,
                                      *lineage_indices,
                                      *(features[bank][key] for bank in BANKS)))
            else:
                quarantine_rows.append((*prefix, 0, *xy, 512, tissue,
                                        *lineage_indices, "slide_absent_from_bound_metadata"))
    def class_support(sids, patient_support):
        row_counts = {c: sum(label == int(c) for sid in sids
                             for label in targets[sid].values()) for c in CLASSES}
        slide_counts = {c: len({sid for sid in sids
                                if int(c) in targets[sid].values()}) for c in CLASSES}
        result = {"rows": sum(len(targets[sid]) for sid in sids), "slides": len(sids),
                  "class_rows": row_counts, "class_slides": slide_counts}
        if patient_support:
            result["patients"] = len({metadata[sid][0] for sid in sids})
            result["class_patients"] = {
                c: len({metadata[sid][0] for sid in sids
                        if int(c) in targets[sid].values()}) for c in CLASSES}
        return result

    _require(class_support(mapped, True)["class_rows"] == expected["mapped_class_counts"]
             and class_support(quarantine, False)["class_rows"]
             == expected["quarantined_class_counts"],
             "historical tissue class counts differ from declared expected")
    def finish():
        paths = [Path(item["path"]) for item in bindings]
        validated_features = {(row[0], row[1]): tuple(map(str, row))
                              for row in feature_rows}
        rows = {"eligible_tiles.csv": [tuple(map(str, row)) for row in eligible_rows],
                "quarantine_tiles.csv": [tuple(map(str, row)) for row in quarantine_rows],
                "slide_inventory.csv": [tuple(map(str, row)) for row in slide_rows],
                "feature_inventory.csv": [tuple(map(str, row)) for row in feature_rows]}
        eligible_rows.clear()
        quarantine_rows.clear()
        summary = {
            "schema_version": 1, "run_id": spec["run_id"],
            "scope": "historical-annotation target keyed binding only",
            "declarations": spec,
            "historical": {"rows": len(historical), "slides": len(historical_slides)},
            "keyed_join_status": "complete",
            "eligible": class_support(mapped, True),
            "quarantine": class_support(quarantine, False),
            "lineage_order_equal": global_mismatches == 0,
            "lineage_csv_target_order_equal": lineage_csv_equal,
            "mapped_feature_csv_order_equal": feature_order_equal,
            "nontarget_source_sets_equal": nontarget_equal,
            "legacy_X_validation": "not_performed",
            "encoder_identity": "declared_unverified",
            "source_inventory_totals": {
                field: sum(int(row[SLIDE_HEADER.index(field)]) for row in slide_rows)
                for field in SLIDE_HEADER[4:15]},
            "additional_canonical_slides": len(extra_canonical),
            "feature_inventory_totals": {
                bank: {field: sum(int(row[FEATURE_HEADER.index(field)])
                                  for row in feature_rows if row[0] == bank)
                       for field in ("feature_rows", "target_rows", "unused_unannotated",
                                     "unused_labelled_nontarget",
                                     "unused_outside_declared_sources")}
                for bank in BANKS},
            "diagnostic_entries": len(diagnostics), "input_bindings": bindings,
            "input_roots": root_bindings,
            "verification_started_at": start,
            "source_recheck_completed_at": None,
            "environment": {"python": sys.version, "executable": sys.executable,
                            "numpy": np.__version__, "pandas": pd.__version__,
                            "hostname": socket.gethostname(), "platform": platform.platform(),
                            "lsf": {key: os.environ[key] for key in
                                    ("LSB_JOBID", "LSB_JOBINDEX", "LSB_QUEUE", "LSB_HOSTS")
                                    if os.environ.get(key)}},
        }
        diagnostic_document = {"schema_version": 1, "run_id": spec["run_id"],
                               "entries": diagnostics}
        slide_rows.clear()
        feature_rows.clear()
        written = {}
        for name, header in OUTPUT_HEADERS.items():
            written[name] = _write_csv(directory_fd, name, header, rows[name])
        for name, root in roots.items():
            _require(str(root.resolve(strict=True)) == root_bindings[name]["resolved_path"],
                     f"input root {name}: late retarget")
        for item in bindings:
            current = _binding(item["path"], item["kind"], item["name"])
            _require(current == item, f"{item['kind']} {item['name']}: late input mutation")
        summary["source_recheck_completed_at"] = _time()
        written["diagnostics.json"] = stage0._encoded(diagnostic_document)
        written["summary.json"] = stage0._encoded(summary)
        _atomic_bytes(directory_fd, "diagnostics.json", written["diagnostics.json"])
        _atomic_bytes(directory_fd, "summary.json", written["summary.json"])
        _require(_output_path(out, out_root, paths, list(roots.values()))
                 == claimed_target, "output retargeted during audit")
        _claimed_path(directory_fd, claimed_target)
        hashes = _verify_outputs(directory_fd, rows, written, summary, diagnostic_document,
                                 metadata, historical, lineage_data, features, sources,
                                 expected, extra_canonical, aliases, validated_features)
        _require(_output_path(out, out_root, paths, list(roots.values()))
                 == claimed_target, "output retargeted before completion")
        _claimed_path(directory_fd, claimed_target)
        for name, item in hashes.items():
            current = _read_output(directory_fd, name)
            _require(len(current) == item["size"]
                     and hashlib.sha256(current).hexdigest() == item["sha256"],
                     f"{name}: final output hash changed")
        receipt = {
            "schema_version": 1, "run_id": spec["run_id"],
            "scope": "historical-annotation target keyed binding only",
            "outputs": hashes, "initial_verification_at": start,
            "final_verification_at": _time(),
            "limitations": [
                "No proof of annotation correctness, encoder/weight/pixel identity, "
                "legacy-X fidelity, folds, training, matrices or scientific acceptance.",
                "Cooperative output-directory ownership; empty substitution before first "
                "observation cannot be detected.",
                "Two passes are not an atomic snapshot: freeze inputs and enforce "
                "recorded hashes downstream. Pickle inputs must be trusted locally.",
            ],
        }
        _atomic_bytes(directory_fd, "COMPLETE.json", stage0._encoded(receipt))
        return diagnostic_document
    return finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        result = audit(args.spec, args.spec_sha256, args.out, os.environ.get("PROSPER_OUT"))
    except Exception as exc:
        print(f"annotation-bank audit FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for entry in result["entries"]:
        print(f"annotation-bank audit WARNING: {entry['kind']} "
              f"{entry.get('slide_id', '')} {entry.get('bank', entry.get('lineage', ''))} "
              f"count={entry['count']}", file=sys.stderr)
    print(f"annotation-bank audit COMPLETE: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
