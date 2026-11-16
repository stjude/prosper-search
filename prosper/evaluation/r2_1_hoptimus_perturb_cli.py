"""Bound CPU population/stimulus manifests; no image, model or feature reads."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import hashlib
import io
from pathlib import Path
import re
import sys
from typing import Mapping

from . import feature_bank_audit as legacy
from . import r2_1_hoptimus_perturb as kernel


TASK = "hoptimus_perturb_manifest"
ARTIFACTS = {"queries.csv", "references.csv", "stimuli.json", "summary.json"}
SOURCE_NAMES = {
    "metadata", "stage0_review", "stage0_complete", "stage0_inventory",
    "stage0_summary", "kernel_review", "template",
}
EXPECTED_KEYS = {
    "raw_slides", "target_slides", "patients", "groups",
    "cross_label_patients", "stimulus_records", "feature_dim",
}
METADATA_COLUMNS = {
    "id_col": "Slide.Scan.File", "patient_col": "UID.Subject", "label_col": "Tumor.group",
}
INVENTORY_HEADER = (
    "bank", "slide_id", "canonical_tiles", "feature_rows", "dtype",
    "feature_path", "resolved_path", "size", "sha256",
)
POPULATION_HEADER = (
    "population_index", "metadata_index", "slide_id", "patient_id",
    "diagnostic_group", "feature_rows", "feature_path", "resolved_path",
    "feature_size", "feature_sha256",
)
LIMITATIONS = [
    "Feature payloads were not read; width and feature bindings inherit accepted Stage0 evidence.",
    "Template bytes are bound but not decoded or fitted; no image transformation was applied.",
    "No model, reference aggregation, extraction, retrieval or identity/reference compatibility is accepted.",
]


def _require(condition: bool, message: str) -> None:
    legacy._require(condition, message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash(value: object, context: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None,
             f"{context}: expected SHA256")
    return value.lower()


def _count(value: object, context: str) -> int:
    legacy._integer(value, context)
    return value


def _integer_cell(value: str, context: str) -> int:
    _require(re.fullmatch(r"0|[1-9][0-9]*", value) is not None,
             f"{context}: expected canonical nonnegative integer")
    return int(value)


def _source_path(spec: dict, name: str, base: Path) -> Path:
    path = Path(spec["sources"][name]["path"])
    return (path if path.is_absolute() else base / path).resolve()


def _csv_rows(data: bytes, header: tuple[str, ...], context: str):
    with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", newline="") as text:
        reader = csv.reader(text, strict=True)
        _require(tuple(next(reader, ())) == header, f"{context}: wrong CSV header")
        for index, row in enumerate(reader):
            _require(len(row) == len(header), f"{context}: malformed/trailing row {index}")
            yield dict(zip(header, row))


def _csv_bytes(header: tuple[str, ...], rows) -> bytes:
    buffer = io.BytesIO()
    text = io.TextIOWrapper(buffer, encoding="utf-8", newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    text.flush()
    data = buffer.getvalue()
    text.detach()
    return data


def _artifact_link(review: dict, review_path: Path, source_path: Path, digest: str) -> None:
    artifacts = review.get("artifacts")
    _require(isinstance(artifacts, dict), "Stage0 review: missing artifacts")
    matches = []
    for relative, value in artifacts.items():
        _require(isinstance(relative, str), "Stage0 review: invalid artifact name")
        _hash(value, "Stage0 review artifact")
        path = Path(relative)
        _require(not path.is_absolute() and ".." not in path.parts,
                 "Stage0 review: escaping artifact link")
        linked = (review_path.parent / path).resolve()
        _require(linked.is_relative_to(review_path.parent),
                 "Stage0 review: artifact outside review directory")
        if linked == source_path:
            matches.append(value.lower())
    _require(matches == [digest], "Stage0 review: selected artifact link/hash mismatch")


def _stage0_controls(spec: dict, sources: Mapping[str, bytes], base: Path) -> dict:
    review = legacy._json(sources["stage0_review"])
    complete = legacy._json(sources["stage0_complete"])
    summary = legacy._json(sources["stage0_summary"])
    _require(all(isinstance(value, dict) for value in (review, complete, summary)),
             "Stage0 controls must be JSON objects")
    _require(isinstance(summary.get("declarations"), dict)
             and isinstance(summary["declarations"].get("metadata"), dict),
             "Stage0 declarations/metadata must be objects")
    _require(review.get("schema_version") == 1 and type(review.get("schema_version")) is int,
             "Stage0 review: wrong schema")
    _require(review.get("disposition") == "accepted"
             and review.get("scope") == "Stage0 feature-bank binding only",
             "Stage0 review: acceptance/scope mismatch")
    _require(type(complete.get("schema_version")) is int and complete["schema_version"] == 1
             and complete.get("scope") == "feature-bank binding only",
             "Stage0 COMPLETE: schema/scope mismatch")
    _require(type(summary.get("schema_version")) is int and summary["schema_version"] == 1
             and summary.get("scope") == "feature-bank binding only; encoder identities declared/unverified",
             "Stage0 summary: schema/scope mismatch")
    legacy._text(complete.get("run_id"), "Stage0 run ID")
    _require(summary.get("run_id") == complete["run_id"]
             == summary.get("declarations", {}).get("run_id"), "Stage0 run identity mismatch")
    review_path = _source_path(spec, "stage0_review", base)
    complete_path = _source_path(spec, "stage0_complete", base)
    for name in ("stage0_complete", "stage0_inventory", "stage0_summary"):
        path = _source_path(spec, name, base)
        _artifact_link(review, review_path, path, _sha(sources[name]))
        if name != "stage0_complete":
            _require(path.parent == complete_path.parent, "Stage0 selected attempt mismatch")
            entry = complete.get("outputs", {}).get(path.name)
            legacy._keys(entry, ("sha256", "size"), "Stage0 consumed output")
            _require(_hash(entry["sha256"], "Stage0 output") == _sha(sources[name])
                     and _count(entry["size"], "Stage0 output size") == len(sources[name]),
                     "Stage0 COMPLETE output binding mismatch")
    _require(_hash(review.get("source_sha256", {}).get("metadata"), "Stage0 metadata hash")
             == _sha(sources["metadata"]), "Stage0 review metadata binding mismatch")
    bindings = summary.get("input_bindings")
    _require(isinstance(bindings, list), "Stage0 summary: missing input bindings")
    metadata_bindings = [
        b for b in bindings if isinstance(b, dict)
        and b.get("kind") == "metadata" and b.get("name") == "metadata"
    ]
    _require(len(metadata_bindings) == 1, "Stage0 metadata binding role is not unique")
    binding = metadata_bindings[0]
    _require(_hash(binding.get("sha256"), "Stage0 metadata") == _sha(sources["metadata"])
             and _count(binding.get("size"), "Stage0 metadata size") == len(sources["metadata"]),
             "Stage0 summary metadata size/hash mismatch")
    metadata_path = _source_path(spec, "metadata", base)
    _require(str(metadata_path) in {binding.get("path"), binding.get("resolved_path")},
             "Stage0 recorded metadata identity mismatch")
    _require(summary.get("declarations", {}).get("metadata", {}).get("id_col")
             == METADATA_COLUMNS["id_col"]
             and summary["declarations"]["metadata"].get("patient_col")
             == METADATA_COLUMNS["patient_col"]
             and summary["declarations"]["metadata"].get("label_col")
             == METADATA_COLUMNS["label_col"], "Stage0 metadata column declaration mismatch")
    return {"review": review, "summary": summary}


def _kernel_control(spec: dict, sources: Mapping[str, bytes]) -> None:
    review = legacy._json(sources["kernel_review"])
    _require(isinstance(review, dict) and isinstance(review.get("source_files"), dict),
             "kernel review must declare source files")
    _require(type(review.get("schema_version")) is int and review["schema_version"] == 1,
             "kernel review: wrong schema")
    _require(review.get("disposition") == "accepted"
             and review.get("scope") == "Repaired pure stimulus/SAMPLER/L1 kernels only",
             "kernel review: acceptance/scope mismatch")
    digest = review.get("source_files", {}).get("prosper/evaluation/r2_1_hoptimus_perturb.py")
    _require(_hash(digest, "kernel review module") == _hash(
        spec["code_files"]["kernel"]["sha256"], "executing kernel"
    ), "kernel review: executing module hash mismatch")


def _population(spec: dict, sources: Mapping[str, bytes], controls: dict):
    summary, review = controls["summary"], controls["review"]
    headers = summary.get("metadata_columns")
    _require(isinstance(headers, list) and all(isinstance(h, str) and h for h in headers)
             and len(headers) == len(set(headers)), "Stage0 metadata header declaration invalid")
    _require(set(METADATA_COLUMNS.values()).issubset(headers), "missing metadata columns")
    metadata = []
    slide_ids: set[str] = set()
    raw_patient_groups = defaultdict(set)
    for index, row in enumerate(_csv_rows(sources["metadata"], tuple(headers), "metadata")):
        sid, patient, group = (row[METADATA_COLUMNS[k]] for k in ("id_col", "patient_col", "label_col"))
        for value in (sid, patient, group):
            legacy._text(value, "metadata identity")
        _require(sid not in slide_ids, "duplicate metadata slide")
        slide_ids.add(sid)
        raw_patient_groups[patient].add(group)
        metadata.append((index, sid, patient, group))
    target = [row for row in metadata if row[3] != "MISC"]
    _require(bool(target), "empty target population")
    patient_groups = defaultdict(set)
    for _, _, patient, group in target:
        patient_groups[patient].add(group)
    counts = {
        "raw_slides": len(metadata), "target_slides": len(target),
        "patients": len(patient_groups), "groups": len({r[3] for r in target}),
        "cross_label_patients": sum(len(groups) > 1 for groups in patient_groups.values()),
        "stimulus_records": 12 * len(target), "feature_dim": kernel.FEATURE_DIM,
    }
    _require(counts == spec["expected"], "observed population counts differ from declaration")
    reconciled = review.get("reconciled", {})
    for key, value in (
        ("raw_slides_per_bank", len(metadata)), ("raw_patients", len(raw_patient_groups)),
        ("raw_multi_label_patients", sum(len(g) > 1 for g in raw_patient_groups.values())),
        ("non_misc_slides", len(target)), ("non_misc_patients", len(patient_groups)),
        ("non_misc_multi_label_patients", counts["cross_label_patients"]),
    ):
        _require(_count(reconciled.get(key), f"Stage0 reconciled {key}") == value,
                 f"Stage0 review count mismatch: {key}")
    for key, value in (
        ("canonical_slides", len(metadata)), ("patient_count", len(raw_patient_groups)),
        ("multi_label_patient_count", sum(len(g) > 1 for g in raw_patient_groups.values())),
    ):
        _require(_count(summary.get(key), f"Stage0 {key}") == value,
                 f"Stage0 summary count mismatch: {key}")
    banks = summary.get("declarations", {}).get("banks")
    _require(isinstance(banks, list) and all(isinstance(b, dict) for b in banks),
             "Stage0 bank declarations missing")
    names = [b.get("name") for b in banks]
    _require(all(isinstance(name, str) and name for name in names)
             and len(names) == len(set(names)), "duplicate/invalid Stage0 banks")
    selected = [b for b in banks if b["name"] == "h_optimus_1"]
    _require(len(selected) == 1
             and _count(selected[0].get("feature_dim"), "H-Optimus width") == kernel.FEATURE_DIM,
             "H-Optimus declared width must be 1536")
    bindings = {}
    for binding in summary["input_bindings"]:
        if isinstance(binding, dict) and binding.get("kind") == "feature":
            name = binding.get("name")
            if isinstance(name, str) and name.startswith("h_optimus_1/"):
                _require(name not in bindings, "duplicate Stage0 H-Optimus binding")
                bindings[name] = binding
    inventory = {}
    seen = set()
    row_count = 0
    tile_total = 0
    for row in _csv_rows(sources["stage0_inventory"], INVENTORY_HEADER, "Stage0 inventory"):
        row_count += 1
        key = (row["bank"], row["slide_id"])
        _require(key not in seen and key[0] in names and key[1] in slide_ids,
                 "duplicate/unknown inventory identity")
        seen.add(key)
        if key[0] != "h_optimus_1":
            continue
        feature_rows = _integer_cell(row["feature_rows"], "feature rows")
        _require(feature_rows == _integer_cell(row["canonical_tiles"], "canonical tiles"),
                 "inventory feature/coordinate counts mismatch")
        _require(row["dtype"] in selected[0].get("dtypes", ()), "undeclared feature dtype")
        size = _integer_cell(row["size"], "feature size")
        digest = _hash(row["sha256"], "feature hash")
        for value in (row["feature_path"], row["resolved_path"]):
            legacy._text(value, "feature identity")
        binding = bindings.get(f"h_optimus_1/{key[1]}", {})
        _require(binding.get("path") == row["feature_path"]
                 and binding.get("resolved_path") == row["resolved_path"]
                 and _count(binding.get("size"), "feature binding size") == size
                 and _hash(binding.get("sha256"), "feature binding hash") == digest,
                 "inventory/summary feature binding mismatch")
        inventory[key[1]] = (
            feature_rows, row["feature_path"], row["resolved_path"], size, row["sha256"],
        )
        tile_total += feature_rows
    _require(set(inventory) == slide_ids
             and set(bindings) == {f"h_optimus_1/{sid}" for sid in slide_ids},
             "H-Optimus raw population coverage mismatch")
    _require(_count(summary.get("inventory_rows"), "inventory rows") == row_count,
             "Stage0 inventory row count mismatch")
    _require(_count(summary.get("canonical_tiles"), "canonical tiles") == tile_total
             and _count(summary.get("bank_tile_totals", {}).get("h_optimus_1"),
                        "H-Optimus tile total") == tile_total,
             "Stage0 tile total mismatch")
    population = tuple(
        (index, metadata_index, sid, patient, group, *inventory[sid])
        for index, (metadata_index, sid, patient, group) in enumerate(target)
    )
    return population, counts


def _read_stimuli(data: bytes, target_ids: set[str], template_digest: str):
    decoded = legacy._json(data)
    legacy._keys(decoded, ("policy", "records"), "stimuli")
    legacy._keys(decoded["policy"], (
        "name", "seed", "key_encoding", "seed_derivation", "algorithm", "numpy_version",
    ), "stimulus policy")
    _require(isinstance(decoded["records"], list), "stimuli records must be an array")
    typed = []
    for record in decoded["records"]:
        legacy._keys(record, ("slide_id", "condition", "level", "patch_size", "parameters"),
                     "stimulus record")
        _require(isinstance(record["parameters"], dict), "stimulus parameters must be an object")
        if record["condition"] == "reinhard":
            _require(record["parameters"].get("template_sha256") == template_digest,
                     "stimulus/bound-template digest mismatch")
        typed.append(kernel.StimulusRecord(
            record["slide_id"], record["condition"], record["level"], record["patch_size"],
            tuple(record["parameters"].items()),
        ))
    manifest = kernel.StimulusManifest(kernel.StimulusPolicy(**decoded["policy"]), tuple(typed))
    _require(kernel.serialize_stimulus_manifest(manifest) == data,
             "stimuli are not canonical serialized bytes")
    _require({r.slide_id for r in manifest.records} == target_ids,
             "serialized stimuli target set mismatch")
    _require(len(manifest.records) == 12 * len(target_ids), "serialized stimulus coverage mismatch")
    return decoded


def validate_artifacts(
    spec: dict, artifacts: Mapping[str, bytes], population: tuple,
    counts: dict, template_digest: str,
) -> None:
    """Reparse and reconcile produced bytes before handing them to shared IO."""
    _require(set(artifacts) == ARTIFACTS and all(type(v) is bytes for v in artifacts.values()),
             "wrong branch artifact set/type")
    for name in ("queries.csv", "references.csv"):
        _require(artifacts[name].endswith(b"\n"), "population CSV must end with LF")
        seen = 0
        for index, row in enumerate(_csv_rows(artifacts[name], POPULATION_HEADER, name)):
            _require(index < len(population)
                     and tuple(row[h] for h in POPULATION_HEADER)
                     == tuple(str(v) for v in population[index]), "population readback mismatch")
            seen += 1
        _require(seen == len(population), "population readback has missing rows")
        _require(_csv_bytes(POPULATION_HEADER, population) == artifacts[name],
                 "population CSV is not canonical")
    target_ids = {row[2] for row in population}
    stimuli = _read_stimuli(artifacts["stimuli.json"], target_ids, template_digest)
    summary = legacy._json(artifacts["summary.json"])
    legacy._keys(summary, ("schema_version", "run_id", "task", "counts", "policy",
                          "details", "limitations"), "summary")
    _require(type(summary["schema_version"]) is int and summary["schema_version"] == 1
             and summary["run_id"] == spec["run_id"] and summary["task"] == TASK,
             "summary run/schema/task mismatch")
    legacy._keys(summary["counts"], EXPECTED_KEYS, "summary counts")
    for name, value in summary["counts"].items():
        _count(value, f"summary count {name}")
    _require(summary["counts"] == counts == spec["expected"], "summary counts mismatch")
    _require(kernel.canonical_json_bytes(summary["policy"])
             == kernel.canonical_json_bytes(stimuli["policy"]),
             "summary stimulus policy mismatch")
    legacy._keys(summary["details"], (
        "condition_order", "condition_counts", "template_sha256", "feature_dim",
    ), "summary details")
    details = summary["details"]
    _require(details["condition_order"] == list(kernel.CONDITION_ORDER),
             "summary condition order mismatch")
    legacy._keys(details["condition_counts"], kernel.CONDITION_ORDER, "condition counts")
    for name, value in details["condition_counts"].items():
        _require(_count(value, name) == len(population), "summary condition count mismatch")
    _require(details["template_sha256"] == template_digest
             and _count(details["feature_dim"], "summary width") == kernel.FEATURE_DIM,
             "summary template/width mismatch")
    _require(summary["limitations"] == LIMITATIONS, "summary limitations mismatch")
    _require(kernel.canonical_json_bytes(summary) == artifacts["summary.json"],
             "summary is not canonical JSON")


def execute(
    spec: dict, sources: Mapping[str, bytes], *, declaration_base: Path,
) -> dict[str, bytes]:
    """Produce artifacts only from the shared layer's explicitly bound bytes."""
    _require(spec.get("task") == TASK and set(sources) == SOURCE_NAMES,
             "wrong perturbation task/source roles")
    legacy._keys(spec["parameters"], ("metadata",), "parameters")
    legacy._keys(spec["parameters"]["metadata"], METADATA_COLUMNS, "metadata parameters")
    _require(spec["parameters"]["metadata"] == METADATA_COLUMNS, "wrong metadata columns")
    legacy._keys(spec["expected"], EXPECTED_KEYS, "expected")
    for name, value in spec["expected"].items():
        _count(value, f"expected {name}")
    controls = _stage0_controls(spec, sources, declaration_base)
    _kernel_control(spec, sources)
    population, counts = _population(spec, sources, controls)
    template_digest = _sha(sources["template"])
    manifest = kernel.generate_stimulus_manifest([row[2] for row in population], template_digest)
    stimuli = kernel.serialize_stimulus_manifest(manifest)
    summary = {
        "schema_version": 1, "run_id": spec["run_id"], "task": TASK, "counts": counts,
        "policy": asdict(manifest.policy),
        "details": {
            "condition_order": list(kernel.CONDITION_ORDER),
            "condition_counts": {name: len(population) for name in kernel.CONDITION_ORDER},
            "template_sha256": template_digest, "feature_dim": kernel.FEATURE_DIM,
        },
        "limitations": LIMITATIONS,
    }
    population_bytes = _csv_bytes(POPULATION_HEADER, population)
    artifacts = {
        "queries.csv": population_bytes, "references.csv": population_bytes,
        "stimuli.json": stimuli, "summary.json": kernel.canonical_json_bytes(summary),
    }
    validate_artifacts(spec, artifacts, population, counts, template_digest)
    return artifacts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    args = parser.parse_args(argv)
    from . import r2_1_hoptimus_io as bound

    code_paths = {
        "runner": Path(__file__), "kernel": Path(kernel.__file__),
        "io": Path(bound.__file__), "legacy_io": Path(legacy.__file__),
        "package_init": Path(sys.modules["prosper"].__file__),
        "evaluation_init": Path(sys.modules["prosper.evaluation"].__file__),
    }
    try:
        bound.run_bound(
            args.spec, args.spec_sha256, args.out, args.out_root,
            task=TASK, code_paths=code_paths,
            execute=lambda spec, sources: execute(
                spec, sources, declaration_base=args.spec.absolute().parent
            ),
        )
    except (ValueError, TypeError, KeyError, UnicodeError, csv.Error,
            kernel.KernelComputationError) as exc:
        parser.exit(1, f"perturbation manifest failure: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
