"""Bound, cooperative-snapshot CPU artifact IO for the three R2.1 branches."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from . import feature_bank_audit as legacy


METHODS = ("median_pooling", "sampler", "ot_cosine", "yottixel")
CODE_NAMES = frozenset(("runner", "kernel", "io", "legacy_io",
                        "package_init", "evaluation_init"))
COMMON_SOURCES = frozenset(("metadata", "stage0_review", "stage0_complete",
                           "stage0_inventory", "stage0_summary", "kernel_review"))
COMMON_EXPECTED = frozenset(("raw_slides", "target_slides", "patients",
                            "groups", "cross_label_patients"))
ARTIFACTS = {
    "hoptimus_main": frozenset((
        "populations.csv", "query_metrics.csv.gz", "estimates.csv", "contrasts.csv",
        "tie_sensitivity.csv.gz", "summary.json", "patient_draws.npy",
        "replicate_index.csv", "replicates.npy", "replicate_valid.npy")),
    "hoptimus_filter_manifest": frozenset((
        "retrieval_population.csv", "patient_folds.csv", "annotation_assignments.csv.gz",
        "training_rows.csv.gz", "head_plan.json", "summary.json")),
    "hoptimus_perturb_manifest": frozenset((
        "queries.csv", "references.csv", "stimuli.json", "summary.json")),
}
EXTRA_SOURCES = {
    "hoptimus_main": frozenset(
        f"{kind}_{method}" for kind in ("ranks", "matrix") for method in METHODS),
    "hoptimus_filter_manifest": frozenset((
        "stage1a_review", "stage1a_complete", "stage1a_eligible",
        "stage1a_feature_inventory", "stage1a_summary")),
    "hoptimus_perturb_manifest": frozenset(("template",)),
}
EXTRA_EXPECTED = {
    "hoptimus_main": frozenset(),
    "hoptimus_filter_manifest": frozenset((
        "annotation_tiles", "annotation_slides", "annotation_patients",
        "quarantine_tiles", "quarantine_slides")),
    "hoptimus_perturb_manifest": frozenset(("stimulus_records", "feature_dim")),
}
SCOPES = {
    "hoptimus_main": "hoptimus_main_saved_rank_evaluation_only",
    "hoptimus_filter_manifest": "hoptimus_filter_manifests_only",
    "hoptimus_perturb_manifest": "hoptimus_perturb_stimulus_manifests_only",
}
KERNEL_SCOPES = {
    "hoptimus_main": "Repaired pure query-metric and whole-patient inference kernels only",
    "hoptimus_filter_manifest": "Pure filtering fold/training-manifest kernels only",
    "hoptimus_perturb_manifest": "Repaired pure stimulus/SAMPLER/L1 kernels only",
}
METADATA_COLUMNS = {
    "id_col": "Slide.Scan.File", "patient_col": "UID.Subject", "label_col": "Tumor.group",
}
SPEC_LIMIT = 1024 * 1024
SOURCE_LIMIT = 512 * 1024 * 1024
RANK_LIMIT = 64 * 1024 * 1024
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
LIMITATIONS = [
    "Cooperative directory ownership and two-pass inputs are not an atomic snapshot.",
    "Unconsumed upstream feature/image/model payloads were not reread or certified.",
    "Trusted local pickle only; this is not safe ingestion of external pickle.",
]


class BoundError(ValueError):
    """A declaration, input, serialized artifact or ownership check failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundError(message)


def keys(obj: object, expected: Sequence[str] | set[str], context: str) -> None:
    require(isinstance(obj, dict) and set(obj) == set(expected),
            f"{context}: missing/unknown keys")


def text(value: object, context: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{context}: blank/nonstring")
    return value


def integer(value: object, context: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"{context}: invalid integer")
    return value


def digest(value: object, context: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None,
            f"{context}: invalid SHA256")
    return value.lower()


def strict_json(data: bytes) -> object:
    try:
        result = legacy._json(data)
    except (ValueError, UnicodeError) as exc:
        raise BoundError(f"invalid finite/unique-key JSON: {exc}") from exc
    pending = [result]
    while pending:
        value = pending.pop()
        if isinstance(value, float):
            require(np.isfinite(value), "JSON numeric overflow/nonfinite value")
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return result


def json_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                           separators=(",", ":")) + "\n").encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise BoundError(f"invalid finite JSON output: {exc}") from exc


class _LimitedReader(io.RawIOBase):
    def __init__(self, source, limit):
        self.source, self.limit, self.count = source, limit, 0

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.source.read(min(len(buffer), self.limit - self.count + 1))
        self.count += len(data)
        require(self.count <= self.limit, "CSV expansion exceeds bounded read limit")
        buffer[:len(data)] = data
        return len(data)


def csv_rows(
    data: bytes, header: Sequence[str], *, compressed: bool = False,
    max_expanded: int = SOURCE_LIMIT,
) -> Iterator[dict[str, str]]:
    """Stream dict rows under a finite explicit expansion budget.

    Rank callers use RANK_LIMIT. Generated manifests may use a larger
    artifact-derived byte budget and must additionally reconcile record counts.
    """
    integer(max_expanded, "CSV expansion limit", 1)
    raw = io.BytesIO(data)
    source = gzip.GzipFile(fileobj=raw, mode="rb") if compressed else raw
    stream = io.TextIOWrapper(io.BufferedReader(_LimitedReader(source, max_expanded)),
                              encoding="utf-8", newline="")
    try:
        reader = csv.reader(stream, strict=True)
        actual = next(reader, None)
        require(actual == list(header) and len(set(actual)) == len(actual),
                "CSV header missing, duplicate or unknown")
        for row in reader:
            require(len(row) == len(header), "CSV extra/missing/trailing fields or records")
            yield dict(zip(header, row))
    except (OSError, EOFError, UnicodeError, csv.Error) as exc:
        raise BoundError(f"invalid CSV/gzip bytes: {exc}") from exc
    finally:
        stream.close()
        source.close()
        raw.close()


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        require(np.isfinite(value), "nonfinite CSV numeric cell")
        return repr(float(value))
    return str(value)


def csv_bytes(
    header: Sequence[str], rows, *, compressed: bool = False,
) -> bytes:
    """Stream rows into bytes without constructing a full uncompressed string."""
    buffer = io.BytesIO()
    sink = gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) if compressed else buffer
    stream = io.TextIOWrapper(sink, encoding="utf-8", newline="")
    try:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            if isinstance(row, Mapping):
                require(set(row) == set(header), "CSV output row keys mismatch")
                row = [row[k] for k in header]
            require(len(row) == len(header), "CSV output row length mismatch")
            writer.writerow([_cell(v) for v in row])
        stream.flush()
        stream.detach()
        if compressed:
            sink.close()
        return buffer.getvalue()
    finally:
        try:
            stream.close()
        except ValueError:
            pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _scan_file(path, kind, name, limit, *, expected=None, retain=True):
    path = Path(path)
    resolved = str(path.resolve(strict=True))
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= limit,
            f"{name}: not a bounded regular file")
    if expected:
        require(before.st_size == expected["size"], f"{name}: declared size mismatch")
    limit = min(limit, before.st_size)
    hasher, total = hashlib.sha256(), 0
    buffer = io.BytesIO() if retain else None
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        require((before.st_dev, before.st_ino, before.st_size) ==
                (opened.st_dev, opened.st_ino, opened.st_size), f"{name}: changed before open")
        while True:
            chunk = handle.read(min(1024 * 1024, limit - total + 1))
            if not chunk:
                break
            total += len(chunk)
            require(total <= limit, f"{name}: exceeded bounded read")
            hasher.update(chunk)
            if buffer is not None:
                buffer.write(chunk)
        after_fd = os.fstat(handle.fileno())
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size) ==
            (after_fd.st_dev, after_fd.st_ino, after_fd.st_size) ==
            (after.st_dev, after.st_ino, after.st_size)
            and str(path.resolve(strict=True)) == resolved and total == before.st_size,
            f"{name}: input changed while reading")
    item = dict(kind=kind, name=name, path=str(path), resolved_path=resolved, size=total,
                sha256=hasher.hexdigest(), device=after.st_dev, inode=after.st_ino)
    if expected:
        require(item["sha256"] == digest(expected["sha256"], name), f"{name}: SHA256 mismatch")
    return buffer.getvalue() if buffer is not None else None, item


def _schema(spec, task):
    require(task in ARTIFACTS, "unknown task")
    keys(spec, ("schema_version", "run_id", "task", "sources", "code_files",
                "parameters", "expected"), "declaration")
    require(type(spec["schema_version"]) is int and spec["schema_version"] == 1,
            "declaration version must be integer 1")
    text(spec["run_id"], "run_id")
    require(spec["task"] == task, "declaration task mismatch")
    keys(spec["sources"], COMMON_SOURCES | EXTRA_SOURCES[task], "sources")
    keys(spec["code_files"], CODE_NAMES, "code_files")
    for collection in ("sources", "code_files"):
        for name, entry in spec[collection].items():
            text(name, "binding name")
            keys(entry, ("path", "sha256", "size"), name)
            text(entry["path"], f"{name} path")
            digest(entry["sha256"], name)
            integer(entry["size"], name)
            require(entry["size"] <= SOURCE_LIMIT, f"{name}: declared source too large")
    keys(spec["parameters"], ("metadata",), "parameters")
    keys(spec["parameters"]["metadata"], METADATA_COLUMNS, "metadata selectors")
    require(spec["parameters"]["metadata"] == METADATA_COLUMNS, "metadata selectors mismatch")
    keys(spec["expected"], COMMON_EXPECTED | EXTRA_EXPECTED[task], "expected")
    for name, count in spec["expected"].items():
        integer(count, f"expected.{name}")
    require(all(spec["expected"][k] > 0 for k in ("raw_slides", "target_slides", "patients", "groups")),
            "required populations must be nonempty")


def _review(review, scope, context):
    require(isinstance(review, dict) and type(review.get("schema_version")) is int
            and review["schema_version"] == 1 and review.get("disposition") == "accepted"
            and review.get("scope") == scope, f"{context}: wrong/unaccepted review scope")


def _review_links(review, review_path):
    require(isinstance(review.get("artifacts"), dict), "review missing artifacts")
    root = Path(review_path).parent.resolve()
    links = {}
    for relative, sha in review["artifacts"].items():
        require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
                and ".." not in relative.split("/") and "." not in relative.split("/")
                and "\\" not in relative, "unsafe relative review artifact link")
        resolved = (root / relative).resolve()
        require(resolved != root and root in resolved.parents, "review link escapes parent")
        require(str(resolved) not in links, "duplicate resolved review artifact")
        links[str(resolved)] = digest(sha, relative)
    return links


def _record_binding(summary, kind, name, current):
    require(isinstance(summary.get("input_bindings"), list), "summary missing input bindings")
    found = [b for b in summary["input_bindings"]
             if isinstance(b, dict) and b.get("kind") == kind and b.get("name") == name]
    require(len(found) == 1, f"summary binding role mismatch: {kind}/{name}")
    item = found[0]
    require(digest(item.get("sha256"), name) == current["sha256"]
            and type(item.get("size")) is int and item["size"] == current["size"]
            and isinstance(item.get("resolved_path"), str)
            and str(Path(item["resolved_path"]).resolve()) == current["resolved_path"],
            f"summary binding content/identity mismatch: {kind}/{name}")
    return item


def _upstream(prefix, sources, bindings):
    review = strict_json(sources[f"{prefix}_review"])
    complete = strict_json(sources[f"{prefix}_complete"])
    summary = strict_json(sources[f"{prefix}_summary"])
    scope = ("feature-bank binding only" if prefix == "stage0"
             else "historical-annotation target keyed binding only")
    _review(review, ("Stage0 feature-bank binding only" if prefix == "stage0"
                     else "Stage1a historical-annotation target keyed binding only"), prefix)
    require(isinstance(complete, dict) and type(complete.get("schema_version")) is int
            and complete["schema_version"] == 1 and complete.get("scope") == scope,
            f"{prefix}: COMPLETE scope/version mismatch")
    require(isinstance(summary, dict) and type(summary.get("schema_version")) is int
            and summary["schema_version"] == 1
            and isinstance(summary.get("scope"), str) and summary["scope"].startswith(scope)
            and text(summary.get("run_id"), f"{prefix} run") == complete.get("run_id")
            and summary.get("declarations", {}).get("run_id") == complete.get("run_id"),
            f"{prefix}: summary/COMPLETE run or scope mismatch")
    names = ({"inventory.csv": "stage0_inventory", "summary.json": "stage0_summary"}
             if prefix == "stage0" else {
                 "eligible_tiles.csv": "stage1a_eligible",
                 "feature_inventory.csv": "stage1a_feature_inventory",
                 "summary.json": "stage1a_summary"})
    output_names = (set(names) if prefix == "stage0" else
                    set(names) | {"quarantine_tiles.csv", "slide_inventory.csv", "diagnostics.json"})
    keys(complete.get("outputs"), output_names, f"{prefix} outputs")
    for filename, entry in complete["outputs"].items():
        keys(entry, ("sha256", "size"), f"{prefix}/{filename}")
        digest(entry["sha256"], filename)
        integer(entry["size"], filename)
    links = _review_links(review, bindings[f"{prefix}_review"]["resolved_path"])
    comp = bindings[f"{prefix}_complete"]
    require(Path(comp["resolved_path"]).name == "COMPLETE.json"
            and links.get(comp["resolved_path"]) == comp["sha256"],
            f"{prefix}: selected COMPLETE not linked")
    for filename, entry in complete["outputs"].items():
        linked = str((Path(comp["resolved_path"]).parent / filename).resolve())
        require(links.get(linked) == entry["sha256"].lower(),
                f"{prefix}: unconsumed/consumed COMPLETE output not linked by accepted review")
    for filename, name in names.items():
        binding = bindings[name]
        require(Path(binding["resolved_path"]) == Path(comp["resolved_path"]).parent / filename
                and links.get(binding["resolved_path"]) == binding["sha256"]
                and complete["outputs"][filename] == {
                    "sha256": binding["sha256"], "size": binding["size"]},
                f"{prefix}: consumed artifact links disagree: {filename}")
    require(isinstance(review.get("source_sha256"), dict)
            and digest(review["source_sha256"].get("metadata"), "upstream metadata")
            == bindings["metadata"]["sha256"], f"{prefix}: review metadata mismatch")
    _record_binding(summary, "metadata", "metadata", bindings["metadata"])
    specs = [b for b in summary["input_bindings"]
             if b.get("kind") == "spec" and b.get("name") == "spec"]
    require(len(specs) == 1 and isinstance(specs[0].get("resolved_path"), str)
            and links.get(str(Path(specs[0]["resolved_path"]).resolve()))
            == digest(specs[0].get("sha256"), "upstream spec"),
            f"{prefix}: original spec binding missing from accepted review")
    declaration = summary["declarations"]
    require(declaration.get("metadata", {}).get("id_col") == METADATA_COLUMNS["id_col"]
            and declaration["metadata"].get("patient_col") == METADATA_COLUMNS["patient_col"]
            and declaration["metadata"].get("label_col") == METADATA_COLUMNS["label_col"],
            f"{prefix}: metadata selectors differ")
    original_path = text(specs[0].get("path"), f"{prefix}: original spec lexical path")
    require("\x00" not in original_path and Path(original_path).is_absolute(),
            f"{prefix}: original spec lexical path must be usable and absolute")
    require(Path(original_path).resolve() == Path(specs[0]["resolved_path"]).resolve(),
            f"{prefix}: original spec lexical/resolved identity mismatch")
    original_base = Path(original_path).parent
    declared_meta = declaration["metadata"].get("path")
    text(declared_meta, "historical metadata path")
    require(str((original_base / declared_meta).resolve()) == bindings["metadata"]["resolved_path"],
            f"{prefix}: original declaration metadata identity mismatch")
    if prefix == "stage1a":
        require(digest(review["source_sha256"].get("stage0_independent_review"), "stage0 review")
                == bindings["stage0_review"]["sha256"], "Stage1a review has different Stage0")
        _record_binding(summary, "stage0", "stage0_review", bindings["stage0_review"])
        declared_review = declaration.get("stage0", {}).get("review")
        text(declared_review, "historical Stage0 review path")
        require(str((original_base / declared_review).resolve())
                == bindings["stage0_review"]["resolved_path"],
                "Stage1a original-spec-relative Stage0 review identity mismatch")
    return review, summary


def _controls(spec, sources, source_bindings, code_bindings):
    review, summary = _upstream("stage0", sources, source_bindings)
    require(digest(review["source_sha256"].get("auditor_module"), "Stage0 helper")
            == code_bindings["legacy_io"]["sha256"], "actual legacy helper not accepted Stage0 code")
    _record_binding(summary, "code", "running_auditor", code_bindings["legacy_io"])
    kernel_review = strict_json(sources["kernel_review"])
    _review(kernel_review, KERNEL_SCOPES[spec["task"]], "kernel")
    files = kernel_review.get("source_files")
    require(isinstance(files, dict), "kernel source_files missing")
    kernel_name = "prosper/evaluation/" + Path(code_bindings["kernel"]["resolved_path"]).name
    require(kernel_name in files and digest(files[kernel_name], "kernel module")
            == code_bindings["kernel"]["sha256"], "actual kernel differs from accepted review")
    for name, sha in files.items():
        text(name, "kernel source name")
        digest(sha, name)
    if spec["task"] == "hoptimus_filter_manifest":
        _upstream("stage1a", sources, source_bindings)
    require(summary.get("canonical_slides") == spec["expected"]["raw_slides"],
            "Stage0 raw population count differs")


def _read_written(fd, name, size):
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    with os.fdopen(descriptor, "rb") as handle:
        observed = os.fstat(handle.fileno())
        require(stat.S_ISREG(observed.st_mode) and observed.st_size == size,
                f"written {name}: not regular or wrong size")
        data = handle.read(size + 1)
        require(len(data) == size, f"written {name}: truncated/trailing bytes")
        return data


def _check_outputs(fd, out, target, out_root, inputs, hashes, *, marker=False):
    legacy._claimed_path(fd, target)
    require(legacy._output_path(out, out_root, inputs, []) == target, "output path retargeted")
    allowed = set(hashes) | ({"COMPLETE.json"} if marker else set())
    require(set(os.listdir(fd)) == allowed, "output allowlist mismatch/abandoned part files")
    for name, expected in hashes.items():
        data = _read_written(fd, name, expected["size"])
        require(_sha(data) == expected["sha256"], f"written {name}: final output hash changed")


def run_bound(
    spec_path, spec_sha256, out, out_root, *, task: str,
    code_paths: Mapping[str, str | Path],
    execute: Callable[[dict, Mapping[str, bytes]], Mapping[str, bytes]],
) -> dict:
    """Execute exactly one bound branch; return only a read-back COMPLETE marker."""
    expected_spec_sha = digest(spec_sha256, "spec SHA256")
    started = legacy._time()
    spec_path = Path(spec_path).absolute()
    spec_data, spec_binding = _scan_file(spec_path, "spec", "spec", SPEC_LIMIT)
    require(spec_binding["sha256"] == expected_spec_sha, "spec SHA256 mismatch")
    spec = strict_json(spec_data)
    _schema(spec, task)
    require(set(code_paths) == CODE_NAMES, "executing code_paths must cover exactly six names")
    require(isinstance(out_root, (str, Path)) and str(out_root).strip(), "explicit out-root required")
    root = Path(out_root).absolute()
    environment_root = os.environ.get("PROSPER_OUT")
    require(environment_root and Path(environment_root).resolve(strict=True) == root.resolve(strict=True),
            "PROSPER_OUT and out-root must identify the same existing root")
    base = spec_path.parent
    sources, source_bindings, code_bindings = {}, {}, {}
    bindings = [spec_binding]
    for name in sorted(spec["sources"]):
        entry = spec["sources"][name]
        data, binding = _scan_file(base / entry["path"], "source", name,
                                   SOURCE_LIMIT, expected=entry)
        sources[name], source_bindings[name] = data, binding
        bindings.append(binding)
    executing = {}
    for name in sorted(spec["code_files"]):
        entry = spec["code_files"][name]
        path = base / entry["path"]
        require(path.resolve(strict=True) == Path(code_paths[name]).resolve(strict=True),
                f"{name}: declaration differs from actual executing code path")
        _, binding = _scan_file(path, "code", name, SOURCE_LIMIT, expected=entry, retain=False)
        code_bindings[name] = binding
        bindings.append(binding)
        executing[name] = str(Path(code_paths[name]).resolve(strict=True))
    _controls(spec, sources, source_bindings, code_bindings)
    inputs = [Path(b["path"]) for b in bindings]
    out = Path(out).absolute()
    target = legacy._output_path(out, root, inputs, [])
    fd = legacy._claim_output(out, target)
    owned = False
    try:
        legacy._claimed_path(fd, target)
        owned = True
        declaration_bytes = json_bytes(spec)
        artifacts = execute(spec, MappingProxyType(sources))
        require(json_bytes(spec) == declaration_bytes, "callback changed validated declaration")
        require(isinstance(artifacts, Mapping) and set(artifacts) == ARTIFACTS[task],
                "callback artifact allowlist mismatch")
        require(all(type(data) is bytes for data in artifacts.values()),
                "callback artifacts must be bytes")
        summary = strict_json(artifacts["summary.json"])
        keys(summary, ("schema_version", "run_id", "task", "counts",
                       "policy", "details", "limitations"), "summary")
        require(type(summary["schema_version"]) is int and summary["schema_version"] == 1
                and summary["run_id"] == spec["run_id"] and summary["task"] == task
                and summary["counts"] == spec["expected"], "summary identity/count mismatch")
        for name, value in summary["counts"].items():
            integer(value, f"summary count {name}")
        require(isinstance(summary["limitations"], list)
                and all(isinstance(x, str) for x in summary["limitations"]), "summary limitations invalid")
        binding_document = {
            "schema_version": 1, "run_id": spec["run_id"], "task": task,
            "spec_sha256": expected_spec_sha, "bindings": bindings,
            "executing_code_paths": executing,
            "environment": {
                "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                "hostname": socket.gethostname(),
                "native_threads": {k: os.environ.get(k) for k in THREAD_KEYS},
            },
            "started_at": started, "captured_at": legacy._time(), "limitations": LIMITATIONS,
        }
        retained = dict(artifacts)
        sources.clear()
        retained["input_bindings.json"] = json_bytes(binding_document)
        hashes = {name: {"sha256": _sha(data), "size": len(data)}
                  for name, data in retained.items()}
        for name, data in retained.items():
            legacy._atomic_bytes(fd, name, data)
            require(_read_written(fd, name, len(data)) == data, f"{name}: written bytes differ")
        require(strict_json(_read_written(fd, "input_bindings.json",
                                          hashes["input_bindings.json"]["size"])) == binding_document,
                "input binding document readback mismatch")
        for original in bindings:
            _, current = _scan_file(
                original["path"], original["kind"], original["name"],
                SPEC_LIMIT if original["kind"] == "spec" else SOURCE_LIMIT,
                expected={"size": original["size"], "sha256": original["sha256"]},
                retain=False)
            require(current == original, f"{original['name']}: final input/code/spec mutation")
        _check_outputs(fd, out, target, root, inputs, hashes)
        marker = {
            "schema_version": 1, "run_id": spec["run_id"], "task": task,
            "scope": SCOPES[task], "spec_sha256": expected_spec_sha,
            "outputs": hashes, "verified_at": legacy._time(), "limitations": LIMITATIONS,
        }
        marker_bytes = json_bytes(marker)
        legacy._atomic_bytes(fd, "COMPLETE.json", marker_bytes)
        observed = strict_json(_read_written(fd, "COMPLETE.json", len(marker_bytes)))
        keys(observed, ("schema_version", "run_id", "task", "scope", "spec_sha256",
                        "outputs", "verified_at", "limitations"), "COMPLETE")
        require(type(observed["schema_version"]) is int and observed == marker,
                "COMPLETE marker binding/readback mismatch")
        _check_outputs(fd, out, target, root, inputs, hashes, marker=True)
        return observed
    except BaseException as exc:
        if owned:
            try:
                os.unlink("COMPLETE.json", dir_fd=fd)
            except FileNotFoundError:
                pass
            try:
                legacy._atomic_bytes(fd, "FAILED.json", json_bytes({
                    "failed_at": legacy._time(), "type": type(exc).__name__, "error": str(exc)}))
            except Exception:
                pass
        raise
    finally:
        os.close(fd)
