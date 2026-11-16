"""Bounded stdlib encoded-input observations; never input/model authorization."""

from __future__ import annotations

import copy
import csv
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import types
import unicodedata
from typing import TypedDict


CLI_NAME = "r21_input_observation_cli_verified_v1"
OBSERVER_NAME = "r21_input_admission_verified_v1"
OBSERVER_SHA = "768e91414e43a86854c6e0088bdf91b2f10f1c1a3fc0a157c355b1bfa464071b"
OBSERVER_SIZE = 41176
OBSERVER_REVIEW_SHA = "ee965236c79c9dd14de337f22670396d5c4553c5343cb20ea0f880467ddfe5a8"
OBSERVER_CONTRACT_SHA = "a561f3b5b085335d14215fef4f70a517c4276ac86483593a7e873a76c4e1273d"
WSI_ROOT = "/PROSPER_WORKSPACE/data/comet_final_HPC"
CONTROLS = (
    "observer_review", "metadata", "targets", "cpu_review", "cpu_complete",
    "cpu_spec", "cpu_input_bindings", "stage1b_review", "stage1b_complete",
    "stage1b_source_inventory",
)
ASSETS = ("template", "model_config", "checkpoint", "selected_wsi")
STAGES = (
    "trusted_basis", "initial_inventory", "template", "config", "checkpoint_header",
    "checkpoint_full", "selected_wsi_full", "final_checks", "final_inventory",
    "serialization",
)
LIMITS = {
    "spec_bytes": 1048576, "source_bytes": 1048576,
    "each_control_bytes": 8388608, "aggregate_control_bytes": 33554432,
    "config_bytes": 65536, "config_nesting": 16, "png_bytes": 1048576,
    "safetensors_header_bytes": 1048576, "stream_block_bytes": 1048576,
    "each_large_asset_bytes": 6442450944, "selected_two_pass_bytes": 21474836480,
    "stdout_packet_bytes": 16777216, "issue_message_characters": 1024,
    "global_issue_records": 128, "secondary_error_records": 32, "stderr_bytes": 65536,
}
EXPECTED = {
    "raw_slides": 2921, "target_slides": 2778, "patients": 1876,
    "groups": 15, "cross_label_patients": 45, "tile_rows": 3835507,
}
JSON_KEYS = {
    "cpu_complete": "limitations outputs run_id schema_version scope spec_sha256 task verified_at",
    "cpu_input_bindings": "bindings captured_at environment executing_code_paths limitations run_id schema_version spec_sha256 started_at task",
    "cpu_review": "artifacts branches contract_sha256 disposition observed_model observed_reasoning_effort prior_exposure remaining_work requested_model requested_reasoning_effort review_agent_id review_boundaries review_date review_turn schema_version scope separate_lead_reconciliation_sha256",
    "cpu_spec": "code_files expected parameters run_id schema_version sources task",
    "observer_review": "bindings disposition finding_dispositions independent_evidence limits recorded_by review_date reviewer schema_version separate_lead_replay stage",
    "stage1b_complete": "code completed_at disposition input_inventory_sha256 outputs run_id schema_version scope spec upstream",
    "stage1b_review": "artifacts disposition execution independent_checks job_id lead_evidence_acknowledgment_turn limits observed_model observed_reasoning_effort reconciled requested_model requested_reasoning_effort review_agent_id review_date review_turn schema_version scope warning_disposition",
}
CSV_HEADERS = {
    "metadata": ["Tumor.group", "Slide.Scan.File", "Sample.ID", "UID.Subject"],
    "targets": ["population_index", "metadata_index", "slide_id", "patient_id",
                "diagnostic_group", "feature_rows", "feature_path", "resolved_path",
                "feature_size", "feature_sha256"],
    "stage1b_source_inventory": [
        "source_id", "kind", "bank", "universe", "slide_id", "path", "resolved_path",
        "expected_presence", "observed_presence", "expected_size_bytes",
        "observed_size_bytes", "expected_sha256", "observed_sha256", "digest_basis",
    ],
}
CPU_OUTPUTS = ("input_bindings.json", "queries.csv", "references.csv", "stimuli.json", "summary.json")
PANEL_OUTPUTS = (
    "panel_tiles.csv", "feature_rows.csv", "source_inventory.csv",
    "lineage_diagnostics.json", "summary.json", "uni2_main.npy", "uni2_annotation.npy",
    "virchow2_main_f16.npy", "virchow2_main_f32.npy", "virchow2_annotation_full.npy",
    "h_optimus_1_main.npy", "h_optimus_1_annotation.npy", "h_optimus_1_auxiliary.npy",
)
LIMITATIONS = [
    "stdlib_observations_not_decoded_pixels", "structure_and_schema_not_model_identity",
    "declared_byte_matches_not_historical_vendor_authentication",
    "inventory_metadata_not_readability_or_content", "prior_bindings_not_current_observations",
    "cooperative_checks_not_atomic_snapshot", "stdout_requires_reviewed_capture",
    "no_complete_executable_software_attestation", "no_input_or_model_use_authorization",
    "logical_io_not_measured_resource_feasibility", "no_population_narrowing_or_substitution",
]
FORBIDDEN = ("PIL", "safetensors", "numpy", "pandas", "torch", "timm", "openslide", "prosper")
PROBE_FIELDS = (
    "lexical_path", "lexical_type", "resolved_path", "resolved_type", "size_bytes",
    "probe_coherent", "status", "error_operation", "error_errno", "error_message",
)
BOOK_COMPARE = (
    "lexical_before", "target_before", "lexical_after", "target_after",
    "resolved_before", "resolved_after",
)


class RunResult(TypedDict):
    packet: dict | None
    stdout_bytes: bytes
    stderr_bytes: bytes
    exit_code: int


class _Fault(ValueError):
    def __init__(self, code: str, message: str, category: str = "declaration",
                 exit_code: int = 1, operation: str | None = None,
                 error_errno: int | None = None):
        super().__init__(message)
        self.code = code
        self.category = category
        self.exit_code = exit_code
        self.operation = operation
        self.error_errno = error_errno


def _need(condition: bool, code: str, message: str) -> None:
    if not condition:
        category = ("integrity" if code in ("digest_mismatch", "declared_size_mismatch",
                                           "stream_growth", "short_read") else
                    "custody" if code in ("opened_identity_changed", "open_path_changed",
                                         "close_path_changed", "region_changed") else "declaration")
        raise _Fault(code, message, category)


def _fields(value: object, names: object) -> dict:
    _need(type(value) is dict and set(value) == set(names), "record_fields", "Invalid closed record")
    return value


def _int(value: object, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    _need(type(value) is int and minimum <= value <= maximum, "integer_field", "Invalid integer field")
    return value


def _text(value: object, maximum: int = 4096) -> str:
    _need(type(value) is str and 0 < len(value) <= maximum
          and value == value.strip(), "text_field", "Invalid literal text")
    _need(not any(unicodedata.category(c) in ("Cc", "Cs") for c in value),
          "text_control", "Control character in literal text")
    return value


def _path(value: object) -> str:
    return _text(value)


def _absolute(path: str, parent: str) -> str:
    return path if path.startswith("/") else parent.rstrip("/") + "/" + path


def _sha(value: object) -> str:
    _need(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
          "digest_field", "Invalid SHA256 declaration")
    return value


def _truncate(text: str) -> tuple[str, bool]:
    return (text, False) if len(text) <= 1024 else (text[:1021] + "...", True)


def _json_bytes(encoded: bytes) -> dict:
    try:
        text = encoded.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise _Fault("json_utf8", str(exc)) from exc
    _need(not text.startswith("\ufeff"), "json_bom", "JSON BOM is prohibited")
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            _need(depth <= 16, "json_depth", "JSON nesting limit")
        elif char in "]}":
            depth -= 1

    def number(token: str, floating: bool = False):
        _need(len(token) <= 128, "json_number_limit", "JSON number token limit")
        result = float(token) if floating else int(token)
        _need(not floating or math.isfinite(result), "json_nonfinite", "Nonfinite JSON number")
        return result

    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "json_duplicate", "Duplicate JSON key")
            _need(len(key.encode("utf-8")) <= 4096, "json_key_limit", "JSON key limit")
            result[key] = value
        return result

    def constant(token):
        raise _Fault("json_nonfinite", "Nonfinite JSON constant: " + token)

    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant,
                            parse_int=number, parse_float=lambda t: number(t, True))
        _need(type(result) is dict, "json_root", "JSON root must be an object")
        pending = [result]
        while pending:
            item = pending.pop()
            if type(item) is dict:
                pending.extend(item.keys())
                pending.extend(item.values())
            elif type(item) is list:
                pending.extend(item)
            elif type(item) is str:
                _need(len(item.encode("utf-8")) <= 65536, "json_string_limit", "JSON string limit")
        return result
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, _Fault):
            raise
        raise _Fault("json_parse", str(exc)) from exc


def _decimal(text: str, minimum: int = 0) -> int:
    _need(type(text) is str and len(text) <= 19
          and re.fullmatch("0|[1-9][0-9]*", text) is not None,
          "csv_decimal", "Noncanonical decimal column")
    return _int(int(text), minimum)


def _csv_bytes(encoded: bytes, role: str) -> list[dict]:
    try:
        text = encoded.decode("utf-8", "strict")
        lines = text.splitlines(keepends=True)
        _need(bool(lines), "csv_empty", "Empty CSV")
        rows = []
        for index, line in enumerate(lines):
            _need(len(line.encode("utf-8")) <= 8448, "csv_record_limit", "CSV record limit")
            fields = next(csv.reader([line], strict=True))
            _need(not any("\r" in f or "\n" in f for f in fields),
                  "csv_multiline", "Multiline CSV field")
            _need(len(fields) == len(CSV_HEADERS[role]), "csv_width", "CSV row width")
            if index == 0:
                _need(fields == CSV_HEADERS[role], "csv_header", "CSV header mismatch")
            else:
                rows.append(dict(zip(CSV_HEADERS[role], fields)))
        _need(len(rows) == {"metadata": 2921, "targets": 2778,
                           "stage1b_source_inventory": 227}[role],
              "csv_count", "CSV population count")
        return rows
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise _Fault("csv_parse", str(exc)) from exc


def _check_environment() -> None:
    _need(bool(sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode
               and sys.dont_write_bytecode),
          "isolated_entry_required", "Require -I -S -B verified-byte entry")
    _need(not any(name == root or name.startswith(root + ".")
                  for name in sys.modules for root in FORBIDDEN),
          "forbidden_module", "Forbidden package/scientific module present")
    _need(all(hasattr(os, name) for name in ("O_NONBLOCK", "O_NOFOLLOW")),
          "posix_flags_required", "Required POSIX open flags unavailable")


def _binding(role: str, path: str, size: int | None, digest: str) -> dict:
    return {
        "role": role, "declared_path": path, "declared_size": size,
        "declared_sha256": digest, "initial_descriptor": None, "final_descriptor": None,
        "whole_file_checks": [_whole("initial"), _whole("final")],
        "custody_state": "unacquired",
    }


def _whole(phase: str) -> dict:
    return {"phase": phase, "state": "not_performed", "bytes_read": 0,
            "observed_sha256": None, "expected_match": None}


def _inventory(ids: list[str] | None = None) -> dict:
    return {"state": "not_started", "visited_ids": [], "in_progress_id": None,
            "unvisited_ids": copy.deepcopy(ids), "probes": {}, "bookends": {},
            "frozen_result": None, "omitted_fields": []}


def _comparison(ids: list[str] | None = None) -> dict:
    return {"state": "not_performed", "compared_ids": [], "changed_ids": [],
            "changes": [], "uncomparable_ids": copy.deepcopy(ids)}


def _packet() -> dict:
    return {
        "schema_version": 1, "task": "hoptimus_encoded_input_observation",
        "scope": "encoded_input_observations_and_selected_integrity_only",
        "run_id": None, "spec_sha256": None, "packet_kind": "partial_error",
        "exit_code": 1, "model_use_authorized": False,
        "entry_binding": None, "observer_binding": None, "control_bindings": [],
        "asset_records": [],
        "stages": [{"name": n, "state": "not_started", "result": None,
                    "blocker_codes": [], "issue_indices": []} for n in STAGES],
        "inventory_initial": _inventory(), "inventory_final": _inventory(),
        "inventory_comparison": _comparison(), "issues": [], "secondary_errors": [],
        "overflow": {"occurred": False, "cause_code": None, "omitted_sections": [],
                     "omitted_counts": {"inventory_initial_rows": 0, "inventory_final_rows": 0,
                                        "issues": 0, "secondary_errors": 0},
                     "serialization_bytes_seen": None, "global_issue_attempts": 0,
                     "secondary_error_attempts": 0},
        "logical_io": {
            "reads": [], "selected": {"requested_bytes": 0, "returned_bytes": 0,
                                     "eof_probe_requests": 0, "eof_probe_returned_bytes": 0,
                                     "limit_bytes": LIMITS["selected_two_pass_bytes"]},
            "controls": {"declared_bytes": None, "limit_bytes": LIMITS["aggregate_control_bytes"]},
        },
        "limitations": list(LIMITATIONS),
    }


def _fingerprint(info) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns]


def _file_type(info) -> str:
    if info is None:
        return "unknown"
    mode = info[2]
    return "regular" if stat.S_ISREG(mode) else (
        "symlink" if stat.S_ISLNK(mode) else "directory" if stat.S_ISDIR(mode) else "other")


def _recognized_loop(exc: Exception) -> bool:
    return (isinstance(exc, OSError) and exc.errno == errno.ELOOP or
            type(exc) is RuntimeError and isinstance(exc.__context__, OSError)
            and exc.__context__.errno == errno.ELOOP)


def _interruption(exc: BaseException | None) -> bool:
    return (isinstance(exc, BaseException) and not isinstance(exc, Exception)
            or isinstance(exc, InterruptedError)
            or isinstance(exc, OSError) and exc.errno == errno.EINTR)


def _custody(record: dict, state: str) -> None:
    if record["custody_state"] not in ("failed", "invalidated"):
        record["custody_state"] = state


def _coherent_bookends(before: dict, after: dict) -> bool:
    coherent = all(before[k] == after[k] for k in ("lexical", "target", "resolved"))
    coherent = coherent and [
        (e["operation"], e["error_type"], e["errno"]) for e in before["errors"]
    ] == [(e["operation"], e["error_type"], e["errno"]) for e in after["errors"]]
    if (after["lexical"] is not None and after["target"] is not None
            and _file_type(after["lexical"]) not in ("symlink", _file_type(after["target"]))):
        return False
    return coherent


def _bookend(path: str, which: str) -> dict:
    result = {"lexical": None, "target": None, "resolved": None, "errors": []}
    operation = "lstat"
    try:
        result["lexical"] = _fingerprint(os.lstat(path))
        operation = "resolve"
        result["resolved"] = str(Path(path).resolve(strict=True))
        operation = "stat"
        result["target"] = _fingerprint(os.stat(result["resolved"], follow_symlinks=False))
    except (OSError, RuntimeError, KeyboardInterrupt) as exc:
        message, truncated = _truncate(str(exc))
        error_record = {
            "bookend": which, "operation": operation, "error_type": type(exc).__name__,
            "errno": exc.errno if isinstance(exc, OSError) else None,
            "message": message, "message_truncated": truncated,
        }
        result["errors"].append(error_record)
        if (_interruption(exc) or not isinstance(exc, OSError)
                and (operation != "resolve" or not _recognized_loop(exc))):
            exc.observed_bookend = result
            exc.bookend_phase = which
            exc.operation = operation
            raise
    return result


def _probe_path(path: str) -> tuple[dict, dict, int]:
    before = None
    try:
        before = _bookend(path, "before")
        after = _bookend(path, "after")
    except BaseException as exc:
        observed = getattr(exc, "observed_bookend", None)
        phase = getattr(exc, "bookend_phase", None)
        if before is not None or observed is not None:
            first = before or (observed if phase == "before" else None)
            last = observed if phase == "after" else None
            exc.partial_bookends = {
                "lexical_before": first["lexical"] if first else None,
                "target_before": first["target"] if first else None,
                "lexical_after": last["lexical"] if last else None,
                "target_after": last["target"] if last else None,
                "resolved_before": first["resolved"] if first else None,
                "resolved_after": last["resolved"] if last else None,
                "errors": (first["errors"] if first else []) + (last["errors"] if last else []),
                "coherent": None,
            }
        raise
    errors = before["errors"] + after["errors"]
    coherent = _coherent_bookends(before, after)
    chosen = after
    lexical_type = _file_type(chosen["lexical"])
    error = errors[-1] if errors else None
    technical = any(e["errno"] not in (errno.ENOENT, errno.EACCES, errno.EPERM,
                                      errno.ENOTDIR, errno.ELOOP, None) for e in errors)
    if not coherent:
        status, exit_code = "changed_during_probe", 1
    elif error:
        number = error["errno"]
        if number in (errno.EACCES, errno.EPERM):
            status = "inaccessible"
        elif number == errno.ENOENT and error["operation"] == "lstat":
            status, lexical_type = "missing_lexical", "absent"
        elif number == errno.ENOENT and lexical_type == "symlink":
            status = "broken_symlink"
        elif number in (errno.ENOTDIR, errno.ELOOP) or error["error_type"] == "RuntimeError":
            status = "resolution_error"
        else:
            status = "io_error"
            technical = True
        exit_code = 1 if technical else 2
    elif _file_type(chosen["target"]) != "regular":
        status, exit_code = "not_regular", 2
    else:
        status, exit_code = "observed_regular", 0
    if status == "changed_during_probe":
        operation, number, message = "compare_bookends", None, "Observed path bookend change"
    else:
        operation = error["operation"] if error else None
        number = error["errno"] if error else None
        message = error["message"] if error else None
    probe = {
        "lexical_path": path, "lexical_type": lexical_type,
        "resolved_path": chosen["resolved"],
        "resolved_type": _file_type(chosen["target"]) if chosen["target"] else None,
        "size_bytes": chosen["target"][3] if chosen["target"] else None,
        "probe_coherent": coherent, "status": status,
        "error_operation": operation, "error_errno": number, "error_message": message,
    }
    if status in ("missing_lexical", "broken_symlink"):
        probe.update(resolved_path=None, resolved_type=None, size_bytes=None)
    bookends = {
        "lexical_before": before["lexical"], "target_before": before["target"],
        "lexical_after": after["lexical"], "target_after": after["target"],
        "resolved_before": before["resolved"], "resolved_after": after["resolved"],
        "errors": errors, "coherent": coherent,
    }
    return probe, bookends, max(exit_code, 1 if technical else 0)


class _Context:
    def __init__(self):
        self.packet = _packet()
        self.exit_code = 0
        self.large_blockers = []
        self.observer = None
        self.consumed = []
        self.regions = {}
        self.active_reader = None
        self.ids = None
        self.targets = []
        self.metadata = []
        self.prior = {}
        self.stage_name = "trusted_basis"
        self.interrupted = False

    def stage(self, name: str) -> dict:
        return self.packet["stages"][STAGES.index(name)]

    def issue(self, exc: BaseException, role: str | None = None, *,
              stage: str | None = None, secondary: int | None = None) -> int:
        stage = stage or self.stage_name
        if _interruption(exc):
            self.interrupted = True
        primitive = (self.observer is not None and hasattr(self.observer, "AdmissionError")
                     and isinstance(exc, self.observer.AdmissionError))
        if primitive:
            negative = exc.kind != "declaration"
            category, code, kind = "observation", exc.code, exc.kind
            outcome = 2 if negative else 1
        elif isinstance(exc, _Fault):
            category, code, kind = exc.category, exc.code, None
            outcome = exc.exit_code
        elif isinstance(exc, OSError):
            category, code, kind, outcome = "unexpected_io", "os_error", None, 1
        else:
            category, code, kind, outcome = "execution", "unexpected_exception", None, 1
        message, truncated = _truncate(str(exc))
        record = {
            "stage": stage, "role": role, "category": category, "kind": kind, "code": code,
            "region": ("header" if role == "checkpoint" else "encoded") if primitive else None,
            "offset": exc.offset if primitive else None,
            "operation": getattr(exc, "operation", None),
            "errno": getattr(exc, "error_errno", getattr(exc, "errno", None)),
            "message": message, "message_truncated": truncated,
        }
        self.exit_code = max(self.exit_code, outcome) if self.exit_code != 1 else 1
        if outcome == 1:
            self.exit_code = 1
        if secondary is None and (stage not in ("initial_inventory", "final_inventory") or outcome == 1):
            if code not in self.large_blockers:
                self.large_blockers.append(code)
        overflow = self.packet["overflow"]
        key = "global_issue_attempts" if secondary is None else "secondary_error_attempts"
        overflow[key] += 1
        records = self.packet["issues"] if secondary is None else self.packet["secondary_errors"]
        maximum = 127 if secondary is None else 31
        if secondary is not None:
            record["primary_issue_index"] = secondary
        if len(records) >= maximum:
            self.exit_code = 1
            code = "issue_limit" if secondary is None else "secondary_limit"
            if code not in self.large_blockers:
                self.large_blockers.append(code)
            overflow.update(occurred=True, cause_code=code)
            omitted_key = "issues" if secondary is None else "secondary_errors"
            overflow["omitted_counts"][omitted_key] += 1
            if len(records) == maximum:
                record.update(category="execution", kind=None, code=code,
                              message="Bounded issue evidence overflow", message_truncated=False)
                records.append(record)
            self.stage(stage)["state"] = "failed"
            return len(self.packet["issues"]) - 1
        records.append(record)
        index = len(self.packet["issues"]) - 1
        if secondary is None:
            item = self.stage(stage)
            item["issue_indices"].append(index)
            if code not in item["blocker_codes"]:
                item["blocker_codes"].append(code)
            item["state"] = "failed" if outcome == 1 else "negative"
        return index

    def read_record(self, role: str, phase: str) -> dict:
        record = {"role": role, "phase": phase, "requested_bytes": 0, "returned_bytes": 0,
                  "eof_probe_requests": 0, "eof_probe_returned_bytes": 0}
        self.packet["logical_io"]["reads"].append(record)
        return record

    def read(self, fd: int, amount: int, record: dict, eof: bool = False) -> bytes:
        _need(not self.interrupted, "interrupted", "No payload reads after interruption")
        selected = self.packet["logical_io"]["selected"]
        if record["role"] in ASSETS:
            _need(selected["returned_bytes"] + amount <= selected["limit_bytes"],
                  "selected_read_budget", "Selected actual-read budget exceeded")
        record["requested_bytes"] += amount
        if eof:
            record["eof_probe_requests"] += amount
        if record["role"] in ASSETS:
            selected["requested_bytes"] += amount
            if eof:
                selected["eof_probe_requests"] += amount
        block = os.read(fd, amount)
        record["returned_bytes"] += len(block)
        if eof:
            record["eof_probe_returned_bytes"] += len(block)
        if record["role"] in ASSETS:
            selected["returned_bytes"] += len(block)
            if eof:
                selected["eof_probe_returned_bytes"] += len(block)
        return block


class _Reader:
    def __init__(self, context: _Context, record: dict, phase: str, ceiling: int):
        self.context, self.record, self.phase = context, record, phase
        self.fd = None
        self.size = 0
        self.position = 0
        self.io = context.read_record(record["role"], phase)
        self.descriptor = {
            "resolved_path": None, "lexical_before": None, "target_before": None,
            "opened": None, "after_fd": None, "lexical_after": None, "target_after": None,
            "coherent": None,
        }
        record[phase + "_descriptor"] = self.descriptor
        try:
            before = _bookend(record["declared_path"], "before")
        except BaseException as exc:
            observed = getattr(exc, "observed_bookend", None)
            if observed is not None:
                self.descriptor.update(resolved_path=observed["resolved"],
                                       lexical_before=observed["lexical"], target_before=observed["target"])
            raise
        self.descriptor.update(resolved_path=before["resolved"], lexical_before=before["lexical"],
                               target_before=before["target"])
        if "current_resolved_path" in record and before["resolved"] is not None:
            record["current_resolved_path"] = before["resolved"]
        if before["errors"] or _file_type(before["target"]) != "regular":
            try:
                after = _bookend(record["declared_path"], "after")
            except BaseException as exc:
                observed = getattr(exc, "observed_bookend", None)
                if observed is not None:
                    self.descriptor.update(lexical_after=observed["lexical"],
                                           target_after=observed["target"])
                raise
            coherent = _coherent_bookends(before, after)
            self.descriptor.update(lexical_after=after["lexical"], target_after=after["target"],
                                   coherent=coherent)
            if "current_resolved_path" in record and after["resolved"] is not None:
                record["current_resolved_path"] = after["resolved"]
            if not coherent:
                _custody(record, "invalidated")
                errors = after["errors"] or before["errors"]
                error = errors[-1] if errors else None
                raise _Fault("negative_metadata_changed", "Negative acquisition bookends disagree",
                             "custody", operation=error["operation"] if error else None,
                             error_errno=error["errno"] if error else None)
            if before["errors"]:
                error = before["errors"][0]
                ordinary = error["errno"] in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP,
                                             errno.EACCES, errno.EPERM) or error["error_type"] == "RuntimeError"
                raise _Fault("asset_unavailable", error["message"],
                             "availability" if ordinary and record["role"] in ASSETS else "unexpected_io",
                             2 if ordinary and record["role"] in ASSETS else 1,
                             error["operation"], error["errno"])
            raise _Fault("not_regular", "Acquisition target is not regular",
                         "availability" if record["role"] in ASSETS else "declaration",
                         2 if record["role"] in ASSETS else 1)
        self.size = before["target"][3]
        _need(self.size <= ceiling, "acquisition_size_limit", "Acquisition exceeds role ceiling")
        declared = record["declared_size"]
        _need(declared is None or self.size == declared, "declared_size_mismatch", "Declared size mismatch")
        try:
            self.fd = os.open(before["resolved"], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            opened = _fingerprint(os.fstat(self.fd))
            self.descriptor["opened"] = opened
            _need(_file_type(opened) == "regular" and opened == before["target"],
                  "opened_identity_changed", "Opened descriptor does not match regular target")
            current = _bookend(record["declared_path"], "after")
            _need(not current["errors"] and all(current[k] == before[k]
                                              for k in ("lexical", "target", "resolved")),
                  "open_path_changed", "Path changed during open")
        except BaseException as original_error:
            if self.fd is not None:
                _custody(record, "failed")
                _cleanup_reader(self, original_error)
            elif (isinstance(original_error, OSError)
                  and original_error.errno in (errno.EACCES, errno.EPERM)
                  and record["role"] in ASSETS):
                after = _bookend(record["declared_path"], "after")
                unchanged = not after["errors"] and all(after[k] == before[k]
                                                       for k in ("lexical", "target", "resolved"))
                self.descriptor.update(lexical_after=after["lexical"], target_after=after["target"],
                                       coherent=unchanged)
                _need(unchanged, "open_path_changed", "Path changed after failed open")
                raise _Fault("asset_inaccessible", str(original_error), "availability", 2,
                             "open", original_error.errno) from original_error
            raise
        _custody(record, "region_only")
        if "current_resolved_path" in record:
            record["current_resolved_path"] = before["resolved"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            invalidated = isinstance(exc, _Fault) and exc.category in ("integrity", "custody")
            _custody(self.record, "invalidated" if invalidated else "failed")
        _cleanup_reader(self, exc)
        return False

    def exact(self, length: int) -> bytes:
        blocks = []
        remaining = length
        while remaining:
            block = self.context.read(self.fd, min(remaining, LIMITS["stream_block_bytes"]), self.io)
            _need(bool(block), "short_read", "Premature EOF")
            blocks.append(block)
            remaining -= len(block)
            self.position += len(block)
        return b"".join(blocks)

    def full(self, prefix: bytes = b"", capture: bool = False) -> bytes | None:
        _need(self.position == len(prefix), "stream_position", "Captured prefix/descriptor mismatch")
        digest = hashlib.sha256(prefix)
        blocks = [prefix] if capture else None
        check = self.record["whole_file_checks"][0 if self.phase == "initial" else 1]
        check["bytes_read"] = self.position
        try:
            while self.position < self.size:
                block = self.context.read(self.fd, min(self.size - self.position,
                                                       LIMITS["stream_block_bytes"]), self.io)
                _need(bool(block), "short_read", "Premature EOF")
                self.position += len(block)
                check["bytes_read"] = self.position
                digest.update(block)
                if blocks is not None:
                    blocks.append(block)
            growth = self.context.read(self.fd, 1, self.io, True)
            _need(not growth, "stream_growth", "Extra byte after declared EOF")
            check["observed_sha256"] = digest.hexdigest()
            check["expected_match"] = digest.hexdigest() == self.record["declared_sha256"]
            check["state"] = "matched" if check["expected_match"] else "rejected"
            _need(check["expected_match"], "digest_mismatch", "Whole-file digest mismatch")
            return b"".join(blocks) if blocks is not None else None
        except BaseException:
            if check["observed_sha256"] is None:
                check["state"] = "failed"
            raise

    def finish(self, *, primary: BaseException | None = None) -> None:
        if self.fd is None:
            return
        fd = self.fd
        validation_error = close_error = None
        if _interruption(primary):
            self.context.interrupted = True
        if self.context.interrupted:
            _custody(self.record, "failed")
        try:
            if primary is None and not self.context.interrupted:
                self.descriptor["after_fd"] = _fingerprint(os.fstat(fd))
                try:
                    after = _bookend(self.record["declared_path"], "after")
                except BaseException as exc:
                    observed = getattr(exc, "observed_bookend", None)
                    if type(observed) is dict and getattr(exc, "bookend_phase", None) == "after":
                        self.descriptor.update(lexical_after=observed["lexical"],
                                               target_after=observed["target"])
                    raise
                self.descriptor.update(lexical_after=after["lexical"], target_after=after["target"])
                coherent = (not after["errors"] and after["resolved"] == self.descriptor["resolved_path"]
                            and self.descriptor["opened"] == self.descriptor["after_fd"]
                            == after["target"] == self.descriptor["target_before"]
                            and after["lexical"] == self.descriptor["lexical_before"])
                self.descriptor["coherent"] = coherent
                _need(coherent, "close_path_changed", "Descriptor/path changed during acquisition")
        except BaseException as exc:
            validation_error = exc
            _custody(self.record, "invalidated")
            if _interruption(exc):
                self.context.interrupted = True
        finally:
            try:
                os.close(fd)
            except BaseException as exc:
                close_error = exc
                _custody(self.record, "failed")
                if _interruption(exc):
                    self.context.interrupted = True
            finally:
                # Ownership ends after the one close attempt; failure is not proof of closure.
                self.fd = None
        if validation_error is not None:
            if close_error is not None:
                index = self.context.issue(validation_error, self.record["role"])
                self.context.issue(close_error, self.record["role"], secondary=index)
            raise validation_error
        if close_error is not None:
            raise close_error


def _cleanup_reader(reader: _Reader, primary: BaseException | None = None) -> None:
    try:
        reader.finish(primary=primary)
    except BaseException as cleanup:
        if primary is None:
            raise
        index = reader.context.issue(primary, reader.record["role"])
        reader.context.issue(cleanup, reader.record["role"], secondary=index)


def _small(context: _Context, record: dict, ceiling: int, phase: str = "initial") -> bytes:
    with _Reader(context, record, phase, ceiling) as reader:
        if phase == "initial":
            context.consumed.append((record, ceiling))
        encoded = reader.full(capture=True)
    if (phase == "final" and record["whole_file_checks"][0]["expected_match"] is True
            and record["initial_descriptor"]["coherent"] is True
            and record["custody_state"] not in ("failed", "invalidated")):
        record["custody_state"] = "full_verified"
    return encoded


def _declaration(spec: dict, parent: str) -> dict:
    _fields(spec, ("schema_version", "run_id", "task", "observer", "controls",
                   "assets", "lexical_root", "expected", "selection", "limits"))
    _need(_int(spec["schema_version"]) == 1 and spec["task"] == "hoptimus_encoded_input_observation",
          "spec_task", "Spec task/schema mismatch")
    _need(type(spec["run_id"]) is str and re.fullmatch("[A-Za-z0-9_.-]{1,128}", spec["run_id"]),
          "run_id", "Invalid run identity")
    for key, expected in (("limits", LIMITS), ("expected", EXPECTED)):
        _fields(spec[key], expected)
        _need(all(_int(spec[key][name]) == value for name, value in expected.items()),
              "fixed_values", "Fixed declarations do not match")
    _need(spec["lexical_root"] == WSI_ROOT, "lexical_root", "WSI lexical root mismatch")
    _need(spec["selection"] == {"selected_wsi_ids": ["10003"], "deferred_wsi_ids": ["16683"]},
          "selection", "Selected/deferred identities mismatch")
    _fields(spec["controls"], CONTROLS)
    _fields(spec["assets"], ASSETS)
    for role, declaration in [("observer", spec["observer"]), *spec["controls"].items(),
                              *spec["assets"].items()]:
        names = ("path", "size", "sha256", "slide_id") if role == "selected_wsi" else (
            "path", "size", "sha256")
        _fields(declaration, names)
        _path(declaration["path"])
        declaration["path"] = _absolute(declaration["path"], parent)
        _path(declaration["path"])
        _int(declaration["size"], 1)
        _sha(declaration["sha256"])
        ceiling = (LIMITS["source_bytes"] if role == "observer" else
                   LIMITS["each_control_bytes"] if role in CONTROLS else
                   LIMITS["png_bytes"] if role == "template" else
                   LIMITS["config_bytes"] if role == "model_config" else
                   LIMITS["each_large_asset_bytes"])
        _need(declaration["size"] <= ceiling, "declaration_size_limit", "Declared size ceiling")
    _need(spec["observer"]["size"] == OBSERVER_SIZE and spec["observer"]["sha256"] == OBSERVER_SHA,
          "observer_binding", "Frozen observer binding mismatch")
    _need(spec["controls"]["observer_review"]["sha256"] == OBSERVER_REVIEW_SHA,
          "observer_receipt", "Frozen observer receipt mismatch")
    _need(spec["assets"]["selected_wsi"]["slide_id"] == "10003"
          and spec["assets"]["selected_wsi"]["path"] == WSI_ROOT + "/10003.svs",
          "selected_path", "Selected WSI must use literal root join")
    _need(sum(r["size"] for r in spec["controls"].values()) <= LIMITS["aggregate_control_bytes"],
          "control_declaration_limit", "Aggregate declared control size")
    _need(2 * sum(r["size"] for r in spec["assets"].values()) + 8 <= LIMITS["selected_two_pass_bytes"],
          "selected_declaration_limit", "Selected planned read budget")
    return spec


def _output_registry(value: dict, names: tuple, size_key: str) -> None:
    _fields(value, names)
    for record in value.values():
        _fields(record, (size_key, "sha256"))
        _int(record[size_key], 1)
        _sha(record["sha256"])


def _same_descriptor(record: dict, declaration: dict, size_key: str = "size") -> bool:
    return record[size_key] == declaration["size"] and record["sha256"] == declaration["sha256"]


def _population(metadata: list[dict], queries: list[dict]) -> tuple[list[dict], list[dict]]:
    meta, targets, seen = [], [], set()
    for index, row in enumerate(metadata):
        sid = str(_decimal(row["Slide.Scan.File"], 1))
        _need(sid not in seen, "metadata_duplicate", "Duplicate metadata slide")
        seen.add(sid)
        meta.append({"metadata_index": index, "slide_id": sid,
                     "patient_id": _text(row["UID.Subject"]),
                     "diagnostic_group": _text(row["Tumor.group"])})
    eligible = [r for r in meta if r["diagnostic_group"] != "MISC"]
    _need(len(eligible) == 2778, "population_exclusion", "MISC-only target exclusion")
    for index, row in enumerate(queries):
        projected = {
            "population_index": _decimal(row["population_index"]),
            "metadata_index": _decimal(row["metadata_index"]),
            "slide_id": str(_decimal(row["slide_id"], 1)),
            "patient_id": _text(row["patient_id"]), "diagnostic_group": _text(row["diagnostic_group"]),
            "expected_tile_rows": _decimal(row["feature_rows"], 1),
        }
        _need(projected["population_index"] == index
              and all(projected[k] == eligible[index][k] for k in eligible[index]),
              "population_join", "Target/metadata order or identity mismatch")
        _decimal(row["feature_size"], 1)
        _sha(row["feature_sha256"])
        _path(row["feature_path"])
        _path(row["resolved_path"])
        targets.append(projected)
    groups = {}
    for row in targets:
        groups.setdefault(row["patient_id"], set()).add(row["diagnostic_group"])
    _need(len(groups) == 1876 and len({r["diagnostic_group"] for r in targets}) == 15
          and sum(len(g) > 1 for g in groups.values()) == 45
          and sum(r["expected_tile_rows"] for r in targets) == 3835507,
          "population_totals", "Target counts/support mismatch")
    return meta, targets


def _reconcile(context: _Context, spec: dict, controls: dict) -> None:
    observer_review = controls["observer_review"]
    _need(observer_review["disposition"] == "accepted_repaired_candidate002_two_files_only"
          and observer_review["stage"] == "encoded_input_observation_primitives_implementation"
          and observer_review["bindings"]["contract_sha256"] == OBSERVER_CONTRACT_SHA
          and _same_descriptor(observer_review["bindings"]["module"], spec["observer"]),
          "observer_review_link", "Observer accepted receipt does not bind source")
    cpu, cpu_review = controls["cpu_complete"], controls["cpu_review"]
    producer, bindings = controls["cpu_spec"], controls["cpu_input_bindings"]
    _need(cpu_review["disposition"] == "accepted"
          and cpu_review["branches"]["perturb"]["disposition"] == "accepted"
          and cpu_review["branches"]["perturb"]["scope"] == "Population and stimulus-manifest preparation only"
          and cpu_review["artifacts"]["perturb_attempt_001/COMPLETE.json"]
          == spec["controls"]["cpu_complete"]["sha256"], "cpu_review_link", "CPU acceptance link")
    _need(_int(cpu["schema_version"]) == 1 and cpu["task"] == "hoptimus_perturb_manifest"
          and cpu["scope"] == "hoptimus_perturb_stimulus_manifests_only",
          "cpu_complete_scope", "CPU COMPLETE scope")
    _output_registry(cpu["outputs"], CPU_OUTPUTS, "size")
    _need(cpu["outputs"]["queries.csv"] == cpu["outputs"]["references.csv"]
          and _same_descriptor(cpu["outputs"]["queries.csv"], spec["controls"]["targets"])
          and _same_descriptor(cpu["outputs"]["input_bindings.json"], spec["controls"]["cpu_input_bindings"])
          and cpu["spec_sha256"] == spec["controls"]["cpu_spec"]["sha256"],
          "cpu_output_links", "CPU output/spec bindings")
    for value in (producer, bindings):
        _need(_int(value["schema_version"]) == 1 and value["task"] == cpu["task"]
              and value["run_id"] == cpu["run_id"], "cpu_producer_identity", "CPU producer identity")
    _need(bindings["spec_sha256"] == cpu["spec_sha256"], "cpu_bindings_spec", "CPU bindings spec hash")
    producer_expected = {"raw_slides": 2921, "target_slides": 2778, "patients": 1876,
                         "groups": 15, "cross_label_patients": 45, "feature_dim": 1536,
                         "stimulus_records": 33336}
    _fields(producer["expected"], producer_expected)
    _need(all(_int(producer["expected"][k]) == v for k, v in producer_expected.items()),
          "producer_expected", "Producer population declaration mismatch")
    _need(producer["parameters"]["metadata"] == {
        "id_col": "Slide.Scan.File", "label_col": "Tumor.group", "patient_col": "UID.Subject"},
        "producer_columns", "Producer metadata column declaration")
    mapped = {}
    _need(type(bindings["bindings"]) is list, "cpu_bindings_type", "CPU bindings list")
    for row in bindings["bindings"]:
        _fields(row, ("device", "inode", "kind", "name", "path", "resolved_path", "sha256", "size"))
        key = (_text(row["kind"]), _text(row["name"]))
        _need(key not in mapped, "cpu_binding_duplicate", "Duplicate source binding identity")
        _int(row["device"])
        _int(row["inode"])
        _int(row["size"], 1)
        _sha(row["sha256"])
        _path(row["path"])
        _path(row["resolved_path"])
        mapped[key] = row
    _need(_same_descriptor(mapped[("spec", "spec")], spec["controls"]["cpu_spec"]),
          "producer_spec_binding", "Producer spec byte binding")
    for name, declaration in (("metadata", spec["controls"]["metadata"]),
                              ("template", spec["assets"]["template"])):
        original = producer["sources"][name]
        row = mapped[("source", name)]
        _fields(original, ("path", "size", "sha256"))
        _need(original["path"] == row["path"] and _same_descriptor(row, original)
              and _same_descriptor(row, declaration), "producer_source_link", "Metadata/template source binding")
    context.metadata, context.targets = _population(controls["metadata"], controls["targets"])
    ids = [r["slide_id"] for r in context.targets]
    panel, review = controls["stage1b_complete"], controls["stage1b_review"]
    _need(review["disposition"] == "accepted"
          and review["scope"] == "Stage1b sampled panel/reference and current input-byte preparation only"
          and review["artifacts"]["stage1b_attempt_001/COMPLETE.json"]
          == spec["controls"]["stage1b_complete"]["sha256"],
          "panel_review_link", "Stage1b review link")
    _need(_int(panel["schema_version"]) == 1 and panel["disposition"] == "prepared"
          and panel["scope"] == "encoder_replay_panel_only", "panel_scope", "Stage1b COMPLETE scope")
    _output_registry(panel["outputs"], PANEL_OUTPUTS, "size_bytes")
    inv_decl = spec["controls"]["stage1b_source_inventory"]
    _need(_same_descriptor(panel["outputs"]["source_inventory.csv"], inv_decl, "size_bytes")
          and panel["input_inventory_sha256"] == inv_decl["sha256"]
          and review["artifacts"]["stage1b_attempt_001/source_inventory.csv"] == inv_decl["sha256"],
          "panel_inventory_link", "Stage1b source inventory bindings")
    rows = {}
    for row in controls["stage1b_source_inventory"]:
        source_id = _text(row["source_id"])
        _need(source_id not in rows, "source_id_duplicate", "Duplicate source identity")
        rows[source_id] = row
    wsi_rows = [r for r in rows.values() if r["kind"] == "wsi" and r["slide_id"] in ids]
    _need(len(wsi_rows) == 10, "prior_wsi_count", "Ten prior WSI roles required")
    selected_rows = {}
    for row in wsi_rows + [rows["model_config/h_optimus_1"], rows["weight/h_optimus_1"]]:
        sid = row["slide_id"]
        if row["kind"] == "wsi":
            _need(row["source_id"] == "wsi/" + sid and row["bank"] == "" and sid in ids,
                  "prior_wsi_identity", "Prior WSI role identity")
        else:
            expected_kind = "model_config" if row["source_id"].startswith("model_config/") else "weight"
            _need(row["kind"] == expected_kind and row["bank"] == "h_optimus_1" and sid == "",
                  "prior_model_identity", "Prior model role identity")
        _need(row["expected_presence"] == "true" and row["observed_presence"] == "true",
              "prior_presence", "Required prior role was not present")
        size = _decimal(row["observed_size_bytes"], 1)
        digest = _sha(row["observed_sha256"])
        _path(row["path"])
        _path(row["resolved_path"])
        _text(row["digest_basis"])
        attribution = {
            "source_control": "stage1b_source_inventory", "source_id": row["source_id"],
            "producer_path": row["path"], "producer_resolved_path": row["resolved_path"],
            "size": size, "sha256": digest, "digest_basis": row["digest_basis"],
            "review_sha256": spec["controls"]["stage1b_review"]["sha256"],
        }
        selected_rows[row["source_id"]] = attribution
    attributions = []
    for record in context.packet["asset_records"]:
        role = record["role"]
        if role == "template":
            original = mapped[("source", "template")]
            attribution = {
                "source_control": "cpu_input_bindings", "source_id": "source:template",
                "producer_path": original["path"], "producer_resolved_path": original["resolved_path"],
                "size": original["size"], "sha256": original["sha256"],
                "digest_basis": "accepted_cpu57_source_binding",
                "review_sha256": spec["controls"]["cpu_review"]["sha256"],
            }
        else:
            source_id = {"model_config": "model_config/h_optimus_1",
                         "checkpoint": "weight/h_optimus_1", "selected_wsi": "wsi/10003"}[role]
            attribution = selected_rows[source_id]
        _need(record["declared_size"] == attribution["size"]
              and record["declared_sha256"] == attribution["sha256"],
              "asset_prior_link", "Asset does not match prior observed bytes")
        attributions.append(copy.deepcopy(attribution))
    context.prior = {
        row["slide_id"]: {
            "resolved_path": row["resolved_path"],
            "size_bytes": _decimal(row["observed_size_bytes"], 1),
            "sha256": row["observed_sha256"],
            "receipt_reference": "stage1b_review:" + spec["controls"]["stage1b_review"]["sha256"]
            + "#source_inventory:" + row["source_id"],
        } for row in wsi_rows
    }
    context.ids = ids
    for record, attribution in zip(context.packet["asset_records"], attributions):
        record["prior_attribution"] = attribution
    context.packet["inventory_initial"] = _inventory(ids)
    context.packet["inventory_final"] = _inventory(ids)
    context.packet["inventory_comparison"] = _comparison(ids)
    context.stage("trusted_basis").update(state="complete", result={
        "expected": copy.deepcopy(spec["expected"]),
        "prior_wsi_ids": [sid for sid in ids if sid in context.prior],
    })


def _inventory_pass(context: _Context, final: bool = False) -> None:
    name = "final_inventory" if final else "initial_inventory"
    context.stage_name = name
    output = context.packet["inventory_final" if final else "inventory_initial"]
    stage = context.stage(name)
    output["state"], stage["state"] = "partial", "partial"
    summarized = set()
    try:
        for index, sid in enumerate(context.ids):
            output["in_progress_id"] = sid
            output["unvisited_ids"] = context.ids[index + 1:]
            probe, bookends, outcome = _probe_path(WSI_ROOT + "/" + sid + ".svs")
            output["probes"][sid], output["bookends"][sid] = probe, bookends
            output["visited_ids"].append(sid)
            output["in_progress_id"] = None
            if outcome and (outcome, probe["status"]) not in summarized:
                summarized.add((outcome, probe["status"]))
                context.issue(_Fault("inventory_" + probe["status"],
                                     "Inventory contains " + probe["status"],
                                     "availability" if outcome == 2 else "integrity",
                                     outcome), stage=name)
            if sid == "10003" and outcome:
                if "selected_wsi_unavailable" not in context.large_blockers:
                    context.large_blockers.append("selected_wsi_unavailable")
        output["frozen_result"] = context.observer.inventory_targets(
            context.targets, context.metadata, output["probes"], context.prior,
            lexical_root=WSI_ROOT)
        output["probes"] = None
        output["state"] = "complete"
        if stage["state"] == "partial":
            stage["state"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            context.interrupted = True
        context.issue(exc, stage=name)
        stage["state"] = "partial"
        if output["in_progress_id"] is not None and hasattr(exc, "partial_bookends"):
            output["bookends"][output["in_progress_id"]] = exc.partial_bookends
        if context.interrupted:
            raise


def _inventory_rows(output: dict) -> dict:
    if output["frozen_result"] is not None:
        return {r["slide_id"]: {k: r[k] for k in PROBE_FIELDS}
                for r in output["frozen_result"]["rows"]}
    return output["probes"] or {}


def _compare_inventories(context: _Context) -> None:
    if context.ids is None:
        return
    before, after = context.packet["inventory_initial"], context.packet["inventory_final"]
    first, last = _inventory_rows(before), _inventory_rows(after)
    shared = [sid for sid in context.ids if sid in first and sid in last]
    result = _comparison(context.ids)
    result.update(state="complete" if len(shared) == len(context.ids) else "partial",
                  compared_ids=shared, uncomparable_ids=[sid for sid in context.ids if sid not in first or sid not in last])
    for sid in shared:
        fields = [k for k in PROBE_FIELDS if k != "error_message" and first[sid][k] != last[sid][k]]
        fields.extend(k for k in BOOK_COMPARE if before["bookends"][sid][k] != after["bookends"][sid][k])
        if fields:
            result["changed_ids"].append(sid)
            result["changes"].append({
                "slide_id": sid, "changed_fields": fields, "initial_status": first[sid]["status"],
                "final_status": last[sid]["status"],
                "affected_custody": "selected_wsi" if sid == "10003" else "inventory_only",
            })
            if sid == "10003":
                _custody(context.packet["asset_records"][3], "invalidated")
    context.packet["inventory_comparison"] = result
    if result["changed_ids"]:
        context.issue(_Fault("inventory_changed", "Initial/final inventory changed", "integrity"),
                      stage="final_inventory")


def _record_region(context: _Context, reader: _Reader, name: str, length: int) -> bytes:
    record = {
        "name": name, "offset": 0 if name == "prefix8" else 8, "length": length,
        "sha256": None, "capture_state": "not_performed", "final_state": "not_performed",
        "final_sha256": None, "byte_equal": None,
    }
    reader.record["regions"].append(record)
    try:
        encoded = reader.exact(length)
        record.update(sha256=hashlib.sha256(encoded).hexdigest(), capture_state="captured")
        context.regions[name] = encoded
        return encoded
    except BaseException:
        record["capture_state"] = "failed"
        raise


def _initial_assets(context: _Context) -> None:
    template, config, checkpoint, selected = context.packet["asset_records"]
    for name, record, ceiling in (("template", template, LIMITS["png_bytes"]),
                                  ("config", config, LIMITS["config_bytes"])):
        context.stage_name = name
        stage = context.stage(name)
        try:
            encoded = _small(context, record, ceiling)
            if name == "template":
                stage["result"] = context.observer.observe_png(encoded, declared_size=record["declared_size"])
                if stage["result"]["unsupported_properties"]:
                    for code in stage["result"]["unsupported_properties"]:
                        context.issue(context.observer.AdmissionError("unsupported", code, "png.encoded"),
                                      record["role"])
                    continue
            else:
                result = {"encoded_size": len(encoded), "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
                          "raw_utf8": None, "parsed_json": None, "comparison_state": "not_performed"}
                stage["result"] = result
                try:
                    result["raw_utf8"] = encoded.decode("utf-8", "strict")
                    result["parsed_json"] = _json_bytes(encoded)
                except (UnicodeError, _Fault) as exc:
                    context.issue(_Fault(getattr(exc, "code", "config_utf8"), str(exc),
                                         "observation", 2), record["role"])
                    continue
            stage["state"] = "complete"
        except Exception as exc:
            context.issue(exc, record["role"])
        if context.interrupted:
            return
    context.stage_name = "checkpoint_header"
    header_stage = context.stage("checkpoint_header")
    header_stage["result"] = {"safetensors_observation": None, "schema_comparison": None}
    reader = None
    try:
        reader = _Reader(context, checkpoint, "initial", LIMITS["each_large_asset_bytes"])
        context.active_reader = reader
        prefix = _record_region(context, reader, "prefix8", 8)
        length = struct.unpack("<Q", prefix)[0]
        if length > LIMITS["safetensors_header_bytes"]:
            context.observer.observe_safetensors(prefix, b"", total_size=reader.size)
        if length + 8 > reader.size:
            raise context.observer.AdmissionError("malformed", "header_total_size", "safetensors.header")
        header = _record_region(context, reader, "header", length)
        observation = context.observer.observe_safetensors(prefix, header, total_size=reader.size)
        header_stage["result"]["safetensors_observation"] = observation
        comparison = context.observer.compare_vit224_schema(observation)
        header_stage["result"]["schema_comparison"] = comparison
        if not comparison["match"]:
            context.issue(_Fault("schema_mismatch", "Checkpoint does not match descriptive schema",
                                 "observation", 2), "checkpoint")
        else:
            header_stage["state"] = "complete"
    except Exception as exc:
        context.issue(exc, "checkpoint")
    if context.interrupted:
        return
    if context.large_blockers:
        for name in ("checkpoint_full", "selected_wsi_full"):
            context.stage(name).update(state="blocked", blocker_codes=list(context.large_blockers))
    else:
        context.stage_name = "checkpoint_full"
        try:
            reader.full(context.regions["prefix8"] + context.regions["header"])
            context.stage("checkpoint_full")["state"] = "complete"
        except Exception as exc:
            context.issue(exc, "checkpoint")
    if reader is not None:
        try:
            reader.finish()
        except Exception as exc:
            context.issue(exc, "checkpoint", stage="checkpoint_header")
        context.active_reader = None
    if context.large_blockers:
        if context.stage("selected_wsi_full")["state"] == "not_started":
            context.stage("selected_wsi_full").update(state="blocked", blocker_codes=list(context.large_blockers))
    else:
        context.stage_name = "selected_wsi_full"
        try:
            with _Reader(context, selected, "initial", LIMITS["each_large_asset_bytes"]) as selected_reader:
                selected_reader.full()
            context.stage("selected_wsi_full")["state"] = "complete"
        except Exception as exc:
            context.issue(exc, "selected_wsi")


def _final_asset(context: _Context, record: dict) -> None:
    role = record["role"]
    if role in ("template", "model_config"):
        ceiling = LIMITS["png_bytes"] if role == "template" else LIMITS["config_bytes"]
        _small(context, record, ceiling, "final")
        return
    try:
        with _Reader(context, record, "final", LIMITS["each_large_asset_bytes"]) as reader:
            prefix = b""
            if role == "checkpoint":
                for region in record["regions"]:
                    if region["capture_state"] != "captured":
                        continue
                    original = context.regions[region["name"]]
                    region["final_state"] = "failed"
                    actual = reader.exact(region["length"])
                    region.update(final_sha256=hashlib.sha256(actual).hexdigest(),
                                  byte_equal=actual == original,
                                  final_state="matched" if actual == original else "rejected")
                    prefix += actual
                    _need(actual == original, "region_changed", "Original checkpoint region changed")
            if record["whole_file_checks"][0]["observed_sha256"] is not None:
                reader.full(prefix)
        _need(record["initial_descriptor"]["opened"] == record["final_descriptor"]["opened"]
              and record["initial_descriptor"]["resolved_path"] == record["final_descriptor"]["resolved_path"]
              and record["initial_descriptor"]["lexical_before"] == record["final_descriptor"]["lexical_before"],
              "region_changed", "Asset identity changed between passes")
        if (all(c["expected_match"] is True for c in record["whole_file_checks"])
                and record["initial_descriptor"]["coherent"] is True
                and record["final_descriptor"]["coherent"] is True):
            _custody(record, "full_verified")
    except BaseException:
        _custody(record, "invalidated")
        raise


def _final_checks(context: _Context) -> None:
    context.stage_name = "final_checks"
    result = {"checked_roles": [], "region_only_roles": [], "failed_roles": []}
    stage = context.stage("final_checks")
    stage["result"] = result
    for record, ceiling in context.consumed:
        if context.interrupted:
            break
        role = record["role"]
        result["checked_roles"].append(role)
        try:
            _small(context, record, ceiling, "final")
            _need(record["initial_descriptor"]["opened"] == record["final_descriptor"]["opened"]
                  and record["initial_descriptor"]["resolved_path"] == record["final_descriptor"]["resolved_path"]
                  and record["initial_descriptor"]["lexical_before"] == record["final_descriptor"]["lexical_before"],
                  "region_changed", "Consumed source identity changed between passes")
        except (Exception, KeyboardInterrupt) as exc:
            _custody(record, "invalidated")
            result["failed_roles"].append(role)
            context.issue(exc, role)
    for record in context.packet["asset_records"]:
        if context.interrupted:
            break
        role = record["role"]
        if role in ("template", "model_config"):
            continue
        full = record["whole_file_checks"][0]["observed_sha256"] is not None
        regions = any(r["capture_state"] == "captured" for r in record["regions"])
        if not full and not regions:
            continue
        result["checked_roles"].append(role)
        if not full:
            result["region_only_roles"].append(role)
        try:
            _final_asset(context, record)
        except (Exception, KeyboardInterrupt) as exc:
            result["failed_roles"].append(role)
            context.issue(exc, role)
    try:
        _check_environment()
        _need(context.observer is None or sys.modules.get(OBSERVER_NAME) is context.observer,
              "observer_registration_changed", "Observer registration substituted")
    except Exception as exc:
        context.issue(exc, stage="final_checks")
    if stage["state"] == "not_started":
        stage["state"] = "complete"


def _load_basis(context: _Context, spec: dict) -> None:
    record = context.packet["observer_binding"]
    _need(OBSERVER_NAME not in sys.modules, "observer_module_collision", "Observer module already registered")
    encoded = _small(context, record, LIMITS["source_bytes"])
    observer = types.ModuleType(OBSERVER_NAME)
    observer.__file__ = record["declared_path"]
    observer.__package__ = ""
    sys.modules[OBSERVER_NAME] = observer
    context.observer = observer
    exec(compile(encoded, record["declared_path"], "exec"), observer.__dict__)
    _check_environment()
    controls = {}
    for record in context.packet["control_bindings"][1:]:
        role = record["role"]
        encoded = _small(context, record, LIMITS["each_control_bytes"])
        if role in CSV_HEADERS:
            controls[role] = _csv_bytes(encoded, role)
        else:
            value = _json_bytes(encoded)
            _fields(value, JSON_KEYS[role].split())
            _need(_int(value["schema_version"]) == 1, "control_schema", "Control schema version")
            controls[role] = value
    _reconcile(context, spec, controls)


def _execute(context: _Context, spec_path: str, spec_sha256: str, entry_binding: dict) -> None:
    # G0 validates independent declarations before any source read.
    errors = []
    spec_record = None
    try:
        context.packet["spec_sha256"] = _sha(spec_sha256)
    except Exception as exc:
        errors.append(exc)
    try:
        path = _absolute(_path(spec_path), os.getcwd())
        _path(path)
        if context.packet["spec_sha256"] is not None:
            spec_record = _binding("spec", path, None, spec_sha256)
            context.packet["control_bindings"] = [spec_record]
    except Exception as exc:
        errors.append(exc)
    try:
        _fields(entry_binding, ("path", "resolved_path", "size", "sha256"))
        path = _path(entry_binding["path"])
        _need(path.startswith("/") and _path(entry_binding["resolved_path"]).startswith("/"),
              "entry_path", "Entry binding must be absolute")
        size = _int(entry_binding["size"], 1, LIMITS["source_bytes"])
        digest = _sha(entry_binding["sha256"])
        context.packet["entry_binding"] = _binding("entry", path, size, digest)
    except Exception as exc:
        errors.append(exc)
    for exc in errors:
        context.issue(exc)
    if errors:
        return
    _check_environment()
    _need(__name__ == CLI_NAME and sys.modules.get(CLI_NAME) is not None
          and sys.modules[CLI_NAME].__dict__ is globals(),
          "verified_entry_required", "Caller must execute captured verified bytes in fixed module")
    _small(context, context.packet["entry_binding"], LIMITS["source_bytes"])
    _need(context.packet["entry_binding"]["initial_descriptor"]["resolved_path"]
          == entry_binding["resolved_path"], "entry_resolved_changed", "Entry resolved path differs from launcher")
    encoded = _small(context, spec_record, LIMITS["spec_bytes"])
    parent = spec_record["declared_path"].rsplit("/", 1)[0] or "/"
    spec = _declaration(_json_bytes(encoded), parent)
    context.packet["run_id"] = spec["run_id"]
    declaration = spec["observer"]
    context.packet["observer_binding"] = _binding("observer", declaration["path"],
                                                 declaration["size"], declaration["sha256"])
    for role in CONTROLS:
        declaration = spec["controls"][role]
        context.packet["control_bindings"].append(
            _binding(role, declaration["path"], declaration["size"], declaration["sha256"]))
    context.packet["logical_io"]["controls"]["declared_bytes"] = sum(
        d["size"] for d in spec["controls"].values())
    for role in ASSETS:
        declaration = spec["assets"][role]
        record = _binding(role, declaration["path"], declaration["size"], declaration["sha256"])
        record.update(prior_attribution=None, current_resolved_path=None, regions=[])
        context.packet["asset_records"].append(record)
    _load_basis(context, spec)
    _inventory_pass(context)
    _initial_assets(context)


def _encode(packet: dict) -> bytes:
    buffer = bytearray()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    for chunk in encoder.iterencode(packet):
        encoded = chunk.encode("utf-8")
        seen = len(buffer) + len(encoded)
        if seen + 1 > LIMITS["stdout_packet_bytes"]:
            exc = _Fault("stdout_limit", "Serialized stdout exceeds actual 16MiB ceiling", "output")
            exc.bytes_seen = seen
            raise exc
        buffer.extend(encoded)
    buffer.extend(b"\n")
    return bytes(buffer)


def _fallback(packet: dict, exc: Exception) -> dict:
    result = copy.deepcopy(packet)
    result.update(packet_kind="partial_error", exit_code=1)
    overflow = result["overflow"]
    overflow.update(occurred=True, cause_code=getattr(exc, "code", "serialization_error"),
                    serialization_bytes_seen=getattr(exc, "bytes_seen", None))
    for name in ("inventory_initial", "inventory_final"):
        inventory = result[name]
        rows = len(inventory["visited_ids"] or [])
        if inventory["in_progress_id"] is not None and inventory["in_progress_id"] in inventory["bookends"]:
            rows += 1
        overflow["omitted_counts"][name + "_rows"] = rows
        overflow["omitted_sections"].append(name)
        fields = [k for k in inventory if k != "state"]
        result[name] = {
            "state": "omitted_output_limit", "visited_ids": None, "in_progress_id": None,
            "unvisited_ids": None, "probes": None, "bookends": None,
            "frozen_result": None, "omitted_fields": fields,
        }
    result["inventory_comparison"] = {
        "state": "omitted_output_limit", "compared_ids": None, "changed_ids": None,
        "changes": None, "uncomparable_ids": None,
    }
    overflow["omitted_sections"].append("inventory_comparison")
    for stage in result["stages"]:
        if stage["result"] is not None:
            overflow["omitted_sections"].append("stages." + stage["name"] + ".result")
            stage["result"] = None
        if stage["name"] in ("initial_inventory", "final_inventory"):
            stage["state"] = "omitted_output_limit"
        if stage["name"] == "serialization":
            stage["state"] = "failed"
            stage["blocker_codes"] = [overflow["cause_code"]]
    return result


def _finish_packet(context: _Context) -> RunResult:
    stage = context.stage("serialization")
    stage.update(state="complete", result={"encoding": "utf-8", "json_format": "canonical_compact"})
    context.packet.update(exit_code=context.exit_code,
                          packet_kind={0: "complete", 2: "complete_negative", 1: "partial_error"}[context.exit_code])
    packet = context.packet
    diagnostic = b""
    try:
        encoded = _encode(packet)
    except Exception as exc:
        context.issue(_Fault(getattr(exc, "code", "serialization_error"), str(exc), "output"),
                      stage="serialization")
        try:
            packet = _fallback(context.packet, exc)
            encoded = _encode(packet)
        except Exception as fallback_error:
            text, _ = _truncate("Ordinary and fallback packet construction failed: " + str(fallback_error))
            return {"packet": None, "stdout_bytes": b"", "stderr_bytes": (text + "\n").encode("utf-8"),
                    "exit_code": 1}
        diagnostic = b"Packet serialization failed; explicit bounded omissions emitted.\n"
    if packet["exit_code"] == 1 and not diagnostic:
        diagnostic = b"Observation caller encountered a technical failure; see packet issues.\n"
    return {"packet": packet, "stdout_bytes": encoded, "stderr_bytes": diagnostic,
            "exit_code": packet["exit_code"]}


def run(spec_path: str, *, spec_sha256: str, entry_binding: dict) -> RunResult:
    """Acquire only declared inputs and return one preconstructed packet without emission."""
    context = _Context()
    try:
        _execute(context, spec_path, spec_sha256, entry_binding)
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            context.interrupted = True
        context.issue(exc)
    finally:
        if context.active_reader is not None:
            try:
                context.active_reader.finish()
            except (Exception, KeyboardInterrupt) as exc:
                prior = len(context.packet["issues"]) - 1
                context.issue(exc, secondary=prior if prior >= 0 else None)
            context.active_reader = None
    try:
        if not context.interrupted:
            _final_checks(context)
            if context.ids is not None and not context.interrupted:
                _inventory_pass(context, True)
                _compare_inventories(context)
            elif context.interrupted:
                context.stage("final_inventory").update(state="blocked", blocker_codes=["interrupted"])
        else:
            context.stage("final_checks").update(state="blocked", blocker_codes=["interrupted"])
            context.stage("final_inventory").update(state="blocked", blocker_codes=["interrupted"])
    except (Exception, KeyboardInterrupt) as exc:
        context.issue(exc)
    finally:
        if context.observer is not None and sys.modules.get(OBSERVER_NAME) is context.observer:
            del sys.modules[OBSERVER_NAME]
    return _finish_packet(context)


def _write_checked(stream, encoded: bytes) -> None:
    position = 0
    while position < len(encoded):
        count = stream.write(encoded[position:])
        if type(count) is not int or count <= 0 or count > len(encoded) - position:
            raise OSError(errno.EIO, "Invalid/short output write")
        position += count
    stream.flush()


def _diagnostic(encoded: bytes) -> bool:
    if len(encoded) > LIMITS["stderr_bytes"]:
        marker = b"\n[caller stderr truncated]\n"
        encoded = encoded[:LIMITS["stderr_bytes"] - len(marker)] + marker
    try:
        _write_checked(sys.stderr.buffer, encoded)
    except Exception:
        # There is no third channel; return status and capture readback expose this failure.
        return False
    return True


def main(argv: list[str] | tuple[str, ...] | None = None, *, entry_binding: dict) -> int:
    """Machine packet emission, distinct from standalone help and usage errors."""
    supplied = sys.argv[1:] if argv is None else argv
    if type(supplied) not in (list, tuple) or not all(type(arg) is str for arg in supplied):
        _diagnostic(b"Usage error: argument list must contain strings.\n")
        return 1
    arguments = list(supplied)
    if arguments in (["-h"], ["--help"]):
        try:
            _write_checked(sys.stdout.buffer, b"Usage: --spec PATH --spec-sha256 HEX\nVerified -I -S -B byte entry required.\n")
            return 0
        except (OSError, ValueError):
            return 1
    if (len(arguments) != 4 or {arguments[0], arguments[2]} != {"--spec", "--spec-sha256"}):
        _diagnostic(b"Usage error: require --spec PATH --spec-sha256 HEX; no positional arguments.\n")
        return 1
    flags = {arguments[0]: arguments[1], arguments[2]: arguments[3]}
    result = run(flags["--spec"], spec_sha256=flags["--spec-sha256"], entry_binding=entry_binding)
    if result["stderr_bytes"]:
        if not _diagnostic(result["stderr_bytes"]):
            return 1
    if result["packet"] is None:
        return 1
    try:
        _write_checked(sys.stdout.buffer, result["stdout_bytes"])
    except Exception as exc:
        text, _ = _truncate("Stdout emission/flush failed; no replacement packet: " + str(exc))
        _diagnostic((text + "\n").encode("utf-8"))
        return 1
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main(entry_binding=None))
