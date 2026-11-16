"""Source-bound CPU preparation; native capture requires a separate private release."""
from __future__ import annotations

import argparse
import ast
import dataclasses
import email.parser
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import socket
import stat
import struct
import sys
import time
import types

MIB = 1024 ** 2
TASK = "hoptimus_filter_cpu_software_preparation"
CALLER = "r2_1_hoptimus_input_observation_cli"
NATIVE_NAMES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
LIMITS = {
    "spec_bytes": 2*MIB, "inventory_bytes_each": 8*MIB,
    "source_bytes_each": 2*MIB, "startup_pth_bytes_each": 65536,
    "synthetic_feature_bytes_each": MIB, "synthetic_npy_bytes_each": 65536,
    "persistent_canonical_files": 16384, "lexical_alias_paths": 32768,
    "aggregate_unique_software_bytes": 16*1024**3,
    "individual_software_bytes": 2*1024**3, "module_records_each_phase": 8192,
    "executable_map_paths_each_phase": 8192, "role_occurrences": 65536,
    "native_pool_records_each_phase": 64,
    "runtime_implementation_files_each_phase": 76, "proc_maps_bytes": 8*MIB,
    "ordinary_json_bytes": 8*MIB, "software_map_json_bytes": 32*MIB,
    "diagnostic_json_bytes": 8*MIB, "handoff_bytes": 16384,
    "stdout_bytes": 16384, "stderr_bytes": MIB,
    "kernel_accounted_working_bytes": 8*MIB, "pair_cost_cells": 4,
    "live_custody_descriptors": 384, "source_binding_records": 512,
}


class StartupLeases:
    def __init__(self, ctx):
        self.ctx, self.held = ctx, []

    def acquire(self):
        sources = self.ctx.spec["startup_sources"]
        declarations = [sources["site_source"]]
        declarations += [v for d in sources["site_directories"] for v in d["pth_files"]]
        declarations += [v["descriptor"] for v in sources["customization"].values() if v["state"] == "present"]
        require(len(declarations)+8 <= 384, "activation descriptor cap")
        for desc in declarations:
            _, record = self.ctx.inventory.read(desc["path"], expected=desc, cap=2*MIB)
            fd = os.open(desc["path"], os.O_RDONLY | os.O_NONBLOCK)
            self.held.append((fd, desc, record))
            require(identity(os.fstat(fd)) == (record["st_dev"], record["st_ino"], record["size"]),
                    "activation descriptor acquisition")

    def close(self, primary=None):
        for fd, desc, original in reversed(self.held):
            try:
                _, final = self.ctx.inventory.read(desc["path"], expected=desc, cap=2*MIB)
                require(final == original and identity(os.fstat(fd)) ==
                        (final["st_dev"], final["st_ino"], final["size"]), "activation final content and held identity")
            except BaseException as extra:
                if primary is None:
                    primary = extra
                else:
                    add_secondary(primary, extra, "activation source custody")
            try:
                os.close(fd)
            except BaseException as extra:
                if primary is None:
                    primary = extra
                else:
                    add_secondary(primary, extra, "activation descriptor close")
        self.held.clear()
        if primary is not None:
            raise primary


PHASES = ("bootstrap", "before", "after")
CAPTURE_NAMES = (
    "startup.json", "runtime_before.json", "runtime_after.json",
    "software_map.json", "exercises.json", "input_bindings.json",
    "summary.json", "capture.json",
)
DATA_NAMES = (
    "query.pkl", "reference.pkl", "baseline.pkl", "coordinates.npy",
    *(f"mask_f{f}_b{b}.npy" for f in range(5) for b in (5, 100)),
)
CONTROL_NAMES = ("synthetic_spec.json", "synthetic_complete.json", "synthetic_review.json")
FIXTURE_NAMES = (*DATA_NAMES, *CONTROL_NAMES, "fixture_manifest.json")
OPERATION_NAMES = (
    "capture.stdout", "capture.stderr", "capture_exit.json", "generated_source.py",
    "verify.stdout", "verify.stderr", "verify_exit.json", "supervisor_result.json",
    "terminal_acceptance.json",
)
INHERITED = (
    "prosper.config", "prosper.pipeline", "prosper.embeddings",
    "prosper.embeddings.patch_embedding", "prosper.embeddings.patch_embedding.base",
    "prosper.embeddings.patch_embedding.gigapath_embedding",
    "prosper.embeddings.patch_embedding.uni2_embedding",
    "prosper.embeddings.patch_embedding.virchow2_embeddings",
    "prosper.embeddings.slide_embedding", "prosper.embeddings.slide_embedding.base",
    "prosper.embeddings.slide_embedding.bob_embedding",
    "prosper.embeddings.slide_embedding.ot_embedding",
    "prosper.embeddings.slide_embedding.percentile_embedding", "prosper.evaluation.metrics",
    "prosper.evaluation.paths", "prosper.evaluation.slide_headers", "prosper.preprocessing",
    "prosper.preprocessing.mosaic", "prosper.search", "prosper.search.roi_search",
    "prosper.search.slide_search",
)
DIRECT_MODULES = {
    "runner": "prosper.evaluation.r2_1_hoptimus_filter_retrieval_cli",
    "head_reader": "prosper.evaluation.r2_1_hoptimus_head_reader",
    "fit": "prosper.evaluation.r2_1_hoptimus_fit",
    "asset_io": "prosper.evaluation.r2_1_hoptimus_asset_io",
    "head_cli": "prosper.evaluation.r2_1_hoptimus_head_cli",
    "filter_kernel": "prosper.evaluation.r2_1_hoptimus_filter",
    "filter_cli": "prosper.evaluation.r2_1_hoptimus_filter_cli",
    "cpu_io": "prosper.evaluation.r2_1_hoptimus_io",
    "legacy_io": "prosper.evaluation.feature_bank_audit",
    "package_init": "prosper", "evaluation_init": "prosper.evaluation",
    "mosaic_runner": "prosper.evaluation.r2_1_hoptimus_mosaic_cli",
    "mosaic_reader": "prosper.evaluation.r2_1_hoptimus_mosaic_reader",
    "main_cli": "prosper.evaluation.r2_1_hoptimus_main_cli",
    "main_kernel": "prosper.evaluation.r2_1_hoptimus_main",
}
CODE_MODULES = {**DIRECT_MODULES, **{"inherited:"+n: n for n in INHERITED}}
PRIOR_NAMES = (
    "original_profile", "cpu_capture", "cpu_verification",
    "cpu_private_binding", "cpu_private_reconciliation",
)
NEW_LIMITATIONS = [
    "software_preparation_not_runtime_acceptance",
    "synthetic_not_real_masks_heads_features_or_results",
    "cooperative_custody_not_atomic_execution_attestation",
    "manual_site_activation_not_normal_CLI_startup_acceptance",
    "no_full_solver_or_row_feasibility", "no_submission_or_scientific_acceptance",
    "fileless_object_creation_origin_unresolved",
    "synthetic_dataframe_decode_not_real_baseline_equivalence",
    "initial_import_and_normal_worker_authority_still_private_gates",
]
GENERATED = "_remote_module_non_scriptable"
GENERATED_HASH = "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
CATALOG_HASH = "6675a37caf5d1da1cea27b5d023283da8da22f5149a51a5eb1afa621b2f3de45"
SIX_HASH = "4355b686a8bb123458c3a907c394f5a4e38c8bdd88a75cd7299abe2ddb9d447a"
BOOTSTRAP_HASH = "ccfc1943509ae7a8d8e6f41eca0160547883a0e8d1490d31c23bca7e0413bb15"
TORCH_NAMESPACES = {
    "torch.ops": ("_ops.py", "torch._ops"),
    "torch.classes": ("_classes.py", "torch._classes"),
}
PROXY_NAMES = (
    "setuptools._vendor.jaraco", "setuptools._vendor.more_itertools",
    "setuptools._vendor.ordered_set", "setuptools._vendor.packaging",
    "setuptools.extern.jaraco", "setuptools.extern.more_itertools",
    "setuptools.extern.ordered_set", "setuptools.extern.packaging",
    "torch.backends.cudnn", "torch.backends.mkldnn", "torch.backends.quantized",
)
_SECONDARIES = {}
_EMITTED = None
_ACTIVE_CONTEXT = None
OWNER_CONTRACT = {
    "path": "/PROSPER_WORKSPACE/"
            "prosper_revision/prosper_revision_output/r2_1_mosaic_stain_review_20261003/"
            "filter_cpu_failure_ledger_contract_candidate001.txt",
    "size": 10794,
    "sha256": "b49d475f79fd021d68fb6bd9e78e95d15da427f40bdb567314ac5f834283b7d7",
}
FAILURE_INPUT_BYTES = 49424896
FAILURE_STAGES = (
    "bootstrap", "site", "import", "profile_before", "fixture", "decode", "prepare", "compute",
    "profile_after", "final_inputs", "write_capture", "release_capture", "verify_inputs",
    "write_verification", "release_verification", "supervisor", "external_terminal",
)


class ProbeError(ValueError):
    """A declaration, record, consumption or namespace gate failed."""


def require(condition, message):
    if not condition:
        raise ProbeError(message)


def keys(value, expected, context):
    require(type(value) is dict, context+": base dict required")
    require(all(type(k) is str for k in value), context+": exact string keys")
    require(set(value) == set(expected), context+": exact keys")


def integer(value, minimum=0, maximum=2**63-1):
    require(type(value) is int and minimum <= value <= maximum, "strict integer domain")
    return value


def boolean(value):
    require(type(value) is bool, "strict Boolean")
    return value


def text(value, ceiling=4096, empty=False):
    require(type(value) is str and (empty or bool(value)), "base text required")
    require(not any(c in value for c in "\x00\r\n"), "unambiguous text required")
    require(len(value) <= ceiling and len(value.encode("utf-8")) <= ceiling, "text byte cap")
    return value


def identifier(value, run=False):
    text(value, 128)
    require(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}" if run else
                         r"[A-Za-z0-9_.:-]{1,128}", value) is not None, "identifier")
    return value


def digest(value):
    require(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None, "SHA256")
    return value


def path_text(value):
    text(value)
    require("\\" not in value and Path(value).is_absolute()
            and os.path.normpath(value) == value, "normalized absolute path")
    return value


def array(value, ceiling):
    require(type(value) is list and len(value) <= ceiling, "base list/count cap")
    return value


def leaf(value):
    identifier(value, run=True)
    require(value not in (".", ".."), "artifact leaf")
    return value


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_depth(data):
    stack, quoted, escaped = [], False, False
    for octet in data:
        if quoted:
            if escaped:
                escaped = False
            elif octet == 92:
                escaped = True
            elif octet == 34:
                quoted = False
        elif octet == 34:
            quoted = True
        elif octet in (91, 123):
            require(len(stack) < 32, "JSON depth32 exceeded before decode")
            stack.append(octet)
        elif octet in (93, 125):
            require(bool(stack) and stack.pop() == (91 if octet == 93 else 123),
                    "JSON bracket mismatch")
    require(not stack and not quoted and not escaped, "JSON incomplete structure")


def strict_json(data, ceiling=8*MIB, canonical=False):
    require(type(data) is bytes and len(data) <= ceiling, "JSON pre-decode byte cap")
    require(not data.startswith(b"\xef\xbb\xbf"), "JSON BOM")
    check_depth(data)

    def pairs(items):
        result = {}
        for name, value in items:
            require(name not in result, "duplicate JSON key")
            result[name] = value
        return result

    def constant(_):
        raise ProbeError("nonfinite JSON constant")

    value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    validate_atoms(value)
    if canonical:
        require(encoded(value, ceiling) == data, "noncanonical original JSON")
    return value


def validate_atoms(value, ancestry=None, depth=0):
    require(depth <= 32, "JSON structural depth")
    kind = type(value)
    if kind is float:
        require(math.isfinite(value), "nonfinite JSON scalar")
    elif kind is int:
        require(value.bit_length() <= 256, "JSON integer256-bit domain")
    elif kind is str:
        require(len(value) <= 32*MIB, "JSON string administrative cap")
    elif kind in (dict, list):
        ancestry = set() if ancestry is None else ancestry
        require(id(value) not in ancestry, "cyclic JSON container")
        require(len(value) <= 65536, "JSON container count cap")
        ancestry.add(id(value))
        try:
            if kind is dict:
                require(all(type(k) is str for k in value), "JSON exact string keys")
                for k, v in value.items():
                    validate_atoms(k, ancestry, depth+1)
                    validate_atoms(v, ancestry, depth+1)
            else:
                for v in value:
                    validate_atoms(v, ancestry, depth+1)
        finally:
            ancestry.remove(id(value))
    else:
        require(value is None or kind is bool, "unsupported JSON value type")


def _char_fragment(character):
    return json.dumps(character, ensure_ascii=True)[1:-1].encode("ascii")


def canonical_fragments(value, ancestry=None, depth=0):
    require(depth <= 32, "encoding depth32")
    kind = type(value)
    if kind is str:
        yield b'"'
        for c in value:
            yield _char_fragment(c)
        yield b'"'
    elif value is None:
        yield b"null"
    elif kind is bool:
        yield b"true" if value else b"false"
    elif kind is int:
        require(value.bit_length() <= 256, "bounded integer encoding")
        yield format(value, "d").encode("ascii")
    elif kind is float:
        require(math.isfinite(value), "finite float encoding")
        yield json.dumps(value, allow_nan=False).encode("ascii")
    elif kind in (dict, list):
        ancestry = set() if ancestry is None else ancestry
        require(id(value) not in ancestry and len(value) <= 65536, "encoding ancestry/count")
        ancestry.add(id(value))
        try:
            if kind is dict:
                require(all(type(k) is str for k in value), "encoding string keys")
                yield b"{"
                for index, key in enumerate(sorted(value)):
                    if index:
                        yield b","
                    yield from canonical_fragments(key, ancestry, depth+1)
                    yield b":"
                    yield from canonical_fragments(value[key], ancestry, depth+1)
                yield b"}"
            else:
                yield b"["
                for index, item in enumerate(value):
                    if index:
                        yield b","
                    yield from canonical_fragments(item, ancestry, depth+1)
                yield b"]"
        finally:
            ancestry.remove(id(value))
    else:
        raise ProbeError("unsupported encoding object")


def encoded(value, ceiling):
    chunks, total, buffer = [], 0, bytearray()
    for fragment in canonical_fragments(value):
        require(total + len(fragment) <= ceiling, "encoding preallocation cap")
        total += len(fragment)
        if len(buffer) + len(fragment) > 65536:
            chunks.append(bytes(buffer))
            buffer.clear()
        buffer.extend(fragment)
    require(total+1 <= ceiling, "encoding newline cap")
    if len(buffer) == 65536:
        chunks.append(bytes(buffer))
        buffer.clear()
    buffer.extend(b"\n")
    chunks.append(bytes(buffer))
    return b"".join(chunks)


def stream_json(fd, value, ceiling):
    count, hasher, buffer = 0, hashlib.sha256(), bytearray()

    def flush():
        view = memoryview(buffer)
        pos = 0
        try:
            while pos < len(view):
                wrote = os.write(fd, view[pos:])
                require(wrote > 0, "short owned write")
                pos += wrote
        finally:
            view.release()
        buffer.clear()

    for fragment in canonical_fragments(value):
        require(ceiling is None or count+len(fragment) <= ceiling, "stream encoded cap")
        if len(buffer)+len(fragment) > 65536:
            flush()
        buffer.extend(fragment)
        hasher.update(fragment)
        count += len(fragment)
    require(ceiling is None or count+1 <= ceiling, "stream newline cap")
    if len(buffer) == 65536:
        flush()
    buffer.extend(b"\n")
    hasher.update(b"\n")
    count += 1
    flush()
    return {"size": count, "sha256": hasher.hexdigest()}


def bounded_detail(value):
    require(type(value) is str, "base string detail")
    suffix = " [source text truncated]"
    suffix_cost = sum(len(_char_fragment(c)) for c in suffix)
    prefix, count, cost = [], 0, 2
    for c in value:
        fragment_cost = len(_char_fragment(c))
        if cost+fragment_cost > 4096:
            break
        prefix.append(c)
        count += 1
        cost += fragment_cost
    truncated = count != len(value)
    if not truncated:
        result = value
    else:
        while cost+suffix_cost > 4096:
            removed = prefix.pop()
            count -= 1
            cost -= len(_char_fragment(removed))
        result = "".join(prefix)+suffix
    return {
        "text": result, "original_characters": len(value),
        "retained_source_characters": count, "omitted_source_characters": len(value)-count,
        "truncated": truncated,
    }


_TYPE_SLOTS = {name: type.__dict__[name] for name in
               ("__dict__", "__mro__", "__name__", "__module__", "__qualname__")}
_MODULE_DICT = types.ModuleType.__dict__["__dict__"]
_ERROR_ARGS = BaseException.__dict__["args"]
_ERROR_DICT = BaseException.__dict__["__dict__"]
_SPEC_DICT = importlib.machinery.ModuleSpec.__dict__["__dict__"]
_NAMESPACE_PATH = importlib._bootstrap_external._NamespacePath
_NAMESPACE_DICT = _NAMESPACE_PATH.__dict__["__dict__"]


def safe_type(value):
    try:
        cls = type(value)
        module = _TYPE_SLOTS["__module__"].__get__(cls)
        name = _TYPE_SLOTS["__qualname__"].__get__(cls)
        require(type(module) is str and type(name) is str, "type metadata")
        candidate = module+"."+name
        require(re.fullmatch(r"[A-Za-z0-9_.<>-]{1,512}", candidate) is not None, "type domain")
        return candidate
    except BaseException:
        return None


def _reason_record(error):
    reason = {
        "kind": "type_only", "fragments": [], "source_argument_count": None,
        "omitted_argument_count": None, "errno": None, "unavailable_argument_count": None,
    }
    try:
        args = _ERROR_ARGS.__get__(error)
        if type(args) is not tuple:
            reason["kind"] = "unavailable"
            return reason, 1
        reason["source_argument_count"] = len(args)
        reason["omitted_argument_count"] = max(0, len(args)-8)
        unknown = 0
        for item in args[:8]:
            if type(item) is str:
                reason["fragments"].append(bounded_detail(item))
            else:
                unknown += 1
        reason["unavailable_argument_count"] = unknown
        if reason["fragments"]:
            reason["kind"] = "known_string_args"
        if _base_kind(error, OSError):
            value = OSError.errno.__get__(error)
            if type(value) is int and -2**63 <= value < 2**63:
                reason["errno"] = value
                reason["kind"] = "known_errno"
    except BaseException:
        reason["kind"] = "unavailable"
        return reason, 1
    return reason, 0


def reason_record(error):
    return _reason_record(error)[0]


def error_record(error, context):
    reason, failures = _reason_record(error)
    kind = safe_type(error)
    try:
        context_text = bounded_detail(context)["text"]
    except BaseException:
        context_text = "error context formatting unavailable"
        failures += 1
    result = {
        "type": kind, "context": context_text,
        "reason": reason, "notes": [], "source_note_count": None,
        "omitted_note_count": None, "type_unavailable": kind is None,
        "notes_unavailable": False, "formatting_failure_count": failures+int(kind is None),
    }
    try:
        notes = _ERROR_DICT.__get__(error).get("__notes__", [])
        if type(notes) not in (list, tuple):
            result["notes_unavailable"] = True
            return result
        result["source_note_count"] = len(notes)
        result["omitted_note_count"] = max(0, len(notes)-64)
        for note in notes[:64]:
            if type(note) is str:
                detail = bounded_detail(note)
                kind = ("returned_solver" if note.startswith("Bounded returned OT interface evidence:")
                        else "warning_context" if note.startswith("Captured retrieval warnings:")
                        else "other_string")
                result["notes"].append({"kind": kind, **detail, "unavailable": False})
            else:
                result["notes"].append({
                    "kind": "unsupported", "text": "unsupported note [detail unavailable]",
                    "original_characters": None, "retained_source_characters": None,
                    "omitted_source_characters": None, "truncated": False, "unavailable": True,
                })
    except BaseException:
        result["notes_unavailable"] = True
        result["formatting_failure_count"] += 1
        if result["source_note_count"] is not None:
            result["omitted_note_count"] = result["source_note_count"]-len(result["notes"])
    return result


def add_secondary(primary, additional, context):
    held = _SECONDARIES.setdefault(id(primary), [primary, [], 0])
    try:
        record = error_record(additional, context)
    except BaseException:
        kind = safe_type(additional)
        record = {
            "type": kind, "context": "secondary error formatting unavailable",
            "reason": {"kind": "unavailable", "fragments": [], "source_argument_count": None,
                       "omitted_argument_count": None, "errno": None, "unavailable_argument_count": None},
            "notes": [], "source_note_count": None, "omitted_note_count": None,
            "type_unavailable": kind is None, "notes_unavailable": True, "formatting_failure_count": 1,
        }
    if len(held[1]) < 64:
        held[1].append(record)
    else:
        held[2] += 1


def secondary_records(primary):
    held = _SECONDARIES.get(id(primary))
    return ([], 0) if held is None else (list(held[1]), held[2])


def validate_detail(value, nullable=False):
    required = ("text", "original_characters", "retained_source_characters", "omitted_source_characters", "truncated")
    require(all(k in value for k in required), "source text detail keys")
    require(type(value["text"]) is str and 2+sum(len(_char_fragment(c)) for c in value["text"]) <= 4096,
            "escaped diagnostic string4096")
    boolean(value["truncated"])
    if nullable and value["original_characters"] is None:
        require(value["retained_source_characters"] is value["omitted_source_characters"] is None, "unavailable text counts")
        return
    for key in required[1:4]:
        integer(value[key])
    require(value["original_characters"] == value["retained_source_characters"]+value["omitted_source_characters"]
            and value["truncated"] == (value["omitted_source_characters"] > 0), "exact source omission accounting")


def validate_error(value):
    keys(value, ("type", "context", "reason", "notes", "source_note_count", "omitted_note_count",
                 "type_unavailable", "notes_unavailable", "formatting_failure_count"), "Error")
    if value["type"] is not None:
        require(type(value["type"]) is str and re.fullmatch("[A-Za-z0-9_.<>-]{1,512}", value["type"]) is not None,
                "safe qualified error type")
    require(value["type_unavailable"] is (value["type"] is None), "error type availability")
    require(type(value["context"]) is str and len(encoded(value["context"], 8192))-1 <= 4096, "fixed escaped context")
    boolean(value["notes_unavailable"])
    integer(value["formatting_failure_count"])
    reason = value["reason"]
    keys(reason, ("kind", "fragments", "source_argument_count", "omitted_argument_count", "errno",
                  "unavailable_argument_count"), "Reason")
    require(reason["kind"] in ("known_string_args", "known_errno", "type_only", "unavailable"), "reason domain")
    for key in ("source_argument_count", "omitted_argument_count", "unavailable_argument_count"):
        if reason[key] is not None:
            integer(reason[key])
    if reason["errno"] is not None:
        integer(reason["errno"], -2**63, 2**63-1)
    for fragment in array(reason["fragments"], 8):
        keys(fragment, ("text", "original_characters", "retained_source_characters",
                        "omitted_source_characters", "truncated"), "TextDetail")
        validate_detail(fragment)
    for note in array(value["notes"], 64):
        keys(note, ("kind", "text", "original_characters", "retained_source_characters",
                    "omitted_source_characters", "truncated", "unavailable"), "Note")
        require(note["kind"] in ("returned_solver", "warning_context", "other_string", "unsupported"), "note domain")
        boolean(note["unavailable"])
        validate_detail(note, note["unavailable"])
    for key in ("source_note_count", "omitted_note_count"):
        if value[key] is not None:
            integer(value[key])
    if value["source_note_count"] is not None and value["omitted_note_count"] is not None:
        require(value["source_note_count"] == len(value["notes"])+value["omitted_note_count"], "note omission accounting")


def identity(observed):
    return observed.st_dev, observed.st_ino, observed.st_size


def close_owned(fd, primary=None):
    try:
        os.close(fd)
    except BaseException as extra:
        if primary is None:
            raise
        add_secondary(primary, extra, "owned descriptor close")


def validate_desc(value, cap=2*1024**3):
    keys(value, ("path", "size", "sha256"), "descriptor")
    path_text(value["path"])
    integer(value["size"], maximum=cap)
    digest(value["sha256"])
    return value


def validate_bind(value):
    keys(value, ("path", "size", "sha256", "st_dev", "st_ino"), "binding")
    validate_desc({k: value[k] for k in ("path", "size", "sha256")})
    integer(value["st_dev"])
    integer(value["st_ino"], 1)
    return value


def validate_dir(value):
    keys(value, ("path", "st_dev", "st_ino", "uid", "mode"), "directory")
    path_text(value["path"])
    integer(value["st_dev"])
    integer(value["st_ino"], 1)
    integer(value["uid"])
    require(type(value["mode"]) is int and value["mode"] == 0o700, "directory owner-only mode")
    return value


def validate_file(value, name=None, cap=None):
    keys(value, ("name", "size", "sha256", "st_dev", "st_ino", "uid", "mode"), "file")
    leaf(value["name"])
    require(name is None or value["name"] == name, "file registry leaf")
    integer(value["size"], maximum=2**63-1 if cap is None else cap)
    digest(value["sha256"])
    integer(value["st_dev"])
    integer(value["st_ino"], 1)
    integer(value["uid"])
    require(type(value["mode"]) is int and value["mode"] in (0o600, 0o700), "file owner-only mode")
    return value


def validate_envelope(value, envelope=None):
    for name in ("schema_version", "task", "run_id", "attempt_nonce", "spec_sha256", "private_sha256"):
        require(name in value, "missing envelope field")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1
            and value["task"] == TASK, "envelope schema/task")
    identifier(value["run_id"], run=True)
    require(type(value["attempt_nonce"]) is str and
            re.fullmatch("[0-9a-f]{32}", value["attempt_nonce"]) is not None, "attempt nonce")
    digest(value["spec_sha256"])
    digest(value["private_sha256"])
    if envelope is not None:
        require(all(type(value[k]) is type(v) and value[k] == v for k, v in envelope.items()),
                "envelope linkage")


def envelope_of(value):
    return {k: value[k] for k in (
        "schema_version", "task", "run_id", "attempt_nonce", "spec_sha256", "private_sha256")}


def clock(start, deadline):
    now = time.monotonic_ns()
    return {
        "started_utc_ns": start[0], "ended_utc_ns": time.time_ns(),
        "started_monotonic_ns": start[1], "ended_monotonic_ns": now,
        "deadline_monotonic_ns": deadline, "deadline_met": now <= deadline,
    }


def validate_clock(value, success=False):
    keys(value, ("started_utc_ns", "ended_utc_ns", "started_monotonic_ns",
                 "ended_monotonic_ns", "deadline_monotonic_ns", "deadline_met"), "clock")
    for k, v in value.items():
        if k != "deadline_met":
            integer(v)
    boolean(value["deadline_met"])
    require(value["started_monotonic_ns"] <= value["ended_monotonic_ns"]
            and value["deadline_met"] == (value["ended_monotonic_ns"] <= value["deadline_monotonic_ns"]),
            "truthful clock")
    require(not success or value["deadline_met"], "late success")


class Inventory:
    """Observed-only serial streamed files and their exact lexical aliases."""

    def __init__(self, deadline, startup_pth=()):
        self.deadline = deadline
        self.startup_pth = frozenset(startup_pth)
        self.records, self.aliases, self.phases, self.alias_phases = {}, {}, {}, {}
        self.current_phase = "bootstrap"
        self.total = 0

    def check_deadline(self):
        require(time.monotonic_ns() <= self.deadline, "input custody deadline")

    def read(self, path, *, expected=None, cap=None, retain=False):
        self.check_deadline()
        path = path_text(str(path))
        require(CALLER not in path, "changing caller acquisition forbidden")
        require(Path(path).suffix.lower() not in (".pkl", ".npy", ".npz", ".pt", ".safetensors",
                                                  ".png", ".jpg", ".jpeg", ".svs", ".ndpi"),
                "payload is not software/control")
        require(not path.endswith(".pth") or path in self.startup_pth, "non-startup .pth forbidden")
        resolved = str(Path(path).resolve(strict=True))
        require(CALLER not in resolved, "caller resolved alias forbidden")
        require(Path(resolved).suffix.lower() not in (".pkl", ".pickle", ".npy", ".npz", ".pt",
                  ".safetensors", ".png", ".jpg", ".jpeg", ".svs", ".ndpi"),
                "resolved payload is not software/control")
        require(not resolved.endswith(".pth") or path in self.startup_pth, "resolved non-startup pth forbidden")
        cap = LIMITS["individual_software_bytes"] if cap is None else cap
        before = os.stat(path)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, "software regular/stat cap")
        if expected is not None:
            require(before.st_size == expected["size"], "declared source size")
        if resolved not in self.records:
            require(len(self.records) < LIMITS["persistent_canonical_files"]
                    and self.total+before.st_size <= LIMITS["aggregate_unique_software_bytes"],
                    "unique software caps")
        if path not in self.aliases:
            require(len(self.aliases) < LIMITS["lexical_alias_paths"], "lexical software cap")
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        primary = None
        try:
            require(identity(os.fstat(fd)) == identity(before), "software open handoff")
            hasher, count, chunks = hashlib.sha256(), 0, [] if retain else None
            while True:
                self.check_deadline()
                chunk = os.read(fd, min(MIB, before.st_size-count+1))
                if not chunk:
                    break
                count += len(chunk)
                require(count <= cap and count <= before.st_size, "software read growth")
                hasher.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            after = os.stat(path)
            require(count == before.st_size and identity(os.fstat(fd)) == identity(before)
                    == identity(after) and str(Path(path).resolve(strict=True)) == resolved,
                    "software content/path custody")
            record = {"path": resolved, "size": count, "sha256": hasher.hexdigest(),
                      "st_dev": after.st_dev, "st_ino": after.st_ino}
            if expected is not None:
                require(record["sha256"] == expected["sha256"], "declared software SHA")
            if resolved in self.records:
                require(record == self.records[resolved], "same-run software changed")
            else:
                self.total += count
                self.records[resolved] = record
            require(path not in self.aliases or self.aliases[path] == record, "alias substitution")
            self.aliases[path] = record
            self.phases.setdefault(resolved, set()).add(self.current_phase)
            self.alias_phases.setdefault(path, set()).add(self.current_phase)
            self.check_deadline()
            return (b"".join(chunks) if chunks is not None else None), record
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close_owned(fd, primary)

    def sweep(self):
        for path, expected in tuple(self.aliases.items()):
            self.read(path, expected=expected)


class OwnedDirectory:
    """Fixed local namespace custody; no submission or terminal certification."""

    def __init__(self, path, uid, deadline, *, create=False, expected=None):
        self.path = Path(path_text(str(path)))
        self.uid, self.deadline, self.files = uid, deadline, {}
        self.completed_files, self.failure_records = {}, {}
        self.fd = self.parent_fd = None
        parent = self.path.parent
        self.parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            self.parent_identity = identity(os.fstat(self.parent_fd))[:2]
            if create:
                os.mkdir(self.path.name, 0o700, dir_fd=self.parent_fd)
                os.fsync(self.parent_fd)
            observed = self.path.lstat()
            require(stat.S_ISDIR(observed.st_mode) and observed.st_uid == uid
                    and stat.S_IMODE(observed.st_mode) == 0o700, "owned directory/mode")
            self.fd = os.open(self.path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=self.parent_fd)
            self.record = {"path": str(self.path), "st_dev": observed.st_dev,
                           "st_ino": observed.st_ino, "uid": uid, "mode": 0o700}
            self.check()
            require(expected is None or expected == self.record, "directory identity linkage")
            if create:
                require(not os.listdir(self.fd), "exclusive empty directory")
        except BaseException as exc:
            if self.fd is not None:
                close_owned(self.fd, exc)
            close_owned(self.parent_fd, exc)
            raise

    def check(self):
        require(time.monotonic_ns() <= self.deadline, "namespace deadline")
        opened, lexical, parent = os.fstat(self.fd), self.path.lstat(), self.path.parent.lstat()
        require(stat.S_ISDIR(opened.st_mode) and stat.S_ISDIR(lexical.st_mode)
                and (opened.st_dev, opened.st_ino) == (lexical.st_dev, lexical.st_ino)
                == (self.record["st_dev"], self.record["st_ino"])
                and (parent.st_dev, parent.st_ino) == self.parent_identity
                and self.path.resolve(strict=True) == self.path, "namespace custody")

    def names(self):
        self.check()
        return set(os.listdir(self.fd))

    def sync(self):
        self.check()
        os.fsync(self.fd)
        self.check()
        os.fsync(self.parent_fd)
        self.check()

    def write(self, name, value, *, raw=False, ceiling=8*MIB):
        leaf(name)
        if name == "FAILED.json":
            require(not raw and ceiling is None, "uncapped structured FAILED writer")
            validate_failure(value)
            require(name not in self.failure_records, "preserve prior FAILED attempt record")
            self.failure_records[name] = value
        self.check()
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=self.fd)
        primary = None
        try:
            initial = os.fstat(fd)
            self.files[name] = {"name": name, "size": 0, "sha256": sha(b""),
                                "st_dev": initial.st_dev, "st_ino": initial.st_ino,
                                "uid": initial.st_uid, "mode": stat.S_IMODE(initial.st_mode)}
            if raw:
                require(type(value) is bytes and (ceiling is None or len(value) <= ceiling),
                        "owned raw byte cap")
                view, offset = memoryview(value), 0
                try:
                    while offset < len(view):
                        self.check()
                        wrote = os.write(fd, view[offset:offset+65536])
                        require(wrote > 0, "owned raw write")
                        offset += wrote
                finally:
                    view.release()
                binding = {"size": len(value), "sha256": sha(value)}
            else:
                binding = stream_json(fd, value, ceiling)
            self.check()
            os.fsync(fd)
            final = os.fstat(fd)
            require(final.st_size == binding["size"], "owned writer size")
            self.files[name].update(binding)
            validate_file(self.files[name], name, ceiling)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close_owned(fd, primary)
        self.sync()
        if name == "FAILED.json":
            self.read_failure(self.files[name], value)
        else:
            self.read(name, self.files[name], ceiling=ceiling)
        self.check()
        self.completed_files[name] = dict(self.files[name])
        return self.files[name]

    def read_failure(self, record, value):
        validate_file(record, "FAILED.json")
        validate_failure(value)
        self.check()
        fd = os.open("FAILED.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
        primary = None
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and identity(before) ==
                    (record["st_dev"], record["st_ino"], record["size"])
                    and before.st_uid == record["uid"]
                    and stat.S_IMODE(before.st_mode) == record["mode"], "original FAILED identity")
            count, hasher = 0, hashlib.sha256()
            for expected in failure_chunks(value):
                actual = bytearray()
                while len(actual) < len(expected):
                    self.check()
                    chunk = os.read(fd, len(expected)-len(actual))
                    require(bool(chunk), "original FAILED short read")
                    actual.extend(chunk)
                require(actual == expected, "original FAILED canonical fragment equality")
                count += len(actual)
                hasher.update(actual)
            require(os.read(fd, 1) == b"", "original FAILED trailing bytes")
            after = os.stat("FAILED.json", dir_fd=self.fd, follow_symlinks=False)
            require(count == record["size"] and hasher.hexdigest() == record["sha256"]
                    and identity(os.fstat(fd)) == identity(before) == identity(after),
                    "original FAILED hash/extent/custody")
            self.check()
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close_owned(fd, primary)
        self.check()
        return record

    def read(self, name, record, *, ceiling=8*MIB):
        validate_file(record, name, ceiling)
        self.check()
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
        primary = None
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode)
                    and identity(before) == (record["st_dev"], record["st_ino"], record["size"])
                    and before.st_uid == record["uid"]
                    and stat.S_IMODE(before.st_mode) == record["mode"], "owned read identity")
            chunks, count, hasher = [], 0, hashlib.sha256()
            while True:
                self.check()
                chunk = os.read(fd, min(65536, record["size"]-count+1))
                if not chunk:
                    break
                count += len(chunk)
                require(count <= record["size"], "owned read growth")
                chunks.append(chunk)
                hasher.update(chunk)
            after = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            require(count == record["size"] and hasher.hexdigest() == record["sha256"]
                    and identity(os.fstat(fd)) == identity(before) == identity(after),
                    "owned original-byte custody")
            self.check()
            return b"".join(chunks)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close_owned(fd, primary)

    def release(self, primary=None):
        try:
            self.sync()
        except BaseException as extra:
            if primary is None:
                primary = extra
            else:
                add_secondary(primary, extra, "directory final sync")
        for fd in (self.fd, self.parent_fd):
            try:
                os.close(fd)
            except BaseException as extra:
                if primary is None:
                    primary = extra
                else:
                    add_secondary(primary, extra, "directory release")
        self.fd = self.parent_fd = None
        if primary is not None:
            raise primary


def software_id(path):
    return "sw:"+sha(path_text(path).encode("utf-8"))


def runtime_id(path):
    return "runtime:impl:"+sha(path_text(path).encode("utf-8"))


def runtime_bridge(binding, phases):
    validate_bind(binding)
    return {"software_asset_id": software_id(binding["path"]),
            "runtime_asset_id": runtime_id(binding["path"]), "canonical_path": binding["path"],
            "binding": binding, "final_binding": dict(binding), "phases": list(phases)}


def returned_export_context(note):
    prefix = "Bounded returned OT interface evidence: "
    if type(note) is not str or not note.startswith(prefix):
        return False
    # Only the source's leading context object, never warning/detail text.
    payload = note[len(prefix):]
    if payload.startswith("formatting unavailable; "):
        match = re.fullmatch(
            r"formatting unavailable; request_id=[A-Za-z0-9_.:-]{1,64}; "
            r"result_code_type=[A-Za-z0-9_.<>-]{1,128}; warning_type=[A-Za-z0-9_.<>-]{1,128}; "
            r"solver_result_code_export=not exported; numeric cell=null; sign=(?:positive|negative); "
            r"bit_length=([0-9]{1,19}); diagnostic domain=256 bits; "
            r"formatting_error_type=[A-Za-z0-9_.<>-]{1,128}", payload)
        return match is not None and int(match[1]) > 256
    quoted = escaped = False
    depth = 0
    end = None
    for index, character in enumerate(payload):
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "{[":
            depth += 1
            require(depth <= 32, "returned context depth")
        elif character in "}]":
            depth -= 1
            if depth == 0:
                end = index+1
                break
    if end is None:
        return False
    try:
        context = strict_json(payload[:end].encode("ascii"), 4096)
    except (ValueError, UnicodeError, ProbeError):
        return False
    if type(context) is not dict or set(context) != {
            "request_id", "result_code_type", "warning_type", "solver_result_code_export"}:
        return False
    marker = context["solver_result_code_export"]
    if type(marker) is not str:
        return False
    match = re.fullmatch(
        r"not exported; numeric cell=null; sign=(?:positive|negative); "
        r"bit_length=([0-9]{1,19}); diagnostic domain=256 bits", marker)
    return match is not None and int(match[1]) > 256


def partial_observation(observation, error, request_id):
    require(type(observation) is dict and all(type(k) is str for k in observation),
            "partial observation exact keys")
    numeric = observation.get("solver_result_code")
    exported = type(numeric) is int and numeric.bit_length() <= 256
    require(numeric is None or exported, "stock exported numeric domain")
    notes = error_record(error, "native computation failure")
    evidence = next((v["text"] for v in notes["notes"]
                     if v["kind"] == "returned_solver"), None)
    reason = notes["reason"]
    args = _ERROR_ARGS.__get__(error)
    boundary = args[0] if len(args) == 1 and type(args[0]) is str else None
    tags = observation.get("nonfinite_cost_tags")
    if tags is not None:
        array(tags, 3)
        require(tags == sorted(set(tags)) and all(v in (
            "nan", "negative_infinity", "positive_infinity") for v in tags), "real cost tags")
    state = "numeric" if exported else "unavailable"
    unsupported = boundary == "nonintegral/Boolean solver status"
    if not exported and evidence is not None:
        state = ("outside_256_bits" if returned_export_context(evidence) else
                 "unsupported_type" if unsupported
                 else "capture_export_not_reached")
    elif not exported and unsupported:
        state = "unsupported_type"
    rejected = boundary == "unsuccessful solver status/warning"
    return {
        "request_id": identifier(request_id),
        "solver_result_code": numeric if exported else None,
        "numeric_origin": "original_observation" if exported else "unavailable",
        "raw_status_evidence": "original_cell" if exported else
        "returned_note" if evidence is not None else "unavailable",
        "export_state": state,
        "solver_success_validation": "rejected" if rejected else
        "accepted" if observation.get("computed_valid") is True and
        observation.get("solver_result_code") == 1 else "not_established",
        "solver_raw_evidence": evidence, "nonfinite_cost_tags": tags,
        "reason": reason, "notes": notes["notes"], "omitted_note_count": notes["omitted_note_count"],
    }


RESERVATION_STAGES = ("query_event", "raw_query", "raw_reference", "prepare_query",
                      "prepare_reference", "baseline_load", "final_cleanup")


def validate_reservations(value, success=False):
    seen, operations = set(), set()
    for event in array(value, 64):
        keys(event, ("operation", "stage", "before_live", "after_live", "before_retained",
                     "after_retained", "new_reservations", "accounting_consistent",
                     "cleanup_complete", "error"), "ReservationEvent")
        identifier(event["operation"])
        require(event["operation"] not in operations and event["stage"] in RESERVATION_STAGES,
                "reservation operation/stage")
        operations.add(event["operation"])
        for name in ("before_live", "after_live"):
            integer(event[name])
        for name in ("before_retained", "after_retained"):
            integer(event[name], maximum=128)
        boolean(event["accounting_consistent"])
        boolean(event["cleanup_complete"])
        if event["error"] is not None:
            validate_error(event["error"])
        for row in array(event["new_reservations"], 128):
            keys(row, ("reservation_id", "size", "disposition"), "Reservation")
            identifier(row["reservation_id"])
            require(row["reservation_id"] not in seen and row["disposition"] in
                    ("retained", "released", "release_failed"), "reservation identity/disposition")
            seen.add(row["reservation_id"])
            integer(row["size"])
            if success:
                require(row["disposition"] != "release_failed", "successful reservation release evidence")
        if success:
            require(event["accounting_consistent"] and event["cleanup_complete"]
                    and event["error"] is None, "successful reservation evidence")
    if success:
        for index, event in enumerate(value):
            if event["stage"] == "final_cleanup":
                require(index == len(value)-1 and event["after_live"] == event["after_retained"] == 0
                        and all(row["disposition"] == "released" for prior in value
                                for row in prior["new_reservations"]), "successful final reservation accounting")


class ReservationLedger:
    def __init__(self, memory):
        self.memory, self.events, self.owned = memory, [], {}
        self.token_events = {}
        self.serial = 0
        self.cleanup_started = False

    @contextmanager
    def operation(self, operation, stage):
        require(stage in RESERVATION_STAGES and len(self.events) < 64, "reservation stage/event cap")
        require(len(self.memory.retained) <= 128, "reservation map cap")
        before = dict(self.memory.retained)
        live = self.memory.live
        peak = integer(self.memory.peak, live)
        primary = None
        event = {"operation": f"operation:{len(self.events):08d}", "stage": stage, "before_live": live,
                 "after_live": live, "before_retained": len(before), "after_retained": len(before),
                 "new_reservations": [], "accounting_consistent": True,
                 "cleanup_complete": True, "error": None}
        try:
            yield
        except BaseException as exc:
            primary = exc
            event["error"] = error_record(exc, "owned reservation operation")
            raise
        finally:
            new = [k for k in self.memory.retained if k not in before]
            try:
                require(all(self.memory.retained.get(k) == v for k, v in before.items()),
                        "preexisting reservation changed")
                for token in new:
                    self.serial += 1
                    row = {"reservation_id": f"reservation:{self.serial:08d}",
                           "size": self.memory.retained[token], "disposition": "retained"}
                    event["new_reservations"].append(row)
                    self.owned[token] = row
                    self.token_events[token] = event
                    if primary is not None:
                        self.release(token, primary)
                require(len(self.memory.retained) <= 128, "after reservation map cap")
                require(self.memory.live == sum(self.memory.retained.values()), "working accounting")
                require(self.memory.peak >= max(peak, self.memory.live), "working peak monotonic accounting")
            except BaseException as extra:
                event["accounting_consistent"] = event["cleanup_complete"] = False
                event["error"] = error_record(extra, "reservation accounting")
                if primary is None:
                    raise
                add_secondary(primary, extra, "reservation accounting")
            finally:
                event.update(after_live=self.memory.live, after_retained=len(self.memory.retained))
                self.events.append(event)
                require(len(self.events) <= 64, "reservation event cap")

    def release(self, token, primary=None):
        try:
            require(token in self.owned, "release only harness-owned token")
            self.memory.release(token)
            self.owned[token]["disposition"] = "released"
            del self.owned[token]
            del self.token_events[token]
        except BaseException as extra:
            if token in self.owned:
                self.owned[token]["disposition"] = "release_failed"
                event = self.token_events[token]
                event["cleanup_complete"] = False
                event["error"] = error_record(extra, "reservation release")
            if primary is None:
                raise
            add_secondary(primary, extra, "reservation release")

    def cleanup(self, primary=None):
        self.cleanup_started = True
        event = {"operation": f"operation:{len(self.events):08d}", "stage": "final_cleanup",
                 "before_live": self.memory.live, "after_live": self.memory.live,
                 "before_retained": len(self.memory.retained), "after_retained": len(self.memory.retained),
                 "new_reservations": [], "accounting_consistent": True,
                 "cleanup_complete": True, "error": None}
        cleanup_error = None
        for token in tuple(self.owned):
            try:
                self.release(token)
            except BaseException as extra:
                if cleanup_error is None:
                    cleanup_error = extra
                if primary is None:
                    primary = extra
                else:
                    add_secondary(primary, extra, "reservation cleanup")
        event.update(after_live=self.memory.live, after_retained=len(self.memory.retained),
                     accounting_consistent=self.memory.live == sum(self.memory.retained.values()),
                     cleanup_complete=not self.owned and cleanup_error is None,
                     error=None if cleanup_error is None else error_record(cleanup_error, "final cleanup"))
        self.events.append(event)
        require(len(self.events) <= 64, "reservation event cap")
        if primary is not None:
            raise primary


def cleanup_exercise(state, ledger, primary=None):
    try:
        ledger.cleanup(primary)
        require(state.memory.live == sum(state.memory.retained.values()) == 0,
                "remaining reservations truthful")
    except BaseException as extra:
        if primary is None:
            primary = extra
        elif extra is not primary:
            add_secondary(primary, extra, "working reservations final cleanup")
    finally:
        state.query_prepared.clear()
        state.query_raw.clear()
    if primary is not None:
        try:
            namespace = _ERROR_DICT.__get__(primary)
            notes = namespace.setdefault("__notes__", [])
            require(type(notes) is list, "owned failure note base container")
            notes.append("WorkingMemory ownership after cleanup: "+encoded({
                "live": state.memory.live, "retained_sum": sum(state.memory.retained.values()),
                "retained_count": len(state.memory.retained), "harness_owned_count": len(ledger.owned),
                "physical_rss_release_not_established": True,
            }, 4096).decode("ascii").rstrip("\n"))
        except BaseException as extra:
            add_secondary(primary, extra, "working ownership note formatting")
        raise primary


def serialize_exercises(value, state, ledger):
    # Publish the actual pre-cleanup snapshot, not a second serialization after cache release.
    value["reservations"] = [
        {**event, "new_reservations": [dict(row) for row in event["new_reservations"]]}
        for event in ledger.events]
    validate_reservations(value["reservations"], True)
    literal = encoded(value, 8*MIB)
    cleanup_exercise(state, ledger)
    return literal


def validate_success_accounting(value):
    events = value["reservations"]
    expected_live = expected_count = 0
    peak = value["working_memory"]["peak_accounted_working_bytes"]
    reservation_ordinal = 0
    for index in range(8):
        query, raw, prepared = events[index*3:index*3+3]
        require(query["before_live"] == expected_live and query["before_retained"] == expected_count
                and len(query["new_reservations"]) == (2 if index == 0 else 1),
                "persistent query reservation boundary")
        for prior, event in ((None, query), (query, raw), (raw, prepared)):
            for row in event["new_reservations"]:
                reservation_ordinal += 1
                require(row["reservation_id"] == f"reservation:{reservation_ordinal:08d}",
                        "successful reservation ordinal")
            require(event["operation"] == f"operation:{index*3+(0 if prior is None else 1 if prior is query else 2):08d}",
                    "successful operation ordinal")
            require(event["after_live"] == event["before_live"] +
                    sum(row["size"] for row in event["new_reservations"])
                    and event["after_retained"] == event["before_retained"] +
                    len(event["new_reservations"]), "successful operation-exit deltas")
            require(peak >= max(event["before_live"], event["after_live"]), "peak covers operation boundaries")
            if prior is not None:
                require(event["before_live"] == prior["after_live"]
                        and event["before_retained"] == prior["after_retained"]
                        and len(event["new_reservations"]) == 1, "reference operation continuity")
            if event is query:
                require(all(row["disposition"] == "retained" for row in event["new_reservations"]),
                        "query cache owned through serialization")
        expected_live, expected_count = query["after_live"], query["after_retained"]
    event, baseline = events[-1], value["baseline"]
    require(event["operation"] == "operation:00000024" and len(event["new_reservations"]) == 1
            and event["new_reservations"][0]["reservation_id"] == f"reservation:{reservation_ordinal+1:08d}"
            and event["new_reservations"][0]["size"] == baseline["reservation_bytes"] ==
            baseline["descriptor"]["size"]+4192
            and event["before_live"] == baseline["working_live_before"] == expected_live
            and event["before_retained"] == expected_count == 9
            and event["after_live"] == expected_live+baseline["reservation_bytes"]
            and event["after_retained"] == expected_count+1
            and baseline["working_live_after"] == expected_live, "baseline hold versus later release")
    require(peak >= max(event["after_live"], baseline["working_live_after"]), "peak covers baseline")


def copy_ledger_error(value):
    source_ledger_records(value)
    validate_error(value)
    return {**value, "reason": {**value["reason"], "fragments": [dict(v) for v in value["reason"]["fragments"]]},
            "notes": [dict(v) for v in value["notes"]]}


def copy_ledger_events(value):
    require(type(value) is list, "source-safe ledger event list")
    require(len(value) <= 64, "source event bound before traversal")
    source_ledger_records(value)
    validate_reservations(value)
    return [{**event, "new_reservations": [dict(row) for row in event["new_reservations"]],
             "error": None if event["error"] is None else copy_ledger_error(event["error"])}
            for event in value]


def local_base_namespace(value):
    # Only actual base storage is observed; a Python property is not a storage slot.
    for cls in _TYPE_SLOTS["__mro__"].__get__(type(value)):
        descriptor = _TYPE_SLOTS["__dict__"].__get__(cls).get("__dict__")
        if type(descriptor) is types.GetSetDescriptorType:
            result = descriptor.__get__(value)
            require(type(result) is dict, "local base storage")
            return result
    raise ProbeError("local base storage unavailable")


def source_ledger_records(value, depth=0):
    require(depth <= 32, "source ledger record depth")
    kind = type(value)
    if kind is dict:
        require(len(value) <= 64, "source ledger record keys")
        for key, item in dict.items(value):
            require(type(key) is str, "source ledger base key")
            source_ledger_records(item, depth+1)
    elif kind is list:
        require(len(value) <= 128, "source ledger list bound before traversal")
        for item in value:
            source_ledger_records(item, depth+1)
    else:
        require(value is None or kind in (str, int, bool), "source ledger base scalar")


_LEDGER_FACTS = (
    "cleanup_attempted", "events", "event_count", "owned_reservations", "live", "peak", "limit_bytes",
    "retained_count", "retained_sum", "owned_count", "unregistered_retained_count",
    "owned_not_retained_count", "live_matches_sum", "registry_matches_retained",
    "registered_sizes_match", "cleanup_complete",
)


def ledger_cleanup_complete(value):
    event = None if value["events"] is None else next(
        (v for v in reversed(value["events"]) if v["stage"] == "final_cleanup"), None)
    needed = ("cleanup_attempted", "live", "owned_count", "retained_count", "retained_sum",
              "live_matches_sum", "registry_matches_retained", "registered_sizes_match")
    if any(value[k] is None for k in needed) or value["events"] is None:
        return None
    return (value["cleanup_attempted"] is True and event is not None
            and event["cleanup_complete"] is True and event["accounting_consistent"] is True
            and event["error"] is None and value["live"] == value["owned_count"] ==
            value["retained_count"] == value["retained_sum"] == 0
            and value["live_matches_sum"] and value["registry_matches_retained"]
            and value["registered_sizes_match"])


def failure_ledger_snapshot(ledger, stage, ordinal, primary):
    if ledger is None:
        return None
    require(stage in FAILURE_STAGES, "failure snapshot closed stage")
    value = {k: None for k in _LEDGER_FACTS}
    value.update(state="unavailable", epoch={
        "kind": "after_harness_unwind_before_failure_write", "observation_ordinal": ordinal,
        "producer_pid": os.getpid(), "observed_utc_ns": time.time_ns(),
        "observed_monotonic_ns": time.monotonic_ns(), "failure_stage": stage,
    }, snapshot_errors=[], omitted_snapshot_error_count=0, physical_rss_release_established=False)

    def error(extra, context):
        previous_count = len(secondary_records(primary)[0])
        add_secondary(primary, extra, context)
        if len(value["snapshot_errors"]) < 8:
            retained, _ = secondary_records(primary)
            if len(retained) > previous_count:
                value["snapshot_errors"].append(copy_ledger_error(retained[-1]))
            else:
                try:
                    value["snapshot_errors"].append(error_record(extra, context))
                except BaseException as formatting:
                    add_secondary(primary, formatting, "snapshot Error formatting")
                    value["omitted_snapshot_error_count"] += 1
        else:
            value["omitted_snapshot_error_count"] += 1

    try:
        require(type(ledger) is ReservationLedger, "local harness ledger representation")
        source = local_base_namespace(ledger)
    except BaseException as extra:
        error(extra, "local ledger acquisition")
        validate_failure_ledger(value, stage)
        return value
    try:
        flag = source["cleanup_started"]
        boolean(flag)
        value["cleanup_attempted"] = flag
    except BaseException as extra:
        error(extra, "ledger cleanup flag acquisition")
    try:
        events = source["events"]
        require(type(events) is list, "source-safe events")
        value["event_count"] = integer(len(events))
        value["events"] = copy_ledger_events(events)
    except BaseException as extra:
        error(extra, "ledger event acquisition")
    owned = retained = None
    try:
        mapping = source["owned"]
        require(type(mapping) is dict, "source-safe owned registry")
        value["owned_count"] = integer(len(mapping))
        require(len(mapping) <= 128, "owned observation128")
        owned = list(dict.items(mapping))
    except BaseException as extra:
        error(extra, "owned registry acquisition")
    try:
        memory = local_base_namespace(source["memory"])
        for field in ("live", "peak", "limit"):
            try:
                value["limit_bytes" if field == "limit" else field] = integer(memory[field])
            except BaseException as extra:
                error(extra, "working scalar acquisition")
        mapping = memory["retained"]
        require(type(mapping) is dict, "source-safe retained registry")
        value["retained_count"] = integer(len(mapping))
        require(len(mapping) <= 128, "retained observation128")
        retained = list(dict.items(mapping))
        for _, size in retained:
            integer(size)
        value["retained_sum"] = integer(sum(size for _, size in retained))
    except BaseException as extra:
        retained = None
        error(extra, "retained registry acquisition")
    if owned is not None and retained is not None:
        value["unregistered_retained_count"] = sum(
            not any(token is held for held, _ in owned) for token, _ in retained)
        value["owned_not_retained_count"] = sum(
            not any(token is held for held, _ in retained) for token, _ in owned)
        value["registry_matches_retained"] = value["unregistered_retained_count"] == \
            value["owned_not_retained_count"] == 0
        try:
            rows = []
            for token, row in owned:
                source_ledger_records(row)
                keys(row, ("reservation_id", "size", "disposition"), "owned source row")
                require(type(row["reservation_id"]) is str and
                        re.fullmatch(r"reservation:[0-9]{8}", row["reservation_id"]) is not None,
                        "existing harness reservation ordinal")
                integer(row["size"])
                require(row["disposition"] in ("retained", "released", "release_failed"), "actual disposition")
                matches = [(held, size) for held, size in retained if token is held]
                operation = None
                if value["events"] is not None:
                    token_events = source.get("token_events")
                    if type(token_events) is dict and len(token_events) <= 128:
                        event_matches = [e for held, e in dict.items(token_events) if held is token]
                    else:
                        event_matches = []
                    matched = [v["operation"] for original, v in zip(source["events"], value["events"])
                               if len(event_matches) == 1 and event_matches[0] is original
                               for r in v["new_reservations"]
                               if r["reservation_id"] == row["reservation_id"] and r["size"] == row["size"]
                               and r["disposition"] == row["disposition"]]
                    if len(matched) == 1:
                        operation = matched[0]
                if operation is None:
                    error(ProbeError("owned ordinal event join unavailable"), "owned event acquisition")
                rows.append({"reservation_id": row["reservation_id"], "operation_id": operation,
                             "registered_size": row["size"], "retained": bool(matches),
                             "retained_size": matches[0][1] if matches else None,
                             "disposition": row["disposition"]})
            rows.sort(key=lambda row: int(row["reservation_id"].split(":")[1]))
            require(len({row["reservation_id"] for row in rows}) == len(rows), "owned ordinal uniqueness")
            value["owned_reservations"] = rows
            value["registered_sizes_match"] = all(
                row["retained_size"] == row["registered_size"] for row in rows if row["retained"])
        except BaseException as extra:
            error(extra, "owned row acquisition")
    if value["live"] is not None and value["retained_sum"] is not None:
        value["live_matches_sum"] = value["live"] == value["retained_sum"]
    value["cleanup_complete"] = ledger_cleanup_complete(value)
    usable = ("events", "event_count", "owned_reservations", "owned_count", "retained_count",
              "retained_sum", "live", "peak", "limit_bytes")
    if not any(value[k] is not None for k in usable):
        for k in _LEDGER_FACTS:
            value[k] = None
    else:
        value["state"] = "incomplete" if value["snapshot_errors"] or \
            value["omitted_snapshot_error_count"] else "observed"
    validate_failure_ledger(value, stage)
    return value


def validate_failure_ledger(value, stage):
    if value is None:
        return
    keys(value, ("state", "epoch", *_LEDGER_FACTS, "snapshot_errors", "omitted_snapshot_error_count",
                 "physical_rss_release_established"), "ReservationFailureLedger")
    require(value["state"] in ("observed", "incomplete", "unavailable"), "snapshot state")
    epoch = value["epoch"]
    keys(epoch, ("kind", "observation_ordinal", "producer_pid", "observed_utc_ns",
                 "observed_monotonic_ns", "failure_stage"), "FailureLedgerEpoch")
    require(epoch["kind"] == "after_harness_unwind_before_failure_write"
            and epoch["failure_stage"] == stage and stage in FAILURE_STAGES, "failure observation epoch")
    for k in ("observation_ordinal", "producer_pid", "observed_utc_ns", "observed_monotonic_ns"):
        integer(epoch[k], 1 if k != "observation_ordinal" else 0)
    require(value["physical_rss_release_established"] is False, "RSS release unestablished")
    for error in array(value["snapshot_errors"], 8):
        validate_error(error)
    integer(value["omitted_snapshot_error_count"])
    for k in ("event_count", "live", "peak", "limit_bytes", "retained_count", "retained_sum",
              "owned_count", "unregistered_retained_count", "owned_not_retained_count"):
        if value[k] is not None:
            integer(value[k])
    for k in ("cleanup_attempted", "live_matches_sum", "registry_matches_retained",
              "registered_sizes_match", "cleanup_complete"):
        if value[k] is not None:
            boolean(value[k])
    if value["state"] == "observed":
        require(all(value[k] is not None for k in _LEDGER_FACTS)
                and not value["snapshot_errors"] and value["omitted_snapshot_error_count"] == 0,
                "completely acquired snapshot")
    else:
        require(bool(value["snapshot_errors"]) or value["omitted_snapshot_error_count"] > 0,
                "explicit snapshot acquisition errors")
    if value["state"] == "unavailable":
        require(all(value[k] is None for k in _LEDGER_FACTS), "unavailable facts never guessed")
    events = value["events"]
    if events is not None:
        validate_reservations(events)
        require(value["event_count"] == len(events), "snapshot event count")
        for event in events:
            require(re.fullmatch(r"operation:[0-9]{8}", event["operation"]) is not None,
                    "actual harness operation ordinal")
            for row in event["new_reservations"]:
                require(re.fullmatch(r"reservation:[0-9]{8}", row["reservation_id"]) is not None,
                        "actual harness reservation ordinal")
    rows = value["owned_reservations"]
    if rows is not None:
        ordinals = []
        for row in array(rows, 128):
            keys(row, ("reservation_id", "operation_id", "registered_size", "retained", "retained_size",
                       "disposition"), "OwnedReservationObservation")
            require(type(row["reservation_id"]) is str and
                    re.fullmatch(r"reservation:[0-9]{8}", row["reservation_id"]) is not None,
                    "owned actual ordinal")
            ordinals.append(int(row["reservation_id"].split(":")[1]))
            integer(row["registered_size"])
            boolean(row["retained"])
            require(row["disposition"] in ("retained", "released", "release_failed"), "owned disposition")
            require((row["retained_size"] is not None) is row["retained"], "independent retained size")
            if row["retained"]:
                integer(row["retained_size"])
            if row["operation_id"] is None:
                require(value["state"] == "incomplete", "missing registered event join")
            else:
                require(type(row["operation_id"]) is str and
                        re.fullmatch(r"operation:[0-9]{8}", row["operation_id"]) is not None,
                        "owned operation ordinal")
                if events is not None:
                    matches = [r for e in events if e["operation"] == row["operation_id"]
                               for r in e["new_reservations"] if r["reservation_id"] == row["reservation_id"]]
                    require(len(matches) == 1 and matches[0]["size"] == row["registered_size"]
                            and matches[0]["disposition"] == row["disposition"], "owned event row join")
        require(ordinals == sorted(set(ordinals)) and value["owned_count"] == len(rows), "exhaustive owned rows")
        if value["owned_not_retained_count"] is not None:
            require(value["owned_not_retained_count"] == sum(not row["retained"] for row in rows),
                    "owned not retained observation")
        if value["registered_sizes_match"] is not None:
            require(value["registered_sizes_match"] == all(row["registered_size"] == row["retained_size"]
                    for row in rows if row["retained"]), "registered versus actual sizes")
        if value["retained_sum"] is not None:
            observed_owned_sum = sum(row["retained_size"] for row in rows if row["retained"])
            require(value["retained_sum"] >= observed_owned_sum, "retained sum covers observed owned sizes")
            if value["unregistered_retained_count"] == 0:
                require(value["retained_sum"] == observed_owned_sum, "complete retained-size sum join")
    if value["live"] is not None and value["retained_sum"] is not None:
        require(value["live_matches_sum"] == (value["live"] == value["retained_sum"]), "actual live versus sum")
    counts = ("owned_count", "retained_count", "unregistered_retained_count", "owned_not_retained_count")
    if all(value[k] is not None for k in counts):
        require(value["retained_count"] == value["owned_count"]-value["owned_not_retained_count"]+
                value["unregistered_retained_count"], "actual registry count differences")
        require(value["registry_matches_retained"] ==
                (value["owned_not_retained_count"] == value["unregistered_retained_count"] == 0),
                "actual registry consistency")
    require(value["cleanup_complete"] is ledger_cleanup_complete(value), "actual final cleanup observation")


def validate_failure(value, envelope=None):
    keys(value, ("schema_version", "task", "run_id", "attempt_nonce", "spec_sha256", "private_sha256",
                 "status", "stage", "primary", "secondary", "omitted_secondary_count", "capture_directory",
                 "fixture_directory", "operations_directory", "partial_artifacts", "marker_state",
                 "capture_outcome", "verify_outcome", "observed_utc_ns", "deadline_exceeded", "limitations",
                 "reservation_ledger"), "Failure")
    validate_envelope(value, envelope)
    require(value["status"] in ("held", "failed") and value["stage"] in FAILURE_STAGES
            and value["marker_state"] in ("absent", "owned_removed", "foreign_preserved", "removal_failed"),
            "Failure status/stage/marker")
    validate_error(value["primary"])
    for error in array(value["secondary"], 64):
        validate_error(error)
    integer(value["omitted_secondary_count"])
    for key in ("capture_directory", "fixture_directory", "operations_directory"):
        if value[key] is not None:
            validate_dir(value[key])
    names = []
    for file in array(value["partial_artifacts"], 32):
        validate_file(file)
        names.append(file["name"])
    require(len(names) == len(set(names)), "Failure partial uniqueness")
    for key in ("capture_outcome", "verify_outcome"):
        if value[key] is not None:
            validate_outcome(value[key])
    integer(value["observed_utc_ns"], 1)
    boolean(value["deadline_exceeded"])
    same_json(value["limitations"], NEW_LIMITATIONS)
    validate_failure_ledger(value["reservation_ledger"], value["stage"])


def decode_failure(data, envelope=None):
    value = strict_json(data, FAILURE_INPUT_BYTES, True)
    validate_failure(value, envelope)
    return value


def validate_diagnostic(value, envelope=None):
    keys(value, ("schema_version", "task", "run_id", "attempt_nonce", "spec_sha256", "private_sha256",
                 "scope", "primary", "secondary", "omitted_secondary_count", "failed_record",
                 "partial_artifacts", "observations", "unavailable"), "Diagnostic")
    validate_envelope(value, envelope)
    require(value["scope"] == "failure_diagnostic_not_success", "failure-only diagnostic")
    validate_error(value["primary"])
    for error in array(value["secondary"], 64):
        validate_error(error)
    integer(value["omitted_secondary_count"])
    files = array(value["partial_artifacts"], 32)
    for file in files:
        validate_file(file)
    if value["failed_record"] is not None:
        keys(value["failed_record"], ("size", "sha256"), "DigestSize")
        integer(value["failed_record"]["size"])
        digest(value["failed_record"]["sha256"])
        matching = [v for v in files if v["name"] == "FAILED.json"]
        require(len(matching) == 1 and {k: matching[0][k] for k in ("size", "sha256")} ==
                value["failed_record"], "original FAILED diagnostic File join")
    for observation in array(value["observations"], 8):
        validate_partial_observation(observation)
    unavailable = array(value["unavailable"], 32)
    for item in unavailable:
        identifier(item)
    require(len(unavailable) == len(set(unavailable)), "diagnostic unavailable uniqueness")


def validate_partial_observation(value):
    keys(value, ("request_id", "solver_result_code", "numeric_origin", "raw_status_evidence", "export_state",
                 "solver_success_validation", "solver_raw_evidence", "nonfinite_cost_tags", "reason",
                 "notes", "omitted_note_count"), "PartialObservation")
    identifier(value["request_id"])
    number = value["solver_result_code"]
    require(number is None or type(number) is int and number.bit_length() <= 256, "original I256 cell")
    require(value["numeric_origin"] in ("original_observation", "unavailable")
            and value["raw_status_evidence"] in ("original_cell", "returned_note", "unavailable")
            and value["export_state"] in ("numeric", "outside_256_bits", "unsupported_type",
                                         "capture_export_not_reached", "unavailable")
            and value["solver_success_validation"] in ("accepted", "rejected", "not_established"),
            "partial source evidence domains")
    require((number is not None) == (value["numeric_origin"] == "original_observation") ==
            (value["export_state"] == "numeric") == (value["raw_status_evidence"] == "original_cell"),
            "original numeric evidence joins")
    raw = value["solver_raw_evidence"]
    if raw is not None:
        require(type(raw) is str and 2+sum(len(_char_fragment(c)) for c in raw) <= 4096,
                "original returned EPS")
    if value["raw_status_evidence"] == "returned_note":
        require(raw is not None, "returned note evidence")
    if value["raw_status_evidence"] == "unavailable":
        require(raw is None, "no invented raw status evidence")
    tags = value["nonfinite_cost_tags"]
    if tags is not None:
        array(tags, 3)
        require(tags == sorted(set(tags)) and all(v in (
            "nan", "negative_infinity", "positive_infinity") for v in tags), "original cost tags")
    # Reuse the effective Reason/Note validator without creating a published Error.
    validate_error({"type": None, "context": "partial observation schema", "reason": value["reason"],
                    "notes": value["notes"], "source_note_count": None,
                    "omitted_note_count": value["omitted_note_count"], "type_unavailable": True,
                    "notes_unavailable": True, "formatting_failure_count": 0})


def failure_chunks(value):
    buffer = bytearray()
    for fragment in canonical_fragments(value):
        require(len(fragment) <= 65536, "bounded Failure fragment")
        if len(buffer)+len(fragment) > 65536:
            yield bytes(buffer)
            buffer.clear()
        buffer.extend(fragment)
    if len(buffer) == 65536:
        yield bytes(buffer)
        buffer.clear()
    buffer.extend(b"\n")
    yield bytes(buffer)


def parse_maps(data):
    require(type(data) is bytes and len(data) <= 8*MIB, "proc maps8MiB")
    result = set()
    for line in data.decode("utf-8").splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and "x" in fields[1] and fields[5].startswith("/"):
            require(not fields[5].endswith(" (deleted)"), "deleted executable mapping")
            result.add(path_text(fields[5]))
    require(len(result) <= 8192, "executable-map count")
    return sorted(result)


def read_maps():
    fd = os.open("/proc/self/maps", os.O_RDONLY)
    primary = None
    try:
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, min(65536, 8*MIB-total+1))
            if not chunk:
                break
            total += len(chunk)
            require(total <= 8*MIB, "proc maps read cap")
            chunks.append(chunk)
        return parse_maps(b"".join(chunks))
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close_owned(fd, primary)


def validate_registry(value, names, cap=8*MIB):
    keys(value, names, "artifact membership")
    for name, record in value.items():
        validate_file(record, name, cap)


def validate_checks(value, cap=64, success=False):
    array(value, cap)
    for item in value:
        keys(item, ("name", "passed"), "check")
        identifier(item["name"])
        boolean(item["passed"])
        require(not success or item["passed"], "failed success check")


def validate_optional_source(value):
    keys(value, ("path", "state", "descriptor"), "optional startup source")
    path_text(value["path"])
    require(value["state"] in ("present", "absent"), "optional source state")
    if value["state"] == "present":
        validate_desc(value["descriptor"], 2*MIB)
        require(value["descriptor"]["path"] == value["path"], "optional source path")
    else:
        require(value["descriptor"] is None, "absent source descriptor")


def validate_startup_sources(value):
    keys(value, ("stdlib_files", "site_source", "site_directories", "customization",
                 "venv_configs"), "startup sources")
    for entry in array(value["stdlib_files"], 256):
        validate_desc(entry, 2*MIB)
    validate_desc(value["site_source"], 2*MIB)
    for site_dir in array(value["site_directories"], 4):
        keys(site_dir, ("path", "state", "pth_files"), "site directory")
        path_text(site_dir["path"])
        require(site_dir["state"] in ("present", "absent"), "site directory state")
        entries = array(site_dir["pth_files"], 32)
        require(site_dir["state"] != "absent" or not entries, "absent site members")
        for entry in entries:
            validate_desc(entry, 65536)
            require(Path(entry["path"]).parent == Path(site_dir["path"])
                    and entry["path"].endswith(".pth"), "startup pth whitelist")
        require([v["path"] for v in entries] == sorted({v["path"] for v in entries}),
                "pth membership order")
    keys(value["customization"], ("sitecustomize", "usercustomize"), "customization")
    for item in value["customization"].values():
        validate_optional_source(item)
    for item in array(value["venv_configs"], 4):
        validate_optional_source(item)


def validate_generated(value):
    fields = ("name", "kind", "persistent_path_required", "factory", "expected_source",
              "stable_sources", "fixed_inputs", "template_terms_concatenated")
    keys(value, fields, "generated declaration")
    require(value["name"] == GENERATED and value["kind"] ==
            "deterministic_import_time_generated_source"
            and value["persistent_path_required"] is False and value["factory"] ==
            "instantiate_non_scriptable_remote_module_template", "generated fixed route")
    keys(value["expected_source"], ("size", "sha256"), "generated bytes")
    require(value["expected_source"] == {"size": 2355, "sha256": GENERATED_HASH},
            "generated exact2355 bytes")
    keys(value["stable_sources"], ("generator", "template", "importing_api"), "generator sources")
    pins = {
        "generator": (5459, "440a619c764e4133564d7956ba060a7223e94664854b94a4a2074d095756db7e"),
        "template": (3463, "0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc"),
        "importing_api": (31308, "55c9c44ba25a2b5edf105fbd740ceff771f937147d7d6a9d6232f05681e7eeaf"),
    }
    for role, entry in value["stable_sources"].items():
        validate_desc(entry, 2*MIB)
        require((entry["size"], entry["sha256"]) == pins[role], "generator stable pin")
    keys(value["fixed_inputs"], ("generated_module_name", "enable_moving_cpu_tensors_to_cuda",
                                 "format_fields"), "generator inputs")
    require(value["fixed_inputs"]["generated_module_name"] == GENERATED
            and value["fixed_inputs"]["enable_moving_cpu_tensors_to_cuda"] is True,
            "generator literal inputs")
    require(value["fixed_inputs"]["format_fields"] == {
        "arg_types": "*args, **kwargs", "args": "*args", "kwargs": "**kwargs",
        "arrow_and_future_return_type": "", "arrow_and_return_type": "",
        "assign_module_interface_cls": "module_interface_cls = None", "jit_script_decorator": "",
    }, "generator format fields")
    require(value["template_terms_concatenated"] == [
        "_TEMPLATE_PREFIX", "_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA"],
        "template terms")


def validate_spec(value):
    keys(value, ("schema_version", "task", "run_id", "output_root", "package_root",
                 "interpreter", "expected_host", "cpu_policy", "probe_source",
                 "bootstrap_source", "code_files", "startup_sources",
                 "prior_inventory_assets", "generated_source", "limits"), "spec")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1
            and value["task"] == TASK, "spec schema/task")
    identifier(value["run_id"], run=True)
    for name in ("output_root", "package_root"):
        path_text(value[name])
    require(Path(value["output_root"]).name == "prosper_revision_output"
            and Path(value["package_root"]).name == "prosper-search", "fixed roots")
    for name in ("interpreter", "probe_source", "bootstrap_source"):
        validate_desc(value[name])
    require(value["interpreter"]["path"] == "/HOME/miniconda3/bin/python3",
            "established interpreter")
    require(value["probe_source"]["path"] == str(Path(value["package_root"])/"scripts"/
                                                "r2_1_filter_cpu_profile_probe.py"),
            "owned probe source path")
    require(value["bootstrap_source"]["path"] == str(Path(value["package_root"])/"scripts"/
                                                    "r2_1_cpu_startup_probe.py")
            and (value["bootstrap_source"]["size"], value["bootstrap_source"]["sha256"]) ==
            (57638, BOOTSTRAP_HASH), "frozen stdlib bootstrap source")
    require(value["expected_host"] == "noderome251"
            and value["cpu_policy"] == "one_allocated_affinitized_cpu", "fixed CPU host/policy")
    keys(value["code_files"], CODE_MODULES, "first-party36 roles")
    for item in value["code_files"].values():
        validate_desc(item, 2*MIB)
        require(CALLER not in item["path"], "caller code declaration forbidden")
    validate_startup_sources(value["startup_sources"])
    keys(value["prior_inventory_assets"], PRIOR_NAMES, "prior metadata roles")
    for item in value["prior_inventory_assets"].values():
        validate_desc(item, 8*MIB)
    validate_generated(value["generated_source"])
    keys(value["limits"], LIMITS, "fixed limits")
    require(all(type(value["limits"][k]) is int and value["limits"][k] == v
                for k, v in LIMITS.items()), "exact limits, no owner changes")
    return value


def validate_initial_authority(value, private):
    keys(value, ("schema_version", "scope", "interpreter", "bootstrap_literal_sha256",
                 "cwd", "pythonpath", "sys_path", "imported_sources", "authority"),
         "initial import authority")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1
            and value["scope"] == "trusted_prelaunch_interpreter_bootstrap_resolution"
            and value["authority"] == "lead_authenticated_external_launcher", "initial trust scope")
    require(value["interpreter"] == private["interpreter"]
            and value["bootstrap_literal_sha256"] == private["bootstrap_literal_sha256"]
            and value["cwd"] == private["cwd"] and value["pythonpath"] == private["pythonpath"],
            "initial authority linkage")
    for item in array(value["sys_path"], 64):
        text(item, empty=True)
    for item in array(value["imported_sources"], 256):
        keys(item, ("module", "origin", "source", "associated_cache"), "initial source")
        text(item["module"], 512)
        require(item["origin"] in ("built-in", "frozen") or Path(item["origin"]).is_absolute(),
                "initial origin")
        for role in ("source", "associated_cache"):
            if item[role] is not None:
                validate_desc(item[role], 2*MIB)
        require(item["origin"] in ("built-in", "frozen") or item["source"] is not None,
                "initial file origin binding")


def validate_private(value, spec):
    fields = (
        "schema_version", "task", "run_id", "attempt_nonce", "owner_uid", "owner_contract",
        "spec", "interpreter", "probe_source", "bootstrap_source", "supervisor_body",
        "bootstrap_literal_sha256", "queue", "host", "cpus", "memory_mib", "wall_seconds",
        "cwd", "pythonpath", "output_root", "sys_path_pre_site", "sys_path_after_site",
        "ordinary_namespaces", "attempt_root", "capture_dir", "fixture_dir", "operations_dir",
        "start_deadline_utc_ns", "terminal_observation_seconds", "environment", "unset",
        "terminal_deadline_utc_ns", "initial_import_authority",
    )
    keys(value, fields, "private launch")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1
            and value["task"] == "hoptimus_filter_cpu_private_launch"
            and value["run_id"] == spec["run_id"], "private schema/run")
    require(type(value["attempt_nonce"]) is str and
            re.fullmatch("[0-9a-f]{32}", value["attempt_nonce"]) is not None, "private nonce")
    integer(value["owner_uid"])
    for name in ("owner_contract", "spec", "interpreter", "probe_source", "bootstrap_source",
                 "supervisor_body", "initial_import_authority"):
        validate_desc(value[name], 8*MIB)
    for name in ("interpreter", "probe_source", "bootstrap_source"):
        require(value[name] == spec[name], "private source equality")
    require(value["owner_contract"] == OWNER_CONTRACT,
            "exact primary authority")
    digest(value["bootstrap_literal_sha256"])
    require(value["bootstrap_literal_sha256"] == sha(BOOT_LITERAL.encode("utf-8")),
            "authenticated fixed BOOT literal")
    identifier(value["queue"])
    require(value["host"] == "noderome251" and type(value["cpus"]) is int
            and value["cpus"] == 1 and type(value["memory_mib"]) is int
            and value["memory_mib"] == 8192 and type(value["wall_seconds"]) is int
            and value["wall_seconds"] == 1800, "selected finite envelope")
    require(value["cwd"] == value["pythonpath"] == spec["package_root"]
            and value["output_root"] == spec["output_root"], "private roots")
    path_text(value["attempt_root"])
    require(Path(value["attempt_root"]).parent == Path(value["output_root"]), "fresh attempt leaf")
    for name, tail in (("capture_dir", "capture"), ("fixture_dir", "fixture"),
                       ("operations_dir", "operations")):
        require(value[name] == str(Path(value["attempt_root"])/tail), "fixed attempt layout")
    for role in ("sys_path_pre_site", "sys_path_after_site"):
        for p in array(value[role], 64):
            text(p, empty=True)
            require(p == "" or Path(p).is_absolute(), "declared sys.path")
    for item in array(value["ordinary_namespaces"], 32):
        keys(item, ("name", "search_locations"), "ordinary namespace")
        text(item["name"], 512)
        require(bool(array(item["search_locations"], 4)), "namespace actual locations")
        for p in item["search_locations"]:
            path_text(p)
    integer(value["start_deadline_utc_ns"])
    require(type(value["terminal_observation_seconds"]) is int and
            value["terminal_observation_seconds"] == 300, "terminal observation300s")
    terminal = value["start_deadline_utc_ns"]+2100*10**9
    integer(terminal)
    require(type(value["terminal_deadline_utc_ns"]) is int
            and value["terminal_deadline_utc_ns"] == terminal, "absolute terminal cutoff")
    environment = {n: "1" for n in NATIVE_NAMES}
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                       PROSPER_OUT=value["output_root"], PYTHONPATH=value["pythonpath"])
    require(value["environment"] == environment, "private exact selected environment")
    require(value["unset"] == ["PROSPER_PARAMS", "PROSPER_MODULE", "R21_FILTER_STABLE_READER_SHA"],
            "private unset controls")
    return value


def validate_stream(value, success=False):
    fields = (
        "descriptor", "identity", "limit_bytes", "received_bytes", "retained_prefix_bytes",
        "written_bytes", "eof_observed", "excess_observed", "reception_scope",
        "received_prefix_sha256", "write_accounting", "persistence", "terminal_read", "error",
    )
    keys(value, fields, "stream accounting")
    require(type(value["limit_bytes"]) is int and value["limit_bytes"] in (16384, MIB), "stream cap")
    for k in ("received_bytes", "retained_prefix_bytes", "written_bytes"):
        integer(value[k], maximum=value["limit_bytes"]+(k == "received_bytes"))
    for k in ("eof_observed", "excess_observed"):
        boolean(value[k])
    digest(value["received_prefix_sha256"])
    require(value["reception_scope"] in ("complete", "known_prefix", "unknown")
            and value["write_accounting"] in ("exact_returned_counts", "unknown_after_failure")
            and value["persistence"] in ("verified_complete", "verified_prefix", "not_written", "unknown")
            and value["terminal_read"] in ("eof", "excess", "receive_deadline",
                                          "drain_deadline", "read_error"), "stream domains")
    if value["descriptor"] is not None:
        validate_desc(value["descriptor"], value["limit_bytes"])
        validate_file(value["identity"], cap=value["limit_bytes"])
        require(value["descriptor"]["size"] == value["identity"]["size"]
                and value["descriptor"]["sha256"] == value["identity"]["sha256"],
                "stream original file linkage")
    else:
        require(value["identity"] is None, "unknown stream identity")
    require(value["retained_prefix_bytes"] <= min(value["received_bytes"], value["limit_bytes"]),
            "received versus retained")
    if value["error"] is not None:
        validate_error(value["error"])
    if success:
        require(value["descriptor"] is not None and value["eof_observed"] is True
                and value["excess_observed"] is False and value["reception_scope"] == "complete"
                and value["received_bytes"] == value["retained_prefix_bytes"] == value["written_bytes"]
                == value["identity"]["size"] and value["received_prefix_sha256"] ==
                value["identity"]["sha256"] and value["write_accounting"] == "exact_returned_counts"
                and value["persistence"] == "verified_complete" and value["terminal_read"] == "eof"
                and value["error"] is None, "complete truthful stream")


def validate_outcome(value, success=False):
    fields = ("pid", "wait_state", "returncode", "signal", "timed_out",
              "started_monotonic_ns", "last_observed_monotonic_ns", "reaped_monotonic_ns",
              "deadline_monotonic_ns", "deadline_met", "signal_attempts")
    keys(value, fields, "actual owned-process outcome")
    integer(value["pid"], 1)
    for k in ("started_monotonic_ns", "last_observed_monotonic_ns", "deadline_monotonic_ns"):
        integer(value[k])
    for k in ("timed_out", "deadline_met"):
        boolean(value[k])
    require(value["started_monotonic_ns"] <= value["last_observed_monotonic_ns"]
            and value["deadline_met"] == (value["last_observed_monotonic_ns"] <=
                                         value["deadline_monotonic_ns"]), "truthful process time")
    require(value["wait_state"] in ("reaped_exit", "reaped_signal", "unresolved"), "wait state")
    if value["wait_state"] == "reaped_exit":
        integer(value["returncode"], maximum=255)
        require(value["signal"] is None, "exit signal null")
    elif value["wait_state"] == "reaped_signal":
        integer(value["signal"], 1)
        require(value["returncode"] is None, "signal returncode null")
    else:
        require(value["returncode"] is value["signal"] is value["reaped_monotonic_ns"] is None,
                "unresolved is not exit0")
    if value["wait_state"] != "unresolved":
        integer(value["reaped_monotonic_ns"])
        require(value["started_monotonic_ns"] <= value["reaped_monotonic_ns"] <=
                value["last_observed_monotonic_ns"], "actual reap observation")
    for attempt in array(value["signal_attempts"], 2):
        keys(attempt, ("signal", "attempted_monotonic_ns", "result", "error"), "signal attempt")
        require(attempt["signal"] in ("SIGTERM", "SIGKILL")
                and attempt["result"] in ("sent", "error", "skipped_already_reaped"),
                "owned signal domain")
        integer(attempt["attempted_monotonic_ns"])
        require(attempt["attempted_monotonic_ns"] >= value["started_monotonic_ns"], "owned signal timestamp")
        if attempt["result"] == "sent" and value["reaped_monotonic_ns"] is not None:
            require(attempt["attempted_monotonic_ns"] <= value["reaped_monotonic_ns"], "no signal after reap")
        if attempt["error"] is not None:
            validate_error(attempt["error"])
        require((attempt["error"] is not None) == (attempt["result"] == "error"), "signal/error accounting")
    require(not success or value["wait_state"] == "reaped_exit" and value["returncode"] == 0
            and value["timed_out"] is False and value["deadline_met"] is True
            and not value["signal_attempts"], "actual successful exit")


def validate_witness(value, envelope, mode, success=False):
    keys(value, (*envelope, "kind", "supervisor_pid", "hostname", "launch_state", "outcome",
                 "stdout", "stderr", "handoff", "operations_directory", "capture_directory",
                 "fixture_directory", "clock", "authenticated_launch_body", "accepted_exit", "checks"),
         "external exit witness")
    validate_envelope(value, envelope)
    require(value["kind"] == ("external_capture_exit_witness" if mode == "capture" else
                             "external_verifier_exit_witness")
            and value["hostname"] == "noderome251", "witness mode/host")
    integer(value["supervisor_pid"], 1)
    require(value["launch_state"] in ("launched", "failed_before_child"), "launch outcome")
    boolean(value["accepted_exit"])
    require(not success or value["accepted_exit"] is True, "witness acceptance")
    if value["launch_state"] == "launched":
        validate_outcome(value["outcome"], success)
        require(value["outcome"]["pid"] != value["supervisor_pid"], "distinct actual child")
        validate_stream(value["stdout"], success)
        validate_stream(value["stderr"], success)
        require(value["stdout"]["limit_bytes"] == 16384
                and value["stderr"]["limit_bytes"] == MIB, "distinct stream caps")
    else:
        require(value["outcome"] is None and not success, "no invented unlaunched PID")
    validate_dir(value["operations_directory"])
    for name in ("capture_directory", "fixture_directory"):
        if value[name] is not None:
            validate_dir(value[name])
        require(not success or value[name] is not None, "successful directory claims")
    if value["handoff"] is not None:
        validate_file(value["handoff"], f"{mode}.stdout", 16384)
        require(value["handoff"] == value["stdout"]["identity"], "original handoff bytes")
    validate_clock(value["clock"], success)
    validate_desc(value["authenticated_launch_body"], 8*MIB)
    validate_checks(value["checks"], 32, success)
    require(not success or value["handoff"] is not None and value["stderr"]["received_bytes"] == 0,
            "nonempty stderr/absent original handoff")


def validate_pool(value, normalized=False):
    require(type(value) is dict, "native pool base dict")
    common = ("user_api", "internal_api", "num_threads", "prefix", "filepath", "version")
    if value.get("internal_api") == "openblas":
        keys(value, (*common, "threading_layer", "architecture"), "OpenBLAS pool")
        require(value["user_api"] == "blas", "OpenBLAS api")
        text(value["threading_layer"])
        text(value["architecture"])
    else:
        keys(value, common, "OpenMP pool")
        require(value["user_api"] == value["internal_api"] == "openmp", "supported pool")
    require(type(value["num_threads"]) is int and value["num_threads"] == 1, "native-one pool")
    text(value["prefix"])
    if value["version"] is not None:
        text(value["version"])
    if normalized:
        require(value["filepath"].startswith("runtime:impl:"), "normalized pool RID")
    else:
        path_text(value["filepath"])


def validate_policy(value):
    keys(value, ("divide", "over", "under", "invalid"), "NumPy policy")
    require(all(v in ("ignore", "warn", "raise", "call", "print", "log")
                for v in value.values()), "NumPy policy domain")


def validate_runtime(value):
    keys(value, ("profile", "observed_paths"), "runtime snapshot")
    p = value["profile"]
    keys(p, ("schema_version", "runtime_scope", "python", "base_profile",
             "distribution_versions", "implementation_assets", "native_environment",
             "numpy_numerics", "pot_interface", "limitations"), "ten-key runtime profile")
    require(type(p["schema_version"]) is int and p["schema_version"] == 1
            and p["runtime_scope"] == "hoptimus_filter_retrieval_pilot_environment_only",
            "unaltered frozen runtime scope")
    keys(p["python"], ("implementation", "version", "byteorder", "executable_asset_id"), "python")
    require(p["python"]["byteorder"] == "little", "CPU byteorder")
    for k in ("implementation", "version"):
        text(p["python"][k])
    b = p["base_profile"]
    keys(b, ("distribution_versions", "implementation", "native_pools"), "base runtime")
    require(b["distribution_versions"] == {
        "numpy": "2.2.4", "scipy": "1.16.0", "scikit-learn": "1.7.0"},
        "fixed native package versions")
    keys(b["implementation"], ("sklearn", "numpy", "scipy", "SGDClassifier"), "base implementations")
    for name, v in b["implementation"].items():
        keys(v, ("path",) if name == "SGDClassifier" else ("version", "path"), "implementation role")
        require(type(v["path"]) is str and v["path"].startswith("runtime:impl:"), "base RID")
        if name != "SGDClassifier":
            text(v["version"])
    for v in array(b["native_pools"], 64):
        validate_pool(v, True)
    keys(p["distribution_versions"], ("POT", "pandas", "threadpoolctl", "pot_module_version"),
         "extra package versions")
    require(p["distribution_versions"]["POT"] == p["distribution_versions"]["pot_module_version"]
            == "0.9.6.post1", "POT version")
    text(p["distribution_versions"]["pandas"])
    text(p["distribution_versions"]["threadpoolctl"])
    env = {n: "1" for n in NATIVE_NAMES}
    env["CUDA_VISIBLE_DEVICES"] = ""
    require(p["native_environment"] == env, "frozen environment keys")
    keys(p["numpy_numerics"], ("error_policy", "baseline", "dispatch", "features",
                              "extension_asset_id"), "NumPy numerics")
    validate_policy(p["numpy_numerics"]["error_policy"])
    for name in ("baseline", "dispatch"):
        for v in array(p["numpy_numerics"][name], 256):
            text(v, 512)
    require(type(p["numpy_numerics"]["features"]) is dict
            and len(p["numpy_numerics"]["features"]) <= 1024, "CPU feature scope")
    for k, v in p["numpy_numerics"]["features"].items():
        text(k, 128)
        boolean(v)
    pot = p["pot_interface"]
    keys(pot, ("signature", "defaults", "call_arguments", "optimal_status_source",
               "numeric_dtype_recipe"), "POT interface")
    text(pot["signature"])
    require(type(pot["defaults"]) is dict and len(pot["defaults"]) <= 16, "POT default scope")
    for name, expected in (("center_dual", True), ("check_marginals", True),
                           ("processes", 1), ("numThreads", 1)):
        require(type(pot["defaults"].get(name)) is type(expected)
                and pot["defaults"][name] == expected, "unchanged scalar defaults")
    require(pot["call_arguments"] == {"dist": {"metric": "cosine"},
            "emd2": {"numItermax": 1000000, "numThreads": 1, "log": True, "return_matrix": False}},
            "unchanged OT log method")
    require(pot["numeric_dtype_recipe"] == {
        "status": "declared", "raw": "float32", "prepared_ot": "float32", "cost": "float32",
        "masses": "float64", "lp_cost": "float64", "distance": "float64",
        "barcodes": "uint8", "barcode_width": 1535}, "declared-versus-observed dtypes")
    require(pot["optimal_status_source"] == {
        "package": "POT", "release": "0.9.6.post1", "optimal_code": 1,
        "emd_h_git_blob": "b56f0601bbcc8f10a6671153ded6a23c408428e8",
        "emd_wrap_git_blob": "53df54fc36b804ff2eb7e179ae25d35071f7194b"}, "optimal source literals")
    implementation = {}
    for v in array(p["implementation_assets"], 76):
        keys(v, ("asset_id", "size", "sha256"), "runtime content declaration")
        require(v["asset_id"].startswith("runtime:impl:"), "runtime prefix")
        integer(v["size"], maximum=2*1024**3)
        digest(v["sha256"])
        require(v["asset_id"] not in implementation, "canonical unique runtime IDs")
        implementation[v["asset_id"]] = v
    logical = {"python", "pandas", "threadpoolctl", "POT", "POT.dist", "POT.network_simplex",
               "POT.emd_wrap", "numpy_cpu_extension", "base:sklearn", "base:numpy", "base:scipy",
               "base:SGDClassifier"}
    logical.update(f"native_pool:{i}" for i in range(len(b["native_pools"])))
    keys(value["observed_paths"], logical, "original logical role occurrence set")
    for v in value["observed_paths"].values():
        keys(v, ("path", "asset_id"), "observed implementation path")
        path_text(v["path"])
        require(v["asset_id"] in implementation, "runtime content reference")
    require(set(implementation) == {v["asset_id"] for v in value["observed_paths"].values()},
            "runtime membership no extra/omitted implementations")
    require(p["python"]["executable_asset_id"] == value["observed_paths"]["python"]["asset_id"]
            and p["numpy_numerics"]["extension_asset_id"] ==
            value["observed_paths"]["numpy_cpu_extension"]["asset_id"], "runtime role identity")
    for name, record in b["implementation"].items():
        require(record["path"] == value["observed_paths"]["base:"+name]["asset_id"],
                "base runtime role identity")
    require(p["limitations"] == [
        "structural_and_unchanged_controls_only_not_cohort_inference",
        "fixed_reference_folds_heads_masks_and_saved_baselines",
        "no_historical_encoder_or_whole_matrix_lineage_certification",
        "exact_control_equality_not_equivalence_margin", "cooperative_file_custody_not_atomic_snapshot",
        "accounted_buffers_not_rss_or_solver_workspace_bound", "runtime_observation_not_full_executable_mapping",
        "no_next_stage_or_job_authorization",
    ], "original frozen runtime limitations")


def parse_entry_arguments(argv):
    required = ("spec", "spec-sha256", "private", "private-sha256", "deadline-monotonic-ns")
    witness = ("capture-witness", "capture-witness-sha256")
    require(type(argv) is list and 1 <= len(argv) <= 17 and all(
        type(v) is str and len(v) <= 4096 and len(v.encode("utf-8")) <= 4096 for v in argv),
        "fixed entry argument domains")
    require(argv[0] in ("capture", "verify") and len(argv) % 2 == 1,
            "fixed entry mode/options")
    permitted = set(required if argv[0] == "capture" else (*required, *witness))
    fields = {}
    for index in range(1, len(argv), 2):
        name, value = argv[index], argv[index+1]
        require(name.startswith("--") and name[2:] in permitted and name[2:] not in fields
                and bool(value) and not value.startswith("--"), "fixed entry option rejection")
        fields[name[2:]] = value
    require(set(fields) == permitted, "fixed entry required options")
    return argparse.Namespace(mode=argv[0], **{
        name.replace("-", "_"): fields.get(name) for name in (*required, *witness)})


def source_handoff_agreement(spec, source):
    validate_bind(source)
    require(all(spec["probe_source"][k] == source[k] for k in ("path", "size", "sha256"))
            and source["path"] == str(Path(spec["package_root"])/"scripts"/
                                      "r2_1_filter_cpu_profile_probe.py"),
            "authenticated source/spec path and bytes")


class Context:
    def __init__(self, argv, source_handoff):
        self.started = time.time_ns(), time.monotonic_ns()
        self.envelope = None
        self.capture_dir = self.fixture_dir = self.operations_dir = None
        self.exercise_state = self.reservation_ledger = None
        self.failure_snapshot = None
        self.failure_snapshot_taken = False
        self.failure_observation_ordinal = 0
        self.observations = []
        self.stage = "bootstrap_args"
        self.args = parse_entry_arguments(argv)
        require(re.fullmatch("[0-9]{1,19}", self.args.deadline_monotonic_ns) is not None,
                "deadline grammar")
        self.deadline = integer(int(self.args.deadline_monotonic_ns), 1)
        require(self.started[1] <= self.deadline, "entry deadline already elapsed")
        self.inventory = Inventory(self.deadline)
        self.stage = "spec_bytes"
        descriptor = {"path": path_text(self.args.spec), "size": os.stat(self.args.spec).st_size,
                      "sha256": digest(self.args.spec_sha256)}
        data, self.spec_binding = self.inventory.read(descriptor["path"], expected=descriptor,
                                                     cap=2*MIB, retain=True)
        self.spec = validate_spec(strict_json(data, 2*MIB, True))
        self.stage = "private_bytes"
        descriptor = {"path": path_text(self.args.private), "size": os.stat(self.args.private).st_size,
                      "sha256": digest(self.args.private_sha256)}
        data, self.private_binding = self.inventory.read(descriptor["path"], expected=descriptor,
                                                        cap=2*MIB, retain=True)
        self.private = validate_private(strict_json(data, 2*MIB, True), self.spec)
        self.stage = "context_agreement"
        source_handoff_agreement(self.spec, source_handoff)
        require(self.private["spec"] == {k: self.spec_binding[k] for k in ("path", "size", "sha256")},
                "trusted executing/source/spec/private agreement")
        self.source_handoff = validate_bind(source_handoff)
        self.envelope = {
            "schema_version": 1, "task": TASK, "run_id": self.spec["run_id"],
            "attempt_nonce": self.private["attempt_nonce"], "spec_sha256": self.spec_binding["sha256"],
            "private_sha256": self.private_binding["sha256"],
        }
        self.stage = "bootstrap"
        pths = [v["path"] for d in self.spec["startup_sources"]["site_directories"]
                for v in d["pth_files"]]
        self.inventory.startup_pth = frozenset(pths)
        descriptor = self.private["initial_import_authority"]
        data, _ = self.inventory.read(descriptor["path"], expected=descriptor, cap=8*MIB, retain=True)
        self.initial_authority = strict_json(data, 8*MIB, True)
        validate_initial_authority(self.initial_authority, self.private)
        require((self.args.mode == "verify") == (self.args.capture_witness is not None)
                and (self.args.capture_witness is None) ==
                (self.args.capture_witness_sha256 is None), "verify-only witness argv")
        self.prior = {}
        for name, descriptor in self.spec["prior_inventory_assets"].items():
            data, _ = self.inventory.read(descriptor["path"], expected=descriptor, cap=8*MIB, retain=True)
            self.prior[name] = strict_json(data, 8*MIB)

    def load_bootstrap(self):
        descriptor = self.spec["bootstrap_source"]
        data, _ = self.inventory.read(descriptor["path"], expected=descriptor, cap=2*MIB, retain=True)
        module = types.ModuleType("_filter_cpu_bootstrap_boundary")
        module.__file__, module.__package__, module.__spec__ = descriptor["path"], "", None
        sys.modules[module.__name__] = module
        exec(compile(data, descriptor["path"], "exec"), module.__dict__)
        return module

    def authenticate_sources(self):
        descriptors = [self.spec["interpreter"], self.spec["probe_source"],
                       self.spec["bootstrap_source"], self.private["supervisor_body"],
                       self.private["owner_contract"], *self.spec["code_files"].values(),
                       *self.spec["startup_sources"]["stdlib_files"],
                       self.spec["startup_sources"]["site_source"],
                       *self.spec["generated_source"]["stable_sources"].values()]
        for item in self.initial_authority["imported_sources"]:
            descriptors += [v for v in (item["source"], item["associated_cache"]) if v is not None]
        require(len(descriptors) <= 512, "source binding scope")
        for descriptor in descriptors:
            self.inventory.read(descriptor["path"], expected=descriptor)
        return descriptors

    def open_operations(self):
        self.operations_dir = OwnedDirectory(self.private["operations_dir"], self.private["owner_uid"],
                                             self.deadline)
        return self.operations_dir

    def failure(self, primary):
        if self.envelope is None:
            return
        if not self.failure_snapshot_taken:
            self.failure_snapshot = failure_ledger_snapshot(
                self.reservation_ledger, self.stage, self.failure_observation_ordinal, primary)
            self.failure_snapshot_taken = True
            self.failure_observation_ordinal += 1
        output = self.capture_dir
        if output is None or output.fd is None:
            add_secondary(primary, ProbeError("failure directory unavailable"), "failure persistence")
            return
        partial = []
        for file in output.files.values():
            try:
                if file["name"] == "FAILED.json":
                    require(file["name"] in output.completed_files, "incomplete original FAILED")
                    output.read_failure(file, output.failure_records[file["name"]])
                else:
                    output.read(file["name"], file, ceiling=None)
                partial.append(file)
            except BaseException as extra:
                add_secondary(primary, extra, "partial artifact verification")
        secondary, omitted = secondary_records(primary)
        record = {
            **self.envelope, "status": "held" if safe_type(primary) ==
            "prosper.evaluation.r2_1_hoptimus_filter_retrieval_cli.PilotPromotionHeld" else "failed",
            "stage": self.stage, "primary": error_record(primary, self.stage),
            "secondary": secondary, "omitted_secondary_count": omitted,
            "capture_directory": output.record,
            "fixture_directory": None if self.fixture_dir is None else self.fixture_dir.record,
            "operations_directory": None if self.operations_dir is None else self.operations_dir.record,
            "partial_artifacts": partial, "marker_state": "absent",
            "capture_outcome": None, "verify_outcome": None, "observed_utc_ns": time.time_ns(),
            "deadline_exceeded": time.monotonic_ns() > self.deadline,
            "limitations": list(NEW_LIMITATIONS),
            "reservation_ledger": self.failure_snapshot,
        }
        failed = None
        try:
            if "FAILED.json" not in output.names():
                output.write("FAILED.json", record, ceiling=None)
            require("FAILED.json" in output.completed_files, "original FAILED not completed")
            verified = output.completed_files["FAILED.json"]
            output.read_failure(verified, output.failure_records["FAILED.json"])
            failed = verified
        except BaseException as extra:
            add_secondary(primary, extra, "FAILED persistence")
        secondary, omitted = secondary_records(primary)
        diagnostic = {
            **self.envelope, "scope": "failure_diagnostic_not_success",
            "primary": error_record(primary, self.stage), "secondary": secondary,
            "omitted_secondary_count": omitted,
            "failed_record": None if failed is None else {k: failed[k] for k in ("size", "sha256")},
            "partial_artifacts": [v for v in partial if v["name"] != "FAILED.json"] +
                                ([] if failed is None else [failed]),
            "observations": [partial_observation(obs, primary, req) for req, obs in self.observations],
            "unavailable": [] if failed is not None else
                           ["failed_record_unverified", "reservation_ledger_unpersisted"],
        }
        try:
            validate_diagnostic(diagnostic, self.envelope)
            if "DIAGNOSTIC.json" not in output.names():
                output.write("DIAGNOSTIC.json", diagnostic, ceiling=8*MIB)
        except BaseException as extra:
            add_secondary(primary, extra, "DIAGNOSTIC persistence")


class CallerDeny:
    def find_spec(self, fullname, path=None, target=None):
        require(CALLER not in fullname, "changing caller import forbidden")
        return None


class ActivationDeny(CallerDeny):
    def find_spec(self, fullname, path=None, target=None):
        super().find_spec(fullname, path, target)
        require(fullname != "prosper" and not fullname.startswith("prosper."),
                "initializer forbidden during site activation")
        return None


def catalog_from_metadata(metadata):
    records = metadata["modules"]
    require(type(records) is dict, "historical module catalog")
    known = [{"key": name, "name": entry["name"]} for name, entry in sorted(records.items())
             if entry["file"] is entry["origin"] is entry["search_locations"] is None]
    require(len(known) == 94 and sha(encoded(known, 65536)) == CATALOG_HASH, "exact Known94")
    six = {"key": "six.moves", "name": "six.moves", "file": None, "origin": None,
           "search_locations": []}
    require(sha(encoded(six, 65536)) == SIX_HASH and records["six.moves"]["file"] is None
            and records["six.moves"]["origin"] is None
            and records["six.moves"]["search_locations"] == [], "exact six.moves")
    return {v["key"]: v["name"] for v in known}


def classify_module(record, catalog, ordinary=(), proxies=None):
    keys(record, ("key", "name", "file", "origin", "search_locations", "cached_path"), "module shape")
    name = record["key"]
    text(name, 512)
    text(record["name"], 512)
    require(CALLER not in name, "caller cached import forbidden")
    f, origin, locations = record["file"], record["origin"], record["search_locations"]
    if name in catalog and f is origin is locations is None:
        require(record["name"] == catalog[name], "historical key/name alias")
        require(record["cached_path"] is None, "known fileless cache")
        return "recorded_fileless_object"
    if name == "six.moves" and f is origin is None and locations == []:
        require(record["name"] == name, "six.moves name")
        require(record["cached_path"] is None, "six.moves cache")
        return "recorded_empty_location_object"
    if name in TORCH_NAMESPACES:
        require(f == TORCH_NAMESPACES[name][0] and origin is locations is None, "Torch namespace marker")
        return "torch_namespace"
    if name == GENERATED:
        require(type(f) is str and Path(f).name == GENERATED+".py" and origin == f,
                "generated source origin")
        return "generated"
    if name in ("__main__", "__mp_main__", "_filter_cpu_bootstrap_boundary"):
        require(type(f) is str and origin is locations is None, "authenticated entry origin")
        require(record["cached_path"] is None, "authenticated entry null cache")
        require(record["name"] == ("__main__" if name == "__mp_main__" else name), "entry alias")
        return "authenticated_entry"
    if name in PROXY_NAMES and origin is locations is None and f is not None:
        expected = proxies[name] if proxies is not None else None
        require(expected is not None and record["name"] == expected["name"] and f == expected["file"],
                "known source proxy")
        return "file_proxy"
    if origin in ("built-in", "frozen"):
        return "builtin" if origin == "built-in" else "frozen"
    if f is None and origin is None:
        candidates = {v["name"]: v["search_locations"] for v in ordinary}
        require(name in candidates and locations == candidates[name] and bool(locations),
                "unsupported null-file object, not invented namespace")
        return "namespace"
    require(type(f) is str and Path(f).is_absolute() and type(origin) is str
            and Path(origin).is_absolute(), "file-backed module origin")
    return "file"


def _base_kind(obj, base):
    return any(item is base for item in _TYPE_SLOTS["__mro__"].__get__(type(obj)))


def _module_shape(name, obj):
    if _base_kind(obj, types.ModuleType):
        namespace = _MODULE_DICT.__get__(obj)
        reported = namespace.get("__name__")
    elif _base_kind(obj, type):
        namespace = _TYPE_SLOTS["__dict__"].__get__(obj)
        reported = _TYPE_SLOTS["__name__"].__get__(obj)
    else:
        raise ProbeError("unsupported loaded module-object representation")
    spec = namespace.get("__spec__")
    require(spec is None or _base_kind(spec, importlib.machinery.ModuleSpec), "accepted module spec")
    spec_slots = {} if spec is None else _SPEC_DICT.__get__(spec)
    require(type(spec_slots) is dict, "loaded spec base namespace")
    locations = spec_slots.get("submodule_search_locations")
    if locations is not None and _base_kind(locations, _NAMESPACE_PATH):
        namespace_path = _NAMESPACE_DICT.__get__(locations)
        require("_path" in namespace_path, "stored namespace locations")
        locations = namespace_path["_path"]
    require(locations is None or type(locations) in (list, tuple), "base search location container")
    require(locations is None or len(locations) <= 64 and all(type(v) is str for v in locations),
            "base search location values")
    require(type(reported) is str and all(namespace.get(k) is None or type(namespace[k]) is str
            for k in ("__file__", "__cached__")) and
            (spec_slots.get("origin") is None or type(spec_slots["origin"]) is str),
            "base module metadata values")
    return {
        "key": name, "name": reported,
        "file": namespace.get("__file__"), "origin": spec_slots.get("origin"),
        "search_locations": None if locations is None else list(locations),
        "cached_path": namespace.get("__cached__"),
    }


def _exposure_rule(name):
    for prefix, provider in (
        ("torch._C.", "torch._C"), ("triton._C.libtriton.", "triton._C.libtriton"),
        ("scipy.optimize._highspy._core.", "scipy.optimize._highspy._core"), ("cv2.", "cv2"),
    ):
        if name.startswith(prefix):
            return provider, name[len(prefix):].split(".")
    exact = {
        "typing.io": ("typing", ["io"]), "typing.re": ("typing", ["re"]),
        "pyexpat.errors": ("pyexpat", ["errors"]), "pyexpat.model": ("pyexpat", ["model"]),
        "xml.parsers.expat.errors": ("pyexpat", ["errors"]),
        "xml.parsers.expat.model": ("pyexpat", ["model"]),
        "torch._VF": ("torch", ["_VF"]), "six.moves": ("six", ["moves"]),
    }
    return exact.get(name, (None, []))


def observe_exposure(name, obj, inventory):
    result = {"status": "unresolved", "provider_key": None, "provider_source_asset_id": None,
              "attribute_path": [], "relation": None}
    provider, route = _exposure_rule(name)
    if provider is None or len(route) > 4 or provider not in sys.modules:
        return result
    root = sys.modules[provider]
    if not _base_kind(root, types.ModuleType):
        return result
    current = root
    for part in route:
        if _base_kind(current, types.ModuleType):
            namespace = _MODULE_DICT.__get__(current)
        elif _base_kind(current, type):
            namespace = _TYPE_SLOTS["__dict__"].__get__(current)
        else:
            return result
        if part not in namespace:
            return result
        current = namespace[part]
    if current is not obj:
        return result
    path = _MODULE_DICT.__get__(root).get("__file__")
    if type(path) is not str or not Path(path).is_absolute():
        return result
    _, binding = inventory.read(path)
    return {"status": "runtime_attribute_identity", "provider_key": provider,
            "provider_source_asset_id": software_id(binding["path"]),
            "attribute_path": route, "relation": "exposes_same_object"}


class SoftwareObservation:
    def __init__(self, context):
        self.ctx, self.inventory = context, context.inventory
        self.catalog = catalog_from_metadata(context.prior["cpu_capture"])
        self.modules, self.maps, self.roles = {p: [] for p in PHASES}, [], []
        self.fileless, self.namespaces, self.pools, self.bridge, self.metadata = [], [], [], {}, []
        self.objects = []
        self.source_descriptors = []
        self.generated = None

    def object_id(self, obj):
        for index, held in enumerate(self.objects):
            if held is obj:
                return f"object:{index:08d}"
        require(len(self.objects) < 3*8192, "object reference registry")
        self.objects.append(obj)
        return f"object:{len(self.objects)-1:08d}"

    def role(self, phase, name, path, occurrence=0):
        self.inventory.current_phase = phase
        _, binding = self.inventory.read(path)
        self.roles.append({"phase": phase, "role": name, "asset_id": software_id(binding["path"]),
                           "reported_path": str(path), "occurrence": occurrence})
        require(len(self.roles) <= 65536, "role occurrence cap")
        return binding

    def observe(self, phase):
        self.inventory.current_phase = phase
        before = tuple(sys.modules.items())
        require(len(before) <= 8192 and not any(CALLER in n for n, _ in before), "module/caller scope")
        for name, obj in sorted(before):
            require(obj is not None, "unavailable module object")
            shape = _module_shape(name, obj)
            kind = classify_module(shape, self.catalog, self.ctx.private["ordinary_namespaces"],
                                   self.ctx.prior["cpu_capture"]["modules"])
            file_id = origin_id = cache_id = None
            cache_state = "not_declared"
            if kind in ("recorded_fileless_object", "recorded_empty_location_object"):
                self.fileless.append({
                    "phase": phase, "key": name, "name": shape["name"], "object_id": self.object_id(obj),
                    "representation": "known_all_null" if kind == "recorded_fileless_object" else
                    "known_empty_locations", "historical_catalog_sha256": CATALOG_HASH if
                    kind == "recorded_fileless_object" else SIX_HASH,
                    "creation_origin": "unresolved", "defining_provider_asset_id": None,
                    "exposure": observe_exposure(name, obj, self.inventory),
                })
            elif kind not in ("torch_namespace", "generated") and shape["file"] is not None:
                binding = self.role(phase, "module:"+name, shape["file"])
                file_id = software_id(binding["path"])
                if kind in ("file", "file_proxy", "authenticated_entry"):
                    if kind == "file":
                        origin_binding = self.role(phase, "origin:"+name, shape["origin"])
                        require(origin_binding == binding,
                                "module file/origin disagreement")
                        origin_id = file_id
                    elif kind == "file_proxy":
                        prior_path = self.ctx.prior["cpu_capture"]["modules"][name]["file"]
                        prior_binding = self.ctx.prior["cpu_capture"]["consumed_paths"][prior_path]
                        require((binding["size"], binding["sha256"]) ==
                                (prior_binding["size"], prior_binding["sha256"]), "proxy source bytes")
            cached = shape["cached_path"]
            if cached is not None:
                path_text(cached)
                require(CALLER not in cached, "caller cache forbidden")
                if os.path.lexists(cached):
                    binding = self.role(phase, "cache:"+name, cached)
                    cache_id, cache_state = software_id(binding["path"]), "present"
                else:
                    cache_state = "absent"
            if kind in ("recorded_fileless_object", "recorded_empty_location_object"):
                require(cached is None, "fileless cache provenance")
            self.modules[phase].append({
                **shape, "phase": phase, "kind": kind, "file_asset_id": file_id,
                "origin_asset_id": origin_id, "cached_asset_id": cache_id, "cache_state": cache_state,
            })
        for path in read_maps():
            binding = self.role(phase, "executable_mapping", path)
            self.maps.append({"phase": phase, "path": path, "asset_id": software_id(binding["path"])})
        require(len(self.fileless) <= 285, "fileless observations<=95 per phase")

    def first_party(self, F):
        paths = F.code_paths()
        require(set(paths) == set(self.ctx.spec["code_files"]), "actual36 code roles")
        actual_names = {n for n in sys.modules if n == "prosper" or n.startswith("prosper.")}
        require(actual_names == set(CODE_MODULES.values()), "exact first-party module graph")
        for role, path in paths.items():
            descriptor = self.ctx.spec["code_files"][role]
            require(str(Path(path).resolve(strict=True)) ==
                    str(Path(descriptor["path"]).resolve(strict=True)), "actual code origin")
            self.inventory.read(path, expected=descriptor, cap=2*MIB)
        return {role: str(Path(path).resolve(strict=True)) for role, path in paths.items()}

    def runtime(self, F, phase, leases):
        base = F.fit._runtime_profile()
        extension = F.np._core._multiarray_umath
        logical = {
            "python": sys.executable, "pandas": F.pd.__file__, "threadpoolctl": F.threadpoolctl.__file__,
            "POT": F.ot.__file__, "POT.dist": F.inspect.getfile(F.ot.dist),
            "POT.network_simplex": F.simplex.__file__, "POT.emd_wrap": F.emd_wrap.__file__,
            "numpy_cpu_extension": extension.__file__,
        }
        logical.update({"base:"+k: v["path"] for k, v in base["implementation"].items()})
        logical.update({f"native_pool:{i}": v["filepath"] for i, v in enumerate(base["native_pools"])})
        require(len(base["native_pools"]) <= 64, "runtime pool cap")
        canonical = {}
        for name, path in logical.items():
            binding = self.role(phase, name, path)
            canonical[binding["path"]] = binding
        require(len(canonical) <= 76, "runtime implementation cap")
        assets = []
        for path, binding in sorted(canonical.items()):
            rid = runtime_id(path)
            declaration = {"asset_id": rid, "role": "runtime_implementation", "path": path,
                           "size": binding["size"], "sha256": binding["sha256"]}
            guard = F.assets.AssetGuard([F.assets.AssetSpec(**declaration)], base_dir=self.ctx.spec["package_root"],
                                       policies={"runtime_implementation":
                                                 F.assets.AssetPolicy("path", 2*1024**3)})
            leases.enter_context(guard)
            leases.enter_context(guard.lease_verified_path(rid))
            leases.guards.append(guard)
            assets.append(declaration)
            bridge = self.bridge.setdefault(rid, runtime_bridge(binding, []))
            require(bridge["binding"] == binding, "bridge same-run bytes")
            if phase not in bridge["phases"]:
                bridge["phases"].append(phase)
        snapshot = F.runtime_snapshot(assets)
        validate_runtime(snapshot)
        for name, value in snapshot["observed_paths"].items():
            binding = self.inventory.records[str(Path(value["path"]).resolve(strict=True))]
            require(value["asset_id"] == runtime_id(binding["path"]), "stock snapshot RID bridge")
        for ordinal, pool in enumerate(base["native_pools"]):
            validate_pool(pool)
            binding = self.inventory.records[str(Path(pool["filepath"]).resolve(strict=True))]
            self.pools.append({"phase": phase, "ordinal": ordinal,
                               "implementation_asset_id": runtime_id(binding["path"]),
                               "software_asset_id": software_id(binding["path"]), "reported": dict(pool)})
        return snapshot

    def versions(self, F):
        expected = {"numpy": "2.2.4", "scipy": "1.16.0", "scikit-learn": "1.7.0", "POT": "0.9.6.post1"}
        for name in ("numpy", "scipy", "scikit-learn", "POT", "pandas", "threadpoolctl"):
            distribution = F.importlib.metadata.distribution(name)
            files = distribution.files
            require(files is not None and len(files) <= 65536, "bounded wheel metadata catalog")
            candidates = [v for v in files if str(v).endswith(".dist-info/METADATA")]
            require(len(candidates) == 1, "single wheel METADATA")
            member = candidates[0]
            record_member = next((v for v in files if str(v) == str(Path(str(member)).parent/"RECORD")), None)
            require(record_member is not None, "matching wheel RECORD")
            meta_path = str(Path(distribution.locate_file(member)).absolute())
            record_path = str(Path(distribution.locate_file(record_member)).absolute())
            data, meta = self.inventory.read(meta_path, cap=8*MIB, retain=True)
            _, rec = self.inventory.read(record_path, cap=8*MIB)
            import email.parser
            message = email.parser.Parser().parsestr(data.decode("utf-8"))
            normal = lambda s: re.sub("[-_.]+", "-", s.lower())
            require(normal(message["Name"]) == normal(name), "distribution metadata Name")
            observed = F.importlib.metadata.version(name)
            require(message["Version"] == observed and (name not in expected or expected[name] == observed),
                    "version metadata/getter linkage")
            self.metadata.append({
                "distribution": name, "required_version": expected.get(name), "observed_version": observed,
                "metadata_asset_id": software_id(meta["path"]), "record_asset_id": software_id(rec["path"]),
                "metadata_path": meta_path, "record_path": record_path,
                "distribution_root": str(Path(meta["path"]).parent),
                "semantic_checks": ["single_wheel_metadata", "name_matches_distribution",
                                    "metadata_version_matches_getter", "original_metadata_and_record_bytes"],
            })

    def historical(self):
        cpu = self.ctx.prior["cpu_capture"]
        known_alias = {k: v["path"] for k, v in cpu["consumed_paths"].items()}
        known_alias.update({k: k for k in cpu["software_bindings"]})
        inventories = {
            "original_profile": self.ctx.prior["original_profile"]["software_bindings"],
            "cpu_capture": cpu["software_bindings"],
            "cpu_private_binding": self.ctx.prior["cpu_private_binding"]["union"],
        }
        by_target = {}
        for inventory, records in inventories.items():
            for path, item in records.items():
                target = known_alias.get(path, path)
                by_target.setdefault(target, []).append((inventory, path, item))
        comparisons, aliases, consumed = [], [], set()
        required_paths = {str(Path(v["path"]).resolve(strict=True)) for v in self.source_descriptors}
        required_paths.update(v["canonical_path"] for v in self.bridge.values())
        namespace_ids = {v[k] for v in self.namespaces for k in ("defining_asset_id", "torch_asset_id")}
        required_paths.update(p for p in self.inventory.records if software_id(p) in namespace_ids)
        for target, binding in sorted(self.inventory.records.items()):
            matches = by_target.get(target, [])
            prior, changed = [], False
            for inventory, path, old in matches:
                aliases.append({"inventory": inventory, "prior_path": path, "asset_id": software_id(target)})
                consumed.add((inventory, path))
                changed |= (old["size"], old["sha256"]) != (binding["size"], binding["sha256"])
                candidate = {"inventory": inventory, "prior_path": path, "prior_size": old["size"],
                             "prior_sha256": old["sha256"], "match_kind": "recorded_alias" if path != target
                             else "recorded_canonical"}
                if not any(v["inventory"] == inventory and v["match_kind"] == candidate["match_kind"]
                           for v in prior):
                    prior.append(candidate)
            required = target in required_paths
            require(not changed or not required, "changed required historical/source binding")
            comparisons.append({"asset_id": software_id(target), "canonical_path": target,
                                "classification": "changed" if changed else "matched" if matches else "new",
                                "prior_matches": prior, "required": required})
        unseen = [{"inventory": name, "path": path, "size": old["size"], "sha256": old["sha256"]}
                  for name, records in inventories.items() for path, old in records.items()
                  if (name, path) not in consumed]
        require(len(unseen) <= 20000 and len(aliases) <= 32768, "historical metadata record caps")
        return comparisons, sorted(aliases, key=lambda v: (v["inventory"], v["prior_path"])), sorted(
            unseen, key=lambda v: (v["inventory"], v["path"]))

    def finish(self):
        comparisons, historical_aliases, unseen = self.historical()
        required = {v["canonical_path"] for v in comparisons if v["required"]}
        files = []
        for path, first in sorted(self.inventory.records.items(), key=lambda kv: software_id(kv[0])):
            phases = [v for v in PHASES if v in self.inventory.phases[path]]
            files.append({"asset_id": software_id(path), "binding": first, "first_phase": phases[0],
                          "phases": phases, "required": path in required, "final_binding": dict(first)})
        aliases = []
        for path, binding in sorted(self.inventory.aliases.items()):
            aliases.append({"lexical_path": path, "asset_id": software_id(binding["path"]),
                            "phases": [v for v in PHASES if v in self.inventory.alias_phases[path]]})
        counts = {
            "canonical_files": len(files), "lexical_alias_paths": len(aliases),
            "role_occurrences": len(self.roles), "unique_content_bytes": self.inventory.total,
            "bootstrap_modules": len(self.modules["bootstrap"]), "before_modules": len(self.modules["before"]),
            "after_modules": len(self.modules["after"]), "bootstrap_maps": sum(v["phase"] == "bootstrap" for v in self.maps),
            "before_maps": sum(v["phase"] == "before" for v in self.maps),
            "after_maps": sum(v["phase"] == "after" for v in self.maps),
            "before_native_pools": sum(v["phase"] == "before" for v in self.pools),
            "after_native_pools": sum(v["phase"] == "after" for v in self.pools),
            "required_files": len(required), "new_files": sum(v["classification"] == "new" for v in comparisons),
            "changed_files": sum(v["classification"] == "changed" for v in comparisons),
            "matched_files": sum(v["classification"] == "matched" for v in comparisons),
            "historical_unobserved_records": len(unseen), "fileless_object_records": len(self.fileless),
            "unresolved_creation_origins": len(self.fileless), "runtime_bridge_rows": len(self.bridge),
        }
        return {
            **self.ctx.envelope, "scope": "observed_startup_and_tiny_exercise_files_only",
            "files": files, "aliases": aliases, "roles": sorted(self.roles, key=lambda v: (
                PHASES.index(v["phase"]), v["role"], v["occurrence"])), "modules": self.modules,
            "executable_mappings": self.maps, "namespaces": self.namespaces, "generated": self.generated,
            "native_pools": self.pools, "version_metadata": self.metadata,
            "historical_comparisons": comparisons, "historical_aliases": historical_aliases,
            "historical_unobserved": unseen, "counts": counts, "fileless_objects": self.fileless,
            "runtime_bridge": sorted(self.bridge.values(), key=lambda v: v["runtime_asset_id"]),
            "limitations": list(NEW_LIMITATIONS),
        }


def startup_gate(ctx, B, observation):
    p = ctx.private
    require(socket.gethostname() == "noderome251" and os.getuid() == p["owner_uid"], "private host/UID")
    require(str(Path.cwd()) == p["cwd"] and list(sys.path) == p["sys_path_pre_site"], "actual initial search context")
    require(sys.flags.no_site and sys.flags.dont_write_bytecode, "controlled -S -B startup")
    require(len(os.sched_getaffinity(0)) == 1, "one allocated affinitized CPU")
    require(not any(n == "prosper" or n.startswith("prosper.") or CALLER in n for n in sys.modules),
            "fresh uncached provider/caller state")
    require(all(os.environ.get(k) == v for k, v in p["environment"].items())
            and all(k not in os.environ for k in p["unset"]), "selected controls, not environment dump")
    descriptors = ctx.authenticate_sources()
    observation.source_descriptors = descriptors
    observation.observe("bootstrap")
    source = ctx.spec["startup_sources"]

    def membership():
        for d in source["site_directories"]:
            if d["state"] == "absent":
                require(not os.path.lexists(d["path"]), "declared absent site directory")
            else:
                require(Path(d["path"]).is_dir() and not Path(d["path"]).is_symlink(), "site directory")
                found = sorted(str(Path(d["path"])/name) for name in os.listdir(d["path"])
                               if name.endswith(".pth"))
                require(found == [v["path"] for v in d["pth_files"]], "complete bounded pth membership")
                for v in d["pth_files"]:
                    ctx.inventory.read(v["path"], expected=v, cap=65536)
        for v in (*source["customization"].values(), *source["venv_configs"]):
            if v["state"] == "absent":
                require(not os.path.lexists(v["path"]), "absent startup candidate")
            else:
                ctx.inventory.read(v["path"], expected=v["descriptor"], cap=2*MIB)

    membership()
    deny = ActivationDeny()
    sys.meta_path.insert(0, deny)
    held = StartupLeases(ctx)
    primary = None
    try:
        held.acquire()
        import site
        require(str(Path(site.__file__).resolve(strict=True)) ==
                str(Path(source["site_source"]["path"]).resolve(strict=True)), "actual site source")
        site.main()
    except BaseException as exc:
        primary = exc
        raise
    finally:
        sys.meta_path.remove(deny)
        held.close(primary)
    caller = CallerDeny()
    sys.meta_path.insert(0, caller)
    membership()
    require(list(sys.path) == p["sys_path_after_site"], "actual manual-site search context")
    root = B.check_effective_prosper_root(ctx.spec["code_files"]["package_init"]["path"])
    original = []
    for descriptor in descriptors:
        original.append(dict(descriptor))
    final = [dict(ctx.inventory.aliases[d["path"]]) for d in descriptors]
    return {
        **ctx.envelope, "scope": "stdlib_first_source_buffer_then_explicit_site",
        "hostname": "noderome251", "capture_pid": os.getpid(), "owner_uid": os.getuid(),
        "source_handoff": ctx.source_handoff,
        "interpreter": ctx.inventory.aliases[ctx.spec["interpreter"]["path"]],
        "bootstrap_source": ctx.inventory.aliases[ctx.spec["bootstrap_source"]["path"]],
        "root_origin": root, "cwd": str(Path.cwd()), "sys_path_pre_site": p["sys_path_pre_site"],
        "sys_path_after_site": list(sys.path), "flags": {
            "no_site": bool(sys.flags.no_site), "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
            "isolated": bool(sys.flags.isolated), "ignore_environment": bool(sys.flags.ignore_environment),
            "safe_path": bool(sys.flags.safe_path)},
        "cpu": {"count": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0))},
        "environment": dict(p["environment"]), "startup_originals": original, "startup_final": final,
        "site_directories": source["site_directories"], "customization": source["customization"],
        "clock": clock(ctx.started, ctx.deadline), "checks": [{"name": "startup_source_gate", "passed": True}],
        "limitations": list(NEW_LIMITATIONS),
    }


class ThreeSourceAdapter:
    def __init__(self, inventory, declaration):
        self.inventory, self.entries, self.bytes = inventory, declaration["stable_sources"], {}

    def read(self, path, limit, *, expected, retain=False):
        require(limit == 16*MIB and retain is True, "narrow generated adapter call")
        matches = [role for role, v in self.entries.items() if v["path"] == path and v == expected]
        require(len(matches) == 1, "only three exact stable inputs")
        data, record = self.inventory.read(path, expected=expected, cap=2*MIB, retain=True)
        self.bytes[matches[0]] = data
        return data, record


def generated_source(ctx, B, observation):
    declaration = ctx.spec["generated_source"]
    adapter = ThreeSourceAdapter(ctx.inventory, declaration)
    returned = B.derive_generated({"baseline": {"generated_module_bindings": {GENERATED: declaration}},
                                  "inventory": adapter})
    require(returned == declaration and len(adapter.bytes) == 3, "actual declaration, not payload")
    constants = {}
    for node in ast.parse(adapter.bytes["template"]).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    data = "".join(constants[n] for n in declaration["template_terms_concatenated"]).format(
        **declaration["fixed_inputs"]["format_fields"]).encode("utf-8")
    require(len(data) == 2355 and sha(data) == GENERATED_HASH, "literal generated bytes")
    module = sys.modules.get(GENERATED)
    require(module is not None, "generated live module")
    path = path_text(module.__file__)
    require(Path(path).name == GENERATED+".py" and module.__spec__ is not None
            and module.__spec__.origin == path, "generated live origin")
    transient = Inventory(ctx.deadline)
    live_data, binding = transient.read(path, expected={"size": 2355, "sha256": GENERATED_HASH},
                                        cap=2*MIB, retain=True)
    require(live_data == data, "actual live generated source")
    retained = ctx.operations_dir.write("generated_source.py", data, raw=True, ceiling=2*MIB)
    observation.generated = {
        "name": GENERATED, "live_binding": binding, "persistent_path_required": False,
        "expected_source": {"size": 2355, "sha256": GENERATED_HASH},
        "stable_source_asset_ids": {r: software_id(ctx.inventory.aliases[v["path"]]["path"])
                                   for r, v in declaration["stable_sources"].items()},
        "phases": ["before", "after"], "retained_bytes_descriptor": retained,
        "post_exit_path_state": "not_required",
    }
    return transient


def requests():
    result = []
    for prefix, method in (("ot", "ot_cosine"), ("bob", "yottixel")):
        for suffix, selection, budget in (("literal", "literal_all_tissue", None),
                                          ("alltrue", "all_true", None),
                                          ("filter5", "filtered", 5), ("filter100", "filtered", 100)):
            result.append({"request_id": prefix+"_"+suffix, "query_index": 0, "query_fold": 0,
                           "method": method, "selection": selection, "budget_percent": budget,
                           "reference_index": 1})
    return result


BASELINE_HEX = [["0x0.0p+0", "0x1.0000000000000p-1"], ["0x1.0000000000000p-1", "0x0.0p+0"]]
BASELINE_RECIPE = {
    "leaf": "baseline.pkl", "pickle_protocol": 4, "container": "pandas.DataFrame",
    "dtype": "<f8", "shape": [2, 2], "index": ["1", "2"], "columns": ["1", "2"],
    "index_dtype": "object", "columns_dtype": "object", "index_name": None, "columns_name": None,
    "values_hex": BASELINE_HEX,
}


def record_projection(record):
    require(dataclasses.is_dataclass(record) and not isinstance(record, type), "typed record projection")
    result = {}
    for field in dataclasses.fields(record):
        value = getattr(record, field.name)
        if dataclasses.is_dataclass(value):
            value = record_projection(value)
        elif type(value) is tuple:
            value = list(value)
        result[field.name] = value
    return result


def descriptor_for(directory, name):
    file = directory.files[name]
    return {"path": str(directory.path/name), "size": file["size"], "sha256": file["sha256"]}


def fixture(ctx, F, R, stack):
    import pickle
    np, A = F.np, F.assets
    d = ctx.fixture_dir
    counts = R.MosaicCounts(2, 2, 1, 2, 2, 4, 10, 20, 40)
    slides = (R.SlideRecord(0, 0, "probe_q", "probe_patient_q", "probe_group", 0, 0, 2, 2),
              R.SlideRecord(1, 1, "probe_r", "probe_patient_r", "probe_group", 1, 2, 4, 2))
    q, r = np.zeros((2, 1536), dtype="<f4", order="C"), np.zeros((2, 1536), dtype="<f4", order="F")
    q[0, 0], q[1, 1], r[0, 0], r[1, 2] = 1, 1, 1, 1
    for name, value in (("query.pkl", q), ("reference.pkl", r)):
        d.write(name, pickle.dumps((value, [{"wsi_loc": [0, 0]}, {"wsi_loc": [16, 0]}]),
                                  protocol=4), raw=True, ceiling=MIB)
    del q, r
    frame = F.pd.DataFrame(np.array([[0, .5], [.5, 0]], dtype="<f8", order="C"),
                          index=F.pd.Index(["1", "2"], dtype="object", name=None),
                          columns=F.pd.Index(["1", "2"], dtype="object", name=None), copy=True)
    d.write("baseline.pkl", pickle.dumps(frame, protocol=4), raw=True, ceiling=MIB)
    del frame
    arrays = {"coordinates.npy": np.array([[0, 0], [16, 0], [0, 0], [16, 0]], dtype="<i8")}
    arrays.update({f"mask_f{f}_b{b}.npy": np.array(
        [False, True, True, False] if b == 5 else [True]*4, dtype="|b1")
                   for f in range(5) for b in (5, 100)})
    for name, value in arrays.items():
        buffer = io.BytesIO()
        np.lib.format.write_array(buffer, value, version=(1, 0), allow_pickle=False)
        d.write(name, buffer.getvalue(), raw=True, ceiling=65536)
    del arrays
    data = {name: descriptor_for(d, name) for name in DATA_NAMES}
    spec = {"schema_version": 1, "task": "synthetic_filter_cpu_fixture", "run_id": ctx.spec["run_id"],
            "scope": "generated_basis_vectors_not_experimental_results", "pickle_protocol": 4,
            "counts": record_projection(counts), "slides": [record_projection(s) for s in slides],
            "requests": requests(), "data": data, "baseline_recipe": BASELINE_RECIPE}
    d.write("synthetic_spec.json", spec)
    complete = {"schema_version": 1, "task": "synthetic_filter_cpu_fixture", "run_id": ctx.spec["run_id"],
                "scope": "generated_data_byte_registry_only", "spec_sha256": d.files["synthetic_spec.json"]["sha256"],
                "outputs": {n: {k: data[n][k] for k in ("size", "sha256")} for n in DATA_NAMES}}
    d.write("synthetic_complete.json", complete)
    review = {"schema_version": 1, "task": "synthetic_filter_cpu_fixture", "run_id": ctx.spec["run_id"],
              "scope": "owner_generated_fixture_not_actual105_or_independent_acceptance",
              "spec_sha256": d.files["synthetic_spec.json"]["sha256"],
              "complete_sha256": d.files["synthetic_complete.json"]["sha256"]}
    d.write("synthetic_review.json", review)
    expected = R.MosaicExpected(d.files["synthetic_review.json"]["sha256"],
                               d.files["synthetic_complete.json"]["sha256"],
                               d.files["synthetic_spec.json"]["sha256"], ctx.spec["run_id"], counts)
    declarations = []
    for name in (*DATA_NAMES, *CONTROL_NAMES):
        aid, role = fixture_identity(name)
        declarations.append(A.AssetSpec(asset_id=aid, role=role, **descriptor_for(d, name)))
    guard = A.AssetGuard(declarations, base_dir=d.path, policies={
        "mosaic_feature": A.AssetPolicy("bytes", MIB), "baseline_matrix": A.AssetPolicy("bytes", MIB),
        "fixture_npy": A.AssetPolicy("bytes", 65536), "fixture_control": A.AssetPolicy("bytes", 8*MIB)})
    stack.enter_context(guard)
    controls, consumed_arrays, backings = [], [], {}
    for name in (*CONTROL_NAMES, *(n for n in DATA_NAMES if n.endswith(".npy"))):
        payload = guard.read_verified("fixture:"+name)
        checks = ("owner_generated_synthetic_control_bytes",) if name in CONTROL_NAMES else (
            "generated_synthetic_NPY1_exact_extent_dtype_shape",)
        consumed = R.ConsumptionRecord("fixture:"+name, name, str(d.path/name), len(payload), sha(payload), checks)
        if name in CONTROL_NAMES:
            require(strict_json(payload, canonical=True) == {"synthetic_spec.json": spec,
                    "synthetic_complete.json": complete, "synthetic_review.json": review}[name],
                    "generated controls exact original bytes")
            controls.append(consumed)
        else:
            consumed_arrays.append(consumed)
            backings[name] = R.NpyBacking(payload, *R._npy_header(payload))
    masks = []
    for f in range(5):
        for b in (5, 100):
            mc = tuple(R.MaskCountRecord(f, b, s.slide_index, s.slide_id, s.patient_id, s.slide_fold,
                       int(s.slide_fold == f), 2, 1 if b == 5 else 2, 0, 0, 1 if b == 5 else 0,
                       0, 1 if b == 5 else 2, 1 if b == 5 else 0, "nonempty") for s in slides)
            masks.append(R.LoadedMask(R.HeadView(f, b), backings[f"mask_f{f}_b{b}.npy"], mc))
    by_id = {v.asset_id: v for v in declarations}
    features = tuple(R.FeatureRecord(s.slide_index, s.slide_id, str(d.path/n), str(d.path/n),
                                    data[n]["size"], data[n]["sha256"], 2, "float32", 1536,
                                    by_id["feature:"+s.slide_id])
                     for s, n in zip(slides, ("query.pkl", "reference.pkl")))
    bundle = R.MosaicBundle(expected, slides, features, backings["coordinates.npy"], tuple(masks),
                           tuple(R.ArtifactDescriptor(n, data[n]["size"], data[n]["sha256"])
                                 for n in sorted(DATA_NAMES)), tuple(controls), tuple(consumed_arrays), ())
    d.write("fixture_manifest.json", {
        "schema_version": 1, "task": "synthetic_filter_cpu_fixture", "run_id": ctx.spec["run_id"],
        "directory": d.record, "expected": record_projection(expected), "files": dict(d.files)})
    require(d.names() == set(FIXTURE_NAMES), "18 synthetic fixture leaves")
    internal = {"pair_requests": requests(), "row_requests": [], "query_indices": [0],
                "limits": {"max_pair_cost_cells": 4, "max_accounted_working_bytes": 8*MIB},
                "baseline_assets": {"matrices": {"ot_cosine": record_projection(
                    by_id["baseline:synthetic:ot_cosine"])}}}
    for req in internal["pair_requests"]:
        F._request(req, pair=True)
    state = F.PilotState(internal, F.WorkingMemory(8*MIB))
    state.guard, state.bundle, state.baseline = guard, bundle, {"raw_ids": ["1", "2"]}
    return state, guard


def array_observation(value, np):
    require(value.dtype.str in ("<f4", "|u1") and value.ndim == 2
            and value.shape[0] in (1, 2) and value.shape[1] ==
            (1536 if value.dtype.str == "<f4" else 1535), "tiny array domain")
    require(np.isfinite(value).all() and ((value == 0) | (value == 1)).all(), "basis/binary atoms")
    nonzero = []
    for row, column in zip(*np.nonzero(value)):
        nonzero.append({"row": int(row), "column": int(column),
                        "float_hex": float(value[row, column]).hex() if value.dtype.str == "<f4" else None,
                        "uint_value": 1 if value.dtype.str == "|u1" else None})
    require(len(nonzero) <= 8, "sparse atom cap")
    payload = value.tobytes(order="C")
    return {"dtype": value.dtype.str, "shape": list(value.shape), "strides": list(value.strides),
            "c_contiguous": bool(value.flags.c_contiguous), "f_contiguous": bool(value.flags.f_contiguous),
            "c_order_bytes": len(payload), "c_order_sha256": sha(payload), "nonzero": nonzero}


def assert_basis(array, dimensions, np):
    expected = np.zeros(array.shape, dtype=array.dtype)
    for row, dim in enumerate(dimensions):
        expected[row, dim] = 1
    require(np.array_equal(array, expected), "actual selected/prepared exact basis values")


def exercise(ctx, F, R, state, ledger):
    np = F.np
    policy = np.geterr()
    cells, decodes, raw_consumptions = [], [], []
    literal = {}
    primary = None

    def decoded(raw, index):
        ordinal = len([v for v in decodes if v["slide_index"] == index])
        observation = array_observation(raw.raw, np)
        require(observation["strides"] == ([6144, 4] if index == 0 else [4, 8]),
                "decoder preserves C/F source layout")
        assert_basis(raw.raw, [0, 1] if index == 0 else [0, 2], np)
        consumed = F._consumption_json(raw.consumption)
        decodes.append({"slide_index": index, "decode_ordinal": ordinal,
                        "layout": record_projection(raw.layout), "consumption": consumed, "raw": observation})
        raw_consumptions.append(consumed)

    try:
        for req in requests():
            ctx.stage = "decode"
            state.current_request, state.stage = req, "pair_probe"
            observation = state.observation(req, 1)
            ctx.observations.append((req["request_id"], observation))
            with ledger.operation("query_event:"+req["request_id"], "query_event"):
                query = state.query_event(req)
            if not any(v["slide_index"] == 0 for v in decodes):
                decoded(state.query_raw[0], 0)
            reference = raw = None
            raw_token = None
            pair_primary = None
            try:
                with ledger.operation("reference_raw:"+req["request_id"], "raw_reference"):
                    raw, raw_token = state.raw(1, query=False)
                decoded(raw, 1)
                ctx.stage = "prepare"
                with ledger.operation("reference_prepare:"+req["request_id"], "prepare_reference"):
                    reference = F.prepare(state, raw, req)
                ctx.stage = "compute"
                actual = F.compute_pair(state, req, query, reference, observation,
                                        scalar_control=req["method"] == "ot_cosine")
                expected = (1.0 if req["method"] == "ot_cosine" else 2.0) if req["budget_percent"] == 5 else (
                    .5 if req["method"] == "ot_cosine" else 1.0)
                require(F.same_double(actual, expected), "analytic exact distance")
                qdims, rdims = ([1], [0]) if req["budget_percent"] == 5 else ([0, 1], [0, 1])
                qbits, rbits = ([1], [0]) if req["budget_percent"] == 5 else ([0, 1], [0, 2])
                assert_basis(query.array, qbits, np)
                assert_basis(reference.array, rbits, np)
                scalar_equal = None if req["method"] != "ot_cosine" else F.same_double(
                    float.fromhex(observation["scalar_distance_hex"]), expected)
                require(scalar_equal is not False, "log/scalar exact analytical control")
                equal = None
                if req["selection"] == "literal_all_tissue":
                    literal[req["method"]] = actual
                elif req["selection"] == "all_true" or req["budget_percent"] == 100:
                    equal = F.same_double(literal[req["method"]], actual)
                    require(equal, "literal selection control")
                cells.append({"request": dict(req), "observation": observation,
                    "query_preparation_id": query.preparation_id,
                    "reference_preparation_id": reference.preparation_id,
                    "query_prepared": array_observation(query.array, np),
                    "reference_prepared": array_observation(reference.array, np),
                    "expected_distance_hex": expected.hex(), "actual_distance_hex": float(actual).hex(),
                    "expected_equal": True, "selected_rows": {"query": qdims, "reference": rdims},
                    "literal_equal": equal, "scalar_equal": scalar_equal})
            except BaseException as exc:
                pair_primary = exc
                raise
            finally:
                if reference is not None:
                    ledger.release(reference.reservation, pair_primary)
                if raw_token is not None:
                    ledger.release(raw_token, pair_primary)
                reference = raw = None
        ctx.stage = "decode"
        before = state.memory.live
        matrix = values = info = None
        baseline_token = None
        baseline_primary = None
        try:
            with ledger.operation("load_baseline", "baseline_load"):
                info = F.load_baseline(state, "ot_cosine")
            matrix, rows, cols, baseline_token = info
            values = matrix.to_numpy(copy=False)
            require(type(matrix) is F.pd.DataFrame and list(matrix.index) == list(matrix.columns) == ["1", "2"]
                    and matrix.index.dtype == matrix.columns.dtype == np.dtype("object")
                    and matrix.index.name is matrix.columns.name is None and values.shape == (2, 2)
                    and values.dtype.str == "<f8" and rows == cols == {"1": 0, "2": 1}, "tiny baseline axes/dtype")
            hexes = [[float(values[i, j]).hex() for j in range(2)] for i in range(2)]
            require(hexes == BASELINE_HEX, "tiny baseline exact FP64 bits")
            descriptor = descriptor_for(ctx.fixture_dir, "baseline.pkl")
            consumed = R.ConsumptionRecord("baseline:synthetic:ot_cosine", "baseline.pkl", descriptor["path"],
                descriptor["size"], descriptor["sha256"],
                ("generated_original_pickle_bytes", "unchanged_baseline_dataframe_decoder",
                 "exact_string_axes_2x2_FP64_values", "working_reservation_accounted"))
            baseline = {
                "method": "ot_cosine", "asset_id": consumed.asset_id, "descriptor": descriptor,
                "consumption": F._consumption_json(consumed), "decoder": "unchanged_F.load_baseline",
                "load_calls": 1, "raw_ids": ["1", "2"], "index": ["1", "2"], "columns": ["1", "2"],
                "row_lookup": rows, "column_lookup": cols, "shape": [2, 2], "dtype": "<f8",
                "strides": list(values.strides), "c_contiguous": bool(values.flags.c_contiguous),
                "f_contiguous": bool(values.flags.f_contiguous), "values_hex": hexes, "expected_equal": True,
                "reservation_bytes": state.memory.retained[baseline_token], "reservation_released": False,
                "working_live_before": before, "working_live_after": state.memory.live,
                "scope": "synthetic_decoder_only_not_baseline_equivalence",
            }
            require(baseline["reservation_bytes"] == descriptor["size"]+4192, "baseline actual reservation")
        except BaseException as exc:
            baseline_primary = exc
            raise
        finally:
            if baseline_token is not None:
                ledger.release(baseline_token, baseline_primary)
            matrix = values = info = None
        baseline["reservation_released"] = True
        baseline["working_live_after"] = state.memory.live
        require(state.memory.live == before, "baseline released without dropping query ownership")
        require(len(decodes) == 9 and len(state.preparations) == 16 and not state.warnings, "exercise exact counts")
        return {
            **ctx.envelope, "scope": "tiny_generated_basis_vectors_only", "fixture_directory": ctx.fixture_dir.record,
            "fixture_manifest": ctx.fixture_dir.files["fixture_manifest.json"],
            "bundle_expected": record_projection(state.bundle.expected), "requests": requests(),
            "decode_records": decodes, "preparations": list(state.preparations), "cells": cells,
            "raw_consumptions": raw_consumptions, "numpy_policy_before": policy, "numpy_policy_after": np.geterr(),
            "dtype_evidence": {"directly_observed": {"raw_query": "<f4", "raw_reference": "<f4",
                               "prepared_ot": "<f4", "prepared_bob": "|u1"},
                               "frozen_call_validated": {"cost": "<f4", "masses": "<f8", "distance": "<f8"},
                               "declared": {"lp_cost": "float64"}},
            "working_memory": state.memory.summary(), "warnings": [], "checks": [
                {"name": "analytic_exact_values", "passed": True}], "limitations": list(NEW_LIMITATIONS),
            "baseline": baseline, "reservations": ledger.events,
        }
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            unchanged = np.geterr() == policy
            np.seterr(**policy)
            require(unchanged, "NumPy policy unexpectedly changed")
        except BaseException as extra:
            if primary is None:
                primary = extra
            else:
                add_secondary(primary, extra, "NumPy policy restore")
        if primary is not None:
            cleanup_exercise(state, ledger, primary)


class RuntimeLeases:
    def __init__(self):
        self.entries, self.guards, self.bindings = [], [], []

    def enter_context(self, context):
        value = context.__enter__()
        self.entries.append(context)
        return value

    def close(self, primary=None):
        while self.entries:
            context = self.entries.pop()
            try:
                context.__exit__(None if primary is None else type(primary), primary,
                                 None if primary is None else primary.__traceback__)
            except BaseException as extra:
                if primary is None:
                    primary = extra
                else:
                    add_secondary(primary, extra, "runtime lease finalization")
            if self.entries and self.entries[-1] in self.guards:
                guard = self.entries.pop()
                try:
                    if primary is None:
                        self.bindings.extend(guard.final_bindings())
                except BaseException as extra:
                    if primary is None:
                        primary = extra
                    else:
                        add_secondary(primary, extra, "runtime guard seal")
                try:
                    guard.__exit__(None if primary is None else type(primary), primary,
                                   None if primary is None else primary.__traceback__)
                except BaseException as extra:
                    if primary is None:
                        primary = extra
                    else:
                        add_secondary(primary, extra, "runtime guard close")
        if primary is not None:
            raise primary


def validate_consumption(value):
    keys(value, ("asset_id", "logical_name", "consumer_path", "producer_size", "producer_sha256",
                 "semantic_checks"), "ConsumptionRecord projection")
    identifier(value["asset_id"])
    text(value["logical_name"])
    path_text(value["consumer_path"])
    integer(value["producer_size"])
    digest(value["producer_sha256"])
    for v in array(value["semantic_checks"], 16):
        text(v)


def validate_asset_binding(value):
    keys(value, ("asset_id", "role", "path", "resolved_path", "size", "sha256", "device", "inode"),
         "actual AssetGuard binding")
    for k in ("asset_id", "role"):
        identifier(value[k])
    for k in ("path", "resolved_path"):
        path_text(value[k])
    for k in ("size", "device"):
        integer(value[k])
    integer(value["inode"], 1)
    digest(value["sha256"])


def fixture_identity(name):
    require(name in (*DATA_NAMES, *CONTROL_NAMES), "guard-bound fixture leaf")
    if name in ("query.pkl", "reference.pkl"):
        return ("feature:probe_q" if name == "query.pkl" else "feature:probe_r"), "mosaic_feature"
    if name == "baseline.pkl":
        return "baseline:synthetic:ot_cosine", "baseline_matrix"
    return "fixture:"+name, "fixture_npy" if name.endswith(".npy") else "fixture_control"


def validate_fixture_bindings(value, manifest, directory):
    require(len(array(value, 17)) == 17, "17 guard-bound fixture assets")
    expected = {fixture_identity(name)[0]: (name, fixture_identity(name)[1])
                for name in (*DATA_NAMES, *CONTROL_NAMES)}
    seen = set()
    for asset in value:
        validate_asset_binding(asset)
        aid = asset["asset_id"]
        require(aid in expected and aid not in seen, "unique exact fixture asset")
        seen.add(aid)
        name, role = expected[aid]
        path = str(Path(directory)/name)
        require(asset["role"] == role and asset["path"] == asset["resolved_path"] == path,
                "fixture asset/role/path membership")
        file = manifest["files"][name]
        require(asset["size"] == file["size"] and asset["sha256"] == file["sha256"]
                and (asset["device"], asset["inode"]) == (file["st_dev"], file["st_ino"]),
                "guard-to-original fixture")
    require(seen == set(expected), "complete exact fixture guard membership")


def validate_phases(value, allowed=PHASES):
    array(value, len(allowed))
    require(bool(value) and value == [p for p in allowed if p in value], "ordered unique phase set")


def validate_software(value, envelope, catalog, private, prior):
    keys(value, (*envelope, "scope", "files", "aliases", "roles", "modules", "executable_mappings",
                 "namespaces", "generated", "native_pools", "version_metadata",
                 "historical_comparisons", "historical_aliases", "historical_unobserved", "counts",
                 "limitations", "fileless_objects", "runtime_bridge"), "SoftwareMap")
    validate_envelope(value, envelope)
    require(value["scope"] == "observed_startup_and_tiny_exercise_files_only"
            and value["limitations"] == NEW_LIMITATIONS, "software evidence scope")
    files = {}
    for v in array(value["files"], 16384):
        keys(v, ("asset_id", "binding", "first_phase", "phases", "required", "final_binding"), "Software")
        validate_bind(v["binding"])
        require(v["binding"] == v["final_binding"] and v["asset_id"] == software_id(v["binding"]["path"])
                and v["asset_id"] not in files, "actual content identity/final binding")
        validate_phases(v["phases"])
        require(v["first_phase"] == v["phases"][0], "first observation phase")
        boolean(v["required"])
        files[v["asset_id"]] = v
    aliases = {}
    for v in array(value["aliases"], 32768):
        keys(v, ("lexical_path", "asset_id", "phases"), "Alias")
        path_text(v["lexical_path"])
        validate_phases(v["phases"])
        require(v["asset_id"] in files and v["lexical_path"] not in aliases, "lexical alias identity")
        aliases[v["lexical_path"]] = v["asset_id"]
    require(set(aliases.values()) == set(files), "all acquired contents have actual aliases")
    for v in array(value["roles"], 65536):
        keys(v, ("phase", "role", "asset_id", "reported_path", "occurrence"), "Role")
        require(v["phase"] in PHASES and aliases.get(v["reported_path"]) == v["asset_id"],
                "role occurrence actual alias")
        text(v["role"])
        integer(v["occurrence"])
    keys(value["modules"], PHASES, "three module phases")
    module_counts, observed_modules, fileless_expected, namespace_expected = {}, {}, {}, set()
    for phase in PHASES:
        entries = array(value["modules"][phase], 8192)
        require([v["key"] for v in entries] == sorted({v["key"] for v in entries}), "module order/unique")
        module_counts[phase] = len(entries)
        for v in entries:
            keys(v, ("key", "name", "file", "origin", "search_locations", "cached_path", "phase", "kind",
                     "file_asset_id", "origin_asset_id", "cached_asset_id", "cache_state"), "Module")
            require(v["phase"] == phase, "module phase")
            shape = {k: v[k] for k in ("key", "name", "file", "origin", "search_locations", "cached_path")}
            require(v["kind"] == classify_module(shape, catalog, private["ordinary_namespaces"],
                                                prior["modules"]), "closed module representation")
            observed_modules[phase, v["key"]] = v
            if v["kind"] in ("recorded_fileless_object", "recorded_empty_location_object"):
                fileless_expected[phase, v["key"]] = v
            if v["kind"] == "torch_namespace":
                namespace_expected.add(v["key"])
            for name in ("file_asset_id", "origin_asset_id", "cached_asset_id"):
                require(v[name] is None or v[name] in files, "module file linkage")
            if v["file_asset_id"] is not None:
                require(aliases.get(v["file"]) == v["file_asset_id"], "module lexical source")
            expected_file = v["file"] is not None and v["kind"] not in ("torch_namespace", "generated")
            require((v["file_asset_id"] is not None) == expected_file, "kind-specific module source")
            if v["kind"] == "file":
                require(v["origin_asset_id"] == v["file_asset_id"] == aliases.get(v["origin"])
                        and v["origin_asset_id"] is not None, "file/origin actual content aliases")
            else:
                require(v["origin_asset_id"] is None, "non-file origin has no invented asset")
            if expected_file:
                require(len([r for r in value["roles"] if r["phase"] == phase and
                        r["role"] == "module:"+v["key"] and r["asset_id"] == v["file_asset_id"]
                        and r["reported_path"] == v["file"]]) == 1, "module source role")
            require(v["cache_state"] in ("present", "absent", "not_declared"), "cache state")
            if v["cached_path"] is not None:
                path_text(v["cached_path"])
            require((v["cached_path"] is None) == (v["cache_state"] == "not_declared")
                    and (v["cached_asset_id"] is not None) == (v["cache_state"] == "present"),
                    "cache presence precision")
            if v["cached_asset_id"] is not None:
                require(aliases.get(v["cached_path"]) == v["cached_asset_id"] and len([
                    r for r in value["roles"] if r["phase"] == phase and
                    r["role"] == "cache:"+v["key"] and r["asset_id"] == v["cached_asset_id"]
                    and r["reported_path"] == v["cached_path"]]) == 1, "actual cache alias/role")
                require(phase in files[v["cached_asset_id"]]["phases"] and any(
                    a["lexical_path"] == v["cached_path"] and phase in a["phases"]
                    for a in value["aliases"]), "actual cache acquired phase")
            if v["kind"] in ("recorded_fileless_object", "recorded_empty_location_object",
                             "authenticated_entry"):
                require(v["cached_path"] is v["cached_asset_id"] is None, "non-file cache provenance")
    bridges = {}
    for v in array(value["runtime_bridge"], 152):
        keys(v, ("software_asset_id", "runtime_asset_id", "canonical_path", "binding",
                 "final_binding", "phases"), "RuntimeBridge")
        require(v == runtime_bridge(v["binding"], v["phases"])
                and v["software_asset_id"] in files and files[v["software_asset_id"]]["binding"] == v["binding"]
                and v["runtime_asset_id"] not in bridges, "typed bridge actual contents")
        validate_phases(v["phases"], ("before", "after"))
        bridges[v["runtime_asset_id"]] = v
    pool_counts = {p: 0 for p in ("before", "after")}
    for v in array(value["native_pools"], 128):
        keys(v, ("phase", "ordinal", "implementation_asset_id", "software_asset_id", "reported"),
             "original native pool occurrence")
        require(v["phase"] in pool_counts and v["ordinal"] == pool_counts[v["phase"]], "pool ordinal")
        pool_counts[v["phase"]] += 1
        require(pool_counts[v["phase"]] <= 64, "pool phase cap")
        validate_pool(v["reported"])
        bridge = bridges[v["implementation_asset_id"]]
        require(bridge["software_asset_id"] == v["software_asset_id"]
                and aliases.get(v["reported"]["filepath"]) == v["software_asset_id"], "pool ID bridge")
    require([v["distribution"] for v in array(value["version_metadata"], 6)] ==
            ["numpy", "scipy", "scikit-learn", "POT", "pandas", "threadpoolctl"], "six distributions")
    for v in value["version_metadata"]:
        keys(v, ("distribution", "required_version", "observed_version", "metadata_asset_id",
                 "record_asset_id", "metadata_path", "record_path", "distribution_root", "semantic_checks"),
             "VersionMetadata")
        require(v["metadata_asset_id"] == aliases.get(v["metadata_path"])
                and v["record_asset_id"] == aliases.get(v["record_path"])
                and Path(v["metadata_path"]).parent == Path(v["record_path"]).parent
                == Path(v["distribution_root"]), "metadata original-file linkage")
        text(v["observed_version"])
        expected_version = {"numpy": "2.2.4", "scipy": "1.16.0", "scikit-learn": "1.7.0",
                            "POT": "0.9.6.post1"}.get(v["distribution"])
        require(v["required_version"] == expected_version and
                (expected_version is None or v["observed_version"] == expected_version), "version requirement")
        require(v["semantic_checks"] == ["single_wheel_metadata", "name_matches_distribution",
                "metadata_version_matches_getter", "original_metadata_and_record_bytes"], "metadata checks")
    map_counts = {p: 0 for p in PHASES}
    for v in array(value["executable_mappings"], 24576):
        keys(v, ("phase", "path", "asset_id"), "Map")
        require(v["phase"] in PHASES and aliases.get(v["path"]) == v["asset_id"], "map lexical linkage")
        map_counts[v["phase"]] += 1
        require(map_counts[v["phase"]] <= 8192, "map phase cap")
    fileless_seen = set()
    for v in array(value["fileless_objects"], 285):
        keys(v, ("phase", "key", "name", "object_id", "representation", "historical_catalog_sha256",
                 "creation_origin", "defining_provider_asset_id", "exposure"), "Fileless")
        require(v["phase"] in PHASES and v["creation_origin"] == "unresolved"
                and v["defining_provider_asset_id"] is None, "no invented creator")
        pair = v["phase"], v["key"]
        require(pair in fileless_expected and pair not in fileless_seen
                and v["name"] == fileless_expected[pair]["name"], "observed-only fileless bijection")
        fileless_seen.add(pair)
        identifier(v["object_id"])
        require((v["representation"] == "known_all_null" and catalog.get(v["key"]) == v["name"]
                 and v["historical_catalog_sha256"] == CATALOG_HASH) or
                (v["representation"] == "known_empty_locations" and v["key"] == v["name"] == "six.moves"
                 and v["historical_catalog_sha256"] == SIX_HASH), "closed historical shape")
        e = v["exposure"]
        keys(e, ("status", "provider_key", "provider_source_asset_id", "attribute_path", "relation"), "Exposure")
        if e["status"] == "unresolved":
            require(e["provider_key"] is e["provider_source_asset_id"] is e["relation"] is None
                    and e["attribute_path"] == [], "unresolved exposure")
        else:
            provider, path = _exposure_rule(v["key"])
            require(e["status"] == "runtime_attribute_identity" and e["provider_key"] == provider
                    and e["attribute_path"] == path and e["relation"] == "exposes_same_object"
                    and e["provider_source_asset_id"] in files, "bounded exposure route not creator proof")
            require((v["phase"], provider) in observed_modules and
                    observed_modules[v["phase"], provider]["file_asset_id"] ==
                    e["provider_source_asset_id"], "observed exposure provider content")
    require(fileless_seen == set(fileless_expected), "every observed fileless sidecar")
    namespace_seen = set()
    for namespace in array(value["namespaces"], 2):
        keys(namespace, ("name", "reported_file", "defining_module", "defining_asset_id", "torch_asset_id"),
             "fixed Torch namespace provenance")
        require(namespace["name"] in TORCH_NAMESPACES and
                (namespace["reported_file"], namespace["defining_module"]) == TORCH_NAMESPACES[namespace["name"]],
                "exact Torch namespace pair")
        require(namespace["name"] in namespace_expected and namespace["name"] not in namespace_seen,
                "observed-only namespace provenance")
        namespace_seen.add(namespace["name"])
        pins = {
            "torch._ops": (55360, "7df9e7998b6c2343f4ae5470a7b21d458236ba9e8d27035c14a670441308028c"),
            "torch._classes": (1721, "2a3dd93d72e9f19450670b89f3a57b5b5adf245709f6a49a551bbfad33c434bf"),
            "torch": (95190, "e1b5bf4151bd246a21e3ac5318f6edfcf2e6ef0fd4f233bfb77a264698ec1319"),
        }
        for role, name in (("defining_asset_id", namespace["defining_module"]), ("torch_asset_id", "torch")):
            binding = files[namespace[role]]["binding"]
            require((binding["size"], binding["sha256"]) == pins[name], "namespace defining-source pin")
            for phase in PHASES:
                if (phase, namespace["name"]) in observed_modules:
                    require((phase, name) in observed_modules and
                            observed_modules[phase, name]["file_asset_id"] == namespace[role],
                            "observed namespace source relation")
    require(namespace_seen == namespace_expected, "every observed Torch namespace sidecar")
    classifications = {}
    for v in array(value["historical_comparisons"], 16384):
        keys(v, ("asset_id", "canonical_path", "classification", "prior_matches", "required"), "HistoricalComparison")
        require(v["asset_id"] in files and v["canonical_path"] == files[v["asset_id"]]["binding"]["path"]
                and v["classification"] in ("matched", "new", "changed")
                and not (v["classification"] == "changed" and v["required"]), "historical classification")
        require(v["asset_id"] not in classifications, "unique historical comparison")
        classifications[v["asset_id"]] = v["classification"]
        for match in array(v["prior_matches"], 9):
            keys(match, ("inventory", "prior_path", "prior_size", "prior_sha256", "match_kind"), "PriorMatch")
            require(match["inventory"] in ("original_profile", "cpu_capture", "cpu_private_binding")
                    and match["match_kind"] in ("recorded_canonical", "recorded_alias", "exact_path"), "prior metadata scope")
            path_text(match["prior_path"])
            integer(match["prior_size"])
            digest(match["prior_sha256"])
    require(set(classifications) == set(files), "all contents reconciled")
    for v in array(value["historical_aliases"], 32768):
        keys(v, ("inventory", "prior_path", "asset_id"), "HistoricalAlias")
        require(v["inventory"] in ("original_profile", "cpu_capture", "cpu_private_binding")
                and v["asset_id"] in files, "prior alias domain")
        path_text(v["prior_path"])
    for v in array(value["historical_unobserved"], 20000):
        keys(v, ("inventory", "path", "size", "sha256"), "HistoricalUnobserved")
        require(v["inventory"] in ("original_profile", "cpu_capture", "cpu_private_binding"), "historical-only inventory")
        validate_desc({k: v[k] for k in ("path", "size", "sha256")})
    counts = {
        "canonical_files": len(files), "lexical_alias_paths": len(aliases),
        "role_occurrences": len(value["roles"]), "unique_content_bytes": sum(v["binding"]["size"] for v in files.values()),
        **{p+"_modules": module_counts[p] for p in PHASES}, **{p+"_maps": map_counts[p] for p in PHASES},
        **{p+"_native_pools": pool_counts[p] for p in pool_counts},
        "required_files": sum(v["required"] for v in files.values()),
        **{c+"_files": sum(v == c for v in classifications.values()) for c in ("new", "changed", "matched")},
        "historical_unobserved_records": len(value["historical_unobserved"]),
        "fileless_object_records": len(value["fileless_objects"]),
        "unresolved_creation_origins": len(value["fileless_objects"]), "runtime_bridge_rows": len(bridges),
    }
    require(value["counts"] == counts and counts["unique_content_bytes"] <= 16*1024**3, "independent software totals")
    return files, aliases, bridges


def capture(ctx):
    namespace = vars(ctx)
    if "reservation_ledger" not in namespace:
        ctx.exercise_state = ctx.reservation_ledger = None
    primary = None
    try:
        return _capture(ctx)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if ctx.reservation_ledger is not None and not ctx.reservation_ledger.cleanup_started:
            cleanup_exercise(ctx.exercise_state, ctx.reservation_ledger, primary)


def _capture(ctx):
    ctx.stage = "bootstrap"
    ctx.open_operations()
    require(ctx.operations_dir.names() <= {"capture.stdout", "capture.stderr"}, "initial operations namespace")
    ctx.capture_dir = OwnedDirectory(ctx.private["capture_dir"], ctx.private["owner_uid"], ctx.deadline, create=True)
    ctx.fixture_dir = OwnedDirectory(ctx.private["fixture_dir"], ctx.private["owner_uid"], ctx.deadline, create=True)
    ctx.authenticate_sources()
    B = ctx.load_bootstrap()
    observation = SoftwareObservation(ctx)
    ctx.stage = "site"
    startup = startup_gate(ctx, B, observation)
    ctx.stage = "import"
    F = importlib.import_module(DIRECT_MODULES["runner"])
    R = importlib.import_module(DIRECT_MODULES["mosaic_reader"])
    paths = observation.first_party(F)
    observation.versions(F)
    leases = RuntimeLeases()
    primary, state = None, None
    stack = ExitStack()
    try:
        ctx.stage = "profile_before"
        before = observation.runtime(F, "before", leases)
        observation.observe("before")
        transient = generated_source(ctx, B, observation)
        provenance = B.namespace_provenance(
            {"inventory": LegacyReadAdapter(ctx.inventory)}, {v["key"]: v for v in observation.modules["before"]})
        observation.namespaces = [{
            "name": n, "reported_file": v["reported_file"], "defining_module": v["defining_module"],
            "defining_asset_id": software_id(v["sources"][v["defining_module"]]["path"]),
            "torch_asset_id": software_id(v["sources"]["torch"]["path"]),
        } for n, v in sorted(provenance.items())]
        ctx.stage = "fixture"
        state, fixture_guard = fixture(ctx, F, R, stack)
        ledger = ReservationLedger(state.memory)
        ctx.exercise_state, ctx.reservation_ledger = state, ledger
        exercises = exercise(ctx, F, R, state, ledger)
        ctx.stage = "profile_after"
        after = observation.runtime(F, "after", leases)
        observation.observe("after")
        require(np_policy_equal(before, after, exercises), "restored policy profile linkage")
        fixture_bindings = list(fixture_guard.final_bindings())
    except BaseException as exc:
        primary = exc
    finally:
        try:
            stack.__exit__(None if primary is None else type(primary), primary,
                           None if primary is None else primary.__traceback__)
        except BaseException as extra:
            if primary is None:
                primary = extra
            else:
                add_secondary(primary, extra, "fixture guard close")
        try:
            leases.close(primary)
        except BaseException as extra:
            if primary is None:
                primary = extra
            elif extra is not primary:
                add_secondary(primary, extra, "runtime close")
    if primary is not None:
        raise primary
    ctx.stage = "final_inputs"
    transient.sweep()
    ctx.inventory.sweep()
    software = observation.finish()
    validate_software(software, ctx.envelope, observation.catalog, ctx.private, ctx.prior["cpu_capture"])
    ctx.stage = "write_capture"
    output = ctx.capture_dir
    output.write("startup.json", startup)
    output.write("runtime_before.json", before)
    output.write("runtime_after.json", after)
    output.write("software_map.json", software, ceiling=32*MIB)
    output.write("exercises.json", serialize_exercises(exercises, state, ledger), raw=True)
    bindings = {
        **ctx.envelope, "source_originals": observation.source_descriptors,
        "source_final": [dict(ctx.inventory.aliases[v["path"]]) for v in observation.source_descriptors],
        "implementation_bindings": leases.bindings, "software_map_binding": {
            k: output.files["software_map.json"][k] for k in ("size", "sha256")},
        "fixture_directory": ctx.fixture_dir.record, "fixture_manifest": ctx.fixture_dir.files["fixture_manifest.json"],
        "fixture_bindings": fixture_bindings, "raw_consumptions": exercises["raw_consumptions"],
        "prior_inventory_originals": ctx.spec["prior_inventory_assets"],
        "prior_inventory_final": {name: {k: ctx.inventory.aliases[d["path"]][k] for k in ("path", "size", "sha256")}
                                  for name, d in ctx.spec["prior_inventory_assets"].items()},
        "executing_code_paths": paths, "limitations": list(NEW_LIMITATIONS),
        "baseline_consumption": exercises["baseline"]["consumption"],
    }
    output.write("input_bindings.json", bindings)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    summary = {
        **ctx.envelope, "disposition": "captured_pending_exit_and_independent_review",
        "runtime_before_after_equal": before == after, "numpy_error_policy_restored": True,
        "exercise_counts": {"requests": 8, "ot_log_calls": 4, "ot_scalar_calls": 4, "bob_calls": 4,
                            "raw_decodes": 9, "preparations": 16, "max_pair_cells": 4, "baseline_decoder_calls": 1},
        "software_counts": software["counts"],
        "usage": {"wall_ns": time.monotonic_ns()-ctx.started[1], "user_seconds_hex": float(usage.ru_utime).hex(),
                  "system_seconds_hex": float(usage.ru_stime).hex(), "maxrss_kib": int(usage.ru_maxrss)},
        "pilot_runtime_accepted": False, "real_load_feasibility_established": False,
        "checks": [{"name": "fixed_tiny_exercise_counts", "passed": True}], "limitations": list(NEW_LIMITATIONS),
    }
    output.write("summary.json", summary)
    capture_record = {
        **ctx.envelope, "status": "captured_pending_actual_process_exit", "capture_pid": os.getpid(),
        "hostname": "noderome251", "capture_directory": output.record, "fixture_directory": ctx.fixture_dir.record,
        "fixture_manifest": ctx.fixture_dir.files["fixture_manifest.json"], "artifacts": dict(output.files),
        "clock": clock(ctx.started, ctx.deadline), "generated_source": ctx.operations_dir.files["generated_source.py"],
        "source_authority": "trusted_source_buffer_bootstrap", "checks": [
            {"name": "original_byte_and_namespace_readback", "passed": True}], "limitations": list(NEW_LIMITATIONS),
    }
    validate_clock(capture_record["clock"], True)
    output.write("capture.json", capture_record)
    require(tuple(sorted(sys.modules)) == tuple(v["key"] for v in observation.modules["after"]),
            "serialization expanded module scope")
    require(read_maps() == [v["path"] for v in observation.maps if v["phase"] == "after"], "late mapping expansion")
    ctx.inventory.sweep()
    transient.sweep()
    require(output.names() == set(CAPTURE_NAMES) and ctx.fixture_dir.names() == set(FIXTURE_NAMES), "capture namespaces")
    for directory in (output, ctx.fixture_dir):
        for name, record in directory.files.items():
            directory.read(name, record, ceiling=32*MIB if name == "software_map.json" else 8*MIB)
    return {
        **ctx.envelope, "kind": "capture_handoff", "capture_pid": os.getpid(), "hostname": "noderome251",
        "capture_directory": output.record, "fixture_directory": ctx.fixture_dir.record,
        "fixture_manifest": ctx.fixture_dir.files["fixture_manifest.json"], "artifacts": dict(output.files),
        "generated_source": ctx.operations_dir.files["generated_source.py"],
    }


class LegacyReadAdapter:
    """Only the fixed namespace helper's three pinned defining sources."""
    def __init__(self, inventory):
        self.inventory = inventory

    def read(self, path, limit=None, *, expected, retain=False):
        allowed = {
            (95190, "e1b5bf4151bd246a21e3ac5318f6edfcf2e6ef0fd4f233bfb77a264698ec1319"),
            (55360, "7df9e7998b6c2343f4ae5470a7b21d458236ba9e8d27035c14a670441308028c"),
            (1721, "2a3dd93d72e9f19450670b89f3a57b5b5adf245709f6a49a551bbfad33c434bf"),
        }
        require((expected["size"], expected["sha256"]) in allowed and retain is False, "fixed namespace source pins")
        return self.inventory.read(path, expected=expected, cap=2*MIB)


def np_policy_equal(before, after, exercise_record):
    return before["profile"]["numpy_numerics"]["error_policy"] == after["profile"]["numpy_numerics"]["error_policy"] \
        == exercise_record["numpy_policy_before"] == exercise_record["numpy_policy_after"]


def runtime_links(runtime, phase, software):
    validate_runtime(runtime)
    bridges = {v["runtime_asset_id"]: v for v in software["runtime_bridge"]}
    aliases = {v["lexical_path"]: v["asset_id"] for v in software["aliases"]}
    roles = [v for v in software["roles"] if v["phase"] == phase]
    for name, path in runtime["observed_paths"].items():
        require(path["asset_id"] in bridges, "observed implementation bridge")
        bridge = bridges[path["asset_id"]]
        require(phase in bridge["phases"] and aliases.get(path["path"]) == bridge["software_asset_id"],
                "original runtime lexical alias to typed bridge")
        occurrences = [v for v in roles if v["role"] == name]
        require(len(occurrences) == 1 and occurrences[0]["asset_id"] == bridge["software_asset_id"]
                and occurrences[0]["reported_path"] == path["path"], "every original runtime logical role")
    for name, implementation in runtime["profile"]["base_profile"]["implementation"].items():
        require(implementation["path"] == runtime["observed_paths"]["base:"+name]["asset_id"]
                and implementation["path"] in bridges, "base implementation role/bridge")
    assets = runtime["profile"]["implementation_assets"]
    require([v["asset_id"] for v in assets] == sorted(v["asset_id"] for v in assets), "stock sorted implementations")
    for asset in assets:
        bridge = bridges[asset["asset_id"]]
        require(all(asset[k] == bridge["binding"][k] for k in ("size", "sha256")), "authenticated runtime content")
    pools = [v for v in software["native_pools"] if v["phase"] == phase]
    normalized = [{**v["reported"], "filepath": v["implementation_asset_id"]} for v in pools]
    normalized.sort(key=lambda v: encoded(v, 65536))
    same_json(normalized, runtime["profile"]["base_profile"]["native_pools"])


def validate_handoff(value, envelope, mode):
    common = (*envelope, "kind", "hostname", "capture_pid", "capture_directory", "fixture_directory", "artifacts")
    keys(value, (*common, "fixture_manifest", "generated_source") if mode == "capture" else
         (*common, "verifier_pid", "verification", "capture_witness"), "fixed handoff")
    validate_envelope(value, envelope)
    require(value["kind"] == ("capture_handoff" if mode == "capture" else "verification_handoff")
            and value["hostname"] == "noderome251", "handoff mode/host")
    integer(value["capture_pid"], 1)
    if mode == "verify":
        integer(value["verifier_pid"], 1)
        require(value["verifier_pid"] != value["capture_pid"], "fresh verifier")
        validate_file(value["verification"], "verification.json", 8*MIB)
        validate_desc(value["capture_witness"], 8*MIB)
    else:
        validate_file(value["fixture_manifest"], "fixture_manifest.json", 8*MIB)
        validate_file(value["generated_source"], "generated_source.py", 2*MIB)
    for name in ("capture_directory", "fixture_directory"):
        validate_dir(value[name])
    for name in CAPTURE_NAMES:
        validate_file(value["artifacts"][name], name, 32*MIB if name == "software_map.json" else 8*MIB)
    keys(value["artifacts"], CAPTURE_NAMES, "capture artifact membership")


PREPARATION_KEYS = (
    "preparation_id", "slide_index", "slide_id", "method", "selection", "budget_percent",
    "query_fold", "raw_rows", "selected_rows", "raw_dtype", "raw_shape", "raw_strides",
    "raw_c_contiguous", "raw_f_contiguous", "selected_dtype", "selected_shape", "selected_strides",
    "selected_c_contiguous", "selected_f_contiguous", "prepared_dtype", "prepared_shape",
    "prepared_strides", "prepared_c_contiguous", "prepared_f_contiguous",
)
OBSERVATION_KEYS = (
    "request_id", "query_index", "reference_index", "reference_id", "query_preparation_id",
    "reference_preparation_id", "same_patient_excluded", "zero_query", "zero_reference", "observation_status",
    "computed_valid", "distance_hex", "cosine_min_hex", "cosine_max_hex", "cosine_negative_count",
    "solver_result_code", "solver_warning_is_none", "captured_warning_count", "scalar_distance_hex",
    "scalar_log_comparison", "baseline_distance_hex", "baseline_comparison", "literal_comparison",
    "candidate_rank", "baseline_rank", "baseline_distance_comparison", "baseline_rank_comparison",
)


def same_json(left, right):
    require(encoded(left, 32*MIB) == encoded(right, 32*MIB), "typed canonical semantic equality")


def array_record(value, dtype, dimensions):
    keys(value, ("dtype", "shape", "strides", "c_contiguous", "f_contiguous",
                 "c_order_bytes", "c_order_sha256", "nonzero"), "ArrayObservation")
    width = 1536 if dtype == "<f4" else 1535
    require(value["dtype"] == dtype, "actual dtype versus declared")
    same_json(value["shape"], [len(dimensions), width])
    for v in array(value["strides"], 2):
        integer(v)
    require(len(value["strides"]) == 2, "stride rank")
    boolean(value["c_contiguous"])
    boolean(value["f_contiguous"])
    data = bytearray(len(dimensions)*width*(4 if dtype == "<f4" else 1))
    atoms = []
    for row, column in enumerate(dimensions):
        if dtype == "<f4":
            struct.pack_into("<f", data, (row*width+column)*4, 1.0)
        else:
            data[row*width+column] = 1
        atoms.append({"row": row, "column": column, "float_hex": "0x1.0000000000000p+0"
                      if dtype == "<f4" else None, "uint_value": None if dtype == "<f4" else 1})
    require(type(value["c_order_bytes"]) is int and value["c_order_bytes"] == len(data)
            and value["c_order_sha256"] == sha(data), "all zero/basis bytes analytical recount")
    same_json(value["nonzero"], atoms)


def validate_exercises(value, envelope):
    keys(value, (*envelope, "scope", "fixture_directory", "fixture_manifest", "bundle_expected", "requests",
                 "decode_records", "preparations", "cells", "raw_consumptions", "numpy_policy_before",
                 "numpy_policy_after", "dtype_evidence", "working_memory", "warnings", "checks",
                 "limitations", "baseline", "reservations"), "Exercises")
    validate_envelope(value, envelope)
    validate_reservations(value["reservations"], True)
    require([event["stage"] for event in value["reservations"]] ==
            [stage for _ in requests() for stage in ("query_event", "raw_reference", "prepare_reference")]
            + ["baseline_load"], "actual successful harness reservation operations")
    for event in value["reservations"]:
        if event["stage"] != "query_event":
            require(all(row["disposition"] == "released" for row in event["new_reservations"]),
                    "non-query reservations released before serialization")
    validate_success_accounting(value)
    require(value["scope"] == "tiny_generated_basis_vectors_only" and value["limitations"] == NEW_LIMITATIONS,
            "exercise scope")
    same_json(value["requests"], requests())
    validate_dir(value["fixture_directory"])
    validate_file(value["fixture_manifest"], "fixture_manifest.json", 8*MIB)
    for policy in ("numpy_policy_before", "numpy_policy_after"):
        validate_policy(value[policy])
    same_json(value["numpy_policy_before"], value["numpy_policy_after"])
    require(value["warnings"] == [], "native warning rejects success")
    validate_checks(value["checks"], 32, True)
    require(len(array(value["decode_records"], 9)) == len(array(value["raw_consumptions"], 9)) == 9,
            "nine actual decoder observations")
    decoded_counts = {0: 0, 1: 0}
    for entry, consumed in zip(value["decode_records"], value["raw_consumptions"]):
        keys(entry, ("slide_index", "decode_ordinal", "layout", "consumption", "raw"), "Decode")
        index = integer(entry["slide_index"], maximum=1)
        require(type(entry["decode_ordinal"]) is int and entry["decode_ordinal"] == decoded_counts[index],
                "actual per-slide decoder ordinal")
        decoded_counts[index] += 1
        array_record(entry["raw"], "<f4", [0, 1] if index == 0 else [0, 2])
        same_json(entry["raw"]["strides"], [6144, 4] if index == 0 else [4, 8])
        same_json(entry["layout"], {"dtype_str": "<f4", "shape": [2, 1536],
            "strides": entry["raw"]["strides"], "c_contiguous": index == 0, "f_contiguous": index == 1})
        validate_consumption(consumed)
        same_json(entry["consumption"], consumed)
        require(consumed["asset_id"] == ("feature:probe_q" if index == 0 else "feature:probe_r"), "raw asset identity")
        same_json(consumed["semantic_checks"], ["accepted_feature_bytes", "frozen_tuple_FP32_1536_decoder",
                  "complete_ordered_coordinate_slice_match", "decoded_layout_preserved"])
    require(decoded_counts == {0: 1, 1: 8}, "actual decoder accounting")
    prepared = {}
    for entry in array(value["preparations"], 16):
        keys(entry, PREPARATION_KEYS, "source-bound preparation24")
        pid = entry["preparation_id"]
        require(type(pid) is str and re.fullmatch("p[0-9]{10}", pid) is not None and pid not in prepared,
                "unique preparation event ID")
        prepared[pid] = entry
    require(len(prepared) == 16 and len(array(value["cells"], 8)) == 8, "exact preparation/pair totals")
    used = set()
    for req, cell in zip(requests(), value["cells"]):
        keys(cell, ("request", "observation", "query_preparation_id", "reference_preparation_id",
                    "query_prepared", "reference_prepared", "expected_distance_hex", "actual_distance_hex",
                    "expected_equal", "selected_rows", "literal_equal", "scalar_equal"), "ExerciseCell")
        same_json(cell["request"], req)
        ot = req["method"] == "ot_cosine"
        filtered = req["budget_percent"] == 5
        expected = (1.0 if ot else 2.0) if filtered else (.5 if ot else 1.0)
        require(cell["expected_distance_hex"] == cell["actual_distance_hex"] == expected.hex()
                and cell["expected_equal"] is True, "analytic exact distances")
        dims = ([1], [0]) if filtered else ([0, 1], [0, 2])
        rows = {"query": [1] if filtered else [0, 1], "reference": [0] if filtered else [0, 1]}
        same_json(cell["selected_rows"], rows)
        for index, side in enumerate(("query", "reference")):
            array_record(cell[side+"_prepared"], "<f4" if ot else "|u1", dims[index])
            pid = cell[side+"_preparation_id"]
            require(pid in prepared and pid not in used, "actual event consumed once")
            used.add(pid)
            p = prepared[pid]
            for key in ("method", "selection", "budget_percent", "query_fold"):
                same_json(p[key], req[key])
            require(type(p["slide_index"]) is int and p["slide_index"] == index
                    and p["slide_id"] == ("probe_q" if index == 0 else "probe_r")
                    and type(p["raw_rows"]) is int and p["raw_rows"] == 2
                    and type(p["selected_rows"]) is int and p["selected_rows"] == len(dims[index]),
                    "preparation input identity")
            for prefix in ("raw", "selected", "prepared"):
                dtype = "<f4" if prefix != "prepared" or ot else "|u1"
                require(p[prefix+"_dtype"] == dtype, "preparation exposed dtype")
                expected_shape = [2, 1536] if prefix == "raw" else [
                    len(dims[index]), 1535 if prefix == "prepared" and not ot else 1536]
                require(p[prefix+"_shape"] == json.dumps(expected_shape, separators=(",", ":")), "preparation shape")
                stride = strict_json(p[prefix+"_strides"].encode(), 65536)
                require(type(stride) is list and len(stride) == 2, "preparation stride rank")
                for atom in stride:
                    integer(atom)
                boolean(p[prefix+"_c_contiguous"])
                boolean(p[prefix+"_f_contiguous"])
                if prefix == "prepared":
                    a = cell[side+"_prepared"]
                    same_json(stride, a["strides"])
                    require(p[prefix+"_c_contiguous"] is a["c_contiguous"]
                            and p[prefix+"_f_contiguous"] is a["f_contiguous"], "prepared array/event linkage")
        obs = cell["observation"]
        keys(obs, (*OBSERVATION_KEYS, "solver_warning") if ot else OBSERVATION_KEYS, "unaltered PairObservation")
        expected_obs = dict(request_id=req["request_id"], query_index=0, reference_index=1,
            reference_id="probe_r", query_preparation_id=cell["query_preparation_id"],
            reference_preparation_id=cell["reference_preparation_id"], same_patient_excluded=False,
            zero_query=False, zero_reference=False, observation_status="valid_distance", computed_valid=True,
            distance_hex=expected.hex(), cosine_min_hex=(1.0 if filtered else 0.0).hex() if ot else None,
            cosine_max_hex=1.0.hex() if ot else None, cosine_negative_count=0 if ot else None,
            solver_result_code=1 if ot else None, solver_warning_is_none=True if ot else None,
            captured_warning_count=0, scalar_distance_hex=expected.hex() if ot else None,
            scalar_log_comparison="equal" if ot else "not_applicable", baseline_distance_hex=None,
            baseline_comparison="not_applicable", literal_comparison="not_applicable", candidate_rank=None,
            baseline_rank=None, baseline_distance_comparison="not_applicable", baseline_rank_comparison="not_applicable")
        if ot:
            expected_obs["solver_warning"] = None
        same_json(obs, expected_obs)
        require(cell["scalar_equal"] is (True if ot else None) and cell["literal_equal"] is
                (True if req["selection"] == "all_true" or req["budget_percent"] == 100 else None), "actual controls")
    baseline = value["baseline"]
    keys(baseline, ("method", "asset_id", "descriptor", "consumption", "decoder", "load_calls", "raw_ids",
                    "index", "columns", "row_lookup", "column_lookup", "shape", "dtype", "strides", "c_contiguous",
                    "f_contiguous", "values_hex", "expected_equal", "reservation_bytes", "reservation_released",
                    "working_live_before", "working_live_after", "scope"), "BaselineExercise")
    require(baseline["method"] == "ot_cosine" and baseline["asset_id"] == "baseline:synthetic:ot_cosine"
            and baseline["decoder"] == "unchanged_F.load_baseline" and type(baseline["load_calls"]) is int
            and baseline["load_calls"] == 1 and baseline["dtype"] == "<f8" and baseline["expected_equal"] is True
            and baseline["reservation_released"] is True
            and baseline["scope"] == "synthetic_decoder_only_not_baseline_equivalence", "one unchanged baseline decoder")
    for name in ("raw_ids", "index", "columns"):
        same_json(baseline[name], ["1", "2"])
    for name in ("row_lookup", "column_lookup"):
        same_json(baseline[name], {"1": 0, "2": 1})
    same_json(baseline["shape"], [2, 2])
    same_json(baseline["values_hex"], BASELINE_HEX)
    validate_desc(baseline["descriptor"], MIB)
    validate_consumption(baseline["consumption"])
    for name in ("reservation_bytes", "working_live_before", "working_live_after"):
        integer(baseline[name], maximum=8*MIB)
    require(baseline["reservation_bytes"] == baseline["descriptor"]["size"]+4192
            and baseline["working_live_before"] == baseline["working_live_after"], "baseline reservation accounting")
    same_json(baseline["consumption"]["semantic_checks"], [
        "generated_original_pickle_bytes", "unchanged_baseline_dataframe_decoder",
        "exact_string_axes_2x2_FP64_values", "working_reservation_accounted"])
    boolean(baseline["c_contiguous"])
    boolean(baseline["f_contiguous"])
    require(len(array(baseline["strides"], 2)) == 2, "baseline observed strides")
    for atom in baseline["strides"]:
        integer(atom)
    keys(value["working_memory"], ("peak_accounted_working_bytes", "limit_bytes", "basis"), "working buffers")
    integer(value["working_memory"]["peak_accounted_working_bytes"], maximum=8*MIB)
    require(type(value["working_memory"]["limit_bytes"]) is int and value["working_memory"]["limit_bytes"] == 8*MIB
            and value["working_memory"]["basis"] == "conservative_buffers_and_known_reservations_not_rss", "not RSS")
    same_json(value["dtype_evidence"], {
        "directly_observed": {"raw_query": "<f4", "raw_reference": "<f4", "prepared_ot": "<f4", "prepared_bob": "|u1"},
        "frozen_call_validated": {"cost": "<f4", "masses": "<f8", "distance": "<f8"}, "declared": {"lp_cost": "float64"}})


def verify(ctx):
    ctx.stage = "verify_inputs"
    require(socket.gethostname() == "noderome251" and os.getuid() == ctx.private["owner_uid"]
            and len(os.sched_getaffinity(0)) == 1, "fresh verifier host/UID/affinity")
    ctx.authenticate_sources()
    ctx.open_operations()
    require(ctx.operations_dir.names() <= set(OPERATION_NAMES[:6]) and
            {"capture.stdout", "capture.stderr", "capture_exit.json", "generated_source.py"} <=
            ctx.operations_dir.names(), "fresh verifier operations namespace")
    witness_path = path_text(ctx.args.capture_witness)
    require(witness_path == str(ctx.operations_dir.path/"capture_exit.json"), "fixed actual witness leaf")
    data, witness_bind = ctx.inventory.read(witness_path, expected={
        "size": os.stat(witness_path).st_size, "sha256": digest(ctx.args.capture_witness_sha256)}, cap=8*MIB, retain=True)
    witness = strict_json(data, canonical=True)
    validate_witness(witness, ctx.envelope, "capture", True)
    require(witness["authenticated_launch_body"] == ctx.private["supervisor_body"]
            and witness["operations_directory"] == ctx.operations_dir.record
            and os.getpid() not in (witness["supervisor_pid"], witness["outcome"]["pid"]), "trusted fresh witness")
    original = ctx.operations_dir.read("capture.stdout", witness["stdout"]["identity"], ceiling=16384)
    require(original.endswith(b"\n") and original.count(b"\n") == 1, "original single-line stdout")
    receipt = strict_json(original, 16384, True)
    validate_handoff(receipt, ctx.envelope, "capture")
    require(receipt["capture_pid"] == witness["outcome"]["pid"], "receipt versus actual capture process")
    for name in ("capture_directory", "fixture_directory"):
        require(receipt[name] == witness[name] and receipt[name]["path"] ==
                ctx.private["capture_dir" if name == "capture_directory" else "fixture_dir"], "original directory witness")
    ctx.capture_dir = OwnedDirectory(ctx.private["capture_dir"], ctx.private["owner_uid"],
                                     ctx.deadline, expected=receipt["capture_directory"])
    ctx.fixture_dir = OwnedDirectory(ctx.private["fixture_dir"], ctx.private["owner_uid"],
                                     ctx.deadline, expected=receipt["fixture_directory"])
    require(ctx.capture_dir.names() == set(CAPTURE_NAMES) and ctx.fixture_dir.names() == set(FIXTURE_NAMES),
            "unverified capture/18 fixture membership")
    artifacts = {}
    for name, record in receipt["artifacts"].items():
        ceiling = 32*MIB if name == "software_map.json" else 8*MIB
        artifacts[name] = strict_json(ctx.capture_dir.read(name, record, ceiling=ceiling), ceiling, True)
        ctx.capture_dir.files[name] = record
    require(ctx.operations_dir.read("capture.stderr", witness["stderr"]["identity"], ceiling=MIB) == b"",
            "original stderr must be empty")
    ctx.operations_dir.read("generated_source.py", receipt["generated_source"], ceiling=2*MIB)
    manifest = strict_json(ctx.fixture_dir.read("fixture_manifest.json", receipt["fixture_manifest"]), canonical=True)
    keys(manifest, ("schema_version", "task", "run_id", "directory", "expected", "files"), "fixture manifest")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
            and manifest["task"] == "synthetic_filter_cpu_fixture" and manifest["run_id"] == ctx.spec["run_id"]
            and manifest["directory"] == ctx.fixture_dir.record, "synthetic manifest scope")
    validate_registry(manifest["files"], (*DATA_NAMES, *CONTROL_NAMES))
    fixture_bytes = {}
    for name, record in manifest["files"].items():
        ceiling = 65536 if name.endswith(".npy") else MIB if name.endswith(".pkl") else 8*MIB
        fixture_bytes[name] = ctx.fixture_dir.read(name, record, ceiling=ceiling)
        ctx.fixture_dir.files[name] = record
    ctx.fixture_dir.files["fixture_manifest.json"] = receipt["fixture_manifest"]
    fixture_controls(ctx, manifest, fixture_bytes)
    catalog = catalog_from_metadata(ctx.prior["cpu_capture"])
    files, aliases, bridges = validate_software(artifacts["software_map.json"], ctx.envelope, catalog,
                                              ctx.private, ctx.prior["cpu_capture"])
    persistent = []
    for alias, sid in aliases.items():
        _, bound = ctx.inventory.read(alias, expected=files[sid]["binding"])
        require(bound == files[sid]["binding"], "same-host captured software identity")
    persistent.extend(v["binding"] for v in files.values())
    for name in ("runtime_before.json", "runtime_after.json"):
        runtime = artifacts[name]
        phase = "before" if name == "runtime_before.json" else "after"
        runtime_links(runtime, phase, artifacts["software_map.json"])
        for asset in runtime["profile"]["implementation_assets"]:
            bridge = bridges[asset["asset_id"]]
            require(phase in bridge["phases"] and all(asset[k] == bridge["binding"][k] for k in ("size", "sha256")),
                    "actual content bridge versus declared runtime profile")
    validate_exercises(artifacts["exercises.json"], ctx.envelope)
    require(np_policy_equal(artifacts["runtime_before.json"], artifacts["runtime_after.json"],
                            artifacts["exercises.json"]), "profile and exercise policy restored")
    validate_capture_join(ctx, receipt, manifest, artifacts, files, bridges)
    ctx.inventory.sweep()
    record = {
        **ctx.envelope, "status": "verified_pending_verifier_exit_and_external_terminal",
        "verifier_pid": os.getpid(), "capture_pid": receipt["capture_pid"], "hostname": "noderome251",
        "capture_witness": {k: witness_bind[k] for k in ("path", "size", "sha256")},
        "capture_directory": ctx.capture_dir.record, "fixture_directory": ctx.fixture_dir.record,
        "capture_artifacts": receipt["artifacts"], "fixture_manifest": receipt["fixture_manifest"],
        "generated_source": receipt["generated_source"], "persistent_checked": persistent,
        "aliases_checked": len(aliases), "source_checked": [
            ctx.inventory.aliases[v["path"]] for v in ctx.authenticate_sources()],
        "before_after_equal": artifacts["runtime_before.json"] == artifacts["runtime_after.json"],
        "clock": clock(ctx.started, ctx.deadline), "checks": [
            {"name": "actual_capture_exit_and_original_semantic_records", "passed": True}],
        "limitations": list(NEW_LIMITATIONS),
    }
    validate_clock(record["clock"], True)
    ctx.stage = "write_verification"
    file = ctx.capture_dir.write("verification.json", record)
    require(ctx.capture_dir.names() == set((*CAPTURE_NAMES, "verification.json")), "nine preterminal capture leaves")
    ctx.inventory.sweep()
    return {**ctx.envelope, "kind": "verification_handoff", "verifier_pid": os.getpid(),
            "capture_pid": receipt["capture_pid"], "hostname": "noderome251",
            "capture_directory": ctx.capture_dir.record, "fixture_directory": ctx.fixture_dir.record,
            "verification": file, "capture_witness": record["capture_witness"], "artifacts": receipt["artifacts"]}


def fixture_controls(ctx, manifest, data):
    spec, complete, review = [strict_json(data[n], canonical=True) for n in CONTROL_NAMES]
    common = ("schema_version", "task", "run_id", "scope")
    keys(spec, (*common, "pickle_protocol", "counts", "slides", "requests", "data", "baseline_recipe"), "synthetic spec")
    keys(complete, (*common, "spec_sha256", "outputs"), "synthetic complete")
    keys(review, (*common, "spec_sha256", "complete_sha256"), "synthetic review")
    for v, scope in zip((spec, complete, review), ("generated_basis_vectors_not_experimental_results",
             "generated_data_byte_registry_only", "owner_generated_fixture_not_actual105_or_independent_acceptance")):
        require(type(v["schema_version"]) is int and v["schema_version"] == 1
                and v["task"] == "synthetic_filter_cpu_fixture" and v["run_id"] == ctx.spec["run_id"]
                and v["scope"] == scope, "synthetic controls not independent acceptance")
    require(type(spec["pickle_protocol"]) is int and spec["pickle_protocol"] == 4, "pickle recipe")
    same_json(spec["baseline_recipe"], BASELINE_RECIPE)
    same_json(spec["requests"], requests())
    same_json(spec["counts"], {"slides": 2, "patients": 2, "diagnostic_groups": 1, "union_patients": 2,
                               "feature_files": 2, "original_tiles": 4, "heads": 10, "slide_head_cells": 20,
                               "tile_head_predictions": 40})
    slide_fields = ("slide_index", "metadata_index", "slide_id", "patient_id", "diagnostic_group",
                    "slide_fold", "tile_start", "tile_stop", "feature_rows")
    same_json(spec["slides"], [dict(zip(slide_fields, v)) for v in (
        (0, 0, "probe_q", "probe_patient_q", "probe_group", 0, 0, 2, 2),
        (1, 1, "probe_r", "probe_patient_r", "probe_group", 1, 2, 4, 2))])
    keys(spec["data"], DATA_NAMES, "14 generated DATA declarations")
    keys(complete["outputs"], DATA_NAMES, "14 generated outputs")
    for name in DATA_NAMES:
        descriptor = spec["data"][name]
        validate_desc(descriptor, 65536 if name.endswith(".npy") else MIB)
        require(descriptor["path"] == str(ctx.fixture_dir.path/name), "fixture-local descriptor")
        same_json(complete["outputs"][name], {k: descriptor[k] for k in ("size", "sha256")})
        file = manifest["files"][name]
        require(file["size"] == descriptor["size"] == len(data[name])
                and file["sha256"] == descriptor["sha256"] == sha(data[name]), "original generated bytes")
        if name.endswith(".npy"):
            check_npy(data[name], name)
    require(complete["spec_sha256"] == review["spec_sha256"] == sha(data["synthetic_spec.json"])
            and review["complete_sha256"] == sha(data["synthetic_complete.json"]), "original control lineage")
    same_json(manifest["expected"], {"review_sha256": sha(data["synthetic_review.json"]),
               "complete_sha256": sha(data["synthetic_complete.json"]), "spec_sha256": sha(data["synthetic_spec.json"]),
               "run_id": ctx.spec["run_id"], "counts": spec["counts"]})


def check_npy(data, name):
    require(data[:8] == b"\x93NUMPY\x01\x00" and len(data) >= 10, "NPY1 framing")
    count = int.from_bytes(data[8:10], "little")
    require(10+count <= len(data) and (10+count) % 16 == 0 and data[9+count] == 10, "NPY exact bounded header")
    node = ast.parse(data[10:10+count].decode("latin1").strip(), mode="eval").body
    require(type(node) is ast.Dict and len(node.keys) == 3, "NPY duplicate-safe header")
    header = ast.literal_eval(node)
    keys(header, ("descr", "fortran_order", "shape"), "NPY three keys")
    if name == "coordinates.npy":
        expected = {"descr": "<i8", "fortran_order": False, "shape": (4, 2)}
        payload = struct.pack("<8q", 0, 0, 16, 0, 0, 0, 16, 0)
    else:
        require(re.fullmatch("mask_f[0-4]_b(5|100)\\.npy", name) is not None, "ten fixed masks")
        expected = {"descr": "|b1", "fortran_order": False, "shape": (4,)}
        payload = b"\x00\x01\x01\x00" if name.endswith("_b5.npy") else b"\x01"*4
    require(type(header["fortran_order"]) is bool and type(header["shape"]) is tuple
            and all(type(v) is int for v in header["shape"]) and header == expected
            and data[10+count:] == payload, "exact generated NPY dtype/extent/bytes")


def validate_capture_join(ctx, receipt, manifest, artifacts, files, bridges):
    cap = artifacts["capture.json"]
    keys(cap, (*ctx.envelope, "status", "capture_pid", "hostname", "capture_directory", "fixture_directory",
               "fixture_manifest", "artifacts", "clock", "generated_source", "source_authority",
               "checks", "limitations"), "capture record")
    validate_envelope(cap, ctx.envelope)
    require(cap["status"] == "captured_pending_actual_process_exit" and cap["capture_pid"] == receipt["capture_pid"]
            and cap["source_authority"] == "trusted_source_buffer_bootstrap"
            and cap["hostname"] == "noderome251" and cap["limitations"] == NEW_LIMITATIONS, "capture scope/source")
    same_json(cap["artifacts"], {n: receipt["artifacts"][n] for n in CAPTURE_NAMES if n != "capture.json"})
    for key in ("capture_directory", "fixture_directory", "fixture_manifest", "generated_source"):
        same_json(cap[key], receipt[key])
    validate_clock(cap["clock"], True)
    validate_checks(cap["checks"], 32, True)
    startup = artifacts["startup.json"]
    keys(startup, (*ctx.envelope, "scope", "hostname", "capture_pid", "owner_uid", "source_handoff", "interpreter",
                   "bootstrap_source", "root_origin", "cwd", "sys_path_pre_site", "sys_path_after_site", "flags",
                   "cpu", "environment", "startup_originals", "startup_final", "site_directories", "customization",
                   "clock", "checks", "limitations"), "Startup")
    validate_envelope(startup, ctx.envelope)
    require(startup["scope"] == "stdlib_first_source_buffer_then_explicit_site"
            and startup["hostname"] == "noderome251" and startup["capture_pid"] == receipt["capture_pid"]
            and startup["owner_uid"] == ctx.private["owner_uid"] and startup["limitations"] == NEW_LIMITATIONS,
            "controlled startup not normal-worker acceptance")
    for name in ("interpreter", "bootstrap_source", "source_handoff"):
        validate_bind(startup[name])
    require(startup["source_handoff"] == ctx.source_handoff, "same-host authentic executing source")
    for name in ("cwd", "sys_path_pre_site", "sys_path_after_site", "environment"):
        same_json(startup[name], ctx.private[name])
    same_json(startup["site_directories"], ctx.spec["startup_sources"]["site_directories"])
    same_json(startup["customization"], ctx.spec["startup_sources"]["customization"])
    same_json(startup["root_origin"], {"origin": str(Path(ctx.spec["code_files"]["package_init"]["path"]).resolve()),
              "search_locations": [str(Path(ctx.spec["code_files"]["package_init"]["path"]).resolve().parent)]})
    keys(startup["flags"], ("no_site", "dont_write_bytecode", "isolated", "ignore_environment", "safe_path"), "startup flags")
    for v in startup["flags"].values():
        boolean(v)
    require(startup["flags"]["no_site"] is startup["flags"]["dont_write_bytecode"] is True, "-S -B authority")
    keys(startup["cpu"], ("count", "affinity"), "one CPU")
    integer(startup["cpu"]["count"], 1)
    require(len(array(startup["cpu"]["affinity"], 1)) == 1, "one observed affinitized CPU")
    integer(startup["cpu"]["affinity"][0])
    validate_clock(startup["clock"], True)
    validate_checks(startup["checks"], 32, True)
    inputs = artifacts["input_bindings.json"]
    keys(inputs, (*ctx.envelope, "source_originals", "source_final", "implementation_bindings", "software_map_binding",
                  "fixture_directory", "fixture_manifest", "fixture_bindings", "raw_consumptions",
                  "prior_inventory_originals", "prior_inventory_final", "executing_code_paths", "limitations",
                  "baseline_consumption"), "InputBindings")
    validate_envelope(inputs, ctx.envelope)
    require(inputs["limitations"] == NEW_LIMITATIONS, "input scope")
    for originals, finals in ((inputs["source_originals"], inputs["source_final"]),
                              (startup["startup_originals"], startup["startup_final"])):
        require(len(array(originals, 512)) == len(array(finals, 512)), "source original/final count")
        for original, final in zip(originals, finals):
            validate_desc(original)
            validate_bind(final)
            _, observed = ctx.inventory.read(original["path"], expected=original)
            require(observed == final, "original to final same-host source bytes")
    same_json(inputs["source_originals"], ctx.authenticate_sources())
    same_json(inputs["software_map_binding"], {k: receipt["artifacts"]["software_map.json"][k] for k in ("size", "sha256")})
    same_json(inputs["fixture_directory"], receipt["fixture_directory"])
    same_json(inputs["fixture_manifest"], receipt["fixture_manifest"])
    same_json(inputs["prior_inventory_originals"], ctx.spec["prior_inventory_assets"])
    same_json(inputs["prior_inventory_final"], ctx.spec["prior_inventory_assets"])
    same_json(inputs["executing_code_paths"], {r: str(Path(d["path"]).resolve()) for r, d in ctx.spec["code_files"].items()})
    for asset in array(inputs["implementation_bindings"], 152):
        validate_asset_binding(asset)
        bridge = bridges[asset["asset_id"]]
        b = bridge["binding"]
        require(asset["role"] == "runtime_implementation" and asset["resolved_path"] == b["path"]
                and asset["size"] == b["size"] and asset["sha256"] == b["sha256"]
                and (asset["device"], asset["inode"]) == (b["st_dev"], b["st_ino"]), "actual lease-bound contents")
    require({v["asset_id"] for v in inputs["implementation_bindings"]} == set(bridges), "no omitted runtime leases")
    validate_fixture_bindings(inputs["fixture_bindings"], manifest, ctx.fixture_dir.path)
    exercise_record = artifacts["exercises.json"]
    same_json(exercise_record["bundle_expected"], manifest["expected"])
    same_json(inputs["raw_consumptions"], exercise_record["raw_consumptions"])
    same_json(inputs["baseline_consumption"], exercise_record["baseline"]["consumption"])
    for consumed in (*inputs["raw_consumptions"], inputs["baseline_consumption"]):
        name = Path(consumed["consumer_path"]).name
        require(name in manifest["files"] and consumed["consumer_path"] == str(ctx.fixture_dir.path/name)
                and consumed["producer_size"] == manifest["files"][name]["size"]
                and consumed["producer_sha256"] == manifest["files"][name]["sha256"], "decoder original bytes")
    summary = artifacts["summary.json"]
    keys(summary, (*ctx.envelope, "disposition", "runtime_before_after_equal", "numpy_error_policy_restored",
                   "exercise_counts", "software_counts", "usage", "pilot_runtime_accepted",
                   "real_load_feasibility_established", "checks", "limitations"), "Summary")
    validate_envelope(summary, ctx.envelope)
    require(summary["disposition"] == "captured_pending_exit_and_independent_review"
            and summary["numpy_error_policy_restored"] is True and summary["pilot_runtime_accepted"] is False
            and summary["real_load_feasibility_established"] is False and summary["limitations"] == NEW_LIMITATIONS,
            "no real/runtime/feasibility acceptance")
    boolean(summary["runtime_before_after_equal"])
    require(summary["runtime_before_after_equal"] == (artifacts["runtime_before.json"] == artifacts["runtime_after.json"]),
            "actual before/after difference preserved")
    same_json(summary["exercise_counts"], {"requests": 8, "ot_log_calls": 4, "ot_scalar_calls": 4, "bob_calls": 4,
              "raw_decodes": 9, "preparations": 16, "max_pair_cells": 4, "baseline_decoder_calls": 1})
    same_json(summary["software_counts"], artifacts["software_map.json"]["counts"])
    keys(summary["usage"], ("wall_ns", "user_seconds_hex", "system_seconds_hex", "maxrss_kib"), "Usage")
    for name in ("wall_ns", "maxrss_kib"):
        integer(summary["usage"][name])
    for name in ("user_seconds_hex", "system_seconds_hex"):
        text(summary["usage"][name], 128)
        number = float.fromhex(summary["usage"][name])
        require(math.isfinite(number) and number >= 0 and number.hex() == summary["usage"][name], "actual finite usage")
    validate_checks(summary["checks"], 32, True)
    software = artifacts["software_map.json"]
    historical = SoftwareObservation(ctx)
    historical.inventory = types.SimpleNamespace(
        records={v["binding"]["path"]: v["binding"] for v in software["files"]})
    historical.source_descriptors = inputs["source_originals"]
    historical.bridge = {v["runtime_asset_id"]: v for v in software["runtime_bridge"]}
    historical.namespaces = software["namespaces"]
    comparisons, prior_aliases, unobserved = historical.historical()
    same_json(software["historical_comparisons"], comparisons)
    same_json(software["historical_aliases"], prior_aliases)
    same_json(software["historical_unobserved"], unobserved)
    for distribution in software["version_metadata"]:
        binding = files[distribution["metadata_asset_id"]]["binding"]
        payload, _ = ctx.inventory.read(distribution["metadata_path"], expected=binding, cap=8*MIB, retain=True)
        message = email.parser.Parser().parsestr(payload.decode("utf-8"))
        normalize = lambda name: re.sub("[-_.]+", "-", name.lower())
        require(type(message["Name"]) is str and normalize(message["Name"]) ==
                normalize(distribution["distribution"]) and message["Version"] == distribution["observed_version"],
                "fresh original version metadata semantics")
        for phase in ("runtime_before.json", "runtime_after.json"):
            profile = artifacts[phase]["profile"]
            versions = {**profile["base_profile"]["distribution_versions"], **profile["distribution_versions"]}
            require(versions[distribution["distribution"]] == distribution["observed_version"],
                    "metadata to both observed getter records")
    generated = artifacts["software_map.json"]["generated"]
    keys(generated, ("name", "live_binding", "persistent_path_required", "expected_source",
                     "stable_source_asset_ids", "phases", "retained_bytes_descriptor", "post_exit_path_state"), "Generated")
    require(generated["name"] == GENERATED and generated["persistent_path_required"] is False
            and generated["post_exit_path_state"] == "not_required", "generated lifetime honesty")
    validate_bind(generated["live_binding"])
    require(generated["live_binding"]["size"] == 2355 and generated["live_binding"]["sha256"] == GENERATED_HASH
            and Path(generated["live_binding"]["path"]).name == GENERATED+".py",
            "retained generated source versus original live observation")
    same_json(generated["phases"], ["before", "after"])
    same_json(generated["expected_source"], {"size": 2355, "sha256": GENERATED_HASH})
    same_json(generated["retained_bytes_descriptor"], receipt["generated_source"])
    raw = ctx.operations_dir.read("generated_source.py", receipt["generated_source"], ceiling=2*MIB)
    require(len(raw) == 2355 and sha(raw) == GENERATED_HASH, "original generated retained bytes")
    B = ctx.load_bootstrap()
    adapter = ThreeSourceAdapter(ctx.inventory, ctx.spec["generated_source"])
    B.derive_generated({"baseline": {"generated_module_bindings": {GENERATED: ctx.spec["generated_source"]}},
                        "inventory": adapter})
    same_json(generated["stable_source_asset_ids"], {r: software_id(ctx.inventory.aliases[d["path"]]["path"])
              for r, d in ctx.spec["generated_source"]["stable_sources"].items()})


def initial_gate(ctx):
    authority = ctx.initial_authority
    require(list(sys.path) == authority["sys_path"] == ctx.private["sys_path_pre_site"], "effective initial search authority")
    require(not any(n == "prosper" or n.startswith("prosper.") or CALLER in n for n in sys.modules),
            "initializer/caller before initial gate")
    for declaration in authority["imported_sources"]:
        name = declaration["module"]
        require(name in sys.modules, "initial authority cannot certify an unobserved import")
        shape = _module_shape(name, sys.modules[name])
        require(shape["origin"] == declaration["origin"], "effective initial stdlib origin")
        if declaration["source"] is not None:
            require(shape["file"] is not None and str(Path(shape["file"]).resolve(strict=True)) ==
                    str(Path(declaration["source"]["path"]).resolve(strict=True)), "initial source association")
        if declaration["associated_cache"] is not None:
            require(shape["cached_path"] == declaration["associated_cache"]["path"], "associated initial cache")
    actual = {n for n, obj in tuple(sys.modules.items()) if
              _module_shape(n, obj)["origin"] not in (None, "built-in", "frozen") and n != "__main__"}
    supplied = {v["module"] for v in authority["imported_sources"]}
    require(actual <= supplied, "uncovered initial file-backed import")


def release_context(ctx, primary=None):
    for directory in (getattr(ctx, "fixture_dir", None), getattr(ctx, "capture_dir", None),
                      getattr(ctx, "operations_dir", None)):
        if directory is not None and directory.fd is not None:
            try:
                directory.release()
            except BaseException as extra:
                if primary is None:
                    primary = extra
                else:
                    add_secondary(primary, extra, "owned namespace release")
    if primary is not None:
        raise primary


def final_namespaces(ctx, mode):
    primary = None
    held = []
    try:
        for previous, membership in ((ctx.capture_dir, set(CAPTURE_NAMES) if mode == "capture" else
                                      set((*CAPTURE_NAMES, "verification.json"))),
                                     (ctx.fixture_dir, set(FIXTURE_NAMES))):
            current = OwnedDirectory(previous.path, ctx.private["owner_uid"], ctx.deadline, expected=previous.record)
            held.append(current)
            require(current.names() == membership, "after-release exact namespace")
            for name, file in previous.files.items():
                current.read(name, file, ceiling=32*MIB if name == "software_map.json" else 8*MIB)
        current = OwnedDirectory(ctx.operations_dir.path, ctx.private["owner_uid"], ctx.deadline,
                                 expected=ctx.operations_dir.record)
        held.append(current)
        allowed = set(OPERATION_NAMES[:4] if mode == "capture" else OPERATION_NAMES[:6])
        require(current.names() <= allowed and "generated_source.py" in current.names(), "after-release operations scope")
        ctx.inventory.sweep()
    except BaseException as exc:
        primary = exc
    finally:
        for directory in reversed(held):
            try:
                directory.release()
            except BaseException as extra:
                if primary is None:
                    primary = extra
                else:
                    add_secondary(primary, extra, "reacquired namespace release")
    if primary is not None:
        raise primary


def emit_handoff(value):
    global _EMITTED
    data = encoded(value, 16384)
    require(data.count(b"\n") == 1 and data.endswith(b"\n"), "literal one-line handoff")
    offset = 0
    while offset < len(data):
        count = os.write(1, data[offset:])
        require(count > 0, "handoff stream write")
        offset += count
    sys.stdout.flush()
    _EMITTED = sha(data)
    return _EMITTED


def validate_normal(value, mode, source_sha256, handoff_sha256):
    keys(value, ("kind", "mode", "handoff_sha256", "executing_source_sha256"), "NormalReturn")
    require(value["kind"] == "normal_success_return" and value["mode"] == mode
            and mode in ("capture", "verify"), "fixed normal return")
    digest(value["handoff_sha256"])
    digest(value["executing_source_sha256"])
    require(handoff_sha256 is not None and value["handoff_sha256"] == handoff_sha256
            and value["executing_source_sha256"] == source_sha256, "original emitted/authenticated-source return")
    return value


def entry(argv, authenticated_source_handoff):
    global _ACTIVE_CONTEXT
    require(_ACTIVE_CONTEXT is None, "entry already initialized; preserve its local failure epoch")
    ctx = Context.__new__(Context)
    ctx.envelope = None
    _ACTIVE_CONTEXT = ctx
    primary = None
    try:
        Context.__init__(ctx, argv, authenticated_source_handoff)
        ctx.authenticate_sources()
        initial_gate(ctx)
        handoff = capture(ctx) if ctx.args.mode == "capture" else verify(ctx)
        validate_handoff(handoff, ctx.envelope, ctx.args.mode)
        ctx.stage = "release_capture" if ctx.args.mode == "capture" else "release_verification"
        release_context(ctx)
        final_namespaces(ctx, ctx.args.mode)
        require(time.monotonic_ns() <= ctx.deadline, "handoff deadline")
        h = emit_handoff(handoff)
        return validate_normal({
            "kind": "normal_success_return", "mode": ctx.args.mode,
            "handoff_sha256": h, "executing_source_sha256": authenticated_source_handoff["sha256"],
        }, ctx.args.mode, authenticated_source_handoff["sha256"], h)
    except BaseException as exc:
        primary = exc
        if ctx.envelope is not None:
            try:
                ctx.failure(primary)
            except BaseException as extra:
                add_secondary(primary, extra, "entry failure recording")
        raise
    finally:
        release_context(ctx, primary)


def boot_error(primary, authenticated_sha256):
    ctx = _ACTIVE_CONTEXT
    secondary, omitted = secondary_records(primary)
    if ctx is not None and ctx.envelope is not None:
        directory = getattr(ctx, "capture_dir", None)
        reopened = None
        try:
            if directory is not None and directory.fd is None:
                reopened = OwnedDirectory(directory.path, ctx.private["owner_uid"], ctx.deadline, expected=directory.record)
                reopened.files = dict(directory.files)
                reopened.completed_files = dict(directory.completed_files)
                reopened.failure_records = dict(directory.failure_records)
                ctx.capture_dir = reopened
            ctx.failure(primary)
        except BaseException as extra:
            add_secondary(primary, extra, "bootstrap failure persistence")
        finally:
            if reopened is not None:
                try:
                    reopened.release()
                except BaseException as extra:
                    add_secondary(primary, extra, "bootstrap failure directory close")
        secondary, omitted = secondary_records(primary)
        record = {"error": error_record(primary, "bootstrap finalization"), "secondary": secondary,
                  "omitted_secondary_count": omitted}
        data = encoded(record, 16384)
    else:
        stage = getattr(ctx, "stage", "bootstrap_finalization") if ctx is not None else "bootstrap_finalization"
        if stage not in ("bootstrap_args", "bootstrap_source", "entry_args", "spec_bytes", "private_bytes",
                         "context_agreement", "bootstrap_finalization"):
            stage = "bootstrap_finalization"
        data = encoded({
            "schema_version": 1, "kind": "pre_envelope_failure", "stage": stage,
            "reason_code": "entry_convention" if type(primary) is ProbeError else "finalization_failure",
            "error_type": safe_type(primary), "reason": reason_record(primary),
            "validated_executing_source_sha256": authenticated_sha256, "secondary_count": len(secondary)+omitted,
        }, 8192)
    offset = 0
    while offset < len(data):
        count = os.write(2, data[offset:])
        require(count > 0, "bootstrap diagnostic write")
        offset += count


BOOT_LITERAL = r'''import os,sys,hashlib,types,json
fd=None
module=None
primary=None
normal=None
authenticated=None
secondary=[]
def fail(message):
    raise ValueError(message)
def scan(handle,limit):
    os.lseek(handle,0,os.SEEK_SET)
    chunks=[]
    total=0
    h=hashlib.sha256()
    while True:
        piece=os.read(handle,min(65536,limit-total+1))
        if not piece:
            break
        total+=len(piece)
        if total>limit:
            fail("source preallocation bound")
        chunks.append(piece)
        h.update(piece)
    return b"".join(chunks),h.hexdigest()
def ident(s):
    return (s.st_dev,s.st_ino,s.st_size)
def remember(error):
    global primary
    if primary is None:
        primary=error
    else:
        secondary.append(error)
        if module is not None and hasattr(module,"add_secondary"):
            module.add_secondary(primary,error,"BOOT finalization")
try:
    if len(sys.argv)<5:
        fail("fixed BOOT arguments")
    path,size_text,expected=sys.argv[1:4]
    if not os.path.isabs(path) or not size_text.isascii() or not size_text.isdigit() or len(size_text)>7:
        fail("fixed source argument domains")
    size=int(size_text)
    if size<1 or size>2097152 or len(expected)!=64 or any(c not in "0123456789abcdef" for c in expected):
        fail("source descriptor domains")
    resolved=os.path.realpath(path)
    cwd=os.getcwd()
    prescribed=os.path.join(cwd,"scripts","r2_1_filter_cpu_profile_probe.py")
    if cwd!=os.path.realpath(cwd) or os.path.basename(cwd)!="prosper-search" or path!=prescribed or resolved!=path:
        fail("prescribed canonical probe source")
    before=os.lstat(path)
    if (before.st_mode&0o170000)!=0o100000:
        fail("regular no-symlink probe source")
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    if ident(os.fstat(fd))!=ident(before) or before.st_size!=size or (before.st_mode&0o170000)!=0o100000:
        fail("source acquisition custody")
    data,actual=scan(fd,size)
    if len(data)!=size or actual!=expected or ident(os.lstat(path))!=ident(before) or os.path.realpath(path)!=path:
        fail("source hash/custody")
    authenticated=actual
    handoff={"path":resolved,"size":size,"sha256":actual,"st_dev":before.st_dev,"st_ino":before.st_ino}
    module=types.ModuleType("__main__")
    module.__file__=path
    module.__spec__=None
    module.__package__=""
    sys.modules["__main__"]=module
    exec(compile(data,path,"exec"),module.__dict__)
    normal=module.entry(sys.argv[4:],handoff)
    if type(normal) is not dict or len(normal)!=4 or not all(type(k) is str and type(v) is str for k,v in normal.items()) or set(normal)!={"kind","mode","handoff_sha256","executing_source_sha256"}:
        fail("entry convention")
    if normal["kind"]!="normal_success_return" or normal["mode"]!=sys.argv[4] or normal["mode"] not in ("capture","verify"):
        fail("normal return mode")
    if type(normal["handoff_sha256"]) is not str or len(normal["handoff_sha256"])!=64 or any(c not in "0123456789abcdef" for c in normal["handoff_sha256"]):
        fail("normal handoff digest")
    if normal["executing_source_sha256"]!=authenticated or normal["handoff_sha256"]!=module._EMITTED:
        fail("normal return source/original stream")
except BaseException as error:
    primary=error
finally:
    if fd is not None:
        try:
            final_data,final_sha=scan(fd,size)
            if len(final_data)!=size or final_sha!=expected or ident(os.fstat(fd))!=ident(before) or ident(os.lstat(path))!=ident(before) or os.path.realpath(path)!=resolved:
                fail("final source bytes/path custody")
        except BaseException as error:
            remember(error)
        try:
            os.close(fd)
        except BaseException as error:
            remember(error)
    if primary is not None:
        try:
            if module is not None and hasattr(module,"boot_error"):
                module.boot_error(primary,authenticated)
            else:
                cls=type(primary)
                qualified=type.__dict__["__module__"].__get__(cls)+"."+type.__dict__["__qualname__"].__get__(cls)
                if type(qualified) is not str or len(qualified)>512:
                    qualified=None
                record={"schema_version":1,"kind":"pre_envelope_failure","stage":"bootstrap_source",
                    "reason_code":"source_custody","error_type":qualified,"reason":{"kind":"type_only","fragments":[],
                    "source_argument_count":None,"omitted_argument_count":None,"errno":None,"unavailable_argument_count":None},
                    "validated_executing_source_sha256":authenticated,"secondary_count":len(secondary)}
                payload=(json.dumps(record,sort_keys=True,ensure_ascii=True,allow_nan=False,separators=(",",":"))+"\n").encode("ascii")
                if len(payload)>8192:
                    fail("early diagnostic cap")
                offset=0
                while offset<len(payload):
                    written=os.write(2,payload[offset:])
                    if written<=0:
                        fail("diagnostic stream write")
                    offset+=written
        except BaseException as error:
            remember(error)
    for stream in (sys.stdout,sys.stderr):
        try:
            stream.flush()
        except BaseException as error:
            remember(error)
status=0 if primary is None and normal is not None else 1
if any(base is SystemExit for base in type.__dict__["__mro__"].__get__(type(primary))):
    code=SystemExit.code.__get__(primary)
    if type(code) is int and 1<=code<=125:
        status=code
os._exit(status)
'''
