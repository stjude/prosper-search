"""Bounded stdlib capture; candidate receipts never authorize inputs or models."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import selectors
import signal
import stat
import subprocess
import sys
import time
from typing import Callable, TypedDict


NAME = "r21_input_observation_capture_verified_v1"
INTERPRETER = "/HOME/miniconda3/bin/python3"
CALLER_SHA = "168f6578c82502b8c9b4602723ebeae33e2f138f06f90371b54f3a1670ebc7d4"
CALLER_SIZE = 77109
EMPTY_SHA = hashlib.sha256(b"").hexdigest()
BLOCK = 65536
SOURCE_CAP = 1048576
STDOUT_CAP = 16777216
MEMBERS = (
    "capture_declaration.json", "caller_source.py", "wrapper_source.py",
    "root_launch.py", "child_bootstrap.py", "child_stdout.bin",
    "child_stderr.bin", "capture_receipt.json",
)
CAPS = dict(zip(MEMBERS, (SOURCE_CAP, SOURCE_CAP, SOURCE_CAP, BLOCK, BLOCK,
                         STDOUT_CAP, BLOCK, SOURCE_CAP)))
LIMITATIONS = [
    "capture_not_input_or_model_authorization", "external_trust_anchor_required",
    "direct_child_only_no_descendant_containment",
    "cooperative_filesystem_checks_not_atomic_snapshot",
    "userspace_deadlines_not_interruptible_filesystem_guarantee",
    "opaque_stderr_no_origin_attribution", "sent_bytes_not_child_consumption_proof",
    "candidate_receipt_not_terminal_acceptance",
    "prefix_digests_not_unread_byte_attestation",
    "no_complete_executable_software_attestation",
]
FORBIDDEN = ("PIL", "safetensors", "numpy", "pandas", "torch", "timm", "openslide", "prosper")
STAGES = (
    "trusted_basis", "initial_inventory", "template", "config", "checkpoint_header",
    "checkpoint_full", "selected_wsi_full", "final_checks", "final_inventory", "serialization",
)
CONTROLS = (
    "observer_review", "metadata", "targets", "cpu_review", "cpu_complete",
    "cpu_spec", "cpu_input_bindings", "stage1b_review", "stage1b_complete",
    "stage1b_source_inventory",
)
ASSETS = ("template", "model_config", "checkpoint", "selected_wsi")
CALLER_LIMITATIONS = [
    "stdlib_observations_not_decoded_pixels", "structure_and_schema_not_model_identity",
    "declared_byte_matches_not_historical_vendor_authentication",
    "inventory_metadata_not_readability_or_content", "prior_bindings_not_current_observations",
    "cooperative_checks_not_atomic_snapshot", "stdout_requires_reviewed_capture",
    "no_complete_executable_software_attestation", "no_input_or_model_use_authorization",
    "logical_io_not_measured_resource_feasibility", "no_population_narrowing_or_substitution",
]


class EntryValidationError(ValueError):
    """Invalid trusted context; no filesystem or child work is permitted."""


class CaptureResult(TypedDict):
    capture_id: str
    classification: str
    intended_exit_code: int
    receipt_binding: dict | None
    ticket_bytes: bytes | None
    storage_state: str
    owned_closures_succeeded: bool
    child: dict
    streams: dict
    packet: dict
    artifacts: dict
    errors: dict
    model_use_authorized: bool


TrustedEntry = dict


def canonical(value: object, *, newline: bool = False) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False).encode("utf-8")
            + (b"\n" if newline else b""))


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fields(value: object, keys: object) -> dict:
    _need(type(value) is dict and set(value) == set(keys), "Closed record membership")
    return value


def _integer(value: object, low: int = 0, high: int | None = None) -> int:
    _need(type(value) is int and value >= low and (high is None or value <= high),
          "Strict integer domain")
    return value


def _sha(value: object) -> str:
    _need(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None, "SHA256 domain")
    return value


def _path(value: object, *, destination: bool = False) -> str:
    import unicodedata
    _need(type(value) is str and value.startswith("/") and 0 < len(value) <= 4096
          and value == value.strip()
          and not any(unicodedata.category(c) in ("Cc", "Cs") for c in value), "Literal path")
    if destination:
        _need(value != "/" and all(p not in ("", ".", "..") for p in value.split("/")[1:]),
              "Destination components")
    return value


def _pin(value: object, cap: int = SOURCE_CAP) -> dict:
    value = _fields(value, ("size", "sha256"))
    _integer(value["size"], 1, cap)
    _sha(value["sha256"])
    return value


def _source_declaration(value: object) -> dict:
    value = _fields(value, ("path", "size", "sha256"))
    _path(value["path"])
    _pin({k: value[k] for k in ("size", "sha256")})
    return value


def _deadlines(value: object) -> dict:
    value = _fields(value, ("overall_ns", "termination_grace_ns", "drain_ns"))
    for number in value.values():
        _integer(number, 1, 2**63 - 1)
    return value


def _json(encoded: bytes, depth_limit: int, *, newline: bool) -> dict:
    _need(type(encoded) is bytes and (not newline or encoded.endswith(b"\n")), "JSON bytes/LF")
    text = encoded.decode("utf-8", "strict")
    _need(not text.startswith("\ufeff"), "JSON BOM")
    depth = 0
    quoted = escaped = False
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
            _need(depth <= depth_limit, "JSON container depth")
        elif char in "]}":
            depth -= 1

    def number(token, floating=False):
        _need(len(token) <= 128, "JSON numeric token")
        result = float(token) if floating else int(token)
        _need(not floating or math.isfinite(result), "Finite JSON number")
        return result

    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result and len(key.encode("utf-8")) <= 4096, "JSON key domain")
            result[key] = value
        return result

    def constant(token):
        raise ValueError("Nonfinite JSON constant: " + token)

    value = json.loads(text, object_pairs_hook=pairs, parse_int=number,
                       parse_float=lambda t: number(t, True), parse_constant=constant)
    _need(type(value) is dict and canonical(value, newline=newline) == encoded, "Canonical JSON object")
    return value


# This fixed source is included in both authenticated entry programs, not loaded from a path.
_ENTRY_SOURCE = r'''
import hashlib, os, stat, time
from pathlib import Path
def fingerprint(info):
    return [info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns,info.st_ctime_ns]
def source_record(role,declaration):
    return {"role":role,"declaration":declaration,"state":"not_started","custody_state":"unacquired",
      "descriptor":dict.fromkeys(("lexical_before","target_before","resolved_before","opened",
        "after_fd","lexical_after","target_after","resolved_after","coherent")),
      "requested_bytes":0,"received_bytes":0,"received_sha256":hashlib.sha256(b"").hexdigest(),
      "receive_scope":"observed_prefix","eof_observed":False,"expected_size_match":None,
      "expected_sha256_match":None,"close_attempts":0,"close_state":"not_acquired","error_indices":[]}
def acquire_source(role,declaration,deadline,report):
    record=source_record(role,declaration)
    fd=None
    digest=hashlib.sha256()
    data=bytearray()
    primary=None
    def check():
        if time.monotonic_ns()>=deadline:
            raise TimeoutError("Completion deadline reached")
    def call(function,*args,**kwargs):
        check()
        value=function(*args,**kwargs)
        return value
    try:
        record["state"]="metadata"
        d=record["descriptor"]
        d["lexical_before"]=fingerprint(call(os.lstat,declaration["path"]))
        d["resolved_before"]=str(call(Path(declaration["path"]).resolve,strict=True))
        d["target_before"]=fingerprint(call(os.stat,d["resolved_before"],follow_symlinks=False))
        if not stat.S_ISREG(d["target_before"][2]) or d["target_before"][3]!=declaration["size"]:
            raise ValueError("Source regular type/size mismatch")
        check()
        fd=os.open(d["resolved_before"],os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
        record.update(state="opened",custody_state="owned",close_state="pending")
        check()
        d["opened"]=fingerprint(call(os.fstat,fd))
        if d["opened"]!=d["target_before"] or fingerprint(call(os.lstat,declaration["path"]))!=d["lexical_before"]:
            raise ValueError("Source opening custody changed")
        if str(call(Path(declaration["path"]).resolve,strict=True))!=d["resolved_before"]:
            raise ValueError("Source opening resolution changed")
        record["state"]="reading"
        while len(data)<declaration["size"]:
            amount=min(65536,declaration["size"]-len(data))
            record["requested_bytes"]+=amount
            block=call(os.read,fd,amount)
            record["received_bytes"]+=len(block)
            digest.update(block)
            record["received_sha256"]=digest.hexdigest()
            data.extend(block)
            check()
            if not block:
                raise ValueError("Short source stream")
        record["requested_bytes"]+=1
        extra=call(os.read,fd,1)
        record["received_bytes"]+=len(extra)
        digest.update(extra)
        record["received_sha256"]=digest.hexdigest()
        record["eof_observed"]=not bool(extra)
        if not extra:
            record.update(receive_scope="complete_file",
              expected_size_match=len(data)==declaration["size"],
              expected_sha256_match=digest.hexdigest()==declaration["sha256"])
        check()
        if extra:
            raise ValueError("Source excess byte")
        record.update(eof_observed=True,receive_scope="complete_file",state="eof_checked",
          expected_size_match=len(data)==declaration["size"],
          expected_sha256_match=digest.hexdigest()==declaration["sha256"])
        if not record["expected_sha256_match"]:
            raise ValueError("Source hash mismatch")
        d["after_fd"]=fingerprint(call(os.fstat,fd))
        d["lexical_after"]=fingerprint(call(os.lstat,declaration["path"]))
        d["resolved_after"]=str(call(Path(declaration["path"]).resolve,strict=True))
        d["target_after"]=fingerprint(call(os.stat,d["resolved_after"],follow_symlinks=False))
        d["coherent"]=(d["lexical_before"]==d["lexical_after"] and
          d["resolved_before"]==d["resolved_after"] and
          d["opened"]==d["after_fd"]==d["target_before"]==d["target_after"])
        if not d["coherent"]:
            raise ValueError("Source closing custody changed")
        check()
    except BaseException as exc:
        primary=exc
        record.update(state="failed",custody_state="invalidated")
        report(exc,record)
    finally:
        if fd is not None:
            record["close_attempts"]=1
            try:
                os.close(fd)
                record["close_state"]="succeeded"
            except BaseException as exc:
                record.update(close_state="failed_unknown",state="failed",custody_state="failed")
                report(exc,record)
                if primary is None:
                    primary=exc
            fd=None
    if primary is not None:
        primary.acquisition=record
        primary.captured_prefix=bytes(data)
        raise primary
    try:
        check()
    except BaseException as exc:
        record.update(state="failed",custody_state="invalidated")
        report(exc,record)
        exc.acquisition=record
        exc.captured_prefix=bytes(data)
        raise
    record.update(state="finalized",custody_state="verified")
    return bytes(data),record
'''
exec(compile(_ENTRY_SOURCE, "<capture-fixed-source-acquisition>", "exec"), globals())


CHILD_BOOTSTRAP_SOURCE = rb'''
import hashlib,json,math,os,re,struct,sys,types,unicodedata
def need(ok,message):
    if not ok: raise ValueError(message)
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("utf-8")
def path(value):
    need(type(value) is str and value.startswith("/") and 0<len(value)<=4096 and value==value.strip()
      and not any(unicodedata.category(c) in ("Cc","Cs") for c in value),"Frame path")
def exact(n):
    parts=[]
    while n:
        block=os.read(0,min(n,65536))
        need(bool(block),"Short frame")
        parts.append(block); n-=len(block)
    return b"".join(parts)
def pairs(items):
    result={}
    for key,value in items:
        need(key not in result,"Duplicate frame key")
        result[key]=value
    return result
owned=None
try:
    need(sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode
      and sys.dont_write_bytecode,"Isolated child required")
    need(sys.orig_argv[:5]==["/HOME/miniconda3/bin/python3","-I","-S","-B","-c"]
      and len(sys.orig_argv)==6,"Child literal invocation")
    m=struct.unpack(">Q",exact(8))[0]
    need(1<=m<=262144,"Frame metadata limit")
    raw=exact(m)
    metadata=json.loads(raw.decode("utf-8"),object_pairs_hook=pairs)
    need(type(metadata) is dict and set(metadata)=={"frame_version","capture_id","entry_binding","argv"}
      and canonical(metadata)==raw,"Closed canonical frame")
    need(type(metadata["frame_version"]) is int and metadata["frame_version"]==1,"Frame version")
    need(type(metadata["capture_id"]) is str and re.fullmatch("[A-Za-z0-9_.-]{1,128}",metadata["capture_id"]),"Capture ID")
    entry=metadata["entry_binding"]
    need(type(entry) is dict and set(entry)=={"path","resolved_path","size","sha256"},"Entry keys")
    path(entry["path"]); path(entry["resolved_path"])
    need(type(entry["size"]) is int and 1<=entry["size"]<=1048576,"Caller size")
    need(type(entry["sha256"]) is str and re.fullmatch("[0-9a-f]{64}",entry["sha256"]),"Caller hash")
    argv=metadata["argv"]
    need(type(argv) is list and len(argv)==4 and argv[0]=="--spec" and argv[2]=="--spec-sha256","Machine arguments")
    path(argv[1])
    need(type(argv[3]) is str and re.fullmatch("[0-9a-f]{64}",argv[3]),"Spec hash")
    encoded=exact(entry["size"])
    need(not os.read(0,1),"Excess frame")
    need(hashlib.sha256(encoded).hexdigest()==entry["sha256"],"Captured caller hash")
    name="r21_input_observation_cli_verified_v1"
    need(name not in sys.modules,"Caller module collision")
    owned=types.ModuleType(name); owned.__file__=entry["path"]; owned.__package__=""
    sys.modules[name]=owned
    exec(compile(encoded,entry["path"],"exec"),owned.__dict__)
    need(sys.modules.get(name) is owned,"Caller registration changed")
    status=owned.main(argv,entry_binding=entry)
    need(type(status) is int and status in (0,1,2),"Caller status")
    need(sys.modules.get(name) is owned,"Caller registration replaced")
    del sys.modules[name]; owned=None
except BaseException as exc:
    if owned is not None and sys.modules.get(owned.__name__) is owned:
        del sys.modules[owned.__name__]
    text=("Caller bootstrap failure: "+str(exc))[:1024].encode("utf-8","backslashreplace")+b"\n"
    try:
        position=0
        while position<len(text):
            n=os.write(2,text[position:position+65536])
            if type(n) is not int or n<=0: break
            position+=n
    except BaseException: pass
    status=1
raise SystemExit(status)
'''


_ROOT_BODY = r'''
import json,re,sys,types,unicodedata
t0=time.monotonic_ns()
def need(ok,message):
    if not ok: raise ValueError(message)
def literal_path(value):
    need(value.startswith("/") and 0<len(value)<=4096 and value==value.strip()
      and not any(unicodedata.category(c) in ("Cc","Cs") for c in value),"Root path")
def decimal(value,cap):
    need(re.fullmatch("[1-9][0-9]*",value) and len(value)<=19,"Root decimal")
    n=int(value); need(n<=cap,"Root numeric ceiling"); return n
def diagnostic(exc):
    rows=[("Root entry failure: "+str(exc))[:1024]]
    rows.extend(("Secondary "+type(e).__name__+": "+str(e))[:1024] for e in root_errors if e is not exc)
    data=("\n".join(rows)+"\n").encode("utf-8","backslashreplace")
    if len(data)>65536:
        data=data[:65500]+b"\n[root diagnostics truncated]\n"
    try:
        pos=0
        while pos<len(data):
            n=os.write(2,data[pos:pos+65536])
            if type(n) is not int or n<=0: break
            pos+=n
    except BaseException: pass
root_errors=[]
def report(exc,record):
    if len(root_errors)<32:
        root_errors.append(exc)
owned=None
try:
    actual=sys.orig_argv
    need(sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode
      and sys.dont_write_bytecode,"Isolated root required")
    need(actual[:5]==["/HOME/miniconda3/bin/python3","-I","-S","-B","-c"]
      and len(actual)==26,"Root actual argument protocol")
    literal=actual[5].encode("ascii")
    need(len(literal)<=65536,"Root literal ceiling")
    flags=actual[6:]
    keys=["--capture-id","--wrapper","--wrapper-size","--wrapper-sha256","--declaration",
      "--declaration-size","--declaration-sha256","--overall-ns","--termination-grace-ns","--drain-ns"]
    need(flags[::2]==keys,"Root flag order")
    values=dict(zip(keys,flags[1::2]))
    cid=values["--capture-id"]; need(re.fullmatch("[A-Za-z0-9_.-]{1,128}",cid),"Root capture ID")
    deadlines={"overall_ns":decimal(values["--overall-ns"],2**63-1),
      "termination_grace_ns":decimal(values["--termination-grace-ns"],2**63-1),
      "drain_ns":decimal(values["--drain-ns"],2**63-1)}
    declarations=[]
    for prefix in ("wrapper","declaration"):
        path=values["--"+prefix]; literal_path(path)
        digest=values["--"+prefix+"-sha256"]; need(re.fullmatch("[0-9a-f]{64}",digest),"Root SHA")
        declarations.append({"path":path,"size":decimal(values["--"+prefix+"-size"],1048576),"sha256":digest})
    wrapper_decl,decl_decl=declarations
    declaration,decl_record=acquire_source("capture_declaration",decl_decl,t0+deadlines["overall_ns"],report)
    encoded,wrapper_record=acquire_source("wrapper",wrapper_decl,t0+deadlines["overall_ns"],report)
    name="r21_input_observation_capture_verified_v1"
    need(name not in sys.modules,"Wrapper module collision")
    owned=types.ModuleType(name); owned.__file__=wrapper_decl["path"]; owned.__package__=""
    sys.modules[name]=owned
    exec(compile(encoded,wrapper_decl["path"],"exec"),owned.__dict__)
    entry={"schema_version":1,"capture_id":cid,"started_monotonic_ns":t0,"deadlines":deadlines,
      "root_literal_bytes":literal,"root_literal_binding":{"size":len(literal),"sha256":hashlib.sha256(literal).hexdigest()},
      "wrapper_bytes":encoded,"wrapper_binding":dict(wrapper_decl,resolved_path=wrapper_record["descriptor"]["resolved_before"]),
      "wrapper_acquisition":wrapper_record,"declaration_binding":decl_decl,"declaration_acquisition":decl_record}
    def release():
        global owned
        need(sys.modules.get(name) is owned,"Wrapper registration replaced")
        del sys.modules[name]; owned=None
    status=owned.main(declaration,trusted_entry=entry,release_registration=release)
except BaseException as exc:
    diagnostic(exc); status=1
finally:
    if owned is not None and sys.modules.get(owned.__name__) is owned:
        del sys.modules[owned.__name__]
raise SystemExit(status)
'''
ROOT_LAUNCH_SOURCE = (_ENTRY_SOURCE + _ROOT_BODY).encode("ascii")


def _fingerprint(info) -> list[int]:
    return fingerprint(info)


def _trusted(entry: object, declaration_bytes: bytes) -> dict:
    try:
        _fields(entry, ("schema_version", "capture_id", "started_monotonic_ns", "deadlines",
                        "root_literal_bytes", "root_literal_binding", "wrapper_bytes",
                        "wrapper_binding", "wrapper_acquisition", "declaration_binding",
                        "declaration_acquisition"))
        _need(type(entry["schema_version"]) is int and entry["schema_version"] == 1, "Entry version")
        _need(type(entry["capture_id"]) is str
              and re.fullmatch("[A-Za-z0-9_.-]{1,128}", entry["capture_id"]) is not None, "Entry ID")
        _integer(entry["started_monotonic_ns"])
        _deadlines(entry["deadlines"])
        _need(type(declaration_bytes) is bytes and 0 < len(declaration_bytes) <= SOURCE_CAP, "Declaration bytes")
        _source_declaration(entry["declaration_binding"])
        for key, data, cap in (("root_literal_binding", entry["root_literal_bytes"], BLOCK),
                               ("wrapper_binding", entry["wrapper_bytes"], SOURCE_CAP)):
            _need(type(data) is bytes, "Captured bytes")
            binding = entry[key]
            if key == "wrapper_binding":
                _fields(binding, ("path", "resolved_path", "size", "sha256"))
                _path(binding["path"])
                _path(binding["resolved_path"])
                binding = {k: binding[k] for k in ("size", "sha256")}
            _pin(binding, cap)
            _need(len(data) == binding["size"] and hashlib.sha256(data).hexdigest() == binding["sha256"],
                  "Captured entry size/hash")
        _need(entry["root_literal_bytes"] == ROOT_LAUNCH_SOURCE, "Root literal binding")
        _need(len(declaration_bytes) == entry["declaration_binding"]["size"]
              and hashlib.sha256(declaration_bytes).hexdigest() == entry["declaration_binding"]["sha256"],
              "Captured declaration binding")
        for role, key in (("wrapper", "wrapper_acquisition"), ("capture_declaration", "declaration_acquisition")):
            record = entry[key]
            _fields(record, source_record(role, None))
            _fields(record["descriptor"], source_record(role, None)["descriptor"])
            for count_key in ("requested_bytes", "received_bytes", "close_attempts"):
                _integer(record[count_key])
            _sequence(record["error_indices"], _integer)
            _need(record["error_indices"] == [] and record["requested_bytes"] >= record["received_bytes"],
                  "Verified acquisition counters/errors")
            for field in ("lexical_before", "target_before", "opened", "after_fd",
                          "lexical_after", "target_after"):
                _fingerprint_value(record["descriptor"][field])
            _path(record["descriptor"]["resolved_before"])
            _path(record["descriptor"]["resolved_after"])
            _need(record["role"] == role and record["state"] == "finalized"
                  and record["custody_state"] == "verified" and record["eof_observed"] is True
                  and record["receive_scope"] == "complete_file"
                  and record["expected_sha256_match"] is True and record["expected_size_match"] is True
                  and record["close_state"] == "succeeded" and record["close_attempts"] == 1
                  and record["descriptor"]["coherent"] is True, "Trusted source acquisition")
            binding = entry["wrapper_binding"] if role == "wrapper" else entry["declaration_binding"]
            _need(record["declaration"] == {k: binding[k] for k in ("path", "size", "sha256")}
                  and record["received_bytes"] == binding["size"]
                  and record["received_sha256"] == binding["sha256"], "Entry acquisition binding")
            if role == "wrapper":
                _need(record["descriptor"]["resolved_before"] == binding["resolved_path"],
                      "Executed wrapper acquisition path")
        _need(__name__ == NAME and sys.modules.get(NAME) is not None
              and sys.modules[NAME].__dict__ is globals(), "Verified wrapper module/global identity")
        _need(sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode
              and sys.dont_write_bytecode, "Isolated wrapper flags")
        _need(not any(n == r or n.startswith(r + ".") for n in sys.modules for r in FORBIDDEN),
              "Forbidden scientific/package import")
        _need(signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL, "SIGCHLD ownership assumption")
        return entry
    except (Exception, KeyboardInterrupt) as exc:
        raise EntryValidationError(str(exc)) from exc


def _declaration(encoded: bytes, entry: dict) -> dict:
    value = _json(encoded, 3, newline=True)
    _fields(value, ("schema_version", "task", "capture_id", "output_dir", "interpreter",
                    "caller", "wrapper", "root_launch", "child_bootstrap", "spec", "deadlines"))
    _need(type(value["schema_version"]) is int and value["schema_version"] == 1
          and value["task"] == "hoptimus_input_observation_capture", "Capture task/version")
    _need(value["capture_id"] == entry["capture_id"] and value["interpreter"] == INTERPRETER,
          "Capture launch identity")
    _path(value["output_dir"], destination=True)
    for role in ("caller", "wrapper"):
        _source_declaration(value[role])
    _need(value["caller"]["size"] == CALLER_SIZE and value["caller"]["sha256"] == CALLER_SHA,
          "Frozen caller source pin")
    _need(value["wrapper"] == {k: entry["wrapper_binding"][k] for k in ("path", "size", "sha256")},
          "Wrapper launch binding")
    for role, actual in (("root_launch", entry["root_literal_bytes"]),
                         ("child_bootstrap", CHILD_BOOTSTRAP_SOURCE)):
        _pin(value[role], BLOCK)
        _need(value[role] == {"size": len(actual), "sha256": hashlib.sha256(actual).hexdigest()},
              "Fixed literal pin")
    _fields(value["spec"], ("path", "sha256"))
    _path(value["spec"]["path"])
    _sha(value["spec"]["sha256"])
    _deadlines(value["deadlines"])
    _need(value["deadlines"] == entry["deadlines"], "Deadline launch agreement")
    return value


def _artifact(name: str) -> dict:
    return dict(name=name, state="not_created",
                byte_scope="candidate_receipt" if name == MEMBERS[-1] else "exact_source",
                size=None, sha256=None, opened=None, before_close=None, path_check="not_performed",
                flush_state="not_performed", fsync_state="not_performed", readback_state="not_performed",
                readback_size=0, readback_sha256=EMPTY_SHA, readback_eof_observed=False,
                close_attempts=0, close_state="not_acquired", error_indices=[])


def _stream(name: str) -> dict:
    return dict(name=name, retention_limit=STDOUT_CAP if name == "stdout" else BLOCK,
                read_state="not_started", received_bytes=0, received_sha256=EMPTY_SHA,
                receive_scope="observed_prefix", eof_observed=False, acknowledged_prefix_bytes=0,
                acknowledged_prefix_sha256=EMPTY_SHA, unknown_retention_start=None,
                unknown_retention_bytes=0, discarded_suffix_start=None, discarded_bytes=0,
                discarded_sha256=EMPTY_SHA, write_state="not_started", post_failure_received_bytes=0,
                unread_bytes_unknown=True, readback_state="not_started", readback_bytes=0,
                readback_sha256=EMPTY_SHA, readback_eof_observed=False, readback_reference="none",
                readback_reference_bytes=0, readback_prefix_sha256=None, readback_reference_match=None,
                error_indices=[])


def _packet_summary() -> dict:
    return dict(state="not_observed", packet_kind=None, packet_exit_code=None, canonical=None,
                envelope_valid=None, entry_binding_match=None, spec_binding_match=None,
                limitations_match=None, status_match=None, model_use_authorized=None, error_indices=[])


class _Context:
    def __init__(self, entry: dict):
        self.entry = entry
        self.deadline = entry["started_monotonic_ns"] + entry["deadlines"]["overall_ns"]
        self.failed_at = None
        self.deadline_failed = False
        self.interrupted = False
        self.closure_uncertain = False
        self.errors = dict(primary=None, secondary=[], attempts=0, secondary_attempts=0,
                           omitted_secondary=0, overflow=False)
        self.result = dict(capture_id=entry["capture_id"], classification="capture_failure",
                           intended_exit_code=1, receipt_binding=None, ticket_bytes=None,
                           storage_state="not_created", owned_closures_succeeded=False,
                           child=dict(state="not_started", spawn_elapsed_ns=None, termination_trigger="none",
                                      termination_elapsed_ns=None, signals=[], returncode=None, reaped=False,
                                      wait_state="not_started", stdout_eof=False, stderr_eof=False,
                                      deadline_exceeded=False),
                           streams={n: _stream(n) for n in ("stdout", "stderr")},
                           packet=_packet_summary(), artifacts={n: _artifact(n) for n in MEMBERS},
                           errors=self.errors, model_use_authorized=False)
        self.fds = {}
        self.pipe_objects = {}
        self.parent_fd = self.directory_fd = None
        self.parent_path = self.output_path = self.leaf = None
        self.parent_identity = self.directory_identity = None
        self.process = None

    def fail(self, exc: BaseException, phase: str, operation: str, *,
             artifact: str | None = None, stream: str | None = None, code: str = "capture_error") -> int:
        now = time.monotonic_ns()
        if self.failed_at is None:
            self.failed_at = now
        if isinstance(exc, (KeyboardInterrupt, InterruptedError)) or (
                isinstance(exc, OSError) and exc.errno == errno.EINTR):
            self.interrupted = True
        if isinstance(exc, TimeoutError):
            self.result["child"]["deadline_exceeded"] = True
        text, kind = str(exc), type(exc).__name__
        errors = self.errors
        index = errors["attempts"]
        errors["attempts"] += 1
        record = dict(index=index, phase=phase, operation=operation[:128], artifact=artifact, stream=stream,
                      code=code, exception_type=kind[:125] + "..." if len(kind) > 128 else kind,
                      exception_type_truncated=len(kind) > 128,
                      errno=exc.errno if isinstance(exc, OSError) else None,
                      message=text[:1021] + "..." if len(text) > 1024 else text,
                      message_truncated=len(text) > 1024,
                      primary_index=0 if errors["primary"] is not None else None)
        if errors["primary"] is None:
            errors["primary"] = record
        else:
            errors["secondary_attempts"] += 1
            if len(errors["secondary"]) < 31:
                errors["secondary"].append(record)
            else:
                errors["omitted_secondary"] += 1
                errors["overflow"] = True
                if len(errors["secondary"]) == 31:
                    errors["secondary"].append(dict(record, code="secondary_error_limit",
                                                     message="Bounded secondary error evidence overflow",
                                                     message_truncated=False))
        self.result.update(classification="capture_failure", intended_exit_code=1, ticket_bytes=None)
        return index

    def check(self) -> None:
        if self.deadline_failed or time.monotonic_ns() >= self.deadline:
            self.deadline_failed = True
            raise TimeoutError("Completion deadline reached")

    def call(self, function, *args, **kwargs):
        self.check()
        value = function(*args, **kwargs)
        self.check()
        return value

    def close(self, fd: int, phase: str, operation: str, record: dict | None = None) -> bool:
        if record is not None:
            record["close_attempts"] = 1
        success = False
        try:
            if time.monotonic_ns() >= self.deadline:
                self.deadline_failed = True
                if self.failed_at is None:
                    self.fail(TimeoutError("Completion deadline before close"), phase, operation, code="deadline")
            os.close(fd)
            success = True
            if record is not None:
                record["close_state"] = "succeeded"
        except BaseException as exc:
            self.closure_uncertain = True
            if record is not None:
                record.update(close_state="failed_unknown", state="failed")
            self.fail(exc, phase, operation)
        if time.monotonic_ns() >= self.deadline:
            self.deadline_failed = True
            if self.failed_at is None:
                self.fail(TimeoutError("Completion deadline after close"), phase, operation, code="deadline")
        return success


def _write(ctx: _Context, name: str, data: bytes, record: dict | None = None) -> None:
    artifact = ctx.result["artifacts"][name]
    artifact["state"] = "writing"
    position = 0
    digest = hashlib.sha256()
    while position < len(data):
        chunk = data[position:position + BLOCK]
        attempted = acknowledged = False
        try:
            ctx.check()
            attempted = True
            n = os.write(ctx.fds[name], chunk)
            _need(type(n) is int and 0 < n <= len(chunk), "Destination write count")
            position += n
            if record is not None:
                record["acknowledged_prefix_bytes"] += n
                record["_ack_digest"].update(chunk[:n])
                record["acknowledged_prefix_sha256"] = record["_ack_digest"].hexdigest()
            digest.update(chunk[:n])
            acknowledged = True
            ctx.check()
        except BaseException as exc:
            if record is not None:
                record["write_state"] = "failed"
                if attempted and not acknowledged:
                    record.update(unknown_retention_start=record["acknowledged_prefix_bytes"],
                                  unknown_retention_bytes=len(chunk))
            artifact["state"] = "failed"
            ctx.fail(exc, "pump" if record is not None else "artifact_finalization", "write",
                     artifact=name, stream=record["name"] if record else None)
            raise
    if record is None:
        artifact.update(size=len(data), sha256=digest.hexdigest())


def _destination(ctx: _Context, path: str) -> None:
    ctx.output_path = path
    ctx.parent_path, ctx.leaf = path.rsplit("/", 1)
    ctx.parent_path = ctx.parent_path or "/"
    parent_before = ctx.call(os.lstat, ctx.parent_path)
    _need(stat.S_ISDIR(parent_before.st_mode), "Existing immediate parent must be a directory")
    ctx.check()
    ctx.parent_fd = os.open(ctx.parent_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    ctx.check()
    parent_open = ctx.call(os.fstat, ctx.parent_fd)
    ctx.parent_identity = _fingerprint(parent_open)[:3]
    _need(_fingerprint(parent_before) == _fingerprint(parent_open), "Parent opening custody")
    ctx.call(os.mkdir, ctx.leaf, 0o700, dir_fd=ctx.parent_fd)
    ctx.result["storage_state"] = "partial"
    ctx.check()
    ctx.directory_fd = os.open(ctx.leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=ctx.parent_fd)
    ctx.check()
    ctx.directory_identity = _fingerprint(ctx.call(os.fstat, ctx.directory_fd))[:3]
    _directory_guard(ctx)
    for name in MEMBERS[:-1]:
        ctx.check()
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=ctx.directory_fd)
        ctx.fds[name] = fd
        record = ctx.result["artifacts"][name]
        record.update(state="created", close_state="pending")
        ctx.check()
        record["opened"] = _fingerprint(ctx.call(os.fstat, fd))


def _directory_guard(ctx: _Context) -> None:
    _need(_fingerprint(ctx.call(os.fstat, ctx.parent_fd))[:3] == ctx.parent_identity
          == _fingerprint(ctx.call(os.lstat, ctx.parent_path))[:3], "Parent namespace substituted")
    _need(_fingerprint(ctx.call(os.fstat, ctx.directory_fd))[:3] == ctx.directory_identity
          == _fingerprint(ctx.call(os.stat, ctx.leaf, dir_fd=ctx.parent_fd, follow_symlinks=False))[:3]
          == _fingerprint(ctx.call(os.lstat, ctx.output_path))[:3], "Destination namespace substituted")


def _finalize_artifact(ctx: _Context, name: str, expected: bytes | None, stream: dict | None = None) -> None:
    fd = ctx.fds.pop(name)
    record = ctx.result["artifacts"][name]
    try:
        if ctx.interrupted:
            return
        record["flush_state"] = "succeeded"  # Unbuffered descriptor writes have no userspace buffer.
        ctx.check()
        record["fsync_state"] = "failed"
        os.fsync(fd)
        record["fsync_state"] = "succeeded"
        ctx.check()
        before = _fingerprint(ctx.call(os.fstat, fd))
        record["before_close"] = before
        _need(before[:3] == record["opened"][:3] and stat.S_ISREG(before[2]), "Artifact descriptor custody")
        matches = before == _fingerprint(ctx.call(os.stat, name, dir_fd=ctx.directory_fd, follow_symlinks=False))
        record["path_check"] = "matched" if matches else "changed"
        _need(matches, "Artifact pathname custody")
        record["path_check"] = "matched"
        _need(before[3] <= CAPS[name], "Artifact size ceiling")
        ctx.call(os.lseek, fd, 0, os.SEEK_SET)
        record["readback_state"] = "partial"
        received = hashlib.sha256()
        prefix = hashlib.sha256()
        seen = 0
        reference_count = len(expected) if expected is not None else (
            stream["acknowledged_prefix_bytes"] if stream is not None else 0)
        reference_sha = hashlib.sha256(expected).hexdigest() if expected is not None else (
            stream["acknowledged_prefix_sha256"] if stream is not None else EMPTY_SHA)
        while seen < before[3]:
            ctx.check()
            block = os.read(fd, min(BLOCK, before[3] - seen))
            _need(bool(block), "Artifact readback short stream")
            if seen < reference_count:
                prefix.update(block[:reference_count - seen])
            seen += len(block)
            received.update(block)
            record.update(readback_size=seen, readback_sha256=received.hexdigest(), readback_state="partial")
            ctx.check()
        ctx.check()
        excess = os.read(fd, 1)
        record["readback_eof_observed"] = not bool(excess)
        ctx.check()
        _need(not excess, "Artifact readback excess")
        _need(before == _fingerprint(ctx.call(os.fstat, fd))
              == _fingerprint(ctx.call(os.stat, name, dir_fd=ctx.directory_fd, follow_symlinks=False)),
              "Artifact changed during readback")
        if stream is not None:
            complete = stream["eof_observed"] and stream["received_bytes"] == stream["acknowledged_prefix_bytes"]
            stream.update(readback_state="verified" if seen == reference_count and prefix.hexdigest() == reference_sha else "mismatch",
                          readback_bytes=seen, readback_sha256=received.hexdigest(), readback_eof_observed=True,
                          readback_reference="received_stream" if complete else "acknowledged_prefix",
                          readback_reference_bytes=reference_count, readback_prefix_sha256=prefix.hexdigest(),
                          readback_reference_match=seen >= reference_count and prefix.hexdigest() == reference_sha)
        _need(seen == reference_count and received.hexdigest() == reference_sha, "Artifact original-byte readback")
        record.update(state="finalized", size=seen, sha256=received.hexdigest(), readback_state="verified")
    except BaseException as exc:
        record["state"] = "failed"
        if record["readback_state"] == "partial":
            record["readback_state"] = "failed"
        if stream is not None:
            stream.update(readback_state="failed", readback_bytes=record["readback_size"],
                          readback_sha256=record["readback_sha256"],
                          readback_eof_observed=record["readback_eof_observed"])
        ctx.fail(exc, "receipt_finalization" if name == MEMBERS[-1] else "artifact_finalization",
                 "finalize", artifact=name)
    finally:
        ctx.close(fd, "artifact_finalization", "close", record)


def _pipe_close(ctx: _Context, key: str) -> None:
    pipe = ctx.pipe_objects.pop(key, None)
    if pipe is None:
        return
    if key in ("stdout", "stderr"):
        record = ctx.result["streams"][key]
        if not record["eof_observed"] and record["read_state"] != "failed":
            record["read_state"] = "abandoned"
    if time.monotonic_ns() >= ctx.deadline:
        ctx.deadline_failed = True
        if ctx.failed_at is None:
            ctx.fail(TimeoutError("Completion deadline before pipe close"), "pump", "pipe_close", code="deadline")
    try:
        pipe.close()
        if key == "stdin":
            ctx.transmission["stdin_close_state"] = "succeeded"
    except BaseException as exc:
        ctx.closure_uncertain = True
        if key == "stdin":
            ctx.transmission["stdin_close_state"] = "failed_unknown"
        ctx.fail(exc, "pump", "pipe_close", stream=key)
    if time.monotonic_ns() >= ctx.deadline:
        ctx.deadline_failed = True
        if ctx.failed_at is None:
            ctx.fail(TimeoutError("Completion deadline after pipe close"), "pump", "pipe_close", code="deadline")


def _send_signal(ctx: _Context, kind: str) -> None:
    child = ctx.result["child"]
    if any(r["signal"] == kind for r in child["signals"]):
        return
    record = dict(signal=kind, elapsed_ns=time.monotonic_ns() - ctx.entry["started_monotonic_ns"],
                  state="attempted", error_index=None)
    child["signals"].append(record)
    try:
        if ctx.process.poll() is not None:
            record["state"] = "already_terminated"
        else:
            ctx.process.send_signal(signal.SIGTERM if kind == "SIGTERM" else signal.SIGKILL)
            record["state"] = "sent"
            child["state"] = "term_requested" if kind == "SIGTERM" else "kill_requested"
    except BaseException as exc:
        record["state"] = "failed"
        record["error_index"] = ctx.fail(exc, "termination", "send_signal")


def _poll_interval(now: int, deadline: int) -> float:
    return min(1000000000, max(0, deadline - now)) / 1000000000


def _termination_trigger(ctx: _Context) -> str:
    primary = ctx.errors["primary"]
    if primary["exception_type"] in ("KeyboardInterrupt", "InterruptedError") or primary["errno"] == errno.EINTR:
        return "interruption"
    return "overall_timeout" if primary["code"] == "deadline" else "failure"


def _exceptional_pump_cleanup(ctx: _Context, natural_deadline: int | None,
                              receive_hashes: dict, discard_hashes: dict, output: bytearray) -> None:
    child = ctx.result["child"]
    failure = ctx.failed_at
    _need(failure is not None and ctx.process is not None, "Owned first-failure cleanup")
    if child["termination_trigger"] == "none":
        child["termination_trigger"] = _termination_trigger(ctx)
    if child["termination_elapsed_ns"] is None:
        child["termination_elapsed_ns"] = failure - ctx.entry["started_monotonic_ns"]
    if ctx.transmission["state"] != "submitted":
        ctx.transmission["state"] = "failed"
    _pipe_close(ctx, "stdin")
    grace = failure + ctx.entry["deadlines"]["termination_grace_ns"]
    end = grace + ctx.entry["deadlines"]["drain_ns"]
    read_end = min(end, natural_deadline) if natural_deadline is not None else end
    for name in ("stdout", "stderr"):
        pipe = ctx.pipe_objects.get(name)
        if pipe is None:
            continue
        try:
            os.set_blocking(pipe.fileno(), False)
        except BaseException as exc:
            record = ctx.result["streams"][name]
            record["read_state"] = "failed"
            record["error_indices"].append(ctx.fail(exc, "pump", "cleanup_nonblocking", stream=name))
            _pipe_close(ctx, name)
    while True:
        now = time.monotonic_ns()
        try:
            status = ctx.process.poll()
        except BaseException as exc:
            status = None
            ctx.fail(exc, "termination", "cleanup_poll")
        if status is not None:
            child.update(state="reaped", reaped=True, returncode=status, wait_state="completed")
        if not child["reaped"]:
            if not any(attempt["signal"] == "SIGTERM" for attempt in child["signals"]):
                _send_signal(ctx, "SIGTERM")
            if (time.monotonic_ns() >= grace
                    and not any(attempt["signal"] == "SIGKILL" for attempt in child["signals"])):
                _send_signal(ctx, "SIGKILL")
        progressed = False
        for name in ("stdout", "stderr"):
            pipe = ctx.pipe_objects.get(name)
            if pipe is None:
                continue
            record = ctx.result["streams"][name]
            allowance = BLOCK - record["post_failure_received_bytes"]
            if time.monotonic_ns() >= read_end or allowance <= 0:
                _pipe_close(ctx, name)
                continue
            try:
                block = os.read(pipe.fileno(), min(BLOCK, allowance))
            except BlockingIOError:
                continue
            except BaseException as exc:
                record["read_state"] = "failed"
                record["error_indices"].append(ctx.fail(exc, "pump", "cleanup_receive", stream=name))
                _pipe_close(ctx, name)
                continue
            progressed = True
            start = record["received_bytes"]
            receive_hashes[name].update(block)
            record.update(received_bytes=start + len(block), received_sha256=receive_hashes[name].hexdigest())
            record["post_failure_received_bytes"] += len(block)
            if not block:
                record.update(read_state="eof", eof_observed=True, receive_scope="complete_stream",
                              unread_bytes_unknown=False)
                child[name + "_eof"] = True
                _pipe_close(ctx, name)
                continue
            if name == "stdout" and start < STDOUT_CAP:
                output.extend(block[:STDOUT_CAP - start])
            if record["discarded_suffix_start"] is None:
                record["discarded_suffix_start"] = start
            record["discarded_bytes"] += len(block)
            discard_hashes[name].update(block)
            record["discarded_sha256"] = discard_hashes[name].hexdigest()
            if record["received_bytes"] > record["retention_limit"]:
                record["error_indices"].append(ctx.fail(ValueError("Child stream ceiling during cleanup"),
                    "pump", "cleanup_overflow", stream=name, code="stream_overflow"))
        if child["reaped"] and not ctx.pipe_objects:
            break
        now = time.monotonic_ns()
        if now >= end:
            if not child["reaped"]:
                child.update(state="unresolved", wait_state="pending")
            for name in list(ctx.pipe_objects):
                _pipe_close(ctx, name)
            break
        next_deadline = end
        if not child["reaped"] and now < grace:
            next_deadline = min(next_deadline, grace)
        if ctx.pipe_objects and now < read_end:
            next_deadline = min(next_deadline, read_end)
        if not progressed:
            try:
                select.select([pipe.fileno() for pipe in ctx.pipe_objects.values()], [], [],
                              _poll_interval(now, next_deadline))
            except BaseException as exc:
                ctx.fail(exc, "termination", "cleanup_readiness")
                try:
                    time.sleep(_poll_interval(time.monotonic_ns(), next_deadline))
                except BaseException as sleep_error:
                    ctx.fail(sleep_error, "termination", "cleanup_sleep")
    if child["reaped"]:
        try:
            _need(ctx.process.wait(timeout=0) == child["returncode"], "Exceptional owned child status")
        except BaseException as exc:
            child["wait_state"] = "failed"
            ctx.fail(exc, "termination", "cleanup_wait")


def _pump(ctx: _Context, frame: bytes) -> bytes:
    child = ctx.result["child"]
    process = subprocess.Popen([INTERPRETER, "-I", "-S", "-B", "-c", CHILD_BOOTSTRAP_SOURCE.decode("ascii")],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               bufsize=0, close_fds=True)
    ctx.process = process
    child.update(state="running", wait_state="pending",
                 spawn_elapsed_ns=time.monotonic_ns() - ctx.entry["started_monotonic_ns"])
    ctx.pipe_objects = {"stdin": process.stdin, "stdout": process.stdout, "stderr": process.stderr}
    selector = None
    source_offset = 0
    sent_digest = hashlib.sha256()
    output = bytearray()
    natural_deadline = None
    receive_hashes = {n: hashlib.sha256() for n in ("stdout", "stderr")}
    discard_hashes = {n: hashlib.sha256() for n in ("stdout", "stderr")}
    for name, stream in ctx.result["streams"].items():
        stream.update(read_state="open", write_state="open")
        stream["_ack_digest"] = hashlib.sha256()
    try:
        selector = selectors.DefaultSelector()
        for name, pipe in ctx.pipe_objects.items():
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_WRITE if name == "stdin" else selectors.EVENT_READ, name)
        while selector.get_map() or not child["reaped"]:
            now = time.monotonic_ns()
            status = process.poll()
            if status is not None and not child["reaped"]:
                child.update(state="reaped", reaped=True, returncode=status, wait_state="completed")
                natural_deadline = min(ctx.deadline, now + ctx.entry["deadlines"]["drain_ns"])
            if now >= ctx.deadline and ctx.failed_at is None:
                ctx.fail(TimeoutError("Overall completion deadline"), "pump", "deadline", code="deadline")
                child["termination_trigger"] = "overall_timeout"
            if ctx.failed_at is not None:
                if child["termination_trigger"] == "none":
                    child["termination_trigger"] = _termination_trigger(ctx)
                if child["termination_elapsed_ns"] is None:
                    child["termination_elapsed_ns"] = ctx.failed_at - ctx.entry["started_monotonic_ns"]
                if "stdin" in ctx.pipe_objects:
                    pipe = ctx.pipe_objects["stdin"]
                    if pipe in selector.get_map():
                        selector.unregister(pipe)
                    _pipe_close(ctx, "stdin")
                if not child["reaped"]:
                    _send_signal(ctx, "SIGTERM")
                    if now >= ctx.failed_at + ctx.entry["deadlines"]["termination_grace_ns"]:
                        _send_signal(ctx, "SIGKILL")
                read_deadline = ctx.failed_at + ctx.entry["deadlines"]["termination_grace_ns"] + ctx.entry["deadlines"]["drain_ns"]
                if natural_deadline is not None:
                    read_deadline = min(read_deadline, natural_deadline)
            else:
                read_deadline = natural_deadline if natural_deadline is not None else ctx.deadline
            if now >= read_deadline:
                if ctx.failed_at is None:
                    ctx.fail(TimeoutError("Natural-exit drain deadline"), "pump", "drain", code="drain_deadline")
                for key in list(ctx.pipe_objects):
                    pipe = ctx.pipe_objects[key]
                    if pipe in selector.get_map():
                        selector.unregister(pipe)
                    if key != "stdin" and not ctx.result["streams"][key]["eof_observed"]:
                        ctx.result["streams"][key]["read_state"] = "abandoned"
                    _pipe_close(ctx, key)
                if not child["reaped"]:
                    _send_signal(ctx, "SIGKILL")
                    status = process.poll()
                    if status is not None:
                        child.update(state="reaped", reaped=True, returncode=status, wait_state="completed")
                    else:
                        child.update(state="unresolved", wait_state="pending")
                break
            wake = read_deadline
            if ctx.failed_at is not None and not child["reaped"]:
                grace = ctx.failed_at + ctx.entry["deadlines"]["termination_grace_ns"]
                if grace > now:
                    wake = min(wake, grace)
            for key, _ in selector.select(_poll_interval(now, wake)):
                if time.monotonic_ns() >= read_deadline:
                    break
                name, pipe = key.data, key.fileobj
                if time.monotonic_ns() >= ctx.deadline and ctx.failed_at is None:
                    ctx.deadline_failed = True
                    ctx.fail(TimeoutError("Completion deadline after selector wait"), "pump", "select", code="deadline")
                if name == "stdin":
                    if ctx.failed_at is not None:
                        continue
                    chunk = frame[source_offset:source_offset + BLOCK]
                    try:
                        n = os.write(pipe.fileno(), chunk)
                        _need(type(n) is int and 0 < n <= len(chunk), "Child stdin write count")
                        source_offset += n
                        sent_digest.update(chunk[:n])
                        ctx.transmission.update(acknowledged_bytes=source_offset,
                                                acknowledged_sha256=sent_digest.hexdigest(), state="partial")
                        if time.monotonic_ns() >= ctx.deadline:
                            ctx.deadline_failed = True
                            ctx.fail(TimeoutError("Completion deadline after stdin write"),
                                     "pump", "stdin_write", code="deadline")
                        if source_offset == len(frame):
                            selector.unregister(pipe)
                            _pipe_close(ctx, name)
                            if ctx.transmission["stdin_close_state"] == "succeeded":
                                ctx.transmission["state"] = "submitted"
                    except BlockingIOError:
                        continue
                    except BaseException as exc:
                        ctx.transmission.update(state="failed", unacknowledged_attempt_bytes=len(chunk))
                        ctx.fail(exc, "pump", "stdin_write", stream="stdin")
                    continue
                record = ctx.result["streams"][name]
                after_failure = ctx.failed_at is not None
                allowance = BLOCK - record["post_failure_received_bytes"] if after_failure else (
                    max(1, min(BLOCK, record["retention_limit"] - record["received_bytes"])))
                if allowance <= 0:
                    selector.unregister(pipe)
                    record["read_state"] = "abandoned"
                    _pipe_close(ctx, name)
                    continue
                try:
                    block = os.read(pipe.fileno(), min(BLOCK, allowance))
                except BlockingIOError:
                    continue
                except BaseException as exc:
                    record["read_state"] = "failed"
                    record["error_indices"].append(ctx.fail(exc, "pump", "receive", stream=name))
                    selector.unregister(pipe)
                    _pipe_close(ctx, name)
                    continue
                start = record["received_bytes"]
                receive_hashes[name].update(block)
                record.update(received_bytes=start + len(block), received_sha256=receive_hashes[name].hexdigest())
                if after_failure:
                    record["post_failure_received_bytes"] += len(block)
                if time.monotonic_ns() >= ctx.deadline and ctx.failed_at is None:
                    ctx.deadline_failed = True
                    ctx.fail(TimeoutError("Completion deadline after receive"), "pump", "receive", code="deadline")
                if not block:
                    record.update(read_state="eof", eof_observed=True, receive_scope="complete_stream",
                                  unread_bytes_unknown=False)
                    child[name + "_eof"] = True
                    selector.unregister(pipe)
                    _pipe_close(ctx, name)
                    continue
                if name == "stdout" and start < STDOUT_CAP:
                    output.extend(block[:STDOUT_CAP - start])
                eligible = len(block) if not after_failure else 0
                if record["received_bytes"] > record["retention_limit"]:
                    eligible = max(0, record["retention_limit"] - start) if not after_failure else 0
                    record["error_indices"].append(ctx.fail(ValueError("Child stream ceiling"), "pump", "overflow",
                                                           stream=name, code="stream_overflow"))
                if eligible and record["write_state"] != "failed":
                    prior_ack = record["acknowledged_prefix_bytes"]
                    try:
                        _write(ctx, "child_" + name + ".bin", block[:eligible], record)
                    except BaseException:
                        eligible = record["acknowledged_prefix_bytes"] - prior_ack + record["unknown_retention_bytes"]
                else:
                    eligible = 0
                discarded = block[eligible:]
                if discarded:
                    if record["discarded_suffix_start"] is None:
                        record["discarded_suffix_start"] = start + eligible
                    record["discarded_bytes"] += len(discarded)
                    discard_hashes[name].update(discarded)
                    record["discarded_sha256"] = discard_hashes[name].hexdigest()
        if child["reaped"]:
            _need(process.wait(timeout=0) == child["returncode"], "Owned child status")
    except BaseException as exc:
        ctx.fail(exc, "pump", "pump")
        _exceptional_pump_cleanup(ctx, natural_deadline, receive_hashes, discard_hashes, output)
    finally:
        try:
            if selector is not None:
                selector.close()
        except BaseException as exc:
            ctx.closure_uncertain = True
            ctx.fail(exc, "pump", "selector_close")
        for name in list(ctx.pipe_objects):
            _pipe_close(ctx, name)
        for record in ctx.result["streams"].values():
            record.pop("_ack_digest", None)
            if record["write_state"] != "failed":
                record["write_state"] = "finished"
    return bytes(output)


def _nullable(value, validator):
    if value is not None:
        validator(value)


def _sequence(value, validator):
    _need(type(value) is list, "Sequence representation")
    for item in value:
        validator(item)


def _text(value):
    _need(type(value) is str, "String representation")


def _bool(value):
    _need(type(value) is bool, "Boolean representation")


def _fingerprint_value(value):
    _need(type(value) is list and len(value) == 6 and all(type(x) is int for x in value)
          and all(x >= 0 for x in value[:4]), "Fingerprint representation")


def _descriptor(value):
    _fields(value, ("resolved_path", "lexical_before", "target_before", "opened",
                    "after_fd", "lexical_after", "target_after", "coherent"))
    _nullable(value["resolved_path"], _text)
    for key in ("lexical_before", "target_before", "opened", "after_fd", "lexical_after", "target_after"):
        _nullable(value[key], _fingerprint_value)
    _nullable(value["coherent"], _bool)


def _binding(value, asset=False):
    keys = ("role", "declared_path", "declared_size", "declared_sha256", "initial_descriptor",
            "final_descriptor", "whole_file_checks", "custody_state")
    _fields(value, keys + (("prior_attribution", "current_resolved_path", "regions") if asset else ()))
    _text(value["role"])
    _path(value["declared_path"])
    _nullable(value["declared_size"], lambda x: _integer(x, 1))
    _sha(value["declared_sha256"])
    _nullable(value["initial_descriptor"], _descriptor)
    _nullable(value["final_descriptor"], _descriptor)
    _need(value["custody_state"] in ("unacquired", "region_only", "full_verified", "failed", "invalidated"),
          "Binding custody state")
    _need(type(value["whole_file_checks"]) is list and len(value["whole_file_checks"]) == 2, "Whole checks")
    for phase, item in zip(("initial", "final"), value["whole_file_checks"]):
        _fields(item, ("phase", "state", "bytes_read", "observed_sha256", "expected_match"))
        _need(item["phase"] == phase and item["state"] in ("not_performed", "matched", "rejected", "failed"),
              "Whole verification state")
        _integer(item["bytes_read"])
        _nullable(item["observed_sha256"], _sha)
        _nullable(item["expected_match"], _bool)
        if item["state"] == "not_performed":
            _need(item["bytes_read"] == 0 and item["observed_sha256"] is None
                  and item["expected_match"] is None, "Unperformed whole check")
        if item["state"] in ("matched", "rejected"):
            _need(item["observed_sha256"] is not None
                  and item["expected_match"] is (item["state"] == "matched"), "Whole check match facts")
    if asset:
        _nullable(value["current_resolved_path"], _text)
        if value["prior_attribution"] is not None:
            prior = _fields(value["prior_attribution"], ("source_control", "source_id", "producer_path",
                            "producer_resolved_path", "size", "sha256", "digest_basis", "review_sha256"))
            for key in ("source_control", "source_id", "producer_path", "producer_resolved_path", "digest_basis"):
                _text(prior[key])
            _integer(prior["size"])
            _sha(prior["sha256"])
            _sha(prior["review_sha256"])
        _need(type(value["regions"]) is list and len(value["regions"]) <= 2, "Region sequence")
        for item in value["regions"]:
            _fields(item, ("name", "offset", "length", "sha256", "capture_state", "final_state",
                          "final_sha256", "byte_equal"))
            _need(item["name"] in ("prefix8", "header") and item["capture_state"] in (
                "not_performed", "captured", "failed") and item["final_state"] in (
                "not_performed", "matched", "rejected", "failed"), "Region states")
            _integer(item["offset"])
            _integer(item["length"])
            _nullable(item["sha256"], _sha)
            _nullable(item["final_sha256"], _sha)
            _nullable(item["byte_equal"], _bool)


def _issue(value, secondary=False):
    keys = ("stage", "role", "category", "kind", "code", "region", "offset",
            "operation", "errno", "message", "message_truncated")
    _fields(value, keys + (("primary_issue_index",) if secondary else ()))
    for key in ("stage", "category", "code", "message"):
        _text(value[key])
    _need(len(value["message"]) <= 1024, "Issue message limit")
    for key in ("role", "kind", "region", "operation"):
        _nullable(value[key], _text)
    _nullable(value["offset"], _integer)
    if value["errno"] is not None:
        _need(type(value["errno"]) is int, "Errno type")
    _bool(value["message_truncated"])
    if secondary:
        _integer(value["primary_issue_index"])


def _bookend(value):
    _fields(value, ("lexical_before", "target_before", "lexical_after", "target_after",
                    "resolved_before", "resolved_after", "errors", "coherent"))
    for key in ("lexical_before", "target_before", "lexical_after", "target_after"):
        _nullable(value[key], _fingerprint_value)
    for key in ("resolved_before", "resolved_after"):
        _nullable(value[key], _text)
    _nullable(value["coherent"], _bool)
    _need(type(value["errors"]) is list, "Bookend errors")
    for error in value["errors"]:
        _fields(error, ("bookend", "operation", "error_type", "errno", "message", "message_truncated"))
        _need(error["bookend"] in ("before", "after") and error["operation"] in ("lstat", "resolve", "stat"),
              "Bookend operation")
        _text(error["error_type"])
        _text(error["message"])
        _need(len(error["message"]) <= 1024, "Bookend message limit")
        _bool(error["message_truncated"])
        _need(error["errno"] is None or type(error["errno"]) is int, "Bookend errno")


_PROBE_KEYS = ("lexical_path", "lexical_type", "resolved_path", "resolved_type", "size_bytes",
               "probe_coherent", "status", "error_operation", "error_errno", "error_message")
_TARGET_KEYS = ("population_index", "metadata_index", "slide_id", "patient_id",
                "diagnostic_group", "expected_tile_rows")
_ROW_EXTRA = ("prior_stage1b", "prior_resolved_path", "prior_size_bytes", "prior_sha256",
              "prior_receipt_reference", "prior_path_match", "prior_size_match", "content_admission")


def _probe(value, row=False):
    _fields(value, _PROBE_KEYS + (_TARGET_KEYS + _ROW_EXTRA if row else ()))
    for key in ("lexical_path", "lexical_type", "status"):
        _text(value[key])
    for key in ("resolved_path", "resolved_type", "error_operation", "error_message"):
        _nullable(value[key], _text)
    _nullable(value["size_bytes"], _integer)
    _nullable(value["probe_coherent"], _bool)
    _need(value["error_errno"] is None or type(value["error_errno"]) is int, "Probe errno")
    _need(value["status"] in ("observed_regular", "missing_lexical", "broken_symlink",
          "inaccessible", "not_regular", "resolution_error", "changed_during_probe", "io_error"), "Probe status")
    if row:
        for key in ("population_index", "metadata_index", "expected_tile_rows"):
            _integer(value[key])
        for key in ("slide_id", "patient_id", "diagnostic_group"):
            _text(value[key])
        _bool(value["prior_stage1b"])
        for key in ("prior_resolved_path", "prior_receipt_reference"):
            _nullable(value[key], _text)
        _nullable(value["prior_sha256"], _sha)
        _nullable(value["prior_size_bytes"], _integer)
        for key in ("prior_path_match", "prior_size_match"):
            _nullable(value[key], _bool)
        _need(value["content_admission"] == "not_performed", "Inventory admission flag")


def _inventory(value):
    _fields(value, ("state", "visited_ids", "in_progress_id", "unvisited_ids", "probes",
                    "bookends", "frozen_result", "omitted_fields"))
    _need(value["state"] in ("not_started", "complete", "partial", "omitted_output_limit"), "Inventory state")
    if value["state"] == "omitted_output_limit":
        _need(all(value[k] is None for k in ("visited_ids", "in_progress_id", "unvisited_ids",
                                           "probes", "bookends", "frozen_result")), "Omitted inventory nulls")
        _sequence(value["omitted_fields"], _text)
        return
    _sequence(value["visited_ids"], _text)
    _nullable(value["in_progress_id"], _text)
    _nullable(value["unvisited_ids"], lambda x: _sequence(x, _text))
    _sequence(value["omitted_fields"], _text)
    _need(type(value["bookends"]) is dict, "Inventory bookends mapping")
    for item in value["bookends"].values():
        _bookend(item)
    if value["state"] == "complete":
        _need(value["probes"] is None and value["in_progress_id"] is None
              and value["unvisited_ids"] == [], "Complete inventory shape")
        frozen = _fields(value["frozen_result"], ("schema_version", "inventory_complete", "rows",
                         "status_counts", "content_admission", "limitations"))
        _need(type(frozen["schema_version"]) is int and frozen["schema_version"] == 1
              and frozen["inventory_complete"] is True and frozen["content_admission"] == "not_performed",
              "Frozen inventory flags")
        _sequence(frozen["rows"], lambda row: _probe(row, True))
        _need(len(frozen["rows"]) == 2778, "Inventory row count representation")
        _fields(frozen["status_counts"], ("observed_regular", "missing_lexical", "broken_symlink",
                "inaccessible", "not_regular", "resolution_error", "changed_during_probe", "io_error"))
        for count in frozen["status_counts"].values():
            _integer(count)
        _sequence(frozen["limitations"], _text)
    else:
        _need(value["frozen_result"] is None and type(value["probes"]) is dict, "Partial inventory shape")
        _need(set(value["probes"]) == set(value["visited_ids"]), "Visited probe membership")
        for probe in value["probes"].values():
            _probe(probe)
    _need(value["omitted_fields"] == [], "Unomitted inventory has omission fields")
    visited, progress, suffix = value["visited_ids"], value["in_progress_id"], value["unvisited_ids"]
    _need(len(visited) == len(set(visited)) and progress not in visited, "Repeated inventory identity")
    _need(set(value["bookends"]).issubset(set(visited) | ({progress} if progress is not None else set())),
          "Inventory bookends without published identity")
    _need(set(visited).issubset(value["bookends"]), "Visited identity without its bookend")
    if suffix is not None:
        partition = visited + ([progress] if progress is not None else []) + suffix
        _need(len(partition) == 2778 and len(set(partition)) == 2778, "Incoherent published inventory partition")
    elif value["state"] == "partial":
        _need(not visited and progress is None and value["probes"] == value["bookends"] == {},
              "Unknown suffix with observed identities")
    if value["state"] == "not_started":
        _need(visited == [] and progress is None and value["probes"] == value["bookends"] == {},
              "Unstarted inventory has observations")
    if value["state"] == "complete":
        _need([row["slide_id"] for row in value["frozen_result"]["rows"]] == visited, "Complete row/visited order")


def _observer_result(value, kind):
    if kind == "png":
        _fields(value, ("schema_version", "declared_size", "supplied_size", "png_framing_valid", "ihdr",
                       "chunks", "palette", "transparency", "colour", "exif", "apng", "unknown_chunks",
                       "consequential_properties", "unsupported_properties"))
        _fields(value["ihdr"], ("width", "height", "bit_depth", "colour_type", "compression_method",
                               "filter_method", "interlace_method"))
        ihdr = value["ihdr"]
        for key in ("width", "height"):
            _integer(ihdr[key], 1, 2**31 - 1)
        _need(type(ihdr["bit_depth"]) is int and ihdr["bit_depth"] in (1, 2, 4, 8, 16),
              "PNG bit-depth domain")
        _need(type(ihdr["colour_type"]) is int and ihdr["colour_type"] in (0, 2, 3, 4, 6),
              "PNG colour-type domain")
        depths = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
        _need(ihdr["bit_depth"] in depths[ihdr["colour_type"]], "PNG colour/bit-depth relationship")
        for key in ("compression_method", "filter_method"):
            _integer(ihdr[key], 0, 0)
        _integer(ihdr["interlace_method"], 0, 1)
        _fields(value["palette"], ("present", "entry_count"))
        _bool(value["palette"]["present"])
        _nullable(value["palette"]["entry_count"], lambda n: _integer(n, 1, 256))
        _need(value["palette"]["present"] is (value["palette"]["entry_count"] is not None),
              "PNG palette presence")
        if value["palette"]["present"]:
            _need(ihdr["colour_type"] in (2, 3, 6), "PNG palette colour type")
        if ihdr["colour_type"] == 3:
            _need(value["palette"]["present"] and value["palette"]["entry_count"] <= 2**ihdr["bit_depth"],
                  "PNG indexed palette domain")
        _fields(value["transparency"], ("present", "kind", "raw_samples_or_alpha_table"))
        _bool(value["transparency"]["present"])
        _nullable(value["transparency"]["kind"], _text)
        transparency = value["transparency"]
        samples = transparency["raw_samples_or_alpha_table"]
        if not transparency["present"]:
            _need(transparency["kind"] is None and samples is None, "Absent PNG transparency")
        else:
            _need(transparency["kind"] in ("grayscale_key", "rgb_key", "palette_alpha"),
                  "PNG transparency kind")
            _need(transparency["kind"] == {0: "grayscale_key", 2: "rgb_key", 3: "palette_alpha"}.get(
                ihdr["colour_type"]), "PNG transparency colour type")
            ceiling = 255 if transparency["kind"] == "palette_alpha" else 2**ihdr["bit_depth"] - 1
            _sequence(samples, lambda n: _integer(n, 0, ceiling))
            if transparency["kind"] == "palette_alpha":
                _need(value["palette"]["entry_count"] is not None
                      and 1 <= len(samples) <= value["palette"]["entry_count"], "PNG alpha-table domain")
            else:
                _need(len(samples) == (1 if transparency["kind"] == "grayscale_key" else 3),
                      "PNG transparency sample count")
        colour = _fields(value["colour"], ("gama_u32", "chrm_u32_values", "srgb_intent", "iccp_descriptor"))
        _nullable(colour["gama_u32"], lambda n: _integer(n, 1, 2**32 - 1))
        _nullable(colour["srgb_intent"], lambda n: _integer(n, 0, 3))
        if colour["chrm_u32_values"] is not None:
            _sequence(colour["chrm_u32_values"], lambda n: _integer(n, 0, 2**32 - 1))
            _need(len(colour["chrm_u32_values"]) == 8, "PNG chromaticity field count")
        if colour["iccp_descriptor"] is not None:
            icc = _fields(colour["iccp_descriptor"], ("keyword", "compression_method", "compressed_length", "compressed_sha256"))
            _text(icc["keyword"])
            _integer(icc["compression_method"], 0, 0)
            _integer(icc["compressed_length"], 1, SOURCE_CAP)
            _sha(icc["compressed_sha256"])
        exif = _fields(value["exif"], ("present", "length", "data_sha256", "orientation", "orientation_status"))
        _bool(exif["present"])
        _nullable(exif["length"], lambda n: _integer(n, 0, SOURCE_CAP))
        _nullable(exif["data_sha256"], _sha)
        _need(exif["orientation"] is None, "PNG EXIF orientation remains uninterpreted")
        _text(exif["orientation_status"])
        apng = _fields(value["apng"], ("present", "actl_raw_fields", "chunk_descriptors", "semantic_status"))
        _bool(apng["present"])
        _text(apng["semantic_status"])
        if apng["actl_raw_fields"] is not None:
            _fields(apng["actl_raw_fields"], ("num_frames", "num_plays"))
            for n in apng["actl_raw_fields"].values():
                _integer(n, 0, 2**32 - 1)
        for descriptors in (value["chunks"], apng["chunk_descriptors"]):
            _need(type(descriptors) is list and len(descriptors) <= 1024, "PNG descriptors")
            for item in descriptors:
                _fields(item, ("index", "offset", "type", "length", "crc_valid", "data_sha256",
                               "critical", "recognised", "interpretation"))
                _integer(item["index"], 0, 1023)
                _integer(item["offset"], 8, SOURCE_CAP)
                _integer(item["length"], 0, SOURCE_CAP)
                for key in ("crc_valid", "critical", "recognised"):
                    _bool(item[key])
                for key in ("type", "interpretation"):
                    _text(item[key])
                _sha(item["data_sha256"])
        for key in ("declared_size", "supplied_size"):
            _integer(value[key], 0, SOURCE_CAP)
        _bool(value["png_framing_valid"])
        _sequence(value["unknown_chunks"], _integer)
    elif kind == "safetensors":
        _fields(value, ("schema_version", "total_size", "header_length", "header_sha256", "payload_offset",
                       "payload_size", "header_structure_valid", "metadata", "tensors", "tensor_count",
                       "scalar_count", "payload_bytes_by_dtype", "unsupported_properties"))
        for key in ("total_size", "header_length", "payload_offset", "payload_size", "tensor_count", "scalar_count"):
            _integer(value[key])
        _sha(value["header_sha256"])
        _bool(value["header_structure_valid"])
        if value["metadata"] is not None:
            _need(type(value["metadata"]) is dict and len(value["metadata"]) <= 128, "Tensor metadata")
            for k, v in value["metadata"].items():
                _need(type(v) is str and len(k.encode("utf-8")) <= 256 and len(v.encode("utf-8")) <= 4096,
                      "Tensor metadata strings")
        _need(type(value["tensors"]) is list and len(value["tensors"]) <= 4096, "Tensor count")
        for item in value["tensors"]:
            _fields(item, ("name", "dtype", "shape", "data_offsets", "numel", "nbytes"))
            _need(type(item["name"]) is str and len(item["name"].encode("utf-8")) <= 256, "Tensor name")
            _text(item["dtype"])
            _need(type(item["shape"]) is list and len(item["shape"]) <= 8, "Tensor rank")
            for dimension in item["shape"]:
                _integer(dimension, 0, 2**31 - 1)
            _need(type(item["data_offsets"]) is list and len(item["data_offsets"]) == 2, "Offsets representation")
            for key in ("data_offsets",):
                _sequence(item[key], _integer)
            _integer(item["numel"])
            _integer(item["nbytes"])
        _need(type(value["payload_bytes_by_dtype"]) is dict, "Dtype totals representation")
        for n in value["payload_bytes_by_dtype"].values():
            _integer(n)
    else:
        _fields(value, ("match", "missing_names", "unexpected_names", "shape_mismatches", "dtype_issues"))
        _bool(value["match"])
        _sequence(value["missing_names"], _text)
        _sequence(value["unexpected_names"], _text)
        for item in value["shape_mismatches"]:
            _fields(item, ("name", "expected_shape", "observed_shape"))
            _text(item["name"])
            _sequence(item["expected_shape"], _integer)
            _sequence(item["observed_shape"], _integer)
        for item in value["dtype_issues"]:
            _fields(item, ("name", "dtype"))
            _text(item["name"])
            _text(item["dtype"])
        return
    _need(type(value["schema_version"]) is int and value["schema_version"] == 1, "Observer version")
    _sequence(value["unsupported_properties"], _text)
    if kind == "png":
        _sequence(value["consequential_properties"], _text)


def _issue_outcome(issue: dict) -> int:
    category, kind = issue["category"], issue["kind"]
    _need(category in ("observation", "availability", "unexpected_io", "execution",
                       "declaration", "integrity", "custody", "output"), "Caller issue category")
    _need(kind is None or category == "observation"
          and kind in ("malformed", "unsupported", "limit", "declaration"), "Primitive issue kind")
    return 2 if category in ("observation", "availability") and kind != "declaration" else 1


def _cap_channels(packet: dict, omitted: set[str]) -> tuple[dict, dict, int, int]:
    overflow = packet["overflow"]
    ng, ns = overflow["global_issue_attempts"], overflow["secondary_error_attempts"]
    og, os_ = overflow["omitted_counts"]["issues"], overflow["omitted_counts"]["secondary_errors"]
    _need(len(packet["issues"]) == min(ng, 128) and og == max(0, ng - 127)
          and len(packet["secondary_errors"]) == min(ns, 32) and os_ == max(0, ns - 31),
          "Coherent source issue channels")
    markers = {}
    for channel, records, count, code in (
            ("global", packet["issues"], og, "issue_limit"),
            ("secondary", packet["secondary_errors"], os_, "secondary_limit")):
        if count:
            marker = records[-1]
            _need(marker["code"] == code and marker["category"] == "execution"
                  and marker["kind"] is None and marker["message"] == "Bounded issue evidence overflow"
                  and marker["message_truncated"] is False, "Reserved source cap marker")
            markers[channel] = marker
        _need(all(record["code"] not in ("issue_limit", "secondary_limit")
                  for record in (records[:-1] if count else records)), "Misplaced reserved cap marker")
    if og or os_:
        _need(overflow["occurred"] is True, "Suppressed issues without error overflow")
    if not omitted:
        if overflow["occurred"]:
            _need((overflow["cause_code"] == "issue_limit" and og > 0
                   or overflow["cause_code"] == "secondary_limit" and os_ > 0),
                  "Unexplained non-serializer overflow")
        else:
            _need(overflow["cause_code"] is None and all(n == 0 for n in overflow["omitted_counts"].values())
                  and overflow["serialization_bytes_seen"] is None, "False overflow retains failure fields")
    global_later, secondary_later = max(0, og - 1), max(0, os_ - 1)
    ordinary_serialization = [(index, issue) for index, issue in enumerate(packet["issues"])
                              if issue["stage"] == "serialization" and issue["code"] != "issue_limit"]
    _need(not any(issue["stage"] == "serialization" for issue in packet["secondary_errors"]),
          "Secondary attempt cannot belong to single-run serialization")
    if omitted:
        cause = overflow["cause_code"]
        serialization = packet["stages"][-1]
        if ng <= 127:
            _need(ng > 0 and len(ordinary_serialization) == 1
                  and ordinary_serialization[0][0] == ng - 1 and packet["issues"][-1]["stage"] == "serialization"
                  and packet["issues"][-1]["category"] == "output"
                  and packet["issues"][-1]["code"] == cause
                  and ng - 1 in serialization["issue_indices"],
                  "Missing last ordinary GLOBAL serialization attempt")
        elif ng == 128:
            _need(not ordinary_serialization and markers["global"]["stage"] == "serialization",
                  "GLOBAL reserved attempt is not serialization")
        else:
            _need(not ordinary_serialization and markers["global"]["stage"] != "serialization" and global_later > 0,
                  "Serialization reserved marker cannot precede later GLOBAL attempts")
            global_later -= 1
    else:
        _need(not ordinary_serialization and all(marker["stage"] != "serialization" for marker in markers.values()),
              "Serialization issue without serializer failure")
    return markers, {"global": ng, "secondary": ns}, global_later, secondary_later


def _capped_phases(packet: dict, omitted: set[str], stages: dict) -> set[str]:
    markers, _, global_later, secondary_later = _cap_channels(packet, omitted)
    explained, missing = set(), []
    for name, stage in stages.items():
        _need(all(index < len(packet["issues"]) for index in stage["issue_indices"]),
              "Out-of-range phase issue index")
        if stage["state"] not in ("failed", "partial") or name == "serialization" and omitted:
            continue
        if any(_issue_outcome(packet["issues"][index]) == 1 for index in stage["issue_indices"]):
            explained.add(name)
            continue
        if stage["state"] == "partial" and stage["issue_indices"]:
            explained.add(name)
            continue
        channels = ("global",) if name in ("initial_inventory", "final_inventory") else ("global", "secondary")
        if any(channel in markers and markers[channel]["stage"] == name for channel in channels):
            explained.add(name)
        else:
            missing.append((name, channels))
    def eligible(channel, name):
        if channel not in markers:
            return False
        previous = markers[channel]["stage"]
        return (STAGES.index(name) >= STAGES.index(previous)
                or previous == "checkpoint_full" and name == "checkpoint_header")
    # Inventory failures need their own global attempts, not secondary capacity.
    for name, channels in sorted(missing, key=lambda item: len(item[1])):
        if "global" in channels and global_later and eligible("global", name):
            global_later -= 1
        elif "secondary" in channels and secondary_later and eligible("secondary", name):
            secondary_later -= 1
        else:
            raise ValueError("No attributable source-channel attempt for " + name)
        explained.add(name)
    return explained


def _published_ids(value: list) -> list:
    _need(type(value) is list and len(value) == 2778 and len(set(value)) == 2778,
          "Published target representation")
    return value


def _publication_failure(packet: dict, stages: dict, omitted: set[str]) -> bool:
    basis = stages["trusted_basis"]
    if basis["state"] != "failed":
        return False
    asset_names = ("template", "config", "checkpoint_header", "checkpoint_full", "selected_wsi_full")
    _need(all(stages[name]["state"] == "not_started" for name in asset_names),
          "Asset-work stage after failed trusted basis")
    assets = packet["asset_records"]
    _need(packet["exit_code"] == 1, "Failed publication cannot complete")
    prefix = [record["prior_attribution"] is not None for record in assets]
    count = sum(prefix)
    _need(prefix == [True] * count + [False] * (len(prefix) - count), "Non-prefix prior publication")
    for record in assets:
        _need(record["current_resolved_path"] is None and record["initial_descriptor"] is None
              and record["final_descriptor"] is None and record["custody_state"] == "unacquired"
              and record["regions"] == [] and all(check == dict(phase=phase, state="not_performed",
                  bytes_read=0, observed_sha256=None, expected_match=None)
                  for check, phase in zip(record["whole_file_checks"], ("initial", "final"))),
              "Current asset work on failed publication")
    _need(all(value == 0 for key, value in packet["logical_io"]["selected"].items() if key != "limit_bytes")
          and not any(read["role"] in ASSETS for read in packet["logical_io"]["reads"]),
          "Selected IO on failed publication")
    initial, final, comparison = packet["inventory_initial"], packet["inventory_final"], packet["inventory_comparison"]
    final_rows = packet["overflow"]["omitted_counts"]["inventory_final_rows"]
    final_work = (final["state"] in ("partial", "complete") or comparison["state"] == "partial"
                  or final_rows > 0 or stages["final_inventory"]["state"] == "failed")
    witness = count or basis["result"] is not None or "trusted_basis" in omitted
    witness = witness or initial["unvisited_ids"] is not None or final_work
    witness = witness or comparison["uncomparable_ids"] is not None or (
        packet["overflow"]["omitted_counts"]["inventory_final_rows"] > 0)
    bindings = [packet["entry_binding"], packet["observer_binding"], *packet["control_bindings"]]
    initial_sources_consumed = all(
        bound is not None and bound["whole_file_checks"][0]["expected_match"] is True
        and bound["initial_descriptor"] is not None and bound["initial_descriptor"]["coherent"] is True
        for bound in bindings)
    if witness:
        _need(packet["run_id"] is not None and len(assets) == 4, "Publication lacks full G1")
        _need(initial_sources_consumed, "Publication lacks consumed initial sources")
    controls = {bound["role"]: bound for bound in packet["control_bindings"]}
    for record in assets[:count]:
        prior = record["prior_attribution"]
        template = record["role"] == "template"
        identity = {"model_config": "model_config/h_optimus_1", "checkpoint": "weight/h_optimus_1",
                    "selected_wsi": "wsi/10003"}.get(record["role"], "source:template")
        review = "cpu_review" if template else "stage1b_review"
        _need(prior["source_control"] == ("cpu_input_bindings" if template else "stage1b_source_inventory")
              and prior["source_id"] == identity and prior["review_sha256"] == controls[review]["declared_sha256"]
              and (prior["size"], prior["sha256"]) == (record["declared_size"], record["declared_sha256"])
              and (not template or prior["digest_basis"] == "accepted_cpu57_source_binding"),
              "Published prior declaration/reference mismatch")
    _need(stages["initial_inventory"]["state"] in ("not_started", "omitted_output_limit"),
          "Initial inventory on failed publication")
    universes = []
    if initial["state"] != "omitted_output_limit":
        _need(initial["state"] == "not_started", "Observed initial pass on failed publication")
        if initial["unvisited_ids"] is not None:
            _need(count == 4, "Partition before complete prior prefix")
            universes.append(_published_ids(initial["unvisited_ids"]))
    if final["state"] == "not_started":
        if final["unvisited_ids"] is not None:
            _need(count == 4 and initial["state"] == "not_started" and initial["unvisited_ids"] is not None,
                  "Reversed final constructor publication")
            universes.append(_published_ids(final["unvisited_ids"]))
        _need(stages["final_inventory"]["state"] in ("not_started", "blocked", "failed"),
              "Unstarted final pass with completed phase")
    elif final["state"] in ("partial", "complete"):
        _need(witness and stages["final_checks"]["state"] not in ("not_started", "blocked")
              and stages["final_inventory"]["state"] in ("partial", "complete", "negative", "failed"),
              "Final-only inventory without returned final checks")
        ids = (final["visited_ids"] if final["state"] == "complete" else
               final["visited_ids"] + ([final["in_progress_id"]] if final["in_progress_id"] is not None else [])
               + final["unvisited_ids"] if final["unvisited_ids"] is not None else None)
        if ids is not None:
            universes.append(_published_ids(ids))
    if comparison["state"] != "omitted_output_limit":
        _need(comparison["compared_ids"] == comparison["changed_ids"] == comparison["changes"] == [],
              "Mutation/comparison of unobserved initial pass")
        if comparison["state"] == "not_performed":
            if comparison["uncomparable_ids"] is not None:
                _need(count == 4 and initial["unvisited_ids"] is not None and final["state"] != "not_started"
                      or count == 4 and initial["unvisited_ids"] is not None and final["unvisited_ids"] is not None,
                      "Reversed comparison constructor publication")
                universes.append(_published_ids(comparison["uncomparable_ids"]))
        else:
            _need(comparison["state"] == "partial" and final["state"] in ("partial", "complete"),
                  "Final comparison without returned final-only pass")
            universes.append(_published_ids(comparison["uncomparable_ids"]))
    _need(not universes or all(value == universes[0] for value in universes), "Conflicting visible target universes")
    if basis["result"] is not None:
        _need(count == 4 and initial["unvisited_ids"] is not None
              and final["state"] != "omitted_output_limit" and comparison["uncomparable_ids"] is not None,
              "Result before complete constructor publication")
        _need(basis["result"]["expected"] == dict(raw_slides=2921, target_slides=2778, patients=1876,
              groups=15, cross_label_patients=45, tile_rows=3835507), "Published basis fixed declarations")
    if "trusted_basis" in omitted:
        _need(count == 4, "Removed result before complete prior publication")
    checks = stages["final_checks"]
    if final_work:
        _need(checks["state"] in ("complete", "negative", "failed")
              and (checks["result"] is not None or "final_checks" in omitted),
              "Final-only work without returning final-check result")
    if checks["result"] is not None:
        roles = ["entry", "spec", "observer", *CONTROLS]
        measured = checks["result"]
        _need(measured["checked_roles"] == roles[:len(measured["checked_roles"])]
              and measured["region_only_roles"] == []
              and all(role in measured["checked_roles"] for role in measured["failed_roles"]),
              "Failed-publication final asset/role checks")
        if final_work or witness and checks["state"] == "complete":
            _need(measured["checked_roles"] == roles, "Completed publication final checks omit consumed roles")
        if witness and checks["state"] == "complete":
            _need(stages["final_inventory"]["state"] != "not_started",
                  "Returned complete checks skipped known-ID finalization")
    if checks["state"] == "blocked":
        _need(checks["blocker_codes"] == ["interrupted"] and stages["final_inventory"]["state"] in (
            "blocked", "omitted_output_limit"), "Stopped finalization entered inventory")
    inventory_stage = stages["final_inventory"]
    inventory_blocked = inventory_stage["state"] == "blocked" or (
        inventory_stage["state"] == "omitted_output_limit"
        and inventory_stage["blocker_codes"] == ["interrupted"] and inventory_stage["issue_indices"] == [])
    if inventory_blocked:
        _need(inventory_stage["blocker_codes"] == ["interrupted"] and inventory_stage["issue_indices"] == [],
              "Unknown blocked finalization")
        _need(final_rows == 0, "Stopped final inventory cannot omit observations")
        _need(checks["state"] in ("blocked", "failed"), "Stopped inventory without stopped or interrupted checks")
        if checks["state"] == "failed":
            _need(checks["result"] is not None or "final_checks" in omitted,
                  "Blocked returned inventory without initialized final checks")
            if len(bindings) == len(CONTROLS) + 3 and initial_sources_consumed:
                _need("final_checks" in omitted or bool(checks["result"]["checked_roles"]),
                      "Blocked returned inventory without an attempted consumed role")
    if omitted:
        _need(packet["overflow"]["omitted_counts"]["inventory_initial_rows"] == 0,
              "Omitted observed initial rows on failed publication")
        if checks["state"] in ("not_started", "blocked"):
            _need(final_rows == 0, "Stopped final checks cannot omit final observations")
    return True


def _serialization_omission(packet: dict) -> set[str]:
    overflow = packet["overflow"]
    inventories = (packet["inventory_initial"], packet["inventory_final"])
    comparison = packet["inventory_comparison"]
    sections = overflow["omitted_sections"]
    requested = bool(sections) or any(i["state"] == "omitted_output_limit" for i in inventories)
    requested = requested or comparison["state"] == "omitted_output_limit"
    if not requested:
        _need(all(s["state"] != "omitted_output_limit" for s in packet["stages"]),
              "Unbound omitted stage")
        return set()
    _need(packet["exit_code"] == 1 and overflow["occurred"] is True
          and type(overflow["cause_code"]) is str and bool(overflow["cause_code"]),
          "Serialization omission outcome")
    _need(sections[:3] == ["inventory_initial", "inventory_final", "inventory_comparison"]
          and len(sections) == len(set(sections)), "Serialization omission sections")
    omitted_inventory_keys = {"visited_ids", "in_progress_id", "unvisited_ids", "probes",
                              "bookends", "frozen_result", "omitted_fields"}
    for inventory in inventories:
        _need(inventory["state"] == "omitted_output_limit"
              and set(inventory["omitted_fields"]) == omitted_inventory_keys
              and len(inventory["omitted_fields"]) == len(omitted_inventory_keys),
              "Serializer inventory omission membership")
    _need(comparison["state"] == "omitted_output_limit", "Serializer comparison omission")
    for key in ("inventory_initial_rows", "inventory_final_rows"):
        _integer(overflow["omitted_counts"][key], 0, 2778)
    result_names = ("trusted_basis", "template", "config", "checkpoint_header", "final_checks", "serialization")
    valid_sections = {"stages." + name + ".result": name for name in result_names}
    _need(all(section in valid_sections for section in sections[3:]), "Unknown stage-result omission")
    names = [valid_sections[section] for section in sections[3:]]
    _need(names == [name for name in result_names if name in names]
          and "serialization" in names, "Ordered serializer result omissions")
    stages = {stage["name"]: stage for stage in packet["stages"]}
    _need(all(stage["result"] is None for stage in stages.values()), "Serializer retained stage result")
    _need(stages["initial_inventory"]["state"] == stages["final_inventory"]["state"] == "omitted_output_limit",
          "Serializer inventory stage states")
    serialization = stages["serialization"]
    _need(serialization["state"] == "failed"
          and serialization["blocker_codes"] == [overflow["cause_code"]], "Serializer failure stage")
    return set(names)


def _packet_relationships(packet: dict) -> None:
    stages = {stage["name"]: stage for stage in packet["stages"]}
    omitted = _serialization_omission(packet)
    capped = _capped_phases(packet, omitted, stages)
    basis = stages["trusted_basis"]
    reached_basis = basis["result"] is not None or "trusted_basis" in omitted
    reached_declaration = packet["run_id"] is not None
    state_domains = {
        "trusted_basis": ("not_started", "complete", "failed"),
        "initial_inventory": ("not_started", "partial", "complete", "negative", "failed", "omitted_output_limit"),
        "template": ("not_started", "complete", "negative", "failed"),
        "config": ("not_started", "complete", "negative", "failed"),
        "checkpoint_header": ("not_started", "complete", "negative", "failed"),
        "checkpoint_full": ("not_started", "complete", "failed", "blocked"),
        "selected_wsi_full": ("not_started", "complete", "negative", "failed", "blocked"),
        "final_checks": ("not_started", "complete", "negative", "failed", "blocked"),
        "final_inventory": ("not_started", "partial", "complete", "negative", "failed", "blocked", "omitted_output_limit"),
        "serialization": ("complete", "failed"),
    }
    issues, secondary = packet["issues"], packet["secondary_errors"]
    outcomes = [_issue_outcome(issue) for issue in issues + secondary]
    mutation = packet["inventory_comparison"]["state"] != "omitted_output_limit" and (
        packet["inventory_comparison"]["changed_ids"] or packet["inventory_comparison"]["changes"])
    expected_outcome = 1 if 1 in outcomes or packet["overflow"]["occurred"] or mutation else 2 if 2 in outcomes else 0
    _need(packet["exit_code"] == expected_outcome, "Caller technical/negative outcome precedence")
    for index, issue in enumerate(issues):
        _need(issue["stage"] in stages, "Issue stage identity")
        stage = stages[issue["stage"]]
        if issue["code"] not in ("issue_limit", "secondary_limit"):
            _need(index in stage["issue_indices"], "Missing primary stage issue link")
            _need(issue["code"] in stage["blocker_codes"], "Missing primary blocker code")
    for issue in secondary:
        _need(issue["stage"] in stages and issue["primary_issue_index"] < len(issues),
              "Secondary primary/stage link")
    for name, stage in stages.items():
        _need(stage["state"] in state_domains[name], "Source-derived stage state")
        indices = stage["issue_indices"]
        _need(len(indices) == len(set(indices))
              and all(index < len(issues) and issues[index]["stage"] == name for index in indices),
              "Primary issue-index ownership")
        if stage["state"] in ("not_started", "blocked"):
            _need(stage["result"] is None and name not in omitted and not indices,
                  "Unreached/blocked measured result")
        if stage["state"] == "not_started":
            _need(stage["blocker_codes"] == [], "Unstarted blocker evidence")
        if stage["state"] == "blocked":
            _need(bool(stage["blocker_codes"]), "Unexplained blocked stage")
        if stage["state"] == "complete":
            _need(not indices, "Completed stage retains primary failure")
            if name in ("trusted_basis", "template", "config", "checkpoint_header", "final_checks", "serialization"):
                _need(stage["result"] is not None or name in omitted, "Unexplained completed-result absence")
        if stage["state"] == "negative":
            _need(bool(indices) and _issue_outcome(issues[indices[-1]]) == 2, "Unexplained negative stage")
        if stage["state"] == "failed":
            _need(name in capped or name == "serialization" and omitted, "Unexplained failed stage")
    if not reached_declaration:
        _need(packet["exit_code"] == 1 and not reached_basis, "G0 cannot complete an observation")
    else:
        _need(packet["entry_binding"] is not None
              and packet["control_bindings"][0]["whole_file_checks"][0]["expected_match"] is True,
              "Declared spec gate lacks acquired matching bytes")
    failed_publication = _publication_failure(packet, stages, omitted)
    if failed_publication:
        pass
    elif reached_basis:
        _need(reached_declaration and basis["state"] == "complete"
              and all(record["prior_attribution"] is not None for record in packet["asset_records"]),
              "Trusted basis/prior declaration gate")
    else:
        _need(basis["state"] != "complete", "Missing trusted-basis result")
        for record in packet["asset_records"]:
            _need(record["prior_attribution"] is None and record["current_resolved_path"] is None
                  and record["initial_descriptor"] is None and record["final_descriptor"] is None
                  and record["custody_state"] == "unacquired" and record["regions"] == []
                  and all(check["state"] == "not_performed" for check in record["whole_file_checks"]),
                  "Pre-G2 acquired/prior asset facts")
        for name in ("template", "config", "checkpoint_header", "checkpoint_full", "selected_wsi_full"):
            _need(stages[name]["state"] == "not_started" and stages[name]["result"] is None,
                  "Pre-G2 downstream observations")
        for slot, name in (("inventory_initial", "initial_inventory"), ("inventory_final", "final_inventory")):
            inventory = packet[slot]
            if inventory["state"] == "omitted_output_limit":
                _need(bool(omitted) and packet["overflow"]["omitted_counts"][slot + "_rows"] == 0,
                      "Pre-G2 omitted observed rows")
            else:
                _need(inventory == {"state": "not_started", "visited_ids": [], "in_progress_id": None,
                      "unvisited_ids": None, "probes": {}, "bookends": {}, "frozen_result": None,
                      "omitted_fields": []}, "Unknown pre-G2 inventory")
                _need(stages[name]["state"] in ("not_started", "blocked"), "Pre-G2 inventory phase")
        comparison = packet["inventory_comparison"]
        if comparison["state"] != "omitted_output_limit":
            _need(comparison == {"state": "not_performed", "compared_ids": [], "changed_ids": [],
                  "changes": [], "uncomparable_ids": None}, "Unknown pre-G2 comparison")
    if packet["exit_code"] in (0, 2):
        _need(reached_basis and not omitted and stages["serialization"]["state"] == "complete",
              "Completed observation gate")
        _need(all(packet[slot]["state"] == "complete" for slot in ("inventory_initial", "inventory_final"))
              and packet["inventory_comparison"]["state"] == "complete", "Completed inventory phases")
        _need(all(stage["state"] in ("complete", "negative", "blocked") for stage in stages.values()),
              "Completed observation has unfinished phases")
    if packet["exit_code"] == 0:
        _need(all(stage["state"] == "complete" for stage in stages.values()), "Positive observation has blockers")
    for slot, name in (("inventory_initial", "initial_inventory"), ("inventory_final", "final_inventory")):
        if stages[name]["state"] in ("complete", "negative"):
            _need(packet[slot]["state"] == "complete", "Completed inventory stage lacks completed inventory")
        if stages[name]["state"] == "partial":
            _need(name in capped, "Unexplained partial inventory stage")
    for name, role in (("template", "template"), ("config", "model_config"),
                       ("checkpoint_full", "checkpoint"), ("selected_wsi_full", "selected_wsi")):
        stage = stages[name]
        if stage["state"] == "complete":
            record = next(record for record in packet["asset_records"] if record["role"] == role)
            _need(record["whole_file_checks"][0]["expected_match"] is True,
                  "Completed asset route lacks initial whole-byte match")
    if stages["config"]["state"] == "complete" and stages["config"]["result"] is not None:
        _need(stages["config"]["result"]["raw_utf8"] is not None
              and stages["config"]["result"]["parsed_json"] is not None, "Completed config parsing")
    if stages["template"]["state"] == "complete" and stages["template"]["result"] is not None:
        _need(stages["template"]["result"]["unsupported_properties"] == [], "Unsupported positive PNG stage")
    if stages["final_checks"]["state"] == "complete" and stages["final_checks"]["result"] is not None:
        _need(stages["final_checks"]["result"]["failed_roles"] == [], "Completed final checks retain failed roles")
    header = stages["checkpoint_header"]
    if header["state"] == "complete" and header["result"] is not None:
        _need(header["result"]["safetensors_observation"] is not None
              and header["result"]["schema_comparison"] is not None
              and header["result"]["schema_comparison"]["match"] is True, "Completed checkpoint-header stage")


def _validate_packet(encoded: bytes, status: int | None, entry: dict, spec: dict) -> tuple[dict, dict]:
    summary = _packet_summary()
    if not encoded:
        raise ValueError("No packet observed; internal packet=None not established")
    summary["state"] = "invalid"
    packet = _json(encoded, 20, newline=True)
    summary["canonical"] = True
    _fields(packet, ("schema_version", "task", "scope", "run_id", "spec_sha256", "packet_kind", "exit_code",
                     "model_use_authorized", "entry_binding", "observer_binding", "control_bindings", "asset_records",
                     "stages", "inventory_initial", "inventory_final", "inventory_comparison", "issues",
                     "secondary_errors", "overflow", "logical_io", "limitations"))
    _need(type(packet["schema_version"]) is int and packet["schema_version"] == 1
          and packet["task"] == "hoptimus_encoded_input_observation"
          and packet["scope"] == "encoded_input_observations_and_selected_integrity_only"
          and packet["model_use_authorized"] is False, "Packet envelope")
    _need(type(packet["exit_code"]) is int and packet["exit_code"] in (0, 1, 2)
          and packet["packet_kind"] == {0: "complete", 1: "partial_error", 2: "complete_negative"}[packet["exit_code"]],
          "Packet status mapping")
    _need(packet["limitations"] == CALLER_LIMITATIONS, "Frozen limitations")
    summary.update(packet_kind=packet["packet_kind"], packet_exit_code=packet["exit_code"],
                   model_use_authorized=False, limitations_match=True)
    _nullable(packet["run_id"], _text)
    _nullable(packet["spec_sha256"], _sha)
    for key in ("entry_binding", "observer_binding"):
        _nullable(packet[key], _binding)
    _sequence(packet["control_bindings"], _binding)
    _sequence(packet["asset_records"], lambda x: _binding(x, True))
    if packet["entry_binding"] is not None:
        bound = packet["entry_binding"]
        _need((bound["declared_path"], bound["declared_size"], bound["declared_sha256"])
              == (entry["path"], entry["size"], entry["sha256"]), "Entry declaration mismatch")
        summary["entry_binding_match"] = True
    _need(packet["spec_sha256"] is None or packet["spec_sha256"] == spec["sha256"], "Spec SHA mismatch")
    if packet["spec_sha256"] is not None:
        summary["spec_binding_match"] = True
    controls = packet["control_bindings"]
    if controls:
        _need(controls[0]["role"] == "spec" and controls[0]["declared_path"] == spec["path"]
              and controls[0]["declared_sha256"] == spec["sha256"], "Spec source declaration")
    if packet["run_id"] is None:
        _need(packet["observer_binding"] is None and packet["asset_records"] == []
              and len(controls) <= 1, "G0 membership")
    else:
        _need(type(packet["run_id"]) is str and re.fullmatch("[A-Za-z0-9_.-]{1,128}", packet["run_id"]),
              "Run ID")
        _need(packet["observer_binding"] is not None and [r["role"] for r in controls] == ["spec", *CONTROLS]
              and [r["role"] for r in packet["asset_records"]] == list(ASSETS), "G1 membership")
    _need(type(packet["stages"]) is list and len(packet["stages"]) == 10, "Ten stage records")
    for expected, stage in zip(STAGES, packet["stages"]):
        _fields(stage, ("name", "state", "result", "blocker_codes", "issue_indices"))
        _need(stage["name"] == expected and stage["state"] in ("not_started", "complete", "partial",
              "failed", "negative", "blocked", "omitted_output_limit"), "Stage representation")
        _sequence(stage["blocker_codes"], _text)
        _sequence(stage["issue_indices"], _integer)
        result = stage["result"]
        if result is None:
            continue
        if expected in ("initial_inventory", "final_inventory", "checkpoint_full", "selected_wsi_full"):
            raise ValueError("Unembedded stage result must be null")
        if expected == "trusted_basis":
            _fields(result, ("expected", "prior_wsi_ids"))
            _fields(result["expected"], ("raw_slides", "target_slides", "patients", "groups", "cross_label_patients", "tile_rows"))
            for n in result["expected"].values():
                _integer(n)
            _sequence(result["prior_wsi_ids"], _text)
        elif expected == "template":
            _observer_result(result, "png")
        elif expected == "config":
            _fields(result, ("encoded_size", "encoded_sha256", "raw_utf8", "parsed_json", "comparison_state"))
            _integer(result["encoded_size"], 0, BLOCK)
            _sha(result["encoded_sha256"])
            _need(result["comparison_state"] == "not_performed", "Config comparison flag")
            if result["raw_utf8"] is not None:
                _need(type(result["raw_utf8"]) is str and len(result["raw_utf8"].encode("utf-8")) <= BLOCK,
                      "Raw config strict UTF8 domain")
            if result["parsed_json"] is not None:
                _need(type(result["parsed_json"]) is dict, "Parsed config object")
                pending = [(result["parsed_json"], 1)]
                while pending:
                    obj, depth = pending.pop()
                    if type(obj) in (dict, list):
                        _need(depth <= 16, "Config depth16")
                        pending.extend((v, depth + 1) for v in (obj.values() if type(obj) is dict else obj))
                        if type(obj) is dict:
                            for key in obj:
                                _need(len(key.encode("utf-8")) <= 4096, "Config key")
                    elif type(obj) is str:
                        _need(len(obj.encode("utf-8")) <= BLOCK, "Config string domain")
        elif expected == "checkpoint_header":
            _fields(result, ("safetensors_observation", "schema_comparison"))
            _nullable(result["safetensors_observation"], lambda x: _observer_result(x, "safetensors"))
            _nullable(result["schema_comparison"], lambda x: _observer_result(x, "comparison"))
        elif expected == "final_checks":
            _fields(result, ("checked_roles", "region_only_roles", "failed_roles"))
            for roles in result.values():
                _sequence(roles, _text)
        elif expected == "serialization":
            _need(result == {"encoding": "utf-8", "json_format": "canonical_compact"}, "Serialization result")
    for key in ("inventory_initial", "inventory_final"):
        _inventory(packet[key])
    comparison = _fields(packet["inventory_comparison"], ("state", "compared_ids", "changed_ids", "changes", "uncomparable_ids"))
    _need(comparison["state"] in ("not_performed", "complete", "partial", "omitted_output_limit"), "Comparison state")
    if comparison["state"] == "omitted_output_limit":
        _need(all(comparison[k] is None for k in comparison if k != "state"), "Comparison omission nulls")
    else:
        for key in ("compared_ids", "changed_ids"):
            _sequence(comparison[key], _text)
        _nullable(comparison["uncomparable_ids"], lambda x: _sequence(x, _text))
        _need(type(comparison["changes"]) is list, "Comparison changes representation")
        for change in comparison["changes"]:
            _fields(change, ("slide_id", "changed_fields", "initial_status", "final_status", "affected_custody"))
            for key in ("slide_id", "initial_status", "final_status", "affected_custody"):
                _text(change[key])
            _sequence(change["changed_fields"], _text)
    _sequence(packet["issues"], _issue)
    _sequence(packet["secondary_errors"], lambda x: _issue(x, True))
    _need(len(packet["issues"]) <= 128 and len(packet["secondary_errors"]) <= 32, "Issue caps")
    overflow = _fields(packet["overflow"], ("occurred", "cause_code", "omitted_sections", "omitted_counts",
                       "serialization_bytes_seen", "global_issue_attempts", "secondary_error_attempts"))
    _bool(overflow["occurred"])
    _nullable(overflow["cause_code"], _text)
    _sequence(overflow["omitted_sections"], _text)
    _fields(overflow["omitted_counts"], ("inventory_initial_rows", "inventory_final_rows", "issues", "secondary_errors"))
    for n in overflow["omitted_counts"].values():
        _integer(n)
    _nullable(overflow["serialization_bytes_seen"], _integer)
    _integer(overflow["global_issue_attempts"])
    _integer(overflow["secondary_error_attempts"])
    logical = _fields(packet["logical_io"], ("reads", "selected", "controls"))
    read_keys = ("role", "phase", "requested_bytes", "returned_bytes", "eof_probe_requests", "eof_probe_returned_bytes")
    for read in logical["reads"]:
        _fields(read, read_keys)
        _text(read["role"])
        _need(read["phase"] in ("initial", "final"), "Read phase")
        for key in read_keys[2:]:
            _integer(read[key])
    _fields(logical["selected"], read_keys[2:] + ("limit_bytes",))
    for n in logical["selected"].values():
        _integer(n)
    _fields(logical["controls"], ("declared_bytes", "limit_bytes"))
    _nullable(logical["controls"]["declared_bytes"], _integer)
    _integer(logical["controls"]["limit_bytes"])
    _packet_relationships(packet)
    summary.update(state="valid", envelope_valid=True, status_match=status == packet["exit_code"])
    return packet, summary


def _receipt(ctx: _Context, declaration: dict, sources: dict, acquisitions: dict, directory: dict) -> dict:
    return dict(schema_version=1, task="hoptimus_input_observation_capture", receipt_kind="candidate",
                capture_id=ctx.entry["capture_id"], declaration_binding={k: ctx.entry["declaration_binding"][k]
                for k in ("size", "sha256")}, classification=ctx.result["classification"],
                intended_exit_code=ctx.result["intended_exit_code"], model_use_authorized=False,
                deadlines=declaration["deadlines"], directory=directory, sources=sources,
                source_acquisitions=acquisitions, transmission=ctx.transmission,
                child=ctx.result["child"], streams=ctx.result["streams"], packet=ctx.result["packet"],
                artifacts={n: ctx.result["artifacts"][n] for n in MEMBERS[:-1]}, errors=ctx.errors,
                closure_requirement="terminal_ticket_and_external_witness", limitations=LIMITATIONS)


def run(declaration_bytes: bytes, *, trusted_entry: TrustedEntry) -> CaptureResult:
    entry = _trusted(trusted_entry, declaration_bytes)
    ctx = _Context(entry)
    ctx.transmission = dict(frame_size=0, frame_sha256=EMPTY_SHA, metadata_size=0, caller_size=0,
                            acknowledged_bytes=0, acknowledged_sha256=EMPTY_SHA, unacknowledged_attempt_bytes=0,
                            state="not_started", stdin_close_state="not_opened", child_consumption_state="not_observed")
    declaration = None
    expected_files = {}
    caller_bytes = b""
    caller_acquisition = source_record("caller", None)
    directory = dict(declared_path=None, resolved_path=None, opened=None, before_receipt=None, path_check="not_performed")
    sources = {}
    try:
        ctx.check()
        declaration = _declaration(declaration_bytes, entry)
        _destination(ctx, declaration["output_dir"])
        directory.update(declared_path=declaration["output_dir"],
                         resolved_path=str(ctx.call(Path(declaration["output_dir"]).resolve, strict=True)),
                         opened=_fingerprint(ctx.call(os.fstat, ctx.directory_fd)), path_check="matched")
        expected_files = {"capture_declaration.json": declaration_bytes, "wrapper_source.py": entry["wrapper_bytes"],
                          "root_launch.py": entry["root_literal_bytes"], "child_bootstrap.py": CHILD_BOOTSTRAP_SOURCE}
        for name, data in expected_files.items():
            _write(ctx, name, data)
        def report(exc, record):
            record["error_indices"].append(ctx.fail(exc, "caller_acquisition", "acquire"))
        try:
            caller_bytes, caller_acquisition = acquire_source("caller", declaration["caller"], ctx.deadline, report)
        except BaseException as exc:
            caller_acquisition = getattr(exc, "acquisition", caller_acquisition)
            caller_bytes = getattr(exc, "captured_prefix", b"")
            expected_files["caller_source.py"] = caller_bytes
            if not ctx.interrupted:
                _write(ctx, "caller_source.py", caller_bytes)
            raise
        expected_files["caller_source.py"] = caller_bytes
        _write(ctx, "caller_source.py", caller_bytes)
        caller_entry = dict(declaration["caller"],
                            resolved_path=caller_acquisition["descriptor"]["resolved_before"])
        _path(caller_entry["resolved_path"])
        metadata = dict(frame_version=1, capture_id=entry["capture_id"], entry_binding=caller_entry,
                        argv=["--spec", declaration["spec"]["path"], "--spec-sha256", declaration["spec"]["sha256"]])
        encoded_metadata = canonical(metadata)
        _need(len(encoded_metadata) <= 262144, "Metadata ceiling")
        frame = len(encoded_metadata).to_bytes(8, "big") + encoded_metadata + caller_bytes
        ctx.transmission.update(frame_size=len(frame), frame_sha256=hashlib.sha256(frame).hexdigest(),
                                metadata_size=len(encoded_metadata), caller_size=len(caller_bytes),
                                stdin_close_state="pending")
        ctx.check()
        output = _pump(ctx, frame)
        ctx.check()
        try:
            packet, summary = _validate_packet(output, ctx.result["child"]["returncode"], caller_entry, declaration["spec"])
            ctx.result["packet"] = summary
            if not summary["status_match"]:
                ctx.fail(ValueError("Actual child status disagrees"), "packet_validation", "status",
                         code="status_disagreement")
            if ctx.failed_at is None:
                status = packet["exit_code"]
                ctx.result.update(classification={0: "observation_complete", 2: "observation_negative",
                                                 1: "diagnostic_packet"}[status], intended_exit_code=status)
        except BaseException as exc:
            if output:
                ctx.result["packet"]["state"] = "invalid"
            ctx.result["packet"]["error_indices"].append(ctx.fail(exc, "packet_validation", "packet"))
    except BaseException as exc:
        if not getattr(exc, "acquisition", None):
            ctx.fail(exc, "declaration" if declaration is None else "destination", "capture")
    finally:
        for name in list(ctx.fds):
            if name in ("child_stdout.bin", "child_stderr.bin"):
                stream = ctx.result["streams"]["stdout" if name == "child_stdout.bin" else "stderr"]
                ctx.result["artifacts"][name]["byte_scope"] = stream["receive_scope"]
                _finalize_artifact(ctx, name, None, stream)
            else:
                _finalize_artifact(ctx, name, expected_files.get(name, b""))
        if declaration is not None and ctx.directory_fd is not None and not ctx.interrupted:
            try:
                sources = {
                    role: dict(artifact=name, origin=origin, size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                               scope="captured_whole" if whole else ("captured_prefix" if data else "not_captured"),
                               execution_attested=attested)
                    for role, name, origin, data, whole, attested in (
                        ("wrapper", "wrapper_source.py", "root_captured_wrapper", entry["wrapper_bytes"], True, True),
                        ("root_launch", "root_launch.py", "executed_root_literal_argument", entry["root_literal_bytes"], True, True),
                        ("caller", "caller_source.py", "wrapper_captured_caller", caller_bytes,
                         caller_acquisition["state"] == "finalized", ctx.result["packet"]["state"] == "valid"),
                        ("child_bootstrap", "child_bootstrap.py", "authenticated_wrapper_child_literal", CHILD_BOOTSTRAP_SOURCE,
                         True, ctx.result["packet"]["state"] == "valid"),
                    )
                }
                directory["before_receipt"] = _fingerprint(ctx.call(os.fstat, ctx.directory_fd))
                acquisitions = {"wrapper": entry["wrapper_acquisition"],
                                "capture_declaration": entry["declaration_acquisition"], "caller": caller_acquisition}
                receipt_bytes = canonical(_receipt(ctx, declaration, sources, acquisitions, directory), newline=True)
                _need(len(receipt_bytes) <= SOURCE_CAP, "Candidate receipt ceiling")
                ctx.check()
                fd = os.open(MEMBERS[-1], os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=ctx.directory_fd)
                ctx.fds[MEMBERS[-1]] = fd
                ctx.result["artifacts"][MEMBERS[-1]].update(state="created", close_state="pending")
                ctx.check()
                ctx.result["artifacts"][MEMBERS[-1]]["opened"] = _fingerprint(ctx.call(os.fstat, fd))
                _write(ctx, MEMBERS[-1], receipt_bytes)
                _finalize_artifact(ctx, MEMBERS[-1], receipt_bytes)
                receipt_record = ctx.result["artifacts"][MEMBERS[-1]]
                if receipt_record["state"] == "finalized" and receipt_record["close_state"] == "succeeded":
                    ctx.result["receipt_binding"] = dict(size=len(receipt_bytes), sha256=hashlib.sha256(receipt_bytes).hexdigest())
                _need(set(ctx.call(os.listdir, ctx.directory_fd)) == set(MEMBERS), "Final fixed membership")
                for name, record in ctx.result["artifacts"].items():
                    _need(record["state"] == "finalized" and record["fsync_state"] == "succeeded"
                          and record["readback_state"] == "verified"
                          and record["before_close"] is not None and record["close_state"] == "succeeded"
                          and _fingerprint(ctx.call(os.stat, name, dir_fd=ctx.directory_fd,
                                                   follow_symlinks=False)) == record["before_close"], "Final member custody")
                _directory_guard(ctx)
                ctx.call(os.fsync, ctx.directory_fd)
                ctx.call(os.fsync, ctx.parent_fd)
                _directory_guard(ctx)
                ctx.result["storage_state"] = "candidate_closed"
            except BaseException as exc:
                ctx.fail(exc, "directory_finalization", "finalize")
                ctx.result["storage_state"] = "failed"
        for name in list(ctx.fds):
            ctx.close(ctx.fds.pop(name), "artifact_finalization", "close", ctx.result["artifacts"][name])
        closed = True
        for attr in ("directory_fd", "parent_fd"):
            fd = getattr(ctx, attr)
            if fd is not None:
                setattr(ctx, attr, None)
                closed = ctx.close(fd, "directory_finalization", attr) and closed
        ctx.result["owned_closures_succeeded"] = (
            closed and not ctx.closure_uncertain and not ctx.fds and not ctx.pipe_objects
            and all(r["close_state"] in ("succeeded", "not_acquired") for r in ctx.result["artifacts"].values())
            and caller_acquisition["close_state"] in ("succeeded", "not_acquired")
            and (ctx.process is None or ctx.result["child"]["reaped"]))
    if (ctx.result["storage_state"] == "candidate_closed" and ctx.result["owned_closures_succeeded"]
            and not ctx.deadline_failed):
        if ctx.failed_at is None or ctx.result["classification"] == "capture_failure":
            ticket = dict(schema_version=1, task="hoptimus_input_observation_capture_ticket",
                          capture_id=entry["capture_id"],
                          receipt=dict(name=MEMBERS[-1], **ctx.result["receipt_binding"]),
                          directory_identity=ctx.directory_identity,
                          classification=ctx.result["classification"], intended_exit_code=ctx.result["intended_exit_code"],
                          owned_closures_succeeded=True, model_use_authorized=False)
            data = canonical(ticket, newline=True)
            if len(data) <= 8192 and time.monotonic_ns() < ctx.deadline:
                ctx.result["ticket_bytes"] = data
    return ctx.result


def _diagnostic(message: str) -> None:
    data = (message[:1021] + "..." if len(message) > 1024 else message).encode("utf-8", "backslashreplace") + b"\n"
    position = 0
    try:
        while position < len(data):
            n = sys.stderr.buffer.write(data[position:position + BLOCK])
            if type(n) is not int or n <= 0 or n > len(data) - position:
                return
            position += n
        sys.stderr.buffer.flush()
    except BaseException:
        return


def main(declaration_bytes: bytes, *, trusted_entry: TrustedEntry,
         release_registration: Callable[[], None]) -> int:
    try:
        result = run(declaration_bytes, trusted_entry=trusted_entry)
        deadline = trusted_entry["started_monotonic_ns"] + trusted_entry["deadlines"]["overall_ns"]
        _need(time.monotonic_ns() < deadline, "Completion deadline before registration release")
        release_registration()
        _need(time.monotonic_ns() < deadline, "Completion deadline after registration release")
        if result["ticket_bytes"] is None:
            if result["errors"]["primary"] is not None:
                _diagnostic("Capture failure: " + result["errors"]["primary"]["message"])
            return 1
        encoded = result["ticket_bytes"]
        position = 0
        while position < len(encoded):
            _need(time.monotonic_ns() < deadline, "Completion deadline before ticket write")
            n = sys.stdout.buffer.write(encoded[position:position + BLOCK])
            _need(type(n) is int and 0 < n <= len(encoded) - position, "Ticket write count")
            position += n
            _need(time.monotonic_ns() < deadline, "Completion deadline after ticket write")
        _need(time.monotonic_ns() < deadline, "Completion deadline before ticket flush")
        sys.stdout.buffer.flush()
        _need(time.monotonic_ns() < deadline, "Completion deadline after ticket flush")
        if result["errors"]["primary"] is not None:
            _diagnostic("Capture failure: " + result["errors"]["primary"]["message"])
        return result["intended_exit_code"]
    except BaseException as exc:
        _diagnostic("Capture terminal failure: " + str(exc))
        return 1


if __name__ == "__main__":
    _diagnostic("Trusted captured-byte root entry required; ordinary execution refused")
    raise SystemExit(1)
