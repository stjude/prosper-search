"""Source-bound synthetic capture evidence; positive schema/WSI/receipt seams are UNIT only."""

import binascii
import copy
import csv
import errno
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import stat
import struct
import subprocess
import sys
import time
import types

import pytest


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / "prosper/evaluation/r2_1_hoptimus_input_observation_capture.py"
CALLER = SOURCE.with_name("r2_1_hoptimus_input_observation_cli.py")
OBSERVER = SOURCE.with_name("r2_1_hoptimus_input_admission.py")
SOURCE_BYTES = SOURCE.read_bytes()
CALLER_BYTES = CALLER.read_bytes()
OBSERVER_BYTES = OBSERVER.read_bytes()
INTERPRETER = "/HOME/miniconda3/bin/python3"
NAME = "r21_input_observation_capture_verified_v1"
WSI_ROOT = "/PROSPER_WORKSPACE/data/comet_final_HPC"
assert hashlib.sha256(CALLER_BYTES).hexdigest() == "168f6578c82502b8c9b4602723ebeae33e2f138f06f90371b54f3a1670ebc7d4"
assert hashlib.sha256(OBSERVER_BYTES).hexdigest() == "768e91414e43a86854c6e0088bdf91b2f10f1c1a3fc0a157c355b1bfa464071b"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canon(value, newline=False):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode() + (b"\n" if newline else b"")


def pin(data):
    return dict(size=len(data), sha256=sha(data))


def load():
    module = types.ModuleType(NAME)
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    sys.modules[NAME] = module
    exec(compile(SOURCE_BYTES, str(SOURCE), "exec"), module.__dict__)
    return module


@pytest.fixture
def capture():
    module = load()
    yield module
    if sys.modules.get(NAME) is module:
        del sys.modules[NAME]


@pytest.fixture
def owned(tmp_path):
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    return root


def csvbytes(header, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode()


def synthetic_controls(root, config=b'{"finite":1e-6}'):
    """Adapted generated data; no production receipt, WSI or model payload is acquired."""
    caller = types.ModuleType("synthetic_source_facts")
    exec(compile(CALLER_BYTES, "<frozen-caller-source-facts>", "exec"), caller.__dict__)
    inputs = root / "inputs"
    inputs.mkdir()
    caller_path = inputs / "caller.py"
    caller_path.write_bytes(CALLER_BYTES)
    observer_path = inputs / "observer.py"
    observer_path.write_bytes(OBSERVER_BYTES)
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", b"not-decoded") + chunk(b"IEND", b"")
    header = canon({"x": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]}})
    assets = {}
    for role, data in (("template", png), ("model_config", config),
                       ("checkpoint", struct.pack("<Q", len(header)) + header + b"abc"),
                       ("selected_wsi", b"not-an-svs")):
        path = inputs / role
        path.write_bytes(data)
        assets[role] = dict(path=str(path), **pin(data))
    wsi_path = assets["selected_wsi"]["path"]
    assets["selected_wsi"].update(path=WSI_ROOT + "/10003.svs", slide_id="10003")
    ids = ["10003", "16683"] + [str(20000 + i) for i in range(2776)]
    labels = ["NORMAL"] + ["G" + str(i) for i in range(1, 15)]
    metadata, queries = [], []
    for i, sid in enumerate(ids):
        patient = i if i < 1876 else i - 1876
        group = i % 15 if i < 1876 else (patient + (1 if patient < 45 else 0)) % 15
        metadata.append([labels[group], sid, "S" + sid, "P" + str(patient)])
        queries.append([str(i), str(i), sid, "P" + str(patient), labels[group],
                        str(3835507 - 2777 if i == 0 else 1), "/referenced/" + sid, "/referenced/" + sid,
                        "1", "0" * 64])
    metadata.extend([["MISC", str(40000 + i), "M", "MP" + str(i)] for i in range(143)])
    controls = {}
    def write(role, data):
        path = inputs / (role + ".control")
        path.write_bytes(data)
        controls[role] = dict(path=str(path), **pin(data))
    def record(role, **values):
        result = dict.fromkeys(caller.JSON_KEYS[role].split())
        result.update(schema_version=1, **values)
        return result
    write("metadata", csvbytes(caller.CSV_HEADERS["metadata"], metadata))
    write("targets", csvbytes(caller.CSV_HEADERS["targets"], queries))
    write("cpu_spec", canon(record("cpu_spec", run_id="synthetic_cpu", task="hoptimus_perturb_manifest",
          sources={"metadata": controls["metadata"], "template": assets["template"]},
          expected={"raw_slides": 2921, "target_slides": 2778, "patients": 1876, "groups": 15,
                    "cross_label_patients": 45, "feature_dim": 1536, "stimulus_records": 33336},
          code_files={}, parameters={"metadata": {"id_col": "Slide.Scan.File", "label_col": "Tumor.group",
                                                "patient_col": "UID.Subject"}})))
    source_bindings = []
    for kind, name, descriptor in (("spec", "spec", controls["cpu_spec"]),
                                   ("source", "metadata", controls["metadata"]),
                                   ("source", "template", assets["template"])):
        source_bindings.append(dict(descriptor, kind=kind, name=name,
                                    resolved_path=descriptor["path"], device=1, inode=1))
    write("cpu_input_bindings", canon(record("cpu_input_bindings", run_id="synthetic_cpu",
          task="hoptimus_perturb_manifest", spec_sha256=controls["cpu_spec"]["sha256"], bindings=source_bindings)))
    outputs = {name: dict(size=1, sha256="0" * 64) for name in caller.CPU_OUTPUTS}
    for name, role in (("queries.csv", "targets"), ("references.csv", "targets"),
                       ("input_bindings.json", "cpu_input_bindings")):
        outputs[name] = {k: controls[role][k] for k in ("size", "sha256")}
    write("cpu_complete", canon(record("cpu_complete", outputs=outputs, run_id="synthetic_cpu",
          task="hoptimus_perturb_manifest", scope="hoptimus_perturb_stimulus_manifests_only",
          spec_sha256=controls["cpu_spec"]["sha256"])))
    write("cpu_review", canon(record("cpu_review", disposition="accepted",
          branches={"perturb": {"disposition": "accepted", "scope": "Population and stimulus-manifest preparation only"}},
          artifacts={"perturb_attempt_001/COMPLETE.json": controls["cpu_complete"]["sha256"]})))
    rows = []
    for sid in ids[:10] + ["40000", "40001"]:
        descriptor = assets["selected_wsi"] if sid == "10003" else dict(size=1, sha256="0" * 64)
        rows.append(["wsi/" + sid, "wsi", "", "", sid, WSI_ROOT + "/" + sid + ".svs",
                     "/prior/" + sid, "true", "true", str(descriptor["size"]), str(descriptor["size"]),
                     "", descriptor["sha256"], "synthetic_observed"])
    for role, kind in (("model_config", "model_config"), ("checkpoint", "weight")):
        descriptor = assets[role]
        rows.append([kind + "/h_optimus_1", kind, "h_optimus_1", "", "", "/prior/" + role,
                     "/prior/resolved/" + role, "true", "true", str(descriptor["size"]),
                     str(descriptor["size"]), "", descriptor["sha256"], "synthetic_observed"])
    rows.extend([["unconsumed/" + str(i), "feature", "", "", "", "/never-read", "/never-read",
                  "true", "true", "1", "1", "", "0" * 64, "synthetic"] for i in range(213)])
    write("stage1b_source_inventory", csvbytes(caller.CSV_HEADERS["stage1b_source_inventory"], rows))
    inventory = controls["stage1b_source_inventory"]
    panel = {name: dict(size_bytes=1, sha256="0" * 64) for name in caller.PANEL_OUTPUTS}
    panel["source_inventory.csv"] = dict(size_bytes=inventory["size"], sha256=inventory["sha256"])
    write("stage1b_complete", canon(record("stage1b_complete", disposition="prepared",
          scope="encoder_replay_panel_only", outputs=panel, input_inventory_sha256=inventory["sha256"])))
    write("stage1b_review", canon(record("stage1b_review", disposition="accepted",
          scope="Stage1b sampled panel/reference and current input-byte preparation only",
          artifacts={"stage1b_attempt_001/COMPLETE.json": controls["stage1b_complete"]["sha256"],
                     "stage1b_attempt_001/source_inventory.csv": inventory["sha256"]})))
    write("observer_review", canon(record("observer_review",
          disposition="accepted_repaired_candidate002_two_files_only",
          stage="encoded_input_observation_primitives_implementation",
          bindings={"contract_sha256": caller.OBSERVER_CONTRACT_SHA, "module": pin(OBSERVER_BYTES)})))
    spec = dict(schema_version=1, run_id="synthetic_caller", task="hoptimus_encoded_input_observation",
                observer=dict(path=str(observer_path), **pin(OBSERVER_BYTES)), controls=controls, assets=assets,
                lexical_root=WSI_ROOT, expected=caller.EXPECTED,
                selection={"selected_wsi_ids": ["10003"], "deferred_wsi_ids": ["16683"]}, limits=caller.LIMITS)
    spec_path = inputs / "spec.json"
    spec_path.write_bytes(canon(spec))
    return dict(caller_path=caller_path, spec_path=spec_path, spec=spec, ids=ids, wsi_path=wsi_path,
                receipt_sha=controls["observer_review"]["sha256"])


def child_unit_seams(fixture, positive):
    """Inserted only into a bound synthetic wrapper copy; production bootstrap has no seams."""
    return """
    import builtins,io
    prescribed=owned.WSI_ROOT
    for operation in ("open","lstat","stat"):
        original=getattr(os,operation)
        def guarded(path,*args,_original=original,**kwargs):
            if isinstance(path,(str,os.PathLike)):
                text=str(path)
                assert not text.startswith(prescribed),"REAL WSI IO FORBIDDEN"
                assert not any(x in text for x in ("r2_1_hoptimus_main","r2_1_hoptimus_filter","mosaic_reader")),"FOREIGN STREAM IO"
            return _original(path,*args,**kwargs)
        setattr(os,operation,guarded)
    original_import=builtins.__import__
    def import_guard(name,*args,**kwargs):
        assert name.split(".")[0] not in owned.FORBIDDEN,"FORBIDDEN IMPORT"
        return original_import(name,*args,**kwargs)
    builtins.__import__=import_guard
    owned.OBSERVER_REVIEW_SHA=%r
    def probe(path):
        fp=[1,1,32768,1,1,1]
        return (dict(lexical_path=path,lexical_type="regular",resolved_path="/synthetic/resolved.svs",
          resolved_type="regular",size_bytes=1,probe_coherent=True,status="observed_regular",
          error_operation=None,error_errno=None,error_message=None),
          dict(lexical_before=fp[:],target_before=fp[:],lexical_after=fp[:],target_after=fp[:],
          resolved_before="/synthetic/resolved.svs",resolved_after="/synthetic/resolved.svs",errors=[],coherent=True),0)
    owned._probe_path=probe
    original_bookend=owned._bookend
    def bookend(path,which):
        return original_bookend(%r if path==prescribed+"/10003.svs" else path,which)
    owned._bookend=bookend
    if %r:
        original_basis=owned._load_basis
        def basis(context,spec):
            original_basis(context,spec)
            context.observer.compare_vit224_schema=lambda observation:dict(match=True,missing_names=[],
              unexpected_names=[],shape_mismatches=[],dtype_issues=[])
        owned._load_basis=basis
""" % (fixture["receipt_sha"], fixture["wsi_path"], positive)


def write_launch(capture, root, *, fixture=None, positive=False, amendment="", malformed=False,
                 durations=None):
    wrapper_bytes = SOURCE_BYTES + amendment.encode("ascii")
    child_bytes = capture.CHILD_BOOTSTRAP_SOURCE
    if fixture is not None:
        seam = child_unit_seams(fixture, positive)
        child_bytes = child_bytes.replace(b"    status=owned.main(argv,entry_binding=entry)",
                                         seam.encode("ascii") + b"    status=owned.main(argv,entry_binding=entry)")
        wrapper_bytes += b"\nCHILD_BOOTSTRAP_SOURCE = " + repr(child_bytes).encode("ascii") + b"\n"
    wrapper = root / "wrapper.py"
    wrapper.write_bytes(wrapper_bytes)
    caller = fixture["caller_path"] if fixture else root / "caller.py"
    if fixture is None:
        caller.write_bytes(CALLER_BYTES)
    spec = fixture["spec_path"] if fixture else root / "spec.json"
    if fixture is None:
        spec.write_bytes(b'{"incomplete":true}')
    deadlines = durations or dict(overall_ns=30_000_000_000, termination_grace_ns=200_000_000, drain_ns=1_000_000_000)
    declaration = dict(schema_version=1, task="hoptimus_input_observation_capture", capture_id="synthetic_capture",
                       output_dir=str(root / "bundle"), interpreter=INTERPRETER,
                       caller=dict(path=str(caller), **pin(CALLER_BYTES)),
                       wrapper=dict(path=str(wrapper), **pin(wrapper_bytes)),
                       root_launch=pin(capture.ROOT_LAUNCH_SOURCE), child_bootstrap=pin(child_bytes),
                       spec=dict(path=str(spec), sha256=sha(spec.read_bytes())), deadlines=deadlines)
    raw = b'{"invalid":true}' if malformed else canon(declaration, True)
    path = root / "capture.json"
    path.write_bytes(raw)
    args = [INTERPRETER, "-I", "-S", "-B", "-c", capture.ROOT_LAUNCH_SOURCE.decode("ascii"),
            "--capture-id", declaration["capture_id"], "--wrapper", str(wrapper),
            "--wrapper-size", str(len(wrapper_bytes)), "--wrapper-sha256", sha(wrapper_bytes),
            "--declaration", str(path), "--declaration-size", str(len(raw)), "--declaration-sha256", sha(raw),
            "--overall-ns", str(deadlines["overall_ns"]),
            "--termination-grace-ns", str(deadlines["termination_grace_ns"]),
            "--drain-ns", str(deadlines["drain_ns"])]
    (root / "root_literal.py").write_bytes(capture.ROOT_LAUNCH_SOURCE)
    (root / "child_literal.py").write_bytes(child_bytes)
    (root / "launch.json").write_bytes(canon({"argv": args, "declaration": declaration,
                                            "unit_seams": fixture is not None}, True))
    return args, declaration


def launch(root, args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    if (root / "hostile").exists():
        env["PYTHONPATH"] = str(root / "hostile")
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=root, env=env)
    try:
        stdout, stderr = process.communicate(timeout=40)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
        (root / "external_timeout").write_text("owned root killed by fixture witness\n")
    (root / "root.stdout").write_bytes(stdout)
    (root / "root.stderr").write_bytes(stderr)
    (root / "root.status.json").write_bytes(canon({"returncode": process.returncode,
                                                 "pid": process.pid,
                                                 "reaped": process.poll() is not None}, True))
    return process.returncode, stdout, stderr


def witness(root, declaration, status, stdout, stderr, hook=None):
    """Fixture-only external consumer; does not call production capture validation."""
    if status not in (0, 2) or stderr or len(stdout) > 8192 or not stdout.endswith(b"\n"):
        return False
    directory_fd = None
    try:
        ticket = json.loads(stdout)
        assert stdout == canon(ticket, True)
        assert set(ticket) == {"schema_version", "task", "capture_id", "receipt", "directory_identity",
                               "classification", "intended_exit_code", "owned_closures_succeeded", "model_use_authorized"}
        assert ticket["capture_id"] == declaration["capture_id"] and ticket["intended_exit_code"] == status
        assert ticket["classification"] == {0: "observation_complete", 2: "observation_negative"}[status]
        assert ticket["owned_closures_succeeded"] is True and ticket["model_use_authorized"] is False
        path = declaration["output_dir"]
        directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        identity = lambda s: [s.st_dev, s.st_ino, s.st_mode]
        assert identity(os.fstat(directory_fd)) == ticket["directory_identity"]
        if hook:
            hook(path)
        names = {"capture_declaration.json", "caller_source.py", "wrapper_source.py", "root_launch.py",
                 "child_bootstrap.py", "child_stdout.bin", "child_stderr.bin", "capture_receipt.json"}
        assert set(os.listdir(directory_fd)) == names
        blobs, fingerprints = {}, {}
        caps = {name: 1048576 for name in names}
        caps.update({"root_launch.py": 65536, "child_bootstrap.py": 65536,
                     "child_stdout.bin": 16777216, "child_stderr.bin": 65536})
        for name in sorted(names):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            try:
                before = os.fstat(fd)
                assert stat.S_ISREG(before.st_mode) and before.st_size <= caps[name]
                chunks, seen = [], 0
                while seen < before.st_size:
                    block = os.read(fd, min(65536, before.st_size - seen))
                    assert block
                    chunks.append(block)
                    seen += len(block)
                assert not os.read(fd, 1)
                after = os.fstat(fd)
                stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                assert stamp(before) == stamp(after) == stamp(os.stat(name, dir_fd=directory_fd, follow_symlinks=False))
                blobs[name], fingerprints[name] = b"".join(chunks), stamp(after)
            finally:
                os.close(fd)
        receipt = json.loads(blobs["capture_receipt.json"])
        assert blobs["capture_receipt.json"] == canon(receipt, True)
        assert pin(blobs["capture_receipt.json"]) == {k: ticket["receipt"][k] for k in ("size", "sha256")}
        assert receipt["receipt_kind"] == "candidate" and receipt["classification"] == ticket["classification"]
        assert receipt["intended_exit_code"] == status and receipt["errors"]["primary"] is None
        assert receipt["child"]["reaped"] and receipt["child"]["returncode"] == status
        assert receipt["packet"]["state"] == "valid" and receipt["packet"]["status_match"]
        assert pin(blobs["capture_declaration.json"]) == receipt["declaration_binding"]
        assert json.loads(blobs["capture_declaration.json"]) == declaration
        for role, record in receipt["sources"].items():
            assert pin(blobs[record["artifact"]]) == {k: record[k] for k in ("size", "sha256")}
            assert record["scope"] == "captured_whole" and record["execution_attested"] is True
        for role in ("stdout", "stderr"):
            stream = receipt["streams"][role]
            data = blobs["child_" + role + ".bin"]
            assert stream["eof_observed"] and stream["receive_scope"] == "complete_stream"
            assert stream["received_bytes"] == stream["acknowledged_prefix_bytes"] == len(data)
            assert stream["received_sha256"] == stream["acknowledged_prefix_sha256"] == sha(data)
            assert stream["unknown_retention_bytes"] == stream["discarded_bytes"] == 0
            assert stream["readback_reference_match"] is True and stream["readback_eof_observed"]
        packet = json.loads(blobs["child_stdout.bin"])
        assert packet["model_use_authorized"] is False and packet["exit_code"] == status
        for name, record in receipt["artifacts"].items():
            assert record["state"] == "finalized" and record["close_state"] == "succeeded"
            assert record["close_attempts"] == 1 and record["fsync_state"] == "succeeded"
            assert (record["size"], record["sha256"]) == (len(blobs[name]), sha(blobs[name]))
            assert tuple(record["before_close"]) == fingerprints[name]
        assert set(os.listdir(directory_fd)) == names
        assert identity(os.fstat(directory_fd)) == ticket["directory_identity"] == identity(os.lstat(path))
        for name, stamp_value in fingerprints.items():
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            assert (current.st_dev, current.st_ino, current.st_mode, current.st_size,
                    current.st_mtime_ns, current.st_ctime_ns) == stamp_value
        owned_fd, directory_fd = directory_fd, None
        os.close(owned_fd)
        return True
    except (AssertionError, OSError, ValueError, KeyError):
        return False
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def test_pristine_root_wrapper_caller_main_partial(capture, owned):
    args, declaration = write_launch(capture, owned)
    status, stdout, stderr = launch(owned, args)
    assert status == 1, stderr
    ticket = json.loads(stdout)
    assert ticket["classification"] == "diagnostic_packet"
    receipt = json.loads((owned / "bundle/capture_receipt.json").read_bytes())
    assert receipt["packet"]["state"] == "valid"
    assert receipt["sources"]["caller"]["execution_attested"]
    assert not witness(owned, declaration, status, stdout, stderr)
    assert not list(owned.rglob("__pycache__"))


@pytest.mark.parametrize("positive", [False, True])
def test_actual_chain_generated_observer_unit_seams(capture, owned, positive):
    fixture = synthetic_controls(owned)
    args, declaration = write_launch(capture, owned, fixture=fixture, positive=positive)
    status, stdout, stderr = launch(owned, args)
    assert status == (0 if positive else 2), stderr.decode(errors="replace")
    assert witness(owned, declaration, status, stdout, stderr)
    packet = json.loads((owned / "bundle/child_stdout.bin").read_bytes())
    assert len(packet["inventory_initial"]["frozen_result"]["rows"]) == 2778
    assert packet["stages"][4]["result"]["safetensors_observation"]["tensor_count"] == 1
    assert not (owned / "bundle/COMPLETE.json").exists()


def test_deep_legal_config_actual_main(capture, owned):
    config = {"signed": -int("9" * 127), "maxfloat": 1.7976931348623157e308, "zero": -0.0}
    nested = 1
    for _ in range(15):
        nested = [nested]
    config["deep"] = nested
    fixture = synthetic_controls(owned, canon(config))
    args, declaration = write_launch(capture, owned, fixture=fixture, positive=False)
    status, stdout, stderr = launch(owned, args)
    assert status == 2, stderr
    assert witness(owned, declaration, status, stdout, stderr)
    packet = json.loads((owned / "bundle/child_stdout.bin").read_bytes())
    assert packet["stages"][3]["result"]["parsed_json"] == config


def test_original_owned_source_mutation(capture, owned):
    amendment = """
_fixture_acquire=acquire_source
def acquire_source(role,declaration,deadline,report):
    data,record=_fixture_acquire(role,declaration,deadline,report)
    if role=="caller":
        Path(declaration["path"]).write_bytes(b"raise RuntimeError('PATH EXECUTION FORBIDDEN')\\n"+b"#"*(len(data)-47))
    return data,record
"""
    args, declaration = write_launch(capture, owned, amendment=amendment)
    status, stdout, stderr = launch(owned, args)
    assert status == 1
    packet = json.loads((owned / "bundle/child_stdout.bin").read_bytes())
    assert any(issue["code"] in ("digest_mismatch", "declared_size_mismatch") for issue in packet["issues"])
    assert (owned / "bundle/caller_source.py").read_bytes() == CALLER_BYTES
    assert not witness(owned, declaration, status, stdout, stderr)


@pytest.mark.parametrize("kind", ["malformed", "bad_wrapper_pin", "flags", "missing", "expired"])
def test_early_root_declaration_routes(capture, owned, kind):
    durations = dict(overall_ns=1, termination_grace_ns=1, drain_ns=1) if kind == "expired" else None
    args, declaration = write_launch(capture, owned, malformed=kind == "malformed", durations=durations)
    if kind == "bad_wrapper_pin":
        args[13] = "0" * 64
    elif kind == "flags":
        args[1] = "-B"
    elif kind == "missing":
        (owned / "wrapper.py").unlink()
    status, stdout, stderr = launch(owned, args)
    assert status == 1 and stdout == b"" and stderr
    assert not (owned / "bundle").exists()


@pytest.mark.parametrize("field", ["schema_version", "capture_id", "wrapper_bytes", "root_literal_bytes",
                                  "declaration_acquisition", "extra"])
def test_invalid_direct_entry_no_io(capture, monkeypatch, field):
    entry = {} if field != "extra" else {"extra": 1}
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid entry performed IO")
    for name in ("open", "lstat", "stat"):
        monkeypatch.setattr(os, name, forbidden)
    with pytest.raises(capture.EntryValidationError):
        capture.run(b"{}\n", trusted_entry=entry)


@pytest.mark.parametrize("n", [1, 1000000000, 2**63 - 1])
def test_integer_deadlines_bounded_poll(capture, n):
    assert 0 <= capture._poll_interval(1, n + 1) <= 1.0
    capture._deadlines(dict(overall_ns=n, termination_grace_ns=n, drain_ns=n))


@pytest.mark.parametrize("n", [0, -1, True, 1.0, 2**63])
def test_deadline_invalid_domains(capture, n):
    with pytest.raises(ValueError):
        capture._deadlines(dict(overall_ns=n, termination_grace_ns=1, drain_ns=1))


def unit_context(capture, root):
    """UNIT context, not evidence of trusted production entry."""
    entry = dict(capture_id="unit", started_monotonic_ns=time.monotonic_ns(),
                 deadlines=dict(overall_ns=20_000_000_000, termination_grace_ns=50_000_000, drain_ns=500_000_000))
    ctx = capture._Context(entry)
    ctx.transmission = dict(frame_size=0, frame_sha256=sha(b""), metadata_size=0, caller_size=0,
                            acknowledged_bytes=0, acknowledged_sha256=sha(b""), unacknowledged_attempt_bytes=0,
                            state="not_started", stdin_close_state="pending", child_consumption_state="not_observed")
    capture._destination(ctx, str(root / "unit_bundle"))
    ctx.fixture_root = root
    return ctx


def close_unit(capture, ctx):
    status = dict(child=ctx.result["child"], streams=ctx.result["streams"],
                  transmission=ctx.transmission, errors=ctx.errors,
                  process_pid=ctx.process.pid if ctx.process is not None else None,
                  process_returncode=ctx.process.poll() if ctx.process is not None else None)
    (ctx.fixture_root / "unit_process_status.json").write_bytes(canon(status, True))
    for name in list(ctx.fds):
        capture._finalize_artifact(ctx, name, b"")
    for attr in ("directory_fd", "parent_fd"):
        fd = getattr(ctx, attr)
        if fd is not None:
            setattr(ctx, attr, None)
            ctx.close(fd, "directory_finalization", "fixture_close")


@pytest.mark.parametrize("stream,length", [("stdout", 16777216), ("stdout", 16777217),
                                          ("stderr", 65536), ("stderr", 65537)])
def test_actual_stream_ceiling_unit_child(capture, owned, monkeypatch, stream, length):
    ctx = unit_context(capture, owned)
    program = ("import os\nwhile os.read(0,65536): pass\n"
               "remaining=%d\nwhile remaining:\n n=os.write(%d,b'x'*min(65536,remaining)); remaining-=n\n"
               % (length, 1 if stream == "stdout" else 2)).encode()
    monkeypatch.setattr(capture, "CHILD_BOOTSTRAP_SOURCE", program)
    try:
        result = capture._pump(ctx, b"x" * 200000)
        record = ctx.result["streams"][stream]
        assert record["received_bytes"] == length
        assert record["received_sha256"] == sha(b"x" * length)
        assert record["acknowledged_prefix_bytes"] == min(length, record["retention_limit"])
        assert record["post_failure_received_bytes"] <= 65536
        assert record["received_bytes"] == record["acknowledged_prefix_bytes"] + record["discarded_bytes"] + record["unknown_retention_bytes"]
        assert (ctx.failed_at is not None) == (length > record["retention_limit"])
        assert len(result) == (min(length, 16777216) if stream == "stdout" else 0)
        assert ctx.result["child"]["reaped"]
    finally:
        close_unit(capture, ctx)


def test_three_pipe_concurrent_pressure_unit(capture, owned, monkeypatch):
    ctx = unit_context(capture, owned)
    program = b"import os\nos.write(2,b'e'*65536)\nos.write(1,b'o'*65536)\nwhile os.read(0,65536): pass\n"
    monkeypatch.setattr(capture, "CHILD_BOOTSTRAP_SOURCE", program)
    try:
        assert capture._pump(ctx, b"i" * 1048576) == b"o" * 65536
        assert ctx.transmission["state"] == "submitted" and ctx.result["child"]["returncode"] == 0
        assert ctx.result["streams"]["stderr"]["received_bytes"] == 65536
        assert ctx.failed_at is None
    finally:
        close_unit(capture, ctx)


@pytest.mark.parametrize("failure", ["eio", "eintr", "keyboard", "zero", "short"])
def test_destination_write_accounting_unit(capture, owned, monkeypatch, failure):
    ctx = unit_context(capture, owned)
    record = capture._stream("stdout")
    record["_ack_digest"] = hashlib.sha256()
    original = os.write
    fd = ctx.fds["child_stdout.bin"]
    calls = []
    def seam(target, data):
        if target == fd:
            calls.append(len(data))
            if failure == "short":
                return original(target, data[:2])
            if failure == "zero":
                return 0
            original(target, data[:1])
            if failure == "keyboard":
                raise KeyboardInterrupt("owned write interruption")
            raise OSError(errno.EINTR if failure == "eintr" else errno.EIO, "owned uncertain write")
        return original(target, data)
    monkeypatch.setattr(os, "write", seam)
    try:
        if failure == "short":
            capture._write(ctx, "child_stdout.bin", b"abcdefgh", record)
            assert record["acknowledged_prefix_bytes"] == 8
            assert record["acknowledged_prefix_sha256"] == sha(b"abcdefgh")
        else:
            with pytest.raises(BaseException):
                capture._write(ctx, "child_stdout.bin", b"abcdefgh", record)
            assert record["acknowledged_prefix_bytes"] == 0 and record["unknown_retention_bytes"] == 8
            assert record["write_state"] == "failed" and len(calls) == 1
    finally:
        close_unit(capture, ctx)


@pytest.mark.parametrize("operation", ["lstat", "resolve", "read", "fstat", "close"])
def test_source_interruption_one_owned_close(capture, owned, monkeypatch, operation):
    path = owned / "source"
    path.write_bytes(b"abc")
    original_close = os.close
    closes, opened = [], []
    original_open = os.open
    def tracked_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd
    def tracked_close(fd):
        closes.append(fd)
        original_close(fd)
        if operation == "close":
            raise OSError(errno.EIO, "closed then synthetic EIO")
    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "close", tracked_close)
    if operation == "resolve":
        monkeypatch.setattr(capture.Path, "resolve", lambda *a, **k: (_ for _ in ()).throw(InterruptedError(errno.EINTR, "resolve")))
    elif operation != "close":
        monkeypatch.setattr(os, operation, lambda *a, **k: (_ for _ in ()).throw(InterruptedError(errno.EINTR, operation)))
    errors = []
    with pytest.raises(BaseException):
        capture.acquire_source("caller", dict(path=str(path), **pin(b"abc")), time.monotonic_ns() + 10**10,
                               lambda exc, record: errors.append((type(exc).__name__, copy.deepcopy(record))))
    assert errors
    assert len(closes) == len(opened)
    assert len(set(closes)) == len(closes)


@pytest.mark.parametrize("kind", ["destination", "parent"])
def test_namespace_fsync_failure_actual_root(capture, owned, kind):
    amendment = """
_fixture_fsync=os.fsync
def fixture_fsync(fd):
    info=os.fstat(fd)
    if stat.S_ISDIR(info.st_mode):
        path=%r
        if info.st_ino==os.stat(path).st_ino:
            raise OSError(5,"UNIT namespace fsync failure")
    return _fixture_fsync(fd)
os.fsync=fixture_fsync
""" % str(owned / "bundle" if kind == "destination" else owned)
    args, declaration = write_launch(capture, owned, amendment=amendment)
    status, stdout, stderr = launch(owned, args)
    assert status == 1 and stdout == b"" and stderr
    assert (owned / "bundle/capture_receipt.json").exists()
    assert not witness(owned, declaration, status, stdout, stderr)


@pytest.mark.parametrize("kind", ["before_open", "after_open"])
def test_witness_substituted_directory(capture, owned, kind):
    fixture = synthetic_controls(owned)
    args, declaration = write_launch(capture, owned, fixture=fixture, positive=False)
    status, stdout, stderr = launch(owned, args)
    assert status == 2
    def substitute(path):
        current = Path(path)
        preserved = owned / "preserved_original"
        current.rename(preserved)
        current.mkdir()
        for file in preserved.iterdir():
            (current / file.name).write_bytes(file.read_bytes())
    if kind == "before_open":
        substitute(declaration["output_dir"])
        hook = None
    else:
        hook = substitute
    assert not witness(owned, declaration, status, stdout, stderr, hook)


@pytest.mark.parametrize("phase", ["release", "ticket_write", "ticket_flush"])
def test_terminal_exact_deadline_unit(capture, monkeypatch, phase):
    t = [1]
    entry = dict(started_monotonic_ns=0, deadlines={"overall_ns": 10})
    result = dict(ticket_bytes=b'{"candidate":true}\n', intended_exit_code=0, errors={"primary": None})
    monkeypatch.setattr(capture, "run", lambda *a, **k: result)
    monkeypatch.setattr(capture.time, "monotonic_ns", lambda: t[0])
    stdout = io.BytesIO()
    class Output:
        def write(self, data):
            n = stdout.write(data)
            if phase == "ticket_write":
                t[0] = 10
            return n
        def flush(self):
            if phase == "ticket_flush":
                t[0] = 10
    monkeypatch.setattr(capture.sys, "stdout", types.SimpleNamespace(buffer=Output()))
    monkeypatch.setattr(capture.sys, "stderr", types.SimpleNamespace(buffer=io.BytesIO()))
    def release():
        if phase == "release":
            t[0] = 10
    assert capture.main(b"", trusted_entry=entry, release_registration=release) == 1
    assert stdout.getvalue() == (b"" if phase == "release" else result["ticket_bytes"])


def test_exit1_flush_failure_ambiguity_unit(capture, monkeypatch):
    result = dict(ticket_bytes=b'{"intended_exit_code":1}\n', intended_exit_code=1, errors={"primary": None})
    monkeypatch.setattr(capture, "run", lambda *a, **k: result)
    emitted = io.BytesIO()
    class Stream:
        def write(self, data):
            return emitted.write(data)
        def flush(self):
            raise OSError(errno.EIO, "after full ticket")
    monkeypatch.setattr(capture.sys, "stdout", types.SimpleNamespace(buffer=Stream()))
    monkeypatch.setattr(capture.sys, "stderr", types.SimpleNamespace(buffer=io.BytesIO()))
    assert capture.main(b"", trusted_entry=dict(started_monotonic_ns=time.monotonic_ns(),
                         deadlines={"overall_ns": 10**10}), release_registration=lambda: None) == 1
    assert emitted.getvalue() == result["ticket_bytes"]


def test_root_stderr_veto_fixture(capture, owned):
    fixture = synthetic_controls(owned)
    args, declaration = write_launch(capture, owned, fixture=fixture, positive=True,
                                     amendment="\nos.write(2,b'UNIT root diagnostic\\n')\n")
    status, stdout, stderr = launch(owned, args)
    assert status == 0 and stderr
    assert not witness(owned, declaration, status, stdout, stderr)


def test_fixed_literal_source_exports(capture):
    assert type(capture.ROOT_LAUNCH_SOURCE) is bytes and len(capture.ROOT_LAUNCH_SOURCE) <= 65536
    assert type(capture.CHILD_BOOTSTRAP_SOURCE) is bytes and len(capture.CHILD_BOOTSTRAP_SOURCE) <= 65536
    capture.ROOT_LAUNCH_SOURCE.decode("ascii")
    capture.CHILD_BOOTSTRAP_SOURCE.decode("ascii")
    assert b"owned.main(argv,entry_binding=entry)" in capture.CHILD_BOOTSTRAP_SOURCE
    assert b"runpy" not in capture.ROOT_LAUNCH_SOURCE + capture.CHILD_BOOTSTRAP_SOURCE


@pytest.mark.parametrize("phase", ["receipt_fsync", "directory_fsync", "parent_fsync",
                                  "directory_close", "parent_close"])
def test_actual_late_exact_deadline(capture, owned, phase):
    amendment = """
_fixture_init=_Context.__init__
_fixture_current=[]
def fixture_init(self,entry):
    _fixture_init(self,entry); _fixture_current.append(self)
_Context.__init__=fixture_init
_fixture_fsync=os.fsync
_fixture_close=os.close
def fixture_expire():
    deadline=_fixture_current[0].deadline
    time.monotonic_ns=lambda:deadline
def fixture_fsync(fd):
    result=_fixture_fsync(fd)
    ctx=_fixture_current[0]
    receipt=ctx.result["artifacts"]["capture_receipt.json"]["opened"]
    match=(%r=="receipt_fsync" and receipt is not None and os.fstat(fd).st_ino==receipt[1] or
      %r=="directory_fsync" and fd==ctx.directory_fd or
      %r=="parent_fsync" and fd==ctx.parent_fd)
    if match: fixture_expire()
    return result
def fixture_close(fd):
    ctx=_fixture_current[0]
    info=os.fstat(fd)
    match=(stat.S_ISDIR(info.st_mode) and (
      %r=="directory_close" and info.st_ino==os.stat(%r).st_ino or
      %r=="parent_close" and info.st_ino==os.stat(%r).st_ino))
    result=_fixture_close(fd)
    if match: fixture_expire()
    return result
os.fsync=fixture_fsync
os.close=fixture_close
""" % (phase, phase, phase, phase, str(owned / "bundle"), phase, str(owned))
    args, declaration = write_launch(capture, owned, amendment=amendment)
    status, stdout, stderr = launch(owned, args)
    assert status == 1 and stdout == b"" and stderr
    assert not witness(owned, declaration, status, stdout, stderr)


@pytest.mark.parametrize("owner", ["directory", "parent", "artifact"])
def test_actual_failed_close_uncertainty(capture, owned, owner):
    amendment = """
_fixture_init=_Context.__init__
_fixture_current=[]
def fixture_init(self,entry):
    _fixture_init(self,entry); _fixture_current.append(self)
_Context.__init__=fixture_init
_fixture_close=os.close
_fixture_fired=[]
def fixture_close(fd):
    ctx=_fixture_current[0]
    info=os.fstat(fd)
    receipt=ctx.result["artifacts"]["capture_receipt.json"]["opened"]
    match=(%r=="directory" and stat.S_ISDIR(info.st_mode) and info.st_ino==os.stat(%r).st_ino or
      %r=="parent" and stat.S_ISDIR(info.st_mode) and info.st_ino==os.stat(%r).st_ino or
      %r=="artifact" and receipt is not None and info.st_ino==receipt[1])
    _fixture_close(fd)
    if match and not _fixture_fired:
        _fixture_fired.append(fd)
        raise OSError(5,"UNIT close after actual owned close")
os.close=fixture_close
""" % (owner, str(owned / "bundle"), owner, str(owned), owner)
    args, declaration = write_launch(capture, owned, amendment=amendment)
    status, stdout, stderr = launch(owned, args)
    assert status == 1 and stdout == b"" and stderr
    assert not witness(owned, declaration, status, stdout, stderr)


def test_registration_release_failure_actual_root(capture, owned):
    amendment = """
_fixture_main=main
def main(declaration_bytes,*,trusted_entry,release_registration):
    def fail_release():
        release_registration()
        raise OSError(5,"UNIT release failure")
    return _fixture_main(declaration_bytes,trusted_entry=trusted_entry,release_registration=fail_release)
"""
    args, declaration = write_launch(capture, owned, amendment=amendment)
    status, stdout, stderr = launch(owned, args)
    assert status == 1 and stdout == b"" and stderr
    assert not witness(owned, declaration, status, stdout, stderr)


@pytest.mark.parametrize("signal_kind", ["SIGTERM", "SIGKILL"])
def test_native_owned_child_signal_unit(capture, owned, monkeypatch, signal_kind):
    ctx = unit_context(capture, owned)
    program = ("import os,signal\nos.kill(os.getpid(),signal.%s)\n" % signal_kind).encode()
    monkeypatch.setattr(capture, "CHILD_BOOTSTRAP_SOURCE", program)
    try:
        capture._pump(ctx, b"x")
        assert ctx.result["child"]["returncode"] == -getattr(signal, signal_kind)
        assert ctx.result["child"]["reaped"]
        assert not ctx.result["child"]["signals"]
    finally:
        close_unit(capture, ctx)


def test_owned_child_timeout_escalation_unit(capture, owned, monkeypatch):
    ctx = unit_context(capture, owned)
    ctx.deadline = time.monotonic_ns() + 200_000_000
    monkeypatch.setattr(capture, "CHILD_BOOTSTRAP_SOURCE",
                        b"import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(10)\n")
    try:
        capture._pump(ctx, b"x")
        assert ctx.result["child"]["reaped"]
        assert ctx.result["child"]["returncode"] == -signal.SIGKILL
        assert [r["signal"] for r in ctx.result["child"]["signals"]] == ["SIGTERM", "SIGKILL"]
    finally:
        close_unit(capture, ctx)


def test_inherited_pipe_drain_not_extended_unit(capture, owned, monkeypatch):
    ctx = unit_context(capture, owned)
    ctx.entry["deadlines"]["drain_ns"] = 100_000_000
    original = subprocess.Popen
    holder = []
    observed_exit = []
    poll_bounds = []
    read_fd, write_fd = os.pipe()
    def factory(args, **kwargs):
        kwargs["stdout"] = write_fd
        process = original(args, **kwargs)
        process.stdout = io.FileIO(read_fd, "rb", closefd=True)
        actual_poll = process.poll
        def poll():
            result = actual_poll()
            if result is not None and not observed_exit:
                observed_exit.append(time.monotonic_ns())
            return result
        process.poll = poll
        holder.append(original([INTERPRETER, "-I", "-S", "-B", "-c", "import time;time.sleep(10)"],
                               pass_fds=(write_fd,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        os.close(write_fd)
        return process
    monkeypatch.setattr(capture.subprocess, "Popen", factory)
    monkeypatch.setattr(capture, "CHILD_BOOTSTRAP_SOURCE",
                        b"import os\nwhile os.read(0,65536):pass\nos.write(1,b'x')\n")
    original_interval = capture._poll_interval
    def interval(now, deadline):
        if observed_exit:
            poll_bounds.append((now, deadline))
        return original_interval(now, deadline)
    monkeypatch.setattr(capture, "_poll_interval", interval)
    try:
        assert capture._pump(ctx, b"x") == b"x"
        assert ctx.result["child"]["reaped"]
        assert ctx.result["streams"]["stdout"]["eof_observed"] is False
        assert ctx.result["streams"]["stdout"]["read_state"] == "abandoned"
        assert observed_exit and poll_bounds
        assert all(deadline <= observed_exit[0] + 100_000_000 for _, deadline in poll_bounds)
        assert ctx.failed_at >= observed_exit[0]
    finally:
        for process in holder:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            (owned / "fixture_holder_status.json").write_bytes(canon({
                "pid": process.pid, "returncode": process.returncode, "reaped": True}, True))
        close_unit(capture, ctx)


@pytest.mark.parametrize("where", ["destination", "parent"])
def test_late_namespace_substitution_actual_root(capture, owned, where):
    target = owned / "bundle" if where == "destination" else owned
    amendment = """
_fixture_fsync=os.fsync
_fixture_fired=[]
def fixture_fsync(fd):
    result=_fixture_fsync(fd)
    info=os.fstat(fd)
    if stat.S_ISDIR(info.st_mode) and not _fixture_fired and info.st_ino==os.stat(%r).st_ino:
        _fixture_fired.append(fd)
        original=Path(%r)
        original.rename(str(original)+".preserved")
        original.mkdir()
    return result
os.fsync=fixture_fsync
""" % (str(target), str(target))
    args, declaration = write_launch(capture, owned, amendment=amendment)
    status, stdout, stderr = launch(owned, args)
    assert status == 1 and stdout == b"" and stderr
    assert not witness(owned, declaration, status, stdout, stderr)


@pytest.mark.parametrize("kind", ["missing", "extra", "noncanonical", "malformed", "status"])
def test_packet_protocol_failures_unit(capture, owned, kind):
    args, _ = write_launch(capture, owned)
    status, _, _ = launch(owned, args)
    assert status == 1
    encoded = (owned / "bundle/child_stdout.bin").read_bytes()
    declaration = json.loads((owned / "capture.json").read_bytes())
    entry = dict(declaration["caller"], resolved_path=declaration["caller"]["path"])
    if kind == "missing":
        encoded = b""
    elif kind == "extra":
        encoded += b"{}\n"
    elif kind == "noncanonical":
        encoded = b" " + encoded
    elif kind == "malformed":
        packet = json.loads(encoded)
        packet["unexpected"] = True
        encoded = canon(packet, True)
    if kind == "status":
        packet, summary = capture._validate_packet(encoded, 0, entry, declaration["spec"])
        assert summary["state"] == "valid" and summary["status_match"] is False
    else:
        with pytest.raises(ValueError):
            capture._validate_packet(encoded, 1, entry, declaration["spec"])


def test_measured_surrogate_not_declaration_domain(capture, owned):
    args, _ = write_launch(capture, owned)
    status, _, _ = launch(owned, args)
    assert status == 1
    packet = json.loads((owned / "bundle/child_stdout.bin").read_bytes())
    packet["entry_binding"]["initial_descriptor"]["resolved_path"] = "/observed/\udc80"
    declaration = json.loads((owned / "capture.json").read_bytes())
    entry = dict(declaration["caller"], resolved_path=declaration["caller"]["path"])
    _, summary = capture._validate_packet(canon(packet, True), 1, entry, declaration["spec"])
    assert summary["state"] == "valid"
    with pytest.raises(ValueError):
        capture._path("/declared/\udc80")


@pytest.mark.parametrize("count", [1, 65536, 65537])
def test_root_stderr_nonempty_overflow_veto(capture, count, owned):
    fixture = synthetic_controls(owned)
    args, declaration = write_launch(capture, owned, fixture=fixture, positive=False,
                                     amendment="\nos.write(2,b'x'*%d)\n" % count)
    status, stdout, stderr = launch(owned, args)
    assert status == 2 and len(stderr) == count
    assert not witness(owned, declaration, status, stdout, stderr)


@pytest.mark.parametrize("kind", ["zero_metadata", "excess_metadata", "at_metadata_cap",
                                  "short_metadata", "short_caller", "excess_frame", "hash",
                                  "keys", "boolean_size", "dirty_path", "bad_order"])
def test_actual_fixed_child_frame_failure(capture, owned, kind):
    caller = owned / "caller.py"
    caller.write_bytes(CALLER_BYTES)
    spec = owned / "invalid_spec.json"
    spec.write_bytes(b"{}")
    metadata = dict(frame_version=1, capture_id="frame_unit",
                    entry_binding=dict(path=str(caller), resolved_path=str(caller), **pin(CALLER_BYTES)),
                    argv=["--spec", str(spec), "--spec-sha256", sha(b"{}")])
    source = CALLER_BYTES
    if kind == "keys":
        metadata["extra"] = True
    elif kind == "boolean_size":
        metadata["entry_binding"]["size"] = True
    elif kind == "dirty_path":
        metadata["entry_binding"]["path"] = "/synthetic/\udc80"
    elif kind == "hash":
        source = b"#" * len(source)
    elif kind == "bad_order":
        metadata["argv"] = ["--spec-sha256", sha(b"{}"), "--spec", str(spec)]
    raw = canon(metadata)
    frame = len(raw).to_bytes(8, "big") + raw + source
    if kind == "zero_metadata":
        frame = (0).to_bytes(8, "big")
    elif kind == "excess_metadata":
        frame = (262145).to_bytes(8, "big")
    elif kind == "at_metadata_cap":
        frame = (262144).to_bytes(8, "big") + b" " * 262144
    elif kind == "short_metadata":
        frame = (len(raw) + 1).to_bytes(8, "big") + raw
    elif kind == "short_caller":
        frame = frame[:-1]
    elif kind == "excess_frame":
        frame += b"x"
    process = subprocess.Popen([INTERPRETER, "-I", "-S", "-B", "-c",
                                capture.CHILD_BOOTSTRAP_SOURCE.decode("ascii")],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(frame, timeout=10)
    (owned / "frame.bin").write_bytes(frame)
    (owned / "child.stdout").write_bytes(stdout)
    (owned / "child.stderr").write_bytes(stderr)
    (owned / "child.status.json").write_bytes(canon({"returncode": process.returncode}, True))
    assert process.returncode == 1 and stdout == b"" and b"bootstrap failure" in stderr


@pytest.mark.parametrize("kind", ["float_count", "bool_count", "unknown_stage", "extra_descriptor",
                                  "invented_g0_population", "wrong_model_flag", "nonfinite",
                                  "config_depth", "dirty_config"])
def test_nested_packet_representation_failures(capture, owned, kind):
    args, _ = write_launch(capture, owned)
    status, _, _ = launch(owned, args)
    assert status == 1
    packet = json.loads((owned / "bundle/child_stdout.bin").read_bytes())
    declaration = json.loads((owned / "capture.json").read_bytes())
    entry = dict(declaration["caller"], resolved_path=declaration["caller"]["path"])
    if kind in ("float_count", "bool_count"):
        packet["entry_binding"]["whole_file_checks"][0]["bytes_read"] = 1.0 if kind == "float_count" else True
    elif kind == "unknown_stage":
        packet["stages"][0]["name"] = "new_stage"
    elif kind == "extra_descriptor":
        packet["entry_binding"]["initial_descriptor"]["new"] = True
    elif kind == "invented_g0_population":
        packet["inventory_initial"]["unvisited_ids"] = ["10003"]
    elif kind == "wrong_model_flag":
        packet["model_use_authorized"] = True
    elif kind == "nonfinite":
        encoded = canon(packet, True).replace(b'"exit_code":1', b'"exit_code":NaN', 1)
        with pytest.raises(ValueError):
            capture._validate_packet(encoded, 1, entry, declaration["spec"])
        return
    else:
        parsed = {"dirty": "\udc80"} if kind == "dirty_config" else {}
        if kind == "config_depth":
            nested = 1
            for _ in range(16):
                nested = [nested]
            parsed["deep"] = nested
        packet["stages"][3]["result"] = dict(encoded_size=2, encoded_sha256=sha(b"{}"), raw_utf8="{}",
                                           parsed_json=parsed, comparison_state="not_performed")
    with pytest.raises(ValueError):
        capture._validate_packet(canon(packet, True), 1, entry, declaration["spec"])


def test_actual_receipt_serialized_ceiling(capture, owned):
    args, declaration = write_launch(capture, owned,
                                     amendment="\nLIMITATIONS = ['UNIT receipt overflow:' + 'x'*1048576]\n")
    status, stdout, stderr = launch(owned, args)
    assert status == 1 and stdout == b"" and b"receipt ceiling" in stderr
    assert not (owned / "bundle/capture_receipt.json").exists()
    assert not witness(owned, declaration, status, stdout, stderr)


@pytest.mark.parametrize("kind", ["write", "flush"])
def test_partial_ticket_no_replacement_unit(capture, monkeypatch, kind):
    encoded = b'{"candidate":true}\n'
    monkeypatch.setattr(capture, "run", lambda *a, **k: dict(ticket_bytes=encoded, intended_exit_code=0,
                                                          errors={"primary": None}))
    emitted = bytearray()
    class Output:
        def write(self, data):
            if kind == "write" and emitted:
                raise OSError(errno.EIO, "partial ticket write")
            n = 3 if kind == "write" else len(data)
            emitted.extend(data[:n])
            return n
        def flush(self):
            if kind == "flush":
                raise InterruptedError(errno.EINTR, "full ticket flush")
    monkeypatch.setattr(capture.sys, "stdout", types.SimpleNamespace(buffer=Output()))
    monkeypatch.setattr(capture.sys, "stderr", types.SimpleNamespace(buffer=io.BytesIO()))
    entry = dict(started_monotonic_ns=time.monotonic_ns(), deadlines={"overall_ns": 10**10})
    assert capture.main(b"", trusted_entry=entry, release_registration=lambda: None) == 1
    assert bytes(emitted) == (encoded[:3] if kind == "write" else encoded)


def test_error_primary_secondary_overflow_unit(capture, owned):
    ctx = unit_context(capture, owned)
    try:
        for i in range(50):
            ctx.fail(OSError(errno.EIO, "x" * 2000 + str(i)), "pump", "unit")
        assert ctx.errors["primary"]["index"] == 0
        assert len(ctx.errors["secondary"]) == 32
        assert ctx.errors["secondary"][-1]["code"] == "secondary_error_limit"
        assert ctx.errors["overflow"] and ctx.errors["omitted_secondary"] == 18
        assert ctx.errors["primary"]["message_truncated"] and len(ctx.errors["primary"]["message"]) == 1024
        assert ctx.result["intended_exit_code"] == 1
    finally:
        close_unit(capture, ctx)


def test_hostile_cwd_pythonpath_bytecode_actual_root(capture, owned):
    hostile = owned / "hostile"
    hostile.mkdir()
    for name in ("prosper", "PIL", "numpy", "torch", "timm", "safetensors", "openslide", "pandas"):
        package = hostile / name
        package.mkdir()
        (package / "__init__.py").write_text("raise RuntimeError('FORBIDDEN PACKAGE')\n")
        (owned / (name + ".py")).write_text("raise RuntimeError('HOSTILE CWD')\n")
    (hostile / "sitecustomize.py").write_text("raise RuntimeError('HOSTILE SITE')\n")
    (owned / "r2_1_hoptimus_input_observation_capture.pyc").write_bytes(b"not-authenticated-bytecode")
    args, declaration = write_launch(capture, owned)
    status, stdout, stderr = launch(owned, args)
    assert status == 1 and json.loads(stdout)["classification"] == "diagnostic_packet"
    assert not stderr and not list(owned.rglob("__pycache__"))
    assert not witness(owned, declaration, status, stdout, stderr)


def test_foreign_registration_not_removed_actual_root(capture, owned):
    amendment = """
_fixture_run=run
def run(declaration_bytes,*,trusted_entry):
    result=_fixture_run(declaration_bytes,trusted_entry=trusted_entry)
    sys.modules[NAME]=__import__("types").ModuleType("foreign_sentinel")
    return result
"""
    args, declaration = write_launch(capture, owned, amendment=amendment)
    status, stdout, stderr = launch(owned, args)
    assert status == 1 and stdout == b"" and b"registration replaced" in stderr
    assert not witness(owned, declaration, status, stdout, stderr)


def test_witness_checked_close_failure_veto(capture, owned, monkeypatch):
    fixture = synthetic_controls(owned)
    args, declaration = write_launch(capture, owned, fixture=fixture, positive=False)
    status, stdout, stderr = launch(owned, args)
    assert status == 2
    original_close = os.close
    identity = os.stat(declaration["output_dir"]).st_ino
    fired = []
    def close(fd):
        info = os.fstat(fd)
        original_close(fd)
        if stat.S_ISDIR(info.st_mode) and info.st_ino == identity:
            fired.append(fd)
            raise OSError(errno.EIO, "UNIT witness close after actual close")
    monkeypatch.setattr(os, "close", close)
    assert not witness(owned, declaration, status, stdout, stderr)
    assert len(fired) == 1


REPAIR37_ROOT = Path(os.environ.get(
    "PERTURB37_EVIDENCE_ROOT",
    str(PACKAGE.parent / "prosper_revision/prosper_revision_output/"
        "r2_1_mosaic_stain_review_20261003/perturb37_capture_repair001")))
REVIEW129_ROOT = REPAIR37_ROOT.parent / "astra129_capture_implementation"


def repair37_packet(number=0):
    bindings = json.loads((REPAIR37_ROOT / "compatibility_bindings.json").read_bytes())
    record = bindings[number]
    data = (REPAIR37_ROOT / record["name"]).read_bytes()
    assert pin(data) == {k: record[k] for k in ("size", "sha256")}
    raw = gzip.decompress(data)
    assert pin(raw) == record["uncompressed"]
    return json.loads(raw)


def repair37_validate(capture, packet):
    binding = packet["entry_binding"]
    entry = {key: binding["declared_" + key] for key in ("path", "size", "sha256")}
    entry["resolved_path"] = entry["path"]
    spec = {"path": packet["control_bindings"][0]["declared_path"], "sha256": packet["spec_sha256"]}
    return capture._validate_packet(canon(packet, True), packet["exit_code"], entry, spec)


@pytest.mark.parametrize("number", range(13))
def test_repair37_retained_legitimate_packets(capture, number):
    _, summary = repair37_validate(capture, repair37_packet(number))
    assert summary["state"] == "valid" and summary["status_match"] is True


@pytest.mark.parametrize("name", [
    "counterfactual_G0_exit0.json", "counterfactual_G0_exit2.json",
    "G0_invented_config_result.packet.json.gz",
    "negative_observation_promoted_zero.packet.json.gz",
    "unstarted_config_with_observed_result.packet.json.gz",
    "png_width_zero.packet.json.gz", "png_width_above_u31.packet.json.gz",
    "png_gamma_above_u32.packet.json.gz",
])
def test_repair37_original_eight_counterexamples(capture, name):
    record = next(r for r in json.loads((REVIEW129_ROOT / "evidence_manifest002.json").read_bytes())["artifacts"]
                  if Path(r["path"]).name == name)
    data = (REVIEW129_ROOT / name).read_bytes()
    assert pin(data) == {k: record[k] for k in ("size", "sha256")}
    raw = gzip.decompress(data) if name.endswith(".gz") else data
    with pytest.raises(ValueError):
        repair37_validate(capture, json.loads(raw))


def repair37_serializer_omission():
    """Actual frozen serializer helper with the original labelled first-encode fault."""
    caller = types.ModuleType("repair37_frozen_serializer_only")
    exec(compile(CALLER_BYTES, "<repair37-frozen-serializer-only>", "exec"), caller.__dict__)
    context = caller._Context()
    context.packet = repair37_packet()
    context.exit_code = context.packet["exit_code"]
    original = caller._encode
    calls = []
    def first_fault(packet):
        calls.append(packet["packet_kind"])
        if len(calls) == 1:
            fault = caller._Fault("stdout_limit", "Astra129 unit-only first-encode ceiling", "output")
            fault.bytes_seen = 16777217
            raise fault
        return original(packet)
    caller._encode = first_fault
    try:
        result = caller._finish_packet(context)
    finally:
        caller._encode = original
    assert len(calls) == 2 and result["exit_code"] == 1
    assert result["stdout_bytes"] == (REVIEW129_ROOT / "frozen_serializer_fallback.stdout.bin").read_bytes()
    assert result["stderr_bytes"] == (REVIEW129_ROOT / "frozen_serializer_fallback.stderr.bin").read_bytes()
    return result


def test_repair37_genuine_frozen_serializer_omission(capture, owned):
    result = repair37_serializer_omission()
    (owned / "serializer.stdout.bin").write_bytes(result["stdout_bytes"])
    (owned / "serializer.stderr.bin").write_bytes(result["stderr_bytes"])
    (owned / "serializer_scope.json").write_bytes(canon({
        "fault_seam": "first frozen _encode call raises stdout_limit",
        "natural_16mib_overflow": False, "caller_main_execution": False,
        "actual_frozen_finish_packet_fallback_second_encode": True,
    }, True))
    packet, summary = repair37_validate(capture, result["packet"])
    assert summary["state"] == "valid"
    assert all(record["prior_attribution"] is not None for record in packet["asset_records"])


REPAIR39_ROOT = Path(os.environ.get("PERTURB39_EVIDENCE_ROOT", str(
    REPAIR37_ROOT.parent / "perturb39_capture_repair002")))
REVIEW130_ROOT = REPAIR37_ROOT.parent / "astra130_capture_repair"


def repair39_retained(name):
    manifest = json.loads((REVIEW130_ROOT / "evidence_manifest.json").read_bytes())
    record = next(item for item in manifest["artifacts"] if Path(item["path"]).name == name)
    data = Path(record["path"]).read_bytes()
    assert pin(data) == {key: record[key] for key in ("size", "sha256")}
    return gzip.decompress(data) if name.endswith(".gz") else data


@pytest.mark.parametrize("cut", ["first_prior", "all_priors_before_result", "after_basis_result"])
@pytest.mark.parametrize("omission", [False, True])
def test_repair39_six_actual_retained130(capture, cut, omission):
    name = "actual_reconcile_" + cut + ("_omitted" if omission else "") + ".stdout.bin"
    packet, summary = repair37_validate(capture, json.loads(repair39_retained(name)))
    assert summary["state"] == "valid" and packet["exit_code"] == 1


@pytest.mark.parametrize("name", ["complete_zero_with_error_overflow", "complete_zero_with_inventory_change"])
@pytest.mark.parametrize("status", [0, 2, 1])
def test_repair39_retained_failure_contradictions(capture, name, status):
    packet = json.loads(repair39_retained(name + ".packet.json.gz"))
    packet.update(exit_code=status, packet_kind={0: "complete", 2: "complete_negative", 1: "partial_error"}[status])
    if status == 1:
        packet["stages"][0]["state"] = "not_started"
    with pytest.raises(ValueError):
        repair37_validate(capture, packet)


REPAIR39_DRIVER = r'''
import binascii,builtins,copy,errno,hashlib,json,os,pathlib,sys,types
root=pathlib.Path(sys.argv[1])
owned_roots=(str(root),str(root.resolve()))
settings=json.loads((root/"settings.json").read_bytes())
caller_path=root/"inputs/caller.py"
encoded=caller_path.read_bytes()
assert hashlib.sha256(encoded).hexdigest()=="168f6578c82502b8c9b4602723ebeae33e2f138f06f90371b54f3a1670ebc7d4"
caller=types.ModuleType("r21_input_observation_cli_verified_v1")
caller.__file__=str(caller_path);caller.__package__=""
assert caller.__name__ not in sys.modules
sys.modules[caller.__name__]=caller
exec(compile(encoded,str(caller_path),"exec"),caller.__dict__)
trace_records=[];probe_calls=[];asset_calls=[]
original_import=builtins.__import__
def guarded_import(name,*args,**kwargs):
    assert name.split(".")[0] not in set(caller.FORBIDDEN)|{"pytest"},"native/provider import"
    return original_import(name,*args,**kwargs)
builtins.__import__=guarded_import
def guard_path(path,metadata=False):
    if isinstance(path,(str,bytes,os.PathLike)):
        text=os.fsdecode(path)
        assert not text.startswith(caller.WSI_ROOT),"real WSI-root access"
        assert not any(x in text for x in ("/comet-feature-gen/","site-packages/")),"real model/software access"
        assert (any(text.startswith(base+"/") or text==base for base in owned_roots)
          or metadata and any(base.startswith(text.rstrip("/")+"/") for base in owned_roots)),"outside owned generated inputs"
for operation in ("open","stat","lstat"):
    original=getattr(os,operation)
    def guarded(path,*args,_original=original,_metadata=operation in ("stat","lstat"),**kwargs):
        if not isinstance(path,int):guard_path(path,_metadata)
        return _original(path,*args,**kwargs)
    setattr(os,operation,guarded)
def audit(event,args):
    if event=="open" and not isinstance(args[0],int):guard_path(args[0])
sys.addaudithook(audit)
caller.OBSERVER_REVIEW_SHA=settings["receipt_sha"]
def forbidden_assets(*args,**kwargs):
    asset_calls.append(True)
    raise AssertionError("initial/current asset work forbidden")
caller._initial_assets=forbidden_assets
def probe(path):
    probe_calls.append(path)
    if settings["final"]=="inventory_error":
        raise RuntimeError("UNIT coherent ordinary final inventory probe error")
    if settings["final"]=="inventory_interrupt":
        raise KeyboardInterrupt("UNIT coherent final inventory interruption")
    fp=[1,1,32768,1,1,1]
    if settings["final"] in ("inventory_progress", "inventory_progress_bookend") and len(probe_calls)==2:
        exc=RuntimeError("UNIT ordinary probe error after one visited identity")
        if settings["final"]=="inventory_progress_bookend":
            exc.partial_bookends=dict(lexical_before=fp[:],target_before=fp[:],
              lexical_after=None,target_after=None,resolved_before="/unit/generated.svs",
              resolved_after=None,errors=[],coherent=False)
        raise exc
    return (dict(lexical_path=path,lexical_type="regular",resolved_path="/unit/generated.svs",
      resolved_type="regular",size_bytes=1,probe_coherent=True,status="observed_regular",
      error_operation=None,error_errno=None,error_message=None),
      dict(lexical_before=fp[:],target_before=fp[:],lexical_after=fp[:],target_after=fp[:],
      resolved_before="/unit/generated.svs",resolved_after="/unit/generated.svs",errors=[],coherent=True),0)
caller._probe_path=probe
original_small=caller._small
def small(context,record,ceiling,phase="initial"):
    if settings["final"]=="checks_return_interrupt" and phase=="final" and record["role"]=="metadata":
        raise KeyboardInterrupt("UNIT caught final-control interruption")
    if settings["final"]=="checks_return_error" and phase=="final" and record["role"]=="metadata":
        raise OSError(errno.EIO,"UNIT ordinary returning final-control error")
    return original_small(context,record,ceiling,phase)
caller._small=small
original_checks=caller._final_checks
def checks(context):
    original_checks(context)
    if settings["final"]=="inventory_before_output":
        sys.settrace(trace)
    if settings.get("cap_setup"):
        for number in range(settings["cap_setup"]):
            context.issue(RuntimeError("UNIT synthetic cap saturation"),stage="final_checks")
caller._final_checks=checks
fired=[]
def trace(frame,event,arg):
    if frame.f_code.co_filename!=str(caller_path):return trace
    context=frame.f_locals.get("context")
    if context is None:return trace
    name=frame.f_code.co_name;line=frame.f_lineno
    count=sum(record["prior_attribution"] is not None for record in context.packet["asset_records"])
    cut=settings["cut"]
    hit=(name=="_reconcile" and (
      event=="line" and (cut=="before_prior" and line==1007 or cut=="before_ids" and line==1016 or
        cut.startswith("prior") and line==1017 and count==int(cut[5:]) or
        cut=="initial" and line==1020 or cut=="final" and line==1021 or
        cut=="comparison" and line==1022 or cut=="result_build" and line==1023)
      or event=="return" and cut=="result_return" and context.packet["stages"][0]["result"] is not None))
    if hit and not fired:
        fired.append(True)
        trace_records.append(dict(function=name,line=line,event=event,prior_count=count,
            basis_result=context.packet["stages"][0]["result"] is not None,
            ids_published=context.ids is not None))
        if settings["fault"]=="keyboard":raise KeyboardInterrupt("UNIT actual publication trace")
        if settings["fault"]=="eintr":raise InterruptedError(errno.EINTR,"UNIT publication EINTR")
        raise RuntimeError("UNIT actual ordinary publication trace")
    if (fired and event=="line" and name=="_final_checks" and line==1253
            and settings["final"]=="checks_escape_interrupt"):
        raise KeyboardInterrupt("UNIT escaping final-check interruption")
    if (fired and event=="line" and name=="_inventory_pass" and line==1034
            and settings["final"]=="inventory_unknown"):
        raise RuntimeError("UNIT before first target publication")
    if (fired and event=="line" and name=="_inventory_pass" and line==1031
            and settings["final"]=="inventory_before_output"):
        trace_records.append(dict(function=name,line=line,event=event,
            seam="rearmed trace after the ordinary publication exception"))
        raise RuntimeError("UNIT entered final inventory before output publication")
    return trace
original_encode=caller._encode
encode_calls=[]
def encode(packet):
    encode_calls.append(packet["packet_kind"])
    if settings["omission"] and len(encode_calls)==1:
        exc=caller._Fault("stdout_limit","UNIT first-encode ceiling seam","output")
        exc.bytes_seen=16777217
        raise exc
    return original_encode(packet)
caller._encode=encode
spec=root/"inputs/spec.json"
sys.settrace(trace)
try:
    result=caller.run(str(spec),spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),
      entry_binding=dict(path=str(caller_path),resolved_path=str(caller_path.resolve()),
                        size=len(encoded),sha256=hashlib.sha256(encoded).hexdigest()))
finally:sys.settrace(None)
assert fired and not asset_calls
assert caller.OBSERVER_NAME not in sys.modules
(root/"actual_source_trace.json").write_text(json.dumps(dict(trace=trace_records,
    probe_calls=len(probe_calls),asset_calls=len(asset_calls),encode_calls=encode_calls,
    actual_run_execute_load_basis_reconcile=True,caller_main=False,
    cap_setup_is_unit_not_natural_single_run_history=bool(settings.get("cap_setup")),
    receipt_pin_seam=True,generated_path_probe_seam=True,natural_16mib_overflow=False)))
os.write(2,result["stderr_bytes"])
data=result["stdout_bytes"]
while data:
    count=os.write(1,data[:65536]);data=data[count:]
del sys.modules[caller.__name__]
raise SystemExit(result["exit_code"])
'''


def repair39_actual_run(capture, owned, *, cut, fault, omission, final="normal", cap_setup=0):
    fixture = synthetic_controls(owned)
    settings = dict(cut=cut, fault=fault, omission=omission, final=final,
                    cap_setup=cap_setup, receipt_sha=fixture["receipt_sha"])
    (owned / "settings.json").write_bytes(canon(settings))
    (owned / "driver_source.py").write_bytes(REPAIR39_DRIVER.encode("ascii"))
    argv = [INTERPRETER, "-I", "-S", "-B", "-c", REPAIR39_DRIVER, str(owned)]
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(timeout=40)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
        (owned / "external_timeout").write_text("fixture-owned caller driver killed/reaped\n")
    (owned / "driver.stdout").write_bytes(stdout)
    (owned / "driver.stderr").write_bytes(stderr)
    (owned / "driver_status.json").write_bytes(canon(dict(pid=process.pid, returncode=process.returncode,
        reaped=process.poll() is not None, argv=argv, settings=settings), True))
    assert process.returncode == 1 and stdout, stderr.decode("utf-8", "replace")
    packet, summary = repair37_validate(capture, json.loads(stdout))
    assert summary["state"] == "valid" and packet["packet_kind"] == "partial_error"
    return packet


@pytest.mark.parametrize("cut", [
    "before_prior", "before_ids", "prior0", "prior1", "prior2", "prior3", "prior4",
    "initial", "final", "comparison", "result_build", "result_return",
])
@pytest.mark.parametrize("fault", ["keyboard", "eintr", "ordinary"])
@pytest.mark.parametrize("omission", [False, True])
def test_repair39_actual_publication_cuts(capture, owned, cut, fault, omission):
    repair39_actual_run(capture, owned, cut=cut, fault=fault, omission=omission)


@pytest.mark.parametrize("final", [
    "checks_return_interrupt", "checks_escape_interrupt", "inventory_error",
    "inventory_interrupt", "inventory_unknown",
])
@pytest.mark.parametrize("omission", [False, True])
def test_repair39_actual_ordinary_finalization(capture, owned, final, omission):
    repair39_actual_run(capture, owned, cut="prior2", fault="ordinary", omission=omission, final=final)


@pytest.mark.parametrize("omission", [False, True])
@pytest.mark.parametrize("cap_setup", [126, 127])
def test_repair39_actual_source_capped_partial_inventory(capture, owned, omission, cap_setup):
    packet = repair39_actual_run(capture, owned, cut="prior2", fault="ordinary", omission=omission,
                                final="inventory_error", cap_setup=cap_setup)
    assert packet["overflow"]["omitted_counts"]["issues"] >= 1
    assert packet["issues"][-1]["stage"] == ("final_inventory" if cap_setup == 126 else "final_checks")
    assert packet["stages"][8]["issue_indices"] == []
    assert packet["stages"][8]["state"] == ("omitted_output_limit" if omission else "partial")


@pytest.mark.parametrize("ng,ns,marker,valid", [
    (1, 33, "final_checks", False), (128, 33, "final_checks", False),
    (128, 0, "serialization", True), (129, 0, "serialization", False),
    (129, 0, "final_checks", True), (130, 0, "final_checks", True),
])
def test_repair39_nonfungible_serializer_channels(capture, owned, ng, ns, marker, valid):
    """Actual frozen issue/fallback helpers with UNIT cap setup, not source-run history."""
    caller = types.ModuleType("repair39_cap_helper_unit")
    exec(compile(CALLER_BYTES, "<frozen-cap-helper-unit>", "exec"), caller.__dict__)
    context = caller._Context()
    context.packet = repair37_packet()
    context.exit_code = 2
    for number in range(max(0, ng - 2)):
        context.issue(RuntimeError("UNIT cap setup"), stage=marker if number == 126 else "final_checks")
    for number in range(ns):
        context.issue(RuntimeError("UNIT secondary setup"), stage="final_checks", secondary=0)
    original_encode = caller._encode
    calls = []
    def encode(packet):
        calls.append(None)
        if len(calls) == 1:
            fault = caller._Fault("stdout_limit", "UNIT serializer-global attempt", "output")
            fault.bytes_seen = 16777217
            raise fault
        return original_encode(packet)
    caller._encode = encode
    result = caller._finish_packet(context)
    packet = result["packet"]
    if ng <= 127:
        # Counterfactual: replace the mandatory last global output with unrelated evidence.
        packet["issues"].pop()
        packet["overflow"]["global_issue_attempts"] -= 1
        packet["stages"][-1]["issue_indices"] = []
    if ng == 128 and marker != "serialization":
        packet["issues"][-1]["stage"] = marker
    (owned / "cap_helper_scope.json").write_bytes(canon(dict(
        actual_frozen_issue_finish_fallback=True, unit_counter_history=True,
        source_run_history_proved=False, ng=packet["overflow"]["global_issue_attempts"],
        ns=packet["overflow"]["secondary_error_attempts"], marker=marker), True))
    (owned / "cap_helper.stdout.bin").write_bytes(canon(packet, True))
    if valid:
        repair37_validate(capture, packet)
    else:
        with pytest.raises(ValueError):
            repair37_validate(capture, packet)


def repair39_mutant(kind):
    before = json.loads(repair39_retained("actual_reconcile_first_prior.stdout.bin"))
    all_priors = json.loads(repair39_retained("actual_reconcile_all_priors_before_result.stdout.bin"))
    after = json.loads(repair39_retained("actual_reconcile_after_basis_result.stdout.bin"))
    packet = before
    if kind == "gap":
        packet = all_priors
        packet["asset_records"][0]["prior_attribution"] = None
    elif kind == "reversed":
        packet = all_priors
        a, b = packet["asset_records"][:2]
        a["prior_attribution"], b["prior_attribution"] = b["prior_attribution"], a["prior_attribution"]
    elif kind == "wrong_size":
        packet["asset_records"][0]["prior_attribution"]["size"] += 1
    elif kind == "wrong_review":
        packet["asset_records"][0]["prior_attribution"]["review_sha256"] = "0" * 64
    elif kind == "early_initial":
        packet["inventory_initial"] = after["inventory_initial"]
    elif kind == "reversed_final":
        packet = all_priors
        packet["inventory_final"] = after["inventory_final"]
    elif kind == "reversed_comparison":
        packet = all_priors
        packet["inventory_comparison"] = after["inventory_comparison"]
    elif kind == "conflicting_universe":
        packet = after
        packet["inventory_final"]["unvisited_ids"] = list(reversed(packet["inventory_final"]["unvisited_ids"]))
    elif kind == "selected_io":
        packet["logical_io"]["selected"]["returned_bytes"] = 1
    elif kind == "invented_asset_result":
        packet["stages"][3].update(state="complete", result=repair37_packet()["stages"][3]["result"])
    elif kind == "incomplete_g1":
        packet["control_bindings"].pop()
    elif kind == "false_basis_omission":
        packet = json.loads(repair39_retained("actual_reconcile_first_prior_omitted.stdout.bin"))
        packet["overflow"]["omitted_sections"].insert(3, "stages.trusted_basis.result")
    elif kind == "false_overflow":
        packet = json.loads(repair39_retained("actual_reconcile_first_prior_omitted.stdout.bin"))
        packet["overflow"]["occurred"] = False
    elif kind == "hybrid_inventory":
        packet["inventory_final"].update(state="partial", frozen_result=repair37_packet()["inventory_final"]["frozen_result"])
    elif kind == "missing_global_count":
        packet["overflow"]["global_issue_attempts"] = 0
    elif kind == "unlinked_issue":
        packet["stages"][0]["issue_indices"] = []
    elif kind == "missing_final_roles":
        packet["stages"][7].update(state="complete", blocker_codes=[], result=dict(
            checked_roles=["entry"], region_only_roles=[], failed_roles=[]))
    return packet


@pytest.mark.parametrize("kind", [
    "gap", "reversed", "wrong_size", "wrong_review", "early_initial", "reversed_final",
    "reversed_comparison", "conflicting_universe", "selected_io", "invented_asset_result",
    "incomplete_g1", "false_basis_omission", "false_overflow", "hybrid_inventory",
    "missing_global_count", "unlinked_issue", "missing_final_roles",
])
def test_repair39_publication_counterfactuals(capture, owned, kind):
    packet = repair39_mutant(kind)
    (owned / "counterfactual.packet.json").write_bytes(canon(packet, True))
    with pytest.raises(ValueError):
        repair37_validate(capture, packet)


@pytest.mark.parametrize("kind", ["gap", "hybrid_inventory"])
def test_repair39_invalid1_raw_capture_fixture(capture, owned, monkeypatch, kind):
    """Fake-child transport seam, NOT caller-main or normal fixed-bootstrap execution."""
    raw = canon(repair39_mutant(kind), True)
    path = owned / "unit_invalid_packet.json"
    path.write_bytes(raw)
    program = ("import os\nwhile os.read(0,65536):pass\n"
               "data=open(%r,'rb').read()\nwhile data:\n n=os.write(1,data[:65536]);data=data[n:]\n"
               "raise SystemExit(1)\n" % str(path)).encode("ascii")
    monkeypatch.setattr(capture, "CHILD_BOOTSTRAP_SOURCE", program)
    args, declaration = write_launch(capture, owned, amendment="\nCHILD_BOOTSTRAP_SOURCE=" + repr(program) + "\n")
    status, stdout, stderr = launch(owned, args)
    assert status == 1 and not witness(owned, declaration, status, stdout, stderr)
    assert (owned / "bundle/child_stdout.bin").read_bytes() == raw
    receipt = json.loads((owned / "bundle/capture_receipt.json").read_bytes())
    assert receipt["classification"] == "capture_failure"
    assert receipt["packet"]["state"] == "invalid"
    (owned / "fake_child_scope.json").write_bytes(canon(dict(
        fake_child=True, caller_main=False, fixed_production_literal_execution=False,
        raw_bytes_retained=True, unmodified_real_literal_values=True), True))


REVIEW134_ROOT = REPAIR37_ROOT.parent / "astra134_capture_repair002"
REPAIR40_NEGATIVES = (
    "failed_basis_with_completed_asset_work",
    "null_final_checks_result_with_completed_final_inventory",
    "prefix_final_checks_with_completed_final_inventory",
    "stopped_finalization_with_positive_omitted_rows",
    "visited_row_without_bookend",
)


def repair40_review_controls():
    records, sources = [], {}
    for name, expected in (
        ("results003.json", {"size": 613426, "sha256": "c11235ad442f53844a16b7c544e4d1106243fd61ad27be2140c12fd242ed6c51"}),
        ("results004.json", {"size": 58991, "sha256": "e1ac1a291e7f657842c5f91d06f8de0994b03222d10330c1abc9b51294f884ba"}),
    ):
        raw = (REVIEW134_ROOT / name).read_bytes()
        assert pin(raw) == expected
        value = json.loads(raw)
        records.extend(value["replays"])
        sources.update((item["sha256"], item) for item in value["sources"].values())
    assert len(records) == 149
    assert sum(item["label"] not in REPAIR40_NEGATIVES for item in records) == 144
    return records, sources


REPAIR40_CONTROLS, REPAIR40_CONTROL_SOURCES = repair40_review_controls()


def repair40_control_packet(record):
    label = record["label"]
    if label.startswith("original13_"):
        packet = repair37_packet(int(label.removeprefix("original13_")))
    elif ".gz_status" in label:
        name, status = label.rsplit("_status", 1)
        packet = json.loads(repair39_retained(name))
        status = int(status)
        packet.update(exit_code=status, packet_kind={0: "complete", 1: "partial_error", 2: "complete_negative"}[status])
        if status == 1:
            packet["stages"][0]["state"] = "not_started"
    elif label.endswith(".packet.json.gz"):
        manifest = json.loads((REVIEW129_ROOT / "evidence_manifest002.json").read_bytes())
        item = next(item for item in manifest["artifacts"] if Path(item["path"]).name == label)
        raw = (REVIEW129_ROOT / label).read_bytes()
        assert pin(raw) == {key: item[key] for key in ("size", "sha256")}
        packet = json.loads(gzip.decompress(raw))
    else:
        if label in REPAIR40_NEGATIVES:
            raw = (REVIEW134_ROOT / (label + ".packet.json")).read_bytes()
        else:
            item = REPAIR40_CONTROL_SOURCES[record["sha256"]]
            raw = Path(item["path"]).read_bytes()
            assert pin(raw) == {key: item[key] for key in ("size", "sha256")}
        packet = json.loads(raw)
    assert pin(canon(packet, True)) == {key: record[key] for key in ("size", "sha256")}
    return packet


@pytest.mark.parametrize("record", REPAIR40_CONTROLS, ids=lambda record: record["label"])
def test_repair40_retained134_controls(capture, owned, record):
    """Authenticated retained parser inputs; no caller or process execution."""
    packet = repair40_control_packet(record)
    accepted, error = False, None
    try:
        _, summary = repair37_validate(capture, packet)
        accepted = summary["state"] == "valid"
    except ValueError as exc:
        error = str(exc)
    (owned / "retained_control_outcome.json").write_bytes(canon(dict(
        label=record["label"], expected=record["expected"], accepted=accepted,
        error=error, packet=pin(canon(packet, True)), owned_process=None,
        source="authenticated retained134 parser input, not fresh caller execution"), True))
    assert accepted is record["expected"]


@pytest.mark.parametrize("kind", REPAIR40_NEGATIVES)
def test_repair40_invalid1_raw_capture_fixture(capture, owned, monkeypatch, kind):
    """Fake-child transport, not production literals or caller.main execution."""
    record = next(item for item in REPAIR40_CONTROLS if item["label"] == kind)
    packet = repair40_control_packet(record)
    raw = (REVIEW134_ROOT / (kind + ".packet.json")).read_bytes()
    assert raw == canon(packet, True)
    path = owned / "unit_invalid_packet.json"
    path.write_bytes(raw)
    program = ("import os\nwhile os.read(0,65536):pass\n"
               "data=open(%r,'rb').read()\nwhile data:\n n=os.write(1,data[:65536]);data=data[n:]\n"
               "raise SystemExit(1)\n" % str(path)).encode("ascii")
    monkeypatch.setattr(capture, "CHILD_BOOTSTRAP_SOURCE", program)
    args, declaration = write_launch(capture, owned, amendment="\nCHILD_BOOTSTRAP_SOURCE=" + repr(program) + "\n")
    status, stdout, stderr = launch(owned, args)
    assert status == 1 and not witness(owned, declaration, status, stdout, stderr)
    assert (owned / "bundle/child_stdout.bin").read_bytes() == raw
    receipt = json.loads((owned / "bundle/capture_receipt.json").read_bytes())
    assert receipt["classification"] == "capture_failure" and receipt["packet"]["state"] == "invalid"
    (owned / "fake_child_scope.json").write_bytes(canon(dict(
        fake_child=True, caller_main=False, fixed_production_literal_execution=False,
        raw_bytes_retained=True, unmodified_real_literal_values=True, finding=kind), True))


@pytest.mark.parametrize("final", ["checks_return_error", "inventory_progress", "inventory_progress_bookend"])
@pytest.mark.parametrize("omission", [False, True])
def test_repair40_actual_returning_finalization(capture, owned, final, omission):
    packet = repair39_actual_run(capture, owned, cut="prior2", fault="ordinary", omission=omission, final=final)
    checks = packet["stages"][7]
    assert checks["state"] == ("failed" if final == "checks_return_error" else "complete")
    if omission:
        expected_rows = {"checks_return_error": 2778, "inventory_progress": 1, "inventory_progress_bookend": 2}
        assert packet["overflow"]["omitted_counts"]["inventory_final_rows"] == expected_rows[final]
        assert "stages.final_checks.result" in packet["overflow"]["omitted_sections"]
    else:
        assert checks["result"]["checked_roles"] == ["entry", "spec", "observer", *capture.CONTROLS]
        if final == "checks_return_error":
            assert checks["result"]["failed_roles"] == ["metadata"]
            assert packet["inventory_final"]["state"] == "complete"
        else:
            inventory = packet["inventory_final"]
            assert len(inventory["visited_ids"]) == 1 and inventory["in_progress_id"] is not None
            assert set(inventory["visited_ids"]).issubset(inventory["bookends"])
            assert (inventory["in_progress_id"] in inventory["bookends"]) is (final == "inventory_progress_bookend")


@pytest.mark.parametrize("omission", [False, True])
def test_repair40_actual_before_final_output(capture, owned, omission):
    packet = repair39_actual_run(capture, owned, cut="prior2", fault="ordinary",
                                omission=omission, final="inventory_before_output")
    assert packet["stages"][7]["state"] == "complete"
    trace = json.loads((owned / "actual_source_trace.json").read_bytes())["trace"]
    assert any(row["function"] == "_inventory_pass" and row["line"] == 1031 for row in trace)
    if omission:
        assert packet["overflow"]["omitted_counts"]["inventory_final_rows"] == 0
    else:
        assert packet["inventory_final"]["state"] == "not_started"
        assert packet["inventory_final"]["unvisited_ids"] is None
        assert packet["stages"][8]["state"] == "failed"
        assert packet["stages"][7]["result"]["checked_roles"] == ["entry", "spec", "observer", *capture.CONTROLS]


@pytest.mark.parametrize("result", [None, {"checked_roles": ["entry"], "region_only_roles": [], "failed_roles": []}])
def test_repair40_entered_final_phase_requires_returning_checks(capture, owned, result):
    """Coordinated CWC134-2 phase-entry negative, not a source-run assertion."""
    record = next(item for item in REPAIR40_CONTROLS
                  if item["label"] == "null_final_checks_result_with_completed_final_inventory")
    packet = repair40_control_packet(record)
    packet["stages"][7]["result"] = result
    packet["inventory_final"] = dict(state="not_started", visited_ids=[], in_progress_id=None,
        unvisited_ids=None, probes={}, bookends={}, frozen_result=None, omitted_fields=[])
    packet["inventory_comparison"] = dict(state="not_performed", compared_ids=[], changed_ids=[],
        changes=[], uncomparable_ids=None)
    issue = copy.deepcopy(packet["issues"][-1])
    issue.update(stage="final_inventory", message="UNIT coordinated before-output phase counterfactual")
    packet["stages"][8].update(state="failed", blocker_codes=[issue["code"]], issue_indices=[len(packet["issues"])])
    packet["issues"].append(issue)
    packet["overflow"]["global_issue_attempts"] += 1
    (owned / "entered_phase_counterfactual.packet.json").write_bytes(canon(packet, True))
    with pytest.raises(ValueError):
        repair37_validate(capture, packet)


REVIEW136_ROOT = REPAIR37_ROOT.parent / "astra136_capture_repair003"
REPAIR41_CASES = (
    ("correction002/case_blocked_final_inventory_omits_1.packet.json", 34129,
     "f9129c19dc74203490b2c17cf6e5d674e7a6439f1422dbe1140d2174e2284fd2"),
    ("correction002/case_blocked_final_inventory_omits_2778.packet.json", 34132,
     "915184df59c9ad5c43d1dcfda11ee89fb1d1da6311c22abbfa15e7e47954e73e"),
    ("correction003/case_escaped_empty_checks_switched_to_blocked_inventory.packet.json", 28869,
     "daa922c174ae0d160f6aace7ab4b0d41ccf291c615110ae7076649b383d1bc27"),
    ("correction003/case_escaped_null_checks_switched_to_blocked_inventory.packet.json", 28812,
     "9703203264c26cfc26edd5ce61b75e73935091778b9a6bc13e0197f86bd2fbfb"),
)


def repair41_case(number):
    name, size, digest = REPAIR41_CASES[number]
    raw = (REVIEW136_ROOT / name).read_bytes()
    assert pin(raw) == dict(size=size, sha256=digest)
    return raw


def repair41_owned_launch(root, argv, prefix):
    """Owned redirected-file streams, not direct pipe-EOF observations."""
    stdout_path, stderr_path = root / (prefix + ".stdout"), root / (prefix + ".stderr")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", NUMEXPR_MAX_THREADS="1",
               BLIS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    controls = {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD")}
    controls["PROSPER_OUT"] = env["PROSPER_OUT"]
    started = time.monotonic_ns()
    with stdout_path.open("xb") as stdout_stream, stderr_path.open("xb") as stderr_stream:
        fds = {"stdout": stdout_stream.fileno(), "stderr": stderr_stream.fileno()}
        (root / (prefix + ".launch.json")).write_bytes(canon(dict(argv=argv, cwd=str(root),
            environment_controls=controls, stream_paths=[str(stdout_path), str(stderr_path)],
            precreated_streams=True, parent_fds=fds, direct_pipe_eof_observed=False), True))
        process = subprocess.Popen(argv, cwd=root, env=env, stdout=stdout_stream, stderr=stderr_stream)
        (root / (prefix + ".pid.json")).write_bytes(canon(dict(pid=process.pid, started_ns=started), True))
        timed_out = False
        try:
            process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait(timeout=5)
        finally:
            (root / (prefix + ".status.json")).write_bytes(canon(dict(
                pid=process.pid, returncode=process.returncode, reaped=process.poll() is not None,
                timed_out=timed_out, elapsed_ns=time.monotonic_ns() - started), True))
    stdout, stderr = stdout_path.read_bytes(), stderr_path.read_bytes()
    (root / (prefix + ".fd_cleanup.json")).write_bytes(canon(dict(
        parent_streams_closed=stdout_stream.closed and stderr_stream.closed,
        stdout=pin(stdout), stderr=pin(stderr), direct_pipe_eof_observed=False), True))
    assert not timed_out
    return process.returncode, stdout, stderr


def repair41_driver():
    """Explicit source/trace/FD seams; the inherited driver and caller stay unchanged."""
    replacements = (
        ("trace_records=[];probe_calls=[];asset_calls=[]",
         "trace_records=[];probe_calls=[];asset_calls=[];fd_events=[];owned_fds={}"),
        ('def guarded(path,*args,_original=original,_metadata=operation in ("stat","lstat"),**kwargs):\n'
         '        if not isinstance(path,int):guard_path(path,_metadata)\n'
         '        return _original(path,*args,**kwargs)',
         'def guarded(path,*args,_original=original,_metadata=operation in ("stat","lstat"),_operation=operation,**kwargs):\n'
         '        if not isinstance(path,int):guard_path(path,_metadata)\n'
         '        result=_original(path,*args,**kwargs)\n'
         '        if _operation=="open":\n'
         '            owned_fds[result]=os.fsdecode(path)\n'
         '            fd_events.append(dict(operation="open",fd=result,path=os.fsdecode(path)))\n'
         '        return result'),
        ('def audit(event,args):',
         'original_close=os.close\n'
         'def tracked_close(fd):\n'
         '    try:original_close(fd)\n'
         '    except OSError as exc:\n'
         '        fd_events.append(dict(operation="close_failed",fd=fd,errno=exc.errno,owned=fd in owned_fds))\n'
         '        raise\n'
         '    else:\n'
         '        fd_events.append(dict(operation="close",fd=fd,owned=fd in owned_fds))\n'
         '        owned_fds.pop(fd,None)\n'
         'os.close=tracked_close\n'
         'def audit(event,args):'),
        ('def small(context,record,ceiling,phase="initial"):',
         'def small(context,record,ceiling,phase="initial"):\n'
         '    if settings["final"]=="checks_return41" and phase=="final" and record["role"]==settings["check_role"]:\n'
         '        trace_records.append(dict(function="_small",phase=phase,role=record["role"],\n'
         '            checked_roles=list(context.packet["stages"][7]["result"]["checked_roles"]),\n'
         '            seam="actual returning control-read interruption"))\n'
         '        if settings["check_fault"]=="eintr":raise InterruptedError(errno.EINTR,"UNIT returning EINTR")\n'
         '        raise KeyboardInterrupt("UNIT returning KeyboardInterrupt")'),
        ('def checks(context):\n    original_checks(context)',
         'def checks(context):\n'
         '    if settings["final"]=="checks_escape41":sys.settrace(trace)\n'
         '    original_checks(context)'),
        ('    if (fired and event=="line" and name=="_final_checks" and line==1253',
         '    if (fired and event=="line" and name=="_final_checks" and line==settings.get("escape_line")\n'
         '            and settings["final"]=="checks_escape41"):\n'
         '        trace_records.append(dict(function=name,line=line,event=event,\n'
         '            result_published=context.packet["stages"][7]["result"] is not None,\n'
         '            seam="explicitly rearmed escaping final checks"))\n'
         '        if settings["check_fault"]=="eintr":raise InterruptedError(errno.EINTR,"UNIT escaping EINTR")\n'
         '        raise KeyboardInterrupt("UNIT escaping KeyboardInterrupt")\n'
         '    if (fired and event=="line" and name=="_final_checks" and line==1253'),
        ('assert fired and not asset_calls',
         '(root/"source_return.stdout.bin").write_bytes(result["stdout_bytes"])\n'
         '(root/"source_return.stderr.bin").write_bytes(result["stderr_bytes"])\n'
         '(root/"source_return.trace.json").write_text(json.dumps(trace_records))\n'
         '(root/"source_fd_lifecycle.json").write_text(json.dumps(dict(events=fd_events,\n'
         '    outstanding_owned_fds=owned_fds,observer_removed=caller.OBSERVER_NAME not in sys.modules,\n'
         '    caller_registration_owned=sys.modules.get(caller.__name__) is caller)))\n'
         'assert not owned_fds\n'
         'assert fired and not asset_calls'),
        ('del sys.modules[caller.__name__]\nraise SystemExit(result["exit_code"])',
         'del sys.modules[caller.__name__]\n'
         '(root/"source_cleanup.json").write_text(json.dumps(dict(caller_removed=caller.__name__ not in sys.modules,\n'
         '    observer_removed=caller.OBSERVER_NAME not in sys.modules,outstanding_owned_fds=owned_fds)))\n'
         'raise SystemExit(result["exit_code"])'),
    )
    program = REPAIR39_DRIVER
    for before, after in replacements:
        assert program.count(before) == 1
        program = program.replace(before, after)
    compile(program, "<repair41-generated-source-driver>", "exec")
    return program


def repair41_actual_run(capture, owned, *, final, omission, cut="prior2", fault="ordinary",
                        escape_line=1253, check_fault="keyboard", check_role="metadata"):
    fixture = synthetic_controls(owned)
    settings = dict(cut=cut, fault=fault, omission=omission, final=final, cap_setup=0,
        receipt_sha=fixture["receipt_sha"], escape_line=escape_line, check_fault=check_fault, check_role=check_role)
    (owned / "settings.json").write_bytes(canon(settings))
    program = repair41_driver().encode("ascii")
    path = owned / "driver_source.py"
    path.write_bytes(program)
    (owned / "fresh_source_scope.json").write_bytes(canon(dict(
        actual_frozen_run_execute_load_basis_reconcile_final_checks=True, caller_main=False,
        receipt_path_probe_trace_fd_seams=True, native_or_real_input=False, driver=pin(program),
        fallback_is_first_encode_fault_not_natural_16mib_overflow=omission,
        returned_buffers_distinct_from_redirected_os_streams=True), True))
    status, stdout, stderr = repair41_owned_launch(owned, [INTERPRETER, "-I", "-S", "-B", str(path), str(owned)], "source")
    assert status == 1 and stdout, stderr.decode("utf-8", "replace")
    assert stdout == (owned / "source_return.stdout.bin").read_bytes()
    assert stderr == (owned / "source_return.stderr.bin").read_bytes()
    cleanup = json.loads((owned / "source_cleanup.json").read_bytes())
    assert cleanup == dict(caller_removed=True, observer_removed=True, outstanding_owned_fds={})
    packet, summary = repair37_validate(capture, json.loads(stdout))
    assert summary["state"] == "valid"
    return packet


@pytest.mark.parametrize("number", range(4))
def test_repair41_exact136_negatives(capture, owned, number):
    raw = repair41_case(number)
    (owned / "original136.packet.json").write_bytes(raw)
    with pytest.raises(ValueError):
        repair37_validate(capture, json.loads(raw))


def test_repair41_retained136_activated_escape(capture, owned):
    manifest_raw = (REVIEW136_ROOT / "evidence_manifest.json").read_bytes()
    assert pin(manifest_raw) == dict(size=13391,
        sha256="759e0b397a138e151debe537830f5e087c50979c8f96fe9f5fc72aa0760435fb")
    manifest = {row["relative_path"]: row for row in json.loads(manifest_raw)["artifacts"]}
    retained = {}
    for name in ("stdout.bin", "stderr.bin", "source_return.stdout.bin", "source_return.stderr.bin",
                 "actual_source_trace.json", "outcome.json"):
        item = manifest["correction003/source_escape/" + name]
        raw = (REVIEW136_ROOT / item["relative_path"]).read_bytes()
        assert pin(raw) == {key: item[key] for key in ("size", "sha256")}
        retained[name] = raw
    assert retained["stdout.bin"] == retained["source_return.stdout.bin"]
    assert retained["stderr.bin"] == retained["source_return.stderr.bin"]
    packet, summary = repair37_validate(capture, json.loads(retained["stdout.bin"]))
    assert summary["state"] == "valid"
    assert packet["stages"][7]["result"] == dict(checked_roles=[], region_only_roles=[], failed_roles=[])
    assert packet["stages"][8]["state"] == "not_started"
    (owned / "retained136_escape_scope.json").write_bytes(canon(dict(
        original_OS_streams=True, original_function_return_buffers=True,
        fresh_execution=False, raw_pipe_eof_observed=False, owned_process=None,
        original_trace=json.loads(retained["actual_source_trace.json"]),
        original_outcome=json.loads(retained["outcome.json"])), True))


@pytest.mark.parametrize("number", range(4))
def test_repair41_exact136_raw_capture_failure(capture, owned, monkeypatch, number):
    raw = repair41_case(number)
    path = owned / "unit_invalid_packet.json"
    path.write_bytes(raw)
    program = ("import os\nwhile os.read(0,65536):pass\n"
               "data=open(%r,'rb').read()\nwhile data:\n n=os.write(1,data[:65536]);data=data[n:]\n"
               "raise SystemExit(1)\n" % str(path)).encode("ascii")
    monkeypatch.setattr(capture, "CHILD_BOOTSTRAP_SOURCE", program)
    args, declaration = write_launch(capture, owned, amendment="\nCHILD_BOOTSTRAP_SOURCE=" + repr(program) + "\n")
    (owned / "fake_child_scope.json").write_bytes(canon(dict(fake_child=True, caller_main=False,
        fixed_production_literal_execution=False, original136=REPAIR41_CASES[number][0]), True))
    status, stdout, stderr = repair41_owned_launch(owned, args, "root")
    assert status == 1 and not witness(owned, declaration, status, stdout, stderr)
    assert (owned / "bundle/child_stdout.bin").read_bytes() == raw
    receipt = json.loads((owned / "bundle/capture_receipt.json").read_bytes())
    assert receipt["classification"] == "capture_failure" and receipt["packet"]["state"] == "invalid"
    assert receipt["child"]["reaped"] is True


@pytest.mark.parametrize("escape_line", [1248, 1253])
@pytest.mark.parametrize("check_fault", ["keyboard", "eintr"])
@pytest.mark.parametrize("omission", [False, True])
def test_repair41_activated_escape(capture, owned, escape_line, check_fault, omission):
    packet = repair41_actual_run(capture, owned, final="checks_escape41", omission=omission,
                                escape_line=escape_line, check_fault=check_fault)
    trace = json.loads((owned / "actual_source_trace.json").read_bytes())
    assert any(row.get("line") == escape_line and row.get("seam") == "explicitly rearmed escaping final checks"
               and row["result_published"] is (escape_line == 1253) for row in trace["trace"])
    assert trace["probe_calls"] == trace["asset_calls"] == 0
    assert trace["encode_calls"] == (["partial_error", "partial_error"] if omission else ["partial_error"])
    checks, inventory = packet["stages"][7:9]
    assert checks["state"] == "failed" and inventory["blocker_codes"] == inventory["issue_indices"] == []
    if omission:
        assert inventory["state"] == "omitted_output_limit"
        assert packet["overflow"]["omitted_counts"]["inventory_final_rows"] == 0
        assert ("stages.final_checks.result" in packet["overflow"]["omitted_sections"]) is (escape_line == 1253)
    else:
        assert inventory["state"] == packet["inventory_final"]["state"] == "not_started"
        assert checks["result"] == (dict(checked_roles=[], region_only_roles=[], failed_roles=[])
                                    if escape_line == 1253 else None)


@pytest.mark.parametrize("check_role", ["entry", "metadata", "stage1b_source_inventory"])
@pytest.mark.parametrize("check_fault", ["keyboard", "eintr"])
@pytest.mark.parametrize("omission", [False, True])
def test_repair41_returned_interruption_prefix(capture, owned, check_role, check_fault, omission):
    packet = repair41_actual_run(capture, owned, final="checks_return41", omission=omission,
                                check_role=check_role, check_fault=check_fault)
    trace = json.loads((owned / "actual_source_trace.json").read_bytes())
    roles = ["entry", "spec", "observer", *capture.CONTROLS]
    prefix = roles[:roles.index(check_role) + 1]
    assert any(row.get("seam") == "actual returning control-read interruption"
               and row["role"] == check_role and row["checked_roles"] == prefix for row in trace["trace"])
    assert trace["probe_calls"] == trace["asset_calls"] == 0
    checks, inventory = packet["stages"][7:9]
    assert checks["state"] == "failed"
    assert inventory["state"] == ("omitted_output_limit" if omission else "blocked")
    assert inventory["blocker_codes"] == ["interrupted"] and inventory["issue_indices"] == []
    assert packet["overflow"]["omitted_counts"]["inventory_final_rows"] == 0
    if omission:
        assert "stages.final_checks.result" in packet["overflow"]["omitted_sections"]
    else:
        assert checks["result"] == dict(checked_roles=prefix, region_only_roles=[], failed_roles=[check_role])
    (owned / "stopped_source_packet.json").write_bytes(canon(packet, True))
    packet["overflow"]["omitted_counts"]["inventory_final_rows"] = 17
    (owned / "stopped_rows_counterfactual.packet.json").write_bytes(canon(packet, True))
    with pytest.raises(ValueError):
        repair37_validate(capture, packet)


@pytest.mark.parametrize("omission", [False, True])
def test_repair41_full_consumption_before_ids(capture, owned, omission):
    packet = repair41_actual_run(capture, owned, final="checks_return41", omission=omission, cut="before_ids")
    assert all(asset["prior_attribution"] is None for asset in packet["asset_records"])
    if not omission:
        packet["stages"][7]["result"]["checked_roles"] = []
        packet["stages"][7]["result"]["failed_roles"] = []
        (owned / "before_ids_blocked_counterfactual.packet.json").write_bytes(canon(packet, True))
        with pytest.raises(ValueError):
            repair37_validate(capture, packet)


@pytest.mark.parametrize("fault", ["keyboard", "eintr"])
@pytest.mark.parametrize("omission", [False, True])
def test_repair41_f_stop_source(capture, owned, fault, omission):
    packet = repair41_actual_run(capture, owned, final="normal", omission=omission, fault=fault)
    checks, inventory = packet["stages"][7:9]
    assert checks["state"] == "blocked" and checks["result"] is None
    assert checks["blocker_codes"] == inventory["blocker_codes"] == ["interrupted"]
    assert inventory["state"] == ("omitted_output_limit" if omission else "blocked")
    assert packet["overflow"]["omitted_counts"]["inventory_final_rows"] == 0
    trace = json.loads((owned / "actual_source_trace.json").read_bytes())
    assert trace["probe_calls"] == trace["asset_calls"] == 0


@pytest.mark.parametrize("rows", [2, 17, 1024, 2777])
def test_repair41_stopped_omission_is_not_a_count_list(capture, rows):
    packet = json.loads(repair41_case(0))
    packet["overflow"]["omitted_counts"]["inventory_final_rows"] = rows
    with pytest.raises(ValueError):
        repair37_validate(capture, packet)


@pytest.mark.parametrize("kind", [
    "unlisted_basis_omission", "missing_inventory_omission", "false_overflow",
    "promoted_omission", "invented_omission_stage", "g0_fake_omission_basis",
    "not_started_basis_prior", "g0_asset_stage", "negative_stage_as_complete",
    "missing_issue_index", "out_of_range_issue_index", "wrong_issue_stage",
    "missing_positive_basis", "complete_config_null", "inventory_stage_complete_without_inventory",
    "invented_partial_inventory", "complete_final_checks_with_failed_roles",
])
def test_repair37_coupled_state_and_omission_negatives(capture, kind):
    packet = (repair37_serializer_omission()["packet"]
              if "omission" in kind or kind == "false_overflow" else repair37_packet())
    if kind == "unlisted_basis_omission":
        packet["overflow"]["omitted_sections"].remove("stages.trusted_basis.result")
    elif kind == "missing_inventory_omission":
        packet["overflow"]["omitted_sections"].remove("inventory_final")
    elif kind == "false_overflow":
        packet["overflow"]["occurred"] = False
    elif kind == "promoted_omission":
        packet.update(packet_kind="complete", exit_code=0)
    elif kind == "invented_omission_stage":
        packet["overflow"]["omitted_sections"].append("stages.checkpoint_full.result")
    elif kind == "g0_fake_omission_basis":
        packet["run_id"] = None
    elif kind == "not_started_basis_prior":
        packet["stages"][0].update(state="not_started", result=None)
    elif kind == "g0_asset_stage":
        packet = repair37_packet(12)
        packet["stages"][3].update(state="complete", result=repair37_packet()["stages"][3]["result"])
    elif kind == "negative_stage_as_complete":
        packet["stages"][4]["state"] = "complete"
    elif kind == "missing_issue_index":
        packet["stages"][4]["issue_indices"] = []
    elif kind == "out_of_range_issue_index":
        packet["stages"][4]["issue_indices"] = [99]
    elif kind == "wrong_issue_stage":
        packet["issues"][0]["stage"] = "template"
    elif kind == "missing_positive_basis":
        packet["stages"][0].update(result=None)
    elif kind == "complete_config_null":
        packet["stages"][3]["result"] = None
    elif kind == "inventory_stage_complete_without_inventory":
        packet["inventory_initial"]["state"] = "partial"
    elif kind == "invented_partial_inventory":
        packet["stages"][1]["state"] = "partial"
    elif kind == "complete_final_checks_with_failed_roles":
        packet["stages"][7]["result"]["failed_roles"] = ["template"]
    with pytest.raises(ValueError):
        repair37_validate(capture, packet)


@pytest.mark.parametrize("path,value,valid", [
    (("ihdr", "width"), 1, True), (("ihdr", "width"), 2**31 - 1, True),
    (("ihdr", "width"), 0, False), (("ihdr", "height"), 2**31, False),
    (("colour", "gama_u32"), None, True), (("colour", "gama_u32"), 1, True),
    (("colour", "gama_u32"), 2**32 - 1, True), (("colour", "gama_u32"), 0, False),
    (("colour", "gama_u32"), 2**32, False), (("colour", "srgb_intent"), 3, True),
    (("colour", "srgb_intent"), 4, False),
    (("colour", "chrm_u32_values"), [0, 2**32 - 1] * 4, True),
    (("colour", "chrm_u32_values"), [2**32] * 8, False),
    (("ihdr", "interlace_method"), 1, True), (("ihdr", "interlace_method"), 2, False),
    (("ihdr", "compression_method"), 1, False),
])
def test_repair37_png_scalar_boundaries(capture, path, value, valid):
    observation = repair37_packet()["stages"][2]["result"]
    observation[path[0]][path[1]] = value
    if valid:
        capture._observer_result(observation, "png")
    else:
        with pytest.raises(ValueError):
            capture._observer_result(observation, "png")


def repair37_receive_trace(ctx, monkeypatch):
    """Independent UNIT read seam; discarded bytes do not become retained capture bytes."""
    original = os.read
    received = {name: bytearray() for name in ("stdout", "stderr")}
    def read(fd, length):
        name = next((name for name in received if name in ctx.pipe_objects
                     and fd == ctx.pipe_objects[name].fileno()), None)
        data = original(fd, length)
        if name is not None:
            received[name].extend(data)
        return data
    monkeypatch.setattr(os, "read", read)
    return received


def repair37_save_receive_trace(owned, received, ctx):
    for name, data in received.items():
        (owned / ("unit_received_" + name + ".bin")).write_bytes(data)
        record = ctx.result["streams"][name]
        assert len(data) == record["received_bytes"] and sha(data) == record["received_sha256"]


@pytest.mark.parametrize("kind", ["KeyboardInterrupt", "selector_EIO", "early_primary",
                                 "early_primary_interrupt"])
def test_repair37_exceptional_real_owned_child(capture, owned, monkeypatch, kind):
    """Astra129 boundary reproduced; actual stdlib child and measured first-failure grace."""
    ctx = unit_context(capture, owned)
    ctx.entry["deadlines"].update(termination_grace_ns=300_000_000, drain_ns=500_000_000)
    child = (REVIEW129_ROOT / "owned_child_literal.py").read_bytes()
    monkeypatch.setattr(capture, "CHILD_BOOTSTRAP_SOURCE", child)
    actual_selector = capture.selectors.DefaultSelector
    actual_signal = capture._send_signal
    trace = []
    received = repair37_receive_trace(ctx, monkeypatch)
    class BrokenSelector(actual_selector):
        def select(self, timeout=None):
            ready = super().select(timeout)
            if any(key.data == "stderr" for key, _ in ready):
                if kind.startswith("early_primary"):
                    ctx.fail(OSError(errno.EPERM, "repair37 preserved primary"), "pump", "unit_primary")
                if kind in ("KeyboardInterrupt", "early_primary_interrupt"):
                    raise KeyboardInterrupt("repair37 selector interruption seam")
                raise OSError(errno.EIO, "repair37 selector EIO seam")
            return ready
    def observed_signal(context, name):
        trace.append(dict(signal=name, monotonic_ns=time.monotonic_ns(),
                          failed_at=context.failed_at, stdin_owned="stdin" in context.pipe_objects))
        return actual_signal(context, name)
    monkeypatch.setattr(capture.selectors, "DefaultSelector", BrokenSelector)
    monkeypatch.setattr(capture, "_send_signal", observed_signal)
    try:
        raw = capture._pump(ctx, b"x" * 1048576)
        (owned / "exception.stdout.bin").write_bytes(raw)
        (owned / "exception_lifecycle_trace.json").write_bytes(canon(trace, True))
        assert ctx.process.poll() is not None and ctx.result["child"]["reaped"]
        assert ctx.result["child"]["returncode"] == -signal.SIGKILL
        assert [row["signal"] for row in trace] == ["SIGTERM", "SIGKILL"]
        assert all(not row["stdin_owned"] for row in trace)
        assert trace[1]["monotonic_ns"] >= ctx.failed_at + 300_000_000
        assert ctx.result["child"]["termination_elapsed_ns"] == ctx.failed_at - ctx.entry["started_monotonic_ns"]
        assert ctx.result["child"]["termination_trigger"] == (
            "interruption" if kind == "KeyboardInterrupt" else "failure")
        if kind.startswith("early_primary"):
            assert ctx.errors["primary"]["errno"] == errno.EPERM
            if kind == "early_primary_interrupt":
                assert ctx.errors["secondary"][0]["exception_type"] == "KeyboardInterrupt"
            else:
                assert ctx.errors["secondary"][0]["errno"] == errno.EIO
            assert trace[0]["failed_at"] == trace[1]["failed_at"]
        for record in ctx.result["streams"].values():
            assert record["read_state"] != "open"
            assert record["eof_observed"] or record["read_state"] in ("failed", "abandoned")
            assert record["unread_bytes_unknown"] is not record["eof_observed"]
            assert record["post_failure_received_bytes"] <= 65536
            assert record["received_bytes"] == (record["acknowledged_prefix_bytes"]
                    + record["unknown_retention_bytes"] + record["discarded_bytes"])
    finally:
        if ctx.process is not None and ctx.process.poll() is None:
            ctx.process.kill()
            ctx.process.wait(timeout=5)
            (owned / "fixture_emergency_cleanup.json").write_bytes(canon({
                "pid": ctx.process.pid, "returncode": ctx.process.returncode, "reaped": True,
            }, True))
        close_unit(capture, ctx)
        repair37_save_receive_trace(owned, received, ctx)


@pytest.mark.parametrize("phase", ["construct", "register"])
def test_repair37_exceptional_selector_setup_owned_child(capture, owned, monkeypatch, phase):
    ctx = unit_context(capture, owned)
    monkeypatch.setattr(capture, "CHILD_BOOTSTRAP_SOURCE",
                        (REVIEW129_ROOT / "owned_child_literal.py").read_bytes())
    actual_selector = capture.selectors.DefaultSelector
    actual_signal = capture._send_signal
    trace = []
    received = repair37_receive_trace(ctx, monkeypatch)
    class SetupFailure(actual_selector):
        def __init__(self):
            if phase == "construct":
                raise OSError(errno.EIO, "repair37 selector construction seam")
            super().__init__()
        def register(self, *args, **kwargs):
            raise OSError(errno.EIO, "repair37 selector registration seam")
    def observed_signal(context, name):
        trace.append(dict(signal=name, monotonic_ns=time.monotonic_ns(),
                          failed_at=context.failed_at, stdin_owned="stdin" in context.pipe_objects))
        return actual_signal(context, name)
    monkeypatch.setattr(capture.selectors, "DefaultSelector", SetupFailure)
    monkeypatch.setattr(capture, "_send_signal", observed_signal)
    try:
        capture._pump(ctx, b"x")
        assert ctx.result["child"]["reaped"] and ctx.process.poll() is not None
        assert ctx.errors["primary"]["errno"] == errno.EIO
        assert ctx.result["child"]["termination_trigger"] == "failure"
        assert trace and all(not row["stdin_owned"] for row in trace)
        assert all(row["monotonic_ns"] >= ctx.failed_at + ctx.entry["deadlines"]["termination_grace_ns"]
                   for row in trace if row["signal"] == "SIGKILL")
        assert not ctx.pipe_objects
        (owned / "setup_lifecycle_trace.json").write_bytes(canon(trace, True))
    finally:
        if ctx.process is not None and ctx.process.poll() is None:
            ctx.process.kill()
            ctx.process.wait(timeout=5)
            (owned / "fixture_emergency_cleanup.json").write_bytes(canon({
                "pid": ctx.process.pid, "returncode": ctx.process.returncode, "reaped": True,
            }, True))
        close_unit(capture, ctx)
        repair37_save_receive_trace(owned, received, ctx)


@pytest.mark.parametrize("fault", ["stdin_close", "stdout_close", "cleanup_read", "cleanup_select"])
def test_repair37_exceptional_secondary_io_owned_child(capture, owned, monkeypatch, fault):
    ctx = unit_context(capture, owned)
    ctx.entry["deadlines"].update(termination_grace_ns=300_000_000, drain_ns=500_000_000)
    monkeypatch.setattr(capture, "CHILD_BOOTSTRAP_SOURCE",
                        (REVIEW129_ROOT / "owned_child_literal.py").read_bytes())
    actual_popen = capture.subprocess.Popen
    actual_selector = capture.selectors.DefaultSelector
    received = repair37_receive_trace(ctx, monkeypatch)
    actual_read = os.read
    actual_signal = capture._send_signal
    fired, trace = [], []
    class FailedClose:
        def __init__(self, pipe):
            self.pipe = pipe
        def fileno(self):
            return self.pipe.fileno()
        def close(self):
            fired.append("close")
            self.pipe.close()
            raise OSError(errno.ENOSPC, "repair37 after-actual-close uncertainty seam")
    def factory(*args, **kwargs):
        process = actual_popen(*args, **kwargs)
        if fault in ("stdin_close", "stdout_close"):
            name = fault.removesuffix("_close")
            setattr(process, name, FailedClose(getattr(process, name)))
        return process
    class BrokenSelector(actual_selector):
        def select(self, timeout=None):
            ready = super().select(timeout)
            if any(key.data == "stderr" for key, _ in ready):
                raise OSError(errno.EIO, "repair37 primary selector seam")
            return ready
    def read(fd, length):
        if (ctx.failed_at is not None and fault == "cleanup_read"
                and "stdout" in ctx.pipe_objects and fd == ctx.pipe_objects["stdout"].fileno()):
            fired.append("read")
            raise OSError(errno.ENOSPC, "repair37 cleanup receive seam")
        return actual_read(fd, length)
    def readiness(*args):
        fired.append("select")
        raise OSError(errno.ENOSPC, "repair37 fallback readiness seam")
    def observed_signal(context, name):
        trace.append(dict(signal=name, monotonic_ns=time.monotonic_ns(),
                          failed_at=context.failed_at, stdin_owned="stdin" in context.pipe_objects))
        return actual_signal(context, name)
    monkeypatch.setattr(capture.subprocess, "Popen", factory)
    monkeypatch.setattr(capture.selectors, "DefaultSelector", BrokenSelector)
    monkeypatch.setattr(capture, "_send_signal", observed_signal)
    monkeypatch.setattr(os, "read", read)
    if fault == "cleanup_select":
        monkeypatch.setattr(capture.select, "select", readiness)
    try:
        capture._pump(ctx, b"x" * 1048576)
        assert ctx.result["child"]["reaped"] and ctx.process.poll() is not None
        assert ctx.result["child"]["returncode"] == -signal.SIGKILL
        assert fired and (len(fired) == 1 if fault != "cleanup_select" else True)
        assert ctx.errors["primary"]["errno"] == errno.EIO
        assert any(record["errno"] == errno.ENOSPC for record in ctx.errors["secondary"])
        assert all(not row["stdin_owned"] for row in trace)
        assert [row["signal"] for row in trace] == ["SIGTERM", "SIGKILL"]
        assert trace[1]["monotonic_ns"] >= ctx.failed_at + 300_000_000
        assert trace[0]["failed_at"] == trace[1]["failed_at"] == ctx.failed_at
        if fault == "stdin_close":
            assert ctx.transmission["stdin_close_state"] == "failed_unknown" and ctx.closure_uncertain
        if fault == "stdout_close":
            assert ctx.closure_uncertain
        if fault == "cleanup_read":
            assert ctx.result["streams"]["stdout"]["read_state"] == "failed"
            assert ctx.result["streams"]["stdout"]["unread_bytes_unknown"] is True
        assert all(record["read_state"] != "open" for record in ctx.result["streams"].values())
        (owned / "secondary_lifecycle_trace.json").write_bytes(canon({
            "fault_seam": fault, "trace": trace, "fault_calls": fired,
        }, True))
    finally:
        if ctx.process is not None and ctx.process.poll() is None:
            ctx.process.kill()
            ctx.process.wait(timeout=5)
            (owned / "fixture_emergency_cleanup.json").write_bytes(canon({
                "pid": ctx.process.pid, "returncode": ctx.process.returncode, "reaped": True,
            }, True))
        close_unit(capture, ctx)
        repair37_save_receive_trace(owned, received, ctx)


def test_repair37_broken_selector_retains_earlier_natural_drain(capture, owned, monkeypatch):
    """Actual owned direct child plus separately fixture-owned inherited-pipe holder."""
    ctx = unit_context(capture, owned)
    ctx.entry["deadlines"].update(termination_grace_ns=300_000_000, drain_ns=100_000_000)
    actual_popen = capture.subprocess.Popen
    actual_selector = capture.selectors.DefaultSelector
    actual_interval = capture._poll_interval
    received = repair37_receive_trace(ctx, monkeypatch)
    holder, observed_exit, bounds = [], [], []
    read_fd, write_fd = os.pipe()
    def factory(args, **kwargs):
        kwargs["stdout"] = write_fd
        process = actual_popen(args, **kwargs)
        process.stdout = io.FileIO(read_fd, "rb", closefd=True)
        actual_poll = process.poll
        def poll():
            result = actual_poll()
            if result is not None and not observed_exit:
                observed_exit.append(time.monotonic_ns())
            return result
        process.poll = poll
        holder.append(actual_popen(
            [INTERPRETER, "-I", "-S", "-B", "-c", "import time;time.sleep(10)"],
            pass_fds=(write_fd,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        os.close(write_fd)
        return process
    class BrokenSelector(actual_selector):
        def select(self, timeout=None):
            if ctx.result["child"]["reaped"]:
                raise OSError(errno.EIO, "repair37 after-natural-exit selector seam")
            return super().select(timeout)
    def interval(now, deadline):
        if observed_exit:
            bounds.append(dict(now=now, deadline=deadline, failed_at=ctx.failed_at))
        return actual_interval(now, deadline)
    monkeypatch.setattr(capture.subprocess, "Popen", factory)
    monkeypatch.setattr(capture.selectors, "DefaultSelector", BrokenSelector)
    monkeypatch.setattr(capture, "_poll_interval", interval)
    monkeypatch.setattr(capture, "CHILD_BOOTSTRAP_SOURCE",
                        b"import os\nwhile os.read(0,65536):pass\nos.write(1,b'x')\n")
    try:
        capture._pump(ctx, b"x")
        assert ctx.result["child"]["reaped"] and ctx.result["child"]["returncode"] == 0
        assert ctx.errors["primary"]["errno"] == errno.EIO
        assert ctx.result["child"]["signals"] == []
        assert observed_exit and any(row["failed_at"] is not None for row in bounds)
        assert all(row["deadline"] <= observed_exit[0] + 100_000_000 for row in bounds)
        assert ctx.failed_at >= observed_exit[0]
        record = ctx.result["streams"]["stdout"]
        assert not record["eof_observed"] and record["read_state"] == "abandoned"
        assert record["unread_bytes_unknown"] is True
        (owned / "natural_drain_trace.json").write_bytes(canon({
            "exit_observed_ns": observed_exit[0], "poll_bounds": bounds,
            "first_failure_ns": ctx.failed_at, "drain_ns": 100_000_000,
            "grace_ns": 300_000_000,
        }, True))
    finally:
        for process in holder:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            (owned / "fixture_holder_status.json").write_bytes(canon({
                "pid": process.pid, "returncode": process.returncode, "reaped": True,
            }, True))
        if ctx.process is not None and ctx.process.poll() is None:
            ctx.process.kill()
            ctx.process.wait(timeout=5)
            (owned / "fixture_emergency_cleanup.json").write_bytes(canon({
                "pid": ctx.process.pid, "returncode": ctx.process.returncode, "reaped": True,
            }, True))
        close_unit(capture, ctx)
        repair37_save_receive_trace(owned, received, ctx)


@pytest.mark.parametrize("key", ["num_frames", "num_plays"])
@pytest.mark.parametrize("number,valid", [(0, True), (2**32 - 1, True), (-1, False), (2**32, False)])
def test_repair37_apng_unsigned_raw_domains(capture, key, number, valid):
    """Raw descriptor validation seam; no APNG semantics or encoded-image observation."""
    observation = repair37_packet()["stages"][2]["result"]
    observation["apng"].update(present=True, actl_raw_fields={"num_frames": 0, "num_plays": 0})
    observation["apng"]["actl_raw_fields"][key] = number
    if valid:
        capture._observer_result(observation, "png")
    else:
        with pytest.raises(ValueError):
            capture._observer_result(observation, "png")


@pytest.mark.parametrize("serializer_omission", [False, True])
def test_repair37_source_run_late_basis_interruption(capture, owned, serializer_omission):
    """Frozen run/issue/finalizer with UNIT G2 seed, not caller-main or input acceptance."""
    caller = types.ModuleType("repair37_frozen_late_basis_unit")
    exec(compile(CALLER_BYTES, "<repair37-frozen-late-basis-unit>", "exec"), caller.__dict__)
    seed = repair37_packet()
    def execute(context, *_):
        context.packet = copy.deepcopy(seed)
        context.packet["issues"] = []
        context.packet["secondary_errors"] = []
        context.packet["overflow"] = caller._Context().packet["overflow"]
        context.packet["logical_io"]["reads"] = [
            record for record in context.packet["logical_io"]["reads"]
            if record["role"] not in caller.ASSETS]
        for key in context.packet["logical_io"]["selected"]:
            if key != "limit_bytes":
                context.packet["logical_io"]["selected"][key] = 0
        for stage in context.packet["stages"]:
            if stage["name"] not in ("trusted_basis", "serialization"):
                stage.update(state="not_started", result=None, issue_indices=[], blocker_codes=[])
        for number, record in enumerate(context.packet["asset_records"]):
            clean = caller._binding(record["role"], record["declared_path"],
                                    record["declared_size"], record["declared_sha256"])
            clean.update(prior_attribution=seed["asset_records"][number]["prior_attribution"],
                         current_resolved_path=None, regions=[])
            context.packet["asset_records"][number] = clean
        ids = [row["slide_id"] for row in seed["inventory_initial"]["frozen_result"]["rows"]]
        context.packet.update(inventory_initial=caller._inventory(ids),
                              inventory_final=caller._inventory(ids),
                              inventory_comparison=caller._comparison(ids))
        context.exit_code = 0
        context.stage_name = "trusted_basis"
        raise KeyboardInterrupt("repair37 UNIT interruption immediately after G2 result")
    caller._execute = execute
    original_encode = caller._encode
    calls = []
    def encode(packet):
        calls.append(None)
        if serializer_omission and len(calls) == 1:
            fault = caller._Fault("stdout_limit", "repair37 UNIT first-encode ceiling", "output")
            fault.bytes_seen = 16777217
            raise fault
        return original_encode(packet)
    caller._encode = encode
    result = caller.run("/unit-unacquired", spec_sha256="0" * 64, entry_binding={})
    (owned / "late_basis.stdout.bin").write_bytes(result["stdout_bytes"])
    (owned / "late_basis.stderr.bin").write_bytes(result["stderr_bytes"])
    (owned / "late_basis_scope.json").write_bytes(canon({
        "unit_execute_seed": "retained G2 declaration facts with unstarted downstream routes",
        "actual_frozen_run_issue_finish_packet": True, "caller_main_execution": False,
        "actual_input_acceptance": False, "first_encode_fault": serializer_omission,
        "natural_16mib_overflow": False,
    }, True))
    packet, summary = repair37_validate(capture, result["packet"])
    assert summary["state"] == "valid" and packet["exit_code"] == 1
    assert packet["stages"][0]["state"] == "failed"
    assert all(record["prior_attribution"] is not None for record in packet["asset_records"])
