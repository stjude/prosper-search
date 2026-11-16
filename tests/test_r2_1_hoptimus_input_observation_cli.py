"""Synthetic caller evidence; schema/stat seams are explicitly unit-only."""

import base64
import binascii
import builtins
import copy
import csv
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import types

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "prosper/evaluation/r2_1_hoptimus_input_observation_cli.py"
OBSERVER = SOURCE.with_name("r2_1_hoptimus_input_admission.py")
SOURCE_BYTES = SOURCE.read_bytes()
OBSERVER_BYTES = OBSERVER.read_bytes()
assert hashlib.sha256(OBSERVER_BYTES).hexdigest() == "768e91414e43a86854c6e0088bdf91b2f10f1c1a3fc0a157c355b1bfa464071b"
INTERPRETER = "/HOME/miniconda3/bin/python3"
NAME = "r21_input_observation_cli_verified_v1"
ROOT = "/PROSPER_WORKSPACE/data/comet_final_HPC"


def sha(encoded):
    return hashlib.sha256(encoded).hexdigest()


def dump(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()


def load(encoded=SOURCE_BYTES, name=NAME):
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(encoded, str(SOURCE), "exec"), module.__dict__)
    return module


@pytest.fixture
def cli(monkeypatch):
    module = load()
    # Parent pytest is not isolated-entry evidence; separate child tests are.
    monkeypatch.setattr(module, "_check_environment", lambda: None)
    originals = {key: getattr(os, key) for key in ("lstat", "stat", "open")}
    forbidden_paths = ("r2_1_hoptimus_main", "r2_1_hoptimus_filter", "r2_1_hoptimus_mosaic_reader",
                       "/r2_1_mosaic_training_", "/r2_1_filter_retrieval_")

    def path_guard(path):
        if isinstance(path, (str, os.PathLike)):
            text = str(path)
            assert not text.startswith(ROOT), "REAL WSI PROBE FORBIDDEN"
            assert not any(word in text for word in forbidden_paths), "MAIN/FILTER IO FORBIDDEN"

    for key, original in originals.items():
        def guarded(path, *args, _original=original, **kwargs):
            path_guard(path)
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(os, key, guarded)
    for owner in (builtins, io):
        original = owner.open

        def guarded_file(path, *args, _original=original, **kwargs):
            path_guard(path)
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(owner, "open", guarded_file)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in module.FORBIDDEN, "PACKAGE/SCIENTIFIC IMPORT FORBIDDEN"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    yield module
    if sys.modules.get(NAME) is module:
        del sys.modules[NAME]
    assert module.OBSERVER_NAME not in sys.modules


def png():
    def chunk(name, body):
        return struct.pack(">I", len(body)) + name + body + struct.pack(">I", binascii.crc32(name + body) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", b"not-decoded") + chunk(b"IEND", b""))


def safefile(header=None):
    header = dump(header or {"x": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]}})
    return struct.pack("<Q", len(header)) + header + b"abc"


def csvbytes(header, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode()


def generated_controls(cli, root, config=b'{"float":1e-6,"url":"https://invalid.example/no-follow"}',
                       template=None, checkpoint=None):
    """No production controls used; observer receipt digest is a labelled seam."""
    root.mkdir()
    assets = {}
    for role, encoded in (("template", template if template is not None else png()),
                          ("model_config", config), ("checkpoint", checkpoint or safefile()),
                          ("selected_wsi", b"synthetic-not-svs")):
        path = root / role
        path.write_bytes(encoded)
        assets[role] = {"path": str(path), "size": len(encoded), "sha256": sha(encoded)}
    wsi_path = assets["selected_wsi"]["path"]
    assets["selected_wsi"].update(path=ROOT + "/10003.svs", slide_id="10003")
    cli_path = root / "captured_cli.py"
    cli_path.write_bytes(SOURCE_BYTES)
    entry = {"path": str(cli_path), "resolved_path": str(cli_path), "size": len(SOURCE_BYTES),
             "sha256": sha(SOURCE_BYTES)}
    observer_path = root / "observer.py"
    observer_path.write_bytes(OBSERVER_BYTES)
    ids = ["10003", "16683"] + [str(20000 + i) for i in range(2776)]
    labels = ["NORMAL"] + ["G" + str(i) for i in range(1, 15)]
    metadata = []
    queries = []
    for i, sid in enumerate(ids):
        if i < 1876:
            patient_index = i
            group_index = i % 15
        else:
            patient_index = i - 1876
            group_index = (patient_index + (1 if patient_index < 45 else 0)) % 15
        patient, group = "P" + str(patient_index), labels[group_index]
        metadata.append([group, sid, "S" + sid, patient])
        queries.append([str(i), str(i), sid, patient, group,
                        str(3835507 - 2777 if i == 0 else 1),
                        "/referenced-only/" + sid, "/referenced-only/" + sid,
                        "1", "0" * 64])
    metadata.extend([["MISC", str(40000 + i), "M" + str(i), "MP" + str(i)] for i in range(143)])
    controls = {}

    def write(role, encoded):
        path = root / (role + ".control")
        path.write_bytes(encoded)
        descriptor = {"path": str(path), "size": len(encoded), "sha256": sha(encoded)}
        controls[role] = descriptor
        return descriptor

    def record(role, **values):
        result = {key: None for key in cli.JSON_KEYS[role].split()}
        result["schema_version"] = 1
        result.update(values)
        return result

    write("metadata", csvbytes(cli.CSV_HEADERS["metadata"], metadata))
    write("targets", csvbytes(cli.CSV_HEADERS["targets"], queries))
    producer = record("cpu_spec", run_id="synthetic_cpu", task="hoptimus_perturb_manifest",
                      sources={"metadata": controls["metadata"], "template": assets["template"]},
                      expected={"raw_slides": 2921, "target_slides": 2778, "patients": 1876,
                                "groups": 15, "cross_label_patients": 45, "feature_dim": 1536,
                                "stimulus_records": 33336}, code_files={},
                      parameters={"metadata": {"id_col": "Slide.Scan.File", "label_col": "Tumor.group",
                                               "patient_col": "UID.Subject"}})
    write("cpu_spec", dump(producer))
    source_bindings = []
    for kind, name, desc in (("spec", "spec", controls["cpu_spec"]),
                             ("source", "metadata", controls["metadata"]),
                             ("source", "template", assets["template"])):
        source_bindings.append(dict(desc, kind=kind, name=name, resolved_path=desc["path"], device=99, inode=1))
    write("cpu_input_bindings", dump(record(
        "cpu_input_bindings", run_id="synthetic_cpu", task="hoptimus_perturb_manifest",
        spec_sha256=controls["cpu_spec"]["sha256"], bindings=source_bindings)))
    outputs = {name: {"size": 1, "sha256": "0" * 64} for name in cli.CPU_OUTPUTS}
    for name, role in (("input_bindings.json", "cpu_input_bindings"), ("queries.csv", "targets"),
                       ("references.csv", "targets")):
        outputs[name] = {k: controls[role][k] for k in ("size", "sha256")}
    write("cpu_complete", dump(record(
        "cpu_complete", outputs=outputs, run_id="synthetic_cpu",
        task="hoptimus_perturb_manifest", scope="hoptimus_perturb_stimulus_manifests_only",
        spec_sha256=controls["cpu_spec"]["sha256"])))
    write("cpu_review", dump(record(
        "cpu_review", disposition="accepted",
        branches={"perturb": {"disposition": "accepted", "scope": "Population and stimulus-manifest preparation only"}},
        artifacts={"perturb_attempt_001/COMPLETE.json": controls["cpu_complete"]["sha256"]})))
    source_rows = []
    for sid in ids[:10] + [str(40000 + i) for i in range(2)]:
        desc = assets["selected_wsi"] if sid == "10003" else {"size": 1, "sha256": "0" * 64}
        source_rows.append(["wsi/" + sid, "wsi", "", "", sid, ROOT + "/" + sid + ".svs",
                            "/prior/physical/" + sid + ".svs", "true", "true",
                            str(desc["size"]), str(desc["size"]), "", desc["sha256"], "observed_only"])
    for role, kind in (("model_config", "model_config"), ("checkpoint", "weight")):
        desc = assets[role]
        source_rows.append([kind + "/h_optimus_1", kind, "h_optimus_1", "", "",
                            "/prior/" + role, "/prior/resolved/" + role, "true", "true",
                            str(desc["size"]), str(desc["size"]), "", desc["sha256"], "observed_only"])
    for i in range(213):
        source_rows.append(["unconsumed/" + str(i), "feature", "", "", "", "/unconsumed/" + str(i),
                            "/unconsumed/" + str(i), "true", "true", "1", "1", "", "0" * 64, "observed_only"])
    write("stage1b_source_inventory", csvbytes(cli.CSV_HEADERS["stage1b_source_inventory"], source_rows))
    panel_outputs = {name: {"size_bytes": 1, "sha256": "0" * 64} for name in cli.PANEL_OUTPUTS}
    inventory = controls["stage1b_source_inventory"]
    panel_outputs["source_inventory.csv"] = {"size_bytes": inventory["size"], "sha256": inventory["sha256"]}
    write("stage1b_complete", dump(record(
        "stage1b_complete", disposition="prepared", scope="encoder_replay_panel_only",
        outputs=panel_outputs, input_inventory_sha256=inventory["sha256"])))
    write("stage1b_review", dump(record(
        "stage1b_review", disposition="accepted",
        scope="Stage1b sampled panel/reference and current input-byte preparation only",
        artifacts={"stage1b_attempt_001/COMPLETE.json": controls["stage1b_complete"]["sha256"],
                   "stage1b_attempt_001/source_inventory.csv": inventory["sha256"]})))
    write("observer_review", dump(record(
        "observer_review", disposition="accepted_repaired_candidate002_two_files_only",
        stage="encoded_input_observation_primitives_implementation",
        bindings={"contract_sha256": cli.OBSERVER_CONTRACT_SHA,
                  "module": {"path": "prosper/evaluation/r2_1_hoptimus_input_admission.py",
                             "size": len(OBSERVER_BYTES), "sha256": sha(OBSERVER_BYTES)}})))
    spec = {"schema_version": 1, "run_id": "synthetic_caller",
            "task": "hoptimus_encoded_input_observation",
            "observer": {"path": str(observer_path), "size": len(OBSERVER_BYTES), "sha256": sha(OBSERVER_BYTES)},
            "controls": controls, "assets": assets, "lexical_root": ROOT,
            "expected": copy.deepcopy(cli.EXPECTED), "selection": {
                "selected_wsi_ids": ["10003"], "deferred_wsi_ids": ["16683"]},
            "limits": copy.deepcopy(cli.LIMITS)}
    path = root / "spec.json"
    path.write_bytes(dump(spec))
    return {"spec": spec, "path": path, "entry": entry, "ids": ids,
            "root": root, "wsi_path": wsi_path, "observer_review_sha": controls["observer_review"]["sha256"]}


def synthetic_probe(cli, path):
    """UNIT metadata seam: no lstat/stat/resolve/open on prescribed WSI paths."""
    fingerprint = [1, 1, 32768, 1, 1, 1]
    probe = {"lexical_path": path, "lexical_type": "regular", "resolved_path": "/synthetic/resolved.svs",
             "resolved_type": "regular", "size_bytes": 1, "probe_coherent": True,
             "status": "observed_regular", "error_operation": None, "error_errno": None, "error_message": None}
    book = {"lexical_before": list(fingerprint), "target_before": list(fingerprint),
            "lexical_after": list(fingerprint), "target_after": list(fingerprint),
            "resolved_before": "/synthetic/resolved.svs", "resolved_after": "/synthetic/resolved.svs",
            "errors": [], "coherent": True}
    return probe, book, 0


def install_unit_seams(cli, fixture, monkeypatch, schema_match=False):
    monkeypatch.setattr(cli, "OBSERVER_REVIEW_SHA", fixture["observer_review_sha"])
    monkeypatch.setattr(cli, "_probe_path", lambda path: synthetic_probe(cli, path))
    original = cli._bookend

    def translated(path, which):
        return original(fixture["wsi_path"] if path == ROOT + "/10003.svs" else path, which)

    monkeypatch.setattr(cli, "_bookend", translated)
    if schema_match:
        load_basis = cli._load_basis

        def unit_schema(context, spec):
            load_basis(context, spec)
            context.observer.compare_vit224_schema = lambda observation: {
                "match": True, "missing_names": [], "unexpected_names": [],
                "shape_mismatches": [], "dtype_issues": []}

        monkeypatch.setattr(cli, "_load_basis", unit_schema)


def execute(cli, fixture):
    return cli.run(str(fixture["path"]), spec_sha256=sha(fixture["path"].read_bytes()),
                   entry_binding=fixture["entry"])


def stage(packet, name):
    return next(r for r in packet["stages"] if r["name"] == name)


def check_packet(result):
    packet = result["packet"]
    assert result["exit_code"] == packet["exit_code"]
    assert packet["model_use_authorized"] is False
    assert json.loads(result["stdout_bytes"]) == packet
    assert result["stdout_bytes"].endswith(b"\n")
    assert len(result["stdout_bytes"]) <= 16 * 1024 * 1024
    assert set(result) == {"packet", "stdout_bytes", "stderr_bytes", "exit_code"}
    return packet


@pytest.mark.parametrize("kind", ["absent", "malformed", "hash", "unreadable"])
def test_g0_spec_constructible_partial_error(cli, tmp_path, kind):
    entry_path = tmp_path / "cli.py"
    entry_path.write_bytes(SOURCE_BYTES)
    entry = {"path": str(entry_path), "resolved_path": str(entry_path),
             "size": len(SOURCE_BYTES), "sha256": sha(SOURCE_BYTES)}
    path = tmp_path / "spec.json"
    if kind != "absent":
        path.write_bytes(b'{"incomplete":true}')
    digest = "x" if kind == "hash" else sha(path.read_bytes()) if path.exists() else "0" * 64
    if kind == "unreadable":
        path.unlink()
        path.mkdir()
    packet = check_packet(cli.run(str(path), spec_sha256=digest, entry_binding=entry))
    assert packet["exit_code"] == 1 and packet["run_id"] is None
    assert packet["observer_binding"] is None and packet["asset_records"] == []
    assert packet["inventory_initial"]["unvisited_ids"] is None
    assert packet["inventory_comparison"]["uncomparable_ids"] is None
    assert packet["logical_io"]["controls"]["declared_bytes"] is None
    assert len(packet["stages"]) == 10


def test_independent_g0_arguments(cli):
    packet = check_packet(cli.run("/synthetic/missing", spec_sha256="a" * 64, entry_binding=None))
    assert packet["spec_sha256"] == "a" * 64
    assert [r["role"] for r in packet["control_bindings"]] == ["spec"]
    assert packet["control_bindings"][0]["declared_size"] is None
    assert packet["entry_binding"] is None


def test_g1_membership_on_control_failure(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch)
    Path(fixture["spec"]["controls"]["metadata"]["path"]).write_bytes(b"broken")
    packet = check_packet(execute(cli, fixture))
    assert packet["run_id"] == "synthetic_caller"
    assert len(packet["control_bindings"]) == 11 and len(packet["asset_records"]) == 4
    assert all(a["prior_attribution"] is None and a["custody_state"] == "unacquired"
               for a in packet["asset_records"])
    assert stage(packet, "trusted_basis")["result"] is None
    assert packet["inventory_final"]["unvisited_ids"] is None


def test_real_observer_complete_negative_schema(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 2
    assert packet["inventory_initial"]["state"] == packet["inventory_final"]["state"] == "complete"
    assert packet["inventory_initial"]["probes"] is None
    assert len(packet["inventory_initial"]["frozen_result"]["rows"]) == 2778
    assert stage(packet, "checkpoint_header")["result"]["safetensors_observation"]["tensor_count"] == 1
    assert stage(packet, "checkpoint_header")["result"]["schema_comparison"]["match"] is False
    assert stage(packet, "checkpoint_full")["state"] == "blocked"
    checkpoint = packet["asset_records"][2]
    assert checkpoint["custody_state"] == "region_only"
    assert all(c["observed_sha256"] is None for c in checkpoint["whole_file_checks"])
    assert all(r["byte_equal"] is True for r in checkpoint["regions"])
    assert stage(packet, "final_checks")["result"]["region_only_roles"] == ["checkpoint"]
    assert len(result_bytes := dump(packet)) < 16 * 1024 * 1024
    assert result_bytes


def test_unit_schema_gate_full_stream_positive(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 0
    assert all(a["custody_state"] == "full_verified" for a in packet["asset_records"])
    assert stage(packet, "config")["result"]["parsed_json"]["float"] == 1e-6
    assert stage(packet, "config")["result"]["raw_utf8"] == '{"float":1e-6,"url":"https://invalid.example/no-follow"}'
    selected = packet["logical_io"]["selected"]
    assert selected["returned_bytes"] == 2 * sum(a["size"] for a in fixture["spec"]["assets"].values())
    assert selected["eof_probe_requests"] == 8
    assert selected["requested_bytes"] == selected["returned_bytes"] + 8
    assert packet["inventory_comparison"]["changed_ids"] == []


@pytest.mark.parametrize("encoded,code,raw", [
    (b"\xff", "config_utf8", None), (b"\xef\xbb\xbf{}", "json_bom", "\ufeff{}"),
    (b"[]", "json_root", "[]"), (b'{"x":NaN}', "json_nonfinite", '{"x":NaN}'),
    (b'{"x":1e999}', "json_nonfinite", '{"x":1e999}'),
    (b'{"x":1,"x":2}', "json_duplicate", '{"x":1,"x":2}'),
    (b'{"x":"\\ud800"}', "json_parse", '{"x":"\\ud800"}'),
    (b'{"x":' + b"[" * 16 + b"0" + b"]" * 16 + b"}", "json_depth", None),
    (b'{"x":' + b"9" * 129 + b"}", "json_number_limit", None),
    (b"{} trailing", "json_parse", "{} trailing"),
])
def test_config_format_retention_negative2(cli, tmp_path, monkeypatch, encoded, code, raw):
    fixture = generated_controls(cli, tmp_path / "owned", config=encoded)
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 2
    result = stage(packet, "config")["result"]
    assert result["encoded_size"] == len(encoded) and result["encoded_sha256"] == sha(encoded)
    assert result["parsed_json"] is None
    assert result["raw_utf8"] == (encoded.decode() if raw is None and encoded != b"\xff" else raw)
    assert stage(packet, "checkpoint_full")["state"] == "blocked"
    assert code in [i["code"] for i in packet["issues"]]


def test_captured_header_limit_is_negative_not_acquisition_limit(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned", checkpoint=struct.pack("<Q", 1048577))
    install_unit_seams(cli, fixture, monkeypatch)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 2
    assert any(i["kind"] == "limit" and i["code"] == "header_length" for i in packet["issues"])
    assert [r["name"] for r in packet["asset_records"][2]["regions"]] == ["prefix8"]


@pytest.mark.parametrize("status", ["missing_lexical", "inaccessible", "not_regular", "resolution_error"])
def test_unselected_negative_mandatory2_does_not_block(cli, tmp_path, monkeypatch, status):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)

    def probe(path):
        row, book, outcome = synthetic_probe(cli, path)
        if path.endswith("/16683.svs"):
            row.update(status=status, error_operation="resolve", error_errno=errno.ENOTDIR,
                       error_message="synthetic unavailable")
            if status == "missing_lexical":
                row.update(lexical_type="absent", resolved_path=None, resolved_type=None, size_bytes=None)
            elif status == "not_regular":
                row.update(lexical_type="directory", resolved_type="directory")
            outcome = 2
        return row, book, outcome

    monkeypatch.setattr(cli, "_probe_path", probe)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 2
    assert stage(packet, "checkpoint_full")["state"] == stage(packet, "selected_wsi_full")["state"] == "complete"
    assert all(a["custody_state"] == "full_verified" for a in packet["asset_records"])
    assert not any(s["state"] == "blocked" for s in packet["stages"])


def test_partial_inventory_real_complements(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch)
    probe = cli._probe_path

    def interruption(path):
        if path.endswith("/" + fixture["ids"][2] + ".svs"):
            raise RuntimeError("not an ELOOP")
        return probe(path)

    monkeypatch.setattr(cli, "_probe_path", interruption)
    packet = check_packet(execute(cli, fixture))
    output = packet["inventory_initial"]
    assert packet["exit_code"] == 1 and output["state"] == "partial"
    assert output["visited_ids"] == fixture["ids"][:2]
    assert output["in_progress_id"] == fixture["ids"][2]
    assert output["unvisited_ids"] == fixture["ids"][3:]
    assert list(output["probes"]) == fixture["ids"][:2]
    assert output["frozen_result"] is None


@pytest.mark.parametrize("payload", [b"\xff", b"{", b"[]", b'{"a":1,"a":2}', b'{"a":Infinity}',
                                  b'{"a":' + b"9" * 10000 + b"}", b'{"a":"\\udfff"}'])
def test_strict_json_rejections(cli, payload):
    with pytest.raises(cli._Fault):
        cli._json_bytes(payload)


@pytest.mark.parametrize("column", ["-1", "+1", "01", "1.0", "1e2", "9" * 10000, ""])
def test_csv_canonical_integer(cli, column):
    with pytest.raises(cli._Fault):
        cli._decimal(column)


@pytest.mark.parametrize("mutation", ["bool", "float", "limit", "role", "root", "digest", "path"])
def test_spec_declaration_rejections(cli, tmp_path, mutation):
    spec = json.loads((SOURCE.parents[2] / "examples/r2_1_hoptimus_input_observation.json").read_bytes())
    if mutation == "bool":
        spec["limits"]["spec_bytes"] = True
    elif mutation == "float":
        spec["expected"]["groups"] = 15.0
    elif mutation == "limit":
        spec["limits"]["stdout_packet_bytes"] += 1
    elif mutation == "role":
        del spec["controls"]["cpu_spec"]
    elif mutation == "root":
        spec["lexical_root"] = "/alternate"
    elif mutation == "digest":
        spec["observer"]["sha256"] = "a" * 64
    else:
        spec["controls"]["metadata"]["path"] = "/bad\npath"
    with pytest.raises(cli._Fault):
        cli._declaration(spec, str(tmp_path))


def test_symlink_spec_parent_and_dotdot_are_not_collapsed(cli, tmp_path):
    assert cli._absolute("../inputs/source", str(tmp_path) + "/lexical/link") == str(tmp_path) + "/lexical/link/../inputs/source"
    assert cli._absolute("link/../spec", str(tmp_path)) == str(tmp_path) + "/link/../spec"


@pytest.mark.parametrize("number,status,outcome", [
    (errno.ENOENT, "missing_lexical", 2), (errno.EACCES, "inaccessible", 2),
    (errno.EPERM, "inaccessible", 2), (errno.ENOTDIR, "resolution_error", 2),
    (errno.ELOOP, "resolution_error", 2), (errno.EIO, "io_error", 1),
])
def test_probe_errno(cli, tmp_path, monkeypatch, number, status, outcome):
    monkeypatch.setattr(cli.os, "lstat", lambda path: (_ for _ in ()).throw(OSError(number, "controlled")))
    probe, book, observed = cli._probe_path(str(tmp_path / "leaf"))
    assert probe["status"] == status and observed == outcome
    assert book["errors"][0]["errno"] == number


def test_runtime_eloop_context_genuine_not_message(cli, tmp_path, monkeypatch):
    path = tmp_path / "leaf"
    path.write_bytes(b"x")

    def loop(self, strict=True):
        try:
            raise OSError(errno.ELOOP, "underlying loop")
        except OSError:
            raise RuntimeError("stdlib loop projection")

    monkeypatch.setattr(cli.Path, "resolve", loop)
    probe, book, outcome = cli._probe_path(str(path))
    assert outcome == 2 and probe["status"] == "resolution_error"
    assert probe["error_errno"] is None
    assert book["errors"][0]["error_type"] == "RuntimeError"

    def unrelated(self, strict=True):
        raise RuntimeError("symlink loop wording alone")

    monkeypatch.setattr(cli.Path, "resolve", unrelated)
    with pytest.raises(RuntimeError):
        cli._probe_path(str(path))


def test_real_file_ancestor_alias_symlink_directory_and_broken(cli, tmp_path):
    directory = tmp_path / "actual"
    directory.mkdir()
    (directory / "leaf").write_bytes(b"abc")
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    probe, book, outcome = cli._probe_path(str(alias / "leaf"))
    assert outcome == 0 and probe["lexical_type"] == "regular"
    assert probe["resolved_path"] != probe["lexical_path"] and book["coherent"] is True
    probe, _, outcome = cli._probe_path(str(alias))
    assert outcome == 2 and probe["status"] == "not_regular" and probe["lexical_type"] == "symlink"
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "absent")
    assert cli._probe_path(str(broken))[0]["status"] == "broken_symlink"


@pytest.mark.parametrize("variant", ["match", "digest", "short", "growth", "final", "fifo", "symlink_race"])
def test_actual_tiny_stream_and_race_boundaries(cli, tmp_path, monkeypatch, variant):
    path = tmp_path / "tiny"
    path.write_bytes(b"abcdefgh")
    context = cli._Context()
    record = cli._binding("checkpoint", str(path), 8, sha(b"abcdefgh"))
    record.update(regions=[], current_resolved_path=None)
    if variant == "digest":
        record["declared_sha256"] = "0" * 64
    if variant == "fifo":
        path.unlink()
        os.mkfifo(path)
        with pytest.raises(cli._Fault, match="not regular"):
            cli._Reader(context, record, "initial", 1024)
        return
    if variant == "symlink_race":
        original_open = cli.os.open

        def raced_open(target, flags):
            path.unlink()
            path.symlink_to(tmp_path / "missing")
            return original_open(target, flags)

        monkeypatch.setattr(cli.os, "open", raced_open)
        with pytest.raises(OSError):
            cli._Reader(context, record, "initial", 1024)
        return
    reader = cli._Reader(context, record, "initial", 1024)
    if variant in ("short", "growth"):
        path.write_bytes(b"abc" if variant == "short" else b"abcdefghi")
    if variant in ("digest", "short", "growth"):
        with pytest.raises(cli._Fault):
            reader.full()
        try:
            reader.finish()
        except cli._Fault:
            pass
        assert record["whole_file_checks"][0]["state"] in ("failed", "rejected")
        if variant == "digest":
            assert record["whole_file_checks"][0]["observed_sha256"] == sha(b"abcdefgh")
        return
    reader.full()
    reader.finish()
    if variant == "final":
        path.write_bytes(b"ABCDEFGH")
        with pytest.raises(cli._Fault):
            cli._final_asset(context, record)
        assert record["custody_state"] == "invalidated"
    else:
        cli._final_asset(context, record)
        assert record["custody_state"] == "full_verified"
        assert context.packet["logical_io"]["selected"]["returned_bytes"] == 16


def test_same_descriptor_prefix_hash_reuse(cli, tmp_path):
    encoded = safefile()
    path = tmp_path / "checkpoint"
    path.write_bytes(encoded)
    context = cli._Context()
    record = cli._binding("checkpoint", str(path), len(encoded), sha(encoded))
    record.update(regions=[], current_resolved_path=None)
    reader = cli._Reader(context, record, "initial", 1024)
    prefix = cli._record_region(context, reader, "prefix8", 8)
    header = cli._record_region(context, reader, "header", struct.unpack("<Q", prefix)[0])
    reader.full(prefix + header)
    reader.finish()
    assert reader.io["returned_bytes"] == len(encoded)
    assert record["whole_file_checks"][0]["bytes_read"] == len(encoded)
    cli._final_asset(context, record)
    assert all(r["byte_equal"] for r in record["regions"])
    assert context.packet["logical_io"]["selected"]["returned_bytes"] == 2 * len(encoded)


def test_frozen_567_tensor_structural_schema_positive(cli):
    observer = types.ModuleType("synthetic_frozen_schema")
    exec(compile(OBSERVER_BYTES, str(OBSERVER), "exec"), observer.__dict__)
    schema = observer.classic_vit224_schema()
    header, offset = {}, 0
    for tensor in schema["tensors"]:
        length = tensor["numel"] * 4
        header[tensor["name"]] = {"dtype": "F32", "shape": tensor["shape"],
                                  "data_offsets": [offset, offset + length]}
        offset += length
    encoded = dump(header)
    observation = observer.observe_safetensors(struct.pack("<Q", len(encoded)), encoded,
                                               total_size=8 + len(encoded) + offset)
    assert observer.compare_vit224_schema(observation)["match"] is True
    assert observation["tensor_count"] == 567


def test_actual_16mib_serialized_threshold_and_fallback(cli):
    packet = cli._packet()
    packet["limitations"] = ["x"]
    base_size = len(cli._encode(packet))
    packet["limitations"] = ["x" * (16777216 - base_size + 1)]
    assert len(cli._encode(packet)) == 16777216
    packet["limitations"][0] += "x"
    with pytest.raises(cli._Fault) as caught:
        cli._encode(packet)
    assert caught.value.code == "stdout_limit"
    context = cli._Context()
    # Real serialization threshold, huge synthetic inventory member, not a reduced cap.
    context.packet["inventory_initial"]["bookends"] = {"synthetic": "x" * 16777216}
    result = cli._finish_packet(context)
    packet = check_packet(result)
    assert packet["exit_code"] == 1
    assert packet["inventory_initial"]["state"] == "omitted_output_limit"
    assert packet["inventory_initial"]["visited_ids"] is None
    assert packet["overflow"]["omitted_counts"]["inventory_initial_rows"] == 0
    assert packet["overflow"]["serialization_bytes_seen"] > 16777216


def test_fallback_only_none(cli, monkeypatch):
    monkeypatch.setattr(cli, "_encode", lambda packet: (_ for _ in ()).throw(ValueError("encoder broken")))
    result = cli._finish_packet(cli._Context())
    assert result["packet"] is None and result["exit_code"] == 1 and result["stdout_bytes"] == b""
    assert b"fallback" in result["stderr_bytes"]


def test_issue_caps_and_messages(cli):
    context = cli._Context()
    for _ in range(130):
        context.issue(cli._Fault("synthetic", "x" * 2000))
    assert len(context.packet["issues"]) == 128 and context.packet["issues"][-1]["code"] == "issue_limit"
    assert context.packet["overflow"]["omitted_counts"]["issues"] == 3
    assert len(context.packet["issues"][0]["message"]) == 1024
    for _ in range(34):
        context.issue(OSError(errno.EIO, "cleanup"), secondary=0)
    assert len(context.packet["secondary_errors"]) == 32
    assert context.packet["secondary_errors"][-1]["code"] == "secondary_limit"
    assert context.packet["overflow"]["omitted_counts"]["secondary_errors"] == 3


def test_selected_actual_read_budget_counts_attempts(cli, tmp_path):
    path = tmp_path / "tiny"
    path.write_bytes(b"x")
    context = cli._Context()
    fd = os.open(path, os.O_RDONLY)
    try:
        record = context.read_record("checkpoint", "initial")
        context.packet["logical_io"]["selected"]["returned_bytes"] = 21474836480
        with pytest.raises(cli._Fault):
            context.read(fd, 1, record)
        assert record["requested_bytes"] == 0
    finally:
        os.close(fd)


class Output:
    def __init__(self, partial=False, flush_error=False):
        self.data = bytearray()
        self.partial, self.flush_error = partial, flush_error
        self.calls = 0
        self.buffer = self

    def write(self, encoded):
        self.calls += 1
        if self.partial:
            if self.calls > 1:
                raise OSError(errno.EPIPE, "broken")
            self.data.extend(encoded[:5])
            return 5
        self.data.extend(encoded)
        return len(encoded)

    def flush(self):
        if self.flush_error:
            raise OSError(errno.EIO, "flush")


@pytest.mark.parametrize("partial,flush", [(True, False), (False, True), (False, False)])
def test_main_no_second_document(cli, monkeypatch, partial, flush):
    packet = cli._packet()
    packet.update(exit_code=0, packet_kind="complete")
    encoded = cli._encode(packet)
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: {
        "packet": packet, "stdout_bytes": encoded, "stderr_bytes": b"", "exit_code": 0})
    out, err = Output(partial, flush), Output()
    monkeypatch.setattr(cli.sys, "stdout", out)
    monkeypatch.setattr(cli.sys, "stderr", err)
    actual = cli.main(["--spec", "synthetic", "--spec-sha256", "0" * 64], entry_binding={})
    assert actual == (1 if partial or flush else 0)
    assert bytes(out.data) == (encoded[:5] if partial else encoded)
    assert b"replacement" in err.data if partial or flush else not err.data


def test_main_none_and_usage_help_separation(cli, monkeypatch):
    out, err = Output(), Output()
    monkeypatch.setattr(cli.sys, "stdout", out)
    monkeypatch.setattr(cli.sys, "stderr", err)
    assert cli.main(["bad"], entry_binding={}) == 1 and not out.data
    assert cli.main(["--help"], entry_binding={}) == 0 and out.data.startswith(b"Usage:")
    out.data.clear()
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: {
        "packet": None, "stdout_bytes": b"", "stderr_bytes": b"fallback failed\n", "exit_code": 1})
    assert cli.main(["--spec", "x", "--spec-sha256", "0" * 64], entry_binding={}) == 1
    assert not out.data
    cli._diagnostic(b"x" * 100000)
    assert err.data.endswith(b"[caller stderr truncated]\n")


ISOLATED = r'''
import base64, builtins, hashlib, json, os, pathlib, sys, types
data = json.loads(sys.stdin.buffer.read())
encoded = base64.b64decode(data["source"])
assert hashlib.sha256(encoded).hexdigest() == data["source_sha"]
assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
module = types.ModuleType("r21_input_observation_cli_verified_v1")
module.__file__ = data["entry"]["path"]
module.__package__ = ""
sys.modules[module.__name__] = module
exec(compile(encoded, module.__file__, "exec"), module.__dict__)
original_import = builtins.__import__
def controlled_import(name, *args, **kwargs):
    assert name.split(".")[0] not in module.FORBIDDEN, name
    return original_import(name, *args, **kwargs)
builtins.__import__ = controlled_import
root = module.WSI_ROOT
actual_probe = module._probe_path
for operation in ("lstat", "stat", "open"):
    original = getattr(os, operation)
    def guarded(path, *args, _original=original, **kwargs):
        assert not isinstance(path, (str, os.PathLike)) or not str(path).startswith(root), "REAL WSI PROBE FORBIDDEN"
        return _original(path, *args, **kwargs)
    setattr(os, operation, guarded)
module.OBSERVER_REVIEW_SHA = data["receipt_sha"]  # UNIT synthetic receipt, not real authority.
def probe(path):
    fp = [1,1,32768,1,1,1]
    return ({"lexical_path":path,"lexical_type":"regular","resolved_path":"/synthetic/resolved",
             "resolved_type":"regular","size_bytes":1,"probe_coherent":True,"status":"observed_regular",
             "error_operation":None,"error_errno":None,"error_message":None},
            {"lexical_before":fp[:],"target_before":fp[:],"lexical_after":fp[:],"target_after":fp[:],
             "resolved_before":"/synthetic/resolved","resolved_after":"/synthetic/resolved",
             "errors":[],"coherent":True},0)
module._probe_path = probe  # UNIT metadata seam; no prescribed-root IO.
original_basis = module._load_basis
def guarded_basis(context, spec):
    original_basis(context, spec)
    def no_io(*args, **kwargs):
        raise AssertionError("Frozen observer attempted primitive IO")
    for name in ("classic_vit224_schema", "observe_png", "observe_safetensors",
                 "compare_vit224_schema", "inventory_targets"):
        function = getattr(context.observer, name)
        def guarded_call(*args, _function=function, **kwargs):
            saved = {key:getattr(os,key) for key in ("open","read","lstat","stat")}
            saved_open, saved_resolve = builtins.open, pathlib.Path.resolve
            try:
                for key in saved:
                    setattr(os,key,no_io)
                builtins.open = no_io
                pathlib.Path.resolve = no_io
                return _function(*args, **kwargs)
            finally:
                for key, value in saved.items():
                    setattr(os,key,value)
                builtins.open, pathlib.Path.resolve = saved_open, saved_resolve
        setattr(context.observer, name, guarded_call)
module._load_basis = guarded_basis
case = data.get("repair_case")
if case == "close_history":
    original_read, original_close = module._Context.read, os.close
    tracked, fired = [], []
    def read(context, fd, amount, record, eof=False):
        if (record["role"],record["phase"]) == ("model_config","initial") and not tracked:
            tracked.append(fd)
        return original_read(context,fd,amount,record,eof)
    def close(fd):
        original_close(fd)
        if tracked and fd == tracked[0] and not fired:
            fired.append(fd)
            raise OSError(5,"isolated initial config close after actual close")
    module._Context.read, os.close = read, close
elif case == "inventory_eintr":
    original_lstat = os.lstat
    calls, target_calls = [], []
    def interrupted(path,*args,**kwargs):
        if str(path) == data["probe_leaf"]:
            calls.append(path)
            raise InterruptedError(4,"isolated actual metadata EINTR")
        return original_lstat(path,*args,**kwargs)
    def inventory(path):
        target_calls.append(path)
        return actual_probe(data["probe_leaf"])
    os.lstat, module._probe_path = interrupted, inventory
result = module.run(data["spec"], spec_sha256=data["spec_sha"], entry_binding=data["entry"])
assert module.OBSERVER_NAME not in sys.modules
assert result["packet"] is not None
assert result["exit_code"] == data.get("expected_exit", 2)
if result["exit_code"] == 2:
    assert result["packet"]["inventory_initial"]["state"] == "complete"
    assert result["packet"]["stages"][4]["result"]["safetensors_observation"]["tensor_count"] == 1
if case == "close_history":
    assert fired and result["exit_code"] == 1
    record = result["packet"]["asset_records"][1]
    assert record["custody_state"] == "failed"
    assert all(check["expected_match"] is True for check in record["whole_file_checks"])
elif case == "inventory_eintr":
    assert len(calls) == len(target_calls) == 1
    assert result["packet"]["inventory_initial"]["state"] == "partial"
    assert result["packet"]["inventory_final"]["state"] == "not_started"
    assert not [r for r in result["packet"]["logical_io"]["reads"] if r["role"] in module.ASSETS]
assert not any(n == r or n.startswith(r+".") for n in sys.modules for r in module.FORBIDDEN)
assert json.loads(result["stdout_bytes"]) == result["packet"]
del sys.modules[module.__name__]
sys.stdout.buffer.write(result["stdout_bytes"])
'''


def test_verified_exact_bytes_hostile_cwd_pythonpath_bytecode(cli, tmp_path):
    fixture = generated_controls(cli, tmp_path / "owned")
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    for name in ("prosper", "numpy", "torch", "PIL", "safetensors", "timm", "openslide", "pandas"):
        package = hostile / name
        package.mkdir()
        (package / "__init__.py").write_text("raise RuntimeError('HOSTILE IMPORT')\n")
    (hostile / "sitecustomize.py").write_text("raise RuntimeError('HOSTILE SITE')\n")
    # This stale source/bytecode is never selected by captured-byte execution.
    (hostile / "r2_1_hoptimus_input_observation_cli.py").write_text("raise RuntimeError('STALE SOURCE')\n")
    (hostile / "r2_1_hoptimus_input_observation_cli.pyc").write_bytes(b"unverified bytecode")
    data = {"source": base64.b64encode(SOURCE_BYTES).decode(), "source_sha": sha(SOURCE_BYTES),
            "entry": fixture["entry"], "receipt_sha": fixture["observer_review_sha"],
            "spec": str(fixture["path"]), "spec_sha": sha(fixture["path"].read_bytes())}
    env = dict(os.environ, PYTHONPATH=str(hostile), PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.run([INTERPRETER, "-I", "-S", "-B", "-c", ISOLATED],
                             input=dump(data), capture_output=True, cwd=hostile, env=env, timeout=60)
    assert process.returncode == 0, process.stderr.decode()
    packet = json.loads(process.stdout)
    assert packet["exit_code"] == 2 and not packet["model_use_authorized"]
    assert not list(fixture["root"].rglob("__pycache__"))


@pytest.mark.parametrize("role", ["entry", "spec", "observer", "metadata", "template",
                                  "model_config", "checkpoint", "selected_wsi"])
def test_final_source_control_and_asset_mutations(cli, tmp_path, monkeypatch, role):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)
    path = (fixture["entry"]["path"] if role == "entry" else str(fixture["path"]) if role == "spec" else
            fixture["spec"]["observer"]["path"] if role == "observer" else
            fixture["spec"]["controls"][role]["path"] if role in cli.CONTROLS else
            fixture["wsi_path"] if role == "selected_wsi" else fixture["spec"]["assets"][role]["path"])
    original = cli._final_checks

    def mutation(context):
        target = Path(path)
        encoded = target.read_bytes()
        target.write_bytes(encoded[:-1] + bytes([encoded[-1] ^ 1]))
        original(context)

    monkeypatch.setattr(cli, "_final_checks", mutation)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 1
    assert role in stage(packet, "final_checks")["result"]["failed_roles"]
    record = (packet["entry_binding"] if role == "entry" else packet["observer_binding"] if role == "observer" else
              next(r for r in packet["control_bindings"] if r["role"] == role) if role in ("spec", *cli.CONTROLS) else
              next(r for r in packet["asset_records"] if r["role"] == role))
    assert record["custody_state"] == "invalidated"
    assert record["whole_file_checks"][0]["observed_sha256"] is not None


@pytest.mark.parametrize("variant", ["unselected_io", "selected_missing", "changed_bookend", "changed_final"])
def test_inventory_negative_unexpected_and_mutation_precedence(cli, tmp_path, monkeypatch, variant):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)
    visits = {}

    def probe(path):
        row, book, outcome = synthetic_probe(cli, path)
        sid = path.rsplit("/", 1)[1][:-4]
        visits[sid] = visits.get(sid, 0) + 1
        if variant == "selected_missing" and sid == "10003":
            row.update(lexical_type="absent", resolved_path=None, resolved_type=None, size_bytes=None,
                       status="missing_lexical", error_operation="lstat", error_errno=errno.ENOENT,
                       error_message="missing")
            outcome = 2
        if variant == "unselected_io" and sid == "16683":
            row.update(status="io_error", error_operation="stat", error_errno=errno.EIO, error_message="io")
            outcome = 1
        if variant == "changed_bookend" and sid == "16683":
            row.update(status="changed_during_probe", probe_coherent=False,
                       error_operation="compare_bookends", error_message="change")
            book["coherent"] = False
            book["target_after"][3] = 2
            outcome = 1
        if variant == "changed_final" and sid == "10003" and visits[sid] == 2:
            row["size_bytes"] = 2
            book["target_after"][3] = book["target_before"][3] = 2
        return row, book, outcome

    monkeypatch.setattr(cli, "_probe_path", probe)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == (2 if variant == "selected_missing" else 1)
    if variant == "changed_final":
        assert packet["inventory_comparison"]["changed_ids"] == ["10003"]
        assert packet["asset_records"][3]["custody_state"] == "invalidated"
    else:
        assert stage(packet, "checkpoint_full")["state"] == "blocked"


@pytest.mark.parametrize("chunk_name,body,blocking", [
    (b"ABCD", b"", True), (b"acTL", struct.pack(">II", 1, 0), True),
    (b"gAMA", struct.pack(">I", 45455), False), (b"eXIf", b"not-decoded-tiff", False),
    (b"iCCP", b"fixture\x00\x00not-decoded-compressed-data", False),
    (b"PLTE", b"\x00\x00\x00", False),
])
def test_png_unsupported_vs_ancillary_preserves_frozen_output(cli, tmp_path, monkeypatch,
                                                            chunk_name, body, blocking):
    encoded_chunk = struct.pack(">I", len(body)) + chunk_name + body + struct.pack(
        ">I", binascii.crc32(chunk_name + body) & 0xffffffff)
    image = png()
    image = image[:33] + encoded_chunk + image[33:]
    fixture = generated_controls(cli, tmp_path / "owned", template=image)
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == (2 if blocking else 0)
    frozen = types.ModuleType("frozen_png")
    exec(compile(OBSERVER_BYTES, str(OBSERVER), "exec"), frozen.__dict__)
    assert stage(packet, "template")["result"] == frozen.observe_png(image, declared_size=len(image))
    assert stage(packet, "checkpoint_full")["state"] == ("blocked" if blocking else "complete")


@pytest.mark.parametrize("role", (
    "observer_review", "cpu_complete", "cpu_input_bindings", "cpu_review",
    "cpu_spec", "stage1b_complete", "stage1b_review"))
def test_closed_control_json_keys(cli, role):
    names = cli.JSON_KEYS[role].split()
    record = {name: None for name in names}
    assert cli._fields(record, names) is record
    record["unreleased"] = None
    with pytest.raises(cli._Fault):
        cli._fields(record, names)


def test_population_join_support_and_source_identity_failures(cli, tmp_path):
    fixture = generated_controls(cli, tmp_path / "owned")
    rows = {role: cli._csv_bytes(Path(fixture["spec"]["controls"][role]["path"]).read_bytes(), role)
            for role in ("metadata", "targets")}
    meta, targets = cli._population(rows["metadata"], rows["targets"])
    assert len(meta) == 2921 and len(targets) == 2778
    for field, replacement in (("metadata_index", "1"), ("patient_id", "wrong"),
                               ("diagnostic_group", "MISC"), ("feature_rows", "0")):
        changed = copy.deepcopy(rows["targets"])
        changed[0][field] = replacement
        with pytest.raises(cli._Fault):
            cli._population(rows["metadata"], changed)
    changed = copy.deepcopy(rows["metadata"])
    changed[1]["Slide.Scan.File"] = changed[0]["Slide.Scan.File"]
    with pytest.raises(cli._Fault):
        cli._population(changed, rows["targets"])


def test_budget_control_aliases_are_counted_per_role(cli, tmp_path):
    spec = json.loads((SOURCE.parents[2] / "examples/r2_1_hoptimus_input_observation.json").read_bytes())
    # UNIT declared controls: same path/hash, still five independently sized roles exceed32MiB.
    for role in ("metadata", "targets", "cpu_spec", "cpu_review", "cpu_complete"):
        spec["controls"][role].update(path="/synthetic/shared", size=8388608, sha256="0" * 64)
    with pytest.raises(cli._Fault, match="Aggregate"):
        cli._declaration(spec, str(tmp_path))
    spec = json.loads((SOURCE.parents[2] / "examples/r2_1_hoptimus_input_observation.json").read_bytes())
    for role in ("checkpoint", "selected_wsi"):
        spec["assets"][role]["size"] = 6442450944
    with pytest.raises(cli._Fault, match="planned"):
        cli._declaration(spec, str(tmp_path))


def test_coherent_failed_asset_open_availability2(cli, tmp_path, monkeypatch):
    path = tmp_path / "denied"
    path.write_bytes(b"x")
    context = cli._Context()
    record = cli._binding("model_config", str(path), 1, sha(b"x"))
    monkeypatch.setattr(cli.os, "open", lambda *args: (_ for _ in ()).throw(OSError(errno.EACCES, "denied")))
    with pytest.raises(cli._Fault) as caught:
        cli._Reader(context, record, "initial", 65536)
    assert caught.value.exit_code == 2 and caught.value.error_errno == errno.EACCES
    assert record["initial_descriptor"]["coherent"] is True


def test_observer_registration_substitution_is_not_removed(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch)
    foreign = types.ModuleType(cli.OBSERVER_NAME)
    initial_assets = cli._initial_assets

    def swap(context):
        initial_assets(context)
        sys.modules[cli.OBSERVER_NAME] = foreign

    monkeypatch.setattr(cli, "_initial_assets", swap)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 1
    assert sys.modules[cli.OBSERVER_NAME] is foreign
    del sys.modules[cli.OBSERVER_NAME]


def test_keyboard_interruption_no_new_payload_reads(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch)
    monkeypatch.setattr(cli, "_probe_path", lambda path: (_ for _ in ()).throw(KeyboardInterrupt()))
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 1 and packet["inventory_initial"]["state"] == "partial"
    assert packet["inventory_final"]["state"] == "not_started"
    assert stage(packet, "final_checks")["state"] == "blocked"
    assert not [r for r in packet["logical_io"]["reads"] if r["role"] in cli.ASSETS]


def test_stderr_failure_cannot_preserve_normal_exit(cli, monkeypatch):
    out, err = Output(), Output(flush_error=True)
    monkeypatch.setattr(cli.sys, "stdout", out)
    monkeypatch.setattr(cli.sys, "stderr", err)
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: {
        "packet": {"exit_code": 2}, "stdout_bytes": b"old negative packet\n",
        "stderr_bytes": b"diagnostic\n", "exit_code": 2})
    assert cli.main(["--spec-sha256", "0" * 64, "--spec", "x"], entry_binding={}) == 1
    assert not out.data


def test_captured_source_not_reopened_for_execution(cli, tmp_path):
    fixture = generated_controls(cli, tmp_path / "owned")
    malicious = b"raise RuntimeError('PATHNAME EXECUTED INSTEAD OF CAPTURED BYTES')\n#"
    malicious += b"#" * (len(SOURCE_BYTES) - len(malicious))
    Path(fixture["entry"]["path"]).write_bytes(malicious)
    data = {"source": base64.b64encode(SOURCE_BYTES).decode(), "source_sha": sha(SOURCE_BYTES),
            "entry": fixture["entry"], "receipt_sha": fixture["observer_review_sha"],
            "spec": str(fixture["path"]), "spec_sha": sha(fixture["path"].read_bytes()),
            "expected_exit": 1}
    process = subprocess.run([INTERPRETER, "-I", "-S", "-B", "-c", ISOLATED],
                             input=dump(data), capture_output=True, cwd=fixture["root"], timeout=60)
    assert process.returncode == 0, process.stderr.decode()
    packet = json.loads(process.stdout)
    assert any(i["code"] == "digest_mismatch" for i in packet["issues"])
    assert packet["run_id"] is None


def test_real_in_progress_bookend_preserved_on_unrelated_runtimeerror(cli, tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_bytes(b"x")
    monkeypatch.setattr(cli.Path, "resolve", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unrelated")))
    with pytest.raises(RuntimeError) as caught:
        cli._probe_path(str(path))
    book = caught.value.partial_bookends
    assert book["lexical_before"] is not None and book["coherent"] is None
    assert book["errors"][0]["error_type"] == "RuntimeError"
    assert book["errors"][0]["errno"] is None


def test_interrupted_small_acquisition_closes_descriptor_and_stops_reads(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch)
    read = cli._Context.read
    closed = []
    original_close = cli.os.close

    def interrupted(context, fd, amount, record, eof=False):
        if record["role"] == "template":
            raise KeyboardInterrupt()
        return read(context, fd, amount, record, eof)

    def close(fd):
        closed.append(fd)
        return original_close(fd)

    monkeypatch.setattr(cli._Context, "read", interrupted)
    monkeypatch.setattr(cli.os, "close", close)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 1
    assert not [r for r in packet["logical_io"]["reads"] if r["role"] in ("model_config", "checkpoint", "selected_wsi")]
    assert len(closed) == 14


def test_stream_close_failure_keeps_primary_and_never_promotes(cli, tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_bytes(b"x")
    context = cli._Context()
    record = cli._binding("model_config", str(path), 1, sha(b"x"))
    reader = cli._Reader(context, record, "initial", 65536)
    reader.full()
    close = cli.os.close

    def failure(fd):
        close(fd)
        raise OSError(errno.EIO, "controlled late close failure")

    monkeypatch.setattr(cli.os, "close", failure)
    with pytest.raises(OSError):
        reader.finish()
    assert record["custody_state"] == "failed"


def test_caller_owns_only_observer_registration(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch)
    foreign = types.ModuleType(cli.OBSERVER_NAME)
    sys.modules[cli.OBSERVER_NAME] = foreign
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 1 and any(i["code"] == "observer_module_collision" for i in packet["issues"])
    assert sys.modules[NAME] is cli and sys.modules[cli.OBSERVER_NAME] is foreign
    del sys.modules[cli.OBSERVER_NAME]


def test_run_results_have_no_nested_aliases(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch)
    first = execute(cli, fixture)
    first["packet"]["inventory_initial"]["frozen_result"]["rows"][0]["slide_id"] = "changed"
    first["packet"]["stages"][2]["result"]["ihdr"]["width"] = -1
    next_result = execute(cli, fixture)
    assert next_result["packet"]["inventory_initial"]["frozen_result"]["rows"][0]["slide_id"] == "10003"
    assert next_result["packet"]["stages"][2]["result"]["ihdr"]["width"] == 1


def test_checkpoint_original_region_mutation_negative(cli, tmp_path):
    encoded = safefile()
    path = tmp_path / "checkpoint"
    path.write_bytes(encoded)
    context = cli._Context()
    record = cli._binding("checkpoint", str(path), len(encoded), sha(encoded))
    record.update(regions=[], current_resolved_path=None)
    reader = cli._Reader(context, record, "initial", 1048576)
    prefix = cli._record_region(context, reader, "prefix8", 8)
    header = cli._record_region(context, reader, "header", struct.unpack("<Q", prefix)[0])
    reader.full(prefix + header)
    reader.finish()
    changed = bytearray(encoded)
    changed[10] ^= 1
    path.write_bytes(changed)
    with pytest.raises(cli._Fault) as caught:
        cli._final_asset(context, record)
    assert caught.value.code == "region_changed"
    assert record["regions"][0]["byte_equal"] is True
    assert record["regions"][1]["byte_equal"] is False
    assert record["whole_file_checks"][1]["observed_sha256"] is None
    assert record["custody_state"] == "invalidated"


def test_declaration_and_capture_are_not_admission_even_exit0(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 0 and packet["model_use_authorized"] is False
    for name in ("inventory_initial", "inventory_final"):
        assert packet[name]["frozen_result"]["content_admission"] == "not_performed"
        assert all(r["content_admission"] == "not_performed" for r in packet[name]["frozen_result"]["rows"])
    assert stage(packet, "config")["result"]["comparison_state"] == "not_performed"
    assert "structure_and_schema_not_model_identity" in packet["limitations"]


def _close_specific_if_open(fd, fstat, close):
    try:
        fstat(fd)
    except OSError as exc:
        assert exc.errno == errno.EBADF
        return False
    close(fd)
    return True


def test_ioc_c1_initial_config_close_history_survives_final_match(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)
    original_read, original_close = cli._Context.read, cli.os.close
    selected_fd = None
    fired = False

    def read(context, fd, amount, record, eof=False):
        nonlocal selected_fd
        if (record["role"], record["phase"]) == ("model_config", "initial"):
            selected_fd = fd
        return original_read(context, fd, amount, record, eof)

    def close(fd):
        nonlocal fired
        original_close(fd)
        if fd == selected_fd and not fired:
            fired = True
            raise OSError(errno.EIO, "IOC-C1 initial config close failed after actual close")

    monkeypatch.setattr(cli._Context, "read", read)
    monkeypatch.setattr(cli.os, "close", close)
    packet = check_packet(execute(cli, fixture))
    assert fired and packet["exit_code"] == 1
    config = packet["asset_records"][1]
    assert stage(packet, "config")["state"] == "failed"
    assert all(c["expected_match"] is True for c in config["whole_file_checks"])
    assert config["custody_state"] in ("failed", "invalidated")


@pytest.mark.parametrize("role,phase", [("selected_wsi", "initial"), ("checkpoint", "final")])
def test_ioc_c2_original_stream_interrupt_closes_specific_fd(cli, tmp_path, monkeypatch, role, phase):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)
    original_read, original_fstat, original_close = cli._Context.read, cli.os.fstat, cli.os.close
    owned = []

    def interrupted(context, fd, amount, record, eof=False):
        if (record["role"], record["phase"]) == (role, phase) and not owned:
            owned.append(fd)
            raise KeyboardInterrupt("IOC-C2 original stream counterexample")
        return original_read(context, fd, amount, record, eof)

    monkeypatch.setattr(cli._Context, "read", interrupted)
    try:
        packet = check_packet(execute(cli, fixture))
        assert packet["exit_code"] == 1 and len(owned) == 1
        was_open = _close_specific_if_open(owned[0], original_fstat, original_close)
        assert not was_open, "An actual specifically owned descriptor leaked"
    finally:
        for fd in owned:
            _close_specific_if_open(fd, original_fstat, original_close)


def test_ioc_c2_original_finish_fstat_interrupt_closes_once(cli, tmp_path, monkeypatch):
    path = tmp_path / "owned_file"
    path.write_bytes(b"tiny")
    context = cli._Context()
    record = cli._binding("checkpoint", str(path), 4, sha(b"tiny"))
    record.update(regions=[], current_resolved_path=None)
    reader = cli._Reader(context, record, "initial", 1048576)
    reader.full()
    fd = reader.fd
    original_fstat, original_close = cli.os.fstat, cli.os.close
    attempts = []

    def interrupted(value):
        assert value == fd
        raise KeyboardInterrupt("IOC-C2 finish fstat counterexample")

    def close(value):
        attempts.append(value)
        return original_close(value)

    monkeypatch.setattr(cli.os, "fstat", interrupted)
    monkeypatch.setattr(cli.os, "close", close)
    try:
        with pytest.raises(KeyboardInterrupt):
            reader.finish()
        was_open = _close_specific_if_open(fd, original_fstat, original_close)
        assert not was_open, "finish lost ownership before its close attempt"
        assert attempts == [fd]
        assert reader.fd is None
    finally:
        _close_specific_if_open(fd, original_fstat, original_close)


def test_ioc_c3_original_inventory_eintr_is_not_normalized(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    actual_probe = cli._probe_path
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)
    leaf = tmp_path / "probe_leaf"
    leaf.write_bytes(b"x")
    original_lstat, original_probe = cli.os.lstat, cli._probe_path
    syscall_calls, target_calls = [], []

    def interrupted(path, *args, **kwargs):
        if str(path) == str(leaf):
            syscall_calls.append(path)
            raise InterruptedError(errno.EINTR, "IOC-C3 lstat interruption")
        return original_lstat(path, *args, **kwargs)

    def probe(path):
        target_calls.append(path)
        if len(target_calls) == 1:
            return actual_probe(str(leaf))
        return original_probe(path)

    # A labelled target-to-owned-path seam; actual _bookend/_probe_path execute.
    monkeypatch.setattr(cli.os, "lstat", interrupted)
    monkeypatch.setattr(cli, "_probe_path", probe)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 1
    assert len(syscall_calls) == len(target_calls) == 1
    assert packet["inventory_initial"]["state"] == "partial"
    assert packet["inventory_final"]["state"] == "not_started"
    assert not [r for r in packet["logical_io"]["reads"] if r["role"] in cli.ASSETS]


def test_ioc_c4_original_negative_to_present_requires_two_bookends(cli, tmp_path, monkeypatch):
    path = tmp_path / "present"
    path.write_bytes(b"tiny")
    context = cli._Context()
    record = cli._binding("template", str(path), 4, sha(b"tiny"))
    original_bookend = cli._bookend
    calls = []

    def absent_first(target, which):
        calls.append(which)
        if len(calls) == 1:
            return {"lexical": None, "target": None, "resolved": None, "errors": [{
                "bookend": which, "operation": "lstat", "error_type": "FileNotFoundError",
                "errno": errno.ENOENT, "message": "IOC-C4 first negative",
                "message_truncated": False}]}
        return original_bookend(target, which)

    monkeypatch.setattr(cli, "_bookend", absent_first)
    with pytest.raises(cli._Fault) as caught:
        cli._Reader(context, record, "initial", 1048576)
    assert caught.value.exit_code == 1
    assert calls == ["before", "after"]
    assert record["initial_descriptor"]["coherent"] is False
    assert record["initial_descriptor"]["target_after"] is not None


@pytest.mark.parametrize("role", ["entry", "spec", "observer", "metadata", "template",
                                  "model_config", "checkpoint", "selected_wsi"])
@pytest.mark.parametrize("history", ["failed", "invalidated"])
def test_ioc_c1_monotone_history_all_phase_readers(cli, tmp_path, role, history):
    path = tmp_path / "asset"
    path.write_bytes(b"tiny")
    context = cli._Context()
    record = cli._binding(role, str(path), 4, sha(b"tiny"))
    record.update(regions=[], current_resolved_path=None)
    with cli._Reader(context, record, "initial", 1048576) as reader:
        reader.full()
    record["custody_state"] = history
    if role in ("checkpoint", "selected_wsi"):
        cli._final_asset(context, record)
    else:
        cli._small(context, record, 1048576, "final")
    assert all(c["expected_match"] is True for c in record["whole_file_checks"])
    assert record["custody_state"] == history


@pytest.mark.parametrize("role,phase,call", [
    ("template", "initial", 1), ("model_config", "final", 1),
    ("checkpoint", "initial", 1), ("checkpoint", "initial", 2),
    ("checkpoint", "initial", 3), ("checkpoint", "final", 2),
    ("checkpoint", "final", 3), ("selected_wsi", "final", 1),
])
def test_ioc_c2_phase_region_full_interrupt_variants(cli, tmp_path, monkeypatch, role, phase, call):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)
    original_read, original_fstat, original_close = cli._Context.read, cli.os.fstat, cli.os.close
    owned, seen, close_attempts = [], [], []

    def interrupted(context, fd, amount, record, eof=False):
        if (record["role"], record["phase"]) == (role, phase):
            seen.append(record.copy())
            if len(seen) == call:
                owned.append(fd)
                raise KeyboardInterrupt("IOC-C2 phase read variant")
        return original_read(context, fd, amount, record, eof)

    def close(fd):
        if owned and fd == owned[0]:
            close_attempts.append(fd)
        return original_close(fd)

    monkeypatch.setattr(cli._Context, "read", interrupted)
    monkeypatch.setattr(cli.os, "close", close)
    try:
        packet = check_packet(execute(cli, fixture))
        assert packet["exit_code"] == 1 and len(owned) == 1
        assert not _close_specific_if_open(owned[0], original_fstat, original_close)
        assert close_attempts == owned
        affected = next(r for r in packet["asset_records"] if r["role"] == role)
        assert affected["custody_state"] in ("failed", "invalidated")
        if role == "checkpoint" and call == 2:
            region = affected["regions"][1]
            if phase == "initial":
                assert region["capture_state"] == "failed" and region["sha256"] is None
            else:
                assert region["final_state"] == "failed" and region["final_sha256"] is None
        if role in ("checkpoint", "selected_wsi") and call == 3:
            check = affected["whole_file_checks"][0 if phase == "initial" else 1]
            assert check["state"] == "failed" and check["observed_sha256"] is None
    finally:
        for fd in owned:
            _close_specific_if_open(fd, original_fstat, original_close)


@pytest.mark.parametrize("close_error", [
    OSError(errno.EIO, "close failed"), KeyboardInterrupt("close interrupted"),
    SystemExit("cleanup termination"),
])
def test_ioc_c2_primary_preserved_cleanup_attempt_once(cli, tmp_path, monkeypatch, close_error):
    path = tmp_path / "owned"
    path.write_bytes(b"x")
    context = cli._Context()
    record = cli._binding("checkpoint", str(path), 1, sha(b"x"))
    record.update(regions=[], current_resolved_path=None)
    original_fstat, original_close = cli.os.fstat, cli.os.close
    reader = cli._Reader(context, record, "initial", 1048576)
    fd = reader.fd
    attempts = []
    primary = cli._Fault("primary_read_failure", "Original primary failure", "unexpected_io")

    def close(value):
        attempts.append(value)
        raise close_error

    monkeypatch.setattr(cli.os, "close", close)
    try:
        with pytest.raises(cli._Fault) as caught:
            with reader:
                raise primary
        assert caught.value is primary
        assert attempts == [fd] and reader.fd is None
        assert record["custody_state"] == "failed"
        assert context.packet["issues"][0]["code"] == "primary_read_failure"
        assert context.packet["secondary_errors"][0]["primary_issue_index"] == 0
        assert context.interrupted == cli._interruption(close_error)
        # A failed close is not claimed closed and is never retried by the caller.
        assert _close_specific_if_open(fd, original_fstat, original_close) is True
        reader.finish()
        assert attempts == [fd]
    finally:
        _close_specific_if_open(fd, original_fstat, original_close)


@pytest.mark.parametrize("phase", ["initial", "final"])
@pytest.mark.parametrize("operation", ["lstat", "resolve", "stat"])
def test_ioc_c3_eintr_actual_probe_operations_both_passes(cli, tmp_path, monkeypatch, phase, operation):
    fixture = generated_controls(cli, tmp_path / "owned")
    actual_probe = cli._probe_path
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)
    leaf = tmp_path / "probe_leaf"
    leaf.write_bytes(b"x")
    originals = {"lstat": cli.os.lstat, "stat": cli.os.stat, "resolve": cli.Path.resolve}
    calls, targets = [], []
    armed = False

    def failure(value, *args, **kwargs):
        if armed and str(value) == str(leaf):
            calls.append(value)
            raise InterruptedError(errno.EINTR, "IOC-C3 actual " + operation)
        return originals[operation](value, *args, **kwargs)

    if operation == "resolve":
        monkeypatch.setattr(cli.Path, operation, failure)
    else:
        monkeypatch.setattr(cli.os, operation, failure)

    def probe(path):
        nonlocal armed
        targets.append(path)
        if len(targets) == (1 if phase == "initial" else 2779):
            armed = True
            return actual_probe(str(leaf))
        return synthetic_probe(cli, path)

    monkeypatch.setattr(cli, "_probe_path", probe)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 1
    assert len(calls) == 1 and len(targets) == (1 if phase == "initial" else 2779)
    inventory = packet["inventory_initial" if phase == "initial" else "inventory_final"]
    assert inventory["state"] == "partial" and inventory["visited_ids"] == []
    assert inventory["in_progress_id"] == "10003"
    assert inventory["unvisited_ids"] == fixture["ids"][1:]
    assert inventory["bookends"]["10003"]["errors"][0]["errno"] == errno.EINTR
    assert inventory["bookends"]["10003"]["coherent"] is None
    if phase == "initial":
        assert packet["inventory_final"]["state"] == "not_started"
        assert not [r for r in packet["logical_io"]["reads"] if r["role"] in cli.ASSETS]
    else:
        assert packet["inventory_initial"]["state"] == "complete"


@pytest.mark.parametrize("number", [errno.ENOENT, errno.EACCES, errno.EPERM, errno.ENOTDIR, errno.ELOOP, errno.EIO])
@pytest.mark.parametrize("changed", [False, True])
def test_ioc_c4_coherent_negative_and_negative_to_positive(cli, tmp_path, monkeypatch, number, changed):
    path = tmp_path / "owned"
    path.write_bytes(b"x")
    context = cli._Context()
    record = cli._binding("template", str(path), 1, sha(b"x"))
    original = cli._bookend
    calls = []

    def negative(target, which):
        calls.append(which)
        if changed and which == "after":
            return original(target, which)
        return {"lexical": None, "target": None, "resolved": None, "errors": [{
            "bookend": which, "operation": "lstat", "error_type": "OSError",
            "errno": number, "message": "Controlled negative metadata", "message_truncated": False}]}

    monkeypatch.setattr(cli, "_bookend", negative)
    monkeypatch.setattr(cli.os, "open", lambda *args: pytest.fail("Negative acquisition must not open"))
    with pytest.raises(cli._Fault) as caught:
        cli._Reader(context, record, "initial", 1048576)
    assert calls == ["before", "after"]
    assert record["initial_descriptor"]["coherent"] is (not changed)
    assert caught.value.exit_code == (1 if changed or number == errno.EIO else 2)
    assert record["initial_descriptor"]["opened"] is None


@pytest.mark.parametrize("changed", [False, True])
def test_ioc_c4_real_nonregular_bookends_and_type_change(cli, tmp_path, monkeypatch, changed):
    directory = tmp_path / "directory"
    directory.mkdir()
    context = cli._Context()
    record = cli._binding("template", str(directory), 1, "0" * 64)
    original = cli._bookend
    calls = []

    def bookend(path, which):
        calls.append(which)
        if changed and which == "after":
            directory.rmdir()
            directory.write_bytes(b"x")
        return original(path, which)

    monkeypatch.setattr(cli, "_bookend", bookend)
    monkeypatch.setattr(cli.os, "open", lambda *args: pytest.fail("Nonregular pre-open must not open"))
    with pytest.raises(cli._Fault) as caught:
        cli._Reader(context, record, "initial", 1048576)
    assert calls == ["before", "after"]
    assert caught.value.exit_code == (1 if changed else 2)
    assert record["initial_descriptor"]["coherent"] is (not changed)


def test_ioc_c2_cleanup_interrupt_stops_later_whole_caller_reads(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)
    original_read, original_close = cli._Context.read, cli.os.close
    primary = OSError(errno.EIO, "original template read failure")
    owned, close_calls = [], []

    def read(context, fd, amount, record, eof=False):
        if record["role"] == "template":
            owned.append(fd)
            raise primary
        return original_read(context, fd, amount, record, eof)

    def close(fd):
        original_close(fd)
        if owned and fd == owned[0]:
            close_calls.append(fd)
            raise KeyboardInterrupt("cleanup interrupt after actual close")

    monkeypatch.setattr(cli._Context, "read", read)
    monkeypatch.setattr(cli.os, "close", close)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 1 and close_calls == owned
    assert any("original template read failure" in e["message"] for e in packet["issues"])
    assert any("cleanup interrupt" in e["message"] for e in packet["secondary_errors"])
    assert not [r for r in packet["logical_io"]["reads"] if r["role"] in ("model_config", "checkpoint", "selected_wsi")]
    assert packet["inventory_final"]["state"] == "not_started"


def test_ioc_c1_whole_caller_large_initial_close_failure_never_promoted(cli, tmp_path, monkeypatch):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch, schema_match=True)
    original_read, original_close = cli._Context.read, cli.os.close
    owned, fired = [], []

    def read(context, fd, amount, record, eof=False):
        if (record["role"], record["phase"]) == ("checkpoint", "initial") and not owned:
            owned.append(fd)
        return original_read(context, fd, amount, record, eof)

    def close(fd):
        original_close(fd)
        if owned and fd == owned[0] and not fired:
            fired.append(fd)
            raise OSError(errno.EIO, "IOC-C1 checkpoint close after actual close")

    monkeypatch.setattr(cli._Context, "read", read)
    monkeypatch.setattr(cli.os, "close", close)
    packet = check_packet(execute(cli, fixture))
    assert packet["exit_code"] == 1 and fired == owned
    checkpoint = packet["asset_records"][2]
    assert all(c["expected_match"] is True for c in checkpoint["whole_file_checks"])
    assert checkpoint["custody_state"] == "failed"


def test_ioc_c2_constructor_interrupt_closes_acquired_fd(cli, tmp_path, monkeypatch):
    path = tmp_path / "owned"
    path.write_bytes(b"x")
    context = cli._Context()
    record = cli._binding("checkpoint", str(path), 1, sha(b"x"))
    original_open, original_close, original_fstat = cli.os.open, cli.os.close, cli.os.fstat
    owned, attempts = [], []

    def opened(target, flags):
        fd = original_open(target, flags)
        owned.append(fd)
        return fd

    def interrupted(fd):
        assert fd == owned[0]
        raise KeyboardInterrupt("IOC-C2 constructor immediate fstat")

    def close(fd):
        attempts.append(fd)
        return original_close(fd)

    monkeypatch.setattr(cli.os, "open", opened)
    monkeypatch.setattr(cli.os, "fstat", interrupted)
    monkeypatch.setattr(cli.os, "close", close)
    try:
        with pytest.raises(KeyboardInterrupt):
            cli._Reader(context, record, "initial", 1048576)
        assert attempts == owned and context.interrupted
        assert not _close_specific_if_open(owned[0], original_fstat, original_close)
    finally:
        for fd in owned:
            _close_specific_if_open(fd, original_fstat, original_close)


def test_ioc_c4_after_bookend_unexpected_error_retains_errno(cli, tmp_path, monkeypatch):
    path = tmp_path / "owned"
    path.write_bytes(b"x")
    context = cli._Context()
    record = cli._binding("template", str(path), 1, sha(b"x"))

    def negative(target, which):
        number = errno.ENOENT if which == "before" else errno.EIO
        return {"lexical": None, "target": None, "resolved": None, "errors": [{
            "bookend": which, "operation": "lstat", "error_type": "OSError",
            "errno": number, "message": "Genuine controlled error", "message_truncated": False}]}

    monkeypatch.setattr(cli, "_bookend", negative)
    with pytest.raises(cli._Fault) as caught:
        cli._Reader(context, record, "initial", 1048576)
    assert caught.value.exit_code == 1
    assert caught.value.error_errno == errno.EIO and caught.value.operation == "lstat"
    assert record["initial_descriptor"]["coherent"] is False


@pytest.mark.parametrize("number", [errno.EINTR, None])
def test_ioc_c4_after_bookend_interruption_retains_observed_metadata(cli, tmp_path, monkeypatch, number):
    path = tmp_path / "owned"
    path.write_bytes(b"x")
    context = cli._Context()
    record = cli._binding("template", str(path), 1, sha(b"x"))
    original_lstat, original_resolve = cli.os.lstat, cli.Path.resolve
    calls = []

    def lstat(target):
        if str(target) == str(path):
            calls.append(target)
            if len(calls) == 1:
                raise OSError(errno.EACCES, "Initial inaccessible metadata")
        return original_lstat(target)

    def resolve(self, *args, **kwargs):
        if str(self) == str(path):
            if number is None:
                raise KeyboardInterrupt("After bookend interrupted")
            raise InterruptedError(number, "After bookend interrupted")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(cli.os, "lstat", lstat)
    monkeypatch.setattr(cli.Path, "resolve", resolve)
    with pytest.raises((KeyboardInterrupt, InterruptedError)):
        cli._Reader(context, record, "initial", 1048576)
    assert len(calls) == 2
    assert record["initial_descriptor"]["lexical_after"] is not None
    assert record["initial_descriptor"]["coherent"] is None


@pytest.mark.parametrize("repair_case", ["close_history", "inventory_eintr"])
def test_ioc_repair_fresh_isolated_same_byte_child(cli, tmp_path, repair_case):
    fixture = generated_controls(cli, tmp_path / "owned")
    leaf = tmp_path / "probe_leaf"
    leaf.write_bytes(b"x")
    data = {"source": base64.b64encode(SOURCE_BYTES).decode(), "source_sha": sha(SOURCE_BYTES),
            "entry": fixture["entry"], "receipt_sha": fixture["observer_review_sha"],
            "spec": str(fixture["path"]), "spec_sha": sha(fixture["path"].read_bytes()),
            "expected_exit": 1, "repair_case": repair_case, "probe_leaf": str(leaf)}
    process = subprocess.run([INTERPRETER, "-I", "-S", "-B", "-c", ISOLATED],
                             input=dump(data), capture_output=True, cwd=fixture["root"], timeout=60)
    assert process.returncode == 0, process.stderr.decode()
    packet = json.loads(process.stdout)
    assert packet["exit_code"] == 1 and packet["model_use_authorized"] is False


@pytest.mark.parametrize("operation", ["resolve", "stat"])
@pytest.mark.parametrize("role", ["checkpoint", "metadata"])
def test_ioc_c2_r1_serialized_closing_partial_evidence(cli, tmp_path, monkeypatch, operation, role):
    fixture = generated_controls(cli, tmp_path / "owned")
    install_unit_seams(cli, fixture, monkeypatch, schema_match=False)
    path = (fixture["spec"]["assets"]["checkpoint"]["path"] if role == "checkpoint"
            else fixture["spec"]["controls"]["metadata"]["path"])
    aliases = {path, str(Path(path).resolve(strict=True))}
    original_read, original_close, original_issue = cli._Context.read, cli.os.close, cli._Context.issue
    owner = cli.Path if operation == "resolve" else cli.os
    original_operation = getattr(owner, operation)
    primary = InterruptedError(errno.EINTR, "IOC-C2-R1 closing " + operation)
    final_reads, owned, close_attempts, issues, late_reads = [], [], [], [], []
    armed = False

    def read(context, fd, amount, record, eof=False):
        nonlocal armed
        if armed:
            late_reads.append(record.copy())
        result = original_read(context, fd, amount, record, eof)
        if (record["role"], record["phase"]) == (role, "final"):
            final_reads.append(record.copy())
            if (role == "checkpoint" and len(final_reads) == 2) or (role == "metadata" and eof):
                armed = True
                owned.append(fd)
        return result

    def failure(value, *args, **kwargs):
        if armed and str(value) in aliases:
            raise primary
        return original_operation(value, *args, **kwargs)

    def close(fd):
        if owned and fd == owned[0]:
            close_attempts.append(fd)
        return original_close(fd)

    def issue(context, exc, *args, **kwargs):
        issues.append(exc)
        return original_issue(context, exc, *args, **kwargs)

    monkeypatch.setattr(cli._Context, "read", read)
    monkeypatch.setattr(owner, operation, failure)
    monkeypatch.setattr(cli.os, "close", close)
    monkeypatch.setattr(cli._Context, "issue", issue)
    packet = check_packet(execute(cli, fixture))
    record = (packet["asset_records"][2] if role == "checkpoint" else
              next(r for r in packet["control_bindings"] if r["role"] == role))
    descriptor = record["final_descriptor"]
    observed = primary.observed_bookend
    assert packet["exit_code"] == 1 and close_attempts == owned and len(owned) == 1
    assert primary in issues and any(exc is primary for exc in issues)
    assert descriptor["after_fd"] is not None
    assert record["custody_state"] == "invalidated"
    assert not late_reads and packet["inventory_final"]["state"] == "not_started"
    assert descriptor["coherent"] is None
    assert stage(packet, "checkpoint_header")["result"]["schema_comparison"]["match"] is False
    assert stage(packet, "checkpoint_full")["state"] == "blocked"
    assert observed["lexical"] is not None and len(observed["lexical"]) == 6
    assert descriptor["lexical_after"] == observed["lexical"]
    assert descriptor["target_after"] is None and observed["target"] is None
    assert descriptor["resolved_path"] == record["initial_descriptor"]["resolved_path"]
    if role == "checkpoint":
        assert all(r["byte_equal"] is True for r in record["regions"])
        assert all(c["observed_sha256"] is None for c in record["whole_file_checks"])
    else:
        assert all(c["expected_match"] is True for c in record["whole_file_checks"])


@pytest.mark.parametrize("where", ["fstat", "close"])
def test_ioc_c2_r1_unrelated_exception_attributes_are_not_closing_evidence(cli, tmp_path, monkeypatch, where):
    path = tmp_path / "owned"
    path.write_bytes(b"tiny")
    context = cli._Context()
    record = cli._binding("metadata", str(path), 4, sha(b"tiny"))
    reader = cli._Reader(context, record, "initial", 1048576)
    reader.full()
    original_fstat, original_close = cli.os.fstat, cli.os.close
    fd = reader.fd
    primary = InterruptedError(errno.EINTR, "Unrelated " + where)
    primary.observed_bookend = {"lexical": [9] * 6, "target": [8] * 6, "resolved": "/not-observed"}
    primary.bookend_phase = "after"
    actual_bookend = cli._bookend
    measured = []

    def bookend(target, phase):
        result = actual_bookend(target, phase)
        measured.append(result)
        return result

    def failure(value):
        assert value == fd
        if where == "close":
            original_close(value)
        raise primary

    monkeypatch.setattr(cli, "_bookend", bookend)
    monkeypatch.setattr(cli.os, where, failure)
    try:
        with pytest.raises(InterruptedError) as caught:
            reader.finish()
        assert caught.value is primary
        if where == "fstat":
            assert record["initial_descriptor"]["lexical_after"] is None
            assert record["initial_descriptor"]["target_after"] is None
            assert not measured
        else:
            assert record["initial_descriptor"]["lexical_after"] == measured[0]["lexical"]
            assert record["initial_descriptor"]["target_after"] == measured[0]["target"]
        assert reader.fd is None
    finally:
        _close_specific_if_open(fd, original_fstat, original_close)
