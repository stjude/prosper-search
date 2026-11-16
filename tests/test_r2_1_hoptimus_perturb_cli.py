"""Normal-package and tiny bound-source fixtures, with no image/feature data."""

from dataclasses import asdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from prosper.evaluation import feature_bank_audit as legacy
from prosper.evaluation import r2_1_hoptimus_perturb as kernel
from prosper.evaluation import r2_1_hoptimus_perturb_cli as cli


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/HOME/miniconda3/bin/python3"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return kernel.canonical_json_bytes(value)


def source_entry(path, data):
    return {"path": str(path), "sha256": sha(data), "size": len(data)}


def code_paths():
    return {
        "runner": Path(cli.__file__), "kernel": Path(kernel.__file__),
        "io": ROOT / "prosper/evaluation/r2_1_hoptimus_io.py",
        "legacy_io": Path(legacy.__file__),
        "package_init": ROOT / "prosper/__init__.py",
        "evaluation_init": ROOT / "prosper/evaluation/__init__.py",
    }


def csv_bytes(header, rows):
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def fixture(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    stage0 = inputs / "stage0"
    attempt = stage0 / "attempt"
    attempt.mkdir(parents=True)
    out_root = tmp_path / "output_root"
    out_root.mkdir()
    paths = {
        "metadata": inputs / "metadata.csv", "template": inputs / "synthetic_template.bin",
        "stage0_review": stage0 / "review.json",
        "stage0_complete": attempt / "COMPLETE.json",
        "stage0_inventory": attempt / "inventory.csv",
        "stage0_summary": attempt / "summary.json",
        "kernel_review": inputs / "kernel_review.json",
    }
    headers = ["Tumor.group", "Slide.Scan.File", "Sample.ID", "UID.Subject"]
    rows = [
        ["DSRCT", "z", "sample-z", "p1"],
        ["MISC", "misc", "sample-misc", "pm"],
        ["NORMAL", "a", "sample-a", "p1"],
        ["A", "b", "sample-b", "p2"],
    ]
    metadata = csv_bytes(headers, rows)
    inventory_rows = []
    bindings = [{
        "kind": "metadata", "name": "metadata", "path": str(paths["metadata"]),
        "resolved_path": str(paths["metadata"]), "size": len(metadata),
        "sha256": sha(metadata), "device": 999, "inode": 999,
    }]
    for row in rows:
        sid = row[1]
        feature_path = str(inputs / "never_read_features" / f"{sid}.pkl")
        digest = sha(f"not feature payload {sid}".encode())
        inventory_rows.append([
            "h_optimus_1", sid, 2, 2, "float32", feature_path, feature_path, 13, digest,
        ])
        bindings.append({
            "kind": "feature", "name": f"h_optimus_1/{sid}",
            "path": feature_path, "resolved_path": feature_path,
            "size": 13, "sha256": digest, "device": 999, "inode": 999,
        })
    inventory = csv_bytes(cli.INVENTORY_HEADER, inventory_rows)
    summary = {
        "schema_version": 1, "run_id": "synthetic-stage0",
        "scope": "feature-bank binding only; encoder identities declared/unverified",
        "declarations": {
            "run_id": "synthetic-stage0",
            "metadata": {**cli.METADATA_COLUMNS, "path": "../metadata.csv"},
            "banks": [{
                "name": "h_optimus_1", "feature_dim": 1536,
                "dtypes": ["float32"], "path": str(inputs / "never_read_features"),
            }],
        },
        "metadata_columns": headers, "input_bindings": bindings,
        "canonical_slides": 4, "patient_count": 3, "multi_label_patient_count": 1,
        "inventory_rows": 4, "canonical_tiles": 8, "bank_tile_totals": {"h_optimus_1": 8},
    }
    historical_spec = json_bytes(summary["declarations"])
    historical_spec_path = stage0 / "original_spec.json"
    bindings.append({
        "kind": "spec", "name": "spec", "path": str(historical_spec_path),
        "resolved_path": str(historical_spec_path), "size": len(historical_spec),
        "sha256": sha(historical_spec), "device": 999, "inode": 999,
    })
    legacy_path = Path(legacy.__file__).resolve()
    legacy_bytes = legacy_path.read_bytes()
    bindings.append({
        "kind": "code", "name": "running_auditor", "path": str(legacy_path),
        "resolved_path": str(legacy_path), "size": len(legacy_bytes),
        "sha256": sha(legacy_bytes), "device": 999, "inode": 999,
    })
    complete = {
        "schema_version": 1, "run_id": "synthetic-stage0", "scope": "feature-bank binding only",
        "outputs": {
            "inventory.csv": {"sha256": sha(inventory), "size": len(inventory)},
            "summary.json": {"sha256": sha(json_bytes(summary)), "size": len(json_bytes(summary))},
        },
    }
    review = {
        "schema_version": 1, "disposition": "accepted", "scope": "Stage0 feature-bank binding only",
        "artifacts": {
            "attempt/COMPLETE.json": sha(json_bytes(complete)),
            "attempt/inventory.csv": sha(inventory),
            "attempt/summary.json": sha(json_bytes(summary)),
            "original_spec.json": sha(historical_spec),
        },
        "source_sha256": {
            "metadata": sha(metadata), "auditor_module": sha(Path(legacy.__file__).read_bytes()),
        },
        "reconciled": {
            "raw_slides_per_bank": 4, "raw_patients": 3, "raw_multi_label_patients": 1,
            "non_misc_slides": 3, "non_misc_patients": 2, "non_misc_multi_label_patients": 1,
        },
    }
    kernel_digest = sha(Path(kernel.__file__).read_bytes())
    kernel_review = {
        "schema_version": 1, "disposition": "accepted",
        "scope": "Repaired pure stimulus/SAMPLER/L1 kernels only",
        "source_files": {"prosper/evaluation/r2_1_hoptimus_perturb.py": kernel_digest},
    }
    sources = {
        "metadata": metadata, "template": b"synthetic bytes, deliberately not a decodable image",
        "stage0_review": json_bytes(review), "stage0_complete": json_bytes(complete),
        "stage0_inventory": inventory, "stage0_summary": json_bytes(summary),
        "kernel_review": json_bytes(kernel_review),
    }
    code = {}
    for name, path in code_paths().items():
        if path.exists():
            code[name] = source_entry(path, path.read_bytes())
    spec = {
        "schema_version": 1, "run_id": "synthetic-perturbation", "task": cli.TASK,
        "sources": {name: source_entry(paths[name], data) for name, data in sources.items()},
        "code_files": code,
        "parameters": {"metadata": dict(cli.METADATA_COLUMNS)},
        "expected": {
            "raw_slides": 4, "target_slides": 3, "patients": 2, "groups": 3,
            "cross_label_patients": 1, "stimulus_records": 36, "feature_dim": 1536,
        },
    }
    return {"spec": spec, "sources": sources, "paths": paths, "inputs": inputs,
            "out_root": out_root}


def produce(bundle):
    return cli.execute(
        bundle["spec"], bundle["sources"], declaration_base=bundle["inputs"],
    )


def change_json(bundle, name, mutate):
    value = json.loads(bundle["sources"][name])
    mutate(value)
    bundle["sources"][name] = json_bytes(value)


def rebind_stage0(bundle):
    def update_complete(value):
        for source, filename in (("stage0_inventory", "inventory.csv"),
                                 ("stage0_summary", "summary.json")):
            data = bundle["sources"][source]
            value["outputs"][filename] = {"sha256": sha(data), "size": len(data)}
    change_json(bundle, "stage0_complete", update_complete)

    def update_review(value):
        for source, filename in (
            ("stage0_complete", "COMPLETE.json"),
            ("stage0_inventory", "inventory.csv"), ("stage0_summary", "summary.json"),
        ):
            value["artifacts"][f"attempt/{filename}"] = sha(bundle["sources"][source])
    change_json(bundle, "stage0_review", update_review)


def materialize(bundle):
    for name, data in bundle["sources"].items():
        path = bundle["paths"][name]
        path.write_bytes(data)
        bundle["spec"]["sources"][name] = source_entry(path, data)
    for name, path in code_paths().items():
        if path.exists():
            bundle["spec"]["code_files"][name] = source_entry(path, path.read_bytes())
    spec_path = bundle["inputs"] / "spec.json"
    spec_path.write_bytes(json_bytes(bundle["spec"]))
    return spec_path


def environment(out_root):
    env = os.environ.copy()
    env.update({
        "PROSPER_OUT": str(out_root), "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    return env


def normal_cli(bundle, out, *, spec_sha=None, out_root=None):
    spec_path = materialize(bundle)
    return subprocess.run([
        PYTHON, "-m", "prosper.evaluation.r2_1_hoptimus_perturb_cli",
        "--spec", str(spec_path), "--spec-sha256", spec_sha or sha(spec_path.read_bytes()),
        "--out", str(out), "--out-root", str(out_root or bundle["out_root"]),
    ], cwd=ROOT, env=environment(bundle["out_root"]), capture_output=True, text=True, timeout=60)


def require_common():
    if not code_paths()["io"].exists():
        pytest.skip("shared r2_1_hoptimus_io.py not yet available; normal CLI integration pending")


def test_normal_package_help_has_no_output_root_side_effect(tmp_path):
    out_root = tmp_path / "must_not_create"
    result = subprocess.run(
        [PYTHON, "-m", "prosper.evaluation.r2_1_hoptimus_perturb_cli", "--help"],
        cwd=ROOT, env=environment(out_root), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--spec-sha256" in result.stdout and "--out-root" in result.stdout
    assert not out_root.exists()


def test_bound_bytes_fixture_population_stimuli_and_buffer_accounting(tmp_path):
    bundle = fixture(tmp_path)
    artifacts = produce(bundle)
    assert set(artifacts) == cli.ARTIFACTS
    population = list(csv.DictReader(io.StringIO(artifacts["queries.csv"].decode())))
    assert [r["slide_id"] for r in population] == ["z", "a", "b"]
    assert [r["metadata_index"] for r in population] == ["0", "2", "3"]
    assert [r["diagnostic_group"] for r in population] == ["DSRCT", "NORMAL", "A"]
    assert artifacts["queries.csv"] == artifacts["references.csv"]
    decoded = json.loads(artifacts["stimuli.json"])
    assert [r["slide_id"] for r in decoded["records"][::12]] == ["a", "b", "z"]
    assert len(decoded["records"]) == 36
    assert all(r["level"] == 0 and r["patch_size"] == 512 for r in decoded["records"])
    assert decoded["policy"] == asdict(kernel.generate_stimulus_manifest([], "0" * 64).policy)
    for record in decoded["records"]:
        if record["condition"] == "reinhard":
            assert record["parameters"]["template_sha256"] == sha(bundle["sources"]["template"])
    summary = json.loads(artifacts["summary.json"])
    assert summary["counts"] == bundle["spec"]["expected"]
    assert summary["policy"] == decoded["policy"]
    assert set(summary["details"]["condition_counts"]) == set(kernel.CONDITION_ORDER)
    assert set(summary) == {"schema_version", "run_id", "task", "counts", "policy",
                           "details", "limitations"}
    assert not (bundle["inputs"] / "never_read_features").exists()
    print("Synthetic source bytes:", sum(map(len, bundle["sources"].values())))
    print("Synthetic retained artifact bytes:", sum(map(len, artifacts.values())))
    print("Synthetic unique artifact-buffer bytes:",
          sum(len(v) for v in {id(v): v for v in artifacts.values()}.values()))


def test_callback_never_opens_feature_or_template_paths(tmp_path, monkeypatch):
    bundle = fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("callback attempted a file read")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert len(produce(bundle)["stimuli.json"]) > 0


@pytest.mark.parametrize("name,value", [
    ("raw_slides", 5), ("target_slides", 2), ("patients", 1),
    ("groups", 2), ("cross_label_patients", 0), ("stimulus_records", 35),
    ("feature_dim", 1280), ("feature_dim", True),
])
def test_wrong_declared_counts_fail(tmp_path, name, value):
    bundle = fixture(tmp_path)
    bundle["spec"]["expected"][name] = value
    with pytest.raises(ValueError):
        produce(bundle)


@pytest.mark.parametrize("control,field,value", [
    ("stage0_review", "disposition", "pending"),
    ("stage0_review", "scope", "different scope"),
    ("stage0_complete", "run_id", "different run"),
    ("stage0_summary", "run_id", "different run"),
    ("stage0_summary", "schema_version", True),
    ("kernel_review", "scope", "not accepted scope"),
    ("kernel_review", "disposition", "pending"),
])
def test_invalid_control_status_and_run_identity(tmp_path, control, field, value):
    bundle = fixture(tmp_path)
    change_json(bundle, control, lambda obj: obj.update({field: value}))
    if control in ("stage0_summary", "stage0_complete"):
        rebind_stage0(bundle)
    with pytest.raises(ValueError):
        produce(bundle)


def test_kernel_review_must_match_executing_kernel(tmp_path):
    bundle = fixture(tmp_path)
    change_json(bundle, "kernel_review", lambda obj: obj["source_files"].update({
        "prosper/evaluation/r2_1_hoptimus_perturb.py": "0" * 64,
    }))
    with pytest.raises(ValueError, match="kernel"):
        produce(bundle)


@pytest.mark.parametrize("mutation", ["hash", "size", "missing", "escaping", "attempt"])
def test_consumed_upstream_links_are_not_optional(tmp_path, mutation):
    bundle = fixture(tmp_path)
    if mutation in ("hash", "size"):
        change_json(bundle, "stage0_complete", lambda obj: obj["outputs"]["inventory.csv"].update({
            "sha256" if mutation == "hash" else "size": "0" * 64 if mutation == "hash" else 1,
        }))
        change_json(bundle, "stage0_review", lambda obj: obj["artifacts"].update({
            "attempt/COMPLETE.json": sha(bundle["sources"]["stage0_complete"]),
        }))
    elif mutation == "missing":
        change_json(bundle, "stage0_review", lambda obj: obj["artifacts"].pop("attempt/summary.json"))
    elif mutation == "escaping":
        change_json(bundle, "stage0_review", lambda obj: obj["artifacts"].update({"../escape": "a" * 64}))
    else:
        bundle["spec"]["sources"]["stage0_complete"]["path"] = str(bundle["inputs"] / "other/COMPLETE.json")
    with pytest.raises(ValueError):
        produce(bundle)


@pytest.mark.parametrize("mutation", ["review_hash", "summary_hash", "summary_size", "identity", "role"])
def test_metadata_must_match_upstream_bindings(tmp_path, mutation):
    bundle = fixture(tmp_path)
    if mutation == "review_hash":
        change_json(bundle, "stage0_review", lambda obj: obj["source_sha256"].update({"metadata": "0" * 64}))
    else:
        binding_field = {
            "summary_hash": ("sha256", "0" * 64), "summary_size": ("size", 1),
            "identity": ("resolved_path", "/not/the/bound/metadata"),
            "role": ("kind", "feature"),
        }[mutation]
        def update(obj):
            obj["input_bindings"][0][binding_field[0]] = binding_field[1]
            if mutation == "identity":
                obj["input_bindings"][0]["path"] = "/not/the/bound/metadata"
        change_json(bundle, "stage0_summary", update)
        rebind_stage0(bundle)
    with pytest.raises(ValueError):
        produce(bundle)


@pytest.mark.parametrize("mutation", ["width", "bool_width", "duplicate_bank", "row_count",
                                    "feature_binding", "missing_binding", "duplicate_binding"])
def test_summary_width_and_feature_lineage(tmp_path, mutation):
    bundle = fixture(tmp_path)
    def update(obj):
        if mutation == "width":
            obj["declarations"]["banks"][0]["feature_dim"] = 1280
        elif mutation == "bool_width":
            obj["declarations"]["banks"][0]["feature_dim"] = True
        elif mutation == "duplicate_bank":
            obj["declarations"]["banks"].append(obj["declarations"]["banks"][0])
        elif mutation == "row_count":
            obj["inventory_rows"] = 5
        elif mutation == "feature_binding":
            obj["input_bindings"][1]["sha256"] = "0" * 64
        elif mutation == "missing_binding":
            obj["input_bindings"].pop(1)
        else:
            obj["input_bindings"].append(obj["input_bindings"][1])
    change_json(bundle, "stage0_summary", update)
    rebind_stage0(bundle)
    with pytest.raises(ValueError):
        produce(bundle)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "unknown", "count", "extra_column",
                                    "trailing", "dimension_column"])
def test_inventory_requires_exact_unique_raw_population_and_schema(tmp_path, mutation):
    bundle = fixture(tmp_path)
    rows = list(csv.reader(io.StringIO(bundle["sources"]["stage0_inventory"].decode())))
    if mutation == "duplicate":
        rows.append(rows[1])
    elif mutation == "missing":
        rows.pop()
    elif mutation == "unknown":
        rows[1][1] = "unknown"
    elif mutation == "count":
        rows[1][3] = "3"
    elif mutation == "extra_column":
        rows[0].append("unexpected")
    elif mutation == "dimension_column":
        rows[0].append("feature_dim")
    else:
        rows.append([])
    bundle["sources"]["stage0_inventory"] = csv_bytes(tuple(rows[0]), rows[1:])
    rebind_stage0(bundle)
    with pytest.raises(ValueError):
        produce(bundle)


@pytest.mark.parametrize("mutation", ["duplicate_slide", "blank_patient", "duplicate_header",
                                    "unknown_header", "trailing"])
def test_metadata_exact_headers_rows_and_patient_mapping(tmp_path, mutation):
    bundle = fixture(tmp_path)
    rows = list(csv.reader(io.StringIO(bundle["sources"]["metadata"].decode())))
    if mutation == "duplicate_slide":
        rows[2][1] = rows[1][1]
    elif mutation == "blank_patient":
        rows[1][3] = ""
    elif mutation == "duplicate_header":
        rows[0][2] = rows[0][1]
    elif mutation == "unknown_header":
        rows[0][2] = "unknown"
    else:
        rows.append([])
    bundle["sources"]["metadata"] = csv_bytes(tuple(rows[0]), rows[1:])
    digest = sha(bundle["sources"]["metadata"])
    change_json(bundle, "stage0_summary", lambda obj: obj["input_bindings"][0].update({
        "sha256": digest, "size": len(bundle["sources"]["metadata"]),
    }))
    change_json(bundle, "stage0_review", lambda obj: obj["source_sha256"].update({"metadata": digest}))
    rebind_stage0(bundle)
    with pytest.raises(ValueError):
        produce(bundle)


@pytest.mark.parametrize("artifact,mutation", [
    ("queries.csv", "missing_row"), ("references.csv", "extra_row"),
    ("queries.csv", "wrong_index"), ("queries.csv", "wrong_feature_hash"),
    ("queries.csv", "duplicate_header"), ("stimuli.json", "missing_condition"),
    ("stimuli.json", "wrong_factor"), ("stimuli.json", "wrong_template"),
    ("stimuli.json", "unknown_key"), ("stimuli.json", "duplicate_key"),
    ("summary.json", "wrong_count"), ("summary.json", "unknown_key"),
    ("summary.json", "bool_width"), ("summary.json", "trailing"),
    ("summary.json", "float_policy_seed"),
])
def test_serialized_artifacts_are_semantically_reparsed(tmp_path, artifact, mutation):
    bundle = fixture(tmp_path)
    artifacts = produce(bundle)
    controls = cli._stage0_controls(bundle["spec"], bundle["sources"], bundle["inputs"])
    population, counts = cli._population(bundle["spec"], bundle["sources"], controls)
    if artifact.endswith(".csv"):
        rows = list(csv.reader(io.StringIO(artifacts[artifact].decode())))
        if mutation == "missing_row":
            rows.pop()
        elif mutation == "extra_row":
            rows.append(rows[-1])
        elif mutation == "wrong_index":
            rows[1][0] = "1"
        elif mutation == "wrong_feature_hash":
            rows[1][-1] = "0" * 64
        else:
            rows[0][1] = rows[0][0]
        artifacts[artifact] = csv_bytes(tuple(rows[0]), rows[1:])
    else:
        obj = json.loads(artifacts[artifact])
        if mutation == "missing_condition":
            obj["records"].pop()
        elif mutation == "wrong_factor":
            obj["records"][6]["parameters"]["factor"] = 5.0
        elif mutation == "wrong_template":
            obj["records"][10]["parameters"]["template_sha256"] = "0" * 64
        elif mutation == "wrong_count":
            obj["counts"]["stimulus_records"] = 1
        elif mutation == "bool_width":
            obj["details"]["feature_dim"] = True
        elif mutation == "float_policy_seed":
            obj["policy"]["seed"] = 1234.0
        elif mutation == "unknown_key":
            obj["unknown"] = "unaccepted"
        artifacts[artifact] = json_bytes(obj)
        if mutation == "duplicate_key":
            artifacts[artifact] = artifacts[artifact].replace(b'{"policy":', b'{"policy":{},"policy":', 1)
        elif mutation == "trailing":
            artifacts[artifact] += b"{}"
    with pytest.raises((ValueError, kernel.StimulusValidationError)):
        cli.validate_artifacts(
            bundle["spec"], artifacts, population, counts, sha(bundle["sources"]["template"])
        )


def test_normal_cli_roundtrip_and_receipt_schema(tmp_path):
    require_common()
    bundle = fixture(tmp_path)
    out = bundle["out_root"] / "attempt"
    result = normal_cli(bundle, out)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {p.name for p in out.iterdir()} == cli.ARTIFACTS | {"COMPLETE.json", "input_bindings.json"}
    complete = json.loads((out / "COMPLETE.json").read_bytes())
    assert set(complete) == {"schema_version", "run_id", "task", "scope", "spec_sha256",
                            "outputs", "verified_at", "limitations"}
    assert complete["scope"] == "hoptimus_perturb_stimulus_manifests_only"
    assert set(complete["outputs"]) == cli.ARTIFACTS | {"input_bindings.json"}
    for name, entry in complete["outputs"].items():
        data = (out / name).read_bytes()
        assert entry == {"sha256": sha(data), "size": len(data)}
    bindings = json.loads((out / "input_bindings.json").read_bytes())
    assert set(bindings) == {"schema_version", "run_id", "task", "spec_sha256", "bindings",
                             "executing_code_paths", "environment", "started_at", "captured_at",
                             "limitations"}
    assert set(bindings["executing_code_paths"]) == set(code_paths())
    assert bindings["environment"]["native_threads"] == {
        key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    }
    again = normal_cli(bundle, out)
    assert again.returncode != 0


def test_normal_cli_spec_symlink_uses_lexical_relative_sources(tmp_path):
    require_common()
    bundle = fixture(tmp_path)
    lexical_spec = materialize(bundle)
    for name, path in bundle["paths"].items():
        bundle["spec"]["sources"][name]["path"] = str(path.relative_to(bundle["inputs"]))
    physical_directory = tmp_path / "physical_spec_directory"
    physical_directory.mkdir()
    physical_spec = physical_directory / "spec.json"
    spec_bytes = json_bytes(bundle["spec"])
    physical_spec.write_bytes(spec_bytes)
    lexical_spec.unlink()
    lexical_spec.symlink_to(physical_spec)
    assert {path.name for path in physical_directory.iterdir()} == {"spec.json"}
    out = bundle["out_root"] / "symlink_attempt"
    result = subprocess.run([
        PYTHON, "-m", "prosper.evaluation.r2_1_hoptimus_perturb_cli",
        "--spec", str(lexical_spec), "--spec-sha256", sha(spec_bytes),
        "--out", str(out), "--out-root", str(bundle["out_root"]),
    ], cwd=ROOT, env=environment(bundle["out_root"]), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {path.name for path in out.iterdir()} == cli.ARTIFACTS | {
        "COMPLETE.json", "input_bindings.json",
    }
    complete = json.loads((out / "COMPLETE.json").read_bytes())
    assert complete["scope"] == "hoptimus_perturb_stimulus_manifests_only"
    assert complete["spec_sha256"] == sha(spec_bytes)
    assert set(complete["outputs"]) == cli.ARTIFACTS | {"input_bindings.json"}
    for name, entry in complete["outputs"].items():
        data = (out / name).read_bytes()
        assert entry == {"sha256": sha(data), "size": len(data)}
    binding_document = json.loads((out / "input_bindings.json").read_bytes())
    spec_binding = binding_document["bindings"][0]
    assert spec_binding["kind"] == "spec" and spec_binding["name"] == "spec"
    assert spec_binding["path"] == str(lexical_spec.absolute())
    assert spec_binding["resolved_path"] == str(physical_spec.resolve())
    source_bindings = {
        entry["name"]: entry for entry in binding_document["bindings"]
        if entry["kind"] == "source"
    }
    assert set(source_bindings) == cli.SOURCE_NAMES
    for name, expected_path in bundle["paths"].items():
        binding = source_bindings[name]
        assert binding["path"] == str(expected_path)
        assert binding["resolved_path"] == str(expected_path.resolve())
        assert binding["sha256"] == sha(bundle["sources"][name])
        assert binding["size"] == len(bundle["sources"][name])
    controls = cli._stage0_controls(bundle["spec"], bundle["sources"], bundle["inputs"])
    population, counts = cli._population(bundle["spec"], bundle["sources"], controls)
    artifacts = {name: (out / name).read_bytes() for name in cli.ARTIFACTS}
    cli.validate_artifacts(
        bundle["spec"], artifacts, population, counts, sha(bundle["sources"]["template"])
    )
    assert {path.name for path in physical_directory.iterdir()} == {"spec.json"}


@pytest.mark.parametrize("mutation", ["wrong_hash", "wrong_count", "root_mismatch",
                                    "output_overlap", "unknown_spec_key", "duplicate_spec_key"])
def test_normal_cli_refuses_invalid_binding_without_complete(tmp_path, mutation):
    require_common()
    bundle = fixture(tmp_path)
    out = bundle["out_root"] / "attempt"
    if mutation == "wrong_count":
        bundle["spec"]["expected"]["target_slides"] = 99
    elif mutation == "unknown_spec_key":
        bundle["spec"]["unknown"] = 1
    elif mutation == "output_overlap":
        out = bundle["inputs"]
    if mutation == "duplicate_spec_key":
        spec_path = materialize(bundle)
        data = spec_path.read_bytes().replace(b'{"code_files":', b'{"run_id":"forged","code_files":', 1)
        spec_path.write_bytes(data)
        result = subprocess.run([
            PYTHON, "-m", "prosper.evaluation.r2_1_hoptimus_perturb_cli",
            "--spec", str(spec_path), "--spec-sha256", sha(data), "--out", str(out),
            "--out-root", str(bundle["out_root"]),
        ], cwd=ROOT, env=environment(bundle["out_root"]), capture_output=True, text=True, timeout=60)
    else:
        result = normal_cli(
            bundle, out, spec_sha="0" * 64 if mutation == "wrong_hash" else None,
            out_root=tmp_path / "different_root" if mutation == "root_mismatch" else None,
        )
    assert result.returncode != 0
    assert not (out / "COMPLETE.json").exists()


@pytest.mark.parametrize("mutation", ["source", "output"])
def test_common_final_sweep_refuses_late_mutation(tmp_path, monkeypatch, mutation):
    require_common()
    from prosper.evaluation import r2_1_hoptimus_io as bound

    bundle = fixture(tmp_path)
    spec_path = materialize(bundle)
    out = bundle["out_root"] / "attempt"
    monkeypatch.setenv("PROSPER_OUT", str(bundle["out_root"]))
    if mutation == "output":
        atomic = legacy._atomic_bytes

        def mutate_after_write(fd, name, data):
            atomic(fd, name, data)
            if name == "summary.json":
                (out / "queries.csv").write_bytes(b"late mutation")

        monkeypatch.setattr(legacy, "_atomic_bytes", mutate_after_write)

    def callback(spec, sources):
        artifacts = cli.execute(spec, sources, declaration_base=bundle["inputs"])
        if mutation == "source":
            bundle["paths"]["template"].write_bytes(b"late source mutation")
        return artifacts

    with pytest.raises(ValueError, match="mutation|changed|size|hash|differ"):
        bound.run_bound(
            spec_path, sha(spec_path.read_bytes()), out, bundle["out_root"],
            task=cli.TASK, code_paths=code_paths(), execute=callback,
        )
    assert not (out / "COMPLETE.json").exists()


def test_portable_example_is_a_placeholder_not_a_production_claim():
    spec = legacy._json((ROOT / "examples/r2_1_hoptimus_perturb.json").read_bytes())
    assert set(spec) == {"schema_version", "run_id", "task", "sources", "code_files",
                         "parameters", "expected"}
    assert spec["task"] == cli.TASK
    assert spec["parameters"] == {"metadata": cli.METADATA_COLUMNS}
    assert set(spec["sources"]) == cli.SOURCE_NAMES
    assert set(spec["code_files"]) == set(code_paths())
    assert set(spec["expected"]) == cli.EXPECTED_KEYS
    assert spec["expected"]["stimulus_records"] == 12 * spec["expected"]["target_slides"]
    assert "replace_all_placeholder_bindings" in spec["run_id"]
    for entry in (*spec["sources"].values(), *spec["code_files"].values()):
        assert set(entry) == {"path", "sha256", "size"}
        assert not Path(entry["path"]).is_absolute()
        assert entry["sha256"] == "0" * 64 and entry["size"] == 0
