"""Synthetic bytes and normal-package integration of the filter-manifest CLI."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from prosper.evaluation import r2_1_hoptimus_filter as kernel
from prosper.evaluation import r2_1_hoptimus_filter_cli as cli


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def csv_bytes(header, rows):
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return text.getvalue().encode()


def csv_read(data, compressed=False):
    if compressed:
        data = gzip.decompress(data)
    return list(csv.DictReader(io.StringIO(data.decode(), newline="")))


def annotation(slide, patient, group, label, index):
    return kernel.AnnotationRecord(
        kernel.physical_tile_id(slide, index * 512, 1024),
        slide, patient, group, 0, index * 512, 1024, 512, label, index,
    )


def fixture_data():
    annotations = []
    for patient_index in range(8):
        patient = f"p{patient_index}"
        for label, count in ((1, 13), (3, 9), (4, 5), (5, 3)):
            slide = f"s{patient_index}_{label}"
            group = "WT" if label in (1, 3) else "NB"
            annotations.extend(annotation(slide, patient, group, label, i) for i in range(count))
    for patient, count in (("immune_a", 47), ("immune_b", 3)):
        annotations.extend(annotation(patient, patient, "NORMAL", 2, i) for i in range(count))
    annotations.append(annotation("misc_training", "p0", "MISC", 1, 0))
    by_slide = {row.slide_id: (row.patient_id, row.diagnostic_group) for row in annotations}
    by_slide.update({
        "query_z": ("query_patient", "EWS"),
        "query_a": ("query_patient", "OS"),
        "query_x": ("other_query_patient", "RB"),
    })
    ordered_slides = tuple(reversed(tuple(by_slide)))
    metadata_rows = [
        (by_slide[slide][1], slide, f"sample_{index}", by_slide[slide][0])
        for index, slide in enumerate(ordered_slides)
    ]
    sources = {
        "metadata": csv_bytes(
            ("Tumor.group", "Slide.Scan.File", "Sample.ID", "UID.Subject"), metadata_rows,
        ),
    }
    eligible_rows = []
    for legacy_index, row in enumerate(reversed(annotations)):
        eligible_rows.append((
            row.tile_id, row.slide_id, row.patient_id, row.diagnostic_group,
            row.level, row.x, row.y, row.patch_size, row.tissue_label,
            10000 + legacy_index, 20000 + legacy_index, 30000 + legacy_index,
            40000 + legacy_index, row.h_optimus_1_feature_row,
        ))
    sources["stage1a_eligible"] = csv_bytes(cli.ELIGIBLE_HEADER, eligible_rows)
    slide_counts = Counter(row.slide_id for row in annotations)
    feature_rows = []
    for bank, width, stop in (("uni2", 1536, 1536), ("virchow2", 2560, 1280),
                              ("h_optimus_1", 1536, 1536)):
        for slide, count in sorted(slide_counts.items()):
            feature_rows.append((
                bank, slide, width, 0, stop, count + 3, "float32", count, 3, 0, 0,
                "false", 1, f"/unread/features/{bank}/{slide}.pkl",
                f"/unread/features/{bank}/{slide}.pkl", 1234, "a" * 64,
            ))
    sources["stage1a_feature_inventory"] = csv_bytes(cli.FEATURE_HEADER, feature_rows)
    eligible = {
        "rows": len(annotations), "slides": len(slide_counts),
        "patients": len({row.patient_id for row in annotations}),
        "class_rows": {
            str(label): sum(row.tissue_label == label for row in annotations)
            for label in range(1, 6)
        },
        "class_slides": {
            str(label): len({row.slide_id for row in annotations if row.tissue_label == label})
            for label in range(1, 6)
        },
        "class_patients": {
            str(label): len({row.patient_id for row in annotations if row.tissue_label == label})
            for label in range(1, 6)
        },
    }
    sources["stage1a_summary"] = json_bytes({
        "schema_version": 1, "run_id": "stage1a_fixture",
        "scope": "historical-annotation target keyed binding only",
        "eligible": eligible, "quarantine": {"rows": 7, "slides": 1},
        "declarations": {"banks": [
            {"name": "h_optimus_1", "feature_dim": 1536, "projection": [0, 1536]},
        ]},
        "feature_inventory_totals": {"h_optimus_1": {
            "feature_rows": len(annotations) + 3 * len(slide_counts),
            "target_rows": len(annotations), "unused_unannotated": 3 * len(slide_counts),
            "unused_labelled_nontarget": 0, "unused_outside_declared_sources": 0,
        }},
    })
    sources["stage1a_review"] = json_bytes({
        "schema_version": 1, "disposition": "accepted",
        "scope": "Stage1a historical-annotation target keyed binding only",
        "reconciled": {
            "eligible_rows": eligible["rows"], "eligible_slides": eligible["slides"],
            "eligible_patients": eligible["patients"], "quarantine_rows": 7,
            "quarantine_slides": 1,
        },
    })
    for name in cli.SOURCE_NAMES - sources.keys():
        sources[name] = json_bytes({})
    spec = {
        "schema_version": 1, "run_id": "filter_fixture", "task": cli.TASK,
        "parameters": {"metadata": dict(cli.METADATA_COLUMNS)},
        "sources": {name: {"path": f"{name}.input", "sha256": digest(data), "size": len(data)}
                    for name, data in sources.items()},
        "code_files": {},
    }
    metadata = cli._metadata(sources["metadata"])
    targets = [row for row in metadata if row.diagnostic_group not in ("MISC", "NORMAL")]
    groups = defaultdict(set)
    for row in targets:
        groups[row.patient_id].add(row.diagnostic_group)
    spec["expected"] = {
        "raw_slides": len(metadata), "target_slides": len(targets), "patients": len(groups),
        "groups": len({row.diagnostic_group for row in targets}),
        "cross_label_patients": sum(len(values) > 1 for values in groups.values()),
        "annotation_tiles": len(annotations), "annotation_slides": len(slide_counts),
        "annotation_patients": eligible["patients"], "quarantine_tiles": 7,
        "quarantine_slides": 1,
    }
    return spec, sources


def context(spec, sources):
    metadata = cli._metadata(sources["metadata"])
    retrieval = tuple(row for row in metadata if row.diagnostic_group not in ("MISC", "NORMAL"))
    lookup = {row.slide_id: row for row in metadata}
    inventory = cli._inventory(sources["stage1a_feature_inventory"], lookup)
    annotations = cli._annotations(sources["stage1a_eligible"], lookup, inventory)
    records = tuple(kernel.RetrievalRecord(
        row.slide_id, row.patient_id, row.diagnostic_group,
    ) for row in retrieval)
    annotations, records = kernel.validate_manifests(annotations, records)
    folds = kernel.build_patient_folds(annotations, records)
    manifests = kernel.build_training_manifests(annotations, records, folds)
    return annotations, retrieval, folds, manifests


def mutate_csv(sources, name, column, value, row_index=0):
    rows = list(csv.reader(io.StringIO(sources[name].decode(), newline="")))
    rows[row_index + 1][rows[0].index(column)] = str(value)
    sources[name] = csv_bytes(rows[0], rows[1:])


def mutate_json(sources, name, mutate):
    value = json.loads(sources[name])
    mutate(value)
    sources[name] = json_bytes(value)


def test_complete_artifact_schemas_order_and_independent_reconstruction():
    spec, sources = fixture_data()
    artifacts = cli.execute(spec, sources)
    assert set(artifacts) == cli.ARTIFACT_NAMES
    population = csv_read(artifacts["retrieval_population.csv"])
    metadata = cli._metadata(sources["metadata"])
    target = [row for row in metadata if row.diagnostic_group not in ("MISC", "NORMAL")]
    assert [row["slide_id"] for row in population] == [row.slide_id for row in target]
    assert [int(row["metadata_index"]) for row in population] == [
        row.metadata_index for row in target
    ]
    assert [int(row["query_index"]) for row in population] == list(range(len(target)))
    assert all(row["query_index"] == row["reference_index"] for row in population)
    fold_rows = csv_read(artifacts["patient_folds.csv"])
    assert [row["patient_id"] for row in fold_rows] == sorted(
        row["patient_id"] for row in fold_rows
    )
    by_order = sorted(fold_rows, key=lambda row: int(row["kernel_order"]))
    patients = [row["patient_id"] for row in by_order]
    assert patients == sorted(patients, key=lambda patient: (
        digest(json.dumps(["r2_1_patient_fold_v1", 1234, patient],
                          separators=(",", ":")).encode()), patient,
    ))
    assert patients != sorted(patients)
    mapping = {row["patient_id"]: int(row["fold"]) for row in fold_rows}
    assigned = csv_read(artifacts["annotation_assignments.csv.gz"], compressed=True)
    by_id = {row["tile_id"]: row for row in assigned}
    assert len(by_id) == spec["expected"]["annotation_tiles"]
    assert assigned == sorted(assigned, key=lambda row: (
        row["slide_id"], int(row["level"]), int(row["x"]), int(row["y"]), int(row["patch_size"]),
    ))
    assert all(int(row["held_out_fold"]) == mapping[row["patient_id"]] for row in assigned)
    heads = json.loads(artifacts["head_plan.json"])["heads"]
    training = csv_read(artifacts["training_rows.csv.gz"], compressed=True)
    groups = defaultdict(list)
    for row in training:
        key = int(row["held_out_fold"]), int(row["budget_percent"])
        groups[key].append(row)
        tile = by_id[row["tile_id"]]
        assert tile["h_optimus_1_feature_row"] == row["h_optimus_1_feature_row"]
        assert mapping[row["patient_id"]] != key[0]
    for head in heads:
        key = head["held_out_fold"], head["budget_percent"]
        rows = groups[key]
        ids = [row["tile_id"] for row in rows]
        assert [int(row["training_position"]) for row in rows] == list(range(len(rows)))
        assert head["training_tile_ids_sha256"] == digest(
            json.dumps(ids, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        )
        assert len(rows) == head["selected_count"]
        counts = Counter(int(row["tissue_label"]) for row in rows)
        for stat in head["class_stats"]:
            assert counts[stat["tissue_label"]] == stat["selected_count"]
            assert stat["balanced_weight"] == len(rows) / (5 * stat["selected_count"])
        if key[1] == 5:
            full = [row["tile_id"] for row in groups[(key[0], 100)]]
            assert ids == [tile for tile in full if tile in set(ids)]
    summary = json.loads(artifacts["summary.json"])
    assert summary["counts"] == spec["expected"]
    assert summary["details"]["quarantine_count_source"] == cli.QUARANTINE_SOURCE
    assert len(summary["details"]["fold_support"]) == 5
    assert len(heads) == 10


def test_deterministic_compression_and_sources_not_mutated():
    spec, sources = fixture_data()
    before = dict(sources)
    first = cli.execute(spec, sources)
    second = cli.execute(spec, sources)
    assert first == second and sources == before
    for name in ("annotation_assignments.csv.gz", "training_rows.csv.gz"):
        assert first[name][4:8] == b"\0" * 4


@pytest.mark.parametrize("mutation", ["bounds", "negative", "boolean", "legacy_confusion"])
def test_dedicated_source_index_validation(mutation):
    spec, sources = fixture_data()
    if mutation == "legacy_confusion":
        mutate_csv(sources, "stage1a_eligible", "h_optimus_1_feature_row", 10000)
    elif mutation == "bounds":
        mutate_csv(sources, "stage1a_eligible", "h_optimus_1_feature_row", 4)
    elif mutation == "negative":
        mutate_csv(sources, "stage1a_eligible", "h_optimus_1_feature_row", -1)
    else:
        mutate_csv(sources, "stage1a_eligible", "h_optimus_1_feature_row", "True")
    with pytest.raises(cli.FilterManifestError):
        cli.execute(spec, sources)


@pytest.mark.parametrize("column,value", [
    ("patient_id", "other_patient"), ("diagnostic_group", "NB"),
    ("tile_id", "0" * 64), ("tissue_label", "0"), ("level", "1"),
])
def test_annotation_identity_metadata_and_quarantine_guards(column, value):
    spec, sources = fixture_data()
    mutate_csv(sources, "stage1a_eligible", column, value)
    with pytest.raises(cli.FilterManifestError):
        cli.execute(spec, sources)


@pytest.mark.parametrize("name", ["metadata", "stage1a_eligible", "stage1a_feature_inventory"])
@pytest.mark.parametrize("mutation", ["duplicate_header", "unknown_header", "duplicate_row",
                                      "extra_cell", "blank_row"])
def test_source_csv_structure_and_duplicates(name, mutation):
    spec, sources = fixture_data()
    rows = list(csv.reader(io.StringIO(sources[name].decode(), newline="")))
    if mutation == "duplicate_header":
        rows[0][-1] = rows[0][0]
    elif mutation == "unknown_header":
        rows[0][-1] = "UNKNOWN"
    elif mutation == "duplicate_row":
        rows.append(rows[1])
    elif mutation == "extra_cell":
        rows[-1].append("extra")
    else:
        rows.append([])
    sources[name] = csv_bytes(rows[0], rows[1:])
    with pytest.raises(cli.FilterManifestError):
        cli.execute(spec, sources)


@pytest.mark.parametrize("key", sorted(cli.EXPECTED_KEYS))
def test_expected_count_mismatch(key):
    spec, sources = fixture_data()
    spec["expected"][key] += 1
    with pytest.raises(cli.FilterManifestError, match="observed counts"):
        cli.execute(spec, sources)


@pytest.mark.parametrize("key", sorted(cli.EXPECTED_KEYS))
def test_boolean_expected_counts_rejected(key):
    spec, sources = fixture_data()
    spec["expected"][key] = True
    with pytest.raises(cli.FilterManifestError, match="integer"):
        cli.execute(spec, sources)


@pytest.mark.parametrize("name", ["stage1a_summary", "stage1a_review"])
def test_quarantine_inherited_controls_must_agree(name):
    spec, sources = fixture_data()
    if name.endswith("summary"):
        mutate_json(sources, name, lambda value: value["quarantine"].update(rows=8))
    else:
        mutate_json(sources, name, lambda value: value["reconciled"].update(quarantine_rows=8))
    with pytest.raises(cli.FilterManifestError, match="quarantine"):
        cli.execute(spec, sources)


@pytest.mark.parametrize("mutation", ["width", "projection", "target_rows", "bank", "feature_hash"])
def test_inventory_validation(mutation):
    spec, sources = fixture_data()
    rows = csv_read(sources["stage1a_feature_inventory"])
    index = next(i for i, row in enumerate(rows) if row["bank"] == "h_optimus_1")
    column, value = {
        "width": ("feature_dim", 1280), "projection": ("projection_stop", 1280),
        "target_rows": ("target_rows", 999), "bank": ("bank", "wrong_encoder"),
        "feature_hash": ("sha256", "wrong"),
    }[mutation]
    mutate_csv(sources, "stage1a_feature_inventory", column, value, index)
    with pytest.raises(cli.FilterManifestError):
        cli.execute(spec, sources)


@pytest.mark.parametrize("mutation", ["duplicate", "nonfinite", "overflow", "trailing"])
def test_strict_source_json(mutation):
    spec, sources = fixture_data()
    if mutation == "duplicate":
        sources["stage1a_summary"] = sources["stage1a_summary"].replace(
            b'"eligible":', b'"eligible": {}, "eligible":', 1,
        )
    elif mutation == "nonfinite":
        sources["stage1a_summary"] = sources["stage1a_summary"].replace(
            b'"rows": 291', b'"rows": NaN',
        )
        if b"NaN" not in sources["stage1a_summary"]:
            sources["stage1a_summary"] = b'{"rows":NaN}'
    elif mutation == "overflow":
        sources["stage1a_summary"] = b'{"unused":1e999}'
    else:
        sources["stage1a_summary"] += b"{}"
    with pytest.raises(cli.FilterManifestError):
        cli.execute(spec, sources)


def test_no_file_access_in_execute(monkeypatch):
    spec, sources = fixture_data()
    def forbidden(*args, **kwargs):
        raise AssertionError("callback cannot read paths or write output")
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    assert set(cli.execute(spec, sources)) == cli.ARTIFACT_NAMES


def test_unicode_quoted_exact_patient_ids_roundtrip():
    spec, sources = fixture_data()
    for name, column in (("metadata", "UID.Subject"), ("stage1a_eligible", "patient_id")):
        rows = list(csv.reader(io.StringIO(sources[name].decode(), newline="")))
        index = rows[0].index(column)
        for row in rows[1:]:
            if row[index] == "p0":
                row[index] = ' patient,é"0 '
        sources[name] = csv_bytes(rows[0], rows[1:])
    artifacts = cli.execute(spec, sources)
    assert any(row["patient_id"] == ' patient,é"0 '
               for row in csv_read(artifacts["patient_folds.csv"]))


@pytest.mark.parametrize("mutation", ["missing_role", "extra_role", "expected_extra",
                                      "parameter_extra", "parameter_changed"])
def test_source_roles_expected_keys_and_parameter_contract(mutation):
    spec, sources = fixture_data()
    if mutation == "missing_role":
        sources.pop("stage0_summary")
    elif mutation == "extra_role":
        sources["feature_payload"] = b"forbidden"
    elif mutation == "expected_extra":
        spec["expected"]["extra"] = 1
    elif mutation == "parameter_extra":
        spec["parameters"]["seed"] = 9
    else:
        spec["parameters"]["metadata"]["patient_col"] = "wrong"
    with pytest.raises(cli.FilterManifestError):
        cli.execute(spec, sources)


@pytest.mark.parametrize("name", ["retrieval_population.csv", "patient_folds.csv",
                                "annotation_assignments.csv.gz", "training_rows.csv.gz"])
@pytest.mark.parametrize("mutation", ["header", "extra", "missing", "reordered"])
def test_serialized_csv_readback_rejects_tampering(name, mutation):
    spec, sources = fixture_data()
    artifacts = cli.execute(spec, sources)
    compressed = name.endswith(".gz")
    data = gzip.decompress(artifacts[name]) if compressed else artifacts[name]
    rows = list(csv.reader(io.StringIO(data.decode(), newline="")))
    if mutation == "header":
        rows[0][-1] = "UNKNOWN"
    elif mutation == "extra":
        rows.append(rows[-1])
    elif mutation == "missing":
        rows.pop()
    else:
        rows[1], rows[2] = rows[2], rows[1]
    altered = csv_bytes(rows[0], rows[1:])
    artifacts[name] = gzip.compress(altered, mtime=0) if compressed else altered
    annotations, retrieval, folds, manifests = context(spec, sources)
    with pytest.raises(cli.FilterManifestError):
        cli.validate_artifacts(
            artifacts, spec, annotations, retrieval, folds, manifests,
            json.loads(cli.execute(spec, sources)["summary.json"]),
        )


@pytest.mark.parametrize("mutation", ["trailing", "extra_member", "truncated", "mtime"])
def test_gzip_framing_rejects_trailing_or_noncanonical_bytes(mutation):
    spec, sources = fixture_data()
    artifacts = cli.execute(spec, sources)
    name = "training_rows.csv.gz"
    if mutation == "trailing":
        artifacts[name] += b"\0"
    elif mutation == "extra_member":
        artifacts[name] += gzip.compress(b"", mtime=0)
    elif mutation == "truncated":
        artifacts[name] = artifacts[name][:-1]
    else:
        data = bytearray(artifacts[name])
        data[4] = 1
        artifacts[name] = bytes(data)
    annotations, retrieval, folds, manifests = context(spec, sources)
    with pytest.raises(cli.FilterManifestError):
        cli.validate_artifacts(
            artifacts, spec, annotations, retrieval, folds, manifests,
            json.loads(cli.execute(spec, sources)["summary.json"]),
        )


@pytest.mark.parametrize(
    "mutation", ["weight", "digest", "position", "kernel_order", "source_index",
                 "summary_bool", "json_extra", "json_duplicate"],
)
def test_serialized_semantics_reject_wrong_weights_order_and_types(mutation):
    spec, sources = fixture_data()
    artifacts = cli.execute(spec, sources)
    original_summary = json.loads(artifacts["summary.json"])
    if mutation in ("weight", "digest", "json_extra", "json_duplicate"):
        head = json.loads(artifacts["head_plan.json"])
        if mutation == "weight":
            head["heads"][0]["class_stats"][0]["balanced_weight"] += 1
        elif mutation == "digest":
            head["heads"][0]["training_tile_ids_sha256"] = "0" * 64
        elif mutation == "json_extra":
            head["extra"] = "invalid"
        artifacts["head_plan.json"] = json_bytes(head)
        if mutation == "json_duplicate":
            artifacts["head_plan.json"] = artifacts["head_plan.json"].replace(
                b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1',
            )
    elif mutation == "summary_bool":
        summary = json.loads(artifacts["summary.json"])
        summary["counts"]["quarantine_slides"] = True
        artifacts["summary.json"] = json_bytes(summary)
    else:
        name = "patient_folds.csv" if mutation == "kernel_order" else "training_rows.csv.gz"
        compressed = name.endswith(".gz")
        data = gzip.decompress(artifacts[name]) if compressed else artifacts[name]
        rows = list(csv.reader(io.StringIO(data.decode(), newline="")))
        column = {"position": "training_position", "kernel_order": "kernel_order",
                  "source_index": "h_optimus_1_feature_row"}[mutation]
        rows[1][rows[0].index(column)] = "999"
        data = csv_bytes(rows[0], rows[1:])
        artifacts[name] = gzip.compress(data, mtime=0) if compressed else data
    annotations, retrieval, folds, manifests = context(spec, sources)
    with pytest.raises(cli.FilterManifestError):
        cli.validate_artifacts(
            artifacts, spec, annotations, retrieval, folds, manifests, original_summary,
        )


def test_streamed_large_gzip_roundtrip_has_bounded_framing_buffers():
    rows = ((index, "x" * 1000) for index in range(2000))
    data = cli._csv_bytes(("index", "value"), rows, compressed=True)
    parsed = cli._csv_rows(data, ("index", "value"), "large fixture", compressed=True)
    assert sum(1 for row in parsed if len(row[1]) == 1000) == 2000


def test_execute_rejects_its_own_faulty_serialized_training_bytes(monkeypatch):
    spec, sources = fixture_data()
    original = cli._csv_bytes
    def faulty(header, rows, *, compressed):
        data = original(header, rows, compressed=compressed)
        if header == cli.TRAINING_HEADER:
            return data + b"\0"
        return data
    monkeypatch.setattr(cli, "_csv_bytes", faulty)
    with pytest.raises(cli.FilterManifestError, match="trailing gzip"):
        cli.execute(spec, sources)


def native_env(out_root=None):
    env = dict(os.environ)
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    })
    if out_root is not None:
        env["PROSPER_OUT"] = str(out_root)
    return env


def test_normal_package_help_has_no_output_root_side_effect(tmp_path):
    out = tmp_path / "must_not_be_created"
    result = subprocess.run(
        [sys.executable, "-m", "prosper.evaluation.r2_1_hoptimus_filter_cli", "--help"],
        cwd=Path(cli.__file__).resolve().parents[2], env=native_env(out),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "--spec-sha256" in result.stdout and "--out-root" in result.stdout
    assert not out.exists()


def test_example_is_explicitly_unbound_and_has_only_released_schema():
    path = Path(cli.__file__).resolve().parents[2] / "examples" / "r2_1_hoptimus_filter.json"
    example = json.loads(path.read_bytes())
    assert set(example) == {
        "schema_version", "run_id", "task", "sources", "code_files", "parameters", "expected",
    }
    assert "REPLACE_ALL_BINDINGS" in example["run_id"]
    assert example["task"] == cli.TASK
    assert set(example["sources"]) == cli.SOURCE_NAMES
    assert set(example["expected"]) == cli.EXPECTED_KEYS
    assert set(example["code_files"]) == {
        "runner", "kernel", "io", "legacy_io", "package_init", "evaluation_init",
    }
    assert all(set(entry) == {"path", "size", "sha256"} for entry in example["sources"].values())
    assert all(entry["size"] == 0 for entry in example["sources"].values())
    assert example["code_files"]["kernel"]["sha256"] == digest(Path(kernel.__file__).read_bytes())


def shared_available():
    return importlib.util.find_spec("prosper.evaluation.r2_1_hoptimus_io") is not None


def make_bound_fixture(tmp_path):
    """Write only synthetic declared inputs; never manufacture the shared guard."""
    from prosper.evaluation import feature_bank_audit as legacy
    from prosper.evaluation import r2_1_hoptimus_io as shared
    import prosper
    from prosper import evaluation

    spec, sources = fixture_data()
    code = {
        "runner": Path(cli.__file__), "kernel": Path(kernel.__file__),
        "io": Path(shared.__file__), "legacy_io": Path(legacy.__file__),
        "package_init": Path(prosper.__file__), "evaluation_init": Path(evaluation.__file__),
    }
    paths = {}
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    filenames = {
        "stage0_complete": "COMPLETE.json", "stage0_inventory": "inventory.csv",
        "stage0_summary": "summary.json", "stage1a_complete": "COMPLETE.json",
        "stage1a_eligible": "eligible_tiles.csv",
        "stage1a_feature_inventory": "feature_inventory.csv",
        "stage1a_summary": "summary.json",
    }
    for name in sorted(cli.SOURCE_NAMES):
        if name.startswith("stage0_") and name not in ("stage0_review",):
            path = inputs / "stage0" / "attempt" / filenames[name]
        elif name.startswith("stage1a_") and name not in ("stage1a_review",):
            path = inputs / "stage1a" / "attempt" / filenames[name]
        elif name in ("stage0_review", "stage1a_review"):
            path = inputs / name.split("_")[0] / (name + ".json")
        else:
            path = inputs / (name + ".input")
        path.parent.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    sources["kernel_review"] = json_bytes({
        "schema_version": 1, "disposition": "accepted",
        "scope": "Pure filtering fold/training-manifest kernels only",
        "source_files": {"prosper/evaluation/r2_1_hoptimus_filter.py": digest(
            code["kernel"].read_bytes(),
        )},
    })
    def binding(name, kind, recorded_name):
        path = paths[name]
        data = sources[name]
        return {
            "kind": kind, "name": recorded_name, "path": str(path),
            "resolved_path": str(path.resolve()), "size": len(data),
            "sha256": digest(data), "device": 987, "inode": 456,
        }
    def historical_spec_binding(stage):
        path = inputs / stage / f"{stage}_spec.json"
        return {
            "kind": "spec", "name": "spec", "path": str(path),
            "resolved_path": str(path.resolve()), "size": 10, "sha256": "b" * 64,
            "device": 123, "inode": 321,
        }
    metadata = cli._metadata(sources["metadata"])
    inventory_header = (
        "bank", "slide_id", "canonical_tiles", "feature_rows", "dtype",
        "feature_path", "resolved_path", "size", "sha256",
    )
    inventory_rows = [
        (bank, row.slide_id, 3, 3, "float32", f"/unread/{bank}/{row.slide_id}.pkl",
         f"/unread/{bank}/{row.slide_id}.pkl", 100, "a" * 64)
        for bank in ("uni2", "virchow2", "h_optimus_1") for row in metadata
    ]
    sources["stage0_inventory"] = csv_bytes(inventory_header, inventory_rows)
    legacy_data = code["legacy_io"].read_bytes()
    legacy_binding = {
        "kind": "code", "name": "running_auditor", "path": str(code["legacy_io"]),
        "resolved_path": str(code["legacy_io"].resolve()), "size": len(legacy_data),
        "sha256": digest(legacy_data), "device": 123, "inode": 456,
    }
    sources["stage0_summary"] = json_bytes({
        "schema_version": 1, "run_id": "stage0_fixture",
        "scope": "feature-bank binding only; encoder identities declared/unverified",
        "canonical_slides": len(metadata), "canonical_tiles": len(metadata) * 3,
        "inventory_rows": len(inventory_rows),
        "metadata_columns": ["Tumor.group", "Slide.Scan.File", "Sample.ID", "UID.Subject"],
        "declarations": {"run_id": "stage0_fixture", "metadata": {
            "path": str(paths["metadata"]), **cli.METADATA_COLUMNS,
        }, "banks": [{"name": name, "feature_dim": width}
                     for name, width in (("uni2", 1536), ("virchow2", 1280),
                                         ("h_optimus_1", 1536))]},
        "input_bindings": [
            binding("metadata", "metadata", "metadata"), historical_spec_binding("stage0"),
            legacy_binding,
        ],
    })
    def complete(stage, scope, names):
        return json_bytes({
            "schema_version": 1, "run_id": f"{stage}_fixture", "scope": scope,
            "outputs": {paths[name].name: {"sha256": digest(sources[name]),
                                         "size": len(sources[name])} for name in names},
        })
    sources["stage0_complete"] = complete(
        "stage0", "feature-bank binding only", ("stage0_inventory", "stage0_summary"),
    )
    sources["stage0_review"] = json_bytes({
        "schema_version": 1, "disposition": "accepted", "scope": "Stage0 feature-bank binding only",
        "artifacts": {
            **{str(paths[name].relative_to(paths["stage0_review"].parent)):
               digest(sources[name]) for name in
               ("stage0_complete", "stage0_inventory", "stage0_summary")},
            "stage0_spec.json": "b" * 64,
        },
        "source_sha256": {
            "metadata": digest(sources["metadata"]),
            "auditor_module": digest(code["legacy_io"].read_bytes()),
        },
    })
    stage1a_summary = json.loads(sources["stage1a_summary"])
    stage1a_summary["input_bindings"] = [
        binding("metadata", "metadata", "metadata"),
        binding("stage0_review", "stage0", "stage0_review"),
        historical_spec_binding("stage1a"),
    ]
    stage1a_summary["declarations"].update({
        "run_id": "stage1a_fixture",
        "metadata": {"path": "../metadata.input", **cli.METADATA_COLUMNS},
        "stage0": {"review": "../stage0/stage0_review.json"},
    })
    sources["stage1a_summary"] = json_bytes(stage1a_summary)
    names = ("stage1a_eligible", "stage1a_feature_inventory", "stage1a_summary")
    stage1a_complete = json.loads(complete(
        "stage1a", "historical-annotation target keyed binding only", names,
    ))
    historical_outputs = ("quarantine_tiles.csv", "slide_inventory.csv", "diagnostics.json")
    for name in historical_outputs:
        stage1a_complete["outputs"][name] = {"sha256": "c" * 64, "size": 12}
    sources["stage1a_complete"] = json_bytes(stage1a_complete)
    review = json.loads(sources["stage1a_review"])
    review["artifacts"] = {
        str(paths[name].relative_to(paths["stage1a_review"].parent)): digest(sources[name])
        for name in ("stage1a_complete", *names)
    }
    review["artifacts"]["stage1a_spec.json"] = "b" * 64
    review["artifacts"].update({f"attempt/{name}": "c" * 64 for name in historical_outputs})
    review["source_sha256"] = {
        "metadata": digest(sources["metadata"]),
        "stage0_independent_review": digest(sources["stage0_review"]),
    }
    sources["stage1a_review"] = json_bytes(review)
    for name, data in sources.items():
        paths[name].write_bytes(data)
    spec["sources"] = {
        name: {"path": str(paths[name]), "sha256": digest(data), "size": len(data)}
        for name, data in sources.items()
    }
    spec["code_files"] = {
        name: {"path": str(path), "sha256": digest(path.read_bytes()),
               "size": path.stat().st_size} for name, path in code.items()
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(json_bytes(spec))
    return spec, paths, spec_path


@pytest.mark.skipif(not shared_available(), reason="shared run_bound module is not available yet")
def test_normal_package_tiny_cli_complete_and_no_reuse(tmp_path):
    spec, paths, spec_path = make_bound_fixture(tmp_path)
    out_root = tmp_path / "out"
    out_root.mkdir()
    target = out_root / "attempt"
    command = [
        sys.executable, "-m", "prosper.evaluation.r2_1_hoptimus_filter_cli",
        "--spec", str(spec_path), "--spec-sha256", digest(spec_path.read_bytes()),
        "--out", str(target), "--out-root", str(out_root),
    ]
    result = subprocess.run(
        command, cwd=Path(cli.__file__).resolve().parents[2], env=native_env(out_root),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    marker = json.loads((target / "COMPLETE.json").read_bytes())
    assert marker["scope"] == "hoptimus_filter_manifests_only"
    assert set(marker["outputs"]) == cli.ARTIFACT_NAMES | {"input_bindings.json"}
    for name, value in marker["outputs"].items():
        data = (target / name).read_bytes()
        assert value == {"size": len(data), "sha256": digest(data)}
    assert json.loads((target / "summary.json").read_bytes())["counts"] == spec["expected"]
    second = subprocess.run(
        command, cwd=Path(cli.__file__).resolve().parents[2], env=native_env(out_root),
        capture_output=True, text=True, timeout=120,
    )
    assert second.returncode != 0


@pytest.mark.skipif(not shared_available(), reason="shared run_bound module is not available yet")
@pytest.mark.parametrize("mutation", ["source_hash", "receipt_link", "extra_key",
                                      "duplicate_key", "wrong_count", "overlap"])
def test_normal_bound_cli_failure_without_complete(tmp_path, mutation):
    spec, paths, spec_path = make_bound_fixture(tmp_path)
    out_root = tmp_path / "out"
    out_root.mkdir()
    target = out_root / "failed"
    if mutation == "source_hash":
        paths["metadata"].write_bytes(paths["metadata"].read_bytes() + b"extra")
    elif mutation == "receipt_link":
        review = json.loads(paths["stage1a_review"].read_bytes())
        review["artifacts"][next(iter(review["artifacts"]))] = "0" * 64
        data = json_bytes(review)
        paths["stage1a_review"].write_bytes(data)
        spec["sources"]["stage1a_review"].update(sha256=digest(data), size=len(data))
    elif mutation == "extra_key":
        spec["extra"] = True
    elif mutation == "wrong_count":
        spec["expected"]["annotation_tiles"] += 1
    elif mutation == "overlap":
        target = paths["metadata"].parent
    spec_path.write_bytes(json_bytes(spec))
    if mutation == "duplicate_key":
        spec_path.write_bytes(spec_path.read_bytes().replace(
            b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1',
        ))
    result = subprocess.run(
        [sys.executable, "-m", "prosper.evaluation.r2_1_hoptimus_filter_cli",
         "--spec", str(spec_path), "--spec-sha256", digest(spec_path.read_bytes()),
         "--out", str(target), "--out-root", str(out_root)],
        cwd=Path(cli.__file__).resolve().parents[2], env=native_env(out_root),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0
    expected_context = {
        "source_hash": ("metadata",),
        "receipt_link": ("stage1a",),
        "extra_key": ("spec", "declaration"),
        "duplicate_key": ("duplicate",),
        "wrong_count": ("observed counts",),
        "overlap": ("output", "PROSPER_OUT", "overlap", "beneath"),
    }[mutation]
    assert any(context in result.stderr for context in expected_context), result.stderr
    assert not (target / "COMPLETE.json").exists()


@pytest.mark.skipif(not shared_available(), reason="shared run_bound module is not available yet")
def test_actual_shared_guard_rejects_late_synthetic_source_mutation(tmp_path, monkeypatch):
    _, paths, spec_path = make_bound_fixture(tmp_path)
    out_root = tmp_path / "out"
    out_root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(out_root))
    original = cli.execute
    mutated = False
    def late_mutation(spec, sources):
        nonlocal mutated
        artifacts = original(spec, sources)
        paths["metadata"].write_bytes(paths["metadata"].read_bytes() + b"late mutation")
        mutated = True
        return artifacts
    monkeypatch.setattr(cli, "execute", late_mutation)
    target = out_root / "failed_late"
    with pytest.raises(Exception, match="metadata|changed|mismatch"):
        cli.run(spec_path, digest(spec_path.read_bytes()), target, out_root)
    assert mutated
    assert not (target / "COMPLETE.json").exists()


@pytest.mark.skipif(not shared_available(), reason="shared run_bound module is not available yet")
def test_actual_shared_guard_rejects_post_readback_output_mutation(tmp_path, monkeypatch):
    """Fault injection retains every original shared read/output guard."""
    from prosper.evaluation import r2_1_hoptimus_io as shared
    _, _, spec_path = make_bound_fixture(tmp_path)
    out_root = tmp_path / "out"
    out_root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(out_root))
    target = out_root / "failed_output_mutation"
    original = shared._read_written
    mutated = False
    def late_mutation(fd, name, size):
        nonlocal mutated
        data = original(fd, name, size)
        if name == "training_rows.csv.gz" and not mutated:
            mutated = True
            path = target / name
            path.write_bytes(path.read_bytes() + b"late mutation")
        return data
    monkeypatch.setattr(shared, "_read_written", late_mutation)
    with pytest.raises(Exception, match="size|changed|mismatch|hash"):
        cli.run(spec_path, digest(spec_path.read_bytes()), target, out_root)
    assert mutated
    assert not (target / "COMPLETE.json").exists()
