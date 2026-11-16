"""Normal-package CLI integration and independent synthetic main-grid recounts."""

import io
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from prosper.evaluation import r2_1_hoptimus_io as bound
from prosper.evaluation import r2_1_hoptimus_main_cli as cli
from tests.test_r2_1_hoptimus_io import build_inputs, replace_source, run_fixture, save_spec, sha


def cli_environment(root):
    return dict(os.environ, PROSPER_OUT=str(root), PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1")


def test_normal_package_help_without_output_side_effect(tmp_path):
    out_root = tmp_path / "does-not-exist"
    result = subprocess.run(
        [sys.executable, "-B", "-m", "prosper.evaluation.r2_1_hoptimus_main_cli", "--help"],
        env=cli_environment(out_root), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "--spec-sha256" in result.stdout and "--out-root" in result.stdout
    assert not out_root.exists()


def test_normal_cli_spec_symlink_preserves_lexical_relative_source_base(tmp_path):
    fixture = build_inputs(tmp_path)
    alias_directory = tmp_path / "lexical_declaration"
    alias_directory.mkdir()
    metadata = Path(fixture["spec"]["sources"]["metadata"]["path"])
    alias_metadata = alias_directory / "metadata-bound.csv"
    alias_metadata.symlink_to(metadata)
    fixture["spec"]["sources"]["metadata"]["path"] = alias_metadata.name
    save_spec(fixture)
    alias_spec = alias_directory / "spec.json"
    alias_spec.symlink_to(fixture["spec_path"])
    assert not (fixture["spec_path"].parent / alias_metadata.name).exists()
    result = subprocess.run([
        sys.executable, "-B", "-m", "prosper.evaluation.r2_1_hoptimus_main_cli",
        "--spec", str(alias_spec), "--spec-sha256", fixture["sha"],
        "--out", str(fixture["out"]), "--out-root", str(fixture["root"])],
        env=cli_environment(fixture["root"]), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    bindings = bound.strict_json((fixture["out"] / "input_bindings.json").read_bytes())["bindings"]
    recorded_spec = next(row for row in bindings if row["kind"] == "spec")
    recorded_metadata = next(row for row in bindings if row["kind"] == "source"
                             and row["name"] == "metadata")
    assert recorded_spec["path"] == str(alias_spec)
    assert recorded_spec["resolved_path"] == str(fixture["spec_path"].resolve())
    assert recorded_metadata["path"] == str(alias_metadata)
    assert recorded_metadata["resolved_path"] == str(metadata.resolve())
    assert (fixture["out"] / "COMPLETE.json").is_file()


def test_tiny_normal_package_cli_complete_and_independent_recounts(tmp_path):
    fixture = build_inputs(tmp_path)
    result = subprocess.run([
        sys.executable, "-B", "-m", "prosper.evaluation.r2_1_hoptimus_main_cli",
        "--spec", str(fixture["spec_path"]), "--spec-sha256", fixture["sha"],
        "--out", str(fixture["out"]), "--out-root", str(fixture["root"])],
        env=cli_environment(fixture["root"]), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = bound.strict_json((fixture["out"] / "COMPLETE.json").read_bytes())
    for name, binding in marker["outputs"].items():
        data = (fixture["out"] / name).read_bytes()
        assert binding == {"sha256": sha(data), "size": len(data)}
    population = list(bound.csv_rows((fixture["out"] / "populations.csv").read_bytes(),
                                     cli.POPULATION_HEADER))
    assert [r["slide_id"] for r in population] == [str(i) for i in range(16)]
    metrics = list(bound.csv_rows((fixture["out"] / "query_metrics.csv.gz").read_bytes(),
                                  cli.QUERY_HEADER, compressed=True))
    assert len(metrics) == 4 * 16 * 4
    estimates = list(bound.csv_rows((fixture["out"] / "estimates.csv").read_bytes(),
                                    cli.ESTIMATE_HEADER))
    contrasts = list(bound.csv_rows((fixture["out"] / "contrasts.csv").read_bytes(),
                                    cli.ESTIMATE_HEADER))
    assert len(estimates) == 32 * 4
    assert len(contrasts) == 12 * 4
    points = {}
    for method in cli.METHODS:
        for endpoint, field in (("hit", "conditional_hit"), ("precision", "conditional_precision")):
            chosen = [r for r in metrics if r["method"] == method and r["k"] == "5"]
            group_points = {g: sum(float(r[field]) for r in chosen if r["diagnostic_group"] == g)
                            / sum(r["diagnostic_group"] == g for r in chosen) for g in ("A", "B", "C")}
            points[method, endpoint] = sum(group_points.values()) / 3
            row = next(r for r in estimates if r["method"] == method and r["k"] == "5"
                       and r["endpoint"] == endpoint and r["level"] == "macro")
            assert float(row["point"]) == pytest.approx(points[method, endpoint])
    for row in contrasts:
        if row["level"] == "macro":
            assert float(row["point"]) == pytest.approx(
                points[row["arm_a"], row["endpoint"]] - points[row["arm_b"], row["endpoint"]])
    ties = list(bound.csv_rows((fixture["out"] / "tie_sensitivity.csv.gz").read_bytes(),
                              cli.TIE_HEADER, compressed=True))
    first = ties[0]
    assert first["block_count"] == "14"  # The fifth-distance tie extends past rank 10.
    assert first["strict_count"] == "0" and first["block_relevant"] == "7"
    assert float(first["precision_expected"]) == 0.5
    assert float(first["hit_expected"]) == pytest.approx(1 - 21 / 2002)
    assert float(first["precision_lower"]) == 0 and float(first["precision_upper"]) == 1
    summary = bound.strict_json((fixture["out"] / "summary.json").read_bytes())
    assert summary["details"]["coverage"] == 1
    assert summary["details"]["matrix_diagnostics"]["yottixel"]["unused_nonfinite_entries"] > 0
    valid = np.load(fixture["out"] / "replicate_valid.npy", allow_pickle=False)
    reps = np.load(fixture["out"] / "replicates.npy", allow_pickle=False)
    assert valid.shape == reps.shape == (5000, 176)
    assert np.isnan(reps[~valid]).all()
    assert any(r["ci_status"] == "unavailable" and r["zero_support_replicates"] != "0"
               for r in estimates if r["group"] == "C")


@pytest.mark.parametrize("mutation", ["same_patient", "duplicate_rank", "wrong_label", "distance",
                                    "missing_row", "extra_row", "unknown_header", "gzip_trailing"])
def test_rank_integrity_failures_never_complete(tmp_path, monkeypatch, mutation):
    fixture = build_inputs(tmp_path)
    name = "ranks_median_pooling"
    data = Path(fixture["spec"]["sources"][name]["path"]).read_bytes()
    rows = list(bound.csv_rows(data, cli.RANK_HEADER, compressed=True))
    if mutation == "same_patient":
        rows[0]["neighbour"] = "1"
        rows[0]["neighbour_label"] = "B"
    elif mutation == "duplicate_rank":
        rows[1]["rank"] = rows[0]["rank"]
    elif mutation == "wrong_label":
        rows[0]["neighbour_label"] = "wrong"
    elif mutation == "distance":
        rows[0]["distance"] = "0.000001"
    elif mutation == "missing_row":
        rows.pop()
    elif mutation == "extra_row":
        rows.append(rows[0])
    elif mutation == "unknown_header":
        data = bound.csv_bytes((*cli.RANK_HEADER, "extra"),
                               ((*[r[k] for k in cli.RANK_HEADER], "x") for r in rows), compressed=True)
    else:
        data += b"trailing corrupt gzip"
    if mutation not in ("unknown_header", "gzip_trailing"):
        data = bound.csv_bytes(cli.RANK_HEADER, rows, compressed=True)
    replace_source(fixture, name, data)
    with pytest.raises(Exception):
        run_fixture(fixture, monkeypatch, cli.execute)
    assert not (fixture["out"] / "COMPLETE.json").exists()


@pytest.mark.parametrize("mutation", ["float_axis", "bool_axis", "duplicate_axis", "used_nan",
                                    "wrong_population", "symmetrize", "trailing_pickle", "integer_dtype"])
def test_matrix_population_numeric_and_directed_integrity(tmp_path, monkeypatch, mutation):
    fixture = build_inputs(tmp_path)
    name = "matrix_yottixel"
    data = Path(fixture["spec"]["sources"][name]["path"]).read_bytes()
    matrix = pickle.loads(data)
    if mutation == "float_axis":
        matrix.index = matrix.index.astype(float)
    elif mutation == "bool_axis":
        matrix.index = [False, True] + list(matrix.index[2:])
    elif mutation == "duplicate_axis":
        matrix.index = [matrix.index[1]] + list(matrix.index[1:])
    elif mutation == "used_nan":
        matrix.loc[2, 3] = np.nan
    elif mutation == "wrong_population":
        matrix.index = ["unknown"] + list(matrix.index[1:])
    elif mutation == "symmetrize":
        matrix.iloc[:, :] = (matrix.to_numpy() + matrix.to_numpy().T) / 2
    elif mutation == "integer_dtype":
        matrix = matrix.fillna(0).replace([np.inf], 0).astype(int)
    if mutation == "trailing_pickle":
        data += b"unbound trailing bytes"
    else:
        data = pickle.dumps(matrix, protocol=5)
    replace_source(fixture, name, data)
    with pytest.raises(Exception):
        run_fixture(fixture, monkeypatch, cli.execute)
    assert not (fixture["out"] / "COMPLETE.json").exists()


@pytest.mark.parametrize("payload", ["trailing", "object", "fortran", "wrong_dtype", "wrong_shape"])
def test_npy_numeric_shape_dtype_order_and_trailing_guards(payload):
    array = np.zeros((2, 3), dtype="<f8")
    if payload == "object":
        buffer = io.BytesIO()
        np.save(buffer, array.astype(object))
        data = buffer.getvalue()
    elif payload == "fortran":
        buffer = io.BytesIO()
        np.lib.format.write_array(buffer, np.asfortranarray(array), version=(1, 0), allow_pickle=False)
        data = buffer.getvalue()
    elif payload == "wrong_dtype":
        data = cli._npy(array, "<f4")
    elif payload == "wrong_shape":
        data = cli._npy(np.zeros((3, 2)), "<f8")
    else:
        data = cli._npy(array, "<f8") + b"trailing"
    with pytest.raises(bound.BoundError):
        cli._load_npy(data, (2, 3), "<f8", "fixture")


@pytest.mark.parametrize("mutation", ["query", "estimate", "contrast", "index", "draws",
                                    "replicates", "validity", "ties", "summary",
                                    "coverage_bool", "cutoff_bool", "matrix_count_bool",
                                    "tie_lower_bool", "count_bool", "bootstrap_float",
                                    "empty_limitations", "missing_limitations",
                                    "short_limitations", "extra_limitations"])
def test_serialized_semantic_readback_detects_corruption(tmp_path, mutation):
    fixture = build_inputs(tmp_path)
    sources = {name: Path(entry["path"]).read_bytes()
               for name, entry in fixture["spec"]["sources"].items()}
    artifacts = cli.execute(fixture["spec"], sources)
    raw, target, groups, _ = cli._metadata(fixture["spec"], sources)
    relevances, ties, diagnostics = {}, [], {}
    for method in cli.METHODS:
        rel, rows, diagnostic = cli._matrix(method, sources, raw, target)
        relevances[method] = rel
        ties.extend(rows)
        diagnostics[method] = diagnostic
    if mutation in ("query", "estimate", "contrast", "index", "ties"):
        name, header, compressed = {
            "query": ("query_metrics.csv.gz", cli.QUERY_HEADER, True),
            "estimate": ("estimates.csv", cli.ESTIMATE_HEADER, False),
            "contrast": ("contrasts.csv", cli.ESTIMATE_HEADER, False),
            "index": ("replicate_index.csv", cli.INDEX_HEADER, False),
            "ties": ("tie_sensitivity.csv.gz", cli.TIE_HEADER, True),
        }[mutation]
        rows = list(bound.csv_rows(artifacts[name], header, compressed=compressed))
        if mutation == "query":
            rows[0]["conditional_precision"] = "0.125"
        elif mutation in ("estimate", "contrast"):
            rows[0]["point"] = "0.125"
        elif mutation == "index":
            rows.pop()
        else:
            rows[0]["block_count"] = "10"
        artifacts[name] = bound.csv_bytes(header, rows, compressed=compressed)
    elif mutation in ("summary", "coverage_bool", "cutoff_bool", "matrix_count_bool",
                      "tie_lower_bool", "count_bool", "bootstrap_float", "empty_limitations",
                      "missing_limitations", "short_limitations", "extra_limitations"):
        summary = bound.strict_json(artifacts["summary.json"])
        if mutation == "summary":
            summary["details"]["tie_summaries"][0]["expected"] = 0.125
        elif mutation == "coverage_bool":
            summary["details"]["coverage"] = True
        elif mutation == "cutoff_bool":
            summary["policy"]["cutoff_order"][0] = True
        elif mutation == "matrix_count_bool":
            assert summary["details"]["matrix_diagnostics"]["median_pooling"]["used_1e6_entries"] == 0
            summary["details"]["matrix_diagnostics"]["median_pooling"]["used_1e6_entries"] = False
        elif mutation == "tie_lower_bool":
            row = next(r for r in summary["details"]["tie_summaries"] if r["lower"] == 0)
            row["lower"] = False
        elif mutation == "count_bool":
            assert summary["counts"]["cross_label_patients"] == 1
            summary["counts"]["cross_label_patients"] = True
        elif mutation == "bootstrap_float":
            summary["policy"]["bootstrap"]["count"] = 5000.0
        elif mutation == "empty_limitations":
            summary["limitations"] = []
        elif mutation == "missing_limitations":
            summary.pop("limitations")
        elif mutation == "short_limitations":
            summary["limitations"].pop()
        else:
            summary["limitations"].append("Unsupported extra assertion")
        artifacts["summary.json"] = bound.json_bytes(summary)
    else:
        name, dtype = {
            "draws": ("patient_draws.npy", "<i8"),
            "replicates": ("replicates.npy", "<f8"),
            "validity": ("replicate_valid.npy", "|b1"),
        }[mutation]
        array = np.load(io.BytesIO(artifacts[name]), allow_pickle=False)
        if mutation == "draws":
            array[0, 0] += 1
        elif mutation == "replicates":
            array[0, 0] = 0.125
        else:
            array[0, 0] = not array[0, 0]
        artifacts[name] = cli._npy(array, dtype)
    with pytest.raises(bound.BoundError):
        cli.validate_artifacts(artifacts, fixture["spec"], target, groups, relevances, ties, diagnostics)
