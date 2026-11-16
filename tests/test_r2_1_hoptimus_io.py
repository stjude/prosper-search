"""Synthetic declaration, bounded-IO and receipt-chain tests."""

import hashlib
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest

from prosper.evaluation import r2_1_hoptimus_io as bound
from prosper.evaluation import r2_1_hoptimus_main_cli as cli


def sha(data):
    return hashlib.sha256(data).hexdigest()


def entry(path):
    data = path.read_bytes()
    return {"path": str(path), "sha256": sha(data), "size": len(data)}


def historical_binding(path, kind, name):
    return dict(kind=kind, name=name, path=str(path), resolved_path=str(path.resolve()),
                sha256=sha(path.read_bytes()), size=path.stat().st_size, device=999, inode=777)


def save_spec(fixture):
    fixture["spec_path"].write_bytes(bound.json_bytes(fixture["spec"]))
    fixture["sha"] = sha(fixture["spec_path"].read_bytes())
    return fixture


def replace_source(fixture, name, data):
    path = Path(fixture["spec"]["sources"][name]["path"])
    path.write_bytes(data)
    fixture["spec"]["sources"][name] = entry(path)
    return save_spec(fixture)


def build_inputs(tmp_path, task="hoptimus_main"):
    inputs, root = tmp_path / "inputs", tmp_path / "outputs"
    inputs.mkdir()
    root.mkdir()
    metadata_path = inputs / "metadata.csv"
    metadata = [(i, str(i), "p0" if i < 2 else f"p{i}",
                 "MISC" if i == 16 else "C" if i == 15 else "A" if i % 2 == 0 else "B")
                for i in range(17)]
    metadata_path.write_bytes(bound.csv_bytes(
        cli.METADATA_HEADER, ((g, q, f"sample{i}", p) for i, q, p, g in metadata)))
    code_paths = cli.code_paths()
    if task != "hoptimus_main":
        from prosper.evaluation import r2_1_hoptimus_filter, r2_1_hoptimus_perturb
        code_paths["kernel"] = (r2_1_hoptimus_filter.__file__
                               if task == "hoptimus_filter_manifest"
                               else r2_1_hoptimus_perturb.__file__)
    upstream = inputs / "upstream"
    attempt = upstream / "attempt"
    attempt.mkdir(parents=True)
    declarations = {
        "schema_version": 1, "run_id": "stage0-fixture",
        "metadata": dict(bound.METADATA_COLUMNS, path="../metadata.csv"),
        "banks": [{"name": name, "feature_dim": width}
                  for name, width in (("uni2", 1536), ("virchow2", 1280), ("h_optimus_1", 1536))],
    }
    original_spec = upstream / "stage0_spec.json"
    original_spec.write_bytes(bound.json_bytes(declarations))
    inventory_path = attempt / "inventory.csv"
    inventory_path.write_bytes(bound.csv_bytes(cli.INVENTORY_HEADER, (
        (b["name"], q, 1, 1, "float32", f"/never-read/{b['name']}/{q}.pkl",
         f"/never-read/{b['name']}/{q}.pkl", 100, sha(q.encode()))
        for b in declarations["banks"] for _, q, _, _ in metadata)))
    summary = {
        "schema_version": 1, "run_id": "stage0-fixture",
        "scope": "feature-bank binding only; encoder identities declared/unverified",
        "canonical_slides": 17, "canonical_tiles": 17, "inventory_rows": 51,
        "bank_tile_totals": {b["name"]: 17 for b in declarations["banks"]},
        "declarations": declarations,
        "input_bindings": [
            historical_binding(original_spec, "spec", "spec"),
            historical_binding(metadata_path, "metadata", "metadata"),
            historical_binding(Path(code_paths["legacy_io"]), "code", "running_auditor"),
        ],
    }
    summary_path = attempt / "summary.json"
    summary_path.write_bytes(bound.json_bytes(summary))
    complete_path = attempt / "COMPLETE.json"
    complete_path.write_bytes(bound.json_bytes({
        "schema_version": 1, "run_id": "stage0-fixture", "scope": "feature-bank binding only",
        "outputs": {p.name: {"sha256": sha(p.read_bytes()), "size": p.stat().st_size}
                    for p in (inventory_path, summary_path)},
    }))
    review_path = upstream / "stage0_review.json"
    review_path.write_bytes(bound.json_bytes({
        "schema_version": 1, "disposition": "accepted", "scope": "Stage0 feature-bank binding only",
        "artifacts": {str(p.relative_to(upstream)): sha(p.read_bytes())
                      for p in (inventory_path, summary_path, complete_path, original_spec)},
        "source_sha256": {"metadata": sha(metadata_path.read_bytes()),
                          "auditor_module": sha(Path(code_paths["legacy_io"]).read_bytes())},
    }))
    kernel_review = inputs / "kernel_review.json"
    kernel_review.write_bytes(bound.json_bytes({
        "schema_version": 1, "disposition": "accepted", "scope": bound.KERNEL_SCOPES[task],
        "source_files": {"prosper/evaluation/" + Path(code_paths["kernel"]).name:
                         sha(Path(code_paths["kernel"]).read_bytes())},
    }))
    source_paths = dict(metadata=metadata_path, stage0_review=review_path,
                        stage0_complete=complete_path, stage0_inventory=inventory_path,
                        stage0_summary=summary_path, kernel_review=kernel_review)
    expected = dict(raw_slides=17, target_slides=16, patients=15, groups=3, cross_label_patients=1)
    if task == "hoptimus_main":
        target = metadata[:16]
        for method_index, method in enumerate(bound.METHODS):
            matrix = np.array([[(j + 3 * method_index) % 17 / 17 + i / 1000
                                for j in range(17)] for i in range(17)], dtype=float)
            matrix[0, 2:16] = 0.0
            for i in range(17):
                for j in range(17):
                    if metadata[i][2] == metadata[j][2]:
                        matrix[i, j] = np.inf
            matrix[16, :] = np.nan
            reversed_axes = np.arange(16, -1, -1)
            frame = pd.DataFrame(matrix[np.ix_(reversed_axes, reversed_axes)],
                                 index=reversed_axes, columns=reversed_axes)
            path = inputs / f"matrix_{method}.pkl"
            path.write_bytes(pickle.dumps(frame, protocol=5))
            source_paths[f"matrix_{method}"] = path
            ranks = []
            for i, q, p, g in target:
                candidates = [j for j in range(16) if target[j][2] != p]
                ordered = sorted(candidates, key=lambda j: matrix[i, j])[:10]
                ranks.extend((q, g, rank, target[j][1], target[j][3], float(matrix[i, j]))
                             for rank, j in enumerate(ordered, 1))
            path = inputs / f"ranks_{method}.csv.gz"
            path.write_bytes(bound.csv_bytes(cli.RANK_HEADER, ranks, compressed=True))
            source_paths[f"ranks_{method}"] = path
    elif task == "hoptimus_perturb_manifest":
        path = inputs / "template.bin"
        path.write_bytes(b"synthetic template bytes; never decoded")
        source_paths["template"] = path
        expected.update(stimulus_records=192, feature_dim=1536)
    else:
        annotation = inputs / "annotation"
        attempt1 = annotation / "attempt"
        attempt1.mkdir(parents=True)
        declarations1 = {
            "run_id": "stage1a-fixture", "metadata": dict(bound.METADATA_COLUMNS, path="../metadata.csv"),
            "stage0": {"review": "../upstream/stage0_review.json"},
        }
        spec1 = annotation / "stage1a_spec.json"
        spec1.write_bytes(bound.json_bytes(declarations1))
        summary1 = {
            "schema_version": 1, "run_id": "stage1a-fixture",
            "scope": "historical-annotation target keyed binding only",
            "declarations": declarations1, "input_bindings": [
                historical_binding(spec1, "spec", "spec"),
                historical_binding(metadata_path, "metadata", "metadata"),
                historical_binding(review_path, "stage0", "stage0_review"),
            ],
        }
        files = {}
        for name, data in (("eligible_tiles.csv", b"synthetic\n"),
                           ("feature_inventory.csv", b"synthetic\n"),
                           ("summary.json", bound.json_bytes(summary1))):
            path = attempt1 / name
            path.write_bytes(data)
            files[name] = path
        outputs = {name: {"sha256": sha(p.read_bytes()), "size": p.stat().st_size}
                   for name, p in files.items()}
        outputs.update({name: {"sha256": "0" * 64, "size": 0}
                        for name in ("quarantine_tiles.csv", "slide_inventory.csv", "diagnostics.json")})
        complete1 = attempt1 / "COMPLETE.json"
        complete1.write_bytes(bound.json_bytes({
            "schema_version": 1, "run_id": "stage1a-fixture",
            "scope": "historical-annotation target keyed binding only", "outputs": outputs,
        }))
        review1 = annotation / "stage1a_review.json"
        artifacts1 = {str(p.relative_to(annotation)): sha(p.read_bytes())
                      for p in (*files.values(), spec1, complete1)}
        artifacts1.update({f"attempt/{name}": "0" * 64
                           for name in ("quarantine_tiles.csv", "slide_inventory.csv", "diagnostics.json")})
        review1.write_bytes(bound.json_bytes({
            "schema_version": 1, "disposition": "accepted",
            "scope": "Stage1a historical-annotation target keyed binding only",
            "artifacts": artifacts1,
            "source_sha256": {"metadata": sha(metadata_path.read_bytes()),
                              "stage0_independent_review": sha(review_path.read_bytes())},
        }))
        source_paths.update(stage1a_review=review1, stage1a_complete=complete1,
                            stage1a_summary=files["summary.json"],
                            stage1a_eligible=files["eligible_tiles.csv"],
                            stage1a_feature_inventory=files["feature_inventory.csv"])
        expected.update(annotation_tiles=5, annotation_slides=5, annotation_patients=5,
                        quarantine_tiles=1, quarantine_slides=1)
    fixture = {
        "root": root, "out": root / "run", "spec_path": inputs / "spec.json",
        "code_paths": code_paths, "metadata": metadata,
        "spec": dict(schema_version=1, run_id="tiny-fixture", task=task,
                     sources={k: entry(p) for k, p in source_paths.items()},
                     code_files={k: entry(Path(p)) for k, p in code_paths.items()},
                     parameters={"metadata": bound.METADATA_COLUMNS.copy()}, expected=expected),
    }
    return save_spec(fixture)


def dummy_execute(spec, sources):
    assert set(sources) == set(spec["sources"])
    with pytest.raises(TypeError):
        sources["new"] = b"no"
    artifacts = {name: b"synthetic artifact\n" for name in bound.ARTIFACTS[spec["task"]]}
    artifacts["summary.json"] = bound.json_bytes({
        "schema_version": 1, "run_id": spec["run_id"], "task": spec["task"],
        "counts": spec["expected"], "policy": {}, "details": {}, "limitations": [],
    })
    return artifacts


def run_fixture(fixture, monkeypatch, execute=dummy_execute):
    monkeypatch.setenv("PROSPER_OUT", str(fixture["root"]))
    return bound.run_bound(
        fixture["spec_path"], fixture["sha"], fixture["out"], fixture["root"],
        task=fixture["spec"]["task"], code_paths=fixture["code_paths"], execute=execute)


@pytest.mark.parametrize("task", list(bound.ARTIFACTS))
def test_common_receipts_all_three_tasks_and_historical_cross_node_bindings(tmp_path, monkeypatch, task):
    fixture = build_inputs(tmp_path, task)
    marker = run_fixture(fixture, monkeypatch)
    assert set(marker) == {"schema_version", "run_id", "task", "scope", "spec_sha256",
                           "outputs", "verified_at", "limitations"}
    assert marker["scope"] == bound.SCOPES[task]
    assert set(marker["outputs"]) == bound.ARTIFACTS[task] | {"input_bindings.json"}
    assert set(p.name for p in fixture["out"].iterdir()) == set(marker["outputs"]) | {"COMPLETE.json"}
    for name, binding in marker["outputs"].items():
        data = (fixture["out"] / name).read_bytes()
        assert binding == {"sha256": sha(data), "size": len(data)}
    bindings = bound.strict_json((fixture["out"] / "input_bindings.json").read_bytes())
    assert [b["kind"] for b in bindings["bindings"]] == (
        ["spec"] + ["source"] * len(fixture["spec"]["sources"]) + ["code"] * 6)
    assert [b["name"] for b in bindings["bindings"] if b["kind"] == "code"] == sorted(bound.CODE_NAMES)
    assert len(bindings["executing_code_paths"]) == 6
    assert bindings["environment"]["numpy"] == np.__version__


@pytest.mark.parametrize("payload", [
    b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{} trailing',
    b'{"nested":[1e400]}',
])
def test_strict_json_rejects_duplicates_nonfinite_and_trailing(payload):
    with pytest.raises(bound.BoundError):
        bound.strict_json(payload)


def test_streaming_csv_and_fixed_gzip_metadata():
    data = bound.csv_bytes(("id", "value"), ((str(i), i / 7) for i in range(20)), compressed=True)
    assert data[4:8] == b"\x00" * 4
    rows = list(bound.csv_rows(data, ("id", "value"), compressed=True))
    assert len(rows) == 20
    assert float(rows[13]["value"]) == 13 / 7
    with pytest.raises(bound.BoundError, match="expansion"):
        list(bound.csv_rows(data, ("id", "value"), compressed=True, max_expanded=20))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, np.inf, None])
def test_csv_expansion_budget_must_be_finite_positive_integer(limit):
    data = bound.csv_bytes(("id",), [("value",)], compressed=True)
    with pytest.raises(bound.BoundError, match="expansion limit"):
        list(bound.csv_rows(data, ("id",), compressed=True, max_expanded=limit))


def test_generated_csv_uses_explicit_artifact_budget_not_rank_limit():
    rows = [("q", "v" * 70)]
    plain = bound.csv_bytes(("id", "value"), rows)
    compressed = bound.csv_bytes(("id", "value"), rows, compressed=True)
    with pytest.raises(bound.BoundError, match="expansion"):
        list(bound.csv_rows(compressed, ("id", "value"), compressed=True, max_expanded=64))
    decoded = list(bound.csv_rows(
        compressed, ("id", "value"), compressed=True, max_expanded=len(plain)))
    assert decoded == [{"id": "q", "value": "v" * 70}]


@pytest.mark.parametrize("payload", [
    b"a,a\n1,2\n", b"a,b,c\n1,2,3\n", b"a,b\n1\n", b"a,b\n1,2\n\n",
    b'a,b\n"unterminated,2\n', b"a,b\n\xff,2\n",
])
def test_csv_strict_headers_records_and_utf8(payload):
    with pytest.raises(bound.BoundError):
        list(bound.csv_rows(payload, ("a", "b")))


@pytest.mark.parametrize("change", [
    lambda s: s.update(extra=1),
    lambda s: s.update(schema_version=True),
    lambda s: s["parameters"].update(seed=123),
    lambda s: s["expected"].update(raw_slides=True),
    lambda s: s["sources"]["metadata"].update(sha256="bad"),
    lambda s: s["sources"]["metadata"].update(size=True),
    lambda s: s["sources"]["metadata"].update(size=bound.SOURCE_LIMIT + 1),
    lambda s: s["sources"].update(undeclared=s["sources"]["metadata"]),
    lambda s: s["code_files"].pop("io"),
])
def test_invalid_declarations_never_complete(tmp_path, monkeypatch, change):
    fixture = build_inputs(tmp_path)
    change(fixture["spec"])
    save_spec(fixture)
    with pytest.raises(bound.BoundError):
        run_fixture(fixture, monkeypatch)
    assert not (fixture["out"] / "COMPLETE.json").exists()


def test_wrong_source_hash_and_actual_executing_code_path(tmp_path, monkeypatch):
    fixture = build_inputs(tmp_path)
    fixture["spec"]["sources"]["metadata"]["sha256"] = "0" * 64
    save_spec(fixture)
    with pytest.raises(bound.BoundError, match="SHA256"):
        run_fixture(fixture, monkeypatch)
    fixture = replace_source(fixture, "metadata", Path(
        fixture["spec"]["sources"]["metadata"]["path"]).read_bytes())
    fixture["code_paths"] = dict(fixture["code_paths"], runner=__file__)
    with pytest.raises(bound.BoundError, match="actual executing"):
        run_fixture(fixture, monkeypatch)


@pytest.mark.parametrize("source", ["stage0_review", "stage0_complete", "kernel_review"])
def test_wrong_review_scope_or_run_links(tmp_path, monkeypatch, source):
    fixture = build_inputs(tmp_path)
    value = bound.strict_json(Path(fixture["spec"]["sources"][source]["path"]).read_bytes())
    value["scope"] = "unaccepted substitute"
    replace_source(fixture, source, bound.json_bytes(value))
    with pytest.raises(bound.BoundError):
        run_fixture(fixture, monkeypatch)


def test_review_artifact_link_containment(tmp_path, monkeypatch):
    fixture = build_inputs(tmp_path)
    review = bound.strict_json(Path(fixture["spec"]["sources"]["stage0_review"]["path"]).read_bytes())
    review["artifacts"]["../escape.json"] = "0" * 64
    replace_source(fixture, "stage0_review", bound.json_bytes(review))
    with pytest.raises(bound.BoundError, match="unsafe"):
        run_fixture(fixture, monkeypatch)


def test_output_reuse_and_input_overlap_are_refused(tmp_path, monkeypatch):
    fixture = build_inputs(tmp_path)
    run_fixture(fixture, monkeypatch)
    with pytest.raises(FileExistsError):
        run_fixture(fixture, monkeypatch)
    fixture["out"] = fixture["spec_path"].parent
    fixture["root"] = tmp_path
    with pytest.raises(Exception, match="overlaps"):
        run_fixture(fixture, monkeypatch)


@pytest.mark.parametrize("mutation", ["extra", "nonbytes", "callback_failure"])
def test_bad_callbacks_leave_failure_not_complete(tmp_path, monkeypatch, mutation):
    fixture = build_inputs(tmp_path)
    def callback(spec, sources):
        result = dummy_execute(spec, sources)
        if mutation == "extra":
            result["COMPLETE.json"] = b"not permitted"
        elif mutation == "nonbytes":
            result["populations.csv"] = "not bytes"
        else:
            raise RuntimeError("synthetic callback failure")
        return result
    with pytest.raises(Exception):
        run_fixture(fixture, monkeypatch, callback)
    assert (fixture["out"] / "FAILED.json").exists()
    assert not (fixture["out"] / "COMPLETE.json").exists()


@pytest.mark.parametrize("mutation", ["source", "output", "extra_output", "code", "spec"])
def test_final_sweep_and_then_output_hash_allowlist_guard(tmp_path, monkeypatch, mutation):
    fixture = build_inputs(tmp_path)
    original = bound._scan_file
    mutated = False
    def scan(path, kind, name, *args, **kwargs):
        nonlocal mutated
        if not mutated and fixture["out"].exists() and kwargs.get("retain") is False:
            mutated = True
            if mutation == "source":
                p = Path(fixture["spec"]["sources"]["metadata"]["path"])
                p.write_bytes(p.read_bytes().replace(b"sample0", b"sampleX", 1))
            elif mutation == "output":
                p = fixture["out"] / "populations.csv"
                p.write_bytes(p.read_bytes().replace(b"synthetic", b"tampered!", 1))
            elif mutation == "extra_output":
                (fixture["out"] / ".abandoned.part").write_bytes(b"partial")
            elif mutation == "spec":
                fixture["spec_path"].write_bytes(fixture["spec_path"].read_bytes() + b" ")
            else:
                if kind == "code":
                    _, binding = original(path, kind, name, *args, **kwargs)
                    binding["sha256"] = "0" * 64
                    return None, binding
                mutated = False
        return original(path, kind, name, *args, **kwargs)
    monkeypatch.setattr(bound, "_scan_file", scan)
    with pytest.raises(Exception):
        run_fixture(fixture, monkeypatch)
    assert not (fixture["out"] / "COMPLETE.json").exists()


def test_marker_full_readback_failure_removes_complete(tmp_path, monkeypatch):
    fixture = build_inputs(tmp_path)
    original = bound.legacy._atomic_bytes
    def write(fd, name, data):
        if name == "COMPLETE.json":
            data = data.replace(b'"schema_version":1', b'"schema_version":0')
        return original(fd, name, data)
    monkeypatch.setattr(bound.legacy, "_atomic_bytes", write)
    with pytest.raises(bound.BoundError, match="COMPLETE"):
        run_fixture(fixture, monkeypatch)
    assert not (fixture["out"] / "COMPLETE.json").exists()


def test_read_limits_do_not_trust_file_or_declaration_sizes(tmp_path):
    path = tmp_path / "bounded.bin"
    path.write_bytes(b"1234")
    with pytest.raises(bound.BoundError, match="bounded"):
        bound._scan_file(path, "source", "tiny", 3)
    with pytest.raises(bound.BoundError, match="size"):
        bound._scan_file(path, "source", "tiny", 10, expected={"size": 3, "sha256": sha(b"123")})


def test_original_stage1a_spec_origin_not_summary_directory(tmp_path, monkeypatch):
    fixture = build_inputs(tmp_path, "hoptimus_filter_manifest")
    run_fixture(fixture, monkeypatch)
    assert (fixture["out"] / "COMPLETE.json").exists()


def test_stage1a_review_requires_links_for_unconsumed_outputs_without_reading_them(tmp_path, monkeypatch):
    fixture = build_inputs(tmp_path, "hoptimus_filter_manifest")
    source = "stage1a_review"
    review = bound.strict_json(Path(fixture["spec"]["sources"][source]["path"]).read_bytes())
    review["artifacts"]["attempt/quarantine_tiles.csv"] = "1" * 64
    replace_source(fixture, source, bound.json_bytes(review))
    with pytest.raises(bound.BoundError, match="COMPLETE output"):
        run_fixture(fixture, monkeypatch)


def test_explicit_workspace_root_agreement_is_required(tmp_path, monkeypatch):
    fixture = build_inputs(tmp_path)
    monkeypatch.delenv("PROSPER_OUT", raising=False)
    with pytest.raises(bound.BoundError, match="PROSPER_OUT"):
        bound.run_bound(fixture["spec_path"], fixture["sha"], fixture["out"], fixture["root"],
                        task="hoptimus_main", code_paths=fixture["code_paths"], execute=dummy_execute)
    assert not fixture["out"].exists()
    monkeypatch.setenv("PROSPER_OUT", str(tmp_path))
    with pytest.raises(bound.BoundError, match="PROSPER_OUT"):
        bound.run_bound(fixture["spec_path"], fixture["sha"], fixture["out"], fixture["root"],
                        task="hoptimus_main", code_paths=fixture["code_paths"], execute=dummy_execute)


def test_output_alias_retarget_during_final_sweep_never_completes(tmp_path, monkeypatch):
    fixture = build_inputs(tmp_path)
    alias = tmp_path / "output_alias"
    alias.symlink_to(fixture["root"], target_is_directory=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    fixture["out"] = alias / "run"
    original = bound._scan_file
    changed = False
    def scan(path, kind, name, *args, **kwargs):
        nonlocal changed
        if not changed and (fixture["root"] / "run").exists() and kwargs.get("retain") is False:
            alias.unlink()
            alias.symlink_to(outside, target_is_directory=True)
            changed = True
        return original(path, kind, name, *args, **kwargs)
    monkeypatch.setattr(bound, "_scan_file", scan)
    with pytest.raises(Exception):
        run_fixture(fixture, monkeypatch)
    assert not (fixture["root"] / "run" / "COMPLETE.json").exists()
    assert not (outside / "run").exists()


def test_example_is_exact_schema_but_not_a_production_specification():
    path = Path(cli.__file__).parents[2] / "examples" / "r2_1_hoptimus_main.json"
    declaration = bound.strict_json(path.read_bytes())
    bound._schema(declaration, "hoptimus_main")
    assert declaration["expected"] == {
        "raw_slides": 2921, "target_slides": 2778, "patients": 1876,
        "groups": 15, "cross_label_patients": 45}
    assert all(source["size"] == 0 for source in declaration["sources"].values())


def refresh_upstream(fixture, prefix, summary, review=None):
    summary_path = Path(fixture["spec"]["sources"][f"{prefix}_summary"]["path"])
    complete_path = Path(fixture["spec"]["sources"][f"{prefix}_complete"]["path"])
    review_path = Path(fixture["spec"]["sources"][f"{prefix}_review"]["path"])
    summary_path.write_bytes(bound.json_bytes(summary))
    complete = bound.strict_json(complete_path.read_bytes())
    complete["outputs"]["summary.json"] = {
        "sha256": sha(summary_path.read_bytes()), "size": summary_path.stat().st_size}
    complete_path.write_bytes(bound.json_bytes(complete))
    if review is None:
        review = bound.strict_json(review_path.read_bytes())
    for path in (summary_path, complete_path):
        review["artifacts"][str(path.relative_to(review_path.parent))] = sha(path.read_bytes())
    if prefix == "stage1a":
        stage0_review = Path(fixture["spec"]["sources"]["stage0_review"]["path"])
        review["source_sha256"]["stage0_independent_review"] = sha(stage0_review.read_bytes())
    review_path.write_bytes(bound.json_bytes(review))
    for name, path in ((f"{prefix}_summary", summary_path),
                       (f"{prefix}_complete", complete_path), (f"{prefix}_review", review_path)):
        fixture["spec"]["sources"][name] = entry(path)
    return save_spec(fixture)


def rebind_historical_spec_symlink(fixture, prefix):
    summary_path = Path(fixture["spec"]["sources"][f"{prefix}_summary"]["path"])
    review_path = Path(fixture["spec"]["sources"][f"{prefix}_review"]["path"])
    summary = bound.strict_json(summary_path.read_bytes())
    review = bound.strict_json(review_path.read_bytes())
    old_spec = next(b for b in summary["input_bindings"] if b["kind"] == "spec")
    review["artifacts"].pop(str(Path(old_spec["resolved_path"]).relative_to(review_path.parent)))
    lexical = review_path.parent / "lexical"
    physical = review_path.parent / "physical"
    lexical.mkdir()
    physical.mkdir()
    meta = Path(fixture["spec"]["sources"]["metadata"]["path"])
    meta_alias = lexical / "metadata-bound.csv"
    meta_alias.symlink_to(meta)
    summary["declarations"]["metadata"]["path"] = meta_alias.name
    if prefix == "stage1a":
        stage0_review = Path(fixture["spec"]["sources"]["stage0_review"]["path"])
        review_alias = lexical / "stage0-bound-review.json"
        review_alias.symlink_to(stage0_review)
        summary["declarations"]["stage0"]["review"] = review_alias.name
    physical_spec = physical / "spec.json"
    physical_spec.write_bytes(bound.json_bytes(summary["declarations"]))
    lexical_spec = lexical / "spec.json"
    lexical_spec.symlink_to(physical_spec)
    for index, binding in enumerate(summary["input_bindings"]):
        if binding["kind"] == "spec":
            summary["input_bindings"][index] = historical_binding(lexical_spec, "spec", "spec")
        elif binding["kind"] == "metadata":
            summary["input_bindings"][index] = historical_binding(meta_alias, "metadata", "metadata")
        elif binding["kind"] == "stage0" and binding["name"] == "stage0_review":
            summary["input_bindings"][index] = historical_binding(review_alias, "stage0", "stage0_review")
    review["artifacts"][str(physical_spec.relative_to(review_path.parent))] = sha(physical_spec.read_bytes())
    refresh_upstream(fixture, prefix, summary, review)
    assert not (physical / meta_alias.name).exists()
    if prefix == "stage1a":
        assert not (physical / review_alias.name).exists()
    return lexical_spec, physical_spec


@pytest.mark.parametrize("stages", [("stage0",), ("stage1a",), ("stage0", "stage1a")])
def test_upstream_spec_file_symlinks_use_recorded_lexical_origins(tmp_path, monkeypatch, stages):
    fixture = build_inputs(tmp_path, "hoptimus_filter_manifest")
    forbidden = set()
    for prefix in stages:
        forbidden.update(rebind_historical_spec_symlink(fixture, prefix))
        if prefix == "stage0" and "stage1a" not in stages:
            path = Path(fixture["spec"]["sources"]["stage1a_summary"]["path"])
            summary = bound.strict_json(path.read_bytes())
            stage0_review = Path(fixture["spec"]["sources"]["stage0_review"]["path"])
            for index, binding in enumerate(summary["input_bindings"]):
                if binding["kind"] == "stage0" and binding["name"] == "stage0_review":
                    summary["input_bindings"][index] = historical_binding(
                        stage0_review, "stage0", "stage0_review")
            refresh_upstream(fixture, "stage1a", summary)
    original_open = Path.open
    def no_historical_payload_read(path, *args, **kwargs):
        assert path not in forbidden, "historical spec payload must not be reread"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_historical_payload_read)
    marker = run_fixture(fixture, monkeypatch)
    assert marker["task"] == "hoptimus_filter_manifest"
    assert (fixture["out"] / "COMPLETE.json").is_file()


@pytest.mark.parametrize("prefix", ["stage0", "stage1a"])
@pytest.mark.parametrize("invalid", ["missing", "blank", "relative", "nul", "different"])
def test_historical_spec_lexical_path_is_required_not_guessed(tmp_path, monkeypatch, prefix, invalid):
    fixture = build_inputs(tmp_path, "hoptimus_filter_manifest")
    path = Path(fixture["spec"]["sources"][f"{prefix}_summary"]["path"])
    summary = bound.strict_json(path.read_bytes())
    binding = next(b for b in summary["input_bindings"] if b["kind"] == "spec")
    if invalid == "missing":
        binding.pop("path")
    else:
        binding["path"] = {
            "blank": "", "relative": "spec.json", "nul": "/invalid\x00spec.json",
            "different": str(tmp_path / "different-spec.json"),
        }[invalid]
    refresh_upstream(fixture, prefix, summary)
    if prefix == "stage0":
        path = Path(fixture["spec"]["sources"]["stage1a_summary"]["path"])
        summary1 = bound.strict_json(path.read_bytes())
        for index, item in enumerate(summary1["input_bindings"]):
            if item["kind"] == "stage0":
                summary1["input_bindings"][index] = historical_binding(
                    Path(fixture["spec"]["sources"]["stage0_review"]["path"]), "stage0", "stage0_review")
        refresh_upstream(fixture, "stage1a", summary1)
    with pytest.raises(bound.BoundError, match="lexical"):
        run_fixture(fixture, monkeypatch)
    assert not fixture["out"].exists()


@pytest.mark.parametrize("abort", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_post_marker_catchable_abort_cleans_owned_complete_and_reraises(tmp_path, monkeypatch, abort):
    fixture = build_inputs(tmp_path)
    original = bound._check_outputs
    def check(fd, out, target, root, inputs, hashes, *, marker=False):
        if marker:
            assert (fixture["out"] / "COMPLETE.json").is_file()
            raise abort("post-marker synthetic abort")
        return original(fd, out, target, root, inputs, hashes, marker=marker)
    monkeypatch.setattr(bound, "_check_outputs", check)
    with pytest.raises(abort, match="post-marker synthetic abort"):
        run_fixture(fixture, monkeypatch)
    assert not (fixture["out"] / "COMPLETE.json").exists()
    failure = bound.strict_json((fixture["out"] / "FAILED.json").read_bytes())
    assert failure["type"] == abort.__name__
    assert failure["error"] == "post-marker synthetic abort"


@pytest.mark.parametrize("abort", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_unverified_ownership_abort_never_cleans_foreign_completion(tmp_path, monkeypatch, abort):
    fixture = build_inputs(tmp_path)
    def unverified(fd, target):
        (fixture["out"] / "COMPLETE.json").write_bytes(b"unowned marker")
        raise abort("ownership not established")
    monkeypatch.setattr(bound.legacy, "_claimed_path", unverified)
    with pytest.raises(abort, match="ownership not established"):
        run_fixture(fixture, monkeypatch)
    assert (fixture["out"] / "COMPLETE.json").read_bytes() == b"unowned marker"
    assert not (fixture["out"] / "FAILED.json").exists()
