"""Tiny, wholly synthetic Stage1b preparation fixtures."""

import csv
import hashlib
import io
import json
import pickle
import weakref
from pathlib import Path

import numpy as np
import pytest

from prosper.evaluation import encoder_replay_panel as panel


def digest(data):
    return hashlib.sha256(data).hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def js(path, obj):
    return put(path, panel._json_bytes(obj))


def csv_data(header, rows):
    return panel._csv_bytes(header, rows)


def features(path, coords, width, dtype="float32", *, offset=0):
    array = (np.arange(len(coords) * width).reshape(len(coords), width) % 31
             + offset).astype(dtype)
    tiles = [{"wsi_loc": [x, y]} for x, y in coords]
    put(path, pickle.dumps((array, tiles), protocol=4))
    return array


def _review(directory, name, files, source, metadata, review0=None):
    attempt = directory / "attempt"
    outputs = {}
    artifacts = {}
    for filename, content in files.items():
        path = put(attempt / filename, content)
        outputs[filename] = {"sha256": digest(content), "size": len(content)}
        artifacts["attempt/" + filename] = digest(content)
    complete = {
        "schema_version": 1, "run_id": name,
        "scope": ("feature-bank binding only" if name == "stage0"
                  else "historical-annotation target keyed binding only"),
        "outputs": outputs,
    }
    complete_path = js(attempt / "COMPLETE.json", complete)
    artifacts["attempt/COMPLETE.json"] = digest(complete_path.read_bytes())
    source_sha = {
        "metadata": digest(metadata),
        "auditor_module": digest(source.read_bytes()),
        "auditor_tests": digest((source.parents[2] / "tests" /
                                ("test_feature_bank_audit.py" if name == "stage0"
                                 else "test_annotation_bank_audit.py")).read_bytes()),
    }
    if review0:
        source_sha.update(
            stage0_helper_module=digest(Path(panel.bank_audit.__file__).read_bytes()),
            stage0_independent_review=digest(review0.read_bytes()))
    review = js(directory / "review.json", {
        "schema_version": 1, "disposition": "accepted",
        "scope": ("Stage0 feature-bank binding only" if name == "stage0"
                  else "Stage1a historical-annotation target keyed binding only"),
        "artifacts": artifacts, "source_sha256": source_sha,
    })
    return {"review": str(review), "review_sha256": digest(review.read_bytes()),
            "complete": str(complete_path)}


def fixture(tmp):
    root = tmp / "input"
    slide_ids = ["s1", "s2"]
    patient_ids = ["p1", "p2"]
    metadata = csv_data(("id", "patient", "group"),
                        [["s1", "p1", "NORMAL"], ["s2", "p2", "MISC"]])
    metadata_path = put(root / "metadata.csv", metadata)
    main = {s: [(i + (0 if s == "s1" else 100), 0) for i in range(12)]
            for s in slide_ids}
    annotated = main["s1"][:10]
    main_roots = {b: root / "main" / b for b in panel.BANKS}
    annotation_roots = {b: root / "annotation" / b for b in panel.BANKS}
    stage0_inventory = []
    stage1_inventory = []
    for universe, slides, roots, inventory in (
        ("main", slide_ids, main_roots, stage0_inventory),
        ("annotation", ["s1"], annotation_roots, stage1_inventory),
    ):
        for bank in panel.BANKS:
            for sid in slides:
                width = (panel.MAIN_DIM if universe == "main"
                         else panel.ANNOTATION_DIM)[bank]
                dtype = ("float16" if bank == "virchow2" and sid == "s1"
                         and universe == "main" else "float32")
                path = roots[bank] / f"{sid}.pkl"
                features(path, main[sid] if universe == "main" else annotated,
                         width, dtype)
                if universe == "main":
                    stage0_inventory.append((
                        bank, sid, len(main[sid]), len(main[sid]), dtype,
                        str(path), str(path.resolve()), path.stat().st_size,
                        digest(path.read_bytes())))
                else:
                    stage1_inventory.append((
                        bank, sid, width, 0, 1280 if bank == "virchow2" else width,
                        len(annotated), dtype, len(annotated), 0, 0, 0,
                        "true", 0, str(path), str(path.resolve()),
                        path.stat().st_size, digest(path.read_bytes())))
    aux_path = root / "auxiliary" / "s2.pkl"
    features(aux_path, main["s2"][:10], 1536)
    aux_csv = put(root / "auxiliary" / "s2.csv", csv_data(
        ("slide_id", "wsi_loc_x", "wsi_loc_y"),
        [["s2", x, y] for x, y in main["s2"][:10]]))
    eligible = csv_data(panel.annotation.ELIGIBLE_HEADER, [
        [panel.annotation._tile_id(panel._physical("s1", xy)), "s1", "p1",
         "NORMAL", 0, *xy, 512, i % 5 + 1, i, i, i, i, i]
        for i, xy in enumerate(annotated)])
    metadata_config = {"path": str(metadata_path), "id_col": "id",
                       "patient_col": "patient", "label_col": "group"}
    base_summary = {"run_id": "stage0", "declarations": {"metadata": metadata_config},
                    "input_bindings": [{"kind": "metadata", "name": "metadata",
                                        "path": str(metadata_path),
                                        "resolved_path": str(metadata_path.resolve()),
                                        "sha256": digest(metadata)}]}
    stage0_dir = root / "stage0"
    stage0 = _review(stage0_dir, "stage0", {
        "summary.json": panel._json_bytes(base_summary),
        "inventory.csv": csv_data((
            "bank", "slide_id", "canonical_tiles", "feature_rows", "dtype",
            "feature_path", "resolved_path", "size", "sha256"), stage0_inventory),
    }, Path(panel.bank_audit.__file__), metadata)
    stage1_summary = {**base_summary, "run_id": "stage1a",
                      "eligible": {"rows": len(annotated)}}
    stage1a = _review(root / "stage1a", "stage1a", {
        "summary.json": panel._json_bytes(stage1_summary),
        "eligible_tiles.csv": eligible,
        "feature_inventory.csv": csv_data(panel.annotation.FEATURE_HEADER,
                                          stage1_inventory),
    }, Path(panel.annotation.__file__), metadata,
        Path(stage0["review"]))
    candidates = []
    for name in panel.ROOTS:
        candidate_root = root / "candidates" / name
        present = name == "C"
        files = []
        if present:
            candidate_root.mkdir(parents=True)
        for sid in slide_ids:
            path = candidate_root / f"{sid}.pkl"
            if present:
                features(path, main[sid], 1280,
                         "float16" if sid == "s1" else "float32")
            files.append({"slide_id": sid, "present": present,
                          "size_bytes": path.stat().st_size if present else None,
                          "expected_sha256": digest(path.read_bytes()) if present else None})
        candidates.append({"name": name, "path": str(candidate_root),
                           "expected_root_presence": present, "files": files})
    wsis = []
    for sid in slide_ids:
        path = put(root / "images" / f"{sid}.svs", (sid * 4).encode())
        wsis.append({"slide_id": sid, "path": str(path),
                     "resolved_path": str(path.resolve()),
                     "size_bytes": path.stat().st_size})
    models = []
    for name in panel.BANKS:
        weight = put(root / "weights" / f"{name}.bin", (name * 3).encode())
        config = put(root / "weights" / f"{name}.json", b'{"model":"fixture"}')
        models.append({
            "name": name, "model_id": "fixture/" + name, "revision": "fixture-v1",
            "config_path": str(config), "config_sha256": digest(config.read_bytes()),
            "weight_path": str(weight), "resolved_weight_path": str(weight.resolve()),
            "weight_size_bytes": weight.stat().st_size,
            "expected_weight_sha256": digest(weight.read_bytes()),
            "digest_basis": "declaration",
        })
    tags = {tag: ("s1" if i % 2 else "s2") for i, tag in enumerate(panel.CONTROL_TAGS)}
    controls = []
    for i, (tag, sid) in enumerate(tags.items()):
        line = f"[extract] {sid}: ({10 + i}, 1536) in 1.5s (7.3 tiles/s)\n"
        path = put(root / "logs" / f"{tag}.txt", line.encode())
        controls.append({"tag": tag, "slide_id": sid, "path": str(path),
                         "sha256": digest(path.read_bytes()), "write_line": 1})
    panels = [{"slide_id": sid, "patient_id": patient_ids[i],
               "group": "NORMAL" if i == 0 else "MISC",
               "virchow_stratum": "older_float16" if i == 0 else "round3_float32",
               "annotation_classes": [1, 2, 3, 4, 5] if i == 0 else [],
               "control_tags": sorted(t for t, s in tags.items() if s == sid)}
              for i, sid in enumerate(slide_ids)]
    expected = {
        "panel_sha256": digest(panel.compact(slide_ids)), "slides": 2,
        "patients": 2, "universes": 4, "tile_rows": 32,
        "feature_rows": 80, "feature_files": 10, "array_files": 8,
        "candidate_files_present": 2, "candidate_files_absent": 12,
        "lineage_pairs": 2,
        "array_payload_bytes": (
            24 * 1536 * 4 + 8 * 1280 * 2 + 8 * 1280 * 4 +
            8 * 2560 * 4 + (16 + 8 + 8) * 1536 * 4),
    }
    spec = {
        "schema_version": 1, "run_id": "synthetic_replay",
        "stage0": stage0, "stage1a": stage1a, "metadata": metadata_config,
        "panel": panels,
        "banks": [{"name": bank, "universe": universe, "path": str(roots[bank]),
                   "feature_dim": (panel.MAIN_DIM if universe == "main"
                                   else panel.ANNOTATION_DIM)[bank],
                   "dtypes": ["float16", "float32"]}
                  for universe, roots in (("main", main_roots),
                                           ("annotation", annotation_roots))
                  for bank in panel.BANKS],
        "auxiliary": {"slide_id": "s2", "path": str(aux_path),
                      "csv": str(aux_csv), "csv_sha256": digest(aux_csv.read_bytes()),
                      "feature_dim": 1536, "dtype": "float32"},
        "wsi": wsis, "models": models, "virchow_candidates": candidates,
        "lineage_pairs": [], "log_controls": controls,
        "code_files": [{"name": "control", "path": str(put(root / "code.txt", b"code\n")),
                        "sha256": digest(b"code\n")}],
        "expected": expected,
    }
    spec["lineage_pairs"] = panel._expected_pairs(
        spec, {item["name"]: item for item in candidates})
    spec_path = js(tmp / "spec.json", spec)
    output_root = tmp / "output"
    output_root.mkdir()
    return spec, spec_path, output_root


def run(tmp, monkeypatch, modify=None):
    spec, spec_path, output_root = fixture(tmp)
    if modify:
        modify(spec, spec_path)
        js(spec_path, spec)
    monkeypatch.setenv("PROSPER_OUT", str(output_root))
    output = output_root / "attempt"
    return spec, spec_path, output, panel.audit(
        str(spec_path), digest(spec_path.read_bytes()), str(output))


def rebind_stage1a(spec, spec_path, eligible_rows=None, changed_features=None):
    complete_path = Path(spec["stage1a"]["complete"])
    complete = json.loads(complete_path.read_bytes())
    attempt = complete_path.parent
    review_path = Path(spec["stage1a"]["review"])
    review = json.loads(review_path.read_bytes())
    if eligible_rows is not None:
        content = csv_data(panel.annotation.ELIGIBLE_HEADER, [
            [row[column] for column in panel.annotation.ELIGIBLE_HEADER]
            for row in eligible_rows])
        put(attempt / "eligible_tiles.csv", content)
        complete["outputs"]["eligible_tiles.csv"] = {
            "sha256": digest(content), "size": len(content)}
        review["artifacts"]["attempt/eligible_tiles.csv"] = digest(content)
    if changed_features:
        inventory_path = attempt / "feature_inventory.csv"
        with inventory_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for bank, path in changed_features.items():
            entry = next(row for row in rows if row["bank"] == bank
                         and row["slide_id"] == "s1")
            entry["size"] = str(path.stat().st_size)
            entry["sha256"] = digest(path.read_bytes())
        content = csv_data(panel.annotation.FEATURE_HEADER, [
            [row[column] for column in panel.annotation.FEATURE_HEADER] for row in rows])
        put(inventory_path, content)
        complete["outputs"]["feature_inventory.csv"] = {
            "sha256": digest(content), "size": len(content)}
        review["artifacts"]["attempt/feature_inventory.csv"] = digest(content)
    js(complete_path, complete)
    review["artifacts"]["attempt/COMPLETE.json"] = digest(complete_path.read_bytes())
    js(review_path, review)
    spec["stage1a"]["review_sha256"] = digest(review_path.read_bytes())
    js(spec_path, spec)


def eligible_records(spec):
    path = Path(spec["stage1a"]["complete"]).parent / "eligible_tiles.csv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def permute_annotation_banks(spec, records):
    changed = {}
    orders = {"uni2": list(range(9, -1, -1)),
              "virchow2": list(range(1, 10)) + [0],
              "h_optimus_1": list(range(2, 10)) + [0, 1]}
    for bank, order in orders.items():
        root = next(row["path"] for row in spec["banks"]
                    if row["universe"] == "annotation" and row["name"] == bank)
        path = Path(root) / "s1.pkl"
        array, tiles = pickle.loads(path.read_bytes())
        put(path, pickle.dumps((array[order], [tiles[i] for i in order]), protocol=4))
        indices = {tuple(tiles[old_index]["wsi_loc"]): new_index
                   for new_index, old_index in enumerate(order)}
        for row in records:
            xy = int(row["x"]), int(row["y"])
            row[f"{bank}_feature_row"] = str(indices[xy])
        changed[bank] = path
    return changed


def test_valid_receipt_and_full_row_bindings(tmp_path, monkeypatch):
    spec, _, out, receipt = run(tmp_path, monkeypatch)
    assert receipt["disposition"] == "prepared"
    assert len(receipt["outputs"]) == 13
    assert set(p.name for p in out.iterdir()) == {*receipt["outputs"], "COMPLETE.json"}
    assert {name: digest((out / name).read_bytes()) for name in receipt["outputs"]} == {
        name: row["sha256"] for name, row in receipt["outputs"].items()}
    assert json.loads((out / "summary.json").read_text())["counts"][
        "array_payload_bytes"] == spec["expected"]["array_payload_bytes"]
    assert json.loads((out / "lineage_diagnostics.json").read_text())["counts"][
        "comparable"] == 2
    with (out / "source_inventory.csv").open(newline="") as handle:
        source = {row["source_id"]: row for row in csv.DictReader(handle)}
    assert source["feature/auxiliary/h_optimus_1/s2"]["digest_basis"] == "observed_only"
    assert source["feature/auxiliary/h_optimus_1/s2"]["expected_sha256"] == ""
    assert source["feature/auxiliary/h_optimus_1/s2"]["observed_sha256"] == digest(
        Path(spec["auxiliary"]["path"]).read_bytes())
    assert source["control/auxiliary_csv"]["digest_basis"] == "declaration"
    assert json.loads((out / "summary.json").read_text())["coverage"]["control_tags"] == {
        control["tag"]: control["slide_id"] for control in spec["log_controls"]}


def test_coherently_permuted_annotation_indices_are_keyed(tmp_path, monkeypatch):
    spec, path, root = fixture(tmp_path)
    records = eligible_records(spec)
    changed = permute_annotation_banks(spec, records)
    rebind_stage1a(spec, path, records, changed)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    out = root / "attempt"
    panel.audit(str(path), digest(path.read_bytes()), str(out))
    by_tile = {row["tile_id"]: row for row in records}
    with (out / "feature_rows.csv").open(newline="") as handle:
        refs = [row for row in csv.DictReader(handle) if row["universe"] == "annotation"]
    assert len(refs) == 24
    assert all(row["source_row"] ==
               by_tile[row["tile_id"]][f"{row['bank']}_feature_row"]
               for row in refs)
    assert (out / "COMPLETE.json").exists()


@pytest.mark.parametrize("change", ("out_of_bounds", "in_range", "swapped_encoders"))
def test_rebound_annotation_indices_cannot_be_silently_remapped(
        tmp_path, monkeypatch, change):
    spec, path, root = fixture(tmp_path)
    records = eligible_records(spec)
    changed_features = None
    if change == "out_of_bounds":
        for row in records:
            row["uni2_feature_row"] = "999999"
    elif change == "in_range":
        for row in records:
            row["h_optimus_1_feature_row"] = str(
                (int(row["h_optimus_1_feature_row"]) + 1) % len(records))
    else:
        changed_features = permute_annotation_banks(spec, records)
        for row in records:
            row["uni2_feature_row"], row["virchow2_feature_row"] = (
                row["virchow2_feature_row"], row["uni2_feature_row"])
    rebind_stage1a(spec, path, records, changed_features)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    with pytest.raises(panel.AuditError, match="Stage1a feature row"):
        panel.audit(str(path), digest(path.read_bytes()), str(root / "attempt"))
    assert (root / "attempt/FAILED.json").exists()
    assert not (root / "attempt/COMPLETE.json").exists()


@pytest.mark.parametrize(("sid", "line"), [
    ("10003", "[extract] 10003: (3868, 1536) in 25.9s (149.1 tiles/s)"),
    ("10003", "[extract] 10003: (25591, 1536) in 121.1s (211.3 tiles/s)"),
    ("15825", "[extract] 15825: (1426, 1536) in 18.6s (76.6 tiles/s)"),
    ("6798", "[extract] 6798: (316, 1536) in 9.4s (33.7 tiles/s)"),
    ("7483", "[extract] 7483: (3408, 1536) in 44.2s (77.1 tiles/s)"),
    ("7483", "[extract] 7483: (22578, 1536) in 114.3s (197.6 tiles/s)"),
])
def test_original_producer_line_shape(sid, line):
    panel._parse_write_line(line, sid, "control/synthetic")


@pytest.mark.parametrize("line", [
    "saved s1.pkl",
    "[extract] s11: (10, 1536) in 1.5s (7.3 tiles/s)",
    "[extract] s1: (10, 1280) in 1.5s (7.3 tiles/s)",
    "[extract] s1: (0, 1536) in 1.5s (7.3 tiles/s)",
    "[extract] s1: (-1, 1536) in 1.5s (7.3 tiles/s)",
    "[extract] s1: (1.5, 1536) in 1.5s (7.3 tiles/s)",
    "[extract] s1: (10, 1536) in 0s (7.3 tiles/s)",
    "[extract] s1: (10, 1536) in -1s (7.3 tiles/s)",
    "[extract] s1: (10, 1536) in nans (7.3 tiles/s)",
    "[extract] s1: (10, 1536) in infs (7.3 tiles/s)",
    "[extract] s1: (10, 1536) in 1s (0 tiles/s)",
    "[extract] s1: (10, 1536) in 1s (nan tiles/s)",
    "[extract] s1: (10, 1536) in 1s (inf tiles/s)",
    "[extract] s1: (10, 1536) in " + "9" * 400 + "s (7.3 tiles/s)",
    "[retry] [extract] s1: (10, 1536) in 1.5s (7.3 tiles/s)",
    "[extract] s1: (10, 1536) in 1.5s (7.3 tiles/s) skipped",
    "[extract] s1: (10, 1536) in 1.5s (7.3 tiles/s) failed",
    "[extract] s1: (10, 1536) in 1.5s (7.3 tiles/s) retried",
    " [extract] s1: (10, 1536) in 1.5s (7.3 tiles/s)",
])
def test_structured_log_rejects_non_success(line):
    with pytest.raises(panel.AuditError, match="structured"):
        panel._parse_write_line(line, "s1", "control/synthetic")


def test_invalid_log_fails_early_before_heavy_reads(tmp_path, monkeypatch):
    spec, path, root = fixture(tmp_path)
    control = spec["log_controls"][0]
    put(Path(control["path"]), b"saved s1.pkl\n")
    control["sha256"] = digest(Path(control["path"]).read_bytes())
    js(path, spec)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    monkeypatch.setattr(panel, "_prepare",
                        lambda *args: pytest.fail("feature inputs read before logs"))
    monkeypatch.setattr(panel, "_remaining_sources",
                        lambda *args: pytest.fail("large inputs hashed before logs"))
    with pytest.raises(panel.AuditError, match="structured"):
        panel.audit(str(path), digest(path.read_bytes()), str(root / "attempt"))
    assert (root / "attempt/FAILED.json").exists()
    assert not (root / "attempt/COMPLETE.json").exists()


@pytest.mark.parametrize("change", ("swap", "duplicate", "missing"))
def test_control_tag_slide_consistency(tmp_path, monkeypatch, change):
    spec, path, root = fixture(tmp_path)
    left, right = spec["panel"]
    if change == "swap":
        left["control_tags"], right["control_tags"] = (
            right["control_tags"], left["control_tags"])
    elif change == "duplicate":
        left["control_tags"] = sorted([*left["control_tags"],
                                       right["control_tags"][0]])
    else:
        left["control_tags"].pop()
    js(path, spec)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    with pytest.raises(panel.AuditError, match="tag"):
        panel.audit(str(path), digest(path.read_bytes()), str(root / "attempt"))
    assert not (root / "attempt/COMPLETE.json").exists()
    if change != "missing":
        assert (root / "attempt/FAILED.json").exists()


def test_auxiliary_array_released_before_main_extraction(tmp_path, monkeypatch):
    spec, path, root = fixture(tmp_path)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    original_array, original_emit = panel._array, panel._emit_vectors
    held = []

    def record_aux(data, context, **kwargs):
        array, coords = original_array(data, context, **kwargs)
        if context == "auxiliary":
            held.append(weakref.ref(array))
        return array, coords

    def check_lifetime(*args, **kwargs):
        assert held and held[0]() is None
        return original_emit(*args, **kwargs)

    monkeypatch.setattr(panel, "_array", record_aux)
    monkeypatch.setattr(panel, "_emit_vectors", check_lifetime)
    panel.audit(str(path), digest(path.read_bytes()), str(root / "attempt"))
    assert (root / "attempt/COMPLETE.json").exists()


def test_existing_output_does_not_read_features(tmp_path, monkeypatch):
    spec, path, root = fixture(tmp_path)
    existing = root / "attempt"
    existing.mkdir()
    marker = put(existing / "third-party", b"keep")
    monkeypatch.setenv("PROSPER_OUT", str(root))
    monkeypatch.setattr(panel, "_prepare", lambda *a: pytest.fail("read feature bank"))
    with pytest.raises(FileExistsError):
        panel.audit(str(path), digest(path.read_bytes()), str(existing))
    assert marker.read_bytes() == b"keep"
    assert sorted(p.name for p in existing.iterdir()) == ["third-party"]


def test_wrong_spec_sha_and_boolean_schema(tmp_path, monkeypatch):
    spec, path, root = fixture(tmp_path)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    with pytest.raises(panel.AuditError, match="spec SHA256"):
        panel.audit(str(path), "0" * 64, str(root / "attempt"))
    spec["expected"]["slides"] = True
    js(path, spec)
    with pytest.raises(panel.AuditError, match="expected.slides"):
        panel.audit(str(path), digest(path.read_bytes()), str(root / "attempt"))
    assert not (root / "attempt").exists()


def test_duplicate_json_key_fails_before_claim(tmp_path, monkeypatch):
    _, path, root = fixture(tmp_path)
    data = path.read_bytes().replace(b'"schema_version": 1,',
                                     b'"schema_version": 1, "schema_version": 1,', 1)
    path.write_bytes(data)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    with pytest.raises(panel.AuditError):
        panel.audit(str(path), digest(data), str(root / "attempt"))
    assert not (root / "attempt").exists()


def test_review_artifact_crosslink_tampering_fails_owned(tmp_path, monkeypatch):
    spec, path, root = fixture(tmp_path)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    complete_path = Path(spec["stage1a"]["complete"])
    complete = json.loads(complete_path.read_bytes())
    complete["outputs"]["summary.json"]["sha256"] = "0" * 64
    js(complete_path, complete)
    review_path = Path(spec["stage1a"]["review"])
    review = json.loads(review_path.read_bytes())
    review["artifacts"]["attempt/COMPLETE.json"] = digest(complete_path.read_bytes())
    js(review_path, review)
    spec["stage1a"]["review_sha256"] = digest(review_path.read_bytes())
    js(path, spec)
    with pytest.raises(panel.AuditError, match="COMPLETE output link"):
        panel.audit(str(path), digest(path.read_bytes()), str(root / "attempt"))
    assert (root / "attempt/FAILED.json").exists()
    assert not (root / "attempt/COMPLETE.json").exists()


@pytest.mark.parametrize("field", ("wsi", "model", "feature", "candidate",
                                    "metadata", "code", "spec", "absent"))
def test_late_input_recheck(tmp_path, monkeypatch, field):
    spec, path, root = fixture(tmp_path)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    original = panel._recheck_inputs

    def mutate(entries):
        source = {
            "wsi": "wsi/s1", "model": "weight/uni2",
            "feature": "feature/main/uni2/s1", "candidate": "candidate/C/s1",
            "metadata": "metadata", "code": "code/control", "spec": "spec",
            "absent": "candidate/T/s1",
        }[field]
        victim = Path(entries[source]["path"])
        if field == "absent":
            put(victim, b"added")
        else:
            data = victim.read_bytes()
            put(victim, data[:-1] + bytes([data[-1] ^ 1]))
        original(entries)

    monkeypatch.setattr(panel, "_recheck_inputs", mutate)
    with pytest.raises(panel.AuditError, match="late"):
        panel.audit(str(path), digest(path.read_bytes()), str(root / "attempt"))
    assert (root / "attempt/FAILED.json").exists()
    assert not (root / "attempt/COMPLETE.json").exists()


@pytest.mark.parametrize("name", ("panel_tiles.csv", "feature_rows.csv",
                                  "summary.json", "lineage_diagnostics.json",
                                  "uni2_main.npy"))
def test_written_semantic_corruption(tmp_path, monkeypatch, name):
    spec, path, root = fixture(tmp_path)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    original = panel.bank_audit._atomic_bytes

    def corrupt(fd, filename, data):
        if filename == name:
            if filename.endswith(".npy"):
                buffer = bytearray(data)
                buffer[-4] ^= 1
                data = bytes(buffer)
            elif filename.endswith(".json"):
                parsed = json.loads(data)
                parsed["scope"] = "wrong"
                data = panel._json_bytes(parsed)
            else:
                data = data.replace(b"s1", b"xx", 1)
        original(fd, filename, data)

    monkeypatch.setattr(panel.bank_audit, "_atomic_bytes", corrupt)
    with pytest.raises(panel.AuditError, match="written"):
        panel.audit(str(path), digest(path.read_bytes()), str(root / "attempt"))
    assert (root / "attempt/FAILED.json").exists()
    assert not (root / "attempt/COMPLETE.json").exists()


def test_output_mutated_after_semantic_readback(tmp_path, monkeypatch):
    spec, path, root = fixture(tmp_path)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    original = panel._recheck_inputs

    def mutate(entries):
        original(entries)
        target = root / "attempt" / "uni2_main.npy"
        data = bytearray(target.read_bytes())
        data[-4] ^= 1
        target.write_bytes(data)

    monkeypatch.setattr(panel, "_recheck_inputs", mutate)
    with pytest.raises(panel.AuditError, match="changed after semantic"):
        panel.audit(str(path), digest(path.read_bytes()), str(root / "attempt"))
    assert not (root / "attempt/COMPLETE.json").exists()


def test_late_extra_output_prevents_complete(tmp_path, monkeypatch):
    spec, path, root = fixture(tmp_path)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    original = panel._recheck_inputs

    def add_extra(entries):
        original(entries)
        put(root / "attempt" / "unexpected.txt", b"not a declared output")

    monkeypatch.setattr(panel, "_recheck_inputs", add_extra)
    with pytest.raises(panel.AuditError, match="output set changed"):
        panel.audit(str(path), digest(path.read_bytes()), str(root / "attempt"))
    assert (root / "attempt/unexpected.txt").read_bytes() == b"not a declared output"
    assert (root / "attempt/FAILED.json").exists()
    assert not (root / "attempt/COMPLETE.json").exists()


def test_selection_order_and_class_reservations():
    coords = [(i, 0) for i in range(13)]
    label = {xy: i % 5 + 1 for i, xy in enumerate(coords)}
    chosen = panel._choose(coords, "s", annotations=(label, [1, 2, 3, 4, 5]),
                           main_coords={(0, 0), (1, 0)})
    assert {label[xy] for xy, _ in chosen} == {1, 2, 3, 4, 5}
    assert all(reason == "class_reservation" for _, reason in chosen[:5])
    assert panel._choose(coords, "s") != panel._choose(coords[::-1], "s")
    assert panel._choose(coords, "s", reserved=[(0, 0)])[0] == (
        (0, 0), "intersection")


def test_declared_missing_side_reports_present_side_observations(tmp_path):
    present = tmp_path / "present.pkl"
    features(present, [(0, 0)], 2560)
    entry, _ = panel._source("candidate/C/s1", "candidate", present,
                             bank="virchow2", slide_id="s1",
                             cap=panel.FEATURE_CAP)
    absent, _ = panel._source("candidate/P3/s1", "candidate",
                             tmp_path / "missing.pkl", expected_present=False)
    pair = panel._pair_report({
        "pair_id": "P3__C__s1", "left_source_id": absent["source_id"],
        "right_source_id": entry["source_id"], "relation": "cls_projection",
    }, {absent["source_id"]: absent, entry["source_id"]: entry})
    assert pair["status"] == "non_comparable"
    assert pair["reason"] == "declared_missing_left"
    assert (pair["left_rows"], pair["right_rows"], pair["right_dimension"],
            pair["common_keys"], pair["exact_value_equal"]) == (
                None, 1, 2560, None, None)


def test_pair_reports_coordinate_and_cast_difference(tmp_path):
    left = tmp_path / "left.pkl"
    right = tmp_path / "right.pkl"
    coords = [(0, 0), (1, 0)]
    l = features(left, coords, 1280, "float16")
    features(right, coords[::-1], 1280, "float32")
    # Same values by physical coordinate; signed-zero bits differ for one.
    l[:] = 0
    r = np.zeros((2, 1280), dtype=np.float32)
    r[1, 0] = -0.0
    tiles_left = [{"wsi_loc": [x, y]} for x, y in coords]
    tiles_right = [{"wsi_loc": [x, y]} for x, y in coords[::-1]]
    put(left, pickle.dumps((l, tiles_left), protocol=4))
    put(right, pickle.dumps((r, tiles_right), protocol=4))
    left_entry, _ = panel._source("candidate/O/s1", "candidate", left)
    right_entry, _ = panel._source("candidate/C/s1", "candidate", right)
    pair = panel._pair_report({
        "pair_id": "O__C__s1", "left_source_id": left_entry["source_id"],
        "right_source_id": right_entry["source_id"], "relation": "candidate_copy",
    }, {left_entry["source_id"]: left_entry, right_entry["source_id"]: right_entry})
    assert pair["status"] == "comparable"
    assert pair["common_keys"] == 2
    assert pair["order_equal"] is False
    assert pair["exact_value_equal"] is True
    assert pair["bitwise_equal"] is False
    assert pair["destination_cast_equal"] is False
    assert pair["cast_mismatch_rows"] == 1


@pytest.mark.parametrize("change", ("wrong_patient", "changed_feature",
                                     "missing_candidate", "wrong_weight",
                                     "extra_eligible_class"))
def test_bound_source_failures_are_owned(tmp_path, monkeypatch, change):
    spec, path, root = fixture(tmp_path)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    if change == "wrong_patient":
        spec["panel"][0]["patient_id"] = "another-patient"
        js(path, spec)
    elif change == "changed_feature":
        source = Path(spec["banks"][0]["path"]) / "s1.pkl"
        data = source.read_bytes()
        source.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    elif change == "missing_candidate":
        Path(spec["virchow_candidates"][5]["path"], "s1.pkl").unlink()
    elif change == "wrong_weight":
        spec["models"][0]["resolved_weight_path"] = "/wrong/weight.bin"
        js(path, spec)
    else:
        source = Path(spec["stage1a"]["complete"]).parent / "eligible_tiles.csv"
        source.write_bytes(source.read_bytes() + b"s1,unexpected\n")
    with pytest.raises((panel.AuditError, FileNotFoundError)):
        panel.audit(str(path), digest(path.read_bytes()), str(root / "attempt"))
    assert (root / "attempt/FAILED.json").exists()
    assert not (root / "attempt/COMPLETE.json").exists()


def test_output_lexical_alias_retarget_does_not_write_elsewhere(tmp_path, monkeypatch):
    spec, path, root = fixture(tmp_path)
    physical = root / "physical"
    physical.mkdir()
    other = root / "other"
    other.mkdir()
    alias = root / "current"
    alias.symlink_to(physical, target_is_directory=True)
    unrelated = put(other / "attempt" / "third-party", b"unaltered")
    monkeypatch.setenv("PROSPER_OUT", str(root))
    original = panel._finish

    def retarget(*args):
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
        return original(*args)

    monkeypatch.setattr(panel, "_finish", retarget)
    with pytest.raises(panel.AuditError, match="retargeted"):
        panel.audit(str(path), digest(path.read_bytes()), str(alias / "attempt"))
    assert unrelated.read_bytes() == b"unaltered"
    assert [p.name for p in (other / "attempt").iterdir()] == ["third-party"]
    assert (physical / "attempt/FAILED.json").exists()
    assert not (physical / "attempt/COMPLETE.json").exists()


def test_bounded_reads_reject_growing_files(tmp_path):
    source = put(tmp_path / "growth.bin", b"ab")
    with pytest.raises(panel.AuditError, match="exceeds declared"):
        panel._hash_stream(source, 1)
    with pytest.raises(panel.AuditError, match="exceeds"):
        panel._bounded(source, 1)


def test_cross_role_file_alias_is_not_silent(tmp_path, monkeypatch):
    spec, path, root = fixture(tmp_path)
    monkeypatch.setenv("PROSPER_OUT", str(root))
    file = spec["virchow_candidates"][5]["files"][0]
    candidate = Path(spec["virchow_candidates"][5]["path"]) / "s1.pkl"
    main = Path(next(b["path"] for b in spec["banks"]
                     if b["universe"] == "main" and b["name"] == "virchow2")) / "s1.pkl"
    candidate.unlink()
    candidate.symlink_to(main)
    file["size_bytes"] = main.stat().st_size
    file["expected_sha256"] = digest(main.read_bytes())
    js(path, spec)
    with pytest.raises(panel.AuditError, match="alias"):
        panel.audit(str(path), digest(path.read_bytes()), str(root / "attempt"))
    assert (root / "attempt/FAILED.json").exists()
    assert not (root / "attempt/COMPLETE.json").exists()


@pytest.mark.parametrize("bad", ("trailing", "nonfinite", "duplicate",
                                 "wrong_width", "wrong_dtype"))
def test_malformed_local_feature_payload_fails(tmp_path, bad):
    coords = [(0, 0), (1, 0)]
    array = np.ones((2, 1280), dtype="<f4")
    if bad == "nonfinite":
        array[0, 0] = np.nan
    if bad == "wrong_width":
        array = array[:, :8]
    if bad == "wrong_dtype":
        array = array.astype(np.int32)
    if bad == "duplicate":
        coords[1] = coords[0]
    payload = pickle.dumps(
        (array, [{"wsi_loc": list(xy)} for xy in coords]), protocol=4)
    if bad == "trailing":
        payload += pickle.dumps("unexpected second object", protocol=4)
    with pytest.raises(panel.AuditError):
        panel._array(payload, bad, width=1280)


@pytest.mark.parametrize("change", ["swapped_rows", "changed_component",
                                     "promoted_dtype", "trailing"])
def test_npy_representation_is_strict(change):
    values = np.arange(12, dtype="<f4").reshape(2, 6)
    out = io.BytesIO()
    np.lib.format.write_array(out, values, version=(1, 0))
    blob = out.getvalue()
    if change == "swapped_rows":
        changed = values[::-1].copy()
    elif change == "changed_component":
        changed = values.copy()
        changed[0, 0] = 900
    elif change == "promoted_dtype":
        changed = values.astype("<f2")
    else:
        changed = None
    if changed is not None:
        out = io.BytesIO()
        np.lib.format.write_array(out, changed, version=(1, 0))
        blob = out.getvalue()
    else:
        blob += b"trailing"
    if change in ("promoted_dtype", "trailing"):
        with pytest.raises(panel.AuditError):
            panel._numeric_npy(blob, values.shape, "<f4")
    else:
        assert panel._numeric_npy(blob, values.shape, "<f4").tobytes() != values.tobytes()
