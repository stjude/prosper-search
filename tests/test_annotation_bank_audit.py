"""Isolated synthetic historical-annotation and three-bank binding fixtures."""

import csv
import hashlib
import io
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

from prosper.evaluation import annotation_bank_audit as audit
from prosper.evaluation import feature_bank_audit as stage0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj))


def write_csv(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_npy(path, values, dtype=np.int64):
    np.save(path, np.asarray(values, dtype=dtype))


def fixture(tmp_path):
    root, out = tmp_path / "inputs", tmp_path / "outputs"
    root.mkdir()
    out.mkdir()
    meta = root / "metadata.csv"
    write_csv(meta, ("Slide.Scan.File", "UID.Subject", "Tumor.group", "extra"),
              (("A", "patient", "NORMAL", "kept"), ("B", "patient", "MISC", "kept")))
    (root / "source.py").write_text("version = 1\n")
    review_dir = root / "stage0"
    stage0_run = review_dir / "stage0_attempt_001"
    stage0_run.mkdir(parents=True)
    (review_dir / "stage0_spec.json").write_text('{"stage0":"synthetic"}')
    (review_dir / "stage0_job.lsf").write_text("synthetic\n")
    (review_dir / "stage0_launch.sha256").write_text("synthetic\n")
    (review_dir / "stage0_held_readback.txt").write_text("synthetic\n")
    (stage0_run / "inventory.csv").write_text("bank,slide_id\na,A\n")
    stage0_summary = {
        "schema_version": 1, "run_id": "accepted_fixture",
        "scope": "feature-bank binding only; declared",
        "declarations": {"metadata": {"id_col": "Slide.Scan.File",
                                       "patient_col": "UID.Subject",
                                       "label_col": "Tumor.group"}},
        "input_bindings": [
            {"kind": "spec", "name": "spec",
             "sha256": sha(review_dir / "stage0_spec.json")},
            {"kind": "metadata", "name": "metadata", "sha256": sha(meta)},
            {"kind": "mosaic", "name": "canonical", "sha256": "a" * 64},
            {"kind": "code", "name": "running_auditor",
             "sha256": sha(stage0.__file__)},
            {"kind": "code", "name": "stage0_tests", "sha256": "b" * 64},
        ],
    }
    write_json(stage0_run / "summary.json", stage0_summary)
    complete = {
        "schema_version": 1, "run_id": "accepted_fixture",
        "scope": "feature-bank binding only",
        "outputs": {name: {"sha256": sha(stage0_run / name),
                           "size": (stage0_run / name).stat().st_size}
                    for name in ("summary.json", "inventory.csv")},
    }
    write_json(stage0_run / "COMPLETE.json", complete)
    (review_dir / "stage0_launch.sha256").write_text(
        "".join(f"{digest}  {name}\n" for digest, name in (
            (sha(stage0.__file__), "auditor"),
            ("b" * 64, "tests"),
            (sha(review_dir / "stage0_spec.json"), "spec"),
            (sha(meta), "metadata"),
            ("a" * 64, "mosaic"),
            ("c" * 64, "wrapper"))))
    artifact_names = ("stage0_attempt_001/COMPLETE.json",
                      "stage0_attempt_001/inventory.csv",
                      "stage0_attempt_001/summary.json", "stage0_spec.json",
                      "stage0_job.lsf", "stage0_launch.sha256", "stage0_held_readback.txt")
    review = {
        "schema_version": 1, "disposition": "accepted",
        "scope": "Stage0 feature-bank binding only",
        "artifacts": {name: sha(review_dir / name) for name in artifact_names},
        "source_sha256": {
            "auditor_module": sha(stage0.__file__), "auditor_tests": "b" * 64,
            "metadata": sha(meta), "canonical_mosaic": "a" * 64,
            "worker_wrapper": "c" * 64,
        },
    }
    review_path = review_dir / "stage0_independent_review.json"
    write_json(review_path, review)
    coords = {"A": [(0, 0), (512, 0), (1024, 0), (1536, 0)],
              "B": [(0, 0), (512, 0)], "Q": [(0, 0)]}
    (root / "all_patches").mkdir()
    (root / "labels").mkdir()
    for sid, pairs in coords.items():
        patch = [*pairs, *(([(2048, 0)]) if sid == "A" else ())]
        labels = ([(512, 0), (0, 0), (1024, 0), (1536, 0)]
                  if sid == "A" else pairs)
        codes = {"A": {(0, 0): 1, (512, 0): 2, (1024, 0): 4, (1536, 0): 0},
                 "B": {(0, 0): 5, (512, 0): 0}, "Q": {(0, 0): 3}}[sid]
        write_csv(root / "all_patches" / f"{sid}.csv",
                  ("slide_id", "wsi_loc_x", "wsi_loc_y", "loc_x"),
                  ((sid, x, y, 99) for x, y in patch))
        write_csv(root / "labels" / f"{sid}.csv",
                  ("slide_id", "wsi_loc_x", "wsi_loc_y", "patch_label", "extra"),
                  ((sid, x, y, codes[(x, y)], "retain") for x, y in labels))
    canonical = [(sid, [{"wsi_loc": [x, y]} for x, y in xy])
                 for sid, xy in coords.items()]
    canonical[0][1].append({"wsi_loc": [2560, 0]})
    canonical.append(("OTHER", [{"wsi_loc": [0, 0]}]))
    (root / "all_patches_combined.pkl").write_bytes(pickle.dumps(canonical, protocol=4))
    orders = {"uni2": [("A", 0, 0), ("A", 512, 0), ("B", 0, 0), ("Q", 0, 0)],
              "virchow2": [("Q", 0, 0), ("A", 512, 0), ("B", 0, 0), ("A", 0, 0)]}
    controls = {"uni2": ([0, 2], [1, 3], [1, 5], [2, 3]),
                "virchow2": ([0, 1], [2, 3], [3, 2], [5, 1])}
    for lineage, order in orders.items():
        folder = root / lineage
        folder.mkdir()
        write_csv(folder / "slide_info.csv", ("slide_id", "wsi_loc_x", "wsi_loc_y"),
                  order)
        for name, values in zip(("idx_train_rand", "idx_test_rand", "y_train_rand",
                                 "y_test_rand"), controls[lineage]):
            write_npy(folder / f"{name}.npy", values)
    bank_specs = []
    for bank, dim in (("uni2", 1536), ("virchow2", 2560), ("h_optimus_1", 1536)):
        folder = root / "features" / bank
        folder.mkdir(parents=True)
        for sid in ("A", "B"):
            row_order = ([(512, 0), (0, 0), (1024, 0), (1536, 0), (4096, 0)]
                         if sid == "A" else coords[sid])
            array = np.ones((len(row_order), dim),
                            dtype=np.float16 if bank == "virchow2" else np.float32)
            (folder / f"{sid}.pkl").write_bytes(pickle.dumps(
                (array, [{"wsi_loc": [x, y]} for x, y in row_order]), protocol=4))
        (folder / "unrelated.pkl").write_bytes(b"unrelated sidecar")
        bank_specs.append({"name": bank, "path": f"features/{bank}", "feature_dim": dim,
                           "projection": [0, 1280 if bank == "virchow2" else dim],
                           "dtypes": ["float16", "float32"]})
    spec = {
        "schema_version": 1, "run_id": "synthetic_annotation",
        "stage0": {"review": "stage0/stage0_independent_review.json",
                   "review_sha256": sha(review_path),
                   "complete": "stage0/stage0_attempt_001/COMPLETE.json"},
        "metadata": {"path": "metadata.csv", "id_col": "Slide.Scan.File",
                     "patient_col": "UID.Subject", "label_col": "Tumor.group"},
        "lineages": [
            {"name": name, "slide_info": f"{name}/slide_info.csv",
             "train_indices": f"{name}/idx_train_rand.npy",
             "test_indices": f"{name}/idx_test_rand.npy",
             "train_labels": f"{name}/y_train_rand.npy",
             "test_labels": f"{name}/y_test_rand.npy"} for name in ("uni2", "virchow2")],
        "annotations": {"patch_csv_dir": "all_patches", "label_csv_dir": "labels",
                        "canonical_path": "all_patches_combined.pkl",
                        "level": 0, "patch_size": 512},
        "banks": bank_specs,
        "expected": {"historical_rows": 4, "historical_slides": 3,
                     "train_rows": 2, "test_rows": 2, "mapped_rows": 3,
                     "mapped_slides": 2, "quarantined_rows": 1,
                     "quarantined_slides": 1,
                     "mapped_class_counts": {str(i): [1, 1, 0, 0, 1][i - 1]
                                             for i in range(1, 6)},
                     "quarantined_class_counts": {str(i): int(i == 3)
                                                  for i in range(1, 6)}},
        "code_files": [{"name": "fixture_source", "path": "source.py"}],
    }
    spec_path = root / "spec.json"
    write_json(spec_path, spec)
    return root, out, spec_path, spec


def run(path, out, name="run"):
    return audit.audit(path, sha(path), out / name, str(out))


def fails(path, out, match, name="run"):
    with pytest.raises(Exception, match=match):
        run(path, out, name)
    assert not (out / name / "COMPLETE.json").exists()


def test_valid_mapped_quarantine_order_diagnostics(tmp_path, monkeypatch, capsys):
    root, out, path, spec = fixture(tmp_path)
    monkeypatch.setenv("PROSPER_OUT", str(out))
    assert audit.main(["--spec", str(path), "--spec-sha256", sha(path),
                       "--out", str(out / "run")]) == 0
    assert "WARNING" in capsys.readouterr().err
    eligible = csv_rows(out / "run/eligible_tiles.csv")
    quarantine = csv_rows(out / "run/quarantine_tiles.csv")
    assert len(eligible) == 3 and len(quarantine) == 1
    assert {(r["diagnostic_group"], r["patient_id"]) for r in eligible} == {
        ("NORMAL", "patient"), ("MISC", "patient")}
    assert ("patient_id" not in quarantine[0]
            and "uni2_feature_row" not in quarantine[0])
    assert quarantine[0]["reason"] == "slide_absent_from_bound_metadata"
    a0 = next(row for row in eligible if row["slide_id"] == "A" and row["x"] == "0")
    assert (a0["uni2_legacy_row"], a0["virchow2_legacy_row"],
            a0["uni2_feature_row"], a0["virchow2_feature_row"],
            a0["h_optimus_1_feature_row"]) == ("0", "3", "1", "1", "1")
    manifest = json.loads((out / "run/summary.json").read_text())
    assert manifest["eligible"]["patients"] == 1
    assert manifest["eligible"]["class_patients"] == {
        str(i): int(i in (1, 2, 5)) for i in range(1, 6)}
    assert manifest["quarantine"]["class_rows"]["3"] == 1
    assert manifest["legacy_X_validation"] == "not_performed"
    assert manifest["lineage_order_equal"] is False
    assert any(entry["kind"] == "lineage_global_order"
               for entry in json.loads((out / "run/diagnostics.json").read_text())["entries"])
    assert not manifest["nontarget_source_sets_equal"]
    inventory = csv_rows(out / "run/feature_inventory.csv")
    assert len(inventory) == 6
    for row in inventory:
        if row["slide_id"] == "A":
            assert (row["target_rows"], row["unused_unannotated"],
                    row["unused_labelled_nontarget"],
                    row["unused_outside_declared_sources"]) == ("2", "1", "1", "1")
    receipt = json.loads((out / "run/COMPLETE.json").read_text())
    assert set(receipt["outputs"]) == set(audit.OUTPUT_HEADERS) | {
        "summary.json", "diagnostics.json"}
    for name, binding in receipt["outputs"].items():
        assert sha(out / "run" / name) == binding["sha256"]


@pytest.mark.parametrize("change,match", [
    (lambda s: s.update(schema_version=True), "schema_version"),
    (lambda s: s["annotations"].update(level=False), "annotations.level"),
    (lambda s: s["banks"][0].update(projection=[False, 1536]), "projection"),
    (lambda s: s["banks"][1].update(feature_dim=1280), "feature_dim"),
    (lambda s: s["banks"][1].update(path="features/uni2"), "bank path|input roots"),
    (lambda s: s["code_files"][0].update(name="stage0_helper"), "reserved"),
    (lambda s: s["expected"].update(mapped_slides=False), "mapped_slides"),
    (lambda s: s["expected"].update(quarantined_rows=0), "class sum|historical_rows"),
])
def test_strict_spec(tmp_path, change, match):
    root, out, path, spec = fixture(tmp_path)
    change(spec)
    write_json(path, spec)
    fails(path, out, match)


def test_duplicate_json_csv_headers_and_bad_identity(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    path.write_text('{"schema_version":1,"schema_version":1}')
    fails(path, out, "duplicate JSON key")
    write_json(path, spec)
    csv_path = root / "labels/A.csv"
    original = csv_path.read_text()
    csv_path.write_text(original.replace("patch_label,extra", "patch_label,slide_id"))
    fails(path, out, "duplicate/blank headers")
    csv_path.write_text(original.replace("A,1024,0", "B,1024,0"))
    fails(path, out, "slide_id disagrees", "different-slide")
    csv_path.write_text(original.replace("A,1024,0", " A,1024,0"))
    fails(path, out, "unsafe or whitespace", "padded-slide")
    csv_path.write_text(original)


def test_metadata_preserves_string_ids_and_rejects_padding():
    config = {"id_col": "scan", "patient_col": "patient", "label_col": "group"}
    assert audit._metadata(b"scan,patient,group,extra\n001,p,NORMAL,kept\n",
                           config) == {"001": ("p", "NORMAL")}
    for data in (b"scan,patient,group\n 001,p,NORMAL\n",
                 b"scan,patient,group\n001, p,NORMAL\n",
                 b"scan,patient,group\n001,p, NORMAL\n"):
        with pytest.raises(audit.AuditError, match="unsafe|padded"):
            audit._metadata(data, config)


def test_symlink_sensitive_relative_code_path(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    external = tmp_path / "external"
    (external / "child").mkdir(parents=True)
    (external / "source.py").write_text("version = 2\n")
    (root / "linked").symlink_to(external / "child", target_is_directory=True)
    spec["code_files"][0]["path"] = "linked/../source.py"
    write_json(path, spec)
    run(path, out)
    bindings = json.loads((out / "run/summary.json").read_text())["input_bindings"]
    code = next(item for item in bindings if item["name"] == "fixture_source")
    assert code["path"] == str(root / "linked/../source.py")
    assert code["resolved_path"] == str(external / "source.py")
    assert code["sha256"] == sha(external / "source.py") != sha(root / "source.py")


@pytest.mark.parametrize("mutate,match", [
    (lambda text: text.replace("A,0,0,1", "A,0,0,4"), "missing/conflicting target"),
    (lambda text: text.replace("A,0,0,1", "A,512,0,1"), "duplicate physical key"),
    (lambda text: text.replace("A,0,0,1", "A,-1,0,1"), "decimal"),
    (lambda text: text.replace("A,0,0,1", "A,0.0,0,1"), "decimal"),
])
def test_label_source_errors(tmp_path, mutate, match):
    root, out, path, spec = fixture(tmp_path)
    p = root / "labels/A.csv"
    p.write_text(mutate(p.read_text()))
    fails(path, out, match)


@pytest.mark.parametrize("name,value,match", [
    ("idx_test_rand.npy", np.array([1, 1], np.int64), "duplicate/out-of-range"),
    ("idx_test_rand.npy", np.array([1, 8], np.int64), "duplicate/out-of-range"),
    ("y_test_rand.npy", np.array([2, 6], np.int64), "invalid tissue label"),
    ("idx_test_rand.npy", np.array([1, 3], np.int32), "signed int64"),
    ("idx_test_rand.npy", np.array([[1, 3]], np.int64), "signed int64"),
    ("idx_test_rand.npy", np.array([1, 3], object), "signed int64"),
])
def test_npy_partitions_and_types(tmp_path, name, value, match):
    root, out, path, spec = fixture(tmp_path)
    np.save(root / "uni2" / name, value)
    fails(path, out, match)


@pytest.mark.parametrize("mode,match", [
    ("extra", "truncated or trailing"),
    ("short", "truncated or trailing"),
    ("header", "invalid NPY"),
])
def test_npy_payload_and_header(tmp_path, mode, match):
    root, out, path, spec = fixture(tmp_path)
    target = root / "uni2/idx_train_rand.npy"
    data = target.read_bytes()
    target.write_bytes(data + b"x" if mode == "extra"
                       else data[:-1] if mode == "short"
                       else b"not-npy")
    fails(path, out, match)


def test_npy_big_endian_and_empty_partitions(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    write_npy(root / "uni2/idx_train_rand.npy", [], dtype=">i8")
    write_npy(root / "uni2/y_train_rand.npy", [], dtype=">i8")
    write_npy(root / "uni2/idx_test_rand.npy", [0, 1, 2, 3], dtype=">i8")
    write_npy(root / "uni2/y_test_rand.npy", [1, 2, 5, 3], dtype=">i8")
    write_npy(root / "virchow2/idx_train_rand.npy", [], dtype=">i8")
    write_npy(root / "virchow2/y_train_rand.npy", [], dtype=">i8")
    write_npy(root / "virchow2/idx_test_rand.npy", [0, 1, 2, 3], dtype=">i8")
    write_npy(root / "virchow2/y_test_rand.npy", [3, 2, 5, 1], dtype=">i8")
    spec["expected"]["train_rows"] = 0
    spec["expected"]["test_rows"] = 4
    write_json(path, spec)
    run(path, out)


def test_npy_version_2_is_supported(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.array([0, 2], dtype=">i8"), version=(2, 0))
    (root / "uni2/idx_train_rand.npy").write_bytes(stream.getvalue())
    run(path, out)


def test_read_cap_detects_growth_during_materialization():
    class Grows(io.BytesIO):
        def __init__(self):
            super().__init__(b"12345")
            self.calls = 0

        def read(self, size=-1):
            result = super().read(size)
            self.calls += 1
            if self.calls == 1:
                position = self.tell()
                self.write(b"6")
                self.seek(position)
            return result

    class Growing:
        def open(self, mode):
            assert mode == "rb"
            return Grows()

        def __str__(self):
            return "growing-input"

    with pytest.raises(audit.AuditError, match="5-byte materialized-read cap"):
        audit._read_capped(Growing(), 5)


def test_feature_read_cap_is_enforced_during_audit(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    monkeypatch.setattr(audit, "FEATURE_LIMIT", 100)
    fails(path, out, "100-byte materialized-read cap")
    assert not (out / "run/COMPLETE.json").exists()


@pytest.mark.parametrize("change,match", [
    (lambda a, t: (np.ones((len(t), 1280), np.float32), t), "raw shape"),
    (lambda a, t: (a.astype(np.float64), t), "dtype"),
    (lambda a, t: (np.full(a.shape, np.nan, np.float32), t), "nonfinite"),
    (lambda a, t: (a, t[:-1]), "row/coordinate"),
    (lambda a, t: (a, t[1:]), "row/coordinate"),
    (lambda a, t: (a, [*t[:-1], {"wsi_loc": [-1, 0]}]), "negative"),
])
def test_feature_errors(tmp_path, change, match):
    root, out, path, spec = fixture(tmp_path)
    f = root / "features/virchow2/A.pkl"
    array, tiles = pickle.loads(f.read_bytes())
    f.write_bytes(pickle.dumps(change(array, tiles), protocol=4))
    fails(path, out, match)


def test_missing_feature_and_false_quarantine(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    (root / "features/uni2/B.pkl").unlink()
    fails(path, out, "No such file")
    (root / "metadata.csv").write_text(
        "Slide.Scan.File,UID.Subject,Tumor.group\nA,p,NORMAL\nB,p,MISC\nQ,p,OTHER\n")
    fails(path, out, "Stage1 metadata bytes differ", "false-quarantine")


def test_feature_missing_target_key_and_canonical_invalid(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    f = root / "features/uni2/A.pkl"
    array, tiles = pickle.loads(f.read_bytes())
    tiles[0]["wsi_loc"] = [8192, 0]
    f.write_bytes(pickle.dumps((array, tiles), protocol=4))
    fails(path, out, "missing target feature coordinates")
    mosaic = root / "all_patches_combined.pkl"
    canonical = pickle.loads(mosaic.read_bytes())
    canonical[0][1][0]["wsi_loc"] = [-1, 0]
    mosaic.write_bytes(pickle.dumps(canonical, protocol=4))
    fails(path, out, "negative physical coordinate", "negative")


def test_canonical_requires_bare_ids_and_all_target_keys(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    mosaic = root / "all_patches_combined.pkl"
    original = pickle.loads(mosaic.read_bytes())
    wrong_id = [(sid + ".svs" if sid == "A" else sid, tiles)
                for sid, tiles in original]
    mosaic.write_bytes(pickle.dumps(wrong_id, protocol=4))
    fails(path, out, "expected bare slide ID")
    original[0][1][0]["wsi_loc"] = [9999, 0]
    mosaic.write_bytes(pickle.dumps(original, protocol=4))
    fails(path, out, "missing/conflicting target", "missing-target")


def test_second_lineage_label_conflict_and_nontarget_duplicate(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    write_npy(root / "virchow2/y_test_rand.npy", [5, 2])
    fails(path, out, "lineages: missing/conflicting")
    write_npy(root / "virchow2/y_test_rand.npy", [5, 1])
    patch = root / "all_patches/A.csv"
    patch.write_text(patch.read_text() + "A,2048,0,99\n")
    fails(path, out, "duplicate physical key", "duplicate-nontarget")


def test_declared_feature_alias_is_reported(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    other = root / "features/h_optimus_1/A.pkl"
    other.unlink()
    os.link(root / "features/uni2/A.pkl", other)
    result = run(path, out)
    assert any(entry["kind"] == "declared_file_alias" for entry in result["entries"])
    assert (out / "run/COMPLETE.json").exists()


def test_exact_order_no_differences_has_empty_diagnostics(tmp_path, monkeypatch, capsys):
    root, out, path, spec = fixture(tmp_path)
    write_csv(root / "virchow2/slide_info.csv",
              ("slide_id", "wsi_loc_x", "wsi_loc_y"),
              (("A", 0, 0), ("A", 512, 0), ("B", 0, 0), ("Q", 0, 0)))
    write_npy(root / "virchow2/idx_train_rand.npy", [0, 2])
    write_npy(root / "virchow2/idx_test_rand.npy", [1, 3])
    write_npy(root / "virchow2/y_train_rand.npy", [1, 5])
    write_npy(root / "virchow2/y_test_rand.npy", [2, 3])
    canonical = []
    for sid, pairs in (("A", [(0, 0), (512, 0)]), ("B", [(0, 0)]),
                       ("Q", [(0, 0)])):
        write_csv(root / "all_patches" / f"{sid}.csv",
                  ("slide_id", "wsi_loc_x", "wsi_loc_y"), ((sid, x, y) for x, y in pairs))
        code = {"A": [1, 2], "B": [5], "Q": [3]}[sid]
        write_csv(root / "labels" / f"{sid}.csv",
                  ("slide_id", "wsi_loc_x", "wsi_loc_y", "patch_label"),
                  ((sid, x, y, label) for (x, y), label in zip(pairs, code)))
        canonical.append((sid, [{"wsi_loc": [x, y]} for x, y in pairs]))
    (root / "all_patches_combined.pkl").write_bytes(pickle.dumps(canonical, protocol=4))
    for bank in audit.BANKS:
        for sid in ("A", "B"):
            f = root / "features" / bank / f"{sid}.pkl"
            arr, tiles = pickle.loads(f.read_bytes())
            arr = arr[[1, 0], :] if sid == "A" else arr[:1, :]
            xy = [(0, 0), (512, 0)] if sid == "A" else [(0, 0)]
            f.write_bytes(pickle.dumps(
                (arr, [{"wsi_loc": [x, y]} for x, y in xy]), protocol=4))
    monkeypatch.setenv("PROSPER_OUT", str(out))
    assert audit.main(["--spec", str(path), "--spec-sha256", sha(path),
                       "--out", str(out / "run")]) == 0
    assert "WARNING" not in capsys.readouterr().err
    assert json.loads((out / "run/diagnostics.json").read_text())["entries"] == []
    assert json.loads((out / "run/summary.json").read_text())["lineage_order_equal"] is True


def test_global_order_difference_even_when_both_csv_orders_agree(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    write_csv(root / "labels/A.csv",
              ("slide_id", "wsi_loc_x", "wsi_loc_y", "patch_label"),
              (("A", 0, 0, 1), ("A", 512, 0, 2),
               ("A", 1024, 0, 4), ("A", 1536, 0, 0)))
    write_csv(root / "virchow2/slide_info.csv",
              ("slide_id", "wsi_loc_x", "wsi_loc_y"),
              (("Q", 0, 0), ("A", 0, 0), ("A", 512, 0), ("B", 0, 0)))
    write_npy(root / "virchow2/idx_train_rand.npy", [0, 2])
    write_npy(root / "virchow2/idx_test_rand.npy", [1, 3])
    write_npy(root / "virchow2/y_train_rand.npy", [3, 2])
    write_npy(root / "virchow2/y_test_rand.npy", [1, 5])
    run(path, out)
    summary = json.loads((out / "run/summary.json").read_text())
    entries = json.loads((out / "run/diagnostics.json").read_text())["entries"]
    assert summary["lineage_csv_target_order_equal"] == {"uni2": True, "virchow2": True}
    assert summary["lineage_order_equal"] is False
    global_difference = next(entry for entry in entries
                             if entry["kind"] == "lineage_global_order")
    assert global_difference["count"] == 4
    assert [example["position"] for example in global_difference["examples"]] == [0, 1, 2, 3]
    a0 = next(row for row in csv_rows(out / "run/eligible_tiles.csv")
              if row["slide_id"] == "A" and row["x"] == "0")
    assert (a0["uni2_legacy_row"], a0["virchow2_legacy_row"]) == ("0", "1")
    assert (out / "run/COMPLETE.json").exists()


def test_zero_quarantine_still_writes_declared_header(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    write_csv(root / "uni2/slide_info.csv",
              ("slide_id", "wsi_loc_x", "wsi_loc_y"),
              (("A", 0, 0), ("A", 512, 0), ("B", 0, 0)))
    write_csv(root / "virchow2/slide_info.csv",
              ("slide_id", "wsi_loc_x", "wsi_loc_y"),
              (("A", 512, 0), ("B", 0, 0), ("A", 0, 0)))
    write_npy(root / "uni2/idx_train_rand.npy", [0, 2])
    write_npy(root / "uni2/idx_test_rand.npy", [1])
    write_npy(root / "uni2/y_test_rand.npy", [2])
    write_npy(root / "virchow2/idx_train_rand.npy", [0, 2])
    write_npy(root / "virchow2/idx_test_rand.npy", [1])
    write_npy(root / "virchow2/y_train_rand.npy", [2, 1])
    write_npy(root / "virchow2/y_test_rand.npy", [5])
    spec["expected"].update(historical_rows=3, historical_slides=2, test_rows=1,
                            quarantined_rows=0, quarantined_slides=0,
                            quarantined_class_counts={str(i): 0 for i in range(1, 6)})
    write_json(path, spec)
    run(path, out)
    assert (out / "run/quarantine_tiles.csv").read_text().splitlines() == [
        ",".join(audit.QUARANTINE_HEADER)]
    assert json.loads((out / "run/summary.json").read_text())["quarantine"]["rows"] == 0


def test_missing_quarantine_source_is_error(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    (root / "labels/Q.csv").unlink()
    fails(path, out, "No such file")


@pytest.mark.parametrize("output,field,bad", [
    ("eligible_tiles.csv", "tissue_label", "5"),
    ("eligible_tiles.csv", "virchow2_legacy_row", "999"),
    ("eligible_tiles.csv", "h_optimus_1_feature_row", "999"),
    ("quarantine_tiles.csv", "uni2_legacy_row", "999"),
    ("feature_inventory.csv", "unused_unannotated", "999"),
    ("slide_inventory.csv", "target_rows", "999"),
])
def test_corrupt_written_output_cannot_complete(tmp_path, monkeypatch, output, field, bad):
    root, out, path, spec = fixture(tmp_path)
    original_write = audit._atomic_bytes

    def corrupt(fd, name, data):
        if name == output:
            lines = list(csv.reader(io.StringIO(data.decode("utf-8"))))
            col = lines[0].index(field)
            lines[1][col] = bad
            stream = io.StringIO(newline="")
            writer = csv.writer(stream)
            writer.writerows(lines)
            data = stream.getvalue().encode("utf-8")
        return original_write(fd, name, data)

    monkeypatch.setattr(audit, "_atomic_bytes", corrupt)
    fails(path, out, "written .*validated bytes changed|written CSV")


def test_corrupt_summary_support_cannot_complete(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    original_write = audit._atomic_bytes

    def corrupt(fd, name, data):
        if name == "summary.json":
            obj = json.loads(data)
            obj["eligible"]["class_patients"]["1"] = 999
            data = json.dumps(obj).encode("utf-8")
        return original_write(fd, name, data)

    monkeypatch.setattr(audit, "_atomic_bytes", corrupt)
    fails(path, out, "written summary.json: validated bytes changed|written JSON")


@pytest.mark.parametrize("kind", ["patient_support", "diagnostic_count"])
def test_serialized_json_fault_rejected_by_semantic_readback(tmp_path, monkeypatch, kind):
    root, out, path, spec = fixture(tmp_path)
    original = audit.stage0._encoded

    def forge(obj):
        if isinstance(obj, dict) and kind == "patient_support" and "eligible" in obj:
            changed = json.loads(json.dumps(obj))
            changed["eligible"]["class_patients"]["1"] = 999
            return original(changed)
        if isinstance(obj, dict) and kind == "diagnostic_count" and "entries" in obj:
            changed = json.loads(json.dumps(obj))
            match = next(entry for entry in changed["entries"]
                         if entry["kind"] == "feature_csv_order")
            match["count"] = 999
            return original(changed)
        return original(obj)

    monkeypatch.setattr(audit.stage0, "_encoded", forge)
    fails(path, out, "written summary: eligible support mismatch|written diagnostics: "
          "missing=|written diagnostics: altered")
    assert (out / "run/FAILED.json").exists()


@pytest.mark.parametrize("output,column,value", [
    ("eligible_tiles.csv", "level", "1"),
    ("eligible_tiles.csv", "patch_size", "256"),
    ("quarantine_tiles.csv", "level", "1"),
    ("quarantine_tiles.csv", "patch_size", "256"),
])
def test_serialized_physical_fields_cannot_reuse_old_tile_id(tmp_path, monkeypatch,
                                                              output, column, value):
    root, out, path, spec = fixture(tmp_path)
    original = audit._write_csv

    def forge(fd, name, header, rows):
        if name == output:
            first = list(rows[0])
            first[header.index(column)] = value
            rows[0] = tuple(first)
        return original(fd, name, header, rows)

    monkeypatch.setattr(audit, "_write_csv", forge)
    fails(path, out, "wrong physical level or patch size")


def test_existing_destination_rejected_before_canonical_and_bank_read(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    (out / "run").mkdir()
    (out / "run/private").write_bytes(b"untouched")
    original = audit._read_capped
    consulted = []

    def guard(path, limit):
        if path.name == "all_patches_combined.pkl" or (
            path.suffix == ".pkl" and path.parent.name in audit.BANKS):
            consulted.append(str(path))
            raise AssertionError("large input consulted before exclusive claim")
        return original(path, limit)

    monkeypatch.setattr(audit, "_read_capped", guard)
    fails(path, out, "File exists")
    assert consulted == []
    assert (out / "run/private").read_bytes() == b"untouched"
    assert not (out / "run/FAILED.json").exists()


def test_label_conflict_after_claim_preserves_failure_context(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    labels = root / "labels/A.csv"
    labels.write_text(labels.read_text().replace("A,0,0,1", "A,0,0,4"))
    fails(path, out, "missing/conflicting target")
    failed_record = json.loads((out / "run/FAILED.json").read_text())
    assert "missing/conflicting target" in failed_record["error"]
    assert not (out / "run/COMPLETE.json").exists()


def test_final_inventory_root_retarget(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    other = root / "features/other_uni2"
    other.mkdir()
    for sid in ("A", "B"):
        (other / f"{sid}.pkl").write_bytes((root / "features/uni2" / f"{sid}.pkl").read_bytes())
    alias = root / "features/uni2_alias"
    alias.symlink_to(root / "features/uni2", target_is_directory=True)
    spec["banks"][0]["path"] = "features/uni2_alias"
    write_json(path, spec)
    original_write = audit._atomic_bytes

    def retarget(fd, name, data):
        result = original_write(fd, name, data)
        if name == "feature_inventory.csv":
            alias.unlink()
            alias.symlink_to(other, target_is_directory=True)
        return result

    monkeypatch.setattr(audit, "_atomic_bytes", retarget)
    fails(path, out, "input root uni2: late retarget")


def test_unowned_nonempty_directory_is_never_overwritten(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    other = out / "preexisting"
    other.mkdir()
    (other / "FAILED.json").write_bytes(b"keep")
    mkdir = os.mkdir

    def replace(name, *args, **kwargs):
        result = mkdir(name, *args, **kwargs)
        if name == "run" and kwargs.get("dir_fd") is not None:
            (out / name).rename(out / "created")
            other.rename(out / name)
        return result

    monkeypatch.setattr(audit.os, "mkdir", replace)
    fails(path, out, "newly claimed output directory is not empty")
    assert (out / "run/FAILED.json").read_bytes() == b"keep"


def test_final_output_readback_truncation(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    original = audit._read_output

    def truncate(fd, name):
        data = original(fd, name)
        if name == "quarantine_tiles.csv":
            (out / "run/quarantine_tiles.csv").write_bytes(data[:-1])
        return data

    monkeypatch.setattr(audit, "_read_output", truncate)
    fails(path, out, "written CSV|late output change|final output hash changed")


@pytest.mark.parametrize("target,match", [
    ("review_sha", "review SHA256 mismatch"),
    ("artifact", "artifact hash mismatch"),
    ("launch", "review/launch source or spec hashes disagree"),
    ("complete", "COMPLETE: inventory.csv output hash"),
    ("metadata", "Stage1 metadata bytes differ"),
    ("columns", "column selectors disagree"),
    ("helper", "helper hash differs"),
])
def test_upstream_crosslinks(tmp_path, target, match):
    root, out, path, spec = fixture(tmp_path)
    review_path = root / spec["stage0"]["review"]
    review = json.loads(review_path.read_text())
    if target == "review_sha":
        spec["stage0"]["review_sha256"] = "0" * 64
    elif target == "artifact":
        (root / "stage0/stage0_job.lsf").write_text("changed")
    elif target == "launch":
        launch = root / "stage0/stage0_launch.sha256"
        launch.write_text(launch.read_text().replace("c" * 64, "d" * 64))
        review["artifacts"]["stage0_launch.sha256"] = sha(launch)
    elif target == "complete":
        complete_path = root / spec["stage0"]["complete"]
        complete = json.loads(complete_path.read_text())
        complete["outputs"]["inventory.csv"]["sha256"] = "0" * 64
        write_json(complete_path, complete)
        review["artifacts"]["stage0_attempt_001/COMPLETE.json"] = sha(complete_path)
    elif target == "metadata":
        (root / "metadata.csv").write_text((root / "metadata.csv").read_text() + "\n")
    elif target == "columns":
        spec["metadata"]["patient_col"] = "extra"
    else:
        review["source_sha256"]["auditor_module"] = "0" * 64
        launch = root / "stage0/stage0_launch.sha256"
        launch.write_text(launch.read_text().replace(sha(stage0.__file__), "0" * 64))
        review["artifacts"]["stage0_launch.sha256"] = sha(launch)
        review["artifacts"]["stage0_attempt_001/summary.json"] = sha(
            root / "stage0/stage0_attempt_001/summary.json")
        upstream_summary = root / "stage0/stage0_attempt_001/summary.json"
        summary = json.loads(upstream_summary.read_text())
        summary["input_bindings"][3]["sha256"] = "0" * 64
        write_json(upstream_summary, summary)
        review["artifacts"]["stage0_attempt_001/summary.json"] = sha(upstream_summary)
        complete_path = root / spec["stage0"]["complete"]
        complete = json.loads(complete_path.read_text())
        complete["outputs"]["summary.json"] = {
            "size": upstream_summary.stat().st_size, "sha256": sha(upstream_summary)}
        write_json(complete_path, complete)
        review["artifacts"]["stage0_attempt_001/COMPLETE.json"] = sha(complete_path)
    if target in ("complete", "helper", "launch"):
        write_json(review_path, review)
        spec["stage0"]["review_sha256"] = sha(review_path)
    write_json(path, spec)
    fails(path, out, match)


def test_output_mutation_and_late_input(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    original = audit._read_output

    def mutate(directory_fd, name):
        data = original(directory_fd, name)
        if name == "summary.json":
            (out / "run/eligible_tiles.csv").write_bytes(
                (out / "run/eligible_tiles.csv").read_bytes()[:-1])
        return data

    monkeypatch.setattr(audit, "_read_output", mutate)
    fails(path, out, "written CSV|late output change")
    assert (out / "run/FAILED.json").exists()


def test_late_code_mutation(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    source = root / "source.py"
    original_write = audit._atomic_bytes

    def change(directory_fd, name, data):
        result = original_write(directory_fd, name, data)
        if name == "feature_inventory.csv":
            stat = source.stat()
            source.write_bytes(b"version = 2\n")
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        return result

    monkeypatch.setattr(audit, "_atomic_bytes", change)
    fails(path, out, "late input mutation")


def test_existing_concurrent_and_unowned_output(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    (out / "run").mkdir()
    (out / "run/private").write_text("preserve")
    fails(path, out, "File exists")
    assert not (out / "run/FAILED.json").exists()
    assert (out / "run/private").read_text() == "preserve"
    original_claimed = audit._claimed_path

    def reject(*args):
        raise audit.AuditError("unowned directory")

    monkeypatch.setattr(audit, "_claimed_path", reject)
    fails(path, out, "unowned directory", "second")
    assert list((out / "second").iterdir()) == []
    monkeypatch.setattr(audit, "_claimed_path", original_claimed)


def test_concurrent_exclusive_claim(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    gate = Barrier(2)
    original = audit._read_capped
    feature_reads = []

    def record(path, limit):
        if path.suffix == ".pkl" and path.parent.name in audit.BANKS:
            feature_reads.append(str(path))
        return original(path, limit)

    monkeypatch.setattr(audit, "_read_capped", record)

    def invoke():
        gate.wait()
        try:
            run(path, out)
            return "won"
        except FileExistsError:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: invoke(), range(2)))
    assert sorted(results) == ["lost", "won"]
    assert len(feature_reads) == 6
    assert (out / "run/COMPLETE.json").exists()
    assert not (out / "run/FAILED.json").exists()
