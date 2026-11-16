"""Tiny trusted synthetic fixtures; no real banks, slides, GPUs or downloads."""

import csv
import hashlib
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from prosper.evaluation import feature_bank_audit as audit


def dump(path, obj):
    path.write_bytes(pickle.dumps(obj, protocol=4))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    (root / "extractor.py").write_text("version = 1\n")
    (root / "metadata.csv").write_text(
        "Slide.Scan.File,UID.Subject,Tumor.group,extra\n"
        "001,patient,A,keep\n002,patient,B,also keep\n")
    coords = {
        "001": [{"wsi_loc": [0, 0], "loc": [1, 2]}, {"wsi_loc": [512, 0]}],
        "002": [{"wsi_loc": [0, 0]}],
    }
    dump(root / "mosaic.pkl", [(f"{sid}.svs", tiles) for sid, tiles in coords.items()])
    for bank, dim in (("a", 4), ("b", 3)):
        folder = root / bank
        folder.mkdir()
        for sid, tiles in coords.items():
            dtype = np.float16 if (bank == "b" and sid == "001") else np.float32
            dump(folder / f"{sid}.pkl", (
                np.ones((len(tiles), dim), dtype=dtype), tiles))
    spec = {
        "schema_version": 1, "run_id": "fixture",
        "metadata": {"path": "metadata.csv", "id_col": "Slide.Scan.File",
                     "patient_col": "UID.Subject", "label_col": "Tumor.group"},
        "mosaic": {"path": "mosaic.pkl", "filename_suffix": ".svs",
                   "level": 0, "patch_size": 512, "expected_slides": 2,
                   "expected_tiles": 3},
        "banks": [
            {"name": "a", "path": "a", "feature_dim": 4, "dtypes": ["float16", "float32"]},
            {"name": "b", "path": "b", "feature_dim": 3, "dtypes": ["float16", "float32"]},
        ],
        "code_files": [{"name": "extractor", "path": "extractor.py"}],
    }
    spec_path = root / "spec.json"
    write_spec(spec_path, spec)
    return root, out_root, spec_path, spec


def write_spec(path, spec):
    path.write_text(json.dumps(spec))


def run(spec_path, out_root, name="run", expected_hash=None):
    return audit.audit(spec_path, expected_hash or sha(spec_path),
                       out_root / name, str(out_root))


def failed(spec_path, out_root, match, *, name="run", expected_hash=None):
    with pytest.raises(Exception, match=match):
        run(spec_path, out_root, name=name, expected_hash=expected_hash)
    assert not (out_root / name / "COMPLETE.json").exists()


def test_valid_multi_bank_receipts_and_metadata(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    receipt = run(path, out)
    assert (out / "run").is_dir()
    assert not (out / "run").is_symlink()
    summary = json.loads((out / "run/summary.json").read_text())
    with (out / "run/inventory.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {(r["bank"], r["slide_id"]) for r in rows} == {
        ("a", "001"), ("a", "002"), ("b", "001"), ("b", "002")}
    assert summary["bank_dtype_counts"]["b"] == {"float16": 1, "float32": 1}
    assert summary["metadata_columns"][-1] == "extra"
    assert summary["multi_label_patient_count"] == 1
    assert summary["bank_tile_totals"] == {"a": 3, "b": 3}
    assert {b["kind"] for b in summary["input_bindings"]} == {
        "spec", "metadata", "mosaic", "feature", "code"}
    assert len([b for b in summary["input_bindings"] if b["kind"] == "feature"]) == 4
    for filename, binding in receipt["outputs"].items():
        assert sha(out / "run" / filename) == binding["sha256"]
    assert "COMPLETE.json" not in receipt["outputs"]
    assert (out / "run/COMPLETE.json").is_file()
    monkeypatch.setenv("PROSPER_OUT", str(out))
    assert audit.main(["--spec", str(path), "--spec-sha256", sha(path),
                       "--out", str(out / "another")]) == 0


def test_wrong_hash_and_json_csv_id_validation(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    failed(path, out, "SHA256 mismatch", expected_hash="0" * 64)
    assert not (out / "run").exists()
    path.write_text('{"schema_version":1,"schema_version":1}')
    failed(path, out, "duplicate JSON key")
    write_spec(path, spec)
    metadata = root / "metadata.csv"
    original = metadata.read_text()
    for text, match in [
        ("Slide.Scan.File,Slide.Scan.File,UID.Subject,Tumor.group\n"
         "001,001,P,A\n", "duplicate CSV headers"),
        (original.replace("002,patient", "001,patient"), "duplicate scan ID"),
        (original.replace("001,patient", " ,patient"), "nonempty string"),
        (original.replace("001,patient", "001, "), "nonempty string"),
        (original.replace("002,patient,B", "002,patient, "), "nonempty string"),
        (original.replace("001,patient,A,keep", "001,patient,A,keep,extra"), "fields"),
    ]:
        metadata.write_text(text)
        failed(path, out, match, name=f"bad-{len(list(out.iterdir()))}")
    metadata.write_text(original)


@pytest.mark.parametrize("change,match", [
    (lambda s: s.update(schema_version=True), "schema_version"),
    (lambda s: s.update(unknown=True), "unknown"),
    (lambda s: s["mosaic"].update(level=True), "mosaic.level"),
    (lambda s: s["mosaic"].update(patch_size=0), "mosaic.patch_size"),
    (lambda s: s["mosaic"].update(expected_tiles=False), "mosaic.expected_tiles"),
    (lambda s: s["banks"][0].update(feature_dim=True), "feature_dim"),
    (lambda s: s["banks"][0].update(dtypes=["float64"]), "dtypes"),
    (lambda s: s.update(code_files=[]), "code_files"),
    (lambda s: s["banks"][1].update(name="a"), "duplicate bank name"),
    (lambda s: s["banks"][1].update(path="a"), "duplicate/resolved bank roots"),
])
def test_schema(tmp_path, change, match):
    root, out, path, spec = fixture(tmp_path)
    change(spec)
    write_spec(path, spec)
    failed(path, out, match)


@pytest.mark.parametrize("loc", [
    [0.0, 0], [True, 0], [0], [0, 0, 1], "0,0", [np.int64(0), 0],
])
def test_invalid_canonical_coordinates(tmp_path, loc):
    root, out, path, spec = fixture(tmp_path)
    entries = pickle.loads((root / "mosaic.pkl").read_bytes())
    entries[0][1][0]["wsi_loc"] = loc
    dump(root / "mosaic.pkl", entries)
    failed(path, out, "wsi_loc")


def test_duplicate_coords_and_bad_canonical_shape(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    entries = pickle.loads((root / "mosaic.pkl").read_bytes())
    entries[0][1][1]["wsi_loc"] = [0, 0]
    dump(root / "mosaic.pkl", entries)
    failed(path, out, "duplicate physical coordinate")
    dump(root / "mosaic.pkl", [(name, tiles) for name, tiles in entries] +
         [("001.svs", [{"wsi_loc": [1, 0]}])])
    failed(path, out, "duplicate physical coordinate", name="bad-second")


def test_bank_missing_extra_and_aliases(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    (root / "a/002.pkl").unlink()
    failed(path, out, "missing=")
    dump(root / "a/002.pkl", (np.ones((1, 4), dtype=np.float32), [{"wsi_loc": [0, 0]}]))
    (root / "a/extra.pkl").write_bytes((root / "a/001.pkl").read_bytes())
    failed(path, out, "extra=", name="extra")
    (root / "a/extra.pkl").unlink()
    (root / "b/002.pkl").unlink()
    (root / "b/002.pkl").symlink_to(root / "a/002.pkl")
    failed(path, out, "file alias", name="alias")
    spec["banks"][1]["path"] = "alias-root"
    (root / "alias-root").symlink_to(root / "a", target_is_directory=True)
    write_spec(path, spec)
    failed(path, out, "duplicate/resolved bank roots", name="root-alias")


@pytest.mark.parametrize("mutation,match", [
    (lambda f, t: (np.ones((2, 3), np.float32), t), "shape/width/rows"),
    (lambda f, t: (np.ones((1, 4), np.float32), t), "shape/width/rows"),
    (lambda f, t: (np.ones((2, 4), np.float64), t), "dtype"),
    (lambda f, t: (np.ones((2, 4), np.int32), t), "dtype"),
    (lambda f, t: (np.full((2, 4), np.nan, np.float32), t), "NaN/Inf"),
    (lambda f, t: (np.full((2, 4), np.inf, np.float32), t), "NaN/Inf"),
    (lambda f, t: (f, list(reversed(t))), "ordered coordinate mismatch"),
    (lambda f, t: (f, t + [{"wsi_loc": [100, 100]}]), "ordered coordinate mismatch"),
    (lambda f, t: [f, t], "expected \\(features, mosaic\\) tuple"),
    (lambda f, t: (f, [{"wsi_loc": [0, 0]}] * 2), "duplicate physical coordinate"),
])
def test_bad_feature_objects(tmp_path, mutation, match):
    root, out, path, spec = fixture(tmp_path)
    f, tiles = pickle.loads((root / "a/001.pkl").read_bytes())
    dump(root / "a/001.pkl", mutation(f, tiles))
    failed(path, out, match)
    assert (out / "run/FAILED.json").is_file()


def test_pickle_trailing_and_malformed(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    feature = root / "a/001.pkl"
    feature.write_bytes(feature.read_bytes() + pickle.dumps("extra"))
    failed(path, out, "trailing bytes")
    feature.write_bytes(b"invalid pickle")
    failed(path, out, "invalid pickle", name="malformed")


def test_framed_pickle_with_additional_object_inside_first_frame(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    feature = root / "a/001.pkl"
    raw = feature.read_bytes()
    assert raw.startswith(b"\x80\x04\x95")
    assert int.from_bytes(raw[3:11], "little") == len(raw) - 11
    appended = pickle.dumps("second object", protocol=4)
    forged = raw[:3] + (len(raw) - 11 + len(appended)).to_bytes(8, "little")
    feature.write_bytes(forged + raw[11:] + appended)
    assert pickle.loads(feature.read_bytes())[0].shape == (2, 4)
    failed(path, out, "trailing bytes or additional serialized objects")


def test_symlink_parent_traversal_uses_actual_target(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    external = tmp_path / "external"
    (external / "child").mkdir(parents=True)
    (external / "extractor.py").write_text("version = 2\n")
    (root / "linked").symlink_to(external / "child", target_is_directory=True)
    spec["code_files"][0]["path"] = "linked/../extractor.py"
    write_spec(path, spec)
    run(path, out)
    bindings = json.loads((out / "run/summary.json").read_text())["input_bindings"]
    code = next(b for b in bindings if b["kind"] == "code" and b["name"] == "extractor")
    assert code["path"] == str(root / "linked/../extractor.py")
    assert code["resolved_path"] == str(external / "extractor.py")
    assert code["sha256"] == sha(external / "extractor.py")
    assert code["sha256"] != sha(root / "extractor.py")


def test_unsafe_existing_concurrent_output(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    with pytest.raises(audit.AuditError, match="PROSPER_OUT"):
        audit.audit(path, sha(path), out / "run", "")
    failed(path, out, "beneath PROSPER_OUT", name="../inputs/a/run")
    failed(path, out, "beneath PROSPER_OUT", name="../outside")
    (out / "run").mkdir()
    (out / "run/winner.txt").write_text("preserve")
    failed(path, out, "File exists")
    assert not (out / "run/FAILED.json").exists()
    assert (out / "run/winner.txt").read_text() == "preserve"
    (out / "symlink").symlink_to(root / "a", target_is_directory=True)
    failed(path, out, "beneath PROSPER_OUT|overlaps", name="symlink/child")
    with pytest.raises(audit.AuditError, match="overlaps bank"):
        audit.audit(path, sha(path), root / "a" / "new", str(root / "a"))


def test_concurrent_exclusive_claim(tmp_path):
    root, out, path, spec = fixture(tmp_path)
    gate = Barrier(2)

    def invoke():
        gate.wait()
        try:
            return run(path, out)
        except FileExistsError:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: invoke(), range(2)))
    assert sum(r == "lost" for r in results) == 1
    assert (out / "run/COMPLETE.json").exists()
    assert not (out / "run/FAILED.json").exists()


@pytest.mark.parametrize("preexisting_failure", [False, True])
def test_create_open_substitution_rejects_existing_directory(tmp_path, monkeypatch,
                                                              preexisting_failure):
    root, out, path, spec = fixture(tmp_path)
    third_party = out / "third-party"
    third_party.mkdir()
    contents = {"private.txt": b"untouched"}
    contents["FAILED.json" if preexisting_failure else "inventory.csv"] = (
        b"previous failure" if preexisting_failure else b"other inventory")
    for name, data in contents.items():
        (third_party / name).write_bytes(data)
    real_mkdir = os.mkdir
    substituted = []

    def substitute(name, *args, **kwargs):
        result = real_mkdir(name, *args, **kwargs)
        if name == "run" and kwargs.get("dir_fd") is not None:
            (out / "run").rename(out / "actual-created-directory")
            third_party.rename(out / "run")
            substituted.append(True)
        return result

    monkeypatch.setattr(audit.os, "mkdir", substitute)
    failed(path, out, "newly claimed output directory is not empty")
    assert substituted == [True]
    assert list((out / "actual-created-directory").iterdir()) == []
    assert {p.name: p.read_bytes() for p in (out / "run").iterdir()} == contents


def test_observed_identity_changes_before_open(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    replacement = out / "empty-replacement"
    replacement.mkdir()
    real_open = os.open

    def substitute(name, flags, *args, **kwargs):
        if name == "run" and flags & os.O_DIRECTORY:
            (out / "run").rename(out / "actual-created-directory")
            replacement.rename(out / "run")
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(audit.os, "open", substitute)
    failed(path, out, "output directory changed during acquisition")
    assert list((out / "run").iterdir()) == []
    assert list((out / "actual-created-directory").iterdir()) == []


@pytest.mark.parametrize("target", [
    "a/001.pkl", "b/002.pkl", "metadata.csv", "mosaic.pkl", "spec.json", "extractor.py",
])
def test_late_same_size_same_timestamp_mutations(tmp_path, monkeypatch, target):
    root, out, path, spec = fixture(tmp_path)
    original_recheck = audit._recheck
    source = root / target

    def mutate(*args):
        stat = source.stat()
        blob = bytearray(source.read_bytes())
        blob[-1] ^= 1
        source.write_bytes(blob)
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        return original_recheck(*args)

    monkeypatch.setattr(audit, "_recheck", mutate)
    failed(path, out, "late input mutation")
    assert (out / "run/FAILED.json").exists()


@pytest.mark.parametrize("target", ["root", "file", "add", "delete"])
def test_late_inventory_and_symlink_retarget(tmp_path, monkeypatch, target):
    root, out, path, spec = fixture(tmp_path)
    original_recheck = audit._recheck
    if target == "root":
        (root / "real-b").mkdir()
        for p in (root / "b").iterdir():
            (root / "real-b" / p.name).write_bytes(p.read_bytes())
        (root / "link-b").symlink_to(root / "b", target_is_directory=True)
        spec["banks"][1]["path"] = "link-b"
        write_spec(path, spec)
    if target == "file":
        p = root / "a/001.pkl"
        (root / "alias-file.pkl").write_bytes(p.read_bytes())
        p.unlink()
        p.symlink_to(root / "alias-file.pkl")

    def mutate(*args):
        if target == "root":
            (root / "link-b").unlink()
            (root / "link-b").symlink_to(root / "real-b", target_is_directory=True)
        elif target == "file":
            (root / "a/001.pkl").unlink()
            (root / "a/001.pkl").symlink_to(root / "b/001.pkl")
        elif target == "add":
            (root / "a/extra.pkl").write_bytes(b"new")
        else:
            (root / "a/001.pkl").unlink()
        return original_recheck(*args)

    monkeypatch.setattr(audit, "_recheck", mutate)
    failed(path, out, "retargeted|inventory|late input mutation|missing=")


@pytest.mark.parametrize("filename", ["inventory.csv", "summary.json"])
def test_truncated_output_never_completes(tmp_path, monkeypatch, filename):
    root, out, path, spec = fixture(tmp_path)
    original = audit._atomic_bytes

    def truncate(directory_fd, name, data):
        if name == filename:
            data = data[:len(data) // 2]
        return original(directory_fd, name, data)

    monkeypatch.setattr(audit, "_atomic_bytes", truncate)
    failed(path, out, "written inventory|written summary.json|manifest")


def test_inventory_modified_during_summary_readback(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    original_json = audit._json

    def mutate(data):
        if b'"input_bindings"' in data:
            inventory = out / "run/inventory.csv"
            inventory.write_bytes(b"\n".join(inventory.read_bytes().splitlines()[:-1]) + b"\n")
        return original_json(data)

    monkeypatch.setattr(audit, "_json", mutate)
    failed(path, out, "written inventory.csv: changed during readback")
    assert (out / "run/FAILED.json").exists()


@pytest.mark.parametrize("inject_failure", [False, True])
def test_output_alias_retarget_cannot_write_to_unclaimed_dir(tmp_path, monkeypatch,
                                                              inject_failure):
    root, out, path, spec = fixture(tmp_path)
    original = out / "original"
    other = out / "other"
    original.mkdir()
    other.mkdir()
    (other / "run").mkdir()
    (other / "run/private.txt").write_text("third-party content")
    current = out / "current"
    current.symlink_to(original, target_is_directory=True)
    original_recheck = audit._recheck

    def retarget(*args):
        current.unlink()
        current.symlink_to(other, target_is_directory=True)
        if inject_failure:
            raise audit.AuditError("injected validation failure")
        return original_recheck(*args)

    monkeypatch.setattr(audit, "_recheck", retarget)
    with pytest.raises(audit.AuditError, match="injected validation failure|retargeted"):
        run(path, out, name="current/run")
    assert (original / "run/FAILED.json").exists()
    assert not (original / "run/COMPLETE.json").exists()
    assert sorted(p.name for p in (other / "run").iterdir()) == ["private.txt"]
    assert (other / "run/private.txt").read_text() == "third-party content"


def test_transient_alias_retarget_writes_only_to_claimed_dir(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    original = out / "original"
    other = out / "other"
    original.mkdir()
    other.mkdir()
    (other / "run").mkdir()
    (other / "run/private.txt").write_text("preserve")
    current = out / "current"
    current.symlink_to(original, target_is_directory=True)
    original_write = audit._atomic_bytes
    original_recheck = audit._recheck
    moved = []

    def retarget(directory_fd, name, data):
        if name == "inventory.csv":
            current.unlink()
            current.symlink_to(other, target_is_directory=True)
            moved.append(True)
        return original_write(directory_fd, name, data)

    def restore(*args):
        assert moved
        assert (original / "run/inventory.csv").exists()
        assert sorted(p.name for p in (other / "run").iterdir()) == ["private.txt"]
        current.unlink()
        current.symlink_to(original, target_is_directory=True)
        return original_recheck(*args)

    monkeypatch.setattr(audit, "_atomic_bytes", retarget)
    monkeypatch.setattr(audit, "_recheck", restore)
    run(path, out, name="current/run")
    assert (original / "run/COMPLETE.json").exists()
    assert sorted(p.name for p in (other / "run").iterdir()) == ["private.txt"]
    assert (other / "run/private.txt").read_text() == "preserve"


def test_unestablished_claim_does_not_write_failure(tmp_path, monkeypatch):
    root, out, path, spec = fixture(tmp_path)
    original = audit._claimed_path

    def reject(directory_fd, target):
        raise audit.AuditError("ownership could not be established")

    monkeypatch.setattr(audit, "_claimed_path", reject)
    failed(path, out, "ownership could not be established")
    assert list((out / "run").iterdir()) == []
    monkeypatch.setattr(audit, "_claimed_path", original)
