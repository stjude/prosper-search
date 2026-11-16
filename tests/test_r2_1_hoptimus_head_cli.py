"""Bound toy controls and tiny features only; no private workspace inputs."""
from collections import Counter
import csv
from dataclasses import asdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from prosper.evaluation import r2_1_hoptimus_head_cli as cli


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jbytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode()


def csv_bytes(header, rows, compressed=False):
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    data = text.getvalue().encode()
    return gzip.compress(data, mtime=0) if compressed else data


class Toy:
    def __init__(self, tmp_path, monkeypatch, *, fold=0, budget=5, zero=False):
        self.work = tmp_path / "toy"
        self.work.mkdir()
        self.controls = self.work / "controls"
        self.controls.mkdir()
        self.bank = self.work / "features"
        self.bank.mkdir()
        self.root = self.work / "out"
        self.root.mkdir()
        monkeypatch.setenv("PROSPER_OUT", str(self.root))
        self.out = self.root / "attempt"
        self.spec_path = self.work / "spec.json"
        self.records, self.raw, self.feature_entries = [], {}, []
        patients = [f"p{i}" for i in range(6)] + ["immune_a", "immune_b", "retrieval_only"]
        ordered = sorted(patients, key=lambda p: (cli.kernel.patient_fold_hash(p), p))
        immune = [p for p in ordered if p.startswith("immune")]
        self.mapping = {p: f for f, p in enumerate(immune)}
        sizes = [1, 1, 0, 0, 0]
        for patient in ordered:
            if patient not in self.mapping:
                f = min(range(5), key=lambda i: (sizes[i], i))
                self.mapping[patient] = f
                sizes[f] += 1
        for patient in patients[:-1]:
            slide = "slide_" + patient
            labels = [1, 3, 4, 5] + ([2] if patient.startswith("immune") else [])
            raw = np.zeros((len(labels) + 2, 1536), np.float32)
            coords = [(i * 512, 512) for i in range(len(raw))]
            for index, label in enumerate(labels):
                source_row = len(labels) - index - 1
                group = "NORMAL" if patient == "immune_a" else "MISC" if patient == "p0" else "WT"
                self.records.append(cli.kernel.AnnotationRecord(
                    cli.kernel.physical_tile_id(slide, *coords[source_row]),
                    slide, patient, group, 0, *coords[source_row], 512, label, source_row))
            if not zero:
                for index in range(len(raw)):
                    raw[index, 0] = (index + 1) * 0.001
                    raw[index, 10] = (len(patient) + index) * 0.0001
            self.raw[slide] = (raw.copy(), coords)
            payload = pickle.dumps((raw, [{"wsi_loc": list(xy)} for xy in coords]), protocol=4)
            path = self.bank / (slide + ".pkl")
            path.write_bytes(payload)
            self.feature_entries.append(dict(
                slide_id=slide, path=str(path), sha256=sha(payload), size=len(payload),
                feature_rows=len(raw), feature_dim=1536, dtype="float32"))
        # Deliberately choose a presentation distinct from physical/hash order.
        self.records = tuple(reversed(self.records))
        annotation_patients = {r.patient_id for r in self.records}
        self.source_paths = {role: self.controls / (role + ".json") for role in cli.SOURCE_LIMITS}
        rows = [(p, self.mapping[p], ordered.index(p), cli.kernel.patient_fold_hash(p),
                 int(p in annotation_patients), 1) for p in sorted(patients)]
        self.source_paths["patient_folds"].write_bytes(csv_bytes(cli.manifests.FOLD_HEADER, rows))
        self.source_paths["annotation_assignments"].write_bytes(csv_bytes(
            cli.manifests.ANNOTATION_HEADER,
            [(r.tile_id, r.slide_id, r.patient_id, r.diagnostic_group, r.level, r.x, r.y,
              r.patch_size, r.tissue_label, r.h_optimus_1_feature_row, self.mapping[r.patient_id])
             for r in self.records], True))
        inventory = []
        for entry in self.feature_entries:
            slide = entry["slide_id"]
            target = sum(r.slide_id == slide for r in self.records)
            inventory.append((
                "h_optimus_1", slide, 1536, 0, 1536, entry["feature_rows"], "float32",
                target, entry["feature_rows"] - target, 0, 0, "false", 1,
                entry["path"], str(Path(entry["path"]).resolve()), entry["size"], entry["sha256"]))
        self.source_paths["feature_inventory"].write_bytes(csv_bytes(
            cli.manifests.FEATURE_HEADER, inventory))
        heads, training = [], []
        self.manifests = {}
        globally_ordered = sorted(self.records,
                                  key=lambda r: (cli.kernel.training_tile_hash(r), r.physical_key))
        for f in range(5):
            pool = [r for r in globally_ordered if self.mapping[r.patient_id] != f]
            by_class = {label: [r for r in pool if r.tissue_label == label] for label in range(1, 6)}
            selected_ids = {r.tile_id for values in by_class.values()
                            for r in values[:cli.kernel.five_percent_count(len(values))]}
            for b in (5, 100):
                selected = tuple(r for r in pool if b == 100 or r.tile_id in selected_ids)
                counts = Counter(r.tissue_label for r in selected)
                stats = tuple(cli.kernel.ClassSelection(
                    label, len(by_class[label]), counts[label],
                    counts[label] / len(by_class[label]), len(selected) / (5 * counts[label]))
                    for label in range(1, 6))
                man = cli.kernel.TrainingManifest(
                    f, b, selected, stats, len(pool), len(selected), len(selected) / len(pool))
                self.manifests[f, b] = man
                heads.append({
                    "held_out_fold": f, "budget_percent": b, "pool_count": len(pool),
                    "selected_count": len(selected), "selected_fraction": len(selected) / len(pool),
                    "class_stats": [asdict(stat) for stat in stats],
                    "training_tile_ids_sha256": cli.manifests._ordered_ids_sha(selected),
                })
                training.extend((f, b, i, r.tile_id, r.slide_id, r.patient_id, r.tissue_label,
                                 r.h_optimus_1_feature_row) for i, r in enumerate(selected))
        self.cpu_run = "toy_cpu_run"
        self.source_paths["training_rows"].write_bytes(csv_bytes(
            cli.manifests.TRAINING_HEADER, training, True))
        self.source_paths["head_plan"].write_bytes(jbytes(dict(
            schema_version=1, run_id=self.cpu_run, task=cli.manifests.TASK,
            classes=[1, 2, 3, 4, 5], heads=heads)))
        man = self.manifests[fold, budget]
        self.held = tuple(r for r in self.records if self.mapping[r.patient_id] == fold)
        assert self.held
        expected = {
            "annotation_tiles": len(self.records), "annotation_slides": len(self.feature_entries),
            "annotation_patients": len(annotation_patients), "union_patients": len(patients),
            "feature_files": len(self.feature_entries), "training_tiles": len(man.tiles),
            "training_slides": len({r.slide_id for r in man.tiles}),
            "training_patients": len({r.patient_id for r in man.tiles}),
            "held_out_tiles": len(self.held), "held_out_slides": len({r.slide_id for r in self.held}),
            "held_out_patients": len({r.patient_id for r in self.held}),
        }
        self.source_paths["cpu_summary"].write_bytes(jbytes(dict(
            schema_version=1, run_id=self.cpu_run, task=cli.manifests.TASK,
            counts={name: expected[name] for name in ("annotation_tiles", "annotation_slides",
                                                       "annotation_patients")},
            policy=cli.manifests.POLICY)))
        self.spec = dict(
            schema_version=1, run_id="toy_head", task=cli.TASK, sources={},
            code_files={role: self.bind(path) for role, path in cli.code_paths().items()},
            features=self.feature_entries,
            parameters=dict(held_out_fold=fold, budget_percent=budget, prediction_batch_rows=4096),
            expected=expected)
        primitive_files = {
            "prosper/evaluation/r2_1_hoptimus_fit.py": self.spec["code_files"]["fit"]["sha256"],
            "prosper/evaluation/r2_1_hoptimus_asset_io.py": self.spec["code_files"]["asset_io"]["sha256"],
        }
        for role, kind in (("asset_fit_review", "implementation"),
                           ("primitive_integration_review", "synthetic_integration")):
            self.source_paths[role].write_bytes(jbytes(dict(
                schema_version=1, disposition="accepted", review_kind=kind,
                contract_sha256=cli.PRIMITIVE_CONTRACT, source_files=primitive_files)))
        self.refresh()

    @staticmethod
    def bind(path):
        data = Path(path).read_bytes()
        return dict(path=str(path), sha256=sha(data), size=len(data))

    def refresh(self):
        old_digest = sha(b"toy historical CPU declaration")
        entries = [dict(kind="spec", name="spec", sha256=old_digest, size=30)]
        inventory = self.bind(self.source_paths["feature_inventory"])
        entries.append(dict(kind="source", name="stage1a_feature_inventory", **inventory))
        for role, old in (("filter_kernel", "kernel"), ("filter_cli", "runner"),
                          ("cpu_io", "io"), ("legacy_io", "legacy_io"),
                          ("package_init", "package_init"), ("evaluation_init", "evaluation_init")):
            entries.append(dict(kind="code", name=old, **self.spec["code_files"][role]))
        self.source_paths["cpu_input_bindings"].write_bytes(jbytes(dict(
            schema_version=1, run_id=self.cpu_run, task=cli.manifests.TASK,
            spec_sha256=old_digest, bindings=entries)))
        outputs = {leaf: {key: self.bind(self.source_paths[role])[key] for key in ("sha256", "size")}
                   for role, leaf in cli.CONSUMED.items()}
        self.source_paths["cpu_complete"].write_bytes(jbytes(dict(
            schema_version=1, run_id=self.cpu_run, task=cli.manifests.TASK,
            scope="hoptimus_filter_manifests_only", spec_sha256=old_digest, outputs=outputs)))
        self.source_paths["cpu_review"].write_bytes(jbytes(dict(
            schema_version=1, disposition="accepted", branches={"filter": {"disposition": "accepted"}},
            contract_sha256=cli.assets._CPU_CONTRACT,
            artifacts={"filter_attempt_001/COMPLETE.json":
                       sha(self.source_paths["cpu_complete"].read_bytes())})))
        self.rebind()

    def rebind(self):
        self.spec["sources"].update({role: self.bind(path) for role, path in self.source_paths.items()})
        self.spec_path.write_bytes(jbytes(self.spec))
        self.spec_digest = sha(self.spec_path.read_bytes())

    def change_json(self, role, edit, *, refresh=True):
        path = self.source_paths[role]
        obj = json.loads(path.read_bytes())
        edit(obj)
        path.write_bytes(jbytes(obj))
        self.refresh() if refresh else self.rebind()

    def change_csv(self, role, header, edit, *, compressed=True):
        path = self.source_paths[role]
        data = gzip.decompress(path.read_bytes()) if compressed else path.read_bytes()
        rows = list(csv.reader(io.StringIO(data.decode(), newline="")))[1:]
        edit(rows)
        path.write_bytes(csv_bytes(header, rows, compressed))
        self.refresh()

    def run(self):
        return cli.run(self.spec_path, self.spec_digest, self.out, self.root)

    def command(self):
        return [sys.executable, "-m", "prosper.evaluation.r2_1_hoptimus_head_cli",
                "--spec", str(self.spec_path), "--spec-sha256", self.spec_digest,
                "--out", str(self.out), "--out-root", str(self.root)]


@pytest.fixture
def toy(tmp_path, monkeypatch):
    return Toy(tmp_path, monkeypatch)


def outputs(toy):
    marker = json.loads((toy.out / "COMPLETE.json").read_bytes())
    assert set(marker["outputs"]) == cli.ARTIFACTS
    assert set(p.name for p in toy.out.iterdir()) == cli.ARTIFACTS | {"COMPLETE.json"}
    for name, binding in marker["outputs"].items():
        data = (toy.out / name).read_bytes()
        assert binding == {"size": len(data), "sha256": sha(data)}
    p = np.load(toy.out / "probabilities.npy", allow_pickle=False)
    y = np.load(toy.out / "predictions.npy", allow_pickle=False)
    truth = np.array([r.tissue_label for r in toy.held], np.int64)
    assert p.dtype.str == "<f4" and y.dtype.str == "<i8"
    assert np.array_equal(y, np.array([1, 2, 3, 4, 5])[p.argmax(axis=1)])
    counts = json.loads((toy.out / "counts.json").read_bytes())
    confusion = np.zeros((5, 5), np.int64)
    for a, b in zip(truth, y):
        confusion[a - 1, b - 1] += 1
    assert counts["tissue_confusion"] == confusion.tolist()
    assert counts["tumour_counts"] == cli.fitter.tumour_counts(truth, y)
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(
        (toy.out / "held_out_rows.csv.gz").read_bytes()).decode(), newline="")))
    assert [r["tile_id"] for r in rows] == [r.tile_id for r in toy.held]
    assert [int(r["tissue_label"]) for r in rows] == truth.tolist()
    assert sum(r["tiles"] for r in counts["held_out_support"]) == len(truth)
    diagnostics = json.loads((toy.out / "fit_diagnostics.json").read_bytes())
    restored = cli.legacy._pickle((toy.out / "head.pkl").read_bytes(), "toy readback")
    assert type(restored) is cli.fitter.FittedHead and restored.diagnostics == diagnostics
    held_features = np.stack([toy.raw[r.slide_id][0][r.h_optimus_1_feature_row] for r in toy.held])
    batch = cli.fitter.predict_head(restored, held_features)
    assert batch.probabilities.tobytes() == p.tobytes() and batch.labels.tobytes() == y.tobytes()
    summary = json.loads((toy.out / "summary.json").read_bytes())
    assert summary["counts"] == toy.spec["expected"]
    assert summary["details"]["scientific_assessment"] == "pending"
    assert summary["limitations"] == cli.LIMITATIONS
    assert summary["details"]["held_out_tile_ids_sha256"] == \
        cli.manifests._ordered_ids_sha(toy.held)
    bindings = json.loads((toy.out / "input_bindings.json").read_bytes())
    assert bindings["environment"]["native_environment"] == {name: "1" for name in cli.NATIVE_VARIABLES}
    assert set(bindings["executing_code_paths"]) == set(cli.code_paths())
    assert len(bindings["feature_consumption"]) == len(toy.feature_entries)
    assert all(len(row["sha256"]) == 64 for row in bindings["bindings"])
    assert not (toy.out / "FAILED").exists()
    return counts, summary


def test_normal_package_help():
    result = subprocess.run(
        [sys.executable, "-m", "prosper.evaluation.r2_1_hoptimus_head_cli", "--help"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "--spec-sha256" in result.stdout and "--out-root" in result.stdout


def test_head1_normal_cli_misclassification_is_success_and_remains_in_counts(tmp_path, monkeypatch):
    toy = Toy(tmp_path, monkeypatch, zero=True)
    result = subprocess.run(toy.command(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(line.startswith("[Parallel(n_jobs=1)]:") for line in result.stderr.splitlines())
    counts, summary = outputs(toy)
    confusion = np.array(counts["tissue_confusion"])
    assert confusion.sum() > np.trace(confusion), "toy must deliberately include misclassification"
    assert counts["tumour_counts"]["fn"] + counts["tumour_counts"]["fp"] > 0
    assert summary["details"]["features_hash_only"] > 0


def test_api_fit_once_exact_source_order_and_serialized_head(toy, monkeypatch):
    original = cli.fitter.fit_head
    calls = []

    def fit(features, manifest, mapping):
        calls.append(manifest)
        expected = np.stack([toy.raw[r.slide_id][0][r.h_optimus_1_feature_row] for r in manifest.tiles])
        assert features.tobytes() == expected.tobytes()
        assert features.flags.c_contiguous and features.dtype == np.float32
        assert manifest.tiles == toy.manifests[0, 5].tiles
        return original(features, manifest, mapping)

    monkeypatch.setattr(cli.fitter, "fit_head", fit)
    marker = toy.run()
    assert marker == json.loads((toy.out / "COMPLETE.json").read_bytes())
    assert len(calls) == 1
    outputs(toy)


def test_full_budget_second_fold_and_missing_held_out_classes(tmp_path, monkeypatch):
    toy = Toy(tmp_path, monkeypatch, fold=2, budget=100)
    toy.run()
    counts, _ = outputs(toy)
    assert counts["held_out_support"][1] == {"tissue_label": 2, "tiles": 0, "slides": 0, "patients": 0}


def test_retained_convergence_warning_has_valid_completed_head(toy, monkeypatch):
    original = cli.fitter.SGDClassifier.fit
    calls = []

    def fit(self, *args, **kwargs):
        calls.append(self)
        result = original(self, *args, **kwargs)
        warnings.warn("toy prescribed convergence warning", ConvergenceWarning)
        return result

    monkeypatch.setattr(cli.fitter.SGDClassifier, "fit", fit)
    toy.run()
    _, summary = outputs(toy)
    assert len(calls) == 1
    assert summary["details"]["convergence_status"] == "iteration_limit_warning_retained"
    assert any(r["message"] == "toy prescribed convergence warning"
               for r in summary["details"]["fit_warnings"])


@pytest.mark.parametrize("section,key,value", [
    ("parameters", "held_out_fold", True), ("parameters", "held_out_fold", 0.0),
    ("parameters", "budget_percent", 50), ("parameters", "prediction_batch_rows", 1),
    ("expected", "annotation_tiles", True), ("expected", "training_tiles", 0),
    ("expected", "held_out_patients", 999),
])
def test_invalid_declarations_fail_before_fit(toy, monkeypatch, section, key, value):
    toy.spec[section][key] = value
    toy.rebind()
    monkeypatch.setattr(cli.fitter, "fit_head", lambda *a: pytest.fail("invalid declaration fitted"))
    with pytest.raises(ValueError):
        toy.run()
    assert not toy.out.exists()


@pytest.mark.parametrize("edit", [
    lambda s: s.update(schema_version=True),
    lambda s: s.update(task="hoptimus_filter_manifest"),
    lambda s: s["sources"].update(extra=s["sources"]["cpu_review"]),
    lambda s: s["code_files"].pop("fit"),
    lambda s: s["features"].append(s["features"][0]),
    lambda s: s["parameters"].update(alpha=0.1),
    lambda s: s["features"][0].update(dtype="float16"),
])
def test_wrong_roles_profiles_and_unknown_keys(toy, edit):
    edit(toy.spec)
    toy.rebind()
    with pytest.raises((ValueError, KeyError)):
        toy.run()
    assert not toy.out.exists()


@pytest.mark.parametrize("role,edit", [
    ("cpu_review", lambda r: r.update(disposition="accepted_with_dependencies")),
    ("cpu_review", lambda r: r.update(contract_sha256="0" * 64)),
    ("cpu_complete", lambda r: r.update(scope=cli.SCOPE)),
    ("cpu_input_bindings", lambda r: r.update(schema_version=1.0)),
    ("asset_fit_review", lambda r: r.update(review_kind="exact_contract")),
    ("primitive_integration_review", lambda r: r.update(contract_sha256="0" * 64)),
    ("asset_fit_review", lambda r: r["source_files"].update(
        {"prosper/evaluation/r2_1_hoptimus_fit.py": "0" * 64})),
])
def test_rehashed_receipt_scope_and_code_negatives(toy, role, edit):
    toy.change_json(role, edit, refresh=False)
    with pytest.raises(ValueError):
        toy.run()
    assert not toy.out.exists()


def test_inventory_binding_role_mismatch_beyond_outer_review_hash(toy):
    obj = json.loads(toy.source_paths["cpu_input_bindings"].read_bytes())
    next(r for r in obj["bindings"] if r["kind"] == "source")["name"] = "other_inventory"
    toy.source_paths["cpu_input_bindings"].write_bytes(jbytes(obj))
    complete = json.loads(toy.source_paths["cpu_complete"].read_bytes())
    data = toy.source_paths["cpu_input_bindings"].read_bytes()
    complete["outputs"]["input_bindings.json"] = {"size": len(data), "sha256": sha(data)}
    toy.source_paths["cpu_complete"].write_bytes(jbytes(complete))
    review = json.loads(toy.source_paths["cpu_review"].read_bytes())
    review["artifacts"]["filter_attempt_001/COMPLETE.json"] = sha(
        toy.source_paths["cpu_complete"].read_bytes())
    toy.source_paths["cpu_review"].write_bytes(jbytes(review))
    toy.rebind()
    with pytest.raises(ValueError, match="inventory"):
        toy.run()


def test_code_path_alias_cannot_substitute_arbitrary_file(toy):
    alternate = toy.work / "copied_runner.py"
    alternate.write_bytes(cli.code_paths()["runner"].read_bytes())
    toy.spec["code_files"]["runner"] = toy.bind(alternate)
    toy.rebind()
    with pytest.raises(ValueError, match="actual executing"):
        toy.run()


@pytest.mark.parametrize("role,header,column,value", [
    ("annotation_assignments", cli.manifests.ANNOTATION_HEADER, "h_optimus_1_feature_row", "999"),
    ("annotation_assignments", cli.manifests.ANNOTATION_HEADER, "held_out_fold", "4"),
    ("annotation_assignments", cli.manifests.ANNOTATION_HEADER, "level", "00"),
    ("annotation_assignments", cli.manifests.ANNOTATION_HEADER, "patient_id", "wrong"),
    ("training_rows", cli.manifests.TRAINING_HEADER, "training_position", "1"),
    ("training_rows", cli.manifests.TRAINING_HEADER, "tissue_label", "5"),
    ("training_rows", cli.manifests.TRAINING_HEADER, "h_optimus_1_feature_row", "999"),
])
def test_rehashed_manifest_identity_and_numeric_negatives(toy, role, header, column, value):
    toy.change_csv(role, header, lambda rows: rows[0].__setitem__(header.index(column), value))
    with pytest.raises(ValueError):
        toy.run()
    assert not toy.out.exists()


def test_annotation_duplicate_and_row_limit(toy):
    toy.change_csv("annotation_assignments", cli.manifests.ANNOTATION_HEADER,
                   lambda rows: rows.append(rows[0]))
    with pytest.raises(ValueError):
        toy.run()


def test_duplicate_patient_kernel_order_and_repeat_fold_values_are_distinct(toy):
    assert len(set(toy.mapping.values())) < len(toy.mapping)
    toy.change_csv("patient_folds", cli.manifests.FOLD_HEADER,
                   lambda rows: rows[1].__setitem__(2, rows[0][2]), compressed=False)
    with pytest.raises(ValueError, match="patient/order"):
        toy.run()


def test_reappearing_training_group_is_not_reused_or_merged(toy):
    def reappear(rows):
        index = next(i for i, row in enumerate(rows) if row[:2] == ["1", "5"])
        rows.insert(index, rows[0])
        rows.pop()

    toy.change_csv("training_rows", cli.manifests.TRAINING_HEADER,
                   reappear)
    with pytest.raises(ValueError, match="reappeared"):
        toy.run()


@pytest.mark.parametrize("edit", [
    lambda p: p["heads"][0]["class_stats"][0].update(balanced_weight=2.0),
    lambda p: p["heads"][0].update(training_tile_ids_sha256="0" * 64),
    lambda p: p["heads"][0].update(selected_count=True),
    lambda p: p["heads"].__setitem__(1, p["heads"][0]),
])
def test_plan_count_weight_digest_and_duplicate_head_negatives(toy, edit):
    toy.change_json("head_plan", edit)
    with pytest.raises(ValueError):
        toy.run()


def test_same_patient_training_leakage_not_just_slide_holdout(toy):
    header = cli.manifests.TRAINING_HEADER
    leaking = toy.held[0]
    replacement = [0, 5, 0, leaking.tile_id, leaking.slide_id, leaking.patient_id,
                   leaking.tissue_label, leaking.h_optimus_1_feature_row]
    toy.change_csv("training_rows", header, lambda rows: rows.__setitem__(0, replacement))
    with pytest.raises(ValueError, match="same-patient"):
        toy.run()


@pytest.mark.parametrize("transform", [
    lambda b: b[:-1], lambda b: b + b"trailing", lambda b: b + gzip.compress(b"extra", mtime=0),
])
def test_rehashed_truncated_trailing_and_multimember_gzip(toy, transform):
    path = toy.source_paths["training_rows"]
    path.write_bytes(transform(path.read_bytes()))
    toy.refresh()
    with pytest.raises(ValueError):
        toy.run()


def test_streamed_gzip_expansion_record_and_row_limits():
    header = ("key", "value")
    data = csv_bytes(header, [("a", "b")] * 1000, True)
    with pytest.raises(ValueError, match="expanded"):
        list(cli.csv_rows(data, header, "toy", compressed=True, expanded_limit=100,
                          row_limit=1000))
    with pytest.raises(ValueError, match="row limit"):
        list(cli.csv_rows(data, header, "toy", compressed=True, expanded_limit=10000,
                          row_limit=1))
    oversized = csv_bytes(header, [("a", "b" * 8192)], True)
    with pytest.raises(ValueError, match="per-record"):
        list(cli.csv_rows(oversized, header, "toy", compressed=True,
                          expanded_limit=10000, row_limit=1))
    multiline = csv_bytes(header, [("a", "b\nc")], True)
    assert list(cli.csv_rows(multiline, header, "toy", compressed=True,
                             expanded_limit=10000, row_limit=1)) == [{"key": "a", "value": "b\nc"}]
    quoted_blank = csv_bytes(header, [("a", "b\n\nc")], True)
    assert list(cli.csv_rows(quoted_blank, header, "toy", compressed=True,
                             expanded_limit=10000, row_limit=1)) == [{"key": "a", "value": "b\n\nc"}]
    blank_row = gzip.compress(gzip.decompress(multiline) + b"\n", mtime=0)
    with pytest.raises(ValueError, match="blank CSV record"):
        list(cli.csv_rows(blank_row, header, "toy", compressed=True,
                          expanded_limit=10000, row_limit=1))


@pytest.mark.parametrize("change", [
    lambda raw, tiles: (raw.astype(np.float64), tiles),
    lambda raw, tiles: (raw[:, :-1], tiles),
    lambda raw, tiles: (raw, tiles[:-1]),
    lambda raw, tiles: (raw, [{"wsi_loc": [False, 512]}] + tiles[1:]),
    lambda raw, tiles: (raw, [{"wsi_loc": [0.0, 512.0]}] + tiles[1:]),
    lambda raw, tiles: (raw, [{"wsi_loc": [-1, 512]}] + tiles[1:]),
    lambda raw, tiles: (np.full_like(raw, np.nan), tiles),
])
def test_feature_dtype_coordinates_rows_and_finiteness_after_ownership(toy, change):
    slide = toy.held[0].slide_id
    entry = next(r for r in toy.feature_entries if r["slide_id"] == slide)
    raw, coords = toy.raw[slide]
    payload = pickle.dumps(change(raw, [{"wsi_loc": list(xy)} for xy in coords]), protocol=4)
    Path(entry["path"]).write_bytes(payload)
    entry.update(size=len(payload), sha256=sha(payload))
    header = cli.manifests.FEATURE_HEADER

    def inventory(rows):
        row = next(r for r in rows if r[1] == slide)
        row[header.index("size")], row[header.index("sha256")] = str(len(payload)), sha(payload)

    toy.change_csv("feature_inventory", header, inventory, compressed=False)
    with pytest.raises(ValueError):
        toy.run()
    assert (toy.out / "FAILED").exists() and not (toy.out / "COMPLETE.json").exists()


def test_roundtrip_invalid_model_state_is_fatal_without_refit(toy, monkeypatch):
    original_dump, original_fit = cli.pickle.dumps, cli.fitter.fit_head
    calls = []

    def fit(*args):
        calls.append(1)
        return original_fit(*args)

    def dump(head, **kwargs):
        head.estimator.coef_[0, 0] += np.float32(1)
        return original_dump(head, **kwargs)

    monkeypatch.setattr(cli.fitter, "fit_head", fit)
    monkeypatch.setattr(cli.pickle, "dumps", dump)
    with pytest.raises(ValueError, match="fingerprint"):
        toy.run()
    assert calls == [1] and (toy.out / "head.pkl").exists()
    assert (toy.out / "FAILED").exists() and not (toy.out / "COMPLETE.json").exists()


def test_same_batch_restored_probability_difference_fails_exactly(toy, monkeypatch):
    original_fit, original_predict = cli.fitter.fit_head, cli.fitter.predict_head
    original_head = []

    def fit(*args):
        head = original_fit(*args)
        original_head.append(head)
        return head

    def predict(head, features):
        batch = original_predict(head, features)
        if head is not original_head[0] and len(features):
            batch.probabilities[0, 0] = np.nextafter(batch.probabilities[0, 0], np.float32(1))
        return batch

    monkeypatch.setattr(cli.fitter, "fit_head", fit)
    monkeypatch.setattr(cli.fitter, "predict_head", predict)
    with pytest.raises(ValueError, match="byte-exact"):
        toy.run()
    assert len(original_head) == 1 and (toy.out / "FAILED").exists()


@pytest.mark.parametrize("array_kind", ["wrong_values", "trailing", "wrong_dtype"])
def test_npy_readback_rejects_wrong_or_extra_serialized_data(toy, monkeypatch, array_kind):
    original = cli.np.lib.format.write_array

    def write(handle, array, **kwargs):
        if array.dtype == np.float32:
            if array_kind == "wrong_values":
                array = array.copy()
                array[0, 0] = np.nextafter(array[0, 0], np.float32(1))
            elif array_kind == "wrong_dtype":
                array = array.astype(np.float64)
            original(handle, array, **kwargs)
            if array_kind == "trailing":
                handle.write(b"extra")
        else:
            original(handle, array, **kwargs)

    monkeypatch.setattr(cli.np.lib.format, "write_array", write)
    with pytest.raises(ValueError):
        toy.run()
    assert (toy.out / "FAILED").exists() and not (toy.out / "COMPLETE.json").exists()


def test_late_source_mutation_is_not_accepted_after_output_readback(toy, monkeypatch):
    original = cli._model_output

    def model(*args):
        original(*args)
        with toy.source_paths["feature_inventory"].open("ab") as handle:
            handle.write(b"\n")

    monkeypatch.setattr(cli, "_model_output", model)
    with pytest.raises(ValueError):
        toy.run()
    assert (toy.out / "summary.json").exists() and (toy.out / "FAILED").exists()
    assert not (toy.out / "COMPLETE.json").exists()


def test_output_mutation_during_final_source_sweep(toy, monkeypatch):
    original = cli.assets.AssetGuard.final_bindings

    def final(guard):
        with (toy.out / "counts.json").open("ab") as handle:
            handle.write(b"\n")
        return original(guard)

    monkeypatch.setattr(cli.assets.AssetGuard, "final_bindings", final)
    with pytest.raises(ValueError):
        toy.run()
    assert (toy.out / "FAILED").exists() and not (toy.out / "COMPLETE.json").exists()


@pytest.mark.parametrize("exception", [KeyboardInterrupt("toy abort"), SystemExit(17)])
def test_post_marker_baseexception_removes_only_owned_marker_closes_fds(toy, monkeypatch, exception):
    original = cli._Output.check
    before = set(os.listdir("/proc/self/fd"))

    def check(output, bindings, *, marker=False):
        original(output, bindings, marker=marker)
        if marker:
            raise exception

    monkeypatch.setattr(cli._Output, "check", check)
    with pytest.raises(type(exception)) as caught:
        toy.run()
    assert caught.value is exception
    assert not (toy.out / "COMPLETE.json").exists()
    assert set(os.listdir("/proc/self/fd")) == before
    assert json.loads((toy.out / "FAILED").read_bytes())["error_type"] == type(exception).__name__


@pytest.mark.parametrize("exception", [KeyboardInterrupt("fdopen abort"), SystemExit(23)])
def test_file_fdopen_handoff_baseexception_closes_fd_and_preserves_failure(toy, monkeypatch, exception):
    original = cli.os.fdopen
    failed = []
    before = set(os.listdir("/proc/self/fd"))

    def fdopen(fd, mode, *args, **kwargs):
        if mode == "wb" and not failed:
            failed.append(fd)
            raise exception
        return original(fd, mode, *args, **kwargs)

    monkeypatch.setattr(cli.os, "fdopen", fdopen)
    with pytest.raises(type(exception)) as caught:
        toy.run()
    assert caught.value is exception and set(os.listdir("/proc/self/fd")) == before
    assert (toy.out / ".head.pkl.part").exists() and (toy.out / "FAILED").exists()


def test_directory_acquisition_abort_before_ownership_does_not_mark_foreign_success(toy, monkeypatch):
    original_open, original_stat = cli.os.open, cli.os.fstat
    claimed = []
    error = KeyboardInterrupt("directory handoff abort")
    before = set(os.listdir("/proc/self/fd"))

    def open_(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        if path == toy.out.name and flags & os.O_DIRECTORY:
            claimed.append(fd)
        return fd

    def fstat(fd):
        if claimed and fd == claimed[0]:
            raise error
        return original_stat(fd)

    monkeypatch.setattr(cli.os, "open", open_)
    monkeypatch.setattr(cli.os, "fstat", fstat)
    with pytest.raises(KeyboardInterrupt) as caught:
        toy.run()
    assert caught.value is error and set(os.listdir("/proc/self/fd")) == before
    assert toy.out.exists() and list(toy.out.iterdir()) == []


def test_existing_foreign_directory_is_untouched(toy):
    toy.out.mkdir()
    marker = toy.out / "COMPLETE.json"
    marker.write_bytes(b"foreign")
    with pytest.raises(FileExistsError):
        toy.run()
    assert marker.read_bytes() == b"foreign" and not (toy.out / "FAILED").exists()


def test_environment_root_and_bank_overlap_rejected(toy, monkeypatch):
    monkeypatch.setenv("PROSPER_OUT", str(toy.bank))
    with pytest.raises(ValueError, match="out-root"):
        toy.run()
    monkeypatch.setenv("PROSPER_OUT", str(toy.work))
    with pytest.raises(ValueError, match="bank"):
        cli.run(toy.spec_path, toy.spec_digest, toy.bank / "attempt", toy.work)


def test_spec_symlink_keeps_lexical_relative_source_origin(toy):
    for role, entry in toy.spec["sources"].items():
        entry["path"] = "controls/" + toy.source_paths[role].name
    toy.spec_path.write_bytes(jbytes(toy.spec))
    toy.spec_digest = sha(toy.spec_path.read_bytes())
    lexical = toy.work / "lexical"
    lexical.mkdir()
    (lexical / "controls").symlink_to(toy.controls, target_is_directory=True)
    linked = lexical / "spec.json"
    linked.symlink_to(toy.spec_path)
    cli.run(linked, toy.spec_digest, toy.out, toy.root)
    bindings = json.loads((toy.out / "input_bindings.json").read_bytes())
    source = next(row for row in bindings["bindings"] if row["asset_id"] == "source:patient_folds")
    assert source["path"] == str(lexical / "controls" / toy.source_paths["patient_folds"].name)
    assert source["resolved_path"] == str(toy.source_paths["patient_folds"].resolve())
    outputs(toy)


def test_unexpected_fit_warning_is_failed_once_with_available_warning_records(toy, monkeypatch):
    original = cli.fitter.SGDClassifier.fit
    calls = []

    def fit(self, *args, **kwargs):
        calls.append(1)
        result = original(self, *args, **kwargs)
        warnings.warn("toy unexpected fit warning", RuntimeWarning)
        return result

    monkeypatch.setattr(cli.fitter.SGDClassifier, "fit", fit)
    with pytest.raises(ValueError, match="unexpected warnings"):
        toy.run()
    assert calls == [1] and not (toy.out / "COMPLETE.json").exists()
    failure = json.loads((toy.out / "FAILED").read_bytes())
    assert any(row["message"] == "toy unexpected fit warning" for row in failure["fit_warnings"])


def test_prediction_batches_are_4096_and_preserve_every_position(monkeypatch):
    class Positions:
        def __len__(self):
            return 4099

        def __getitem__(self, item):
            return item.start, min(item.stop, len(self))

    calls = []

    def predict(head, bounds):
        start, stop = bounds
        calls.append(bounds)
        p = np.full((stop - start, 5), 0.2, np.float32)
        y = np.full(stop - start, 1 if start == 0 else 2, np.int64)
        return cli.fitter.HeadPredictions(p, y)

    monkeypatch.setattr(cli.fitter, "predict_head", predict)
    probabilities, labels = cli._predict(None, Positions())
    assert calls == [(0, 4096), (4096, 4099)]
    assert probabilities.shape == (4099, 5) and probabilities.dtype == np.float32
    assert labels.dtype == np.int64 and labels[:4096].tolist() == [1] * 4096
    assert labels[4096:].tolist() == [2] * 3


def test_hash_only_feature_late_mutation_is_fatal_without_decoding_it(toy, monkeypatch):
    needed = {row.slide_id for row in toy.manifests[0, 5].tiles + toy.held}
    unused = next(row for row in toy.feature_entries if row["slide_id"] not in needed)
    original = cli._model_output

    def model(*args):
        original(*args)
        with Path(unused["path"]).open("ab") as handle:
            handle.write(b"late")

    monkeypatch.setattr(cli, "_model_output", model)
    with pytest.raises(ValueError):
        toy.run()
    assert (toy.out / "FAILED").exists() and not (toy.out / "COMPLETE.json").exists()


def test_output_directory_substitution_preserves_foreign_files(toy, monkeypatch):
    original = cli.assets.AssetGuard.final_bindings
    owned = toy.root / "owned_moved"

    def final(guard):
        bindings = original(guard)
        toy.out.rename(owned)
        toy.out.mkdir()
        (toy.out / "COMPLETE.json").write_bytes(b"foreign")
        return bindings

    monkeypatch.setattr(cli.assets.AssetGuard, "final_bindings", final)
    with pytest.raises(ValueError):
        toy.run()
    assert (toy.out / "COMPLETE.json").read_bytes() == b"foreign"
    assert not (toy.out / "FAILED").exists()
    assert (owned / "FAILED").exists() and not (owned / "COMPLETE.json").exists()


def test_foreign_post_marker_replacement_is_not_unlinked(toy, monkeypatch):
    original = cli._Output.check

    def check(output, bindings, *, marker=False):
        if marker:
            foreign = toy.root / "foreign-marker"
            foreign.write_bytes(b"foreign")
            foreign.replace(toy.out / "COMPLETE.json")
        original(output, bindings, marker=marker)

    monkeypatch.setattr(cli._Output, "check", check)
    with pytest.raises(ValueError):
        toy.run()
    assert (toy.out / "COMPLETE.json").read_bytes() == b"foreign"
    failure = json.loads((toy.out / "FAILED").read_bytes())
    assert any("foreign COMPLETE" in note for note in failure["notes"])


def test_failure_record_error_is_attached_without_masking_baseexception(toy, monkeypatch):
    error = KeyboardInterrupt("preserve primary")
    original = cli._Output.check

    def check(output, bindings, *, marker=False):
        original(output, bindings, marker=marker)
        if marker:
            raise error

    def fail(*args):
        raise OSError("synthetic failure recorder error")

    monkeypatch.setattr(cli._Output, "check", check)
    monkeypatch.setattr(cli._Output, "fail", fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        toy.run()
    assert caught.value is error
    assert any("synthetic failure recorder error" in note for note in error.__notes__)


def test_no_manifest_generator_is_called(toy, monkeypatch):
    monkeypatch.setattr(cli.manifests, "execute", lambda *args: pytest.fail("manifest generator"))
    monkeypatch.setattr(cli.kernel, "build_patient_folds", lambda *args: pytest.fail("fold generator"))
    monkeypatch.setattr(cli.kernel, "build_training_manifests", lambda *args: pytest.fail("sampling"))
    toy.run()
    outputs(toy)


def test_example_is_explicitly_unbound_and_has_exact_public_roles():
    path = Path(__file__).resolve().parents[1] / "examples" / "r2_1_hoptimus_head.json"
    example = json.loads(path.read_bytes())
    assert example["run_id"].startswith("EXAMPLE_UNBOUND")
    assert set(example["sources"]) == set(cli.SOURCE_LIMITS)
    assert set(example["code_files"]) == set(cli.code_paths())
    assert example["parameters"] == {
        "held_out_fold": 0, "budget_percent": 5, "prediction_batch_rows": 4096}
    assert all(binding["size"] == 0 for binding in example["sources"].values())


def _older_estimator_version(data):
    key = data.index(b"_sklearn_version")
    start = data.index(b"1.7.0", key)
    changed = data[:start] + b"1.6.0" + data[start + 5:]
    assert len(changed) == len(data) and changed != data
    return changed


def test_hcli1_same_size_inode_model_change_fails_before_loader(toy, monkeypatch):
    original_write, original_decode = cli._Output.write, cli.legacy._pickle
    original_fit = cli.fitter.fit_head
    calls = []

    def fit(*args):
        calls.append(1)
        return original_fit(*args)

    def write(output, name, emit):
        original_write(output, name, emit)
        if name == "head.pkl":
            path = toy.out / name
            before = path.stat()
            changed = _older_estimator_version(path.read_bytes())
            with path.open("r+b") as handle:
                handle.write(changed)
            after = path.stat()
            assert (before.st_dev, before.st_ino, before.st_size) == \
                (after.st_dev, after.st_ino, after.st_size)

    def decode(data, context):
        if context == "just-written head":
            pytest.fail("changed serialized bytes reached the loader")
        return original_decode(data, context)

    monkeypatch.setattr(cli._Output, "write", write)
    monkeypatch.setattr(cli.legacy, "_pickle", decode)
    monkeypatch.setattr(cli.fitter, "fit_head", fit)
    with pytest.raises(ValueError, match="original serialized size/hash"):
        toy.run()
    assert calls == [1] and not (toy.out / "COMPLETE.json").exists()
    assert (toy.out / "FAILED").exists()


def test_hcli1_loader_version_warning_fails_even_under_outer_ignore(toy, monkeypatch):
    original = cli.pickle.dumps
    calls = []

    def dumps(head, **kwargs):
        calls.append(1)
        return _older_estimator_version(original(head, **kwargs))

    monkeypatch.setattr(cli.pickle, "dumps", dumps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="model-loader warnings") as caught:
            toy.run()
    assert calls == [1] and not (toy.out / "COMPLETE.json").exists()
    notes = "\n".join(caught.value.__notes__)
    assert "InconsistentVersionWarning" in notes and "not fit warnings" in notes
    failure = json.loads((toy.out / "FAILED").read_bytes())
    assert failure["fit_warnings"] == []
    assert any("InconsistentVersionWarning" in note for note in failure["notes"])


def test_hcli1_loader_warnings_are_distinct_from_retained_fit_warning(toy, monkeypatch):
    original_fit, original_decode = cli.fitter.SGDClassifier.fit, cli.legacy._pickle

    def fit(self, *args, **kwargs):
        result = original_fit(self, *args, **kwargs)
        warnings.warn("retained fit-only warning", ConvergenceWarning)
        return result

    def decode(data, context):
        result = original_decode(data, context)
        if context == "just-written head":
            warnings.warn("fatal loader-only warning", RuntimeWarning)
        return result

    monkeypatch.setattr(cli.fitter.SGDClassifier, "fit", fit)
    monkeypatch.setattr(cli.legacy, "_pickle", decode)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="model-loader warnings"):
            toy.run()
    failure = json.loads((toy.out / "FAILED").read_bytes())
    assert any(row["message"] == "retained fit-only warning" for row in failure["fit_warnings"])
    assert all(row["message"] != "fatal loader-only warning" for row in failure["fit_warnings"])
    assert any("fatal loader-only warning" in note for note in failure["notes"])
    assert not (toy.out / "COMPLETE.json").exists()


@pytest.mark.parametrize("error", [RuntimeError("loader failed"),
                                  KeyboardInterrupt("loader abort"), SystemExit(31)])
def test_hcli1_warn_then_loader_error_preserves_same_exception(toy, monkeypatch, error):
    original = cli.legacy._pickle

    def decode(data, context):
        if context == "just-written head":
            warnings.warn("loader warning before failure", UserWarning)
            raise error
        return original(data, context)

    monkeypatch.setattr(cli.legacy, "_pickle", decode)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(type(error)) as caught:
            toy.run()
    assert caught.value is error
    assert any("loader warning before failure" in note for note in error.__notes__)
    failure = json.loads((toy.out / "FAILED").read_bytes())
    assert failure["fit_warnings"] == [] and failure["error_type"] == type(error).__name__
    assert not (toy.out / "COMPLETE.json").exists()


@pytest.mark.parametrize("error", [KeyboardInterrupt("released then interrupted"),
                                  OSError("released then error")])
def test_hcli2_post_release_failure_recovers_owned_directory(toy, monkeypatch, error):
    original_close, original_os_close = cli._Output.close, cli.os.close
    outputs_seen, released = [], []

    def close(output, primary=None):
        if primary is None and output.fd is not None:
            outputs_seen.append(output)
        return original_close(output, primary)

    def os_close(fd):
        if outputs_seen and not released:
            released.append(fd)
            original_os_close(fd)
            raise error
        return original_os_close(fd)

    monkeypatch.setattr(cli._Output, "close", close)
    monkeypatch.setattr(cli.os, "close", os_close)
    with pytest.raises(type(error)) as caught:
        toy.run()
    assert caught.value is error and len(released) == 1
    output = outputs_seen[0]
    assert output._cleanup_recovery_attempted and output.fd is None
    assert output.release_state == "released"
    assert not (toy.out / "COMPLETE.json").exists()
    failure = json.loads((toy.out / "FAILED").read_bytes())
    assert failure["error_type"] == type(error).__name__ and failure["message"] == str(error)


def test_hcli2_stale_fd_left_by_external_release_is_recovered_without_retry(toy, monkeypatch):
    original_close, original_os_close = cli._Output.close, cli.os.close
    error = KeyboardInterrupt("external release before close state update")
    outputs_seen = []

    def close(output, primary=None):
        if primary is None and output.fd is not None:
            outputs_seen.append(output)
            original_os_close(output.fd)
            raise error
        return original_close(output, primary)

    monkeypatch.setattr(cli._Output, "close", close)
    with pytest.raises(KeyboardInterrupt) as caught:
        toy.run()
    assert caught.value is error
    assert outputs_seen[0]._cleanup_recovery_attempted
    assert not (toy.out / "COMPLETE.json").exists() and (toy.out / "FAILED").exists()


def test_hcli2_close_error_before_release_uses_only_verified_live_access(toy, monkeypatch):
    original_close, original_os_close = cli._Output.close, cli.os.close
    error = OSError("close failed before descriptor release")
    active, failed = [], []

    def close(output, primary=None):
        if primary is None and output.fd is not None:
            active.append(output)
        return original_close(output, primary)

    def os_close(fd):
        if active and not failed:
            failed.append(fd)
            raise error
        return original_os_close(fd)

    monkeypatch.setattr(cli._Output, "close", close)
    monkeypatch.setattr(cli.os, "close", os_close)
    with pytest.raises(OSError) as caught:
        toy.run()
    assert caught.value is error and not active[0]._cleanup_recovery_attempted
    assert active[0].fd is None and active[0].release_state == "released"
    assert not (toy.out / "COMPLETE.json").exists() and (toy.out / "FAILED").exists()


def test_hcli2_reused_foreign_fd_is_neither_used_nor_closed(toy, monkeypatch):
    original_close, original_os_close = cli._Output.close, cli.os.close
    error = OSError("release and reuse descriptor")
    foreign = toy.work / "foreign-descriptor"
    foreign.write_bytes(b"foreign unchanged")
    active, foreign_fds, attempted = [], [], []

    def close(output, primary=None):
        if primary is None and output.fd is not None:
            active.append(output)
        return original_close(output, primary)

    def os_close(fd):
        attempted.append(fd)
        if active and not foreign_fds:
            original_os_close(fd)
            new = os.open(foreign, os.O_RDONLY)
            if new != fd:
                os.dup2(new, fd)
                original_os_close(new)
            foreign_fds.append(fd)
            raise error
        return original_os_close(fd)

    monkeypatch.setattr(cli._Output, "close", close)
    monkeypatch.setattr(cli.os, "close", os_close)
    try:
        with pytest.raises(OSError) as caught:
            toy.run()
        assert caught.value is error
        fd = foreign_fds[0]
        assert attempted.count(fd) == 1
        assert os.read(fd, 100) == b"foreign unchanged"
        assert foreign.read_bytes() == b"foreign unchanged"
        assert not (toy.out / "COMPLETE.json").exists() and (toy.out / "FAILED").exists()
        assert any("stale/reused" in note for note in error.__notes__)
    finally:
        for fd in foreign_fds:
            original_os_close(fd)


@pytest.mark.parametrize("substitution", ["directory", "symlink", "foreign_marker"])
def test_hcli2_cleanup_recovery_refuses_substitution_or_foreign_marker(
        toy, monkeypatch, substitution):
    original_close, original_os_close = cli._Output.close, cli.os.close
    error = KeyboardInterrupt("release followed by substitution")
    active, released = [], []
    moved = toy.root / "original_owned"

    def close(output, primary=None):
        if primary is None and output.fd is not None:
            active.append(output)
        return original_close(output, primary)

    def os_close(fd):
        if active and not released:
            released.append(fd)
            original_os_close(fd)
            if substitution == "foreign_marker":
                marker = toy.root / "foreign_complete"
                marker.write_bytes(b"foreign marker")
                marker.replace(toy.out / "COMPLETE.json")
            else:
                toy.out.rename(moved)
                replacement = toy.root / "replacement"
                replacement.mkdir()
                (replacement / "COMPLETE.json").write_bytes(b"foreign marker")
                if substitution == "symlink":
                    toy.out.symlink_to(replacement, target_is_directory=True)
                else:
                    replacement.rename(toy.out)
            raise error
        return original_os_close(fd)

    monkeypatch.setattr(cli._Output, "close", close)
    monkeypatch.setattr(cli.os, "close", os_close)
    with pytest.raises(KeyboardInterrupt) as caught:
        toy.run()
    assert caught.value is error and active[0]._cleanup_recovery_attempted
    assert (toy.out / "COMPLETE.json").read_bytes() == b"foreign marker"
    assert not (toy.out / "FAILED").exists()
    assert not (moved / "FAILED").exists()
    assert any("cleanup recovery refused" in note for note in error.__notes__)


def test_hcli2_recovery_acquisition_handoff_abort_closes_new_descriptor(toy, monkeypatch):
    original_close, original_os_close = cli._Output.close, cli.os.close
    original_open, original_fstat = cli.os.open, cli.os.fstat
    error = KeyboardInterrupt("initial release error")
    handoff = SystemExit("recovery handoff error")
    active, released, recovery = [], [], []

    def close(output, primary=None):
        if primary is None and output.fd is not None:
            active.append(output)
        return original_close(output, primary)

    def os_close(fd):
        if active and not released:
            released.append(fd)
            original_os_close(fd)
            raise error
        return original_os_close(fd)

    def open_(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        if released and Path(path) == toy.out and flags & os.O_DIRECTORY:
            recovery.append(fd)
        return fd

    def fstat(fd):
        if recovery and fd == recovery[0]:
            raise handoff
        return original_fstat(fd)

    monkeypatch.setattr(cli._Output, "close", close)
    monkeypatch.setattr(cli.os, "close", os_close)
    monkeypatch.setattr(cli.os, "open", open_)
    monkeypatch.setattr(cli.os, "fstat", fstat)
    with pytest.raises(KeyboardInterrupt) as caught:
        toy.run()
    assert caught.value is error and len(recovery) == 1
    with pytest.raises(OSError) as missing:
        original_fstat(recovery[0])
    assert missing.value.errno == cli.errno.EBADF
    assert any("recovery handoff error" in note for note in error.__notes__)


def test_hcli_repair_normal_fold1_full_budget_cli_roundtrip(tmp_path, monkeypatch):
    toy = Toy(tmp_path, monkeypatch, fold=1, budget=100, zero=True)
    result = subprocess.run(toy.command(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    counts, _ = outputs(toy)
    matrix = np.array(counts["tissue_confusion"])
    assert matrix.sum() > np.trace(matrix)
