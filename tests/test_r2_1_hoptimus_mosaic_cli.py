"""Tiny synthetic full populations, saved states and custody graphs only."""
from collections import Counter
from dataclasses import asdict
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from prosper.evaluation import r2_1_hoptimus_mosaic_cli as cli


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return cli.cpu.json_bytes(value)


def binding(data):
    return {"size": len(data), "sha256": sha(data)}


def descriptor(path):
    return {"path": str(path), **binding(Path(path).read_bytes())}


def csv_bytes(header, rows):
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return text.getvalue().encode()


def synthetic_state(manifest, *, warning=False):
    """Construct a valid tiny fixture state, not a fit or an alternative loader."""
    weights = {row.tissue_label: row.balanced_weight for row in manifest.class_stats}
    estimator = cli.fitter.SGDClassifier(**cli.fitter._parameters(weights))
    estimator.classes_ = np.arange(1, 6, dtype=np.int64)
    estimator.coef_ = np.zeros((5, 1536), np.float32)
    estimator.coef_[0, 0], estimator.coef_[1, 0] = 20.0, 30.0
    direction = 1 if manifest.held_out_fold % 2 == 0 else -1
    if manifest.budget_percent == 100:
        direction *= -1
    estimator.coef_[0, 1], estimator.coef_[1, 1] = 20.0 * direction, -20.0 * direction
    estimator.intercept_ = np.array([-3.0, 3.0, -3.0, -3.0, -3.0], np.float32)
    estimator.n_features_in_, estimator.n_iter_, estimator.t_ = 1536, 1, 20.0
    records = manifest.tiles
    warnings = ([{"category": "sklearn.exceptions.ConvergenceWarning",
                  "message": "retained synthetic warning", "filename": "fixture.py", "lineno": 1}]
                if warning else [])
    diagnostic = {
        "schema_version": 1, "held_out_fold": manifest.held_out_fold,
        "budget_percent": manifest.budget_percent, "selected_tiles": len(records),
        "selected_slides": len({row.slide_id for row in records}),
        "selected_patients": len({row.patient_id for row in records}),
        "pool_count": manifest.pool_count, "selected_fraction": manifest.selected_fraction,
        "class_support": [{
            "tissue_label": stat.tissue_label, "pool_tiles": stat.pool_count,
            "selected_tiles": stat.selected_count, "selected_fraction": stat.fraction,
            "balanced_weight": stat.balanced_weight,
            "selected_slides": len({row.slide_id for row in records if row.tissue_label == stat.tissue_label}),
            "selected_patients": len({row.patient_id for row in records if row.tissue_label == stat.tissue_label}),
        } for stat in manifest.class_stats],
        "ordered_tile_ids_sha256": cli.manifests._ordered_ids_sha(records),
        "feature_sha256": sha(b"synthetic selected feature declaration"),
        "feature_shape": [len(records), 1536], "feature_dtype": "<f4",
        "parameters": estimator.get_params(deep=False), "classes": [1, 2, 3, 4, 5],
        "coefficient_shape": [5, 1536], "coefficient_dtype": "<f4",
        "intercept_shape": [5], "intercept_dtype": "<f4",
        "n_features_in": 1536, "n_iter": 1, "t": 20.0, "finite_state": True,
        "n_iter_semantics": "maximum_across_five_one_vs_rest_fits",
        "warnings": warnings, "iteration_limit_warning": bool(warnings),
        "convergence_status": "iteration_limit_warning_retained" if warnings else
                              "no_warning_not_optimality_claim",
        "runtime": cli.fitter._runtime_profile(),
        "fitted_state_sha256": cli.fitter._fingerprint(estimator),
        "limits": ["synthetic_only_not_a_training_result"],
    }
    return cli.fitter.FittedHead(estimator, diagnostic)


class Toy:
    def __init__(self, tmp_path, monkeypatch):
        self.work = tmp_path / "mosaic_toy"
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
        self.source_paths = {role: self.controls / (role + ".json") for role in cli.SOURCES}
        self.stage_dir = self.controls / "stage0_attempt_001"
        self.stage_dir.mkdir()
        for role, leaf in (("stage0_complete", "COMPLETE.json"), ("stage0_inventory", "inventory.csv"),
                           ("stage0_summary", "summary.json")):
            self.source_paths[role] = self.stage_dir / leaf
        self.retrieval = (
            cli.kernel.RetrievalRecord("slide_B", "patient_B", "RB"),
            cli.kernel.RetrievalRecord("slide_Z", "patient_A", "WT"),
            cli.kernel.RetrievalRecord("slide_C", "patient_A", "EWS"),
        )
        self.annotations = []
        for patient in ("patient_A", "patient_B", "immune_A", "immune_B", "annotation_C", "annotation_D"):
            labels = [1, 3, 4, 5] + ([2] if patient.startswith("immune") else [])
            slide = "annotation_" + patient
            for index, label in enumerate(labels):
                x, y = index * 512, 512
                self.annotations.append(cli.kernel.AnnotationRecord(
                    cli.kernel.physical_tile_id(slide, x, y), slide, patient,
                    "NORMAL" if patient == "immune_A" else "WT", 0, x, y, 512, label, index))
        folds = cli.kernel.build_patient_folds(self.annotations, self.retrieval)
        self.mapping = {row.patient_id: row.fold for row in folds.patients}
        self.manifests = {(row.held_out_fold, row.budget_percent): row
                          for row in cli.kernel.build_training_manifests(
                              self.annotations, self.retrieval, folds)}
        patients = sorted(self.mapping)
        ordered = sorted(patients, key=lambda p: (cli.kernel.patient_fold_hash(p), p))
        retrieval_patients = {row.patient_id for row in self.retrieval}
        self.source_paths["patient_folds"].write_bytes(csv_bytes(
            cli.manifests.FOLD_HEADER,
            ((p, self.mapping[p], ordered.index(p), cli.kernel.patient_fold_hash(p),
              1, int(p in retrieval_patients)) for p in patients)))
        self.source_paths["retrieval_population"].write_bytes(csv_bytes(
            cli.manifests.RETRIEVAL_HEADER,
            ((i, i, i + 10, row.slide_id, row.patient_id, row.diagnostic_group,
              self.mapping[row.patient_id]) for i, row in enumerate(self.retrieval))))
        self.features, self.raw = [], {}
        for index, row in enumerate(self.retrieval):
            raw = np.zeros((3, 1536), np.float32)
            if row.slide_id == "slide_B":
                raw[:, 1] = 1.0
            if row.slide_id == "slide_C":
                raw[1, 1], raw[2, 0] = 1.0, 20.0
            xy = [(1024, 512), (0, 1024), (512, 0)]
            self.raw[row.slide_id] = raw, xy
            payload = pickle.dumps((raw, [{"wsi_loc": list(pair)} for pair in xy]), protocol=4)
            path = self.bank / (row.slide_id + ".pkl")
            path.write_bytes(payload)
            self.features.append(dict(slide_id=row.slide_id, **descriptor(path),
                                      rows=3, columns=1536, dtype="float32"))
        self.cpu_run = "synthetic_cpu_filter"
        self.plan = {
            "schema_version": 1, "run_id": self.cpu_run, "task": cli.manifests.TASK,
            "classes": [1, 2, 3, 4, 5],
            "heads": [{"held_out_fold": f, "budget_percent": b,
                       "pool_count": man.pool_count, "selected_count": man.selected_count,
                       "selected_fraction": man.selected_fraction,
                       "class_stats": [asdict(row) for row in man.class_stats],
                       "training_tile_ids_sha256": cli.manifests._ordered_ids_sha(man.tiles)}
                      for (f, b), man in self.manifests.items()],
        }
        self.source_paths["head_plan"].write_bytes(encode(self.plan))
        self.cpu_summary = {
            "schema_version": 1, "run_id": self.cpu_run, "task": cli.manifests.TASK,
            "policy": cli.manifests.POLICY,
            "counts": {"raw_slides": 4, "target_slides": 3, "patients": 2, "groups": 3,
                       "annotation_tiles": len(self.annotations), "annotation_slides": 6,
                       "annotation_patients": 6, "cross_label_patients": 1},
            "details": {"fold_support": [asdict(row) for row in folds.support]},
        }
        self.source_paths["cpu_summary"].write_bytes(encode(self.cpu_summary))
        self.spec = {
            "schema_version": 1, "run_id": "synthetic_full_mosaic_population", "task": cli.TASK,
            "sources": {}, "code_files": {
                role: descriptor(path) for role, path in cli.code_paths(cli._reader()).items()},
            "features": self.features, "heads": [], "parameters": {"prediction_batch_rows": 4096},
            "expected": dict(slides=3, patients=2, diagnostic_groups=3, union_patients=6,
                             feature_files=3, original_tiles=9, heads=10,
                             slide_head_cells=30, tile_head_predictions=90),
        }
        self.heads = {}
        self.head_objects = {}
        for key, manifest in self.manifests.items():
            head = synthetic_state(manifest, warning=(key == (0, 5)))
            self.head_objects[key] = head
            name = cli._name(key)
            folder = self.controls / name
            folder.mkdir()
            paths = {role: folder / (role + ".bin") for role in cli.HEAD_ASSETS}
            self.heads[key] = paths
            historical = "/historical/" + ("pilots/" if key in ((0, 5), (1, 100)) else "remaining/") + name
            registry_keys = {"complete": historical + "/COMPLETE.json",
                             **{leaf: historical + "/" + leaf for leaf in cli.REGISTRY_NAMES}}
            producer_path = "/historical/declarations/" + name + ".json"
            held = [row for row in self.annotations if self.mapping[row.patient_id] == key[0]]
            counts = dict(annotation_tiles=len(self.annotations), annotation_slides=6,
                          annotation_patients=6, union_patients=6, feature_files=6,
                          training_tiles=len(manifest.tiles),
                          training_slides=len({r.slide_id for r in manifest.tiles}),
                          training_patients=len({r.patient_id for r in manifest.tiles}),
                          held_out_tiles=len(held), held_out_slides=len({r.slide_id for r in held}),
                          held_out_patients=len({r.patient_id for r in held}))
            producer_spec = {
                "schema_version": 1, "run_id": "synthetic_" + name, "task": cli.producer.TASK,
                "parameters": dict(held_out_fold=key[0], budget_percent=key[1],
                                   prediction_batch_rows=4096),
                "sources": {"head_plan": descriptor(self.source_paths["head_plan"])},
                "code_files": {"fit": self.spec["code_files"]["fit"]},
                "features": [{"slide_id": "annotation_" + p} for p in patients],
                "expected": counts,
            }
            paths["spec"].write_bytes(encode(producer_spec))
            paths["diagnostics"].write_bytes(encode(head.diagnostics))
            paths["model"].write_bytes(pickle.dumps(head, protocol=4))
            head_inputs = {
                "schema_version": 1, "run_id": producer_spec["run_id"], "task": cli.producer.TASK,
                "spec_sha256": descriptor(paths["spec"])["sha256"],
                "head": dict(held_out_fold=key[0], budget_percent=key[1]),
                "bindings": [
                    dict(asset_id="spec", role="spec", path=producer_path, resolved_path=producer_path,
                         **binding(paths["spec"].read_bytes())),
                    dict(asset_id="source:head_plan", role="head_plan",
                         path="/historical/CPU/head_plan.json", resolved_path="/historical/CPU/head_plan.json",
                         **binding(self.source_paths["head_plan"].read_bytes()))],
                "consumed_cpu_artifacts": {"head_plan.json": binding(self.source_paths["head_plan"].read_bytes())},
            }
            paths["input_bindings"].write_bytes(encode(head_inputs))
            head_summary = {
                "schema_version": 1, "run_id": producer_spec["run_id"], "task": cli.producer.TASK,
                "counts": counts,
                "policy": dict(held_out_fold=key[0], budget_percent=key[1], classes=[1, 2, 3, 4, 5],
                               recipe="r2_1_fixed_sgd_ovr_v1", feature_dtype="float32",
                               prediction_batch_rows=4096, decision="probability_argmax_class1",
                               scope=cli.producer.SCOPE, tuning="none",
                               encoder_replay="not_performed_existing_annotation_bank"),
                "details": dict(serialization_protocol=4, held_out_prediction_rows=len(held),
                                training_tile_ids_sha256=head.diagnostics["ordered_tile_ids_sha256"],
                                fitted_state_sha256=head.diagnostics["fitted_state_sha256"],
                                fit_warnings=head.diagnostics["warnings"],
                                convergence_status=head.diagnostics["convergence_status"]),
            }
            paths["summary"].write_bytes(encode(head_summary))
            roles = dict(zip(cli.REGISTRY_NAMES, ("model", "diagnostics", "input_bindings", "summary")))
            outputs = {leaf: binding(paths[role].read_bytes()) for leaf, role in roles.items()}
            outputs.update({leaf: binding(b"unconsumed synthetic producer reference")
                            for leaf in ("held_out_rows.csv.gz", "probabilities.npy",
                                         "predictions.npy", "counts.json")})
            paths["complete"].write_bytes(encode(dict(
                schema_version=1, run_id=producer_spec["run_id"], task=cli.producer.TASK,
                scope=cli.producer.SCOPE, spec_sha256=head_inputs["spec_sha256"], outputs=outputs)))
            self.spec["heads"].append(dict(
                held_out_fold=key[0], budget_percent=key[1],
                assets={role: descriptor(path) for role, path in paths.items()},
                registry_keys=registry_keys, producer_spec_path=producer_path))
        self.refresh_stage()
        self.refresh_head_acceptance()
        self.refresh_cpu()

    def refresh_stage(self):
        inventory, feature_bindings, totals, dtypes = [], [], Counter(), {}
        entries = [(feature["slide_id"], feature["rows"], feature) for feature in self.features]
        entries.append(("excluded_NORMAL", 2, dict(
            path=str(self.bank / "excluded_NORMAL.pkl"), size=1, sha256=sha(b"not read"))))
        for bank in ("uni2", "virchow2", "h_optimus_1"):
            for slide, rows, feature in entries:
                info = feature if bank == "h_optimus_1" else dict(
                    path=f"/historical/{bank}/{slide}.pkl", size=1, sha256=sha(bank.encode()))
                inventory.append((bank, slide, rows, rows, "float32", info["path"],
                                  str(Path(info["path"]).resolve()), info["size"], info["sha256"]))
                feature_bindings.append(dict(
                    kind="feature", name=bank + "/" + slide,
                    resolved_path=str(Path(info["path"]).resolve()),
                    **{name: info[name] for name in ("path", "size", "sha256")}))
                totals[bank] += rows
            dtypes[bank] = {"float32": len(entries)}
        self.source_paths["stage0_inventory"].write_bytes(csv_bytes(cli.INVENTORY_HEADER, inventory))
        old_spec = str(self.controls / "stage0_spec.json")
        self.metadata = dict(kind="metadata", name="metadata", path="/historical/metadata.csv",
                             resolved_path="/historical/metadata.csv",
                             **binding(b"synthetic historical metadata"))
        canonical = dict(kind="mosaic", name="canonical", path="/historical/canonical.pkl",
                         resolved_path="/historical/canonical.pkl",
                         **binding(b"synthetic historical canonical not opened"))
        summary = dict(
            schema_version=1, run_id="synthetic_stage0", scope="feature-bank binding only; unverified encoder",
            canonical_slides=len(entries), canonical_tiles=sum(row[1] for row in entries),
            inventory_rows=len(inventory), bank_tile_totals=dict(totals), bank_dtype_counts=dtypes,
            declarations=dict(
                schema_version=1, run_id="synthetic_stage0",
                mosaic=dict(path=canonical["path"], level=0, patch_size=512,
                            expected_slides=len(entries), expected_tiles=sum(row[1] for row in entries)),
                metadata=dict(path=self.metadata["path"], **cli.cpu.METADATA_COLUMNS),
                banks=[dict(name=bank, feature_dim=1280 if bank == "virchow2" else 1536,
                            path=str(self.bank) if bank == "h_optimus_1" else "/historical/" + bank,
                            dtypes=["float32"]) for bank in ("uni2", "virchow2", "h_optimus_1")]),
            input_bindings=[
                dict(kind="spec", name="spec", path=old_spec, resolved_path=old_spec,
                     **binding(b"synthetic historical stage0 spec")),
                self.metadata, canonical,
                dict(kind="code", name="running_auditor",
                     resolved_path=self.spec["code_files"]["legacy_io"]["path"],
                     **self.spec["code_files"]["legacy_io"]),
                *feature_bindings],
        )
        self.source_paths["stage0_summary"].write_bytes(encode(summary))
        outputs = {leaf: binding(self.source_paths[role].read_bytes()) for leaf, role in
                   (("inventory.csv", "stage0_inventory"), ("summary.json", "stage0_summary"))}
        self.source_paths["stage0_complete"].write_bytes(encode(dict(
            schema_version=1, run_id="synthetic_stage0", scope="feature-bank binding only", outputs=outputs)))
        review = dict(
            schema_version=1, disposition="accepted", scope="Stage0 feature-bank binding only",
            artifacts={f"stage0_attempt_001/{leaf}": descriptor(self.source_paths[role])["sha256"]
                       for leaf, role in (("COMPLETE.json", "stage0_complete"),
                                          ("inventory.csv", "stage0_inventory"), ("summary.json", "stage0_summary"))},
            source_sha256=dict(metadata=self.metadata["sha256"], canonical_mosaic=canonical["sha256"],
                               auditor_module=self.spec["code_files"]["legacy_io"]["sha256"]))
        review["artifacts"]["stage0_spec.json"] = summary["input_bindings"][0]["sha256"]
        self.source_paths["stage0_review"].write_bytes(encode(review))

    def refresh_head_acceptance(self):
        registry = dict(
            schema_version=1,
            scope="Lead eight-output plus retained-pilot control/row/count reconciliation; not production aggregation",
            exact_ten_tuple_and_once_per_budget_coverage=True, heads={}, output_hashes={})
        for head in self.spec["heads"]:
            key = cli._key(head)
            paths = self.heads[key]
            head["assets"] = {role: descriptor(path) for role, path in paths.items()}
            registry["heads"][cli._name(key)] = dict(
                complete_sha256=descriptor(paths["complete"])["sha256"],
                retained_pilot=key in ((0, 5), (1, 100)))
            registry["output_hashes"][head["registry_keys"]["complete"]] = descriptor(paths["complete"])["sha256"]
            for leaf, role in zip(cli.REGISTRY_NAMES, ("model", "diagnostics", "input_bindings", "summary")):
                registry["output_hashes"][head["registry_keys"][leaf]] = descriptor(paths[role])["sha256"]
        pilot = dict(
            schema_version=1, disposition="accepted", review_kind="actual_outputs_and_observed_resources",
            scope="Actual source/model/prediction/count/resource outcomes of the two prescribed pilot heads only",
            complete_bindings={name + "_attempt001/COMPLETE.json": registry["heads"][name]["complete_sha256"]
                               for name in ("fold0_budget5", "fold1_budget100")})
        self.source_paths["head_registry"].write_bytes(encode(registry))
        self.source_paths["pilot_acceptance"].write_bytes(encode(pilot))
        registry_sha = descriptor(self.source_paths["head_registry"])["sha256"]
        acceptance = dict(
            schema_version=1, disposition="accepted",
            review_kind="actual_outputs_observed_resources_and_combined_ten_head_coverage",
            scope="Eight new prescribed annotation heads plus exact combined coverage with two unchanged accepted pilots",
            lead_reconciliation_sha256=registry_sha,
            retained_pilot_actual_acceptance_sha256=descriptor(self.source_paths["pilot_acceptance"])["sha256"],
            complete_and_artifact_bindings=dict(
                file="lead_actual_output_reconciliation.json", sha256=registry_sha,
                fields=["heads.*.complete_sha256", "output_hashes"], complete_records=10))
        self.source_paths["head_acceptance"].write_bytes(encode(acceptance))

    def refresh_cpu(self):
        old_spec = binding(b"synthetic historical cpu spec")
        entries = [dict(kind="spec", name="spec", **old_spec),
                   dict(kind="source", name="metadata", **{k: self.metadata[k]
                        for k in ("path", "resolved_path", "size", "sha256")})]
        entries.extend(dict(kind="source", name=role, **descriptor(self.source_paths[role]))
                       for role in ("stage0_review", "stage0_complete", "stage0_inventory", "stage0_summary"))
        entries.extend(dict(kind="code", name=old, **self.spec["code_files"][role]) for role, old in (
            ("filter_kernel", "kernel"), ("filter_cli", "runner"), ("cpu_io", "io"),
            ("legacy_io", "legacy_io"), ("package_init", "package_init"), ("evaluation_init", "evaluation_init")))
        self.source_paths["cpu_input_bindings"].write_bytes(encode(dict(
            schema_version=1, run_id=self.cpu_run, task=cli.manifests.TASK,
            spec_sha256=old_spec["sha256"], bindings=entries)))
        outputs = {leaf: binding(self.source_paths[role].read_bytes()) for leaf, role in (
            ("input_bindings.json", "cpu_input_bindings"), ("retrieval_population.csv", "retrieval_population"),
            ("patient_folds.csv", "patient_folds"), ("summary.json", "cpu_summary"), ("head_plan.json", "head_plan"))}
        outputs.update({leaf: binding(b"unconsumed synthetic source")
                        for leaf in ("annotation_assignments.csv.gz", "training_rows.csv.gz")})
        self.source_paths["cpu_complete"].write_bytes(encode(dict(
            schema_version=1, run_id=self.cpu_run, task=cli.manifests.TASK,
            scope="hoptimus_filter_manifests_only", spec_sha256=old_spec["sha256"], outputs=outputs)))
        self.source_paths["cpu_review"].write_bytes(encode(dict(
            schema_version=1, disposition="accepted", branches={"filter": {"disposition": "accepted"}},
            contract_sha256=cli.assets._CPU_CONTRACT,
            artifacts={"filter_attempt_001/COMPLETE.json": descriptor(self.source_paths["cpu_complete"])["sha256"]})))
        self.rebind()

    def rebind(self):
        self.spec["sources"] = {role: descriptor(path) for role, path in self.source_paths.items()}
        self.spec_path.write_bytes(encode(self.spec))
        self.digest = sha(self.spec_path.read_bytes())

    def sources(self):
        return {role: path.read_bytes() for role, path in self.source_paths.items()}

    def change_json(self, role, edit, *, cpu=True):
        path = self.source_paths[role]
        value = json.loads(path.read_bytes())
        edit(value)
        path.write_bytes(encode(value))
        if cpu:
            self.refresh_cpu()
        else:
            self.rebind()

    def run(self):
        return cli.run(self.spec_path, self.digest, self.out, self.root)

    def command(self):
        return [sys.executable, "-m", "prosper.evaluation.r2_1_hoptimus_mosaic_cli",
                "--spec", str(self.spec_path), "--spec-sha256", self.digest,
                "--out", str(self.out), "--out-root", str(self.root)]


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("mosaic stage must not fit, partial-fit or use estimator.predict")
    monkeypatch.setattr(cli.fitter, "fit_head", forbidden)
    monkeypatch.setattr(cli.fitter.SGDClassifier, "fit", forbidden)
    monkeypatch.setattr(cli.fitter.SGDClassifier, "partial_fit", forbidden)
    monkeypatch.setattr(cli.fitter.SGDClassifier, "predict", forbidden)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    return Toy(tmp_path, monkeypatch)


@pytest.fixture
def stable_reader():
    expected = os.environ.get("R21_MOSAIC_STABLE_READER_SHA")
    if not expected:
        pytest.skip("full reader integration awaits lead-coordinated stable reader SHA")
    path = Path(cli._reader().__file__)
    assert descriptor(path)["sha256"] == expected
    yield expected
    assert descriptor(path)["sha256"] == expected, "reader bytes changed during integration"


def check_outputs(toy):
    marker = json.loads((toy.out / "COMPLETE.json").read_bytes())
    assert set(marker["outputs"]) == cli.artifact_names() and len(marker["outputs"]) == 36
    assert set(p.name for p in toy.out.iterdir()) == cli.artifact_names() | {"COMPLETE.json"}
    for name, expected in marker["outputs"].items():
        assert binding((toy.out / name).read_bytes()) == expected
    assert not (toy.out / "FAILED").exists()
    assert (toy.out / "retrieval_population.csv").read_bytes() == toy.source_paths["retrieval_population"].read_bytes()
    xy = np.load(toy.out / "coordinates.npy", allow_pickle=False)
    assert xy.dtype.str == "<i8" and xy.tolist() == [
        list(pair) for row in toy.retrieval for pair in toy.raw[row.slide_id][1]]
    offsets = list(csv.DictReader(io.StringIO((toy.out / "slide_offsets.csv").read_text())))
    assert [row["slide_id"] for row in offsets] == [row.slide_id for row in toy.retrieval]
    counts = list(csv.DictReader(io.StringIO(gzip.decompress(
        (toy.out / "mask_counts.csv.gz").read_bytes()).decode())))
    assert len(counts) == 30
    zero = 0
    for number, key in enumerate(cli.TUPLES):
        p = np.load(toy.out / (cli._name(key) + "_probabilities.npy"), allow_pickle=False)
        y = np.load(toy.out / (cli._name(key) + "_predictions.npy"), allow_pickle=False)
        mask = np.load(toy.out / (cli._name(key) + "_mask.npy"), allow_pickle=False)
        assert p.dtype.str == "<f4" and y.dtype.str == "<i8" and mask.dtype.str == "|b1"
        assert np.array_equal(y, p.argmax(axis=1) + 1) and np.array_equal(mask, y == 1)
        for index, row in enumerate(toy.retrieval):
            source = toy.raw[row.slide_id][0]
            want = cli.fitter.predict_head(toy.head_objects[key], source)
            assert p[index * 3:(index + 1) * 3].tobytes() == want.probabilities.tobytes()
            assert y[index * 3:(index + 1) * 3].tobytes() == want.labels.tobytes()
            count = counts[number * 3 + index]
            kept = int(mask[index * 3:(index + 1) * 3].sum())
            assert int(count["retained_tiles"]) == kept
            assert int(count["query_fold_eligible"]) == int(toy.mapping[row.patient_id] == key[0])
            assert [int(count[f"predicted_class_{label}"]) for label in range(1, 6)] == \
                np.bincount(want.labels, minlength=6)[1:6].tolist()
            assert count["status"] == ("nonempty" if kept else "zero_predicted_tumor")
            zero += int(kept == 0)
        assert not mask[3:6].any(), "zero-tumor slide must be retained explicitly"
        assert mask[8], "saturated PROBA tie is tumor despite lower decision score"
        assert p[8, 0] < 0.5, "mask must not use binary .5 threshold"
    summary = json.loads((toy.out / "summary.json").read_bytes())
    assert summary["counts"] == dict(toy.spec["expected"], zero_mask_cells=zero, nonzero_mask_cells=30 - zero)
    assert len(summary["details"]["head_provenance"]) == 10
    assert summary["details"]["packed_array_data_bytes"] == 9 * 306
    assert summary["limitations"] == list(cli.LIMITATIONS)
    records = json.loads((toy.out / "input_bindings.json").read_bytes())
    assert records["current_environment"]["native_environment"] == {name: "1" for name in cli.producer.NATIVE_VARIABLES}
    assert len(records["feature_consumption"]) == 3
    assert all(row["status"] == "decoded_once" for row in records["feature_consumption"])
    assert set(records["executing_code_paths"]) == set(toy.spec["code_files"])
    assert all(str(Path(row["path"]).resolve()) == row["resolved_path"] for row in records["bindings"])
    assert [record["held_out_fold"] for record in records["head_provenance"]] == [key[0] for key in cli.TUPLES]
    assert len(records["producer_artifact_links"]) == 10
    assert all(set(link["referenced_not_reread"]) ==
               {"probabilities.npy", "predictions.npy", "counts.json", "held_out_rows.csv.gz"}
               for link in records["producer_artifact_links"])
    return summary


def test_normal_package_help():
    result = subprocess.run([sys.executable, "-m", "prosper.evaluation.r2_1_hoptimus_mosaic_cli", "--help"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "--spec-sha256" in result.stdout and "--out-root" in result.stdout


def test_normal_cli_full_population_actual_shared_reader(toy, stable_reader):
    result = subprocess.run(toy.command(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["outputs"] == json.loads((toy.out / "COMPLETE.json").read_bytes())["outputs"]
    summary = check_outputs(toy)
    assert json.loads(summary["details"]["head_provenance"][0]["diagnostics_json"])["iteration_limit_warning"]


def test_api_once_decode_all_heads_and_query_fold_reference_routing(toy, stable_reader, monkeypatch):
    original = cli.decode_feature
    seen = []
    def decode(data, entry):
        seen.append(entry["slide_id"])
        return original(data, entry)
    monkeypatch.setattr(cli, "decode_feature", decode)
    toy.run()
    assert seen == [row.slide_id for row in toy.retrieval]
    check_outputs(toy)
    reference_fold = toy.mapping["patient_B"]
    key = next(key for key in cli.TUPLES if key[1] == 5 and key[0] % 2 != reference_fold % 2)
    reference_own = np.load(toy.out / f"fold{reference_fold}_budget5_mask.npy")
    query_head_reference = np.load(toy.out / (cli._name(key) + "_mask.npy"))
    assert not np.array_equal(reference_own[:3], query_head_reference[:3])


def test_controls_and_full_population_reconciliation_without_head_loading(toy):
    cli._schema(toy.spec, cli.code_paths(cli._reader()))
    summary, plans, _ = cli.authenticate(toy.spec, toy.sources())
    population, offsets = cli.reconcile(toy.spec, toy.sources(), summary)
    assert set(plans) == set(cli.TUPLES) and len(population) == 3
    assert [row["tile_start"] for row in offsets] == [0, 3, 6]
    assert [row["tile_stop"] for row in offsets] == [3, 6, 9]


@pytest.mark.parametrize("name,value", [
    ("schema_version", True), ("schema_version", 1.0), ("task", "hoptimus_main"),
])
def test_top_level_type_or_family(toy, name, value):
    toy.spec[name] = value
    with pytest.raises(ValueError):
        cli._schema(toy.spec, cli.code_paths(cli._reader()))


@pytest.mark.parametrize("name", sorted(cli.EXPECTED))
@pytest.mark.parametrize("value", [True, 1.0, -1, 0])
def test_strict_expected_counts(toy, name, value):
    toy.spec["expected"][name] = value
    with pytest.raises(ValueError):
        cli._schema(toy.spec, cli.code_paths(cli._reader()))


@pytest.mark.parametrize("edit", [
    lambda s: s["sources"].pop("stage0_review"),
    lambda s: s["sources"].update(extra=s["sources"]["cpu_review"]),
    lambda s: s["code_files"].pop("inherited:prosper.pipeline"),
    lambda s: s["code_files"].update({"inherited:prosper.arbitrary_new_helper": s["code_files"]["runner"]}),
    lambda s: s["parameters"].update(subset=True),
    lambda s: s["parameters"].update(prediction_batch_rows=4095),
    lambda s: s["heads"].pop(),
    lambda s: s["heads"].__setitem__(1, s["heads"][0]),
    lambda s: s["heads"][0].update(held_out_fold=True),
    lambda s: s["heads"][0].update(budget_percent=100.0),
    lambda s: s["features"].reverse(),
    lambda s: s["features"][0].update(dtype="float16"),
    lambda s: s["features"][0].update(columns=1535),
    lambda s: s["features"][0].update(rows=True),
    lambda s: s["features"][0].update(path="relative.pkl"),
    lambda s: s["sources"]["head_plan"].update(size=4 * cli.MIB + 1),
    lambda s: s["sources"]["head_registry"].update(size=2 * cli.MIB + 1),
])
def test_declared_schema_or_population_failure(toy, edit):
    edit(toy.spec)
    with pytest.raises(ValueError):
        cli._schema(toy.spec, cli.code_paths(cli._reader()))
        summary, _, _ = cli.authenticate(toy.spec, toy.sources())
        cli.reconcile(toy.spec, toy.sources(), summary)


def test_packed_resource_limit_before_allocate(toy, monkeypatch):
    toy.spec["expected"]["original_tiles"] = cli.ARRAY_LIMIT // 306 + 1
    toy.spec["expected"]["tile_head_predictions"] = toy.spec["expected"]["original_tiles"] * 10
    monkeypatch.setattr(cli.np, "full", lambda *a, **k: pytest.fail("must not allocate"))
    with pytest.raises(ValueError, match="2GiB"):
        cli._schema(toy.spec, cli.code_paths(cli._reader()))
    assert cli.packed_bytes(3635673) == 1112515938


@pytest.mark.parametrize("role,limit", [
    *[(role, cli.source_limit(role)) for role in sorted(cli.SOURCES)],
    ("spec", 8 * cli.MIB), ("code", 2 * cli.MIB),
    ("feature", 64 * cli.MIB), ("head_control", 2 * cli.MIB), ("head_model", 2 * cli.MIB),
])
def test_pre_read_policy_before_byte_scan(toy, monkeypatch, role, limit):
    declarations, policies = cli.declarations(toy.spec, toy.spec_path,
                                              toy.spec_path.read_bytes(), toy.digest)
    target = next(entry for entry in declarations if entry.role == role)
    oversized = cli.assets.AssetSpec(target.asset_id, role, target.path, target.sha256, limit + 1)
    entries = [oversized if entry is target else entry for entry in declarations]
    monkeypatch.setattr(cli.assets.AssetGuard, "_scan", lambda *a, **k: pytest.fail("no byte allocation"))
    with cli.assets.AssetGuard(entries, base_dir=toy.work, policies=policies) as guard:
        with pytest.raises(cli.assets.AssetError, match="encoded size exceeds policy"):
            guard.read_verified(target.asset_id)


@pytest.mark.parametrize("edit", [
    lambda s: s["declarations"]["banks"][2].update(feature_dim=1280),
    lambda s: s["declarations"]["mosaic"].update(level=True),
    lambda s: s["declarations"]["metadata"].update(patient_col="wrong"),
    lambda s: s["input_bindings"][1].update(sha256="a" * 64),
])
def test_authenticated_stage0_declaration_or_lineage_negative(toy, edit):
    toy.change_json("stage0_summary", edit)
    # Rebind outer receipts so this exercises semantics, not merely a stale SHA.
    complete = json.loads(toy.source_paths["stage0_complete"].read_bytes())
    complete["outputs"]["summary.json"] = binding(toy.source_paths["stage0_summary"].read_bytes())
    toy.source_paths["stage0_complete"].write_bytes(encode(complete))
    review = json.loads(toy.source_paths["stage0_review"].read_bytes())
    review["artifacts"]["stage0_attempt_001/summary.json"] = descriptor(toy.source_paths["stage0_summary"])["sha256"]
    review["artifacts"]["stage0_attempt_001/COMPLETE.json"] = descriptor(toy.source_paths["stage0_complete"])["sha256"]
    toy.source_paths["stage0_review"].write_bytes(encode(review))
    toy.refresh_cpu()
    with pytest.raises(ValueError):
        cli.authenticate(toy.spec, toy.sources())


@pytest.mark.parametrize("role", ["stage0_review", "stage0_complete", "cpu_review", "cpu_complete"])
def test_upstream_acceptance_or_task_negative(toy, role):
    toy.change_json(role, lambda value: value.update(disposition="rejected", scope="wrong", task="wrong"),
                    cpu=False)
    with pytest.raises(ValueError):
        cli.authenticate(toy.spec, toy.sources())


@pytest.mark.parametrize("kind", ["row_count", "width", "float64", "float16", "nan", "infinity",
                                  "bool_xy", "float_xy", "negative_xy", "overflow_xy",
                                  "duplicate_xy", "coordinate_rows", "list_matrix", "trailing"])
def test_feature_geometry_profile_and_framing(toy, kind):
    entry = toy.features[0]
    raw, xy = toy.raw[entry["slide_id"]]
    raw, xy = raw.copy(), [list(pair) for pair in xy]
    if kind == "row_count": raw = raw[:-1]
    if kind == "width": raw = raw[:, :-1]
    if kind in ("float64", "float16"): raw = raw.astype(kind)
    if kind in ("nan", "infinity"): raw[0, 0] = np.nan if kind == "nan" else np.inf
    if kind == "bool_xy": xy[0][0] = True
    if kind == "float_xy": xy[0][0] = 1024.0
    if kind == "negative_xy": xy[0][0] = -1
    if kind == "overflow_xy": xy[0][0] = 2**63
    if kind == "duplicate_xy": xy[1] = xy[0]
    if kind == "coordinate_rows": xy = xy[:-1]
    if kind == "list_matrix": raw = raw.tolist()
    data = pickle.dumps((raw, [{"wsi_loc": pair} for pair in xy]), protocol=4)
    if kind == "trailing": data += pickle.dumps(None, protocol=4)
    with pytest.raises(ValueError):
        cli.decode_feature(data, entry)


def test_decode_preserves_original_order_and_all_ten_slices(toy):
    summary, _, _ = cli.authenticate(toy.spec, toy.sources())
    _, offsets = cli.reconcile(toy.spec, toy.sources(), summary)
    coordinates, arrays, cells, rows = cli.allocate(9, 3)
    for entry, row in zip(toy.features, offsets):
        raw, xy = cli.decode_feature(Path(entry["path"]).read_bytes(), entry)
        cli.score_slide(toy.head_objects, raw, xy, row, coordinates, arrays, cells, rows)
    assert cells.all() and rows.sum() == 90 and coordinates[:3].tolist() == [
        list(pair) for pair in toy.raw["slide_B"][1]]
    for key in cli.TUPLES:
        cli.validate_array_slice(*arrays[key])
        assert not arrays[key][2][3:6].any()
        assert arrays[key][2][8] and arrays[key][0][8, 0] < 0.5
    with pytest.raises(ValueError, match="already written"):
        cli.score_slide(toy.head_objects, *toy.raw["slide_B"][:1],
                        np.array(toy.raw["slide_B"][1]), offsets[0], coordinates, arrays, cells, rows)


def test_fortran_input_layout_only_batch_copy_no_precision_change(toy):
    raw = np.asfortranarray(toy.raw["slide_B"][0])
    xy = np.array(toy.raw["slide_B"][1], np.int64)
    coordinates, arrays, cells, rows = cli.allocate(3, 1)
    row = dict(slide_index=0, tile_start=0, tile_stop=3)
    cli.score_slide(toy.head_objects, raw, xy, row, coordinates, arrays, cells, rows)
    assert raw.flags.f_contiguous
    for key in cli.TUPLES:
        expected = cli.fitter.predict_head(toy.head_objects[key], np.ascontiguousarray(raw))
        assert arrays[key][0].tobytes() == expected.probabilities.tobytes()


@pytest.mark.parametrize("edit", [
    lambda rows: rows[1].update(tile_start=2),
    lambda rows: rows[1].update(tile_start=4),
    lambda rows: rows[1].update(slide_id=rows[0]["slide_id"]),
    lambda rows: rows.pop(),
])
def test_offset_overlap_gap_duplicate_missing(toy, edit):
    summary, _, _ = cli.authenticate(toy.spec, toy.sources())
    _, offsets = cli.reconcile(toy.spec, toy.sources(), summary)
    edit(offsets)
    with pytest.raises(ValueError):
        cli.validate_offsets(offsets, toy.spec["expected"])


def test_borrowed_reader_failure_has_failed_and_no_fallback(toy, monkeypatch):
    error = RuntimeError("shared reader failure; no alternative model path")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(cli._reader(), "load_saved_annotation_head", fail)
    with pytest.raises(RuntimeError) as caught:
        toy.run()
    assert caught.value is error
    assert (toy.out / "FAILED").exists() and not (toy.out / "COMPLETE.json").exists()
    assert not (toy.out / "coordinates.npy").exists()


@pytest.mark.parametrize("error", [KeyboardInterrupt("cancel"), SystemExit(9)])
def test_borrowed_reader_baseexception_preserved(toy, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(cli._reader(), "load_saved_annotation_head", fail)
    with pytest.raises(type(error)) as caught:
        toy.run()
    assert caught.value is error
    assert (toy.out / "FAILED").exists() and not (toy.out / "COMPLETE.json").exists()


def test_existing_output_refused_without_touching(toy, monkeypatch):
    toy.out.mkdir()
    marker = toy.out / "COMPLETE.json"
    marker.write_bytes(b"foreign marker")
    monkeypatch.setattr(cli._reader(), "load_saved_annotation_head",
                        lambda *a, **k: pytest.fail("existing output must fail before reader"))
    with pytest.raises(FileExistsError):
        toy.run()
    assert marker.read_bytes() == b"foreign marker" and not (toy.out / "FAILED").exists()


def test_initial_feature_hash_failure_remains_failed_not_complete(toy, stable_reader, monkeypatch):
    path = Path(toy.features[0]["path"])
    data = bytearray(path.read_bytes())
    data[-2] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError):
        toy.run()
    assert (toy.out / "FAILED").exists() and not (toy.out / "COMPLETE.json").exists()


def test_final_feature_sweep_rejects_late_same_inode_mutation(toy, stable_reader, monkeypatch):
    original = cli.score_slide
    def score(*args, **kwargs):
        result = original(*args, **kwargs)
        row = args[3]
        if row["slide_index"] == 2:
            path = Path(toy.features[0]["path"])
            data = bytearray(path.read_bytes())
            data[-2] ^= 1
            path.write_bytes(data)
        return result
    monkeypatch.setattr(cli, "score_slide", score)
    with pytest.raises(ValueError, match="SHA256"):
        toy.run()
    assert (toy.out / "FAILED").exists() and not (toy.out / "COMPLETE.json").exists()


def tiny_output(toy):
    output = cli.producer._Output(toy.out, toy.root, [], [])
    output.claim()
    return output


def test_boolean_npy_readback_and_integer_disguise_rejection(toy, monkeypatch):
    output = tiny_output(toy)
    try:
        bindings = {}
        cli._bool_output(output, "mask.npy", np.array([True, False, False], bool), bindings)
        assert np.load(toy.out / "mask.npy").dtype.str == "|b1"
        with pytest.raises(ValueError, match="Boolean"):
            cli._bool_output(output, "integer.npy", np.array([1, 0, 0], np.int8), bindings)
        original = output.readback
        def readback(name, limit, semantic, **kwargs):
            if name == "corrupt.npy":
                np.save(toy.out / name, np.array([1, 0, 0], np.int8), allow_pickle=False)
            return original(name, limit, semantic, **kwargs)
        monkeypatch.setattr(output, "readback", readback)
        with pytest.raises(ValueError, match="header"):
            cli._bool_output(output, "corrupt.npy", np.array([True, False, False], bool), bindings)
    finally:
        output.close()


@pytest.mark.parametrize("error", [KeyboardInterrupt("post-close"), OSError("post-close")])
def test_post_release_failure_recovers_owned_complete_only(toy, monkeypatch, error):
    output = tiny_output(toy)
    original = cli.producer.os.close
    fd = output.fd
    bindings = {}
    cli.producer._json_output(output, "COMPLETE.json", {"synthetic": True}, bindings)
    def release(descriptor):
        original(descriptor)
        if descriptor == fd:
            raise error
    monkeypatch.setattr(cli.producer.os, "close", release)
    with pytest.raises(type(error)) as caught:
        output.close()
    assert caught.value is error
    monkeypatch.setattr(cli.producer.os, "close", original)
    output.fail(error, [])
    output.close(error)
    assert (toy.out / "FAILED").exists() and not (toy.out / "COMPLETE.json").exists()


@pytest.mark.parametrize("kind", ["replacement", "foreign_marker", "reused_descriptor"])
def test_owned_cleanup_never_uses_replaced_or_foreign_resources(toy, kind):
    output = tiny_output(toy)
    bindings = {}
    cli.producer._json_output(output, "COMPLETE.json", {"synthetic": True}, bindings)
    fd = output.fd
    os.close(fd)
    foreign_fd = None
    if kind == "replacement":
        old = toy.root / "moved_owned_attempt"
        toy.out.rename(old)
        toy.out.mkdir()
        (toy.out / "COMPLETE.json").write_bytes(b"foreign")
    elif kind == "foreign_marker":
        old_marker = toy.out / "COMPLETE.json"
        old_marker.rename(toy.out / "owned_old_marker")
        old_marker.write_bytes(b"foreign")
    else:
        foreign = toy.root / "foreign"
        foreign.mkdir()
        foreign_fd = os.open(foreign, os.O_RDONLY | os.O_DIRECTORY)
        if foreign_fd != fd:
            os.dup2(foreign_fd, fd)
            os.close(foreign_fd)
            foreign_fd = fd
    error = RuntimeError("cleanup must preserve original")
    try:
        if kind == "reused_descriptor":
            output.fail(error, [])
            assert os.fstat(foreign_fd)
            assert os.listdir(foreign_fd) == []
            assert not (toy.out / "COMPLETE.json").exists()
        else:
            with pytest.raises(ValueError, match="recovery refused"):
                output.fail(error, [])
            assert (toy.out / "COMPLETE.json").read_bytes() == b"foreign"
            assert not (toy.out / "FAILED").exists()
    finally:
        output.close(error)
        if foreign_fd is not None:
            os.close(foreign_fd)


@pytest.mark.parametrize("kind", ["labels", "mask", "probability", "sum", "nan"])
def test_semantic_prediction_corruption(kind):
    p = np.full((3, 5), 0.2, np.float32)
    y = np.ones(3, np.int64)
    mask = np.ones(3, bool)
    if kind == "labels": y[0] = 2
    if kind == "mask": mask[0] = False
    if kind == "probability": p[0, 0] = 1.1
    if kind == "sum": p[0] = 0.0
    if kind == "nan": p[0, 0] = np.nan
    with pytest.raises(ValueError):
        cli.validate_array_slice(p, y, mask)


def test_current_startup_graph_is_explicit_not_new_helper(toy):
    paths = cli.code_paths(cli._reader())
    inherited = {key.removeprefix("inherited:") for key in paths if key.startswith("inherited:")}
    assert inherited == cli.INHERITED
    assert len(paths) == 11 + len(cli.INHERITED)
    assert all(Path(toy.spec["code_files"][role]["path"]).resolve() == path.resolve()
               for role, path in paths.items())


def test_example_is_explicit_full_population_unbound_template():
    value = json.loads((Path(__file__).parents[1] / "examples/r2_1_hoptimus_mosaic.json").read_bytes())
    assert set(value["sources"]) == cli.SOURCES
    assert set(value["code_files"]) == set(cli.code_paths(cli._reader()))
    assert [(head["held_out_fold"], head["budget_percent"]) for head in value["heads"]] == list(cli.TUPLES)
    assert value["expected"]["slides"] == 2604 and value["expected"]["original_tiles"] == 3635673
    assert "UNBOUND" in value["run_id"]
    assert all(Path(entry["path"]).is_absolute() and entry["size"] == 0 and entry["sha256"] == "0" * 64
               for collection in ("sources", "code_files") for entry in value[collection].values())


def prepare_written(toy):
    stage, _, _ = cli.authenticate(toy.spec, toy.sources())
    _, offsets = cli.reconcile(toy.spec, toy.sources(), stage)
    xy, arrays, cells, rows = cli.allocate(9, 3)
    for entry, row in zip(toy.features, offsets):
        raw, coordinates = cli.decode_feature(Path(entry["path"]).read_bytes(), entry)
        cli.score_slide(toy.head_objects, raw, coordinates, row, xy, arrays, cells, rows)
    output = tiny_output(toy)
    bindings = {}
    cli.producer._array_output(output, "coordinates.npy", xy, bindings)
    cli._csv_output(output, "slide_offsets.csv", cli.OFFSET_HEADER,
                    lambda: (tuple(row[name] for name in cli.OFFSET_HEADER) for row in offsets),
                    3, bindings)
    factory = lambda: cli.counts_rows(offsets, arrays)
    cli._csv_output(output, "mask_counts.csv.gz", cli.COUNT_HEADER, factory, 30, bindings, compressed=True)
    zero = sum(row[-1] == "zero_predicted_tumor" for row in factory())
    for key in cli.TUPLES:
        p, y, keep = arrays[key]
        cli.producer._array_output(output, cli._name(key) + "_probabilities.npy", p, bindings)
        cli.producer._array_output(output, cli._name(key) + "_predictions.npy", y, bindings)
        cli._bool_output(output, cli._name(key) + "_mask.npy", keep, bindings)
    return output, offsets, bindings, zero


def test_independent_written_array_csv_coverage_recount_without_reader_execution(toy):
    output, offsets, bindings, zero = prepare_written(toy)
    try:
        cli._written_recount(output, offsets, toy.spec["expected"], zero, bindings)
        assert zero >= 10
    finally:
        output.close()


@pytest.mark.parametrize("kind", ["probabilities", "predictions", "mask", "coordinates", "counts", "offsets"])
def test_independent_written_recount_rejects_semantic_corruption_even_rebound_hash(toy, kind):
    output, offsets, bindings, zero = prepare_written(toy)
    try:
        if kind in ("probabilities", "predictions", "mask", "coordinates"):
            name = "coordinates.npy" if kind == "coordinates" else "fold0_budget5_" + kind + ".npy"
            array = np.load(toy.out / name)
            if kind == "probabilities": array[0, 0] = np.nan
            elif kind == "predictions": array[0] = 5
            elif kind == "mask": array[0] = not array[0]
            else: array[1] = array[0]
            with (toy.out / name).open("wb") as handle:
                np.lib.format.write_array(handle, array, version=(1, 0), allow_pickle=False)
        else:
            name = "mask_counts.csv.gz" if kind == "counts" else "slide_offsets.csv"
            data = (toy.out / name).read_bytes()
            if kind == "counts":
                parsed = list(csv.reader(io.StringIO(gzip.decompress(data).decode())))
                parsed[1][13] = "999"
                (toy.out / name).write_bytes(gzip.compress(csv_bytes(parsed[0], parsed[1:]), mtime=0))
            else:
                parsed = list(csv.reader(io.StringIO(data.decode())))
                parsed[2][5] = "2"
                (toy.out / name).write_bytes(csv_bytes(parsed[0], parsed[1:]))
        bindings[name] = binding((toy.out / name).read_bytes())
        with pytest.raises(ValueError):
            cli._written_recount(output, offsets, toy.spec["expected"], zero, bindings)
    finally:
        output.close()


def test_prediction_batches_at_4096_and_last_partial(toy, monkeypatch):
    raw = np.zeros((4097, 1536), np.float32)
    raw[:, 1] = 1.0
    xy = np.column_stack((np.arange(4097, dtype=np.int64), np.zeros(4097, np.int64)))
    coordinates, arrays, cells, rows = cli.allocate(4097, 1)
    seen = []
    original = cli.fitter.predict_head
    def predict(head, batch):
        seen.append(len(batch))
        return original(head, batch)
    monkeypatch.setattr(cli.fitter, "predict_head", predict)
    cli.score_slide(toy.head_objects, raw, xy, dict(slide_index=0, tile_start=0, tile_stop=4097),
                    coordinates, arrays, cells, rows)
    assert seen == [4096, 1] * 10 and cells.all() and rows.sum() == 40970


def test_mutable_restored_state_rechecked_per_batch(toy, stable_reader, monkeypatch):
    original = cli.score_slide
    def corrupt(heads, *args, **kwargs):
        heads[0, 5].estimator.coef_[0, 0] += 1
        return original(heads, *args, **kwargs)
    monkeypatch.setattr(cli, "score_slide", corrupt)
    with pytest.raises(ValueError, match="fingerprint"):
        toy.run()
    assert (toy.out / "FAILED").exists() and not (toy.out / "COMPLETE.json").exists()


def test_custody_record_historical_keys_separate_from_current_paths_without_reader_execution(toy):
    declarations, policies = cli.declarations(toy.spec, toy.spec_path, toy.spec_path.read_bytes(), toy.digest)
    with cli.assets.AssetGuard(declarations, base_dir=toy.work, policies=policies) as guard:
        links = cli.producer_links(guard, {cli._key(head): head for head in toy.spec["heads"]})
        for key, link in zip(cli.TUPLES, links):
            assert link["complete_current_path"] == str(toy.heads[key]["complete"])
            assert link["complete_historical_registry_key"].startswith("/historical/")
            assert set(link["consumed_artifacts"]) == set(cli.REGISTRY_NAMES)
            assert set(link["referenced_not_reread"]) == \
                {"probabilities.npy", "predictions.npy", "counts.json", "held_out_rows.csv.gz"}
        assert len(guard.final_bindings()) == len(declarations)


@pytest.mark.parametrize("field,value", [
    ("patient_id", "unknown"), ("diagnostic_group", "NORMAL"), ("diagnostic_group", "MISC"),
    ("query_index", "1"), ("reference_index", "1"), ("metadata_index", "11"),
    ("fold", "9"), ("slide_id", "slide_Z"),
])
def test_retrieval_metadata_and_population_identity_negative(toy, field, value):
    path = toy.source_paths["retrieval_population"]
    parsed = list(csv.DictReader(io.StringIO(path.read_text())))
    parsed[0][field] = value
    path.write_bytes(csv_bytes(cli.manifests.RETRIEVAL_HEADER,
                               (tuple(row[name] for name in cli.manifests.RETRIEVAL_HEADER) for row in parsed)))
    toy.refresh_cpu()
    stage, _, _ = cli.authenticate(toy.spec, toy.sources())
    with pytest.raises(ValueError):
        cli.reconcile(toy.spec, toy.sources(), stage)


@pytest.mark.parametrize("field,value", [
    ("has_retrieval", "1"), ("has_annotations", "0"), ("has_retrieval", "True"),
    ("kernel_order", "0"), ("patient_hash", "a" * 64), ("fold", "5"),
])
def test_complete_patient_fold_flags_order_and_hash_negative(toy, field, value):
    path = toy.source_paths["patient_folds"]
    parsed = list(csv.DictReader(io.StringIO(path.read_text())))
    row = next(row for row in parsed if row["patient_id"] == "immune_A")
    row[field] = value
    if field == "kernel_order":
        row[field] = next(other[field] for other in parsed if other is not row)
    path.write_bytes(csv_bytes(cli.manifests.FOLD_HEADER,
                               (tuple(row[name] for name in cli.manifests.FOLD_HEADER) for row in parsed)))
    toy.refresh_cpu()
    stage, _, _ = cli.authenticate(toy.spec, toy.sources())
    with pytest.raises(ValueError):
        cli.reconcile(toy.spec, toy.sources(), stage)


def test_json_output_cap_is_not_relaxed_to_input_control_cap(toy):
    output = tiny_output(toy)
    try:
        with pytest.raises(ValueError, match="JSON limit"):
            cli.producer._json_output(output, "oversized.json", {"value": "x" * (8 * cli.MIB)}, {})
        assert not (toy.out / "oversized.json").exists()
    finally:
        output.close()


@pytest.mark.parametrize("kind", ["missing_cell", "missing_rows", "duplicate_rows", "wrong_dtype"])
def test_explicit_coverage_not_only_valid_sentinels(kind):
    cells = np.ones((10, 2), bool)
    processed = np.full((10, 2), 3, np.int64)
    offsets = [{"feature_rows": 3}, {"feature_rows": 3}]
    expected = {"slides": 2, "tile_head_predictions": 60}
    cli.validate_coverage(cells, processed, offsets, expected)
    if kind == "missing_cell": cells[0, 1] = False
    if kind == "missing_rows": processed[0, 1] = 2
    if kind == "duplicate_rows": processed[0, 1] = 6
    if kind == "wrong_dtype": cells = cells.astype(np.int8)
    with pytest.raises(ValueError, match="coverage"):
        cli.validate_coverage(cells, processed, offsets, expected)
