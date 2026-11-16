"""A-1 encoder head adapter: bound toy controls and tiny features only.

Includes the synthetic gate H-0 (eligible-tile row mapping) and a synthetic H-1
analogue: in h_optimus_1 mode the adapter must reproduce the frozen H-Optimus-1
head driver's numerical outputs byte for byte on identical toy inputs.
"""
from collections import Counter
import csv
from dataclasses import asdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from prosper.evaluation import r2_1_encoder_head_cli as cli
from prosper.evaluation import r2_1_hoptimus_head_cli as frozen

NATIVE = {name: "1" for name in cli.NATIVE_VARIABLES}
NUMERIC_OUTPUTS = ("head.pkl", "fit_diagnostics.json", "held_out_rows.csv.gz",
                   "probabilities.npy", "predictions.npy")


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


def bind(path):
    data = Path(path).read_bytes()
    return dict(path=str(path), sha256=sha(data), size=len(data))


class Toy:
    """Synthetic Stage1a/CPU controls; the selected bank's inventory rows are bound."""

    def __init__(self, tmp_path, monkeypatch, *, bank="uni2", fold=0, budget=5, bank_dir=None):
        for name, value in NATIVE.items():
            monkeypatch.setenv(name, value)
        self.bank_name = bank
        self.work = tmp_path / "toy"
        self.work.mkdir()
        self.controls = self.work / "controls"
        self.controls.mkdir()
        # Addendum A3: UNI2 annotation pickles must lie under pathologist_annotated_uni2_features/.
        self.bank = self.work / (bank_dir or cli.ANNOTATION_BANK_DIRS.get(bank, "features"))
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
        self.records = tuple(reversed(self.records))
        annotation_patients = {r.patient_id for r in self.records}
        self.source_paths = {role: self.controls / (role + ".json") for role in cli.SOURCE_LIMITS}
        self.source_paths["eligible_tiles"] = self.controls / "eligible_tiles.csv"
        rows = [(p, self.mapping[p], ordered.index(p), cli.kernel.patient_fold_hash(p),
                 int(p in annotation_patients), 1) for p in sorted(patients)]
        self.source_paths["patient_folds"].write_bytes(csv_bytes(cli.manifests.FOLD_HEADER, rows))
        self.source_paths["annotation_assignments"].write_bytes(csv_bytes(
            cli.manifests.ANNOTATION_HEADER,
            [(r.tile_id, r.slide_id, r.patient_id, r.diagnostic_group, r.level, r.x, r.y,
              r.patch_size, r.tissue_label, r.h_optimus_1_feature_row, self.mapping[r.patient_id])
             for r in self.records], True))
        self.write_eligible()
        inventory = []
        for entry in self.feature_entries:
            slide = entry["slide_id"]
            target = sum(r.slide_id == slide for r in self.records)
            for bank_name in ("h_optimus_1", "uni2", "virchow2"):
                selected = bank_name == bank
                # Decoy rows of the other banks deliberately carry a wrong hash/size; the
                # adapter must select only the requested bank.
                inventory.append((
                    bank_name, slide, 1536, 0, 1536, entry["feature_rows"], "float32",
                    target, entry["feature_rows"] - target, 0, 0, "false", 1,
                    entry["path"], str(Path(entry["path"]).resolve()),
                    entry["size"] if selected else entry["size"] + 1,
                    entry["sha256"] if selected else "0" * 64))
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
            code_files={role: bind(path) for role, path in cli.code_paths().items()},
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

    def write_eligible(self, edit=None):
        rows = []
        for r in sorted(self.records, key=lambda r: r.tile_id):
            # uni2_legacy_row/virchow2_legacy_row are deliberately non-numeric: the
            # adapter must never select or convert them (specification §12 item 14).
            row = [r.tile_id, r.slide_id, r.patient_id, r.diagnostic_group, r.level, r.x, r.y,
                   r.patch_size, r.tissue_label, "LEGACY_NOT_READ", "LEGACY_NOT_READ",
                   r.h_optimus_1_feature_row, r.h_optimus_1_feature_row,
                   r.h_optimus_1_feature_row]
            rows.append(row)
        if edit:
            edit(rows)
        self.source_paths["eligible_tiles"].write_bytes(
            csv_bytes(cli.manifests.ELIGIBLE_HEADER, rows))

    def refresh(self):
        old_digest = sha(b"toy historical CPU declaration")
        entries = [dict(kind="spec", name="spec", sha256=old_digest, size=30)]
        entries.append(dict(kind="source", name="stage1a_feature_inventory",
                            **bind(self.source_paths["feature_inventory"])))
        entries.append(dict(kind="source", name="stage1a_eligible",
                            **bind(self.source_paths["eligible_tiles"])))
        for role, old in (("filter_kernel", "kernel"), ("filter_cli", "runner"),
                          ("cpu_io", "io"), ("legacy_io", "legacy_io"),
                          ("package_init", "package_init"), ("evaluation_init", "evaluation_init")):
            entries.append(dict(kind="code", name=old, **self.spec["code_files"][role]))
        self.source_paths["cpu_input_bindings"].write_bytes(jbytes(dict(
            schema_version=1, run_id=self.cpu_run, task=cli.manifests.TASK,
            spec_sha256=old_digest, bindings=entries)))
        outputs = {leaf: {key: bind(self.source_paths[role])[key] for key in ("sha256", "size")}
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
        self.spec["sources"] = {role: bind(path) for role, path in self.source_paths.items()}
        self.spec_path.write_bytes(jbytes(self.spec))
        self.spec_digest = sha(self.spec_path.read_bytes())

    def run(self):
        return cli.run(self.spec_path, self.spec_digest, self.out, self.root, bank=self.bank_name)

    def frozen_run(self, out):
        """Same toy inputs through the byte-unchanged frozen H-Optimus-1 head driver."""
        spec = json.loads(self.spec_path.read_bytes())
        spec["task"] = frozen.TASK
        spec["sources"] = {role: spec["sources"][role] for role in frozen.SOURCE_LIMITS}
        spec["code_files"] = {role: bind(path) for role, path in frozen.code_paths().items()}
        path = self.work / "frozen_spec.json"
        path.write_bytes(jbytes(spec))
        return frozen.run(path, sha(path.read_bytes()), out, self.root)

    def command(self, bank=None):
        return [sys.executable, "-m", "prosper.evaluation.r2_1_encoder_head_cli",
                "--spec", str(self.spec_path), "--spec-sha256", self.spec_digest,
                "--out", str(self.out), "--out-root", str(self.root),
                "--bank", bank or self.bank_name]


def check_outputs(toy):
    marker = json.loads((toy.out / "COMPLETE.json").read_bytes())
    assert set(marker["outputs"]) == cli.ARTIFACTS
    assert set(p.name for p in toy.out.iterdir()) == cli.ARTIFACTS | {"COMPLETE.json"}
    for name, binding in marker["outputs"].items():
        data = (toy.out / name).read_bytes()
        assert binding == {"size": len(data), "sha256": sha(data)}
    assert marker["encoder"] == marker["bank"] == toy.bank_name
    p = np.load(toy.out / "probabilities.npy", allow_pickle=False)
    y = np.load(toy.out / "predictions.npy", allow_pickle=False)
    assert np.array_equal(y, np.array([1, 2, 3, 4, 5])[p.argmax(axis=1)])
    restored = cli.legacy._pickle((toy.out / "head.pkl").read_bytes(), "toy readback")
    held = np.stack([toy.raw[r.slide_id][0][r.h_optimus_1_feature_row] for r in toy.held])
    batch = cli.fitter.predict_head(restored, held)
    assert batch.probabilities.tobytes() == p.tobytes() and batch.labels.tobytes() == y.tobytes()
    for name in ("summary.json", "counts.json", "input_bindings.json"):
        record = json.loads((toy.out / name).read_bytes())
        assert record["encoder"] == record["bank"] == toy.bank_name, name
    summary = json.loads((toy.out / "summary.json").read_bytes())
    assert summary["policy"]["recipe"] == "r2_1_fixed_sgd_ovr_v1"
    assert summary["policy"]["prediction_batch_rows"] == 4096
    assert summary["policy"]["decision"] == "probability_argmax_class1"
    bindings = json.loads((toy.out / "input_bindings.json").read_bytes())
    assert bindings["h0_row_identity"] == {
        "source_role": "eligible_tiles", "bank": toy.bank_name,
        "column": toy.bank_name + "_feature_row", "reference_column": "h_optimus_1_feature_row",
        "tiles_checked": len(toy.records), "held_out_tiles_checked": len(toy.held),
        "all_equal": True}
    reads = bindings["annotation_feature_reads"]
    assert reads and [r["slide_id"] for r in reads] == sorted({r["slide_id"] for r in reads})
    for read in reads:
        assert Path(read["path"]).read_bytes() and sha(Path(read["path"]).read_bytes()) == read["sha256"]
        if toy.bank_name == "uni2":
            assert Path(read["path"]).parent.name == "pathologist_annotated_uni2_features"
    assert Path(bindings["executing_code_paths"]["runner"]) == Path(cli.__file__).resolve()
    assert bindings["environment"]["native_environment"] == NATIVE
    assert not (toy.out / "FAILED").exists()
    return marker


def failed(toy, match):
    with pytest.raises(cli.HeadTaskError, match=match):
        toy.run()


def test_help_requires_bank():
    result = subprocess.run(
        [sys.executable, "-m", "prosper.evaluation.r2_1_encoder_head_cli", "--help"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "--bank" in result.stdout and "{h_optimus_1,uni2}" in result.stdout
    assert cli.main.__module__ == cli.__name__


def test_unknown_bank_rejected_by_parser_and_api(tmp_path, monkeypatch):
    toy = Toy(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        cli.main(["--spec", str(toy.spec_path), "--spec-sha256", toy.spec_digest,
                  "--out", str(toy.out), "--out-root", str(toy.root), "--bank", "virchow2"])
    with pytest.raises(cli.HeadTaskError, match="unsupported bank"):
        cli.run(toy.spec_path, toy.spec_digest, toy.out, toy.root, bank="virchow2")
    assert not toy.out.exists()


def test_uni2_normal_cli_success(tmp_path, monkeypatch):
    toy = Toy(tmp_path, monkeypatch, bank="uni2")
    result = subprocess.run(toy.command(), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    check_outputs(toy)


@pytest.mark.parametrize("fold,budget", [(0, 5), (2, 100)])
def test_h1_synthetic_h_optimus_mode_matches_frozen_driver_bytes(tmp_path, monkeypatch, fold, budget):
    toy = Toy(tmp_path, monkeypatch, bank="h_optimus_1", fold=fold, budget=budget)
    toy.run()
    check_outputs(toy)
    reference = toy.root / "frozen_attempt"
    toy.frozen_run(reference)
    for name in NUMERIC_OUTPUTS:
        assert (toy.out / name).read_bytes() == (reference / name).read_bytes(), name
    ours = json.loads((toy.out / "summary.json").read_bytes())
    theirs = json.loads((reference / "summary.json").read_bytes())
    assert ours["policy"] == theirs["policy"]
    assert ours["counts"] == theirs["counts"]
    ours_counts = json.loads((toy.out / "counts.json").read_bytes())
    theirs_counts = json.loads((reference / "counts.json").read_bytes())
    for key in ("tumour_counts", "tissue_confusion", "held_out_support"):
        assert ours_counts[key] == theirs_counts[key]


def test_uni2_and_h_optimus_modes_identical_numerics_on_identical_features(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = Toy(tmp_path / "a", monkeypatch, bank="uni2")
    a.run()
    b = Toy(tmp_path / "b", monkeypatch, bank="h_optimus_1")
    b.run()
    for name in ("probabilities.npy", "predictions.npy", "held_out_rows.csv.gz"):
        assert (a.out / name).read_bytes() == (b.out / name).read_bytes(), name


def test_h0_mapped_row_mismatch_fails_before_fit(tmp_path, monkeypatch):
    toy = Toy(tmp_path, monkeypatch, bank="uni2")
    column = cli.manifests.ELIGIBLE_HEADER.index("uni2_feature_row")
    target = next(r.tile_id for r in toy.held)

    def edit(rows):
        for row in rows:
            if row[0] == target:
                row[column] = str(int(row[column]) + 1)

    toy.write_eligible(edit)
    toy.refresh()
    calls = []
    monkeypatch.setattr(cli.fitter, "fit_head", lambda *a: calls.append(a))
    failed(toy, "H-0: uni2_feature_row != h_optimus_1_feature_row")
    assert not calls and not (toy.out / "head.pkl").exists()


def test_h0_is_bank_specific(tmp_path, monkeypatch):
    """A UNI2-only discrepancy is irrelevant in h_optimus_1 mode (identity mapping)."""
    toy = Toy(tmp_path, monkeypatch, bank="h_optimus_1")
    column = cli.manifests.ELIGIBLE_HEADER.index("uni2_feature_row")

    def edit(rows):
        rows[0][column] = "999"

    toy.write_eligible(edit)
    toy.refresh()
    toy.run()
    check_outputs(toy)


@pytest.mark.parametrize("change,match", [
    (lambda rows: rows.pop(), "eligible/annotation tile set mismatch"),
    (lambda rows: rows.append(list(rows[0])), "duplicate eligible tile"),
    (lambda rows: rows[0].__setitem__(8, "2" if rows[0][8] != "2" else "3"),
     "eligible/annotation tile identity mismatch"),
    (lambda rows: rows[0].__setitem__(13, str(int(rows[0][13]) + 1)),
     "eligible/annotation h_optimus_1_feature_row mismatch"),
    (lambda rows: rows[0].pop(), "eligible tiles ragged row"),
])
def test_eligible_identity_negatives(tmp_path, monkeypatch, change, match):
    toy = Toy(tmp_path, monkeypatch, bank="uni2")
    toy.write_eligible(change)
    toy.refresh()
    failed(toy, match)


def test_eligible_header_must_match(tmp_path, monkeypatch):
    toy = Toy(tmp_path, monkeypatch)
    data = toy.source_paths["eligible_tiles"].read_bytes().split(b"\n", 1)
    toy.source_paths["eligible_tiles"].write_bytes(data[0].replace(b"tissue_label", b"label") +
                                                   b"\n" + data[1])
    toy.refresh()
    failed(toy, "eligible tiles header mismatch")


def test_eligible_must_match_cpu_stage1a_binding(tmp_path, monkeypatch):
    toy = Toy(tmp_path, monkeypatch)
    toy.write_eligible(lambda rows: rows.reverse())
    toy.rebind()  # spec re-bound but CPU Stage1a binding still names the old bytes
    failed(toy, "eligible tiles (size|hash)")


def test_selected_bank_rows_are_used_not_decoys(tmp_path, monkeypatch):
    toy = Toy(tmp_path, monkeypatch, bank="uni2")
    inventory = toy.source_paths["feature_inventory"]
    rows = list(csv.reader(io.StringIO(inventory.read_text(), newline="")))
    header, body = rows[0], [row for row in rows[1:] if row[0] != "uni2"]
    inventory.write_bytes(csv_bytes(header, body))
    toy.refresh()
    failed(toy, "undeclared|missing uni2 inventory slides")


def test_width_is_kernel_width_not_alternative(tmp_path, monkeypatch):
    toy = Toy(tmp_path, monkeypatch)
    toy.spec["features"][0]["feature_dim"] = 1024
    toy.rebind()
    failed(toy, "fit-kernel width")


def test_decoded_width_mismatch_fails_after_claim(tmp_path, monkeypatch):
    toy = Toy(tmp_path, monkeypatch)
    slide = toy.held[0].slide_id
    entry = next(e for e in toy.feature_entries if e["slide_id"] == slide)
    raw, coords = toy.raw[slide]
    payload = pickle.dumps((np.zeros((len(raw), 1024), np.float32),
                            [{"wsi_loc": list(xy)} for xy in coords]), protocol=4)
    Path(entry["path"]).write_bytes(payload)
    entry.update(sha256=sha(payload), size=len(payload))
    inventory = toy.source_paths["feature_inventory"]
    rows = list(csv.reader(io.StringIO(inventory.read_text(), newline="")))
    for row in rows[1:]:
        if row[0] == "uni2" and row[1] == slide:
            row[-2], row[-1] = str(len(payload)), sha(payload)
    inventory.write_bytes(csv_bytes(rows[0], rows[1:]))
    toy.refresh()
    failed(toy, "feature width 1024 != fit-kernel width 1536")
    assert (toy.out / "FAILED").exists() and not (toy.out / "COMPLETE.json").exists()


def test_code_identity_names_adapter_file(tmp_path, monkeypatch):
    paths = cli.code_paths()
    assert paths["runner"].name == "r2_1_encoder_head_cli.py"
    toy = Toy(tmp_path, monkeypatch)
    toy.spec["code_files"]["runner"] = bind(frozen.code_paths()["runner"])
    toy.rebind()
    failed(toy, "runner: declaration is not the actual executing file")


def test_frozen_parent_is_byte_unchanged():
    assert sha(Path(frozen.__file__).read_bytes()) == \
        "7c0b5ca726e8be8a57be58197eb28e5a6048434ef10665ff020800af6c1538ef"


def test_a3_uni2_annotation_pickles_outside_uni2_root_rejected(tmp_path, monkeypatch):
    toy = Toy(tmp_path, monkeypatch, bank="uni2", bank_dir="h_optimus_1_all_patches")
    failed(toy, "A3: .* is not under pathologist_annotated_uni2_features/")
    assert (toy.out / "FAILED").exists() and not (toy.out / "COMPLETE.json").exists()


def test_a3_routing_unit_checks(tmp_path):
    root = tmp_path / "pathologist_annotated_uni2_features"
    root.mkdir()
    path = root / "s.pkl"
    path.write_bytes(b"payload")
    row = {"bank": "uni2", "sha256": sha(b"payload"), "resolved_path": str(path)}
    assert cli._routed("s", row, b"payload", "uni2", path) == {
        "slide_id": "s", "path": str(path.resolve()), "sha256": sha(b"payload")}
    with pytest.raises(cli.HeadTaskError, match="A3: s inventory row bank 'h_optimus_1'"):
        cli._routed("s", dict(row, bank="h_optimus_1"), b"payload", "uni2", path)
    with pytest.raises(cli.HeadTaskError, match="A3: s sha256 of bytes read"):
        cli._routed("s", row, b"other bytes", "uni2", path)
    other = tmp_path / "h_optimus_1_all_patches"
    other.mkdir()
    (other / "s.pkl").write_bytes(b"payload")
    with pytest.raises(cli.HeadTaskError, match="not under pathologist_annotated_uni2_features"):
        cli._routed("s", dict(row, resolved_path=str(other / "s.pkl")), b"payload", "uni2", path)
    # H-Optimus-1 mode records the read but carries no UNI2 root assertion.
    assert cli._routed("s", dict(row, bank="h_optimus_1", resolved_path=str(other / "s.pkl")),
                       b"payload", "h_optimus_1", other / "s.pkl")["sha256"] == sha(b"payload")
