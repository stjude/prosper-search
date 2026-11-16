"""A-2 route-(b) mosaic adapter: registry head source, --bank Stage0 selection, M-1; synthetic only."""
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from prosper.evaluation import r2_1_encoder_head_registry as registry
from prosper.evaluation import r2_1_encoder_mosaic_cli as enc
from prosper.evaluation import r2_1_hoptimus_mosaic_cli as frozen
from tests import test_r2_1_hoptimus_mosaic_cli as base
from tests.test_r2_1_hoptimus_mosaic_cli import no_fit  # noqa: F401  (autouse: no fitting here)

sha, encode, binding, descriptor, csv_bytes = (
    base.sha, base.encode, base.binding, base.descriptor, base.csv_bytes)
COMPARED = ("coordinates.npy", "mask_counts.csv.gz", "slide_offsets.csv", "retrieval_population.csv") + tuple(
    f"{frozen._name(key)}_{kind}.npy" for key in frozen.TUPLES for kind in ("mask", "predictions", "probabilities"))


def _rewrite_json(path, edit):
    value = json.loads(path.read_bytes())
    edit(value)
    path.write_bytes(encode(value))


class EncToy(base.Toy):
    """The frozen fixture with real producer-spec paths plus leaf-named head directories."""

    def __init__(self, tmp_path, monkeypatch, *, bank="h_optimus_1"):
        super().__init__(tmp_path, monkeypatch)
        self.bank_name = bank
        for head in self.spec["heads"]:
            key = frozen._key(head)
            paths = self.heads[key]
            spec_path = str(paths["spec"])
            _rewrite_json(paths["input_bindings"], lambda v: v["bindings"][0].update(
                path=spec_path, resolved_path=spec_path))
            head["producer_spec_path"] = spec_path
            self._rebind_complete(paths)
        if bank == "uni2":
            self._swap_stage0("uni2", "h_optimus_1")
            monkeypatch.setitem(enc.M1_FEATURE_DIRS, "uni2", str(self.bank))
        self.refresh_head_acceptance()
        self.refresh_cpu()
        self.head_dirs = {}
        for head in self.spec["heads"]:
            key = frozen._key(head)
            folder = self.work / "heads" / frozen._name(key)
            folder.mkdir(parents=True)
            for role, leaf in registry.LEAVES.items():
                shutil.copyfile(self.heads[key][role], folder / leaf)
            if bank == "uni2":
                self._declare_bank(folder, "uni2")
            self.head_dirs[key] = folder
        self.registry_path = self.work / "HEAD_REGISTRY.json"
        self.accepted_path = None
        if bank == "uni2":
            self.accepted_path = self.work / "accepted_heads.json"
            self.accepted_path.write_bytes(encode(self.accepted_document()))
        self.enc_out = self.root / "encoder_attempt"
        self.enc_spec_path = self.work / "encoder_spec.json"

    @staticmethod
    def _rebind_complete(paths, leaves=None):
        leaves = leaves or paths
        roles = {"head.pkl": "model", "fit_diagnostics.json": "diagnostics",
                 "input_bindings.json": "input_bindings", "summary.json": "summary"}
        _rewrite_json(leaves["complete"], lambda v: v["outputs"].update(
            {leaf: binding(Path(leaves[role]).read_bytes()) for leaf, role in roles.items()}))

    def _declare_bank(self, folder, bank):
        leaves = {role: folder / leaf for role, leaf in registry.LEAVES.items()}
        for role in ("summary", "input_bindings"):
            _rewrite_json(leaves[role], lambda v: v.update(encoder=bank, bank=bank))
        self._rebind_complete(None, leaves)
        _rewrite_json(leaves["complete"], lambda v: v.update(encoder=bank, bank=bank))

    def _swap_stage0(self, selected, other):
        """Make the selected bank's Stage0 rows/bindings/declaration point at the toy features."""
        inventory = self.source_paths["stage0_inventory"]
        rows = list(csv.reader(io.StringIO(inventory.read_text(), newline="")))
        info = {}
        for row in rows[1:]:
            info[row[0], row[1]] = row[5:]
        for row in rows[1:]:
            if row[0] in (selected, other):
                row[5:] = info[other if row[0] == selected else selected, row[1]]
        inventory.write_bytes(csv_bytes(rows[0], rows[1:]))

        def summary_edit(value):
            entries = {(e["kind"], e["name"]): dict(e) for e in value["input_bindings"]}
            for entry in value["input_bindings"]:
                if entry["kind"] == "feature":
                    bank, slide = entry["name"].split("/", 1)
                    if bank in (selected, other):
                        source = entries["feature", (other if bank == selected else selected) + "/" + slide]
                        entry.update({k: source[k] for k in ("path", "resolved_path", "size", "sha256")})
            paths = {b["name"]: b["path"] for b in value["declarations"]["banks"]}
            for b in value["declarations"]["banks"]:
                if b["name"] in (selected, other):
                    b["path"] = paths[other if b["name"] == selected else selected]
        _rewrite_json(self.source_paths["stage0_summary"], summary_edit)
        self._rebind_stage0()

    def _rebind_stage0(self):
        inventory = self.source_paths["stage0_inventory"]
        _rewrite_json(self.source_paths["stage0_complete"], lambda v: v["outputs"].update({
            "inventory.csv": binding(inventory.read_bytes()),
            "summary.json": binding(self.source_paths["stage0_summary"].read_bytes())}))
        _rewrite_json(self.source_paths["stage0_review"], lambda v: v["artifacts"].update({
            f"stage0_attempt_001/{leaf}": sha(self.source_paths[role].read_bytes()) for leaf, role in (
                ("COMPLETE.json", "stage0_complete"), ("inventory.csv", "stage0_inventory"),
                ("summary.json", "stage0_summary"))}))

    def accepted_document(self, override=None):
        heads = {frozen._name(key): {"head.pkl_sha256": sha((folder / "head.pkl").read_bytes())}
                 for key, folder in self.head_dirs.items()}
        heads.update(override or {})
        return {"schema_version": 1, "task": registry.ACCEPTED_TASK, "encoder": self.bank_name,
                "bank": self.bank_name, "heads": heads}

    def build_registry(self, **kwargs):
        kwargs.setdefault("accepted_path", self.accepted_path)
        return registry.build_registry(
            self.bank_name, [self.head_dirs[key] for key in frozen.TUPLES],
            self.source_paths["head_plan"], **kwargs)

    def write_registry(self, value=None):
        self.registry_path.write_bytes(encode(value if value is not None else self.build_registry()))

    def write_enc_spec(self, edit=None):
        spec = json.loads(encode(self.spec))
        spec["sources"] = {role: descriptor(path) for role, path in self.source_paths.items()
                           if role in enc.SOURCES}
        spec["sources"]["head_registry"] = descriptor(self.registry_path)
        if self.accepted_path is not None:
            spec["sources"]["accepted_heads"] = descriptor(self.accepted_path)
        spec["code_files"] = {role: descriptor(path) for role, path in enc.code_paths(enc._reader()).items()}
        spec["heads"] = []
        for key in frozen.TUPLES:
            folder = self.head_dirs[key]
            assets = {role: descriptor(folder / leaf) for role, leaf in registry.LEAVES.items()}
            assets["spec"] = descriptor(self.heads[key]["spec"])
            spec["heads"].append({"held_out_fold": key[0], "budget_percent": key[1], "assets": assets})
        if edit:
            edit(spec)
        self.enc_spec_path.write_bytes(encode(spec))
        self.enc_digest = sha(self.enc_spec_path.read_bytes())
        return spec

    def enc_run(self, bank=None):
        return enc.run(self.enc_spec_path, self.enc_digest, self.enc_out, self.root,
                       bank=bank or self.bank_name)

    def enc_command(self):
        return [sys.executable, "-m", "prosper.evaluation.r2_1_encoder_mosaic_cli",
                "--spec", str(self.enc_spec_path), "--spec-sha256", self.enc_digest,
                "--out", str(self.enc_out), "--out-root", str(self.root), "--bank", self.bank_name]


def prepared(tmp_path, monkeypatch, bank="h_optimus_1"):
    toy = EncToy(tmp_path, monkeypatch, bank=bank)
    toy.write_registry()
    toy.write_enc_spec()
    return toy


def enc_failed(toy, match, bank=None):
    with pytest.raises((enc.MosaicError, registry.RegistryError, ValueError), match=match):
        toy.enc_run(bank)
    assert not (toy.enc_out / "COMPLETE.json").exists()


def check_enc_outputs(toy):
    marker = json.loads((toy.enc_out / "COMPLETE.json").read_bytes())
    assert set(marker["outputs"]) == enc.artifact_names() and len(marker["outputs"]) == 36
    for name, expected in marker["outputs"].items():
        assert binding((toy.enc_out / name).read_bytes()) == expected
    assert marker["encoder"] == marker["bank"] == toy.bank_name
    summary = json.loads((toy.enc_out / "summary.json").read_bytes())
    records = json.loads((toy.enc_out / "input_bindings.json").read_bytes())
    for record in (summary, records):
        assert record["encoder"] == record["bank"] == toy.bank_name
    assert summary["details"]["feature_columns"] == 1536
    assert summary["details"]["consumed_source_roles"] == sorted(enc.sources_for(toy.bank_name))
    assert [p["accepted_head_sha256"] for p in summary["details"]["head_provenance"]] == [
        sha((toy.head_dirs[key] / "head.pkl").read_bytes()) for key in frozen.TUPLES]
    m1 = records["m1_stage0_binding"]
    assert m1["slides"] == m1["decoded_bytes_sha256_equal_offsets_and_stage0"] == 3
    offsets = list(csv.DictReader(io.StringIO((toy.enc_out / "slide_offsets.csv").read_text())))
    for row in offsets:
        assert row["feature_sha256"] == sha((toy.bank / (row["slide_id"] + ".pkl")).read_bytes())
    assert len(records["producer_artifact_links"]) == 10
    assert Path(records["executing_code_paths"]["runner"]) == Path(enc.__file__).resolve()
    assert Path(records["executing_code_paths"]["head_registry"]) == Path(registry.__file__).resolve()
    return marker


# ----------------------------------------------------------------------------- registry writer


def test_registry_writer_h_optimus_mode_and_parse(tmp_path, monkeypatch):
    toy = EncToy(tmp_path, monkeypatch)
    value = toy.build_registry()
    assert value["bank"] == value["encoder"] == "h_optimus_1"
    assert set(value["heads"]) == {frozen._name(key) for key in frozen.TUPLES}
    for key in frozen.TUPLES:
        entry = value["heads"][frozen._name(key)]
        assert (entry["held_out_fold"], entry["budget_percent"]) == key
        complete = json.loads((toy.head_dirs[key] / "COMPLETE.json").read_bytes())
        assert entry["accepted_head_sha256"] == complete["outputs"]["head.pkl"]["sha256"]
        assert entry["artifacts"]["model"]["sha256"] == entry["accepted_head_sha256"]
        assert entry["head_bank_identity"] == "pre_adapter_h_optimus_1_reference_head"
    assert value["accepted_hash_source"]["kind"] == "reference_head_complete_json_output_binding"
    heads = registry.parse_registry(encode(value), bank="h_optimus_1",
                                    head_plan_data=toy.source_paths["head_plan"].read_bytes())
    assert set(heads) == set(value["heads"]) and len(heads) == 10
    with pytest.raises(registry.RegistryError, match="head registry bank is not uni2"):
        registry.parse_registry(encode(value), bank="uni2",
                                head_plan_data=toy.source_paths["head_plan"].read_bytes())
    with pytest.raises(registry.RegistryError, match="head_plan binding mismatch"):
        registry.parse_registry(encode(value), bank="h_optimus_1", head_plan_data=b"{}")


def test_registry_cli_writes_once_and_refuses_overwrite(tmp_path, monkeypatch):
    toy = EncToy(tmp_path, monkeypatch)
    argv = ["--bank", "h_optimus_1", "--head-plan", str(toy.source_paths["head_plan"]),
            "--out", str(toy.registry_path)]
    for key in frozen.TUPLES:
        argv += ["--head-dir", str(toy.head_dirs[key])]
    assert registry.main(argv) == 0
    assert json.loads(toy.registry_path.read_bytes())["task"] == registry.TASK
    with pytest.raises(registry.RegistryError, match="refusing to overwrite"):
        registry.main(argv)


def test_registry_acceptance_registry_cross_check(tmp_path, monkeypatch):
    toy = EncToy(tmp_path, monkeypatch)
    heads, hashes = {}, {}
    for key in frozen.TUPLES:
        folder = toy.head_dirs[key]
        heads[frozen._name(key)] = {"complete_sha256": sha((folder / "COMPLETE.json").read_bytes())}
        hashes[str(folder / "head.pkl")] = sha((folder / "head.pkl").read_bytes())
    path = toy.work / "reconciliation.json"
    path.write_bytes(encode({"heads": heads, "output_hashes": hashes}))
    value = toy.build_registry(acceptance_registry_path=path)
    assert value["accepted_hash_source"]["acceptance_registry"]["sha256"] == sha(path.read_bytes())
    hashes[str(toy.head_dirs[(2, 5)] / "head.pkl")] = "0" * 64
    path.write_bytes(encode({"heads": heads, "output_hashes": hashes}))
    with pytest.raises(registry.RegistryError, match="fold2_budget5: acceptance registry head.pkl"):
        toy.build_registry(acceptance_registry_path=path)


def test_registry_assertion_wrong_tuple_rejected(tmp_path, monkeypatch):
    toy = EncToy(tmp_path, monkeypatch)
    # Head directory whose summary declares (0, 5) but whose spec/plan/bindings are (0, 100).
    folder = toy.head_dirs[(0, 100)]
    _rewrite_json(folder / "summary.json", lambda v: v["policy"].update(budget_percent=5))
    toy._rebind_complete(None, {role: folder / leaf for role, leaf in registry.LEAVES.items()})
    with pytest.raises(registry.RegistryError, match="fold0_budget5|duplicate head tuple"):
        toy.build_registry()


def test_registry_assertion_head_sha_must_equal_complete_binding(tmp_path, monkeypatch):
    toy = EncToy(tmp_path, monkeypatch)
    folder = toy.head_dirs[(3, 100)]
    _rewrite_json(folder / "COMPLETE.json", lambda v: v["outputs"]["head.pkl"].update(sha256="1" * 64))
    with pytest.raises(registry.RegistryError, match="head.pkl sha256 .* != accepted 1111"):
        toy.build_registry()


def test_registry_uni2_requires_g5_accepted_and_bank_declaration(tmp_path, monkeypatch):
    toy = EncToy(tmp_path, monkeypatch, bank="uni2")
    value = toy.build_registry()
    assert value["accepted_hash_source"]["kind"] == "g5_accepted_heads_json"
    assert all(entry["head_bank_identity"] == "declared_by_head" for entry in value["heads"].values())
    with pytest.raises(registry.RegistryError, match="UNI2 mode requires --accepted-heads"):
        toy.build_registry(accepted_path=None)
    toy.accepted_path.write_bytes(encode(toy.accepted_document({"fold4_budget5": {"head.pkl_sha256": "2" * 64}})))
    with pytest.raises(registry.RegistryError, match="!= accepted 2222"):
        toy.build_registry()
    toy.accepted_path.write_bytes(encode(toy.accepted_document()))
    # A head that declares h_optimus_1 (or nothing) cannot enter a UNI2 registry.
    folder = toy.head_dirs[(1, 5)]
    for role in ("summary.json", "input_bindings.json", "COMPLETE.json"):
        _rewrite_json(folder / role, lambda v: v.update(encoder="h_optimus_1", bank="h_optimus_1"))
    toy._rebind_complete(None, {role: folder / leaf for role, leaf in registry.LEAVES.items()})
    with pytest.raises(registry.RegistryError, match="head encoder/bank is not uni2"):
        toy.build_registry()


def test_registry_uni2_rejects_pre_adapter_heads(tmp_path, monkeypatch):
    toy = EncToy(tmp_path, monkeypatch, bank="uni2")
    folder = toy.head_dirs[(2, 100)]
    for role in ("summary.json", "input_bindings.json", "COMPLETE.json"):
        _rewrite_json(folder / role, lambda v: (v.pop("encoder"), v.pop("bank")))
    toy._rebind_complete(None, {role: folder / leaf for role, leaf in registry.LEAVES.items()})
    with pytest.raises(registry.RegistryError, match="only pre-adapter H-Optimus-1 reference heads"):
        toy.build_registry()


# ----------------------------------------------------------------------------- mosaic CLI


def test_help_requires_bank():
    result = subprocess.run([sys.executable, "-m", "prosper.evaluation.r2_1_encoder_mosaic_cli", "--help"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "--bank" in result.stdout and "{h_optimus_1,uni2}" in result.stdout


def test_m0_synthetic_h_optimus_registry_path_byte_identical_to_frozen(tmp_path, monkeypatch):
    """Synthetic M-0: frozen reader chain versus the registry head source, same features/heads."""
    toy = prepared(tmp_path, monkeypatch)
    frozen_marker = toy.run()
    assert frozen_marker["task"] == frozen.TASK
    result = subprocess.run(toy.enc_command(), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    check_enc_outputs(toy)
    for name in COMPARED:
        assert (toy.out / name).read_bytes() == (toy.enc_out / name).read_bytes(), name
    old = json.loads((toy.out / "COMPLETE.json").read_bytes())["outputs"]
    new = json.loads((toy.enc_out / "COMPLETE.json").read_bytes())["outputs"]
    assert {k: v for k, v in old.items() if k in COMPARED} == {k: v for k, v in new.items() if k in COMPARED}


def test_uni2_mode_selects_uni2_rows_and_records_m1(tmp_path, monkeypatch):
    toy = prepared(tmp_path, monkeypatch, bank="uni2")
    toy.enc_run()
    marker = check_enc_outputs(toy)
    records = json.loads((toy.enc_out / "input_bindings.json").read_bytes())
    assert records["m1_stage0_binding"]["required_directory"] == str(toy.bank)
    assert marker["task"] == enc.TASK
    p = np.load(toy.enc_out / "fold0_budget5_probabilities.npy", allow_pickle=False)
    assert p.shape == (9, 5)


def test_uni2_mode_rejects_h_optimus_bank_mode_mismatch(tmp_path, monkeypatch):
    toy = prepared(tmp_path, monkeypatch, bank="uni2")
    enc_failed(toy, "mosaic source roles", bank="h_optimus_1")


def test_m1_bank_directory_must_be_comet_uni2_tmp_final(tmp_path, monkeypatch):
    toy = prepared(tmp_path, monkeypatch, bank="uni2")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setitem(enc.M1_FEATURE_DIRS, "uni2", str(other))
    enc_failed(toy, "M-1: Stage0 uni2 bank directory is not")


def test_m1_feature_path_must_be_bank_dir_slide_pkl(tmp_path, monkeypatch):
    toy = prepared(tmp_path, monkeypatch, bank="uni2")
    elsewhere = tmp_path / "copy"
    elsewhere.mkdir()
    feature = toy.features[0]
    target = elsewhere / Path(feature["path"]).name
    shutil.copyfile(feature["path"], target)
    inventory = toy.source_paths["stage0_inventory"]
    rows = list(csv.reader(io.StringIO(inventory.read_text(), newline="")))
    for row in rows[1:]:
        if row[0] == "uni2" and row[1] == feature["slide_id"]:
            row[5], row[6] = str(target), str(target)
    inventory.write_bytes(csv_bytes(rows[0], rows[1:]))

    def moved(value):
        for entry in value["input_bindings"]:
            if entry["kind"] == "feature" and entry["name"] == "uni2/" + feature["slide_id"]:
                entry.update(path=str(target), resolved_path=str(target))
    _rewrite_json(toy.source_paths["stage0_summary"], moved)
    toy._rebind_stage0()
    toy.refresh_cpu()
    toy.write_enc_spec(lambda spec: spec["features"][0].update(path=str(target)))
    enc_failed(toy, "M-1: slide_\\w+ feature_path does not resolve to")


def test_m1_decoded_bytes_hash_is_checked(tmp_path, monkeypatch):
    toy = prepared(tmp_path, monkeypatch, bank="uni2")
    original = enc.assets.AssetGuard.read_verified

    def substituted(self, asset_id):
        data = original(self, asset_id)
        # Pickles end in the STOP opcode b"."; append a byte so the decoded bytes truly differ.
        return data if asset_id != "feature:slide_C" else data + b"\x00"
    monkeypatch.setattr(enc.assets.AssetGuard, "read_verified", substituted)
    enc_failed(toy, "M-1: decoded slide_C bytes != slide_offsets/Stage0 sha256")
    assert (toy.enc_out / "FAILED").exists()


def test_registry_artifacts_must_equal_spec_head_assets(tmp_path, monkeypatch):
    toy = prepared(tmp_path, monkeypatch)
    value = json.loads(toy.registry_path.read_bytes())
    value["heads"]["fold4_budget100"]["artifacts"]["summary"]["path"] = "/elsewhere/summary.json"
    toy.registry_path.write_bytes(encode(value))
    toy.write_enc_spec()
    enc_failed(toy, "fold4_budget100 summary: spec head asset differs from HEAD_REGISTRY.json")


def test_registry_accepted_value_tamper_rejected_in_job(tmp_path, monkeypatch):
    toy = prepared(tmp_path, monkeypatch, bank="uni2")
    toy.accepted_path.write_bytes(encode(toy.accepted_document({"fold0_budget5": {"head.pkl_sha256": "3" * 64}})))
    toy.write_enc_spec()
    enc_failed(toy, "accepted-heads binding mismatch")


def test_width_read_from_input_not_alternative(tmp_path, monkeypatch):
    toy = prepared(tmp_path, monkeypatch)
    toy.write_enc_spec(lambda spec: [f.update(columns=1280) for f in spec["features"]])
    enc_failed(toy, "width must equal the declared input width and fit _WIDTH")


def test_code_identity_names_adapter_and_registry(tmp_path, monkeypatch):
    paths = enc.code_paths(enc._reader())
    assert paths["runner"].name == "r2_1_encoder_mosaic_cli.py"
    assert paths["head_registry"].name == "r2_1_encoder_head_registry.py"
    toy = prepared(tmp_path, monkeypatch)
    toy.write_enc_spec(lambda spec: spec["code_files"].update(
        runner=descriptor(frozen.code_paths(frozen._reader())["runner"])))
    enc_failed(toy, "runner: wrong actual imported source location")


def test_frozen_parents_are_byte_unchanged():
    assert sha(Path(frozen.__file__).read_bytes()) == \
        "94844fefdbe675547a4936d17b81f4287bf2e2ec6f4ef165f40e1fa8f19a50aa"
    assert sha(Path(enc._reader().__file__).read_bytes()) == \
        "1cfd3448ddb152a2002b1cda44e3088337ad62df7d6a4bcafd7f6e910927f31e"
