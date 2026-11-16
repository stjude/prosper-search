"""Tiny synthetic ten-view mosaic packets; no real payloads or fitted heads."""

from dataclasses import FrozenInstanceError, asdict, fields, replace
import csv
import gzip
import hashlib
import inspect
import io
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from prosper.evaluation import r2_1_hoptimus_asset_io as assets
from prosper.evaluation import r2_1_hoptimus_mosaic_reader as reader


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def descriptor(path, data):
    return dict(path=str(path), size=len(data), sha256=sha(data))


def registered(data):
    return dict(size=len(data), sha256=sha(data))


def csv_bytes(header, rows):
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return handle.getvalue().encode()


def npy(array, version=(1, 0)):
    handle = io.BytesIO()
    np.lib.format.write_array(handle, array, version=version, allow_pickle=False)
    return handle.getvalue()


def manual_npy(shape, descr="<i8", fortran=False, body=b""):
    handle = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        handle, dict(shape=shape, descr=descr, fortran_order=fortran))
    return handle.getvalue() + body


def binding(identity, role, declaration):
    return dict(asset_id=identity, role=role, **{k: declaration[k] for k in ("path", "size", "sha256")},
                resolved_path=declaration["path"], device=987, inode=654)


class PublicOnlyGuard:
    """Deliberately exposes read_verified and nothing else to the reader."""

    def __init__(self, guard):
        object.__setattr__(self, "_wrapped", guard)
        object.__setattr__(self, "reads", [])

    def read_verified(self, identity):
        object.__getattribute__(self, "reads").append(identity)
        return object.__getattribute__(self, "_wrapped").read_verified(identity)

    def __getattribute__(self, name):
        if name not in ("read_verified",):
            raise AssertionError("reader inspected guard attribute " + name)
        return object.__getattribute__(self, name)


class Packet:
    """Producer-style controls; only consumer-declared bytes exist on disk."""

    def __init__(self, root, layout="C", selected=((0, 5), (0, 100)), requested=(0, 1, 2)):
        self.root = root
        self.root.mkdir()
        self.selected = selected
        self.requested = requested
        self.population_rows = [
            [0, 0, 10, "s0", "p0", "A", 0],
            [1, 1, 11, "s1", "p0", "B", 0],
            [2, 2, 12, "s2", "p1", "A", 1],
        ]
        self.xy = np.array([[0, 0], [4, 8], [0, 0], [3, 5], [6, 10], [0, 0]], dtype="<i8")
        self.rows = (2, 3, 1)
        self.feature_bytes = {}
        self.coordinates = []
        self.feature_values = []
        start = 0
        for i, n in enumerate(self.rows):
            raw = np.array(np.arange(n * 1536).reshape(n, 1536), dtype=np.float32, order=layout)
            xy = self.xy[start:start + n]
            tiles = [dict(wsi_loc=[int(x), int(y)]) for x, y in xy]
            self.feature_bytes[i] = pickle.dumps((raw, tiles), protocol=4)
            self.coordinates.append(tiles)
            self.feature_values.append(raw)
            start += n
        self.counts = reader.MosaicCounts(3, 2, 2, 4, 3, 6, 10, 30, 60)
        sources = {}
        for role in sorted(reader.mosaic.SOURCES):
            data = f"synthetic source {role}".encode()
            sources[role] = descriptor("/producer/sources/" + role, data)
        population = csv_bytes(reader.POPULATION_HEADER, self.population_rows)
        sources["retrieval_population"] = descriptor("/producer/retrieval_population.csv", population)
        code = {role: descriptor("/producer/code/" + role + ".py", role.encode())
                for role in reader.mosaic.code_paths(reader.mosaic._reader())}
        features = [
            dict(slide_id="s" + str(i), **descriptor("/producer/features/s" + str(i) + ".pkl", data),
                 rows=self.rows[i], dtype="float32", columns=1536)
            for i, data in self.feature_bytes.items()
        ]
        heads, provenance, links = [], [], []
        for fold, budget in reader.VIEW_ORDER:
            name = f"fold{fold}_budget{budget}"
            historical = "/historical/heads/" + name
            head = dict(
                held_out_fold=fold, budget_percent=budget,
                assets={role: descriptor("/producer/heads/" + name + "/" + role, (name + role).encode())
                        for role in reader.mosaic.HEAD_ASSETS},
                registry_keys={name: historical + "/" + name
                               for name in ("complete", *reader.mosaic.REGISTRY_NAMES)},
                producer_spec_path="/historical/specs/" + name + ".json",
            )
            state = sha(name.encode())
            common = {"acceptance": "head_acceptance", "registry": "head_registry",
                      "pilot_acceptance": "pilot_acceptance", "head_plan": "head_plan"}
            p = dict(
                held_out_fold=fold, budget_percent=budget, run_id="synthetic_" + name,
                producer_spec_path=head["producer_spec_path"],
                registry_complete_key=head["registry_keys"]["complete"],
                registry_artifact_keys=[[file, head["registry_keys"][file]] for file in reader.mosaic.REGISTRY_NAMES],
                artifact_sha256=[
                    [role, (sources[common[role]] if role in common else head["assets"][role])["sha256"]]
                    for role in ("acceptance", "registry", "pilot_acceptance",
                                 *reader.mosaic.HEAD_ASSETS, "head_plan")
                ],
                fitted_state_sha256=state,
                diagnostics_json=encode(dict(held_out_fold=fold, budget_percent=budget,
                                             fitted_state_sha256=state)).decode(),
                consumer_runtime_json=encode(dict(runtime="synthetic_reference")).decode(),
            )
            file_roles = {"head.pkl": "model", "fit_diagnostics.json": "diagnostics",
                          "input_bindings.json": "input_bindings", "summary.json": "summary"}
            link = dict(
                held_out_fold=fold, budget_percent=budget,
                complete_current_path=head["assets"]["complete"]["path"],
                complete_historical_registry_key=head["registry_keys"]["complete"],
                producer_spec_current_path=head["assets"]["spec"]["path"],
                producer_spec_historical_path=head["producer_spec_path"],
                consumed_artifacts={
                    file: dict(asset_id=f"head:{name}:{role}", current_path=head["assets"][role]["path"],
                               historical_registry_key=head["registry_keys"][file],
                               **{k: head["assets"][role][k] for k in ("size", "sha256")})
                    for file, role in file_roles.items()
                },
                referenced_not_reread={file: registered(("head reference " + file).encode())
                                      for file in ("counts.json", "held_out_rows.csv.gz",
                                                   "predictions.npy", "probabilities.npy")},
            )
            heads.append(head)
            provenance.append(p)
            links.append(link)
        self.spec = dict(schema_version=1, run_id="synthetic_mosaic", task=reader.mosaic.TASK,
                         expected=asdict(self.counts), parameters=dict(prediction_batch_rows=4096),
                         sources=sources, code_files=code, features=features, heads=heads)
        self.files = {"coordinates.npy": npy(self.xy), "retrieval_population.csv": population}
        offsets = []
        start = 0
        for i, n in enumerate(self.rows):
            p = self.population_rows[i]
            offsets.append([i, p[3], p[4], p[5], p[6], start, start + n, n, features[i]["sha256"]])
            start += n
        self.offset_rows = offsets
        self.files["slide_offsets.csv"] = csv_bytes(reader.OFFSET_HEADER, offsets)
        count_rows = []
        for fold, budget in reader.VIEW_ORDER:
            values = np.array([True, budget == 100, fold == 1, False, budget == 100, fold != 2], dtype=np.bool_)
            name = f"fold{fold}_budget{budget}"
            self.files[name + "_mask.npy"] = npy(values)
            self.files[name + "_predictions.npy"] = npy(np.where(values, 1, 2).astype("<i8"))
            self.files[name + "_probabilities.npy"] = npy(np.zeros((6, 5), dtype="<f4"))
            for i, offset in enumerate(offsets):
                retained = int(values[offset[5]:offset[6]].sum())
                count_rows.append([
                    fold, budget, i, offset[1], offset[2], offset[4], int(offset[4] == fold),
                    self.rows[i], retained, self.rows[i] - retained, 0, 0, 0,
                    retained, self.rows[i] - retained, "nonempty" if retained else "zero_predicted_tumor",
                ])
        self.count_rows = count_rows
        self.files["mask_counts.csv.gz"] = gzip.compress(csv_bytes(reader.COUNT_HEADER, count_rows), mtime=0)
        zero = sum(r[-1] == "zero_predicted_tumor" for r in count_rows)
        total_bytes = sum(len(data) for data in self.feature_bytes.values())
        self.summary = dict(
            schema_version=1, run_id=self.spec["run_id"], task=reader.mosaic.TASK,
            counts={**asdict(self.counts), "zero_mask_cells": zero, "nonzero_mask_cells": 30 - zero},
            policy=dict(reader._POLICY),
            details=dict(
                head_provenance=provenance, tuple_order=[list(k) for k in reader.VIEW_ORDER],
                feature_columns=1536, coordinate_columns=2, packed_array_data_bytes=306 * 6,
                array_count=31, decoded_feature_files=3, consumed_feature_bytes=total_bytes,
                minimum_feature_bytes_with_final_sweep=2 * total_bytes,
                maximum_encoded_feature_bytes=max(map(len, self.feature_bytes.values())),
                maximum_dense_feature_bytes=3 * 6144, maximum_contiguous_feature_batch_bytes=3 * 6144,
                coverage_accounting_bytes=270, written_slide_head_cells=30, written_tile_head_predictions=60,
                consumed_source_roles=sorted(reader.mosaic.SOURCES),
                semantic_readback="all_arrays_offsets_counts_masks_and_coverage_recounted",
            ),
            limitations=list(reader.mosaic.LIMITATIONS),
        )
        entries = [binding("spec", "spec", descriptor("/producer/mosaic_spec.json", b"temporary"))]
        entries.extend(binding("source:" + role, role, value) for role, value in sources.items())
        entries.extend(binding("code:" + role, "code", value) for role, value in code.items())
        entries.extend(binding("feature:" + f["slide_id"], "feature", f) for f in features)
        for head in heads:
            for role, value in head["assets"].items():
                entries.append(binding(f"head:fold{head['held_out_fold']}_budget{head['budget_percent']}:{role}",
                                       "head_model" if role == "model" else "head_control", value))
        cpu_roles = {"head_plan.json": "head_plan", "input_bindings.json": "cpu_input_bindings",
                     "patient_folds.csv": "patient_folds", "retrieval_population.csv": "retrieval_population",
                     "summary.json": "cpu_summary"}
        self.inputs = dict(
            schema_version=1, run_id=self.spec["run_id"], task=reader.mosaic.TASK, spec_sha256=sha(b"temporary"),
            bindings=entries, executing_code_paths={role: value["path"] for role, value in code.items()},
            consumed_cpu_artifacts={name: {k: sources[role][k] for k in ("size", "sha256")}
                                    for name, role in cpu_roles.items()},
            head_provenance=provenance, producer_artifact_links=links,
            feature_consumption=[dict(slide_id=f["slide_id"], status="decoded_once",
                                      asset_id="feature:" + f["slide_id"]) for f in features],
            current_environment=dict(python="synthetic", hostname="synthetic",
                                     native_environment={name: "1" for name in reader.producer.NATIVE_VARIABLES},
                                     head_consumer_runtime=[p["consumer_runtime_json"] for p in provenance]),
            started_at="synthetic_start", captured_at="synthetic_end", limitations=list(reader.mosaic.LIMITATIONS),
        )
        self.complete_changes = {}
        self.review_changes = {}
        self.registry_changes = {}
        self.aliases = {}
        self.control_overrides = {}

    def sync_population(self):
        data = csv_bytes(reader.POPULATION_HEADER, self.population_rows)
        self.files["retrieval_population.csv"] = data
        source = descriptor(self.spec["sources"]["retrieval_population"]["path"], data)
        self.spec["sources"]["retrieval_population"] = source
        entry = next(b for b in self.inputs["bindings"] if b["asset_id"] == "source:retrieval_population")
        entry.update({k: source[k] for k in ("size", "sha256")})
        self.inputs["consumed_cpu_artifacts"]["retrieval_population.csv"] = registered(data)

    def sync_counts(self):
        self.files["mask_counts.csv.gz"] = gzip.compress(csv_bytes(reader.COUNT_HEADER, self.count_rows), mtime=0)

    def publish(self):
        spec_data = self.control_overrides.get("spec", encode(self.spec))
        self.inputs["spec_sha256"] = sha(spec_data)
        entry = next((b for b in self.inputs["bindings"] if b["asset_id"] == "spec"), None)
        if entry is not None:
            entry.update(registered(spec_data))
        self.files["summary.json"] = self.control_overrides.get("summary", encode(self.summary))
        self.files["input_bindings.json"] = self.control_overrides.get("input_bindings", encode(self.inputs))
        registry = {name: registered(data) for name, data in self.files.items()}
        registry.update(self.registry_changes)
        for name in [k for k, value in registry.items() if value is None]:
            del registry[name]
        complete = dict(schema_version=1, run_id=self.spec["run_id"], task=reader.mosaic.TASK,
                        scope=reader.mosaic.SCOPE, spec_sha256=sha(spec_data), outputs=registry,
                        verified_at="synthetic_end", limitations=list(reader.mosaic.LIMITATIONS))
        complete.update(self.complete_changes)
        complete_data = self.control_overrides.get("complete", encode(complete))
        review = dict(
            schema_version=1, stage=reader.REVIEW_STAGE, disposition=reader.REVIEW_DISPOSITION,
            bindings=dict(complete_sha256=sha(complete_data), spec_sha256=sha(spec_data),
                          summary_sha256=sha(self.files["summary.json"]),
                          input_bindings_sha256=sha(self.files["input_bindings.json"]),
                          lead_reconciliation_sha256=sha(b"lead"), runtime_acceptance_sha256=sha(b"runtime"),
                          cpu_preflight_receipt_sha256=sha(b"cpu")),
            review_date="synthetic", reviewer="synthetic", job_id=1, independent_evidence={},
            query_fold_eligible_mask_counts={}, resources={}, limits=[],
        )
        review.update(self.review_changes)
        review_data = self.control_overrides.get("review", encode(review))
        data = dict(self.files, **{"COMPLETE.json": complete_data, "mosaic_spec.json": spec_data,
                                  "actual_outputs_independent_review.json": review_data})
        specs = {}
        for field, identity, role, _, logical in reader._CONTROLS:
            path = self.root / logical
            path.write_bytes(data[logical])
            specs[field] = assets.AssetSpec(identity, role, self.aliases.get(field, str(path)), sha(data[logical]),
                                           len(data[logical]))
        path = self.root / "coordinates.npy"
        path.write_bytes(data[path.name])
        specs["coordinates"] = assets.AssetSpec("mosaic:coordinates", "mosaic_coordinates",
                                               self.aliases.get("coordinates", str(path)),
                                               sha(data[path.name]), len(data[path.name]))
        masks = []
        for fold, budget in self.selected:
            name = f"fold{fold}_budget{budget}_mask.npy"
            path = self.root / name
            path.write_bytes(data[name])
            masks.append(reader.MaskAsset(reader.HeadView(fold, budget), assets.AssetSpec(
                f"mosaic:mask:fold{fold}:budget{budget}", "mosaic_mask",
                self.aliases.get(name, str(path)), sha(data[name]), len(data[name]))))
        features = []
        for index in self.requested:
            path = self.root / f"raw{index}.pkl"
            path.write_bytes(self.feature_bytes[index])
            features.append(reader.FeatureAsset(index, assets.AssetSpec(
                "feature:s" + str(index), "mosaic_feature", self.aliases.get("feature" + str(index), str(path)),
                sha(self.feature_bytes[index]), len(self.feature_bytes[index]))))
        self.bundle_assets = reader.MosaicBundleAssets(**specs, masks=tuple(masks), features=tuple(features))
        self.expected = reader.MosaicExpected(sha(review_data), sha(complete_data), sha(spec_data),
                                              self.spec["run_id"], self.counts)
        declared = list(specs.values()) + [m.asset for m in masks] + [f.asset for f in features]
        policies = {role: assets.AssetPolicy("bytes", limit) for _, _, role, limit, _ in reader._CONTROLS}
        policies.update(mosaic_coordinates=assets.AssetPolicy("bytes", 64 * reader.MIB),
                        mosaic_mask=assets.AssetPolicy("bytes", 8 * reader.MIB),
                        mosaic_feature=assets.AssetPolicy("bytes", 64 * reader.MIB))
        self.guard = assets.AssetGuard(declared, base_dir=self.root, policies=policies)
        return self

    def load(self, guard=None):
        return reader.load_mosaic_bundle(self.guard if guard is None else guard,
                                         bundle_assets=self.bundle_assets, expected=self.expected)


@pytest.fixture
def packet(tmp_path):
    return Packet(tmp_path / "packet")


@pytest.fixture(autouse=True)
def no_fit_or_head_load(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("mosaic reader must not fit, reload heads, or predict")
    monkeypatch.setattr(reader.mosaic.fitter, "fit_head", forbidden)
    monkeypatch.setattr(reader.mosaic.fitter.SGDClassifier, "fit", forbidden)
    monkeypatch.setattr(reader.mosaic._reader(), "load_saved_annotation_head", forbidden)


def rejected(packet, match=None):
    packet.publish()
    with packet.guard:
        with pytest.raises((ValueError, OSError), match=match):
            packet.load()


def test_signatures_and_exact_record_fields():
    assert list(inspect.signature(reader.load_mosaic_bundle).parameters) == ["guard", "bundle_assets", "expected"]
    assert inspect.signature(reader.load_mosaic_bundle).parameters["bundle_assets"].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(inspect.signature(reader.load_raw_slide).parameters) == ["guard", "bundle", "slide_index"]
    assert tuple(f.name for f in fields(reader.MaskCountRecord)) == reader.COUNT_HEADER
    assert tuple(f.name for f in fields(reader.LoadedMask)) == ("view", "backing", "counts")
    assert tuple(f.name for f in fields(reader.NpyBacking)) == ("encoded", "data_offset", "dtype_str", "shape", "fortran_order")


@pytest.mark.parametrize("layout", ["C", "F"])
def test_complete_positive_public_api_custody_and_decoder_identity(tmp_path, monkeypatch, layout):
    p = Packet(tmp_path / "packet", layout=layout).publish()
    original = reader.mosaic.decode_feature
    decoded = []
    def capture(data, feature):
        result = original(data, feature)
        decoded.append(result[0])
        return result
    monkeypatch.setattr(reader.mosaic, "decode_feature", capture)
    with p.guard:
        public = PublicOnlyGuard(p.guard)
        bundle = p.load(public)
        assert not decoded
        assert len(bundle.output_registry) == 36
        assert [d.name for d in bundle.output_registry] == sorted(reader.mosaic.artifact_names())
        assert [s.patient_id for s in bundle.slides] == ["p0", "p0", "p1"]
        assert len(bundle.consumed_controls) == 8
        assert len(bundle.consumed_arrays) == 3
        assert len([r for r in bundle.referenced_only if r.kind == "source"]) == 14
        assert len([r for r in bundle.referenced_only if r.kind == "head"]) == 10
        for i in range(3):
            slide = reader.load_raw_slide(public, bundle, i)
            assert slide.raw is decoded[-1]
            np.testing.assert_array_equal(slide.raw, p.feature_values[i])
            np.testing.assert_array_equal(slide.coordinates, p.xy[bundle.slides[i].tile_start:bundle.slides[i].tile_stop])
            assert slide.layout.shape == slide.raw.shape and slide.layout.strides == slide.raw.strides
            assert slide.layout.dtype_str == "<f4"
            assert not slide.raw.flags.writeable and not slide.coordinates.flags.writeable
            if i == 2:
                assert slide.layout.c_contiguous and slide.layout.f_contiguous
            else:
                assert slide.layout.c_contiguous == (layout == "C")
                assert slide.layout.f_contiguous == (layout == "F")
        assert p.guard.state == "active"
        final = p.guard.final_bindings()
        assert len(final) == 14
        assert p.guard.state == "sealed"
    consumed = {r.asset_id for r in (*bundle.consumed_controls, *bundle.consumed_arrays)}
    assert set(object.__getattribute__(public, "reads")) == consumed | {"feature:s0", "feature:s1", "feature:s2"}
    for reference in bundle.referenced_only:
        assert encode(json.loads(reference.details_json)).decode() == reference.details_json


def test_only_selected_arrays_and_requested_feature_are_read(tmp_path):
    p = Packet(tmp_path / "packet", selected=((0, 5),), requested=(0,)).publish()
    with p.guard:
        public = PublicOnlyGuard(p.guard)
        b = p.load(public)
        assert [f.consumer_asset is None for f in b.feature_records] == [False, True, True]
        assert len([r for r in b.referenced_only if r.kind == "output"]) == 29
        assert not any(identity.startswith("feature:") for identity in object.__getattribute__(public, "reads"))
        with pytest.raises(ValueError, match="not declared"):
            reader.load_raw_slide(public, b, 1)
        reader.load_raw_slide(public, b, 0)
        assert object.__getattribute__(public, "reads")[-1] == "feature:s0"
        p.guard.final_bindings()
    assert not list(p.root.glob("*probabilities*"))
    assert not list(p.root.glob("*predictions*"))


def test_empty_selected_subset_still_checks_all_ten_view_counts(tmp_path):
    p = Packet(tmp_path / "packet", selected=(), requested=()).publish()
    with p.guard:
        b = p.load()
        assert b.masks == () and len(b.consumed_arrays) == 1
        assert all(f.consumer_asset is None for f in b.feature_records)
        p.guard.final_bindings()


def test_immutable_snapshots_and_fresh_array_views(packet):
    packet.publish()
    with packet.guard:
        b = packet.load()
        packet.spec["features"][0]["path"] = "/mutated"
        assert b.feature_records[0].producer_path != "/mutated"
        with pytest.raises(FrozenInstanceError):
            b.slides[0].slide_index = 8
        mask = b.masks[0]
        assert type(mask.counts) is tuple
        with pytest.raises(FrozenInstanceError):
            mask.counts[0].retained_tiles = 500
        first, second = b.coordinates.array_view(), b.coordinates.array_view()
        assert first is not second
        first.shape = (12,)
        assert b.coordinates.shape == second.shape == (6, 2)
        for view in (second, mask.backing.array_view()):
            with pytest.raises(ValueError):
                view.setflags(write=True)
            with pytest.raises(ValueError):
                view.flat[0] = 1


@pytest.mark.parametrize("value", [True, False, 0.0, 1.0, -1, "0"])
@pytest.mark.parametrize("which", ["fold", "budget", "slide_count", "slide_index", "artifact_size", "npy_offset"])
def test_integer_record_fields_reject_alias_types(packet, value, which):
    with pytest.raises(ValueError):
        if which == "fold":
            reader.HeadView(value, 5)
        elif which == "budget":
            reader.HeadView(0, value)
        elif which == "slide_count":
            replace(packet.counts, slides=value)
        elif which == "slide_index":
            reader.FeatureAsset(value, assets.AssetSpec("feature:s0", "mosaic_feature", "/synthetic", sha(b""), 0))
        elif which == "artifact_size":
            reader.ArtifactDescriptor("x", value, sha(b""))
        else:
            reader.NpyBacking(npy(packet.xy), value, "<i8", (6, 2), False)


@pytest.mark.parametrize("field,value", [
    ("heads", 9), ("feature_files", 2), ("original_tiles", 2), ("slide_head_cells", 29),
    ("tile_head_predictions", 59), ("patients", 4), ("diagnostic_groups", 4), ("union_patients", 1),
])
def test_incoherent_expected_counts(packet, field, value):
    with pytest.raises(ValueError):
        replace(packet.counts, **{field: value})


@pytest.mark.parametrize("which", ["review", "complete", "spec"])
def test_wrong_expected_anchor_rejected_before_reads(packet, which):
    packet.publish()
    expected = replace(packet.expected, **{which + "_sha256": sha(b"wrong")})
    class NoReads:
        def read_verified(self, identity):
            pytest.fail("mismatching expected anchor must fail before read")
    with pytest.raises(ValueError, match="anchor"):
        reader.load_mosaic_bundle(NoReads(), bundle_assets=packet.bundle_assets, expected=expected)


@pytest.mark.parametrize("field", [name for name, *_ in reader._CONTROLS] + ["coordinates"])
@pytest.mark.parametrize("change", ["role", "id", "size"])
def test_fixed_control_descriptor_caps_before_reads(packet, field, change):
    packet.publish()
    original = getattr(packet.bundle_assets, field)
    value = replace(original, **({"role": "wrong"} if change == "role" else
                                {"asset_id": "wrong"} if change == "id" else {"size": 65 * reader.MIB}))
    with pytest.raises(ValueError):
        replace(packet.bundle_assets, **{field: value})


@pytest.mark.parametrize("change", ["id", "role", "cap", "duplicate", "duplicate_masks"])
def test_mask_and_feature_declaration_admission(packet, change):
    packet.publish()
    a = packet.bundle_assets
    with pytest.raises(ValueError):
        if change in ("id", "role", "cap"):
            mask = a.masks[0]
            new = replace(mask.asset, **({"asset_id": a.masks[1].asset.asset_id} if change == "id" else
                                        {"role": "feature"} if change == "role" else {"size": 8 * reader.MIB + 1}))
            replace(mask, asset=new)
        elif change == "duplicate":
            replace(a, features=(a.features[0], a.features[0]))
        else:
            replace(a, masks=(a.masks[0], a.masks[0]))


@pytest.mark.parametrize("value", [True, 0.0, -1, 3])
def test_raw_index_before_read(packet, value):
    packet.publish()
    with packet.guard:
        b = packet.load()
        class NoReads:
            def read_verified(self, identity):
                pytest.fail("invalid index must precede read")
        with pytest.raises(ValueError):
            reader.load_raw_slide(NoReads(), b, value)


def test_active_guard_required_and_reader_does_not_enter(packet):
    packet.publish()
    with pytest.raises(assets.AssetError):
        packet.load()


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("stage", "cpu_preflight"), ("disposition", "accepted_implementation"),
    ("branches", {}), ("outputs", {}), ("unknown", None),
])
def test_review_schema_rejection(packet, field, value):
    packet.review_changes[field] = value
    rejected(packet, "review105")


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("run_id", "wrong"), ("task", "head"), ("scope", "all_features"),
    ("spec_sha256", sha(b"wrong")), ("limitations", []), ("unknown", 1),
])
def test_complete_schema_rejection(packet, field, value):
    packet.complete_changes[field] = value
    rejected(packet)


@pytest.mark.parametrize("which", ["missing", "extra", "boolean_size", "invalid_hash", "wrong_consumed_hash", "wrong_consumed_size"])
def test_full_registry_not_just_consumed_subset(packet, which):
    name = "fold4_budget100_probabilities.npy"
    if which == "missing":
        packet.registry_changes[name] = None
    elif which == "extra":
        packet.registry_changes["extra.npy"] = registered(b"x")
    elif which == "boolean_size":
        packet.registry_changes[name] = dict(size=True, sha256=sha(b"x"))
    elif which == "invalid_hash":
        packet.registry_changes[name] = dict(size=1, sha256="z" * 64)
    else:
        actual = registered(packet.files["coordinates.npy"])
        actual["sha256" if which == "wrong_consumed_hash" else "size"] = sha(b"wrong") if which == "wrong_consumed_hash" else 3
        packet.registry_changes["coordinates.npy"] = actual
    rejected(packet)


@pytest.mark.parametrize("which", ["spec", "summary", "input_bindings", "review", "complete"])
@pytest.mark.parametrize("bad", ["duplicate", "nonfinite", "list"])
def test_strict_control_json(packet, which, bad):
    packet.control_overrides[which] = {
        "duplicate": b'{"schema_version":1,"schema_version":1}',
        "nonfinite": b'{"bad":NaN}', "list": b"[]",
    }[bad]
    rejected(packet)


@pytest.mark.parametrize("which", [
    "spec_count", "parameters", "feature_width", "feature_dtype", "feature_rows_bool",
    "feature_duplicate", "head_missing", "head_duplicate", "head_bool", "source_missing",
    "source_extra", "source_hash", "source_size", "code_missing", "binding_duplicate",
    "binding_missing", "binding_extra", "binding_role", "binding_path", "binding_hash",
    "binding_size_bool", "binding_device_bool", "executing_code", "cpu_link",
    "feature_consumption", "provenance_order", "provenance_hash", "provenance_registry",
    "provenance_json", "head_link", "head_link_control", "head_link_reference",
    "summary_policy", "summary_count", "summary_zero", "summary_detail", "summary_source_roles",
    "environment_type", "environment_reference", "limitations",
])
def test_producer_control_and_reference_joins(packet, which):
    s, i, u = packet.spec, packet.inputs, packet.summary
    first = next(b for b in i["bindings"] if b["asset_id"] == "feature:s0")
    if which == "spec_count":
        s["expected"]["patients"] = 1
    elif which == "parameters":
        s["parameters"]["prediction_batch_rows"] = 128
    elif which == "feature_width":
        s["features"][0]["columns"] = 1535
    elif which == "feature_dtype":
        s["features"][0]["dtype"] = "float16"
    elif which == "feature_rows_bool":
        s["features"][0]["rows"] = True
    elif which == "feature_duplicate":
        s["features"][1] = dict(s["features"][0])
    elif which == "head_missing":
        s["heads"].pop()
    elif which == "head_duplicate":
        s["heads"][1] = dict(s["heads"][0])
    elif which == "head_bool":
        s["heads"][0]["held_out_fold"] = False
    elif which == "source_missing":
        s["sources"].pop("stage0_inventory")
    elif which == "source_extra":
        s["sources"]["extra"] = dict(s["sources"]["stage0_inventory"])
    elif which == "source_hash":
        s["sources"]["stage0_inventory"]["sha256"] = sha(b"wrong")
    elif which == "source_size":
        s["sources"]["stage0_inventory"]["size"] = 256 * reader.MIB
    elif which == "code_missing":
        s["code_files"].pop("runner")
    elif which == "binding_duplicate":
        i["bindings"].append(dict(first))
    elif which == "binding_missing":
        i["bindings"].remove(first)
    elif which == "binding_extra":
        i["bindings"].append(dict(first, asset_id="extra"))
    elif which == "binding_role":
        first["role"] = "head_model"
    elif which == "binding_path":
        first["path"] = "/wrong"
    elif which == "binding_hash":
        first["sha256"] = sha(b"wrong")
    elif which == "binding_size_bool":
        first["size"] = True
    elif which == "binding_device_bool":
        first["device"] = False
    elif which == "executing_code":
        i["executing_code_paths"]["runner"] = "/wrong"
    elif which == "cpu_link":
        i["consumed_cpu_artifacts"]["head_plan.json"]["sha256"] = sha(b"wrong")
    elif which == "feature_consumption":
        i["feature_consumption"].reverse()
    elif which == "provenance_order":
        i["head_provenance"] = list(reversed(i["head_provenance"]))
    elif which == "provenance_hash":
        i["head_provenance"][0]["artifact_sha256"][0][1] = sha(b"wrong")
    elif which == "provenance_registry":
        i["head_provenance"][0]["registry_complete_key"] = "/wrong"
    elif which == "provenance_json":
        i["head_provenance"][0]["diagnostics_json"] = '{"x":NaN}'
    elif which == "head_link":
        i["producer_artifact_links"][0]["producer_spec_current_path"] = "/wrong"
    elif which == "head_link_control":
        i["producer_artifact_links"][0]["consumed_artifacts"]["head.pkl"]["asset_id"] = "wrong"
    elif which == "head_link_reference":
        i["producer_artifact_links"][0]["referenced_not_reread"].pop("probabilities.npy")
    elif which == "summary_policy":
        u["policy"]["decision"] = "threshold"
    elif which == "summary_count":
        u["counts"]["original_tiles"] = 7
    elif which == "summary_zero":
        u["counts"]["zero_mask_cells"] += 1
        u["counts"]["nonzero_mask_cells"] -= 1
    elif which == "summary_detail":
        u["details"]["maximum_dense_feature_bytes"] += 4
    elif which == "summary_source_roles":
        u["details"]["consumed_source_roles"].pop()
    elif which == "environment_type":
        i["current_environment"]["native_environment"][reader.producer.NATIVE_VARIABLES[0]] = 1
    elif which == "environment_reference":
        i["current_environment"]["head_consumer_runtime"][0] = "wrong"
    else:
        i["limitations"] = []
    rejected(packet)


@pytest.mark.parametrize("which", [
    "population_header", "missing", "extra", "order", "duplicate_slide", "duplicate_metadata",
    "cross_label_fold", "normal", "misc", "float", "signed", "leading_zero", "bool",
    "record_limit", "offset_header", "offset_gap", "offset_stop", "offset_hash", "offset_order",
    "offset_patient", "offset_label", "count_header", "count_order", "count_missing", "count_extra",
    "count_fold", "count_budget", "count_slide", "count_patient", "count_eligibility",
    "count_class_sum", "count_retained", "count_discarded", "count_status", "count_float", "gzip",
])
def test_csv_schema_order_population_offsets_and_all_counts(packet, which):
    p, o, c = packet.population_rows, packet.offset_rows, packet.count_rows
    if which == "population_header":
        packet.files["retrieval_population.csv"] = b"wrong\n"
    elif which in ("missing", "extra"):
        p.pop() if which == "missing" else p.append(list(p[-1]))
        packet.sync_population()
    elif which == "order":
        p.reverse(); packet.sync_population()
    elif which == "duplicate_slide":
        p[1][3] = p[0][3]; packet.sync_population()
    elif which == "duplicate_metadata":
        p[1][2] = p[0][2]; packet.sync_population()
    elif which == "cross_label_fold":
        p[1][6] = 1; o[1][4] = 1; packet.sync_population()
    elif which in ("normal", "misc"):
        p[1][5] = which.upper(); o[1][3] = which.upper(); packet.sync_population()
    elif which in ("float", "signed", "leading_zero", "bool"):
        p[0][0] = dict(float="0.0", signed="+0", leading_zero="00", bool="False")[which]
        packet.sync_population()
    elif which == "record_limit":
        p[0][3] = "x" * 8500; packet.sync_population()
    elif which == "offset_header":
        packet.files["slide_offsets.csv"] = b"wrong\n"
    elif which == "offset_gap":
        o[1][5] += 1
    elif which == "offset_stop":
        o[-1][6] += 1
    elif which == "offset_hash":
        o[0][-1] = sha(b"wrong")
    elif which == "offset_order":
        o.reverse()
    elif which == "offset_patient":
        o[0][2] = "wrong"
    elif which == "offset_label":
        o[0][3] = "wrong"
    elif which == "count_header":
        packet.files["mask_counts.csv.gz"] = gzip.compress(b"wrong\n", mtime=0)
    elif which == "count_order":
        c.reverse()
    elif which == "count_missing":
        c.pop()
    elif which == "count_extra":
        c.append(list(c[-1]))
    elif which == "gzip":
        packet.files["mask_counts.csv.gz"] = b"invalid gzip"
    else:
        col, val = {
            "count_fold": (0, 1), "count_budget": (1, 100), "count_slide": (2, 2),
            "count_patient": (4, "wrong"), "count_eligibility": (6, 0), "count_class_sum": (9, 4),
            "count_retained": (13, 0), "count_discarded": (14, 0), "count_status": (15, "wrong"),
            "count_float": (0, "0.0"),
        }[which]
        c[0][col] = val
    if which.startswith("offset_") and which != "offset_header" or which in ("cross_label_fold", "normal", "misc"):
        packet.files["slide_offsets.csv"] = csv_bytes(reader.OFFSET_HEADER, o)
    if which.startswith("count_") and which != "count_header":
        packet.sync_counts()
    rejected(packet)


@pytest.mark.parametrize("which", [
    "version", "fortran", "big_endian", "float", "structured", "object", "shape", "truncated",
    "trailing", "second_array", "overflow_shape", "boolean_shape", "negative", "duplicate", "swapped_order",
])
def test_coordinate_array_admission_and_complete_raw_join(packet, which):
    xy = packet.xy.copy()
    if which == "version":
        data = npy(xy, (2, 0))
    elif which == "fortran":
        data = npy(np.asfortranarray(xy))
    elif which == "big_endian":
        data = npy(xy.astype(">i8"))
    elif which == "float":
        data = npy(xy.astype("<f8"))
    elif which == "structured":
        data = npy(np.zeros((6, 2), dtype=[("x", "<i8")]))
    elif which == "object":
        data = manual_npy((6, 2), "|O", body=b"x")
    elif which == "shape":
        data = npy(xy.reshape(12))
    elif which == "truncated":
        data = packet.files["coordinates.npy"][:-1]
    elif which == "trailing":
        data = packet.files["coordinates.npy"] + b"x"
    elif which == "second_array":
        data = packet.files["coordinates.npy"] * 2
    elif which == "overflow_shape":
        data = manual_npy((2**63, 2))
    elif which == "boolean_shape":
        data = manual_npy((True, 2), body=b"\0" * 16)
    elif which == "negative":
        xy[0, 0] = -1; data = npy(xy)
    elif which == "duplicate":
        xy[1] = xy[0]; data = npy(xy)
    else:
        xy[[0, 1]] = xy[[1, 0]]; data = npy(xy)
    packet.files["coordinates.npy"] = data
    if which == "swapped_order":
        packet.publish()
        with packet.guard:
            bundle = packet.load()
            with pytest.raises(ValueError, match="ordered coordinate"):
                reader.load_raw_slide(packet.guard, bundle, 0)
    else:
        rejected(packet)


@pytest.mark.parametrize("which", ["dtype", "shape", "fortran", "version", "noncanonical", "membership", "trailing", "swapped_views"])
def test_selected_mask_arrays_and_named_membership(packet, which):
    name = "fold0_budget5_mask.npy"
    if which == "dtype":
        data = npy(np.ones(6, dtype=np.uint8))
    elif which == "shape":
        data = npy(np.ones((6, 1), dtype=np.bool_))
    elif which == "fortran":
        data = manual_npy((6,), "|b1", True, body=b"\0" * 6)
    elif which == "version":
        data = npy(np.ones(6, dtype=np.bool_), (2, 0))
    elif which == "noncanonical":
        data = packet.files[name][:-1] + b"\2"
    elif which == "membership":
        data = npy(np.zeros(6, dtype=np.bool_))
    elif which == "trailing":
        data = packet.files[name] + b"x"
    else:
        data = packet.files["fold0_budget100_mask.npy"]
    packet.files[name] = data
    rejected(packet)


@pytest.mark.parametrize("which", ["dtype", "width", "rows", "nonfinite", "tuple", "coordinates", "trailing"])
def test_exact_frozen_raw_decoder_rejections(packet, which):
    raw, tiles = packet.feature_values[0], packet.coordinates[0]
    if which == "dtype":
        raw = raw.astype(np.float16)
    elif which == "width":
        raw = raw[:, :-1]
    elif which == "rows":
        raw = raw[:1]
    elif which == "nonfinite":
        raw = raw.copy(); raw[0, 0] = np.nan
    elif which == "coordinates":
        tiles = list(reversed(tiles))
    value = [raw, tiles] if which == "tuple" else (raw, tiles)
    data = pickle.dumps(value, protocol=4) + (b"x" if which == "trailing" else b"")
    packet.feature_bytes[0] = data
    feature = packet.spec["features"][0]
    feature.update(registered(data))
    packet.offset_rows[0][-1] = feature["sha256"]
    packet.files["slide_offsets.csv"] = csv_bytes(reader.OFFSET_HEADER, packet.offset_rows)
    next(b for b in packet.inputs["bindings"] if b["asset_id"] == "feature:s0").update(registered(data))
    details = packet.summary["details"]
    total = sum(map(len, packet.feature_bytes.values()))
    details.update(consumed_feature_bytes=total, minimum_feature_bytes_with_final_sweep=2 * total,
                   maximum_encoded_feature_bytes=max(map(len, packet.feature_bytes.values())))
    packet.publish()
    with packet.guard:
        bundle = packet.load()
        with pytest.raises(ValueError):
            reader.load_raw_slide(packet.guard, bundle, 0)


def test_current_feature_bytes_must_match_original_before_decode(packet, monkeypatch):
    packet.publish()
    def forbidden(*args):
        pytest.fail("changed feature bytes must not be decoded")
    with packet.guard:
        b = packet.load()
        path = Path(packet.bundle_assets.features[0].asset.path)
        path.write_bytes(b"bad")
        monkeypatch.setattr(reader.mosaic, "decode_feature", forbidden)
        with pytest.raises(assets.AssetError):
            reader.load_raw_slide(packet.guard, b, 0)


@pytest.mark.parametrize("field", ["coordinates", "population", "feature0"])
def test_current_path_alias_preserves_historical_paths(packet, field):
    name = dict(coordinates="coordinates.npy", population="retrieval_population.csv", feature0="raw0.pkl")[field]
    alias = packet.root / ("alias_" + name)
    alias.symlink_to(packet.root / name)
    packet.aliases[field] = str(alias)
    packet.publish()
    with packet.guard:
        b = packet.load()
        raw = reader.load_raw_slide(packet.guard, b, 0)
        assert b.feature_records[0].producer_path == "/producer/features/s0.pkl"
        assert b.feature_records[0].producer_resolved_path == "/producer/features/s0.pkl"
        if field == "feature0":
            assert raw.consumption.consumer_path == str(alias)
        packet.guard.final_bindings()


def test_same_role_alias_collapse_is_rejected(packet):
    packet.publish()
    mask = packet.bundle_assets.masks[0].asset
    duplicate = replace(mask, asset_id="mosaic:mask:fold1:budget5")
    with pytest.raises(assets.AssetError):
        assets.AssetGuard((mask, duplicate), base_dir=packet.root,
                          policies={"mosaic_mask": assets.AssetPolicy("bytes", 8 * reader.MIB)})


@pytest.mark.parametrize("target", ["coordinates", "mask", "unread_feature"])
def test_caller_final_sweep_detects_late_mutation(packet, target):
    packet.publish()
    with packet.guard:
        b = packet.load()
        assert packet.guard.state == "active"
        asset = {"coordinates": packet.bundle_assets.coordinates,
                 "mask": packet.bundle_assets.masks[0].asset,
                 "unread_feature": packet.bundle_assets.features[2].asset}[target]
        path = Path(asset.path)
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        with pytest.raises(assets.AssetError):
            packet.guard.final_bindings()
        assert b.expected == packet.expected


def test_alias_retarget_after_bundle_load_blocks_raw_before_decode(packet, monkeypatch):
    alias = packet.root / "raw_alias.pkl"
    alias.symlink_to(packet.root / "raw0.pkl")
    packet.aliases["feature0"] = str(alias)
    packet.publish()
    with packet.guard:
        b = packet.load()
        replacement = packet.root / "replacement.pkl"
        replacement.write_bytes(packet.feature_bytes[0])
        alias.unlink()
        alias.symlink_to(replacement)
        monkeypatch.setattr(reader.mosaic, "decode_feature", lambda *args: pytest.fail("retargeted alias decoded"))
        with pytest.raises(assets.AssetError):
            reader.load_raw_slide(packet.guard, b, 0)


def test_npy_header_io_never_copies_whole_body(packet, monkeypatch):
    packet.publish()
    original = reader.io.BytesIO
    observed = []
    def bounded(data=b""):
        observed.append(len(data))
        assert len(data) <= 65545
        return original(data)
    monkeypatch.setattr(reader.io, "BytesIO", bounded)
    monkeypatch.setattr(reader.np, "load", lambda *args, **kwargs: pytest.fail("np.load forbidden"))
    with packet.guard:
        packet.load()
    assert observed


def test_extent_rejection_precedes_array_view_allocation(packet, monkeypatch):
    packet.files["coordinates.npy"] = manual_npy((2**63, 2))
    packet.publish()
    monkeypatch.setattr(reader.NpyBacking, "array_view", lambda *args: pytest.fail("extent must precede view"))
    with packet.guard:
        with pytest.raises(ValueError, match="extent"):
            packet.load()


def test_raw_assets_not_retained_in_bundle_and_selected_route_is_not_implicit(packet, monkeypatch):
    packet.publish()
    calls = []
    original = reader.mosaic.decode_feature
    def capture(data, feature):
        calls.append(feature["slide_id"])
        return original(data, feature)
    monkeypatch.setattr(reader.mosaic, "decode_feature", capture)
    with packet.guard:
        bundle = packet.load()
        assert calls == []
        assert not any(f.name == "raw" for f in fields(bundle))
        a = reader.load_raw_slide(packet.guard, bundle, 0)
        b = reader.load_raw_slide(packet.guard, bundle, 0)
        assert calls == ["s0", "s0"] and a.raw is not b.raw
        assert len(a.raw) == bundle.slides[0].feature_rows


def test_read_verified_return_cannot_be_success_shaped_wrong_bytes(packet):
    packet.publish()
    with packet.guard:
        class WrongBytes:
            def read_verified(self, identity):
                packet.guard.read_verified(identity)
                return b"{}"
        with pytest.raises(ValueError, match="returned bytes"):
            packet.load(WrongBytes())


@pytest.mark.parametrize("descr", ["=i8", "i8", "|i8"])
def test_coordinate_header_requires_literal_little_endian_descriptor(packet, descr):
    packet.files["coordinates.npy"] = manual_npy((6, 2), descr, body=packet.xy.tobytes())
    rejected(packet, "dtype|descriptor")


@pytest.mark.parametrize("descr", ["bool", "bool_"])
def test_mask_header_requires_literal_boolean_descriptor(packet, descr):
    values = np.array([True, False, False, False, False, True], dtype=np.bool_)
    packet.files["fold0_budget5_mask.npy"] = manual_npy((6,), descr, body=values.tobytes())
    rejected(packet, "dtype|descriptor")


def test_selected_input_order_does_not_define_returned_view_order(packet):
    packet.selected = tuple(reversed(packet.selected))
    packet.publish()
    with packet.guard:
        b = packet.load()
        assert [asdict(m.view) for m in b.masks] == [
            dict(held_out_fold=0, budget_percent=5), dict(held_out_fold=0, budget_percent=100)]


def test_producer_head_list_order_is_not_json_iteration_order(packet):
    packet.spec["heads"].reverse()
    packet.publish()
    with packet.guard:
        b = packet.load()
        heads = [json.loads(r.details_json) for r in b.referenced_only if r.kind == "head"]
        assert [(h["held_out_fold"], h["budget_percent"]) for h in heads] == list(reader.VIEW_ORDER)


def test_npy_header_duplicate_dictionary_keys_rejected(packet):
    text = "{'descr':'<i8', 'descr':'<i8', 'fortran_order':False, 'shape':(6,2)}"
    padding = 16 - (10 + len(text) + 1) % 16
    header = (text + " " * padding + "\n").encode()
    packet.files["coordinates.npy"] = b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + packet.xy.tobytes()
    rejected(packet, "header")


def test_all_ten_selected_views_have_named_population_ordered_counts(packet):
    packet.selected = tuple(reversed(reader.VIEW_ORDER))
    packet.publish()
    with packet.guard:
        b = packet.load()
        assert [(m.view.held_out_fold, m.view.budget_percent) for m in b.masks] == list(reader.VIEW_ORDER)
        for mask in b.masks:
            assert [r.slide_index for r in mask.counts] == [0, 1, 2]
            assert [r.slide_id for r in mask.counts] == ["s0", "s1", "s2"]
        packet.guard.final_bindings()


@pytest.mark.parametrize("which", ["namespace", "role", "cap", "relative"])
def test_feature_descriptor_pre_read_admission(which):
    a = assets.AssetSpec("feature:s0", "mosaic_feature", "/synthetic/raw", sha(b""), 0)
    changed = {
        "namespace": dict(asset_id="head:s0"), "role": dict(role="head_model"),
        "cap": dict(size=64 * reader.MIB + 1), "relative": dict(path="raw.pkl"),
    }[which]
    with pytest.raises(ValueError):
        reader.FeatureAsset(0, replace(a, **changed))


def test_out_of_population_feature_index_rejected_before_control_reads(packet):
    packet.publish()
    a = replace(packet.bundle_assets, features=(replace(packet.bundle_assets.features[0], slide_index=3),))
    class NoReads:
        def read_verified(self, identity):
            pytest.fail("out-of-range declaration must fail before control read")
    with pytest.raises(ValueError, match="expected population"):
        reader.load_mosaic_bundle(NoReads(), bundle_assets=a, expected=packet.expected)


def test_swapped_feature_ids_rejected_without_raw_acquisition(packet):
    packet.publish()
    first, second, third = packet.bundle_assets.features
    a = replace(packet.bundle_assets, features=(replace(first, asset=second.asset),
                                                replace(second, asset=first.asset), third))
    with packet.guard:
        public = PublicOnlyGuard(packet.guard)
        with pytest.raises(ValueError, match="ID/role"):
            reader.load_mosaic_bundle(public, bundle_assets=a, expected=packet.expected)
        assert not any(i.startswith("feature:") for i in object.__getattribute__(public, "reads"))


@pytest.mark.parametrize("field", [n for n in reader.COUNT_HEADER if n not in ("slide_id", "patient_id", "status")])
@pytest.mark.parametrize("value", [True, 1.0])
def test_all_count_integer_fields_are_strict(packet, field, value):
    original = reader.MaskCountRecord(**dict(zip(reader.COUNT_HEADER, packet.count_rows[0])))
    with pytest.raises(ValueError):
        replace(original, **{field: value})


@pytest.mark.parametrize("which", ["coordinates", "mask"])
def test_selected_array_descriptor_limit_checked_before_guard(which):
    a = assets.AssetSpec("mosaic:coordinates", "mosaic_coordinates", "/synthetic/coordinates", sha(b""), 64 * reader.MIB + 1)
    if which == "mask":
        a = assets.AssetSpec("mosaic:mask:fold0:budget5", "mosaic_mask", "/synthetic/mask", sha(b""), 8 * reader.MIB + 1)
        with pytest.raises(ValueError):
            reader.MaskAsset(reader.HeadView(0, 5), a)
    else:
        with pytest.raises(ValueError):
            reader._asset(a, "mosaic:coordinates", "mosaic_coordinates", 64 * reader.MIB)


@pytest.mark.parametrize("constant", ["OFFSET_EXPANSION_LIMIT", "COUNT_EXPANSION_LIMIT"])
def test_bounded_csv_expansion_failures_are_visible(packet, monkeypatch, constant):
    packet.publish()
    monkeypatch.setattr(reader, constant, 16)
    with packet.guard:
        with pytest.raises(ValueError, match="expanded"):
            packet.load()


def test_header_admission_copies_at_most_bounded_prefix(monkeypatch):
    body = np.zeros((5000, 2), dtype="<i8")
    data = npy(body)
    assert len(data) > 65545
    original = reader.io.BytesIO
    lengths = []
    def observe(value):
        lengths.append(len(value))
        return original(value)
    monkeypatch.setattr(reader.io, "BytesIO", observe)
    offset, dtype, shape, order = reader._npy_header(data)
    backing = reader.NpyBacking(data, offset, dtype, shape, order)
    assert backing.encoded is data and lengths == [65545, 65545]
    assert backing.array_view().shape == (5000, 2)


@pytest.mark.parametrize("which", ["list_shape", "bool_shape", "float_stride", "bool_flag", "list_checks", "noncanonical_reference"])
def test_remaining_frozen_metadata_types(which):
    with pytest.raises(ValueError):
        if which == "list_shape":
            reader.ArrayLayout("<f4", [2, 1536], (6144, 4), True, False)
        elif which == "bool_shape":
            reader.ArrayLayout("<f4", (True, 1536), (6144, 4), True, False)
        elif which == "float_stride":
            reader.ArrayLayout("<f4", (2, 1536), (6144.0, 4), True, False)
        elif which == "bool_flag":
            reader.ArrayLayout("<f4", (2, 1536), (6144, 4), 1, False)
        elif which == "list_checks":
            reader.ConsumptionRecord("x", "x", "/x", 0, sha(b""), ["check"])
        else:
            reader.ReferenceRecord("source", "x", '{"x":1}')


def test_every_public_record_has_exact_contract_fields():
    expected = {
        "HeadView": "held_out_fold budget_percent",
        "MosaicCounts": "slides patients diagnostic_groups union_patients feature_files original_tiles heads slide_head_cells tile_head_predictions",
        "MosaicExpected": "review_sha256 complete_sha256 spec_sha256 run_id counts",
        "MaskAsset": "view asset",
        "FeatureAsset": "slide_index asset",
        "MosaicBundleAssets": "review complete spec summary input_bindings population offsets mask_counts coordinates masks features",
        "SlideRecord": "slide_index metadata_index slide_id patient_id diagnostic_group slide_fold tile_start tile_stop feature_rows",
        "FeatureRecord": "slide_index slide_id producer_path producer_resolved_path size sha256 rows dtype columns consumer_asset",
        "NpyBacking": "encoded data_offset dtype_str shape fortran_order",
        "LoadedMask": "view backing counts",
        "MosaicBundle": "expected slides feature_records coordinates masks output_registry consumed_controls consumed_arrays referenced_only",
        "RawSlide": "slide raw coordinates layout consumption",
        "ArrayLayout": "dtype_str shape strides c_contiguous f_contiguous",
        "ArtifactDescriptor": "name size sha256",
        "ConsumptionRecord": "asset_id logical_name consumer_path producer_size producer_sha256 semantic_checks",
        "ReferenceRecord": "kind logical_name details_json",
    }
    for name, columns in expected.items():
        assert [f.name for f in fields(getattr(reader, name))] == columns.split()
        assert getattr(reader, name).__dataclass_params__.frozen
