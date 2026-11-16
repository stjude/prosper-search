"""Pure synthetic fixtures; no package-wide model/image imports or real inputs."""

from dataclasses import FrozenInstanceError, replace
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import ModuleType

import numpy as np
import pytest


# Loading just this pure module avoids the legacy package's eager model imports.
_PATH = Path(__file__).resolve().parents[1] / "prosper/evaluation/r2_1_hoptimus_perturb.py"
_SPEC = importlib.util.spec_from_file_location("r2_1_hoptimus_perturb_synthetic", _PATH)
p = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = p
_SPEC.loader.exec_module(p)

TEMPLATE = "0123456789abcdef" * 4


def records(manifest):
    return {(r.slide_id, r.condition): r for r in manifest.records}


def features(values, dtype=np.float64):
    return np.repeat(np.asarray(values, dtype=dtype)[:, None], 1536, axis=1)


def test_manifest_order_definitions_and_metadata():
    manifest = p.generate_stimulus_manifest(["z", "a"], TEMPLATE)
    assert [(r.slide_id, r.condition) for r in manifest.records] == [
        (slide, condition) for slide in ("a", "z") for condition in p.CONDITION_ORDER
    ]
    assert len(manifest.records) == 24
    assert all(r.level == 0 and r.patch_size == 512 for r in manifest.records)
    keyed = records(manifest)
    assert dict(keyed["a", "identity"].parameters) == {"treatment": "untreated"}
    for angle in (90, 180, 270):
        params = dict(keyed["a", f"rotate_{angle}"].parameters)
        assert params == {"angle": angle}
        assert type(params["angle"]) is int
    for axis in ("horizontal", "vertical"):
        assert dict(keyed["a", f"flip_{axis}"].parameters) == {"axis": axis}
    assert dict(keyed["a", "reinhard"].parameters) == {"template_sha256": TEMPLATE}
    assert dict(keyed["a", "gaussian_blur"].parameters) == {
        "kernel_size": 3, "sigma": 0.5, "application_stage": "post_preprocessing"
    }
    assert manifest.policy.name == "r2_1_stimulus_v1"
    assert manifest.policy.seed == 1234
    assert manifest.policy.algorithm == "PCG64"
    assert manifest.policy.numpy_version == np.__version__
    with pytest.raises(FrozenInstanceError):
        manifest.records[0].patch_size = 224


def test_parameters_reproduce_exact_independent_full_digest_draws():
    manifest = p.generate_stimulus_manifest(["exact/id", "é"], TEMPLATE)
    keyed = records(manifest)
    keys, seeds = [], []
    for slide in ("exact/id", "é"):
        for condition, parameter in (
            ("brightness", "factor"), ("saturation", "factor"), ("contrast", "factor"),
            ("spatial_shift", "x_shift"), ("spatial_shift", "y_shift"),
        ):
            expected_key = json.dumps(
                ["r2_1_stimulus_v1", 1234, slide, condition, parameter],
                ensure_ascii=True, separators=(",", ":"),
            ).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(expected_key).digest(), "big")
            assert p.stimulus_parameter_key(slide, condition, parameter) == expected_key
            assert p.stimulus_parameter_seed(slide, condition, parameter) == seed
            rng = np.random.Generator(np.random.PCG64(seed))
            if condition == "spatial_shift":
                expected = int(rng.integers(-16, 17))
            else:
                low, high = (0.9, 1.1) if condition == "contrast" else (0.85, 1.15)
                expected = float(rng.uniform(low, high))
            assert dict(keyed[slide, condition].parameters)[parameter] == expected
            keys.append(expected_key)
            seeds.append(seed)
    assert len(set(keys)) == len(keys)
    assert len(set(seeds)) == len(seeds)
    assert max(seeds).bit_length() > 64


def test_order_chunk_resume_and_unrelated_slide_invariance():
    ids = ["002", "é", "001", "unrelated"]
    original = tuple(ids)
    whole = p.generate_stimulus_manifest(ids, TEMPLATE)
    assert p.serialize_stimulus_manifest(whole) == p.serialize_stimulus_manifest(
        p.generate_stimulus_manifest(ids[::-1], TEMPLATE)
    )
    keyed = records(whole)
    for chunk in (ids[:2], ids[2:], ["001"], ["é"]):
        partial = p.generate_stimulus_manifest(chunk, TEMPLATE)
        assert partial.policy == whole.policy
        assert all(keyed[r.slide_id, r.condition] == r for r in partial.records)
        assert partial == p.generate_stimulus_manifest(chunk, TEMPLATE)
    assert tuple(ids) == original


def test_ranges_types_and_slide_wide_parameter_unit():
    manifest = p.generate_stimulus_manifest([f"slide{i}" for i in range(30)], TEMPLATE)
    for record in manifest.records:
        params = dict(record.parameters)
        if record.condition in ("brightness", "saturation", "contrast"):
            low, high = (0.9, 1.1) if record.condition == "contrast" else (0.85, 1.15)
            assert type(params["factor"]) is float
            assert low <= params["factor"] <= high
        elif record.condition == "spatial_shift":
            assert all(type(v) is int and -16 <= v <= 16 for v in params.values())
    assert all(not hasattr(r, "tile_id") for r in manifest.records)


def test_equal_shift_draws_are_legal(monkeypatch):
    class EqualDraws:
        def integers(self, low, high):
            assert (low, high) == (-16, 17)
            return np.int64(0)

        def uniform(self, low, high):
            return low

    monkeypatch.setattr(p, "_rng", lambda *args: EqualDraws())
    manifest = p.generate_stimulus_manifest(["a"], TEMPLATE)
    assert dict(records(manifest)["a", "spatial_shift"].parameters) == {
        "x_shift": 0, "y_shift": 0
    }


def test_rounded_uniform_upper_endpoint_is_not_clipped_or_redrawn(monkeypatch):
    calls = []

    class EndpointDraws:
        def uniform(self, low, high):
            calls.append((low, high))
            return high

        def integers(self, low, high):
            return low

    monkeypatch.setattr(p, "_rng", lambda *args: EndpointDraws())
    keyed = records(p.generate_stimulus_manifest(["a"], TEMPLATE))
    assert dict(keyed["a", "brightness"].parameters)["factor"] == 1.15
    assert dict(keyed["a", "saturation"].parameters)["factor"] == 1.15
    assert dict(keyed["a", "contrast"].parameters)["factor"] == 1.1
    assert len(calls) == 3


def test_global_random_state_and_numpy_error_policy_unchanged():
    py_state = random.getstate()
    np_state = np.random.get_state()
    errors = np.geterr()
    p.generate_stimulus_manifest(["a", "b"], TEMPLATE)
    p.sampler_embedding(features([1, 2]))
    p.cityblock_distances(np.zeros(15360), np.ones((1, 15360)))
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert np_state[0] == after[0]
    np.testing.assert_array_equal(np_state[1], after[1])
    assert np_state[2:] == after[2:]
    assert np.geterr() == errors


def test_module_import_has_no_file_or_global_rng_side_effects(monkeypatch):
    source = compile(_PATH.read_text(), str(_PATH), "exec")
    isolated = ModuleType("r2_1_hoptimus_perturb_import_fixture")
    monkeypatch.setitem(sys.modules, isolated.__name__, isolated)

    def forbidden(*args, **kwargs):
        raise AssertionError("pure module import attempted IO or global RNG seeding")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(random, "seed", forbidden)
    monkeypatch.setattr(np.random, "seed", forbidden)
    exec(source, isolated.__dict__)
    assert isolated.CONDITION_ORDER == p.CONDITION_ORDER


@pytest.mark.parametrize("ids", ["abc", b"abc", None, [""], [1], [True], ["a", "a"]])
def test_invalid_slide_ids(ids):
    with pytest.raises(p.StimulusValidationError):
        p.generate_stimulus_manifest(ids, TEMPLATE)


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, "g" * 64, 1, None])
def test_invalid_template_digest(digest):
    with pytest.raises(p.StimulusValidationError):
        p.generate_stimulus_manifest(["a"], digest)


@pytest.mark.parametrize("condition,parameter", [
    ("identity", "factor"), ("brightness", "x_shift"),
    ("spatial_shift", "factor"), ("blur", "factor"), ("contrast", "FACTOR"),
])
def test_invalid_stochastic_key(condition, parameter):
    with pytest.raises(p.StimulusValidationError):
        p.stimulus_parameter_key("a", condition, parameter)


def test_exact_nonempty_ids_not_normalized_and_empty_population():
    manifest = p.generate_stimulus_manifest(["001", "1", " a ", "é"], TEMPLATE)
    assert {r.slide_id for r in manifest.records} == {"001", "1", " a ", "é"}
    assert p.generate_stimulus_manifest([], TEMPLATE).records == ()


def test_canonical_serialization_roundtrip_ascii_and_sorted_keys():
    manifest = p.generate_stimulus_manifest(["é"], TEMPLATE)
    encoded = p.serialize_stimulus_manifest(manifest)
    decoded = json.loads(encoded)
    assert b"\\u00e9" in encoded
    assert b" " not in p.canonical_json_bytes({"z": 1, "a": 2})
    assert p.canonical_json_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}'
    assert p.canonical_json_bytes(decoded) == encoded
    assert decoded["records"][0]["parameters"] == {"treatment": "untreated"}
    assert len(decoded["records"]) == 12


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_json_and_manifest_reject_nonfinite_parameters(value):
    with pytest.raises(p.StimulusValidationError):
        p.canonical_json_bytes({"nested": [value]})
    manifest = p.generate_stimulus_manifest(["a"], TEMPLATE)
    bad_record = replace(manifest.records[6], parameters=(("factor", value),))
    with pytest.raises(p.StimulusValidationError):
        p.serialize_stimulus_manifest(replace(manifest, records=(bad_record,)))


def altered_record(manifest, condition, **changes):
    return replace(manifest, records=tuple(
        replace(record, **changes) if record.condition == condition else record
        for record in manifest.records
    ))


@pytest.mark.parametrize("field,value", [
    ("name", "invented_policy"), ("seed", 999), ("seed", 1234.0),
    ("seed", True), ("algorithm", "MT19937"), ("key_encoding", "different encoding"),
    ("seed_derivation", "first 64 bits"), ("numpy_version", "invented version"),
])
def test_serialization_rejects_forged_policy(field, value):
    manifest = p.generate_stimulus_manifest(["a"], TEMPLATE)
    forged = replace(manifest, policy=replace(manifest.policy, **{field: value}))
    with pytest.raises(p.StimulusValidationError, match="policy"):
        p.serialize_stimulus_manifest(forged)


@pytest.mark.parametrize("changes", [
    {"level": True}, {"level": False}, {"level": 0.0}, {"level": 1},
    {"patch_size": 224}, {"patch_size": 512.0}, {"patch_size": True},
    {"patch_size": np.int64(512)}, {"condition": "unknown"},
    {"slide_id": ""}, {"slide_id": 1},
])
def test_serialization_rejects_forged_record_geometry_and_ids(changes):
    manifest = p.generate_stimulus_manifest(["a"], TEMPLATE)
    forged = replace(manifest, records=(
        replace(manifest.records[0], **changes), *manifest.records[1:],
    ))
    with pytest.raises(p.StimulusValidationError):
        p.serialize_stimulus_manifest(forged)


@pytest.mark.parametrize("condition,parameters", [
    ("identity", (("treatment", "changed"),)),
    ("rotate_90", (("angle", 90.0),)),
    ("rotate_90", (("angle", True),)),
    ("rotate_90", (("angle", 180),)),
    ("flip_horizontal", (("axis", "vertical"),)),
    ("brightness", (("factor", 5.0),)),
    ("brightness", (("factor", 1),)),
    ("saturation", (("factor", np.float64(1.0)),)),
    ("contrast", (("factor", float("nan")),)),
    ("contrast", (("factor", float("inf")),)),
    ("spatial_shift", (("x_shift", True), ("y_shift", 0))),
    ("spatial_shift", (("x_shift", 0.0), ("y_shift", 0))),
    ("spatial_shift", (("x_shift", 17), ("y_shift", 0))),
    ("gaussian_blur", (("kernel_size", 3.0), ("sigma", 0.5),
                       ("application_stage", "post_preprocessing"))),
    ("gaussian_blur", (("kernel_size", 3), ("sigma", 0.6),
                       ("application_stage", "post_preprocessing"))),
    ("gaussian_blur", (("kernel_size", 3), ("sigma", 0.5),
                       ("application_stage", "pre_preprocessing"))),
])
def test_serialization_rejects_forged_parameter_types_and_values(condition, parameters):
    manifest = p.generate_stimulus_manifest(["a"], TEMPLATE)
    with pytest.raises(p.StimulusValidationError):
        p.serialize_stimulus_manifest(altered_record(
            manifest, condition, parameters=parameters
        ))


@pytest.mark.parametrize("condition", ["brightness", "saturation", "contrast"])
def test_serialization_rejects_wrong_but_within_range_keyed_factor(condition):
    manifest = p.generate_stimulus_manifest(["a"], TEMPLATE)
    actual = dict(records(manifest)["a", condition].parameters)["factor"]
    low, high = (0.9, 1.1) if condition == "contrast" else (0.85, 1.15)
    wrong = low if actual != low else high
    assert low <= wrong <= high and wrong != actual
    with pytest.raises(p.StimulusValidationError, match="keyed"):
        p.serialize_stimulus_manifest(altered_record(
            manifest, condition, parameters=(("factor", wrong),)
        ))


def test_serialization_rejects_wrong_but_within_range_keyed_shift():
    manifest = p.generate_stimulus_manifest(["a"], TEMPLATE)
    original = dict(records(manifest)["a", "spatial_shift"].parameters)
    wrong = -16 if original["x_shift"] != -16 else 16
    with pytest.raises(p.StimulusValidationError, match="keyed"):
        p.serialize_stimulus_manifest(altered_record(
            manifest, "spatial_shift",
            parameters=(("x_shift", wrong), ("y_shift", original["y_shift"])),
        ))


@pytest.mark.parametrize("parameters", [
    (), (("unexpected", 1.0),), (("factor", 1.0), ("unexpected", 0)),
    (("factor", 1.0), ("factor", 1.0)),
    (("factor", float("nan")), ("factor", 1.0)),
    {"factor": 1.0}, [("factor", 1.0)], (["factor", 1.0],),
    (("factor",),), ((1, 1.0),), (("factor", object()),),
])
def test_serialization_rejects_parameter_name_collisions_and_malformed_pairs(parameters):
    manifest = p.generate_stimulus_manifest(["a"], TEMPLATE)
    with pytest.raises(p.StimulusValidationError):
        p.serialize_stimulus_manifest(altered_record(
            manifest, "brightness", parameters=parameters
        ))


def test_duplicate_parameter_would_otherwise_hide_a_finite_invalid_value():
    manifest = p.generate_stimulus_manifest(["a"], TEMPLATE)
    actual = dict(records(manifest)["a", "brightness"].parameters)["factor"]
    pairs = (("factor", 5.0), ("factor", actual))
    assert dict(pairs) == {"factor": actual}
    with pytest.raises(p.StimulusValidationError, match="duplicate"):
        p.serialize_stimulus_manifest(altered_record(
            manifest, "brightness", parameters=pairs
        ))


@pytest.mark.parametrize("index", range(12))
def test_serialization_rejects_every_missing_condition(index):
    manifest = p.generate_stimulus_manifest(["a"], TEMPLATE)
    incomplete = replace(
        manifest, records=manifest.records[:index] + manifest.records[index + 1:]
    )
    with pytest.raises(p.StimulusValidationError):
        p.serialize_stimulus_manifest(incomplete)


@pytest.mark.parametrize("mutation", [
    "duplicate_condition", "extra_condition", "condition_order",
    "slide_order", "duplicate_slide_block",
])
def test_serialization_rejects_duplicate_or_noncanonical_coverage(mutation):
    manifest = p.generate_stimulus_manifest(["a", "b"], TEMPLATE)
    rs = manifest.records
    if mutation == "duplicate_condition":
        altered = (rs[0], rs[0], *rs[2:])
    elif mutation == "extra_condition":
        altered = (*rs, rs[0])
    elif mutation == "condition_order":
        altered = (rs[1], rs[0], *rs[2:])
    elif mutation == "slide_order":
        altered = (*rs[12:], *rs[:12])
    else:
        altered = (*rs[:12], *rs[:12], *rs[12:])
    with pytest.raises(p.StimulusValidationError, match="complete|ordered|unique"):
        p.serialize_stimulus_manifest(replace(manifest, records=altered))


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64, 1])
def test_serialization_rejects_forged_template_digests(digest):
    manifest = p.generate_stimulus_manifest(["a"], TEMPLATE)
    with pytest.raises(p.StimulusValidationError):
        p.serialize_stimulus_manifest(altered_record(
            manifest, "reinhard", parameters=(("template_sha256", digest),)
        ))


def test_serialization_rejects_inconsistent_template_bindings_across_slides():
    manifest = p.generate_stimulus_manifest(["a", "b"], TEMPLATE)
    rs = tuple(
        replace(r, parameters=(("template_sha256", "a" * 64),))
        if r.slide_id == "b" and r.condition == "reinhard" else r
        for r in manifest.records
    )
    with pytest.raises(p.StimulusValidationError, match="consistent"):
        p.serialize_stimulus_manifest(replace(manifest, records=rs))


@pytest.mark.parametrize("part,value", [
    ("policy", {}), ("records", []), ("records", (object(),)),
])
def test_serialization_rejects_malformed_manifest_records(part, value):
    manifest = p.generate_stimulus_manifest(["a"], TEMPLATE)
    with pytest.raises(p.StimulusValidationError):
        p.serialize_stimulus_manifest(replace(manifest, **{part: value}))


def test_empty_generated_manifest_is_valid_but_still_checks_policy():
    empty = p.generate_stimulus_manifest([], TEMPLATE)
    assert json.loads(p.serialize_stimulus_manifest(empty))["records"] == []
    with pytest.raises(p.StimulusValidationError, match="policy"):
        p.serialize_stimulus_manifest(replace(
            empty, policy=replace(empty.policy, seed=999)
        ))


def test_semantic_serialization_validation_is_pure_and_preserves_manifest():
    manifest = p.generate_stimulus_manifest(["b", "a"], TEMPLATE)
    original = manifest
    python_state, numpy_state = random.getstate(), np.random.get_state()
    encoded = p.serialize_stimulus_manifest(manifest)
    assert manifest == original
    assert p.serialize_stimulus_manifest(manifest) == encoded
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]


def test_canonical_object_serialization_is_independent_of_parameter_pair_order():
    manifest = p.generate_stimulus_manifest(["a"], TEMPLATE)
    reordered = replace(manifest, records=tuple(
        replace(record, parameters=record.parameters[::-1])
        for record in manifest.records
    ))
    assert p.serialize_stimulus_manifest(reordered) == p.serialize_stimulus_manifest(manifest)


def test_sampler_precast_discriminator_and_dtype():
    tile_features = features([1.0002, 1.0008, 1.0021, 1.0043])
    original = tile_features.copy()
    expected = np.percentile(
        tile_features.astype(np.float16), p.PERCENTILES, axis=0, method="linear"
    ).ravel(order="C")
    actual = p.sampler_embedding(tile_features)
    uncast = np.percentile(tile_features, p.PERCENTILES, axis=0, method="linear").ravel()
    postcast = uncast.astype(np.float16)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == (15360,)
    assert not np.array_equal(actual, uncast)
    assert not np.array_equal(actual, postcast)
    np.testing.assert_array_equal(tile_features, original)


def test_sampler_component_and_percentile_major_flatten_order():
    tile_features = np.vstack([np.arange(1536), np.arange(1536) + 10]).astype(np.float32)
    pooled = p.sampler_embedding(tile_features)
    expected = np.percentile(
        tile_features.astype(np.float16), p.PERCENTILES, axis=0, method="linear"
    )
    np.testing.assert_array_equal(pooled.reshape(10, 1536), expected)
    assert pooled[0] == expected[0, 0]
    assert pooled[1536] == expected[1, 0]
    assert pooled[-1] == expected[-1, -1]
    assert not np.array_equal(pooled, expected.ravel(order="F"))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_sampler_single_tile_and_noncontiguous_input(dtype):
    one = features([1.25], dtype)
    np.testing.assert_array_equal(p.sampler_embedding(one), np.full(15360, 1.25))
    backing = np.zeros((4, 3072), dtype=dtype)
    backing[:, ::2] = features([1, 2, 4, 8], dtype)
    sliced = backing[:, ::2]
    assert not sliced.flags.c_contiguous
    np.testing.assert_array_equal(
        p.sampler_embedding(sliced), p.sampler_embedding(sliced.copy())
    )


@pytest.mark.parametrize("shape", [(0, 1536), (2, 1280), (2, 1537), (1536,), (2, 1, 1536)])
def test_sampler_rejects_empty_or_wrong_dimensions(shape):
    with pytest.raises(p.KernelValidationError):
        p.sampler_embedding(np.zeros(shape, dtype=np.float64))


@pytest.mark.parametrize("bad", [
    [[0.0] * 1536], np.zeros((1, 1536), dtype=int),
    np.zeros((1, 1536), dtype=bool), np.zeros((1, 1536), dtype=complex),
    np.zeros((1, 1536), dtype=object),
])
def test_sampler_rejects_nonfloating_arrays(bad):
    with pytest.raises(p.KernelValidationError):
        p.sampler_embedding(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_sampler_rejects_nonfinite_inputs(bad):
    tile_features = features([1])
    tile_features[0, -1] = bad
    with pytest.raises(p.KernelValidationError):
        p.sampler_embedding(tile_features)


def test_sampler_rejects_finite_fp16_cast_overflow_and_preserves_input():
    tile_features = features([70000])
    original = tile_features.copy()
    with pytest.raises(p.KernelComputationError, match="overflow|nonfinite"):
        p.sampler_embedding(tile_features)
    np.testing.assert_array_equal(tile_features, original)


def test_fp16_rounding_underflow_does_not_depend_on_external_error_policy():
    errors = np.geterr()
    with np.errstate(under="raise"):
        np.testing.assert_array_equal(
            p.sampler_embedding(features([1e-300])), np.zeros(15360)
        )
    assert np.geterr() == errors


def test_cityblock_known_answers_row_order_and_no_input_mutation():
    query = np.zeros(15360)
    query[:3] = [1, 2, 3]
    refs = np.zeros((3, 15360))
    refs[0, :3] = [1, 2, 3]
    refs[1, :3] = [4, 0, 2]
    refs[2, :3] = [-1, -2, -3]
    q_original, r_original = query.copy(), refs.copy()
    np.testing.assert_array_equal(p.cityblock_distances(query, refs), [0, 6, 12])
    np.testing.assert_array_equal(p.cityblock_distances(query, refs[[2, 0, 1]]), [12, 0, 6])
    np.testing.assert_array_equal(query, q_original)
    np.testing.assert_array_equal(refs, r_original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cityblock_no_extra_cast_and_empty_library(dtype):
    query = np.ones(15360, dtype=dtype)
    refs = np.zeros((1, 15360), dtype=dtype)
    assert p.cityblock_distances(query, refs).dtype == dtype
    empty = p.cityblock_distances(query, refs[:0])
    assert empty.shape == (0,)
    assert empty.dtype == dtype


@pytest.mark.parametrize("q_shape,r_shape", [
    ((1536,), (1, 15360)), ((1, 15360), (1, 15360)),
    ((15360,), (15360,)), ((15360,), (0, 1536)), ((15360,), (1, 1, 15360)),
])
def test_cityblock_bad_dimensions_even_when_reference_is_empty(q_shape, r_shape):
    with pytest.raises(p.KernelValidationError):
        p.cityblock_distances(np.zeros(q_shape), np.zeros(r_shape))


@pytest.mark.parametrize("dtype", [np.int64, bool, complex, object])
def test_cityblock_rejects_nonfloating_arrays(dtype):
    with pytest.raises(p.KernelValidationError):
        p.cityblock_distances(
            np.zeros(15360, dtype=dtype), np.zeros((0, 15360), dtype=dtype)
        )


def test_masked_arrays_cannot_hide_invalid_or_absent_values():
    tiles = np.ma.array(features([1]), mask=True)
    query = np.ma.array(np.zeros(15360), mask=True)
    with pytest.raises(p.KernelValidationError):
        p.sampler_embedding(tiles)
    with pytest.raises(p.KernelValidationError):
        p.cityblock_distances(query, np.zeros((1, 15360)))


@pytest.mark.filterwarnings("ignore:the matrix subclass is not the recommended way:PendingDeprecationWarning")
def test_matrix_tiles_are_rejected_with_specific_validation_error():
    tiles = np.matrix(np.ones((2, 1536)))
    with pytest.raises(p.KernelValidationError, match="base floating"):
        p.sampler_embedding(tiles)


@pytest.mark.filterwarnings("ignore:the matrix subclass is not the recommended way:PendingDeprecationWarning")
def test_matrix_references_cannot_produce_column_matrix_distances():
    refs = np.matrix(np.ones((2, 15360)))
    with pytest.raises(p.KernelValidationError, match="base floating"):
        p.cityblock_distances(np.zeros(15360), refs)


def test_other_ndarray_subclasses_are_explicitly_rejected():
    class ArraySubclass(np.ndarray):
        pass

    with pytest.raises(p.KernelValidationError, match="base floating"):
        p.sampler_embedding(np.ones((2, 1536)).view(ArraySubclass))
    with pytest.raises(p.KernelValidationError, match="base floating"):
        p.cityblock_distances(
            np.zeros(15360), np.ones((2, 15360)).view(ArraySubclass)
        )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_all_public_numeric_kernel_outputs_are_one_dimensional_base_arrays(dtype):
    pooled = p.sampler_embedding(features([1, 2], dtype))
    distances = p.cityblock_distances(
        np.zeros(15360, dtype=dtype), np.ones((2, 15360), dtype=dtype)
    )
    empty = p.cityblock_distances(
        np.zeros(15360, dtype=dtype), np.empty((0, 15360), dtype=dtype)
    )
    for result, shape in ((pooled, (15360,)), (distances, (2,)), (empty, (0,))):
        assert type(result) is np.ndarray
        assert result.ndim == 1 and result.shape == shape


@pytest.mark.parametrize("which", ["query", "reference"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_cityblock_rejects_nonfinite_inputs(which, value):
    query, refs = np.zeros(15360), np.zeros((1, 15360))
    if which == "query":
        query[0] = value
    else:
        refs[0, 0] = value
    with pytest.raises(p.KernelValidationError):
        p.cityblock_distances(query, refs)


def test_empty_reference_does_not_hide_invalid_query():
    query = np.zeros(15360)
    query[0] = np.nan
    with pytest.raises(p.KernelValidationError):
        p.cityblock_distances(query, np.empty((0, 15360)))


@pytest.mark.parametrize("operation", ["subtract", "reduce"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cityblock_finite_input_arithmetic_overflow(operation, dtype):
    maximum = np.finfo(dtype).max
    query = np.zeros(15360, dtype=dtype)
    refs = np.zeros((1, 15360), dtype=dtype)
    if operation == "subtract":
        query[0], refs[0, 0] = -maximum, maximum
    else:
        refs[0, :2] = maximum
    assert np.isfinite(query).all() and np.isfinite(refs).all()
    with pytest.raises(p.KernelComputationError, match="overflow|nonfinite"):
        p.cityblock_distances(query, refs)
