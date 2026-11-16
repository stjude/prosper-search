"""Only materialized synthetic images/readers/bridges; no payload or model IO."""

from dataclasses import FrozenInstanceError, replace
from importlib import metadata
import random

import numpy as np
from PIL import Image, ImageEnhance
import pytest
import torch
from torch.nn import functional as TF
import torchvision
from torchvision.transforms import functional as F

from prosper.evaluation import r2_1_hoptimus_pixels as pixels
from prosper.evaluation import r2_1_hoptimus_perturb as core
from prosper.evaluation.r2_1_hoptimus_perturb import StimulusRecord


DIGEST = "a" * 64
KEY = ("synthetic-slide", 0, 3, 9, 512)


def record(condition="identity", **parameters):
    if not parameters:
        parameters = {
            "identity": {"treatment": "untreated"},
            "rotate_90": {"angle": 90},
            "rotate_180": {"angle": 180},
            "rotate_270": {"angle": 270},
            "flip_horizontal": {"axis": "horizontal"},
            "flip_vertical": {"axis": "vertical"},
            "brightness": {"factor": 0.85},
            "saturation": {"factor": 1.15},
            "contrast": {"factor": 0.9},
            "spatial_shift": {"x_shift": -16, "y_shift": -16},
            "reinhard": {"template_sha256": DIGEST},
            "gaussian_blur": {
                "kernel_size": 3, "sigma": 0.5,
                "application_stage": "post_preprocessing",
            },
        }[condition]
    return StimulusRecord(KEY[0], condition, 0, 512, tuple(parameters.items()))


@pytest.fixture
def image():
    y, x = np.indices((512, 512))
    array = np.stack(((3*x + y) % 256, (x + 5*y) % 256, (7*x + 11*y) % 256), axis=-1)
    return Image.fromarray(array.astype(np.uint8))


def normalized(image):
    resized = image.resize((224, 224), resample=Image.Resampling.BILINEAR)
    array = np.array(resized, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().to(torch.float32).div(255)
    mean = torch.tensor([0.707223, 0.578729, 0.703617]).reshape(3, 1, 1)
    std = torch.tensor([0.211883, 0.230117, 0.177517]).reshape(3, 1, 1)
    return (tensor - mean) / std


def exact(actual, expected):
    assert type(actual) is torch.Tensor
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.float32
    assert tuple(actual.shape) == (3, 224, 224)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_runtime_profile_records_distribution_and_separate_module_identity():
    profile = pixels.runtime_profile()
    assert [(m.distribution, m.distribution_version) for m in profile.modules] == [
        ("torch", "2.5.1"), ("torchvision", "0.20.1"), ("Pillow", "11.3.0"),
    ]
    for entry, module in zip(profile.modules, (torch, torchvision, Image)):
        assert entry.module_version == str(module.__version__)
        assert entry.module_path == module.__file__
    assert "compiled-code identity" in profile.limitations[0]
    with pytest.raises(FrozenInstanceError):
        profile.modules = ()


@pytest.mark.parametrize("distribution", ["torch", "torchvision", "Pillow"])
@pytest.mark.parametrize("replacement", ["99.0.0", None])
def test_wrong_or_missing_distribution_rejects_before_pixel_operations(
    image, monkeypatch, distribution, replacement,
):
    original = metadata.version
    def version(name):
        if name == distribution:
            if replacement is None:
                raise metadata.PackageNotFoundError(name)
            return replacement
        return original(name)
    monkeypatch.setattr(pixels.metadata, "version", version)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("pixel operation must not run")
    monkeypatch.setattr(Image.Image, "copy", forbidden)
    with pytest.raises(pixels.PixelValidationError):
        pixels.apply_pixels(image, record())
    assert not calls


def test_non_fp32_default_rejected_without_mutating_it(image):
    before = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(pixels.PixelValidationError, match="default floating dtype FP32"):
            pixels.apply_pixels(image, record())
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(before)


@pytest.mark.parametrize("condition", ["identity", "spatial_shift"])
def test_identity_shift_exact_resize_and_normalize(image, condition):
    before = image.tobytes()
    exact(pixels.apply_pixels(image, record(condition)), normalized(image))
    assert image.tobytes() == before


def test_rgba_drops_alpha_instead_of_white_compositing():
    rgba = Image.new("RGBA", (512, 512), (25, 90, 170, 0))
    before = rgba.tobytes()
    expected = normalized(rgba.convert("RGB"))
    actual = pixels.apply_pixels(rgba, record())
    exact(actual, expected)
    assert not torch.equal(actual, normalized(Image.new("RGB", (512, 512), "white")))
    assert rgba.mode == "RGBA" and rgba.tobytes() == before


@pytest.mark.parametrize("condition,expected", [
    ("identity", ["red", "green", "blue", "yellow"]),
    ("rotate_90", ["green", "yellow", "red", "blue"]),
    ("rotate_180", ["yellow", "blue", "green", "red"]),
    ("rotate_270", ["blue", "red", "yellow", "green"]),
    ("flip_horizontal", ["green", "red", "yellow", "blue"]),
    ("flip_vertical", ["blue", "yellow", "red", "green"]),
])
def test_512_landmarks_have_ccw_rotation_and_exact_flip_direction(condition, expected):
    colours = {"red": (255, 0, 0), "green": (0, 255, 0),
               "blue": (0, 0, 255), "yellow": (255, 255, 0)}
    source = Image.new("RGB", (512, 512))
    for colour, box in zip(
        ("red", "green", "blue", "yellow"),
        ((0, 0, 256, 256), (256, 0, 512, 256), (0, 256, 256, 512), (256, 256, 512, 512)),
    ):
        source.paste(colours[colour], box)
    before = source.tobytes()
    result = pixels.apply_pixels(source, record(condition))
    for colour, (y, x) in zip(expected, ((0, 0), (0, 223), (223, 0), (223, 223))):
        value = torch.tensor(colours[colour], dtype=torch.float32).div(255)
        wanted = (value - torch.tensor(pixels.MEAN)) / torch.tensor(pixels.STD)
        torch.testing.assert_close(result[:, y, x], wanted, rtol=0, atol=0)
    assert source.tobytes() == before


@pytest.mark.parametrize("condition,enhancer,factors", [
    ("brightness", ImageEnhance.Brightness, (0.85, 1.0, 1.15)),
    ("saturation", ImageEnhance.Color, (0.85, 1.0, 1.15)),
    ("contrast", ImageEnhance.Contrast, (0.9, 1.0, 1.1)),
])
def test_independent_pil_colour_conditions_and_rounded_inclusive_endpoints(
    image, condition, enhancer, factors,
):
    before = image.tobytes()
    for factor in factors:
        exact(pixels.apply_pixels(image, record(condition, factor=factor)),
              normalized(enhancer(image).enhance(factor)))
        assert image.tobytes() == before
    exact(pixels.apply_pixels(image, record()), normalized(image))


def test_blur_is_functional_reflect_and_after_normalization(image, monkeypatch):
    expected_input = normalized(image)
    calls = []
    original = F.gaussian_blur
    def blur(value, *, kernel_size, sigma):
        exact(value, expected_input)
        calls.append((kernel_size, sigma))
        return original(value, kernel_size=kernel_size, sigma=sigma)
    def forbidden(*args, **kwargs):
        raise AssertionError("GaussianBlur transform must not be constructed")
    monkeypatch.setattr(pixels.F, "gaussian_blur", blur)
    monkeypatch.setattr(torchvision.transforms, "GaussianBlur", forbidden)
    before = image.tobytes()
    actual = pixels.apply_pixels(image, record("gaussian_blur"))
    weights = torch.exp(-0.5 * (torch.tensor([-1., 0., 1.]) / 0.5).pow(2))
    weights /= weights.sum()
    kernel = torch.mm(weights[:, None], weights[None, :]).expand(3, 1, 3, 3)
    padded = TF.pad(expected_input.unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    expected = TF.conv2d(padded, kernel, groups=3)[0]
    exact(actual, expected)
    wrong_order = normalized(original(image, kernel_size=[3, 3], sigma=[0.5, 0.5]))
    assert not torch.equal(actual, wrong_order)
    assert calls == [([3, 3], [0.5, 0.5])]
    assert image.tobytes() == before


def test_read_plan_keeps_nominal_key_and_negative_actual_coordinates(image):
    stimulus = record("spatial_shift")
    plan = pixels.read_plan(KEY, stimulus)
    assert plan.nominal_key is KEY
    assert plan.actual_read == ((-13, -7), 0, (512, 512))
    with pytest.raises(FrozenInstanceError):
        plan.location = (0, 0)
    calls = []
    def reader(location, level, size):
        calls.append((location, level, size))
        return image
    returned_plan, tensor = pixels.read_and_apply(KEY, stimulus, reader)
    assert returned_plan == plan
    assert calls == [plan.actual_read]
    exact(tensor, normalized(image))


@pytest.mark.parametrize("condition", core.CONDITION_ORDER)
def test_only_shift_changes_read_origin_and_reader_is_called_once(image, condition):
    stimulus = record(condition)
    bridge = pixels.ReinhardBridge(DIGEST, lambda private: private)
    calls = []
    def reader(*args):
        calls.append(args)
        return image
    plan, tensor = pixels.read_and_apply(KEY, stimulus, reader, reinhard_bridge=bridge)
    expected = (-13, -7) if condition == "spatial_shift" else (3, 9)
    assert plan.nominal_key == KEY and plan.location == expected
    assert calls == [(expected, 0, (512, 512))]
    assert tensor.dtype == torch.float32 and tensor.shape == (3, 224, 224)


@pytest.mark.parametrize("bad_key", [
    None, list(KEY), KEY[:-1], KEY + (0,), ("other-slide", 0, 3, 9, 512),
    ("", 0, 3, 9, 512), (True, 0, 3, 9, 512),
    (KEY[0], False, 3, 9, 512), (KEY[0], 0.0, 3, 9, 512),
    (KEY[0], 1, 3, 9, 512), (KEY[0], 0, -1, 9, 512),
    (KEY[0], 0, 3, -1, 512), (KEY[0], 0, True, 9, 512),
    (KEY[0], 0, 3, False, 512), (KEY[0], 0, 3.0, 9, 512),
    (KEY[0], 0, 3, np.int64(9), 512), (KEY[0], 0, 3, 9, 224),
    (KEY[0], 0, 3, 9, 512.0), (KEY[0], 0, 3, 9, True),
])
def test_invalid_nominal_key_rejected_before_reader(bad_key):
    calls = []
    with pytest.raises(pixels.PixelValidationError):
        pixels.read_and_apply(bad_key, record(), lambda *args: calls.append(args))
    assert not calls


@pytest.mark.parametrize("changes", [
    {"slide_id": ""}, {"slide_id": None}, {"slide_id": True},
    {"slide_id": np.str_(KEY[0])}, {"level": True}, {"level": 1},
    {"level": 0.0}, {"level": np.int64(0)}, {"patch_size": 224},
    {"patch_size": True}, {"patch_size": 512.0}, {"condition": "unknown"},
    {"condition": True}, {"parameters": []}, {"parameters": {"treatment": "untreated"}},
    {"parameters": (("treatment", "untreated"), ("treatment", "untreated"))},
    {"parameters": (["treatment", "untreated"],)},
    {"parameters": (("treatment", "untreated", "extra"),)},
    {"parameters": ((42, "untreated"),)}, {"parameters": (([], "untreated"),)},
    {"parameters": ()}, {"parameters": (("treatment", "treated"),)},
    {"parameters": (("treatment", True),)},
    {"parameters": (("treatment", "untreated"), ("extra", 1))},
])
def test_forged_record_fields_reject_before_pixel_operations(image, monkeypatch, changes):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("malformed stimulus must not apply pixels")
    monkeypatch.setattr(Image.Image, "copy", forbidden)
    with pytest.raises(pixels.PixelValidationError):
        pixels.apply_pixels(image, replace(record(), **changes))
    assert not calls


@pytest.mark.parametrize("condition,parameters", [
    ("rotate_90", {"angle": 180}), ("rotate_90", {"angle": 90.0}),
    ("rotate_180", {"angle": True}), ("rotate_270", {"angle": np.int64(270)}),
    ("flip_horizontal", {"axis": "vertical"}), ("flip_vertical", {"axis": "horizontal"}),
    ("flip_horizontal", {"axis": True}), ("spatial_shift", {"x_shift": True, "y_shift": 0}),
    ("spatial_shift", {"x_shift": 0, "y_shift": 16.0}),
    ("spatial_shift", {"x_shift": -17, "y_shift": 0}),
    ("spatial_shift", {"x_shift": 0, "y_shift": 17}),
    ("spatial_shift", {"x_shift": 0}),
    ("spatial_shift", {"x_shift": 0, "y_shift": 0, "extra": 1}),
    ("gaussian_blur", {"kernel_size": 3.0, "sigma": 0.5, "application_stage": "post_preprocessing"}),
    ("gaussian_blur", {"kernel_size": True, "sigma": 0.5, "application_stage": "post_preprocessing"}),
    ("gaussian_blur", {"kernel_size": 3, "sigma": 1, "application_stage": "post_preprocessing"}),
    ("gaussian_blur", {"kernel_size": 3, "sigma": float("nan"), "application_stage": "post_preprocessing"}),
    ("gaussian_blur", {"kernel_size": 3, "sigma": 0.5, "application_stage": "pre_preprocessing"}),
    ("reinhard", {"template_sha256": "A" * 64}), ("reinhard", {"template_sha256": "a" * 63}),
    ("reinhard", {"template_sha256": "z" * 64}), ("reinhard", {"template_sha256": True}),
    ("reinhard", {"template_sha256": DIGEST, "extra": 0}),
])
def test_wrong_condition_parameters_rejected(condition, parameters):
    with pytest.raises(pixels.PixelValidationError):
        pixels.read_plan(KEY, record(condition, **parameters))


@pytest.mark.parametrize("condition", ["brightness", "saturation", "contrast"])
@pytest.mark.parametrize("factor", [
    float("nan"), float("inf"), -float("inf"), 0.84, 1.16,
    1, True, np.float64(1.0), "1.0", None,
])
def test_colour_bad_factor_types_nonfinite_and_outside_range(condition, factor):
    with pytest.raises(pixels.PixelValidationError):
        pixels.read_plan(KEY, record(condition, factor=factor))


@pytest.mark.parametrize("factor", [0.89, 1.11])
def test_contrast_has_its_own_narrower_range(factor):
    with pytest.raises(pixels.PixelValidationError):
        pixels.read_plan(KEY, record("contrast", factor=factor))


def test_duplicate_stochastic_parameters_are_not_collapsed():
    stimulus = replace(record("brightness"), parameters=(("factor", 0.85), ("factor", 0.85)))
    with pytest.raises(pixels.PixelValidationError, match="duplicate"):
        pixels.read_plan(KEY, stimulus)


def test_record_subclass_and_untyped_object_are_rejected():
    class SubRecord(StimulusRecord):
        pass
    for stimulus in (object(), SubRecord(KEY[0], "identity", 0, 512, (("treatment", "untreated"),))):
        with pytest.raises(pixels.PixelValidationError, match="StimulusRecord"):
            pixels.read_plan(KEY, stimulus)


class LazySentinel(Image.Image):
    def __init__(self, calls, *, materialized=False):
        super().__init__()
        self._mode = "RGB"
        self._size = (512, 512)
        self.calls = calls
        if materialized:
            self._im = Image.new("RGB", (512, 512))._im
            self.fp = object()

    def load(self, *args, **kwargs):
        self.calls.append("load")
        raise AssertionError("lazy sentinel must not load")

    convert = copy = resize = load


@pytest.mark.parametrize("route", ["direct", "reader", "bridge"])
@pytest.mark.parametrize("kind", ["lazy_subclass", "filebacked_subclass", "base_unmaterialized"])
def test_lazy_loader_never_runs_in_any_route(image, route, kind):
    calls = []
    if kind == "base_unmaterialized":
        invalid = Image.Image()
        invalid._mode = "RGB"
        invalid._size = (512, 512)
        def forbidden(*args, **kwargs):
            calls.append("load")
            raise AssertionError("unmaterialized base must not load")
        invalid.load = invalid.convert = invalid.copy = invalid.resize = forbidden
    else:
        invalid = LazySentinel(calls, materialized=kind == "filebacked_subclass")
    with pytest.raises(pixels.PixelValidationError, match="base PIL|materialized pixel core"):
        if route == "direct":
            pixels.apply_pixels(invalid, record())
        elif route == "reader":
            pixels.read_and_apply(KEY, record(), lambda *args: invalid)
        else:
            bridge = pixels.ReinhardBridge(DIGEST, lambda private: invalid)
            pixels.apply_pixels(image, record("reinhard"), reinhard_bridge=bridge)
    assert calls == []


@pytest.mark.parametrize("route", ["direct", "reader", "bridge"])
@pytest.mark.parametrize("kind", ["object", "wrong_core", "wrong_mode", "wrong_dimensions"])
def test_invalid_images_all_routes_raise_explicitly(image, route, kind):
    if kind == "object":
        invalid = object()
    elif kind == "wrong_core":
        invalid = Image.Image()
        invalid._mode = "RGB"
        invalid._size = (512, 512)
        invalid._im = object()
    elif kind == "wrong_mode":
        invalid = Image.new("L", (512, 512))
    else:
        invalid = Image.new("RGB", (511, 512))
    with pytest.raises(pixels.PixelValidationError):
        if route == "direct":
            pixels.apply_pixels(invalid, record())
        elif route == "reader":
            pixels.read_and_apply(KEY, record(), lambda *args: invalid)
        else:
            pixels.apply_pixels(image, record("reinhard"),
                                reinhard_bridge=pixels.ReinhardBridge(DIGEST, lambda private: invalid))


def test_bridge_rgba_return_is_not_coerced(image):
    rgba = Image.new("RGBA", (512, 512))
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid bridge return must not be converted")
    rgba.convert = forbidden
    with pytest.raises(pixels.PixelValidationError, match="wrong image mode"):
        pixels.apply_pixels(image, record("reinhard"),
                            reinhard_bridge=pixels.ReinhardBridge(DIGEST, lambda private: rgba))
    assert not calls


def test_reinhard_requires_bridge_and_matching_digest_before_reader(image):
    for bridge in (None, object(), pixels.ReinhardBridge("b" * 64, lambda private: private)):
        calls = []
        with pytest.raises(pixels.PixelValidationError):
            pixels.read_and_apply(KEY, record("reinhard"), lambda *args: calls.append(args),
                                  reinhard_bridge=bridge)
        assert not calls
        with pytest.raises(pixels.PixelValidationError):
            pixels.apply_pixels(image, record("reinhard"), reinhard_bridge=bridge)


@pytest.mark.parametrize("digest", [None, True, "", "A" * 64, "a" * 63, "g" * 64])
def test_invalid_bridge_digest_rejected_at_construction(digest):
    with pytest.raises(pixels.PixelValidationError):
        pixels.ReinhardBridge(digest, lambda private: private)


def test_bridge_is_immutable_and_revalidates_forged_fields(image):
    with pytest.raises(pixels.PixelValidationError, match="callable"):
        pixels.ReinhardBridge(DIGEST, None)
    bridge = pixels.ReinhardBridge(DIGEST, lambda private: private)
    with pytest.raises(FrozenInstanceError):
        bridge.template_sha256 = "b" * 64
    object.__setattr__(bridge, "transform", None)
    with pytest.raises(pixels.PixelValidationError, match="callable"):
        pixels.apply_pixels(image, record("reinhard"), reinhard_bridge=bridge)
    object.__setattr__(bridge, "transform", lambda private: private)
    object.__setattr__(bridge, "template_sha256", "invalid")
    with pytest.raises(pixels.PixelValidationError, match="digest"):
        pixels.apply_pixels(image, record("reinhard"), reinhard_bridge=bridge)


@pytest.mark.parametrize("rgba", [False, True])
def test_bridge_can_mutate_private_rgb_copy_without_mutating_original(image, rgba):
    source = image.convert("RGBA") if rgba else image
    before = source.tobytes()
    calls = []
    def transform(private):
        assert type(private) is Image.Image and private.mode == "RGB"
        assert private.size == (512, 512) and private is not source
        calls.append(private)
        private.paste((10, 20, 30), (0, 0, 512, 512))
        return private
    actual = pixels.apply_pixels(source, record("reinhard"),
                                 reinhard_bridge=pixels.ReinhardBridge(DIGEST, transform))
    exact(actual, normalized(Image.new("RGB", (512, 512), (10, 20, 30))))
    assert len(calls) == 1 and source.tobytes() == before
    assert not torch.equal(actual, normalized(source.convert("RGB")))


@pytest.mark.parametrize("rgba", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_bridge_nested_info_is_isolated_on_return_and_exception(image, rgba, raises):
    class OpaqueMetadata:
        def __deepcopy__(self, memo):
            raise AssertionError("arbitrary metadata deepcopy hook must not run")

    class BridgeFailure(RuntimeError):
        pass

    source = image.convert("RGBA") if rgba else image
    steps = ["original"]
    annotation = {"steps": steps}
    opaque = OpaqueMetadata()
    source.info = {"synthetic_annotation": annotation, "opaque": opaque}
    original_info = source.info
    before_pixels = source.tobytes()
    received_empty_info = []

    def transform(private):
        assert type(private) is Image.Image and private.mode == "RGB"
        assert private is not source and private.info is not original_info
        received_empty_info.append(not private.info)
        private.info.setdefault("synthetic_annotation", {"steps": []})["steps"].append("bridge")
        private.paste((10, 20, 30), (0, 0, 512, 512))
        if raises:
            raise BridgeFailure("synthetic bridge exception")
        return private

    bridge = pixels.ReinhardBridge(DIGEST, transform)
    if raises:
        with pytest.raises(BridgeFailure, match="synthetic bridge exception"):
            pixels.apply_pixels(source, record("reinhard"), reinhard_bridge=bridge)
    else:
        actual = pixels.apply_pixels(source, record("reinhard"), reinhard_bridge=bridge)
        exact(actual, normalized(Image.new("RGB", (512, 512), (10, 20, 30))))
    assert source.tobytes() == before_pixels
    assert source.info is original_info
    assert source.info == {"synthetic_annotation": {"steps": ["original"]}, "opaque": opaque}
    assert source.info["synthetic_annotation"] is annotation
    assert source.info["synthetic_annotation"]["steps"] is steps
    assert source.info["opaque"] is opaque
    assert received_empty_info == [True]


def test_trusted_callback_errors_propagate_without_fallback(image):
    class CallbackError(RuntimeError):
        pass
    def fail(*args):
        raise CallbackError("synthetic failure")
    with pytest.raises(CallbackError, match="synthetic failure"):
        pixels.read_and_apply(KEY, record(), fail)
    with pytest.raises(CallbackError, match="synthetic failure"):
        pixels.apply_pixels(image, record("reinhard"),
                            reinhard_bridge=pixels.ReinhardBridge(DIGEST, fail))
    with pytest.raises(pixels.PixelValidationError, match="callable"):
        pixels.read_and_apply(KEY, record(), None)


@pytest.mark.parametrize("stage", ["to_tensor", "normalize", "gaussian_blur"])
@pytest.mark.parametrize("kind", ["not_tensor", "wrong_shape", "wrong_dtype", "nan", "inf", "meta"])
def test_invalid_nonfinite_wrong_device_tensor_results_raise(image, monkeypatch, stage, kind):
    if kind == "not_tensor":
        invalid = np.zeros((3, 224, 224), dtype=np.float32)
    elif kind == "wrong_shape":
        invalid = torch.zeros((3, 223, 224))
    elif kind == "wrong_dtype":
        invalid = torch.zeros((3, 224, 224), dtype=torch.float64)
    elif kind in ("nan", "inf"):
        invalid = torch.full((3, 224, 224), float(kind))
    else:
        invalid = torch.empty((3, 224, 224), device="meta")
    monkeypatch.setattr(pixels.F, stage, lambda *args, **kwargs: invalid)
    with pytest.raises(pixels.PixelValidationError):
        pixels.apply_pixels(image, record("gaussian_blur"))


def test_realized_values_are_not_regenerated_and_rng_states_are_unchanged(image, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("adapter must not regenerate or draw parameters")
    monkeypatch.setattr(core, "_rng", forbidden)
    monkeypatch.setattr(core, "stimulus_parameter_key", forbidden)
    monkeypatch.setattr(core, "stimulus_parameter_seed", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "PCG64", forbidden)
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    torch_before = torch.get_rng_state().clone()
    dtype_before = torch.get_default_dtype()
    threads_before = torch.get_num_threads()
    bridge = pixels.ReinhardBridge(DIGEST, lambda private: private)
    for condition in core.CONDITION_ORDER:
        stimulus = record(condition)
        pixels.apply_pixels(image, stimulus, reinhard_bridge=bridge)
        pixels.read_and_apply(KEY, stimulus, lambda *args: image, reinhard_bridge=bridge)
    for x, y in ((-16, 16), (16, -16), (0, 0), (16, 16)):
        stimulus = record("spatial_shift", y_shift=y, x_shift=x)
        assert pixels.read_plan(KEY, stimulus).location == (3 + x, 9 + y)
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert random.getstate() == python_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert torch.get_default_dtype() == dtype_before
    assert torch.get_num_threads() == threads_before
    assert not torch.cuda.is_initialized()
