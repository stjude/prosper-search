"""Synthetic-only fixtures, runnable only with the independently verified private pin."""

from dataclasses import FrozenInstanceError, fields, replace
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.overrides import _get_current_function_mode_stack

from prosper.evaluation import r2_1_hoptimus_reinhard as stain
from prosper.evaluation.r2_1_hoptimus_pixels import ReinhardBridge, apply_pixels
from prosper.evaluation.r2_1_hoptimus_perturb import StimulusRecord


DIGEST = "a" * 64


def synthetic(width=32, height=24, variant=0):
    y, x = np.indices((height, width))
    array = np.stack(((3*x + y + 11*variant) % 190 + 25,
                      (x + 5*y + 17*variant) % 180 + 35,
                      (7*x + 11*y + 23*variant) % 170 + 40), axis=-1)
    return Image.fromarray(array.astype(np.uint8))


@pytest.fixture
def template():
    return synthetic(31, 23, 1)


@pytest.fixture
def query():
    return synthetic(27, 19, 2)


def fitted(template):
    return stain.make_reinhard_bridge(template, template_sha256=DIGEST)


def upstream(template, query):
    # No imports occur before the same preconditions/source closure as the owner.
    stain._execution_context()
    stain._candidate()
    cls = import_module("torchstain.torch.normalizers.reinhard").TorchReinhardNormalizer
    normalizer = cls(method=None)
    target = stain._working(np.array(template, copy=True, dtype=np.uint8))
    source = stain._working(np.array(query, copy=True, dtype=np.uint8))
    target_before, source_before = target.clone(), source.clone()
    normalizer.fit(target)
    result = normalizer.normalize(source)
    assert torch.equal(target, target_before) and torch.equal(source, source_before)
    return result.numpy().copy()


def test_pinned_all_package_members_and_actual_locations():
    locations = stain._member_locations()
    assert tuple(locations) == tuple(name for name, _, _ in stain.PACKAGE_MEMBERS)
    assert len(locations) == 49
    primitives = stain._candidate()
    assert primitives.rgb2lab.__module__ == "torchstain.torch.utils.rgb2lab"
    assert metadata.version("torchstain") == "1.4.1"
    assert import_module("torchstain").__version__ == "1.3.0"
    assert torch.get_num_threads() == 1
    assert all(os.environ[name] == "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    print("Verified synthetic runtime:", {
        "python": sys.version.split()[0],
        "package_python_members": len(locations),
        "package_root": str(locations["torchstain/__init__.py"].parent),
        "torch": (metadata.version("torch"), str(torch.__version__), torch.__file__),
        "numpy": (metadata.version("numpy"), np.__version__, np.__file__),
        "Pillow": (metadata.version("Pillow"), Image.__version__, Image.__file__),
        "default_dtype": str(torch.get_default_dtype()),
        "underlying_default_device": torch._C._get_default_device(),
        "cpu_autocast": torch.is_autocast_enabled("cpu"),
        "torch_threads": torch.get_num_threads(),
        "PROSPER_OUT": os.environ["PROSPER_OUT"],
    })


@pytest.mark.parametrize("variant", [0, 1, 2, 3])
def test_exact_direct_upstream_bytes_nonidentity_and_repetition(template, variant):
    query = synthetic(29, 17, variant)
    bridge, _ = fitted(template)
    expected = upstream(template, query)
    result = bridge.transform(query)
    assert type(result) is Image.Image and result.mode == "RGB" and result.size == query.size
    assert np.array_equal(np.array(result), expected)
    assert result.tobytes() != query.tobytes()
    assert bridge.transform(query).tobytes() == result.tobytes()


def test_exact_out_of_gamut_clipping_and_selected_negative_branch(template):
    primitives = stain._candidate()
    L = torch.tensor([[10., 90., 50., 60.], [20., 80., 45., 70.]])
    a = torch.tensor([[-120., 120., -100., 100.], [110., -110., 130., -130.]])
    b = torch.tensor([[120., -120., 100., -100.], [-100., 100., -120., 120.]])
    lab = primitives.lab_merge(L, a, b)
    actual = stain._lab2rgb_checked(lab, primitives)
    expected = primitives.inverse_module.lab2rgb(lab)
    assert torch.equal(actual, expected)
    assert bool((actual == 0).any()) and bool((actual == 1).any())
    assert torch.equal(stain._quantize(actual), (expected * 255).type(torch.uint8))
    source = Image.new("RGB", (3, 2))
    source.putdata([(0, 1, 254), (255, 0, 1), (1, 254, 255),
                    (254, 255, 0), (30, 180, 90), (210, 20, 150)])
    bridge, _ = fitted(template)
    assert np.array_equal(np.array(bridge.transform(source)), upstream(template, source))


def test_full_ordinary_upstream_oracle_with_finite_out_of_gamut_transfer(template):
    query = Image.new("RGB", (24, 24), (120, 130, 140))
    for x, colour in enumerate(((0, 0, 0), (255, 255, 255),
                                (255, 0, 0), (0, 255, 0), (0, 0, 255))):
        query.putpixel((x, 0), colour)
    primitives = stain._candidate()
    target_channels, target_stats = stain._lab_statistics(stain._image_data(template), primitives)
    channels, source_stats = stain._lab_statistics(stain._image_data(query), primitives)
    transferred_L = primitives.standardize(
        channels[0], source_stats[0, 0], source_stats[0, 1],
    ) * target_stats[0, 1] + target_stats[0, 0]
    assert bool(torch.isfinite(transferred_L).all())
    assert bool((transferred_L < 0).any()) and bool((transferred_L > 100).any())
    expected = upstream(template, query)
    actual = np.array(fitted(template)[0].transform(query))
    assert np.array_equal(actual, expected)
    assert bool((actual == 0).any()) and bool((actual == 255).any())


def test_positive_small_source_variance_is_not_thresholded(template):
    query = Image.new("RGB", (64, 64), (127, 127, 127))
    query.putpixel((0, 0), (128, 127, 127))
    primitives = stain._candidate()
    _, source_stats = stain._lab_statistics(stain._image_data(query), primitives)
    _, target_stats = stain._lab_statistics(stain._image_data(template), primitives)
    assert bool((source_stats[:, 1] > 0).all())
    assert bool((source_stats[:, 1] < target_stats[:, 1]).all())
    actual = np.array(fitted(template)[0].transform(query))
    assert np.array_equal(actual, upstream(template, query))


def test_channel_numeric_range_and_truncation_are_independent():
    image = Image.new("RGB", (3, 2))
    image.putdata([(0, 1, 254), (255, 0, 1), (1, 254, 255),
                   (254, 255, 0), (30, 180, 90), (210, 20, 150)])
    data = stain._image_data(image)
    working = stain._working(data)
    assert working.shape == (3, 2, 3) and working.dtype == torch.float32
    assert working[:, 0, 0].tolist() == [0., 1., 254.]
    working[0, 0, 0] = 99
    assert data[0, 0, 0] == 0 and image.getpixel((0, 0)) == (0, 1, 254)
    values = torch.tensor([[[0., 1., 254.5 / 255]]])
    assert stain._quantize(values).tolist() == [[[0, 255, 254]]]


def test_provenance_and_target_are_deeply_immutable_and_pixel_digest_exact(template, query):
    bridge, provenance = fitted(template)
    header = json.dumps({"mode": "RGB", "width": template.width, "height": template.height},
                        sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    assert provenance.decoded_pixel_sha256 == hashlib.sha256(
        header + b"\n" + np.array(template, dtype=np.uint8).tobytes(order="C"),
    ).hexdigest()
    assert provenance.encoded_template_sha256 == DIGEST
    assert provenance.encoded_digest_verified_by_adapter is False
    assert provenance.distribution_version == "1.4.1" and provenance.module_version == "1.3.0"
    assert provenance.backend == "torch" and provenance.method is None
    assert provenance.device == "cpu" and provenance.dtype == "float32"
    assert provenance.source_commit == stain.SOURCE_COMMIT
    assert provenance.package_members == stain.PACKAGE_MEMBERS
    assert provenance.library_versions == (
        ("torch", "2.5.1"), ("numpy", "2.2.4"), ("Pillow", "11.3.0"),
    )
    for field in fields(provenance):
        assert not isinstance(getattr(provenance, field.name), (dict, list, np.ndarray, torch.Tensor))
    assert isinstance(provenance.target_state_bytes, bytes) and len(provenance.target_state_bytes) == 24
    assert stain._state_hash(provenance.target_state_bytes) == provenance.target_state_sha256
    with pytest.raises(FrozenInstanceError):
        provenance.width = 4
    with pytest.raises(FrozenInstanceError):
        bridge.transform.target_state = b"broken"
    before = bridge.transform(query).tobytes()
    template.paste("white", (0, 0, template.width, template.height))
    assert bridge.transform(query).tobytes() == before


def test_target_fit_happens_once_and_working_tensors_are_private(template, query, monkeypatch):
    original = stain._lab_statistics
    observed = []
    def statistics(data, primitives):
        observed.append(data.shape)
        return original(data, primitives)
    monkeypatch.setattr(stain, "_lab_statistics", statistics)
    bridge, _ = fitted(template)
    for _ in range(3):
        bridge.transform(query)
    assert observed == [(template.height, template.width, 3)] + [
        (query.height, query.width, 3),
    ] * 3


def test_finite_target_zero_is_retained_but_source_zero_always_fails(template, query, monkeypatch):
    original = stain._lab_statistics
    def target_zero(data, primitives):
        channels, stats = original(data, primitives)
        stats[:, 1] = 0
        return channels, stats
    with monkeypatch.context() as m:
        m.setattr(stain, "_lab_statistics", target_zero)
        bridge, provenance = fitted(template)
    assert np.frombuffer(provenance.target_state_bytes, dtype="<f4")[3:].tolist() == [0., 0., 0.]
    assert bridge.transform(query).mode == "RGB"
    def source_zero(data, primitives):
        channels, stats = original(data, primitives)
        stats[0, 1] = 0
        return channels, stats
    with monkeypatch.context() as m:
        m.setattr(stain, "_lab_statistics", source_zero)
        with pytest.raises(stain.ReinhardValidationError, match="zero source"):
            bridge.transform(query)


def test_actual_constant_source_fails(template):
    bridge, _ = fitted(template)
    with pytest.raises(stain.ReinhardValidationError, match="zero source"):
        bridge.transform(Image.new("RGB", (32, 24), (0, 0, 0)))


def test_actual_finite_constant_target_is_a_valid_channel_collapse(query):
    bridge, provenance = fitted(Image.new("RGB", (12, 9), (0, 0, 0)))
    assert np.frombuffer(provenance.target_state_bytes, dtype="<f4")[3:].tolist() == [0., 0., 0.]
    assert np.array_equal(np.array(bridge.transform(query)), np.zeros(
        (query.height, query.width, 3), dtype=np.uint8,
    ))
    with pytest.raises(stain.ReinhardValidationError, match="zero source"):
        bridge.transform(Image.new("RGB", (12, 9), (0, 0, 0)))


@pytest.mark.parametrize("bad_std", [-1., float("nan"), float("inf")])
def test_invalid_target_statistics_raise_without_aliases(template, monkeypatch, bad_std):
    primitives = stain._candidate()
    original = primitives.get_mean_std
    fake = replace(primitives, get_mean_std=lambda value: (original(value)[0], torch.tensor(bad_std)))
    monkeypatch.setattr(stain, "_candidate", lambda: fake)
    with pytest.raises(stain.ReinhardValidationError, match="nonfinite|negative std"):
        fitted(template)


@pytest.mark.parametrize("point", ["inverse_lab", "matrix", "linear_gamma", "quantization"])
def test_nonfinite_preclamp_or_prequantization_is_not_hidden(monkeypatch, point):
    primitives = stain._candidate()
    lab = primitives.lab_merge(torch.ones((2, 2)) * 50, torch.zeros((2, 2)),
                               torch.zeros((2, 2)))
    if point == "inverse_lab":
        lab[0, 0, 0] = torch.finfo(torch.float32).max
    elif point in ("matrix", "linear_gamma"):
        original = torch.tensordot
        def tensordot(*args, **kwargs):
            result = original(*args, **kwargs)
            result[0, 0, 0] = float("inf") if point == "matrix" else -torch.finfo(torch.float32).max
            return result
        monkeypatch.setattr(torch, "tensordot", tensordot)
        hidden = primitives.inverse_module.lab2rgb(lab)
        assert bool(torch.isfinite(hidden).all())
        assert hidden[0, 0, 0].item() == (1.0 if point == "matrix" else 0.0)
    else:
        with pytest.raises(stain.ReinhardValidationError, match="before original uint8"):
            stain._quantize(torch.full((2, 2, 3), float("inf")))
        return
    with pytest.raises(stain.ReinhardValidationError, match="nonfinite"):
        stain._lab2rgb_checked(lab, primitives)


@pytest.mark.parametrize("role", ["template", "query"])
@pytest.mark.parametrize("kind", ["one_pixel", "mode", "unmaterialized", "lazy", "wrong_core"])
def test_invalid_images_rejected_without_loading(template, role, kind):
    calls = []
    class Lazy(Image.Image):
        def load(self, *args, **kwargs):
            calls.append(True)
            raise AssertionError("must not load")
    if kind == "one_pixel":
        invalid = Image.new("RGB", (1, 1))
    elif kind == "mode":
        invalid = Image.new("RGBA", (4, 4))
    elif kind in ("unmaterialized", "wrong_core"):
        invalid = Image.Image()
        invalid._size, invalid._mode = (4, 4), "RGB"
        if kind == "wrong_core":
            invalid._im = object()
        invalid.load = lambda: calls.append(True)
    else:
        invalid = Lazy()
    with pytest.raises((stain.ReinhardValidationError, ValueError)):
        if role == "template":
            fitted(invalid)
        else:
            fitted(template)[0].transform(invalid)
    assert not calls


@pytest.mark.parametrize("digest", ["A"*64, "a"*63, True, None, "g"*64])
def test_invalid_encoded_digest_rejected(template, digest):
    with pytest.raises(stain.ReinhardValidationError, match="encoded-template"):
        stain.make_reinhard_bridge(template, template_sha256=digest)


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("role", ["template", "query"])
def test_original_pixels_and_nested_metadata_survive_failure_or_cancellation(
    template, query, monkeypatch, role, exception,
):
    image = template if role == "template" else query
    steps = ["original"]
    image.info = {"synthetic_annotation": {"steps": steps}}
    old_info = image.info
    old_pixels = image.tobytes()
    bridge = fitted(template)[0] if role == "query" else None
    def fail(data, primitives):
        data[:] = 0
        raise exception("synthetic failure")
    monkeypatch.setattr(stain, "_lab_statistics", fail)
    with pytest.raises(exception, match="synthetic failure"):
        if bridge is None:
            fitted(template)
        else:
            bridge.transform(query)
    assert image.tobytes() == old_pixels and image.info is old_info
    assert image.info == {"synthetic_annotation": {"steps": ["original"]}}
    assert image.info["synthetic_annotation"]["steps"] is steps


def test_bridge_pixel_contract_integration_preserves_pixels_and_info(template):
    query = synthetic(512, 512, 3)
    query.info = {"synthetic_annotation": {"steps": ["original"]}}
    before = query.tobytes()
    bridge, _ = fitted(template)
    record = StimulusRecord("synthetic", "reinhard", 0, 512, (("template_sha256", DIGEST),))
    tensor = apply_pixels(query, record, reinhard_bridge=bridge)
    assert type(bridge) is ReinhardBridge and tensor.shape == (3, 224, 224)
    assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
    assert query.tobytes() == before
    assert query.info == {"synthetic_annotation": {"steps": ["original"]}}


def test_template_and_query_metadata_and_pixels_are_unchanged_on_success(template, query):
    class Opaque:
        def __deepcopy__(self, memo):
            raise AssertionError("metadata hooks must not be invoked")
    snapshots = []
    for image in (template, query):
        image.info = {"synthetic_annotation": {"steps": ["original"]}, "opaque": Opaque()}
        snapshots.append((image.tobytes(), image.info, image.info["synthetic_annotation"]["steps"],
                          image.info["opaque"]))
    bridge, _ = fitted(template)
    output = bridge.transform(query)
    output.paste("white", (0, 0, output.width, output.height))
    output.info["new"] = True
    for image, (pixels, info, steps, opaque) in zip((template, query), snapshots):
        assert image.tobytes() == pixels and image.info is info
        assert image.info["synthetic_annotation"]["steps"] is steps and steps == ["original"]
        assert image.info["opaque"] is opaque and "new" not in image.info


@pytest.mark.parametrize("phase", ["factory", "cached_bridge"])
@pytest.mark.parametrize("context", ["float64", "meta_context", "global_meta", "autocast", "cpu_context"])
def test_context_guard_before_arithmetic_preserves_caller(template, query, monkeypatch, phase, context):
    bridge = fitted(template)[0] if phase == "cached_bridge" else None
    calls = []
    original = stain._lab_statistics
    def arithmetic(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(stain, "_lab_statistics", arithmetic)
    dtype = torch.get_default_dtype()
    global_context = getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", None)
    if context == "float64":
        torch.set_default_dtype(torch.float64)
        manager = None
    elif context == "global_meta":
        torch.set_default_device("meta")
        manager = None
    elif context == "autocast":
        manager = torch.autocast("cpu")
    else:
        manager = torch.device("cpu" if context == "cpu_context" else "meta")
    def run():
        stack = tuple(_get_current_function_mode_stack())
        ambient_dtype = torch.get_default_dtype()
        autocast = torch.is_autocast_enabled("cpu")
        if context == "cpu_context":
            fitted(template) if bridge is None else bridge.transform(query)
            assert calls
        else:
            with pytest.raises(stain.ReinhardValidationError, match="dtype|CPU|autocast"):
                fitted(template) if bridge is None else bridge.transform(query)
            assert not calls
        assert tuple(_get_current_function_mode_stack()) == stack
        assert torch.get_default_dtype() == ambient_dtype
        assert torch.is_autocast_enabled("cpu") == autocast
    try:
        if manager is None:
            run()
        else:
            with manager:
                run()
    finally:
        if context == "float64":
            torch.set_default_dtype(dtype)
        elif context == "global_meta":
            previous = None if global_context is None else global_context.device
            torch.set_default_device(previous)


@pytest.mark.parametrize("device_type", ["cuda", "meta", "xpu"])
def test_stored_accelerator_guard_never_probes_or_creates_tensor(template, monkeypatch, device_type):
    calls = []
    monkeypatch.setattr(torch._C, "_get_default_device", lambda: device_type)
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("must not create/probe a tensor")
    monkeypatch.setattr(torch, "tensor", forbidden)
    monkeypatch.setattr(torch, "get_default_device", forbidden)
    with pytest.raises(stain.ReinhardValidationError, match="underlying default"):
        fitted(template)
    assert not calls


def test_mocked_accelerator_device_context_before_candidate_import(template, monkeypatch):
    context = object.__new__(stain.DeviceContext)
    context.device = torch.device("cuda")
    monkeypatch.setattr(stain, "_get_current_function_mode_stack", lambda: [context])
    calls = []
    monkeypatch.setattr(stain, "import_module", lambda *args: calls.append(args))
    with pytest.raises(stain.ReinhardValidationError, match="DeviceContext"):
        fitted(template)
    assert not calls and context.device.type == "cuda"


@pytest.mark.parametrize("context", ["float64", "meta", "autocast"])
def test_fresh_process_rejects_before_initial_candidate_import(context):
    script = r'''
import sys,torch
from prosper.evaluation import r2_1_hoptimus_reinhard as s
from PIL import Image
assert not any(n == "torchstain" or n.startswith("torchstain.") for n in sys.modules)
kind=sys.argv[1]
if kind=="float64":
    torch.set_default_dtype(torch.float64)
    manager=None
elif kind=="meta":
    manager=torch.device("meta")
else:
    manager=torch.autocast("cpu")
def run():
    try:
        s.make_reinhard_bridge(Image.new("RGB",(2,2)),template_sha256="a"*64)
    except s.ReinhardValidationError:
        pass
    else:
        raise AssertionError("bad ambient context was accepted")
    assert not any(n == "torchstain" or n.startswith("torchstain.") for n in sys.modules)
if manager is None:
    run()
else:
    with manager:
        run()
'''
    process = subprocess.run([sys.executable, "-c", script, context],
                             env=os.environ.copy(), capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr


def test_fresh_cwd_shadow_rejected_before_initializer_executes(tmp_path):
    shadow = tmp_path / "torchstain"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(
        "from pathlib import Path\n"
        "Path('shadow_initializer_executed').write_text('sentinel')\n"
        "raise RuntimeError('SHADOW_INITIALIZER_EXECUTED')\n",
        encoding="ascii",
    )
    script = r'''
import sys
from importlib import metadata
from PIL import Image
from prosper.evaluation import r2_1_hoptimus_reinhard as s
assert not any(n == "torchstain" or n.startswith("torchstain.") for n in sys.modules)
assert metadata.version("torchstain") == "1.4.1"
paths_before = tuple(sys.path)
try:
    s.make_reinhard_bridge(Image.new("RGB", (2,2)), template_sha256="a"*64)
except s.ReinhardValidationError as exc:
    assert "import specification" in str(exc), str(exc)
else:
    raise AssertionError("cwd shadow accepted")
assert tuple(sys.path) == paths_before
assert not any(n == "torchstain" or n.startswith("torchstain.") for n in sys.modules)
'''
    environment = os.environ.copy()
    process = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                             env=environment, capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
    assert not (tmp_path / "shadow_initializer_executed").exists()
    assert "SHADOW_INITIALIZER_EXECUTED" not in process.stdout + process.stderr


def test_target_or_conversion_constant_mutation_rejected(template, query):
    bridge, _ = fitted(template)
    transform = bridge.transform
    object.__setattr__(transform, "target_state", b"\0"*24)
    with pytest.raises(stain.ReinhardValidationError, match="target state changed"):
        transform(query)
    bridge, _ = fitted(template)
    primitives = stain._candidate()
    matrix = primitives.rgb_module._rgb2xyz
    before = matrix.clone()
    try:
        matrix[0, 0] += 0.01
        with pytest.raises(stain.ReinhardValidationError, match="constants changed"):
            bridge.transform(query)
    finally:
        matrix.copy_(before)


def test_wrong_loaded_candidate_location_rejected(template, monkeypatch):
    stain._candidate()
    module = import_module("torchstain.torch.utils.rgb2lab")
    monkeypatch.setattr(module, "__file__", "/synthetic/not-the-verified-file.py")
    with pytest.raises(stain.ReinhardValidationError, match="location mismatch"):
        fitted(template)


@pytest.mark.parametrize("mismatch", ["origin", "foreign_search", "not_package"])
def test_cached_root_import_spec_mismatch_rejected_before_import(template, monkeypatch, mismatch):
    stain._candidate()
    root = import_module("torchstain")
    if mismatch == "origin":
        monkeypatch.setattr(root.__spec__, "origin", "/synthetic/foreign/__init__.py")
    else:
        search = None if mismatch == "not_package" else ["/synthetic/foreign"]
        monkeypatch.setattr(root.__spec__, "submodule_search_locations", search)
    imports = []
    monkeypatch.setattr(stain, "import_module", lambda *args: imports.append(args))
    with pytest.raises(stain.ReinhardValidationError, match="import specification|location mismatch"):
        fitted(template)
    assert not imports


@pytest.mark.parametrize("phase", ["factory", "cached_bridge"])
def test_loaded_primitive_code_mutation_is_rejected(template, query, phase):
    primitives = stain._candidate()
    bridge = fitted(template)[0] if phase == "cached_bridge" else None
    function = primitives.get_mean_std
    before = function.__code__
    try:
        function.__code__ = primitives.standardize.__code__
        with pytest.raises(stain.ReinhardValidationError, match="primitive code changed"):
            fitted(template) if bridge is None else bridge.transform(query)
    finally:
        function.__code__ = before


@pytest.mark.parametrize("change", ["wrong_version", "extra", "duplicate", "changed_leaf"])
def test_distribution_member_closure_checked_before_import(template, monkeypatch, change):
    actual = metadata.distribution("torchstain")
    paths = list(actual.files)
    if change == "extra":
        paths.append(metadata.PackagePath("torchstain/unreviewed.py"))
    elif change == "duplicate":
        paths.append(metadata.PackagePath("torchstain/__init__.py"))
    fake = SimpleNamespace(
        version="1.3.0" if change == "wrong_version" else "1.4.1", files=paths,
        locate_file=actual.locate_file,
    )
    if change == "changed_leaf":
        fake.locate_file = lambda name: Path(__file__) if name == stain.PACKAGE_MEMBERS[-1][0] \
            else actual.locate_file(name)
    original = metadata.distribution
    monkeypatch.setattr(stain.metadata, "distribution",
                        lambda name: fake if name == "torchstain" else original(name))
    imports = []
    monkeypatch.setattr(stain, "import_module", lambda *args: imports.append(args))
    with pytest.raises(stain.ReinhardValidationError):
        fitted(template)
    assert not imports


def test_no_rng_mutation_or_default_changes(template, query):
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    torch_before = torch.get_rng_state().clone()
    dtype_before = torch.get_default_dtype()
    stack_before = tuple(_get_current_function_mode_stack())
    bridge, _ = fitted(template)
    bridge.transform(query)
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0] and np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert random.getstate() == python_before and torch.equal(torch.get_rng_state(), torch_before)
    assert torch.get_default_dtype() == dtype_before
    assert tuple(_get_current_function_mode_stack()) == stack_before
