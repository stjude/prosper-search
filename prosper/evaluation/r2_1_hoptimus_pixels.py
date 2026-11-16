"""Synthetic-only in-memory pixel adapters for frozen R2.1 stimuli.

Readers and bridges are trusted callbacks: they must not perform payload IO,
draw random values or change process state. This module is not a sandbox for
arbitrary callback code. No real stain normalization or model inference is
implemented or certified here.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import re
from typing import Callable

from PIL import Image
import torch
import torchvision
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from .r2_1_hoptimus_perturb import CONDITION_ORDER, ParameterValue, StimulusRecord


NominalKey = tuple[str, int, int, int, int]
Location = tuple[int, int]
Size = tuple[int, int]
Reader = Callable[[Location, int, Size], Image.Image]
MEAN = (0.707223, 0.578729, 0.703617)
STD = (0.211883, 0.230117, 0.177517)


class PixelValidationError(ValueError):
    """A realized record, materialized image, bridge or pixel result is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PixelValidationError(message)


def _digest(value: object) -> None:
    _require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "template digest must be 64 lowercase hexadecimal characters",
    )


@dataclass(frozen=True)
class ReadPlan:
    nominal_key: NominalKey
    location: Location
    level: int
    size: Size

    @property
    def actual_read(self) -> tuple[Location, int, Size]:
        return self.location, self.level, self.size


@dataclass(frozen=True)
class ReinhardBridge:
    """Only dispatch is supported; actual stain semantics remain unconfigured."""

    template_sha256: str
    transform: Callable[[Image.Image], Image.Image]

    def __post_init__(self) -> None:
        _digest(self.template_sha256)
        _require(callable(self.transform), "Reinhard bridge transform must be callable")


@dataclass(frozen=True)
class ModuleProfile:
    distribution: str
    distribution_version: str
    module_version: str
    module_path: str


@dataclass(frozen=True)
class PixelRuntimeProfile:
    modules: tuple[ModuleProfile, ...]
    limitations: tuple[str, ...] = (
        "Distribution versions and module paths do not establish full compiled-code identity.",
    )


def runtime_profile() -> PixelRuntimeProfile:
    """Assert the released distribution profile without reconfiguring globals."""
    modules: list[ModuleProfile] = []
    for name, expected, module in (
        ("torch", "2.5.1", torch),
        ("torchvision", "0.20.1", torchvision),
        ("Pillow", "11.3.0", Image),
    ):
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise PixelValidationError(f"missing released distribution: {name}") from exc
        _require(actual == expected, f"{name} distribution must be {expected}, got {actual}")
        module_version = getattr(module, "__version__", None)
        module_path = getattr(module, "__file__", None)
        _require(isinstance(module_version, str) and bool(module_version)
                 and type(module_path) is str and bool(module_path),
                 f"{name} module version/path metadata is missing")
        modules.append(ModuleProfile(name, actual, str(module_version), module_path))
    _require(torch.get_default_dtype() == torch.float32,
             "pixel preprocessing requires default floating dtype FP32")
    return PixelRuntimeProfile(tuple(modules))


def _integer(value: ParameterValue) -> int:
    if type(value) is not int:
        raise PixelValidationError("parameter must be a Python integer, not bool")
    return value


def _factor(value: ParameterValue, condition: str) -> float:
    low, high = (0.9, 1.1) if condition == "contrast" else (0.85, 1.15)
    if type(value) is not float or not low <= value <= high:
        raise PixelValidationError("factor must be a finite Python float within the rounded inclusive range")
    return value


def _parameters(stimulus: StimulusRecord) -> dict[str, ParameterValue]:
    _require(type(stimulus) is StimulusRecord, "expected a frozen StimulusRecord")
    _require(type(stimulus.slide_id) is str and bool(stimulus.slide_id),
             "stimulus slide ID must be an exact nonempty string")
    _require(type(stimulus.level) is int and stimulus.level == 0,
             "stimulus level must be integer 0")
    _require(type(stimulus.patch_size) is int and stimulus.patch_size == 512,
             "stimulus patch size must be integer 512")
    _require(type(stimulus.condition) is str and stimulus.condition in CONDITION_ORDER,
             "unknown stimulus condition")
    _require(type(stimulus.parameters) is tuple, "parameters must be an immutable tuple")
    parameters: dict[str, ParameterValue] = {}
    for pair in stimulus.parameters:
        _require(type(pair) is tuple and len(pair) == 2, "expected parameter name/value pair")
        name, value = pair
        _require(type(name) is str and name not in parameters,
                 "invalid or duplicate parameter name")
        parameters[name] = value

    condition = stimulus.condition
    if condition == "identity":
        expected = {"treatment": "untreated"}
    elif condition in ("rotate_90", "rotate_180", "rotate_270"):
        expected = {"angle": int(condition.removeprefix("rotate_"))}
    elif condition in ("flip_horizontal", "flip_vertical"):
        expected = {"axis": condition.removeprefix("flip_")}
    elif condition in ("brightness", "saturation", "contrast"):
        _require(set(parameters) == {"factor"}, "colour condition requires exactly factor")
        _factor(parameters["factor"], condition)
        return parameters
    elif condition == "spatial_shift":
        _require(set(parameters) == {"x_shift", "y_shift"},
                 "spatial shift requires exactly x_shift/y_shift")
        _require(all(-16 <= _integer(value) <= 16 for value in parameters.values()),
                 "shift offsets must be integers in [-16,16]")
        return parameters
    elif condition == "reinhard":
        _require(set(parameters) == {"template_sha256"},
                 "Reinhard requires exactly template_sha256")
        _digest(parameters["template_sha256"])
        return parameters
    else:
        expected = {"kernel_size": 3, "sigma": 0.5, "application_stage": "post_preprocessing"}
    _require(set(parameters) == set(expected), "parameter names do not match condition")
    _require(all(type(parameters[name]) is type(value) and parameters[name] == value
                 for name, value in expected.items()), "fixed parameter value/type mismatch")
    return parameters


def read_plan(nominal_key: NominalKey, stimulus: StimulusRecord) -> ReadPlan:
    """Preserve nominal identity and signed actual origins; no image access."""
    parameters = _parameters(stimulus)
    _require(type(nominal_key) is tuple and len(nominal_key) == 5,
             "nominal key must be (slide_id,0,x,y,512)")
    slide_id, level, x, y, patch_size = nominal_key
    _require(type(slide_id) is str and bool(slide_id) and slide_id == stimulus.slide_id,
             "nominal/stimulus slide IDs must agree exactly")
    _require(type(level) is int and level == 0 and type(patch_size) is int and patch_size == 512,
             "nominal geometry must be integer level0/patch512")
    _require(type(x) is int and type(y) is int and x >= 0 and y >= 0,
             "nominal coordinates must be nonnegative integers")
    location = (x, y)
    if stimulus.condition == "spatial_shift":
        location = (x + _integer(parameters["x_shift"]), y + _integer(parameters["y_shift"]))
    return ReadPlan(nominal_key, location, level, (patch_size, patch_size))


def _materialized_image(
    image: Image.Image, *, size: Size, modes: tuple[str, ...], context: str,
) -> None:
    _require(type(image) is Image.Image, f"{context}: require base PIL.Image.Image, not a subclass")
    # Pillow 11.3 exposes the materialized core privately; .load() is not a guard.
    pixel_core = image.__dict__.get("_im")
    core_type = type(Image.core.new("RGB", (0, 0)))
    _require(type(pixel_core) is core_type, f"{context}: existing materialized pixel core required")
    _require(image.mode in modes and pixel_core.mode == image.mode,
             f"{context}: wrong image mode")
    _require(image.size == size and pixel_core.size == size,
             f"{context}: wrong image dimensions")


def _bridge(
    parameters: dict[str, ParameterValue], condition: str, bridge: ReinhardBridge | None,
) -> None:
    if bridge is not None:
        _require(type(bridge) is ReinhardBridge, "expected a ReinhardBridge")
        _digest(bridge.template_sha256)
        _require(callable(bridge.transform), "Reinhard bridge transform must be callable")
    if condition == "reinhard":
        if bridge is None:
            raise PixelValidationError("Reinhard is unconfigured: a synthetic bridge is required")
        _require(bridge.template_sha256 == parameters["template_sha256"],
                 "Reinhard bridge/stimulus template digest mismatch")


def _tensor(value: torch.Tensor, context: str) -> torch.Tensor:
    _require(type(value) is torch.Tensor, f"{context}: expected base torch.Tensor")
    _require(value.device.type == "cpu" and value.dtype == torch.float32,
             f"{context}: expected CPU FP32 tensor")
    _require(tuple(value.shape) == (3, 224, 224), f"{context}: wrong tensor dimensions")
    _require(bool(torch.isfinite(value).all()), f"{context}: nonfinite tensor")
    return value


def apply_pixels(
    image: Image.Image, stimulus: StimulusRecord, *,
    reinhard_bridge: ReinhardBridge | None = None,
) -> torch.Tensor:
    """Apply one independent realized condition to a materialized original patch."""
    parameters = _parameters(stimulus)
    runtime_profile()
    _materialized_image(image, size=(512, 512), modes=("RGB", "RGBA"), context="input")
    _bridge(parameters, stimulus.condition, reinhard_bridge)
    rgb = image.copy() if image.mode == "RGB" else image.convert("RGB")
    condition = stimulus.condition
    if condition in ("rotate_90", "rotate_180", "rotate_270"):
        rgb = rgb.rotate(_integer(parameters["angle"]), expand=True)
    elif condition == "flip_horizontal":
        rgb = F.hflip(rgb)
    elif condition == "flip_vertical":
        rgb = F.vflip(rgb)
    elif condition == "brightness":
        rgb = F.adjust_brightness(rgb, _factor(parameters["factor"], condition))
    elif condition == "saturation":
        rgb = F.adjust_saturation(rgb, _factor(parameters["factor"], condition))
    elif condition == "contrast":
        rgb = F.adjust_contrast(rgb, _factor(parameters["factor"], condition))
    elif condition == "reinhard":
        if reinhard_bridge is None:
            raise PixelValidationError("Reinhard is unconfigured: a synthetic bridge is required")
        # Pillow's shallow info copy can still alias nested caller metadata.
        rgb.info = {}
        rgb = reinhard_bridge.transform(rgb)
    _materialized_image(rgb, size=(512, 512), modes=("RGB",), context="treated RGB/bridge return")
    resized = F.resize(
        rgb, [224, 224], interpolation=InterpolationMode.BILINEAR, antialias=True,
    )
    _materialized_image(resized, size=(224, 224), modes=("RGB",), context="resized RGB")
    tensor = _tensor(F.to_tensor(resized), "ToTensor")
    tensor = _tensor(F.normalize(tensor, MEAN, STD, inplace=False), "Normalize")
    if condition == "gaussian_blur":
        tensor = F.gaussian_blur(tensor, kernel_size=[3, 3], sigma=[0.5, 0.5])
    return _tensor(tensor, "pixel result")


def read_and_apply(
    nominal_key: NominalKey, stimulus: StimulusRecord, read_region: Reader, *,
    reinhard_bridge: ReinhardBridge | None = None,
) -> tuple[ReadPlan, torch.Tensor]:
    """Invoke one trusted synthetic reader, then apply the independent condition."""
    plan = read_plan(nominal_key, stimulus)
    runtime_profile()
    _bridge(_parameters(stimulus), stimulus.condition, reinhard_bridge)
    _require(callable(read_region), "read_region must be an explicitly supplied callable")
    image = read_region(*plan.actual_read)
    return plan, apply_pixels(image, stimulus, reinhard_bridge=reinhard_bridge)
