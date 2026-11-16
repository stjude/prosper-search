"""Pure R2.1 stimulus manifests and H-Optimus SAMPLER/L1 kernels.

This slice does not apply transformations, read templates or features, load a
model, assess cache compatibility, or evaluate retrieval outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import re
from typing import Literal, Sequence

import numpy as np


Condition = Literal[
    "identity", "rotate_90", "rotate_180", "rotate_270",
    "flip_horizontal", "flip_vertical", "brightness", "saturation",
    "contrast", "spatial_shift", "reinhard", "gaussian_blur",
]
ParameterValue = str | int | float
CONDITION_ORDER: tuple[Condition, ...] = (
    "identity", "rotate_90", "rotate_180", "rotate_270",
    "flip_horizontal", "flip_vertical", "brightness", "saturation",
    "contrast", "spatial_shift", "reinhard", "gaussian_blur",
)
FEATURE_DIM = 1536
SAMPLER_DIM = 15360
PERCENTILES = (5, 15, 25, 35, 45, 55, 65, 75, 85, 95)
STIMULUS_POLICY = "r2_1_stimulus_v1"
STIMULUS_SEED = 1234


class StimulusValidationError(ValueError):
    """A stimulus identity, parameter key or serialization is invalid."""


class KernelValidationError(ValueError):
    """An input array does not satisfy the mathematical kernel specification."""


class KernelComputationError(ArithmeticError):
    """A finite input produced overflow or another nonfinite result."""


@dataclass(frozen=True)
class StimulusRecord:
    slide_id: str
    condition: Condition
    level: int
    patch_size: int
    parameters: tuple[tuple[str, ParameterValue], ...]


@dataclass(frozen=True)
class StimulusPolicy:
    name: str
    seed: int
    key_encoding: str
    seed_derivation: str
    algorithm: str
    numpy_version: str


@dataclass(frozen=True)
class StimulusManifest:
    policy: StimulusPolicy
    records: tuple[StimulusRecord, ...]


def canonical_json_bytes(value: object) -> bytes:
    """Encode canonical ASCII JSON; reject NaN and infinity."""
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StimulusValidationError(f"invalid canonical JSON: {exc}") from exc


def _slide_id(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise StimulusValidationError("slide IDs must be exact nonempty strings")


def stimulus_parameter_key(
    slide_id: str, condition: str, parameter: str,
) -> bytes:
    """Return the exact pre-hash key for one independently drawn parameter."""
    _slide_id(slide_id)
    valid = (
        condition in ("brightness", "saturation", "contrast")
        and parameter == "factor"
    ) or (
        condition == "spatial_shift" and parameter in ("x_shift", "y_shift")
    )
    if not valid:
        raise StimulusValidationError("unsupported stochastic condition/parameter")
    return canonical_json_bytes(
        [STIMULUS_POLICY, STIMULUS_SEED, slide_id, condition, parameter]
    )


def stimulus_parameter_seed(
    slide_id: str, condition: str, parameter: str,
) -> int:
    """Use the full SHA256 digest as an unsigned big-endian PCG64 seed."""
    key = stimulus_parameter_key(slide_id, condition, parameter)
    return int.from_bytes(hashlib.sha256(key).digest(), "big", signed=False)


def _rng(slide_id: str, condition: str, parameter: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(
        stimulus_parameter_seed(slide_id, condition, parameter)
    ))


def _parameters(
    slide_id: str, condition: Condition, template_sha256: str,
) -> tuple[tuple[str, ParameterValue], ...]:
    if condition == "identity":
        return (("treatment", "untreated"),)
    if condition in ("rotate_90", "rotate_180", "rotate_270"):
        return (("angle", int(condition.removeprefix("rotate_"))),)
    if condition in ("flip_horizontal", "flip_vertical"):
        return (("axis", condition.removeprefix("flip_")),)
    if condition in ("brightness", "saturation", "contrast"):
        low, high = (0.9, 1.1) if condition == "contrast" else (0.85, 1.15)
        factor = float(_rng(slide_id, condition, "factor").uniform(low, high))
        # NumPy documents possible rounding to high; do not clip or redraw.
        if not np.isfinite(factor) or not low <= factor <= high:
            raise KernelComputationError("uniform stimulus factor outside range")
        return (("factor", factor),)
    if condition == "spatial_shift":
        return tuple(
            (parameter, int(_rng(slide_id, condition, parameter).integers(-16, 17)))
            for parameter in ("x_shift", "y_shift")
        )
    if condition == "reinhard":
        return (("template_sha256", template_sha256),)
    if condition == "gaussian_blur":
        return (
            ("kernel_size", 3), ("sigma", 0.5),
            ("application_stage", "post_preprocessing"),
        )
    raise StimulusValidationError("unsupported condition")


def _policy() -> StimulusPolicy:
    return StimulusPolicy(
        name=STIMULUS_POLICY,
        seed=STIMULUS_SEED,
        key_encoding="UTF-8 compact ASCII JSON array",
        seed_derivation="full SHA256 digest as unsigned big-endian integer",
        algorithm="PCG64",
        numpy_version=np.__version__,
    )


def _template_digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise StimulusValidationError("template SHA256 must be 64 lowercase hex characters")


def generate_stimulus_manifest(
    slide_ids: Sequence[str], template_sha256: str,
) -> StimulusManifest:
    """Materialize all twelve slide-wide independent conditions, without IO."""
    if isinstance(slide_ids, (str, bytes)):
        raise StimulusValidationError("slide IDs must be a sequence, not a string")
    try:
        ids = tuple(slide_ids)
    except TypeError as exc:
        raise StimulusValidationError("slide IDs must be a sequence") from exc
    for slide_id in ids:
        _slide_id(slide_id)
    if len(set(ids)) != len(ids):
        raise StimulusValidationError("duplicate slide IDs")
    _template_digest(template_sha256)
    records = tuple(
        StimulusRecord(
            slide_id=slide_id, condition=condition, level=0, patch_size=512,
            parameters=_parameters(slide_id, condition, template_sha256),
        )
        for slide_id in sorted(ids)
        for condition in CONDITION_ORDER
    )
    return StimulusManifest(policy=_policy(), records=records)


def _validate_manifest(manifest: StimulusManifest) -> None:
    if type(manifest) is not StimulusManifest:
        raise StimulusValidationError("expected a StimulusManifest")
    if type(manifest.policy) is not StimulusPolicy:
        raise StimulusValidationError("expected a StimulusPolicy")
    expected_policy = _policy()
    for field in fields(expected_policy):
        actual = getattr(manifest.policy, field.name)
        expected = getattr(expected_policy, field.name)
        if type(actual) is not type(expected) or actual != expected:
            raise StimulusValidationError(f"invalid stimulus policy field: {field.name}")
    if type(manifest.records) is not tuple:
        raise StimulusValidationError("manifest records must be an immutable tuple")
    if not manifest.records:
        return

    slide_ids: set[str] = set()
    templates: set[str] = set()
    for record in manifest.records:
        if type(record) is not StimulusRecord:
            raise StimulusValidationError("expected a StimulusRecord")
        _slide_id(record.slide_id)
        if type(record.condition) is not str or record.condition not in CONDITION_ORDER:
            raise StimulusValidationError("invalid stimulus condition")
        if type(record.level) is not int or record.level != 0:
            raise StimulusValidationError("stimulus level must be integer 0")
        if type(record.patch_size) is not int or record.patch_size != 512:
            raise StimulusValidationError("stimulus patch size must be integer 512")
        if type(record.parameters) is not tuple:
            raise StimulusValidationError("parameters must be an immutable tuple")
        names: set[str] = set()
        for pair in record.parameters:
            if type(pair) is not tuple or len(pair) != 2:
                raise StimulusValidationError("parameters must contain name/value pairs")
            name, value = pair
            if type(name) is not str or name in names:
                raise StimulusValidationError("invalid or duplicate parameter name")
            names.add(name)
            if type(value) not in (str, int, float):
                raise StimulusValidationError("invalid parameter value type")
            if type(value) is float and not np.isfinite(value):
                raise StimulusValidationError("parameter values must be finite")
        if record.condition == "reinhard":
            if names != {"template_sha256"}:
                raise StimulusValidationError("Reinhard requires exactly template_sha256")
            digest = record.parameters[0][1]
            _template_digest(digest)
            templates.add(digest)
        slide_ids.add(record.slide_id)

    if len(templates) != 1:
        raise StimulusValidationError("manifest must bind one consistent Reinhard template")
    template_sha256 = next(iter(templates))
    expected_manifest = generate_stimulus_manifest(tuple(slide_ids), template_sha256)
    if tuple((r.slide_id, r.condition) for r in manifest.records) != tuple(
        (r.slide_id, r.condition) for r in expected_manifest.records
    ):
        raise StimulusValidationError("conditions must be complete, unique and canonically ordered")
    for actual, expected in zip(manifest.records, expected_manifest.records):
        actual_parameters = dict(actual.parameters)
        if set(actual_parameters) != {name for name, _ in expected.parameters}:
            raise StimulusValidationError("parameter names do not match condition")
        for name, expected_value in expected.parameters:
            value = actual_parameters[name]
            if type(value) is not type(expected_value) or value != expected_value:
                raise StimulusValidationError("parameter does not match the keyed stimulus recipe")


def serialize_stimulus_manifest(manifest: StimulusManifest) -> bytes:
    """Validate the fixed recipe, then serialize actual values without disk IO."""
    _validate_manifest(manifest)
    records = [
        {
            "slide_id": record.slide_id, "condition": record.condition,
            "level": record.level, "patch_size": record.patch_size,
            "parameters": dict(record.parameters),
        }
        for record in manifest.records
    ]
    return canonical_json_bytes({"policy": asdict(manifest.policy), "records": records})


def _finite_floating_array(value: np.ndarray, name: str) -> None:
    if type(value) is not np.ndarray or value.dtype.kind != "f":
        raise KernelValidationError(f"{name} must be a base floating NumPy array")
    if not np.isfinite(value).all():
        raise KernelValidationError(f"{name} must contain only finite values")


def sampler_embedding(tile_features: np.ndarray) -> np.ndarray:
    """Precast tiles to FP16, take linear percentiles, then C-order flatten."""
    _finite_floating_array(tile_features, "tile features")
    if (
        tile_features.ndim != 2
        or tile_features.shape[0] == 0
        or tile_features.shape[1] != FEATURE_DIM
    ):
        raise KernelValidationError("tile features must have nonempty shape (N, 1536)")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            precast = tile_features.astype(np.float16)
            if not np.isfinite(precast).all():
                raise KernelComputationError("FP16 conversion produced nonfinite features")
            pooled = np.percentile(precast, PERCENTILES, axis=0, method="linear")
    except FloatingPointError as exc:
        raise KernelComputationError("SAMPLER conversion/percentile overflow") from exc
    if not np.isfinite(pooled).all():
        raise KernelComputationError("SAMPLER produced nonfinite values")
    return pooled.ravel(order="C")


def cityblock_distances(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Return ordered L1 distances; an empty reference library returns no rows."""
    _finite_floating_array(query, "query")
    _finite_floating_array(references, "references")
    if query.shape != (SAMPLER_DIM,):
        raise KernelValidationError("query must have shape (15360,)")
    if references.ndim != 2 or references.shape[1] != SAMPLER_DIM:
        raise KernelValidationError("references must have shape (R, 15360)")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            distances = np.abs(references - query).sum(axis=1)
    except FloatingPointError as exc:
        raise KernelComputationError("cityblock arithmetic overflow") from exc
    if not np.isfinite(distances).all():
        raise KernelComputationError("cityblock produced nonfinite distances")
    return distances
