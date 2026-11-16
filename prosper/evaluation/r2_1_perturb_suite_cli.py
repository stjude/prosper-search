"""R2.1 reproducible perturbation suite (PS-1; [decision record] candidate 002, amendments A1-A8).

Additive to the frozen extractor ``r2_1_perturb_extract_cli`` (module X, sha 1b97b4b0...).
X is imported and never patched; only the functions named by the specification are called.

Sub-commands (every one needs an explicit ``--out-root``; nothing is written outside it):

  gen-manifest-v2      stimulus manifest v2 = v1 verbatim + one ``gaussian_blur_image`` record
                       per slide (v1 slide order); ``stimuli_v2.json`` + ``.sha256`` sidecar
  capture-env          fixed-environment block + reference-library closure capture (JSON)
  pin-reference-libs   identical captures -> ``environment/reference_libs.json``
  env-check            assert the fixed environment (no rendering)
  capture-host-tags    ``lshosts -w`` GPU driver tag per host -> ``environment/host_driver_tags.json``
  plan-tasks           fingerprint task packing by tile count -> ``fingerprints/TASKS.json``
  fingerprint          per-slide 13-picture SHA-256 fingerprints (CPU only)
  finalize             ``fingerprints/INDEX.json`` and ``fingerprints/COMPLETE`` (R3/R5a); full
                       report -> ``fingerprints/finalize_reports/`` + compact stdout (C4-1)
  finalize-features    ``features/<encoder>/INDEX.json`` and ``COMPLETE`` over the extension plan's
                       slides (R3/R5a/R6b/R6d); report -> ``features/<encoder>/finalize_reports/``
                       + the same compact stdout (C4-2)
  quarantine-orphans   move-only, recorded quarantine of lone fingerprint/feature files (R3)
  reference-png        lossless PNGs of the first four tiles x 13 pictures of pilot slides
  select-pilot         stratified bit-identity pilot slide selection
  pilot-render         pilot (a): in-line re-render on a GPU host vs the CPU fingerprints
  pilot-process-slide  pilot (b): frozen ``X.process_slide`` vs first-run pickles (+ twice diagnostic)
  plan-extension       H-Optimus extension population (A3 predicate) -> ``extension/PLAN.json``
  extract              verifying extractor wrapper (streaming per-tile picture verification)

Conditions (``SUITE_CONDITIONS``): ``X.CONDITIONS`` in order, then ``gaussian_blur_image``.
The legacy ``gaussian_blur`` arm is post-preprocessing (encoder-specific), the historical
Fig. 4A definition; its *picture* is the untreated tile (digest == identity digest).

Output formats
--------------
``fingerprints/<slide>.fp.npy``  uint8 ``(n_tiles, 13, 32)``: raw SHA-256 of
    ``picture.tobytes()`` (packed row-major RGB, 512x512) per tile (bank ``coords`` order) and
    condition (``SUITE_CONDITIONS`` order).
``fingerprints/<slide>.fp.json`` schema ``r2_1_perturb_suite_fingerprint_v1`` (see
    :func:`_fp_json`): ``array.content_sha256`` = SHA-256 of the C-order array bytes (independent
    of the .npy header), ``aggregate_digests[c]`` = SHA-256 over the concatenated raw 32-byte
    per-tile digests of condition ``c`` in tile order.
``features/<encoder>/<slide>.pkl`` = ``pickle.dumps((feats, coords), protocol=4)`` with
    ``feats`` a dict ``condition -> np.float32 (n_tiles, D)`` holding ONLY the encoded
    conditions (``record["conditions_encoded"]``) and ``coords`` the bank coords list.
``features/<encoder>/<slide>.record.json`` schema ``r2_1_perturb_suite_record_v1`` with
    ``pickle_sha256``/``pickle_bytes``, ``conditions_encoded``, ``manifest_v2_sha256``,
    ``extractor_code_sha256`` (X), ``suite_code_sha256`` (this module),
    ``reinhard_nonfinite_tile_indices``, ``fingerprint_verification``, ``first_run`` (status
    ``present``/``absent``/``not_applicable``, ``direct_run``, ``identity_equal``), WSI
    ``st_size``/``st_mtime_ns``, GPU model/driver and the environment block.
Pilot (b) keeps X's native ``r2_1_perturb_record_v1`` outputs under
``pilot_bit_identity/process_slide/`` (A1).
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import io
import json
import math
import os
import pickle
import platform
import re
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from . import paths
from . import r2_1_perturb_extract_cli as X

# --------------------------------------------------------------------------
# Frozen extractor identity (asserted at import)
# --------------------------------------------------------------------------

EXTRACTOR_SHA256 = "14eb457ebfe2243a924bb7928483538985120b6987de51eb4a6cc6ccf3a8d0bb"
EXPECTED_X_CONDITIONS: Tuple[str, ...] = (
    "identity", "rotate_90", "rotate_180", "rotate_270",
    "flip_horizontal", "flip_vertical", "brightness", "saturation",
    "contrast", "spatial_shift", "reinhard", "gaussian_blur",
)


class SuiteError(RuntimeError):
    """A suite invariant failed; the operation stops without publishing."""


class EnvironmentMismatch(SuiteError):
    pass


class PinnedFitError(SuiteError):
    pass


class PopulationStop(SuiteError):
    """Extension population rule violated (A3): stop before encoding."""


def check_extractor(path: Path, conditions: Sequence[str]) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != EXTRACTOR_SHA256:
        raise SuiteError(f"frozen extractor {path} sha256 {digest} != {EXTRACTOR_SHA256}")
    if tuple(conditions) != EXPECTED_X_CONDITIONS:
        raise SuiteError(f"X.CONDITIONS {tuple(conditions)} != {EXPECTED_X_CONDITIONS}")
    return digest


check_extractor(Path(X.__file__).resolve(), X.CONDITIONS)
SUITE_CODE_PATH = Path(__file__).resolve()
SUITE_CODE_SHA256 = X.sha256_file(SUITE_CODE_PATH)

# --------------------------------------------------------------------------
# Specification constants
# --------------------------------------------------------------------------

BLUR_CONDITION = "gaussian_blur_image"
SUITE_CONDITIONS: Tuple[str, ...] = tuple(X.CONDITIONS) + (BLUR_CONDITION,)
BLUR_WEIGHTS_HEX: Tuple[str, ...] = (
    "0x1.6d67623129833p-7", "0x1.35b4c9bd6812dp-4", "0x1.e84adbabd1817p-3",
    "0x1.6604495267dd0p-2", "0x1.e84adbabd1817p-3", "0x1.35b4c9bd6812dp-4",
    "0x1.6d67623129833p-7",
)
# Run-time weights are the hex literals themselves (specification item 2).
BLUR_WEIGHTS: Tuple[float, ...] = tuple(float.fromhex(h) for h in BLUR_WEIGHTS_HEX)
BLUR_SIGMA = 1.1428571428571428
BLUR_KERNEL = 7
BLUR_PARAMETERS: Dict[str, Any] = {
    "application_stage": "image_rgb_512", "kernel_size": BLUR_KERNEL, "sigma": BLUR_SIGMA,
    "padding": "reflect_101", "arithmetic": "float64_separable_explicit_order",
    "rounding": "rint_half_even_clip_uint8", "weights_hex": list(BLUR_WEIGHTS_HEX),
}
# Specification [A4] golden: default_rng(0).integers(0, 256, (512, 512, 3), uint8) -> blur -> tobytes.
BLUR_GOLDEN_SHA256 = "3bb4aa71d9861b91fb661b40f1a0892beb55872d9cf96b2c50a67a10ae15f94f"

V1_POLICY_NAME = "r2_1_stimulus_v1"
V2_POLICY_NAME = "r2_1_stimulus_v2"
V1_RECORD_COUNT = 33336                      # production v1 (asserted for the production sha)
MANIFEST_RECORD_KEYS = frozenset({"condition", "level", "parameters", "patch_size", "slide_id"})

PINNED_MEANS = (67.08672332763672, 23.116905212402344, -15.08007526397705)
PINNED_STDS = (18.064807891845703, 10.140364646911621, 11.612739562988281)
DIRECT_MEANS = (67.08672332763672, 23.11690330505371, -15.080077171325684)
DIRECT_MAIN_THREADS = 121

EXPECTED_ENV = {
    "lsf_model": "Opteron140",
    "cpu_vendor": "AuthenticAMD",
    "cpu_capability": "AVX2",
    "python_major_minor": "3.12",
    "numpy": "2.2.4",
    "torch": "2.5.1+cu124",
    "torchvision": "0.20.1+cu124",
    "pillow": "11.3.0",
    "openslide_python": "1.4.3",
    "openslide_library": "3.4.1",
}
OPENSLIDE_LIB = "/lib64/libopenslide.so.0"

FP_SCHEMA = "r2_1_perturb_suite_fingerprint_v1"
RECORD_SCHEMA = "r2_1_perturb_suite_record_v1"
REFLIBS_SCHEMA = "r2_1_perturb_suite_reference_libs_v3"      # [R4]/[R6a] mapped ELF closure
ENV_CAPTURE_SCHEMA = "r2_1_perturb_suite_env_capture_v3"      # [R6a] + non_elf, locale
PLAN_SCHEMA = "r2_1_perturb_suite_extension_plan_v1"
FIRST_RUN_SCHEMA = "r2_1_perturb_record_v1"

FP_TASK_MAX_TILES = 15000                    # [A7] "<= ~15,000 tiles"
PILOT_MIN_TILES = 500
PILOT_MIN_SLIDES = 10
REFERENCE_PNG_TILES = 4
EXPECTED_DIRECT_COUNT = 655   # [R5b] direct-run set size (A3); override only with a reason
PILOT_PROCESS_SLIDE = {"forward_batch": 256, "loader_batch": 8, "num_workers": 7,
                       "prefetch_factor": 2}  # first-run loader settings (A1; X parser defaults)


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------


def _canon_tag(obj: Any) -> Any:
    """Tagged JSON-able form for :func:`_canon` [R2]: numpy scalars become
    ``["np", dtype.str, value]`` and arrays ``["np", dtype.str, shape, values]`` so that
    ``np.int64(7)`` and ``7`` canonicalise differently; tuples become lists."""
    if isinstance(obj, np.generic):
        return ["np", obj.dtype.str, _canon_tag(obj.item())]
    if isinstance(obj, np.ndarray):
        return ["np", obj.dtype.str, list(obj.shape), _canon_tag(obj.tolist())]
    if isinstance(obj, dict):
        return {k: _canon_tag(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon_tag(v) for v in obj]
    return obj


def _canon(obj: Any) -> str:
    """Type-strict canonical JSON (distinguishes 7 / 7.0 / True / np.int64(7)) [R2].
    Non-JSON types other than numpy raise TypeError/ValueError, which callers on the
    identity paths convert into per-slide failures."""
    return json.dumps(_canon_tag(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _dump_compact(doc: Any) -> bytes:
    # Same serialization as the v1 manifest (sorted keys, compact separators, no newline).
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def _dump_pretty(doc: Any) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n").encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _tag() -> str:
    return (f"{os.environ.get('LSB_JOBID', 'local')}_{os.environ.get('LSB_JOBINDEX', '0')}_"
            f"{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}_{time.time_ns() % 10**9:09d}")


def _inside(root: Path, dest: Path) -> Path:
    r = Path(root).resolve()
    d = Path(dest).resolve()
    if d != r and r not in d.parents:
        raise SuiteError(f"refusing to write {d}: outside --out-root {r}")
    return d


def publish(root: Path, dest: Path, data: bytes) -> None:
    """Atomic, never-overwrite publish (X._publish_bytes) confined to ``root``."""
    _inside(root, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    X._publish_bytes(dest, data)


def publish_idempotent(root: Path, dest: Path, data: bytes) -> str:
    """Publish deterministic content; an existing identical file is accepted, a different one
    is an error (never overwritten). Returns 'written' or 'existing'."""
    _inside(root, dest)
    if dest.exists():
        if dest.read_bytes() != data:
            raise SuiteError(f"{dest} exists with different content; refusing to overwrite")
        return "existing"
    publish(root, dest, data)
    return "written"


def _cpu_seconds() -> float:
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def _split_list(s: str) -> List[str]:
    return [v.strip() for v in (s or "").split(",") if v.strip()]


# --------------------------------------------------------------------------
# Stimulus manifest v2 (item 1)
# --------------------------------------------------------------------------


def blur_record(slide_id: str) -> dict:
    return {"condition": BLUR_CONDITION, "level": 0, "parameters": dict(BLUR_PARAMETERS,
            weights_hex=list(BLUR_WEIGHTS_HEX)), "patch_size": 512, "slide_id": slide_id}


def _v1_document(v1_path: Path, v1_sha256: str) -> Tuple[str, Dict[str, Dict[str, dict]], dict]:
    if not v1_sha256:
        raise SuiteError("the v1 manifest sha256 must be given explicitly")
    digest, by_slide = X.load_manifest(Path(v1_path), v1_sha256)
    raw = Path(v1_path).read_bytes()
    if _sha(raw) != digest:
        raise SuiteError("v1 manifest changed between validation and read")
    doc = json.loads(raw)
    if set(doc) != {"policy", "records"}:
        raise SuiteError(f"v1 top-level keys {sorted(doc)}")
    if not isinstance(doc["policy"], dict) or doc["policy"].get("name") != V1_POLICY_NAME:
        raise SuiteError(f"v1 policy name {doc['policy'].get('name')!r} != {V1_POLICY_NAME}")
    if digest == X.MANIFEST_SHA256 and len(doc["records"]) != V1_RECORD_COUNT:
        raise SuiteError(f"production v1 has {len(doc['records'])} records, not {V1_RECORD_COUNT}")
    for rec in doc["records"]:
        if set(rec) != MANIFEST_RECORD_KEYS:
            raise SuiteError(f"v1 record keys {sorted(rec)}")
    return digest, by_slide, doc


def _slide_order(records: Sequence[dict]) -> List[str]:
    seen: Dict[str, None] = {}
    for rec in records:
        seen.setdefault(rec["slide_id"], None)
    return list(seen)


def expected_v2_policy(v1_policy: dict, v1_sha256: str) -> dict:
    pol = dict(v1_policy)
    pol["name"] = V2_POLICY_NAME
    pol["supersedes"] = {"name": V1_POLICY_NAME, "sha256": v1_sha256}
    return pol


def build_manifest_v2(v1_path: Path, v1_sha256: str) -> Tuple[bytes, dict]:
    v1_digest, _, v1_doc = _v1_document(v1_path, v1_sha256)
    order = _slide_order(v1_doc["records"])
    doc = {"policy": expected_v2_policy(v1_doc["policy"], v1_digest),
           "records": list(v1_doc["records"]) + [blur_record(s) for s in order]}
    raw = _dump_compact(doc)
    info = {"v1_sha256": v1_digest, "v1_records": len(v1_doc["records"]),
            "v1_slides": len(order), "v2_records": len(doc["records"]),
            "v1_reserialization_byte_identical": _dump_compact(v1_doc) == Path(v1_path).read_bytes()}
    return raw, info


@dataclass
class ManifestV2:
    path: Path
    sha256: str
    v1_path: Path
    v1_sha256: str
    by_slide: Dict[str, Dict[str, dict]]     # slide -> condition -> record (13 conditions)
    slide_order: List[str]

    def stimuli_v1(self, slide_id: str) -> Dict[str, dict]:
        return {c: self.by_slide[slide_id][c] for c in X.CONDITIONS}


def parse_manifest_v2(raw: bytes, v1_path: Path, v1_sha256: str) -> Tuple[Dict[str, Dict[str, dict]], List[str]]:
    """Validate v2 bytes: v1 subset through X.load_manifest + exact new records."""
    v1_digest, v1_by_slide, v1_doc = _v1_document(v1_path, v1_sha256)
    try:
        doc = json.loads(raw)
    except Exception as exc:                              # noqa: BLE001
        raise SuiteError(f"v2 manifest is not JSON: {exc}") from exc
    if not isinstance(doc, dict) or set(doc) != {"policy", "records"}:
        raise SuiteError("v2 top-level keys must be exactly policy, records")
    if _canon(doc["policy"]) != _canon(expected_v2_policy(v1_doc["policy"], v1_digest)):
        raise SuiteError(f"v2 policy {doc['policy']!r} is not the v1 policy renamed with supersedes")
    v1_recs = v1_doc["records"]
    recs = doc["records"]
    n1 = len(v1_recs)
    if not isinstance(recs, list) or len(recs) < n1:
        raise SuiteError("v2 has fewer records than v1")
    for i in range(n1):
        if _canon(recs[i]) != _canon(v1_recs[i]):
            raise SuiteError(f"v2 record {i} differs from v1 record {i}")
    order = _slide_order(v1_recs)
    extra = recs[n1:]
    if len(extra) != len(order):
        raise SuiteError(f"v2 has {len(extra)} new records for {len(order)} v1 slides")
    by_slide: Dict[str, Dict[str, dict]] = {s: dict(v1_by_slide[s]) for s in order}
    for sid, rec in zip(order, extra):
        if not isinstance(rec, dict) or set(rec) != MANIFEST_RECORD_KEYS:
            raise SuiteError(f"v2 new record keys {sorted(rec) if isinstance(rec, dict) else rec!r}")
        if _canon(rec) != _canon(blur_record(sid)):
            raise SuiteError(f"v2 new record for {sid} is not the fixed {BLUR_CONDITION} record "
                             f"(or is out of v1 slide order): {rec!r}")
        p = rec["parameters"]
        if type(p["kernel_size"]) is not int or type(p["sigma"]) is not float:
            raise SuiteError(f"{sid}: {BLUR_CONDITION} parameter types")
        by_slide[sid][BLUR_CONDITION] = rec
    return by_slide, order


def load_manifest_v2(path: Path, expected_sha256: str, v1_path: Path, v1_sha256: str) -> ManifestV2:
    if not expected_sha256:
        raise SuiteError("the v2 manifest sha256 must be given explicitly")
    raw = Path(path).read_bytes()
    digest = _sha(raw)
    if digest != expected_sha256:
        raise SuiteError(f"v2 manifest sha256 {digest} != expected {expected_sha256}")
    by_slide, order = parse_manifest_v2(raw, v1_path, v1_sha256)
    return ManifestV2(Path(path), digest, Path(v1_path), v1_sha256, by_slide, order)


def gen_manifest_v2(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gen-manifest-v2")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--v1", type=Path, required=True)
    p.add_argument("--v1-sha256", default=X.MANIFEST_SHA256)
    a = p.parse_args(argv)
    raw, info = build_manifest_v2(a.v1, a.v1_sha256)
    by_slide, order = parse_manifest_v2(raw, a.v1, a.v1_sha256)   # round trip
    if len(order) != info["v1_slides"]:
        raise SuiteError("round trip slide count")
    digest = _sha(raw)
    dest = a.out_root / "stimuli_v2.json"
    st = publish_idempotent(a.out_root, dest, raw)
    publish_idempotent(a.out_root, a.out_root / "stimuli_v2.json.sha256",
                       f"{digest}  stimuli_v2.json\n".encode())
    print(json.dumps({"stimuli_v2": str(dest), "sha256": digest, "status": st, **info},
                     sort_keys=True), flush=True)
    return 0


# --------------------------------------------------------------------------
# Deterministic image-space blur primitive (item 2)
# --------------------------------------------------------------------------

_W0, _W1, _W2, _W3, _W4, _W5, _W6 = BLUR_WEIGHTS


def _blur_pass(x: np.ndarray, axis: int) -> np.ndarray:
    """One 7-tap pass along ``axis``: reflect (= reflect_101) pad by 3, then
    ``acc = w0*x[-3]; acc = acc + w1*x[-2]; ...; acc = acc + w6*x[+3]`` with separate
    numpy multiply/add ufunc calls (no BLAS/conv/einsum, no fused expression)."""
    pad = [(0, 0)] * x.ndim
    pad[axis] = (3, 3)
    xp = np.pad(x, pad, mode="reflect")
    n = x.shape[axis]

    def tap(k: int) -> np.ndarray:
        idx = [slice(None)] * x.ndim
        idx[axis] = slice(k, k + n)
        return xp[tuple(idx)]

    acc = np.multiply(_W0, tap(0))
    acc = np.add(acc, np.multiply(_W1, tap(1)))
    acc = np.add(acc, np.multiply(_W2, tap(2)))
    acc = np.add(acc, np.multiply(_W3, tap(3)))
    acc = np.add(acc, np.multiply(_W4, tap(4)))
    acc = np.add(acc, np.multiply(_W5, tap(5)))
    acc = np.add(acc, np.multiply(_W6, tap(6)))
    return acc


def gaussian_blur_image(tile_uint8_hwc: np.ndarray) -> np.ndarray:
    """Specification item 2: uint8 HWC -> float64; horizontal pass (columns padded) then
    vertical pass on the float64 intermediate (rows padded), no intermediate rounding;
    ``np.clip(np.rint(acc), 0, 255).astype(np.uint8)`` (round-half-even)."""
    a = np.asarray(tile_uint8_hwc)
    if a.dtype != np.uint8 or a.ndim != 3:
        raise ValueError(f"expected uint8 HWC, got {a.dtype} {a.shape}")
    if a.shape[0] < 4 or a.shape[1] < 4:
        raise ValueError("reflect padding of 3 needs at least 4 pixels per axis")
    x = a.astype(np.float64)
    h = _blur_pass(x, 1)
    v = _blur_pass(h, 0)
    return np.clip(np.rint(v), 0, 255).astype(np.uint8)


def blur_golden_check() -> dict:
    img = np.random.default_rng(0).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    digest = _sha(gaussian_blur_image(img).tobytes())
    return {"expected": BLUR_GOLDEN_SHA256, "observed": digest, "ok": digest == BLUR_GOLDEN_SHA256}


def blur_picture(identity_picture: Image.Image) -> Image.Image:
    """``gaussian_blur_image`` of the identity picture (unshifted read) [A6]."""
    return Image.fromarray(gaussian_blur_image(np.asarray(identity_picture)))


# --------------------------------------------------------------------------
# Fixed rendering environment [A2]
# --------------------------------------------------------------------------


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _cpuinfo(field_name: str) -> Optional[str]:
    for line in _read_text("/proc/cpuinfo").splitlines():
        if line.split(":", 1)[0].strip() == field_name:
            return line.split(":", 1)[1].strip()
    return None


def _os_release() -> Optional[str]:
    for line in _read_text("/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def parse_lshosts(text: str, host: str) -> Dict[str, Any]:
    """``lshosts -w <host>`` -> {'model': ..., 'resources': [...], 'driver_tags': [...]}."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise SuiteError(f"lshosts output for {host} has no host line")
    header = lines[0].split()
    if "model" not in header or header[0] != "HOST_NAME":
        raise SuiteError(f"unexpected lshosts header {lines[0]!r}")
    for ln in lines[1:]:
        cols = ln.split()
        if cols and cols[0] == host:
            m = re.search(r"\(([^)]*)\)\s*$", ln)
            res = m.group(1).split() if m else []
            return {"model": cols[header.index("model")], "resources": res,
                    "driver_tags": sorted(t for t in res if re.fullmatch(r"driver_\d+", t))}
    raise SuiteError(f"lshosts output has no line for {host}")


def lshosts_info(host: str) -> Dict[str, Any]:
    out = subprocess.run(["lshosts", "-w", host], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise SuiteError(f"lshosts -w {host} failed: {out.stderr.strip()[:300]}")
    return parse_lshosts(out.stdout, host)


def _versions() -> Dict[str, Any]:
    import openslide
    import PIL
    import torchvision
    return {"python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "python": sys.version.split()[0], "numpy": np.__version__,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "pillow": PIL.__version__, "openslide_python": openslide.__version__,
            "openslide_library": getattr(openslide, "__library_version__", None)}


def parse_ldd(text: str) -> List[Tuple[str, str]]:
    """``ldd`` output -> [(soname, path)]; vdso skipped; unresolved entries are errors."""
    out: List[Tuple[str, str]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or "vdso" in ln.split()[0]:
            continue
        if "not found" in ln:
            raise SuiteError(f"ldd: unresolved dependency {ln!r}")
        m = re.match(r"^(\S+)\s+=>\s+(\S+)\s+\(0x[0-9a-f]+\)$", ln)
        if m:
            out.append((m.group(1), m.group(2)))
            continue
        m = re.match(r"^(/\S+)\s+\(0x[0-9a-f]+\)$", ln)
        if m:
            out.append((os.path.basename(m.group(1)), m.group(1)))
            continue
        raise SuiteError(f"ldd: unparsed line {ln!r}")
    return out


# [R4] host-driver/display objects: mapped but recorded only (fnmatchcase on the basename).
DRIVER_LIB_PATTERNS: Tuple[str, ...] = (
    "libcuda*", "libnvidia*", "libcudadebugger*", "libGL*", "libEGL*", "libGLX*",
    "libGLdispatch*", "libOpenGL*", "libdrm*", "libX*", "libxcb*")
# [R4] the asserted set must contain a mapped object matching each (decode + numeric path).
# ``libz.so*`` (not ``libz*``) so that e.g. ``libzstd.so`` cannot satisfy the zlib requirement.
REQUIRED_LIB_PATTERNS: Tuple[str, ...] = (
    "libopenslide*", "libjpeg*", "libz.so*", "libpng*", "libtiff*", "libopenjp2*",
    "_imaging*", "libtorch_cpu*")
SONAME_SOURCE = "basename of realpath (not the ELF DT_SONAME)"
ELF_MAGIC = b"\x7fELF"                       # [R6a] only ELF objects are asserted
LOCALE_ENV_KEYS: Tuple[str, ...] = ("LC_ALL", "LANG", "LC_CTYPE")     # [R6a] recorded only
PROBE_READ_LOCATION = (0, 0)                 # one level-0 512x512 read, as X/the suite render


def parse_maps(text: str) -> List[str]:
    """``/proc/<pid>/maps`` -> distinct pathname fields that start with ``/`` (file-backed
    mappings, possibly ending in `` (deleted)``), in first-seen order."""
    out: List[str] = []
    seen = set()
    for ln in text.splitlines():
        parts = ln.split(None, 5)
        if len(parts) < 6:
            continue
        path = parts[5].strip()
        if path.startswith("/") and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def ldd_closure(openslide_lib: str = OPENSLIDE_LIB, ldd_text: Optional[str] = None) -> dict:
    """``ldd /lib64/libopenslide.so.0`` closure -- recorded only [R4]; errors are recorded."""
    doc: Dict[str, Any] = {"target": openslide_lib, "libraries": [], "error": None}
    try:
        if ldd_text is None:
            r = subprocess.run(["ldd", openslide_lib], capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                raise SuiteError(f"ldd {openslide_lib} failed: {r.stderr.strip()[:300]}")
            ldd_text = r.stdout
        for name, path in parse_ldd(ldd_text):
            real = os.path.realpath(path)
            doc["libraries"].append({"name": name, "realpath": real,
                                     "sha256": X.sha256_file(Path(real))})
        doc["libraries"].sort(key=lambda d: (d["realpath"], d["name"]))
    except Exception as exc:                              # noqa: BLE001
        doc["error"] = f"{type(exc).__name__}: {exc}"
    return doc


def capture_mapped_libs(*, maps_text: Optional[str] = None, ldd_text: Optional[str] = None,
                        openslide_lib: str = OPENSLIDE_LIB) -> dict:
    """[R4]/[R6a] the shared objects/files actually mapped by THIS process
    (``/proc/self/maps``), deduplicated by realpath. A mapped regular file is asserted
    (realpath, SHA-256, SONAME = basename) only if its first four bytes are the ELF magic
    ``\\x7fELF`` and its basename does not match :data:`DRIVER_LIB_PATTERNS`. Every other
    mapped regular file is recorded only (realpath, size, SHA-256): driver-pattern matches and
    ``non_elf`` (locale archives, gconv cache, zoneinfo, data files). Deleted/unstat-able/
    non-regular mappings are recorded (not hashed). The ``ldd`` closure of libopenslide is
    recorded only. Call it after the import/probe steps (see :func:`reference_lib_probe`)."""
    if maps_text is None:
        maps_text = Path("/proc/self/maps").read_text()
    asserted: Dict[str, dict] = {}
    driver: Dict[str, dict] = {}
    non_elf: Dict[str, dict] = {}
    not_hashed: List[dict] = []
    for path in parse_maps(maps_text):
        if path.endswith(" (deleted)"):
            not_hashed.append({"path": path, "reason": "deleted"})
            continue
        real = os.path.realpath(path)
        try:
            st = os.stat(real)
        except OSError as exc:
            not_hashed.append({"path": path, "reason": f"stat: {exc}"})
            continue
        if not stat.S_ISREG(st.st_mode):
            not_hashed.append({"path": path, "reason": "not a regular file"})
            continue
        base = os.path.basename(real)
        if real in asserted or real in driver or real in non_elf:
            continue
        with open(real, "rb") as fh:
            is_elf = fh.read(4) == ELF_MAGIC
        if any(fnmatch.fnmatchcase(base, pat) for pat in DRIVER_LIB_PATTERNS):
            driver[real] = {"realpath": real, "size": st.st_size,
                            "sha256": X.sha256_file(Path(real)), "soname": base}
        elif not is_elf:
            non_elf[real] = {"realpath": real, "size": st.st_size,
                             "sha256": X.sha256_file(Path(real)), "soname": base}
        else:
            asserted[real] = {"realpath": real, "sha256": X.sha256_file(Path(real)),
                              "soname": base}
    return {
        "method": "proc_self_maps", "soname_source": SONAME_SOURCE,
        "asserted": [asserted[k] for k in sorted(asserted)],
        "recorded_only": {
            "driver_pattern_matches": [driver[k] for k in sorted(driver)],
            "non_elf": [non_elf[k] for k in sorted(non_elf)],
            "not_hashed": not_hashed,
            "ldd_openslide_closure": ldd_closure(openslide_lib, ldd_text),
        },
        "driver_patterns": list(DRIVER_LIB_PATTERNS),
        "required_patterns": list(REQUIRED_LIB_PATTERNS),
    }


def required_lib_matches(asserted: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    return {pat: sorted(e["realpath"] for e in asserted
                        if fnmatch.fnmatchcase(os.path.basename(e["realpath"]), pat))
            for pat in REQUIRED_LIB_PATTERNS}


def check_required_libs(asserted: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    """[R4] the capture fails unless every required pattern has a mapped asserted object."""
    m = required_lib_matches(asserted)
    missing = [pat for pat, hits in m.items() if not hits]
    if missing:
        raise SuiteError(f"mapped reference-library set lacks {missing} (decode/numeric path "
                         "not covered); capture refused")
    return m


def torchstain_member_hashes(prefix: Path) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for member in X.TORCHSTAIN_MEMBERS:
        p = Path(prefix) / member
        out[member] = X.sha256_file(p) if p.is_file() else None
    return out


@dataclass
class EnvProbe:
    """Injectable probes for the environment block (tests replace them)."""
    hostname: Callable[[], str] = socket.gethostname
    lsf_host_info: Callable[[str], Dict[str, Any]] = lshosts_info
    cpu_vendor: Callable[[], Optional[str]] = lambda: _cpuinfo("vendor_id")
    cpu_model: Callable[[], Optional[str]] = lambda: _cpuinfo("model name")
    cpu_capability: Callable[[], str] = lambda: torch.backends.cpu.get_cpu_capability()
    versions: Callable[[], Dict[str, Any]] = _versions
    kernel: Callable[[], str] = platform.release
    os_release: Callable[[], Optional[str]] = _os_release
    reference_libs: Callable[[], Dict[str, Any]] = capture_mapped_libs
    torchstain_members: Callable[[Path], Dict[str, Optional[str]]] = torchstain_member_hashes
    extractor_sha256: Callable[[], str] = lambda: X.sha256_file(Path(X.__file__).resolve())


def load_reference_libs(path: Path) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != REFLIBS_SCHEMA:
        # [R6a] v1/v2 pins (pre-ELF-only asserted set) cannot be mixed with v3 observations
        raise EnvironmentMismatch(f"{path}: reference-library pin schema {doc.get('schema')!r} "
                                  f"is refused; only {REFLIBS_SCHEMA} pins are accepted "
                                  "(re-pin with pin-reference-libs)")
    if not isinstance(doc.get("asserted"), list):
        raise EnvironmentMismatch(f"{path}: not a {REFLIBS_SCHEMA} document")
    return doc


def environment_block(probe: EnvProbe, *, torchstain_prefix: Optional[Path],
                      reference_libs_path: Optional[Path],
                      allow_missing_reference_libs: bool = False,
                      enforce: bool = True, require_threads: bool = True,
                      probe_steps: Optional[Callable[[], dict]] = None) -> dict:
    """Asserted/recorded environment split [A2]. ``enforce`` raises EnvironmentMismatch on
    any asserted deviation (before rendering); capture mode records deviations instead.

    [R4] order: all non-library checks (and the presence/schema of ``--reference-libs``) are
    evaluated first and, when enforcing, raise before anything else; then ``probe_steps``
    performs the import/WSI-read/Reinhard/blur-golden probe (:func:`reference_lib_probe`);
    only then is ``/proc/self/maps`` read and the asserted mapped set compared with the pin."""
    host = probe.hostname()
    deviations: List[str] = []
    asserted: Dict[str, Any] = {}

    def check(key: str, observed: Any, expected: Any) -> None:
        asserted[key] = {"observed": observed, "expected": expected}
        if observed != expected:
            deviations.append(f"{key}: observed {observed!r} != expected {expected!r}")

    lsf_info: Dict[str, Any] = {}
    try:
        lsf_info = probe.lsf_host_info(host)
        lsf_model = lsf_info.get("model")
    except Exception as exc:                              # noqa: BLE001
        lsf_model = f"unavailable: {type(exc).__name__}: {exc}"
    check("lsf_model", lsf_model, EXPECTED_ENV["lsf_model"])
    check("cpu_vendor", probe.cpu_vendor(), EXPECTED_ENV["cpu_vendor"])
    check("cpu_capability", probe.cpu_capability(), EXPECTED_ENV["cpu_capability"])
    vers = probe.versions()
    for k in ("python_major_minor", "numpy", "torch", "torchvision", "pillow",
              "openslide_python", "openslide_library"):
        check(k, vers.get(k), EXPECTED_ENV[k])
    if torchstain_prefix is not None:
        check("torchstain_members", probe.torchstain_members(Path(torchstain_prefix)),
              dict(X.TORCHSTAIN_MEMBERS))
    else:
        asserted["torchstain_members"] = {"observed": None, "expected": dict(X.TORCHSTAIN_MEMBERS),
                                          "note": "no prefix given (capture without torchstain)"}
        if enforce:
            deviations.append("torchstain_members: no --torchstain-prefix")
    check("extractor_code_sha256", probe.extractor_sha256(), EXTRACTOR_SHA256)
    threads = {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
               "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
               "torch_num_threads": torch.get_num_threads()}
    if require_threads:
        check("threads", threads, {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                   "torch_num_threads": 1})
    else:
        asserted["threads"] = {"observed": threads, "note": "not asserted (capture)"}
    ref: Optional[dict] = None
    if reference_libs_path is not None:
        ref = load_reference_libs(reference_libs_path)
    elif not allow_missing_reference_libs and enforce:
        deviations.append("reference_libs: --reference-libs is required "
                          "(or --allow-missing-reference-libs explicitly)")
    if enforce and deviations:
        raise EnvironmentMismatch("fixed rendering environment violated: " + "; ".join(deviations))
    probe_record = probe_steps() if probe_steps is not None else None
    libs = probe.reference_libs()
    observed = libs["asserted"]
    if ref is not None:
        check("reference_libs", observed, ref["asserted"])
        asserted["reference_libs"]["pinned_file"] = str(reference_libs_path)
        asserted["reference_libs"]["pinned_file_sha256"] = X.sha256_file(Path(reference_libs_path))
        asserted["reference_libs"].pop("expected")
        if observed != ref["asserted"]:
            exp = {(e["realpath"], e["sha256"]) for e in ref["asserted"]}
            obs = {(e["realpath"], e["sha256"]) for e in observed}
            asserted["reference_libs"]["only_observed"] = sorted(obs - exp)[:200]
            asserted["reference_libs"]["only_pinned"] = sorted(exp - obs)[:200]
    elif allow_missing_reference_libs:
        asserted["reference_libs"] = {"observed": observed, "status": "NOT_ASSERTED",
                                      "note": "--allow-missing-reference-libs given explicitly"}
    else:
        asserted["reference_libs"] = {"observed": observed, "status": "NOT_ASSERTED"}
    asserted["reference_libs"]["method"] = libs.get("method")
    asserted["reference_libs"]["soname_source"] = libs.get("soname_source")
    recorded = {"cpu_model": probe.cpu_model(), "kernel": probe.kernel(),
                "os_release": probe.os_release(), "hostname": host,
                "lsf_job": os.environ.get("LSB_JOBID"), "lsf_index": os.environ.get("LSB_JOBINDEX"),
                "lsf_queue": os.environ.get("LSB_QUEUE"),
                "lsf_resources": lsf_info.get("resources"),
                "gpu_driver_tags": lsf_info.get("driver_tags"),
                "python": vers.get("python"), "argv": list(sys.argv),
                "reference_libs_recorded_only": libs.get("recorded_only"),
                "reference_lib_probe": probe_record}
    # [R6a] locale variables are recorded only (the job wrapper exports LC_ALL=C); never a
    # deviation. A non-"C" LC_ALL leaves a recorded note.
    locale_env = {k: os.environ.get(k) for k in LOCALE_ENV_KEYS}
    recorded["locale"] = locale_env
    if locale_env["LC_ALL"] != "C":
        recorded["locale_note"] = (f"LC_ALL={locale_env['LC_ALL']!r} != 'C' (suite jobs and "
                                   "[A2] captures export LC_ALL=C); recorded only")
    block = {"asserted": asserted, "recorded": recorded, "deviations": deviations,
             "ok": not deviations, "suite_code_sha256": SUITE_CODE_SHA256,
             "allow_missing_reference_libs": bool(allow_missing_reference_libs),
             "time": _now()}
    if enforce and deviations:
        raise EnvironmentMismatch("fixed rendering environment violated: " + "; ".join(deviations))
    return block


def set_single_thread() -> None:
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise EnvironmentMismatch(f"torch.get_num_threads() == {torch.get_num_threads()} after "
                                  "set_num_threads(1)")


def _f32_bits(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        if values.dtype != torch.float32:
            raise PinnedFitError(f"Reinhard target dtype {values.dtype} != float32")
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray([float(v) for v in values], dtype=np.float64)
        if not np.array_equal(arr.astype(np.float32).astype(np.float64), arr):
            return np.full(len(arr), 0xFFFFFFFF, np.uint64)          # not a float32 value
        arr = arr.astype(np.float32)
    arr = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    return arr.view(np.uint32).astype(np.uint64)


def _bits_equal(values: Any, ref: Sequence[float]) -> bool:
    try:
        a = _f32_bits(values)
    except PinnedFitError:
        raise
    except Exception:                                    # noqa: BLE001
        return False
    b = np.asarray(ref, np.float32).view(np.uint32).astype(np.uint64)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def check_pinned_fit(means: Any, stds: Any) -> None:
    if not (_bits_equal(means, PINNED_MEANS) and _bits_equal(stds, PINNED_STDS)):
        raise PinnedFitError(f"Reinhard fit {list(np.asarray(_as_list(means)))} / "
                             f"{list(np.asarray(_as_list(stds)))} is not the pinned 1-thread fit "
                             f"{list(PINNED_MEANS)} / {list(PINNED_STDS)}")


def _as_list(v: Any) -> List[float]:
    if isinstance(v, torch.Tensor):
        return [float(t) for t in v.detach().cpu().reshape(-1)]
    return [float(t) for t in v]


def make_pinned_reinhard(template: Path, torchstain_prefix: Optional[Path],
                         factory: Optional[Callable] = None):
    """``torch.set_num_threads(1)`` BEFORE constructing ``X.TorchstainReinhard``, then assert
    float32 bit-equality of ``normalizer.target_means/target_stds`` with the pinned fit."""
    set_single_thread()
    factory = factory or X.TorchstainReinhard
    rh = factory(Path(template), X.TEMPLATE_SHA256, torchstain_prefix)
    nz = rh.normalizer
    check_pinned_fit(nz.target_means, nz.target_stds)
    return rh


def gpu_info_default(device: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"device": device,
                            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
                            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if not device.startswith("cuda"):
        info.update({"gpu_model": None, "driver": None})
        return info
    idx = torch.cuda.current_device()
    info["gpu_model"] = torch.cuda.get_device_name(idx)
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,uuid",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
        info["nvidia_smi"] = r.stdout.strip().splitlines()
        info["driver"] = (info["nvidia_smi"][0].split(",")[2].strip()
                          if info["nvidia_smi"] else None)
    except Exception as exc:                              # noqa: BLE001
        info["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
        info["driver"] = None
    return info


# --------------------------------------------------------------------------
# Inputs shared by the rendering sub-commands
# --------------------------------------------------------------------------


@dataclass
class Inputs:
    manifest: ManifestV2
    queries_path: Path
    queries_sha256: str
    rows: List[dict]
    row_by_slide: Dict[str, dict]


def add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest-v2", type=Path, required=True)
    p.add_argument("--manifest-v2-sha256", required=True)
    p.add_argument("--v1", type=Path, required=True, help="frozen manifest v1 (stimuli.json)")
    p.add_argument("--v1-sha256", default=X.MANIFEST_SHA256)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--queries-sha256", default=X.QUERIES_SHA256)


def add_env_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--template", type=Path, default=X.DEFAULT_TEMPLATE)
    p.add_argument("--torchstain-prefix", type=Path, required=True)
    p.add_argument("--reference-libs", type=Path, default=None)
    p.add_argument("--allow-missing-reference-libs", action="store_true",
                   help="explicit, recorded waiver of the pinned reference-library assertion")
    p.add_argument("--probe-wsi", type=Path, default=None,
                   help="WSI opened for one read_region before the mapped-library check [R4] "
                        "(default: the WSI of queries.csv row 0)")


def load_inputs(a: argparse.Namespace) -> Inputs:
    if not a.queries_sha256:
        raise SuiteError("--queries-sha256 must be non-empty")
    m = load_manifest_v2(a.manifest_v2, a.manifest_v2_sha256, a.v1, a.v1_sha256)
    qsha, rows = X.load_queries(a.queries, a.queries_sha256)
    by = {r["slide_id"]: r for r in rows}
    missing = [r["slide_id"] for r in rows if r["slide_id"] not in m.by_slide]
    if missing:
        raise SuiteError(f"{len(missing)} query slides have no v2 stimuli, e.g. {missing[:5]}")
    return Inputs(m, Path(a.queries), qsha, rows, by)


def check_template_digest(stimuli: Mapping[str, dict], sid: str) -> None:
    d = stimuli["reinhard"]["parameters"]["template_sha256"]
    if d != X.TEMPLATE_SHA256:
        raise SuiteError(f"{sid}: manifest template digest {d} != {X.TEMPLATE_SHA256}")


def wsi_stat(path: Path) -> Dict[str, Any]:
    st = os.stat(path)
    return {"path": str(path), "st_size": st.st_size, "st_mtime_ns": st.st_mtime_ns}


def openslide_open(path: str):
    """Same call as X.openslide_opener (X line ~401): ``openslide.OpenSlide(path)``."""
    import openslide
    return openslide.OpenSlide(path)


# --------------------------------------------------------------------------
# Rendering: 13 pictures per tile, digests, streaming verification
# --------------------------------------------------------------------------


def picture_digests(pictures: Mapping[str, Image.Image]) -> np.ndarray:
    out = np.empty((len(SUITE_CONDITIONS), 32), np.uint8)
    for j, c in enumerate(SUITE_CONDITIONS):
        im = pictures[c]
        if im.mode != "RGB" or im.size != (512, 512):
            raise SuiteError(f"{c}: picture mode/size {im.mode} {im.size} != RGB (512, 512)")
        out[j] = np.frombuffer(hashlib.sha256(im.tobytes()).digest(), np.uint8)
    return out


def render_pictures(slide: Any, loc: Tuple[int, int], shift: Tuple[int, int],
                    stimuli_v1: Dict[str, dict], reinhard: Callable,
                    patch_size: int = 512) -> Tuple[Dict[str, Image.Image], bool]:
    """Mirrors X.SlideTiles.__getitem__ (X lines 423-436): two level-0 ``read_region`` calls
    of ``(patch_size, patch_size)`` converted to RGB -- the tile at ``(x, y)`` and the
    shifted read at ``(x + dx, y + dy)`` -- then ``X.condition_images``. The blur arm is
    applied to the identity (unshifted) picture [A6]."""
    x, y = loc
    region = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
    dx, dy = shift
    shifted = slide.read_region((x + dx, y + dy), 0, (patch_size, patch_size)).convert("RGB")
    images, finite = X.condition_images(region, shifted, stimuli_v1, reinhard)
    pictures = {c: images[c] for c in X.CONDITIONS}
    pictures[BLUR_CONDITION] = blur_picture(images["identity"])
    return pictures, finite


class RenderTiles(torch.utils.data.Dataset):
    """Per-tile renderer; one reader handle per worker process.

    ``expected`` (n, 13, 32): verify every picture's digest before any transform; on a
    mismatch the item carries ``mismatch`` and no tensors. ``transform`` + ``select``:
    tensors of the selected conditions (v1 arms via ``X.encode_conditions`` so the bytes
    are X's; ``gaussian_blur_image`` via ``transform(picture)``)."""

    def __init__(self, slide_path: str, coords: List[dict], stimuli_v1: Dict[str, dict],
                 reinhard: Callable, opener: Callable, *, expected: Optional[np.ndarray] = None,
                 transform: Optional[Callable] = None, select: Sequence[str] = (),
                 patch_size: int = 512):
        self.slide_path = str(slide_path)
        self.locs = [(int(c["wsi_loc"][0]), int(c["wsi_loc"][1])) for c in coords]
        self.stimuli = stimuli_v1
        self.reinhard = reinhard
        self.opener = opener
        self.expected = expected
        self.transform = transform
        self.select = tuple(select)
        self.patch_size = patch_size
        self.shift = X.shift_of(stimuli_v1)
        self._slide = None
        if expected is not None and expected.shape != (len(self.locs), len(SUITE_CONDITIONS), 32):
            raise SuiteError(f"expected digests shape {expected.shape}")

    def __len__(self) -> int:
        return len(self.locs)

    def __getitem__(self, i: int) -> dict:
        x, y = self.locs[i]
        try:
            if self._slide is None:
                self._slide = self.opener(self.slide_path)
            pictures, finite = render_pictures(self._slide, (x, y), self.shift, self.stimuli,
                                               self.reinhard, self.patch_size)
            digests = picture_digests(pictures)
        except Exception as exc:                          # noqa: BLE001
            return {"index": i, "error": {"type": type(exc).__name__, "message": str(exc)[:500],
                                          "wsi_loc": [x, y]}}
        item: Dict[str, Any] = {"index": i, "reinhard_finite": finite}
        if self.expected is None:
            item["digests"] = digests
            return item
        bad = [c for j, c in enumerate(SUITE_CONDITIONS)
               if not np.array_equal(digests[j], self.expected[i, j])]
        if bad:
            item["mismatch"] = bad
            return item
        if self.transform is not None:
            try:
                tensors: Dict[str, torch.Tensor] = {}
                if any(c in X.CONDITIONS for c in self.select):
                    stack = X.encode_conditions({c: pictures[c] for c in X.CONDITIONS},
                                                self.transform, self.stimuli)
                    for j, c in enumerate(X.CONDITIONS):
                        if c in self.select:
                            tensors[c] = stack[j]
                if BLUR_CONDITION in self.select:
                    tensors[BLUR_CONDITION] = self.transform(pictures[BLUR_CONDITION])
            except Exception as exc:                      # noqa: BLE001
                return {"index": i, "error": {"type": type(exc).__name__,
                                              "message": f"transform: {exc}"[:500],
                                              "wsi_loc": [x, y]}}
            item["tensors"] = tensors
        return item


def _collate_list(batch):
    return batch


def _suite_worker_init(_):
    torch.set_num_threads(1)


# Worker processes return tensors through AF_UNIX sockets created under the temp directory
# (<tmp>/pymp-XXXXXXXX/listener-XXXXXXXX); a path longer than sun_path (107 bytes) makes the
# DataLoader hang silently rather than fail, so refuse such a TMPDIR up front.
UNIX_SOCKET_PATH_MAX = 107
MP_SOCKET_SUFFIX_LEN = len("/pymp-XXXXXXXX/listener-XXXXXXXX")


def check_worker_tempdir(n_workers: int) -> None:
    if n_workers <= 0:
        return
    tmp = tempfile.gettempdir()
    if len(os.fsencode(tmp)) + MP_SOCKET_SUFFIX_LEN > UNIX_SOCKET_PATH_MAX:
        raise SuiteError(f"temp directory {tmp!r} is too long for worker AF_UNIX sockets "
                         f"(> {UNIX_SOCKET_PATH_MAX - MP_SOCKET_SUFFIX_LEN} bytes); set a short TMPDIR")


def make_loader(ds: torch.utils.data.Dataset, processes: int, loader_batch: int,
                prefetch_factor: int = 2):
    check_worker_tempdir(processes)
    return torch.utils.data.DataLoader(
        ds, batch_size=loader_batch, shuffle=False, num_workers=processes,
        collate_fn=_collate_list, worker_init_fn=_suite_worker_init if processes else None,
        prefetch_factor=prefetch_factor if processes else None, persistent_workers=False)


def render_digests(slide_path: str, coords: List[dict], stimuli_v1: Dict[str, dict],
                   reinhard: Callable, opener: Callable, processes: int,
                   loader_batch: int) -> Tuple[np.ndarray, List[int], List[dict]]:
    n = len(coords)
    arr = np.zeros((n, len(SUITE_CONDITIONS), 32), np.uint8)
    seen = np.zeros(n, bool)
    nonfinite: List[int] = []
    errors: List[dict] = []
    ds = RenderTiles(slide_path, coords, stimuli_v1, reinhard, opener)
    for batch in make_loader(ds, processes, loader_batch):
        for it in batch:
            i = it["index"]
            if "error" in it:
                errors.append(it["error"] | {"index": i})
                continue
            arr[i] = it["digests"]
            seen[i] = True
            if not it["reinhard_finite"]:
                nonfinite.append(i)
    if not errors and not seen.all():
        errors.append({"type": "Incomplete", "message": f"{int((~seen).sum())} tiles not rendered"})
    return arr, sorted(nonfinite), errors


def aggregate_digests(arr: np.ndarray) -> Dict[str, str]:
    return {c: _sha(np.ascontiguousarray(arr[:, j, :]).tobytes())
            for j, c in enumerate(SUITE_CONDITIONS)}


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr, dtype=np.uint8), allow_pickle=False)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Fingerprint pass (item 3)
# --------------------------------------------------------------------------


def fp_paths(fp_dir: Path, sid: str) -> Tuple[Path, Path]:
    return fp_dir / f"{sid}.fp.npy", fp_dir / f"{sid}.fp.json"


def validate_fp_pair(fp_dir: Path, sid: str, *, n_tiles: int, manifest_v2_sha256: str,
                     load_array: bool = True) -> Tuple[str, str, Optional[dict], Optional[np.ndarray]]:
    """('absent'|'complete'|'invalid', reason, json, array): validates sha, n_tiles,
    manifest sha, shape/dtype, content and aggregate digests -- not mere existence."""
    npy, js = fp_paths(fp_dir, sid)
    has_n, has_j = npy.exists(), js.exists()
    if not has_n and not has_j:
        return "absent", "", None, None
    if not (has_n and has_j):
        return "invalid", f"orphan fingerprint (npy={has_n}, json={has_j})", None, None
    try:
        doc = json.loads(js.read_text())
    except Exception as exc:                              # noqa: BLE001
        return "invalid", f"unreadable fp.json: {exc}", None, None
    checks = [(doc.get("schema") == FP_SCHEMA, "schema"),
              (doc.get("slide_id") == sid, "slide_id"),
              (doc.get("n_tiles") == n_tiles, "n_tiles"),
              (doc.get("manifest_v2_sha256") == manifest_v2_sha256, "manifest_v2_sha256"),
              (doc.get("conditions") == list(SUITE_CONDITIONS), "conditions"),
              (isinstance(doc.get("array"), dict)
               and doc["array"].get("npy_bytes") == npy.stat().st_size, "npy_bytes")]
    for ok, what in checks:
        if not ok:
            return "invalid", f"fp.json {what} mismatch", doc, None
    raw = npy.read_bytes()
    if _sha(raw) != doc["array"].get("npy_sha256"):
        return "invalid", "npy sha256 != fp.json", doc, None
    if not load_array:
        return "complete", "", doc, None
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:                              # noqa: BLE001
        return "invalid", f"unloadable npy: {exc}", doc, None
    if arr.dtype != np.uint8 or arr.shape != (n_tiles, len(SUITE_CONDITIONS), 32):
        return "invalid", f"array {arr.dtype} {arr.shape}", doc, None
    if _sha(np.ascontiguousarray(arr).tobytes()) != doc["array"].get("content_sha256"):
        return "invalid", "array content sha256 != fp.json", doc, None
    if aggregate_digests(arr) != doc.get("aggregate_digests"):
        return "invalid", "aggregate digests != fp.json", doc, None
    if not np.array_equal(arr[:, SUITE_CONDITIONS.index("gaussian_blur")], arr[:, 0]):
        return "invalid", "legacy gaussian_blur picture digest != identity digest", doc, None
    return "complete", "", doc, arr


def write_failed(root: Path, failed_dir: Path, sid: str, obj: dict) -> Path:
    obj = dict(obj, slide_id=sid, time=_now(), host=socket.gethostname(),
               lsf_job=os.environ.get("LSB_JOBID"), lsf_index=os.environ.get("LSB_JOBINDEX"),
               suite_code_sha256=SUITE_CODE_SHA256, extractor_code_sha256=EXTRACTOR_SHA256)
    dest = failed_dir / f"{sid}.{_tag()}.json"
    publish(root, dest, _dump_pretty(obj))
    print(f"[suite] FAILED {sid}: {json.dumps(obj, default=str)[:500]}", flush=True)
    return dest


def fingerprint_slide(root: Path, fp_dir: Path, inp: Inputs, sid: str, *, reinhard: Any,
                      reinhard_info: dict, env: dict, opener: Callable,
                      slide_path_of: Callable[[str], Path], processes: int,
                      loader_batch: int, golden: dict) -> str:
    """'written' | 'skipped' | 'failed'."""
    row = inp.row_by_slide[sid]
    n_decl = int(row["feature_rows"])
    failed_dir = fp_dir / "failed"
    st, why, _, _ = validate_fp_pair(fp_dir, sid, n_tiles=n_decl,
                                     manifest_v2_sha256=inp.manifest.sha256)
    if st == "complete":
        return "skipped"
    if st == "invalid":
        write_failed(root, failed_dir, sid, {"stage": "existing_output", "reason": why})
        return "failed"
    t0, c0 = time.time(), _cpu_seconds()
    stimuli = inp.manifest.stimuli_v1(sid)
    try:
        check_template_digest(stimuli, sid)
        coords = X.load_bank_coords(row)
        if len(coords) != n_decl:
            raise SuiteError(f"{len(coords)} coords != feature_rows {n_decl}")
        slide_path = Path(slide_path_of(sid))
        stat0 = wsi_stat(slide_path)
    except Exception as exc:                              # noqa: BLE001
        write_failed(root, failed_dir, sid, {"stage": "inputs",
                                             "reason": f"{type(exc).__name__}: {exc}"})
        return "failed"
    try:
        arr, nonfinite, errors = render_digests(str(slide_path), coords, stimuli, reinhard,
                                                opener, processes, loader_batch)
    except Exception as exc:                              # noqa: BLE001
        write_failed(root, failed_dir, sid, {"stage": "render",
                                             "reason": f"{type(exc).__name__}: {exc}",
                                             "traceback": traceback.format_exc()[-4000:]})
        return "failed"
    if errors:
        write_failed(root, failed_dir, sid, {"stage": "tile_render", "n_tiles": n_decl,
                                             "n_failed_tiles": len(errors),
                                             "tile_errors": errors[:200]})
        return "failed"
    if not np.array_equal(arr[:, SUITE_CONDITIONS.index("gaussian_blur")], arr[:, 0]):
        write_failed(root, failed_dir, sid, {"stage": "invariant",
                                             "reason": "legacy gaussian_blur digest != identity"})
        return "failed"
    stat1 = wsi_stat(slide_path)
    if stat1 != stat0:
        write_failed(root, failed_dir, sid, {"stage": "wsi_changed", "before": stat0,
                                             "after": stat1})
        return "failed"
    raw = npy_bytes(arr)
    doc = {
        "schema": FP_SCHEMA, "slide_id": sid, "n_tiles": n_decl,
        "conditions": list(SUITE_CONDITIONS),
        "array": {"file": f"{sid}.fp.npy", "dtype": "uint8",
                  "shape": [n_decl, len(SUITE_CONDITIONS), 32],
                  "content_sha256": _sha(np.ascontiguousarray(arr).tobytes()),
                  "npy_sha256": _sha(raw), "npy_bytes": len(raw)},
        "aggregate_digests": aggregate_digests(arr),
        "legacy_gaussian_blur_picture": "untreated tile (post-preprocessing arm); "
                                        "digest == identity digest",
        "legacy_gaussian_blur_equals_identity": True,
        "reinhard_nonfinite_tile_indices": nonfinite,
        "wsi": stat0,
        "environment": env,
        "manifest_v2_path": str(inp.manifest.path), "manifest_v2_sha256": inp.manifest.sha256,
        "manifest_v1_sha256": inp.manifest.v1_sha256,
        "queries_path": str(inp.queries_path), "queries_sha256": inp.queries_sha256,
        "coords_source": row["resolved_path"], "coords_source_sha256": row["feature_sha256"],
        "extractor_code_sha256": EXTRACTOR_SHA256, "suite_code_sha256": SUITE_CODE_SHA256,
        "reinhard": {"target_means": list(PINNED_MEANS), "target_stds": list(PINNED_STDS),
                     "info": reinhard_info},
        "blur_golden": golden,
        "render_processes": processes, "loader_batch": loader_batch,
        "wall_seconds": round(time.time() - t0, 3),
        "cpu_seconds": round(_cpu_seconds() - c0, 3),
        "written_at": _now(),
    }
    npy, js = fp_paths(fp_dir, sid)
    try:
        publish(root, npy, raw)
        publish(root, js, _dump_pretty(doc))
    except FileExistsError as exc:
        # [R6d] the refused content's digest, so finalize can tell an identical duplicate
        # (superseded) from non-determinism (blocking)
        write_failed(root, failed_dir, sid, {"stage": "publish",
                                             "reason": f"refused to overwrite: {exc}",
                                             "npy_sha256": doc["array"]["npy_sha256"],
                                             "content_sha256": doc["array"]["content_sha256"]})
        return "failed"
    print(f"[suite] fingerprint {sid}: n={n_decl} wall={doc['wall_seconds']}s "
          f"nonfinite_reinhard={len(nonfinite)}", flush=True)
    return "written"


def plan_tasks(rows: Sequence[dict], cap: int = FP_TASK_MAX_TILES) -> List[dict]:
    """[A7] pack by tile count: slides sorted by tile count (desc, ties by slide_id) and
    first-fit into tasks of <= ``cap`` tiles; a slide with >= ``cap`` tiles gets its own task."""
    items = sorted(((int(r["feature_rows"]), r["slide_id"]) for r in rows),
                   key=lambda t: (-t[0], t[1]))
    bins: List[dict] = []
    for n, sid in items:
        if n >= cap:
            bins.append({"slides": [sid], "tiles": n, "solo": True})
            continue
        for b in bins:
            if not b["solo"] and b["tiles"] + n <= cap:
                b["slides"].append(sid)
                b["tiles"] += n
                break
        else:
            bins.append({"slides": [sid], "tiles": n, "solo": False})
    return [{"task_index": k, "slides": b["slides"], "tiles": b["tiles"], "solo": b["solo"]}
            for k, b in enumerate(bins, 1)]


def plan_tasks_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="plan-tasks")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--queries-sha256", default=X.QUERIES_SHA256)
    p.add_argument("--max-tiles", type=int, default=FP_TASK_MAX_TILES)
    a = p.parse_args(argv)
    if not a.queries_sha256:
        raise SuiteError("--queries-sha256 must be non-empty")
    qsha, rows = X.load_queries(a.queries, a.queries_sha256)
    tasks = plan_tasks(rows, a.max_tiles)
    doc = {"schema": "r2_1_perturb_suite_tasks_v1", "queries_sha256": qsha,
           "max_tiles": a.max_tiles, "n_tasks": len(tasks), "n_slides": len(rows),
           "total_tiles": sum(t["tiles"] for t in tasks),
           "rule": "sort by tile count desc (ties slide_id), first-fit <= max_tiles; "
                   ">= max_tiles alone", "tasks": tasks}
    dest = a.out_root / "fingerprints" / "TASKS.json"
    st = publish_idempotent(a.out_root, dest, _dump_pretty(doc))
    print(json.dumps({"tasks": str(dest), "status": st, "n_tasks": len(tasks),
                      "total_tiles": doc["total_tiles"]}), flush=True)
    return 0


def _task_slides(tasks_path: Path, task_index: int, queries_sha256: str) -> List[str]:
    doc = json.loads(Path(tasks_path).read_text())
    if doc.get("queries_sha256") != queries_sha256:
        raise SuiteError(f"TASKS.json queries sha {doc.get('queries_sha256')} != {queries_sha256}")
    hits = [t for t in doc["tasks"] if t["task_index"] == task_index]
    if len(hits) != 1:
        raise SuiteError(f"task index {task_index} not in {tasks_path}")
    return list(hits[0]["slides"])


def _select_slides(a: argparse.Namespace, inp: Inputs) -> List[str]:
    if getattr(a, "tasks", None) is not None:
        if a.slides:
            raise SuiteError("--tasks and --slides are exclusive")
        if not a.task_index or a.task_index < 1:
            raise SuiteError("--task-index (1-based, e.g. $LSB_JOBINDEX) is required with --tasks")
        sel = _task_slides(a.tasks, a.task_index, inp.queries_sha256)
    else:
        sel = _split_list(a.slides)
        if not sel:
            raise SuiteError("give --tasks/--task-index or --slides")
    unknown = [s for s in sel if s not in inp.row_by_slide]
    if unknown or len(set(sel)) != len(sel):
        raise SuiteError(f"slides not in queries.csv or duplicated: {unknown[:10]}")
    return sel


def reference_lib_probe(*, template: Path, torchstain_prefix: Optional[Path],
                        reinhard_factory: Optional[Callable], opener: Callable,
                        probe_wsi: Path) -> Tuple[dict, Any]:
    """[R4] the steps after which the mapped closure is read: the suite's import order
    (prosper -> torch/torchvision/PIL/cv2 -> openslide -> pinned torchstain), one pinned
    1-thread Reinhard fit (bit-asserted), one blur-golden evaluation and one level-0
    512x512 ``read_region`` of ``probe_wsi`` through ``opener`` (no output is produced from
    the pixels). Returns (record, pinned Reinhard object)."""
    imports: Dict[str, Any] = {"prosper": True, "torch": torch.__version__}
    import torchvision
    import PIL
    imports["torchvision"] = torchvision.__version__
    imports["PIL"] = PIL.__version__
    try:
        import cv2                                   # noqa: F401 -- part of the import order
        imports["cv2"] = cv2.__version__
    except Exception as exc:                          # noqa: BLE001
        imports["cv2"] = f"unavailable: {type(exc).__name__}: {exc}"
    import openslide
    imports["openslide"] = openslide.__version__
    rh = make_pinned_reinhard(template, torchstain_prefix, reinhard_factory)
    golden = blur_golden_check()                     # recorded; callers act on it as before
    st = wsi_stat(Path(probe_wsi))
    slide = opener(str(probe_wsi))
    try:
        im = slide.read_region(PROBE_READ_LOCATION, 0, (512, 512)).convert("RGB")
        read = {"location": list(PROBE_READ_LOCATION), "level": 0, "size": [512, 512],
                "mode": im.mode, "returned_size": list(im.size)}
    finally:
        close = getattr(slide, "close", None)
        if callable(close):
            close()
    rec = {"import_order": ["prosper", "torch", "torchvision", "PIL", "cv2", "openslide",
                            "torchstain (pinned prefix)"],
           "imports": imports, "probe_wsi": st, "read_region": read,
           "pinned_reinhard_fit_asserted": True, "blur_golden": golden,
           "maps_read": "after these steps, before any slide is rendered or output published"}
    return rec, rh


def _default_probe_wsi(a: argparse.Namespace, rows: Optional[Sequence[Mapping[str, Any]]],
                       slide_path_of: Optional[Callable[[str], Path]]) -> Path:
    """``--probe-wsi`` if given, else the WSI of queries.csv row 0 (as the capture uses)."""
    if getattr(a, "probe_wsi", None) is not None:
        return Path(a.probe_wsi)
    if not rows or slide_path_of is None:
        raise SuiteError("--probe-wsi is required (no queries.csv row 0 to default to)")
    return Path(slide_path_of(rows[0]["slide_id"]))


def _prepare_render_env(a: argparse.Namespace, probe: Optional[EnvProbe],
                        reinhard_factory: Optional[Callable], *, opener: Optional[Callable],
                        probe_wsi: Path):
    """Environment assertion [A2]/[R4]; runs before any slide is rendered or any output is
    published by the calling sub-command."""
    set_single_thread()
    held: Dict[str, Any] = {}

    def steps() -> dict:
        rec, held["rh"] = reference_lib_probe(
            template=a.template, torchstain_prefix=a.torchstain_prefix,
            reinhard_factory=reinhard_factory, opener=opener or openslide_open,
            probe_wsi=probe_wsi)
        return rec

    env = environment_block(probe or EnvProbe(), torchstain_prefix=a.torchstain_prefix,
                            reference_libs_path=a.reference_libs,
                            allow_missing_reference_libs=a.allow_missing_reference_libs,
                            probe_steps=steps)
    rh = held["rh"]
    env["pinned_reinhard_fit_asserted"] = True
    return env, rh, dict(getattr(rh, "info", {}) or {})


def fingerprint_main(argv=None, *, opener: Optional[Callable] = None,
                     slide_path_of: Callable[[str], Path] = paths.slide_wsi_path,
                     reinhard_factory: Optional[Callable] = None,
                     probe: Optional[EnvProbe] = None) -> int:
    p = argparse.ArgumentParser(prog="fingerprint")
    p.add_argument("--out-root", type=Path, required=True)
    add_input_args(p)
    add_env_args(p)
    p.add_argument("--tasks", type=Path, default=None)
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--slides", default="")
    p.add_argument("--processes", type=int, default=3)
    p.add_argument("--loader-batch", type=int, default=8)
    a = p.parse_args(argv)
    inp = load_inputs(a)
    sel = _select_slides(a, inp)
    env, rh, rh_info = _prepare_render_env(
        a, probe, reinhard_factory, opener=opener,
        probe_wsi=_default_probe_wsi(a, inp.rows, slide_path_of))
    golden = blur_golden_check()
    if not golden["ok"]:
        raise SuiteError(f"blur golden mismatch {golden}")
    fp_dir = a.out_root / "fingerprints"
    _inside(a.out_root, fp_dir)
    counts = {"written": 0, "skipped": 0, "failed": 0}
    t0 = time.time()
    for sid in sel:
        st = fingerprint_slide(a.out_root, fp_dir, inp, sid, reinhard=rh, reinhard_info=rh_info,
                               env=env, opener=opener or openslide_open,
                               slide_path_of=slide_path_of, processes=a.processes,
                               loader_batch=a.loader_batch, golden=golden)
        counts[st] += 1
    summary = {"argv": list(argv) if argv is not None else sys.argv[1:], "slides": sel,
               "counts": counts, "wall_seconds": round(time.time() - t0, 2),
               "cpu_seconds": round(_cpu_seconds(), 2), "environment": env,
               "manifest_v2_sha256": inp.manifest.sha256, "queries_sha256": inp.queries_sha256}
    publish(a.out_root, fp_dir / "runs" / f"{_tag()}.json", _dump_pretty(summary))
    print(f"[suite] fingerprint done {counts}", flush=True)
    return 1 if counts["failed"] else 0


COMPLETE_SCHEMA = "r2_1_perturb_suite_fingerprint_complete_v2"


def _env_waived(doc: Mapping[str, Any]) -> bool:
    """[R5a] the pair was produced under ``--allow-missing-reference-libs`` (or without an
    asserted reference-library set)."""
    env = doc.get("environment") if isinstance(doc, Mapping) else None
    if not isinstance(env, Mapping):
        return True
    rl = (env.get("asserted") or {}).get("reference_libs") or {}
    return bool(env.get("allow_missing_reference_libs")) or rl.get("status") == "NOT_ASSERTED"


PUBLISH_CONFLICT_DIGEST_KEYS: Tuple[str, ...] = ("npy_sha256", "content_sha256")   # [R6d]
FINALIZE_REPORT_SCHEMA = "r2_1_perturb_suite_finalize_report_v1"                    # [C4-1]
FINALIZE_SUMMARY_WOULD_BE_KEYS: Tuple[str, ...] = ("sha256", "bytes", "n_slides", "n_tiles")


def _file_digest(path: Path) -> Dict[str, Any]:
    data = Path(path).read_bytes()
    return {"path": str(path), "sha256": _sha(data), "bytes": len(data)}


def publish_finalize_report(root: Path, report_dir: Path, report: Mapping[str, Any], *, rc: int,
                            published: Mapping[str, Any], meta: Mapping[str, Any],
                            schema: str = FINALIZE_REPORT_SCHEMA) -> Dict[str, Any]:
    """[C4-1] the complete finalize report (unchanged keys plus ``report_schema``, ``rc``, the
    published INDEX/COMPLETE digests and run metadata) published immutably (``publish``) to
    ``<report_dir>/<UTC>_<pid>_<ns>.json``; never overwritten. Any failure is a SuiteError
    (in the complete branch INDEX.json/COMPLETE are already published: OR-8 applies)."""
    extra = dict(report_schema=schema, rc=rc, written_at=_now(),
                 suite_code_sha256=SUITE_CODE_SHA256, **meta, **published)
    clash = sorted(set(extra) & set(report))
    if clash:
        raise SuiteError(f"finalize report key clash {clash}")
    ns = time.time_ns()
    dest = report_dir / (f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(ns // 10**9))}_"
                         f"{os.getpid()}_{ns % 10**9:09d}.json")
    try:
        data = _dump_pretty({**report, **extra})
        publish(root, dest, data)
    except Exception as exc:                              # noqa: BLE001
        state = ("INDEX.json and COMPLETE were already published (OR-8)" if report.get("complete")
                 else "finalize blocked; nothing else was written")
        raise SuiteError(f"finalize report write failed ({type(exc).__name__}: {exc}); "
                         f"{state}") from exc
    return {"path": str(dest), "sha256": _sha(data), "bytes": len(data)}


def finalize_summary(report: Mapping[str, Any], rc: int, report_file: Mapping[str, Any],
                     published: Mapping[str, Any]) -> Dict[str, Any]:
    """[C4-1] compact, never-truncated stdout summary: the complete
    ``blocking_failed_records`` list, counts, the report file digest and either the published
    ``index`` digest or the ``would_be_index`` digest/summary (null when no index could be
    built because population slides lack a validated pair; ``n_problems`` counts those)."""
    s: Dict[str, Any] = {
        "complete": bool(report["complete"]), "rc": rc, "report": dict(report_file),
        "n_slides": report["n_slides"], "n_problems": len(report["problems"]),
        "n_blocking": len(report["blocking_failed_records"]),
        "blocking_failed_records": report["blocking_failed_records"],
        "n_superseded_failures": len(report["superseded_failures"]),
        "n_reference_lib_waived": len(report["reference_lib_waived"])}
    if report["complete"]:
        s["index"] = dict(published["index_published"])
    else:
        wbi = report.get("would_be_index")
        s["would_be_index"] = None if wbi is None else {
            k: wbi[k] for k in FINALIZE_SUMMARY_WOULD_BE_KEYS}
    if "waiver_block" in report:
        s["waiver_block"] = report["waiver_block"]
    return s


FP_SUPERSEDING_KEYS: Tuple[str, ...] = ("content_sha256", "npy_sha256", "fp_json_sha256")


def classify_failed_records(base_dir: Path, failed_dir: Path, population: Any,
                            by_sid: Mapping[str, Mapping[str, Any]], *,
                            digest_keys: Sequence[str], superseding_keys: Sequence[str],
                            supersede_problem: Optional[
                                Callable[[Mapping[str, Any], Mapping[str, Any]], Optional[str]]
                            ] = None
                            ) -> Tuple[List[str], List[dict], List[dict]]:
    """[R3]/[R6d] FAILED-record classification shared by ``finalize`` (``fingerprints/failed``,
    digest keys ``npy_sha256``/``content_sha256``) and ``finalize-features`` [C4-2]
    (``features/failed/<E>``, digest key ``pickle_sha256``). ``failed_dir/*.json`` is read
    non-recursively; a record is BLOCKING when unreadable or not a JSON object, has a
    ``slide_id`` that is neither a string nor missing (repair 005, finding 4: blocking with a
    reason, never a traceback), is of a slide outside ``population``, of a slide without a
    validated pair (``by_sid``), when ``supersede_problem(record, pair)`` (``finalize-features``
    only; ``finalize`` passes None) returns a reason, or is a ``publish`` conflict whose refused
    digest is missing/empty or differs from the pair's; otherwise it is SUPERSEDED by the pair
    (listed, never moved/deleted). Returns (record paths relative to ``base_dir``, blocking,
    superseded)."""
    failed_files = sorted(failed_dir.glob("*.json")) if failed_dir.is_dir() else []
    blocking_failed, superseded = [], []
    for f in failed_files:
        rel = str(f.relative_to(base_dir))
        try:
            fd = json.loads(f.read_text())
            if not isinstance(fd, dict):
                raise ValueError("FAILED record is not a JSON object")
            sid = fd.get("slide_id")
        except Exception as exc:                          # noqa: BLE001
            blocking_failed.append({"record": rel, "reason": f"unreadable: {exc}"})
            continue
        if sid is not None and not isinstance(sid, str):
            blocking_failed.append({"record": rel, "slide_id": sid, "stage": fd.get("stage"),
                                    "reason": "slide_id is not a string"})
        elif sid not in population:
            blocking_failed.append({"record": rel, "slide_id": sid, "stage": fd.get("stage"),
                                    "reason": "slide_id not in the population"})
        elif sid not in by_sid:
            blocking_failed.append({"record": rel, "slide_id": sid, "stage": fd.get("stage"),
                                    "reason": "no validated pair for this slide"})
        else:
            e = by_sid[sid]
            why = supersede_problem(fd, e) if supersede_problem is not None else None
            if why is not None:
                blocking_failed.append({"record": rel, "slide_id": sid, "stage": fd.get("stage"),
                                        "reason": why})
                continue
            if fd.get("stage") == "publish":
                # [R6d] a publish conflict is superseded only when the refused duplicate's
                # digest equals the pair's; digest-less (older) records fail closed.
                refused = {k: fd.get(k) for k in digest_keys}
                pair = {k: e[k] for k in digest_keys}
                why = None
                if any(not isinstance(fd.get(k), str) or not fd.get(k) for k in digest_keys):
                    why = "publish conflict without digest"
                elif refused != pair:
                    why = "duplicate produced different content"
                if why is not None:
                    blocking_failed.append({"record": rel, "slide_id": sid, "stage": "publish",
                                            "reason": why, "refused_digest": refused,
                                            "pair_digest": pair})
                    continue
            superseded.append({"record": rel, "record_sha256": X.sha256_file(f),
                               "slide_id": sid, "stage": fd.get("stage"),
                               "failure_time": fd.get("time"),
                               "superseding_pair": {k: e[k] for k in superseding_keys}})
    return [str(f.relative_to(base_dir)) for f in failed_files], blocking_failed, superseded


def finalize_publish(root: Path, out_dir: Path, report: Dict[str, Any], index: Mapping[str, Any],
                     *, waived: Sequence[str], blocking: Sequence[dict],
                     accept_waivers: Optional[str],
                     complete_doc: Callable[[str], Mapping[str, Any]],
                     would_be: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> None:
    """[R5a]/[R6b] shared by ``finalize`` and ``finalize-features`` (called only when every
    population slide has a validated pair). ``out_dir/INDEX.json`` is published only in the
    step that publishes ``out_dir/COMPLETE`` (INDEX first, both ``publish_idempotent``: an
    existing file with different bytes raises); a blocked finalize writes no INDEX and puts the
    would-be index (digest, ``would_be(index)`` summary, content) into the report."""
    index_bytes = _dump_pretty(index)
    if waived and accept_waivers is None:
        report["waiver_block"] = (f"{len(waived)} pair(s) produced under a reference-library "
                                  "waiver; COMPLETE needs --accept-waivers <reason>")
    if not blocking and (not waived or accept_waivers is not None):
        report["index"] = publish_idempotent(root, out_dir / "INDEX.json", index_bytes)
        complete = complete_doc(X.sha256_file(out_dir / "INDEX.json"))
        report["complete_status"] = publish_idempotent(root, out_dir / "COMPLETE",
                                                       _dump_pretty(complete))
        report["complete"] = True
    else:
        report["index_not_written"] = "finalize blocked (waivers or blocking FAILED records)"
        report["would_be_index"] = {"file": "INDEX.json", "sha256": _sha(index_bytes),
                                    "bytes": len(index_bytes), "schema": index["schema"],
                                    **would_be(index)}
        report["would_be_index_content"] = index


def finalize_emit(root: Path, out_dir: Path, report: Mapping[str, Any],
                  meta: Mapping[str, Any]) -> int:
    """[C4-1] report file (``out_dir/finalize_reports/``) then the compact stdout summary;
    returns the exit code (0 complete / 1 blocked). In the complete branch the published
    INDEX/COMPLETE are re-read for the recorded digests; a read failure there is a SuiteError
    (OR-8: both are already published), as is any report-write failure."""
    rc = 0 if report["complete"] else 1
    published: Dict[str, Any] = {}
    if report["complete"]:
        try:
            published = {"index_published": _file_digest(out_dir / "INDEX.json"),
                         "complete_published": _file_digest(out_dir / "COMPLETE")}
        except OSError as exc:
            raise SuiteError(f"finalize report: cannot read the published INDEX.json/COMPLETE "
                             f"({type(exc).__name__}: {exc}); INDEX.json and COMPLETE were "
                             "already published (OR-8)") from exc
    rep_file = publish_finalize_report(root, out_dir / "finalize_reports", report,
                                       rc=rc, published=published, meta=meta)
    print(json.dumps(finalize_summary(report, rc, rep_file, published), sort_keys=True,
                     default=str), flush=True)
    return rc


def finalize_main(argv=None) -> int:
    """[R3] COMPLETE only when every population slide has a validated pair. FAILED records
    (``fingerprints/failed/*.json``) of slides WITHOUT a validated pair block and are reported;
    those of slides WITH a validated pair are listed (never deleted/moved) as
    ``superseded_failures`` in INDEX.json and COMPLETE. [R5a] pairs produced under a
    reference-library waiver are flagged per slide (``reference_lib_waived``) and block
    COMPLETE unless ``--accept-waivers <reason>`` is given (written into COMPLETE).
    [R6b] INDEX.json is published only in the step that publishes COMPLETE (INDEX first); a
    blocked finalize writes no INDEX and reports the would-be index. [R6d] a ``publish``
    FAILED record of a paired slide is superseded only if its refused digest equals the
    pair's; otherwise (or without a digest) it blocks.
    [C4-1] The complete report is published (never overwritten) to
    ``fingerprints/finalize_reports/<UTC>_<pid>_<ns>.json`` (complete branch: INDEX ->
    COMPLETE -> report, the report recording both published digests; blocked branch: report
    only) and stdout carries only :func:`finalize_summary` (candidate 003 printed the full
    report cut at 20,000 characters, OR-9; the summary is printed whole, never cut). INDEX.json
    and COMPLETE bytes are unchanged from candidate 003; exit codes 0 complete / 1 blocked.
    [C4-2] the FAILED classification and the INDEX/COMPLETE/report steps are shared with
    ``finalize-features`` (:func:`classify_failed_records`, :func:`finalize_publish`,
    :func:`finalize_emit`) with this command's behaviour and bytes unchanged."""
    p = argparse.ArgumentParser(prog="finalize")
    p.add_argument("--out-root", type=Path, required=True)
    add_input_args(p)
    p.add_argument("--accept-waivers", default=None, metavar="REASON",
                   help="[R5a] explicit reason accepting reference-library-waived pairs")
    a = p.parse_args(argv)
    if a.accept_waivers is not None and not a.accept_waivers.strip():
        raise SuiteError("--accept-waivers needs a non-empty reason")
    inp = load_inputs(a)
    fp_dir = a.out_root / "fingerprints"
    entries, problems = [], []
    by_sid: Dict[str, dict] = {}
    for r in inp.rows:
        sid = r["slide_id"]
        st, why, doc, _ = validate_fp_pair(fp_dir, sid, n_tiles=int(r["feature_rows"]),
                                           manifest_v2_sha256=inp.manifest.sha256)
        if st != "complete":
            problems.append({"slide_id": sid, "status": st, "reason": why})
            continue
        rec = doc["environment"]["recorded"]
        e = {"slide_id": sid, "n_tiles": doc["n_tiles"],
             "content_sha256": doc["array"]["content_sha256"],
             "npy_sha256": doc["array"]["npy_sha256"],
             "fp_json_sha256": X.sha256_file(fp_paths(fp_dir, sid)[1]),
             "aggregate_digests": doc["aggregate_digests"],
             "cpu_model": rec.get("cpu_model"), "host": rec.get("hostname"),
             "lsf_job": rec.get("lsf_job"), "lsf_index": rec.get("lsf_index"),
             "reinhard_nonfinite_tile_indices": doc["reinhard_nonfinite_tile_indices"],
             "reference_lib_waived": _env_waived(doc)}
        entries.append(e)
        by_sid[sid] = e
    population = {r["slide_id"] for r in inp.rows}
    failed_records, blocking_failed, superseded = classify_failed_records(
        fp_dir, fp_dir / "failed", population, by_sid,
        digest_keys=PUBLISH_CONFLICT_DIGEST_KEYS, superseding_keys=FP_SUPERSEDING_KEYS)
    waived = [e["slide_id"] for e in entries if e["reference_lib_waived"]]
    report = {"n_slides": len(inp.rows), "n_valid": len(entries), "problems": problems,
              "failed_records": failed_records,
              "blocking_failed_records": blocking_failed,
              "superseded_failures": superseded, "reference_lib_waived": waived,
              "index": None, "complete": False}
    if not problems:
        index = {"schema": "r2_1_perturb_suite_fingerprint_index_v2",
                 "manifest_v2_sha256": inp.manifest.sha256, "queries_sha256": inp.queries_sha256,
                 "conditions": list(SUITE_CONDITIONS), "n_slides": len(entries),
                 "n_tiles": sum(e["n_tiles"] for e in entries),
                 "cpu_models": sorted({str(e["cpu_model"]) for e in entries}),
                 "n_reference_lib_waived": len(waived),
                 "superseded_failures": superseded,
                 "slides": entries}
        finalize_publish(
            a.out_root, fp_dir, report, index, waived=waived, blocking=blocking_failed,
            accept_waivers=a.accept_waivers,
            complete_doc=lambda index_sha256: {
                "schema": COMPLETE_SCHEMA, "index_file": "INDEX.json",
                "index_sha256": index_sha256,
                "n_slides": len(entries), "superseded_failures": superseded,
                "reference_lib_waivers": {
                    "n_waived": len(waived), "slides": waived,
                    "accepted_reason": a.accept_waivers if waived else None}},
            would_be=lambda idx: {
                "n_slides": idx["n_slides"], "n_tiles": idx["n_tiles"],
                "cpu_models": idx["cpu_models"],
                "n_reference_lib_waived": idx["n_reference_lib_waived"],
                "n_superseded_failures": len(superseded)})
    meta = {"manifest_v2_sha256": inp.manifest.sha256, "queries_sha256": inp.queries_sha256,
            "accept_waivers": a.accept_waivers}
    return finalize_emit(a.out_root, fp_dir, report, meta)


# --------------------------------------------------------------------------
# [C4-2] features-tree finalize
# --------------------------------------------------------------------------

FEATURES_INDEX_SCHEMA = "r2_1_perturb_suite_features_index_v1"
FEATURES_COMPLETE_SCHEMA = "r2_1_perturb_suite_features_complete_v1"
FEATURES_PUBLISH_DIGEST_KEYS: Tuple[str, ...] = ("pickle_sha256",)                  # [R6d]
FEATURES_SUPERSEDING_KEYS: Tuple[str, ...] = ("pickle_sha256", "record_sha256")
# The exact status string the frozen 003 ``extract`` writes after all tiles x 13 pictures
# were verified (its ``fingerprint_verification`` block carries no ``ok`` key).
FINGERPRINT_VERIFICATION_STATUS = "all tiles x 13 pictures verified before transform"
FEATURES_INDEX_FIELD_SOURCES = {
    "pickle_sha256": "record.pickle_sha256 (== sha256 of the .pkl, verified)",
    "pickle_bytes": "record.pickle_bytes (== .pkl st_size, verified)",
    "record_sha256": "sha256 of the .record.json bytes read by finalize-features",
    "conditions_encoded": "record.conditions_encoded (duplicate-free subset of the 13 suite "
                          "conditions, superset of the plan's; order as recorded)",
    "n_tiles": "record.n_tiles (== queries feature_rows, verified)",
    "host": "record.environment.recorded.hostname (JSON null if absent)",
    "gpu": "record.gpu (null if absent)",
    "driver_tag": "'+'.join(record.environment.recorded.gpu_driver_tags); 'none' for an empty "
                  "list; null if absent/null",
    "lsf_job": "record.environment.recorded.lsf_job (null if absent)",
    "invocation_tag": "record.invocation_tag (null if absent; the frozen 003 extract writes "
                      "no invocation_tag)",
    "plan_sha256": "record.plan_sha256: the --plan-sha256 of the extract run that produced the "
                   "pair (null for a run without --plan)",
    "reference_lib_waived": "[R5a] record.environment (as finalize)",
    "hosts": "distinct per-slide host values; null (JSON) listed last when any is absent",
    "driver_tags": "distinct per-slide driver_tag values; null (JSON) listed last when any is "
                   "absent",
    "n_pairs_other_plan": "number of slides whose record.plan_sha256 != this plan's sha256 "
                          "(null included)",
    "fingerprints_complete_sha256": "sha256 of the fingerprints/COMPLETE bytes; COMPLETE must "
                                    "exist, parse as the fingerprint COMPLETE schema with "
                                    "index_file 'INDEX.json', and its index_sha256 must equal "
                                    "the sha256 of the actual fingerprints/INDEX.json bytes "
                                    "(else SuiteError, nothing written)",
    "fingerprints_index_sha256": "verified sha256 of the fingerprints/INDEX.json bytes, bound to "
                                 "fingerprints/COMPLETE.index_sha256; the INDEX must parse, "
                                 "re-serialise to the same bytes, match the schema, "
                                 "--manifest-v2 and n_slides (INDEX and COMPLETE), and each "
                                 "validated pair's fp_json/npy/content digests must equal its "
                                 "INDEX entry (any failure: SuiteError, nothing written)",
}
FIRST_RUN_SUPERSEDE_REASON = ("first_run_identity failure not superseded: the pair has no "
                              "passing first-run identity check")


def _distinct_sorted(values: Sequence[Any]) -> List[Any]:
    """Distinct values (JSON-canonical equality) sorted: strings, then other values by their
    canonical JSON, then null (repair 005: a missing value is JSON null, not "None")."""
    seen: Dict[str, Any] = {}
    for v in values:
        seen.setdefault(json.dumps(v, sort_keys=True, default=str), v)
    return [seen[k] for k in sorted(seen, key=lambda k: (
        seen[k] is None, not isinstance(seen[k], str),
        seen[k] if isinstance(seen[k], str) else k))]


def fingerprint_verification_problem(fv: Any, n_tiles: int) -> Optional[str]:
    """[C4-2] None when the record's ``fingerprint_verification`` block states that every tile
    x 13 pictures was verified: exact status string, ``mismatches == 0`` and the verified
    tile/picture counts equal ``n_tiles`` / ``13 * n_tiles`` (ints, not bools); an ``ok`` key,
    if present, must be ``True``. Otherwise the reason."""
    if not isinstance(fv, Mapping):
        return "fingerprint_verification missing or not an object"
    if "ok" in fv and fv["ok"] is not True:
        return "fingerprint_verification.ok is not True"
    if fv.get("status") != FINGERPRINT_VERIFICATION_STATUS:
        return "fingerprint_verification status mismatch"
    for key, want in (("mismatches", 0), ("n_tiles_verified", n_tiles),
                      ("n_pictures_verified", n_tiles * len(SUITE_CONDITIONS))):
        v = fv.get(key)
        if type(v) is not int or v != want:
            return f"fingerprint_verification {key} {v!r} != {want}"
    return None


def fingerprint_tree_problem(fp_dir: Path, sid: str, fv: Mapping[str, Any]) -> Optional[str]:
    """[C4-2] repair 005 (amendment 3): None when the record's ``fingerprint_verification``
    digests equal the fingerprint tree: ``fp_json_sha256 == sha256(fingerprints/<sid>.fp.json)``
    and ``fp_npy_sha256`` / ``fp_content_sha256`` equal that file's ``array.npy_sha256`` /
    ``array.content_sha256`` (the values the frozen ``extract`` copies into the record).
    Otherwise the reason."""
    js = fp_paths(fp_dir, sid)[1]
    try:
        raw = js.read_bytes()
        doc = json.loads(raw)
        arr = doc.get("array") if isinstance(doc, dict) else None
    except Exception as exc:                              # noqa: BLE001
        return f"fingerprint tree {js.name} unreadable: {type(exc).__name__}: {exc}"
    if fv.get("fp_json_sha256") != _sha(raw):
        return f"record fp_json_sha256 != sha256 of fingerprint tree {js.name}"
    if not isinstance(arr, Mapping):
        return f"fingerprint tree {js.name} has no array object"
    for key, fkey in (("fp_npy_sha256", "npy_sha256"), ("fp_content_sha256", "content_sha256")):
        if fv.get(key) != arr.get(fkey):
            return f"record {key} != fingerprint tree {js.name} array.{fkey}"
    return None


def first_run_check_passed(r: Mapping[str, Any]) -> bool:
    """[C4-2] repair 005: the record carries a passing first-run identity check, as the frozen
    ``extract --plan`` writes it for a plan-"present" slide (``first_run.status == "present"``,
    ``identity_equal is True`` and a non-empty ``record_sha256``)."""
    fr = r.get("first_run")
    return (isinstance(fr, Mapping) and fr.get("status") == "present"
            and fr.get("identity_equal") is True
            and isinstance(fr.get("record_sha256"), str) and bool(fr.get("record_sha256")))


def first_run_problem(r: Mapping[str, Any], plan_entry: Mapping[str, Any]) -> Optional[str]:
    """[C4-2] repair 005 (finding 1): for a plan-"present" slide the record must carry a passing
    first-run identity check against the SAME first-run record as the plan entry:
    ``first_run.status == "present"``, ``identity_equal is True``, ``first_run.record_sha256 ==
    plan_entry.record_sha256``, and the class / first-run pickle digest the frozen ``extract``
    copies from the plan entry / that record equal the entry's ``first_run_class`` /
    ``pickle_sha256``. None for a plan-"absent" slide. Otherwise the reason."""
    if plan_entry.get("first_run_record") != "present":
        return None
    fr = r.get("first_run")
    status = fr.get("status") if isinstance(fr, Mapping) else None
    if status != "present":
        return (f"plan entry first_run_record is 'present' but the record has no first-run "
                f"identity check (first_run.status {status!r})")
    if fr.get("identity_equal") is not True:
        return "record first_run.identity_equal is not True"
    if fr.get("record_sha256") != plan_entry.get("record_sha256"):
        return "record first_run.record_sha256 != plan entry record_sha256"
    if fr.get("class") != plan_entry.get("first_run_class"):
        return "record first_run.class != plan entry first_run_class"
    if fr.get("pickle_sha256") != plan_entry.get("pickle_sha256"):
        return "record first_run.pickle_sha256 != plan entry pickle_sha256"
    return None


def validate_feature_pair(feat_dir: Path, sid: str, *, encoder: str, manifest_v2_sha256: str,
                          n_tiles: int, conditions: Sequence[str],
                          plan_entry: Mapping[str, Any], fp_dir: Path
                          ) -> Tuple[str, str, Optional[dict], Optional[str]]:
    """[C4-2] ('absent'|'complete'|'invalid', reason, record, record_sha256) of
    ``feat_dir/<sid>.pkl`` + ``<sid>.record.json``: record ``schema == RECORD_SCHEMA``,
    ``slide_id``, ``encoder``, ``manifest_v2_sha256``, ``extractor_code_sha256 ==
    EXTRACTOR_SHA256``, ``n_tiles``, ``conditions_encoded`` a duplicate-free subset of the 13
    suite conditions (order not checked) and a superset of ``conditions``, ``pickle_bytes ==
    st_size``, :func:`fingerprint_verification_problem` None, the record's fingerprint digests
    equal to the fingerprint tree (:func:`fingerprint_tree_problem`), the plan-"present"
    first-run identity check (:func:`first_run_problem`) and ``pickle_sha256 ==
    sha256(pkl)``."""
    pkl, recp = feat_dir / f"{sid}.pkl", feat_dir / f"{sid}.record.json"
    has_p, has_r = pkl.exists(), recp.exists()
    if not has_p and not has_r:
        return "absent", "", None, None
    if not (has_p and has_r):
        return "invalid", f"orphan output (pickle={has_p}, record={has_r})", None, None
    try:
        raw = recp.read_bytes()
        r = json.loads(raw)
        if not isinstance(r, dict):
            raise ValueError("record is not a JSON object")
        pkl_size = pkl.stat().st_size
    except Exception as exc:                              # noqa: BLE001
        return "invalid", f"unreadable record: {exc}", None, None
    conds = r.get("conditions_encoded")
    checks = [(r.get("schema") == RECORD_SCHEMA, "schema"),
              (r.get("slide_id") == sid, "slide_id"),
              (r.get("encoder") == encoder, "encoder"),
              (r.get("manifest_v2_sha256") == manifest_v2_sha256, "manifest_v2_sha256"),
              (r.get("extractor_code_sha256") == EXTRACTOR_SHA256, "extractor_code_sha256"),
              (type(r.get("n_tiles")) is int and r.get("n_tiles") == n_tiles, "n_tiles"),
              (type(r.get("pickle_bytes")) is int and r.get("pickle_bytes") == pkl_size,
               "pickle_bytes"),
              (isinstance(r.get("pickle_sha256"), str) and bool(r.get("pickle_sha256")),
               "pickle_sha256")]
    for ok, what in checks:
        if not ok:
            return "invalid", f"record {what} mismatch", r, None
    if not (isinstance(conds, list) and all(isinstance(c, str) for c in conds)):
        return "invalid", "record conditions_encoded is not a list of strings", r, None
    dups = sorted({c for c in conds if conds.count(c) > 1})
    if dups:
        return "invalid", f"record conditions_encoded has duplicates {dups}", r, None
    unknown = [c for c in conds if c not in SUITE_CONDITIONS]
    if unknown:
        return "invalid", f"record conditions_encoded has unknown conditions {unknown}", r, None
    lacking = [c for c in conditions if c not in conds]
    if lacking:
        return "invalid", f"record conditions_encoded lacks planned conditions {lacking}", r, None
    fv = r.get("fingerprint_verification")
    why = fingerprint_verification_problem(fv, n_tiles)
    if why is None:
        why = fingerprint_tree_problem(fp_dir, sid, fv)
    if why is None:
        why = first_run_problem(r, plan_entry)
    if why is not None:
        return "invalid", why, r, None
    try:
        digest = X.sha256_file(pkl)
    except OSError as exc:
        return "invalid", f"unreadable pickle: {exc}", r, None
    if digest != r["pickle_sha256"]:
        return "invalid", "pickle sha256 != record", r, None
    return "complete", "", r, _sha(raw)


def load_extension_plan(path: Path, expected_sha256: str, encoder: str, inp: Inputs
                        ) -> Tuple[dict, str, List[Tuple[str, List[str], Mapping[str, Any]]]]:
    """[C4-2] the extension plan checked as ``extract --plan`` does (raw-byte sha256 ==
    non-empty ``--plan-sha256``; schema, encoder, queries sha and manifest v1 sha), plus: a
    ``slides`` list of unique query slides, each with valid ``conditions`` (SUITE order) and
    ``first_run_record`` "present" (then with a non-empty ``record_sha256``, as
    ``plan-extension`` writes it) or "absent". Returns (plan, sha256,
    [(slide_id, conditions, plan entry)] in plan order)."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise SuiteError(f"cannot read plan {path}: {exc}") from exc
    plan_sha = _sha(raw)
    if not expected_sha256 or plan_sha != expected_sha256:
        raise SuiteError(f"plan sha256 {plan_sha} != --plan-sha256 {expected_sha256!r}")
    try:
        plan = json.loads(raw)
    except ValueError as exc:
        raise SuiteError(f"plan is not JSON: {exc}") from exc
    if (not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA
            or plan.get("encoder") != encoder
            or plan.get("queries_sha256") != inp.queries_sha256
            or plan.get("manifest_v1_sha256") != inp.manifest.v1_sha256):
        raise SuiteError("plan schema/encoder/queries sha/manifest v1 sha mismatch")
    slides = plan.get("slides")
    if not isinstance(slides, list) or not slides:
        raise SuiteError("plan has no slides list")
    population: List[Tuple[str, List[str], Mapping[str, Any]]] = []
    seen = set()
    for s in slides:
        sid = s.get("slide_id") if isinstance(s, dict) else None
        if not isinstance(sid, str) or sid in seen:
            raise SuiteError(f"plan slide entry without a unique slide_id: {str(s)[:200]}")
        if sid not in inp.row_by_slide:
            raise SuiteError(f"plan slide {sid} is not in the queries")
        conds = s.get("conditions")
        if not isinstance(conds, list):
            raise SuiteError(f"plan slide {sid} has no conditions list")
        frr = s.get("first_run_record")
        if frr not in ("present", "absent"):
            raise SuiteError(f"plan slide {sid} first_run_record {frr!r} is not "
                             "'present'/'absent'")
        if frr == "present" and not (isinstance(s.get("record_sha256"), str)
                                     and s.get("record_sha256")):
            raise SuiteError(f"plan slide {sid} is first-run 'present' without record_sha256")
        seen.add(sid)
        population.append((sid, _ordered(conds), s))
    return plan, plan_sha, population


def _recorded_driver_tag(recorded: Mapping[str, Any]) -> Optional[str]:
    tags = recorded.get("gpu_driver_tags")
    if tags is None:
        return None
    if isinstance(tags, (list, tuple)):
        return "+".join(str(t) for t in tags) if tags else "none"
    return str(tags)


def verified_fingerprint_tree(fp_dir: Path, manifest_v2_sha256: str
                              ) -> Tuple[str, str, Dict[str, Mapping[str, Any]]]:
    """[C4-2] repair 006 (review 005 F1): ``fingerprints/COMPLETE`` is parsed and bound to
    ``fingerprints/INDEX.json``, not merely stat'ed. COMPLETE must be a ``COMPLETE_SCHEMA``
    object naming ``INDEX.json`` whose ``index_sha256`` equals the sha256 of the actual INDEX
    bytes; the INDEX must parse to a fingerprint-index object (schema as written by
    ``finalize``) for this manifest whose re-serialisation (``_dump_pretty``, exactly as
    ``finalize`` publishes it) reproduces those bytes, with ``n_slides`` (INDEX and COMPLETE)
    equal to the number of entries and unique string ``slide_id`` entries. Any failure is a
    SuiteError, raised before anything is written. Returns (COMPLETE sha256, verified INDEX
    sha256, INDEX entries by slide_id)."""
    comp_path, idx_path = fp_dir / "COMPLETE", fp_dir / "INDEX.json"
    if not comp_path.is_file():
        raise SuiteError(f"{comp_path} does not exist: finalize the fingerprint tree first "
                         "(finalize-features requires fingerprints/COMPLETE)")

    def bad(what: str) -> SuiteError:
        return SuiteError(f"fingerprints/COMPLETE not verified ({comp_path}): {what}; "
                          "nothing written")
    try:
        comp_raw = comp_path.read_bytes()
    except OSError as exc:
        raise SuiteError(f"cannot read {comp_path}: {exc}") from exc
    try:
        comp = json.loads(comp_raw)
    except Exception as exc:                              # noqa: BLE001
        raise bad(f"COMPLETE does not parse as JSON ({type(exc).__name__}: {exc})") from exc
    if not isinstance(comp, dict) or comp.get("schema") != COMPLETE_SCHEMA:
        raise bad(f"COMPLETE is not a {COMPLETE_SCHEMA} object")
    if comp.get("index_file") != "INDEX.json":
        raise bad("COMPLETE index_file is not 'INDEX.json'")
    want = comp.get("index_sha256")
    if not isinstance(want, str) or not want:
        raise bad("COMPLETE has no index_sha256")
    try:
        idx_raw = idx_path.read_bytes()
    except OSError as exc:
        raise bad(f"cannot read fingerprints/INDEX.json ({type(exc).__name__}: {exc})") from exc
    idx_sha = _sha(idx_raw)
    if idx_sha != want:
        raise bad(f"COMPLETE index_sha256 {want} != sha256 of fingerprints/INDEX.json "
                  f"{idx_sha}")
    try:
        idx = json.loads(idx_raw)
        reserialised = _dump_pretty(idx)
    except Exception as exc:                              # noqa: BLE001
        raise bad(f"fingerprints/INDEX.json does not parse ({type(exc).__name__}: "
                  f"{exc})") from exc
    if reserialised != idx_raw:
        raise bad("fingerprints/INDEX.json does not re-serialise to its own bytes")
    if not isinstance(idx, dict) or idx.get("schema") != "r2_1_perturb_suite_fingerprint_index_v2":
        raise bad("fingerprints/INDEX.json is not a r2_1_perturb_suite_fingerprint_index_v2 "
                  "object")
    if idx.get("manifest_v2_sha256") != manifest_v2_sha256:
        raise bad("fingerprints/INDEX.json manifest_v2_sha256 != --manifest-v2")
    slides = idx.get("slides")
    if (not isinstance(slides, list) or type(idx.get("n_slides")) is not int
            or idx["n_slides"] != len(slides) or type(comp.get("n_slides")) is not int
            or comp["n_slides"] != len(slides)):
        raise bad("fingerprints INDEX/COMPLETE n_slides inconsistent with the INDEX slides")
    by_sid: Dict[str, Mapping[str, Any]] = {}
    for e in slides:
        sid = e.get("slide_id") if isinstance(e, dict) else None
        if not isinstance(sid, str) or sid in by_sid:
            raise bad("fingerprints/INDEX.json has a malformed or duplicate slide entry")
        by_sid[sid] = e
    return _sha(comp_raw), idx_sha, by_sid


def fingerprint_index_problem(entry: Optional[Mapping[str, Any]], fv: Mapping[str, Any]
                              ) -> Optional[str]:
    """[C4-2] repair 006 (review 005 F1): None when the record's ``fingerprint_verification``
    digests equal the verified fingerprint INDEX entry of that slide (``fp_json_sha256``,
    ``fp_npy_sha256`` -> ``npy_sha256``, ``fp_content_sha256`` -> ``content_sha256``);
    otherwise the reason."""
    if entry is None:
        return "slide has no entry in the verified fingerprints/INDEX.json"
    for key, ikey in (("fp_json_sha256", "fp_json_sha256"), ("fp_npy_sha256", "npy_sha256"),
                      ("fp_content_sha256", "content_sha256")):
        if fv.get(key) != entry.get(ikey):
            return f"record {key} != fingerprints/INDEX.json entry {ikey}"
    return None


def finalize_features_main(argv=None) -> int:
    """[C4-2] features-tree finalize for one encoder ``E``. Population = the extension plan's
    slides (plan checked as ``extract --plan``); a validated pair is
    ``features/E/<sid>.pkl`` + ``<sid>.record.json`` (:func:`validate_feature_pair`, with the
    plan's conditions for that slide). FAILED records ``features/failed/E/*.json``
    (non-recursive; where ``extract`` and ``quarantine-orphans --tree features`` write them)
    follow ``finalize``'s exact [R3]/[R5a]/[R6b]/[R6d] semantics, the [R6d] digest key being
    ``pickle_sha256``. Publishes ``features/E/INDEX.json`` (``FEATURES_INDEX_SCHEMA``) then
    ``features/E/COMPLETE`` (``FEATURES_COMPLETE_SCHEMA``) only together; the report goes to
    ``features/E/finalize_reports/`` and stdout carries the same compact summary as
    ``finalize``; exit codes 0 complete / 1 blocked. A rerun after COMPLETE is idempotent; a
    later FAILED record changes the index bytes and raises (``publish_idempotent``).
    Repair 005: ``fingerprints/COMPLETE`` must exist (SuiteError otherwise, before anything is
    written); a plan-"present" slide's pair must carry the first-run identity check against
    the plan entry's first-run record; a ``first_run_identity`` FAILED record is superseded
    only by a pair carrying a passing first-run identity check; INDEX records per-slide
    ``plan_sha256`` and ``n_pairs_other_plan``. Repair 006 (review 005 F1):
    ``fingerprints/COMPLETE`` is parsed and bound to ``fingerprints/INDEX.json``
    (:func:`verified_fingerprint_tree`) and every validated pair's fingerprint digests must
    equal its slide's INDEX entry (:func:`fingerprint_index_problem`); any failure is a
    SuiteError before anything is written. INDEX/COMPLETE/report record the verified
    fingerprint COMPLETE and INDEX sha256."""
    p = argparse.ArgumentParser(prog="finalize-features")
    p.add_argument("--out-root", type=Path, required=True)
    add_input_args(p)
    p.add_argument("--encoder", required=True, choices=sorted(X.AMP_DEFAULTS))
    p.add_argument("--plan", type=Path, required=True, help="extension/PLAN.json")
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--accept-waivers", default=None, metavar="REASON",
                   help="[R5a] explicit reason accepting reference-library-waived pairs")
    a = p.parse_args(argv)
    if a.accept_waivers is not None and not a.accept_waivers.strip():
        raise SuiteError("--accept-waivers needs a non-empty reason")
    inp = load_inputs(a)
    _, plan_sha, population = load_extension_plan(a.plan, a.plan_sha256, a.encoder, inp)
    fp_dir = a.out_root / "fingerprints"
    fp_complete_sha, fp_index_sha, fp_index = verified_fingerprint_tree(
        fp_dir, inp.manifest.sha256)
    feat_root = a.out_root / "features"
    feat_dir = feat_root / a.encoder
    _inside(a.out_root, feat_dir)
    entries, problems = [], []
    by_sid: Dict[str, dict] = {}
    first_run_ok: Dict[str, bool] = {}
    fp_index_mismatch: List[str] = []
    for sid, conds, plan_entry in population:
        st, why, rec, rec_sha = validate_feature_pair(
            feat_dir, sid, encoder=a.encoder, manifest_v2_sha256=inp.manifest.sha256,
            n_tiles=int(inp.row_by_slide[sid]["feature_rows"]), conditions=conds,
            plan_entry=plan_entry, fp_dir=fp_dir)
        if st != "complete":
            problems.append({"slide_id": sid, "status": st, "reason": why})
            continue
        why = fingerprint_index_problem(fp_index.get(sid), rec["fingerprint_verification"])
        if why is not None:
            fp_index_mismatch.append(f"{sid}: {why}")
            continue
        env = rec.get("environment")
        recorded = env.get("recorded") if isinstance(env, Mapping) else None
        recorded = recorded if isinstance(recorded, Mapping) else {}
        e = {"slide_id": sid, "pickle_sha256": rec["pickle_sha256"],
             "pickle_bytes": rec["pickle_bytes"], "record_sha256": rec_sha,
             "conditions_encoded": rec["conditions_encoded"], "n_tiles": rec["n_tiles"],
             "host": recorded.get("hostname"), "gpu": rec.get("gpu"),
             "driver_tag": _recorded_driver_tag(recorded), "lsf_job": recorded.get("lsf_job"),
             "invocation_tag": rec.get("invocation_tag"), "plan_sha256": rec.get("plan_sha256"),
             "reference_lib_waived": _env_waived(rec)}
        entries.append(e)
        by_sid[sid] = e
        first_run_ok[sid] = first_run_check_passed(rec)
    if fp_index_mismatch:
        raise SuiteError("fingerprints/INDEX.json (certified by fingerprints/COMPLETE) does not "
                         "match the feature records: " + "; ".join(fp_index_mismatch)
                         + "; nothing written")

    def supersede_problem(fd: Mapping[str, Any], e: Mapping[str, Any]) -> Optional[str]:
        if fd.get("stage") == "first_run_identity" and not first_run_ok[e["slide_id"]]:
            return FIRST_RUN_SUPERSEDE_REASON
        return None
    failed_records, blocking_failed, superseded = classify_failed_records(
        feat_root, feat_root / "failed" / a.encoder, {sid for sid, _, _ in population}, by_sid,
        digest_keys=FEATURES_PUBLISH_DIGEST_KEYS, superseding_keys=FEATURES_SUPERSEDING_KEYS,
        supersede_problem=supersede_problem)
    waived = [e["slide_id"] for e in entries if e["reference_lib_waived"]]
    n_other_plan = sum(e["plan_sha256"] != plan_sha for e in entries)
    report = {"n_slides": len(population), "n_valid": len(entries), "problems": problems,
              "failed_records": failed_records,
              "blocking_failed_records": blocking_failed,
              "superseded_failures": superseded, "reference_lib_waived": waived,
              "index": None, "complete": False}
    if not problems:
        coverage = {c: {"planned": sum(c in conds for _, conds, _ in population),
                        "encoded": sum(c in e["conditions_encoded"] for e in entries)}
                    for c in SUITE_CONDITIONS}
        index = {"schema": FEATURES_INDEX_SCHEMA, "encoder": a.encoder,
                 "plan_sha256": plan_sha, "manifest_v2_sha256": inp.manifest.sha256,
                 "queries_sha256": inp.queries_sha256,
                 "extractor_code_sha256": EXTRACTOR_SHA256,
                 "fingerprints_complete_sha256": fp_complete_sha,
                 "fingerprints_index_sha256": fp_index_sha,
                 "conditions": list(SUITE_CONDITIONS), "n_slides": len(entries),
                 "n_tiles": sum(e["n_tiles"] for e in entries),
                 "condition_coverage": coverage,
                 "hosts": _distinct_sorted([e["host"] for e in entries]),
                 "driver_tags": _distinct_sorted([e["driver_tag"] for e in entries]),
                 "n_pairs_other_plan": n_other_plan,
                 "n_reference_lib_waived": len(waived), "reference_lib_waived": waived,
                 "superseded_failures": superseded,
                 "field_sources": FEATURES_INDEX_FIELD_SOURCES,
                 "slides": entries}
        finalize_publish(
            a.out_root, feat_dir, report, index, waived=waived, blocking=blocking_failed,
            accept_waivers=a.accept_waivers,
            complete_doc=lambda index_sha256: {
                "schema": FEATURES_COMPLETE_SCHEMA, "encoder": a.encoder,
                "plan_sha256": plan_sha,
                "fingerprints_complete_sha256": fp_complete_sha,
                "fingerprints_index_sha256": fp_index_sha, "index_file": "INDEX.json",
                "index_sha256": index_sha256,
                "n_slides": len(entries), "superseded_failures": superseded,
                "reference_lib_waivers": {
                    "n_waived": len(waived), "slides": waived,
                    "accepted_reason": a.accept_waivers if waived else None}},
            would_be=lambda idx: {
                "n_slides": idx["n_slides"], "n_tiles": idx["n_tiles"],
                "hosts": idx["hosts"], "driver_tags": idx["driver_tags"],
                "n_reference_lib_waived": idx["n_reference_lib_waived"],
                "n_pairs_other_plan": idx["n_pairs_other_plan"],
                "n_superseded_failures": len(superseded)})
    meta = {"encoder": a.encoder, "plan_path": str(a.plan), "plan_sha256": plan_sha,
            "fingerprints_complete_sha256": fp_complete_sha,
            "fingerprints_index_sha256": fp_index_sha,
            "manifest_v2_sha256": inp.manifest.sha256, "queries_sha256": inp.queries_sha256,
            "accept_waivers": a.accept_waivers}
    return finalize_emit(a.out_root, feat_dir, report, meta)


QUARANTINE_PAIRS = {"fingerprints": (".fp.npy", ".fp.json"),
                    "features": (".pkl", ".record.json")}


def quarantine_orphans_main(argv=None) -> int:
    """[R3] explicit, recorded, move-only orphan repair. A lone ``<sid>.fp.npy`` /
    ``<sid>.fp.json`` (``--tree fingerprints``) or lone ``features/<enc>/<sid>.pkl`` /
    ``<sid>.record.json`` (``--tree features``) is MOVED (``os.replace`` within --out-root;
    never deleted, never overwritten) to ``<tree>/failed/quarantine/<UTC>/<relative path>``
    and a FAILED record (stage ``orphan_quarantine``: original path, size, sha256, new path)
    is written. Run only when no job writing that tree is active (an in-flight publish is
    indistinguishable from an orphan)."""
    p = argparse.ArgumentParser(prog="quarantine-orphans")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--tree", choices=sorted(QUARANTINE_PAIRS), default="fingerprints")
    p.add_argument("--encoder", default=None,
                   help="features tree: restrict to this encoder directory (default: all)")
    a = p.parse_args(argv)
    tree = a.out_root / a.tree
    if not tree.is_dir():
        raise SuiteError(f"{tree} does not exist")
    first, second = QUARANTINE_PAIRS[a.tree]
    if a.tree == "fingerprints":
        dirs = [(tree, tree / "failed")]
    else:
        encs = [a.encoder] if a.encoder else sorted(
            d.name for d in tree.iterdir() if d.is_dir() and d.name not in ("failed", "runs"))
        dirs = [(tree / e, tree / "failed" / e) for e in encs if (tree / e).is_dir()]
    lone: List[Tuple[Path, Path, str]] = []
    for d, failed_dir in dirs:
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.name.startswith("."):
                continue
            for suf, other in ((first, second), (second, first)):
                if f.name.endswith(suf):
                    sid = f.name[:-len(suf)]
                    if sid and not (d / f"{sid}{other}").exists():
                        lone.append((f, failed_dir, sid))
    report: Dict[str, Any] = {"tree": a.tree, "n_orphans": len(lone), "quarantined": []}
    if lone:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        qroot = tree / "failed" / "quarantine" / stamp
        _inside(a.out_root, qroot)
        qroot.mkdir(parents=True, exist_ok=False)          # never reuse a quarantine dir
        for f, failed_dir, sid in lone:
            rel = f.relative_to(tree)
            dest = qroot / rel
            _inside(a.out_root, dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() or os.path.lexists(dest):
                raise SuiteError(f"{dest} exists; refusing to overwrite")
            size, sha = f.stat().st_size, X.sha256_file(f)
            os.replace(f, dest)
            if dest.stat().st_size != size or X.sha256_file(dest) != sha:
                raise SuiteError(f"{dest}: moved file size/sha256 changed")
            rec = write_failed(a.out_root, failed_dir, sid, {
                "stage": "orphan_quarantine", "tree": a.tree,
                "reason": f"lone {f.name} (partner absent)",
                "original_path": str(f), "size": size, "sha256": sha, "new_path": str(dest)})
            report["quarantined"].append({"slide_id": sid, "original_path": str(f),
                                          "new_path": str(dest), "size": size, "sha256": sha,
                                          "failed_record": str(rec)})
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


# --------------------------------------------------------------------------
# Reference PNG set (item 4)
# --------------------------------------------------------------------------


def _pilot_slides(a: argparse.Namespace) -> List[str]:
    if getattr(a, "pilot", None) is not None:
        if a.slides:
            raise SuiteError("--pilot and --slides are exclusive")
        doc = json.loads(Path(a.pilot).read_text())
        return [s["slide_id"] for s in doc["slides"]]
    sel = _split_list(a.slides)
    if not sel:
        raise SuiteError("give --pilot PILOT_SLIDES.json or --slides")
    return sel


def _png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _decode_png_rgb(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        if im.mode != "RGB":
            raise SuiteError(f"PNG decodes to mode {im.mode}")
        return im.copy()


def reference_png_slide(root: Path, inp: Inputs, sid: str, *, reinhard: Any, opener: Callable,
                        slide_path_of: Callable[[str], Path], env: dict) -> str:
    fp_dir = root / "fingerprints"
    row = inp.row_by_slide[sid]
    st, why, _, arr = validate_fp_pair(fp_dir, sid, n_tiles=int(row["feature_rows"]),
                                       manifest_v2_sha256=inp.manifest.sha256)
    if st != "complete":
        raise SuiteError(f"{sid}: no validated fingerprint pair ({st}: {why})")
    out_dir = root / "reference_png" / sid
    index_path = out_dir / "INDEX.json"
    if index_path.exists():
        doc = json.loads(index_path.read_text())
        for e in doc["pictures"]:
            data = (out_dir / e["png_file"]).read_bytes()
            if _sha(data) != e["png_sha256"] or _sha(_decode_png_rgb(data).tobytes()) != e["raw_rgb_sha256"]:
                raise SuiteError(f"{sid}: existing reference PNG {e['png_file']} invalid")
        return "skipped"
    stimuli = inp.manifest.stimuli_v1(sid)
    check_template_digest(stimuli, sid)
    coords = X.load_bank_coords(row)
    slide_path = Path(slide_path_of(sid))
    stat0 = wsi_stat(slide_path)
    slide = opener(str(slide_path))
    shift = X.shift_of(stimuli)
    pictures_out: List[Tuple[Path, bytes, dict]] = []
    for i in range(min(REFERENCE_PNG_TILES, len(coords))):
        loc = (int(coords[i]["wsi_loc"][0]), int(coords[i]["wsi_loc"][1]))
        pics, _ = render_pictures(slide, loc, shift, stimuli, reinhard)
        dig = picture_digests(pics)
        for j, c in enumerate(SUITE_CONDITIONS):
            raw_sha = dig[j].tobytes().hex()
            if not np.array_equal(dig[j], arr[i, j]):
                raise SuiteError(f"{sid} tile {i} {c}: rendered digest != stored fingerprint")
            data = _png_bytes(pics[c])
            if _sha(_decode_png_rgb(data).tobytes()) != raw_sha:
                raise SuiteError(f"{sid} tile {i} {c}: PNG round trip changed the pixels")
            name = f"{i}_{c}.png"
            pictures_out.append((out_dir / name, data, {
                "tile_index": i, "condition": c, "wsi_loc": list(loc), "png_file": name,
                "raw_rgb_sha256": raw_sha, "png_sha256": _sha(data), "png_bytes": len(data)}))
    if hasattr(slide, "close"):
        slide.close()
    if wsi_stat(slide_path) != stat0:
        raise SuiteError(f"{sid}: WSI changed during reference rendering")
    for dest, data, _ in pictures_out:
        publish(root, dest, data)
    doc = {"schema": "r2_1_perturb_suite_reference_png_v1", "slide_id": sid,
           "fingerprint_npy_sha256": X.sha256_file(fp_paths(fp_dir, sid)[0]),
           "decode_check": "PIL PNG decode -> RGB tobytes SHA-256 == raw_rgb_sha256",
           "pictures": [m for _, _, m in pictures_out], "wsi": stat0, "environment": env,
           "suite_code_sha256": SUITE_CODE_SHA256, "extractor_code_sha256": EXTRACTOR_SHA256,
           "written_at": _now()}
    publish(root, index_path, _dump_pretty(doc))
    return "written"


def reference_png_main(argv=None, *, opener: Optional[Callable] = None,
                       slide_path_of: Callable[[str], Path] = paths.slide_wsi_path,
                       reinhard_factory: Optional[Callable] = None,
                       probe: Optional[EnvProbe] = None) -> int:
    p = argparse.ArgumentParser(prog="reference-png")
    p.add_argument("--out-root", type=Path, required=True)
    add_input_args(p)
    add_env_args(p)
    p.add_argument("--pilot", type=Path, default=None)
    p.add_argument("--slides", default="")
    a = p.parse_args(argv)
    inp = load_inputs(a)
    sel = _pilot_slides(a)
    env, rh, _ = _prepare_render_env(a, probe, reinhard_factory, opener=opener,
                                     probe_wsi=_default_probe_wsi(a, inp.rows, slide_path_of))
    res = {sid: reference_png_slide(a.out_root, inp, sid, reinhard=rh,
                                    opener=opener or openslide_open,
                                    slide_path_of=slide_path_of, env=env) for sid in sel}
    print(json.dumps(res, sort_keys=True), flush=True)
    return 0


# --------------------------------------------------------------------------
# First-run records: fit classification, host tags, pilot selection (item 5)
# --------------------------------------------------------------------------


def classify_fit(means: Any, stds: Any) -> str:
    if _bits_equal(means, PINNED_MEANS) and _bits_equal(stds, PINNED_STDS):
        return "pinned_1thread"
    if _bits_equal(means, DIRECT_MEANS) and _bits_equal(stds, PINNED_STDS):
        return "direct_121thread"
    return "other:" + _canon([_as_list(means), _as_list(stds)])


def direct_predicate(rec: dict) -> Dict[str, bool]:
    """[A3] three criteria of the direct-run set (all must agree)."""
    rh = rec.get("reinhard") or {}
    flags = rec.get("torch_backend_flags") or {}
    threads = flags.get("main_process_num_threads")
    return {"means_direct": _bits_equal(rh.get("target_means", []), DIRECT_MEANS),
            "lsf_job_null": rec.get("lsf_job", "missing") is None,
            "threads_121": type(threads) is int and threads == DIRECT_MAIN_THREADS}


def first_run_record_paths(first_run_root: Path, encoder: str) -> Dict[str, Path]:
    d = Path(first_run_root) / "features" / encoder
    return {p.name[:-len(".record.json")]: p for p in sorted(d.glob("*.record.json"))}


def capture_host_tags_main(argv=None, *, host_info: Callable[[str], Dict[str, Any]] = lshosts_info) -> int:
    p = argparse.ArgumentParser(prog="capture-host-tags")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--hosts", default="")
    p.add_argument("--first-run-root", type=Path, default=None)
    p.add_argument("--encoder", default="h_optimus_1")
    a = p.parse_args(argv)
    hosts = set(_split_list(a.hosts))
    unreadable = []
    if a.first_run_root is not None:
        for sid, path in first_run_record_paths(a.first_run_root, a.encoder).items():
            try:
                rec = json.loads(path.read_text())
            except (OSError, ValueError) as e:
                unreadable.append({"slide_id": sid, "reason": f"{type(e).__name__}: {e}"})
                continue
            h = rec.get("host") if isinstance(rec, dict) else None
            if h:
                hosts.add(h)
    if not hosts:
        raise SuiteError("no hosts given")
    doc = {"schema": "r2_1_perturb_suite_host_tags_v1", "source": "lshosts -w <host>",
           "hosts": {h: host_info(h) for h in sorted(hosts)},
           "unreadable_first_run_records": unreadable}
    dest = a.out_root / "environment" / "host_driver_tags.json"
    st = publish_idempotent(a.out_root, dest, _dump_pretty(doc))
    print(json.dumps({"host_tags": str(dest), "status": st, "n_hosts": len(hosts)}), flush=True)
    return 0


def _driver_tag(host_tags: Mapping[str, Any], host: str) -> str:
    info = host_tags.get(host)
    if info is None:
        raise SuiteError(f"no lshosts driver tag captured for host {host}")
    tags = info.get("driver_tags") or []
    return "+".join(tags) if tags else "none"


def select_pilot(records: Sequence[dict], host_tags: Mapping[str, Any], index: Mapping[str, Any],
                 *, min_tiles: int = PILOT_MIN_TILES, min_slides: int = PILOT_MIN_SLIDES) -> dict:
    """[A5] group by (host, LSF/direct, Reinhard fit, GPU driver tag); lexically smallest
    >= min_tiles slide per group; fill to min_slides with the lexically smallest remaining
    >= min_tiles slides; add the lexically smallest >= min_tiles slide fingerprinted on each
    CPU model string of INDEX.json not yet covered."""
    groups: Dict[Tuple[str, str, str, str], List[dict]] = {}
    for r in records:
        rh = r.get("reinhard") or {}
        key = (str(r.get("host")), "direct" if r.get("lsf_job") is None else "lsf",
               classify_fit(rh.get("target_means", []), rh.get("target_stds", [])),
               _driver_tag(host_tags, str(r.get("host"))))
        groups.setdefault(key, []).append(r)
    chosen: Dict[str, dict] = {}

    def add(sid: str, reason: str, **kw) -> None:
        e = chosen.setdefault(sid, {"slide_id": sid, "reasons": [], **kw})
        e["reasons"].append(reason)

    by_sid = {r["slide_id"]: r for r in records}
    group_out = []
    for key in sorted(groups):
        elig = sorted(r["slide_id"] for r in groups[key] if int(r.get("n_tiles", 0)) >= min_tiles)
        g = {"host": key[0], "run": key[1], "fit": key[2], "driver_tag": key[3],
             "n_records": len(groups[key]), "selected": elig[0] if elig else None}
        group_out.append(g)
        if elig:
            add(elig[0], "group:" + "|".join(key))
    for sid in sorted(s for s, r in by_sid.items() if int(r.get("n_tiles", 0)) >= min_tiles):
        if len(chosen) >= min_slides:
            break
        if sid not in chosen:
            add(sid, "fill")
    idx_slides = {e["slide_id"]: e for e in index.get("slides", [])}
    for sid, e in chosen.items():
        e["has_first_run_record"] = sid in by_sid
    models = sorted({str(e.get("cpu_model")) for e in idx_slides.values()})
    for model in models:
        if any(str(idx_slides.get(s, {}).get("cpu_model")) == model for s in chosen):
            continue
        elig = sorted(s for s, e in idx_slides.items()
                      if str(e.get("cpu_model")) == model and int(e["n_tiles"]) >= min_tiles)
        if elig:
            add(elig[0], f"cpu_model:{model}")
            chosen[elig[0]]["has_first_run_record"] = elig[0] in by_sid
    for sid, e in chosen.items():
        r = by_sid.get(sid)
        e["n_tiles"] = int(r["n_tiles"]) if r else int(idx_slides[sid]["n_tiles"])
        e["first_run_host"] = r.get("host") if r else None
        e["fingerprint_cpu_model"] = idx_slides.get(sid, {}).get("cpu_model")
        e["fingerprint_host"] = idx_slides.get(sid, {}).get("host")
    return {"groups": group_out,
            "uncovered_groups": [g for g in group_out if g["selected"] is None],
            "driver_tags": sorted({g["driver_tag"] for g in group_out}),
            "cpu_models": models,
            "slides": [chosen[s] for s in sorted(chosen)]}


def select_pilot_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="select-pilot")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--first-run-root", type=Path, required=True)
    p.add_argument("--encoder", default="h_optimus_1")
    p.add_argument("--host-tags", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True, help="fingerprints/INDEX.json")
    a = p.parse_args(argv)
    host_tags = json.loads(a.host_tags.read_text())["hosts"]
    index = json.loads(a.index.read_text())
    records, invalid = [], []
    for sid, path in first_run_record_paths(a.first_run_root, a.encoder).items():
        try:
            r = json.loads(path.read_text())
            if r.get("schema") != FIRST_RUN_SCHEMA or r.get("slide_id") != sid:
                raise ValueError("schema/slide_id")
            records.append(r)
        except Exception as exc:                          # noqa: BLE001
            invalid.append({"slide_id": sid, "reason": f"{type(exc).__name__}: {exc}"})
    sel = select_pilot(records, host_tags, index)
    doc = {"schema": "r2_1_perturb_suite_pilot_slides_v1", "n_first_run_records": len(records),
           "invalid_records": invalid, "index_sha256": X.sha256_file(a.index),
           "host_tags_sha256": X.sha256_file(a.host_tags), **sel}
    dest = a.out_root / "pilot_bit_identity" / "PILOT_SLIDES.json"
    st = publish_idempotent(a.out_root, dest, _dump_pretty(doc))
    print(json.dumps({"pilot": str(dest), "status": st, "n_slides": len(sel["slides"]),
                      "uncovered_groups": len(sel["uncovered_groups"])}), flush=True)
    return 0


# --------------------------------------------------------------------------
# Pilot (a): in-line re-render vs CPU fingerprints
# --------------------------------------------------------------------------


def _reference_pixels(root: Path, sid: str, tile: int, cond: str) -> Optional[np.ndarray]:
    d = root / "reference_png" / sid
    if not (d / "INDEX.json").exists():
        return None
    p = d / f"{tile}_{cond}.png"
    return np.asarray(_decode_png_rgb(p.read_bytes())) if p.exists() else None


def compare_digest_arrays(ref: np.ndarray, new: np.ndarray) -> Dict[str, Any]:
    out = {}
    for j, c in enumerate(SUITE_CONDITIONS):
        diff = np.flatnonzero(~(ref[:, j] == new[:, j]).all(axis=1))
        out[c] = {"differing_tiles": int(diff.size), "first_differing": diff[:50].tolist()}
    return out


def pilot_render_main(argv=None, *, opener: Optional[Callable] = None,
                      slide_path_of: Callable[[str], Path] = paths.slide_wsi_path,
                      reinhard_factory: Optional[Callable] = None,
                      probe: Optional[EnvProbe] = None,
                      gpu_info: Callable[[str], Dict[str, Any]] = gpu_info_default) -> int:
    p = argparse.ArgumentParser(prog="pilot-render")
    p.add_argument("--out-root", type=Path, required=True)
    add_input_args(p)
    add_env_args(p)
    p.add_argument("--pilot", type=Path, default=None)
    p.add_argument("--slides", default="")
    p.add_argument("--device", default="cuda", help="GPU recorded for the host stratum")
    p.add_argument("--processes", type=int, default=3)
    p.add_argument("--loader-batch", type=int, default=8)
    a = p.parse_args(argv)
    inp = load_inputs(a)
    sel = _pilot_slides(a)
    env, rh, _ = _prepare_render_env(a, probe, reinhard_factory, opener=opener,
                                     probe_wsi=_default_probe_wsi(a, inp.rows, slide_path_of))
    golden = blur_golden_check()
    gpu = gpu_info(a.device)
    opener = opener or openslide_open
    fp_dir = a.out_root / "fingerprints"
    any_fail = not golden["ok"]
    reports = {}
    for sid in sel:
        row = inp.row_by_slide[sid]
        st, why, _, ref = validate_fp_pair(fp_dir, sid, n_tiles=int(row["feature_rows"]),
                                           manifest_v2_sha256=inp.manifest.sha256)
        stimuli = inp.manifest.stimuli_v1(sid)
        rep: Dict[str, Any] = {"slide_id": sid, "fingerprint_status": st, "reason": why}
        if st == "complete":
            coords = X.load_bank_coords(row)
            slide_path = Path(slide_path_of(sid))
            arr, nonfinite, errors = render_digests(str(slide_path), coords, stimuli, rh, opener,
                                                    a.processes, a.loader_batch)
            rep["tile_errors"] = errors[:200]
            rep["reinhard_nonfinite_tile_indices"] = nonfinite
            cmp_ = compare_digest_arrays(ref, arr) if not errors else {}
            for c, d in cmp_.items():
                diffs = []
                for t in d["first_differing"]:
                    if t >= REFERENCE_PNG_TILES:
                        continue
                    want = _reference_pixels(a.out_root, sid, t, c)
                    if want is None:
                        continue
                    loc = (int(coords[t]["wsi_loc"][0]), int(coords[t]["wsi_loc"][1]))
                    s = opener(str(slide_path))
                    pics, _ = render_pictures(s, loc, X.shift_of(stimuli), stimuli, rh)
                    got = np.asarray(pics[c])
                    diffs.append(int(np.abs(got.astype(np.int16) - want.astype(np.int16)).max()))
                # [D5] max-abs is over SAVED tiles only (reference PNG tiles 0-3), not over
                # all differing tiles; other tiles have no stored pixels.
                d["max_abs_pixel_difference_saved_tiles_only"] = max(diffs) if diffs else None
                d["max_abs_pixel_difference_basis"] = (
                    "max-abs over saved tiles only (reference PNG tiles 0-3 that differ); "
                    "other differing tiles have no stored pixels and are not included")
            rep["comparison"] = cmp_
            rep["exact_all_13"] = bool(cmp_) and all(d["differing_tiles"] == 0 for d in cmp_.values())
        else:
            rep["exact_all_13"] = False
        any_fail |= not rep["exact_all_13"]
        reports[sid] = rep
    doc = {"schema": "r2_1_perturb_suite_pilot_render_v1", "slides": reports,
           "blur_golden": golden, "gpu": gpu, "environment": env,
           "manifest_v2_sha256": inp.manifest.sha256, "pass": not any_fail,
           "hf_cache_disclosure": hf_cache_disclosure(False),
           "suite_code_sha256": SUITE_CODE_SHA256, "extractor_code_sha256": EXTRACTOR_SHA256,
           "written_at": _now()}
    publish(a.out_root, a.out_root / "pilot_bit_identity" / "render" / f"{_tag()}.json",
            _dump_pretty(doc))
    print(json.dumps({"pass": not any_fail, "slides": {s: r["exact_all_13"] for s, r in reports.items()}}),
          flush=True)
    return 0 if not any_fail else 1


# --------------------------------------------------------------------------
# Pilot (b): frozen X.process_slide vs first-run pickles [A1]
# --------------------------------------------------------------------------


def _load_verified_pickle(pkl: Path, rec_path: Path) -> Tuple[dict, Any, dict]:
    rec = json.loads(Path(rec_path).read_text())
    data = Path(pkl).read_bytes()
    if len(data) != rec.get("pickle_bytes") or _sha(data) != rec.get("pickle_sha256"):
        raise SuiteError(f"{pkl}: pickle size/sha256 != record")
    feats, coords = pickle.loads(data)
    return feats, coords, rec


def _first_coord_difference(a: Any, b: Any) -> Dict[str, Any]:
    """[R2] report the first differing coordinate in the tagged (type-strict, numpy-aware)
    form, e.g. ``["np", "<i8", 7]`` vs ``7``."""
    la, lb = list(a), list(b)
    for i in range(min(len(la), len(lb))):
        if _canon(la[i]) != _canon(lb[i]):
            return {"index": i, "a": _canon_tag(la[i]), "b": _canon_tag(lb[i])}
    return {"index": min(len(la), len(lb)), "len_a": len(la), "len_b": len(lb)}


def compare_arms(a: Mapping[str, np.ndarray], a_coords: Any, b: Mapping[str, np.ndarray],
                 b_coords: Any, conditions: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"coords_equal": _canon(a_coords) == _canon(b_coords), "arms": {}}
    if not out["coords_equal"]:
        out["coords_first_difference"] = _first_coord_difference(a_coords, b_coords)
    for c in conditions:
        x, y = a.get(c), b.get(c)
        if x is None or y is None:
            out["arms"][c] = {"exact": False, "reason": "missing arm"}
            continue
        same = (x.dtype == y.dtype and x.shape == y.shape and bool(np.array_equal(x, y)))
        e: Dict[str, Any] = {"exact": same, "dtype": [str(x.dtype), str(y.dtype)],
                             "shape": [list(x.shape), list(y.shape)]}
        if not same and x.shape == y.shape:
            neq = ~(x == y).all(axis=1) if x.ndim == 2 else np.array([True])
            e["differing_rows"] = int(neq.sum())
            e["max_abs_difference"] = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        out["arms"][c] = e
    return out


def _first_run_class(rec: dict) -> str:
    pred = direct_predicate(rec)
    if all(pred.values()):
        return "direct"
    if not any(pred.values()):
        return "lsf"
    return "predicate_disagreement"


INVOCATION_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,254}")
# [R1]/[R5c] loader settings compared per slide with the first-run record. X records no
# separate worker-init field; its worker initialisation is X's own code, identified by the
# record's ``code_sha256`` (asserted == EXTRACTOR_SHA256). X does not record prefetch_factor.
PILOT_LOADER_KEYS = ("forward_batch", "loader_batch", "num_workers")


def hf_cache_disclosure(reads_hf_cache: bool = True) -> Dict[str, Any]:
    """[R1] disclosure for the GPU sub-commands: encoder weights/config come from the Hugging
    Face cache, which is outside --out-root (read only; nothing is written there by the suite)."""
    doc = {"hf_cache_read_outside_out_root": bool(reads_hf_cache),
           "HF_HOME": os.environ.get("HF_HOME"), "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
           "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")}
    doc["note"] = ("the encoder (weights, config) is loaded from the Hugging Face cache outside "
                   "--out-root" if reads_hf_cache else
                   "no encoder is built by this sub-command; the Hugging Face cache is not read")
    return doc


class _CountingRuntime:
    """Wraps an ``X.EncoderRuntime`` so that every ``rt.forward`` call made by X._forward is
    counted (X itself is not patched): returns a new ``X.EncoderRuntime`` whose ``forward``
    delegates to the original."""

    def __init__(self, rt: Any):
        self.inner = rt
        self.calls = 0
        self.images = 0

        def forward(x):
            self.calls += 1
            self.images += int(x.shape[0])
            return rt.forward(x)
        self.runtime = X.EncoderRuntime(rt.name, rt.model, rt.transform, forward, rt.amp,
                                        rt.dim, rt.meta)

    def reset(self) -> None:
        self.calls = 0
        self.images = 0


def _slide_outputs_under(run_root: Path, sid: str) -> List[str]:
    """Existing outputs of ``sid`` under an X run root: any ``features/*/<sid>.*`` file
    (incl. X's tmp names) or a ``failures.jsonl`` line for the slide."""
    hits: List[str] = []
    fdir = run_root / "features"
    if fdir.is_dir():
        for enc_dir in sorted(fdir.iterdir()):
            if enc_dir.is_dir():
                hits += [str(q) for q in sorted(enc_dir.iterdir())
                         if q.name.startswith(f"{sid}.") or q.name.startswith(f".{sid}.")]
    fj = run_root / "failures.jsonl"
    if fj.is_file():
        for ln in fj.read_text().splitlines():
            try:
                if json.loads(ln).get("slide_id") == sid:
                    hits.append(f"{fj} (slide_id={sid})")
                    break
            except ValueError:
                hits.append(f"{fj} (unparseable line)")
                break
    return hits


def _slide_failures(run_root: Path, sid: str) -> List[dict]:
    fj = run_root / "failures.jsonl"
    out: List[dict] = []
    if fj.is_file():
        for ln in fj.read_text().splitlines():
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if d.get("slide_id") == sid:
                out.append({k: d.get(k) for k in ("stage", "reason", "encoder", "time")})
    return out


# [R6c] X ``_hf_weights_info`` emits ``repo_id``, ``file`` (the specification's "filename"),
# ``snapshot_revision``, ``cache_path``, ``blob``, ``bytes`` and (when hashing) ``sha256``.
# Only content keys are compared; path keys (and any other key) are recorded, not compared.
WEIGHTS_CONTENT_KEYS: Tuple[str, ...] = ("repo_id", "file", "snapshot_revision", "bytes",
                                         "sha256")


def weights_content_check(first_run: Any, current: Any) -> Dict[str, Any]:
    """[R6c] equal iff both are dicts carrying every content key with equal (type-strict)
    values; a missing content key on either side is a mismatch."""
    if not isinstance(first_run, dict) or not isinstance(current, dict):
        return {"equal": False, "compared_keys": list(WEIGHTS_CONTENT_KEYS),
                "missing_keys": list(WEIGHTS_CONTENT_KEYS), "differing_keys": [],
                "reason": "weights info missing or not a dict"}
    missing = [k for k in WEIGHTS_CONTENT_KEYS if k not in first_run or k not in current]
    differing = [k for k in WEIGHTS_CONTENT_KEYS if k not in missing
                 and _canon(first_run[k]) != _canon(current[k])]
    not_compared = sorted((set(first_run) | set(current)) - set(WEIGHTS_CONTENT_KEYS))
    return {"equal": not missing and not differing, "compared_keys": list(WEIGHTS_CONTENT_KEYS),
            "missing_keys": missing, "differing_keys": differing,
            "recorded_not_compared_keys": not_compared}


def pilot_process_slide_main(argv=None, *, encoder_factory: Optional[Callable] = None,
                             opener: Callable = X.openslide_opener,
                             slide_path_of: Callable[[str], Path] = paths.slide_wsi_path,
                             reinhard_factory: Optional[Callable] = None,
                             probe: Optional[EnvProbe] = None,
                             gpu_info: Callable[[str], Dict[str, Any]] = gpu_info_default) -> int:
    p = argparse.ArgumentParser(prog="pilot-process-slide")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--v1", type=Path, required=True)
    p.add_argument("--v1-sha256", default=X.MANIFEST_SHA256)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--queries-sha256", default=X.QUERIES_SHA256)
    add_env_args(p)
    p.add_argument("--pilot", type=Path, default=None)
    p.add_argument("--slides", default="")
    p.add_argument("--first-run-root", type=Path, required=True)
    p.add_argument("--encoder", default="h_optimus_1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--invocation-tag", required=True,
                   help="[R1] unique per invocation; run root is pilot_bit_identity/process_slide/"
                        "<driver_tag>_<host>_<invocation_tag>/ ([A-Za-z0-9][A-Za-z0-9._-]{0,63})")
    p.add_argument("--twice", action="store_true",
                   help="pre-specified diagnostic: encode the slides twice in this job")
    p.add_argument("--no-hash-weights", action="store_true")
    a = p.parse_args(argv)
    if not INVOCATION_TAG_RE.fullmatch(a.invocation_tag):
        raise SuiteError(f"--invocation-tag {a.invocation_tag!r} must match "
                         f"{INVOCATION_TAG_RE.pattern}")
    check_worker_tempdir(PILOT_PROCESS_SLIDE["num_workers"])
    if not a.v1_sha256 or not a.queries_sha256:
        raise SuiteError("explicit non-empty sha256 values are required")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable (no silent CPU fallback)")
    v1_sha, manifest = X.load_manifest(a.v1, a.v1_sha256)
    q_sha, rows = X.load_queries(a.queries, a.queries_sha256)
    row_by = {r["slide_id"]: r for r in rows}
    sel = _pilot_slides(a)
    for sid in sel:
        if sid not in row_by or sid not in manifest:
            raise SuiteError(f"{sid}: not in queries/manifest")
        check_template_digest(manifest[sid], sid)
    env, rh, rh_info = _prepare_render_env(a, probe, reinhard_factory, opener=opener,
                                           probe_wsi=_default_probe_wsi(a, rows, slide_path_of))
    # [R1] per-invocation run root: driver tag = the lshosts GPU driver tag(s) of this host as
    # recorded in the environment block (same source as capture-host-tags), host = hostname.
    host = str(env["recorded"]["hostname"])
    tags = env["recorded"].get("gpu_driver_tags") or []
    driver = "+".join(tags) if tags else "none"
    run_name = f"{driver}_{host}_{a.invocation_tag}"
    if not RUN_NAME_RE.fullmatch(run_name) or ".." in run_name:
        raise SuiteError(f"run root name {run_name!r} is not a safe path component")
    base_root = a.out_root / "pilot_bit_identity"
    roots = {"process_slide": base_root / "process_slide" / run_name}
    if a.twice:
        roots.update({"twice_run1": base_root / "twice" / run_name / "run1",
                      "twice_run2": base_root / "twice" / run_name / "run2"})
    for root in roots.values():
        _inside(a.out_root, root)
    pre = {sid: hits for sid in sel
           for hits in [[h for root in roots.values() for h in _slide_outputs_under(root, sid)]]
           if hits}
    if pre:
        raise SuiteError(f"[R1] refusing to start: selected slides already have outputs under "
                         f"the run root(s) of invocation {run_name!r}: "
                         f"{json.dumps(pre, sort_keys=True)[:2000]}")
    golden = blur_golden_check()
    gpu = gpu_info(a.device)
    amp = X.AMP_DEFAULTS[a.encoder]
    hf_offline = os.environ.setdefault("HF_HUB_OFFLINE", "1")     # [R1] as extract / X.run
    factory = encoder_factory or (lambda name, device, amp_flag: X.build_encoder(
        name, device, amp_flag, hash_weights=not a.no_hash_weights))
    rt = factory(a.encoder, a.device, amp)
    counter = _CountingRuntime(rt)
    encoders = {a.encoder: counter.runtime}
    job_meta = X._job_meta(a.device)
    s = PILOT_PROCESS_SLIDE
    current_weights = (rt.meta or {}).get("weights") if isinstance(rt.meta, dict) else None

    def ctx_for(run_root: Path):
        _inside(a.out_root, run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        return X.RunContext(run_root=run_root, manifest_path=a.v1, manifest_sha256=v1_sha,
                            queries_path=a.queries, queries_sha256=q_sha, device=a.device,
                            forward_batch=s["forward_batch"], loader_batch=s["loader_batch"],
                            num_workers=s["num_workers"], prefetch_factor=s["prefetch_factor"],
                            reinhard=rh, reinhard_info=rh_info, opener=opener,
                            slide_path_of=slide_path_of, limit_tiles=0, job_meta=job_meta)

    first_recs = first_run_record_paths(a.first_run_root, a.encoder)
    all_ok = golden["ok"]
    reports: Dict[str, Any] = {}
    for sid in sel:
        rep: Dict[str, Any] = {"slide_id": sid, "runs": {}}
        failures: List[str] = []
        outs: Dict[str, Any] = {}
        for label, root in roots.items():
            counter.reset()
            st = X.process_slide(ctx_for(root), row_by[sid], manifest[sid], encoders)
            status = st.get(a.encoder)
            run = {"status": status, "run_root": str(root), "forward_calls": counter.calls,
                   "forward_images": counter.images}
            rep["runs"][label] = run
            if status == "written" and counter.calls > 0:
                # process_slide's own output location (X.output_paths, X line 552; not called
                # directly -- specification [A1] lists it as used only through process_slide)
                pkl, rec = suite_output_paths(root, a.encoder, sid)
                try:
                    outs[label] = _load_verified_pickle(pkl, rec)
                    run["reinhard_nonfinite_tile_indices"] = outs[label][2].get(
                        "reinhard_nonfinite_tile_indices")
                except Exception as exc:                  # noqa: BLE001
                    run["output_error"] = f"{type(exc).__name__}: {exc}"
                    failures.append(f"{label}:output_unreadable")
            elif status == "failed":
                run["failures"] = _slide_failures(root, sid)
                failures.append(f"{label}:process_slide_failed")
            else:
                # [R1] 'skipped' (or written without a single forward) is not a fresh encode
                failures.append(f"{label}:not_encoded")
        if sid not in first_recs:
            rep["first_run"] = {"status": "absent"}
            failures.append("first_run_absent")
        else:
            fpkl = first_recs[sid].with_name(f"{sid}.pkl")
            try:
                ff, fc, frec = _load_verified_pickle(fpkl, first_recs[sid])
            except Exception as exc:                      # noqa: BLE001
                rep["first_run"] = {"status": "invalid", "reason": f"{type(exc).__name__}: {exc}"}
                failures.append("first_run_invalid")
                ff = None
            if ff is not None:
                try:
                    cls = _first_run_class(frec)
                    rep["first_run"] = {
                        "status": "present", "class": cls, "predicate": direct_predicate(frec),
                        "fit": classify_fit(frec["reinhard"]["target_means"],
                                            frec["reinhard"]["target_stds"]),
                        "host": frec.get("host"), "lsf_job": frec.get("lsf_job"),
                        "loader_settings": {k: frec.get(k) for k in s}}
                    if cls == "predicate_disagreement":
                        failures.append("predicate_disagreement")
                    # [R1] weights info (X _hf_weights_info, record encoder_meta.weights)
                    em = frec.get("encoder_meta")
                    fw = em.get("weights") if isinstance(em, dict) else None
                    wchk = weights_content_check(fw, current_weights)
                    wok = wchk["equal"]
                    rep["weights_check"] = dict(wchk, first_run=fw,
                                                this_invocation=current_weights)
                    if not wok:
                        failures.append("weights_mismatch")
                    # [R5c] loader settings + X code (worker init) vs this invocation's
                    want = {k: s[k] for k in PILOT_LOADER_KEYS}
                    want["code_sha256"] = EXTRACTOR_SHA256
                    got = {k: frec.get(k) for k in want}
                    lok = _canon(got) == _canon(want)
                    rep["loader_check"] = {"equal": lok, "first_run": got,
                                           "this_invocation": want,
                                           "not_recorded_by_X": ["prefetch_factor"]}
                    if not lok:
                        failures.append("loader_mismatch")
                    allowed = {"reinhard"} if cls == "direct" else set()
                    for label, (f, c, _) in outs.items():
                        cmp_ = compare_arms(f, c, ff, fc, X.CONDITIONS)
                        bad = [k for k, v in cmp_["arms"].items()
                               if not v["exact"] and k not in allowed]
                        cmp_["unexpected_non_exact_arms"] = bad
                        cmp_["pass"] = (cmp_["coords_equal"] and not bad
                                        and cls != "predicate_disagreement")
                        rep["runs"][label]["vs_first_run"] = cmp_
                        if not cmp_["coords_equal"]:
                            failures.append(f"{label}:coords_not_equal")
                        if bad:
                            failures.append(f"{label}:arms_not_exact")
                except Exception as exc:                  # noqa: BLE001  [R2]
                    rep["comparison_error"] = f"{type(exc).__name__}: {exc}"
                    failures.append("comparison_error")
        if "twice_run1" in outs and "twice_run2" in outs:
            f1, c1, _ = outs["twice_run1"]
            f2, c2, _ = outs["twice_run2"]
            try:
                rep["twice_run1_vs_run2"] = compare_arms(f1, c1, f2, c2, X.CONDITIONS)
            except Exception as exc:                      # noqa: BLE001  [R2]
                rep["twice_run1_vs_run2"] = {"comparison_error": f"{type(exc).__name__}: {exc}"}
        if "process_slide" not in outs and not any(f.startswith("process_slide:")
                                                   for f in failures):
            failures.append("process_slide:not_encoded")
        rep["failures"] = failures
        rep["pass"] = not failures
        all_ok = all_ok and rep["pass"]
        reports[sid] = rep
    doc = {"schema": "r2_1_perturb_suite_pilot_process_slide_v2", "slides": reports,
           "pass": bool(all_ok), "blur_golden": golden, "gpu": gpu, "environment": env,
           "invocation": {"tag": a.invocation_tag, "driver_tag": driver, "host": host,
                          "run_name": run_name,
                          "run_roots": {k: str(v) for k, v in roots.items()}},
           "hf_hub_offline": hf_offline, "hf_cache_disclosure": hf_cache_disclosure(True),
           "run_context": dict(s, device=a.device, manifest_sha256=v1_sha, queries_sha256=q_sha,
                               amp=amp, main_process_num_threads=torch.get_num_threads()),
           "forward_count_method": "X.EncoderRuntime wrapper counting rt.forward calls made by "
                                   "X._forward (X unpatched); reset per process_slide call",
           "expected": "LSF slides: all 12 arms exact; direct-run (A3 predicate) slides: "
                       "reinhard may differ (G3), other 11 exact; every compared slide freshly "
                       "encoded ('written', >0 forwards) in this invocation [R1]",
           "suite_code_sha256": SUITE_CODE_SHA256, "extractor_code_sha256": EXTRACTOR_SHA256,
           "written_at": _now()}
    publish(a.out_root, base_root / "process_slide_reports" / f"{_tag()}.json", _dump_pretty(doc))
    print(json.dumps({"pass": bool(all_ok), "run_name": run_name}), flush=True)
    return 0 if all_ok else 1


# --------------------------------------------------------------------------
# H-Optimus extension population [A3] (item 7)
# --------------------------------------------------------------------------

DIRECT_CONDITIONS = ("identity", "reinhard", BLUR_CONDITION)
PINNED_CONDITIONS = ("identity", BLUR_CONDITION)


def validate_first_run_record(rec_path: Path, sid: str, row: Mapping[str, Any], encoder: str,
                              *, manifest_sha256: str = X.MANIFEST_SHA256,
                              verify_pickle: bool = True) -> Tuple[Optional[dict], str]:
    """(record, '') if valid else (record-or-None, reason)."""
    try:
        rec = json.loads(Path(rec_path).read_text())
    except Exception as exc:                              # noqa: BLE001
        return None, f"unreadable record: {type(exc).__name__}: {exc}"
    if not isinstance(rec, dict):
        return None, "record is not an object"
    checks = [(rec.get("schema") == FIRST_RUN_SCHEMA, "schema"),
              (rec.get("slide_id") == sid, "slide_id"),
              (rec.get("encoder") == encoder, "encoder"),
              (rec.get("manifest_sha256") == manifest_sha256, "manifest_sha256"),
              (rec.get("code_sha256") == EXTRACTOR_SHA256, "code_sha256"),
              (rec.get("n_tiles") == int(row["feature_rows"]), "n_tiles"),
              (rec.get("conditions") == list(X.CONDITIONS), "conditions"),
              (rec.get("limit_tiles") is None, "limit_tiles")]
    for ok, what in checks:
        if not ok:
            return rec, f"record {what} mismatch"
    pkl = Path(rec_path).with_name(f"{sid}.pkl")
    if not pkl.is_file() or pkl.stat().st_size != rec.get("pickle_bytes"):
        return rec, "pickle missing or size != record"
    if verify_pickle and X.sha256_file(pkl) != rec.get("pickle_sha256"):
        return rec, "pickle sha256 != record"
    return rec, ""


def plan_extension(rows: Sequence[dict], record_paths: Mapping[str, Path], encoder: str, *,
                   expected_direct: int, manifest_sha256: str = X.MANIFEST_SHA256,
                   verify_pickle: bool = True) -> dict:
    """Per-slide conditions (A3). Raises PopulationStop (with a report) on predicate
    disagreement, a third Reinhard fit, or a direct-set count != expected."""
    slides, stops = [], []
    n_direct = 0
    for r in rows:
        sid = r["slide_id"]
        path = record_paths.get(sid)
        entry: Dict[str, Any] = {"slide_id": sid}
        rec, why = (None, "no first-run record") if path is None else \
            validate_first_run_record(path, sid, r, encoder, manifest_sha256=manifest_sha256,
                                      verify_pickle=verify_pickle)
        cls = None
        if rec is not None and isinstance(rec.get("reinhard"), dict):
            pred = direct_predicate(rec)
            fit = classify_fit(rec["reinhard"].get("target_means", []),
                               rec["reinhard"].get("target_stds", []))
            entry.update(predicate=pred, fit=fit)
            if len(set(pred.values())) != 1:
                stops.append({"slide_id": sid, "stop": "direct predicate criteria disagree",
                              "predicate": pred, "fit": fit})
            elif all(pred.values()):
                cls = "direct"
                if fit != "direct_121thread":
                    stops.append({"slide_id": sid, "stop": "direct-run record stds are not the "
                                  "recorded values", "fit": fit})
            else:
                cls = "pinned"
                if fit != "pinned_1thread":
                    stops.append({"slide_id": sid, "stop": "third Reinhard fit", "fit": fit})
        elif rec is not None and not why:
            why = "record has no reinhard block"
        if path is not None:
            entry["record_path"] = str(path)
        if rec is None or why:
            entry.update(first_run_record="absent", reason=why, conditions=list(SUITE_CONDITIONS),
                         **({"invalid_record_class": cls} if cls else {}))
        else:
            entry.update(first_run_record="present", first_run_class=cls,
                         record_sha256=X.sha256_file(path),
                         pickle_sha256=rec["pickle_sha256"], pickle_bytes=rec["pickle_bytes"],
                         conditions=list(DIRECT_CONDITIONS if cls == "direct" else PINNED_CONDITIONS))
            n_direct += cls == "direct"
        slides.append(entry)
    if n_direct != expected_direct:
        stops.append({"stop": f"direct-run set has {n_direct} valid records, expected "
                              f"{expected_direct}"})
    counts = {"slides": len(slides), "direct": n_direct,
              "pinned": sum(s.get("first_run_class") == "pinned" for s in slides),
              "absent": sum(s["first_run_record"] == "absent" for s in slides)}
    doc = {"schema": PLAN_SCHEMA, "encoder": encoder, "expected_direct_count": expected_direct,
           "pickle_sha256_verified": verify_pickle, "counts": counts, "slides": slides,
           "rules": {"direct": "means bit-equal DIRECT_MEANS AND lsf_job null AND "
                               "main_process_num_threads == 121 (all agree) -> "
                               + ",".join(DIRECT_CONDITIONS),
                     "pinned": "pinned 1-thread means and stds -> " + ",".join(PINNED_CONDITIONS),
                     "absent": "no valid record -> all 13 conditions, first_run_record absent"}}
    if stops:
        raise PopulationStop(_canon({"stops": stops, "counts": counts}))
    return doc


def plan_extension_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="plan-extension")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--queries-sha256", default=X.QUERIES_SHA256)
    p.add_argument("--first-run-root", type=Path, required=True)
    p.add_argument("--v1-sha256", default=X.MANIFEST_SHA256,
                   help="manifest v1 sha256 the first-run records must carry")
    p.add_argument("--encoder", default="h_optimus_1")
    p.add_argument("--expected-direct-count", type=int, default=None,
                   help=f"[R5b] optional override of EXPECTED_DIRECT_COUNT="
                        f"{EXPECTED_DIRECT_COUNT}; needs --expected-direct-count-reason; "
                        "value and reason are written into PLAN")
    p.add_argument("--expected-direct-count-reason", default=None)
    p.add_argument("--size-only", action="store_true",
                   help="do not hash first-run pickles (size check only); recorded")
    a = p.parse_args(argv)
    if not a.queries_sha256 or not a.v1_sha256:
        raise SuiteError("--queries-sha256/--v1-sha256 must be non-empty")
    if a.expected_direct_count is None:
        if a.expected_direct_count_reason is not None:
            raise SuiteError("--expected-direct-count-reason given without an override")
        expected = EXPECTED_DIRECT_COUNT
        override = None
    else:
        if not (a.expected_direct_count_reason or "").strip():
            raise SuiteError("--expected-direct-count override needs a non-empty "
                             "--expected-direct-count-reason [R5b]")
        expected = a.expected_direct_count
        override = {"value": a.expected_direct_count, "reason": a.expected_direct_count_reason,
                    "module_constant": EXPECTED_DIRECT_COUNT}
    qsha, rows = X.load_queries(a.queries, a.queries_sha256)
    recs = first_run_record_paths(a.first_run_root, a.encoder)
    ext = a.out_root / "extension"
    try:
        doc = plan_extension(rows, recs, a.encoder, expected_direct=expected,
                             manifest_sha256=a.v1_sha256, verify_pickle=not a.size_only)
    except PopulationStop as exc:
        rep = {"schema": "r2_1_perturb_suite_extension_stop_v1", "queries_sha256": qsha,
               "first_run_root": str(a.first_run_root), "detail": json.loads(str(exc)),
               "expected_direct_count": expected,
               "expected_direct_count_override": override, "time": _now()}
        publish(a.out_root, ext / f"STOP_{_tag()}.json", _dump_pretty(rep))
        print(f"[suite] plan-extension STOP: {str(exc)[:2000]}", flush=True)
        return 2
    doc.update(queries_sha256=qsha, first_run_root=str(a.first_run_root),
               manifest_v1_sha256=a.v1_sha256,
               expected_direct_count_module_constant=EXPECTED_DIRECT_COUNT,
               expected_direct_count_override=override,
               records_not_in_queries=sorted(set(recs) - {r["slide_id"] for r in rows}),
               suite_code_sha256=SUITE_CODE_SHA256, created_at=_now())
    raw = _dump_pretty(doc)
    dest = ext / "PLAN.json"
    publish(a.out_root, dest, raw)
    publish(a.out_root, ext / "PLAN.json.sha256", f"{_sha(raw)}  PLAN.json\n".encode())
    print(json.dumps({"plan": str(dest), "sha256": _sha(raw), **doc["counts"]}), flush=True)
    return 0


# --------------------------------------------------------------------------
# Verifying extractor wrapper (item 6)
# --------------------------------------------------------------------------


def suite_output_paths(root: Path, encoder: str, sid: str) -> Tuple[Path, Path]:
    d = root / "features" / encoder
    return d / f"{sid}.pkl", d / f"{sid}.record.json"


def existing_suite_output(pkl: Path, rec: Path, *, sid: str, encoder: str, manifest_v2_sha256: str,
                          n_tiles: int, conditions: Sequence[str]) -> Tuple[str, str]:
    has_p, has_r = pkl.exists(), rec.exists()
    if not has_p and not has_r:
        return "absent", ""
    if not (has_p and has_r):
        return "invalid", f"orphan output (pickle={has_p}, record={has_r})"
    try:
        r = json.loads(rec.read_text())
    except Exception as exc:                              # noqa: BLE001
        return "invalid", f"unreadable record: {exc}"
    checks = [(r.get("schema") == RECORD_SCHEMA, "schema"), (r.get("slide_id") == sid, "slide_id"),
              (r.get("encoder") == encoder, "encoder"),
              (r.get("manifest_v2_sha256") == manifest_v2_sha256, "manifest_v2_sha256"),
              (r.get("n_tiles") == n_tiles, "n_tiles"),
              (r.get("conditions_encoded") == list(conditions), "conditions_encoded"),
              (r.get("pickle_bytes") == pkl.stat().st_size, "pickle_bytes")]
    for ok, what in checks:
        if not ok:
            return "invalid", f"record {what} mismatch"
    if X.sha256_file(pkl) != r.get("pickle_sha256"):
        return "invalid", "pickle sha256 != record"
    return "complete", ""


def cpu_request() -> int:
    v = os.environ.get("LSB_DJOB_NUMPROC")
    if v:
        return int(v)
    return len(os.sched_getaffinity(0))


@dataclass
class ExtractContext:
    root: Path
    inp: Inputs
    encoder: str
    runtime: Any
    device: str
    forward_batch: int
    loader_batch: int
    num_workers: int
    prefetch_factor: int
    reinhard: Any
    reinhard_info: dict
    opener: Callable
    slide_path_of: Callable[[str], Path]
    env: dict
    gpu: dict
    plan: Optional[dict] = None
    plan_sha256: Optional[str] = None


def _ordered(conds: Sequence[str]) -> List[str]:
    bad = [c for c in conds if c not in SUITE_CONDITIONS]
    if bad or len(set(conds)) != len(conds) or not conds:
        raise SuiteError(f"bad conditions {list(conds)}")
    return [c for c in SUITE_CONDITIONS if c in conds]


def extract_slide(ctx: ExtractContext, sid: str, conditions: Sequence[str],
                  plan_entry: Optional[dict]) -> str:
    """'written' | 'skipped' | 'failed'. Nothing is published unless every tile x 13 pictures
    verified against the fingerprint and (when a first-run record is reused) the identity
    arm equals the first-run identity arm bit for bit."""
    root, inp, rt = ctx.root, ctx.inp, ctx.runtime
    row = inp.row_by_slide[sid]
    n_decl = int(row["feature_rows"])
    conds = _ordered(conditions)
    failed_dir = root / "features" / "failed" / ctx.encoder
    pkl, recp = suite_output_paths(root, ctx.encoder, sid)
    st, why = existing_suite_output(pkl, recp, sid=sid, encoder=ctx.encoder,
                                    manifest_v2_sha256=inp.manifest.sha256, n_tiles=n_decl,
                                    conditions=conds)
    if st == "complete":
        return "skipped"
    if st == "invalid":
        write_failed(root, failed_dir, sid, {"stage": "existing_output", "reason": why,
                                             "encoder": ctx.encoder})
        return "failed"

    def fail(**kw) -> str:
        write_failed(root, failed_dir, sid, dict(kw, encoder=ctx.encoder, conditions=conds))
        return "failed"

    fst, fwhy, fdoc, expected = validate_fp_pair(root / "fingerprints", sid, n_tiles=n_decl,
                                                 manifest_v2_sha256=inp.manifest.sha256)
    if fst != "complete":
        return fail(stage="fingerprint", reason=f"no validated fingerprint pair ({fst}: {fwhy})")
    stimuli = inp.manifest.stimuli_v1(sid)
    try:
        check_template_digest(stimuli, sid)
        coords = X.load_bank_coords(row)
        if len(coords) != n_decl:
            raise SuiteError(f"{len(coords)} coords != feature_rows {n_decl}")
        slide_path = Path(ctx.slide_path_of(sid))
        stat0 = wsi_stat(slide_path)
    except Exception as exc:                              # noqa: BLE001
        return fail(stage="inputs", reason=f"{type(exc).__name__}: {exc}")
    n = n_decl
    ds = RenderTiles(str(slide_path), coords, stimuli, ctx.reinhard, ctx.opener,
                     expected=expected, transform=rt.transform, select=conds)
    dl = make_loader(ds, ctx.num_workers, ctx.loader_batch, ctx.prefetch_factor)
    out = {c: np.empty((n, rt.dim), np.float32) for c in conds}
    filled = np.zeros(n, bool)
    verified = np.zeros(n, bool)
    mismatch_counts = {c: 0 for c in SUITE_CONDITIONS}
    mismatched_tiles: List[dict] = []
    tile_errors: List[dict] = []
    nonfinite_reinhard: List[int] = []
    pending: List[dict] = []
    fwd_seconds = 0.0
    t0 = time.time()

    def encode(items: List[dict]) -> None:
        nonlocal fwd_seconds
        idx = np.array([it["index"] for it in items])
        ts = time.time()
        for c in conds:
            feats = X._forward(rt, torch.stack([it["tensors"][c] for it in items]).contiguous(),
                               ctx.device)
            if feats.shape != (len(idx), rt.dim):
                raise SuiteError(f"{c}: feature shape {feats.shape}")
            out[c][idx] = feats
        fwd_seconds += time.time() - ts
        filled[idx] = True

    try:
        for batch in dl:
            for it in batch:
                i = it["index"]
                if "error" in it:
                    tile_errors.append(it["error"] | {"index": i})
                elif "mismatch" in it:
                    for c in it["mismatch"]:
                        mismatch_counts[c] += 1
                    mismatched_tiles.append({"index": i, "conditions": it["mismatch"]})
                else:
                    verified[i] = True
                    if not it["reinhard_finite"]:
                        nonfinite_reinhard.append(i)
                if tile_errors or mismatched_tiles:
                    pending.clear()       # stop all further forwards; keep draining to count
                    continue
                pending.append(it)
            while not (tile_errors or mismatched_tiles) and len(pending) >= ctx.forward_batch:
                encode(pending[:ctx.forward_batch])
                del pending[:ctx.forward_batch]
        if pending and not (tile_errors or mismatched_tiles):
            encode(pending)
        pending.clear()
    except Exception as exc:                              # noqa: BLE001
        return fail(stage="extract", reason=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc()[-4000:])
    wall = time.time() - t0
    if mismatched_tiles or tile_errors:
        return fail(stage="fingerprint_verification" if mismatched_tiles else "tile_render",
                    n_tiles=n, n_verified_tiles=int(verified.sum()),
                    n_mismatched_tiles=len(mismatched_tiles), mismatch_counts=mismatch_counts,
                    mismatched_tiles=mismatched_tiles[:500], n_error_tiles=len(tile_errors),
                    tile_errors=tile_errors[:200], published=False)
    if not verified.all() or not filled.all():
        return fail(stage="extract", reason=f"{int((~verified).sum())} unverified / "
                                            f"{int((~filled).sum())} unencoded tiles")
    bad = {c: int((~np.isfinite(a)).any(axis=1).sum()) for c, a in out.items()}
    if any(bad.values()):
        return fail(stage="nonfinite_features", nonfinite_rows=bad)
    stat1 = wsi_stat(slide_path)
    if stat1 != stat0:
        return fail(stage="wsi_changed", before=stat0, after=stat1)
    first_run: Dict[str, Any] = {"status": "not_applicable"}
    if plan_entry is not None:
        if plan_entry["first_run_record"] == "present":
            rp = Path(plan_entry["record_path"])
            try:
                if X.sha256_file(rp) != plan_entry["record_sha256"]:
                    raise SuiteError("first-run record changed since the plan")
                ff, fc, frec = _load_verified_pickle(rp.with_name(f"{sid}.pkl"), rp)
            except Exception as exc:                      # noqa: BLE001
                return fail(stage="first_run_identity", reason=f"{type(exc).__name__}: {exc}")
            # [R2] the comparison (type-strict, numpy-aware _canon) and compare_arms run under
            # fail(): any exception becomes this slide's FAILED record; the task continues.
            try:
                fi = ff.get("identity")
                eq = (isinstance(fi, np.ndarray) and fi.dtype == out["identity"].dtype
                      and fi.shape == out["identity"].shape
                      and bool(np.array_equal(fi, out["identity"]))
                      and _canon(fc) == _canon(coords))
                comparison = None if eq else compare_arms(out, coords, ff, fc, ["identity"])
            except Exception as exc:                      # noqa: BLE001
                return fail(stage="first_run_identity",
                            reason=f"comparison error {type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc()[-4000:])
            if not eq:
                return fail(stage="first_run_identity",
                            reason="identity arm != first-run identity arm (bit-exact)",
                            comparison=comparison)
            first_run = {"status": "present", "class": plan_entry["first_run_class"],
                         "record_path": str(rp), "record_sha256": plan_entry["record_sha256"],
                         "pickle_sha256": frec["pickle_sha256"], "identity_equal": True}
        else:
            first_run = {"status": "absent", "reason": plan_entry.get("reason")}
    payload = pickle.dumps((out, coords), protocol=4)
    record = {
        "schema": RECORD_SCHEMA, "slide_id": sid, "encoder": ctx.encoder, "n_tiles": n,
        "dim": rt.dim, "conditions_encoded": conds,
        "stimuli": [inp.manifest.by_slide[sid][c] for c in conds],
        "manifest_v2_path": str(inp.manifest.path), "manifest_v2_sha256": inp.manifest.sha256,
        "manifest_v1_sha256": inp.manifest.v1_sha256,
        "queries_path": str(inp.queries_path), "queries_sha256": inp.queries_sha256,
        "query_row": {k: row[k] for k in ("population_index", "metadata_index", "slide_id",
                                          "feature_rows", "resolved_path", "feature_sha256")},
        "coords_source": row["resolved_path"], "coords_source_sha256": row["feature_sha256"],
        "slide_path": str(slide_path), "wsi": {"st_size": stat0["st_size"],
                                               "st_mtime_ns": stat0["st_mtime_ns"]},
        "reinhard": {"target_means": list(PINNED_MEANS), "target_stds": list(PINNED_STDS),
                     "info": ctx.reinhard_info},
        "reinhard_nonfinite_tile_indices": sorted(nonfinite_reinhard),
        "fingerprint_verification": {
            "status": "all tiles x 13 pictures verified before transform",
            "n_tiles_verified": int(verified.sum()), "n_pictures_verified": int(verified.sum()) * 13,
            "mismatches": 0, "fp_npy_sha256": fdoc["array"]["npy_sha256"],
            "fp_content_sha256": fdoc["array"]["content_sha256"],
            "fp_json_sha256": X.sha256_file(fp_paths(root / "fingerprints", sid)[1])},
        "first_run": first_run,
        "first_run_record": first_run["status"] if plan_entry is not None else "not_applicable",
        "plan_sha256": ctx.plan_sha256,
        "encoder_meta": rt.meta, "amp": rt.amp, "forward_batch": ctx.forward_batch,
        "forward_batching": "per condition, consecutive forward_batch-tile chunks (as X)",
        "loader_batch": ctx.loader_batch, "num_workers": ctx.num_workers,
        "prefetch_factor": ctx.prefetch_factor, "device": ctx.device, "gpu": ctx.gpu,
        "cpu_request": cpu_request(), "environment": ctx.env,
        "hf_cache_disclosure": hf_cache_disclosure(True),
        "extractor_code_sha256": EXTRACTOR_SHA256, "suite_code_sha256": SUITE_CODE_SHA256,
        "legacy_gaussian_blur_label": "post-preprocessing (encoder-specific), historical "
                                      "Fig. 4A definition",
        "slide_wall_seconds": round(wall, 3), "encoder_forward_seconds": round(fwd_seconds, 3),
        "pickle_sha256": _sha(payload), "pickle_bytes": len(payload), "written_at": _now(),
    }
    try:
        publish(root, pkl, payload)
        publish(root, recp, _dump_pretty(record))
    except FileExistsError as exc:
        # [R6d] the refused content's digest is carried by the FAILED record
        return fail(stage="publish", reason=f"refused to overwrite: {exc}",
                    pickle_sha256=record["pickle_sha256"])
    print(f"[suite] extract {sid}: n={n} conds={conds} wall={wall:.1f}s", flush=True)
    return "written"


def extract_main(argv=None, *, encoder_factory: Optional[Callable] = None,
                 opener: Optional[Callable] = None,
                 slide_path_of: Callable[[str], Path] = paths.slide_wsi_path,
                 reinhard_factory: Optional[Callable] = None,
                 probe: Optional[EnvProbe] = None,
                 gpu_info: Callable[[str], Dict[str, Any]] = gpu_info_default) -> int:
    p = argparse.ArgumentParser(prog="extract")
    p.add_argument("--out-root", type=Path, required=True)
    add_input_args(p)
    add_env_args(p)
    p.add_argument("--encoder", required=True, choices=sorted(X.AMP_DEFAULTS))
    p.add_argument("--conditions", default="",
                   help="comma list (default all 13); exclusive with --plan")
    p.add_argument("--plan", type=Path, default=None, help="extension/PLAN.json")
    p.add_argument("--plan-sha256", default="")
    p.add_argument("--lo", type=int, default=1)
    p.add_argument("--hi", type=int, default=0)
    p.add_argument("--slides", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--forward-batch", type=int, default=256)
    p.add_argument("--loader-batch", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=7)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--no-hash-weights", action="store_true")
    a = p.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if a.plan is not None and a.conditions:
        raise SuiteError("--conditions and --plan are exclusive")
    if cpu_request() < a.num_workers + 1:
        raise SuiteError(f"CPU request {cpu_request()} < num_workers + 1 = {a.num_workers + 1}")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable (no silent CPU fallback)")
    inp = load_inputs(a)
    plan, plan_sha, plan_by = None, None, {}
    if a.plan is not None:
        raw = a.plan.read_bytes()
        plan_sha = _sha(raw)
        if not a.plan_sha256 or plan_sha != a.plan_sha256:
            raise SuiteError(f"plan sha256 {plan_sha} != --plan-sha256 {a.plan_sha256!r}")
        plan = json.loads(raw)
        if (plan.get("schema") != PLAN_SCHEMA or plan.get("encoder") != a.encoder
                or plan.get("queries_sha256") != inp.queries_sha256
                or plan.get("manifest_v1_sha256") != inp.manifest.v1_sha256):
            raise SuiteError("plan schema/encoder/queries sha/manifest v1 sha mismatch")
        plan_by = {s["slide_id"]: s for s in plan["slides"]}
    default_conds = _ordered(_split_list(a.conditions)) if a.conditions else list(SUITE_CONDITIONS)
    hi = a.hi or len(inp.rows)
    sel = [r["slide_id"] for r in inp.rows[max(1, a.lo) - 1:hi]]
    if a.slides:
        want = _split_list(a.slides)
        missing = [s for s in want if s not in sel]
        if missing:
            raise SuiteError(f"--slides not in range: {missing[:10]}")
        sel = [s for s in sel if s in set(want)]
    if plan is not None:
        missing = [s for s in sel if s not in plan_by]
        if missing:
            raise SuiteError(f"slides missing from the plan: {missing[:10]}")
    env, rh, rh_info = _prepare_render_env(
        a, probe, reinhard_factory, opener=opener,
        probe_wsi=_default_probe_wsi(a, inp.rows, slide_path_of))
    golden = blur_golden_check()
    if not golden["ok"]:
        raise SuiteError(f"blur golden mismatch {golden}")
    env["blur_golden"] = golden
    gpu = gpu_info(a.device)
    factory = encoder_factory or (lambda name, device, amp_flag: X.build_encoder(
        name, device, amp_flag, hash_weights=not a.no_hash_weights))
    rt = factory(a.encoder, a.device, X.AMP_DEFAULTS[a.encoder])
    ctx = ExtractContext(a.out_root, inp, a.encoder, rt, a.device, a.forward_batch,
                         a.loader_batch, a.num_workers, a.prefetch_factor, rh, rh_info,
                         opener or openslide_open, slide_path_of, env, gpu, plan, plan_sha)
    counts = {"written": 0, "skipped": 0, "failed": 0}
    t0 = time.time()
    for sid in sel:
        pe = plan_by.get(sid) if plan is not None else None
        conds = pe["conditions"] if pe is not None else default_conds
        counts[extract_slide(ctx, sid, conds, pe)] += 1
    summary = {"argv": list(argv) if argv is not None else sys.argv[1:], "slides": sel,
               "counts": counts, "wall_seconds": round(time.time() - t0, 2), "gpu": gpu,
               "environment": env, "plan_sha256": plan_sha, "encoder_meta": rt.meta,
               "hf_cache_disclosure": hf_cache_disclosure(True)}
    publish(a.out_root, a.out_root / "features" / "runs" / f"{_tag()}.json", _dump_pretty(summary))
    print(f"[suite] extract done {counts}", flush=True)
    return 1 if counts["failed"] else 0


# --------------------------------------------------------------------------
# Environment sub-commands
# --------------------------------------------------------------------------


def capture_env_main(argv=None, *, probe: Optional[EnvProbe] = None,
                     reinhard_factory: Optional[Callable] = None,
                     opener: Optional[Callable] = None) -> int:
    """[R4] capture: the mapped closure is read from /proc/self/maps after the import order,
    one pinned 1-thread Reinhard fit, one blur-golden evaluation and one level-0 512x512
    read_region of ``--probe-wsi`` (no output is produced from the probe pixels). Fails
    (SuiteError, nothing published) unless the asserted set covers REQUIRED_LIB_PATTERNS."""
    p = argparse.ArgumentParser(prog="capture-env")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--torchstain-prefix", type=Path, required=True)
    p.add_argument("--template", type=Path, default=X.DEFAULT_TEMPLATE)
    p.add_argument("--probe-wsi", type=Path, required=True,
                   help="[R4] WSI opened for one read_region before /proc/self/maps is read "
                        "(the lead passes the queries.csv row-0 WSI)")
    a = p.parse_args(argv)
    set_single_thread()

    def steps() -> dict:
        rec, _ = reference_lib_probe(template=a.template, torchstain_prefix=a.torchstain_prefix,
                                     reinhard_factory=reinhard_factory,
                                     opener=opener or openslide_open, probe_wsi=a.probe_wsi)
        return rec

    block = environment_block(probe or EnvProbe(), torchstain_prefix=a.torchstain_prefix,
                              reference_libs_path=None, enforce=False, require_threads=False,
                              probe_steps=steps)
    required = check_required_libs(block["asserted"]["reference_libs"]["observed"])
    doc = {"schema": ENV_CAPTURE_SCHEMA, "environment": block,
           "required_library_matches": required, "blur_golden": blur_golden_check()}
    host = block["recorded"]["hostname"]
    dest = a.out_root / "environment" / "captures" / f"{host}.{_tag()}.json"
    publish(a.out_root, dest, _dump_pretty(doc))
    print(json.dumps({"capture": str(dest), "deviations": block["deviations"],
                      "n_asserted_libraries": len(block["asserted"]["reference_libs"]["observed"])}),
          flush=True)
    return 0


def pin_reference_libs_main(argv=None) -> int:
    """[R4] pins the ASSERTED mapped sets (identical across captures required); the
    recorded-only sets (driver-pattern matches, non-ELF files [R6a], unhashed mappings, ldd
    closure) are stored
    per capture host and never compared."""
    p = argparse.ArgumentParser(prog="pin-reference-libs")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--captures", type=Path, nargs="+", required=True)
    p.add_argument("--expect-queues", default="", help="comma list that must all be captured")
    p.add_argument("--expect-driver-tags", default="", help="comma list that must all be captured")
    a = p.parse_args(argv)
    docs = [(c, json.loads(c.read_text())) for c in a.captures]
    problems = []
    for c, d in docs:
        if d.get("schema") != ENV_CAPTURE_SCHEMA:
            problems.append(f"{c}: schema {d.get('schema')!r} != {ENV_CAPTURE_SCHEMA}")
    if problems:
        print(json.dumps({"pinned": False, "problems": problems}, indent=1), flush=True)
        return 1
    asserted0 = docs[0][1]["environment"]["asserted"]["reference_libs"]["observed"]
    libs0 = _canon(asserted0)
    for c, d in docs:
        envb = d["environment"]
        obs = envb["asserted"]["reference_libs"]["observed"]
        if _canon(obs) != libs0:
            problems.append(f"{c}: asserted mapped reference-library set differs from {docs[0][0]}")
        try:
            check_required_libs(obs)
        except SuiteError as exc:
            problems.append(f"{c}: {exc}")
        if envb["deviations"]:
            problems.append(f"{c}: asserted-field deviations {envb['deviations']}")
        if not d.get("blur_golden", {}).get("ok"):
            problems.append(f"{c}: blur golden mismatch")
    queues = {d["environment"]["recorded"].get("lsf_queue") for _, d in docs}
    tags = {t for _, d in docs for t in (d["environment"]["recorded"].get("gpu_driver_tags") or [])}
    for q in _split_list(a.expect_queues):
        if q not in queues:
            problems.append(f"no capture from queue {q}")
    for t in _split_list(a.expect_driver_tags):
        if t not in tags:
            problems.append(f"no capture with driver tag {t}")
    if problems:
        print(json.dumps({"pinned": False, "problems": problems}, indent=1), flush=True)
        return 1
    caps = [{"file": str(c), "sha256": X.sha256_file(c),
             "host": d["environment"]["recorded"]["hostname"],
             "lsf_queue": d["environment"]["recorded"].get("lsf_queue"),
             "gpu_driver_tags": d["environment"]["recorded"].get("gpu_driver_tags")}
            for c, d in docs]
    doc = {"schema": REFLIBS_SCHEMA, "method": "proc_self_maps", "soname_source": SONAME_SOURCE,
           "driver_patterns": list(DRIVER_LIB_PATTERNS),
           "required_patterns": list(REQUIRED_LIB_PATTERNS),
           "asserted": json.loads(libs0),
           "required_library_matches": required_lib_matches(asserted0),
           "recorded_only_by_capture": [
               dict(cap, recorded_only=d["environment"]["recorded"].get(
                   "reference_libs_recorded_only"))
               for cap, (_, d) in zip(caps, docs)],
           "captures": caps}
    dest = a.out_root / "environment" / "reference_libs.json"
    st = publish_idempotent(a.out_root, dest, _dump_pretty(doc))
    print(json.dumps({"pinned": True, "file": str(dest), "status": st,
                      "n_asserted_libraries": len(doc["asserted"])}), flush=True)
    return 0


def env_check_main(argv=None, *, probe: Optional[EnvProbe] = None,
                   reinhard_factory: Optional[Callable] = None,
                   opener: Optional[Callable] = None) -> int:
    p = argparse.ArgumentParser(prog="env-check")
    p.add_argument("--out-root", type=Path, required=True)
    add_env_args(p)
    a = p.parse_args(argv)
    doc: Dict[str, Any] = {"schema": "r2_1_perturb_suite_env_check_v1", "time": _now()}
    try:
        if a.probe_wsi is None:
            p.error("--probe-wsi is required for env-check [R4]")
        env, _, _ = _prepare_render_env(a, probe, reinhard_factory, opener=opener,
                                        probe_wsi=Path(a.probe_wsi))
        doc.update(ok=True, environment=env)
    except (EnvironmentMismatch, PinnedFitError) as exc:
        doc.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    doc["blur_golden"] = blur_golden_check()
    doc["ok"] = doc["ok"] and doc["blur_golden"]["ok"]
    publish(a.out_root, a.out_root / "environment" / "checks" / f"{socket.gethostname()}.{_tag()}.json",
            _dump_pretty(doc))
    print(json.dumps({"ok": doc["ok"], "error": doc.get("error")}), flush=True)
    return 0 if doc["ok"] else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

COMMANDS: Dict[str, Callable[..., int]] = {
    "gen-manifest-v2": gen_manifest_v2,
    "capture-env": capture_env_main,
    "pin-reference-libs": pin_reference_libs_main,
    "env-check": env_check_main,
    "capture-host-tags": capture_host_tags_main,
    "plan-tasks": plan_tasks_main,
    "fingerprint": fingerprint_main,
    "finalize": finalize_main,
    "finalize-features": finalize_features_main,
    "quarantine-orphans": quarantine_orphans_main,
    "reference-png": reference_png_main,
    "select-pilot": select_pilot_main,
    "pilot-render": pilot_render_main,
    "pilot-process-slide": pilot_process_slide_main,
    "plan-extension": plan_extension_main,
    "extract": extract_main,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("commands: " + ", ".join(COMMANDS))
        return 2
    try:
        return COMMANDS[argv[0]](argv[1:])
    except SuiteError as exc:
        print(f"[suite] ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
