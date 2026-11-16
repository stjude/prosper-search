"""R2.1 Stage V1: Virchow2 fp32 main-bank regeneration (identity tiles only).

Implements the three-block exact specification in
``[internal record]`` (candidate 001 V1.1–V1.9,
candidate 002 A-F1…A-F12, candidate 003 B-N1…B-N9) for the PROSPER revision
workspace.  The frozen perturbation extractor ``X``
(``r2_1_perturb_extract_cli.py``, sha256 pinned below) is imported for the
encoder construction, the forward helper, the OpenSlide opener, the worker
initialiser and the no-replace publish protocol so that the identity-tile
feature path is byte-equivalent to X's identity arm.  ``load_bank_coords`` and
``SlideTiles`` from X are deliberately NOT imported (A-F10): coordinates are
read from the inventory-bound old Virchow2 pickle and cross-checked against the
H-Optimus bank for the same slide.

Subcommands
-----------
extract           production / pilot extractor (one contiguous inventory chunk)
make-params       optimal contiguous K-way partition of the virchow2 rows
extend-allowlist  admit observed GPU tuples from passing pilot records
calibrate         step 4.0 CPU comparison published fp32 vs X first-run identity
fidelity          step 4.1/4.2 classification of fp32 / fp16 historical slides
check-mosaic      ordered mosaic wsi_loc == published pickle coords (A-F8/B-N8)
seed-allowlist    write the seed GPU tuple allowlist to a NEW path
make-code-manifest  record the sha256 of every loaded module under a code root

CUDA is never initialised at import time; ``extract`` resolves the device,
observes the GPU tuple and builds the encoder only after the start-up
refusals, through injectable seams used by the synthetic tests.

``extract`` requires ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1`` to be
exported before Python starts (``import prosper`` imports huggingface_hub,
which freezes its offline flag at import); it refuses otherwise and asserts
the live library flag before and after building the encoder.  No path
component below the run root may be a symlink.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import socket
import stat as stat_mod
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import paths
from . import r2_1_perturb_extract_cli as X

# --------------------------------------------------------------------------
# Frozen constants
# --------------------------------------------------------------------------

SCHEMA = "r2_1_virchow2_fp32_record_v1"
TASK = "r2_1_virchow2_fp32_main_bank"
ENCODER = "virchow2_fp32"
BANK = "virchow2"
HOPTIMUS_BANK = "h_optimus_1"
WIDTH = 1280
MODEL_DIM = 2560
PATCH = 512
FORWARD_BATCH = 256
LOADER_BATCH = 8
PREFETCH_FACTOR = 2
PICKLE_PROTOCOL = 4

X_SHA256_PINNED = "14eb457ebfe2243a924bb7928483538985120b6987de51eb4a6cc6ccf3a8d0bb"
HF_REPO_ID = "paige-ai/Virchow2"
HF_SNAPSHOT_REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
WEIGHTS_SHA256 = "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
WEIGHTS_BYTES = 2525001112
MOSAIC_SHA256 = "13fdd889e91dcce88a7b5d94b81a8599ef54082dba520a69238d2efa6cfa98e7"
MOSAIC_BYTES = 63781420

RUN_PARENT_NAME = "r2_1_virchow2_fp32_20261006"     # $PROSPER_OUT/<this>/run_001 (V1.4, M1)
RUN_ROOT_NAME = "run_001"
CODE_FROZEN_NAME = "code_frozen"                    # run_001/code_frozen (V1.7/A-F4(1), L3)
PRODUCTION_DEVICES = ("cuda", "cuda:0")             # M2
RELATIVE_FILE_MODULES = ("torch.ops", "torch.classes")  # L8: only these may have a relative __file__

# A-F2 step 4.0: the 16 stored-fp32 slides X's first run already extracted (41,714 tiles).
CALIBRATION_SLIDES: Tuple[str, ...] = (
    "10079", "12543", "14490", "14613", "14628", "15358", "15414", "16839",
    "17125", "17141", "17147", "17148", "19115", "19131", "7419", "9934",
)
CALIBRATION_TILES = 41714
CALIBRATE_SCHEMA = SCHEMA + ".calibrate"
X_RECORD_SCHEMA = "r2_1_perturb_record_v1"

FORBIDDEN_ENV = ("NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
HF_OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
FORBIDDEN_DIR_NAMES = ("comet-virchow2", "comet-robustness")

TUPLE_KEYS = ("torch_device_name", "nvidia_smi_name", "nvidia_smi_memory_total",
              "nvidia_smi_driver_version", "torch_cuda", "torch_cudnn")
SEED_TUPLE = {
    "torch_device_name": "NVIDIA A100-SXM4-80GB",
    "nvidia_smi_name": "NVIDIA A100-SXM4-80GB",
    "nvidia_smi_memory_total": "81920 MiB",
    "nvidia_smi_driver_version": "595.58.03",
    "torch_cuda": "12.4",
    "torch_cudnn": 90100,
}
NVIDIA_SMI_QUERY = ["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]

RC_OK, RC_SLIDE_FAILURES, RC_REFUSED = 0, 1, 2

# Frozen record key list (declaration §3.1 + candidate 002 additional fields +
# the independent enumerated additions).  Tests assert every published record has
# exactly this key set.
RECORD_KEYS: Tuple[str, ...] = (
    "schema", "task", "encoder", "bank", "slide_id", "n_tiles", "dim", "width", "dtype",
    "c_contiguous", "amp", "forward_batch", "loader_batch", "num_workers", "prefetch_factor",
    "device", "pickle_bytes", "pickle_sha256", "pickle_protocol",
    "inventory_path", "inventory_sha256", "inventory_row_index",
    "coords_source", "coords_source_sha256", "hoptimus_coords_source", "old_bank_dtype",
    "slide_path", "wsi", "encoder_meta", "weights_sha256", "gpu",
    "allowlist_mode", "allowlist_path", "allowlist_sha256", "observed_tuple", "matched_tuple",
    "host", "platform", "python", "versions", "torch_backend_flags", "threads", "hf_offline_env",
    "git", "code_root", "code_path", "code_sha256", "extractor_code_path", "extractor_code_sha256",
    "code_manifest_path", "code_manifest_sha256", "code_manifest_excluded_modules",
    "prosper_file", "sys_path_head",
    "lsf_job", "lsf_index", "lsf_queue", "lsf_djob_numproc", "argv", "run_root", "out_root",
    "run_started_utc", "slide_started_utc", "slide_ended_utc", "slide_wall_seconds",
    "encoder_forward_seconds", "images_per_second_forward", "tiles_per_second_slide",
    "n_forward_batches", "written_at",
)

# Record fields compared by the skip predicate (A-F5 + B-N1).
SKIP_FIELDS: Tuple[str, ...] = (
    "slide_id", "encoder", "n_tiles", "pickle_bytes", "pickle_sha256", "inventory_sha256",
    "coords_source", "hoptimus_coords_source", "code_sha256", "extractor_code_sha256",
    "code_manifest_sha256", "weights_sha256", "width", "dtype", "gpu_env_null",
    "allowlist_mode", "matched_tuple", "device",
)

ULP_BINS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5-7", 5, 7),
    ("8-15", 8, 15), ("16-31", 16, 31), ("32-63", 32, 63), ("64-127", 64, 127),
    ("128-255", 128, 255), (">=256", 256, None),
)


class Refusal(RuntimeError):
    """Start-up refusal (exit code 2); carries a short machine-readable stage."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message


class SlideFailure(RuntimeError):
    def __init__(self, stage: str, message: str, **extra):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message
        self.extra = extra


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


sha256_file = X.sha256_file


def read_checked(path: Path, expected_sha256: str, expected_size: Optional[int] = None,
                 what: str = "file") -> bytes:
    """Return the bytes of ``path`` only after size and sha256 match."""
    p = Path(path)
    if not p.is_file():
        raise Refusal(what, f"missing file {p}")
    st = p.stat()
    if expected_size is not None and st.st_size != int(expected_size):
        raise Refusal(what, f"size mismatch for {p}: {st.st_size} != {expected_size}")
    data = p.read_bytes()
    got = sha256_bytes(data)
    if got != expected_sha256:
        raise Refusal(what, f"sha256 mismatch for {p}: {got} != {expected_sha256}")
    return data


def load_json_checked(path: Path, expected_sha256: str, what: str) -> Tuple[Any, str]:
    data = read_checked(path, expected_sha256, None, what)
    return json.loads(data.decode("utf-8")), expected_sha256


def is_under(child: Path, parent: Path) -> bool:
    try:
        Path(child).relative_to(Path(parent))
        return True
    except ValueError:
        return False


def realpath(p) -> Path:
    return Path(os.path.realpath(str(p)))


def lsf_components() -> Tuple[str, str]:
    """``(<LSB_JOBID>, <LSB_JOBINDEX>)`` or ``(local.<pid>, local.<pid>)`` (B-N4)."""
    job = os.environ.get("LSB_JOBID")
    idx = os.environ.get("LSB_JOBINDEX")
    if job and idx:
        return job, idx
    tag = f"local.{os.getpid()}"
    return tag, tag


def failed_path(failed_dir: Path, slide_id: str) -> Path:
    job, idx = lsf_components()
    return Path(failed_dir) / f"{slide_id}.{job}.{idx}.json"


def write_failed(failed_dir: Path, slide_id: str, obj: dict) -> Path:
    """No-replace FAILED write.  On a name collision a ``.<time_ns>`` component is
    inserted before ``.json`` so that neither the old nor the new evidence is lost."""
    failed_dir = Path(failed_dir)
    failed_dir.mkdir(parents=True, exist_ok=True)
    dest = failed_path(failed_dir, slide_id)
    data = (json.dumps(obj, sort_keys=True, indent=2, default=str) + "\n").encode()
    try:
        X._publish_bytes(dest, data)
        return dest
    except FileExistsError:
        alt = dest.with_name(dest.name[:-len(".json")] + f".{time.time_ns()}.json")
        X._publish_bytes(alt, data)
        print(f"[warn] FAILED file {dest} already existed; wrote {alt}", file=sys.stderr)
        return alt


def validate_tuple(t: Any, where: str) -> Dict[str, Any]:
    if not isinstance(t, dict) or set(t.keys()) != set(TUPLE_KEYS):
        raise Refusal("gpu_tuple", f"{where}: tuple keys must be exactly {list(TUPLE_KEYS)}")
    for k in TUPLE_KEYS[:5]:
        if not isinstance(t[k], str) or not t[k]:
            raise Refusal("gpu_tuple", f"{where}: field {k} must be a non-empty string")
    if not isinstance(t["torch_cudnn"], int) or isinstance(t["torch_cudnn"], bool):
        raise Refusal("gpu_tuple", f"{where}: torch_cudnn must be an int")
    return {k: t[k] for k in TUPLE_KEYS}


def validate_allowlist(obj: Any, where: str) -> List[Dict[str, Any]]:
    if not isinstance(obj, list) or not obj:
        raise Refusal("gpu_tuple", f"{where}: allowlist must be a non-empty JSON list")
    out = [validate_tuple(t, f"{where}[{i}]") for i, t in enumerate(obj)]
    seen = set()
    for t in out:
        key = tuple(t[k] for k in TUPLE_KEYS)
        if key in seen:
            raise Refusal("gpu_tuple", f"{where}: duplicate tuple {key}")
        seen.add(key)
    return out


def tuple_in(t: Dict[str, Any], allowlist: Sequence[Dict[str, Any]]) -> bool:
    return any(all(t.get(k) == a[k] for k in TUPLE_KEYS) for a in allowlist)


def normalise_device(device: str) -> str:
    """``"cpu"`` | ``"cuda:<index>"``; ``"cuda"`` is torch's default device 0 (M2)."""
    d = str(device)
    if d == "cpu":
        return "cpu"
    if d == "cuda":
        return "cuda:0"
    if d.startswith("cuda:") and d[5:].isdigit():
        return f"cuda:{int(d[5:])}"
    raise Refusal("args", f"unsupported --device {device!r} (cpu | cuda | cuda:<index>)")


def nvidia_smi_selector(index: int) -> str:
    """``nvidia-smi -i`` selector for torch device ``cuda:<index>``.  nvidia-smi
    ignores ``CUDA_VISIBLE_DEVICES``, so the torch index is mapped through it
    (entry may be a physical index or a GPU UUID; an unsupported form makes
    nvidia-smi fail and the observer refuse)."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is None:
        return str(index)
    entries = [e.strip() for e in vis.split(",")]
    if not vis.strip() or index >= len(entries) or not entries[index]:
        raise Refusal("gpu_tuple", f"cuda:{index} not resolvable through CUDA_VISIBLE_DEVICES={vis!r}")
    return entries[index]


def observe_gpu_tuple(device: str) -> Tuple[Dict[str, Any], str]:
    """Default tuple observer (R1): torch device name + nvidia-smi raw fields,
    both for the actual device index (M2)."""
    import torch
    norm = normalise_device(device)
    if not norm.startswith("cuda:"):
        raise Refusal("gpu_tuple", f"GPU tuple observation needs a CUDA device, got {device!r}")
    index = int(norm.split(":", 1)[1])
    cmd = ["nvidia-smi"] + NVIDIA_SMI_QUERY + ["-i", nvidia_smi_selector(index)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise Refusal("gpu_tuple", f"nvidia-smi failed rc={proc.returncode}: {proc.stderr.strip()}")
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) != 1:
        raise Refusal("gpu_tuple", f"nvidia-smi -i returned {len(lines)} GPU lines, expected 1: {lines!r}")
    fields = [f.strip() for f in lines[0].split(",")]
    if len(fields) != 3:
        raise Refusal("gpu_tuple", f"unexpected nvidia-smi output {lines!r}")
    name, mem, drv = fields
    t = {"torch_device_name": torch.cuda.get_device_name(index), "nvidia_smi_name": name,
         "nvidia_smi_memory_total": mem, "nvidia_smi_driver_version": drv,
         "torch_cuda": str(torch.version.cuda), "torch_cudnn": int(torch.backends.cudnn.version())}
    return t, proc.stdout


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------


def load_inventory(path: Path, expected_sha256: str) -> Tuple[List[dict], List[dict]]:
    """Return ``(virchow2_rows_in_file_order, all_rows)`` after sha verification.

    File order of the virchow2 rows is asserted equal to lexicographic
    ``slide_id`` order (V1.3 / make-params specification)."""
    data = read_checked(path, expected_sha256, None, "inventory")
    rows = list(csv.DictReader(data.decode("utf-8").splitlines()))
    need = {"bank", "slide_id", "canonical_tiles", "feature_rows", "dtype", "feature_path",
            "resolved_path", "size", "sha256"}
    if not rows or not need.issubset(rows[0].keys()):
        raise Refusal("inventory", f"inventory columns {sorted(rows[0].keys()) if rows else []} "
                                   f"lack {sorted(need)}")
    v2 = [r for r in rows if r["bank"] == BANK]
    if not v2:
        raise Refusal("inventory", "no virchow2 rows")
    ids = [r["slide_id"] for r in v2]
    if ids != sorted(ids):
        raise Refusal("inventory", "virchow2 rows are not in lexicographic slide_id order")
    if len(set(ids)) != len(ids):
        raise Refusal("inventory", "duplicate virchow2 slide_id")
    for i, r in enumerate(v2):
        r["_row_index"] = i + 1
    return v2, rows


def hoptimus_row(all_rows: List[dict], slide_id: str) -> dict:
    hits = [r for r in all_rows if r["bank"] == HOPTIMUS_BANK and r["slide_id"] == slide_id]
    if len(hits) != 1:
        raise SlideFailure("coords", f"expected exactly one {HOPTIMUS_BANK} row for {slide_id}, "
                                     f"found {len(hits)}")
    return hits[0]


def _validate_coords(coords: Any, n_expected: int, what: str) -> None:
    if not isinstance(coords, list):
        raise SlideFailure("coords", f"{what}: coords is {type(coords).__name__}, not list")
    if len(coords) != n_expected:
        raise SlideFailure("coords", f"{what}: len(coords)={len(coords)} != canonical_tiles={n_expected}")
    for i, c in enumerate(coords):
        if not isinstance(c, dict) or "wsi_loc" not in c:
            raise SlideFailure("coords", f"{what}: coords[{i}] lacks wsi_loc")
        loc = c["wsi_loc"]
        if len(loc) != 2:
            raise SlideFailure("coords", f"{what}: coords[{i}].wsi_loc has {len(loc)} entries")
        for v in loc:
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 0:
                raise SlideFailure("coords", f"{what}: coords[{i}].wsi_loc non-integer/negative {loc!r}")


def read_bank_pickle(row: dict, what: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """Read an inventory-bound bank pickle only after size+sha match (V1.3).

    Returns ``(feats, coords, source_dict)``; ``coords`` is the exact list
    object read (no numpy retyping)."""
    p = Path(row["resolved_path"])
    try:
        data = read_checked(p, row["sha256"], int(row["size"]), what)
    except Refusal as exc:
        raise SlideFailure(what, exc.message) from exc
    obj = pickle.loads(data)
    if not isinstance(obj, tuple) or len(obj) != 2:
        raise SlideFailure(what, f"{p}: pickle is not a (feats, coords) pair")
    feats, coords = obj
    n = int(row["canonical_tiles"])
    _validate_coords(coords, n, what)
    if not hasattr(feats, "shape") or feats.shape[0] != n:
        raise SlideFailure(what, f"{p}: feats.shape[0] != canonical_tiles ({n})")
    src = {"path": str(p), "size": int(row["size"]), "sha256": row["sha256"]}
    return feats, coords, src


def locs_of(coords: list) -> List[Tuple[int, int]]:
    return [(int(c["wsi_loc"][0]), int(c["wsi_loc"][1])) for c in coords]


# --------------------------------------------------------------------------
# Code manifest (A-F4)
# --------------------------------------------------------------------------


NON_FILE_MODULE_EXCLUSIONS: List[str] = []


def loaded_code_modules(code_root: Path) -> Dict[str, str]:
    """``{relative_posix_path: sha256}`` of every loaded module whose ``__file__``
    resolves under ``code_root``.  A loaded module whose file is not a ``.py``
    source (e.g. a sourceless ``.pyc``) is a refusal.

    Only ``torch.ops`` and ``torch.classes`` (torch binds them to the relative
    names ``"_ops.py"`` / ``"_classes.py"``) may have a relative ``__file__``;
    they are not filesystem-bound and are excluded, their names appended to
    ``NON_FILE_MODULE_EXCLUSIONS`` so the record can list them.  Any other
    module with a relative ``__file__`` is a refusal (L8)."""
    root = realpath(code_root)
    out: Dict[str, str] = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        if not os.path.isabs(f):
            if name not in RELATIVE_FILE_MODULES:
                raise Refusal("code_manifest", f"module {name} has a relative __file__ {f!r}; only "
                                               f"{list(RELATIVE_FILE_MODULES)} are exempt")
            if name not in NON_FILE_MODULE_EXCLUSIONS:
                NON_FILE_MODULE_EXCLUSIONS.append(name)
            continue
        p = realpath(f)
        if not is_under(p, root):
            continue
        if p.suffix != ".py":
            raise Refusal("code_manifest", f"module {name} loaded from non-source file {p}")
        if not p.is_file():
            raise Refusal("code_manifest", f"module {name} __file__ {p} does not exist")
        out[p.relative_to(root).as_posix()] = sha256_file(p)
    return out


def check_code_manifest(manifest: Any, code_root: Path) -> Dict[str, str]:
    if not isinstance(manifest, dict) or not manifest:
        raise Refusal("code_manifest", "code manifest must be a non-empty JSON object path->sha256")
    loaded = loaded_code_modules(code_root)
    extra = sorted(set(loaded) - set(manifest))
    missing = sorted(set(manifest) - set(loaded))
    mism = sorted(k for k in set(loaded) & set(manifest) if loaded[k] != manifest[k])
    if extra or missing or mism:
        raise Refusal("code_manifest", f"loaded modules differ from manifest: extra={extra} "
                                       f"missing={missing} mismatch={mism}")
    return loaded


# --------------------------------------------------------------------------
# Identity-tile dataset (byte-equivalent to X's identity arm)
# --------------------------------------------------------------------------


class IdentityTiles:
    """Dataset yielding ``{"index", "tensor"}`` or ``{"index", "error"}``.

    Mirrors X.SlideTiles' identity path exactly:
    ``read_region((x, y), 0, (512, 512)).convert("RGB")`` -> ``rt.transform``."""

    def __init__(self, slide_path: str, locs: Sequence[Tuple[int, int]], transform: Callable,
                 opener: Callable, patch_size: int = PATCH):
        self.slide_path = str(slide_path)
        self.locs = list(locs)
        self.transform = transform
        self.opener = opener
        self.patch_size = patch_size
        self._slide = None

    def __len__(self) -> int:
        return len(self.locs)

    def __getitem__(self, i: int) -> dict:
        x, y = self.locs[i]
        ps = self.patch_size
        try:
            if self._slide is None:
                self._slide = self.opener(self.slide_path)
            region = self._slide.read_region((x, y), 0, (ps, ps)).convert("RGB")
            tensor = self.transform(region)
        except Exception as exc:                          # noqa: BLE001
            return {"index": i, "error": {"type": type(exc).__name__, "message": str(exc),
                                          "wsi_loc": [x, y]}}
        return {"index": i, "tensor": tensor}


def encode_slide(rt: "X.EncoderRuntime", ds: IdentityTiles, *, device: str, num_workers: int,
                 loader_batch: int, forward_batch: int) -> Tuple[np.ndarray, int, float]:
    """Encode all tiles; returns ``(feats (n, WIDTH) float32 C-contiguous,
    n_forward_batches, forward_seconds)``.  Row ``i`` is placed at coords ``i``
    regardless of worker completion order."""
    import torch
    n = len(ds)
    out = np.empty((n, WIDTH), dtype=np.float32, order="C")
    filled = np.zeros(n, dtype=bool)
    errors: List[dict] = []
    loader = torch.utils.data.DataLoader(
        ds, batch_size=loader_batch, shuffle=False, num_workers=num_workers,
        collate_fn=X._collate, worker_init_fn=X._worker_init,
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
        persistent_workers=False)
    pending: List[dict] = []
    n_batches = 0
    fwd_seconds = 0.0

    def flush(items: List[dict]) -> None:
        nonlocal n_batches, fwd_seconds
        if not items:
            return
        idx = np.asarray([it["index"] for it in items], dtype=np.int64)
        xb = torch.stack([it["tensor"] for it in items])
        t0 = time.perf_counter()
        feats = X._forward(rt, xb, device)
        fwd_seconds += time.perf_counter() - t0
        n_batches += 1
        if feats.shape != (len(items), rt.dim):
            raise SlideFailure("encode", f"forward returned {feats.shape}, expected {(len(items), rt.dim)}")
        out[idx] = feats[:, :WIDTH]
        filled[idx] = True

    for batch in loader:
        for item in batch:
            if "error" in item:
                errors.append(item)
                continue
            pending.append(item)
        # Encode in exact forward_batch chunks in mosaic order; the loader is
        # sequential (shuffle=False), so pending is already ordered.
        while len(pending) >= forward_batch:
            flush(pending[:forward_batch])
            pending = pending[forward_batch:]
    flush(pending)
    if errors:
        raise SlideFailure("tile_read", f"{len(errors)} tile read/transform errors",
                           tile_errors=errors[:20], n_tile_errors=len(errors))
    if not filled.all():
        raise SlideFailure("encode", f"{int((~filled).sum())} rows never assembled")
    if not np.isfinite(out).all():
        raise SlideFailure("encode", "non-finite values in features")
    if not out.flags["C_CONTIGUOUS"] or out.dtype != np.float32 or out.shape != (n, WIDTH):
        raise SlideFailure("encode", "feature array not (n, WIDTH) float32 C-contiguous")
    return out, n_batches, fwd_seconds


# --------------------------------------------------------------------------
# Start-up checks (B-N9 order)
# --------------------------------------------------------------------------


def check_env() -> Dict[str, Any]:
    for k in FORBIDDEN_ENV:
        if k in os.environ:
            raise Refusal("env", f"{k} must not be present in the environment (B-N3)")
    for k in THREAD_ENV:
        if os.environ.get(k) != "1":
            raise Refusal("env", f"{k} must be set to '1', found {os.environ.get(k)!r}")
    if not os.environ.get("PROSPER_OUT"):
        raise Refusal("env", "PROSPER_OUT must be set (forbidden-root and run-root checks need it)")
    # L4/NM1: huggingface_hub (imported by ``import prosper``) freezes its offline
    # flag at import, so setting the variables here would be too late.  They must
    # be inherited as "1"; the CLI never sets them.
    for k in HF_OFFLINE_ENV:
        if os.environ.get(k) != "1":
            raise Refusal("env", f"{k} must be exported as '1' before the process starts, "
                                 f"found {os.environ.get(k)!r}")
    env = {k: os.environ.get(k) for k in HF_OFFLINE_ENV}
    env["HF_HOME"] = os.environ.get("HF_HOME")
    env["library_offline_at_start"] = check_hf_offline_effective()
    return env


def hf_offline_library_state() -> Dict[str, Optional[bool]]:
    """Observed (not intended) offline flags of the HF libraries in this process.
    ``transformers_offline_mode`` is None when transformers is not imported."""
    import huggingface_hub.constants as hfc
    tf = sys.modules.get("transformers.utils.hub")
    return {"huggingface_hub_constants_HF_HUB_OFFLINE": hfc.HF_HUB_OFFLINE,
            "transformers_offline_mode": tf.is_offline_mode() if tf is not None else None}


def check_hf_offline_effective() -> Dict[str, Optional[bool]]:
    """NM1: refuse unless the live library flags say offline."""
    st = hf_offline_library_state()
    if st["huggingface_hub_constants_HF_HUB_OFFLINE"] is not True:
        raise Refusal("hf_offline", "huggingface_hub.constants.HF_HUB_OFFLINE is "
                                    f"{st['huggingface_hub_constants_HF_HUB_OFFLINE']!r}; export "
                                    "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 before starting Python")
    if st["transformers_offline_mode"] not in (None, True):
        raise Refusal("hf_offline", f"transformers offline mode is {st['transformers_offline_mode']!r}")
    return st


def check_prosper_file(code_root: Path) -> Path:
    import prosper
    pf = realpath(prosper.__file__)
    root = realpath(code_root)
    if not is_under(pf, root):
        raise Refusal("code_root", f"prosper.__file__ {pf} is not under code root {root}")
    return pf


def forbidden_parents(v2_rows: List[dict], all_rows: List[dict]) -> List[Path]:
    pout = os.environ.get("PROSPER_OUT")
    if not pout:
        raise Refusal("env", "PROSPER_OUT must be set")
    roots: List[Path] = [realpath(Path(pout) / "features")]
    for r in all_rows:
        roots.append(realpath(Path(r["resolved_path"]).parent))
    return roots


def check_out_root(out_root: Path, run_root: Path, pilot: bool, v2_rows: List[dict],
                   all_rows: List[dict]) -> Tuple[Path, Path]:
    """M1 (design decision (a)): run root is exactly
    ``realpath($PROSPER_OUT)/r2_1_virchow2_fp32_20261006/run_001``; production
    writes at the run root itself (features at ``run_001/features/virchow2_fp32``);
    a pilot writes at a fresh direct child of ``run_001/pilot``; no production
    out root may contain a ``pilot`` path component."""
    ro = realpath(out_root)
    rr = realpath(run_root)
    if any(part in FORBIDDEN_DIR_NAMES for part in ro.parts):
        raise Refusal("out_root", f"{ro} has a forbidden path component {FORBIDDEN_DIR_NAMES}")
    for fp in forbidden_parents(v2_rows, all_rows):
        if is_under(ro, fp):
            raise Refusal("out_root", f"{ro} is under forbidden parent {fp}")
    want_parent = realpath(Path(os.environ["PROSPER_OUT"])) / RUN_PARENT_NAME
    if rr.name != RUN_ROOT_NAME or rr.parent != want_parent:
        raise Refusal("out_root", f"run root {rr} must be {want_parent / RUN_ROOT_NAME}")
    pilot_root = rr / "pilot"
    if pilot:
        if ro.parent != pilot_root:
            raise Refusal("out_root", f"pilot out root {ro} must be a direct child of {pilot_root}")
        if ro.exists():
            raise Refusal("out_root", f"pilot out root {ro} already exists (single-shot pilot)")
    else:
        if "pilot" in ro.parts:
            raise Refusal("out_root", f"production out root {ro} has a 'pilot' path component")
        if ro != rr:
            raise Refusal("out_root", f"production out root {ro} must equal the run root {rr}")
    return ro, rr


def check_code_frozen(code_root: Path, run_root: Path) -> None:
    """L3 (V1.7/A-F4(1)): production code runs from ``run_001/code_frozen``."""
    want = realpath(run_root) / CODE_FROZEN_NAME
    if realpath(code_root) != want:
        raise Refusal("code_root", f"production --code-root {realpath(code_root)} must be {want}")


def check_no_symlink_below(root: Path, target: Path, stage: str = "symlink") -> None:
    """NL1: no existing path component of ``target`` strictly below the run
    root ``root`` may be a symlink (``os.lstat``), so a pre-existing link such
    as ``run_001/features -> elsewhere`` cannot redirect a CLI write or read."""
    root, target = Path(root), Path(target)
    try:
        rel = target.relative_to(root)
    except ValueError:
        raise Refusal(stage, f"{target} is not under the run root {root}") from None
    cur = root
    for part in rel.parts:
        cur = cur / part
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            return
        if stat_mod.S_ISLNK(st.st_mode):
            raise Refusal(stage, f"{cur} is a symlink; no path below the run root may be a symlink")


def output_dirs(out_root: Path) -> Tuple[Path, ...]:
    return (Path(out_root), Path(out_root) / "features" / ENCODER, Path(out_root) / "FAILED",
            Path(out_root) / "runs")


def check_weights(rt: "X.EncoderRuntime", expected_sha256: str, expected_revision: Optional[str]) -> dict:
    w = (rt.meta or {}).get("weights")
    if not isinstance(w, dict):
        raise Refusal("weights", f"encoder weights info missing/None: {w!r}")
    if "error" in w:
        raise Refusal("weights", f"encoder weights info error: {w['error']}")
    if w.get("repo_id") != HF_REPO_ID:
        raise Refusal("weights", f"repo_id {w.get('repo_id')!r} != {HF_REPO_ID!r}")
    if not w.get("sha256"):
        raise Refusal("weights", "encoder weights sha256 missing (None)")
    if w["sha256"] != expected_sha256:
        raise Refusal("weights", f"weights sha256 {w['sha256']} != expected {expected_sha256}")
    if w.get("bytes") is not None and w["bytes"] != WEIGHTS_BYTES:
        raise Refusal("weights", f"weights bytes {w['bytes']!r} != pinned {WEIGHTS_BYTES}")
    if not expected_revision or w.get("snapshot_revision") != expected_revision:
        raise Refusal("weights", f"snapshot_revision {w.get('snapshot_revision')!r} != {expected_revision!r}")
    if rt.dim != MODEL_DIM:
        raise Refusal("weights", f"encoder dim {rt.dim} != {MODEL_DIM}")
    if rt.amp:
        raise Refusal("weights", "encoder runtime amp must be False")
    if rt.name != BANK:
        raise Refusal("weights", f"encoder name {rt.name!r} != {BANK!r}")
    return w


# --------------------------------------------------------------------------
# Skip predicate (A-F5 + B-N1)
# --------------------------------------------------------------------------


def output_paths(out_root: Path, slide_id: str) -> Tuple[Path, Path, Path]:
    feat = Path(out_root) / "features" / ENCODER
    return feat / f"{slide_id}.pkl", feat / f"{slide_id}.record.json", Path(out_root) / "FAILED"


def expected_skip_view(ctx: "Ctx", row: dict, hop_src: Dict[str, Any], coords_src: Dict[str, Any]) -> dict:
    return {
        "slide_id": row["slide_id"], "encoder": ENCODER, "n_tiles": int(row["canonical_tiles"]),
        "inventory_sha256": ctx.inventory_sha256, "coords_source": coords_src,
        "hoptimus_coords_source": hop_src, "code_sha256": ctx.code_sha256,
        "extractor_code_sha256": ctx.extractor_code_sha256,
        "code_manifest_sha256": ctx.code_manifest_sha256, "weights_sha256": ctx.weights_sha256,
        "width": WIDTH, "dtype": "float32", "allowlist_mode": "production", "device": ctx.device_norm,
    }


def existing_status(pkl: Path, rec: Path, expected: dict, allowlist: Sequence[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """``("absent", None)``, ``("complete", None)`` or ``("conflict", reason)``."""
    pe, re_ = pkl.exists(), rec.exists()
    if not pe and not re_:
        return "absent", None
    if pe != re_:
        return "conflict", f"orphan: pickle_exists={pe} record_exists={re_}"
    try:
        r = json.loads(rec.read_text())
    except Exception as exc:                              # noqa: BLE001
        return "conflict", f"unreadable record: {exc}"
    for k, v in expected.items():
        if r.get(k) != v:
            return "conflict", f"record field {k!r} mismatch: {r.get(k)!r} != {v!r}"
    st = pkl.stat()
    if r.get("pickle_bytes") != st.st_size:
        return "conflict", f"pickle_bytes {r.get('pickle_bytes')} != st_size {st.st_size}"
    if r.get("pickle_sha256") != sha256_file(pkl):
        return "conflict", "pickle sha256 mismatch"
    g = r.get("gpu") or {}
    if not isinstance(g, dict) or g.get("NVIDIA_TF32_OVERRIDE") is not None \
            or g.get("CUBLAS_WORKSPACE_CONFIG") is not None:
        return "conflict", "record gpu env fields are not both null"
    mt = r.get("matched_tuple")
    if not isinstance(mt, dict) or not tuple_in(mt, allowlist):
        return "conflict", "record matched_tuple is not in the current allowlist"
    return "complete", None


# --------------------------------------------------------------------------
# Run context and per-slide processing
# --------------------------------------------------------------------------


class Ctx:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _versions() -> Dict[str, Any]:
    import torch
    import torchvision
    import timm
    import PIL
    import openslide
    import huggingface_hub
    import safetensors
    return {"torch": torch.__version__, "torchvision": torchvision.__version__,
            "timm": timm.__version__, "numpy": np.__version__, "pillow": PIL.__version__,
            "openslide_python": openslide.__version__,
            "openslide_library": getattr(openslide, "__library_version__", None),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "huggingface_hub": huggingface_hub.__version__,
            "safetensors": safetensors.__version__}


def _backend_flags() -> Dict[str, Any]:
    import torch
    return {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "main_process_num_threads": torch.get_num_threads()}


def build_record(ctx: Ctx, row: dict, *, coords_src, hop_src, old_dtype, slide_path, wsi_stat,
                 payload: bytes, n_tiles: int, t_slide0: str, t_slide1: str, wall: float,
                 fwd_seconds: float, n_batches: int) -> dict:
    rec = {
        "schema": SCHEMA, "task": TASK, "encoder": ENCODER, "bank": BANK,
        "slide_id": row["slide_id"], "n_tiles": n_tiles, "dim": MODEL_DIM, "width": WIDTH,
        "dtype": "float32", "c_contiguous": True, "amp": False,
        "forward_batch": ctx.forward_batch, "loader_batch": ctx.loader_batch,
        "num_workers": ctx.num_workers, "prefetch_factor": PREFETCH_FACTOR, "device": ctx.device_norm,
        "pickle_bytes": len(payload), "pickle_sha256": sha256_bytes(payload),
        "pickle_protocol": PICKLE_PROTOCOL,
        "inventory_path": str(ctx.inventory_path), "inventory_sha256": ctx.inventory_sha256,
        "inventory_row_index": row["_row_index"],
        "coords_source": coords_src, "coords_source_sha256": coords_src["sha256"],
        "hoptimus_coords_source": hop_src, "old_bank_dtype": old_dtype,
        "slide_path": str(slide_path),
        "wsi": {"st_size": wsi_stat.st_size, "st_mtime_ns": wsi_stat.st_mtime_ns,
                "realpath": str(realpath(slide_path))},
        "encoder_meta": ctx.rt.meta, "weights_sha256": ctx.weights_sha256,
        "gpu": {"gpu_model": ctx.observed_tuple["torch_device_name"],
                "driver": ctx.observed_tuple["nvidia_smi_driver_version"],
                "nvidia_smi": ctx.nvidia_smi_raw,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE")},
        "allowlist_mode": ctx.allowlist_mode, "allowlist_path": str(ctx.allowlist_path),
        "allowlist_sha256": ctx.allowlist_sha256, "observed_tuple": ctx.observed_tuple,
        "matched_tuple": ctx.matched_tuple,
        "host": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version.split()[0], "versions": ctx.versions,
        "torch_backend_flags": _backend_flags(),
        "threads": {k: os.environ.get(k) for k in THREAD_ENV}, "hf_offline_env": ctx.hf_env,
        "git": ctx.git, "code_root": str(ctx.code_root),
        "code_path": str(ctx.code_path), "code_sha256": ctx.code_sha256,
        "extractor_code_path": str(ctx.extractor_code_path),
        "extractor_code_sha256": ctx.extractor_code_sha256,
        "code_manifest_path": str(ctx.code_manifest_path),
        "code_manifest_sha256": ctx.code_manifest_sha256,
        "code_manifest_excluded_modules": sorted(NON_FILE_MODULE_EXCLUSIONS),
        "prosper_file": str(ctx.prosper_file), "sys_path_head": list(sys.path[0:3]),
        "lsf_job": os.environ.get("LSB_JOBID"), "lsf_index": os.environ.get("LSB_JOBINDEX"),
        "lsf_queue": os.environ.get("LSB_QUEUE"), "lsf_djob_numproc": os.environ.get("LSB_DJOB_NUMPROC"),
        "argv": list(ctx.argv), "run_root": str(ctx.run_root), "out_root": str(ctx.out_root),
        "run_started_utc": ctx.run_started_utc, "slide_started_utc": t_slide0,
        "slide_ended_utc": t_slide1, "slide_wall_seconds": wall,
        "encoder_forward_seconds": fwd_seconds,
        "images_per_second_forward": (n_tiles / fwd_seconds) if fwd_seconds > 0 else None,
        "tiles_per_second_slide": (n_tiles / wall) if wall > 0 else None,
        "n_forward_batches": n_batches, "written_at": utc_now(),
    }
    if tuple(rec.keys()) != RECORD_KEYS:
        raise SlideFailure("record", f"record keys diverge from RECORD_KEYS: "
                                     f"{sorted(set(rec) ^ set(RECORD_KEYS))}")
    return rec


def process_slide(ctx: Ctx, row: dict) -> str:
    """Returns ``"skipped" | "written"``; raises SlideFailure otherwise."""
    import torch
    sid = row["slide_id"]
    pkl, rec_path, failed_dir = output_paths(ctx.out_root, sid)
    for target in (pkl, rec_path):
        try:
            check_no_symlink_below(ctx.run_root, target)
        except Refusal as exc:
            raise SlideFailure("symlink", exc.message) from None
    t0 = time.perf_counter()
    t_slide0 = utc_now()
    if row["dtype"] not in ("float16", "float32"):
        raise SlideFailure("inventory", f"unexpected old bank dtype {row['dtype']!r}")
    n = int(row["canonical_tiles"])
    old_feats, coords, coords_src = read_bank_pickle(row, "coords")
    old_dtype = str(getattr(old_feats, "dtype", None))
    if old_dtype != row["dtype"]:
        raise SlideFailure("coords", f"old bank dtype {old_dtype} != inventory {row['dtype']}")
    hrow = hoptimus_row(ctx.all_rows, sid)
    if int(hrow["canonical_tiles"]) != n:
        raise SlideFailure("coords", f"h_optimus_1 canonical_tiles {hrow['canonical_tiles']} != {n}")
    _, hcoords, hop_src = read_bank_pickle(hrow, "hoptimus_coords")
    if locs_of(coords) != locs_of(hcoords):
        raise SlideFailure("coords", "virchow2 and h_optimus_1 coords differ elementwise")
    del old_feats
    expected = expected_skip_view(ctx, row, hop_src, coords_src)
    status, reason = existing_status(pkl, rec_path, expected, ctx.allowlist)
    if status == "complete":
        return "skipped"
    if status == "conflict":
        raise SlideFailure("existing_output", reason or "conflict")
    slide_path = ctx.slide_path_of(sid)
    if not Path(slide_path).is_file():
        raise SlideFailure("wsi", f"WSI missing {slide_path}")
    wsi_stat = Path(slide_path).stat()
    ds = IdentityTiles(str(slide_path), locs_of(coords), ctx.rt.transform, ctx.opener)
    feats, n_batches, fwd_seconds = encode_slide(
        ctx.rt, ds, device=ctx.device, num_workers=ctx.num_workers,
        loader_batch=ctx.loader_batch, forward_batch=ctx.forward_batch)
    payload = pickle.dumps((feats, coords), protocol=PICKLE_PROTOCOL)
    t_slide1 = utc_now()
    wall = time.perf_counter() - t0
    rec = build_record(ctx, row, coords_src=coords_src, hop_src=hop_src, old_dtype=old_dtype,
                       slide_path=slide_path, wsi_stat=wsi_stat, payload=payload, n_tiles=n,
                       t_slide0=t_slide0, t_slide1=t_slide1, wall=wall, fwd_seconds=fwd_seconds,
                       n_batches=n_batches)
    for target in (pkl, rec_path):
        try:
            check_no_symlink_below(ctx.run_root, target)
        except Refusal as exc:
            raise SlideFailure("symlink", exc.message) from None
    pkl.parent.mkdir(parents=True, exist_ok=True)
    try:
        X._publish_bytes(pkl, payload)
    except FileExistsError as exc:
        raise SlideFailure("publish", f"pickle appeared concurrently: {pkl}") from exc
    try:
        X._publish_bytes(rec_path, (json.dumps(rec, sort_keys=True, indent=2, default=str) + "\n").encode())
    except FileExistsError as exc:
        raise SlideFailure("publish", f"record appeared concurrently: {rec_path}") from exc
    return "written"


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------


def _default_num_workers() -> Optional[int]:
    v = os.environ.get("LSB_DJOB_NUMPROC")
    if v is None:
        return None
    try:
        return int(v) - 1
    except ValueError:
        return None


def add_extract_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--inventory-sha256", required=True)
    p.add_argument("--weights-sha256", required=True)
    p.add_argument("--expected-snapshot-revision", default=HF_SNAPSHOT_REVISION)
    p.add_argument("--allowlist", required=True, type=Path)
    p.add_argument("--allowlist-sha256", required=True)
    p.add_argument("--code-manifest", required=True, type=Path)
    p.add_argument("--code-manifest-sha256", required=True)
    p.add_argument("--code-root", required=True, type=Path)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--lo", required=True, type=int, help="1-based inclusive over virchow2 rows")
    p.add_argument("--hi", required=True, type=int)
    p.add_argument("--slides", nargs="*", default=None, help="optional slide_id filter within lo..hi")
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--forward-batch", type=int, default=FORWARD_BATCH)
    p.add_argument("--loader-batch", type=int, default=LOADER_BATCH)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pilot-observe-tuple", action="store_true")
    p.add_argument("--hf-snapshot", type=Path, default=None,
                   help="optional local HF snapshot dir; recorded only, build_encoder resolves via HF cache")
    p.add_argument("--tag", default=None, help="runs/<tag>.json name; default job.index.pid")


def run_extract(argv: Sequence[str], *, encoder_factory: Optional[Callable] = None,
                opener: Optional[Callable] = None, slide_path_of: Optional[Callable] = None,
                tuple_observer: Optional[Callable] = None,
                test_only_allow_cpu_production: bool = False,
                test_only_code_root_anywhere: bool = False) -> int:
    """The ``extract`` subcommand.  Seams (tests only): ``encoder_factory(name,
    device, amp)``, ``opener(path)``, ``slide_path_of(slide_id)``,
    ``tuple_observer(device) -> (tuple_dict, raw_nvidia_smi)``.

    ``test_only_allow_cpu_production`` / ``test_only_code_root_anywhere`` are
    keyword-only, not reachable from argv/``main``, and are refused unless
    both ``encoder_factory`` and ``tuple_observer`` doubles are injected, so a
    real-encoder production run cannot use them (M2, L3)."""
    p = argparse.ArgumentParser(prog="r2_1_virchow2_fp32_extract_cli extract")
    add_extract_args(p)
    a = p.parse_args(list(argv))
    run_started = utc_now()
    seams_injected = encoder_factory is not None and tuple_observer is not None
    encoder_factory = encoder_factory or X.build_encoder
    opener = opener or X.openslide_opener
    slide_path_of = slide_path_of or paths.slide_wsi_path
    tuple_observer = tuple_observer or observe_gpu_tuple
    failed_dir: Optional[Path] = None
    run_root: Optional[Path] = None

    def refuse(exc: Refusal) -> int:
        print(f"REFUSED {exc}", file=sys.stderr)
        if failed_dir is not None:
            try:
                check_no_symlink_below(run_root, failed_dir)
                write_failed(failed_dir, "startup", {
                    "slide_id": None, "stage": exc.stage, "error": exc.message,
                    "argv": list(argv), "time": utc_now(), "lsf_job": os.environ.get("LSB_JOBID"),
                    "lsf_index": os.environ.get("LSB_JOBINDEX"), "pid": os.getpid()})
            except Exception as e2:                       # noqa: BLE001
                print(f"[warn] could not write startup FAILED: {e2}", file=sys.stderr)
        return RC_REFUSED

    try:
        # 1. environment
        hf_env = check_env()
        pilot = bool(a.pilot_observe_tuple)
        if (test_only_allow_cpu_production or test_only_code_root_anywhere) and not seams_injected:
            raise Refusal("test_seam", "test-only seams require injected encoder_factory and tuple_observer")
        device_norm = normalise_device(a.device)
        if not pilot and a.device not in PRODUCTION_DEVICES and not (
                test_only_allow_cpu_production and device_norm == "cpu"):
            raise Refusal("args", f"production --device must be one of {list(PRODUCTION_DEVICES)}, got {a.device!r}")
        if a.weights_sha256 != WEIGHTS_SHA256:
            raise Refusal("args", f"--weights-sha256 {a.weights_sha256} != pinned {WEIGHTS_SHA256}")
        if not a.expected_snapshot_revision:
            raise Refusal("args", "--expected-snapshot-revision must not be empty")
        if a.width != WIDTH:
            raise Refusal("args", f"--width must be {WIDTH}")
        if a.forward_batch != FORWARD_BATCH:
            raise Refusal("args", f"--forward-batch must be {FORWARD_BATCH}")
        if a.loader_batch != LOADER_BATCH:
            raise Refusal("args", f"--loader-batch must be {LOADER_BATCH}")
        nw = a.num_workers if a.num_workers is not None else _default_num_workers()
        if nw is None:
            raise Refusal("args", "--num-workers not given and LSB_DJOB_NUMPROC unset")
        if nw < 1:
            raise Refusal("args", f"num_workers={nw} < 1 refused")
        # 2. prosper.__file__ under code root
        prosper_file = check_prosper_file(a.code_root)
        # 3. inventory sha + row selection
        v2_rows, all_rows = load_inventory(a.inventory, a.inventory_sha256)
        if not (1 <= a.lo <= a.hi <= len(v2_rows)):
            raise Refusal("args", f"--lo/--hi {a.lo}..{a.hi} outside 1..{len(v2_rows)}")
        rows = v2_rows[a.lo - 1:a.hi]
        if a.slides:
            want = set(a.slides)
            rows = [r for r in rows if r["slide_id"] in want]
            if {r["slide_id"] for r in rows} != want:
                raise Refusal("args", "--slides contains ids outside lo..hi or unknown")
        # 4. out-root realpath forbidden-parent check + mode confinement
        out_root, run_root = check_out_root(a.out_root, a.run_root, pilot, v2_rows, all_rows)
        for d in output_dirs(out_root):                  # NL1: before any write
            check_no_symlink_below(run_root, d)
        if not pilot and not test_only_code_root_anywhere:
            check_code_frozen(a.code_root, run_root)
        # 5. allowlist and manifest files (sha-checked); refusals from here on
        #    have an out root and may write FAILED/startup.*
        allow_obj, allow_sha = load_json_checked(a.allowlist, a.allowlist_sha256, "allowlist")
        allowlist = validate_allowlist(allow_obj, str(a.allowlist))
        manifest, manifest_sha = load_json_checked(a.code_manifest, a.code_manifest_sha256, "code_manifest")
        out_root.mkdir(parents=True, exist_ok=True)
        failed_dir = out_root / "FAILED"
        # 6. numerics
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if a.device.startswith("cuda") and not torch.cuda.is_available():
            raise Refusal("device", "CUDA requested but torch.cuda.is_available() is False")
        # 7. encoder + weights (NM1: live library offline flag asserted right before)
        hf_env["library_offline_before_encoder"] = check_hf_offline_effective()
        rt = encoder_factory(BANK, a.device, False)
        for k in HF_OFFLINE_ENV:
            if os.environ.get(k) != "1":
                raise Refusal("env", f"{k} is {os.environ.get(k)!r} after build_encoder, expected '1'")
        hf_env["library_offline_after_encoder"] = check_hf_offline_effective()
        weights = check_weights(rt, a.weights_sha256, a.expected_snapshot_revision)
        # 8. GPU tuple
        observed, smi_raw = tuple_observer(a.device)
        observed = validate_tuple(observed, "observed")
        if pilot:
            matched = None
            mode = "pilot"
        else:
            if not tuple_in(observed, allowlist):
                raise Refusal("gpu_tuple", f"observed GPU tuple not in allowlist: {observed}")
            matched = observed
            mode = "production"
        # 9. third-party versions, then code manifest over everything loaded
        versions = _versions()
        check_code_manifest(manifest, a.code_root)
    except Refusal as exc:
        return refuse(exc)

    this_file = realpath(__file__)
    x_file = realpath(X.__file__)
    ctx = Ctx(
        argv=list(argv), rt=rt, opener=opener, slide_path_of=slide_path_of, device=a.device,
        device_norm=device_norm, num_workers=nw, loader_batch=a.loader_batch, forward_batch=a.forward_batch,
        inventory_path=realpath(a.inventory), inventory_sha256=a.inventory_sha256, all_rows=all_rows,
        allowlist=allowlist, allowlist_path=realpath(a.allowlist), allowlist_sha256=allow_sha,
        allowlist_mode=mode, observed_tuple=observed, matched_tuple=matched, nvidia_smi_raw=smi_raw,
        weights_sha256=weights["sha256"], versions=versions, hf_env=hf_env,
        git=X._git_state(realpath(a.code_root)), code_root=realpath(a.code_root),
        code_path=this_file, code_sha256=sha256_file(this_file),
        extractor_code_path=x_file, extractor_code_sha256=sha256_file(x_file),
        code_manifest_path=realpath(a.code_manifest), code_manifest_sha256=manifest_sha,
        prosper_file=prosper_file, run_root=run_root, out_root=out_root, run_started_utc=run_started,
    )
    try:
        for d in output_dirs(out_root):                  # NL1: re-checked around the mkdirs
            check_no_symlink_below(run_root, d)
            d.mkdir(parents=True, exist_ok=True)
            check_no_symlink_below(run_root, d)
    except Refusal as exc:
        failed_dir = None                                 # never write through a link
        return refuse(exc)

    def slide_failed(sid: str, obj: dict) -> Optional[Path]:
        try:
            check_no_symlink_below(run_root, failed_dir)
        except Refusal as exc:
            print(f"[{sid}] FAILED evidence not written: {exc}; record: "
                  f"{json.dumps(obj, sort_keys=True, default=str)}", file=sys.stderr)
            return None
        return write_failed(failed_dir, sid, obj)

    counts = {"written": 0, "skipped": 0, "failed": 0}
    failures: List[dict] = []
    for row in rows:
        sid = row["slide_id"]
        try:
            res = process_slide(ctx, row)
            counts[res] += 1
            print(f"[{sid}] {res}", file=sys.stderr)
        except SlideFailure as exc:
            counts["failed"] += 1
            obj = {"slide_id": sid, "stage": exc.stage, "error": exc.message, "time": utc_now(),
                   "inventory_row_index": row["_row_index"], "allowlist_mode": mode,
                   "lsf_job": os.environ.get("LSB_JOBID"), "lsf_index": os.environ.get("LSB_JOBINDEX"),
                   "pid": os.getpid(), **exc.extra}
            fp = slide_failed(sid, obj)
            failures.append({"slide_id": sid, "stage": exc.stage, "file": None if fp is None else str(fp)})
            print(f"[{sid}] FAILED {exc}", file=sys.stderr)
        except Exception as exc:                          # noqa: BLE001
            counts["failed"] += 1
            obj = {"slide_id": sid, "stage": "unexpected", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(), "time": utc_now(), "pid": os.getpid(),
                   "lsf_job": os.environ.get("LSB_JOBID"), "lsf_index": os.environ.get("LSB_JOBINDEX")}
            fp = slide_failed(sid, obj)
            failures.append({"slide_id": sid, "stage": "unexpected", "file": None if fp is None else str(fp)})
            print(f"[{sid}] FAILED unexpected {exc}", file=sys.stderr)

    job, idx = lsf_components()
    tag = a.tag or f"{job}.{idx}.{os.getpid()}.{time.time_ns()}"
    run_rec = {
        "schema": SCHEMA + ".run", "tag": tag, "argv": list(argv), "run_started_utc": run_started,
        "run_ended_utc": utc_now(), "lo": a.lo, "hi": a.hi, "n_rows": len(rows),
        "slide_ids": [r["slide_id"] for r in rows], "counts": counts, "failures": failures,
        "allowlist_mode": mode, "observed_tuple": observed, "matched_tuple": matched,
        "allowlist_sha256": allow_sha, "code_manifest_sha256": manifest_sha,
        "inventory_sha256": a.inventory_sha256, "weights_sha256": weights["sha256"],
        "code_sha256": ctx.code_sha256, "extractor_code_sha256": ctx.extractor_code_sha256,
        "num_workers": nw, "device": a.device, "host": socket.gethostname(),
        "lsf_job": os.environ.get("LSB_JOBID"), "lsf_index": os.environ.get("LSB_JOBINDEX"),
        "lsf_queue": os.environ.get("LSB_QUEUE"), "pid": os.getpid(),
        "torch_backend_flags": _backend_flags(), "versions": versions, "hf_offline_env": hf_env,
    }
    try:
        check_no_symlink_below(run_root, out_root / "runs" / f"{tag}.json")
    except Refusal as exc:
        failed_dir = None
        return refuse(exc)
    X._publish_bytes(out_root / "runs" / f"{tag}.json",
                     (json.dumps(run_rec, sort_keys=True, indent=2, default=str) + "\n").encode())
    print(json.dumps(counts), file=sys.stderr)
    return RC_SLIDE_FAILURES if counts["failed"] else RC_OK


# --------------------------------------------------------------------------
# make-params: optimal contiguous K-way partition
# --------------------------------------------------------------------------


def _greedy_count(sizes: Sequence[int], cap: int) -> int:
    k, cur = 1, 0
    for s in sizes:
        if cur + s > cap:
            k += 1
            cur = s
        else:
            cur += s
    return k


def _greedy_partition(sizes: Sequence[int], cap: int) -> List[Tuple[int, int]]:
    """Contiguous chunks ``(lo_idx, hi_idx)`` 0-based inclusive, each sum <= cap."""
    chunks: List[Tuple[int, int]] = []
    start, cur = 0, 0
    for i, s in enumerate(sizes):
        if cur + s > cap and i > start:
            chunks.append((start, i - 1))
            start, cur = i, s
        else:
            cur += s
    chunks.append((start, len(sizes) - 1))
    return chunks


def optimal_partition(sizes: Sequence[int], k: int) -> Tuple[List[Tuple[int, int]], int]:
    """Minimise the maximum chunk sum over contiguous partitions into exactly ``k``
    chunks (binary search on the cap with a greedy feasibility test, then split
    chunks until exactly ``k``)."""
    n = len(sizes)
    if k < 1 or k > n:
        raise Refusal("params", f"K={k} must be within 1..{n}")
    lo, hi = max(sizes), sum(sizes)
    while lo < hi:
        mid = (lo + hi) // 2
        if _greedy_count(sizes, mid) <= k:
            hi = mid
        else:
            lo = mid + 1
    chunks = _greedy_partition(sizes, lo)
    while len(chunks) < k:
        # split the chunk with the most slides (ties: first) at its midpoint
        j = max(range(len(chunks)), key=lambda i: (chunks[i][1] - chunks[i][0], -i))
        a, b = chunks[j]
        if b == a:
            raise Refusal("params", "cannot split further to reach K chunks")
        m = (a + b) // 2
        chunks[j:j + 1] = [(a, m), (m + 1, b)]
    return chunks, lo


def run_make_params(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(prog="make-params")
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--inventory-sha256", required=True)
    p.add_argument("--k", type=int, default=60)
    p.add_argument("--out", required=True, type=Path, help="params.txt path (must not exist)")
    a = p.parse_args(list(argv))
    try:
        v2, _ = load_inventory(a.inventory, a.inventory_sha256)
        sizes = [int(r["canonical_tiles"]) for r in v2]
        chunks, cap = optimal_partition(sizes, a.k)
    except Refusal as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        return RC_REFUSED
    total = sum(sizes)
    bound = math.ceil(total / a.k) + max(sizes)
    sums = [sum(sizes[lo:hi + 1]) for lo, hi in chunks]
    covered = [i for lo, hi in chunks for i in range(lo, hi + 1)]
    assert covered == list(range(len(sizes))), "exactly-once coverage violated"
    assert max(sums) <= bound, f"max chunk {max(sums)} > bound {bound}"
    assert len(chunks) == a.k
    lines = [f"{i + 1} {lo + 1} {hi + 1} {hi - lo + 1} {sums[i]}" for i, (lo, hi) in enumerate(chunks)]
    text = ("\n".join(lines) + "\n").encode()
    X._publish_bytes(a.out, text)
    sha = sha256_bytes(text)
    side = {"inventory": str(a.inventory), "inventory_sha256": a.inventory_sha256, "k": a.k,
            "n_slides": len(sizes), "total_tiles": total, "max_slide_tiles": max(sizes),
            "max_chunk_tiles": max(sums), "min_chunk_tiles": min(sums), "optimal_cap": cap,
            "bound": bound, "max_le_bound": max(sums) <= bound, "exactly_once_coverage": True,
            "params_sha256": sha, "line_format": "index lo hi n_slides n_tiles (1-based inclusive)",
            "chunks": [{"index": i + 1, "lo": lo + 1, "hi": hi + 1, "n_slides": hi - lo + 1,
                        "n_tiles": sums[i], "first_slide": v2[lo]["slide_id"],
                        "last_slide": v2[hi]["slide_id"]} for i, (lo, hi) in enumerate(chunks)]}
    X._publish_bytes(a.out.with_suffix(".json"),
                     (json.dumps(side, indent=2, sort_keys=True) + "\n").encode())
    print(sha)
    return RC_OK


# --------------------------------------------------------------------------
# Allowlist helpers
# --------------------------------------------------------------------------


def run_seed_allowlist(argv: Sequence[str]) -> int:
    """B-N1 seed: exactly ``[SEED_TUPLE]``; ``<out>.provenance.json`` beside it
    binds the seed to the files that attest it (``--provenance``, sha-recorded) (L9)."""
    p = argparse.ArgumentParser(prog="seed-allowlist")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--provenance", nargs="+", required=True, type=Path,
                   help="files attesting the seed tuple (e.g. X first-run records); sha-recorded")
    a = p.parse_args(list(argv))
    sources = []
    for f in a.provenance:
        if not f.is_file():
            print(f"REFUSED [allowlist] --provenance {f} is not a regular file", file=sys.stderr)
            return RC_REFUSED
        sources.append({"path": str(realpath(f)), "bytes": f.stat().st_size, "sha256": sha256_file(f)})
    text = (json.dumps([SEED_TUPLE], indent=2, sort_keys=True) + "\n").encode()
    X._publish_bytes(a.out, text)
    sha = sha256_bytes(text)
    side = {"kind": "seed", "seed_tuple": SEED_TUPLE, "out_sha256": sha, "sources": sources,
            "written_at": utc_now()}
    X._publish_bytes(Path(str(a.out) + ".provenance.json"),
                     (json.dumps(side, indent=2, sort_keys=True) + "\n").encode())
    print(sha)
    return RC_OK


def run_extend_allowlist(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(prog="extend-allowlist")
    p.add_argument("--allowlist", required=True, type=Path)
    p.add_argument("--allowlist-sha256", required=True)
    p.add_argument("--pilot-root", required=True, type=Path, help="<run_root>/pilot")
    p.add_argument("--host-tags", nargs="+", required=True)
    p.add_argument("--passed-host-tags", nargs="+", required=True,
                   help="subset of --host-tags that passed the lead's pilot gate; only these are admitted")
    p.add_argument("--out", required=True, type=Path, help="NEW allowlist path (must not exist)")
    a = p.parse_args(list(argv))
    try:
        obj, _ = load_json_checked(a.allowlist, a.allowlist_sha256, "allowlist")
        allow = validate_allowlist(obj, str(a.allowlist))
        if a.out.exists():
            raise Refusal("allowlist", f"--out {a.out} exists; never written in place")
        if len(set(a.host_tags)) != len(a.host_tags) or len(set(a.passed_host_tags)) != len(a.passed_host_tags):
            raise Refusal("allowlist", "duplicate host tags")
        if not set(a.passed_host_tags) <= set(a.host_tags):
            raise Refusal("allowlist", f"--passed-host-tags {a.passed_host_tags} not a subset of --host-tags")
        admitted: List[Dict[str, Any]] = []
        prov: List[dict] = []
        for tag in a.passed_host_tags:
            troot = a.pilot_root / tag
            fdir = troot / "FAILED"
            if fdir.exists() and any(fdir.iterdir()):
                raise Refusal("allowlist", f"pilot {tag} has FAILED files; not admitted")
            fe = troot / "features" / ENCODER
            recs = sorted(fe.glob("*.record.json"))
            if not recs:
                raise Refusal("allowlist", f"pilot {tag} has no records")
            rec_ids = {rp.name[:-len(".record.json")] for rp in recs}
            pkl_ids = {pp.name[:-len(".pkl")] for pp in fe.glob("*.pkl")}
            if rec_ids != pkl_ids:
                raise Refusal("allowlist", f"pilot {tag} orphan pickle/record: "
                                           f"{sorted(rec_ids ^ pkl_ids)}")
            tuples: List[Dict[str, Any]] = []
            for rp in recs:
                r = json.loads(rp.read_text())
                if r.get("allowlist_mode") != "pilot":
                    raise Refusal("allowlist", f"{rp} allowlist_mode={r.get('allowlist_mode')!r} != 'pilot'")
                if r.get("matched_tuple") is not None:
                    raise Refusal("allowlist", f"{rp} pilot record has non-null matched_tuple")
                pk = fe / (rp.name[:-len(".record.json")] + ".pkl")
                if r.get("pickle_bytes") != pk.stat().st_size or r.get("pickle_sha256") != sha256_file(pk):
                    raise Refusal("allowlist", f"{pk} size/sha256 does not match {rp}")
                t = validate_tuple(r.get("observed_tuple"), str(rp))
                if tuples and t != tuples[0]:
                    raise Refusal("allowlist", f"pilot {tag} records disagree on observed tuple")
                tuples.append(t)
                prov.append({"host_tag": tag, "record": str(rp), "record_sha256": sha256_file(rp),
                             "pickle": str(pk), "pickle_sha256": r.get("pickle_sha256"),
                             "slide_id": r.get("slide_id")})
            t = tuples[0]
            if not tuple_in(t, allow) and not tuple_in(t, admitted):
                admitted.append(t)
    except Refusal as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        return RC_REFUSED
    new = allow + admitted
    text = (json.dumps(new, indent=2, sort_keys=True) + "\n").encode()
    X._publish_bytes(a.out, text)
    sha = sha256_bytes(text)
    side = {"source_allowlist": str(a.allowlist), "source_sha256": a.allowlist_sha256,
            "pilot_root": str(a.pilot_root), "host_tags": a.host_tags,
            "passed_host_tags": a.passed_host_tags, "admitted": admitted,
            "n_before": len(allow), "n_after": len(new), "records": prov, "out_sha256": sha,
            "written_at": utc_now()}
    X._publish_bytes(Path(str(a.out) + ".provenance.json"),
                     (json.dumps(side, indent=2, sort_keys=True) + "\n").encode())
    print(sha)
    return RC_OK


def run_make_code_manifest(argv: Sequence[str]) -> int:
    """Convenience: import the extractor chain, then dump ``loaded_code_modules``.

    Limits: the recorded set is whatever is loaded in THIS process after the
    listed imports; the production process may load additional modules under
    the code root (e.g. via ``build_encoder``) and will refuse if so.  The analyst
    must therefore generate the manifest with the same interpreter, cwd and
    sys.path as the production job, and verify against a dry start-up."""
    p = argparse.ArgumentParser(prog="make-code-manifest")
    p.add_argument("--code-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--import", dest="imports", nargs="*", default=[
        "prosper.evaluation.r2_1_virchow2_fp32_extract_cli",
        "prosper.evaluation.r2_1_perturb_extract_cli",
        "prosper.embeddings.patch_embedding.virchow2_embeddings"])
    a = p.parse_args(list(argv))
    import importlib
    for m in a.imports:
        importlib.import_module(m)
    try:
        man = loaded_code_modules(a.code_root)
    except Refusal as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        return RC_REFUSED
    text = (json.dumps(man, indent=2, sort_keys=True) + "\n").encode()
    X._publish_bytes(a.out, text)
    print(sha256_bytes(text))
    return RC_OK


# --------------------------------------------------------------------------
# Numerical comparison helpers (calibrate / fidelity)
# --------------------------------------------------------------------------


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Integer ULP distance between same-dtype float arrays (float32 or float16),
    via the monotone signed-integer bit mapping ``i<0 -> INT_MIN - i``."""
    if a.dtype != b.dtype or a.dtype not in (np.float32, np.float16):
        raise Refusal("ulp", f"ulp_distance needs matching float16/float32 arrays, got {a.dtype},{b.dtype}")
    if a.dtype == np.float32:
        it, imin = np.int32, np.int32(-2 ** 31)
    else:
        it, imin = np.int16, np.int16(-2 ** 15)
    ia = a.view(it).astype(np.int64)
    ib = b.view(it).astype(np.int64)
    ia = np.where(ia < 0, np.int64(imin) - ia, ia)
    ib = np.where(ib < 0, np.int64(imin) - ib, ib)
    return np.abs(ia - ib)


def ulp_histogram(d: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for label, lo, hi in ULP_BINS:
        m = (d >= lo) if hi is None else ((d >= lo) & (d <= hi))
        out[label] = int(m.sum())
    return out


def compare_arrays(new: np.ndarray, old: np.ndarray, *, what: str) -> dict:
    """Per-slide statistics in float64.  Zero-norm old rows fail closed."""
    if new.shape != old.shape:
        raise Refusal(what, f"shape mismatch {new.shape} vs {old.shape}")
    n64 = np.asarray(new, dtype=np.float64)
    o64 = np.asarray(old, dtype=np.float64)
    d = n64 - o64
    onorm = np.linalg.norm(o64, axis=1)
    nnorm = np.linalg.norm(n64, axis=1)
    if (onorm == 0).any() or (nnorm == 0).any():
        raise Refusal(what, "zero-norm row encountered; refusing (fail closed)")
    rel = np.linalg.norm(d, axis=1) / onorm
    cos = (n64 * o64).sum(axis=1) / (nnorm * onorm)
    abs_row = np.abs(d).max(axis=1)
    bit_identical = (np.ascontiguousarray(new).tobytes() == np.ascontiguousarray(old).tobytes()
                     and new.dtype == old.dtype)
    stats = {
        "n_rows": int(new.shape[0]), "max_abs": float(abs_row.max()), "max_rel_l2": float(rel.max()),
        "min_cosine": float(cos.min()), "median_rel_l2": float(np.median(rel)),
        "median_cosine": float(np.median(cos)), "bit_identical": bool(bit_identical),
        "frac_rows_identical": float((abs_row == 0).mean()),
        "rel_l2_quantiles": {q: float(np.quantile(rel, float(q))) for q in ("0.5", "0.9", "0.99", "1.0")},
        "cosine_quantiles": {q: float(np.quantile(cos, float(q))) for q in ("0.0", "0.01", "0.1", "0.5")},
    }
    if new.dtype == old.dtype and new.dtype in (np.float32, np.float16):
        u = ulp_distance(np.ascontiguousarray(new), np.ascontiguousarray(old))
        stats["ulp_histogram_elements"] = ulp_histogram(u.ravel())
        stats["ulp_histogram_row_max"] = ulp_histogram(u.max(axis=1))
        stats["ulp_max"] = int(u.max())
        stats["frac_rows_le_2ulp"] = float((u.max(axis=1) <= 2).mean())
    return stats, {"rel": rel, "cos": cos, "abs_row": abs_row, "d": d}


def _load_new_pickle(features_dir: Path, slide_id: str) -> Tuple[np.ndarray, list, dict]:
    pkl = Path(features_dir) / f"{slide_id}.pkl"
    rec_p = Path(features_dir) / f"{slide_id}.record.json"
    if not pkl.is_file() or not rec_p.is_file():
        raise Refusal("new_pickle", f"missing pickle/record for {slide_id} under {features_dir}")
    rec = json.loads(rec_p.read_text())
    data = read_checked(pkl, rec["pickle_sha256"], rec["pickle_bytes"], "new_pickle")
    feats, coords = pickle.loads(data)
    return feats, coords, rec


def _load_old_fp32(row: dict) -> Tuple[np.ndarray, list, dict]:
    try:
        feats, coords, src = read_bank_pickle(row, "old_pickle")
    except SlideFailure as exc:
        raise Refusal("old_pickle", exc.message) from exc
    return feats, coords, src


def _check_new_controls(feats: np.ndarray, n: int, what: str) -> None:
    if not isinstance(feats, np.ndarray) or feats.dtype != np.float32:
        raise Refusal(what, "new feats not float32 ndarray")
    if feats.shape != (n, WIDTH):
        raise Refusal(what, f"new feats shape {feats.shape} != {(n, WIDTH)}")
    if not feats.flags["C_CONTIGUOUS"]:
        raise Refusal(what, "new feats not C-contiguous")
    if not np.isfinite(feats).all():
        raise Refusal(what, "new feats non-finite")


def _is_sha256_hex(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _load_x_identity(x_features_dir: Path, row: dict, first_run: Dict[str, str]
                     ) -> Tuple[np.ndarray, list, dict, Path]:
    """X first-run ``identity[:, :1280]`` for one slide (A-F2 step 4.0).

    The pickle is read only after its size+sha256 match the X record; the
    record must carry ``code_sha256 == X_SHA256_PINNED``, ``encoder ==
    "virchow2"``, ``amp is False`` (N7), the row's ``slide_id`` and
    ``n_tiles == canonical_tiles``, and (NL3) the first-run ``lsf_job``,
    ``manifest_path`` (realpath-equal) and ``manifest_sha256`` given in
    ``first_run``.  A malformed record is a Refusal (NL4)."""
    sid = row["slide_id"]
    n = int(row["canonical_tiles"])
    xp = Path(x_features_dir) / f"{sid}.pkl"
    xr = Path(x_features_dir) / f"{sid}.record.json"
    if not xp.is_file() or not xr.is_file():
        raise Refusal("calibrate", f"X pickle/record missing for {sid} under {x_features_dir}")
    try:
        xrec = json.loads(xr.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Refusal("calibrate", f"{sid}: X record {xr} is not valid UTF-8 JSON: {exc}") from None
    if not isinstance(xrec, dict):
        raise Refusal("calibrate", f"{sid}: X record {xr} is a JSON {type(xrec).__name__}, not an object")
    for k in ("lsf_job", "manifest_path", "manifest_sha256"):
        if not isinstance(xrec.get(k), str):
            raise Refusal("calibrate", f"{sid}: X record {k} {xrec.get(k)!r} is not a string")
    if xrec["lsf_job"] != first_run["lsf_job"]:
        raise Refusal("calibrate", f"{sid}: X record lsf_job {xrec['lsf_job']!r} != first run "
                                   f"{first_run['lsf_job']!r}")
    if realpath(Path(xrec["manifest_path"])) != realpath(Path(first_run["manifest_path"])):
        raise Refusal("calibrate", f"{sid}: X record manifest_path {xrec['manifest_path']!r} != first run "
                                   f"{first_run['manifest_path']!r}")
    if xrec["manifest_sha256"] != first_run["manifest_sha256"]:
        raise Refusal("calibrate", f"{sid}: X record manifest_sha256 {xrec['manifest_sha256']!r} != first run "
                                   f"{first_run['manifest_sha256']!r}")
    if xrec.get("schema") != X_RECORD_SCHEMA:
        raise Refusal("calibrate", f"{sid}: X record schema {xrec.get('schema')!r} != {X_RECORD_SCHEMA!r}")
    if xrec.get("code_sha256") != X_SHA256_PINNED:
        raise Refusal("calibrate", f"{sid}: X record code_sha256 {xrec.get('code_sha256')!r} != pinned X")
    if xrec.get("encoder") != BANK:
        raise Refusal("calibrate", f"{sid}: X record encoder {xrec.get('encoder')!r} != {BANK!r}")
    if xrec.get("amp") is not False:
        raise Refusal("calibrate", f"{sid}: X record amp {xrec.get('amp')!r} is not False")
    if str(xrec.get("slide_id")) != sid or xrec.get("n_tiles") != n:
        raise Refusal("calibrate", f"{sid}: X record slide_id/n_tiles {xrec.get('slide_id')!r}/"
                                   f"{xrec.get('n_tiles')!r} != {sid!r}/{n}")
    pb, ps = xrec.get("pickle_bytes"), xrec.get("pickle_sha256")
    if not isinstance(pb, int) or isinstance(pb, bool) or pb < 0:
        raise Refusal("calibrate", f"{sid}: X record pickle_bytes {pb!r} is not a non-negative integer")
    if not _is_sha256_hex(ps):
        raise Refusal("calibrate", f"{sid}: X record pickle_sha256 {ps!r} is not 64 lowercase hex characters")
    xdata = read_checked(xp, xrec["pickle_sha256"], xrec["pickle_bytes"], "x_pickle")
    xobj = pickle.loads(xdata)
    if not isinstance(xobj, tuple) or len(xobj) != 2 or not isinstance(xobj[0], dict) \
            or "identity" not in xobj[0]:
        raise Refusal("calibrate", f"{sid}: X pickle is not a ({{condition: feats}}, coords) pair with 'identity'")
    ident = np.asarray(xobj[0]["identity"])
    if ident.dtype != np.float32 or ident.ndim != 2 or ident.shape != (n, MODEL_DIM):
        raise Refusal("calibrate", f"{sid}: X identity {ident.dtype} {ident.shape} != float32 {(n, MODEL_DIM)}")
    xfeats = np.ascontiguousarray(ident[:, :WIDTH])
    if not np.isfinite(xfeats).all():
        raise Refusal("calibrate", f"{sid}: X identity[:, :{WIDTH}] non-finite")
    return xfeats, xobj[1], xrec, xr


def run_calibrate(argv: Sequence[str]) -> int:
    """Step 4.0 (A-F2 / B-N2), CPU only, before any GPU hour: for the 16
    stored-fp32 slides that X's first run already extracted
    (``CALIBRATION_SLIDES``), compare the **published** fp32 pickle (read via
    the Stage0 inventory row, size+sha256 checked, ``dtype == "float32"``)
    with X first-run ``identity[:, :1280]`` (size+sha256 checked against the
    X record).  Coordinates are asserted equal.  The published row is the
    rel-L2 denominator (``compare_arrays(x, published)``).  No V1 output and
    no record beside the published pickles is needed."""
    p = argparse.ArgumentParser(prog="calibrate")
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--inventory-sha256", required=True)
    p.add_argument("--x-features-dir", required=True, type=Path,
                   help="X first-run features/virchow2 dir with <sid>.pkl and <sid>.record.json")
    p.add_argument("--x-lsf-job", required=True,
                   help="X first-run LSF job id; every X record's lsf_job must equal it (NL3)")
    p.add_argument("--x-manifest-path", required=True, type=Path,
                   help="X first-run stimulus manifest; records' manifest_path must resolve to it (NL3)")
    p.add_argument("--x-manifest-sha256", required=True,
                   help="sha256 of --x-manifest-path; records' manifest_sha256 must equal it (NL3)")
    p.add_argument("--slides", nargs="+", default=None,
                   help="optional; if given must equal the pinned CALIBRATION_SLIDES set")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(list(argv))
    try:
        if not a.x_lsf_job:
            raise Refusal("calibrate", "--x-lsf-job must not be empty")
        if not _is_sha256_hex(a.x_manifest_sha256):
            raise Refusal("calibrate", "--x-manifest-sha256 must be 64 lowercase hex characters")
        if not Path(a.x_manifest_path).is_file():
            raise Refusal("calibrate", f"--x-manifest-path {a.x_manifest_path} is not a file")
        got = sha256_file(a.x_manifest_path)
        if got != a.x_manifest_sha256:
            raise Refusal("calibrate", f"--x-manifest-path sha256 {got} != --x-manifest-sha256 "
                                       f"{a.x_manifest_sha256}")
        first_run = {"lsf_job": a.x_lsf_job, "manifest_path": str(a.x_manifest_path),
                     "manifest_sha256": a.x_manifest_sha256}
        pinned = list(CALIBRATION_SLIDES)
        if a.slides is not None and (sorted(a.slides) != sorted(pinned) or len(set(a.slides)) != len(a.slides)):
            raise Refusal("calibrate", f"--slides {a.slides} != pinned step-4.0 set {pinned}")
        v2, _ = load_inventory(a.inventory, a.inventory_sha256)
        by_id = {r["slide_id"]: r for r in v2}
        per: Dict[str, dict] = {}
        n_total = 0
        for sid in pinned:
            if sid not in by_id:
                raise Refusal("calibrate", f"{sid} not a virchow2 inventory row")
            row = by_id[sid]
            if row["dtype"] != "float32":
                raise Refusal("calibrate", f"{sid}: inventory dtype {row['dtype']!r} != 'float32'")
            n = int(row["canonical_tiles"])
            published, pcoords, src = _load_old_fp32(row)
            published = np.asarray(published)
            if published.dtype != np.float32 or published.shape != (n, WIDTH):
                raise Refusal("calibrate", f"{sid}: published {published.dtype} {published.shape} "
                                           f"!= float32 {(n, WIDTH)}")
            published = np.ascontiguousarray(published)
            xfeats, xcoords, xrec, xr = _load_x_identity(a.x_features_dir, row, first_run)
            if locs_of(xcoords) != locs_of(pcoords):
                raise Refusal("calibrate", f"{sid}: X coords != published coords")
            stats, _ = compare_arrays(xfeats, published, what=f"calibrate:{sid}")
            stats.update({"published_pickle": src, "inventory_row_index": row["_row_index"],
                          "x_pickle_sha256": xrec["pickle_sha256"], "x_pickle_bytes": xrec["pickle_bytes"],
                          "x_record_path": str(xr), "x_record_sha256": sha256_file(xr),
                          "x_code_sha256": xrec["code_sha256"], "x_encoder": xrec["encoder"],
                          "x_amp": xrec["amp"], "x_lsf_job": xrec["lsf_job"],
                          "x_manifest_path": xrec["manifest_path"],
                          "x_manifest_sha256": xrec["manifest_sha256"], "denominator": "published"})
            per[sid] = stats
            n_total += n
        if n_total != CALIBRATION_TILES:
            raise Refusal("calibrate", f"step-4.0 tiles {n_total} != pinned {CALIBRATION_TILES}")
        max_abs = max(s["max_abs"] for s in per.values())
        max_rel = max(s["max_rel_l2"] for s in per.values())
        band_abs = min(1e-4, max(1e-6, 10.0 * max_abs))
        band_rel = min(1e-5, max(1e-7, 10.0 * max_rel))
        # NL2: classify 4.0 itself against its own derived band (A-F2: "If 4.0
        # itself lands in (ii)/(iii) it is reported").
        for st in per.values():
            st["class"] = classify_fp32(st, band_abs, band_rel)
        class_counts = {c: sum(1 for st in per.values() if st["class"] == c) for c in ("i", "ii", "iii")}
        report_to_author = any(st["class"] != "i" for st in per.values())
        report = {"schema": CALIBRATE_SCHEMA, "step": "4.0", "slides": pinned, "n_tiles": n_total,
                  "comparison": "published fp32 (inventory) vs X first-run identity[:, :1280]; "
                                "rel-L2 denominator = published row",
                  "per_slide": per,
                  "max_abs_over_slides": max_abs, "max_rel_l2_over_slides": max_rel,
                  "all_bit_identical": all(s["bit_identical"] for s in per.values()),
                  "band_abs": band_abs, "band_rel": band_rel,
                  "band_rule": "band_abs=min(1e-4,max(1e-6,10*max_abs)); band_rel=min(1e-5,max(1e-7,10*max_rel_l2))",
                  "class_counts": class_counts, "report_to_author": report_to_author,
                  "report_to_author_rule": "true iff any slide's class (classify_fp32 against this "
                                           "report's band) is not 'i' (A-F2: 4.0 in (ii)/(iii) is "
                                           "reported to the author before any GPU hour)",
                  "x_first_run": {"lsf_job": a.x_lsf_job,
                                  "manifest_path": str(realpath(a.x_manifest_path)),
                                  "manifest_sha256": a.x_manifest_sha256},
                  "inventory_path": str(realpath(a.inventory)), "inventory_sha256": a.inventory_sha256,
                  "x_features_dir": str(a.x_features_dir), "x_code_sha256_pinned": X_SHA256_PINNED,
                  "written_at": utc_now()}
    except Refusal as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        return RC_REFUSED
    text = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    X._publish_bytes(a.out, text)
    if report_to_author:
        print(f"WARNING step 4.0 outside its own fp32 band (class counts {class_counts}); "
              "report_to_author=true: the lead reports to the author before any GPU hour", file=sys.stderr)
    print(sha256_bytes(text))
    return RC_OK


def classify_fp32(stats: dict, band_abs: float, band_rel: float) -> str:
    if stats["bit_identical"] or (stats["max_abs"] <= band_abs and stats["max_rel_l2"] <= band_rel):
        return "i"
    if stats["max_rel_l2"] <= 1e-2 and stats["min_cosine"] >= 0.9999:
        return "ii"
    return "iii"


def bimodality_summary(values: np.ndarray) -> dict:
    """Two-group split of log10 per-slide median rel-L2 maximising between-group
    variance (Otsu-like); descriptive only."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    v = v[np.isfinite(v)]
    if v.size < 2:
        return {"n": int(v.size), "note": "too few finite values"}
    best = None
    for i in range(1, v.size):
        a, b = v[:i], v[i:]
        between = (a.size * b.size / v.size ** 2) * (a.mean() - b.mean()) ** 2
        if best is None or between > best[0]:
            best = (between, i)
    between, i = best
    total = v.var()
    gaps = np.diff(v)
    j = int(gaps.argmax())
    return {"n": int(v.size), "threshold": float((v[i - 1] + v[i]) / 2), "n_low": int(i),
            "n_high": int(v.size - i), "between_variance_fraction": float(between / total) if total > 0 else None,
            "largest_gap": float(gaps[j]), "largest_gap_at": float((v[j] + v[j + 1]) / 2),
            "min": float(v.min()), "max": float(v.max()), "median": float(np.median(v))}


def run_fidelity(argv: Sequence[str]) -> int:
    """Steps 4.1/4.2: classify stored-fp32 slides (B-N2) and describe fp16 slides."""
    p = argparse.ArgumentParser(prog="fidelity")
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--inventory-sha256", required=True)
    p.add_argument("--new-features-dir", required=True, type=Path)
    p.add_argument("--calibration-report", required=True, type=Path)
    p.add_argument("--calibration-report-sha256", required=True)
    p.add_argument("--slides", nargs="*", default=None)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(list(argv))
    try:
        v2, _ = load_inventory(a.inventory, a.inventory_sha256)
        cal, cal_sha = load_json_checked(a.calibration_report, a.calibration_report_sha256, "calibration_report")
        if not isinstance(cal, dict) or cal.get("schema") != CALIBRATE_SCHEMA or cal.get("step") != "4.0":
            raise Refusal("calibration_report", f"report schema/step {cal.get('schema') if isinstance(cal, dict) else None!r}/"
                                                f"{cal.get('step') if isinstance(cal, dict) else None!r} != "
                                                f"{CALIBRATE_SCHEMA!r}/'4.0'")
        band_abs, band_rel = float(cal["band_abs"]), float(cal["band_rel"])
        rows = v2 if not a.slides else [r for r in v2 if r["slide_id"] in set(a.slides)]
        fp32: Dict[str, dict] = {}
        fp16: Dict[str, dict] = {}
        pooled_dim_sum = np.zeros(WIDTH, dtype=np.float64)
        pooled_rows = 0
        pooled_ulp = {k[0]: 0 for k in ULP_BINS}
        for row in rows:
            sid = row["slide_id"]
            n = int(row["canonical_tiles"])
            new, ncoords, rec = _load_new_pickle(a.new_features_dir, sid)
            if rec.get("allowlist_mode") != "production":
                raise Refusal("fidelity", f"{sid}: record allowlist_mode {rec.get('allowlist_mode')!r} != 'production'")
            _check_new_controls(new, n, f"fidelity:{sid}")
            old, ocoords, src = _load_old_fp32(row)
            if locs_of(ocoords) != locs_of(ncoords):
                raise Refusal("fidelity", f"{sid}: coords differ")
            if row["dtype"] == "float32":
                stats, det = compare_arrays(new, np.ascontiguousarray(old), what=f"fidelity:{sid}")
                stats["class"] = classify_fp32(stats, band_abs, band_rel)
                stats["rows_out_of_band"] = int(((det["abs_row"] > band_abs) | (det["rel"] > band_rel)).sum())
                stats["rows_out_of_class_ii"] = int(((det["rel"] > 1e-2) | (det["cos"] < 0.9999)).sum())
                fp32[sid] = stats
            elif row["dtype"] == "float16":
                new16 = np.ascontiguousarray(new.astype(np.float16))
                old16 = np.ascontiguousarray(np.asarray(old))
                if old16.dtype != np.float16:
                    raise Refusal("fidelity", f"{sid}: old dtype {old16.dtype} != float16")
                stats, det = compare_arrays(new16, old16, what=f"fidelity:{sid}")
                stats["rows_out_of_band"] = int(((det["abs_row"] > band_abs) | (det["rel"] > band_rel)).sum())
                dim_mean = det["d"].mean(axis=0)
                stats["per_dim_mean_signed_diff_summary"] = {
                    "min": float(dim_mean.min()), "max": float(dim_mean.max()),
                    "mean": float(dim_mean.mean()), "abs_max_dim": int(np.abs(dim_mean).argmax())}
                pooled_dim_sum += det["d"].sum(axis=0)
                pooled_rows += int(new16.shape[0])
                for k, v in stats["ulp_histogram_elements"].items():
                    pooled_ulp[k] += v
                fp16[sid] = stats
            else:
                raise Refusal("fidelity", f"{sid}: unexpected dtype {row['dtype']}")
        fp16_med = np.array([s["median_rel_l2"] for s in fp16.values()], dtype=np.float64)
        report = {
            "schema": SCHEMA + ".fidelity", "steps": ["4.1", "4.2"],
            "calibration_report": str(a.calibration_report),
            "calibration_report_sha256": cal_sha,
            "band_abs": band_abs, "band_rel": band_rel,
            "class_rule": {"i": "bit_identical or (max_abs<=band_abs and max_rel_l2<=band_rel)",
                           "ii": "max_rel_l2<=1e-2 and min_cosine>=0.9999", "iii": "otherwise"},
            "fp32": {"n": len(fp32), "counts": {c: sum(1 for s in fp32.values() if s["class"] == c)
                                                 for c in ("i", "ii", "iii")},
                     "per_slide": fp32},
            "fp16": {"n": len(fp16), "per_slide": fp16,
                     "pooled_ulp_histogram_elements": pooled_ulp,
                     "pooled_per_dim_mean_signed_diff": (pooled_dim_sum / pooled_rows).tolist() if pooled_rows else None,
                     "pooled_rows": pooled_rows,
                     "median_rel_l2_distribution": {q: float(np.quantile(fp16_med, float(q)))
                                                    for q in ("0.0", "0.25", "0.5", "0.75", "1.0")} if fp16_med.size else None,
                     "min_cosine_distribution": {q: float(np.quantile([s["min_cosine"] for s in fp16.values()], float(q)))
                                                 for q in ("0.0", "0.25", "0.5", "0.75", "1.0")} if fp16 else None,
                     "slides_by_frac_rows_le_2ulp": {
                         ">=0.99": sum(1 for s in fp16.values() if s["frac_rows_le_2ulp"] >= 0.99),
                         "0.5-0.99": sum(1 for s in fp16.values() if 0.5 <= s["frac_rows_le_2ulp"] < 0.99),
                         "<0.5": sum(1 for s in fp16.values() if s["frac_rows_le_2ulp"] < 0.5)},
                     "bimodality_log10_median_rel_l2": bimodality_summary(np.log10(fp16_med)) if fp16_med.size else None,
                     "note": "descriptive only; fp16 comparison casts new fp32 -> fp16"},
            "inventory_sha256": a.inventory_sha256, "new_features_dir": str(a.new_features_dir),
            "n_slides": len(rows), "written_at": utc_now()}
    except Refusal as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        return RC_REFUSED
    text = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    X._publish_bytes(a.out, text)
    print(sha256_bytes(text))
    return RC_OK


# --------------------------------------------------------------------------
# check-mosaic (A-F8 / B-N8)
# --------------------------------------------------------------------------


def run_check_mosaic(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(prog="check-mosaic")
    p.add_argument("--mosaic", required=True, type=Path)
    p.add_argument("--mosaic-sha256", default=MOSAIC_SHA256)
    p.add_argument("--mosaic-bytes", type=int, default=MOSAIC_BYTES)
    p.add_argument("--features-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(list(argv))
    try:
        data = read_checked(a.mosaic, a.mosaic_sha256, a.mosaic_bytes, "mosaic")
        mosaic = pickle.loads(data)
        if not isinstance(mosaic, list):
            raise Refusal("mosaic", f"mosaic pickle is {type(mosaic).__name__}, not list")
        per = {}
        n_ok = 0
        for entry in mosaic:
            fname, tiles = entry[0], entry[1]
            sid = Path(str(fname)).stem
            want = [(int(t["wsi_loc"][0]), int(t["wsi_loc"][1])) for t in tiles]
            pkl = Path(a.features_dir) / f"{sid}.pkl"
            if not pkl.is_file():
                per[sid] = {"status": "missing_pickle"}
                continue
            with open(pkl, "rb") as fh:
                _, coords = pickle.load(fh)
            got = locs_of(coords)
            ok = got == want
            n_ok += int(ok)
            per[sid] = {"status": "equal" if ok else "mismatch", "n_mosaic": len(want), "n_pickle": len(got)}
        all_ok = n_ok == len(mosaic) and len(mosaic) > 0
        report = {"schema": SCHEMA + ".check_mosaic", "mosaic": str(a.mosaic), "mosaic_sha256": a.mosaic_sha256,
                  "mosaic_bytes": a.mosaic_bytes, "features_dir": str(a.features_dir),
                  "n_mosaic_slides": len(mosaic), "n_equal": n_ok, "all_equal": all_ok,
                  "per_slide": per, "written_at": utc_now()}
    except Refusal as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        return RC_REFUSED
    X._publish_bytes(a.out, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print("ALL_EQUAL" if all_ok else "MISMATCH", file=sys.stderr)
    return RC_OK if all_ok else RC_SLIDE_FAILURES


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

SUBCOMMANDS: Dict[str, Callable[[Sequence[str]], int]] = {
    "extract": lambda argv: run_extract(argv),
    "make-params": run_make_params,
    "extend-allowlist": run_extend_allowlist,
    "seed-allowlist": run_seed_allowlist,
    "make-code-manifest": run_make_code_manifest,
    "calibrate": run_calibrate,
    "fidelity": run_fidelity,
    "check-mosaic": run_check_mosaic,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in SUBCOMMANDS:
        print(f"usage: r2_1_virchow2_fp32_extract_cli {{{'|'.join(SUBCOMMANDS)}}} ...", file=sys.stderr)
        return RC_REFUSED
    return SUBCOMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
