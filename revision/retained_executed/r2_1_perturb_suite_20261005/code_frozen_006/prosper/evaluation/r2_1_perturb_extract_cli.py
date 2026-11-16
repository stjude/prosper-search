"""R2.1 stimulus-driven perturbation feature extractor (Fig. 4A / Table S17).

For each slide of the frozen R2.1 query population and each of the twelve
conditions of the frozen stimulus manifest, encode every mosaic tile with the
condition applied. Condition semantics follow the historical robustness harness
(``comet-robustness/src/lib_perturbation_univ2.py:126-190`` and
``lib_perturbation.py:110-174``); the only intended difference is the parameter
source: brightness/saturation/contrast factors and spatial-shift offsets are
taken verbatim from the manifest, never drawn here.

* tile        = ``read_region((x, y), 0, (512, 512)).convert("RGB")``
* identity    = untreated tile (the main-bank recipe)
* rotate_*    = PIL ``rotate(angle, expand=True)``
* flip_*      = PIL ``transpose(FLIP_LEFT_RIGHT / FLIP_TOP_BOTTOM)``
* colour      = ``torchvision.transforms.functional.adjust_*`` on the PIL tile
* spatial_shift = a second read at ``(x + x_shift, y + y_shift)``
* reinhard    = torchstain torch-backend Reinhard fitted on template41.png,
                same call pattern as the harness (template via matplotlib
                imread [:, :, :3]; ``ToTensor()*255``; HWC uint8 output ->
                ``permute(2,0,1)`` -> ``ToPILImage``)
* gaussian_blur = ``gaussian_blur(kernel, sigma)`` applied AFTER the encoder's
                preprocessing transform, on the normalized tensor

Output per encoder/slide: ``features/<encoder>/<slide_id>.pkl`` =
``(dict condition -> float32 (n_tiles, D), coords)`` plus a
``<slide_id>.record.json`` completion record. Outputs are never overwritten.

Subcommand ``pilot-report`` compares the identity arm against saved banks.
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
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from . import paths

CONDITIONS: Tuple[str, ...] = (
    "identity", "rotate_90", "rotate_180", "rotate_270",
    "flip_horizontal", "flip_vertical", "brightness", "saturation",
    "contrast", "spatial_shift", "reinhard", "gaussian_blur",
)
FIXED_PARAMETERS: Dict[str, dict] = {
    "identity": {"treatment": "untreated"},
    "rotate_90": {"angle": 90},
    "rotate_180": {"angle": 180},
    "rotate_270": {"angle": 270},
    "flip_horizontal": {"axis": "horizontal"},
    "flip_vertical": {"axis": "vertical"},
}
VARIABLE_KEYS: Dict[str, set] = {
    "brightness": {"factor"},
    "saturation": {"factor"},
    "contrast": {"factor"},
    "spatial_shift": {"x_shift", "y_shift"},
    "reinhard": {"template_sha256"},
    "gaussian_blur": {"application_stage", "kernel_size", "sigma"},
}

_HAORAN = Path("/PROSPER_WORKSPACE")
DEFAULT_TEMPLATE = _HAORAN / "comet-robustness" / "fig" / "template41.png"
TEMPLATE_SHA256 = "2ec8a01c09af3b9ac357bfaa812c2c0b42f4fe332baf423b44df99f8d1d1db09"
MANIFEST_SHA256 = "014513af5920ed621cfad2dc5cf527747936bb21c1db8336d282a187d669d6ce"
QUERIES_SHA256 = "5ff0ea3bd2999a4dbebb4c73e679d15870450d38f5bb6fc91be071ee72cbe471"
INPUT_SUBDIR = "r2_1_cpu_20261002/perturb_attempt_001"
RUN_SUBDIR = "r2_1_perturb_20261003"
TORCHSTAIN_SUBDIR = "r2_1_mosaic_stain_review_20261003/torchstain_runtime_001"
TORCHSTAIN_VERSION = "1.4.1"
# Pinned torchstain 1.4.1 members on the Reinhard path (same digests as the
# accepted r2_1_hoptimus_reinhard.PACKAGE_MEMBERS pin).
TORCHSTAIN_MEMBERS = {
    "torchstain/torch/normalizers/reinhard.py":
        "411d237be788a9f97bb6d31869ed826303fbfd3f663bd97eb57491f483e74031",
    "torchstain/torch/utils/lab2rgb.py":
        "aab1885c752fc270cfd1770e6f7feae1a7dce86fd3b820d674100911d2e9e0c0",
    "torchstain/torch/utils/rgb2lab.py":
        "2e3214a2cf01a8d854c85b5e2265325291fc33c5814832d2a934cf37ce7bec2b",
    "torchstain/torch/utils/split.py":
        "3ebc25b1e3f6555e396bd211486fb54809e14f5a201aeefa633dbd1cc7307bc1",
    "torchstain/torch/utils/stats.py":
        "5fc037b61babd1ffc917ce14a903b51b1e813c2599562c14c2aa2c8b7b667353",
}

# Saved banks used only by the pilot identity-fidelity report.
SAVED_BANKS = {
    "uni2": _HAORAN / "comet-feature-gen" / "comet-uni2" / "data" / "tmp_final",
    "virchow2": _HAORAN / "comet-feature-gen" / "comet-virchow2" / "data" / "tmp_final",
}
HISTORICAL_PERTURBED = {
    "uni2": _HAORAN / "comet-robustness" / "data" / "tmp_final_uni2",
    "virchow2": _HAORAN / "comet-robustness" / "data" / "tmp_final_virchow2",
}
HISTORICAL_NAMES = {"flip_horizontal": "flip_h", "flip_vertical": "flip_v",
                    "gaussian_blur": "blur"}


class InputIntegrityError(RuntimeError):
    """A frozen input (manifest, queries, bank pickle, template) failed verification."""


class TileReadError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _publish_bytes(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` atomically; never replace an existing file."""
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{time.time_ns()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp, dest)          # FileExistsError if dest exists
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(obj, sort_keys=True, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def _git_state(repo: Path) -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                               capture_output=True, text=True, timeout=20).stdout
        return {"head": head or None, "worktree_dirty": bool(dirty.strip())}
    except Exception:                                    # noqa: BLE001
        return {"head": None, "worktree_dirty": None}


# --------------------------------------------------------------------------
# Frozen inputs: manifest, query population, bank coordinates
# --------------------------------------------------------------------------


def _validate_parameters(condition: str, params: dict) -> None:
    if not isinstance(params, dict):
        raise InputIntegrityError(f"{condition}: parameters must be an object")
    if condition in FIXED_PARAMETERS:
        exp = FIXED_PARAMETERS[condition]
        if params != exp or any(type(params[k]) is not type(v) for k, v in exp.items()):
            raise InputIntegrityError(f"{condition}: parameters {params!r} != {exp!r}")
        return
    if set(params) != VARIABLE_KEYS[condition]:
        raise InputIntegrityError(f"{condition}: parameter keys {sorted(params)}")
    if condition in ("brightness", "saturation", "contrast"):
        f = params["factor"]
        if type(f) is not float or not math.isfinite(f) or f <= 0:
            raise InputIntegrityError(f"{condition}: bad factor {f!r}")
    elif condition == "spatial_shift":
        if any(type(params[k]) is not int for k in ("x_shift", "y_shift")):
            raise InputIntegrityError("spatial_shift: offsets must be integers")
    elif condition == "reinhard":
        d = params["template_sha256"]
        if type(d) is not str or len(d) != 64:
            raise InputIntegrityError("reinhard: bad template digest")
    elif condition == "gaussian_blur":
        k, s = params["kernel_size"], params["sigma"]
        if (params["application_stage"] != "post_preprocessing" or type(k) is not int
                or k < 1 or k % 2 == 0 or type(s) is not float or not s > 0):
            raise InputIntegrityError(f"gaussian_blur: bad parameters {params!r}")


def load_manifest(path: Path, expected_sha256: Optional[str]) -> Tuple[str, Dict[str, Dict[str, dict]]]:
    """Return (sha256, slide_id -> condition -> verbatim record)."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise InputIntegrityError(f"manifest sha256 {digest} != expected {expected_sha256}")
    doc = json.loads(raw)
    by_slide: Dict[str, Dict[str, dict]] = {}
    for rec in doc["records"]:
        sid, cond = rec["slide_id"], rec["condition"]
        if type(sid) is not str or cond not in CONDITIONS:
            raise InputIntegrityError(f"bad manifest record {rec!r}")
        if rec.get("level") != 0 or rec.get("patch_size") != 512:
            raise InputIntegrityError(f"{sid}/{cond}: level/patch_size must be 0/512")
        _validate_parameters(cond, rec["parameters"])
        slot = by_slide.setdefault(sid, {})
        if cond in slot:
            raise InputIntegrityError(f"{sid}: duplicate condition {cond}")
        slot[cond] = rec
    for sid, slot in by_slide.items():
        if set(slot) != set(CONDITIONS):
            raise InputIntegrityError(f"{sid}: conditions {sorted(slot)}")
    return digest, by_slide


def load_queries(path: Path, expected_sha256: Optional[str]) -> Tuple[str, List[dict]]:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise InputIntegrityError(f"queries sha256 {digest} != expected {expected_sha256}")
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    if not rows or len({r["slide_id"] for r in rows}) != len(rows):
        raise InputIntegrityError("queries.csv empty or slide_id not unique")
    return digest, rows


def load_bank_coords(row: dict) -> List[dict]:
    """Mosaic coordinates from the sha256-verified saved H-Optimus bank pickle."""
    path = Path(row["resolved_path"])
    size = path.stat().st_size
    if size != int(row["feature_size"]):
        raise InputIntegrityError(f"{path}: size {size} != {row['feature_size']}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != row["feature_sha256"]:
        raise InputIntegrityError(f"{path}: sha256 {digest} != {row['feature_sha256']}")
    feats, coords = pickle.loads(data)
    n = int(row["feature_rows"])
    if len(coords) != n or getattr(feats, "shape", (None,))[0] != n:
        raise InputIntegrityError(f"{path}: {len(coords)} coords for {n} declared rows")
    for c in coords:
        loc = c["wsi_loc"]
        if len(loc) != 2 or any(isinstance(v, bool) or int(v) != v for v in loc):
            raise InputIntegrityError(f"{path}: bad coordinate {c!r}")
    return coords


# --------------------------------------------------------------------------
# Reinhard (torchstain torch backend, harness call pattern)
# --------------------------------------------------------------------------


def import_torchstain(prefix: Optional[Path]):
    """Import the pinned private torchstain 1.4.1 and verify its Reinhard path."""
    if prefix is not None:
        prefix = Path(prefix).resolve()
        if str(prefix) not in sys.path:
            sys.path.insert(0, str(prefix))
    import importlib.metadata as md
    import torchstain
    pkg_root = Path(torchstain.__file__).resolve().parent.parent
    if prefix is not None and pkg_root != prefix:
        raise InputIntegrityError(f"torchstain imported from {pkg_root}, not {prefix}")
    version = md.version("torchstain")
    if version != TORCHSTAIN_VERSION:
        raise InputIntegrityError(f"torchstain {version} != {TORCHSTAIN_VERSION}")
    for member, digest in TORCHSTAIN_MEMBERS.items():
        if sha256_file(pkg_root / member) != digest:
            raise InputIntegrityError(f"torchstain member hash mismatch: {member}")
    import torchstain.torch.normalizers  # noqa: F401  (loads backend)
    return torchstain, {"distribution_version": version, "prefix": str(pkg_root),
                        "module_version": getattr(torchstain, "__version__", None)}


def _times_255(x):
    return x * 255


class TorchstainReinhard:
    """Harness-equivalent Reinhard: ``normalizer.normalize(I=T(tile))``.

    ``__call__`` executes TorchReinhardNormalizer.normalize (method=None)
    step for step with the library's own primitives, so the output bytes are
    those of the library call while the transferred Lab can be checked for
    non-finite values (zero-variance tiles) before the uint8 cast.
    """

    def __init__(self, template_path: Path, template_sha256: str, prefix: Optional[Path]):
        actual = sha256_file(template_path)
        if actual != template_sha256:
            raise InputIntegrityError(f"template sha256 {actual} != {template_sha256}")
        ts, self.info = import_torchstain(prefix)
        import matplotlib.image as mpimg
        from torchvision import transforms
        from torchstain.torch.utils.lab2rgb import lab2rgb
        from torchstain.torch.utils.rgb2lab import rgb2lab
        from torchstain.torch.utils.split import lab_merge, lab_split
        from torchstain.torch.utils.stats import get_mean_std, standardize
        self._T = transforms.Compose([transforms.ToTensor(), transforms.Lambda(_times_255)])
        self._to_pil = transforms.ToPILImage()
        template = mpimg.imread(str(template_path))[:, :, :3]
        self.normalizer = ts.normalizers.ReinhardNormalizer(backend="torch")
        self.normalizer.fit(self._T(template))
        self._p = (rgb2lab, lab_split, lab_merge, lab2rgb, get_mean_std, standardize)
        self.info.update({"template_path": str(template_path), "template_sha256": actual,
                          "template_decode": "matplotlib.image.imread[:, :, :3]",
                          "target_means": [float(v) for v in self.normalizer.target_means],
                          "target_stds": [float(v) for v in self.normalizer.target_stds]})

    def __call__(self, tile: Image.Image) -> Tuple[Image.Image, bool]:
        rgb2lab, lab_split, lab_merge, lab2rgb, get_mean_std, standardize = self._p
        nz = self.normalizer
        I = self._T(tile).type(torch.float32) / 255
        lab = rgb2lab(I)
        labs = lab_split(lab)
        stack_ = torch.tensor([get_mean_std(x) for x in labs])
        mus, stds = stack_[:, 0], stack_[:, 1]
        result = [standardize(x, mu_, std_) * std_T + mu_T for x, mu_, std_, mu_T, std_T
                  in zip(labs, mus, stds, nz.target_means, nz.target_stds)]
        finite = all(bool(torch.isfinite(r).all()) for r in result)
        out = (lab2rgb(lab_merge(*result)) * 255).type(torch.uint8)
        return self._to_pil(out.permute(2, 0, 1)), finite


# --------------------------------------------------------------------------
# Condition semantics (pure; exercised directly by the tests)
# --------------------------------------------------------------------------


def shift_of(stimuli: Dict[str, dict]) -> Tuple[int, int]:
    p = stimuli["spatial_shift"]["parameters"]
    return int(p["x_shift"]), int(p["y_shift"])


def condition_images(region: Image.Image, shifted: Image.Image,
                     stimuli: Dict[str, dict], reinhard: Callable) -> Tuple[Dict[str, Image.Image], bool]:
    """PIL image per condition before encoder preprocessing.

    ``gaussian_blur`` maps to the untreated tile: blur is applied after the
    encoder transform (see :func:`encode_conditions`).
    """
    from torchvision.transforms import functional as F
    p = {c: stimuli[c]["parameters"] for c in CONDITIONS}
    out: Dict[str, Image.Image] = {"identity": region}
    for c in ("rotate_90", "rotate_180", "rotate_270"):
        out[c] = region.rotate(p[c]["angle"], expand=True)
    out["flip_horizontal"] = region.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    out["flip_vertical"] = region.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    out["brightness"] = F.adjust_brightness(region, p["brightness"]["factor"])
    out["saturation"] = F.adjust_saturation(region, p["saturation"]["factor"])
    out["contrast"] = F.adjust_contrast(region, p["contrast"]["factor"])
    out["spatial_shift"] = shifted
    out["reinhard"], finite = reinhard(region)
    out["gaussian_blur"] = region
    return out, finite


def encode_conditions(images: Dict[str, Image.Image], transform: Callable,
                      stimuli: Dict[str, dict]) -> torch.Tensor:
    """(12, C, H, W) preprocessed tensors in CONDITIONS order."""
    from torchvision.transforms import functional as F
    base = transform(images["identity"])
    tensors = []
    for c in CONDITIONS:
        if c == "identity":
            t = base
        elif c == "gaussian_blur":
            g = stimuli[c]["parameters"]
            k, s = g["kernel_size"], g["sigma"]
            # == torchvision GaussianBlur(kernel_size=3, sigma=0.5)(transform(tile))
            t = F.gaussian_blur(base, [k, k], [s, s])
        else:
            t = transform(images[c])
        tensors.append(t)
    return torch.stack(tensors)


# --------------------------------------------------------------------------
# Tile dataset: one reader handle per DataLoader worker
# --------------------------------------------------------------------------


def openslide_opener(path: str):
    import openslide
    return openslide.OpenSlide(path)


class SlideTiles(torch.utils.data.Dataset):
    def __init__(self, slide_path: str, coords: List[dict], stimuli: Dict[str, dict],
                 transforms: Dict[str, Callable], reinhard: Callable,
                 opener: Callable = openslide_opener, patch_size: int = 512):
        self.slide_path = str(slide_path)
        self.locs = [(int(c["wsi_loc"][0]), int(c["wsi_loc"][1])) for c in coords]
        self.stimuli = stimuli
        self.transforms = transforms
        self.reinhard = reinhard
        self.opener = opener
        self.patch_size = patch_size
        self.shift = shift_of(stimuli)
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
            dx, dy = self.shift
            shifted = self._slide.read_region((x + dx, y + dy), 0, (ps, ps)).convert("RGB")
        except Exception as exc:                          # noqa: BLE001
            return {"index": i, "error": {"type": type(exc).__name__, "message": str(exc),
                                          "wsi_loc": [x, y]}}
        images, finite = condition_images(region, shifted, self.stimuli, self.reinhard)
        return {"index": i, "reinhard_finite": finite,
                "tensors": {n: encode_conditions(images, tf, self.stimuli)
                            for n, tf in self.transforms.items()}}


def _collate(batch):
    return batch


def _worker_init(_):
    torch.set_num_threads(1)


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------


@dataclass
class EncoderRuntime:
    name: str
    model: Any
    transform: Callable
    forward: Callable[[torch.Tensor], torch.Tensor]
    amp: bool
    dim: int
    meta: dict = field(default_factory=dict)


AMP_DEFAULTS = {"h_optimus_1": True, "uni2": False, "virchow2": False}
HF_WEIGHTS = {
    "h_optimus_1": ("bioptimus/H-optimus-1", ("model.safetensors", "pytorch_model.bin")),
    "uni2": ("MahmoodLab/UNI2-h", ("pytorch_model.bin", "model.safetensors")),
    "virchow2": ("paige-ai/Virchow2", ("model.safetensors", "pytorch_model.bin")),
}
RECIPES = {
    "h_optimus_1": ("prosper.embeddings.tile_features._build_h_optimus_1 (accepted main-bank "
                    "builder); extract_slide forward: fp16 CUDA autocast when amp, "
                    "feats.float().cpu() -> float32; 1536-d model output"),
    "uni2": ("prosper.embeddings.tile_features._build_uni2 (== comet-uni2/src/lib.py:57-79 "
             "load_univ2_model); forward model(x) under no_grad, no autocast "
             "(lib.py:124 and lib_perturbation_univ2.py:114); 1536-d"),
    "virchow2": ("prosper.embeddings.patch_embedding.virchow2_embeddings."
                 "Virchow2EmbeddingExtractor.load_model (== comet-new-scans/src/"
                 "create_virchow2_features/lib.py:14-19, lib_perturbation.py:23-26); forward "
                 "model(x) no autocast; cat(tokens[:,0], tokens[:,5:].mean(1)) -> 2560-d "
                 "(lib.py:68-73, lib_perturbation.py:96-99). Saved main bank keeps the "
                 "first 1280 (CLS) only"),
}


def _hf_weights_info(name: str, hash_weights: bool) -> dict:
    repo, files = HF_WEIGHTS[name]
    info: dict = {"repo_id": repo}
    try:
        from huggingface_hub import try_to_load_from_cache
        for fn in files:
            p = try_to_load_from_cache(repo, fn)
            if isinstance(p, str):
                pp = Path(p)
                parts = pp.parts
                rev = parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
                real = Path(os.path.realpath(p))
                info.update({"file": fn, "snapshot_revision": rev, "cache_path": p,
                             "blob": str(real), "bytes": real.stat().st_size})
                if hash_weights:
                    info["sha256"] = sha256_file(real)
                break
    except Exception as exc:                              # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def build_encoder(name: str, device: str, amp: bool, hash_weights: bool = True) -> EncoderRuntime:
    if name in ("h_optimus_1", "uni2"):
        from ..embeddings import tile_features as T
        model, transform = T.load_encoder(name, device=device)
        forward = model
        dim = T.REGISTRY[name].dim
    elif name == "virchow2":
        from ..embeddings.patch_embedding.virchow2_embeddings import Virchow2EmbeddingExtractor
        model, transform = Virchow2EmbeddingExtractor(device=device, verbose=False).load_model()

        def forward(x):
            tokens = model(x)
            return torch.cat([tokens[:, 0], tokens[:, 5:].mean(1)], dim=-1)
        dim = 2560
    else:
        raise KeyError(f"unknown encoder {name!r}")
    model.eval()
    meta = {"recipe": RECIPES[name], "transform": repr(transform),
            "pretrained_cfg": {k: v for k, v in (getattr(model, "pretrained_cfg", {}) or {}).items()
                               if isinstance(v, (str, int, float, bool, list, tuple, type(None)))},
            "weights": _hf_weights_info(name, hash_weights),
            "autocast_dtype": "float16" if amp else None, "storage_dtype": "float32"}
    if name == "virchow2":
        meta["token_slicing"] = "cat(tokens[:,0], tokens[:,5:].mean(1))"
    return EncoderRuntime(name, model, transform, forward, amp, dim, meta)


def _forward(rt: EncoderRuntime, x: torch.Tensor, device: str) -> np.ndarray:
    x = x.to(device, non_blocking=True)
    with torch.no_grad():
        if rt.amp and device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.float16):
                f = rt.forward(x)
        else:
            f = rt.forward(x)
    return f.float().cpu().numpy()


# --------------------------------------------------------------------------
# Per-slide processing
# --------------------------------------------------------------------------


def output_paths(run_root: Path, encoder: str, slide_id: str) -> Tuple[Path, Path]:
    d = run_root / "features" / encoder
    return d / f"{slide_id}.pkl", d / f"{slide_id}.record.json"


def existing_status(pkl: Path, rec: Path, *, slide_id: str, encoder: str,
                    manifest_sha256: str, n_tiles: int) -> Tuple[str, str]:
    """('absent'|'complete'|'invalid', reason)."""
    has_p, has_r = pkl.exists(), rec.exists()
    if not has_p and not has_r:
        return "absent", ""
    if not (has_p and has_r):
        return "invalid", f"orphan output (pickle={has_p}, record={has_r})"
    try:
        r = json.loads(rec.read_text())
    except Exception as exc:                              # noqa: BLE001
        return "invalid", f"unreadable record: {exc}"
    checks = [(r.get("slide_id") == slide_id, "slide_id"),
              (r.get("encoder") == encoder, "encoder"),
              (r.get("manifest_sha256") == manifest_sha256, "manifest_sha256"),
              (r.get("n_tiles") == n_tiles, "n_tiles"),
              (r.get("pickle_bytes") == pkl.stat().st_size, "pickle_bytes")]
    for ok, what in checks:
        if not ok:
            return "invalid", f"record {what} mismatch"
    if sha256_file(pkl) != r.get("pickle_sha256"):
        return "invalid", "pickle sha256 != record"
    return "complete", ""


@dataclass
class RunContext:
    run_root: Path
    manifest_path: Path
    manifest_sha256: str
    queries_path: Path
    queries_sha256: str
    device: str
    forward_batch: int
    loader_batch: int
    num_workers: int
    prefetch_factor: int
    reinhard: Callable
    reinhard_info: dict
    opener: Callable = openslide_opener
    slide_path_of: Callable[[str], Path] = paths.slide_wsi_path
    limit_tiles: int = 0
    job_meta: dict = field(default_factory=dict)


def _fail(ctx: RunContext, **obj) -> None:
    obj.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    obj.setdefault("lsf_job", os.environ.get("LSB_JOBID"))
    obj.setdefault("lsf_index", os.environ.get("LSB_JOBINDEX"))
    obj.setdefault("host", socket.gethostname())
    append_jsonl(ctx.run_root / "failures.jsonl", obj)
    print(f"[perturb] FAILURE {json.dumps(obj, default=str)[:600]}", flush=True)


def process_slide(ctx: RunContext, row: dict, stimuli: Dict[str, dict],
                  encoders: Dict[str, EncoderRuntime]) -> Dict[str, str]:
    """Returns encoder -> 'written' | 'skipped' | 'failed'."""
    sid = row["slide_id"]
    status: Dict[str, str] = {}
    n_decl = int(row["feature_rows"])
    todo: Dict[str, EncoderRuntime] = {}
    for name, rt in encoders.items():
        pkl, rec = output_paths(ctx.run_root, name, sid)
        st, why = existing_status(pkl, rec, slide_id=sid, encoder=name,
                                  manifest_sha256=ctx.manifest_sha256,
                                  n_tiles=min(n_decl, ctx.limit_tiles) if ctx.limit_tiles else n_decl)
        if st == "complete":
            status[name] = "skipped"
        elif st == "invalid":
            status[name] = "failed"
            _fail(ctx, slide_id=sid, encoder=name, stage="existing_output", reason=why)
        else:
            todo[name] = rt
    if not todo:
        return status

    def fail_all(**kw):
        for name in todo:
            status[name] = "failed"
            _fail(ctx, slide_id=sid, encoder=name, **kw)

    try:
        coords = load_bank_coords(row)
    except Exception as exc:                              # noqa: BLE001
        fail_all(stage="bank_coords", reason=f"{type(exc).__name__}: {exc}")
        return status
    if ctx.limit_tiles:
        coords = coords[:ctx.limit_tiles]
    slide_path = ctx.slide_path_of(sid)
    try:
        h = ctx.opener(str(slide_path))
        if hasattr(h, "close"):
            h.close()
    except Exception as exc:                              # noqa: BLE001
        fail_all(stage="slide_open", reason=f"{type(exc).__name__}: {exc}",
                 slide_path=str(slide_path))
        return status

    n = len(coords)
    ds = SlideTiles(str(slide_path), coords, stimuli,
                    {k: rt.transform for k, rt in todo.items()}, ctx.reinhard, ctx.opener)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=ctx.loader_batch, shuffle=False, num_workers=ctx.num_workers,
        collate_fn=_collate, worker_init_fn=_worker_init if ctx.num_workers else None,
        prefetch_factor=ctx.prefetch_factor if ctx.num_workers else None,
        persistent_workers=False)
    out = {k: {c: np.empty((n, rt.dim), np.float32) for c in CONDITIONS} for k, rt in todo.items()}
    filled = np.zeros(n, bool)
    tile_errors: List[dict] = []
    nonfinite_reinhard: List[int] = []
    fwd_seconds = {k: 0.0 for k in todo}
    pending: List[dict] = []
    t0 = time.time()

    def encode(items: List[dict]) -> None:
        # Condition-homogeneous forwards over consecutive mosaic tiles, so the
        # identity batches are the same 256-tile chunks the main bank used.
        idx = np.array([it["index"] for it in items])
        for name, rt in todo.items():
            stack = torch.stack([it["tensors"][name] for it in items])  # (B,12,C,H,W)
            ts = time.time()
            for j, c in enumerate(CONDITIONS):
                feats = _forward(rt, stack[:, j].contiguous(), ctx.device)
                if feats.shape != (len(idx), rt.dim):
                    raise RuntimeError(f"{name}: feature shape {feats.shape}")
                out[name][c][idx] = feats
            fwd_seconds[name] += time.time() - ts
        filled[idx] = True

    try:
        for batch in dl:
            for it in batch:
                if "error" in it:
                    tile_errors.append(it["error"] | {"index": it["index"]})
                    pending.clear()       # keep draining only to enumerate failures
                    continue
                if not it["reinhard_finite"]:
                    nonfinite_reinhard.append(it["index"])
                if not tile_errors:
                    pending.append(it)
            while len(pending) >= ctx.forward_batch:
                encode(pending[:ctx.forward_batch])
                del pending[:ctx.forward_batch]
        if pending and not tile_errors:
            encode(pending)
        pending.clear()
    except Exception as exc:                              # noqa: BLE001
        fail_all(stage="extract", reason=f"{type(exc).__name__}: {exc}",
                 traceback=traceback.format_exc()[-4000:])
        return status
    wall = time.time() - t0
    if tile_errors:
        fail_all(stage="tile_read", n_failed_tiles=len(tile_errors), n_tiles=n,
                 tile_errors=tile_errors[:200])
        return status
    if not filled.all():
        fail_all(stage="extract", reason=f"{int((~filled).sum())} tiles not encoded")
        return status

    stim_list = [stimuli[c] for c in CONDITIONS]
    for name, rt in todo.items():
        bad = {c: int((~np.isfinite(a)).any(axis=1).sum()) for c, a in out[name].items()}
        if any(bad.values()):
            status[name] = "failed"
            _fail(ctx, slide_id=sid, encoder=name, stage="nonfinite_features",
                  nonfinite_rows=bad)
            continue
        payload = pickle.dumps((out[name], coords), protocol=4)
        pkl, rec = output_paths(ctx.run_root, name, sid)
        pkl.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "r2_1_perturb_record_v1",
            "slide_id": sid, "encoder": name, "n_tiles": n, "dim": rt.dim,
            "conditions": list(CONDITIONS),
            "stimuli": stim_list,
            "manifest_path": str(ctx.manifest_path), "manifest_sha256": ctx.manifest_sha256,
            "queries_path": str(ctx.queries_path), "queries_sha256": ctx.queries_sha256,
            "query_row": {k: row[k] for k in ("population_index", "metadata_index", "slide_id",
                                              "feature_rows", "resolved_path", "feature_sha256")},
            "coords_source": row["resolved_path"], "coords_source_sha256": row["feature_sha256"],
            "limit_tiles": ctx.limit_tiles or None,
            "slide_path": str(slide_path),
            "reinhard": ctx.reinhard_info,
            "reinhard_nonfinite_tile_indices": nonfinite_reinhard,
            "encoder_meta": rt.meta, "amp": rt.amp,
            "forward_batch": ctx.forward_batch, "loader_batch": ctx.loader_batch,
            "num_workers": ctx.num_workers, "device": ctx.device,
            "slide_wall_seconds": round(wall, 3),
            "encoder_forward_seconds": round(fwd_seconds[name], 3),
            "tiles_per_second_slide": round(n / wall, 3) if wall > 0 else None,
            "images_per_second_forward": (round(n * len(CONDITIONS) / fwd_seconds[name], 3)
                                          if fwd_seconds[name] > 0 else None),
            "pickle_sha256": hashlib.sha256(payload).hexdigest(),
            "pickle_bytes": len(payload),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **ctx.job_meta,
        }
        try:
            _publish_bytes(pkl, payload)
            _publish_bytes(rec, (json.dumps(record, indent=1, sort_keys=True, default=str)
                                 + "\n").encode())
            status[name] = "written"
        except FileExistsError as exc:
            status[name] = "failed"
            _fail(ctx, slide_id=sid, encoder=name, stage="publish",
                  reason=f"refused to overwrite: {exc}")
    print(f"[perturb] {sid}: n={n} wall={wall:.1f}s ({n / max(wall, 1e-9):.2f} tiles/s) "
          + " ".join(f"{k}={v}" for k, v in status.items())
          + (f" reinhard_nonfinite={len(nonfinite_reinhard)}" if nonfinite_reinhard else ""),
          flush=True)
    return status


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _job_meta(device: str) -> dict:
    import timm
    import torchvision
    import openslide
    import PIL
    meta = {
        "host": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version.split()[0],
        "lsf_job": os.environ.get("LSB_JOBID"), "lsf_index": os.environ.get("LSB_JOBINDEX"),
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "timm": timm.__version__, "numpy": np.__version__,
                     "pillow": PIL.__version__, "openslide_python": openslide.__version__,
                     "openslide_library": getattr(openslide, "__library_version__", None),
                     "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "torch_backend_flags": {
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "main_process_num_threads": torch.get_num_threads(),
            "lsb_djob_numproc": os.environ.get("LSB_DJOB_NUMPROC")},
        "code_path": str(Path(__file__).resolve()),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "git": _git_state(Path(__file__).resolve().parents[2]),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    }
    try:
        import importlib.metadata as md
        import torchstain
        meta["versions"]["torchstain"] = md.version("torchstain")
        meta["versions"]["torchstain_module"] = getattr(torchstain, "__version__", None)
    except Exception:                                    # noqa: BLE001
        pass
    return meta


def build_parser() -> argparse.ArgumentParser:
    out = paths.output_root()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoders", default="h_optimus_1",
                   help="comma list from h_optimus_1,uni2,virchow2")
    p.add_argument("--amp", action="append", default=[], metavar="ENCODER",
                   help="force fp16 autocast for ENCODER (repeatable)")
    p.add_argument("--no-amp", action="append", default=[], metavar="ENCODER",
                   help="force fp32 for ENCODER (repeatable)")
    p.add_argument("--manifest", type=Path, default=out / INPUT_SUBDIR / "stimuli.json")
    p.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    p.add_argument("--queries", type=Path, default=out / INPUT_SUBDIR / "queries.csv")
    p.add_argument("--queries-sha256", default=QUERIES_SHA256)
    p.add_argument("--out-root", type=Path, default=out / RUN_SUBDIR)
    p.add_argument("--lo", type=int, default=1, help="1-based first queries.csv row")
    p.add_argument("--hi", type=int, default=0, help="1-based last row (0 = all)")
    p.add_argument("--slides", default="", help="comma list restricting the range")
    p.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    p.add_argument("--torchstain-prefix", type=Path, default=out / TORCHSTAIN_SUBDIR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--forward-batch", type=int, default=256)
    p.add_argument("--loader-batch", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=7)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--main-threads", type=int, default=0,
                   help="torch intra-op threads in the main process "
                        "(default on CUDA: LSB_DJOB_NUMPROC - num_workers, min 1)")
    p.add_argument("--limit-tiles", type=int, default=0,
                   help="smoke test only; refused for the default output root")
    p.add_argument("--no-hash-weights", action="store_true")
    p.add_argument("--allow-network", action="store_true",
                   help="do not force HF_HUB_OFFLINE=1")
    return p


def resolve_amp(encoders: Sequence[str], amp: Sequence[str], no_amp: Sequence[str]) -> Dict[str, bool]:
    res = {e: AMP_DEFAULTS[e] for e in encoders}
    for e in amp:
        res[e] = True
    for e in no_amp:
        if e in amp:
            raise SystemExit(f"--amp and --no-amp both given for {e}")
        res[e] = False
    unknown = (set(amp) | set(no_amp)) - set(encoders)
    if unknown:
        raise SystemExit(f"--amp/--no-amp for unselected encoders {sorted(unknown)}")
    return res


def run(argv=None, *, encoder_factory: Optional[Callable] = None,
        reinhard: Optional[Callable] = None, reinhard_info: Optional[dict] = None,
        opener: Callable = openslide_opener,
        slide_path_of: Callable[[str], Path] = paths.slide_wsi_path) -> int:
    a = build_parser().parse_args(argv)
    if not a.allow_network:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    encs = [e.strip() for e in a.encoders.split(",") if e.strip()]
    if not encs or len(set(encs)) != len(encs) or set(encs) - set(AMP_DEFAULTS):
        raise SystemExit(f"bad --encoders {a.encoders!r}")
    amp = resolve_amp(encs, a.amp, a.no_amp)
    default_root = (paths.output_root() / RUN_SUBDIR).resolve()
    if a.limit_tiles and a.out_root.resolve() == default_root:
        raise SystemExit("--limit-tiles is refused for the production output root")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable (no silent CPU fallback)")
    if a.main_threads:
        torch.set_num_threads(a.main_threads)
    elif a.device.startswith("cuda"):
        # Main process only feeds the GPU; leave the LSF allocation to workers.
        alloc = int(os.environ.get("LSB_DJOB_NUMPROC") or os.cpu_count() or 1)
        torch.set_num_threads(max(1, alloc - a.num_workers))

    manifest_sha, manifest = load_manifest(a.manifest, a.manifest_sha256 or None)
    queries_sha, rows = load_queries(a.queries, a.queries_sha256 or None)
    hi = a.hi or len(rows)
    lo = max(1, a.lo)
    sel = rows[lo - 1:hi]
    if a.slides:
        want = {s.strip() for s in a.slides.split(",") if s.strip()}
        sel = [r for r in sel if r["slide_id"] in want]
        missing = want - {r["slide_id"] for r in sel}
        if missing:
            raise SystemExit(f"--slides not in range {lo}..{hi}: {sorted(missing)}")
    for r in sel:
        if r["slide_id"] not in manifest:
            raise InputIntegrityError(f"{r['slide_id']}: no stimuli in manifest")
        rec = manifest[r["slide_id"]]["reinhard"]["parameters"]["template_sha256"]
        if rec != TEMPLATE_SHA256 and reinhard is None:
            raise InputIntegrityError(f"{r['slide_id']}: manifest template digest {rec}")

    a.out_root.mkdir(parents=True, exist_ok=True)
    if reinhard is None:
        rh = TorchstainReinhard(a.template, TEMPLATE_SHA256, a.torchstain_prefix)
        reinhard, reinhard_info = rh, rh.info
    print(f"[perturb] encoders={encs} amp={amp} rows {lo}..{hi} of {len(rows)} "
          f"({len(sel)} slides) out={a.out_root}", flush=True)

    job_meta = _job_meta(a.device)
    factory = encoder_factory or (lambda name, device, amp_flag: build_encoder(
        name, device, amp_flag, hash_weights=not a.no_hash_weights))
    t_load = time.time()
    runtimes = {e: factory(e, a.device, amp[e]) for e in encs}
    load_seconds = round(time.time() - t_load, 2)
    print(f"[perturb] models loaded in {load_seconds}s on {a.device} "
          f"gpu={job_meta.get('gpu')}", flush=True)

    ctx = RunContext(a.out_root, a.manifest, manifest_sha, a.queries, queries_sha,
                     a.device, a.forward_batch, a.loader_batch, a.num_workers,
                     a.prefetch_factor, reinhard, reinhard_info or {}, opener,
                     slide_path_of, a.limit_tiles, job_meta)
    counts = {e: {"written": 0, "skipped": 0, "failed": 0} for e in encs}
    tiles_written = {e: 0 for e in encs}
    t0 = time.time()
    for r in sel:
        st = process_slide(ctx, r, manifest[r["slide_id"]], runtimes)
        for e, s in st.items():
            counts[e][s] += 1
            if s == "written":
                tiles_written[e] += (min(int(r["feature_rows"]), a.limit_tiles)
                                     if a.limit_tiles else int(r["feature_rows"]))
    wall = time.time() - t0
    summary = {"argv": list(argv) if argv is not None else sys.argv[1:],
               "encoders": encs, "amp": amp, "rows": [lo, hi],
               "slides": [r["slide_id"] for r in sel], "counts": counts,
               "tiles_written": tiles_written, "model_load_seconds": load_seconds,
               "wall_seconds": round(wall, 2),
               "encoder_meta": {e: rt.meta for e, rt in runtimes.items()},
               "manifest_sha256": manifest_sha, "queries_sha256": queries_sha,
               "reinhard": reinhard_info, **job_meta}
    tag = (f"{os.environ.get('LSB_JOBID', 'local')}_{os.environ.get('LSB_JOBINDEX', '0')}"
           f"_{lo}-{hi}_{'+'.join(encs)}_{time.strftime('%Y%m%dT%H%M%S')}"
           f"_{os.getpid()}_{time.time_ns() % 10**9:09d}")
    rdir = a.out_root / "runs"
    rdir.mkdir(parents=True, exist_ok=True)
    _publish_bytes(rdir / f"{tag}.json",
                   (json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n").encode())
    print(f"[perturb] done counts={counts} wall={wall:.1f}s", flush=True)
    return 1 if any(c["failed"] for c in counts.values()) else 0


# --------------------------------------------------------------------------
# Pilot report: identity fidelity and condition sanity (descriptive)
# --------------------------------------------------------------------------


def _rowwise(a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {"cos": cos, "maxabs": np.abs(a - b).max(1),
            "rel_l2": np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)}


def _summ(m: Dict[str, np.ndarray]) -> dict:
    c, x, r = m["cos"], m["maxabs"], m["rel_l2"]
    q = lambda v, p: float(np.percentile(v, p))  # noqa: E731
    return {"n_tiles": int(c.size),
            "cosine": {"min": float(c.min()), "p1": q(c, 1), "median": q(c, 50),
                       "p99": q(c, 99), "max": float(c.max())},
            "frac_cosine_ge_0.999": float((c >= 0.999).mean()),
            "frac_cosine_ge_0.9999": float((c >= 0.9999).mean()),
            "maxabs": {"min": float(x.min()), "median": q(x, 50), "p99": q(x, 99),
                       "max": float(x.max())},
            "rel_l2": {"median": q(r, 50), "p99": q(r, 99), "max": float(r.max())}}


def _coords_equal(c1, c2) -> bool:
    return len(c1) == len(c2) and all(list(a["wsi_loc"]) == list(b["wsi_loc"])
                                      for a, b in zip(c1, c2))


def pilot_report(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pilot-report")
    out = paths.output_root()
    p.add_argument("--run-root", type=Path, default=out / RUN_SUBDIR)
    p.add_argument("--slides", required=True)
    p.add_argument("--encoders", default="h_optimus_1,uni2,virchow2")
    p.add_argument("--queries", type=Path, default=out / INPUT_SUBDIR / "queries.csv")
    p.add_argument("--manifest", type=Path, default=out / INPUT_SUBDIR / "stimuli.json")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    _, rows = load_queries(a.queries, QUERIES_SHA256)
    _, manifest = load_manifest(a.manifest, MANIFEST_SHA256)
    by_id = {r["slide_id"]: r for r in rows}
    slides = [s.strip() for s in a.slides.split(",") if s.strip()]
    report: dict = {"slides": slides, "identity_fidelity": {}, "historical_arms": {},
                    "condition_sanity": {}, "throughput": {}, "notes": []}
    for enc in [e.strip() for e in a.encoders.split(",") if e.strip()]:
        fid, hist, sanity, thr = {}, {}, {}, []
        pooled: Dict[str, List[np.ndarray]] = {}
        for sid in slides:
            pkl, rec = output_paths(a.run_root, enc, sid)
            if not (pkl.exists() and rec.exists()):
                fid[sid] = {"error": "missing output"}
                continue
            record = json.loads(rec.read_text())
            if sha256_file(pkl) != record["pickle_sha256"]:
                fid[sid] = {"error": "pickle sha256 mismatch"}
                continue
            feats, coords = pickle.loads(pkl.read_bytes())
            thr.append({"slide_id": sid, "n_tiles": record["n_tiles"],
                        "slide_wall_seconds": record["slide_wall_seconds"],
                        "encoder_forward_seconds": record["encoder_forward_seconds"],
                        "images_per_second_forward": record["images_per_second_forward"],
                        "gpu": record.get("gpu")})
            ident = feats["identity"]
            entry: dict = {}
            if enc == "h_optimus_1":
                row = by_id[sid]
                bank_path = Path(row["resolved_path"])
                if sha256_file(bank_path) != row["feature_sha256"]:
                    entry["error"] = "bank sha256 mismatch"
                else:
                    bf, bc = pickle.loads(bank_path.read_bytes())
                    entry["bank"] = {"path": str(bank_path), "dtype": str(bf.dtype),
                                     "shape": list(bf.shape), "sha256": row["feature_sha256"],
                                     "coords_equal": _coords_equal(coords, bc)}
                    m = _rowwise(ident, bf)
                    entry["vs_bank"] = _summ(m)
                    pooled.setdefault("vs_bank", []).append(m)
            else:
                bank_path = SAVED_BANKS[enc] / f"{sid}.pkl"
                bf, bc = pickle.loads(bank_path.read_bytes())
                entry["bank"] = {"path": str(bank_path), "dtype": str(bf.dtype),
                                 "shape": list(bf.shape), "sha256": sha256_file(bank_path),
                                 "coords_equal": _coords_equal(coords, bc)}
                ours = ident[:, :bf.shape[1]] if bf.shape[1] < ident.shape[1] else ident
                if bf.shape[1] < ident.shape[1]:
                    entry["bank"]["note"] = (f"saved bank has {bf.shape[1]} columns; compared "
                                             f"against our first {bf.shape[1]} (CLS) columns")
                m = _rowwise(ours, bf.astype(np.float32))
                entry["vs_bank"] = _summ(m)
                pooled.setdefault("vs_bank", []).append(m)
                if bf.dtype == np.float16:
                    m16 = _rowwise(ours.astype(np.float16).astype(np.float32),
                                   bf.astype(np.float32))
                    entry["vs_bank_after_casting_ours_to_float16"] = _summ(m16)
                    pooled.setdefault("vs_bank_after_casting_ours_to_float16", []).append(m16)
                hp = HISTORICAL_PERTURBED[enc] / f"{sid}.pkl"
                if hp.exists():
                    hd, hc = pickle.loads(hp.read_bytes())
                    h_entry = {"path": str(hp), "sha256": sha256_file(hp),
                               "coords_equal": _coords_equal(coords, hc), "arms": {}}
                    for c in ("rotate_90", "rotate_180", "rotate_270", "flip_horizontal",
                              "flip_vertical", "gaussian_blur", "reinhard"):
                        hk = HISTORICAL_NAMES.get(c, c)
                        ha = hd.get(hk) if isinstance(hd, dict) else None
                        if ha is None or getattr(ha, "ndim", 0) != 2:
                            h_entry["arms"][c] = {"note": f"historical '{hk}' absent/empty"}
                            continue
                        hm = _rowwise(feats[c], ha)
                        h_entry["arms"][c] = _summ(hm)
                        pooled.setdefault(f"historical_{c}", []).append(hm)
                    hist[sid] = h_entry
            # Condition sanity: per-tile cosine to identity
            san = {}
            stim = manifest[sid]
            for c in CONDITIONS[1:]:
                m = _rowwise(feats[c], ident)
                san[c] = {"mean_cosine_to_identity": float(m["cos"].mean()),
                          "median_cosine_to_identity": float(np.median(m["cos"])),
                          "mean_rel_l2_to_identity": float(m["rel_l2"].mean()),
                          "frac_rows_identical": float((m["maxabs"] == 0).mean()),
                          "parameters": stim[c]["parameters"]}
            sanity[sid] = san
            fid[sid] = entry
        report["identity_fidelity"][enc] = fid | {
            "_pooled": {k: _summ({kk: np.concatenate([d[kk] for d in v])
                                  for kk in ("cos", "maxabs", "rel_l2")})
                        for k, v in pooled.items() if not k.startswith("historical_")}}
        if hist:
            report["historical_arms"][enc] = hist | {
                "_pooled": {k[len("historical_"):]: _summ(
                    {kk: np.concatenate([d[kk] for d in v]) for kk in ("cos", "maxabs", "rel_l2")})
                    for k, v in pooled.items() if k.startswith("historical_")}}
        report["condition_sanity"][enc] = sanity
        tiles = sum(t["n_tiles"] for t in thr)
        fwd = sum(t["encoder_forward_seconds"] for t in thr)
        report["throughput"][enc] = {"per_slide": thr, "tiles": tiles,
                                     "forward_seconds": round(fwd, 2),
                                     "tiles_per_second_forward_12_conditions":
                                         round(tiles / fwd, 3) if fwd else None}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(a.out, (json.dumps(report, indent=1, sort_keys=True, default=str)
                           + "\n").encode())
    print(f"[pilot-report] wrote {a.out}", flush=True)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "pilot-report":
        return pilot_report(argv[1:])
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
