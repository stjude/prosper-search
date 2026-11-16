"""Tile-level feature extraction over a mosaic, as an encoder *registry*.

Ported from ``comet-feature-gen/comet-<enc>/src/create_comet_features_by_<enc>.py``
rather than edited in place (standing rule 6: the ``comet-*`` trees are
read-only inputs; a path fix is a port, not an edit). Four defects in the
original are deliberately not reproduced:

1. ``args = parser.parse_args([...])`` with a hardcoded literal list, which
   made the whole argparse surface dead code -- the ``--percentage_selected``
   style of bug that also killed the mosaic generator's flag.
2. ``slides = slides[2400:]``, a hardcoded slice left in the shipped file.
3. Patches read **serially** with ``openslide.read_region`` in the same
   process that runs the forward pass. Measured aggregate throughput of the
   published UNI2 pass was 47 tiles/s, i.e. ~2-5% of an A100's rate for this
   model class: the job was slide-I/O bound while reserving 64 slots and
   using 8. Here the reads go through a ``DataLoader`` with one OpenSlide
   handle **per worker**, so decode overlaps compute.
4. A ``Pool`` that pickled a live CUDA model to each child.

Output specification is kept byte-compatible with the published artifacts: a
2-tuple ``(features, mosaic)`` per slide, ``features`` a float32
``(n_tiles, dim)`` array in mosaic order, so the existing aggregation layer
(median pooling / SAMPLER / OT / Yottixel) consumes it unchanged.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------
# Encoder registry. One entry per encoder -- NOT one cloned directory tree,
# which is what the 15 comet-* trees are and what [decision record] costs us.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Encoder:
    name: str
    dim: int
    builder: str          # dotted key resolved in _BUILDERS
    note: str = ""


REGISTRY: Dict[str, Encoder] = {
    "resnet50": Encoder(
        "resnet50", 2048, "timm",
        "ImageNet-supervised floor for R3.1/E4. timm resnet50.tv_in1k, "
        "torchvision weights, num_classes=0 -> 2048-d pooled features."),
    "uni2": Encoder(
        "uni2", 1536, "uni2",
        "Reproduces load_univ2_model() from comet-uni2/src/lib.py, minus the "
        "defunct /cnvrg/model local_dir default."),
    "h_optimus_1": Encoder(
        "h_optimus_1", 1536, "h_optimus_1",
        "R2.1 (reviewer-requested). ViT-g/14, architecturally identical to "
        "H-optimus-0 (also 1536-d) per bioptimus/H-optimus-1. Reproduces "
        "load_h_optimus_model() from comet-h-optimus/src/lib.py: "
        "dynamic_img_size=False (fixed 224x224, unlike UNI2's dynamic sizing), "
        "and Bioptimus's own normalisation stats, NOT ImageNet's -- using "
        "resolve_data_config()/timm defaults here would silently produce a "
        "plausible but wrong transform."),
    "retccl": Encoder(
        "retccl", 2048, "retccl",
        "No comet-retccl/ precedent anywhere on this filesystem -- unlike "
        "every other encoder here, this one has no published number to "
        "reproduce and no reference extractor to port from, only a weights "
        "file. Checkpoint verified as a bare torchvision resnet50 trunk "
        "(conv1/bn1/layer1-4, no fc: exact key-for-key match against "
        "torchvision.models.resnet50() minus fc.*) -- 'CCL' is the "
        "contrastive pretraining objective, not an architecture change. "
        "Preprocessing (resize 256, ImageNet mean/std) follows the official "
        "RetCCL repo's convention; PROVISIONAL, sanity-checked only "
        "(non-degenerate variance, no NaNs), since there is nothing local to "
        "fidelity-check it against."),
}

_TIMM_IDS = {"resnet50": "resnet50.tv_in1k"}

# Bioptimus's own normalisation, not ImageNet's -- see comet-h-optimus/src/lib.py.
_H_OPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
_H_OPTIMUS_STD = (0.211883, 0.230117, 0.177517)

RETCCL_WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "retccl" / "retccl.pth"


def _build_timm(name: str, device: str):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    model = timm.create_model(_TIMM_IDS[name], pretrained=True, num_classes=0)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg,
                                                       model=model))
    return model.to(device).eval(), transform


def _build_uni2(name: str, device: str):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24,
              init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
              num_classes=0, no_embed_class=True,
              mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
              reg_tokens=8, dynamic_img_size=True)
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg,
                                                       model=model))
    return model.to(device).eval(), transform


def _build_h_optimus_1(name: str, device: str):
    import timm
    import torchvision.transforms as T
    model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True,
                              init_values=1e-5, dynamic_img_size=False)
    # dynamic_img_size=False -> the reference resizes to a fixed 224x224
    # itself rather than relying on timm's data-config pipeline.
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=_H_OPTIMUS_MEAN, std=_H_OPTIMUS_STD),
    ])
    return model.to(device).eval(), transform


def _build_retccl(name: str, device: str):
    import torchvision.models as tvm
    import torchvision.transforms as T
    if not RETCCL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"RetCCL weights not found at {RETCCL_WEIGHTS}. Expected a "
            "318-key torchvision-resnet50-shaped state_dict (no fc.*).")
    model = tvm.resnet50(weights=None)
    model.fc = torch.nn.Identity()
    state = torch.load(RETCCL_WEIGHTS, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # fc.* is expected to be absent (that's why we replaced it with Identity);
    # anything else missing/unexpected means this is not the checkpoint shape
    # verified during the port and must not be silently trusted.
    bad_missing = [k for k in missing if not k.startswith("fc.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"RetCCL state_dict shape mismatch: missing={bad_missing}, "
            f"unexpected={unexpected}")
    # No local reference extractor exists for RetCCL (see REGISTRY note).
    # This is the official RetCCL repo's stated preprocessing; PROVISIONAL.
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model.to(device).eval(), transform


_BUILDERS: Dict[str, Callable] = {
    "timm": _build_timm, "uni2": _build_uni2,
    "h_optimus_1": _build_h_optimus_1, "retccl": _build_retccl,
}


def load_encoder(name: str, device: str = "cuda"):
    if name not in REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; known: {sorted(REGISTRY)}")
    enc = REGISTRY[name]
    return _BUILDERS[enc.builder](name, device)


# --------------------------------------------------------------------------
# Patch dataset: one OpenSlide handle per worker process.
# --------------------------------------------------------------------------


class _PatchSet(Dataset):
    """Patches of ONE slide. ``__getitem__`` opens the slide lazily so each
    DataLoader worker gets its own handle -- an OpenSlide object is not safe to
    share across a fork."""

    def __init__(self, slide_path: Path, locs: List[Tuple[int, int]],
                 patch_size: int, transform):
        self.slide_path = str(slide_path)
        self.locs = locs
        self.patch_size = patch_size
        self.transform = transform
        self._slide = None

    def __len__(self) -> int:
        return len(self.locs)

    def _handle(self):
        if self._slide is None:
            import openslide
            self._slide = openslide.OpenSlide(self.slide_path)
        return self._slide

    def __getitem__(self, i: int):
        w, h = self.locs[i]
        region = self._handle().read_region(
            (int(w), int(h)), 0, (self.patch_size, self.patch_size)).convert("RGB")
        return self.transform(region), i


def extract_slide(slide_path: Path, mosaic: List[dict], model, transform, *,
                  patch_size: int = 512, batch_size: int = 256,
                  num_workers: int = 8, device: str = "cuda",
                  amp: bool = True) -> Optional[np.ndarray]:
    """Features for one slide, in mosaic order. ``None`` if the slide fails."""
    if not mosaic:
        return None
    locs = [tuple(p["wsi_loc"]) for p in mosaic]
    ds = _PatchSet(slide_path, locs, patch_size, transform)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                    shuffle=False, pin_memory=True,
                    persistent_workers=False,
                    prefetch_factor=4 if num_workers else None)
    out = np.empty((len(locs), 0), dtype=np.float32)
    chunks: List[Tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for imgs, idx in dl:
            imgs = imgs.to(device, non_blocking=True)
            if amp and device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    feats = model(imgs)
            else:
                feats = model(imgs)
            chunks.append((feats.float().cpu().numpy(), idx.numpy()))
    if not chunks:
        return None
    dim = chunks[0][0].shape[1]
    out = np.empty((len(locs), dim), dtype=np.float32)
    for feats, idx in chunks:
        out[idx] = feats
    return out


def load_mosaic(path: Path) -> List[Tuple[str, List[dict]]]:
    """The published mosaic pkl: a sorted list of ``(filename, [tile dicts])``."""
    with open(path, "rb") as fh:
        return sorted(pickle.load(fh))


def slide_id_of(entry) -> str:
    return str(entry[0]).split(".")[0]


# --------------------------------------------------------------------------
# The R2.2 tile universe: all *fully*-tissue tiles, per slide.
# --------------------------------------------------------------------------

ALL_PATCHES_DIR = Path(
    "/PROSPER_WORKSPACE/"
    "comet-annotation/output/all_patches")


def load_all_patches(scan_id: str,
                     root: Optional[Path] = None) -> Optional[List[dict]]:
    """Tile locations for one slide from the all-patches CSV, or ``None``.

    Returns the same ``[{'wsi_loc': [x, y]}, ...]`` shape the mosaic pkl uses,
    so the extraction path downstream is identical.

    Standing rule 4 applies and is the reason this is a separate loader rather
    than a flag on the mosaic: these are all tiles at ``tissue_coverage == 1.0``,
    whereas the published 15% mosaic admitted partial-tissue tiles. The two sets
    are **not nested** (pooled 93.7% of mosaic tiles appear here; 1,063 of 2,887
    slides below 90%), so a tile-fraction sweep must run *inside one universe*
    and must never be presented as reproducing the published mosaic numbers.

    ``None`` means the slide has no list at all -- 34 of the 2,921 mosaic
    slides. The caller is expected to report those, not drop them quietly.
    """
    root = root or ALL_PATCHES_DIR
    csv_path = root / f"{scan_id}.csv"
    if not csv_path.exists():
        return None
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["wsi_loc_x", "wsi_loc_y"])
    if df.empty:
        return None
    return [{"wsi_loc": [int(x), int(y)]}
            for x, y in zip(df["wsi_loc_x"].to_numpy(),
                            df["wsi_loc_y"].to_numpy())]
