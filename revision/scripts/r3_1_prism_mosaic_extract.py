#!/usr/bin/env python
"""PRISM-mosaic row of R3.1 ([decision record], Decision item 2).

Same Virchow v1 -> PRISM chain as the PRISM-native arm, applied to the
published 512-px mosaic tile positions instead of trident's dense 20x/224 grid.
Each mosaic tile is read as a 512x512 level-0 region and passed through the
trident Virchow wrapper's own eval transform (Resize 224 bicubic, ToTensor,
ImageNet normalise) -- the 512->224 convention of the other grid encoders.
This is "PRISM applied to our mosaic", not PRISM as released.

The numerics mirror trident (comet-titan/trident, read-only):
  * patch features: ``WSI.extract_patch_features`` -- ``torch.autocast('cuda',
    dtype=encoder.precision, enabled=precision != float32)`` then
    ``.cpu().numpy()`` (float32: Virchow's final LayerNorm runs in fp32 under
    autocast; asserted below rather than cast silently);
  * slide embedding: ``WSI.extract_slide_features`` --
    ``torch.from_numpy(feats).float().to(device).unsqueeze(0)`` under
    ``torch.autocast('cuda', enabled=precision != float32)`` then
    ``.float().cpu().numpy().squeeze()``.

Outputs (``--out-root``):
  features_virchow/<id>.h5      features f32 (n, 2560), coords i64 (n, 2)
  slide_features_prism/<id>.h5  features f32 (1280,),   coords i64 (n, 2)
  provenance/<tag>_<pid>.json   one per invocation
The PRISM directory is read unchanged by
``scripts/r3_1_prism_whole_patient.py --prism-dir <out-root>/slide_features_prism``.

Production runs need no debug flags; ``--device cpu``, ``--no-gpu-guard`` and
``--limit-tiles`` exist only for smoke tests (``--limit-tiles`` refuses the
default production out-root).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pickle
import platform
import socket
import sys
import time
import traceback
from pathlib import Path

PROSPER_OUT_DEFAULT = (
    "/PROSPER_WORKSPACE/prosper_revision/prosper_revision_output")
PROSPER_OUT = os.environ.get("PROSPER_OUT", PROSPER_OUT_DEFAULT)
OUT_ROOT_DEFAULT = os.path.join(PROSPER_OUT, "r3_1_prism_20261018", "mosaic512")
MOSAIC_DEFAULT = ("/PROSPER_WORKSPACE/comet-new-scans/"
                  "output/mosaic/mosaic512_final_simplified.pkl")
MOSAIC_SHA256 = "13fdd889e91dcce88a7b5d94b81a8599ef54082dba520a69238d2efa6cfa98e7"
MOSAIC_N_SLIDES = 2921
MOSAIC_N_TILES = 4049810
SLIDES_DIR_DEFAULT = ("/PROSPER_WORKSPACE/"
                      "data/comet_final_HPC")
META_DEFAULT = ("/PROSPER_WORKSPACE/comet-feature-gen/"
                "comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv")
HF_CACHE_DEFAULT = os.path.join(PROSPER_OUT, "third_party", "hf_cache")
PRISM_REVISION = "b5ef311e78a0811c71ecf618f6f7140e9ebb5fc6"
VIRCHOW_REVISION = "19eebc84ae33e79f1b2d866e6ff90ae50e522f9a"
PATCH_SIZE_LEVEL0 = 512
MODEL_INPUT = 224
VIRCHOW_DIM = 2560
PRISM_DIM = 1280


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", help="CSV with column `wsi` (<id>.svs; CRLF tolerated); default: all mosaic slides")
    ap.add_argument("--tag", default="run", help="invocation tag (provenance file name, HF_MODULES_CACHE)")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--mosaic", default=MOSAIC_DEFAULT)
    ap.add_argument("--slides-dir", default=SLIDES_DIR_DEFAULT)
    ap.add_argument("--meta", default=META_DEFAULT, help="metadata CSV (`Slide.Scan.File`): slide universe only")
    ap.add_argument("--hf-cache", default=os.environ.get("HF_HUB_CACHE", HF_CACHE_DEFAULT))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="recompute slides whose PRISM h5 already exists")
    ap.add_argument("--provenance", help="override provenance JSON path (default <out-root>/provenance/<tag>_<pid>.json)")
    dbg = ap.add_argument_group("debug only (not for production)")
    dbg.add_argument("--limit-tiles", type=int, default=0, help="first N mosaic tiles per slide")
    dbg.add_argument("--device", default="cuda")
    dbg.add_argument("--no-gpu-guard", action="store_true", help="skip the A100-only guard")
    return ap.parse_args(argv)


def setup_hf_env(args) -> None:
    """Offline, pinned model loading. Must run before importing timm/transformers/trident."""
    os.environ["HF_HUB_CACHE"] = args.hf_cache
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Per-process dynamic-module cache: transformers' remote-code loading of
    # PRISM breaks on a shared modules cache (see r3_1_prism_fidelity_check.py).
    os.environ.setdefault("HF_MODULES_CACHE",
                          os.path.join(args.out_root, "hf_modules", f"{args.tag}.{os.getpid()}"))


def hf_snapshot(hf_cache: str, repo: str, expected_rev: str) -> dict:
    """Revision actually resolved offline (refs/main) and the snapshot's files -> blob names.
    LFS blob names in the HF cache are the files' sha256."""
    root = Path(hf_cache) / f"models--{repo.replace('/', '--')}"
    rev = (root / "refs" / "main").read_text().strip()
    if rev != expected_rev:
        raise RuntimeError(f"{repo}: refs/main={rev} != pinned {expected_rev}")
    snap = root / "snapshots" / rev
    files = {p.name: os.path.basename(os.readlink(p)) if p.is_symlink() else None
             for p in sorted(snap.iterdir()) if not p.is_dir()}
    return dict(repo=repo, revision=rev, snapshot=str(snap), blobs=files)


# --------------------------------------------------------------------------
# Mosaic reading. Copied from prosper-search
# prosper/embeddings/tile_features.py (load_mosaic, slide_id_of, _PatchSet);
# not imported, so this script has no dependency on that tree.
# --------------------------------------------------------------------------

def load_mosaic(path):
    """The published mosaic pkl: a sorted list of ``(filename, [tile dicts])``."""
    with open(path, "rb") as fh:
        return sorted(pickle.load(fh))


def slide_id_of(entry) -> str:
    return str(entry[0]).split(".")[0]


def make_patchset(slide_path, locs, patch_size, transform):
    from torch.utils.data import Dataset

    class _PatchSet(Dataset):
        """Patches of ONE slide; one OpenSlide handle per DataLoader worker."""

        def __init__(self):
            self.slide_path = str(slide_path)
            self.locs = locs
            self.patch_size = patch_size
            self.transform = transform
            self._slide = None

        def __len__(self):
            return len(self.locs)

        def _handle(self):
            if self._slide is None:
                import openslide
                self._slide = openslide.OpenSlide(self.slide_path)
            return self._slide

        def __getitem__(self, i):
            w, h = self.locs[i]
            region = self._handle().read_region(
                (int(w), int(h)), 0, (self.patch_size, self.patch_size)).convert("RGB")
            return self.transform(region), i

    return _PatchSet()


# --------------------------------------------------------------------------
# Chain (mirrors trident WSI.extract_patch_features / extract_slide_features)
# --------------------------------------------------------------------------

def virchow_features(venc, slide_path, locs, device, batch_size, num_workers):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    precision = venc.precision
    ds = make_patchset(slide_path, locs, PATCH_SIZE_LEVEL0, venc.eval_transforms)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                    pin_memory=str(device).startswith("cuda"), persistent_workers=False,
                    prefetch_factor=4 if num_workers else None)
    feats, order = [], []
    with torch.inference_mode():
        for imgs, idx in dl:
            imgs = imgs.to(device)
            with torch.autocast(device_type="cuda", dtype=precision, enabled=(precision != torch.float32)):
                batch_features = venc(imgs)
            feats.append(batch_features.cpu().numpy())
            order.append(idx.numpy())
    order = np.concatenate(order)
    if not np.array_equal(order, np.arange(len(locs))):
        raise RuntimeError("DataLoader returned tiles out of mosaic order")
    out = np.concatenate(feats, axis=0)
    if out.dtype != np.float32 or out.shape != (len(locs), VIRCHOW_DIM):
        raise RuntimeError(f"Virchow output {out.dtype} {out.shape}, expected float32 ({len(locs)}, {VIRCHOW_DIM})")
    return out


def prism_embedding(senc, feats, coords, coords_attrs, device):
    import torch

    with torch.inference_mode():
        x = torch.from_numpy(feats).float().to(device).unsqueeze(0)
        c = torch.from_numpy(coords).to(device).unsqueeze(0)
        batch = {"features": x, "coords": c, "attributes": coords_attrs}
        with torch.autocast(device_type="cuda", enabled=(senc.precision != torch.float32)):
            z = senc(batch, device)
        z = z.float().cpu().numpy().squeeze()
    if z.shape != (PRISM_DIM,):
        raise RuntimeError(f"PRISM output shape {z.shape}, expected ({PRISM_DIM},)")
    return z


# --------------------------------------------------------------------------
# h5 I/O
# --------------------------------------------------------------------------

def write_h5_atomic(path: Path, features, coords, feat_attrs: dict, coords_attrs: dict) -> None:
    import h5py
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with h5py.File(tmp, "w") as f:
        d = f.create_dataset("features", data=features)
        for k, v in feat_attrs.items():
            d.attrs[k] = v
        c = f.create_dataset("coords", data=coords)
        for k, v in coords_attrs.items():
            c.attrs[k] = v
    os.replace(tmp, path)


def prism_done(path: Path, n_tiles: int) -> bool:
    """Existing PRISM h5 opens, has a finite (1280,) vector and the expected tile count."""
    import h5py
    import numpy as np
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            v = f["features"][()]
            n = f["coords"].shape[0]
        return v.shape == (PRISM_DIM,) and bool(np.isfinite(v).all()) and n == n_tiles
    except Exception:
        return False


def read_list(path) -> list:
    with open(path, newline="") as f:
        lines = [ln.replace("\r", "").strip() for ln in f.read().split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines or lines[0] != "wsi":
        raise ValueError(f"{path}: first line must be the header `wsi`, got {lines[:1]}")
    return [ln[:-4] if ln.endswith(".svs") else ln for ln in lines[1:]]


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = dt.datetime.now().isoformat(timespec="seconds")
    out_root = Path(args.out_root)
    if args.limit_tiles and os.path.realpath(out_root) == os.path.realpath(OUT_ROOT_DEFAULT):
        print("REFUSED: --limit-tiles writes truncated outputs; give a non-production --out-root", flush=True)
        return 2
    setup_hf_env(args)

    import numpy as np
    import torch

    # ---- GPU guard (identical intent to trident_native/run_feat.sh): G1 passed only on A100.
    if args.no_gpu_guard:
        if str(args.device).startswith("cuda"):
            gpu = torch.cuda.get_device_name(torch.device(args.device))
        else:
            gpu = f"cpu ({platform.processor() or platform.machine()})"
    else:
        if not torch.cuda.is_available():
            print("REFUSED: non-A100 GPU (no CUDA device)", flush=True)
            sys.exit(3)
        gpu = torch.cuda.get_device_name(torch.device(args.device))
        if "A100" not in gpu:
            print(f"REFUSED: non-A100 GPU ({gpu})", flush=True)
            sys.exit(3)
    device = torch.device(args.device)
    log(f"host={socket.gethostname()} gpu={gpu} device={device} tag={args.tag} pid={os.getpid()}")

    # ---- inputs
    mosaic_sha = sha256_file(args.mosaic)
    if mosaic_sha != MOSAIC_SHA256:
        raise RuntimeError(f"mosaic sha256 {mosaic_sha} != expected {MOSAIC_SHA256}")
    mosaic = load_mosaic(args.mosaic)
    tiles = {slide_id_of(e): e[1] for e in mosaic}
    n_tiles_all = sum(len(v) for v in tiles.values())
    if len(mosaic) != MOSAIC_N_SLIDES or len(tiles) != MOSAIC_N_SLIDES or n_tiles_all != MOSAIC_N_TILES:
        raise RuntimeError(f"mosaic has {len(mosaic)} entries / {len(tiles)} ids / {n_tiles_all} tiles; "
                           f"expected {MOSAIC_N_SLIDES} / {MOSAIC_N_SLIDES} / {MOSAIC_N_TILES}")
    import pandas as pd
    meta_ids = set(pd.read_csv(args.meta, usecols=["Slide.Scan.File"])["Slide.Scan.File"].astype(str)
                   .str.replace(r"\.svs$", "", regex=True))
    if meta_ids != set(tiles):
        raise RuntimeError(f"mosaic ids != metadata ids: {len(set(tiles) - meta_ids)} mosaic-only, "
                           f"{len(meta_ids - set(tiles))} metadata-only")
    if args.list:
        slides = read_list(args.list)
        unknown = [s for s in slides if s not in tiles]
        if unknown:
            raise RuntimeError(f"{len(unknown)} listed slides not in the mosaic, e.g. {unknown[:5]}")
    else:
        slides = sorted(tiles, key=int)
    if len(set(slides)) != len(slides):
        raise RuntimeError("duplicate slide ids in --list")

    def slide_locs(sid):
        locs = [tuple(t["wsi_loc"]) for t in tiles[sid]]
        return locs[:args.limit_tiles] if args.limit_tiles else locs

    vdir = out_root / "features_virchow"
    pdir = out_root / "slide_features_prism"
    # Resume: decided before loading models so a finished list costs seconds.
    skipped = [] if args.force else [s for s in slides if prism_done(pdir / f"{s}.h5", len(slide_locs(s)))]
    skip_set = set(skipped)
    todo = [s for s in slides if s not in skip_set]
    log(f"listed={len(slides)} skip_existing={len(skipped)} todo={len(todo)}")

    # ---- models (trident wrappers: byte-for-byte the native arm's chain)
    virchow_snap = hf_snapshot(args.hf_cache, "paige-ai/Virchow", VIRCHOW_REVISION)
    prism_snap = hf_snapshot(args.hf_cache, "paige-ai/Prism", PRISM_REVISION)
    import timm
    import transformers
    import h5py
    import openslide
    from trident.patch_encoder_models.load import encoder_factory as patch_factory
    from trident.slide_encoder_models.load import encoder_factory as slide_factory
    import trident

    if not todo:
        venc = senc = None
    else:
        venc, senc = patch_factory("virchow"), slide_factory("prism")
        venc.to(device)
        venc.eval()
        senc.to(device)
        senc.eval()
    if venc is None:
        transform_repr = None
    elif venc.enc_name != "virchow" or senc.enc_name != "prism":
        raise RuntimeError(f"unexpected encoders {venc.enc_name}/{senc.enc_name}")
    elif venc.precision != torch.float16 or senc.precision != torch.float16:
        raise RuntimeError(f"unexpected precisions {venc.precision}/{senc.precision}")
    elif getattr(venc, "return_cls", None) is not False:
        raise RuntimeError("Virchow wrapper must return cls (+) mean-of-patch-tokens")
    else:
        transform_repr = repr(venc.eval_transforms)
        log(f"virchow eval_transforms: {transform_repr}")

    prov_path = Path(args.provenance) if args.provenance else \
        out_root / "provenance" / f"{args.tag}_{os.getpid()}.json"
    script_path = Path(__file__).resolve()
    prov = dict(
        script=str(script_path), script_sha256=sha256_file(script_path), argv=sys.argv,
        tag=args.tag, pid=os.getpid(), hostname=socket.gethostname(), gpu=gpu, device=str(device),
        start=t_start, end=None, status="running",
        mosaic=dict(path=args.mosaic, sha256=mosaic_sha, n_slides=len(tiles), n_tiles=n_tiles_all),
        metadata=dict(path=args.meta, sha256=sha256_file(args.meta), n_ids=len(meta_ids)),
        list=dict(path=args.list, sha256=sha256_file(args.list) if args.list else None, n=len(slides)),
        slides_dir=args.slides_dir, out_root=str(out_root),
        models=dict(virchow=virchow_snap, prism=prism_snap,
                    virchow_precision=str(venc.precision) if venc else None,
                    prism_precision=str(senc.precision) if senc else None,
                    virchow_eval_transforms=transform_repr),
        env=dict(python=platform.python_version(), executable=sys.executable, torch=torch.__version__,
                 cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), timm=timm.__version__,
                 transformers=transformers.__version__, h5py=h5py.__version__,
                 openslide=openslide.__version__, numpy=np.__version__,
                 trident_path=os.path.dirname(trident.__file__),
                 **{k: os.environ.get(k) for k in ("HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                                                   "HF_MODULES_CACHE", "CUDA_VISIBLE_DEVICES", "LSB_JOBID")}),
        params=dict(batch_size=args.batch_size, num_workers=args.num_workers, limit_tiles=args.limit_tiles,
                    force=args.force, patch_size_level0=PATCH_SIZE_LEVEL0, model_input=MODEL_INPUT),
        counts=None, done=[], skipped=skipped, failed=[])

    def write_prov():
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = prov_path.with_name(prov_path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(prov, f, indent=2, default=str)
        os.replace(tmp, prov_path)

    write_prov()

    vdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)

    for k, sid in enumerate(todo, 1):
        locs = slide_locs(sid)
        ppath, vpath = pdir / f"{sid}.h5", vdir / f"{sid}.h5"
        t0 = time.time()
        try:
            slide_path = Path(args.slides_dir) / f"{sid}.svs"
            with openslide.OpenSlide(str(slide_path)) as osr:
                w0, h0 = osr.dimensions
                mag = osr.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
            coords = np.asarray(locs, dtype=np.int64).reshape(-1, 2)
            coords_attrs = dict(patch_size_level0=PATCH_SIZE_LEVEL0, patch_size=MODEL_INPUT,
                                source=os.path.basename(args.mosaic), mosaic_sha256=mosaic_sha, name=sid,
                                level0_width=int(w0), level0_height=int(h0),
                                level0_magnification=int(float(mag)) if mag else -1, limit_tiles=int(args.limit_tiles))
            feats = virchow_features(venc, slide_path, locs, device, args.batch_size, args.num_workers)
            t_v = time.time() - t0
            emb = prism_embedding(senc, feats, coords, coords_attrs, device)
            if not np.isfinite(feats).all() or not np.isfinite(emb).all():
                raise RuntimeError("non-finite features/embedding")
            write_h5_atomic(vpath, feats, coords,
                            dict(encoder="virchow", virchow_revision=VIRCHOW_REVISION, name=sid, gpu=gpu),
                            coords_attrs)
            write_h5_atomic(ppath, emb.astype(np.float32, copy=False), coords,
                            dict(encoder="prism", prism_revision=PRISM_REVISION, patch_encoder="virchow",
                                 virchow_revision=VIRCHOW_REVISION, name=sid, gpu=gpu),
                            coords_attrs)
        except Exception as e:
            traceback.print_exc()
            prov["failed"].append(dict(slide=sid, error=f"{type(e).__name__}: {e}"))
            log(f"[{k}/{len(todo)}] {sid} FAILED {type(e).__name__}: {e}")
            continue
        sec = time.time() - t0
        prov["done"].append(dict(slide=sid, n_tiles=len(locs), seconds=round(sec, 2),
                                 virchow_seconds=round(t_v, 2)))
        log(f"[{k}/{len(todo)}] {sid} n_tiles={len(locs)} seconds={sec:.1f} "
            f"tiles/s={len(locs) / max(sec, 1e-9):.1f} gpu={gpu}")
        if k % 10 == 0:
            write_prov()

    prov["end"] = dt.datetime.now().isoformat(timespec="seconds")
    prov["counts"] = dict(listed=len(slides), done=len(prov["done"]), skipped=len(prov["skipped"]),
                          failed=len(prov["failed"]),
                          tiles_done=sum(d["n_tiles"] for d in prov["done"]))
    prov["status"] = "ok" if not prov["failed"] else "failed"
    write_prov()
    log(f"end counts={prov['counts']} provenance={prov_path}")
    return 0 if not prov["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
