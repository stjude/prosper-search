#!/usr/bin/env python
"""G2 identity gate for the Virchow v1 tile encoder ([decision record]).

Distinct-route check of trident's ``features_virchow/<id>.h5``: re-embeds the
first N tiles of one slide with an independent implementation that follows the
paige-ai/Virchow model card literally (timm ``vit_huge_patch14_224`` with the
card's kwargs, strict load from the pinned ``model.safetensors``, bicubic
resize to 224, ImageNet normalisation, fp16 autocast, cls ⊕ mean patch token)
and compares with the saved features. Also asserts dim 2,560 and records GPU
type and file hashes.

Usage: python r3_1_virchow_identity_check.py --job-root <trident_native> --slide 5643 --n 64 --out <dir>
"""
import argparse
import hashlib
import json
import os
import platform
import socket
from pathlib import Path

VIRCHOW_REVISION = "19eebc84ae33e79f1b2d866e6ff90ae50e522f9a"
WSI_DIR = Path("/PROSPER_WORKSPACE/data/comet_final_HPC")
COORDS = "20x_224px_0px_overlap"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-root", required=True)
    ap.add_argument("--slide", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-cache", default=os.environ.get("HF_HUB_CACHE"))
    ap.add_argument("--layout", choices=["native", "mosaic"], default="native",
                    help="native: trident job root (<root>/%s/{features_virchow,patches}); "
                         "mosaic: r3_1_prism_mosaic_extract.py root (<root>/features_virchow, "
                         "coords and patch_size_level0 read from the feature file itself)" % COORDS)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import h5py
    import numpy as np
    import openslide
    import timm
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from torchvision import transforms

    snap = Path(args.hf_cache) / "models--paige-ai--Virchow" / "snapshots" / VIRCHOW_REVISION
    weights = snap / "model.safetensors"
    root = Path(args.job_root)
    if args.layout == "mosaic":
        feat_path = root / "features_virchow" / f"{args.slide}.h5"
        coord_path = feat_path
    else:
        feat_path = root / COORDS / "features_virchow" / f"{args.slide}.h5"
        coord_path = root / COORDS / "patches" / f"{args.slide}_patches.h5"

    rec = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "virchow_revision": VIRCHOW_REVISION,
        "virchow_safetensors_sha256": sha256(weights),
        "slide": args.slide,
        "layout": args.layout,
        "n_tiles_checked": args.n,
    }

    with h5py.File(feat_path) as h:
        saved = np.asarray(h["features"][: args.n])
        saved_coords = np.asarray(h["coords"][: args.n])
        rec["saved_dtype"] = str(h["features"].dtype)
        rec["saved_shape"] = list(h["features"].shape)
    with h5py.File(coord_path) as h:
        coords = np.asarray(h["coords"][: args.n])
        level0 = int(h["coords"].attrs["patch_size_level0"])
    assert saved.shape[1] == 2560, saved.shape
    assert np.array_equal(saved_coords, coords), "feature/coord order mismatch"
    rec["patch_size_level0"] = level0

    # Model-card reference: independent of trident's wrapper.
    model = timm.create_model(
        "vit_huge_patch14_224", pretrained=False, img_size=224, init_values=1e-5,
        num_classes=0, mlp_ratio=5.3375, global_pool="", dynamic_img_size=True,
        mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
    )
    missing, unexpected = model.load_state_dict(load_file(str(weights)), strict=True), None
    model = model.eval().cuda()
    tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    slide = openslide.OpenSlide(str(WSI_DIR / f"{args.slide}.svs"))
    imgs = [tf(slide.read_region((int(x), int(y)), 0, (level0, level0)).convert("RGB")) for x, y in coords]
    x = torch.stack(imgs).cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        o = model(x)
        ref = torch.cat([o[:, 0], o[:, 1:].mean(1)], dim=-1)
    ref = ref.float().cpu().numpy()
    rec["ref_shape"] = list(ref.shape)
    rec["n_patch_tokens"] = int(o.shape[1] - 1)

    s = saved.astype(np.float32)
    cos = (s * ref).sum(1) / (np.linalg.norm(s, axis=1) * np.linalg.norm(ref, axis=1))
    rec["cos_min"] = float(cos.min())
    rec["cos_mean"] = float(cos.mean())
    rec["max_abs_diff"] = float(np.abs(s - ref).max())
    rec["ref_abs_max"] = float(np.abs(ref).max())
    rec["n_nonfinite_saved"] = int((~np.isfinite(s)).sum())
    # Component-order check: swapping cls/mean halves must not match better.
    swapped = np.concatenate([ref[:, 1280:], ref[:, :1280]], 1)
    cos_sw = (s * swapped).sum(1) / (np.linalg.norm(s, axis=1) * np.linalg.norm(swapped, axis=1))
    rec["cos_mean_if_halves_swapped"] = float(cos_sw.mean())
    ok = rec["cos_min"] >= 0.999 and rec["n_nonfinite_saved"] == 0 and rec["n_patch_tokens"] == 256 \
        and rec["cos_mean"] > rec["cos_mean_if_halves_swapped"]
    rec["status"] = "PASS" if ok else "FAIL"
    tag = rec["gpu"].replace(" ", "_")
    (out / f"g2_virchow_identity_{args.slide}_{tag}.json").write_text(json.dumps(rec, indent=2))
    np.save(out / f"g2_ref_{args.slide}.npy", ref)
    print(json.dumps(rec, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
