#!/usr/bin/env python
"""G1 fidelity gate for the PRISM comparator ([decision record]).

Loads the pinned PRISM snapshot offline, runs ``slide_representations`` on the
TCGA example shipped with the model and compares the first eight values of
``image_embedding[0]`` with the values printed in the model card. Records GPU
type, library versions and file hashes. Exits non-zero if the check fails.

Usage: python r3_1_prism_fidelity_check.py --out <dir>
"""
import argparse
import hashlib
import json
import os
import platform
import socket
import time
from pathlib import Path

PRISM_REVISION = "b5ef311e78a0811c71ecf618f6f7140e9ebb5fc6"
TCGA_FILE = "tcga/TCGA-B6-A0WZ-01Z-00-DX1.6CFB236E-36F5-43D6-8DE3-C4ECBD3C14C6.pth"
# Values printed in the paige-ai/Prism README for image_embedding[0, :8].
EXPECTED8 = [0.0620, -0.2983, 0.5849, -0.7383, 0.1242, -0.7440, -0.4719, -0.2920]
TOL = 5e-4  # README prints 4 decimals; allow half a unit in the last place


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-cache", default=os.environ.get("HF_HUB_CACHE"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Per-process dynamic-module cache: the shared default raced across jobs,
    # and transformers' local-directory loader misses second-level relative
    # imports (biogpt -> biogpt_hf), so the model is loaded by pinned hub id.
    os.environ.setdefault("HF_MODULES_CACHE", str(out / f"hf_modules_{os.getpid()}"))
    import torch
    import transformers
    from transformers import AutoModel

    snap = Path(args.hf_cache) / "models--paige-ai--Prism" / "snapshots" / PRISM_REVISION
    assert snap.is_dir(), snap
    rec = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "prism_revision": PRISM_REVISION,
        "prism_safetensors_sha256": sha256(snap / "model.safetensors"),
        "tcga_pth_sha256": sha256(snap / TCGA_FILE),
        "expected8": EXPECTED8,
        "tol": TOL,
    }
    if not torch.cuda.is_available():
        rec["status"] = "FAIL: no CUDA device"
        (out / "g1_prism_fidelity.json").write_text(json.dumps(rec, indent=2))
        print(rec["status"])
        return 2

    t0 = time.time()
    model = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True, revision=PRISM_REVISION)
    model.text_decoder = None
    model = model.to("cuda").eval()
    rec["load_s"] = round(time.time() - t0, 1)

    d = torch.load(snap / TCGA_FILE, map_location="cpu", weights_only=False)
    emb = d["embeddings"]
    rec["tcga_n_tiles"] = int(emb.shape[0])
    rec["tcga_dim"] = int(emb.shape[1])
    assert emb.shape[1] == 2560, emb.shape

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        t1 = time.time()
        reprs = model.slide_representations(emb.unsqueeze(0).to("cuda"))
        torch.cuda.synchronize()
        rec["forward_s"] = round(time.time() - t1, 3)
    z = reprs["image_embedding"].float().cpu()
    rec["image_embedding_shape"] = list(z.shape)
    got8 = [round(float(v), 4) for v in z[0, :8]]
    rec["got8"] = got8
    rec["max_abs_diff8"] = max(abs(a - b) for a, b in zip(got8, EXPECTED8))
    rec["norm"] = float(z[0].norm())
    rec["n_nan"] = int(torch.isnan(z).sum())
    ok = rec["max_abs_diff8"] <= TOL and rec["n_nan"] == 0 and list(z.shape) == [1, 1280]
    rec["status"] = "PASS" if ok else "FAIL"
    torch.save(z, out / "g1_tcga_image_embedding.pt")
    tag = rec["gpu"].replace(" ", "_")
    (out / f"g1_prism_fidelity_{tag}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
