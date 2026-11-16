#!/usr/bin/env python3
"""R2.2-ALL Stage B: fp32 UNI2 extraction for explicit tile-coordinate manifests.

Matches the accepted 15% mosaic bank (``comet-uni2/data/tmp_final``; [analysis plan] R2.2-ALL D3):
the frozen ``code_frozen_007/prosper/embeddings/tile_features.py`` (loaded by path, SHA-guarded)
supplies the UNI2 model + timm transform and the per-worker OpenSlide ``_PatchSet``; forward is
``model(imgs)`` without autocast at batch 256, as in ``extract_slide(..., amp=False)``, with
timing instrumentation added (data wait vs synchronised forward).

Run under ``/HOME/miniconda3/envs/comet-uni2/bin/python3`` (the launcher enforces it).

Inputs: ``--coords-dir`` holding ``<slide>.coords.csv.gz`` (wsi_loc_x, wsi_loc_y, coverage),
and a slide list (``--slides``, ``--slides-file`` or ``--tasks-json`` + ``--task-id``).
Outputs per slide in ``--out`` (atomic): ``<slide>.npz`` (feats float32 N x 1536, xy int64,
coverage float64) and ``<slide>.json`` (provenance, GPU, timings). On a slide error a
``<slide>.FAILED.json`` is written and the job continues; exit code 1 if any slide failed.
Resume: a slide whose json exists with identical code/config/coords SHA is skipped; a json with
different SHAs is never overwritten (recorded as a failure).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_b_lib as L  # noqa: E402

TILE_FEATURES_DEFAULT = Path(
    "/PROSPER_WORKSPACE/prosper_revision/"
    "prosper_revision_output/r2_1_perturb_suite_20261005/code_frozen_007/prosper/embeddings/tile_features.py")
TILE_FEATURES_SHA = "6a1a7ff38378e786b11722581af55f1ba81dbaead8f85e50e8646f18488ada0f"
WEIGHTS = Path.home() / (".cache/huggingface/hub/models--MahmoodLab--UNI2-h/snapshots/"
                         "d517a8dd47902dd7c308b3c36f63bce47e7b9a43/pytorch_model.bin")
WEIGHTS_SHA = "6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e"

CONFIG = {
    "encoder": "uni2",
    "precision": "fp32 (no autocast)",
    "patch_size": 512,
    "level": 0,
    "forward_batch": 256,
    "tile_features_sha256": TILE_FEATURES_SHA,
    "weights_sha256": WEIGHTS_SHA,
    "expected": {"torch": "2.7.1+cu126", "cudnn": 90501, "timm": "1.0.16",
                 "python": "3.10.18", "numpy": "2.2.6", "Pillow": "11.3.0",
                 "openslide_python": "1.4.2", "libopenslide": "3.4.1",
                 "matmul_tf32": False, "cudnn_tf32": True, "cudnn_benchmark": False,
                 "flash_sdp": True, "mem_eff_sdp": True},
}


def config_sha():
    return L.sha256_bytes(json.dumps(CONFIG, sort_keys=True).encode())


def load_frozen(path, sha):
    got = L.sha256_file(path)
    if got != sha:
        raise SystemExit(f"tile_features sha mismatch: {got} != {sha}")
    spec = importlib.util.spec_from_file_location("frozen_tile_features", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


def environment(torch):
    import openslide
    import PIL
    import timm
    b = torch.backends
    env = {
        "python": platform.python_version(), "executable": sys.executable,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "timm": timm.__version__,
        "numpy": np.__version__, "Pillow": PIL.__version__,
        "openslide_python": openslide.__version__,
        "libopenslide": openslide.__library_version__,
        "matmul_tf32": b.cuda.matmul.allow_tf32, "cudnn_tf32": b.cudnn.allow_tf32,
        "cudnn_benchmark": b.cudnn.benchmark, "flash_sdp": b.cuda.flash_sdp_enabled(),
        "mem_eff_sdp": b.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": b.cuda.math_sdp_enabled(),
        "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"),
        "lsf_index": os.environ.get("LSB_JOBINDEX"), "lsf_queue": os.environ.get("LSB_QUEUE"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(gpu_model=p.name, gpu_capability=f"{p.major}.{p.minor}",
                   gpu_mem_mib=p.total_memory // (1 << 20), gpu_sm=p.multi_processor_count)
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,uuid",
                            "--format=csv,noheader"], capture_output=True, text=True,
                           timeout=60).stdout.strip().splitlines()
        env["nvidia_smi"] = q
        env["driver"] = q[0].split(", ")[1] if q else None
    except Exception as exc:  # pragma: no cover
        env["nvidia_smi_error"] = repr(exc)
    return env


def check_env(env, strict, accept_libopenslide=None):
    exp = dict(CONFIG["expected"])
    if accept_libopenslide:  # RHEL9 hosts: no system libopenslide 3.4.1; disclosed per output
        exp["libopenslide"] = accept_libopenslide
    bad = {k: (env.get(k), v) for k, v in exp.items() if env.get(k) != v}
    if env.get("NVIDIA_TF32_OVERRIDE") is not None or env.get("CUBLAS_WORKSPACE_CONFIG") is not None:
        bad["tf32_env"] = (env.get("NVIDIA_TF32_OVERRIDE"), env.get("CUBLAS_WORKSPACE_CONFIG"))
    if bad and strict:
        raise SystemExit(f"environment mismatch: {bad}")
    return bad


def slide_list(args):
    if args.tasks_json:
        tasks = json.loads(Path(args.tasks_json).read_text())["tasks"]
        return [str(s) for s in tasks[args.task_id]["slides"]]
    if args.slides_file:
        return [ln.strip() for ln in Path(args.slides_file).read_text().splitlines() if ln.strip()]
    return [s.strip() for s in args.slides.split(",") if s.strip()]


def extract(torch, frozen, model, transform, wsi, xy, args):
    from torch.utils.data import DataLoader
    ds = frozen._PatchSet(wsi, [tuple(map(int, p)) for p in xy], 512, transform)
    dl = DataLoader(ds, batch_size=CONFIG["forward_batch"], num_workers=args.num_workers,
                    shuffle=False, pin_memory=True, persistent_workers=False,
                    prefetch_factor=args.prefetch if args.num_workers else None)
    out = np.full((len(xy), L.DIM), np.nan, dtype=np.float32)
    seen = np.zeros(len(xy), dtype=bool)
    t_wait = t_fwd = 0.0
    t0 = time.perf_counter()
    it = iter(dl)
    with torch.no_grad():
        while True:
            a = time.perf_counter()
            try:
                imgs, idx = next(it)
            except StopIteration:
                break
            b = time.perf_counter()
            imgs = imgs.to("cuda", non_blocking=True)
            feats = model(imgs)
            arr = feats.float().cpu().numpy()
            c = time.perf_counter()
            t_wait += b - a
            t_fwd += c - b
            idx = idx.numpy()
            out[idx] = arr
            seen[idx] = True
    wall = time.perf_counter() - t0
    if not seen.all():
        raise RuntimeError(f"{int((~seen).sum())} tiles not returned by loader")
    return out, {"wall_s": wall, "data_wait_s": t_wait, "forward_s": t_fwd,
                 "tiles_per_s": len(xy) / wall if wall else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coords-dir")
    ap.add_argument("--out")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slides")
    g.add_argument("--slides-file")
    g.add_argument("--tasks-json")
    g.add_argument("--segments-json", help="probe mode: [{label, coords_dir, out, slides}] in one job")
    ap.add_argument("--task-id", type=int)
    ap.add_argument("--wsi-root", default=str(L.WSI_ROOT))
    ap.add_argument("--tile-features", default=str(TILE_FEATURES_DEFAULT))
    ap.add_argument("--num-workers", type=int, default=7)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--verify-weights", action="store_true")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    ap.add_argument("--accept-libopenslide", default=None,
                    help="accept this libopenslide version instead of 3.4.1 (RHEL9 probe only)")
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)
    if args.tasks_json and args.task_id is None:
        ap.error("--tasks-json needs --task-id")
    if args.segments_json:
        segments = [(d.get("label", ""), Path(d["coords_dir"]), Path(d["out"]), [str(x) for x in d["slides"]])
                    for d in json.loads(Path(args.segments_json).read_text())]
    else:
        if not (args.coords_dir and args.out):
            ap.error("--coords-dir and --out are required")
        segments = [(args.label, Path(args.coords_dir), Path(args.out), slide_list(args))]

    t_job = time.perf_counter()
    slides = [s for seg in segments for s in seg[3]]
    code_sha = {"stage_b_extract.py": L.sha256_file(Path(__file__).resolve()),
                "stage_b_lib.py": L.sha256_file(HERE / "stage_b_lib.py")}
    cfg_sha = config_sha()

    import torch
    frozen = load_frozen(Path(args.tile_features), TILE_FEATURES_SHA)
    env = environment(torch)
    mismatch = check_env(env, strict=not args.allow_env_mismatch,
                         accept_libopenslide=args.accept_libopenslide)
    env["accepted_libopenslide_override"] = args.accept_libopenslide
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    wsha = None
    if args.verify_weights:
        wsha = L.sha256_file(WEIGHTS)
        if wsha != WEIGHTS_SHA:
            raise SystemExit(f"weights sha mismatch {wsha}")
    t_load = time.perf_counter()
    model, transform = frozen.load_encoder("uni2", device="cuda")
    t_load = time.perf_counter() - t_load
    print(f"[extract] host={env['host']} job={env['lsf_job']} queue={env['lsf_queue']} "
          f"gpu={env.get('gpu_model')} driver={env.get('driver')} torch={env['torch']} "
          f"slides={len(slides)} model_load_s={t_load:.1f} env_mismatch={mismatch}", flush=True)

    n_fail = n_done = n_skip = n_tiles = 0
    work = [(lab, cdir, out, s) for lab, cdir, out, ss in segments for s in ss]
    for _, _, out, _ in work:
        out.mkdir(parents=True, exist_ok=True)
    for label, coords_dir, out, slide in work:
        js = out / f"{slide}.json"
        try:
            cpath = coords_dir / f"{slide}.coords.csv.gz"
            csha = L.sha256_file(cpath)
            if js.exists():
                prev = json.loads(js.read_text())
                if (prev.get("code_sha256") == code_sha and prev.get("config_sha256") == cfg_sha
                        and prev.get("coords_sha256") == csha):
                    n_skip += 1
                    print(f"[skip] {slide} (complete, same bindings)", flush=True)
                    continue
                raise RuntimeError("existing output with different bindings; refusing to overwrite")
            xy, cov = L.read_coords(cpath)
            if len(xy) == 0:
                raise RuntimeError("empty coordinate manifest")
            if len(np.unique(L.coord_key(xy))) != len(xy):
                raise RuntimeError("duplicate coordinates in manifest")
            wsi = Path(args.wsi_root) / f"{slide}.svs"
            st = wsi.stat()
            feats, timing = extract(torch, frozen, model, transform, wsi, xy, args)
            if feats.shape != (len(xy), L.DIM) or feats.dtype != np.float32 or not np.isfinite(feats).all():
                raise RuntimeError(f"bad features {feats.shape} {feats.dtype} finite={np.isfinite(feats).all()}")
            npz = out / f"{slide}.npz"
            tmp = out / f".{slide}.npz.tmp{os.getpid()}.npz"
            np.savez(tmp, feats=feats, xy=xy, coverage=cov)
            os.replace(tmp, npz)
            rec = {"slide": slide, "label": label, "n_tiles": int(len(xy)),
                   "coords_file": str(cpath), "coords_sha256": csha,
                   "npz_sha256": L.sha256_file(npz), "code_sha256": code_sha,
                   "config_sha256": cfg_sha, "config": CONFIG,
                   "wsi": str(wsi), "wsi_realpath": os.path.realpath(wsi),
                   "wsi_size": st.st_size, "wsi_mtime": st.st_mtime,
                   "gpu_model": env.get("gpu_model"), "driver": env.get("driver"),
                   "env": env, "env_mismatch": mismatch, "weights_sha256_verified": wsha,
                   "model_load_s": t_load, "timing": timing,
                   "num_workers": args.num_workers, "prefetch": args.prefetch,
                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            L.dump_json(js, rec)
            n_done += 1
            n_tiles += len(xy)
            print(f"[ok] {slide} n={len(xy)} wall={timing['wall_s']:.1f}s wait={timing['data_wait_s']:.1f}s "
                  f"fwd={timing['forward_s']:.1f}s tps={timing['tiles_per_s']:.1f}", flush=True)
        except Exception as exc:
            n_fail += 1
            L.dump_json(out / f"{slide}.FAILED.{env.get('lsf_job') or 'local'}.json",
                        {"slide": slide, "error": repr(exc), "traceback": traceback.format_exc(),
                         "host": env["host"], "lsf_job": env["lsf_job"], "queue": env["lsf_queue"]})
            print(f"[FAIL] {slide} {exc!r}", flush=True)
    print(f"[done] ok={n_done} skip={n_skip} fail={n_fail} tiles={n_tiles} "
          f"job_wall_s={time.perf_counter() - t_job:.1f}", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
