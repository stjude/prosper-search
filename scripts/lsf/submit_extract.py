#!/usr/bin/env python3
"""Submit tile-feature extraction (GPU) as a canary-then-fan-out array.

Task C5's runnable encoder rung. R3.1/E4 asked for stronger baselines and an
ImageNet floor; of the three new encoders in the plan, this is the only one
that can run tonight:

  * H-Optimus-1  -- blocked on the Bioptimus gated licence (H1, requested).
  * RetCCL       -- no weights on this filesystem and none in the HF cache;
                    upstream distributes them by Google Drive link, so it is a
                    download task, not a compute task.
  * ResNet50     -- cached (`models--timm--resnet50.tv_in1k`), loads offline,
                    2048-d. Runnable now.

Slides are split into contiguous ranges over the sorted mosaic order, one
array element per range, so array *count* is what we manage. Each element is
idempotent (skip-if-loadable), which makes resubmission a resume.

Usage:
    submit_extract.py queues
    submit_extract.py canary --encoder resnet50 --go
    submit_extract.py fanout --encoder resnet50 --epilogue <.out> --go
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from prosper.hpc import submit as S             # noqa: E402
from prosper.hpc import queues as Q             # noqa: E402
from prosper.evaluation import paths            # noqa: E402
from prosper.embeddings import tile_features as T  # noqa: E402

WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
MODULE = "prosper.embeddings.extract_cli"
OUT = paths.output_root()

# 16 slots so the DataLoader has real workers to read patches with: the
# published pass reserved 64 and used 8, and was slide-I/O bound at 47
# tiles/s. num_workers is set to slots-2, leaving room for the main process
# and the CUDA context.
SLOTS = 16
WORKERS = SLOTS - 2
BATCH = 60          # slides per array element -> ~49 elements over 2,921


def _mosaic_len() -> int:
    return len(T.load_mosaic(paths.require(_mosaic_path(), "mosaic pkl")))


def _mosaic_path() -> Path:
    return (Path(os.environ.get(
        "COMET_SCANS_ROOT",
        "/PROSPER_WORKSPACE/comet-new-scans"))
        / "output" / "mosaic" / "mosaic512_final_simplified.pkl")


def _tag(encoder: str, tile_source: str) -> str:
    """Output/log tag. all_patches results must never land in the same
    directory as the 15% mosaic results -- they are a different tile universe
    (standing rule 4), not a refinement of the same one."""
    return encoder if tile_source == "mosaic" else f"{encoder}_all_patches"


def _task(encoder: str, tile_source: str) -> str:
    return f"features/{_tag(encoder, tile_source)}"


def canary(encoder: str, go: bool, tile_source: str = "mosaic",
           n_slides: int = 4, wall: int = 28) -> None:
    """A few slides on a 30-minute GPU queue: enough to measure tiles/s and to
    prove the model loads offline on a compute node, which is the real risk
    (the login node has the HF cache; egress from compute nodes is untested)."""
    tag = _tag(encoder, tile_source)
    res = S.Resources(slots=SLOTS, mem_mb_per_slot=4096, wall_minutes=wall,
                      gpus=1)
    S.canary(f"PROSPER_MODULE={MODULE} bash {WORKER} "
             f"--encoder {encoder} --lo 1 --hi {n_slides} "
             f"--tile-source {tile_source} --task {_task(encoder, tile_source)} "
             f"--num-workers {WORKERS} --batch-size 256",
             job_name=f"prosper_extract_{tag}", res=res, kind="gpu",
             log_dir=OUT / "logs" / f"extract_{tag}", dry_run=not go)


def fanout(encoder: str, epilogue_path: Path, go: bool, batch: int,
           wall: int, n_queues: int, tile_source: str = "mosaic") -> None:
    epi = S.parse_epilogue(epilogue_path.read_text())
    if not epi:
        raise SystemExit(f"no LSF epilogue in {epilogue_path}")
    tag = _tag(encoder, tile_source)
    n_slides = _mosaic_len()
    n_elem = math.ceil(n_slides / batch)
    params = OUT / f"features_{tag}" / "params.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n_elem):
        lo = i * batch + 1
        hi = min(n_slides, (i + 1) * batch)
        lines.append(f"--encoder {encoder} --lo {lo} --hi {hi} "
                     f"--tile-source {tile_source} "
                     f"--task {_task(encoder, tile_source)} "
                     f"--num-workers {WORKERS} --batch-size 256")
    params.write_text("\n".join(lines) + "\n")
    print(f"[extract] {n_slides} slides -> {n_elem} elements of <= {batch} "
          f"-> {params}")

    # Memory is right-sized from the canary, but slots are NOT cut here:
    # effective_cores averages over the whole element and the DataLoader's
    # workers are I/O-blocked rather than CPU-busy, so the epilogue
    # systematically understates the parallelism this job actually needs.
    # right_size() documents exactly this caveat.
    peak = epi.get("max_mem_mb") or 4096
    mem = max(2048, int(peak * 2.0 / SLOTS) + 1)
    res = S.Resources(slots=SLOTS, mem_mb_per_slot=mem, wall_minutes=wall,
                      gpus=1)
    print(f"[extract] canary peak {peak:.0f} MB -> -n {SLOTS} -M {mem} "
          f"({res.total_reserved_gb:.1f} GB total)")

    # fan_out() calls right_size() again on whatever epilogue it is handed, and
    # right_size cuts slots toward `effective_cores = CPU time / Run time`. For
    # this job that metric is actively misleading: the canary measured 405 s CPU
    # over 110 s wall = 3.7 effective cores, so slots were cut 16 -> 4 and the
    # DataLoader's 14 patch-reader workers landed on 4 reserved cores. Those
    # workers block on GPFS `read_region` rather than burning CPU, which is
    # exactly the caveat right_size documents. Overriding the key keeps the
    # memory right-sizing (a real 16x saving) while refusing the slot cut --
    # rather than passing epilogue=None, which fan_out rejects outright.
    epi = dict(epi)
    epi["effective_cores"] = float(SLOTS)
    S.fan_out(f"PROSPER_MODULE={MODULE} PROSPER_PARAMS={params} bash {WORKER}",
              n=n_elem, job_name=f"prosper_extract_{tag}", res=res,
              kind="gpu", log_dir=OUT / "logs" / f"extract_{tag}",
              n_queues=n_queues, canary_epilogue=epi, dry_run=not go)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["queues", "canary", "fanout"])
    ap.add_argument("--encoder", default="resnet50", choices=sorted(T.REGISTRY))
    ap.add_argument("--tile-source", default="mosaic",
                    choices=["mosaic", "all_patches"])
    ap.add_argument("--canary-slides", type=int, default=4)
    ap.add_argument("--epilogue", type=Path)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--wall", type=int, default=600)
    ap.add_argument("--n-queues", type=int, default=2)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if a.action == "queues":
        print(Q.report(kind="gpu", wall_minutes=a.wall, slots=SLOTS))
        return 0
    if a.action == "canary":
        canary(a.encoder, a.go, a.tile_source, a.canary_slides)
    else:
        if not a.epilogue:
            raise SystemExit("fanout needs --epilogue <canary .out>")
        fanout(a.encoder, a.epilogue, a.go, a.batch, a.wall, a.n_queues,
               a.tile_source)
    if not a.go:
        print("\n[dry run] nothing submitted. Re-run with --go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
