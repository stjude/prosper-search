#!/usr/bin/env python3
"""Submit OT / BoB(Yottixel) pairwise-similarity rows as a canary-then-fan-out
array, then chain the combine + evaluation steps behind it.

New for the H-Optimus-1 (R2.1) and RetCCL rungs of C5: neither encoder has
ever had an OT or Yottixel matrix built in this pipeline (ResNet50's C5 rung
only got median pooling), and there is no existing batch driver for either
method -- only the per-slide algorithm classes
(``prosper.embeddings.slide_embedding``). This is that driver.

Cost note, measured 2026-08-28 (not assumed from the historical LSF request):
the published reference driver (``comet-uni2/src/compute_ot_similarities_lsf.py``)
requested a 24h wall limit per query slide, but a local timing probe --
one real pair of mosaic-scale (~1,300-tile) UNI2 slides, ``ot.emd2`` only --
took 0.34s, projecting to ~16-20 min for a full 2,920-pair row. That request
may have been generously over-provisioned rather than reflecting actual cost;
this script's canary measures the REAL number, including the full-library
load this local probe did not do, before anything is sized or believed.

Usage:
    submit_similarity.py canary --encoder h_optimus_1 --method ot --go
    submit_similarity.py fanout --encoder h_optimus_1 --method ot \\
        --epilogue <canary .out> --go
    submit_similarity.py combine --encoder h_optimus_1 --method ot \\
        --after <fanout job ids> --go
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from prosper.hpc import submit as S             # noqa: E402
from prosper.evaluation import paths            # noqa: E402
from prosper.embeddings import tile_features as T  # noqa: E402

WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
MODULE = "prosper.embeddings.similarity_cli"
COMBINE_MODULE = "prosper.embeddings.combine_similarity_cli"


def _out() -> Path:
    return paths.output_root()

# uni2 is not a new C5 rung -- it is included so this driver can also run the
# pre-fanout canary/fidelity probe against UNI2's real cached comet features
# (the only mosaic-scale library with a published OT/BoB matrix to check
# against) before trusting the same code on either new encoder.
FEATURE_DIM = {"h_optimus_1": 1536, "retccl": 2048, "uni2": 1536}


def _task(encoder: str, method: str) -> str:
    return f"similarity/{encoder}_{method}"


def _library_len(encoder: str, features: Path) -> int:
    import glob
    return len(glob.glob(str(features / "*.pkl")))


def canary(encoder: str, method: str, features: Path, go: bool,
           n_queries: int = 3, wall: int = 120) -> None:
    """A few query rows against the FULL library -- the real cost, including
    the full-library load this driver's own docstring flags as unmeasured
    by a login-node probe."""
    tag = f"{encoder}_{method}"
    # 2,921 mosaic-scale slides x ~1,300 tiles x up to 2048-d x float32 is
    # ~30 GB for one normalised-only copy of the whole reference library --
    # measured directly: a 16 GB reservation hit TERM_MEMLIMIT loading UNI2's
    # real library before the fix that dropped the redundant raw copy. 40 GB
    # leaves real headroom above that, not just above the smaller post-fix
    # estimate.
    res = S.Resources(slots=1, mem_mb_per_slot=40960, wall_minutes=wall)
    cmd = (f"PROSPER_MODULE={MODULE} bash {WORKER} "
          f"--encoder {encoder} --method {method} --features {features} "
          f"--feature-dim {FEATURE_DIM[encoder]} --lo 1 --hi {n_queries} "
          f"--task {_task(encoder, method)}")
    S.canary(cmd, job_name=f"prosper_sim_{tag}", res=res, kind="cpu",
             log_dir=_out() / "logs" / f"similarity_{tag}", dry_run=not go)


def fanout(encoder: str, method: str, features: Path, epilogue_path: Path,
           go: bool, batch: int, wall: int, n_queues: int,
           mem_mb_per_slot: int | None = None) -> None:
    epi = S.parse_epilogue(epilogue_path.read_text())
    if not epi:
        raise SystemExit(f"no LSF epilogue in {epilogue_path}")
    tag = f"{encoder}_{method}"
    n_slides = _library_len(encoder, features)
    n_elem = math.ceil(n_slides / batch)
    params = _out() / f"similarity_{tag}" / "params.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n_elem):
        lo = i * batch + 1
        hi = min(n_slides, (i + 1) * batch)
        lines.append(f"--encoder {encoder} --method {method} "
                     f"--features {features} "
                     f"--feature-dim {FEATURE_DIM[encoder]} "
                     f"--lo {lo} --hi {hi} --task {_task(encoder, method)}")
    params.write_text("\n".join(lines) + "\n")
    print(f"[similarity] {n_slides} query slides -> {n_elem} elements of "
          f"<= {batch} -> {params}")

    # 2,921 mosaic-scale slides x ~1,300 tiles x up to 2048-d x float32 is
    # ~30 GB for one normalised-only copy of the whole reference library --
    # measured directly: a 16 GB reservation hit TERM_MEMLIMIT loading UNI2's
    # real library before the fix that dropped the redundant raw copy. 40 GB
    # leaves real headroom above that, not just above the smaller post-fix
    # estimate.
    res = S.Resources(slots=1, mem_mb_per_slot=40960, wall_minutes=wall)
    if mem_mb_per_slot is not None:
        # Floor after right_size(): use a proxy canary from another encoder
        # (e.g. uni2 OT for retccl OT) without under-reserving memory.
        peak = mem_mb_per_slot / 2.0
        epi = dict(epi, max_mem_mb=max(epi.get("max_mem_mb", 0), peak))
    S.fan_out(f"PROSPER_MODULE={MODULE} PROSPER_PARAMS={params} bash {WORKER}",
              n=n_elem, job_name=f"prosper_sim_{tag}", res=res, kind="cpu",
              log_dir=_out() / "logs" / f"similarity_{tag}",
              n_queues=n_queues, canary_epilogue=epi, dry_run=not go)


def _bsub(args, name, queue, cmd, dry, subdir):
    log_dir = _out() / "logs" / subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    full = ["bsub", "-q", queue, "-J", name,
            "-oo", str(log_dir / f"{name}.%J.out"),
            "-eo", str(log_dir / f"{name}.%J.err")] + args + [cmd]
    if dry:
        print("  DRY:", name, "->", queue)
        return None
    env = os.environ.copy()
    env["LSB_JOB_REPORT_MAIL"] = "N"
    r = subprocess.run(full, capture_output=True, text=True, env=env)
    print("  " + (r.stdout or r.stderr).strip())
    if r.returncode != 0:
        raise SystemExit(f"bsub failed for {name}")
    return r.stdout.split("<")[1].split(">")[0]


def combine(encoder: str, method: str, after: str, go: bool,
           min_slides: int) -> None:
    """Chained on `ended`, not `done` (rule 9): the similarity array can have
    a partial straggler and the combine step's own --min-slides gate is the
    correctness check, not LSF's dependency semantics."""
    tag = f"{encoder}_{method}"
    dep = " && ".join(f"ended({j})" for j in after.split(",") if j.strip())
    args = ["-n", "4", "-R", "span[hosts=1]",
            "-R", "select[mem>=8192] rusage[mem=8192]", "-M", "8192",
            "-W", "60"]
    if dep:
        args += ["-w", dep]
    cmd = (f"PROSPER_MODULE={COMBINE_MODULE} bash {WORKER} "
          f"--encoder {encoder} --method {method} "
          f"--similarity-dir {_out()}/{_task(encoder, method)} "
          f"--min-slides {min_slides}")
    _bsub(args, f"prosper_combine_{tag}", "priority", cmd, not go, "combine_similarity")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["canary", "fanout", "combine"])
    ap.add_argument("--encoder", required=True, choices=sorted(FEATURE_DIM))
    ap.add_argument("--method", required=True, choices=["ot", "bob"])
    ap.add_argument("--features", type=Path, default=None,
                    help="default $PROSPER_OUT/features/<encoder>")
    ap.add_argument("--n-queries", type=int, default=3)
    ap.add_argument("--epilogue", type=Path)
    ap.add_argument("--batch", type=int, default=None,
                    help="query slides per array element (default: 1 for ot, "
                         "20 for bob)")
    ap.add_argument("--wall", type=int, default=None,
                    help="wall-clock minutes per element (default: 2880 ot, "
                         "720 bob)")
    ap.add_argument("--mem-mb-per-slot", type=int, default=None,
                    help="minimum -M after right_size (for proxy canaries)")
    ap.add_argument("--n-queues", type=int, default=3)
    ap.add_argument("--after", default="")
    ap.add_argument("--min-slides", type=int, default=2900)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()

    features = a.features or (_out() / "features" / a.encoder)

    if a.action == "canary":
        canary(a.encoder, a.method, features, a.go, a.n_queries, a.wall)
    elif a.action == "fanout":
        if not a.epilogue:
            raise SystemExit("fanout needs --epilogue <canary .out>")
        batch = a.batch if a.batch is not None else (1 if a.method == "ot" else 20)
        wall = a.wall if a.wall is not None else (2880 if a.method == "ot" else 720)
        fanout(a.encoder, a.method, features, a.epilogue, a.go, batch,
               wall, a.n_queues, a.mem_mb_per_slot)
    else:
        if not a.after:
            raise SystemExit("combine needs --after <fanout job ids>")
        combine(a.encoder, a.method, a.after, a.go, a.min_slides)

    if not a.go:
        print("\n[dry run] nothing submitted. Re-run with --go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
