#!/usr/bin/env python3
"""Submit S1a (WSI header/scanner-metadata extraction) to LSF.

Canary-then-fan-out, per ``prosper.hpc.submit`` and execution-spec rule 3:

  submit_s1a.py canary --go
  submit_s1a.py fanout --epilogue <canary .out> --go
  submit_s1a.py merge                      # after the sweep completes, no LSF needed
  submit_s1a.py queues                     # live queue pressure, cpu, -n 1

2,921 slides are split into ``BATCH``-sized contiguous ranges over the sorted
``Slide.Scan.File`` id list (see ``prosper.evaluation.slide_headers``), one
LSF array element per range. Each element writes its own chunk CSV -- no
shared-file writer contention -- so ``merge`` is the only step that touches
all of them at once, and it is cheap enough to run without LSF.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from prosper.hpc import submit as S             # noqa: E402
from prosper.hpc import queues as Q             # noqa: E402
from prosper.evaluation import paths            # noqa: E402
from prosper.evaluation import slide_headers as SH  # noqa: E402

WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
OUT = paths.output_root()
TASK = "s1a_slide_headers"
# ~0.27 s/slide measured locally (mostly network-filesystem latency reading
# the header, not CPU) -> a 50-slide batch is ~15 s of work per array element,
# well inside even the 30-minute `short`/`heavy_io` wall limit, while keeping
# the array small enough (~59 elements) that per-job LSF overhead does not
# dominate.
BATCH = 50


def _ranges(n: int, batch: int):
    return [(s + 1, min(s + batch, n)) for s in range(0, n, batch)]


def _cmd(args: str) -> str:
    return f"PROSPER_MODULE=prosper.evaluation.slide_headers bash {WORKER} {args}"


def canary(go: bool) -> None:
    n = len(SH._sorted_slide_ids())
    ranges = _ranges(n, BATCH)
    lo, hi = ranges[0]
    log = OUT / "logs" / TASK
    # Header reads are I/O-bound, not CPU-bound: 1 slot is enough. Memory for
    # opening a tiled TIFF header is well under 1 GB; reserve modestly and
    # right-size from the canary epilogue before fanning out.
    res = S.Resources(slots=1, mem_mb_per_slot=1024, wall_minutes=15)
    print(f"[s1a] {n} slides -> {len(ranges)} batches of <= {BATCH}; "
          f"canary covers rows [{lo},{hi}]")
    S.canary(_cmd(f"extract --start {lo} --end {hi} --task {TASK}"),
             job_name="prosper_s1a", res=res, log_dir=log, dry_run=not go)


def fanout(epilogue_path: Path, go: bool) -> None:
    n = len(SH._sorted_slide_ids())
    ranges = _ranges(n, BATCH)[1:]  # the canary already covered the first batch
    log = OUT / "logs" / TASK
    epi = S.parse_epilogue(epilogue_path.read_text())
    if not epi:
        raise SystemExit(
            f"no LSF epilogue found in {epilogue_path}. The canary must have "
            "completed and -oo must have preserved it.")
    print(f"[s1a-fanout] canary epilogue: {epi}")

    params = OUT / TASK / "params.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"extract --start {lo} --end {hi} --task {TASK}" for lo, hi in ranges]
    params.write_text("\n".join(lines) + "\n")
    print(f"[s1a-fanout] wrote {len(lines)} params to {params}")

    res = S.Resources(slots=1, mem_mb_per_slot=1024, wall_minutes=15)
    S.fan_out(
        f"PROSPER_MODULE=prosper.evaluation.slide_headers PROSPER_PARAMS={params} "
        f"bash {WORKER}",
        n=len(lines), job_name="prosper_s1a_arr", res=res, log_dir=log,
        n_queues=2, canary_epilogue=epi, dry_run=not go)


def merge_cmd() -> None:
    summary = SH.merge(TASK)
    print(json.dumps(summary, indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["canary", "fanout", "merge", "queues"])
    ap.add_argument("--epilogue", type=Path)
    ap.add_argument("--go", action="store_true",
                    help="actually submit (default is a dry run)")
    a = ap.parse_args()

    if a.cmd == "queues":
        print(Q.report(kind="cpu", wall_minutes=15, slots=1))
        return 0
    if a.cmd == "canary":
        canary(a.go)
    elif a.cmd == "fanout":
        if not a.epilogue:
            raise SystemExit("fanout needs --epilogue <canary .out file>")
        fanout(a.epilogue, a.go)
    else:
        merge_cmd()
        return 0

    if not a.go:
        print("\n[dry run] nothing submitted. Re-run with --go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
