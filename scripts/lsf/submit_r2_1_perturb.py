#!/usr/bin/env python3
"""Plan (and optionally submit) the R2.1 perturbation-extraction GPU array.

Dry run by default: prints the exact ``bsub`` line plus a GPU-hour / wall
estimate and writes the per-element parameter file the line refers to. Nothing
is submitted without ``--go``.

Elements are contiguous ``--lo/--hi`` ranges over the frozen ``queries.csv``
order, packed greedily by tile count so that each element's predicted wall
(model load + tiles / measured tiles-per-second) stays within ``--budget-hours``
(default 6 h of an 8 h ``-W``). Throughput is per *slide tile* for one
invocation of the extractor, i.e. all twelve conditions and every requested
encoder of that pass; take it from the pilot run summaries
(``--pilot-run-root``) or give it with ``--tiles-per-sec``.

Each element runs ``prosper_job.sh`` in array mode (one params line per
``$LSB_JOBINDEX``). The extractor is idempotent (verified skip, never
overwrite), so resubmitting the same params file resumes.

Usage:
    submit_r2_1_perturb.py --encoders h_optimus_1 --pilot-run-root <pilot/run>
    submit_r2_1_perturb.py --encoders uni2,virchow2 --tiles-per-sec 9.5 --go
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from prosper.evaluation import paths                          # noqa: E402
from prosper.evaluation import r2_1_perturb_extract_cli as X  # noqa: E402

WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
MODULE = "prosper.evaluation.r2_1_perturb_extract_cli"


def pilot_throughput(run_root: Path, encoders: list) -> dict:
    """Aggregate tiles/s and LSF-free timings of pilot run summaries whose
    encoder set equals ``encoders`` exactly."""
    best = None
    for p in sorted((run_root / "runs").glob("*.json")):
        s = json.loads(p.read_text())
        if sorted(s.get("encoders", [])) != sorted(encoders):
            continue
        tiles = sum(s.get("tiles_written", {}).values()) / max(1, len(encoders))
        compute = s["wall_seconds"]          # slide loop only; model load is separate
        if tiles <= 0 or compute <= 0:
            continue
        cand = {"summary": str(p), "slide_tiles": tiles, "compute_seconds": compute,
                "model_load_seconds": s.get("model_load_seconds", 0.0),
                "tiles_per_sec": tiles / compute}
        if best is None or cand["slide_tiles"] > best["slide_tiles"]:
            best = cand
    if best is None:
        raise SystemExit(f"no pilot run summary for encoders={encoders} under {run_root}/runs")
    return best


def chunk(rows: list, tiles_per_sec: float, budget_s: float, load_s: float) -> list:
    """Greedy contiguous packing; a single slide larger than the budget still
    gets its own element (reported)."""
    chunks, lo, acc = [], 1, 0.0
    for i, r in enumerate(rows, start=1):
        t = int(r["feature_rows"]) / tiles_per_sec
        if acc and load_s + acc + t > budget_s:
            chunks.append((lo, i - 1, acc))
            lo, acc = i, 0.0
        acc += t
    chunks.append((lo, len(rows), acc))
    return chunks


def main() -> int:
    out = paths.output_root()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", required=True,
                    help="exactly as passed to the extractor, e.g. h_optimus_1 or uni2,virchow2")
    ap.add_argument("--pilot-run-root", type=Path,
                    default=out / X.RUN_SUBDIR / "pilot" / "run")
    ap.add_argument("--tiles-per-sec", type=float, default=0.0,
                    help="override the pilot-derived slide-tiles/s for this pass")
    ap.add_argument("--model-load-seconds", type=float, default=0.0,
                    help="override the pilot-derived model-load time")
    ap.add_argument("--queries", type=Path, default=out / X.INPUT_SUBDIR / "queries.csv")
    ap.add_argument("--queue", default="gpu")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0, help="default slots-1")
    ap.add_argument("--peak-mem-mb", type=float, default=0.0,
                    help="pilot LSF 'Max Memory' (MB) for this pass; per-slot -M = peak*1.5/slots")
    ap.add_argument("--wall-minutes", type=int, default=480)
    ap.add_argument("--budget-hours", type=float, default=6.0,
                    help="predicted compute per element; leaves slack under -W")
    ap.add_argument("--max-concurrent", type=int, default=0,
                    help="optional LSF %%N array limit (0 = none)")
    ap.add_argument("--extra-args", default="",
                    help="appended verbatim to every params line (e.g. '--no-hash-weights')")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()

    encs = [e.strip() for e in a.encoders.split(",") if e.strip()]
    for e in encs:
        if e not in X.AMP_DEFAULTS:
            raise SystemExit(f"unknown encoder {e}")
    _, rows = X.load_queries(a.queries, X.QUERIES_SHA256)
    total_tiles = sum(int(r["feature_rows"]) for r in rows)

    if a.tiles_per_sec > 0:
        tps, load_s, src = a.tiles_per_sec, a.model_load_seconds or 300.0, "command line"
    else:
        pt = pilot_throughput(a.pilot_run_root, encs)
        tps = pt["tiles_per_sec"]
        load_s = a.model_load_seconds or pt["model_load_seconds"]
        src = pt["summary"]
    budget_s = a.budget_hours * 3600.0
    chunks = chunk(rows, tps, budget_s, load_s)
    over = [c for c in chunks if load_s + c[2] > budget_s]
    if load_s + max(c[2] for c in chunks) > a.wall_minutes * 60:
        raise SystemExit("an element's predicted wall exceeds -W; lower --budget-hours "
                         "or raise --wall-minutes")

    tag = "_".join(encs)
    workers = a.num_workers or max(1, a.slots - 1)
    run_root = out / X.RUN_SUBDIR
    lines = [f"--encoders {','.join(encs)} --lo {lo} --hi {hi} --num-workers {workers}"
             + (f" {a.extra_args}" if a.extra_args else "") for lo, hi, _ in chunks]
    text = "\n".join(lines) + "\n"
    # Content-addressed: a params file an array may be reading is never rewritten.
    digest = hashlib.sha256(text.encode()).hexdigest()
    params = run_root / "params" / f"{tag}_{len(chunks)}el_{digest[:12]}.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    if params.exists():
        if params.read_text() != text:
            raise SystemExit(f"{params} exists with different content")
    else:
        params.write_text(text)

    if a.peak_mem_mb > 0:
        mem = max(2048, int(math.ceil(a.peak_mem_mb * 1.5 / a.slots)))
    else:
        mem = 4096
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    n = len(chunks)
    arr = f"[1-{n}]" + (f"%{a.max_concurrent}" if a.max_concurrent else "")
    cmd = (f"PROSPER_OUT={out} HF_HUB_OFFLINE=1 PROSPER_MODULE={MODULE} "
           f"PROSPER_PARAMS={params} bash {WORKER}")
    argv = ["bsub", "-J", f"r21_perturb_{tag}{arr}", "-q", a.queue,
            "-n", str(a.slots), "-M", str(mem),
            "-R", f"rusage[mem={mem}] span[hosts=1]", "-gpu", "num=1",
            "-W", str(a.wall_minutes),
            "-oo", str(log_dir / f"r21_perturb_{tag}.%J_%I.out"),
            "-eo", str(log_dir / f"r21_perturb_{tag}.%J_%I.err"), cmd]

    gpu_h = (total_tiles / tps + n * load_s) / 3600.0
    elem_h = [(load_s + c[2]) / 3600.0 for c in chunks]
    print(f"[plan] encoders={encs} slides={len(rows)} tiles={total_tiles}")
    print(f"[plan] throughput {tps:.2f} slide-tiles/s (all 12 conditions x {len(encs)} encoder(s)); "
          f"model load {load_s:.0f}s; source: {src}")
    print(f"[plan] {n} elements; predicted element wall min/median/max = "
          f"{min(elem_h):.2f}/{sorted(elem_h)[n // 2]:.2f}/{max(elem_h):.2f} h "
          f"(budget {a.budget_hours} h, -W {a.wall_minutes} min)")
    if over:
        print(f"[plan] WARNING {len(over)} single-slide element(s) exceed the budget")
    print(f"[plan] total ~{gpu_h:.1f} GPU-hours for this pass")
    for c in (16, 24):
        waves = math.ceil(n / c)
        print(f"[plan] at {c} concurrent elements: ~{gpu_h / min(c, n):.1f} h if perfectly packed, "
              f"<= {waves} waves x {max(elem_h):.2f} h = {waves * max(elem_h):.1f} h worst case")
    print(f"[plan] -n {a.slots} -M {mem} per slot ({a.slots * mem / 1024:.1f} GB total); params {params}")
    printable = " ".join(shlex.quote(x) for x in argv)
    print(printable)
    if not a.go:
        print("\n[dry run] nothing submitted. Re-run with --go.")
        return 0
    env = os.environ.copy()
    env["LSB_JOB_REPORT_MAIL"] = "N"
    res = subprocess.run(argv, capture_output=True, text=True, env=env)
    print(res.stdout.strip())
    print(res.stderr.strip(), file=sys.stderr)
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
