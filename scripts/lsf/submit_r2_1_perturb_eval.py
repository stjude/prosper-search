#!/usr/bin/env python3
"""R2.1 perturbation evaluation (P2): SAMPLER -> L1 -> hit@k/P@k + paired deltas.

Without --go every stage only PRINTS the exact bsub lines; with --go it submits and appends one
JSON line per submitted job to $PROSPER_OUT/r2_1_perturb_20261003/eval/pilot_jobs.jsonl.

Stages
  production  r2_1_perturb_eval_cli run, one job per encoder (coordinator submits)
  fidelity    historical Fig. 4A fidelity chain per encoder (uni2, virchow2):
              hist-sampler shards -> hist-l1 per arm (ended(sampler)) -> hist-eval (ended(l1))
  timing      bounded --allow-partial pilot on a slide subset of the production features

Usage: submit_r2_1_perturb_eval.py production [--encoders ...] [--go]
       submit_r2_1_perturb_eval.py fidelity [--encoders uni2,virchow2] [--go]
       submit_r2_1_perturb_eval.py timing --encoder h_optimus_1 --slides-file F --out-root D [--go]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from prosper.hpc import submit as S              # noqa: E402
from prosper.evaluation import paths             # noqa: E402

OUT = paths.output_root()
WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
ROOT = OUT / "r2_1_perturb_20261003"
EVAL = ROOT / "eval"
FID = EVAL / "fidelity"
JOURNAL = EVAL / "pilot_jobs.jsonl"
MODULE = "prosper.evaluation.r2_1_perturb_eval_cli"
QUEUE = "large_core_count"
HIST_ARMS = ("rotate_90", "rotate_180", "rotate_270", "flip_h", "flip_v", "reinhard",
             "brightness", "saturation", "contrast", "blur", "spatial_shift", "identity_fp32",
             "identity_fp16_reference")
# Production sizing from the timing pilot (see eval/IMPLEMENTATION_REPORT.md).
# Measured: SAMPLER of 88 slides/116,503 tiles x 12 conditions in 6.6 s on 8 workers; full-scale
# 2,778^2 L1 in 14 s (15,360-d) / 17 s (25,600-d) on 8 threads. Parent memory is dominated by the
# float64 (2778, 10, D) x 12 SAMPLER arrays (4.1 GB at 1,536-d, 6.8 GB at 2,560-d); each worker
# holds one slide (<= 5,051 tiles x 12 conditions). LSF -M here is the per-slot share of a
# per-job limit.
PROD = {"h_optimus_1": S.Resources(slots=8, mem_mb_per_slot=4096, wall_minutes=120),
        "uni2": S.Resources(slots=8, mem_mb_per_slot=4096, wall_minutes=120),
        "virchow2": S.Resources(slots=8, mem_mb_per_slot=5120, wall_minutes=180)}


def _journal(entry: dict) -> None:
    entry = {"time": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), **entry}
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _params(path: Path, lines: list) -> Path:
    data = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != data:
            raise SystemExit(f"{path} exists with different content; refusing to overwrite")
    else:
        path.write_text(data)
    return path


def _submit(go, cmd, *, name, res, log_dir, array=None, dependency=None, purpose=""):
    # Never move a submitted job with `bmod -q`: the site wrapper resets its memory request to
    # the 2.5 GB default. Choose the queue at submission (--queue).
    jid = S._bsub(cmd, job_name=name, queue=QUEUE, res=res, log_dir=log_dir, array=array,
                  dry_run=not go, dependency=dependency)
    if go:
        _journal({"job_id": jid, "name": name, "array": array, "queue": QUEUE,
                  "slots": res.slots, "mem_mb_per_slot": res.mem_mb_per_slot,
                  "wall_minutes": res.wall_minutes, "dependency": dependency, "cmd": cmd,
                  "purpose": purpose})
    return jid


def fidelity(go: bool, encoders, n_shards: int) -> None:
    logs = FID / "logs"
    for enc in encoders:
        p = _params(FID / "params" / f"hist_sampler_{enc}.txt",
                    [f"hist-sampler --encoder {enc} --shard {i} --n-shards {n_shards} "
                     f"--out-dir {FID / 'sampler'}" for i in range(n_shards)])
        j1 = _submit(go, f"PROSPER_MODULE={MODULE} PROSPER_PARAMS={p} bash {WORKER}",
                     name=f"r21pe_hs_{enc}", array=f"1-{n_shards}", log_dir=logs,
                     res=S.Resources(slots=1, mem_mb_per_slot=8192, wall_minutes=60),
                     purpose="fidelity: historical SAMPLER recompute + stored comparison")
        p = _params(FID / "params" / f"hist_l1_{enc}.txt",
                    [f"hist-l1 --encoder {enc} --arm {arm} --sampler-dir {FID / 'sampler'} "
                     f"--n-shards {n_shards} --out-dir {FID / 'l1'}" for arm in HIST_ARMS])
        dep = f"ended({j1})" if j1 else None
        j2 = _submit(go, f"PROSPER_MODULE={MODULE} PROSPER_PARAMS={p} bash {WORKER}",
                     name=f"r21pe_hl_{enc}", array=f"1-{len(HIST_ARMS)}", log_dir=logs,
                     res=S.Resources(slots=1, mem_mb_per_slot=6144, wall_minutes=60),
                     dependency=dep, purpose="fidelity: L1 of one arm vs historical references")
        dep = f"ended({j2})" if j2 else None
        _submit(go, f"PROSPER_MODULE={MODULE} bash {WORKER} hist-eval --encoder {enc} "
                    f"--sampler-dir {FID / 'sampler'} --l1-dir {FID / 'l1'} "
                    f"--out-dir {FID / 'eval'}",
                name=f"r21pe_he_{enc}", log_dir=logs,
                res=S.Resources(slots=1, mem_mb_per_slot=16384, wall_minutes=120),
                dependency=dep, purpose="fidelity: Fig. 4A comparison")


def production(go: bool, encoders, cls_only: bool) -> None:
    for enc in encoders:
        res = PROD[enc]
        extra = " --virchow2-cls-only" if cls_only else ""
        _submit(go, f"PROSPER_MODULE={MODULE} bash {WORKER} run --encoders {enc} "
                    f"--workers {res.slots} --threads {res.slots}{extra}",
                name=f"r21pe_run_{enc}{'_cls' if cls_only else ''}", res=res,
                log_dir=EVAL / "logs", purpose="production evaluation")


def timing(go: bool, encoder: str, slides_file: Path, out_root: Path, slots: int) -> None:
    res = S.Resources(slots=slots, mem_mb_per_slot=4096, wall_minutes=60)
    _submit(go, f"PROSPER_MODULE={MODULE} bash {WORKER} run --encoders {encoder} "
                f"--slides-subset {slides_file} --allow-partial --out-root {out_root} "
                f"--workers {slots} --threads {slots}",
            name=f"r21pe_timing_{encoder}", res=res, log_dir=EVAL / "logs",
            purpose="timing pilot on available production features (descriptive only)")


def main() -> int:
    global QUEUE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("production", "fidelity", "timing"))
    ap.add_argument("--encoders", default="")
    ap.add_argument("--encoder", default="h_optimus_1")
    ap.add_argument("--n-shards", type=int, default=16)
    ap.add_argument("--virchow2-cls-only", action="store_true")
    ap.add_argument("--slides-file", type=Path)
    ap.add_argument("--out-root", type=Path)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--queue", default=QUEUE, choices=("large_core_count", "standard"))
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    QUEUE = a.queue
    if a.stage == "fidelity":
        fidelity(a.go, (a.encoders or "uni2,virchow2").split(","), a.n_shards)
    elif a.stage == "production":
        encs = (a.encoders or ("virchow2" if a.virchow2_cls_only
                               else "h_optimus_1,uni2,virchow2")).split(",")
        production(a.go, encs, a.virchow2_cls_only)
    else:
        timing(a.go, a.encoder, a.slides_file, a.out_root, a.slots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
