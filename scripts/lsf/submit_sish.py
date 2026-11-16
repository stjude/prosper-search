#!/usr/bin/env python3
"""C14: SISH full-system baseline over the internal 2,921-slide index (R3.1).

Chain: extract (GPU, canary->fanout) -> build (CPU) -> search (CPU,
canary->fanout) -> matrix -> evaluate (standard cli rows + metrics + ties +
paired). Every stage is idempotent (skip-if-loadable), so resubmission is a
resume. See prosper/baselines/sish_cli.py for the design decisions and ADR
0015 in the revision workspace for why this was reopened.

Usage:
    submit_sish.py extract-canary --go
    submit_sish.py extract-fanout --epilogue <canary.out> --go
    submit_sish.py build --go [--after <ids>]
    submit_sish.py search-canary --go [--after <ids>]
    submit_sish.py search-fanout --epilogue <canary.out> --go
    submit_sish.py matrix --go [--after <ids>]
    submit_sish.py evaluate --go [--after <ids>]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from prosper.hpc import submit as S             # noqa: E402
from prosper.evaluation import paths            # noqa: E402

WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
MODULE = "prosper.baselines.sish_cli"
OUT = paths.output_root()
N_SLIDES = 2921
LOGS = OUT / "logs" / "sish"

EXTRACT_SLOTS = 8
EXTRACT_BATCH = 60          # slides per element -> 49 elements
SEARCH_BATCH = 30           # queries per element -> 98 elements


def _dep(ids, kind="ended") -> str:
    ids = [i for i in ids if i]
    return " && ".join(f"{kind}({i})" for i in ids)


def extract_canary(go: bool, n_slides: int = 4, wall: int = 30):
    res = S.Resources(slots=EXTRACT_SLOTS, mem_mb_per_slot=4096,
                      wall_minutes=wall, gpus=1)
    return S.canary(f"PROSPER_MODULE={MODULE} bash {WORKER} extract "
                    f"--lo 1 --hi {n_slides} --num-workers {EXTRACT_SLOTS-2} "
                    f"--batch-size 8",
                    job_name="prosper_sish_extract", res=res, kind="gpu",
                    log_dir=LOGS, dry_run=not go)


def extract_fanout(epilogue_path: Path, go: bool, batch: int, wall: int,
                   n_queues: int):
    epi = S.parse_epilogue(epilogue_path.read_text())
    if not epi:
        raise SystemExit(f"no LSF epilogue in {epilogue_path}")
    n_elem = math.ceil(N_SLIDES / batch)
    params = OUT / "sish" / "params_extract.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n_elem):
        lo, hi = i * batch + 1, min(N_SLIDES, (i + 1) * batch)
        lines.append(f"extract --lo {lo} --hi {hi} "
                     f"--num-workers {EXTRACT_SLOTS-2} --batch-size 8")
    params.write_text("\n".join(lines) + "\n")
    peak = epi.get("max_mem_mb") or 8192
    mem = max(2048, int(peak * 2.0 / EXTRACT_SLOTS) + 1)
    res = S.Resources(slots=EXTRACT_SLOTS, mem_mb_per_slot=mem,
                      wall_minutes=wall, gpus=1)
    return S.fan_out(f"PROSPER_MODULE={MODULE} PROSPER_PARAMS={params} bash {WORKER}",
                     n=n_elem, job_name="prosper_sish_extract", res=res,
                     kind="gpu", n_queues=n_queues, canary_epilogue=epi,
                     log_dir=LOGS, dry_run=not go)


def build(after, go: bool):
    res = S.Resources(slots=4, mem_mb_per_slot=8192, wall_minutes=120)
    return S.one(f"PROSPER_MODULE={MODULE} bash {WORKER} build",
                 job_name="prosper_sish_build", res=res, kind="cpu",
                 queue="priority", log_dir=LOGS, dry_run=not go,
                 dependency=_dep(after) or None)


def search_canary(after, go: bool, n_queries: int = 3, wall: int = 60):
    # veb.pkl is 4.75 GB on disk and ~38 GB unpickled (the build's peak): each
    # search element holds the whole tree, so reserve accordingly.
    res = S.Resources(slots=8, mem_mb_per_slot=8192, wall_minutes=wall)
    return S.canary(f"PROSPER_MODULE={MODULE} bash {WORKER} search "
                    f"--lo 1 --hi {n_queries}",
                    job_name="prosper_sish_search", res=res, kind="cpu",
                    log_dir=LOGS, dry_run=not go)


def search_fanout(epilogue_path: Path, go: bool, batch: int, wall: int,
                  n_queues: int):
    epi = S.parse_epilogue(epilogue_path.read_text())
    if not epi:
        raise SystemExit(f"no LSF epilogue in {epilogue_path}")
    n_elem = math.ceil(N_SLIDES / batch)
    params = OUT / "sish" / "params_search.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n_elem):
        lo, hi = i * batch + 1, min(N_SLIDES, (i + 1) * batch)
        lines.append(f"search --lo {lo} --hi {hi}")
    params.write_text("\n".join(lines) + "\n")
    res = S.Resources(slots=8, mem_mb_per_slot=8192, wall_minutes=wall)
    return S.fan_out(f"PROSPER_MODULE={MODULE} PROSPER_PARAMS={params} bash {WORKER}",
                     n=n_elem, job_name="prosper_sish_search", res=res,
                     kind="cpu", n_queues=n_queues, canary_epilogue=epi,
                     log_dir=LOGS, dry_run=not go)


def matrix(after, go: bool):
    res = S.Resources(slots=2, mem_mb_per_slot=8192, wall_minutes=60)
    return S.one(f"PROSPER_MODULE={MODULE} bash {WORKER} matrix",
                 job_name="prosper_sish_matrix", res=res, kind="cpu",
                 queue="priority", log_dir=LOGS, dry_run=not go,
                 dependency=_dep(after) or None)


def evaluate(after, go: bool):
    """Standard rows: voting F1 grid (2 rules x 4 k) -> c14_sish; IR metric
    suite -> c2_metric_suite (so paired.py sees it); tie floor; paired v2."""
    dep = _dep(after)
    epi = {"max_mem_mb": 1600.0, "effective_cores": 8.0}   # C2 evaluator sizing
    lines = []
    for rule in ("generous_majority", "absolute"):
        for k in (1, 3, 5, 10):
            lines.append(f"--model sish --method system --topk {k} "
                         f"--tie-break {rule} --iter 5000 --n-jobs 8 "
                         f"--task c14_sish")
    params = OUT / "sish" / "params_votes.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    params.write_text("\n".join(lines) + "\n")
    res = S.Resources(slots=8, mem_mb_per_slot=2048, wall_minutes=30)
    vote_ids = S.fan_out(f"PROSPER_PARAMS={params} bash {WORKER}", n=len(lines),
                         job_name="prosper_sish_votes", res=res,
                         log_dir=LOGS, n_queues=2, canary_epilogue=epi,
                         dry_run=not go, dependency=dep or None)
    mres = S.Resources(slots=4, mem_mb_per_slot=4096, wall_minutes=40)
    met = S.one(f"PROSPER_MODULE=prosper.evaluation.metrics_cli bash {WORKER} "
                f"--model sish --method system --depth 10 --iter 5000 "
                f"--task c2_metric_suite",
                job_name="prosper_sish_metrics", res=mres, kind="cpu",
                queue="priority", log_dir=LOGS, dry_run=not go,
                dependency=dep or None)
    tres = S.Resources(slots=8, mem_mb_per_slot=4096, wall_minutes=180)
    tie = S.one(f"PROSPER_MODULE=prosper.evaluation.tie_sensitivity bash {WORKER} "
                f"--model sish --method system --topk 5 "
                f"--tie-break generous_majority --n-orders 200 "
                f"--task c2_tie_sensitivity",
                job_name="prosper_sish_ties", res=tres, kind="cpu",
                log_dir=LOGS, dry_run=not go, dependency=dep or None)
    pres = S.Resources(slots=8, mem_mb_per_slot=4096, wall_minutes=120)
    paired = []
    for k in (5, 10):
        paired.append(S.one(
            f"PROSPER_MODULE=prosper.evaluation.paired bash {WORKER} "
            f"--reference uni2_ot_cosine --k {k} --depth 10 --iter 5000 "
            f"--task c5_paired_v2",
            job_name=f"prosper_sish_paired_k{k}", res=pres, kind="cpu",
            dry_run=not go, log_dir=LOGS,
            dependency=_dep([met]) or None))
    return vote_ids, met, tie, paired


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["extract-canary", "extract-fanout",
                                       "build", "search-canary",
                                       "search-fanout", "matrix", "evaluate"])
    ap.add_argument("--after", default="")
    ap.add_argument("--epilogue", default="")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--wall", type=int, default=0)
    ap.add_argument("--n-queues", type=int, default=2)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    after = [x for x in a.after.split(",") if x.strip()]
    if a.action == "extract-canary":
        print(extract_canary(a.go))
    elif a.action == "extract-fanout":
        print(extract_fanout(Path(a.epilogue), a.go,
                             a.batch or EXTRACT_BATCH, a.wall or 120,
                             a.n_queues))
    elif a.action == "build":
        print(build(after, a.go))
    elif a.action == "search-canary":
        print(search_canary(after, a.go))
    elif a.action == "search-fanout":
        print(search_fanout(Path(a.epilogue), a.go,
                            a.batch or SEARCH_BATCH, a.wall or 240,
                            a.n_queues))
    elif a.action == "matrix":
        print(matrix(after, a.go))
    elif a.action == "evaluate":
        print(evaluate(after, a.go))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
