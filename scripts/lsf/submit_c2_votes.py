#!/usr/bin/env python3
"""Submit the C2 voting-rule sweep (R4.b, execution spec task C2 part (b)).

C2 has three parts. Parts (a) and (c) are done: the archived ``any_hit`` grid
reproduced as the fidelity check, and ``prosper.evaluation.metrics_cli`` has
written the full 34-configuration IR-metric grid with persisted ranked lists.
What remains is part (b) -- the voting rule that is *implemented but never
swept*: ``analysis_lib``'s ``absolute`` (a genuine >k/2 majority that emits
``'Ambiguous'`` when no label clears the bar) against the published
``generous_majority`` (plurality, credited whenever the truth is anywhere in
the tied argmax set).

Grid: 33 configurations x k in {1,3,5,10} x 2 rules = **264 jobs**.

The spec says 256, counting only the 8 benchmark encoders x 4 aggregations.
CONCH/TITAN is added here (+8) because C5's paired arm found it is the *only*
configuration that significantly beats the declared primary on precision@5, so
its behaviour under the voting rule is now load-bearing for R3.1 rather than a
curiosity. The 8 uni2/ot_cosine ``generous_majority`` cells duplicate G2 and
are kept deliberately: they cost nothing and make the sweep self-checking
against a gate that already passed to 6 dp.

Usage:
    submit_c2_votes.py queues                     # live pressure
    submit_c2_votes.py canary --go                # one `absolute` element
    submit_c2_votes.py fanout --epilogue <.out> --go
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from prosper.hpc import submit as S             # noqa: E402
from prosper.hpc import queues as Q             # noqa: E402
from prosper.evaluation import paths            # noqa: E402

WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
OUT = paths.output_root()
TASK = "c2_vote_rules"

ENCODERS = ["uni2", "virchow2", "gigapath", "exaonepath",
            "phikon-v2", "ctranspath", "hibou", "simclr"]
AGGS = ["ot_cosine", "median_pooling", "sampler", "yottixel"]
COMBOS = [(e, a) for e in ENCODERS for a in AGGS] + [("titan", "titan")]
KS = (1, 3, 5, 10)
RULES = ("generous_majority", "absolute")

# 8 slots. Not 32, because `prosper.hpc.queues` steers -n>=32 exclusively to
# large_core_count (measured -- `short`/`priority` refused -n 32) and rule 7 is
# about spreading the sweep. Not 16 either: the canary measured
# effective_cores = 941.98 s CPU / 137 s wall = 6.9 against 16 reserved, so 16
# was over-provisioned. Unlike the GPU extractor, this slot count is *real*
# parallelism -- the bootstrap fans B=5000 replicates across a
# ProcessPoolExecutor -- so the measurement is trustworthy here and 8 is the
# honest size. It must also match `--n-jobs` in params.txt, which is written
# before right_size() runs; hence the pinning in fanout().
SLOTS = 8


def _params() -> Path:
    p = OUT / TASK / "params.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"--model {e} --method {a} --topk {k} --tie-break {r} "
        f"--iter 5000 --n-jobs {SLOTS} --task {TASK}"
        for r in RULES for e, a in COMBOS for k in KS
    ]
    p.write_text("\n".join(lines) + "\n")
    print(f"[c2-votes] {len(lines)} jobs -> {p}")
    return p


def canary(go: bool) -> None:
    """One `absolute` element. That rule has never run, so it is the new code
    path and therefore the only honest thing to canary."""
    res = S.Resources(slots=SLOTS, mem_mb_per_slot=2048, wall_minutes=25)
    S.canary(f"bash {WORKER} --model uni2 --method ot_cosine --topk 5 "
             f"--tie-break absolute --iter 5000 --n-jobs {SLOTS} --task {TASK}",
             job_name="prosper_c2_votes", res=res,
             log_dir=OUT / "logs" / TASK, dry_run=not go)


def fanout(epilogue_path: Path, go: bool) -> None:
    epi = S.parse_epilogue(epilogue_path.read_text())
    if not epi:
        raise SystemExit(f"no LSF epilogue in {epilogue_path}")
    params = _params()
    n = sum(1 for _ in params.read_text().splitlines() if _.strip())
    res = S.Resources(slots=SLOTS, mem_mb_per_slot=2048, wall_minutes=25)
    # Pin slots at SLOTS so they cannot drift away from the `--n-jobs SLOTS`
    # already baked into params.txt. right_size() would otherwise derive
    # int(6.9)+1 = 7 from the canary and leave 8 ProcessPoolExecutor workers
    # oversubscribing 7 reserved cores. Memory right-sizing is kept.
    epi = dict(epi)
    epi["effective_cores"] = float(SLOTS)
    S.fan_out(f"PROSPER_PARAMS={params} bash {WORKER}",
              n=n, job_name="prosper_c2_votes", res=res,
              log_dir=OUT / "logs" / TASK, n_queues=3,
              canary_epilogue=epi, dry_run=not go)


def tumor_metrics(epilogue_path: Path, go: bool) -> None:
    """The tumour-only column of the C2 IR-metric grid.

    ``c2_metric_suite`` holds 33 all-tissue configurations and exactly one
    tumour-only cell (uni2/ot_cosine, produced as the C5 ladder's rung-3
    input). Table S5 is the tumour-only analogue of Table S3 and R2.6 asks
    specifically about tumour-region restriction, so the other 32 cells are
    needed. Same code path as the all-tissue grid -- only ``--tumor_only``
    differs, which also flips ``drop_labels`` to ['MISC','NORMAL'] and the
    population to n=2,604 (standing rule 8: drop_labels is coupled to the
    population, never free-standing).
    """
    epi = S.parse_epilogue(epilogue_path.read_text())
    if not epi:
        raise SystemExit(f"no LSF epilogue in {epilogue_path}")
    task = "c2_metric_suite"
    params = OUT / task / "params_tumor_only.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"--model {e} --method {a} --depth 10 --iter 5000 "
             f"--tumor_only --task {task}" for e, a in COMBOS]
    params.write_text("\n".join(lines) + "\n")
    print(f"[c2-tumor] {len(lines)} configurations -> {params}")
    res = S.Resources(slots=4, mem_mb_per_slot=4096, wall_minutes=25)
    S.fan_out(f"PROSPER_MODULE=prosper.evaluation.metrics_cli "
              f"PROSPER_PARAMS={params} bash {WORKER}",
              n=len(lines), job_name="prosper_c2_tumor", res=res,
              log_dir=OUT / "logs" / task, n_queues=3,
              canary_epilogue=epi, dry_run=not go)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["queues", "canary", "fanout",
                                       "tumor-metrics"])
    ap.add_argument("--epilogue", type=Path)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if a.action == "queues":
        print(Q.report(kind="cpu", wall_minutes=25, slots=SLOTS))
        return 0
    if a.action == "canary":
        canary(a.go)
    elif a.action == "tumor-metrics":
        if not a.epilogue:
            raise SystemExit("tumor-metrics needs --epilogue <a completed .out>")
        tumor_metrics(a.epilogue, a.go)
    else:
        if not a.epilogue:
            raise SystemExit("fanout needs --epilogue <canary .out>")
        fanout(a.epilogue, a.go)
    if not a.go:
        print("\n[dry run] nothing submitted. Re-run with --go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
