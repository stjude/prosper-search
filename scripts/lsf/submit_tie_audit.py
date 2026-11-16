#!/usr/bin/env python3
"""Re-run the C2 voting sweep deterministically, and measure the tie-break floor.

Two sweeps, one cause. The C2 sweep already ran (264 all-tissue + 32 tumour-only
cells) but it ran against a reference library whose row order depended on
PYTHONHASHSEED, so every configuration with tied distances got an arbitrary
top-1 that changed between array elements. The tell was that `absolute` and
`generous_majority` -- which are mathematically identical at k=1 -- disagreed at
k=1 for 9 of 33 configurations, all of them Yottixel or TITAN.

  (1) `votes`  re-runs the whole sweep under commit 94dc8b7, into *new* task
      directories so the pre-fix results stay on disk for comparison rather than
      being overwritten. The 24 unaffected configurations are re-run too: they
      cost little and it means the delivered grid has one provenance instead of
      two.

  (2) `ties`   quantifies what the fix cannot fix. A deterministic tie-break is
      still an arbitrary one, so for each configuration this measures the
      interval macro F1 spans over `--n-orders` random library orders. The width
      of that interval is a floor on the differences this evaluation can
      resolve; the declared primary (uni2/ot_cosine) has zero ties and so a
      floor of exactly 0.

Usage:
    submit_tie_audit.py queues
    submit_tie_audit.py canary --which {votes,ties} --go
    submit_tie_audit.py fanout --which {votes,ties} --epilogue <.out> --go
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

ENCODERS = ["uni2", "virchow2", "gigapath", "exaonepath",
            "phikon-v2", "ctranspath", "hibou", "simclr"]
AGGS = ["ot_cosine", "median_pooling", "sampler", "yottixel"]
COMBOS = [(e, a) for e in ENCODERS for a in AGGS] + [("titan", "titan")]
KS = (1, 3, 5, 10)
RULES = ("generous_majority", "absolute")

# The four tumour-only masked matrices that exist cluster-wide. The other 29
# configurations have no tumour-only matrix, so `require()` fails by design.
TUMOR_COMBOS = [("uni2", "ot_cosine"), ("uni2", "yottixel"),
                ("virchow2", "ot_cosine"), ("virchow2", "yottixel")]

VOTE_TASK = "c2_vote_rules_v2"
VOTE_TUMOR_TASK = "c2_vote_rules_tumor_v2"
TIE_TASK = "c2_tie_sensitivity"

# Matches --n-jobs in params.txt; see submit_c2_votes.py for why 8 and not 32.
SLOTS = 8
# Each order is a full 2778-query prediction pass (~3 s), so 200 orders is
# ~10 min of real work -- too long for `short`'s 30 min wall once queueing and
# the 4-k sweep are added, hence the wall below and no `short` for `ties`.
N_ORDERS = 200


def _vote_params() -> Path:
    p = OUT / VOTE_TASK / "params.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"--model {e} --method {a} --topk {k} --tie-break {r} "
             f"--iter 5000 --n-jobs {SLOTS} --task {VOTE_TASK}"
             for r in RULES for e, a in COMBOS for k in KS]
    lines += [f"--model {e} --method {a} --topk {k} --tie-break {r} "
              f"--iter 5000 --n-jobs {SLOTS} --tumor_only --task {VOTE_TUMOR_TASK}"
              for r in RULES for e, a in TUMOR_COMBOS for k in KS]
    p.write_text("\n".join(lines) + "\n")
    print(f"[votes] {len(lines)} cells -> {p}")
    return p


def _tie_params() -> Path:
    p = OUT / TIE_TASK / "params.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    # k=5 is the reported depth for every configuration; the declared primary
    # additionally gets all four k so the zero-tie result is shown across the
    # whole grid rather than at one depth.
    lines = [f"--model {e} --method {a} --topk 5 "
             f"--tie-break generous_majority --n-orders {N_ORDERS} "
             f"--task {TIE_TASK}" for e, a in COMBOS]
    lines += [f"--model uni2 --method ot_cosine --topk {k} "
              f"--tie-break generous_majority --n-orders {N_ORDERS} "
              f"--task {TIE_TASK}" for k in (1, 3, 10)]
    lines += [f"--model {e} --method {a} --topk 5 "
              f"--tie-break generous_majority --n-orders {N_ORDERS} "
              f"--tumor_only --task {TIE_TASK}" for e, a in TUMOR_COMBOS]
    p.write_text("\n".join(lines) + "\n")
    print(f"[ties] {len(lines)} cells -> {p}")
    return p


def canary(which: str, go: bool) -> None:
    if which == "votes":
        res = S.Resources(slots=SLOTS, mem_mb_per_slot=2048, wall_minutes=25)
        S.canary(f"bash {WORKER} --model uni2 --method yottixel --topk 1 "
                 f"--tie-break absolute --iter 5000 --n-jobs {SLOTS} "
                 f"--task {VOTE_TASK}",
                 job_name="prosper_votes_v2", res=res,
                 log_dir=OUT / "logs" / VOTE_TASK, dry_run=not go)
    else:
        # New module, so this is the only honest thing to canary. uni2/yottixel
        # is the worst case: most ties, so the most permutation work.
        res = S.Resources(slots=SLOTS, mem_mb_per_slot=4096, wall_minutes=90)
        S.canary(f"PROSPER_MODULE=prosper.evaluation.tie_sensitivity bash {WORKER} "
                 f"--model uni2 --method yottixel --topk 5 "
                 f"--tie-break generous_majority --n-orders 20 --task {TIE_TASK}",
                 job_name="prosper_ties", res=res,
                 log_dir=OUT / "logs" / TIE_TASK, dry_run=not go)


def fanout(which: str, epilogue_path: Path, go: bool, n_queues: int) -> None:
    epi = S.parse_epilogue(epilogue_path.read_text())
    if not epi:
        raise SystemExit(f"no LSF epilogue in {epilogue_path}")
    # Pin slots so they cannot drift from the --n-jobs baked into params.txt.
    epi = dict(epi)
    epi["effective_cores"] = float(SLOTS)
    if which == "votes":
        params = _vote_params()
        res = S.Resources(slots=SLOTS, mem_mb_per_slot=2048, wall_minutes=25)
        name, task = "prosper_votes_v2", VOTE_TASK
        module = ""
    else:
        params = _tie_params()
        res = S.Resources(slots=SLOTS, mem_mb_per_slot=4096, wall_minutes=180)
        name, task = "prosper_ties", TIE_TASK
        module = "PROSPER_MODULE=prosper.evaluation.tie_sensitivity "
    n = sum(1 for ln in params.read_text().splitlines() if ln.strip())
    S.fan_out(f"{module}PROSPER_PARAMS={params} bash {WORKER}",
              n=n, job_name=name, res=res,
              log_dir=OUT / "logs" / task, n_queues=n_queues,
              canary_epilogue=epi, dry_run=not go)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["queues", "canary", "fanout"])
    ap.add_argument("--which", choices=["votes", "ties"], default="votes")
    ap.add_argument("--epilogue", type=Path)
    ap.add_argument("--n-queues", type=int, default=3)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if a.action == "queues":
        print(Q.report(kind="cpu", wall_minutes=25, slots=SLOTS))
        return 0
    if a.action == "canary":
        canary(a.which, a.go)
    else:
        if not a.epilogue:
            raise SystemExit("fanout needs --epilogue <canary .out>")
        fanout(a.which, a.epilogue, a.go, a.n_queues)
    if not a.go:
        print("\n[dry run] nothing submitted. Re-run with --go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
