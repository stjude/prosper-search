#!/usr/bin/env python3
"""Finish the C5 new-encoder rungs (H-Optimus-1 = R2.1, RetCCL = R3.1/E4).

The OT and BoB per-query rows for both encoders completed overnight
2026-08-28 -> 29 (2,921/2,921 each; recorded in prosper_revision [decision record] B2).
What remained: combine -> mask -> evaluate -> IR metrics -> tie floors ->
paired comparison against the declared primary. This driver submits that
whole chain with LSF dependencies (`ended` between stages, per standing rule
8), so it runs unattended and every result records a commit that contains
this code ([decision record] C1 -- commit before submitting).

Sizing: the evaluators are the same code paths the C2 sweep ran 296 times
(188 s / 1.5 GB peak per cell at -n 8), so the C2 canary epilogue is reused as
the right-sizing measurement rather than re-canarying an unchanged code path.

Usage:
    submit_new_encoders.py all --go        # combine -> evals+metrics -> ties -> paired
    submit_new_encoders.py combine --go
    submit_new_encoders.py evals --after <combine job ids> --go
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from prosper.hpc import submit as S             # noqa: E402
from prosper.evaluation import paths            # noqa: E402

WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
OUT = paths.output_root()

ENCODERS = ["h_optimus_1", "retccl"]
SIM_METHODS = ["ot", "bob"]                          # generator keys
AGGS = ["ot_cosine", "yottixel", "median_pooling", "sampler"]
KS = (1, 3, 5, 10)
RULES = ("generous_majority", "absolute")
SLOTS = 8


def _c2_epilogue() -> dict:
    """The C2 sweep's canary epilogue: same evaluator, same matrix shape."""
    cands = sorted(glob.glob(str(OUT / "logs" / "c2_vote_rules_v2" / "*canary*.out")))
    cands += sorted(glob.glob(str(OUT / "logs" / "c2_vote_rules" / "*canary*.out")))
    for c in cands:
        epi = S.parse_epilogue(Path(c).read_text())
        if epi.get("max_mem_mb"):
            epi = dict(epi)
            epi["effective_cores"] = float(SLOTS)   # pin, as submit_c2_votes does
            print(f"[new-enc] sizing from {c}: {epi}")
            return epi
    raise SystemExit("no C2 canary epilogue found to size from")


def _dep(ids, kind="ended") -> str:
    ids = [i for i in ids if i]
    return " && ".join(f"{kind}({i})" for i in ids)


def combine(go: bool) -> list:
    res = S.Resources(slots=4, mem_mb_per_slot=8192, wall_minutes=60)
    jids = []
    for enc in ENCODERS:
        for m in SIM_METHODS:
            cmd = (f"PROSPER_MODULE=prosper.embeddings.combine_similarity_cli bash {WORKER} "
                   f"--encoder {enc} --method {m} "
                   f"--similarity-dir {OUT}/similarity/{enc}_{m} --min-slides 2900")
            jids.append(S.one(cmd, job_name=f"prosper_combine_{enc}_{m}", res=res,
                              log_dir=OUT / "logs" / "combine_similarity",
                              kind="cpu", queue="priority", dry_run=not go))
    return jids


def evals(after: list, go: bool) -> tuple:
    """slide_ci (2 rules x 4 k) + metrics_cli for every new encoder x aggregation."""
    epi = _c2_epilogue()
    dep = _dep(after)
    # --- voting F1 grid ---
    lines = []
    for enc in ENCODERS:
        for agg in AGGS:
            for rule in RULES:
                for k in KS:
                    dest = OUT / f"c5_{enc}" / f"slide_ci_{enc}_{agg}_top{k}_{rule}.json"
                    if dest.exists():
                        continue          # h_optimus_1 median/sampler generous already ran
                    lines.append(f"--model {enc} --method {agg} --topk {k} --tie-break {rule} "
                                 f"--iter 5000 --n-jobs {SLOTS} --task c5_{enc}")
    params = OUT / "c5_new_encoders" / "params_votes.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    params.write_text("\n".join(lines) + "\n")
    print(f"[new-enc] {len(lines)} voting cells -> {params}")
    res = S.Resources(slots=SLOTS, mem_mb_per_slot=2048, wall_minutes=30)
    vote_ids = S.fan_out(f"PROSPER_PARAMS={params} bash {WORKER}", n=len(lines),
                         job_name="prosper_newenc_votes", res=res,
                         log_dir=OUT / "logs" / "c5_new_encoders", n_queues=3,
                         canary_epilogue=epi, dry_run=not go, dependency=dep or None)
    # --- IR metric suite (lands in c2_metric_suite so paired.py sees it) ---
    mlines = [f"--model {enc} --method {agg} --depth 10 --iter 5000 --task c2_metric_suite"
              for enc in ENCODERS for agg in AGGS]
    mparams = OUT / "c5_new_encoders" / "params_metrics.txt"
    mparams.write_text("\n".join(mlines) + "\n")
    mres = S.Resources(slots=4, mem_mb_per_slot=4096, wall_minutes=40)
    met_ids = S.fan_out(f"PROSPER_MODULE=prosper.evaluation.metrics_cli "
                        f"PROSPER_PARAMS={mparams} bash {WORKER}", n=len(mlines),
                        job_name="prosper_newenc_metrics", res=mres,
                        log_dir=OUT / "logs" / "c5_new_encoders", n_queues=2,
                        canary_epilogue=epi, dry_run=not go, dependency=dep or None)
    return vote_ids, met_ids


def ties(after: list, go: bool) -> list:
    """Tie floors for the new encoders' set-based cells ([decision record] amendment 2)."""
    lines = [f"--model {enc} --method {agg} --topk 5 --tie-break generous_majority "
             f"--n-orders 200 --task c2_tie_sensitivity"
             for enc in ENCODERS for agg in ("yottixel", "ot_cosine")]
    params = OUT / "c5_new_encoders" / "params_ties.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    params.write_text("\n".join(lines) + "\n")
    epi = {"max_mem_mb": 1600.0, "effective_cores": float(SLOTS)}  # from the 40-cell tie sweep
    res = S.Resources(slots=SLOTS, mem_mb_per_slot=4096, wall_minutes=180)
    return S.fan_out(f"PROSPER_MODULE=prosper.evaluation.tie_sensitivity "
                     f"PROSPER_PARAMS={params} bash {WORKER}", n=len(lines),
                     job_name="prosper_newenc_ties", res=res,
                     log_dir=OUT / "logs" / "c2_tie_sensitivity", n_queues=2,
                     canary_epilogue=epi, dry_run=not go, dependency=_dep(after) or None)


PAIRED_TASK = "c5_paired_v3"   # v3 (2026-09-22): adds p_two_sided + Holm; v2 kept as recorded


def paired(after: list, go: bool) -> list:
    """Paired Δ vs the declared primary, every ranked list in c2_metric_suite,
    into PAIRED_TASK (earlier c5_paired/, c5_paired_v2/ artifacts stay as recorded)."""
    res = S.Resources(slots=8, mem_mb_per_slot=4096, wall_minutes=120)
    jids = []
    for k in (5, 10):
        cmd = (f"PROSPER_MODULE=prosper.evaluation.paired bash {WORKER} "
               f"--reference uni2_ot_cosine --k {k} --depth 10 --iter 5000 "
               f"--task {PAIRED_TASK}")
        jids.append(S.one(cmd, job_name=f"prosper_newenc_paired_k{k}", res=res,
                          log_dir=OUT / "logs" / "c5_paired", kind="cpu",
                          dry_run=not go, dependency=_dep(after) or None))
    return jids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["all", "combine", "evals", "ties", "paired"])
    ap.add_argument("--after", default="")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    after = [x for x in a.after.split(",") if x.strip()]
    if a.action == "combine":
        print(combine(a.go))
    elif a.action == "evals":
        print(evals(after, a.go))
    elif a.action == "ties":
        print(ties(after, a.go))
    elif a.action == "paired":
        print(paired(after, a.go))
    else:
        c = combine(a.go)
        v, m = evals(c, a.go)
        t = ties(c, a.go)
        p = paired(m, a.go)
        print(f"[new-enc] combine={c} votes={v} metrics={m} ties={t} paired={p}")
    if not a.go:
        print("\n[dry run] nothing submitted. Re-run with --go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
