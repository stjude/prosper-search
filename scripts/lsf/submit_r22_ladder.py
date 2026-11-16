#!/usr/bin/env python3
"""R2.2: does retrieval need all tiles, or is the 15% mosaic enough?

A review comment asked for a comparison against all tiles. The plan of record
budgeted 2-3 weeks of GPU for it; measured, extraction runs at 290 tiles/s and
the whole ~26M-tile pass is about an hour, so the cost was never the obstacle.
The *confound* is.

Standing rule 4: the published 15% mosaic and the all-patches lists are NOT
nested. all-patches admits only fully-tissue tiles (`tissue_coverage == 1.0`)
while the mosaic retained partial-tissue tiles, and pooled overlap is 93.7%
(66.1% on small slides). So subsetting all-patches features can never reproduce
a published 15% number, and a naive "15% vs 100%" table would confound tile
*count* with tile *identity* -- exactly the thing a reader is trying to
isolate.

This ladder therefore runs entirely INSIDE the all-patches universe: 15/30/50%
are nested prefixes of one per-slide permutation seeded from
`sha256(slide_id) ^ 20260827`, so every rung is a strict subset of every larger
one and only tile count varies. The published mosaic arm stays separate and
unchanged.

Why this exists as a script rather than as eight inline bsubs: the first attempt
chained on `-w done(<extraction array>)`, and when one element hit the memory
limit `done` became unsatisfiable, so LSF killed all 8 downstream jobs with no
exit code and no log. The chain now uses `ended` -- which fires whether or not
every element succeeded -- and `aggregate_cli --min-slides` does the
completeness check where it can fail loudly.

Usage:
    submit_r22_ladder.py --after 317769442,317776224 --go
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from prosper.evaluation import paths            # noqa: E402
from prosper.hpc import queues as Q             # noqa: E402

WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
OUT = paths.output_root()
FEATURES = OUT / "features" / "uni2_all_patches"

# 2,921 mosaic slides minus the 34 with no all-patches coordinate list. The
# guard is deliberately a floor, not equality: the 34 are named in each
# extraction provenance JSON, and if that count is off the aggregation should
# still refuse rather than quietly build a short matrix.
EXPECT_SLIDES = 2887

FRACTIONS = [(15, 0.15), (30, 0.30), (50, 0.50), (100, 1.0)]
KS = (1, 3, 5, 10)


def _bsub(args: list, name: str, queue: str, cmd: str, dry: bool) -> str | None:
    full = ["bsub", "-q", queue, "-J", name,
            "-oo", str(OUT / "logs" / "r22_ladder" / f"{name}.%J.out"),
            "-eo", str(OUT / "logs" / "r22_ladder" / f"{name}.%J.err")] + args + [cmd]
    if dry:
        print("  DRY:", " ".join(full[:14]), "...")
        return None
    r = subprocess.run(full, capture_output=True, text=True)
    print("  " + (r.stdout or r.stderr).strip())
    if r.returncode != 0:
        raise SystemExit(f"bsub failed for {name}")
    return r.stdout.split("<")[1].split(">")[0]


def paired(go: bool) -> None:
    """Phase 2: bound the tile-count effect instead of merely failing to see it.

    The rung evaluations give four *unpaired* CIs, each ~0.045 wide, and the
    point estimates sit 0.001-0.011 apart. Overlapping unpaired intervals do not
    establish equivalence -- they only fail to establish a difference, which is a
    much weaker statement and not one that answers R2.2. The rungs share their
    query set exactly (same 2,887 slides, nested tile subsets), so a paired
    bootstrap on the per-query difference removes the between-slide variance
    that dominates those intervals and yields a CI on the *effect* itself.

    `prosper.evaluation.paired` consumes the ranked-list CSVs that
    `metrics_cli` persists, not the slide-CI JSONs, so each rung is re-scored
    through metrics_cli first (which also produces the hit@k / precision@k /
    nDCG@k that R4.b asks for, at every rung, for free).
    """
    dry = not go
    ranked = Q.rank(kind="cpu", wall_minutes=120, slots=8)
    qs = [r.name for r in ranked[:2]] or ["priority"]
    (OUT / "logs" / "r22_paired").mkdir(parents=True, exist_ok=True)
    jids = []
    for i, (pct, _) in enumerate(FRACTIONS):
        model = f"uni2apf{pct}"
        args = ["-n", "8", "-R", "span[hosts=1]",
                "-R", "select[mem>=4096] rusage[mem=4096]", "-M", "4096",
                "-W", "120"]
        cmd = (f"PROSPER_MODULE=prosper.evaluation.metrics_cli bash {WORKER} "
               f"--model {model} --method median_pooling --depth 10 "
               f"--iter 5000 --task r22_metrics")
        print(f"[r22-paired] {pct}% metrics + ranked lists:")
        j = _bsub2(args, f"prosper_r22_met_f{pct}", qs[i % len(qs)], cmd, dry,
                   "r22_paired")
        if j:
            jids.append(j)

    dep = " && ".join(f"done({j})" for j in jids)
    for k in (1, 3, 5, 10):
        args = ["-n", "8", "-R", "span[hosts=1]",
                "-R", "select[mem>=4096] rusage[mem=4096]", "-M", "4096",
                "-W", "120"]
        if dep:
            args += ["-w", dep]
        # 15% is the reference: the question is what ADDING tiles buys.
        cmd = (f"PROSPER_MODULE=prosper.evaluation.paired bash {WORKER} "
               f"--ranked-dir {OUT}/r22_metrics "
               f"--reference uni2apf15_median_pooling --k {k} --depth 10 "
               f"--iter 5000 --task r22_paired")
        print(f"[r22-paired] paired delta vs 15% at k={k}:")
        _bsub2(args, f"prosper_r22_paired_k{k}", qs[(k) % len(qs)], cmd, dry,
               "r22_paired")


def _bsub2(args, name, queue, cmd, dry, subdir):
    full = ["bsub", "-q", queue, "-J", name,
            "-oo", str(OUT / "logs" / subdir / f"{name}.%J.out"),
            "-eo", str(OUT / "logs" / subdir / f"{name}.%J.err")] + args + [cmd]
    if dry:
        print("  DRY:", name, "->", queue)
        return None
    r = subprocess.run(full, capture_output=True, text=True)
    print("  " + (r.stdout or r.stderr).strip())
    if r.returncode != 0:
        raise SystemExit(f"bsub failed for {name}")
    return r.stdout.split("<")[1].split(">")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["ladder", "paired"], default="ladder")
    ap.add_argument("--after", default="",
                    help="comma-separated job ids to wait on with `ended` "
                         "(NOT `done` -- see module docstring)")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    dry = not a.go
    (OUT / "logs" / "r22_ladder").mkdir(parents=True, exist_ok=True)

    if a.phase == "paired":
        paired(a.go)
        if not a.go:
            print("\n[dry run] nothing submitted. Re-run with --go.")
        return 0

    dep = ""
    ids = [x.strip() for x in a.after.split(",") if x.strip()]
    if ids:
        dep = " && ".join(f"ended({i})" for i in ids)
        print(f"[r22] dependency: {dep}")

    ranked = Q.rank(kind="cpu", wall_minutes=180, slots=8)
    agg_q = ranked[0].name if ranked else "heavy_io"
    ev_q = ranked[1].name if len(ranked) > 1 else agg_q
    print(f"[r22] aggregation -> {agg_q}, evaluation -> {ev_q}")

    for pct, frac in FRACTIONS:
        model = f"uni2apf{pct}"
        agg_args = ["-n", "8", "-R", "span[hosts=1]",
                    "-R", "select[mem>=4096] rusage[mem=4096]", "-M", "4096",
                    "-W", "180"]
        if dep:
            agg_args += ["-w", dep]
        agg_cmd = (f"PROSPER_MODULE=prosper.embeddings.aggregate_cli bash {WORKER} "
                   f"--encoder uni2 --features {FEATURES} "
                   f"--matrix-model {model} --method median_pooling "
                   f"--metric cityblock --tile-fraction {frac} "
                   f"--min-slides {EXPECT_SLIDES}")
        print(f"[r22] {pct}% aggregation:")
        jid = _bsub(agg_args, f"prosper_r22_agg_f{pct}", agg_q, agg_cmd, dry)

        ev_args = ["-n", "8", "-R", "span[hosts=1]",
                   "-R", "select[mem>=2048] rusage[mem=2048]", "-M", "2048",
                   "-W", "60"]
        # `done` is correct here: if the aggregation refused (incomplete
        # features) there is no matrix, and the evaluation MUST NOT run.
        if jid:
            ev_args += ["-w", f"done({jid})"]
        print(f"[r22] {pct}% evaluation:")
        for k in KS:
            ev_cmd = (f"bash {WORKER} --model {model} --method median_pooling "
                      f"--topk {k} --tie-break generous_majority --iter 5000 "
                      f"--n-jobs 8 --task r22_ladder")
            _bsub(ev_args, f"prosper_r22_eval_f{pct}_k{k}", ev_q, ev_cmd, dry)

    if dry:
        print("\n[dry run] nothing submitted. Re-run with --go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
