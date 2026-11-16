#!/usr/bin/env python3
"""Submit the execution spec's gates to LSF, with queue discipline enforced.

Every submission goes through ``prosper.hpc.submit``, so the rules in
``[internal record]`` are mechanical rather than
aspirational: queues are ranked live at submit time, a sweep cannot be fanned
out before a canary has returned an epilogue, and ``-oo/-eo`` preserve that
epilogue.

  submit_gate.py g1                      # harness smoke, both populations
  submit_gate.py g2-canary               # Table S3 k=5, all-tissue, --assert
  submit_gate.py g2-fanout --epilogue <canary .out>   # the remaining 7 runs

Nothing submits without ``--go``; the default is a dry run that prints the
exact bsub argv.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from prosper.hpc import submit as S          # noqa: E402
from prosper.hpc import queues as Q          # noqa: E402

WORKER = REPO / "scripts" / "lsf" / "prosper_job.sh"
OUT = Path(os.environ.get(
    "PROSPER_OUT",
    "/PROSPER_WORKSPACE/prosper_revision/prosper_revision_output"))


def _cmd(args: str) -> str:
    return f"bash {WORKER} {args}"


def g1(go: bool) -> None:
    """Harness portability smoke. Reduced --iter, both populations.

    The spec's G1 command runs the tumour-only path only. Both are run here
    because G2 depends on the *all-tissue* matrix, and a smoke test that does
    not touch the artifact the next gate needs is not much of a smoke test.
    """
    log = OUT / "logs" / "g1_smoke"
    res = S.Resources(slots=4, mem_mb_per_slot=4096, wall_minutes=20)
    for tag, extra in (("alltissue", ""), ("tumoronly", "--tumor_only")):
        S.one(_cmd(f"--model uni2 --method ot_cosine --topk 5 {extra} "
                   f"--iter 50 --n-jobs 4 --task g1_smoke"),
              job_name=f"prosper_g1_{tag}", res=res, log_dir=log,
              dry_run=not go)


def g2_canary(go: bool) -> None:
    """The trust anchor, one k. Read its epilogue before fanning out."""
    log = OUT / "logs" / "g2_table_s3"
    # Memory is sized from G1's epilogue (peak 422 MB against 16 GB reserved at
    # -n 4), scaled for 32 ProcessPoolExecutor workers each holding a copy of
    # the prediction frame: ~5 GB expected, 64 GB reserved. That is still 12x
    # headroom but half the naive -M 4096 x 32 = 128 GB, and the canary's own
    # epilogue is what the fan-out gets sized from.
    #
    # Slots stay at 32 despite G1 measuring effective_cores = 0.22: at
    # --iter 50 there is essentially no parallel phase to measure. This is the
    # exact case right_size() documents as advisory-only.
    res = S.Resources(slots=32, mem_mb_per_slot=2048, wall_minutes=120)
    S.canary(_cmd("--model uni2 --method ot_cosine --topk 5 "
                  "--iter 5000 --n-jobs 32 --task g2_table_s3 "
                  "--assert-table-s3"),
             job_name="prosper_g2", res=res, log_dir=log, dry_run=not go)


def g2_fanout(epilogue_path: Path, go: bool) -> None:
    """The remaining 7 runs: k in {1,3,10} all-tissue + all four tumour-only.

    Both arms assert their known answers, so this is 7 independent
    known-answer tests, not 7 exploratory runs.
    """
    log = OUT / "logs" / "g2_table_s3"
    epi = S.parse_epilogue(epilogue_path.read_text())
    if not epi:
        raise SystemExit(f"no LSF epilogue found in {epilogue_path}. The canary "
                         "must have completed and -oo must have preserved it.")
    print(f"[g2-fanout] canary epilogue: {epi}")

    params = OUT / "g2_table_s3" / "params.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for k in (1, 3, 10):
        lines.append(f"--model uni2 --method ot_cosine --topk {k} --iter 5000 "
                     f"--n-jobs 32 --task g2_table_s3 --assert-table-s3")
    for k in (1, 3, 5, 10):
        lines.append(f"--model uni2 --method ot_cosine --topk {k} --tumor_only "
                     f"--iter 5000 --n-jobs 32 --task g2_table_s3 "
                     f"--assert-table-s3")
    params.write_text("\n".join(lines) + "\n")
    print(f"[g2-fanout] wrote {len(lines)} params to {params}")

    # Wall is set to 25 min, not 120, so that `short` (30 min hard limit) is
    # eligible -- it is the emptiest CPU queue here and the canary proved the
    # job runs in 94 s. right_size() will cut slots from the canary's measured
    # effective_cores, which also drops large_core_count (>=32 slots) from the
    # ranking, leaving short + priority. Both were near-zero pressure.
    res = S.Resources(slots=32, mem_mb_per_slot=2048, wall_minutes=25)
    S.fan_out(f"PROSPER_PARAMS={params} bash {WORKER}",
              n=len(lines), job_name="prosper_g2_arr", res=res, log_dir=log,
              n_queues=2, canary_epilogue=epi, dry_run=not go)


def g3(go: bool) -> None:
    """ROI harness fidelity. One job, so it is its own canary.

    Reproduces the published ROI macro F1 = 0.7903 using the *legacy,
    patient-free* metadata map -- which is the only way to reproduce it, and
    itself the evidence that the ROI branch never had patient information. Must
    pass before C1b changes anything, or a movement in Fig 6A is not
    attributable to patient exclusion.
    """
    log = OUT / "logs" / "g3_roi_fidelity"
    res = S.Resources(slots=32, mem_mb_per_slot=2048, wall_minutes=240)
    S.one(f"PROSPER_MODULE=prosper.evaluation.roi_cli bash {WORKER} "
          f"--n-workers 32 --task g3_roi_fidelity --assert-published-roi",
          job_name="prosper_g3_roi", res=res, log_dir=log, dry_run=not go)


def c1b_regression(go: bool) -> None:
    """C1b's own fidelity check (i): re-run G3's exact command through the
    *modified* ``roi_cli.py``/``roi_eval.py`` (patient-exclusion code added)
    with ``--exclude-same-patient`` absent, and confirm macro F1 still equals
    0.7903 to 4 dp. Same shape as G3 (see ``g3()`` above) -- it is the same
    code path by construction -- so it is also this sweep's canary.
    """
    log = OUT / "logs" / "c1b_roi_patient_excluded"
    res = S.Resources(slots=32, mem_mb_per_slot=2048, wall_minutes=240)
    # Explicit queue, not S.canary's live-ranked pick: measured live, `priority`
    # and `large_core_count` both showed pressure 0.00, so pressure-then-PRIO
    # ranking put `priority` first -- but `priority` draws from the shared
    # rhel8_cpu host group and this -n 32 span[hosts=1] request PEND'd there on
    # "Affinity resource requirement cannot be met ... not enough processor
    # units" even at zero queue-depth pressure. `large_core_count` is the
    # dedicated wide-host queue G3 itself ran on (see g3()'s epilogue) and is
    # the right fit for a single 32-slot request regardless of what the
    # pending-count heuristic says.
    S.one(f"PROSPER_MODULE=prosper.evaluation.roi_cli bash {WORKER} "
          f"--n-workers 32 --task c1b_roi_patient_excluded "
          f"--assert-published-roi",
          job_name="prosper_c1b_regression_canary", res=res, log_dir=log,
          queue="large_core_count", dry_run=not go)


def c1b_run(epilogue_path: Path, go: bool) -> None:
    """C1b's two compute jobs: primary (top_k_per_roi=5) and sensitivity
    (top_k_per_roi=10) patient-excluded runs, both with the patient-clustered
    PPS bootstrap (B=5000, seed=123 -- ``roi_cli``'s defaults once
    ``--exclude-same-patient`` is set).

    Sized from the regression canary's epilogue with extra headroom: same
    per-ROI scan as G3/the regression job, plus a per-hit patient-membership
    check, plus 5x the bootstrap replicates (5000 vs the as-run 1000) -- so
    memory is doubled and wall is tripled off the measured peak rather than
    reused verbatim.
    """
    log = OUT / "logs" / "c1b_roi_patient_excluded"
    epi = S.parse_epilogue(epilogue_path.read_text())
    if not epi:
        raise SystemExit(f"no LSF epilogue in {epilogue_path}. Run "
                         "c1b-regression first and pass its .out file.")
    print(f"[c1b] regression-canary epilogue: {epi}")
    res = S.right_size(
        S.Resources(slots=32, mem_mb_per_slot=2048, wall_minutes=240),
        epi, headroom=4.0)
    res = S.Resources(slots=max(32, res.slots), mem_mb_per_slot=res.mem_mb_per_slot,
                      wall_minutes=180)
    print(f"[c1b] right-sized to -n {res.slots} -M {res.mem_mb_per_slot} "
          f"-W {res.wall_minutes} ({res.total_reserved_gb:.1f} GB total)")

    # Explicit queues, not the live-ranked pick -- see the comment in
    # c1b_regression(): this job's `-n 32 span[hosts=1]` shape PEND'd on
    # "Affinity resource requirement cannot be met" on *both* `priority`
    # (tried twice) and `heavy_io` (tried once) despite each showing PEND=0 at
    # submit time, all on 2026-08-27. `large_core_count`, the dedicated
    # wide-host queue G3 itself ran on, started every one of these jobs within
    # seconds. Both C1b runs landed there in the end -- a departure from the
    # spec's "priority/large_core_count" pairing, recorded and reconciled in
    # `[internal record]` -- because a queue that
    # cannot actually place a wide job is not a second queue in any sense that
    # matters, and `heavy_io` and `priority` were both genuinely tried first.
    S.one(f"PROSPER_MODULE=prosper.evaluation.roi_cli bash {WORKER} "
          f"--top-k-per-roi 5 --exclude-same-patient --n-workers {res.slots} "
          f"--task c1b_roi_patient_excluded",
          job_name="prosper_c1b_primary_k5", res=res, log_dir=log,
          queue="large_core_count", dry_run=not go)
    S.one(f"PROSPER_MODULE=prosper.evaluation.roi_cli bash {WORKER} "
          f"--top-k-per-roi 10 --exclude-same-patient --n-workers {res.slots} "
          f"--task c1b_roi_patient_excluded",
          job_name="prosper_c1b_sensitivity_k10", res=res, log_dir=log,
          queue="large_core_count", dry_run=not go)


def c2_fidelity(epilogue_path: Path, go: bool) -> None:
    """C2 fidelity check: reproduce the archived any_hit (hit@k) grid.

    Four known-answer tests, not four results. Proves the hit@k relabelling is a
    relabelling. Sized from the G2 canary because the job shape is identical --
    same matrix, same population, same bootstrap, only the tie-break differs.
    """
    log = OUT / "logs" / "c2_fidelity"
    epi = S.parse_epilogue(epilogue_path.read_text())
    if not epi:
        raise SystemExit(f"no LSF epilogue in {epilogue_path}")
    params = OUT / "c2_fidelity" / "params.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    params.write_text("\n".join(
        f"--model uni2 --method ot_cosine --topk {k} --tie-break any_hit "
        f"--iter 5000 --n-jobs 32 --task c2_fidelity --assert-table-s3"
        for k in (1, 3, 5, 10)) + "\n")
    res = S.Resources(slots=32, mem_mb_per_slot=2048, wall_minutes=25)
    S.fan_out(f"PROSPER_PARAMS={params} bash {WORKER}",
              n=4, job_name="prosper_c2_fid", res=res, log_dir=log,
              n_queues=2, canary_epilogue=epi, dry_run=not go)


def c2_canary(go: bool) -> None:
    """C2 metric-suite canary: uni2/ot_cosine with the hit@k known-answer test.

    Deliberately the one combination that has archived answers, so the sweep is
    only launched once the layer has reproduced a published retrieval.
    """
    log = OUT / "logs" / "c2_metric_suite"
    res = S.Resources(slots=16, mem_mb_per_slot=4096, wall_minutes=120)
    S.canary(f"PROSPER_MODULE=prosper.evaluation.metrics_cli bash {WORKER} "
             f"--model uni2 --method ot_cosine --depth 10 --iter 5000 "
             f"--task c2_metric_suite --assert-hit-at-k",
             job_name="prosper_c2_metrics", res=res, log_dir=log, dry_run=not go)


def c2_sweep(epilogue_path: Path, go: bool) -> None:
    """The full C2 metric grid: 33 published configurations.

    8 benchmark encoders x 4 aggregations + CONCH/TITAN = 33, which is exactly
    the count behind Table S3's 132 rows (33 configs x 4 k values). Sized from
    the canary, which measured 552 MB peak against 65,536 MB reserved and
    effective_cores 0.83 -- the bootstrap here is single-threaded numpy
    indexing over precomputed per-query vectors, so a slot cut to 1 is correct
    rather than the artefact it would be for the F1 path's ProcessPoolExecutor.
    """
    log = OUT / "logs" / "c2_metric_suite"
    epi = S.parse_epilogue(epilogue_path.read_text())
    if not epi:
        raise SystemExit(f"no LSF epilogue in {epilogue_path}")

    encoders = ["uni2", "virchow2", "gigapath", "exaonepath",
                "phikon-v2", "ctranspath", "hibou", "simclr"]
    aggs = ["ot_cosine", "median_pooling", "sampler", "yottixel"]
    combos = [(e, a) for e in encoders for a in aggs] + [("titan", "titan")]

    params = OUT / "c2_metric_suite" / "params.txt"
    params.parent.mkdir(parents=True, exist_ok=True)
    params.write_text("\n".join(
        f"--model {e} --method {a} --depth 10 --iter 5000 --task c2_metric_suite"
        for e, a in combos) + "\n")
    print(f"[c2-sweep] {len(combos)} configurations -> {params}")

    res = S.Resources(slots=16, mem_mb_per_slot=4096, wall_minutes=25)
    S.fan_out(f"PROSPER_MODULE=prosper.evaluation.metrics_cli "
              f"PROSPER_PARAMS={params} bash {WORKER}",
              n=len(combos), job_name="prosper_c2_sweep", res=res, log_dir=log,
              n_queues=3, canary_epilogue=epi, dry_run=not go)


def c5_paired(go: bool) -> None:
    """C5 paired-bootstrap comparisons. One job: all 32 Δ share the replicates."""
    log = OUT / "logs" / "c5_paired"
    res = S.Resources(slots=4, mem_mb_per_slot=4096, wall_minutes=240)
    for k in (5, 10):
        S.one(f"PROSPER_MODULE=prosper.evaluation.paired bash {WORKER} "
              f"--k {k} --iter 5000 --reference uni2_ot_cosine --task c5_paired",
              job_name=f"prosper_c5_paired_k{k}", res=res, log_dir=log,
              queue="priority", dry_run=not go)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g1", "g2-canary", "g2-fanout", "g3",
                                     "c1b-regression", "c1b-run",
                                     "c2-fidelity", "c2-canary", "c2-sweep", "c5-paired",
                                     "queues"])
    ap.add_argument("--epilogue", type=Path)
    ap.add_argument("--go", action="store_true",
                    help="actually submit (default is a dry run)")
    a = ap.parse_args()

    if a.gate == "queues":
        print(Q.report(kind="cpu", wall_minutes=120, slots=32))
        print()
        print(Q.report(kind="gpu", wall_minutes=120, slots=4))
        return 0
    if a.gate == "g1":
        g1(a.go)
    elif a.gate == "g2-canary":
        g2_canary(a.go)
    elif a.gate == "g3":
        g3(a.go)
    elif a.gate == "c1b-regression":
        c1b_regression(a.go)
    elif a.gate == "c1b-run":
        if not a.epilogue:
            raise SystemExit("c1b-run needs --epilogue <c1b-regression's .out file>")
        c1b_run(a.epilogue, a.go)
    elif a.gate == "c5-paired":
        c5_paired(a.go)
    elif a.gate == "c2-canary":
        c2_canary(a.go)
    elif a.gate == "c2-sweep":
        if not a.epilogue:
            raise SystemExit("c2-sweep needs --epilogue <c2 canary .out>")
        c2_sweep(a.epilogue, a.go)
    elif a.gate == "c2-fidelity":
        if not a.epilogue:
            raise SystemExit("c2-fidelity needs --epilogue <a completed .out file>")
        c2_fidelity(a.epilogue, a.go)
    else:
        if not a.epilogue:
            raise SystemExit("g2-fanout needs --epilogue <canary .out file>")
        g2_fanout(a.epilogue, a.go)
    if not a.go:
        print("\n[dry run] nothing submitted. Re-run with --go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
