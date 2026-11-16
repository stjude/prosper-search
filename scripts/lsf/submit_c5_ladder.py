#!/usr/bin/env python3
"""Submit the C5 module ablation ladder to LSF ([internal record], task C5).

Separate from ``scripts/lsf/submit_gate.py`` by instruction (another task may
be editing that file concurrently) -- this is its own driver, following the
same pattern: everything goes through ``prosper.hpc.submit``, nothing submits
without ``--go``, queues are ranked live, and a fan-out never runs before a
canary has returned an epilogue.

Three stages:

  submit_c5_ladder.py gen-tumor-only-canary        # the one missing input:
                                                    # uni2/ot_cosine tumour-only
                                                    # ranked list (rungs 2->3 and
                                                    # 4->5 both need it; nothing
                                                    # else does, per the artifact
                                                    # audit in module_ladder.py)
  submit_c5_ladder.py ladder --epilogue <canary .out>   # the three rung jobs
  submit_c5_ladder.py queues                        # live queue pressure

Why only one canary generates a ranked list rather than all three rungs having
their own: rung 3->4 needs only all-tissue ranked lists, which the C2 sweep
already produced (``ranked_lists_uni2_{median_pooling,ot_cosine}.csv.gz``);
rung 2->3 and rung 4->5 both need the tumour-only ot_cosine ranked list, which
does not exist yet. Generating it is the only new upstream compute this ladder
requires -- everything else re-analyzes existing artifacts.
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


def gen_tumor_only_canary(go: bool) -> None:
    """The one new upstream artifact: uni2/ot_cosine tumour-only ranked list.

    Same job shape as the already-run ``c2-canary``/``c2-sweep`` (metrics_cli on
    a cached 2921x2921 matrix, B=5000 patient-clustered bootstrap, no GPU) --
    sized identically (16 slots, 4 GB/slot, 120 min), because it is the same
    code path with ``--tumor_only`` added.
    """
    log = OUT / "logs" / "c5_ladder"
    res = S.Resources(slots=16, mem_mb_per_slot=4096, wall_minutes=120)
    S.canary(f"PROSPER_MODULE=prosper.evaluation.metrics_cli bash {WORKER} "
             f"--model uni2 --method ot_cosine --tumor_only --depth 10 "
             f"--iter 5000 --task c2_metric_suite",
             job_name="prosper_c5_ladder_genranked", res=res, log_dir=log,
             dry_run=not go)


def ladder(epilogue_path: Path, go: bool) -> None:
    """The three rung-transition jobs, right-sized from the canary's epilogue.

    Each is its own single job (not a job array -- there are only three, and
    they read different inputs), submitted with an explicit queue so the sweep
    is still spread across more than one queue as required, rather than a
    single ``recommend()`` pick reused three times.
    """
    log = OUT / "logs" / "c5_ladder"
    epi = S.parse_epilogue(epilogue_path.read_text())
    if not epi:
        raise SystemExit(f"no LSF epilogue found in {epilogue_path}. The ranked-list "
                         "canary must have completed and -oo must have preserved it.")
    print(f"[c5-ladder] canary epilogue: {epi}")
    res = S.right_size(
        S.Resources(slots=16, mem_mb_per_slot=4096, wall_minutes=120), epi, headroom=4.0)
    print(f"[c5-ladder] right-sized to -n {res.slots} -M {res.mem_mb_per_slot} "
          f"-W {res.wall_minutes} ({res.total_reserved_gb:.1f} GB total)")

    jobs = [
        ("prosper_c5_ladder_2to3",
         f"PROSPER_MODULE=prosper.evaluation.module_ladder bash {WORKER} "
         f"--rung 2to3 --model uni2 --method ot_cosine --k 5 --iter 5000 "
         f"--task c5_ladder", "priority"),
        ("prosper_c5_ladder_3to4",
         f"PROSPER_MODULE=prosper.evaluation.module_ladder bash {WORKER} "
         f"--rung 3to4 --model uni2 --k 5 --iter 5000 --task c5_ladder",
         "heavy_io"),
        ("prosper_c5_ladder_4to5",
         f"PROSPER_MODULE=prosper.evaluation.module_ladder bash {WORKER} "
         f"--rung 4to5 --model uni2 --method ot_cosine --n-neighbors 5 "
         f"--temperature 20.0 --iter 5000 --task c5_ladder "
         f"--assert-bayesian-fidelity "
         f"--s11-path {S11_DEFAULT} --s13-path {S13_DEFAULT}",
         "priority"),
    ]
    for name, cmd, queue in jobs:
        S.one(cmd, job_name=name, res=res, log_dir=log, queue=queue, dry_run=not go)


# Supplementary tables live in the manuscript workspace, not this repo -- a
# read-only cross-mount input for the (best-effort, non-fatal) Bayesian
# fidelity check only. Both mount aliases resolve to the same filesystem
# (docs/[internal record]); the /PROSPER_STORAGE form is used here since it is the one the
# manuscript repo documents as canonical.
S11_DEFAULT = ("/PROSPER_WORKSPACE/"
              "prosper_revision/docs/prosper_submission/supplementary/"
              "supplementary_tables/Table_S11_Interp_combined_similarity_matrix_"
              "uni2_ot_cosine_tumor_only_no_same_patient_probs_N5.xlsx")
S13_DEFAULT = ("/PROSPER_WORKSPACE/"
              "prosper_revision/docs/prosper_submission/supplementary/"
              "supplementary_tables/Table_S13_Interp_combined_similarity_matrix_"
              "uni2_ot_cosine_tumor_only_no_same_patient_probs_N18.xlsx")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["gen-tumor-only-canary", "ladder", "queues"])
    ap.add_argument("--epilogue", type=Path)
    ap.add_argument("--go", action="store_true",
                    help="actually submit (default is a dry run)")
    a = ap.parse_args()

    if a.gate == "queues":
        print(Q.report(kind="cpu", wall_minutes=120, slots=16))
        return 0
    if a.gate == "gen-tumor-only-canary":
        gen_tumor_only_canary(a.go)
    else:
        if not a.epilogue:
            raise SystemExit("ladder needs --epilogue <gen-tumor-only-canary's .out file>")
        ladder(a.epilogue, a.go)
    if not a.go:
        print("\n[dry run] nothing submitted. Re-run with --go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
