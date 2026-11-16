"""Slide-level evaluation entry point.

Ported from ``comet-analysis/src/calculate_slide_CI.py``. The evaluation logic in
``slide_eval.py`` is a byte-identical copy of the script that produced the
submitted manuscript's numbers; this module only changes *how it is configured*.

Two defects in the original are fixed here, and both are provenance defects
rather than numerical ones:

1. ``TIE_BREAK``, ``DROP_LABELS``, ``BOOTSTRAP_MODE``, ``SEED`` and the column
   names were module-level constants. Sweeping tie-break rules meant editing the
   source between runs, and nothing recorded which edit produced which JSON. The
   headline metric changed from ``any_hit`` to ``generous_majority`` in Nov 2025
   as exactly such an edit, and it was never committed -- the change had to be
   dated from file mtimes.
2. The output JSON recorded ``exclude_same_patient`` (an unused evaluator flag)
   but not the input matrix path -- while the actual same-patient masking lived
   in the matrix, signalled only by a filename token. That ambiguity is what made
   a full patient-exclusion audit necessary.

So the emitted config block now records the resolved input paths, the git commit,
and every knob. Defaults reproduce the published configuration exactly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from . import paths
from .slide_eval import (
    compute_metrics_from_predictions,
    filter_similarity_matrix,
    load_metadata,
    load_similarity_matrix,
    predict_labels_from_matrix,
    stratified_bootstrap_cis,
)

# Published configuration, read back from the archived CI JSONs.
#
# CRITICAL: there are TWO published configurations, and they differ in more than
# the input matrix. Conflating them was a real error caught by audit on
# 2026-08-27, before G2 ran:
#
#   all-tissue (Table S3)  n_slides=2778  dropped_labels=['MISC']
#                          -> 15 classes, NORMAL RETAINED
#   tumour-only            n_slides=2604  dropped_labels=['MISC','NORMAL']
#                          -> 14 classes, NORMAL dropped entirely
#
# 2778 = 2921 - 143 MISC. 2604 = 2778 - 174 NORMAL. Methods (revised ms, ROI /
# tumour-only section) state that all 174 NORMAL slides were excluded from the
# tumour-only evaluation, which is why the two populations differ by exactly 174.
#
# Beware a THIRD population: Supplementary Tables S10-S13 carry 2,766 rows. That
# is the row count of the tumour-only *distance matrix* (which retains 163 NORMAL
# slides), not of any CI evaluation. Do not use 2,766 as a CI assertion.
PUBLISHED = {
    "topk": 5,
    "bootstrap_B": 5000,
    "seed": 123,
    "larger_is_better": False,
    "min_class_count": None,
    "id_col": "Slide.Scan.File",
    "label_col": "Tumor.group",
    "patient_col": "UID.Subject",
    "tie_break": "generous_majority",
    "bootstrap_mode": "pps_cluster_one_slide",
}

# drop_labels is coupled to the population in every published run, so it is
# derived from --tumor_only rather than being a free-standing default.
PUBLISHED_DROP_LABELS = {
    False: ["MISC"],
    True: ["MISC", "NORMAL"],
}

PUBLISHED_N_SLIDES = {False: 2778, True: 2604}

# Known-answer targets for UNIv2 + Optimal Transport (ot_cosine), same-patient
# masked, generous_majority. Verified against Supplementary Table S3 'Overall
# Metrics' and against the archived JSONs on 2026-08-27.
#
# Table S3 is the ALL-TISSUE analysis: every one of its 132 rows carries
# per-class support summing to 2,778 with NORMAL=174. It contains no tumour-only
# rows and its caption states no population.
REGRESSION_MACRO_F1 = {
    # tumor_only -> {topk -> macro F1}
    False: {1: 0.769515, 3: 0.821829, 5: 0.791156, 10: 0.765022},   # == Table S3
    True:  {1: 0.791487, 3: 0.832427, 5: 0.801831, 10: 0.796423},   # not tabulated
}

# Back-compat alias; Table S3 is the all-tissue arm.
TABLE_S3_UNI2_OT_MACRO_F1 = REGRESSION_MACRO_F1[False]

# Known answers for the OTHER published tie-break rule, read back from the
# archived comet-uni2/confidence_interval/ JSONs (tie_break: any_hit, 160 files)
# on 2026-08-27. These are task C2's fidelity check: reproducing them proves the
# hit@k relabelling is a *relabelling* and not a recomputation.
#
# `any_hit` overwrites the prediction with the ground truth whenever the truth
# appears anywhere in the top-k, then computes F1 on that -- so it is hit@k
# laundered through an F1 wrapper, and monotone non-decreasing in k by
# construction. The 0.7695 -> 0.8684 -> 0.8981 -> 0.9411 progression is that
# monotonicity, not a model improving with k.
#
# At k=1 the two rules coincide exactly (a single neighbour cannot tie), which
# is why this table and REGRESSION_MACRO_F1[False] share 0.769515. That shared
# value is a known-answer test in its own right.
KNOWN_MACRO_F1 = {
    ("generous_majority", False): REGRESSION_MACRO_F1[False],
    ("generous_majority", True): REGRESSION_MACRO_F1[True],
    ("any_hit", False): {1: 0.769515, 3: 0.868444, 5: 0.898134, 10: 0.941109},
}


def _git_commit() -> str:
    # timeout=10: this repo's .git has been observed to hang under heavy
    # concurrent filesystem contention (GPFS, multiple agents active on
    # 2026-08-27). Provenance is best-effort -- a job must not wedge on an LSF
    # node waiting on a git call that will not return.
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode().strip()
    except Exception:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Slide-level retrieval evaluation with patient-clustered bootstrap CIs."
    )
    p.add_argument("--model", required=True, help="encoder name, e.g. uni2")
    p.add_argument("--method", required=True,
                   help="aggregation, e.g. ot_cosine|median_pooling|yottixel|sampler")
    p.add_argument("--topk", type=int, default=PUBLISHED["topk"])
    p.add_argument("--iter", type=int, default=PUBLISHED["bootstrap_B"],
                   dest="bootstrap_B",
                   help="bootstrap replicates (published: 5000; the original "
                        "default was 100, which no published run used)")
    p.add_argument("--seed", type=int, default=PUBLISHED["seed"])
    p.add_argument("--tumor_only", action="store_true")
    p.add_argument("--no-same-patient", dest="no_same_patient",
                   action="store_true", default=True,
                   help="use the same-patient-masked input matrix (default, as published)")
    p.add_argument("--allow-same-patient", dest="no_same_patient",
                   action="store_false",
                   help="use the UNMASKED matrix; for leakage-quantification runs only")
    p.add_argument("--tie-break", default=PUBLISHED["tie_break"],
                   choices=["generous_majority", "any_hit", "absolute", "closest"],
                   help="generous_majority = plurality, correct if truth is in the "
                        "tied argmax set (published headline). any_hit = hit@k "
                        "laundered through an F1 wrapper. absolute = true >k/2 "
                        "majority. closest = plurality + nearest-neighbour tie-break")
    p.add_argument("--drop-labels", nargs="*", default=None,
                   help="default is the published value for the chosen population: "
                        "['MISC'] all-tissue, ['MISC','NORMAL'] tumour-only")
    p.add_argument("--min-class-count", type=int, default=None)
    p.add_argument("--bootstrap-mode", default=PUBLISHED["bootstrap_mode"],
                   choices=["pps_cluster_one_slide", "one_per_patient"])
    p.add_argument("--n-jobs", type=int, default=32,
                   help="bootstrap replicates run across a ProcessPoolExecutor; "
                        "match this to the LSF slot count")
    p.add_argument("--task", default="slide_ci",
                   help="subdirectory under $PROSPER_OUT for the result")
    p.add_argument("--out", default=None, help="explicit output JSON path")
    p.add_argument("--assert-table-s3", action="store_true",
                   help="G2 gate: require macro F1 to match the published Table S3 "
                        "value for uni2+ot_cosine to 6 dp, and exit non-zero if not")
    return p


def run(args: argparse.Namespace) -> dict:
    if args.drop_labels is None:
        args.drop_labels = list(PUBLISHED_DROP_LABELS[bool(args.tumor_only)])

    sim_path = paths.require(
        paths.similarity_matrix(args.model, args.method,
                                tumor_only=args.tumor_only,
                                no_same_patient=args.no_same_patient),
        "similarity matrix")
    meta_path = paths.require(paths.slide_metadata(), "slide metadata")

    sim = load_similarity_matrix(str(sim_path))
    meta = load_metadata(
        str(meta_path),
        label_col=PUBLISHED["label_col"],
        id_col=PUBLISHED["id_col"],
        patient_col=PUBLISHED["patient_col"],
        drop_labels=list(args.drop_labels),
        min_class_count=args.min_class_count,
    )
    sim_f, meta_f = filter_similarity_matrix(sim, meta, id_col=PUBLISHED["id_col"])

    df_pred = predict_labels_from_matrix(
        sim=sim_f, meta=meta_f, topk=args.topk,
        id_col=PUBLISHED["id_col"], label_col=PUBLISHED["label_col"],
        patient_col=PUBLISHED["patient_col"],
        larger_is_better=PUBLISHED["larger_is_better"],
        tie_break=args.tie_break,
        # The evaluator's own exclusion branch is dead code in the published
        # pipeline: masking happens upstream in the matrix. Keep it off so the
        # port stays faithful, and control exclusion by matrix choice instead.
        exclude_same_patient=False,
    )

    labels_order = sorted(meta_f[PUBLISHED["label_col"]].unique().tolist())
    metrics = compute_metrics_from_predictions(df_pred, labels_order=labels_order)
    cis = stratified_bootstrap_cis(
        df_pred=df_pred, meta=meta_f, B=args.bootstrap_B, seed=args.seed,
        id_col=PUBLISHED["id_col"], label_col=PUBLISHED["label_col"],
        patient_col=PUBLISHED["patient_col"], labels_order=labels_order,
        mode=args.bootstrap_mode, n_jobs=args.n_jobs,
    )

    results = {
        "config": {
            "model": args.model,
            "method": args.method,
            "top_k": args.topk,
            "larger_is_better": PUBLISHED["larger_is_better"],
            "bootstrap_B": args.bootstrap_B,
            "seed": args.seed,
            "n_jobs": args.n_jobs,
            "dropped_labels": list(args.drop_labels),
            "min_class_count": args.min_class_count,
            "tie_break": args.tie_break,
            "bootstrap_mode": args.bootstrap_mode,
            "tumor_only": args.tumor_only,
            # The two fields the published JSONs lacked, and the reason the
            # patient-exclusion question could not be settled from them:
            "input_similarity_matrix": str(sim_path),
            "input_metadata": str(meta_path),
            "same_patient_masked_input": args.no_same_patient,
            "prosper_commit": _git_commit(),
        },
        "n_slides": metrics["n_slides"],
        "labels_order": metrics["labels_order"],
        "confusion_matrix": metrics["confusion_matrix"],
        "point_estimates": {k: metrics[k] for k in
                            ("micro_f1", "macro_f1", "macro_precision", "macro_recall")
                            if k in metrics},
        "per_class_f1": metrics.get("per_class_f1"),
        "confidence_intervals": cis,
    }

    out = Path(args.out) if args.out else paths.result_json(
        args.task,
        f"slide_ci_{args.model}_{args.method}_top{args.topk}_{args.tie_break}"
        f"{'_tumor_only' if args.tumor_only else ''}"
        f"{'' if args.no_same_patient else '_SAME_PATIENT_ALLOWED'}")
    out.write_text(json.dumps(results, indent=2))
    print(f"[prosper.evaluation] wrote {out}")
    print(f"[prosper.evaluation] macro_f1 = {metrics['macro_f1']:.6f}  "
          f"micro_f1 = {metrics['micro_f1']:.6f}  n_slides = {metrics['n_slides']}")
    return results


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    results = run(args)

    if args.assert_table_s3:
        tumor_only = bool(args.tumor_only)
        key = (args.tie_break, tumor_only)
        expected = KNOWN_MACRO_F1.get(key, {}).get(args.topk)
        expected_n = PUBLISHED_N_SLIDES[tumor_only]
        arm = ("tumour-only" if tumor_only else "all-tissue (Table S3)")
        arm = f"{arm} / {args.tie_break}"
        if not (args.model == "uni2" and args.method == "ot_cosine"
                and expected is not None):
            print("[G2] no known answer for "
                  f"tie_break={args.tie_break} tumor_only={tumor_only} "
                  f"topk={args.topk} on {args.model}/{args.method}. Known "
                  f"combinations: {sorted(KNOWN_MACRO_F1)} with k in {{1,3,5,10}}, "
                  "uni2/ot_cosine only.")
            return 2

        got = results["point_estimates"]["macro_f1"]
        got_n = results["n_slides"]
        ok_f1 = abs(got - expected) < 5e-7
        ok_n = got_n == expected_n

        # n_slides is checked as well as macro F1 because the two published
        # configurations differ in population AND in dropped_labels. A run that
        # matched macro F1 on the wrong population would be a false pass.
        print(f"[G2] arm            : {arm}")
        print(f"[G2] macro_f1       : {got:.6f} vs {expected:.6f}  "
              f"{'OK' if ok_f1 else f'MISMATCH (delta {got - expected:+.2e})'}")
        print(f"[G2] n_slides       : {got_n} vs {expected_n}  "
              f"{'OK' if ok_n else 'MISMATCH'}")
        print(f"[G2] dropped_labels : {results['config']['dropped_labels']}")
        if ok_f1 and ok_n:
            print("[G2] PASS")
            return 0
        if ok_f1 and not ok_n:
            print("[G2] FAIL on population. macro F1 matched but n_slides did not, "
                  "which means the label set differs. Check --drop-labels.")
        elif ok_n and not ok_f1:
            print("[G2] FAIL on metric. Population is right, so the divergence is "
                  "in the evaluator or the input matrix, not the cohort.")
        print("[G2] STOP. Do not run downstream tasks: the pipeline no longer "
              "reproduces the manuscript, so every derived number is suspect.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
