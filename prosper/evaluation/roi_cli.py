"""ROI-level evaluation entry point -- the G3 fidelity gate.

Ported from ``comet-new-scans/src/calculate_CI/calculate_agg_ROI_CI.py``, which
had no argparse: every knob was a module-level constant, so sweeping anything
meant editing the source and nothing recorded which edit produced which JSON.

G3 reproduces the *as-run* ROI configuration, and the distinction between
"as-run" and "published" matters enough to state here rather than in a commit
message. The Methods say the ROI bootstrap used 5,000 iterations resampling
**ROIs**; the code that produced the number used **1,000** iterations resampling
**slides**, i.i.d. within class. Reproducing the as-run config is still the
correct port test -- it generated the manuscript's macro F1 = 0.79 -- but the
divergence from the Methods is a claims-audit item (C9), not a bug to fix here.

Reproducing it also requires loading the *legacy, patient-free* metadata map.
That is not an accident of the port: the ROI branch never loaded a patient
column, which is why it could not have excluded same-patient slides even in
principle. It excluded only the query's own slide, by string comparison. That is
the one confirmed patient-leakage exposure in the paper, and correcting it is
C1b -- which is gated on this file passing first, so that any change in Fig 6A
is attributable to patient exclusion rather than to the reimplementation.

C1b (``--exclude-same-patient``)
---------------------------------
Adds the patient-excluded mode. It changes exactly three things relative to the
G3 path above, and nothing else:

1. Loads ``paths.slide_metadata()`` (carries ``UID.Subject``) instead of the
   legacy map, and filters same-patient hits out of every query's vote pool
   via ``roi_eval.predict_all_slides_agg_patient_excluded`` -- which reuses the
   published voting (``_dedup_preserve_order``) and tie-break
   (``_tie_break_by_counts_lex``) unmodified.
2. Bootstraps with ``slide_eval.stratified_bootstrap_cis`` (the audited
   patient-clustered PPS implementation) instead of ``roi_eval``'s i.i.d.
   within-class bootstrap, at B=5000/seed=123 by default -- the published
   Table S3 design, not the as-run ROI design.
3. Defaults to those B/seed values instead of the as-run 1000/17, unless
   overridden.

With ``--exclude-same-patient`` absent, this module's behaviour is completely
unchanged from the G3 gate above -- same functions, same defaults, same
metadata file. That is deliberate: it is what makes "run without
--exclude-same-patient and confirm G3 still passes" a real regression check on
this file rather than a check of a different code path.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from itertools import combinations
from pathlib import Path

from . import paths
from . import slide_eval
from .roi_eval import (
    compute_metrics_from_predictions,
    load_metadata,
    predict_all_slides_agg,
    predict_all_slides_agg_patient_excluded,
    stratified_bootstrap_cis,
)

# The as-run ROI config, read back from the published
# comet-new-scans/output/confidence_interval/agg_roi_slide_eval_results.json.
# Note what this config block does NOT contain: no exclude_same_patient key and
# no bootstrap_mode key. Their absence is the finding.
AS_RUN = {
    "top_k_per_roi": 5,
    "vote_mode": "raw_hits",
    "bootstrap_B": 1000,
    "seed": 17,
    "drop_labels": ["MISC"],
    "min_class_count": None,
    "id_col": "Slide.Scan.File",
    "label_col": "Tumor.group",
}

# C1b's patient-clustered PPS bootstrap: the published Table S3 design
# ([decision record]), not the as-run ROI design AS_RUN carries above.
PATIENT_EXCLUDED_DEFAULTS = {
    "bootstrap_B": 5000,
    "seed": 123,
    "bootstrap_mode": "pps_cluster_one_slide",
    "patient_col": "UID.Subject",
}

# Known answers for the fidelity check. 0.7903 is the value in the published
# JSON; the manuscript states only "macro F1 = 0.79". micro F1 is what Fig 6A
# plots as its "Overall" bar -- which is why the figure reads ~0.87 while the
# Results text says 0.79. Two different metrics, not an error, but undisclosed.
PUBLISHED_ROI = {
    "macro_f1": 0.7903,
    "micro_f1": 0.875450,
    "sum_total_hits_counted": 18_664_831,
}

# [decision record]'s within-patient pair count -- computed over the full 2,921-slide
# metadata population, i.e. *before* the ROI evaluator's own MISC drop.
ADR_0002_WITHIN_PATIENT_PAIRS_FULL_POPULATION = 1_949


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def count_glob_fallbacks(slide_ids, roi_hits_dir: str) -> dict:
    """How many slides would resolve via the substring-glob fallback.

    ``find_roi_file_for_slide`` tries the exact filename first and then falls
    back to ``*{slide_id}*.pkl``, returning ``matches[0]`` with no warning -- so
    a miss on slide ``1002`` would silently evaluate slide ``10021``. Whether it
    ever fired is unrecorded.

    This measures it without changing the evaluator's behaviour, which is the
    right order: a fix applied before measuring would destroy the evidence that
    the published run was or was not affected. If the count is zero, the latent
    bug never fired and the published numbers are unaffected by it.
    """
    exact_missing, ambiguous = [], []
    for sid in slide_ids:
        exact = os.path.join(roi_hits_dir, f"top_10_hit_slides_slide_{sid}.pkl")
        if os.path.exists(exact):
            continue
        matches = sorted(glob.glob(os.path.join(roi_hits_dir, f"*{sid}*.pkl")))
        if matches:
            ambiguous.append({"slide": sid, "n_matches": len(matches),
                              "would_use": os.path.basename(matches[0])})
        else:
            exact_missing.append(sid)
    return {"n_would_use_glob_fallback": len(ambiguous),
            "n_no_file_at_all": len(exact_missing),
            "fallback_examples": ambiguous[:20]}


def masked_pair_reconciliation(meta_full, meta_eval, id_col: str,
                               patient_col: str) -> dict:
    """C1b fidelity check (iii): count within-patient pairs directly, both over
    the full 2,921-slide metadata population (must reconcile against ADR
    0002's 1,949) and over the ROI-evaluated population (after ``drop_labels``,
    i.e. MISC dropped -- the population this run actually scores).

    A direct pairwise count (``itertools.combinations``), not a formula, so
    this is not just re-deriving the same arithmetic [decision record] used -- it is an
    independent recount against this run's own metadata load.
    """
    def _pairs(meta) -> dict:
        by_patient = meta.groupby(patient_col)[id_col].apply(list)
        n_pairs = 0
        n_multi = 0
        for slides in by_patient:
            if len(slides) > 1:
                n_multi += 1
                n_pairs += sum(1 for _ in combinations(slides, 2))
        return {"n_slides": int(len(meta)),
                "n_patients": int(meta[patient_col].nunique()),
                "n_multi_slide_patients": int(n_multi),
                "n_within_patient_pairs": int(n_pairs)}

    full = _pairs(meta_full)
    ev = _pairs(meta_eval)
    return {
        "full_population": full,
        "roi_evaluated_population": ev,
        "adr_0002_expected_full_population": ADR_0002_WITHIN_PATIENT_PAIRS_FULL_POPULATION,
        "full_population_matches_adr_0002": full["n_within_patient_pairs"]
            == ADR_0002_WITHIN_PATIENT_PAIRS_FULL_POPULATION,
        "pairs_lost_to_misc_drop": full["n_within_patient_pairs"]
            - ev["n_within_patient_pairs"],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ROI-level aggregated retrieval evaluation (G3 fidelity gate).")
    p.add_argument("--top-k-per-roi", type=int, default=AS_RUN["top_k_per_roi"],
                   help="as-run value is 5; the Methods say ten (a C9 item)")
    p.add_argument("--vote-mode", default=AS_RUN["vote_mode"],
                   choices=["raw_hits", "unique_per_roi", "unique_library"],
                   help="raw_hits counts every hit row across all ~1,381 ROIs "
                        "of a query with no dedup, so one library slide can cast "
                        "up to ~1,381 votes. That amplification is why ROI-level "
                        "leakage is worse than slide-level, not milder.")
    p.add_argument("--iter", type=int, default=None,
                   dest="bootstrap_B",
                   help="default is as-run (1000) unless --exclude-same-patient, "
                        "in which case it is C1b's 5000 (the published Table S3 "
                        "design, per ADR 0002)")
    p.add_argument("--seed", type=int, default=None,
                   help="default is as-run (17) unless --exclude-same-patient, "
                        "in which case it is C1b's 123")
    p.add_argument("--drop-labels", nargs="*", default=AS_RUN["drop_labels"])
    p.add_argument("--n-workers", type=int, default=32,
                   help="match to the LSF slot count; the original hardcoded 128")
    p.add_argument("--task", default="g3_roi_fidelity")
    p.add_argument("--out", default=None)
    p.add_argument("--assert-published-roi", action="store_true",
                   help="G3 gate: require macro F1 == 0.7903 to 4 dp and the "
                        "published hit count, and exit non-zero if not. Only "
                        "meaningful without --exclude-same-patient.")
    p.add_argument("--exclude-same-patient", action="store_true",
                   help="C1b: filter same-patient hits from the vote pool "
                        "(paths.slide_metadata(), patient-clustered PPS "
                        "bootstrap) instead of reproducing the as-run config")
    p.add_argument("--patient-col", default=PATIENT_EXCLUDED_DEFAULTS["patient_col"])
    p.add_argument("--bootstrap-mode",
                   default=PATIENT_EXCLUDED_DEFAULTS["bootstrap_mode"],
                   choices=["pps_cluster_one_slide", "one_per_patient"])
    return p


def run(args: argparse.Namespace) -> dict:
    roi_dir = paths.require(paths.roi_hits_dir(), "ROI hit pickles")
    excl = args.exclude_same_patient
    bootstrap_B = args.bootstrap_B if args.bootstrap_B is not None else (
        PATIENT_EXCLUDED_DEFAULTS["bootstrap_B"] if excl else AS_RUN["bootstrap_B"])
    seed = args.seed if args.seed is not None else (
        PATIENT_EXCLUDED_DEFAULTS["seed"] if excl else AS_RUN["seed"])

    pair_recon = None
    if excl:
        meta_path = paths.require(paths.slide_metadata(),
                                  "patient-carrying slide metadata")
        # Loaded twice, undropped and dropped, purely for fidelity check (iii)
        # -- the pair reconciliation needs both populations and this is the
        # only place both are read.
        meta_full = load_metadata(str(meta_path), id_col=AS_RUN["id_col"],
                                  label_col=AS_RUN["label_col"],
                                  drop_labels=None,
                                  min_class_count=AS_RUN["min_class_count"],
                                  patient_col=args.patient_col)
        meta = load_metadata(str(meta_path), id_col=AS_RUN["id_col"],
                             label_col=AS_RUN["label_col"],
                             drop_labels=list(args.drop_labels),
                             min_class_count=AS_RUN["min_class_count"],
                             patient_col=args.patient_col)
        pair_recon = masked_pair_reconciliation(
            meta_full, meta, id_col=AS_RUN["id_col"], patient_col=args.patient_col)
    else:
        meta_path = paths.require(paths.roi_metadata_legacy(),
                                  "legacy (patient-free) ROI metadata")
        meta = load_metadata(str(meta_path), id_col=AS_RUN["id_col"],
                             label_col=AS_RUN["label_col"],
                             drop_labels=list(args.drop_labels),
                             min_class_count=AS_RUN["min_class_count"])

    fallbacks = count_glob_fallbacks(
        meta[AS_RUN["id_col"]].astype(str).tolist(), str(roi_dir))

    if excl:
        df_pred = predict_all_slides_agg_patient_excluded(
            roi_hits_dir=str(roi_dir), meta_df=meta,
            id_col=AS_RUN["id_col"], label_col=AS_RUN["label_col"],
            patient_col=args.patient_col,
            top_k_per_roi=args.top_k_per_roi, vote_mode=args.vote_mode,
            n_workers=args.n_workers)
    else:
        df_pred = predict_all_slides_agg(
            roi_hits_dir=str(roi_dir), meta_df=meta,
            id_col=AS_RUN["id_col"], label_col=AS_RUN["label_col"],
            top_k_per_roi=args.top_k_per_roi, vote_mode=args.vote_mode,
            n_workers=args.n_workers)

    labels_order = sorted(df_pred["true_label"].unique().tolist())
    metrics = compute_metrics_from_predictions(df_pred, labels_order=labels_order)

    if excl:
        # Fidelity check (ii): a direct count over the whole vote pool, not an
        # assumption that filtering worked. Must be exactly 0.
        surviving = int(df_pred["same_patient_hits_surviving"].sum())
        dropped = int(df_pred["same_patient_hits_dropped"].sum())
        if surviving != 0:
            raise AssertionError(
                f"C1b fidelity check (ii) FAILED: {surviving} same-patient hits "
                f"survived filtering into the counted vote pool (dropped "
                f"{dropped} correctly, but {surviving} leaked through). Do not "
                f"interpret this run's macro F1.")

        # Reuse the audited patient-clustered bootstrap (slide_eval.py), not a
        # new implementation. df_pred needs id_col in place of "slide", and
        # meta needs [id_col, label_col, patient_col] -- both required by its
        # merge(..., validate="one_to_one").
        df_pred_for_bootstrap = df_pred.rename(columns={"slide": AS_RUN["id_col"]})
        cis_raw = slide_eval.stratified_bootstrap_cis(
            df_pred=df_pred_for_bootstrap, meta=meta, B=bootstrap_B, seed=seed,
            id_col=AS_RUN["id_col"], label_col=AS_RUN["label_col"],
            patient_col=args.patient_col, labels_order=labels_order,
            mode=args.bootstrap_mode, alpha=0.05, n_jobs=args.n_workers)
        cis = {
            "micro_f1_ci": list(cis_raw["micro_f1_ci"]),
            "macro_f1_ci": list(cis_raw["macro_f1_ci"]),
            "macro_precision_ci": list(cis_raw["macro_precision_ci"]),
            "macro_recall_ci": list(cis_raw["macro_recall_ci"]),
            "per_class_f1_ci": {c: list(v) for c, v in
                                cis_raw["per_class_f1_ci"].items()},
            "B": bootstrap_B, "seed": seed, "mode": args.bootstrap_mode,
        }
        hit_at_5_rate = float(df_pred["hit_at_5"].mean()) if "hit_at_5" in df_pred else None
    else:
        cis = stratified_bootstrap_cis(df_pred=df_pred, B=bootstrap_B,
                                       seed=seed, labels_order=labels_order)
        surviving = dropped = None
        hit_at_5_rate = float(df_pred["hit_at_5"].mean()) if "hit_at_5" in df_pred else None

    diag = {c: int(df_pred[c].sum()) for c in
            ("n_rois", "skipped_rois", "total_hits_seen",
             "total_hits_counted", "missing_label_hits",
             "same_patient_hits_dropped", "same_patient_hits_surviving")
            if c in df_pred.columns}

    delta_vs_published = {
        "macro_f1": metrics["macro_f1"] - PUBLISHED_ROI["macro_f1"],
        "micro_f1": metrics["micro_f1"] - PUBLISHED_ROI["micro_f1"],
    }

    config = {
        **{k: v for k, v in AS_RUN.items() if k not in ("drop_labels",)},
        "top_k_per_roi": args.top_k_per_roi,
        "vote_mode": args.vote_mode,
        "bootstrap_B": bootstrap_B,
        "seed": seed,
        "dropped_labels": list(args.drop_labels),
        "n_workers": args.n_workers,
        "input_roi_hits_dir": str(roi_dir),
        "input_metadata": str(meta_path),
        "prosper_commit": _git_commit(),
    }
    if excl:
        config.update({
            "exclude_same_patient": True,
            "patient_column_available": True,
            "patient_col": args.patient_col,
            "exclusion": "same-patient (paths.slide_metadata()) + self, "
                        "filtered from the vote pool before top_k_per_roi",
            "bootstrap_mode": args.bootstrap_mode,
        })
    else:
        config.update({
            "exclude_same_patient": False,
            "patient_column_available": False,
            "exclusion": "self only, by string comparison (as published)",
            "bootstrap_mode": "iid_within_class (as published; the Methods "
                              "describe 5000 ROI resamples instead)",
        })

    results = {
        "config": config,
        "n_slides": int(len(df_pred)),
        "labels_order": labels_order,
        "point_estimates": {k: metrics[k] for k in
                            ("micro_f1", "macro_f1", "macro_precision",
                             "macro_recall") if k in metrics},
        "per_class_f1": {c: v["f1"] for c, v in metrics.get("per_class", {}).items()},
        "per_class": metrics.get("per_class"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "confidence_intervals": cis,
        "roi_diagnostics": diag,
        "glob_fallback_audit": fallbacks,
        "hit_at_5_rate": hit_at_5_rate,
        "delta_vs_published_roi": delta_vs_published,
        "masked_pair_reconciliation": pair_recon,
    }

    out = Path(args.out) if args.out else paths.result_json(
        args.task, f"roi_agg_k{args.top_k_per_roi}_{args.vote_mode}"
                  f"{'_patient_excluded' if excl else ''}")
    out.write_text(json.dumps(results, indent=2))
    print(f"[prosper.roi] wrote {out}")
    print(f"[prosper.roi] macro_f1 = {metrics['macro_f1']:.6f}  "
          f"micro_f1 = {metrics['micro_f1']:.6f}  n_slides = {len(df_pred)}")
    print(f"[prosper.roi] hits counted = {diag.get('total_hits_counted')}  "
          f"seen = {diag.get('total_hits_seen')}")
    print(f"[prosper.roi] glob-fallback slides = "
          f"{fallbacks['n_would_use_glob_fallback']}  "
          f"no-file slides = {fallbacks['n_no_file_at_all']}")
    if excl:
        print(f"[C1b] same-patient hits dropped = {dropped}  "
              f"surviving in counted pool = {surviving} (must be 0)")
        print(f"[C1b] delta macro_f1 vs published = {delta_vs_published['macro_f1']:+.6f}  "
              f"delta micro_f1 = {delta_vs_published['micro_f1']:+.6f}")
        if pair_recon:
            print(f"[C1b] within-patient pairs: full pop = "
                  f"{pair_recon['full_population']['n_within_patient_pairs']} "
                  f"(ADR 0002 expects {ADR_0002_WITHIN_PATIENT_PAIRS_FULL_POPULATION}, "
                  f"match={pair_recon['full_population_matches_adr_0002']}); "
                  f"ROI-evaluated pop = "
                  f"{pair_recon['roi_evaluated_population']['n_within_patient_pairs']}")
    return results


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    r = run(args)
    if not args.assert_published_roi:
        return 0
    if args.exclude_same_patient:
        raise SystemExit(
            "--assert-published-roi checks the as-run (unmasked) config; it is "
            "not meaningful with --exclude-same-patient, which is expected to "
            "move macro F1 -- that movement is the point of C1b.")

    got_f1 = r["point_estimates"]["macro_f1"]
    got_hits = r["roi_diagnostics"].get("total_hits_counted")
    exp_f1 = PUBLISHED_ROI["macro_f1"]
    exp_hits = PUBLISHED_ROI["sum_total_hits_counted"]
    ok_f1 = abs(got_f1 - exp_f1) < 5e-5
    ok_hits = got_hits == exp_hits

    print(f"[G3] macro_f1     : {got_f1:.6f} vs {exp_f1:.4f}  "
          f"{'OK' if ok_f1 else f'MISMATCH (delta {got_f1 - exp_f1:+.2e})'}")
    print(f"[G3] hits counted : {got_hits} vs {exp_hits}  "
          f"{'OK' if ok_hits else 'MISMATCH'}")
    print(f"[G3] micro_f1     : {r['point_estimates']['micro_f1']:.6f}  "
          f"(Fig 6A plots this as 'Overall'; the Results text quotes macro)")
    if ok_f1 and ok_hits:
        print("[G3] PASS")
        return 0
    if ok_hits and not ok_f1:
        print("[G3] FAIL on metric with the hit count intact: the vote pool is "
              "right, so the divergence is in voting or scoring.")
    elif ok_f1 and not ok_hits:
        print("[G3] FAIL on hit count with macro F1 intact: suspect top_k or "
              "vote_mode -- the published run counted top-5 of stored top-10.")
    print("[G3] STOP. Do not run C1b: a change in Fig 6A would not be "
          "attributable to patient exclusion.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
