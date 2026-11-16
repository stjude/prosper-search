"""
R3.1 addendum (design decision F1, 2026-10-16): whole-patient bootstrap intervals for the RetCCL encoder
(Wang et al., 2023) under the four aggregation methods, from SAVED ranked lists, so that RetCCL can appear in
Supplementary Table S-R3.1 beside SISH and the ResNet-50 floor.

This script does not modify the frozen r3_1_comparators_whole_patient_20261008/ bundle. It imports the accepted
functions of scripts/r3_1_comparators_whole_patient.py (loader, per-query scoring, bootstrap block) unchanged and
runs ONE all-tissue block (2,778 queries, 15 groups incl. NORMAL, same-patient masked) containing the reference
configurations needed for pairing plus the RetCCL configurations. Because the block uses the same whole-patient
sampler, B and seed (5,000, PCG64 seed 123) as the frozen run, the reference configurations' points/intervals must
reproduce the frozen summary exactly; this is asserted and recorded in reconciliation.json.

Configurations (ranked lists, c2_metric_suite/ and c5_resnet50/):
  primary, median, yottixel      uni2 + exact OT / median pooling / Yottixel-style   (frozen references)
  resnet50                       ImageNet ResNet-50 + median pooling                 (frozen reference, floor)
  retccl_median / retccl_ot / retccl_yottixel / retccl_sampler
                                 RetCCL tile encoder + the same four aggregations
Paired differences (minuend - subtrahend; same queries, same draws):
  retccl_<agg> - primary (x4); retccl_median - median and retccl_yottixel - yottixel (encoder swap at fixed
  aggregation); retccl_median - resnet50 (self-supervised pathology encoder vs ImageNet floor at fixed aggregation).
Intervals are nominal 95 %, unadjusted. An interval including 0 is not evidence of equivalence.

RetCCL preprocessing caveat (c5_new_encoders.md): RetCCL has no local reference extractor and its preprocessing
follows the official-repo convention, marked PROVISIONAL in the registry; disclosed, not re-extracted, here.

Outputs (--out-dir, default $PROSPER_OUT/r3_1_retccl_whole_patient_<YYYYMMDD>/):
  summary.csv, paired_differences.csv, per_group.csv, per_query_frame.csv.gz, reconciliation.json, README.md,
  sha256.txt, COMPLETE
"""
import argparse
import datetime as dt
import importlib.util
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
OUT = os.environ.get("PROSPER_OUT", os.path.join(R, "prosper_revision_output"))
BASE = os.path.join(HERE, "r3_1_comparators_whole_patient.py")
FROZEN = os.path.join(OUT, "r3_1_comparators_whole_patient_20261008")

spec = importlib.util.spec_from_file_location("r31base", BASE)
B_ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B_)
L = B_.L

REFERENCE = [c for c in B_.MAIN if c[0] != "sish"]
RETCCL = [
    ("retccl_median", "RetCCL + median pooling", "c2_metric_suite/ranked_lists_retccl_median_pooling.csv.gz", "c2_metric_suite/metrics_retccl_median_pooling.json"),
    ("retccl_ot", "RetCCL + exact OT (cosine)", "c2_metric_suite/ranked_lists_retccl_ot_cosine.csv.gz", "c2_metric_suite/metrics_retccl_ot_cosine.json"),
    ("retccl_yottixel", "RetCCL + Yottixel-style", "c2_metric_suite/ranked_lists_retccl_yottixel.csv.gz", "c2_metric_suite/metrics_retccl_yottixel.json"),
    ("retccl_sampler", "RetCCL + sampler aggregation", "c2_metric_suite/ranked_lists_retccl_sampler.csv.gz", "c2_metric_suite/metrics_retccl_sampler.json"),
]
CONTRASTS = [(t, "primary") for t, *_ in RETCCL] + [("retccl_median", "median"), ("retccl_yottixel", "yottixel"), ("retccl_median", "resnet50")]
CHECK_METRICS = ["hit1", "hit3", "hit5", "hit10", "precision5", "precision10", "plurality5"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=L.META)
    ap.add_argument("--out-dir", default=os.path.join(OUT, f"r3_1_retccl_whole_patient_{dt.date.today():%Y%m%d}"))
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--frozen", default=FROZEN)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=False)
    meta, meta_sha = L.load_meta(a.meta)
    configs = REFERENCE + RETCCL
    hashes, frames, padded_n, saved = {}, {}, {}, {}
    for tag, desc, rel, mj in configs:
        r, sha, npad = B_.load_ranks_padded(os.path.join(OUT, rel), meta, B_.CLASSES15)
        hashes[rel] = sha
        padded_n[tag] = npad
        frames[tag] = B_.per_query_simple(r, meta)
        if len(frames[tag]) != L.MAIN_POPULATION_QUERIES:
            raise ValueError(f"{tag}: {len(frames[tag])} queries != {L.MAIN_POPULATION_QUERIES}")
        saved[tag] = B_.saved_points(os.path.join(OUT, mj))
    ref = frames["primary"]
    for tag, f in frames.items():
        if not f.index.equals(ref.index) or not (f["true_label"] == ref["true_label"]).all() or not (f["patient"] == ref["patient"]).all():
            raise ValueError(f"{tag}: query/label/patient frame does not align with primary")
    if any(v for v in padded_n.values()):
        raise ValueError(f"unexpected padding: {padded_n}")
    block = B_.boot_block(frames, [t for t, *_ in configs], B_.CLASSES15, a.B, a.seed, CONTRASTS, "all_tissue_2778")
    summary, paired, per_group = block["summary"], block["paired"], block["per_group"]

    # 1) reference configurations must reproduce the frozen bundle exactly (same sampler, B, seed, draws)
    frozen_summary = pd.read_csv(os.path.join(a.frozen, "summary.csv"))
    frozen_summary = frozen_summary[frozen_summary["population"].eq("all_tissue_2778")]
    ref_tags = [t for t, *_ in REFERENCE]
    key = ["config", "metric", "average"]
    m = summary[summary["config"].isin(ref_tags)].merge(frozen_summary, on=key, suffixes=("", "_frozen"))
    if len(m) != len(ref_tags) * len(CHECK_METRICS) * 2:
        raise ValueError(f"frozen reconciliation join incomplete: {len(m)} rows")
    ref_diff = {c: float((m[c] - m[f"{c}_frozen"]).abs().max()) for c in ("point", "ci_lo", "ci_hi")}
    if max(ref_diff.values()) > 1e-12:
        raise ValueError(f"reference configurations do not reproduce the frozen bundle: {ref_diff}")

    # 2) RetCCL points vs the saved legacy JSON points (pps-legacy intervals there are not quotable; points must agree)
    rec_points = []
    for t, *_ in RETCCL:
        sp = saved.get(t)
        if not sp or sp[0] is None:
            continue
        pts, mode = sp
        for k, val in pts.items():
            if val is None:
                continue
            avg, metric = k.split("_", 1)
            metric = metric.replace("_at_", "")
            mine = block["points"].get((t, metric, avg))
            rec_points.append(dict(config=t, metric=metric, average=avg, saved_point=val, recomputed_point=mine,
                                   abs_diff=(None if mine is None else abs(mine - val)), saved_bootstrap_mode=mode))

    frame = pd.concat(frames, names=["config", "query"])
    recon = dict(
        script=os.path.relpath(__file__, R), script_sha256=L.sha256_bytes(L.read_bytes(__file__)),
        imported_base_script=os.path.relpath(BASE, R), imported_base_sha256=L.sha256_bytes(L.read_bytes(BASE)),
        imported_ladder_script=os.path.relpath(B_.LADDER, R), imported_ladder_sha256=L.sha256_bytes(L.read_bytes(B_.LADDER)),
        metadata=a.meta, metadata_sha256=meta_sha, inputs_sha256=hashes, B=a.B, seed=a.seed,
        bootstrap="ADR0021 whole-patient: patients with replacement, identical draws for all configurations in the block",
        population=dict(all_tissue_2778=dict(n_queries=block["n_queries"], n_patients=block["n_patients"], n_groups=15)),
        frozen_bundle=a.frozen, frozen_summary_sha256=L.sha256_bytes(L.read_bytes(os.path.join(a.frozen, "summary.csv"))),
        reference_vs_frozen_max_abs_diff=ref_diff,
        retccl_saved_point_reconciliation=rec_points,
        retccl_max_abs_point_diff=max([x["abs_diff"] for x in rec_points if x["abs_diff"] is not None] or [None]),
        retccl_preprocessing="PROVISIONAL (c5_new_encoders.md): official-repo convention, no local reference extractor; disclosed, not re-extracted",
        padding={t: int(v) for t, v in padded_n.items()},
    )
    summary.to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)
    paired.to_csv(os.path.join(a.out_dir, "paired_differences.csv"), index=False)
    per_group.to_csv(os.path.join(a.out_dir, "per_group.csv"), index=False)
    frame.to_csv(os.path.join(a.out_dir, "per_query_frame.csv.gz"), compression="gzip")
    json.dump(recon, open(os.path.join(a.out_dir, "reconciliation.json"), "w"), indent=2, default=str)
    lines = [__doc__.strip(), "", "## Run", "", f"B = {a.B}, seed = {a.seed}; {dt.datetime.now():%Y-%m-%dT%H:%M:%S}",
             f"reference configurations vs frozen bundle max |diff| (point/ci_lo/ci_hi): {ref_diff}",
             f"RetCCL max |saved point - recomputed point| over hit@5/10, precision@5: {recon['retccl_max_abs_point_diff']}",
             "", "## RetCCL points (macro, hit@5 / precision@5)", ""]
    for t, *_ in RETCCL:
        h = summary[(summary.config == t) & (summary.metric == "hit5") & (summary.average == "macro")].iloc[0]
        p = summary[(summary.config == t) & (summary.metric == "precision5") & (summary.average == "macro")].iloc[0]
        lines.append(f"- {t}: hit5 {h.point:.3f} [{h.ci_lo:.3f}, {h.ci_hi:.3f}]; precision5 {p.point:.3f} [{p.ci_lo:.3f}, {p.ci_hi:.3f}]")
    lines += ["", "## Paired differences (macro, hit@5 / precision@5)", ""]
    for _, row in paired[(paired.average == "macro") & (paired.metric.isin(["hit5", "precision5"]))].iterrows():
        lines.append(f"- {row.minuend} - {row.subtrahend} {row.metric}: {row.point:+.3f} [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}]")
    open(os.path.join(a.out_dir, "README.md"), "w").write("\n".join(lines) + "\n")
    with open(os.path.join(a.out_dir, "sha256.txt"), "w") as f:
        for fn in ("summary.csv", "paired_differences.csv", "per_group.csv", "per_query_frame.csv.gz", "reconciliation.json", "README.md"):
            f.write(f"{L.sha256_bytes(L.read_bytes(os.path.join(a.out_dir, fn)))}  {fn}\n")
    open(os.path.join(a.out_dir, "COMPLETE"), "w").write(f"{dt.datetime.now():%Y-%m-%dT%H:%M:%S}\n")
    print("\n".join(lines[lines.index("## Run"):]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
