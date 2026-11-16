#!/usr/bin/env python3
"""R2.5-ii: [decision record] whole-patient bootstrap intervals for the C4 ROI-heatmap localisation points.

Design decision 2026-10-08 (Q4 option b): before the moderated Fig. 5/6 localisation statement is
settled, attach whole-patient intervals to the quoted point estimates. Reads ONLY the saved per-pair table
$PROSPER_OUT/c4_localization_roi8/per_pair_c4_localization_roi8.csv (job 319177602, 2026-08-29); no heatmap,
feature, model or package code is touched or re-run.

Population: variant A (labelled patches only; 1 = tumour, 2-5 = non-tumour, 0 excluded) on the 44 annotated
reference slides with >= 20 tumour- and >= 20 non-tumour-labelled patches that have BOTH a same-group and an
other-group arm (one further reference, 13748, has only an other-group arm and is excluded from every
paired estimand; its all-45 other-group means are recorded for reconciliation with the saved JSON only).
Each arm has exactly 10 queries per reference.

Estimands (per reference = mean over its 10 queries; then mean over references):
  auroc_same, auroc_other, auroc_diff = same - other (paired on the same reference)
  pointing_same, pointing_other, pointing_diff = same - other,
  pointing_minus_share = pointing_same - tumour_fraction_A (the per-reference tumour share among labelled
                         patches, i.e. the chance level quoted as 0.683)
  ap_same, ap_other, ap_diff (recorded, not currently quoted)
  frac_refs_auroc_diff_positive (count, not resampled)

Bootstrap: patients drawn with replacement, B = 5,000, numpy PCG64 seed 123, every reference weighted by its
patient's draw multiplicity (identical draws for every estimand, so paired differences are exact);
percentile 2.5/97.5 (numpy linear), nominal 95 %, unadjusted for multiplicity. `make_patient_draws` is
byte-for-byte the function of the accepted scripts/r2_2_tile_ladder_whole_patient.py. Because the 44 references
belong to 44 distinct patients, the whole-patient bootstrap coincides with a whole-reference bootstrap here;
this is recorded, not assumed.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

B_DEFAULT = 5000
SEED_DEFAULT = 123
EXPECTED_CSV_SHA = "df0f103e"  # prefix recorded in the [internal record] (R2.5-T5 sources); full hash written to the record


def make_patient_draws(patient_ids, count=5000, seed=123):
    count, seed = int(count), int(seed)
    if any(not isinstance(v, str) or not v.strip() for v in patient_ids):
        raise ValueError("patient_ids must contain nonempty string IDs")
    patients = tuple(sorted(set(patient_ids)))
    p = len(patients)
    multiplicities = np.zeros((count, p), dtype=np.int64)
    if p:
        rng = np.random.Generator(np.random.PCG64(seed))
        for b in range(count):
            multiplicities[b] = np.bincount(rng.integers(p, size=p), minlength=p)
    return patients, multiplicities


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ci(v):
    v = np.asarray(v, float)
    if not v.size or (~np.isfinite(v)).any():
        return None, None
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(os.environ["PROSPER_OUT"], "c4_localization_roi8",
                                                  "per_pair_c4_localization_roi8.csv"))
    ap.add_argument("--json", default=os.path.join(os.environ["PROSPER_OUT"], "c4_localization_roi8",
                                                   "c4_localization_roi8.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"--out exists: {a.out}")

    csv_sha = sha256_file(a.csv)
    json_sha = sha256_file(a.json)
    if not csv_sha.startswith(EXPECTED_CSV_SHA):
        sys.exit(f"per-pair CSV sha256 {csv_sha} does not start with {EXPECTED_CSV_SHA}")
    df = pd.read_csv(a.csv, dtype={"reference": str, "query_slide": str, "ref_patient": str})
    needed = ["reference", "ref_patient", "arm", "auroc_A", "pointing_A", "ap_A", "tumour_fraction_A"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"missing columns: {missing}")

    va = df[df["auroc_A"].notna()].copy()
    va["pointing_A"] = va["pointing_A"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    if va["pointing_A"].isna().any():
        sys.exit("pointing_A has values other than True/False among variant-A rows")

    per = (va.groupby(["arm", "reference", "ref_patient"], as_index=False)
             .agg(auroc=("auroc_A", "mean"), pointing=("pointing_A", "mean"), ap=("ap_A", "mean"),
                  share=("tumour_fraction_A", "mean"), share_min=("tumour_fraction_A", "min"),
                  share_max=("tumour_fraction_A", "max"), n_queries=("auroc_A", "size")))
    if (per["n_queries"] != 10).any():
        sys.exit("expected exactly 10 queries per reference per arm")
    if (per["share_max"] - per["share_min"]).abs().max() > 0:
        sys.exit("tumour_fraction_A differs across queries of one reference")

    same = per[per["arm"] == "same_group"].set_index("reference")
    other = per[per["arm"] == "other_group"].set_index("reference")
    paired = same.index.intersection(other.index)
    other_only = sorted(set(other.index) - set(same.index))
    same_only = sorted(set(same.index) - set(other.index))
    if len(paired) != 44 or same_only:
        sys.exit(f"expected 44 paired references and no same-only references; got {len(paired)}, {same_only}")
    s = same.loc[paired]
    o = other.loc[paired]
    if (s["ref_patient"].values != o["ref_patient"].values).any():
        sys.exit("ref_patient differs between arms for a reference")
    if (s["share"].values != o["share"].values).any():
        sys.exit("tumour share differs between arms for a reference")

    patients, mult = make_patient_draws(list(s["ref_patient"]), count=a.B, seed=a.seed)
    idx = pd.Index(patients).get_indexer(list(s["ref_patient"]))
    W = mult[:, idx].astype(float)
    den = W.sum(1)

    def wmean(v):
        return (W @ np.asarray(v, float)) / den

    est = {
        "auroc_same": s["auroc"].values,
        "auroc_other": o["auroc"].values,
        "auroc_diff": s["auroc"].values - o["auroc"].values,
        "pointing_same": s["pointing"].values,
        "pointing_other": o["pointing"].values,
        "pointing_diff": s["pointing"].values - o["pointing"].values,
        "tumour_share": s["share"].values,
        "pointing_minus_share": s["pointing"].values - s["share"].values,
        "ap_same": s["ap"].values,
        "ap_other": o["ap"].values,
        "ap_diff": s["ap"].values - o["ap"].values,
    }
    results = {}
    for name, v in est.items():
        reps = wmean(v)
        lo, hi = ci(reps)
        results[name] = {"point": float(np.mean(v)), "ci95_lo": lo, "ci95_hi": hi,
                         "n_references": int(len(v)), "n_replicates_finite": int(np.isfinite(reps).sum())}
    counts = {
        "n_refs_auroc_diff_positive": int((est["auroc_diff"] > 0).sum()),
        "n_refs_auroc_diff_zero": int((est["auroc_diff"] == 0).sum()),
        "n_refs_pointing_diff_positive": int((est["pointing_diff"] > 0).sum()),
        "n_refs_pointing_diff_negative": int((est["pointing_diff"] < 0).sum()),
        "n_refs_pointing_minus_share_positive": int((est["pointing_minus_share"] > 0).sum()),
    }
    recon = {
        "other_group_all_references": {"n_references": int(len(other)),
                                       "auroc_mean": float(other["auroc"].mean()),
                                       "pointing_mean": float(other["pointing"].mean()),
                                       "ap_mean": float(other["ap"].mean())},
        "other_only_references": other_only,
        "saved_json_same_group_auroc_A": None, "saved_json_other_group_auroc_A": None,
    }
    try:
        j = json.load(open(a.json))
        recon["saved_json_same_group_auroc_A"] = j["summary"]["same_group"]["mean_over_references"]["auroc_A"]
        recon["saved_json_other_group_auroc_A"] = j["summary"]["other_group"]["mean_over_references"]["auroc_A"]
        recon["saved_json_same_group_pointing_A_present"] = "pointing_A" in j["summary"]["same_group"]["mean_over_references"]
    except Exception as e:  # reconciliation only; never fatal
        recon["saved_json_error"] = repr(e)
    recon["same_group_matches_saved_json"] = (recon["saved_json_same_group_auroc_A"] is not None and
                                              abs(results["auroc_same"]["point"] - recon["saved_json_same_group_auroc_A"]) < 1e-12)
    recon["other_group_all45_matches_saved_json"] = (recon["saved_json_other_group_auroc_A"] is not None and
                                                     abs(recon["other_group_all_references"]["auroc_mean"] - recon["saved_json_other_group_auroc_A"]) < 1e-12)

    os.makedirs(a.out, exist_ok=False)
    per_ref = pd.DataFrame({"reference": paired, "ref_patient": s["ref_patient"].values,
                            "ref_group": va.drop_duplicates("reference").set_index("reference").loc[paired, "ref_group"].values,
                            **{k: v for k, v in est.items()}})
    per_ref_path = os.path.join(a.out, "per_reference_variant_A.csv")
    per_ref.to_csv(per_ref_path, index=False, float_format="%.12g")

    rec = {
        "script": os.path.basename(__file__), "script_sha256": sha256_file(os.path.abspath(__file__)),
        "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {"per_pair_csv": a.csv, "per_pair_csv_sha256": csv_sha, "json": a.json, "json_sha256": json_sha},
        "config": {"B": a.B, "seed": a.seed, "generator": "numpy PCG64",
                   "resampling_unit": "ref_patient (drawn with replacement; references weighted by multiplicity)",
                   "n_patients": len(patients), "n_references": int(len(paired)),
                   "patients_equal_references": len(patients) == len(paired),
                   "ci": "percentile 2.5/97.5 (numpy linear), nominal 95%, unadjusted for multiplicity",
                   "variant": "A (labelled patches only; 1 tumour, 2-5 non-tumour, 0 excluded)",
                   "queries_per_reference_per_arm": 10},
        "results": results, "counts": counts, "reconciliation": recon,
        "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0],
    }
    with open(os.path.join(a.out, "c4_whole_patient_ci.json"), "w") as f:
        json.dump(rec, f, indent=2)

    def fmt(k):
        r = results[k]
        return f"{r['point']:.3f} [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]"
    md = [
        "# C4 ROI-heatmap localisation: whole-patient intervals (variant A, 44 paired references / 44 patients)", "",
        f"B = {a.B}, PCG64 seed {a.seed}; percentile 95 %; per-pair CSV sha256 {csv_sha}.", "",
        "| estimand | point [95 % CI] |", "|---|---|",
    ] + [f"| {k} | {fmt(k)} |" for k in results] + ["", f"Counts: {json.dumps(counts)}", "",
        f"Reconciliation: same-group AUROC matches saved JSON: {recon['same_group_matches_saved_json']}; "
        f"other-group all-{recon['other_group_all_references']['n_references']} AUROC {recon['other_group_all_references']['auroc_mean']:.6f} "
        f"matches saved JSON: {recon['other_group_all45_matches_saved_json']}; other-only reference(s): {other_only}.", "",
        "Whole-patient = whole-reference here because every paired reference is from a distinct patient "
        f"({len(patients)} patients): {rec['config']['patients_equal_references']}."]
    with open(os.path.join(a.out, "c4_whole_patient_ci.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(a.out, "sha256.txt"), "w") as f:
        for name in ["c4_whole_patient_ci.json", "c4_whole_patient_ci.md", "per_reference_variant_A.csv"]:
            f.write(f"{sha256_file(os.path.join(a.out, name))}  {name}\n")
    open(os.path.join(a.out, "COMPLETE"), "w").write(rec["finished"] + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
