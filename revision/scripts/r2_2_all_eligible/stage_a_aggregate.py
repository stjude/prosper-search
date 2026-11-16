"""R2.2-ALL Stage A aggregation over per-slide outputs (small, local).

Reads stage_a/slides/<sid>.{json,manifest.csv.gz,bankcmp.npz} and
stage_a/bindings/<sid>.json for every target slide; writes
  slide_summary.csv.gz, gap_manifest.csv.gz, eligible_counts.json,
  bankcmp_pooled.json, aggregate.json, input_bindings.json
under stage_a/. Does not write COMPLETE/HOLD (the analyst does, after review).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
H = Path("/PROSPER_WORKSPACE")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def q(a, ps=(0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1)):
    a = np.asarray(a)
    return {str(p): float(np.quantile(a, p)) for p in ps} if a.size else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", required=True)
    a = ap.parse_args()
    A = Path(a.stage_a)
    S = A / "slides"
    sids = [s.strip() for s in open(A / "inputs/target_slides.txt") if s.strip()]
    pop = json.loads((A / "inputs/population.json").read_text())
    agg = {"n_target": len(sids)}

    recs, missing_json = [], []
    for sid in sids:
        p = S / f"{sid}.json"
        if not p.exists():
            missing_json.append(sid)
            continue
        recs.append(json.loads(p.read_text()))
    agg["missing_json"] = missing_json
    agg["failed_files_present"] = sorted(p.name for p in S.glob("*.FAILED.txt"))
    code_shas = {json.dumps(r["code_sha256"], sort_keys=True) for r in recs}
    agg["code_sha256_variants"] = [json.loads(c) for c in code_shas]

    rows = []
    check_fail = {}
    bank_fail = {}
    for r in recs:
        for k in r["check_failures"]:
            check_fail.setdefault(k, []).append(r["slide"])
        for k in r["bank_binding_failures"]:
            bank_fail.setdefault(k, []).append(r["slide"])
        row = {k: r.get(k) for k in (
            "slide", "width", "height", "level_count", "objective_power", "mpp_x", "thumbnail_route",
            "grid_rows", "grid_cols", "n_cells", "n_eligible", "n_full", "n_partial_eligible", "n_mosaic",
            "n_cluster_locs", "n_csv", "csv_present", "full_bank_present", "n_both_banks",
            "n_src_mosaic_bank", "n_src_full_bank_only", "n_src_missing", "n_missing_full",
            "n_missing_partial", "n_full_bank_not_eligible", "mosaic_recon_cov_min",
            "mosaic_recon_cov_n_full", "manifest_rows", "host", "lsf_jobid", "t_mask_s", "t_total_s")}
        row["thumbnail_size"] = "x".join(map(str, r["thumbnail_size"]))
        row["check_failures"] = ";".join(r["check_failures"])
        row["bank_binding_failures"] = ";".join(r["bank_binding_failures"])
        for k, v in r["checks"].items():
            row["chk_" + k] = v
        for k, v in r["bank"].items():
            row["bank_" + k] = v if not isinstance(v, list) else "x".join(map(str, v))
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(A / "slide_summary.csv.gz", index=False)
    agg["check_failures_by_check"] = {k: {"n_slides": len(v), "slides": v[:50]} for k, v in check_fail.items()}
    agg["bank_binding_failures_by_check"] = {k: {"n_slides": len(v), "slides": v[:50]} for k, v in bank_fail.items()}

    tot = {k: int(summ[k].sum()) for k in (
        "n_cells", "n_eligible", "n_full", "n_partial_eligible", "n_mosaic", "n_src_mosaic_bank",
        "n_src_full_bank_only", "n_src_missing", "n_missing_full", "n_missing_partial",
        "n_full_bank_not_eligible", "n_both_banks")}
    tot["n_cluster_locs"] = int(summ["n_cluster_locs"].fillna(0).sum())
    tot["n_mosaic_partial"] = int(tot["n_mosaic"] - summ["mosaic_recon_cov_n_full"].sum())
    nocsv = summ[~summ["csv_present"].astype(bool)]
    tot["no_csv_target_slides"] = len(nocsv)
    tot["no_csv_missing_tiles"] = int(nocsv["n_src_missing"].sum())
    tot["no_csv_missing_full_tiles"] = int(nocsv["n_missing_full"].sum())
    agg["totals"] = tot
    agg["thumbnail_routes"] = summ["thumbnail_route"].value_counts().to_dict()
    agg["per_slide_eligible"] = q(summ["n_eligible"], (0, 0.5, 0.9, 0.99, 1))
    agg["per_slide_missing"] = q(summ["n_src_missing"], (0, 0.5, 0.9, 0.99, 1))
    agg["largest_eligible_slides"] = summ.nlargest(10, "n_eligible")[["slide", "n_eligible", "n_mosaic"]].to_dict("records")
    agg["eligible_to_mosaic_ratio"] = q(summ["n_eligible"] / summ["n_mosaic"], (0, 0.5, 1))
    agg["timing_s"] = {"t_total_sum": float(summ["t_total_s"].sum()), "t_total_median": float(summ["t_total_s"].median()),
                       "t_total_max": float(summ["t_total_s"].max()), "t_mask_median": float(summ["t_mask_s"].median())}
    agg["hosts"] = summ["host"].value_counts().to_dict()

    # gap manifest + manifest consistency
    gaps, man_tot = [], {"eligible": 0, "missing": 0, "mosaic": 0, "full_bank_only": 0}
    for sid in summ["slide"]:
        m = pd.read_csv(S / f"{sid}.manifest.csv.gz")
        e = m[m.eligible == 1]
        man_tot["eligible"] += len(e)
        man_tot["missing"] += int((e.source == "missing").sum())
        man_tot["mosaic"] += int((e.source == "mosaic_bank").sum())
        man_tot["full_bank_only"] += int((e.source == "full_bank").sum())
        g = e[e.source == "missing"][["wsi_loc_x", "wsi_loc_y", "grid_i", "grid_j", "coverage", "full"]].copy()
        g.insert(0, "slide", sid)
        gaps.append(g)
    gap = pd.concat(gaps, ignore_index=True)
    gap.to_csv(A / "gap_manifest.csv.gz", index=False, float_format="%.17g")
    agg["manifest_totals"] = man_tot
    agg["manifest_totals_match_json"] = (man_tot["eligible"] == tot["n_eligible"] and man_tot["missing"] == tot["n_src_missing"]
                                         and man_tot["mosaic"] == tot["n_src_mosaic_bank"]
                                         and man_tot["full_bank_only"] == tot["n_src_full_bank_only"])
    agg["gap_rows"] = len(gap)
    agg["gap_slides"] = int(gap["slide"].nunique())
    agg["gap_storage_bytes_fp32"] = int(len(gap) * 1536 * 4)
    du = shutil.disk_usage(os.environ["PROSPER_OUT"])
    agg["disk_free_bytes"] = du.free

    # pooled cross-bank comparison
    cos, mad, agr, ex, nsl = [], [], [], [], 0
    for sid in summ["slide"]:
        p = S / f"{sid}.bankcmp.npz"
        if p.exists():
            z = np.load(p)
            cos.append(z["cosine"]); mad.append(z["max_abs"]); agr.append(z["barcode_agree"]); ex.append(z["exact"])
            nsl += 1
    cos, mad, agr, ex = map(np.concatenate, (cos, mad, agr, ex))
    bc = {"n_slides": nsl, "n_tiles": int(cos.size), "n_exact": int(ex.sum()),
          "cosine": q(cos), "max_abs": q(mad), "barcode_agree": q(agr), "barcode_agree_mean": float(agr.mean()),
          "n_cos_below_0.999879": int((cos < 0.999879).sum()), "n_cos_below_0.999": int((cos < 0.999).sum()),
          "n_agree_below_0.99": int((agr < 0.99).sum()), "n_agree_below_0.95": int((agr < 0.95).sum())}
    bc["material"] = not (cos.min() >= 0.999 and np.quantile(agr, 0.01) >= 0.99 and agr.min() >= 0.95)
    (A / "bankcmp_pooled.json").write_text(json.dumps(bc, indent=1))
    agg["bankcmp"] = bc

    agg["population"] = {k: pop[k] for k in ("n_labels", "n_mosaic", "n_misc", "n_target", "n_target_patients",
                                             "n_classes", "n_target_mosaic_tiles", "n_no_csv", "n_no_csv_in_target")}
    agg["population_mosaic_tiles_match"] = tot["n_mosaic"] == pop["n_target_mosaic_tiles"]

    # bindings
    B = A / "bindings"
    per = {}
    nb_missing = []
    for sid in sids:
        p = B / f"{sid}.json"
        if p.exists():
            per[sid] = json.loads(p.read_text())
        else:
            nb_missing.append(sid)
    glob_in = {
        "labels_csv": H / "comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv",
        "mosaic_simplified_pkl": H / "comet-new-scans/output/mosaic/mosaic512_final_simplified.pkl",
    }
    bind = {"global": {k: {"path": str(v), "size": v.stat().st_size, "sha256": sha256_file(v)} for k, v in glob_in.items()},
            "per_slide": per, "per_slide_missing": nb_missing,
            "wsi_note": "WSIs fingerprinted by size/mtime/realpath (not hashed; 2,778 files of ~1-5 GB)."}
    (A / "input_bindings.json").write_text(json.dumps(bind, indent=1))
    agg["bindings_missing"] = nb_missing

    checks_ok = not missing_json and not check_fail and not agg["failed_files_present"] and len(code_shas) == 1
    agg["containment_all_pass"] = bool(checks_ok and agg["manifest_totals_match_json"] and agg["population_mosaic_tiles_match"])
    agg["bank_binding_all_pass"] = not bank_fail
    (A / "aggregate.json").write_text(json.dumps(agg, indent=1, default=str))
    print(json.dumps({k: agg[k] for k in ("containment_all_pass", "bank_binding_all_pass", "totals", "gap_rows")}, indent=1))


if __name__ == "__main__":
    main()
