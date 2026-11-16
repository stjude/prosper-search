#!/usr/bin/env python3
"""R3.5: cosine ROI heat maps on the 44-reference annotation design of Fig. 6B / Supplementary Table S25 block C.

decision 2026-11-11: "let's try cosine for both roi search and heatmap"; 2026-11-12: draft the dependent text
and tables. The submitted Fig. 6B maps used L1 distance; the manuscript's annotation check (C4, 44 paired
references, 0.691 [0.614, 0.764]) evaluated those L1 maps. This script re-tabulates the same comparison for
cosine maps from saved per-pair values only; no map, feature or pair is recomputed.

Source: the accepted R3.5-L1 parameter study, s2/run_003/per_pair_A.csv.gz, design P2 (the C4-ROI8 pair
construction: ten same-group and ten other-group annotated queries per reference, each query feature the mean
of up to eight tumour-labelled tiles), settings A|l1|mean (submitted map) and A|cosine|mean, variant A
(labelled patches only; 1 tumour, 2-5 non-tumour). Restricted to the 44 C4 references with both arms.

Estimands as in scripts/r2_5_c4_whole_patient_ci.py (per reference = mean over its ten queries per arm,
then mean over references): same-group, other-group and same - other AUROC / pointing / AP for each
distance, and the paired cosine - L1 contrasts. Bootstrap: that script's make_patient_draws (B = 5,000,
PCG64 seed 123, whole reference patient; 44 patients = 44 references), percentile 95 %, nominal, unadjusted.

Gates (all must pass; the record is written only if they do):
  G1  L1 per-reference same/other AUROC, pointing and AP equal the accepted C4 per_reference_variant_A.csv (1e-9)
  G2  L1 point and interval of every C4 estimand equal c4_whole_patient_ci.json (1e-9): same draws, same kernel
  G3  ten queries per reference per arm; 440 + 440 pairs; one patient per reference
  G4  distinct route: per-pair cosine AUROC equals auroc_roi_cos of the separately computed R3.5-TCM run
      (r3_5_tumour_correspondence_map_20261111/per_pair.csv.gz, in_manuscript_44 pairs) to 1e-9
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ["PROSPER_OUT"])
PP = OUT_ROOT / "r3_5_localisation_maps_20261017" / "s2" / "run_003" / "per_pair_A.csv.gz"
S2_COMPLETE = PP.parent / "COMPLETE.json"
C4 = OUT_ROOT / "r2_5_c4_whole_patient_ci_20261008"
TCM = OUT_ROOT / "r3_5_tumour_correspondence_map_20261111" / "per_pair.csv.gz"
SETTINGS = {"l1": "A|l1|mean", "cosine": "A|cosine|mean"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def c4_module():
    spec = importlib.util.spec_from_file_location("c4ci", REPO / "scripts" / "r2_5_c4_whole_patient_ci.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_ROOT / "r3_5_cosine_heatmap_c4_44_20261112")
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"--out exists: {a.out}")
    M = c4_module()
    s2c = json.loads(S2_COMPLETE.read_text())
    assert s2c["outputs"]["per_pair_A.csv.gz"] == sha256(PP), "per_pair_A.csv.gz differs from its COMPLETE record"
    c4j = json.loads((C4 / "c4_whole_patient_ci.json").read_text())
    c4r = pd.read_csv(C4 / "per_reference_variant_A.csv", dtype={"reference": str}).set_index("reference")
    refs = list(c4r.index)
    pp = pd.read_csv(PP, dtype={"reference": str, "query": str, "ref_patient": str})
    pp = pp[(pp.design == "P2") & pp.setting.isin(SETTINGS.values()) & pp.reference.isin(refs)].copy()
    pp["pointing_A"] = pp.pointing_A.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    gates = {}
    per = {}
    for dist, setting in SETTINGS.items():
        d = pp[pp.setting == setting]
        g = d.groupby(["arm", "reference"]).agg(auroc=("auroc_A", "mean"), pointing=("pointing_A", "mean"),
                                                 ap=("ap_A", "mean"), share=("tumour_fraction_A", "mean"),
                                                 n=("auroc_A", "size"), patient=("ref_patient", "first"))
        gates[f"G3_{dist}_ten_queries_per_reference_arm"] = bool((g.n == 10).all() and len(d) == 880
                                                                  and d.auroc_A.notna().all())
        per[dist] = {arm: g.loc[arm].loc[refs] for arm in ("same_group", "other_group")}
    s1, o1 = per["l1"]["same_group"], per["l1"]["other_group"]
    gates["G3_one_patient_per_reference"] = bool((s1.patient.values == c4r.ref_patient.values).all()
                                                 and c4r.ref_patient.nunique() == 44)
    dev = max(np.abs(s1.auroc.values - c4r.auroc_same.values).max(), np.abs(o1.auroc.values - c4r.auroc_other.values).max(),
              np.abs(s1.pointing.values - c4r.pointing_same.values).max(), np.abs(o1.pointing.values - c4r.pointing_other.values).max(),
              np.abs(s1.ap.values - c4r.ap_same.values).max(), np.abs(o1.ap.values - c4r.ap_other.values).max(),
              np.abs(s1.share.values - c4r.tumour_share.values).max())
    gates["G1_l1_per_reference_equals_c4"] = bool(dev < 1e-9)

    patients, mult = M.make_patient_draws(list(c4r.ref_patient), count=5000, seed=123)
    W = mult[:, pd.Index(patients).get_indexer(list(c4r.ref_patient))].astype(float)
    den = W.sum(1)

    def summarise(v):
        v = np.asarray(v, float)
        reps = (W @ v) / den
        lo, hi = M.ci(reps)
        return {"point": float(v.mean()), "ci95_lo": lo, "ci95_hi": hi, "n_references": int(len(v))}

    est = {}
    for dist in SETTINGS:
        s, o = per[dist]["same_group"], per[dist]["other_group"]
        for m in ("auroc", "pointing", "ap"):
            est[f"{dist}_{m}_same"] = s[m].values
            est[f"{dist}_{m}_other"] = o[m].values
            est[f"{dist}_{m}_diff"] = s[m].values - o[m].values
    for m in ("auroc", "pointing", "ap"):
        for arm in ("same", "other", "diff"):
            est[f"cosine_minus_l1_{m}_{arm}"] = est[f"cosine_{m}_{arm}"] - est[f"l1_{m}_{arm}"]
    est["tumour_share"] = s1.share.values
    est["cosine_pointing_same_minus_share"] = est["cosine_pointing_same"] - s1.share.values
    res = {k: summarise(v) for k, v in est.items()}

    dev2 = 0.0
    for mine, theirs in (("l1_auroc_same", "auroc_same"), ("l1_auroc_other", "auroc_other"), ("l1_auroc_diff", "auroc_diff"),
                         ("l1_pointing_same", "pointing_same"), ("l1_pointing_other", "pointing_other"),
                         ("l1_pointing_diff", "pointing_diff"), ("l1_ap_same", "ap_same"), ("tumour_share", "tumour_share")):
        for f in ("point", "ci95_lo", "ci95_hi"):
            dev2 = max(dev2, abs(res[mine][f] - c4j["results"][theirs][f]))
    gates["G2_l1_points_and_intervals_equal_c4"] = bool(dev2 < 1e-9)

    t = pd.read_csv(TCM, dtype={"reference": str, "query": str})
    t = t[(t.design == "P2") & t.in_manuscript_44]
    cos = pp[pp.setting == SETTINGS["cosine"]].set_index("pair_id").auroc_A
    j = t.set_index("pair_id").auroc_roi_cos
    common = cos.index.intersection(j.index)
    dev4 = float(np.abs(cos.loc[common] - j.loc[common]).max())
    gates["G4_cosine_per_pair_equals_tcm_run"] = bool(len(common) == 880 and dev4 < 1e-9)
    passed = all(gates.values())
    print(json.dumps(gates, indent=1), f"\nG1 max|d| {dev:.1e}; G2 max|d| {dev2:.1e}; G4 max|d| {dev4:.1e} on {len(common)} pairs")
    if not passed:
        return 1

    a.out.mkdir(parents=True)
    rows = []
    for dist in SETTINGS:
        s, o = per[dist]["same_group"], per[dist]["other_group"]
        for r in refs:
            rows.append(dict(reference=r, ref_group=c4r.loc[r, "ref_group"], distance=dist,
                             auroc_same=s.loc[r, "auroc"], auroc_other=o.loc[r, "auroc"],
                             pointing_same=s.loc[r, "pointing"], pointing_other=o.loc[r, "pointing"],
                             ap_same=s.loc[r, "ap"], ap_other=o.loc[r, "ap"], tumour_share=s.loc[r, "share"]))
    pd.DataFrame(rows).to_csv(a.out / "per_reference.csv", index=False, float_format="%.12g")
    summ = pd.DataFrame([{"estimand": k, **v} for k, v in res.items()])
    summ.to_csv(a.out / "summary.csv", index=False, float_format="%.12g")
    rec = {"script": str(Path(__file__).relative_to(REPO)), "script_sha256": sha256(Path(__file__)),
           "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "inputs": {str(p): sha256(p) for p in (PP, S2_COMPLETE, C4 / "c4_whole_patient_ci.json",
                                                  C4 / "per_reference_variant_A.csv", TCM)},
           "config": {"B": 5000, "seed": 123, "generator": "numpy PCG64", "draws": "make_patient_draws of "
                      "scripts/r2_5_c4_whole_patient_ci.py (identical to C4)", "n_references": 44, "n_patients": 44,
                      "settings": SETTINGS, "variant": "A (labelled patches only; 1 tumour, 2-5 non-tumour, 0 excluded)",
                      "ci": "percentile 2.5/97.5 (numpy linear), nominal 95%, unadjusted for multiplicity"},
           "gates": gates, "gate_deviations": {"G1": dev, "G2": dev2, "G4": dev4, "G4_pairs": int(len(common))},
           "results": res, "counts": {"n_refs_cosine_gt_l1_same": int((est["cosine_minus_l1_auroc_same"] > 0).sum()),
                                      "n_refs_cosine_same_gt_other": int((est["cosine_auroc_diff"] > 0).sum())},
           "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]}
    (a.out / "result.json").write_text(json.dumps(rec, indent=1) + "\n")
    with open(a.out / "SHA256SUMS.txt", "w") as f:
        for n in ("per_reference.csv", "summary.csv", "result.json"):
            f.write(f"{sha256(a.out / n)}  {n}\n")
    (a.out / "COMPLETE").write_text(rec["finished"] + "\n")
    for k in ("cosine_auroc_same", "cosine_auroc_other", "cosine_auroc_diff", "cosine_minus_l1_auroc_same",
              "cosine_minus_l1_auroc_diff", "cosine_pointing_same", "cosine_pointing_same_minus_share", "l1_auroc_same"):
        r = res[k]
        print(f"{k:38s} {r['point']:+.4f} [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}]")
    print("counts", rec["counts"], "->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
