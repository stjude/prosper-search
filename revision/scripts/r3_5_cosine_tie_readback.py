#!/usr/bin/env python3
"""R3.5 revision: tie census of the cosine ROI heat maps on the 44-reference C4 design (S25 block C).

Plan record: [internal record], "Fig. 6B revert and cosine tie census,
pre-run record, 2026-11-12". decision 2026-11-12: "Yes, run it and add the cosine counts to block C".
Read-only on every input; writes only --out.

Rebuilds the 880 C4 pairs (s0_pairs pairs_P2.csv on the 44 C4 references) as the R3.5-L1 s1 stage scored
family A: query vector = mean of the listed query_tiles rows of the query localization_cli.Slide.X; map =
r3_5_l1_maps.roi_map(reference Slide, q, distance, "mean"); variant-A scores by r3_5_l1_maps.score_continuous
(min_neg = 1). The tie functions (tie_metrics, boot) are imported unchanged from the revision L1 readback.

Gates (all required before the census is read):
  F1  cosine per pair equals s2 run_003 per_pair_A (A|cosine|mean, design P2): AUROC, AP, tumour fraction
      <= 1e-9; pointing and annotated count 0 mismatches; 880 pairs
  F2  cosine same / other / difference points of AUROC, AP and pointing equal
      r3_5_cosine_heatmap_c4_44_20261112/result.json (<= 1e-9)
  F3  positive control: the same route with L1 reproduces the revision per-pair tied-at-maximum count,
      any/all pointing bounds, stored-convention AP and grouped AP on all 880 pairs
Outcome rule as revision: a stored aggregate stands unless threshold-grouped AP or tie-fraction pointing moves
it by >= 0.005 (AP) or >= 0.01 (pointing).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ["PROSPER_OUT"])
ROOT = OUT / "r3_5_localisation_maps_20261017"
S0 = ROOT / "s0_pairs" / "run_001"
S2 = ROOT / "s2" / "run_003"
C4 = OUT / "r2_5_c4_whole_patient_ci_20261008"
COS = OUT / "r3_5_cosine_heatmap_c4_44_20261112"
TIE029 = OUT / "localization_tie_readback_20261110"
L1_SCRIPT = REPO / "scripts" / "r3_5_l1_maps.py"
TIE_SCRIPT = TIE029 / "readback_localization_ties.py"
CI_SCRIPT = REPO / "scripts" / "r2_5_c4_whole_patient_ci.py"
DISTANCES = ("cosine", "l1")
TOL = 1e-9
AP_THR, PT_THR = 0.005, 0.01
INPUTS: dict = {}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bind(p: Path, role: str) -> None:
    INPUTS[str(p)] = {"role": role, "sha256": sha256(p)}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def as_float(s: pd.Series) -> pd.Series:
    return s.map(lambda v: float(v in (True, "True")) if isinstance(v, (bool, np.bool_, str)) else float(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT / "localization_tie_readback_cosine_20261112")
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"--out exists: {a.out}")
    t0 = time.time()
    s0c = json.loads((S0 / "COMPLETE.json").read_text())
    meta_pkl = Path(s0c["s0_bind"]) / "meta_tiles.pkl"
    for p, r in ((S0 / "pairs_P2.csv", "C4 pair construction (P2)"), (S0 / "COMPLETE.json", "s0_pairs record"),
                 (meta_pkl, "mosaic metadata (s0_bind)"), (S2 / "per_pair_A.csv.gz", "stored family-A values"),
                 (S2 / "COMPLETE.json", "s2 record"), (C4 / "per_reference_variant_A.csv", "C4 44 references"),
                 (COS / "result.json", "cosine C4-44 aggregates"), (TIE029 / "per_pair.csv", "pass-029 L1 census"),
                 (L1_SCRIPT, "map/score functions (imported)"), (TIE_SCRIPT, "tie functions (imported)"),
                 (CI_SCRIPT, "whole-patient draws (imported)")):
        bind(p, r)
    s2c = json.loads((S2 / "COMPLETE.json").read_text())
    assert s2c["outputs"]["per_pair_A.csv.gz"] == INPUTS[str(S2 / "per_pair_A.csv.gz")]["sha256"]
    L1 = load_module(L1_SCRIPT, "r3_5_l1_maps")
    TIE = load_module(TIE_SCRIPT, "readback_localization_ties")
    CI = load_module(CI_SCRIPT, "r2_5_c4_whole_patient_ci")
    LC, _ = L1.import_prosper()

    c4r = pd.read_csv(C4 / "per_reference_variant_A.csv", dtype={"reference": str, "ref_patient": str})
    refs = list(c4r.reference)
    p2 = pd.read_csv(S0 / "pairs_P2.csv", dtype=str)
    p2 = p2[p2.reference.isin(set(refs))].reset_index(drop=True)
    pp = pd.read_csv(S2 / "per_pair_A.csv.gz", dtype={"reference": str, "query": str, "ref_patient": str},
                     float_precision="round_trip")
    pp = pp[(pp.design == "P2") & pp.reference.isin(set(refs))]
    stored = {d: pp[pp.setting == f"A|{d}|mean"].set_index("pair_id") for d in DISTANCES}
    assert len(p2) == 880 and (p2.arm == "same_group").sum() == 440
    assert all(len(v) == 880 and v.index.is_unique for v in stored.values())
    # C4 draws with replacement repeat a few (reference, query, tiles) pairs, so align by row order, not by key
    old = pd.read_csv(TIE029 / "per_pair.csv", dtype={"reference": str, "query_slide": str, "query_tiles": str},
                      float_precision="round_trip")
    assert len(old) == 880 and all((old[c].values == p2[d].values).all() for c, d in (
        ("reference", "reference"), ("arm", "arm"), ("query_slide", "query"), ("query_tiles", "query_tiles")))
    old = old.set_index(p2.pair_id.values)

    meta = L1.meta_cache_load(meta_pkl)
    slides: dict = {}

    def slide(s):
        if s not in slides:
            slides[s] = LC.Slide(s, L1.FEATURES, meta[s], L1.LABELS)
        return slides[s]

    keys = ("auroc_A", "ap_A", "pointing_A", "tumour_fraction_A", "n_annotated")
    rows = []
    for r in L1.numeric_sorted(set(p2.reference)):
        ref = slide(r)
        y_t = ref.patch_label == LC.TUMOR
        ann = y_t | np.isin(ref.patch_label, LC.NONTUMOR)
        for _, row in p2[p2.reference == r].iterrows():
            q = slide(row.query).X[L1.parse_idx(row.query_tiles)].mean(axis=0)
            rec = {"pair_id": row.pair_id, "reference": r, "ref_patient": row.ref_patient, "arm": row.arm,
                   "query": row.query, "query_tiles": row.query_tiles}
            for dist in DISTANCES:
                h = L1.roi_map(ref, q, dist, "mean")
                sc = L1.score_continuous(h, ref.patch_label, LC, min_neg=1)
                st = stored[dist].loc[row.pair_id]
                rec.update({f"{dist}_stored_{k}": st[k] for k in keys})
                rec.update({f"{dist}_re_{k}": sc.get(k, np.nan) for k in keys})
                rec.update({f"{dist}_{k}": v for k, v in TIE.tie_metrics(h[ann], y_t[ann].astype(int)).items()})
            rows.append(rec)
        print(f"[ties] {r} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    for dist in DISTANCES:
        for k in ("re", "stored"):
            df[f"{dist}_{k}_pointing_num"] = as_float(df[f"{dist}_{k}_pointing_A"])

    gates, dev = {}, {}
    for dist in DISTANCES:
        d = {f"{k}_max_abs_diff": float((df[f"{dist}_re_{k}"].astype(float) - df[f"{dist}_stored_{k}"].astype(float))
                                        .abs().max()) for k in ("auroc_A", "ap_A", "tumour_fraction_A")}
        d["pointing_A_mismatches"] = int((df[f"{dist}_re_pointing_num"] != df[f"{dist}_stored_pointing_num"]).sum())
        d["n_annotated_mismatches"] = int((df[f"{dist}_re_n_annotated"].astype(float)
                                           != df[f"{dist}_stored_n_annotated"].astype(float)).sum())
        dev[f"F1_{dist}"] = d
        gates[f"F1_{dist}_per_pair_equals_s2"] = bool(len(df) == 880 and all(
            v <= TOL for k, v in d.items() if k.endswith("diff")) and d["pointing_A_mismatches"] == 0
            and d["n_annotated_mismatches"] == 0)

    def per_ref(col):
        g = df.groupby(["arm", "reference"])[col].mean()
        return g.loc["same_group"].reindex(refs).values, g.loc["other_group"].reindex(refs).values

    share = per_ref("cosine_re_tumour_fraction_A")[0]

    def estimands(apcol, ptcol):
        aps, apo = per_ref(apcol)
        pts, pto = per_ref(ptcol)
        return {"ap_same": aps, "ap_other": apo, "ap_diff": aps - apo, "pointing_same": pts,
                "pointing_other": pto, "pointing_diff": pts - pto, "pointing_minus_share": pts - share}

    cres = json.loads((COS / "result.json").read_text())["results"]
    f2 = {}
    for m, col in (("auroc", "cosine_re_auroc_A"), ("ap", "cosine_re_ap_A"), ("pointing", "cosine_re_pointing_num")):
        s, o = per_ref(col)
        for arm, v in (("same", s), ("other", o), ("diff", s - o)):
            f2[f"cosine_{m}_{arm}"] = float(abs(np.mean(v) - cres[f"cosine_{m}_{arm}"]["point"]))
    dev["F2"] = f2
    gates["F2_cosine_points_equal_c4_44_result"] = bool(all(v <= TOL for v in f2.values()))

    j = df.set_index("pair_id")
    common = j.index.intersection(old.index)
    jo, oo = j.loc[common], old.loc[common]
    f3 = {"n_pairs_joined": int(len(common)),
          "n_tied_at_max_mismatches": int((jo.l1_n_tied_at_max != oo.n_tied_at_max).sum()),
          "pointing_any_mismatches": int((jo.l1_pointing_any != oo.pointing_any).sum()),
          "pointing_all_mismatches": int((jo.l1_pointing_all != oo.pointing_all).sum()),
          "ap_stored_convention_max_abs_diff": float((jo.l1_re_ap_A - oo.re_ap_A).abs().max()),
          "ap_grouped_max_abs_diff": float((jo.l1_ap_grouped - oo.ap_grouped).abs().max())}
    dev["F3"] = f3
    gates["F3_l1_route_reproduces_pass029"] = bool(
        f3["n_pairs_joined"] == 880 and f3["n_tied_at_max_mismatches"] == 0 and f3["pointing_any_mismatches"] == 0
        and f3["pointing_all_mismatches"] == 0 and f3["ap_stored_convention_max_abs_diff"] <= TOL
        and f3["ap_grouped_max_abs_diff"] <= TOL)
    passed = all(gates.values())
    print(json.dumps({"gates": gates, "deviations": dev}, indent=1), flush=True)

    a.out.mkdir(parents=True)
    df.to_csv(a.out / "per_pair.csv", index=False, float_format="%.17g")
    res = {"script": str(Path(__file__).relative_to(REPO)), "script_sha256": sha256(Path(__file__)),
           "host": platform.node(), "gates": gates, "gate_deviations": dev, "passed": passed}
    if passed:
        census, aggs, verdict = {}, {}, {}
        for dist in DISTANCES:
            p = f"{dist}_"
            census[dist] = {
                "annotated_patches_per_pair_median": float(df[p + "n_scored"].median()),
                "distinct_scores_per_pair_median": float(df[p + "n_distinct_scores"].median()),
                "share_in_tied_groups_pooled": float((df[p + "share_in_tied_groups"] * df[p + "n_scored"]).sum()
                                                     / df[p + "n_scored"].sum()),
                "share_in_mixed_label_tied_groups_pooled": float(
                    (df[p + "share_in_mixed_tied_groups"] * df[p + "n_scored"]).sum() / df[p + "n_scored"].sum()),
                "share_in_mixed_label_tied_groups_max_pair": float(df[p + "share_in_mixed_tied_groups"].max()),
                "pairs_with_any_mixed_tied_group": int((df[p + "share_in_mixed_tied_groups"] > 0).sum()),
                "pairs_with_tie_at_max": int((df[p + "n_tied_at_max"] > 1).sum()),
                "pairs_with_mixed_label_tie_at_max": int((df[p + "pointing_any"] != df[p + "pointing_all"]).sum()),
                "max_pair_abs_diff_ap_grouped_minus_stored": float((df[p + "ap_grouped"] - df[p + "re_ap_A"]).abs().max()),
                "pairs_where_tiefrac_pointing_differs_from_stored": int(
                    ((df[p + "pointing_tiefrac"] - df[p + "re_pointing_num"]).abs() > 0).sum()),
            }
        conv = {"stored (stable-sort AP, first-argmax pointing)": ("cosine_re_ap_A", "cosine_re_pointing_num"),
                "threshold-grouped AP / tie-fraction pointing (replacement conventions)":
                    ("cosine_ap_grouped", "cosine_pointing_tiefrac"),
                "optimistic tie order AP / pointing any-tied-is-tumour": ("cosine_ap_optimistic", "cosine_pointing_any"),
                "pessimistic tie order AP / pointing all-tied-are-tumour": ("cosine_ap_pessimistic", "cosine_pointing_all")}
        aggs = {n: {k: float(np.mean(v)) for k, v in estimands(*c).items()} for n, c in conv.items()}
        names = list(conv)
        diffs = {n: {k: aggs[n][k] - aggs[names[0]][k] for k in aggs[names[0]]} for n in names[1:]}
        for k in aggs[names[0]]:
            thr = AP_THR if k.startswith("ap") else PT_THR
            d = abs(diffs[names[1]][k])
            verdict[k] = {"abs_diff_replacement_convention": d, "threshold": thr, "replacement_triggered": bool(d >= thr),
                          "max_abs_diff_tie_order_bounds": max(abs(diffs[n][k]) for n in names[2:])}
        res.update({"tie_census": census, "aggregates_cosine": aggs, "differences_vs_stored_cosine": diffs,
                    "outcome_rule_cosine": verdict})
        if any(v["replacement_triggered"] for v in verdict.values()):
            rb = TIE.boot(estimands(*conv[names[1]]), list(c4r.ref_patient), CI.make_patient_draws)
            res["replacement_cells_cosine"] = {k: rb[k] for k in verdict if verdict[k]["replacement_triggered"]}
        print(json.dumps({"tie_census": census, "outcome_rule_cosine": verdict}, indent=1), flush=True)
    res.update({"inputs": INPUTS, "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "elapsed_s": round(time.time() - t0, 1),
                "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                             "sklearn": __import__("sklearn").__version__}})
    (a.out / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    with open(a.out / "SHA256SUMS.txt", "w") as f:
        for n in ("per_pair.csv", "results.json"):
            f.write(f"{sha256(a.out / n)}  {n}\n")
    if passed:
        (a.out / "COMPLETE").write_text(res["finished"] + "\n")
    print("passed" if passed else "FAILED", a.out, f"{res['elapsed_s']}s")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
