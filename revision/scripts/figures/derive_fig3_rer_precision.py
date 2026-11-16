"""Relative error reduction rebased on hit@5 / precision@5 (author-requested comparison, 2026-10-21 ≈16:30).

Derived endpoint, not pre-registered: a transformation of the TF-2 S6 primary endpoints with no new data and no new
random draws. For every setting (encoder, method, arm; 100 % budget, k = 5) and every diagnostic group plus the
macro average (mean of the 14 group means), on the queries returned in both arms (population both_returned):

    error      = 1 - metric                      (metric = conditional precision@5 or hit@5, group mean)
    RER        = (error_all - error_filtered) / error_all = (metric_filt - metric_all) / (1 - metric_all)

Uncertainty: the S6 shared whole-patient draw matrix (s6/draws_001/patient_draws.npy, 5,000 x 1,800 multiplicities;
query weight = its patient's multiplicity), percentile 2.5 / 97.5 (linear). A replicate is undefined when
error_all = 0; the interval is withheld when more than 1 % of replicates are undefined (the rule of the frozen
F1-based RER). Checks: the point change (metric_filt - metric_all) must equal contrasts.csv to 1e-9 for every group
and macro cell, and the replicate percentile interval of that change is compared with the stored S6 interval
(reported as max deviation; it validates the weighting against the frozen evaluation).

Writes rer_rebased_contrasts.csv (+ .json, SHA256SUMS.txt) under figures_20261021/fig3_rer_rebased_derivation/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig3_tumor_filtering as f3  # noqa: E402
from _common import out_dir, sha256  # noqa: E402

QUERY_METRICS = f3.TF2 / "s6/evaluation_primary_001/query_metrics.csv.gz"
DRAWS = f3.TF2 / "s6/draws_001/patient_draws.npy"
DRAW_PATIENTS = f3.TF2 / "s6/draws_001/patient_draws_patients.txt"
OUT = "fig3_rer_rebased_derivation"
UNDEFINED_MAX_FRACTION = 0.01
TOL = 1e-9
ENDPOINTS = {"precision": "conditional_precision", "hit": "conditional_hit"}


def percentile(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5, method="linear")), float(np.percentile(x, 97.5, method="linear"))


def main():
    draws = np.load(DRAWS)
    patients = DRAW_PATIENTS.read_text().split()
    assert draws.shape == (5000, len(patients)) == (5000, 1800) and (draws.sum(1) == 1800).all()
    ppos = {p: i for i, p in enumerate(patients)}

    q = pd.read_csv(QUERY_METRICS, low_memory=False)
    q = q[(q.k == 5) & (q.status == "returned")]
    base = q[q.arm == "all_tissue"].set_index(["encoder", "method", "query"])[
        ["conditional_hit", "conditional_precision", "diagnostic_group", "patient_id"]]
    c = pd.read_csv(f3.CONTRASTS)
    c = c[(c.family == "a_filtered_minus_all_tissue") & (c.k == 5) & (c.budget == 100) & (c.population == "both_returned")]

    rows, dev_point, dev_ci = [], 0.0, 0.0
    for enc, meth, arm in f3.SETTINGS:
        filt = q[(q.encoder == enc) & (q.method == meth) & (q.arm == arm) & (q.budget == 100)]
        j = filt.set_index(["encoder", "method", "query"]).join(base, rsuffix="_all", how="inner")
        assert (j.diagnostic_group == j.diagnostic_group_all).all() and (j.patient_id == j.patient_id_all).all()
        W = draws[:, [ppos[p] for p in j.patient_id]]  # 5000 x n_queries multiplicities
        groups = sorted(j.diagnostic_group.unique())
        assert len(groups) == 14
        for ep, col in ENDPOINTS.items():
            m_filt = j[col].to_numpy(float)
            m_all = j[f"{col}_all"].to_numpy(float)
            P_all_g, P_filt_g, cells = [], [], []
            for grp in groups:
                sel = (j.diagnostic_group == grp).to_numpy()
                Wg = W[:, sel]
                wsum = Wg.sum(1)
                assert (wsum > 0).all(), (enc, meth, arm, grp, "zero-support replicate")
                pa = Wg @ m_all[sel] / wsum
                pf = Wg @ m_filt[sel] / wsum
                P_all_g.append(pa)
                P_filt_g.append(pf)
                cells.append((grp, int(sel.sum()), m_all[sel].mean(), m_filt[sel].mean(), pa, pf))
            macro_all = np.mean(P_all_g, axis=0)
            macro_filt = np.mean(P_filt_g, axis=0)
            cells.append(("MACRO", int(len(j)), np.mean([x[2] for x in cells]), np.mean([x[3] for x in cells]),
                          macro_all, macro_filt))
            for grp, n, pt_all, pt_filt, pa, pf in cells:
                ref = c[(c.encoder == enc) & (c.method == meth) & (c.arm == arm) & (c.endpoint == ep)
                        & ((c.level == "macro") if grp == "MACRO" else ((c.level == "group") & (c.group == grp)))].iloc[0]
                delta = pt_filt - pt_all
                dev_point = max(dev_point, abs(delta - ref.point))
                assert abs(delta - ref.point) < TOL and n == int(ref.n_queries), (enc, meth, arm, ep, grp, delta, ref.point)
                d_lo, d_hi = percentile(pf - pa)
                dev_ci = max(dev_ci, abs(d_lo - ref.ci_low), abs(d_hi - ref.ci_high))
                err_all = 1.0 - pa
                defined = err_all > 0
                n_undef = int((~defined).sum())
                rer_rep = (pf[defined] - pa[defined]) / err_all[defined]
                rer_point = delta / (1.0 - pt_all) if pt_all < 1.0 else np.nan
                if n_undef / len(pa) > UNDEFINED_MAX_FRACTION or np.isnan(rer_point):
                    lo = hi = np.nan
                    status = "not_estimable_undefined_gt_1pct" if not np.isnan(rer_point) else "point_undefined"
                else:
                    lo, hi = percentile(rer_rep)
                    status = "reported"
                rows.append(dict(encoder=enc, method=meth, arm=arm, budget=100, k=5, base_metric=ep,
                                 **{"class": grp}, n_queries=n, metric_all=pt_all, metric_filt=pt_filt,
                                 delta=delta, delta_ci_low=d_lo, delta_ci_high=d_hi,
                                 delta_ci_low_s6=ref.ci_low, delta_ci_high_s6=ref.ci_high,
                                 rer=rer_point, rer_ci_low=lo, rer_ci_high=hi, rer_ci_status=status,
                                 n_undefined_reps=n_undef))
    out = pd.DataFrame(rows)
    assert len(out) == len(f3.SETTINGS) * 15 * 2
    d = out_dir(OUT)
    csv = d / "rer_rebased_contrasts.csv"
    out.to_csv(csv, index=False)
    meta = dict(status="PASS", rows=len(out), settings=len(f3.SETTINGS), endpoints=list(ENDPOINTS),
                population="status == returned in both arms, k = 5, budget 100",
                draws=str(DRAWS), draws_sha256=sha256(DRAWS), query_metrics=str(QUERY_METRICS),
                query_metrics_sha256=sha256(QUERY_METRICS), contrasts=str(f3.CONTRASTS),
                point_tolerance=TOL, max_abs_point_deviation=dev_point,
                max_abs_delta_interval_deviation_vs_s6=dev_ci,
                undefined_max_fraction=UNDEFINED_MAX_FRACTION, percentile="2.5 / 97.5, linear",
                definition="RER = (metric_filt - metric_all) / (1 - metric_all); derived endpoint, not pre-registered",
                not_estimable=out[out.rer_ci_status != "reported"][["encoder", "method", "arm", "base_metric", "class",
                                                                     "n_undefined_reps"]].to_dict("records"))
    js = d / "rer_rebased_contrasts.json"
    js.write_text(json.dumps(meta, indent=1) + "\n")
    with open(d / "SHA256SUMS.txt", "w") as fh:
        for p in (csv, js):
            fh.write(f"{sha256(p)}  {p.name}\n")
    print(f"PASS {len(out)} rows; point dev {dev_point:.1e}; delta-interval dev vs S6 {dev_ci:.2e}; "
          f"not estimable: {len(meta['not_estimable'])}")
    print(out[(out['class'] == 'MACRO') & (out.arm == 'both_filtered')][
        ["encoder", "method", "base_metric", "delta", "rer", "rer_ci_low", "rer_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
