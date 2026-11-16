"""Distinct-route check for Fig. 3C (per-group filtering effect).

Recomputes every group-level point estimate drawn in Fig. 3C from the per-query
table of the TF-2 S6 primary evaluation (query_metrics.csv.gz), independently of
contrasts.csv: for each encoder / method / arm at the 100 % budget and k = 5,
the change in hit@5 and precision@5 = mean over the queries returned in both
the filtered and the all-tissue arm (population both_returned) of the filtered
minus the all-tissue conditional metric, within each diagnostic group and for
the macro average (mean of the 14 group means). Bootstrap intervals are not
recomputed (the replicate draws are not re-seeded here); only the point
estimates and the per-group query counts are checked. Writes
panelC_group_check.json beside the figure.
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
TOL = 1e-9


def main():
    q = pd.read_csv(QUERY_METRICS, low_memory=False)
    q = q[(q.k == 5) & (q.status == "returned")]
    base = q[q.arm == "all_tissue"].set_index(["encoder", "method", "query"])[
        ["conditional_hit", "conditional_precision", "diagnostic_group"]]
    g = f3.load_group_contrasts()
    d = f3.load_contrasts()
    d = d[d.budget == 100]
    out = []
    max_dev = 0.0
    for enc, meth, arm in f3.SETTINGS:
        filt = q[(q.encoder == enc) & (q.method == meth) & (q.arm == arm) & (q.budget == 100)]
        filt = filt.set_index(["encoder", "method", "query"])
        j = filt.join(base, rsuffix="_all", how="inner")
        assert (j.diagnostic_group == j.diagnostic_group_all).all()
        j = j.assign(d_hit=j.conditional_hit - j.conditional_hit_all,
                     d_prec=j.conditional_precision - j.conditional_precision_all)
        per = j.groupby("diagnostic_group").agg(hit=("d_hit", "mean"), precision=("d_prec", "mean"), n=("d_hit", "size"))
        assert len(per) == 14
        for grp, r in per.iterrows():
            ref = g[(g.encoder == enc) & (g.method == meth) & (g.arm == arm) & (g.group == grp)].iloc[0]
            dev = max(abs(r.hit - ref.hit_point), abs(r.precision - ref.precision_point))
            max_dev = max(max_dev, dev)
            assert dev < TOL and int(r.n) == ref.hit_n_queries == ref.precision_n_queries, (enc, meth, arm, grp, dev, r.n)
            out.append(dict(encoder=enc, method=meth, arm=arm, group=grp, n=int(r.n), hit=r.hit, precision=r.precision,
                            hit_contrasts=ref.hit_point, precision_contrasts=ref.precision_point))
        macro = per[["hit", "precision"]].mean()
        mref = d[(d.encoder == enc) & (d.method == meth) & (d.arm == arm)].iloc[0]
        mdev = max(abs(macro.hit - mref.hit_point), abs(macro.precision - mref.precision_point))
        max_dev = max(max_dev, mdev)
        assert mdev < TOL and int(per.n.sum()) == mref.hit_n_queries, (enc, meth, arm, mdev, per.n.sum())
    res = dict(status="PASS", cells_checked=len(out), macro_settings_checked=len(f3.SETTINGS), tolerance=TOL,
               max_abs_deviation=max_dev, query_metrics=str(QUERY_METRICS), query_metrics_sha256=sha256(QUERY_METRICS),
               population="status == returned in both arms, k = 5, budget 100",
               limits="point estimates and query counts only; bootstrap intervals not recomputed")
    dst = out_dir(f3.FIGURE) / "panelC_group_check.json"
    dst.write_text(json.dumps(res, indent=1) + "\n")
    pd.DataFrame(out).to_csv(dst.with_suffix(".csv"), index=False)
    with open(out_dir(f3.FIGURE) / "SHA256SUMS.txt", "a") as fh:
        for p in (dst, dst.with_suffix(".csv")):
            fh.write(f"{sha256(p)}  {p.name}\n")
    print(f"PASS {len(out)} group cells + {len(f3.SETTINGS)} macro settings; max |dev| = {max_dev:.2e}")


if __name__ == "__main__":
    main()
