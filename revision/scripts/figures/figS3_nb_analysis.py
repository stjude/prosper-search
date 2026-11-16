"""Supplementary Fig. S3 (rebuilt): CCDI-MCI neuroblastoma queries, revised
FFPE metadata-scope population (89 queries), hit@5 outcomes.

[internal record] block R1.2-T4 (SFig. S3 part). Sources
(`$PROSPER_OUT/r1_2_scope_correction_20260930/`):
  main_nb_descriptive_factors.csv            panels a, c, d (hit@5 by factor)
  whole_slide_ot_ffpe_scope_clear_per_query.csv + decision_table_765.csv
                                              panel b (age by outcome), panel e
  main_nb_raw_diagnosis_closest_vote5_confusion.csv  panel e cross-check
  decision_table_765.csv                     panel f (eligibility flow)

Panel e uses the explicitly named rule: closest-rank plurality of the five
nearest neighbours (pred_closest_k5). No p-values; no "182 / 35.2 %".
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GREY, HIT_COLOR, PREC_COLOR, prosper_out, save, style  # noqa: E402

SRC = prosper_out() / "r1_2_scope_correction_20260930"
FIGURE = "figS3_nb_analysis"
AGE_ORDER = ["<1 yr", "1-2 yr", "2-5 yr", "5-10 yr", "10-18 yr", ">=18 yr"]


def first_days(v):
    try:
        return float(str(v).split(";")[0])
    except ValueError:
        return np.nan


def load():
    f = pd.read_csv(SRC / "main_nb_descriptive_factors.csv")
    d = pd.read_csv(SRC / "decision_table_765.csv")
    q = pd.read_csv(SRC / "whole_slide_ot_ffpe_scope_clear_per_query.csv")
    conf = pd.read_csv(SRC / "main_nb_raw_diagnosis_closest_vote5_confusion.csv")
    nb = d[d.tumor_group == "NB"].copy()
    main = nb[nb.ffpe_scope_clear]
    m = q.merge(main[["slide_id", "Age at Diagnosis (days)"]], left_on="query", right_on="slide_id")
    assert len(m) == 89 and m.true_label.eq("NB").all()
    m["age_years"] = m["Age at Diagnosis (days)"].map(first_days) / 365.25
    flow = {
        "NB queries among the 765 CCDI-MCI participants with a diagnosis in the COMET vocabulary": len(nb),
        "verified FFPE": int((nb.prep_dicom == "FFPE").sum()),
        "FFPE, metadata scope clear (main analysis)": int(main.shape[0]),
    }
    withheld = nb[(nb.prep_dicom == "FFPE") & ~nb.ffpe_scope_clear].scope_reason.value_counts().to_dict()
    assert flow["NB queries among the 765 CCDI-MCI participants with a diagnosis in the COMET vocabulary"] == 178 and flow["verified FFPE"] == 103 and flow["FFPE, metadata scope clear (main analysis)"] == 89
    assert withheld.get("known_foxr2", 0) == 4 and sum(withheld.values()) == 14
    # panel e cross-check against the confusion table
    miss = m[m.pred_closest_k5 != "NB"].pred_closest_k5.value_counts()
    conf_miss = conf[conf.pred_closest_k5 != "NB"].groupby("pred_closest_k5").n.sum()
    assert miss.to_dict() == conf_miss.to_dict(), (miss.to_dict(), conf_miss.to_dict())
    return f, m, flow, withheld, miss


def bar_rate(ax, rows, order, title, xlabel_rot=0):
    rows = rows.set_index("source_value").reindex([o for o in order if o in rows.source_value.values])
    x = np.arange(len(rows))
    ax.bar(x, rows.hit5_rate * 100, color=HIT_COLOR, width=0.7)
    for xi, (n, h) in enumerate(zip(rows.n, rows.hit5_count)):
        ax.text(xi, rows.hit5_rate.iloc[xi] * 100 + 1.5, f"{int(h)}/{int(n)}", ha="center", fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(rows.index, rotation=xlabel_rot, ha="right" if xlabel_rot else "center")
    ax.set_ylim(0, 112); ax.set_ylabel("hit@5 (%)"); ax.set_title(title)


def main():
    style()
    f, m, flow, withheld, miss = load()
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.4))
    fig.subplots_adjust(hspace=0.75, wspace=0.4, left=0.07, right=0.98, top=0.92, bottom=0.14)
    a, b, c, d, e, fx = axes.ravel()
    # a: age bucket
    bar_rate(a, f[f.factor == "age_bucket"], AGE_ORDER, "a  hit@5 by age at diagnosis")
    # b: age distribution by outcome
    hit = m[m.hit_k5 == 1].age_years.dropna(); mis = m[m.hit_k5 == 0].age_years.dropna()
    bins = np.arange(0, max(m.age_years.max(), 18) + 1, 1)
    b.hist(hit, bins=bins, color=HIT_COLOR, alpha=0.7, label=f"hit@5 (n = {len(hit)})")
    b.hist(mis, bins=bins, color=PREC_COLOR, alpha=0.7, label=f"no hit@5 (n = {len(mis)})")
    b.set_xlabel("age at diagnosis (years)"); b.set_ylabel("queries"); b.set_title("b  age by hit@5 outcome")
    b.legend(frameon=False)
    # c: anatomic region heuristic
    reg = f[f.factor == "source_site_region_heuristic"].sort_values("n", ascending=False)
    bar_rate(c, reg, reg.source_value.tolist(), "c  hit@5 by recorded site region\n(grouping of site text, not adjudicated)", 35)
    # d: source diagnosis
    dg = f[f.factor == "Diagnosis"].sort_values("n", ascending=False)
    bar_rate(d, dg, dg.source_value.tolist(), "d  hit@5 by source diagnosis label", 20)
    # e: misclassification targets (closest-rank plurality, k = 5)
    miss = miss.sort_values(ascending=False)
    e.bar(np.arange(len(miss)), miss.values, color=PREC_COLOR, width=0.7)
    for xi, v in enumerate(miss.values):
        e.text(xi, v + 0.15, str(int(v)), ha="center", fontsize=6)
    e.set_xticks(np.arange(len(miss))); e.set_xticklabels(miss.index)
    e.set_ylabel("queries"); e.set_title(f"e  predicted group when not NB\n(closest-rank plurality, k = 5; {int(miss.sum())} of 89)")
    e.set_ylim(0, max(miss.values) + 1.5)
    # f: eligibility flow
    fx.axis("off")
    y = 0.95
    for lab, n in flow.items():
        fx.text(0.02, y, f"{n}", fontsize=10, fontweight="bold", va="top")
        fx.text(0.2, y, lab, fontsize=6.5, va="top", wrap=True)
        y -= 0.3
        if lab != list(flow)[-1]:
            fx.annotate("", xy=(0.08, y + 0.08), xytext=(0.08, y + 0.2), arrowprops=dict(arrowstyle="->", lw=0.8))
    fx.text(0.02, 0.05, "14 FFPE NB queries withheld: 4 known CNS neuroblastoma, FOXR2-activated;\n"
            f"10 unresolved metadata ({withheld.get('site_only', 0)} site-only, {withheld.get('compound_support', 0)} compound-diagnosis records).", fontsize=5.8, va="bottom", color="black")
    fx.set_title("f  NB query eligibility flow")
    note = ("89 FFPE metadata-scope NB queries (UNI2 + exact Optimal Transport, whole_slide_ot, fixed COMET library). "
            "Outcomes are hit@5 (a correct-group slide among the five nearest); panel e uses the closest-rank plurality "
            "prediction. Descriptive; no inferential statistics. Age buckets are the source table's buckets.")
    save(fig, FIGURE, "Figure_S3_NB_analysis_revised",
         [SRC / "main_nb_descriptive_factors.csv", SRC / "decision_table_765.csv",
          SRC / "whole_slide_ot_ffpe_scope_clear_per_query.csv", SRC / "main_nb_raw_diagnosis_closest_vote5_confusion.csv"],
         note, {"source_data_factors": f, "source_data_nb89_queries": m[["query", "true_label", "hit_k5", "pred_closest_k5", "age_years"]]})


if __name__ == "__main__":
    main()
