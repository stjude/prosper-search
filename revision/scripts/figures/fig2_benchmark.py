"""Fig. 2 (revised, Design 2 — design decision 2026-10-21): retrieval benchmark on
hit@k and majority-vote accuracy, every bar/cell with a 95 % whole-patient interval.

Sources (accepted; no result is recomputed here)
  * $PROSPER_OUT/r2_1_fig2_retrieval_20261021/run_001/{summary,per_group}.csv
    hit@k (k = 1,3,5,10) for the 37 RR-5 grid cells, TITAN, PRISM (native tiling),
    SISH, RetCCL x 4, ResNet-50 floor; gates G1-G3 reproduce RR-5 / R3.1 / Table S25.
  * $PROSPER_OUT/r2_1_virchow2_fp32_main_20261013/run_003/rr5_replacement_001/
    table_s3_revised.csv (macro recall = mMV@5, plurality rule, canonical tie order)
    and per_class.csv (per-group recall at k = 5)
  * R3.1 summaries (plurality5 macro for SISH, PRISM native, RetCCL)

Panel A  macro hit@k for the best aggregation per encoder (ten tile encoders incl.
         RetCCL) and the three end-to-end systems (TITAN, PRISM native tiling, SISH);
         bars sorted by value within each k.
Panel B  k = 5 heat-map of macro hit@5: four aggregations x ten encoders plus an
         end-to-end row (mMV@5 computed into the source data / Table S-R2.1-E, not drawn;
         author 2026-10-21 13:49: one metric, compact).
Panel C  per-group hit@5 for TITAN, UNIv2+OT, Virchow2+Yottixel and H-Optimus-1+Yottixel
         (groups ordered by mean hit@5 over the four pipelines); MV@5 in the source data.

All intervals: percentile bootstrap over 5,000 pinned whole-patient draws (PCG64
seed 123), 2,778 queries / 1,876 patients / 15 groups.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import AGG_COLORS, GREY, prosper_out, save, style  # noqa: E402

DER = prosper_out() / "r2_1_fig2_retrieval_20261021/run_001"
RR5 = prosper_out() / "r2_1_virchow2_fp32_main_20261013/run_003/rr5_replacement_001"
R31 = {
    "sish": prosper_out() / "r3_1_comparators_whole_patient_20261008/summary.csv",
    "prism_native": prosper_out() / "r3_1_prism_20261018/eval_native_20261019/summary.csv",
    "retccl": prosper_out() / "r3_1_retccl_whole_patient_20261016/summary.csv",
}
FIGURE = "fig2_benchmark"
E2E_COLORS = {"TITAN": AGG_COLORS["TITAN"], "PRISM (native tiling)": "#d62728", "SISH": "#7f7f7f"}
E2E_COL = {"TITAN": "TITAN (CONCH v1.5)", "PRISM (native tiling)": "PRISM (Virchow)", "SISH": "SISH (DenseNet)"}
# Display names on the figure face follow the manuscript (figure review 2026-11-03 D3): the derivation CSVs
# label the uni2 pipeline "UNIv2" and the TITAN tile encoder "Conchv1.5".
SYSTEM_DISPLAY = {"UNIv2": "UNI2"}
AGG_ROWS = ["Optimal Transport", "Yottixel", "SAMPLER", "Median Pooling"]
PIPELINES = {  # panel C; pair labels "encoder + aggregation" since revision (2026-11-10)
    "titan/titan_repaired": ("CONCH v1.5 + TITAN", AGG_COLORS["TITAN"]),
    "uni2/ot_cosine": ("UNI2 + Optimal Transport", AGG_COLORS["Optimal Transport"]),
    "virchow2_fp32/yottixel": ("Virchow2 + Yottixel", AGG_COLORS["Yottixel"]),
    "h_optimus_1/yottixel": ("H-Optimus-1 + Yottixel", "#17becf"),
}
R31_CFG = {"sish": "sish", "prism_native": "prism_native", "retccl_median": "retccl_median", "retccl_ot": "retccl_ot",
           "retccl_yottixel": "retccl_yottixel", "retccl_sampler": "retccl_sampler"}


def load():
    s = pd.read_csv(DER / "summary.csv")
    s = s[(s.metric == "hit_at_k") & (s.average == "macro") & (s.kind != "floor")].copy()
    s["system"] = s.system.replace(SYSTEM_DISPLAY)
    pg = pd.read_csv(DER / "per_group.csv")
    pg = pg[(pg.metric == "hit_at_k") & (pg.k == 5) & pg.cell_id.isin(PIPELINES)].copy()
    # mMV@5 (macro recall under the plurality rule) for the grid cells and TITAN
    t = pd.read_csv(RR5 / "table_s3_revised.csv")
    t = t[(t.k == 5) & (t.rule == "plurality") & (t.tie_order == "canonical") & (t.metric == "macro_recall")
          & (t.cell_id != "titan/titan_published")]
    assert (t.ci_status == "ok").all() and len(t) == 37
    mv = t[["cell_id", "value", "ci_low", "ci_high"]].rename(columns={"value": "point", "ci_lo": "ci_lo"})
    mv = mv.rename(columns={"ci_low": "ci_lo", "ci_high": "ci_hi"})
    # comparators' mMV@5 from the R3.1 summaries (same draws)
    rows = []
    for cid, cfg in R31_CFG.items():
        f = R31["retccl" if cid.startswith("retccl") else cid]
        d = pd.read_csv(f)
        d = d[(d.population == "all_tissue_2778") & (d.config == cfg) & (d.metric == "plurality5") & (d.average == "macro")]
        assert len(d) == 1, cid
        rows.append(dict(cell_id=cid, point=d.point.iloc[0], ci_lo=d.ci_lo.iloc[0], ci_hi=d.ci_hi.iloc[0]))
    # cross-check: R3.1 primary == RR-5 uni2/ot_cosine macro recall (identical draws)
    d = pd.read_csv(R31["sish"]); d = d[(d.population == "all_tissue_2778") & (d.config == "primary") & (d.metric == "plurality5") & (d.average == "macro")].iloc[0]
    u = mv[mv.cell_id == "uni2/ot_cosine"].iloc[0]
    assert abs(d.point - u.point) < 1e-6 and abs(d.ci_lo - u.ci_lo) < 1e-6 and abs(d.ci_hi - u.ci_hi) < 1e-6  # RR-5 CSV is 6 dp
    mv = pd.concat([mv, pd.DataFrame(rows)], ignore_index=True)
    meta = s.drop_duplicates("cell_id")[["cell_id", "system", "aggregation", "kind"]]
    mv = mv.merge(meta, on="cell_id", how="left"); mv["metric"] = "mmv_at_5"
    assert mv.system.notna().all()
    # per-group MV@5 (= per-group recall) for panel C pipelines
    pc = pd.read_csv(RR5 / "per_class.csv")
    pc = pc[(pc.k == 5) & (pc.rule == "plurality") & (pc.tie_order == "canonical") & (pc.metric == "recall") & pc.cell_id.isin(PIPELINES)]
    assert (pc.ci_status == "ok").all() and len(pc) == 60
    pc = pc.rename(columns={"value": "point", "ci_low": "ci_lo", "ci_high": "ci_hi"})[["cell_id", "group", "point", "ci_lo", "ci_hi"]]
    pc["metric"] = "mv_at_5"
    return s, mv, pg, pc


def _err(ax, x, d, color="black", lw=0.6, cap=1.5):
    ax.errorbar(x, d.point, yerr=[d.point - d.ci_lo, d.ci_hi - d.point], fmt="none", ecolor=color, elinewidth=lw, capsize=cap)


def panel_a(axes, s):
    grid = s[s.kind == "grid"]; e2e = s[s.kind == "e2e"]
    out = []
    for ax, k in zip(axes.ravel(), [1, 3, 5, 10]):
        g = grid[grid.k == k].sort_values("point", ascending=False).drop_duplicates("system")  # best aggregation per encoder
        e = e2e[e2e.k == k]
        d = pd.concat([g, e]).sort_values("point", ascending=False).reset_index(drop=True)
        assert len(d) == 13, (k, len(d))
        d["panel"] = "A"; out.append(d)
        x = np.arange(len(d))
        colors = [E2E_COLORS[r.system] if r.kind == "e2e" else AGG_COLORS[r.aggregation] for r in d.itertuples()]
        hatch = ["////" if r.kind == "e2e" else None for r in d.itertuples()]
        bars = ax.bar(x, d.point, color=colors, width=0.72, edgecolor="black", linewidth=0.3)
        for b, h in zip(bars, hatch):
            if h:
                b.set_hatch(h); b.set_edgecolor("white"); b.set_linewidth(0.3)
        _err(ax, x, d)
        labels = [r.system if r.kind == "e2e" else r.system for r in d.itertuples()]
        ax.set_xticks(x); ax.set_xticklabels([l.replace(" (native tiling)", "") for l in labels], rotation=45, ha="right")
        ax.set_ylim(0.4, 1.0); ax.set_title(f"k = {k}"); ax.set_ylabel(f"Macro hit@{k}")
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    used = [g for g in AGG_ROWS if g in set(pd.concat(out).query("kind == 'grid'").aggregation)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=AGG_COLORS[g]) for g in used]
    labels = [f"{g} (best aggregation for that encoder)" for g in used]
    for nm, c in E2E_COLORS.items():
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=c, hatch="////", edgecolor="white")); labels.append(f"{nm}, end-to-end")
    return handles, labels, pd.concat(out)


def _heat(ax, piv, label, fmt="{:.2f}"):
    data = piv.values.astype(float)
    masked = np.ma.masked_invalid(data)
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad(GREY)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.5,
                        color="white" if v < vmin + 0.6 * (vmax - vmin) else "black")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=6)
    ax.spines[:].set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.ax.tick_params(labelsize=6)


def panel_b(ax_h, s, mv):
    h5 = s[s.k == 5].copy(); h5["metric"] = "hit_at_5"
    enc_order = h5[h5.kind == "grid"].groupby("system").point.max().sort_values(ascending=False).index.tolist()
    e2e_order = ["TITAN", "PRISM (native tiling)", "SISH"]
    cols = enc_order + [E2E_COL[e] for e in e2e_order]
    rows = ["End-to-end"] + AGG_ROWS

    def pivot(d):
        d = d.copy()
        d["row"] = np.where(d.kind == "e2e", "End-to-end", d.aggregation)
        d["col"] = np.where(d.kind == "e2e", d.system.map(E2E_COL), d.system)
        return d.pivot_table(index="row", columns="col", values="point", aggfunc="first").reindex(index=rows, columns=cols)
    ph = pivot(h5)
    _heat(ax_h, ph, "Macro hit@5")
    ax_h.set_ylabel("Aggregation"); ax_h.set_title("Macro hit@5")
    src = pd.concat([h5, mv])[["cell_id", "system", "aggregation", "kind", "metric", "point", "ci_lo", "ci_hi"]]  # mMV@5 rows retained, not drawn[["cell_id", "system", "aggregation", "kind", "metric", "point", "ci_lo", "ci_hi"]]
    return src, cols


def panel_c(ax_h, pg, pc):
    order = pg.groupby("group").point.mean().sort_values(ascending=False).index.tolist()
    x = np.arange(len(order)); w = 0.2
    for ax, d, ylab in [(ax_h, pg, "Per-group hit@5")]:
        for i, (cid, (label, color)) in enumerate(PIPELINES.items()):
            dd = d[d.cell_id == cid].set_index("group").reindex(order)
            assert dd.point.notna().all(), cid
            xi = x + (i - 1.5) * w
            ax.bar(xi, dd.point, width=w, color=color, edgecolor="black", linewidth=0.3, label=label)
            _err(ax, xi, dd, lw=0.5, cap=1.0)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_ylim(0, 1.0); ax.set_ylabel(ylab); ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax_h.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=4)
    src = pd.concat([pg.assign(metric="hit_at_5")[["cell_id", "group", "metric", "point", "ci_lo", "ci_hi"]], pc])  # MV@5 rows retained, not drawn
    return order, src


def main():
    style()
    s, mv, pg, pc = load()
    fig = plt.figure(figsize=(8.5, 9.8))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.3, 0.05, 0.95, 0.9], hspace=0.55, left=0.08, right=0.98, top=0.97, bottom=0.05)
    gsa = gs[0].subgridspec(2, 2, wspace=0.25, hspace=0.75)
    axes_a = np.array([fig.add_subplot(gsa[i, j]) for i in range(2) for j in range(2)])
    ax_leg = fig.add_subplot(gs[1]); ax_leg.axis("off")  # panel A legend strip, between A and B
    ax_b = fig.add_subplot(gs[2])
    ax_c = fig.add_subplot(gs[3])

    h, l, src_a = panel_a(axes_a, s)
    src_b, cols = panel_b(ax_b, s, mv)
    order, src_c = panel_c(ax_c, pg, pc)
    ax_leg.legend(h, l, loc="center", ncol=3, frameon=False)
    for ax, lab, dx in [(axes_a[0], "A", -0.25), (ax_b, "B", -0.12), (ax_c, "C", -0.12)]:
        ax.text(dx, 1.15, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    note = ("Design 2, single metric (author decisions 2026-10-21 13:1x and 13:49: hit@5 only in B and C; mMV@5 in the table). "
            "Panel A: best aggregation per encoder chosen by macro hit@k at each k; end-to-end systems hatched. Panel B: point "
            "estimates; the 95 % intervals of every cell, and the mMV@5 rows, are in source_data_panelB.csv and Table S-R2.1-E. "
            "Panel C: per-group hit@5 (derivation per_group.csv); per-group MV@5 rows retained in source_data_panelC.csv, not drawn. "
            f"Panel C group order (mean hit@5 over the four pipelines): {', '.join(order)}. "
            f"Panel B column order: {', '.join(cols)}. ImageNet ResNet-50 floor is in Table S-R2.1-E only.")
    save(fig, FIGURE, "Figure_2_revised",
         [DER / "summary.csv", DER / "per_group.csv", RR5 / "table_s3_revised.csv", RR5 / "per_class.csv",
          R31["sish"], R31["prism_native"], R31["retccl"]], note,
         {"source_data_panelA": src_a, "source_data_panelB": src_b, "source_data_panelC": src_c})


if __name__ == "__main__":
    main()
