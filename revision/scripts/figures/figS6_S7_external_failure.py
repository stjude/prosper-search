"""Supplementary Figs. S6 and S7 (R1.3-T10): external failure structure and descriptive context.

Rendering of the audit-complete R1.3-T10 specification (`docs/manuscript_supplement_edits.md`,
"R1.3-T10 — Figs.S6-S7"), cited in the revised Results (l.121) but never drawn until 2026-10-21.

Sources (read-only, accepted synthesis `$PROSPER_OUT/r1_3_external_synthesis_20260930/`):
  all_returned_label_slots.csv        S6-A  true group x retrieved label, slot_fraction of all top-5 slots
  first_relevant_rank_distribution.csv S6-B  first same-group rank: 1 | 2-5 | 6-10 | not observed through 10
  descriptive_strata_points.csv        S7    conditional hit@5 / precision@5 by KidRare source, WT pattern,
                                             CCDI-MCI preparation (points only; no intervals or tests exist)

Identities (T5): (stv, main_696), (kidrare, amended_1581), (ccdi, ffpe_scope_clear), configuration
whole_slide_ot, analysis_role main. Fixed label order ACT/ACC ... WT in every facet; absent true groups
are grey "not estimable" rows, never zero rows. Failure-conditioned slots are not used (they stay in Table S20).

Spec acceptance checks are asserted: S7-A never includes the 1,168 non-WT KidRare queries (n sums to 413);
S7-B sums 413 within each configuration; S7-C sums 735 and keeps MIXED / NONE_UNKNOWN separate from FROZEN_OCT.

Outputs  $PROSPER_OUT/figures_20261021/figS6_S7_external_failure/
         Figure_S6_revised.{png,pdf}, Figure_S7_revised.{png,pdf}, source_s6a.csv, source_s6b.csv, source_s7.csv,
         legibility_check_S6.json, legibility_check_S7.json, SHA256SUMS.txt, PROVENANCE.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GREY, GROUP_ORDER_ALPHA, HIT_COLOR, PREC_COLOR, out_dir, prosper_out, style, write_provenance  # noqa: E402
from figS1_S2_neighbor_ladder import legibility  # noqa: E402

SYN = prosper_out() / "r1_3_external_synthesis_20260930"
SLOTS = SYN / "all_returned_label_slots.csv"
RANKS = SYN / "first_relevant_rank_distribution.csv"
STRATA = SYN / "descriptive_strata_points.csv"
COMPLETE = SYN / "COMPLETE.json"
FIGURE = "figS6_S7_external_failure"
WIDTH_IN = 180 / 25.4

COHORTS = [("stv", "main_696", "STV"), ("kidrare", "amended_1581", "KidRare"), ("ccdi", "ffpe_scope_clear", "CCDI-MCI FFPE")]
RANK_ORDER = ["1", "2-5", "6-10", "not_observed_through_10"]
RANK_LABELS = ["rank 1", "rank 2–5", "rank 6–10", "not in top 10"]
RANK_COLORS = ["#08519c", "#6baed6", "#c6dbef", "#bdbdbd"]
CONFIGS = [("amended_1581", "whole_slide_ot", "All tissue", "#1f77b4"),
           ("expanded_query_only_1581", "query_only", "Query-only filtering", "#2ca02c"),
           ("expanded_both_sided_1531", "both_sided", "Both-sided filtering", "#9467bd")]
PATTERNS = [("Mixed blastemal and epithelial nephroblastoma", "Mixed blastemal/epithelial"),
            ("Mixed blastemal and stromal nephroblastoma", "Mixed blastemal/stromal"),
            ("Mixed blastemal, epithelial and stromal nephroblastoma", "Mixed all three"),
            ("Nephroblastoma, blastemal type", "Blastemal"),
            ("Nephroblastoma, stromal type", "Stromal")]
SOURCES = [("ORIGINAL_KEEP", "KEEP annotations"), ("ADDED_PATHPT", "PathPT annotations (added)")]  # KidRare WT label source
PREPS = [("VERIFIED_FFPE", "FFPE"), ("FROZEN_OCT", "Frozen/OCT"), ("MIXED", "Mixed"), ("NONE_UNKNOWN", "Unresolved")]
ENDPOINTS = [("conditional_hit_at_5", "hit@5", "o", HIT_COLOR), ("conditional_precision_at_5", "precision@5", "D", PREC_COLOR)]


def load():
    if not COMPLETE.exists():
        raise SystemExit(f"synthesis not COMPLETE: {SYN}")
    slots = pd.read_csv(SLOTS)
    ranks = pd.read_csv(RANKS)
    strata = pd.read_csv(STRATA)
    slots = slots[(slots.analysis_role == "main") & (slots.configuration == "whole_slide_ot")]
    ranks = ranks[(ranks.analysis_role == "main") & (ranks.configuration == "whole_slide_ot")]
    strata = strata[strata.endpoint.isin([e for e, *_ in ENDPOINTS])]
    return slots, ranks, strata


def draw_s6(slots, ranks):
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH_IN, 7.4),
                             gridspec_kw=dict(width_ratios=[6.2, 0.9, 4.2], wspace=0.05, hspace=0.32, left=0.08,
                                              right=0.985, top=0.93, bottom=0.07))
    src_a, src_b = [], []
    for row, (cohort, pop, title) in enumerate(COHORTS):
        ax, axn, axb = axes[row]
        s = slots[(slots.cohort == cohort) & (slots.population == pop)]
        r = ranks[(ranks.cohort == cohort) & (ranks.population == pop)]
        M = np.full((15, 15), np.nan)
        n = {}
        for g in s.tumor_group.unique():
            sg = s[s.tumor_group == g]
            n[g] = int(sg.n_returned_queries.iloc[0])
            assert sg.denominator_top5_slots.iloc[0] == 5 * n[g]
            i = GROUP_ORDER_ALPHA.index(g)
            M[i, :] = 0.0
            for rr in sg.itertuples():
                M[i, GROUP_ORDER_ALPHA.index(rr.neighbour_label)] = rr.slot_fraction
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#e6e6e6")
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, g in enumerate(GROUP_ORDER_ALPHA):
            if g not in n:
                ax.text(7, i, "not estimable", ha="center", va="center", fontsize=6, color="#7a7a7a")
        ax.set_xticks(range(15))
        ax.set_xticklabels(GROUP_ORDER_ALPHA, rotation=90)
        ax.set_yticks(range(15))
        ax.set_yticklabels(GROUP_ORDER_ALPHA)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_ylabel(f"{title}\nquery group")
        if row == 2:
            ax.set_xlabel("Retrieved reference label (all top-5 slots)")
        axn.axis("off")
        axn.set_ylim(14.5, -0.5)
        axn.set_xlim(0, 1)
        for i, g in enumerate(GROUP_ORDER_ALPHA):
            axn.text(0.1, i, str(n.get(g, "–")), ha="left", va="center", fontsize=6.5)
        if row == 0:
            axn.text(0.1, -1.0, "n", ha="left", va="center", fontsize=6.5, color=GREY)
        # B: stacked rank profile
        axb.set_ylim(14.5, -0.5)
        axb.set_xlim(0, 1)
        for i, g in enumerate(GROUP_ORDER_ALPHA):
            rg = r[r.tumor_group == g]
            if rg.empty:
                continue
            tot = int(rg.n_returned.iloc[0])
            assert tot == n[g] and int(rg.n_queries.sum()) == tot
            left = 0.0
            for cat, col in zip(RANK_ORDER, RANK_COLORS):
                frac = int(rg[rg.first_relevant_rank == cat].n_queries.sum()) / tot
                axb.barh(i, frac, left=left, color=col, height=0.72, linewidth=0)
                src_b.append(dict(cohort=cohort, population=pop, tumor_group=g, first_relevant_rank=cat,
                                  n_queries=int(rg[rg.first_relevant_rank == cat].n_queries.sum()), n_returned=tot, fraction=frac))
                left += frac
        axb.set_yticks([])
        axb.set_xticks([0, 0.5, 1])
        if row == 2:
            axb.set_xlabel("Share of returned queries")
        axb.spines["left"].set_visible(False)
        for i, g in enumerate(GROUP_ORDER_ALPHA):
            for j, lab in enumerate(GROUP_ORDER_ALPHA):
                if g in n:
                    src_a.append(dict(cohort=cohort, population=pop, tumor_group=g, neighbour_label=lab,
                                      slot_fraction=M[i, j], n_returned_queries=n[g]))
    cax = fig.add_axes([0.16, 0.962, 0.2, 0.011])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Share of top-5 slots", fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    axes[0, 2].legend(handles=[Patch(color=c, label=l) for c, l in zip(RANK_COLORS, RANK_LABELS)], loc="lower center",
                      bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=6.5, title="First same-group reference",
                      title_fontsize=6.5)
    for ax, letter in ((axes[0, 0], "A"), (axes[0, 2], "B")):
        ax.text(-0.02 if letter == "B" else -0.3, 1.1, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    return fig, pd.DataFrame(src_a), pd.DataFrame(src_b)


def dotpanel(ax, d, strata, configs, title_rows):
    """Horizontal dot plot: rows = strata, x = 0-1, configuration colours, endpoint shapes."""
    nconf = len(configs)
    offs = np.linspace(-0.25, 0.25, nconf) if nconf > 1 else [0.0]
    rows = []
    for i, (key, label) in enumerate(strata):
        for off, (pop, cfg, cname, col) in zip(offs, configs):
            dd = d[(d.population == pop) & (d.configuration == cfg) & (d.stratum == key)]
            if dd.empty:
                continue
            ntar, nret = int(dd.n_target.iloc[0]), int(dd.n_returned.iloc[0])
            for ep, ename, mk, _ in ENDPOINTS:
                v = dd[dd.endpoint == ep]
                if v.empty:
                    continue
                pt = float(v.point.iloc[0])
                ax.plot(pt, i + off, marker=mk, ms=4 if mk == "o" else 3.3, color=col, mec=col,
                        mfc=col if mk == "o" else "white", mew=0.9, ls="none", zorder=3)
                rows.append(dict(population=pop, configuration=cfg, stratum=key, endpoint=ep, point=pt, n_target=ntar, n_returned=nret))
        ax.axhline(i + 0.5, color="#eeeeee", lw=0.5, zorder=0)
    ax.set_yticks(range(len(strata)))
    ax.set_yticklabels([lab for _, lab in strata])
    ax.set_ylim(len(strata) - 0.5, -0.5)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", color="#eeeeee", lw=0.5, zorder=0)
    return pd.DataFrame(rows)


def ncolumn(axn, strata, src, configs):
    axn.axis("off")
    axn.set_ylim(len(strata) - 0.5, -0.5)
    axn.set_xlim(0, 1)
    for i, (key, _) in enumerate(strata):
        ns = [int(src[(src.stratum == key) & (src.configuration == cfg)].n_target.iloc[0]) for _, cfg, _, _ in configs
              if not src[(src.stratum == key) & (src.configuration == cfg)].empty]
        txt = str(ns[0]) if len(set(ns)) == 1 else "/".join(map(str, ns))
        axn.text(0.05, i, txt, ha="left", va="center", fontsize=6.5)
    axn.text(0.05, -0.95, "n", ha="left", va="center", fontsize=6.5, color=GREY)


def draw_s7(strata):
    kid = strata[strata.cohort == "kidrare"]
    ccd = strata[(strata.cohort == "ccdi") & (strata.population == "allprep_scope_clear") & (strata.configuration == "whole_slide_ot")]
    fig = plt.figure(figsize=(WIDTH_IN * 0.62, 4.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 5, 4], width_ratios=[6, 0.9], wspace=0.04, hspace=0.45,
                          left=0.3, right=0.97, top=0.88, bottom=0.08)
    axA, nA, axB, nB, axC, nC = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
    sA = dotpanel(axA, kid[kid.stratum_type == "source_group"], SOURCES, CONFIGS, None)
    sB = dotpanel(axB, kid[kid.stratum_type == "wt_pathpt_pattern"], PATTERNS, CONFIGS, None)
    ccfg = [("allprep_scope_clear", "whole_slide_ot", "All tissue", "#1f77b4")]
    sC = dotpanel(axC, ccd[ccd.stratum_type == "preparation_status"], PREPS, ccfg, None)
    for cfg in CONFIGS:
        a = sA[(sA.configuration == cfg[1]) & (sA.endpoint == ENDPOINTS[0][0])]
        b = sB[(sB.configuration == cfg[1]) & (sB.endpoint == ENDPOINTS[0][0])]
        assert a.n_target.sum() == 413, (cfg[1], a.n_target.sum())  # never the 1,168 non-WT KidRare queries
        assert b.n_target.sum() == 413, (cfg[1], b.n_target.sum())
    c = sC[sC.endpoint == ENDPOINTS[0][0]]
    assert c.n_target.sum() == 735 and set(c.stratum) == {k for k, _ in PREPS}
    ncolumn(nA, SOURCES, sA, CONFIGS)
    ncolumn(nB, PATTERNS, sB, CONFIGS)
    ncolumn(nC, PREPS, sC, ccfg)
    for ax in (axA, axB):
        ax.tick_params(labelbottom=False)
    axC.set_xlabel("Conditional hit@5 / precision@5")
    for ax, letter in ((axA, "A"), (axB, "B"), (axC, "C")):
        ax.text(-0.42, 1.0, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
    handles = [Line2D([], [], marker="o", color=c, ls="none", ms=4, label=n) for _, _, n, c in CONFIGS]
    handles += [Line2D([], [], marker="o", color="black", ls="none", ms=4, label="hit@5"),
                Line2D([], [], marker="D", color="black", mfc="white", ls="none", ms=3.3, label="precision@5")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.99), ncol=3, frameon=False, fontsize=6.5)
    sA["panel"], sB["panel"], sC["panel"] = "A", "B", "C"
    return fig, pd.concat([sA, sB, sC], ignore_index=True)


def main():
    style()
    slots, ranks, strata = load()
    out = out_dir(FIGURE)
    written = []
    fig6, a6, b6 = draw_s6(slots, ranks)
    c6 = legibility(fig6, out, "S6")
    fig7, s7 = draw_s7(strata)
    c7 = legibility(fig7, out, "S7")
    for fig, stem in ((fig6, "Figure_S6_revised"), (fig7, "Figure_S7_revised")):
        for ext in ("png", "pdf"):
            p = out / f"{stem}.{ext}"
            fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white",
                        **({"metadata": {"CreationDate": None}} if ext == "pdf" else {}))
            written.append(p)
    for name, frame in (("source_s6a", a6), ("source_s6b", b6), ("source_s7", s7)):
        frame.to_csv(out / f"{name}.csv", index=False)
        written.append(out / f"{name}.csv")
    written += [out / "legibility_check_S6.json", out / "legibility_check_S7.json"]
    note = ("R1.3-T10 rendering from the accepted r1_3_external_synthesis_20260930 tables; identities (stv, main_696), "
            "(kidrare, amended_1581), (ccdi, ffpe_scope_clear), whole_slide_ot, analysis_role main. S6-A slot_fraction = "
            "slot_count / (5 x n_returned_queries); diagonal = precision@5. S6-B first same-group reference rank; counts / "
            "n_returned. S7 conditional hit@5 (filled circles) and precision@5 (open diamonds), points only (the source has "
            "no intervals or tests); configuration colours all-tissue / query-only / both-sided. Spec acceptance asserted: "
            "S7-A and S7-B n sum to 413 WT-labelled KidRare queries per configuration; S7-C sums 735 with mixed/unresolved "
            f"kept separate from frozen. Legibility S6 pass={c6['pass']} overlaps={c6['overlaps']}; S7 pass={c7['pass']} "
            f"overlaps={c7['overlaps']}.")
    write_provenance(out, written, [SLOTS, RANKS, STRATA, COMPLETE], note, Path(sys.argv[0]), title=FIGURE)
    print("S6 legibility", c6["pass"], c6["overlaps"])
    print("S7 legibility", c7["pass"], c7["overlaps"])


if __name__ == "__main__":
    main()
