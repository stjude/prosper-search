#!/usr/bin/env python3
"""Table S-R3.5-L1 (current final number Supplementary Table S26): ROI heat-map parameter study
(pre-registered settings) and ROI-search values.

Spec: docs/manuscript_supplement_edits.md, '### R3.5-T5 — new Supplementary Table S-R3.5-L1 ...'
including its 2026-10-29 paired ROI candidate 001 assembly addendum (ROI aggregation block; retained
C4 annotation check), the 'Table:' instruction (development grid + report half, split, outcome rule,
degenerate flag) and the consistency corrections at R2.5-T3 (0.52 lower bound; link-agreement naming;
0.030 [0.005, 0.058] belongs to the development-selected C setting).

Blocks (sheets):
  A  heat-map parameter grid: A1 development grid (6 settings x P1/P2 x arm), A2 report half
  B  matched-cost map + graph/cluster settings: B1/B2 cost map dev/report, B3/B4 graph dev/report,
     B5 settings summary (counts, selection, outcome rule, degenerate and failed-setting handling)
  C  44-reference annotation comparison (C4): C1 endpoints, C2 per reference (+ coverage)
  D  ROI-search aggregation: D1 overall, D2 paired, D3 per group, D4 per-group paired

Assembly/re-tabulation only. Writes $PROSPER_OUT/supp_tables_20261109/C_localization/S-R3.5-L1/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c_localization_common_20261109 import (  # noqa: E402
    BUILD_DATE, BUILT_BY, ES, GROUP_DIR, LG, MINUS, MS, OUT_ROOT, REPO, Build, ci, fmt, numbering,
    rel, report_header, report_tail, sha256)

WID = "S-R3.5-L1"
TITLE = "ROI heat-map parameter study (pre-registered settings) and ROI-search values"
LOC = OUT_ROOT / "r3_5_localisation_maps_20261017"
S2 = LOC / "s2/run_003"
C4CI = OUT_ROOT / "r2_5_c4_whole_patient_ci_20261008"
C4R8 = OUT_ROOT / "c4_localization_roi8"
R4A = OUT_ROOT / "r4a_vii_whole_patient_20261015/production"
C6 = OUT_ROOT / "c6_roi_per_group_whole_patient_20261021/run_001"
C1B = OUT_ROOT / "c1b_roi_patient_excluded"
FIG6 = OUT_ROOT / "figures_20261021/fig6_roi_search"

b = Build(WID, GROUP_DIR / WID, __file__)
final = numbering(b, WID)
b.check("numbering map: S-R3.5-L1 → S26", final == "S26", final)

# ------------------------------------------------------------------ inputs
s2c = b.json(S2 / "COMPLETE.json", "L1 S2 run_003 completion record (producer hashes)")
gA = b.csv(S2 / "grid_development_A.csv", "development grid, ROI heatmap family")
gB = b.csv(S2 / "grid_development_B.csv", "development grid, matched-cost map family")
gC = b.csv(S2 / "grid_development_C.csv", "development grid, graph/cluster family")
rep = b.json(S2 / "report_estimates.json", "report-half estimates (strict and sensitivity reference sets)")
sel = b.json(S2 / "selection.json", "development selection per family")
outc = b.json(S2 / "outcome.json", "pre-registered outcome rule result per family")
s2chk = b.json(S2 / "checks.json", "S2 fidelity checks (F1-F3)")
pairs_chk = b.json(LOC / "s0_pairs/run_001/checks.json", "reference split counts (F5)")
split = b.csv(LOC / "s0_pairs/run_001/split.csv", "reference-patient split (patient codes not copied)")
ppC = b.csv(S2 / "per_pair_C.csv.gz", "per-pair graph scores (re-tabulation of non-terminated calls only)",
            usecols=["reference", "query", "setting", "terminated", "design"])
for f in ("grid_development_A.csv", "grid_development_B.csv", "grid_development_C.csv",
          "report_estimates.json", "selection.json", "outcome.json", "checks.json", "per_pair_C.csv.gz"):
    b.producer_hash(S2 / f, s2c["outputs"][f], "s2/run_003/COMPLETE.json")
b.inp(LOC / "LEAD_REVIEW_001.md", "lead ratification of L1 (implementer decisions, family outcomes)")
b.inp(LOC / "IMPLEMENTER_REPORT.md", "L1 implementer report (incidents, failed-setting rule)")

c4 = b.json(C4CI / "c4_whole_patient_ci.json", "C4 44-reference whole-patient intervals")
c4ref = b.csv(C4CI / "per_reference_variant_A.csv", "C4 per-reference values (44 paired references)",
              dtype={"reference": str})
c4sha = dict(l.split()[::-1] for l in b.text(C4CI / "sha256.txt", "C4 interval package hash record").splitlines())
b.producer_hash(C4CI / "c4_whole_patient_ci.json", c4sha["c4_whole_patient_ci.json"], "r2_5 sha256.txt")
b.producer_hash(C4CI / "per_reference_variant_A.csv", c4sha["per_reference_variant_A.csv"], "r2_5 sha256.txt")
c4pp = b.csv(C4R8 / "per_pair_c4_localization_roi8.csv", "C4 per-pair records (coverage, query tiles)",
             float_precision="round_trip", dtype={"reference": str, "query_tiles": str})
b.producer_hash(C4R8 / "per_pair_c4_localization_roi8.csv", c4["inputs"]["per_pair_csv_sha256"],
                "c4_whole_patient_ci.json inputs")
b.inp(C4R8 / "c4_localization_roi8.json", "C4 run configuration (label semantics, ROI = 8 tiles)")
b.producer_hash(C4R8 / "c4_localization_roi8.json", c4["inputs"]["json_sha256"], "c4_whole_patient_ci.json inputs")

summ = b.csv(R4A / "summary.csv", "R4.a-vii whole-patient summary (arm c1b rows)")
pdif = b.csv(R4A / "paired_differences.csv", "R4.a-vii paired differences (arm c1b rows)")
r4sha = dict(l.split()[::-1] for l in b.text(R4A / "sha256.txt", "R4.a-vii hash record").splitlines())
b.producer_hash(R4A / "summary.csv", r4sha["summary.csv"], "r4a_vii production/sha256.txt")
b.producer_hash(R4A / "paired_differences.csv", r4sha["paired_differences.csv"], "r4a_vii production/sha256.txt")
pg = b.csv(C6 / "per_group_f1.csv", "C6 per-group F1 with whole-patient intervals")
pgd = b.csv(C6 / "paired_differences.csv", "C6 per-group paired differences (published − patient_excluded)")
c6sha = dict(l.split()[::-1] for l in b.text(C6 / "SHA256SUMS.txt", "C6 hash record").splitlines())
b.producer_hash(C6 / "per_group_f1.csv", c6sha["per_group_f1.csv"], "c6 run_001 SHA256SUMS.txt")
b.producer_hash(C6 / "paired_differences.csv", c6sha["paired_differences.csv"], "c6 run_001 SHA256SUMS.txt")
c6gates = b.json(C6 / "gates.json", "C6 gates (per-group F1 = c1b JSON; intervals = R4.a-vii)")
c6bind = b.text(C6 / "input_bindings.json", "C6 input bindings")
c1b = {}
for arm, fn in (("published_k5", "roi_agg_k5_raw_hits.json"), ("patient_excluded_k5", "roi_agg_k5_raw_hits_patient_excluded.json"),
                ("published_k10", "roi_agg_k10_raw_hits.json"), ("patient_excluded_k10", "roi_agg_k10_raw_hits_patient_excluded.json")):
    c1b[arm] = b.json(C1B / fn, f"C1b ROI aggregation record ({arm}; point estimates, cross-check)")
    b.check(f"C1b {fn} sha256 is bound by C6 input_bindings.json", sha256(C1B / fn) in c6bind)
fig6 = b.csv(FIG6 / "source_data_panelA.csv", "Fig. 6A source data (consistency check)")
f6sha = dict(l.split()[::-1] for l in b.text(FIG6 / "SHA256SUMS.txt", "Fig. 6 hash record").splitlines())
b.producer_hash(FIG6 / "source_data_panelA.csv", f6sha["source_data_panelA.csv"], "fig6_roi_search/SHA256SUMS.txt")
b.inp(REPO / "scripts/figures/fig6_roi_search.py", "accepted Fig. 6A layout (read)")
b.inp(REPO / "scripts/r4a_c6_roi_per_group_whole_patient.py", "accepted C6 layout (read)")

# ------------------------------------------------------------------ labels
DESIGN = {"P1": "natural top-ten retrieval pairs (P1)", "P2": "constructed annotated pairs (P2)"}
REFSET = {"strict": "≥20 tumour / ≥20 non-tumour locations (primary)", "loose": "≥10 / ≥10 (sensitivity)"}
METRIC = {
    "auroc_A": "AUROC, tumour vs pooled non-tumour (annotated locations)",
    "ap_A": "average precision", "pointing_A": "pointing: highest-scoring annotated location is tumour",
    "tumour_fraction_A": "annotated tumour share", "pointing_minus_share": "pointing minus tumour share",
    "coverage_tiles": "share of reference tiles with a map value",
    "coverage_annotated_patches": "share of annotated locations with a map value",
    "excess_bin": "transport-link class agreement minus chance (tumour / non-tumour)",
    "agreement_bin": "transport-link class agreement (tumour / non-tumour)", "chance_bin": "composition-based chance (binary)",
    "purity_bin": "mass-weighted cluster purity (binary)", "nmi_bin": "reference-side NMI (binary)",
    "n_clusters": "number of clusters", "median_cluster_size": "median cluster size (tiles)",
    "excess_5": "transport-link class agreement minus chance (five classes)", "purity_5": "cluster purity (five classes)",
    "nmi_5": "NMI (five classes)"}
COMP = {"selected": "development-selected setting", "submitted": "submitted setting",
        "selected_minus_submitted": "selected − submitted (paired)", "same_minus_other": "same-group − other-group"}
ARM = {"same_group": "same-group", "other_group": "other-group", "same_minus_other": "same − other"}


def parse_setting(s):
    p = s.split("|")
    if p[0] == "A":
        return {"distance": {"l1": "L1", "l2": "Euclidean", "cosine": "cosine"}[p[1]], "overlap aggregation": p[2]}
    if p[0] == "B":
        return {"cost statistic": {"meancost": "transport-weighted mean link cost", "mincost": "minimum link cost"}[p[1]],
                "sparsification": p[2], "k": "" if p[3] == "None" else p[3]}
    return {"sparsification": p[1], "sparsification parameter": "" if p[2] == "None" else p[2],
            "clustering": p[3].replace("_", " "), "Louvain resolution": "" if p[4] == "None" else p[4]}


def grid_sheets(g, fam, metrics):
    s = sel[fam]
    rows, raw = [], []
    for _, r in g.iterrows():
        tag = []
        if r.setting == s["selected"]:
            tag.append("development-selected")
        if r.setting == s["submitted"]:
            tag.append("submitted")
        d = {"design": DESIGN[r.design], "arm": ARM[r.arm], "setting key": r.setting, **parse_setting(r.setting),
             "role": "; ".join(tag)}
        for m in metrics:
            d[METRIC[m]] = round(float(r[m]), 4) if pd.notna(r[m]) else None
        d["n reference patients (development)"] = int(r.n_references)
        if fam == "C":
            d["degenerate on development (excluded from selection)"] = bool(r.degenerate_setting)
            d["development same-group degenerate share"] = round(float(r.dev_same_group_degenerate_share), 4)
            d["development same-group non-terminated share"] = round(float(r.dev_same_group_nonterminated_share), 4)
        d["grid order"] = int(r.grid_order)
        rows.append(d)
        raw.append({**r.to_dict(), "role": "; ".join(tag)})
    return pd.DataFrame(rows), pd.DataFrame(raw)


def report_sheet(fam):
    rows, raw = [], []
    for rs in ("strict", "loose"):
        for key, v in rep["estimates"][rs][fam].items():
            design, comp, arm, metric = key.split("|")
            setting = v.get("setting") or (f"{v.get('selected', sel[fam]['selected'])} − {v.get('submitted', sel[fam]['submitted'])}")
            rows.append({"reference set": REFSET[rs], "design": DESIGN[design], "comparison": COMP[comp],
                         "arm": ARM[arm], "metric": METRIC[metric], "setting": setting,
                         "estimate [95% CI]": ci(v["point"], v["ci95_lo"], v["ci95_hi"],
                                                 dp=1 if metric in ("n_clusters", "median_cluster_size") else 3,
                                                 plus=comp == "selected_minus_submitted" or arm == "same_minus_other"),
                         "n reference patients (report half)": v["n_references"]})
            raw.append({"reference_set": rs, "key": key, "design": design, "comparison": comp, "arm": arm,
                        "metric": metric, "setting": setting, "point": v["point"], "ci95_lo": v["ci95_lo"],
                        "ci95_hi": v["ci95_hi"], "n_references": v["n_references"],
                        "n_replicates_finite": v.get("n_replicates_finite"), "method": v.get("method", "")})
    return pd.DataFrame(rows), pd.DataFrame(raw)


# ------------------------------------------------------------------ block A
mA = ["auroc_A", "ap_A", "pointing_A", "tumour_fraction_A", "pointing_minus_share"]
A1, A1r = grid_sheets(gA, "A", mA)
A2, A2r = report_sheet("A")
b.check("A: 6 heatmap settings (3 distances × 2 overlap aggregations)", gA.setting.nunique() == 6)
b.check("A: development grid = 6 settings × 2 designs × 2 arms", len(gA) == 24)
tie = all(np.allclose(gA[gA.setting == f"A|{d}|mean"].drop(columns=["setting", "grid_order"]).select_dtypes("number").values,
                      gA[gA.setting == f"A|{d}|min"].drop(columns=["setting", "grid_order"]).select_dtypes("number").values)
          for d in ("l1", "l2", "cosine"))
b.check("A: mean and minimum overlap aggregation give identical development values for every distance", tie)
b.check("A: selection = cosine|mean (2 tied at max, grid order), criterion P1 same-group AUROC",
        sel["A"]["selected"] == "A|cosine|mean" and sel["A"]["n_tied_at_max"] == 2 and sel["A"]["design"] == "P1")
devA = gA[(gA.design == "P1") & (gA.arm == "same_group")].set_index("setting").auroc_A
b.check("A: selection.json dev values equal the grid (P1 same-group AUROC)",
        abs(devA["A|cosine|mean"] - sel["A"]["dev_value_selected"]) < 1e-12 and abs(devA["A|l1|mean"] - sel["A"]["dev_value_submitted"]) < 1e-12)
b.check("A: outcome rule adopted the selected setting (paired report-half lower bound > 0)",
        outc["A"]["adopt_selected"] is True and rep["estimates"]["strict"]["A"]["P1|selected_minus_submitted|same_group|auroc_A"]["ci95_lo"] > 0)

# ------------------------------------------------------------------ block B
mB = mA + ["coverage_tiles", "coverage_annotated_patches"]
B1, B1r = grid_sheets(gB, "B", mB)
B2, B2r = report_sheet("B")
mC = ["excess_bin", "agreement_bin", "chance_bin", "purity_bin", "nmi_bin", "n_clusters", "median_cluster_size",
      "excess_5", "purity_5", "nmi_5"]
B3, B3r = grid_sheets(gC, "C", mC)
B4, B4r = report_sheet("C")
b.check("B: 14 matched-cost settings", gB.setting.nunique() == 14 and len(gB) == 56)
b.check("B: 380 graph/cluster settings", gC.setting.nunique() == 380 and len(gC) == 760)
b.check("B: total pre-registered settings 6 + 14 + 380 = 400", gA.setting.nunique() + gB.setting.nunique() + gC.setting.nunique() == 400)
deg = gC[gC.arm == "same_group"].degenerate_setting
b.check("B: 51 graph settings degenerate on development; 380 − 51 = 329 = selection candidates",
        int(deg.sum()) == 51 and sel["C"]["n_candidates"] == 380 - int(deg.sum()), f"{int(deg.sum())} / {sel['C']['n_candidates']}")
sub_deg = float(gC[(gC.setting == sel["C"]["submitted"]) & (gC.arm == "same_group")].dev_same_group_degenerate_share.iloc[0])
b.check("B: submitted graph setting is non-degenerate (development degenerate share 0.27)",
        round(sub_deg, 2) == 0.27 and not bool(gC[gC.setting == sel["C"]["submitted"]].degenerate_setting.iloc[0]), f"{sub_deg:.4f}")
nt = ppC[~ppC.terminated.astype(bool)]
ntu = nt.drop_duplicates(["reference", "query", "setting"])
n_plans = ppC[["reference", "query"]].drop_duplicates().shape[0]
b.check("B: non-terminated Louvain calls = 251 plan×setting rows of 292,220 (135 settings, 13 pairs; LEAD_REVIEW_001 §4)",
        len(ntu) == 251 and n_plans * 380 == 292220 and ntu.setting.nunique() == 135
        and ntu[["reference", "query"]].drop_duplicates().shape[0] == 13,
        f"{len(ntu)} rows ({len(nt)} pair rows with draw multiplicity), {ntu.setting.nunique()} settings, "
        f"{ntu[['reference', 'query']].drop_duplicates().shape[0]} pairs, {n_plans} unique plans")
csg = gC[gC.arm == "same_group"].excess_bin
b.check("B: development same-group link agreement − chance spans 0.023–0.034 over all 380 settings",
        round(csg.min(), 3) == 0.023 and round(csg.max(), 3) == 0.034, f"{csg.min():.4f}–{csg.max():.4f}")
devB = gB[(gB.design == "P1") & (gB.arm == "same_group")].auroc_A
b.check("B: development P1 same-group AUROC of all 14 cost-map settings ≤ 0.68", round(devB.max(), 2) == 0.68,
        f"{devB.min():.4f}–{devB.max():.4f}")
b.check("B/C: selection.json dev values equal the grids",
        abs(gB[(gB.design == "P1") & (gB.arm == "same_group") & (gB.setting == sel["B"]["selected"])].auroc_A.iloc[0] - sel["B"]["dev_value_selected"]) < 1e-12
        and abs(gC[(gC.arm == "same_group") & (gC.setting == sel["C"]["selected"])].excess_bin.iloc[0] - sel["C"]["dev_value_selected"]) < 1e-12)
b.check("B/C: outcome rule kept the submitted setting for both families",
        outc["B"]["adopt_selected"] is False and outc["C"]["adopt_selected"] is False)
same_all = True
for fam in "ABC":
    for k, v in rep["estimates"]["strict"][fam].items():
        w = rep["estimates"]["loose"][fam].get(k)
        if w is None or v["point"] != w["point"] or v["ci95_lo"] != w["ci95_lo"]:
            same_all = False
b.check("report half: the ≥10/≥10 sensitivity set gives identical estimates to the strict set (all families)", same_all)
b.check("split: 24 development / 22 report patients (strict 23 / 22); 45 strict + 1 sensitivity-only reference",
        pairs_chk["F5"]["n_dev_patients"] == 24 and pairs_chk["F5"]["n_report_patients"] == 22
        and (split["set"] == "strict").sum() == 45 and len(split) == 46)
b.check("S2 fidelity checks F1, F1b (P2 submitted = C4 on 880 pairs), F2, F3 pass",
        s2chk["F1"] and s2chk["F2"] and s2chk["F3"] and s2chk["F1b_P2_submitted_equals_C4"]["pass"]
        and s2chk["F1b_P2_submitted_equals_C4"]["n_pairs_compared"] == 880)

E = rep["estimates"]["strict"]
summ_rows = []
for fam, label, n_set in (("A", "ROI heatmap (Fig. 6B)", 6), ("B", "matched-cost (transport-weighted) map", 14),
                          ("C", "graph / correspondence clusters (Fig. 5)", 380)):
    s, o = sel[fam], outc[fam]
    d = E[fam][o["rule_key"]]
    summ_rows.append({
        "family": label, "settings pre-registered": n_set,
        "eligible for selection": s["n_candidates"],
        "excluded from selection": "51 degenerate on development (one cluster or >50 % singleton clusters)" if fam == "C" else "none",
        "selection criterion (development half)": f"{METRIC[s['criterion']]}, {DESIGN[s['design']]}, same-group",
        "development-selected": s["selected"], "development value (selected)": round(s["dev_value_selected"], 4),
        "submitted": s["submitted"], "development value (submitted)": round(s["dev_value_submitted"], 4),
        "settings tied at maximum": s["n_tied_at_max"],
        "outcome rule": "adopt selected if the paired report-half interval (selected − submitted, same-group) lies above 0",
        "report-half paired difference [95% CI]": ci(d["point"], d["ci95_lo"], d["ci95_hi"], plus=True),
        "n report-half reference patients": d["n_references"],
        "outcome": "selected adopted" if o["adopt_selected"] else "submitted kept",
        "failed-setting handling": (
            "Louvain calls not terminating within 30 s recorded terminated = False, counted as degenerate and excluded from "
            f"that setting's means: {len(ntu)} of {n_plans * 380:,} plan×setting rows ({ntu.setting.nunique()} settings, "
            f"{ntu[['reference', 'query']].drop_duplicates().shape[0]} pairs)") if fam == "C" else "none needed (no partition step)"})
B5 = pd.DataFrame(summ_rows)

# ------------------------------------------------------------------ block C (C4)
R = c4["results"]
refs44 = set(c4ref.reference)
pp44 = c4pp[c4pp.reference.isin(refs44)].copy()
b.check("C4: 44 paired references, 880 pairs (440 same-group, 440 other-group), all variant-A scored",
        len(refs44) == 44 and len(pp44) == 880 and (pp44.arm == "same_group").sum() == 440 and pp44.auroc_A.notna().all())
b.check("C4: B = 5,000, PCG64 seed 123, whole-reference-patient resampling, 44 patients = 44 references",
        c4["config"]["B"] == 5000 and c4["config"]["seed"] == 123 and c4["config"]["generator"] == "numpy PCG64"
        and c4["config"]["patients_equal_references"])
sg = pp44[pp44.arm == "same_group"]
n_point = int((sg.pointing_A == True).sum())  # noqa: E712
ntiles = pp44.query_tiles.str.split(";").str.len()
short = pp44[ntiles < 8]
u = pp44.drop_duplicates("reference").set_index("reference")
b.check("C4: one n_annotated / n_patches / tumour share per reference in the per-pair file",
        pp44.groupby("reference")[["n_annotated", "n_patches", "tumour_fraction_A"]].nunique().max().max() == 1)
pooled_share = float((u.tumour_fraction_A * u.n_annotated).sum() / u.n_annotated.sum())
b.check("C4: per-reference means reproduce the interval-package points (auroc_same, tumour_share)",
        abs(c4ref.auroc_same.mean() - R["auroc_same"]["point"]) < 1e-9 and abs(c4ref.tumour_share.mean() - R["tumour_share"]["point"]) < 1e-9)
b.check("C4: per-pair same-group pointing mean equals pointing_same", abs(n_point / 440 - R["pointing_same"]["point"]) < 1e-12)
b.check("C4: same-group AUROC higher on 34/44 references", int((c4ref.auroc_diff > 0).sum()) == 34 == c4["counts"]["n_refs_auroc_diff_positive"])

C1rows = [
    ("Same-group AUROC (L1 heatmap)", "auroc_same", "44 references; mean of ten constructed same-group queries per reference"),
    ("Other-group AUROC", "auroc_other", "same 44 references, not the unpaired 45-reference control"),
    ("Same − other AUROC", "auroc_diff", "paired reference contrast"),
    ("Pointing: tumour at highest-scoring annotated location, same-group", "pointing_same", f"{n_point}/440 same-group pairs; ten per reference"),
    ("Pointing, other-group", "pointing_other", "440 other-group pairs"),
    ("Pointing, same − other", "pointing_diff", "paired reference contrast"),
    ("Annotated tumour share", "tumour_share", "average per-reference share; pooled (location-weighted) share "
                                                 f"{pooled_share:.3f} is not the reported estimand"),
    ("Pointing (same-group) minus tumour share", "pointing_minus_share", "paired excess, not interval-overlap inference"),
    ("Average precision, same-group", "ap_same", "44 references"),
    ("Average precision, other-group", "ap_other", "44 references"),
    ("Average precision, same − other", "ap_diff", "paired reference contrast"),
]
C1 = pd.DataFrame([{"endpoint": n, "estimate [95% CI]": ci(R[k]["point"], R[k]["ci95_lo"], R[k]["ci95_hi"], plus=k.endswith("diff") or k == "pointing_minus_share"),
                    "n references": R[k]["n_references"], "denominator / interpretation": t} for n, k, t in C1rows]
                  + [{"endpoint": "Same-group AUROC higher than other-group (descriptive count)", "estimate [95% CI]": "34/44",
                      "n references": 44, "denominator / interpretation": "descriptive, not another hypothesis test"},
                     {"endpoint": "Paired-reference draws with fewer than eight query tiles", "estimate [95% CI]": f"{len(short)}/880",
                      "n references": 44, "denominator / interpretation":
                          f"{int((short.arm == 'same_group').sum())} same-group / {int((short.arm == 'other_group').sum())} other-group"},
                     {"endpoint": "Annotation coverage: annotated / total projected locations (median, range over references)",
                      "estimate [95% CI]": f"{(u.n_annotated / u.n_patches).median():.3f} ({(u.n_annotated / u.n_patches).min():.3f}–{(u.n_annotated / u.n_patches).max():.3f})",
                      "n references": 44, "denominator / interpretation":
                          f"pooled {int(u.n_annotated.sum()):,} / {int(u.n_patches.sum()):,} = {u.n_annotated.sum() / u.n_patches.sum():.3f}; builder re-tabulation of the per-pair file"},
                     {"endpoint": "Unpaired other-group control (all 45 references; not the reported population)",
                      "estimate [95% CI]": fmt(c4["reconciliation"]["other_group_all_references"]["auroc_mean"]),
                      "n references": 45, "denominator / interpretation": "includes reference 13748 (no same-group candidate); point only"}])
C1r = pd.DataFrame([{"endpoint": k, **v} for k, v in R.items()])
C2r = c4ref.drop(columns=["ref_patient"]).merge(
    u[["n_annotated", "n_patches"]].reset_index(), on="reference", how="left").merge(
    short.groupby("reference").size().rename("n_draws_lt8_tiles").reset_index(), on="reference", how="left")
C2r["n_draws_lt8_tiles"] = C2r.n_draws_lt8_tiles.fillna(0).astype(int)
C2r["annotated_share_of_locations"] = C2r.n_annotated / C2r.n_patches
C2 = C2r.copy()
for c in C2.columns:
    if C2[c].dtype == float and c not in ("n_annotated", "n_patches"):
        C2[c] = C2[c].round(3)
C2 = C2.rename(columns={"reference": "reference slide (internal index)", "ref_group": "reference tumour group",
                        "n_annotated": "annotated locations", "n_patches": "projected locations",
                        "n_draws_lt8_tiles": "query draws with < 8 tiles"})

# ------------------------------------------------------------------ block D (ROI aggregation)
CFG = {"published_k5": ("query slide only excluded (submitted aggregation)", 5, "primary"),
       "patient_excluded_k5": ("all same-patient slides excluded (reported, Fig. 6A)", 5, "primary"),
       "published_k10": ("query slide only excluded (submitted aggregation)", 10, "sensitivity"),
       "patient_excluded_k10": ("all same-patient slides excluded", 10, "sensitivity")}
c1bs = summ[(summ.arm == "c1b")]
D1r = c1bs[c1bs.metric == "f1"].copy()
b.check("D: four configurations × macro/micro F1, each 2,778 slides / 1,876 patients / 15 groups, B 5,000, 0 non-finite",
        len(D1r) == 8 and (D1r.n_queries == 2778).all() and (D1r.n_patients == 1876).all() and (D1r.n_groups == 15).all()
        and (D1r.n_replicates == 5000).all() and (D1r.n_nonfinite == 0).all())
D1 = pd.DataFrame([{"exclusion": CFG[r.config][0], "hits per ROI (k)": CFG[r.config][1], "role": CFG[r.config][2],
                    "F1 average": r.average, "F1 [95% CI]": ci(r.point, r.ci_lo, r.ci_hi),
                    "slides": int(r.n_queries), "patients": int(r.n_patients), "configuration key": r.config}
                   for r in D1r.itertuples()])
D2r = pdif[(pdif.arm == "c1b") & (pdif.metric == "f1")].copy()
D2 = pd.DataFrame([{"contrast": "all same-patient slides excluded − query slide only excluded", "hits per ROI (k)": int(r.minuend.split("_k")[1]),
                    "role": r.role, "F1 average": r.average, "paired difference [95% CI]": ci(r.point, r.ci_lo, r.ci_hi, plus=True),
                    "slides": int(r.n_queries), "patients": int(r.n_patients)} for r in D2r.itertuples()])
b.check("D: 4 paired contrasts (k 5/10 × macro/micro), patient_excluded − published", len(D2r) == 4
        and set(D2r.minuend) == {"patient_excluded_k5", "patient_excluded_k10"})
b.check("D: 60 per-group rows = 4 configurations × 15 groups; B 5,000; 0 non-finite", len(pg) == 60
        and pg.group.nunique() == 15 and (pg.n_replicates == 5000).all() and (pg.n_nonfinite == 0).all())
pgx = pg.assign(cfg=pg.arm)
D3 = pd.DataFrame([{"exclusion": CFG[r.arm][0], "hits per ROI (k)": int(r.k), "role": CFG[r.arm][2], "tumour group": r.group,
                    "slides": int(r.n_slides), "patients": int(r.n_patients), "F1 [95% CI]": ci(r.f1, r.ci_lo, r.ci_hi)}
                   for r in pg.itertuples()])
D4 = pd.DataFrame([{"contrast (sign as in source)": "query slide only excluded − all same-patient slides excluded "
                    "(opposite sign to sheet D2)", "hits per ROI (k)": int(r.k), "tumour group": r.group,
                    "paired difference [95% CI]": ci(r.point, r.ci_lo, r.ci_hi, plus=True),
                    "interval excludes 0 (nominal)": bool(r.excludes_zero)} for r in pgd.itertuples()])
b.check("D: 30 per-group paired rows (15 groups × k 5/10), published − patient_excluded", len(pgd) == 30
        and set(pgd.minuend) == {"published"} and set(pgd.subtrahend) == {"patient_excluded"})
b.check("D: C6 gates all passed (per-group F1 = C1b JSON; macro/micro intervals = R4.a-vii)", all(g["passed"] for g in c6gates))
mx = 0.0
for arm, js in c1b.items():
    sub = pg[pg.arm == arm].set_index("group").f1
    mx = max(mx, max(abs(sub[g] - js["per_class_f1"][g]) for g in sub.index))
    s_ = c1bs[(c1bs.config == arm) & (c1bs.metric == "f1")].set_index("average").point
    mx = max(mx, abs(s_["macro"] - js["point_estimates"]["macro_f1"]), abs(s_["micro"] - js["point_estimates"]["micro_f1"]))
b.check("D: per-group F1 and macro/micro points equal the four C1b JSON records (distinct source)", mx < 1e-12, f"max |Δ| {mx:.1e}")
f6g = fig6[fig6.kind == "group"].set_index("category")
p5 = pg[pg.arm == "published_k5"].set_index("group")
e5 = pg[pg.arm == "patient_excluded_k5"].set_index("group")
d6 = max(max(abs(f6g.loc[g, "f1_submitted"] - p5.loc[g, "f1"]), abs(f6g.loc[g, "ci_lo_patient_excluded"] - e5.loc[g, "ci_lo"]),
             abs(f6g.loc[g, "ci_hi_patient_excluded"] - e5.loc[g, "ci_hi"]), abs(f6g.loc[g, "f1_patient_excluded"] - e5.loc[g, "f1"]))
         for g in f6g.index)
f6o = fig6[fig6.kind == "overall"].set_index("category")
so = c1bs[(c1bs.metric == "f1") & (c1bs.config == "patient_excluded_k5")].set_index("average")
d6 = max(d6, abs(f6o.loc["Macro F1", "f1_patient_excluded"] - so.loc["macro", "point"]), abs(f6o.loc["Micro F1", "ci_hi_patient_excluded"] - so.loc["micro", "ci_hi"]))
b.check("D: Fig. 6A source data equal the C6 per-group (k = 5) and R4.a-vii overall rows used here", d6 < 1e-12, f"max |Δ| {d6:.1e}")
dk5 = pgd[pgd.k == 5].sort_values("point", ascending=False)
top3 = list(dk5.group.head(3))
b.check("D: largest per-group point decreases (k = 5) are MRT, MPNST, DSRCT", set(top3) == {"MRT", "MPNST", "DSRCT"}, ", ".join(top3))

# ------------------------------------------------------------------ manuscript value checks
EA = E["A"]
b.value("ROI-heatmap settings", "6", S2 / "grid_development_A.csv", "nunique(setting)", int(gA.setting.nunique()), exact=True,
        doc=LG, needle="six ROI-heatmap, 14 matched-cost-map and 380 graph/clustering settings")
b.value("matched-cost-map settings", "14", S2 / "grid_development_B.csv", "nunique(setting)", int(gB.setting.nunique()), exact=True,
        doc=LG, needle="six ROI-heatmap, 14 matched-cost-map and 380 graph/clustering settings")
b.value("graph/clustering settings", "380", S2 / "grid_development_C.csv", "nunique(setting)", int(gC.setting.nunique()), exact=True,
        doc=LG, needle="six ROI-heatmap, 14 matched-cost-map and 380 graph/clustering settings")
k = "P1|selected|same_group|auroc_A"
for lab, pr, f in (("cosine heatmap AUROC", "0.85", "point"), ("cosine AUROC lower", "0.75", "ci95_lo"), ("cosine AUROC upper", "0.93", "ci95_hi")):
    b.value(lab, pr, S2 / "report_estimates.json", f"estimates.strict.A['{k}'].{f}", EA[k][f], 2, LG, "AUROC 0.85 (95% CI 0.75–0.93)")
k = "P1|submitted|same_group|auroc_A"
for lab, pr, f in (("L1 heatmap AUROC", "0.72", "point"), ("L1 AUROC lower", "0.59", "ci95_lo"), ("L1 AUROC upper", "0.84", "ci95_hi")):
    b.value(lab, pr, S2 / "report_estimates.json", f"estimates.strict.A['{k}'].{f}", EA[k][f], 2, LG, "against 0.72 (0.59–0.84) for L1")
k = "P1|selected_minus_submitted|same_group|auroc_A"
for lab, pr, f in (("cosine − L1 paired", "+0.13", "point"), ("paired lower", "0.07", "ci95_lo"), ("paired upper", "0.20", "ci95_hi")):
    b.value(lab, pr, S2 / "report_estimates.json", f"estimates.strict.A['{k}'].{f}", EA[k][f], 2, LG, "paired difference +0.13, 0.07–0.20")
b.value("report-half reference patients (heatmap)", "16", S2 / "report_estimates.json", f"estimates.strict.A['{k}'].n_references",
        EA[k]["n_references"], exact=True, doc=LG, needle="+0.13, 0.07–0.20; 16 report-half reference patients")
b.value("development-selected heatmap setting", "A|cosine|mean", S2 / "selection.json", "A.selected", sel["A"]["selected"], exact=True,
        doc=LG, needle="development-selected cosine-distance heat map")
k = "P1|submitted|same_group|auroc_A"
EB = E["B"]
for lab, pr, f in (("matched-cost map AUROC", "0.65", "point"), ("matched-cost lower", "0.52", "ci95_lo"), ("matched-cost upper", "0.78", "ci95_hi")):
    b.value(lab, pr, S2 / "report_estimates.json", f"estimates.strict.B['{k}'].{f} (submitted mean-cost setting)", EB[k][f], 2,
            MS, "AUROC 0.65, 0.52–0.78; 16 report-half reference patients")
b.value("report-half reference patients (cost map)", "16", S2 / "report_estimates.json", f"estimates.strict.B['{k}'].n_references",
        EB[k]["n_references"], exact=True, doc=MS, needle="0.52–0.78; 16 report-half reference patients")
b.value("max development AUROC over 14 cost-map settings", "0.68", S2 / "grid_development_B.csv",
        "max auroc_A (P1, same_group)", float(devB.max()), 2, ES, "at most 0.68 at any of the 14 settings on the development half")
k = "P2|selected|same_group|excess_bin"
EC = E["C"]
for lab, pr, f in (("graph setting link agreement − chance", "0.030", "point"), ("agreement lower", "0.005", "ci95_lo"),
                   ("agreement upper", "0.058", "ci95_hi")):
    b.value(lab, pr, S2 / "report_estimates.json", f"estimates.strict.C['{k}'].{f} (development-selected setting)", EC[k][f], 3,
            LG, "0.030 (95% CI 0.005–0.058; 22 report-half reference patients)")
b.value("report-half reference patients (graph)", "22", S2 / "report_estimates.json", f"estimates.strict.C['{k}'].n_references",
        EC[k]["n_references"], exact=True, doc=LG, needle="0.005–0.058; 22 report-half reference patients")
k = "P2|submitted|same_group|excess_bin"
for lab, pr, f in (("full-plan setting link agreement − chance", "0.030", "point"), ("full-plan lower", "0.006", "ci95_lo"),
                   ("full-plan upper", "0.057", "ci95_hi")):
    b.value(lab, pr, S2 / "report_estimates.json", f"estimates.strict.C['{k}'].{f} (submitted full-plan setting)", EC[k][f], 3,
            ES, "full-plan setting has its own")
k = "P2|selected_minus_submitted|same_group|excess_bin"
b.check("graph: development-selected − full-plan link agreement interval includes 0 ('did not establish improved agreement')",
        EC[k]["ci95_lo"] < 0 < EC[k]["ci95_hi"], ci(EC[k]["point"], EC[k]["ci95_lo"], EC[k]["ci95_hi"], plus=True))
for lab, pr, key in (("C4 same-group AUROC", "0.691", "auroc_same"), ("C4 other-group AUROC", "0.588", "auroc_other"),
                     ("C4 paired difference", "0.103", "auroc_diff")):
    needle = {"auroc_same": "AUROC 0.691 (95% CI 0.614–0.764)", "auroc_other": "0.588 (0.502–0.673)",
              "auroc_diff": "paired difference 0.103, 0.064–0.146"}[key]
    b.value(lab, pr, C4CI / "c4_whole_patient_ci.json", f"results.{key}.point", R[key]["point"], 3, MS, needle)
    lo, hi = {"auroc_same": ("0.614", "0.764"), "auroc_other": ("0.502", "0.673"), "auroc_diff": ("0.064", "0.146")}[key]
    b.value(lab + " lower", lo, C4CI / "c4_whole_patient_ci.json", f"results.{key}.ci95_lo", R[key]["ci95_lo"], 3, MS, needle)
    b.value(lab + " upper", hi, C4CI / "c4_whole_patient_ci.json", f"results.{key}.ci95_hi", R[key]["ci95_hi"], 3, MS, needle)
b.value("C4 references", "44", C4CI / "c4_whole_patient_ci.json", "config.n_references", c4["config"]["n_references"], exact=True,
        doc=MS, needle="on 44 reference slides with pathologist annotations")
for lab, pr, key, f in (("pointing same-group", "0.816", "pointing_same", "point"), ("pointing lower", "0.748", "pointing_same", "ci95_lo"),
                        ("pointing upper", "0.882", "pointing_same", "ci95_hi"), ("annotated tumour share", "0.683", "tumour_share", "point"),
                        ("pointing − share", "0.133", "pointing_minus_share", "point"), ("pointing − share lower", "0.049", "pointing_minus_share", "ci95_lo"),
                        ("pointing − share upper", "0.220", "pointing_minus_share", "ci95_hi")):
    b.value(lab, pr, C4CI / "c4_whole_patient_ci.json", f"results.{key}.{f}", R[key][f], 3, ES,
            {"pointing_same": "| Tumor at highest-scoring annotated location | 0.816",
             "tumour_share": "| Annotated tumor share | 0.683", "pointing_minus_share": "| Pointing minus that share | 0.133"}[key])
b.value("pointing count", "359", C4R8 / "per_pair_c4_localization_roi8.csv", "count(pointing_A) same_group, 44 refs", str(n_point),
        exact=True, doc=ES, needle="359/440 same-group pairs")
b.value("pooled tumour share (not quoted)", "0.707", C4R8 / "per_pair_c4_localization_roi8.csv",
        "Σ(tumour_fraction_A·n_annotated)/Σ n_annotated over 44 refs", pooled_share, 3, ES, "not pooled 0.707")
b.value("references with same > other AUROC", "34", C4CI / "c4_whole_patient_ci.json", "counts.n_refs_auroc_diff_positive",
        str(c4["counts"]["n_refs_auroc_diff_positive"]), exact=True, doc=ES, needle="higher on 34/44 references")
b.value("draws with < 8 query tiles", "35", C4R8 / "per_pair_c4_localization_roi8.csv", "count(len(query_tiles) < 8), 880 pairs",
        str(len(short)), exact=True, doc=ES, needle="Thirty-five of 880 paired-reference draws")
b.value("short draws same-group", "24", C4R8 / "per_pair_c4_localization_roi8.csv", "… arm = same_group",
        str(int((short.arm == "same_group").sum())), exact=True, doc=ES, needle="(24 same-group/11 other-group)")
b.value("short draws other-group", "11", C4R8 / "per_pair_c4_localization_roi8.csv", "… arm = other_group",
        str(int((short.arm == "other_group").sum())), exact=True, doc=ES, needle="(24 same-group/11 other-group)")
sv = lambda cfg, avg: c1bs[(c1bs.metric == "f1") & (c1bs.config == cfg) & (c1bs.average == avg)].iloc[0]  # noqa: E731
pv = lambda mn, avg: D2r[(D2r.minuend == mn) & (D2r.average == avg)].iloc[0]  # noqa: E731
r_ = sv("patient_excluded_k5", "macro")
for lab, pr, f in (("ROI macro F1 (all same-patient excluded)", "0.765", "point"), ("macro lower", "0.741", "ci_lo"), ("macro upper", "0.787", "ci_hi")):
    b.value(lab, pr, R4A / "summary.csv", f"c1b/patient_excluded_k5/f1/macro.{f}", float(r_[f]), 3, MS, "macro F1 0.765 (95% CI 0.741–0.787)")
r_ = pv("patient_excluded_k5", "macro")
for lab, pr, f in (("paired difference vs query-slide-only", "−0.025", "point"), ("paired lower", "−0.038", "ci_lo"), ("paired upper", "−0.013", "ci_hi")):
    b.value(lab, pr, R4A / "paired_differences.csv", f"c1b/patient_excluded_k5−published_k5/f1/macro.{f}", float(r_[f]), 3,
            LG, "paired difference −0.025, −0.038 to −0.013")
b.value("micro F1 query-slide-only", "0.875", R4A / "summary.csv", "c1b/published_k5/f1/micro.point", float(sv("published_k5", "micro").point), 3,
        LG, "micro F1 decreased from 0.875 to 0.863")
b.value("micro F1 all same-patient excluded", "0.863", R4A / "summary.csv", "c1b/patient_excluded_k5/f1/micro.point",
        float(sv("patient_excluded_k5", "micro").point), 3, LG, "micro F1 decreased from 0.875 to 0.863")
b.value("ROI slides", "2778", R4A / "summary.csv", "c1b n_queries", str(int(sv("patient_excluded_k5", "macro").n_queries)), exact=True,
        doc=MS, needle="(2,778 slides, 1,876 patients)")
b.value("ROI patients", "1876", R4A / "summary.csv", "c1b n_patients", str(int(sv("patient_excluded_k5", "macro").n_patients)), exact=True,
        doc=MS, needle="(2,778 slides, 1,876 patients)")
b.value("k = 10 macro F1 query-slide-only", "0.755", R4A / "summary.csv", "c1b/published_k10/f1/macro.point",
        float(sv("published_k10", "macro").point), 3, ES, "macro F1 is 0.755 versus 0.737")
b.value("k = 10 macro F1 all same-patient excluded", "0.737", R4A / "summary.csv", "c1b/patient_excluded_k10/f1/macro.point",
        float(sv("patient_excluded_k10", "macro").point), 3, ES, "macro F1 is 0.755 versus 0.737")
r_ = pv("patient_excluded_k10", "macro")
for lab, pr, f in (("k = 10 paired", "−0.018", "point"), ("k = 10 paired lower", "−0.026", "ci_lo"), ("k = 10 paired upper", "−0.010", "ci_hi")):
    b.value(lab, pr, R4A / "paired_differences.csv", f"c1b/patient_excluded_k10−published_k10/f1/macro.{f}", float(r_[f]), 3,
            ES, "[−0.026, −0.010], with no backfill beyond the stored ten")
b.value("largest per-group decreases", "MRT, MPNST and DSRCT", C6 / "paired_differences.csv", "top-3 point (k = 5, published − patient_excluded)",
        "MRT, MPNST and DSRCT" if set(top3) == {"MRT", "MPNST", "DSRCT"} else ", ".join(top3), exact=True,
        doc=LG, needle="The largest decreases in point estimates were in MRT, MPNST and DSRCT")
b.value("submitted ROI macro F1 slot (query slide only)", "0.79", R4A / "summary.csv", "c1b/published_k5/f1/macro.point",
        float(sv("published_k5", "macro").point), 2, ES, "submitted 0.79 slot")

# ------------------------------------------------------------------ caption
cap_text = (
    f"Supplementary Table {final} | {TITLE}. Exploratory evaluation of the similarity maps against pathologist tile annotations "
    "and the values behind ROI search (Fig. 6A). "
    "Block A (sheets A1, A2): six ROI-heatmap settings (L1, Euclidean or cosine distance between the query-ROI feature and reference "
    "UNI2 mosaic tiles × mean or minimum where mosaic projections overlap). Block B (B1–B5): 14 matched-cost (transport-weighted) "
    "map settings and 380 graph/clustering settings of the exact Optimal Transport plan. All 400 settings, the selection rule and the "
    "outcome rule were fixed before scoring. Annotations: tumor (class 1) versus pooled immune cells, necrosis, non-tumor and "
    "artifacts (classes 2–5); unannotated locations excluded. References had at least 20 tumor and 20 non-tumor annotated locations "
    "(sensitivity: at least ten of each; identical estimates). Reference patients were split by seeded, group-stratified assignment "
    "(seed 123) into a development half (24 patients), on which each family's setting was selected, and a report half (22 patients), "
    "on which the selected and submitted settings were compared; a selected setting replaced the submitted one only if the paired "
    "report-half interval lay above zero. Natural pairs (P1) are top-ten slide retrievals after same-patient exclusion, with query "
    "features averaged over up to eight annotated or predicted tumor tiles; constructed pairs (P2) are ten same-group and ten "
    "other-group annotated queries per reference. For graph settings the class endpoint is transport-link class agreement "
    "(the share of annotated-to-annotated transported mass joining the same class) minus composition-based chance; it is invariant "
    "to partitioning a fixed graph and is not cluster purity, which is shown beside it with NMI and cluster counts. "
    "Block C (C1, C2): retained 44-reference comparison of the submitted L1 heatmap with constructed averaged-tumor queries. "
    "Block D (D1–D4): slide-level plurality F1 of aggregated ROI hits over 2,778 slides from 1,876 patients and 15 tumor groups, "
    "excluding only the query slide (submitted aggregation) or all slides of the query's patient (reported in Fig. 6A), with five "
    "hits per ROI and a ten-hit sensitivity. Intervals are nominal percentile 95% intervals from 5,000 whole-patient resamples "
    "(PCG64 seed 123; reference patients in blocks A–C), paired for contrasts, unadjusted for multiplicity and conditional on the "
    "fixed query draws, features and annotations. Map results quantify overlap with annotated tissue, not diagnostic accuracy, "
    "tumor segmentation or pathologist-assessed usefulness.")
footnotes = " ".join([
    "(a) Mean and minimum overlap aggregation gave identical values for every distance, because each annotated location has one "
    "owning mosaic tile; cosine/mean was selected over cosine/minimum by grid order.",
    "(b) Fig. 6B retains the submitted L1 display; the cosine setting was not used to redraw it.",
    "(c) Report half, P1: 16 reference patients with same-group and 9 with other-group natural pairs; 204 of 210 P1 query slides "
    "were unannotated and used the patient-held-out tumor classifier's predicted-tumor tiles. P2: 22 reference patients. The "
    "other-group control reversed direction on the development half (cosine 0.579 versus L1 0.680, 17 references) and is unstable "
    "at these sizes.",
    "(d) Matched-cost map: the submitted mean-cost setting is reported (0.65 [0.52, 0.78]); every development setting lay between "
    f"{devB.min():.3f} and {devB.max():.3f}.",
    "(e) Graph settings: 51 settings were degenerate on development (one cluster or more than half singleton clusters on same-group "
    "pairs) and were not eligible for selection; the submitted full-plan Louvain setting (resolution 4×10⁻⁵) was degenerate on 27% "
    "of development same-group pairs. Louvain calls not terminating within 30 s were retained as failed setting–pair rows and "
    f"excluded from that setting's means ({len(ntu)} of {n_plans * 380:,} plan×setting rows; {ntu.setting.nunique()} settings; "
    f"{ntu[['reference', 'query']].drop_duplicates().shape[0]} pairs). The selection criterion does not depend on the partition, so "
    "20 settings tied and threshold 0.6 with connected components was chosen by tie-break (purity, then grid order); this choice "
    "carries no information about granularity. Seeded Louvain depends on node order.",
    "(f) Block C: label 0 excluded; ten same-group and ten non-NORMAL other-group query slides per reference drawn with replacement "
    "(seed 123), each query feature averaging up to eight tumor-labelled tiles drawn without replacement; 35 of 880 draws used "
    "fewer than eight tiles (24 same-group, 11 other-group). Pair scores were averaged within reference, then across references; "
    "44 whole-reference-patient resampling units. The tumor share is the per-reference average (0.683), not the pooled "
    "location-weighted share (0.707). Reference patients and source queries are not a new independent clinical test set.",
    "(g) Block D: each ROI contributed up to five hits from its stored ten in stored order; same-patient removal preceded selection "
    "of the first five remaining hits, with no backfill beyond the stored ten (also at k = 10). Every hit cast one unweighted vote; "
    "MISC or missing-label hits occupied their rank slot without a vote; ties were broken alphabetically. Sheet D2 gives "
    "patient-excluded minus query-slide-only contrasts; sheet D4 keeps the source sign (query-slide-only minus patient-excluded).",
])
sources = "; ".join(f"{k} (sha256 {v['sha256'][:16]}…)" for k, v in sorted(b.inputs.items()) if not k.startswith("docs/"))
caption = [
    ("workspace_id", WID),
    ("final_number", f"Supplementary Table {final} (scripts/figures/supp_table_numbering.json; applied at assembly)"),
    ("title", TITLE),
    ("caption", cap_text),
    ("footnotes", footnotes),
    ("sheets", "A1 heatmap development grid; A2 heatmap report half; B1 cost-map development grid; B2 cost-map report half; "
               "B3 graph development grid; B4 graph report half; B5 settings summary (selection, outcome rule, degenerate and failed "
               "settings); C1 44-reference endpoints; C2 44-reference per-reference values and coverage; D1 ROI aggregation overall; "
               "D2 overall paired contrasts; D3 per-group F1; D4 per-group paired contrasts. Unrounded values: <stem>__<sheet>_raw.csv."),
    ("caption_status", "DRAFT drafted by the builder from the edit-sheet spec (R3.5-T5 + 2026-10-29 addendum) and the current "
                       "legends file; the lead's Supplementary Information Markdown is the source of truth for caption wording."),
    ("sources", sources),
    ("built_by/date", f"{BUILT_BY}; {BUILD_DATE}; builder {rel(b.script)}"),
]

sheets = [("A1_heatmap_dev_grid", A1, A1r), ("A2_heatmap_report_half", A2, A2r),
          ("B1_costmap_dev_grid", B1, B1r), ("B2_costmap_report_half", B2, B2r),
          ("B3_graph_dev_grid", B3, B3r), ("B4_graph_report_half", B4, B4r), ("B5_settings_summary", B5, None),
          ("C1_annot44_endpoints", C1, C1r), ("C2_annot44_per_reference", C2, C2r),
          ("D1_roi_overall", D1, D1r), ("D2_roi_overall_paired", D2, D2r), ("D3_roi_per_group", D3, pg),
          ("D4_roi_per_group_paired", D4, pgd)]
xlsx = b.write_workbook(f"Table_{WID}", caption, sheets)
b.check("workbook written with caption first and 13 data sheets", xlsx.exists())

# ------------------------------------------------------------------ notes, open items, report
b.note("Block letters A–D are this workbook's blocks; the L1 study's internal family codes (A heatmap, B matched-cost map, "
       "C graph/cluster) appear only in the 'setting key' columns.")
b.note("Natural annotated subset (P1 ∩ P2) is a single pair (amendment 002 had expected ≈41); it is descriptive only and is not tabulated "
       f"(report_estimates.json natural_annotated_subset keys: {list(rep.get('natural_annotated_subset', {}))[:4]}).")
b.note("C2 drops the reference patient code (restricted identifier); the internal reference slide index and tumour group are kept.")
b.note("D3/D4 per-group intervals are those already in C6 run_001 (Fig. 6A source); the 2026-10-21 '[PENDING Fig. 6A per-group "
       "intervals]' legend marker is superseded by C6 per the R2.5-T5/R4.a-T3 record.")
b.note("Re-tabulations performed by the builder (counting only): non-terminated Louvain rows (per_pair_C.csv.gz), C4 pointing count, "
       "short-query draws, coverage and the location-weighted tumour share (per_pair_c4_localization_roi8.csv); each is cross-checked "
       "against a figure quoted in the lead reviews or the edit sheet.")
b.open_item("Caption and footnotes are builder drafts from the edit-sheet spec and the current legends file; final wording comes from the "
            "lead's Supplementary Information Markdown and Haoran's reading approval.")
b.open_item("The legends file still reads 'Supplementary Table S26 | ROI heat-map parameter study (pre-registered settings) and "
            "ROI-search values. [PENDING: caption]' and was being edited while this table was built; value-check line numbers refer "
            f"to the bound version (sha256 {sha256(LG)[:16]}…). Its block definitions (A heat-map study, B matched-cost map and "
            "graph/clustering, C 44-reference evaluation, D ROI aggregation) match this workbook's blocks. Rebuild after the legends "
            "settle to refresh locations.")
b.open_item("Supplementary Fig. S-R3.5-L1 (six rule-selected montages, s3/run_002) is a figure product and is not part of this workbook; "
            "its legend basis is s3/run_002/LEGEND.txt.")
b.open_item("Fig. 5 legend in the current manuscript ends 'Ranks exclude the query's patient (Supplementary Table S26)'; S26 (this "
            "table) holds no Fig. 5 rank data. The citation target should be checked by the lead (possible stale cross-reference).")
b.open_item("C2 per-reference rows (internal slide indices, tumour group, per-reference AUROC/coverage) derive from restricted "
            "annotation data; whether they are published or kept as controlled source data needs the author's access decision.")
b.open_item("The edit-sheet table row 'Tumor at highest-scoring annotated location 0.816' and the main-text C4 values use the "
            "44-reference paired population; the unpaired 45-reference other-group mean (0.583) is shown only as a labelled control.")
b.open_item("Historical edit-sheet candidate text (l. ~9100) prints the matched-cost lower bound as 0.53 and calls the ≈0.03 endpoint "
            "cluster/tissue agreement; the retained source gives 0.52 and the endpoint is transport-link class agreement (R2.5-T3 "
            "consistency corrections). This table follows the source; other occurrences are not reconciled here.")
b.write_bindings()
vc = b.write_value_checks()
spec = [f"`docs/manuscript_supplement_edits.md` (sha256 {sha256(ES)[:16]}…): '### R3.5-T5 — new Supplementary Table S-R3.5-L1 and "
        f"Supplementary Fig. S-R3.5-L1' ({b.locate(ES, '### R3.5-T5 — new Supplementary Table S-R3.5-L1')}) with its 'Paired ROI "
        "candidate 001 assembly addendum (2026-10-29, 16:05 CDT)' (ROI aggregation block; retained L1 annotation check, C4 table) and "
        "its 'Table:' line (development grid A 6 / B 14 / C 380, report half, split, outcome rule, degenerate flag).",
        f"Consistency corrections at R2.5-T3 ({b.locate(ES, 'supplement prose must call the ≈0.03 endpoint')}): link-agreement naming; "
        "0.52 lower bound; 0.030 [0.005, 0.058] = development-selected setting, 0.030 [0.006, 0.057] = full-plan setting.",
        f"R4.a-T3/R2.5-T5 2026-10-29 records ({b.locate(ES, 'retained for the S-R3.5-L1 assembly addendum below')}): C4 pointing/"
        "prevalence/control statistics and ten-hit aggregation sensitivity move to this table.",
        f"Current manuscript citations of Supplementary Table S26: `docs/prosper_revised/prosper_manuscript_revised.md` (sha256 "
        f"{sha256(MS)[:16]}…) and notes in `docs/prosper_revised/prosper_supplementary_legends.md` (sha256 {sha256(LG)[:16]}…)."]
lines = report_header(b, TITLE, final, spec) + [
    "", "## Workbook", "", f"`{xlsx.name}` — sheet `caption` (field | text) then 13 data sheets; each data sheet is also written as "
    f"`Table_{WID}__<sheet>.csv`, with unrounded source values in `Table_{WID}__<sheet>_raw.csv` (B5 has no raw counterpart; it is "
    "assembled from selection.json/outcome.json/report_estimates.json, which are bound)."] + report_tail(b)
(b.out / "BUILD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
b.write_sums()
n_fail_c = sum(not ok for _, ok, _ in b.checks)
print(f"{WID}: checks {len(b.checks) - n_fail_c}/{len(b.checks)} pass; values {int((vc.result == 'pass').sum())}/{len(vc)} pass")
sys.exit(1 if n_fail_c or (vc.result != "pass").any() else 0)
