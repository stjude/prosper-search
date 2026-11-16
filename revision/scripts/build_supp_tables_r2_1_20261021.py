#!/usr/bin/env python3
"""Assemble the supplementary-table sheets owed by the 2026-10-21 Fig. 2 / Fig. 4 decisions.

Table S-R2.1-E, sheet 5 (retrieval metrics, all systems) and sheet 6 (per group, k = 5), from the
Fig. 2 derivation `r2_1_fig2_retrieval_20261021/run_001/` (hit@k, precision@k, mAP@k; whole-patient
intervals) joined with mMV@5 (macro recall under the plurality rule) from RR-5
(`r2_1_virchow2_fp32_main_20261013/run_003/rr5_replacement_001/`) and, for the comparators, from
the R3.1 summaries (same pinned draws). Table S-R2.1-C, mMV@5 sheet, from
`r2_1_fig4a_mmv5_20261021/run_001/{deltas,identity}.csv`.

No value is computed here: every number is copied from an accepted or independently audited output and the
script asserts the joins (cell coverage, interval presence, the RR-5 = R3.1 identity on the main
configuration). Output: $PROSPER_OUT/supp_tables_20261021/ (CSV per sheet, one XLSX, README,
SHA256SUMS.txt). Reading-only on inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

OUT_ROOT = Path(os.environ.get("PROSPER_OUT") or Path.cwd() / "prosper_revision_output")
DER = OUT_ROOT / "r2_1_fig2_retrieval_20261021/run_001"
RR5 = OUT_ROOT / "r2_1_virchow2_fp32_main_20261013/run_003/rr5_replacement_001"
R31 = {
    "sish": OUT_ROOT / "r3_1_comparators_whole_patient_20261008",
    "prism_native": OUT_ROOT / "r3_1_prism_20261018/eval_native_20261019",
    "retccl": OUT_ROOT / "r3_1_retccl_whole_patient_20261016",
}
MMV = OUT_ROOT / "r2_1_fig4a_mmv5_20261021/run_001"
OUT = OUT_ROOT / "supp_tables_20261021"
R31_CFG = {"sish": ("sish", "sish"), "prism_native": ("prism_native", "prism_native"),
           "retccl_median": ("retccl", "retccl_median"), "retccl_ot": ("retccl", "retccl_ot"),
           "retccl_yottixel": ("retccl", "retccl_yottixel"), "retccl_sampler": ("retccl", "retccl_sampler"),
           "resnet50": ("sish", "resnet50")}
METRIC_NAME = {"hit_at_k": "hit@k", "precision_at_k": "precision@k", "map_at_k": "mAP@k"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {}

    # ---------------- Sheet 5: retrieval metrics (long format) ----------------
    s = pd.read_csv(DER / "summary.csv"); inputs[str(DER / "summary.csv")] = sha(DER / "summary.csv")
    assert s.cell_id.nunique() == 44 and set(s.k) == {1, 3, 5, 10}
    assert s[["point", "ci_lo", "ci_hi"]].notna().all().all(), "every interval must be reportable"
    s5 = s.rename(columns={"ci_lo": "ci_low", "ci_hi": "ci_high"}).copy()
    s5["metric"] = s5.metric.map(METRIC_NAME)
    s5 = s5[s5.metric.notna()]
    s5 = s5[~((s5.metric == "mAP@k") & (~s5.k.isin([5, 10])))]
    # mMV@5 (macro) rows
    t = pd.read_csv(RR5 / "table_s3_revised.csv"); inputs[str(RR5 / "table_s3_revised.csv")] = sha(RR5 / "table_s3_revised.csv")
    t = t[(t.k == 5) & (t.rule == "plurality") & (t.tie_order == "canonical") & (t.metric == "macro_recall")
          & (t.cell_id != "titan/titan_published")]
    assert len(t) == 37 and (t.ci_status == "ok").all()
    mv = t[["cell_id", "value", "ci_low", "ci_high"]].rename(columns={"value": "point"})
    rows = []
    for cid, (key, cfg) in R31_CFG.items():
        f = R31[key] / "summary.csv"; inputs[str(f)] = sha(f)
        d = pd.read_csv(f)
        d = d[(d.population == "all_tissue_2778") & (d.config == cfg) & (d.metric == "plurality5") & (d.average == "macro")]
        assert len(d) == 1, cid
        rows.append(dict(cell_id=cid, point=d.point.iloc[0], ci_low=d.ci_lo.iloc[0], ci_high=d.ci_hi.iloc[0]))
    d = pd.read_csv(R31["sish"] / "summary.csv")
    d = d[(d.population == "all_tissue_2778") & (d.config == "primary") & (d.metric == "plurality5") & (d.average == "macro")].iloc[0]
    u = mv[mv.cell_id == "uni2/ot_cosine"].iloc[0]
    assert abs(d.point - u.point) < 1e-6 and abs(d.ci_lo - u.ci_low) < 1e-6 and abs(d.ci_hi - u.ci_high) < 1e-6
    mv = pd.concat([mv, pd.DataFrame(rows)], ignore_index=True)
    meta = s.drop_duplicates("cell_id")[["cell_id", "system", "aggregation", "kind", "lineage", "n_queries", "n_patients", "n_replicates", "n_padded"]]
    mv = mv.merge(meta, on="cell_id", how="left")
    assert mv.system.notna().all() and len(mv) == 44, len(mv)  # 37 grid + TITAN... (TITAN is in RR-5) + 6 comparators + floor
    mv["k"] = 5; mv["metric"] = "mMV@k"; mv["average"] = "macro"
    s5 = pd.concat([s5, mv[s5.columns]], ignore_index=True)
    s5 = s5.sort_values(["kind", "system", "aggregation", "metric", "k", "average"], kind="stable")
    s5.insert(0, "sheet", "S-R2.1-E sheet 5")
    s5["ci"] = "95 % whole-patient percentile bootstrap (5,000 pinned draws, 1,876 patients)"

    # ---------------- Sheet 6: per group, k = 5 ----------------
    g = pd.read_csv(DER / "per_group.csv"); inputs[str(DER / "per_group.csv")] = sha(DER / "per_group.csv")
    g6 = g.rename(columns={"ci_lo": "ci_low", "ci_hi": "ci_high"}).copy()
    g6["metric"] = g6.metric.map({"hit_at_k": "hit@5", "precision_at_k": "precision@5", "map_at_k": "AP@5"})
    assert g6.metric.notna().all() and set(g6.k) == {5}
    pc = pd.read_csv(RR5 / "per_class.csv"); inputs[str(RR5 / "per_class.csv")] = sha(RR5 / "per_class.csv")
    pc = pc[(pc.k == 5) & (pc.rule == "plurality") & (pc.tie_order == "canonical") & (pc.metric == "recall")
            & (pc.cell_id != "titan/titan_published")]
    assert (pc.ci_status == "ok").all() and pc.cell_id.nunique() == 37 and len(pc) == 37 * 15
    pc = pc.rename(columns={"value": "point"})[["cell_id", "group", "point", "ci_low", "ci_high"]]
    rows = []
    for cid, (key, cfg) in R31_CFG.items():
        f = R31[key] / "per_group.csv"; inputs[str(f)] = sha(f)
        d = pd.read_csv(f)
        d = d[(d.population == "all_tissue_2778") & (d.config == cfg) & (d.metric == "plurality5")]
        assert len(d) == 15, (cid, len(d))
        for _, r in d.iterrows():
            rows.append(dict(cell_id=cid, group=r.group, point=r.point, ci_low=r.ci_lo, ci_high=r.ci_hi))
    pc = pd.concat([pc, pd.DataFrame(rows)], ignore_index=True)
    nq = g6.drop_duplicates(["cell_id", "group"])[["cell_id", "group", "n_queries"]]
    pc = pc.merge(nq, on=["cell_id", "group"], how="left").merge(meta[["cell_id", "system", "aggregation", "kind"]], on="cell_id", how="left")
    assert pc.system.notna().all() and pc.n_queries.notna().all() and len(pc) == 44 * 15
    pc["k"] = 5; pc["metric"] = "MV@5"
    g6 = pd.concat([g6, pc[g6.columns]], ignore_index=True)
    g6 = g6.sort_values(["kind", "system", "aggregation", "group", "metric"], kind="stable")
    g6.insert(0, "sheet", "S-R2.1-E sheet 6")

    # ---------------- S-R2.1-C: mMV@5 sheet ----------------
    dl = pd.read_csv(MMV / "deltas.csv"); inputs[str(MMV / "deltas.csv")] = sha(MMV / "deltas.csv")
    idn = pd.read_csv(MMV / "identity.csv"); inputs[str(MMV / "identity.csv")] = sha(MMV / "identity.csv")
    gates = json.load(open(MMV / "gates.json")); assert gates["all_pass"]
    c = dl[dl.metric == "mmv"].copy()
    assert len(c) == 3 * 12 * 16 and (c.point_status == "estimable").all() and (c.ci_status == "estimable").all()
    c["metric"] = "mMV@5 (group rows: MV@5)"
    c = c.rename(columns={"delta": "delta_perturbed_minus_identity"})
    c.insert(0, "sheet", "S-R2.1-C mMV@5")
    pipe_name = {"hopt": "H-Optimus-1 + SAMPLER (fp16 autocast)", "uni2_sampler": "UNI2 + SAMPLER (fp32)", "virchow2_sampler": "Virchow2 + SAMPLER (fp32)"}
    c.insert(2, "pipeline_label", c.pipeline.map(pipe_name))
    i = idn[idn.metric == "mmv"].copy(); i["metric"] = "mMV@5 (group rows: MV@5)"; i.insert(0, "sheet", "S-R2.1-C mMV@5 identity")
    i.insert(2, "pipeline_label", i.pipeline.map(pipe_name))

    # ---------------- write ----------------
    s5.to_csv(OUT / "S-R2.1-E_sheet5_retrieval_metrics.csv", index=False)
    g6.to_csv(OUT / "S-R2.1-E_sheet6_per_group_k5.csv", index=False)
    c.to_csv(OUT / "S-R2.1-C_mmv5_deltas.csv", index=False)
    i.to_csv(OUT / "S-R2.1-C_mmv5_identity.csv", index=False)
    notes = pd.DataFrame({"note": [
        "S-R2.1-E sheet 5. hit@k: at least one same-group other-patient slide among the k nearest neighbours; precision@k: share of the k nearest that are same-group; "
        "mAP@k: mean over queries of AP@k, where AP@k normalises by min(k, number of same-group other-patient slides in the library) [footnote 6: the main configuration's values equal Table S25, 0.657 and 0.604 macro at k = 5 and 10]; "
        "mMV@k: the query's label is the plurality label of its k nearest neighbours (ties toward the closest neighbour), scored as macro recall over the 15 groups (majority-vote accuracy). "
        "micro = over the 2,778 queries; macro = unweighted mean over the 15 groups. Intervals: 95 % whole-patient percentile bootstrap, 5,000 pinned draws over 1,876 patients, shared by every system.",
        "Footnote 7: SISH returned fewer than ten neighbours for 35 queries (206 positions); padded positions are scored as non-relevant, as in Supplementary Table S-R3.1.",
        "Systems: 37 encoder x aggregation cells (nine tile encoders x four aggregations, Virchow2 from the fp32 feature bank, plus CONCH v1.5 + TITAN with the repaired embeddings), RetCCL x four aggregations, PRISM (native tiling), SISH, and the ImageNet ResNet-50 + median-pooling floor. "
        "Sources: hit@k / precision@k / mAP@k from r2_1_fig2_retrieval_20261021/run_001 (gated against RR-5 and R3.1); mMV@5 from RR-5 (table_s3_revised.csv, per_class.csv) and the R3.1 summaries (plurality5).",
        "S-R2.1-E sheet 6. Per group at k = 5: hit@5, precision@5, AP@5 (as above) and MV@5 (share of the group's queries whose plurality label is correct) with intervals; n_queries = group size among the 2,778 queries.",
        "S-R2.1-C mMV@5 sheet. Paired change (perturbed minus unperturbed) in MV@5 per group and in mMV@5 (MACRO) for three SAMPLER pipelines under twelve simulated query perturbations; "
        "same per-query vote rule as Fig. 2; identical 5,000 whole-patient draws for every pipeline and arm; intervals unadjusted for multiplicity. Derived 2026-10-21 (scripts/r2_1_fig4a_mmv5.py, gates G1-G4 pass) from the accepted perturbation ranked lists; "
        "lead distinct-route check 2026-10-21 (r2_1_fig4a_mmv5_20261021/audit_001): own vote and own bootstrap reproduce every row on the pinned draws to 1e-15. The identity sheet gives the unperturbed MV@5 / mMV@5 values.",
    ]})
    with pd.ExcelWriter(OUT / "Tables_S-R2.1-E_sheets5-6_and_S-R2.1-C_mmv5.xlsx", engine="openpyxl") as xw:
        notes.to_excel(xw, sheet_name="Notes", index=False)
        s5.to_excel(xw, sheet_name="S-R2.1-E s5 retrieval", index=False)
        g6.to_excel(xw, sheet_name="S-R2.1-E s6 per-group k5", index=False)
        c.to_excel(xw, sheet_name="S-R2.1-C mMV5 deltas", index=False)
        i.to_excel(xw, sheet_name="S-R2.1-C mMV5 identity", index=False)
    json.dump({"inputs_sha256": inputs, "rows": {"sheet5": len(s5), "sheet6": len(g6), "c_deltas": len(c), "c_identity": len(i)}},
              open(OUT / "input_bindings.json", "w"), indent=1)
    (OUT / "README.md").write_text(
        "# Supplementary-table sheets assembled 2026-10-21 (lead)\n\n"
        "Built by `scripts/build_supp_tables_r2_1_20261021.py`; values copied from accepted/lead-audited outputs (see input_bindings.json), nothing recomputed.\n\n"
        f"- S-R2.1-E sheet 5: {len(s5)} rows (44 systems; hit@k, precision@k at k = 1/3/5/10, mAP@5/@10, mMV@5; micro/macro; intervals).\n"
        f"- S-R2.1-E sheet 6: {len(g6)} rows (44 systems x 15 groups x hit@5 / precision@5 / AP@5 / MV@5).\n"
        f"- S-R2.1-C mMV@5: {len(c)} delta rows (3 pipelines x 12 arms x 15 groups + MACRO) and {len(i)} identity rows.\n\n"
        "Candidates for Haoran's reading; numbering (S-R2.1-E = provisional S35, S-R2.1-C = provisional S34) is assigned at assembly.\n")
    lines = []
    for p in sorted(OUT.iterdir()):
        if p.name != "SHA256SUMS.txt":
            lines.append(f"{sha(p)}  {p.name}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(f"sheet5 {len(s5)} rows; sheet6 {len(g6)} rows; C deltas {len(c)}; identity {len(i)} -> {OUT}")


if __name__ == "__main__":
    main()
