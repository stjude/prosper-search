#!/usr/bin/env python3
"""Frozen/OCT CCDI-MCI queries: hub-slide collapse by aggregation (revision, 2026-11-10).

Read-only re-tabulation of the accepted R1.2 scope-correction outputs
(`$PROSPER_OUT/r1_2_scope_correction_20260930/`, LSF 324464955; [decision record]). No retrieval is
recomputed. For the clear-scope all-preparation population (735 queries) the ranked top-10
lists of the six configurations are split by `preparation_status` and, for the 201 frozen/OCT
queries against the 524 verified-FFPE queries, the script counts how many distinct reference
slides fill the top-5 positions and which reference slides sit in the top-5 of most queries
("hub" slides), broken down by the query's tumor group.

Outputs (`$PROSPER_OUT/ccdi_frozen_hub_20261110/`, refused if non-empty):
  hub_summary.csv     one row per configuration x stratum (FROZEN_OCT, VERIFIED_FFPE): n, micro
                      and macro hit@5, hit@1, hit@10, distinct reference slides in the top-5
                      positions, share of queries whose top-5 contains the most frequent slide
  hub_slides.csv      per configuration x stratum: the ten reference slides most often in the
                      top-5, their label, number of queries, and per query-group counts
  per_class_hit5.csv  per configuration x stratum x query group: n, hit@5, generous vote k=5
  provenance.json     input paths and sha256, counts, script sha256

Usage:  PROSPER_OUT=... python3 scripts/ccdi_frozen_hub_tabulation.py
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CONFIGS = ["whole_slide_ot", "yottixel", "query_tumor_ot", "both_tumor_ot", "median_pooling", "sampler"]
STRATA = ["FROZEN_OCT", "VERIFIED_FFPE"]
TOP_K = 5
N_HUBS = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    out_root = os.environ.get("PROSPER_OUT")
    if not out_root:
        sys.exit("export PROSPER_OUT first")
    src = Path(out_root) / "r1_2_scope_correction_20260930"
    dst = Path(out_root) / "ccdi_frozen_hub_20261110"
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"refusing to write into non-empty {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    inputs = {}
    manifest_path = src / "manifest_allprep_scope_clear.csv"
    manifest = pd.read_csv(manifest_path)
    inputs[str(manifest_path)] = sha256(manifest_path)
    assert manifest["slide_id"].is_unique
    label = manifest.set_index("slide_id")["tumor_group"]
    prep = manifest.set_index("slide_id")["preparation_status"]
    assert len(manifest) == 735, len(manifest)
    assert (prep == "FROZEN_OCT").sum() == 201 and (prep == "VERIFIED_FFPE").sum() == 524

    summary_rows, hub_rows, class_rows = [], [], []
    for cfg in CONFIGS:
        pq_path = src / f"{cfg}_allprep_scope_clear_per_query.csv"
        rk_path = src / f"{cfg}_allprep_scope_clear_ranked.csv.gz"
        inputs[str(pq_path)] = sha256(pq_path)
        inputs[str(rk_path)] = sha256(rk_path)
        pq = pd.read_csv(pq_path)
        rk = pd.read_csv(rk_path)
        assert set(pq["query"]) == set(manifest["slide_id"])
        pq["preparation_status"] = pq["query"].map(prep)
        rk["preparation_status"] = rk["query"].map(prep)
        rk["query_group"] = rk["query"].map(label)
        top = rk[rk["rank"] <= TOP_K]
        for stratum in STRATA:
            f = pq[pq["preparation_status"] == stratum]
            t = top[top["preparation_status"] == stratum]
            n = len(f)
            assert t["query"].nunique() == n and len(t) == n * TOP_K
            per_class = f.groupby("true_label").agg(
                n=("hit_k5", "size"), hit_at_5=("hit_k5", "mean"),
                generous_vote_accuracy_k5=("generous_correct_k5", "mean"))
            counts = t.groupby("neighbour")["query"].nunique().sort_values(ascending=False)
            summary_rows.append({
                "configuration": cfg, "preparation": stratum, "n_queries": n,
                "n_query_groups": int(f["true_label"].nunique()),
                "micro_hit_at_5": f["hit_k5"].mean(), "macro_hit_at_5": per_class["hit_at_5"].mean(),
                "hit_at_1": f["hit_k1"].mean(), "hit_at_10": f["hit_k10"].mean(),
                "generous_vote_accuracy_k5": f["generous_correct_k5"].mean(),
                "top5_positions": n * TOP_K,
                "distinct_reference_slides_in_top5": int(t["neighbour"].nunique()),
                "most_frequent_slide": int(counts.index[0]),
                "most_frequent_slide_label": t.loc[t["neighbour"] == counts.index[0], "neighbour_label"].iloc[0],
                "most_frequent_slide_n_queries": int(counts.iloc[0]),
                "most_frequent_slide_share": counts.iloc[0] / n,
            })
            for slide, n_q in counts.head(N_HUBS).items():
                sub = t[t["neighbour"] == slide]
                by_group = sub.groupby("query_group")["query"].nunique()
                row = {"configuration": cfg, "preparation": stratum, "reference_slide": int(slide),
                       "reference_label": sub["neighbour_label"].iloc[0], "n_queries_with_slide_in_top5": int(n_q),
                       "share_of_stratum": n_q / n}
                for g in sorted(label.unique()):
                    row[f"q_{g}"] = int(by_group.get(g, 0))
                hub_rows.append(row)
            for g, r in per_class.iterrows():
                class_rows.append({"configuration": cfg, "preparation": stratum, "query_group": g,
                                   "n": int(r["n"]), "hit_at_5": r["hit_at_5"],
                                   "generous_vote_accuracy_k5": r["generous_vote_accuracy_k5"]})

    pd.DataFrame(summary_rows).to_csv(dst / "hub_summary.csv", index=False)
    pd.DataFrame(hub_rows).to_csv(dst / "hub_slides.csv", index=False)
    pd.DataFrame(class_rows).to_csv(dst / "per_class_hit5.csv", index=False)
    prov = {
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_directory": str(src), "source_job": "LSF 324464955 (ADR 0018, 2026-09-30)",
        "inputs_sha256": inputs, "population": "allprep_scope_clear (735)", "strata": STRATA,
        "top_k": TOP_K, "n_hubs_listed": N_HUBS, "configurations": CONFIGS,
        "note": "Descriptive re-tabulation of accepted outputs; no retrieval recomputed, no intervals, post-result stratum.",
    }
    (dst / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(pd.DataFrame(summary_rows).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
