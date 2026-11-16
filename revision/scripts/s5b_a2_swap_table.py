#!/usr/bin/env python3
"""S5b pilot — analyst precomputation of the A2 swap table (specification §3, F2).

For each of the 20 pilot queries: neighbour tumour groups from the deployed
loader (``analysis_lib.load_metadata``) joined on ``neighbor_weights.csv``;
plurality-tied set from the harness ``plurality()``; S drawn as
``numpy.random.default_rng([20261011, int(query)]).choice(candidates)`` with
candidates = sorted(G15 - {MISC, NORMAL} - A0 neighbour groups - {true group});
A2 probabilities = weights aggregated by relabelled group (6 dp, > 1e-8).

Writes ``a2_swap_table.csv`` (query, relabelled_ranks, S) and ``a2_swap_check.json``
to the key directory and compares every query with the independent reference
(``e6_check.json -> recheck_v2``). KEY MATERIAL: the exclusion rule uses the true
group; the output directory must stay independent only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

G15 = ["ACT/ACC", "EWS", "LIVER", "MEL", "MISC", "MPNST", "MRT", "NB", "NF", "NORMAL",
       "OS", "RB", "RMS", "SS", "WT"]
SEED0 = 20261011


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True, help="r3_5_s5b_posterior_20q_20261009_v2")
    ap.add_argument("--reference", type=Path, required=True, help="e6_check.json (recheck_v2)")
    ap.add_argument("--harness", type=Path, required=True, help="prosper-search repo root")
    ap.add_argument("--out", type=Path, required=True, help="key directory (lead/reviewer only)")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(a.harness))
    from prosper.evaluation import llm_grounding_cli as H  # noqa: E402
    analysis_lib, _ = H._import_pipeline("manuscript")
    meta = analysis_lib.load_metadata(H.RICH_META)
    assert set(meta["tumor_group"].dropna().unique()) <= set(G15), sorted(meta["tumor_group"].unique())
    g_of = meta.drop_duplicates("slide_id").set_index("slide_id")["tumor_group"].to_dict()

    qids = [l.strip() for l in (a.inputs / "harness_query_ids.txt").read_text().splitlines() if l.strip()]
    w = pd.read_csv(a.inputs / "neighbor_weights.csv", float_precision="round_trip",
                    dtype={"query": str, "neighbor_slide_id": str})
    key = pd.read_csv(a.inputs / "scoring_key_DO_NOT_SEND.csv", dtype={"query": str}).set_index("query")
    ref = json.load(open(a.reference))["recheck_v2"]
    ref = ref[next(k for k in ref if k.startswith("A2_swap_reference"))]

    rows, check, mismatches = [], {}, []
    for q in qids:
        nb = w[w["query"] == q].sort_values("rank")
        assert len(nb) == 5, (q, len(nb))
        groups = [g_of[s] for s in nb["neighbor_slide_id"]]
        assert all(groups), (q, groups)
        true = key.loc[q, "true_group"]
        _, tied = H.plurality(groups)
        cands = sorted(set(G15) - {"MISC", "NORMAL"} - set(groups) - {true})
        assert cands, q
        S = str(np.random.default_rng([SEED0, int(q)]).choice(cands))
        relabelled = [int(r) for r, g in zip(nb["rank"], groups) if g in tied]
        new_groups = [S if g in tied else g for g in groups]
        agg = {}
        for g, wt in zip(new_groups, nb["weight"]):
            agg[g] = agg.get(g, 0.0) + float(wt)
        probs = {g: round(v, 6) for g, v in agg.items() if round(v, 6) > 1e-8}
        probs = dict(sorted(probs.items(), key=lambda kv: (-kv[1], kv[0])))
        tot = sum(probs.values())
        assert abs(tot - 1) <= 5e-6, (q, tot)
        assert set(probs) == set(new_groups), (q, probs, new_groups)
        assert S != true and S not in groups
        r = ref.get(q)
        same = bool(r) and r["S"] == S and list(r["relabelled_ranks"]) == relabelled \
            and r["A2_probabilities"] == probs and r["n_candidates"] == len(cands)
        if not same:
            mismatches.append({"query": q, "lead": {"S": S, "ranks": relabelled, "probs": probs,
                                                    "n_candidates": len(cands)}, "reference": r})
        rows.append({"query": q, "relabelled_ranks": ";".join(map(str, relabelled)), "S": S})
        check[q] = {"groups": groups, "tied": tied, "S": S, "n_candidates": len(cands),
                    "relabelled_ranks": relabelled, "A2_probabilities": probs, "sum": tot,
                    "S_is_argmax": max(probs, key=probs.get) == S,
                    "true_in_A2_labels": true in new_groups, "matches_reference": same}

    a.out.mkdir(parents=True, exist_ok=True)
    tab = a.out / "a2_swap_table.csv"
    pd.DataFrame(rows).to_csv(tab, index=False)
    out = {"inputs": {p.name: sha(p) for p in [a.inputs / "harness_query_ids.txt",
                                                a.inputs / "neighbor_weights.csv",
                                                a.inputs / "scoring_key_DO_NOT_SEND.csv",
                                                a.reference, H.RICH_META, Path(H.__file__)]},
           "script_sha256": sha(Path(__file__)), "seed_rule": f"default_rng([{SEED0}, int(query)])",
           "n_queries": len(qids), "n_matching_reference": sum(v["matches_reference"] for v in check.values()),
           "conflicting_evidence(S_not_argmax)": sorted(q for q, v in check.items() if not v["S_is_argmax"]),
           "true_group_remains_in_A2": sorted(q for q, v in check.items() if v["true_in_A2_labels"]),
           "swap_table_sha256": sha(tab), "mismatches": mismatches, "per_query": check}
    (a.out / "a2_swap_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_queries", "n_matching_reference", "conflicting_evidence(S_not_argmax)",
                                          "true_group_remains_in_A2", "swap_table_sha256")}, indent=1))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
