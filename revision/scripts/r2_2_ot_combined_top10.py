#!/usr/bin/env python3
"""R2.2 OT re-solve: combined top-10 substitution across re-solve tiers (pre-exec review C5).

Each tier's `summarize` substitutes only its own CAPPED values. This step substitutes the
corrections of all given tiers into ONE copy of the saved `_no_same_patient` matrix and
reports top-10 changes for the saved 2,750-query population, under three scenarios:

  A  CAPPED high-solve values of all tiers                      (the correction)
  C  A + NON_REPRODUCED high-solve values of all tiers            (output-review scenario C)
  D  C + BIT_EQUAL_CODE_* rows, i.e. every non-CONVERGED direction with a converged
     high-solve value (CONVERGED directions are bit-equal no-op substitutions). D is
     value-equivalent to the OT-30 output review's scenario D but its row counts and
     margins are not comparable to that review's figures (7,470 / 1.34e-4), which
     counted every direction including no-ops.

Checks before any arithmetic (C5 a-d, h): tier artifact hashes unchanged when expected
hashes are given (full 64-hex or >= 16-hex prefix); each tier's non-converged file is
self-consistent with its summary.json (pairs hash, row/class counts, membership in the
tier's directed set, no duplicates); all tiers cite the same matrix/ranked-list hashes;
directed sets disjoint with the expected union size; every CAPPED row has high_code == 1
and delta > 0; any non-CONVERGED row without a converged high solve is an anomaly that
stops the run; every substituted matrix entry equals the row's stored_hex value. The
original top-10 is reconstructed from the matrix and must equal the saved ranked lists for
every query (the same tie rule as the re-solve tool, imported from it; no `prosper` import).

Login node, `OMP_NUM_THREADS=1`, CPU-minutes. Writes `<out>/combined_top10.json` and
`<out>/combined_top10_<scenario>.csv`; never modifies tier directories or the matrix.
"""
import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ot_resolve", HERE / "r2_2_ot30_resolve.py")
ot_resolve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ot_resolve)
_topk, MASK_VALUE, TOPK = ot_resolve._topk, ot_resolve.MASK_VALUE, ot_resolve.TOPK


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tier(d: Path, expect: dict) -> dict:
    if not (d / "COMPLETE").exists():
        sys.exit(f"{d}: no COMPLETE marker")
    hashes = {f: sha256(d / f) for f in ("capped_solves.csv", "summary.json", "non_converged_or_non_reproduced.csv", "pairs.csv")}
    for f, want in expect.items():
        if not hashes[f].startswith(want):
            sys.exit(f"{d}/{f}: sha256 {hashes[f][:16]} does not match expected {want}")
    summ = json.load(open(d / "summary.json"))
    nonconv = pd.read_csv(d / "non_converged_or_non_reproduced.csv", dtype={"query": str, "other": str},
                          float_precision="round_trip", keep_default_na=False)
    pairs = pd.read_csv(d / "pairs.csv", dtype={"slide_a": str, "slide_b": str})
    directed = {(a, b) for a, b in zip(pairs["slide_a"], pairs["slide_b"])} | {(b, a) for a, b in zip(pairs["slide_a"], pairs["slide_b"])}
    if len(directed) != 2 * len(pairs) or len(directed) != summ["directed_solves"]:
        sys.exit(f"{d}: directed set size {len(directed)} inconsistent with pairs/summary")
    # X1: self-consistency of the non-converged file against summary.json
    if hashes["pairs.csv"] != summ["pairs_sha256"]:
        sys.exit(f"{d}: pairs.csv hash {hashes['pairs.csv'][:16]} != summary pairs_sha256")
    cc = {k: int(v) for k, v in summ["classification_counts"].items() if k != "CONVERGED"}
    if len(nonconv) != summ["directed_solves"] - int(summ["classification_counts"].get("CONVERGED", 0)):
        sys.exit(f"{d}: {len(nonconv)} non-converged rows vs summary {summ['directed_solves']} - CONVERGED")
    if {k: int(v) for k, v in nonconv["classification"].value_counts().items()} != cc:
        sys.exit(f"{d}: classification counts of non-converged file differ from summary")
    if int((nonconv["classification"] == "NON_REPRODUCED").sum()) != summ["non_reproduced_directed"]:
        sys.exit(f"{d}: NON_REPRODUCED count differs from summary")
    keys = list(zip(nonconv["query"], nonconv["other"]))
    if len(keys) != len(set(keys)) or not set(keys) <= directed:
        sys.exit(f"{d}: non-converged rows duplicated or outside the tier's directed set")
    capped = nonconv[nonconv["classification"] == "CAPPED"].copy()
    if len(capped) != summ["capped_directed"]:
        sys.exit(f"{d}: {len(capped)} CAPPED rows vs summary {summ['capped_directed']}")
    if len(capped):
        if (capped["high_code"].astype(str) != "1").any():
            sys.exit(f"{d}: CAPPED row with high_code != 1")
        if (capped["delta_stored_minus_high"].astype(float) <= 0).any():
            sys.exit(f"{d}: CAPPED row with non-positive delta")
    # X2: a non-CONVERGED row without a converged high solve is an anomaly, not a value to drop
    bad = nonconv[(nonconv["high_cost"].astype(str) == "") | (nonconv["high_code"].astype(str) != "1")]
    if len(bad):
        sys.exit(f"{d}: {len(bad)} non-CONVERGED rows without a converged high solve (ANOMALY): "
                 + " ".join(f"{q}->{o}" for q, o in zip(bad["query"], bad["other"])))
    for hc, hx in zip(nonconv["high_cost"], nonconv["high_cost_hex"]):
        if float(hc) != float.fromhex(hx):
            sys.exit(f"{d}: high_cost decimal/hex mismatch")
    nonrep = nonconv[nonconv["classification"] == "NON_REPRODUCED"]
    return {"dir": str(d), "hashes": hashes, "summary": summ, "directed": directed, "capped": capped,
            "nonrep": nonrep, "resolved": nonconv.copy(), "n_nonrep": int(len(nonrep)),
            "composition": {k: int(v) for k, v in nonconv["classification"].value_counts().items()},
            "rows_excluded_high_not_converged": 0}


def substitute(sub, pos, rows):
    corr = sub.copy()
    n = 0
    skipped = {"query_not_in_population": 0, "other_not_in_population": 0, "masked_entry": 0}
    for q, o, hc, sx in zip(rows["query"], rows["other"], rows["high_cost"], rows["stored_hex"]):
        if q not in pos:
            skipped["query_not_in_population"] += 1; continue
        if o not in pos:
            skipped["other_not_in_population"] += 1; continue
        if sub[pos[q], pos[o]] >= MASK_VALUE:
            skipped["masked_entry"] += 1; continue
        # X3: the matrix entry being replaced must be the row's stored value
        if sub[pos[q], pos[o]] != float.fromhex(sx):
            sys.exit(f"stored-value mismatch at {q}->{o}: matrix {sub[pos[q], pos[o]]!r} vs stored_hex {sx}")
        corr[pos[q], pos[o]] = float(hc); n += 1
    return corr, n, skipped


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiers", nargs="+", required=True, help="tier output directories (each with COMPLETE)")
    p.add_argument("--expect", action="append", default=[],
                   help="TIERINDEX:FILE:SHA256PREFIX, e.g. 0:capped_solves.csv:b0183844 (repeatable)")
    p.add_argument("--expect-directions", type=int, required=True, help="expected size of the union of directed sets")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    hashed_files = ("capped_solves.csv", "summary.json", "non_converged_or_non_reproduced.csv", "pairs.csv")
    expects = {}
    for e in a.expect:
        try:
            i, f, h = e.split(":", 2); i = int(i)
        except ValueError:
            sys.exit(f"bad --expect {e!r}; use TIERINDEX:FILE:SHA256")
        if not 0 <= i < len(a.tiers):
            sys.exit(f"--expect tier index {i} out of range for {len(a.tiers)} tiers")
        if f not in hashed_files:
            sys.exit(f"--expect file {f!r} is not one of {hashed_files}")
        if len(h) < 16 or len(h) > 64 or any(c not in "0123456789abcdef" for c in h):
            sys.exit(f"--expect hash {h!r} must be 16-64 lowercase hex characters")
        expects.setdefault(i, {})[f] = h
    out = Path(a.out)
    if out.exists():
        sys.exit(f"--out {out} exists; refusing to overwrite")
    if any(out.resolve() == Path(d).resolve() for d in a.tiers):
        sys.exit("--out must not be a tier directory")
    tiers = [load_tier(Path(d), expects.get(i, {})) for i, d in enumerate(a.tiers)]

    # (b) same matrix / ranked lists; (c) disjoint union
    mats = {t["summary"]["top10_impact"]["matrix"]["sha256"] for t in tiers}
    rls = {t["summary"]["top10_impact"]["ranked_lists"]["sha256"] for t in tiers}
    if len(mats) != 1 or len(rls) != 1:
        sys.exit(f"tiers cite different matrix/ranked-list hashes: {mats} {rls}")
    union = set().union(*(t["directed"] for t in tiers))
    if len(union) != sum(len(t["directed"]) for t in tiers):
        sys.exit("tier directed sets overlap")
    if len(union) != a.expect_directions:
        sys.exit(f"union has {len(union)} directions, expected {a.expect_directions}")
    matrix_path = Path(tiers[0]["summary"]["top10_impact"]["matrix"]["path"])
    rl_path = Path(tiers[0]["summary"]["top10_impact"]["ranked_lists"]["path"])
    if sha256(matrix_path) != next(iter(mats)) or sha256(rl_path) != next(iter(rls)):
        sys.exit("matrix or ranked-list file on disk no longer matches the hash cited by the tiers")

    # (e) population and original top-10 reconstruction
    saved = pd.read_csv(rl_path, dtype={"query": str, "neighbour": str})
    m = pickle.load(open(matrix_path, "rb"))
    all_ids = [str(i) for i in m.index]
    qset = set(saved["query"])
    ids = [i for i in all_ids if i in qset]
    if len(ids) != len(qset):
        sys.exit("saved queries not all present in matrix")
    sub = m.loc[ids, ids].to_numpy(dtype=np.float64)
    pos = {s_: i for i, s_ in enumerate(ids)}
    saved_top = saved.sort_values(["query", "rank"]).groupby("query")["neighbour"].apply(list).to_dict()
    orig = [_topk(sub[i], ids) for i in range(len(ids))]
    mism = sum(1 for i, q in enumerate(ids) if orig[i] != saved_top[q][:TOPK])
    if mism:
        sys.exit(f"reconstructed original top-10 differs from saved lists for {mism} queries")

    out.mkdir(parents=True, exist_ok=False)
    scenarios = {
        "A_capped": pd.concat([t["capped"] for t in tiers], ignore_index=True),
        "C_capped_plus_nonreproduced": pd.concat([t["capped"] for t in tiers] + [t["nonrep"] for t in tiers], ignore_index=True),
        "D_all_nonconverged_resolved": pd.concat([t["resolved"] for t in tiers], ignore_index=True),
    }
    report = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "script_sha256": sha256(Path(__file__)),
        "resolve_tool_sha256": sha256(HERE / "r2_2_ot30_resolve.py"),
        "tiers": [{"dir": t["dir"], "hashes": t["hashes"], "directed": len(t["directed"]), "capped": int(len(t["capped"])),
                   "capped_pairs": t["summary"]["capped_pairs"], "non_reproduced": t["n_nonrep"],
                   "non_converged_rows": int(len(t["resolved"])), "composition_by_classification": t["composition"],
                   "rows_excluded_high_not_converged": t["rows_excluded_high_not_converged"],
                   "exact_count_valid": t["summary"]["exact_count_valid"]} for t in tiers],
        "scenario_definitions": {
            "A_capped": "CAPPED high-solve values of all tiers (the correction)",
            "C_capped_plus_nonreproduced": "A plus NON_REPRODUCED high-solve values",
            "D_all_nonconverged_resolved": "C plus BIT_EQUAL_CODE_* rows = every non-CONVERGED direction with a converged high "
                                           "solve; CONVERGED directions are bit-equal no-ops. Value-equivalent to the OT-30 output "
                                           "review's scenario D; counts/margins NOT comparable to its 7,470 / 1.34e-4 figures."},
        "union_directions": len(union), "population": len(ids), "saved_top10_reconstructed": len(ids) - mism,
        "matrix": {"path": str(matrix_path), "sha256": next(iter(mats))}, "ranked_lists": {"path": str(rl_path), "sha256": next(iter(rls))},
        "combined": {"capped_directed": int(sum(len(t["capped"]) for t in tiers)),
                     "capped_pairs": len({tuple(sorted(x)) for t in tiers for x in zip(t["capped"]["query"], t["capped"]["other"])})},
        "scenarios": {},
    }
    for name, rows in scenarios.items():
        if len(rows) != len(set(zip(rows["query"], rows["other"]))):
            sys.exit(f"{name}: duplicate directed rows")
        corr, n_rep, skipped = substitute(sub, pos, rows)
        recs, moves = [], []
        for i, q in enumerate(ids):
            t1 = _topk(corr[i], ids)
            recs.append({"query": q, "same_set": set(orig[i]) == set(t1), "same_order": orig[i] == t1,
                         "orig_top10": " ".join(orig[i]), "corrected_top10": " ".join(t1)})
        for q, o, dl in zip(rows["query"], rows["other"], rows["delta_stored_minus_high"]):
            if q not in pos or o not in pos or sub[pos[q], pos[o]] >= MASK_VALUE:
                continue
            i = pos[q]
            full0 = _topk(sub[i], ids, len(ids)); full1 = _topk(corr[i], ids, len(ids))
            d10 = sub[i][pos[full0[TOPK - 1]]]
            c10 = corr[i][pos[full1[TOPK - 1]]]; c11 = corr[i][pos[full1[TOPK]]]
            moves.append({"query": q, "other": o, "orig_rank": full0.index(o) + 1, "corrected_rank": full1.index(o) + 1,
                          "delta_stored_minus_high": float(dl),
                          "margin_to_10th": float(sub[i, pos[o]] - d10),                 # pre-correction, outside rows
                          "post_slack_to_10th": float(corr[i, pos[o]] - c10),           # post-correction, outside rows
                          "post_slack_11th_minus_self": float(c11 - corr[i, pos[o]])})  # post-correction, inside rows
        imp = pd.DataFrame(recs); mv = pd.DataFrame(moves)
        imp.to_csv(out / f"combined_top10_{name}.csv", index=False)
        if len(mv):
            mv.to_csv(out / f"combined_moves_{name}.csv", index=False)
        outside = mv[mv["orig_rank"] > TOPK] if len(mv) else mv
        inside = mv[mv["orig_rank"] <= TOPK] if len(mv) else mv
        min_out = outside.loc[outside["post_slack_to_10th"].idxmin()] if len(outside) else None
        report["scenarios"][name] = {
            "rows_offered": int(len(rows)), "entries_replaced": n_rep, "skipped": skipped,
            "composition_by_classification": {k: int(v) for k, v in rows["classification"].value_counts().items()},
            "rows_excluded_high_not_converged": 0,
            "queries_order_changed": int((~imp["same_order"]).sum()), "queries_set_changed": int((~imp["same_set"]).sum()),
            "substituted_neighbours_entering_top10": int(((mv["orig_rank"] > TOPK) & (mv["corrected_rank"] <= TOPK)).sum()) if len(mv) else 0,
            "substituted_neighbours_leaving_top10": int(((mv["orig_rank"] <= TOPK) & (mv["corrected_rank"] > TOPK)).sum()) if len(mv) else 0,
            "substituted_originally_outside_top10": int(len(outside)), "substituted_originally_inside_top10": int(len(inside)),
            "min_margin_to_10th_over_replaced_outside_top10": float(outside["margin_to_10th"].min()) if len(outside) else None,
            "min_post_correction_slack_to_10th_outside": (
                {"slack": float(min_out["post_slack_to_10th"]), "query": min_out["query"], "other": min_out["other"],
                 "delta_stored_minus_high": float(min_out["delta_stored_minus_high"])} if min_out is not None else None),
            "min_post_correction_slack_11th_minus_self_inside": float(inside["post_slack_11th_minus_self"].min()) if len(inside) else None,
        }
    json.dump(report, open(out / "combined_top10.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
