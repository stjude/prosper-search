"""Run the two-round top-k ILP (scripts/slide_selector) on a within-cohort
distance matrix and write per-round lists, checks and class descriptives.

Usage:
  PROSPER_OUT=... python3 scripts/ilp_representatives/run_ilp.py --cohort stv \
      --matrix $PROSPER_OUT/similarity_matrix/combined_similarity_matrix_stv_inlib_uni2_ot_cosine.pkl

Outputs under $PROSPER_OUT/ilp_representatives/<cohort>/:
  selected_round1.txt, selected_round2.txt   sorted slide ids
  run.json                                   sizes, checks, parameters, provenance
  selected_round{1,2}_with_labels.csv        selection joined to population.csv
  class_summary.csv                          per tumor_group: population n, round1 n, round2 n
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "slide_selector"))
from slide_selector import get_top_k_similar_slides, solve_ilp  # noqa: E402

SENTINEL = 1e6


def coverage_ok(selected: set, neighbors: dict) -> tuple[bool, int]:
    """Mirror of the ILP constraint: slide s is covered if it is selected or
    some selected slide n has s among ITS top-k (s in neighbors[n])."""
    covered = set(selected)
    for n in selected:
        covered |= neighbors[n]
    uncovered = [s for s in neighbors if s not in covered]
    return not uncovered, len(uncovered)


def is_minimal(selected: set, neighbors: dict) -> bool:
    for s in list(selected):
        ok, _ = coverage_ok(selected - {s}, neighbors)
        if ok:
            return False
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--matrix", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--rounds", type=int, default=2)
    a = p.parse_args()

    out = Path(os.environ["PROSPER_OUT"]) / "ilp_representatives" / a.cohort
    pop = pd.read_csv(out / "population.csv", dtype={"slide_id": str})
    sm = pd.read_pickle(a.matrix)
    sm.index = sm.index.astype(str)
    sm.columns = sm.columns.astype(str)
    assert list(sm.index) == list(sm.columns), "matrix not square/aligned"
    assert set(sm.index) == set(pop.slide_id), "matrix ids != population"
    vals = sm.to_numpy()
    n = len(sm)
    assert (vals >= SENTINEL).sum() == n, "expected exactly one sentinel per row (self)"
    assert not pd.isna(vals).any(), "NaNs in matrix"
    off = vals[vals < SENTINEL]
    asym = float(abs(vals - vals.T)[vals < SENTINEL].max())

    run = {
        "cohort": a.cohort, "matrix": a.matrix, "top_k": a.top_k,
        "n_population": n, "distance_min": float(off.min()),
        "distance_median": float(pd.Series(off).median()),
        "distance_max": float(off.max()), "max_abs_asymmetry": asym,
        "rounds": [], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    cur = sm
    for r in range(1, a.rounds + 1):
        nb = get_top_k_similar_slides(cur, top_k=a.top_k)
        t0 = time.time()
        sel = set(solve_ilp(nb, None, solver_verbose=False))
        cov, n_unc = coverage_ok(sel, nb)
        info = {
            "round": r, "n_in": cur.shape[0], "n_selected": len(sel),
            "coverage_100pct": cov, "n_uncovered": n_unc,
            "minimal": is_minimal(sel, nb), "solve_seconds": round(time.time() - t0, 2),
        }
        run["rounds"].append(info)
        print(info, flush=True)
        ids = sorted(sel)
        (out / f"selected_round{r}.txt").write_text("\n".join(ids) + "\n")
        pop[pop.slide_id.isin(sel)].to_csv(out / f"selected_round{r}_with_labels.csv", index=False)
        if len(sel) >= cur.shape[0]:
            break
        cur = sm.loc[ids, ids]

    summ = pop.groupby("tumor_group").size().rename("population").to_frame()
    for r in range(1, len(run["rounds"]) + 1):
        ids = set((out / f"selected_round{r}.txt").read_text().split())
        summ[f"round{r}"] = pop[pop.slide_id.isin(ids)].groupby("tumor_group").size()
    summ = summ.fillna(0).astype(int).sort_values("population", ascending=False)
    summ.loc["TOTAL"] = summ.sum()
    if "patient_id" in pop.columns:
        pats = {"population": pop.patient_id.nunique()}
        for r in range(1, len(run["rounds"]) + 1):
            ids = set((out / f"selected_round{r}.txt").read_text().split())
            pats[f"round{r}"] = pop[pop.slide_id.isin(ids)].patient_id.nunique()
        run["patients"] = pats
    summ.to_csv(out / "class_summary.csv")
    (out / "run.json").write_text(json.dumps(run, indent=2))
    print(summ.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
