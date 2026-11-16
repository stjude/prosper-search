"""[analysis plan] v3: one maximum-coverage ILP with strict co-author/class/patient quotas.

CCDI uses the staged FFPE-only population. No preliminary ILP output is read.
Produces a review workbook plus CSV/JSON provenance; refuses to overwrite runs.

Usage (export PROSPER_OUT first):
  POOL_VER=v3 POOL_TAG=20260928 python3 scripts/ilp_representatives/build_review_pool.py --cohort stv --nosignal-min 0
  POOL_VER=v3 POOL_TAG=20260928 python3 scripts/ilp_representatives/build_review_pool.py --cohort ccdi
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pulp import (LpBinary, LpMaximize, LpProblem, LpStatus, LpVariable,
                  PULP_CBC_CMD, lpSum, value)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "slide_selector"))
from slide_selector import get_top_k_similar_slides  # noqa: E402

OUT = Path(os.environ["PROSPER_OUT"])
XLSX = REPO / "docs/pathologist_study/Ali_CCDI_STV_pathologist_shortlist_20260925.xlsx"
DATE = os.environ.get("POOL_TAG", "20260928")
VER = os.environ.get("POOL_VER", "v3")
CFG = {
    "stv": dict(work_cohort="stv", sheet="StJude",
                ali_key=lambda d: d["file_name"].str.replace(r"\.tiff$", "", regex=True),
                features=OUT / "features/stv_inlib_uni2",
                matrix=OUT / "similarity_matrix/combined_similarity_matrix_stv_inlib_uni2_ot_cosine.pkl",
                per_query=OUT / "external_stv_labelled/per_query_stv_uni2_ot_cosine.csv"),
    "ccdi": dict(work_cohort="ccdi_ffpe", sheet="CCDI-MCI",
                 ali_key=lambda d: d["patient_id"].astype(str),
                 features=OUT / "features/ccdi_inlib_uni2",
                 matrix=OUT / "similarity_matrix/combined_similarity_matrix_ccdi_ffpe_inlib_uni2_ot_cosine.pkl",
                 per_query=OUT / "external_ccdi_inscope/per_query_ccdi_uni2_ot_cosine.csv"),
}
HARD_TIERS = {"very_difficult", "difficult"}
NOSIGNAL_TIERS = {"no_model_signal", "no_signal"}
EASY_CLASS_CAP = {"RB": 1, "MEL": 2}


def tile_counts(cohort: str, feat_dir: Path) -> pd.Series:
    cache = OUT / "ilp_representatives" / cohort / "tile_counts.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"slide_id": str}).set_index("slide_id")["n_tiles"]
    rows = []
    for fp in sorted(glob.glob(str(feat_dir / "*.pkl"))):
        with open(fp, "rb") as fh:
            obj = pickle.load(fh)
        feats = obj[0] if isinstance(obj, tuple) and len(obj) == 2 else obj
        if feats is None:
            raise ValueError(f"Missing features: {fp}")
        rows.append((Path(fp).stem, int(np.asarray(feats).shape[0])))
    counts = pd.DataFrame(rows, columns=["slide_id", "n_tiles"])
    counts.to_csv(cache, index=False)
    return counts.set_index("slide_id")["n_tiles"]


def selection_checks(selected, pop, eligible, ali, a) -> pd.DataFrame:
    """Audit a pool, including a human-edited pool, without solver variables."""
    selected = list(selected)
    if not set(selected) <= set(pop.index):
        raise ValueError("Selected slides outside the population")
    chosen = set(selected)
    tiers = ali.loc[ali.index.isin(chosen)]
    counts = pop.loc[selected, "tumor_group"].value_counts()
    quota = math.ceil(a.ali_quota_frac * a.budget)
    checks = []

    def add(name, observed, lower, upper):
        checks.append(dict(constraint=name, observed=observed, minimum=lower,
                           maximum=upper, passed=lower <= observed <= upper))

    add("pool_size", len(selected), a.budget, a.budget)
    add("duplicate_slides", len(selected) - len(chosen), 0, 0)
    add("ineligible_slides", len(chosen - eligible), 0, 0)
    add("ali_tier1", int((tiers.selection_tier == "Tier1").sum()), quota, a.budget)
    add("ali_hard", int(tiers.tier.isin(HARD_TIERS).sum()), a.hard_min, a.budget)
    add("ali_moderate", int((tiers.tier == "moderate").sum()), a.moderate_min, a.budget)
    add("ali_no_signal", int(tiers.tier.isin(NOSIGNAL_TIERS).sum()), a.nosignal_min, a.budget)
    add("ali_easy", int((tiers.tier == "easy").sum()), 0, a.ali_easy_max)
    for c in sorted(pop.tumor_group.unique()):
        cap = min(math.floor(a.class_max_frac * a.budget), EASY_CLASS_CAP.get(c, a.budget))
        add(f"class:{c}", int(counts.get(c, 0)), 1, cap)
    add("duplicate_patients", int(pop.loc[selected, "patient_id"].duplicated().sum()), 0, 0)
    return pd.DataFrame(checks)


def solve_pool(pop, nb, eligible, ali, a):
    ids = list(pop.index)
    covered_by = {s: {s} for s in ids}
    for n, ns in nb.items():
        for s in ns:
            covered_by[s].add(n)
    model = LpProblem("review_pool", LpMaximize)
    x = {s: LpVariable(f"x_{i}", cat=LpBinary) for i, s in enumerate(ids)}
    y = {s: LpVariable(f"y_{i}", cat=LpBinary) for i, s in enumerate(ids)}
    model += lpSum(y.values())
    for s in ids:
        model += y[s] <= lpSum(x[n] for n in sorted(covered_by[s]))
        if s not in eligible:
            model += x[s] == 0
    model += lpSum(x.values()) == a.budget
    tier1 = ali.index[ali.selection_tier == "Tier1"]
    model += lpSum(x[s] for s in tier1) >= math.ceil(a.ali_quota_frac * a.budget)
    for labels, minimum in [(HARD_TIERS, a.hard_min), ({"moderate"}, a.moderate_min),
                            (NOSIGNAL_TIERS, a.nosignal_min)]:
        model += lpSum(x[s] for s in ali.index[ali.tier.isin(labels)]) >= minimum
    model += lpSum(x[s] for s in ali.index[ali.tier == "easy"]) <= a.ali_easy_max
    for c, group in pop.groupby("tumor_group"):
        model += lpSum(x[s] for s in group.index) >= 1
        model += lpSum(x[s] for s in group.index) <= math.floor(a.class_max_frac * a.budget)
        if c in EASY_CLASS_CAP:
            model += lpSum(x[s] for s in group.index) <= EASY_CLASS_CAP[c]
    for _, group in pop.groupby("patient_id"):
        model += lpSum(x[s] for s in group.index) <= 1
    t0 = time.time()
    model.solve(PULP_CBC_CMD(msg=False, threads=1))
    status = LpStatus[model.status]
    print(f"status={status} objective={value(model.objective)} in {time.time()-t0:.1f}s")
    if status != "Optimal":
        raise ValueError(f"ILP status {status}; no pool written. Constraints were NOT relaxed.")
    selected = [s for s in ids if x[s].varValue > 0.5]
    covered = set().union(*(nb[s] | {s} for s in selected))
    solver_covered = {s for s in ids if y[s].varValue > 0.5}
    if covered != solver_covered or not math.isclose(value(model.objective), len(covered)):
        raise ValueError("Solver coverage does not match the selected neighbourhood union")
    checks = selection_checks(selected, pop, eligible, ali, a)
    if not checks.passed.all():
        raise ValueError("Selection violates constraints:\n" + checks[~checks.passed].to_string(index=False))
    return selected, covered, checks


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True, choices=list(CFG))
    p.add_argument("--budget", type=int, default=40)
    p.add_argument("--alternates", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--tile-floor", type=int, default=None,
                   help="absolute floor; otherwise use --tile-floor-quantile")
    p.add_argument("--tile-floor-relaxed", type=int, default=None)
    p.add_argument("--relax-tile-floor-for", nargs="*", default=[],
                   help="explicitly approved classes using a relaxed floor; none by default")
    p.add_argument("--tile-floor-quantile", type=float, default=0.25)
    p.add_argument("--ali-quota-frac", type=float, default=0.5)
    p.add_argument("--hard-min", type=int, default=8)
    p.add_argument("--moderate-min", type=int, default=5)
    p.add_argument("--nosignal-min", type=int, default=4,
                   help="strict minimum; approved STV exception must be passed explicitly as 0")
    p.add_argument("--ali-easy-max", type=int, default=3)
    p.add_argument("--class-max-frac", type=float, default=0.35)
    a = p.parse_args()
    if a.budget <= 0 or a.top_k <= 0 or min(a.alternates, a.hard_min, a.moderate_min,
                                           a.nosignal_min, a.ali_easy_max) < 0:
        p.error("budget/top-k must be positive and counts must be nonnegative")
    if not (0 <= a.ali_quota_frac <= 1 and 0 < a.class_max_frac <= 1
            and 0 <= a.tile_floor_quantile <= 1):
        p.error("invalid quota, class-cap or tile-quantile fraction")
    cfg = CFG[a.cohort]
    work = OUT / "ilp_representatives" / cfg["work_cohort"]
    rp = work / "review_pool"
    stem = f"{a.cohort}_{VER}_{DATE}"
    xlsx = REPO / "docs/pathologist_study" / f"review_pool_{stem}.xlsx"
    targets = [rp / f"{name}_{stem}.csv" for name in
               ("review_pool", "alternates", "summary", "constraint_checks")]
    targets += [rp / f"provenance_{stem}.json", xlsx]
    if any(path.exists() for path in targets):
        p.error(f"Output already exists for {stem}; use a new POOL_VER/POOL_TAG, not an overwrite")

    pop = pd.read_csv(work / "population.csv", dtype=str).set_index("slide_id").sort_index()
    if pop.empty or pop.index.has_duplicates or pop.index.isna().any() or pop.tumor_group.isna().any():
        raise ValueError("Population must have unique, nonmissing slide IDs and tumour groups")
    if (pop.tumor_group == "NORMAL").any():
        raise ValueError("NORMAL slides are outside the review population")
    prep_path = OUT / "external_ccdi/slide_preparation.csv"
    if a.cohort == "ccdi":
        if pop.index.str.contains("_").any():
            raise ValueError("CCDI requires unsuffixed participant IDs")
        prep = pd.read_csv(prep_path, dtype=str).set_index("slide_id")
        if prep.index.has_duplicates:
            raise ValueError("Duplicate slide IDs in preparation records")
        current = prep.reindex(pop.index)
        for col in ("prep_dicom", "prep_manifest"):
            if current[col].isna().any() or not pop[col].equals(current[col]):
                raise ValueError(f"CCDI staged {col} is stale or missing; re-stage the population")
        if not (current.prep_dicom == "FFPE").all():
            raise ValueError("CCDI population contains non-FFPE, mixed or unknown preparation")
        pop["patient_id"] = pop.index
    if "patient_id" not in pop or pop.patient_id.isna().any() or (pop.patient_id.str.strip() == "").any():
        raise ValueError("Every slide must have a patient_id")
    ids = list(pop.index)
    sm = pd.read_pickle(cfg["matrix"])
    sm.index = sm.index.astype(str)
    sm.columns = sm.columns.astype(str)
    if (sm.index.has_duplicates or sm.columns.has_duplicates or set(sm.index) != set(ids)
            or set(sm.columns) != set(ids)):
        raise ValueError("Matrix IDs must match the population exactly")
    sm = sm.loc[ids, ids]
    if not np.isfinite(sm.to_numpy()).all() or a.top_k >= len(ids):
        raise ValueError("Nonfinite matrix entries or top-k too large for the population")
    nb = get_top_k_similar_slides(sm, top_k=a.top_k)

    tiles = tile_counts(cfg["work_cohort"], cfg["features"]).reindex(ids)
    if tiles.isna().any() or (tiles <= 0).any() or (tiles % 1 != 0).any():
        raise ValueError("Tile counts must be known positive integers for every population slide")
    tiles = tiles.astype(int)
    ali = pd.read_excel(XLSX, sheet_name=cfg["sheet"]).dropna(axis=1, how="all")
    ali["slide_id"] = cfg["ali_key"](ali)
    ali = ali[ali.slide_id.isin(ids)].set_index("slide_id").sort_index()
    if ali.index.has_duplicates or ali[["tier", "selection_tier"]].isna().any().any():
        raise ValueError("Ali's matching rows must have unique slide IDs and known tiers")
    if not ali.selection_tier.isin(["Tier1", "Tier2"]).all():
        raise ValueError("Unrecognised Ali selection tier")
    tier1 = set(ali.index[ali.selection_tier == "Tier1"])
    tier2 = set(ali.index[ali.selection_tier == "Tier2"])
    quantile_used = a.tile_floor_quantile if a.tile_floor is None else None
    if a.tile_floor is None:
        a.tile_floor = int(tiles.quantile(a.tile_floor_quantile))
    if a.tile_floor_relaxed is None and a.relax_tile_floor_for:
        a.tile_floor_relaxed = int(tiles.quantile(0.10)) if quantile_used is not None else int(0.8 * a.tile_floor)
    if a.tile_floor < 0 or (a.tile_floor_relaxed is not None
                           and not 0 <= a.tile_floor_relaxed <= a.tile_floor):
        p.error("tile floors must be nonnegative; relaxed floor cannot exceed the main floor")
    cls = pop["tumor_group"]
    if not set(a.relax_tile_floor_for) <= set(cls):
        p.error("tile-floor relaxation names a tumour group outside the population")
    floor_by_class = {c: a.tile_floor_relaxed if c in a.relax_tile_floor_for else a.tile_floor
                      for c in sorted(cls.unique())}
    eligible = {s for s in ids if tiles[s] >= floor_by_class[cls[s]]}
    print(f"[{a.cohort}] tile floors {floor_by_class}; eligible={len(eligible)}")
    quota = math.ceil(a.ali_quota_frac * a.budget)
    t = ali["tier"]
    print("Eligible Ali tiers:", ali.loc[ali.index.isin(eligible), "tier"].value_counts().to_dict())
    sel, covered, checks = solve_pool(pop, nb, eligible, ali, a)

    alts = []
    pool_cov = set(covered)
    for _ in range(a.alternates):
        candidates = [s for s in ids if s not in sel and s not in alts and s in eligible]
        if not candidates:
            raise ValueError(f"Cannot supply the requested {a.alternates} eligible alternates")
        best = max(candidates, key=lambda s: len((nb[s] | {s}) - pool_cov))
        alts.append(best)
        pool_cov |= nb[best] | {best}

    # Outcomes are read only after selection and never enter the objective.
    pq = pd.read_csv(cfg["per_query"], dtype={"query": str}).set_index("query")
    if pq.index.has_duplicates:
        raise ValueError("Duplicate per-query outcome rows")
    pq = pq.reindex(ids)
    if pq[["true_label", "pred_closest_k5", "top1_label", "hit_k5"]].isna().any().any():
        raise ValueError("Missing per-query outcome metadata")
    if not pq.true_label.equals(cls.rename("true_label")):
        raise ValueError("Per-query labels disagree with population tumour groups")
    vote_correct5 = pq.pred_closest_k5 == pq.true_label
    if not pq.hit_k5.isin([0, 1]).all():
        raise ValueError("Unrecognised hit_k5 values")
    hit5 = pq.hit_k5.astype(bool)

    def source(s):
        if s in tier1:
            return "ali_tier1"
        if s in tier2:
            return "ali_tier2"
        return "ilp_coverage"

    def table(slides, kind):
        d = pop.loc[slides].copy()
        d.insert(0, "slide_id", slides)
        d["kind"] = kind
        d["source"] = [source(s) for s in slides]
        d["ali_tier"] = [t.get(s) for s in slides]
        d["ali_selection_tier"] = [ali.selection_tier.get(s) for s in slides]
        d["n_tiles"] = [int(tiles[s]) for s in slides]
        d["n_covered_by_this_slide"] = [len(nb[s]) + 1 for s in slides]
        d["prosper_top1_label"] = pq.loc[slides, "top1_label"].values
        d["prosper_pred_k5"] = pq.loc[slides, "pred_closest_k5"].values
        d["prosper_vote_correct_at5"] = vote_correct5.loc[slides].values
        d["prosper_hit_at5"] = hit5.loc[slides].values
        d["decision"] = ""
        d["reason"] = ""
        return d.reset_index(drop=True)

    pool_df = table(sel, "pool").sort_values(["tumor_group", "source"]).reset_index(drop=True)
    alt_df = table(alts, "alternate")
    summ = pd.DataFrame({
        "population": cls.value_counts(),
        "eligible_tiles": cls[sorted(eligible)].value_counts(),
        "pool": pool_df.tumor_group.value_counts(),
        "pool_from_ali": pool_df[pool_df.source.str.startswith("ali")].tumor_group.value_counts(),
        "covered": cls[sorted(covered)].value_counts(),
    }).fillna(0).astype(int).sort_values("population", ascending=False)
    summ.loc["TOTAL"] = summ.sum()
    summ["covered_frac"] = (summ["covered"] / summ["population"]).round(3)
    prov = {
        "cohort": a.cohort, "date": DATE, "version": VER, "budget": a.budget, "n_alternates": len(alts),
        "model": "single_ilp_maximum_neighbourhood_coverage",
        "secondary_objective": None, "preliminary_ilp_used": False,
        "work_cohort": cfg["work_cohort"], "population_file": str(work / "population.csv"),
        "population": len(ids), "eligible_after_tile_floor": len(eligible),
        "tile_floor": a.tile_floor, "tile_floor_relaxed": a.tile_floor_relaxed,
        "tile_floor_quantile": quantile_used, "approved_tile_floor_relaxations": a.relax_tile_floor_for,
        "tile_count_quantiles": {str(q): int(tiles.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "tile_floor_by_class": floor_by_class,
        "ali_in_population": {"tier1": len(tier1), "tier2": len(tier2)},
        "forced_both": [], "ali_quota": quota,
        "tier_minima": {"hard": a.hard_min, "moderate": a.moderate_min,
                        "no_signal": a.nosignal_min, "ali_easy_max": a.ali_easy_max},
        "easy_class_cap": EASY_CLASS_CAP, "class_max_frac": a.class_max_frac, "top_k": a.top_k,
        "n_covered": len(covered), "coverage_frac": round(len(covered) / len(ids), 4),
        "source_counts": pool_df.source.value_counts().to_dict(),
        "ali_tier_counts_in_pool": pool_df.ali_tier.value_counts().to_dict(),
        "vote_correct_at5": int(pool_df.prosper_vote_correct_at5.sum()),
        "retrieval_hit_at5": int(pool_df.prosper_hit_at5.sum()),
        "preparation_in_pool": pool_df.prep_dicom.value_counts().to_dict() if a.cohort == "ccdi" else "not independently audited",
        "preparation_in_alternates": alt_df.prep_dicom.value_counts().to_dict() if a.cohort == "ccdi" else "not independently audited",
        "preparation_file": str(prep_path) if a.cohort == "ccdi" else None,
        "all_constraints_passed": bool(checks.passed.all()),
        "solver": "PuLP CBC exact", "status": "Optimal", "matrix": str(cfg["matrix"]),
        "per_query": str(cfg["per_query"]), "ali_xlsx": str(XLSX),
        "arguments": vars(a), "command": sys.argv,
    }
    readme = pd.DataFrame({"README": [
        f"{a.cohort.upper()} pathologist review pool {VER} ({DATE}) - plan 0007.",
        f"Pool = {a.budget} slides per cohort, 80 total at budget 40; reviewers may trim to ~30 each.",
        "Fill decision (keep/drop/swap) and reason. Recheck final-size quotas, class coverage and patient uniqueness after edits.",
        "Alternates are ranked by marginal coverage, NOT pre-approved replacements. A swap can violate constraints.",
        "One ILP: maximise neighbourhood union size; no preliminary representative list, forced consensus or secondary score.",
        "source: ali_tier1/2 = Ali membership; ilp_coverage = not on Ali's list. All selected slides contribute coverage.",
        "Ali tiers retain his original labels (CCDI based on MELD); our distances use UNI2+OT for both cohorts.",
        "prosper_vote_correct_at5 = pred_closest_k5 equals true label; prosper_hit_at5 = any relevant slide in the top 5.",
        "Historical prosper_hit@5 columns actually recorded closest-vote correctness, not retrieval hit@5.",
        f"Tile floor: {a.tile_floor} UNI2 tiles; quantile={quantile_used}; approved class exceptions={a.relax_tile_floor_for}.",
        f"Constraints: >=1 per tumour group, none >{int(a.class_max_frac*100)}% of pool; RB<=1, MEL<=2; 1 slide/patient;",
        f">={quota} Ali Tier1; hard>={a.hard_min}, moderate>={a.moderate_min}, no_signal>={a.nosignal_min}, Ali-easy<={a.ali_easy_max}.",
        f"Neighbourhood coverage: {len(covered)}/{len(ids)} ({prov['coverage_frac']:.1%}); at most {(a.top_k+1)*a.budget} at this budget.",
        "CCDI population, pool and alternates must be FFPE-only; STV preparation was not independently audited here.",
        "This is a curated, difficulty-enriched review set, not a prevalence-representative performance sample.",
        "Do NOT forward source/ali_tier/ali_selection_tier/prosper_* columns to the pathologists.",
    ]})
    rp.mkdir(parents=True, exist_ok=True)
    pool_df.to_csv(rp / f"review_pool_{stem}.csv", index=False)
    alt_df.to_csv(rp / f"alternates_{stem}.csv", index=False)
    summ.to_csv(rp / f"summary_{stem}.csv")
    checks.to_csv(rp / f"constraint_checks_{stem}.csv", index=False)
    (rp / f"provenance_{stem}.json").write_text(json.dumps(prov, indent=2, default=str))
    with pd.ExcelWriter(xlsx) as w:
        pool_df.to_excel(w, sheet_name="pool", index=False)
        alt_df.to_excel(w, sheet_name="alternates", index=False)
        summ.to_excel(w, sheet_name="summary")
        checks.to_excel(w, sheet_name="constraints", index=False)
        readme.to_excel(w, sheet_name="README", index=False)
    print(summ.to_string())
    print(json.dumps({k: prov[k] for k in ("source_counts", "ali_tier_counts_in_pool",
                                         "vote_correct_at5", "retrieval_hit_at5", "coverage_frac")},
                     indent=1))
    print("wrote", xlsx)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
