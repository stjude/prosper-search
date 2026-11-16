"""[analysis plan]: uniformly tissue-qualified, two-stage hierarchical author-review pool.

Only run on accepted inputs after separate implementation review. Export the
repository-local PROSPER_OUT explicitly. This does not release reader cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd
from pulp import (LpBinary, LpMinimize, LpProblem, LpSolutionOptimal, LpStatus,
                  LpStatusOptimal, LpVariable, PULP_CBC_CMD, lpSum, value)

import build_hierarchical_review_pool as v4

REPO = Path(__file__).resolve().parents[2]
TEST_FILE = Path(__file__).with_name("test_build_hierarchical_review_pool_v5.py")
FROZEN_V4 = {
    Path(v4.__file__).resolve(): "86581e042604a75ac78ea5aa6607dbb6422d9b5dbcf0fd243619045c0a110fb1",
    v4.TEST_FILE.resolve(): "76aabc7681634ad28184a334b8b868201edf7463c7bfeb6a35f41a0127f6d110",
}
EXPECTED = {"stv": (631, 615, 57, 14), "ccdi": (441, 411, 80, 12)}
SWAP_COLUMNS = ["incoming_slide_id", "outgoing_slide_id", "tissue_cost_delta",
                "incoming_tissue_band", "incoming_qc_flag", "qc_status",
                "valid_only_as_individual_swap"]


def source_bindings(cohort: str, work: Path, inventory: Path) -> dict[str, str]:
    for path, expected in FROZEN_V4.items():
        if v4.digest(path) != expected:
            raise ValueError(f"Frozen v4 dependency has changed: {path}")
    bindings = v4.source_bindings(cohort, work, inventory)
    for path in (Path(__file__).resolve(), TEST_FILE.resolve()):
        v4.add_binding(bindings, path, v4.digest(path))
    return bindings


def study_subset(cohort: str, base: pd.DataFrame, matrix: pd.DataFrame,
                 ali: pd.DataFrame, base_candidates: set[str],
                 counts: pd.DataFrame):
    ids = list(base.index)
    if list(matrix.index) != ids or list(counts.index) != ids:
        raise ValueError("Base population/matrix/count order mismatch")
    quantity_ids = [s for s in ids if counts.at[s, "detected_tissue_tiles"] >= 800]
    eligible = set(quantity_ids) & base_candidates
    expected_p, expected_e, expected_tier1, expected_groups = EXPECTED[cohort]
    tier1 = set(ali.index[ali.selection_tier == "Tier1"]) & set(quantity_ids)
    if (len(quantity_ids), len(eligible), len(tier1)) != (
            expected_p, expected_e, expected_tier1):
        raise ValueError("Unexpected study population, candidates or Tier1 supply")
    pop = base.loc[quantity_ids].copy()
    if pop.tumor_group.nunique() != expected_groups:
        raise ValueError("Filtering lost or introduced a tumour group")
    filtered_counts = counts.loc[quantity_ids].copy()
    filtered_matrix = matrix.loc[quantity_ids, quantity_ids].copy()
    if not set(eligible) <= set(pop.index) or (filtered_counts.detected_tissue_tiles < 800).any():
        raise ValueError("Quantity-ineligible candidate or population slide")
    table = base[["slide_id", "patient_id", "tumor_group"]].copy().reset_index(drop=True)
    table.insert(0, "cohort", cohort)
    table["ali_priority"] = [ali.at[s, "selection_tier"] if s in ali.index else "not_on_ali_list"
                             for s in ids]
    table["ali_difficulty_tier1_only"] = [
        ali.at[s, "tier"] if s in ali.index and ali.at[s, "selection_tier"] == "Tier1"
        else "" for s in ids
    ]
    table["detected_tissue_tiles"] = counts.loc[ids, "detected_tissue_tiles"].tolist()
    table["mosaic_tiles"] = counts.loc[ids, "mosaic_tiles"].tolist()
    table["tissue_band"] = counts.loc[ids, "tissue_band"].tolist()
    table["quantity_eligible"] = [s in quantity_ids for s in ids]
    table["final_candidate"] = [s in eligible for s in ids]
    table["eligibility_reason"] = [
        "BELOW_800_TISSUE_TILES" if s not in quantity_ids else
        "ALI_TIER2_NOT_SELECTABLE" if s not in eligible else "FINAL_CANDIDATE"
        for s in ids
    ]
    if table.slide_id.tolist() != ids or table.final_candidate.sum() != len(eligible):
        raise ValueError("Base exclusion accounting is incomplete")
    return pop, filtered_matrix, filtered_counts, eligible, table


def neighborhoods(matrix: pd.DataFrame) -> dict[str, list[str]]:
    ids = list(matrix.index)
    if len(ids) <= 5 or list(matrix.columns) != ids:
        raise ValueError("Filtered matrix must have ordered square population identity")
    return {s: [s, *sorted((other for other in ids if other != s),
                           key=lambda other: (float(matrix.at[s, other]), other))[:5]]
            for s in ids}


def solve_model(model: LpProblem, stage: str, scratch_parent: Path,
                solver=None) -> dict:
    if solver is not None:
        model.solve(solver)
    else:
        if not scratch_parent.is_dir() or not scratch_parent.resolve().is_relative_to(REPO):
            raise ValueError("CBC requires an existing repository-local scratch parent")
        owned = scratch_parent.resolve() / f".v5_{stage}_cbc_{uuid.uuid4().hex}"
        owned.mkdir(mode=0o700, exist_ok=False)
        try:
            cbc = PULP_CBC_CMD(msg=False, threads=1, timeLimit=300)
            cbc.tmpDir = str(owned)
            model.solve(cbc)
        finally:
            shutil.rmtree(owned)
    info = {"problem_status": LpStatus.get(model.status, str(model.status)),
            "solution_status": model.sol_status, "solution_optimal_code": LpSolutionOptimal,
            "name": "CBC", "threads": 1, "time_limit_seconds": 300}
    if model.status != LpStatusOptimal or model.sol_status != LpSolutionOptimal:
        raise ValueError(f"{stage} not proven optimal; no incumbent released: {info}")
    return info


def solve_stage1(nb: dict[str, list[str]], scratch_parent: Path, solver=None):
    ids = list(nb)
    if set().union(*(set(v) for v in nb.values())) != set(ids):
        raise ValueError("Stage1 outgoing neighborhoods cannot cover filtered population")
    covered_by = {s: [] for s in ids}
    for candidate, targets in nb.items():
        if len(targets) != 6 or targets[0] != candidate or len(set(targets)) != 6:
            raise ValueError("Stage1 must use self plus five OTHER outgoing neighbors")
        for target in targets:
            if target not in covered_by:
                raise ValueError("Stage1 neighborhood contains slide outside population")
            covered_by[target].append(candidate)
    model = LpProblem("v5_minimum_filtered_anchor_cover", LpMinimize)
    z = {s: LpVariable(f"z_{j}", cat=LpBinary) for j, s in enumerate(ids)}
    model += lpSum(z.values())
    for target in ids:
        model += lpSum(z[s] for s in covered_by[target]) >= 1
    info = solve_model(model, "stage1", scratch_parent, solver)
    if any(z[s].varValue not in (0, 1) for s in ids):
        raise ValueError("Stage1 lacks integral solution values")
    anchors = [s for s in ids if z[s].varValue == 1]
    witness = {target: [a for a in anchors if target in nb[a]] for target in ids}
    if not anchors or any(not cover for cover in witness.values()):
        raise ValueError("Stage1 independently recounted coverage failed")
    objective = value(model.objective)
    if objective is None or int(round(objective)) != len(anchors):
        raise ValueError("Stage1 objective disagrees with selected anchor count")
    info["objective"] = len(anchors)
    return anchors, witness, info


def solve_stage2(pop: pd.DataFrame, ali: pd.DataFrame, counts: pd.DataFrame,
                 eligible: set[str], anchors: list[str], original: dict[str, list[str]],
                 rectangular: dict[str, list[str]], scratch_parent: Path, solver=None):
    ids = [s for s in pop.index if s in eligible]
    if not ids or any(counts.at[s, "detected_tissue_tiles"] < 800 for s in ids):
        raise ValueError("Stage2 contains an ineligible tissue count")
    model = LpProblem("v5_tissue_qualified_author_review", LpMinimize)
    x = {s: LpVariable(f"x_{j}", cat=LpBinary) for j, s in enumerate(ids)}
    borderline = [s for s in ids if counts.at[s, "detected_tissue_tiles"] < 1000]
    model += lpSum(x[s] for s in borderline)
    model += lpSum(x.values()) == v4.BUDGET
    for anchor in anchors:
        support = [x[s] for s in ids if anchor in rectangular[s]]
        if not support:
            raise ValueError(f"Stage2 cannot cover anchor: {anchor}")
        model += lpSum(support) >= 1
    tier1 = [s for s in ids if s in ali.index and ali.at[s, "selection_tier"] == "Tier1"]
    model += lpSum(x[s] for s in tier1) >= 20
    for labels, minimum in ((v4.HARD, 8), ({"moderate"}, 5),
                            ({"no_model_signal"}, 0 if "stv_dx" in pop else 4)):
        model += lpSum(x[s] for s in tier1 if ali.at[s, "tier"] in labels) >= minimum
    model += lpSum(x[s] for s in tier1 if ali.at[s, "tier"] == "easy") <= 3
    for group, subset in pop.groupby("tumor_group"):
        subset_x = [x[s] for s in subset.index if s in x]
        if not subset_x:
            raise ValueError(f"Stage2 has no eligible final candidate for group {group}")
        model += lpSum(subset_x) >= 1
        model += lpSum(subset_x) <= {"RB": 1, "MEL": 2}.get(group, 14)
    for _, subset in pop.groupby("patient_id"):
        model += lpSum(x[s] for s in subset.index if s in x) <= 1
    info = solve_model(model, "stage2", scratch_parent, solver)
    if any(x[s].varValue not in (0, 1) for s in ids):
        raise ValueError("Stage2 lacks integral solution values")
    selected = [s for s in ids if x[s].varValue == 1]
    checks = v4.selection_checks(selected, pop, ali, eligible, original, rectangular, anchors)
    if not all(row["passed"] for row in checks.values()):
        raise ValueError(f"Stage2 independent constraint recount failed: {checks}")
    cost = v4.tissue_cost(selected, counts)
    raw_objective = value(model.objective)
    if (raw_objective is None and borderline) or (
            raw_objective is not None and int(round(raw_objective)) != cost):
        raise ValueError("Stage2 CBC objective differs from selected borderline count")
    if raw_objective is None and cost != 0:
        raise ValueError("Stage2 zero objective has nonzero selected cost")
    info["objective"] = cost
    return selected, checks, info


def file_bindings(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): v4.digest(path) for path in paths}


def text_frame(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kwargs)


def excel_frame(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet, dtype=str, keep_default_na=False)


def equal_frame(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    if not actual.equals(expected.astype(str)):
        raise ValueError(f"{label} output readback mismatch")


def stage1_write(run: Path, pop: pd.DataFrame, matrix: pd.DataFrame,
                 counts: pd.DataFrame, eligibility: pd.DataFrame,
                 anchors: list[str], witness: dict[str, list[str]], info: dict,
                 nb: dict[str, list[str]], source: dict[str, str]) -> dict[str, str]:
    if run.exists():
        raise FileExistsError("Stage1 run directory already exists")
    if (nb != neighborhoods(matrix) or info.get("objective") != len(anchors) or
            info.get("problem_status") != "Optimal" or
            info.get("solution_status") != LpSolutionOptimal or
            set(witness) != set(pop.index) or
            any(witness[s] != [a for a in anchors if s in nb[a]] or not witness[s]
                for s in pop.index)):
        raise ValueError("Stage1 matrix, proven solution and coverage evidence disagree")
    run.mkdir(exist_ok=False)
    pop_file = run / "filtered_population.csv"
    matrix_file = run / "filtered_distance_matrix.pkl"
    counts_file = run / "filtered_tissue_counts.csv"
    base_file = run / "base_eligibility.csv"
    anchor_file = run / "stage1_anchors.txt"
    report_file = run / "stage1.json"
    pop.to_csv(pop_file, index=False)
    matrix.to_pickle(matrix_file)
    counts.rename_axis("slide_id").to_csv(counts_file)
    eligibility.to_csv(base_file, index=False)
    anchor_file.write_text("\n".join(anchors) + "\n")
    anchor_sha = v4.digest(anchor_file)
    report = {
        "design": "plan0009_v5_new_filtered_minimum_cover", "population": len(pop),
        "selected_anchor_ids": anchors, "anchor_list_sha256": anchor_sha,
        "solver": info, "original_neighborhoods": nb,
        "coverage": {"numerator": sum(bool(w) for w in witness.values()),
                     "denominator": len(pop), "population_to_selected_anchor_witnesses": witness},
        "source_bindings_sha256": source,
        "generated_inputs_sha256": file_bindings([pop_file, matrix_file, counts_file, base_file]),
        "warning": "STAGE1 ONLY: no phase2 feasibility or final author pool is established",
    }
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    paths = [pop_file, matrix_file, counts_file, base_file, anchor_file, report_file]
    bound = file_bindings(paths)
    equal_frame(text_frame(pop_file), pop.reset_index(drop=True), "filtered population")
    equal_frame(text_frame(counts_file), counts.rename_axis("slide_id").reset_index(),
                "filtered count inventory")
    equal_frame(text_frame(base_file), eligibility, "base eligibility")
    reread_matrix = v4.load_distance(matrix_file, list(pop.index))
    if not reread_matrix.equals(matrix) or anchor_file.read_text().splitlines() != anchors:
        raise ValueError("Filtered matrix or new anchor identity readback failed")
    if json.loads(report_file.read_text()) != report or report["anchor_list_sha256"] != bound[str(anchor_file.resolve())]:
        raise ValueError("Stage1 evidence readback failed")
    v4.verify_bindings(source)
    v4.verify_bindings(bound)
    v4.verify_bindings(source)
    v4.verify_bindings(bound)
    marker = run / "STAGE1_COMPLETE"
    with marker.open("x") as f:
        f.write(json.dumps({"status": "STAGE1_OPTIMAL_ONLY_NO_FINAL_POOL",
                            "source_binding_set_sha256": hashlib.sha256(
                                json.dumps(source, sort_keys=True).encode()).hexdigest(),
                            "artifact_sha256": bound}, indent=2) + "\n")
    bound.update(file_bindings([marker]))
    return bound


def final_write(run: Path, workbook: Path, cohort: str, base_n: int,
                pop: pd.DataFrame, ali: pd.DataFrame, counts: pd.DataFrame,
                eligible: set[str], anchors: list[str], original: dict[str, list[str]],
                rectangular: dict[str, list[str]], selected: list[str], checks: dict,
                stage1_info: dict, stage2_info: dict, eligibility: pd.DataFrame,
                source: dict[str, str], staged: dict[str, str]) -> None:
    pool_file, swaps_file, ids_file, report_file = (
        run / name for name in ("pool.csv", "single_swaps.csv", "selected_ids.txt", "run.json")
    )
    destinations = (pool_file, swaps_file, ids_file, report_file, workbook, run / "COMPLETE")
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("Final output already exists; no partial artifact may be overwritten")
    if not workbook.parent.is_dir():
        raise ValueError("Workbook output parent is missing")
    marker = run / "STAGE1_COMPLETE"
    receipt = json.loads(marker.read_text())
    anchor_file, stage1_report_file = run / "stage1_anchors.txt", run / "stage1.json"
    marker_key = str(marker.resolve())
    stage1_report = json.loads(stage1_report_file.read_text())
    required_stage1 = {"filtered_population.csv", "filtered_distance_matrix.pkl",
                       "filtered_tissue_counts.csv", "base_eligibility.csv",
                       "stage1_anchors.txt", "stage1.json", "STAGE1_COMPLETE"}
    if (receipt.get("status") != "STAGE1_OPTIMAL_ONLY_NO_FINAL_POOL" or
            {Path(name).name for name in staged} != required_stage1 or
            receipt.get("source_binding_set_sha256") != hashlib.sha256(
                json.dumps(source, sort_keys=True).encode()).hexdigest() or
            staged.get(marker_key) != v4.digest(marker) or
            receipt.get("artifact_sha256") != {name: sha for name, sha in staged.items()
                                                if name != marker_key} or
            anchor_file.read_text().splitlines() != anchors or
            stage1_report.get("selected_anchor_ids") != anchors or
            stage1_report.get("anchor_list_sha256") != staged.get(str(anchor_file.resolve()))):
        raise ValueError("Stage1 anchor evidence/receipt does not bind the chosen solution")
    if (len(eligibility) != base_n or
            int(eligibility.final_candidate.sum()) != len(eligible) or
            stage1_info.get("objective") != len(anchors)):
        raise ValueError("Stage1 and final population denominators disagree")
    if any(counts.at[s, "detected_tissue_tiles"] < 800 for s in selected):
        raise ValueError("Pool includes slide under the hard 800-tile floor")
    pool = v4.case_table(selected, cohort, pop, ali, counts)
    swaps = v4.single_swaps(selected, pop, ali, eligible, original, rectangular, anchors, counts)
    for swap in swaps:
        incoming, outgoing = swap["incoming_slide_id"], swap["outgoing_slide_id"]
        candidate = [incoming if s == outgoing else s for s in selected]
        if (counts.at[incoming, "detected_tissue_tiles"] < 800 or
                not all(row["passed"] for row in v4.selection_checks(
                    candidate, pop, ali, eligible, original, rectangular, anchors).values())):
            raise ValueError("Unqualified or invalid individual swap")
    swaps_frame = pd.DataFrame(swaps, columns=SWAP_COLUMNS)
    groups = pd.DataFrame(
        [{"group": g, "population": int(n), "selected": int(pool.tumor_group.eq(g).sum())}
         for g, n in sorted(pop.tumor_group.value_counts().items())]
    )
    checks_frame = pd.DataFrame([{"constraint": label, **detail} for label, detail in checks.items()])
    if (checks != v4.selection_checks(selected, pop, ali, eligible, original, rectangular, anchors)
            or not all(row["passed"] for row in checks.values()) or
            stage2_info.get("objective") != int(pool.tissue_band.eq("borderline_800_999").sum())):
        raise ValueError("Final constraints or objective differ from independently recounted pool")
    readme = pd.DataFrame({"README": [
        "Plan0009/v5 author/QC review ONLY; no blinded-reader release.",
        f"Base {base_n}; quantity-eligible study population {len(pop)}; final candidates {len(eligible)}.",
        "Uniform FULL detector-positive tissue count >=800 applied BEFORE both ILPs to Ali and non-Ali.",
        "Tier2 can be a filtered-population anchor/target but is never a final candidate.",
        "Stage1: minimum self-plus-five outgoing cover on the freshly filtered square matrix.",
        "Stage2: candidate covers its five nearest OTHER new anchors, plus itself if an anchor.",
        "Covered new anchors compose through their freshly rebuilt population neighborhoods.",
        "800-999 is borderline quantity; >=1000 is quantity-preferred, NOT image-QC passed.",
        "ALL selected cases QC PENDING. Counts do not certify diagnostic tumour area or image adequacy.",
        "STV full cluster coordinates unavailable: duplicate count NOT_ASSESSED; RET→RB provisional.",
        "CCDI full-union clusters are duplicate-free; own-cluster membership is not an eligibility test.",
        "CCDI Ali difficulty is MELD-derived; distances use UNI2+OT. STV–COMET overlap unresolved.",
        "Swaps pass hard constraints alone, not jointly; listed outgoing alternatives not exhaustive.",
        "Other optimal pools may exist; frozen stage1 anchors are not retried to improve stage2.",
        "No retrieval outcomes or PROSPER scores are used or displayed.",
    ]})
    covered_a, covered_p = v4.coverage(selected, original, rectangular, anchors)
    report = {
        "design": "plan0009_v5_uniform_ge800_two_stage_hierarchical",
        "cohort": cohort, "denominators": {"base": base_n, "quantity_eligible_population": len(pop),
                                         "final_candidates": len(eligible), "new_anchors": len(anchors)},
        "stage1_solver": stage1_info, "stage2_solver": stage2_info,
        "anchor_ids": anchors, "selected_ids": selected,
        "selected_borderline": int(pool.tissue_band.eq("borderline_800_999").sum()),
        "selected_below_800": int((pool.detected_tissue_tiles < 800).sum()),
        "class_counts": {"population": {k: int(v) for k, v in pop.tumor_group.value_counts().items()},
                         "selected": {k: int(v) for k, v in pool.tumor_group.value_counts().items()}},
        "constraint_checks": checks,
        "coverage": {"anchors": {"numerator": len(covered_a), "denominator": len(anchors)},
                     "filtered_population": {"numerator": len(covered_p), "denominator": len(pop)}},
        "graph": {"anchor_to_filtered_population": {a: original[a] for a in anchors},
                  "eligible_candidate_to_anchor": rectangular},
        "witnesses": {"anchor_to_selected": {a: [s for s in selected if a in rectangular[s]] for a in anchors},
                      "population_to_covered_anchor": {s: [a for a in anchors if a in covered_a
                                                            and s in original[a]] for s in pop.index}},
        "single_swap_options": swaps, "single_swap_option_count": len(swaps),
        "source_bindings_sha256": source, "stage1_artifact_sha256": staged,
        "warning": "AUTHOR/QC REVIEW ONLY; all QC PENDING; no clinical adequacy or reader release",
        "argv": sys.argv,
    }
    with pool_file.open("x", newline="") as f:
        pool.to_csv(f, index=False)
    with swaps_file.open("x", newline="") as f:
        swaps_frame.to_csv(f, index=False)
    with ids_file.open("x") as f:
        f.write("\n".join(selected) + "\n")
    with report_file.open("x") as f:
        f.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with pd.ExcelWriter(workbook, engine="openpyxl", mode="x") as writer:
        pool.to_excel(writer, sheet_name="pool", index=False)
        swaps_frame.to_excel(writer, sheet_name="single_swaps", index=False)
        groups.to_excel(writer, sheet_name="groups", index=False)
        checks_frame.to_excel(writer, sheet_name="constraints", index=False)
        eligibility.to_excel(writer, sheet_name="eligibility", index=False)
        readme.to_excel(writer, sheet_name="README", index=False)
    final_bound = file_bindings([pool_file, swaps_file, ids_file, report_file, workbook])
    equal_frame(text_frame(pool_file), pool, "final pool CSV")
    equal_frame(excel_frame(workbook, "pool"), pool, "final pool workbook")
    equal_frame(text_frame(swaps_file), swaps_frame, "single-swap CSV")
    equal_frame(excel_frame(workbook, "single_swaps"), swaps_frame, "single-swap workbook")
    for sheet, frame in (("groups", groups), ("constraints", checks_frame),
                         ("eligibility", eligibility), ("README", readme)):
        equal_frame(excel_frame(workbook, sheet), frame, f"workbook {sheet}")
    if (json.loads(report_file.read_text()) != report or ids_file.read_text().splitlines() != selected or
            len(pool) != v4.BUDGET or not pool.qc_status.eq("PENDING").all() or
            not set(swaps_frame.incoming_slide_id) <= eligible):
        raise ValueError("Final report, identities or pending QC readback failed")
    v4.verify_bindings(source)
    v4.verify_bindings(staged)
    v4.verify_bindings(source)
    v4.verify_bindings(staged)
    v4.verify_bindings(final_bound)
    complete = {"status": "V5_SOURCE_BOUND_AUTHOR_QC_REVIEW_ONLY",
                "source_binding_set_sha256": hashlib.sha256(
                    json.dumps(source, sort_keys=True).encode()).hexdigest(),
                "output_sha256": {**staged, **final_bound},
                "warning": "QC PENDING; not a blinded-reader release"}
    with (run / "COMPLETE").open("x") as f:
        f.write(json.dumps(complete, indent=2) + "\n")


def execute(cohort: str, run: Path, workbook: Path, base: pd.DataFrame,
            matrix: pd.DataFrame, ali: pd.DataFrame, base_candidates: set[str],
            counts: pd.DataFrame, source: dict[str, str], scratch_parent: Path) -> dict:
    if run.exists() or workbook.exists():
        raise FileExistsError("Existing run directory/workbook: outputs may not be overwritten")
    pop, filtered_matrix, filtered_counts, eligible, eligibility = study_subset(
        cohort, base, matrix, ali, base_candidates, counts)
    nb = neighborhoods(filtered_matrix)
    anchors, witness, stage1_info = solve_stage1(nb, scratch_parent)
    staged = stage1_write(run, pop, filtered_matrix, filtered_counts, eligibility,
                          anchors, witness, stage1_info, nb, source)
    original, rectangular = v4.graphs(filtered_matrix, anchors,
                                       [s for s in pop.index if s in eligible])
    selected, checks, stage2_info = solve_stage2(
        pop, ali, filtered_counts, eligible, anchors, original, rectangular, scratch_parent)
    final_write(run, workbook, cohort, len(base), pop, ali, filtered_counts, eligible,
                anchors, original, rectangular, selected, checks,
                stage1_info, stage2_info, eligibility, source, staged)
    return {"cohort": cohort, "stage1_anchors": len(anchors),
            "borderline_selected": stage2_info["objective"],
            "run_dir": str(run), "workbook": str(workbook)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("stv", "ccdi"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    args = parser.parse_args()
    out_raw = os.environ.get("PROSPER_OUT")
    out = (REPO / "prosper_revision_output").resolve()
    if not out_raw or Path(out_raw).resolve() != out:
        parser.error("Export PROSPER_OUT=$PWD/prosper_revision_output from repository root")
    if args.run_dir.exists() or args.workbook.exists():
        parser.error("Existing run path/workbook: never overwrite")
    if (args.run_dir.parent.resolve() != out / "ilp_representatives" or
            args.workbook.parent.resolve() != REPO / "docs/pathologist_study" or
            args.workbook.suffix.lower() != ".xlsx"):
        parser.error("Run must be directly under local OUT/ilp_representatives; workbook in docs/pathologist_study")
    work = out / "ilp_representatives" / f"{args.cohort}_original_core_eval_20261001"
    inventory = out / "ilp_representatives" / v4.INVENTORY
    source = source_bindings(args.cohort, work, inventory)
    base, _legacy_anchors_for_lineage_only = v4.load_population(args.cohort, work, out)
    ids = list(base.index)
    matrix = v4.load_distance(work / "distance_matrix.pkl", ids)
    ali, base_candidates = v4.load_ali(args.cohort, v4.ALI_FILE, ids)
    counts = v4.load_counts(args.cohort, inventory / f"{args.cohort}_tissue_counts.csv", ids, source)
    outcome = execute(args.cohort, args.run_dir, args.workbook, base, matrix, ali,
                      base_candidates, counts, source, out / "ilp_representatives")
    print(json.dumps({"status": "v5_author_qc_review_only", **outcome}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from exc
