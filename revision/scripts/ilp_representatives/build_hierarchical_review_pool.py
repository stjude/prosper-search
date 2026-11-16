"""Build [analysis plan]'s source-bound, hierarchical v4 author/QC-review pool.

Export PROSPER_OUT to the repository-local output root. No real-data invocation
is authorized until the input and implementation reviews have passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pulp import (
    LpBinary, LpMinimize, LpProblem, LpSolutionOptimal, LpStatus, LpStatusOptimal,
    LpVariable, PULP_CBC_CMD, lpSum, value,
)

REPO = Path(__file__).resolve().parents[2]
ALI_FILE = REPO / "docs/pathologist_study/Ali_CCDI_STV_pathologist_shortlist_20260925.xlsx"
TEST_FILE = Path(__file__).with_name("test_build_hierarchical_review_pool.py")
INVENTORY = "hierarchical_v4_inputs_20261001_attempt002"
SOURCE_MANIFEST_SHA = "c1464a68b401beb31b905a58be288719644f19d545c9b4963a6cbb73c7a37f8c"
COUNT_SHA = {
    "stv": "5e9c242b36a70105aef3f0c6ca7969cbca7443163fe34955ad603b1b73600648",
    "ccdi": "eea794c89e91a3b3b8b70f2faa765328a552f985b7a3ea21a2a20f3b1ae311ae",
}
ANCHOR_SHA = {
    "stv": "f2c2f65ba1884960b4a1953787112e490f1534263554c9644df04ed0eb414700",
    "ccdi": "f17062126bf16f5dd716e226eea359b1c374174ad296235011de1bfefa768053",
}
EXPECTED = {"stv": (688, 158, 671, 14), "ccdi": (524, 130, 489, 12)}
DIFFICULTIES = {"easy", "moderate", "difficult", "very_difficult", "no_model_signal", "other"}
HARD = {"difficult", "very_difficult"}
BUDGET = 40


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add_binding(bindings: dict[str, str], path: Path, sha: str) -> None:
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise ValueError(f"Invalid source SHA256: {path}")
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError(f"Source path is not absolute/resolved: {path}")
    key = str(path)
    if key in bindings and bindings[key] != sha:
        raise ValueError(f"Conflicting source SHA256: {path}")
    bindings[key] = sha


def verify_bindings(bindings: dict[str, str]) -> None:
    for name, expected in bindings.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Source binding changed or was invalid: {name}")


def source_bindings(cohort: str, work: Path, inventory: Path) -> dict[str, str]:
    manifest = inventory / "source_sha256.json"
    if digest(manifest) != SOURCE_MANIFEST_SHA:
        raise ValueError("Unaccepted source inventory manifest")
    if not (inventory / "COMPLETE").is_file():
        raise ValueError("Source inventory is incomplete")
    raw = json.loads(manifest.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Invalid source inventory bindings")
    bindings: dict[str, str] = {}
    for name, sha in raw.items():
        add_binding(bindings, Path(name), sha)
    original = work / "input_sha256.json"
    for name, sha in json.loads(original.read_text()).items():
        add_binding(bindings, Path(name), sha)
    counts = inventory / f"{cohort}_tissue_counts.csv"
    if bindings.get(str(counts.resolve())) != COUNT_SHA[cohort]:
        raise ValueError("Frozen tissue-count CSV binding missing or changed")
    for path in (work / "population.csv", work / "distance_matrix.pkl", ALI_FILE):
        if str(path.resolve()) not in bindings:
            raise ValueError(f"Required input not bound in inventory: {path}")
    if str((work / "population.csv").resolve()) not in json.loads(original.read_text()):
        raise ValueError("Original core evaluation did not bind the current population")
    if str((work / "distance_matrix.pkl").resolve()) not in json.loads(original.read_text()):
        raise ValueError("Original core evaluation did not bind the current matrix")
    if cohort == "ccdi":
        accepted = work.parents[1] / "r1_2_scope_correction_20260930/manifest_ffpe_scope_clear.csv"
        if str(accepted.resolve()) not in bindings:
            raise ValueError("Accepted CCDI scope manifest is unbound")
    add_binding(bindings, manifest.resolve(), SOURCE_MANIFEST_SHA)
    add_binding(bindings, (work / "selected_round1.txt").resolve(), ANCHOR_SHA[cohort])
    for path in (original, inventory / "COMPLETE", Path(__file__).resolve(), TEST_FILE.resolve()):
        add_binding(bindings, path.resolve(), digest(path))
    verify_bindings(bindings)
    return bindings


def unique_ids(ids: list[str], label: str) -> None:
    if not ids or any(not isinstance(i, str) or not i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{label} has missing or duplicate IDs")


def load_population(cohort: str, work: Path, out: Path) -> tuple[pd.DataFrame, list[str]]:
    pop = pd.read_csv(work / "population.csv", dtype=str, keep_default_na=False)
    if not {"slide_id", "tumor_group", "patient_id"} <= set(pop.columns):
        raise ValueError("Population lacks required identifiers")
    ids = pop.slide_id.tolist()
    unique_ids(ids, "Population")
    expected_n, expected_a, _, expected_groups = EXPECTED[cohort]
    if len(ids) != expected_n or pop.tumor_group.eq("").any() or pop.patient_id.str.strip().eq("").any():
        raise ValueError("Unexpected population size or missing group/patient")
    if pop.tumor_group.eq("NORMAL").any() or pop.tumor_group.nunique() != expected_groups:
        raise ValueError("Unexpected in-library group membership")
    pop = pop.set_index("slide_id", drop=False)
    anchors = (work / "selected_round1.txt").read_text().splitlines()
    unique_ids(anchors, "Original round-1 anchors")
    if len(anchors) != expected_a or not set(anchors) <= set(ids):
        raise ValueError("Original round-1 anchor identity/count differs")
    if cohort == "stv":
        if not {"stv_dx", "stv_subclass", "comet_diagnosis"} <= set(pop.columns):
            raise ValueError("Missing source STV diagnosis fields")
    else:
        required = {"ccdi_diagnosis", "participant", "scope_status", "preparation_status", "prep_dicom"}
        if not required <= set(pop.columns):
            raise ValueError("Missing accepted CCDI metadata")
        if not (pop.scope_status == "CLEAR").all() or not (
            pop.preparation_status == "VERIFIED_FFPE"
        ).all() or not (pop.prep_dicom == "FFPE").all():
            raise ValueError("CCDI contains unresolved scope/preparation")
        if not (pop.patient_id == pop.participant).all() or not (pop.slide_id == pop.participant).all():
            raise ValueError("CCDI patient/participant/slide identity mismatch")
        manifest = pd.read_csv(out / "r1_2_scope_correction_20260930/manifest_ffpe_scope_clear.csv",
                               dtype=str, keep_default_na=False)
        unique_ids(manifest.slide_id.tolist(), "Accepted CCDI manifest")
        if set(manifest.slide_id) != set(ids):
            raise ValueError("Current CCDI population differs from accepted manifest")
        manifest = manifest.set_index("slide_id").loc[ids]
        for column in ("tumor_group", "ccdi_diagnosis", "participant",
                       "scope_status", "preparation_status", "prep_dicom"):
            if not pop[column].equals(manifest[column]):
                raise ValueError(f"CCDI population differs from accepted {column}")
    return pop, anchors


def load_distance(path: Path, ids: list[str]) -> pd.DataFrame:
    matrix = pd.read_pickle(path)
    if not isinstance(matrix, pd.DataFrame) or matrix.shape != (len(ids), len(ids)):
        raise ValueError("Distance matrix must be an exact square DataFrame")
    if (list(matrix.index) != ids or list(matrix.columns) != ids or
            matrix.index.has_duplicates or matrix.columns.has_duplicates):
        raise ValueError("Distance matrix ordered rows/columns must exactly match population")
    try:
        values = matrix.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Nonnumeric distance matrix") from exc
    if not (np.isfinite(values).all() and (values >= 0).all() and
            np.array_equal(np.diag(values), np.full(len(ids), 1_000_000.0))):
        raise ValueError("Nonfinite/negative distance or wrong original self-distance sentinel")
    if np.any(values[~np.eye(len(ids), dtype=bool)] >= 1_000_000):
        raise ValueError("Off-diagonal distances must be below the self-distance sentinel")
    return matrix


def graphs(matrix: pd.DataFrame, anchors: list[str], candidates: list[str], k: int = 5):
    ids = list(matrix.index)
    if len(anchors) <= k or len(ids) <= k or not set(anchors) <= set(ids) or not set(candidates) <= set(ids):
        raise ValueError("Insufficient anchors, candidates or population for the original graph")

    def nearest(i: str, targets: list[str]) -> list[str]:
        return sorted((a for a in targets if a != i),
                      key=lambda a: (float(matrix.at[i, a]), a))[:k]

    original = {s: [s, *nearest(s, ids)] for s in ids}
    if set().union(*(set(original[a]) for a in anchors)) != set(ids):
        raise ValueError("Original round-1 anchors do not cover the source population")
    rectangular = {s: [*([s] if s in anchors else []), *nearest(s, anchors)]
                   for s in candidates}
    supported = set().union(*(set(v) for v in rectangular.values()))
    if supported != set(anchors):
        raise ValueError(f"Eligible candidates cannot reach all anchors: {sorted(set(anchors)-supported)}")
    return original, rectangular


def load_ali(cohort: str, path: Path, ids: list[str]) -> tuple[pd.DataFrame, set[str]]:
    sheet, key = ("StJude", "file_name") if cohort == "stv" else ("CCDI-MCI", "patient_id")
    df = pd.read_excel(path, sheet_name=sheet, dtype=str, keep_default_na=False)
    if not {key, "selection_tier", "tier"} <= set(df.columns):
        raise ValueError("Ali sheet missing priority/difficulty/ID")
    if cohort == "stv":
        if not df[key].str.endswith(".tiff").all():
            raise ValueError("Ali STV file_name must have a .tiff suffix")
        df["slide_id"] = df[key].str[:-5]
    else:
        df["slide_id"] = df[key]
    unique_ids(df.slide_id.tolist(), "Ali workbook")
    df["tier"] = df.tier.replace({"no_signal": "no_model_signal"})
    if not df.selection_tier.isin(("Tier1", "Tier2")).all() or not df.tier.isin(DIFFICULTIES).all():
        raise ValueError("Unrecognized Ali priority/difficulty; no miscellaneous coercion")
    df = df[df.slide_id.isin(ids)].set_index("slide_id")
    eligible = set(ids) - set(df.index[df.selection_tier == "Tier2"])
    if len(eligible) != EXPECTED[cohort][2]:
        raise ValueError("Unexpected Tier2 exclusions; do not silently change eligibility")
    return df, eligible


def integer_column(df: pd.DataFrame, column: str, allow_zero: bool = False) -> pd.Series:
    if not df[column].str.fullmatch(r"(0|[1-9][0-9]*)").all():
        raise ValueError(f"Missing, duplicate-format or noninteger {column}")
    nums = df[column].map(int)
    if ((nums < 0) if allow_zero else (nums <= 0)).any():
        raise ValueError(f"Invalid nonpositive {column}")
    return nums


def load_counts(cohort: str, path: Path, ids: list[str],
                bindings: dict[str, str]) -> pd.DataFrame:
    if digest(path) != COUNT_SHA[cohort]:
        raise ValueError("Count manifest differs from frozen inventory")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"slide_id", "detected_tissue_tiles", "mosaic_tiles", "tissue_band",
                "count_method", "count_source_path", "count_source_sha256", "mosaic_meta_path",
                "mosaic_meta_sha256", "tile_size_px", "tissue_mask_threshold",
                "cluster_duplicate_count"}
    if not required <= set(df.columns):
        raise ValueError("Count manifest is missing required lineage/count columns")
    unique_ids(df.slide_id.tolist(), "Tissue inventory")
    if set(df.slide_id) != set(ids) or len(df) != len(ids):
        raise ValueError("Tissue inventory must cover exactly the population once")
    df = df.set_index("slide_id").loc[ids]
    tissue = integer_column(df, "detected_tissue_tiles")
    mosaic = integer_column(df, "mosaic_tiles")
    if (tissue < mosaic).any():
        raise ValueError("Detected tissue count must cover mosaic count")
    if cohort == "ccdi":
        duplicates = integer_column(df, "cluster_duplicate_count", allow_zero=True)
        if not (duplicates == 0).all():
            raise ValueError("CCDI full-union cluster locations must be duplicate-free")
    if not (df.tile_size_px == "512").all() or not (df.tissue_mask_threshold == "0.7").all():
        raise ValueError("Count inventory uses unexpected tile/mask definitions")
    methods = ({"original_provenance", "original_stdout"} if cohort == "stv"
               else {"feature_and_mosaic_unique_cluster_locs"})
    if not df.count_method.isin(methods).all():
        raise ValueError("Unknown full tissue derivation")
    expected_bands = tissue.map(tissue_band)
    if not df.tissue_band.equals(expected_bands.rename("tissue_band")):
        raise ValueError("Tissue bands disagree with integer detector counts")
    for _, row in df.iterrows():
        for prefix in ("count_source", "mosaic_meta"):
            path_string, sha = row[f"{prefix}_path"], row[f"{prefix}_sha256"]
            if bindings.get(path_string) != sha or str(Path(path_string).resolve()) != path_string:
                raise ValueError(f"Unbound {prefix} lineage for {row.name}")
    df["detected_tissue_tiles"] = tissue
    df["mosaic_tiles"] = mosaic
    df["cluster_duplicate_count"] = duplicates if cohort == "ccdi" else "NOT_ASSESSED"
    return df


def tissue_band(n: int) -> str:
    return "low_lt800" if n < 800 else ("borderline_800_999" if n < 1000 else "preferred_ge1000")


def qc_flag(n: int) -> str:
    return ("EXPLICIT_ADEQUACY_REQUIRED" if n < 800 else
            "BORDERLINE_QUANTITY_REVIEW" if n < 1000 else "ORDINARY_IMAGE_QC_REQUIRED")


def tissue_cost(selected: list[str], counts: pd.DataFrame) -> int:
    vals = counts.loc[selected, "detected_tissue_tiles"]
    return 41 * int((vals < 800).sum()) + int(((vals >= 800) & (vals < 1000)).sum())


def coverage(selected: list[str], original: dict[str, list[str]],
             rectangular: dict[str, list[str]], anchors: list[str]):
    selected_anchors = set().union(*(set(rectangular[s]) for s in selected)) if selected else set()
    composed = set().union(*(set(original[a]) for a in selected_anchors)) if selected_anchors else set()
    return selected_anchors, composed


def selection_checks(selected: list[str], pop: pd.DataFrame, ali: pd.DataFrame,
                     eligible: set[str], original: dict[str, list[str]],
                     rectangular: dict[str, list[str]], anchors: list[str]) -> dict[str, dict]:
    chosen = set(selected)
    observed: dict[str, dict] = {}

    def check(name: str, amount: int, minimum: int, maximum: int):
        amount = int(amount)
        observed[name] = {"observed": amount, "minimum": minimum,
                          "maximum": maximum, "passed": bool(minimum <= amount <= maximum)}

    check("selected", len(selected), BUDGET, BUDGET)
    check("duplicate_slide", len(selected)-len(chosen), 0, 0)
    check("ineligible_or_tier2", len(chosen-eligible), 0, 0)
    tiers = ali.loc[ali.index.intersection(chosen)]
    tier1 = tiers[tiers.selection_tier == "Tier1"]
    check("ali_tier1", len(tier1), 20, BUDGET)
    check("ali_hard", tier1.tier.isin(HARD).sum(), 8, BUDGET)
    check("ali_moderate", tier1.tier.eq("moderate").sum(), 5, BUDGET)
    check("ali_no_model_signal", tier1.tier.eq("no_model_signal").sum(),
          0 if "stv_dx" in pop else 4, BUDGET)
    check("ali_easy", tier1.tier.eq("easy").sum(), 0, 3)
    groups = pop.loc[list(chosen), "tumor_group"].value_counts() if chosen <= set(pop.index) else pd.Series(dtype=int)
    for group in sorted(pop.tumor_group.unique()):
        check(f"class:{group}", int(groups.get(group, 0)), 1,
              {"RB": 1, "MEL": 2}.get(group, 14))
    check("duplicate_patient", int(pop.loc[list(chosen), "patient_id"].duplicated().sum())
          if chosen <= set(pop.index) else BUDGET, 0, 0)
    covered_a, covered_p = coverage(selected, original, rectangular, anchors)
    check("anchors_covered", len(covered_a), len(anchors), len(anchors))
    check("composed_population_covered", len(covered_p), len(pop), len(pop))
    return observed


def solve(pop: pd.DataFrame, ali: pd.DataFrame, eligible: set[str],
          rectangular: dict[str, list[str]], anchors: list[str], counts: pd.DataFrame,
          solver=None, solver_dir: Path | None = None) -> tuple[list[str], dict]:
    ids = [s for s in pop.index if s in eligible]
    support = {a: [s for s in ids if a in rectangular[s]] for a in anchors}
    if any(not v for v in support.values()):
        raise ValueError("An anchor has no eligible covering candidate")
    model = LpProblem("v4_hierarchical_author_review", LpMinimize)
    x = {s: LpVariable(f"x_{j}", cat=LpBinary) for j, s in enumerate(ids)}
    model += lpSum((41 if counts.at[s, "detected_tissue_tiles"] < 800 else
                    1 if counts.at[s, "detected_tissue_tiles"] < 1000 else 0) * x[s] for s in ids)
    model += lpSum(x.values()) == BUDGET
    for a in anchors:
        model += lpSum(x[s] for s in support[a]) >= 1
    tier1 = ali.index[(ali.selection_tier == "Tier1") & ali.index.isin(ids)]
    model += lpSum(x[s] for s in tier1) >= 20
    for category, minimum in ((HARD, 8), ({"moderate"}, 5), ({"no_model_signal"}, 0 if "stv_dx" in pop else 4)):
        model += lpSum(x[s] for s in tier1 if ali.at[s, "tier"] in category) >= minimum
    model += lpSum(x[s] for s in tier1 if ali.at[s, "tier"] == "easy") <= 3
    for group, subset in pop.groupby("tumor_group"):
        items = [x[s] for s in subset.index if s in x]
        model += lpSum(items) >= 1
        model += lpSum(items) <= {"RB": 1, "MEL": 2}.get(group, 14)
    for _, subset in pop.groupby("patient_id"):
        model += lpSum(x[s] for s in subset.index if s in x) <= 1
    if solver is not None:
        model.solve(solver)
    else:
        if solver_dir is None or not solver_dir.is_dir() or not solver_dir.resolve().is_relative_to(REPO):
            raise ValueError("CBC needs an existing repository-local solver scratch parent")
        owned = solver_dir.resolve() / f".v4_cbc_{uuid.uuid4().hex}"
        owned.mkdir(mode=0o700, exist_ok=False)
        try:
            cbc = PULP_CBC_CMD(msg=False, threads=1, timeLimit=300)
            cbc.tmpDir = str(owned)
            model.solve(cbc)
        finally:
            shutil.rmtree(owned)
    info = {"problem_status": LpStatus.get(model.status, str(model.status)),
            "solution_status": model.sol_status, "solution_optimal_code": LpSolutionOptimal}
    if model.status != LpStatusOptimal or model.sol_status != LpSolutionOptimal:
        raise ValueError(f"No proven optimal shortlist; no incumbent released: {info}")
    selected = [s for s in ids if x[s].varValue is not None and x[s].varValue > 0.5]
    if len(selected) != BUDGET or any(x[s].varValue not in (0, 1) for s in ids):
        raise ValueError("Unusable integer solution values")
    info["objective"] = int(round(value(model.objective)))
    if info["objective"] != tissue_cost(selected, counts):
        raise ValueError("Objective disagrees with independent tissue-band recount")
    return selected, info


def single_swaps(selected: list[str], pop: pd.DataFrame, ali: pd.DataFrame, eligible: set[str],
                 original: dict[str, list[str]], rectangular: dict[str, list[str]],
                 anchors: list[str], counts: pd.DataFrame) -> list[dict]:
    counts_a = Counter(a for s in selected for a in rectangular[s])
    base_cost = tissue_cost(selected, counts)
    options = []
    for incoming in sorted(eligible - set(selected)):
        alternatives = []
        incoming_support = set(rectangular[incoming])
        for outgoing in selected:
            if any(counts_a[a] == 1 and a not in incoming_support for a in rectangular[outgoing]):
                continue
            candidate = [incoming if s == outgoing else s for s in selected]
            checks = selection_checks(candidate, pop, ali, eligible, original, rectangular, anchors)
            if all(item["passed"] for item in checks.values()):
                alternatives.append((tissue_cost(candidate, counts) - base_cost, outgoing))
        if alternatives:
            delta, outgoing = min(alternatives)
            options.append({"incoming_slide_id": incoming, "outgoing_slide_id": outgoing,
                            "tissue_cost_delta": delta, "incoming_tissue_band": counts.at[incoming, "tissue_band"],
                            "incoming_qc_flag": qc_flag(counts.at[incoming, "detected_tissue_tiles"]),
                            "qc_status": "PENDING",
                            "valid_only_as_individual_swap": True})
    return sorted(options, key=lambda o: (o["tissue_cost_delta"], o["incoming_slide_id"],
                                          o["outgoing_slide_id"]))[:8]


def case_table(selected: list[str], cohort: str, pop: pd.DataFrame,
               ali: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    columns = (["slide_id", "patient_id", "tumor_group", "comet_diagnosis"] +
               (["file_name", "stv_dx", "stv_subclass"] if cohort == "stv" else
                ["ccdi_diagnosis", "participant", "scope_status", "preparation_status", "prep_dicom"]))
    rows = pop.loc[selected, columns].copy().reset_index(drop=True)
    rows.insert(0, "cohort", cohort)
    rows["ali_priority"] = [ali.at[s, "selection_tier"] if s in ali.index else "not_on_ali_list"
                            for s in selected]
    rows["ali_difficulty"] = [ali.at[s, "tier"] if s in ali.index and
                              ali.at[s, "selection_tier"] == "Tier1" else ""
                              for s in selected]
    for column in counts.columns:
        if column != "slide_id":
            rows[column] = ("NOT_ASSESSED" if cohort == "stv" and column == "cluster_duplicate_count"
                            else counts.loc[selected, column].tolist())
    rows["qc_status"] = "PENDING"
    rows["quantity_qc_flag"] = [qc_flag(counts.at[s, "detected_tissue_tiles"]) for s in selected]
    rows["adequacy_decision"] = ""
    rows["label_qualification"] = ([
        ("RET→RB provisional; " if pop.at[s, "stv_dx"] == "RET" else "") +
        "cross-library patient overlap unresolved" for s in selected
    ] if cohort == "stv" else "")
    return rows


def write_outputs(run_dir: Path, workbook: Path, cohort: str, selected: list[str],
                  pop: pd.DataFrame, ali: pd.DataFrame, eligible: set[str],
                  counts: pd.DataFrame, original: dict[str, list[str]],
                  rectangular: dict[str, list[str]], anchors: list[str],
                  solver_info: dict, bindings: dict[str, str]) -> None:
    if run_dir.exists() or workbook.exists():
        raise FileExistsError("Existing run directory/workbook: outputs may not be overwritten")
    if not run_dir.parent.is_dir() or not workbook.parent.is_dir():
        raise ValueError("Output parents must exist (no implicit production directories)")
    checks = selection_checks(selected, pop, ali, eligible, original, rectangular, anchors)
    if not all(v["passed"] for v in checks.values()):
        raise ValueError(f"Selection violates hard constraints: {checks}")
    cov_a, cov_p = coverage(selected, original, rectangular, anchors)
    pool = case_table(selected, cohort, pop, ali, counts)
    swaps = single_swaps(selected, pop, ali, eligible, original, rectangular, anchors, counts)
    witnesses_a = {a: [s for s in selected if a in rectangular[s]] for a in anchors}
    witnesses_p = {s: [a for a in anchors if a in cov_a and s in original[a]] for s in pop.index}
    report = {
        "cohort": cohort, "design": "plan0008_v4_hierarchical_candidate_to_fixed_round1_anchors",
        "solver": {"name": "CBC", "threads": 1, "time_limit_seconds": 300, **solver_info},
        "objective": {"low_weight": 41, "borderline_weight": 1, "value": tissue_cost(selected, counts),
                      "selected_low": int((pool.detected_tissue_tiles < 800).sum()),
                      "selected_borderline": int(pool.tissue_band.eq("borderline_800_999").sum())},
        "selected_ids": selected, "eligible_ids": sorted(eligible), "anchor_ids": anchors,
        "population_class_counts": {s: int(v) for s, v in pop.tumor_group.value_counts().items()},
        "selected_class_counts": {s: int(v) for s, v in pool.tumor_group.value_counts().items()},
        "constraints": checks,
        "coverage": {"anchors": {"numerator": len(cov_a), "denominator": len(anchors)},
                     "composed_population": {"numerator": len(cov_p), "denominator": len(pop)},
                     "original_anchor_union": sorted(set().union(*(set(original[a]) for a in anchors)))},
        "graph": {"direction": "selected candidate -> five OTHER nearest anchors plus self if anchor; "
                               "covered anchor -> its original self-plus-five population slides",
                  "original_anchor_to_population": {a: original[a] for a in anchors},
                  "candidate_to_anchors": rectangular},
        "witnesses": {"anchor_to_selected_candidates": witnesses_a,
                      "population_to_covered_anchors": witnesses_p},
        "single_swaps": swaps, "single_swap_option_count": len(swaps),
        "single_swap_warning": "Options each pass alone; they are not mutually compatible, and "
                               "listed outgoing choices are not exhaustive.",
        "source_bindings_sha256": bindings, "argv": sys.argv,
        "optimality_qualification": "Minimum for this frozen graph/population/constraints only; "
                                     "other equally optimal solutions may exist.",
        "release_warning": "AUTHOR/QC REVIEW ONLY; all selected QC statuses PENDING. "
                           "<800 requires explicit adequacy decision, 800-999 borderline review, "
                           ">=1000 quantity preferred but NOT image-QC passed. No blinded-reader release.",
    }
    readme = pd.DataFrame({"README": [
        "v4 author/QC review only; not blinded reader material or a released reader pool.",
        report["graph"]["direction"], report["release_warning"],
        "Tier1 is Ali's preferred list; only Tier1 contributes difficulty quotas. No Tier2 selected.",
        "Ali CCDI difficulty is MELD-derived; selection distance is original UNI2+OT.",
        "Full detector-positive tissue tiles differ from sampled mosaic tiles; neither certifies tumour area.",
        "STV full cluster coordinates were unavailable; duplicate-count status is NOT_ASSESSED. "
        "CCDI duplicate-free counts refer to full-union cluster locations, not each representative's own cluster.",
        "STV RET→RB mapping is provisional; COMET/STV patient non-overlap unresolved.",
        "Single swaps pass all constraints and composed coverage ALONE; options are not combinable without recheck.",
        report["optimality_qualification"],
        "No PROSPER retrieval outcome or diagnosis-performance values are included.",
    ]})
    swaps_frame = pd.DataFrame(swaps, columns=["incoming_slide_id", "outgoing_slide_id",
                                               "tissue_cost_delta", "incoming_tissue_band",
                                               "incoming_qc_flag", "qc_status",
                                               "valid_only_as_individual_swap"])
    groups_frame = pd.DataFrame(
        [{"group": k, "population": v, "selected": report["selected_class_counts"].get(k, 0)}
         for k, v in sorted(report["population_class_counts"].items())]
    )
    checks_frame = pd.DataFrame([{"constraint": k, **v} for k, v in checks.items()])
    run_dir.mkdir(exist_ok=False)
    pool.to_csv(run_dir / "pool.csv", index=False)
    swaps_frame.to_csv(run_dir / "single_swaps.csv", index=False)
    (run_dir / "selected_ids.txt").write_text("\n".join(selected) + "\n")
    (run_dir / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with pd.ExcelWriter(workbook, engine="openpyxl", mode="x") as writer:
        pool.to_excel(writer, sheet_name="pool", index=False)
        swaps_frame.to_excel(writer, sheet_name="single_swaps", index=False)
        groups_frame.to_excel(writer, sheet_name="groups", index=False)
        checks_frame.to_excel(writer, sheet_name="constraints", index=False)
        readme.to_excel(writer, sheet_name="README", index=False)
    output_paths = [run_dir / name for name in
                    ("pool.csv", "single_swaps.csv", "selected_ids.txt", "run.json")] + [workbook]
    output_bindings = {str(path.resolve()): digest(path) for path in output_paths}
    csv_pool = pd.read_csv(run_dir / "pool.csv", dtype=str, keep_default_na=False)
    xlsx_pool = pd.read_excel(workbook, sheet_name="pool", dtype=str, keep_default_na=False)
    csv_swaps = pd.read_csv(run_dir / "single_swaps.csv", dtype=str, keep_default_na=False)
    xlsx_swaps = pd.read_excel(workbook, sheet_name="single_swaps", dtype=str, keep_default_na=False)
    workbook_groups = pd.read_excel(workbook, sheet_name="groups", dtype=str, keep_default_na=False)
    workbook_checks = pd.read_excel(workbook, sheet_name="constraints", dtype=str, keep_default_na=False)
    workbook_readme = pd.read_excel(workbook, sheet_name="README", dtype=str, keep_default_na=False)
    saved = json.loads((run_dir / "run.json").read_text())
    if (csv_pool.slide_id.tolist() != selected or xlsx_pool.slide_id.tolist() != selected or
            not csv_pool.equals(xlsx_pool) or
            not csv_swaps.equals(swaps_frame.astype(str)) or
            not xlsx_swaps.equals(csv_swaps) or
            not workbook_groups.equals(groups_frame.astype(str)) or
            not workbook_checks.equals(checks_frame.astype(str)) or
            not workbook_readme.equals(readme.astype(str)) or
            csv_pool.qc_status.tolist() != ["PENDING"] * BUDGET or
            xlsx_pool.qc_status.tolist() != ["PENDING"] * BUDGET or
            saved != report or
            (run_dir / "selected_ids.txt").read_text().splitlines() != selected):
        raise ValueError("Output readback failed; partial output left without COMPLETE")
    verify_bindings(bindings)
    verify_bindings(output_bindings)
    receipt = {"status": "SOURCE_BOUND_AUTHOR_QC_REVIEW_ONLY",
               "source_binding_set_sha256": hashlib.sha256(
                   json.dumps(bindings, sort_keys=True).encode()).hexdigest(),
               "output_sha256": output_bindings,
               "warning": "QC PENDING; not a blinded-reader release"}
    with (run_dir / "COMPLETE").open("x") as f:
        f.write(json.dumps(receipt, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=("stv", "ccdi"))
    parser.add_argument("--run-dir", required=True, type=Path, help="new empty/nonexistent per-cohort run path")
    parser.add_argument("--workbook", required=True, type=Path, help="new author-review .xlsx, never overwritten")
    args = parser.parse_args()
    out_raw = os.environ.get("PROSPER_OUT")
    if not out_raw or Path(out_raw).resolve() != (REPO / "prosper_revision_output").resolve():
        parser.error("Export PROSPER_OUT=$PWD/prosper_revision_output from the repository root")
    out = Path(out_raw).resolve()
    if args.run_dir.exists() or args.workbook.exists():
        parser.error("Run path/workbook already exists; never overwrite")
    if args.workbook.suffix.lower() != ".xlsx":
        parser.error("--workbook must be an .xlsx")
    if (args.run_dir.parent.resolve() != out / "ilp_representatives" or
            args.workbook.parent.resolve() != REPO / "docs/pathologist_study"):
        parser.error("Run must be directly under repository-local OUT/ilp_representatives; "
                     "workbook must be under docs/pathologist_study")
    work = out / "ilp_representatives" / f"{args.cohort}_original_core_eval_20261001"
    inventory = out / "ilp_representatives" / INVENTORY
    bindings = source_bindings(args.cohort, work, inventory)
    pop, anchors = load_population(args.cohort, work, out)
    ids = list(pop.index)
    matrix = load_distance(work / "distance_matrix.pkl", ids)
    ali, eligible = load_ali(args.cohort, ALI_FILE, ids)
    counts = load_counts(args.cohort, inventory / f"{args.cohort}_tissue_counts.csv", ids, bindings)
    original, rectangular = graphs(matrix, anchors, [s for s in ids if s in eligible])
    selected, solver_info = solve(pop, ali, eligible, rectangular, anchors, counts,
                                  solver_dir=out / "ilp_representatives")
    write_outputs(args.run_dir, args.workbook, args.cohort, selected, pop, ali, eligible,
                  counts, original, rectangular, anchors, solver_info, bindings)
    print(json.dumps({"status": "proven_optimal", "cohort": args.cohort,
                      "objective": solver_info["objective"], "selected": len(selected),
                      "run_dir": str(args.run_dir), "workbook": str(args.workbook)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
