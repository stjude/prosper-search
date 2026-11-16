#!/usr/bin/env python3
"""[decision record] external synthesis from accepted saved rankings; no retrieval jobs."""

from __future__ import annotations

import hashlib
import json
import argparse
import os
import platform
import socket
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import r1_2_scope_correction as ccdi
import r1_3_external_correction as correction
import r1_3_kidrare_filter_extension as pilot
import r1_3_kidrare_filter_full as full
from prosper.evaluation import external_cli, metrics

DEST = "r1_3_external_synthesis_20260930"
ENDPOINTS = ("hit_at_5", "precision_at_5")
FAILURES = ("hit_miss", "closest_mismatch", "generous_incorrect")
PREPARATIONS = {"VERIFIED_FFPE": 524, "FROZEN_OCT": 201,
                "MIXED": 6, "NONE_UNKNOWN": 4}
PATTERNS = {
    "Mixed blastemal and epithelial nephroblastoma": (1, 55),
    "Mixed blastemal and stromal nephroblastoma": (19, 32),
    "Mixed blastemal, epithelial and stromal nephroblastoma": (29, 175),
    "Nephroblastoma, blastemal type": (8, 4),
    "Nephroblastoma, stromal type": (2, 88),
}
OUTPUTS = {
    "inventory": "query_configuration_inventory.csv",
    "broad": "broad_group_primary.csv",
    "aggregate": "aggregate_context.csv",
    "paired": "paired_primary_differences.csv",
    "coverage": "coverage_and_hit_yield.csv",
    "composition": "all_returned_label_slots.csv",
    "failure_composition": "failure_conditioned_label_slots.csv",
    "rank": "first_relevant_rank_distribution.csv",
    "recurrence": "reference_recurrence.csv",
    "transitions": "matched_endpoint_transitions.csv",
    "losses": "separate_coverage_losses.csv",
    "strata": "descriptive_strata_points.csv",
    "pattern_context": "wt_source_by_pattern_membership.csv",
    "secondary": "expanded_secondary_retrieval_points.csv",
    "accepted": "accepted_result_references.json",
    "provenance": "provenance.json",
}
EMPTY_COLUMNS = {
    "secondary": ["cohort", "population", "configuration", "metric", "point",
                  "interval_status"],
    "paired": ["cohort", "population", "configuration", "metric", "tumor_group",
               "difference", "difference_ci_lo", "difference_ci_hi"],
    "coverage": ["cohort", "population", "configuration", "metric", "tumor_group",
                 "n_returned", "n_confirmed_empty", "point", "ci_lo", "ci_hi"],
    "losses": ["cohort", "population", "configuration", "query", "tumor_group",
               "coverage_loss", "baseline_hit_at_5"],
    "transitions": ["cohort", "population", "configuration", "query", "endpoint",
                    "transition"],
    "strata": ["cohort", "population", "configuration", "stratum_type",
               "stratum", "endpoint", "n_target", "n_returned", "point"],
    "pattern_context": ["population", "configuration", "pathpt_pattern",
                        "source_group", "n_queries"],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    require(path.is_file(), f"Missing synthesis input: {path}")
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def unique(frame: pd.DataFrame, key: str, name: str) -> None:
    require(key in frame and frame[key].notna().all() and
            frame[key].astype(str).str.len().gt(0).all() and
            frame[key].is_unique, f"{name}: missing/duplicate {key}")


def csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


class BoundInputs(dict):
    """Snapshot each declared input on first binding, before it is consumed."""

    def __init__(self):
        super().__init__()
        self.initial: dict[str, str] = {}

    def __setitem__(self, name: str, path: Path) -> None:
        path = Path(path)
        if name in self:
            require(self[name].resolve() == path.resolve() and
                    digest(path) == self.initial[name],
                    f"Bound input replaced: {name}")
        else:
            self.initial[name] = digest(path)
        super().__setitem__(name, path)

    def update(self, other=(), **kwargs) -> None:
        for name, path in dict(other, **kwargs).items():
            self[name] = path

    def snapshot(self) -> dict[str, str]:
        require({name: digest(path) for name, path in self.items()} == self.initial,
                "Consumed input changed after its initial binding")
        return self.initial.copy()


def bind_provenance_pin(name: str, record: dict, bindings: dict[str, Path],
                        actual_path: Path | None = None) -> None:
    require(isinstance(record, dict) and
            set(("path", "sha256")) <= set(record) and
            isinstance(record["sha256"], str) and len(record["sha256"]) == 64,
            f"{name}: malformed accepted provenance binding")
    path = Path(record["path"])
    if actual_path is not None:
        require(path.resolve() == actual_path.resolve(),
                f"{name}: imported helper/source path changed")
    bindings[f"pinned_{name}"] = path
    bound = (bindings.initial[f"pinned_{name}"] if isinstance(bindings, BoundInputs)
             else digest(path))
    require(bound == record["sha256"] and digest(path) == bound,
            f"{name}: accepted source/helper bytes changed during binding")


def frozen_dependencies(out: Path, bindings: dict[str, Path], spec: Spec) -> None:
    r18 = out / ccdi.DEST_NAME / "provenance.json"
    r19 = out / correction.DEST / "provenance.json"
    for name, path, pinned in (("ccdi_provenance", r18, spec.ccdi_sha),
                               ("correction_provenance", r19, spec.correction_sha)):
        bind_provenance_pin(name, {"path": str(path), "sha256": pinned}, bindings)
        bindings[name] = path
    original = read(r18)["input_sha256"]
    corrected = read(r19)["inputs"]
    required18 = {"script", "shared_metrics", "shared_external_cli",
                  "shared_resample", "comet_metadata", "policy",
                  "manifest_all", "manifest_inscope", "slide_preparation",
                  "series_preparation", "ccdi_metadata"} | {
                      f"{method}_{kind}" for method in ccdi.CONFIGS
                      for kind in ("per_query", "ranked", "json")}
    required19 = {"implementation", "implementation_tests", "metrics_helper",
                  "external_helper", "comet_metadata", "pathpt", "keep_wt",
                  "stv_source"}
    require(required18 <= set(original) and required19 <= set(corrected),
            "Missing accepted provenance dependencies")
    actual_code = {
        "script": Path(ccdi.__file__), "shared_metrics": Path(metrics.__file__),
        "metrics_helper": Path(metrics.__file__),
        "shared_external_cli": Path(external_cli.__file__),
        "external_helper": Path(external_cli.__file__),
        "shared_resample": Path(ccdi.resample.__file__),
        "implementation": Path(correction.__file__),
        "implementation_tests": correction.ROOT / "scripts/test_r1_3_external_correction.py"}
    for name, record in (
        *((f"ccdi_{key}", original[key]) for key in sorted(required18)),
        *((f"correction_{key}", corrected[key]) for key in sorted(required19)),
    ):
        code_key = name.removeprefix("ccdi_").removeprefix("correction_")
        bind_provenance_pin(name, record, bindings, actual_code.get(code_key))
    require(original["shared_metrics"]["sha256"] ==
            corrected["metrics_helper"]["sha256"] and
            original["shared_external_cli"]["sha256"] ==
            corrected["external_helper"]["sha256"] and
            original["comet_metadata"]["sha256"] ==
            corrected["comet_metadata"]["sha256"],
            "Accepted CCDI/KidRare shared metadata/helpers disagree")
    if isinstance(bindings, BoundInputs):
        bindings.snapshot()


@dataclass(frozen=True)
class Spec:
    b: int = 5000
    seed: int = 123
    stv_main: int = 696
    stv_patients: int = 350
    stv_both: int = 688
    stv_both_patients: int = 346
    ccdi_main: int = 524
    ccdi_allprep: int = 735
    kid_main: int = 1581
    kid_legacy: int = 1227
    kid_both: int = 1531
    kid_returned: int = 1580
    new_kid: int = 354
    old_query_summary_sha: str = "43979841c9fe840c7aa85a4103066ddbe7ba7120a9fa93c4f3e3fc70f528eda4"
    ccdi_sha: str = "6b62801c9fb944eb210082a16662c993036a6dd800c758eb535021f7a002e8c4"
    correction_sha: str = "f7f5bdf344c2011a3dda46457cc35b4620d35dbc5657db518ecd047ff61c6311"
    correction_manifest_sha: str = "da8df106e4ae4d6693d886ec7f0013a3621043f448be5bf26302b4e22c3b91c2"
    full_driver_sha: str = "cb310f64c1adbc224d8675c5f205a777c9695606801fead7359f6ebe00bee10d"
    full_tests_sha: str = "d3aebc7cf7b76b55514c50ab3c8d4c409009522253b5150f5b3c61b4fca4df63"
    full_manifest_sha: str = "02f7012a48f0cb5d50d3ec41cd59a311f6894294f106793ed88680cbfb348b37"
    full_preflight_sha: str = "c8152f53cfd3066521c6d73323634ffd22db063ad3376078542c619794e9b01f"
    reviewer: str = "3524c027-feb9-412e-83ed-b8967b6183e9"


CONTRACT = Spec()


@dataclass
class Analysis:
    cohort: str
    population: str
    configuration: str
    role: str
    target: pd.DataFrame
    ranked: pd.DataFrame
    query: pd.DataFrame
    empty: set[str]
    source: str


def bind_retrieval_modules(manifest: dict, spec: Spec,
                           bindings: dict[str, Path]) -> None:
    records = manifest["bindings"]
    modules = {
        "pilot_driver": Path(pilot.__file__).resolve(),
        "pilot_tests": Path(pilot.__file__).resolve().with_name(
            "test_r1_3_kidrare_filter_extension.py"),
        "full_driver": Path(full.__file__).resolve(),
        "full_tests": Path(full.__file__).resolve().with_name(
            "test_r1_3_kidrare_filter_full.py"),
    }
    require(records["full_driver"]["sha256"] == spec.full_driver_sha and
            records["full_tests"]["sha256"] == spec.full_tests_sha,
            "Accepted full-driver/test pin disagrees with contract")
    for name, actual in modules.items():
        bind_provenance_pin(f"retrieval_{name}", records[name], bindings, actual)


def full_gate(out: Path, spec: Spec) -> dict[str, Path]:
    root = out / full.DEST
    files = {name: root / name for name in
             ("COMPLETE.json", "FULL_INDEPENDENT_REVIEW.json", "manifest.json",
              "merge_receipt.json", "preflight_receipt.json")}
    require(digest(files["manifest.json"]) == spec.full_manifest_sha,
            "Frozen full manifest changed")
    manifest = read(files["manifest.json"])
    require(digest(files["manifest.json"]) == spec.full_manifest_sha,
            "Frozen full manifest changed during read")
    module_bindings = BoundInputs()
    bind_retrieval_modules(manifest, spec, module_bindings)
    module_snapshot = module_bindings.snapshot()
    files.update(module_bindings)
    snapshot = {name: digest(path) for name, path in files.items()}
    require(digest(Path(full.__file__)) == spec.full_driver_sha and
            digest(files["pinned_retrieval_full_tests"]) == spec.full_tests_sha and
            digest(files["preflight_receipt.json"]) == spec.full_preflight_sha,
            "Frozen full implementation/preflight binding changed")
    complete = read(files["COMPLETE.json"])
    review = read(files["FULL_INDEPENDENT_REVIEW.json"])
    require(complete["status"] == "technical_full_extension_for_independent_review" and
            review["status"] == "accepted_technical_full_extension_for_scientific_synthesis"
            and review["complete_sha256"] == digest(files["COMPLETE.json"]) and
            review["manifest_sha256"] == digest(files["manifest.json"]) and
            review["merge_receipt_sha256"] == digest(files["merge_receipt.json"]) and
            review["reviewer"]["agent_id"] == spec.reviewer and
            review["blocking_findings"] == [] and
            review["not_scientific_gate_c_approval"] is True,
            "Independent full-extension acceptance/linkage missing or invalid")
    full.Full(out).verify_complete_chain()
    require(module_bindings.snapshot() == module_snapshot,
            "Imported retrieval helper changed during full gate")
    require({name: digest(path) for name, path in files.items()} == snapshot,
            "Full gate record changed during chain verification")
    return files


def weights(pop: pd.DataFrame, cohort: str, spec: Spec) -> tuple[np.ndarray, dict]:
    unique(pop, "slide_id", "resampling population")
    require("tumor_group" in pop and pop.tumor_group.ne("").all() and
            pop.tumor_group.notna().all(), "Missing bootstrap truth")
    if cohort == "ccdi":
        unique(pop, "participant", "CCDI singleton participant")
        require(pop.participant.ne("").all(), "Missing CCDI participant")
        w = ccdi.draw_weights(pop, b=spec.b, seed=spec.seed)
        unit = "unique supplied CCDI participant (singleton)"
    else:
        clusters = "patient_id" if cohort == "stv" else "slide_id"
        require(clusters in pop and pop[clusters].notna().all() and
                pop[clusters].ne("").all(), "Missing patient/nominal-slide ID")
        w = correction.cluster_weights(pop, clusters, b=spec.b, seed=spec.seed)
        unit = ("supplied ST_ patient; whole-cluster multiplicity" if cohort == "stv"
                else "nominal slide; no patient-level independence claim")
    require(w.shape == (spec.b, len(pop)) and (w >= 0).all() and
            (w.sum(axis=1) > 0).all(), "Invalid shared population draw weights")
    return w, {"B": spec.b, "seed": spec.seed, "unit": unit,
               "ordered_ids_sha256": hashlib.sha256(
                   "".join(q + "\n" for q in pop.slide_id).encode()).hexdigest(),
               "weight_sha256": hashlib.sha256(w.tobytes()).hexdigest(),
               "min_resampled_slides": int(w.sum(axis=1).min()),
               "max_resampled_slides": int(w.sum(axis=1).max()),
               "fixed_classes": sorted(pop.tumor_group.unique())}


def unit_ids(pop: pd.DataFrame, cohort: str) -> tuple[str, ...]:
    unique(pop, "slide_id", "resampling population")
    require("tumor_group" in pop and pop.tumor_group.notna().all() and
            pop.tumor_group.ne("").all(), "Missing bootstrap truth")
    column = {"stv": "patient_id", "kidrare": "slide_id",
              "ccdi": "participant"}.get(cohort)
    require(column is not None and column in pop and pop[column].notna().all()
            and pop[column].ne("").all(), "Missing resampling unit")
    if cohort == "ccdi":
        unique(pop, "participant", "CCDI singleton participant")
    elif cohort == "stv":
        require(pop.groupby("patient_id").tumor_group.nunique().eq(1).all(),
                "Cross-class patient cluster")
    return tuple(pop[column])


def shared_weights(cache: dict, pop: pd.DataFrame, cohort: str,
                   spec: Spec, estimand: str = "conditional_returned"
                   ) -> tuple[np.ndarray, dict]:
    require(estimand in ("conditional_returned", "full_target"),
            "Unknown resampling estimand")
    units = unit_ids(pop, cohort)
    key = (cohort, estimand, tuple(pop.slide_id), units, spec.b, spec.seed)
    truth = tuple(pop.tumor_group)
    if key not in cache:
        cache[key] = (*weights(pop, cohort, spec), truth)
    require(cache[key][2] == truth, "Shared weight population truth changed")
    return cache[key][:2]


def ratio(values: np.ndarray, pop: pd.DataFrame, w: np.ndarray,
          classes: list[str]) -> dict[str, tuple[float, np.ndarray, int]]:
    require(len(values) == len(pop) and np.isfinite(values).all() and
            w.shape[1] == len(pop) and
            sorted(pop.tumor_group.unique()) == classes,
            "Ratio population/values/classes mismatch")
    data: dict[str, tuple[float, np.ndarray, int]] = {}
    class_draws = []
    for label in classes:
        ix = pop.tumor_group.to_numpy() == label
        denom = w[:, ix].sum(axis=1)
        require((denom > 0).all(), f"Missing fixed class in draw: {label}")
        draws = (w[:, ix] @ values[ix]) / denom
        data[label] = (float(values[ix].mean()), draws, int(ix.sum()))
        class_draws.append(draws)
    denom = w.sum(axis=1)
    require((denom > 0).all(), "Empty weighted total")
    data["micro"] = (float(values.mean()), (w @ values) / denom, len(values))
    data["macro"] = (float(np.mean([data[c][0] for c in classes])),
                     np.mean(class_draws, axis=0), len(values))
    return data


def estimate_rows(pop: pd.DataFrame, w: np.ndarray, values: np.ndarray,
                  metric: str, identity: dict, classes: list[str],
                  baseline: np.ndarray | None = None) -> list[dict]:
    current = ratio(values, pop, w, classes)
    previous = ratio(baseline, pop, w, classes) if baseline is not None else None
    unit = unit_ids(pop, identity["cohort"])
    rows = []
    for group in [*classes, "micro", "macro"]:
        point, draws, n = current[group]
        mask = (pop.tumor_group.eq(group).to_numpy() if group not in
                ("micro", "macro") else np.ones(len(pop), dtype=bool))
        n_units = len(set(u for u, selected in zip(unit, mask) if selected))
        row = {**identity, "metric": metric, "tumor_group": group, "n": n,
               "n_slides": n, "n_resampling_units": n_units,
               "point": point, "ci_lo": float(np.percentile(draws, 2.5)),
               "ci_hi": float(np.percentile(draws, 97.5)),
               "B_effective": len(draws),
               "sparse_units": n_units < 8,
               "degenerate_interval": bool(np.all(draws == draws[0])),
               "sparse_or_degenerate": n_units < 8 or
               bool(np.all(draws == draws[0]))}
        if previous is not None:
            diff = draws - previous[group][1]
            row.update({"baseline_point": previous[group][0],
                        "difference": point - previous[group][0],
                        "difference_ci_lo": float(np.percentile(diff, 2.5)),
                        "difference_ci_hi": float(np.percentile(diff, 97.5))})
        rows.append(row)
    return rows


def validate_ranking(ranked: pd.DataFrame, query: pd.DataFrame,
                     truth: dict[str, str], source: str,
                     expected_labels: dict[str, str] | None = None) -> None:
    require(list(ranked.columns) == ["query", "rank", "neighbour",
                                     "neighbour_label", "distance"] and
            {"query", "true_label", "hit_k5", "generous_correct_k5",
             "pred_closest_k5", "tied_set_k5"} <= set(query.columns),
            f"{source}: ranking/query schema")
    unique(query, "query", source)
    require(len(ranked) == 10 * len(query) and set(ranked["query"]) ==
            set(query["query"]) and not ranked.duplicated(["query", "neighbour"]).any(),
            f"{source}: incomplete/duplicate ranking")
    for qid, group in ranked.groupby("query", sort=False):
        require(group["rank"].tolist() == list(range(1, 11)) and
                group["distance"].notna().all() and
                np.isfinite(group["distance"].to_numpy(dtype=float)).all() and
                (group["distance"].to_numpy(dtype=float) >= 0).all() and
                (group["distance"].to_numpy(dtype=float) < 1e6).all() and
                (expected_labels is None or
                 all(expected_labels.get(sid) == label for sid, label in zip(
                     group.neighbour, group.neighbour_label))),
                f"{source}/{qid}: rank order/reference/label/distance invalid")
    require(set(query["query"]) <= set(truth) and
            all(truth[q] == label for q, label in zip(query["query"], query.true_label)),
            f"{source}: truth/query membership changed")
    ordered = ranked.set_index(["query", "rank"]).loc[
        pd.MultiIndex.from_product([query["query"], range(1, 11)])].reset_index()
    n = len(query)
    derived = external_cli.per_query_frame(
        query["query"].tolist(), query.true_label.tolist(),
        ordered.neighbour.to_numpy().reshape(n, 10),
        ordered.neighbour_label.to_numpy().reshape(n, 10),
        ordered.distance.to_numpy(dtype=float).reshape(n, 10), (1, 3, 5, 10), 10)
    for col in ("hit_k5", "generous_correct_k5", "pred_closest_k5",
                "pred_generous_k5", "tied_set_k5", "top1_id", "top1_label"):
        require(derived[col].astype(str).tolist() == query[col].astype(str).tolist()
                if col not in ("hit_k5", "generous_correct_k5") else
                np.allclose(derived[col].to_numpy(dtype=float),
                            query[col].to_numpy(dtype=float), atol=0, rtol=0),
                f"{source}: saved endpoint differs from ranked labels ({col})")
    require(np.allclose(derived["mean_top5_distance"].to_numpy(dtype=float),
                        query["mean_top5_distance"].to_numpy(dtype=float),
                        atol=1e-10, rtol=0),
            f"{source}: saved mean distance differs")


def inventory(a: Analysis, descriptors: pd.DataFrame) -> pd.DataFrame:
    target = a.target
    unique(target, "slide_id", f"{a.cohort}/{a.population}")
    q = a.query.set_index("query")
    r = a.ranked
    require(set(q.index) == set(target.slide_id) - a.empty and
            set(a.empty) <= set(target.slide_id) and
            not (set(q.index) & a.empty), f"{a.configuration}: missing ranks are technical failures")
    ranks = {sid: group for sid, group in r.groupby("query", sort=False)}
    descriptor = descriptors.set_index("slide_id")
    require(set(target.slide_id) <= set(descriptor.index),
            f"{a.cohort}: missing source descriptors")
    rows = []
    for entry in target.itertuples(index=False):
        sid, truth = entry.slide_id, entry.tumor_group
        status = "confirmed_empty" if sid in a.empty else "returned"
        labels = ranks[sid].neighbour_label.tolist() if status == "returned" else []
        match = [i + 1 for i, lab in enumerate(labels) if lab == truth]
        first = (None if not labels else
                 "1" if match and match[0] == 1 else
                 "2-5" if match and match[0] <= 5 else
                 "6-10" if match else "not_observed_through_10")
        query = q.loc[sid] if status == "returned" else None
        row = {"cohort": a.cohort, "population": a.population,
               "configuration": a.configuration, "analysis_role": a.role,
               "query": sid, "tumor_group": truth,
               "resampling_unit": (entry.patient_id if a.cohort == "stv" else
                                   entry.participant if a.cohort == "ccdi" else sid),
               "source": a.source, "status": status,
               "filter_scored": a.configuration in (
                   "query_only", "both_sided", "query_tumor_ot", "both_tumor_ot"),
               "ot_attempted": status == "returned" and a.configuration in (
                   "whole_slide_ot", "query_only", "both_sided",
                   "query_tumor_ot", "both_tumor_ot"),
               "rank_complete": status == "returned",
               "metric_scored": status == "returned",
               "confirmed_empty": status == "confirmed_empty",
               "technical_failure": False, "paired": False,
               "relevant_top5_count": sum(l == truth for l in labels[:5])
               if labels else None,
               "precision_at_5": sum(l == truth for l in labels[:5]) / 5
               if labels else None,
               "hit_at_5": float(truth in labels[:5]) if labels else None,
               "first_relevant_rank": first,
               "closest_prediction": query.pred_closest_k5 if query is not None else None,
               "closest_correct": bool(query.pred_closest_k5 == truth)
               if query is not None else None,
               "generous_correct": bool(float(query.generous_correct_k5))
               if query is not None else None,
               "tied_set_k5": query.tied_set_k5 if query is not None else None,
               "vote_share_k5": float(query.vote_share_k5) if query is not None else None,
               "rank5_distance": float(ranks[sid].distance.iloc[4])
               if labels else None,
               "rank10_distance": float(ranks[sid].distance.iloc[9])
               if labels else None,
               "beyond_top10_cutoff_tie": "unavailable_not_saved" if labels else None}
        for key in ("corrected_label_source", "pathpt_subtype", "source_group",
                    "source_tiles", "retained_tumor_tiles",
                    "preparation_status", "collection", "overlap_with_COMET",
                    "RET_to_RB"):
            row[key] = descriptor.at[sid, key] if key in descriptor else None
        rows.append(row)
    return pd.DataFrame(rows)


def failure_views(a: Analysis, inv: pd.DataFrame) -> dict[str, list[dict]]:
    found = inv[inv.status == "returned"].set_index("query")
    ranks = a.ranked[a.ranked["rank"] <= 5]
    require(set(found.index) == set(ranks["query"]) and
            len(ranks) == 5 * len(found), "Failure denominators/ranks incomplete")
    rows = {"composition": [], "failure_composition": [],
            "rank": [], "recurrence": []}
    identity = {"cohort": a.cohort, "population": a.population,
                "configuration": a.configuration, "analysis_role": a.role}
    misses = {sid for sid in found.index if found.at[sid, "hit_at_5"] == 0}
    closest = {sid for sid in found.index if not found.at[sid, "closest_correct"]}
    generous = {sid for sid in found.index if not found.at[sid, "generous_correct"]}
    failures = dict(zip(FAILURES, (misses, closest, generous)))
    for truth, group in found.groupby("tumor_group", sort=True):
        ids = set(group.index)
        slots = ranks[ranks["query"].isin(ids)]
        labels = sorted(set(slots.neighbour_label) | {truth})
        for label in labels:
            count = int(slots.neighbour_label.eq(label).sum())
            rows["composition"].append({**identity, "tumor_group": truth,
                                        "neighbour_label": label, "slot_count": count,
                                        "n_returned_queries": len(group),
                                        "denominator_top5_slots": 5 * len(group),
                                        "slot_fraction": count / (5 * len(group))})
        for endpoint, failing in failures.items():
            selected = slots[slots["query"].isin(ids & failing)]
            for label in labels:
                n = len(ids & failing)
                rows["failure_composition"].append({
                    **identity, "tumor_group": truth, "endpoint": endpoint,
                    "neighbour_label": label,
                    "slot_count": int(selected.neighbour_label.eq(label).sum()),
                    "n_failure_queries": n, "denominator_failure_top5_slots": 5 * n,
                    "slot_fraction": (float(selected.neighbour_label.eq(label).sum() /
                                            (5 * n)) if n else None)})
        for category in ("1", "2-5", "6-10", "not_observed_through_10"):
            rows["rank"].append({**identity, "tumor_group": truth,
                                 "first_relevant_rank": category, "n_returned": len(group),
                                 "n_queries": int(group.first_relevant_rank.eq(category).sum())})
    for (ref, label), group in ranks.groupby(["neighbour", "neighbour_label"], sort=True):
        require(group["query"].is_unique, f"{ref}: duplicate reference exposure")
        base = {**identity, "reference_id": ref, "neighbour_label": label,
                "all_query_exposure": len(group),
                "n_all_returned_queries": len(found)}
        for endpoint, failing in failures.items():
            rows["recurrence"].append({**base, "endpoint": endpoint,
                                       "failure_exposure": int(group["query"].isin(failing).sum()),
                                       "n_failure_queries": len(failing)})
    return rows


def summarize(a: Analysis, inv: pd.DataFrame, spec: Spec, cache: dict,
              baseline: pd.DataFrame | None = None) -> tuple[list[dict], list[dict]]:
    target = a.target
    require(inv["query"].tolist() == target.slide_id.tolist(),
            "Analysis inventory/target order changed")
    classes = sorted(target.tumor_group.unique())
    identity = {"cohort": a.cohort, "population": a.population,
                "configuration": a.configuration, "analysis_role": a.role}
    if baseline is not None:
        identity.update({
            "alternative_configuration": a.configuration,
            "baseline_configuration": "whole_slide_ot",
            "comparison_id": f"{a.cohort}:{a.population}:{a.configuration}:vs:whole_slide_ot",
        })
    conditional = inv[inv.status == "returned"].copy()
    require(len(conditional) > 0 and
            set(conditional.tumor_group) == set(classes),
            "Undefined conditional class; do not drop a class")
    paired_pop = target.set_index("slide_id").loc[conditional["query"]].reset_index()
    w, provenance = shared_weights(cache, paired_pop, a.cohort, spec)
    broad = []
    for metric in ENDPOINTS:
        values = conditional[metric].to_numpy(dtype=float)
        previous = (baseline.set_index("query").loc[conditional["query"], metric]
                    .to_numpy(dtype=float) if baseline is not None else None)
        broad += [{**row, "estimand": "conditional_returned",
                   "population_order_sha256": provenance["ordered_ids_sha256"],
                   "weights_sha256": provenance["weight_sha256"],
                   "resampling_unit": provenance["unit"]}
                  for row in estimate_rows(paired_pop, w, values, metric,
                                           identity, classes, previous)]
    coverage = []
    if baseline is not None or (a.configuration == "whole_slide_ot" and
                                a.role in ("main", "historical_sensitivity")):
        w_all, prov_all = shared_weights(cache, target, a.cohort, spec,
                                        "full_target")
        prev = baseline.set_index("query").loc[target.slide_id] if baseline is not None else None
        if prev is not None:
            require(prev.status.eq("returned").all() and
                    (prev.tumor_group.to_numpy() == target.tumor_group.to_numpy()).all(),
                    "Same-target all-tissue baseline incomplete/misaligned")
        got = inv.set_index("query").loc[target.slide_id]
        status = got.status.eq("returned").to_numpy(dtype=float)
        hit = got.hit_at_5.fillna(0).to_numpy(dtype=float)
        for metric, values, control in (
            ("coverage", status, np.ones(len(target)) if prev is not None else None),
            ("hit_yield", hit, prev.hit_at_5.to_numpy(dtype=float)
             if prev is not None else None)):
            coverage += [{**row, "estimand": "full_target",
                          "n_returned": int(status[
                              target.tumor_group.eq(row["tumor_group"]).to_numpy()
                              if row["tumor_group"] not in ("micro", "macro")
                              else np.ones(len(target), dtype=bool)].sum()),
                          "n_confirmed_empty": row["n_slides"] - int(status[
                              target.tumor_group.eq(row["tumor_group"]).to_numpy()
                              if row["tumor_group"] not in ("micro", "macro")
                              else np.ones(len(target), dtype=bool)].sum()),
                          "population_order_sha256": prov_all["ordered_ids_sha256"],
                          "weights_sha256": prov_all["weight_sha256"],
                          "resampling_unit": prov_all["unit"]}
                         for row in estimate_rows(target, w_all, values, metric,
                                                  identity, classes, control)]
    return broad, coverage


def transitions(a: Analysis, inv: pd.DataFrame,
                baseline: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    old = baseline.set_index("query")
    require(set(inv["query"]) <= set(old.index), "Matched baseline missing queries")
    rows, losses = [], []
    identity = {"cohort": a.cohort, "population": a.population,
                "configuration": a.configuration, "analysis_role": a.role,
                "baseline_configuration": "whole_slide_ot"}
    for entry in inv.itertuples(index=False):
        prior = old.loc[entry.query]
        if entry.status == "confirmed_empty":
            losses.append({**identity, "query": entry.query,
                           "tumor_group": entry.tumor_group,
                           "baseline_hit_at_5": prior.hit_at_5,
                           "baseline_closest_correct": prior.closest_correct,
                           "baseline_generous_correct": prior.generous_correct,
                           "coverage_loss": True})
            continue
        for endpoint, col in (("hit_at_5", "hit_at_5"),
                              ("closest_correct", "closest_correct"),
                              ("generous_correct", "generous_correct")):
            before, after = bool(prior[col]), bool(getattr(entry, col))
            rows.append({**identity, "query": entry.query,
                         "tumor_group": entry.tumor_group,
                         "endpoint": endpoint, "baseline_correct": before,
                         "alternative_correct": after,
                         "transition": ("fixed" if not before and after else
                                        "broken" if before and not after else
                                        "unchanged_correct" if before else "unchanged_incorrect")})
    return rows, losses


def accepted_ranking(root: Path, prefix: str, truth: dict[str, str],
                     expected: set[str], bindings: dict[str, Path],
                     manifest: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    files = {kind: root / (prefix + suffix) for kind, suffix in (
        ("report", ".json"), ("query", "_per_query.csv"),
        ("ranked", "_ranked.csv.gz"))}
    consumed = {kind: digest(path) for kind, path in files.items()}
    for kind, path in files.items():
        if manifest is not None:
            entry = manifest["artifacts"][path.name]
            require(consumed[kind] == entry["sha256"] and
                    path.stat().st_size == entry["bytes"],
                    f"{prefix}: accepted output manifest mismatch ({kind})")
        bindings[f"{prefix}_{kind}"] = path
    report = read(files["report"])
    q = pd.read_csv(files["query"], dtype={"query": str, "true_label": str},
                    keep_default_na=False)
    r = pd.read_csv(files["ranked"], dtype={"query": str, "neighbour": str,
                                            "neighbour_label": str}, keep_default_na=False)
    require(set(q["query"]) == expected and len(q) == len(expected) and
            report.get("n", report["point_estimates"]["n_queries"]) == len(expected),
            f"{prefix}: accepted query count/set changed")
    validate_ranking(r, q, truth, prefix)
    qidx = q.set_index("query")
    for metric, column in (("micro_hit_at_5", "hit_k5"),
                           ("micro_precision_at_5", None)):
        if column is None:
            slot = r[r["rank"] <= 5]
            count = sum(truth[sid] == lab for sid, lab in zip(
                slot["query"], slot["neighbour_label"]))
            point = count / (5 * len(q))
        else:
            point = qidx[column].astype(float).mean()
        require(np.isclose(point, report["point_estimates"][metric],
                           atol=1e-12, rtol=0),
                f"{prefix}: accepted report and ranked outcomes disagree")
    for label, group in q.groupby("true_label"):
        slots = r[(r["rank"] <= 5) & r["query"].isin(group["query"])]
        hit = float(group.hit_k5.mean())
        prec = float(slots.neighbour_label.eq(label).sum() / (5 * len(group)))
        reported = report["per_class_primary" if "per_class_primary" in report
                          else "per_class_hit_precision"]
        if isinstance(reported, dict):
            entry = reported[label]
        else:
            entry = next(row for row in reported if row.get("tumor_group") == label)
        for key, value in (("hit_at_5", hit), ("precision_at_5", prec)):
            require(np.isclose(float(entry[key]), value, atol=1e-12, rtol=0),
                    f"{prefix}/{label}: accepted per-group point differs")
    require(np.isclose(float(report["point_estimates"]["macro_hit_at_5"]),
                       np.mean([float(group.hit_k5.mean()) for _, group in
                                q.groupby("true_label")]), atol=1e-12, rtol=0),
            f"{prefix}: accepted macro hit differs")
    require(np.isclose(float(report["point_estimates"]["macro_precision_at_5"]),
                       np.mean([float(r[(r["rank"] <= 5) &
                                        r["query"].isin(group["query"])]
                                      .neighbour_label.eq(label).sum() /
                                      (5 * len(group))) for label, group in
                                q.groupby("true_label")]), atol=1e-12, rtol=0),
            f"{prefix}: accepted macro precision differs")
    require({kind: digest(path) for kind, path in files.items()} == consumed,
            f"{prefix}: accepted source changed during read")
    return q, r, report


def same_scientific_values(actual, expected, label: str) -> None:
    require(type(actual) is type(expected) or
            (isinstance(actual, (int, float)) and isinstance(expected, (int, float))),
            f"{label}: accepted value type changed")
    if isinstance(actual, dict):
        require(set(actual) == set(expected), f"{label}: keys changed")
        for key in actual:
            same_scientific_values(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(actual, list):
        require(len(actual) == len(expected), f"{label}: length changed")
        for i, (left, right) in enumerate(zip(actual, expected)):
            same_scientific_values(left, right, f"{label}/{i}")
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        require(np.isfinite(actual) and np.isfinite(expected) and
                np.isclose(actual, expected, rtol=0, atol=1e-12),
                f"{label}: accepted numerical result changed")
    else:
        require(actual == expected, f"{label}: accepted value changed")


def verify_ccdi_configuration(
        root: Path, method: str, populations: dict[str, pd.DataFrame],
        original_q: pd.DataFrame, original_r: pd.DataFrame,
        configuration: dict, accepted: dict, draws: dict[str, np.ndarray],
        reference_labels: dict[str, str], truth: dict[str, str]) -> dict[str, dict]:
    require(set(original_q["query"]) == set(truth) and
            all(dict(zip(pop.slide_id, pop.tumor_group))[sid] == truth[sid]
                for pop in populations.values() for sid in pop.slide_id),
            f"{method}: original scope membership changed")
    validate_ranking(original_r, original_q, truth,
                     f"original_{method}", reference_labels)
    entries = {}
    for name, pop in populations.items():
        prefix = f"{method}_{name}"
        saved_q = pd.read_csv(root / f"{prefix}_per_query.csv",
                              dtype={"query": str, "true_label": str})
        saved_r = pd.read_csv(root / f"{prefix}_ranked.csv.gz",
                              dtype={"query": str, "neighbour": str,
                                     "neighbour_label": str})
        selected_q = original_q.loc[original_q["query"].isin(pop.slide_id)]
        selected_r = original_r.loc[original_r["query"].isin(pop.slide_id)]
        ccdi.compare_frames(selected_q.reset_index(drop=True), saved_q,
                            f"{prefix}: pinned original query export")
        ccdi.compare_frames(selected_r.reset_index(drop=True), saved_r,
                            f"{prefix}: pinned original reference/rank export")
        ordered = original_q.set_index("query").loc[pop.slide_id].reset_index()
        ranked = original_r.set_index(["query", "rank"]).loc[
            pd.MultiIndex.from_product([pop.slide_id, range(1, 11)])]
        ordered.attrs["ranked_labels"] = ranked.neighbour_label.to_numpy().reshape(
            len(pop), 10)
        recomputed, replicates = ccdi.score(pop, ordered, configuration, draws[name])
        reported = accepted[prefix]["report"]
        for field in ("query_class_counts", "point_estimates",
                      "per_class_hit_precision", "voting_k5",
                      "macro_generous_accuracy_k5", "confidence_intervals_95",
                      "secondary_endpoint_CI"):
            same_scientific_values(recomputed[field], reported[field],
                                   f"{prefix}/{field}")
        require(reported["population"] == name and
                reported["configuration"] == method and
                reported["n_participants"] == pop.participant.nunique() and
                reported["observed_classes"] == sorted(pop.tumor_group.unique()),
                f"{prefix}: accepted report population/configuration changed")
        entries[name] = {"draws": replicates, "point": {
            **{key: recomputed["point_estimates"][key] for key in (
                "micro_hit_at_5", "macro_hit_at_5", "micro_precision_at_5",
                "macro_precision_at_5")},
            "generous_accuracy_k5": recomputed["voting_k5"]["generous"]["accuracy"],
            "macro_generous_accuracy_k5": recomputed["macro_generous_accuracy_k5"]}}
    return entries


def verify_ccdi_paired(path: Path, entries: dict[str, dict[str, dict]],
                       populations: dict[str, pd.DataFrame],
                       configurations: dict[str, dict],
                       methods: tuple[str, ...]) -> None:
    require("whole_slide_ot" in methods and
            set(entries) == set(populations) and
            all(set(arms) == set(methods) for arms in entries.values()),
            "CCDI paired population/method replay incomplete")
    saved = read(path)
    require(isinstance(saved, list), "CCDI paired comparison schema changed")
    expected_ids = {(pop, arm) for pop in populations for arm in methods
                    if arm != "whole_slide_ot"}
    ids = [(row["population"], row["configuration"]) for row in saved]
    require(len(saved) == len(expected_ids) and len(ids) == len(set(ids)) and
            set(ids) == expected_ids, "CCDI paired comparison identity/coverage changed")
    for row in saved:
        name, arm = row["population"], row["configuration"]
        population = populations[name]
        expected = {
            "population": name, "configuration": arm,
            "baseline": "whole_slide_ot", "n_paired": len(population),
            "difference": ccdi.paired(entries[name][arm],
                                      entries[name]["whole_slide_ot"]),
            "observed_truth_classes": sorted(population.tumor_group.unique()),
            "library_label_sets_equal": (
                configurations[arm]["library_classes"] ==
                configurations["whole_slide_ot"]["library_classes"]),
            "note": "No macro F1 compared across distinct library label sets"}
        same_scientific_values(expected, row, f"CCDI paired/{name}/{arm}")


def replay_ccdi(out: Path, accepted: dict, bindings: dict[str, Path],
                spec: Spec) -> None:
    root = out / ccdi.DEST_NAME
    source_keys = (*ccdi.SOURCES, "ccdi_metadata", "policy", "comet_metadata")
    provenance = read(root / "provenance.json")["input_sha256"]
    source = {key: ccdi.read_csv(Path(provenance[key]["path"]))
              for key in source_keys}
    decision = ccdi.build_decisions(source, source["policy"])
    decision_path = root / "decision_table_765.csv"
    bindings["ccdi_decision_table"] = decision_path
    ccdi.compare_frames(decision, pd.read_csv(decision_path), "CCDI reconstructed decisions")
    library = source["comet_metadata"].set_index("Slide.Scan.File")
    require(library.index.is_unique, "CCDI COMET reference IDs duplicate")
    labels = library["Tumor.group"].to_dict()
    scope = ("ffpe_scope_clear", "ffpe_add_site_only",
             "ffpe_add_all_unresolved", "allprep_scope_clear")
    populations = {}
    draws_for_population = {}
    paired_entries: dict[str, dict[str, dict]] = {name: {} for name in scope}
    configurations = {}
    for name in scope:
        pop = decision.loc[decision[name]].copy()
        populations[name] = pop
        draws_for_population[name] = ccdi.draw_weights(pop, b=spec.b, seed=spec.seed)
        manifest_path = root / f"manifest_{name}.csv"
        bindings[f"ccdi_reconstructed_{name}"] = manifest_path
        ccdi.compare_frames(pop.reset_index(drop=True), pd.read_csv(manifest_path),
                            f"CCDI reconstructed membership/{name}")
    for method in ccdi.CONFIGS:
        original_q = pd.read_csv(Path(provenance[f"{method}_per_query"]["path"]),
                                 dtype={"query": str, "true_label": str})
        original_r = pd.read_csv(Path(provenance[f"{method}_ranked"]["path"]),
                                 dtype={"query": str, "neighbour": str,
                                        "neighbour_label": str})
        configuration = read(Path(provenance[f"{method}_json"]["path"]))["config"]
        configurations[method] = configuration
        require(len(original_q) == len(decision) == 765 and
                len(original_r) == 7650 and
                configuration["depth"] == 10 and
                configuration["k_values"] == list(ccdi.K),
                f"{method}: original source interface changed")
        entries = verify_ccdi_configuration(
            root, method, populations, original_q, original_r,
            configuration, accepted, draws_for_population, labels,
            dict(zip(decision.slide_id, decision.tumor_group)))
        for name, entry in entries.items():
            paired_entries[name][method] = entry
        if isinstance(bindings, BoundInputs):
            bindings.snapshot()
    paired_path = root / "paired_differences.json"
    bindings["ccdi_paired_differences.json"] = paired_path
    verify_ccdi_paired(paired_path, paired_entries, populations, configurations,
                       tuple(ccdi.CONFIGS))
    if isinstance(bindings, BoundInputs):
        bindings.snapshot()


def expanded_ranking(out: Path, arm: str, target: pd.DataFrame,
                     empty: set[str], metadata: pd.DataFrame, m: dict,
                     bindings: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = out / "similarity_matrix" / (
        f"combined_similarity_matrix_{full.MODELS[arm]}_ot_cosine.pkl")
    refs = m["refs"][arm]
    returned = set(target.slide_id) - empty
    expected = set(m["historic"][arm]) | set(m["ids"])
    require(returned == (expected if arm == "query_only" else
                         expected & set(target.slide_id)) and
            empty == {full.ZERO_LIVER},
            f"{arm}: expanded raw/target/empty membership differs")
    bindings[f"expanded_{arm}_matrix"] = path
    frame = full.matrix(path, sorted(expected), refs)
    meta = metadata.set_index("Slide.Scan.File")
    require(meta.index.is_unique and set(refs) <= set(meta.index),
            "Expanded COMET reference metadata incomplete")
    labels = meta["Tumor.group"].to_dict()
    patient = meta["UID.Subject"].to_dict()
    eligible = [sid for sid in refs if labels[sid] not in (
        ("MISC",) if arm == "query_only" else ("MISC", "NORMAL"))]
    require(eligible == [sid for sid in refs if sid in set(m["eligible"][arm])],
            f"{arm}: accepted eligible reference set/order differs")
    order = [sid for sid in target.slide_id if sid in returned]
    truth = target.set_index("slide_id").loc[order, "tumor_group"].tolist()
    ids, labs, distances, short = external_cli.rank_block(
        frame, eligible, labels, patient, order, None, 10)
    require(short == 0, f"{arm}: incomplete expanded top10 ranking")
    q = external_cli.per_query_frame(order, truth, ids, labs, distances,
                                     (1, 3, 5, 10), 10)
    r = pd.DataFrame({"query": np.repeat(order, 10),
                      "rank": np.tile(np.arange(1, 11), len(order)),
                      "neighbour": ids.ravel(), "neighbour_label": labs.ravel(),
                      "distance": distances.ravel()})
    validate_ranking(r, q, dict(zip(order, truth)), f"expanded_{arm}", labels)
    if isinstance(bindings, BoundInputs):
        require(digest(path) == bindings.initial[f"expanded_{arm}_matrix"],
                "Expanded matrix changed during ranking")
    return q, r


def descriptor_inputs(out: Path, spec: Spec, bindings: dict[str, Path]
                      ) -> tuple[dict[str, pd.DataFrame], dict, dict, set[str]]:
    r18 = out / ccdi.DEST_NAME
    r19 = out / correction.DEST
    paths = {
        "ccdi_complete": r18 / "COMPLETE.json",
        "ccdi_provenance": r18 / "provenance.json",
        "ccdi_main": r18 / "manifest_ffpe_scope_clear.csv",
        "ccdi_prep": r18 / "manifest_allprep_scope_clear.csv",
        "stv": r19 / "stv_source_782_eligibility.csv",
        "kid": r19 / "kidrare_source_2331_eligibility.csv",
        "correction_provenance": r19 / "provenance.json",
        "correction_manifest": r19 / "output_manifest.json",
        "comet_metadata": pilot.production_inputs(out).metadata,
        "old_query_filter": out / "st2p2/kidrare/features_in/_filter_summary.csv",
        "new_query_filter": out / pilot.DEST / "features_in/_filter_summary.csv",
        "new_filter_receipt": out / pilot.DEST / "filter_receipt.json",
    }
    if isinstance(bindings, BoundInputs):
        require("pinned_correction_comet_metadata" in bindings and
                paths["comet_metadata"].resolve() ==
                bindings["pinned_correction_comet_metadata"].resolve(),
                "Expanded reference labels use a different COMET metadata path")
    bindings.update(paths)
    require(digest(paths["ccdi_provenance"]) == spec.ccdi_sha and
            read(paths["ccdi_complete"])["provenance_sha256"] == spec.ccdi_sha and
            digest(paths["correction_provenance"]) == spec.correction_sha and
            digest(paths["correction_manifest"]) == spec.correction_manifest_sha and
            digest(paths["old_query_filter"]) == spec.old_query_summary_sha,
            "Accepted correction provenance/output/query summary changed")
    index = read(paths["correction_manifest"])
    for filename, receipt in index["artifacts"].items():
        path = r19 / filename
        require(digest(path) == receipt["sha256"] and
                path.stat().st_size == receipt["bytes"],
                f"Accepted correction output changed: {filename}")
        bindings[f"correction_artifact_{filename}"] = path
    for filename in ("paired_differences.json",
                     "allprep_clear_preparation_descriptive.csv"):
        bindings[f"ccdi_{filename}"] = r18 / filename
    for key in ("stv", "kid"):
        require(digest(paths[key]) ==
                index["artifacts"][paths[key].name]["sha256"],
                f"Accepted {key} eligibility source changed")
    stv, kid, ccdi_main, ccdi_prep = (csv(paths[k]) for k in
                                      ("stv", "kid", "ccdi_main", "ccdi_prep"))
    for name, d in (("stv", stv), ("kid", kid),
                    ("ccdi_main", ccdi_main), ("ccdi_prep", ccdi_prep)):
        unique(d, "slide_id", name)
    stv_main = stv[stv.main_eligible.eq("True")].copy()
    stv_both = stv[stv.both_eligible.eq("True")].copy()
    kid_main = kid[kid.amended_eligible.eq("True")].copy()
    kid_both = kid_main[kid_main.corrected_tumor_group.ne("NORMAL")].copy()
    require(set(stv_both.slide_id) <= set(stv_main.slide_id) and
            set(stv_main.slide_id) - set(stv_both.slide_id) ==
            set(stv_main.loc[stv_main.tumor_group.eq("NORMAL"), "slide_id"]),
            "STV both-sided target excludes more than its eight NORMAL queries")
    require(len(stv_main) == spec.stv_main and
            stv_main.patient_id.nunique() == spec.stv_patients and
            len(stv_both) == spec.stv_both and
            stv_both.patient_id.nunique() == spec.stv_both_patients and
            len(ccdi_main) == spec.ccdi_main and
            ccdi_main.participant.nunique() == spec.ccdi_main and
            len(ccdi_prep) == spec.ccdi_allprep and
            ccdi_prep.participant.nunique() == spec.ccdi_allprep and
            len(kid_main) == spec.kid_main and len(kid_both) == spec.kid_both and
            len(kid[kid.legacy_eligible.eq("True")]) == spec.kid_legacy and
            kid_main.corrected_tumor_group.value_counts().to_dict() ==
            ({"LIVER": 815, "NB": 303, "WT": 413, "NORMAL": 50}
             if spec == CONTRACT else
             kid_main.corrected_tumor_group.value_counts().to_dict()),
            "Accepted main/conditional populations changed")
    stv_main["tumor_group"] = stv_main.tumor_group
    stv_both["tumor_group"] = stv_both.tumor_group
    for d in (kid_main, kid_both):
        d["tumor_group"] = d.corrected_tumor_group
        d["source_group"] = np.where(
            d.tumor_group.ne("WT"), "NOT_APPLICABLE_NON_WT",
            np.where(d.pathpt_only_new_label.eq("True"),
                     "ADDED_PATHPT", "ORIGINAL_KEEP"))
    prep_counts = ccdi_prep.preparation_status.value_counts().to_dict()
    require(prep_counts == (PREPARATIONS if spec == CONTRACT else
                            prep_counts) and set(prep_counts) == set(PREPARATIONS),
            "Accepted CCDI preparation categories/margins changed")
    changed = set(kid.loc[kid.processed.eq("True") &
                          kid.pathpt_only_new_label.eq("True"), "slide_id"])
    require(len(changed) == spec.new_kid and changed <= set(kid_main.slide_id) and
            not (changed & set(kid.loc[kid.query_only_eligible.eq("True"), "slide_id"])),
            "New corrected-WT additions or old filter flags changed")
    summaries = {}
    for name, count in (("old_query_filter", spec.kid_legacy),
                        ("new_query_filter", spec.new_kid)):
        s = csv(paths[name])
        unique(s, "slide", name)
        require(len(s) == count and {"n_tiles", "kept_tumor_only"} <= set(s.columns),
                f"{name}: filter coverage/columns changed")
        summaries[name] = s.set_index("slide")
        if name == "old_query_filter":
            require(set(s.slide) == set(kid.loc[kid.legacy_eligible.eq("True"), "slide_id"])
                    and set(s.loc[s.kept_tumor_only.astype(int).eq(0), "slide"]) ==
                    {full.ZERO_LIVER}, "Legacy filtered-query zero/membership changed")
            if spec == CONTRACT:
                require((s.n_tiles.astype(int).sum(),
                         s.kept_tumor_only.astype(int).sum()) == (2682742, 2098376)
                        and int(s.kept_tumor_only.astype(int).between(1, 7).sum()) == 6,
                        "Legacy filtered-query tile margins changed")
        else:
            require(set(s.slide) == changed and
                    s.kept_tumor_only.astype(int).gt(0).all() and
                    read(paths["new_filter_receipt"])["filter_sha256"].keys() == set(changed),
                    "New filtered-query coverage/receipt changed")
            if spec == CONTRACT:
                require((s.n_tiles.astype(int).sum(),
                         s.kept_tumor_only.astype(int).sum()) == (1054560, 892620)
                        and int(s.kept_tumor_only.astype(int).between(1, 7).sum()) == 2,
                        "New filtered-query tile margins changed")
    kid_main["source_tiles"] = kid_main.slide_id.map(pd.concat(
        [summaries["old_query_filter"]["n_tiles"],
         summaries["new_query_filter"]["n_tiles"]]).astype(int))
    kid_main["retained_tumor_tiles"] = kid_main.slide_id.map(pd.concat(
        [summaries["old_query_filter"]["kept_tumor_only"],
         summaries["new_query_filter"]["kept_tumor_only"]]).astype(int))
    require(kid_main[["source_tiles", "retained_tumor_tiles"]].notna().all().all(),
            "Missing accepted query feature-row descriptors")
    kid_both = kid_main[kid_main.tumor_group.ne("NORMAL")].copy()
    return {"stv": stv_main, "stv_both": stv_both, "kid": kid_main,
            "kid_both": kid_both, "ccdi": ccdi_main,
            "ccdi_prep": ccdi_prep}, index, paths, changed


def analyses(out: Path, spec: Spec, bindings: dict[str, Path]
             ) -> tuple[list[Analysis], dict, dict[str, pd.DataFrame]]:
    populations, index, paths, additions = descriptor_inputs(out, spec, bindings)
    kid = populations["kid"]
    legacy = kid[kid.legacy_eligible.eq("True")].copy()
    legacy_both = legacy[legacy.tumor_group.ne("NORMAL")].copy()
    targets = {
        ("stv", "main_696"): populations["stv"],
        ("stv", "query_only_696"): populations["stv"],
        ("stv", "both_sided_688"): populations["stv_both"],
        ("kidrare", "amended_1581"): kid,
        ("kidrare", "legacy_1227"): legacy,
        ("kidrare", "query_only_1226"): legacy,
        ("kidrare", "both_sided_1176"): legacy_both,
        ("kidrare", "expanded_query_only_1581"): kid,
        ("kidrare", "expanded_both_sided_1531"): populations["kid_both"],
        ("ccdi", "ffpe_scope_clear"): populations["ccdi"],
        ("ccdi", "allprep_scope_clear"): populations["ccdi_prep"],
    }
    labels = {cohort: dict(zip(frame.slide_id, frame.tumor_group))
              for cohort, frame in (("stv", populations["stv"]),
                                     ("kidrare", kid),
                                     ("ccdi", populations["ccdi_prep"]))}
    r19 = out / correction.DEST
    r18 = out / ccdi.DEST_NAME
    accepted: dict = {}
    result: list[Analysis] = []

    def add(cohort: str, population: str, configuration: str, role: str,
            root: Path, manifest: dict | None = None,
            empty: set[str] | None = None) -> None:
        target = targets[cohort, population]
        absent = empty or set()
        prefix = f"{cohort}_{population}_{configuration}" if cohort != "ccdi" else (
            f"{configuration}_{population}")
        expected = set(target.slide_id) - absent
        q, ranked, report = accepted_ranking(root, prefix, labels[cohort],
                                             expected, bindings, manifest)
        accepted[prefix] = {"report": report, "source": str(root / f"{prefix}.json"),
                            "sha256": digest(root / f"{prefix}.json")}
        result.append(Analysis(cohort, population, configuration, role,
                               target, ranked, q, absent, prefix))

    for cohort, pop in (("stv", "main_696"), ("kidrare", "amended_1581"),
                        ("kidrare", "legacy_1227")):
        for method in ("whole_slide_ot", "median_pooling", "sampler", "yottixel"):
            add(cohort, pop, method, "main" if pop != "legacy_1227" else
                "historical_sensitivity", r19, index)
    for cohort, arms in (("stv", (("query_only_696", "query_only"),
                                 ("both_sided_688", "both_sided"))),
                         ("kidrare", (("query_only_1226", "query_only"),
                                      ("both_sided_1176", "both_sided")))):
        for pop, method in arms:
            absent = {full.ZERO_LIVER} if cohort == "kidrare" else set()
            if cohort == "stv":
                add(cohort, pop, "whole_slide_ot", "matched_baseline", r19, index)
            else:
                prefix = f"{cohort}_{pop}_whole_slide_ot"
                q, ranked, report = accepted_ranking(
                    r19, prefix, labels[cohort],
                    set(targets[cohort, pop].slide_id) - absent, bindings, index)
                accepted[prefix] = {
                    "report": report, "source": str(r19 / f"{prefix}.json"),
                    "sha256": digest(r19 / f"{prefix}.json")}
                baseline_ranked = next(a.ranked for a in result if a.cohort == cohort
                                       and a.population == "legacy_1227" and
                                       a.configuration == "whole_slide_ot")
                original = baseline_ranked.set_index(["query", "rank"]).loc[
                    pd.MultiIndex.from_frame(ranked[["query", "rank"]])].reset_index()
                require((original[["query", "rank", "neighbour",
                                   "neighbour_label"]].to_numpy() ==
                         ranked[["query", "rank", "neighbour",
                                  "neighbour_label"]].to_numpy()).all() and
                        np.allclose(original.distance, ranked.distance,
                                    atol=1e-10, rtol=0),
                        f"{prefix}: saved matched baseline differs from main ranks")
            add(cohort, pop, method, "historical_filter" if cohort == "kidrare"
                else "current_filter", r19, index, absent)
    for pop in ("ffpe_scope_clear", "allprep_scope_clear"):
        for method in ccdi.CONFIGS:
            add("ccdi", pop, method, "main" if pop == "ffpe_scope_clear"
                else "preparation_sensitivity", r18)
    ccdi_provenance = read(out / ccdi.DEST_NAME / "provenance.json")
    for pop, n in (("ffpe_add_site_only", 536),
                   ("ffpe_add_all_unresolved", 542)):
        manifest_path = r18 / f"manifest_{pop}.csv"
        bindings[f"ccdi_manifest_{pop}"] = manifest_path
        sensitivity = csv(manifest_path)
        unique(sensitivity, "slide_id", f"CCDI {pop}")
        unique(sensitivity, "participant", f"CCDI {pop} singleton")
        require(len(sensitivity) == n and
                sensitivity.tumor_group.value_counts().to_dict() ==
                ccdi_provenance["populations"][pop]["observed_groups"],
                f"{pop}: sensitivity scope/group membership changed")
        sensitivity_truth = dict(zip(sensitivity.slide_id, sensitivity.tumor_group))
        for method in ccdi.CONFIGS:
            prefix = f"{method}_{pop}"
            path = r18 / f"{prefix}.json"
            _, _, report = accepted_ranking(
                r18, prefix, sensitivity_truth, set(sensitivity.slide_id), bindings)
            accepted[prefix] = {"report": report, "source": str(path),
                                "sha256": digest(path)}
    replay_ccdi(out, accepted, bindings, spec)
    # Preserve the published 735-query context, without appending it to the
    # overlapping 524-query main or changing any accepted confidence interval.
    m = read(out / full.DEST / "manifest.json")
    metadata = csv(paths["comet_metadata"])
    require(set(m["ids"]) == additions and set(m["historic"]["query_only"]) ==
            set(legacy.slide_id) - {full.ZERO_LIVER} and
            set(m["historic"]["both_sided"]) ==
            set(legacy.slide_id) - {full.ZERO_LIVER},
            "Full raw/legacy query membership disagrees with corrected truth")
    for arm, pop in (("query_only", "expanded_query_only_1581"),
                     ("both_sided", "expanded_both_sided_1531")):
        target = targets["kidrare", pop]
        q, ranked = expanded_ranking(out, arm, target, {full.ZERO_LIVER},
                                     metadata, m, bindings)
        require(len(q) == spec.kid_returned -
                (0 if arm == "query_only" else 50),
                f"{arm}: expanded return count changed")
        ref_labels = metadata.set_index("Slide.Scan.File")["Tumor.group"]
        library_classes = Counter(ref_labels.loc[m["eligible"][arm]].tolist())
        ordered = ranked.set_index(["query", "rank"]).loc[
            pd.MultiIndex.from_product([q["query"], range(1, 11)])]
        ranked_labels = ordered.neighbour_label.to_numpy().reshape(len(q), 10)
        truth = q.true_label.to_numpy()
        require(set(truth) <= set(library_classes),
                f"{arm}: expanded reference library lacks a truth class")
        point = metrics.compute_all(
            ranked_labels, truth, (1, 3, 5, 10),
            n_relevant=np.array([library_classes[label] for label in truth]))
        require(len(point) == 44 and
                np.isclose(point["micro_hit_at_5"], q.hit_k5.mean(), atol=1e-12),
                f"{arm}: secondary point/endpoints disagree")
        accepted[f"expanded_{arm}"] = {
            "point_estimates": point, "population": pop,
            "reference_library_eligible_counts": dict(library_classes),
            "conditional_returned_only": True,
            "source": str(bindings[f"expanded_{arm}_matrix"]),
            "sha256": digest(bindings[f"expanded_{arm}_matrix"])}
        result.append(Analysis("kidrare", pop, arm, "expanded_filter",
                               target, ranked, q, {full.ZERO_LIVER},
                               f"full_{arm}_matrix"))
    return result, accepted, populations


def baseline_for(a: Analysis, all_analyses: list[Analysis],
                 inventories: dict) -> pd.DataFrame | None:
    if a.configuration == "whole_slide_ot":
        return None
    candidates = [x for x in all_analyses if x.cohort == a.cohort and
                  x.population == a.population and
                  x.configuration == "whole_slide_ot" and
                  not x.empty]
    if candidates:
        return inventories[id(candidates[0])]
    main = next(x for x in all_analyses if x.cohort == a.cohort and
                x.configuration == "whole_slide_ot" and x.role in (
                    ("historical_sensitivity",) if a.population.startswith(
                        ("query_only_1226", "both_sided_1176")) else ("main",)))
    base = inventories[id(main)].set_index("query")
    require(set(a.target.slide_id) <= set(base.index),
            f"{a.population}: main all-tissue baseline cannot cover target")
    matched = base.loc[a.target.slide_id].reset_index()
    require(matched.tumor_group.tolist() == a.target.tumor_group.tolist(),
            f"{a.population}: all-tissue baseline truth differs")
    return matched


def descriptive(a: Analysis, inv: pd.DataFrame, additions: set[str],
                spec: Spec) -> tuple[list[dict], list[dict]]:
    if a.cohort == "kidrare":
        if a.population not in ("amended_1581", "expanded_query_only_1581",
                                "expanded_both_sided_1531"):
            return [], []
        wt = inv[inv.tumor_group.eq("WT")]
        source_groups = [("source_group", value, wt[wt["query"].isin(additions)
                          if value == "ADDED_PATHPT" else ~wt["query"].isin(additions)])
                         for value in ("ORIGINAL_KEEP", "ADDED_PATHPT")]
        pattern_groups = [("wt_pathpt_pattern", pattern,
                           wt[wt.pathpt_subtype.eq(pattern)]) for pattern in PATTERNS]
        groups = source_groups + pattern_groups
        context = []
        for pattern, (old, new) in PATTERNS.items():
            for label, expected in (("ORIGINAL_KEEP", old), ("ADDED_PATHPT", new)):
                n = int((wt.pathpt_subtype.eq(pattern) &
                         wt["query"].isin(additions).eq(label == "ADDED_PATHPT")).sum())
                if spec == CONTRACT:
                    require(n == expected, f"{pattern}/{label}: WT source margin changed")
                context.append({"population": a.population,
                                "configuration": a.configuration,
                                "pathpt_pattern": pattern, "source_group": label,
                                "n_queries": n})
        require(sum(len(group) for _, _, group in pattern_groups) == len(wt),
                "Unrecognized/missing WT PathPT query pattern")
    elif a.cohort == "ccdi" and a.population == "allprep_scope_clear":
        groups = [("preparation_status", name,
                   inv[inv.preparation_status.eq(name)]) for name in PREPARATIONS]
        context = []
        require(sum(len(group) for _, _, group in groups) == len(inv),
                "Unrecognized CCDI preparation status")
    else:
        return [], []
    rows = []
    for key, value, subset in groups:
        returned = subset[subset.status.eq("returned")]
        n = len(subset)
        require(n > 0, f"Empty descriptive stratum: {key}/{value}")
        for endpoint in ("coverage", "hit_yield", "conditional_hit_at_5",
                         "conditional_precision_at_5", "conditional_closest_correct",
                         "conditional_generous_correct"):
            source = subset if endpoint in ("coverage", "hit_yield") else returned
            values = (
                source.status.eq("returned").astype(float) if endpoint == "coverage" else
                source.hit_at_5.fillna(0).astype(float) if endpoint == "hit_yield" else
                source[endpoint.removeprefix("conditional_")].astype(float))
            rows.append({"cohort": a.cohort, "population": a.population,
                         "configuration": a.configuration, "stratum_type": key,
                         "stratum": value, "endpoint": endpoint,
                         "n_target": n, "n_returned": len(returned),
                         "point": float(values.mean()) if len(values) else None,
                         "interval": None, "test": None,
                         "interpretation": ("broad-WT retrieval, not subtype accuracy"
                                            if key == "wt_pathpt_pattern" else
                                            "unadjusted descriptive point")})
    return rows, context


def production_identities() -> tuple[set[tuple[str, str, str]], set[str], set[str]]:
    identities = {
        (cohort, pop, method)
        for cohort, pop in (("stv", "main_696"), ("kidrare", "amended_1581"),
                            ("kidrare", "legacy_1227"))
        for method in ("whole_slide_ot", "median_pooling", "sampler", "yottixel")
    }
    sources = {f"{cohort}_{pop}_{method}" for cohort, pop, method in identities}
    for cohort, arms in (("stv", (("query_only_696", "query_only"),
                                 ("both_sided_688", "both_sided"))),
                         ("kidrare", (("query_only_1226", "query_only"),
                                      ("both_sided_1176", "both_sided")))):
        for pop, arm in arms:
            identities.add((cohort, pop, arm))
            if cohort == "stv":
                identities.add((cohort, pop, "whole_slide_ot"))
            sources.update((f"{cohort}_{pop}_whole_slide_ot",
                            f"{cohort}_{pop}_{arm}"))
    for pop in ("ffpe_scope_clear", "allprep_scope_clear"):
        for method in ccdi.CONFIGS:
            identities.add(("ccdi", pop, method))
    for pop in ("ffpe_scope_clear", "allprep_scope_clear",
                "ffpe_add_site_only", "ffpe_add_all_unresolved"):
        sources.update(f"{method}_{pop}" for method in ccdi.CONFIGS)
    identities.update({("kidrare", "expanded_query_only_1581", "query_only"),
                       ("kidrare", "expanded_both_sided_1531", "both_sided")})
    contrasts = {
        f"{cohort}:{pop}:{method}:vs:whole_slide_ot"
        for cohort, pop, method in identities if method != "whole_slide_ot"}
    require(len(identities) == 32 and len(sources) == 44 and
            len(contrasts) == 25,
            "Production configuration/source contract changed")
    return identities, sources, contrasts


def verify_output_snapshot(destination: Path, artifacts: dict,
                           provenance_sha: str, manifest_sha: str) -> None:
    expected = set(artifacts) | {"output_manifest.json"}
    require({entry.name for entry in destination.iterdir()} == expected and
            digest(destination / "output_manifest.json") == manifest_sha and
            read(destination / "output_manifest.json") ==
            {"schema_version": 1, "artifacts": artifacts} and
            digest(destination / OUTPUTS["provenance"]) == provenance_sha and
            all(digest(destination / filename) == record["sha256"] and
                (destination / filename).stat().st_size == record["bytes"]
                for filename, record in artifacts.items()),
            "Synthesized output changed during final gate/finalization")


def output_frame(rows: list[dict], path: Path) -> None:
    require(bool(rows), f"No rows to write: {path.name}")
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def synthesize(out: Path, spec: Spec = CONTRACT,
               gate=full_gate, loader=analyses) -> Path:
    destination = out / DEST
    correction.refuse_nonempty(destination)
    gate_files = gate(out, spec)
    bindings = BoundInputs()
    bindings.update({f"full_{key}": path for key, path in gate_files.items()})
    bindings["synthesis_driver"] = Path(__file__).resolve()
    bindings["synthesis_tests"] = Path(__file__).with_name(
        "test_r1_3_external_synthesis.py")
    if loader is analyses:
        frozen_dependencies(out, bindings, spec)
    all_analyses, accepted, populations = loader(out, spec, bindings)
    identities = {(a.cohort, a.population, a.configuration) for a in all_analyses}
    require(len(identities) == len(all_analyses),
            "Duplicate configuration/population identity")
    if loader is analyses:
        required, sources, expected_contrasts = production_identities()
        require(identities == required and set(accepted) == sources | {
            "expanded_query_only", "expanded_both_sided"},
            "Production analysis/accepted-source inventory changed")
    for a in all_analyses:
        require(not a.target.empty and
                set(a.query["query"]) == set(a.target.slide_id) - a.empty and
                all(dict(zip(a.target.slide_id, a.target.tumor_group))[sid] == truth
                    for sid, truth in zip(a.query["query"], a.query.true_label)),
                "Analysis target/query/truth membership changed")
    # Fail before creating an output when a bound source is missing or altered.
    initial = bindings.snapshot()
    inventories = {
        id(a): inventory(a, populations["stv"] if a.cohort == "stv" else
                         populations["kid"] if a.cohort == "kidrare" else
                         populations["ccdi_prep"]) for a in all_analyses
    }
    cache: dict = {}
    tables: dict[str, list[dict]] = {name: [] for name in OUTPUTS if name not in
                                     ("accepted", "provenance")}
    for arm in ("query_only", "both_sided"):
        report = accepted.get(f"expanded_{arm}")
        if report is not None:
            tables["secondary"].extend(
                {"cohort": "kidrare", "population": report["population"],
                 "configuration": arm, "metric": metric, "point": point,
                 "n_queries": report["point_estimates"]["n_queries"],
                 "retrieval_depth": report["point_estimates"]["retrieval_depth"],
                 "estimand": "conditional_returned",
                 "interval_status": "secondary_point_only_no_new_CI"}
                for metric, point in report["point_estimates"].items()
                if metric not in ("n_queries", "retrieval_depth"))
    for a in all_analyses:
        inv = inventories[id(a)]
        base = baseline_for(a, all_analyses, inventories)
        if base is not None:
            inv.loc[inv.status.eq("returned"), "paired"] = True
        tables["inventory"].extend(inv.to_dict("records"))
        primary, coverage = summarize(a, inv, spec, cache, base)
        tables["broad"].extend(primary)
        tables["coverage"].extend(coverage)
        tables["paired"].extend(row for row in primary + coverage
                                if "difference" in row)
        if a.role in ("main", "current_filter", "expanded_filter"):
            views = failure_views(a, inv)
            for name, rows in views.items():
                tables[name].extend(rows)
        if base is not None:
            changed, losses = transitions(a, inv, base)
            tables["transitions"].extend(changed)
            tables["losses"].extend(losses)
        points, pattern = descriptive(a, inv, set(
            populations["kid"].loc[populations["kid"].pathpt_only_new_label.eq("True"),
                                   "slide_id"]), spec)
        tables["strata"].extend(points)
        tables["pattern_context"].extend(pattern)
        for metric in ("closest_correct", "generous_correct"):
            returned = inv[inv.status.eq("returned")]
            row = {"cohort": a.cohort, "population": a.population,
                   "configuration": a.configuration, "metric": metric,
                   "tumor_group": "micro",
                   "n_returned": len(returned),
                   "conditional_point": float(returned[metric].astype(float).mean())}
            if base is not None:
                before = base.set_index("query").loc[returned["query"], metric]
                row["paired_baseline_point"] = float(before.astype(float).mean())
                row["paired_difference"] = (row["conditional_point"] -
                                             row["paired_baseline_point"])
                row["n_fixed"] = sum(r["transition"] == "fixed" for r in changed
                                     if r["endpoint"] == metric)
                row["n_broken"] = sum(r["transition"] == "broken" for r in changed
                                      if r["endpoint"] == metric)
            if a.role == "expanded_filter" and metric == "generous_correct":
                pop = a.target.set_index("slide_id").loc[returned["query"]].reset_index()
                draws, desc = shared_weights(cache, pop, a.cohort, spec)
                classes = sorted(pop.tumor_group.unique())
                value = returned[metric].to_numpy(dtype=float)
                comparison = base.set_index("query").loc[returned["query"], metric]
                comparison = comparison.to_numpy(dtype=float)
                current = ratio(value, pop, draws, classes)
                prior = ratio(comparison, pop, draws, classes)
                for group in ("micro", "macro"):
                    point, series, _ = current[group]
                    delta = series - prior[group][1]
                    entry = {**row, "tumor_group": group,
                             "conditional_point": point,
                             "paired_baseline_point": prior[group][0],
                             "paired_difference": point - prior[group][0],
                             "ci_lo": float(np.percentile(series, 2.5)),
                             "ci_hi": float(np.percentile(series, 97.5)),
                             "paired_difference_ci_lo": float(np.percentile(delta, 2.5)),
                             "paired_difference_ci_hi": float(np.percentile(delta, 97.5)),
                             "weights_sha256": desc["weight_sha256"],
                             "B_effective": len(series),
                             "comparison_id": (
                                 f"{a.cohort}:{a.population}:{a.configuration}"
                                 ":vs:whole_slide_ot")}
                    tables["aggregate"].append(entry)
                continue
            tables["aggregate"].append(row)
    require(set(tables["inventory"][i]["query"] for i in range(
        len(tables["inventory"]))) ==
            set(q for a in all_analyses for q in a.target.slide_id),
            "Inventory query coverage changed")
    if loader is analyses:
        require({row["comparison_id"] for row in tables["paired"]} ==
                expected_contrasts,
                "Production paired-contrast inventory changed")
    correction.refuse_nonempty(destination)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    def record_output(filename: str) -> None:
        path = destination / filename
        artifacts[filename] = {"sha256": digest(path),
                               "bytes": path.stat().st_size}

    for name, filename in OUTPUTS.items():
        if name not in ("accepted", "provenance"):
            if tables[name]:
                output_frame(tables[name], destination / filename)
            else:
                require(name in EMPTY_COLUMNS, f"Missing required output: {name}")
                pd.DataFrame(columns=EMPTY_COLUMNS[name]).to_csv(
                    destination / filename, index=False)
            record_output(filename)
    accepted["source_summary"] = {
        role: {"source": str(bindings[key]), "sha256": initial[key]}
        for role, key in (
            ("correction_primary_and_secondary",
             "correction_artifact_points_all_44_and_primary_intervals.csv"),
            ("correction_paired", "correction_artifact_paired_differences.json"),
            ("ccdi_paired", "ccdi_paired_differences.json"))
    }
    write_json(destination / OUTPUTS["accepted"], accepted)
    record_output(OUTPUTS["accepted"])
    require(bindings.snapshot() == initial,
            "Input changed during synthesis; partial output retained")
    gate(out, spec)
    require(bindings.snapshot() == initial,
            "Input changed during intermediate gate")
    provenance = {
        "schema_version": 1, "scientific_scope": "ADR0020; fixed-library external-query",
        "execution": {
            "job_id": os.environ.get("LSB_JOBID"),
            "queue": os.environ.get("LSB_QUEUE"),
            "hosts": os.environ.get("LSB_HOSTS"),
            "host": socket.gethostname(), "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": {key: os.environ.get(key)
                                   for key in ("OPENBLAS_NUM_THREADS",
                                               "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                               "CUDA_VISIBLE_DEVICES")}},
        "inputs": {name: {"path": str(path.resolve()), "sha256": initial[name]}
                   for name, path in sorted(bindings.items())},
        "populations": [{
            "cohort": a.cohort, "population": a.population,
            "configuration": a.configuration, "analysis_role": a.role,
            "n_target": len(a.target), "n_returned": len(a.query),
            "n_confirmed_empty": len(a.empty),
            "n_technical_failure": 0, "source": a.source,
            "target_ids_sha256": hashlib.sha256(
                "".join(sid + "\n" for sid in a.target.slide_id).encode()).hexdigest()
        } for a in all_analyses],
        "draws": [{**details[1], "cohort": key[0], "estimand": key[1],
                   "n_query": len(key[2]),
                   "ordered_units_sha256": hashlib.sha256(
                       json.dumps(key[3]).encode()).hexdigest()}
                  for key, details in cache.items()],
        "interpretation": [
            "STV repeated ST_ identifiers are supplied patient clusters; KidRare uses nominal slide units.",
            "CCDI reuses ADR0018 singleton-participant draws in accepted population order.",
            "Fixed broad classes; percentile95 nominal B/seed intervals, no stratum-specific inference.",
            "Confirmed empty has zero coverage/hit yield; conditional rank, precision and votes unavailable.",
            "Historical KidRare filter rows remain historical and are never substituted with replay controls.",
            "Accepted main/sensitivity estimates and confidence intervals are preserved verbatim by hash-bound report reference.",
            "Differences are paired replicate-wise on declared identical populations and shared weights.",
            "No cross-library macro-F1 inference, inferred patient identity, or pathological-cause attribution.",
        ],
        "output_index": OUTPUTS,
    }
    write_json(destination / OUTPUTS["provenance"], provenance)
    record_output(OUTPUTS["provenance"])
    require(set(artifacts) == set(OUTPUTS.values()),
            "Synthesis output inventory incomplete")
    provenance_sha = artifacts[OUTPUTS["provenance"]]["sha256"]
    write_json(destination / "output_manifest.json",
               {"schema_version": 1, "artifacts": artifacts})
    manifest_sha = digest(destination / "output_manifest.json")
    require(bindings.snapshot() == initial,
            "Input changed during finalization; partial output retained")
    verify_output_snapshot(destination, artifacts, provenance_sha, manifest_sha)
    gate(out, spec)
    require(bindings.snapshot() == initial,
            "Input changed during final gate; partial output retained")
    verify_output_snapshot(destination, artifacts, provenance_sha, manifest_sha)
    require(read(destination / OUTPUTS["provenance"])["inputs"] ==
            {name: {"path": str(path.resolve()), "sha256": initial[name]}
             for name, path in sorted(bindings.items())},
            "Synthesized provenance input chain changed")
    write_json(destination / "COMPLETE.json",
               {"status": "technical_synthesis_for_independent_scientific_review",
                "n_configurations": len(all_analyses),
                "provenance_sha256": provenance_sha,
                "output_manifest_sha256": manifest_sha,
                "not_scientific_gate_c_approval": True})
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    synthesize(correction.out_root())


if __name__ == "__main__":
    main()
