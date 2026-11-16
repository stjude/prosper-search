#!/usr/bin/env python3
"""[decision record] full KidRare filtered-query OT extension; no scheduler submissions.

Stages: preflight (global source audit and frozen plan), run-task (one scalar
one/two-query OT job), merge (strict union, stock combine, global final audit).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import r1_3_kidrare_filter_extension as pilot

DEST = "r1_3_kidrare_filter_full_20260930"
ZERO_LIVER = "NULL-20250103-133602"
ARMS = pilot.ARMS
MODELS = {
    "query_only": "r1_3_kidrare_expanded_query_only_20260930",
    "both_sided": "r1_3_kidrare_expanded_both_sided_20260930",
}


@dataclass(frozen=True)
class Spec:
    n_old: int = 1226
    n_new: int = 354
    n_pilot: int = 3
    n_source: int = 2331
    n_legacy: int = 1227
    n_normal: int = 50
    membership_sha: str = "4af6f341cd25df74df7baef1662b28d121f254dda3133ab34d5bebd59b08140d"
    marker_sha: str = "60b7db5e7e60c999f40245ceeb4cd8ccae95e21a8e4bf80ceb1561fb11a93d46"
    review_sha: str = "130a04590357e917c8502213ac9c73997f8ac9b7da90e0d0267e0ca7b9f0decf"
    pilot_driver_sha: str = "43000a73767746845b10dba12881af6bc15a893dda2b0928b0e6fc92c6e817f8"
    pilot_tests_sha: str = "9b8584bb2e104388090e44137ae1e2180f98c6255c3f5f04c5a911936ca57a62"
    provenance_sha: str = "f7f5bdf344c2011a3dda46457cc35b4620d35dbc5657db518ecd047ff61c6311"
    matrix_all_sha: str = "660f58e5c97d4c6cb52390c6de2d87d409a282b295172b51c484d6a5b2bc3886"
    matrix_both_sha: str = "8ef6081f3a5c3b565b1dc470c5ec6b035768bf9d9ccb52b3e014b6c8b06dfb6c"
    rank_all_sha: str = "b463bb3440b31c732f322bad73b0f689df9fc95b1469a27636682b980a58f77d"
    rank_both_sha: str = "c062dbb973287d15b91ff308209a511406004239b0bcb706e065fad7399dc6f7"
    reviewer: str = "3524c027-feb9-412e-83ed-b8967b6183e9"


CONTRACT = Spec()


@dataclass(frozen=True)
class Historical:
    matrix: Path
    cache: Path
    ranked: Path


@dataclass(frozen=True)
class Inputs:
    pilot_inputs: pilot.Inputs
    provenance: Path
    historical: dict[str, Historical]
    corrected_membership: Path


def production_inputs(out: Path) -> Inputs:
    p = pilot.production_inputs(out)
    correction = out / "r1_3_external_correction_20260930"
    return Inputs(p, correction / "provenance.json", {
        "query_only": Historical(
            out / "similarity_matrix/combined_similarity_matrix_st2p2_kidrare_tumor_only_ot_cosine.pkl",
            out / "similarity_st2p2_kidrare_tumor_only_ot",
            p.old_query_rank),
        "both_sided": Historical(
            out / "similarity_matrix/combined_similarity_matrix_st2p2_kidrare_tumor_only_both_ot_cosine.pkl",
            out / "similarity_st2p2_kidrare_tumor_only_both_ot",
            p.old_both_rank),
    }, correction / "kidrare_source_2331_eligibility.csv")


def ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("".join(f"{sid}\n" for sid in ids).encode()).hexdigest()


def bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.array(a, dtype=np.float64, order="C")
    b = np.array(b, dtype=np.float64, order="C")
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def bound(path: Path) -> dict:
    pilot.require(path.is_file(), f"Missing fixed input: {path}")
    return {"path": str(path.resolve()), "sha256": pilot.digest(path)}


def check_bound(binding: dict) -> None:
    path = Path(binding["path"])
    pilot.require(path.is_file() and str(path.resolve()) == binding["path"] and
                  pilot.digest(path) == binding["sha256"],
                  f"Bound input changed: {path}")


def matrix(path: Path, queries: list[str], refs: list[str]) -> pd.DataFrame:
    with path.open("rb") as stream:
        frame = pickle.load(stream)
    pilot.require(isinstance(frame, pd.DataFrame) and
                  frame.shape == (len(queries), len(refs)) and
                  list(frame.index) == queries and list(frame.columns) == refs and
                  all(dtype == np.dtype("float64") for dtype in frame.dtypes),
                  f"Wrong matrix shape/order/dtype: {path}")
    values = frame.to_numpy()
    pilot.require(np.isfinite(values).all() and (values >= 0).all() and
                  (values < 1e6).all(), f"Invalid matrix cells: {path}")
    return frame


def ranked_bridge(path: Path, queries: list[str], refs: list[str],
                  eligible: set[str], labels: dict[str, str],
                  cached: dict[str, np.ndarray]) -> None:
    saved = pd.read_csv(path, dtype={"query": str, "neighbour": str,
                                     "neighbour_label": str}, keep_default_na=False)
    pilot.require(list(saved.columns) ==
                  ["query", "rank", "neighbour", "neighbour_label", "distance"] and
                  len(saved) == len(queries) * 10 and
                  set(saved["query"]) == set(queries) and
                  not saved.duplicated(["query", "rank"]).any(),
                  f"Historical ranking inventory/schema changed: {path}")
    for qid, rows in saved.groupby("query", sort=False):
        rows = rows.sort_values("rank")
        neighbours = pilot.top10(refs, cached[qid], eligible, qid)
        distances = dict(zip(refs, cached[qid]))
        expected = np.array([distances[sid] for sid in neighbours])
        pilot.require(rows["rank"].tolist() == list(range(1, 11)) and
                      rows["neighbour"].tolist() == neighbours and
                      rows["neighbour_label"].tolist() ==
                      [labels[sid] for sid in neighbours] and
                      np.isfinite(rows["distance"].to_numpy(dtype=float)).all() and
                      (np.abs(rows["distance"].to_numpy(dtype=float) - expected)
                       <= 1e-10).all(),
                      f"{qid}: historical ranked IDs/labels/distances diverge")


def corrected_old_ids(path: Path, new_ids: list[str], spec: Spec) -> list[str]:
    pilot.require(path.is_file() and pilot.digest(path) == spec.membership_sha,
                  "Accepted ADR0019 KidRare eligibility source hash changed")
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = {"slide_id", "processed", "tumor_group", "original_tumor_group",
               "corrected_tumor_group", "pathpt_only_new_label",
               "legacy_eligible", "amended_eligible", "query_only_eligible",
               "both_sided_eligible"}
    flags = ("processed", "pathpt_only_new_label", "legacy_eligible",
             "amended_eligible", "query_only_eligible", "both_sided_eligible")
    pilot.require(columns <= set(d.columns) and len(d) == spec.n_source and
                  d.slide_id.is_unique and
                  all(d[key].isin({"True", "False"}).all() for key in flags),
                  "Corrected KidRare eligibility source schema/flags invalid")
    source = d.set_index("slide_id")
    old = set(d.loc[d.query_only_eligible.eq("True"), "slide_id"])
    legacy = set(d.loc[d.legacy_eligible.eq("True"), "slide_id"])
    both = set(d.loc[d.both_sided_eligible.eq("True"), "slide_id"])
    normal = {sid for sid in old if source.at[sid, "tumor_group"] == "NORMAL"}
    pilot.require(ZERO_LIVER in source.index and len(old) == spec.n_old and
                  len(legacy) == spec.n_legacy and
                  legacy - old == {ZERO_LIVER} and
                  old <= legacy and len(normal) == spec.n_normal and
                  both == old - normal and len(both) == spec.n_old - spec.n_normal and
                  source.at[ZERO_LIVER, "tumor_group"] == "LIVER" and
                  source.at[ZERO_LIVER, "processed"] == "True" and
                  source.at[ZERO_LIVER, "query_only_eligible"] == "False" and
                  source.at[ZERO_LIVER, "both_sided_eligible"] == "False" and
                  all(source.at[sid, "processed"] == "True" and
                      source.at[sid, "pathpt_only_new_label"] == "True" and
                      source.at[sid, "legacy_eligible"] == "False" and
                      source.at[sid, "amended_eligible"] == "True" and
                      source.at[sid, "query_only_eligible"] == "False" and
                      source.at[sid, "both_sided_eligible"] == "False" and
                      source.at[sid, "original_tumor_group"] == "WT_UNVERIFIED" and
                      source.at[sid, "corrected_tumor_group"] == "WT"
                      for sid in new_ids if sid in source.index) and
                  set(new_ids) <= set(source.index) and
                  not set(new_ids) & legacy,
                  "Historical/new query membership or zero-tile LIVER exclusion changed")
    return sorted(old)


def partition(new_ids: list[str], reused: list[str]) -> list[dict]:
    pilot.require(len(set(new_ids)) == len(new_ids) and new_ids == sorted(new_ids)
                  and len(reused) == 3 and len(set(reused)) == 3 and
                  set(reused) <= set(new_ids),
                  "New/reused query partition invalid")
    remaining = sorted(set(new_ids) - set(reused))
    tasks = []
    for arm in ARMS:
        for i, start in enumerate(range(0, len(remaining), 2), 1):
            qids = remaining[start:start + 2]
            task_id = f"{arm}_{i:03d}"
            task = f"{DEST}/ot/{task_id}"
            tasks.append({"id": task_id, "arm": arm, "queries": qids, "task": task})
    pilot.require(len(tasks) == 2 * ((len(new_ids) - 3 + 1) // 2) and
                  len({t["task"].replace("/", "_") for t in tasks}) == len(tasks) and
                  len({t["id"] for t in tasks}) == len(tasks),
                  "Task or normalized log namespace collision")
    for arm in ARMS:
        planned = [sid for t in tasks if t["arm"] == arm for sid in t["queries"]]
        pilot.require(planned == remaining and len(planned) == len(set(planned)),
                      f"{arm}: task partition overlap/omission")
    return tasks


class Full:
    def __init__(self, out: Path, inputs: Inputs | None = None,
                 spec: Spec = CONTRACT, pilot_spec: pilot.Spec = pilot.CONTRACT) -> None:
        self.out = out.resolve()
        self.root = self.out / DEST
        self.inputs = inputs or production_inputs(self.out)
        self.spec = spec
        self.pilot = pilot.Pilot(self.out, self.inputs.pilot_inputs, pilot_spec)

    def sources(self) -> dict[str, Path]:
        p = self.pilot.root
        files = {
            "pilot_marker": p / "PILOT_COMPLETE.json",
            "pilot_review": p / "PILOT_INDEPENDENT_REVIEW.json",
            "pilot_preflight": p / "preflight.json",
            "pilot_selection": p / "selection.json",
            "pilot_score_receipt": p / "score_receipt.json",
            "pilot_filter_receipt": p / "filter_receipt.json",
            "pilot_driver": pilot.ROOT / "scripts/r1_3_kidrare_filter_extension.py",
            "pilot_tests": pilot.ROOT / "scripts/test_r1_3_kidrare_filter_extension.py",
            "full_driver": Path(__file__).resolve(),
            "full_tests": pilot.ROOT / "scripts/test_r1_3_kidrare_filter_full.py",
            "correction_provenance": self.inputs.provenance,
            "correction_membership": self.inputs.corrected_membership,
        }
        files.update({f"worker_{name}": path for name, path in
                      pilot.worker_files(self.inputs.pilot_inputs.worker_root).items()})
        for arm in ARMS:
            h = self.inputs.historical[arm]
            files[f"{arm}_matrix"] = h.matrix
            files[f"{arm}_ranked"] = h.ranked
        return files

    def accepted(self) -> tuple[list[str], list[str], dict, dict]:
        p = self.pilot
        marker_path = p.root / "PILOT_COMPLETE.json"
        review_path = p.root / "PILOT_INDEPENDENT_REVIEW.json"
        pilot.require(pilot.digest(marker_path) == self.spec.marker_sha and
                      pilot.digest(review_path) == self.spec.review_sha and
                      pilot.digest(self.sources()["pilot_driver"]) ==
                      self.spec.pilot_driver_sha and
                      pilot.digest(self.sources()["pilot_tests"]) ==
                      self.spec.pilot_tests_sha,
                      "Accepted pilot marker/review/code hash changed")
        marker, review = pilot.json_read(marker_path), pilot.json_read(review_path)
        ids, pre, sel = p.selection_receipt()
        reused = list(sel["selection"].values())
        pilot.require(set(sel["selection"]) == {"median", "p95", "maximum"} and
                      len(ids) == self.spec.n_new and len(reused) == self.spec.n_pilot
                      and len(set(reused)) == 3 and set(reused) <= set(ids) and
                      marker["status"] == "technical_pilot_complete_for_independent_review"
                      and marker["not_scientific_or_author_gate_c_approval"] is True and
                      marker["n_new_scored"] == len(ids) and
                      marker["n_ot_tasks"] == len(sel["tasks"]) == 8 and
                      marker["preflight_sha256"] == pilot.digest(p.root / "preflight.json")
                      and marker["selection_sha256"] == pilot.digest(p.root / "selection.json")
                      and marker["score_receipt_sha256"] == pilot.digest(p.root / "score_receipt.json")
                      and marker["filter_receipt_sha256"] == pilot.digest(p.root / "filter_receipt.json")
                      and review["status"] == "accepted_technical_pilot_for_full_filter_extension"
                      and review["pilot_complete_sha256"] == self.spec.marker_sha and
                      review["preflight_sha256"] == marker["preflight_sha256"] and
                      review["selection_sha256"] == marker["selection_sha256"] and
                      review["score_receipt_sha256"] == marker["score_receipt_sha256"] and
                      review["filter_receipt_sha256"] == marker["filter_receipt_sha256"] and
                      review["pilot_driver_sha256"] == self.spec.pilot_driver_sha and
                      review["pilot_tests_sha256"] == self.spec.pilot_tests_sha and
                      review["reviewer"]["agent_id"] == self.spec.reviewer and
                      review["author_full_extension_condition_satisfied"] is True and
                      review["preserve_original_historical_control_rows"] is True and
                      review["blocking_findings"] == [],
                      "Pilot acceptance, selection or independent review linkage invalid")
        return ids, reused, marker, pre

    def global_audit(self) -> dict:
        ids, reused, marker, pre = self.accepted()
        self.pilot.check_bindings(pre, ids, full=True)
        expected_old = corrected_old_ids(self.inputs.corrected_membership,
                                         ids, self.spec)
        all_ids, both_ids, eligible = self.pilot.reference_ids()
        refs = {"query_only": all_ids, "both_sided": both_ids}
        metadata = pd.read_csv(self.inputs.pilot_inputs.metadata, dtype=str,
                               keep_default_na=False)
        labels = dict(zip(metadata["Slide.Scan.File"], metadata["Tumor.group"]))
        pilot.require(pilot.digest(self.inputs.provenance) == self.spec.provenance_sha,
                      "Accepted ADR0019 provenance hash changed")
        provenance = pilot.json_read(self.inputs.provenance)
        files = self.sources()
        binding = {key: bound(path) for key, path in files.items()}
        historic = {}
        for arm in ARMS:
            h, raw = self.inputs.historical[arm], refs[arm]
            pinned_matrix = (self.spec.matrix_all_sha if arm == "query_only"
                             else self.spec.matrix_both_sha)
            pinned_rank = (self.spec.rank_all_sha if arm == "query_only"
                           else self.spec.rank_both_sha)
            rank_key = ("kidrare_query_only_ranked" if arm == "query_only"
                        else "kidrare_both_sided_ranked")
            pilot.require(binding[f"{arm}_matrix"]["sha256"] == pinned_matrix and
                          binding[f"{arm}_ranked"]["sha256"] == pinned_rank and
                          provenance["inputs"][rank_key]["sha256"] == pinned_rank and
                          Path(provenance["inputs"][rank_key]["path"]).resolve() ==
                          h.ranked.resolve(), f"{arm}: historical source provenance diverges")
            pilot.require(h.cache.is_dir(), f"{arm}: missing historical cache directory")
            names = sorted(h.cache.iterdir())
            pilot.require(all(file.is_file() and
                              re.fullmatch(r"similarity_[A-Za-z0-9_-]+\.pkl",
                                           file.name) for file in names),
                          f"{arm}: foreign historical cache file")
            old_ids = [file.name[len("similarity_"):-len(".pkl")] for file in names]
            pilot.require(old_ids == expected_old and
                          not set(old_ids) & set(ids) and pilot.CONTROL in old_ids,
                          f"{arm}: historical cache differs from corrected eligibility")
            frame = matrix(h.matrix, old_ids, raw)
            cache_rows = {}
            for sid in old_ids:
                file = h.cache / f"similarity_{sid}.pkl"
                values = pilot.row(file, sid, raw)
                pilot.require(bits_equal(values, frame.loc[sid, raw].to_numpy()),
                              f"{arm}/{sid}: historical cache/matrix bits differ")
                cache_rows[sid] = values
                binding[f"{arm}_cache_{sid}"] = bound(file)
            ranked_bridge(h.ranked, old_ids, raw, eligible[arm], labels, cache_rows)
            historic[arm] = old_ids
        pilot.require(historic["query_only"] == historic["both_sided"],
                      "Historical arm query sets differ")
        for entry in marker["tasks"]:
            arm, sid, task = (entry[key] for key in ("arm", "query", "task"))
            pilot.require({"arm": arm, "query": sid, "task": task} in
                          pilot.json_read(self.pilot.root / "selection.json")["tasks"],
                          "Accepted pilot task not in frozen selection")
            receipt_path = self.pilot.root / f"receipt_{arm}_{sid}.json"
            receipt = pilot.json_read(receipt_path)
            row_path = self.out / task / f"similarity_{sid}.pkl"
            sidecar = self.out / "logs" / task.replace("/", "_") / "provenance_1_1.json"
            result = self.pilot.check_row(arm, sid, task, eligible[arm], refs[arm])
            pilot.require(pilot.digest(receipt_path) == entry["receipt_sha256"] and
                          receipt["result"] == entry["result"] == result and
                          receipt["runtime"]["lsf_job"] == result["source_job"] and
                          receipt["arm"] == arm and receipt["query"] == sid and
                          receipt["task"] == task, f"{arm}/{sid}: pilot receipt mismatch")
            for kind, file in (("receipt", receipt_path), ("row", row_path),
                               ("sidecar", sidecar)):
                binding[f"pilot_{kind}_{arm}_{sid}"] = bound(file)
        filter_receipt = pilot.json_read(self.pilot.root / "filter_receipt.json")
        pilot.require(set(filter_receipt["filter_sha256"]) == set(ids),
                      "Accepted filtered feature inventory changed")
        for sid in ids:
            file = self.pilot.root / "features_in/tumor_only" / f"{sid}.pkl"
            binding[f"feature_{sid}"] = bound(file)
            pilot.require(binding[f"feature_{sid}"]["sha256"] ==
                          filter_receipt["filter_sha256"][sid],
                          f"{sid}: filtered feature differs from accepted pilot")
        return {"ids": ids, "reused": reused, "historic": historic,
                "corrected_historical_ids": expected_old, "refs": refs,
                "libraries": {
                    "query_only": str(self.inputs.pilot_inputs.raw_all.resolve()),
                    "both_sided": str(self.inputs.pilot_inputs.raw_both.resolve())},
                "eligible": {a: sorted(eligible[a]) for a in ARMS},
                "bindings": binding}

    def matrix_path(self, arm: str) -> Path:
        return self.out / "similarity_matrix" / (
            f"combined_similarity_matrix_{MODELS[arm]}_ot_cosine.pkl")

    def sidecar_path(self, arm: str) -> Path:
        return self.out / "logs/combine_similarity" / (
            f"provenance_{MODELS[arm]}_ot_cosine.json")

    def fresh_destinations(self, tasks: list[dict]) -> None:
        for arm in ARMS:
            pilot.require(not self.matrix_path(arm).exists() and
                          not self.sidecar_path(arm).exists(),
                          f"{arm}: combine output/sidecar collision")
        for task in tasks:
            path = self.out / task["task"]
            log = self.out / "logs" / task["task"].replace("/", "_")
            pilot.require(not path.exists() and not log.exists(),
                          f"{task['id']}: task output/log collision")

    def preflight(self) -> None:
        pilot.require(not self.root.exists(), f"Refusing existing full root: {self.root}")
        runtime = pilot.runtime_info("score")
        audited = self.global_audit()
        tasks = partition(audited["ids"], audited["reused"])
        pilot.require(len(audited["historic"]["query_only"]) + len(audited["ids"]) ==
                      self.spec.n_old + self.spec.n_new and
                      all(re.fullmatch(r"[A-Za-z0-9_-]+", sid)
                          for sid in audited["ids"] + audited["historic"]["query_only"]),
                      "Union count or unsafe source query ID")
        self.fresh_destinations(tasks)
        manifest = {"schema_version": 1, **audited, "tasks": tasks,
                    "references": {arm: {"sha256": ids_sha(audited["refs"][arm]),
                                         "count": len(audited["refs"][arm])}
                                   for arm in ARMS},
                    "source_payload_policy": (
                        "Reference directory IDs are checked per task; file payloads are "
                        "read-only and rehashed globally at preflight and final merge.")}
        self.root.mkdir(parents=True, exist_ok=False)
        pilot.json_write(self.root / "manifest.json", manifest)
        pilot.json_write(self.root / "preflight_receipt.json", {
            "stage": "preflight", "manifest_sha256": pilot.digest(self.root / "manifest.json"),
            "runtime": runtime})

    def plan(self) -> dict:
        record = pilot.json_read(self.root / "preflight_receipt.json")
        manifest_path = self.root / "manifest.json"
        pilot.require(record["stage"] == "preflight" and
                      record["manifest_sha256"] == pilot.digest(manifest_path),
                      "Frozen full preflight/manifest changed")
        m = pilot.json_read(manifest_path)
        pilot.require(m["schema_version"] == 1 and
                      m["tasks"] == partition(m["ids"], m["reused"]) and
                      len(m["ids"]) == self.spec.n_new and
                      len(m["historic"]["query_only"]) ==
                      len(m["historic"]["both_sided"]) == self.spec.n_old and
                      m["historic"]["query_only"] == m["historic"]["both_sided"] and
                      m["historic"]["query_only"] == m["corrected_historical_ids"] and
                      not set(m["ids"]) & set(m["historic"]["query_only"]) and
                      set(m["libraries"]) == set(ARMS) and
                      all(m["references"][a] == {"sha256": ids_sha(m["refs"][a]),
                                                   "count": len(m["refs"][a])}
                          for a in ARMS),
                      "Full plan partition/historical/reference identities changed")
        return m

    def small_bindings(self, m: dict) -> None:
        small = self.sources()
        for key, path in small.items():
            if key.endswith(("_matrix", "_ranked")):
                continue
            pilot.require(m["bindings"][key]["path"] == str(path.resolve()),
                          f"Full small source path changed: {key}")
            check_bound(m["bindings"][key])
        accepted = pilot.json_read(self.pilot.root / "preflight.json")["input_bindings"]
        for key, path in (("manifest", self.inputs.pilot_inputs.changed_manifest),
                          ("comet_metadata", self.inputs.pilot_inputs.metadata),
                          ("classifier", self.inputs.pilot_inputs.classifier)):
            pilot.require(accepted[key]["path"] == str(path.resolve()),
                          f"Accepted pilot {key} path changed")
            check_bound(accepted[key])
        pilot.verify_worker_head(self.inputs.pilot_inputs.worker_root)
        commit = subprocess.check_output(
            ["git", "-C", str(self.inputs.pilot_inputs.worker_root), "rev-parse", "HEAD"],
            text=True).strip()
        pilot.require(commit == self.pilot.spec.source_commit,
                      "Pinned stock worker commit changed")

    def references(self, m: dict, arm: str) -> list[str]:
        source = self.inputs.pilot_inputs.raw_all if arm == "query_only" else \
            self.inputs.pilot_inputs.raw_both
        pilot.require(str(source.resolve()) == m["libraries"][arm],
                      f"{arm}: library path differs from bound canonical source")
        refs = pilot.ids_in(source)
        pilot.require(refs == m["refs"][arm] and
                      ids_sha(refs) == m["references"][arm]["sha256"] and
                      len(m["eligible"][arm]) == (self.pilot.spec.eligible_all
                                                   if arm == "query_only" else
                                                   self.pilot.spec.eligible_both),
                      f"{arm}: ordered reference membership/eligibility changed")
        return refs

    def task_paths(self, t: dict) -> tuple[Path, Path, Path, Path]:
        output = self.out / t["task"]
        log = self.out / "logs" / t["task"].replace("/", "_")
        subset = self.root / "subsets" / t["id"]
        receipt = self.root / "receipts" / f"{t['id']}.json"
        return output, log, subset, receipt

    def execution_log_paths(self, t: dict) -> tuple[Path, Path]:
        directory = self.root / "execution_logs"
        return directory / f"{t['id']}.stdout.log", directory / f"{t['id']}.stderr.log"

    def execution_log_bindings(self, t: dict) -> dict:
        stdout, stderr = self.execution_log_paths(t)
        return {"stdout": bound(stdout), "stderr": bound(stderr)}

    def failure_diagnostic(self, t: dict, reason: str, returncode: int,
                           logs: dict) -> None:
        destination = self.root / "diagnostics" / f"{t['id']}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pilot.json_write(destination, {"status": reason, "task": t,
                                       "returncode": returncode, "execution_logs": logs})

    def run_ot_worker(self, t: dict, args: list[str]) -> dict:
        worker_root = self.inputs.pilot_inputs.worker_root
        env = os.environ.copy()
        pilot.require(env.get("PROSPER_PARAMS") in (None, "") and
                      env.get("LSB_JOBINDEX") in (None, "", "0"),
                      "Full OT tasks require scalar LSF mode without PROSPER_PARAMS")
        env.update(pilot.THREAD_ENV)
        env.update({"PROSPER_ROOT": str(worker_root.resolve()),
                    "COMET_ROOT": str(pilot.COMET), "PROSPER_OUT": str(self.out),
                    "PROSPER_PYTHON": str(pilot.PYTHON),
                    "PROSPER_MODULE": "prosper.embeddings.similarity_cli"})
        env.pop("PROSPER_PARAMS", None)
        env.pop("LSB_JOBINDEX", None)
        env.pop("PYTHONWARNINGS", None)
        pilot.require(pilot.PYTHON.is_file() and worker_root.is_dir() and
                      all(env[key] == value for key, value in pilot.THREAD_ENV.items()),
                      "Worker executable/root/native thread binding invalid")
        stdout, stderr = self.execution_log_paths(t)
        stdout.parent.mkdir(parents=True, exist_ok=True)
        command = ["bash", str(worker_root / "scripts/lsf/prosper_job.sh"), *args]
        try:
            with stdout.open("xb") as out_stream, stderr.open("xb") as err_stream:
                subprocess.run(command, cwd=worker_root, env=env, check=True,
                               stdout=out_stream, stderr=err_stream)
        except subprocess.CalledProcessError as exc:
            self.failure_diagnostic(t, "worker_nonzero_exit", exc.returncode,
                                    self.execution_log_bindings(t))
            raise
        logs = self.execution_log_bindings(t)
        if stderr.stat().st_size:
            self.failure_diagnostic(t, "worker_stderr_nonempty", 0, logs)
            raise ValueError(f"{t['id']}: OT worker stderr nonempty; inspect {stderr}")
        return logs

    def subset(self, t: dict, m: dict) -> None:
        _, _, subset, _ = self.task_paths(t)
        pilot.require(subset.is_dir() and pilot.ids_in(subset) == t["queries"],
                      f"{t['id']}: query subset incomplete/extra")
        for sid in t["queries"]:
            link = subset / f"{sid}.pkl"
            source = self.pilot.root / "features_in/tumor_only" / f"{sid}.pkl"
            pilot.require(link.is_symlink() and link.resolve() == source.resolve() and
                          m["bindings"][f"feature_{sid}"]["path"] == str(source.resolve()),
                          f"{t['id']}: subset symlink source changed")
            check_bound(m["bindings"][f"feature_{sid}"])

    def task_rows(self, t: dict, m: dict) -> dict:
        output, log, subset, _ = self.task_paths(t)
        refs = self.references(m, t["arm"])
        n = len(t["queries"])
        pilot.require(output.is_dir() and
                      {p.name for p in output.iterdir()} ==
                      {f"similarity_{sid}.pkl" for sid in t["queries"]} and
                      log.is_dir() and {p.name for p in log.iterdir()} ==
                      {f"provenance_1_{n}.json"},
                      f"{t['id']}: incomplete/extra OT rows or normalized logs")
        sidecar = log / f"provenance_1_{n}.json"
        prov = pilot.json_read(sidecar)
        library = (self.inputs.pilot_inputs.raw_all if t["arm"] == "query_only"
                   else self.inputs.pilot_inputs.raw_both)
        pilot.require(prov["task"] == t["task"] and prov["encoder"] == "uni2" and
                      prov["method"] == "ot" and prov["feature_dim"] == 1536 and
                      prov["metric"] == "cosine" and prov["reg"] == 0 and
                      prov["external"] is True and prov["num_workers"] == 8 and
                      prov["tile_fraction"] is None and
                      prov.get("tile_fraction_nesting") is None and
                      prov["n_library"] == len(refs) and
                      prov["n_queries_total"] == n and prov["query_range"] == [1, n] and
                      prov["n_queries_this_element"] == n and prov["n_done"] == n and
                      prov["git_commit"] == self.pilot.spec.source_commit and
                      isinstance(prov["lsf_job"], str) and bool(prov["lsf_job"]) and
                      prov.get("lsf_index") is None and
                      Path(prov["query_features"]).resolve() == subset.resolve() and
                      Path(prov["library_features"]).resolve() == library.resolve(),
                      f"{t['id']}: OT sidecar task/range/job/array/source mismatch")
        rows = {}
        for sid in t["queries"]:
            source = output / f"similarity_{sid}.pkl"
            pilot.row(source, sid, refs)
            rows[sid] = pilot.digest(source)
        return {"rows": rows, "sidecar_sha256": pilot.digest(sidecar),
                "source_job": prov["lsf_job"]}

    def run_task(self, task_id: str) -> None:
        runtime = pilot.runtime_info("ot")
        pilot.require(len(runtime["affinity_cpus"]) >= 8 and
                      runtime["lsf_slots"] == "8",
                      "Scalar OT task needs eight allocated/affinity CPUs")
        m = self.plan()
        matches = [t for t in m["tasks"] if t["id"] == task_id]
        pilot.require(len(matches) == 1 and re.fullmatch(r"[A-Za-z0-9_]+", task_id),
                      f"Unknown/unsafe frozen task ID: {task_id}")
        t = matches[0]
        self.small_bindings(m)
        self.references(m, t["arm"])
        output, log, subset, receipt = self.task_paths(t)
        stdout, stderr = self.execution_log_paths(t)
        diagnostic = self.root / "diagnostics" / f"{t['id']}.json"
        pilot.require(not output.exists() and not log.exists() and
                      not subset.exists() and not receipt.exists() and
                      not stdout.exists() and not stderr.exists() and
                      not diagnostic.exists(),
                      f"{task_id}: refusing pre-existing task/log/subset/receipt/diagnostic")
        subset.mkdir(parents=True, exist_ok=False)
        for sid in t["queries"]:
            source = self.pilot.root / "features_in/tumor_only" / f"{sid}.pkl"
            (subset / f"{sid}.pkl").symlink_to(source.resolve())
        self.subset(t, m)
        library = (self.inputs.pilot_inputs.raw_all if t["arm"] == "query_only"
                   else self.inputs.pilot_inputs.raw_both)
        n = len(t["queries"])
        logs = self.run_ot_worker(
            t,
            ["--encoder", "uni2", "--method", "ot", "--feature-dim", "1536",
             "--metric", "cosine", "--reg", "0", "--num-workers", "8",
             "--features", str(subset), "--library-features", str(library),
             "--lo", "1", "--hi", str(n), "--task", t["task"]])
        self.subset(t, m)
        result = {**self.task_rows(t, m), "execution_logs": logs}
        pilot.require(result["source_job"] == runtime["lsf_job"],
                      f"{task_id}: worker job differs from active LSF job")
        self.small_bindings(m)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        pilot.json_write(receipt, {"stage": "run-task", "task": t,
                                   "manifest_sha256": pilot.digest(self.root / "manifest.json"),
                                   "result": result, "runtime": runtime})

    def global_against_manifest(self, m: dict) -> None:
        audit = self.global_audit()
        pilot.require(all(audit[key] == m[key] for key in (
            "ids", "reused", "historic", "corrected_historical_ids", "refs",
            "libraries", "eligible", "bindings")),
            "Global source/input payloads or historical ranking bridge changed")

    def receipt_inventory(self, m: dict) -> dict:
        directory = self.root / "receipts"
        pilot.require(directory.is_dir() and
                      {path.name for path in directory.iterdir()} ==
                      {f"{t['id']}.json" for t in m["tasks"]},
                      "Full task receipt inventory incomplete/extra")
        return {t["id"]: bound(directory / f"{t['id']}.json") for t in m["tasks"]}

    def verified_union(self, arm: str, m: dict) -> dict[str, Path]:
        refs = self.references(m, arm)
        rows = {}
        for sid in m["historic"][arm]:
            rows[sid] = self.inputs.historical[arm].cache / f"similarity_{sid}.pkl"
        marker = pilot.json_read(self.pilot.root / "PILOT_COMPLETE.json")
        for sid in m["reused"]:
            entry = next(x for x in marker["tasks"] if
                         x["arm"] == arm and x["query"] == sid)
            source = self.out / entry["task"] / f"similarity_{sid}.pkl"
            pilot.require(pilot.digest(source) ==
                          m["bindings"][f"pilot_row_{arm}_{sid}"]["sha256"],
                          f"{arm}/{sid}: accepted pilot row changed")
            rows[sid] = source
        for t in m["tasks"]:
            if t["arm"] != arm:
                continue
            _, _, _, receipt_path = self.task_paths(t)
            rec = pilot.json_read(receipt_path)
            result = {**self.task_rows(t, m),
                      "execution_logs": self.execution_log_bindings(t)}
            _, stderr = self.execution_log_paths(t)
            pilot.require(stderr.stat().st_size == 0 and
                          not (self.root / "diagnostics" / f"{t['id']}.json").exists(),
                          f"{t['id']}: OT worker stderr/diagnostic prevents merge")
            self.subset(t, m)
            pilot.require(rec["stage"] == "run-task" and rec["task"] == t and
                          rec["manifest_sha256"] == pilot.digest(self.root / "manifest.json")
                          and rec["result"] == result and
                          rec["runtime"]["lsf_job"] == result["source_job"] and
                          rec["runtime"]["joblib_workers"] == 8 and
                          rec["runtime"]["lsf_slots"] == "8" and
                          len(rec["runtime"]["affinity_cpus"]) >= 8 and
                          rec["runtime"]["thread_env"] == pilot.THREAD_ENV and
                          all(pool["num_threads"] == 1 for pool in
                              rec["runtime"]["thread_pools"]) and
                          rec["runtime"]["lsf_index"] in (None, "", "0"),
                          f"{t['id']}: task receipt/job/source changed")
            for sid in t["queries"]:
                pilot.require(sid not in rows, f"{arm}/{sid}: duplicate union row")
                rows[sid] = self.out / t["task"] / f"similarity_{sid}.pkl"
        expected = set(m["historic"][arm]) | set(m["ids"])
        pilot.require(set(rows) == expected and len(rows) ==
                      self.spec.n_old + self.spec.n_new and
                      pilot.CONTROL in m["historic"][arm] and
                      rows[pilot.CONTROL] ==
                      self.inputs.historical[arm].cache /
                      f"similarity_{pilot.CONTROL}.pkl",
                      f"{arm}: missing/extra queries or replayed control")
        for sid, source in rows.items():
            pilot.row(source, sid, refs)
        return rows

    def verify_combined(self, arm: str, m: dict, rows: dict[str, Path]) -> dict:
        refs = m["refs"][arm]
        frame = matrix(self.matrix_path(arm), sorted(rows), refs)
        original = matrix(self.inputs.historical[arm].matrix,
                          m["historic"][arm], refs)
        for sid, source in rows.items():
            values = pilot.row(source, sid, refs)
            pilot.require(bits_equal(values, frame.loc[sid, refs].to_numpy()),
                          f"{arm}/{sid}: combined row differs in float64 bits")
            if sid in m["historic"][arm]:
                pilot.require(bits_equal(frame.loc[sid, refs].to_numpy(),
                                         original.loc[sid, refs].to_numpy()),
                              f"{arm}/{sid}: original matrix cells changed")
        prov_path = self.sidecar_path(arm)
        prov = pilot.json_read(prov_path)
        union = self.root / "unions" / arm
        pilot.require(prov["model"] == MODELS[arm] and
                      prov["method"] == "ot_cosine" and prov["external"] is True and
                      prov["n_query_slides"] == len(rows) and
                      prov["n_reference_slides"] == len(refs) and
                      prov["same_patient_pairs_masked"] is None and
                      Path(prov["similarity_dir"]).resolve() == union.resolve() and
                      Path(prov["out_dir"]).resolve() ==
                      (self.out / "similarity_matrix").resolve(),
                      f"{arm}: stock combine sidecar mismatch")
        return {"matrix_sha256": pilot.digest(self.matrix_path(arm)),
                "sidecar_sha256": pilot.digest(prov_path),
                "n_queries": len(rows), "n_references": len(refs)}

    def merge(self) -> None:
        runtime = pilot.runtime_info("score")
        m = self.plan()
        preflight_sha = pilot.digest(self.root / "preflight_receipt.json")
        pilot.require(not (self.root / "COMPLETE.json").exists() and
                      not (self.root / "merge_receipt.json").exists() and
                      not (self.root / "unions").exists(),
                      "Refusing pre-existing full merge artifacts")
        for arm in ARMS:
            pilot.require(not self.matrix_path(arm).exists() and
                          not self.sidecar_path(arm).exists(),
                          f"{arm}: refusing existing combine output/sidecar")
        self.global_against_manifest(m)
        directory = self.root / "execution_logs"
        pilot.require(directory.is_dir() and
                      {path.name for path in directory.iterdir()} ==
                      {f"{t['id']}.{stream}.log" for t in m["tasks"]
                       for stream in ("stdout", "stderr")},
                      "Full OT execution-log inventory incomplete/extra")
        unions = {arm: self.verified_union(arm, m) for arm in ARMS}
        task_receipts = self.receipt_inventory(m)
        results = {}
        for arm in ARMS:
            directory = self.root / "unions" / arm
            directory.mkdir(parents=True, exist_ok=False)
            for sid, source in unions[arm].items():
                (directory / f"similarity_{sid}.pkl").symlink_to(source.resolve())
            pilot.require({p.name for p in directory.iterdir()} ==
                          {f"similarity_{sid}.pkl" for sid in unions[arm]} and
                          all((directory / f"similarity_{sid}.pkl").is_symlink() and
                              (directory / f"similarity_{sid}.pkl").resolve() ==
                              source.resolve() for sid, source in unions[arm].items()),
                          f"{arm}: union symlinks differ from bound sources")
            self.pilot.run_worker(
                "prosper.embeddings.combine_similarity_cli",
                ["--encoder", "uni2", "--method", "ot", "--external",
                 "--similarity-dir", str(directory), "--matrix-model", MODELS[arm],
                 "--min-slides", str(len(unions[arm]))], "merge")
            results[arm] = self.verify_combined(arm, m, unions[arm])
        self.global_against_manifest(m)
        final_unions = {arm: self.verified_union(arm, m) for arm in ARMS}
        final_results = {arm: self.verify_combined(arm, m, final_unions[arm])
                         for arm in ARMS}
        pilot.require(final_results == results and
                      self.receipt_inventory(m) == task_receipts and
                      pilot.digest(self.root / "preflight_receipt.json") ==
                      preflight_sha and
                      all(pilot.digest(self.matrix_path(arm)) ==
                          final_results[arm]["matrix_sha256"] and
                          pilot.digest(self.sidecar_path(arm)) ==
                          final_results[arm]["sidecar_sha256"] for arm in ARMS),
                      "Merge output/task receipts/preflight changed before finalization")
        pilot.json_write(self.root / "merge_receipt.json", {
            "stage": "merge", "manifest_sha256": pilot.digest(self.root / "manifest.json"),
            "preflight_receipt_sha256": preflight_sha,
            "task_receipts": task_receipts,
            "results": final_results, "runtime": runtime})
        pilot.json_write(self.root / "COMPLETE.json", {
            "status": "technical_full_extension_for_independent_review",
            "not_scientific_or_author_gate_c_approval": True,
            "manifest_sha256": pilot.digest(self.root / "manifest.json"),
            "merge_receipt_sha256": pilot.digest(self.root / "merge_receipt.json"),
            "pilot_marker_sha256": self.spec.marker_sha,
            "pilot_review_sha256": self.spec.review_sha})

    def verify_complete_chain(self) -> None:
        m = self.plan()
        complete = pilot.json_read(self.root / "COMPLETE.json")
        merge = pilot.json_read(self.root / "merge_receipt.json")
        pilot.require(complete["status"] ==
                      "technical_full_extension_for_independent_review" and
                      complete["not_scientific_or_author_gate_c_approval"] is True and
                      complete["manifest_sha256"] == pilot.digest(self.root / "manifest.json")
                      and complete["merge_receipt_sha256"] ==
                      pilot.digest(self.root / "merge_receipt.json") and
                      complete["pilot_marker_sha256"] == self.spec.marker_sha and
                      complete["pilot_review_sha256"] == self.spec.review_sha and
                      merge["stage"] == "merge" and
                      merge["manifest_sha256"] == complete["manifest_sha256"] and
                      merge["preflight_receipt_sha256"] ==
                      pilot.digest(self.root / "preflight_receipt.json") and
                      merge["task_receipts"] == self.receipt_inventory(m),
                      "Full completion/preflight/task receipt hash chain changed")
        unions = {arm: self.verified_union(arm, m) for arm in ARMS}
        pilot.require(merge["results"] ==
                      {arm: self.verify_combined(arm, m, unions[arm]) for arm in ARMS},
                      "Full completion row/sidecar/log/matrix chain changed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "run-task", "merge"),
                        required=True)
    parser.add_argument("--task-id")
    args = parser.parse_args(argv)
    pilot.require((args.stage == "run-task") == (args.task_id is not None),
                  "Only run-task accepts and requires --task-id")
    full = Full(pilot.out_root())
    if args.stage == "preflight":
        full.preflight()
    elif args.stage == "run-task":
        full.run_task(args.task_id)
    else:
        full.merge()


if __name__ == "__main__":
    main()
