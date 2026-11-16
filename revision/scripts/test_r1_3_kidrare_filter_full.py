"""Isolated [decision record] full-extension source, OT receipt and real stock-combine tests."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import r1_3_kidrare_filter_extension as p
import r1_3_kidrare_filter_full as f
import test_r1_3_kidrare_filter_extension as pilot_fixtures
from prosper.embeddings import combine_similarity_cli

dump = pilot_fixtures.dump
REAL_RUN = subprocess.run


class FullTests(unittest.TestCase):
    def setUp(self):
        self.fx = pilot_fixtures.PilotTests(methodName="runTest")
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        for sid, n in (("new_d", 4), ("new_e", 5)):
            feats = np.ones((n, 1536), dtype=np.float32)
            coords = [{"wsi_loc": [i * 512, i * 512]} for i in range(n)]
            dump(self.fx.inputs.source_features / f"{sid}.pkl", (feats, coords))
            self.fx.new.append(sid)
        sid = "new_empty"
        feats = np.ones((2, 1536), dtype=np.float32)
        dump(self.fx.inputs.source_features / f"{sid}.pkl",
             (feats, [{"wsi_loc": [i * 512, i * 512]} for i in range(2)]))
        manifest = pd.read_csv(self.fx.inputs.changed_manifest, dtype=str)
        for sid in ("new_d", "new_e"):
            row = manifest.iloc[0].copy()
            row["slide_id"] = sid
            manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
        manifest.to_csv(self.fx.inputs.changed_manifest, index=False)
        self.fx.spec = replace(
            self.fx.spec, n_new=6, manifest_sha=p.digest(self.fx.inputs.changed_manifest),
            ids_sha=hashlib.sha256("".join(
                sid + "\n" for sid in sorted(self.fx.new)).encode()).hexdigest())
        labels = dict(zip(self.fx.refids, ["NORMAL", "MISC", *["WT"] * 10]))
        historical = {}
        old_queries = sorted([p.CONTROL, "hist2"])
        for arm, refs in (("query_only", self.fx.refids),
                          ("both_sided", self.fx.refids[1:])):
            source = self.fx.oldrows[arm]
            control = p.row(source, p.CONTROL, refs)
            cache = self.fx.out / f"historical_{arm}"
            cache_rows = {p.CONTROL: control, "hist2": control + 0.01}
            for qid in old_queries:
                dump(cache / f"similarity_{qid}.pkl",
                     list(zip(refs, cache_rows[qid].tolist())))
            path = self.fx.out / f"matrix_{arm}.pkl"
            frame = pd.DataFrame([cache_rows[qid] for qid in old_queries],
                                 index=old_queries, columns=refs, dtype=np.float64)
            dump(path, frame)
            eligible = set(refs) - {"ref01"}
            if arm == "both_sided":
                eligible -= {"ref00"}
            records = []
            for qid in old_queries:
                top = p.top10(refs, cache_rows[qid], eligible, qid)
                for n, sid in enumerate(top, 1):
                    records.append({"query": qid, "rank": n, "neighbour": sid,
                                    "neighbour_label": labels[sid],
                                    "distance": cache_rows[qid][refs.index(sid)]})
            ranked = self.fx.out / f"historical_rank_{arm}.csv.gz"
            pd.DataFrame(records).to_csv(ranked, index=False, compression="gzip")
            historical[arm] = f.Historical(path, cache, ranked)
        self.fx.pilot = p.Pilot(self.fx.out, self.fx.inputs, self.fx.spec)
        self.fx.stages()
        root = self.fx.pilot.root
        marker = p.json_read(root / "PILOT_COMPLETE.json")
        review = {
            "status": "accepted_technical_pilot_for_full_filter_extension",
            "pilot_complete_sha256": p.digest(root / "PILOT_COMPLETE.json"),
            **{key: marker[key] for key in ("preflight_sha256", "selection_sha256",
                                            "score_receipt_sha256",
                                            "filter_receipt_sha256")},
            "pilot_driver_sha256": p.digest(p.ROOT / "scripts/r1_3_kidrare_filter_extension.py"),
            "pilot_tests_sha256": p.digest(p.ROOT / "scripts/test_r1_3_kidrare_filter_extension.py"),
            "reviewer": {"agent_id": "fixture-reviewer"},
            "author_full_extension_condition_satisfied": True,
            "preserve_original_historical_control_rows": True,
            "blocking_findings": [],
        }
        p.json_write(root / "PILOT_INDEPENDENT_REVIEW.json", review)
        provenance = self.fx.out / "correction_provenance.json"
        p.json_write(provenance, {"inputs": {
            "kidrare_query_only_ranked": f.bound(historical["query_only"].ranked),
            "kidrare_both_sided_ranked": f.bound(historical["both_sided"].ranked),
        }})
        membership = self.fx.out / "kidrare_source_2331_eligibility.csv"
        legacy = [
            (p.CONTROL, "WT"), ("hist2", "NORMAL"), (f.ZERO_LIVER, "LIVER")]
        entries = [{
            "slide_id": sid, "processed": "True", "tumor_group": label,
            "original_tumor_group": label, "corrected_tumor_group": label,
            "pathpt_only_new_label": "False", "legacy_eligible": "True",
            "amended_eligible": "True",
            "query_only_eligible": str(sid != f.ZERO_LIVER),
            "both_sided_eligible": str(sid == p.CONTROL),
        } for sid, label in legacy]
        entries += [{
            "slide_id": sid, "processed": "True", "tumor_group": "WT_UNVERIFIED",
            "original_tumor_group": "WT_UNVERIFIED",
            "corrected_tumor_group": "WT", "pathpt_only_new_label": "True",
            "legacy_eligible": "False", "amended_eligible": "True",
            "query_only_eligible": "False", "both_sided_eligible": "False",
        } for sid in self.fx.new]
        pd.DataFrame(entries).to_csv(membership, index=False)
        self.spec = f.Spec(
            n_old=2, n_new=6, n_pilot=3, n_source=9, n_legacy=3, n_normal=1,
            membership_sha=p.digest(membership),
            marker_sha=p.digest(root / "PILOT_COMPLETE.json"),
            review_sha=p.digest(root / "PILOT_INDEPENDENT_REVIEW.json"),
            pilot_driver_sha=review["pilot_driver_sha256"],
            pilot_tests_sha=review["pilot_tests_sha256"],
            provenance_sha=p.digest(provenance),
            matrix_all_sha=p.digest(historical["query_only"].matrix),
            matrix_both_sha=p.digest(historical["both_sided"].matrix),
            rank_all_sha=p.digest(historical["query_only"].ranked),
            rank_both_sha=p.digest(historical["both_sided"].ranked),
            reviewer="fixture-reviewer",
        )
        self.full = f.Full(
            self.fx.out, f.Inputs(self.fx.inputs, provenance, historical, membership),
            self.spec, self.fx.spec)
        self.corrupt_combine = None
        self.after_second_combine = None
        self.ot_stderr = b""
        self.ot_exit_code = 0

    def worker(self, instance, module, args, stage):
        if module == "prosper.embeddings.combine_similarity_cli":
            self.assertEqual(stage, "merge")
            with patch.dict(os.environ, {"PROSPER_OUT": str(self.fx.out)}):
                self.assertEqual(combine_similarity_cli.main(args), 0)
            model = args[args.index("--matrix-model") + 1]
            arm = next(arm for arm in f.ARMS if f.MODELS[arm] == model)
            if self.corrupt_combine:
                path = self.full.matrix_path(arm)
                with path.open("rb") as stream:
                    frame = pickle.load(stream)
                if self.corrupt_combine == "dtype":
                    frame = frame.astype(np.float32)
                elif self.corrupt_combine == "order":
                    frame = frame.iloc[::-1]
                elif self.corrupt_combine == "shape":
                    frame = frame.iloc[:-1]
                elif self.corrupt_combine == "bits":
                    frame.iloc[0, 0] += 0.01
                dump(path, frame)
            if stage == "merge" and arm == "both_sided" and self.after_second_combine:
                self.after_second_combine()
            return
        self.assertEqual((module, stage), ("prosper.embeddings.similarity_cli", "ot"))
        a = dict(zip(args[::2], args[1::2]))
        self.assertEqual(a["--lo"], "1")
        self.assertEqual(a["--encoder"], "uni2")
        self.assertEqual(a["--method"], "ot")
        self.assertEqual(a["--metric"], "cosine")
        self.assertEqual(a["--num-workers"], "8")
        queries = p.ids_in(Path(a["--features"]))
        refs = p.ids_in(Path(a["--library-features"]))
        self.assertEqual(a["--hi"], str(len(queries)))
        for index, sid in enumerate(queries):
            values = [0.3 + index * 0.01 + i * 0.001 for i in range(len(refs))]
            dump(self.fx.out / a["--task"] / f"similarity_{sid}.pkl",
                 list(zip(refs, values)))
        log = self.fx.out / "logs" / a["--task"].replace("/", "_")
        log.mkdir(parents=True)
        p.json_write(log / f"provenance_1_{len(queries)}.json", {
            "task": a["--task"], "encoder": "uni2", "method": "ot",
            "feature_dim": 1536, "metric": "cosine", "reg": 0,
            "external": True, "num_workers": 8, "tile_fraction": None,
            "tile_fraction_nesting": None, "n_library": len(refs),
            "n_queries_total": len(queries), "query_range": [1, len(queries)],
            "n_queries_this_element": len(queries), "n_done": len(queries),
            "git_commit": self.fx.spec.source_commit,
            "query_features": a["--features"],
            "library_features": a["--library-features"],
            "lsf_job": "fixture-job", "lsf_index": None,
        })

    def invoke(self, stage, task_id=None):
        runtime = {"lsf_job": "fixture-job", "joblib_workers": 8,
                   "lsf_index": "0", "lsf_slots": "8",
                   "affinity_cpus": list(range(8)), "thread_env": p.THREAD_ENV,
                   "thread_pools": [{"num_threads": 1}]}
        with patch.object(p, "runtime_info", return_value=runtime), \
                patch.object(p.Pilot, "run_worker", autospec=True,
                             side_effect=self.worker), \
                patch.object(f.subprocess, "run", side_effect=self.subprocess_worker):
            if stage == "run_task":
                return self.full.run_task(task_id)
            return getattr(self.full, stage)()

    def subprocess_worker(self, command, **kwargs):
        if command[0] == "git":
            return REAL_RUN(command, **kwargs)
        self.assertEqual(command[:2], [
            "bash", str(p.PROSPER / "scripts/lsf/prosper_job.sh")])
        self.assertTrue(kwargs["check"])
        self.assertEqual(kwargs["cwd"], p.PROSPER)
        env = kwargs["env"]
        self.assertNotIn("PROSPER_PARAMS", env)
        self.assertNotIn("LSB_JOBINDEX", env)
        self.assertNotIn("PYTHONWARNINGS", env)
        self.assertEqual(env["PROSPER_ROOT"], str(p.PROSPER))
        self.assertEqual(env["PROSPER_OUT"], str(self.fx.out.resolve()))
        self.assertEqual(env["COMET_ROOT"], str(p.COMET))
        self.assertEqual(env["PROSPER_PYTHON"], str(p.PYTHON))
        self.assertEqual(env["PROSPER_MODULE"], "prosper.embeddings.similarity_cli")
        self.assertEqual({key: env[key] for key in p.THREAD_ENV}, p.THREAD_ENV)
        self.worker(None, env["PROSPER_MODULE"], command[2:], "ot")
        kwargs["stdout"].write(b"[worker] scalar OT fixture completed\n")
        kwargs["stderr"].write(self.ot_stderr)
        if self.ot_exit_code:
            raise subprocess.CalledProcessError(self.ot_exit_code, command)
        return subprocess.CompletedProcess(command, 0)

    def completed_tasks(self):
        self.invoke("preflight")
        m = self.full.plan()
        self.assertEqual(len(m["tasks"]), 4)
        for t in m["tasks"]:
            self.invoke("run_task", t["id"])
        return m

    def test_preflight_pair_singleton_stock_combine_and_source_immutability(self):
        sources = {key: f.bound(path) for key, path in self.full.sources().items()}
        m = self.completed_tasks()
        self.assertEqual([len(t["queries"]) for t in m["tasks"]], [2, 1, 2, 1])
        self.assertEqual(m["corrected_historical_ids"],
                         sorted([p.CONTROL, "hist2"]))
        self.assertEqual(set(m["historic"]["both_sided"]),
                         {p.CONTROL, "hist2"})
        self.assertEqual(set(m["ids"]) - set(m["reused"]),
                         {sid for t in m["tasks"] for sid in t["queries"]})
        for task in m["tasks"]:
            receipt = p.json_read(self.full.task_paths(task)[-1])
            for stream in ("stdout", "stderr"):
                log = self.full.execution_log_paths(task)[0 if stream == "stdout" else 1]
                self.assertEqual(receipt["result"]["execution_logs"][stream],
                                 f.bound(log))
            self.assertEqual(self.full.execution_log_paths(task)[1].read_bytes(), b"")
        for key, binding in m["bindings"].items():
            with self.subTest(binding=key):
                f.check_bound(binding)
        for arm in f.ARMS:
            control = self.fx.out / f"historical_{arm}" / f"similarity_{p.CONTROL}.pkl"
            self.assertEqual(m["bindings"][f"{arm}_cache_{p.CONTROL}"]["sha256"],
                             p.digest(control))
        self.invoke("merge")
        self.full.verify_complete_chain()
        marker = p.json_read(self.full.root / "COMPLETE.json")
        self.assertEqual(marker["status"],
                         "technical_full_extension_for_independent_review")
        merged = p.json_read(self.full.root / "merge_receipt.json")
        self.assertEqual(merged["preflight_receipt_sha256"],
                         p.digest(self.full.root / "preflight_receipt.json"))
        self.assertEqual(merged["task_receipts"], self.full.receipt_inventory(m))
        self.assertEqual(len(merged["task_receipts"]), 4)
        for arm in f.ARMS:
            with self.full.matrix_path(arm).open("rb") as stream:
                result = pickle.load(stream)
            self.assertEqual(result.shape, (8, len(m["refs"][arm])))
            self.assertEqual(result.dtypes.unique().tolist(), [np.dtype("float64")])
        self.assertEqual({key: f.bound(path) for key, path in
                          self.full.sources().items()}, sources)
        with self.assertRaisesRegex(ValueError, "existing full root"):
            self.invoke("preflight")
        with self.assertRaisesRegex(ValueError, "pre-existing task"):
            self.invoke("run_task", m["tasks"][0]["id"])
        with self.assertRaisesRegex(ValueError, "pre-existing full merge"):
            self.invoke("merge")

    def test_missing_or_altered_pilot_acceptance_refused(self):
        marker = self.fx.pilot.root / "PILOT_COMPLETE.json"
        marker.write_bytes(marker.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "pilot marker/review/code hash"):
            self.invoke("preflight")
        self.assertFalse(self.full.root.exists())

    def test_missing_review_and_changed_filtered_feature_refused(self):
        review = self.fx.pilot.root / "PILOT_INDEPENDENT_REVIEW.json"
        review.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.invoke("preflight")
        self.assertFalse(self.full.root.exists())

    def test_review_status_and_marker_linkage_refused_even_with_rebound_fixture_hash(self):
        path = self.fx.pilot.root / "PILOT_INDEPENDENT_REVIEW.json"
        review = p.json_read(path)
        for change in ({"status": "pending_review"},
                       {"pilot_complete_sha256": "not-the-marker"}):
            with self.subTest(change=change):
                path.write_text(json.dumps(review | change))
                self.full.spec = replace(self.spec, review_sha=p.digest(path))
                with self.assertRaisesRegex(ValueError, "independent review linkage invalid"):
                    self.invoke("preflight")
                self.assertFalse(self.full.root.exists())
        path.write_text(json.dumps(review))

    def test_global_source_feature_mutation_refused(self):
        sid = self.fx.new[0]
        source = self.fx.inputs.source_features / f"{sid}.pkl"
        source.write_bytes(source.read_bytes() + b"mutation")
        with self.assertRaisesRegex(ValueError, "Pilot source input changed"):
            self.invoke("preflight")

    def test_accepted_pilot_new_row_mutation_refused(self):
        entry = next(x for x in p.json_read(
            self.fx.pilot.root / "PILOT_COMPLETE.json")["tasks"]
            if x["query"] != p.CONTROL)
        row = self.fx.out / entry["task"] / f"similarity_{entry['query']}.pkl"
        row.write_bytes(row.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "pilot receipt mismatch"):
            self.invoke("preflight")
        self.assertFalse(self.full.root.exists())

    def test_cache_matrix_bit_mismatch_refused(self):
        path = self.full.inputs.historical["query_only"].cache / f"similarity_{p.CONTROL}.pkl"
        with path.open("rb") as stream:
            entries = pickle.load(stream)
        entries[0] = (entries[0][0], entries[0][1] + 0.01)
        dump(path, entries)
        with self.assertRaisesRegex(ValueError, "cache/matrix bits differ"):
            self.invoke("preflight")

    def test_historical_cache_membership_refused_even_when_count_unchanged(self):
        cache = self.full.inputs.historical["query_only"].cache
        (cache / "similarity_hist2.pkl").rename(cache / "similarity_wrong.pkl")
        with self.assertRaisesRegex(ValueError, "differs from corrected eligibility"):
            self.invoke("preflight")

    def test_corrected_source_hash_and_zero_exclusion_refused(self):
        path = self.full.inputs.corrected_membership
        original = pd.read_csv(path, dtype=str, keep_default_na=False)
        altered = original.copy()
        altered.loc[altered.slide_id == f.ZERO_LIVER, "query_only_eligible"] = "True"
        altered.loc[altered.slide_id == "hist2", "query_only_eligible"] = "False"
        altered.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "eligibility source hash changed"):
            self.invoke("preflight")
        with self.assertRaisesRegex(ValueError, "zero-tile LIVER exclusion changed"):
            f.corrected_old_ids(path, self.fx.new,
                                replace(self.spec, membership_sha=p.digest(path)))
        self.assertFalse(self.full.root.exists())

    def test_corrected_source_normal_evaluation_exclusion_not_raw_exclusion(self):
        path = self.full.inputs.corrected_membership
        original = pd.read_csv(path, dtype=str, keep_default_na=False)
        self.assertEqual(f.corrected_old_ids(path, self.fx.new, self.spec),
                         sorted([p.CONTROL, "hist2"]))
        original.loc[original.slide_id == "hist2", "both_sided_eligible"] = "True"
        original.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Historical/new query membership"):
            f.corrected_old_ids(path, self.fx.new,
                                replace(self.spec, membership_sha=p.digest(path)))

    def test_historical_matrix_cells_mismatch_even_with_rebound_fixture_hash(self):
        path = self.full.inputs.historical["both_sided"].matrix
        with path.open("rb") as stream:
            frame = pickle.load(stream)
        frame.iloc[0, 0] = np.nextafter(frame.iloc[0, 0], np.inf)
        dump(path, frame)
        self.full.spec = replace(self.spec, matrix_both_sha=p.digest(path))
        with self.assertRaisesRegex(ValueError, "cache/matrix bits differ"):
            self.invoke("preflight")

    def test_historical_rank_label_distance_and_id_mismatch(self):
        for column, value in (("neighbour_label", "WRONG"),
                              ("distance", 9.0), ("neighbour", "ref11")):
            with self.subTest(column=column):
                path = self.full.inputs.historical["query_only"].ranked
                original = pd.read_csv(path)
                tampered = original.copy()
                tampered.loc[0, column] = value
                tampered.to_csv(path, index=False, compression="gzip")
                prov = self.full.inputs.provenance
                payload = p.json_read(prov)
                payload["inputs"]["kidrare_query_only_ranked"]["sha256"] = p.digest(path)
                prov.write_text(json.dumps(payload))
                self.full.spec = replace(
                    self.full.spec, rank_all_sha=p.digest(path),
                    provenance_sha=p.digest(prov))
                with self.assertRaisesRegex(ValueError, "historical ranked"):
                    self.invoke("preflight")
                original.to_csv(path, index=False, compression="gzip")
                payload["inputs"]["kidrare_query_only_ranked"]["sha256"] = p.digest(path)
                prov.write_text(json.dumps(payload))
                self.full.spec = replace(
                    self.full.spec, rank_all_sha=p.digest(path),
                    provenance_sha=p.digest(prov))

    def test_task_subset_source_manifest_and_collision_refused(self):
        self.invoke("preflight")
        task = self.full.plan()["tasks"][0]
        sid = task["queries"][0]
        feature = self.fx.pilot.root / "features_in/tumor_only" / f"{sid}.pkl"
        feature.write_bytes(feature.read_bytes() + b"mutation")
        with self.assertRaisesRegex(ValueError, "Bound input changed"):
            self.invoke("run_task", task["id"])
        self.assertFalse((self.fx.out / task["task"]).exists())

    def test_task_metadata_binding_and_existing_output_log_collisions(self):
        self.invoke("preflight")
        t = self.full.plan()["tasks"][0]
        metadata = self.fx.inputs.metadata
        metadata.write_bytes(metadata.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Bound input changed"):
            self.invoke("run_task", t["id"])
        self.assertFalse((self.fx.out / t["task"]).exists())

    def test_preflight_normalized_log_collision_refused(self):
        ids = self.fx.pilot.new_ids()
        selected = list(p.json_read(self.fx.pilot.root / "selection.json")["selection"].values())
        task = f.partition(ids, selected)[0]
        log = self.fx.out / "logs" / task["task"].replace("/", "_")
        log.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "task output/log collision"):
            self.invoke("preflight")
        self.assertFalse(self.full.root.exists())

    def test_per_task_foreign_same_id_library_path_refused_before_worker(self):
        self.invoke("preflight")
        task = self.full.plan()["tasks"][0]
        foreign = self.fx.out / "foreign_reference_bank"
        shutil.copytree(self.fx.raw_all, foreign)
        alias = self.fx.out / "repointed_reference_bank"
        alias.symlink_to(foreign, target_is_directory=True)
        self.assertEqual(p.ids_in(alias), p.ids_in(self.fx.raw_all))
        self.full.inputs = replace(
            self.full.inputs,
            pilot_inputs=replace(self.fx.inputs, raw_all=alias))
        with self.assertRaisesRegex(ValueError, "library path differs from bound"):
            self.invoke("run_task", task["id"])
        self.assertFalse(self.full.task_paths(task)[0].exists())
        self.assertFalse(self.full.task_paths(task)[-1].exists())

    def test_preflight_combine_matrix_collision_refused(self):
        target = self.full.matrix_path("query_only")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"existing unrelated output")
        with self.assertRaisesRegex(ValueError, "combine output/sidecar collision"):
            self.invoke("preflight")
        self.assertFalse(self.full.root.exists())

    def test_task_plan_overlap_and_omission_refused(self):
        self.invoke("preflight")
        path = self.full.root / "manifest.json"
        receipt_path = self.full.root / "preflight_receipt.json"
        original = path.read_bytes()
        original_receipt = receipt_path.read_bytes()
        for operation in ("omit", "overlap"):
            with self.subTest(operation=operation):
                manifest = json.loads(original)
                if operation == "omit":
                    manifest["tasks"].pop()
                else:
                    manifest["tasks"][1]["queries"] = manifest["tasks"][0]["queries"]
                path.write_text(json.dumps(manifest))
                receipt = json.loads(original_receipt)
                receipt["manifest_sha256"] = p.digest(path)
                receipt_path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "plan partition"):
                    self.full.plan()
                path.write_bytes(original)
                receipt_path.write_bytes(original_receipt)

    def test_unsafe_or_unknown_task_refused(self):
        self.invoke("preflight")
        with self.assertRaisesRegex(ValueError, "Unknown/unsafe"):
            self.invoke("run_task", "../query_only_001")
        with self.assertRaisesRegex(ValueError, "Unknown/unsafe"):
            self.invoke("run_task", "query_only_999")

    def test_corrupt_task_rows_sidecar_and_subset_refused(self):
        m = self.completed_tasks()
        t = m["tasks"][0]
        output, log, subset, receipt = self.full.task_paths(t)
        sid = t["queries"][0]
        path = output / f"similarity_{sid}.pkl"
        with path.open("rb") as stream:
            original = pickle.load(stream)
        for changed in (original[:-1],
                        [original[0], original[0], *original[2:]],
                        list(reversed(original)),
                        [(ref, 1e6 if i == 0 else value)
                         for i, (ref, value) in enumerate(original)]):
            with self.subTest(kind=str(changed[:2])[:50]):
                dump(path, changed)
                with self.assertRaisesRegex(ValueError, "incomplete/reordered|sentinel"):
                    self.full.task_rows(t, m)
        dump(path, original)
        prov_path = log / "provenance_1_2.json"
        prov = p.json_read(prov_path)
        for key, bad in (("n_done", 1), ("query_range", [1, 1]),
                         ("lsf_job", ""), ("lsf_index", "1"),
                         ("task", "foreign"), ("n_library", 9)):
            with self.subTest(key=key):
                prov_path.write_text(json.dumps(prov | {key: bad}))
                with self.assertRaisesRegex(ValueError, "OT sidecar"):
                    self.full.task_rows(t, m)
        prov_path.write_text(json.dumps(prov))
        link = subset / f"{sid}.pkl"
        link.unlink()
        link.symlink_to(self.fx.inputs.source_features / f"{sid}.pkl")
        with self.assertRaisesRegex(ValueError, "subset symlink source"):
            self.full.subset(t, m)
        self.assertTrue(receipt.exists())

    def test_missing_or_extra_task_rows_and_logs_refused(self):
        m = self.completed_tasks()
        t = m["tasks"][-1]
        output, log, _, _ = self.full.task_paths(t)
        sid = t["queries"][0]
        (output / f"similarity_{sid}.pkl").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete/extra"):
            self.full.task_rows(t, m)
        dump(output / f"similarity_{sid}.pkl",
             [(ref, 0.5) for ref in m["refs"][t["arm"]]])
        (log / "provenance_1_1.json").rename(log / "provenance_1_2.json")
        with self.assertRaisesRegex(ValueError, "incomplete/extra"):
            self.full.task_rows(t, m)
        (log / "provenance_1_2.json").rename(log / "provenance_1_1.json")
        dump(output / "similarity_foreign.pkl", [])
        with self.assertRaisesRegex(ValueError, "incomplete/extra"):
            self.full.task_rows(t, m)

    def test_wrong_nonempty_worker_job_rejected_before_receipt(self):
        self.invoke("preflight")
        t = self.full.plan()["tasks"][0]
        original_worker = self.subprocess_worker

        def wrong_job(command, **kwargs):
            result = original_worker(command, **kwargs)
            if command[0] != "git":
                _, log, _, _ = self.full.task_paths(t)
                path = log / "provenance_1_2.json"
                prov = p.json_read(path)
                prov["lsf_job"] = "different-job"
                path.write_text(json.dumps(prov))
            return result

        runtime = {"lsf_job": "fixture-job", "lsf_slots": "8",
                   "affinity_cpus": list(range(8))}
        with patch.object(p, "runtime_info", return_value=runtime), \
                patch.object(f.subprocess, "run", side_effect=wrong_job):
            with self.assertRaisesRegex(ValueError, "worker job differs"):
                self.full.run_task(t["id"])
        self.assertFalse((self.full.root / "receipts" / f"{t['id']}.json").exists())

    def test_exit_zero_complete_finite_ot_warning_refuses_receipt(self):
        self.invoke("preflight")
        m = self.full.plan()
        t = m["tasks"][0]
        self.ot_stderr = b"UserWarning: numItermax reached; convergence not certified\n"
        with patch.dict(os.environ, {"PYTHONWARNINGS": "ignore"}):
            with self.assertRaisesRegex(ValueError, "OT worker stderr nonempty"):
                self.invoke("run_task", t["id"])
        self.assertEqual(set(self.full.task_rows(t, m)["rows"]), set(t["queries"]))
        stdout, stderr = self.full.execution_log_paths(t)
        self.assertTrue(stdout.is_file() and stderr.is_file())
        self.assertEqual(stderr.read_bytes(), self.ot_stderr)
        diagnostic = p.json_read(self.full.root / "diagnostics" / f"{t['id']}.json")
        self.assertEqual(diagnostic["status"], "worker_stderr_nonempty")
        self.assertEqual(diagnostic["returncode"], 0)
        self.assertEqual(diagnostic["execution_logs"], {
            "stdout": f.bound(stdout), "stderr": f.bound(stderr)})
        self.assertFalse(self.full.task_paths(t)[-1].exists())
        with self.assertRaisesRegex(ValueError, "pre-existing task"):
            self.invoke("run_task", t["id"])

    def test_nonzero_ot_exit_preserves_hashed_diagnostics_without_receipt(self):
        self.invoke("preflight")
        t = self.full.plan()["tasks"][0]
        self.ot_exit_code = 7
        self.ot_stderr = b"solver fatal\n"
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke("run_task", t["id"])
        stdout, stderr = self.full.execution_log_paths(t)
        diagnostic = p.json_read(self.full.root / "diagnostics" / f"{t['id']}.json")
        self.assertEqual(diagnostic["status"], "worker_nonzero_exit")
        self.assertEqual(diagnostic["returncode"], 7)
        self.assertEqual(diagnostic["execution_logs"], {
            "stdout": f.bound(stdout), "stderr": f.bound(stderr)})
        self.assertFalse(self.full.task_paths(t)[-1].exists())

    def test_changed_execution_log_hash_refuses_merge(self):
        m = self.completed_tasks()
        t = m["tasks"][0]
        stdout, _ = self.full.execution_log_paths(t)
        stdout.write_bytes(stdout.read_bytes() + b"altered")
        with self.assertRaisesRegex(ValueError, "task receipt/job/source changed"):
            self.invoke("merge")
        self.assertFalse((self.full.root / "COMPLETE.json").exists())

    def test_late_nonempty_stderr_refuses_merge(self):
        m = self.completed_tasks()
        _, stderr = self.full.execution_log_paths(m["tasks"][0])
        stderr.write_bytes(b"UserWarning: later unexpected OT stderr\n")
        with self.assertRaisesRegex(ValueError, "OT worker stderr/diagnostic"):
            self.invoke("merge")
        self.assertFalse((self.full.root / "COMPLETE.json").exists())

    def test_changed_global_bank_payload_refused_at_merge(self):
        self.completed_tasks()
        raw = self.fx.raw_all / "ref02.pkl"
        raw.write_bytes(raw.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Pilot source input changed"):
            self.invoke("merge")
        self.assertFalse((self.full.root / "COMPLETE.json").exists())

    def test_historical_control_not_replayed(self):
        m = self.completed_tasks()
        arm = "query_only"
        original = m["bindings"][f"{arm}_cache_{p.CONTROL}"]["sha256"]
        control = self.full.inputs.historical[arm].cache / f"similarity_{p.CONTROL}.pkl"
        replay = self.fx.out / f"{p.DEST}/pilot_ot/{arm}/{p.CONTROL}" / (
            f"similarity_{p.CONTROL}.pkl")
        shutil.copyfile(replay, control)
        self.assertNotEqual(p.digest(control), original)
        with self.assertRaisesRegex(ValueError, "Global source/input payloads|cache/matrix bits"):
            self.invoke("merge")
        self.assertFalse((self.full.root / "COMPLETE.json").exists())

    def assert_bad_combine(self, failure, message):
        self.corrupt_combine = failure
        self.completed_tasks()
        with self.assertRaisesRegex(ValueError, message):
            self.invoke("merge")
        self.assertFalse((self.full.root / "COMPLETE.json").exists())
        self.assertTrue(self.full.matrix_path("query_only").is_file())

    def test_combine_wrong_shape(self):
        self.assert_bad_combine("shape", "Wrong matrix shape/order/dtype")

    def test_combine_wrong_order(self):
        self.assert_bad_combine("order", "Wrong matrix shape/order/dtype")

    def test_combine_wrong_dtype(self):
        self.assert_bad_combine("dtype", "Wrong matrix shape/order/dtype")

    def test_combine_changed_cells(self):
        self.assert_bad_combine("bits", "combined row differs")

    def test_signed_zero_cells_require_bitwise_agreement(self):
        self.assertFalse(f.bits_equal(np.array([-0.0]), np.array([0.0])))

    def test_combine_sidecar_count_spoof_refused(self):
        self.completed_tasks()
        original_worker = self.worker

        def bad_sidecar(instance, module, args, stage):
            original_worker(instance, module, args, stage)
            if stage == "merge":
                model = args[args.index("--matrix-model") + 1]
                arm = next(arm for arm in f.ARMS if f.MODELS[arm] == model)
                sidecar = self.full.sidecar_path(arm)
                prov = p.json_read(sidecar)
                prov["n_query_slides"] = 999
                sidecar.write_text(json.dumps(prov))

        with patch.object(p, "runtime_info",
                          return_value={"lsf_job": "fixture-job"}), \
                patch.object(p.Pilot, "run_worker", autospec=True,
                             side_effect=bad_sidecar):
            with self.assertRaisesRegex(ValueError, "stock combine sidecar mismatch"):
                self.full.merge()
        self.assertFalse((self.full.root / "COMPLETE.json").exists())

    def test_late_query_only_matrix_mutation_during_both_combine_refused(self):
        self.completed_tasks()

        def mutate():
            path = self.full.matrix_path("query_only")
            with path.open("rb") as stream:
                frame = pickle.load(stream)
            frame.iloc[0, 0] += 0.01
            dump(path, frame)

        self.after_second_combine = mutate
        with self.assertRaisesRegex(ValueError, "combined row differs"):
            self.invoke("merge")
        self.assertFalse((self.full.root / "COMPLETE.json").exists())

    def test_late_row_and_matching_matrix_mutation_refuses_old_receipt(self):
        m = self.completed_tasks()
        t = m["tasks"][0]
        sid = t["queries"][0]

        def mutate():
            path = self.fx.out / t["task"] / f"similarity_{sid}.pkl"
            with path.open("rb") as stream:
                row = pickle.load(stream)
            row[0] = (row[0][0], row[0][1] + 0.01)
            dump(path, row)
            matrix = self.full.matrix_path("query_only")
            with matrix.open("rb") as stream:
                frame = pickle.load(stream)
            frame.loc[sid, row[0][0]] = row[0][1]
            dump(matrix, frame)

        self.after_second_combine = mutate
        with self.assertRaisesRegex(ValueError, "task receipt/job/source changed"):
            self.invoke("merge")
        self.assertFalse((self.full.root / "COMPLETE.json").exists())

    def test_late_task_receipt_job_mutation_refused_at_finalization(self):
        m = self.completed_tasks()
        receipt = self.full.task_paths(m["tasks"][0])[-1]

        def mutate():
            payload = p.json_read(receipt)
            payload["runtime"]["lsf_job"] = "different-job"
            receipt.write_text(json.dumps(payload))

        self.after_second_combine = mutate
        with self.assertRaisesRegex(ValueError, "task receipt/job/source changed"):
            self.invoke("merge")
        self.assertFalse((self.full.root / "COMPLETE.json").exists())

    def test_late_execution_log_mutation_refused_at_finalization(self):
        m = self.completed_tasks()
        stdout, _ = self.full.execution_log_paths(m["tasks"][0])
        self.after_second_combine = lambda: stdout.write_bytes(
            stdout.read_bytes() + b"late")
        with self.assertRaisesRegex(ValueError, "task receipt/job/source changed"):
            self.invoke("merge")
        self.assertFalse((self.full.root / "COMPLETE.json").exists())

    def test_late_sidecar_mutation_refused_at_finalization(self):
        m = self.completed_tasks()
        t = m["tasks"][0]
        log = self.full.task_paths(t)[1]
        sidecar = log / "provenance_1_2.json"

        def mutate():
            payload = p.json_read(sidecar)
            payload["n_done"] = 1
            sidecar.write_text(json.dumps(payload))

        self.after_second_combine = mutate
        with self.assertRaisesRegex(ValueError, "OT sidecar"):
            self.invoke("merge")
        self.assertFalse((self.full.root / "COMPLETE.json").exists())

    def test_late_preflight_receipt_mutation_refused_at_finalization(self):
        self.completed_tasks()
        path = self.full.root / "preflight_receipt.json"
        self.after_second_combine = lambda: path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "preflight changed before finalization"):
            self.invoke("merge")
        self.assertFalse((self.full.root / "COMPLETE.json").exists())

    def test_postcompletion_changed_task_receipt_breaks_bound_chain(self):
        m = self.completed_tasks()
        self.invoke("merge")
        self.full.verify_complete_chain()
        marker = self.full.root / "COMPLETE.json"
        merge_receipt = self.full.root / "merge_receipt.json"
        preserved = (p.digest(marker), p.digest(merge_receipt))
        receipt = self.full.task_paths(m["tasks"][0])[-1]
        payload = p.json_read(receipt)
        payload["runtime"]["lsf_job"] = "different-job"
        receipt.write_text(json.dumps(payload))
        self.assertEqual((p.digest(marker), p.digest(merge_receipt)), preserved)
        with self.assertRaisesRegex(ValueError, "completion/preflight/task receipt hash chain"):
            self.full.verify_complete_chain()

    def test_default_rank_pins_match_accepted_provenance_and_sha_format(self):
        spec = f.CONTRACT
        for name, value in vars(spec).items():
            if name.endswith("_sha"):
                with self.subTest(pin=name):
                    self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}", value))
        self.assertEqual(
            spec.rank_both_sha,
            "c062dbb973287d15b91ff308209a511406004239b0bcb706e065fad7399dc6f7")
        provenance = p.json_read(
            p.ROOT / "prosper_revision_output/r1_3_external_correction_20260930/provenance.json")
        self.assertEqual(spec.provenance_sha, p.digest(
            p.ROOT / "prosper_revision_output/r1_3_external_correction_20260930/provenance.json"))
        for arm, key in (("query_only", "kidrare_query_only_ranked"),
                         ("both_sided", "kidrare_both_sided_ranked")):
            expected = (spec.rank_all_sha if arm == "query_only"
                        else spec.rank_both_sha)
            self.assertEqual(provenance["inputs"][key]["sha256"], expected)

    def test_cli_requires_task_selector_only_for_run_task(self):
        for stage, args in (("preflight", ["--task-id", "query_only_001"]),
                            ("merge", ["--task-id", "query_only_001"]),
                            ("run-task", [])):
            with self.subTest(stage=stage), \
                    patch.object(p, "out_root",
                                 side_effect=AssertionError("entered production")):
                with self.assertRaisesRegex(ValueError, "Only run-task"):
                    f.main(["--stage", stage, *args])

    def test_partition_and_normalized_namespace_guard(self):
        self.assertEqual([len(t["queries"]) for t in
                          f.partition(["a", "b", "c", "d", "e", "f"],
                                      ["a", "b", "c"])], [2, 1, 2, 1])
        ids = [f"new{i:03d}" for i in range(354)]
        tasks = f.partition(ids, ids[:3])
        self.assertEqual(len(tasks), 352)
        self.assertEqual(sum(map(lambda task: len(task["queries"]), tasks)), 702)
        for arm in f.ARMS:
            sizes = [len(task["queries"]) for task in tasks if task["arm"] == arm]
            self.assertEqual((sizes.count(2), sizes.count(1)), (175, 1))
        self.assertEqual(len({t["task"].replace("/", "_") for t in tasks}), 352)
        for ids, reused in ((["a", "a", "b"], ["a", "b", "c"]),
                            (["a", "b", "c"], ["a", "b", "z"])):
            with self.assertRaisesRegex(ValueError, "partition invalid"):
                f.partition(ids, reused)
