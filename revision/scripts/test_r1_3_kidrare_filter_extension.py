"""Isolated [decision record] pilot guards; no real bank, classifier or OT work."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import r1_3_kidrare_filter_extension as e
from prosper.preprocessing import tumor_tiles_cli


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


class Classifier:
    classes_ = np.array([1, 2, 3, 4, 5])
    n_features_in_ = 1536

    def predict_proba(self, x):
        result = np.zeros((len(x), 5), dtype=float)
        result[:, 0] = np.where(x[:, 0] > 0, .9, .1)
        result[:, 1] = 1 - result[:, 0]
        return result


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.out = e.ROOT / "prosper_revision_output" / f"r1_3_pilot_fixture_{os.getpid()}"
        self.assertFalse(self.out.exists(), "Refusing an existing fixture directory")
        self.out.mkdir()
        self.addCleanup(lambda: shutil.rmtree(self.out))
        self.new = ["new_a", "new_b", "new_c", "new_empty"]
        source = self.out / "source_features"
        for sid, predictions in (
            ("new_a", [1, 0]), ("new_b", [1, 0, 1]),
            ("new_c", [1] * 9), ("new_empty", [0])):
            feats = np.zeros((len(predictions), 1536), dtype=np.float32)
            feats[:, 0] = np.where(predictions, 1, -1)
            coords = [{"wsi_loc": [i * 512, i * 512]} for i in range(len(predictions))]
            dump(source / f"{sid}.pkl", (feats, coords))
        self.raw_all = self.out / "raw_all"
        self.raw_both = self.out / "raw_both"
        self.refids = [f"ref{i:02d}" for i in range(12)]
        for i, sid in enumerate(self.refids):
            feats = np.ones((1, 1536), dtype=np.float32)
            coords = [{"wsi_loc": [i * 512, 0]}]
            dump(self.raw_all / f"{sid}.pkl", (feats, coords))
            if i:
                dump(self.raw_both / f"{sid}.pkl", (feats, coords))
        metadata = self.out / "comet_metadata.csv"
        pd.DataFrame({"Slide.Scan.File": self.refids,
                      "Tumor.group": ["NORMAL", "MISC", *["WT"] * 10]}).to_csv(
                          metadata, index=False)
        summary = self.out / "old_filter_summary.csv"
        pd.DataFrame({"slide": self.refids, "kept_tumor_only":
                      [0] + [1] * 11}).to_csv(summary, index=False)
        control = self.out / "old_control.pkl"
        dump(control, (np.ones((2, 1536), dtype=np.float32),
                       [{"wsi_loc": [0, 0]}, {"wsi_loc": [512, 512]}]))
        classifier = self.out / "classifier.pkl"
        classifier.write_bytes(b"fixture classifier: monkeypatched for tiny test")
        manifest = self.out / "changed.csv"
        pd.DataFrame({"slide_id": self.new, "processed": ["True"] * 4,
                      "pathpt_only_new_label": ["True"] * 4,
                      "corrected_tumor_group": ["WT"] * 4,
                      "original_tumor_group": ["WT_UNVERIFIED"] * 4,
                      "legacy_eligible": ["False"] * 4,
                      "amended_eligible": ["True"] * 4,
                      "query_only_eligible": ["False"] * 4,
                      "both_sided_eligible": ["False"] * 4}).to_csv(manifest, index=False)
        self.oldrows = {}
        self.oldranks = {}
        for arm, ids in (("query_only", self.refids),
                         ("both_sided", self.refids[1:])):
            rowpath = self.out / f"old_{arm}.pkl"
            distances = [0.4 + i * .01 for i in range(len(ids))]
            distances[ids.index("ref02")] = .50000001
            distances[ids.index("ref03")] = .50000002
            dump(rowpath, list(zip(ids, distances)))
            self.oldrows[arm] = rowpath
            rankpath = self.out / f"rank_{arm}.csv.gz"
            keep = set(self.refids[2:]) if arm == "both_sided" else set(
                self.refids) - {self.refids[1]}
            ranked = e.top10(ids, np.asarray(distances), keep, e.CONTROL)
            pd.DataFrame({"query": [e.CONTROL] * 10,
                          "rank": range(1, 11), "neighbour": ranked}).to_csv(
                              rankpath, index=False, compression="gzip")
            self.oldranks[arm] = rankpath
        self.inputs = e.Inputs(
            source_features=source, raw_all=self.raw_all, raw_both=self.raw_both,
            classifier=classifier, changed_manifest=manifest, metadata=metadata,
            old_filter_summary=summary, old_control_feature=control,
            old_query_row=self.oldrows["query_only"],
            old_both_row=self.oldrows["both_sided"],
            old_query_rank=self.oldranks["query_only"],
            old_both_rank=self.oldranks["both_sided"],
            worker_root=e.PROSPER)
        self.spec = e.Spec(
            n_new=4, raw_all=12, raw_both=11, eligible_all=11, eligible_both=10,
            control_tiles=2, manifest_sha=e.digest(manifest),
            ids_sha=hashlib.sha256("".join(s + "\n" for s in
                                         sorted(self.new)).encode()).hexdigest(),
            classifier_sha=e.digest(classifier), metadata_sha=e.digest(metadata))
        self.pilot = e.Pilot(self.out, self.inputs, self.spec)
        self.calls = []

    def worker(self, command, *, cwd=None, env=None, check=None,
               capture_output=False, text=False):
        if command[0] == "git":
            self.assertEqual(command[:6], [
                "git", "-C", str(e.PROSPER), "diff", "--quiet", "HEAD"])
            self.assertEqual(command[6], "--")
            self.assertEqual(set(command[7:]), set(e.worker_files(e.PROSPER)))
            self.assertNotIn("prosper/evaluation/metrics.py", command)
            self.assertFalse(check)
            return subprocess.CompletedProcess(command, 0, "", "")
        self.assertTrue(check)
        self.assertEqual(command[:2], ["bash", str(e.PROSPER / "scripts/lsf/prosper_job.sh")])
        self.assertNotIn("PROSPER_PARAMS", env)
        self.assertNotIn("LSB_JOBINDEX", env)
        self.assertEqual(env["PROSPER_ROOT"], str(e.PROSPER))
        self.assertEqual(env["PROSPER_OUT"], str(self.out.resolve()))
        self.assertEqual(env["COMET_ROOT"], str(e.COMET))
        self.assertEqual(env["PROSPER_PYTHON"], str(e.PYTHON))
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
        self.calls.append((env["PROSPER_MODULE"], command[2:]))
        argv = command[2:]
        if env["PROSPER_MODULE"] == "prosper.preprocessing.tumor_tiles_cli":
            tumor_tiles_cli.main(argv)
        else:
            a = dict(zip(argv[::2], argv[1::2]))
            sid = next(Path(a["--features"]).glob("*.pkl")).stem
            refs = e.ids_in(Path(a["--library-features"]))
            arm = "query_only" if Path(a["--library-features"]) == self.raw_all else "both_sided"
            if sid == e.CONTROL:
                old = e.row(self.oldrows[arm], sid, refs)
                distances = old + 1e-9
            else:
                distances = np.array([.4 + i * .01 for i in range(len(refs))])
            task = a["--task"]
            dump(self.out / task / f"similarity_{sid}.pkl",
                 list(zip(refs, distances.tolist())))
            log = self.out / "logs" / task.replace("/", "_")
            log.mkdir(parents=True)
            e.json_write(log / "provenance_1_1.json", {
                "task": task, "encoder": "uni2", "method": "ot", "feature_dim": 1536,
                "metric": "cosine", "reg": 0, "external": True, "num_workers": 8,
                "n_library": len(refs), "n_queries_total": 1, "query_range": [1, 1],
                "n_queries_this_element": 1, "n_done": 1, "tile_fraction": None,
                "git_commit": self.spec.source_commit, "query_features": a["--features"],
                "library_features": a["--library-features"], "lsf_job": "fixture",
                "lsf_index": None})
        return subprocess.CompletedProcess(command, 0)

    def stages(self):
        with patch.dict(os.environ, e.THREAD_ENV | {
                "PROSPER_OUT": str(self.out), "LSB_JOBINDEX": "0"}), \
                patch.object(e, "runtime_info",
                             return_value={"fixture": "bounded", "lsf_job": "fixture"}), \
                patch.object(e.subprocess, "check_output",
                             return_value=self.spec.source_commit + "\n"), \
                patch.object(e.subprocess, "run", side_effect=self.worker), \
                patch.object(tumor_tiles_cli, "_load_clf", return_value=Classifier()):
            self.pilot.score()
            self.pilot.filter()
            selection = e.json_read(self.pilot.root / "selection.json")
            for x in selection["tasks"]:
                self.pilot.ot(x["arm"], x["query"])
            self.pilot.verify()
        return selection

    def stock_filtered_fixture(self, predictions=None):
        if predictions is not None:
            self.assertEqual(set(predictions), set(self.new))
            for sid, labels in predictions.items():
                feats = np.zeros((len(labels), 1536), dtype=np.float32)
                feats[:, 0] = np.where(labels, 1, -1)
                coords = [{"wsi_loc": [i * 512, i * 512]}
                          for i in range(len(labels))]
                dump(self.inputs.source_features / f"{sid}.pkl", (feats, coords))
        with patch.dict(os.environ, e.THREAD_ENV | {"PROSPER_OUT": str(self.out)}), \
                patch.object(e, "runtime_info", return_value={"lsf_job": "fixture"}), \
                patch.object(e.subprocess, "check_output",
                             return_value=self.spec.source_commit + "\n"), \
                patch.object(e.subprocess, "run", side_effect=self.worker), \
                patch.object(tumor_tiles_cli, "_load_clf", return_value=Classifier()):
            self.pilot.score()
            self.pilot.run_worker(
                "prosper.preprocessing.tumor_tiles_cli",
                ["filter", "--features-dir", str(self.pilot.root / "source_subset"),
                 "--scores-dir", str(self.pilot.root / "scores"),
                 "--out-root", str(self.pilot.root / "features_in"),
                 "--arms", "tumor_only", "--min-tiles", "8", "--skip-empty"],
                "filter")
        ids = self.pilot.new_ids()
        return ids, self.pilot.verify_scores(ids)

    def test_isolated_full_pilot_receipts_and_source_immutability(self):
        original = {k: e.digest(p) for k, p in
                    self.pilot.file_inputs(self.new).items()}
        selection = self.stages()
        self.assertEqual(len(self.calls), 10)
        self.assertEqual(len(selection["tasks"]), 8)
        self.assertEqual(set(selection["selection"].values()),
                         {"new_a", "new_b", "new_c"})
        marker = e.json_read(self.pilot.root / "PILOT_COMPLETE.json")
        self.assertEqual((marker["n_new_scored"], marker["n_ot_tasks"]), (4, 8))
        self.assertEqual({k: e.digest(p) for k, p in
                          self.pilot.file_inputs(self.new).items()}, original)
        summaries = pd.read_csv(self.pilot.root / "features_in/_filter_summary.csv")
        self.assertEqual(summaries.set_index("slide").loc["new_empty",
                                                          "kept_tumor_only"], 0)
        self.assertFalse((self.pilot.root / "features_in/tumor_only/new_empty.pkl").exists())
        self.assertTrue((self.pilot.root / "features_in/tumor_only/new_a.pkl").exists())
        for arm in e.ARMS:
            receipt = e.json_read(self.pilot.root /
                                  f"receipt_{arm}_{e.CONTROL}.json")
            self.assertLess(receipt["result"]["max_relative_discrepancy"], 1e-6)
            self.assertEqual(len(receipt["result"]["top10_ids"]), 10)
        with self.assertRaisesRegex(ValueError, "pre-existing pilot root"):
            self.pilot.score()
        with patch.object(e, "runtime_info", return_value={"lsf_job": "fixture"}):
            with self.assertRaisesRegex(ValueError, "pre-existing task"):
                self.pilot.ot("query_only", e.CONTROL)

    def test_cli_requires_both_selectors_only_for_ot(self):
        for stage in ("score", "filter", "verify"):
            for selectors in (["--arm", "query_only"], ["--query", "new_a"],
                              ["--arm", "query_only", "--query", "new_a"]):
                with self.subTest(stage=stage, selectors=selectors):
                    with patch.object(e, "out_root",
                                      side_effect=AssertionError("entered production")):
                        with self.assertRaisesRegex(ValueError, "Non-OT stages refuse"):
                            e.main(["--stage", stage, *selectors])
        for selectors in ([], ["--arm", "query_only"], ["--query", "new_a"]):
            with self.subTest(stage="ot", selectors=selectors):
                with patch.object(e, "out_root",
                                  side_effect=AssertionError("entered production")):
                    with self.assertRaisesRegex(ValueError, "OT stage requires both"):
                        e.main(["--stage", "ot", *selectors])
        with patch.object(e, "out_root", return_value=self.out), \
                patch.object(e, "Pilot") as pilot_class:
            e.main(["--stage", "score"])
            pilot_class.return_value.score.assert_called_once_with()

    def test_dirty_relevant_worker_refused_unrelated_paths_not_checked(self):
        def dirty_worker(command, **kwargs):
            self.assertEqual(command[0], "git")
            self.assertEqual(set(command[7:]), set(e.worker_files(e.PROSPER)))
            self.assertNotIn("prosper/evaluation/metrics.py", command)
            return subprocess.CompletedProcess(command, 1, "", "worker modified")

        with patch.object(e, "runtime_info", return_value={"lsf_job": "fixture"}), \
                patch.object(e.subprocess, "check_output",
                             return_value=self.spec.source_commit + "\n"), \
                patch.object(e.subprocess, "run", side_effect=dirty_worker):
            with self.assertRaisesRegex(ValueError, "differs from pinned HEAD"):
                self.pilot.preflight()
        self.assertFalse(self.pilot.root.exists())
        e.verify_worker_head(e.PROSPER)

    def test_identity_collision_and_unmodified_source_guard(self):
        with patch.dict(os.environ, e.THREAD_ENV | {"PROSPER_OUT": str(self.out)}), \
                patch.object(e, "runtime_info", return_value={"lsf_job": "fixture"}), \
                patch.object(e.subprocess, "check_output",
                             return_value=self.spec.source_commit + "\n"), \
                patch.object(e.subprocess, "run", side_effect=self.worker), \
                patch.object(tumor_tiles_cli, "_load_clf", return_value=Classifier()):
            self.pilot.score()
            self.pilot.filter()
            sel = e.json_read(self.pilot.root / "selection.json")
            task = sel["tasks"][0]["task"]
            (self.out / "logs" / task.replace("/", "_")).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "pre-existing task"):
                self.pilot.ot(sel["tasks"][0]["arm"], sel["tasks"][0]["query"])
            (self.out / "logs" / task.replace("/", "_")).rmdir()
            self.pilot.ot(sel["tasks"][0]["arm"], sel["tasks"][0]["query"])
            first = self.new[0]
            source = self.inputs.source_features / f"{first}.pkl"
            source.write_bytes(source.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "source input changed"):
                self.pilot.check_bindings(
                    e.json_read(self.pilot.root / "preflight.json"), self.new,
                    full=True)

    def test_manifest_and_id_membership_guard(self):
        self.assertEqual(self.pilot.new_ids(), sorted(self.new))
        changed = pd.read_csv(self.inputs.changed_manifest, dtype=str)
        changed.loc[0, "legacy_eligible"] = "True"
        changed.to_csv(self.inputs.changed_manifest, index=False)
        with self.assertRaisesRegex(ValueError, "manifest hash changed"):
            self.pilot.new_ids()

    def test_final_comet_label_change_rejected_despite_same_eligible_counts(self):
        self.assertEqual(tuple(len(x) for x in self.pilot.reference_ids()[:2]),
                         (12, 11))
        metadata = pd.read_csv(self.inputs.metadata, dtype=str)
        metadata.loc[metadata["Slide.Scan.File"] == "ref02", "Tumor.group"] = "NB"
        metadata.to_csv(self.inputs.metadata, index=False)
        self.assertEqual(metadata["Slide.Scan.File"].tolist(), self.refids)
        self.assertEqual(int(metadata["Tumor.group"].ne("MISC").sum()), 11)
        with self.assertRaisesRegex(ValueError, "metadata hash changed"):
            self.pilot.reference_ids()

    def test_explicit_repository_output_and_pinned_classifier(self):
        with patch.dict(os.environ, {"PROSPER_OUT": ""}):
            with self.assertRaisesRegex(ValueError, "explicitly resolve"):
                e.out_root()
        with patch.dict(os.environ, {"PROSPER_OUT": str(self.out)}):
            with self.assertRaisesRegex(ValueError, "repository-local"):
                e.out_root()
        self.inputs.classifier.write_bytes(b"different classifier")
        with patch.object(e, "runtime_info", return_value={"lsf_job": "fixture"}):
            with self.assertRaisesRegex(ValueError, "classifier hash changed"):
                self.pilot.preflight()
        self.assertFalse(self.pilot.root.exists())

    def test_exact_subsets_empty_and_small_tile_warning(self):
        with patch.dict(os.environ, e.THREAD_ENV | {"PROSPER_OUT": str(self.out)}), \
                patch.object(e, "runtime_info", return_value={}), \
                patch.object(e.subprocess, "check_output",
                             return_value=self.spec.source_commit + "\n"), \
                patch.object(e.subprocess, "run", side_effect=self.worker), \
                patch.object(tumor_tiles_cli, "_load_clf", return_value=Classifier()):
            self.pilot.score()
            scores = self.pilot.verify_scores(sorted(self.new))
            path = self.pilot.root / "scores/new_c.csv.gz"
            original_bytes = path.read_bytes()
            recorded_sha = e.json_read(self.pilot.root /
                                       "score_receipt.json")["score_sha256"]["new_c"]
            self.assertEqual(e.digest(path), recorded_sha)
            scores["new_c"].loc[0, "wsi_loc_x"] += 1
            scores["new_c"].to_csv(path, index=False, compression="gzip")
            with self.assertRaisesRegex(ValueError, "coordinate"):
                self.pilot.verify_scores(sorted(self.new))
            path.write_bytes(original_bytes)
            self.assertEqual(e.digest(path), recorded_sha)
            compressed = bytearray(original_bytes)
            compressed[4] ^= 1
            self.assertEqual(gzip.decompress(compressed),
                             gzip.decompress(original_bytes))
            path.write_bytes(compressed)
            self.assertNotEqual(e.digest(path), recorded_sha)
            with self.assertRaisesRegex(ValueError, "Scoring files changed"):
                self.pilot.filter()

    def test_stock_filter_all_nonempty_no_optional_status_columns(self):
        ids, scores = self.stock_filtered_fixture({
            sid: [1] * 8 for sid in self.new})
        summary = pd.read_csv(self.pilot.root / "features_in/_filter_summary.csv",
                              dtype=str, keep_default_na=False)
        self.assertEqual(summary.columns.tolist(), [
            "slide", "n_tiles", "n_tumor", "n_immune", "n_necrosis",
            "n_nontumor", "n_artifacts", "kept_tumor_only"])
        self.assertEqual(self.pilot.verify_filter(ids, scores),
                         {sid: 8 for sid in ids})

    def test_stock_filter_below_min_only_optional_status_column(self):
        ids, scores = self.stock_filtered_fixture({
            "new_a": [1] * 3, "new_b": [1] * 7,
            "new_c": [1] * 8, "new_empty": [1] * 9})
        summary = pd.read_csv(self.pilot.root / "features_in/_filter_summary.csv",
                              dtype=str, keep_default_na=False).set_index("slide")
        self.assertIn("below_min_tumor_only", summary.columns)
        self.assertNotIn("skipped_tumor_only", summary.columns)
        self.assertEqual(summary.loc["new_a", "below_min_tumor_only"], "True")
        self.assertEqual(summary.loc["new_b", "below_min_tumor_only"], "True")
        self.assertEqual(summary.loc["new_c", "below_min_tumor_only"], "")
        self.assertEqual(self.pilot.verify_filter(ids, scores),
                         {"new_a": 3, "new_b": 7, "new_c": 8, "new_empty": 9})

    def test_stock_filter_skipped_only_optional_status_column(self):
        ids, scores = self.stock_filtered_fixture({
            "new_a": [0], "new_b": [1] * 8,
            "new_c": [1] * 9, "new_empty": [1] * 8})
        summary = pd.read_csv(self.pilot.root / "features_in/_filter_summary.csv",
                              dtype=str, keep_default_na=False).set_index("slide")
        self.assertIn("skipped_tumor_only", summary.columns)
        self.assertNotIn("below_min_tumor_only", summary.columns)
        self.assertEqual(summary.loc["new_a", "skipped_tumor_only"], "True")
        self.assertEqual(summary.loc["new_b", "skipped_tumor_only"], "")
        self.assertEqual(self.pilot.verify_filter(ids, scores),
                         {"new_b": 8, "new_c": 9, "new_empty": 8})

    def test_stock_filter_mixed_optional_status_columns(self):
        ids, scores = self.stock_filtered_fixture()
        summary = pd.read_csv(self.pilot.root / "features_in/_filter_summary.csv",
                              dtype=str, keep_default_na=False).set_index("slide")
        self.assertEqual(summary.loc["new_empty", "skipped_tumor_only"], "True")
        self.assertEqual(summary.loc["new_a", "below_min_tumor_only"], "True")
        self.assertEqual(summary.loc["new_c", "below_min_tumor_only"], "")
        self.assertEqual(self.pilot.verify_filter(ids, scores),
                         {"new_a": 1, "new_b": 2, "new_c": 9})

    def test_filter_missing_positive_flags_and_required_counts_refused(self):
        ids, scores = self.stock_filtered_fixture()
        path = self.pilot.root / "features_in/_filter_summary.csv"
        summary = pd.read_csv(path, dtype=str, keep_default_na=False)
        cases = (
            ("skipped_tumor_only", None, None, "zero retained tiles not explicitly skipped"),
            ("below_min_tumor_only", None, None, "below-eight warning"),
            ("skipped_tumor_only", "new_empty", "", "zero retained tiles not explicitly skipped"),
            ("below_min_tumor_only", "new_a", "", "below-eight warning"),
            ("skipped_tumor_only", "new_c", "True", "below-eight warning"),
            ("below_min_tumor_only", "new_c", "True", "below-eight warning"),
            ("n_tumor", "new_b", "0", "summary/mask count mismatch"),
            ("n_tiles", None, None, "missing required"),
        )
        for column, sid, value, error in cases:
            with self.subTest(column=column, slide=sid, value=value):
                modified = summary.drop(columns=[column]) if sid is None else summary.copy()
                if sid is not None:
                    modified.loc[modified.slide == sid, column] = value
                modified.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, error):
                    self.pilot.verify_filter(ids, scores)

    def test_missing_score_not_mislabeled_algorithmic_empty(self):
        with patch.dict(os.environ, e.THREAD_ENV | {"PROSPER_OUT": str(self.out)}), \
                patch.object(e, "runtime_info", return_value={}), \
                patch.object(e.subprocess, "check_output",
                             return_value=self.spec.source_commit + "\n"), \
                patch.object(e.subprocess, "run", side_effect=self.worker), \
                patch.object(tumor_tiles_cli, "_load_clf", return_value=Classifier()):
            self.pilot.score()
            (self.pilot.root / "scores/new_a.csv.gz").unlink()
            with self.assertRaisesRegex(ValueError, "Score output set incomplete"):
                self.pilot.filter()
            self.assertFalse((self.pilot.root / "selection.json").exists())

    def test_fewer_than_three_nonempty_fails_with_diagnostics(self):
        source = self.inputs.source_features / "new_c.pkl"
        feats, coords = e.feature(source, "new_c")
        feats[:, 0] = -1
        dump(source, (feats, coords))
        with patch.dict(os.environ, e.THREAD_ENV | {"PROSPER_OUT": str(self.out)}), \
                patch.object(e, "runtime_info", return_value={}), \
                patch.object(e.subprocess, "check_output",
                             return_value=self.spec.source_commit + "\n"), \
                patch.object(e.subprocess, "run", side_effect=self.worker), \
                patch.object(tumor_tiles_cli, "_load_clf", return_value=Classifier()):
            self.pilot.score()
            with self.assertRaisesRegex(ValueError, "Only 2 nonempty"):
                self.pilot.filter()
            self.assertEqual(e.json_read(self.pilot.root /
                                         "selection_diagnostics.json")["n_nonempty"], 2)
            self.assertFalse((self.pilot.root / "selection.json").exists())

    def test_incomplete_sentinel_and_exact_rank_mismatch(self):
        path = self.out / "probe.pkl"
        ids = self.refids
        dump(path, list(zip(ids[:-1], [1.] * 11)))
        with self.assertRaisesRegex(ValueError, "incomplete/reordered"):
            e.row(path, e.CONTROL, ids)
        dump(path, list(zip(ids, [1.] * 11 + [1e6])))
        with self.assertRaisesRegex(ValueError, "sentinel"):
            e.row(path, e.CONTROL, ids)
        old = e.row(self.oldrows["query_only"], e.CONTROL, ids)
        new = old.copy()
        new[2], new[3] = new[3], new[2]
        self.assertLess(e.compatibility(new, old)["max_relative_discrepancy"], 1e-6)
        self.assertNotEqual(e.top10(ids, new, set(ids) - {ids[1]}, e.CONTROL),
                            e.top10(ids, old, set(ids) - {ids[1]}, e.CONTROL))

    def test_control_rank_mismatch_blocks_even_after_distance_pass(self):
        self.stages()
        task = f"{e.DEST}/pilot_ot/query_only/{e.CONTROL}"
        path = self.out / task / f"similarity_{e.CONTROL}.pkl"
        original = e.row(path, e.CONTROL, self.refids)
        modified = original.copy()
        modified[2], modified[3] = modified[3], modified[2]
        self.assertLess(e.compatibility(
            modified, e.row(self.oldrows["query_only"], e.CONTROL,
                            self.refids))["max_relative_discrepancy"], 1e-6)
        dump(path, list(zip(self.refids, modified.tolist())))
        with self.assertRaisesRegex(ValueError, "top10 rank/ID differs"):
            self.pilot.check_row("query_only", e.CONTROL, task,
                                 set(self.refids) - {self.refids[1]},
                                 self.refids)

    def test_sidecar_scalar_index_and_exact_task_range_job_bindings(self):
        selection = self.stages()
        task = selection["tasks"][0]["task"]
        sid = selection["tasks"][0]["query"]
        sidecar = self.out / "logs" / task.replace("/", "_") / "provenance_1_1.json"
        original = e.json_read(sidecar)
        _, _, eligible = self.pilot.reference_ids()
        for changes in ({"lsf_index": "0"}, {"lsf_index": "1"},
                        {"query_range": [1, 2]}, {"task": task + "_other"}):
            with self.subTest(changes=changes):
                sidecar.write_text(json.dumps(original | changes))
                with self.assertRaisesRegex(ValueError, "worker sidecar does not bind"):
                    self.pilot.check_row("query_only", sid, task,
                                         eligible["query_only"], self.refids)
        sidecar.write_text(json.dumps(original | {"lsf_job": "other-job"}))
        with patch.object(e, "runtime_info", return_value={"lsf_job": "fixture"}):
            with self.assertRaisesRegex(ValueError, "OT receipt/source changed"):
                self.pilot.verify()
        without_index = {k: v for k, v in original.items() if k != "lsf_index"}
        sidecar.write_text(json.dumps(without_index))
        self.assertEqual(self.pilot.check_row("query_only", sid, task,
                                               eligible["query_only"],
                                               self.refids)["source_job"], "fixture")

    def test_selection_tie_and_insufficient_candidates(self):
        selected = e.choose_three({"c": 3, "b": 3, "a": 3, "d": 3})
        self.assertEqual(selected["selection"], {
            "median": "b", "p95": "c", "maximum": "a"})
        self.assertEqual(selected["distance_ties"], {
            "median": ["b", "c", "d"], "p95": ["c", "d"],
            "maximum": ["a", "b", "c", "d"]})
        self.assertEqual(selected["retained_tiles"],
                         {"median": 3, "p95": 3, "maximum": 3})
        self.assertEqual(selected["target_distances"],
                         {"median": 0.0, "p95": 0.0, "maximum": 0.0})
        self.assertEqual(e.choose_three({"d": 3, "a": 3, "c": 3, "b": 3}),
                         selected)
        three = e.choose_three({"max": 9, "low": 1, "middle": 2})
        self.assertEqual(three["selection"], {
            "median": "middle", "p95": "low", "maximum": "max"})
        self.assertEqual(three["targets"]["median"], 2.0)
        self.assertAlmostEqual(three["targets"]["p95"], 8.3)
        self.assertEqual(three["targets"]["maximum"], 9.0)
        self.assertEqual(three["retained_tiles"],
                         {"median": 2, "p95": 1, "maximum": 9})
        self.assertAlmostEqual(three["target_distances"]["p95"], 7.3)
        maximum_tie = e.choose_three({"z": 9, "b": 1, "a": 9})
        self.assertEqual(maximum_tie["selection"]["maximum"], "a")
        self.assertEqual(maximum_tie["distance_ties"]["maximum"], ["a", "z"])
        with self.assertRaisesRegex(ValueError, "have 2"):
            e.choose_three({"a": 1, "b": 2})

    def test_relative_fidelity_threshold_strict_boundary(self):
        ref = np.array([1.0, 0.0])
        ok = np.array([1 + 9e-7, 0.0])
        self.assertLess(e.compatibility(ok, ref)["max_relative_discrepancy"], 1e-6)
        bad = np.array([np.nextafter(1.000001, np.inf), 0.0])
        with self.assertRaisesRegex(ValueError, "not <1e-6"):
            e.compatibility(bad, ref)

    def test_runtime_native_refusal_and_affinity_record(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "exported before Python"):
                e.runtime_info("ot")
        base = e.THREAD_ENV | {"LSB_JOBID": "fixture", "LSB_DJOB_NUMPROC": "8"}
        for index in (None, "", "0"):
            with self.subTest(index=index), \
                    patch.dict(os.environ, base | (
                        {} if index is None else {"LSB_JOBINDEX": index}), clear=True), \
                    patch("threadpoolctl.threadpool_info",
                          return_value=[{"num_threads": 1}]):
                actual = e.runtime_info("ot")
                self.assertEqual(actual["joblib_workers"], 8)
                self.assertEqual(actual["pot_emd2_numThreads"], 1)
                self.assertTrue(actual["affinity_cpus"])
                self.assertEqual(actual["lsf_index"], index)
        for index in ("1", "12", "garbage", " 0", "-1"):
            with self.subTest(rejected=index), \
                    patch.dict(os.environ, base | {"LSB_JOBINDEX": index}, clear=True):
                with self.assertRaisesRegex(ValueError, "scalar LSF job index"):
                    e.runtime_info("ot")
        with patch.dict(os.environ, base | {"LSB_JOBINDEX": "0",
                                             "PROSPER_PARAMS": "old"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unset PROSPER_PARAMS"):
                e.runtime_info("ot")


if __name__ == "__main__":
    unittest.main()
